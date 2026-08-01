// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! [`ParquetFileReaderFactory`] and [`DefaultParquetFileReaderFactory`] for
//! low level control of parquet file readers

use crate::ParquetFileMetrics;
use crate::metadata::DFParquetMetadata;
use arrow::array::new_null_array;
use arrow::record_batch::{RecordBatch, RecordBatchReader};
use bytes::Bytes;
use datafusion_datasource::PartitionedFile;
use datafusion_execution::cache::cache_manager::FileMetadata;
use datafusion_execution::cache::cache_manager::FileMetadataCache;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use futures::FutureExt;
use futures::future::BoxFuture;
use object_store::ObjectStore;
use parking_lot::Mutex;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReader,
    ParquetRecordBatchReaderBuilder,
};
use parquet::arrow::async_reader::{AsyncFileReader, ParquetObjectReader};
use parquet::errors::{ParquetError as ArrowParquetError, Result as ParquetResult};
use parquet::file::metadata::ParquetMetaData;
use parquet::file::page_index::column_index::ColumnIndexMetaData;
use parquet::file::reader::{ChunkReader, Length};
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::io::Cursor;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// A forward-only, page-lazy Parquet batch reader backed by either a DataFusion
/// [`AsyncFileReader`] or an existing synchronous [`ChunkReader`].
///
/// Unlike [`crate::source::ParquetSource`], this reader does not require a
/// [`parquet::arrow::arrow_reader::RowSelection`] to be known when it is
/// created. [`Self::read_batch_at`] advances one retained Arrow reader, so
/// complete pages between the current position and the requested row are
/// skipped without fetching or decoding them.
///
/// The projected columns must have an OffsetIndex for every non-empty row
/// group. This lets Arrow request complete page ranges with `get_bytes`; the
/// full compressed column chunk is never buffered.
pub struct ParquetForwardBatchReader {
    reader: ParquetRecordBatchReader,
    physical_position: usize,
    position: usize,
    row_count: usize,
    repeated: bool,
    pages: Vec<ParquetForwardPage>,
    metadata: Arc<ParquetMetaData>,
    projected_leaf_column: usize,
}

/// Reusable DataFusion-backed factory for independent forward readers over the
/// same file, metadata, and projection.
pub struct ParquetForwardBatchReaderFactory {
    reader_factory: Arc<dyn ParquetFileReaderFactory>,
    file: PartitionedFile,
    metadata: Arc<ParquetMetaData>,
    projection: ProjectionMask,
    batch_size: usize,
    runtime: Arc<Runtime>,
    local_file: Option<PathBuf>,
}

impl ParquetForwardBatchReaderFactory {
    /// Creates a factory whose readers all use the supplied DataFusion file
    /// reader factory and cached Parquet metadata.
    pub fn new(
        reader_factory: Arc<dyn ParquetFileReaderFactory>,
        file: PartitionedFile,
        metadata: Arc<ParquetMetaData>,
        projection: ProjectionMask,
        batch_size: usize,
        runtime: Arc<Runtime>,
    ) -> Self {
        Self {
            reader_factory,
            file,
            metadata,
            projection,
            batch_size,
            runtime,
            local_file: None,
        }
    }

    /// Uses a synchronous retained file descriptor when `path` identifies an
    /// existing local file. Other files continue through DataFusion's reader
    /// factory.
    pub fn with_local_file_if_exists(mut self, path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        self.local_file = path.is_file().then_some(path);
        self
    }

    /// Opens a new retained forward reader.
    pub fn open(&self) -> ParquetResult<ParquetForwardBatchReader> {
        if let Some(path) = self.local_file.as_ref() {
            let file = std::fs::File::open(path)
                .map_err(|error| ArrowParquetError::External(Box::new(error)))?;
            return ParquetForwardBatchReader::try_new_with_chunk_reader(
                file,
                Arc::clone(&self.metadata),
                self.projection.clone(),
                self.batch_size,
            );
        }
        let metrics = ExecutionPlanMetricsSet::new();
        let async_reader = self
            .reader_factory
            .create_reader(0, self.file.clone(), None, &metrics)
            .map_err(|error| ArrowParquetError::External(Box::new(error)))?;
        ParquetForwardBatchReader::try_new(
            async_reader,
            self.file.object_meta.size,
            Arc::clone(&self.metadata),
            self.projection.clone(),
            self.batch_size,
            Arc::clone(&self.runtime),
        )
    }
}

/// OffsetIndex and ColumnIndex information for one projected data page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParquetForwardPage {
    pub row_group_index: usize,
    pub page_index: usize,
    pub first_row: usize,
    pub row_count: usize,
    pub file_offset: i64,
    pub compressed_size: i32,
    pub null_count: Option<i64>,
    pub all_null: bool,
}

impl Debug for ParquetForwardBatchReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParquetForwardBatchReader")
            .field("position", &self.position)
            .field("physical_position", &self.physical_position)
            .field("row_count", &self.row_count)
            .field("page_count", &self.pages.len())
            .finish_non_exhaustive()
    }
}

impl ParquetForwardBatchReader {
    /// Creates a retained Arrow reader over the full file.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        async_reader: Box<dyn AsyncFileReader + Send>,
        file_len: u64,
        metadata: Arc<ParquetMetaData>,
        projection: ProjectionMask,
        batch_size: usize,
        runtime: Arc<Runtime>,
    ) -> ParquetResult<Self> {
        let chunk_reader = AsyncFileChunkReader {
            reader: Mutex::new(async_reader),
            file_len,
            runtime,
        };
        Self::try_new_with_chunk_reader(chunk_reader, metadata, projection, batch_size)
    }

    /// Creates a retained Arrow reader over an existing synchronous chunk
    /// reader, while reusing metadata loaded by DataFusion.
    pub fn try_new_with_chunk_reader<T>(
        chunk_reader: T,
        metadata: Arc<ParquetMetaData>,
        projection: ProjectionMask,
        batch_size: usize,
    ) -> ParquetResult<Self>
    where
        T: ChunkReader + 'static,
    {
        let (projected_leaf_column, repeated, pages) =
            projected_pages(&metadata, &projection)?;

        let row_count =
            usize::try_from(metadata.file_metadata().num_rows()).map_err(|_| {
                ArrowParquetError::General(
                    "Parquet row count does not fit in usize".to_string(),
                )
            })?;
        let arrow_metadata = ArrowReaderMetadata::try_new(
            Arc::clone(&metadata),
            ArrowReaderOptions::new(),
        )?;
        let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(
            chunk_reader,
            arrow_metadata,
        )
        .with_projection(projection)
        .with_batch_size(batch_size.max(1))
        .build()?;

        Ok(Self {
            reader,
            physical_position: 0,
            position: 0,
            row_count,
            repeated,
            pages,
            metadata,
            projected_leaf_column,
        })
    }

    /// Skips to `target_row` and decodes the next Arrow batch.
    ///
    /// Returns `None` when `target_row == self.row_count()`. Backward seeks and
    /// rows beyond the end of the file return an error.
    pub fn read_batch_at(
        &mut self,
        target_row: usize,
        max_rows: usize,
    ) -> ParquetResult<Option<RecordBatch>> {
        if target_row > self.row_count {
            return Err(ArrowParquetError::General(format!(
                "row {target_row} is beyond Parquet row count {}",
                self.row_count
            )));
        }
        if target_row < self.position {
            return Err(ArrowParquetError::General(format!(
                "backward seek from {} to {target_row} is not supported",
                self.position
            )));
        }
        if target_row == self.row_count {
            return Ok(None);
        }
        if max_rows == 0 {
            return Err(ArrowParquetError::General(
                "forward batch size must be greater than zero".to_string(),
            ));
        }

        let page = self.page_at(target_row)?.clone();
        let rows_to_read = max_rows.min(page.first_row + page.row_count - target_row);
        if page.all_null {
            let page_end = page.first_row + page.row_count;
            if self.physical_position < page_end {
                let to_skip = page_end - self.physical_position;
                let skipped = self.reader.skip_rows(to_skip)?;
                if skipped != to_skip {
                    return Err(ArrowParquetError::General(format!(
                        "requested all-null page skip of {to_skip} rows but skipped {skipped}"
                    )));
                }
                self.physical_position = page_end;
            }
            let schema = self.reader.schema();
            let columns = schema
                .fields()
                .iter()
                .map(|field| new_null_array(field.data_type(), rows_to_read))
                .collect();
            let batch = RecordBatch::try_new(schema, columns)?;
            self.position = target_row + rows_to_read;
            return Ok(Some(batch));
        }

        if target_row < self.physical_position {
            return Err(ArrowParquetError::General(format!(
                "physical reader is at {} before non-null row {target_row}",
                self.physical_position
            )));
        }
        let to_skip = target_row - self.physical_position;
        let skipped = self.reader.skip_rows(to_skip)?;
        if skipped != to_skip {
            return Err(ArrowParquetError::General(format!(
                "requested skip of {to_skip} rows but skipped {skipped}"
            )));
        }

        let batch = self.reader.read_next_batch(rows_to_read)?.ok_or_else(|| {
            ArrowParquetError::General(format!(
                "Parquet reader exhausted before row {target_row}"
            ))
        })?;
        self.physical_position = target_row + batch.num_rows();
        self.position = self.physical_position;
        Ok(Some(batch))
    }

    /// Skips to `target_row` and decodes up to `max_rows`, crossing data-page
    /// and row-group boundaries without rebuilding the retained Arrow reader.
    pub fn read_range_at(
        &mut self,
        target_row: usize,
        max_rows: usize,
    ) -> ParquetResult<Option<RecordBatch>> {
        let Some(first) = self.read_batch_at(target_row, max_rows)? else {
            return Ok(None);
        };
        let end = target_row.saturating_add(max_rows).min(self.row_count);
        if self.position >= end {
            return Ok(Some(first));
        }

        let schema = first.schema();
        let mut batches = vec![first];
        while self.position < end {
            let position = self.position;
            let batch =
                self.read_batch_at(position, end - position)?
                    .ok_or_else(|| {
                        ArrowParquetError::General(format!(
                            "Parquet reader exhausted before row {position}"
                        ))
                    })?;
            if batch.num_rows() == 0 {
                return Err(ArrowParquetError::General(format!(
                    "Parquet reader made no progress at row {position}"
                )));
            }
            batches.push(batch);
        }
        Ok(Some(arrow::compute::concat_batches(&schema, &batches)?))
    }

    /// Current physical row position of the retained Arrow reader.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Total rows in the Parquet file.
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Number of physical rows in the page containing `target_row`.
    pub fn page_row_count(&self, target_row: usize) -> ParquetResult<usize> {
        Ok(self.page_at(target_row)?.row_count)
    }

    /// Number of physical rows from `target_row` through the end of its page.
    pub fn rows_remaining_in_page(&self, target_row: usize) -> ParquetResult<usize> {
        let page = self.page_at(target_row)?;
        Ok(page.first_row + page.row_count - target_row)
    }

    /// Page metadata for the projected leaf column.
    pub fn pages(&self) -> &[ParquetForwardPage] {
        &self.pages
    }

    /// Parquet metadata containing the scoped OffsetIndex and ColumnIndex.
    pub fn metadata(&self) -> &ParquetMetaData {
        &self.metadata
    }

    /// Projected Parquet leaf-column index.
    pub fn projected_leaf_column(&self) -> usize {
        self.projected_leaf_column
    }

    /// Whether the projected leaf belongs to a repeated Parquet field.
    pub fn is_repeated(&self) -> bool {
        self.repeated
    }

    fn page_at(&self, target_row: usize) -> ParquetResult<&ParquetForwardPage> {
        let index = self
            .pages
            .partition_point(|page| page.first_row + page.row_count <= target_row);
        self.pages
            .get(index)
            .filter(|page| {
                target_row >= page.first_row
                    && target_row < page.first_row + page.row_count
            })
            .ok_or_else(|| {
                ArrowParquetError::General(format!(
                    "OffsetIndex does not contain row {target_row}"
                ))
            })
    }
}

fn projected_pages(
    metadata: &ParquetMetaData,
    projection: &ProjectionMask,
) -> ParquetResult<(usize, bool, Vec<ParquetForwardPage>)> {
    let schema = metadata.file_metadata().schema_descr();
    let projected_columns = (0..schema.num_columns())
        .filter(|&column_idx| projection.leaf_included(column_idx))
        .collect::<Vec<_>>();
    let [column_idx] = projected_columns.as_slice() else {
        return Err(ArrowParquetError::General(format!(
            "ParquetForwardBatchReader requires exactly one projected leaf column, got {}",
            projected_columns.len()
        )));
    };
    let repeated = schema.column(*column_idx).max_rep_level() > 0;

    let offset_index = metadata.offset_index().ok_or_else(|| {
        ArrowParquetError::General(
            "ParquetForwardBatchReader requires an OffsetIndex".to_string(),
        )
    })?;
    let mut row_group_start = 0usize;
    let column_index = metadata.column_index();
    let mut pages = vec![];
    for (row_group_idx, row_group) in metadata.row_groups().iter().enumerate() {
        let row_group_rows = usize::try_from(row_group.num_rows()).map_err(|_| {
            ArrowParquetError::General(format!(
                "negative row count for row group {row_group_idx}"
            ))
        })?;
        if row_group_rows == 0 {
            continue;
        }
        let locations = &offset_index
            .get(row_group_idx)
            .and_then(|row_group| row_group.get(*column_idx))
            .filter(|index| !index.page_locations.is_empty())
            .ok_or_else(|| {
                ArrowParquetError::General(format!(
                    "OffsetIndex missing for row group {row_group_idx}, column {column_idx}"
                ))
            })?
            .page_locations;

        // Repeated values can span data-page boundaries, and consecutive
        // PageLocations may therefore identify the same first logical row.
        // Arrow's retained ArrayReader still skips and decodes those pages
        // lazily using the full OffsetIndex. For cursor planning, expose one
        // non-overlapping logical region per row group instead of pretending
        // page boundaries are record boundaries.
        if repeated {
            let first = locations
                .first()
                .ok_or_else(|| {
                    ArrowParquetError::General(format!(
                        "OffsetIndex missing for row group {row_group_idx}, column {column_idx}"
                    ))
                })?;
            let compressed_size = locations
                .iter()
                .try_fold(0i64, |total, page| {
                    total.checked_add(i64::from(page.compressed_page_size))
                })
                .and_then(|total| i32::try_from(total).ok())
                .unwrap_or(i32::MAX);
            pages.push(ParquetForwardPage {
                row_group_index: row_group_idx,
                page_index: 0,
                first_row: row_group_start,
                row_count: row_group_rows,
                file_offset: first.offset,
                compressed_size,
                null_count: None,
                all_null: false,
            });
            row_group_start += row_group_rows;
            continue;
        }

        let page_statistics = column_index
            .and_then(|index| index.get(row_group_idx))
            .and_then(|row_group| row_group.get(*column_idx))
            .filter(|index| !matches!(index, ColumnIndexMetaData::NONE));

        for (page_idx, location) in locations.iter().enumerate() {
            let start = usize::try_from(location.first_row_index).map_err(|_| {
                ArrowParquetError::General(format!(
                    "negative first row for row group {row_group_idx}, page {page_idx}"
                ))
            })?;
            let end = match locations.get(page_idx + 1) {
                Some(next) => usize::try_from(next.first_row_index).map_err(|_| {
                    ArrowParquetError::General(format!(
                        "negative first row for row group {row_group_idx}, page {}",
                        page_idx + 1
                    ))
                })?,
                None => row_group_rows,
            };
            if start >= end || end > row_group_rows {
                return Err(ArrowParquetError::General(format!(
                    "invalid OffsetIndex row range {start}..{end} for row group {row_group_idx}"
                )));
            }
            let null_count = page_statistics.and_then(|index| {
                (page_idx < index.num_pages() as usize)
                    .then(|| index.null_count(page_idx))
                    .flatten()
            });
            let all_null = page_statistics.is_some_and(|index| {
                page_idx < index.num_pages() as usize && index.is_null_page(page_idx)
            }) || null_count == Some((end - start) as i64);
            pages.push(ParquetForwardPage {
                row_group_index: row_group_idx,
                page_index: page_idx,
                first_row: row_group_start + start,
                row_count: end - start,
                file_offset: location.offset,
                compressed_size: location.compressed_page_size,
                null_count,
                all_null,
            });
        }
        row_group_start += row_group_rows;
    }
    Ok((*column_idx, repeated, pages))
}

/// Adapts DataFusion's cache-aware async reader to Arrow's lazy synchronous
/// page reader. Calls are serialized because `AsyncFileReader` takes `&mut
/// self`, matching the single-threaded cursor contract.
struct AsyncFileChunkReader {
    reader: Mutex<Box<dyn AsyncFileReader + Send>>,
    file_len: u64,
    runtime: Arc<Runtime>,
}

impl Length for AsyncFileChunkReader {
    fn len(&self) -> u64 {
        self.file_len
    }
}

impl ChunkReader for AsyncFileChunkReader {
    type T = Cursor<Bytes>;

    fn get_read(&self, start: u64) -> ParquetResult<Self::T> {
        Err(ArrowParquetError::General(format!(
            "page-header scanning at byte {start} is disabled; an OffsetIndex is required"
        )))
    }

    fn get_bytes(&self, start: u64, length: usize) -> ParquetResult<Bytes> {
        let end = start.checked_add(length as u64).ok_or_else(|| {
            ArrowParquetError::General("page range overflow".to_string())
        })?;
        if end > self.file_len {
            return Err(ArrowParquetError::General(format!(
                "page range {start}..{end} exceeds file length {}",
                self.file_len
            )));
        }
        self.runtime
            .block_on(self.reader.lock().get_bytes(start..end))
    }
}

#[cfg(test)]
mod forward_batch_reader_tests {
    use super::*;
    use arrow::array::{Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::file::metadata::{PageIndexPolicy, ParquetMetaDataReader};
    use parquet::file::properties::WriterProperties;
    use std::fs::File;

    fn test_reader() -> ParquetForwardBatchReader {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from_iter_values(0..20))],
        )
        .unwrap();
        let file = tempfile::tempfile().unwrap();
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(8)
            .set_data_page_row_count_limit(3)
            .set_offset_index_disabled(false)
            .build();
        let mut writer =
            ArrowWriter::try_new(file.try_clone().unwrap(), schema, Some(properties))
                .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let metadata = ParquetMetaDataReader::new()
            .with_page_index_policy(PageIndexPolicy::Required)
            .parse_and_finish(&file)
            .unwrap();
        let metadata = Arc::new(metadata);
        let projection =
            ProjectionMask::leaves(metadata.file_metadata().schema_descr(), [0]);
        ParquetForwardBatchReader::try_new_with_chunk_reader(
            File::from(file),
            metadata,
            projection,
            20,
        )
        .unwrap()
    }

    fn values(batch: &RecordBatch) -> Vec<i32> {
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    #[test]
    fn read_range_crosses_page_boundary() {
        let batch = test_reader().read_range_at(2, 4).unwrap().unwrap();
        assert_eq!(values(&batch), vec![2, 3, 4, 5]);
    }

    #[test]
    fn read_range_crosses_row_group_boundary() {
        let batch = test_reader().read_range_at(6, 5).unwrap().unwrap();
        assert_eq!(values(&batch), vec![6, 7, 8, 9, 10]);
    }
}

/// Interface for reading Apache Parquet files.
///
/// The combined implementations of [`ParquetFileReaderFactory`] and
/// [`AsyncFileReader`] can be used to provide custom data access operations
/// such as pre-cached metadata, I/O coalescing, etc.
///
/// See [`DefaultParquetFileReaderFactory`] for a simple implementation.
pub trait ParquetFileReaderFactory: Debug + Send + Sync + 'static {
    /// Provides an `AsyncFileReader` for reading data from a parquet file specified
    ///
    /// # Notes
    ///
    /// If the resulting [`AsyncFileReader`]  returns `ParquetMetaData` without
    /// page index information, the reader will load it on demand. Thus it is important
    /// to ensure that the returned `ParquetMetaData` has the necessary information
    /// if you wish to avoid a subsequent I/O
    ///
    /// # Arguments
    /// * partition_index - Index of the partition (for reporting metrics)
    /// * file - The file to be read
    /// * metadata_size_hint - If specified, the first IO reads this many bytes from the footer
    /// * metrics - Execution metrics
    fn create_reader(
        &self,
        partition_index: usize,
        partitioned_file: PartitionedFile,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> datafusion_common::Result<Box<dyn AsyncFileReader + Send>>;
}

/// Default implementation of [`ParquetFileReaderFactory`]
///
/// This implementation:
/// 1. Reads parquet directly from an underlying [`ObjectStore`] instance.
/// 2. Reads the footer and page metadata on demand.
/// 3. Does not cache metadata or coalesce I/O operations.
#[derive(Debug)]
pub struct DefaultParquetFileReaderFactory {
    store: Arc<dyn ObjectStore>,
}

impl DefaultParquetFileReaderFactory {
    /// Create a new `DefaultParquetFileReaderFactory`.
    pub fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self { store }
    }
}

/// Implements [`AsyncFileReader`] for a parquet file in object storage.
///
/// This implementation uses the [`ParquetObjectReader`] to read data from the
/// object store on demand, as required, tracking the number of bytes read.
///
/// This implementation does not coalesce I/O operations or cache bytes. Such
/// optimizations can be done either at the object store level or by providing a
/// custom implementation of [`ParquetFileReaderFactory`].
pub struct ParquetFileReader {
    pub file_metrics: ParquetFileMetrics,
    pub inner: ParquetObjectReader,
    pub partitioned_file: PartitionedFile,
}

impl AsyncFileReader for ParquetFileReader {
    fn get_bytes(
        &mut self,
        range: Range<u64>,
    ) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        let bytes_scanned = range.end - range.start;
        self.file_metrics.bytes_scanned.add(bytes_scanned as usize);
        self.inner.get_bytes(range)
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>>
    where
        Self: Send,
    {
        let total: u64 = ranges.iter().map(|r| r.end - r.start).sum();
        self.file_metrics.bytes_scanned.add(total as usize);
        self.inner.get_byte_ranges(ranges)
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, parquet::errors::Result<Arc<ParquetMetaData>>> {
        self.inner.get_metadata(options)
    }
}

impl Drop for ParquetFileReader {
    fn drop(&mut self) {
        self.file_metrics
            .scan_efficiency_ratio
            .add_part(self.file_metrics.bytes_scanned.value());
        // Multiple ParquetFileReaders may run, so we set_total to avoid adding the total multiple times
        self.file_metrics
            .scan_efficiency_ratio
            .set_total(self.partitioned_file.object_meta.size as usize);
    }
}

impl ParquetFileReaderFactory for DefaultParquetFileReaderFactory {
    fn create_reader(
        &self,
        partition_index: usize,
        partitioned_file: PartitionedFile,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> datafusion_common::Result<Box<dyn AsyncFileReader + Send>> {
        let file_metrics = ParquetFileMetrics::new(
            partition_index,
            partitioned_file.object_meta.location.as_ref(),
            metrics,
        );
        let store = Arc::clone(&self.store);
        let mut inner = ParquetObjectReader::new(
            store,
            partitioned_file.object_meta.location.clone(),
        )
        .with_file_size(partitioned_file.object_meta.size);

        if let Some(hint) = metadata_size_hint {
            inner = inner.with_footer_size_hint(hint)
        };

        Ok(Box::new(ParquetFileReader {
            inner,
            file_metrics,
            partitioned_file,
        }))
    }
}

/// Implementation of [`ParquetFileReaderFactory`] supporting the caching of footer and page
/// metadata. Reads and updates the [`FileMetadataCache`] with the [`ParquetMetaData`] data.
/// This reader always loads the entire metadata (including page index, unless the file is
/// encrypted), even if not required by the current query, to ensure it is always available for
/// those that need it.
#[derive(Debug)]
pub struct CachedParquetFileReaderFactory {
    store: Arc<dyn ObjectStore>,
    metadata_cache: Arc<dyn FileMetadataCache>,
}

impl CachedParquetFileReaderFactory {
    pub fn new(
        store: Arc<dyn ObjectStore>,
        metadata_cache: Arc<dyn FileMetadataCache>,
    ) -> Self {
        Self {
            store,
            metadata_cache,
        }
    }
}

impl ParquetFileReaderFactory for CachedParquetFileReaderFactory {
    fn create_reader(
        &self,
        partition_index: usize,
        partitioned_file: PartitionedFile,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> datafusion_common::Result<Box<dyn AsyncFileReader + Send>> {
        let file_metrics = ParquetFileMetrics::new(
            partition_index,
            partitioned_file.object_meta.location.as_ref(),
            metrics,
        );
        let store = Arc::clone(&self.store);

        let mut inner = ParquetObjectReader::new(
            store,
            partitioned_file.object_meta.location.clone(),
        )
        .with_file_size(partitioned_file.object_meta.size);

        if let Some(hint) = metadata_size_hint {
            inner = inner.with_footer_size_hint(hint)
        };

        Ok(Box::new(CachedParquetFileReader::new(
            file_metrics,
            Arc::clone(&self.store),
            inner,
            partitioned_file,
            Arc::clone(&self.metadata_cache),
            metadata_size_hint,
        )))
    }
}

/// Implements [`AsyncFileReader`] for a Parquet file in object storage. Reads the file metadata
/// from the [`FileMetadataCache`], if available, otherwise reads it directly from the file and then
/// updates the cache.
pub struct CachedParquetFileReader {
    pub file_metrics: ParquetFileMetrics,
    store: Arc<dyn ObjectStore>,
    pub inner: ParquetObjectReader,
    partitioned_file: PartitionedFile,
    metadata_cache: Arc<dyn FileMetadataCache>,
    metadata_size_hint: Option<usize>,
}

impl CachedParquetFileReader {
    pub fn new(
        file_metrics: ParquetFileMetrics,
        store: Arc<dyn ObjectStore>,
        inner: ParquetObjectReader,
        partitioned_file: PartitionedFile,
        metadata_cache: Arc<dyn FileMetadataCache>,
        metadata_size_hint: Option<usize>,
    ) -> Self {
        Self {
            file_metrics,
            store,
            inner,
            partitioned_file,
            metadata_cache,
            metadata_size_hint,
        }
    }
}

impl AsyncFileReader for CachedParquetFileReader {
    fn get_bytes(
        &mut self,
        range: Range<u64>,
    ) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        let bytes_scanned = range.end - range.start;
        self.file_metrics.bytes_scanned.add(bytes_scanned as usize);
        self.inner.get_bytes(range)
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>>
    where
        Self: Send,
    {
        let total: u64 = ranges.iter().map(|r| r.end - r.start).sum();
        self.file_metrics.bytes_scanned.add(total as usize);
        self.inner.get_byte_ranges(ranges)
    }

    fn get_metadata<'a>(
        &'a mut self,
        #[cfg_attr(not(feature = "parquet_encryption"), expect(unused_variables))]
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, parquet::errors::Result<Arc<ParquetMetaData>>> {
        let object_meta = self.partitioned_file.object_meta.clone();
        let metadata_cache = Arc::clone(&self.metadata_cache);

        async move {
            #[cfg(feature = "parquet_encryption")]
            let file_decryption_properties = options
                .and_then(|o| o.file_decryption_properties())
                .map(Arc::clone);

            #[cfg(not(feature = "parquet_encryption"))]
            let file_decryption_properties = None;

            DFParquetMetadata::new(&self.store, &object_meta)
                .with_decryption_properties(file_decryption_properties)
                .with_file_metadata_cache(Some(Arc::clone(&metadata_cache)))
                .with_metadata_size_hint(self.metadata_size_hint)
                .fetch_metadata()
                .await
                .map_err(|e| {
                    parquet::errors::ParquetError::General(format!(
                        "Failed to fetch metadata for file {}: {e}",
                        object_meta.location,
                    ))
                })
        }
        .boxed()
    }
}

impl Drop for CachedParquetFileReader {
    fn drop(&mut self) {
        self.file_metrics
            .scan_efficiency_ratio
            .add_part(self.file_metrics.bytes_scanned.value());
        // Multiple ParquetFileReaders may run, so we set_total to avoid adding the total multiple times
        self.file_metrics
            .scan_efficiency_ratio
            .set_total(self.partitioned_file.object_meta.size as usize);
    }
}

/// Wrapper to implement [`FileMetadata`] for [`ParquetMetaData`].
pub struct CachedParquetMetaData(Arc<ParquetMetaData>);

impl CachedParquetMetaData {
    pub fn new(metadata: Arc<ParquetMetaData>) -> Self {
        Self(metadata)
    }

    pub fn parquet_metadata(&self) -> &Arc<ParquetMetaData> {
        &self.0
    }
}

impl FileMetadata for CachedParquetMetaData {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn memory_size(&self) -> usize {
        self.0.memory_size()
    }

    fn extra_info(&self) -> HashMap<String, String> {
        let page_index =
            self.0.column_index().is_some() && self.0.offset_index().is_some();
        HashMap::from([("page_index".to_owned(), page_index.to_string())])
    }
}
