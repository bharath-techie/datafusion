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

//! Selectively decoded Parquet page indexes, cached with the file metadata.
//!
//! The Parquet opener derives which page-index entries a scan needs
//! ([`PageIndexSelection::for_scan`]), reuses entries already stored on the
//! file's `FileMetadataCache` entry
//! ([`CachedParquetMetaData`](crate::metadata::CachedParquetMetaData)),
//! decodes only the missing entries, merges them back, and hands the result
//! to the reader as a
//! [`PageIndexProvider`](parquet::file::page_index::provider::PageIndexProvider).
//!
//! The footer and its scoped page indexes share one cache entry: admitted,
//! memory-accounted, and evicted together. The shared footer
//! [`ParquetMetaData`](parquet::file::metadata::ParquetMetaData) itself is
//! never cloned or rebuilt: page indexes travel out of band through the
//! provider.

use parquet::file::page_index::column_index::ColumnIndexMetaData;
use parquet::file::page_index::offset_index::OffsetIndexMetaData;
use parquet::file::page_index::provider::PageIndexProvider;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

/// Identifies one Parquet column chunk within a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageIndexKey {
    /// Zero-based row-group index.
    pub row_group_index: usize,
    /// Zero-based physical Parquet leaf-column index.
    pub column_index: usize,
}

impl PageIndexKey {
    /// Creates a new page-index key.
    pub fn new(row_group_index: usize, column_index: usize) -> Self {
        Self {
            row_group_index,
            column_index,
        }
    }
}

/// Page-index entries required by a Parquet scan.
///
/// Column-index and offset-index selections are independent because page
/// pruning needs column indexes only for predicate columns, while applying
/// the resulting row selection requires offset indexes for all projected
/// columns.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PageIndexSelection {
    column_indexes: HashSet<PageIndexKey>,
    offset_indexes: HashSet<PageIndexKey>,
}

impl PageIndexSelection {
    /// Creates an empty selection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a column-index entry to this selection.
    pub fn with_column_index(mut self, key: PageIndexKey) -> Self {
        self.column_indexes.insert(key);
        self
    }

    /// Adds an offset-index entry to this selection.
    pub fn with_offset_index(mut self, key: PageIndexKey) -> Self {
        self.offset_indexes.insert(key);
        self
    }

    /// Returns the selected column-index entries.
    pub fn column_indexes(&self) -> &HashSet<PageIndexKey> {
        &self.column_indexes
    }

    /// Returns the selected offset-index entries.
    pub fn offset_indexes(&self) -> &HashSet<PageIndexKey> {
        &self.offset_indexes
    }

    /// Returns `true` if no page-index entries are selected.
    pub fn is_empty(&self) -> bool {
        self.column_indexes.is_empty() && self.offset_indexes.is_empty()
    }

    /// Builds the selection for a scan after footer-based row-group pruning.
    ///
    /// The inputs correspond to information the Parquet opener has once row
    /// groups have been pruned with footer statistics:
    ///
    /// - `surviving_row_groups`: row groups remaining in the access plan.
    /// - `predicate_leaf_columns`: physical leaf columns referenced by the
    ///   page-pruning predicate. These need **column indexes** (min/max/null
    ///   statistics per page) to evaluate the predicate, and **offset
    ///   indexes** to convert matching pages into row ranges.
    /// - `projected_leaf_columns`: physical leaf columns read by the decoder,
    ///   including columns decoded by pushed-down row filters. These need
    ///   **offset indexes** so the resulting row selection can be applied
    ///   while reading, but no column indexes.
    pub fn for_scan(
        surviving_row_groups: impl IntoIterator<Item = usize>,
        predicate_leaf_columns: &[usize],
        projected_leaf_columns: &[usize],
    ) -> Self {
        let mut selection = Self::new();
        for row_group_index in surviving_row_groups {
            for &column in predicate_leaf_columns {
                let key = PageIndexKey::new(row_group_index, column);
                selection.column_indexes.insert(key);
                selection.offset_indexes.insert(key);
            }
            for &column in projected_leaf_columns {
                selection
                    .offset_indexes
                    .insert(PageIndexKey::new(row_group_index, column));
            }
        }
        selection
    }
}

/// A sparse collection of decoded page-index entries.
///
/// Entries not present have not been supplied by the cache; the Parquet
/// footer remains the source of truth for whether an index exists in the
/// file.
///
/// Implements [`PageIndexProvider`], so once the opener has resolved a
/// scan's selection the collection is handed directly to page pruning and
/// the Arrow reader. The shared footer metadata is never modified.
#[derive(Debug, Clone, Default)]
pub struct CachedPageIndexes {
    column_indexes: HashMap<PageIndexKey, Arc<ColumnIndexMetaData>>,
    offset_indexes: HashMap<PageIndexKey, Arc<OffsetIndexMetaData>>,
}

impl CachedPageIndexes {
    /// Creates an empty collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a decoded column index.
    pub fn with_column_index(
        mut self,
        key: PageIndexKey,
        index: Arc<ColumnIndexMetaData>,
    ) -> Self {
        self.column_indexes.insert(key, index);
        self
    }

    /// Adds a decoded offset index.
    pub fn with_offset_index(
        mut self,
        key: PageIndexKey,
        index: Arc<OffsetIndexMetaData>,
    ) -> Self {
        self.offset_indexes.insert(key, index);
        self
    }

    /// Returns a decoded column index, if present.
    pub fn column_index(&self, key: &PageIndexKey) -> Option<&Arc<ColumnIndexMetaData>> {
        self.column_indexes.get(key)
    }

    /// Returns a decoded offset index, if present.
    pub fn offset_index(&self, key: &PageIndexKey) -> Option<&Arc<OffsetIndexMetaData>> {
        self.offset_indexes.get(key)
    }

    /// Returns the subset of `selection` not present in this collection.
    pub fn missing(&self, selection: &PageIndexSelection) -> PageIndexSelection {
        PageIndexSelection {
            column_indexes: selection
                .column_indexes
                .iter()
                .filter(|key| !self.column_indexes.contains_key(key))
                .copied()
                .collect(),
            offset_indexes: selection
                .offset_indexes
                .iter()
                .filter(|key| !self.offset_indexes.contains_key(key))
                .copied()
                .collect(),
        }
    }

    /// Returns the subset of this collection matching `selection`.
    pub fn subset(&self, selection: &PageIndexSelection) -> CachedPageIndexes {
        let mut result = CachedPageIndexes::new();
        for key in selection.column_indexes() {
            if let Some(index) = self.column_indexes.get(key) {
                result = result.with_column_index(*key, Arc::clone(index));
            }
        }
        for key in selection.offset_indexes() {
            if let Some(index) = self.offset_indexes.get(key) {
                result = result.with_offset_index(*key, Arc::clone(index));
            }
        }
        result
    }

    /// Returns the number of decoded entries (column plus offset indexes).
    pub fn len(&self) -> usize {
        self.column_indexes.len() + self.offset_indexes.len()
    }

    /// Returns `true` if no entries are present.
    pub fn is_empty(&self) -> bool {
        self.column_indexes.is_empty() && self.offset_indexes.is_empty()
    }

    /// Rough estimate of the heap memory held by the decoded entries, used
    /// for cache memory accounting.
    ///
    /// The `parquet` crate's precise `HeapSize` trait is not public, so this
    /// approximates each entry by its page count times the per-page footprint
    /// (page location plus min/max statistics references).
    pub fn memory_estimate(&self) -> usize {
        use parquet::file::page_index::offset_index::PageLocation;
        use std::mem::{size_of, size_of_val};

        let offset_bytes: usize = self
            .offset_indexes
            .values()
            .map(|index| {
                size_of::<OffsetIndexMetaData>()
                    + index.page_locations().len() * size_of::<PageLocation>()
            })
            .sum();
        let column_bytes: usize = self
            .column_indexes
            .values()
            .map(|index| {
                // ColumnIndexMetaData stores per-page min/max/null-count
                // vectors; approximate 64 bytes per page.
                size_of_val(index.as_ref()) + index.num_pages() as usize * 64
            })
            .sum();
        offset_bytes + column_bytes
    }

    /// Merges another sparse collection into this one.
    pub fn extend(&mut self, other: Self) {
        self.column_indexes.extend(other.column_indexes);
        self.offset_indexes.extend(other.offset_indexes);
    }
}

/// The decoded entries are served to the Arrow reader and page pruning
/// through the `parquet` crate's provider interface. Cells that were not
/// requested (pruned row groups, unprojected columns) are `None` and
/// consumers fall back conservatively.
impl PageIndexProvider for CachedPageIndexes {
    fn column_index(
        &self,
        row_group_index: usize,
        column_index: usize,
    ) -> Option<&ColumnIndexMetaData> {
        self.column_indexes
            .get(&PageIndexKey::new(row_group_index, column_index))
            .map(Arc::as_ref)
    }

    fn offset_index(
        &self,
        row_group_index: usize,
        column_index: usize,
    ) -> Option<&OffsetIndexMetaData> {
        self.offset_indexes
            .get(&PageIndexKey::new(row_group_index, column_index))
            .map(Arc::as_ref)
    }
}

/// Computes the byte ranges needed to load the page indexes in `selection`,
/// using the index locations recorded in the footer.
///
/// The logical selection determines *which entries* to decode, while the
/// caller decides *which bytes* to fetch (exact ranges, coalesced ranges, or
/// one covering range). Entries whose footer locations are absent do not
/// exist in the file and produce no range.
pub fn required_ranges(
    footer: &parquet::file::metadata::ParquetMetaData,
    selection: &PageIndexSelection,
) -> Vec<Range<u64>> {
    let mut ranges = Vec::new();
    for key in selection.column_indexes() {
        let column = footer
            .row_group(key.row_group_index)
            .column(key.column_index);
        if let (Some(offset), Some(length)) =
            (column.column_index_offset(), column.column_index_length())
        {
            ranges.push(offset as u64..offset as u64 + length as u64);
        }
    }
    for key in selection.offset_indexes() {
        let column = footer
            .row_group(key.row_group_index)
            .column(key.column_index);
        if let (Some(offset), Some(length)) =
            (column.offset_index_offset(), column.offset_index_length())
        {
            ranges.push(offset as u64..offset as u64 + length as u64);
        }
    }
    ranges.sort_by_key(|range| range.start);
    ranges
}

/// Coalesces sorted ranges whose gap is at most `max_gap` bytes.
///
/// Storage decides the trade-off: local files can use `max_gap = 0` (exact
/// reads), remote object stores can use a large gap or fetch the covering
/// range, while decoding stays limited to the logical selection either way.
pub fn coalesce_ranges(ranges: Vec<Range<u64>>, max_gap: u64) -> Vec<Range<u64>> {
    let mut merged: Vec<Range<u64>> = Vec::new();
    for range in ranges {
        match merged.last_mut() {
            Some(last) if range.start <= last.end.saturating_add(max_gap) => {
                last.end = last.end.max(range.end);
            }
            _ => merged.push(range),
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use bytes::Bytes;
    use parquet::arrow::ArrowWriter;
    use parquet::file::metadata::{PageIndexPolicy, ParquetMetaDataReader};
    use parquet::file::properties::WriterProperties;

    /// Writes a real Parquet file with columns `a`, `b`, `c` and two row
    /// groups of four rows each. Page indexes are written by default.
    fn write_parquet_file() -> Bytes {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
            Field::new("c", DataType::Int32, false),
        ]));
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(4))
            .build();
        let mut buf = Vec::new();
        let mut writer =
            ArrowWriter::try_new(&mut buf, Arc::clone(&schema), Some(props)).unwrap();
        let column = |base: i32| -> ArrayRef {
            Arc::new(Int32Array::from_iter_values(base..base + 8))
        };
        let batch =
            RecordBatch::try_new(schema, vec![column(0), column(100), column(200)])
                .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        Bytes::from(buf)
    }

    /// Stand-in for a scoped Arrow decode API: decodes the file's page
    /// indexes and copies only the entries in `selection` into a sparse
    /// result.
    fn decode_selected_page_indexes(
        file: &Bytes,
        selection: &PageIndexSelection,
    ) -> CachedPageIndexes {
        let full = ParquetMetaDataReader::new()
            .with_page_index_policy(PageIndexPolicy::Required)
            .parse_and_finish(file)
            .unwrap();
        let column_index = full.column_index().unwrap();
        let offset_index = full.offset_index().unwrap();

        let mut decoded = CachedPageIndexes::new();
        for key in selection.column_indexes() {
            decoded = decoded.with_column_index(
                *key,
                Arc::new(column_index[key.row_group_index][key.column_index].clone()),
            );
        }
        for key in selection.offset_indexes() {
            decoded = decoded.with_offset_index(
                *key,
                Arc::new(offset_index[key.row_group_index][key.column_index].clone()),
            );
        }
        decoded
    }

    #[test]
    fn reports_only_uncached_entries_as_missing() {
        let cached_column_key = PageIndexKey::new(1, 2);
        let missing_column_key = PageIndexKey::new(1, 3);
        let missing_offset_key = PageIndexKey::new(1, 4);
        let selection = PageIndexSelection::new()
            .with_column_index(cached_column_key)
            .with_column_index(missing_column_key)
            .with_offset_index(missing_offset_key);

        let cached = CachedPageIndexes::new()
            .with_column_index(cached_column_key, Arc::new(ColumnIndexMetaData::NONE));
        let missing = cached.missing(&selection);

        assert!(!missing.column_indexes().contains(&cached_column_key));
        assert!(missing.column_indexes().contains(&missing_column_key));
        assert!(missing.offset_indexes().contains(&missing_offset_key));
    }

    #[test]
    fn extend_retains_existing_entries_and_adds_new_entries() {
        let first_key = PageIndexKey::new(0, 1);
        let second_key = PageIndexKey::new(2, 3);
        let mut cached = CachedPageIndexes::new()
            .with_column_index(first_key, Arc::new(ColumnIndexMetaData::NONE));

        cached.extend(
            CachedPageIndexes::new()
                .with_column_index(second_key, Arc::new(ColumnIndexMetaData::NONE)),
        );

        assert!(cached.column_index(&first_key).is_some());
        assert!(cached.column_index(&second_key).is_some());
    }

    /// End-to-end flow for `SELECT a, c FROM t WHERE a > 5`:
    ///
    /// 1. Footer statistics prune row group 0, so only row group 1 survives.
    /// 2. The plan is consumed into a `PageIndexSelection`.
    /// 3. The metadata-cache entry has no entries yet; the missing ones are
    ///    decoded and merged into the entry.
    /// 4. A second query with projection `a, b` reuses the merged entries
    ///    and only needs the offset index for `b`.
    /// 5. The decoded entries serve the reader directly as a
    ///    `PageIndexProvider`; the shared footer is never touched.
    #[test]
    fn scan_plan_is_consumed_into_selection_cache_and_provider() {
        use crate::metadata::CachedParquetMetaData;

        let file = write_parquet_file();

        // Shared footer-only metadata, as FileMetadataCache stores it.
        let footer = Arc::new(
            ParquetMetaDataReader::new()
                .with_page_index_policy(PageIndexPolicy::Skip)
                .parse_and_finish(&file)
                .unwrap(),
        );
        assert_eq!(footer.num_row_groups(), 2);
        assert!(footer.column_index().is_none());

        // Plan inputs after footer pruning -> logical selection.
        let selection = PageIndexSelection::for_scan(vec![1], &[0], &[0, 2]);

        // Fetch planning: 3 entries -> 3 exact ranges, or 1 coalesced.
        let exact = required_ranges(&footer, &selection);
        assert_eq!(exact.len(), 3);
        assert_eq!(coalesce_ranges(exact, u64::MAX).len(), 1);

        // The metadata-cache entry holds footer + (initially empty) scoped
        // page indexes, as it would sit inside `FileMetadataCache`.
        let entry = CachedParquetMetaData::new(Arc::clone(&footer));

        // Miss -> decode only the missing entries and merge into the entry.
        let mut indexes = entry.page_indexes_for(&selection);
        let missing = indexes.missing(&selection);
        assert_eq!(missing, selection, "first query misses everything");
        let decoded = decode_selected_page_indexes(&file, &missing);
        entry.merge_page_indexes(decoded.clone());
        indexes.extend(decoded);

        // An overlapping query (SELECT a, b WHERE a > 5) only needs the
        // offset index for column b (leaf 1).
        let second_selection = PageIndexSelection::for_scan(vec![1], &[0], &[0, 1]);
        let second_missing = entry
            .page_indexes_for(&second_selection)
            .missing(&second_selection);
        assert_eq!(second_missing.column_indexes().len(), 0);
        assert_eq!(
            second_missing.offset_indexes(),
            &HashSet::from([PageIndexKey::new(1, 1)])
        );

        // The decoded entries ARE the provider: selected cells are served,
        // unrequested cells are None, and the footer never changed.
        let provider: &dyn PageIndexProvider = &indexes;
        assert!(provider.column_index(1, 0).is_some());
        assert!(provider.offset_index(1, 2).is_some());
        assert!(provider.column_index(0, 0).is_none()); // pruned row group
        assert!(provider.offset_index(1, 1).is_none()); // unprojected column
        assert!(footer.column_index().is_none());
        assert!(footer.offset_index().is_none());
    }
}
