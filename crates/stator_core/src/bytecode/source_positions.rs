//! Source-position table: maps bytecode offsets to locations in JavaScript
//! source code.
//!
//! The table is used for:
//!
//! - **Error stack traces** — translating a bytecode offset from an exception's
//!   call-frame into a human-readable `line:column`.
//! - **Debugger stepping** — entries marked [`SourcePositionEntry::is_statement`]
//!   indicate valid breakpoint locations.
//! - **Source maps** — building a V3 source-map that relates generated
//!   bytecode back to the original source file.
//!
//! # Example
//!
//! ```
//! use stator_core::bytecode::source_positions::{SourcePositionEntry, SourcePositionTable};
//!
//! let mut table = SourcePositionTable::new();
//! table.add_entry(SourcePositionEntry::new(0,  1, 1, true));
//! table.add_entry(SourcePositionEntry::new(4,  1, 5, false));
//! table.add_entry(SourcePositionEntry::new(8,  2, 1, true));
//!
//! // Exact offset hit.
//! let entry = table.lookup(4).unwrap();
//! assert_eq!(entry.line, 1);
//! assert_eq!(entry.column, 5);
//! assert!(!entry.is_statement);
//!
//! // Offset between entries falls back to the last preceding entry.
//! let entry = table.lookup(6).unwrap();
//! assert_eq!(entry.line, 1);
//! assert_eq!(entry.column, 5);
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// SourcePositionEntry
// ─────────────────────────────────────────────────────────────────────────────

/// A single record in a [`SourcePositionTable`].
///
/// Each entry associates a bytecode offset with the corresponding location in
/// the original JavaScript source and notes whether the position is the start
/// of a statement (relevant for debugger breakpoints and stepping).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourcePositionEntry {
    /// Byte offset within the encoded bytecode stream.
    pub bytecode_offset: u32,
    /// 1-based source line number.
    pub line: u32,
    /// 1-based source column number.
    pub column: u32,
    /// `true` when this position marks the beginning of a statement.
    ///
    /// Statement positions are valid debugger breakpoint locations; expression
    /// positions (e.g. sub-expressions inside a larger statement) are not.
    pub is_statement: bool,
}

impl SourcePositionEntry {
    /// Construct a new [`SourcePositionEntry`].
    pub fn new(bytecode_offset: u32, line: u32, column: u32, is_statement: bool) -> Self {
        Self {
            bytecode_offset,
            line,
            column,
            is_statement,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SourcePositionTable
// ─────────────────────────────────────────────────────────────────────────────

/// A sorted table that maps bytecode offsets to source positions.
///
/// Entries must be added in non-decreasing `bytecode_offset` order (the same
/// order in which the bytecode generator emits them).  [`SourcePositionTable::lookup`]
/// then performs a binary-search to find the entry whose `bytecode_offset` is
/// closest to (and ≤) the queried offset.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SourcePositionTable {
    entries: Vec<SourcePositionEntry>,
}

impl SourcePositionTable {
    /// Create an empty `SourcePositionTable`.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Append a new entry to the table.
    ///
    /// Entries **must** be appended in non-decreasing `bytecode_offset` order
    /// to maintain the binary-search invariant used by [`lookup`](Self::lookup).
    pub fn add_entry(&mut self, entry: SourcePositionEntry) {
        debug_assert!(
            self.entries
                .last()
                .is_none_or(|last| last.bytecode_offset <= entry.bytecode_offset),
            "entries must be added in non-decreasing bytecode_offset order"
        );
        self.entries.push(entry);
    }

    /// Return the number of entries in the table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the table contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The full ordered slice of entries.
    pub fn entries(&self) -> &[SourcePositionEntry] {
        &self.entries
    }

    /// Look up the [`SourcePositionEntry`] that covers `bytecode_offset`.
    ///
    /// Returns the last entry whose `bytecode_offset` is ≤ `bytecode_offset`,
    /// or `None` if the table is empty or all entries follow the given offset.
    ///
    /// The lookup is O(log n) via binary search.
    pub fn lookup(&self, bytecode_offset: u32) -> Option<&SourcePositionEntry> {
        let idx = self
            .entries
            .partition_point(|e| e.bytecode_offset <= bytecode_offset);
        idx.checked_sub(1).map(|i| &self.entries[i])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_table() -> SourcePositionTable {
        let mut t = SourcePositionTable::new();
        t.add_entry(SourcePositionEntry::new(0, 1, 1, true));
        t.add_entry(SourcePositionEntry::new(4, 1, 5, false));
        t.add_entry(SourcePositionEntry::new(8, 2, 1, true));
        t
    }

    // ── build ────────────────────────────────────────────────────────────────

    #[test]
    fn test_build_table_len() {
        let t = make_table();
        assert_eq!(t.len(), 3);
        assert!(!t.is_empty());
    }

    #[test]
    fn test_build_empty_table() {
        let t = SourcePositionTable::new();
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
    }

    #[test]
    fn test_entries_slice() {
        let t = make_table();
        let entries = t.entries();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0], SourcePositionEntry::new(0, 1, 1, true));
        assert_eq!(entries[1], SourcePositionEntry::new(4, 1, 5, false));
        assert_eq!(entries[2], SourcePositionEntry::new(8, 2, 1, true));
    }

    // ── lookup ───────────────────────────────────────────────────────────────

    #[test]
    fn test_lookup_exact_first_entry() {
        let t = make_table();
        let e = t.lookup(0).unwrap();
        assert_eq!(e.bytecode_offset, 0);
        assert_eq!(e.line, 1);
        assert_eq!(e.column, 1);
        assert!(e.is_statement);
    }

    #[test]
    fn test_lookup_exact_middle_entry() {
        let t = make_table();
        let e = t.lookup(4).unwrap();
        assert_eq!(e.bytecode_offset, 4);
        assert_eq!(e.line, 1);
        assert_eq!(e.column, 5);
        assert!(!e.is_statement);
    }

    #[test]
    fn test_lookup_exact_last_entry() {
        let t = make_table();
        let e = t.lookup(8).unwrap();
        assert_eq!(e.bytecode_offset, 8);
        assert_eq!(e.line, 2);
        assert_eq!(e.column, 1);
        assert!(e.is_statement);
    }

    #[test]
    fn test_lookup_between_entries_returns_preceding() {
        let t = make_table();
        // Offset 2 is between entry 0 (offset 0) and entry 1 (offset 4).
        let e = t.lookup(2).unwrap();
        assert_eq!(e.bytecode_offset, 0);

        // Offset 6 is between entry 1 (offset 4) and entry 2 (offset 8).
        let e = t.lookup(6).unwrap();
        assert_eq!(e.bytecode_offset, 4);
    }

    #[test]
    fn test_lookup_beyond_last_entry() {
        let t = make_table();
        // Any offset past the last entry should return the last entry.
        let e = t.lookup(100).unwrap();
        assert_eq!(e.bytecode_offset, 8);
    }

    #[test]
    fn test_lookup_empty_table_returns_none() {
        let t = SourcePositionTable::new();
        assert!(t.lookup(0).is_none());
        assert!(t.lookup(42).is_none());
    }

    #[test]
    fn test_lookup_before_first_entry_returns_none() {
        // If the first entry is at offset 4, querying offset 2 returns None.
        let mut t = SourcePositionTable::new();
        t.add_entry(SourcePositionEntry::new(4, 1, 1, true));
        assert!(t.lookup(0).is_none());
        assert!(t.lookup(3).is_none());
        assert!(t.lookup(4).is_some());
    }

    // ── is_statement flag ────────────────────────────────────────────────────

    #[test]
    fn test_is_statement_field() {
        let mut t = SourcePositionTable::new();
        t.add_entry(SourcePositionEntry::new(0, 1, 1, true));
        t.add_entry(SourcePositionEntry::new(2, 1, 3, false));

        assert!(t.lookup(0).unwrap().is_statement);
        assert!(!t.lookup(2).unwrap().is_statement);
    }

    // ── single-entry table ───────────────────────────────────────────────────

    #[test]
    fn test_single_entry_table() {
        let mut t = SourcePositionTable::new();
        t.add_entry(SourcePositionEntry::new(10, 5, 3, true));

        assert!(t.lookup(9).is_none());
        let e = t.lookup(10).unwrap();
        assert_eq!(e.line, 5);
        assert_eq!(e.column, 3);
        assert!(t.lookup(99).is_some());
    }

    // ── default ──────────────────────────────────────────────────────────────

    #[test]
    fn test_default_is_empty() {
        let t = SourcePositionTable::default();
        assert!(t.is_empty());
    }
}
