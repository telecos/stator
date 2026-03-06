//! String interning and forwarding types for property-key deduplication.
//!
//! This module extends [`crate::objects::string`] with the remaining V8-style
//! string types needed for efficient property-key handling:
//!
//! - [`InternalizedString`] — an interned string for O(1) identity comparison
//! - [`ThinString`] — a forwarding wrapper that redirects to an interned copy
//! - [`StringTable`] — the global deduplication table for property keys
//!
//! These types are designed to sit alongside the existing [`JsString`] enum
//! without replacing it.  A future integration phase will unify both into a
//! single representation.

use std::collections::HashMap;
use std::sync::Arc;

use crate::gc::trace::{Trace, Tracer};
use crate::objects::string::JsString;

// ──────────────────────────────────────────────────────────────────────────────
// InternalizedString
// ──────────────────────────────────────────────────────────────────────────────

/// An interned (internalized) string used for property keys.
///
/// Two `InternalizedString` values with the same content are guaranteed to
/// share the same [`Arc`] allocation when created through a [`StringTable`],
/// enabling O(1) identity comparison via [`InternalizedString::ptr_eq`].
///
/// The FNV-1a hash is cached at creation time so that string-table lookups
/// and hash-map operations never need to re-traverse the character data.
#[derive(Debug, Clone)]
pub struct InternalizedString {
    /// The underlying flat string data (always `SeqOneByte` or `SeqTwoByte`).
    inner: JsString,
    /// Cached FNV-1a hash of the string's UTF-16 code units.
    hash: u32,
}

impl InternalizedString {
    /// Returns the cached FNV-1a hash.
    #[inline]
    pub fn hash(&self) -> u32 {
        self.hash
    }

    /// Returns the number of UTF-16 code units.
    #[inline]
    pub fn length(&self) -> usize {
        self.inner.length()
    }

    /// Returns the UTF-16 code unit at `index`, or `None` if out of bounds.
    #[inline]
    pub fn char_at(&self, index: usize) -> Option<u16> {
        self.inner.char_at(index)
    }

    /// Returns a UTF-8 `String` representation.
    pub fn to_utf8(&self) -> String {
        self.inner.to_utf8()
    }

    /// Returns a reference to the underlying [`JsString`].
    #[inline]
    pub fn as_js_string(&self) -> &JsString {
        &self.inner
    }

    /// Returns `true` if two `Arc<InternalizedString>` values point to the
    /// same interned instance (pointer equality).
    #[inline]
    pub fn ptr_eq(a: &Arc<Self>, b: &Arc<Self>) -> bool {
        Arc::ptr_eq(a, b)
    }
}

impl Trace for InternalizedString {
    /// Trace the underlying flat string.
    fn trace(&self, tracer: &mut Tracer) {
        self.inner.trace(tracer);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ThinString
// ──────────────────────────────────────────────────────────────────────────────

/// A forwarding wrapper that redirects to an [`InternalizedString`].
///
/// When a string is internalized the original value may be logically replaced
/// by a `ThinString` that points to the canonical interned copy.  All
/// character-level operations delegate to the target.
///
/// This corresponds to V8's `ThinString` type: a one-word indirection used
/// after internalization to avoid duplicating character data.
#[derive(Debug, Clone)]
pub struct ThinString {
    /// The canonical interned string this thin string forwards to.
    actual: Arc<InternalizedString>,
}

impl ThinString {
    /// Creates a `ThinString` forwarding to the given interned string.
    pub fn new(actual: Arc<InternalizedString>) -> Self {
        Self { actual }
    }

    /// Returns a reference to the target [`InternalizedString`].
    #[inline]
    pub fn actual(&self) -> &InternalizedString {
        &self.actual
    }

    /// Returns a shared reference-counted handle to the target.
    pub fn actual_arc(&self) -> Arc<InternalizedString> {
        Arc::clone(&self.actual)
    }

    /// Returns the number of UTF-16 code units.
    #[inline]
    pub fn length(&self) -> usize {
        self.actual.length()
    }

    /// Returns the UTF-16 code unit at `index`, or `None` if out of bounds.
    #[inline]
    pub fn char_at(&self, index: usize) -> Option<u16> {
        self.actual.char_at(index)
    }

    /// Returns a UTF-8 `String` representation.
    pub fn to_utf8(&self) -> String {
        self.actual.to_utf8()
    }

    /// Returns the cached FNV-1a hash.
    #[inline]
    pub fn hash(&self) -> u32 {
        self.actual.hash()
    }
}

impl Trace for ThinString {
    /// Trace the target interned string.
    fn trace(&self, tracer: &mut Tracer) {
        self.actual.trace(tracer);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// StringTable
// ──────────────────────────────────────────────────────────────────────────────

/// The string interning table for property-key deduplication.
///
/// `StringTable` maintains a set of [`InternalizedString`] values keyed by
/// their FNV-1a hash.  Inserting a string that already exists returns the
/// existing [`Arc<InternalizedString>`] rather than creating a new allocation,
/// guaranteeing identity-based equality for all interned strings.
///
/// # Examples
///
/// ```
/// use stator_core::objects::js_string::{InternalizedString, StringTable};
///
/// let mut table = StringTable::new();
/// let a = table.intern_str("hello");
/// let b = table.intern_str("hello");
/// assert!(InternalizedString::ptr_eq(&a, &b));
/// ```
pub struct StringTable {
    /// Hash → bucket of interned strings sharing that hash value.
    buckets: HashMap<u32, Vec<Arc<InternalizedString>>>,
    /// Total number of interned strings in the table.
    count: usize,
}

impl StringTable {
    /// Creates an empty string table.
    pub fn new() -> Self {
        Self {
            buckets: HashMap::new(),
            count: 0,
        }
    }

    /// Creates an empty string table with pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buckets: HashMap::with_capacity(cap),
            count: 0,
        }
    }

    /// Returns the number of interned strings in the table.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if the table contains no interned strings.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Interns a [`JsString`], returning the canonical [`Arc<InternalizedString>`].
    ///
    /// If an equal string is already present the existing entry is returned
    /// (O(1) amortised for typical workloads where hash collisions are rare).
    /// Otherwise a new [`InternalizedString`] is created, inserted, and
    /// returned.
    pub fn intern(&mut self, s: JsString) -> Arc<InternalizedString> {
        let flat = s.flatten();
        let hash = flat.hash();

        if let Some(bucket) = self.buckets.get(&hash) {
            for existing in bucket {
                if js_strings_equal(&flat, existing.as_js_string()) {
                    return Arc::clone(existing);
                }
            }
        }

        let interned = Arc::new(InternalizedString { inner: flat, hash });
        self.buckets
            .entry(hash)
            .or_default()
            .push(Arc::clone(&interned));
        self.count += 1;
        interned
    }

    /// Convenience method to intern a UTF-8 string slice.
    pub fn intern_str(&mut self, s: &str) -> Arc<InternalizedString> {
        self.intern(JsString::new(s))
    }

    /// Looks up a string in the table without inserting it.
    ///
    /// Returns `Some(arc)` if an identical string is already interned,
    /// `None` otherwise.
    pub fn lookup(&self, s: &JsString) -> Option<Arc<InternalizedString>> {
        let flat = s.flatten();
        let hash = flat.hash();
        if let Some(bucket) = self.buckets.get(&hash) {
            for existing in bucket {
                if js_strings_equal(&flat, existing.as_js_string()) {
                    return Some(Arc::clone(existing));
                }
            }
        }
        None
    }

    /// Convenience method to look up a UTF-8 string slice.
    pub fn lookup_str(&self, s: &str) -> Option<Arc<InternalizedString>> {
        self.lookup(&JsString::new(s))
    }
}

impl Default for StringTable {
    fn default() -> Self {
        Self::new()
    }
}

impl Trace for StringTable {
    /// Trace every interned string in the table.
    fn trace(&self, tracer: &mut Tracer) {
        for bucket in self.buckets.values() {
            for interned in bucket {
                interned.trace(tracer);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper
// ──────────────────────────────────────────────────────────────────────────────

/// Compares two [`JsString`] values for code-unit-level equality.
fn js_strings_equal(a: &JsString, b: &JsString) -> bool {
    let len = a.length();
    if len != b.length() {
        return false;
    }
    for i in 0..len {
        if a.char_at(i) != b.char_at(i) {
            return false;
        }
    }
    true
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── InternalizedString ───────────────────────────────────────────────────

    #[test]
    fn test_internalized_length_and_char_at() {
        let mut table = StringTable::new();
        let s = table.intern_str("abc");
        assert_eq!(s.length(), 3);
        assert_eq!(s.char_at(0), Some(b'a' as u16));
        assert_eq!(s.char_at(2), Some(b'c' as u16));
        assert_eq!(s.char_at(3), None);
    }

    #[test]
    fn test_internalized_to_utf8() {
        let mut table = StringTable::new();
        let s = table.intern_str("hello");
        assert_eq!(s.to_utf8(), "hello");
    }

    #[test]
    fn test_internalized_hash_matches_js_string() {
        let mut table = StringTable::new();
        let interned = table.intern_str("test");
        let js = JsString::new("test");
        assert_eq!(interned.hash(), js.hash());
    }

    #[test]
    fn test_internalized_ptr_eq_same_content() {
        let mut table = StringTable::new();
        let a = table.intern_str("key");
        let b = table.intern_str("key");
        assert!(InternalizedString::ptr_eq(&a, &b));
    }

    #[test]
    fn test_internalized_ptr_ne_different_content() {
        let mut table = StringTable::new();
        let a = table.intern_str("foo");
        let b = table.intern_str("bar");
        assert!(!InternalizedString::ptr_eq(&a, &b));
    }

    // ── ThinString ───────────────────────────────────────────────────────────

    #[test]
    fn test_thin_string_delegates_length() {
        let mut table = StringTable::new();
        let interned = table.intern_str("hello");
        let thin = ThinString::new(interned);
        assert_eq!(thin.length(), 5);
    }

    #[test]
    fn test_thin_string_delegates_char_at() {
        let mut table = StringTable::new();
        let interned = table.intern_str("abc");
        let thin = ThinString::new(interned);
        assert_eq!(thin.char_at(0), Some(b'a' as u16));
        assert_eq!(thin.char_at(2), Some(b'c' as u16));
        assert_eq!(thin.char_at(3), None);
    }

    #[test]
    fn test_thin_string_delegates_to_utf8() {
        let mut table = StringTable::new();
        let interned = table.intern_str("world");
        let thin = ThinString::new(interned);
        assert_eq!(thin.to_utf8(), "world");
    }

    #[test]
    fn test_thin_string_delegates_hash() {
        let mut table = StringTable::new();
        let interned = table.intern_str("test");
        let thin = ThinString::new(Arc::clone(&interned));
        assert_eq!(thin.hash(), interned.hash());
    }

    #[test]
    fn test_thin_string_actual_returns_same_arc() {
        let mut table = StringTable::new();
        let interned = table.intern_str("key");
        let thin = ThinString::new(Arc::clone(&interned));
        assert!(InternalizedString::ptr_eq(&interned, &thin.actual_arc()));
    }

    // ── StringTable ──────────────────────────────────────────────────────────

    #[test]
    fn test_string_table_new_is_empty() {
        let table = StringTable::new();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn test_string_table_default_is_empty() {
        let table = StringTable::default();
        assert!(table.is_empty());
    }

    #[test]
    fn test_string_table_intern_increments_count() {
        let mut table = StringTable::new();
        table.intern_str("a");
        assert_eq!(table.len(), 1);
        table.intern_str("b");
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn test_string_table_intern_deduplicates() {
        let mut table = StringTable::new();
        table.intern_str("dup");
        table.intern_str("dup");
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn test_string_table_intern_js_string_variants() {
        let mut table = StringTable::new();
        // Intern a ConsString — it should be flattened and deduplicated.
        let cons = JsString::new("foo").concat(JsString::new("bar"));
        let from_cons = table.intern(cons);
        let from_flat = table.intern_str("foobar");
        assert!(InternalizedString::ptr_eq(&from_cons, &from_flat));
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn test_string_table_lookup_existing() {
        let mut table = StringTable::new();
        let interned = table.intern_str("hello");
        let found = table.lookup_str("hello");
        assert!(found.is_some());
        assert!(InternalizedString::ptr_eq(&interned, &found.unwrap()));
    }

    #[test]
    fn test_string_table_lookup_missing() {
        let table = StringTable::new();
        assert!(table.lookup_str("missing").is_none());
    }

    #[test]
    fn test_string_table_with_capacity() {
        let table = StringTable::with_capacity(64);
        assert!(table.is_empty());
    }

    #[test]
    fn test_string_table_intern_non_latin1() {
        let mut table = StringTable::new();
        let a = table.intern_str("日本語");
        let b = table.intern_str("日本語");
        assert!(InternalizedString::ptr_eq(&a, &b));
        assert_eq!(a.to_utf8(), "日本語");
    }

    #[test]
    fn test_string_table_intern_empty_string() {
        let mut table = StringTable::new();
        let a = table.intern_str("");
        let b = table.intern_str("");
        assert!(InternalizedString::ptr_eq(&a, &b));
        assert_eq!(a.length(), 0);
    }

    #[test]
    fn test_string_table_intern_latin1_chars() {
        let mut table = StringTable::new();
        let a = table.intern_str("caf\u{00E9}");
        let b = table.intern_str("café");
        assert!(InternalizedString::ptr_eq(&a, &b));
    }

    // ── js_strings_equal helper ──────────────────────────────────────────────

    #[test]
    fn test_js_strings_equal_same_content() {
        let a = JsString::new("hello");
        let b = JsString::new("hello");
        assert!(js_strings_equal(&a, &b));
    }

    #[test]
    fn test_js_strings_equal_different_content() {
        let a = JsString::new("hello");
        let b = JsString::new("world");
        assert!(!js_strings_equal(&a, &b));
    }

    #[test]
    fn test_js_strings_equal_different_length() {
        let a = JsString::new("hi");
        let b = JsString::new("hello");
        assert!(!js_strings_equal(&a, &b));
    }
}
