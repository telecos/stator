//! Shape-indexed flat property storage for lightweight plain objects.
//!
//! [`PlainObjectStorage`] replaces `HashMap<String, JsValue>` as the backing
//! store for [`JsValue::PlainObject`][crate::objects::value::JsValue::PlainObject].
//! Properties are stored in a flat slot array whose indices are determined by
//! a shared [`ShapeTable`], yielding O(1) property access via the inline-cache
//! system with much lower constant overhead than a hash map.
//!
//! # In-object properties
//!
//! The first [`INLINE_SLOTS`] properties are stored directly inside the struct
//! to avoid a `Vec` heap allocation for small objects (the common case).
//! Properties beyond that limit overflow into a heap-allocated `Vec<JsValue>`.
//!
//! # Thread-local shape table
//!
//! All `PlainObjectStorage` instances on a thread share a single
//! [`ShapeTable`] via [`SHAPE_TABLE`].  Shape transitions are automatically
//! deduplicated so objects created with the same property sequence converge on
//! the same shape.

use std::cell::RefCell;

use crate::objects::shapes::{ShapeId, ShapeTable};
use crate::objects::value::JsValue;

/// Number of property slots stored inline in the struct before overflowing
/// to a heap-allocated `Vec`.
pub const INLINE_SLOTS: usize = 4;

thread_local! {
    /// Per-thread shape table shared by all [`PlainObjectStorage`] instances.
    static SHAPE_TABLE: RefCell<ShapeTable> = RefCell::new(ShapeTable::new());
}

/// Shape-indexed flat property storage for plain JavaScript objects.
///
/// Properties are stored in contiguous slot arrays indexed by shape
/// descriptors, giving O(1) lookup via the inline-cache system.  The first
/// [`INLINE_SLOTS`] values live directly in the struct; additional values
/// spill to `overflow`.  Property names are also stored in insertion order
/// in `names` so that the `HashMap`-compatible API (`keys`, `values`, `iter`)
/// can return references without borrowing the global shape table.
#[derive(Clone)]
pub struct PlainObjectStorage {
    /// Current shape in the global shape table.
    shape_id: ShapeId,
    /// Property names in insertion order (index matches slot index).
    names: Vec<String>,
    /// Inline property slots (avoids heap allocation for small objects).
    inline: [JsValue; INLINE_SLOTS],
    /// Overflow property slots for objects with more than [`INLINE_SLOTS`]
    /// properties.
    overflow: Vec<JsValue>,
    /// Bit-set tracking deleted slot indices.  Lazily allocated on first
    /// deletion to keep the common (no-delete) path allocation-free.
    deleted: Option<Vec<u32>>,
}

impl PlainObjectStorage {
    /// Creates an empty `PlainObjectStorage` backed by the thread-local shape
    /// table.
    pub fn new() -> Self {
        let root = SHAPE_TABLE.with(|t| t.borrow().root());
        Self {
            shape_id: root,
            names: Vec::new(),
            inline: [
                JsValue::Undefined,
                JsValue::Undefined,
                JsValue::Undefined,
                JsValue::Undefined,
            ],
            overflow: Vec::new(),
            deleted: None,
        }
    }

    /// Returns the number of live (non-deleted) properties.
    #[inline]
    pub fn len(&self) -> usize {
        let total = self.names.len();
        let del = self.deleted.as_ref().map_or(0, |d| d.len());
        total - del
    }

    /// Returns `true` if the storage contains no properties.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a reference to the value associated with `key`, or `None`.
    pub fn get(&self, key: &str) -> Option<&JsValue> {
        let idx = self.name_index(key)?;
        if self.is_deleted(idx) {
            return None;
        }
        Some(self.slot_ref(idx))
    }

    /// Returns a mutable reference to the value associated with `key`, or
    /// `None`.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut JsValue> {
        let idx = self.name_index(key)?;
        if self.is_deleted(idx) {
            return None;
        }
        Some(self.slot_mut(idx))
    }

    /// Returns `true` if a property with the given `key` exists.
    pub fn contains_key(&self, key: &str) -> bool {
        match self.name_index(key) {
            Some(idx) => !self.is_deleted(idx),
            None => false,
        }
    }

    /// Inserts or updates the property `key` with `value`.
    ///
    /// If the property already exists, the value is updated in place.
    /// Otherwise a shape transition is performed and the value is appended
    /// to the slot array.
    pub fn insert(&mut self, key: String, value: JsValue) {
        // Check for existing property (update in place).
        if let Some(idx) = self.name_index(&key) {
            if self.is_deleted(idx) {
                self.mark_undeleted(idx);
            }
            *self.slot_mut(idx) = value;
            return;
        }

        // New property: transition shape and append slot.
        let attrs = crate::objects::map::PropertyAttributes::WRITABLE
            | crate::objects::map::PropertyAttributes::ENUMERABLE
            | crate::objects::map::PropertyAttributes::CONFIGURABLE;

        let new_shape = SHAPE_TABLE.with(|t| {
            let mut table = t.borrow_mut();
            table.transition(self.shape_id, &key, attrs)
        });
        self.shape_id = new_shape;

        let slot_idx = self.names.len();
        self.names.push(key);

        // Write value into the new slot.
        if slot_idx < INLINE_SLOTS {
            self.inline[slot_idx] = value;
        } else {
            let overflow_idx = slot_idx - INLINE_SLOTS;
            if overflow_idx >= self.overflow.len() {
                self.overflow.resize(overflow_idx + 1, JsValue::Undefined);
            }
            self.overflow[overflow_idx] = value;
        }
    }

    /// Removes the property `key` and returns its value, or `None` if not
    /// found.
    pub fn remove(&mut self, key: &str) -> Option<JsValue> {
        let idx = self.name_index(key)?;
        if self.is_deleted(idx) {
            return None;
        }
        let old = std::mem::replace(self.slot_mut(idx), JsValue::Undefined);
        self.mark_deleted(idx);
        Some(old)
    }

    /// Returns an iterator over live property keys in insertion order.
    pub fn keys(&self) -> Keys<'_> {
        Keys {
            names: &self.names,
            deleted: &self.deleted,
            pos: 0,
        }
    }

    /// Returns an iterator over live property values in insertion order.
    pub fn values(&self) -> Values<'_> {
        Values {
            storage: self,
            pos: 0,
        }
    }

    /// Returns an iterator over live `(&String, &JsValue)` pairs.
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            storage: self,
            pos: 0,
        }
    }

    /// Returns the [`ShapeId`] of this storage (for inline-cache integration).
    #[inline]
    pub fn shape_id(&self) -> ShapeId {
        self.shape_id
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    /// Returns the slot index for `key`, or `None` if not present.
    #[inline]
    fn name_index(&self, key: &str) -> Option<usize> {
        self.names.iter().position(|n| n == key)
    }

    /// Returns a reference to the slot at the given index.
    #[inline]
    fn slot_ref(&self, idx: usize) -> &JsValue {
        if idx < INLINE_SLOTS {
            &self.inline[idx]
        } else {
            &self.overflow[idx - INLINE_SLOTS]
        }
    }

    /// Returns a mutable reference to the slot at the given index.
    #[inline]
    fn slot_mut(&mut self, idx: usize) -> &mut JsValue {
        if idx < INLINE_SLOTS {
            &mut self.inline[idx]
        } else {
            &mut self.overflow[idx - INLINE_SLOTS]
        }
    }

    /// Returns `true` if the slot at `idx` has been deleted.
    #[inline]
    fn is_deleted(&self, idx: usize) -> bool {
        self.deleted
            .as_ref()
            .is_some_and(|d| d.contains(&(idx as u32)))
    }

    /// Marks a slot as deleted.
    fn mark_deleted(&mut self, idx: usize) {
        let del = self.deleted.get_or_insert_with(Vec::new);
        if !del.contains(&(idx as u32)) {
            del.push(idx as u32);
        }
    }

    /// Removes a slot from the deleted set.
    fn mark_undeleted(&mut self, idx: usize) {
        if let Some(ref mut del) = self.deleted {
            del.retain(|&d| d != idx as u32);
        }
    }
}

impl Default for PlainObjectStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl From<std::collections::HashMap<String, JsValue>> for PlainObjectStorage {
    fn from(map: std::collections::HashMap<String, JsValue>) -> Self {
        let mut storage = Self::new();
        // Sort by key to ensure deterministic shape transitions.
        let mut entries: Vec<_> = map.into_iter().collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        for (k, v) in entries {
            storage.insert(k, v);
        }
        storage
    }
}

impl std::fmt::Debug for PlainObjectStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Iterators
// ─────────────────────────────────────────────────────────────────────────────

/// Iterator over live property keys of a [`PlainObjectStorage`].
pub struct Keys<'a> {
    names: &'a [String],
    deleted: &'a Option<Vec<u32>>,
    pos: usize,
}

impl<'a> Iterator for Keys<'a> {
    type Item = &'a String;

    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.names.len() {
            let idx = self.pos;
            self.pos += 1;
            let is_del = self
                .deleted
                .as_ref()
                .is_some_and(|d| d.contains(&(idx as u32)));
            if !is_del {
                return Some(&self.names[idx]);
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.names.len() - self.pos;
        (0, Some(remaining))
    }
}

/// Iterator over live property values of a [`PlainObjectStorage`].
pub struct Values<'a> {
    storage: &'a PlainObjectStorage,
    pos: usize,
}

impl<'a> Iterator for Values<'a> {
    type Item = &'a JsValue;

    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.storage.names.len() {
            let idx = self.pos;
            self.pos += 1;
            if !self.storage.is_deleted(idx) {
                return Some(self.storage.slot_ref(idx));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.storage.names.len() - self.pos;
        (0, Some(remaining))
    }
}

/// Iterator over live `(&String, &JsValue)` pairs of a [`PlainObjectStorage`].
pub struct Iter<'a> {
    storage: &'a PlainObjectStorage,
    pos: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = (&'a String, &'a JsValue);

    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.storage.names.len() {
            let idx = self.pos;
            self.pos += 1;
            if !self.storage.is_deleted(idx) {
                return Some((&self.storage.names[idx], self.storage.slot_ref(idx)));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.storage.names.len() - self.pos;
        (0, Some(remaining))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_empty() {
        let s = PlainObjectStorage::new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn test_insert_and_get() {
        let mut s = PlainObjectStorage::new();
        s.insert("x".to_string(), JsValue::Smi(42));
        assert_eq!(s.get("x"), Some(&JsValue::Smi(42)));
        assert_eq!(s.len(), 1);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_insert_multiple_inline() {
        let mut s = PlainObjectStorage::new();
        for i in 0..INLINE_SLOTS {
            s.insert(format!("p{i}"), JsValue::Smi(i as i32));
        }
        assert_eq!(s.len(), INLINE_SLOTS);
        for i in 0..INLINE_SLOTS {
            assert_eq!(s.get(&format!("p{i}")), Some(&JsValue::Smi(i as i32)));
        }
    }

    #[test]
    fn test_overflow_slots() {
        let mut s = PlainObjectStorage::new();
        for i in 0..(INLINE_SLOTS + 3) {
            s.insert(format!("p{i}"), JsValue::Smi(i as i32));
        }
        assert_eq!(s.len(), INLINE_SLOTS + 3);
        for i in 0..(INLINE_SLOTS + 3) {
            assert_eq!(s.get(&format!("p{i}")), Some(&JsValue::Smi(i as i32)));
        }
    }

    #[test]
    fn test_update_existing() {
        let mut s = PlainObjectStorage::new();
        s.insert("x".to_string(), JsValue::Smi(1));
        s.insert("x".to_string(), JsValue::Smi(99));
        assert_eq!(s.get("x"), Some(&JsValue::Smi(99)));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn test_contains_key() {
        let mut s = PlainObjectStorage::new();
        assert!(!s.contains_key("x"));
        s.insert("x".to_string(), JsValue::Smi(1));
        assert!(s.contains_key("x"));
    }

    #[test]
    fn test_remove() {
        let mut s = PlainObjectStorage::new();
        s.insert("x".to_string(), JsValue::Smi(42));
        let removed = s.remove("x");
        assert_eq!(removed, Some(JsValue::Smi(42)));
        assert!(!s.contains_key("x"));
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut s = PlainObjectStorage::new();
        assert_eq!(s.remove("ghost"), None);
    }

    #[test]
    fn test_reinsert_after_remove() {
        let mut s = PlainObjectStorage::new();
        s.insert("x".to_string(), JsValue::Smi(1));
        s.remove("x");
        s.insert("x".to_string(), JsValue::Smi(2));
        assert_eq!(s.get("x"), Some(&JsValue::Smi(2)));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn test_keys() {
        let mut s = PlainObjectStorage::new();
        s.insert("a".to_string(), JsValue::Smi(1));
        s.insert("b".to_string(), JsValue::Smi(2));
        let keys: Vec<&String> = s.keys().collect();
        assert_eq!(keys, vec![&"a".to_string(), &"b".to_string()]);
    }

    #[test]
    fn test_keys_after_remove() {
        let mut s = PlainObjectStorage::new();
        s.insert("a".to_string(), JsValue::Smi(1));
        s.insert("b".to_string(), JsValue::Smi(2));
        s.remove("a");
        let keys: Vec<&String> = s.keys().collect();
        assert_eq!(keys, vec![&"b".to_string()]);
    }

    #[test]
    fn test_values() {
        let mut s = PlainObjectStorage::new();
        s.insert("x".to_string(), JsValue::Smi(10));
        s.insert("y".to_string(), JsValue::Smi(20));
        let vals: Vec<&JsValue> = s.values().collect();
        assert_eq!(vals, vec![&JsValue::Smi(10), &JsValue::Smi(20)]);
    }

    #[test]
    fn test_iter() {
        let mut s = PlainObjectStorage::new();
        s.insert("a".to_string(), JsValue::Smi(1));
        s.insert("b".to_string(), JsValue::Smi(2));
        let items: Vec<_> = s.iter().collect();
        assert_eq!(
            items,
            vec![
                (&"a".to_string(), &JsValue::Smi(1)),
                (&"b".to_string(), &JsValue::Smi(2)),
            ]
        );
    }

    #[test]
    fn test_get_mut() {
        let mut s = PlainObjectStorage::new();
        s.insert("x".to_string(), JsValue::Smi(1));
        if let Some(v) = s.get_mut("x") {
            *v = JsValue::Smi(42);
        }
        assert_eq!(s.get("x"), Some(&JsValue::Smi(42)));
    }

    #[test]
    fn test_default() {
        let s = PlainObjectStorage::default();
        assert!(s.is_empty());
    }

    #[test]
    fn test_debug_format() {
        let mut s = PlainObjectStorage::new();
        s.insert("x".to_string(), JsValue::Smi(1));
        let debug = format!("{s:?}");
        assert!(debug.contains("x"));
    }

    #[test]
    fn test_from_hashmap() {
        let mut map = std::collections::HashMap::new();
        map.insert("x".to_string(), JsValue::Smi(1));
        map.insert("y".to_string(), JsValue::Smi(2));
        let s = PlainObjectStorage::from(map);
        assert_eq!(s.len(), 2);
        assert_eq!(s.get("x"), Some(&JsValue::Smi(1)));
        assert_eq!(s.get("y"), Some(&JsValue::Smi(2)));
    }

    #[test]
    fn test_shape_id_available() {
        let s = PlainObjectStorage::new();
        assert_eq!(s.shape_id().index(), 0);
    }
}
