//! A property map that stores ECMAScript property values alongside their
//! attribute flags (`[[Writable]]`, `[[Enumerable]]`, `[[Configurable]]`).
//!
//! [`PropertyMap`] is the backing store for [`JsValue::PlainObject`] and
//! provides a HashMap-like API that transparently attaches
//! [`PropertyAttributes`] to every entry.  Newly inserted properties default
//! to `WRITABLE | ENUMERABLE | CONFIGURABLE` (the ECMAScript default for
//! ordinary user-created data properties).
//!
//! Internally, property values are stored in a flat `Vec<JsValue>` for
//! cache-friendly iteration and O(1) slot-based access.  A secondary
//! `HashMap<String, usize>` maps property names to slot indices for
//! efficient lookup.
//!
//! An [`INLINE_CACHE_CAP`]-entry inline cache of recently accessed property
//! names sits in front of the `HashMap` to avoid hash-table probing on
//! repeated lookups of the same hot property names.

use std::cell::Cell;
use std::collections::HashMap;

use crate::objects::map::PropertyAttributes;
use crate::objects::value::JsValue;

/// Default attributes for properties created by ordinary JS assignment:
/// writable, enumerable, and configurable.
const DEFAULT_ATTRS: PropertyAttributes = PropertyAttributes::from_bits_truncate(
    PropertyAttributes::WRITABLE.bits()
        | PropertyAttributes::ENUMERABLE.bits()
        | PropertyAttributes::CONFIGURABLE.bits(),
);

/// Maximum number of entries in the inline property-name cache.
///
/// Four entries fit comfortably in a single cache line and cover the vast
/// majority of tight property-access loops in typical JavaScript code.
const INLINE_CACHE_CAP: usize = 4;

/// Cheap 64-bit FNV-1a hash for inline-cache probing.
///
/// This is intentionally *not* `SipHash`: it trades collision resistance for
/// raw speed on the short property names typical of JavaScript.
#[inline]
fn name_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// A map of named properties with ECMAScript attribute flags.
///
/// Property values are stored in a flat `Vec` for cache-friendly iteration
/// and O(1) slot-based access.  The `index` HashMap provides O(1) name
/// lookups into the flat storage.
///
/// An [`INLINE_CACHE_CAP`]-entry inline cache of recently accessed property
/// name hashes sits in front of the `HashMap` to avoid hash-table probing
/// on repeated lookups of the same hot property names.  The cache uses
/// [`Cell`]-based interior mutability so that read-only (`&self`) lookups
/// can still populate the cache.
#[derive(Debug, Clone)]
pub struct PropertyMap {
    /// Property names in insertion order.
    keys: Vec<String>,
    /// Property values, one per key.
    values: Vec<JsValue>,
    /// Property attributes, one per key.
    attrs: Vec<PropertyAttributes>,
    /// Name → slot-index mapping for O(1) lookup.
    index: HashMap<String, usize>,
    /// Inline cache: FNV-1a hashes of recently accessed property names.
    cache_hashes: [Cell<u64>; INLINE_CACHE_CAP],
    /// Inline cache: corresponding slot indices for `cache_hashes`.
    cache_slots: [Cell<u32>; INLINE_CACHE_CAP],
    /// Number of valid entries in the inline cache (0..=INLINE_CACHE_CAP).
    cache_len: Cell<u8>,
    /// Circular replacement cursor into the cache arrays.
    cache_cursor: Cell<u8>,
}

impl PartialEq for PropertyMap {
    fn eq(&self, other: &Self) -> bool {
        self.keys == other.keys
            && self.values == other.values
            && self.attrs == other.attrs
            && self.index == other.index
    }
}

impl PropertyMap {
    /// Creates an empty property map.
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            attrs: Vec::new(),
            index: HashMap::new(),
            cache_hashes: Default::default(),
            cache_slots: Default::default(),
            cache_len: Cell::new(0),
            cache_cursor: Cell::new(0),
        }
    }

    /// Creates a property map pre-allocated for `capacity` entries.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            keys: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
            attrs: Vec::with_capacity(capacity),
            index: HashMap::with_capacity(capacity),
            cache_hashes: Default::default(),
            cache_slots: Default::default(),
            cache_len: Cell::new(0),
            cache_cursor: Cell::new(0),
        }
    }

    // ── Inline cache helpers ──────────────────────────────────────────────

    /// Probes the inline cache for `key`, returning its slot index on hit.
    #[inline]
    fn cache_probe(&self, key: &str) -> Option<usize> {
        let h = name_hash(key);
        let len = self.cache_len.get() as usize;
        for i in 0..len {
            if self.cache_hashes[i].get() == h {
                let slot = self.cache_slots[i].get() as usize;
                // Verify against the canonical key to guard against hash
                // collisions (extremely rare but must be handled).
                if slot < self.keys.len() && self.keys[slot] == key {
                    return Some(slot);
                }
            }
        }
        None
    }

    /// Records a `(key, slot)` pair in the inline cache.
    #[inline]
    fn cache_record(&self, key: &str, slot: usize) {
        let h = name_hash(key);
        let len = self.cache_len.get() as usize;
        if len < INLINE_CACHE_CAP {
            self.cache_hashes[len].set(h);
            self.cache_slots[len].set(slot as u32);
            self.cache_len.set((len + 1) as u8);
        } else {
            let cursor = self.cache_cursor.get() as usize;
            self.cache_hashes[cursor].set(h);
            self.cache_slots[cursor].set(slot as u32);
            self.cache_cursor
                .set(((cursor + 1) % INLINE_CACHE_CAP) as u8);
        }
    }

    /// Invalidates all inline cache entries.
    #[inline]
    fn cache_invalidate(&self) {
        self.cache_len.set(0);
        self.cache_cursor.set(0);
    }

    // ── HashMap-compatible API ────────────────────────────────────────────

    /// Returns the value for `key`, ignoring attributes.
    pub fn get(&self, key: &str) -> Option<&JsValue> {
        if let Some(slot) = self.cache_probe(key) {
            return Some(&self.values[slot]);
        }
        self.index.get(key).map(|&i| {
            self.cache_record(key, i);
            &self.values[i]
        })
    }

    /// Returns a clone of the value for `key`, ignoring attributes.
    pub fn get_cloned(&self, key: &str) -> Option<JsValue> {
        if let Some(slot) = self.cache_probe(key) {
            return Some(self.values[slot].clone());
        }
        self.index.get(key).map(|&i| {
            self.cache_record(key, i);
            self.values[i].clone()
        })
    }

    /// Returns `true` if the map contains an entry for `key`.
    pub fn contains_key(&self, key: &str) -> bool {
        if self.cache_probe(key).is_some() {
            return true;
        }
        self.index.contains_key(key)
    }

    /// Inserts a property with default attributes (writable, enumerable,
    /// configurable).  If the key already exists the value is replaced but
    /// the existing attributes are preserved.
    pub fn insert(&mut self, key: String, value: JsValue) {
        if let Some(&i) = self.index.get(&key) {
            self.values[i] = value;
        } else {
            let i = self.keys.len();
            self.index.insert(key.clone(), i);
            self.keys.push(key);
            self.values.push(value);
            self.attrs.push(DEFAULT_ATTRS);
        }
    }

    /// Removes the entry for `key`, returning the old value (if any).
    pub fn remove(&mut self, key: &str) -> Option<JsValue> {
        if let Some(i) = self.index.remove(key) {
            let val = self.values[i].clone();
            // Swap-remove to keep storage compact.
            let last = self.keys.len() - 1;
            if i != last {
                self.keys.swap(i, last);
                self.values.swap(i, last);
                self.attrs.swap(i, last);
                // Update the index for the swapped-in key.
                self.index.insert(self.keys[i].clone(), i);
            }
            self.keys.pop();
            self.values.pop();
            self.attrs.pop();
            // Invalidate cache: swap-remove may have changed slot indices.
            self.cache_invalidate();
            Some(val)
        } else {
            None
        }
    }

    /// Returns an iterator over the property keys.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.keys.iter()
    }

    /// Returns an iterator over `(key, value)` pairs, ignoring attributes.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &JsValue)> {
        self.keys.iter().zip(self.values.iter())
    }

    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Returns `true` if the map contains no entries.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    // ── Attribute-aware API ───────────────────────────────────────────────

    /// Returns the value and attribute flags for `key`.
    pub fn get_with_attrs(&self, key: &str) -> Option<(&JsValue, PropertyAttributes)> {
        if let Some(slot) = self.cache_probe(key) {
            return Some((&self.values[slot], self.attrs[slot]));
        }
        self.index.get(key).map(|&i| {
            self.cache_record(key, i);
            (&self.values[i], self.attrs[i])
        })
    }

    /// Inserts a property with explicit attribute flags.
    pub fn insert_with_attrs(&mut self, key: String, value: JsValue, attrs: PropertyAttributes) {
        if let Some(&i) = self.index.get(&key) {
            self.values[i] = value;
            self.attrs[i] = attrs;
        } else {
            let i = self.keys.len();
            self.index.insert(key.clone(), i);
            self.keys.push(key);
            self.values.push(value);
            self.attrs.push(attrs);
        }
    }

    /// Updates the attribute flags for an existing property.
    /// Returns `true` if the property existed and was updated.
    pub fn set_attrs(&mut self, key: &str, attrs: PropertyAttributes) -> bool {
        if let Some(&i) = self.index.get(key) {
            self.attrs[i] = attrs;
            true
        } else {
            false
        }
    }

    /// Returns the attribute flags for `key`, or `None` if absent.
    pub fn attrs(&self, key: &str) -> Option<PropertyAttributes> {
        if let Some(slot) = self.cache_probe(key) {
            return Some(self.attrs[slot]);
        }
        self.index.get(key).map(|&i| self.attrs[i])
    }

    /// Returns `true` if the property exists and is writable.
    pub fn is_writable(&self, key: &str) -> bool {
        self.index
            .get(key)
            .map(|&i| self.attrs[i].contains(PropertyAttributes::WRITABLE))
            .unwrap_or(false)
    }

    /// Returns `true` if the property exists and is configurable.
    pub fn is_configurable(&self, key: &str) -> bool {
        self.index
            .get(key)
            .map(|&i| self.attrs[i].contains(PropertyAttributes::CONFIGURABLE))
            .unwrap_or(false)
    }

    /// Returns `true` if the property exists and is enumerable.
    pub fn is_enumerable(&self, key: &str) -> bool {
        self.index
            .get(key)
            .map(|&i| self.attrs[i].contains(PropertyAttributes::ENUMERABLE))
            .unwrap_or(false)
    }

    /// Set or clear the `WRITABLE` flag for an existing property.
    pub fn set_writable(&mut self, key: &str, writable: bool) {
        if let Some(&i) = self.index.get(key) {
            if writable {
                self.attrs[i].insert(PropertyAttributes::WRITABLE);
            } else {
                self.attrs[i].remove(PropertyAttributes::WRITABLE);
            }
        }
    }

    /// Set or clear the `ENUMERABLE` flag for an existing property.
    pub fn set_enumerable(&mut self, key: &str, enumerable: bool) {
        if let Some(&i) = self.index.get(key) {
            if enumerable {
                self.attrs[i].insert(PropertyAttributes::ENUMERABLE);
            } else {
                self.attrs[i].remove(PropertyAttributes::ENUMERABLE);
            }
        }
    }

    /// Set or clear the `CONFIGURABLE` flag for an existing property.
    pub fn set_configurable(&mut self, key: &str, configurable: bool) {
        if let Some(&i) = self.index.get(key) {
            if configurable {
                self.attrs[i].insert(PropertyAttributes::CONFIGURABLE);
            } else {
                self.attrs[i].remove(PropertyAttributes::CONFIGURABLE);
            }
        }
    }

    /// Returns an iterator over only the enumerable property keys.
    pub fn enumerable_keys(&self) -> impl Iterator<Item = &String> {
        self.keys
            .iter()
            .zip(self.attrs.iter())
            .filter(|(_, a)| a.contains(PropertyAttributes::ENUMERABLE))
            .map(|(k, _)| k)
    }

    /// Returns an iterator over `(key, value, attrs)` triples.
    pub fn iter_with_attrs(&self) -> impl Iterator<Item = (&String, &JsValue, PropertyAttributes)> {
        self.keys
            .iter()
            .zip(self.values.iter())
            .zip(self.attrs.iter())
            .map(|((k, v), a)| (k, v, *a))
    }
}

impl Default for PropertyMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_get_default_attrs() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(42));
        assert_eq!(pm.get("x"), Some(&JsValue::Smi(42)));
        let attrs = pm.attrs("x").unwrap();
        assert!(attrs.contains(PropertyAttributes::WRITABLE));
        assert!(attrs.contains(PropertyAttributes::ENUMERABLE));
        assert!(attrs.contains(PropertyAttributes::CONFIGURABLE));
    }

    #[test]
    fn test_insert_with_attrs() {
        let mut pm = PropertyMap::new();
        pm.insert_with_attrs(
            "ro".to_string(),
            JsValue::Smi(1),
            PropertyAttributes::empty(),
        );
        assert!(!pm.is_writable("ro"));
        assert!(!pm.is_enumerable("ro"));
        assert!(!pm.is_configurable("ro"));
    }

    #[test]
    fn test_insert_preserves_existing_attrs() {
        let mut pm = PropertyMap::new();
        pm.insert_with_attrs(
            "p".to_string(),
            JsValue::Smi(1),
            PropertyAttributes::ENUMERABLE,
        );
        // Re-insert with default insert — should preserve ENUMERABLE-only.
        pm.insert("p".to_string(), JsValue::Smi(2));
        assert_eq!(pm.get("p"), Some(&JsValue::Smi(2)));
        let attrs = pm.attrs("p").unwrap();
        assert_eq!(attrs, PropertyAttributes::ENUMERABLE);
    }

    #[test]
    fn test_remove() {
        let mut pm = PropertyMap::new();
        pm.insert("a".to_string(), JsValue::Boolean(true));
        assert!(pm.contains_key("a"));
        let removed = pm.remove("a");
        assert_eq!(removed, Some(JsValue::Boolean(true)));
        assert!(!pm.contains_key("a"));
    }

    #[test]
    fn test_enumerable_keys() {
        let mut pm = PropertyMap::new();
        pm.insert("visible".to_string(), JsValue::Smi(1));
        pm.insert_with_attrs(
            "hidden".to_string(),
            JsValue::Smi(2),
            PropertyAttributes::WRITABLE,
        );
        let enum_keys: Vec<&String> = pm.enumerable_keys().collect();
        assert!(enum_keys.contains(&&"visible".to_string()));
        assert!(!enum_keys.contains(&&"hidden".to_string()));
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut pm = PropertyMap::new();
        assert!(pm.is_empty());
        assert_eq!(pm.len(), 0);
        pm.insert("k".to_string(), JsValue::Null);
        assert!(!pm.is_empty());
        assert_eq!(pm.len(), 1);
    }

    #[test]
    fn test_set_attrs() {
        let mut pm = PropertyMap::new();
        pm.insert("p".to_string(), JsValue::Smi(1));
        assert!(pm.is_writable("p"));
        pm.set_attrs("p", PropertyAttributes::ENUMERABLE);
        assert!(!pm.is_writable("p"));
        assert!(pm.is_enumerable("p"));
        assert!(!pm.is_configurable("p"));
    }

    #[test]
    fn test_get_with_attrs() {
        let mut pm = PropertyMap::new();
        pm.insert_with_attrs(
            "x".to_string(),
            JsValue::Smi(5),
            PropertyAttributes::WRITABLE | PropertyAttributes::ENUMERABLE,
        );
        let (val, attrs) = pm.get_with_attrs("x").unwrap();
        assert_eq!(val, &JsValue::Smi(5));
        assert!(attrs.contains(PropertyAttributes::WRITABLE));
        assert!(attrs.contains(PropertyAttributes::ENUMERABLE));
        assert!(!attrs.contains(PropertyAttributes::CONFIGURABLE));
    }

    #[test]
    fn test_iter_with_attrs() {
        let mut pm = PropertyMap::new();
        pm.insert_with_attrs(
            "a".to_string(),
            JsValue::Smi(1),
            PropertyAttributes::WRITABLE,
        );
        let entries: Vec<_> = pm.iter_with_attrs().collect();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "a");
        assert_eq!(entries[0].1, &JsValue::Smi(1));
        assert_eq!(entries[0].2, PropertyAttributes::WRITABLE);
    }

    #[test]
    fn test_missing_key_attr_queries() {
        let pm = PropertyMap::new();
        assert!(!pm.is_writable("nope"));
        assert!(!pm.is_enumerable("nope"));
        assert!(!pm.is_configurable("nope"));
        assert!(pm.attrs("nope").is_none());
    }

    #[test]
    fn test_with_capacity() {
        let pm = PropertyMap::with_capacity(16);
        assert!(pm.is_empty());
    }

    // ── Inline cache tests ───────────────────────────────────────────────

    #[test]
    fn test_cache_populated_on_get() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(1));
        pm.insert("y".to_string(), JsValue::Smi(2));

        // First access populates the cache.
        assert_eq!(pm.get("x"), Some(&JsValue::Smi(1)));
        assert_eq!(pm.cache_len.get(), 1);

        // Second access of same key hits the cache.
        assert_eq!(pm.get("x"), Some(&JsValue::Smi(1)));
        assert_eq!(pm.cache_len.get(), 1); // no new entry

        // Different key adds another cache entry.
        assert_eq!(pm.get("y"), Some(&JsValue::Smi(2)));
        assert_eq!(pm.cache_len.get(), 2);
    }

    #[test]
    fn test_cache_invalidated_on_remove() {
        let mut pm = PropertyMap::new();
        pm.insert("a".to_string(), JsValue::Smi(1));
        pm.insert("b".to_string(), JsValue::Smi(2));

        // Populate cache.
        assert_eq!(pm.get("a"), Some(&JsValue::Smi(1)));
        assert_eq!(pm.cache_len.get(), 1);

        // Remove invalidates cache.
        pm.remove("a");
        assert_eq!(pm.cache_len.get(), 0);
    }

    #[test]
    fn test_cache_wraps_around() {
        let mut pm = PropertyMap::new();
        for i in 0..6 {
            pm.insert(format!("k{i}"), JsValue::Smi(i));
        }
        // Access 6 keys: first 4 fill the cache, next 2 replace via cursor.
        for i in 0..6 {
            assert_eq!(pm.get(&format!("k{i}")), Some(&JsValue::Smi(i)));
        }
        assert_eq!(pm.cache_len.get(), INLINE_CACHE_CAP as u8);
        // All lookups should still work (cache or HashMap fallback).
        for i in 0..6 {
            assert_eq!(pm.get(&format!("k{i}")), Some(&JsValue::Smi(i)));
        }
    }

    #[test]
    fn test_cache_contains_key_fast_path() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(1));
        // Populate cache via get.
        let _ = pm.get("x");
        // contains_key should hit the cache.
        assert!(pm.contains_key("x"));
        assert!(!pm.contains_key("missing"));
    }

    #[test]
    fn test_cache_get_with_attrs_fast_path() {
        let mut pm = PropertyMap::new();
        pm.insert_with_attrs(
            "p".to_string(),
            JsValue::Smi(42),
            PropertyAttributes::WRITABLE | PropertyAttributes::ENUMERABLE,
        );
        // First call: populates cache.
        let (val, attrs) = pm.get_with_attrs("p").unwrap();
        assert_eq!(val, &JsValue::Smi(42));
        assert!(attrs.contains(PropertyAttributes::WRITABLE));
        // Second call: should hit cache.
        let (val2, attrs2) = pm.get_with_attrs("p").unwrap();
        assert_eq!(val2, &JsValue::Smi(42));
        assert_eq!(attrs, attrs2);
    }

    #[test]
    fn test_cache_equality_ignores_cache_state() {
        let mut pm1 = PropertyMap::new();
        let mut pm2 = PropertyMap::new();
        pm1.insert("x".to_string(), JsValue::Smi(1));
        pm2.insert("x".to_string(), JsValue::Smi(1));
        // pm1 has a populated cache, pm2 does not.
        let _ = pm1.get("x");
        assert_eq!(pm1.cache_len.get(), 1);
        assert_eq!(pm2.cache_len.get(), 0);
        // They should still be equal.
        assert_eq!(pm1, pm2);
    }
}
