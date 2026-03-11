//! A property map that stores ECMAScript property values alongside their
//! attribute flags (`[[Writable]]`, `[[Enumerable]]`, `[[Configurable]]`).
//!
//! [`PropertyMap`] is the backing store for [`JsValue::PlainObject`] and
//! provides a HashMap-like API that transparently attaches
//! [`PropertyAttributes`] to every entry.  Newly inserted properties default
//! to `WRITABLE | ENUMERABLE | CONFIGURABLE` (the ECMAScript default for
//! ordinary user-created data properties).
//!
//! ## ECMAScript enumeration order (§10.1.11)
//!
//! Properties are stored in **ECMAScript enumeration order**: integer-indexed
//! keys (array indices `0 ..= 2^32 − 2`) occupy the front of the storage in
//! ascending numeric order, followed by non-index string keys in the order
//! they were first inserted.  All iteration methods (`keys`, `iter`,
//! `enumerable_keys`, `iter_with_attrs`) naturally produce this order.
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
use std::sync::atomic::{AtomicU64, Ordering};

use crate::objects::map::PropertyAttributes;
use crate::objects::value::JsValue;

/// Global monotonically-increasing counter used to assign unique shape
/// identifiers to each [`PropertyMap`] structural configuration.
static NEXT_SHAPE_ID: AtomicU64 = AtomicU64::new(1);

/// Default attributes for properties created by ordinary JS assignment:
/// writable, enumerable, and configurable.
const DEFAULT_ATTRS: PropertyAttributes = PropertyAttributes::from_bits_truncate(
    PropertyAttributes::WRITABLE.bits()
        | PropertyAttributes::ENUMERABLE.bits()
        | PropertyAttributes::CONFIGURABLE.bits(),
);

/// Attributes for built-in methods per ES spec §10.4.7: writable,
/// non-enumerable, configurable.
const BUILTIN_ATTRS: PropertyAttributes = PropertyAttributes::from_bits_truncate(
    PropertyAttributes::WRITABLE.bits() | PropertyAttributes::CONFIGURABLE.bits(),
);

/// Maximum number of entries in the inline property-name cache.
///
/// Four entries fit comfortably in a single cache line and cover the vast
/// majority of tight property-access loops in typical JavaScript code.
const INLINE_CACHE_CAP: usize = 4;

/// Returns `Some(n)` if `key` is a valid ECMAScript array index — a canonical
/// decimal string representing an integer in `0 ..= 2^32 − 2`.
///
/// Per ECMA-262 §6.1.7, an integer index is a String value that is a canonical
/// numeric string (no leading zeros except `"0"` itself) whose numeric value
/// *i* satisfies `0 ≤ i < 2^32 − 1`.
#[inline]
fn parse_integer_index(key: &str) -> Option<u32> {
    if key.is_empty() || (key.len() > 1 && key.as_bytes()[0] == b'0') {
        return None;
    }
    let n: u32 = key.parse().ok()?;
    // Array indices are 0 ..= 2^32 − 2 (u32::MAX is *not* an index).
    if n < u32::MAX { Some(n) } else { None }
}

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
/// Property values are stored in a flat `Vec` in ECMAScript enumeration
/// order (integer indices ascending, then string keys in insertion order)
/// for cache-friendly, spec-compliant iteration.  The `index` HashMap
/// provides O(1) name lookups into the flat storage.
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
    /// Shape identifier — a monotonically-increasing stamp that changes on
    /// every structural mutation (property add/remove or attribute change).
    shape_id: u64,
    /// Whether new properties may be added to this object (§10.1 `[[Extensible]]`).
    pub extensible: bool,
}

impl PartialEq for PropertyMap {
    fn eq(&self, other: &Self) -> bool {
        self.keys == other.keys
            && self.values == other.values
            && self.attrs == other.attrs
            && self.index == other.index
            && self.extensible == other.extensible
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
            shape_id: NEXT_SHAPE_ID.fetch_add(1, Ordering::Relaxed),
            extensible: true,
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
            shape_id: NEXT_SHAPE_ID.fetch_add(1, Ordering::Relaxed),
            extensible: true,
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

    /// Assigns a fresh shape identifier, signalling that the structural layout
    /// (set of property names or their attribute flags) has changed.
    #[inline]
    fn bump_shape_id(&mut self) {
        self.shape_id = NEXT_SHAPE_ID.fetch_add(1, Ordering::Relaxed);
    }

    // ── Shape / offset API ───────────────────────────────────────────────

    /// Returns the current shape identifier.
    ///
    /// The value changes on every structural mutation (property add, remove,
    /// or attribute change) but is stable across value-only updates.
    #[inline]
    pub fn shape_id(&self) -> u64 {
        self.shape_id
    }

    /// Returns the slot index (offset) for `key`, or `None` if absent.
    ///
    /// The offset is valid as long as `shape_id()` does not change.
    #[inline]
    pub fn offset_of(&self, key: &str) -> Option<usize> {
        self.index.get(key).copied()
    }

    /// Returns the value at a raw slot offset.
    ///
    /// # Safety contract (logical)
    ///
    /// The caller must ensure that `offset` was obtained from
    /// [`offset_of`](Self::offset_of) while `shape_id()` has not changed
    /// since.
    #[inline]
    pub fn get_by_offset(&self, offset: usize) -> Option<&JsValue> {
        self.values.get(offset)
    }

    /// Overwrites the value at a raw slot offset, returning `true` on
    /// success.
    ///
    /// The same validity constraint as [`get_by_offset`](Self::get_by_offset)
    /// applies.
    #[inline]
    pub fn set_by_offset(&mut self, offset: usize, value: JsValue) -> bool {
        if let Some(slot) = self.values.get_mut(offset) {
            *slot = value;
            true
        } else {
            false
        }
    }

    /// Returns `true` if the property at `offset` has the `WRITABLE` flag.
    #[inline]
    pub fn is_writable_by_offset(&self, offset: usize) -> bool {
        self.attrs
            .get(offset)
            .is_some_and(|a| a.contains(PropertyAttributes::WRITABLE))
    }

    // ── ECMAScript enumeration-order helpers ──────────────────────────────

    /// Returns the position at which `key` should be inserted to maintain
    /// ECMAScript §10.1.11 enumeration order: integer indices (ascending)
    /// first, then string keys in insertion order.
    fn spec_insert_pos(&self, key: &str) -> usize {
        if let Some(n) = parse_integer_index(key) {
            // Integer index: find the correct sorted position among the
            // existing integer-indexed keys that occupy the front of `keys`.
            let mut pos = 0;
            while pos < self.keys.len() {
                if let Some(existing) = parse_integer_index(&self.keys[pos]) {
                    if existing > n {
                        return pos;
                    }
                    pos += 1;
                } else {
                    return pos;
                }
            }
            pos
        } else {
            // String property: append after all existing entries.
            self.keys.len()
        }
    }

    /// Increments all HashMap index values `>= pos` by one, preparing for
    /// an element insertion at `pos`.
    fn index_shift_right(&mut self, pos: usize) {
        for idx in self.index.values_mut() {
            if *idx >= pos {
                *idx += 1;
            }
        }
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
    ///
    /// New properties are placed according to ECMAScript enumeration order:
    /// integer-indexed keys occupy the front of the storage (sorted
    /// numerically), followed by string keys in insertion order.
    pub fn insert(&mut self, key: String, value: JsValue) {
        if let Some(&i) = self.index.get(&key) {
            self.values[i] = value;
        } else {
            // Non-extensible objects reject new properties (except internal __dunder__ keys).
            if !self.extensible && !key.starts_with("__") {
                return;
            }
            // __proto__ is an internal link and must never be enumerable (ES §B.2.2.1).
            let attrs = if key == "__proto__" {
                BUILTIN_ATTRS
            } else {
                DEFAULT_ATTRS
            };
            let pos = self.spec_insert_pos(&key);
            self.index_shift_right(pos);
            self.index.insert(key.clone(), pos);
            self.keys.insert(pos, key);
            self.values.insert(pos, value);
            self.attrs.insert(pos, attrs);
            self.bump_shape_id();
            if pos != self.keys.len() - 1 {
                self.cache_invalidate();
            }
        }
    }

    /// Insert a built-in method or constructor property (writable,
    /// non-enumerable, configurable — per ES spec).
    pub fn insert_builtin(&mut self, key: String, value: JsValue) {
        if let Some(&i) = self.index.get(&key) {
            self.values[i] = value;
            self.attrs[i] = BUILTIN_ATTRS;
        } else {
            let pos = self.spec_insert_pos(&key);
            self.index_shift_right(pos);
            self.index.insert(key.clone(), pos);
            self.keys.insert(pos, key);
            self.values.insert(pos, value);
            self.attrs.insert(pos, BUILTIN_ATTRS);
            self.bump_shape_id();
            if pos != self.keys.len() - 1 {
                self.cache_invalidate();
            }
        }
    }

    /// Set all existing properties to non-enumerable.
    ///
    /// Called after populating built-in prototype objects whose methods
    /// should not appear in `for…in` or `Object.keys()`.
    pub fn make_all_non_enumerable(&mut self) {
        for attr in &mut self.attrs {
            attr.remove(PropertyAttributes::ENUMERABLE);
        }
    }

    /// Removes the entry for `key`, returning the old value (if any).
    ///
    /// Uses an order-preserving shift-remove so that the ECMAScript
    /// enumeration order of the remaining properties is maintained.
    pub fn remove(&mut self, key: &str) -> Option<JsValue> {
        if let Some(i) = self.index.remove(key) {
            let val = self.values[i].clone();
            self.keys.remove(i);
            self.values.remove(i);
            self.attrs.remove(i);
            // Decrement indices for every slot that shifted left.
            for idx in self.index.values_mut() {
                if *idx > i {
                    *idx -= 1;
                }
            }
            self.bump_shape_id();
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
    ///
    /// New properties are placed according to ECMAScript enumeration order
    /// (see [`insert`][Self::insert]).
    pub fn insert_with_attrs(&mut self, key: String, value: JsValue, attrs: PropertyAttributes) {
        if let Some(&i) = self.index.get(&key) {
            self.values[i] = value;
            self.attrs[i] = attrs;
        } else {
            let pos = self.spec_insert_pos(&key);
            self.index_shift_right(pos);
            self.index.insert(key.clone(), pos);
            self.keys.insert(pos, key);
            self.values.insert(pos, value);
            self.attrs.insert(pos, attrs);
            self.bump_shape_id();
            if pos != self.keys.len() - 1 {
                self.cache_invalidate();
            }
        }
    }

    /// Updates the attribute flags for an existing property.
    /// Returns `true` if the property existed and was updated.
    pub fn set_attrs(&mut self, key: &str, attrs: PropertyAttributes) -> bool {
        if let Some(&i) = self.index.get(key) {
            self.attrs[i] = attrs;
            self.bump_shape_id();
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
            self.bump_shape_id();
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
            self.bump_shape_id();
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
            self.bump_shape_id();
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

    /// Returns an iterator over `(key, value)` pairs for only enumerable
    /// properties — the set that ES `EnumerableOwnProperties` would return.
    pub fn enumerable_iter(&self) -> impl Iterator<Item = (&String, &JsValue)> {
        self.keys
            .iter()
            .zip(self.values.iter())
            .zip(self.attrs.iter())
            .filter(|((_, _), a)| a.contains(PropertyAttributes::ENUMERABLE))
            .map(|((k, v), _)| (k, v))
    }

    /// Returns an iterator over `(key, value, attrs)` triples.
    pub fn iter_with_attrs(&self) -> impl Iterator<Item = (&String, &JsValue, PropertyAttributes)> {
        self.keys
            .iter()
            .zip(self.values.iter())
            .zip(self.attrs.iter())
            .map(|((k, v), a)| (k, v, *a))
    }

    /// Mark all properties as non-writable and non-configurable, and prevent
    /// new properties from being added (ES §20.1.2.6).
    pub fn freeze(&mut self) {
        for a in &mut self.attrs {
            a.remove(PropertyAttributes::WRITABLE);
            a.remove(PropertyAttributes::CONFIGURABLE);
        }
        self.extensible = false;
        self.bump_shape_id();
    }

    /// Returns `true` if the object is frozen: non-extensible with all
    /// properties non-writable and non-configurable (ES §20.1.2.15).
    pub fn is_frozen(&self) -> bool {
        if self.extensible {
            return false;
        }
        self.attrs.iter().all(|a| {
            !a.contains(PropertyAttributes::WRITABLE)
                && !a.contains(PropertyAttributes::CONFIGURABLE)
        })
    }

    /// Mark all properties as non-configurable and prevent new properties
    /// from being added (ES §20.1.2.20).
    pub fn seal(&mut self) {
        for a in &mut self.attrs {
            a.remove(PropertyAttributes::CONFIGURABLE);
        }
        self.extensible = false;
        self.bump_shape_id();
    }

    /// Returns `true` if the object is sealed: non-extensible with all
    /// properties non-configurable (ES §20.1.2.16).
    pub fn is_sealed(&self) -> bool {
        if self.extensible {
            return false;
        }
        self.attrs
            .iter()
            .all(|a| !a.contains(PropertyAttributes::CONFIGURABLE))
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

    // ── ECMAScript enumeration-order tests ───────────────────────────────

    #[test]
    fn test_parse_integer_index() {
        assert_eq!(parse_integer_index("0"), Some(0));
        assert_eq!(parse_integer_index("1"), Some(1));
        assert_eq!(parse_integer_index("42"), Some(42));
        assert_eq!(parse_integer_index("4294967294"), Some(u32::MAX - 1));
        // u32::MAX is NOT a valid array index.
        assert_eq!(parse_integer_index("4294967295"), None);
        // Leading zeros are not canonical.
        assert_eq!(parse_integer_index("01"), None);
        assert_eq!(parse_integer_index("007"), None);
        // Non-numeric strings.
        assert_eq!(parse_integer_index(""), None);
        assert_eq!(parse_integer_index("abc"), None);
        assert_eq!(parse_integer_index("-1"), None);
        assert_eq!(parse_integer_index("1.5"), None);
    }

    #[test]
    fn test_integer_indices_sorted_before_strings() {
        let mut pm = PropertyMap::new();
        // Insert in non-spec order: strings first, then integers.
        pm.insert("b".to_string(), JsValue::Smi(1));
        pm.insert("2".to_string(), JsValue::Smi(2));
        pm.insert("a".to_string(), JsValue::Smi(3));
        pm.insert("0".to_string(), JsValue::Smi(4));
        // Expected spec order: 0, 2, b, a
        let keys: Vec<&String> = pm.keys().collect();
        assert_eq!(
            keys.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            vec!["0", "2", "b", "a"]
        );
    }

    #[test]
    fn test_integer_indices_ascending_numeric_order() {
        let mut pm = PropertyMap::new();
        pm.insert("10".to_string(), JsValue::Smi(10));
        pm.insert("2".to_string(), JsValue::Smi(2));
        pm.insert("1".to_string(), JsValue::Smi(1));
        pm.insert("20".to_string(), JsValue::Smi(20));
        let keys: Vec<&str> = pm.keys().map(|s| s.as_str()).collect();
        assert_eq!(keys, vec!["1", "2", "10", "20"]);
    }

    #[test]
    fn test_string_keys_preserve_insertion_order() {
        let mut pm = PropertyMap::new();
        pm.insert("z".to_string(), JsValue::Smi(1));
        pm.insert("a".to_string(), JsValue::Smi(2));
        pm.insert("m".to_string(), JsValue::Smi(3));
        let keys: Vec<&str> = pm.keys().map(|s| s.as_str()).collect();
        assert_eq!(keys, vec!["z", "a", "m"]);
    }

    #[test]
    fn test_mixed_integer_and_string_order() {
        let mut pm = PropertyMap::new();
        // Simulate: obj.z = 1; obj[5] = 2; obj.a = 3; obj[1] = 4; obj.m = 5; obj[3] = 6;
        pm.insert("z".to_string(), JsValue::Smi(1));
        pm.insert("5".to_string(), JsValue::Smi(2));
        pm.insert("a".to_string(), JsValue::Smi(3));
        pm.insert("1".to_string(), JsValue::Smi(4));
        pm.insert("m".to_string(), JsValue::Smi(5));
        pm.insert("3".to_string(), JsValue::Smi(6));
        // Spec: integer indices ascending, then strings in insertion order.
        let keys: Vec<&str> = pm.keys().map(|s| s.as_str()).collect();
        assert_eq!(keys, vec!["1", "3", "5", "z", "a", "m"]);
    }

    #[test]
    fn test_remove_preserves_order() {
        let mut pm = PropertyMap::new();
        pm.insert("a".to_string(), JsValue::Smi(1));
        pm.insert("b".to_string(), JsValue::Smi(2));
        pm.insert("c".to_string(), JsValue::Smi(3));
        pm.remove("b");
        let keys: Vec<&str> = pm.keys().map(|s| s.as_str()).collect();
        assert_eq!(keys, vec!["a", "c"]);
        // Remaining entries still accessible.
        assert_eq!(pm.get("a"), Some(&JsValue::Smi(1)));
        assert_eq!(pm.get("c"), Some(&JsValue::Smi(3)));
    }

    #[test]
    fn test_remove_preserves_spec_order() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(1));
        pm.insert("3".to_string(), JsValue::Smi(2));
        pm.insert("y".to_string(), JsValue::Smi(3));
        pm.insert("1".to_string(), JsValue::Smi(4));
        // Before remove: 1, 3, x, y
        pm.remove("3");
        let keys: Vec<&str> = pm.keys().map(|s| s.as_str()).collect();
        assert_eq!(keys, vec!["1", "x", "y"]);
    }

    #[test]
    fn test_enumerable_keys_spec_order() {
        let mut pm = PropertyMap::new();
        pm.insert("b".to_string(), JsValue::Smi(1));
        pm.insert("2".to_string(), JsValue::Smi(2));
        pm.insert_with_attrs(
            "hidden".to_string(),
            JsValue::Smi(99),
            PropertyAttributes::WRITABLE, // not enumerable
        );
        pm.insert("0".to_string(), JsValue::Smi(3));
        // Enumerable keys should follow spec order, excluding "hidden".
        let keys: Vec<&str> = pm.enumerable_keys().map(|s| s.as_str()).collect();
        assert_eq!(keys, vec!["0", "2", "b"]);
    }

    #[test]
    fn test_iter_values_match_spec_ordered_keys() {
        let mut pm = PropertyMap::new();
        pm.insert("b".to_string(), JsValue::Smi(10));
        pm.insert("1".to_string(), JsValue::Smi(20));
        pm.insert("0".to_string(), JsValue::Smi(30));
        // Spec order: 0, 1, b — values should follow.
        let pairs: Vec<(&str, &JsValue)> = pm.iter().map(|(k, v)| (k.as_str(), v)).collect();
        assert_eq!(
            pairs,
            vec![
                ("0", &JsValue::Smi(30)),
                ("1", &JsValue::Smi(20)),
                ("b", &JsValue::Smi(10)),
            ]
        );
    }

    #[test]
    fn test_insert_existing_integer_key_no_reorder() {
        let mut pm = PropertyMap::new();
        pm.insert("1".to_string(), JsValue::Smi(10));
        pm.insert("a".to_string(), JsValue::Smi(20));
        // Re-insert "1" — should update value, not move it.
        pm.insert("1".to_string(), JsValue::Smi(99));
        let keys: Vec<&str> = pm.keys().map(|s| s.as_str()).collect();
        assert_eq!(keys, vec!["1", "a"]);
        assert_eq!(pm.get("1"), Some(&JsValue::Smi(99)));
    }

    // ── Shape ID / offset API tests ──────────────────────────────────────

    #[test]
    fn test_shape_id_stable_on_value_update() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(1));
        let id_after_insert = pm.shape_id();
        // Updating an existing value does not change the shape.
        pm.insert("x".to_string(), JsValue::Smi(2));
        assert_eq!(pm.shape_id(), id_after_insert);
    }

    #[test]
    fn test_shape_id_changes_on_new_property() {
        let mut pm = PropertyMap::new();
        let id0 = pm.shape_id();
        pm.insert("x".to_string(), JsValue::Smi(1));
        assert_ne!(pm.shape_id(), id0);
    }

    #[test]
    fn test_shape_id_changes_on_remove() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(1));
        let id1 = pm.shape_id();
        pm.remove("x");
        assert_ne!(pm.shape_id(), id1);
    }

    #[test]
    fn test_shape_id_changes_on_attr_change() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(1));
        let id1 = pm.shape_id();
        pm.set_writable("x", false);
        assert_ne!(pm.shape_id(), id1);
    }

    #[test]
    fn test_offset_of_and_get_by_offset() {
        let mut pm = PropertyMap::new();
        pm.insert("a".to_string(), JsValue::Smi(10));
        pm.insert("b".to_string(), JsValue::Smi(20));
        let off_a = pm.offset_of("a").unwrap();
        let off_b = pm.offset_of("b").unwrap();
        assert_eq!(pm.get_by_offset(off_a), Some(&JsValue::Smi(10)));
        assert_eq!(pm.get_by_offset(off_b), Some(&JsValue::Smi(20)));
        assert!(pm.offset_of("missing").is_none());
        assert!(pm.get_by_offset(999).is_none());
    }

    #[test]
    fn test_set_by_offset() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(1));
        let off = pm.offset_of("x").unwrap();
        assert!(pm.set_by_offset(off, JsValue::Smi(42)));
        assert_eq!(pm.get("x"), Some(&JsValue::Smi(42)));
        assert!(!pm.set_by_offset(999, JsValue::Null));
    }

    #[test]
    fn test_is_writable_by_offset() {
        let mut pm = PropertyMap::new();
        pm.insert("w".to_string(), JsValue::Smi(1));
        pm.insert_with_attrs(
            "ro".to_string(),
            JsValue::Smi(2),
            PropertyAttributes::ENUMERABLE,
        );
        let off_w = pm.offset_of("w").unwrap();
        let off_ro = pm.offset_of("ro").unwrap();
        assert!(pm.is_writable_by_offset(off_w));
        assert!(!pm.is_writable_by_offset(off_ro));
        assert!(!pm.is_writable_by_offset(999));
    }

    #[test]
    fn test_unique_shape_ids_across_maps() {
        let pm1 = PropertyMap::new();
        let pm2 = PropertyMap::new();
        assert_ne!(pm1.shape_id(), pm2.shape_id());
    }
}
