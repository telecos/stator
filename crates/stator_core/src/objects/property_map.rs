//! A property map that stores ECMAScript property values alongside their
//! attribute flags (`[[Writable]]`, `[[Enumerable]]`, `[[Configurable]]`).
//!
//! [`PropertyMap`] is the backing store for [`JsValue::PlainObject`] and
//! provides a HashMap-like API that transparently attaches
//! [`PropertyAttributes`] to every entry.  Newly inserted properties default
//! to `WRITABLE | ENUMERABLE | CONFIGURABLE` (the ECMAScript default for
//! ordinary user-created data properties).

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

/// A map of named properties with ECMAScript attribute flags.
///
/// Each entry carries a [`JsValue`] and its [`PropertyAttributes`].
/// The API mirrors `HashMap<String, JsValue>` for ease of migration while
/// adding attribute-aware accessors.
#[derive(Debug, Clone, PartialEq)]
pub struct PropertyMap {
    entries: HashMap<String, (JsValue, PropertyAttributes)>,
}

impl PropertyMap {
    /// Creates an empty property map.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Creates a property map pre-allocated for `capacity` entries.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
        }
    }

    // ── HashMap-compatible API ────────────────────────────────────────────

    /// Returns the value for `key`, ignoring attributes.
    pub fn get(&self, key: &str) -> Option<&JsValue> {
        self.entries.get(key).map(|(v, _)| v)
    }

    /// Returns a clone of the value for `key`, ignoring attributes.
    pub fn get_cloned(&self, key: &str) -> Option<JsValue> {
        self.entries.get(key).map(|(v, _)| v.clone())
    }

    /// Returns `true` if the map contains an entry for `key`.
    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Inserts a property with default attributes (writable, enumerable,
    /// configurable).  If the key already exists the value is replaced but
    /// the existing attributes are preserved.
    pub fn insert(&mut self, key: String, value: JsValue) {
        let attrs = self
            .entries
            .get(&key)
            .map(|(_, a)| *a)
            .unwrap_or(DEFAULT_ATTRS);
        self.entries.insert(key, (value, attrs));
    }

    /// Removes the entry for `key`, returning the old value (if any).
    pub fn remove(&mut self, key: &str) -> Option<JsValue> {
        self.entries.remove(key).map(|(v, _)| v)
    }

    /// Returns an iterator over the property keys.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }

    /// Returns an iterator over `(key, value)` pairs, ignoring attributes.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &JsValue)> {
        self.entries.iter().map(|(k, (v, _))| (k, v))
    }

    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the map contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // ── Attribute-aware API ───────────────────────────────────────────────

    /// Returns the value and attribute flags for `key`.
    pub fn get_with_attrs(&self, key: &str) -> Option<(&JsValue, PropertyAttributes)> {
        self.entries.get(key).map(|(v, a)| (v, *a))
    }

    /// Inserts a property with explicit attribute flags.
    pub fn insert_with_attrs(&mut self, key: String, value: JsValue, attrs: PropertyAttributes) {
        self.entries.insert(key, (value, attrs));
    }

    /// Updates the attribute flags for an existing property.
    /// Returns `true` if the property existed and was updated.
    pub fn set_attrs(&mut self, key: &str, attrs: PropertyAttributes) -> bool {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.1 = attrs;
            true
        } else {
            false
        }
    }

    /// Returns the attribute flags for `key`, or `None` if absent.
    pub fn attrs(&self, key: &str) -> Option<PropertyAttributes> {
        self.entries.get(key).map(|(_, a)| *a)
    }

    /// Returns `true` if the property exists and is writable.
    pub fn is_writable(&self, key: &str) -> bool {
        self.entries
            .get(key)
            .map(|(_, a)| a.contains(PropertyAttributes::WRITABLE))
            .unwrap_or(false)
    }

    /// Returns `true` if the property exists and is configurable.
    pub fn is_configurable(&self, key: &str) -> bool {
        self.entries
            .get(key)
            .map(|(_, a)| a.contains(PropertyAttributes::CONFIGURABLE))
            .unwrap_or(false)
    }

    /// Returns `true` if the property exists and is enumerable.
    pub fn is_enumerable(&self, key: &str) -> bool {
        self.entries
            .get(key)
            .map(|(_, a)| a.contains(PropertyAttributes::ENUMERABLE))
            .unwrap_or(false)
    }

    /// Returns an iterator over only the enumerable property keys.
    pub fn enumerable_keys(&self) -> impl Iterator<Item = &String> {
        self.entries
            .iter()
            .filter(|(_, (_, a))| a.contains(PropertyAttributes::ENUMERABLE))
            .map(|(k, _)| k)
    }

    /// Returns an iterator over `(key, value, attrs)` triples.
    pub fn iter_with_attrs(&self) -> impl Iterator<Item = (&String, &JsValue, PropertyAttributes)> {
        self.entries.iter().map(|(k, (v, a))| (k, v, *a))
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
}
