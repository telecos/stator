//! JavaScript ordinary objects with V8-style property storage.
//!
//! # Storage model
//!
//! A [`JsObject`] starts life in **fast mode**: named properties are backed by
//! a [`SmallVec`] whose indices mirror the [`Map`]'s descriptor table.  When
//! the number of named properties exceeds [`MAX_FAST_PROPERTIES`] the object
//! is *normalised* into **slow (dictionary) mode**, where each entry carries
//! both its value and its [`PropertyAttributes`] inside a [`HashMap`].
//!
//! Indexed properties (u32-keyed per ECMAScript, stored via `usize` for Vec
//! indexing) are always stored in a separate `Vec<JsValue>` elements backing
//! store, independent of the named-property mode.
//!
//! # Prototype chain
//!
//! Each `JsObject` optionally holds a reference-counted pointer to a prototype
//! object (`Rc<RefCell<JsObject>>`).  Property lookup, existence tests, and
//! write-through checks all walk the chain automatically.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use smallvec::SmallVec;

use crate::error::{StatorError, StatorResult};
use crate::gc::trace::{Trace, Tracer};
use crate::objects::map::{InstanceType, Map, PropertyAttributes, PropertyDescriptor};
use crate::objects::value::JsValue;

/// Number of named-property slots stored directly in the object before the
/// property store overflows to a [`HashMap`] (slow / dictionary mode).
pub const MAX_FAST_PROPERTIES: usize = 8;

/// A named property entry in slow (dictionary-mode) storage.
///
/// Combines the property value and its attribute flags so that the `HashMap`
/// key alone is sufficient to look up both.
#[derive(Debug, Clone)]
pub struct SlowProperty {
    value: JsValue,
    attributes: PropertyAttributes,
}

impl SlowProperty {
    /// Creates a `SlowProperty` with the given value and attribute flags.
    pub fn new(value: JsValue, attributes: PropertyAttributes) -> Self {
        Self { value, attributes }
    }

    /// Returns a reference to the stored value.
    pub fn value(&self) -> &JsValue {
        &self.value
    }

    /// Returns the property attribute flags.
    pub fn attributes(&self) -> PropertyAttributes {
        self.attributes
    }
}

/// Named-property backing store: fast (descriptor-indexed flat array) or slow
/// (dictionary / `HashMap`).
enum NamedProperties {
    /// Fast mode: values stored at the same index as the corresponding
    /// [`PropertyDescriptor`] in the object's [`Map`].  Up to
    /// [`MAX_FAST_PROPERTIES`] properties are held inline via [`SmallVec`].
    Fast(Box<SmallVec<[JsValue; MAX_FAST_PROPERTIES]>>),
    /// Slow (dictionary) mode: each entry carries both its value and its
    /// [`PropertyAttributes`].  Used once more than [`MAX_FAST_PROPERTIES`]
    /// properties have been defined.
    Slow(HashMap<String, SlowProperty>),
}

/// A JavaScript ordinary object per ECMAScript §10.1.
///
/// # Property storage
///
/// Named properties live in one of two backing stores selected by the internal
/// `NamedProperties` enum:
///
/// * **Fast mode** — values stored in a [`SmallVec`] aligned with the [`Map`]'s
///   descriptor table.  Provides O(n) lookup where n ≤ [`MAX_FAST_PROPERTIES`].
///
/// * **Slow / dictionary mode** — values and attribute flags stored in a
///   [`HashMap`].  The object transitions to slow mode automatically when more
///   than [`MAX_FAST_PROPERTIES`] distinct named properties are defined.
///
/// Indexed properties (u32-keyed, array-like) are always stored in a separate
/// `Vec<JsValue>` elements backing store regardless of the named-property mode.
///
/// # Prototype chain
///
/// An optional `Rc<RefCell<JsObject>>` prototype link implements the standard
/// ECMAScript prototype chain.  [`get_property`][JsObject::get_property] and
/// [`has_property`][JsObject::has_property] traverse the chain automatically.
pub struct JsObject {
    /// Hidden class (shape descriptor) for named fast properties.
    map: Map,
    /// Backing store for named (string-keyed) properties.
    named_properties: NamedProperties,
    /// Backing store for indexed (u32-keyed per ECMAScript) properties.
    elements: Vec<JsValue>,
    /// Prototype object, or `None` for base objects.
    prototype: Option<Rc<RefCell<JsObject>>>,
}

impl JsObject {
    /// Creates an empty ordinary object with no prototype and no properties.
    pub fn new() -> Self {
        Self {
            map: Map::new(InstanceType::JsObject, 0),
            named_properties: NamedProperties::Fast(Box::new(SmallVec::new())),
            elements: Vec::new(),
            prototype: None,
        }
    }

    /// Creates an empty object using `instance_type` as the hidden-class tag.
    ///
    /// This is used internally by subtypes such as [`JsArray`][super::js_array::JsArray]
    /// that need to stamp the correct [`InstanceType`] into the object's [`Map`]
    /// while reusing the same storage layout.
    pub fn new_with_instance_type(instance_type: InstanceType) -> Self {
        Self {
            map: Map::new(instance_type, 0),
            named_properties: NamedProperties::Fast(Box::new(SmallVec::new())),
            elements: Vec::new(),
            prototype: None,
        }
    }

    /// Creates an empty ordinary object with the given prototype.
    pub fn with_prototype(prototype: Rc<RefCell<JsObject>>) -> Self {
        Self {
            map: Map::new(InstanceType::JsObject, 0),
            named_properties: NamedProperties::Fast(Box::new(SmallVec::new())),
            elements: Vec::new(),
            prototype: Some(prototype),
        }
    }

    /// Returns a reference to this object's hidden class ([`Map`]).
    pub fn map(&self) -> &Map {
        &self.map
    }

    /// Returns `true` if this object is in fast (descriptor-backed) mode.
    pub fn is_fast_mode(&self) -> bool {
        matches!(self.named_properties, NamedProperties::Fast(_))
    }

    /// Returns the prototype of this object, if any.
    pub fn prototype(&self) -> Option<&Rc<RefCell<JsObject>>> {
        self.prototype.as_ref()
    }

    /// Sets (or removes) the prototype of this object.
    pub fn set_prototype(&mut self, prototype: Option<Rc<RefCell<JsObject>>>) {
        self.prototype = prototype;
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Normalises this object from fast to slow mode.
    ///
    /// Builds a `HashMap` from the current `Map` descriptors + value slots and
    /// replaces `named_properties` with `NamedProperties::Slow`.
    fn normalise_to_slow(&mut self) {
        let new_storage = if let NamedProperties::Fast(ref values) = self.named_properties {
            let mut map = HashMap::new();
            for (i, desc) in self.map.descriptors().iter().enumerate() {
                if let Some(val) = values.get(i) {
                    map.insert(
                        desc.key().to_string(),
                        SlowProperty::new(val.clone(), desc.attributes()),
                    );
                }
            }
            Some(NamedProperties::Slow(map))
        } else {
            None
        };
        if let Some(storage) = new_storage {
            self.named_properties = storage;
        }
    }

    /// Returns the descriptor index and attribute flags for `key` in fast mode,
    /// or `None` if not found (or if already in slow mode).
    fn fast_index_and_attrs(&self, key: &str) -> Option<(usize, PropertyAttributes)> {
        self.map
            .descriptors()
            .iter()
            .enumerate()
            .find(|(_, d)| d.key() == key)
            .map(|(i, d)| (i, d.attributes()))
    }

    /// Returns the attribute flags of an own property, or `None` if the
    /// property does not exist on this object.
    fn own_property_attrs(&self, key: &str) -> Option<PropertyAttributes> {
        match &self.named_properties {
            NamedProperties::Fast(_) => self.fast_index_and_attrs(key).map(|(_, a)| a),
            NamedProperties::Slow(map) => map.get(key).map(|e| e.attributes),
        }
    }

    /// Returns `true` if `key` exists anywhere in the prototype chain with
    /// `WRITABLE` **not** set (i.e., a read-only data property).
    fn is_readonly_in_chain(&self, key: &str) -> bool {
        if let Some(attrs) = self.own_property_attrs(key) {
            return !attrs.contains(PropertyAttributes::WRITABLE);
        }
        if let Some(proto) = &self.prototype {
            return proto.borrow().is_readonly_in_chain(key);
        }
        false
    }

    // ── Own property operations ───────────────────────────────────────────────

    /// Returns the value of an own property, or `None` if it does not exist.
    pub fn get_own_property(&self, key: &str) -> Option<JsValue> {
        match &self.named_properties {
            NamedProperties::Fast(values) => self
                .fast_index_and_attrs(key)
                .and_then(|(i, _)| values.get(i).cloned()),
            NamedProperties::Slow(map) => map.get(key).map(|e| e.value.clone()),
        }
    }

    /// Returns `true` if this object has an own property named `key`.
    pub fn has_own_property(&self, key: &str) -> bool {
        match &self.named_properties {
            NamedProperties::Fast(_) => self.fast_index_and_attrs(key).is_some(),
            NamedProperties::Slow(map) => map.contains_key(key),
        }
    }

    // ── Prototype-chain traversal (ECMAScript §10.1) ──────────────────────────

    /// ECMAScript §10.1.8 `[[Get]]`.
    ///
    /// Returns the value of property `key` found on this object or anywhere in
    /// its prototype chain, or [`JsValue::Undefined`] if not found.
    pub fn get_property(&self, key: &str) -> JsValue {
        if let Some(v) = self.get_own_property(key) {
            return v;
        }
        if let Some(proto) = &self.prototype {
            return proto.borrow().get_property(key);
        }
        JsValue::Undefined
    }

    /// ECMAScript §10.1.7 `[[HasProperty]]`.
    ///
    /// Returns `true` if property `key` exists on this object or anywhere in
    /// its prototype chain.
    pub fn has_property(&self, key: &str) -> bool {
        if self.has_own_property(key) {
            return true;
        }
        if let Some(proto) = &self.prototype {
            return proto.borrow().has_property(key);
        }
        false
    }

    /// ECMAScript §10.1.9 `[[Set]]`.
    ///
    /// Updates an existing own property or creates a new one with default
    /// attributes (`WRITABLE | ENUMERABLE | CONFIGURABLE`).
    ///
    /// Returns [`StatorError::TypeError`] if:
    /// * the own property exists but is non-writable, or
    /// * the property is not own but is found in the prototype chain as
    ///   non-writable.
    pub fn set_property(&mut self, key: &str, value: JsValue) -> StatorResult<()> {
        let default_attrs = PropertyAttributes::WRITABLE
            | PropertyAttributes::ENUMERABLE
            | PropertyAttributes::CONFIGURABLE;

        // Determine existing state (index + attributes) before any mutation.
        let existing = match &self.named_properties {
            NamedProperties::Fast(values) => self
                .fast_index_and_attrs(key)
                .and_then(|(i, a)| values.get(i).map(|_| (i, a))),
            NamedProperties::Slow(map) => map.get(key).map(|e| (usize::MAX, e.attributes)),
        };

        if let Some((idx, attrs)) = existing {
            // Property already exists on this object.
            if !attrs.contains(PropertyAttributes::WRITABLE) {
                return Err(StatorError::TypeError(format!(
                    "Cannot assign to read-only property '{key}'"
                )));
            }
            match &mut self.named_properties {
                NamedProperties::Fast(values) => values[idx] = value,
                NamedProperties::Slow(map) => {
                    if let Some(entry) = map.get_mut(key) {
                        entry.value = value;
                    }
                }
            }
        } else {
            // Property does not exist on this object: check prototype chain.
            if let Some(proto) = &self.prototype
                && proto.borrow().is_readonly_in_chain(key)
            {
                return Err(StatorError::TypeError(format!(
                    "Cannot assign to read-only property '{key}' in prototype chain"
                )));
            }

            // Create a new own property.
            let fast_len = match &self.named_properties {
                NamedProperties::Fast(v) => Some(v.len()),
                NamedProperties::Slow(_) => None,
            };
            if let Some(len) = fast_len {
                if len < MAX_FAST_PROPERTIES {
                    // Map transition: add a new descriptor.
                    self.map
                        .add_descriptor(PropertyDescriptor::new(key, default_attrs));
                    if let NamedProperties::Fast(ref mut values) = self.named_properties {
                        values.push(value);
                    }
                } else {
                    // Exceeded fast-mode capacity: normalise then add.
                    self.normalise_to_slow();
                    if let NamedProperties::Slow(ref mut map) = self.named_properties {
                        map.insert(key.to_string(), SlowProperty::new(value, default_attrs));
                    }
                }
            } else if let NamedProperties::Slow(ref mut map) = self.named_properties {
                map.insert(key.to_string(), SlowProperty::new(value, default_attrs));
            }
        }
        Ok(())
    }

    /// ECMAScript §10.1.6 `[[DefineOwnProperty]]`.
    ///
    /// Defines or redefines an own property with explicit attribute flags,
    /// enforcing the following constraints on existing non-configurable
    /// properties:
    ///
    /// * `[[Configurable]]` cannot be changed from `false` to `true`.
    /// * `[[Enumerable]]` cannot be changed.
    /// * `[[Writable]]` cannot be changed from `false` to `true`.
    ///
    /// Returns [`StatorError::TypeError`] when a constraint is violated.
    pub fn define_own_property(
        &mut self,
        key: &str,
        value: JsValue,
        attributes: PropertyAttributes,
    ) -> StatorResult<()> {
        if let Some(existing_attrs) = self.own_property_attrs(key) {
            // Property exists: validate attribute changes.
            if !existing_attrs.contains(PropertyAttributes::CONFIGURABLE) {
                if attributes.contains(PropertyAttributes::CONFIGURABLE) {
                    return Err(StatorError::TypeError(format!(
                        "Cannot redefine property '{key}': \
                         [[Configurable]] cannot change from false to true"
                    )));
                }
                if attributes.contains(PropertyAttributes::ENUMERABLE)
                    != existing_attrs.contains(PropertyAttributes::ENUMERABLE)
                {
                    return Err(StatorError::TypeError(format!(
                        "Cannot redefine property '{key}': \
                         [[Enumerable]] cannot change on a non-configurable property"
                    )));
                }
                if !existing_attrs.contains(PropertyAttributes::WRITABLE)
                    && attributes.contains(PropertyAttributes::WRITABLE)
                {
                    return Err(StatorError::TypeError(format!(
                        "Cannot redefine property '{key}': \
                         [[Writable]] cannot change from false to true"
                    )));
                }
            }
            // Validation passed: update in slow mode (normalise if currently fast).
            if self.is_fast_mode() {
                self.normalise_to_slow();
            }
            if let NamedProperties::Slow(ref mut map) = self.named_properties {
                map.insert(key.to_string(), SlowProperty::new(value, attributes));
            }
        } else {
            // New property: insert it.
            let fast_len = match &self.named_properties {
                NamedProperties::Fast(v) => Some(v.len()),
                NamedProperties::Slow(_) => None,
            };
            if let Some(len) = fast_len {
                if len < MAX_FAST_PROPERTIES {
                    self.map
                        .add_descriptor(PropertyDescriptor::new(key, attributes));
                    if let NamedProperties::Fast(ref mut values) = self.named_properties {
                        values.push(value);
                    }
                } else {
                    self.normalise_to_slow();
                    if let NamedProperties::Slow(ref mut map) = self.named_properties {
                        map.insert(key.to_string(), SlowProperty::new(value, attributes));
                    }
                }
            } else if let NamedProperties::Slow(ref mut map) = self.named_properties {
                map.insert(key.to_string(), SlowProperty::new(value, attributes));
            }
        }
        Ok(())
    }

    /// Deletes an own property (ECMAScript §10.1.10 `[[Delete]]`).
    ///
    /// Returns `Ok(true)` if the property was deleted or did not exist.
    /// Returns `Ok(false)` if the property is non-configurable and therefore
    /// cannot be deleted.
    pub fn delete_own_property(&mut self, key: &str) -> StatorResult<bool> {
        match self.own_property_attrs(key) {
            None => Ok(true),
            Some(attrs) if !attrs.contains(PropertyAttributes::CONFIGURABLE) => Ok(false),
            Some(_) => {
                if self.is_fast_mode() {
                    self.normalise_to_slow();
                }
                if let NamedProperties::Slow(ref mut map) = self.named_properties {
                    map.remove(key);
                }
                Ok(true)
            }
        }
    }

    // ── Indexed element operations ────────────────────────────────────────────

    /// Returns the element at `index`, or [`JsValue::Undefined`] if the index
    /// is out of bounds or the slot has not been set.
    pub fn get_element(&self, index: usize) -> JsValue {
        self.elements
            .get(index)
            .cloned()
            .unwrap_or(JsValue::Undefined)
    }

    /// Sets the element at `index`.
    ///
    /// If `index` is beyond the current length of the backing store, the store
    /// is extended with [`JsValue::Undefined`] hole entries.
    pub fn set_element(&mut self, index: usize, value: JsValue) {
        if index >= self.elements.len() {
            self.elements.resize(index + 1, JsValue::Undefined);
        }
        self.elements[index] = value;
    }

    /// Returns `true` if the element at `index` is within bounds and is not
    /// [`JsValue::Undefined`].
    ///
    /// # Note
    /// [`JsValue::Undefined`] is used for both out-of-bounds access and
    /// explicitly-stored `undefined` values, so those two cases are
    /// indistinguishable via this predicate.
    pub fn has_element(&self, index: usize) -> bool {
        self.elements
            .get(index)
            .map(|v| !v.is_undefined())
            .unwrap_or(false)
    }

    /// Removes the element at `index` by replacing it with
    /// [`JsValue::Undefined`].
    ///
    /// Returns `true` if the element existed (was not already `undefined` or
    /// out of bounds), `false` otherwise.
    pub fn delete_element(&mut self, index: usize) -> bool {
        if index < self.elements.len() && !self.elements[index].is_undefined() {
            self.elements[index] = JsValue::Undefined;
            true
        } else {
            false
        }
    }

    /// Returns the number of element slots in the backing store (including
    /// `undefined` holes created by sparse assignments).
    pub fn elements_length(&self) -> usize {
        self.elements.len()
    }

    /// Truncates the element backing store to `new_len` slots.
    ///
    /// If `new_len` is greater than or equal to the current length this is a
    /// no-op.  Slots beyond `new_len` are dropped.
    pub fn truncate_elements(&mut self, new_len: usize) {
        self.elements.truncate(new_len);
    }

    // ── Write-barrier–aware store operations ─────────────────────────────────

    /// ECMAScript `[[Set]]` with a generational write barrier.
    ///
    /// Equivalent to [`set_property`][JsObject::set_property] but also invokes
    /// the write barrier so that old-generation → young-generation pointer
    /// edges are recorded in the remembered set.
    ///
    /// # Parameters
    ///
    /// * `host` – raw pointer to the [`HeapObject`] header of the object that
    ///   *contains* this `JsObject`.  Required by the write barrier to decide
    ///   whether the store creates an old→young edge.
    /// * `key` – property name.
    /// * `value` – new property value.
    /// * `barrier` – mutable reference to the active [`WriteBarrier`].
    ///
    /// # Safety
    ///
    /// `host` must be non-null and point to the live [`HeapObject`] header that
    /// was allocated to back this `JsObject` instance.
    pub unsafe fn set_property_with_barrier(
        &mut self,
        host: *mut crate::objects::heap_object::HeapObject,
        key: &str,
        value: JsValue,
        barrier: &mut crate::gc::write_barrier::WriteBarrier<'_>,
    ) -> StatorResult<()> {
        // SAFETY: caller guarantees `host` is a valid live HeapObject.
        unsafe { barrier.record(host, std::ptr::null(), &value) };
        self.set_property(key, value)
    }

    /// Sets the element at `index` with a generational write barrier.
    ///
    /// Equivalent to [`set_element`][JsObject::set_element] but also invokes
    /// the write barrier so that old-generation → young-generation pointer
    /// edges are recorded in the remembered set.
    ///
    /// # Parameters
    ///
    /// * `host` – raw pointer to the [`HeapObject`] header of the object that
    ///   *contains* this `JsObject`.
    /// * `index` – element index.
    /// * `value` – new element value.
    /// * `barrier` – mutable reference to the active [`WriteBarrier`].
    ///
    /// # Safety
    ///
    /// `host` must be non-null and point to the live [`HeapObject`] header that
    /// was allocated to back this `JsObject` instance.
    pub unsafe fn set_element_with_barrier(
        &mut self,
        host: *mut crate::objects::heap_object::HeapObject,
        index: usize,
        value: JsValue,
        barrier: &mut crate::gc::write_barrier::WriteBarrier<'_>,
    ) {
        // SAFETY: caller guarantees `host` is a valid live HeapObject.
        unsafe { barrier.record(host, std::ptr::null(), &value) };
        self.set_element(index, value);
    }
}

impl Default for JsObject {
    fn default() -> Self {
        Self::new()
    }
}

impl Trace for JsObject {
    /// Visit every GC-managed heap reference reachable through this object.
    ///
    /// Traces:
    /// * all values in the named-property store (fast or slow mode),
    /// * all indexed elements,
    /// * the prototype chain (via the `Rc<RefCell<JsObject>>` link).
    fn trace(&self, tracer: &mut Tracer) {
        match &self.named_properties {
            NamedProperties::Fast(props) => {
                for v in props.iter() {
                    v.trace(tracer);
                }
            }
            NamedProperties::Slow(map) => {
                for prop in map.values() {
                    prop.value().trace(tracer);
                }
            }
        }
        for v in &self.elements {
            v.trace(tracer);
        }
        if let Some(proto) = &self.prototype {
            proto.borrow().trace(tracer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Property CRUD ─────────────────────────────────────────────────────────

    #[test]
    fn test_set_and_get_own_property() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(42)).unwrap();
        assert_eq!(obj.get_own_property("x"), Some(JsValue::Smi(42)));
    }

    #[test]
    fn test_get_missing_own_property_returns_none() {
        let obj = JsObject::new();
        assert_eq!(obj.get_own_property("missing"), None);
    }

    #[test]
    fn test_get_missing_property_returns_undefined() {
        let obj = JsObject::new();
        assert_eq!(obj.get_property("missing"), JsValue::Undefined);
    }

    #[test]
    fn test_update_existing_property() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        obj.set_property("x", JsValue::Smi(2)).unwrap();
        assert_eq!(obj.get_own_property("x"), Some(JsValue::Smi(2)));
    }

    #[test]
    fn test_delete_own_property() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(99)).unwrap();
        assert!(obj.has_own_property("x"));
        let deleted = obj.delete_own_property("x").unwrap();
        assert!(deleted);
        assert!(!obj.has_own_property("x"));
    }

    #[test]
    fn test_delete_nonexistent_property_returns_true() {
        let mut obj = JsObject::new();
        assert!(obj.delete_own_property("ghost").unwrap());
    }

    #[test]
    fn test_has_own_property() {
        let mut obj = JsObject::new();
        assert!(!obj.has_own_property("x"));
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        assert!(obj.has_own_property("x"));
    }

    #[test]
    fn test_multiple_properties() {
        let mut obj = JsObject::new();
        obj.set_property("a", JsValue::Smi(1)).unwrap();
        obj.set_property("b", JsValue::Boolean(true)).unwrap();
        obj.set_property("c", JsValue::String("hi".to_string()))
            .unwrap();
        assert_eq!(obj.get_own_property("a"), Some(JsValue::Smi(1)));
        assert_eq!(obj.get_own_property("b"), Some(JsValue::Boolean(true)));
        assert_eq!(
            obj.get_own_property("c"),
            Some(JsValue::String("hi".to_string()))
        );
    }

    // ── Prototype chain ───────────────────────────────────────────────────────

    #[test]
    fn test_get_property_traverses_prototype_chain() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        proto
            .borrow_mut()
            .set_property("inherited", JsValue::Smi(7))
            .unwrap();
        let child = JsObject::with_prototype(Rc::clone(&proto));
        assert_eq!(child.get_property("inherited"), JsValue::Smi(7));
    }

    #[test]
    fn test_own_property_shadows_prototype() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        proto
            .borrow_mut()
            .set_property("x", JsValue::Smi(1))
            .unwrap();
        let mut child = JsObject::with_prototype(Rc::clone(&proto));
        child.set_property("x", JsValue::Smi(99)).unwrap();
        assert_eq!(child.get_property("x"), JsValue::Smi(99));
        assert_eq!(proto.borrow().get_property("x"), JsValue::Smi(1));
    }

    #[test]
    fn test_has_property_traverses_chain() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        proto
            .borrow_mut()
            .set_property("y", JsValue::Boolean(true))
            .unwrap();
        let child = JsObject::with_prototype(Rc::clone(&proto));
        assert!(!child.has_own_property("y"));
        assert!(child.has_property("y"));
    }

    #[test]
    fn test_missing_property_with_prototype_returns_undefined() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        let child = JsObject::with_prototype(Rc::clone(&proto));
        assert_eq!(child.get_property("nope"), JsValue::Undefined);
    }

    #[test]
    fn test_multi_level_prototype_chain() {
        let grandparent = Rc::new(RefCell::new(JsObject::new()));
        grandparent
            .borrow_mut()
            .set_property("gp", JsValue::Smi(100))
            .unwrap();
        let parent = Rc::new(RefCell::new(JsObject::with_prototype(Rc::clone(
            &grandparent,
        ))));
        let child = JsObject::with_prototype(Rc::clone(&parent));
        assert_eq!(child.get_property("gp"), JsValue::Smi(100));
    }

    // ── Map transitions ───────────────────────────────────────────────────────

    #[test]
    fn test_map_descriptor_count_grows_on_add() {
        let mut obj = JsObject::new();
        assert_eq!(obj.map().descriptors().len(), 0);
        obj.set_property("a", JsValue::Smi(1)).unwrap();
        assert_eq!(obj.map().descriptors().len(), 1);
        obj.set_property("b", JsValue::Smi(2)).unwrap();
        assert_eq!(obj.map().descriptors().len(), 2);
    }

    #[test]
    fn test_fast_to_slow_transition_on_overflow() {
        let mut obj = JsObject::new();
        for i in 0..MAX_FAST_PROPERTIES {
            obj.set_property(&format!("p{i}"), JsValue::Smi(i as i32))
                .unwrap();
        }
        assert!(
            obj.is_fast_mode(),
            "should still be fast after 8 properties"
        );
        obj.set_property("overflow", JsValue::Smi(99)).unwrap();
        assert!(
            !obj.is_fast_mode(),
            "should be slow after exceeding MAX_FAST_PROPERTIES"
        );
        assert_eq!(obj.get_property("overflow"), JsValue::Smi(99));
    }

    #[test]
    fn test_slow_mode_properties_are_accessible() {
        let mut obj = JsObject::new();
        // Fill fast slots.
        for i in 0..MAX_FAST_PROPERTIES {
            obj.set_property(&format!("p{i}"), JsValue::Smi(i as i32))
                .unwrap();
        }
        // Overflow into slow mode.
        obj.set_property("slow_key", JsValue::Smi(42)).unwrap();
        assert!(!obj.is_fast_mode());
        assert_eq!(obj.get_property("slow_key"), JsValue::Smi(42));
        // Previously-fast properties remain readable.
        assert_eq!(obj.get_property("p0"), JsValue::Smi(0));
        assert_eq!(obj.get_property("p7"), JsValue::Smi(7));
    }

    #[test]
    fn test_update_slow_mode_property() {
        let mut obj = JsObject::new();
        for i in 0..=MAX_FAST_PROPERTIES {
            obj.set_property(&format!("p{i}"), JsValue::Smi(i as i32))
                .unwrap();
        }
        assert!(!obj.is_fast_mode());
        obj.set_property("p0", JsValue::Smi(999)).unwrap();
        assert_eq!(obj.get_property("p0"), JsValue::Smi(999));
    }

    // ── Non-writable / non-configurable ──────────────────────────────────────

    #[test]
    fn test_write_to_readonly_property_is_type_error() {
        let mut obj = JsObject::new();
        // Define a property with ENUMERABLE only (no WRITABLE).
        obj.define_own_property(
            "ro",
            JsValue::Smi(1),
            PropertyAttributes::ENUMERABLE | PropertyAttributes::CONFIGURABLE,
        )
        .unwrap();
        let err = obj.set_property("ro", JsValue::Smi(2)).unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    #[test]
    fn test_define_non_configurable_cannot_become_configurable() {
        let mut obj = JsObject::new();
        obj.define_own_property(
            "frozen",
            JsValue::Smi(1),
            PropertyAttributes::WRITABLE | PropertyAttributes::ENUMERABLE,
        )
        .unwrap();
        let err = obj
            .define_own_property(
                "frozen",
                JsValue::Smi(1),
                PropertyAttributes::WRITABLE
                    | PropertyAttributes::ENUMERABLE
                    | PropertyAttributes::CONFIGURABLE,
            )
            .unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    #[test]
    fn test_define_non_configurable_enumerable_cannot_change() {
        let mut obj = JsObject::new();
        obj.define_own_property("p", JsValue::Smi(0), PropertyAttributes::WRITABLE)
            .unwrap();
        // Trying to add ENUMERABLE to a non-configurable property should fail.
        let err = obj
            .define_own_property(
                "p",
                JsValue::Smi(0),
                PropertyAttributes::WRITABLE | PropertyAttributes::ENUMERABLE,
            )
            .unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    #[test]
    fn test_define_non_configurable_writable_false_to_true_rejected() {
        let mut obj = JsObject::new();
        // No WRITABLE, no CONFIGURABLE.
        obj.define_own_property("nw", JsValue::Smi(0), PropertyAttributes::empty())
            .unwrap();
        let err = obj
            .define_own_property("nw", JsValue::Smi(0), PropertyAttributes::WRITABLE)
            .unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    #[test]
    fn test_define_writable_true_to_false_allowed() {
        let mut obj = JsObject::new();
        obj.define_own_property(
            "p",
            JsValue::Smi(1),
            PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE,
        )
        .unwrap();
        // One-way: writable true → false is allowed.
        obj.define_own_property(
            "p",
            JsValue::Smi(1),
            PropertyAttributes::CONFIGURABLE, // writable removed
        )
        .unwrap();
        // Now writing should fail.
        let err = obj.set_property("p", JsValue::Smi(2)).unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    #[test]
    fn test_delete_non_configurable_property_returns_false() {
        let mut obj = JsObject::new();
        obj.define_own_property("nc", JsValue::Smi(0), PropertyAttributes::empty())
            .unwrap();
        let result = obj.delete_own_property("nc").unwrap();
        assert!(!result);
        assert!(obj.has_own_property("nc"));
    }

    #[test]
    fn test_delete_configurable_property_succeeds() {
        let mut obj = JsObject::new();
        obj.set_property("del", JsValue::Smi(1)).unwrap();
        assert!(obj.has_own_property("del"));
        assert!(obj.delete_own_property("del").unwrap());
        assert!(!obj.has_own_property("del"));
    }

    // ── Indexed element operations ────────────────────────────────────────────

    #[test]
    fn test_set_and_get_element() {
        let mut obj = JsObject::new();
        obj.set_element(0, JsValue::Smi(10));
        obj.set_element(1, JsValue::Smi(20));
        assert_eq!(obj.get_element(0), JsValue::Smi(10));
        assert_eq!(obj.get_element(1), JsValue::Smi(20));
    }

    #[test]
    fn test_get_element_out_of_bounds_returns_undefined() {
        let obj = JsObject::new();
        assert_eq!(obj.get_element(100), JsValue::Undefined);
    }

    #[test]
    fn test_set_element_beyond_length_extends_with_holes() {
        let mut obj = JsObject::new();
        obj.set_element(5, JsValue::Smi(42));
        assert_eq!(obj.elements_length(), 6);
        assert_eq!(obj.get_element(0), JsValue::Undefined);
        assert_eq!(obj.get_element(5), JsValue::Smi(42));
    }

    #[test]
    fn test_has_element() {
        let mut obj = JsObject::new();
        assert!(!obj.has_element(0));
        obj.set_element(0, JsValue::Smi(1));
        assert!(obj.has_element(0));
    }

    #[test]
    fn test_delete_element() {
        let mut obj = JsObject::new();
        obj.set_element(0, JsValue::Smi(5));
        assert!(obj.has_element(0));
        assert!(obj.delete_element(0));
        assert!(!obj.has_element(0));
    }

    #[test]
    fn test_delete_element_out_of_bounds_returns_false() {
        let mut obj = JsObject::new();
        assert!(!obj.delete_element(99));
    }

    #[test]
    fn test_elements_length() {
        let mut obj = JsObject::new();
        assert_eq!(obj.elements_length(), 0);
        obj.set_element(3, JsValue::Smi(1));
        assert_eq!(obj.elements_length(), 4);
    }

    // ── Default / constructor ─────────────────────────────────────────────────

    #[test]
    fn test_new_object_is_fast_mode() {
        let obj = JsObject::new();
        assert!(obj.is_fast_mode());
    }

    #[test]
    fn test_default_equals_new() {
        let obj: JsObject = JsObject::default();
        assert!(obj.is_fast_mode());
        assert!(obj.prototype().is_none());
    }

    #[test]
    fn test_set_and_clear_prototype() {
        let mut obj = JsObject::new();
        assert!(obj.prototype().is_none());
        let proto = Rc::new(RefCell::new(JsObject::new()));
        obj.set_prototype(Some(Rc::clone(&proto)));
        assert!(obj.prototype().is_some());
        obj.set_prototype(None);
        assert!(obj.prototype().is_none());
    }
}
