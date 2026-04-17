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

use crate::builtins::symbol::is_symbol_property_key;
use crate::error::{StatorError, StatorResult};
use crate::gc::trace::{Trace, Tracer};
use crate::objects::map::{InstanceType, Map, PropertyAttributes, PropertyDescriptor};
use crate::objects::property_descriptor::FullPropertyDescriptor;
use crate::objects::shapes::{ShapeId, ShapeTable};
use crate::objects::value::JsValue;

/// Returns `Some(n)` if `key` is a valid ECMAScript array index — a canonical
/// decimal string representing an integer in `0 ..= 2^32 − 2`.
#[inline]
fn parse_integer_index(key: &str) -> Option<u32> {
    if key.is_empty() || (key.len() > 1 && key.as_bytes()[0] == b'0') {
        return None;
    }
    let n: u32 = key.parse().ok()?;
    if n < u32::MAX { Some(n) } else { None }
}

/// Sort `keys` in ECMAScript §10.1.11 enumeration order: integer indices
/// ascending, then non-symbol string keys in their original relative order,
/// then symbol keys in their original relative order.
fn sort_keys_spec_order(keys: &mut Vec<String>) {
    // Partition into three buckets preserving relative insertion order.
    let mut indices: Vec<(u32, String)> = Vec::new();
    let mut strings: Vec<String> = Vec::new();
    let mut symbols: Vec<String> = Vec::new();
    for k in keys.drain(..) {
        if let Some(n) = parse_integer_index(&k) {
            indices.push((n, k));
        } else if is_symbol_property_key(&k) {
            symbols.push(k);
        } else {
            strings.push(k);
        }
    }
    indices.sort_by_key(|(n, _)| *n);
    keys.extend(indices.into_iter().map(|(_, k)| k));
    keys.extend(strings);
    keys.extend(symbols);
}

/// Number of named-property slots stored directly in the object before the
/// property store overflows to a [`HashMap`] (slow / dictionary mode).
pub const MAX_FAST_PROPERTIES: usize = 8;

/// The kind of a named property: data or accessor.
#[derive(Debug, Clone)]
pub enum PropertyKind {
    /// A data property with a single value.
    Data(JsValue),
    /// An accessor property with `[[Get]]` and `[[Set]]` slots.
    Accessor {
        /// The getter function, or `Undefined` if absent.
        get: JsValue,
        /// The setter function, or `Undefined` if absent.
        set: JsValue,
    },
}

/// A named property entry in slow (dictionary-mode) storage.
///
/// Combines the property kind (data or accessor) and its attribute flags so
/// that the `HashMap` key alone is sufficient to look up both.
#[derive(Debug, Clone)]
pub struct SlowProperty {
    kind: PropertyKind,
    attributes: PropertyAttributes,
}

impl SlowProperty {
    /// Creates a data `SlowProperty` with the given value and attribute flags.
    pub fn new(value: JsValue, attributes: PropertyAttributes) -> Self {
        Self {
            kind: PropertyKind::Data(value),
            attributes,
        }
    }

    /// Creates an accessor `SlowProperty` with getter/setter and attribute
    /// flags.
    pub fn new_accessor(get: JsValue, set: JsValue, attributes: PropertyAttributes) -> Self {
        Self {
            kind: PropertyKind::Accessor { get, set },
            attributes,
        }
    }

    /// Returns a reference to the stored value.
    ///
    /// For accessor properties this returns the getter.
    pub fn value(&self) -> &JsValue {
        match &self.kind {
            PropertyKind::Data(v) => v,
            PropertyKind::Accessor { get, .. } => get,
        }
    }

    /// Returns `true` if this is an accessor property.
    pub fn is_accessor(&self) -> bool {
        matches!(self.kind, PropertyKind::Accessor { .. })
    }

    /// Returns the property kind (data or accessor).
    pub fn kind(&self) -> &PropertyKind {
        &self.kind
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
    /// [`PropertyAttributes`].  `key_order` preserves ECMAScript property
    /// enumeration order (integer indices ascending, then string keys in
    /// insertion order).
    Slow {
        map: HashMap<String, SlowProperty>,
        key_order: Vec<String>,
    },
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
    /// Optional shape identifier for the new transition-tree based shape
    /// system.  When `Some`, shape-based property lookup via a [`ShapeTable`]
    /// is available alongside the legacy [`Map`]-based path.
    shape_id: Option<ShapeId>,
    /// Backing store for named (string-keyed) properties.
    named_properties: NamedProperties,
    /// Backing store for indexed (u32-keyed per ECMAScript) properties.
    elements: Vec<JsValue>,
    /// Prototype object, or `None` for base objects.
    prototype: Option<Rc<RefCell<JsObject>>>,
    /// Whether new own properties may be added to this object (ECMAScript
    /// `[[Extensible]]` internal slot, §10.1).  Defaults to `true`.
    extensible: bool,
}

impl JsObject {
    /// Creates an empty ordinary object with no prototype and no properties.
    pub fn new() -> Self {
        Self {
            map: Map::new(InstanceType::JsObject, 0),
            shape_id: None,
            named_properties: NamedProperties::Fast(Box::new(SmallVec::new())),
            elements: Vec::new(),
            prototype: None,
            extensible: true,
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
            shape_id: None,
            named_properties: NamedProperties::Fast(Box::new(SmallVec::new())),
            elements: Vec::new(),
            prototype: None,
            extensible: true,
        }
    }

    /// Creates an empty ordinary object with the given prototype.
    pub fn with_prototype(prototype: Rc<RefCell<JsObject>>) -> Self {
        Self {
            map: Map::new(InstanceType::JsObject, 0),
            shape_id: None,
            named_properties: NamedProperties::Fast(Box::new(SmallVec::new())),
            elements: Vec::new(),
            prototype: Some(prototype),
            extensible: true,
        }
    }

    /// Returns a reference to this object's hidden class ([`Map`]).
    pub fn map(&self) -> &Map {
        &self.map
    }

    /// Returns the [`ShapeId`] associated with this object, if any.
    ///
    /// When `Some`, the object participates in the transition-tree shape
    /// system and its properties can be looked up via a [`ShapeTable`].
    pub fn shape_id(&self) -> Option<ShapeId> {
        self.shape_id
    }

    /// Associates this object with a [`ShapeId`] in the global shape
    /// transition tree.
    pub fn set_shape_id(&mut self, id: ShapeId) {
        self.shape_id = Some(id);
    }

    /// Looks up an own property value using the shape system, falling back to
    /// the legacy [`Map`]-based / `HashMap`-based lookup.
    ///
    /// When the object has an associated [`ShapeId`] and the property is
    /// found in the shape's descriptor array, the value is read directly by
    /// its `field_index` from the fast-mode backing store.  Otherwise the
    /// lookup delegates to [`get_own_property`][Self::get_own_property].
    pub fn get_property_by_shape(&self, table: &ShapeTable, key: &str) -> Option<JsValue> {
        if let Some(sid) = self.shape_id
            && let Some(desc) = table.lookup(sid, key)
        {
            return self.get_fast_property_at_index(desc.field_index() as usize);
        }
        self.get_own_property(key)
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

    /// Returns `true` if new own properties may be added to this object
    /// (ECMAScript `[[Extensible]]` internal slot, §10.1).
    pub fn is_extensible(&self) -> bool {
        self.extensible
    }

    /// Marks this object as non-extensible: no new own properties may be
    /// added after this call (ECMAScript `[[PreventExtensions]]`, §10.1.3).
    ///
    /// Existing properties are unaffected.
    pub fn prevent_extensions(&mut self) {
        self.extensible = false;
    }

    /// Returns the names of all own named (string-keyed) properties in
    /// ECMAScript §10.1.11 enumeration order: integer indices ascending,
    /// then non-symbol string keys in insertion order, then symbol keys in
    /// insertion order.
    pub fn own_property_keys(&self) -> Vec<String> {
        let mut keys = match &self.named_properties {
            NamedProperties::Fast(_) => self
                .map
                .descriptors()
                .iter()
                .map(|d| d.key().to_string())
                .collect(),
            NamedProperties::Slow { key_order, .. } => key_order.clone(),
        };
        sort_keys_spec_order(&mut keys);
        keys
    }

    /// Returns the value **and** attribute flags of an own named property, or
    /// `None` if the property does not exist on this object.
    ///
    /// Corresponds to ECMAScript `[[GetOwnProperty]]` (§10.1.5).
    pub fn get_own_property_descriptor(&self, key: &str) -> Option<(JsValue, PropertyAttributes)> {
        match &self.named_properties {
            NamedProperties::Fast(values) => self
                .fast_index_and_attrs(key)
                .and_then(|(i, attrs)| values.get(i).map(|v| (v.clone(), attrs))),
            NamedProperties::Slow { map, .. } => {
                map.get(key).map(|e| (e.value().clone(), e.attributes))
            }
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Normalises this object from fast to slow mode.
    ///
    /// Builds a `HashMap` from the current `Map` descriptors + value slots and
    /// replaces `named_properties` with `NamedProperties::Slow`.
    fn normalise_to_slow(&mut self) {
        let new_storage = if let NamedProperties::Fast(ref values) = self.named_properties {
            let mut map = HashMap::new();
            let mut key_order = Vec::with_capacity(self.map.descriptors().len());
            for (i, desc) in self.map.descriptors().iter().enumerate() {
                if let Some(val) = values.get(i) {
                    let k = desc.key().to_string();
                    map.insert(k.clone(), SlowProperty::new(val.clone(), desc.attributes()));
                    key_order.push(k);
                }
            }
            Some(NamedProperties::Slow { map, key_order })
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
            NamedProperties::Slow { map, .. } => map.get(key).map(|e| e.attributes),
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
            NamedProperties::Slow { map, .. } => map.get(key).map(|e| e.value().clone()),
        }
    }

    /// Returns `true` if this object has an own property named `key`.
    pub fn has_own_property(&self, key: &str) -> bool {
        match &self.named_properties {
            NamedProperties::Fast(_) => self.fast_index_and_attrs(key).is_some(),
            NamedProperties::Slow { map, .. } => map.contains_key(key),
        }
    }

    /// Read a fast-mode named property by its zero-based descriptor index.
    ///
    /// Returns `None` if the object is in slow (dictionary) mode or `index`
    /// is out of range.  This is intended for use by the inline-cache runtime,
    /// which caches the fast index after verifying the object's hidden-class
    /// shape.
    pub fn get_fast_property_at_index(&self, index: usize) -> Option<JsValue> {
        if let NamedProperties::Fast(ref values) = self.named_properties {
            values.get(index).cloned()
        } else {
            None
        }
    }

    /// Write a fast-mode named property by its zero-based descriptor index.
    ///
    /// Returns `true` on success.  Returns `false` if the object is in slow
    /// mode, `index` is out of range, or the existing value slot does not
    /// exist (i.e., the descriptor exists but the value array is shorter than
    /// expected).  Intended for the inline-cache runtime fast path.
    pub fn set_fast_property_at_index(&mut self, index: usize, value: JsValue) -> bool {
        if let NamedProperties::Fast(ref mut values) = self.named_properties
            && index < values.len()
        {
            values[index] = value;
            return true;
        }
        false
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
            NamedProperties::Slow { map, .. } => map.get(key).map(|e| (usize::MAX, e.attributes)),
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
                NamedProperties::Slow { map, .. } => {
                    if let Some(entry) = map.get_mut(key) {
                        entry.kind = PropertyKind::Data(value);
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

            // Reject new properties when the object is non-extensible.
            if !self.extensible {
                return Err(StatorError::TypeError(format!(
                    "Cannot add property '{key}' to a non-extensible object"
                )));
            }

            // Create a new own property.
            let fast_len = match &self.named_properties {
                NamedProperties::Fast(v) => Some(v.len()),
                NamedProperties::Slow { .. } => None,
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
                    if let NamedProperties::Slow {
                        ref mut map,
                        ref mut key_order,
                    } = self.named_properties
                    {
                        map.insert(key.to_string(), SlowProperty::new(value, default_attrs));
                        key_order.push(key.to_string());
                    }
                }
            } else if let NamedProperties::Slow {
                ref mut map,
                ref mut key_order,
            } = self.named_properties
            {
                map.insert(key.to_string(), SlowProperty::new(value, default_attrs));
                key_order.push(key.to_string());
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
                // §10.1.6.3 step 9.a.i: non-writable → value must not change.
                if !existing_attrs.contains(PropertyAttributes::WRITABLE)
                    && let Some(current_value) = self.get_own_property(key)
                    && !value.same_value(&current_value)
                {
                    return Err(StatorError::TypeError(format!(
                        "Cannot redefine property '{key}': \
                         value of a non-writable, non-configurable property \
                         cannot be changed"
                    )));
                }
            }
            // Validation passed: update in slow mode (normalise if currently fast).
            if self.is_fast_mode() {
                self.normalise_to_slow();
            }
            if let NamedProperties::Slow { ref mut map, .. } = self.named_properties {
                map.insert(key.to_string(), SlowProperty::new(value, attributes));
            }
        } else {
            // New property: reject if non-extensible.
            if !self.extensible {
                return Err(StatorError::TypeError(format!(
                    "Cannot define property '{key}' on a non-extensible object"
                )));
            }
            let fast_len = match &self.named_properties {
                NamedProperties::Fast(v) => Some(v.len()),
                NamedProperties::Slow { .. } => None,
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
                    if let NamedProperties::Slow {
                        ref mut map,
                        ref mut key_order,
                    } = self.named_properties
                    {
                        map.insert(key.to_string(), SlowProperty::new(value, attributes));
                        key_order.push(key.to_string());
                    }
                }
            } else if let NamedProperties::Slow {
                ref mut map,
                ref mut key_order,
            } = self.named_properties
            {
                map.insert(key.to_string(), SlowProperty::new(value, attributes));
                key_order.push(key.to_string());
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
                if let NamedProperties::Slow {
                    ref mut map,
                    ref mut key_order,
                } = self.named_properties
                {
                    map.remove(key);
                    key_order.retain(|k| k != key);
                }
                Ok(true)
            }
        }
    }

    // ── ECMAScript §10.1 abstract operations ────────────────────────────────

    /// ECMAScript §10.1.5 *OrdinaryGetOwnProperty* ( *O*, *P* ).
    ///
    /// Returns a [`FullPropertyDescriptor`] describing the own property named
    /// `key`, or `None` if the property does not exist.  The descriptor shape
    /// correctly distinguishes data and accessor properties.
    pub fn ordinary_get_own_property(&self, key: &str) -> Option<FullPropertyDescriptor> {
        match &self.named_properties {
            NamedProperties::Fast(values) => {
                self.fast_index_and_attrs(key).and_then(|(i, attrs)| {
                    values.get(i).map(|v| FullPropertyDescriptor::Data {
                        value: v.clone(),
                        writable: attrs.contains(PropertyAttributes::WRITABLE),
                        enumerable: attrs.contains(PropertyAttributes::ENUMERABLE),
                        configurable: attrs.contains(PropertyAttributes::CONFIGURABLE),
                    })
                })
            }
            NamedProperties::Slow { map, .. } => map.get(key).map(|entry| match entry.kind() {
                PropertyKind::Data(v) => FullPropertyDescriptor::Data {
                    value: v.clone(),
                    writable: entry.attributes.contains(PropertyAttributes::WRITABLE),
                    enumerable: entry.attributes.contains(PropertyAttributes::ENUMERABLE),
                    configurable: entry.attributes.contains(PropertyAttributes::CONFIGURABLE),
                },
                PropertyKind::Accessor { get, set } => FullPropertyDescriptor::Accessor {
                    get: get.clone(),
                    set: set.clone(),
                    enumerable: entry.attributes.contains(PropertyAttributes::ENUMERABLE),
                    configurable: entry.attributes.contains(PropertyAttributes::CONFIGURABLE),
                },
            }),
        }
    }

    /// ECMAScript §10.1.6 *OrdinaryDefineOwnProperty* ( *O*, *P*, *Desc* ).
    ///
    /// Full algorithm with [`FullPropertyDescriptor`] validation including
    /// data↔accessor conversion checks.  Delegates validation to
    /// [`FullPropertyDescriptor::validate_against`].
    pub fn ordinary_define_own_property(
        &mut self,
        key: &str,
        desc: &FullPropertyDescriptor,
    ) -> StatorResult<bool> {
        let current = self.ordinary_get_own_property(key);

        if let Some(ref current_desc) = current {
            let current_attrs = current_desc.to_attributes();
            let is_configurable = current_attrs.contains(PropertyAttributes::CONFIGURABLE);

            // §10.1.6.3 step 2: if every field is absent, return true.
            if desc.is_generic() && desc.enumerable().is_none() && desc.configurable().is_none() {
                return Ok(true);
            }

            // §10.1.6.3 step 4: non-configurable property restrictions.
            if !is_configurable {
                if desc.configurable() == Some(true) {
                    return Ok(false);
                }
                if let Some(e) = desc.enumerable()
                    && e != current_attrs.contains(PropertyAttributes::ENUMERABLE)
                {
                    return Ok(false);
                }
            }

            // §10.1.6.3 step 5: generic descriptor — just update shared attrs.
            if desc.is_generic() {
                // Merge shared attributes only.
            }
            // §10.1.6.3 step 6: data↔accessor kind mismatch.
            else if current_desc.is_data() != desc.is_data() {
                if !is_configurable {
                    return Ok(false);
                }
            }
            // §10.1.6.3 step 7: both are data descriptors.
            else if current_desc.is_data() && desc.is_data() {
                if !is_configurable && !current_attrs.contains(PropertyAttributes::WRITABLE) {
                    if let FullPropertyDescriptor::Data { writable, .. } = desc
                        && *writable
                    {
                        return Ok(false);
                    }
                    if let FullPropertyDescriptor::Data { value, .. } = desc
                        && let FullPropertyDescriptor::Data { value: cur_val, .. } = current_desc
                        && !value.same_value(cur_val)
                    {
                        return Ok(false);
                    }
                }
            }
            // §10.1.6.3 step 8: both are accessor descriptors.
            else if !is_configurable
                && let (
                    FullPropertyDescriptor::Accessor {
                        get: cur_get,
                        set: cur_set,
                        ..
                    },
                    FullPropertyDescriptor::Accessor {
                        get: new_get,
                        set: new_set,
                        ..
                    },
                ) = (current_desc, desc)
                && (!new_get.same_value(cur_get) || !new_set.same_value(cur_set))
            {
                return Ok(false);
            }
        } else {
            // Property does not exist — check extensible.
            if !self.extensible {
                return Ok(false);
            }
        }

        // Apply the descriptor.
        self.apply_property_descriptor(key, desc);
        Ok(true)
    }

    /// Internal helper: writes a [`FullPropertyDescriptor`] to the property
    /// store (normalises to slow mode if necessary).
    fn apply_property_descriptor(&mut self, key: &str, desc: &FullPropertyDescriptor) {
        if self.is_fast_mode() {
            self.normalise_to_slow();
        }
        if let NamedProperties::Slow {
            ref mut map,
            ref mut key_order,
        } = self.named_properties
        {
            let entry = match desc {
                FullPropertyDescriptor::Data {
                    value,
                    writable,
                    enumerable,
                    configurable,
                } => {
                    let mut attrs = PropertyAttributes::empty();
                    if *writable {
                        attrs |= PropertyAttributes::WRITABLE;
                    }
                    if *enumerable {
                        attrs |= PropertyAttributes::ENUMERABLE;
                    }
                    if *configurable {
                        attrs |= PropertyAttributes::CONFIGURABLE;
                    }
                    SlowProperty::new(value.clone(), attrs)
                }
                FullPropertyDescriptor::Accessor {
                    get,
                    set,
                    enumerable,
                    configurable,
                } => {
                    let mut attrs = PropertyAttributes::empty();
                    if *enumerable {
                        attrs |= PropertyAttributes::ENUMERABLE;
                    }
                    if *configurable {
                        attrs |= PropertyAttributes::CONFIGURABLE;
                    }
                    SlowProperty::new_accessor(get.clone(), set.clone(), attrs)
                }
                FullPropertyDescriptor::Generic { .. } => {
                    // For a generic descriptor on an existing property, merge.
                    let existing = map.get(key);
                    let base_attrs = existing
                        .map(|e| e.attributes)
                        .unwrap_or(PropertyAttributes::empty());
                    let attrs = desc.merge_into(base_attrs);
                    let base_kind = existing
                        .map(|e| e.kind.clone())
                        .unwrap_or(PropertyKind::Data(JsValue::Undefined));
                    SlowProperty {
                        kind: base_kind,
                        attributes: attrs,
                    }
                }
            };
            if !map.contains_key(key) {
                key_order.push(key.to_string());
            }
            map.insert(key.to_string(), entry);
        }
    }

    /// ECMAScript §10.1.8 *OrdinaryGet* ( *O*, *P*, *Receiver* ).
    ///
    /// Walks the prototype chain and invokes accessor getters.  For data
    /// properties, the receiver is unused.  `receiver` is provided for
    /// Proxy / getter integration but is not called through in this
    /// simplified implementation.
    pub fn ordinary_get(&self, key: &str, _receiver: &JsValue) -> JsValue {
        if let Some(desc) = self.ordinary_get_own_property(key) {
            return match desc {
                FullPropertyDescriptor::Data { value, .. } => value,
                FullPropertyDescriptor::Accessor { get, .. } => {
                    // In a full engine, we'd call the getter here.
                    // Return the getter itself for observability.
                    get
                }
                FullPropertyDescriptor::Generic { .. } => JsValue::Undefined,
            };
        }
        if let Some(proto) = &self.prototype {
            return proto.borrow().ordinary_get(key, _receiver);
        }
        JsValue::Undefined
    }

    /// ECMAScript §10.1.9 *OrdinarySet* ( *O*, *P*, *V*, *Receiver* ).
    ///
    /// Handles setter invocation on prototype chain and receiver semantics.
    /// Returns `Ok(true)` on success, `Ok(false)` on rejection.
    pub fn ordinary_set(
        &mut self,
        key: &str,
        value: JsValue,
        _receiver: &JsValue,
    ) -> StatorResult<bool> {
        // Step 1: Get own descriptor.
        if let Some(own_desc) = self.ordinary_get_own_property(key) {
            match own_desc {
                FullPropertyDescriptor::Data { writable, .. } => {
                    if !writable {
                        return Ok(false);
                    }
                    // Update the data property value.
                    let new_desc = FullPropertyDescriptor::Data {
                        value,
                        writable: true,
                        enumerable: own_desc.enumerable().unwrap_or(false),
                        configurable: own_desc.configurable().unwrap_or(false),
                    };
                    return self.ordinary_define_own_property(key, &new_desc);
                }
                FullPropertyDescriptor::Accessor { set, .. } => {
                    if set.is_undefined() {
                        return Ok(false);
                    }
                    // In a full engine we'd call the setter.
                    return Ok(true);
                }
                _ => {}
            }
        }

        // Step 2: Walk prototype chain.
        let parent_desc = self.find_in_prototype_chain(key);
        if let Some(parent) = parent_desc {
            match parent {
                FullPropertyDescriptor::Data { writable, .. } => {
                    if !writable {
                        return Ok(false);
                    }
                }
                FullPropertyDescriptor::Accessor { set, .. } => {
                    if set.is_undefined() {
                        return Ok(false);
                    }
                    // In a full engine we'd call the setter.
                    return Ok(true);
                }
                _ => {}
            }
        }

        // Step 3: Create a new data property on receiver.
        if !self.extensible {
            return Ok(false);
        }
        let new_desc = FullPropertyDescriptor::Data {
            value,
            writable: true,
            enumerable: true,
            configurable: true,
        };
        self.ordinary_define_own_property(key, &new_desc)
    }

    /// ECMAScript §10.1.7 *OrdinaryHasProperty* ( *O*, *P* ).
    ///
    /// Walks the prototype chain. Equivalent to `[[HasProperty]]`.
    pub fn ordinary_has_property(&self, key: &str) -> bool {
        self.has_property(key)
    }

    /// ECMAScript §10.1.10 *OrdinaryDelete* ( *O*, *P* ).
    ///
    /// Respects the `[[Configurable]]` attribute.  Returns `Ok(true)` on
    /// success, `Ok(false)` if the property is non-configurable.
    pub fn ordinary_delete(&mut self, key: &str) -> StatorResult<bool> {
        self.delete_own_property(key)
    }

    /// ECMAScript §7.3.5 *CreateDataProperty* ( *O*, *P*, *V* ).
    ///
    /// Creates a new own data property with `{ [[Value]]: V, [[Writable]]:
    /// true, [[Enumerable]]: true, [[Configurable]]: true }`.
    pub fn create_data_property(&mut self, key: &str, value: JsValue) -> StatorResult<bool> {
        let desc = FullPropertyDescriptor::Data {
            value,
            writable: true,
            enumerable: true,
            configurable: true,
        };
        self.ordinary_define_own_property(key, &desc)
    }

    /// ECMAScript §7.3.6 *CreateMethodProperty* ( *O*, *P*, *V* ).
    ///
    /// Creates a new own data property with `{ [[Value]]: V, [[Writable]]:
    /// true, [[Enumerable]]: false, [[Configurable]]: true }`.
    pub fn create_method_property(&mut self, key: &str, value: JsValue) -> StatorResult<bool> {
        let desc = FullPropertyDescriptor::Data {
            value,
            writable: true,
            enumerable: false,
            configurable: true,
        };
        self.ordinary_define_own_property(key, &desc)
    }

    /// ECMAScript §7.3.8 *DefinePropertyOrThrow* ( *O*, *P*, *desc* ).
    ///
    /// Like [`ordinary_define_own_property`](Self::ordinary_define_own_property)
    /// but throws a `TypeError` instead of returning `false`.
    pub fn define_property_or_throw(
        &mut self,
        key: &str,
        desc: &FullPropertyDescriptor,
    ) -> StatorResult<()> {
        let success = self.ordinary_define_own_property(key, desc)?;
        if !success {
            return Err(StatorError::TypeError(format!(
                "Cannot define property '{key}'"
            )));
        }
        Ok(())
    }

    /// ECMAScript §7.3.12 *HasProperty* ( *O*, *P* ) — inherited + own.
    pub fn spec_has_property(&self, key: &str) -> bool {
        self.has_property(key)
    }

    /// ECMAScript §7.3.13 *HasOwnProperty* ( *O*, *P* ) — own only.
    pub fn spec_has_own_property(&self, key: &str) -> bool {
        self.has_own_property(key)
    }

    /// ECMAScript §7.3.2 *Get* ( *O*, *P* ) with implicit receiver = *O*.
    pub fn spec_get(&self, key: &str) -> JsValue {
        self.get_property(key)
    }

    /// ECMAScript §7.3.2 *Get* ( *O*, *P* ) with explicit *Receiver*.
    ///
    /// The receiver parameter is relevant for Proxy objects and accessor
    /// property getters.
    pub fn spec_get_with_receiver(&self, key: &str, receiver: &JsValue) -> JsValue {
        self.ordinary_get(key, receiver)
    }

    /// Searches the prototype chain (excluding `self`) for a property
    /// descriptor matching `key`.
    fn find_in_prototype_chain(&self, key: &str) -> Option<FullPropertyDescriptor> {
        let mut current = self.prototype.clone();
        while let Some(proto_ref) = current {
            let proto = proto_ref.borrow();
            if let Some(desc) = proto.ordinary_get_own_property(key) {
                return Some(desc);
            }
            current = proto.prototype.clone();
        }
        None
    }

    /// Defines an accessor property on this object.
    ///
    /// Useful for setting up getter/setter pairs without going through
    /// the full `OrdinaryDefineOwnProperty` path.
    pub fn define_accessor_property(
        &mut self,
        key: &str,
        get: JsValue,
        set: JsValue,
        enumerable: bool,
        configurable: bool,
    ) -> StatorResult<bool> {
        let desc = FullPropertyDescriptor::Accessor {
            get,
            set,
            enumerable,
            configurable,
        };
        self.ordinary_define_own_property(key, &desc)
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

    /// Returns an immutable slice of the element backing store.
    ///
    /// This provides zero-copy access to the underlying `Vec<JsValue>`,
    /// enabling optimised fast paths in array built-ins.
    pub fn elements_as_slice(&self) -> &[JsValue] {
        &self.elements
    }

    /// Returns a mutable slice of the element backing store.
    ///
    /// This provides zero-copy mutable access for in-place operations such as
    /// `Array.prototype.reverse` and `Array.prototype.sort`.
    pub fn elements_as_mut_slice(&mut self) -> &mut [JsValue] {
        &mut self.elements
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
            NamedProperties::Slow { map, .. } => {
                for prop in map.values() {
                    match prop.kind() {
                        PropertyKind::Data(v) => v.trace(tracer),
                        PropertyKind::Accessor { get, set } => {
                            get.trace(tracer);
                            set.trace(tracer);
                        }
                    }
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
        obj.set_property("c", JsValue::String("hi".to_string().into()))
            .unwrap();
        assert_eq!(obj.get_own_property("a"), Some(JsValue::Smi(1)));
        assert_eq!(obj.get_own_property("b"), Some(JsValue::Boolean(true)));
        assert_eq!(
            obj.get_own_property("c"),
            Some(JsValue::String("hi".to_string().into()))
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

    // ── Property enumeration order conformance ────────────────────────────

    #[test]
    fn test_own_keys_fast_mode_integer_sorted() {
        let mut obj = JsObject::new();
        obj.set_property("2", JsValue::Smi(2)).unwrap();
        obj.set_property("0", JsValue::Smi(0)).unwrap();
        obj.set_property("1", JsValue::Smi(1)).unwrap();
        assert!(obj.is_fast_mode());
        let keys = obj.own_property_keys();
        assert_eq!(keys, &["0", "1", "2"]);
    }

    #[test]
    fn test_own_keys_fast_mode_strings_after_integers() {
        let mut obj = JsObject::new();
        obj.set_property("b", JsValue::Smi(1)).unwrap();
        obj.set_property("1", JsValue::Smi(2)).unwrap();
        obj.set_property("a", JsValue::Smi(3)).unwrap();
        obj.set_property("0", JsValue::Smi(4)).unwrap();
        let keys = obj.own_property_keys();
        assert_eq!(keys, &["0", "1", "b", "a"]);
    }

    #[test]
    fn test_own_keys_slow_mode_integer_sorted() {
        let mut obj = JsObject::new();
        // Force slow mode by exceeding MAX_FAST_PROPERTIES
        for i in 0..=MAX_FAST_PROPERTIES {
            obj.set_property(&format!("p{i}"), JsValue::Smi(i as i32))
                .unwrap();
        }
        assert!(!obj.is_fast_mode());
        // Now add integer-indexed properties
        obj.set_property("5", JsValue::Smi(50)).unwrap();
        obj.set_property("2", JsValue::Smi(20)).unwrap();
        obj.set_property("10", JsValue::Smi(100)).unwrap();
        let keys = obj.own_property_keys();
        // Integers first in ascending order
        assert_eq!(keys[0], "2");
        assert_eq!(keys[1], "5");
        assert_eq!(keys[2], "10");
    }

    #[test]
    fn test_own_keys_slow_mode_strings_insertion_order() {
        let mut obj = JsObject::new();
        for i in 0..=MAX_FAST_PROPERTIES {
            obj.set_property(&format!("fill{i}"), JsValue::Smi(i as i32))
                .unwrap();
        }
        assert!(!obj.is_fast_mode());
        obj.set_property("z", JsValue::Smi(1)).unwrap();
        obj.set_property("a", JsValue::Smi(2)).unwrap();
        obj.set_property("m", JsValue::Smi(3)).unwrap();
        let keys = obj.own_property_keys();
        // String keys should be in insertion order relative to each other
        let z_pos = keys.iter().position(|k| k == "z").unwrap();
        let a_pos = keys.iter().position(|k| k == "a").unwrap();
        let m_pos = keys.iter().position(|k| k == "m").unwrap();
        assert!(z_pos < a_pos);
        assert!(a_pos < m_pos);
    }

    #[test]
    fn test_own_keys_element_indices_not_in_named() {
        let mut obj = JsObject::new();
        obj.set_element(0, JsValue::Smi(10));
        obj.set_element(1, JsValue::Smi(20));
        obj.set_property("a", JsValue::Smi(1)).unwrap();
        // own_property_keys only returns named properties
        let keys = obj.own_property_keys();
        assert_eq!(keys, &["a"]);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // ECMAScript abstract operations — 40+ e2e spec compliance tests
    // ══════════════════════════════════════════════════════════════════════════

    // ── OrdinaryGetOwnProperty ──────────────────────────────────────────────

    #[test]
    fn test_ordinary_get_own_property_data_shape() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(42)).unwrap();
        let desc = obj.ordinary_get_own_property("x").unwrap();
        assert!(desc.is_data());
        if let FullPropertyDescriptor::Data {
            value,
            writable,
            enumerable,
            configurable,
        } = desc
        {
            assert_eq!(value, JsValue::Smi(42));
            assert!(writable);
            assert!(enumerable);
            assert!(configurable);
        }
    }

    #[test]
    fn test_ordinary_get_own_property_missing_returns_none() {
        let obj = JsObject::new();
        assert!(obj.ordinary_get_own_property("absent").is_none());
    }

    #[test]
    fn test_ordinary_get_own_property_accessor_shape() {
        let mut obj = JsObject::new();
        obj.define_accessor_property(
            "acc",
            JsValue::Boolean(true),  // getter stand-in
            JsValue::Boolean(false), // setter stand-in
            true,
            true,
        )
        .unwrap();
        let desc = obj.ordinary_get_own_property("acc").unwrap();
        assert!(desc.is_accessor());
        if let FullPropertyDescriptor::Accessor {
            get,
            set,
            enumerable,
            configurable,
        } = desc
        {
            assert_eq!(get, JsValue::Boolean(true));
            assert_eq!(set, JsValue::Boolean(false));
            assert!(enumerable);
            assert!(configurable);
        }
    }

    #[test]
    fn test_ordinary_get_own_property_non_writable() {
        let mut obj = JsObject::new();
        obj.define_own_property("ro", JsValue::Smi(7), PropertyAttributes::ENUMERABLE)
            .unwrap();
        let desc = obj.ordinary_get_own_property("ro").unwrap();
        if let FullPropertyDescriptor::Data { writable, .. } = desc {
            assert!(!writable);
        } else {
            panic!("expected Data descriptor");
        }
    }

    // ── OrdinaryDefineOwnProperty / ValidateAndApplyPropertyDescriptor ──────

    #[test]
    fn test_ordinary_define_own_property_new_data() {
        let mut obj = JsObject::new();
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(10),
            writable: true,
            enumerable: true,
            configurable: true,
        };
        assert!(obj.ordinary_define_own_property("p", &desc).unwrap());
        assert_eq!(obj.get_property("p"), JsValue::Smi(10));
    }

    #[test]
    fn test_ordinary_define_rejects_on_non_extensible() {
        let mut obj = JsObject::new();
        obj.prevent_extensions();
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: true,
            enumerable: true,
            configurable: true,
        };
        assert!(!obj.ordinary_define_own_property("p", &desc).unwrap());
    }

    #[test]
    fn test_ordinary_define_non_configurable_rejects_configurable_change() {
        let mut obj = JsObject::new();
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: false,
            enumerable: false,
            configurable: false,
        };
        obj.ordinary_define_own_property("p", &desc).unwrap();
        let redesc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: false,
            enumerable: false,
            configurable: true,
        };
        assert!(!obj.ordinary_define_own_property("p", &redesc).unwrap());
    }

    #[test]
    fn test_ordinary_define_non_configurable_rejects_enumerable_change() {
        let mut obj = JsObject::new();
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: false,
            enumerable: false,
            configurable: false,
        };
        obj.ordinary_define_own_property("p", &desc).unwrap();
        let redesc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: false,
            enumerable: true,
            configurable: false,
        };
        assert!(!obj.ordinary_define_own_property("p", &redesc).unwrap());
    }

    #[test]
    fn test_ordinary_define_non_configurable_non_writable_rejects_value_change() {
        let mut obj = JsObject::new();
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: false,
            enumerable: false,
            configurable: false,
        };
        obj.ordinary_define_own_property("p", &desc).unwrap();
        let redesc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(2),
            writable: false,
            enumerable: false,
            configurable: false,
        };
        assert!(!obj.ordinary_define_own_property("p", &redesc).unwrap());
    }

    #[test]
    fn test_ordinary_define_non_configurable_non_writable_same_value_ok() {
        let mut obj = JsObject::new();
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: false,
            enumerable: false,
            configurable: false,
        };
        obj.ordinary_define_own_property("p", &desc).unwrap();
        // Redefining with same value is fine.
        assert!(obj.ordinary_define_own_property("p", &desc).unwrap());
    }

    #[test]
    fn test_ordinary_define_non_configurable_rejects_writable_false_to_true() {
        let mut obj = JsObject::new();
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: false,
            enumerable: false,
            configurable: false,
        };
        obj.ordinary_define_own_property("p", &desc).unwrap();
        let redesc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: true,
            enumerable: false,
            configurable: false,
        };
        assert!(!obj.ordinary_define_own_property("p", &redesc).unwrap());
    }

    #[test]
    fn test_ordinary_define_configurable_allows_writable_change() {
        let mut obj = JsObject::new();
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: true,
            enumerable: true,
            configurable: true,
        };
        obj.ordinary_define_own_property("p", &desc).unwrap();
        let redesc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: false,
            enumerable: true,
            configurable: true,
        };
        assert!(obj.ordinary_define_own_property("p", &redesc).unwrap());
    }

    #[test]
    fn test_ordinary_define_data_to_accessor_on_configurable() {
        let mut obj = JsObject::new();
        let data = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: true,
            enumerable: true,
            configurable: true,
        };
        obj.ordinary_define_own_property("p", &data).unwrap();
        // Convert to accessor.
        let acc = FullPropertyDescriptor::Accessor {
            get: JsValue::Boolean(true),
            set: JsValue::Undefined,
            enumerable: true,
            configurable: true,
        };
        assert!(obj.ordinary_define_own_property("p", &acc).unwrap());
        let desc = obj.ordinary_get_own_property("p").unwrap();
        assert!(desc.is_accessor());
    }

    #[test]
    fn test_ordinary_define_data_to_accessor_on_non_configurable_rejected() {
        let mut obj = JsObject::new();
        let data = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: true,
            enumerable: false,
            configurable: false,
        };
        obj.ordinary_define_own_property("p", &data).unwrap();
        let acc = FullPropertyDescriptor::Accessor {
            get: JsValue::Undefined,
            set: JsValue::Undefined,
            enumerable: false,
            configurable: false,
        };
        assert!(!obj.ordinary_define_own_property("p", &acc).unwrap());
    }

    #[test]
    fn test_ordinary_define_accessor_to_data_on_configurable() {
        let mut obj = JsObject::new();
        let acc = FullPropertyDescriptor::Accessor {
            get: JsValue::Boolean(true),
            set: JsValue::Undefined,
            enumerable: true,
            configurable: true,
        };
        obj.ordinary_define_own_property("p", &acc).unwrap();
        let data = FullPropertyDescriptor::Data {
            value: JsValue::Smi(99),
            writable: true,
            enumerable: true,
            configurable: true,
        };
        assert!(obj.ordinary_define_own_property("p", &data).unwrap());
        let desc = obj.ordinary_get_own_property("p").unwrap();
        assert!(desc.is_data());
    }

    #[test]
    fn test_ordinary_define_non_configurable_accessor_same_values_ok() {
        let mut obj = JsObject::new();
        let acc = FullPropertyDescriptor::Accessor {
            get: JsValue::Boolean(true),
            set: JsValue::Boolean(false),
            enumerable: false,
            configurable: false,
        };
        obj.ordinary_define_own_property("p", &acc).unwrap();
        // Redefine with same getter/setter.
        assert!(obj.ordinary_define_own_property("p", &acc).unwrap());
    }

    #[test]
    fn test_ordinary_define_non_configurable_accessor_different_getter_rejected() {
        let mut obj = JsObject::new();
        let acc = FullPropertyDescriptor::Accessor {
            get: JsValue::Boolean(true),
            set: JsValue::Undefined,
            enumerable: false,
            configurable: false,
        };
        obj.ordinary_define_own_property("p", &acc).unwrap();
        let changed = FullPropertyDescriptor::Accessor {
            get: JsValue::Boolean(false),
            set: JsValue::Undefined,
            enumerable: false,
            configurable: false,
        };
        assert!(!obj.ordinary_define_own_property("p", &changed).unwrap());
    }

    #[test]
    fn test_ordinary_define_empty_generic_on_existing_returns_true() {
        let mut obj = JsObject::new();
        obj.set_property("p", JsValue::Smi(1)).unwrap();
        let generic = FullPropertyDescriptor::Generic {
            enumerable: None,
            configurable: None,
        };
        assert!(obj.ordinary_define_own_property("p", &generic).unwrap());
    }

    // ── OrdinarySet ─────────────────────────────────────────────────────────

    #[test]
    fn test_ordinary_set_creates_new_property() {
        let mut obj = JsObject::new();
        let receiver = JsValue::Undefined;
        assert!(obj.ordinary_set("x", JsValue::Smi(42), &receiver).unwrap());
        assert_eq!(obj.get_property("x"), JsValue::Smi(42));
    }

    #[test]
    fn test_ordinary_set_updates_existing_writable_property() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        let receiver = JsValue::Undefined;
        assert!(obj.ordinary_set("x", JsValue::Smi(99), &receiver).unwrap());
        assert_eq!(obj.get_property("x"), JsValue::Smi(99));
    }

    #[test]
    fn test_ordinary_set_rejects_non_writable() {
        let mut obj = JsObject::new();
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: false,
            enumerable: true,
            configurable: true,
        };
        obj.ordinary_define_own_property("x", &desc).unwrap();
        let receiver = JsValue::Undefined;
        assert!(!obj.ordinary_set("x", JsValue::Smi(2), &receiver).unwrap());
    }

    #[test]
    fn test_ordinary_set_rejects_non_writable_in_prototype() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        {
            let mut p = proto.borrow_mut();
            let desc = FullPropertyDescriptor::Data {
                value: JsValue::Smi(1),
                writable: false,
                enumerable: true,
                configurable: true,
            };
            p.ordinary_define_own_property("x", &desc).unwrap();
        }
        let mut child = JsObject::with_prototype(Rc::clone(&proto));
        let receiver = JsValue::Undefined;
        assert!(!child.ordinary_set("x", JsValue::Smi(2), &receiver).unwrap());
    }

    #[test]
    fn test_ordinary_set_respects_accessor_setter_undefined() {
        let mut obj = JsObject::new();
        let acc = FullPropertyDescriptor::Accessor {
            get: JsValue::Boolean(true),
            set: JsValue::Undefined,
            enumerable: true,
            configurable: true,
        };
        obj.ordinary_define_own_property("x", &acc).unwrap();
        let receiver = JsValue::Undefined;
        assert!(!obj.ordinary_set("x", JsValue::Smi(2), &receiver).unwrap());
    }

    #[test]
    fn test_ordinary_set_accessor_setter_present_succeeds() {
        let mut obj = JsObject::new();
        let acc = FullPropertyDescriptor::Accessor {
            get: JsValue::Undefined,
            set: JsValue::Boolean(true), // stand-in for a setter
            enumerable: true,
            configurable: true,
        };
        obj.ordinary_define_own_property("x", &acc).unwrap();
        let receiver = JsValue::Undefined;
        assert!(obj.ordinary_set("x", JsValue::Smi(2), &receiver).unwrap());
    }

    #[test]
    fn test_ordinary_set_proto_accessor_setter_undefined_rejected() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        {
            let mut p = proto.borrow_mut();
            let acc = FullPropertyDescriptor::Accessor {
                get: JsValue::Undefined,
                set: JsValue::Undefined,
                enumerable: true,
                configurable: true,
            };
            p.ordinary_define_own_property("x", &acc).unwrap();
        }
        let mut child = JsObject::with_prototype(Rc::clone(&proto));
        let receiver = JsValue::Undefined;
        assert!(!child.ordinary_set("x", JsValue::Smi(1), &receiver).unwrap());
    }

    #[test]
    fn test_ordinary_set_non_extensible_rejects_new_property() {
        let mut obj = JsObject::new();
        obj.prevent_extensions();
        let receiver = JsValue::Undefined;
        assert!(!obj.ordinary_set("x", JsValue::Smi(1), &receiver).unwrap());
    }

    // ── OrdinaryGet ─────────────────────────────────────────────────────────

    #[test]
    fn test_ordinary_get_data_property() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(42)).unwrap();
        let receiver = JsValue::Undefined;
        assert_eq!(obj.ordinary_get("x", &receiver), JsValue::Smi(42));
    }

    #[test]
    fn test_ordinary_get_accessor_returns_getter() {
        let mut obj = JsObject::new();
        let acc = FullPropertyDescriptor::Accessor {
            get: JsValue::Boolean(true),
            set: JsValue::Undefined,
            enumerable: true,
            configurable: true,
        };
        obj.ordinary_define_own_property("x", &acc).unwrap();
        let receiver = JsValue::Undefined;
        // Returns the getter (stand-in for calling it).
        assert_eq!(obj.ordinary_get("x", &receiver), JsValue::Boolean(true));
    }

    #[test]
    fn test_ordinary_get_walks_prototype_chain() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        proto
            .borrow_mut()
            .set_property("inherited", JsValue::Smi(7))
            .unwrap();
        let child = JsObject::with_prototype(Rc::clone(&proto));
        let receiver = JsValue::Undefined;
        assert_eq!(child.ordinary_get("inherited", &receiver), JsValue::Smi(7));
    }

    #[test]
    fn test_ordinary_get_missing_returns_undefined() {
        let obj = JsObject::new();
        let receiver = JsValue::Undefined;
        assert_eq!(obj.ordinary_get("nope", &receiver), JsValue::Undefined);
    }

    #[test]
    fn test_ordinary_get_accessor_on_prototype() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        {
            let mut p = proto.borrow_mut();
            let acc = FullPropertyDescriptor::Accessor {
                get: JsValue::Smi(123),
                set: JsValue::Undefined,
                enumerable: true,
                configurable: true,
            };
            p.ordinary_define_own_property("x", &acc).unwrap();
        }
        let child = JsObject::with_prototype(Rc::clone(&proto));
        let receiver = JsValue::Undefined;
        assert_eq!(child.ordinary_get("x", &receiver), JsValue::Smi(123));
    }

    // ── OrdinaryHasProperty ─────────────────────────────────────────────────

    #[test]
    fn test_ordinary_has_property_own() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        assert!(obj.ordinary_has_property("x"));
        assert!(!obj.ordinary_has_property("y"));
    }

    #[test]
    fn test_ordinary_has_property_walks_chain() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        proto
            .borrow_mut()
            .set_property("y", JsValue::Smi(2))
            .unwrap();
        let child = JsObject::with_prototype(Rc::clone(&proto));
        assert!(child.ordinary_has_property("y"));
        assert!(!child.ordinary_has_property("z"));
    }

    // ── OrdinaryDelete ──────────────────────────────────────────────────────

    #[test]
    fn test_ordinary_delete_configurable_succeeds() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        assert!(obj.ordinary_delete("x").unwrap());
        assert!(!obj.has_own_property("x"));
    }

    #[test]
    fn test_ordinary_delete_non_configurable_returns_false() {
        let mut obj = JsObject::new();
        obj.define_own_property("nc", JsValue::Smi(1), PropertyAttributes::empty())
            .unwrap();
        assert!(!obj.ordinary_delete("nc").unwrap());
        assert!(obj.has_own_property("nc"));
    }

    #[test]
    fn test_ordinary_delete_absent_returns_true() {
        let mut obj = JsObject::new();
        assert!(obj.ordinary_delete("ghost").unwrap());
    }

    // ── CreateDataProperty ──────────────────────────────────────────────────

    #[test]
    fn test_create_data_property_attributes() {
        let mut obj = JsObject::new();
        obj.create_data_property("p", JsValue::Smi(5)).unwrap();
        let desc = obj.ordinary_get_own_property("p").unwrap();
        if let FullPropertyDescriptor::Data {
            value,
            writable,
            enumerable,
            configurable,
        } = desc
        {
            assert_eq!(value, JsValue::Smi(5));
            assert!(writable);
            assert!(enumerable);
            assert!(configurable);
        } else {
            panic!("expected Data descriptor");
        }
    }

    #[test]
    fn test_create_data_property_non_extensible_fails() {
        let mut obj = JsObject::new();
        obj.prevent_extensions();
        assert!(!obj.create_data_property("p", JsValue::Smi(1)).unwrap());
    }

    // ── CreateMethodProperty ────────────────────────────────────────────────

    #[test]
    fn test_create_method_property_attributes() {
        let mut obj = JsObject::new();
        obj.create_method_property("m", JsValue::Smi(42)).unwrap();
        let desc = obj.ordinary_get_own_property("m").unwrap();
        if let FullPropertyDescriptor::Data {
            writable,
            enumerable,
            configurable,
            ..
        } = desc
        {
            assert!(writable);
            assert!(!enumerable, "method properties are NOT enumerable");
            assert!(configurable);
        } else {
            panic!("expected Data descriptor");
        }
    }

    // ── DefinePropertyOrThrow vs CreateDataProperty ─────────────────────────

    #[test]
    fn test_define_property_or_throw_success() {
        let mut obj = JsObject::new();
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: false,
            enumerable: false,
            configurable: false,
        };
        obj.define_property_or_throw("p", &desc).unwrap();
        assert_eq!(obj.get_property("p"), JsValue::Smi(1));
    }

    #[test]
    fn test_define_property_or_throw_fails_on_non_extensible() {
        let mut obj = JsObject::new();
        obj.prevent_extensions();
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: true,
            enumerable: true,
            configurable: true,
        };
        let err = obj.define_property_or_throw("p", &desc).unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    #[test]
    fn test_define_property_or_throw_default_attrs_differ_from_create_data() {
        let mut obj = JsObject::new();
        // DefinePropertyOrThrow with restrictive attrs.
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: false,
            enumerable: false,
            configurable: false,
        };
        obj.define_property_or_throw("strict", &desc).unwrap();
        // CreateDataProperty with full attrs.
        obj.create_data_property("loose", JsValue::Smi(2)).unwrap();

        let strict = obj.ordinary_get_own_property("strict").unwrap();
        let loose = obj.ordinary_get_own_property("loose").unwrap();
        assert_ne!(strict.to_attributes(), loose.to_attributes());
    }

    // ── HasProperty vs HasOwnProperty ───────────────────────────────────────

    #[test]
    fn test_has_property_vs_has_own_property() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        proto
            .borrow_mut()
            .set_property("inherited", JsValue::Smi(1))
            .unwrap();
        let mut child = JsObject::with_prototype(Rc::clone(&proto));
        child.set_property("own", JsValue::Smi(2)).unwrap();

        // HasProperty sees both.
        assert!(child.spec_has_property("own"));
        assert!(child.spec_has_property("inherited"));
        // HasOwnProperty sees only own.
        assert!(child.spec_has_own_property("own"));
        assert!(!child.spec_has_own_property("inherited"));
    }

    #[test]
    fn test_has_property_multi_level_chain() {
        let gp = Rc::new(RefCell::new(JsObject::new()));
        gp.borrow_mut()
            .set_property("deep", JsValue::Smi(1))
            .unwrap();
        let parent = Rc::new(RefCell::new(JsObject::with_prototype(Rc::clone(&gp))));
        let child = JsObject::with_prototype(Rc::clone(&parent));
        assert!(child.spec_has_property("deep"));
        assert!(!child.spec_has_own_property("deep"));
    }

    // ── Get with Receiver ───────────────────────────────────────────────────

    #[test]
    fn test_spec_get_implicit_receiver() {
        let mut obj = JsObject::new();
        obj.set_property("v", JsValue::Smi(100)).unwrap();
        assert_eq!(obj.spec_get("v"), JsValue::Smi(100));
    }

    #[test]
    fn test_spec_get_with_receiver_data() {
        let mut obj = JsObject::new();
        obj.set_property("v", JsValue::Smi(100)).unwrap();
        let receiver = JsValue::Smi(0); // arbitrary receiver
        assert_eq!(
            obj.spec_get_with_receiver("v", &receiver),
            JsValue::Smi(100)
        );
    }

    #[test]
    fn test_spec_get_with_receiver_accessor() {
        let mut obj = JsObject::new();
        let getter = JsValue::String("getter_fn".into());
        obj.define_accessor_property("x", getter.clone(), JsValue::Undefined, true, true)
            .unwrap();
        let receiver = JsValue::Smi(0);
        // Returns the getter value.
        assert_eq!(obj.spec_get_with_receiver("x", &receiver), getter);
    }

    #[test]
    fn test_spec_get_with_receiver_prototype_accessor() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        {
            let mut p = proto.borrow_mut();
            p.define_accessor_property("x", JsValue::Smi(999), JsValue::Undefined, true, true)
                .unwrap();
        }
        let child = JsObject::with_prototype(Rc::clone(&proto));
        let receiver = JsValue::Smi(0);
        assert_eq!(
            child.spec_get_with_receiver("x", &receiver),
            JsValue::Smi(999)
        );
    }

    // ── Accessor property round-trip ────────────────────────────────────────

    #[test]
    fn test_accessor_property_survives_define_and_read() {
        let mut obj = JsObject::new();
        obj.define_accessor_property(
            "acc",
            JsValue::Smi(10), // getter stand-in
            JsValue::Smi(20), // setter stand-in
            false,
            true,
        )
        .unwrap();
        let desc = obj.ordinary_get_own_property("acc").unwrap();
        assert!(desc.is_accessor());
        assert_eq!(desc.enumerable(), Some(false));
        assert_eq!(desc.configurable(), Some(true));
    }

    // ── Narrowing writable via OrdinaryDefineOwnProperty ────────────────────

    #[test]
    fn test_narrowing_writable_true_to_false_on_non_configurable() {
        let mut obj = JsObject::new();
        // non-configurable, writable
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: true,
            enumerable: false,
            configurable: false,
        };
        obj.ordinary_define_own_property("p", &desc).unwrap();
        // narrow writable true→false
        let narrow = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: false,
            enumerable: false,
            configurable: false,
        };
        assert!(obj.ordinary_define_own_property("p", &narrow).unwrap());
        let final_desc = obj.ordinary_get_own_property("p").unwrap();
        if let FullPropertyDescriptor::Data { writable, .. } = final_desc {
            assert!(!writable);
        }
    }

    // ── Prototype chain OrdinarySet with setter propagation ─────────────────

    #[test]
    fn test_ordinary_set_proto_accessor_with_setter_succeeds() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        {
            let mut p = proto.borrow_mut();
            let acc = FullPropertyDescriptor::Accessor {
                get: JsValue::Undefined,
                set: JsValue::Boolean(true), // non-undefined setter
                enumerable: true,
                configurable: true,
            };
            p.ordinary_define_own_property("x", &acc).unwrap();
        }
        let mut child = JsObject::with_prototype(Rc::clone(&proto));
        let receiver = JsValue::Undefined;
        assert!(child.ordinary_set("x", JsValue::Smi(1), &receiver).unwrap());
    }

    // ── CreateDataProperty overwrites existing ──────────────────────────────

    #[test]
    fn test_create_data_property_overwrites_accessor() {
        let mut obj = JsObject::new();
        obj.define_accessor_property(
            "x",
            JsValue::Boolean(true),
            JsValue::Boolean(false),
            true,
            true,
        )
        .unwrap();
        assert!(obj.ordinary_get_own_property("x").unwrap().is_accessor());
        obj.create_data_property("x", JsValue::Smi(42)).unwrap();
        let desc = obj.ordinary_get_own_property("x").unwrap();
        assert!(desc.is_data());
        if let FullPropertyDescriptor::Data { value, .. } = desc {
            assert_eq!(value, JsValue::Smi(42));
        }
    }

    // ── Generic descriptor changes only specified attributes ────────────────

    #[test]
    fn test_generic_descriptor_changes_only_specified_attrs() {
        let mut obj = JsObject::new();
        let desc = FullPropertyDescriptor::Data {
            value: JsValue::Smi(1),
            writable: true,
            enumerable: false,
            configurable: true,
        };
        obj.ordinary_define_own_property("p", &desc).unwrap();
        // Generic: only flip enumerable.
        let generic = FullPropertyDescriptor::Generic {
            enumerable: Some(true),
            configurable: None,
        };
        assert!(obj.ordinary_define_own_property("p", &generic).unwrap());
        let final_desc = obj.ordinary_get_own_property("p").unwrap();
        assert_eq!(final_desc.enumerable(), Some(true));
        assert_eq!(final_desc.configurable(), Some(true));
    }
}
