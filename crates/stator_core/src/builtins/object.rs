//! ECMAScript §20.1 `Object` built-in static methods.
//!
//! Every function in this module is a direct Rust equivalent of a static
//! property of the JavaScript `Object` constructor.  They operate on
//! [`JsObject`] and [`JsValue`] values and have no side-effects beyond the
//! objects passed in.
//!
//! # Naming convention
//!
//! Each function is prefixed `object_` to avoid ambiguity with similarly-named
//! standard-library items (e.g. `object_keys` vs `Object.keys`).
//!
//! # References
//!
//! * ECMAScript 2025 Language Specification §20.1 — *The Object Constructor*

use std::cell::RefCell;
use std::rc::Rc;

use crate::builtins::symbol::{is_symbol_property_key, property_key_to_symbol};
use crate::error::{StatorError, StatorResult};
use crate::objects::js_object::JsObject;
use crate::objects::map::PropertyAttributes;
use crate::objects::property_descriptor::FullPropertyDescriptor;
use crate::objects::value::JsValue;

/// Returns `true` if `key` is an internal accessor-storage key
/// (`__get_<name>__` or `__set_<name>__`).
fn is_internal_accessor_key(key: &str) -> bool {
    (key.starts_with("__get_") || key.starts_with("__set_")) && key.ends_with("__")
}

// ── Object.create ─────────────────────────────────────────────────────────────

/// ECMAScript §20.1.2.2 `Object.create(proto)`.
///
/// Creates a new ordinary object whose `[[Prototype]]` internal slot is set to
/// `proto`.  Pass `None` to create an object with a `null` prototype (i.e.
/// `Object.create(null)`).
///
/// # Examples
///
/// ```
/// use stator_core::builtins::object::object_create;
///
/// let child = object_create(None);
/// assert!(child.prototype().is_none());
/// ```
pub fn object_create(proto: Option<Rc<RefCell<JsObject>>>) -> JsObject {
    match proto {
        Some(p) => JsObject::with_prototype(p),
        None => JsObject::new(),
    }
}

/// ECMAScript §20.1.2.2 `Object.create(proto, propertiesObject)`.
///
/// Creates a new ordinary object with the given prototype, then defines
/// properties on it using the same semantics as
/// [`object_define_properties`] (i.e. the second argument to `Object.create`).
///
/// If `properties` is `None`, behaves identically to [`object_create`].
pub fn object_create_with_properties(
    proto: Option<Rc<RefCell<JsObject>>>,
    properties: Option<&JsValue>,
) -> StatorResult<JsObject> {
    let mut obj = object_create(proto);
    if let Some(props) = properties {
        object_define_properties(&mut obj, props)?;
    }
    Ok(obj)
}

// ── Object.assign ─────────────────────────────────────────────────────────────

/// ECMAScript §20.1.2.1 `Object.assign(target, ...sources)`.
///
/// Copies every **enumerable own** property from each source object into
/// `target`, in the order the sources appear in `sources`.  Within a single
/// source, properties are copied in the order returned by
/// [`JsObject::own_property_keys`].
///
/// Returns [`StatorError::TypeError`] if writing to a read-only property on
/// `target` fails.
pub fn object_assign(target: &mut JsObject, sources: &[&JsObject]) -> StatorResult<()> {
    for src in sources {
        for key in src.own_property_keys() {
            if is_internal_accessor_key(&key) {
                continue;
            }
            // Only copy enumerable own properties (string and symbol keys).
            if let Some((value, attrs)) = src.get_own_property_descriptor(&key)
                && attrs.contains(PropertyAttributes::ENUMERABLE)
            {
                target.set_property(&key, value)?;
            }
        }
    }
    Ok(())
}

// ── Object.keys / values / entries ───────────────────────────────────────────

/// ECMAScript §20.1.2.17 `Object.keys(obj)`.
///
/// Returns the names of all own **enumerable** string-keyed properties of
/// `obj`, in the same order as [`JsObject::own_property_keys`].
pub fn object_keys(obj: &JsObject) -> Vec<String> {
    obj.own_property_keys()
        .into_iter()
        .filter(|k| {
            !is_internal_accessor_key(k)
                && !is_symbol_property_key(k)
                && obj
                    .get_own_property_descriptor(k)
                    .map(|(_, a)| a.contains(PropertyAttributes::ENUMERABLE))
                    .unwrap_or(false)
        })
        .collect()
}

/// ECMAScript §20.1.2.22 `Object.values(obj)`.
///
/// Returns the values of all own **enumerable** string-keyed properties of
/// `obj`, in the same order as [`object_keys`].
pub fn object_values(obj: &JsObject) -> Vec<JsValue> {
    object_keys(obj)
        .into_iter()
        .filter_map(|k| obj.get_own_property(k.as_str()))
        .collect()
}

/// ECMAScript §20.1.2.5 `Object.entries(obj)`.
///
/// Returns `(key, value)` pairs for all own **enumerable** string-keyed
/// properties of `obj`, in the same order as [`object_keys`].
pub fn object_entries(obj: &JsObject) -> Vec<(String, JsValue)> {
    object_keys(obj)
        .into_iter()
        .filter_map(|k| obj.get_own_property(k.as_str()).map(|v| (k, v)))
        .collect()
}

// ── Object.defineProperty ────────────────────────────────────────────────────

/// ECMAScript §20.1.2.4 `Object.defineProperty(obj, key, descriptor)`.
///
/// Defines or redefines an own property with explicit attribute flags.
/// Delegates to [`JsObject::define_own_property`], which enforces all
/// ECMAScript invariants for non-configurable properties.
pub fn object_define_property(
    obj: &mut JsObject,
    key: &str,
    value: JsValue,
    attributes: PropertyAttributes,
) -> StatorResult<()> {
    obj.define_own_property(key, value, attributes)
}

// ── Object.getOwnPropertyDescriptor ──────────────────────────────────────────

/// ECMAScript §20.1.2.8 `Object.getOwnPropertyDescriptor(obj, key)`.
///
/// Returns `Some((value, attributes))` if `obj` has an own property named
/// `key`, or `None` otherwise.
pub fn object_get_own_property_descriptor(
    obj: &JsObject,
    key: &str,
) -> Option<(JsValue, PropertyAttributes)> {
    obj.get_own_property_descriptor(key)
}

// ── Object.getPrototypeOf / setPrototypeOf ────────────────────────────────────

/// ECMAScript §20.1.2.11 `Object.getPrototypeOf(obj)`.
///
/// Returns a clone of the reference-counted prototype pointer, or `None` if
/// the object has a `null` prototype.
pub fn object_get_prototype_of(obj: &JsObject) -> Option<Rc<RefCell<JsObject>>> {
    obj.prototype().cloned()
}

/// ECMAScript §20.1.2.21 `Object.setPrototypeOf(obj, proto)`.
///
/// Sets the `[[Prototype]]` of `obj` to `proto` (or `null` when `None`).
///
/// Returns [`StatorError::TypeError`] if `obj` is non-extensible and the new
/// prototype differs from the current one.
pub fn object_set_prototype_of(
    obj: &mut JsObject,
    proto: Option<Rc<RefCell<JsObject>>>,
) -> StatorResult<()> {
    // ECMAScript §10.1.2: If obj is non-extensible, the prototype may not
    // be changed unless it is already the same object (pointer equality).
    if !obj.is_extensible() {
        let same = match (obj.prototype(), &proto) {
            (Some(cur), Some(new)) => Rc::ptr_eq(cur, new),
            (None, None) => true,
            _ => false,
        };
        if !same {
            return Err(StatorError::TypeError(
                "Cannot set prototype of a non-extensible object".to_string(),
            ));
        }
    }
    obj.set_prototype(proto);
    Ok(())
}

// ── Object.freeze / seal / isFrozen ──────────────────────────────────────────

/// ECMAScript §20.1.2.6 `Object.freeze(obj)`.
///
/// Makes all own properties non-writable and non-configurable, then marks the
/// object as non-extensible.  After this call the object is *frozen*:
/// [`object_is_frozen`] will return `true`.
pub fn object_freeze(obj: &mut JsObject) -> StatorResult<()> {
    // Collect keys first to avoid borrow issues while mutating.
    let keys: Vec<String> = obj.own_property_keys();
    for key in keys {
        if let Some((value, attrs)) = obj.get_own_property_descriptor(&key) {
            // Remove WRITABLE and CONFIGURABLE; preserve ENUMERABLE.
            let new_attrs =
                attrs & !(PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE);
            // If the property is already non-configurable we may not be able
            // to strengthen the constraint via define_own_property, but
            // freeze only ever *removes* flags so validation always passes.
            obj.define_own_property(&key, value, new_attrs)?;
        }
    }
    obj.prevent_extensions();
    Ok(())
}

/// ECMAScript §20.1.2.20 `Object.seal(obj)`.
///
/// Makes all own properties non-configurable and marks the object as
/// non-extensible.  Properties retain their current `[[Writable]]` flag.
pub fn object_seal(obj: &mut JsObject) -> StatorResult<()> {
    let keys: Vec<String> = obj.own_property_keys();
    for key in keys {
        if let Some((value, attrs)) = obj.get_own_property_descriptor(&key) {
            let new_attrs = attrs & !PropertyAttributes::CONFIGURABLE;
            obj.define_own_property(&key, value, new_attrs)?;
        }
    }
    obj.prevent_extensions();
    Ok(())
}

/// ECMAScript §20.1.2.12 `Object.isFrozen(obj)`.
///
/// Returns `true` if and only if the object is non-extensible and every own
/// property is both non-writable and non-configurable.
pub fn object_is_frozen(obj: &JsObject) -> bool {
    if obj.is_extensible() {
        return false;
    }
    for key in obj.own_property_keys() {
        if is_internal_accessor_key(&key) {
            continue;
        }
        if let Some((_, attrs)) = obj.get_own_property_descriptor(&key)
            && (attrs.contains(PropertyAttributes::WRITABLE)
                || attrs.contains(PropertyAttributes::CONFIGURABLE))
        {
            return false;
        }
    }
    true
}
/// ECMAScript §20.1.2.16 `Object.isSealed(obj)`.
///
/// Returns `true` if and only if the object is non-extensible and every own
/// property is non-configurable.
pub fn object_is_sealed(obj: &JsObject) -> bool {
    if obj.is_extensible() {
        return false;
    }
    for key in obj.own_property_keys() {
        if is_internal_accessor_key(&key) {
            continue;
        }
        if let Some((_, attrs)) = obj.get_own_property_descriptor(&key)
            && attrs.contains(PropertyAttributes::CONFIGURABLE)
        {
            return false;
        }
    }
    true
}

/// ECMAScript §20.1.2.9 `Object.getOwnPropertyNames(obj)`.
///
/// Returns an array of all own property names (including non-enumerable ones).
pub fn object_get_own_property_names(obj: &JsObject) -> Vec<String> {
    obj.own_property_keys()
        .into_iter()
        .filter(|k| !is_internal_accessor_key(k) && !is_symbol_property_key(k))
        .collect()
}
// ── Object.is ────────────────────────────────────────────────────────────────

/// ECMAScript §20.1.2.15 `Object.is(x, y)` — **SameValue** (§7.2.11).
///
/// Unlike `===`, `Object.is` distinguishes `+0` from `-0` and considers
/// `NaN` equal to `NaN`.
///
/// | `x` | `y` | result |
/// |---|---|---|
/// | `NaN` | `NaN` | `true` |
/// | `+0` | `-0` | `false` |
/// | `-0` | `+0` | `false` |
/// | anything else | same value | `true` |
pub fn object_is(x: &JsValue, y: &JsValue) -> bool {
    match (x, y) {
        // Both undefined / null → same.
        (JsValue::Undefined, JsValue::Undefined) => true,
        (JsValue::Null, JsValue::Null) => true,
        // Booleans.
        (JsValue::Boolean(a), JsValue::Boolean(b)) => a == b,
        // Strings.
        (JsValue::String(a), JsValue::String(b)) => a == b,
        // Symbols: same descriptor → same.
        (JsValue::Symbol(a), JsValue::Symbol(b)) => a == b,
        // BigInt.
        (JsValue::BigInt(a), JsValue::BigInt(b)) => a == b,
        // Smi: straightforward integer equality.
        (JsValue::Smi(a), JsValue::Smi(b)) => a == b,
        // Cross-type numeric: Smi vs HeapNumber.
        (JsValue::Smi(a), JsValue::HeapNumber(b)) => {
            let af = f64::from(*a);
            // +0 / -0 check: Smi 0 is always +0.
            if af == 0.0 && b.is_sign_negative() {
                return false;
            }
            af == *b
        }
        (JsValue::HeapNumber(a), JsValue::Smi(b)) => {
            let bf = f64::from(*b);
            if bf == 0.0 && a.is_sign_negative() {
                return false;
            }
            *a == bf
        }
        // HeapNumber: SameValue for floats — NaN == NaN, +0 ≠ -0.
        (JsValue::HeapNumber(a), JsValue::HeapNumber(b)) => {
            if a.is_nan() && b.is_nan() {
                return true;
            }
            // Distinguish +0 from -0 via bit patterns.
            a.to_bits() == b.to_bits()
        }
        // Object: pointer equality.
        (JsValue::Object(a), JsValue::Object(b)) => std::ptr::eq(*a, *b),
        // All other combinations are not the same value.
        _ => false,
    }
}

// ── Object.fromEntries ───────────────────────────────────────────────────────

/// ECMAScript §20.1.2.7 `Object.fromEntries(iterable)`.
///
/// Creates a new object from an iterable of `(key, value)` pairs.  The
/// resulting properties are created with default attributes
/// (`WRITABLE | ENUMERABLE | CONFIGURABLE`).
pub fn object_from_entries(
    entries: impl IntoIterator<Item = (String, JsValue)>,
) -> StatorResult<JsObject> {
    let mut obj = JsObject::new();
    for (key, value) in entries {
        obj.set_property(&key, value)?;
    }
    Ok(obj)
}

// ── Object.hasOwn ────────────────────────────────────────────────────────────

/// ECMAScript §20.1.2.13 `Object.hasOwn(obj, key)`.
///
/// Returns `true` if `obj` has an own property named `key`, regardless of
/// its enumerability or other attributes.  This is the static-method
/// replacement for `Object.prototype.hasOwnProperty`.
pub fn object_has_own(obj: &JsObject, key: &str) -> bool {
    obj.has_own_property(key)
}

// ── Object.preventExtensions ─────────────────────────────────────────────────

/// ECMAScript §20.1.2.18 `Object.preventExtensions(obj)`.
///
/// Marks `obj` as non-extensible so that no new own properties may be added.
/// Existing properties are unaffected.
pub fn object_prevent_extensions(obj: &mut JsObject) {
    obj.prevent_extensions();
}

// ── Object.isExtensible ──────────────────────────────────────────────────────

/// ECMAScript §20.1.2.14 `Object.isExtensible(obj)`.
///
/// Returns `true` if new properties may be added to `obj`.
pub fn object_is_extensible(obj: &JsObject) -> bool {
    obj.is_extensible()
}

// ── Object.getOwnPropertyDescriptors ─────────────────────────────────────────

/// ECMAScript §20.1.2.10 `Object.getOwnPropertyDescriptors(obj)`.
///
/// Returns a `Vec` of `(key, value, attributes)` tuples for every own
/// property of `obj`, regardless of enumerability.
pub fn object_get_own_property_descriptors(
    obj: &JsObject,
) -> Vec<(String, JsValue, PropertyAttributes)> {
    obj.own_property_keys()
        .into_iter()
        .filter(|k| !is_internal_accessor_key(k) && !is_symbol_property_key(k))
        .filter_map(|k| obj.get_own_property_descriptor(&k).map(|(v, a)| (k, v, a)))
        .collect()
}

/// ECMAScript §20.1.2.10 `Object.getOwnPropertyDescriptors(obj)` — returns a
/// descriptor *object*.
///
/// Returns a [`JsValue::PlainObject`] whose own properties are property names
/// (including symbol keys) mapped to their corresponding descriptor objects
/// (as returned by [`object_get_own_property_descriptor_as_object`]).
pub fn object_get_own_property_descriptors_as_object(obj: &JsObject) -> JsValue {
    use crate::objects::property_map::PropertyMap;
    let mut map = PropertyMap::new();
    for key in obj.own_property_keys() {
        if is_internal_accessor_key(&key) {
            continue;
        }
        if let Some(desc_obj) = object_get_own_property_descriptor_as_object(obj, &key) {
            map.insert(key, desc_obj);
        }
    }
    JsValue::PlainObject(Rc::new(RefCell::new(map)))
}

// ── Object.getOwnPropertySymbols ─────────────────────────────────────────────

/// ECMAScript §20.1.2.11 `Object.getOwnPropertySymbols(obj)`.
///
/// Returns an array of all own symbol-keyed properties of `obj`.
pub fn object_get_own_property_symbols(obj: &JsObject) -> Vec<JsValue> {
    obj.own_property_keys()
        .into_iter()
        .filter_map(|k| property_key_to_symbol(&k).map(JsValue::Symbol))
        .collect()
}

// ── Object.defineProperty (descriptor object) ────────────────────────────────

/// ECMAScript §20.1.2.4 `Object.defineProperty(obj, key, descriptorObj)`.
///
/// Accepts a [`JsValue::PlainObject`] descriptor (as JS code would pass) and
/// converts it to a [`FullPropertyDescriptor`] before applying it.
///
/// Returns [`StatorError::TypeError`] if the descriptor mixes data and
/// accessor fields, or if the redefinition violates non-configurable
/// invariants.
pub fn object_define_property_from_descriptor(
    obj: &mut JsObject,
    key: &str,
    descriptor: &JsValue,
) -> StatorResult<()> {
    let desc = FullPropertyDescriptor::from_object(descriptor)?;

    let getter_key = format!("__get_{key}__");
    let setter_key = format!("__set_{key}__");
    let is_current_accessor =
        obj.has_own_property(&getter_key) || obj.has_own_property(&setter_key);

    let attrs = if let Some((_, current_attrs)) = obj.get_own_property_descriptor(key) {
        let is_configurable = current_attrs.contains(PropertyAttributes::CONFIGURABLE);
        if !is_configurable {
            // Non-configurable: cannot switch between data ↔ accessor.
            if is_current_accessor && desc.is_data() {
                return Err(StatorError::TypeError(format!(
                    "Cannot redefine property '{key}': \
                     cannot convert accessor to data on a non-configurable property"
                )));
            }
            if !is_current_accessor && desc.is_accessor() {
                return Err(StatorError::TypeError(format!(
                    "Cannot redefine property '{key}': \
                     cannot convert data to accessor on a non-configurable property"
                )));
            }
            // Non-configurable accessor → accessor: getter/setter must not change.
            if let FullPropertyDescriptor::Accessor { get, set, .. } = &desc
                && is_current_accessor
            {
                let cur_get = obj
                    .get_own_property(&getter_key)
                    .unwrap_or(JsValue::Undefined);
                if *get != cur_get {
                    return Err(StatorError::TypeError(format!(
                        "Cannot redefine property '{key}': \
                         cannot change getter of a non-configurable accessor"
                    )));
                }
                let cur_set = obj
                    .get_own_property(&setter_key)
                    .unwrap_or(JsValue::Undefined);
                if *set != cur_set {
                    return Err(StatorError::TypeError(format!(
                        "Cannot redefine property '{key}': \
                         cannot change setter of a non-configurable accessor"
                    )));
                }
            }
        }
        desc.validate_against(key, current_attrs)?
    } else {
        desc.to_attributes()
    };

    match &desc {
        FullPropertyDescriptor::Data { value, .. } => {
            // Transition from accessor → data: remove getter/setter entries.
            if is_current_accessor {
                let _ = obj.delete_own_property(&getter_key);
                let _ = obj.delete_own_property(&setter_key);
            }
            obj.define_own_property(key, value.clone(), attrs)
        }
        FullPropertyDescriptor::Accessor { get, set, .. } => {
            let internal_attrs = PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE;
            // Update or create the internal getter/setter entries.
            if obj.has_own_property(&getter_key) {
                obj.set_property(&getter_key, get.clone())?;
            } else {
                obj.define_own_property(&getter_key, get.clone(), internal_attrs)?;
            }
            if obj.has_own_property(&setter_key) {
                obj.set_property(&setter_key, set.clone())?;
            } else {
                obj.define_own_property(&setter_key, set.clone(), internal_attrs)?;
            }
            // Store the property key with accessor attributes (no WRITABLE).
            obj.define_own_property(key, JsValue::Undefined, attrs)
        }
        FullPropertyDescriptor::Generic { .. } => {
            let value = obj.get_own_property(key).unwrap_or(JsValue::Undefined);
            obj.define_own_property(key, value, attrs)
        }
    }
}

// ── Object.defineProperties ──────────────────────────────────────────────────

/// ECMAScript §20.1.2.3 `Object.defineProperties(obj, props)`.
///
/// For each own enumerable property of `props`, extracts a
/// [`FullPropertyDescriptor`] and applies it to `obj` via
/// [`object_define_property_from_descriptor`].
pub fn object_define_properties(obj: &mut JsObject, props: &JsValue) -> StatorResult<()> {
    let entries: Vec<(String, JsValue)> = match props {
        JsValue::PlainObject(map) => map
            .borrow()
            .enumerable_iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
        _ => {
            return Err(StatorError::TypeError(
                "Property descriptor must be an object".to_string(),
            ));
        }
    };
    for (key, desc_val) in &entries {
        object_define_property_from_descriptor(obj, key, desc_val)?;
    }
    Ok(())
}

// ── Object.getOwnPropertyDescriptor (as descriptor object) ──────────────────

/// ECMAScript §20.1.2.8 `Object.getOwnPropertyDescriptor(obj, key)` — returns
/// a descriptor *object*.
///
/// If the property uses the `__get_<key>__`/`__set_<key>__` accessor
/// convention, returns an accessor descriptor with `get`, `set`,
/// `enumerable`, and `configurable` keys.  Otherwise returns a data
/// descriptor with `value`, `writable`, `enumerable`, and `configurable`.
pub fn object_get_own_property_descriptor_as_object(obj: &JsObject, key: &str) -> Option<JsValue> {
    obj.get_own_property_descriptor(key).map(|(value, attrs)| {
        let getter_key = format!("__get_{key}__");
        let setter_key = format!("__set_{key}__");
        if obj.has_own_property(&getter_key) || obj.has_own_property(&setter_key) {
            let get = obj
                .get_own_property(&getter_key)
                .unwrap_or(JsValue::Undefined);
            let set = obj
                .get_own_property(&setter_key)
                .unwrap_or(JsValue::Undefined);
            let desc = FullPropertyDescriptor::Accessor {
                get,
                set,
                enumerable: attrs.contains(PropertyAttributes::ENUMERABLE),
                configurable: attrs.contains(PropertyAttributes::CONFIGURABLE),
            };
            desc.to_object()
        } else {
            let desc = FullPropertyDescriptor::Data {
                value,
                writable: attrs.contains(PropertyAttributes::WRITABLE),
                enumerable: attrs.contains(PropertyAttributes::ENUMERABLE),
                configurable: attrs.contains(PropertyAttributes::CONFIGURABLE),
            };
            desc.to_object()
        }
    })
}

// ── Object.groupBy ────────────────────────────────────────────────────────────

/// ECMAScript §22.1.2.5 `Object.groupBy(items, callbackFn)`.
///
/// Groups the elements of `items` into a null-prototype object by the string
/// keys returned by `key_fn(element, index)`.  Each group is a [`Vec<JsValue>`].
///
/// The result is an [`IndexMap`] preserving first-seen key order, which mirrors
/// the insertion-order semantics of a null-prototype object.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::object::object_group_by;
/// use stator_core::objects::value::JsValue;
///
/// let items = vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3), JsValue::Smi(4)];
/// let groups = object_group_by(&items, |v, _idx| {
///     if let JsValue::Smi(n) = v { if n % 2 == 0 { "even" } else { "odd" } }
///     else { "other" }
///     .to_string()
/// });
/// assert_eq!(groups.len(), 2);
/// assert_eq!(groups["odd"].len(), 2);
/// assert_eq!(groups["even"].len(), 2);
/// ```
pub fn object_group_by(
    items: &[JsValue],
    mut key_fn: impl FnMut(&JsValue, usize) -> String,
) -> std::collections::HashMap<String, Vec<JsValue>> {
    let mut groups: std::collections::HashMap<String, Vec<JsValue>> =
        std::collections::HashMap::new();
    for (i, item) in items.iter().enumerate() {
        let key = key_fn(item, i);
        groups.entry(key).or_default().push(item.clone());
    }
    groups
}

// ── Object.prototype.hasOwnProperty ──────────────────────────────────────────

/// ECMAScript §20.1.3.2 `Object.prototype.hasOwnProperty(V)`.
///
/// Works on any [`JsValue`] by applying **ToObject** for primitives (auto-boxing)
/// and **ToPropertyKey** for the argument.  Returns `true` if the resulting
/// object has an own property with the given key.
///
/// For primitives (numbers, booleans, strings, symbols, bigint), the value is
/// first wrapped into a temporary object wrapper via [`JsValue::to_object`].
/// Because the wrapper is temporary, only intrinsic properties (e.g. string
/// indices for strings) may appear as own properties.
///
/// Returns [`StatorError::TypeError`] if `this_value` is `null` or `undefined`.
pub fn object_prototype_has_own_property(
    this_value: &JsValue,
    key: &JsValue,
) -> StatorResult<bool> {
    let prop_key = key.to_property_key()?;
    match this_value {
        JsValue::PlainObject(map) => Ok(map.borrow().contains_key(&prop_key)),
        _ => {
            let obj = this_value.to_object()?;
            if let JsValue::PlainObject(map) = &obj {
                Ok(map.borrow().contains_key(&prop_key))
            } else {
                Ok(false)
            }
        }
    }
}

// ── Object.prototype.isPrototypeOf ──────────────────────────────────────────

/// ECMAScript §20.1.3.4 `Object.prototype.isPrototypeOf(V)`.
///
/// Returns `true` if `this_obj` appears anywhere in the prototype chain of
/// the [`JsObject`] `target`.  If `target` is not an object, returns `false`.
///
/// This performs a simple walk up the `[[Prototype]]` chain of `target`,
/// comparing each link to `this_obj` by pointer identity.
pub fn object_prototype_is_prototype_of(
    this_obj: &Rc<RefCell<JsObject>>,
    target: &JsObject,
) -> bool {
    let mut current = target.prototype().cloned();
    while let Some(proto) = current {
        if Rc::ptr_eq(&proto, this_obj) {
            return true;
        }
        current = proto.borrow().prototype().cloned();
    }
    false
}

// ── Object.prototype.propertyIsEnumerable ────────────────────────────────────

/// ECMAScript §20.1.3.5 `Object.prototype.propertyIsEnumerable(V)`.
///
/// Returns `true` if the object has an **own** property named `key` that is
/// also **enumerable**.  Inherited properties always return `false`.
pub fn object_prototype_property_is_enumerable(obj: &JsObject, key: &str) -> bool {
    obj.get_own_property_descriptor(key)
        .map(|(_, attrs)| attrs.contains(PropertyAttributes::ENUMERABLE))
        .unwrap_or(false)
}

/// [`object_prototype_property_is_enumerable`] variant for [`PropertyMap`]-backed
/// plain objects.
pub fn object_prototype_property_is_enumerable_plain(
    map: &crate::objects::property_map::PropertyMap,
    key: &str,
) -> bool {
    map.get_with_attrs(key)
        .map(|(_, attrs)| attrs.contains(PropertyAttributes::ENUMERABLE))
        .unwrap_or(false)
}

// ── Object.prototype.valueOf ────────────────────────────────────────────────

/// ECMAScript §20.1.3.7 `Object.prototype.valueOf()`.
///
/// Returns the object itself.  For primitive values, applies **ToObject**
/// first to return the boxed wrapper.
pub fn object_prototype_value_of(this_value: &JsValue) -> StatorResult<JsValue> {
    this_value.to_object()
}

// ── Object.prototype.toLocaleString ─────────────────────────────────────────

/// ECMAScript §20.1.3.6 `Object.prototype.toLocaleString()`.
///
/// Calls `this.toString()`.  For a plain [`JsObject`], this delegates to
/// [`JsValue::to_js_string`].
pub fn object_prototype_to_locale_string(this_value: &JsValue) -> StatorResult<String> {
    this_value.to_js_string()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use crate::objects::js_object::JsObject;
    use crate::objects::map::PropertyAttributes;
    use crate::objects::value::JsValue;

    // ── object_create ────────────────────────────────────────────────────────

    #[test]
    fn test_object_create_null_prototype() {
        let obj = object_create(None);
        assert!(obj.prototype().is_none());
    }

    #[test]
    fn test_object_create_with_prototype() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        proto
            .borrow_mut()
            .set_property("x", JsValue::Smi(42))
            .unwrap();
        let child = object_create(Some(Rc::clone(&proto)));
        assert!(child.prototype().is_some());
        // Prototype-chain lookup works.
        assert_eq!(child.get_property("x"), JsValue::Smi(42));
    }

    // ── object_assign ────────────────────────────────────────────────────────

    #[test]
    fn test_object_assign_copies_enumerable_own_properties() {
        let mut target = JsObject::new();
        target.set_property("a", JsValue::Smi(1)).unwrap();

        let mut src = JsObject::new();
        src.set_property("b", JsValue::Smi(2)).unwrap();
        src.set_property("c", JsValue::Boolean(true)).unwrap();

        object_assign(&mut target, &[&src]).unwrap();

        assert_eq!(target.get_own_property("a"), Some(JsValue::Smi(1)));
        assert_eq!(target.get_own_property("b"), Some(JsValue::Smi(2)));
        assert_eq!(target.get_own_property("c"), Some(JsValue::Boolean(true)));
    }

    #[test]
    fn test_object_assign_skips_non_enumerable() {
        let mut target = JsObject::new();
        let mut src = JsObject::new();
        // Define a non-enumerable property.
        src.define_own_property("hidden", JsValue::Smi(99), PropertyAttributes::WRITABLE)
            .unwrap();

        object_assign(&mut target, &[&src]).unwrap();
        // Non-enumerable property must NOT be copied.
        assert_eq!(target.get_own_property("hidden"), None);
    }

    #[test]
    fn test_object_assign_multiple_sources() {
        let mut target = JsObject::new();
        let mut s1 = JsObject::new();
        s1.set_property("x", JsValue::Smi(1)).unwrap();
        let mut s2 = JsObject::new();
        s2.set_property("y", JsValue::Smi(2)).unwrap();

        object_assign(&mut target, &[&s1, &s2]).unwrap();
        assert_eq!(target.get_own_property("x"), Some(JsValue::Smi(1)));
        assert_eq!(target.get_own_property("y"), Some(JsValue::Smi(2)));
    }

    #[test]
    fn test_object_assign_later_source_overwrites() {
        let mut target = JsObject::new();
        let mut s1 = JsObject::new();
        s1.set_property("k", JsValue::Smi(1)).unwrap();
        let mut s2 = JsObject::new();
        s2.set_property("k", JsValue::Smi(2)).unwrap();

        object_assign(&mut target, &[&s1, &s2]).unwrap();
        assert_eq!(target.get_own_property("k"), Some(JsValue::Smi(2)));
    }

    // ── object_keys / values / entries ───────────────────────────────────────

    #[test]
    fn test_object_keys_returns_enumerable_own_keys() {
        let mut obj = JsObject::new();
        obj.set_property("a", JsValue::Smi(1)).unwrap();
        obj.set_property("b", JsValue::Smi(2)).unwrap();
        // Non-enumerable.
        obj.define_own_property("c", JsValue::Smi(3), PropertyAttributes::WRITABLE)
            .unwrap();

        let keys = object_keys(&obj);
        assert!(keys.contains(&"a".to_string()));
        assert!(keys.contains(&"b".to_string()));
        assert!(!keys.contains(&"c".to_string()));
    }

    #[test]
    fn test_object_keys_excludes_prototype_properties() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        proto
            .borrow_mut()
            .set_property("inherited", JsValue::Smi(0))
            .unwrap();
        let child = object_create(Some(proto));
        // Object.keys only returns own enumerable keys.
        assert!(!object_keys(&child).contains(&"inherited".to_string()));
    }

    #[test]
    fn test_object_values() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(10)).unwrap();
        obj.set_property("y", JsValue::Smi(20)).unwrap();

        let vals = object_values(&obj);
        assert_eq!(vals.len(), 2);
        assert!(vals.contains(&JsValue::Smi(10)));
        assert!(vals.contains(&JsValue::Smi(20)));
    }

    #[test]
    fn test_object_entries() {
        let mut obj = JsObject::new();
        obj.set_property("p", JsValue::Boolean(true)).unwrap();

        let entries = object_entries(&obj);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "p");
        assert_eq!(entries[0].1, JsValue::Boolean(true));
    }

    // ── object_define_property ───────────────────────────────────────────────

    #[test]
    fn test_object_define_property_creates_property() {
        let mut obj = JsObject::new();
        object_define_property(
            &mut obj,
            "x",
            JsValue::Smi(7),
            PropertyAttributes::WRITABLE | PropertyAttributes::ENUMERABLE,
        )
        .unwrap();
        assert_eq!(obj.get_own_property("x"), Some(JsValue::Smi(7)));
    }

    #[test]
    fn test_object_define_property_non_writable_rejects_assignment() {
        let mut obj = JsObject::new();
        object_define_property(&mut obj, "c", JsValue::Smi(1), PropertyAttributes::empty())
            .unwrap();
        // Attempt to set the property via normal assignment should fail.
        let err = obj.set_property("c", JsValue::Smi(2));
        assert!(err.is_err());
    }

    // ── object_get_own_property_descriptor ───────────────────────────────────

    #[test]
    fn test_get_own_property_descriptor_exists() {
        let mut obj = JsObject::new();
        let attrs = PropertyAttributes::WRITABLE | PropertyAttributes::ENUMERABLE;
        obj.define_own_property("k", JsValue::Smi(5), attrs)
            .unwrap();

        let desc = object_get_own_property_descriptor(&obj, "k");
        assert!(desc.is_some());
        let (val, a) = desc.unwrap();
        assert_eq!(val, JsValue::Smi(5));
        assert_eq!(a, attrs);
    }

    #[test]
    fn test_get_own_property_descriptor_missing() {
        let obj = JsObject::new();
        assert!(object_get_own_property_descriptor(&obj, "nope").is_none());
    }

    // ── object_get_prototype_of / set_prototype_of ───────────────────────────

    #[test]
    fn test_get_prototype_of_none() {
        let obj = JsObject::new();
        assert!(object_get_prototype_of(&obj).is_none());
    }

    #[test]
    fn test_get_prototype_of_some() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        let child = object_create(Some(Rc::clone(&proto)));
        let got = object_get_prototype_of(&child);
        assert!(got.is_some());
        assert!(Rc::ptr_eq(&proto, &got.unwrap()));
    }

    #[test]
    fn test_set_prototype_of_changes_prototype() {
        let mut obj = JsObject::new();
        let proto = Rc::new(RefCell::new(JsObject::new()));
        object_set_prototype_of(&mut obj, Some(Rc::clone(&proto))).unwrap();
        assert!(obj.prototype().is_some());
    }

    #[test]
    fn test_set_prototype_of_non_extensible_same_proto_allowed() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        let mut obj = object_create(Some(Rc::clone(&proto)));
        obj.prevent_extensions();
        // Setting the same prototype on a non-extensible object is allowed.
        object_set_prototype_of(&mut obj, Some(Rc::clone(&proto))).unwrap();
    }

    #[test]
    fn test_set_prototype_of_non_extensible_different_proto_rejected() {
        let mut obj = JsObject::new();
        obj.prevent_extensions();
        let new_proto = Rc::new(RefCell::new(JsObject::new()));
        let err = object_set_prototype_of(&mut obj, Some(new_proto));
        assert!(matches!(err, Err(StatorError::TypeError(_))));
    }

    // ── object_freeze ────────────────────────────────────────────────────────

    #[test]
    fn test_object_freeze_makes_non_extensible() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        object_freeze(&mut obj).unwrap();
        assert!(!obj.is_extensible());
    }

    #[test]
    fn test_object_freeze_prevents_write() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        object_freeze(&mut obj).unwrap();
        let err = obj.set_property("x", JsValue::Smi(2));
        assert!(matches!(err, Err(StatorError::TypeError(_))));
    }

    #[test]
    fn test_object_freeze_prevents_new_property() {
        let mut obj = JsObject::new();
        object_freeze(&mut obj).unwrap();
        let err = obj.set_property("new_key", JsValue::Smi(1));
        assert!(matches!(err, Err(StatorError::TypeError(_))));
    }

    #[test]
    fn test_object_is_frozen_after_freeze() {
        let mut obj = JsObject::new();
        obj.set_property("a", JsValue::Smi(1)).unwrap();
        assert!(!object_is_frozen(&obj));
        object_freeze(&mut obj).unwrap();
        assert!(object_is_frozen(&obj));
    }

    #[test]
    fn test_object_is_frozen_empty_non_extensible() {
        let mut obj = JsObject::new();
        obj.prevent_extensions();
        // An empty non-extensible object is considered frozen.
        assert!(object_is_frozen(&obj));
    }

    // ── object_seal ──────────────────────────────────────────────────────────

    #[test]
    fn test_object_seal_makes_non_extensible() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        object_seal(&mut obj).unwrap();
        assert!(!obj.is_extensible());
    }

    #[test]
    fn test_object_seal_allows_write_to_existing() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        object_seal(&mut obj).unwrap();
        // Writing an existing property is still allowed after seal.
        obj.set_property("x", JsValue::Smi(99)).unwrap();
        assert_eq!(obj.get_own_property("x"), Some(JsValue::Smi(99)));
    }

    #[test]
    fn test_object_seal_prevents_new_property() {
        let mut obj = JsObject::new();
        object_seal(&mut obj).unwrap();
        let err = obj.set_property("new_key", JsValue::Smi(1));
        assert!(matches!(err, Err(StatorError::TypeError(_))));
    }

    // ── object_is ────────────────────────────────────────────────────────────

    #[test]
    fn test_object_is_undefined_undefined() {
        assert!(object_is(&JsValue::Undefined, &JsValue::Undefined));
    }

    #[test]
    fn test_object_is_null_null() {
        assert!(object_is(&JsValue::Null, &JsValue::Null));
    }

    #[test]
    fn test_object_is_null_undefined_differs() {
        assert!(!object_is(&JsValue::Null, &JsValue::Undefined));
    }

    #[test]
    fn test_object_is_nan_equals_nan() {
        assert!(object_is(
            &JsValue::HeapNumber(f64::NAN),
            &JsValue::HeapNumber(f64::NAN)
        ));
    }

    #[test]
    fn test_object_is_positive_zero_not_equal_negative_zero() {
        assert!(!object_is(
            &JsValue::HeapNumber(0.0),
            &JsValue::HeapNumber(-0.0)
        ));
    }

    #[test]
    fn test_object_is_negative_zero_not_equal_positive_zero() {
        assert!(!object_is(
            &JsValue::HeapNumber(-0.0),
            &JsValue::HeapNumber(0.0)
        ));
    }

    #[test]
    fn test_object_is_smi_zero_not_equal_negative_zero() {
        // Smi 0 is +0; HeapNumber -0.0 is -0 → not the same value.
        assert!(!object_is(&JsValue::Smi(0), &JsValue::HeapNumber(-0.0)));
        assert!(!object_is(&JsValue::HeapNumber(-0.0), &JsValue::Smi(0)));
    }

    #[test]
    fn test_object_is_same_smi() {
        assert!(object_is(&JsValue::Smi(42), &JsValue::Smi(42)));
        assert!(!object_is(&JsValue::Smi(1), &JsValue::Smi(2)));
    }

    #[test]
    fn test_object_is_same_string() {
        assert!(object_is(
            &JsValue::String("hello".to_string().into()),
            &JsValue::String("hello".to_string().into())
        ));
        assert!(!object_is(
            &JsValue::String("a".to_string().into()),
            &JsValue::String("b".to_string().into())
        ));
    }

    #[test]
    fn test_object_is_same_boolean() {
        assert!(object_is(&JsValue::Boolean(true), &JsValue::Boolean(true)));
        assert!(!object_is(
            &JsValue::Boolean(true),
            &JsValue::Boolean(false)
        ));
    }

    #[test]
    fn test_object_is_object_pointer_equality() {
        let mut o1 = JsObject::new();
        // We need raw pointers; create a HeapObject wrapper manually.
        use crate::objects::heap_object::HeapObject;
        let mut h = HeapObject::new_null();
        let ptr: *mut HeapObject = &raw mut h;
        let v1 = JsValue::Object(ptr);
        let v2 = JsValue::Object(ptr);
        let mut h2 = HeapObject::new_null();
        let ptr2: *mut HeapObject = &raw mut h2;
        let v3 = JsValue::Object(ptr2);
        // Suppress unused variable warning.
        let _ = &mut o1;
        assert!(object_is(&v1, &v2));
        assert!(!object_is(&v1, &v3));
    }

    // ── object_from_entries ──────────────────────────────────────────────────

    #[test]
    fn test_object_from_entries_basic() {
        let entries = vec![
            ("a".to_string(), JsValue::Smi(1)),
            ("b".to_string(), JsValue::Boolean(false)),
        ];
        let obj = object_from_entries(entries).unwrap();
        assert_eq!(obj.get_own_property("a"), Some(JsValue::Smi(1)));
        assert_eq!(obj.get_own_property("b"), Some(JsValue::Boolean(false)));
    }

    #[test]
    fn test_object_from_entries_empty() {
        let obj = object_from_entries(vec![]).unwrap();
        assert!(obj.own_property_keys().is_empty());
    }

    #[test]
    fn test_object_from_entries_later_overrides_earlier() {
        let entries = vec![
            ("k".to_string(), JsValue::Smi(1)),
            ("k".to_string(), JsValue::Smi(2)),
        ];
        let obj = object_from_entries(entries).unwrap();
        assert_eq!(obj.get_own_property("k"), Some(JsValue::Smi(2)));
    }

    // ── object_is_sealed ─────────────────────────────────────────────────

    #[test]
    fn test_object_is_sealed_extensible_returns_false() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        assert!(!object_is_sealed(&obj));
    }

    #[test]
    fn test_object_is_sealed_after_seal() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        object_seal(&mut obj).unwrap();
        assert!(object_is_sealed(&obj));
    }

    #[test]
    fn test_object_is_sealed_empty_non_extensible() {
        let mut obj = JsObject::new();
        obj.prevent_extensions();
        assert!(object_is_sealed(&obj));
    }

    // ── object_get_own_property_names ────────────────────────────────────

    #[test]
    fn test_get_own_property_names_includes_non_enumerable() {
        let mut obj = JsObject::new();
        obj.set_property("visible", JsValue::Smi(1)).unwrap();
        obj.define_own_property("hidden", JsValue::Smi(2), PropertyAttributes::WRITABLE)
            .unwrap();

        let names = object_get_own_property_names(&obj);
        assert!(names.contains(&"visible".to_string()));
        assert!(names.contains(&"hidden".to_string()));
    }

    #[test]
    fn test_get_own_property_names_empty() {
        let obj = JsObject::new();
        assert!(object_get_own_property_names(&obj).is_empty());
    }

    // ── object_has_own ──────────────────────────────────────────────────

    #[test]
    fn test_has_own_existing_property() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        assert!(object_has_own(&obj, "x"));
    }

    #[test]
    fn test_has_own_missing_property() {
        let obj = JsObject::new();
        assert!(!object_has_own(&obj, "x"));
    }

    #[test]
    fn test_has_own_does_not_check_prototype() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        proto
            .borrow_mut()
            .set_property("inherited", JsValue::Smi(1))
            .unwrap();
        let child = object_create(Some(proto));
        assert!(!object_has_own(&child, "inherited"));
    }

    // ── object_prevent_extensions / is_extensible ───────────────────────

    #[test]
    fn test_is_extensible_default_true() {
        let obj = JsObject::new();
        assert!(object_is_extensible(&obj));
    }

    #[test]
    fn test_prevent_extensions_makes_non_extensible() {
        let mut obj = JsObject::new();
        object_prevent_extensions(&mut obj);
        assert!(!object_is_extensible(&obj));
    }

    // ── object_get_own_property_descriptors ──────────────────────────────

    #[test]
    fn test_get_own_property_descriptors_basic() {
        let mut obj = JsObject::new();
        obj.set_property("a", JsValue::Smi(1)).unwrap();
        obj.set_property("b", JsValue::Smi(2)).unwrap();

        let descs = object_get_own_property_descriptors(&obj);
        assert_eq!(descs.len(), 2);
    }

    #[test]
    fn test_get_own_property_descriptors_empty() {
        let obj = JsObject::new();
        assert!(object_get_own_property_descriptors(&obj).is_empty());
    }

    // ── object_get_own_property_symbols ──────────────────────────────────

    #[test]
    fn test_get_own_property_symbols_returns_empty_without_symbols() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        assert!(object_get_own_property_symbols(&obj).is_empty());
    }

    // ── object_define_property_from_descriptor ───────────────────────────

    #[test]
    fn test_define_property_from_data_descriptor() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        let mut desc_map = PropertyMap::new();
        desc_map.insert("value".to_string(), JsValue::Smi(42));
        desc_map.insert("writable".to_string(), JsValue::Boolean(true));
        desc_map.insert("enumerable".to_string(), JsValue::Boolean(true));
        desc_map.insert("configurable".to_string(), JsValue::Boolean(true));
        let desc = JsValue::PlainObject(Rc::new(RefCell::new(desc_map)));

        object_define_property_from_descriptor(&mut obj, "x", &desc).unwrap();
        assert_eq!(obj.get_own_property("x"), Some(JsValue::Smi(42)));
        // Verify attributes via get_own_property_descriptor.
        let (_, attrs) = obj.get_own_property_descriptor("x").unwrap();
        assert!(attrs.contains(PropertyAttributes::WRITABLE));
        assert!(attrs.contains(PropertyAttributes::ENUMERABLE));
        assert!(attrs.contains(PropertyAttributes::CONFIGURABLE));
    }

    #[test]
    fn test_define_property_from_descriptor_non_writable() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        let mut desc_map = PropertyMap::new();
        desc_map.insert("value".to_string(), JsValue::Smi(7));
        desc_map.insert("writable".to_string(), JsValue::Boolean(false));
        let desc = JsValue::PlainObject(Rc::new(RefCell::new(desc_map)));

        object_define_property_from_descriptor(&mut obj, "ro", &desc).unwrap();
        // Attempt to write should fail.
        let err = obj.set_property("ro", JsValue::Smi(99));
        assert!(matches!(err, Err(StatorError::TypeError(_))));
        // Value stays the same.
        assert_eq!(obj.get_own_property("ro"), Some(JsValue::Smi(7)));
    }

    #[test]
    fn test_define_property_from_descriptor_rejects_mixed() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        let mut desc_map = PropertyMap::new();
        desc_map.insert("value".to_string(), JsValue::Smi(1));
        desc_map.insert("get".to_string(), JsValue::Undefined);
        let desc = JsValue::PlainObject(Rc::new(RefCell::new(desc_map)));

        let err = object_define_property_from_descriptor(&mut obj, "bad", &desc);
        assert!(matches!(err, Err(StatorError::TypeError(_))));
    }

    #[test]
    fn test_define_property_from_descriptor_non_configurable_redefine_rejected() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        // First: define non-configurable property.
        obj.define_own_property("nc", JsValue::Smi(1), PropertyAttributes::WRITABLE)
            .unwrap();
        // Try to make it configurable.
        let mut desc_map = PropertyMap::new();
        desc_map.insert("value".to_string(), JsValue::Smi(2));
        desc_map.insert("configurable".to_string(), JsValue::Boolean(true));
        let desc = JsValue::PlainObject(Rc::new(RefCell::new(desc_map)));

        let err = object_define_property_from_descriptor(&mut obj, "nc", &desc);
        assert!(matches!(err, Err(StatorError::TypeError(_))));
    }

    // ── object_define_properties ─────────────────────────────────────────

    #[test]
    fn test_define_properties_multiple_keys() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();

        let mut desc_a = PropertyMap::new();
        desc_a.insert("value".to_string(), JsValue::Smi(10));
        desc_a.insert("writable".to_string(), JsValue::Boolean(true));
        desc_a.insert("enumerable".to_string(), JsValue::Boolean(true));
        desc_a.insert("configurable".to_string(), JsValue::Boolean(true));

        let mut desc_b = PropertyMap::new();
        desc_b.insert("value".to_string(), JsValue::Smi(20));
        desc_b.insert("writable".to_string(), JsValue::Boolean(false));
        desc_b.insert("enumerable".to_string(), JsValue::Boolean(false));
        desc_b.insert("configurable".to_string(), JsValue::Boolean(false));

        let mut props_map = PropertyMap::new();
        props_map.insert(
            "a".to_string(),
            JsValue::PlainObject(Rc::new(RefCell::new(desc_a))),
        );
        props_map.insert(
            "b".to_string(),
            JsValue::PlainObject(Rc::new(RefCell::new(desc_b))),
        );
        let props = JsValue::PlainObject(Rc::new(RefCell::new(props_map)));

        object_define_properties(&mut obj, &props).unwrap();
        assert_eq!(obj.get_own_property("a"), Some(JsValue::Smi(10)));
        assert_eq!(obj.get_own_property("b"), Some(JsValue::Smi(20)));

        // 'a' is writable and enumerable.
        let (_, a_attrs) = obj.get_own_property_descriptor("a").unwrap();
        assert!(a_attrs.contains(PropertyAttributes::WRITABLE));
        assert!(a_attrs.contains(PropertyAttributes::ENUMERABLE));

        // 'b' is non-writable and non-enumerable.
        let (_, b_attrs) = obj.get_own_property_descriptor("b").unwrap();
        assert!(!b_attrs.contains(PropertyAttributes::WRITABLE));
        assert!(!b_attrs.contains(PropertyAttributes::ENUMERABLE));
    }

    #[test]
    fn test_define_properties_rejects_non_object() {
        let mut obj = JsObject::new();
        let err = object_define_properties(&mut obj, &JsValue::Smi(1));
        assert!(matches!(err, Err(StatorError::TypeError(_))));
    }

    // ── object_get_own_property_descriptor_as_object ─────────────────────

    #[test]
    fn test_get_own_property_descriptor_as_object_data() {
        let mut obj = JsObject::new();
        let attrs = PropertyAttributes::WRITABLE | PropertyAttributes::ENUMERABLE;
        obj.define_own_property("k", JsValue::Smi(5), attrs)
            .unwrap();

        let desc_obj = object_get_own_property_descriptor_as_object(&obj, "k");
        assert!(desc_obj.is_some());
        let desc_obj = desc_obj.unwrap();
        if let JsValue::PlainObject(map) = &desc_obj {
            let map = map.borrow();
            assert_eq!(map.get("value"), Some(&JsValue::Smi(5)));
            assert_eq!(map.get("writable"), Some(&JsValue::Boolean(true)));
            assert_eq!(map.get("enumerable"), Some(&JsValue::Boolean(true)));
            assert_eq!(map.get("configurable"), Some(&JsValue::Boolean(false)));
        } else {
            panic!("expected PlainObject");
        }
    }

    #[test]
    fn test_get_own_property_descriptor_as_object_missing() {
        let obj = JsObject::new();
        assert!(object_get_own_property_descriptor_as_object(&obj, "nope").is_none());
    }

    // ── Property enforcement via freeze/seal with descriptors ────────────

    #[test]
    fn test_freeze_sets_non_writable_non_configurable() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        object_freeze(&mut obj).unwrap();

        let desc_obj = object_get_own_property_descriptor_as_object(&obj, "x").unwrap();
        if let JsValue::PlainObject(map) = &desc_obj {
            let map = map.borrow();
            assert_eq!(map.get("writable"), Some(&JsValue::Boolean(false)));
            assert_eq!(map.get("configurable"), Some(&JsValue::Boolean(false)));
        } else {
            panic!("expected PlainObject");
        }
    }

    #[test]
    fn test_seal_sets_non_configurable_preserves_writable() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        object_seal(&mut obj).unwrap();

        let desc_obj = object_get_own_property_descriptor_as_object(&obj, "x").unwrap();
        if let JsValue::PlainObject(map) = &desc_obj {
            let map = map.borrow();
            // Writable is preserved (it was true from set_property).
            assert_eq!(map.get("writable"), Some(&JsValue::Boolean(true)));
            assert_eq!(map.get("configurable"), Some(&JsValue::Boolean(false)));
        } else {
            panic!("expected PlainObject");
        }
    }

    #[test]
    fn test_non_enumerable_property_hidden_from_keys() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        let mut desc_map = PropertyMap::new();
        desc_map.insert("value".to_string(), JsValue::Smi(42));
        desc_map.insert("writable".to_string(), JsValue::Boolean(true));
        desc_map.insert("enumerable".to_string(), JsValue::Boolean(false));
        desc_map.insert("configurable".to_string(), JsValue::Boolean(true));
        let desc = JsValue::PlainObject(Rc::new(RefCell::new(desc_map)));

        object_define_property_from_descriptor(&mut obj, "hidden", &desc).unwrap();
        let keys = object_keys(&obj);
        assert!(!keys.contains(&"hidden".to_string()));
        // But Object.getOwnPropertyNames includes it.
        let names = object_get_own_property_names(&obj);
        assert!(names.contains(&"hidden".to_string()));
    }

    #[test]
    fn test_non_configurable_property_cannot_be_deleted() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        let mut desc_map = PropertyMap::new();
        desc_map.insert("value".to_string(), JsValue::Smi(1));
        desc_map.insert("configurable".to_string(), JsValue::Boolean(false));
        let desc = JsValue::PlainObject(Rc::new(RefCell::new(desc_map)));

        object_define_property_from_descriptor(&mut obj, "sticky", &desc).unwrap();
        let deleted = obj.delete_own_property("sticky").unwrap();
        assert!(!deleted);
    }

    // ── Non-writable + non-configurable value change rejection ──────────

    #[test]
    fn test_define_own_property_rejects_value_change_on_non_writable_non_configurable() {
        let mut obj = JsObject::new();
        // Create a non-writable, non-configurable property.
        obj.define_own_property("x", JsValue::Smi(1), PropertyAttributes::empty())
            .unwrap();
        // Attempting to change the value should fail.
        let err = obj.define_own_property("x", JsValue::Smi(2), PropertyAttributes::empty());
        assert!(matches!(err, Err(StatorError::TypeError(_))));
        // Value should remain unchanged.
        assert_eq!(obj.get_own_property("x"), Some(JsValue::Smi(1)));
    }

    #[test]
    fn test_define_own_property_allows_same_value_on_non_writable_non_configurable() {
        let mut obj = JsObject::new();
        obj.define_own_property("x", JsValue::Smi(1), PropertyAttributes::empty())
            .unwrap();
        // Setting the same value is allowed per spec.
        obj.define_own_property("x", JsValue::Smi(1), PropertyAttributes::empty())
            .unwrap();
        assert_eq!(obj.get_own_property("x"), Some(JsValue::Smi(1)));
    }

    #[test]
    fn test_define_property_from_descriptor_rejects_value_change_non_writable() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        // Define non-writable, non-configurable property.
        obj.define_own_property("p", JsValue::Smi(10), PropertyAttributes::empty())
            .unwrap();
        // Try to change value via descriptor.
        let mut desc_map = PropertyMap::new();
        desc_map.insert("value".to_string(), JsValue::Smi(20));
        let desc = JsValue::PlainObject(Rc::new(RefCell::new(desc_map)));

        let err = object_define_property_from_descriptor(&mut obj, "p", &desc);
        assert!(matches!(err, Err(StatorError::TypeError(_))));
    }

    // ── Accessor descriptor support ─────────────────────────────────────

    #[test]
    fn test_define_property_accessor_stores_getter_setter() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        let mut desc_map = PropertyMap::new();
        desc_map.insert("get".to_string(), JsValue::Boolean(true));
        desc_map.insert("set".to_string(), JsValue::Boolean(false));
        desc_map.insert("enumerable".to_string(), JsValue::Boolean(true));
        desc_map.insert("configurable".to_string(), JsValue::Boolean(true));
        let desc = JsValue::PlainObject(Rc::new(RefCell::new(desc_map)));

        object_define_property_from_descriptor(&mut obj, "x", &desc).unwrap();
        // Internal getter/setter entries must be stored.
        assert_eq!(
            obj.get_own_property("__get_x__"),
            Some(JsValue::Boolean(true))
        );
        assert_eq!(
            obj.get_own_property("__set_x__"),
            Some(JsValue::Boolean(false))
        );
    }

    #[test]
    fn test_get_own_property_descriptor_returns_accessor() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        let mut desc_map = PropertyMap::new();
        desc_map.insert("get".to_string(), JsValue::Boolean(true));
        desc_map.insert("set".to_string(), JsValue::Boolean(false));
        desc_map.insert("enumerable".to_string(), JsValue::Boolean(true));
        desc_map.insert("configurable".to_string(), JsValue::Boolean(true));
        let desc_val = JsValue::PlainObject(Rc::new(RefCell::new(desc_map)));
        object_define_property_from_descriptor(&mut obj, "x", &desc_val).unwrap();

        let result = object_get_own_property_descriptor_as_object(&obj, "x");
        assert!(result.is_some());
        let desc = FullPropertyDescriptor::from_object(&result.unwrap()).unwrap();
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
    fn test_define_property_accessor_to_data_transition() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        // Define as accessor first.
        let mut acc_map = PropertyMap::new();
        acc_map.insert("get".to_string(), JsValue::Boolean(true));
        acc_map.insert("configurable".to_string(), JsValue::Boolean(true));
        let acc_desc = JsValue::PlainObject(Rc::new(RefCell::new(acc_map)));
        object_define_property_from_descriptor(&mut obj, "x", &acc_desc).unwrap();
        assert!(obj.has_own_property("__get_x__"));

        // Redefine as data.
        let mut data_map = PropertyMap::new();
        data_map.insert("value".to_string(), JsValue::Smi(42));
        data_map.insert("writable".to_string(), JsValue::Boolean(true));
        let data_desc = JsValue::PlainObject(Rc::new(RefCell::new(data_map)));
        object_define_property_from_descriptor(&mut obj, "x", &data_desc).unwrap();

        // Accessor entries must be removed.
        assert!(!obj.has_own_property("__get_x__"));
        assert!(!obj.has_own_property("__set_x__"));
        assert_eq!(obj.get_own_property("x"), Some(JsValue::Smi(42)));
    }

    #[test]
    fn test_define_property_data_to_accessor_transition() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(10)).unwrap();

        // Redefine as accessor.
        let mut acc_map = PropertyMap::new();
        acc_map.insert("get".to_string(), JsValue::Boolean(true));
        acc_map.insert("set".to_string(), JsValue::Boolean(false));
        acc_map.insert("configurable".to_string(), JsValue::Boolean(true));
        let acc_desc = JsValue::PlainObject(Rc::new(RefCell::new(acc_map)));
        object_define_property_from_descriptor(&mut obj, "x", &acc_desc).unwrap();

        assert!(obj.has_own_property("__get_x__"));
        assert!(obj.has_own_property("__set_x__"));
        // Descriptor should be accessor.
        let result = object_get_own_property_descriptor_as_object(&obj, "x");
        let desc = FullPropertyDescriptor::from_object(&result.unwrap()).unwrap();
        assert!(desc.is_accessor());
    }

    #[test]
    fn test_define_property_nonconfigurable_accessor_rejects_data_conversion() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        // Non-configurable accessor.
        let mut acc_map = PropertyMap::new();
        acc_map.insert("get".to_string(), JsValue::Boolean(true));
        acc_map.insert("configurable".to_string(), JsValue::Boolean(false));
        let acc_desc = JsValue::PlainObject(Rc::new(RefCell::new(acc_map)));
        object_define_property_from_descriptor(&mut obj, "x", &acc_desc).unwrap();

        // Try to redefine as data → must fail.
        let mut data_map = PropertyMap::new();
        data_map.insert("value".to_string(), JsValue::Smi(1));
        let data_desc = JsValue::PlainObject(Rc::new(RefCell::new(data_map)));
        let err = object_define_property_from_descriptor(&mut obj, "x", &data_desc);
        assert!(matches!(err, Err(StatorError::TypeError(_))));
    }

    #[test]
    fn test_define_property_nonconfigurable_data_rejects_accessor_conversion() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        // Non-configurable data.
        obj.define_own_property("x", JsValue::Smi(1), PropertyAttributes::empty())
            .unwrap();

        // Try to redefine as accessor → must fail.
        let mut acc_map = PropertyMap::new();
        acc_map.insert("get".to_string(), JsValue::Boolean(true));
        let acc_desc = JsValue::PlainObject(Rc::new(RefCell::new(acc_map)));
        let err = object_define_property_from_descriptor(&mut obj, "x", &acc_desc);
        assert!(matches!(err, Err(StatorError::TypeError(_))));
    }

    #[test]
    fn test_define_property_nonconfigurable_accessor_rejects_getter_change() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        // Non-configurable accessor.
        let mut acc_map = PropertyMap::new();
        acc_map.insert("get".to_string(), JsValue::Boolean(true));
        acc_map.insert("set".to_string(), JsValue::Boolean(false));
        acc_map.insert("configurable".to_string(), JsValue::Boolean(false));
        let acc_desc = JsValue::PlainObject(Rc::new(RefCell::new(acc_map)));
        object_define_property_from_descriptor(&mut obj, "x", &acc_desc).unwrap();

        // Try to change getter → must fail.
        let mut new_acc = PropertyMap::new();
        new_acc.insert("get".to_string(), JsValue::Smi(999));
        new_acc.insert("set".to_string(), JsValue::Boolean(false));
        let new_desc = JsValue::PlainObject(Rc::new(RefCell::new(new_acc)));
        let err = object_define_property_from_descriptor(&mut obj, "x", &new_desc);
        assert!(matches!(err, Err(StatorError::TypeError(_))));
    }

    #[test]
    fn test_accessor_hidden_from_own_property_names() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        let mut desc_map = PropertyMap::new();
        desc_map.insert("get".to_string(), JsValue::Boolean(true));
        desc_map.insert("configurable".to_string(), JsValue::Boolean(true));
        let desc = JsValue::PlainObject(Rc::new(RefCell::new(desc_map)));
        object_define_property_from_descriptor(&mut obj, "x", &desc).unwrap();

        let names = object_get_own_property_names(&obj);
        assert!(names.contains(&"x".to_string()));
        assert!(!names.contains(&"__get_x__".to_string()));
        assert!(!names.contains(&"__set_x__".to_string()));
    }

    #[test]
    fn test_accessor_attributes_no_writable() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        let mut desc_map = PropertyMap::new();
        desc_map.insert("get".to_string(), JsValue::Boolean(true));
        desc_map.insert("enumerable".to_string(), JsValue::Boolean(true));
        desc_map.insert("configurable".to_string(), JsValue::Boolean(true));
        let desc = JsValue::PlainObject(Rc::new(RefCell::new(desc_map)));
        object_define_property_from_descriptor(&mut obj, "x", &desc).unwrap();

        let (_, attrs) = obj.get_own_property_descriptor("x").unwrap();
        assert!(!attrs.contains(PropertyAttributes::WRITABLE));
        assert!(attrs.contains(PropertyAttributes::ENUMERABLE));
        assert!(attrs.contains(PropertyAttributes::CONFIGURABLE));
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  E2E property-attribute deep-conformance tests
    // ══════════════════════════════════════════════════════════════════════════

    // ── 1. Object.defineProperties ──────────────────────────────────────────

    #[test]
    fn test_e2e_define_properties_defines_multiple_with_correct_attrs() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();

        let mut desc_x = PropertyMap::new();
        desc_x.insert("value".to_string(), JsValue::Smi(1));
        desc_x.insert("writable".to_string(), JsValue::Boolean(false));
        desc_x.insert("enumerable".to_string(), JsValue::Boolean(true));
        desc_x.insert("configurable".to_string(), JsValue::Boolean(false));

        let mut desc_y = PropertyMap::new();
        desc_y.insert("value".to_string(), JsValue::Smi(2));
        desc_y.insert("writable".to_string(), JsValue::Boolean(true));
        desc_y.insert("enumerable".to_string(), JsValue::Boolean(false));
        desc_y.insert("configurable".to_string(), JsValue::Boolean(true));

        let mut props_map = PropertyMap::new();
        props_map.insert(
            "x".to_string(),
            JsValue::PlainObject(Rc::new(RefCell::new(desc_x))),
        );
        props_map.insert(
            "y".to_string(),
            JsValue::PlainObject(Rc::new(RefCell::new(desc_y))),
        );
        let props = JsValue::PlainObject(Rc::new(RefCell::new(props_map)));

        object_define_properties(&mut obj, &props).unwrap();

        // x: non-writable, enumerable, non-configurable
        let (xv, xa) = obj.get_own_property_descriptor("x").unwrap();
        assert_eq!(xv, JsValue::Smi(1));
        assert!(!xa.contains(PropertyAttributes::WRITABLE));
        assert!(xa.contains(PropertyAttributes::ENUMERABLE));
        assert!(!xa.contains(PropertyAttributes::CONFIGURABLE));

        // y: writable, non-enumerable, configurable
        let (yv, ya) = obj.get_own_property_descriptor("y").unwrap();
        assert_eq!(yv, JsValue::Smi(2));
        assert!(ya.contains(PropertyAttributes::WRITABLE));
        assert!(!ya.contains(PropertyAttributes::ENUMERABLE));
        assert!(ya.contains(PropertyAttributes::CONFIGURABLE));
    }

    #[test]
    fn test_e2e_define_properties_only_reads_enumerable_props() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();

        let mut desc_a = PropertyMap::new();
        desc_a.insert("value".to_string(), JsValue::Smi(10));
        let mut desc_b = PropertyMap::new();
        desc_b.insert("value".to_string(), JsValue::Smi(20));

        let mut props_map = PropertyMap::new();
        props_map.insert(
            "a".to_string(),
            JsValue::PlainObject(Rc::new(RefCell::new(desc_a))),
        );
        // Make 'b' non-enumerable in the props object.
        props_map.insert_with_attrs(
            "b".to_string(),
            JsValue::PlainObject(Rc::new(RefCell::new(desc_b))),
            PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE,
        );
        let props = JsValue::PlainObject(Rc::new(RefCell::new(props_map)));

        object_define_properties(&mut obj, &props).unwrap();

        assert!(obj.has_own_property("a"), "'a' should be defined");
        assert!(
            !obj.has_own_property("b"),
            "'b' should NOT be defined (non-enumerable in props)"
        );
    }

    // ── 2. Non-writable property semantics ──────────────────────────────────

    #[test]
    fn test_e2e_nonwritable_rejects_set_property() {
        let mut obj = JsObject::new();
        obj.define_own_property("ro", JsValue::Smi(42), PropertyAttributes::ENUMERABLE)
            .unwrap();
        let err = obj.set_property("ro", JsValue::Smi(99));
        assert!(err.is_err(), "set_property on non-writable must fail");
        assert_eq!(obj.get_own_property("ro"), Some(JsValue::Smi(42)));
    }

    #[test]
    fn test_e2e_nonwritable_value_preserved_after_failed_set() {
        let mut obj = JsObject::new();
        obj.define_own_property(
            "x",
            JsValue::String("original".to_string().into()),
            PropertyAttributes::ENUMERABLE | PropertyAttributes::CONFIGURABLE,
        )
        .unwrap();
        let _ = obj.set_property("x", JsValue::String("changed".to_string().into()));
        assert_eq!(
            obj.get_own_property("x"),
            Some(JsValue::String("original".to_string().into()))
        );
    }

    // ── 3. Non-enumerable property semantics ────────────────────────────────

    #[test]
    fn test_e2e_nonenumerable_hidden_from_object_keys() {
        let mut obj = JsObject::new();
        obj.set_property("visible", JsValue::Smi(1)).unwrap();
        obj.define_own_property(
            "hidden",
            JsValue::Smi(2),
            PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE,
        )
        .unwrap();

        let keys = object_keys(&obj);
        assert!(keys.contains(&"visible".to_string()));
        assert!(!keys.contains(&"hidden".to_string()));
    }

    #[test]
    fn test_e2e_nonenumerable_hidden_from_object_values() {
        let mut obj = JsObject::new();
        obj.set_property("a", JsValue::Smi(1)).unwrap();
        obj.define_own_property(
            "b",
            JsValue::Smi(2),
            PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE,
        )
        .unwrap();

        let vals = object_values(&obj);
        assert_eq!(vals, vec![JsValue::Smi(1)]);
    }

    #[test]
    fn test_e2e_nonenumerable_hidden_from_object_entries() {
        let mut obj = JsObject::new();
        obj.set_property("a", JsValue::Smi(1)).unwrap();
        obj.define_own_property(
            "b",
            JsValue::Smi(2),
            PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE,
        )
        .unwrap();

        let entries = object_entries(&obj);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "a");
    }

    #[test]
    fn test_e2e_nonenumerable_visible_to_get_own_property_names() {
        let mut obj = JsObject::new();
        obj.define_own_property(
            "hidden",
            JsValue::Smi(1),
            PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE,
        )
        .unwrap();

        let names = object_get_own_property_names(&obj);
        assert!(names.contains(&"hidden".to_string()));
    }

    #[test]
    fn test_e2e_nonenumerable_hidden_from_object_assign() {
        let mut target = JsObject::new();
        let mut src = JsObject::new();
        src.set_property("vis", JsValue::Smi(1)).unwrap();
        src.define_own_property(
            "hid",
            JsValue::Smi(2),
            PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE,
        )
        .unwrap();

        object_assign(&mut target, &[&src]).unwrap();
        assert!(target.has_own_property("vis"));
        assert!(!target.has_own_property("hid"));
    }

    // ── 4. Non-configurable property semantics ──────────────────────────────

    #[test]
    fn test_e2e_nonconfigurable_cannot_delete() {
        let mut obj = JsObject::new();
        obj.define_own_property("nc", JsValue::Smi(1), PropertyAttributes::WRITABLE)
            .unwrap();
        assert!(!obj.delete_own_property("nc").unwrap());
        assert!(obj.has_own_property("nc"));
    }

    #[test]
    fn test_e2e_nonconfigurable_cannot_change_configurable_flag() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        obj.define_own_property("p", JsValue::Smi(1), PropertyAttributes::WRITABLE)
            .unwrap();

        let mut desc_map = PropertyMap::new();
        desc_map.insert("configurable".to_string(), JsValue::Boolean(true));
        let desc = JsValue::PlainObject(Rc::new(RefCell::new(desc_map)));

        let err = object_define_property_from_descriptor(&mut obj, "p", &desc);
        assert!(err.is_err());
    }

    #[test]
    fn test_e2e_nonconfigurable_cannot_change_enumerable() {
        let mut obj = JsObject::new();
        obj.define_own_property("p", JsValue::Smi(1), PropertyAttributes::WRITABLE)
            .unwrap();

        let err = obj.define_own_property(
            "p",
            JsValue::Smi(1),
            PropertyAttributes::WRITABLE | PropertyAttributes::ENUMERABLE,
        );
        assert!(err.is_err());
    }

    // ── 5. Accessor ↔ data conversion ───────────────────────────────────────

    #[test]
    fn test_e2e_configurable_accessor_to_data_conversion() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();

        let mut acc = PropertyMap::new();
        acc.insert("get".to_string(), JsValue::Boolean(true));
        acc.insert("configurable".to_string(), JsValue::Boolean(true));
        let acc_desc = JsValue::PlainObject(Rc::new(RefCell::new(acc)));
        object_define_property_from_descriptor(&mut obj, "p", &acc_desc).unwrap();

        let desc = object_get_own_property_descriptor_as_object(&obj, "p").unwrap();
        let fpd = FullPropertyDescriptor::from_object(&desc).unwrap();
        assert!(fpd.is_accessor());

        let mut data = PropertyMap::new();
        data.insert("value".to_string(), JsValue::Smi(42));
        data.insert("writable".to_string(), JsValue::Boolean(true));
        let data_desc = JsValue::PlainObject(Rc::new(RefCell::new(data)));
        object_define_property_from_descriptor(&mut obj, "p", &data_desc).unwrap();

        assert_eq!(obj.get_own_property("p"), Some(JsValue::Smi(42)));
        assert!(!obj.has_own_property("__get_p__"));
    }

    #[test]
    fn test_e2e_configurable_data_to_accessor_conversion() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        obj.set_property("p", JsValue::Smi(10)).unwrap();

        let mut acc = PropertyMap::new();
        acc.insert("get".to_string(), JsValue::Boolean(true));
        acc.insert("set".to_string(), JsValue::Boolean(false));
        acc.insert("configurable".to_string(), JsValue::Boolean(true));
        let acc_desc = JsValue::PlainObject(Rc::new(RefCell::new(acc)));
        object_define_property_from_descriptor(&mut obj, "p", &acc_desc).unwrap();

        assert!(obj.has_own_property("__get_p__"));
        assert!(obj.has_own_property("__set_p__"));
    }

    #[test]
    fn test_e2e_nonconfigurable_rejects_data_to_accessor() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        obj.define_own_property("p", JsValue::Smi(1), PropertyAttributes::empty())
            .unwrap();

        let mut acc = PropertyMap::new();
        acc.insert("get".to_string(), JsValue::Boolean(true));
        let acc_desc = JsValue::PlainObject(Rc::new(RefCell::new(acc)));

        let err = object_define_property_from_descriptor(&mut obj, "p", &acc_desc);
        assert!(err.is_err());
    }

    #[test]
    fn test_e2e_nonconfigurable_rejects_accessor_to_data() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();

        let mut acc = PropertyMap::new();
        acc.insert("get".to_string(), JsValue::Boolean(true));
        acc.insert("configurable".to_string(), JsValue::Boolean(false));
        let acc_desc = JsValue::PlainObject(Rc::new(RefCell::new(acc)));
        object_define_property_from_descriptor(&mut obj, "p", &acc_desc).unwrap();

        let mut data = PropertyMap::new();
        data.insert("value".to_string(), JsValue::Smi(1));
        let data_desc = JsValue::PlainObject(Rc::new(RefCell::new(data)));

        let err = object_define_property_from_descriptor(&mut obj, "p", &data_desc);
        assert!(err.is_err());
    }

    // ── 6. Object.getOwnPropertyDescriptors (as object) ────────────────────

    #[test]
    fn test_e2e_get_own_property_descriptors_as_object_basic() {
        let mut obj = JsObject::new();
        obj.set_property("a", JsValue::Smi(1)).unwrap();
        obj.define_own_property(
            "b",
            JsValue::Smi(2),
            PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE,
        )
        .unwrap();

        let result = object_get_own_property_descriptors_as_object(&obj);
        if let JsValue::PlainObject(map) = result {
            let borrow = map.borrow();
            assert!(borrow.contains_key("a"));
            assert!(borrow.contains_key("b"));

            if let Some(JsValue::PlainObject(desc_a)) = borrow.get("a") {
                let desc_a = desc_a.borrow();
                assert_eq!(desc_a.get("value"), Some(&JsValue::Smi(1)));
                assert_eq!(desc_a.get("enumerable"), Some(&JsValue::Boolean(true)));
            } else {
                panic!("expected PlainObject for descriptor 'a'");
            }

            if let Some(JsValue::PlainObject(desc_b)) = borrow.get("b") {
                let desc_b = desc_b.borrow();
                assert_eq!(desc_b.get("value"), Some(&JsValue::Smi(2)));
                assert_eq!(desc_b.get("enumerable"), Some(&JsValue::Boolean(false)));
            } else {
                panic!("expected PlainObject for descriptor 'b'");
            }
        } else {
            panic!("expected PlainObject");
        }
    }

    #[test]
    fn test_e2e_get_own_property_descriptors_as_object_empty() {
        let obj = JsObject::new();
        let result = object_get_own_property_descriptors_as_object(&obj);
        if let JsValue::PlainObject(map) = result {
            assert!(map.borrow().is_empty());
        } else {
            panic!("expected PlainObject");
        }
    }

    // ── 7. Redefining properties ────────────────────────────────────────────

    #[test]
    fn test_e2e_configurable_property_allows_all_changes() {
        let mut obj = JsObject::new();
        obj.set_property("p", JsValue::Smi(1)).unwrap();

        obj.define_own_property("p", JsValue::Smi(2), PropertyAttributes::CONFIGURABLE)
            .unwrap();
        let (v, a) = obj.get_own_property_descriptor("p").unwrap();
        assert_eq!(v, JsValue::Smi(2));
        assert!(!a.contains(PropertyAttributes::WRITABLE));
        assert!(!a.contains(PropertyAttributes::ENUMERABLE));
        assert!(a.contains(PropertyAttributes::CONFIGURABLE));
    }

    #[test]
    fn test_e2e_nonconfigurable_allows_narrowing_writable() {
        let mut obj = JsObject::new();
        obj.define_own_property("p", JsValue::Smi(1), PropertyAttributes::WRITABLE)
            .unwrap();
        obj.define_own_property("p", JsValue::Smi(1), PropertyAttributes::empty())
            .unwrap();
        let (_, a) = obj.get_own_property_descriptor("p").unwrap();
        assert!(!a.contains(PropertyAttributes::WRITABLE));
    }

    #[test]
    fn test_e2e_nonconfigurable_rejects_widening_writable() {
        let mut obj = JsObject::new();
        obj.define_own_property("p", JsValue::Smi(1), PropertyAttributes::empty())
            .unwrap();
        let err = obj.define_own_property("p", JsValue::Smi(1), PropertyAttributes::WRITABLE);
        assert!(err.is_err());
    }

    #[test]
    fn test_e2e_nonconfigurable_nonwritable_same_value_ok() {
        let mut obj = JsObject::new();
        obj.define_own_property("p", JsValue::Smi(42), PropertyAttributes::empty())
            .unwrap();
        obj.define_own_property("p", JsValue::Smi(42), PropertyAttributes::empty())
            .unwrap();
    }

    #[test]
    fn test_e2e_nonconfigurable_nonwritable_different_value_rejected() {
        let mut obj = JsObject::new();
        obj.define_own_property("p", JsValue::Smi(1), PropertyAttributes::empty())
            .unwrap();
        let err = obj.define_own_property("p", JsValue::Smi(2), PropertyAttributes::empty());
        assert!(err.is_err());
    }

    // ── 8. Property order ───────────────────────────────────────────────────

    #[test]
    fn test_e2e_property_order_slow_mode_preserves_insertion() {
        let mut obj = JsObject::new();
        for i in 0..crate::objects::js_object::MAX_FAST_PROPERTIES {
            obj.set_property(&format!("p{i}"), JsValue::Smi(i as i32))
                .unwrap();
        }
        obj.set_property("extra1", JsValue::Smi(100)).unwrap();
        obj.set_property("extra2", JsValue::Smi(200)).unwrap();

        let keys = obj.own_property_keys();
        assert!(keys.contains(&"extra1".to_string()));
        assert!(keys.contains(&"extra2".to_string()));
        let i1 = keys.iter().position(|k| k == "extra1").unwrap();
        let i2 = keys.iter().position(|k| k == "extra2").unwrap();
        assert!(i1 < i2, "insertion order must be preserved in slow mode");
    }

    #[test]
    fn test_e2e_property_order_integer_indices_first_in_property_map() {
        use crate::objects::property_map::PropertyMap;
        let mut pm = PropertyMap::new();
        pm.insert("z".to_string(), JsValue::Smi(1));
        pm.insert("10".to_string(), JsValue::Smi(2));
        pm.insert("a".to_string(), JsValue::Smi(3));
        pm.insert("2".to_string(), JsValue::Smi(4));

        let keys: Vec<&str> = pm.keys().map(|s| s.as_str()).collect();
        assert_eq!(keys, vec!["2", "10", "z", "a"]);
    }

    // ── 9. Object.create with propertiesObject ──────────────────────────────

    #[test]
    fn test_e2e_create_with_properties_defines_props() {
        use crate::objects::property_map::PropertyMap;
        let proto = Rc::new(RefCell::new(JsObject::new()));

        let mut desc_x = PropertyMap::new();
        desc_x.insert("value".to_string(), JsValue::Smi(42));
        desc_x.insert("writable".to_string(), JsValue::Boolean(true));
        desc_x.insert("enumerable".to_string(), JsValue::Boolean(true));
        desc_x.insert("configurable".to_string(), JsValue::Boolean(true));

        let mut props_map = PropertyMap::new();
        props_map.insert(
            "x".to_string(),
            JsValue::PlainObject(Rc::new(RefCell::new(desc_x))),
        );
        let props = JsValue::PlainObject(Rc::new(RefCell::new(props_map)));

        let obj = object_create_with_properties(Some(proto), Some(&props)).unwrap();
        assert_eq!(obj.get_own_property("x"), Some(JsValue::Smi(42)));
        assert!(obj.prototype().is_some());
    }

    #[test]
    fn test_e2e_create_with_properties_null_proto() {
        use crate::objects::property_map::PropertyMap;
        let mut desc = PropertyMap::new();
        desc.insert("value".to_string(), JsValue::Smi(1));
        let mut props_map = PropertyMap::new();
        props_map.insert(
            "k".to_string(),
            JsValue::PlainObject(Rc::new(RefCell::new(desc))),
        );
        let props = JsValue::PlainObject(Rc::new(RefCell::new(props_map)));

        let obj = object_create_with_properties(None, Some(&props)).unwrap();
        assert!(obj.prototype().is_none());
        assert_eq!(obj.get_own_property("k"), Some(JsValue::Smi(1)));
    }

    #[test]
    fn test_e2e_create_with_no_properties() {
        let obj = object_create_with_properties(None, None).unwrap();
        assert!(obj.own_property_keys().is_empty());
    }

    #[test]
    fn test_e2e_create_with_properties_respects_attributes() {
        use crate::objects::property_map::PropertyMap;
        let mut desc = PropertyMap::new();
        desc.insert("value".to_string(), JsValue::Smi(7));
        desc.insert("writable".to_string(), JsValue::Boolean(false));
        desc.insert("enumerable".to_string(), JsValue::Boolean(false));
        desc.insert("configurable".to_string(), JsValue::Boolean(false));
        let mut props_map = PropertyMap::new();
        props_map.insert(
            "readonly".to_string(),
            JsValue::PlainObject(Rc::new(RefCell::new(desc))),
        );
        let props = JsValue::PlainObject(Rc::new(RefCell::new(props_map)));

        let obj = object_create_with_properties(None, Some(&props)).unwrap();
        let (_, attrs) = obj.get_own_property_descriptor("readonly").unwrap();
        assert!(!attrs.contains(PropertyAttributes::WRITABLE));
        assert!(!attrs.contains(PropertyAttributes::ENUMERABLE));
        assert!(!attrs.contains(PropertyAttributes::CONFIGURABLE));
    }

    // ── 10. Frozen / sealed property behaviour ──────────────────────────────

    #[test]
    fn test_e2e_frozen_object_rejects_define_property_value_change() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        object_freeze(&mut obj).unwrap();

        let err = obj.define_own_property("x", JsValue::Smi(2), PropertyAttributes::empty());
        assert!(err.is_err(), "frozen obj must reject value change");
    }

    #[test]
    fn test_e2e_frozen_object_rejects_new_property() {
        let mut obj = JsObject::new();
        object_freeze(&mut obj).unwrap();
        let err = obj.define_own_property("new", JsValue::Smi(1), PropertyAttributes::empty());
        assert!(err.is_err());
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_e2e_frozen_object_allows_same_value_define() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        object_freeze(&mut obj).unwrap();

        obj.define_own_property("x", JsValue::Smi(1), PropertyAttributes::empty())
            .unwrap();
    }

    #[test]
    fn test_e2e_frozen_object_rejects_writable_change() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        object_freeze(&mut obj).unwrap();

        let mut desc = PropertyMap::new();
        desc.insert("writable".to_string(), JsValue::Boolean(true));
        let d = JsValue::PlainObject(Rc::new(RefCell::new(desc)));

        let err = object_define_property_from_descriptor(&mut obj, "x", &d);
        assert!(err.is_err());
    }

    #[test]
    fn test_e2e_sealed_object_allows_value_change_on_writable() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        object_seal(&mut obj).unwrap();

        obj.set_property("x", JsValue::Smi(99)).unwrap();
        assert_eq!(obj.get_own_property("x"), Some(JsValue::Smi(99)));
    }

    #[test]
    fn test_e2e_sealed_object_rejects_new_property() {
        let mut obj = JsObject::new();
        object_seal(&mut obj).unwrap();
        let err = obj.set_property("new", JsValue::Smi(1));
        assert!(err.is_err());
    }

    #[test]
    fn test_e2e_sealed_object_rejects_delete() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        object_seal(&mut obj).unwrap();
        assert!(!obj.delete_own_property("x").unwrap());
    }

    // ── 11. Descriptor defaults ─────────────────────────────────────────────

    #[test]
    fn test_e2e_define_property_defaults_writable_false() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        let mut desc = PropertyMap::new();
        desc.insert("value".to_string(), JsValue::Smi(1));
        let d = JsValue::PlainObject(Rc::new(RefCell::new(desc)));
        object_define_property_from_descriptor(&mut obj, "p", &d).unwrap();

        let (_, attrs) = obj.get_own_property_descriptor("p").unwrap();
        assert!(!attrs.contains(PropertyAttributes::WRITABLE));
        assert!(!attrs.contains(PropertyAttributes::ENUMERABLE));
        assert!(!attrs.contains(PropertyAttributes::CONFIGURABLE));
    }

    #[test]
    fn test_e2e_set_property_defaults_all_true() {
        let mut obj = JsObject::new();
        obj.set_property("p", JsValue::Smi(1)).unwrap();

        let (_, attrs) = obj.get_own_property_descriptor("p").unwrap();
        assert!(attrs.contains(PropertyAttributes::WRITABLE));
        assert!(attrs.contains(PropertyAttributes::ENUMERABLE));
        assert!(attrs.contains(PropertyAttributes::CONFIGURABLE));
    }

    // ── 12. Generic descriptor ──────────────────────────────────────────────

    #[test]
    fn test_e2e_generic_descriptor_only_changes_specified_attrs() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();
        obj.set_property("p", JsValue::Smi(1)).unwrap();

        let mut desc = PropertyMap::new();
        desc.insert("enumerable".to_string(), JsValue::Boolean(false));
        let d = JsValue::PlainObject(Rc::new(RefCell::new(desc)));
        object_define_property_from_descriptor(&mut obj, "p", &d).unwrap();

        let (v, a) = obj.get_own_property_descriptor("p").unwrap();
        assert_eq!(v, JsValue::Smi(1));
        assert!(a.contains(PropertyAttributes::WRITABLE));
        assert!(!a.contains(PropertyAttributes::ENUMERABLE));
        assert!(a.contains(PropertyAttributes::CONFIGURABLE));
    }

    // ── 13. Non-configurable accessor — getter/setter immutability ──────────

    #[test]
    fn test_e2e_nonconfigurable_accessor_same_getter_setter_ok() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();

        let mut acc = PropertyMap::new();
        acc.insert("get".to_string(), JsValue::Boolean(true));
        acc.insert("set".to_string(), JsValue::Boolean(false));
        acc.insert("configurable".to_string(), JsValue::Boolean(false));
        let d = JsValue::PlainObject(Rc::new(RefCell::new(acc)));
        object_define_property_from_descriptor(&mut obj, "x", &d).unwrap();

        let mut acc2 = PropertyMap::new();
        acc2.insert("get".to_string(), JsValue::Boolean(true));
        acc2.insert("set".to_string(), JsValue::Boolean(false));
        let d2 = JsValue::PlainObject(Rc::new(RefCell::new(acc2)));
        object_define_property_from_descriptor(&mut obj, "x", &d2).unwrap();
    }

    #[test]
    fn test_e2e_nonconfigurable_accessor_rejects_setter_change() {
        use crate::objects::property_map::PropertyMap;
        let mut obj = JsObject::new();

        let mut acc = PropertyMap::new();
        acc.insert("get".to_string(), JsValue::Boolean(true));
        acc.insert("set".to_string(), JsValue::Boolean(false));
        acc.insert("configurable".to_string(), JsValue::Boolean(false));
        let d = JsValue::PlainObject(Rc::new(RefCell::new(acc)));
        object_define_property_from_descriptor(&mut obj, "x", &d).unwrap();

        let mut acc2 = PropertyMap::new();
        acc2.insert("get".to_string(), JsValue::Boolean(true));
        acc2.insert("set".to_string(), JsValue::Smi(999));
        let d2 = JsValue::PlainObject(Rc::new(RefCell::new(acc2)));
        let err = object_define_property_from_descriptor(&mut obj, "x", &d2);
        assert!(err.is_err());
    }

    // ── 14. Extensibility ───────────────────────────────────────────────────

    #[test]
    fn test_e2e_preventextensions_rejects_new_define() {
        let mut obj = JsObject::new();
        obj.prevent_extensions();
        let err = obj.define_own_property("x", JsValue::Smi(1), PropertyAttributes::empty());
        assert!(err.is_err());
    }

    #[test]
    fn test_e2e_preventextensions_allows_existing_redefine() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        obj.prevent_extensions();
        obj.define_own_property(
            "x",
            JsValue::Smi(2),
            PropertyAttributes::WRITABLE
                | PropertyAttributes::ENUMERABLE
                | PropertyAttributes::CONFIGURABLE,
        )
        .unwrap();
        assert_eq!(obj.get_own_property("x"), Some(JsValue::Smi(2)));
    }

    // ── 15. has_own vs prototype chain ──────────────────────────────────────

    #[test]
    fn test_e2e_object_keys_exclude_prototype_chain() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        proto
            .borrow_mut()
            .set_property("inherited", JsValue::Smi(0))
            .unwrap();
        let mut child = object_create(Some(proto));
        child.set_property("own", JsValue::Smi(1)).unwrap();

        let keys = object_keys(&child);
        assert_eq!(keys, vec!["own".to_string()]);
    }

    // ── 16. Slow-mode delete preserves order ────────────────────────────────

    #[test]
    fn test_e2e_slow_mode_delete_preserves_key_order() {
        let mut obj = JsObject::new();
        for i in 0..crate::objects::js_object::MAX_FAST_PROPERTIES + 2 {
            obj.set_property(&format!("p{i}"), JsValue::Smi(i as i32))
                .unwrap();
        }
        assert!(!obj.is_fast_mode());

        obj.delete_own_property("p0").unwrap();
        let keys = obj.own_property_keys();
        assert!(!keys.contains(&"p0".to_string()));
        assert_eq!(keys[0], "p1");
    }

    // ── 17. is_frozen / is_sealed comprehensive ─────────────────────────────

    #[test]
    fn test_e2e_is_frozen_requires_all_conditions() {
        let mut obj = JsObject::new();
        obj.set_property("a", JsValue::Smi(1)).unwrap();
        assert!(!object_is_frozen(&obj));

        obj.prevent_extensions();
        assert!(!object_is_frozen(&obj));

        let keys: Vec<String> = obj.own_property_keys();
        for key in keys {
            if let Some((value, _)) = obj.get_own_property_descriptor(&key) {
                obj.define_own_property(&key, value, PropertyAttributes::empty())
                    .unwrap();
            }
        }
        assert!(object_is_frozen(&obj));
    }

    #[test]
    fn test_e2e_frozen_is_also_sealed() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        object_freeze(&mut obj).unwrap();
        assert!(object_is_frozen(&obj));
        assert!(object_is_sealed(&obj));
    }

    // ── 18. defineProperty on new property sets correct attrs ──────────────

    #[test]
    fn test_e2e_define_own_property_new_prop_sets_exact_attrs() {
        let mut obj = JsObject::new();
        obj.define_own_property(
            "x",
            JsValue::Smi(1),
            PropertyAttributes::ENUMERABLE | PropertyAttributes::CONFIGURABLE,
        )
        .unwrap();
        let (_, attrs) = obj.get_own_property_descriptor("x").unwrap();
        assert!(!attrs.contains(PropertyAttributes::WRITABLE));
        assert!(attrs.contains(PropertyAttributes::ENUMERABLE));
        assert!(attrs.contains(PropertyAttributes::CONFIGURABLE));
    }

    // ── object_group_by ──────────────────────────────────────────────────────

    #[test]
    fn test_object_group_by_even_odd() {
        let items = vec![
            JsValue::Smi(1),
            JsValue::Smi(2),
            JsValue::Smi(3),
            JsValue::Smi(4),
        ];
        let groups = object_group_by(&items, |v, _| {
            if let JsValue::Smi(n) = v {
                if n % 2 == 0 { "even" } else { "odd" }
            } else {
                "other"
            }
            .to_string()
        });
        assert_eq!(groups.len(), 2);
        assert_eq!(groups["odd"].len(), 2);
        assert_eq!(groups["even"].len(), 2);
    }

    #[test]
    fn test_object_group_by_empty() {
        let items: Vec<JsValue> = vec![];
        let groups = object_group_by(&items, |_, _| "x".to_string());
        assert!(groups.is_empty());
    }

    #[test]
    fn test_object_group_by_single_group() {
        let items = vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)];
        let groups = object_group_by(&items, |_, _| "all".to_string());
        assert_eq!(groups.len(), 1);
        assert_eq!(groups["all"].len(), 3);
    }

    #[test]
    fn test_object_group_by_preserves_values() {
        let items = vec![
            JsValue::String("apple".into()),
            JsValue::String("avocado".into()),
            JsValue::String("banana".into()),
        ];
        let groups = object_group_by(&items, |v, _| {
            if let JsValue::String(s) = v {
                s.chars().next().unwrap_or('?').to_string()
            } else {
                "?".to_string()
            }
        });
        assert_eq!(groups.len(), 2);
        assert_eq!(groups["a"].len(), 2);
        assert_eq!(groups["b"].len(), 1);
    }

    #[test]
    fn test_object_group_by_callback_receives_index() {
        let items = vec![JsValue::Smi(10), JsValue::Smi(20), JsValue::Smi(30)];
        let mut indices = Vec::new();
        object_group_by(&items, |_, idx| {
            indices.push(idx);
            "g".to_string()
        });
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_object_group_by_same_key_all_items() {
        let items = vec![JsValue::Boolean(true), JsValue::Boolean(false)];
        let groups = object_group_by(&items, |_, _| "k".to_string());
        assert_eq!(groups["k"].len(), 2);
        assert_eq!(groups["k"][0], JsValue::Boolean(true));
        assert_eq!(groups["k"][1], JsValue::Boolean(false));
    }

    #[test]
    fn test_object_group_by_unique_keys() {
        let items = vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)];
        let groups = object_group_by(&items, |v, _| {
            if let JsValue::Smi(n) = v {
                n.to_string()
            } else {
                "?".to_string()
            }
        });
        assert_eq!(groups.len(), 3);
        assert_eq!(groups["1"].len(), 1);
        assert_eq!(groups["2"].len(), 1);
        assert_eq!(groups["3"].len(), 1);
    }

    #[test]
    fn test_object_group_by_with_undefined_values() {
        let items = vec![JsValue::Undefined, JsValue::Null, JsValue::Undefined];
        let groups = object_group_by(&items, |v, _| {
            if v.is_undefined() { "undef" } else { "other" }.to_string()
        });
        assert_eq!(groups["undef"].len(), 2);
        assert_eq!(groups["other"].len(), 1);
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  E2E Object.prototype and Object static methods conformance tests
    // ══════════════════════════════════════════════════════════════════════════

    // ── Object.prototype.hasOwnProperty ─────────────────────────────────────

    #[test]
    fn test_e2e_has_own_property_own_key_returns_true() {
        use crate::objects::property_map::PropertyMap;
        let mut map = PropertyMap::new();
        map.insert("x".to_string(), JsValue::Smi(1));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        assert!(object_prototype_has_own_property(&obj, &JsValue::String("x".into())).unwrap());
    }

    #[test]
    fn test_e2e_has_own_property_missing_key_returns_false() {
        use crate::objects::property_map::PropertyMap;
        let map = PropertyMap::new();
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        assert!(
            !object_prototype_has_own_property(&obj, &JsValue::String("missing".into())).unwrap()
        );
    }

    #[test]
    fn test_e2e_has_own_property_inherited_returns_false() {
        use crate::objects::property_map::PropertyMap;
        // Simulate: child PlainObject that doesn't have the key itself.
        let map = PropertyMap::new();
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        // hasOwnProperty only checks own, even though prototype may have it.
        assert!(
            !object_prototype_has_own_property(&obj, &JsValue::String("toString".into())).unwrap()
        );
    }

    #[test]
    fn test_e2e_has_own_property_symbol_key() {
        use crate::builtins::symbol::{symbol_create, symbol_to_property_key};
        use crate::objects::property_map::PropertyMap;
        let sym_id = symbol_create(Some("mySymbol".to_string()));
        let sym_key = symbol_to_property_key(sym_id);
        let mut map = PropertyMap::new();
        map.insert(sym_key, JsValue::Smi(42));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        assert!(object_prototype_has_own_property(&obj, &JsValue::Symbol(sym_id)).unwrap());
    }

    #[test]
    fn test_e2e_has_own_property_autoboxes_string_primitive() {
        // Calling hasOwnProperty on a string primitive auto-boxes it.
        // String wrapper has "0", "1", ... for character indices.
        let str_val = JsValue::String("hi".into());
        let obj = str_val.to_object().unwrap();
        // The wrapper should have [[PrimitiveValue]].
        assert!(
            object_prototype_has_own_property(&obj, &JsValue::String("[[PrimitiveValue]]".into()))
                .unwrap()
        );
    }

    #[test]
    fn test_e2e_has_own_property_null_throws() {
        let result =
            object_prototype_has_own_property(&JsValue::Null, &JsValue::String("x".into()));
        assert!(result.is_err());
    }

    #[test]
    fn test_e2e_has_own_property_undefined_throws() {
        let result =
            object_prototype_has_own_property(&JsValue::Undefined, &JsValue::String("x".into()));
        assert!(result.is_err());
    }

    #[test]
    fn test_e2e_has_own_property_number_key_coercion() {
        use crate::objects::property_map::PropertyMap;
        let mut map = PropertyMap::new();
        map.insert("42".to_string(), JsValue::Boolean(true));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        // Number 42 should coerce to "42".
        assert!(object_prototype_has_own_property(&obj, &JsValue::Smi(42)).unwrap());
    }

    // ── Object.prototype.isPrototypeOf ──────────────────────────────────────

    #[test]
    fn test_e2e_is_prototype_of_direct_proto() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        let child = JsObject::with_prototype(Rc::clone(&proto));
        assert!(object_prototype_is_prototype_of(&proto, &child));
    }

    #[test]
    fn test_e2e_is_prototype_of_indirect_proto() {
        let grandparent = Rc::new(RefCell::new(JsObject::new()));
        let parent = Rc::new(RefCell::new(JsObject::with_prototype(Rc::clone(
            &grandparent,
        ))));
        let child = JsObject::with_prototype(Rc::clone(&parent));
        assert!(object_prototype_is_prototype_of(&grandparent, &child));
    }

    #[test]
    fn test_e2e_is_prototype_of_returns_false_when_not_in_chain() {
        let unrelated = Rc::new(RefCell::new(JsObject::new()));
        let proto = Rc::new(RefCell::new(JsObject::new()));
        let child = JsObject::with_prototype(proto);
        assert!(!object_prototype_is_prototype_of(&unrelated, &child));
    }

    #[test]
    fn test_e2e_is_prototype_of_returns_false_for_null_prototype() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        let child = JsObject::new(); // null prototype
        assert!(!object_prototype_is_prototype_of(&proto, &child));
    }

    #[test]
    fn test_e2e_is_prototype_of_self_reference_returns_false() {
        // An object is not its own prototype.
        let obj = Rc::new(RefCell::new(JsObject::new()));
        assert!(!object_prototype_is_prototype_of(&obj, &obj.borrow()));
    }

    // ── Object.prototype.propertyIsEnumerable ───────────────────────────────

    #[test]
    fn test_e2e_property_is_enumerable_own_enumerable() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        assert!(object_prototype_property_is_enumerable(&obj, "x"));
    }

    #[test]
    fn test_e2e_property_is_enumerable_own_non_enumerable() {
        let mut obj = JsObject::new();
        obj.define_own_property("hidden", JsValue::Smi(1), PropertyAttributes::WRITABLE)
            .unwrap();
        assert!(!object_prototype_property_is_enumerable(&obj, "hidden"));
    }

    #[test]
    fn test_e2e_property_is_enumerable_inherited_returns_false() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        proto
            .borrow_mut()
            .set_property("inherited", JsValue::Smi(1))
            .unwrap();
        let child = object_create(Some(proto));
        // Even though "inherited" is enumerable on the prototype,
        // propertyIsEnumerable only checks own properties.
        assert!(!object_prototype_property_is_enumerable(
            &child,
            "inherited"
        ));
    }

    #[test]
    fn test_e2e_property_is_enumerable_missing_returns_false() {
        let obj = JsObject::new();
        assert!(!object_prototype_property_is_enumerable(&obj, "nope"));
    }

    #[test]
    fn test_e2e_property_is_enumerable_plain_object() {
        use crate::objects::property_map::PropertyMap;
        let mut map = PropertyMap::new();
        map.insert("a".to_string(), JsValue::Smi(1));
        map.insert_with_attrs(
            "b".to_string(),
            JsValue::Smi(2),
            PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE,
        );
        assert!(object_prototype_property_is_enumerable_plain(&map, "a"));
        assert!(!object_prototype_property_is_enumerable_plain(&map, "b"));
    }

    // ── Object.prototype.valueOf ────────────────────────────────────────────

    #[test]
    fn test_e2e_value_of_object_returns_itself() {
        use crate::objects::property_map::PropertyMap;
        let map = PropertyMap::new();
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        let result = object_prototype_value_of(&obj).unwrap();
        // PlainObject should return itself (same Rc).
        assert_eq!(result, obj);
    }

    #[test]
    fn test_e2e_value_of_number_wraps() {
        let val = JsValue::Smi(42);
        let result = object_prototype_value_of(&val).unwrap();
        // Should return a PlainObject wrapper.
        assert!(matches!(result, JsValue::PlainObject(_)));
        if let JsValue::PlainObject(map) = &result {
            assert_eq!(
                map.borrow().get("[[PrimitiveValue]]"),
                Some(&JsValue::Smi(42))
            );
        }
    }

    #[test]
    fn test_e2e_value_of_boolean_wraps() {
        let val = JsValue::Boolean(true);
        let result = object_prototype_value_of(&val).unwrap();
        assert!(matches!(result, JsValue::PlainObject(_)));
        if let JsValue::PlainObject(map) = &result {
            assert_eq!(
                map.borrow().get("[[PrimitiveValue]]"),
                Some(&JsValue::Boolean(true))
            );
        }
    }

    #[test]
    fn test_e2e_value_of_string_wraps() {
        let val = JsValue::String("hello".into());
        let result = object_prototype_value_of(&val).unwrap();
        assert!(matches!(result, JsValue::PlainObject(_)));
    }

    #[test]
    fn test_e2e_value_of_null_throws() {
        assert!(object_prototype_value_of(&JsValue::Null).is_err());
    }

    #[test]
    fn test_e2e_value_of_undefined_throws() {
        assert!(object_prototype_value_of(&JsValue::Undefined).is_err());
    }

    // ── Object.prototype.toLocaleString ─────────────────────────────────────

    #[test]
    fn test_e2e_to_locale_string_number() {
        let val = JsValue::Smi(42);
        assert_eq!(object_prototype_to_locale_string(&val).unwrap(), "42");
    }

    #[test]
    fn test_e2e_to_locale_string_string() {
        let val = JsValue::String("hello".into());
        assert_eq!(object_prototype_to_locale_string(&val).unwrap(), "hello");
    }

    #[test]
    fn test_e2e_to_locale_string_boolean() {
        assert_eq!(
            object_prototype_to_locale_string(&JsValue::Boolean(true)).unwrap(),
            "true"
        );
    }

    #[test]
    fn test_e2e_to_locale_string_null() {
        assert_eq!(
            object_prototype_to_locale_string(&JsValue::Null).unwrap(),
            "null"
        );
    }

    // ── Object.create(null) — no prototype ──────────────────────────────────

    #[test]
    fn test_e2e_create_null_has_no_prototype() {
        let obj = object_create(None);
        assert!(obj.prototype().is_none());
    }

    #[test]
    fn test_e2e_create_null_has_no_inherited_methods() {
        let obj = object_create(None);
        // A null-prototype object has no "toString" or "hasOwnProperty".
        assert!(!obj.has_own_property("toString"));
        assert!(!obj.has_own_property("hasOwnProperty"));
        assert!(obj.get_property("toString") == JsValue::Undefined);
    }

    #[test]
    fn test_e2e_create_null_can_add_properties() {
        let mut obj = object_create(None);
        obj.set_property("x", JsValue::Smi(42)).unwrap();
        assert_eq!(obj.get_own_property("x"), Some(JsValue::Smi(42)));
    }

    // ── Object.create(proto, properties) ────────────────────────────────────

    #[test]
    fn test_e2e_create_with_props_accessor_descriptor() {
        use crate::objects::property_map::PropertyMap;
        let mut desc = PropertyMap::new();
        desc.insert("get".to_string(), JsValue::Boolean(true));
        desc.insert("enumerable".to_string(), JsValue::Boolean(true));
        desc.insert("configurable".to_string(), JsValue::Boolean(true));
        let mut props_map = PropertyMap::new();
        props_map.insert(
            "acc".to_string(),
            JsValue::PlainObject(Rc::new(RefCell::new(desc))),
        );
        let props = JsValue::PlainObject(Rc::new(RefCell::new(props_map)));

        let obj = object_create_with_properties(None, Some(&props)).unwrap();
        // The accessor internals should be set up.
        assert!(obj.has_own_property("__get_acc__"));
    }

    // ── Object.assign — enumerable own + Symbol keys ────────────────────────

    #[test]
    fn test_e2e_assign_copies_symbol_keyed_properties() {
        use crate::builtins::symbol::{symbol_create, symbol_to_property_key};
        let sym_id = symbol_create(Some("testSym".to_string()));
        let sym_key = symbol_to_property_key(sym_id);

        let mut src = JsObject::new();
        src.set_property(&sym_key, JsValue::Smi(99)).unwrap();

        let mut target = JsObject::new();
        object_assign(&mut target, &[&src]).unwrap();
        assert_eq!(target.get_own_property(&sym_key), Some(JsValue::Smi(99)));
    }

    #[test]
    fn test_e2e_assign_does_not_copy_non_enumerable_symbol() {
        use crate::builtins::symbol::{symbol_create, symbol_to_property_key};
        let sym_id = symbol_create(Some("hidden".to_string()));
        let sym_key = symbol_to_property_key(sym_id);

        let mut src = JsObject::new();
        src.define_own_property(
            &sym_key,
            JsValue::Smi(1),
            PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE,
        )
        .unwrap();

        let mut target = JsObject::new();
        object_assign(&mut target, &[&src]).unwrap();
        assert!(!target.has_own_property(&sym_key));
    }

    #[test]
    fn test_e2e_assign_overwrites_in_order() {
        let mut target = JsObject::new();
        let mut s1 = JsObject::new();
        s1.set_property("a", JsValue::Smi(1)).unwrap();
        s1.set_property("b", JsValue::Smi(2)).unwrap();
        let mut s2 = JsObject::new();
        s2.set_property("b", JsValue::Smi(3)).unwrap();
        s2.set_property("c", JsValue::Smi(4)).unwrap();

        object_assign(&mut target, &[&s1, &s2]).unwrap();
        assert_eq!(target.get_own_property("a"), Some(JsValue::Smi(1)));
        assert_eq!(target.get_own_property("b"), Some(JsValue::Smi(3)));
        assert_eq!(target.get_own_property("c"), Some(JsValue::Smi(4)));
    }

    // ── Object.is — SameValue ───────────────────────────────────────────────

    #[test]
    fn test_e2e_object_is_nan_nan_true() {
        assert!(object_is(
            &JsValue::HeapNumber(f64::NAN),
            &JsValue::HeapNumber(f64::NAN)
        ));
    }

    #[test]
    fn test_e2e_object_is_pos_zero_neg_zero_false() {
        assert!(!object_is(
            &JsValue::HeapNumber(0.0),
            &JsValue::HeapNumber(-0.0)
        ));
    }

    #[test]
    fn test_e2e_object_is_neg_zero_pos_zero_false() {
        assert!(!object_is(
            &JsValue::HeapNumber(-0.0),
            &JsValue::HeapNumber(0.0)
        ));
    }

    #[test]
    fn test_e2e_object_is_pos_zero_pos_zero_true() {
        assert!(object_is(
            &JsValue::HeapNumber(0.0),
            &JsValue::HeapNumber(0.0)
        ));
    }

    #[test]
    fn test_e2e_object_is_smi_heap_number_cross_type() {
        // Smi(5) and HeapNumber(5.0) are the same value.
        assert!(object_is(&JsValue::Smi(5), &JsValue::HeapNumber(5.0)));
        assert!(object_is(&JsValue::HeapNumber(5.0), &JsValue::Smi(5)));
    }

    #[test]
    fn test_e2e_object_is_different_types_false() {
        assert!(!object_is(&JsValue::Smi(0), &JsValue::Boolean(false)));
        assert!(!object_is(&JsValue::String("0".into()), &JsValue::Smi(0)));
        assert!(!object_is(&JsValue::Null, &JsValue::Undefined));
    }

    // ── Object.fromEntries — duplicate keys last-wins ───────────────────────

    #[test]
    fn test_e2e_from_entries_duplicate_keys_last_wins() {
        let entries = vec![
            ("x".to_string(), JsValue::Smi(1)),
            ("x".to_string(), JsValue::Smi(2)),
            ("x".to_string(), JsValue::Smi(3)),
        ];
        let obj = object_from_entries(entries).unwrap();
        assert_eq!(obj.get_own_property("x"), Some(JsValue::Smi(3)));
    }

    #[test]
    fn test_e2e_from_entries_preserves_insertion_order() {
        let entries = vec![
            ("b".to_string(), JsValue::Smi(2)),
            ("a".to_string(), JsValue::Smi(1)),
            ("c".to_string(), JsValue::Smi(3)),
        ];
        let obj = object_from_entries(entries).unwrap();
        let keys = object_keys(&obj);
        assert_eq!(keys, vec!["b", "a", "c"]);
    }

    #[test]
    fn test_e2e_from_entries_properties_are_enumerable() {
        let entries = vec![("k".to_string(), JsValue::Smi(42))];
        let obj = object_from_entries(entries).unwrap();
        assert!(object_prototype_property_is_enumerable(&obj, "k"));
    }

    // ── Object.entries / Object.values — own enumerable string-keyed only ───

    #[test]
    fn test_e2e_entries_excludes_symbol_keys() {
        use crate::builtins::symbol::{symbol_create, symbol_to_property_key};
        let sym_id = symbol_create(Some("sym".to_string()));
        let sym_key = symbol_to_property_key(sym_id);

        let mut obj = JsObject::new();
        obj.set_property("str", JsValue::Smi(1)).unwrap();
        obj.set_property(&sym_key, JsValue::Smi(2)).unwrap();

        let entries = object_entries(&obj);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "str");
    }

    #[test]
    fn test_e2e_values_excludes_symbol_keys() {
        use crate::builtins::symbol::{symbol_create, symbol_to_property_key};
        let sym_id = symbol_create(Some("sym".to_string()));
        let sym_key = symbol_to_property_key(sym_id);

        let mut obj = JsObject::new();
        obj.set_property("str", JsValue::Smi(1)).unwrap();
        obj.set_property(&sym_key, JsValue::Smi(2)).unwrap();

        let vals = object_values(&obj);
        assert_eq!(vals.len(), 1);
        assert_eq!(vals[0], JsValue::Smi(1));
    }

    #[test]
    fn test_e2e_entries_excludes_non_enumerable() {
        let mut obj = JsObject::new();
        obj.set_property("a", JsValue::Smi(1)).unwrap();
        obj.define_own_property(
            "b",
            JsValue::Smi(2),
            PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE,
        )
        .unwrap();

        let entries = object_entries(&obj);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "a");
    }

    #[test]
    fn test_e2e_values_excludes_non_enumerable() {
        let mut obj = JsObject::new();
        obj.set_property("a", JsValue::Smi(1)).unwrap();
        obj.define_own_property(
            "b",
            JsValue::Smi(2),
            PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE,
        )
        .unwrap();

        let vals = object_values(&obj);
        assert_eq!(vals, vec![JsValue::Smi(1)]);
    }

    // ── Object.getOwnPropertyNames vs Object.getOwnPropertySymbols ──────────

    #[test]
    fn test_e2e_get_own_property_names_excludes_symbols() {
        use crate::builtins::symbol::{symbol_create, symbol_to_property_key};
        let sym_id = symbol_create(Some("s".to_string()));
        let sym_key = symbol_to_property_key(sym_id);

        let mut obj = JsObject::new();
        obj.set_property("str", JsValue::Smi(1)).unwrap();
        obj.set_property(&sym_key, JsValue::Smi(2)).unwrap();

        let names = object_get_own_property_names(&obj);
        assert!(names.contains(&"str".to_string()));
        assert!(!names.iter().any(|k| is_symbol_property_key(k)));
    }

    #[test]
    fn test_e2e_get_own_property_symbols_excludes_strings() {
        use crate::builtins::symbol::{symbol_create, symbol_to_property_key};
        let sym_id = symbol_create(Some("s".to_string()));
        let sym_key = symbol_to_property_key(sym_id);

        let mut obj = JsObject::new();
        obj.set_property("str", JsValue::Smi(1)).unwrap();
        obj.set_property(&sym_key, JsValue::Smi(2)).unwrap();

        let symbols = object_get_own_property_symbols(&obj);
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0], JsValue::Symbol(sym_id));
    }

    #[test]
    fn test_e2e_get_own_property_names_includes_non_enumerable() {
        let mut obj = JsObject::new();
        obj.set_property("vis", JsValue::Smi(1)).unwrap();
        obj.define_own_property("hid", JsValue::Smi(2), PropertyAttributes::WRITABLE)
            .unwrap();

        let names = object_get_own_property_names(&obj);
        assert!(names.contains(&"vis".to_string()));
        assert!(names.contains(&"hid".to_string()));
    }

    #[test]
    fn test_e2e_get_own_property_symbols_empty_when_no_symbols() {
        let mut obj = JsObject::new();
        obj.set_property("a", JsValue::Smi(1)).unwrap();
        assert!(object_get_own_property_symbols(&obj).is_empty());
    }

    // ── Combined workflow tests ─────────────────────────────────────────────

    #[test]
    fn test_e2e_create_null_then_has_own_property() {
        let mut obj = object_create(None);
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        assert!(object_has_own(&obj, "x"));
        assert!(!object_has_own(&obj, "toString"));
    }

    #[test]
    fn test_e2e_assign_then_entries() {
        let mut target = JsObject::new();
        let mut src = JsObject::new();
        src.set_property("a", JsValue::Smi(1)).unwrap();
        src.set_property("b", JsValue::Smi(2)).unwrap();
        object_assign(&mut target, &[&src]).unwrap();

        let entries = object_entries(&target);
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_e2e_from_entries_then_values() {
        let entries = vec![
            ("x".to_string(), JsValue::Smi(10)),
            ("y".to_string(), JsValue::Smi(20)),
        ];
        let obj = object_from_entries(entries).unwrap();
        let vals = object_values(&obj);
        assert!(vals.contains(&JsValue::Smi(10)));
        assert!(vals.contains(&JsValue::Smi(20)));
    }

    #[test]
    fn test_e2e_freeze_then_property_is_enumerable() {
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        object_freeze(&mut obj).unwrap();
        // The property remains enumerable after freezing.
        assert!(object_prototype_property_is_enumerable(&obj, "x"));
    }

    #[test]
    fn test_e2e_value_of_array_returns_self() {
        let arr = JsValue::new_array(vec![JsValue::Smi(1), JsValue::Smi(2)]);
        let result = object_prototype_value_of(&arr).unwrap();
        assert!(matches!(result, JsValue::Array(_)));
    }

    #[test]
    fn test_e2e_to_locale_string_undefined() {
        assert_eq!(
            object_prototype_to_locale_string(&JsValue::Undefined).unwrap(),
            "undefined"
        );
    }

    #[test]
    fn test_e2e_to_locale_string_heap_number() {
        let val = JsValue::HeapNumber(3.14);
        assert_eq!(object_prototype_to_locale_string(&val).unwrap(), "3.14");
    }

    #[test]
    fn test_e2e_object_is_bigint_equality() {
        assert!(object_is(&JsValue::BigInt(42), &JsValue::BigInt(42)));
        assert!(!object_is(&JsValue::BigInt(1), &JsValue::BigInt(2)));
    }

    #[test]
    fn test_e2e_object_is_symbol_equality() {
        use crate::builtins::symbol::symbol_create;
        let s1 = symbol_create(Some("a".to_string()));
        let s2 = symbol_create(Some("a".to_string()));
        assert!(object_is(&JsValue::Symbol(s1), &JsValue::Symbol(s1)));
        assert!(!object_is(&JsValue::Symbol(s1), &JsValue::Symbol(s2)));
    }
}
