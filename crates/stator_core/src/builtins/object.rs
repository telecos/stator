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

use crate::error::{StatorError, StatorResult};
use crate::objects::js_object::JsObject;
use crate::objects::map::PropertyAttributes;
use crate::objects::property_descriptor::FullPropertyDescriptor;
use crate::objects::value::JsValue;

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
            // Only copy enumerable own properties.
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
            obj.get_own_property_descriptor(k)
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
        .filter_map(|k| obj.get_own_property_descriptor(&k).map(|(v, a)| (k, v, a)))
        .collect()
}

// ── Object.getOwnPropertySymbols ─────────────────────────────────────────────

/// ECMAScript §20.1.2.11 `Object.getOwnPropertySymbols(obj)`.
///
/// Returns an array of all own symbol-keyed properties of `obj`.
/// Currently returns an empty `Vec` since `JsObject` does not track
/// symbol-keyed properties.
pub fn object_get_own_property_symbols(_obj: &JsObject) -> Vec<JsValue> {
    Vec::new()
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
    let attrs = if let Some((_, current_attrs)) = obj.get_own_property_descriptor(key) {
        desc.validate_against(key, current_attrs)?
    } else {
        desc.to_attributes()
    };

    let value = match &desc {
        FullPropertyDescriptor::Data { value, .. } => value.clone(),
        FullPropertyDescriptor::Accessor { .. } | FullPropertyDescriptor::Generic { .. } => {
            // For accessor/generic descriptors, preserve the existing value
            // if present, otherwise use undefined.
            obj.get_own_property(key).unwrap_or(JsValue::Undefined)
        }
    };
    obj.define_own_property(key, value, attrs)
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
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
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
/// Wraps the internal `(value, attributes)` pair into a
/// [`FullPropertyDescriptor::Data`] and converts it to a
/// [`JsValue::PlainObject`] with `value`, `writable`, `enumerable`, and
/// `configurable` keys.
pub fn object_get_own_property_descriptor_as_object(obj: &JsObject, key: &str) -> Option<JsValue> {
    obj.get_own_property_descriptor(key).map(|(value, attrs)| {
        let desc = FullPropertyDescriptor::Data {
            value,
            writable: attrs.contains(PropertyAttributes::WRITABLE),
            enumerable: attrs.contains(PropertyAttributes::ENUMERABLE),
            configurable: attrs.contains(PropertyAttributes::CONFIGURABLE),
        };
        desc.to_object()
    })
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
    fn test_get_own_property_symbols_returns_empty() {
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
}
