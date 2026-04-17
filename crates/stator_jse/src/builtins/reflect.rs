//! ECMAScript §28 `Reflect` built-in.
//!
//! `Reflect` is a plain object whose properties are functions that correspond
//! exactly to the 13 fundamental object operations (the *essential internal
//! methods*) defined in ECMAScript §10.  Each function is a pure, trap-free
//! invocation of the underlying [`JsObject`] internal method.
//!
//! # Naming convention
//!
//! Each function is prefixed `reflect_` to mirror the ECMAScript `Reflect.*`
//! spelling and to avoid ambiguity with standard-library items.
//!
//! # Receiver-aware overloads
//!
//! The `_with_receiver` variants of `reflect_get` and `reflect_set` accept an
//! additional *receiver* argument.  When the resolved property turns out to be
//! an accessor, the receiver is used as the `this` value for the getter / setter
//! call — matching the ECMAScript spec semantics for `Reflect.get` / `Reflect.set`.
//!
//! # References
//!
//! * ECMAScript 2025 Language Specification §28 — *The Reflect Object*

use std::cell::RefCell;
use std::rc::Rc;

use crate::error::{StatorError, StatorResult};
use crate::objects::js_object::JsObject;
use crate::objects::map::PropertyAttributes;
use crate::objects::property_map::PropertyMap;
use crate::objects::value::JsValue;

// ── reflect_get ───────────────────────────────────────────────────────────────

/// ECMAScript §28.1.4 `Reflect.get(target, propertyKey)`.
///
/// Equivalent to `target[propertyKey]`.  Walks the prototype chain.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::reflect::reflect_get;
/// use stator_jse::objects::js_object::JsObject;
/// use stator_jse::objects::value::JsValue;
///
/// let mut target = JsObject::new();
/// target.set_property("x", JsValue::Smi(42)).unwrap();
/// assert_eq!(reflect_get(&target, "x"), JsValue::Smi(42));
/// assert_eq!(reflect_get(&target, "missing"), JsValue::Undefined);
/// ```
pub fn reflect_get(target: &JsObject, key: &str) -> JsValue {
    target.get_property(key)
}

/// ECMAScript §28.1.4 `Reflect.get(target, propertyKey, receiver)`.
///
/// Like [`reflect_get`] but stores the *receiver* so that, when the resolved
/// property is an accessor, the getter is invoked with `receiver` as `this`.
/// The basic [`JsObject`] layer does not dispatch accessor calls, so this
/// falls back to [`reflect_get`] — the full receiver-aware path is handled at
/// the JS-level wrapper in `install_globals`.
pub fn reflect_get_with_receiver(target: &JsObject, key: &str, _receiver: &JsValue) -> JsValue {
    target.get_property(key)
}

// ── reflect_set ───────────────────────────────────────────────────────────────

/// ECMAScript §28.1.9 `Reflect.set(target, propertyKey, value)`.
///
/// Equivalent to `target[propertyKey] = value`.  Returns `true` on success.
///
/// Returns `Ok(false)` if the property is non-writable or the object is
/// non-extensible, propagating the error as a boolean rather than throwing
/// (consistent with the ECMAScript spec which returns a boolean).
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::reflect::reflect_set;
/// use stator_jse::objects::js_object::JsObject;
/// use stator_jse::objects::value::JsValue;
///
/// let mut target = JsObject::new();
/// assert!(reflect_set(&mut target, "x", JsValue::Smi(1)).unwrap());
/// assert_eq!(target.get_property("x"), JsValue::Smi(1));
/// ```
pub fn reflect_set(target: &mut JsObject, key: &str, value: JsValue) -> StatorResult<bool> {
    Ok(target.set_property(key, value).is_ok())
}

/// ECMAScript §28.1.9 `Reflect.set(target, propertyKey, value, receiver)`.
///
/// Like [`reflect_set`] but stores the *receiver* for accessor dispatch.
/// At the `JsObject` layer this falls back to [`reflect_set`]; the full
/// receiver-aware path is handled at the JS-level wrapper.
pub fn reflect_set_with_receiver(
    target: &mut JsObject,
    key: &str,
    value: JsValue,
    _receiver: &JsValue,
) -> StatorResult<bool> {
    Ok(target.set_property(key, value).is_ok())
}

// ── reflect_has ───────────────────────────────────────────────────────────────

/// ECMAScript §28.1.5 `Reflect.has(target, propertyKey)`.
///
/// Equivalent to the `in` operator: `propertyKey in target`.  Walks the
/// prototype chain.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::reflect::reflect_has;
/// use stator_jse::objects::js_object::JsObject;
/// use stator_jse::objects::value::JsValue;
///
/// let mut target = JsObject::new();
/// target.set_property("y", JsValue::Boolean(true)).unwrap();
/// assert!(reflect_has(&target, "y"));
/// assert!(!reflect_has(&target, "z"));
/// ```
pub fn reflect_has(target: &JsObject, key: &str) -> bool {
    target.has_property(key)
}

// ── reflect_delete_property ───────────────────────────────────────────────────

/// ECMAScript §28.1.3 `Reflect.deleteProperty(target, propertyKey)`.
///
/// Equivalent to `delete target[propertyKey]`.  Returns `true` if the
/// property was deleted or did not exist, `false` if it is non-configurable.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::reflect::reflect_delete_property;
/// use stator_jse::objects::js_object::JsObject;
/// use stator_jse::objects::value::JsValue;
///
/// let mut target = JsObject::new();
/// target.set_property("p", JsValue::Smi(1)).unwrap();
/// assert!(reflect_delete_property(&mut target, "p").unwrap());
/// assert!(!target.has_own_property("p"));
/// ```
pub fn reflect_delete_property(target: &mut JsObject, key: &str) -> StatorResult<bool> {
    target.delete_own_property(key)
}

// ── reflect_define_property ───────────────────────────────────────────────────

/// ECMAScript §28.1.3 `Reflect.defineProperty(target, propertyKey, attributes)`.
///
/// Defines or redefines an own property on `target` with explicit
/// [`PropertyAttributes`] flags.  Returns `true` on success.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::reflect::reflect_define_property;
/// use stator_jse::objects::js_object::JsObject;
/// use stator_jse::objects::map::PropertyAttributes;
/// use stator_jse::objects::value::JsValue;
///
/// let mut target = JsObject::new();
/// assert!(reflect_define_property(
///     &mut target, "c", JsValue::Smi(9),
///     PropertyAttributes::WRITABLE | PropertyAttributes::ENUMERABLE,
/// ).unwrap());
/// ```
pub fn reflect_define_property(
    target: &mut JsObject,
    key: &str,
    value: JsValue,
    attributes: PropertyAttributes,
) -> StatorResult<bool> {
    Ok(target.define_own_property(key, value, attributes).is_ok())
}

// ── reflect_get_own_property_descriptor ──────────────────────────────────────

/// ECMAScript §28.1.6 `Reflect.getOwnPropertyDescriptor(target, propertyKey)`.
///
/// Returns `Some((value, attributes))` if `target` has an own property named
/// `key`, or `None` otherwise.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::reflect::reflect_get_own_property_descriptor;
/// use stator_jse::objects::js_object::JsObject;
/// use stator_jse::objects::map::PropertyAttributes;
/// use stator_jse::objects::value::JsValue;
///
/// let mut target = JsObject::new();
/// target.set_property("k", JsValue::Smi(7)).unwrap();
/// assert!(reflect_get_own_property_descriptor(&target, "k").is_some());
/// assert!(reflect_get_own_property_descriptor(&target, "nope").is_none());
/// ```
pub fn reflect_get_own_property_descriptor(
    target: &JsObject,
    key: &str,
) -> Option<(JsValue, PropertyAttributes)> {
    target.get_own_property_descriptor(key)
}

// ── reflect_get_prototype_of ──────────────────────────────────────────────────

/// ECMAScript §28.1.7 `Reflect.getPrototypeOf(target)`.
///
/// Returns the `[[Prototype]]` of `target`, or `None` for objects with a
/// `null` prototype.
///
/// # Examples
///
/// ```
/// use std::cell::RefCell;
/// use std::rc::Rc;
/// use stator_jse::builtins::reflect::reflect_get_prototype_of;
/// use stator_jse::objects::js_object::JsObject;
///
/// let target = JsObject::new();
/// assert!(reflect_get_prototype_of(&target).is_none());
/// ```
pub fn reflect_get_prototype_of(target: &JsObject) -> Option<Rc<RefCell<JsObject>>> {
    target.prototype().cloned()
}

// ── reflect_set_prototype_of ──────────────────────────────────────────────────

/// ECMAScript §28.1.11 `Reflect.setPrototypeOf(target, proto)`.
///
/// Sets the `[[Prototype]]` of `target` to `proto`.  Returns `true` on
/// success, `false` if the object is non-extensible and the prototype
/// differs from the current one.
///
/// # Examples
///
/// ```
/// use std::cell::RefCell;
/// use std::rc::Rc;
/// use stator_jse::builtins::reflect::reflect_set_prototype_of;
/// use stator_jse::objects::js_object::JsObject;
///
/// let proto = Rc::new(RefCell::new(JsObject::new()));
/// let mut target = JsObject::new();
/// assert!(reflect_set_prototype_of(&mut target, Some(Rc::clone(&proto))));
/// ```
pub fn reflect_set_prototype_of(
    target: &mut JsObject,
    proto: Option<Rc<RefCell<JsObject>>>,
) -> bool {
    use crate::builtins::object::object_set_prototype_of;
    object_set_prototype_of(target, proto).is_ok()
}

// ── reflect_is_extensible ─────────────────────────────────────────────────────

/// ECMAScript §28.1.8 `Reflect.isExtensible(target)`.
///
/// Returns `true` if new own properties may be added to `target`.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::reflect::reflect_is_extensible;
/// use stator_jse::objects::js_object::JsObject;
///
/// let mut target = JsObject::new();
/// assert!(reflect_is_extensible(&target));
/// target.prevent_extensions();
/// assert!(!reflect_is_extensible(&target));
/// ```
pub fn reflect_is_extensible(target: &JsObject) -> bool {
    target.is_extensible()
}

// ── reflect_prevent_extensions ───────────────────────────────────────────────

/// ECMAScript §28.1.10 `Reflect.preventExtensions(target)`.
///
/// Marks `target` as non-extensible.  Always returns `true`.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::reflect::reflect_prevent_extensions;
/// use stator_jse::objects::js_object::JsObject;
///
/// let mut target = JsObject::new();
/// assert!(reflect_prevent_extensions(&mut target));
/// assert!(!target.is_extensible());
/// ```
pub fn reflect_prevent_extensions(target: &mut JsObject) -> bool {
    target.prevent_extensions();
    true
}

// ── reflect_own_keys ──────────────────────────────────────────────────────────

/// ECMAScript §28.1.12 `Reflect.ownKeys(target)`.
///
/// Returns all own property keys (string-keyed) of `target` as a `Vec<String>`.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::reflect::reflect_own_keys;
/// use stator_jse::objects::js_object::JsObject;
/// use stator_jse::objects::value::JsValue;
///
/// let mut target = JsObject::new();
/// target.set_property("a", JsValue::Smi(1)).unwrap();
/// target.set_property("b", JsValue::Smi(2)).unwrap();
/// let keys = reflect_own_keys(&target);
/// assert_eq!(keys.len(), 2);
/// assert!(keys.contains(&"a".to_string()));
/// assert!(keys.contains(&"b".to_string()));
/// ```
pub fn reflect_own_keys(target: &JsObject) -> Vec<String> {
    target.own_property_keys()
}

// ── reflect_apply ─────────────────────────────────────────────────────────────

/// ECMAScript §28.1.1 `Reflect.apply(target, thisArgument, argumentsList)`.
///
/// In the pure-Rust engine model the target is represented as a callable
/// Rust closure rather than a [`JsValue::Function`].  `this_argument` is
/// passed through for completeness; the built-in layer does not use it.
///
/// Returns an error if invoking the target closure returns an error.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::reflect::reflect_apply;
/// use stator_jse::objects::value::JsValue;
///
/// let result = reflect_apply(
///     |_this, args| Ok(args.first().cloned().unwrap_or(JsValue::Undefined)),
///     JsValue::Undefined,
///     vec![JsValue::Smi(7)],
/// ).unwrap();
/// assert_eq!(result, JsValue::Smi(7));
/// ```
pub fn reflect_apply(
    target: impl Fn(JsValue, Vec<JsValue>) -> crate::error::StatorResult<JsValue>,
    this_argument: JsValue,
    arguments_list: Vec<JsValue>,
) -> crate::error::StatorResult<JsValue> {
    target(this_argument, arguments_list)
}

// ── reflect_construct ─────────────────────────────────────────────────────────

/// ECMAScript §28.1.2 `Reflect.construct(target, argumentsList [, newTarget])`.
///
/// In the pure-Rust engine model the constructor is a Rust closure that
/// receives the argument list and returns a new [`JsObject`].  The optional
/// `new_target` parameter determines which prototype to use; when omitted the
/// target itself acts as `newTarget`.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::reflect::reflect_construct;
/// use stator_jse::objects::js_object::JsObject;
/// use stator_jse::objects::value::JsValue;
///
/// let obj = reflect_construct(
///     |args| {
///         let mut o = JsObject::new();
///         if let Some(v) = args.first() {
///             o.set_property("val", v.clone()).unwrap();
///         }
///         Ok(o)
///     },
///     vec![JsValue::Smi(99)],
/// ).unwrap();
/// assert_eq!(obj.get_property("val"), JsValue::Smi(99));
/// ```
pub fn reflect_construct(
    target: impl Fn(Vec<JsValue>) -> crate::error::StatorResult<JsObject>,
    arguments_list: Vec<JsValue>,
) -> crate::error::StatorResult<JsObject> {
    target(arguments_list)
}

/// ECMAScript §28.1.2 `Reflect.construct(target, argumentsList, newTarget)`.
///
/// Like [`reflect_construct`] but accepts a separate `new_target` constructor
/// whose `prototype` property is set on the created instance.  This mirrors
/// the `Reflect.construct(target, args, newTarget)` semantics where the
/// created object inherits from `newTarget.prototype` instead of
/// `target.prototype`.
pub fn reflect_construct_with_new_target(
    target: impl Fn(Vec<JsValue>) -> crate::error::StatorResult<JsObject>,
    arguments_list: Vec<JsValue>,
    new_target_proto: Option<Rc<RefCell<JsObject>>>,
) -> crate::error::StatorResult<JsObject> {
    let mut result = target(arguments_list)?;
    if let Some(proto) = new_target_proto {
        result.set_prototype(Some(proto));
    }
    Ok(result)
}

// ── JsValue-based bridge functions ────────────────────────────────────────────
//
// The Proxy layer represents prototypes as `JsValue` (either `Null` or an
// object-like variant), while the Reflect/JsObject layer uses
// `Option<Rc<RefCell<JsObject>>>`.  The bridge functions below convert between
// the two representations so that round-trip conformance tests can compare
// Proxy and Reflect results directly.

/// Returns the prototype of `target` as a [`JsValue`].
///
/// Returns [`JsValue::Null`] when the target has no prototype, or a
/// [`JsValue::PlainObject`] snapshot of the prototype's own properties.
///
/// This mirrors the proxy's `[[GetPrototypeOf]]` default path so that
/// `proxy_get_prototype_of` (no trap) and `reflect_get_prototype_of_value`
/// agree on the representation.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::reflect::reflect_get_prototype_of_value;
/// use stator_jse::objects::js_object::JsObject;
/// use stator_jse::objects::value::JsValue;
///
/// let target = JsObject::new();
/// assert_eq!(reflect_get_prototype_of_value(&target), JsValue::Null);
/// ```
pub fn reflect_get_prototype_of_value(target: &JsObject) -> JsValue {
    match target.prototype() {
        Some(proto) => {
            let borrowed = proto.borrow();
            let mut map = PropertyMap::new();
            for key in borrowed.own_property_keys() {
                if let Some(val) = borrowed.get_own_property(&key) {
                    map.insert(key, val);
                }
            }
            JsValue::PlainObject(Rc::new(RefCell::new(map)))
        }
        None => JsValue::Null,
    }
}

/// Sets the prototype of `target` from a [`JsValue`].
///
/// Accepts [`JsValue::Null`] to remove the prototype, or any object-like
/// [`JsValue`] to set one.  Returns `Ok(true)` on success, `Ok(false)` if
/// the target is non-extensible with a different prototype, or `Err` for
/// non-object, non-null values.
///
/// This mirrors the proxy's `[[SetPrototypeOf]]` default path so that
/// `proxy_set_prototype_of` (no trap) and `reflect_set_prototype_of_value`
/// agree on the representation.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::reflect::reflect_set_prototype_of_value;
/// use stator_jse::objects::js_object::JsObject;
/// use stator_jse::objects::value::JsValue;
///
/// let mut target = JsObject::new();
/// assert!(reflect_set_prototype_of_value(&mut target, JsValue::Null).unwrap());
/// ```
pub fn reflect_set_prototype_of_value(target: &mut JsObject, proto: JsValue) -> StatorResult<bool> {
    use crate::builtins::object::object_set_prototype_of;
    if matches!(proto, JsValue::Null) {
        Ok(object_set_prototype_of(target, None).is_ok())
    } else if proto.is_object_like() {
        let mut js_obj = JsObject::new();
        if let JsValue::PlainObject(map) = &proto {
            let borrowed = map.borrow();
            for (key, val) in borrowed.iter() {
                let _ = js_obj.set_property(key, val.clone());
            }
        }
        Ok(object_set_prototype_of(target, Some(Rc::new(RefCell::new(js_obj)))).is_ok())
    } else {
        Err(StatorError::TypeError(
            "Object prototype may only be an Object or null".to_string(),
        ))
    }
}

/// Returns all own property keys of `target` as [`JsValue`] entries.
///
/// Each string key becomes [`JsValue::String`].  Symbol keys are detected by
/// checking for the internal `Symbol(…)` property-key encoding and returned
/// as [`JsValue::Symbol`].
///
/// This mirrors the proxy's `[[OwnPropertyKeys]]` default path so that
/// `proxy_own_keys` (no trap) and `reflect_own_keys_values` produce the
/// same representation.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::reflect::reflect_own_keys_values;
/// use stator_jse::objects::js_object::JsObject;
/// use stator_jse::objects::value::JsValue;
///
/// let mut target = JsObject::new();
/// target.set_property("a", JsValue::Smi(1)).unwrap();
/// let keys = reflect_own_keys_values(&target);
/// assert_eq!(keys, vec![JsValue::String("a".into())]);
/// ```
pub fn reflect_own_keys_values(target: &JsObject) -> Vec<JsValue> {
    target
        .own_property_keys()
        .into_iter()
        .map(|key| {
            if let Some(id) = crate::builtins::symbol::property_key_to_symbol(&key) {
                JsValue::Symbol(id)
            } else {
                JsValue::String(key.into())
            }
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use crate::objects::js_object::JsObject;
    use crate::objects::map::PropertyAttributes;
    use crate::objects::value::JsValue;

    // ── reflect_get ──────────────────────────────────────────────────────────

    #[test]
    fn test_reflect_get_own_property() {
        let mut t = JsObject::new();
        t.set_property("x", JsValue::Smi(5)).unwrap();
        assert_eq!(reflect_get(&t, "x"), JsValue::Smi(5));
    }

    #[test]
    fn test_reflect_get_missing_returns_undefined() {
        let t = JsObject::new();
        assert_eq!(reflect_get(&t, "nope"), JsValue::Undefined);
    }

    #[test]
    fn test_reflect_get_walks_prototype_chain() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        proto
            .borrow_mut()
            .set_property("inherited", JsValue::Smi(10))
            .unwrap();
        let child = JsObject::with_prototype(proto);
        assert_eq!(reflect_get(&child, "inherited"), JsValue::Smi(10));
    }

    #[test]
    fn test_reflect_get_with_receiver_returns_value() {
        let mut t = JsObject::new();
        t.set_property("k", JsValue::Smi(77)).unwrap();
        let receiver = JsValue::Undefined;
        assert_eq!(
            reflect_get_with_receiver(&t, "k", &receiver),
            JsValue::Smi(77)
        );
    }

    #[test]
    fn test_reflect_get_with_receiver_missing_returns_undefined() {
        let t = JsObject::new();
        let receiver = JsValue::Undefined;
        assert_eq!(
            reflect_get_with_receiver(&t, "nope", &receiver),
            JsValue::Undefined
        );
    }

    // ── reflect_set ──────────────────────────────────────────────────────────

    #[test]
    fn test_reflect_set_creates_property() {
        let mut t = JsObject::new();
        assert!(reflect_set(&mut t, "k", JsValue::Smi(99)).unwrap());
        assert_eq!(t.get_property("k"), JsValue::Smi(99));
    }

    #[test]
    fn test_reflect_set_non_writable_returns_false() {
        let mut t = JsObject::new();
        t.define_own_property("ro", JsValue::Smi(1), PropertyAttributes::empty())
            .unwrap();
        assert!(!reflect_set(&mut t, "ro", JsValue::Smi(2)).unwrap());
    }

    #[test]
    fn test_reflect_set_with_receiver_creates_property() {
        let mut t = JsObject::new();
        let receiver = JsValue::Undefined;
        assert!(reflect_set_with_receiver(&mut t, "k", JsValue::Smi(50), &receiver).unwrap());
        assert_eq!(t.get_property("k"), JsValue::Smi(50));
    }

    #[test]
    fn test_reflect_set_overwrites_existing() {
        let mut t = JsObject::new();
        t.set_property("k", JsValue::Smi(1)).unwrap();
        assert!(reflect_set(&mut t, "k", JsValue::Smi(2)).unwrap());
        assert_eq!(t.get_property("k"), JsValue::Smi(2));
    }

    // ── reflect_has ──────────────────────────────────────────────────────────

    #[test]
    fn test_reflect_has_own_property() {
        let mut t = JsObject::new();
        t.set_property("p", JsValue::Null).unwrap();
        assert!(reflect_has(&t, "p"));
        assert!(!reflect_has(&t, "q"));
    }

    #[test]
    fn test_reflect_has_prototype_chain() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        proto
            .borrow_mut()
            .set_property("up", JsValue::Boolean(true))
            .unwrap();
        let child = JsObject::with_prototype(proto);
        assert!(reflect_has(&child, "up"));
    }

    #[test]
    fn test_reflect_has_empty_object() {
        let t = JsObject::new();
        assert!(!reflect_has(&t, "anything"));
    }

    // ── reflect_delete_property ───────────────────────────────────────────────

    #[test]
    fn test_reflect_delete_existing_property() {
        let mut t = JsObject::new();
        t.set_property("d", JsValue::Smi(1)).unwrap();
        assert!(reflect_delete_property(&mut t, "d").unwrap());
        assert!(!t.has_own_property("d"));
    }

    #[test]
    fn test_reflect_delete_non_configurable_returns_false() {
        let mut t = JsObject::new();
        t.define_own_property("nc", JsValue::Smi(1), PropertyAttributes::WRITABLE)
            .unwrap();
        assert!(!reflect_delete_property(&mut t, "nc").unwrap());
    }

    #[test]
    fn test_reflect_delete_missing_property_returns_true() {
        let mut t = JsObject::new();
        assert!(reflect_delete_property(&mut t, "ghost").unwrap());
    }

    #[test]
    fn test_reflect_delete_configurable_property() {
        let mut t = JsObject::new();
        t.define_own_property(
            "cfg",
            JsValue::Smi(1),
            PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE,
        )
        .unwrap();
        assert!(reflect_delete_property(&mut t, "cfg").unwrap());
        assert!(!t.has_own_property("cfg"));
    }

    // ── reflect_define_property ───────────────────────────────────────────────

    #[test]
    fn test_reflect_define_property_creates_property() {
        let mut t = JsObject::new();
        assert!(
            reflect_define_property(
                &mut t,
                "n",
                JsValue::Smi(3),
                PropertyAttributes::WRITABLE | PropertyAttributes::ENUMERABLE,
            )
            .unwrap()
        );
        assert_eq!(t.get_own_property("n"), Some(JsValue::Smi(3)));
    }

    #[test]
    fn test_reflect_define_property_non_extensible_returns_false() {
        let mut t = JsObject::new();
        t.prevent_extensions();
        assert!(
            !reflect_define_property(&mut t, "new", JsValue::Smi(1), PropertyAttributes::WRITABLE,)
                .unwrap()
        );
    }

    #[test]
    fn test_reflect_define_property_read_only() {
        let mut t = JsObject::new();
        assert!(
            reflect_define_property(&mut t, "ro", JsValue::Smi(7), PropertyAttributes::empty(),)
                .unwrap()
        );
        let desc = reflect_get_own_property_descriptor(&t, "ro");
        assert!(desc.is_some());
        let (_val, attrs) = desc.unwrap();
        assert!(!attrs.contains(PropertyAttributes::WRITABLE));
    }

    #[test]
    fn test_reflect_define_property_all_attributes() {
        let mut t = JsObject::new();
        let all = PropertyAttributes::WRITABLE
            | PropertyAttributes::ENUMERABLE
            | PropertyAttributes::CONFIGURABLE;
        assert!(reflect_define_property(&mut t, "full", JsValue::Boolean(true), all).unwrap());
        let (val, attrs) = reflect_get_own_property_descriptor(&t, "full").unwrap();
        assert_eq!(val, JsValue::Boolean(true));
        assert_eq!(attrs, all);
    }

    // ── reflect_get_own_property_descriptor ──────────────────────────────────

    #[test]
    fn test_reflect_get_own_property_descriptor_exists() {
        let mut t = JsObject::new();
        let attrs = PropertyAttributes::WRITABLE | PropertyAttributes::ENUMERABLE;
        t.define_own_property("k", JsValue::Smi(5), attrs).unwrap();
        let desc = reflect_get_own_property_descriptor(&t, "k");
        assert!(desc.is_some());
        let (val, a) = desc.unwrap();
        assert_eq!(val, JsValue::Smi(5));
        assert_eq!(a, attrs);
    }

    #[test]
    fn test_reflect_get_own_property_descriptor_missing_returns_none() {
        let t = JsObject::new();
        assert!(reflect_get_own_property_descriptor(&t, "nope").is_none());
    }

    #[test]
    fn test_reflect_get_own_property_descriptor_does_not_check_prototype() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        proto
            .borrow_mut()
            .set_property("inherited", JsValue::Smi(1))
            .unwrap();
        let child = JsObject::with_prototype(proto);
        assert!(reflect_get_own_property_descriptor(&child, "inherited").is_none());
    }

    // ── reflect_get_prototype_of / set_prototype_of ───────────────────────────

    #[test]
    fn test_reflect_get_prototype_of_none() {
        let t = JsObject::new();
        assert!(reflect_get_prototype_of(&t).is_none());
    }

    #[test]
    fn test_reflect_get_prototype_of_some() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        let child = JsObject::with_prototype(Rc::clone(&proto));
        let got = reflect_get_prototype_of(&child);
        assert!(got.is_some());
        assert!(Rc::ptr_eq(&proto, &got.unwrap()));
    }

    #[test]
    fn test_reflect_set_prototype_of_success() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        let mut t = JsObject::new();
        assert!(reflect_set_prototype_of(&mut t, Some(Rc::clone(&proto))));
        assert!(t.prototype().is_some());
    }

    #[test]
    fn test_reflect_set_prototype_of_null() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        let mut t = JsObject::with_prototype(proto);
        assert!(reflect_set_prototype_of(&mut t, None));
        assert!(t.prototype().is_none());
    }

    #[test]
    fn test_reflect_set_prototype_of_non_extensible_same_is_ok() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        let mut t = JsObject::with_prototype(Rc::clone(&proto));
        t.prevent_extensions();
        assert!(reflect_set_prototype_of(&mut t, Some(Rc::clone(&proto))));
    }

    #[test]
    fn test_reflect_set_prototype_of_non_extensible_different_returns_false() {
        let mut t = JsObject::new();
        t.prevent_extensions();
        let new_proto = Rc::new(RefCell::new(JsObject::new()));
        assert!(!reflect_set_prototype_of(&mut t, Some(new_proto)));
    }

    // ── reflect_is_extensible ─────────────────────────────────────────────────

    #[test]
    fn test_reflect_is_extensible_new_object() {
        let t = JsObject::new();
        assert!(reflect_is_extensible(&t));
    }

    #[test]
    fn test_reflect_is_extensible_after_prevent() {
        let mut t = JsObject::new();
        t.prevent_extensions();
        assert!(!reflect_is_extensible(&t));
    }

    // ── reflect_prevent_extensions ───────────────────────────────────────────

    #[test]
    fn test_reflect_prevent_extensions_returns_true() {
        let mut t = JsObject::new();
        assert!(reflect_prevent_extensions(&mut t));
        assert!(!t.is_extensible());
    }

    #[test]
    fn test_reflect_prevent_extensions_idempotent() {
        let mut t = JsObject::new();
        assert!(reflect_prevent_extensions(&mut t));
        assert!(reflect_prevent_extensions(&mut t));
        assert!(!t.is_extensible());
    }

    // ── reflect_own_keys ──────────────────────────────────────────────────────

    #[test]
    fn test_reflect_own_keys_empty() {
        let t = JsObject::new();
        assert!(reflect_own_keys(&t).is_empty());
    }

    #[test]
    fn test_reflect_own_keys_multiple() {
        let mut t = JsObject::new();
        t.set_property("a", JsValue::Smi(1)).unwrap();
        t.set_property("b", JsValue::Smi(2)).unwrap();
        t.set_property("c", JsValue::Smi(3)).unwrap();
        let keys = reflect_own_keys(&t);
        assert_eq!(keys.len(), 3);
    }

    // ── reflect_apply ─────────────────────────────────────────────────────────

    #[test]
    fn test_reflect_apply_with_this() {
        let result = reflect_apply(
            |this, _args| {
                if this == JsValue::Smi(42) {
                    Ok(JsValue::Boolean(true))
                } else {
                    Ok(JsValue::Boolean(false))
                }
            },
            JsValue::Smi(42),
            vec![],
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_reflect_apply_empty_args() {
        let result = reflect_apply(
            |_this, args| Ok(JsValue::Smi(args.len() as i32)),
            JsValue::Undefined,
            vec![],
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(0));
    }

    #[test]
    fn test_reflect_apply_multiple_args() {
        let result = reflect_apply(
            |_this, args| Ok(JsValue::Smi(args.len() as i32)),
            JsValue::Undefined,
            vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    // ── reflect_construct ─────────────────────────────────────────────────────

    #[test]
    fn test_reflect_construct_empty_args() {
        let obj = reflect_construct(|_args| Ok(JsObject::new()), vec![]).unwrap();
        assert!(reflect_own_keys(&obj).is_empty());
    }

    #[test]
    fn test_reflect_construct_with_new_target_sets_prototype() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        proto
            .borrow_mut()
            .set_property("tag", JsValue::Boolean(true))
            .unwrap();
        let obj =
            reflect_construct_with_new_target(|_args| Ok(JsObject::new()), vec![], Some(proto))
                .unwrap();
        assert!(obj.prototype().is_some());
    }

    // ── reflect_own_keys ─────────────────────────────────────────────────────

    #[test]
    fn test_reflect_own_keys_empty_v2() {
        let t = JsObject::new();
        assert!(reflect_own_keys(&t).is_empty());
    }

    #[test]
    fn test_reflect_own_keys_returns_all_own_keys() {
        let mut t = JsObject::new();
        t.set_property("a", JsValue::Smi(1)).unwrap();
        t.set_property("b", JsValue::Smi(2)).unwrap();
        let keys = reflect_own_keys(&t);
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"a".to_string()));
        assert!(keys.contains(&"b".to_string()));
    }

    #[test]
    fn test_reflect_own_keys_does_not_include_prototype_keys() {
        let proto = Rc::new(RefCell::new(JsObject::new()));
        proto
            .borrow_mut()
            .set_property("inherited", JsValue::Smi(0))
            .unwrap();
        let mut child = JsObject::with_prototype(proto);
        child.set_property("own", JsValue::Smi(1)).unwrap();
        let keys = reflect_own_keys(&child);
        assert!(keys.contains(&"own".to_string()));
        assert!(!keys.contains(&"inherited".to_string()));
    }

    // ── reflect_apply ────────────────────────────────────────────────────────

    #[test]
    fn test_reflect_apply_calls_target() {
        let result = reflect_apply(
            |_this, args| Ok(args.first().cloned().unwrap_or(JsValue::Undefined)),
            JsValue::Undefined,
            vec![JsValue::Smi(42)],
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn test_reflect_apply_passes_this() {
        let result = reflect_apply(|this, _args| Ok(this), JsValue::Boolean(true), vec![]).unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── reflect_construct ────────────────────────────────────────────────────

    #[test]
    fn test_reflect_construct_creates_object() {
        let obj = reflect_construct(
            |args| {
                let mut o = JsObject::new();
                if let Some(v) = args.first() {
                    o.set_property("val", v.clone()).unwrap();
                }
                Ok(o)
            },
            vec![JsValue::Smi(7)],
        )
        .unwrap();
        assert_eq!(obj.get_property("val"), JsValue::Smi(7));
    }

    // ── Reflect.ownKeys enumeration order conformance ────────────────────────

    #[test]
    fn test_reflect_own_keys_integer_sorted() {
        let mut obj = JsObject::new();
        obj.set_property("2", JsValue::Smi(1)).unwrap();
        obj.set_property("0", JsValue::Smi(2)).unwrap();
        obj.set_property("1", JsValue::Smi(3)).unwrap();
        let keys = reflect_own_keys(&obj);
        assert_eq!(keys, &["0", "1", "2"]);
    }

    #[test]
    fn test_reflect_own_keys_integers_strings_order() {
        let mut obj = JsObject::new();
        obj.set_property("b", JsValue::Smi(1)).unwrap();
        obj.set_property("5", JsValue::Smi(2)).unwrap();
        obj.set_property("a", JsValue::Smi(3)).unwrap();
        obj.set_property("0", JsValue::Smi(4)).unwrap();
        let keys = reflect_own_keys(&obj);
        assert_eq!(keys, &["0", "5", "b", "a"]);
    }

    #[test]
    fn test_reflect_own_keys_includes_non_enumerable() {
        let mut obj = JsObject::new();
        obj.set_property("a", JsValue::Smi(1)).unwrap();
        obj.define_own_property(
            "b",
            JsValue::Smi(2),
            PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE,
        )
        .unwrap();
        let keys = reflect_own_keys(&obj);
        assert!(keys.contains(&"a".to_string()));
        assert!(keys.contains(&"b".to_string()));
    }

    #[test]
    fn test_reflect_own_keys_empty_v3() {
        let obj = JsObject::new();
        assert!(reflect_own_keys(&obj).is_empty());
    }
}
