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
//! # References
//!
//! * ECMAScript 2025 Language Specification §28 — *The Reflect Object*

use std::cell::RefCell;
use std::rc::Rc;

use crate::error::StatorResult;
use crate::objects::js_object::JsObject;
use crate::objects::map::PropertyAttributes;
use crate::objects::value::JsValue;

// ── reflect_get ───────────────────────────────────────────────────────────────

/// ECMAScript §28.1.4 `Reflect.get(target, propertyKey)`.
///
/// Equivalent to `target[propertyKey]`.  Walks the prototype chain.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::reflect::reflect_get;
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::value::JsValue;
///
/// let mut target = JsObject::new();
/// target.set_property("x", JsValue::Smi(42)).unwrap();
/// assert_eq!(reflect_get(&target, "x"), JsValue::Smi(42));
/// assert_eq!(reflect_get(&target, "missing"), JsValue::Undefined);
/// ```
pub fn reflect_get(target: &JsObject, key: &str) -> JsValue {
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
/// use stator_core::builtins::reflect::reflect_set;
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::value::JsValue;
///
/// let mut target = JsObject::new();
/// assert!(reflect_set(&mut target, "x", JsValue::Smi(1)).unwrap());
/// assert_eq!(target.get_property("x"), JsValue::Smi(1));
/// ```
pub fn reflect_set(target: &mut JsObject, key: &str, value: JsValue) -> StatorResult<bool> {
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
/// use stator_core::builtins::reflect::reflect_has;
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::value::JsValue;
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
/// use stator_core::builtins::reflect::reflect_delete_property;
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::value::JsValue;
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
/// use stator_core::builtins::reflect::reflect_define_property;
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::map::PropertyAttributes;
/// use stator_core::objects::value::JsValue;
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
/// use stator_core::builtins::reflect::reflect_get_own_property_descriptor;
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::map::PropertyAttributes;
/// use stator_core::objects::value::JsValue;
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
/// use stator_core::builtins::reflect::reflect_get_prototype_of;
/// use stator_core::objects::js_object::JsObject;
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
/// use stator_core::builtins::reflect::reflect_set_prototype_of;
/// use stator_core::objects::js_object::JsObject;
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
/// use stator_core::builtins::reflect::reflect_is_extensible;
/// use stator_core::objects::js_object::JsObject;
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
/// use stator_core::builtins::reflect::reflect_prevent_extensions;
/// use stator_core::objects::js_object::JsObject;
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
/// use stator_core::builtins::reflect::reflect_own_keys;
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::value::JsValue;
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
/// use stator_core::builtins::reflect::reflect_apply;
/// use stator_core::objects::value::JsValue;
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

/// ECMAScript §28.1.2 `Reflect.construct(target, argumentsList)`.
///
/// In the pure-Rust engine model the constructor is a Rust closure that
/// receives the argument list and returns a new [`JsObject`].
///
/// # Examples
///
/// ```
/// use stator_core::builtins::reflect::reflect_construct;
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::value::JsValue;
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

    // ── reflect_own_keys ─────────────────────────────────────────────────────

    #[test]
    fn test_reflect_own_keys_empty() {
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
}
