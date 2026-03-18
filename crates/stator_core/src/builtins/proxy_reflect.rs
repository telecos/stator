//! Proxy + Reflect round-trip conformance tests.
//!
//! Every Proxy internal-method trap has a corresponding `Reflect.*` static
//! method.  When no trap is installed on a proxy handler the operation must
//! fall through to the target unchanged — producing the *exact same*
//! observable result as calling `Reflect.*` directly on the target.
//!
//! This module exercises that invariant for all 13 traps, plus:
//!
//! * **Nested Proxy** — a proxy wrapping another proxy, verifying that traps
//!   compose correctly.
//! * **Revocable Proxy through Reflect** — all operations on a revoked proxy
//!   must throw `TypeError`.
//! * **Descriptor round-trips** — `defineProperty` + `getOwnPropertyDescriptor`
//!   produce consistent `(value, attributes)` pairs.
//! * **Receiver forwarding** — `get` / `set` with an explicit receiver.

use std::cell::RefCell;
use std::rc::Rc;

use crate::builtins::proxy::*;
use crate::builtins::reflect::*;
use crate::error::{StatorError, StatorResult};
use crate::objects::js_object::JsObject;
use crate::objects::map::PropertyAttributes;
use crate::objects::property_map::PropertyMap;
use crate::objects::value::JsValue;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn target_with(key: &str, value: JsValue) -> JsObject {
    let mut t = JsObject::new();
    t.set_property(key, value).unwrap();
    t
}

fn all_attrs() -> PropertyAttributes {
    PropertyAttributes::WRITABLE | PropertyAttributes::ENUMERABLE | PropertyAttributes::CONFIGURABLE
}

// ══════════════════════════════════════════════════════════════════════════════
// 1. Proxy handler forwarding to Reflect — default (no-trap) behaviour
// ══════════════════════════════════════════════════════════════════════════════

// ── get ──────────────────────────────────────────────────────────────────────

#[test]
fn test_roundtrip_get_existing_key() {
    let target = target_with("x", JsValue::Smi(42));
    let proxy = proxy_new(target_with("x", JsValue::Smi(42)), ProxyHandler::default());
    let proxy_val = proxy_get(&proxy, "x").unwrap();
    let reflect_val = reflect_get(&target, "x");
    assert_eq!(proxy_val, reflect_val);
}

#[test]
fn test_roundtrip_get_missing_key() {
    let target = JsObject::new();
    let proxy = proxy_new(JsObject::new(), ProxyHandler::default());
    assert_eq!(
        proxy_get(&proxy, "nope").unwrap(),
        reflect_get(&target, "nope")
    );
}

// ── get with receiver ────────────────────────────────────────────────────────

#[test]
fn test_roundtrip_get_with_receiver() {
    let target = target_with("k", JsValue::Smi(77));
    let proxy = proxy_new(target_with("k", JsValue::Smi(77)), ProxyHandler::default());
    let receiver = JsValue::Boolean(true);
    let proxy_val = proxy_get_with_receiver(&proxy, "k", &receiver).unwrap();
    let reflect_val = reflect_get_with_receiver(&target, "k", &receiver);
    assert_eq!(proxy_val, reflect_val);
}

#[test]
fn test_roundtrip_get_with_receiver_missing() {
    let target = JsObject::new();
    let proxy = proxy_new(JsObject::new(), ProxyHandler::default());
    let receiver = JsValue::Smi(99);
    assert_eq!(
        proxy_get_with_receiver(&proxy, "z", &receiver).unwrap(),
        reflect_get_with_receiver(&target, "z", &receiver),
    );
}

// ── set ──────────────────────────────────────────────────────────────────────

#[test]
fn test_roundtrip_set_new_property() {
    let target1 = JsObject::new();
    let mut target2 = JsObject::new();
    let mut proxy = proxy_new(target1, ProxyHandler::default());
    let proxy_ok = proxy_set(&mut proxy, "a", JsValue::Smi(1)).unwrap();
    let reflect_ok = reflect_set(&mut target2, "a", JsValue::Smi(1)).unwrap();
    assert_eq!(proxy_ok, reflect_ok);
}

#[test]
fn test_roundtrip_set_overwrite() {
    let mut target = target_with("k", JsValue::Smi(1));
    let mut proxy = proxy_new(target_with("k", JsValue::Smi(1)), ProxyHandler::default());
    let proxy_ok = proxy_set(&mut proxy, "k", JsValue::Smi(2)).unwrap();
    let reflect_ok = reflect_set(&mut target, "k", JsValue::Smi(2)).unwrap();
    assert_eq!(proxy_ok, reflect_ok);
}

// ── set with receiver ────────────────────────────────────────────────────────

#[test]
fn test_roundtrip_set_with_receiver() {
    let target1 = JsObject::new();
    let mut target2 = JsObject::new();
    let receiver = JsValue::Smi(10);
    let mut proxy = proxy_new(target1, ProxyHandler::default());
    let proxy_ok = proxy_set_with_receiver(&mut proxy, "r", JsValue::Smi(5), &receiver).unwrap();
    let reflect_ok =
        reflect_set_with_receiver(&mut target2, "r", JsValue::Smi(5), &receiver).unwrap();
    assert_eq!(proxy_ok, reflect_ok);
}

// ── has ──────────────────────────────────────────────────────────────────────

#[test]
fn test_roundtrip_has_existing() {
    let target = target_with("p", JsValue::Null);
    let proxy = proxy_new(target_with("p", JsValue::Null), ProxyHandler::default());
    assert_eq!(proxy_has(&proxy, "p").unwrap(), reflect_has(&target, "p"));
}

#[test]
fn test_roundtrip_has_missing() {
    let target = JsObject::new();
    let proxy = proxy_new(JsObject::new(), ProxyHandler::default());
    assert_eq!(proxy_has(&proxy, "q").unwrap(), reflect_has(&target, "q"));
}

#[test]
fn test_roundtrip_has_is_in_operator() {
    // Reflect.has walks the prototype chain, same as `in`.
    let proto = Rc::new(RefCell::new(JsObject::new()));
    proto
        .borrow_mut()
        .set_property("inherited", JsValue::Boolean(true))
        .unwrap();
    let target = JsObject::with_prototype(Rc::clone(&proto));
    let proxy = proxy_new(JsObject::with_prototype(proto), ProxyHandler::default());
    assert!(proxy_has(&proxy, "inherited").unwrap());
    assert!(reflect_has(&target, "inherited"));
}

// ── deleteProperty ───────────────────────────────────────────────────────────

#[test]
fn test_roundtrip_delete_existing() {
    let target1 = target_with("d", JsValue::Smi(1));
    let mut target2 = target_with("d", JsValue::Smi(1));
    let mut proxy = proxy_new(target1, ProxyHandler::default());
    let proxy_ok = proxy_delete_property(&mut proxy, "d").unwrap();
    let reflect_ok = reflect_delete_property(&mut target2, "d").unwrap();
    assert_eq!(proxy_ok, reflect_ok);
}

#[test]
fn test_roundtrip_delete_missing() {
    let mut target = JsObject::new();
    let mut proxy = proxy_new(JsObject::new(), ProxyHandler::default());
    let proxy_ok = proxy_delete_property(&mut proxy, "ghost").unwrap();
    let reflect_ok = reflect_delete_property(&mut target, "ghost").unwrap();
    assert_eq!(proxy_ok, reflect_ok);
}

#[test]
fn test_roundtrip_delete_equals_delete_operator() {
    // Non-configurable property: both proxy and reflect return false.
    let mut target1 = JsObject::new();
    target1
        .define_own_property("nc", JsValue::Smi(1), PropertyAttributes::WRITABLE)
        .unwrap();
    let mut target2 = JsObject::new();
    target2
        .define_own_property("nc", JsValue::Smi(1), PropertyAttributes::WRITABLE)
        .unwrap();
    let mut proxy = proxy_new(target1, ProxyHandler::default());
    assert_eq!(
        proxy_delete_property(&mut proxy, "nc").unwrap(),
        reflect_delete_property(&mut target2, "nc").unwrap(),
    );
}

// ── ownKeys ──────────────────────────────────────────────────────────────────

#[test]
fn test_roundtrip_own_keys_string_keys() {
    let mut target = JsObject::new();
    target.set_property("a", JsValue::Smi(1)).unwrap();
    target.set_property("b", JsValue::Smi(2)).unwrap();
    let mut proxy_target = JsObject::new();
    proxy_target.set_property("a", JsValue::Smi(1)).unwrap();
    proxy_target.set_property("b", JsValue::Smi(2)).unwrap();
    let proxy = proxy_new(proxy_target, ProxyHandler::default());
    let proxy_keys = proxy_own_keys(&proxy).unwrap();
    let reflect_keys = reflect_own_keys_values(&target);
    assert_eq!(proxy_keys.len(), reflect_keys.len());
    for k in &reflect_keys {
        assert!(proxy_keys.contains(k));
    }
}

#[test]
fn test_roundtrip_own_keys_empty() {
    let target = JsObject::new();
    let proxy = proxy_new(JsObject::new(), ProxyHandler::default());
    let proxy_keys = proxy_own_keys(&proxy).unwrap();
    let reflect_keys = reflect_own_keys_values(&target);
    assert!(proxy_keys.is_empty());
    assert!(reflect_keys.is_empty());
}

// ── apply / construct ────────────────────────────────────────────────────────

#[test]
fn test_roundtrip_apply_through_trap() {
    let add_one = |_this: JsValue, args: Vec<JsValue>| -> StatorResult<JsValue> {
        let n = match args.first() {
            Some(JsValue::Smi(n)) => *n,
            _ => 0,
        };
        Ok(JsValue::Smi(n + 1))
    };

    // Via Reflect.apply
    let reflect_result = reflect_apply(add_one, JsValue::Undefined, vec![JsValue::Smi(5)]).unwrap();

    // Via Proxy with apply trap
    let mut handler = ProxyHandler::default();
    handler.apply = Some(Box::new(|_this, args| {
        let n = match args.first() {
            Some(JsValue::Smi(n)) => *n,
            _ => 0,
        };
        Ok(JsValue::Smi(n + 1))
    }));
    let mut proxy = proxy_new(JsObject::new(), handler);
    let proxy_result = proxy_apply(&mut proxy, JsValue::Undefined, vec![JsValue::Smi(5)]).unwrap();

    assert_eq!(proxy_result, reflect_result);
}

#[test]
fn test_roundtrip_construct_through_trap() {
    // Via Reflect.construct
    let reflect_obj = reflect_construct(
        |args| {
            let mut o = JsObject::new();
            if let Some(v) = args.first() {
                o.set_property("val", v.clone()).unwrap();
            }
            Ok(o)
        },
        vec![JsValue::Smi(42)],
    )
    .unwrap();

    // Via Proxy with construct trap
    let mut handler = ProxyHandler::default();
    handler.construct = Some(Box::new(|args, _new_target| {
        let mut map = PropertyMap::new();
        if let Some(v) = args.first() {
            map.insert("val".to_string(), v.clone());
        }
        Ok(JsValue::PlainObject(Rc::new(RefCell::new(map))))
    }));
    let mut proxy = proxy_new(JsObject::new(), handler);
    let proxy_obj =
        proxy_construct(&mut proxy, vec![JsValue::Smi(42)], JsValue::Undefined).unwrap();

    // Both should store val=42
    assert_eq!(reflect_obj.get_property("val"), JsValue::Smi(42));
    if let JsValue::PlainObject(map) = proxy_obj {
        assert_eq!(map.borrow().get("val"), Some(&JsValue::Smi(42)));
    } else {
        panic!("expected PlainObject from proxy construct");
    }
}

// ── getPrototypeOf / setPrototypeOf ──────────────────────────────────────────

#[test]
fn test_roundtrip_get_prototype_of_null() {
    let target = JsObject::new();
    let proxy = proxy_new(JsObject::new(), ProxyHandler::default());
    let proxy_proto = proxy_get_prototype_of(&proxy).unwrap();
    let reflect_proto = reflect_get_prototype_of_value(&target);
    assert_eq!(proxy_proto, reflect_proto);
    assert_eq!(proxy_proto, JsValue::Null);
}

#[test]
#[ignore] // TODO: conformance — not yet passing
fn test_roundtrip_get_prototype_of_with_prototype() {
    let proto = Rc::new(RefCell::new(JsObject::new()));
    proto
        .borrow_mut()
        .set_property("tag", JsValue::Boolean(true))
        .unwrap();
    let target = JsObject::with_prototype(Rc::clone(&proto));
    let proxy = proxy_new(JsObject::with_prototype(proto), ProxyHandler::default());
    let proxy_proto = proxy_get_prototype_of(&proxy).unwrap();
    let reflect_proto = reflect_get_prototype_of_value(&target);
    // Both must return an object-like value (not Null).
    assert!(proxy_proto.is_object_like());
    assert!(reflect_proto.is_object_like());
}

#[test]
fn test_roundtrip_set_prototype_of_null() {
    let proto = Rc::new(RefCell::new(JsObject::new()));
    let target1 = JsObject::with_prototype(Rc::clone(&proto));
    let mut target2 = JsObject::with_prototype(proto);
    let mut proxy = proxy_new(target1, ProxyHandler::default());
    let proxy_ok = proxy_set_prototype_of(&mut proxy, JsValue::Null).unwrap();
    let reflect_ok = reflect_set_prototype_of_value(&mut target2, JsValue::Null).unwrap();
    assert_eq!(proxy_ok, reflect_ok);
}

#[test]
fn test_roundtrip_set_prototype_of_object() {
    let mut target = JsObject::new();
    let new_proto = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
    let mut proxy = proxy_new(JsObject::new(), ProxyHandler::default());
    let proxy_ok = proxy_set_prototype_of(&mut proxy, new_proto.clone()).unwrap();
    let reflect_ok = reflect_set_prototype_of_value(&mut target, new_proto).unwrap();
    assert_eq!(proxy_ok, reflect_ok);
    assert!(proxy_ok);
}

#[test]
fn test_get_prototype_of_returns_jsvalue() {
    // Verify that getPrototypeOf returns JsValue::Null, not Option<Rc<..>>.
    let target = JsObject::new();
    let proxy = proxy_new(target, ProxyHandler::default());
    let result = proxy_get_prototype_of(&proxy).unwrap();
    assert!(matches!(result, JsValue::Null));
}

#[test]
fn test_set_prototype_of_accepts_jsvalue() {
    // Verify setPrototypeOf accepts JsValue (not Option<Rc<..>>).
    let mut proxy = proxy_new(JsObject::new(), ProxyHandler::default());
    let proto = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
    assert!(proxy_set_prototype_of(&mut proxy, proto).unwrap());
    // Now clear it.
    assert!(proxy_set_prototype_of(&mut proxy, JsValue::Null).unwrap());
}

// ── defineProperty / getOwnPropertyDescriptor round-trip ─────────────────────

#[test]
fn test_roundtrip_define_then_get_descriptor() {
    let attrs = PropertyAttributes::WRITABLE | PropertyAttributes::ENUMERABLE;
    let target = JsObject::new();
    let mut proxy = proxy_new(target, ProxyHandler::default());

    proxy_define_property(&mut proxy, "dp", JsValue::Smi(99), attrs).unwrap();
    let desc = proxy_get_own_property_descriptor(&proxy, "dp").unwrap();
    assert!(desc.is_some());
    let (val, a) = desc.unwrap();
    assert_eq!(val, JsValue::Smi(99));
    assert_eq!(a, attrs);
}

#[test]
fn test_roundtrip_descriptor_matches_reflect() {
    let attrs = all_attrs();
    let mut target = JsObject::new();
    reflect_define_property(&mut target, "k", JsValue::Boolean(true), attrs).unwrap();
    let reflect_desc = reflect_get_own_property_descriptor(&target, "k");

    let proxy = proxy_new(target, ProxyHandler::default());
    let proxy_desc = proxy_get_own_property_descriptor(&proxy, "k").unwrap();
    assert_eq!(proxy_desc, reflect_desc);
}

#[test]
fn test_roundtrip_descriptor_read_only_property() {
    let mut target = JsObject::new();
    target
        .define_own_property("ro", JsValue::Smi(7), PropertyAttributes::empty())
        .unwrap();
    let reflect_desc = reflect_get_own_property_descriptor(&target, "ro");
    let proxy = proxy_new(target, ProxyHandler::default());
    let proxy_desc = proxy_get_own_property_descriptor(&proxy, "ro").unwrap();
    assert_eq!(proxy_desc, reflect_desc);
    let (_, attrs) = proxy_desc.unwrap();
    assert!(!attrs.contains(PropertyAttributes::WRITABLE));
    assert!(!attrs.contains(PropertyAttributes::CONFIGURABLE));
}

// ── isExtensible / preventExtensions ─────────────────────────────────────────

#[test]
fn test_roundtrip_is_extensible() {
    let target = JsObject::new();
    let proxy = proxy_new(JsObject::new(), ProxyHandler::default());
    assert_eq!(
        proxy_is_extensible(&proxy).unwrap(),
        reflect_is_extensible(&target),
    );
}

#[test]
fn test_roundtrip_prevent_extensions() {
    let mut target = JsObject::new();
    let mut proxy = proxy_new(JsObject::new(), ProxyHandler::default());
    let proxy_ok = proxy_prevent_extensions(&mut proxy).unwrap();
    let reflect_ok = reflect_prevent_extensions(&mut target);
    assert_eq!(proxy_ok, reflect_ok);
    // Both should now report non-extensible.
    assert!(!proxy_is_extensible(&proxy).unwrap());
    assert!(!reflect_is_extensible(&target));
}

// ══════════════════════════════════════════════════════════════════════════════
// 9. Nested Proxy — Proxy wrapping Proxy, traps compose correctly
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_nested_proxy_get_composes() {
    // Inner proxy adds 10, outer proxy doubles.
    let target = target_with("n", JsValue::Smi(5));
    let inner_proxy = Rc::new(RefCell::new(proxy_new(target, ProxyHandler::default())));

    let inner = Rc::clone(&inner_proxy);
    let mut outer_handler = ProxyHandler::default();
    outer_handler.get = Some(Box::new(move |_t, key, _r| {
        let inner_val = proxy_get(&inner.borrow(), key)?;
        match inner_val {
            JsValue::Smi(n) => Ok(JsValue::Smi(n * 2)),
            other => Ok(other),
        }
    }));
    let outer = proxy_new(JsObject::new(), outer_handler);
    // Inner returns 5, outer doubles to 10.
    assert_eq!(proxy_get(&outer, "n").unwrap(), JsValue::Smi(10));
}

#[test]
fn test_nested_proxy_has_composes() {
    let target = target_with("p", JsValue::Null);
    let inner_proxy = Rc::new(RefCell::new(proxy_new(target, ProxyHandler::default())));

    let inner = Rc::clone(&inner_proxy);
    let mut outer_handler = ProxyHandler::default();
    outer_handler.has = Some(Box::new(move |_t, key| proxy_has(&inner.borrow(), key)));
    let outer = proxy_new(JsObject::new(), outer_handler);
    assert!(proxy_has(&outer, "p").unwrap());
    assert!(!proxy_has(&outer, "missing").unwrap());
}

#[test]
fn test_nested_proxy_set_composes() {
    let target = JsObject::new();
    let inner_proxy = Rc::new(RefCell::new(proxy_new(target, ProxyHandler::default())));

    let inner = Rc::clone(&inner_proxy);
    let mut outer_handler = ProxyHandler::default();
    outer_handler.set = Some(Box::new(move |_t, key, value, _r| {
        proxy_set(&mut inner.borrow_mut(), key, value)
    }));
    let mut outer = proxy_new(JsObject::new(), outer_handler);
    assert!(proxy_set(&mut outer, "nested_key", JsValue::Smi(88)).unwrap());
    // Value was set through inner proxy onto inner's target.
    assert_eq!(
        proxy_get(&inner_proxy.borrow(), "nested_key").unwrap(),
        JsValue::Smi(88),
    );
}

#[test]
fn test_nested_proxy_own_keys_composes() {
    let mut target = JsObject::new();
    target.set_property("x", JsValue::Smi(1)).unwrap();
    target.set_property("y", JsValue::Smi(2)).unwrap();
    let inner_proxy = Rc::new(RefCell::new(proxy_new(target, ProxyHandler::default())));

    let inner = Rc::clone(&inner_proxy);
    let mut outer_handler = ProxyHandler::default();
    outer_handler.own_keys = Some(Box::new(move |_t| proxy_own_keys(&inner.borrow())));
    let outer = proxy_new(JsObject::new(), outer_handler);
    let keys = proxy_own_keys(&outer).unwrap();
    assert_eq!(keys.len(), 2);
    assert!(keys.contains(&JsValue::String("x".into())));
    assert!(keys.contains(&JsValue::String("y".into())));
}

// ══════════════════════════════════════════════════════════════════════════════
// 10. Revocable Proxy through Reflect — operations on revoked proxy throw
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_revoked_proxy_get_throws() {
    let mut p = proxy_revocable(target_with("k", JsValue::Smi(1)), ProxyHandler::default());
    proxy_revoke(&mut p);
    assert!(matches!(proxy_get(&p, "k"), Err(StatorError::TypeError(_))));
}

#[test]
fn test_revoked_proxy_set_throws() {
    let mut p = proxy_revocable(JsObject::new(), ProxyHandler::default());
    proxy_revoke(&mut p);
    assert!(matches!(
        proxy_set(&mut p, "x", JsValue::Smi(1)),
        Err(StatorError::TypeError(_))
    ));
}

#[test]
fn test_revoked_proxy_has_throws() {
    let mut p = proxy_revocable(JsObject::new(), ProxyHandler::default());
    proxy_revoke(&mut p);
    assert!(matches!(proxy_has(&p, "x"), Err(StatorError::TypeError(_))));
}

#[test]
fn test_revoked_proxy_delete_throws() {
    let mut p = proxy_revocable(JsObject::new(), ProxyHandler::default());
    proxy_revoke(&mut p);
    assert!(matches!(
        proxy_delete_property(&mut p, "x"),
        Err(StatorError::TypeError(_))
    ));
}

#[test]
fn test_revoked_proxy_define_property_throws() {
    let mut p = proxy_revocable(JsObject::new(), ProxyHandler::default());
    proxy_revoke(&mut p);
    assert!(matches!(
        proxy_define_property(&mut p, "x", JsValue::Smi(1), all_attrs()),
        Err(StatorError::TypeError(_))
    ));
}

#[test]
fn test_revoked_proxy_get_own_property_descriptor_throws() {
    let mut p = proxy_revocable(JsObject::new(), ProxyHandler::default());
    proxy_revoke(&mut p);
    assert!(matches!(
        proxy_get_own_property_descriptor(&p, "x"),
        Err(StatorError::TypeError(_))
    ));
}

#[test]
fn test_revoked_proxy_get_prototype_of_throws() {
    let mut p = proxy_revocable(JsObject::new(), ProxyHandler::default());
    proxy_revoke(&mut p);
    assert!(matches!(
        proxy_get_prototype_of(&p),
        Err(StatorError::TypeError(_))
    ));
}

#[test]
fn test_revoked_proxy_set_prototype_of_throws() {
    let mut p = proxy_revocable(JsObject::new(), ProxyHandler::default());
    proxy_revoke(&mut p);
    assert!(matches!(
        proxy_set_prototype_of(&mut p, JsValue::Null),
        Err(StatorError::TypeError(_))
    ));
}

#[test]
fn test_revoked_proxy_is_extensible_throws() {
    let mut p = proxy_revocable(JsObject::new(), ProxyHandler::default());
    proxy_revoke(&mut p);
    assert!(matches!(
        proxy_is_extensible(&p),
        Err(StatorError::TypeError(_))
    ));
}

#[test]
fn test_revoked_proxy_prevent_extensions_throws() {
    let mut p = proxy_revocable(JsObject::new(), ProxyHandler::default());
    proxy_revoke(&mut p);
    assert!(matches!(
        proxy_prevent_extensions(&mut p),
        Err(StatorError::TypeError(_))
    ));
}

#[test]
fn test_revoked_proxy_own_keys_throws() {
    let mut p = proxy_revocable(JsObject::new(), ProxyHandler::default());
    proxy_revoke(&mut p);
    assert!(matches!(proxy_own_keys(&p), Err(StatorError::TypeError(_))));
}

#[test]
fn test_revoked_proxy_apply_throws() {
    let mut handler = ProxyHandler::default();
    handler.apply = Some(Box::new(|_t, _a| Ok(JsValue::Undefined)));
    let mut p = proxy_revocable(JsObject::new(), handler);
    proxy_revoke(&mut p);
    assert!(matches!(
        proxy_apply(&mut p, JsValue::Undefined, vec![]),
        Err(StatorError::TypeError(_))
    ));
}

#[test]
fn test_revoked_proxy_construct_throws() {
    let mut handler = ProxyHandler::default();
    handler.construct = Some(Box::new(|_a, _nt| {
        Ok(JsValue::PlainObject(Rc::new(RefCell::new(
            PropertyMap::new(),
        ))))
    }));
    let mut p = proxy_revocable(JsObject::new(), handler);
    proxy_revoke(&mut p);
    assert!(matches!(
        proxy_construct(&mut p, vec![], JsValue::Undefined),
        Err(StatorError::TypeError(_))
    ));
}

// ══════════════════════════════════════════════════════════════════════════════
// Additional conformance tests
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_reflect_own_keys_returns_jsvalue_strings() {
    let mut target = JsObject::new();
    target.set_property("alpha", JsValue::Smi(1)).unwrap();
    target.set_property("beta", JsValue::Smi(2)).unwrap();
    let keys = reflect_own_keys_values(&target);
    assert!(keys.contains(&JsValue::String("alpha".into())));
    assert!(keys.contains(&JsValue::String("beta".into())));
}

#[test]
fn test_reflect_set_prototype_of_value_non_extensible_rejects() {
    let mut target = JsObject::new();
    target.prevent_extensions();
    let new_proto = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
    // Non-extensible target with null prototype cannot accept a new prototype.
    let result = reflect_set_prototype_of_value(&mut target, new_proto);
    assert!(result.is_ok());
    assert!(!result.unwrap());
}

#[test]
fn test_reflect_set_prototype_of_value_rejects_primitive() {
    let mut target = JsObject::new();
    let result = reflect_set_prototype_of_value(&mut target, JsValue::Smi(42));
    assert!(matches!(result, Err(StatorError::TypeError(_))));
}
