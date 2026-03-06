//! ECMAScript §22.2 `RegExp` built-in constructor and prototype methods.
//!
//! This module provides the functions wired into [`super::install_globals`] so
//! that JavaScript code can construct `RegExp` objects and call their prototype
//! methods (`test`, `exec`, `toString`, and the `Symbol.match/replace/search/
//! split/matchAll` protocols).
//!
//! The heavy lifting is done by [`JsRegExp`] in
//! [`crate::objects::regexp`]; this module converts between [`JsValue`] and
//! the Rust-level API.

use std::cell::RefCell;
use std::rc::Rc;

use crate::objects::property_map::PropertyMap;

use crate::error::StatorResult;
use crate::objects::regexp::{JsRegExp, RegExpFlags, SymbolMatchResult};
use crate::objects::value::{JsValue, NativeIterator};

/// Create a new [`JsRegExp`] from positional `JsValue` arguments.
///
/// `args[0]` — pattern (string), `args[1]` — flags (string, default `""`).
/// Returns a `JsValue::PlainObject` with the regexp stored under `__regexp__`
/// together with accessor properties (`source`, `flags`, `global`, etc.) and
/// prototype methods.
pub fn regexp_construct(args: &[JsValue]) -> StatorResult<JsValue> {
    let pattern = match args.first() {
        Some(v) => v.to_js_string()?,
        None => String::new(),
    };
    let flags = match args.get(1) {
        Some(JsValue::Undefined) | None => String::new(),
        Some(v) => v.to_js_string()?,
    };
    let re = JsRegExp::new(&pattern, &flags)?;
    Ok(wrap_regexp(re))
}

/// Convert a [`JsRegExp`] into a `JsValue::PlainObject` exposing all
/// ECMAScript `RegExp.prototype` properties and methods.
pub fn wrap_regexp(re: JsRegExp) -> JsValue {
    let re = Rc::new(re);
    let mut props = PropertyMap::new();

    // ── Read-only accessors ─────────────────────────────────────────────
    props.insert("source".into(), JsValue::String(re.pattern().to_string()));
    props.insert(
        "flags".into(),
        JsValue::String(re.flags().to_flags_string()),
    );
    props.insert(
        "global".into(),
        JsValue::Boolean(re.flags().contains(RegExpFlags::GLOBAL)),
    );
    props.insert(
        "ignoreCase".into(),
        JsValue::Boolean(re.flags().contains(RegExpFlags::IGNORE_CASE)),
    );
    props.insert(
        "multiline".into(),
        JsValue::Boolean(re.flags().contains(RegExpFlags::MULTILINE)),
    );
    props.insert(
        "dotAll".into(),
        JsValue::Boolean(re.flags().contains(RegExpFlags::DOT_ALL)),
    );
    props.insert(
        "unicode".into(),
        JsValue::Boolean(re.flags().contains(RegExpFlags::UNICODE)),
    );
    props.insert(
        "unicodeSets".into(),
        JsValue::Boolean(re.flags().contains(RegExpFlags::UNICODE_SETS)),
    );
    props.insert(
        "sticky".into(),
        JsValue::Boolean(re.flags().contains(RegExpFlags::STICKY)),
    );
    props.insert(
        "hasIndices".into(),
        JsValue::Boolean(re.flags().contains(RegExpFlags::HAS_INDICES)),
    );

    // ── lastIndex (read/write) ──────────────────────────────────────────
    props.insert("lastIndex".into(), JsValue::Smi(re.last_index() as i32));

    // ── Prototype methods ───────────────────────────────────────────────

    // test(string)
    let re_test = Rc::clone(&re);
    props.insert(
        "test".into(),
        JsValue::NativeFunction(Rc::new(move |args: Vec<JsValue>| {
            let input = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::Boolean(re_test.test(&input)))
        })),
    );

    // exec(string)
    let re_exec = Rc::clone(&re);
    props.insert(
        "exec".into(),
        JsValue::NativeFunction(Rc::new(move |args: Vec<JsValue>| {
            let input = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            match re_exec.exec(&input) {
                Some(m) => Ok(match_to_js(&m)),
                None => Ok(JsValue::Null),
            }
        })),
    );

    // toString()
    let re_str = Rc::clone(&re);
    props.insert(
        "toString".into(),
        JsValue::NativeFunction(Rc::new(move |_args: Vec<JsValue>| {
            Ok(JsValue::String(re_str.to_string()))
        })),
    );

    // [Symbol.match](string)
    let re_match = Rc::clone(&re);
    props.insert(
        "__symbol_match__".into(),
        JsValue::NativeFunction(Rc::new(move |args: Vec<JsValue>| {
            let input = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            match re_match.symbol_match(&input) {
                None => Ok(JsValue::Null),
                Some(SymbolMatchResult::Single(m)) => Ok(match_to_js(&m)),
                Some(SymbolMatchResult::All(v)) => {
                    let arr: Vec<JsValue> = v.into_iter().map(JsValue::String).collect();
                    Ok(JsValue::Array(Rc::new(arr)))
                }
            }
        })),
    );

    // [Symbol.replace](string, replacement)
    let re_replace = Rc::clone(&re);
    props.insert(
        "__symbol_replace__".into(),
        JsValue::NativeFunction(Rc::new(move |args: Vec<JsValue>| {
            let input = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let replacement = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(
                re_replace.symbol_replace(&input, &replacement),
            ))
        })),
    );

    // [Symbol.search](string)
    let re_search = Rc::clone(&re);
    props.insert(
        "__symbol_search__".into(),
        JsValue::NativeFunction(Rc::new(move |args: Vec<JsValue>| {
            let input = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let idx = re_search.symbol_search(&input);
            Ok(if idx >= 0 {
                JsValue::Smi(idx as i32)
            } else {
                JsValue::Smi(-1)
            })
        })),
    );

    // [Symbol.split](string[, limit])
    let re_split = Rc::clone(&re);
    props.insert(
        "__symbol_split__".into(),
        JsValue::NativeFunction(Rc::new(move |args: Vec<JsValue>| {
            let input = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let limit = match args.get(1) {
                Some(JsValue::Undefined) | None => None,
                Some(v) => Some(v.to_number()? as usize),
            };
            let parts = re_split.symbol_split(&input, limit);
            let arr: Vec<JsValue> = parts.into_iter().map(JsValue::String).collect();
            Ok(JsValue::Array(Rc::new(arr)))
        })),
    );

    // [Symbol.matchAll](string)
    let re_match_all = Rc::clone(&re);
    props.insert(
        "__symbol_match_all__".into(),
        JsValue::NativeFunction(Rc::new(move |args: Vec<JsValue>| {
            let input = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let matches = re_match_all.symbol_match_all(&input);
            let items: Vec<JsValue> = matches.iter().map(match_to_js).collect();
            Ok(JsValue::Iterator(NativeIterator::from_items(items)))
        })),
    );

    // Sentinel so the interpreter can identify this PlainObject as a RegExp.
    props.insert("__is_regexp__".into(), JsValue::Boolean(true));

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

/// Convert a [`RegExpMatch`] into a `JsValue::PlainObject` matching the
/// ECMAScript `RegExpExecResult` shape:
///
/// * Index `0` → full match (via numeric string key)
/// * Index `1..` → capture groups
/// * `index` → byte offset
/// * `input` → original input string
/// * `groups` → named groups object (or `undefined`)
/// * `indices` → `{ 0: [s,e], 1: [s,e], …, groups: {…} }` when `/d` flag
fn match_to_js(m: &crate::objects::regexp::RegExpMatch) -> JsValue {
    let mut props = PropertyMap::new();

    // [0] = full match
    props.insert("0".into(), JsValue::String(m.matched.clone()));
    // [1..] = captures
    for (i, cap) in m.captures.iter().enumerate() {
        let key = (i + 1).to_string();
        props.insert(
            key,
            match cap {
                Some(s) => JsValue::String(s.clone()),
                None => JsValue::Undefined,
            },
        );
    }
    props.insert("index".into(), JsValue::Smi(m.index as i32));
    props.insert("input".into(), JsValue::String(m.input.clone()));

    // groups
    if m.named_groups.is_empty() {
        props.insert("groups".into(), JsValue::Undefined);
    } else {
        let mut groups = PropertyMap::new();
        for (k, v) in &m.named_groups {
            groups.insert(k.clone(), JsValue::String(v.clone()));
        }
        props.insert(
            "groups".into(),
            JsValue::PlainObject(Rc::new(RefCell::new(groups))),
        );
    }

    // indices (only present when /d flag was used)
    if let Some(ref idx) = m.indices {
        let mut idx_props = PropertyMap::new();
        for (i, pair) in idx.pairs.iter().enumerate() {
            let val = match pair {
                Some((s, e)) => JsValue::Array(Rc::new(vec![
                    JsValue::Smi(*s as i32),
                    JsValue::Smi(*e as i32),
                ])),
                None => JsValue::Undefined,
            };
            idx_props.insert(i.to_string(), val);
        }
        if !idx.groups.is_empty() {
            let mut g = PropertyMap::new();
            for (k, (s, e)) in &idx.groups {
                g.insert(
                    k.clone(),
                    JsValue::Array(Rc::new(vec![
                        JsValue::Smi(*s as i32),
                        JsValue::Smi(*e as i32),
                    ])),
                );
            }
            idx_props.insert(
                "groups".into(),
                JsValue::PlainObject(Rc::new(RefCell::new(g))),
            );
        }
        props.insert(
            "indices".into(),
            JsValue::PlainObject(Rc::new(RefCell::new(idx_props))),
        );
    }

    // length = 1 + captures.len() (spec compatibility)
    props.insert("length".into(), JsValue::Smi((1 + m.captures.len()) as i32));

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: extract a string property from a PlainObject JsValue.
    fn get_str(obj: &JsValue, key: &str) -> Option<String> {
        if let JsValue::PlainObject(map) = obj {
            match map.borrow().get(key)? {
                JsValue::String(s) => Some(s.clone()),
                _ => None,
            }
        } else {
            None
        }
    }

    /// Helper: extract a bool property from a PlainObject JsValue.
    fn get_bool(obj: &JsValue, key: &str) -> Option<bool> {
        if let JsValue::PlainObject(map) = obj {
            match map.borrow().get(key)? {
                JsValue::Boolean(b) => Some(*b),
                _ => None,
            }
        } else {
            None
        }
    }

    /// Helper: extract an i32 property.
    fn get_smi(obj: &JsValue, key: &str) -> Option<i32> {
        if let JsValue::PlainObject(map) = obj {
            match map.borrow().get(key)? {
                JsValue::Smi(n) => Some(*n),
                _ => None,
            }
        } else {
            None
        }
    }

    #[test]
    fn test_regexp_construct_basic() {
        let re = regexp_construct(&[JsValue::String(r"\d+".into()), JsValue::String("g".into())])
            .unwrap();
        assert_eq!(get_str(&re, "source").as_deref(), Some(r"\d+"));
        assert_eq!(get_str(&re, "flags").as_deref(), Some("g"));
        assert_eq!(get_bool(&re, "global"), Some(true));
        assert_eq!(get_bool(&re, "sticky"), Some(false));
    }

    #[test]
    fn test_regexp_construct_defaults() {
        let re = regexp_construct(&[]).unwrap();
        assert_eq!(get_str(&re, "source").as_deref(), Some(""));
        assert_eq!(get_str(&re, "flags").as_deref(), Some(""));
    }

    #[test]
    fn test_exec_returns_match_object() {
        let re = regexp_construct(&[JsValue::String(r"(\d+)".into()), JsValue::String("".into())])
            .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let result = f(vec![JsValue::String("price 42 dollars".into())]).unwrap();
                assert_eq!(get_str(&result, "0").as_deref(), Some("42"));
                assert_eq!(get_str(&result, "1").as_deref(), Some("42"));
                assert_eq!(get_smi(&result, "index"), Some(6));
            } else {
                panic!("exec should be NativeFunction");
            }
        }
    }

    #[test]
    fn test_exec_named_groups() {
        let re = regexp_construct(&[
            JsValue::String(r"(?<year>\d{4})-(?<month>\d{2})".into()),
            JsValue::String("u".into()),
        ])
        .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let result = f(vec![JsValue::String("2024-07".into())]).unwrap();
                assert_eq!(get_str(&result, "0").as_deref(), Some("2024-07"));
                // Check groups object
                if let JsValue::PlainObject(groups_map) = &result {
                    let groups = groups_map.borrow().get("groups").cloned();
                    if let Some(JsValue::PlainObject(g)) = groups {
                        assert_eq!(
                            g.borrow().get("year"),
                            Some(&JsValue::String("2024".into()))
                        );
                        assert_eq!(g.borrow().get("month"), Some(&JsValue::String("07".into())));
                    } else {
                        panic!("groups should be PlainObject");
                    }
                }
            }
        }
    }

    #[test]
    fn test_exec_with_indices() {
        let re = regexp_construct(&[
            JsValue::String(r"(\d+)".into()),
            JsValue::String("d".into()),
        ])
        .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let result = f(vec![JsValue::String("abc 42 end".into())]).unwrap();
                assert_eq!(get_str(&result, "0").as_deref(), Some("42"));
                // Check indices
                if let JsValue::PlainObject(result_map) = &result {
                    let indices = result_map.borrow().get("indices").cloned();
                    assert!(indices.is_some(), "indices should be present with /d flag");
                    if let Some(JsValue::PlainObject(idx)) = indices {
                        // indices[0] = [4, 6]
                        let idx0 = idx.borrow().get("0").cloned();
                        if let Some(JsValue::Array(arr)) = idx0 {
                            assert_eq!(arr[0], JsValue::Smi(4));
                            assert_eq!(arr[1], JsValue::Smi(6));
                        } else {
                            panic!("indices[0] should be an array");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_test_method() {
        let re = regexp_construct(&[JsValue::String(r"\d+".into()), JsValue::String("".into())])
            .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let test_fn = map.borrow().get("test").cloned().unwrap();
            if let JsValue::NativeFunction(f) = test_fn {
                assert_eq!(
                    f(vec![JsValue::String("hello 42".into())]).unwrap(),
                    JsValue::Boolean(true)
                );
                assert_eq!(
                    f(vec![JsValue::String("no digits".into())]).unwrap(),
                    JsValue::Boolean(false)
                );
            }
        }
    }

    #[test]
    fn test_to_string_method() {
        let re = regexp_construct(&[JsValue::String("foo".into()), JsValue::String("gi".into())])
            .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let ts_fn = map.borrow().get("toString").cloned().unwrap();
            if let JsValue::NativeFunction(f) = ts_fn {
                assert_eq!(f(vec![]).unwrap(), JsValue::String("/foo/gi".into()));
            }
        }
    }

    #[test]
    fn test_symbol_match_all_produces_iterator() {
        let re = regexp_construct(&[JsValue::String(r"\d+".into()), JsValue::String("g".into())])
            .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let ma_fn = map.borrow().get("__symbol_match_all__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = ma_fn {
                let result = f(vec![JsValue::String("a1 b22 c333".into())]).unwrap();
                if let JsValue::Iterator(iter) = result {
                    let first = iter.borrow_mut().next_item();
                    assert!(first.is_some());
                    if let Some(JsValue::PlainObject(m)) = first {
                        assert_eq!(m.borrow().get("0"), Some(&JsValue::String("1".into())));
                    }
                } else {
                    panic!("expected Iterator");
                }
            }
        }
    }

    #[test]
    fn test_flag_accessors() {
        let re = regexp_construct(&[
            JsValue::String("a".into()),
            JsValue::String("gimsdy".into()),
        ])
        .unwrap();
        assert_eq!(get_bool(&re, "global"), Some(true));
        assert_eq!(get_bool(&re, "ignoreCase"), Some(true));
        assert_eq!(get_bool(&re, "multiline"), Some(true));
        assert_eq!(get_bool(&re, "dotAll"), Some(true));
        assert_eq!(get_bool(&re, "sticky"), Some(true));
        assert_eq!(get_bool(&re, "hasIndices"), Some(true));
        assert_eq!(get_bool(&re, "unicode"), Some(false));
        assert_eq!(get_bool(&re, "unicodeSets"), Some(false));
    }
}
