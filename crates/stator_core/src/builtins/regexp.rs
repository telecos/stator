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
use std::rc::{Rc, Weak};

use crate::builtins::string::encode_utf16;
use crate::objects::map::PropertyAttributes;
use crate::objects::property_map::{INTERNAL_PROTO_PROPERTY_KEY, PropertyMap};

use crate::error::StatorResult;
use crate::interpreter::dispatch_call_value;
use crate::objects::regexp::{JsRegExp, RegExpFlags, RegExpMatch, SymbolMatchResult};
use crate::objects::value::{JsValue, NativeIterator};

// ── Annex B legacy static properties ─────────────────────────────────────────

// Per-thread storage for the legacy `RegExp.$1`–`$9`, `RegExp.input`,
// `RegExp.lastMatch`, `RegExp.lastParen`, `RegExp.leftContext`, and
// `RegExp.rightContext` static properties (Annex B §B.2.4).
//
// Updated after every successful `exec` / `test`.
thread_local! {
    static REGEXP_STATICS: RefCell<RegExpStatics> = RefCell::new(RegExpStatics::default());
}

/// Snapshot of the most recent successful match.
#[derive(Default, Clone)]
struct RegExpStatics {
    input: String,
    last_match: String,
    last_paren: String,
    left_context: String,
    right_context: String,
    captures: [String; 9],
}

/// Record a successful match into the thread-local statics.
fn update_regexp_statics(m: &RegExpMatch) {
    REGEXP_STATICS.with(|cell| {
        let mut s = cell.borrow_mut();
        s.input.clone_from(&m.input);
        s.last_match.clone_from(&m.matched);
        s.left_context = m.input[..m.index].to_string();
        let end = m.index + m.matched.len();
        s.right_context = if end <= m.input.len() {
            m.input[end..].to_string()
        } else {
            String::new()
        };
        // last paren = last non-empty capture
        s.last_paren = m
            .captures
            .iter()
            .rev()
            .find_map(|c| c.clone())
            .unwrap_or_default();
        for (i, slot) in s.captures.iter_mut().enumerate() {
            *slot = m
                .captures
                .get(i)
                .and_then(|c| c.clone())
                .unwrap_or_default();
        }
    });
}

/// Read a legacy static property by key (e.g. `"$1"`, `"input"`, `"lastMatch"`).
pub(crate) fn regexp_static_get(key: &str) -> JsValue {
    REGEXP_STATICS.with(|cell| {
        let s = cell.borrow();
        let val = match key {
            "$1" => &s.captures[0],
            "$2" => &s.captures[1],
            "$3" => &s.captures[2],
            "$4" => &s.captures[3],
            "$5" => &s.captures[4],
            "$6" => &s.captures[5],
            "$7" => &s.captures[6],
            "$8" => &s.captures[7],
            "$9" => &s.captures[8],
            "input" | "$_" => &s.input,
            "lastMatch" | "$&" => &s.last_match,
            "lastParen" | "$+" => &s.last_paren,
            "leftContext" | "$`" => &s.left_context,
            "rightContext" | "$'" => &s.right_context,
            _ => return JsValue::Undefined,
        };
        JsValue::String(val.clone().into())
    })
}

/// Create a new [`JsRegExp`] from positional `JsValue` arguments.
///
/// Handles three cases per §22.2.3.1:
///
/// 1. `new RegExp("pattern", "flags")` — compile from strings.
/// 2. `new RegExp(existingRegExp)` — clone with the same pattern and flags.
/// 3. `new RegExp(existingRegExp, "newFlags")` — clone with overridden flags.
pub fn regexp_construct(args: &[JsValue]) -> StatorResult<JsValue> {
    let first = args.first().unwrap_or(&JsValue::Undefined);

    // §22.2.3.1 step 4: if the first argument is itself a RegExp, extract
    // its source and flags rather than stringifying to "/pattern/flags".
    if let JsValue::PlainObject(map) = first {
        let is_regexp = matches!(
            map.borrow().get("__is_regexp__"),
            Some(JsValue::Boolean(true))
        );
        if is_regexp {
            let borrow = map.borrow();
            let source = match borrow.get("source") {
                Some(JsValue::String(s)) => s.to_string(),
                _ => String::new(),
            };
            let existing_flags = match borrow.get("flags") {
                Some(JsValue::String(s)) => s.to_string(),
                _ => String::new(),
            };
            drop(borrow);

            let flags = match args.get(1) {
                Some(JsValue::Undefined) | None => existing_flags,
                Some(v) => v.to_js_string()?,
            };
            let re = JsRegExp::new(&source, &flags)?;
            return Ok(wrap_regexp(re));
        }
    }

    let pattern = match first {
        JsValue::Undefined => String::new(),
        v => v.to_js_string()?,
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
///
/// The closures for `test`, `exec`, and the `Symbol.*` methods hold a
/// [`Weak`] back-reference to the property map so they can synchronise the
/// user-visible `lastIndex` property with the internal [`JsRegExp`] state
/// before and after every invocation.
pub fn wrap_regexp(re: JsRegExp) -> JsValue {
    let re = Rc::new(re);
    let props_rc: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));
    let weak = Rc::downgrade(&props_rc);

    // ── Read-only accessors ─────────────────────────────────────────────
    {
        let mut props = props_rc.borrow_mut();
        props.insert(
            "source".into(),
            JsValue::String(re.pattern().to_string().into()),
        );
        props.insert(
            "flags".into(),
            JsValue::String(re.flags().to_flags_string().into()),
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
            JsValue::Boolean(
                re.flags().contains(RegExpFlags::UNICODE)
                    || re.flags().contains(RegExpFlags::UNICODE_SETS),
            ),
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
        insert_regexp_getter(
            &mut props,
            "source",
            JsValue::NativeFunction(Rc::new({
                let re = Rc::clone(&re);
                move |_args: Vec<JsValue>| Ok(JsValue::String(re.source_text().into()))
            })),
        );
        insert_regexp_getter(
            &mut props,
            "flags",
            JsValue::NativeFunction(Rc::new({
                let re = Rc::clone(&re);
                move |_args: Vec<JsValue>| Ok(JsValue::String(re.flags().to_flags_string().into()))
            })),
        );
        insert_regexp_getter(&mut props, "global", bool_getter(&re, RegExpFlags::GLOBAL));
        insert_regexp_getter(
            &mut props,
            "ignoreCase",
            bool_getter(&re, RegExpFlags::IGNORE_CASE),
        );
        insert_regexp_getter(
            &mut props,
            "multiline",
            bool_getter(&re, RegExpFlags::MULTILINE),
        );
        insert_regexp_getter(&mut props, "dotAll", bool_getter(&re, RegExpFlags::DOT_ALL));
        insert_regexp_getter(
            &mut props,
            "unicode",
            JsValue::NativeFunction(Rc::new({
                let re = Rc::clone(&re);
                move |_args: Vec<JsValue>| {
                    Ok(JsValue::Boolean(
                        re.flags().contains(RegExpFlags::UNICODE)
                            || re.flags().contains(RegExpFlags::UNICODE_SETS),
                    ))
                }
            })),
        );
        insert_regexp_getter(
            &mut props,
            "unicodeSets",
            bool_getter(&re, RegExpFlags::UNICODE_SETS),
        );
        insert_regexp_getter(&mut props, "sticky", bool_getter(&re, RegExpFlags::STICKY));
        insert_regexp_getter(
            &mut props,
            "hasIndices",
            bool_getter(&re, RegExpFlags::HAS_INDICES),
        );

        // ── lastIndex (read/write) ──────────────────────────────────────
        props.insert("lastIndex".into(), JsValue::Smi(re.last_index() as i32));

        // Sentinel so the interpreter can identify this PlainObject as a RegExp.
        props.insert("__is_regexp__".into(), JsValue::Boolean(true));
    }

    // ── Prototype methods ───────────────────────────────────────────────

    // test(string)
    {
        let re_test = Rc::clone(&re);
        let re_test_exec = Rc::clone(&re);
        let w = weak.clone();
        props_rc.borrow_mut().insert(
            "test".into(),
            JsValue::NativeFunction(Rc::new(move |args: Vec<JsValue>| {
                sync_last_index_from_props(&w, &re_test);
                let input = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
                // Use exec internally so we can capture match data for statics.
                let matched = re_test_exec.exec(&input);
                if let Some(ref m) = matched {
                    update_regexp_statics(m);
                }
                sync_last_index_to_props(&w, &re_test);
                Ok(JsValue::Boolean(matched.is_some()))
            })),
        );
    }

    // exec(string)
    {
        let re_exec = Rc::clone(&re);
        let w = weak.clone();
        props_rc.borrow_mut().insert(
            "exec".into(),
            JsValue::NativeFunction(Rc::new(move |args: Vec<JsValue>| {
                sync_last_index_from_props(&w, &re_exec);
                let input = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
                let result = match re_exec.exec(&input) {
                    Some(m) => {
                        update_regexp_statics(&m);
                        match_to_js(&m)
                    }
                    None => JsValue::Null,
                };
                sync_last_index_to_props(&w, &re_exec);
                Ok(result)
            })),
        );
    }

    // toString()
    {
        let re_str = Rc::clone(&re);
        props_rc.borrow_mut().insert(
            "toString".into(),
            JsValue::NativeFunction(Rc::new(move |_args: Vec<JsValue>| {
                Ok(JsValue::String(re_str.to_string().into()))
            })),
        );
    }

    // [Symbol.match](string)
    {
        let re_match = Rc::clone(&re);
        let w = weak.clone();
        props_rc.borrow_mut().insert(
            "__symbol_match__".into(),
            JsValue::NativeFunction(Rc::new(move |args: Vec<JsValue>| {
                sync_last_index_from_props(&w, &re_match);
                let input = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
                let result = match re_match.symbol_match(&input) {
                    None => JsValue::Null,
                    Some(SymbolMatchResult::Single(m)) => match_to_js(&m),
                    Some(SymbolMatchResult::All(v)) => {
                        let arr: Vec<JsValue> =
                            v.into_iter().map(|s| JsValue::String(s.into())).collect();
                        JsValue::new_array(arr)
                    }
                };
                sync_last_index_to_props(&w, &re_match);
                Ok(result)
            })),
        );
    }

    // [Symbol.replace](string, replacement)
    {
        let re_replace = Rc::clone(&re);
        let w = weak.clone();
        props_rc.borrow_mut().insert(
            "__symbol_replace__".into(),
            JsValue::NativeFunction(Rc::new(move |args: Vec<JsValue>| {
                sync_last_index_from_props(&w, &re_replace);
                let input = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
                let replacement = args.get(1).unwrap_or(&JsValue::Undefined).clone();
                let result = if is_callable(&replacement) {
                    regexp_replace_with_callback(&re_replace, &input, &replacement, &w)?
                } else {
                    re_replace.symbol_replace(&input, &replacement.to_js_string()?)
                };
                sync_last_index_to_props(&w, &re_replace);
                Ok(JsValue::String(result.into()))
            })),
        );
    }

    // [Symbol.search](string)
    {
        let re_search = Rc::clone(&re);
        let w = weak.clone();
        props_rc.borrow_mut().insert(
            "__symbol_search__".into(),
            JsValue::NativeFunction(Rc::new(move |args: Vec<JsValue>| {
                sync_last_index_from_props(&w, &re_search);
                let input = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
                let idx = re_search.symbol_search(&input);
                sync_last_index_to_props(&w, &re_search);
                Ok(if idx >= 0 {
                    JsValue::Smi(utf16_index(&input, idx as usize))
                } else {
                    JsValue::Smi(-1)
                })
            })),
        );
    }

    // [Symbol.split](string[, limit])
    {
        let re_split = Rc::clone(&re);
        let w = weak.clone();
        props_rc.borrow_mut().insert(
            "__symbol_split__".into(),
            JsValue::NativeFunction(Rc::new(move |args: Vec<JsValue>| {
                sync_last_index_from_props(&w, &re_split);
                let input = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
                let limit = match args.get(1) {
                    Some(JsValue::Undefined) | None => None,
                    Some(v) => Some(crate::builtins::util::clamped_f64_to_usize(v.to_number()?)),
                };
                let parts = re_split.symbol_split(&input, limit);
                sync_last_index_to_props(&w, &re_split);
                Ok(JsValue::new_array(
                    parts
                        .into_iter()
                        .map(|s| match s {
                            Some(s) => JsValue::String(s.into()),
                            None => JsValue::Undefined,
                        })
                        .collect(),
                ))
            })),
        );
    }

    // [Symbol.matchAll](string)
    {
        let re_match_all = Rc::clone(&re);
        let w = weak.clone();
        props_rc.borrow_mut().insert(
            "__symbol_match_all__".into(),
            JsValue::NativeFunction(Rc::new(move |args: Vec<JsValue>| {
                sync_last_index_from_props(&w, &re_match_all);
                let input = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
                let matches = re_match_all.symbol_match_all(&input);
                sync_last_index_to_props(&w, &re_match_all);
                let items: Vec<JsValue> = matches.iter().map(match_to_js).collect();
                Ok(JsValue::Iterator(NativeIterator::from_items(items)))
            })),
        );
    }

    JsValue::PlainObject(props_rc)
}

/// Read the user-visible `lastIndex` property from the object and push it
/// into the internal [`JsRegExp`] state so the next `exec`/`test` starts
/// at the right position.
fn sync_last_index_from_props(weak: &Weak<RefCell<PropertyMap>>, re: &JsRegExp) {
    if let Some(rc) = weak.upgrade()
        && let Some(val) = rc.borrow().get("lastIndex").cloned()
    {
        let idx = val.to_length().unwrap_or(0) as usize;
        re.set_last_index(idx);
    }
}

/// Write the internal [`JsRegExp`] `lastIndex` back to the property map so
/// user code sees the updated value.
fn sync_last_index_to_props(weak: &Weak<RefCell<PropertyMap>>, re: &JsRegExp) {
    if let Some(rc) = weak.upgrade() {
        rc.borrow_mut()
            .insert("lastIndex".into(), JsValue::Smi(re.last_index() as i32));
    }
}

fn is_callable(value: &JsValue) -> bool {
    match value {
        JsValue::Function(_) | JsValue::NativeFunction(_) => true,
        JsValue::PlainObject(map) => map.borrow().contains_key("__call__"),
        _ => false,
    }
}

fn insert_regexp_getter(props: &mut PropertyMap, name: &str, getter: JsValue) {
    props.insert_with_attrs(
        format!("__get_{name}__"),
        getter,
        PropertyAttributes::CONFIGURABLE,
    );
}

fn bool_getter(re: &Rc<JsRegExp>, flag: RegExpFlags) -> JsValue {
    let re = Rc::clone(re);
    JsValue::NativeFunction(Rc::new(move |_args: Vec<JsValue>| {
        Ok(JsValue::Boolean(re.flags().contains(flag)))
    }))
}

fn utf16_index(input: &str, byte_index: usize) -> i32 {
    encode_utf16(&input[..byte_index]).len() as i32
}

fn call_replace_callback(
    callback: &JsValue,
    input: &str,
    matched: &RegExpMatch,
) -> StatorResult<String> {
    let mut args = Vec::with_capacity(
        4 + matched.captures.len() + usize::from(!matched.named_groups.is_empty()),
    );
    args.push(JsValue::String(matched.matched.clone().into()));
    for capture in &matched.captures {
        match capture {
            Some(value) => args.push(JsValue::String(value.clone().into())),
            None => args.push(JsValue::Undefined),
        }
    }
    args.push(JsValue::Smi(utf16_index(input, matched.index)));
    args.push(JsValue::String(input.to_string().into()));
    if !matched.named_groups.is_empty() {
        let mut groups = PropertyMap::new();
        groups.insert(INTERNAL_PROTO_PROPERTY_KEY.into(), JsValue::Null);
        for (key, value) in &matched.named_groups {
            groups.insert(
                key.clone(),
                match value {
                    Some(value) => JsValue::String(value.clone().into()),
                    None => JsValue::Undefined,
                },
            );
        }
        args.push(JsValue::PlainObject(Rc::new(RefCell::new(groups))));
    }
    dispatch_call_value(callback, args)?.to_js_string()
}

fn regexp_replace_with_callback(
    re: &JsRegExp,
    input: &str,
    replacement: &JsValue,
    _weak: &Weak<RefCell<PropertyMap>>,
) -> StatorResult<String> {
    let global = re.flags().contains(RegExpFlags::GLOBAL);

    if global {
        re.set_last_index(0);
        let matches = re.symbol_match_all(input);
        if matches.is_empty() {
            re.set_last_index(0);
            return Ok(input.to_string());
        }

        let mut result = String::new();
        let mut next_source_position = 0usize;
        for matched in matches {
            let end = matched.index + matched.matched.len();
            result.push_str(&input[next_source_position..matched.index]);
            result.push_str(&call_replace_callback(replacement, input, &matched)?);
            next_source_position = end;
        }
        result.push_str(&input[next_source_position..]);
        re.set_last_index(0);
        Ok(result)
    } else if let Some(matched) = re.exec(input) {
        let mut result = String::new();
        let end = matched.index + matched.matched.len();
        result.push_str(&input[..matched.index]);
        result.push_str(&call_replace_callback(replacement, input, &matched)?);
        result.push_str(&input[end..]);
        Ok(result)
    } else {
        Ok(input.to_string())
    }
}

/// Convert a [`RegExpMatch`] into a `JsValue::PlainObject` matching the
/// ECMAScript `RegExpExecResult` shape:
///
/// * Index `0` → full match (via numeric string key)
/// * Index `1..` → capture groups
/// * `index` → UTF-16 code unit offset of the match start
/// * `input` → original input string
/// * `groups` → named groups object (or `undefined`)
/// * `indices` → `{ 0: [s,e], 1: [s,e], …, groups: {…} }` when `/d` flag
fn match_to_js(m: &crate::objects::regexp::RegExpMatch) -> JsValue {
    let mut props = PropertyMap::new();

    // [0] = full match
    props.insert("0".into(), JsValue::String(m.matched.clone().into()));
    // [1..] = captures
    for (i, cap) in m.captures.iter().enumerate() {
        let key = (i + 1).to_string();
        props.insert(
            key,
            match cap {
                Some(s) => JsValue::String(s.clone().into()),
                None => JsValue::Undefined,
            },
        );
    }
    // ECMAScript specifies `index` as a UTF-16 code unit offset.
    props.insert("index".into(), JsValue::Smi(utf16_index(&m.input, m.index)));
    props.insert("input".into(), JsValue::String(m.input.clone().into()));

    // groups — null-prototype object (Object.create(null) per spec)
    props.insert("groups".into(), named_groups_to_js(m));

    // indices (only present when /d flag was used)
    if let Some(ref idx) = m.indices {
        let mut idx_props = PropertyMap::new();
        for (i, pair) in idx.pairs.iter().enumerate() {
            let val = match pair {
                Some((s, e)) => JsValue::new_array(vec![
                    JsValue::Smi(utf16_index(&m.input, *s)),
                    JsValue::Smi(utf16_index(&m.input, *e)),
                ]),
                None => JsValue::Undefined,
            };
            idx_props.insert(i.to_string(), val);
        }
        // indices.groups — null-prototype object when named groups exist,
        // undefined otherwise (per §22.2.7.8 step 26-28).
        if m.named_groups.is_empty() {
            idx_props.insert("groups".into(), JsValue::Undefined);
        } else {
            let mut g = PropertyMap::new();
            g.insert(INTERNAL_PROTO_PROPERTY_KEY.into(), JsValue::Null);
            for (k, (s, e)) in &idx.groups {
                g.insert(
                    k.clone(),
                    JsValue::new_array(vec![
                        JsValue::Smi(utf16_index(&m.input, *s)),
                        JsValue::Smi(utf16_index(&m.input, *e)),
                    ]),
                );
            }
            idx_props.insert(
                "groups".into(),
                JsValue::PlainObject(Rc::new(RefCell::new(g))),
            );
        }
        // Mark indices as array-like (spec: indices is an Array exotic).
        idx_props.insert("length".into(), JsValue::Smi(idx.pairs.len() as i32));
        idx_props.insert("__is_array__".into(), JsValue::Boolean(true));
        props.insert(
            "indices".into(),
            JsValue::PlainObject(Rc::new(RefCell::new(idx_props))),
        );
    }

    // length = 1 + captures.len() (spec compatibility)
    props.insert("length".into(), JsValue::Smi((1 + m.captures.len()) as i32));

    // Mark as array-like so Array.isArray() returns true (spec: exec
    // returns an Array exotic object).
    props.insert("__is_array__".into(), JsValue::Boolean(true));

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

fn named_groups_to_js(m: &RegExpMatch) -> JsValue {
    if m.named_groups.is_empty() {
        JsValue::Undefined
    } else {
        let mut groups = PropertyMap::new();
        groups.insert(INTERNAL_PROTO_PROPERTY_KEY.into(), JsValue::Null);
        for (k, v) in &m.named_groups {
            groups.insert(
                k.clone(),
                match v {
                    Some(s) => JsValue::String(s.clone().into()),
                    None => JsValue::Undefined,
                },
            );
        }
        JsValue::PlainObject(Rc::new(RefCell::new(groups)))
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: extract a string property from a PlainObject JsValue.
    fn get_str(obj: &JsValue, key: &str) -> Option<String> {
        if let JsValue::PlainObject(map) = obj {
            match map.borrow().get(key)? {
                JsValue::String(s) => Some(s.to_string()),
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
                            assert_eq!(arr.borrow()[0], JsValue::Smi(4));
                            assert_eq!(arr.borrow()[1], JsValue::Smi(6));
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
    fn test_symbol_match_global_resets_last_index() {
        let re =
            regexp_construct(&[JsValue::String("a".into()), JsValue::String("g".into())]).unwrap();
        if let JsValue::PlainObject(map) = &re {
            let match_fn = map.borrow().get("__symbol_match__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = match_fn {
                let result = f(vec![JsValue::String("a_a".into())]).unwrap();
                assert!(matches!(result, JsValue::Array(_)));
                assert_eq!(get_smi(&re, "lastIndex"), Some(0));
            }
        }
    }

    #[test]
    fn test_symbol_search_preserves_last_index() {
        let re =
            regexp_construct(&[JsValue::String("a".into()), JsValue::String("g".into())]).unwrap();
        if let JsValue::PlainObject(map) = &re {
            map.borrow_mut().insert("lastIndex".into(), JsValue::Smi(2));
            let search_fn = map.borrow().get("__symbol_search__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = search_fn {
                let result = f(vec![JsValue::String("ba".into())]).unwrap();
                assert_eq!(result, JsValue::Smi(1));
                assert_eq!(get_smi(&re, "lastIndex"), Some(2));
            }
        }
    }

    #[test]
    fn test_symbol_split_keeps_undefined_capture() {
        let re = regexp_construct(&[JsValue::String("-(x)?".into()), JsValue::String("".into())])
            .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let split_fn = map.borrow().get("__symbol_split__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = split_fn {
                let result = f(vec![JsValue::String("a-b".into())]).unwrap();
                if let JsValue::Array(items) = result {
                    let items = items.borrow();
                    assert_eq!(items[0], JsValue::String("a".into()));
                    assert_eq!(items[1], JsValue::Undefined);
                    assert_eq!(items[2], JsValue::String("b".into()));
                } else {
                    panic!("expected Array");
                }
            }
        }
    }

    #[test]
    fn test_replace_callback_receives_groups_argument() {
        let re = regexp_construct(&[
            JsValue::String(r"(?<y>\d{4})-(?<m>\d{2})".into()),
            JsValue::String("".into()),
        ])
        .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let replace_fn = map.borrow().get("__symbol_replace__").cloned().unwrap();
            let callback = JsValue::NativeFunction(Rc::new(|args: Vec<JsValue>| {
                let groups = args.last().cloned().unwrap_or(JsValue::Undefined);
                if let JsValue::PlainObject(groups) = groups {
                    let year = groups
                        .borrow()
                        .get("y")
                        .cloned()
                        .unwrap_or(JsValue::Undefined);
                    let month = groups
                        .borrow()
                        .get("m")
                        .cloned()
                        .unwrap_or(JsValue::Undefined);
                    Ok(JsValue::String(
                        format!("{}-{}", year.to_js_string()?, month.to_js_string()?).into(),
                    ))
                } else {
                    Ok(JsValue::String("missing".into()))
                }
            }));
            if let JsValue::NativeFunction(f) = replace_fn {
                let result = f(vec![JsValue::String("2024-07".into()), callback]).unwrap();
                assert_eq!(result, JsValue::String("2024-07".into()));
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

    // ── RegExp constructor with existing RegExp (clone) ──────────────────

    #[test]
    fn test_construct_from_regexp_clones() {
        let original =
            regexp_construct(&[JsValue::String("abc".into()), JsValue::String("gi".into())])
                .unwrap();
        let cloned = regexp_construct(&[original]).unwrap();
        assert_eq!(get_str(&cloned, "source").as_deref(), Some("abc"));
        assert_eq!(get_str(&cloned, "flags").as_deref(), Some("gi"));
        assert_eq!(get_bool(&cloned, "global"), Some(true));
        assert_eq!(get_bool(&cloned, "ignoreCase"), Some(true));
    }

    #[test]
    fn test_construct_from_regexp_overrides_flags() {
        let original =
            regexp_construct(&[JsValue::String("abc".into()), JsValue::String("gi".into())])
                .unwrap();
        let cloned = regexp_construct(&[original, JsValue::String("m".into())]).unwrap();
        assert_eq!(get_str(&cloned, "source").as_deref(), Some("abc"));
        assert_eq!(get_str(&cloned, "flags").as_deref(), Some("m"));
        assert_eq!(get_bool(&cloned, "global"), Some(false));
        assert_eq!(get_bool(&cloned, "multiline"), Some(true));
    }

    #[test]
    fn test_construct_from_regexp_undefined_flags_keeps_original() {
        let original =
            regexp_construct(&[JsValue::String("xyz".into()), JsValue::String("s".into())])
                .unwrap();
        let cloned = regexp_construct(&[original, JsValue::Undefined]).unwrap();
        assert_eq!(get_str(&cloned, "flags").as_deref(), Some("s"));
        assert_eq!(get_bool(&cloned, "dotAll"), Some(true));
    }

    // ── lastIndex synchronisation ────────────────────────────────────────

    #[test]
    fn test_last_index_synced_after_exec_global() {
        let re =
            regexp_construct(&[JsValue::String("a".into()), JsValue::String("g".into())]).unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let _ = f(vec![JsValue::String("aaa".into())]).unwrap();
                // After first exec, lastIndex should advance past the first 'a'.
                assert_eq!(get_smi(&re, "lastIndex"), Some(1));
            }
        }
    }

    #[test]
    fn test_last_index_readable_after_test_global() {
        let re =
            regexp_construct(&[JsValue::String("b".into()), JsValue::String("g".into())]).unwrap();
        if let JsValue::PlainObject(map) = &re {
            let test_fn = map.borrow().get("test").cloned().unwrap();
            if let JsValue::NativeFunction(f) = test_fn {
                assert_eq!(
                    f(vec![JsValue::String("abc".into())]).unwrap(),
                    JsValue::Boolean(true)
                );
                assert_eq!(get_smi(&re, "lastIndex"), Some(2));
            }
        }
    }

    #[test]
    fn test_last_index_writable_affects_exec() {
        let re =
            regexp_construct(&[JsValue::String("a".into()), JsValue::String("g".into())]).unwrap();
        if let JsValue::PlainObject(map) = &re {
            // Set lastIndex = 2 so exec starts searching at position 2.
            map.borrow_mut().insert("lastIndex".into(), JsValue::Smi(2));
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let result = f(vec![JsValue::String("a_a_a".into())]).unwrap();
                // The first 'a' at index 0 should be skipped; match at index 2.
                assert_eq!(get_smi(&result, "index"), Some(2));
                assert_eq!(get_smi(&re, "lastIndex"), Some(3));
            }
        }
    }

    #[test]
    fn test_last_index_zero_width_match_advances() {
        let re = regexp_construct(&[JsValue::String("(?:)".into()), JsValue::String("g".into())])
            .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let _ = f(vec![JsValue::String("ab".into())]).unwrap();
                assert_eq!(get_smi(&re, "lastIndex"), Some(1));
            }
        }
    }

    // ── exec result is array-like ────────────────────────────────────────

    #[test]
    fn test_exec_result_has_array_marker() {
        let re =
            regexp_construct(&[JsValue::String("a".into()), JsValue::String("".into())]).unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let result = f(vec![JsValue::String("a".into())]).unwrap();
                assert_eq!(get_bool(&result, "__is_array__"), Some(true));
                assert_eq!(get_smi(&result, "length"), Some(1));
            }
        }
    }

    // ── groups is null-prototype object ──────────────────────────────────

    #[test]
    fn test_exec_groups_null_prototype() {
        let re = regexp_construct(&[
            JsValue::String(r"(?<a>\d+)".into()),
            JsValue::String("".into()),
        ])
        .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let result = f(vec![JsValue::String("42".into())]).unwrap();
                if let JsValue::PlainObject(result_map) = &result {
                    let groups = result_map.borrow().get("groups").cloned().unwrap();
                    if let JsValue::PlainObject(g) = groups {
                        // Null-prototype: __proto__ is Null
                        assert_eq!(
                            g.borrow().get(INTERNAL_PROTO_PROPERTY_KEY),
                            Some(&JsValue::Null)
                        );
                    } else {
                        panic!("groups should be PlainObject");
                    }
                }
            }
        }
    }

    #[test]
    fn test_exec_no_named_groups_returns_undefined() {
        let re = regexp_construct(&[JsValue::String(r"\d+".into()), JsValue::String("".into())])
            .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let result = f(vec![JsValue::String("42".into())]).unwrap();
                if let JsValue::PlainObject(result_map) = &result {
                    let groups = result_map.borrow().get("groups").cloned().unwrap();
                    assert_eq!(groups, JsValue::Undefined);
                }
            }
        }
    }

    // ── /d flag indices integration tests ────────────────────────────────

    #[test]
    fn test_exec_indices_array_like() {
        let re = regexp_construct(&[JsValue::String(r"\d+".into()), JsValue::String("d".into())])
            .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let result = f(vec![JsValue::String("abc 42".into())]).unwrap();
                if let JsValue::PlainObject(result_map) = &result {
                    let indices = result_map.borrow().get("indices").cloned().unwrap();
                    // indices should be array-like
                    assert_eq!(get_bool(&indices, "__is_array__"), Some(true));
                    assert_eq!(get_smi(&indices, "length"), Some(1));
                }
            }
        }
    }

    #[test]
    fn test_exec_indices_with_named_groups_null_proto() {
        let re = regexp_construct(&[
            JsValue::String(r"(?<word>\w+)".into()),
            JsValue::String("d".into()),
        ])
        .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let result = f(vec![JsValue::String("hello".into())]).unwrap();
                if let JsValue::PlainObject(result_map) = &result {
                    let indices = result_map.borrow().get("indices").cloned().unwrap();
                    if let JsValue::PlainObject(idx_map) = indices {
                        let ig = idx_map.borrow().get("groups").cloned().unwrap();
                        if let JsValue::PlainObject(g) = ig {
                            assert_eq!(
                                g.borrow().get(INTERNAL_PROTO_PROPERTY_KEY),
                                Some(&JsValue::Null)
                            );
                            // "word" should be [0, 5]
                            let word_idx = g.borrow().get("word").cloned().unwrap();
                            if let JsValue::Array(arr) = word_idx {
                                assert_eq!(arr.borrow()[0], JsValue::Smi(0));
                                assert_eq!(arr.borrow()[1], JsValue::Smi(5));
                            } else {
                                panic!("word index should be array");
                            }
                        } else {
                            panic!("indices.groups should be PlainObject");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_exec_indices_groups_undefined_without_named() {
        let re = regexp_construct(&[
            JsValue::String(r"(\d+)".into()),
            JsValue::String("d".into()),
        ])
        .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let result = f(vec![JsValue::String("42".into())]).unwrap();
                if let JsValue::PlainObject(result_map) = &result {
                    let indices = result_map.borrow().get("indices").cloned().unwrap();
                    if let JsValue::PlainObject(idx_map) = indices {
                        let groups = idx_map.borrow().get("groups").cloned().unwrap();
                        assert_eq!(groups, JsValue::Undefined);
                    }
                }
            }
        }
    }

    #[test]
    fn test_exec_indices_capture_pairs() {
        let re = regexp_construct(&[
            JsValue::String(r"(\d+)-(\d+)".into()),
            JsValue::String("d".into()),
        ])
        .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let result = f(vec![JsValue::String("abc 12-34 end".into())]).unwrap();
                if let JsValue::PlainObject(result_map) = &result {
                    let indices = result_map.borrow().get("indices").cloned().unwrap();
                    assert_eq!(get_smi(&indices, "length"), Some(3));
                    // indices[0] = [4, 9] full match
                    if let JsValue::PlainObject(idx) = &indices {
                        let pair0 = idx.borrow().get("0").cloned().unwrap();
                        if let JsValue::Array(arr) = pair0 {
                            assert_eq!(arr.borrow()[0], JsValue::Smi(4));
                            assert_eq!(arr.borrow()[1], JsValue::Smi(9));
                        }
                        // indices[1] = [4, 6]
                        let pair1 = idx.borrow().get("1").cloned().unwrap();
                        if let JsValue::Array(arr) = pair1 {
                            assert_eq!(arr.borrow()[0], JsValue::Smi(4));
                            assert_eq!(arr.borrow()[1], JsValue::Smi(6));
                        }
                        // indices[2] = [7, 9]
                        let pair2 = idx.borrow().get("2").cloned().unwrap();
                        if let JsValue::Array(arr) = pair2 {
                            assert_eq!(arr.borrow()[0], JsValue::Smi(7));
                            assert_eq!(arr.borrow()[1], JsValue::Smi(9));
                        }
                    }
                }
            }
        }
    }

    // ── /v flag stub integration ─────────────────────────────────────────

    #[test]
    fn test_v_flag_construct() {
        let re =
            regexp_construct(&[JsValue::String("a".into()), JsValue::String("v".into())]).unwrap();
        assert_eq!(get_str(&re, "flags").as_deref(), Some("v"));
        assert_eq!(get_bool(&re, "unicodeSets"), Some(true));
        assert_eq!(get_bool(&re, "unicode"), Some(true));
    }

    #[test]
    fn test_v_flag_has_indices_combined() {
        let re = regexp_construct(&[JsValue::String(r"\d+".into()), JsValue::String("dv".into())])
            .unwrap();
        assert_eq!(get_str(&re, "flags").as_deref(), Some("dv"));
        assert_eq!(get_bool(&re, "hasIndices"), Some(true));
        assert_eq!(get_bool(&re, "unicodeSets"), Some(true));
    }

    // ── flags ordering integration ───────────────────────────────────────

    #[test]
    fn test_flags_accessor_order() {
        let re = regexp_construct(&[
            JsValue::String("a".into()),
            JsValue::String("ymisgd".into()),
        ])
        .unwrap();
        assert_eq!(get_str(&re, "flags").as_deref(), Some("dgimsy"));
    }

    #[test]
    fn test_flags_with_v_flag_order() {
        let re = regexp_construct(&[JsValue::String("a".into()), JsValue::String("ygdv".into())])
            .unwrap();
        assert_eq!(get_str(&re, "flags").as_deref(), Some("dgvy"));
    }

    // ── exec no match returns null ───────────────────────────────────────

    #[test]
    fn test_exec_no_match_returns_null() {
        let re =
            regexp_construct(&[JsValue::String("xyz".into()), JsValue::String("".into())]).unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let result = f(vec![JsValue::String("hello".into())]).unwrap();
                assert_eq!(result, JsValue::Null);
            }
        }
    }

    // ── named backreference integration ──────────────────────────────────

    #[test]
    fn test_named_backreference_exec() {
        let re = regexp_construct(&[
            JsValue::String(r"(?<char>.)\k<char>".into()),
            JsValue::String("".into()),
        ])
        .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let result = f(vec![JsValue::String("aabbcc".into())]).unwrap();
                assert_eq!(get_str(&result, "0").as_deref(), Some("aa"));
                // groups.char = "a"
                if let JsValue::PlainObject(result_map) = &result {
                    let groups = result_map.borrow().get("groups").cloned().unwrap();
                    if let JsValue::PlainObject(g) = groups {
                        assert_eq!(g.borrow().get("char"), Some(&JsValue::String("a".into())));
                    }
                }
            }
        }
    }

    // ── replace with named groups integration ────────────────────────────

    #[test]
    fn test_replace_named_capture_integration() {
        let re = regexp_construct(&[
            JsValue::String(r"(?<y>\d{4})-(?<m>\d{2})-(?<d>\d{2})".into()),
            JsValue::String("".into()),
        ])
        .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let replace_fn = map.borrow().get("__symbol_replace__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = replace_fn {
                let result = f(vec![
                    JsValue::String("2024-07-15".into()),
                    JsValue::String("$<d>/$<m>/$<y>".into()),
                ])
                .unwrap();
                assert_eq!(result, JsValue::String("15/07/2024".into()));
            }
        }
    }

    // ── legacy statics ───────────────────────────────────────────────────

    #[test]
    fn test_legacy_statics_updated_after_exec() {
        let re = regexp_construct(&[
            JsValue::String(r"(\d+)-(\w+)".into()),
            JsValue::String("".into()),
        ])
        .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let _ = f(vec![JsValue::String("abc 42-foo end".into())]).unwrap();
                assert_eq!(regexp_static_get("$1"), JsValue::String("42".into()));
                assert_eq!(regexp_static_get("$2"), JsValue::String("foo".into()));
                assert_eq!(
                    regexp_static_get("lastMatch"),
                    JsValue::String("42-foo".into())
                );
            }
        }
    }

    // ── d flag with nonparticipating capture ──────────────────────────────

    #[test]
    fn test_exec_indices_nonparticipating() {
        let re = regexp_construct(&[
            JsValue::String(r"(x)?(\d+)".into()),
            JsValue::String("d".into()),
        ])
        .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let result = f(vec![JsValue::String("42".into())]).unwrap();
                if let JsValue::PlainObject(result_map) = &result {
                    let indices = result_map.borrow().get("indices").cloned().unwrap();
                    if let JsValue::PlainObject(idx) = indices {
                        // indices[1] should be undefined (group didn't participate)
                        let pair1 = idx.borrow().get("1").cloned().unwrap();
                        assert_eq!(pair1, JsValue::Undefined);
                    }
                }
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Integration tests — lookbehind, dotAll, Unicode property escapes,
    //                      named backreferences, flag combinations
    // ══════════════════════════════════════════════════════════════════════

    // ── dotAll getter via wrap_regexp ─────────────────────────────────────

    #[test]
    fn test_dotall_getter_true() {
        let re = regexp_construct(&[JsValue::String("a.b".into()), JsValue::String("s".into())])
            .unwrap();
        assert_eq!(get_bool(&re, "dotAll"), Some(true));
    }

    #[test]
    fn test_dotall_getter_false() {
        let re =
            regexp_construct(&[JsValue::String("a.b".into()), JsValue::String("".into())]).unwrap();
        assert_eq!(get_bool(&re, "dotAll"), Some(false));
    }

    // ── flags includes 's' when dotAll ───────────────────────────────────

    #[test]
    fn test_flags_includes_s_when_dotall() {
        let re =
            regexp_construct(&[JsValue::String("a".into()), JsValue::String("gs".into())]).unwrap();
        let flags = get_str(&re, "flags").unwrap();
        assert!(flags.contains('s'));
        assert_eq!(flags, "gs");
    }

    // ── dotAll exec integration ──────────────────────────────────────────

    #[test]
    fn test_dotall_exec_matches_newline() {
        let re = regexp_construct(&[JsValue::String("a.b".into()), JsValue::String("s".into())])
            .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let test_fn = map.borrow().get("test").cloned().unwrap();
            if let JsValue::NativeFunction(f) = test_fn {
                assert_eq!(
                    f(vec![JsValue::String("a\nb".into())]).unwrap(),
                    JsValue::Boolean(true)
                );
            }
        }
    }

    #[test]
    fn test_no_dotall_exec_rejects_newline() {
        let re =
            regexp_construct(&[JsValue::String("a.b".into()), JsValue::String("".into())]).unwrap();
        if let JsValue::PlainObject(map) = &re {
            let test_fn = map.borrow().get("test").cloned().unwrap();
            if let JsValue::NativeFunction(f) = test_fn {
                assert_eq!(
                    f(vec![JsValue::String("a\nb".into())]).unwrap(),
                    JsValue::Boolean(false)
                );
            }
        }
    }

    // ── Lookbehind via exec integration ──────────────────────────────────

    #[test]
    fn test_lookbehind_positive_exec() {
        let re = regexp_construct(&[
            JsValue::String(r"(?<=\$)\d+".into()),
            JsValue::String("".into()),
        ])
        .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let result = f(vec![JsValue::String("$100".into())]).unwrap();
                assert_eq!(get_str(&result, "0").as_deref(), Some("100"));
            }
        }
    }

    #[test]
    fn test_lookbehind_negative_exec() {
        let re = regexp_construct(&[
            JsValue::String(r"(?<!\$)\d+".into()),
            JsValue::String("".into()),
        ])
        .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let result = f(vec![JsValue::String("42 items".into())]).unwrap();
                assert_eq!(get_str(&result, "0").as_deref(), Some("42"));
            }
        }
    }

    // ── Unicode property escapes via integration ─────────────────────────

    #[test]
    fn test_unicode_property_letter_exec() {
        let re = regexp_construct(&[
            JsValue::String(r"\p{Letter}+".into()),
            JsValue::String("u".into()),
        ])
        .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let result = f(vec![JsValue::String("42 hello".into())]).unwrap();
                assert_eq!(get_str(&result, "0").as_deref(), Some("hello"));
            }
        }
    }

    #[test]
    fn test_unicode_property_negated_exec() {
        let re = regexp_construct(&[
            JsValue::String(r"\P{Number}+".into()),
            JsValue::String("u".into()),
        ])
        .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let result = f(vec![JsValue::String("42abc99".into())]).unwrap();
                assert_eq!(get_str(&result, "0").as_deref(), Some("abc"));
            }
        }
    }

    #[test]
    fn test_unicode_property_script_greek_exec() {
        let re = regexp_construct(&[
            JsValue::String(r"\p{Script=Greek}+".into()),
            JsValue::String("u".into()),
        ])
        .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let test_fn = map.borrow().get("test").cloned().unwrap();
            if let JsValue::NativeFunction(f) = test_fn {
                assert_eq!(
                    f(vec![JsValue::String("αβγ".into())]).unwrap(),
                    JsValue::Boolean(true)
                );
                assert_eq!(
                    f(vec![JsValue::String("abc".into())]).unwrap(),
                    JsValue::Boolean(false)
                );
            }
        }
    }

    // ── Named backreference \k<name> integration ─────────────────────────

    #[test]
    fn test_named_backreference_exec_match() {
        let re = regexp_construct(&[
            JsValue::String(r"(?<w>\w+)=\k<w>".into()),
            JsValue::String("".into()),
        ])
        .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let result = f(vec![JsValue::String("abc=abc".into())]).unwrap();
                assert_eq!(get_str(&result, "0").as_deref(), Some("abc=abc"));
            }
        }
    }

    #[test]
    fn test_named_backreference_exec_no_match() {
        let re = regexp_construct(&[
            JsValue::String(r"(?<w>\w+)=\k<w>".into()),
            JsValue::String("".into()),
        ])
        .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let exec_fn = map.borrow().get("exec").cloned().unwrap();
            if let JsValue::NativeFunction(f) = exec_fn {
                let result = f(vec![JsValue::String("abc=xyz".into())]).unwrap();
                assert_eq!(result, JsValue::Null);
            }
        }
    }

    // ── Flag combinations integration ────────────────────────────────────

    #[test]
    fn test_flags_gimus_integration() {
        let re = regexp_construct(&[JsValue::String(".".into()), JsValue::String("gimus".into())])
            .unwrap();
        assert_eq!(get_str(&re, "flags").as_deref(), Some("gimsu"));
        assert_eq!(get_bool(&re, "global"), Some(true));
        assert_eq!(get_bool(&re, "ignoreCase"), Some(true));
        assert_eq!(get_bool(&re, "multiline"), Some(true));
        assert_eq!(get_bool(&re, "dotAll"), Some(true));
        assert_eq!(get_bool(&re, "unicode"), Some(true));
    }

    #[test]
    fn test_flags_gimsuy_integration() {
        let re = regexp_construct(&[
            JsValue::String(".".into()),
            JsValue::String("gimsuy".into()),
        ])
        .unwrap();
        assert_eq!(get_str(&re, "flags").as_deref(), Some("gimsuy"));
        assert_eq!(get_bool(&re, "sticky"), Some(true));
    }

    // ── Lookbehind with replace ──────────────────────────────────────────

    #[test]
    fn test_lookbehind_global_replace_integration() {
        let re = regexp_construct(&[
            JsValue::String(r"(?<=\$)\d+".into()),
            JsValue::String("g".into()),
        ])
        .unwrap();
        if let JsValue::PlainObject(map) = &re {
            let replace_fn = map.borrow().get("__symbol_replace__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = replace_fn {
                let result = f(vec![
                    JsValue::String("$100 and $200".into()),
                    JsValue::String("XXX".into()),
                ])
                .unwrap();
                assert_eq!(result, JsValue::String("$XXX and $XXX".into()));
            }
        }
    }
}
