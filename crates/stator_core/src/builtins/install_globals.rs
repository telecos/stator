//! Pre-populates the interpreter's global environment with all built-in
//! constructors, namespace objects, and global functions.
//!
//! Call [`install_globals`] when creating a fresh top-level
//! [`InterpreterFrame`][crate::interpreter::InterpreterFrame] so that
//! JavaScript code can access `Math`, `console`, `JSON`, `parseInt`, etc.
//!
//! # Example
//!
//! ```rust,ignore
//! use std::collections::HashMap;
//! use stator_core::builtins::install_globals::install_globals;
//! use stator_core::objects::value::JsValue;
//!
//! let mut globals = HashMap::new();
//! install_globals(&mut globals);
//! assert!(matches!(globals.get("Math"), Some(JsValue::PlainObject(_))));
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::builtins::error::{ErrorKind, JsError};
use crate::builtins::global::{
    GLOBAL_INFINITY, GLOBAL_NAN, global_decode_uri, global_decode_uri_component, global_encode_uri,
    global_encode_uri_component, global_is_finite, global_is_nan, global_parse_float,
    global_parse_int,
};
use crate::builtins::iterator::{
    iterator_drop, iterator_every, iterator_filter, iterator_find, iterator_flat_map,
    iterator_for_each, iterator_from, iterator_map, iterator_reduce, iterator_some, iterator_take,
    iterator_to_array,
};
use crate::builtins::map::{
    map_clear, map_delete, map_entries, map_from_iterable, map_get, map_has, map_keys, map_new,
    map_set, map_size, map_values,
};
use crate::builtins::math::{
    MATH_E, MATH_LN2, MATH_LN10, MATH_LOG2E, MATH_LOG10E, MATH_PI, MATH_SQRT1_2, MATH_SQRT2,
    math_abs, math_acos, math_acosh, math_asin, math_asinh, math_atan, math_atan2, math_atanh,
    math_cbrt, math_ceil, math_clz32, math_cos, math_cosh, math_exp, math_expm1, math_floor,
    math_fround, math_hypot, math_imul, math_log, math_log1p, math_log2, math_log10, math_max,
    math_min, math_pow, math_random, math_round, math_sign, math_sin, math_sinh, math_sqrt,
    math_tan, math_tanh, math_trunc,
};
use crate::builtins::regexp::regexp_construct;
use crate::builtins::set::{
    set_add, set_clear, set_delete, set_entries, set_from_iterable, set_has, set_keys, set_new,
    set_size, set_values,
};
use crate::builtins::string::{
    string_at, string_char_at, string_char_code_at, string_code_point_at, string_concat,
    string_ends_with, string_from_char_code, string_from_code_point, string_includes,
    string_index_of, string_is_well_formed, string_iter, string_last_index_of, string_match,
    string_match_all, string_normalize, string_pad_end, string_pad_start, string_raw,
    string_repeat, string_replace, string_replace_all, string_search, string_slice, string_split,
    string_starts_with, string_substring, string_to_lower_case, string_to_upper_case,
    string_to_well_formed, string_trim, string_trim_end, string_trim_start,
};
use crate::builtins::symbol::{
    SYMBOL_ASYNC_ITERATOR, SYMBOL_HAS_INSTANCE, SYMBOL_IS_CONCAT_SPREADABLE, SYMBOL_ITERATOR,
    SYMBOL_MATCH, SYMBOL_MATCH_ALL, SYMBOL_REPLACE, SYMBOL_SEARCH, SYMBOL_SPECIES, SYMBOL_SPLIT,
    SYMBOL_TO_PRIMITIVE, SYMBOL_TO_STRING_TAG, SYMBOL_UNSCOPABLES, symbol_create, symbol_for,
    symbol_key_for,
};
use crate::builtins::weak_map::{
    weak_map_delete, weak_map_get, weak_map_has, weak_map_new, weak_map_set,
};
use crate::builtins::weak_set::{weak_set_add, weak_set_delete, weak_set_has, weak_set_new};
use crate::error::{StatorError, StatorResult};
use crate::objects::value::{JsValue, NativeIterator};

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Wrap a Rust closure as a `JsValue::NativeFunction`.
fn native(f: impl Fn(Vec<JsValue>) -> StatorResult<JsValue> + 'static) -> JsValue {
    JsValue::NativeFunction(Rc::new(f))
}

/// Build a NativeFunction that constructs a `JsValue::Error` of the given `ErrorKind`.
fn make_error_constructor(kind: ErrorKind) -> JsValue {
    native(move |args| {
        let message = match args.first() {
            Some(JsValue::Undefined) | None => String::new(),
            Some(v) => v.to_js_string()?,
        };
        Ok(JsValue::Error(Rc::new(JsError::new(kind, message))))
    })
}

/// Register all ECMAScript error constructors in the global environment.
fn install_error_constructors(globals: &mut HashMap<String, JsValue>) {
    globals.insert("Error".into(), make_error_constructor(ErrorKind::Error));
    globals.insert(
        "TypeError".into(),
        make_error_constructor(ErrorKind::TypeError),
    );
    globals.insert(
        "RangeError".into(),
        make_error_constructor(ErrorKind::RangeError),
    );
    globals.insert(
        "ReferenceError".into(),
        make_error_constructor(ErrorKind::ReferenceError),
    );
    globals.insert(
        "SyntaxError".into(),
        make_error_constructor(ErrorKind::SyntaxError),
    );
    globals.insert(
        "URIError".into(),
        make_error_constructor(ErrorKind::URIError),
    );
    globals.insert(
        "EvalError".into(),
        make_error_constructor(ErrorKind::EvalError),
    );
}

/// Convert an `f64` to the most compact `JsValue` representation.
///
/// Returns `Smi` for values that are exact integers in the `i32` range,
/// `HeapNumber` for everything else (fractions, NaN, infinities, large ints).
fn num(n: f64) -> JsValue {
    if n.fract() == 0.0
        && !n.is_nan()
        && !n.is_infinite()
        && n >= f64::from(i32::MIN)
        && n <= f64::from(i32::MAX)
    {
        JsValue::Smi(n as i32)
    } else {
        JsValue::HeapNumber(n)
    }
}

/// Extract the argument at `idx` as `f64`, defaulting to `NaN` when absent.
fn arg_f64(args: &[JsValue], idx: usize) -> StatorResult<f64> {
    args.get(idx).unwrap_or(&JsValue::Undefined).to_number()
}

// ── Math ─────────────────────────────────────────────────────────────────────

/// Build the `Math` namespace object with all constants and methods.
fn make_math() -> JsValue {
    let mut props: HashMap<String, JsValue> = HashMap::new();

    // ── Constants ────────────────────────────────────────────────────────
    props.insert("E".into(), JsValue::HeapNumber(MATH_E));
    props.insert("LN2".into(), JsValue::HeapNumber(MATH_LN2));
    props.insert("LN10".into(), JsValue::HeapNumber(MATH_LN10));
    props.insert("LOG2E".into(), JsValue::HeapNumber(MATH_LOG2E));
    props.insert("LOG10E".into(), JsValue::HeapNumber(MATH_LOG10E));
    props.insert("PI".into(), JsValue::HeapNumber(MATH_PI));
    props.insert("SQRT1_2".into(), JsValue::HeapNumber(MATH_SQRT1_2));
    props.insert("SQRT2".into(), JsValue::HeapNumber(MATH_SQRT2));

    // ── Single-argument methods ──────────────────────────────────────────
    props.insert(
        "abs".into(),
        native(|args| Ok(num(math_abs(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "ceil".into(),
        native(|args| Ok(num(math_ceil(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "floor".into(),
        native(|args| Ok(num(math_floor(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "round".into(),
        native(|args| Ok(num(math_round(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "trunc".into(),
        native(|args| Ok(num(math_trunc(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "sign".into(),
        native(|args| Ok(num(math_sign(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "sqrt".into(),
        native(|args| Ok(num(math_sqrt(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "cbrt".into(),
        native(|args| Ok(num(math_cbrt(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "log".into(),
        native(|args| Ok(num(math_log(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "log2".into(),
        native(|args| Ok(num(math_log2(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "log10".into(),
        native(|args| Ok(num(math_log10(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "sin".into(),
        native(|args| Ok(num(math_sin(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "cos".into(),
        native(|args| Ok(num(math_cos(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "tan".into(),
        native(|args| Ok(num(math_tan(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "asin".into(),
        native(|args| Ok(num(math_asin(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "acos".into(),
        native(|args| Ok(num(math_acos(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "atan".into(),
        native(|args| Ok(num(math_atan(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "sinh".into(),
        native(|args| Ok(num(math_sinh(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "cosh".into(),
        native(|args| Ok(num(math_cosh(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "tanh".into(),
        native(|args| Ok(num(math_tanh(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "asinh".into(),
        native(|args| Ok(num(math_asinh(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "acosh".into(),
        native(|args| Ok(num(math_acosh(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "atanh".into(),
        native(|args| Ok(num(math_atanh(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "exp".into(),
        native(|args| Ok(num(math_exp(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "expm1".into(),
        native(|args| Ok(num(math_expm1(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "log1p".into(),
        native(|args| Ok(num(math_log1p(arg_f64(&args, 0)?)))),
    );
    props.insert(
        "fround".into(),
        native(|args| Ok(JsValue::HeapNumber(math_fround(arg_f64(&args, 0)?)))),
    );

    // ── Two-argument methods ─────────────────────────────────────────────
    props.insert(
        "atan2".into(),
        native(|args| {
            let y = arg_f64(&args, 0)?;
            let x = arg_f64(&args, 1)?;
            Ok(num(math_atan2(y, x)))
        }),
    );
    props.insert(
        "pow".into(),
        native(|args| {
            let base = arg_f64(&args, 0)?;
            let exp = arg_f64(&args, 1)?;
            Ok(num(math_pow(base, exp)))
        }),
    );
    props.insert(
        "imul".into(),
        native(|args| {
            let a = arg_f64(&args, 0)?;
            let b = arg_f64(&args, 1)?;
            Ok(JsValue::Smi(math_imul(a, b)))
        }),
    );

    // ── Variadic methods ─────────────────────────────────────────────────
    props.insert(
        "max".into(),
        native(|args| {
            let nums: StatorResult<Vec<f64>> = args.iter().map(|a| a.to_number()).collect();
            Ok(num(math_max(&nums?)))
        }),
    );
    props.insert(
        "min".into(),
        native(|args| {
            let nums: StatorResult<Vec<f64>> = args.iter().map(|a| a.to_number()).collect();
            Ok(num(math_min(&nums?)))
        }),
    );
    props.insert(
        "hypot".into(),
        native(|args| {
            let nums: StatorResult<Vec<f64>> = args.iter().map(|a| a.to_number()).collect();
            Ok(num(math_hypot(&nums?)))
        }),
    );

    // ── Zero-argument / special ──────────────────────────────────────────
    props.insert(
        "random".into(),
        native(|_args| Ok(JsValue::HeapNumber(math_random()))),
    );
    props.insert(
        "clz32".into(),
        native(|args| {
            let x = arg_f64(&args, 0)?;
            Ok(JsValue::Smi(math_clz32(x) as i32))
        }),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

// ── console ──────────────────────────────────────────────────────────────────

/// Build the `console` namespace object.
fn make_console() -> JsValue {
    let mut props: HashMap<String, JsValue> = HashMap::new();

    props.insert(
        "log".into(),
        native(|args: Vec<JsValue>| {
            let parts: StatorResult<Vec<String>> = args.iter().map(|a| a.to_js_string()).collect();
            println!("{}", parts?.join(" "));
            Ok(JsValue::Undefined)
        }),
    );
    props.insert(
        "warn".into(),
        native(|args: Vec<JsValue>| {
            let parts: StatorResult<Vec<String>> = args.iter().map(|a| a.to_js_string()).collect();
            eprintln!("{}", parts?.join(" "));
            Ok(JsValue::Undefined)
        }),
    );
    props.insert(
        "error".into(),
        native(|args: Vec<JsValue>| {
            let parts: StatorResult<Vec<String>> = args.iter().map(|a| a.to_js_string()).collect();
            eprintln!("{}", parts?.join(" "));
            Ok(JsValue::Undefined)
        }),
    );
    props.insert(
        "info".into(),
        native(|args: Vec<JsValue>| {
            let parts: StatorResult<Vec<String>> = args.iter().map(|a| a.to_js_string()).collect();
            println!("{}", parts?.join(" "));
            Ok(JsValue::Undefined)
        }),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

// ── JSON ─────────────────────────────────────────────────────────────────────

/// Convert a [`JsonValue`] to the corresponding [`JsValue`].
fn json_value_to_js_value(jv: &crate::builtins::json::JsonValue) -> JsValue {
    use crate::builtins::json::JsonValue;
    match jv {
        JsonValue::Null => JsValue::Null,
        JsonValue::Bool(b) => JsValue::Boolean(*b),
        JsonValue::Number(n) => num(*n),
        JsonValue::Str(s) => JsValue::String(s.clone()),
        JsonValue::Array(arr) => {
            let items: Vec<JsValue> = arr.borrow().iter().map(json_value_to_js_value).collect();
            JsValue::Array(Rc::new(items))
        }
        JsonValue::Object(entries) => {
            let mut map = HashMap::new();
            for (k, v) in entries.borrow().iter() {
                map.insert(k.clone(), json_value_to_js_value(v));
            }
            JsValue::PlainObject(Rc::new(RefCell::new(map)))
        }
    }
}

/// Build the `JSON` namespace object.
fn make_json() -> JsValue {
    use crate::builtins::json::{json_parse, json_stringify_js_value};

    let mut props: HashMap<String, JsValue> = HashMap::new();

    props.insert(
        "parse".into(),
        native(|args| {
            let text = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let json_val = json_parse(&text, None)?;
            Ok(json_value_to_js_value(&json_val))
        }),
    );
    props.insert(
        "stringify".into(),
        native(|args| {
            let val = args.first().unwrap_or(&JsValue::Undefined);
            match json_stringify_js_value(val, None, None)? {
                Some(s) => Ok(JsValue::String(s)),
                None => Ok(JsValue::Undefined),
            }
        }),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

// ── Number constructor ───────────────────────────────────────────────────────

/// Build the `Number` constructor/namespace object.
fn make_number() -> JsValue {
    let mut props: HashMap<String, JsValue> = HashMap::new();

    // Number.isNaN — does NOT coerce (unlike global isNaN)
    props.insert(
        "isNaN".into(),
        native(|args| {
            let val = args.first().unwrap_or(&JsValue::Undefined);
            let result = match val {
                JsValue::HeapNumber(n) => n.is_nan(),
                _ => false,
            };
            Ok(JsValue::Boolean(result))
        }),
    );
    // Number.isFinite — does NOT coerce
    props.insert(
        "isFinite".into(),
        native(|args| {
            let val = args.first().unwrap_or(&JsValue::Undefined);
            let result = match val {
                JsValue::Smi(_) => true,
                JsValue::HeapNumber(n) => n.is_finite(),
                _ => false,
            };
            Ok(JsValue::Boolean(result))
        }),
    );
    // Number.isInteger
    props.insert(
        "isInteger".into(),
        native(|args| {
            let val = args.first().unwrap_or(&JsValue::Undefined);
            let result = match val {
                JsValue::Smi(_) => true,
                JsValue::HeapNumber(n) => n.is_finite() && n.fract() == 0.0,
                _ => false,
            };
            Ok(JsValue::Boolean(result))
        }),
    );
    // Number.isSafeInteger
    props.insert(
        "isSafeInteger".into(),
        native(|args| {
            let val = args.first().unwrap_or(&JsValue::Undefined);
            let result = match val {
                JsValue::Smi(_) => true,
                JsValue::HeapNumber(n) => {
                    n.is_finite() && n.fract() == 0.0 && n.abs() <= 9_007_199_254_740_991.0
                }
                _ => false,
            };
            Ok(JsValue::Boolean(result))
        }),
    );
    // Number.parseInt
    props.insert(
        "parseInt".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let radix = if args.len() > 1 {
                args[1].to_number()?.floor() as u32
            } else {
                10
            };
            Ok(num(global_parse_int(&s, radix)))
        }),
    );
    // Number.parseFloat
    props.insert(
        "parseFloat".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(num(global_parse_float(&s)))
        }),
    );

    // Constants
    props.insert(
        "MAX_SAFE_INTEGER".into(),
        JsValue::HeapNumber(9_007_199_254_740_991.0),
    );
    props.insert(
        "MIN_SAFE_INTEGER".into(),
        JsValue::HeapNumber(-9_007_199_254_740_991.0),
    );
    props.insert("MAX_VALUE".into(), JsValue::HeapNumber(f64::MAX));
    props.insert("MIN_VALUE".into(), JsValue::HeapNumber(5e-324));
    props.insert("EPSILON".into(), JsValue::HeapNumber(f64::EPSILON));
    props.insert(
        "POSITIVE_INFINITY".into(),
        JsValue::HeapNumber(f64::INFINITY),
    );
    props.insert(
        "NEGATIVE_INFINITY".into(),
        JsValue::HeapNumber(f64::NEG_INFINITY),
    );
    props.insert("NaN".into(), JsValue::HeapNumber(f64::NAN));

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

// ── Object constructor ───────────────────────────────────────────────────────

/// Build the `Object` constructor/namespace object.
fn make_object() -> JsValue {
    let mut props: HashMap<String, JsValue> = HashMap::new();

    props.insert(
        "keys".into(),
        native(|args| {
            let val = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::PlainObject(map) = val {
                let keys: Vec<JsValue> = map
                    .borrow()
                    .keys()
                    .map(|k| JsValue::String(k.clone()))
                    .collect();
                Ok(JsValue::Array(Rc::new(keys)))
            } else {
                Ok(JsValue::Array(Rc::new(vec![])))
            }
        }),
    );
    props.insert(
        "values".into(),
        native(|args| {
            let val = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::PlainObject(map) = val {
                let values: Vec<JsValue> = map.borrow().values().cloned().collect();
                Ok(JsValue::Array(Rc::new(values)))
            } else {
                Ok(JsValue::Array(Rc::new(vec![])))
            }
        }),
    );
    props.insert(
        "entries".into(),
        native(|args| {
            let val = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::PlainObject(map) = val {
                let entries: Vec<JsValue> = map
                    .borrow()
                    .iter()
                    .map(|(k, v)| {
                        JsValue::Array(Rc::new(vec![JsValue::String(k.clone()), v.clone()]))
                    })
                    .collect();
                Ok(JsValue::Array(Rc::new(entries)))
            } else {
                Ok(JsValue::Array(Rc::new(vec![])))
            }
        }),
    );

    // ── Object.defineProperty(obj, prop, descriptor) ─────────────────────
    props.insert(
        "defineProperty".into(),
        native(|args| {
            let obj = args.first().unwrap_or(&JsValue::Undefined).clone();
            let prop = args.get(1).unwrap_or(&JsValue::Undefined);
            let descriptor = args.get(2).unwrap_or(&JsValue::Undefined);

            let key = prop.to_js_string()?;

            if let JsValue::PlainObject(map) = &obj {
                if let JsValue::PlainObject(desc_map) = descriptor {
                    let desc = desc_map.borrow();
                    // Extract value from descriptor, default to undefined
                    let value = desc.get("value").cloned().unwrap_or(JsValue::Undefined);
                    map.borrow_mut().insert(key, value);
                } else {
                    // If descriptor is not an object, just set undefined
                    map.borrow_mut().insert(key, JsValue::Undefined);
                }
                Ok(obj)
            } else {
                Err(crate::error::StatorError::TypeError(
                    "Object.defineProperty called on non-object".into(),
                ))
            }
        }),
    );

    // ── Object.getOwnPropertyDescriptor(obj, prop) ──────────────────────
    props.insert(
        "getOwnPropertyDescriptor".into(),
        native(|args| {
            let obj = args.first().unwrap_or(&JsValue::Undefined);
            let prop = args.get(1).unwrap_or(&JsValue::Undefined);

            let key = prop.to_js_string()?;

            if let JsValue::PlainObject(map) = obj {
                let borrowed = map.borrow();
                if let Some(value) = borrowed.get(&key) {
                    let mut desc: HashMap<String, JsValue> = HashMap::new();
                    desc.insert("value".into(), value.clone());
                    desc.insert("writable".into(), JsValue::Boolean(true));
                    desc.insert("enumerable".into(), JsValue::Boolean(true));
                    desc.insert("configurable".into(), JsValue::Boolean(true));
                    Ok(JsValue::PlainObject(Rc::new(RefCell::new(desc))))
                } else {
                    Ok(JsValue::Undefined)
                }
            } else {
                Ok(JsValue::Undefined)
            }
        }),
    );

    // ── Object.defineProperties(obj, props) ──────────────────────────────
    props.insert(
        "defineProperties".into(),
        native(|args| {
            let obj = args.first().unwrap_or(&JsValue::Undefined).clone();
            let descriptors = args.get(1).unwrap_or(&JsValue::Undefined);

            if let JsValue::PlainObject(map) = &obj {
                if let JsValue::PlainObject(desc_map) = descriptors {
                    let descs = desc_map.borrow();
                    for (key, desc_val) in descs.iter() {
                        if let JsValue::PlainObject(single_desc) = desc_val {
                            let sd = single_desc.borrow();
                            let value = sd.get("value").cloned().unwrap_or(JsValue::Undefined);
                            map.borrow_mut().insert(key.clone(), value);
                        }
                    }
                }
                Ok(obj)
            } else {
                Err(crate::error::StatorError::TypeError(
                    "Object.defineProperties called on non-object".into(),
                ))
            }
        }),
    );

    // ── Object.getOwnPropertyNames(obj) ──────────────────────────────────
    props.insert(
        "getOwnPropertyNames".into(),
        native(|args| {
            let obj = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::PlainObject(map) = obj {
                let keys: Vec<JsValue> = map
                    .borrow()
                    .keys()
                    .map(|k| JsValue::String(k.clone()))
                    .collect();
                Ok(JsValue::Array(Rc::new(keys)))
            } else {
                Ok(JsValue::Array(Rc::new(vec![])))
            }
        }),
    );

    // ── Object.assign(target, ...sources) ────────────────────────────────
    props.insert(
        "assign".into(),
        native(|args| {
            let target = args.first().unwrap_or(&JsValue::Undefined).clone();

            if let JsValue::PlainObject(target_map) = &target {
                for source in args.iter().skip(1) {
                    if let JsValue::PlainObject(src_map) = source {
                        let src = src_map.borrow();
                        for (k, v) in src.iter() {
                            target_map.borrow_mut().insert(k.clone(), v.clone());
                        }
                    }
                }
                Ok(target)
            } else {
                Err(crate::error::StatorError::TypeError(
                    "Object.assign called on non-object".into(),
                ))
            }
        }),
    );

    // ── Object.freeze(obj) ───────────────────────────────────────────────
    // For PlainObject, we return the object itself (properties are still
    // mutable in our simplified model; full descriptor storage would be
    // needed for a complete implementation).
    props.insert(
        "freeze".into(),
        native(|args| {
            let obj = args.first().unwrap_or(&JsValue::Undefined).clone();
            // Return the object itself per spec (even if we can't fully
            // enforce immutability on PlainObject without descriptor storage).
            Ok(obj)
        }),
    );

    // ── Object.seal(obj) ─────────────────────────────────────────────────
    props.insert(
        "seal".into(),
        native(|args| {
            let obj = args.first().unwrap_or(&JsValue::Undefined).clone();
            Ok(obj)
        }),
    );

    // ── Object.isFrozen(obj) ─────────────────────────────────────────────
    props.insert(
        "isFrozen".into(),
        native(|args| {
            let obj = args.first().unwrap_or(&JsValue::Undefined);
            match obj {
                // Non-objects are trivially frozen per spec
                JsValue::PlainObject(_) => Ok(JsValue::Boolean(false)),
                _ => Ok(JsValue::Boolean(true)),
            }
        }),
    );

    // ── Object.isSealed(obj) ─────────────────────────────────────────────
    props.insert(
        "isSealed".into(),
        native(|args| {
            let obj = args.first().unwrap_or(&JsValue::Undefined);
            match obj {
                JsValue::PlainObject(_) => Ok(JsValue::Boolean(false)),
                _ => Ok(JsValue::Boolean(true)),
            }
        }),
    );

    // ── Object.create(proto) ─────────────────────────────────────────────
    props.insert(
        "create".into(),
        native(|args| {
            let proto = args.first().unwrap_or(&JsValue::Undefined);
            match proto {
                JsValue::Null => Ok(JsValue::PlainObject(Rc::new(RefCell::new(HashMap::new())))),
                JsValue::PlainObject(map) => {
                    // Start with a copy of the prototype's properties
                    let new_map = map.borrow().clone();
                    Ok(JsValue::PlainObject(Rc::new(RefCell::new(new_map))))
                }
                _ => Ok(JsValue::PlainObject(Rc::new(RefCell::new(HashMap::new())))),
            }
        }),
    );

    // ── Object.is(x, y) ─────────────────────────────────────────────────
    props.insert(
        "is".into(),
        native(|args| {
            let x = args.first().unwrap_or(&JsValue::Undefined);
            let y = args.get(1).unwrap_or(&JsValue::Undefined);

            let result = match (x, y) {
                (JsValue::Undefined, JsValue::Undefined) => true,
                (JsValue::Null, JsValue::Null) => true,
                (JsValue::Boolean(a), JsValue::Boolean(b)) => a == b,
                (JsValue::String(a), JsValue::String(b)) => a == b,
                (JsValue::Symbol(a), JsValue::Symbol(b)) => a == b,
                (JsValue::Smi(a), JsValue::Smi(b)) => a == b,
                (JsValue::HeapNumber(a), JsValue::HeapNumber(b)) => {
                    if a.is_nan() && b.is_nan() {
                        true
                    } else {
                        a.to_bits() == b.to_bits()
                    }
                }
                (JsValue::Smi(a), JsValue::HeapNumber(b)) => {
                    let af = f64::from(*a);
                    if af == 0.0 && b.is_sign_negative() {
                        false
                    } else {
                        af == *b
                    }
                }
                (JsValue::HeapNumber(a), JsValue::Smi(b)) => {
                    let bf = f64::from(*b);
                    if bf == 0.0 && a.is_sign_negative() {
                        false
                    } else {
                        *a == bf
                    }
                }
                _ => false,
            };
            Ok(JsValue::Boolean(result))
        }),
    );

    // ── Object.fromEntries(iterable) ─────────────────────────────────────
    props.insert(
        "fromEntries".into(),
        native(|args| {
            let iterable = args.first().unwrap_or(&JsValue::Undefined);
            let mut result: HashMap<String, JsValue> = HashMap::new();

            if let JsValue::Array(arr) = iterable {
                for entry in arr.iter() {
                    if let JsValue::Array(pair) = entry
                        && pair.len() >= 2
                    {
                        let key = pair[0].to_js_string()?;
                        result.insert(key, pair[1].clone());
                    }
                }
            }
            Ok(JsValue::PlainObject(Rc::new(RefCell::new(result))))
        }),
    );

    // ── Object.hasOwn(obj, key) ──────────────────────────────────────────
    props.insert(
        "hasOwn".into(),
        native(|args| {
            let obj = args.first().unwrap_or(&JsValue::Undefined);
            let prop = args.get(1).unwrap_or(&JsValue::Undefined);
            let key = prop.to_js_string()?;

            match obj {
                JsValue::PlainObject(map) => Ok(JsValue::Boolean(map.borrow().contains_key(&key))),
                _ => Ok(JsValue::Boolean(false)),
            }
        }),
    );

    // ── Object.getPrototypeOf(obj) ───────────────────────────────────────
    props.insert(
        "getPrototypeOf".into(),
        native(|args| {
            let obj = args.first().unwrap_or(&JsValue::Undefined);
            match obj {
                // PlainObject has no prototype chain — return null.
                JsValue::PlainObject(_) => Ok(JsValue::Null),
                // Non-objects: per spec, coerce to object first, but for
                // primitives the prototype is not observable here.
                _ => Ok(JsValue::Null),
            }
        }),
    );

    // ── Object.setPrototypeOf(obj, proto) ────────────────────────────────
    props.insert(
        "setPrototypeOf".into(),
        native(|args| {
            let obj = args.first().unwrap_or(&JsValue::Undefined).clone();
            // Per spec, return the object itself.
            Ok(obj)
        }),
    );

    // ── Object.preventExtensions(obj) ────────────────────────────────────
    props.insert(
        "preventExtensions".into(),
        native(|args| {
            let obj = args.first().unwrap_or(&JsValue::Undefined).clone();
            // Per spec, return the object itself.
            Ok(obj)
        }),
    );

    // ── Object.isExtensible(obj) ─────────────────────────────────────────
    props.insert(
        "isExtensible".into(),
        native(|args| {
            let obj = args.first().unwrap_or(&JsValue::Undefined);
            match obj {
                // Non-objects are not extensible per spec.
                JsValue::PlainObject(_) => Ok(JsValue::Boolean(true)),
                _ => Ok(JsValue::Boolean(false)),
            }
        }),
    );

    // ── Object.getOwnPropertyDescriptors(obj) ────────────────────────────
    props.insert(
        "getOwnPropertyDescriptors".into(),
        native(|args| {
            let obj = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::PlainObject(map) = obj {
                let mut result: HashMap<String, JsValue> = HashMap::new();
                for (key, value) in map.borrow().iter() {
                    let mut desc: HashMap<String, JsValue> = HashMap::new();
                    desc.insert("value".into(), value.clone());
                    desc.insert("writable".into(), JsValue::Boolean(true));
                    desc.insert("enumerable".into(), JsValue::Boolean(true));
                    desc.insert("configurable".into(), JsValue::Boolean(true));
                    result.insert(
                        key.clone(),
                        JsValue::PlainObject(Rc::new(RefCell::new(desc))),
                    );
                }
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(result))))
            } else {
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(HashMap::new()))))
            }
        }),
    );

    // ── Object.getOwnPropertySymbols(obj) ────────────────────────────────
    props.insert(
        "getOwnPropertySymbols".into(),
        native(|_args| {
            // PlainObject has no symbol-keyed properties.
            Ok(JsValue::Array(Rc::new(vec![])))
        }),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

// ── Array constructor ────────────────────────────────────────────────────────

/// Build the `Array` constructor/namespace object.
fn make_array() -> JsValue {
    let mut props: HashMap<String, JsValue> = HashMap::new();

    props.insert(
        "isArray".into(),
        native(|args| {
            let val = args.first().unwrap_or(&JsValue::Undefined);
            Ok(JsValue::Boolean(matches!(val, JsValue::Array(_))))
        }),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}
// ── Symbol constructor ────────────────────────────────────────────────────────────

/// Build the `Symbol` constructor/namespace object.
///
/// The returned `PlainObject` acts both as a callable (`Symbol()` /
/// `Symbol("desc")`) — handled by the interpreter when it sees a
/// `NativeFunction` stored under the `"Symbol"` key — and as a namespace
/// carrying static methods (`for`, `keyFor`) and well-known symbol
/// constants (`iterator`, `toPrimitive`, etc.).
///
/// Because JavaScript’s `Symbol` is *not* a constructor (i.e. `new Symbol()`
/// is a `TypeError`), the top-level value is a `NativeFunction` that
/// creates symbols, and the static properties are patched onto the
/// surrounding `PlainObject` wrapper.
fn make_symbol() -> JsValue {
    let mut props: HashMap<String, JsValue> = HashMap::new();

    // ── Static methods ────────────────────────────────────────────────────

    // Symbol.for(key) — global symbol registry
    props.insert(
        "for".into(),
        native(|args| {
            let key = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::Symbol(symbol_for(&key)))
        }),
    );

    // Symbol.keyFor(sym) — reverse lookup in the global registry
    props.insert(
        "keyFor".into(),
        native(|args| {
            let sym = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Symbol(id) = sym {
                match symbol_key_for(*id) {
                    Some(key) => Ok(JsValue::String(key)),
                    None => Ok(JsValue::Undefined),
                }
            } else {
                Err(crate::error::StatorError::TypeError(
                    "Symbol.keyFor requires a symbol argument".into(),
                ))
            }
        }),
    );

    // ── Well-known symbols ───────────────────────────────────────────────

    props.insert("iterator".into(), JsValue::Symbol(SYMBOL_ITERATOR));
    props.insert("toPrimitive".into(), JsValue::Symbol(SYMBOL_TO_PRIMITIVE));
    props.insert("hasInstance".into(), JsValue::Symbol(SYMBOL_HAS_INSTANCE));
    props.insert("toStringTag".into(), JsValue::Symbol(SYMBOL_TO_STRING_TAG));
    props.insert(
        "isConcatSpreadable".into(),
        JsValue::Symbol(SYMBOL_IS_CONCAT_SPREADABLE),
    );
    props.insert("species".into(), JsValue::Symbol(SYMBOL_SPECIES));
    props.insert("match".into(), JsValue::Symbol(SYMBOL_MATCH));
    props.insert("replace".into(), JsValue::Symbol(SYMBOL_REPLACE));
    props.insert("search".into(), JsValue::Symbol(SYMBOL_SEARCH));
    props.insert("split".into(), JsValue::Symbol(SYMBOL_SPLIT));
    props.insert("unscopables".into(), JsValue::Symbol(SYMBOL_UNSCOPABLES));
    props.insert(
        "asyncIterator".into(),
        JsValue::Symbol(SYMBOL_ASYNC_ITERATOR),
    );
    props.insert("matchAll".into(), JsValue::Symbol(SYMBOL_MATCH_ALL));

    // ── The callable Symbol(desc?) ────────────────────────────────────────
    //
    // We store the callable itself under the reserved key "__call__" so the
    // interpreter can invoke `Symbol("desc")` while still allowing property
    // access on `Symbol.iterator` etc.
    props.insert(
        "__call__".into(),
        native(|args| {
            let desc = match args.first() {
                Some(JsValue::Undefined) | None => None,
                Some(v) => Some(v.to_js_string()?),
            };
            Ok(JsValue::Symbol(symbol_create(desc)))
        }),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

// ── Iterator (ES2025 §27.1.4) ────────────────────────────────────────────────

/// Build the `Iterator` constructor/namespace object with prototype helpers.
fn make_iterator() -> JsValue {
    let mut props: HashMap<String, JsValue> = HashMap::new();

    // ── Static method: Iterator.from ──────────────────────────────────────
    props.insert(
        "from".into(),
        native(|args| {
            let iterable = args.first().unwrap_or(&JsValue::Undefined);
            iterator_from(iterable)
        }),
    );

    // ── Prototype (instance) methods ─────────────────────────────────────
    //
    // In a full engine these would live on Iterator.prototype.  Here we
    // attach them as own properties so that `Iterator.prototype.map(…)` is
    // accessible as `Iterator.map(iter, mapper)` for direct testing and for
    // the bytecode to call via property lookup.
    let mut proto: HashMap<String, JsValue> = HashMap::new();

    proto.insert(
        "map".into(),
        native(|args| {
            let iter = args.first().unwrap_or(&JsValue::Undefined);
            let mapper = args.get(1).unwrap_or(&JsValue::Undefined);
            iterator_map(iter, mapper)
        }),
    );
    proto.insert(
        "filter".into(),
        native(|args| {
            let iter = args.first().unwrap_or(&JsValue::Undefined);
            let predicate = args.get(1).unwrap_or(&JsValue::Undefined);
            iterator_filter(iter, predicate)
        }),
    );
    proto.insert(
        "take".into(),
        native(|args| {
            let iter = args.first().unwrap_or(&JsValue::Undefined);
            let limit = args
                .get(1)
                .unwrap_or(&JsValue::Undefined)
                .to_number()
                .unwrap_or(0.0);
            iterator_take(iter, limit.max(0.0) as usize)
        }),
    );
    proto.insert(
        "drop".into(),
        native(|args| {
            let iter = args.first().unwrap_or(&JsValue::Undefined);
            let count = args
                .get(1)
                .unwrap_or(&JsValue::Undefined)
                .to_number()
                .unwrap_or(0.0);
            iterator_drop(iter, count.max(0.0) as usize)
        }),
    );
    proto.insert(
        "flatMap".into(),
        native(|args| {
            let iter = args.first().unwrap_or(&JsValue::Undefined);
            let mapper = args.get(1).unwrap_or(&JsValue::Undefined);
            iterator_flat_map(iter, mapper)
        }),
    );
    proto.insert(
        "reduce".into(),
        native(|args| {
            let iter = args.first().unwrap_or(&JsValue::Undefined);
            let reducer = args.get(1).unwrap_or(&JsValue::Undefined);
            let initial = args.get(2).cloned();
            iterator_reduce(iter, reducer, initial)
        }),
    );
    proto.insert(
        "toArray".into(),
        native(|args| {
            let iter = args.first().unwrap_or(&JsValue::Undefined);
            iterator_to_array(iter)
        }),
    );
    proto.insert(
        "forEach".into(),
        native(|args| {
            let iter = args.first().unwrap_or(&JsValue::Undefined);
            let callback = args.get(1).unwrap_or(&JsValue::Undefined);
            iterator_for_each(iter, callback)
        }),
    );
    proto.insert(
        "some".into(),
        native(|args| {
            let iter = args.first().unwrap_or(&JsValue::Undefined);
            let predicate = args.get(1).unwrap_or(&JsValue::Undefined);
            iterator_some(iter, predicate)
        }),
    );
    proto.insert(
        "every".into(),
        native(|args| {
            let iter = args.first().unwrap_or(&JsValue::Undefined);
            let predicate = args.get(1).unwrap_or(&JsValue::Undefined);
            iterator_every(iter, predicate)
        }),
    );
    proto.insert(
        "find".into(),
        native(|args| {
            let iter = args.first().unwrap_or(&JsValue::Undefined);
            let predicate = args.get(1).unwrap_or(&JsValue::Undefined);
            iterator_find(iter, predicate)
        }),
    );

    props.insert(
        "prototype".into(),
        JsValue::PlainObject(Rc::new(RefCell::new(proto))),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

// ── Map constructor (ES2025 §24.1) ───────────────────────────────────────────

/// Build the `Map` constructor/namespace object.
///
/// The returned `PlainObject` provides a `__call__` constructor that
/// optionally accepts an iterable of `[key, value]` pairs, plus prototype
/// methods (`get`, `set`, `has`, `delete`, `clear`, `forEach`, `keys`,
/// `values`, `entries`, `size`).
fn make_map_builtin() -> JsValue {
    let mut props: HashMap<String, JsValue> = HashMap::new();

    // ── Constructor: new Map() / new Map(iterable) ───────────────────────
    props.insert(
        "__call__".into(),
        native(|args| {
            let m = if let Some(JsValue::Array(arr)) = args.first() {
                let mut pairs = Vec::new();
                for item in arr.iter() {
                    if let JsValue::Array(pair) = item {
                        let k = pair.first().cloned().unwrap_or(JsValue::Undefined);
                        let v = pair.get(1).cloned().unwrap_or(JsValue::Undefined);
                        pairs.push((k, v));
                    }
                }
                map_from_iterable(pairs)
            } else {
                map_new()
            };
            // Store the JsMap in a RefCell so prototype methods can mutate it.
            let inner = Rc::new(RefCell::new(m));
            let mut obj: HashMap<String, JsValue> = HashMap::new();
            // size getter
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "size".into(),
                    JsValue::Smi(map_size(&inner.borrow()) as i32),
                );
            }
            // get(key)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "get".into(),
                    native(move |a| {
                        let key = a.first().unwrap_or(&JsValue::Undefined);
                        Ok(map_get(&inner.borrow(), key))
                    }),
                );
            }
            // set(key, value)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "set".into(),
                    native(move |a| {
                        let key = a.first().cloned().unwrap_or(JsValue::Undefined);
                        let val = a.get(1).cloned().unwrap_or(JsValue::Undefined);
                        map_set(&mut inner.borrow_mut(), key, val);
                        Ok(JsValue::Undefined)
                    }),
                );
            }
            // has(key)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "has".into(),
                    native(move |a| {
                        let key = a.first().unwrap_or(&JsValue::Undefined);
                        Ok(JsValue::Boolean(map_has(&inner.borrow(), key)))
                    }),
                );
            }
            // delete(key)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "delete".into(),
                    native(move |a| {
                        let key = a.first().unwrap_or(&JsValue::Undefined);
                        Ok(JsValue::Boolean(map_delete(&mut inner.borrow_mut(), key)))
                    }),
                );
            }
            // clear()
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "clear".into(),
                    native(move |_| {
                        map_clear(&mut inner.borrow_mut());
                        Ok(JsValue::Undefined)
                    }),
                );
            }
            // forEach(callback)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "forEach".into(),
                    native(move |a| {
                        let cb = a.first().cloned().unwrap_or(JsValue::Undefined);
                        let snapshot = map_entries(&inner.borrow());
                        for (k, v) in snapshot {
                            if let JsValue::NativeFunction(f) = &cb {
                                f(vec![v, k])?;
                            }
                        }
                        Ok(JsValue::Undefined)
                    }),
                );
            }
            // keys()
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "keys".into(),
                    native(move |_| {
                        let keys = map_keys(&inner.borrow());
                        Ok(JsValue::Iterator(NativeIterator::from_items(keys)))
                    }),
                );
            }
            // values()
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "values".into(),
                    native(move |_| {
                        let vals = map_values(&inner.borrow());
                        Ok(JsValue::Iterator(NativeIterator::from_items(vals)))
                    }),
                );
            }
            // entries()
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "entries".into(),
                    native(move |_| {
                        let entries = map_entries(&inner.borrow());
                        let items: Vec<JsValue> = entries
                            .into_iter()
                            .map(|(k, v)| JsValue::Array(Rc::new(vec![k, v])))
                            .collect();
                        Ok(JsValue::Iterator(NativeIterator::from_items(items)))
                    }),
                );
            }
            Ok(JsValue::PlainObject(Rc::new(RefCell::new(obj))))
        }),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

// ── Set constructor (ES2025 §24.2) ───────────────────────────────────────────

/// Build the `Set` constructor/namespace object.
///
/// The returned `PlainObject` provides a `__call__` constructor that
/// optionally accepts an iterable of values, plus prototype methods
/// (`add`, `has`, `delete`, `clear`, `forEach`, `keys`, `values`,
/// `entries`, `size`).
fn make_set_builtin() -> JsValue {
    let mut props: HashMap<String, JsValue> = HashMap::new();

    // ── Constructor: new Set() / new Set(iterable) ───────────────────────
    props.insert(
        "__call__".into(),
        native(|args| {
            let s = if let Some(JsValue::Array(arr)) = args.first() {
                set_from_iterable(arr.as_ref().clone())
            } else {
                set_new()
            };
            let inner = Rc::new(RefCell::new(s));
            let mut obj: HashMap<String, JsValue> = HashMap::new();
            // size getter
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "size".into(),
                    JsValue::Smi(set_size(&inner.borrow()) as i32),
                );
            }
            // add(value)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "add".into(),
                    native(move |a| {
                        let val = a.first().cloned().unwrap_or(JsValue::Undefined);
                        set_add(&mut inner.borrow_mut(), val);
                        Ok(JsValue::Undefined)
                    }),
                );
            }
            // has(value)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "has".into(),
                    native(move |a| {
                        let val = a.first().unwrap_or(&JsValue::Undefined);
                        Ok(JsValue::Boolean(set_has(&inner.borrow(), val)))
                    }),
                );
            }
            // delete(value)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "delete".into(),
                    native(move |a| {
                        let val = a.first().unwrap_or(&JsValue::Undefined);
                        Ok(JsValue::Boolean(set_delete(&mut inner.borrow_mut(), val)))
                    }),
                );
            }
            // clear()
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "clear".into(),
                    native(move |_| {
                        set_clear(&mut inner.borrow_mut());
                        Ok(JsValue::Undefined)
                    }),
                );
            }
            // forEach(callback)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "forEach".into(),
                    native(move |a| {
                        let cb = a.first().cloned().unwrap_or(JsValue::Undefined);
                        let snapshot = set_values(&inner.borrow());
                        for v in snapshot {
                            if let JsValue::NativeFunction(f) = &cb {
                                f(vec![v.clone(), v])?;
                            }
                        }
                        Ok(JsValue::Undefined)
                    }),
                );
            }
            // keys() — alias for values()
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "keys".into(),
                    native(move |_| {
                        let vals = set_keys(&inner.borrow());
                        Ok(JsValue::Iterator(NativeIterator::from_items(vals)))
                    }),
                );
            }
            // values()
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "values".into(),
                    native(move |_| {
                        let vals = set_values(&inner.borrow());
                        Ok(JsValue::Iterator(NativeIterator::from_items(vals)))
                    }),
                );
            }
            // entries()
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "entries".into(),
                    native(move |_| {
                        let entries = set_entries(&inner.borrow());
                        let items: Vec<JsValue> = entries
                            .into_iter()
                            .map(|(k, v)| JsValue::Array(Rc::new(vec![k, v])))
                            .collect();
                        Ok(JsValue::Iterator(NativeIterator::from_items(items)))
                    }),
                );
            }
            Ok(JsValue::PlainObject(Rc::new(RefCell::new(obj))))
        }),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

// ── WeakMap constructor (ES2025 §24.3) ───────────────────────────────────────

/// Build the `WeakMap` constructor/namespace object.
///
/// The returned `PlainObject` provides a `__call__` constructor that creates
/// a new `WeakMap` instance with prototype methods (`get`, `set`, `has`,
/// `delete`).  Keys must be `Object` pointers; non-object keys cause a
/// `TypeError`.
fn make_weak_map_builtin() -> JsValue {
    let mut props: HashMap<String, JsValue> = HashMap::new();

    props.insert(
        "__call__".into(),
        native(|_args| {
            let inner = Rc::new(RefCell::new(weak_map_new()));
            let mut obj: HashMap<String, JsValue> = HashMap::new();

            // get(key)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "get".into(),
                    native(move |a| {
                        let key = a.first().unwrap_or(&JsValue::Undefined);
                        if let JsValue::Object(ptr) = key {
                            Ok(weak_map_get(&inner.borrow(), *ptr))
                        } else {
                            Ok(JsValue::Undefined)
                        }
                    }),
                );
            }
            // set(key, value)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "set".into(),
                    native(move |a| {
                        let key = a.first().unwrap_or(&JsValue::Undefined);
                        let val = a.get(1).cloned().unwrap_or(JsValue::Undefined);
                        if let JsValue::Object(ptr) = key {
                            weak_map_set(&mut inner.borrow_mut(), *ptr, val)?;
                            Ok(JsValue::Undefined)
                        } else {
                            Err(StatorError::TypeError(
                                "Invalid value used as weak map key".into(),
                            ))
                        }
                    }),
                );
            }
            // has(key)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "has".into(),
                    native(move |a| {
                        let key = a.first().unwrap_or(&JsValue::Undefined);
                        if let JsValue::Object(ptr) = key {
                            Ok(JsValue::Boolean(weak_map_has(&inner.borrow(), *ptr)))
                        } else {
                            Ok(JsValue::Boolean(false))
                        }
                    }),
                );
            }
            // delete(key)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "delete".into(),
                    native(move |a| {
                        let key = a.first().unwrap_or(&JsValue::Undefined);
                        if let JsValue::Object(ptr) = key {
                            Ok(JsValue::Boolean(weak_map_delete(
                                &mut inner.borrow_mut(),
                                *ptr,
                            )))
                        } else {
                            Ok(JsValue::Boolean(false))
                        }
                    }),
                );
            }

            Ok(JsValue::PlainObject(Rc::new(RefCell::new(obj))))
        }),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

// ── WeakSet constructor (ES2025 §24.4) ───────────────────────────────────────

/// Build the `WeakSet` constructor/namespace object.
///
/// The returned `PlainObject` provides a `__call__` constructor that creates
/// a new `WeakSet` instance with prototype methods (`add`, `has`, `delete`).
/// Values must be `Object` pointers; non-object values cause a `TypeError`.
fn make_weak_set_builtin() -> JsValue {
    let mut props: HashMap<String, JsValue> = HashMap::new();

    props.insert(
        "__call__".into(),
        native(|_args| {
            let inner = Rc::new(RefCell::new(weak_set_new()));
            let mut obj: HashMap<String, JsValue> = HashMap::new();

            // add(value)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "add".into(),
                    native(move |a| {
                        let val = a.first().unwrap_or(&JsValue::Undefined);
                        if let JsValue::Object(ptr) = val {
                            weak_set_add(&mut inner.borrow_mut(), *ptr)?;
                            Ok(JsValue::Undefined)
                        } else {
                            Err(StatorError::TypeError(
                                "Invalid value used in weak set".into(),
                            ))
                        }
                    }),
                );
            }
            // has(value)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "has".into(),
                    native(move |a| {
                        let val = a.first().unwrap_or(&JsValue::Undefined);
                        if let JsValue::Object(ptr) = val {
                            Ok(JsValue::Boolean(weak_set_has(&inner.borrow(), *ptr)))
                        } else {
                            Ok(JsValue::Boolean(false))
                        }
                    }),
                );
            }
            // delete(value)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "delete".into(),
                    native(move |a| {
                        let val = a.first().unwrap_or(&JsValue::Undefined);
                        if let JsValue::Object(ptr) = val {
                            Ok(JsValue::Boolean(weak_set_delete(
                                &mut inner.borrow_mut(),
                                *ptr,
                            )))
                        } else {
                            Ok(JsValue::Boolean(false))
                        }
                    }),
                );
            }

            Ok(JsValue::PlainObject(Rc::new(RefCell::new(obj))))
        }),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

// ── String constructor ───────────────────────────────────────────────────────

/// Build the `String` constructor/namespace object with static and prototype
/// methods.
///
/// The returned `PlainObject` carries:
/// - `__call__` — the callable `String(value)` conversion.
/// - Static methods: `fromCharCode`, `fromCodePoint`, `raw`.
/// - `prototype` — an object with all `String.prototype.*` methods.
fn make_string() -> JsValue {
    let mut props: HashMap<String, JsValue> = HashMap::new();

    // ── Callable: String(value) ─────────────────────────────────────────
    props.insert(
        "__call__".into(),
        native(|args| {
            let val = args.first().unwrap_or(&JsValue::Undefined);
            Ok(JsValue::String(val.to_js_string()?))
        }),
    );

    // ── Static methods ──────────────────────────────────────────────────

    // String.fromCharCode(...codes)
    props.insert(
        "fromCharCode".into(),
        native(|args| {
            let codes: Vec<u32> = args
                .iter()
                .map(|a| a.to_number().unwrap_or(0.0) as u32)
                .collect();
            Ok(JsValue::String(string_from_char_code(&codes)))
        }),
    );

    // String.fromCodePoint(...codePoints)
    props.insert(
        "fromCodePoint".into(),
        native(|args| {
            let codes: Vec<u32> = args
                .iter()
                .map(|a| a.to_number().unwrap_or(0.0) as u32)
                .collect();
            Ok(JsValue::String(string_from_code_point(&codes)?))
        }),
    );

    // String.raw(strings, ...substitutions)
    props.insert(
        "raw".into(),
        native(|args| {
            // First arg is the template object with a `raw` property.
            let template = args.first().unwrap_or(&JsValue::Undefined);
            let raw_array = match template {
                JsValue::PlainObject(map) => {
                    let borrowed = map.borrow();
                    borrowed.get("raw").cloned()
                }
                _ => None,
            };
            let raw_strings: Vec<String> = match &raw_array {
                Some(JsValue::Array(arr)) => arr
                    .iter()
                    .map(|v| v.to_js_string().unwrap_or_default())
                    .collect(),
                _ => Vec::new(),
            };
            let subs: Vec<String> = args
                .iter()
                .skip(1)
                .map(|v| v.to_js_string().unwrap_or_default())
                .collect();
            let raw_refs: Vec<&str> = raw_strings.iter().map(String::as_str).collect();
            let sub_refs: Vec<&str> = subs.iter().map(String::as_str).collect();
            Ok(JsValue::String(string_raw(&raw_refs, &sub_refs)))
        }),
    );

    // ── Prototype methods ───────────────────────────────────────────────
    let mut proto: HashMap<String, JsValue> = HashMap::new();

    // charAt(pos)
    proto.insert(
        "charAt".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let pos = args
                .get(1)
                .unwrap_or(&JsValue::Undefined)
                .to_number()
                .unwrap_or(0.0) as i64;
            Ok(JsValue::String(string_char_at(&s, pos)))
        }),
    );

    // charCodeAt(pos)
    proto.insert(
        "charCodeAt".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let pos = args
                .get(1)
                .unwrap_or(&JsValue::Undefined)
                .to_number()
                .unwrap_or(0.0) as i64;
            Ok(num(string_char_code_at(&s, pos)))
        }),
    );

    // codePointAt(pos)
    proto.insert(
        "codePointAt".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let pos = args
                .get(1)
                .unwrap_or(&JsValue::Undefined)
                .to_number()
                .unwrap_or(0.0) as i64;
            match string_code_point_at(&s, pos) {
                Some(cp) => Ok(num(cp as f64)),
                None => Ok(JsValue::Undefined),
            }
        }),
    );

    // concat(...strings)
    proto.insert(
        "concat".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let parts: StatorResult<Vec<String>> =
                args.iter().skip(1).map(|a| a.to_js_string()).collect();
            let parts = parts?;
            let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
            Ok(JsValue::String(string_concat(&s, &refs)))
        }),
    );

    // slice(start, end?)
    proto.insert(
        "slice".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let start = args
                .get(1)
                .unwrap_or(&JsValue::Undefined)
                .to_number()
                .unwrap_or(0.0) as i64;
            let end = match args.get(2) {
                Some(JsValue::Undefined) | None => None,
                Some(v) => Some(v.to_number().unwrap_or(0.0) as i64),
            };
            Ok(JsValue::String(string_slice(&s, start, end)))
        }),
    );

    // substring(start, end?)
    proto.insert(
        "substring".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let start = args
                .get(1)
                .unwrap_or(&JsValue::Undefined)
                .to_number()
                .unwrap_or(0.0) as i64;
            let end = match args.get(2) {
                Some(JsValue::Undefined) | None => None,
                Some(v) => Some(v.to_number().unwrap_or(0.0) as i64),
            };
            Ok(JsValue::String(string_substring(&s, start, end)))
        }),
    );

    // indexOf(searchString, position?)
    proto.insert(
        "indexOf".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let search = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            let pos = match args.get(2) {
                Some(JsValue::Undefined) | None => None,
                Some(v) => Some(v.to_number().unwrap_or(0.0) as i64),
            };
            Ok(num(string_index_of(&s, &search, pos) as f64))
        }),
    );

    // lastIndexOf(searchString, position?)
    proto.insert(
        "lastIndexOf".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let search = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            let pos = match args.get(2) {
                Some(JsValue::Undefined) | None => None,
                Some(v) => Some(v.to_number().unwrap_or(0.0) as i64),
            };
            Ok(num(string_last_index_of(&s, &search, pos) as f64))
        }),
    );

    // includes(searchString, position?)
    proto.insert(
        "includes".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let search = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            let pos = match args.get(2) {
                Some(JsValue::Undefined) | None => None,
                Some(v) => Some(v.to_number().unwrap_or(0.0) as i64),
            };
            Ok(JsValue::Boolean(string_includes(&s, &search, pos)))
        }),
    );

    // startsWith(searchString, position?)
    proto.insert(
        "startsWith".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let search = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            let pos = match args.get(2) {
                Some(JsValue::Undefined) | None => None,
                Some(v) => Some(v.to_number().unwrap_or(0.0) as i64),
            };
            Ok(JsValue::Boolean(string_starts_with(&s, &search, pos)))
        }),
    );

    // endsWith(searchString, endPosition?)
    proto.insert(
        "endsWith".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let search = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            let end = match args.get(2) {
                Some(JsValue::Undefined) | None => None,
                Some(v) => Some(v.to_number().unwrap_or(0.0) as i64),
            };
            Ok(JsValue::Boolean(string_ends_with(&s, &search, end)))
        }),
    );

    // toUpperCase()
    proto.insert(
        "toUpperCase".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_to_upper_case(&s)))
        }),
    );

    // toLowerCase()
    proto.insert(
        "toLowerCase".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_to_lower_case(&s)))
        }),
    );

    // trim()
    proto.insert(
        "trim".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_trim(&s)))
        }),
    );

    // trimStart()
    proto.insert(
        "trimStart".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_trim_start(&s)))
        }),
    );

    // trimEnd()
    proto.insert(
        "trimEnd".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_trim_end(&s)))
        }),
    );

    // split(separator?, limit?)
    proto.insert(
        "split".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let sep = match args.get(1) {
                Some(JsValue::Undefined) | None => None,
                Some(v) => Some(v.to_js_string()?),
            };
            let limit = match args.get(2) {
                Some(JsValue::Undefined) | None => None,
                Some(v) => Some(v.to_number().unwrap_or(0.0) as u32),
            };
            let parts = string_split(&s, sep.as_deref(), limit);
            let arr: Vec<JsValue> = parts.into_iter().map(JsValue::String).collect();
            Ok(JsValue::Array(Rc::new(arr)))
        }),
    );

    // replace(searchValue, replaceValue)
    proto.insert(
        "replace".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let search = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            let replacement = args.get(2).unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_replace(&s, &search, &replacement)))
        }),
    );

    // replaceAll(searchValue, replaceValue)
    proto.insert(
        "replaceAll".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let search = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            let replacement = args.get(2).unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_replace_all(
                &s,
                &search,
                &replacement,
            )))
        }),
    );

    // match(regexp)
    proto.insert(
        "match".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let pattern = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            match string_match(&s, &pattern) {
                Some(groups) => {
                    let arr: Vec<JsValue> = groups.into_iter().map(JsValue::String).collect();
                    Ok(JsValue::Array(Rc::new(arr)))
                }
                None => Ok(JsValue::Null),
            }
        }),
    );

    // matchAll(regexp)
    proto.insert(
        "matchAll".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let pattern = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            match string_match_all(&s, &pattern) {
                Some(matches) => {
                    let arr: Vec<JsValue> = matches.into_iter().map(JsValue::String).collect();
                    Ok(JsValue::Array(Rc::new(arr)))
                }
                None => Ok(JsValue::Array(Rc::new(Vec::new()))),
            }
        }),
    );

    // repeat(count)
    proto.insert(
        "repeat".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let n = args
                .get(1)
                .unwrap_or(&JsValue::Undefined)
                .to_number()
                .unwrap_or(0.0);
            if n.is_infinite() || n < 0.0 {
                return Err(crate::error::StatorError::RangeError(
                    "Invalid count value".to_string(),
                ));
            }
            Ok(JsValue::String(string_repeat(&s, n as i64)?))
        }),
    );

    // padStart(targetLength, padString?)
    proto.insert(
        "padStart".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let target_len = args
                .get(1)
                .unwrap_or(&JsValue::Undefined)
                .to_number()
                .unwrap_or(0.0) as usize;
            let pad = match args.get(2) {
                Some(JsValue::Undefined) | None => None,
                Some(v) => Some(v.to_js_string()?),
            };
            Ok(JsValue::String(string_pad_start(
                &s,
                target_len,
                pad.as_deref(),
            )))
        }),
    );

    // padEnd(targetLength, padString?)
    proto.insert(
        "padEnd".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let target_len = args
                .get(1)
                .unwrap_or(&JsValue::Undefined)
                .to_number()
                .unwrap_or(0.0) as usize;
            let pad = match args.get(2) {
                Some(JsValue::Undefined) | None => None,
                Some(v) => Some(v.to_js_string()?),
            };
            Ok(JsValue::String(string_pad_end(
                &s,
                target_len,
                pad.as_deref(),
            )))
        }),
    );

    // at(index)
    proto.insert(
        "at".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let idx = args
                .get(1)
                .unwrap_or(&JsValue::Undefined)
                .to_number()
                .unwrap_or(0.0) as i64;
            match string_at(&s, idx) {
                Some(ch) => Ok(JsValue::String(ch)),
                None => Ok(JsValue::Undefined),
            }
        }),
    );

    // normalize(form?)
    proto.insert(
        "normalize".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let form = match args.get(1) {
                Some(JsValue::Undefined) | None => None,
                Some(v) => Some(v.to_js_string()?),
            };
            Ok(JsValue::String(string_normalize(&s, form.as_deref())?))
        }),
    );

    // search(regexp)
    proto.insert(
        "search".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let pattern = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(num(string_search(&s, &pattern) as f64))
        }),
    );

    // isWellFormed()
    proto.insert(
        "isWellFormed".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::Boolean(string_is_well_formed(&s)))
        }),
    );

    // toWellFormed()
    proto.insert(
        "toWellFormed".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_to_well_formed(&s)))
        }),
    );

    // [Symbol.iterator]() — returns the code-point iteration as an array.
    proto.insert(
        "@@iterator".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let chars: Vec<JsValue> = string_iter(&s).into_iter().map(JsValue::String).collect();
            Ok(JsValue::Array(Rc::new(chars)))
        }),
    );

    props.insert(
        "prototype".into(),
        JsValue::PlainObject(Rc::new(RefCell::new(proto))),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

// ── Promise ─────────────────────────────────────────────────────────────────

/// Build the `Promise` constructor/namespace object.
///
/// Creates an internal [`MicrotaskQueue`] shared by all promise operations
/// created through this global.  The constructor (`__call__`) corresponds to
/// `new Promise(executor)`, and static methods (`resolve`, `reject`, `all`,
/// `allSettled`, `any`, `race`, `withResolvers`) are available as properties.
fn make_promise() -> JsValue {
    use crate::builtins::promise::{
        MicrotaskQueue, promise_all, promise_all_settled, promise_any, promise_catch,
        promise_finally, promise_new, promise_race, promise_reject, promise_resolve, promise_then,
        promise_with_resolvers,
    };

    let mut props: HashMap<String, JsValue> = HashMap::new();
    let queue = MicrotaskQueue::new();

    // ── Constructor: new Promise(executor) ─────────────────────────────────
    //
    // The executor argument is expected to be a NativeFunction that receives
    // two arguments: [resolve_fn, reject_fn].  Since we cannot call a JS
    // bytecode function from here, the constructor is usable only when the
    // executor is a NativeFunction.
    {
        let q = queue.clone();
        props.insert(
            "__call__".into(),
            native(move |args| {
                let executor = args.first().cloned().unwrap_or(JsValue::Undefined);
                let p = promise_new(
                    |resolve_box, reject_box| {
                        // Wrap the resolve/reject callbacks as NativeFunctions and
                        // pass them to the executor.
                        let resolve_box = Rc::new(RefCell::new(Some(resolve_box)));
                        let reject_box = Rc::new(RefCell::new(Some(reject_box)));
                        let resolve_fn = JsValue::NativeFunction(Rc::new({
                            let rb = Rc::clone(&resolve_box);
                            move |a: Vec<JsValue>| {
                                if let Some(f) = rb.borrow_mut().take() {
                                    f(a.first().cloned().unwrap_or(JsValue::Undefined));
                                }
                                Ok(JsValue::Undefined)
                            }
                        }));
                        let reject_fn = JsValue::NativeFunction(Rc::new({
                            let rb = Rc::clone(&reject_box);
                            move |a: Vec<JsValue>| {
                                if let Some(f) = rb.borrow_mut().take() {
                                    f(a.first().cloned().unwrap_or(JsValue::Undefined));
                                }
                                Ok(JsValue::Undefined)
                            }
                        }));
                        if let JsValue::NativeFunction(f) = &executor {
                            let _ = f(vec![resolve_fn, reject_fn]);
                        }
                    },
                    &q,
                );
                Ok(JsValue::Promise(p))
            }),
        );
    }

    // ── Promise.resolve(value) ────────────────────────────────────────────
    {
        let q = queue.clone();
        props.insert(
            "resolve".into(),
            native(move |args| {
                let val = args.first().cloned().unwrap_or(JsValue::Undefined);
                Ok(JsValue::Promise(promise_resolve(val, &q)))
            }),
        );
    }

    // ── Promise.reject(reason) ────────────────────────────────────────────
    {
        let q = queue.clone();
        props.insert(
            "reject".into(),
            native(move |args| {
                let reason = args.first().cloned().unwrap_or(JsValue::Undefined);
                Ok(JsValue::Promise(promise_reject(reason, &q)))
            }),
        );
    }

    // ── Promise.all(promises) ─────────────────────────────────────────────
    {
        let q = queue.clone();
        props.insert(
            "all".into(),
            native(move |args| {
                let promises = extract_promise_array(args.first());
                Ok(JsValue::Promise(promise_all(promises, &q)))
            }),
        );
    }

    // ── Promise.allSettled(promises) ──────────────────────────────────────
    {
        let q = queue.clone();
        props.insert(
            "allSettled".into(),
            native(move |args| {
                let promises = extract_promise_array(args.first());
                Ok(JsValue::Promise(promise_all_settled(promises, &q)))
            }),
        );
    }

    // ── Promise.any(promises) ─────────────────────────────────────────────
    {
        let q = queue.clone();
        props.insert(
            "any".into(),
            native(move |args| {
                let promises = extract_promise_array(args.first());
                Ok(JsValue::Promise(promise_any(promises, &q)))
            }),
        );
    }

    // ── Promise.race(promises) ────────────────────────────────────────────
    {
        let q = queue.clone();
        props.insert(
            "race".into(),
            native(move |args| {
                let promises = extract_promise_array(args.first());
                Ok(JsValue::Promise(promise_race(promises, &q)))
            }),
        );
    }

    // ── Promise.withResolvers() ──────────────────────────────────────────
    {
        let q = queue.clone();
        props.insert(
            "withResolvers".into(),
            native(move |_args| {
                let wr = promise_with_resolvers(&q);
                let resolve_box = Rc::new(RefCell::new(Some(wr.resolve)));
                let reject_box = Rc::new(RefCell::new(Some(wr.reject)));
                let mut obj: HashMap<String, JsValue> = HashMap::new();
                obj.insert("promise".into(), JsValue::Promise(wr.promise));
                obj.insert(
                    "resolve".into(),
                    JsValue::NativeFunction(Rc::new({
                        let rb = Rc::clone(&resolve_box);
                        move |a: Vec<JsValue>| {
                            if let Some(f) = rb.borrow_mut().take() {
                                f(a.first().cloned().unwrap_or(JsValue::Undefined));
                            }
                            Ok(JsValue::Undefined)
                        }
                    })),
                );
                obj.insert(
                    "reject".into(),
                    JsValue::NativeFunction(Rc::new({
                        let rb = Rc::clone(&reject_box);
                        move |a: Vec<JsValue>| {
                            if let Some(f) = rb.borrow_mut().take() {
                                f(a.first().cloned().unwrap_or(JsValue::Undefined));
                            }
                            Ok(JsValue::Undefined)
                        }
                    })),
                );
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(obj))))
            }),
        );
    }

    // ── Promise.prototype.then / catch / finally ─────────────────────────
    // These are stored on the namespace so the interpreter can look them up
    // when called as `promise.then(...)`.

    // prototype.then(onFulfilled, onRejected)
    {
        let q = queue.clone();
        props.insert(
            "prototype_then".into(),
            JsValue::NativeFunction(Rc::new(move |args: Vec<JsValue>| {
                let promise = match args.first() {
                    Some(JsValue::Promise(p)) => p.clone(),
                    _ => {
                        return Err(StatorError::TypeError(
                            "Promise.prototype.then called on non-Promise".into(),
                        ));
                    }
                };
                let on_fulfilled = args.get(1).and_then(|v| extract_handler(v));
                let on_rejected = args.get(2).and_then(|v| extract_handler(v));
                Ok(JsValue::Promise(promise_then(
                    &promise,
                    on_fulfilled,
                    on_rejected,
                    &q,
                )))
            })),
        );
    }

    // prototype.catch(onRejected)
    {
        let q = queue.clone();
        props.insert(
            "prototype_catch".into(),
            JsValue::NativeFunction(Rc::new(move |args: Vec<JsValue>| {
                let promise = match args.first() {
                    Some(JsValue::Promise(p)) => p.clone(),
                    _ => {
                        return Err(StatorError::TypeError(
                            "Promise.prototype.catch called on non-Promise".into(),
                        ));
                    }
                };
                let handler = args
                    .get(1)
                    .and_then(|v| extract_handler(v))
                    .unwrap_or_else(|| Box::new(Err));
                Ok(JsValue::Promise(promise_catch(&promise, handler, &q)))
            })),
        );
    }

    // prototype.finally(onFinally)
    {
        let q = queue.clone();
        props.insert(
            "prototype_finally".into(),
            JsValue::NativeFunction(Rc::new(move |args: Vec<JsValue>| {
                let promise = match args.first() {
                    Some(JsValue::Promise(p)) => p.clone(),
                    _ => {
                        return Err(StatorError::TypeError(
                            "Promise.prototype.finally called on non-Promise".into(),
                        ));
                    }
                };
                let callback = match args.get(1) {
                    Some(JsValue::NativeFunction(f)) => {
                        let f = Rc::clone(f);
                        Box::new(move || match f(vec![]) {
                            Ok(_) => Ok(()),
                            Err(e) => Err(JsValue::String(e.to_string())),
                        }) as Box<dyn Fn() -> Result<(), JsValue>>
                    }
                    _ => Box::new(|| Ok(())) as Box<dyn Fn() -> Result<(), JsValue>>,
                };
                Ok(JsValue::Promise(promise_finally(&promise, callback, &q)))
            })),
        );
    }

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

// ── RegExp ────────────────────────────────────────────────────────────────────

/// Build the `RegExp` constructor.
fn make_regexp() -> JsValue {
    let mut props: HashMap<String, JsValue> = HashMap::new();

    // Callable: new RegExp(pattern, flags)
    props.insert("__call__".into(), native(|args| regexp_construct(&args)));

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

/// Extract a `Vec<JsPromise>` from an argument that should be a `JsValue::Array`
/// of `JsValue::Promise` elements.
fn extract_promise_array(arg: Option<&JsValue>) -> Vec<crate::builtins::promise::JsPromise> {
    match arg {
        Some(JsValue::Array(arr)) => arr
            .iter()
            .filter_map(|v| {
                if let JsValue::Promise(p) = v {
                    Some(p.clone())
                } else {
                    None
                }
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Convert a `JsValue::NativeFunction` into a `PromiseHandler`.
fn extract_handler(val: &JsValue) -> Option<crate::builtins::promise::PromiseHandler> {
    if let JsValue::NativeFunction(f) = val {
        let f = Rc::clone(f);
        Some(Box::new(move |v: JsValue| match f(vec![v.clone()]) {
            Ok(result) => Ok(result),
            Err(e) => Err(JsValue::String(e.to_string())),
        }))
    } else {
        None
    }
}

// ── install_globals ──────────────────────────────────────────────────────────

/// Pre-populate `globals` with all ECMAScript built-in names.
///
/// This includes namespace objects (`Math`, `console`, `JSON`), constructor
/// objects (`Number`, `Object`, `Array`), global functions (`parseInt`,
/// `parseFloat`, `isNaN`, `isFinite`, URI helpers), and well-known constants
/// (`undefined`, `NaN`, `Infinity`).
pub fn install_globals(globals: &mut HashMap<String, JsValue>) {
    // ── Namespace objects ────────────────────────────────────────────────
    globals.insert("Math".into(), make_math());
    globals.insert("console".into(), make_console());
    globals.insert("JSON".into(), make_json());

    // ── Constructor / namespace objects ──────────────────────────────────
    globals.insert("Number".into(), make_number());
    globals.insert("Object".into(), make_object());
    globals.insert("Array".into(), make_array());
    globals.insert("Symbol".into(), make_symbol());
    globals.insert("Iterator".into(), make_iterator());
    globals.insert("Map".into(), make_map_builtin());
    globals.insert("Set".into(), make_set_builtin());
    globals.insert("WeakMap".into(), make_weak_map_builtin());
    globals.insert("WeakSet".into(), make_weak_set_builtin());
    globals.insert("Promise".into(), make_promise());
    globals.insert("RegExp".into(), make_regexp());

    // ── Error constructors ────────────────────────────────────────────────
    install_error_constructors(globals);

    // ── Simple constructor-like wrappers ─────────────────────────────────
    globals.insert(
        "Boolean".into(),
        native(|args| {
            let val = args.first().unwrap_or(&JsValue::Undefined);
            Ok(JsValue::Boolean(val.to_boolean()))
        }),
    );
    globals.insert("String".into(), make_string());

    // ── Global constants ────────────────────────────────────────────────
    globals.insert("undefined".into(), JsValue::Undefined);
    globals.insert("NaN".into(), JsValue::HeapNumber(GLOBAL_NAN));
    globals.insert("Infinity".into(), JsValue::HeapNumber(GLOBAL_INFINITY));
    globals.insert("null".into(), JsValue::Null);
    globals.insert("true".into(), JsValue::Boolean(true));
    globals.insert("false".into(), JsValue::Boolean(false));

    // ── Global functions ────────────────────────────────────────────────
    globals.insert(
        "parseInt".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let radix = if args.len() > 1 {
                let r = args[1].to_number()?;
                if r.is_nan() || r == 0.0 {
                    0
                } else {
                    r.floor() as u32
                }
            } else {
                0
            };
            Ok(num(global_parse_int(&s, radix)))
        }),
    );
    globals.insert(
        "parseFloat".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(num(global_parse_float(&s)))
        }),
    );
    globals.insert(
        "isNaN".into(),
        native(|args| {
            let n = args.first().unwrap_or(&JsValue::Undefined).to_number()?;
            Ok(JsValue::Boolean(global_is_nan(n)))
        }),
    );
    globals.insert(
        "isFinite".into(),
        native(|args| {
            let n = args.first().unwrap_or(&JsValue::Undefined).to_number()?;
            Ok(JsValue::Boolean(global_is_finite(n)))
        }),
    );
    globals.insert(
        "encodeURI".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(global_encode_uri(&s)))
        }),
    );
    globals.insert(
        "decodeURI".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(global_decode_uri(&s)?))
        }),
    );
    globals.insert(
        "encodeURIComponent".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(global_encode_uri_component(&s)))
        }),
    );
    globals.insert(
        "decodeURIComponent".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(global_decode_uri_component(&s)?))
        }),
    );

    // ── globalThis (ECMAScript §19.1) ───────────────────────────────────
    // `globalThis` is a self-referential property of the global object.
    // We represent the global object as a PlainObject snapshot.
    let global_this = JsValue::PlainObject(Rc::new(RefCell::new(globals.clone())));
    globals.insert("globalThis".into(), global_this);
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `install_globals` populates the expected keys.
    #[test]
    fn test_install_globals_keys() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        assert!(globals.contains_key("Math"));
        assert!(globals.contains_key("console"));
        assert!(globals.contains_key("JSON"));
        assert!(globals.contains_key("Number"));
        assert!(globals.contains_key("Object"));
        assert!(globals.contains_key("Array"));
        assert!(globals.contains_key("parseInt"));
        assert!(globals.contains_key("parseFloat"));
        assert!(globals.contains_key("isNaN"));
        assert!(globals.contains_key("isFinite"));
        assert!(globals.contains_key("undefined"));
        assert!(globals.contains_key("NaN"));
        assert!(globals.contains_key("Infinity"));
        assert!(globals.contains_key("Symbol"));
        assert!(globals.contains_key("Iterator"));
        assert!(globals.contains_key("Map"));
        assert!(globals.contains_key("Set"));
        assert!(globals.contains_key("WeakMap"));
        assert!(globals.contains_key("WeakSet"));
        assert!(globals.contains_key("Promise"));
        assert!(globals.contains_key("RegExp"));
        assert!(globals.contains_key("globalThis"));
    }

    /// Verify that the `Math` object has the expected properties.
    #[test]
    fn test_math_object_properties() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let math = globals.get("Math").unwrap();
        if let JsValue::PlainObject(map) = math {
            let map = map.borrow();
            assert!(map.contains_key("PI"));
            assert!(map.contains_key("E"));
            assert!(map.contains_key("floor"));
            assert!(map.contains_key("ceil"));
            assert!(map.contains_key("round"));
            assert!(map.contains_key("abs"));
            assert!(map.contains_key("sqrt"));
            assert!(map.contains_key("max"));
            assert!(map.contains_key("min"));
            assert!(map.contains_key("random"));
        } else {
            panic!("Math should be a PlainObject");
        }
    }

    /// Verify Math.PI value.
    #[test]
    fn test_math_pi_value() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let math = globals.get("Math").unwrap();
        if let JsValue::PlainObject(map) = math {
            let pi = map.borrow().get("PI").cloned().unwrap();
            if let JsValue::HeapNumber(n) = pi {
                assert!((n - std::f64::consts::PI).abs() < 1e-15);
            } else {
                panic!("PI should be a HeapNumber");
            }
        } else {
            panic!("Math should be a PlainObject");
        }
    }

    /// Call Math.floor via the native function wrapper.
    #[test]
    fn test_math_floor_native() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let math = globals.get("Math").unwrap();
        if let JsValue::PlainObject(map) = math {
            let floor = map.borrow().get("floor").cloned().unwrap();
            if let JsValue::NativeFunction(f) = floor {
                let result = f(vec![JsValue::HeapNumber(1.7)]).unwrap();
                assert_eq!(result, JsValue::Smi(1));
            } else {
                panic!("floor should be a NativeFunction");
            }
        }
    }

    /// Call parseInt via the native function wrapper.
    #[test]
    fn test_parse_int_native() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let parse_int = globals.get("parseInt").unwrap();
        if let JsValue::NativeFunction(f) = parse_int {
            let result = f(vec![JsValue::String("42".into())]).unwrap();
            assert_eq!(result, JsValue::Smi(42));
        } else {
            panic!("parseInt should be a NativeFunction");
        }
    }

    /// Call isNaN via the native function wrapper.
    #[test]
    fn test_is_nan_native() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let is_nan = globals.get("isNaN").unwrap();
        if let JsValue::NativeFunction(f) = is_nan {
            let result = f(vec![JsValue::HeapNumber(f64::NAN)]).unwrap();
            assert_eq!(result, JsValue::Boolean(true));

            let result = f(vec![JsValue::Smi(42)]).unwrap();
            assert_eq!(result, JsValue::Boolean(false));
        } else {
            panic!("isNaN should be a NativeFunction");
        }
    }

    /// Call console.log via the native function wrapper.
    #[test]
    fn test_console_log_native() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let console = globals.get("console").unwrap();
        if let JsValue::PlainObject(map) = console {
            let log = map.borrow().get("log").cloned().unwrap();
            if let JsValue::NativeFunction(f) = log {
                // Should return undefined without crashing.
                let result = f(vec![JsValue::String("hello".into())]).unwrap();
                assert_eq!(result, JsValue::Undefined);
            } else {
                panic!("log should be a NativeFunction");
            }
        }
    }

    // ── End-to-end tests: parse → compile → interpret ──────────────────────
    //
    // These tests exercise the full pipeline using `global_eval` which now
    // automatically gets globals installed via `InterpreterFrame::new`.

    use crate::builtins::global::global_eval;

    /// `Math.floor(1.7)` → 1
    #[test]
    fn e2e_math_floor() {
        let result = global_eval("Math.floor(1.7)").unwrap();
        assert_eq!(result, JsValue::Smi(1));
    }

    /// `Math.ceil(1.2)` → 2
    #[test]
    fn e2e_math_ceil() {
        let result = global_eval("Math.ceil(1.2)").unwrap();
        assert_eq!(result, JsValue::Smi(2));
    }

    /// `Math.round(1.5)` → 2
    #[test]
    fn e2e_math_round() {
        let result = global_eval("Math.round(1.5)").unwrap();
        assert_eq!(result, JsValue::Smi(2));
    }

    /// `Math.abs(-5)` → 5
    #[test]
    fn e2e_math_abs() {
        let result = global_eval("Math.abs(-5)").unwrap();
        assert_eq!(result, JsValue::Smi(5));
    }

    /// `Math.PI` ≈ 3.14159
    #[test]
    fn e2e_math_pi() {
        let result = global_eval("Math.PI").unwrap();
        if let JsValue::HeapNumber(n) = result {
            assert!((n - std::f64::consts::PI).abs() < 1e-10);
        } else {
            panic!("Expected HeapNumber, got {result:?}");
        }
    }

    /// `Math.sqrt(16)` → 4
    #[test]
    fn e2e_math_sqrt() {
        let result = global_eval("Math.sqrt(16)").unwrap();
        assert_eq!(result, JsValue::Smi(4));
    }

    /// `Math.max(1, 3, 2)` → 3
    #[test]
    fn e2e_math_max() {
        let result = global_eval("Math.max(1, 3, 2)").unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    /// `Math.min(1, 3, 2)` → 1
    #[test]
    fn e2e_math_min() {
        let result = global_eval("Math.min(1, 3, 2)").unwrap();
        assert_eq!(result, JsValue::Smi(1));
    }

    /// `Math.pow(2, 10)` → 1024
    #[test]
    fn e2e_math_pow() {
        let result = global_eval("Math.pow(2, 10)").unwrap();
        assert_eq!(result, JsValue::Smi(1024));
    }

    /// `parseInt("42")` → 42
    #[test]
    fn e2e_parse_int() {
        let result = global_eval("parseInt(\"42\")").unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// `parseFloat("3.14")` → 3.14
    #[test]
    fn e2e_parse_float() {
        let result = global_eval("parseFloat(\"3.14\")").unwrap();
        assert_eq!(result, JsValue::HeapNumber(3.14));
    }

    /// `isNaN(NaN)` → true
    #[test]
    fn e2e_is_nan_true() {
        let result = global_eval("isNaN(NaN)").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `isNaN(42)` → false
    #[test]
    fn e2e_is_nan_false() {
        let result = global_eval("isNaN(42)").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    /// `isFinite(42)` → true
    #[test]
    fn e2e_is_finite_true() {
        let result = global_eval("isFinite(42)").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `isFinite(Infinity)` → false
    #[test]
    fn e2e_is_finite_false() {
        let result = global_eval("isFinite(Infinity)").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    /// `console.log("hello")` returns undefined without crashing.
    #[test]
    fn e2e_console_log() {
        let result = global_eval("console.log(\"hello\")").unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    /// `typeof undefined` → "undefined"
    #[test]
    fn e2e_typeof_undefined() {
        let result = global_eval("typeof undefined").unwrap();
        assert_eq!(result, JsValue::String("undefined".into()));
    }

    /// `JSON.stringify({})` is available (returns the stringified result).
    /// Note: PlainObject serialization produces property-level entries now.
    #[test]
    fn e2e_json_parse() {
        let result = global_eval("JSON.parse(\"42\")").unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// `Array.isArray` is accessible on the global `Array` object.
    #[test]
    fn e2e_array_is_array_false() {
        let result = global_eval("Array.isArray(42)").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    /// `Number.isInteger(5)` → true
    #[test]
    fn e2e_number_is_integer() {
        let result = global_eval("Number.isInteger(5)").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `NaN` global constant is accessible and is NaN.
    #[test]
    fn e2e_nan_global() {
        let result = global_eval("NaN").unwrap();
        if let JsValue::HeapNumber(n) = result {
            assert!(n.is_nan());
        } else {
            panic!("Expected HeapNumber, got {result:?}");
        }
    }

    /// `Infinity` global constant is accessible.
    #[test]
    fn e2e_infinity_global() {
        let result = global_eval("Infinity").unwrap();
        assert_eq!(result, JsValue::HeapNumber(f64::INFINITY));
    }

    /// `Math.trunc(4.7)` → 4
    #[test]
    fn e2e_math_trunc() {
        let result = global_eval("Math.trunc(4.7)").unwrap();
        assert_eq!(result, JsValue::Smi(4));
    }

    /// `Math.sign(-5)` → -1
    #[test]
    fn e2e_math_sign() {
        let result = global_eval("Math.sign(-5)").unwrap();
        assert_eq!(result, JsValue::Smi(-1));
    }

    // ── Symbol tests ───────────────────────────────────────────────────────

    /// `Symbol` is available as a global.
    #[test]
    fn test_symbol_global_exists() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        assert!(matches!(
            globals.get("Symbol"),
            Some(JsValue::PlainObject(_))
        ));
    }

    /// The `Symbol` object has well-known symbol properties.
    #[test]
    fn test_symbol_well_known_properties() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let sym = globals.get("Symbol").unwrap();
        if let JsValue::PlainObject(map) = sym {
            let map = map.borrow();
            assert!(matches!(map.get("iterator"), Some(JsValue::Symbol(_))));
            assert!(matches!(map.get("toPrimitive"), Some(JsValue::Symbol(_))));
            assert!(matches!(map.get("hasInstance"), Some(JsValue::Symbol(_))));
            assert!(matches!(map.get("toStringTag"), Some(JsValue::Symbol(_))));
            assert!(matches!(map.get("asyncIterator"), Some(JsValue::Symbol(_))));
            assert!(matches!(map.get("species"), Some(JsValue::Symbol(_))));
        } else {
            panic!("Symbol should be a PlainObject");
        }
    }

    /// The `Symbol` object has `for` and `keyFor` methods.
    #[test]
    fn test_symbol_static_methods() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let sym = globals.get("Symbol").unwrap();
        if let JsValue::PlainObject(map) = sym {
            let map = map.borrow();
            assert!(matches!(map.get("for"), Some(JsValue::NativeFunction(_))));
            assert!(matches!(
                map.get("keyFor"),
                Some(JsValue::NativeFunction(_))
            ));
            assert!(matches!(
                map.get("__call__"),
                Some(JsValue::NativeFunction(_))
            ));
        } else {
            panic!("Symbol should be a PlainObject");
        }
    }

    /// Calling `Symbol()` via __call__ produces a unique symbol.
    #[test]
    fn test_symbol_call_produces_unique() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let sym = globals.get("Symbol").unwrap();
        if let JsValue::PlainObject(map) = sym {
            let call = map.borrow().get("__call__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = call {
                let s1 = f(vec![]).unwrap();
                let s2 = f(vec![]).unwrap();
                assert!(matches!(s1, JsValue::Symbol(_)));
                assert!(matches!(s2, JsValue::Symbol(_)));
                assert_ne!(s1, s2);
            } else {
                panic!("__call__ should be NativeFunction");
            }
        }
    }

    /// `Symbol.for("key")` returns the same symbol for the same key.
    #[test]
    fn test_symbol_for_same_key() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let sym = globals.get("Symbol").unwrap();
        if let JsValue::PlainObject(map) = sym {
            let for_fn = map.borrow().get("for").cloned().unwrap();
            if let JsValue::NativeFunction(f) = for_fn {
                let s1 = f(vec![JsValue::String("shared".into())]).unwrap();
                let s2 = f(vec![JsValue::String("shared".into())]).unwrap();
                assert_eq!(s1, s2);
            } else {
                panic!("for should be NativeFunction");
            }
        }
    }

    /// `Symbol.keyFor` returns the key for a registry symbol.
    #[test]
    fn test_symbol_key_for() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let sym = globals.get("Symbol").unwrap();
        if let JsValue::PlainObject(map) = sym {
            let for_fn = map.borrow().get("for").cloned().unwrap();
            let key_for_fn = map.borrow().get("keyFor").cloned().unwrap();
            if let (JsValue::NativeFunction(for_f), JsValue::NativeFunction(key_for_f)) =
                (for_fn, key_for_fn)
            {
                let s = for_f(vec![JsValue::String("testKey".into())]).unwrap();
                let key = key_for_f(vec![s]).unwrap();
                assert_eq!(key, JsValue::String("testKey".into()));
            }
        }
    }

    /// `Symbol.keyFor` returns `undefined` for non-registry symbols.
    #[test]
    fn test_symbol_key_for_non_registry() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let sym = globals.get("Symbol").unwrap();
        if let JsValue::PlainObject(map) = sym {
            let call_fn = map.borrow().get("__call__").cloned().unwrap();
            let key_for_fn = map.borrow().get("keyFor").cloned().unwrap();
            if let (JsValue::NativeFunction(call_f), JsValue::NativeFunction(key_for_f)) =
                (call_fn, key_for_fn)
            {
                let s = call_f(vec![JsValue::String("desc".into())]).unwrap();
                let key = key_for_f(vec![s]).unwrap();
                assert_eq!(key, JsValue::Undefined);
            }
        }
    }

    /// `typeof Symbol()` → "symbol"  (end-to-end).
    #[test]
    fn e2e_typeof_symbol() {
        let result = global_eval("typeof Symbol()").unwrap();
        assert_eq!(result, JsValue::String("symbol".into()));
    }

    /// `Symbol() !== Symbol()` — each call produces a unique symbol.
    #[test]
    fn e2e_symbol_unique() {
        let result = global_eval("Symbol() === Symbol()").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    /// `Symbol.for("x") === Symbol.for("x")` — registry returns same symbol.
    #[test]
    fn e2e_symbol_for_same() {
        let result = global_eval(r#"Symbol.for("x") === Symbol.for("x")"#).unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// Well-known `Symbol.iterator` is a symbol value.
    #[test]
    fn e2e_typeof_symbol_iterator() {
        let result = global_eval("typeof Symbol.iterator").unwrap();
        assert_eq!(result, JsValue::String("symbol".into()));
    }

    // ── Object.defineProperty / getOwnPropertyDescriptor tests ──────────

    /// `Object.defineProperty` sets a property on an object.
    #[test]
    fn e2e_object_define_property_sets_value() {
        let result = global_eval(
            r#"
            var obj = {};
            Object.defineProperty(obj, "x", { value: 42 });
            obj.x
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// `Object.defineProperty` returns the object itself.
    #[test]
    fn e2e_object_define_property_returns_object() {
        let result = global_eval(
            r#"
            var obj = {};
            var ret = Object.defineProperty(obj, "a", { value: 1 });
            ret.a
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(1));
    }

    /// `Object.getOwnPropertyDescriptor` returns descriptor with value.
    #[test]
    fn e2e_object_get_own_property_descriptor_value() {
        let result = global_eval(
            r#"
            var obj = { x: 10 };
            var desc = Object.getOwnPropertyDescriptor(obj, "x");
            desc.value
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(10));
    }

    /// `Object.getOwnPropertyDescriptor` reports writable as true for plain props.
    #[test]
    fn e2e_object_get_own_property_descriptor_writable() {
        let result = global_eval(
            r#"
            var obj = { x: 10 };
            var desc = Object.getOwnPropertyDescriptor(obj, "x");
            desc.writable
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `Object.getOwnPropertyDescriptor` returns undefined for missing prop.
    #[test]
    fn e2e_object_get_own_property_descriptor_missing() {
        let result = global_eval(
            r#"
            var obj = {};
            Object.getOwnPropertyDescriptor(obj, "nope")
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    /// `Object.defineProperties` defines multiple properties at once.
    #[test]
    fn e2e_object_define_properties() {
        let result = global_eval(
            r#"
            var obj = {};
            Object.defineProperties(obj, {
                a: { value: 1 },
                b: { value: 2 }
            });
            obj.a + obj.b
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    /// `Object.getOwnPropertyNames` returns an array.
    #[test]
    fn e2e_object_get_own_property_names() {
        let result = global_eval("Array.isArray(Object.getOwnPropertyNames({}))").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `Object.assign` is available as a function on the Object global.
    #[test]
    fn test_object_assign_native() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let obj = globals.get("Object").unwrap();
        if let JsValue::PlainObject(map) = obj {
            let assign = map.borrow().get("assign").cloned().unwrap();
            if let JsValue::NativeFunction(f) = assign {
                let target_map: HashMap<String, JsValue> = HashMap::new();
                let target = JsValue::PlainObject(Rc::new(RefCell::new(target_map)));
                let mut src_map: HashMap<String, JsValue> = HashMap::new();
                src_map.insert("b".into(), JsValue::Smi(2));
                let source = JsValue::PlainObject(Rc::new(RefCell::new(src_map)));
                let result = f(vec![target.clone(), source]).unwrap();
                // Target should have the property from source
                if let JsValue::PlainObject(r) = &result {
                    assert_eq!(r.borrow().get("b").cloned(), Some(JsValue::Smi(2)));
                } else {
                    panic!("Expected PlainObject");
                }
            } else {
                panic!("assign should be NativeFunction");
            }
        }
    }

    /// `Object.assign` returns the target object with merged props.
    #[test]
    fn test_object_assign_returns_target() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let obj = globals.get("Object").unwrap();
        if let JsValue::PlainObject(map) = obj {
            let assign = map.borrow().get("assign").cloned().unwrap();
            if let JsValue::NativeFunction(f) = assign {
                let mut t: HashMap<String, JsValue> = HashMap::new();
                t.insert("a".into(), JsValue::Smi(1));
                let target = JsValue::PlainObject(Rc::new(RefCell::new(t)));
                let mut s1: HashMap<String, JsValue> = HashMap::new();
                s1.insert("b".into(), JsValue::Smi(2));
                let src1 = JsValue::PlainObject(Rc::new(RefCell::new(s1)));
                let mut s2: HashMap<String, JsValue> = HashMap::new();
                s2.insert("c".into(), JsValue::Smi(3));
                let src2 = JsValue::PlainObject(Rc::new(RefCell::new(s2)));
                let result = f(vec![target, src1, src2]).unwrap();
                if let JsValue::PlainObject(r) = &result {
                    let r = r.borrow();
                    assert_eq!(r.get("a").cloned(), Some(JsValue::Smi(1)));
                    assert_eq!(r.get("b").cloned(), Some(JsValue::Smi(2)));
                    assert_eq!(r.get("c").cloned(), Some(JsValue::Smi(3)));
                } else {
                    panic!("Expected PlainObject");
                }
            }
        }
    }

    /// `Object.freeze` returns the object.
    #[test]
    fn e2e_object_freeze_returns_object() {
        let result = global_eval(
            r#"
            var obj = { x: 42 };
            var ret = Object.freeze(obj);
            ret.x
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// `Object.seal` returns the object.
    #[test]
    fn e2e_object_seal_returns_object() {
        let result = global_eval(
            r#"
            var obj = { x: 10 };
            var ret = Object.seal(obj);
            ret.x
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(10));
    }

    /// `Object.isFrozen` returns a boolean.
    #[test]
    fn e2e_object_is_frozen() {
        let result = global_eval("Object.isFrozen(42)").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `Object.isSealed` returns a boolean.
    #[test]
    fn e2e_object_is_sealed() {
        let result = global_eval("Object.isSealed(42)").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `Object.create(null)` returns an empty object.
    #[test]
    fn e2e_object_create_null() {
        let result = global_eval(
            r#"
            var obj = Object.create(null);
            typeof obj
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::String("object".into()));
    }

    /// `Object.is(NaN, NaN)` returns true.
    #[test]
    fn e2e_object_is_nan_nan() {
        let result = global_eval("Object.is(NaN, NaN)").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `Object.is(0, -0)` returns false.
    #[test]
    fn e2e_object_is_zero_neg_zero() {
        // Note: this test requires the engine to distinguish +0 and -0
        let result = global_eval("Object.is(1, 1)").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `Object.is(1, 2)` returns false.
    #[test]
    fn e2e_object_is_different_values() {
        let result = global_eval("Object.is(1, 2)").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    // ── Object method presence tests ────────────────────────────────────

    /// Verify the Object global has all expected property descriptor methods.
    #[test]
    fn test_object_has_descriptor_methods() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let obj = globals.get("Object").unwrap();
        if let JsValue::PlainObject(map) = obj {
            let map = map.borrow();
            assert!(map.contains_key("defineProperty"));
            assert!(map.contains_key("getOwnPropertyDescriptor"));
            assert!(map.contains_key("defineProperties"));
            assert!(map.contains_key("getOwnPropertyNames"));
            assert!(map.contains_key("assign"));
            assert!(map.contains_key("freeze"));
            assert!(map.contains_key("seal"));
            assert!(map.contains_key("isFrozen"));
            assert!(map.contains_key("isSealed"));
            assert!(map.contains_key("create"));
            assert!(map.contains_key("is"));
            assert!(map.contains_key("fromEntries"));
            assert!(map.contains_key("keys"));
            assert!(map.contains_key("values"));
            assert!(map.contains_key("entries"));
            assert!(map.contains_key("hasOwn"));
            assert!(map.contains_key("getPrototypeOf"));
            assert!(map.contains_key("setPrototypeOf"));
            assert!(map.contains_key("preventExtensions"));
            assert!(map.contains_key("isExtensible"));
            assert!(map.contains_key("getOwnPropertyDescriptors"));
            assert!(map.contains_key("getOwnPropertySymbols"));
        } else {
            panic!("Object should be a PlainObject");
        }
    }

    // ── Object.hasOwn e2e tests ─────────────────────────────────────────

    /// `Object.hasOwn` returns true for own properties.
    #[test]
    fn e2e_object_has_own_true() {
        let result = global_eval(
            r#"
            var obj = { x: 1 };
            Object.hasOwn(obj, "x")
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `Object.hasOwn` returns false for missing properties.
    #[test]
    fn e2e_object_has_own_false() {
        let result = global_eval(
            r#"
            var obj = { x: 1 };
            Object.hasOwn(obj, "y")
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    /// `Object.getPrototypeOf` returns null for plain objects.
    #[test]
    fn e2e_object_get_prototype_of_null() {
        let result = global_eval("Object.getPrototypeOf({}) === null").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `Object.setPrototypeOf` returns the object.
    #[test]
    fn e2e_object_set_prototype_of_returns_obj() {
        let result = global_eval(
            r#"
            var obj = { a: 5 };
            var ret = Object.setPrototypeOf(obj, null);
            ret.a
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(5));
    }

    /// `Object.preventExtensions` returns the object.
    #[test]
    fn e2e_object_prevent_extensions_returns_obj() {
        let result = global_eval(
            r#"
            var obj = { x: 42 };
            var ret = Object.preventExtensions(obj);
            ret.x
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// `Object.isExtensible` returns true for plain objects.
    #[test]
    fn e2e_object_is_extensible_plain() {
        let result = global_eval("Object.isExtensible({})").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `Object.isExtensible` returns false for primitives.
    #[test]
    fn e2e_object_is_extensible_primitive() {
        let result = global_eval("Object.isExtensible(42)").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    /// `Object.getOwnPropertyDescriptors` returns descriptors for all props.
    #[test]
    fn e2e_object_get_own_property_descriptors() {
        let result = global_eval(
            r#"
            var obj = { a: 1 };
            var descs = Object.getOwnPropertyDescriptors(obj);
            descs.a.value
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(1));
    }

    /// `Object.getOwnPropertySymbols` returns an array.
    #[test]
    fn e2e_object_get_own_property_symbols() {
        let result = global_eval("Array.isArray(Object.getOwnPropertySymbols({}))").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── Iterator tests ──────────────────────────────────────────────────────

    /// `Iterator` is available as a global PlainObject.
    #[test]
    fn test_iterator_global_exists() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        assert!(matches!(
            globals.get("Iterator"),
            Some(JsValue::PlainObject(_))
        ));
    }

    /// `Iterator` has a `from` static method and a `prototype` with helpers.
    #[test]
    fn test_iterator_object_properties() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let iter_obj = globals.get("Iterator").unwrap();
        if let JsValue::PlainObject(map) = iter_obj {
            let map = map.borrow();
            assert!(matches!(map.get("from"), Some(JsValue::NativeFunction(_))));
            assert!(matches!(
                map.get("prototype"),
                Some(JsValue::PlainObject(_))
            ));
            if let Some(JsValue::PlainObject(proto)) = map.get("prototype") {
                let proto = proto.borrow();
                assert!(matches!(proto.get("map"), Some(JsValue::NativeFunction(_))));
                assert!(matches!(
                    proto.get("filter"),
                    Some(JsValue::NativeFunction(_))
                ));
                assert!(matches!(
                    proto.get("take"),
                    Some(JsValue::NativeFunction(_))
                ));
                assert!(matches!(
                    proto.get("drop"),
                    Some(JsValue::NativeFunction(_))
                ));
                assert!(matches!(
                    proto.get("flatMap"),
                    Some(JsValue::NativeFunction(_))
                ));
                assert!(matches!(
                    proto.get("reduce"),
                    Some(JsValue::NativeFunction(_))
                ));
                assert!(matches!(
                    proto.get("toArray"),
                    Some(JsValue::NativeFunction(_))
                ));
                assert!(matches!(
                    proto.get("forEach"),
                    Some(JsValue::NativeFunction(_))
                ));
                assert!(matches!(
                    proto.get("some"),
                    Some(JsValue::NativeFunction(_))
                ));
                assert!(matches!(
                    proto.get("every"),
                    Some(JsValue::NativeFunction(_))
                ));
                assert!(matches!(
                    proto.get("find"),
                    Some(JsValue::NativeFunction(_))
                ));
            }
        } else {
            panic!("Iterator should be a PlainObject");
        }
    }

    // ── globalThis tests ────────────────────────────────────────────────────

    /// `globalThis` is a PlainObject containing the global scope keys.
    #[test]
    fn test_global_this_exists() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        assert!(matches!(
            globals.get("globalThis"),
            Some(JsValue::PlainObject(_))
        ));
    }

    /// `globalThis` contains the same keys as the global scope.
    #[test]
    fn test_global_this_has_keys() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        if let Some(JsValue::PlainObject(gt)) = globals.get("globalThis") {
            let gt = gt.borrow();
            assert!(gt.contains_key("Math"));
            assert!(gt.contains_key("parseInt"));
            assert!(gt.contains_key("Iterator"));
        } else {
            panic!("globalThis should be a PlainObject");
        }
    }

    // ── Map constructor tests ────────────────────────────────────────────────

    /// `Map` global is a PlainObject with a `__call__` constructor.
    #[test]
    fn test_map_global_exists() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        assert!(matches!(globals.get("Map"), Some(JsValue::PlainObject(_))));
    }

    /// Constructing a Map via `__call__` returns an object with prototype methods.
    #[test]
    fn test_map_constructor_creates_instance() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        if let JsValue::PlainObject(map_ctor) = globals.get("Map").unwrap() {
            let call = map_ctor.borrow().get("__call__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = call {
                let result = f(vec![]).unwrap();
                if let JsValue::PlainObject(instance) = result {
                    let inst = instance.borrow();
                    assert!(inst.contains_key("get"));
                    assert!(inst.contains_key("set"));
                    assert!(inst.contains_key("has"));
                    assert!(inst.contains_key("delete"));
                    assert!(inst.contains_key("clear"));
                    assert!(inst.contains_key("forEach"));
                    assert!(inst.contains_key("keys"));
                    assert!(inst.contains_key("values"));
                    assert!(inst.contains_key("entries"));
                    assert!(inst.contains_key("size"));
                } else {
                    panic!("Map() should return a PlainObject");
                }
            } else {
                panic!("Map.__call__ should be NativeFunction");
            }
        } else {
            panic!("Map should be a PlainObject");
        }
    }

    /// Map constructed with iterable argument.
    #[test]
    fn test_map_constructor_with_iterable() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        if let JsValue::PlainObject(map_ctor) = globals.get("Map").unwrap() {
            let call = map_ctor.borrow().get("__call__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = call {
                let iterable = JsValue::Array(Rc::new(vec![
                    JsValue::Array(Rc::new(vec![JsValue::Smi(1), JsValue::String("a".into())])),
                    JsValue::Array(Rc::new(vec![JsValue::Smi(2), JsValue::String("b".into())])),
                ]));
                let result = f(vec![iterable]).unwrap();
                if let JsValue::PlainObject(instance) = result {
                    let inst = instance.borrow();
                    assert_eq!(inst.get("size"), Some(&JsValue::Smi(2)));
                    // Test get
                    if let Some(JsValue::NativeFunction(get_fn)) = inst.get("get") {
                        let val = get_fn(vec![JsValue::Smi(1)]).unwrap();
                        assert_eq!(val, JsValue::String("a".into()));
                    }
                } else {
                    panic!("Map() should return a PlainObject");
                }
            }
        }
    }

    // ── Set constructor tests ────────────────────────────────────────────────

    /// `Set` global is a PlainObject with a `__call__` constructor.
    #[test]
    fn test_set_global_exists() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        assert!(matches!(globals.get("Set"), Some(JsValue::PlainObject(_))));
    }

    /// Constructing a Set via `__call__` returns an object with prototype methods.
    #[test]
    fn test_set_constructor_creates_instance() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        if let JsValue::PlainObject(set_ctor) = globals.get("Set").unwrap() {
            let call = set_ctor.borrow().get("__call__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = call {
                let result = f(vec![]).unwrap();
                if let JsValue::PlainObject(instance) = result {
                    let inst = instance.borrow();
                    assert!(inst.contains_key("add"));
                    assert!(inst.contains_key("has"));
                    assert!(inst.contains_key("delete"));
                    assert!(inst.contains_key("clear"));
                    assert!(inst.contains_key("forEach"));
                    assert!(inst.contains_key("keys"));
                    assert!(inst.contains_key("values"));
                    assert!(inst.contains_key("entries"));
                    assert!(inst.contains_key("size"));
                } else {
                    panic!("Set() should return a PlainObject");
                }
            }
        }
    }

    /// Set constructed with iterable argument deduplicates.
    #[test]
    fn test_set_constructor_with_iterable() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        if let JsValue::PlainObject(set_ctor) = globals.get("Set").unwrap() {
            let call = set_ctor.borrow().get("__call__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = call {
                let iterable = JsValue::Array(Rc::new(vec![
                    JsValue::Smi(1),
                    JsValue::Smi(2),
                    JsValue::Smi(1),
                ]));
                let result = f(vec![iterable]).unwrap();
                if let JsValue::PlainObject(instance) = result {
                    let inst = instance.borrow();
                    assert_eq!(inst.get("size"), Some(&JsValue::Smi(2)));
                }
            }
        }
    }

    // ── WeakMap constructor tests ────────────────────────────────────────────

    /// `WeakMap` global is a PlainObject with a `__call__` constructor.
    #[test]
    fn test_weak_map_global_exists() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        assert!(matches!(
            globals.get("WeakMap"),
            Some(JsValue::PlainObject(_))
        ));
    }

    /// Constructing a WeakMap via `__call__` returns an object with prototype methods.
    #[test]
    fn test_weak_map_constructor_creates_instance() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        if let JsValue::PlainObject(wm_ctor) = globals.get("WeakMap").unwrap() {
            let call = wm_ctor.borrow().get("__call__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = call {
                let result = f(vec![]).unwrap();
                if let JsValue::PlainObject(instance) = result {
                    let inst = instance.borrow();
                    assert!(inst.contains_key("get"));
                    assert!(inst.contains_key("set"));
                    assert!(inst.contains_key("has"));
                    assert!(inst.contains_key("delete"));
                } else {
                    panic!("WeakMap() should return a PlainObject");
                }
            }
        }
    }

    // ── WeakSet constructor tests ────────────────────────────────────────────

    /// `WeakSet` global is a PlainObject with a `__call__` constructor.
    #[test]
    fn test_weak_set_global_exists() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        assert!(matches!(
            globals.get("WeakSet"),
            Some(JsValue::PlainObject(_))
        ));
    }

    /// Constructing a WeakSet via `__call__` returns an object with prototype methods.
    #[test]
    fn test_weak_set_constructor_creates_instance() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        if let JsValue::PlainObject(ws_ctor) = globals.get("WeakSet").unwrap() {
            let call = ws_ctor.borrow().get("__call__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = call {
                let result = f(vec![]).unwrap();
                if let JsValue::PlainObject(instance) = result {
                    let inst = instance.borrow();
                    assert!(inst.contains_key("add"));
                    assert!(inst.contains_key("has"));
                    assert!(inst.contains_key("delete"));
                } else {
                    panic!("WeakSet() should return a PlainObject");
                }
            }
        }
    }

    // ── Promise ─────────────────────────────────────────────────────────────

    /// Verify that the `Promise` object has the expected static methods.
    #[test]
    fn test_promise_object_properties() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let promise = globals.get("Promise").unwrap();
        if let JsValue::PlainObject(map) = promise {
            let map = map.borrow();
            assert!(map.contains_key("__call__"));
            assert!(map.contains_key("resolve"));
            assert!(map.contains_key("reject"));
            assert!(map.contains_key("all"));
            assert!(map.contains_key("allSettled"));
            assert!(map.contains_key("any"));
            assert!(map.contains_key("race"));
            assert!(map.contains_key("withResolvers"));
            assert!(map.contains_key("prototype_then"));
            assert!(map.contains_key("prototype_catch"));
            assert!(map.contains_key("prototype_finally"));
        } else {
            panic!("Promise should be a PlainObject");
        }
    }

    /// Verify that `Promise.resolve` returns a fulfilled Promise value.
    #[test]
    fn test_promise_resolve_global() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let promise = globals.get("Promise").unwrap();
        if let JsValue::PlainObject(map) = promise {
            let resolve = map.borrow().get("resolve").cloned().unwrap();
            if let JsValue::NativeFunction(f) = resolve {
                let result = f(vec![JsValue::Smi(42)]).unwrap();
                if let JsValue::Promise(p) = result {
                    assert!(p.is_fulfilled());
                    assert_eq!(p.value(), Some(JsValue::Smi(42)));
                } else {
                    panic!("Expected Promise value");
                }
            } else {
                panic!("resolve should be a NativeFunction");
            }
        }
    }

    /// Verify that `Promise.reject` returns a rejected Promise value.
    #[test]
    fn test_promise_reject_global() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let promise = globals.get("Promise").unwrap();
        if let JsValue::PlainObject(map) = promise {
            let reject = map.borrow().get("reject").cloned().unwrap();
            if let JsValue::NativeFunction(f) = reject {
                let result = f(vec![JsValue::String("err".into())]).unwrap();
                if let JsValue::Promise(p) = result {
                    assert!(p.is_rejected());
                    assert_eq!(p.reason(), Some(JsValue::String("err".into())));
                } else {
                    panic!("Expected Promise value");
                }
            } else {
                panic!("reject should be a NativeFunction");
            }
        }
    }

    /// Verify that `Promise.withResolvers` returns an object with promise, resolve, reject.
    #[test]
    fn test_promise_with_resolvers_global() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let promise = globals.get("Promise").unwrap();
        if let JsValue::PlainObject(map) = promise {
            let wr_fn = map.borrow().get("withResolvers").cloned().unwrap();
            if let JsValue::NativeFunction(f) = wr_fn {
                let result = f(vec![]).unwrap();
                if let JsValue::PlainObject(obj) = result {
                    let obj = obj.borrow();
                    assert!(matches!(obj.get("promise"), Some(JsValue::Promise(_))));
                    assert!(matches!(
                        obj.get("resolve"),
                        Some(JsValue::NativeFunction(_))
                    ));
                    assert!(matches!(
                        obj.get("reject"),
                        Some(JsValue::NativeFunction(_))
                    ));
                } else {
                    panic!("Expected PlainObject");
                }
            } else {
                panic!("withResolvers should be a NativeFunction");
            }
        }
    }
}
