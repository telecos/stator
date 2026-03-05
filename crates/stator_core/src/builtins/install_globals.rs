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
    global_decode_uri, global_decode_uri_component, global_encode_uri,
    global_encode_uri_component, global_is_finite, global_is_nan, global_parse_float,
    global_parse_int, GLOBAL_INFINITY, GLOBAL_NAN,
};
use crate::builtins::symbol::{
    symbol_create, symbol_for, symbol_key_for, SYMBOL_ASYNC_ITERATOR,
    SYMBOL_HAS_INSTANCE, SYMBOL_IS_CONCAT_SPREADABLE, SYMBOL_ITERATOR, SYMBOL_MATCH,
    SYMBOL_REPLACE, SYMBOL_SEARCH, SYMBOL_SPECIES, SYMBOL_SPLIT, SYMBOL_TO_PRIMITIVE,
    SYMBOL_TO_STRING_TAG, SYMBOL_UNSCOPABLES,
};
use crate::builtins::math::{
    math_abs, math_acos, math_asin, math_atan, math_atan2, math_cbrt, math_ceil, math_clz32,
    math_cos, math_floor, math_fround, math_hypot, math_imul, math_log, math_log10, math_log2,
    math_max, math_min, math_pow, math_random, math_round, math_sign, math_sin, math_sqrt,
    math_tan, math_trunc, MATH_E, MATH_LN10, MATH_LN2, MATH_LOG10E, MATH_LOG2E, MATH_PI,
    MATH_SQRT1_2, MATH_SQRT2,
};
use crate::error::StatorResult;
use crate::objects::value::JsValue;

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
    globals.insert("TypeError".into(), make_error_constructor(ErrorKind::TypeError));
    globals.insert("RangeError".into(), make_error_constructor(ErrorKind::RangeError));
    globals.insert("ReferenceError".into(), make_error_constructor(ErrorKind::ReferenceError));
    globals.insert("SyntaxError".into(), make_error_constructor(ErrorKind::SyntaxError));
    globals.insert("URIError".into(), make_error_constructor(ErrorKind::URIError));
    globals.insert("EvalError".into(), make_error_constructor(ErrorKind::EvalError));
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
            let parts: StatorResult<Vec<String>> =
                args.iter().map(|a| a.to_js_string()).collect();
            println!("{}", parts?.join(" "));
            Ok(JsValue::Undefined)
        }),
    );
    props.insert(
        "warn".into(),
        native(|args: Vec<JsValue>| {
            let parts: StatorResult<Vec<String>> =
                args.iter().map(|a| a.to_js_string()).collect();
            eprintln!("{}", parts?.join(" "));
            Ok(JsValue::Undefined)
        }),
    );
    props.insert(
        "error".into(),
        native(|args: Vec<JsValue>| {
            let parts: StatorResult<Vec<String>> =
                args.iter().map(|a| a.to_js_string()).collect();
            eprintln!("{}", parts?.join(" "));
            Ok(JsValue::Undefined)
        }),
    );
    props.insert(
        "info".into(),
        native(|args: Vec<JsValue>| {
            let parts: StatorResult<Vec<String>> =
                args.iter().map(|a| a.to_js_string()).collect();
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
            let text = args
                .first()
                .unwrap_or(&JsValue::Undefined)
                .to_js_string()?;
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
    // Number.parseInt
    props.insert(
        "parseInt".into(),
        native(|args| {
            let s = args
                .first()
                .unwrap_or(&JsValue::Undefined)
                .to_js_string()?;
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
            let s = args
                .first()
                .unwrap_or(&JsValue::Undefined)
                .to_js_string()?;
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
    props.insert("MIN_VALUE".into(), JsValue::HeapNumber(f64::MIN_POSITIVE));
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
                            let value =
                                sd.get("value").cloned().unwrap_or(JsValue::Undefined);
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
                JsValue::Null => {
                    Ok(JsValue::PlainObject(Rc::new(RefCell::new(HashMap::new()))))
                }
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
            let key = args
                .first()
                .unwrap_or(&JsValue::Undefined)
                .to_js_string()?;
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
    props.insert(
        "toStringTag".into(),
        JsValue::Symbol(SYMBOL_TO_STRING_TAG),
    );
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
    globals.insert(
        "String".into(),
        native(|args| {
            let val = args.first().unwrap_or(&JsValue::Undefined);
            Ok(JsValue::String(val.to_js_string()?))
        }),
    );

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
            let s = args
                .first()
                .unwrap_or(&JsValue::Undefined)
                .to_js_string()?;
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
            let s = args
                .first()
                .unwrap_or(&JsValue::Undefined)
                .to_js_string()?;
            Ok(num(global_parse_float(&s)))
        }),
    );
    globals.insert(
        "isNaN".into(),
        native(|args| {
            let n = args
                .first()
                .unwrap_or(&JsValue::Undefined)
                .to_number()?;
            Ok(JsValue::Boolean(global_is_nan(n)))
        }),
    );
    globals.insert(
        "isFinite".into(),
        native(|args| {
            let n = args
                .first()
                .unwrap_or(&JsValue::Undefined)
                .to_number()?;
            Ok(JsValue::Boolean(global_is_finite(n)))
        }),
    );
    globals.insert(
        "encodeURI".into(),
        native(|args| {
            let s = args
                .first()
                .unwrap_or(&JsValue::Undefined)
                .to_js_string()?;
            Ok(JsValue::String(global_encode_uri(&s)))
        }),
    );
    globals.insert(
        "decodeURI".into(),
        native(|args| {
            let s = args
                .first()
                .unwrap_or(&JsValue::Undefined)
                .to_js_string()?;
            Ok(JsValue::String(global_decode_uri(&s)?))
        }),
    );
    globals.insert(
        "encodeURIComponent".into(),
        native(|args| {
            let s = args
                .first()
                .unwrap_or(&JsValue::Undefined)
                .to_js_string()?;
            Ok(JsValue::String(global_encode_uri_component(&s)))
        }),
    );
    globals.insert(
        "decodeURIComponent".into(),
        native(|args| {
            let s = args
                .first()
                .unwrap_or(&JsValue::Undefined)
                .to_js_string()?;
            Ok(JsValue::String(global_decode_uri_component(&s)?))
        }),
    );
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
        assert!(matches!(globals.get("Symbol"), Some(JsValue::PlainObject(_))));
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
            assert!(matches!(map.get("keyFor"), Some(JsValue::NativeFunction(_))));
            assert!(matches!(map.get("__call__"), Some(JsValue::NativeFunction(_))));
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
        } else {
            panic!("Object should be a PlainObject");
        }
    }
}
