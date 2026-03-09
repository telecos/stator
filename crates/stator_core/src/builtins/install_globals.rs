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

use crate::builtins::date::{
    date_construct_components, date_construct_now, date_construct_value, date_get_date,
    date_get_day, date_get_full_year, date_get_hours, date_get_milliseconds, date_get_minutes,
    date_get_month, date_get_seconds, date_get_time, date_get_timezone_offset, date_get_utc_date,
    date_get_utc_day, date_get_utc_full_year, date_get_utc_hours, date_get_utc_milliseconds,
    date_get_utc_minutes, date_get_utc_month, date_get_utc_seconds, date_now, date_parse,
    date_set_date, date_set_full_year, date_set_hours, date_set_milliseconds, date_set_minutes,
    date_set_month, date_set_seconds, date_set_time, date_set_utc_date, date_set_utc_full_year,
    date_set_utc_hours, date_set_utc_milliseconds, date_set_utc_minutes, date_set_utc_month,
    date_set_utc_seconds, date_to_date_string, date_to_iso_string, date_to_json,
    date_to_locale_date_string, date_to_locale_string, date_to_locale_time_string, date_to_string,
    date_to_time_string, date_to_utc_string, date_utc, date_value_of,
};
use crate::builtins::error::{
    ErrorKind, JsError, error_capture_stack_trace, get_stack_trace_limit,
};
use crate::builtins::finalization_registry::{
    finalization_registry_drain, finalization_registry_new, finalization_registry_notify,
    finalization_registry_register, finalization_registry_register_plain,
    finalization_registry_sweep_plain, finalization_registry_unregister,
    finalization_registry_unregister_plain,
};
use crate::builtins::function::{
    function_apply, function_bind, function_call, function_constructor, function_has_instance,
    function_to_string,
};
use crate::builtins::global::{
    GLOBAL_INFINITY, GLOBAL_NAN, global_decode_uri, global_decode_uri_component, global_encode_uri,
    global_encode_uri_component, global_escape, global_eval, global_is_finite, global_is_nan,
    global_parse_float, global_parse_int, global_unescape,
};
use crate::builtins::intl::{
    collator_compare_js, date_time_format_js, date_time_format_to_parts_js, display_names_of,
    list_format_js, locale_base_name, locale_language, number_format_js, number_format_to_parts_js,
    plural_rules_select_js, relative_time_format_js, segmenter_segment,
};
use crate::builtins::iterator::{
    async_iterator_drop, async_iterator_every, async_iterator_filter, async_iterator_find,
    async_iterator_flat_map, async_iterator_for_each, async_iterator_from, async_iterator_map,
    async_iterator_reduce, async_iterator_some, async_iterator_take, async_iterator_to_array,
    iterator_drop, iterator_every, iterator_filter, iterator_find, iterator_flat_map,
    iterator_for_each, iterator_from, iterator_map, iterator_reduce, iterator_some, iterator_take,
    iterator_to_array,
};
use crate::builtins::map::{
    MapIteratorKind, map_clear, map_create_iterator, map_delete, map_entries, map_from_iterable,
    map_get, map_has, map_new, map_set, map_size,
};
use crate::builtins::math::{
    MATH_E, MATH_LN2, MATH_LN10, MATH_LOG2E, MATH_LOG10E, MATH_PI, MATH_SQRT1_2, MATH_SQRT2,
    math_abs, math_acos, math_acosh, math_asin, math_asinh, math_atan, math_atan2, math_atanh,
    math_cbrt, math_ceil, math_clz32, math_cos, math_cosh, math_exp, math_expm1, math_floor,
    math_fround, math_hypot, math_imul, math_log, math_log1p, math_log2, math_log10, math_max,
    math_min, math_pow, math_random, math_round, math_sign, math_sin, math_sinh, math_sqrt,
    math_tan, math_tanh, math_trunc,
};
use crate::builtins::proxy::{
    ProxyHandler, proxy_new, proxy_new_callable, proxy_revocable, proxy_revoke,
};
use crate::builtins::reflect::{
    reflect_define_property, reflect_delete_property, reflect_get,
    reflect_get_own_property_descriptor, reflect_get_prototype_of, reflect_has,
    reflect_is_extensible, reflect_own_keys, reflect_prevent_extensions, reflect_set,
    reflect_set_prototype_of,
};
use crate::builtins::regexp::regexp_construct;
use crate::builtins::set::{
    SetIteratorKind, set_add, set_clear, set_create_iterator, set_delete, set_difference,
    set_from_iterable, set_has, set_intersection, set_is_disjoint_from, set_is_subset_of,
    set_is_superset_of, set_new, set_size, set_symmetric_difference, set_union, set_values,
};
use crate::builtins::string::{
    string_anchor, string_at, string_big, string_blink, string_bold, string_char_at,
    string_char_code_at, string_code_point_at, string_concat, string_ends_with, string_fixed,
    string_fontcolor, string_fontsize, string_from_char_code, string_from_code_point,
    string_includes, string_index_of, string_is_well_formed, string_italics, string_iter,
    string_last_index_of, string_link, string_locale_compare, string_match, string_match_all,
    string_normalize, string_pad_end, string_pad_start, string_raw, string_repeat, string_replace,
    string_replace_all, string_search, string_slice, string_small, string_split,
    string_starts_with, string_strike, string_sub, string_substr, string_substring, string_sup,
    string_to_locale_lower_case, string_to_locale_upper_case, string_to_lower_case,
    string_to_upper_case, string_to_well_formed, string_trim, string_trim_end, string_trim_start,
};
use crate::builtins::symbol::{
    SYMBOL_ASYNC_DISPOSE, SYMBOL_ASYNC_ITERATOR, SYMBOL_DISPOSE, SYMBOL_HAS_INSTANCE,
    SYMBOL_IS_CONCAT_SPREADABLE, SYMBOL_ITERATOR, SYMBOL_MATCH, SYMBOL_MATCH_ALL, SYMBOL_REPLACE,
    SYMBOL_SEARCH, SYMBOL_SPECIES, SYMBOL_SPLIT, SYMBOL_TO_PRIMITIVE, SYMBOL_TO_STRING_TAG,
    SYMBOL_UNSCOPABLES, symbol_create, symbol_description, symbol_for, symbol_key_for,
};
use crate::builtins::typed_array::{
    JsArrayBuffer, TypedArrayKind, arraybuffer_is_view, arraybuffer_new, dataview_get_bigint64,
    dataview_get_biguint64, dataview_get_float32, dataview_get_float64, dataview_get_int8,
    dataview_get_int16, dataview_get_int32, dataview_get_uint8, dataview_get_uint16,
    dataview_get_uint32, dataview_new, dataview_set_bigint64, dataview_set_biguint64,
    dataview_set_float32, dataview_set_float64, dataview_set_int8, dataview_set_int16,
    dataview_set_int32, dataview_set_uint8, dataview_set_uint16, dataview_set_uint32,
    typed_array_at, typed_array_copy_within, typed_array_entries, typed_array_fill,
    typed_array_from_values, typed_array_get, typed_array_includes, typed_array_index_of,
    typed_array_join, typed_array_keys, typed_array_last_index_of, typed_array_new_from_buffer,
    typed_array_new_from_length, typed_array_reverse, typed_array_set_from, typed_array_slice,
    typed_array_sort, typed_array_subarray, typed_array_values,
};
use crate::builtins::weak_map::{
    weak_map_delete, weak_map_get, weak_map_has, weak_map_new, weak_map_set,
};
use crate::builtins::weak_ref::{weak_ref_deref, weak_ref_new, weak_ref_new_plain};
use crate::builtins::weak_set::{weak_set_add, weak_set_delete, weak_set_has, weak_set_new};
use crate::error::{StatorError, StatorResult};
use crate::objects::js_object::JsObject;
use crate::objects::map::PropertyAttributes;
use crate::objects::property_map::PropertyMap;
use crate::objects::value::JsValue;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Wrap a Rust closure as a `JsValue::NativeFunction`.
fn native(f: impl Fn(Vec<JsValue>) -> StatorResult<JsValue> + 'static) -> JsValue {
    JsValue::NativeFunction(Rc::new(f))
}

/// ECMAScript §23.1.3.1 step 5.b — check `@@isConcatSpreadable`.
///
/// Returns `true` when the value should be spread by `Array.prototype.concat`:
/// arrays are spreadable by default unless `@@isConcatSpreadable` is `false`;
/// other objects are spreadable only when `@@isConcatSpreadable` is `true`.
fn is_concat_spreadable(value: &JsValue) -> bool {
    match value {
        JsValue::PlainObject(map) => match map.borrow().get("@@isConcatSpreadable").cloned() {
            Some(v) => v.to_boolean(),
            None => false,
        },
        JsValue::Array(items) => {
            // If the array was wrapped in an object with @@isConcatSpreadable,
            // we can't see it here.  Bare arrays are always spreadable unless
            // they carry an internal property override — our representation
            // does not support per-value internal slots, so default to `true`.
            let _ = items;
            true
        }
        _ => false,
    }
}

/// If `value` is a RegExp `PlainObject` (has `__is_regexp__`), invoke the
/// given `__symbol_*__` method with the supplied arguments and return the
/// result.  Otherwise return `None` so the caller can fall through to the
/// plain-string implementation.
fn try_regexp_symbol(
    value: &JsValue,
    symbol_key: &str,
    args: Vec<JsValue>,
) -> Option<StatorResult<JsValue>> {
    if let JsValue::PlainObject(map) = value {
        let borrow = map.borrow();
        let is_re = matches!(borrow.get("__is_regexp__"), Some(JsValue::Boolean(true)));
        if is_re && let Some(JsValue::NativeFunction(f)) = borrow.get(symbol_key).cloned() {
            drop(borrow);
            return Some(f(args));
        }
    }
    None
}

/// Extract a [`JsSet`] from a Set-like `PlainObject` by calling its `values()`
/// method and collecting the resulting iterator.
fn extract_set_from_arg(arg: &JsValue) -> StatorResult<crate::builtins::set::JsSet> {
    use crate::builtins::iterator::iterator_next;
    if let JsValue::PlainObject(map) = arg {
        let borrow = map.borrow();
        if let Some(JsValue::NativeFunction(values_fn)) = borrow.get("values") {
            let iter = values_fn(vec![])?;
            drop(borrow);
            let mut items = Vec::new();
            loop {
                let rec = iterator_next(&iter)?;
                if rec.done {
                    break;
                }
                items.push(rec.value);
            }
            return Ok(set_from_iterable(items));
        }
    }
    Err(StatorError::TypeError(
        "argument is not a Set-like object".into(),
    ))
}

/// Build a full `Set` instance (PlainObject with prototype methods) from a
/// [`JsSet`].  Used by ES2025 Set composition methods that return new sets.
fn build_set_instance(s: crate::builtins::set::JsSet) -> StatorResult<JsValue> {
    let inner = Rc::new(RefCell::new(s));
    let mut obj = PropertyMap::new();
    obj.insert(
        "size".into(),
        JsValue::Smi(set_size(&inner.borrow()) as i32),
    );
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
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "keys".into(),
            native(move |_| Ok(set_create_iterator(&inner.borrow(), SetIteratorKind::Keys))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "values".into(),
            native(move |_| {
                Ok(set_create_iterator(
                    &inner.borrow(),
                    SetIteratorKind::Values,
                ))
            }),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "entries".into(),
            native(move |_| {
                Ok(set_create_iterator(
                    &inner.borrow(),
                    SetIteratorKind::Entries,
                ))
            }),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "@@iterator".into(),
            native(move |_| {
                Ok(set_create_iterator(
                    &inner.borrow(),
                    SetIteratorKind::Values,
                ))
            }),
        );
    }
    Ok(JsValue::PlainObject(Rc::new(RefCell::new(obj))))
}

/// Build a NativeFunction that constructs a `JsValue::Error` of the given `ErrorKind`.
///
/// Supports the ES2022 options parameter: `new Error(message, { cause })`.
fn make_error_constructor(kind: ErrorKind) -> JsValue {
    native(move |args| {
        let message = match args.first() {
            Some(JsValue::Undefined) | None => String::new(),
            Some(v) => v.to_js_string()?,
        };
        let mut err = JsError::new(kind, message);
        // ES2022: extract `cause` from the optional second options argument.
        err.cause = extract_cause(args.get(1));
        Ok(JsValue::Error(Rc::new(err)))
    })
}

/// Extract the `cause` value from an options argument, if present.
fn extract_cause(options: Option<&JsValue>) -> Option<JsValue> {
    if let Some(JsValue::PlainObject(map)) = options {
        map.borrow().get("cause").cloned()
    } else {
        None
    }
}

/// Build the `AggregateError` constructor.
///
/// Signature: `AggregateError(errors, message [, options])`.
///
/// The `errors` argument is consumed as an `Array`; the optional `options`
/// object may contain a `cause` property (ES2022).
fn make_aggregate_error_constructor() -> JsValue {
    native(|args| {
        // First arg: errors (iterable — we accept Array).
        let errors_val = args.first().unwrap_or(&JsValue::Undefined);
        let inner_errors: Vec<Rc<JsError>> = match errors_val {
            JsValue::Array(arr) => arr
                .borrow()
                .iter()
                .map(|v| match v {
                    JsValue::Error(e) => Rc::clone(e),
                    other => Rc::new(JsError::new(
                        ErrorKind::Error,
                        other.to_js_string().unwrap_or_default(),
                    )),
                })
                .collect(),
            _ => Vec::new(),
        };
        // Second arg: message.
        let message = match args.get(1) {
            Some(JsValue::Undefined) | None => String::new(),
            Some(v) => v.to_js_string()?,
        };
        let mut err = JsError::new_aggregate(inner_errors, message);
        // Third arg: optional options object with `cause`.
        err.cause = extract_cause(args.get(2));
        Ok(JsValue::Error(Rc::new(err)))
    })
}

/// Build the `SuppressedError` constructor (TC39 explicit resource management).
///
/// `SuppressedError(error, suppressed, message)` creates an Error with
/// `.error`, `.suppressed`, and `.message` properties.
fn make_suppressed_error_constructor() -> JsValue {
    native(|args| {
        let error_val = args.first().cloned().unwrap_or(JsValue::Undefined);
        let suppressed_val = args.get(1).cloned().unwrap_or(JsValue::Undefined);
        let message = match args.get(2) {
            Some(JsValue::Undefined) | None => String::new(),
            Some(v) => v.to_js_string()?,
        };
        let mut props = PropertyMap::new();
        props.insert("name".into(), JsValue::String("SuppressedError".into()));
        props.insert("message".into(), JsValue::String(message.into()));
        props.insert("error".into(), error_val);
        props.insert("suppressed".into(), suppressed_val);
        Ok(JsValue::PlainObject(Rc::new(RefCell::new(props))))
    })
}

/// Register all ECMAScript error constructors in the global environment.
/// Register all ECMAScript error constructors in the global environment.
///
/// The `Error` constructor additionally exposes the V8-compatible
/// `Error.captureStackTrace(target)` and `Error.stackTraceLimit` extensions.
fn install_error_constructors(globals: &mut HashMap<String, JsValue>) {
    // The `Error` constructor is a PlainObject so it can carry static methods.
    let mut error_props = PropertyMap::new();
    error_props.insert("__call__".into(), make_error_constructor(ErrorKind::Error));
    // V8 extension: Error.captureStackTrace(targetObject [, constructorOpt])
    error_props.insert(
        "captureStackTrace".into(),
        native(|args| {
            let target = args.first().cloned().unwrap_or(JsValue::Undefined);
            if let JsValue::Error(e) = target {
                // Rc::try_unwrap is unlikely to succeed for shared Rcs, so
                // clone-and-mutate.
                let mut cloned = (*e).clone();
                error_capture_stack_trace(&mut cloned, None);
                Ok(JsValue::Error(Rc::new(cloned)))
            } else {
                Ok(JsValue::Undefined)
            }
        }),
    );
    // V8 extension: Error.stackTraceLimit (getter/setter via property)
    error_props.insert(
        "stackTraceLimit".into(),
        JsValue::Smi(get_stack_trace_limit() as i32),
    );
    globals.insert(
        "Error".into(),
        JsValue::PlainObject(Rc::new(RefCell::new(error_props))),
    );

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
    globals.insert("AggregateError".into(), make_aggregate_error_constructor());
    globals.insert(
        "SuppressedError".into(),
        make_suppressed_error_constructor(),
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

/// Deep-clone a `JsValue`, recursing into objects and arrays.
fn structured_clone(val: &JsValue) -> JsValue {
    match val {
        // Primitives: value types or immutable — return as-is.
        JsValue::Undefined
        | JsValue::Null
        | JsValue::Boolean(_)
        | JsValue::Smi(_)
        | JsValue::HeapNumber(_)
        | JsValue::String(_)
        | JsValue::Symbol(_)
        | JsValue::BigInt(_) => val.clone(),

        // PlainObject: recursively clone every property.
        JsValue::PlainObject(map) => {
            let mut cloned = PropertyMap::new();
            for (k, v) in map.borrow().iter() {
                cloned.insert(k.clone(), structured_clone(v));
            }
            JsValue::PlainObject(Rc::new(RefCell::new(cloned)))
        }

        // Array: recursively clone every element.
        JsValue::Array(arr) => {
            let cloned: Vec<JsValue> = arr.borrow().iter().map(structured_clone).collect();
            JsValue::new_array(cloned)
        }

        // Error: clone the error.
        JsValue::Error(e) => JsValue::Error(Rc::new(JsError::clone(e))),

        // Best-effort shallow clone for remaining types.
        _ => val.clone(),
    }
}

// ── Math ─────────────────────────────────────────────────────────────────────

/// Build the `Math` namespace object with all constants and methods.
fn make_math() -> JsValue {
    let mut props = PropertyMap::new();

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
    let mut props = PropertyMap::new();

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
        JsonValue::Str(s) => JsValue::String(s.clone().into()),
        JsonValue::Array(arr) => {
            let items: Vec<JsValue> = arr.borrow().iter().map(json_value_to_js_value).collect();
            JsValue::new_array(items)
        }
        JsonValue::Object(entries) => {
            let mut map = PropertyMap::new();
            for (k, v) in entries.borrow().iter() {
                map.insert(k.clone(), json_value_to_js_value(v));
            }
            JsValue::PlainObject(Rc::new(RefCell::new(map)))
        }
    }
}

/// A boxed replacer function for `JSON.stringify`.
type JsonReplacerFn = Box<
    dyn Fn(
        &str,
        &crate::builtins::json::JsonValue,
    ) -> StatorResult<Option<crate::builtins::json::JsonValue>>,
>;

/// Build the `JSON` namespace object.
fn make_json() -> JsValue {
    use crate::builtins::json::{
        JsonReplacer, JsonSpace, JsonValue, js_value_to_json, json_parse, json_stringify_js_value,
    };

    let mut props = PropertyMap::new();

    props.insert(
        "parse".into(),
        native(|args| {
            let text = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let json_val = json_parse(&text, None)?;
            let js_val = json_value_to_js_value(&json_val);

            // §25.5.1 — apply the optional reviver function bottom-up.
            match args.get(1) {
                Some(JsValue::NativeFunction(reviver)) => apply_js_reviver(js_val, "", reviver),
                _ => Ok(js_val),
            }
        }),
    );
    props.insert(
        "stringify".into(),
        native(|args| {
            let val = args.first().unwrap_or(&JsValue::Undefined);

            // Build optional replacer function closure from the second argument.
            let repl_fn_closure: Option<JsonReplacerFn> = match args.get(1) {
                Some(JsValue::NativeFunction(f)) => {
                    let f = f.clone();
                    Some(Box::new(
                        move |key: &str, val: &JsonValue| -> StatorResult<Option<JsonValue>> {
                            let js_key = JsValue::String(key.to_string().into());
                            let js_val = json_value_to_js_value(val);
                            let result = f(vec![js_key, js_val])?;
                            match result {
                                JsValue::Undefined => Ok(None),
                                other => match js_value_to_json(&other)? {
                                    Some(jv) => Ok(Some(jv)),
                                    None => Ok(None),
                                },
                            }
                        },
                    ))
                }
                _ => None,
            };

            // Build optional replacer array from the second argument.
            let repl_strings: Vec<String> = match args.get(1) {
                Some(JsValue::Array(items)) => items
                    .borrow()
                    .iter()
                    .filter_map(|v| {
                        if let JsValue::String(s) = v {
                            Some(s.to_string())
                        } else {
                            None
                        }
                    })
                    .collect(),
                _ => Vec::new(),
            };

            // Assemble the JsonReplacer enum.
            let replacer: Option<JsonReplacer<'_>> = if let Some(ref f) = repl_fn_closure {
                Some(JsonReplacer::Function(f.as_ref()))
            } else if matches!(args.get(1), Some(JsValue::Array(_))) {
                Some(JsonReplacer::Array(repl_strings))
            } else {
                None
            };

            // Build optional space from the third argument.
            let space: Option<JsonSpace> = match args.get(2) {
                Some(JsValue::Smi(n)) => Some(JsonSpace::Count((*n).max(0) as u32)),
                Some(JsValue::HeapNumber(n)) => Some(JsonSpace::Count(n.clamp(0.0, 10.0) as u32)),
                Some(JsValue::String(s)) => Some(JsonSpace::Str(s.to_string())),
                _ => None,
            };

            match json_stringify_js_value(val, replacer.as_ref(), space.as_ref())? {
                Some(s) => Ok(JsValue::String(s.into())),
                None => Ok(JsValue::Undefined),
            }
        }),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

/// Walk a `JsValue` tree bottom-up, calling `reviver(key, value)` at each
/// node — the ECMAScript `InternalizeJSONProperty` algorithm (§25.5.1.1).
fn apply_js_reviver(
    value: JsValue,
    key: &str,
    reviver: &crate::objects::value::NativeFn,
) -> StatorResult<JsValue> {
    let value = match value {
        JsValue::PlainObject(ref map) => {
            let keys: Vec<String> = map.borrow().keys().cloned().collect();
            for k in keys {
                let child = map.borrow().get(&k).cloned().unwrap_or(JsValue::Undefined);
                let new_child = apply_js_reviver(child, &k, reviver)?;
                if matches!(new_child, JsValue::Undefined) {
                    map.borrow_mut().remove(&k);
                } else {
                    map.borrow_mut().insert(k, new_child);
                }
            }
            value
        }
        JsValue::Array(ref items) => {
            let mut new_items = Vec::with_capacity(items.borrow().len());
            for (i, item) in items.borrow().iter().enumerate() {
                let new_item = apply_js_reviver(item.clone(), &i.to_string(), reviver)?;
                new_items.push(new_item);
            }
            JsValue::new_array(new_items)
        }
        other => other,
    };
    reviver(vec![JsValue::String(key.to_string().into()), value])
}

// ── Date constructor ─────────────────────────────────────────────────────────

/// Build the `Date` constructor/namespace object.
///
/// The returned `PlainObject` has:
/// - `__call__`: the constructor (`new Date(...)` / `Date()`)
/// - `now`: `Date.now()`
/// - `parse`: `Date.parse(string)`
/// - `UTC`: `Date.UTC(year, month, ...)`
fn make_date() -> JsValue {
    let mut props = PropertyMap::new();

    // ── Constructor: new Date() / Date(value) / Date(y, m, d, ...) ──────
    props.insert(
        "__call__".into(),
        native(|args| {
            let timestamp = match args.len() {
                0 => date_construct_now(),
                1 => {
                    let arg = args.first().unwrap();
                    match arg {
                        JsValue::String(s) => date_parse(s),
                        _ => date_construct_value(arg.to_number()?),
                    }
                }
                _ => {
                    let year = args[0].to_number()?;
                    let month = args[1].to_number()?;
                    let date_val = args
                        .get(2)
                        .map(|v| v.to_number())
                        .transpose()?
                        .unwrap_or(1.0);
                    let hours = args
                        .get(3)
                        .map(|v| v.to_number())
                        .transpose()?
                        .unwrap_or(0.0);
                    let minutes = args
                        .get(4)
                        .map(|v| v.to_number())
                        .transpose()?
                        .unwrap_or(0.0);
                    let seconds = args
                        .get(5)
                        .map(|v| v.to_number())
                        .transpose()?
                        .unwrap_or(0.0);
                    let ms = args
                        .get(6)
                        .map(|v| v.to_number())
                        .transpose()?
                        .unwrap_or(0.0);
                    date_construct_components(year, month, date_val, hours, minutes, seconds, ms)
                }
            };
            Ok(make_date_instance(timestamp))
        }),
    );

    // ── Date.now() ──────────────────────────────────────────────────────
    props.insert("now".into(), native(|_| Ok(num(date_now()))));

    // ── Date.parse(string) ──────────────────────────────────────────────
    props.insert(
        "parse".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(num(date_parse(&s)))
        }),
    );

    // ── Date.UTC(year, month, ...) ──────────────────────────────────────
    props.insert(
        "UTC".into(),
        native(|args| {
            let year = arg_f64(&args, 0)?;
            let month = if args.len() > 1 {
                args[1].to_number()?
            } else {
                0.0
            };
            let date_val = if args.len() > 2 {
                args[2].to_number()?
            } else {
                1.0
            };
            let hours = if args.len() > 3 {
                args[3].to_number()?
            } else {
                0.0
            };
            let minutes = if args.len() > 4 {
                args[4].to_number()?
            } else {
                0.0
            };
            let seconds = if args.len() > 5 {
                args[5].to_number()?
            } else {
                0.0
            };
            let ms = if args.len() > 6 {
                args[6].to_number()?
            } else {
                0.0
            };
            Ok(num(date_utc(
                year, month, date_val, hours, minutes, seconds, ms,
            )))
        }),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

/// Create a Date instance object with all prototype methods.
///
/// The returned `PlainObject` holds a shared `Rc<RefCell<f64>>` timestamp
/// that all getter/setter methods close over.
fn make_date_instance(t: f64) -> JsValue {
    let inner = Rc::new(RefCell::new(t));
    let mut obj = PropertyMap::new();

    // §20.1.3.6 — identify as Date for Object.prototype.toString.
    obj.insert("@@toStringTag".into(), JsValue::String("Date".into()));

    // ── getTime / valueOf ────────────────────────────────────────────────
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "getTime".into(),
            native(move |_| Ok(num(date_get_time(*inner.borrow())))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "valueOf".into(),
            native(move |_| Ok(num(date_value_of(*inner.borrow())))),
        );
    }

    // ── Local getters ───────────────────────────────────────────────────
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "getFullYear".into(),
            native(move |_| Ok(num(date_get_full_year(*inner.borrow())))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "getMonth".into(),
            native(move |_| Ok(num(date_get_month(*inner.borrow())))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "getDate".into(),
            native(move |_| Ok(num(date_get_date(*inner.borrow())))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "getDay".into(),
            native(move |_| Ok(num(date_get_day(*inner.borrow())))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "getHours".into(),
            native(move |_| Ok(num(date_get_hours(*inner.borrow())))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "getMinutes".into(),
            native(move |_| Ok(num(date_get_minutes(*inner.borrow())))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "getSeconds".into(),
            native(move |_| Ok(num(date_get_seconds(*inner.borrow())))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "getMilliseconds".into(),
            native(move |_| Ok(num(date_get_milliseconds(*inner.borrow())))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "getTimezoneOffset".into(),
            native(move |_| Ok(num(date_get_timezone_offset(*inner.borrow())))),
        );
    }

    // ── UTC getters ─────────────────────────────────────────────────────
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "getUTCFullYear".into(),
            native(move |_| Ok(num(date_get_utc_full_year(*inner.borrow())))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "getUTCMonth".into(),
            native(move |_| Ok(num(date_get_utc_month(*inner.borrow())))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "getUTCDate".into(),
            native(move |_| Ok(num(date_get_utc_date(*inner.borrow())))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "getUTCDay".into(),
            native(move |_| Ok(num(date_get_utc_day(*inner.borrow())))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "getUTCHours".into(),
            native(move |_| Ok(num(date_get_utc_hours(*inner.borrow())))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "getUTCMinutes".into(),
            native(move |_| Ok(num(date_get_utc_minutes(*inner.borrow())))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "getUTCSeconds".into(),
            native(move |_| Ok(num(date_get_utc_seconds(*inner.borrow())))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "getUTCMilliseconds".into(),
            native(move |_| Ok(num(date_get_utc_milliseconds(*inner.borrow())))),
        );
    }

    // ── Local setters ───────────────────────────────────────────────────
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "setTime".into(),
            native(move |args| {
                let v = arg_f64(&args, 0)?;
                let result = date_set_time(v);
                *inner.borrow_mut() = result;
                Ok(num(result))
            }),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "setMilliseconds".into(),
            native(move |args| {
                let ms = arg_f64(&args, 0)?;
                let result = date_set_milliseconds(*inner.borrow(), ms);
                *inner.borrow_mut() = result;
                Ok(num(result))
            }),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "setSeconds".into(),
            native(move |args| {
                let sec = arg_f64(&args, 0)?;
                let ms = if args.len() > 1 {
                    Some(args[1].to_number()?)
                } else {
                    None
                };
                let result = date_set_seconds(*inner.borrow(), sec, ms);
                *inner.borrow_mut() = result;
                Ok(num(result))
            }),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "setMinutes".into(),
            native(move |args| {
                let min = arg_f64(&args, 0)?;
                let sec = if args.len() > 1 {
                    Some(args[1].to_number()?)
                } else {
                    None
                };
                let ms = if args.len() > 2 {
                    Some(args[2].to_number()?)
                } else {
                    None
                };
                let result = date_set_minutes(*inner.borrow(), min, sec, ms);
                *inner.borrow_mut() = result;
                Ok(num(result))
            }),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "setHours".into(),
            native(move |args| {
                let hour = arg_f64(&args, 0)?;
                let min = if args.len() > 1 {
                    Some(args[1].to_number()?)
                } else {
                    None
                };
                let sec = if args.len() > 2 {
                    Some(args[2].to_number()?)
                } else {
                    None
                };
                let ms = if args.len() > 3 {
                    Some(args[3].to_number()?)
                } else {
                    None
                };
                let result = date_set_hours(*inner.borrow(), hour, min, sec, ms);
                *inner.borrow_mut() = result;
                Ok(num(result))
            }),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "setDate".into(),
            native(move |args| {
                let date_val = arg_f64(&args, 0)?;
                let result = date_set_date(*inner.borrow(), date_val);
                *inner.borrow_mut() = result;
                Ok(num(result))
            }),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "setMonth".into(),
            native(move |args| {
                let month = arg_f64(&args, 0)?;
                let date_val = if args.len() > 1 {
                    Some(args[1].to_number()?)
                } else {
                    None
                };
                let result = date_set_month(*inner.borrow(), month, date_val);
                *inner.borrow_mut() = result;
                Ok(num(result))
            }),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "setFullYear".into(),
            native(move |args| {
                let year = arg_f64(&args, 0)?;
                let month = if args.len() > 1 {
                    Some(args[1].to_number()?)
                } else {
                    None
                };
                let date_val = if args.len() > 2 {
                    Some(args[2].to_number()?)
                } else {
                    None
                };
                let result = date_set_full_year(*inner.borrow(), year, month, date_val);
                *inner.borrow_mut() = result;
                Ok(num(result))
            }),
        );
    }

    // ── UTC setters ─────────────────────────────────────────────────────
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "setUTCMilliseconds".into(),
            native(move |args| {
                let ms = arg_f64(&args, 0)?;
                let result = date_set_utc_milliseconds(*inner.borrow(), ms);
                *inner.borrow_mut() = result;
                Ok(num(result))
            }),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "setUTCSeconds".into(),
            native(move |args| {
                let sec = arg_f64(&args, 0)?;
                let ms = if args.len() > 1 {
                    Some(args[1].to_number()?)
                } else {
                    None
                };
                let result = date_set_utc_seconds(*inner.borrow(), sec, ms);
                *inner.borrow_mut() = result;
                Ok(num(result))
            }),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "setUTCMinutes".into(),
            native(move |args| {
                let min = arg_f64(&args, 0)?;
                let sec = if args.len() > 1 {
                    Some(args[1].to_number()?)
                } else {
                    None
                };
                let ms = if args.len() > 2 {
                    Some(args[2].to_number()?)
                } else {
                    None
                };
                let result = date_set_utc_minutes(*inner.borrow(), min, sec, ms);
                *inner.borrow_mut() = result;
                Ok(num(result))
            }),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "setUTCHours".into(),
            native(move |args| {
                let hour = arg_f64(&args, 0)?;
                let min = if args.len() > 1 {
                    Some(args[1].to_number()?)
                } else {
                    None
                };
                let sec = if args.len() > 2 {
                    Some(args[2].to_number()?)
                } else {
                    None
                };
                let ms = if args.len() > 3 {
                    Some(args[3].to_number()?)
                } else {
                    None
                };
                let result = date_set_utc_hours(*inner.borrow(), hour, min, sec, ms);
                *inner.borrow_mut() = result;
                Ok(num(result))
            }),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "setUTCDate".into(),
            native(move |args| {
                let date_val = arg_f64(&args, 0)?;
                let result = date_set_utc_date(*inner.borrow(), date_val);
                *inner.borrow_mut() = result;
                Ok(num(result))
            }),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "setUTCMonth".into(),
            native(move |args| {
                let month = arg_f64(&args, 0)?;
                let date_val = if args.len() > 1 {
                    Some(args[1].to_number()?)
                } else {
                    None
                };
                let result = date_set_utc_month(*inner.borrow(), month, date_val);
                *inner.borrow_mut() = result;
                Ok(num(result))
            }),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "setUTCFullYear".into(),
            native(move |args| {
                let year = arg_f64(&args, 0)?;
                let month = if args.len() > 1 {
                    Some(args[1].to_number()?)
                } else {
                    None
                };
                let date_val = if args.len() > 2 {
                    Some(args[2].to_number()?)
                } else {
                    None
                };
                let result = date_set_utc_full_year(*inner.borrow(), year, month, date_val);
                *inner.borrow_mut() = result;
                Ok(num(result))
            }),
        );
    }

    // ── String conversion methods ───────────────────────────────────────
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "toString".into(),
            native(move |_| Ok(JsValue::String(date_to_string(*inner.borrow()).into()))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "toDateString".into(),
            native(move |_| Ok(JsValue::String(date_to_date_string(*inner.borrow()).into()))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "toTimeString".into(),
            native(move |_| Ok(JsValue::String(date_to_time_string(*inner.borrow()).into()))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "toISOString".into(),
            native(move |_| Ok(JsValue::String(date_to_iso_string(*inner.borrow())?.into()))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "toUTCString".into(),
            native(move |_| Ok(JsValue::String(date_to_utc_string(*inner.borrow()).into()))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "toGMTString".into(),
            native(move |_| Ok(JsValue::String(date_to_utc_string(*inner.borrow()).into()))),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "toJSON".into(),
            native(move |_| match date_to_json(*inner.borrow()) {
                Some(s) => Ok(JsValue::String(s.into())),
                None => Ok(JsValue::Null),
            }),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "toLocaleDateString".into(),
            native(move |_| {
                Ok(JsValue::String(
                    date_to_locale_date_string(*inner.borrow()).into(),
                ))
            }),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "toLocaleString".into(),
            native(move |_| {
                Ok(JsValue::String(
                    date_to_locale_string(*inner.borrow()).into(),
                ))
            }),
        );
    }
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "toLocaleTimeString".into(),
            native(move |_| {
                Ok(JsValue::String(
                    date_to_locale_time_string(*inner.borrow()).into(),
                ))
            }),
        );
    }

    JsValue::PlainObject(Rc::new(RefCell::new(obj)))
}

// ── Number constructor ───────────────────────────────────────────────────────

/// Build the `Number` constructor/namespace object.
fn make_number() -> JsValue {
    let mut props = PropertyMap::new();

    // Number(value) — type conversion when called as a function
    props.insert(
        "__call__".into(),
        native(|args| {
            let val = args.first().unwrap_or(&JsValue::Undefined);
            if matches!(val, JsValue::Undefined) && args.is_empty() {
                return Ok(JsValue::Smi(0));
            }
            Ok(num(val.to_number()?))
        }),
    );

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
    let mut props = PropertyMap::new();

    props.insert(
        "keys".into(),
        native(|args| {
            let val = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::PlainObject(map) = val {
                let keys: Vec<JsValue> = map
                    .borrow()
                    .keys()
                    .map(|k| JsValue::String(k.clone().into()))
                    .collect();
                Ok(JsValue::new_array(keys))
            } else {
                Ok(JsValue::new_array(vec![]))
            }
        }),
    );
    props.insert(
        "values".into(),
        native(|args| {
            let val = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::PlainObject(map) = val {
                let values: Vec<JsValue> = map.borrow().iter().map(|(_, v)| v.clone()).collect();
                Ok(JsValue::new_array(values))
            } else {
                Ok(JsValue::new_array(vec![]))
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
                        JsValue::new_array(vec![JsValue::String(k.clone().into()), v.clone()])
                    })
                    .collect();
                Ok(JsValue::new_array(entries))
            } else {
                Ok(JsValue::new_array(vec![]))
            }
        }),
    );

    // ── Object.is(x, y) — SameValue comparison (ECMAScript §19.1.2.10) ──
    props.insert(
        "is".into(),
        native(|args| {
            let x = args.first().unwrap_or(&JsValue::Undefined);
            let y = args.get(1).unwrap_or(&JsValue::Undefined);
            Ok(JsValue::Boolean(x.same_value(y)))
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
                    // Check for accessor descriptor (get/set)
                    let has_get = desc.contains_key("get");
                    let has_set = desc.contains_key("set");
                    if has_get || has_set {
                        // Accessor property: store getter/setter using __get/__set convention.
                        if let Some(getter) = desc.get("get").cloned() {
                            let getter_key = format!("__get_{key}__");
                            map.borrow_mut().insert(getter_key, getter);
                        }
                        if let Some(setter) = desc.get("set").cloned() {
                            let setter_key = format!("__set_{key}__");
                            map.borrow_mut().insert(setter_key, setter);
                        }
                    } else {
                        // Data descriptor: extract value.
                        let value = desc.get("value").cloned().unwrap_or(JsValue::Undefined);
                        map.borrow_mut().insert(key.clone(), value);
                    }
                    // Apply writable attribute
                    if let Some(JsValue::Boolean(false)) = desc.get("writable") {
                        map.borrow_mut().set_writable(&key, false);
                    }
                    // Apply enumerable attribute
                    if let Some(JsValue::Boolean(false)) = desc.get("enumerable") {
                        map.borrow_mut().set_enumerable(&key, false);
                    }
                    // Apply configurable attribute
                    if let Some(JsValue::Boolean(false)) = desc.get("configurable") {
                        map.borrow_mut().set_configurable(&key, false);
                    }
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
                // Check for accessor property first
                let getter_key = format!("__get_{key}__");
                let setter_key = format!("__set_{key}__");
                let has_getter = borrowed.contains_key(&getter_key);
                let has_setter = borrowed.contains_key(&setter_key);
                if has_getter || has_setter {
                    let mut desc = PropertyMap::new();
                    desc.insert(
                        "get".into(),
                        borrowed
                            .get(&getter_key)
                            .cloned()
                            .unwrap_or(JsValue::Undefined),
                    );
                    desc.insert(
                        "set".into(),
                        borrowed
                            .get(&setter_key)
                            .cloned()
                            .unwrap_or(JsValue::Undefined),
                    );
                    desc.insert("enumerable".into(), JsValue::Boolean(true));
                    desc.insert("configurable".into(), JsValue::Boolean(true));
                    Ok(JsValue::PlainObject(Rc::new(RefCell::new(desc))))
                } else if let Some(value) = borrowed.get(&key) {
                    let mut desc = PropertyMap::new();
                    desc.insert("value".into(), value.clone());
                    desc.insert(
                        "writable".into(),
                        JsValue::Boolean(borrowed.is_writable(&key)),
                    );
                    desc.insert(
                        "enumerable".into(),
                        JsValue::Boolean(borrowed.is_enumerable(&key)),
                    );
                    desc.insert(
                        "configurable".into(),
                        JsValue::Boolean(borrowed.is_configurable(&key)),
                    );
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
                    .map(|k| JsValue::String(k.clone().into()))
                    .collect();
                Ok(JsValue::new_array(keys))
            } else {
                Ok(JsValue::new_array(vec![]))
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
    props.insert(
        "freeze".into(),
        native(|args| {
            let obj = args.first().unwrap_or(&JsValue::Undefined).clone();
            if let JsValue::PlainObject(map) = &obj {
                let keys: Vec<String> = map.borrow().keys().cloned().collect();
                let mut pm = map.borrow_mut();
                for key in &keys {
                    pm.set_writable(key, false);
                    pm.set_configurable(key, false);
                }
            }
            Ok(obj)
        }),
    );

    // ── Object.seal(obj) ─────────────────────────────────────────────────
    props.insert(
        "seal".into(),
        native(|args| {
            let obj = args.first().unwrap_or(&JsValue::Undefined).clone();
            if let JsValue::PlainObject(map) = &obj {
                let keys: Vec<String> = map.borrow().keys().cloned().collect();
                let mut pm = map.borrow_mut();
                for key in &keys {
                    pm.set_configurable(key, false);
                }
            }
            Ok(obj)
        }),
    );

    // ── Object.isFrozen(obj) ─────────────────────────────────────────────
    props.insert(
        "isFrozen".into(),
        native(|args| {
            let obj = args.first().unwrap_or(&JsValue::Undefined);
            match obj {
                JsValue::PlainObject(map) => {
                    let pm = map.borrow();
                    let frozen = pm
                        .keys()
                        .all(|k| !pm.is_writable(k) && !pm.is_configurable(k));
                    Ok(JsValue::Boolean(frozen))
                }
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
                JsValue::PlainObject(map) => {
                    let pm = map.borrow();
                    let sealed = pm.keys().all(|k| !pm.is_configurable(k));
                    Ok(JsValue::Boolean(sealed))
                }
                _ => Ok(JsValue::Boolean(true)),
            }
        }),
    );

    // ── Object.create(proto) ─────────────────────────────────────────────
    props.insert(
        "create".into(),
        native(|args| {
            let proto = args.first().unwrap_or(&JsValue::Undefined);
            let mut obj = PropertyMap::new();
            match proto {
                JsValue::Null | JsValue::Undefined => {}
                _ => {
                    obj.insert("__proto__".to_string(), proto.clone());
                }
            }
            Ok(JsValue::PlainObject(Rc::new(RefCell::new(obj))))
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
            let mut result = PropertyMap::new();

            if let JsValue::Array(arr) = iterable {
                for entry in arr.borrow().iter() {
                    if let JsValue::Array(pair) = entry
                        && pair.borrow().len() >= 2
                    {
                        let key = pair.borrow()[0].to_js_string()?;
                        result.insert(key, pair.borrow()[1].clone());
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

    // ── Object.groupBy(items, callbackFn) ────────────────────────────────
    props.insert(
        "groupBy".into(),
        native(|args| {
            let items = args.first().unwrap_or(&JsValue::Undefined).clone();
            let cb = args.get(1).cloned().unwrap_or(JsValue::Undefined);
            let arr = match &items {
                JsValue::Array(a) => a.clone(),
                _ => {
                    return Err(StatorError::TypeError(
                        "Object.groupBy: first argument must be iterable".into(),
                    ));
                }
            };
            let result = PropertyMap::new();
            let result_rc = Rc::new(RefCell::new(result));
            for (i, item) in arr.borrow().iter().enumerate() {
                let key = if let JsValue::NativeFunction(f) = &cb {
                    f(vec![item.clone(), JsValue::Smi(i as i32)])?
                } else {
                    JsValue::Undefined
                };
                let group_key = key.to_js_string()?;
                let mut borrow = result_rc.borrow_mut();
                if let Some(JsValue::Array(existing)) = borrow.get(&group_key).cloned() {
                    let mut v = existing.borrow().clone();
                    v.push(item.clone());
                    borrow.insert(group_key, JsValue::new_array(v));
                } else {
                    borrow.insert(group_key, JsValue::new_array(vec![item.clone()]));
                }
            }
            Ok(JsValue::PlainObject(result_rc))
        }),
    );

    // ── Object.getPrototypeOf(obj) ───────────────────────────────────────
    props.insert(
        "getPrototypeOf".into(),
        native(|args| {
            let obj = args.first().unwrap_or(&JsValue::Undefined);
            match obj {
                JsValue::PlainObject(map) => {
                    let borrow = map.borrow();
                    match borrow.get("__proto__") {
                        Some(proto) => Ok(proto.clone()),
                        None => Ok(JsValue::Null),
                    }
                }
                _ => Ok(JsValue::Null),
            }
        }),
    );

    // ── Object.setPrototypeOf(obj, proto) ────────────────────────────────
    props.insert(
        "setPrototypeOf".into(),
        native(|args| {
            let obj = args.first().unwrap_or(&JsValue::Undefined).clone();
            let proto = args.get(1).unwrap_or(&JsValue::Undefined).clone();
            if let JsValue::PlainObject(map) = &obj {
                // Cycle detection: walk the new proto chain and check for obj.
                if let JsValue::PlainObject(_) = &proto {
                    let mut current = proto.clone();
                    for _ in 0..256 {
                        if let JsValue::PlainObject(cur_map) = &current {
                            if Rc::ptr_eq(cur_map, map) {
                                return Err(StatorError::TypeError(
                                    "Cyclic __proto__ value".to_string(),
                                ));
                            }
                            let next = cur_map.borrow().get("__proto__").cloned();
                            match next {
                                Some(v) => current = v,
                                None => break,
                            }
                        } else {
                            break;
                        }
                    }
                }
                match &proto {
                    JsValue::Null => {
                        map.borrow_mut().remove("__proto__");
                    }
                    _ => {
                        map.borrow_mut().insert("__proto__".to_string(), proto);
                    }
                }
            }
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
                let mut result = PropertyMap::new();
                for (key, value) in map.borrow().iter() {
                    let mut desc = PropertyMap::new();
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
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(
                    PropertyMap::new(),
                ))))
            }
        }),
    );

    // ── Object.getOwnPropertySymbols(obj) ────────────────────────────────
    props.insert(
        "getOwnPropertySymbols".into(),
        native(|_args| {
            // PlainObject has no symbol-keyed properties.
            Ok(JsValue::new_array(vec![]))
        }),
    );

    // ── Object.prototype ─────────────────────────────────────────────────
    let mut obj_proto = PropertyMap::new();

    // Object.prototype.hasOwnProperty(key)
    obj_proto.insert(
        "hasOwnProperty".into(),
        native(|args| {
            let this = args.first().unwrap_or(&JsValue::Undefined);
            let key = args.get(1).unwrap_or(&JsValue::Undefined);
            let prop = key.to_js_string()?;
            match this {
                JsValue::PlainObject(map) => {
                    let has = map.borrow().contains_key(&prop) && prop != "__proto__";
                    Ok(JsValue::Boolean(has))
                }
                _ => Ok(JsValue::Boolean(false)),
            }
        }),
    );

    // Object.prototype.isPrototypeOf(obj)
    obj_proto.insert(
        "isPrototypeOf".into(),
        native(|args| {
            let this = args.first().unwrap_or(&JsValue::Undefined);
            let target = args.get(1).unwrap_or(&JsValue::Undefined);
            if let (JsValue::PlainObject(this_map), JsValue::PlainObject(_)) = (this, target) {
                let mut current = target.clone();
                for _ in 0..256 {
                    if let JsValue::PlainObject(cur_map) = &current {
                        let next = cur_map.borrow().get("__proto__").cloned();
                        match next {
                            Some(JsValue::PlainObject(ref p)) if Rc::ptr_eq(p, this_map) => {
                                return Ok(JsValue::Boolean(true));
                            }
                            Some(v) => current = v,
                            None => break,
                        }
                    } else {
                        break;
                    }
                }
            }
            Ok(JsValue::Boolean(false))
        }),
    );

    // Annex B §B.2.2.1 — Object.prototype.__proto__ (terminal null)
    obj_proto.insert("__proto__".into(), JsValue::Null);

    // Object.prototype.toString()
    // ECMAScript §20.1.3.6 — returns "[object X]" classification.
    obj_proto.insert(
        "toString".into(),
        native(|args| {
            // When called via .call(value), args[0] is the value.
            if let Some(value) = args.first() {
                return Ok(JsValue::String(value.obj_to_string_tag().into()));
            }
            Ok(JsValue::String("[object Object]".to_string().into()))
        }),
    );

    props.insert(
        "prototype".into(),
        JsValue::PlainObject(Rc::new(RefCell::new(obj_proto))),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

// ── Array constructor ────────────────────────────────────────────────────────

/// Build the `Array` constructor/namespace object.
///
/// The returned `PlainObject` carries:
/// - `isArray` — `Array.isArray(value)`.
/// - `from` — `Array.from(iterable)`.
/// - `of` — `Array.of(...items)`.
/// - `prototype` — an object with all `Array.prototype.*` methods.
fn make_array() -> JsValue {
    let mut props = PropertyMap::new();

    // ── Static methods ──────────────────────────────────────────────────

    // Array.isArray(value)
    props.insert(
        "isArray".into(),
        native(|args| {
            let val = args.first().unwrap_or(&JsValue::Undefined);
            Ok(JsValue::Boolean(matches!(val, JsValue::Array(_))))
        }),
    );

    // Array.from(iterable)
    props.insert(
        "from".into(),
        native(|args| {
            let iterable = args.first().unwrap_or(&JsValue::Undefined);
            let items: Vec<JsValue> = match iterable {
                JsValue::Array(arr) => arr.borrow().clone(),
                JsValue::String(s) => s
                    .chars()
                    .map(|c| JsValue::String(c.to_string().into()))
                    .collect(),
                _ => Vec::new(),
            };
            Ok(JsValue::new_array(items))
        }),
    );

    // Array.of(...items)
    props.insert("of".into(), native(|args| Ok(JsValue::new_array(args))));

    // ── Prototype methods ───────────────────────────────────────────────
    //
    // Each method receives `(this_array, ...args)` where the first argument is
    // the array instance (`this`), and the remaining arguments are the method's
    // parameters.  The interpreter rewrites `arr.borrow_mut().push(x)` into
    // `Array.prototype.push(arr, x)` at the bytecode level.

    let mut proto = PropertyMap::new();

    // push(...items)
    proto.insert(
        "push".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let mut vec = items.borrow().clone();
                vec.extend_from_slice(&args[1..]);
                Ok(JsValue::Smi(vec.len() as i32))
            } else {
                Ok(JsValue::Undefined)
            }
        }),
    );

    // pop()
    proto.insert(
        "pop".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                Ok(items.borrow().last().cloned().unwrap_or(JsValue::Undefined))
            } else {
                Ok(JsValue::Undefined)
            }
        }),
    );

    // shift()
    proto.insert(
        "shift".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                Ok(items
                    .borrow()
                    .first()
                    .cloned()
                    .unwrap_or(JsValue::Undefined))
            } else {
                Ok(JsValue::Undefined)
            }
        }),
    );

    // unshift(...items)
    proto.insert(
        "unshift".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let new_len = items.borrow().len() + args.len() - 1;
                Ok(JsValue::Smi(new_len as i32))
            } else {
                Ok(JsValue::Undefined)
            }
        }),
    );

    // indexOf(searchElement, fromIndex?)
    proto.insert(
        "indexOf".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let search = args.get(1).unwrap_or(&JsValue::Undefined);
                let from = args
                    .get(2)
                    .unwrap_or(&JsValue::Smi(0))
                    .to_number()
                    .unwrap_or(0.0) as i64;
                let len = items.borrow().len() as i64;
                let start = if from < 0 { (len + from).max(0) } else { from } as usize;
                for (i, v) in items.borrow().iter().enumerate().skip(start) {
                    if v == search {
                        return Ok(JsValue::Smi(i as i32));
                    }
                }
                Ok(JsValue::Smi(-1))
            } else {
                Ok(JsValue::Smi(-1))
            }
        }),
    );

    // lastIndexOf(searchElement, fromIndex?)
    proto.insert(
        "lastIndexOf".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let search = args.get(1).unwrap_or(&JsValue::Undefined);
                let len = items.borrow().len() as i64;
                let from = args
                    .get(2)
                    .map(|v| v.to_number().unwrap_or((len - 1) as f64) as i64)
                    .unwrap_or(len - 1);
                let start = if from < 0 {
                    (len + from).max(0) as usize
                } else {
                    from.min(len - 1) as usize
                };
                for i in (0..=start).rev() {
                    if items.borrow().get(i) == Some(search) {
                        return Ok(JsValue::Smi(i as i32));
                    }
                }
                Ok(JsValue::Smi(-1))
            } else {
                Ok(JsValue::Smi(-1))
            }
        }),
    );

    // includes(searchElement, fromIndex?)
    proto.insert(
        "includes".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let search = args.get(1).unwrap_or(&JsValue::Undefined);
                let from = args
                    .get(2)
                    .unwrap_or(&JsValue::Smi(0))
                    .to_number()
                    .unwrap_or(0.0) as i64;
                let len = items.borrow().len() as i64;
                let start = if from < 0 { (len + from).max(0) } else { from } as usize;
                for v in items.borrow().iter().skip(start) {
                    if v == search {
                        return Ok(JsValue::Boolean(true));
                    }
                }
                Ok(JsValue::Boolean(false))
            } else {
                Ok(JsValue::Boolean(false))
            }
        }),
    );

    // join(separator?)
    proto.insert(
        "join".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let sep = match args.get(1) {
                    Some(JsValue::Undefined) | None => ",".to_string(),
                    Some(v) => v.to_js_string()?,
                };
                let parts: Vec<String> = items
                    .borrow()
                    .iter()
                    .map(|v| match v {
                        JsValue::Undefined | JsValue::Null => Ok(String::new()),
                        other => other.to_js_string(),
                    })
                    .collect::<StatorResult<_>>()?;
                Ok(JsValue::String(parts.join(&sep).into()))
            } else {
                Ok(JsValue::String(String::new().into()))
            }
        }),
    );

    // concat(...arrays) — §23.1.3.1, respects @@isConcatSpreadable
    proto.insert(
        "concat".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            let mut result: Vec<JsValue> = if let JsValue::Array(items) = arr {
                items.borrow().clone()
            } else {
                Vec::new()
            };
            for other in args.iter().skip(1) {
                if is_concat_spreadable(other) {
                    match other {
                        JsValue::Array(items) => {
                            result.extend(items.borrow().iter().cloned());
                        }
                        JsValue::PlainObject(map) => {
                            let borrow = map.borrow();
                            let len = borrow
                                .get("length")
                                .and_then(|v| v.to_number().ok())
                                .unwrap_or(0.0) as usize;
                            for i in 0..len {
                                if let Some(v) = borrow.get(&i.to_string()) {
                                    result.push(v.clone());
                                } else {
                                    result.push(JsValue::Undefined);
                                }
                            }
                        }
                        _ => result.push(other.clone()),
                    }
                } else {
                    result.push(other.clone());
                }
            }
            Ok(JsValue::new_array(result))
        }),
    );

    // slice(start?, end?)
    proto.insert(
        "slice".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let len = items.borrow().len() as i64;
                let start = args
                    .get(1)
                    .unwrap_or(&JsValue::Smi(0))
                    .to_number()
                    .unwrap_or(0.0) as i64;
                let end = args
                    .get(2)
                    .map(|v| v.to_number().unwrap_or(len as f64) as i64)
                    .unwrap_or(len);
                let s = if start < 0 {
                    (len + start).max(0)
                } else {
                    start.min(len)
                } as usize;
                let e = if end < 0 {
                    (len + end).max(0)
                } else {
                    end.min(len)
                } as usize;
                Ok(JsValue::new_array(items.borrow()[s..e].to_vec()))
            } else {
                Ok(JsValue::new_array(Vec::new()))
            }
        }),
    );

    // reverse()
    proto.insert(
        "reverse".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let mut v = items.borrow().clone();
                v.reverse();
                Ok(JsValue::new_array(v))
            } else {
                Ok(JsValue::Undefined)
            }
        }),
    );

    // sort(compareFn?)
    proto.insert(
        "sort".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let mut v = items.borrow().clone();
                let cmp_fn = args.get(1).cloned();
                if let Some(JsValue::NativeFunction(cmp)) = cmp_fn {
                    v.sort_by(|a, b| {
                        let result = cmp(vec![a.clone(), b.clone()]).unwrap_or(JsValue::Smi(0));
                        let n = match result {
                            JsValue::Smi(n) => n as f64,
                            JsValue::HeapNumber(n) => n,
                            _ => 0.0,
                        };
                        n.partial_cmp(&0.0).unwrap_or(std::cmp::Ordering::Equal)
                    });
                } else {
                    v.sort_by(|a, b| {
                        let sa = a.to_js_string().unwrap_or_default();
                        let sb = b.to_js_string().unwrap_or_default();
                        sa.cmp(&sb)
                    });
                }
                Ok(JsValue::new_array(v))
            } else {
                Ok(JsValue::Undefined)
            }
        }),
    );

    // fill(value, start?, end?)
    proto.insert(
        "fill".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let value = args.get(1).cloned().unwrap_or(JsValue::Undefined);
                let len = items.borrow().len() as i64;
                let start = args
                    .get(2)
                    .unwrap_or(&JsValue::Smi(0))
                    .to_number()
                    .unwrap_or(0.0) as i64;
                let end = args
                    .get(3)
                    .map(|v| v.to_number().unwrap_or(len as f64) as i64)
                    .unwrap_or(len);
                let s = if start < 0 {
                    (len + start).max(0)
                } else {
                    start.min(len)
                } as usize;
                let e = if end < 0 {
                    (len + end).max(0)
                } else {
                    end.min(len)
                } as usize;
                let mut v = items.borrow().clone();
                for item in v.iter_mut().take(e).skip(s) {
                    *item = value.clone();
                }
                Ok(JsValue::new_array(v))
            } else {
                Ok(JsValue::Undefined)
            }
        }),
    );

    // at(index)
    proto.insert(
        "at".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let index = args
                    .get(1)
                    .unwrap_or(&JsValue::Smi(0))
                    .to_number()
                    .unwrap_or(0.0) as i64;
                let len = items.borrow().len() as i64;
                let actual = if index < 0 { len + index } else { index };
                if actual < 0 || actual >= len {
                    Ok(JsValue::Undefined)
                } else {
                    Ok(items.borrow()[actual as usize].clone())
                }
            } else {
                Ok(JsValue::Undefined)
            }
        }),
    );

    // flat(depth?)
    proto.insert(
        "flat".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let depth = args
                    .get(1)
                    .unwrap_or(&JsValue::Smi(1))
                    .to_number()
                    .unwrap_or(1.0) as u32;
                fn flatten(items: &[JsValue], depth: u32) -> Vec<JsValue> {
                    let mut result = Vec::new();
                    for item in items {
                        if depth > 0
                            && let JsValue::Array(inner) = item
                        {
                            result.extend(flatten(&inner.borrow(), depth - 1));
                            continue;
                        }
                        result.push(item.clone());
                    }
                    result
                }
                Ok(JsValue::new_array(flatten(&items.borrow(), depth)))
            } else {
                Ok(JsValue::new_array(Vec::new()))
            }
        }),
    );

    // flatMap(callback)
    proto.insert(
        "flatMap".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let cb = args.get(1).cloned().unwrap_or(JsValue::Undefined);
                let mut result = Vec::new();
                for (i, item) in items.borrow().iter().enumerate() {
                    let mapped = if let JsValue::NativeFunction(f) = &cb {
                        f(vec![item.clone(), JsValue::Smi(i as i32)])?
                    } else {
                        item.clone()
                    };
                    if let JsValue::Array(inner) = mapped {
                        result.extend(inner.borrow().iter().cloned());
                    } else {
                        result.push(mapped);
                    }
                }
                Ok(JsValue::new_array(result))
            } else {
                Ok(JsValue::new_array(Vec::new()))
            }
        }),
    );

    // copyWithin(target, start, end?)
    proto.insert(
        "copyWithin".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let len = items.borrow().len() as i64;
                let target = args
                    .get(1)
                    .unwrap_or(&JsValue::Smi(0))
                    .to_number()
                    .unwrap_or(0.0) as i64;
                let start = args
                    .get(2)
                    .unwrap_or(&JsValue::Smi(0))
                    .to_number()
                    .unwrap_or(0.0) as i64;
                let end = args
                    .get(3)
                    .map(|v| v.to_number().unwrap_or(len as f64) as i64)
                    .unwrap_or(len);
                let to = if target < 0 {
                    (len + target).max(0)
                } else {
                    target.min(len)
                } as usize;
                let from = if start < 0 {
                    (len + start).max(0)
                } else {
                    start.min(len)
                } as usize;
                let fin = if end < 0 {
                    (len + end).max(0)
                } else {
                    end.min(len)
                } as usize;
                let count = (fin.saturating_sub(from)).min(items.borrow().len().saturating_sub(to));
                let buf: Vec<JsValue> = items.borrow()[from..from + count].to_vec();
                let mut v = items.borrow().clone();
                for (i, val) in buf.into_iter().enumerate() {
                    v[to + i] = val;
                }
                Ok(JsValue::new_array(v))
            } else {
                Ok(JsValue::Undefined)
            }
        }),
    );

    // splice(start, deleteCount?, ...items)
    proto.insert(
        "splice".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let len = items.borrow().len() as i64;
                let start = args
                    .get(1)
                    .unwrap_or(&JsValue::Smi(0))
                    .to_number()
                    .unwrap_or(0.0) as i64;
                let s = if start < 0 {
                    (len + start).max(0)
                } else {
                    start.min(len)
                } as usize;
                let max_del = (len - s as i64).max(0) as usize;
                let del = args
                    .get(2)
                    .map(|v| {
                        crate::builtins::util::clamped_f64_to_usize(
                            v.to_number().unwrap_or(max_del as f64),
                        )
                        .min(max_del)
                    })
                    .unwrap_or(max_del);
                let new_items = if args.len() > 3 { &args[3..] } else { &[] };
                let deleted: Vec<JsValue> = items.borrow()[s..s + del].to_vec();
                let mut v: Vec<JsValue> = items.borrow()[..s].to_vec();
                v.extend_from_slice(new_items);
                v.extend_from_slice(&items.borrow()[s + del..]);
                // Return the deleted elements as an array.
                Ok(JsValue::new_array(deleted))
            } else {
                Ok(JsValue::new_array(Vec::new()))
            }
        }),
    );

    // map(callback)
    proto.insert(
        "map".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let cb = args.get(1).cloned().unwrap_or(JsValue::Undefined);
                let mut result = Vec::with_capacity(items.borrow().len());
                for (i, item) in items.borrow().iter().enumerate() {
                    let mapped = if let JsValue::NativeFunction(f) = &cb {
                        f(vec![item.clone(), JsValue::Smi(i as i32)])?
                    } else {
                        item.clone()
                    };
                    result.push(mapped);
                }
                Ok(JsValue::new_array(result))
            } else {
                Ok(JsValue::new_array(Vec::new()))
            }
        }),
    );

    // filter(callback)
    proto.insert(
        "filter".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let cb = args.get(1).cloned().unwrap_or(JsValue::Undefined);
                let mut result = Vec::new();
                for (i, item) in items.borrow().iter().enumerate() {
                    let keep = if let JsValue::NativeFunction(f) = &cb {
                        let v = f(vec![item.clone(), JsValue::Smi(i as i32)])?;
                        v.to_boolean()
                    } else {
                        false
                    };
                    if keep {
                        result.push(item.clone());
                    }
                }
                Ok(JsValue::new_array(result))
            } else {
                Ok(JsValue::new_array(Vec::new()))
            }
        }),
    );

    // reduce(callback, initialValue?)
    proto.insert(
        "reduce".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let cb = args.get(1).cloned().unwrap_or(JsValue::Undefined);
                let (mut acc, start) = if let Some(init) = args.get(2) {
                    (init.clone(), 0usize)
                } else {
                    if items.borrow().is_empty() {
                        return Err(StatorError::TypeError(
                            "Reduce of empty array with no initial value".into(),
                        ));
                    }
                    (items.borrow()[0].clone(), 1)
                };
                for (i, item) in items.borrow().iter().enumerate().skip(start) {
                    if let JsValue::NativeFunction(f) = &cb {
                        acc = f(vec![acc, item.clone(), JsValue::Smi(i as i32)])?;
                    }
                }
                Ok(acc)
            } else {
                Err(StatorError::TypeError(
                    "Reduce of empty array with no initial value".into(),
                ))
            }
        }),
    );

    // reduceRight(callback, initialValue?)
    proto.insert(
        "reduceRight".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let cb = args.get(1).cloned().unwrap_or(JsValue::Undefined);
                let (mut acc, end_exclusive) = if let Some(init) = args.get(2) {
                    (init.clone(), items.borrow().len())
                } else {
                    if items.borrow().is_empty() {
                        return Err(StatorError::TypeError(
                            "Reduce of empty array with no initial value".into(),
                        ));
                    }
                    (
                        items.borrow()[items.borrow().len() - 1].clone(),
                        items.borrow().len() - 1,
                    )
                };
                for i in (0..end_exclusive).rev() {
                    if let JsValue::NativeFunction(f) = &cb {
                        acc = f(vec![acc, items.borrow()[i].clone(), JsValue::Smi(i as i32)])?;
                    }
                }
                Ok(acc)
            } else {
                Err(StatorError::TypeError(
                    "Reduce of empty array with no initial value".into(),
                ))
            }
        }),
    );

    // forEach(callback)
    proto.insert(
        "forEach".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let cb = args.get(1).cloned().unwrap_or(JsValue::Undefined);
                for (i, item) in items.borrow().iter().enumerate() {
                    if let JsValue::NativeFunction(f) = &cb {
                        f(vec![item.clone(), JsValue::Smi(i as i32)])?;
                    }
                }
            }
            Ok(JsValue::Undefined)
        }),
    );

    // find(callback)
    proto.insert(
        "find".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let cb = args.get(1).cloned().unwrap_or(JsValue::Undefined);
                for (i, item) in items.borrow().iter().enumerate() {
                    if let JsValue::NativeFunction(f) = &cb {
                        let v = f(vec![item.clone(), JsValue::Smi(i as i32)])?;
                        if v.to_boolean() {
                            return Ok(item.clone());
                        }
                    }
                }
            }
            Ok(JsValue::Undefined)
        }),
    );

    // findIndex(callback)
    proto.insert(
        "findIndex".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let cb = args.get(1).cloned().unwrap_or(JsValue::Undefined);
                for (i, item) in items.borrow().iter().enumerate() {
                    if let JsValue::NativeFunction(f) = &cb {
                        let v = f(vec![item.clone(), JsValue::Smi(i as i32)])?;
                        if v.to_boolean() {
                            return Ok(JsValue::Smi(i as i32));
                        }
                    }
                }
            }
            Ok(JsValue::Smi(-1))
        }),
    );

    // findLast(callback)
    proto.insert(
        "findLast".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let cb = args.get(1).cloned().unwrap_or(JsValue::Undefined);
                for i in (0..items.borrow().len()).rev() {
                    if let JsValue::NativeFunction(f) = &cb {
                        let v = f(vec![items.borrow()[i].clone(), JsValue::Smi(i as i32)])?;
                        if v.to_boolean() {
                            return Ok(items.borrow()[i].clone());
                        }
                    }
                }
            }
            Ok(JsValue::Undefined)
        }),
    );

    // findLastIndex(callback)
    proto.insert(
        "findLastIndex".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let cb = args.get(1).cloned().unwrap_or(JsValue::Undefined);
                for i in (0..items.borrow().len()).rev() {
                    if let JsValue::NativeFunction(f) = &cb {
                        let v = f(vec![items.borrow()[i].clone(), JsValue::Smi(i as i32)])?;
                        if v.to_boolean() {
                            return Ok(JsValue::Smi(i as i32));
                        }
                    }
                }
            }
            Ok(JsValue::Smi(-1))
        }),
    );

    // some(callback)
    proto.insert(
        "some".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let cb = args.get(1).cloned().unwrap_or(JsValue::Undefined);
                for (i, item) in items.borrow().iter().enumerate() {
                    if let JsValue::NativeFunction(f) = &cb {
                        let v = f(vec![item.clone(), JsValue::Smi(i as i32)])?;
                        if v.to_boolean() {
                            return Ok(JsValue::Boolean(true));
                        }
                    }
                }
            }
            Ok(JsValue::Boolean(false))
        }),
    );

    // every(callback)
    proto.insert(
        "every".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let cb = args.get(1).cloned().unwrap_or(JsValue::Undefined);
                for (i, item) in items.borrow().iter().enumerate() {
                    if let JsValue::NativeFunction(f) = &cb {
                        let v = f(vec![item.clone(), JsValue::Smi(i as i32)])?;
                        if !v.to_boolean() {
                            return Ok(JsValue::Boolean(false));
                        }
                    }
                }
            }
            Ok(JsValue::Boolean(true))
        }),
    );

    // keys()
    proto.insert(
        "keys".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let keys: Vec<JsValue> = (0..items.borrow().len())
                    .map(|i| JsValue::Smi(i as i32))
                    .collect();
                Ok(JsValue::new_array(keys))
            } else {
                Ok(JsValue::new_array(Vec::new()))
            }
        }),
    );

    // values()
    proto.insert(
        "values".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                Ok(JsValue::new_array(items.borrow().clone()))
            } else {
                Ok(JsValue::new_array(Vec::new()))
            }
        }),
    );

    // entries()
    proto.insert(
        "entries".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let entries: Vec<JsValue> = items
                    .borrow()
                    .iter()
                    .enumerate()
                    .map(|(i, v)| JsValue::new_array(vec![JsValue::Smi(i as i32), v.clone()]))
                    .collect();
                Ok(JsValue::new_array(entries))
            } else {
                Ok(JsValue::new_array(Vec::new()))
            }
        }),
    );

    // toReversed()
    proto.insert(
        "toReversed".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let mut v = items.borrow().clone();
                v.reverse();
                Ok(JsValue::new_array(v))
            } else {
                Ok(JsValue::new_array(Vec::new()))
            }
        }),
    );

    // toSorted(compareFn?)
    proto.insert(
        "toSorted".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let mut v = items.borrow().clone();
                let cmp_fn = args.get(1).cloned();
                if let Some(JsValue::NativeFunction(cmp)) = cmp_fn {
                    v.sort_by(|a, b| {
                        let result = cmp(vec![a.clone(), b.clone()]).unwrap_or(JsValue::Smi(0));
                        let n = match result {
                            JsValue::Smi(n) => n as f64,
                            JsValue::HeapNumber(n) => n,
                            _ => 0.0,
                        };
                        n.partial_cmp(&0.0).unwrap_or(std::cmp::Ordering::Equal)
                    });
                } else {
                    v.sort_by(|a, b| {
                        let sa = a.to_js_string().unwrap_or_default();
                        let sb = b.to_js_string().unwrap_or_default();
                        sa.cmp(&sb)
                    });
                }
                Ok(JsValue::new_array(v))
            } else {
                Ok(JsValue::new_array(Vec::new()))
            }
        }),
    );

    // toSpliced(start, deleteCount, ...items)
    proto.insert(
        "toSpliced".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let len = items.borrow().len() as i64;
                let start = args
                    .get(1)
                    .unwrap_or(&JsValue::Smi(0))
                    .to_number()
                    .unwrap_or(0.0) as i64;
                let s = if start < 0 {
                    (len + start).max(0)
                } else {
                    start.min(len)
                } as usize;
                let max_del = (len - s as i64).max(0) as usize;
                let del = args
                    .get(2)
                    .map(|v| {
                        crate::builtins::util::clamped_f64_to_usize(
                            v.to_number().unwrap_or(max_del as f64),
                        )
                        .min(max_del)
                    })
                    .unwrap_or(max_del);
                let new_items = if args.len() > 3 { &args[3..] } else { &[] };
                let mut v: Vec<JsValue> = items.borrow()[..s].to_vec();
                v.extend_from_slice(new_items);
                v.extend_from_slice(&items.borrow()[s + del..]);
                Ok(JsValue::new_array(v))
            } else {
                Ok(JsValue::new_array(Vec::new()))
            }
        }),
    );

    // with(index, value)
    proto.insert(
        "with".into(),
        native(|args| {
            let arr = args.first().unwrap_or(&JsValue::Undefined);
            if let JsValue::Array(items) = arr {
                let index = args
                    .get(1)
                    .unwrap_or(&JsValue::Smi(0))
                    .to_number()
                    .unwrap_or(0.0) as i64;
                let value = args.get(2).cloned().unwrap_or(JsValue::Undefined);
                let len = items.borrow().len() as i64;
                let actual = if index < 0 { len + index } else { index };
                if actual < 0 || actual >= len {
                    return Err(StatorError::RangeError(format!("Invalid index : {index}")));
                }
                let mut v = items.borrow().clone();
                v[actual as usize] = value;
                Ok(JsValue::new_array(v))
            } else {
                Err(StatorError::TypeError(
                    "Array.prototype.with called on non-array".into(),
                ))
            }
        }),
    );

    props.insert(
        "prototype".into(),
        JsValue::PlainObject(Rc::new(RefCell::new(proto))),
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
    let mut props = PropertyMap::new();

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
                    Some(key) => Ok(JsValue::String(key.into())),
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
    props.insert("dispose".into(), JsValue::Symbol(SYMBOL_DISPOSE));
    props.insert("asyncDispose".into(), JsValue::Symbol(SYMBOL_ASYNC_DISPOSE));

    // ── Symbol.prototype ─────────────────────────────────────────────────
    {
        let mut proto = PropertyMap::new();

        // Symbol.prototype.description — getter returning the description.
        proto.insert(
            "description".into(),
            native(|args| {
                let this = args.first().unwrap_or(&JsValue::Undefined);
                if let JsValue::Symbol(id) = this {
                    match symbol_description(*id) {
                        Some(desc) => Ok(JsValue::String(desc.into())),
                        None => Ok(JsValue::Undefined),
                    }
                } else {
                    Err(crate::error::StatorError::TypeError(
                        "Symbol.prototype.description requires a symbol".into(),
                    ))
                }
            }),
        );

        // Symbol.prototype.toString()
        proto.insert(
            "toString".into(),
            native(|args| {
                let this = args.first().unwrap_or(&JsValue::Undefined);
                if let JsValue::Symbol(id) = this {
                    match symbol_description(*id) {
                        Some(desc) => Ok(JsValue::String(format!("Symbol({desc})").into())),
                        None => Ok(JsValue::String("Symbol()".to_string().into())),
                    }
                } else {
                    Err(crate::error::StatorError::TypeError(
                        "Symbol.prototype.toString requires a symbol".into(),
                    ))
                }
            }),
        );

        // Symbol.prototype.valueOf()
        proto.insert(
            "valueOf".into(),
            native(|args| {
                let this = args.first().unwrap_or(&JsValue::Undefined);
                if let JsValue::Symbol(id) = this {
                    Ok(JsValue::Symbol(*id))
                } else {
                    Err(crate::error::StatorError::TypeError(
                        "Symbol.prototype.valueOf requires a symbol".into(),
                    ))
                }
            }),
        );

        // Symbol.prototype[@@toStringTag] = "Symbol"
        proto.insert("@@toStringTag".into(), JsValue::String("Symbol".into()));

        props.insert(
            "prototype".into(),
            JsValue::PlainObject(Rc::new(RefCell::new(proto))),
        );
    }

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
    let mut props = PropertyMap::new();

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
    let mut proto = PropertyMap::new();

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

// ── AsyncIterator (ES2025 §27.1.4.2) ─────────────────────────────────────────

/// Build the `AsyncIterator` constructor/namespace object with prototype
/// helpers mirroring [`make_iterator`].  Each method returns a `Promise`.
fn make_async_iterator() -> JsValue {
    use crate::builtins::promise::MicrotaskQueue;

    let mut props = PropertyMap::new();
    let queue = MicrotaskQueue::new();

    // ── Static method: AsyncIterator.from ─────────────────────────────────
    {
        let q = queue.clone();
        props.insert(
            "from".into(),
            native(move |args| {
                let iterable = args.first().unwrap_or(&JsValue::Undefined);
                Ok(async_iterator_from(iterable, &q))
            }),
        );
    }

    // ── Prototype (instance) methods ─────────────────────────────────────
    let mut proto = PropertyMap::new();

    {
        let q = queue.clone();
        proto.insert(
            "map".into(),
            native(move |args| {
                let iter = args.first().unwrap_or(&JsValue::Undefined);
                let mapper = args.get(1).unwrap_or(&JsValue::Undefined);
                Ok(async_iterator_map(iter, mapper, &q))
            }),
        );
    }
    {
        let q = queue.clone();
        proto.insert(
            "filter".into(),
            native(move |args| {
                let iter = args.first().unwrap_or(&JsValue::Undefined);
                let predicate = args.get(1).unwrap_or(&JsValue::Undefined);
                Ok(async_iterator_filter(iter, predicate, &q))
            }),
        );
    }
    {
        let q = queue.clone();
        proto.insert(
            "take".into(),
            native(move |args| {
                let iter = args.first().unwrap_or(&JsValue::Undefined);
                let limit = args
                    .get(1)
                    .unwrap_or(&JsValue::Undefined)
                    .to_number()
                    .unwrap_or(0.0);
                Ok(async_iterator_take(iter, limit.max(0.0) as usize, &q))
            }),
        );
    }
    {
        let q = queue.clone();
        proto.insert(
            "drop".into(),
            native(move |args| {
                let iter = args.first().unwrap_or(&JsValue::Undefined);
                let count = args
                    .get(1)
                    .unwrap_or(&JsValue::Undefined)
                    .to_number()
                    .unwrap_or(0.0);
                Ok(async_iterator_drop(iter, count.max(0.0) as usize, &q))
            }),
        );
    }
    {
        let q = queue.clone();
        proto.insert(
            "flatMap".into(),
            native(move |args| {
                let iter = args.first().unwrap_or(&JsValue::Undefined);
                let mapper = args.get(1).unwrap_or(&JsValue::Undefined);
                Ok(async_iterator_flat_map(iter, mapper, &q))
            }),
        );
    }
    {
        let q = queue.clone();
        proto.insert(
            "reduce".into(),
            native(move |args| {
                let iter = args.first().unwrap_or(&JsValue::Undefined);
                let reducer = args.get(1).unwrap_or(&JsValue::Undefined);
                let initial = args.get(2).cloned();
                Ok(async_iterator_reduce(iter, reducer, initial, &q))
            }),
        );
    }
    {
        let q = queue.clone();
        proto.insert(
            "toArray".into(),
            native(move |args| {
                let iter = args.first().unwrap_or(&JsValue::Undefined);
                Ok(async_iterator_to_array(iter, &q))
            }),
        );
    }
    {
        let q = queue.clone();
        proto.insert(
            "forEach".into(),
            native(move |args| {
                let iter = args.first().unwrap_or(&JsValue::Undefined);
                let callback = args.get(1).unwrap_or(&JsValue::Undefined);
                Ok(async_iterator_for_each(iter, callback, &q))
            }),
        );
    }
    {
        let q = queue.clone();
        proto.insert(
            "some".into(),
            native(move |args| {
                let iter = args.first().unwrap_or(&JsValue::Undefined);
                let predicate = args.get(1).unwrap_or(&JsValue::Undefined);
                Ok(async_iterator_some(iter, predicate, &q))
            }),
        );
    }
    {
        let q = queue.clone();
        proto.insert(
            "every".into(),
            native(move |args| {
                let iter = args.first().unwrap_or(&JsValue::Undefined);
                let predicate = args.get(1).unwrap_or(&JsValue::Undefined);
                Ok(async_iterator_every(iter, predicate, &q))
            }),
        );
    }
    {
        let q = queue.clone();
        proto.insert(
            "find".into(),
            native(move |args| {
                let iter = args.first().unwrap_or(&JsValue::Undefined);
                let predicate = args.get(1).unwrap_or(&JsValue::Undefined);
                Ok(async_iterator_find(iter, predicate, &q))
            }),
        );
    }

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
    let mut props = PropertyMap::new();

    // ── Constructor: new Map() / new Map(iterable) ───────────────────────
    props.insert(
        "__call__".into(),
        native(|args| {
            let m = if let Some(JsValue::Array(arr)) = args.first() {
                let mut pairs = Vec::new();
                for item in arr.borrow().iter() {
                    if let JsValue::Array(pair) = item {
                        let k = pair.borrow().first().cloned().unwrap_or(JsValue::Undefined);
                        let v = pair.borrow().get(1).cloned().unwrap_or(JsValue::Undefined);
                        pairs.push((k, v));
                    }
                }
                map_from_iterable(pairs)
            } else {
                map_new()
            };
            // Store the JsMap in a RefCell so prototype methods can mutate it.
            let inner = Rc::new(RefCell::new(m));
            let mut obj = PropertyMap::new();
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
                        Ok(map_create_iterator(&inner.borrow(), MapIteratorKind::Keys))
                    }),
                );
            }
            // values()
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "values".into(),
                    native(move |_| {
                        Ok(map_create_iterator(
                            &inner.borrow(),
                            MapIteratorKind::Values,
                        ))
                    }),
                );
            }
            // entries()
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "entries".into(),
                    native(move |_| {
                        Ok(map_create_iterator(
                            &inner.borrow(),
                            MapIteratorKind::Entries,
                        ))
                    }),
                );
            }
            // [Symbol.iterator]() — same as entries() per §24.1.3.13
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "@@iterator".into(),
                    native(move |_| {
                        Ok(map_create_iterator(
                            &inner.borrow(),
                            MapIteratorKind::Entries,
                        ))
                    }),
                );
            }
            Ok(JsValue::PlainObject(Rc::new(RefCell::new(obj))))
        }),
    );

    // ── Map.groupBy(items, callbackFn) ──────────────────────────────────
    props.insert(
        "groupBy".into(),
        native(|args| {
            let items = args.first().unwrap_or(&JsValue::Undefined).clone();
            let cb = args.get(1).cloned().unwrap_or(JsValue::Undefined);
            let arr = match &items {
                JsValue::Array(a) => a.clone(),
                _ => {
                    return Err(StatorError::TypeError(
                        "Map.groupBy: first argument must be iterable".into(),
                    ));
                }
            };
            let result_map = Rc::new(RefCell::new(map_new()));
            for (i, item) in arr.borrow().iter().enumerate() {
                let key = if let JsValue::NativeFunction(f) = &cb {
                    f(vec![item.clone(), JsValue::Smi(i as i32)])?
                } else {
                    JsValue::Undefined
                };
                let existing = map_get(&result_map.borrow(), &key);
                if let JsValue::Array(existing_arr) = existing {
                    existing_arr.borrow_mut().push(item.clone());
                } else {
                    map_set(
                        &mut result_map.borrow_mut(),
                        key,
                        JsValue::new_array(vec![item.clone()]),
                    );
                }
            }
            // Build a Map instance with prototype methods from the result
            let inner = result_map;
            let mut obj = PropertyMap::new();
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "size".into(),
                    JsValue::Smi(map_size(&inner.borrow()) as i32),
                );
            }
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
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "entries".into(),
                    native(move |_| {
                        Ok(map_create_iterator(
                            &inner.borrow(),
                            MapIteratorKind::Entries,
                        ))
                    }),
                );
            }
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
    let mut props = PropertyMap::new();

    // ── Constructor: new Set() / new Set(iterable) ───────────────────────
    props.insert(
        "__call__".into(),
        native(|args| {
            let s = if let Some(JsValue::Array(arr)) = args.first() {
                set_from_iterable(arr.borrow().clone())
            } else {
                set_new()
            };
            let inner = Rc::new(RefCell::new(s));
            let mut obj = PropertyMap::new();
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
                        Ok(set_create_iterator(&inner.borrow(), SetIteratorKind::Keys))
                    }),
                );
            }
            // values()
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "values".into(),
                    native(move |_| {
                        Ok(set_create_iterator(
                            &inner.borrow(),
                            SetIteratorKind::Values,
                        ))
                    }),
                );
            }
            // entries()
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "entries".into(),
                    native(move |_| {
                        Ok(set_create_iterator(
                            &inner.borrow(),
                            SetIteratorKind::Entries,
                        ))
                    }),
                );
            }
            // [Symbol.iterator]() — same as values() per §24.2.3.11
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "@@iterator".into(),
                    native(move |_| {
                        Ok(set_create_iterator(
                            &inner.borrow(),
                            SetIteratorKind::Values,
                        ))
                    }),
                );
            }
            // ── ES2025 Set composition methods ──────────────────────────
            // union(other)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "union".into(),
                    native(move |a| {
                        let other_val = a.first().unwrap_or(&JsValue::Undefined);
                        let other_set = extract_set_from_arg(other_val)?;
                        let result = set_union(&inner.borrow(), &other_set);
                        build_set_instance(result)
                    }),
                );
            }
            // intersection(other)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "intersection".into(),
                    native(move |a| {
                        let other_val = a.first().unwrap_or(&JsValue::Undefined);
                        let other_set = extract_set_from_arg(other_val)?;
                        let result = set_intersection(&inner.borrow(), &other_set);
                        build_set_instance(result)
                    }),
                );
            }
            // difference(other)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "difference".into(),
                    native(move |a| {
                        let other_val = a.first().unwrap_or(&JsValue::Undefined);
                        let other_set = extract_set_from_arg(other_val)?;
                        let result = set_difference(&inner.borrow(), &other_set);
                        build_set_instance(result)
                    }),
                );
            }
            // symmetricDifference(other)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "symmetricDifference".into(),
                    native(move |a| {
                        let other_val = a.first().unwrap_or(&JsValue::Undefined);
                        let other_set = extract_set_from_arg(other_val)?;
                        let result = set_symmetric_difference(&inner.borrow(), &other_set);
                        build_set_instance(result)
                    }),
                );
            }
            // isSubsetOf(other)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "isSubsetOf".into(),
                    native(move |a| {
                        let other_val = a.first().unwrap_or(&JsValue::Undefined);
                        let other_set = extract_set_from_arg(other_val)?;
                        Ok(JsValue::Boolean(set_is_subset_of(
                            &inner.borrow(),
                            &other_set,
                        )))
                    }),
                );
            }
            // isSupersetOf(other)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "isSupersetOf".into(),
                    native(move |a| {
                        let other_val = a.first().unwrap_or(&JsValue::Undefined);
                        let other_set = extract_set_from_arg(other_val)?;
                        Ok(JsValue::Boolean(set_is_superset_of(
                            &inner.borrow(),
                            &other_set,
                        )))
                    }),
                );
            }
            // isDisjointFrom(other)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "isDisjointFrom".into(),
                    native(move |a| {
                        let other_val = a.first().unwrap_or(&JsValue::Undefined);
                        let other_set = extract_set_from_arg(other_val)?;
                        Ok(JsValue::Boolean(set_is_disjoint_from(
                            &inner.borrow(),
                            &other_set,
                        )))
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
    let mut props = PropertyMap::new();

    props.insert(
        "__call__".into(),
        native(|_args| {
            let inner = Rc::new(RefCell::new(weak_map_new()));
            let mut obj = PropertyMap::new();

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
    let mut props = PropertyMap::new();

    props.insert(
        "__call__".into(),
        native(|_args| {
            let inner = Rc::new(RefCell::new(weak_set_new()));
            let mut obj = PropertyMap::new();

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

// ── WeakRef constructor (ES2025 §26.1) ───────────────────────────────────────

/// Build the `WeakRef` constructor/namespace object.
///
/// The returned `PlainObject` provides a `__call__` constructor that creates
/// a new `WeakRef` instance with a `deref` prototype method.  The target
/// must be an `Object` pointer; non-object targets cause a `TypeError`.
fn make_weak_ref_builtin() -> JsValue {
    let mut props = PropertyMap::new();

    props.insert(
        "__call__".into(),
        native(|args| {
            let target = args.first().unwrap_or(&JsValue::Undefined);
            let wr = match target {
                JsValue::Object(ptr) => weak_ref_new(*ptr)?,
                JsValue::PlainObject(rc) => weak_ref_new_plain(rc),
                _ => {
                    return Err(StatorError::TypeError(
                        "WeakRef target must be an object".into(),
                    ));
                }
            };
            let inner = Rc::new(RefCell::new(wr));
            let mut obj = PropertyMap::new();

            // deref()
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "deref".into(),
                    native(move |_a| Ok(weak_ref_deref(&inner.borrow()))),
                );
            }

            Ok(JsValue::PlainObject(Rc::new(RefCell::new(obj))))
        }),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

// ── FinalizationRegistry constructor (ES2025 §26.2) ─────────────────────────

/// Build the `FinalizationRegistry` constructor/namespace object.
///
/// The returned `PlainObject` provides a `__call__` constructor that creates
/// a new `FinalizationRegistry` instance with `register` and `unregister`
/// prototype methods.  The cleanup callback is stored as a JS-level value;
/// actual invocation happens when the GC integration is complete.
fn make_finalization_registry_builtin() -> JsValue {
    let mut props = PropertyMap::new();

    props.insert(
        "__call__".into(),
        native(|args| {
            let callback = args.first().cloned().unwrap_or(JsValue::Undefined);
            if !matches!(callback, JsValue::NativeFunction(_)) {
                return Err(StatorError::TypeError(
                    "FinalizationRegistry requires a callable cleanup callback".into(),
                ));
            }

            let inner = Rc::new(RefCell::new(finalization_registry_new()));
            let callback = Rc::new(callback);
            let mut obj = PropertyMap::new();

            // register(target, heldValue [, unregisterToken])
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "register".into(),
                    native(move |a| {
                        let target = a.first().unwrap_or(&JsValue::Undefined);
                        let held_value = a.get(1).cloned().unwrap_or(JsValue::Undefined);
                        let token = a.get(2).unwrap_or(&JsValue::Undefined);

                        match target {
                            JsValue::Object(ptr) => {
                                let unregister_token = match token {
                                    JsValue::Object(tok_ptr) => Some(*tok_ptr),
                                    JsValue::Undefined => None,
                                    _ => {
                                        return Err(StatorError::TypeError(
                                            "unregister token must be an object or undefined"
                                                .into(),
                                        ));
                                    }
                                };
                                finalization_registry_register(
                                    &mut inner.borrow_mut(),
                                    *ptr,
                                    held_value,
                                    unregister_token,
                                )?;
                                Ok(JsValue::Undefined)
                            }
                            JsValue::PlainObject(rc) => {
                                let unregister_token = match token {
                                    JsValue::PlainObject(tok_rc) => Some(tok_rc),
                                    JsValue::Undefined => None,
                                    _ => {
                                        return Err(StatorError::TypeError(
                                            "unregister token must be an object or undefined"
                                                .into(),
                                        ));
                                    }
                                };
                                finalization_registry_register_plain(
                                    &mut inner.borrow_mut(),
                                    rc,
                                    held_value,
                                    unregister_token,
                                );
                                Ok(JsValue::Undefined)
                            }
                            _ => Err(StatorError::TypeError(
                                "FinalizationRegistry target must be an object".into(),
                            )),
                        }
                    }),
                );
            }

            // unregister(unregisterToken)
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "unregister".into(),
                    native(move |a| {
                        let token = a.first().unwrap_or(&JsValue::Undefined);
                        match token {
                            JsValue::Object(ptr) => Ok(JsValue::Boolean(
                                finalization_registry_unregister(&mut inner.borrow_mut(), *ptr)?,
                            )),
                            JsValue::PlainObject(rc) => Ok(JsValue::Boolean(
                                finalization_registry_unregister_plain(&mut inner.borrow_mut(), rc),
                            )),
                            _ => Err(StatorError::TypeError(
                                "unregister token must be an object".into(),
                            )),
                        }
                    }),
                );
            }

            // cleanupSome() — sweep plain targets, then drain the queue
            {
                let inner = Rc::clone(&inner);
                let cb = Rc::clone(&callback);
                obj.insert(
                    "cleanupSome".into(),
                    native(move |_a| {
                        finalization_registry_sweep_plain(&mut inner.borrow_mut());
                        let held_values = finalization_registry_drain(&mut inner.borrow_mut());
                        if let JsValue::NativeFunction(ref f) = *cb {
                            for held in held_values {
                                f(vec![held])?;
                            }
                        }
                        Ok(JsValue::Undefined)
                    }),
                );
            }

            // __notify__ — internal GC hook exposed for testing
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "__notify__".into(),
                    native(move |a| {
                        let target = a.first().unwrap_or(&JsValue::Undefined);
                        if let JsValue::Object(ptr) = target {
                            finalization_registry_notify(&mut inner.borrow_mut(), *ptr);
                        }
                        Ok(JsValue::Undefined)
                    }),
                );
            }

            Ok(JsValue::PlainObject(Rc::new(RefCell::new(obj))))
        }),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

// ── String constructor ───────────────────────────────────────────────────────

// ── Function constructor (ES2025 §20.2) ──────────────────────────────────────

/// Build the `Function` constructor/namespace object.
///
/// The returned `PlainObject` carries:
/// - `__call__` — the `Function(…args, body)` dynamic constructor.
/// - `prototype` — an object with `bind`, `call`, `apply`, `toString`,
///   `Symbol.hasInstance`, `name`, and `length`.
fn make_function() -> JsValue {
    let mut props = PropertyMap::new();

    // ── Constructor: new Function(…args, body) ──────────────────────────
    props.insert(
        "__call__".into(),
        native(|args| function_constructor(&args)),
    );

    // ── Function.prototype ──────────────────────────────────────────────
    let mut proto = PropertyMap::new();

    // Function.prototype.call(thisArg, ...args)
    proto.insert(
        "call".into(),
        native(|args| {
            let func = args.first().cloned().unwrap_or(JsValue::Undefined);
            let this_arg = args.get(1).cloned().unwrap_or(JsValue::Undefined);
            let call_args: Vec<JsValue> = args.get(2..).unwrap_or(&[]).to_vec();
            match func {
                JsValue::NativeFunction(f) => function_call(&f, &this_arg, &call_args),
                _ => Err(StatorError::TypeError(
                    "Function.prototype.call requires a callable".into(),
                )),
            }
        }),
    );

    // Function.prototype.apply(thisArg, argsArray)
    proto.insert(
        "apply".into(),
        native(|args| {
            let func = args.first().cloned().unwrap_or(JsValue::Undefined);
            let this_arg = args.get(1).cloned().unwrap_or(JsValue::Undefined);
            let args_array = match args.get(2) {
                Some(JsValue::Array(arr)) => Some(arr.borrow().clone()),
                Some(JsValue::Null) | Some(JsValue::Undefined) | None => None,
                _ => {
                    return Err(StatorError::TypeError(
                        "CreateListFromArrayLike called on non-object".into(),
                    ));
                }
            };
            match func {
                JsValue::NativeFunction(f) => function_apply(&f, &this_arg, &args_array),
                _ => Err(StatorError::TypeError(
                    "Function.prototype.apply requires a callable".into(),
                )),
            }
        }),
    );

    // Function.prototype.bind(thisArg, ...args)
    proto.insert(
        "bind".into(),
        native(|args| {
            let func = args.first().cloned().unwrap_or(JsValue::Undefined);
            let this_arg = args.get(1).cloned().unwrap_or(JsValue::Undefined);
            let bound_args: Vec<JsValue> = args.get(2..).unwrap_or(&[]).to_vec();
            match func {
                JsValue::NativeFunction(ref f) => {
                    let bound = function_bind(f, &this_arg, &bound_args);
                    Ok(JsValue::NativeFunction(bound))
                }
                _ => Err(StatorError::TypeError(
                    "Function.prototype.bind requires a callable".into(),
                )),
            }
        }),
    );

    // Function.prototype.toString()
    proto.insert(
        "toString".into(),
        native(|args| {
            let _func = args.first().cloned().unwrap_or(JsValue::Undefined);
            Ok(JsValue::String(function_to_string("").into()))
        }),
    );

    // Function.prototype[Symbol.hasInstance](V)
    proto.insert(
        format!("Symbol({})", SYMBOL_HAS_INSTANCE),
        native(|args| {
            let constructor_proto = args.first().cloned().unwrap_or(JsValue::Undefined);
            let value = args.get(1).cloned().unwrap_or(JsValue::Undefined);
            Ok(JsValue::Boolean(function_has_instance(
                &value,
                &constructor_proto,
            )))
        }),
    );

    // Function.prototype.name (empty string for the prototype itself)
    proto.insert("name".into(), JsValue::String(String::new().into()));

    // Function.prototype.length (0 for the prototype itself)
    proto.insert("length".into(), JsValue::Smi(0));

    // Function.length = 1 (the constructor expects 1 argument)
    props.insert("length".into(), JsValue::Smi(1));

    // Function.name = "Function"
    props.insert("name".into(), JsValue::String("Function".into()));

    props.insert(
        "prototype".into(),
        JsValue::PlainObject(Rc::new(RefCell::new(proto))),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

/// Build the `String` constructor/namespace object with static and prototype
/// methods.
///
/// The returned `PlainObject` carries:
/// - `__call__` — the callable `String(value)` conversion.
/// - Static methods: `fromCharCode`, `fromCodePoint`, `raw`.
/// - `prototype` — an object with all `String.prototype.*` methods.
fn make_string() -> JsValue {
    let mut props = PropertyMap::new();

    // ── Callable: String(value) ─────────────────────────────────────────
    props.insert(
        "__call__".into(),
        native(|args| {
            let val = args.first().unwrap_or(&JsValue::Undefined);
            Ok(JsValue::String(val.to_js_string()?.into()))
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
            Ok(JsValue::String(string_from_char_code(&codes).into()))
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
            Ok(JsValue::String(string_from_code_point(&codes)?.into()))
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
                    .borrow()
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
            Ok(JsValue::String(string_raw(&raw_refs, &sub_refs).into()))
        }),
    );

    // ── Prototype methods ───────────────────────────────────────────────
    let mut proto = PropertyMap::new();

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
            Ok(JsValue::String(string_char_at(&s, pos).into()))
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
            Ok(JsValue::String(string_concat(&s, &refs).into()))
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
            Ok(JsValue::String(string_slice(&s, start, end).into()))
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
            Ok(JsValue::String(string_substring(&s, start, end).into()))
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
            Ok(JsValue::String(string_to_upper_case(&s).into()))
        }),
    );

    // toLowerCase()
    proto.insert(
        "toLowerCase".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_to_lower_case(&s).into()))
        }),
    );

    // trim()
    proto.insert(
        "trim".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_trim(&s).into()))
        }),
    );

    // trimStart()
    proto.insert(
        "trimStart".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_trim_start(&s).into()))
        }),
    );

    // trimEnd()
    proto.insert(
        "trimEnd".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_trim_end(&s).into()))
        }),
    );

    // trimLeft() — §B.2.3.1 legacy alias for trimStart
    proto.insert(
        "trimLeft".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_trim_start(&s).into()))
        }),
    );

    // trimRight() — §B.2.3.2 legacy alias for trimEnd
    proto.insert(
        "trimRight".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_trim_end(&s).into()))
        }),
    );

    // split(separator?, limit?)
    proto.insert(
        "split".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let sep_arg = args.get(1).unwrap_or(&JsValue::Undefined);
            // Delegate to RegExp[@@split] when separator is a regexp.
            if let Some(result) = try_regexp_symbol(sep_arg, "__symbol_split__", {
                let mut a = vec![JsValue::String(s.clone().into())];
                if let Some(lim) = args.get(2) {
                    a.push(lim.clone());
                }
                a
            }) {
                return result;
            }
            let sep = match sep_arg {
                JsValue::Undefined => None,
                v => Some(v.to_js_string()?),
            };
            let limit = match args.get(2) {
                Some(JsValue::Undefined) | None => None,
                Some(v) => Some(v.to_number().unwrap_or(0.0) as u32),
            };
            let parts = string_split(&s, sep.as_deref(), limit);
            let arr: Vec<JsValue> = parts
                .into_iter()
                .map(|s| JsValue::String(s.into()))
                .collect();
            Ok(JsValue::new_array(arr))
        }),
    );

    // replace(searchValue, replaceValue)
    proto.insert(
        "replace".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let search_arg = args.get(1).unwrap_or(&JsValue::Undefined);
            // Delegate to RegExp[@@replace] when searchValue is a regexp.
            if let Some(result) = try_regexp_symbol(
                search_arg,
                "__symbol_replace__",
                vec![
                    JsValue::String(s.clone().into()),
                    args.get(2).unwrap_or(&JsValue::Undefined).clone(),
                ],
            ) {
                return result;
            }
            let search = search_arg.to_js_string()?;
            let replacement = args.get(2).unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(
                string_replace(&s, &search, &replacement).into(),
            ))
        }),
    );

    // replaceAll(searchValue, replaceValue)
    proto.insert(
        "replaceAll".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let search_arg = args.get(1).unwrap_or(&JsValue::Undefined);
            // Delegate to RegExp[@@replace] when searchValue is a regexp.
            // Per spec, replaceAll with a non-global regexp throws TypeError.
            if let JsValue::PlainObject(map) = search_arg {
                let borrow = map.borrow();
                if matches!(borrow.get("__is_regexp__"), Some(JsValue::Boolean(true))) {
                    let is_global = matches!(borrow.get("global"), Some(JsValue::Boolean(true)));
                    if !is_global {
                        return Err(crate::error::StatorError::TypeError(
                            "String.prototype.replaceAll called with a non-global RegExp argument"
                                .to_string(),
                        ));
                    }
                    if let Some(JsValue::NativeFunction(f)) =
                        borrow.get("__symbol_replace__").cloned()
                    {
                        drop(borrow);
                        return f(vec![
                            JsValue::String(s.into()),
                            args.get(2).unwrap_or(&JsValue::Undefined).clone(),
                        ]);
                    }
                }
            }
            let search = search_arg.to_js_string()?;
            let replacement = args.get(2).unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(
                string_replace_all(&s, &search, &replacement).into(),
            ))
        }),
    );

    // match(regexp)
    proto.insert(
        "match".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let pattern_arg = args.get(1).unwrap_or(&JsValue::Undefined);
            // Delegate to RegExp[@@match] when argument is a regexp.
            if let Some(result) = try_regexp_symbol(
                pattern_arg,
                "__symbol_match__",
                vec![JsValue::String(s.clone().into())],
            ) {
                return result;
            }
            let pattern = pattern_arg.to_js_string()?;
            match string_match(&s, &pattern) {
                Some(groups) => {
                    let arr: Vec<JsValue> = groups
                        .into_iter()
                        .map(|s| JsValue::String(s.into()))
                        .collect();
                    Ok(JsValue::new_array(arr))
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
            let pattern_arg = args.get(1).unwrap_or(&JsValue::Undefined);
            // Delegate to RegExp[@@matchAll] when argument is a regexp.
            if let Some(result) = try_regexp_symbol(
                pattern_arg,
                "__symbol_match_all__",
                vec![JsValue::String(s.clone().into())],
            ) {
                return result;
            }
            let pattern = pattern_arg.to_js_string()?;
            match string_match_all(&s, &pattern) {
                Some(matches) => {
                    let arr: Vec<JsValue> = matches
                        .into_iter()
                        .map(|s| JsValue::String(s.into()))
                        .collect();
                    Ok(JsValue::new_array(arr))
                }
                None => Ok(JsValue::new_array(Vec::new())),
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
            Ok(JsValue::String(string_repeat(&s, n as i64)?.into()))
        }),
    );

    // padStart(targetLength, padString?)
    proto.insert(
        "padStart".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let raw = args
                .get(1)
                .unwrap_or(&JsValue::Undefined)
                .to_number()
                .unwrap_or(0.0);
            let target_len = if raw.is_nan() || raw < 0.0 {
                0usize
            } else {
                crate::builtins::util::clamped_f64_to_usize(raw)
            };
            let pad = match args.get(2) {
                Some(JsValue::Undefined) | None => None,
                Some(v) => Some(v.to_js_string()?),
            };
            Ok(JsValue::String(
                string_pad_start(&s, target_len, pad.as_deref())?.into(),
            ))
        }),
    );

    // padEnd(targetLength, padString?)
    proto.insert(
        "padEnd".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let raw = args
                .get(1)
                .unwrap_or(&JsValue::Undefined)
                .to_number()
                .unwrap_or(0.0);
            let target_len = if raw.is_nan() || raw < 0.0 {
                0usize
            } else {
                crate::builtins::util::clamped_f64_to_usize(raw)
            };
            let pad = match args.get(2) {
                Some(JsValue::Undefined) | None => None,
                Some(v) => Some(v.to_js_string()?),
            };
            Ok(JsValue::String(
                string_pad_end(&s, target_len, pad.as_deref())?.into(),
            ))
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
                Some(ch) => Ok(JsValue::String(ch.into())),
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
            Ok(JsValue::String(
                string_normalize(&s, form.as_deref())?.into(),
            ))
        }),
    );

    // search(regexp)
    proto.insert(
        "search".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let pattern_arg = args.get(1).unwrap_or(&JsValue::Undefined);
            // Delegate to RegExp[@@search] when argument is a regexp.
            if let Some(result) = try_regexp_symbol(
                pattern_arg,
                "__symbol_search__",
                vec![JsValue::String(s.clone().into())],
            ) {
                return result;
            }
            let pattern = pattern_arg.to_js_string()?;
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
            Ok(JsValue::String(string_to_well_formed(&s).into()))
        }),
    );

    // localeCompare(that)
    proto.insert(
        "localeCompare".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let that = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(num(string_locale_compare(&s, &that) as f64))
        }),
    );

    // toLocaleLowerCase()
    proto.insert(
        "toLocaleLowerCase".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_to_locale_lower_case(&s).into()))
        }),
    );

    // toLocaleUpperCase()
    proto.insert(
        "toLocaleUpperCase".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_to_locale_upper_case(&s).into()))
        }),
    );

    // toString()
    proto.insert(
        "toString".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(s.into()))
        }),
    );

    // valueOf()
    proto.insert(
        "valueOf".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(s.into()))
        }),
    );

    // [Symbol.iterator]() — returns the code-point iteration as an array.
    proto.insert(
        "@@iterator".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let chars: Vec<JsValue> = string_iter(&s)
                .into_iter()
                .map(|s| JsValue::String(s.into()))
                .collect();
            Ok(JsValue::new_array(chars))
        }),
    );

    // ── Annex B prototype methods ───────────────────────────────────────

    // substr(start, length?)
    proto.insert(
        "substr".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let start = args
                .get(1)
                .unwrap_or(&JsValue::Undefined)
                .to_number()
                .unwrap_or(0.0) as i64;
            let length = match args.get(2) {
                Some(JsValue::Undefined) | None => None,
                Some(v) => Some(v.to_number().unwrap_or(0.0) as i64),
            };
            Ok(JsValue::String(string_substr(&s, start, length).into()))
        }),
    );

    // anchor(name)
    proto.insert(
        "anchor".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let name = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_anchor(&s, &name).into()))
        }),
    );

    // big()
    proto.insert(
        "big".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_big(&s).into()))
        }),
    );

    // blink()
    proto.insert(
        "blink".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_blink(&s).into()))
        }),
    );

    // bold()
    proto.insert(
        "bold".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_bold(&s).into()))
        }),
    );

    // fixed()
    proto.insert(
        "fixed".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_fixed(&s).into()))
        }),
    );

    // fontcolor(color)
    proto.insert(
        "fontcolor".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let color = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_fontcolor(&s, &color).into()))
        }),
    );

    // fontsize(size)
    proto.insert(
        "fontsize".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let size = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_fontsize(&s, &size).into()))
        }),
    );

    // italics()
    proto.insert(
        "italics".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_italics(&s).into()))
        }),
    );

    // link(url)
    proto.insert(
        "link".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let url = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_link(&s, &url).into()))
        }),
    );

    // small()
    proto.insert(
        "small".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_small(&s).into()))
        }),
    );

    // strike()
    proto.insert(
        "strike".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_strike(&s).into()))
        }),
    );

    // sub()
    proto.insert(
        "sub".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_sub(&s).into()))
        }),
    );

    // sup()
    proto.insert(
        "sup".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(string_sup(&s).into()))
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

    let mut props = PropertyMap::new();
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
                let mut obj = PropertyMap::new();
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
                            Err(e) => Err(JsValue::String(e.to_string().into())),
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
    let mut props = PropertyMap::new();

    // Callable: new RegExp(pattern, flags)
    props.insert("__call__".into(), native(|args| regexp_construct(&args)));

    // Annex B legacy static properties (stubs)
    for i in 1..=9 {
        props.insert(format!("${i}"), JsValue::String(String::new().into()));
    }
    props.insert("input".into(), JsValue::String(String::new().into()));
    props.insert("lastMatch".into(), JsValue::String(String::new().into()));
    props.insert("lastParen".into(), JsValue::String(String::new().into()));
    props.insert("leftContext".into(), JsValue::String(String::new().into()));
    props.insert("rightContext".into(), JsValue::String(String::new().into()));

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

/// Build the `BigInt` global constructor with `asIntN` and `asUintN` static methods.
fn make_bigint() -> JsValue {
    let mut props = PropertyMap::new();

    // BigInt(value) — callable constructor (must not be called with `new`)
    props.insert(
        "__call__".into(),
        native(|args| {
            let val = args.first().unwrap_or(&JsValue::Undefined);
            match val {
                JsValue::BigInt(n) => Ok(JsValue::BigInt(*n)),
                JsValue::Smi(n) => Ok(JsValue::BigInt(i128::from(*n))),
                JsValue::HeapNumber(n) => {
                    if n.is_nan() || n.is_infinite() || n.fract() != 0.0 {
                        Err(StatorError::RangeError(format!(
                            "The number {n} cannot be converted to a BigInt because it is not an integer"
                        )))
                    } else {
                        Ok(JsValue::BigInt(*n as i128))
                    }
                }
                JsValue::Boolean(b) => Ok(JsValue::BigInt(if *b { 1 } else { 0 })),
                JsValue::String(s) => {
                    let trimmed = s.trim();
                    if trimmed.is_empty() {
                        return Err(StatorError::SyntaxError(
                            "Cannot convert  to a BigInt".to_string(),
                        ));
                    }
                    let parsed = if let Some(hex) =
                        trimmed.strip_prefix("0x").or_else(|| trimmed.strip_prefix("0X"))
                    {
                        i128::from_str_radix(hex, 16)
                    } else if let Some(oct) =
                        trimmed.strip_prefix("0o").or_else(|| trimmed.strip_prefix("0O"))
                    {
                        i128::from_str_radix(oct, 8)
                    } else if let Some(bin) =
                        trimmed.strip_prefix("0b").or_else(|| trimmed.strip_prefix("0B"))
                    {
                        i128::from_str_radix(bin, 2)
                    } else {
                        trimmed.parse::<i128>()
                    };
                    parsed.map(JsValue::BigInt).map_err(|_| {
                        StatorError::SyntaxError(format!(
                            "Cannot convert {s} to a BigInt"
                        ))
                    })
                }
                _ => Err(StatorError::TypeError(format!(
                    "Cannot convert {} to a BigInt",
                    val.to_js_string().unwrap_or_default()
                ))),
            }
        }),
    );

    // BigInt.asIntN(bits, bigint)
    props.insert(
        "asIntN".into(),
        native(|args| {
            let bits = args.first().unwrap_or(&JsValue::Undefined).to_number()? as u32;
            let bigint = match args.get(1) {
                Some(JsValue::BigInt(n)) => *n,
                _ => {
                    return Err(StatorError::TypeError(
                        "Cannot convert a non-BigInt value to a BigInt".to_string(),
                    ));
                }
            };
            if bits == 0 {
                return Ok(JsValue::BigInt(0));
            }
            if bits >= 128 {
                return Ok(JsValue::BigInt(bigint));
            }
            let mask = (1i128 << bits) - 1;
            let truncated = bigint & mask;
            // Sign extension
            if truncated & (1i128 << (bits - 1)) != 0 {
                Ok(JsValue::BigInt(truncated | !mask))
            } else {
                Ok(JsValue::BigInt(truncated))
            }
        }),
    );

    // BigInt.asUintN(bits, bigint)
    props.insert(
        "asUintN".into(),
        native(|args| {
            let bits = args.first().unwrap_or(&JsValue::Undefined).to_number()? as u32;
            let bigint = match args.get(1) {
                Some(JsValue::BigInt(n)) => *n,
                _ => {
                    return Err(StatorError::TypeError(
                        "Cannot convert a non-BigInt value to a BigInt".to_string(),
                    ));
                }
            };
            if bits == 0 {
                return Ok(JsValue::BigInt(0));
            }
            if bits >= 128 {
                return Ok(JsValue::BigInt(bigint));
            }
            let mask = (1i128 << bits) - 1;
            Ok(JsValue::BigInt(bigint & mask))
        }),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

/// Extract a `Vec<JsPromise>` from an argument that should be a `JsValue::Array`
/// of `JsValue::Promise` elements.
fn extract_promise_array(arg: Option<&JsValue>) -> Vec<crate::builtins::promise::JsPromise> {
    match arg {
        Some(JsValue::Array(arr)) => arr
            .borrow()
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
            Err(e) => Err(JsValue::String(e.to_string().into())),
        }))
    } else {
        None
    }
}

// ── Intl ─────────────────────────────────────────────────────────────────────

/// Create a `supportedLocalesOf` native function shared by all Intl constructors.
///
/// Stub: returns all requested locales (we fall back to en-US for everything).
fn make_supported_locales_of() -> JsValue {
    native(|args| {
        let locales: Vec<JsValue> = match args.first() {
            Some(JsValue::Array(arr)) => arr.borrow().iter().cloned().collect(),
            Some(JsValue::String(s)) => vec![JsValue::String(s.clone())],
            _ => Vec::new(),
        };
        Ok(JsValue::new_array(locales))
    })
}

/// Build the `Intl` namespace object (ECMA-402).
///
/// Each property is a constructor-like `PlainObject` with a `__call__` method
/// that returns an instance (another `PlainObject`) carrying a `format` (or
/// `compare` / `select`) method.
fn make_intl() -> JsValue {
    let mut ns = PropertyMap::new();

    // ── Intl.NumberFormat ────────────────────────────────────────────────
    ns.insert("NumberFormat".into(), {
        let mut ctor = PropertyMap::new();
        ctor.insert(
            "__call__".into(),
            native(|_args| {
                let mut obj = PropertyMap::new();
                obj.insert("format".into(), native(|a| number_format_js(&a)));
                obj.insert(
                    "formatToParts".into(),
                    native(|a| number_format_to_parts_js(&a)),
                );
                obj.insert(
                    "resolvedOptions".into(),
                    native(|_| {
                        let mut opts = PropertyMap::new();
                        opts.insert("locale".into(), JsValue::String("en-US".into()));
                        opts.insert("numberingSystem".into(), JsValue::String("latn".into()));
                        Ok(JsValue::PlainObject(Rc::new(RefCell::new(opts))))
                    }),
                );
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(obj))))
            }),
        );
        ctor.insert("supportedLocalesOf".into(), make_supported_locales_of());
        let mut proto = PropertyMap::new();
        proto.insert("format".into(), native(|a| number_format_js(&a)));
        proto.insert(
            "formatToParts".into(),
            native(|a| number_format_to_parts_js(&a)),
        );
        proto.insert(
            "resolvedOptions".into(),
            native(|_| {
                let mut opts = PropertyMap::new();
                opts.insert("locale".into(), JsValue::String("en-US".into()));
                opts.insert("numberingSystem".into(), JsValue::String("latn".into()));
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(opts))))
            }),
        );
        ctor.insert(
            "prototype".into(),
            JsValue::PlainObject(Rc::new(RefCell::new(proto))),
        );
        JsValue::PlainObject(Rc::new(RefCell::new(ctor)))
    });

    // ── Intl.DateTimeFormat ──────────────────────────────────────────────
    ns.insert("DateTimeFormat".into(), {
        let mut ctor = PropertyMap::new();
        ctor.insert(
            "__call__".into(),
            native(|_args| {
                let mut obj = PropertyMap::new();
                obj.insert("format".into(), native(|a| date_time_format_js(&a)));
                obj.insert(
                    "formatToParts".into(),
                    native(|a| date_time_format_to_parts_js(&a)),
                );
                obj.insert(
                    "resolvedOptions".into(),
                    native(|_| {
                        let mut opts = PropertyMap::new();
                        opts.insert("locale".into(), JsValue::String("en-US".into()));
                        opts.insert("calendar".into(), JsValue::String("gregory".into()));
                        opts.insert("timeZone".into(), JsValue::String("UTC".into()));
                        Ok(JsValue::PlainObject(Rc::new(RefCell::new(opts))))
                    }),
                );
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(obj))))
            }),
        );
        ctor.insert("supportedLocalesOf".into(), make_supported_locales_of());
        let mut proto = PropertyMap::new();
        proto.insert("format".into(), native(|a| date_time_format_js(&a)));
        proto.insert(
            "formatToParts".into(),
            native(|a| date_time_format_to_parts_js(&a)),
        );
        proto.insert(
            "resolvedOptions".into(),
            native(|_| {
                let mut opts = PropertyMap::new();
                opts.insert("locale".into(), JsValue::String("en-US".into()));
                opts.insert("calendar".into(), JsValue::String("gregory".into()));
                opts.insert("timeZone".into(), JsValue::String("UTC".into()));
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(opts))))
            }),
        );
        ctor.insert(
            "prototype".into(),
            JsValue::PlainObject(Rc::new(RefCell::new(proto))),
        );
        JsValue::PlainObject(Rc::new(RefCell::new(ctor)))
    });

    // ── Intl.Collator ───────────────────────────────────────────────────
    ns.insert("Collator".into(), {
        let mut ctor = PropertyMap::new();
        ctor.insert(
            "__call__".into(),
            native(|_args| {
                let mut obj = PropertyMap::new();
                obj.insert("compare".into(), native(|a| collator_compare_js(&a)));
                obj.insert(
                    "resolvedOptions".into(),
                    native(|_| {
                        let mut opts = PropertyMap::new();
                        opts.insert("locale".into(), JsValue::String("en-US".into()));
                        opts.insert("sensitivity".into(), JsValue::String("variant".into()));
                        Ok(JsValue::PlainObject(Rc::new(RefCell::new(opts))))
                    }),
                );
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(obj))))
            }),
        );
        ctor.insert("supportedLocalesOf".into(), make_supported_locales_of());
        let mut proto = PropertyMap::new();
        proto.insert("compare".into(), native(|a| collator_compare_js(&a)));
        proto.insert(
            "resolvedOptions".into(),
            native(|_| {
                let mut opts = PropertyMap::new();
                opts.insert("locale".into(), JsValue::String("en-US".into()));
                opts.insert("sensitivity".into(), JsValue::String("variant".into()));
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(opts))))
            }),
        );
        ctor.insert(
            "prototype".into(),
            JsValue::PlainObject(Rc::new(RefCell::new(proto))),
        );
        JsValue::PlainObject(Rc::new(RefCell::new(ctor)))
    });

    // ── Intl.PluralRules ────────────────────────────────────────────────
    ns.insert("PluralRules".into(), {
        let mut ctor = PropertyMap::new();
        ctor.insert(
            "__call__".into(),
            native(|_args| {
                let mut obj = PropertyMap::new();
                obj.insert("select".into(), native(|a| plural_rules_select_js(&a)));
                obj.insert(
                    "resolvedOptions".into(),
                    native(|_| {
                        let mut opts = PropertyMap::new();
                        opts.insert("locale".into(), JsValue::String("en-US".into()));
                        opts.insert("type".into(), JsValue::String("cardinal".into()));
                        Ok(JsValue::PlainObject(Rc::new(RefCell::new(opts))))
                    }),
                );
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(obj))))
            }),
        );
        ctor.insert("supportedLocalesOf".into(), make_supported_locales_of());
        let mut proto = PropertyMap::new();
        proto.insert("select".into(), native(|a| plural_rules_select_js(&a)));
        proto.insert(
            "resolvedOptions".into(),
            native(|_| {
                let mut opts = PropertyMap::new();
                opts.insert("locale".into(), JsValue::String("en-US".into()));
                opts.insert("type".into(), JsValue::String("cardinal".into()));
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(opts))))
            }),
        );
        ctor.insert(
            "prototype".into(),
            JsValue::PlainObject(Rc::new(RefCell::new(proto))),
        );
        JsValue::PlainObject(Rc::new(RefCell::new(ctor)))
    });

    // ── Intl.ListFormat ─────────────────────────────────────────────────
    ns.insert("ListFormat".into(), {
        let mut ctor = PropertyMap::new();
        ctor.insert(
            "__call__".into(),
            native(|args| {
                let list_type = if let Some(JsValue::PlainObject(opts)) = args.get(1) {
                    opts.borrow()
                        .get("type")
                        .and_then(|v| {
                            if let JsValue::String(s) = v {
                                Some(s.to_string())
                            } else {
                                None
                            }
                        })
                        .unwrap_or_else(|| "conjunction".to_string())
                } else {
                    "conjunction".to_string()
                };
                let lt = list_type.clone();
                let mut obj = PropertyMap::new();
                obj.insert(
                    "format".into(),
                    native(move |a| list_format_js(&a, &list_type)),
                );
                obj.insert(
                    "formatToParts".into(),
                    native(move |a| list_format_js(&a, &lt)),
                );
                obj.insert(
                    "resolvedOptions".into(),
                    native(|_| {
                        let mut opts = PropertyMap::new();
                        opts.insert("locale".into(), JsValue::String("en-US".into()));
                        opts.insert("type".into(), JsValue::String("conjunction".into()));
                        opts.insert("style".into(), JsValue::String("long".into()));
                        Ok(JsValue::PlainObject(Rc::new(RefCell::new(opts))))
                    }),
                );
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(obj))))
            }),
        );
        ctor.insert("supportedLocalesOf".into(), make_supported_locales_of());
        let mut proto = PropertyMap::new();
        proto.insert(
            "format".into(),
            native(|a| list_format_js(&a, "conjunction")),
        );
        proto.insert(
            "formatToParts".into(),
            native(|a| list_format_js(&a, "conjunction")),
        );
        proto.insert(
            "resolvedOptions".into(),
            native(|_| {
                let mut opts = PropertyMap::new();
                opts.insert("locale".into(), JsValue::String("en-US".into()));
                opts.insert("type".into(), JsValue::String("conjunction".into()));
                opts.insert("style".into(), JsValue::String("long".into()));
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(opts))))
            }),
        );
        ctor.insert(
            "prototype".into(),
            JsValue::PlainObject(Rc::new(RefCell::new(proto))),
        );
        JsValue::PlainObject(Rc::new(RefCell::new(ctor)))
    });

    // ── Intl.RelativeTimeFormat ─────────────────────────────────────────
    ns.insert("RelativeTimeFormat".into(), {
        let mut ctor = PropertyMap::new();
        ctor.insert(
            "__call__".into(),
            native(|_args| {
                let mut obj = PropertyMap::new();
                obj.insert("format".into(), native(|a| relative_time_format_js(&a)));
                obj.insert(
                    "formatToParts".into(),
                    native(|a| relative_time_format_js(&a)),
                );
                obj.insert(
                    "resolvedOptions".into(),
                    native(|_| {
                        let mut opts = PropertyMap::new();
                        opts.insert("locale".into(), JsValue::String("en-US".into()));
                        opts.insert("style".into(), JsValue::String("long".into()));
                        opts.insert("numeric".into(), JsValue::String("always".into()));
                        Ok(JsValue::PlainObject(Rc::new(RefCell::new(opts))))
                    }),
                );
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(obj))))
            }),
        );
        ctor.insert("supportedLocalesOf".into(), make_supported_locales_of());
        let mut proto = PropertyMap::new();
        proto.insert("format".into(), native(|a| relative_time_format_js(&a)));
        proto.insert(
            "formatToParts".into(),
            native(|a| relative_time_format_js(&a)),
        );
        proto.insert(
            "resolvedOptions".into(),
            native(|_| {
                let mut opts = PropertyMap::new();
                opts.insert("locale".into(), JsValue::String("en-US".into()));
                opts.insert("style".into(), JsValue::String("long".into()));
                opts.insert("numeric".into(), JsValue::String("always".into()));
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(opts))))
            }),
        );
        ctor.insert(
            "prototype".into(),
            JsValue::PlainObject(Rc::new(RefCell::new(proto))),
        );
        JsValue::PlainObject(Rc::new(RefCell::new(ctor)))
    });

    // ── Intl.Segmenter ──────────────────────────────────────────────────
    ns.insert("Segmenter".into(), {
        let mut ctor = PropertyMap::new();
        ctor.insert(
            "__call__".into(),
            native(|_args| {
                let mut obj = PropertyMap::new();
                obj.insert(
                    "segment".into(),
                    native(|a| {
                        let s = a.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
                        let segs: Vec<JsValue> = segmenter_segment(&s)
                            .into_iter()
                            .map(|s| JsValue::String(s.into()))
                            .collect();
                        Ok(JsValue::new_array(segs))
                    }),
                );
                obj.insert(
                    "resolvedOptions".into(),
                    native(|_| {
                        let mut opts = PropertyMap::new();
                        opts.insert("locale".into(), JsValue::String("en-US".into()));
                        opts.insert("granularity".into(), JsValue::String("grapheme".into()));
                        Ok(JsValue::PlainObject(Rc::new(RefCell::new(opts))))
                    }),
                );
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(obj))))
            }),
        );
        ctor.insert("supportedLocalesOf".into(), make_supported_locales_of());
        let mut proto = PropertyMap::new();
        proto.insert(
            "segment".into(),
            native(|a| {
                let s = a.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
                let segs: Vec<JsValue> = segmenter_segment(&s)
                    .into_iter()
                    .map(|s| JsValue::String(s.into()))
                    .collect();
                Ok(JsValue::new_array(segs))
            }),
        );
        proto.insert(
            "resolvedOptions".into(),
            native(|_| {
                let mut opts = PropertyMap::new();
                opts.insert("locale".into(), JsValue::String("en-US".into()));
                opts.insert("granularity".into(), JsValue::String("grapheme".into()));
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(opts))))
            }),
        );
        ctor.insert(
            "prototype".into(),
            JsValue::PlainObject(Rc::new(RefCell::new(proto))),
        );
        JsValue::PlainObject(Rc::new(RefCell::new(ctor)))
    });

    // ── Intl.DisplayNames ───────────────────────────────────────────────
    ns.insert("DisplayNames".into(), {
        let mut ctor = PropertyMap::new();
        ctor.insert(
            "__call__".into(),
            native(|_args| {
                let mut obj = PropertyMap::new();
                obj.insert(
                    "of".into(),
                    native(|a| {
                        let code = a.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
                        Ok(JsValue::String(display_names_of(&code).into()))
                    }),
                );
                obj.insert(
                    "resolvedOptions".into(),
                    native(|_| {
                        let mut opts = PropertyMap::new();
                        opts.insert("locale".into(), JsValue::String("en-US".into()));
                        opts.insert("type".into(), JsValue::String("language".into()));
                        opts.insert("style".into(), JsValue::String("long".into()));
                        Ok(JsValue::PlainObject(Rc::new(RefCell::new(opts))))
                    }),
                );
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(obj))))
            }),
        );
        ctor.insert("supportedLocalesOf".into(), make_supported_locales_of());
        let mut proto = PropertyMap::new();
        proto.insert(
            "of".into(),
            native(|a| {
                let code = a.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
                Ok(JsValue::String(display_names_of(&code).into()))
            }),
        );
        proto.insert(
            "resolvedOptions".into(),
            native(|_| {
                let mut opts = PropertyMap::new();
                opts.insert("locale".into(), JsValue::String("en-US".into()));
                opts.insert("type".into(), JsValue::String("language".into()));
                opts.insert("style".into(), JsValue::String("long".into()));
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(opts))))
            }),
        );
        ctor.insert(
            "prototype".into(),
            JsValue::PlainObject(Rc::new(RefCell::new(proto))),
        );
        JsValue::PlainObject(Rc::new(RefCell::new(ctor)))
    });

    // ── Intl.Locale ─────────────────────────────────────────────────────
    ns.insert("Locale".into(), {
        let mut ctor = PropertyMap::new();
        ctor.insert(
            "__call__".into(),
            native(|args| {
                let tag = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
                let mut obj = PropertyMap::new();
                obj.insert(
                    "language".into(),
                    JsValue::String(locale_language(&tag).into()),
                );
                obj.insert(
                    "baseName".into(),
                    JsValue::String(locale_base_name(&tag).into()),
                );
                obj.insert(
                    "toString".into(),
                    native(move |_| Ok(JsValue::String(tag.clone().into()))),
                );
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(obj))))
            }),
        );
        ctor.insert("supportedLocalesOf".into(), make_supported_locales_of());
        JsValue::PlainObject(Rc::new(RefCell::new(ctor)))
    });

    // ── Intl.getCanonicalLocales ────────────────────────────────────────
    ns.insert(
        "getCanonicalLocales".into(),
        native(|args| {
            let locales: Vec<JsValue> = match args.first() {
                Some(JsValue::Array(arr)) => arr
                    .borrow()
                    .iter()
                    .map(|v| match v {
                        JsValue::String(s) => Ok(JsValue::String(s.clone())),
                        other => Ok(JsValue::String(other.to_js_string()?.into())),
                    })
                    .collect::<StatorResult<Vec<_>>>()?,
                Some(JsValue::String(s)) => vec![JsValue::String(s.clone())],
                _ => Vec::new(),
            };
            Ok(JsValue::new_array(locales))
        }),
    );

    // ── Intl.supportedValuesOf ──────────────────────────────────────────
    ns.insert(
        "supportedValuesOf".into(),
        native(|args| {
            let key = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let values: Vec<JsValue> = match key.as_str() {
                "calendar" => vec![JsValue::String("gregory".into())],
                "collation" => vec![JsValue::String("default".into())],
                "currency" => vec![JsValue::String("USD".into())],
                "numberingSystem" => vec![JsValue::String("latn".into())],
                "timeZone" => vec![JsValue::String("UTC".into())],
                "unit" => vec![],
                _ => Vec::new(),
            };
            Ok(JsValue::new_array(values))
        }),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(ns)))
}

// ── Proxy ────────────────────────────────────────────────────────────────────

/// Build the `Proxy` constructor namespace.
///
/// Provides `Proxy(target, handler)` as a callable constructor and
/// `Proxy.revocable(target, handler)` as a static method.
fn make_proxy() -> JsValue {
    let mut props = PropertyMap::new();

    // Proxy as a constructor: new Proxy(target, handler)
    props.insert(
        "__call__".into(),
        native(|args| {
            let target_val = args.first().cloned().unwrap_or(JsValue::Undefined);
            let handler_val = args.get(1).cloned().unwrap_or(JsValue::Undefined);
            // target must be an object-like value; detect callability
            let (target, callable) = match &target_val {
                JsValue::PlainObject(map) => {
                    let mut obj = JsObject::new();
                    for (k, v) in map.borrow().iter() {
                        obj.set_property(k, v.clone()).ok();
                    }
                    (obj, false)
                }
                JsValue::Function(_) | JsValue::NativeFunction(_) => (JsObject::new(), true),
                _ => {
                    return Err(StatorError::TypeError(
                        "Proxy: target must be an object".to_string(),
                    ));
                }
            };
            // Build a ProxyHandler from the handler PlainObject's trap functions
            let handler = build_proxy_handler(&handler_val, &target_val);
            let proxy = if callable {
                proxy_new_callable(target, handler)
            } else {
                proxy_new(target, handler)
            };
            Ok(JsValue::Proxy(Rc::new(RefCell::new(proxy))))
        }),
    );

    // Proxy.revocable(target, handler)
    props.insert(
        "revocable".into(),
        native(|args| {
            let target_val = args.first().cloned().unwrap_or(JsValue::Undefined);
            let handler_val = args.get(1).cloned().unwrap_or(JsValue::Undefined);
            let (target, callable) = match &target_val {
                JsValue::PlainObject(map) => {
                    let mut obj = JsObject::new();
                    for (k, v) in map.borrow().iter() {
                        obj.set_property(k, v.clone()).ok();
                    }
                    (obj, false)
                }
                JsValue::Function(_) | JsValue::NativeFunction(_) => (JsObject::new(), true),
                _ => {
                    return Err(StatorError::TypeError(
                        "Proxy.revocable: target must be an object".to_string(),
                    ));
                }
            };
            let handler = build_proxy_handler(&handler_val, &target_val);
            let proxy = if callable {
                proxy_new_callable(target, handler)
            } else {
                proxy_revocable(target, handler)
            };
            let proxy_rc = Rc::new(RefCell::new(proxy));
            let proxy_val = JsValue::Proxy(Rc::clone(&proxy_rc));
            let revoke_fn = JsValue::NativeFunction(Rc::new(move |_args| {
                proxy_revoke(&mut proxy_rc.borrow_mut());
                Ok(JsValue::Undefined)
            }));
            let mut result = PropertyMap::new();
            result.insert("proxy".to_string(), proxy_val);
            result.insert("revoke".to_string(), revoke_fn);
            Ok(JsValue::PlainObject(Rc::new(RefCell::new(result))))
        }),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

/// Extract proxy handler traps from a JS handler object.
///
/// `target_val` is the original JS target value passed to the Proxy
/// constructor so that handler traps receive the correct `target` argument
/// per ECMAScript §10.5.
fn build_proxy_handler(handler_val: &JsValue, target_val: &JsValue) -> ProxyHandler {
    let mut handler = ProxyHandler::default();
    if let JsValue::PlainObject(map) = handler_val {
        let borrow = map.borrow();

        if let Some(JsValue::NativeFunction(f)) = borrow.get("get").cloned() {
            let target = target_val.clone();
            handler.get = Some(Box::new(move |_target, key| {
                f(vec![
                    target.clone(),
                    JsValue::String(key.to_string().into()),
                ])
            }));
        }
        if let Some(JsValue::NativeFunction(f)) = borrow.get("set").cloned() {
            let target = target_val.clone();
            handler.set = Some(Box::new(move |_target, key, value| {
                let result = f(vec![
                    target.clone(),
                    JsValue::String(key.to_string().into()),
                    value,
                ])?;
                Ok(result.to_boolean())
            }));
        }
        if let Some(JsValue::NativeFunction(f)) = borrow.get("has").cloned() {
            let target = target_val.clone();
            handler.has = Some(Box::new(move |_target, key| {
                let result = f(vec![
                    target.clone(),
                    JsValue::String(key.to_string().into()),
                ])?;
                Ok(result.to_boolean())
            }));
        }
        if let Some(JsValue::NativeFunction(f)) = borrow.get("deleteProperty").cloned() {
            let target = target_val.clone();
            handler.delete_property = Some(Box::new(move |_target, key| {
                let result = f(vec![
                    target.clone(),
                    JsValue::String(key.to_string().into()),
                ])?;
                Ok(result.to_boolean())
            }));
        }
        if let Some(JsValue::NativeFunction(f)) = borrow.get("apply").cloned() {
            let target = target_val.clone();
            handler.apply = Some(Box::new(move |this, args| {
                f(vec![target.clone(), this, JsValue::new_array(args)])
            }));
        }
        if let Some(JsValue::NativeFunction(f)) = borrow.get("construct").cloned() {
            let target = target_val.clone();
            handler.construct = Some(Box::new(move |args| {
                let result = f(vec![target.clone(), JsValue::new_array(args)])?;
                let mut obj = JsObject::new();
                if let JsValue::PlainObject(map) = &result {
                    for (k, v) in map.borrow().iter() {
                        obj.set_property(k, v.clone()).ok();
                    }
                }
                Ok(obj)
            }));
        }
        if let Some(JsValue::NativeFunction(f)) = borrow.get("ownKeys").cloned() {
            let target = target_val.clone();
            handler.own_keys = Some(Box::new(move |_target| {
                let result = f(vec![target.clone()])?;
                if let JsValue::Array(items) = &result {
                    Ok(items
                        .borrow()
                        .iter()
                        .filter_map(|v| match v {
                            JsValue::String(s) => Some(s.to_string()),
                            JsValue::Smi(n) => Some(n.to_string()),
                            _ => None,
                        })
                        .collect())
                } else {
                    Err(StatorError::TypeError(
                        "ownKeys trap must return an array".to_string(),
                    ))
                }
            }));
        }
        if let Some(JsValue::NativeFunction(f)) = borrow.get("defineProperty").cloned() {
            let target = target_val.clone();
            handler.define_property = Some(Box::new(move |_target, key, value, _attrs| {
                let result = f(vec![
                    target.clone(),
                    JsValue::String(key.to_string().into()),
                    value,
                ])?;
                Ok(result.to_boolean())
            }));
        }
        if let Some(JsValue::NativeFunction(f)) = borrow.get("getOwnPropertyDescriptor").cloned() {
            let target = target_val.clone();
            handler.get_own_property_descriptor = Some(Box::new(move |_target, key| {
                let result = f(vec![
                    target.clone(),
                    JsValue::String(key.to_string().into()),
                ])
                .ok()?;
                if result.is_undefined() || result.is_null() {
                    return None;
                }
                if let JsValue::PlainObject(desc) = &result {
                    let desc_borrow = desc.borrow();
                    let value = desc_borrow
                        .get("value")
                        .cloned()
                        .unwrap_or(JsValue::Undefined);
                    let mut attrs = PropertyAttributes::empty();
                    if desc_borrow.get("writable").is_some_and(|v| v.to_boolean()) {
                        attrs |= PropertyAttributes::WRITABLE;
                    }
                    if desc_borrow
                        .get("enumerable")
                        .is_some_and(|v| v.to_boolean())
                    {
                        attrs |= PropertyAttributes::ENUMERABLE;
                    }
                    if desc_borrow
                        .get("configurable")
                        .is_some_and(|v| v.to_boolean())
                    {
                        attrs |= PropertyAttributes::CONFIGURABLE;
                    }
                    Some((value, attrs))
                } else {
                    None
                }
            }));
        }
    }
    handler
}

// ── Reflect ──────────────────────────────────────────────────────────────────

/// Build the `Reflect` namespace object with all 13 static methods.
fn make_reflect() -> JsValue {
    let mut props = PropertyMap::new();

    props.insert(
        "get".into(),
        native(|args| {
            let target = require_object_arg(&args, 0, "Reflect.get")?;
            let key = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(reflect_get(&target, &key))
        }),
    );

    props.insert(
        "set".into(),
        native(|args| {
            let mut target = require_object_arg(&args, 0, "Reflect.set")?;
            let key = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            let value = args.get(2).cloned().unwrap_or(JsValue::Undefined);
            Ok(JsValue::Boolean(reflect_set(&mut target, &key, value)?))
        }),
    );

    props.insert(
        "has".into(),
        native(|args| {
            let target = require_object_arg(&args, 0, "Reflect.has")?;
            let key = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::Boolean(reflect_has(&target, &key)))
        }),
    );

    props.insert(
        "deleteProperty".into(),
        native(|args| {
            let mut target = require_object_arg(&args, 0, "Reflect.deleteProperty")?;
            let key = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::Boolean(reflect_delete_property(
                &mut target,
                &key,
            )?))
        }),
    );

    props.insert(
        "defineProperty".into(),
        native(|args| {
            let mut target = require_object_arg(&args, 0, "Reflect.defineProperty")?;
            let key = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            let value = args.get(2).cloned().unwrap_or(JsValue::Undefined);
            let attrs = PropertyAttributes::WRITABLE
                | PropertyAttributes::ENUMERABLE
                | PropertyAttributes::CONFIGURABLE;
            Ok(JsValue::Boolean(reflect_define_property(
                &mut target,
                &key,
                value,
                attrs,
            )?))
        }),
    );

    props.insert(
        "getOwnPropertyDescriptor".into(),
        native(|args| {
            let target = require_object_arg(&args, 0, "Reflect.getOwnPropertyDescriptor")?;
            let key = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
            match reflect_get_own_property_descriptor(&target, &key) {
                Some((val, attrs)) => {
                    let mut desc = PropertyMap::new();
                    desc.insert("value".to_string(), val);
                    desc.insert(
                        "writable".to_string(),
                        JsValue::Boolean(attrs.contains(PropertyAttributes::WRITABLE)),
                    );
                    desc.insert(
                        "enumerable".to_string(),
                        JsValue::Boolean(attrs.contains(PropertyAttributes::ENUMERABLE)),
                    );
                    desc.insert(
                        "configurable".to_string(),
                        JsValue::Boolean(attrs.contains(PropertyAttributes::CONFIGURABLE)),
                    );
                    Ok(JsValue::PlainObject(Rc::new(RefCell::new(desc))))
                }
                None => Ok(JsValue::Undefined),
            }
        }),
    );

    props.insert(
        "getPrototypeOf".into(),
        native(|args| {
            let target = require_object_arg(&args, 0, "Reflect.getPrototypeOf")?;
            match reflect_get_prototype_of(&target) {
                Some(_) => Ok(JsValue::PlainObject(Rc::new(RefCell::new(
                    PropertyMap::new(),
                )))),
                None => Ok(JsValue::Null),
            }
        }),
    );

    props.insert(
        "setPrototypeOf".into(),
        native(|args| {
            let mut target = require_object_arg(&args, 0, "Reflect.setPrototypeOf")?;
            let proto_arg = args.get(1).cloned().unwrap_or(JsValue::Null);
            let proto = if proto_arg.is_null() {
                None
            } else {
                Some(Rc::new(RefCell::new(JsObject::new())))
            };
            Ok(JsValue::Boolean(reflect_set_prototype_of(
                &mut target,
                proto,
            )))
        }),
    );

    props.insert(
        "isExtensible".into(),
        native(|args| {
            let target = require_object_arg(&args, 0, "Reflect.isExtensible")?;
            Ok(JsValue::Boolean(reflect_is_extensible(&target)))
        }),
    );

    props.insert(
        "preventExtensions".into(),
        native(|args| {
            let mut target = require_object_arg(&args, 0, "Reflect.preventExtensions")?;
            Ok(JsValue::Boolean(reflect_prevent_extensions(&mut target)))
        }),
    );

    props.insert(
        "ownKeys".into(),
        native(|args| {
            let target = require_object_arg(&args, 0, "Reflect.ownKeys")?;
            let keys: Vec<JsValue> = reflect_own_keys(&target)
                .into_iter()
                .map(|s| JsValue::String(s.into()))
                .collect();
            Ok(JsValue::new_array(keys))
        }),
    );

    props.insert(
        "apply".into(),
        native(|args| {
            let _target = args.first().cloned().unwrap_or(JsValue::Undefined);
            let _this_arg = args.get(1).cloned().unwrap_or(JsValue::Undefined);
            let arg_list = match args.get(2) {
                Some(JsValue::Array(arr)) => arr.borrow().clone(),
                _ => vec![],
            };
            match &_target {
                JsValue::NativeFunction(f) => f(arg_list),
                _ => Err(StatorError::TypeError(
                    "Reflect.apply: target is not callable".to_string(),
                )),
            }
        }),
    );

    props.insert(
        "construct".into(),
        native(|args| {
            let target = args.first().cloned().unwrap_or(JsValue::Undefined);
            let arg_list = match args.get(1) {
                Some(JsValue::Array(arr)) => arr.borrow().clone(),
                _ => vec![],
            };
            match &target {
                JsValue::NativeFunction(f) => f(arg_list),
                _ => Err(StatorError::TypeError(
                    "Reflect.construct: target is not a constructor".to_string(),
                )),
            }
        }),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

/// Helper to extract a `JsObject` from the first argument, converting from
/// `PlainObject` when necessary.
fn require_object_arg(args: &[JsValue], idx: usize, name: &str) -> StatorResult<JsObject> {
    match args.get(idx) {
        Some(JsValue::PlainObject(map)) => {
            let mut obj = JsObject::new();
            for (k, v) in map.borrow().iter() {
                obj.set_property(k, v.clone()).ok();
            }
            Ok(obj)
        }
        _ => Err(StatorError::TypeError(format!(
            "{name}: argument is not an object"
        ))),
    }
}

// ── ArrayBuffer / DataView / TypedArray constructors ─────────────────────────

/// Build the `ArrayBuffer` constructor object.
fn make_arraybuffer() -> JsValue {
    let mut props = PropertyMap::new();

    // ArrayBuffer(byteLength)
    props.insert(
        "__call__".into(),
        native(|args| {
            let len = match args.first() {
                Some(v) => crate::builtins::util::checked_f64_to_length(v.to_number()?)?,
                None => 0,
            };
            let buf = arraybuffer_new(len);
            Ok(JsValue::ArrayBuffer(Rc::new(RefCell::new(buf))))
        }),
    );

    // ArrayBuffer.isView(arg)
    props.insert(
        "isView".into(),
        native(|args| {
            let arg = args.first().unwrap_or(&JsValue::Undefined);
            Ok(JsValue::Boolean(arraybuffer_is_view(arg)))
        }),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

/// Build the `DataView` constructor object.
fn make_dataview() -> JsValue {
    let mut props = PropertyMap::new();

    props.insert(
        "__call__".into(),
        native(|args| {
            let buf_rc = match args.first() {
                Some(JsValue::ArrayBuffer(b)) => Rc::clone(b),
                _ => {
                    return Err(StatorError::TypeError(
                        "First argument must be an ArrayBuffer".into(),
                    ));
                }
            };
            let offset = match args.get(1) {
                Some(v) if !v.is_undefined() => {
                    crate::builtins::util::checked_f64_to_length(v.to_number()?)?
                }
                _ => 0,
            };
            let length = match args.get(2) {
                Some(v) if !v.is_undefined() => Some(crate::builtins::util::checked_f64_to_length(
                    v.to_number()?,
                )?),
                _ => None,
            };
            let dv = dataview_new(buf_rc, offset, length)?;
            let inner = Rc::new(RefCell::new(dv));
            let mut obj: HashMap<String, JsValue> = HashMap::new();

            // byteLength
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "byteLength".into(),
                    JsValue::Smi(inner.borrow().byte_length as i32),
                );
            }
            // byteOffset
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "byteOffset".into(),
                    JsValue::Smi(inner.borrow().byte_offset as i32),
                );
            }
            // buffer
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "buffer".into(),
                    JsValue::ArrayBuffer(Rc::clone(&inner.borrow().buffer)),
                );
            }

            // DataView get/set methods helper macro
            macro_rules! dv_getter {
                ($name:expr, $fn_get:expr) => {{
                    let inner = Rc::clone(&inner);
                    obj.insert(
                        $name.into(),
                        native(move |a| {
                            let off = a
                                .first()
                                .map(|v| {
                                    crate::builtins::util::clamped_f64_to_usize(
                                        v.to_number().unwrap_or(0.0),
                                    )
                                })
                                .unwrap_or(0);
                            let le = a.get(1).map(|v| v.to_boolean()).unwrap_or(false);
                            let dv_ref = inner.borrow();
                            let val = $fn_get(&dv_ref, off, le)?;
                            Ok(num_value(val))
                        }),
                    );
                }};
            }

            macro_rules! dv_setter {
                ($name:expr, $fn_set:expr, $conv:expr) => {{
                    let inner = Rc::clone(&inner);
                    obj.insert(
                        $name.into(),
                        native(move |a| {
                            let off = a
                                .first()
                                .map(|v| {
                                    crate::builtins::util::clamped_f64_to_usize(
                                        v.to_number().unwrap_or(0.0),
                                    )
                                })
                                .unwrap_or(0);
                            let raw_val = a.get(1).unwrap_or(&JsValue::Undefined);
                            let le = a.get(2).map(|v| v.to_boolean()).unwrap_or(false);
                            let dv_ref = inner.borrow();
                            $fn_set(&dv_ref, off, $conv(raw_val)?, le)?;
                            Ok(JsValue::Undefined)
                        }),
                    );
                }};
            }

            dv_getter!("getInt8", dataview_get_int8);
            dv_getter!("getUint8", dataview_get_uint8);
            dv_getter!("getInt16", dataview_get_int16);
            dv_getter!("getUint16", dataview_get_uint16);
            dv_getter!("getInt32", dataview_get_int32);
            dv_getter!("getUint32", dataview_get_uint32);
            dv_getter!("getFloat32", dataview_get_float32);
            dv_getter!("getFloat64", dataview_get_float64);

            // BigInt getters return JsValue::BigInt directly.
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "getBigInt64".into(),
                    native(move |a| {
                        let off = a
                            .first()
                            .map(|v| {
                                crate::builtins::util::clamped_f64_to_usize(
                                    v.to_number().unwrap_or(0.0),
                                )
                            })
                            .unwrap_or(0);
                        let le = a.get(1).map(|v| v.to_boolean()).unwrap_or(false);
                        let val = dataview_get_bigint64(&inner.borrow(), off, le)?;
                        Ok(JsValue::BigInt(i128::from(val)))
                    }),
                );
            }
            {
                let inner = Rc::clone(&inner);
                obj.insert(
                    "getBigUint64".into(),
                    native(move |a| {
                        let off = a
                            .first()
                            .map(|v| {
                                crate::builtins::util::clamped_f64_to_usize(
                                    v.to_number().unwrap_or(0.0),
                                )
                            })
                            .unwrap_or(0);
                        let le = a.get(1).map(|v| v.to_boolean()).unwrap_or(false);
                        let val = dataview_get_biguint64(&inner.borrow(), off, le)?;
                        Ok(JsValue::BigInt(i128::from(val)))
                    }),
                );
            }

            dv_setter!("setInt8", dataview_set_int8, |v: &JsValue| Ok::<
                i8,
                StatorError,
            >(
                v.to_int32()? as i8
            ));
            dv_setter!("setUint8", dataview_set_uint8, |v: &JsValue| Ok::<
                u8,
                StatorError,
            >(
                v.to_int32()? as u8
            ));
            dv_setter!("setInt16", dataview_set_int16, |v: &JsValue| Ok::<
                i16,
                StatorError,
            >(
                v.to_int32()? as i16
            ));
            dv_setter!("setUint16", dataview_set_uint16, |v: &JsValue| Ok::<
                u16,
                StatorError,
            >(
                v.to_int32()? as u16
            ));
            dv_setter!("setInt32", dataview_set_int32, |v: &JsValue| v.to_int32());
            dv_setter!("setUint32", dataview_set_uint32, |v: &JsValue| v
                .to_uint32());
            dv_setter!("setFloat32", dataview_set_float32, |v: &JsValue| Ok::<
                f32,
                StatorError,
            >(
                v.to_number()? as f32
            ));
            dv_setter!("setFloat64", dataview_set_float64, |v: &JsValue| v
                .to_number());
            dv_setter!("setBigInt64", dataview_set_bigint64, |v: &JsValue| {
                match v {
                    JsValue::BigInt(n) => Ok::<i64, StatorError>(*n as i64),
                    _ => Ok(v.to_number()? as i64),
                }
            });
            dv_setter!("setBigUint64", dataview_set_biguint64, |v: &JsValue| {
                match v {
                    JsValue::BigInt(n) => Ok::<u64, StatorError>(*n as u64),
                    _ => Ok(v.to_number()? as u64),
                }
            });

            Ok(JsValue::DataView(inner))
        }),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

/// Helper to convert a numeric value to `JsValue`.
fn num_value<T: Into<f64>>(v: T) -> JsValue {
    let f: f64 = v.into();
    if f.fract() == 0.0 && f >= f64::from(i32::MIN) && f <= f64::from(i32::MAX) {
        JsValue::Smi(f as i32)
    } else {
        JsValue::HeapNumber(f)
    }
}

/// Build a typed-array constructor for the given `TypedArrayKind`.
fn make_typed_array_constructor(kind: TypedArrayKind) -> JsValue {
    let mut props = PropertyMap::new();

    // BYTES_PER_ELEMENT
    props.insert(
        "BYTES_PER_ELEMENT".into(),
        JsValue::Smi(kind.bytes_per_element() as i32),
    );

    // Constructor: TypedArray(length) | TypedArray(array) | TypedArray(buffer, offset?, length?)
    props.insert(
        "__call__".into(),
        native(move |args| {
            let ta = match args.first() {
                // From ArrayBuffer
                Some(JsValue::ArrayBuffer(buf)) => {
                    let offset = match args.get(1) {
                        Some(v) if !v.is_undefined() => {
                            crate::builtins::util::checked_f64_to_length(v.to_number()?)?
                        }
                        _ => 0,
                    };
                    let length = match args.get(2) {
                        Some(v) if !v.is_undefined() => Some(
                            crate::builtins::util::checked_f64_to_length(v.to_number()?)?,
                        ),
                        _ => None,
                    };
                    typed_array_new_from_buffer(kind, Rc::clone(buf), offset, length)?
                }
                // From another TypedArray
                Some(JsValue::TypedArray(src_rc)) => {
                    let src = src_rc.borrow();
                    let vals: Vec<JsValue> =
                        (0..src.length).map(|i| typed_array_get(&src, i)).collect();
                    typed_array_from_values(kind, &vals)?
                }
                // From Array
                Some(JsValue::Array(arr)) => typed_array_from_values(kind, &arr.borrow())?,
                // From length (number)
                Some(v) => {
                    let len = crate::builtins::util::checked_f64_to_length(v.to_number()?)?;
                    typed_array_new_from_length(kind, len)
                }
                None => typed_array_new_from_length(kind, 0),
            };
            let inner = Rc::new(RefCell::new(ta));
            Ok(make_typed_array_instance(kind, inner))
        }),
    );

    // TypedArray.from(source)
    props.insert(
        "from".into(),
        native(move |args| {
            let source = match args.first() {
                Some(JsValue::Array(arr)) => arr.borrow().clone(),
                _ => Vec::new(),
            };
            let ta = typed_array_from_values(kind, &source)?;
            let inner = Rc::new(RefCell::new(ta));
            Ok(make_typed_array_instance(kind, inner))
        }),
    );

    // TypedArray.of(...items)
    props.insert(
        "of".into(),
        native(move |args| {
            let ta = typed_array_from_values(kind, &args)?;
            let inner = Rc::new(RefCell::new(ta));
            Ok(make_typed_array_instance(kind, inner))
        }),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

/// Build the prototype methods for a `JsValue::TypedArray` instance.
fn make_typed_array_instance(
    kind: TypedArrayKind,
    inner: Rc<RefCell<crate::builtins::typed_array::JsTypedArray>>,
) -> JsValue {
    let _ = kind;
    let typed_array_val = JsValue::TypedArray(Rc::clone(&inner));
    let mut obj = PropertyMap::new();

    // BYTES_PER_ELEMENT
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "BYTES_PER_ELEMENT".into(),
            JsValue::Smi(inner.borrow().kind.bytes_per_element() as i32),
        );
    }
    // length
    {
        let inner = Rc::clone(&inner);
        obj.insert("length".into(), JsValue::Smi(inner.borrow().length as i32));
    }
    // byteLength
    {
        let inner = Rc::clone(&inner);
        let ta = inner.borrow();
        obj.insert(
            "byteLength".into(),
            JsValue::Smi((ta.length * ta.kind.bytes_per_element()) as i32),
        );
    }
    // byteOffset
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "byteOffset".into(),
            JsValue::Smi(inner.borrow().byte_offset as i32),
        );
    }
    // buffer
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "buffer".into(),
            JsValue::ArrayBuffer(Rc::clone(&inner.borrow().buffer)),
        );
    }
    // at(index)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "at".into(),
            native(move |a| {
                let idx = a
                    .first()
                    .map(|v| v.to_number().unwrap_or(0.0) as i64)
                    .unwrap_or(0);
                Ok(typed_array_at(&inner.borrow(), idx))
            }),
        );
    }
    // copyWithin(target, start, end?)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "copyWithin".into(),
            native(move |a| {
                let target = a
                    .first()
                    .map(|v| v.to_number().unwrap_or(0.0) as i64)
                    .unwrap_or(0);
                let start = a
                    .get(1)
                    .map(|v| v.to_number().unwrap_or(0.0) as i64)
                    .unwrap_or(0);
                let end = a
                    .get(2)
                    .map(|v| v.to_number().unwrap_or(inner.borrow().length as f64) as i64)
                    .unwrap_or(inner.borrow().length as i64);
                typed_array_copy_within(&inner.borrow(), target, start, end);
                Ok(JsValue::Undefined)
            }),
        );
    }
    // entries()
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "entries".into(),
            native(move |_| {
                let items = typed_array_entries(&inner.borrow());
                Ok(JsValue::new_array(items))
            }),
        );
    }
    // every(callbackfn)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "every".into(),
            native(move |a| {
                let cb = a.first().cloned().unwrap_or(JsValue::Undefined);
                let ta = inner.borrow();
                for i in 0..ta.length {
                    let v = typed_array_get(&ta, i);
                    if let JsValue::NativeFunction(f) = &cb {
                        let result = f(vec![v, JsValue::Smi(i as i32)])?;
                        if !result.to_boolean() {
                            return Ok(JsValue::Boolean(false));
                        }
                    }
                }
                Ok(JsValue::Boolean(true))
            }),
        );
    }
    // fill(value, start?, end?)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "fill".into(),
            native(move |a| {
                let val = a.first().unwrap_or(&JsValue::Undefined).clone();
                let ta = inner.borrow();
                let start = a
                    .get(1)
                    .map(|v| v.to_number().unwrap_or(0.0) as i64)
                    .unwrap_or(0);
                let end = a
                    .get(2)
                    .map(|v| v.to_number().unwrap_or(ta.length as f64) as i64)
                    .unwrap_or(ta.length as i64);
                typed_array_fill(&ta, &val, start, end)?;
                Ok(JsValue::Undefined)
            }),
        );
    }
    // filter(callbackfn)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "filter".into(),
            native(move |a| {
                let cb = a.first().cloned().unwrap_or(JsValue::Undefined);
                let ta = inner.borrow();
                let mut kept = Vec::new();
                for i in 0..ta.length {
                    let v = typed_array_get(&ta, i);
                    if let JsValue::NativeFunction(f) = &cb {
                        let result = f(vec![v.clone(), JsValue::Smi(i as i32)])?;
                        if result.to_boolean() {
                            kept.push(v);
                        }
                    }
                }
                let result = typed_array_from_values(ta.kind, &kept)?;
                Ok(JsValue::TypedArray(Rc::new(RefCell::new(result))))
            }),
        );
    }
    // find(callbackfn)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "find".into(),
            native(move |a| {
                let cb = a.first().cloned().unwrap_or(JsValue::Undefined);
                let ta = inner.borrow();
                for i in 0..ta.length {
                    let v = typed_array_get(&ta, i);
                    if let JsValue::NativeFunction(f) = &cb {
                        let result = f(vec![v.clone(), JsValue::Smi(i as i32)])?;
                        if result.to_boolean() {
                            return Ok(v);
                        }
                    }
                }
                Ok(JsValue::Undefined)
            }),
        );
    }
    // findIndex(callbackfn)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "findIndex".into(),
            native(move |a| {
                let cb = a.first().cloned().unwrap_or(JsValue::Undefined);
                let ta = inner.borrow();
                for i in 0..ta.length {
                    let v = typed_array_get(&ta, i);
                    if let JsValue::NativeFunction(f) = &cb {
                        let result = f(vec![v, JsValue::Smi(i as i32)])?;
                        if result.to_boolean() {
                            return Ok(JsValue::Smi(i as i32));
                        }
                    }
                }
                Ok(JsValue::Smi(-1))
            }),
        );
    }
    // findLast(callbackfn)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "findLast".into(),
            native(move |a| {
                let cb = a.first().cloned().unwrap_or(JsValue::Undefined);
                let ta = inner.borrow();
                for i in (0..ta.length).rev() {
                    let v = typed_array_get(&ta, i);
                    if let JsValue::NativeFunction(f) = &cb {
                        let result = f(vec![v.clone(), JsValue::Smi(i as i32)])?;
                        if result.to_boolean() {
                            return Ok(v);
                        }
                    }
                }
                Ok(JsValue::Undefined)
            }),
        );
    }
    // findLastIndex(callbackfn)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "findLastIndex".into(),
            native(move |a| {
                let cb = a.first().cloned().unwrap_or(JsValue::Undefined);
                let ta = inner.borrow();
                for i in (0..ta.length).rev() {
                    let v = typed_array_get(&ta, i);
                    if let JsValue::NativeFunction(f) = &cb {
                        let result = f(vec![v, JsValue::Smi(i as i32)])?;
                        if result.to_boolean() {
                            return Ok(JsValue::Smi(i as i32));
                        }
                    }
                }
                Ok(JsValue::Smi(-1))
            }),
        );
    }
    // forEach(callbackfn)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "forEach".into(),
            native(move |a| {
                let cb = a.first().cloned().unwrap_or(JsValue::Undefined);
                let ta = inner.borrow();
                for i in 0..ta.length {
                    let v = typed_array_get(&ta, i);
                    if let JsValue::NativeFunction(f) = &cb {
                        f(vec![v, JsValue::Smi(i as i32)])?;
                    }
                }
                Ok(JsValue::Undefined)
            }),
        );
    }
    // includes(searchElement, fromIndex?)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "includes".into(),
            native(move |a| {
                let search = a.first().unwrap_or(&JsValue::Undefined);
                let from = a
                    .get(1)
                    .map(|v| v.to_number().unwrap_or(0.0) as i64)
                    .unwrap_or(0);
                Ok(JsValue::Boolean(typed_array_includes(
                    &inner.borrow(),
                    search,
                    from,
                )))
            }),
        );
    }
    // indexOf(searchElement, fromIndex?)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "indexOf".into(),
            native(move |a| {
                let search = a.first().unwrap_or(&JsValue::Undefined);
                let from = a
                    .get(1)
                    .map(|v| v.to_number().unwrap_or(0.0) as i64)
                    .unwrap_or(0);
                Ok(JsValue::Smi(
                    typed_array_index_of(&inner.borrow(), search, from) as i32,
                ))
            }),
        );
    }
    // join(separator?)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "join".into(),
            native(move |a| {
                let sep = match a.first() {
                    Some(v) if !v.is_undefined() => v.to_js_string()?,
                    _ => ",".to_string(),
                };
                Ok(JsValue::String(
                    typed_array_join(&inner.borrow(), &sep)?.into(),
                ))
            }),
        );
    }
    // keys()
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "keys".into(),
            native(move |_| {
                let items = typed_array_keys(&inner.borrow());
                Ok(JsValue::new_array(items))
            }),
        );
    }
    // lastIndexOf(searchElement, fromIndex?)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "lastIndexOf".into(),
            native(move |a| {
                let search = a.first().unwrap_or(&JsValue::Undefined);
                let ta = inner.borrow();
                let from = a
                    .get(1)
                    .map(|v| v.to_number().unwrap_or(ta.length as f64 - 1.0) as i64)
                    .unwrap_or(ta.length as i64 - 1);
                Ok(JsValue::Smi(
                    typed_array_last_index_of(&ta, search, from) as i32
                ))
            }),
        );
    }
    // map(callbackfn)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "map".into(),
            native(move |a| {
                let cb = a.first().cloned().unwrap_or(JsValue::Undefined);
                let ta = inner.borrow();
                let mut mapped = Vec::with_capacity(ta.length);
                for i in 0..ta.length {
                    let v = typed_array_get(&ta, i);
                    if let JsValue::NativeFunction(f) = &cb {
                        mapped.push(f(vec![v, JsValue::Smi(i as i32)])?);
                    } else {
                        mapped.push(v);
                    }
                }
                let result = typed_array_from_values(ta.kind, &mapped)?;
                Ok(JsValue::TypedArray(Rc::new(RefCell::new(result))))
            }),
        );
    }
    // reduce(callbackfn, initialValue?)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "reduce".into(),
            native(move |a| {
                let cb = a.first().cloned().unwrap_or(JsValue::Undefined);
                let ta = inner.borrow();
                let mut start = 0;
                let mut acc = if a.len() > 1 {
                    a[1].clone()
                } else {
                    if ta.length == 0 {
                        return Err(StatorError::TypeError(
                            "Reduce of empty array with no initial value".into(),
                        ));
                    }
                    start = 1;
                    typed_array_get(&ta, 0)
                };
                for i in start..ta.length {
                    let v = typed_array_get(&ta, i);
                    if let JsValue::NativeFunction(f) = &cb {
                        acc = f(vec![acc, v, JsValue::Smi(i as i32)])?;
                    }
                }
                Ok(acc)
            }),
        );
    }
    // reduceRight(callbackfn, initialValue?)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "reduceRight".into(),
            native(move |a| {
                let cb = a.first().cloned().unwrap_or(JsValue::Undefined);
                let ta = inner.borrow();
                let len = ta.length;
                let mut acc = if a.len() > 1 {
                    a[1].clone()
                } else {
                    if len == 0 {
                        return Err(StatorError::TypeError(
                            "Reduce of empty array with no initial value".into(),
                        ));
                    }
                    typed_array_get(&ta, len - 1)
                };
                let end = if a.len() <= 1 && len > 0 {
                    len - 1
                } else {
                    len
                };
                for i in (0..end).rev() {
                    let v = typed_array_get(&ta, i);
                    if let JsValue::NativeFunction(f) = &cb {
                        acc = f(vec![acc, v, JsValue::Smi(i as i32)])?;
                    }
                }
                Ok(acc)
            }),
        );
    }
    // reverse()
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "reverse".into(),
            native(move |_| {
                typed_array_reverse(&inner.borrow());
                Ok(JsValue::Undefined)
            }),
        );
    }
    // set(source, offset?)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "set".into(),
            native(move |a| {
                let source: Vec<JsValue> = match a.first() {
                    Some(JsValue::Array(arr)) => arr.borrow().clone(),
                    Some(JsValue::TypedArray(src_rc)) => {
                        let src = src_rc.borrow();
                        (0..src.length).map(|i| typed_array_get(&src, i)).collect()
                    }
                    _ => Vec::new(),
                };
                let offset = a
                    .get(1)
                    .map(|v| {
                        crate::builtins::util::clamped_f64_to_usize(v.to_number().unwrap_or(0.0))
                    })
                    .unwrap_or(0);
                typed_array_set_from(&inner.borrow(), &source, offset)?;
                Ok(JsValue::Undefined)
            }),
        );
    }
    // slice(start?, end?)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "slice".into(),
            native(move |a| {
                let ta = inner.borrow();
                let start = a
                    .first()
                    .map(|v| v.to_number().unwrap_or(0.0) as i64)
                    .unwrap_or(0);
                let end = a
                    .get(1)
                    .map(|v| v.to_number().unwrap_or(ta.length as f64) as i64)
                    .unwrap_or(ta.length as i64);
                let result = typed_array_slice(&ta, start, end)?;
                Ok(JsValue::TypedArray(Rc::new(RefCell::new(result))))
            }),
        );
    }
    // some(callbackfn)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "some".into(),
            native(move |a| {
                let cb = a.first().cloned().unwrap_or(JsValue::Undefined);
                let ta = inner.borrow();
                for i in 0..ta.length {
                    let v = typed_array_get(&ta, i);
                    if let JsValue::NativeFunction(f) = &cb {
                        let result = f(vec![v, JsValue::Smi(i as i32)])?;
                        if result.to_boolean() {
                            return Ok(JsValue::Boolean(true));
                        }
                    }
                }
                Ok(JsValue::Boolean(false))
            }),
        );
    }
    // sort(comparefn?)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "sort".into(),
            native(move |_| {
                typed_array_sort(&inner.borrow());
                Ok(JsValue::Undefined)
            }),
        );
    }
    // subarray(begin?, end?)
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "subarray".into(),
            native(move |a| {
                let ta = inner.borrow();
                let begin = a
                    .first()
                    .map(|v| v.to_number().unwrap_or(0.0) as i64)
                    .unwrap_or(0);
                let end = a
                    .get(1)
                    .map(|v| v.to_number().unwrap_or(ta.length as f64) as i64)
                    .unwrap_or(ta.length as i64);
                let sub = typed_array_subarray(&ta, begin, end);
                let sub_inner = Rc::new(RefCell::new(sub));
                Ok(make_typed_array_instance(ta.kind, sub_inner))
            }),
        );
    }
    // toLocaleString()
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "toLocaleString".into(),
            native(move |_| {
                Ok(JsValue::String(
                    typed_array_join(&inner.borrow(), ",")?.into(),
                ))
            }),
        );
    }
    // toString()
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "toString".into(),
            native(move |_| {
                Ok(JsValue::String(
                    typed_array_join(&inner.borrow(), ",")?.into(),
                ))
            }),
        );
    }
    // values()
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "values".into(),
            native(move |_| {
                let items = typed_array_values(&inner.borrow());
                Ok(JsValue::new_array(items))
            }),
        );
    }
    // @@iterator — same as values()
    {
        let inner = Rc::clone(&inner);
        obj.insert(
            "@@iterator".into(),
            native(move |_| {
                let items = typed_array_values(&inner.borrow());
                Ok(JsValue::new_array(items))
            }),
        );
    }
    // Store the TypedArray value for identity purposes.
    obj.insert("__typed_array__".into(), typed_array_val);

    JsValue::PlainObject(Rc::new(RefCell::new(obj)))
}

// ── install_globals ──────────────────────────────────────────────────────────

/// Build the `ShadowRealm` constructor.
///
/// Each `new ShadowRealm()` creates an isolated evaluation environment with
/// its own set of global builtins.  `evaluate(code)` parses and runs code
/// inside the realm.
fn make_shadow_realm() -> JsValue {
    use crate::bytecode::bytecode_generator::BytecodeGenerator;
    use crate::interpreter::{Interpreter, InterpreterFrame};
    native(|_args| {
        // Create a fresh set of globals for this realm.
        let mut realm_globals = HashMap::new();
        install_globals(&mut realm_globals);
        let realm_globals = Rc::new(RefCell::new(realm_globals));

        let mut props = PropertyMap::new();

        // evaluate(sourceText)
        {
            let globals = Rc::clone(&realm_globals);
            props.insert(
                "evaluate".into(),
                native(move |args| {
                    let source = args
                        .first()
                        .map(|v| v.to_js_string())
                        .transpose()?
                        .unwrap_or_default();
                    let program = crate::parser::recursive_descent::parse(&source)?;
                    let bc = BytecodeGenerator::compile_program(&program)?;
                    let mut frame =
                        InterpreterFrame::new_with_globals(bc, vec![], Rc::clone(&globals));
                    let result = Interpreter::run(&mut frame)?;
                    // ShadowRealm boundary: only primitives pass through.
                    match &result {
                        JsValue::Smi(_)
                        | JsValue::HeapNumber(_)
                        | JsValue::String(_)
                        | JsValue::Boolean(_)
                        | JsValue::Null
                        | JsValue::Undefined
                        | JsValue::BigInt(_)
                        | JsValue::Symbol(_) => Ok(result),
                        _ => Err(StatorError::TypeError(
                            "ShadowRealm evaluate: cannot pass non-primitive across realm boundary"
                                .into(),
                        )),
                    }
                }),
            );
        }

        // importValue(specifier, exportName) — stub returning undefined
        props.insert("importValue".into(), native(|_args| Ok(JsValue::Undefined)));

        Ok(JsValue::PlainObject(Rc::new(RefCell::new(props))))
    })
}

/// Build the `SharedArrayBuffer` constructor.
fn make_shared_arraybuffer() -> JsValue {
    native(|args| {
        let byte_length = match args.first() {
            Some(v) => crate::builtins::util::checked_f64_to_length(v.to_number()?)?,
            None => 0,
        };
        let buf = arraybuffer_new(byte_length);
        let mut props = PropertyMap::new();
        let inner = Rc::new(RefCell::new(buf));
        {
            let inner2 = Rc::clone(&inner);
            props.insert(
                "byteLength".into(),
                native(move |_| Ok(JsValue::Smi(inner2.borrow().data.len() as i32))),
            );
        }
        props.insert(
            "slice".into(),
            native({
                let inner2 = Rc::clone(&inner);
                move |args| {
                    let len = inner2.borrow().data.len();
                    let start = args
                        .first()
                        .map(|v| v.to_number().unwrap_or(0.0) as i64)
                        .unwrap_or(0);
                    let end = args
                        .get(1)
                        .map(|v| v.to_number().unwrap_or(len as f64) as i64)
                        .unwrap_or(len as i64);
                    let s = start.max(0) as usize;
                    let e = end.clamp(0, len as i64) as usize;
                    let data = inner2.borrow().data[s..e].to_vec();
                    Ok(JsValue::ArrayBuffer(Rc::new(RefCell::new(JsArrayBuffer {
                        data,
                    }))))
                }
            }),
        );
        props.insert("__buffer__".into(), JsValue::ArrayBuffer(Rc::clone(&inner)));
        Ok(JsValue::PlainObject(Rc::new(RefCell::new(props))))
    })
}

/// Helper: read an integer value from a buffer at a given byte offset.
fn atomics_read(buf: &[u8], byte_offset: usize, kind: TypedArrayKind) -> JsValue {
    let bpe = kind.bytes_per_element();
    if byte_offset + bpe > buf.len() {
        return JsValue::Undefined;
    }
    let slice = &buf[byte_offset..byte_offset + bpe];
    match kind {
        TypedArrayKind::Int8 => JsValue::Smi(i8::from_ne_bytes([slice[0]]) as i32),
        TypedArrayKind::Uint8 | TypedArrayKind::Uint8Clamped => JsValue::Smi(slice[0] as i32),
        TypedArrayKind::Int16 => JsValue::Smi(i16::from_ne_bytes([slice[0], slice[1]]) as i32),
        TypedArrayKind::Uint16 => JsValue::Smi(u16::from_ne_bytes([slice[0], slice[1]]) as i32),
        TypedArrayKind::Int32 => {
            JsValue::Smi(i32::from_ne_bytes([slice[0], slice[1], slice[2], slice[3]]))
        }
        TypedArrayKind::Uint32 => {
            let v = u32::from_ne_bytes([slice[0], slice[1], slice[2], slice[3]]);
            JsValue::HeapNumber(v as f64)
        }
        TypedArrayKind::Float32 => {
            JsValue::HeapNumber(f32::from_ne_bytes([slice[0], slice[1], slice[2], slice[3]]) as f64)
        }
        TypedArrayKind::Float64 => {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(slice);
            JsValue::HeapNumber(f64::from_ne_bytes(bytes))
        }
        TypedArrayKind::BigInt64 => {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(slice);
            JsValue::HeapNumber(i64::from_ne_bytes(bytes) as f64)
        }
        TypedArrayKind::BigUint64 => {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(slice);
            JsValue::HeapNumber(u64::from_ne_bytes(bytes) as f64)
        }
    }
}

/// Helper: write a numeric JS value into a buffer at a given byte offset.
fn atomics_write(buf: &mut [u8], byte_offset: usize, kind: TypedArrayKind, val: f64) {
    let bpe = kind.bytes_per_element();
    if byte_offset + bpe > buf.len() {
        return;
    }
    let slice = &mut buf[byte_offset..byte_offset + bpe];
    match kind {
        TypedArrayKind::Int8 => slice[0] = (val as i8) as u8,
        TypedArrayKind::Uint8 | TypedArrayKind::Uint8Clamped => slice[0] = val as u8,
        TypedArrayKind::Int16 => slice.copy_from_slice(&(val as i16).to_ne_bytes()),
        TypedArrayKind::Uint16 => slice.copy_from_slice(&(val as u16).to_ne_bytes()),
        TypedArrayKind::Int32 => slice.copy_from_slice(&(val as i32).to_ne_bytes()),
        TypedArrayKind::Uint32 => slice.copy_from_slice(&(val as u32).to_ne_bytes()),
        TypedArrayKind::Float32 => slice.copy_from_slice(&(val as f32).to_ne_bytes()),
        TypedArrayKind::Float64 => slice.copy_from_slice(&val.to_ne_bytes()),
        TypedArrayKind::BigInt64 => slice.copy_from_slice(&(val as i64).to_ne_bytes()),
        TypedArrayKind::BigUint64 => slice.copy_from_slice(&(val as u64).to_ne_bytes()),
    }
}

/// Extract the buffer, kind, and element index from the first two Atomics args.
fn atomics_extract_ta(
    args: &[JsValue],
) -> StatorResult<(Rc<RefCell<JsArrayBuffer>>, TypedArrayKind, usize)> {
    let ta = args.first().ok_or_else(|| {
        StatorError::TypeError("Atomics: first argument must be a TypedArray".into())
    })?;
    let (buf, kind) = match ta {
        JsValue::TypedArray(inner) => {
            let inner_ref = inner.borrow();
            (Rc::clone(&inner_ref.buffer), inner_ref.kind)
        }
        _ => {
            return Err(StatorError::TypeError(
                "Atomics: first argument must be a TypedArray".into(),
            ));
        }
    };
    let idx = args
        .get(1)
        .map(|v| crate::builtins::util::clamped_f64_to_usize(v.to_number().unwrap_or(0.0)))
        .unwrap_or(0);
    let byte_offset = idx * kind.bytes_per_element();
    Ok((buf, kind, byte_offset))
}

/// Build the `Atomics` namespace object.
///
/// Since stator is single-threaded, all atomic operations are plain
/// reads/writes — sequentially consistent by default.
fn make_atomics() -> JsValue {
    let mut props = PropertyMap::new();

    // Atomics.load(typedArray, index)
    props.insert(
        "load".into(),
        native(|args| {
            let (buf, kind, byte_offset) = atomics_extract_ta(&args)?;
            Ok(atomics_read(&buf.borrow().data, byte_offset, kind))
        }),
    );

    // Atomics.store(typedArray, index, value)
    props.insert(
        "store".into(),
        native(|args| {
            let (buf, kind, byte_offset) = atomics_extract_ta(&args)?;
            let val = args
                .get(2)
                .map(|v| v.to_number().unwrap_or(0.0))
                .unwrap_or(0.0);
            atomics_write(&mut buf.borrow_mut().data, byte_offset, kind, val);
            Ok(JsValue::HeapNumber(val))
        }),
    );

    // Atomics.add(typedArray, index, value)
    props.insert(
        "add".into(),
        native(|args| {
            let (buf, kind, byte_offset) = atomics_extract_ta(&args)?;
            let old = atomics_read(&buf.borrow().data, byte_offset, kind);
            let old_num = old.to_number()?;
            let val = args
                .get(2)
                .map(|v| v.to_number().unwrap_or(0.0))
                .unwrap_or(0.0);
            atomics_write(&mut buf.borrow_mut().data, byte_offset, kind, old_num + val);
            Ok(old)
        }),
    );

    // Atomics.sub(typedArray, index, value)
    props.insert(
        "sub".into(),
        native(|args| {
            let (buf, kind, byte_offset) = atomics_extract_ta(&args)?;
            let old = atomics_read(&buf.borrow().data, byte_offset, kind);
            let old_num = old.to_number()?;
            let val = args
                .get(2)
                .map(|v| v.to_number().unwrap_or(0.0))
                .unwrap_or(0.0);
            atomics_write(&mut buf.borrow_mut().data, byte_offset, kind, old_num - val);
            Ok(old)
        }),
    );

    // Atomics.and(typedArray, index, value)
    props.insert(
        "and".into(),
        native(|args| {
            let (buf, kind, byte_offset) = atomics_extract_ta(&args)?;
            let old = atomics_read(&buf.borrow().data, byte_offset, kind);
            let old_i = old.to_number()? as i64;
            let val = args
                .get(2)
                .map(|v| v.to_number().unwrap_or(0.0) as i64)
                .unwrap_or(0);
            atomics_write(
                &mut buf.borrow_mut().data,
                byte_offset,
                kind,
                (old_i & val) as f64,
            );
            Ok(old)
        }),
    );

    // Atomics.or(typedArray, index, value)
    props.insert(
        "or".into(),
        native(|args| {
            let (buf, kind, byte_offset) = atomics_extract_ta(&args)?;
            let old = atomics_read(&buf.borrow().data, byte_offset, kind);
            let old_i = old.to_number()? as i64;
            let val = args
                .get(2)
                .map(|v| v.to_number().unwrap_or(0.0) as i64)
                .unwrap_or(0);
            atomics_write(
                &mut buf.borrow_mut().data,
                byte_offset,
                kind,
                (old_i | val) as f64,
            );
            Ok(old)
        }),
    );

    // Atomics.xor(typedArray, index, value)
    props.insert(
        "xor".into(),
        native(|args| {
            let (buf, kind, byte_offset) = atomics_extract_ta(&args)?;
            let old = atomics_read(&buf.borrow().data, byte_offset, kind);
            let old_i = old.to_number()? as i64;
            let val = args
                .get(2)
                .map(|v| v.to_number().unwrap_or(0.0) as i64)
                .unwrap_or(0);
            atomics_write(
                &mut buf.borrow_mut().data,
                byte_offset,
                kind,
                (old_i ^ val) as f64,
            );
            Ok(old)
        }),
    );

    // Atomics.exchange(typedArray, index, value)
    props.insert(
        "exchange".into(),
        native(|args| {
            let (buf, kind, byte_offset) = atomics_extract_ta(&args)?;
            let old = atomics_read(&buf.borrow().data, byte_offset, kind);
            let val = args
                .get(2)
                .map(|v| v.to_number().unwrap_or(0.0))
                .unwrap_or(0.0);
            atomics_write(&mut buf.borrow_mut().data, byte_offset, kind, val);
            Ok(old)
        }),
    );

    // Atomics.compareExchange(typedArray, index, expected, replacement)
    props.insert(
        "compareExchange".into(),
        native(|args| {
            let (buf, kind, byte_offset) = atomics_extract_ta(&args)?;
            let old = atomics_read(&buf.borrow().data, byte_offset, kind);
            let old_num = old.to_number()?;
            let expected = args
                .get(2)
                .map(|v| v.to_number().unwrap_or(0.0))
                .unwrap_or(0.0);
            if (old_num - expected).abs() < f64::EPSILON {
                let replacement = args
                    .get(3)
                    .map(|v| v.to_number().unwrap_or(0.0))
                    .unwrap_or(0.0);
                atomics_write(&mut buf.borrow_mut().data, byte_offset, kind, replacement);
            }
            Ok(old)
        }),
    );

    // Atomics.isLockFree(size)
    props.insert(
        "isLockFree".into(),
        native(|args| {
            let size = args
                .first()
                .map(|v| crate::builtins::util::clamped_f64_to_usize(v.to_number().unwrap_or(0.0)))
                .unwrap_or(0);
            Ok(JsValue::Boolean(matches!(size, 1 | 2 | 4 | 8)))
        }),
    );

    // Atomics.wait — single-threaded, always returns "not-equal" or "timed-out"
    props.insert(
        "wait".into(),
        native(|args| {
            let (buf, kind, byte_offset) = atomics_extract_ta(&args)?;
            let current = atomics_read(&buf.borrow().data, byte_offset, kind);
            let expected = args
                .get(2)
                .map(|v| v.to_number().unwrap_or(0.0))
                .unwrap_or(0.0);
            let current_num = current.to_number()?;
            if (current_num - expected).abs() >= f64::EPSILON {
                Ok(JsValue::String("not-equal".into()))
            } else {
                Ok(JsValue::String("timed-out".into()))
            }
        }),
    );

    // Atomics.notify — single-threaded no-op, returns 0
    props.insert("notify".into(), native(|_args| Ok(JsValue::Smi(0))));

    // Atomics.waitAsync — returns { async: false, value: "timed-out" }
    props.insert(
        "waitAsync".into(),
        native(|_args| {
            let mut props = PropertyMap::new();
            props.insert("async".into(), JsValue::Boolean(false));
            props.insert("value".into(), JsValue::String("timed-out".into()));
            Ok(JsValue::PlainObject(Rc::new(RefCell::new(props))))
        }),
    );

    // @@toStringTag
    props.insert("@@toStringTag".into(), JsValue::String("Atomics".into()));

    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

/// Build the `DisposableStack` constructor.
///
/// `DisposableStack` manages a stack of disposable resources and calls their
/// `[Symbol.dispose]()` methods in reverse order when `.dispose()` is called.
fn make_disposable_stack() -> JsValue {
    native(|_args| {
        let resources: Rc<RefCell<Vec<JsValue>>> = Rc::new(RefCell::new(Vec::new()));
        let disposed: Rc<RefCell<bool>> = Rc::new(RefCell::new(false));
        let mut props = PropertyMap::new();

        // use(value) — add a disposable resource.
        {
            let res = Rc::clone(&resources);
            props.insert(
                "use".into(),
                native(move |args| {
                    let val = args.first().cloned().unwrap_or(JsValue::Undefined);
                    res.borrow_mut().push(val.clone());
                    Ok(val)
                }),
            );
        }

        // disposed — whether the stack has been disposed.
        {
            let d = Rc::clone(&disposed);
            props.insert(
                "disposed".into(),
                native(move |_args| Ok(JsValue::Boolean(*d.borrow()))),
            );
        }

        // dispose() — dispose all resources in reverse order.
        {
            let res = Rc::clone(&resources);
            let d = Rc::clone(&disposed);
            props.insert(
                "dispose".into(),
                native(move |_args| {
                    if *d.borrow() {
                        return Ok(JsValue::Undefined);
                    }
                    *d.borrow_mut() = true;
                    let items: Vec<JsValue> = res.borrow_mut().drain(..).rev().collect();
                    for item in &items {
                        if let JsValue::PlainObject(obj) = item
                            && let Some(JsValue::NativeFunction(f)) =
                                obj.borrow().get("@@dispose").cloned()
                        {
                            let _ = f(vec![item.clone()]);
                        }
                    }
                    Ok(JsValue::Undefined)
                }),
            );
        }

        // [Symbol.dispose]() — aliases dispose() for using protocol.
        {
            let res2 = Rc::clone(&resources);
            let d2 = Rc::clone(&disposed);
            props.insert(
                "@@dispose".into(),
                native(move |_args| {
                    if *d2.borrow() {
                        return Ok(JsValue::Undefined);
                    }
                    *d2.borrow_mut() = true;
                    let items: Vec<JsValue> = res2.borrow_mut().drain(..).rev().collect();
                    for item in &items {
                        if let JsValue::PlainObject(obj) = item
                            && let Some(JsValue::NativeFunction(f)) =
                                obj.borrow().get("@@dispose").cloned()
                        {
                            let _ = f(vec![item.clone()]);
                        }
                    }
                    Ok(JsValue::Undefined)
                }),
            );
        }

        Ok(JsValue::PlainObject(Rc::new(RefCell::new(props))))
    })
}

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
    globals.insert("Intl".into(), make_intl());

    // ── Constructor / namespace objects ──────────────────────────────────
    globals.insert("Number".into(), make_number());
    globals.insert("Date".into(), make_date());
    globals.insert("Object".into(), make_object());
    globals.insert("Array".into(), make_array());
    globals.insert("Symbol".into(), make_symbol());
    globals.insert("Iterator".into(), make_iterator());
    globals.insert("AsyncIterator".into(), make_async_iterator());
    globals.insert("Map".into(), make_map_builtin());
    globals.insert("Set".into(), make_set_builtin());
    globals.insert("WeakMap".into(), make_weak_map_builtin());
    globals.insert("WeakSet".into(), make_weak_set_builtin());
    globals.insert("WeakRef".into(), make_weak_ref_builtin());
    globals.insert(
        "FinalizationRegistry".into(),
        make_finalization_registry_builtin(),
    );
    globals.insert("Promise".into(), make_promise());
    globals.insert("RegExp".into(), make_regexp());
    globals.insert("BigInt".into(), make_bigint());
    globals.insert("Function".into(), make_function());
    globals.insert("Proxy".into(), make_proxy());
    globals.insert("Reflect".into(), make_reflect());

    // ── Atomics / SharedArrayBuffer ─────────────────────────────────────
    globals.insert("Atomics".into(), make_atomics());
    globals.insert("SharedArrayBuffer".into(), make_shared_arraybuffer());

    // ── ShadowRealm ────────────────────────────────────────────────────
    globals.insert("ShadowRealm".into(), make_shadow_realm());

    // ── Error constructors ────────────────────────────────────────────────
    install_error_constructors(globals);

    // ── Explicit resource management ────────────────────────────────────
    globals.insert("DisposableStack".into(), make_disposable_stack());

    // ── Simple constructor-like wrappers ─────────────────────────────────
    globals.insert(
        "Boolean".into(),
        native(|args| {
            let val = args.first().unwrap_or(&JsValue::Undefined);
            Ok(JsValue::Boolean(val.to_boolean()))
        }),
    );
    globals.insert("String".into(), make_string());

    // ── TypedArray / ArrayBuffer / DataView constructors ─────────────────
    globals.insert("ArrayBuffer".into(), make_arraybuffer());
    globals.insert("DataView".into(), make_dataview());
    globals.insert(
        "Int8Array".into(),
        make_typed_array_constructor(TypedArrayKind::Int8),
    );
    globals.insert(
        "Uint8Array".into(),
        make_typed_array_constructor(TypedArrayKind::Uint8),
    );
    globals.insert(
        "Uint8ClampedArray".into(),
        make_typed_array_constructor(TypedArrayKind::Uint8Clamped),
    );
    globals.insert(
        "Int16Array".into(),
        make_typed_array_constructor(TypedArrayKind::Int16),
    );
    globals.insert(
        "Uint16Array".into(),
        make_typed_array_constructor(TypedArrayKind::Uint16),
    );
    globals.insert(
        "Int32Array".into(),
        make_typed_array_constructor(TypedArrayKind::Int32),
    );
    globals.insert(
        "Uint32Array".into(),
        make_typed_array_constructor(TypedArrayKind::Uint32),
    );
    globals.insert(
        "Float32Array".into(),
        make_typed_array_constructor(TypedArrayKind::Float32),
    );
    globals.insert(
        "Float64Array".into(),
        make_typed_array_constructor(TypedArrayKind::Float64),
    );
    globals.insert(
        "BigInt64Array".into(),
        make_typed_array_constructor(TypedArrayKind::BigInt64),
    );
    globals.insert(
        "BigUint64Array".into(),
        make_typed_array_constructor(TypedArrayKind::BigUint64),
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
            Ok(JsValue::String(global_encode_uri(&s).into()))
        }),
    );
    globals.insert(
        "decodeURI".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(global_decode_uri(&s)?.into()))
        }),
    );
    globals.insert(
        "encodeURIComponent".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(global_encode_uri_component(&s).into()))
        }),
    );
    globals.insert(
        "decodeURIComponent".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(global_decode_uri_component(&s)?.into()))
        }),
    );
    globals.insert(
        "eval".into(),
        native(|args| {
            // Indirect eval: non-string arguments are returned as-is (§19.2.1 step 2).
            let source = match args.first() {
                Some(JsValue::String(s)) => s.clone(),
                Some(other) => return Ok(other.clone()),
                None => return Ok(JsValue::Undefined),
            };
            global_eval(&source)
        }),
    );

    // ── Annex B global functions ─────────────────────────────────────────
    globals.insert(
        "escape".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(global_escape(&s).into()))
        }),
    );
    globals.insert(
        "unescape".into(),
        native(|args| {
            let s = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            Ok(JsValue::String(global_unescape(&s).into()))
        }),
    );

    // ── structuredClone(value) ──────────────────────────────────────────
    globals.insert(
        "structuredClone".into(),
        native(|args| {
            let val = args.first().unwrap_or(&JsValue::Undefined);
            Ok(structured_clone(val))
        }),
    );

    // ── queueMicrotask(callback) ────────────────────────────────────────
    globals.insert(
        "queueMicrotask".into(),
        native(|args| {
            let cb = args.first().unwrap_or(&JsValue::Undefined);
            match cb {
                JsValue::NativeFunction(f) => {
                    f(vec![])?;
                }
                _ => {
                    return Err(StatorError::TypeError(
                        "queueMicrotask: argument must be a function".into(),
                    ));
                }
            }
            Ok(JsValue::Undefined)
        }),
    );

    // ── globalThis (ECMAScript §19.1) ───────────────────────────────────
    // `globalThis` is a self-referential property of the global object.
    // The inner map is shared via `Rc` so that `globalThis.globalThis`
    // resolves back to the same object (configurable, writable, not enumerable).
    let mut inner_props = PropertyMap::new();
    for (k, v) in globals.iter() {
        inner_props.insert(k.clone(), v.clone());
    }
    let inner = Rc::new(RefCell::new(inner_props));
    inner
        .borrow_mut()
        .insert("globalThis".into(), JsValue::PlainObject(Rc::clone(&inner)));
    globals.insert("globalThis".into(), JsValue::PlainObject(inner));
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
        assert!(globals.contains_key("Date"));
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
        assert!(globals.contains_key("WeakRef"));
        assert!(globals.contains_key("FinalizationRegistry"));
        assert!(globals.contains_key("Promise"));
        assert!(globals.contains_key("RegExp"));
        assert!(globals.contains_key("BigInt"));
        assert!(globals.contains_key("Function"));
        assert!(globals.contains_key("Proxy"));
        assert!(globals.contains_key("Reflect"));
        assert!(globals.contains_key("globalThis"));
        assert!(globals.contains_key("Intl"));
        // Error constructors
        assert!(globals.contains_key("Error"));
        assert!(globals.contains_key("TypeError"));
        assert!(globals.contains_key("RangeError"));
        assert!(globals.contains_key("ReferenceError"));
        assert!(globals.contains_key("SyntaxError"));
        assert!(globals.contains_key("URIError"));
        assert!(globals.contains_key("EvalError"));
        assert!(globals.contains_key("AggregateError"));
        // TypedArray family
        assert!(globals.contains_key("ArrayBuffer"));
        assert!(globals.contains_key("DataView"));
        assert!(globals.contains_key("Int8Array"));
        assert!(globals.contains_key("Uint8Array"));
        assert!(globals.contains_key("Uint8ClampedArray"));
        assert!(globals.contains_key("Int16Array"));
        assert!(globals.contains_key("Uint16Array"));
        assert!(globals.contains_key("Int32Array"));
        assert!(globals.contains_key("Uint32Array"));
        assert!(globals.contains_key("Float32Array"));
        assert!(globals.contains_key("Float64Array"));
        assert!(globals.contains_key("BigInt64Array"));
        assert!(globals.contains_key("BigUint64Array"));
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

    // ── JSON Phase 2: stringify with space ──────────────────────────────────

    #[test]
    fn e2e_json_stringify_with_space() {
        use crate::builtins::json::{JsonSpace, JsonValue, json_stringify};
        use std::cell::RefCell;
        use std::rc::Rc;

        let obj = JsonValue::Object(Rc::new(RefCell::new(vec![(
            "x".to_string(),
            JsonValue::Number(1.0),
        )])));
        let s = json_stringify(&obj, None, Some(&JsonSpace::Count(2)), None)
            .unwrap()
            .unwrap();
        assert_eq!(s, "{\n  \"x\": 1\n}");
    }

    // ── JSON Phase 2: stringify replacer function ───────────────────────────

    #[test]
    fn e2e_json_stringify_replacer_fn() {
        use crate::builtins::json::{JsonReplacer, JsonValue, json_stringify};
        use std::cell::RefCell;
        use std::rc::Rc;

        let obj = JsonValue::Object(Rc::new(RefCell::new(vec![
            ("a".to_string(), JsonValue::Number(1.0)),
            ("b".to_string(), JsonValue::Number(2.0)),
        ])));
        let replacer = JsonReplacer::Function(&|key, val| {
            if key == "b" {
                Ok(None)
            } else {
                Ok(Some(val.clone()))
            }
        });
        let s = json_stringify(&obj, Some(&replacer), None, None)
            .unwrap()
            .unwrap();
        assert_eq!(s, r#"{"a":1}"#);
    }

    // ── JSON Phase 2: stringify NaN / Infinity → null ───────────────────────

    #[test]
    fn e2e_json_stringify_nan_infinity() {
        use crate::builtins::json::json_stringify_js_value;

        let s = json_stringify_js_value(&JsValue::HeapNumber(f64::NAN), None, None)
            .unwrap()
            .unwrap();
        assert_eq!(s, "null");

        let s = json_stringify_js_value(&JsValue::HeapNumber(f64::INFINITY), None, None)
            .unwrap()
            .unwrap();
        assert_eq!(s, "null");
    }

    // ── JSON Phase 2: stringify BigInt → TypeError ──────────────────────────

    #[test]
    fn e2e_json_stringify_bigint_error() {
        use crate::builtins::json::json_stringify_js_value;

        let result = json_stringify_js_value(&JsValue::BigInt(42), None, None);
        assert!(result.is_err());
    }

    // ── JSON Phase 2: parse reviver ─────────────────────────────────────────

    #[test]
    fn e2e_json_parse_reviver() {
        use crate::builtins::json::{JsonValue, json_parse};

        let v = json_parse(
            "[1, 2, 3]",
            Some(&|_key, val| {
                Ok(Some(match val {
                    JsonValue::Number(n) => JsonValue::Number(n * 10.0),
                    other => other,
                }))
            }),
        )
        .unwrap();
        if let JsonValue::Array(arr) = &v {
            let b = arr.borrow();
            assert_eq!(b[0], JsonValue::Number(10.0));
            assert_eq!(b[1], JsonValue::Number(20.0));
            assert_eq!(b[2], JsonValue::Number(30.0));
        } else {
            panic!("expected array");
        }
    }

    // ── JSON Phase 2: stringify toJSON method ───────────────────────────────

    #[test]
    fn e2e_json_stringify_to_json_method() {
        use crate::builtins::json::json_stringify_js_value;
        use std::cell::RefCell;
        use std::rc::Rc;

        // Build a PlainObject with a toJSON method.
        let mut inner = PropertyMap::new();
        inner.insert("value".into(), JsValue::Smi(42));
        inner.insert(
            "toJSON".into(),
            JsValue::NativeFunction(Rc::new(|_args| {
                Ok(JsValue::String("custom-serialized".into()))
            })),
        );
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(inner)));

        let s = json_stringify_js_value(&obj, None, None).unwrap().unwrap();
        assert_eq!(s, r#""custom-serialized""#);
    }

    // ── JSON Phase 2: apply_js_reviver ──────────────────────────────────────

    #[test]
    fn test_apply_js_reviver_doubles_numbers() {
        use crate::builtins::json::{JsonValue, json_parse};
        use std::rc::Rc;

        let json_val = json_parse("[1, 2, 3]", None).unwrap();
        let js_val = json_value_to_js_value(&json_val);

        let reviver: crate::objects::value::NativeFn = Rc::new(|args| {
            let val = args.get(1).cloned().unwrap_or(JsValue::Undefined);
            match val {
                JsValue::Smi(n) => Ok(JsValue::Smi(n * 2)),
                other => Ok(other),
            }
        });

        let result = apply_js_reviver(js_val, "", &reviver).unwrap();
        // The top-level array itself is passed through the reviver too,
        // so the result should be the array (reviver returns it unchanged).
        if let JsValue::Array(arr) = result {
            assert_eq!(arr.borrow()[0], JsValue::Smi(2));
            assert_eq!(arr.borrow()[1], JsValue::Smi(4));
            assert_eq!(arr.borrow()[2], JsValue::Smi(6));
        } else {
            panic!("expected array, got {result:?}");
        }
    }

    #[test]
    fn test_apply_js_reviver_removes_undefined() {
        use crate::builtins::json::json_parse;
        use std::rc::Rc;

        let json_val = json_parse(r#"{"a":1,"b":2,"c":3}"#, None).unwrap();
        let js_val = json_value_to_js_value(&json_val);

        // Remove "b" by returning undefined
        let reviver: crate::objects::value::NativeFn = Rc::new(|args| {
            let key = args.first().cloned().unwrap_or(JsValue::Undefined);
            let val = args.get(1).cloned().unwrap_or(JsValue::Undefined);
            if key == JsValue::String("b".into()) {
                Ok(JsValue::Undefined)
            } else {
                Ok(val)
            }
        });

        let result = apply_js_reviver(js_val, "", &reviver).unwrap();
        if let JsValue::PlainObject(map) = result {
            let m = map.borrow();
            assert!(m.contains_key("a"));
            assert!(!m.contains_key("b"), "key 'b' should be removed");
            assert!(m.contains_key("c"));
        } else {
            panic!("expected PlainObject");
        }
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

    // ── Symbol.prototype.description tests ──────────────────────────────

    /// `Symbol("foo").description` → "foo"
    #[test]
    fn e2e_symbol_description_with_value() {
        let result = global_eval(r#"Symbol("foo").description"#).unwrap();
        assert_eq!(result, JsValue::String("foo".into()));
    }

    /// `Symbol().description` → undefined
    #[test]
    fn e2e_symbol_description_undefined() {
        let result = global_eval("Symbol().description").unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    /// `Symbol("").description` → ""
    #[test]
    fn e2e_symbol_description_empty_string() {
        let result = global_eval(r#"Symbol("").description"#).unwrap();
        assert_eq!(result, JsValue::String("".into()));
    }

    /// Well-known `Symbol.iterator.description` → "Symbol.iterator"
    #[test]
    fn e2e_symbol_iterator_description() {
        let result = global_eval("Symbol.iterator.description").unwrap();
        assert_eq!(result, JsValue::String("Symbol.iterator".into()));
    }

    /// `Symbol.toPrimitive.description` → "Symbol.toPrimitive"
    #[test]
    fn e2e_symbol_to_primitive_description() {
        let result = global_eval("Symbol.toPrimitive.description").unwrap();
        assert_eq!(result, JsValue::String("Symbol.toPrimitive".into()));
    }

    /// `Symbol.hasInstance.description` → "Symbol.hasInstance"
    #[test]
    fn e2e_symbol_has_instance_description() {
        let result = global_eval("Symbol.hasInstance.description").unwrap();
        assert_eq!(result, JsValue::String("Symbol.hasInstance".into()));
    }

    /// `Symbol.toStringTag.description` → "Symbol.toStringTag"
    #[test]
    fn e2e_symbol_to_string_tag_description() {
        let result = global_eval("Symbol.toStringTag.description").unwrap();
        assert_eq!(result, JsValue::String("Symbol.toStringTag".into()));
    }

    /// `Symbol.species.description` → "Symbol.species"
    #[test]
    fn e2e_symbol_species_description() {
        let result = global_eval("Symbol.species.description").unwrap();
        assert_eq!(result, JsValue::String("Symbol.species".into()));
    }

    /// `Symbol.isConcatSpreadable.description` → "Symbol.isConcatSpreadable"
    #[test]
    fn e2e_symbol_is_concat_spreadable_description() {
        let result = global_eval("Symbol.isConcatSpreadable.description").unwrap();
        assert_eq!(result, JsValue::String("Symbol.isConcatSpreadable".into()));
    }

    /// `Symbol.match.description` → "Symbol.match"
    #[test]
    fn e2e_symbol_match_description() {
        let result = global_eval("Symbol.match.description").unwrap();
        assert_eq!(result, JsValue::String("Symbol.match".into()));
    }

    /// `Symbol.replace.description` → "Symbol.replace"
    #[test]
    fn e2e_symbol_replace_description() {
        let result = global_eval("Symbol.replace.description").unwrap();
        assert_eq!(result, JsValue::String("Symbol.replace".into()));
    }

    /// `Symbol.search.description` → "Symbol.search"
    #[test]
    fn e2e_symbol_search_description() {
        let result = global_eval("Symbol.search.description").unwrap();
        assert_eq!(result, JsValue::String("Symbol.search".into()));
    }

    /// `Symbol.split.description` → "Symbol.split"
    #[test]
    fn e2e_symbol_split_description() {
        let result = global_eval("Symbol.split.description").unwrap();
        assert_eq!(result, JsValue::String("Symbol.split".into()));
    }

    /// `Symbol.unscopables.description` → "Symbol.unscopables"
    #[test]
    fn e2e_symbol_unscopables_description() {
        let result = global_eval("Symbol.unscopables.description").unwrap();
        assert_eq!(result, JsValue::String("Symbol.unscopables".into()));
    }

    /// `Symbol.asyncIterator.description` → "Symbol.asyncIterator"
    #[test]
    fn e2e_symbol_async_iterator_description() {
        let result = global_eval("Symbol.asyncIterator.description").unwrap();
        assert_eq!(result, JsValue::String("Symbol.asyncIterator".into()));
    }

    /// `Symbol.matchAll.description` → "Symbol.matchAll"
    #[test]
    fn e2e_symbol_match_all_description() {
        let result = global_eval("Symbol.matchAll.description").unwrap();
        assert_eq!(result, JsValue::String("Symbol.matchAll".into()));
    }

    // ── Symbol.prototype.toString tests ─────────────────────────────────

    /// `Symbol("foo").toString()` → "Symbol(foo)"
    #[test]
    fn e2e_symbol_to_string_with_desc() {
        let result = global_eval(r#"Symbol("foo").toString()"#).unwrap();
        assert_eq!(result, JsValue::String("Symbol(foo)".into()));
    }

    /// `Symbol().toString()` → "Symbol()"
    #[test]
    fn e2e_symbol_to_string_no_desc() {
        let result = global_eval("Symbol().toString()").unwrap();
        assert_eq!(result, JsValue::String("Symbol()".into()));
    }

    /// `typeof Symbol("x").toString()` → "string"
    #[test]
    fn e2e_symbol_to_string_is_string() {
        let result = global_eval(r#"typeof Symbol("x").toString()"#).unwrap();
        assert_eq!(result, JsValue::String("string".into()));
    }

    // ── Symbol.prototype.valueOf tests ──────────────────────────────────

    /// `typeof Symbol("x").valueOf()` → "symbol"
    #[test]
    fn e2e_symbol_value_of_is_symbol() {
        let result = global_eval(r#"typeof Symbol("x").valueOf()"#).unwrap();
        assert_eq!(result, JsValue::String("symbol".into()));
    }

    /// `Symbol.iterator.valueOf() === Symbol.iterator`
    #[test]
    fn e2e_symbol_value_of_identity() {
        let result = global_eval("Symbol.iterator.valueOf() === Symbol.iterator").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── Symbol as property key tests ────────────────────────────────────

    /// Symbols can be used as property keys on objects.
    #[test]
    fn e2e_symbol_as_property_key() {
        let result = global_eval(
            r#"
            var s = Symbol("myKey");
            var obj = {};
            obj[s] = 42;
            obj[s]
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// Different symbols produce different property keys.
    #[test]
    fn e2e_symbol_different_keys() {
        let result = global_eval(
            r#"
            var s1 = Symbol("a");
            var s2 = Symbol("a");
            var obj = {};
            obj[s1] = 1;
            obj[s2] = 2;
            obj[s1]
            "#,
        )
        .unwrap();
        // s1 and s2 are different symbols despite same description,
        // so they produce different property keys.
        assert_eq!(result, JsValue::Smi(1));
    }

    /// `Symbol.for` symbol as property key is shared.
    #[test]
    fn e2e_symbol_for_as_property_key() {
        let result = global_eval(
            r#"
            var obj = {};
            obj[Symbol.for("shared")] = 99;
            obj[Symbol.for("shared")]
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    /// Well-known symbol as property key.
    #[test]
    fn e2e_well_known_symbol_as_property_key() {
        let result = global_eval(
            r#"
            var obj = {};
            obj[Symbol.iterator] = "iter";
            obj[Symbol.iterator]
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::String("iter".into()));
    }

    // ── Symbol.for / Symbol.keyFor e2e tests ────────────────────────────

    /// `Symbol.for("x") !== Symbol("x")` — registry symbols ≠ non-registry.
    #[test]
    fn e2e_symbol_for_not_same_as_symbol() {
        let result = global_eval(r#"Symbol.for("x") === Symbol("x")"#).unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    /// `Symbol.for` with different keys returns different symbols.
    #[test]
    fn e2e_symbol_for_different_keys() {
        let result = global_eval(r#"Symbol.for("a") === Symbol.for("b")"#).unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    /// `Symbol.keyFor(Symbol.for("test"))` → "test"
    #[test]
    fn e2e_symbol_key_for_returns_key() {
        let result = global_eval(r#"Symbol.keyFor(Symbol.for("test"))"#).unwrap();
        assert_eq!(result, JsValue::String("test".into()));
    }

    /// `Symbol.keyFor` returns undefined for non-registry symbols.
    #[test]
    fn e2e_symbol_key_for_undefined() {
        let result = global_eval(r#"Symbol.keyFor(Symbol("desc")) === undefined"#).unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `Symbol.keyFor` returns undefined for well-known symbols.
    #[test]
    fn e2e_symbol_key_for_well_known_undefined() {
        let result = global_eval("Symbol.keyFor(Symbol.iterator) === undefined").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── typeof symbol tests ─────────────────────────────────────────────

    /// `typeof Symbol.for("x")` → "symbol"
    #[test]
    fn e2e_typeof_symbol_for() {
        let result = global_eval(r#"typeof Symbol.for("x")"#).unwrap();
        assert_eq!(result, JsValue::String("symbol".into()));
    }

    /// `typeof Symbol("desc")` → "symbol"
    #[test]
    fn e2e_typeof_symbol_with_desc() {
        let result = global_eval(r#"typeof Symbol("desc")"#).unwrap();
        assert_eq!(result, JsValue::String("symbol".into()));
    }

    /// `typeof Symbol.toPrimitive` → "symbol"
    #[test]
    fn e2e_typeof_symbol_to_primitive() {
        let result = global_eval("typeof Symbol.toPrimitive").unwrap();
        assert_eq!(result, JsValue::String("symbol".into()));
    }

    /// `typeof Symbol.hasInstance` → "symbol"
    #[test]
    fn e2e_typeof_symbol_has_instance() {
        let result = global_eval("typeof Symbol.hasInstance").unwrap();
        assert_eq!(result, JsValue::String("symbol".into()));
    }

    /// `typeof Symbol.species` → "symbol"
    #[test]
    fn e2e_typeof_symbol_species() {
        let result = global_eval("typeof Symbol.species").unwrap();
        assert_eq!(result, JsValue::String("symbol".into()));
    }

    /// `typeof Symbol.toStringTag` → "symbol"
    #[test]
    fn e2e_typeof_symbol_to_string_tag() {
        let result = global_eval("typeof Symbol.toStringTag").unwrap();
        assert_eq!(result, JsValue::String("symbol".into()));
    }

    // ── Object.prototype.toString tests ─────────────────────────────────

    /// `Object.prototype.toString.call(undefined)` → "[object Undefined]"
    #[test]
    fn e2e_obj_tostring_call_undefined() {
        let result = global_eval("Object.prototype.toString.call(undefined)").unwrap();
        assert_eq!(result, JsValue::String("[object Undefined]".into()));
    }

    /// `Object.prototype.toString.call(null)` → "[object Null]"
    #[test]
    fn e2e_obj_tostring_call_null() {
        let result = global_eval("Object.prototype.toString.call(null)").unwrap();
        assert_eq!(result, JsValue::String("[object Null]".into()));
    }

    /// `Object.prototype.toString.call([])` → "[object Array]"
    #[test]
    fn e2e_obj_tostring_call_array() {
        let result = global_eval("Object.prototype.toString.call([])").unwrap();
        assert_eq!(result, JsValue::String("[object Array]".into()));
    }

    /// `Object.prototype.toString.call(true)` → "[object Boolean]"
    #[test]
    fn e2e_obj_tostring_call_boolean() {
        let result = global_eval("Object.prototype.toString.call(true)").unwrap();
        assert_eq!(result, JsValue::String("[object Boolean]".into()));
    }

    /// `Object.prototype.toString.call(42)` → "[object Number]"
    #[test]
    fn e2e_obj_tostring_call_number() {
        let result = global_eval("Object.prototype.toString.call(42)").unwrap();
        assert_eq!(result, JsValue::String("[object Number]".into()));
    }

    /// `Object.prototype.toString.call("hi")` → "[object String]"
    #[test]
    fn e2e_obj_tostring_call_string() {
        let result = global_eval(r#"Object.prototype.toString.call("hi")"#).unwrap();
        assert_eq!(result, JsValue::String("[object String]".into()));
    }

    /// `Object.prototype.toString.call({})` → "[object Object]"
    #[test]
    fn e2e_obj_tostring_call_object() {
        let result = global_eval("Object.prototype.toString.call({})").unwrap();
        assert_eq!(result, JsValue::String("[object Object]".into()));
    }

    /// Direct `({}).toString()` → "[object Object]"
    #[test]
    fn e2e_plain_object_tostring_direct() {
        let result = global_eval("({}).toString()").unwrap();
        assert_eq!(result, JsValue::String("[object Object]".into()));
    }

    /// `Object.prototype.toString.call(function(){})` → "[object Function]"
    #[test]
    fn e2e_obj_tostring_call_function() {
        let result = global_eval("Object.prototype.toString.call(function(){})").unwrap();
        assert_eq!(result, JsValue::String("[object Function]".into()));
    }

    /// `Object.prototype.toString.call(new Error())` → "[object Error]"
    #[test]
    fn e2e_obj_tostring_call_error() {
        let result = global_eval("Object.prototype.toString.call(new Error())").unwrap();
        assert_eq!(result, JsValue::String("[object Error]".into()));
    }

    /// `Object.prototype.toString.call(new Date())` → "[object Date]"
    #[test]
    fn e2e_obj_tostring_call_date() {
        let result = global_eval("Object.prototype.toString.call(new Date())").unwrap();
        assert_eq!(result, JsValue::String("[object Date]".into()));
    }

    /// `typeof Symbol.match` → "symbol"
    #[test]
    fn e2e_typeof_symbol_match() {
        let result = global_eval("typeof Symbol.match").unwrap();
        assert_eq!(result, JsValue::String("symbol".into()));
    }

    /// `typeof Symbol.replace` → "symbol"
    #[test]
    fn e2e_typeof_symbol_replace() {
        let result = global_eval("typeof Symbol.replace").unwrap();
        assert_eq!(result, JsValue::String("symbol".into()));
    }

    /// `typeof Symbol.search` → "symbol"
    #[test]
    fn e2e_typeof_symbol_search() {
        let result = global_eval("typeof Symbol.search").unwrap();
        assert_eq!(result, JsValue::String("symbol".into()));
    }

    /// `typeof Symbol.split` → "symbol"
    #[test]
    fn e2e_typeof_symbol_split() {
        let result = global_eval("typeof Symbol.split").unwrap();
        assert_eq!(result, JsValue::String("symbol".into()));
    }

    /// `typeof Symbol.unscopables` → "symbol"
    #[test]
    fn e2e_typeof_symbol_unscopables() {
        let result = global_eval("typeof Symbol.unscopables").unwrap();
        assert_eq!(result, JsValue::String("symbol".into()));
    }

    /// `typeof Symbol.asyncIterator` → "symbol"
    #[test]
    fn e2e_typeof_symbol_async_iterator() {
        let result = global_eval("typeof Symbol.asyncIterator").unwrap();
        assert_eq!(result, JsValue::String("symbol".into()));
    }

    /// `typeof Symbol.isConcatSpreadable` → "symbol"
    #[test]
    fn e2e_typeof_symbol_is_concat_spreadable() {
        let result = global_eval("typeof Symbol.isConcatSpreadable").unwrap();
        assert_eq!(result, JsValue::String("symbol".into()));
    }

    /// `typeof Symbol.matchAll` → "symbol"
    #[test]
    fn e2e_typeof_symbol_match_all() {
        let result = global_eval("typeof Symbol.matchAll").unwrap();
        assert_eq!(result, JsValue::String("symbol".into()));
    }

    // ── Symbol identity / equality tests ────────────────────────────────

    /// `Symbol("a") === Symbol("a")` → false (unique).
    #[test]
    fn e2e_symbol_same_desc_not_equal() {
        let result = global_eval(r#"Symbol("a") === Symbol("a")"#).unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    /// A symbol is strictly equal to itself.
    #[test]
    fn e2e_symbol_identity() {
        let result = global_eval(
            r#"
            var s = Symbol("id");
            s === s
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── Symbol.prototype exists ─────────────────────────────────────────

    /// `Symbol` object has a `prototype` property.
    #[test]
    fn test_symbol_prototype_exists() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let sym = globals.get("Symbol").unwrap();
        if let JsValue::PlainObject(map) = sym {
            let map = map.borrow();
            assert!(
                matches!(map.get("prototype"), Some(JsValue::PlainObject(_))),
                "Symbol.prototype should be a PlainObject"
            );
        } else {
            panic!("Symbol should be a PlainObject");
        }
    }

    /// `Symbol.prototype` has `toString`, `valueOf`, `description` methods.
    #[test]
    fn test_symbol_prototype_methods() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let sym = globals.get("Symbol").unwrap();
        if let JsValue::PlainObject(map) = sym {
            let map = map.borrow();
            let proto = map.get("prototype").unwrap();
            if let JsValue::PlainObject(proto_map) = proto {
                let proto_map = proto_map.borrow();
                assert!(matches!(
                    proto_map.get("toString"),
                    Some(JsValue::NativeFunction(_))
                ));
                assert!(matches!(
                    proto_map.get("valueOf"),
                    Some(JsValue::NativeFunction(_))
                ));
                assert!(matches!(
                    proto_map.get("description"),
                    Some(JsValue::NativeFunction(_))
                ));
            } else {
                panic!("Symbol.prototype should be a PlainObject");
            }
        } else {
            panic!("Symbol should be a PlainObject");
        }
    }

    // ── Well-known symbol completeness test ─────────────────────────────

    /// All 13 well-known symbols are present on the Symbol constructor.
    #[test]
    fn test_symbol_all_well_known_present() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let sym = globals.get("Symbol").unwrap();
        if let JsValue::PlainObject(map) = sym {
            let map = map.borrow();
            let expected = [
                "iterator",
                "toPrimitive",
                "hasInstance",
                "toStringTag",
                "isConcatSpreadable",
                "species",
                "match",
                "replace",
                "search",
                "split",
                "unscopables",
                "asyncIterator",
                "matchAll",
            ];
            for name in &expected {
                assert!(
                    matches!(map.get(*name), Some(JsValue::Symbol(_))),
                    "Symbol.{name} should be present as a Symbol value"
                );
            }
        } else {
            panic!("Symbol should be a PlainObject");
        }
    }

    /// Each well-known symbol is distinct from the others.
    #[test]
    fn test_symbol_well_known_all_distinct() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let sym = globals.get("Symbol").unwrap();
        if let JsValue::PlainObject(map) = sym {
            let map = map.borrow();
            let names = [
                "iterator",
                "toPrimitive",
                "hasInstance",
                "toStringTag",
                "isConcatSpreadable",
                "species",
                "match",
                "replace",
                "search",
                "split",
                "unscopables",
                "asyncIterator",
                "matchAll",
            ];
            let ids: Vec<u64> = names
                .iter()
                .map(|n| {
                    if let Some(JsValue::Symbol(id)) = map.get(*n) {
                        *id
                    } else {
                        panic!("Symbol.{n} missing");
                    }
                })
                .collect();
            // All IDs should be unique.
            let mut deduped = ids.clone();
            deduped.sort();
            deduped.dedup();
            assert_eq!(
                ids.len(),
                deduped.len(),
                "All well-known symbols must have distinct IDs"
            );
        }
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
                let target_map = PropertyMap::new();
                let target = JsValue::PlainObject(Rc::new(RefCell::new(target_map)));
                let mut src_map = PropertyMap::new();
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
                let mut t = PropertyMap::new();
                t.insert("a".into(), JsValue::Smi(1));
                let target = JsValue::PlainObject(Rc::new(RefCell::new(t)));
                let mut s1 = PropertyMap::new();
                s1.insert("b".into(), JsValue::Smi(2));
                let src1 = JsValue::PlainObject(Rc::new(RefCell::new(s1)));
                let mut s2 = PropertyMap::new();
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

    /// `Object.groupBy` groups array elements by callback return value.
    #[test]
    fn test_object_group_by_native() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let obj = globals.get("Object").unwrap();
        if let JsValue::PlainObject(map) = obj {
            let group_by = map.borrow().get("groupBy").cloned().unwrap();
            if let JsValue::NativeFunction(f) = group_by {
                let items = JsValue::new_array(vec![
                    JsValue::Smi(1),
                    JsValue::Smi(2),
                    JsValue::Smi(3),
                    JsValue::Smi(4),
                ]);
                let cb = JsValue::NativeFunction(Rc::new(|args: Vec<JsValue>| {
                    let v = args.first().unwrap_or(&JsValue::Undefined).clone();
                    let n = v.to_number().unwrap_or(0.0) as i32;
                    if n % 2 == 0 {
                        Ok(JsValue::String("even".into()))
                    } else {
                        Ok(JsValue::String("odd".into()))
                    }
                }));
                let result = f(vec![items, cb]).unwrap();
                if let JsValue::PlainObject(r) = &result {
                    let borrow = r.borrow();
                    let odd = borrow.get("odd").cloned().unwrap();
                    let even = borrow.get("even").cloned().unwrap();
                    if let JsValue::Array(odd_arr) = odd {
                        assert_eq!(odd_arr.borrow().len(), 2);
                        assert_eq!(odd_arr.borrow()[0], JsValue::Smi(1));
                        assert_eq!(odd_arr.borrow()[1], JsValue::Smi(3));
                    } else {
                        panic!("odd should be Array");
                    }
                    if let JsValue::Array(even_arr) = even {
                        assert_eq!(even_arr.borrow().len(), 2);
                        assert_eq!(even_arr.borrow()[0], JsValue::Smi(2));
                        assert_eq!(even_arr.borrow()[1], JsValue::Smi(4));
                    } else {
                        panic!("even should be Array");
                    }
                } else {
                    panic!("Expected PlainObject");
                }
            } else {
                panic!("groupBy should be NativeFunction");
            }
        }
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

    /// `arguments.length` returns the number of arguments passed.
    #[test]
    fn e2e_arguments_length() {
        let result = global_eval(
            r#"
            function f(a, b) { return arguments.length; }
            f(1, 2)
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(2));
    }

    /// `arguments[i]` returns the i-th argument.
    #[test]
    fn e2e_arguments_indexing() {
        let result = global_eval(
            r#"
            function f(a, b) { return arguments[1]; }
            f(10, 20)
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(20));
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

    /// `globalThis.globalThis` resolves back to the same object (self-referential).
    #[test]
    fn test_global_this_is_self_referential() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        if let Some(JsValue::PlainObject(gt)) = globals.get("globalThis") {
            // globalThis.globalThis should exist and be a PlainObject
            let inner = gt.borrow();
            assert!(
                inner.contains_key("globalThis"),
                "globalThis should contain a 'globalThis' key"
            );
            if let Some(JsValue::PlainObject(gt2)) = inner.get("globalThis") {
                // The inner Rc should point to the same allocation.
                assert!(
                    Rc::ptr_eq(gt, gt2),
                    "globalThis.globalThis should be the same Rc"
                );
            } else {
                panic!("globalThis.globalThis should be a PlainObject");
            }
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
                let iterable = JsValue::new_array(vec![
                    JsValue::new_array(vec![JsValue::Smi(1), JsValue::String("a".into())]),
                    JsValue::new_array(vec![JsValue::Smi(2), JsValue::String("b".into())]),
                ]);
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
                let iterable =
                    JsValue::new_array(vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(1)]);
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

    // ── WeakRef constructor tests ────────────────────────────────────────────

    /// `WeakRef` global is a PlainObject with a `__call__` constructor.
    #[test]
    fn test_weak_ref_global_exists() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        assert!(matches!(
            globals.get("WeakRef"),
            Some(JsValue::PlainObject(_))
        ));
    }

    /// Constructing a WeakRef via `__call__` returns an object with `deref`.
    #[test]
    fn test_weak_ref_constructor_creates_instance() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        if let JsValue::PlainObject(wr_ctor) = globals.get("WeakRef").unwrap() {
            let call = wr_ctor.borrow().get("__call__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = call {
                let mut obj = crate::objects::heap_object::HeapObject::new_null();
                let ptr = &raw mut obj;
                let result = f(vec![JsValue::Object(ptr)]).unwrap();
                if let JsValue::PlainObject(instance) = result {
                    let inst = instance.borrow();
                    assert!(inst.contains_key("deref"));
                } else {
                    panic!("WeakRef() should return a PlainObject");
                }
            }
        }
    }

    /// `WeakRef` constructor with non-object argument returns TypeError.
    #[test]
    fn test_weak_ref_constructor_non_object_error() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        if let JsValue::PlainObject(wr_ctor) = globals.get("WeakRef").unwrap() {
            let call = wr_ctor.borrow().get("__call__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = call {
                let result = f(vec![JsValue::Smi(42)]);
                assert!(result.is_err());
            }
        }
    }

    /// `WeakRef.prototype.deref()` returns the target object.
    #[test]
    fn test_weak_ref_deref_returns_target() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        if let JsValue::PlainObject(wr_ctor) = globals.get("WeakRef").unwrap() {
            let call = wr_ctor.borrow().get("__call__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = call {
                let mut obj = crate::objects::heap_object::HeapObject::new_null();
                let ptr = &raw mut obj;
                let instance = f(vec![JsValue::Object(ptr)]).unwrap();
                if let JsValue::PlainObject(inst) = instance {
                    let deref_fn = inst.borrow().get("deref").cloned().unwrap();
                    if let JsValue::NativeFunction(deref) = deref_fn {
                        let result = deref(vec![]).unwrap();
                        assert!(matches!(result, JsValue::Object(p) if p == ptr));
                    }
                }
            }
        }
    }

    // ── FinalizationRegistry constructor tests ───────────────────────────────

    /// `FinalizationRegistry` global is a PlainObject with a `__call__` constructor.
    #[test]
    fn test_finalization_registry_global_exists() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        assert!(matches!(
            globals.get("FinalizationRegistry"),
            Some(JsValue::PlainObject(_))
        ));
    }

    /// Constructing a FinalizationRegistry via `__call__` returns an object.
    #[test]
    fn test_finalization_registry_constructor_creates_instance() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        if let JsValue::PlainObject(fr_ctor) = globals.get("FinalizationRegistry").unwrap() {
            let call = fr_ctor.borrow().get("__call__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = call {
                let cb = native(|_| Ok(JsValue::Undefined));
                let result = f(vec![cb]).unwrap();
                if let JsValue::PlainObject(instance) = result {
                    let inst = instance.borrow();
                    assert!(inst.contains_key("register"));
                    assert!(inst.contains_key("unregister"));
                    assert!(inst.contains_key("cleanupSome"));
                } else {
                    panic!("FinalizationRegistry() should return a PlainObject");
                }
            }
        }
    }

    /// `FinalizationRegistry` constructor without callback returns TypeError.
    #[test]
    fn test_finalization_registry_constructor_no_callback_error() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        if let JsValue::PlainObject(fr_ctor) = globals.get("FinalizationRegistry").unwrap() {
            let call = fr_ctor.borrow().get("__call__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = call {
                let result = f(vec![]);
                assert!(result.is_err());
            }
        }
    }

    /// `FinalizationRegistry.prototype.register` with non-object target returns TypeError.
    #[test]
    fn test_finalization_registry_register_non_object_error() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        if let JsValue::PlainObject(fr_ctor) = globals.get("FinalizationRegistry").unwrap() {
            let call = fr_ctor.borrow().get("__call__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = call {
                let cb = native(|_| Ok(JsValue::Undefined));
                let instance = f(vec![cb]).unwrap();
                if let JsValue::PlainObject(inst) = instance {
                    let register_fn = inst.borrow().get("register").cloned().unwrap();
                    if let JsValue::NativeFunction(register) = register_fn {
                        let result = register(vec![JsValue::Smi(42), JsValue::Smi(1)]);
                        assert!(result.is_err());
                    }
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

    // ── eval: direct vs indirect (end-to-end) ───────────────────────────────

    /// Direct eval `eval("1+2")` is recognised by the bytecode generator
    /// and executed sharing the caller's global environment.
    #[test]
    fn e2e_eval_direct_expression() {
        let result = global_eval("eval(42)").unwrap();
        // Non-string argument returns the value directly.
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn e2e_eval_direct_string() {
        let result = global_eval("eval('1 + 2')").unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    #[test]
    fn e2e_eval_direct_var_hoisting() {
        // Direct eval: `var x` should be visible after eval.
        let result = global_eval("eval('var x = 10'); x").unwrap();
        assert_eq!(result, JsValue::Smi(10));
    }

    #[test]
    fn e2e_eval_indirect_expression() {
        // Indirect eval via comma operator: `(0, eval)("1+2")`.
        let result = global_eval("(0, eval)('1 + 2')").unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    #[test]
    fn e2e_eval_indirect_no_caller_scope() {
        // Indirect eval runs in global scope: variables defined in the
        // outer scope should NOT be accessible.  Unknown names resolve to
        // `undefined` (not an error) because LdaGlobal returns undefined
        // for unbound names.
        let result = global_eval("var a = 5; (0, eval)('a')").unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    #[test]
    fn e2e_eval_with_closures() {
        // Direct eval at the top level can declare variables via var
        // hoisting that persist in the same top-level scope.
        let result = global_eval("eval('var x = 10'); eval('x + 5')").unwrap();
        assert_eq!(result, JsValue::Smi(15));
    }

    #[test]
    fn e2e_eval_no_args_returns_undefined() {
        let result = global_eval("eval()").unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    /// `eval` is accessible as a global identifier.
    #[test]
    fn e2e_eval_is_global() {
        let result = global_eval("typeof eval").unwrap();
        assert_eq!(result, JsValue::String("function".into()));
    }

    // ── BigInt tests ────────────────────────────────────────────────────────

    // -- Literal parsing --

    #[test]
    fn e2e_bigint_literal_zero() {
        let result = global_eval("0n").unwrap();
        assert_eq!(result, JsValue::BigInt(0));
    }

    #[test]
    fn e2e_bigint_literal_positive() {
        let result = global_eval("42n").unwrap();
        assert_eq!(result, JsValue::BigInt(42));
    }

    #[test]
    fn e2e_bigint_literal_negative() {
        let result = global_eval("-1n").unwrap();
        assert_eq!(result, JsValue::BigInt(-1));
    }

    #[test]
    fn e2e_bigint_literal_large() {
        let result = global_eval("9007199254740993n").unwrap();
        assert_eq!(result, JsValue::BigInt(9_007_199_254_740_993));
    }

    #[test]
    fn e2e_bigint_literal_hex() {
        let result = global_eval("0xFFn").unwrap();
        assert_eq!(result, JsValue::BigInt(255));
    }

    #[test]
    fn e2e_bigint_literal_octal() {
        let result = global_eval("0o77n").unwrap();
        assert_eq!(result, JsValue::BigInt(63));
    }

    #[test]
    fn e2e_bigint_literal_binary() {
        let result = global_eval("0b1010n").unwrap();
        assert_eq!(result, JsValue::BigInt(10));
    }

    #[test]
    fn e2e_bigint_literal_large_hex() {
        let result = global_eval("0x1Fn").unwrap();
        assert_eq!(result, JsValue::BigInt(31));
    }

    // -- typeof --

    #[test]
    fn e2e_bigint_typeof() {
        let result = global_eval("typeof 42n").unwrap();
        assert_eq!(result, JsValue::String("bigint".into()));
    }

    #[test]
    fn e2e_bigint_typeof_zero() {
        let result = global_eval("typeof 0n").unwrap();
        assert_eq!(result, JsValue::String("bigint".into()));
    }

    #[test]
    fn e2e_bigint_typeof_negative() {
        let result = global_eval("typeof -1n").unwrap();
        assert_eq!(result, JsValue::String("bigint".into()));
    }

    // -- Arithmetic: addition --

    #[test]
    fn e2e_bigint_add() {
        let result = global_eval("1n + 2n").unwrap();
        assert_eq!(result, JsValue::BigInt(3));
    }

    #[test]
    fn e2e_bigint_add_zero() {
        let result = global_eval("0n + 0n").unwrap();
        assert_eq!(result, JsValue::BigInt(0));
    }

    #[test]
    fn e2e_bigint_add_negative() {
        let result = global_eval("-1n + -2n").unwrap();
        assert_eq!(result, JsValue::BigInt(-3));
    }

    #[test]
    fn e2e_bigint_add_large() {
        let result = global_eval("9007199254740993n + 1n").unwrap();
        assert_eq!(result, JsValue::BigInt(9_007_199_254_740_994));
    }

    #[test]
    fn e2e_bigint_add_mixed_type_error() {
        let result = global_eval("1n + 1");
        assert!(result.is_err());
    }

    #[test]
    fn e2e_bigint_add_mixed_type_error_reverse() {
        let result = global_eval("1 + 1n");
        assert!(result.is_err());
    }

    // -- Arithmetic: subtraction --

    #[test]
    fn e2e_bigint_sub() {
        let result = global_eval("5n - 3n").unwrap();
        assert_eq!(result, JsValue::BigInt(2));
    }

    #[test]
    fn e2e_bigint_sub_negative_result() {
        let result = global_eval("3n - 5n").unwrap();
        assert_eq!(result, JsValue::BigInt(-2));
    }

    #[test]
    fn e2e_bigint_sub_zero() {
        let result = global_eval("0n - 0n").unwrap();
        assert_eq!(result, JsValue::BigInt(0));
    }

    #[test]
    fn e2e_bigint_sub_mixed_error() {
        assert!(global_eval("1n - 1").is_err());
    }

    // -- Arithmetic: multiplication --

    #[test]
    fn e2e_bigint_mul() {
        let result = global_eval("3n * 4n").unwrap();
        assert_eq!(result, JsValue::BigInt(12));
    }

    #[test]
    fn e2e_bigint_mul_zero() {
        let result = global_eval("100n * 0n").unwrap();
        assert_eq!(result, JsValue::BigInt(0));
    }

    #[test]
    fn e2e_bigint_mul_negative() {
        let result = global_eval("-3n * 4n").unwrap();
        assert_eq!(result, JsValue::BigInt(-12));
    }

    #[test]
    fn e2e_bigint_mul_both_negative() {
        let result = global_eval("-3n * -4n").unwrap();
        assert_eq!(result, JsValue::BigInt(12));
    }

    #[test]
    fn e2e_bigint_mul_mixed_error() {
        assert!(global_eval("2n * 3").is_err());
    }

    // -- Arithmetic: division --

    #[test]
    fn e2e_bigint_div() {
        let result = global_eval("10n / 3n").unwrap();
        assert_eq!(result, JsValue::BigInt(3)); // truncates
    }

    #[test]
    fn e2e_bigint_div_exact() {
        let result = global_eval("10n / 2n").unwrap();
        assert_eq!(result, JsValue::BigInt(5));
    }

    #[test]
    fn e2e_bigint_div_negative() {
        let result = global_eval("-10n / 3n").unwrap();
        assert_eq!(result, JsValue::BigInt(-3));
    }

    #[test]
    fn e2e_bigint_div_by_zero_error() {
        assert!(global_eval("10n / 0n").is_err());
    }

    #[test]
    fn e2e_bigint_div_mixed_error() {
        assert!(global_eval("10n / 2").is_err());
    }

    // -- Arithmetic: modulo --

    #[test]
    fn e2e_bigint_mod() {
        let result = global_eval("10n % 3n").unwrap();
        assert_eq!(result, JsValue::BigInt(1));
    }

    #[test]
    fn e2e_bigint_mod_zero_result() {
        let result = global_eval("10n % 5n").unwrap();
        assert_eq!(result, JsValue::BigInt(0));
    }

    #[test]
    fn e2e_bigint_mod_negative() {
        let result = global_eval("-10n % 3n").unwrap();
        assert_eq!(result, JsValue::BigInt(-1));
    }

    #[test]
    fn e2e_bigint_mod_by_zero_error() {
        assert!(global_eval("10n % 0n").is_err());
    }

    #[test]
    fn e2e_bigint_mod_mixed_error() {
        assert!(global_eval("10n % 3").is_err());
    }

    // -- Arithmetic: exponentiation --

    #[test]
    fn e2e_bigint_exp() {
        let result = global_eval("2n ** 10n").unwrap();
        assert_eq!(result, JsValue::BigInt(1024));
    }

    #[test]
    fn e2e_bigint_exp_zero() {
        let result = global_eval("2n ** 0n").unwrap();
        assert_eq!(result, JsValue::BigInt(1));
    }

    #[test]
    fn e2e_bigint_exp_one() {
        let result = global_eval("5n ** 1n").unwrap();
        assert_eq!(result, JsValue::BigInt(5));
    }

    #[test]
    fn e2e_bigint_exp_negative_exponent_error() {
        assert!(global_eval("2n ** -1n").is_err());
    }

    #[test]
    fn e2e_bigint_exp_mixed_error() {
        assert!(global_eval("2n ** 3").is_err());
    }

    // -- Unary: negate --

    #[test]
    fn e2e_bigint_negate() {
        let result = global_eval("-42n").unwrap();
        assert_eq!(result, JsValue::BigInt(-42));
    }

    #[test]
    fn e2e_bigint_negate_zero() {
        let result = global_eval("-0n").unwrap();
        assert_eq!(result, JsValue::BigInt(0));
    }

    #[test]
    fn e2e_bigint_negate_negative() {
        // Double negate: -(-5n) = 5n; but this is parsed as `-(-(5n))`
        let result = global_eval("let x = -5n; -x").unwrap();
        assert_eq!(result, JsValue::BigInt(5));
    }

    // -- Unary: bitwise not --

    #[test]
    fn e2e_bigint_bitwise_not() {
        let result = global_eval("~0n").unwrap();
        assert_eq!(result, JsValue::BigInt(-1));
    }

    #[test]
    fn e2e_bigint_bitwise_not_positive() {
        let result = global_eval("~5n").unwrap();
        assert_eq!(result, JsValue::BigInt(-6));
    }

    #[test]
    fn e2e_bigint_bitwise_not_negative() {
        let result = global_eval("~-1n").unwrap();
        assert_eq!(result, JsValue::BigInt(0));
    }

    // -- Bitwise: OR --

    #[test]
    fn e2e_bigint_bitwise_or() {
        let result = global_eval("5n | 3n").unwrap();
        assert_eq!(result, JsValue::BigInt(7));
    }

    #[test]
    fn e2e_bigint_bitwise_or_zero() {
        let result = global_eval("0n | 0n").unwrap();
        assert_eq!(result, JsValue::BigInt(0));
    }

    #[test]
    fn e2e_bigint_bitwise_or_mixed_error() {
        assert!(global_eval("5n | 3").is_err());
    }

    // -- Bitwise: AND --

    #[test]
    fn e2e_bigint_bitwise_and() {
        let result = global_eval("5n & 3n").unwrap();
        assert_eq!(result, JsValue::BigInt(1));
    }

    #[test]
    fn e2e_bigint_bitwise_and_zero() {
        let result = global_eval("5n & 0n").unwrap();
        assert_eq!(result, JsValue::BigInt(0));
    }

    #[test]
    fn e2e_bigint_bitwise_and_mixed_error() {
        assert!(global_eval("5n & 3").is_err());
    }

    // -- Bitwise: XOR --

    #[test]
    fn e2e_bigint_bitwise_xor() {
        let result = global_eval("5n ^ 3n").unwrap();
        assert_eq!(result, JsValue::BigInt(6));
    }

    #[test]
    fn e2e_bigint_bitwise_xor_same() {
        let result = global_eval("7n ^ 7n").unwrap();
        assert_eq!(result, JsValue::BigInt(0));
    }

    #[test]
    fn e2e_bigint_bitwise_xor_mixed_error() {
        assert!(global_eval("5n ^ 3").is_err());
    }

    // -- Bitwise: shifts --

    #[test]
    fn e2e_bigint_shift_left() {
        let result = global_eval("1n << 10n").unwrap();
        assert_eq!(result, JsValue::BigInt(1024));
    }

    #[test]
    fn e2e_bigint_shift_left_zero() {
        let result = global_eval("5n << 0n").unwrap();
        assert_eq!(result, JsValue::BigInt(5));
    }

    #[test]
    fn e2e_bigint_shift_left_mixed_error() {
        assert!(global_eval("1n << 2").is_err());
    }

    #[test]
    fn e2e_bigint_shift_right() {
        let result = global_eval("1024n >> 5n").unwrap();
        assert_eq!(result, JsValue::BigInt(32));
    }

    #[test]
    fn e2e_bigint_shift_right_zero() {
        let result = global_eval("5n >> 0n").unwrap();
        assert_eq!(result, JsValue::BigInt(5));
    }

    #[test]
    fn e2e_bigint_shift_right_mixed_error() {
        assert!(global_eval("1024n >> 5").is_err());
    }

    // -- Comparison: strict equality --

    #[test]
    fn e2e_bigint_strict_eq_same() {
        let result = global_eval("42n === 42n").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_bigint_strict_eq_different() {
        let result = global_eval("42n === 43n").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn e2e_bigint_strict_eq_number_false() {
        let result = global_eval("42n === 42").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn e2e_bigint_strict_neq() {
        let result = global_eval("42n !== 42").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_bigint_strict_eq_zero() {
        let result = global_eval("0n === 0n").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // -- Comparison: abstract equality --

    #[test]
    fn e2e_bigint_abstract_eq_number() {
        let result = global_eval("42n == 42").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_bigint_abstract_eq_number_reverse() {
        let result = global_eval("42 == 42n").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_bigint_abstract_eq_different() {
        let result = global_eval("42n == 43").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn e2e_bigint_abstract_eq_string() {
        let result = global_eval("42n == '42'").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_bigint_abstract_eq_string_reverse() {
        let result = global_eval("'42' == 42n").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_bigint_abstract_eq_string_mismatch() {
        let result = global_eval("42n == 'hello'").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn e2e_bigint_abstract_neq() {
        let result = global_eval("42n != 43").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_bigint_abstract_eq_zero() {
        let result = global_eval("0n == 0").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_bigint_abstract_eq_false() {
        let result = global_eval("0n == false").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_bigint_abstract_eq_true() {
        let result = global_eval("1n == true").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // -- Comparison: relational --

    #[test]
    fn e2e_bigint_less_than() {
        let result = global_eval("1n < 2n").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_bigint_less_than_false() {
        let result = global_eval("2n < 1n").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn e2e_bigint_less_than_equal() {
        let result = global_eval("2n < 2n").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn e2e_bigint_greater_than() {
        let result = global_eval("2n > 1n").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_bigint_greater_than_false() {
        let result = global_eval("1n > 2n").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn e2e_bigint_lte() {
        let result = global_eval("2n <= 2n").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_bigint_gte() {
        let result = global_eval("2n >= 2n").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_bigint_mixed_lt_number() {
        let result = global_eval("1n < 2").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_bigint_mixed_gt_number() {
        let result = global_eval("2n > 1").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_bigint_mixed_lt_reverse() {
        let result = global_eval("1 < 2n").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_bigint_mixed_gt_reverse() {
        let result = global_eval("2 > 1n").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // -- No implicit coercion --

    #[test]
    fn e2e_bigint_to_number_error() {
        assert!(global_eval("Number(1n)").is_err());
    }

    // -- String coercion --

    #[test]
    fn e2e_bigint_string_concat() {
        let result = global_eval("'' + 42n").unwrap();
        assert_eq!(result, JsValue::String("42".into()));
    }

    #[test]
    fn e2e_bigint_string_concat_reverse() {
        let result = global_eval("42n + ''").unwrap();
        assert_eq!(result, JsValue::String("42".into()));
    }

    #[test]
    fn e2e_bigint_string_concat_negative() {
        let result = global_eval("'' + -42n").unwrap();
        assert_eq!(result, JsValue::String("-42".into()));
    }

    // -- BigInt() constructor --

    #[test]
    fn e2e_bigint_constructor_number() {
        let result = global_eval("BigInt(42)").unwrap();
        assert_eq!(result, JsValue::BigInt(42));
    }

    #[test]
    fn e2e_bigint_constructor_zero() {
        let result = global_eval("BigInt(0)").unwrap();
        assert_eq!(result, JsValue::BigInt(0));
    }

    #[test]
    fn e2e_bigint_constructor_negative() {
        let result = global_eval("BigInt(-5)").unwrap();
        assert_eq!(result, JsValue::BigInt(-5));
    }

    #[test]
    fn e2e_bigint_constructor_string() {
        let result = global_eval("BigInt('123')").unwrap();
        assert_eq!(result, JsValue::BigInt(123));
    }

    #[test]
    fn e2e_bigint_constructor_string_hex() {
        let result = global_eval("BigInt('0xff')").unwrap();
        assert_eq!(result, JsValue::BigInt(255));
    }

    #[test]
    fn e2e_bigint_constructor_string_octal() {
        let result = global_eval("BigInt('0o77')").unwrap();
        assert_eq!(result, JsValue::BigInt(63));
    }

    #[test]
    fn e2e_bigint_constructor_string_binary() {
        let result = global_eval("BigInt('0b1010')").unwrap();
        assert_eq!(result, JsValue::BigInt(10));
    }

    #[test]
    fn e2e_bigint_constructor_bool_true() {
        let result = global_eval("BigInt(true)").unwrap();
        assert_eq!(result, JsValue::BigInt(1));
    }

    #[test]
    fn e2e_bigint_constructor_bool_false() {
        let result = global_eval("BigInt(false)").unwrap();
        assert_eq!(result, JsValue::BigInt(0));
    }

    #[test]
    fn e2e_bigint_constructor_bigint() {
        let result = global_eval("BigInt(42n)").unwrap();
        assert_eq!(result, JsValue::BigInt(42));
    }

    #[test]
    fn e2e_bigint_constructor_float_error() {
        assert!(global_eval("BigInt(1.5)").is_err());
    }

    #[test]
    fn e2e_bigint_constructor_nan_error() {
        assert!(global_eval("BigInt(NaN)").is_err());
    }

    #[test]
    fn e2e_bigint_constructor_infinity_error() {
        assert!(global_eval("BigInt(Infinity)").is_err());
    }

    #[test]
    fn e2e_bigint_constructor_invalid_string_error() {
        assert!(global_eval("BigInt('hello')").is_err());
    }

    #[test]
    fn e2e_bigint_constructor_undefined_error() {
        assert!(global_eval("BigInt(undefined)").is_err());
    }

    #[test]
    fn e2e_bigint_constructor_null_error() {
        assert!(global_eval("BigInt(null)").is_err());
    }

    // -- BigInt.asIntN --

    #[test]
    fn e2e_bigint_as_int_n_positive() {
        let result = global_eval("BigInt.asIntN(8, 127n)").unwrap();
        assert_eq!(result, JsValue::BigInt(127));
    }

    #[test]
    fn e2e_bigint_as_int_n_overflow() {
        let result = global_eval("BigInt.asIntN(8, 128n)").unwrap();
        assert_eq!(result, JsValue::BigInt(-128));
    }

    #[test]
    fn e2e_bigint_as_int_n_negative() {
        let result = global_eval("BigInt.asIntN(8, -129n)").unwrap();
        assert_eq!(result, JsValue::BigInt(127));
    }

    #[test]
    fn e2e_bigint_as_int_n_zero_bits() {
        let result = global_eval("BigInt.asIntN(0, 42n)").unwrap();
        assert_eq!(result, JsValue::BigInt(0));
    }

    #[test]
    fn e2e_bigint_as_int_n_16() {
        let result = global_eval("BigInt.asIntN(16, 32768n)").unwrap();
        assert_eq!(result, JsValue::BigInt(-32768));
    }

    #[test]
    fn e2e_bigint_as_int_n_32() {
        let result = global_eval("BigInt.asIntN(32, 2147483648n)").unwrap();
        assert_eq!(result, JsValue::BigInt(-2147483648));
    }

    // -- BigInt.asUintN --

    #[test]
    fn e2e_bigint_as_uint_n_positive() {
        let result = global_eval("BigInt.asUintN(8, 255n)").unwrap();
        assert_eq!(result, JsValue::BigInt(255));
    }

    #[test]
    fn e2e_bigint_as_uint_n_overflow() {
        let result = global_eval("BigInt.asUintN(8, 256n)").unwrap();
        assert_eq!(result, JsValue::BigInt(0));
    }

    #[test]
    fn e2e_bigint_as_uint_n_large() {
        let result = global_eval("BigInt.asUintN(8, 257n)").unwrap();
        assert_eq!(result, JsValue::BigInt(1));
    }

    #[test]
    fn e2e_bigint_as_uint_n_zero_bits() {
        let result = global_eval("BigInt.asUintN(0, 42n)").unwrap();
        assert_eq!(result, JsValue::BigInt(0));
    }

    #[test]
    fn e2e_bigint_as_uint_n_16() {
        let result = global_eval("BigInt.asUintN(16, 65536n)").unwrap();
        assert_eq!(result, JsValue::BigInt(0));
    }

    #[test]
    fn e2e_bigint_as_uint_n_negative() {
        let result = global_eval("BigInt.asUintN(8, -1n)").unwrap();
        assert_eq!(result, JsValue::BigInt(255));
    }

    // -- ToBoolean --

    #[test]
    fn e2e_bigint_to_boolean_truthy() {
        let result = global_eval("!!42n").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_bigint_to_boolean_falsy() {
        let result = global_eval("!!0n").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn e2e_bigint_to_boolean_negative_truthy() {
        let result = global_eval("!!-1n").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // -- Increment / Decrement --

    #[test]
    fn e2e_bigint_increment() {
        let result = global_eval("let x = 5n; ++x").unwrap();
        assert_eq!(result, JsValue::BigInt(6));
    }

    #[test]
    fn e2e_bigint_decrement() {
        let result = global_eval("let x = 5n; --x").unwrap();
        assert_eq!(result, JsValue::BigInt(4));
    }

    #[test]
    fn e2e_bigint_increment_zero() {
        let result = global_eval("let x = 0n; ++x").unwrap();
        assert_eq!(result, JsValue::BigInt(1));
    }

    #[test]
    fn e2e_bigint_decrement_zero() {
        let result = global_eval("let x = 0n; --x").unwrap();
        assert_eq!(result, JsValue::BigInt(-1));
    }

    #[test]
    fn e2e_bigint_postfix_increment() {
        let result = global_eval("let x = 5n; x++; x").unwrap();
        assert_eq!(result, JsValue::BigInt(6));
    }

    #[test]
    fn e2e_bigint_postfix_decrement() {
        let result = global_eval("let x = 5n; x--; x").unwrap();
        assert_eq!(result, JsValue::BigInt(4));
    }

    // -- Variables and assignment --

    #[test]
    fn e2e_bigint_let_variable() {
        let result = global_eval("let x = 42n; x").unwrap();
        assert_eq!(result, JsValue::BigInt(42));
    }

    #[test]
    fn e2e_bigint_reassign() {
        let result = global_eval("let x = 1n; x = 2n; x").unwrap();
        assert_eq!(result, JsValue::BigInt(2));
    }

    #[test]
    fn e2e_bigint_add_assign() {
        let result = global_eval("let x = 10n; x += 5n; x").unwrap();
        assert_eq!(result, JsValue::BigInt(15));
    }

    #[test]
    fn e2e_bigint_sub_assign() {
        let result = global_eval("let x = 10n; x -= 3n; x").unwrap();
        assert_eq!(result, JsValue::BigInt(7));
    }

    #[test]
    fn e2e_bigint_mul_assign() {
        let result = global_eval("let x = 3n; x *= 4n; x").unwrap();
        assert_eq!(result, JsValue::BigInt(12));
    }

    #[test]
    fn e2e_bigint_div_assign() {
        let result = global_eval("let x = 10n; x /= 3n; x").unwrap();
        assert_eq!(result, JsValue::BigInt(3));
    }

    #[test]
    fn e2e_bigint_mod_assign() {
        let result = global_eval("let x = 10n; x %= 3n; x").unwrap();
        assert_eq!(result, JsValue::BigInt(1));
    }

    #[test]
    fn e2e_bigint_exp_assign() {
        let result = global_eval("let x = 2n; x **= 10n; x").unwrap();
        assert_eq!(result, JsValue::BigInt(1024));
    }

    #[test]
    fn e2e_bigint_bitwise_or_assign() {
        let result = global_eval("let x = 5n; x |= 3n; x").unwrap();
        assert_eq!(result, JsValue::BigInt(7));
    }

    #[test]
    fn e2e_bigint_bitwise_and_assign() {
        let result = global_eval("let x = 5n; x &= 3n; x").unwrap();
        assert_eq!(result, JsValue::BigInt(1));
    }

    #[test]
    fn e2e_bigint_bitwise_xor_assign() {
        let result = global_eval("let x = 5n; x ^= 3n; x").unwrap();
        assert_eq!(result, JsValue::BigInt(6));
    }

    #[test]
    fn e2e_bigint_shift_left_assign() {
        let result = global_eval("let x = 1n; x <<= 10n; x").unwrap();
        assert_eq!(result, JsValue::BigInt(1024));
    }

    #[test]
    fn e2e_bigint_shift_right_assign() {
        let result = global_eval("let x = 1024n; x >>= 5n; x").unwrap();
        assert_eq!(result, JsValue::BigInt(32));
    }

    // -- Control flow with BigInt --

    #[test]
    fn e2e_bigint_if_truthy() {
        let result = global_eval("if (1n) { 42n } else { 0n }").unwrap();
        assert_eq!(result, JsValue::BigInt(42));
    }

    #[test]
    fn e2e_bigint_if_falsy() {
        let result = global_eval("if (0n) { 42n } else { 0n }").unwrap();
        assert_eq!(result, JsValue::BigInt(0));
    }

    #[test]
    fn e2e_bigint_ternary() {
        let result = global_eval("1n ? 'yes' : 'no'").unwrap();
        assert_eq!(result, JsValue::String("yes".into()));
    }

    #[test]
    fn e2e_bigint_ternary_falsy() {
        let result = global_eval("0n ? 'yes' : 'no'").unwrap();
        assert_eq!(result, JsValue::String("no".into()));
    }

    // -- Loop with BigInt --

    #[test]
    fn e2e_bigint_while_loop() {
        let result =
            global_eval("let x = 0n; let i = 0n; while (i < 5n) { x += i; i += 1n; } x").unwrap();
        assert_eq!(result, JsValue::BigInt(10)); // 0+1+2+3+4
    }

    #[test]
    fn e2e_bigint_for_loop() {
        let result =
            global_eval("let sum = 0n; for (let i = 1n; i <= 5n; i += 1n) { sum += i; } sum")
                .unwrap();
        assert_eq!(result, JsValue::BigInt(15)); // 1+2+3+4+5
    }

    // -- Function with BigInt --

    #[test]
    fn e2e_bigint_function_return() {
        let result = global_eval("function f() { return 42n; } f()").unwrap();
        assert_eq!(result, JsValue::BigInt(42));
    }

    #[test]
    fn e2e_bigint_function_parameter() {
        let result = global_eval("function f(x) { return x + 1n; } f(41n)").unwrap();
        assert_eq!(result, JsValue::BigInt(42));
    }

    #[test]
    fn e2e_bigint_function_multiple_params() {
        let result = global_eval("function add(a, b) { return a + b; } add(20n, 22n)").unwrap();
        assert_eq!(result, JsValue::BigInt(42));
    }

    // -- Complex expressions --

    #[test]
    fn e2e_bigint_chained_arithmetic() {
        let result = global_eval("1n + 2n + 3n + 4n + 5n").unwrap();
        assert_eq!(result, JsValue::BigInt(15));
    }

    #[test]
    fn e2e_bigint_parenthesized() {
        let result = global_eval("(2n + 3n) * 4n").unwrap();
        assert_eq!(result, JsValue::BigInt(20));
    }

    #[test]
    fn e2e_bigint_nested_operations() {
        let result = global_eval("2n ** 3n + 4n * 2n - 1n").unwrap();
        assert_eq!(result, JsValue::BigInt(15)); // 8 + 8 - 1
    }

    #[test]
    fn e2e_bigint_factorial_iterative() {
        let result = global_eval(
            "function factorial(n) { let r = 1n; for (let i = 2n; i <= n; i += 1n) { r *= i; } return r; } factorial(10n)"
        ).unwrap();
        assert_eq!(result, JsValue::BigInt(3_628_800));
    }

    #[test]
    fn e2e_bigint_fibonacci() {
        let result = global_eval(
            "function fib(n) { let a = 0n; let b = 1n; for (let i = 0n; i < n; i += 1n) { let t = b; b = a + b; a = t; } return a; } fib(20n)"
        ).unwrap();
        assert_eq!(result, JsValue::BigInt(6765));
    }

    // -- Edge cases --

    #[test]
    fn e2e_bigint_max_safe_integer_plus_one() {
        let result = global_eval("9007199254740991n + 1n").unwrap();
        assert_eq!(result, JsValue::BigInt(9_007_199_254_740_992));
    }

    #[test]
    fn e2e_bigint_very_large_mul() {
        let result = global_eval("1000000000n * 1000000000n").unwrap();
        assert_eq!(result, JsValue::BigInt(1_000_000_000_000_000_000));
    }

    #[test]
    fn e2e_bigint_negative_large() {
        let result = global_eval("-9007199254740993n").unwrap();
        assert_eq!(result, JsValue::BigInt(-9_007_199_254_740_993));
    }

    // -- install_globals includes BigInt --

    #[test]
    fn test_install_globals_has_bigint() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        assert!(globals.contains_key("BigInt"));
    }

    #[test]
    fn test_bigint_object_has_as_int_n() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let bigint = globals.get("BigInt").unwrap();
        if let JsValue::PlainObject(map) = bigint {
            let map = map.borrow();
            assert!(map.contains_key("asIntN"));
            assert!(map.contains_key("asUintN"));
            assert!(map.contains_key("__call__"));
        } else {
            panic!("BigInt should be a PlainObject");
        }
    }

    // -- BigInt constructor direct call tests --

    #[test]
    fn test_bigint_constructor_from_smi() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let bigint = globals.get("BigInt").unwrap();
        if let JsValue::PlainObject(map) = bigint {
            let call = map.borrow().get("__call__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = call {
                let result = f(vec![JsValue::Smi(42)]).unwrap();
                assert_eq!(result, JsValue::BigInt(42));
            }
        }
    }

    #[test]
    fn test_bigint_constructor_from_heap_number() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let bigint = globals.get("BigInt").unwrap();
        if let JsValue::PlainObject(map) = bigint {
            let call = map.borrow().get("__call__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = call {
                let result = f(vec![JsValue::HeapNumber(100.0)]).unwrap();
                assert_eq!(result, JsValue::BigInt(100));
            }
        }
    }

    #[test]
    fn test_bigint_constructor_from_float_fails() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let bigint = globals.get("BigInt").unwrap();
        if let JsValue::PlainObject(map) = bigint {
            let call = map.borrow().get("__call__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = call {
                assert!(f(vec![JsValue::HeapNumber(1.5)]).is_err());
            }
        }
    }

    #[test]
    fn test_bigint_constructor_from_nan_fails() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let bigint = globals.get("BigInt").unwrap();
        if let JsValue::PlainObject(map) = bigint {
            let call = map.borrow().get("__call__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = call {
                assert!(f(vec![JsValue::HeapNumber(f64::NAN)]).is_err());
            }
        }
    }

    #[test]
    fn test_bigint_constructor_from_string_negative() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let bigint = globals.get("BigInt").unwrap();
        if let JsValue::PlainObject(map) = bigint {
            let call = map.borrow().get("__call__").cloned().unwrap();
            if let JsValue::NativeFunction(f) = call {
                let result = f(vec![JsValue::String("-123".into())]).unwrap();
                assert_eq!(result, JsValue::BigInt(-123));
            }
        }
    }

    #[test]
    fn test_bigint_as_int_n_direct() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let bigint = globals.get("BigInt").unwrap();
        if let JsValue::PlainObject(map) = bigint {
            let as_int_n = map.borrow().get("asIntN").cloned().unwrap();
            if let JsValue::NativeFunction(f) = as_int_n {
                // asIntN(8, 255n) = -1n
                let result = f(vec![JsValue::Smi(8), JsValue::BigInt(255)]).unwrap();
                assert_eq!(result, JsValue::BigInt(-1));
            }
        }
    }

    #[test]
    fn test_bigint_as_uint_n_direct() {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        let bigint = globals.get("BigInt").unwrap();
        if let JsValue::PlainObject(map) = bigint {
            let as_uint_n = map.borrow().get("asUintN").cloned().unwrap();
            if let JsValue::NativeFunction(f) = as_uint_n {
                // asUintN(8, 256n) = 0n
                let result = f(vec![JsValue::Smi(8), JsValue::BigInt(256)]).unwrap();
                assert_eq!(result, JsValue::BigInt(0));
            }
        }
    }

    // -- Additional edge cases and combinations --

    #[test]
    fn e2e_bigint_logical_and_truthy() {
        let result = global_eval("1n && 2n").unwrap();
        assert_eq!(result, JsValue::BigInt(2));
    }

    #[test]
    fn e2e_bigint_logical_and_falsy() {
        let result = global_eval("0n && 2n").unwrap();
        assert_eq!(result, JsValue::BigInt(0));
    }

    #[test]
    fn e2e_bigint_logical_or_truthy() {
        let result = global_eval("1n || 2n").unwrap();
        assert_eq!(result, JsValue::BigInt(1));
    }

    #[test]
    fn e2e_bigint_logical_or_falsy() {
        let result = global_eval("0n || 2n").unwrap();
        assert_eq!(result, JsValue::BigInt(2));
    }

    #[test]
    fn e2e_bigint_nullish_coalescing() {
        let result = global_eval("0n ?? 42n").unwrap();
        assert_eq!(result, JsValue::BigInt(0));
    }

    #[test]
    fn e2e_bigint_power_of_two() {
        let result = global_eval("2n ** 32n").unwrap();
        assert_eq!(result, JsValue::BigInt(4_294_967_296));
    }

    #[test]
    fn e2e_bigint_large_shift() {
        let result = global_eval("1n << 32n").unwrap();
        assert_eq!(result, JsValue::BigInt(4_294_967_296));
    }

    #[test]
    fn e2e_bigint_xor_identity() {
        let result = global_eval("42n ^ 42n").unwrap();
        assert_eq!(result, JsValue::BigInt(0));
    }

    #[test]
    fn e2e_bigint_and_all_ones() {
        let result = global_eval("255n & 15n").unwrap();
        assert_eq!(result, JsValue::BigInt(15));
    }

    #[test]
    fn e2e_bigint_or_complement() {
        let result = global_eval("240n | 15n").unwrap();
        assert_eq!(result, JsValue::BigInt(255));
    }

    #[test]
    fn e2e_bigint_complex_expression() {
        let result = global_eval("(10n + 20n) * 2n - 5n / 1n").unwrap();
        assert_eq!(result, JsValue::BigInt(55));
    }

    #[test]
    fn e2e_bigint_switch_case() {
        let result = global_eval(
            "let x = 2n; let r; switch (true) { case x === 1n: r = 'one'; break; case x === 2n: r = 'two'; break; default: r = 'other'; } r"
        ).unwrap();
        assert_eq!(result, JsValue::String("two".into()));
    }

    #[test]
    fn e2e_bigint_conditional_chain() {
        let result =
            global_eval("let x = 5n; x > 3n ? x > 4n ? 'big' : 'medium' : 'small'").unwrap();
        assert_eq!(result, JsValue::String("big".into()));
    }

    #[test]
    fn e2e_bigint_recursive_sum() {
        let result = global_eval(
            "function sum(n) { if (n <= 0n) return 0n; return n + sum(n - 1n); } sum(10n)",
        )
        .unwrap();
        assert_eq!(result, JsValue::BigInt(55));
    }

    #[test]
    fn e2e_bigint_power_iterative() {
        let result = global_eval(
            "function pow(b, e) { let r = 1n; for (let i = 0n; i < e; i += 1n) { r *= b; } return r; } pow(3n, 5n)"
        ).unwrap();
        assert_eq!(result, JsValue::BigInt(243));
    }

    #[test]
    fn e2e_bigint_gcd() {
        let result = global_eval(
            "function gcd(a, b) { while (b !== 0n) { let t = b; b = a % b; a = t; } return a; } gcd(48n, 18n)"
        ).unwrap();
        assert_eq!(result, JsValue::BigInt(6));
    }

    #[test]
    fn e2e_bigint_abs() {
        let result = global_eval("let x = -42n; x < 0n ? -x : x").unwrap();
        assert_eq!(result, JsValue::BigInt(42));
    }

    #[test]
    fn e2e_bigint_min_of_two() {
        let result = global_eval("let a = 10n; let b = 20n; a < b ? a : b").unwrap();
        assert_eq!(result, JsValue::BigInt(10));
    }

    #[test]
    fn e2e_bigint_max_of_two() {
        let result = global_eval("let a = 10n; let b = 20n; a > b ? a : b").unwrap();
        assert_eq!(result, JsValue::BigInt(20));
    }

    #[test]
    fn e2e_bigint_div_truncates_toward_zero() {
        // JS BigInt division truncates toward zero
        let result = global_eval("7n / 2n").unwrap();
        assert_eq!(result, JsValue::BigInt(3));
    }

    #[test]
    fn e2e_bigint_div_negative_truncates_toward_zero() {
        let result = global_eval("-7n / 2n").unwrap();
        assert_eq!(result, JsValue::BigInt(-3));
    }

    #[test]
    fn e2e_bigint_mod_sign_follows_dividend() {
        let result = global_eval("-7n % 2n").unwrap();
        assert_eq!(result, JsValue::BigInt(-1));
    }

    #[test]
    fn e2e_bigint_constructor_empty_string_error() {
        assert!(global_eval("BigInt('')").is_err());
    }

    #[test]
    fn e2e_bigint_string_with_suffix_n() {
        // BigInt('42n') should fail — n suffix not valid in constructor
        assert!(global_eval("BigInt('42n')").is_err());
    }

    #[test]
    fn e2e_bigint_constructor_large_positive() {
        let result = global_eval("BigInt('170141183460469231731687303715884105727')").unwrap();
        assert_eq!(result, JsValue::BigInt(i128::MAX));
    }

    // ── Prototype chain tests ───────────────────────────────────────────

    /// `Object.create(proto)` sets up prototype chain for property lookup.
    #[test]
    fn e2e_object_create_prototype_chain() {
        let result = global_eval(
            r#"
            var proto = { x: 42 };
            var child = Object.create(proto);
            child.x
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// Own property shadows prototype property.
    #[test]
    fn e2e_own_property_shadows_prototype() {
        let result = global_eval(
            r#"
            var proto = { x: 1 };
            var child = Object.create(proto);
            child.x = 99;
            child.x
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    /// `Object.getPrototypeOf` returns the prototype set by Object.create.
    #[test]
    fn e2e_get_prototype_of_created_object() {
        let result = global_eval(
            r#"
            var proto = { marker: true };
            var child = Object.create(proto);
            var p = Object.getPrototypeOf(child);
            p.marker
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `Object.setPrototypeOf` changes the prototype chain.
    #[test]
    fn e2e_set_prototype_of_changes_chain() {
        let result = global_eval(
            r#"
            var a = { val: 10 };
            var b = { val: 20 };
            var obj = Object.create(a);
            Object.setPrototypeOf(obj, b);
            obj.val
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(20));
    }

    /// `Object.hasOwn` returns false for inherited properties.
    #[test]
    fn e2e_has_own_false_for_inherited() {
        let result = global_eval(
            r#"
            var proto = { x: 1 };
            var child = Object.create(proto);
            Object.hasOwn(child, "x")
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    /// Multi-level prototype chain property resolution.
    #[test]
    fn e2e_multi_level_prototype_chain() {
        let result = global_eval(
            r#"
            var grandparent = { deep: 777 };
            var parent = Object.create(grandparent);
            var child = Object.create(parent);
            child.deep
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(777));
    }

    // ── Error spec-compliance e2e tests (issue #295) ─────────────────────

    /// `new Error("msg").name` → "Error"
    #[test]
    fn e2e_error_name_property() {
        let result = global_eval(r#"var e = new Error("msg"); e.name"#).unwrap();
        assert_eq!(result, JsValue::String("Error".to_string().into()));
    }

    /// `new TypeError("msg").name` → "TypeError"
    #[test]
    fn e2e_type_error_name() {
        let result = global_eval(r#"var e = new TypeError("msg"); e.name"#).unwrap();
        assert_eq!(result, JsValue::String("TypeError".to_string().into()));
    }

    /// `new RangeError("msg").name` → "RangeError"
    #[test]
    fn e2e_range_error_name() {
        let result = global_eval(r#"var e = new RangeError("msg"); e.name"#).unwrap();
        assert_eq!(result, JsValue::String("RangeError".to_string().into()));
    }

    /// `new ReferenceError("msg").name` → "ReferenceError"
    #[test]
    fn e2e_reference_error_name() {
        let result = global_eval(r#"var e = new ReferenceError("msg"); e.name"#).unwrap();
        assert_eq!(result, JsValue::String("ReferenceError".to_string().into()));
    }

    /// `new SyntaxError("msg").name` → "SyntaxError"
    #[test]
    fn e2e_syntax_error_name() {
        let result = global_eval(r#"var e = new SyntaxError("msg"); e.name"#).unwrap();
        assert_eq!(result, JsValue::String("SyntaxError".to_string().into()));
    }

    /// `new URIError("msg").name` → "URIError"
    #[test]
    fn e2e_uri_error_name() {
        let result = global_eval(r#"var e = new URIError("msg"); e.name"#).unwrap();
        assert_eq!(result, JsValue::String("URIError".to_string().into()));
    }

    /// `new EvalError("msg").name` → "EvalError"
    #[test]
    fn e2e_eval_error_name() {
        let result = global_eval(r#"var e = new EvalError("msg"); e.name"#).unwrap();
        assert_eq!(result, JsValue::String("EvalError".to_string().into()));
    }

    /// `new Error("msg").message` → "msg"
    #[test]
    fn e2e_error_message_property() {
        let result = global_eval(r#"var e = new Error("hello"); e.message"#).unwrap();
        assert_eq!(result, JsValue::String("hello".to_string().into()));
    }

    /// `new Error("msg").stack` starts with "Error: msg"
    #[test]
    fn e2e_error_stack_property() {
        let result = global_eval(r#"var e = new Error("msg"); e.stack"#).unwrap();
        if let JsValue::String(s) = result {
            assert!(
                s.starts_with("Error: msg"),
                "stack should start with error string: {s}"
            );
        } else {
            panic!("expected String for .stack");
        }
    }

    /// Error without cause: `.cause` → undefined
    #[test]
    fn e2e_error_cause_undefined_when_absent() {
        let result = global_eval(r#"var e = new Error("msg"); e.cause"#).unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    /// AggregateError constructor: `.name` → "AggregateError"
    #[test]
    fn e2e_aggregate_error_name() {
        let result = global_eval(r#"var e = new AggregateError([], "msg"); e.name"#).unwrap();
        assert_eq!(result, JsValue::String("AggregateError".to_string().into()));
    }

    /// AggregateError constructor: `.message` → "msg"
    #[test]
    fn e2e_aggregate_error_message() {
        let result = global_eval(r#"var e = new AggregateError([], "msg"); e.message"#).unwrap();
        assert_eq!(result, JsValue::String("msg".to_string().into()));
    }

    /// `new Error("msg").toString()` → "Error: msg"
    #[test]
    fn e2e_error_to_string() {
        let result = global_eval(r#"var e = new Error("msg"); e.toString()"#).unwrap();
        assert_eq!(result, JsValue::String("Error: msg".to_string().into()));
    }

    /// `new TypeError("").toString()` → "TypeError"
    #[test]
    fn e2e_type_error_to_string_empty_message() {
        let result = global_eval(r#"var e = new TypeError(); e.toString()"#).unwrap();
        assert_eq!(result, JsValue::String("TypeError".to_string().into()));
    }

    /// `Error.captureStackTrace` is accessible as a function.
    #[test]
    fn e2e_error_capture_stack_trace_exists() {
        let result = global_eval("typeof Error.captureStackTrace").unwrap();
        assert_eq!(result, JsValue::String("function".into()));
    }

    /// `Error.stackTraceLimit` is a number.
    #[test]
    fn e2e_error_stack_trace_limit_exists() {
        let result = global_eval("typeof Error.stackTraceLimit").unwrap();
        assert_eq!(result, JsValue::String("number".into()));
    }

    /// `new.target` is the constructor when called via `new`.
    #[test]
    fn e2e_new_target_is_defined_in_constructor() {
        // When called via new, the Construct handler creates a this object
        // and returns it. new.target points to the constructor.
        // For now, verify basic construction works.
        let result = global_eval(
            r#"
            function Foo() {}
            var x = new Foo();
            typeof x
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::String("object".into()));
    }

    /// `new.target` is undefined in normal function calls.
    #[test]
    fn e2e_new_target_undefined_in_normal_call() {
        let result = global_eval(
            r#"
            function Bar() { return typeof new.target; }
            Bar()
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::String("undefined".into()));
    }

    // ── String ↔ RegExp delegation tests ────────────────────────────────────

    /// Helper: build a RegExp `JsValue` via `regexp_construct`.
    fn make_re(pattern: &str, flags: &str) -> JsValue {
        regexp_construct(&[
            JsValue::String(pattern.into()),
            JsValue::String(flags.into()),
        ])
        .unwrap()
    }

    /// Helper: build a `String` proto object via `install_globals` and extract it.
    fn string_proto() -> Rc<RefCell<PropertyMap>> {
        let mut globals = HashMap::new();
        install_globals(&mut globals);
        if let Some(JsValue::PlainObject(string_ctor)) = globals.get("String") {
            if let Some(JsValue::PlainObject(proto)) = string_ctor.borrow().get("prototype") {
                return Rc::clone(proto);
            }
        }
        panic!("String.prototype not found");
    }

    /// Helper: call a String.prototype method (first arg = this string).
    fn call_string_method(
        proto: &Rc<RefCell<PropertyMap>>,
        method: &str,
        args: Vec<JsValue>,
    ) -> StatorResult<JsValue> {
        if let Some(JsValue::NativeFunction(f)) = proto.borrow().get(method).cloned() {
            f(args)
        } else {
            panic!("String.prototype.{method} not found");
        }
    }

    #[test]
    fn test_string_match_delegates_to_regexp() {
        let proto = string_proto();
        let re = make_re(r"(\d+)", "");
        let result = call_string_method(
            &proto,
            "match",
            vec![JsValue::String("price 42 dollars".into()), re],
        )
        .unwrap();
        // Should delegate to @@match → exec-like result with "0" = "42"
        if let JsValue::PlainObject(map) = &result {
            assert_eq!(map.borrow().get("0"), Some(&JsValue::String("42".into())));
            assert_eq!(map.borrow().get("index"), Some(&JsValue::Smi(6)));
        } else {
            panic!("expected PlainObject match result, got {result:?}");
        }
    }

    #[test]
    fn test_string_search_delegates_to_regexp() {
        let proto = string_proto();
        let re = make_re(r"\d+", "");
        let result =
            call_string_method(&proto, "search", vec![JsValue::String("abc 42".into()), re])
                .unwrap();
        assert_eq!(result, JsValue::Smi(4));
    }

    #[test]
    fn test_string_replace_delegates_to_regexp() {
        let proto = string_proto();
        let re = make_re(r"\d+", "g");
        let result = call_string_method(
            &proto,
            "replace",
            vec![
                JsValue::String("a1 b2 c3".into()),
                re,
                JsValue::String("X".into()),
            ],
        )
        .unwrap();
        assert_eq!(result, JsValue::String("aX bX cX".into()));
    }

    #[test]
    fn test_string_split_delegates_to_regexp() {
        let proto = string_proto();
        let re = make_re(r"\s+", "");
        let result =
            call_string_method(&proto, "split", vec![JsValue::String("a  b  c".into()), re])
                .unwrap();
        if let JsValue::Array(arr) = &result {
            let arr = arr.borrow();
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], JsValue::String("a".into()));
            assert_eq!(arr[1], JsValue::String("b".into()));
            assert_eq!(arr[2], JsValue::String("c".into()));
        } else {
            panic!("expected Array, got {result:?}");
        }
    }

    #[test]
    fn test_string_replace_all_regexp_non_global_throws() {
        let proto = string_proto();
        let re = make_re(r"\d+", ""); // no 'g' flag
        let result = call_string_method(
            &proto,
            "replaceAll",
            vec![
                JsValue::String("a1 b2".into()),
                re,
                JsValue::String("X".into()),
            ],
        );
        assert!(
            result.is_err(),
            "replaceAll with non-global regexp should error"
        );
    }

    #[test]
    fn test_string_replace_all_regexp_global_delegates() {
        let proto = string_proto();
        let re = make_re(r"\d+", "g");
        let result = call_string_method(
            &proto,
            "replaceAll",
            vec![
                JsValue::String("a1 b2 c3".into()),
                re,
                JsValue::String("X".into()),
            ],
        )
        .unwrap();
        assert_eq!(result, JsValue::String("aX bX cX".into()));
    }

    // ── trimLeft / trimRight aliases ─────────────────────────────────────

    /// `trimLeft` is a legacy alias for `trimStart`.
    #[test]
    fn test_string_trim_left_alias() {
        let proto = string_proto();
        let result = call_string_method(
            &proto,
            "trimLeft",
            vec![JsValue::String("  hello  ".into())],
        )
        .unwrap();
        assert_eq!(result, JsValue::String("hello  ".into()));
    }

    /// `trimRight` is a legacy alias for `trimEnd`.
    #[test]
    fn test_string_trim_right_alias() {
        let proto = string_proto();
        let result = call_string_method(
            &proto,
            "trimRight",
            vec![JsValue::String("  hello  ".into())],
        )
        .unwrap();
        assert_eq!(result, JsValue::String("  hello".into()));
    }
}
