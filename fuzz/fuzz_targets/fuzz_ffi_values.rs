#![no_main]

//! Fuzz value creation, type-checking predicates, and ECMAScript coercions at
//! the FFI boundary.
//!
//! For each chunk of input bytes the fuzzer:
//!   1. Creates a value of a randomly selected type.
//!   2. Runs all `stator_value_is_*` predicates and asserts exactly the right
//!      one returns `true`.
//!   3. Runs `stator_value_type` and asserts the returned C string matches.
//!   4. Runs ECMAScript coercions (`to_number`, `to_boolean`, `to_string`,
//!      `to_int32`, `to_uint32`) and checks known invariants.
//!   5. Destroys the value.
//!
//! Verifies correctness and that no crash or undefined behaviour occurs.

use std::ffi::CStr;

use libfuzzer_sys::fuzz_target;
use stator_ffi::{
    stator_isolate_new, stator_isolate_dispose,
    stator_value_new_number, stator_value_new_string, stator_value_new_boolean,
    stator_value_new_undefined, stator_value_new_null, stator_value_new_object,
    stator_value_new_function_tag, stator_value_new_array_tag, stator_value_new_date_tag,
    stator_value_destroy,
    stator_value_type,
    stator_value_is_undefined, stator_value_is_null, stator_value_is_boolean,
    stator_value_is_number, stator_value_is_string, stator_value_is_object,
    stator_value_is_function, stator_value_is_array, stator_value_is_int32,
    stator_value_is_uint32, stator_value_is_date, stator_value_is_regexp,
    stator_value_is_promise, stator_value_is_map, stator_value_is_set,
    stator_value_as_number, stator_value_as_string,
    stator_value_to_number, stator_value_to_boolean, stator_value_to_string,
    stator_value_to_int32, stator_value_to_uint32,
    stator_value_strict_equals,
};

/// Number of value-type variants we cycle through.
const NUM_TYPES: u8 = 13;

fuzz_target!(|data: &[u8]| {
    // Create a single isolate for all operations in this run.
    let iso = stator_isolate_new();
    assert!(!iso.is_null(), "stator_isolate_new must not return null");

    // Process input in 4-byte chunks:
    //   byte[0]: value type selector (mod NUM_TYPES)
    //   byte[1]: numeric/boolean payload byte
    //   byte[2..3]: string length hint (used for string type)
    let mut chunk = data;
    while chunk.len() >= 4 {
        let type_sel = chunk[0] % NUM_TYPES;
        let payload = chunk[1];
        let str_len_hint = usize::from(chunk[2]) & 0x1f; // cap at 31
        let str_payload = chunk[3];
        chunk = &chunk[4..];

        // Build a short string from the payload bytes for the string case.
        let str_bytes: Vec<u8> = (0..str_len_hint).map(|i| str_payload.wrapping_add(i as u8)).collect();

        // --- Create the value ---
        let val = unsafe {
            match type_sel {
                0 => stator_value_new_undefined(iso),
                1 => stator_value_new_null(iso),
                2 => stator_value_new_boolean(iso, payload & 1 != 0),
                3 => stator_value_new_number(iso, f64::from(payload)),
                4 => stator_value_new_number(iso, f64::NAN),
                5 => stator_value_new_number(iso, f64::INFINITY),
                6 => stator_value_new_number(iso, f64::NEG_INFINITY),
                7 => stator_value_new_number(iso, 0.0),
                8 => {
                    let p = str_bytes.as_ptr() as *const std::ffi::c_char;
                    stator_value_new_string(iso, p, str_bytes.len())
                }
                9 => stator_value_new_object(iso),
                10 => stator_value_new_function_tag(iso),
                11 => stator_value_new_array_tag(iso),
                12 => stator_value_new_date_tag(iso),
                _ => stator_value_new_undefined(iso),
            }
        };
        assert!(!val.is_null(), "value constructor must not return null");

        // --- Run all type predicates ---
        // SAFETY: `val` is a live pointer we own.
        let is_undefined = unsafe { stator_value_is_undefined(val) };
        let is_null = unsafe { stator_value_is_null(val) };
        let is_boolean = unsafe { stator_value_is_boolean(val) };
        let is_number = unsafe { stator_value_is_number(val) };
        let is_string = unsafe { stator_value_is_string(val) };
        let is_object = unsafe { stator_value_is_object(val) };
        let is_function = unsafe { stator_value_is_function(val) };
        let is_array = unsafe { stator_value_is_array(val) };
        let is_date = unsafe { stator_value_is_date(val) };
        // These predicates cover types not generated in this fuzz target;
        // they are called to verify they don't crash.
        let _is_regexp = unsafe { stator_value_is_regexp(val) };
        let _is_promise = unsafe { stator_value_is_promise(val) };
        let _is_map = unsafe { stator_value_is_map(val) };
        let _is_set = unsafe { stator_value_is_set(val) };

        // Exactly one of the "leaf" type checks must be true for primitive types.
        // Object-subtypes (array, date, …) also satisfy is_object.
        match type_sel {
            0 => {
                assert!(is_undefined, "undefined value must satisfy is_undefined");
                assert!(!is_null && !is_boolean && !is_number && !is_string
                    && !is_function && !is_array && !is_date);
            }
            1 => {
                assert!(is_null, "null value must satisfy is_null");
                assert!(!is_undefined && !is_boolean && !is_number && !is_string
                    && !is_object && !is_function);
            }
            2 => {
                assert!(is_boolean, "boolean value must satisfy is_boolean");
                assert!(!is_undefined && !is_null && !is_number && !is_string
                    && !is_object && !is_function);
            }
            3..=7 => {
                assert!(is_number, "number value must satisfy is_number");
                assert!(!is_undefined && !is_null && !is_boolean && !is_string
                    && !is_object && !is_function);
            }
            8 => {
                assert!(is_string, "string value must satisfy is_string");
                assert!(!is_undefined && !is_null && !is_boolean && !is_number
                    && !is_object && !is_function);
            }
            9 => {
                assert!(is_object, "object value must satisfy is_object");
                assert!(!is_undefined && !is_null && !is_boolean && !is_number
                    && !is_string && !is_function);
            }
            10 => {
                assert!(is_function, "function value must satisfy is_function");
                assert!(!is_undefined && !is_null && !is_boolean && !is_number
                    && !is_string);
            }
            11 => {
                assert!(is_array, "array value must satisfy is_array");
                assert!(is_object, "array must also satisfy is_object");
            }
            12 => {
                assert!(is_date, "date value must satisfy is_date");
                assert!(is_object, "date must also satisfy is_object");
            }
            _ => {}
        }

        // --- Verify stator_value_type ---
        // SAFETY: `val` is a live pointer we own.
        let type_str = unsafe { stator_value_type(val) };
        assert!(!type_str.is_null(), "stator_value_type must not return null");
        // SAFETY: the returned pointer is a static C string literal.
        let type_cstr = unsafe { CStr::from_ptr(type_str) };
        let type_name = type_cstr.to_str().unwrap_or("");
        match type_sel {
            0 => assert_eq!(type_name, "undefined"),
            1 => assert_eq!(type_name, "object"), // typeof null === "object"
            2 => assert_eq!(type_name, "boolean"),
            3..=7 => assert_eq!(type_name, "number"),
            8 => assert_eq!(type_name, "string"),
            9 | 11 | 12 => assert_eq!(type_name, "object"),
            10 => assert_eq!(type_name, "function"),
            _ => {}
        }

        // --- Coercion invariants ---

        // to_number must not crash for any type.
        // SAFETY: `val` is a live pointer we own.
        let as_num = unsafe { stator_value_to_number(val) };
        // stator_value_as_number is the "raw" accessor (no coercion).
        let raw_num = unsafe { stator_value_as_number(val) };

        // For actual number values, to_number should equal the stored value.
        if (3..=7).contains(&type_sel) {
            // Both representations (NaN, inf, finite) must agree.
            if as_num.is_nan() {
                assert!(raw_num.is_nan(), "NaN number: as_number must also be NaN");
            } else {
                assert_eq!(as_num, raw_num, "number coercion must round-trip");
            }
        }

        // to_boolean must not crash.
        // SAFETY: `val` is a live pointer we own.
        let _as_bool = unsafe { stator_value_to_boolean(val) };

        // to_int32 / to_uint32 must not crash.
        // SAFETY: `val` is a live pointer we own.
        let _as_i32 = unsafe { stator_value_to_int32(val) };
        let _as_u32 = unsafe { stator_value_to_uint32(val) };

        // is_int32 / is_uint32 consistency:
        // If is_int32 is true the value must be a number.
        let is_i32 = unsafe { stator_value_is_int32(val) };
        let is_u32 = unsafe { stator_value_is_uint32(val) };
        if is_i32 {
            assert!(is_number, "is_int32 implies is_number");
        }
        if is_u32 {
            assert!(is_number, "is_uint32 implies is_number");
        }

        // stator_value_as_string must not crash and must return a valid pointer.
        // SAFETY: `val` is a live pointer we own.
        let raw_str = unsafe { stator_value_as_string(val) };
        assert!(!raw_str.is_null(), "stator_value_as_string must not return null");

        // to_string must produce a valid value.
        // SAFETY: `iso` is live; `val` is live.
        let str_val = unsafe { stator_value_to_string(iso, val) };
        assert!(!str_val.is_null(), "stator_value_to_string must not return null");
        // The result must satisfy is_string.
        // SAFETY: `str_val` is a live pointer we own.
        let str_is_string = unsafe { stator_value_is_string(str_val) };
        assert!(str_is_string, "stator_value_to_string result must be a string");

        // strict_equals(v, v) must be true for non-NaN primitives.
        // SAFETY: both pointers are live.
        let self_eq = unsafe { stator_value_strict_equals(val, val) };
        // NaN !== NaN; object handles are never equal.
        let nan_or_obj = ((3..=7).contains(&type_sel) && as_num.is_nan())
            || type_sel >= 9;
        if !nan_or_obj {
            assert!(self_eq, "strict_equals(v, v) must be true for non-NaN primitives");
        }

        // Destroy the to_string result (we own it; no scope is open).
        // SAFETY: `str_val` is a live Box-backed pointer we own.
        unsafe { stator_value_destroy(str_val) };

        // Destroy the original value.
        // SAFETY: `val` is a live Box-backed pointer we own.
        unsafe { stator_value_destroy(val) };
    }

    // Dispose the isolate.
    // SAFETY: `iso` is a live pointer we own; all values have been destroyed.
    unsafe { stator_isolate_dispose(iso) };
});
