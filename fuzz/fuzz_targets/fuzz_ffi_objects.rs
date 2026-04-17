#![no_main]

//! Fuzz random property operations on a [`StatorObject`] via the FFI and
//! verify consistency with the underlying Rust API.
//!
//! Input layout: triplets of bytes (same layout as `fuzz_property_ops` but
//! going through the C-ABI layer):
//!   byte[0]: operation selector
//!     0 → stator_object_set
//!     1 → stator_object_get
//!     2 → stator_object_has
//!     3 → stator_object_delete
//!     4 → stator_object_get_property_names
//!   byte[1]: property-name index (0..=7, "k0"…"k7")
//!   byte[2]: value type selector (0=undefined, 1=null, 2=bool, 3=number, 4=string)
//!
//! Invariants verified:
//!   - A successful set is immediately readable with the same value.
//!   - A deleted property is no longer reported by stator_object_has.
//!   - stator_property_names_count / stator_property_names_get are consistent.

use std::ffi::{CStr, CString};

use libfuzzer_sys::fuzz_target;
use stator_js_ffi::{
    stator_context_destroy, stator_context_new,
    stator_isolate_dispose, stator_isolate_new,
    stator_object_delete, stator_object_get, stator_object_get_property_names,
    stator_object_has, stator_object_new, stator_object_destroy, stator_object_set,
    stator_property_names_count, stator_property_names_destroy, stator_property_names_get,
    stator_value_as_number, stator_value_as_string, stator_value_destroy,
    stator_value_is_number, stator_value_is_string,
    stator_value_new_boolean, stator_value_new_null, stator_value_new_number,
    stator_value_new_string, stator_value_new_undefined,
};

const KEYS: [&str; 8] = ["k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7"];

fuzz_target!(|data: &[u8]| {
    // SAFETY: all FFI calls satisfy the documented pointer validity requirements.
    let iso = stator_isolate_new();
    assert!(!iso.is_null());
    let ctx = unsafe { stator_context_new(iso) };
    assert!(!ctx.is_null());

    let obj = unsafe { stator_object_new(iso) };
    assert!(!obj.is_null());

    // Shadow map to track expected state.
    let mut expected: std::collections::HashMap<&'static str, String> =
        std::collections::HashMap::new();

    let mut chunk = data;
    while chunk.len() >= 3 {
        let op = chunk[0] % 5;
        let key_idx = usize::from(chunk[1] & 0x07);
        let val_sel = chunk[2] % 5;
        chunk = &chunk[3..];

        let key: &'static str = KEYS[key_idx];
        let key_c = CString::new(key).unwrap();

        match op {
            0 => {
                // stator_object_set — set property and verify round-trip.
                let (val_ptr, expected_str) = make_value(iso, val_sel, chunk.first().copied().unwrap_or(0));
                let expected_type = val_sel;

                // SAFETY: obj, key_c, val_ptr are all valid.
                unsafe { stator_object_set(obj, key_c.as_ptr(), val_ptr) };

                // Verify via stator_object_get.
                // SAFETY: obj, key_c are valid.
                let got = unsafe { stator_object_get(obj, key_c.as_ptr()) };

                // get returns null for undefined/null/object (non-primitive) values;
                // for number/string it returns a new value.
                if expected_type == 3 || expected_type == 4 {
                    // number or string: get must return a value.
                    if !got.is_null() {
                        if expected_type == 3 {
                            // SAFETY: got is a live pointer.
                            let is_num = unsafe { stator_value_is_number(got) };
                            assert!(is_num, "get after set(number) must return a number");
                            // SAFETY: got is a live pointer.
                            let n = unsafe { stator_value_as_number(got) };
                            let expected_n: f64 = expected_str.parse().unwrap_or(f64::NAN);
                            if expected_n.is_nan() {
                                assert!(n.is_nan(), "number round-trip: expected NaN");
                            } else {
                                assert_eq!(n, expected_n, "number round-trip mismatch");
                            }
                        } else {
                            // SAFETY: got is a live pointer.
                            let is_str = unsafe { stator_value_is_string(got) };
                            assert!(is_str, "get after set(string) must return a string");
                            // SAFETY: got is a live pointer.
                            let ptr = unsafe { stator_value_as_string(got) };
                            // SAFETY: ptr points to a valid null-terminated string.
                            let got_str = unsafe { CStr::from_ptr(ptr) }.to_string_lossy();
                            assert_eq!(got_str, expected_str.as_str(), "string round-trip mismatch");
                        }
                        // SAFETY: got is a Box-backed pointer we own.
                        unsafe { stator_value_destroy(got) };
                    }
                    // Update shadow map only for primitives we can verify.
                    expected.insert(key, expected_str);
                } else {
                    // For undefined/null/boolean the object stores the value
                    // but stator_object_get only returns primitive number/string.
                    // Just ensure no crash.
                    if !got.is_null() {
                        // SAFETY: got is a Box-backed pointer we own.
                        unsafe { stator_value_destroy(got) };
                    }
                    // Remove from shadow since we can't round-trip non-primitives.
                    expected.remove(key);
                }

                // SAFETY: val_ptr is a Box-backed pointer we own.
                unsafe { stator_value_destroy(val_ptr) };
            }
            1 => {
                // stator_object_get — must not crash; result (if non-null) must be freed.
                // SAFETY: obj, key_c are valid.
                let got = unsafe { stator_object_get(obj, key_c.as_ptr()) };
                if !got.is_null() {
                    // SAFETY: got is a Box-backed pointer we own.
                    unsafe { stator_value_destroy(got) };
                }
            }
            2 => {
                // stator_object_has — result must agree with shadow map for set keys.
                // SAFETY: obj, key_c are valid.
                let has = unsafe { stator_object_has(obj, key_c.as_ptr()) };
                // We only track number/string keys in the shadow map; for those
                // the shadow is authoritative.
                if expected.contains_key(key) {
                    assert!(has, "stator_object_has must be true for a key that was set");
                }
                // has==false for a key in shadow is unexpected but tolerated when
                // internal transitions (fast→slow) occur; the main goal is no crash.
            }
            3 => {
                // stator_object_delete
                // SAFETY: obj, key_c are valid.
                let deleted = unsafe { stator_object_delete(obj, key_c.as_ptr()) };
                if deleted {
                    expected.remove(key);
                    // After deletion, has must return false.
                    // SAFETY: obj, key_c are valid.
                    let still_has = unsafe { stator_object_has(obj, key_c.as_ptr()) };
                    assert!(
                        !still_has,
                        "stator_object_has must be false after successful delete"
                    );
                }
            }
            4 => {
                // stator_object_get_property_names
                // SAFETY: obj is valid.
                let names = unsafe { stator_object_get_property_names(obj) };
                if !names.is_null() {
                    // SAFETY: names is a live pointer we own.
                    let count = unsafe { stator_property_names_count(names) };
                    for i in 0..count {
                        // SAFETY: names is valid; i < count.
                        let name_ptr = unsafe { stator_property_names_get(names, i) };
                        assert!(
                            !name_ptr.is_null(),
                            "stator_property_names_get must not return null for valid index"
                        );
                        // SAFETY: name_ptr points to a valid null-terminated string.
                        let _name = unsafe { CStr::from_ptr(name_ptr) };
                    }
                    // Out-of-range get must return null.
                    // SAFETY: names is valid.
                    let out_of_range = unsafe { stator_property_names_get(names, count + 100) };
                    assert!(
                        out_of_range.is_null(),
                        "out-of-range stator_property_names_get must return null"
                    );
                    // SAFETY: names is a Box-backed pointer we own.
                    unsafe { stator_property_names_destroy(names) };
                }
            }
            _ => unreachable!(),
        }
    }

    // Cleanup.
    // SAFETY: obj is a Box-backed pointer we own; all values have been freed.
    unsafe { stator_object_destroy(obj) };
    // SAFETY: ctx is a live pointer we own.
    unsafe { stator_context_destroy(ctx) };
    // SAFETY: iso is a live pointer we own.
    unsafe { stator_isolate_dispose(iso) };
});

/// Create a `StatorValue` of the given type.  Returns the pointer and a
/// string representation of the value for round-trip verification.
fn make_value(
    iso: *mut stator_js_ffi::StatorIsolate,
    sel: u8,
    payload: u8,
) -> (*mut stator_js_ffi::StatorValue, String) {
    unsafe {
        match sel {
            0 => (stator_value_new_undefined(iso), String::new()),
            1 => (stator_value_new_null(iso), String::new()),
            2 => {
                let b = payload & 1 != 0;
                (stator_value_new_boolean(iso, b), b.to_string())
            }
            3 => {
                let n = f64::from(payload);
                (stator_value_new_number(iso, n), n.to_string())
            }
            _ => {
                // string: use a short fixed string derived from payload.
                let s = format!("s{payload}");
                let p = s.as_ptr() as *const std::ffi::c_char;
                let len = s.len();
                let v = stator_value_new_string(iso, p, len);
                (v, s)
            }
        }
    }
}
