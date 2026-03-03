#![no_main]

//! Stress-test realistic embedder patterns by interleaving FFI calls randomly.
//!
//! Each byte selects an operation from a rich menu covering the entire FFI
//! surface: isolates, contexts, values, objects, scripts, and handle scopes.
//! The goal is to catch crashes and memory errors that only manifest when
//! operations are interleaved in unexpected ways (e.g., script execution while
//! a handle scope is open, mixing value creation and GC, etc.).
//!
//! A small pool of live values / objects / scripts is maintained; each
//! operation either creates a new item (bounded pool size) or acts on a live
//! one.

use std::ffi::{CStr, CString};

use libfuzzer_sys::fuzz_target;
use stator_ffi::{
    StatorContext, StatorIsolate, StatorObject, StatorScript, StatorValue,
    stator_context_destroy, stator_context_enter, stator_context_exit, stator_context_new,
    stator_gc_collect,
    stator_handle_scope_close, stator_handle_scope_new,
    stator_isolate_dispose, stator_isolate_enter, stator_isolate_exit, stator_isolate_new,
    stator_live_object_count,
    stator_object_delete, stator_object_get, stator_object_get_property_names,
    stator_object_has, stator_object_new, stator_object_destroy, stator_object_set,
    stator_property_names_count, stator_property_names_destroy, stator_property_names_get,
    stator_script_compile, stator_script_free, stator_script_get_error, stator_script_run,
    stator_value_destroy, stator_value_new_boolean, stator_value_new_null,
    stator_value_new_number, stator_value_new_string, stator_value_new_undefined,
    stator_value_to_boolean, stator_value_to_int32, stator_value_to_number,
    stator_value_to_string, stator_value_type,
};

/// Maximum number of live values / objects / scripts at one time to keep
/// memory usage bounded during a single fuzz run.
const MAX_VALUES: usize = 8;
const MAX_OBJECTS: usize = 4;
const MAX_SCRIPTS: usize = 4;
const MAX_CONTEXTS: usize = 2;
/// Property names used for object operations.
const KEYS: [&str; 4] = ["a", "b", "c", "d"];
/// A small set of JS snippets used for script compilation.
const JS_SNIPPETS: [&str; 8] = [
    "1 + 2",
    "var x = 42; x",
    "function f() { return 1; } f()",
    "null",
    "undefined",
    "true",
    "'hello'",
    "1 / 0",
];

struct State {
    iso: *mut StatorIsolate,
    contexts: Vec<*mut StatorContext>,
    /// Values NOT owned by a handle scope (must be explicitly freed).
    values: Vec<*mut StatorValue>,
    objects: Vec<*mut StatorObject>,
    scripts: Vec<*mut StatorScript>,
    /// Currently open handle scope, or null.
    handle_scope: *mut stator_ffi::StatorHandleScope,
}

impl State {
    fn current_ctx(&self) -> *mut StatorContext {
        self.contexts.last().copied().unwrap_or(std::ptr::null_mut())
    }
}

fuzz_target!(|data: &[u8]| {
    // SAFETY: all FFI calls below satisfy their documented pointer validity
    // requirements.  Pointers are removed from the pools before being freed,
    // so no use-after-free is possible within the fuzzer itself.

    let iso = stator_isolate_new();
    assert!(!iso.is_null());

    // Always start with one context.
    let ctx = unsafe { stator_context_new(iso) };
    assert!(!ctx.is_null());

    let mut st = State {
        iso,
        contexts: vec![ctx],
        values: Vec::new(),
        objects: Vec::new(),
        scripts: Vec::new(),
        handle_scope: std::ptr::null_mut(),
    };

    for &byte in data {
        let op = byte & 0x1f; // 32 operations
        let aux = byte >> 5;  // 3-bit auxiliary index

        match op {
            // ── Isolate lifecycle ──────────────────────────────────────────
            0 => unsafe { stator_isolate_enter(st.iso) },
            1 => unsafe { stator_isolate_exit(st.iso) },
            2 => unsafe { stator_gc_collect(st.iso) },

            // ── Context lifecycle ──────────────────────────────────────────
            3 if st.contexts.len() < MAX_CONTEXTS => {
                let c = unsafe { stator_context_new(st.iso) };
                if !c.is_null() {
                    st.contexts.push(c);
                }
            }
            4 => {
                if let Some(&c) = st.contexts.last() {
                    unsafe { stator_context_enter(c) };
                }
            }
            5 => {
                if let Some(&c) = st.contexts.last() {
                    unsafe { stator_context_exit(c) };
                }
            }
            6 if st.contexts.len() > 1 => {
                // Destroy the most recently created context (keep at least one).
                let c = st.contexts.pop().unwrap();
                unsafe { stator_context_destroy(c) };
            }

            // ── Value creation ─────────────────────────────────────────────
            7 if st.values.len() < MAX_VALUES => {
                let v = unsafe { stator_value_new_undefined(st.iso) };
                if !v.is_null() { st.values.push(v); }
            }
            8 if st.values.len() < MAX_VALUES => {
                let v = unsafe { stator_value_new_null(st.iso) };
                if !v.is_null() { st.values.push(v); }
            }
            9 if st.values.len() < MAX_VALUES => {
                let v = unsafe { stator_value_new_boolean(st.iso, aux & 1 != 0) };
                if !v.is_null() { st.values.push(v); }
            }
            10 if st.values.len() < MAX_VALUES => {
                let v = unsafe { stator_value_new_number(st.iso, f64::from(aux) * std::f64::consts::PI) };
                if !v.is_null() { st.values.push(v); }
            }
            11 if st.values.len() < MAX_VALUES => {
                let s = format!("str{aux}");
                let v = unsafe {
                    stator_value_new_string(
                        st.iso,
                        s.as_ptr() as *const std::ffi::c_char,
                        s.len(),
                    )
                };
                if !v.is_null() { st.values.push(v); }
            }

            // ── Value operations ───────────────────────────────────────────
            12 if !st.values.is_empty() => {
                // Inspect and coerce a random live value.
                let idx = usize::from(aux) % st.values.len();
                let v = st.values[idx];
                let _type_str = unsafe { stator_value_type(v) };
                let _num = unsafe { stator_value_to_number(v) };
                let _bool = unsafe { stator_value_to_boolean(v) };
                let _i32 = unsafe { stator_value_to_int32(v) };
                // to_string allocates; destroy immediately (no scope open).
                if st.handle_scope.is_null() {
                    let s = unsafe { stator_value_to_string(st.iso, v) };
                    if !s.is_null() {
                        unsafe { stator_value_destroy(s) };
                    }
                }
            }
            13 if !st.values.is_empty() => {
                // Destroy a random live value.
                let idx = usize::from(aux) % st.values.len();
                let v = st.values.remove(idx);
                // Values created via stator_value_new_* are registered with
                // the active handle scope (if any) by the engine.  When a scope
                // is open we must not call stator_value_destroy — the scope owns
                // the value and will free it when closed.  Op 24 clears
                // st.values before closing the scope, so no value is freed twice.
                if st.handle_scope.is_null() {
                    unsafe { stator_value_destroy(v) };
                }
                // If a scope is open, the value is scope-owned; don't free it.
            }

            // ── Object operations ──────────────────────────────────────────
            14 if st.objects.len() < MAX_OBJECTS => {
                let o = unsafe { stator_object_new(st.iso) };
                if !o.is_null() { st.objects.push(o); }
            }
            15 if !st.objects.is_empty() && !st.values.is_empty() => {
                // Set a property on a random object using a random live value.
                let oi = usize::from(aux) % st.objects.len();
                let vi = usize::from(aux) % st.values.len();
                let key = KEYS[usize::from(aux) % KEYS.len()];
                let key_c = CString::new(key).unwrap();
                unsafe { stator_object_set(st.objects[oi], key_c.as_ptr(), st.values[vi]) };
            }
            16 if !st.objects.is_empty() => {
                // Get a property.
                // stator_object_get does not register the returned value with
                // the active handle scope; the caller always owns it.
                let oi = usize::from(aux) % st.objects.len();
                let key = KEYS[usize::from(aux) % KEYS.len()];
                let key_c = CString::new(key).unwrap();
                let got = unsafe { stator_object_get(st.objects[oi], key_c.as_ptr()) };
                if !got.is_null() {
                    unsafe { stator_value_destroy(got) };
                }
            }
            17 if !st.objects.is_empty() => {
                // has + delete.
                let oi = usize::from(aux) % st.objects.len();
                let key = KEYS[usize::from(aux) % KEYS.len()];
                let key_c = CString::new(key).unwrap();
                let _has = unsafe { stator_object_has(st.objects[oi], key_c.as_ptr()) };
                unsafe { stator_object_delete(st.objects[oi], key_c.as_ptr()) };
            }
            18 if !st.objects.is_empty() => {
                // Property names snapshot.
                let oi = usize::from(aux) % st.objects.len();
                let names = unsafe { stator_object_get_property_names(st.objects[oi]) };
                if !names.is_null() {
                    let count = unsafe { stator_property_names_count(names) };
                    for i in 0..count {
                        let p = unsafe { stator_property_names_get(names, i) };
                        if !p.is_null() {
                            let _ = unsafe { CStr::from_ptr(p) };
                        }
                    }
                    unsafe { stator_property_names_destroy(names) };
                }
            }
            19 if !st.objects.is_empty() => {
                // Destroy a random object.
                let oi = usize::from(aux) % st.objects.len();
                let o = st.objects.remove(oi);
                unsafe { stator_object_destroy(o) };
            }

            // ── Script operations ──────────────────────────────────────────
            20 if st.scripts.len() < MAX_SCRIPTS => {
                let snippet = JS_SNIPPETS[usize::from(aux) % JS_SNIPPETS.len()];
                let s = unsafe {
                    stator_script_compile(
                        st.current_ctx(),
                        snippet.as_ptr() as *const std::ffi::c_char,
                        snippet.len(),
                    )
                };
                if !s.is_null() {
                    st.scripts.push(s);
                }
            }
            21 if !st.scripts.is_empty() => {
                // Run a random live script.
                let si = usize::from(aux) % st.scripts.len();
                let script = st.scripts[si];
                let err_ptr = unsafe { stator_script_get_error(script) };
                if err_ptr.is_null() {
                    // Only run scripts that compiled successfully.
                    let result = unsafe { stator_script_run(script, st.current_ctx()) };
                    if !result.is_null() {
                        unsafe { stator_value_destroy(result) };
                    }
                }
            }
            22 if !st.scripts.is_empty() => {
                // Free a random script.
                let si = usize::from(aux) % st.scripts.len();
                let s = st.scripts.remove(si);
                unsafe { stator_script_free(s) };
            }

            // ── Handle scope ───────────────────────────────────────────────
            23 if st.handle_scope.is_null() => {
                // Open a handle scope (at most one at a time for simplicity).
                let scope = unsafe { stator_handle_scope_new(st.iso) };
                st.handle_scope = scope;
            }
            24 if !st.handle_scope.is_null() => {
                // Close the handle scope; all scope-owned values become invalid.
                // Any values that were created while this scope was active
                // are owned by it.  We drain them from st.values to prevent
                // any future operations from touching freed pointers.
                st.values.clear();
                unsafe { stator_handle_scope_close(st.handle_scope) };
                st.handle_scope = std::ptr::null_mut();
            }

            // ── Live-object counter ────────────────────────────────────────
            25 => {
                // Query the live-object counter; must not crash.
                let _count = unsafe { stator_live_object_count(st.iso) };
            }

            _ => {} // ops 26-31 and unmatched guards: no-op
        }
    }

    // ── Teardown: free everything in safe order ────────────────────────────

    // Close any open handle scope (frees scope-owned values).
    if !st.handle_scope.is_null() {
        st.values.clear(); // scope-owned; must not double-free
        unsafe { stator_handle_scope_close(st.handle_scope) };
    }

    // Free remaining values.
    for v in st.values.drain(..) {
        unsafe { stator_value_destroy(v) };
    }

    // Free objects.
    for o in st.objects.drain(..) {
        unsafe { stator_object_destroy(o) };
    }

    // Free scripts.
    for s in st.scripts.drain(..) {
        unsafe { stator_script_free(s) };
    }

    // Destroy contexts (the mandatory initial one last).
    for c in st.contexts.drain(..) {
        unsafe { stator_context_destroy(c) };
    }

    // Dispose the isolate.
    unsafe { stator_isolate_dispose(st.iso) };
});
