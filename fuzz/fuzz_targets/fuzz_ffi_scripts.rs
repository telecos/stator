#![no_main]

//! Fuzz `stator_script_compile` + `stator_script_run` with arbitrary input.
//!
//! Any byte sequence is interpreted as the source of a JavaScript program.
//! The fuzzer verifies:
//!   - `stator_script_compile` never panics (returns a valid pointer).
//!   - `stator_script_get_error` returns either null (success) or a valid C
//!     string on parse / compile error.
//!   - `stator_script_bytecode_count` returns a non-zero count on success.
//!   - `stator_script_run` never panics; a null return is allowed on runtime
//!     error.
//!   - All returned `StatorValue` pointers are properly freed.

use std::ffi::CStr;

use libfuzzer_sys::fuzz_target;
use stator_js_ffi::{
    stator_context_destroy, stator_context_new,
    stator_isolate_dispose, stator_isolate_new,
    stator_script_bytecode_count, stator_script_compile, stator_script_free,
    stator_script_get_error, stator_script_run,
    stator_value_destroy,
};

fuzz_target!(|data: &[u8]| {
    // Treat fuzz data as the JS source; replace invalid UTF-8 sequences.
    let source = String::from_utf8_lossy(data);

    // SAFETY: all FFI calls satisfy their documented pointer validity requirements.
    let iso = stator_isolate_new();
    assert!(!iso.is_null(), "stator_isolate_new must not return null");

    let ctx = unsafe { stator_context_new(iso) };
    assert!(!ctx.is_null(), "stator_context_new must not return null");

    // --- Compile ---
    let src_bytes = source.as_bytes();
    // SAFETY: `src_bytes` is a valid UTF-8 byte slice backed by `source`.
    let script = unsafe {
        stator_script_compile(
            ctx,
            src_bytes.as_ptr() as *const std::ffi::c_char,
            src_bytes.len(),
        )
    };
    assert!(!script.is_null(), "stator_script_compile must not return null");

    // SAFETY: `script` is a valid pointer returned by `stator_script_compile`.
    let err_ptr = unsafe { stator_script_get_error(script) };
    let compiled_ok = err_ptr.is_null();

    if !compiled_ok {
        // SAFETY: `err_ptr` is a static-lifetime string owned by `script`.
        let err_cstr = unsafe { CStr::from_ptr(err_ptr) };
        // Just verify it's valid UTF-8 / doesn't crash.
        let _ = err_cstr.to_string_lossy();
    } else {
        // Bytecode count must be > 0 for a successfully compiled script.
        // SAFETY: `script` is valid.
        let count = unsafe { stator_script_bytecode_count(script) };
        assert!(count > 0, "a successfully compiled script must have at least one instruction");

        // --- Run ---
        // SAFETY: `script` and `ctx` are both valid live pointers.
        let result = unsafe { stator_script_run(script, ctx) };
        // `result` may be null on runtime error; that is allowed.
        if !result.is_null() {
            // SAFETY: `result` is a Box-backed pointer we own.
            unsafe { stator_value_destroy(result) };
        }
    }

    // --- Cleanup ---
    // SAFETY: `script` is a live pointer returned by `stator_script_compile`.
    unsafe { stator_script_free(script) };
    // SAFETY: `ctx` is a live pointer we own.
    unsafe { stator_context_destroy(ctx) };
    // SAFETY: `iso` is a live pointer we own.
    unsafe { stator_isolate_dispose(iso) };
});
