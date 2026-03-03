#![no_main]

use libfuzzer_sys::fuzz_target;
use stator_core::builtins::global::global_eval;

fuzz_target!(|data: &[u8]| {
    // Accept arbitrary bytes as UTF-8 source (replace invalid sequences).
    let source = String::from_utf8_lossy(data);

    // eval() must either return a valid result or a graceful error — never panic.
    let _ = global_eval(&source);
});
