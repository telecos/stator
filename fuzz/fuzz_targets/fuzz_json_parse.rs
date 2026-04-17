#![no_main]

use libfuzzer_sys::fuzz_target;
use stator_js::builtins::json::{json_parse, json_stringify, JsonSpace};

fuzz_target!(|data: &[u8]| {
    // Treat the raw bytes as a JSON text candidate (RFC 8259 §8.1 allows UTF-8).
    // Invalid UTF-8 sequences are replaced so the parser always receives a &str.
    let text = String::from_utf8_lossy(data);

    // json_parse must not panic on any input; it may return an error.
    let Ok(value) = json_parse(&text, None) else {
        return;
    };

    // If we parsed successfully, round-trip through stringify must also not panic.
    let _ = json_stringify(&value, None, Some(&JsonSpace::Count(0)), None);
});
