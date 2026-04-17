#![no_main]

use libfuzzer_sys::fuzz_target;
use stator_js::parser::preparser::preparse;

fuzz_target!(|data: &[u8]| {
    // Convert raw bytes to a string — invalid UTF-8 is silently replaced so
    // the pre-parser always receives a valid &str.
    let source = String::from_utf8_lossy(data);

    // The pre-parser must not panic on any input; errors are acceptable.
    let _ = preparse(&source);
});
