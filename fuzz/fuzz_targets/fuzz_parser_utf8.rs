#![no_main]

use libfuzzer_sys::fuzz_target;
use stator_core::parser::preparser::preparse;

fuzz_target!(|data: &[u8]| {
    // Only process valid UTF-8 inputs so the fuzzer exercises real JS text
    // patterns instead of raw binary noise.
    let Ok(source) = std::str::from_utf8(data) else {
        return;
    };

    // The pre-parser must not panic on any valid UTF-8 input.
    let _ = preparse(source);

    // Additionally exercise the scanner on the same UTF-8 source.
    let _ = stator_core::parser::scanner::Scanner::tokenize_all(source);
});
