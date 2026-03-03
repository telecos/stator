#![no_main]

use libfuzzer_sys::fuzz_target;
use stator_core::builtins::string::{string_match, string_match_all};

fuzz_target!(|data: &[u8]| {
    // Need at least 2 bytes: one length byte for the pattern, then the rest.
    if data.is_empty() {
        return;
    }

    // First byte controls how many bytes form the pattern (1..=64).
    let pat_len = (usize::from(data[0]) % 64) + 1;
    let rest = &data[1..];

    if rest.len() < pat_len {
        return;
    }

    let pattern = String::from_utf8_lossy(&rest[..pat_len]);
    let input = String::from_utf8_lossy(&rest[pat_len..]);

    // string_match and string_match_all must not panic on any pattern/input pair;
    // invalid patterns return None, which is the expected graceful failure path.
    let _ = string_match(&input, &pattern);
    let _ = string_match_all(&input, &pattern);
});
