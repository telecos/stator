#![no_main]

use libfuzzer_sys::fuzz_target;
use stator_jse::builtins::string::{
    string_char_at, string_char_code_at, string_code_point_at, string_concat, string_ends_with,
    string_from_char_code, string_includes, string_index_of, string_last_index_of,
    string_match, string_match_all, string_replace, string_replace_all, string_repeat,
    string_slice, string_split, string_starts_with, string_substring, string_to_lower_case,
    string_to_upper_case, string_trim, string_trim_end, string_trim_start,
};

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let op = data[0];
    // Use the remaining bytes as UTF-8 text (replacing invalid sequences).
    let text = String::from_utf8_lossy(&data[1..]).into_owned();

    // Additional helper: second string derived from first half of text.
    let mid = text.len() / 2;
    let sub = &text[..mid];

    // Index derived from last byte (may be negative).
    let idx = i64::from(data[data.len() - 1]) - 64;

    match op % 22 {
        0 => {
            // charAt / charCodeAt / codePointAt — must not panic for any index
            let _ = string_char_at(&text, idx);
            let _ = string_char_code_at(&text, idx);
            let _ = string_code_point_at(&text, idx);
        }
        1 => {
            // String.fromCharCode
            let codes: Vec<u32> = data.iter().map(|&b| u32::from(b)).collect();
            let _ = string_from_char_code(&codes);
        }
        2 => {
            // concat
            let _ = string_concat(&text, &[sub, " suffix"]);
        }
        3 => {
            // slice / substring
            let end = idx.wrapping_add(8);
            let _ = string_slice(&text, idx, Some(end));
            let _ = string_substring(&text, idx, Some(end));
        }
        4 => {
            // indexOf / lastIndexOf
            let _ = string_index_of(&text, sub, None);
            let _ = string_last_index_of(&text, sub, None);
        }
        5 => {
            // includes / startsWith / endsWith
            let _ = string_includes(&text, sub, None);
            let _ = string_starts_with(&text, sub, None);
            let _ = string_ends_with(&text, sub, None);
        }
        6 => {
            // toUpperCase / toLowerCase
            let upper = string_to_upper_case(&text);
            let lower = string_to_lower_case(&text);
            // Idempotency: applying the same conversion twice is a no-op.
            assert_eq!(string_to_upper_case(&upper), upper);
            assert_eq!(string_to_lower_case(&lower), lower);
        }
        7 => {
            // trim / trimStart / trimEnd
            let trimmed = string_trim(&text);
            // A trimmed string trimmed again must be unchanged.
            assert_eq!(string_trim(&trimmed), trimmed);
            let _ = string_trim_start(&text);
            let _ = string_trim_end(&text);
        }
        8 => {
            // split — cap limit to avoid unbounded output
            let limit_val = (data[1] % 16) as u32 + 1;
            let parts = string_split(&text, Some(","), Some(limit_val));
            // Number of parts must not exceed the requested limit.
            assert!(parts.len() <= limit_val as usize);
        }
        9 => {
            // replace / replaceAll
            let _ = string_replace(&text, sub, "X");
            let _ = string_replace_all(&text, sub, "X");
        }
        10 => {
            // repeat — cap count to avoid out-of-memory
            let count = i64::from(data[1] % 8);
            let _ = string_repeat(&text, count);
        }
        11 => {
            // match — invalid patterns return None (no panic)
            let _ = string_match(&text, sub);
        }
        12 => {
            // match_all — invalid patterns return None (no panic)
            let _ = string_match_all(&text, sub);
        }
        13 => {
            // charCodeAt round-trip: code units should be ≤ 0xFFFF or NaN
            for i in 0..text.len() as i64 {
                let code = string_char_code_at(&text, i);
                assert!(code.is_nan() || (0.0..=65535.0).contains(&code));
            }
        }
        14 => {
            // charAt length invariant: result is "" or a single UTF-16 code unit string
            for i in -2i64..=(text.len() as i64 + 2) {
                let ch = string_char_at(&text, i);
                // Result is either empty or a single UTF-16 code unit (1 or 2 bytes).
                assert!(ch.encode_utf16().count() <= 1);
            }
        }
        15 => {
            // slice with both endpoints at the same index → empty
            let _ = string_slice(&text, idx, Some(idx));
        }
        16 => {
            // concat with empty string must return same content
            let result = string_concat(&text, &[]);
            assert_eq!(result, text);
        }
        17 => {
            // includes with empty search must always be true
            assert!(string_includes(&text, "", None));
        }
        18 => {
            // startsWith with empty search must always be true
            assert!(string_starts_with(&text, "", None));
        }
        19 => {
            // endsWith with empty search must always be true
            assert!(string_ends_with(&text, "", None));
        }
        20 => {
            // split with no separator must return single-element vec
            let parts = string_split(&text, None, None);
            assert_eq!(parts.len(), 1);
            assert_eq!(parts[0], text);
        }
        _ => {
            // repeat(0) must return empty string
            let result = string_repeat(&text, 0).unwrap_or_default();
            assert!(result.is_empty());
        }
    }
});
