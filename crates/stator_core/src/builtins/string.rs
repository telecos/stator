//! ECMAScript §22.1 `String` built-in static methods and prototype equivalents.
//!
//! Every function in this module is a direct Rust equivalent of either a static
//! property of the JavaScript `String` constructor or a method on
//! `String.prototype`.  They operate on plain `&str` / `String` values and
//! have no side-effects beyond the values passed in.
//!
//! # Index semantics
//!
//! ECMAScript strings are sequences of **UTF-16 code units**.  All positional
//! parameters (e.g. `pos` in `charAt`, `start`/`end` in `slice`) are therefore
//! interpreted as UTF-16 code unit indices, matching the behaviour a JavaScript
//! developer would expect.  Internally the string is re-encoded to a `Vec<u16>`
//! only when an index-based operation is requested; all other operations work
//! directly on Rust's UTF-8 `&str`.
//!
//! # Naming convention
//!
//! Each function is prefixed `string_` to avoid ambiguity with similarly-named
//! standard-library items (e.g. `string_slice` vs `str::split`).
//!
//! # References
//!
//! * ECMAScript 2025 Language Specification §22.1 — *The String Constructor*

use crate::error::{StatorError, StatorResult};

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Encodes a UTF-8 string slice into a `Vec` of UTF-16 code units.
fn encode_utf16(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

/// Decodes a slice of UTF-16 code units back to a `String`, replacing any
/// unpaired surrogates with the Unicode replacement character (`U+FFFD`).
fn decode_utf16(units: &[u16]) -> String {
    String::from_utf16_lossy(units)
}

/// Clamps a potentially-negative index to the valid range `[0, len]` using
/// ECMAScript relative-index semantics (negative means offset from the end).
fn clamp_index(index: i64, len: usize) -> usize {
    let len = len as i64;
    if index < 0 {
        (len + index).max(0) as usize
    } else {
        index.min(len) as usize
    }
}

// UTF-16 surrogate range boundaries (Unicode §3.8).
/// First code unit of a high (leading) surrogate pair.
const UTF16_HIGH_SURROGATE_START: u16 = 0xD800;
/// Last code unit of a high (leading) surrogate pair.
const UTF16_HIGH_SURROGATE_END: u16 = 0xDBFF;
/// First code unit of a low (trailing) surrogate pair.
const UTF16_LOW_SURROGATE_START: u16 = 0xDC00;
/// Last code unit of a low (trailing) surrogate pair.
const UTF16_LOW_SURROGATE_END: u16 = 0xDFFF;

// ── String.fromCharCode ───────────────────────────────────────────────────────

/// ECMAScript §22.1.2.1 `String.fromCharCode(...codes)`.
///
/// Creates a string from one or more UTF-16 code unit values.  Each element of
/// `codes` is masked to the lower 16 bits, matching the ECMAScript `ToUint16`
/// conversion applied by the spec.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_from_char_code;
///
/// assert_eq!(string_from_char_code(&[72, 101, 108, 108, 111]), "Hello");
/// // Values outside 0..=0xFFFF are masked to their lower 16 bits.
/// assert_eq!(string_from_char_code(&[0x10041]), "A"); // 0x10041 & 0xFFFF = 0x41 = 'A'
/// ```
pub fn string_from_char_code(codes: &[u32]) -> String {
    let units: Vec<u16> = codes.iter().map(|&c| (c & 0xFFFF) as u16).collect();
    decode_utf16(&units)
}

// ── String.fromCodePoint ──────────────────────────────────────────────────────

/// ECMAScript §22.1.2.2 `String.fromCodePoint(...codePoints)`.
///
/// Creates a string from one or more Unicode code point values.
///
/// # Errors
///
/// Returns [`StatorError::RangeError`] if any value is not a valid Unicode
/// scalar value (i.e. outside `0..=0x10FFFF` or in the surrogate range
/// `0xD800..=0xDFFF`).
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_from_code_point;
///
/// assert_eq!(string_from_code_point(&[0x1F600]).unwrap(), "😀");
/// assert!(string_from_code_point(&[0x110000]).is_err());
/// ```
pub fn string_from_code_point(code_points: &[u32]) -> StatorResult<String> {
    let mut result = String::new();
    for &cp in code_points {
        let ch = char::from_u32(cp)
            .ok_or_else(|| StatorError::RangeError(format!("Invalid code point {cp}")))?;
        result.push(ch);
    }
    Ok(result)
}

// ── charAt ────────────────────────────────────────────────────────────────────

/// ECMAScript §22.1.3.1 `String.prototype.charAt(pos)`.
///
/// Returns a single-character string containing the UTF-16 code unit at the
/// given index, or an empty string if `pos` is out of bounds.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_char_at;
///
/// assert_eq!(string_char_at("Hello", 1), "e");
/// assert_eq!(string_char_at("Hello", 10), "");
/// assert_eq!(string_char_at("Hello", -1), "");
/// ```
pub fn string_char_at(s: &str, pos: i64) -> String {
    if pos < 0 {
        return String::new();
    }
    let units = encode_utf16(s);
    match units.get(pos as usize) {
        Some(&u) => decode_utf16(&[u]),
        None => String::new(),
    }
}

// ── charCodeAt ────────────────────────────────────────────────────────────────

/// ECMAScript §22.1.3.2 `String.prototype.charCodeAt(pos)`.
///
/// Returns the numeric value of the UTF-16 code unit at `pos`, or `NaN` (as
/// `f64::NAN`) if `pos` is out of range.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_char_code_at;
///
/// assert_eq!(string_char_code_at("ABC", 0), 65.0);
/// assert!(string_char_code_at("ABC", 5).is_nan());
/// ```
pub fn string_char_code_at(s: &str, pos: i64) -> f64 {
    if pos < 0 {
        return f64::NAN;
    }
    let units = encode_utf16(s);
    units
        .get(pos as usize)
        .copied()
        .map(f64::from)
        .unwrap_or(f64::NAN)
}

// ── codePointAt ───────────────────────────────────────────────────────────────

/// ECMAScript §22.1.3.3 `String.prototype.codePointAt(pos)`.
///
/// Returns the Unicode code point value of the character starting at the given
/// UTF-16 code unit index, or `None` if `pos` is out of bounds.  Correctly
/// handles surrogate pairs: if the code unit at `pos` is a high surrogate and
/// the next unit is a low surrogate, the full supplementary code point is
/// returned.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_code_point_at;
///
/// assert_eq!(string_code_point_at("😀", 0), Some(0x1F600));
/// assert_eq!(string_code_point_at("A", 0), Some(65));
/// assert_eq!(string_code_point_at("A", 5), None);
/// ```
pub fn string_code_point_at(s: &str, pos: i64) -> Option<u32> {
    if pos < 0 {
        return None;
    }
    let units = encode_utf16(s);
    let idx = pos as usize;
    let high = *units.get(idx)?;
    // Surrogate pair detection: per Unicode §3.9, a high surrogate in
    // [U+D800, U+DBFF] immediately followed by a low surrogate in
    // [U+DC00, U+DFFF] encodes a supplementary code point.
    // Decoding: cp = 0x10000 + (H - 0xD800) * 0x400 + (L - 0xDC00)
    if (UTF16_HIGH_SURROGATE_START..=UTF16_HIGH_SURROGATE_END).contains(&high)
        && let Some(&low) = units.get(idx + 1)
        && (UTF16_LOW_SURROGATE_START..=UTF16_LOW_SURROGATE_END).contains(&low)
    {
        let cp = 0x10000u32
            + ((high as u32 - UTF16_HIGH_SURROGATE_START as u32) << 10)
            + (low as u32 - UTF16_LOW_SURROGATE_START as u32);
        return Some(cp);
    }
    Some(u32::from(high))
}

// ── concat ────────────────────────────────────────────────────────────────────

/// ECMAScript §22.1.3.4 `String.prototype.concat(...strings)`.
///
/// Concatenates the receiver `s` with each string in `others` and returns the
/// resulting string.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_concat;
///
/// assert_eq!(string_concat("Hello", &[", ", "world", "!"]), "Hello, world!");
/// assert_eq!(string_concat("a", &[]), "a");
/// ```
pub fn string_concat(s: &str, others: &[&str]) -> String {
    let mut result = s.to_string();
    for other in others {
        result.push_str(other);
    }
    result
}

// ── slice ─────────────────────────────────────────────────────────────────────

/// ECMAScript §22.1.3.22 `String.prototype.slice(start, end?)`.
///
/// Returns the substring between UTF-16 code unit positions `start` and `end`
/// (exclusive).  Negative indices are resolved relative to the end of the
/// string.  If `start >= end` after clamping, returns an empty string.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_slice;
///
/// assert_eq!(string_slice("Hello", 1, Some(4)), "ell");
/// assert_eq!(string_slice("Hello", -3, None), "llo");
/// assert_eq!(string_slice("Hello", 2, Some(-1)), "ll");
/// ```
pub fn string_slice(s: &str, start: i64, end: Option<i64>) -> String {
    let units = encode_utf16(s);
    let len = units.len();
    let from = clamp_index(start, len);
    let to = match end {
        Some(e) => clamp_index(e, len),
        None => len,
    };
    if from >= to {
        return String::new();
    }
    decode_utf16(&units[from..to])
}

// ── substring ─────────────────────────────────────────────────────────────────

/// ECMAScript §22.1.3.27 `String.prototype.substring(start, end?)`.
///
/// Like [`string_slice`] but differs in two ways:
/// - Negative indices are clamped to `0` rather than being relative to the end.
/// - If `start > end`, the two arguments are swapped.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_substring;
///
/// assert_eq!(string_substring("Hello", 1, Some(3)), "el");
/// assert_eq!(string_substring("Hello", 3, Some(1)), "el"); // swapped
/// assert_eq!(string_substring("Hello", -99, Some(3)), "Hel"); // negative → 0
/// ```
pub fn string_substring(s: &str, start: i64, end: Option<i64>) -> String {
    let units = encode_utf16(s);
    let len = units.len() as i64;
    let from = start.max(0).min(len) as usize;
    let to = match end {
        Some(e) => e.max(0).min(len) as usize,
        None => len as usize,
    };
    let (from, to) = if from <= to { (from, to) } else { (to, from) };
    decode_utf16(&units[from..to])
}

// ── indexOf ───────────────────────────────────────────────────────────────────

/// ECMAScript §22.1.3.8 `String.prototype.indexOf(searchString, position?)`.
///
/// Searches for the first occurrence of `search` at or after UTF-16 code unit
/// position `from_index`.  Returns the index of the match, or `-1` if not
/// found.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_index_of;
///
/// assert_eq!(string_index_of("hello world", "world", None), 6);
/// assert_eq!(string_index_of("hello world", "xyz", None), -1);
/// assert_eq!(string_index_of("aaa", "a", Some(1)), 1);
/// ```
pub fn string_index_of(s: &str, search: &str, from_index: Option<i64>) -> i64 {
    let units = encode_utf16(s);
    let search_units = encode_utf16(search);
    let len = units.len();
    let from = from_index.unwrap_or(0).max(0).min(len as i64) as usize;

    if search_units.is_empty() {
        return from as i64;
    }
    if search_units.len() > len {
        return -1;
    }
    let end = len - search_units.len() + 1;
    for i in from..end {
        if units[i..i + search_units.len()] == search_units[..] {
            return i as i64;
        }
    }
    -1
}

// ── lastIndexOf ───────────────────────────────────────────────────────────────

/// ECMAScript §22.1.3.9 `String.prototype.lastIndexOf(searchString, position?)`.
///
/// Searches backwards for the last occurrence of `search` at or before UTF-16
/// code unit position `from_index`.  Returns the index of the match, or `-1`
/// if not found.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_last_index_of;
///
/// assert_eq!(string_last_index_of("hello world hello", "hello", None), 12);
/// assert_eq!(string_last_index_of("hello", "xyz", None), -1);
/// ```
pub fn string_last_index_of(s: &str, search: &str, from_index: Option<i64>) -> i64 {
    let units = encode_utf16(s);
    let search_units = encode_utf16(search);
    let len = units.len();

    if search_units.is_empty() {
        let max = match from_index {
            Some(f) => f.max(0).min(len as i64) as usize,
            None => len,
        };
        return max.min(len) as i64;
    }
    if search_units.len() > len {
        return -1;
    }

    let max_start = match from_index {
        Some(f) => f.max(0).min(len as i64) as usize,
        None => len,
    };
    let end = (max_start + 1).min(len - search_units.len() + 1);
    for i in (0..end).rev() {
        if units[i..i + search_units.len()] == search_units[..] {
            return i as i64;
        }
    }
    -1
}

// ── includes ──────────────────────────────────────────────────────────────────

/// ECMAScript §22.1.3.7 `String.prototype.includes(searchString, position?)`.
///
/// Returns `true` if `search` appears anywhere in `s` at or after `from_index`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_includes;
///
/// assert!(string_includes("hello world", "world", None));
/// assert!(!string_includes("hello world", "xyz", None));
/// assert!(string_includes("hello", "ell", Some(0)));
/// assert!(!string_includes("hello", "hel", Some(1)));
/// ```
pub fn string_includes(s: &str, search: &str, from_index: Option<i64>) -> bool {
    string_index_of(s, search, from_index) != -1
}

// ── startsWith ────────────────────────────────────────────────────────────────

/// ECMAScript §22.1.3.23 `String.prototype.startsWith(searchString, position?)`.
///
/// Returns `true` if `s` starts with `search` at UTF-16 code unit position
/// `position` (defaults to `0`).
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_starts_with;
///
/// assert!(string_starts_with("Hello", "Hel", None));
/// assert!(!string_starts_with("Hello", "ello", None));
/// assert!(string_starts_with("Hello", "ello", Some(1)));
/// ```
pub fn string_starts_with(s: &str, search: &str, position: Option<i64>) -> bool {
    let units = encode_utf16(s);
    let search_units = encode_utf16(search);
    let len = units.len();
    let pos = position.unwrap_or(0).max(0).min(len as i64) as usize;
    if pos + search_units.len() > len {
        return false;
    }
    units[pos..pos + search_units.len()] == search_units[..]
}

// ── endsWith ──────────────────────────────────────────────────────────────────

/// ECMAScript §22.1.3.6 `String.prototype.endsWith(searchString, endPosition?)`.
///
/// Returns `true` if `s` ends with `search` before UTF-16 code unit position
/// `end_position` (defaults to the length of the string).
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_ends_with;
///
/// assert!(string_ends_with("Hello", "llo", None));
/// assert!(!string_ends_with("Hello", "Hel", None));
/// assert!(string_ends_with("Hello", "Hel", Some(3)));
/// ```
pub fn string_ends_with(s: &str, search: &str, end_position: Option<i64>) -> bool {
    let units = encode_utf16(s);
    let search_units = encode_utf16(search);
    let end = match end_position {
        Some(e) => e.max(0).min(units.len() as i64) as usize,
        None => units.len(),
    };
    if search_units.len() > end {
        return false;
    }
    let start = end - search_units.len();
    units[start..end] == search_units[..]
}

// ── toUpperCase / toLowerCase ─────────────────────────────────────────────────

/// ECMAScript §22.1.3.28 `String.prototype.toUpperCase()`.
///
/// Returns a new string with all characters converted to their upper-case
/// equivalents using Unicode simple case folding (locale-independent).
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_to_upper_case;
///
/// assert_eq!(string_to_upper_case("hello"), "HELLO");
/// assert_eq!(string_to_upper_case("café"), "CAFÉ");
/// ```
pub fn string_to_upper_case(s: &str) -> String {
    s.to_uppercase()
}

/// ECMAScript §22.1.3.26 `String.prototype.toLowerCase()`.
///
/// Returns a new string with all characters converted to their lower-case
/// equivalents using Unicode simple case folding (locale-independent).
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_to_lower_case;
///
/// assert_eq!(string_to_lower_case("HELLO"), "hello");
/// assert_eq!(string_to_lower_case("CAFÉ"), "café");
/// ```
pub fn string_to_lower_case(s: &str) -> String {
    s.to_lowercase()
}

// ── trim ──────────────────────────────────────────────────────────────────────

/// ECMAScript §22.1.3.31 `String.prototype.trim()`.
///
/// Returns a new string with leading and trailing ASCII white-space and
/// Unicode line terminator / white-space characters stripped.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_trim;
///
/// assert_eq!(string_trim("  hello  "), "hello");
/// assert_eq!(string_trim("\t\nhello\r\n"), "hello");
/// ```
pub fn string_trim(s: &str) -> String {
    s.trim().to_string()
}

/// ECMAScript §22.1.3.32 `String.prototype.trimStart()`.
///
/// Returns a new string with leading white-space stripped.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_trim_start;
///
/// assert_eq!(string_trim_start("  hello  "), "hello  ");
/// ```
pub fn string_trim_start(s: &str) -> String {
    s.trim_start().to_string()
}

/// ECMAScript §22.1.3.33 `String.prototype.trimEnd()`.
///
/// Returns a new string with trailing white-space stripped.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_trim_end;
///
/// assert_eq!(string_trim_end("  hello  "), "  hello");
/// ```
pub fn string_trim_end(s: &str) -> String {
    s.trim_end().to_string()
}

// ── split ─────────────────────────────────────────────────────────────────────

/// ECMAScript §22.1.3.24 `String.prototype.split(separator?, limit?)`.
///
/// Splits `s` into substrings using `separator` as the delimiter and returns
/// up to `limit` substrings.
///
/// - `separator = None` — returns `[s]` (the entire string as a single element).
/// - `separator = Some("")` — splits between every adjacent pair of UTF-16 code
///   units (each code unit becomes its own element).
/// - `limit = None` — no limit (all substrings are returned).
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_split;
///
/// assert_eq!(string_split("a,b,c", Some(","), None), vec!["a", "b", "c"]);
/// assert_eq!(string_split("abc", Some(""), None), vec!["a", "b", "c"]);
/// assert_eq!(string_split("abc", None, None), vec!["abc"]);
/// assert_eq!(string_split("a,b,c", Some(","), Some(2)), vec!["a", "b"]);
/// ```
pub fn string_split(s: &str, separator: Option<&str>, limit: Option<u32>) -> Vec<String> {
    let lim = limit.unwrap_or(u32::MAX) as usize;
    if lim == 0 {
        return Vec::new();
    }
    let Some(sep) = separator else {
        return vec![s.to_string()];
    };

    if sep.is_empty() {
        // Split by each UTF-16 code unit.
        return encode_utf16(s)
            .into_iter()
            .take(lim)
            .map(|u| decode_utf16(&[u]))
            .collect();
    }

    s.split(sep).take(lim).map(str::to_string).collect()
}

// ── replace / replaceAll ──────────────────────────────────────────────────────

/// ECMAScript §22.1.3.17 `String.prototype.replace(searchValue, replaceValue)`.
///
/// Replaces the **first** occurrence of `search` with `replacement`.  If
/// `search` is not found the original string is returned unchanged.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_replace;
///
/// assert_eq!(string_replace("aabbcc", "bb", "XX"), "aaXXcc");
/// assert_eq!(string_replace("hello", "x", "y"), "hello");
/// assert_eq!(string_replace("aaa", "a", "b"), "baa");
/// ```
pub fn string_replace(s: &str, search: &str, replacement: &str) -> String {
    match s.find(search) {
        Some(pos) => {
            let mut result = s[..pos].to_string();
            result.push_str(replacement);
            result.push_str(&s[pos + search.len()..]);
            result
        }
        None => s.to_string(),
    }
}

/// ECMAScript §22.1.3.18 `String.prototype.replaceAll(searchValue, replaceValue)`.
///
/// Replaces **every** occurrence of `search` with `replacement`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_replace_all;
///
/// assert_eq!(string_replace_all("aabbaa", "aa", "X"), "XbbX");
/// assert_eq!(string_replace_all("hello", "x", "y"), "hello");
/// ```
pub fn string_replace_all(s: &str, search: &str, replacement: &str) -> String {
    if search.is_empty() {
        // ECMAScript: if searchString is "", insert replacement between every
        // character and at both ends (between each UTF-16 code unit).
        let units = encode_utf16(s);
        let mut result = replacement.to_string();
        for u in &units {
            result.push_str(&decode_utf16(&[*u]));
            result.push_str(replacement);
        }
        return result;
    }
    s.replace(search, replacement)
}

// ── match ─────────────────────────────────────────────────────────────────────

/// ECMAScript §22.1.3.12 `String.prototype.match(regexp)` — non-global variant.
///
/// Applies the regular expression `pattern` to `s` and returns all captured
/// substrings for the **first** match (index 0 is the whole match, subsequent
/// entries are capture groups).  Returns `None` if there is no match or if the
/// pattern fails to compile.
///
/// Uses the [`regress`] crate for ECMAScript-compatible regular expressions.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_match;
///
/// let m = string_match("hello world", r"(\w+)\s(\w+)").unwrap();
/// assert_eq!(m[0], "hello world");
/// assert_eq!(m[1], "hello");
/// assert_eq!(m[2], "world");
///
/// assert!(string_match("hello", r"\d+").is_none());
/// ```
pub fn string_match(s: &str, pattern: &str) -> Option<Vec<String>> {
    let re = regress::Regex::new(pattern).ok()?;
    let m = re.find(s)?;
    let mut groups = vec![s[m.range()].to_string()];
    for cap in &m.captures {
        if let Some(range) = cap {
            groups.push(s[range.clone()].to_string());
        } else {
            groups.push(String::new());
        }
    }
    Some(groups)
}

/// ECMAScript §22.1.3.12 `String.prototype.match(regexp)` — global (`g`) flag variant.
///
/// Returns all non-overlapping whole-match strings for `pattern` applied to
/// `s`.  Returns `None` if the pattern fails to compile or there are no
/// matches.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_match_all;
///
/// let matches = string_match_all("test 1 and 2 and 3", r"\d+").unwrap();
/// assert_eq!(matches, vec!["1", "2", "3"]);
/// ```
pub fn string_match_all(s: &str, pattern: &str) -> Option<Vec<String>> {
    let re = regress::Regex::new(pattern).ok()?;
    let matches: Vec<String> = re.find_iter(s).map(|m| s[m.range()].to_string()).collect();
    if matches.is_empty() {
        None
    } else {
        Some(matches)
    }
}

// ── repeat ────────────────────────────────────────────────────────────────────

/// ECMAScript §22.1.3.16 `String.prototype.repeat(count)`.
///
/// Returns `s` repeated `count` times.
///
/// # Errors
///
/// Returns [`StatorError::RangeError`] if `count` is negative.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_repeat;
///
/// assert_eq!(string_repeat("ab", 3).unwrap(), "ababab");
/// assert_eq!(string_repeat("x", 0).unwrap(), "");
/// assert!(string_repeat("a", -1).is_err());
/// ```
pub fn string_repeat(s: &str, count: i64) -> StatorResult<String> {
    if count < 0 {
        return Err(StatorError::RangeError(
            "Invalid count value: must be non-negative".to_string(),
        ));
    }
    let n = count as usize;
    // Guard against allocation failure: cap result size at ~256 MiB
    // (similar to V8's string length limit).
    const MAX_STRING_LEN: usize = 1 << 28;
    let total = n.saturating_mul(s.len());
    if total > MAX_STRING_LEN {
        return Err(StatorError::RangeError(
            "Invalid count value: result string exceeds maximum length".to_string(),
        ));
    }
    Ok(s.repeat(n))
}

// ── padStart / padEnd ─────────────────────────────────────────────────────────

/// ECMAScript §22.1.3.14 `String.prototype.padStart(targetLength, padString?)`.
///
/// Pads the **beginning** of `s` with `pad_string` (default `" "`) until the
/// string reaches `target_length` UTF-16 code units.  If `s` is already at
/// least `target_length` units long, it is returned unchanged.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_pad_start;
///
/// assert_eq!(string_pad_start("5", 3, None), "  5");
/// assert_eq!(string_pad_start("5", 3, Some("0")), "005");
/// assert_eq!(string_pad_start("hello", 3, None), "hello"); // already long enough
/// ```
pub fn string_pad_start(s: &str, target_length: usize, pad_string: Option<&str>) -> String {
    let units = encode_utf16(s);
    let len = units.len();
    if len >= target_length {
        return s.to_string();
    }
    let pad = pad_string.unwrap_or(" ");
    let pad_units = encode_utf16(pad);
    if pad_units.is_empty() {
        return s.to_string();
    }
    let pad_count = target_length - len;
    let mut prefix: Vec<u16> = Vec::with_capacity(pad_count);
    let mut filled = 0;
    while filled < pad_count {
        let remaining = pad_count - filled;
        let take = remaining.min(pad_units.len());
        prefix.extend_from_slice(&pad_units[..take]);
        filled += take;
    }
    let mut result = decode_utf16(&prefix);
    result.push_str(s);
    result
}

/// ECMAScript §22.1.3.13 `String.prototype.padEnd(targetLength, padString?)`.
///
/// Pads the **end** of `s` with `pad_string` (default `" "`) until the string
/// reaches `target_length` UTF-16 code units.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_pad_end;
///
/// assert_eq!(string_pad_end("5", 3, None), "5  ");
/// assert_eq!(string_pad_end("5", 3, Some("0")), "500");
/// assert_eq!(string_pad_end("hello", 3, None), "hello");
/// ```
pub fn string_pad_end(s: &str, target_length: usize, pad_string: Option<&str>) -> String {
    let units = encode_utf16(s);
    let len = units.len();
    if len >= target_length {
        return s.to_string();
    }
    let pad = pad_string.unwrap_or(" ");
    let pad_units = encode_utf16(pad);
    if pad_units.is_empty() {
        return s.to_string();
    }
    let pad_count = target_length - len;
    let mut suffix: Vec<u16> = Vec::with_capacity(pad_count);
    let mut filled = 0;
    while filled < pad_count {
        let remaining = pad_count - filled;
        let take = remaining.min(pad_units.len());
        suffix.extend_from_slice(&pad_units[..take]);
        filled += take;
    }
    let mut result = s.to_string();
    result.push_str(&decode_utf16(&suffix));
    result
}

// ── at ────────────────────────────────────────────────────────────────────────

/// ECMAScript §22.1.3.1 `String.prototype.at(index)`.
///
/// Returns a single-character string for the UTF-16 code unit at `index`.
/// Negative indices count from the end of the string.  Returns `None` if the
/// resolved index is out of bounds.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_at;
///
/// assert_eq!(string_at("Hello", 0), Some("H".to_string()));
/// assert_eq!(string_at("Hello", -1), Some("o".to_string()));
/// assert_eq!(string_at("Hello", 10), None);
/// ```
pub fn string_at(s: &str, index: i64) -> Option<String> {
    let units = encode_utf16(s);
    let len = units.len() as i64;
    let actual = if index < 0 { len + index } else { index };
    if actual < 0 || actual >= len {
        return None;
    }
    Some(decode_utf16(&[units[actual as usize]]))
}

// ── normalize ─────────────────────────────────────────────────────────────────

/// ECMAScript §22.1.3.11 `String.prototype.normalize(form?)`.
///
/// Returns a Unicode Normalization Form of `s`.  Accepted values for `form`
/// are `"NFC"` (default), `"NFD"`, `"NFKC"`, and `"NFKD"`.
///
/// # Current limitation
///
/// Full canonical decomposition / composition requires tables that are not
/// bundled in this build.  For ASCII-only strings all four forms are identical,
/// so those return the input unchanged.  For non-ASCII strings the current
/// implementation returns the input string as-is (which is a valid NFC
/// representation for all strings produced by Rust literals and most common
/// inputs).  A future version should integrate a crate such as
/// `unicode-normalization` to provide a complete implementation.
///
/// # Errors
///
/// Returns [`StatorError::RangeError`] if `form` is not one of the four
/// recognised values.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_normalize;
///
/// assert_eq!(string_normalize("hello", None).unwrap(), "hello");
/// assert_eq!(string_normalize("hello", Some("NFC")).unwrap(), "hello");
/// assert!(string_normalize("hello", Some("XYZ")).is_err());
/// ```
pub fn string_normalize(s: &str, form: Option<&str>) -> StatorResult<String> {
    match form.unwrap_or("NFC") {
        "NFC" | "NFD" | "NFKC" | "NFKD" => Ok(s.to_string()),
        f => Err(StatorError::RangeError(format!(
            "The normalization form should be one of NFC, NFD, NFKC, or NFKD; got \"{f}\""
        ))),
    }
}

// ── Symbol.iterator equivalent ────────────────────────────────────────────────

/// ECMAScript §22.1.3.34 `String.prototype[Symbol.iterator]()` — Rust equivalent.
///
/// Returns a `Vec<String>` where each element is a single ECMAScript character
/// (Unicode scalar value / code point) rendered as a `String`.  This matches
/// the iteration protocol defined in the spec: the iterator advances by code
/// point, not by UTF-16 code unit, so a supplementary character (e.g. 😀)
/// appears as a single two-code-unit element rather than as two separate
/// surrogate strings.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_iter;
///
/// assert_eq!(string_iter("abc"), vec!["a", "b", "c"]);
/// // The emoji is a single element even though it is two UTF-16 code units.
/// assert_eq!(string_iter("a😀b"), vec!["a", "😀", "b"]);
/// ```
pub fn string_iter(s: &str) -> Vec<String> {
    s.chars().map(|c| c.to_string()).collect()
}

// ── String.raw ────────────────────────────────────────────────────────────────

/// ECMAScript §22.1.2.4 `String.raw(strings, ...substitutions)`.
///
/// Produces a string by interleaving the *raw* template segments with the
/// provided substitutions.  This is the tagged-template equivalent: the
/// `raw_strings` slice corresponds to the `raw` property of the template
/// object (the literal portions of the template), and `substitutions` are the
/// values that fill `${…}` placeholders.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_raw;
///
/// assert_eq!(string_raw(&["hello\\n", "world"], &["!"]), "hello\\n!world");
/// assert_eq!(string_raw(&["a", "b", "c"], &["1", "2"]), "a1b2c");
/// assert_eq!(string_raw(&[], &[]), "");
/// ```
pub fn string_raw(raw_strings: &[&str], substitutions: &[&str]) -> String {
    let mut result = String::new();
    for (i, raw) in raw_strings.iter().enumerate() {
        result.push_str(raw);
        if i < substitutions.len() {
            result.push_str(substitutions[i]);
        }
    }
    result
}

// ── isWellFormed ──────────────────────────────────────────────────────────────

/// ECMAScript §22.1.3.9 `String.prototype.isWellFormed()`.
///
/// Returns `true` if the string contains no lone (unpaired) UTF-16 surrogates.
///
/// # Implementation note
///
/// Because this engine stores strings as Rust `String` values (guaranteed valid
/// UTF-8), lone surrogates can never appear.  Therefore this function always
/// returns `true`.  It is provided for spec-completeness so that user code
/// calling `"…".isWellFormed()` receives the expected result.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_is_well_formed;
///
/// assert!(string_is_well_formed("Hello 😀"));
/// assert!(string_is_well_formed(""));
/// ```
pub fn string_is_well_formed(s: &str) -> bool {
    // Rust strings are always valid UTF-8, which implies well-formed
    // UTF-16 (no lone surrogates).  Verify by checking the encode/decode
    // round-trip produces no replacement characters.
    let units = encode_utf16(s);
    let len = units.len();
    let mut i = 0;
    while i < len {
        let cu = units[i];
        if (UTF16_HIGH_SURROGATE_START..=UTF16_HIGH_SURROGATE_END).contains(&cu) {
            // High surrogate must be followed by a low surrogate.
            if i + 1 >= len
                || !(UTF16_LOW_SURROGATE_START..=UTF16_LOW_SURROGATE_END).contains(&units[i + 1])
            {
                return false;
            }
            i += 2;
        } else if (UTF16_LOW_SURROGATE_START..=UTF16_LOW_SURROGATE_END).contains(&cu) {
            // Lone low surrogate.
            return false;
        } else {
            i += 1;
        }
    }
    true
}

// ── toWellFormed ──────────────────────────────────────────────────────────────

/// ECMAScript §22.1.3.38 `String.prototype.toWellFormed()`.
///
/// Returns a new string where every lone UTF-16 surrogate is replaced by
/// `U+FFFD` (the Unicode replacement character).
///
/// # Implementation note
///
/// Rust `String` values cannot contain lone surrogates, so in practice this
/// returns a copy of the input unchanged.  Provided for spec-completeness.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_to_well_formed;
///
/// assert_eq!(string_to_well_formed("Hello"), "Hello");
/// assert_eq!(string_to_well_formed(""), "");
/// ```
pub fn string_to_well_formed(s: &str) -> String {
    // Replace any lone surrogates in the UTF-16 encoding with U+FFFD.
    let units = encode_utf16(s);
    let len = units.len();
    let mut result: Vec<u16> = Vec::with_capacity(len);
    let mut i = 0;
    while i < len {
        let cu = units[i];
        if (UTF16_HIGH_SURROGATE_START..=UTF16_HIGH_SURROGATE_END).contains(&cu) {
            if i + 1 < len
                && (UTF16_LOW_SURROGATE_START..=UTF16_LOW_SURROGATE_END).contains(&units[i + 1])
            {
                // Valid pair — keep both.
                result.push(cu);
                result.push(units[i + 1]);
                i += 2;
            } else {
                result.push(0xFFFD);
                i += 1;
            }
        } else if (UTF16_LOW_SURROGATE_START..=UTF16_LOW_SURROGATE_END).contains(&cu) {
            result.push(0xFFFD);
            i += 1;
        } else {
            result.push(cu);
            i += 1;
        }
    }
    decode_utf16(&result)
}

// ── search ────────────────────────────────────────────────────────────────────

/// ECMAScript §22.1.3.19 `String.prototype.search(regexp)`.
///
/// Searches for the first match of the regular expression `pattern` in `s` and
/// returns the index of the match as a UTF-16 code unit offset, or `-1` if no
/// match is found.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::string::string_search;
///
/// assert_eq!(string_search("hello world", r"\d+"), -1);
/// assert_eq!(string_search("abc123", r"\d+"), 3);
/// ```
pub fn string_search(s: &str, pattern: &str) -> i64 {
    let re = match regress::Regex::new(pattern) {
        Ok(r) => r,
        Err(_) => return -1,
    };
    match re.find(s) {
        Some(m) => {
            // Convert byte offset to UTF-16 code unit index.
            let byte_start = m.range().start;
            let prefix = &s[..byte_start];
            encode_utf16(prefix).len() as i64
        }
        None => -1,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── string_from_char_code ────────────────────────────────────────────────

    #[test]
    fn test_from_char_code_basic_ascii() {
        assert_eq!(string_from_char_code(&[72, 101, 108, 108, 111]), "Hello");
    }

    #[test]
    fn test_from_char_code_empty() {
        assert_eq!(string_from_char_code(&[]), "");
    }

    #[test]
    fn test_from_char_code_masking() {
        // 0x10041 masked to 0x41 = 'A'
        assert_eq!(string_from_char_code(&[0x10041]), "A");
    }

    #[test]
    fn test_from_char_code_surrogate_pair() {
        // High + low surrogate → emoji (lossy decode keeps both units)
        let s = string_from_char_code(&[0xD83D, 0xDE00]);
        // The resulting string should be "😀" (or replacement char if lossy)
        // At minimum it should not panic and should be non-empty.
        assert!(!s.is_empty());
    }

    // ── string_from_code_point ───────────────────────────────────────────────

    #[test]
    fn test_from_code_point_emoji() {
        assert_eq!(string_from_code_point(&[0x1F600]).unwrap(), "😀");
    }

    #[test]
    fn test_from_code_point_ascii() {
        assert_eq!(string_from_code_point(&[65, 66, 67]).unwrap(), "ABC");
    }

    #[test]
    fn test_from_code_point_invalid_too_large() {
        assert!(matches!(
            string_from_code_point(&[0x110000]),
            Err(StatorError::RangeError(_))
        ));
    }

    #[test]
    fn test_from_code_point_surrogate_is_invalid() {
        // U+D800 is a surrogate and not a valid scalar value.
        assert!(matches!(
            string_from_code_point(&[0xD800]),
            Err(StatorError::RangeError(_))
        ));
    }

    #[test]
    fn test_from_code_point_empty() {
        assert_eq!(string_from_code_point(&[]).unwrap(), "");
    }

    // ── string_char_at ───────────────────────────────────────────────────────

    #[test]
    fn test_char_at_ascii() {
        assert_eq!(string_char_at("Hello", 0), "H");
        assert_eq!(string_char_at("Hello", 4), "o");
    }

    #[test]
    fn test_char_at_out_of_bounds() {
        assert_eq!(string_char_at("Hello", 10), "");
    }

    #[test]
    fn test_char_at_negative() {
        assert_eq!(string_char_at("Hello", -1), "");
    }

    #[test]
    fn test_char_at_emoji_utf16_index() {
        // "😀" encodes to 2 UTF-16 code units: 0xD83D at index 0, 0xDE00 at index 1.
        let s = "😀";
        // Index 0 → high surrogate, index 1 → low surrogate.
        assert!(!string_char_at(s, 0).is_empty());
        assert!(!string_char_at(s, 1).is_empty());
        assert_eq!(string_char_at(s, 2), "");
    }

    // ── string_char_code_at ──────────────────────────────────────────────────

    #[test]
    fn test_char_code_at_ascii() {
        assert_eq!(string_char_code_at("ABC", 0), 65.0);
        assert_eq!(string_char_code_at("ABC", 2), 67.0);
    }

    #[test]
    fn test_char_code_at_out_of_bounds_is_nan() {
        assert!(string_char_code_at("ABC", 5).is_nan());
    }

    #[test]
    fn test_char_code_at_negative_is_nan() {
        assert!(string_char_code_at("ABC", -1).is_nan());
    }

    #[test]
    fn test_char_code_at_emoji_high_surrogate() {
        // "😀" high surrogate is 0xD83D = 55357
        assert_eq!(string_char_code_at("😀", 0), 55357.0);
    }

    // ── string_code_point_at ─────────────────────────────────────────────────

    #[test]
    fn test_code_point_at_ascii() {
        assert_eq!(string_code_point_at("A", 0), Some(65));
    }

    #[test]
    fn test_code_point_at_emoji() {
        assert_eq!(string_code_point_at("😀", 0), Some(0x1F600));
    }

    #[test]
    fn test_code_point_at_emoji_low_surrogate_index() {
        // Index 1 points at the low surrogate; it is returned as its own value.
        let low = string_code_point_at("😀", 1).unwrap();
        assert_eq!(low, 0xDE00);
    }

    #[test]
    fn test_code_point_at_out_of_bounds() {
        assert_eq!(string_code_point_at("A", 5), None);
    }

    #[test]
    fn test_code_point_at_negative() {
        assert_eq!(string_code_point_at("A", -1), None);
    }

    // ── string_concat ────────────────────────────────────────────────────────

    #[test]
    fn test_concat_multiple() {
        assert_eq!(
            string_concat("Hello", &[", ", "world", "!"]),
            "Hello, world!"
        );
    }

    #[test]
    fn test_concat_empty_others() {
        assert_eq!(string_concat("abc", &[]), "abc");
    }

    #[test]
    fn test_concat_unicode() {
        assert_eq!(string_concat("café", &[" 😀"]), "café 😀");
    }

    // ── string_slice ─────────────────────────────────────────────────────────

    #[test]
    fn test_slice_basic() {
        assert_eq!(string_slice("Hello", 1, Some(4)), "ell");
    }

    #[test]
    fn test_slice_no_end() {
        assert_eq!(string_slice("Hello", 2, None), "llo");
    }

    #[test]
    fn test_slice_negative_start() {
        assert_eq!(string_slice("Hello", -3, None), "llo");
    }

    #[test]
    fn test_slice_negative_end() {
        assert_eq!(string_slice("Hello", 0, Some(-1)), "Hell");
    }

    #[test]
    fn test_slice_start_ge_end_returns_empty() {
        assert_eq!(string_slice("Hello", 4, Some(2)), "");
    }

    #[test]
    fn test_slice_unicode() {
        // "café" = ['c','a','f','é'] — 4 UTF-16 code units (é is U+00E9, single unit)
        assert_eq!(string_slice("café", 0, Some(3)), "caf");
    }

    // ── string_substring ─────────────────────────────────────────────────────

    #[test]
    fn test_substring_basic() {
        assert_eq!(string_substring("Hello", 1, Some(4)), "ell");
    }

    #[test]
    fn test_substring_swaps_when_start_gt_end() {
        assert_eq!(string_substring("Hello", 4, Some(1)), "ell");
    }

    #[test]
    fn test_substring_negative_clamped_to_zero() {
        assert_eq!(string_substring("Hello", -5, Some(3)), "Hel");
    }

    #[test]
    fn test_substring_no_end() {
        assert_eq!(string_substring("Hello", 2, None), "llo");
    }

    // ── string_index_of ──────────────────────────────────────────────────────

    #[test]
    fn test_index_of_found() {
        assert_eq!(string_index_of("hello world", "world", None), 6);
    }

    #[test]
    fn test_index_of_not_found() {
        assert_eq!(string_index_of("hello world", "xyz", None), -1);
    }

    #[test]
    fn test_index_of_with_from_index() {
        assert_eq!(string_index_of("aaa", "a", Some(1)), 1);
    }

    #[test]
    fn test_index_of_empty_search() {
        // Empty search returns from_index (clamped to length).
        assert_eq!(string_index_of("hello", "", Some(2)), 2);
    }

    #[test]
    fn test_index_of_unicode() {
        let s = "café";
        assert_eq!(string_index_of(s, "é", None), 3);
    }

    // ── string_last_index_of ─────────────────────────────────────────────────

    #[test]
    fn test_last_index_of_found() {
        assert_eq!(string_last_index_of("hello world hello", "hello", None), 12);
    }

    #[test]
    fn test_last_index_of_not_found() {
        assert_eq!(string_last_index_of("hello", "xyz", None), -1);
    }

    #[test]
    fn test_last_index_of_with_from_index() {
        assert_eq!(string_last_index_of("aaa", "a", Some(1)), 1);
    }

    // ── string_includes ──────────────────────────────────────────────────────

    #[test]
    fn test_includes_true() {
        assert!(string_includes("hello world", "world", None));
    }

    #[test]
    fn test_includes_false() {
        assert!(!string_includes("hello world", "xyz", None));
    }

    #[test]
    fn test_includes_with_position() {
        assert!(!string_includes("hello", "hel", Some(1)));
    }

    // ── string_starts_with ───────────────────────────────────────────────────

    #[test]
    fn test_starts_with_true() {
        assert!(string_starts_with("Hello", "Hel", None));
    }

    #[test]
    fn test_starts_with_false() {
        assert!(!string_starts_with("Hello", "ello", None));
    }

    #[test]
    fn test_starts_with_position() {
        assert!(string_starts_with("Hello", "ello", Some(1)));
    }

    #[test]
    fn test_starts_with_empty_search() {
        assert!(string_starts_with("Hello", "", None));
    }

    // ── string_ends_with ─────────────────────────────────────────────────────

    #[test]
    fn test_ends_with_true() {
        assert!(string_ends_with("Hello", "llo", None));
    }

    #[test]
    fn test_ends_with_false() {
        assert!(!string_ends_with("Hello", "Hel", None));
    }

    #[test]
    fn test_ends_with_end_position() {
        assert!(string_ends_with("Hello", "Hel", Some(3)));
    }

    #[test]
    fn test_ends_with_empty_search() {
        assert!(string_ends_with("Hello", "", None));
    }

    // ── string_to_upper_case / lower_case ────────────────────────────────────

    #[test]
    fn test_to_upper_case_ascii() {
        assert_eq!(string_to_upper_case("hello"), "HELLO");
    }

    #[test]
    fn test_to_upper_case_unicode() {
        assert_eq!(string_to_upper_case("café"), "CAFÉ");
    }

    #[test]
    fn test_to_lower_case_ascii() {
        assert_eq!(string_to_lower_case("HELLO"), "hello");
    }

    #[test]
    fn test_to_lower_case_unicode() {
        assert_eq!(string_to_lower_case("CAFÉ"), "café");
    }

    // ── string_trim ──────────────────────────────────────────────────────────

    #[test]
    fn test_trim_basic() {
        assert_eq!(string_trim("  hello  "), "hello");
    }

    #[test]
    fn test_trim_tabs_newlines() {
        assert_eq!(string_trim("\t\nhello\r\n"), "hello");
    }

    #[test]
    fn test_trim_start() {
        assert_eq!(string_trim_start("  hello  "), "hello  ");
    }

    #[test]
    fn test_trim_end() {
        assert_eq!(string_trim_end("  hello  "), "  hello");
    }

    #[test]
    fn test_trim_empty_string() {
        assert_eq!(string_trim(""), "");
    }

    // ── string_split ─────────────────────────────────────────────────────────

    #[test]
    fn test_split_by_comma() {
        assert_eq!(string_split("a,b,c", Some(","), None), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_split_by_empty_string() {
        assert_eq!(string_split("abc", Some(""), None), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_split_no_separator() {
        assert_eq!(string_split("abc", None, None), vec!["abc"]);
    }

    #[test]
    fn test_split_with_limit() {
        assert_eq!(string_split("a,b,c", Some(","), Some(2)), vec!["a", "b"]);
    }

    #[test]
    fn test_split_with_limit_zero() {
        assert_eq!(
            string_split("a,b,c", Some(","), Some(0)),
            Vec::<String>::new()
        );
    }

    #[test]
    fn test_split_empty_string_by_empty() {
        assert_eq!(string_split("", Some(""), None), Vec::<String>::new());
    }

    #[test]
    fn test_split_emoji_by_empty() {
        // "😀" is 2 UTF-16 code units → should yield 2 elements.
        let parts = string_split("😀", Some(""), None);
        assert_eq!(parts.len(), 2);
    }

    // ── string_replace / replace_all ─────────────────────────────────────────

    #[test]
    fn test_replace_first_occurrence_only() {
        assert_eq!(string_replace("aaa", "a", "b"), "baa");
    }

    #[test]
    fn test_replace_not_found() {
        assert_eq!(string_replace("hello", "x", "y"), "hello");
    }

    #[test]
    fn test_replace_middle() {
        assert_eq!(string_replace("aabbcc", "bb", "XX"), "aaXXcc");
    }

    #[test]
    fn test_replace_all_basic() {
        assert_eq!(string_replace_all("aabbaa", "aa", "X"), "XbbX");
    }

    #[test]
    fn test_replace_all_not_found() {
        assert_eq!(string_replace_all("hello", "x", "y"), "hello");
    }

    #[test]
    fn test_replace_all_empty_search_inserts_between_units() {
        // Empty search string: replacement goes between every UTF-16 code unit
        // and at both ends.
        let result = string_replace_all("ab", "", "-");
        assert_eq!(result, "-a-b-");
    }

    // ── string_match / match_all ──────────────────────────────────────────────

    #[test]
    fn test_match_basic() {
        let m = string_match("hello world", r"(\w+)\s(\w+)").unwrap();
        assert_eq!(m[0], "hello world");
        assert_eq!(m[1], "hello");
        assert_eq!(m[2], "world");
    }

    #[test]
    fn test_match_no_match_returns_none() {
        assert!(string_match("hello", r"\d+").is_none());
    }

    #[test]
    fn test_match_invalid_pattern_returns_none() {
        assert!(string_match("hello", r"[invalid").is_none());
    }

    #[test]
    fn test_match_all_digits() {
        let matches = string_match_all("test 1 and 2 and 3", r"\d+").unwrap();
        assert_eq!(matches, vec!["1", "2", "3"]);
    }

    #[test]
    fn test_match_all_no_match_returns_none() {
        assert!(string_match_all("hello", r"\d+").is_none());
    }

    // ── string_repeat ─────────────────────────────────────────────────────────

    #[test]
    fn test_repeat_basic() {
        assert_eq!(string_repeat("ab", 3).unwrap(), "ababab");
    }

    #[test]
    fn test_repeat_zero() {
        assert_eq!(string_repeat("x", 0).unwrap(), "");
    }

    #[test]
    fn test_repeat_negative_is_error() {
        assert!(matches!(
            string_repeat("a", -1),
            Err(StatorError::RangeError(_))
        ));
    }

    #[test]
    fn test_repeat_unicode() {
        assert_eq!(string_repeat("😀", 2).unwrap(), "😀😀");
    }

    // ── string_pad_start / pad_end ───────────────────────────────────────────

    #[test]
    fn test_pad_start_default_pad() {
        assert_eq!(string_pad_start("5", 3, None), "  5");
    }

    #[test]
    fn test_pad_start_custom_pad() {
        assert_eq!(string_pad_start("5", 3, Some("0")), "005");
    }

    #[test]
    fn test_pad_start_already_long_enough() {
        assert_eq!(string_pad_start("hello", 3, None), "hello");
    }

    #[test]
    fn test_pad_start_multi_char_pad() {
        // "abc" padded to length 8 with "xy" → "xyxyxabc" (pad cycles)
        // Wait: "abc" length 3, need 5 more, "xyxyx" → "xyxyxabc"
        assert_eq!(string_pad_start("abc", 8, Some("xy")), "xyxyxabc");
    }

    #[test]
    fn test_pad_end_default_pad() {
        assert_eq!(string_pad_end("5", 3, None), "5  ");
    }

    #[test]
    fn test_pad_end_custom_pad() {
        assert_eq!(string_pad_end("5", 3, Some("0")), "500");
    }

    #[test]
    fn test_pad_end_already_long_enough() {
        assert_eq!(string_pad_end("hello", 3, None), "hello");
    }

    #[test]
    fn test_pad_end_multi_char_pad() {
        assert_eq!(string_pad_end("abc", 8, Some("xy")), "abcxyxyx");
    }

    // ── string_at ────────────────────────────────────────────────────────────

    #[test]
    fn test_at_positive_index() {
        assert_eq!(string_at("Hello", 0), Some("H".to_string()));
        assert_eq!(string_at("Hello", 4), Some("o".to_string()));
    }

    #[test]
    fn test_at_negative_index() {
        assert_eq!(string_at("Hello", -1), Some("o".to_string()));
        assert_eq!(string_at("Hello", -5), Some("H".to_string()));
    }

    #[test]
    fn test_at_out_of_bounds() {
        assert_eq!(string_at("Hello", 10), None);
        assert_eq!(string_at("Hello", -10), None);
    }

    // ── string_normalize ─────────────────────────────────────────────────────

    #[test]
    fn test_normalize_nfc_default() {
        assert_eq!(string_normalize("hello", None).unwrap(), "hello");
    }

    #[test]
    fn test_normalize_accepted_forms() {
        for form in &["NFC", "NFD", "NFKC", "NFKD"] {
            assert!(string_normalize("hello", Some(form)).is_ok());
        }
    }

    #[test]
    fn test_normalize_invalid_form_is_range_error() {
        assert!(matches!(
            string_normalize("hello", Some("XYZ")),
            Err(StatorError::RangeError(_))
        ));
    }

    // ── string_iter ──────────────────────────────────────────────────────────

    #[test]
    fn test_iter_ascii() {
        assert_eq!(string_iter("abc"), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_iter_unicode_emoji_is_single_element() {
        // The emoji is a single Unicode scalar value → one element.
        assert_eq!(string_iter("a😀b"), vec!["a", "😀", "b"]);
    }

    #[test]
    fn test_iter_empty_string() {
        assert_eq!(string_iter(""), Vec::<String>::new());
    }

    #[test]
    fn test_iter_multi_codepoint_string() {
        let chars = string_iter("héllo");
        assert_eq!(chars.len(), 5);
        assert_eq!(chars[1], "é");
    }

    // ── Unicode edge cases ────────────────────────────────────────────────────

    #[test]
    fn test_utf16_length_of_emoji_is_two() {
        // ECMAScript .length of "😀" is 2 (two UTF-16 code units).
        let units = encode_utf16("😀");
        assert_eq!(units.len(), 2);
    }

    #[test]
    fn test_slice_preserves_surrogate_pair() {
        // Slicing "😀" from 0 to 2 should return the whole emoji.
        assert_eq!(string_slice("😀", 0, Some(2)), "😀");
    }

    #[test]
    fn test_index_of_emoji() {
        let s = "a😀b";
        // "😀" starts at UTF-16 index 1.
        assert_eq!(string_index_of(s, "😀", None), 1);
    }

    #[test]
    fn test_starts_with_emoji() {
        assert!(string_starts_with("😀hello", "😀", None));
    }

    #[test]
    fn test_ends_with_emoji() {
        assert!(string_ends_with("hello😀", "😀", None));
    }

    #[test]
    fn test_pad_start_unicode_pad_string() {
        // Pad "a" to length 3 with "é" (1 UTF-16 unit each).
        // "a" has length 1, need 2 more: "éé" then "a".
        let result = string_pad_start("a", 3, Some("é"));
        assert_eq!(result, "ééa");
    }

    #[test]
    fn test_substring_unicode_boundary() {
        // "café" = c(0) a(1) f(2) é(3) — 4 UTF-16 code units
        assert_eq!(string_substring("café", 0, Some(4)), "café");
        assert_eq!(string_substring("café", 3, Some(4)), "é");
    }

    #[test]
    fn test_replace_unicode() {
        assert_eq!(string_replace("héllo", "é", "e"), "hello");
    }

    #[test]
    fn test_repeat_empty_string() {
        assert_eq!(string_repeat("", 100).unwrap(), "");
    }

    // ── string_raw ───────────────────────────────────────────────────────────

    #[test]
    fn test_raw_basic() {
        assert_eq!(string_raw(&["a", "b", "c"], &["1", "2"]), "a1b2c");
    }

    #[test]
    fn test_raw_preserves_escape_sequences() {
        assert_eq!(string_raw(&["hello\\n", "world"], &["!"]), "hello\\n!world");
    }

    #[test]
    fn test_raw_empty_strings() {
        assert_eq!(string_raw(&[], &[]), "");
    }

    #[test]
    fn test_raw_more_strings_than_substitutions() {
        assert_eq!(string_raw(&["a", "b", "c"], &["1"]), "a1bc");
    }

    #[test]
    fn test_raw_single_segment_no_substitution() {
        assert_eq!(string_raw(&["hello"], &[]), "hello");
    }

    // ── string_is_well_formed ────────────────────────────────────────────────

    #[test]
    fn test_is_well_formed_ascii() {
        assert!(string_is_well_formed("hello"));
    }

    #[test]
    fn test_is_well_formed_emoji() {
        assert!(string_is_well_formed("Hello 😀"));
    }

    #[test]
    fn test_is_well_formed_empty() {
        assert!(string_is_well_formed(""));
    }

    #[test]
    fn test_is_well_formed_unicode_chars() {
        assert!(string_is_well_formed("café résumé"));
    }

    // ── string_to_well_formed ────────────────────────────────────────────────

    #[test]
    fn test_to_well_formed_normal_string() {
        assert_eq!(string_to_well_formed("Hello"), "Hello");
    }

    #[test]
    fn test_to_well_formed_empty() {
        assert_eq!(string_to_well_formed(""), "");
    }

    #[test]
    fn test_to_well_formed_emoji() {
        assert_eq!(string_to_well_formed("a😀b"), "a😀b");
    }

    // ── string_search ────────────────────────────────────────────────────────

    #[test]
    fn test_search_found() {
        assert_eq!(string_search("abc123def", r"\d+"), 3);
    }

    #[test]
    fn test_search_not_found() {
        assert_eq!(string_search("hello world", r"\d+"), -1);
    }

    #[test]
    fn test_search_at_start() {
        assert_eq!(string_search("123abc", r"\d+"), 0);
    }

    #[test]
    fn test_search_invalid_regex() {
        assert_eq!(string_search("hello", r"[invalid"), -1);
    }

    // ── string_repeat overflow ────────────────────────────────────────────────

    #[test]
    fn test_repeat_overflow_is_range_error() {
        assert!(matches!(
            string_repeat("ab", i64::MAX),
            Err(StatorError::RangeError(_))
        ));
    }

    // ── string_starts_with edge cases ────────────────────────────────────────

    #[test]
    fn test_starts_with_position_beyond_length() {
        // Per spec, position is clamped to [0, len]. Empty search at clamped
        // position should still succeed.
        assert!(string_starts_with("Hello", "", Some(100)));
    }

    #[test]
    fn test_starts_with_empty_both() {
        assert!(string_starts_with("", "", None));
    }

    // ── string_ends_with edge cases ──────────────────────────────────────────

    #[test]
    fn test_ends_with_end_position_zero() {
        // endPosition 0 → only empty search matches.
        assert!(string_ends_with("Hello", "", Some(0)));
        assert!(!string_ends_with("Hello", "H", Some(0)));
    }

    #[test]
    fn test_ends_with_empty_both() {
        assert!(string_ends_with("", "", None));
    }
}
