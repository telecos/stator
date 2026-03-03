//! ECMAScript §19 global object properties and global functions.
//!
//! This module implements the global-scope constants and functions that are
//! available without any namespace qualifier in JavaScript:
//!
//! - [`GLOBAL_UNDEFINED`], [`GLOBAL_NAN`], [`GLOBAL_INFINITY`] — well-known
//!   constant values (`undefined`, `NaN`, `Infinity`).
//! - [`global_is_nan`] — global `isNaN(value)` with `ToNumber` coercion.
//! - [`global_is_finite`] — global `isFinite(value)` with `ToNumber` coercion.
//! - [`global_parse_int`] — global `parseInt(string, radix)`.
//! - [`global_parse_float`] — global `parseFloat(string)`.
//! - [`global_encode_uri`] — global `encodeURI(string)`.
//! - [`global_decode_uri`] — global `decodeURI(string)`.
//! - [`global_encode_uri_component`] — global `encodeURIComponent(string)`.
//! - [`global_decode_uri_component`] — global `decodeURIComponent(string)`.
//! - [`global_eval`] — global `eval(source)`: parses and executes JavaScript
//!   source code at runtime.
//!
//! # Difference from `Number.isNaN` / `Number.isFinite`
//!
//! The global `isNaN` and `isFinite` coerce their argument to a number before
//! testing (ECMAScript §19.2.2/§19.2.3), whereas `Number.isNaN` and
//! `Number.isFinite` return `false` for any non-`Number` argument without
//! coercion (ECMAScript §21.1.2.2/§21.1.2.4).
//!
//! # Naming convention
//!
//! All public items are prefixed `global_` to avoid ambiguity with standard-
//! library items and with the `Number.*` equivalents.
//!
//! # References
//!
//! * ECMAScript 2025 Language Specification §19 — *The Global Object*
//! * ECMAScript 2025 Language Specification §19.2 — *Function Properties of the Global Object*

use crate::builtins::number::{number_parse_float, number_parse_int};
use crate::error::{StatorError, StatorResult};
use crate::objects::value::JsValue;

// ── Constants (ECMAScript §19.1) ──────────────────────────────────────────────

/// The ECMAScript `undefined` value.
pub const GLOBAL_UNDEFINED: JsValue = JsValue::Undefined;

/// The IEEE 754 `NaN` (Not-a-Number) value exposed as `NaN` on the global object.
pub const GLOBAL_NAN: f64 = f64::NAN;

/// Positive infinity, exposed as `Infinity` on the global object.
pub const GLOBAL_INFINITY: f64 = f64::INFINITY;

// ── isNaN ─────────────────────────────────────────────────────────────────────

/// ECMAScript §19.2.3 global `isNaN(number)`.
///
/// Coerces `value` to a number via `ToNumber`, then returns `true` if the
/// result is `NaN`.  Unlike [`crate::builtins::number::number_is_nan`], this
/// function performs coercion, so passing a non-numeric value such as `NaN`
/// (already a number) or an unparseable string would both yield `true`.
///
/// The caller is responsible for converting a `JsValue` to `f64` via
/// `JsValue::to_number` before calling this function, matching the coercion
/// semantics.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::global::global_is_nan;
///
/// assert!(global_is_nan(f64::NAN));
/// assert!(!global_is_nan(1.0));
/// assert!(!global_is_nan(f64::INFINITY));
/// // String coercion: "abc" → NaN → true
/// assert!(global_is_nan("abc".parse::<f64>().unwrap_or(f64::NAN)));
/// // String coercion: "1" → 1.0 → false
/// assert!(!global_is_nan("1".parse::<f64>().unwrap_or(f64::NAN)));
/// ```
pub fn global_is_nan(value: f64) -> bool {
    value.is_nan()
}

// ── isFinite ──────────────────────────────────────────────────────────────────

/// ECMAScript §19.2.2 global `isFinite(number)`.
///
/// Coerces `value` to a number via `ToNumber`, then returns `true` if the
/// result is finite (not `NaN`, `+Infinity`, or `-Infinity`).  Unlike
/// [`crate::builtins::number::number_is_finite`], this function is intended
/// to be called *after* the coercion step has been performed by the caller.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::global::global_is_finite;
///
/// assert!(global_is_finite(42.0));
/// assert!(!global_is_finite(f64::INFINITY));
/// assert!(!global_is_finite(f64::NEG_INFINITY));
/// assert!(!global_is_finite(f64::NAN));
/// // String coercion: "1" → 1.0 → true
/// assert!(global_is_finite("1".parse::<f64>().unwrap_or(f64::NAN)));
/// ```
pub fn global_is_finite(value: f64) -> bool {
    value.is_finite()
}

// ── parseInt ──────────────────────────────────────────────────────────────────

/// ECMAScript §19.2.5 global `parseInt(string, radix)`.
///
/// Identical in behaviour to [`crate::builtins::number::number_parse_int`];
/// the global function is a direct alias per the specification.
///
/// Parses `string` as an integer in the given `radix` (2–36).  A `radix` of
/// `0` is treated as `10` unless the string starts with `"0x"`/`"0X"`, in
/// which case the radix is 16.  Returns `NaN` if parsing fails.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::global::global_parse_int;
///
/// assert_eq!(global_parse_int("42", 10), 42.0);
/// assert_eq!(global_parse_int("0xff", 0), 255.0);
/// assert_eq!(global_parse_int("10", 2), 2.0);
/// assert!(global_parse_int("xyz", 10).is_nan());
/// assert!(global_parse_int("", 10).is_nan());
/// ```
pub fn global_parse_int(string: &str, radix: u32) -> f64 {
    number_parse_int(string, radix)
}

// ── parseFloat ────────────────────────────────────────────────────────────────

/// ECMAScript §19.2.4 global `parseFloat(string)`.
///
/// Identical in behaviour to [`crate::builtins::number::number_parse_float`];
/// the global function is a direct alias per the specification.
///
/// Parses `string` as an IEEE 754 double.  Returns `NaN` if the string does
/// not represent a valid number.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::global::global_parse_float;
///
/// assert_eq!(global_parse_float("3.14"), 3.14);
/// assert_eq!(global_parse_float("  -2.5  "), -2.5);
/// assert!(global_parse_float("abc").is_nan());
/// assert!(global_parse_float("").is_nan());
/// ```
pub fn global_parse_float(string: &str) -> f64 {
    number_parse_float(string)
}

// ── encodeURI ─────────────────────────────────────────────────────────────────

/// ECMAScript §19.2.6.4 `encodeURI(uri)`.
///
/// Percent-encodes all characters in `uri` except those in the **unreserved
/// set** and the URI-reserved characters that have syntactic significance:
///
/// ```text
/// unreserved    = ALPHA / DIGIT / "-" / "_" / "." / "!" / "~" / "*" / "'" / "(" / ")"
/// reserved      = ";" / "," / "/" / "?" / ":" / "@" / "&" / "=" / "+" / "$" / "#"
/// ```
///
/// Characters in either set are left unencoded.  All other characters
/// (including non-ASCII bytes) are UTF-8-encoded and then each byte is
/// percent-encoded as `%XX`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::global::global_encode_uri;
///
/// assert_eq!(global_encode_uri("https://example.com/path?q=hello world"),
///            "https://example.com/path?q=hello%20world");
/// assert_eq!(global_encode_uri("foo bar"), "foo%20bar");
/// assert_eq!(global_encode_uri("http://a/b#c"), "http://a/b#c");
/// ```
pub fn global_encode_uri(uri: &str) -> String {
    percent_encode(uri, is_encode_uri_safe)
}

/// ECMAScript §19.2.6.2 `decodeURI(encodedURI)`.
///
/// Decodes percent-encoded sequences in `encoded_uri`, but does **not** decode
/// sequences that represent URI-reserved characters (`;,/?:@&=+$#`) or `%`
/// itself when followed by valid hex digits.
///
/// Returns [`StatorError::URIError`] if the string contains a malformed
/// percent-encoded sequence.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::global::global_decode_uri;
///
/// assert_eq!(global_decode_uri("hello%20world").unwrap(), "hello world");
/// assert_eq!(global_decode_uri("http://a/b#c").unwrap(), "http://a/b#c");
/// assert!(global_decode_uri("%GG").is_err());
/// ```
pub fn global_decode_uri(encoded_uri: &str) -> StatorResult<String> {
    percent_decode(encoded_uri, is_decode_uri_reserved)
}

/// ECMAScript §19.2.6.3 `encodeURIComponent(uriComponent)`.
///
/// Percent-encodes all characters in `component` except those in the
/// **unreserved set** (`ALPHA / DIGIT / "-" / "_" / "." / "!" / "~" / "*" /
/// "'" / "(" / ")"`).  Unlike [`global_encode_uri`], URI-reserved characters
/// are also encoded.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::global::global_encode_uri_component;
///
/// assert_eq!(global_encode_uri_component("hello world"), "hello%20world");
/// assert_eq!(global_encode_uri_component("a=1&b=2"), "a%3D1%26b%3D2");
/// assert_eq!(global_encode_uri_component("foo/bar"), "foo%2Fbar");
/// ```
pub fn global_encode_uri_component(component: &str) -> String {
    percent_encode(component, is_encode_uri_component_safe)
}

/// ECMAScript §19.2.6.1 `decodeURIComponent(encodedURIComponent)`.
///
/// Decodes **all** percent-encoded sequences in `encoded_component`, including
/// those representing URI-reserved characters.
///
/// Returns [`StatorError::URIError`] if the string contains a malformed
/// percent-encoded sequence.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::global::global_decode_uri_component;
///
/// assert_eq!(global_decode_uri_component("hello%20world").unwrap(), "hello world");
/// assert_eq!(global_decode_uri_component("a%3D1%26b%3D2").unwrap(), "a=1&b=2");
/// assert!(global_decode_uri_component("%GG").is_err());
/// ```
pub fn global_decode_uri_component(encoded_component: &str) -> StatorResult<String> {
    percent_decode(encoded_component, |_| false)
}

// ── eval ──────────────────────────────────────────────────────────────────────

/// ECMAScript §19.2.1 global `eval(x)`.
///
/// Parses `source` as a JavaScript program, compiles it to bytecode, and
/// executes it in a fresh top-level frame.  Returns the **completion value**:
/// if the last statement is an expression statement its value is returned,
/// otherwise `undefined` is returned.
///
/// # Errors
///
/// - [`StatorError::SyntaxError`] — the source is not valid JavaScript.
/// - Any runtime error produced during execution.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::global::global_eval;
/// use stator_core::objects::value::JsValue;
///
/// let result = global_eval("1 + 2").unwrap();
/// assert_eq!(result, JsValue::Smi(3));
///
/// let result = global_eval("true").unwrap();
/// assert_eq!(result, JsValue::Boolean(true));
/// ```
pub fn global_eval(source: &str) -> StatorResult<JsValue> {
    use crate::bytecode::bytecode_generator::BytecodeGenerator;
    use crate::interpreter::{Interpreter, InterpreterFrame};
    use crate::parser::ast::{ProgramItem, ReturnStmt, Stmt};
    use crate::parser::parse;

    let mut program = parse(source)?;

    // ECMAScript completion value: if the last item is an expression statement,
    // convert it to a `return` so the interpreter propagates the value.
    if let Some(ProgramItem::Stmt(Stmt::Expr(expr_stmt))) = program.body.last_mut() {
        let return_stmt = ReturnStmt {
            loc: expr_stmt.loc,
            argument: Some(expr_stmt.expr.clone()),
        };
        *program.body.last_mut().unwrap() = ProgramItem::Stmt(Stmt::Return(return_stmt));
    }

    let bytecode = BytecodeGenerator::compile_program(&program)?;
    let mut frame = InterpreterFrame::new(bytecode, vec![]);
    Interpreter::run(&mut frame)
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Returns `true` for characters that `encodeURI` must **not** encode:
/// unreserved characters plus URI-reserved characters.
fn is_encode_uri_safe(c: char) -> bool {
    matches!(c,
        'A'..='Z' | 'a'..='z' | '0'..='9'
        | '-' | '_' | '.' | '!' | '~' | '*' | '\'' | '(' | ')'
        // URI-reserved characters preserved by encodeURI
        | ';' | ',' | '/' | '?' | ':' | '@' | '&' | '=' | '+' | '$' | '#'
    )
}

/// Returns `true` for characters that `encodeURIComponent` must **not**
/// encode: the unreserved set only (no URI-reserved characters).
fn is_encode_uri_component_safe(c: char) -> bool {
    matches!(c,
        'A'..='Z' | 'a'..='z' | '0'..='9'
        | '-' | '_' | '.' | '!' | '~' | '*' | '\'' | '(' | ')'
    )
}

/// Returns `true` for percent-encoded sequences that `decodeURI` must **not**
/// decode because they represent URI-reserved characters.
fn is_decode_uri_reserved(byte: u8) -> bool {
    matches!(
        byte,
        b';' | b',' | b'/' | b'?' | b':' | b'@' | b'&' | b'=' | b'+' | b'$' | b'#'
    )
}

/// Percent-encode `input`, leaving characters for which `safe(c)` returns
/// `true` unencoded.  All other characters are UTF-8-encoded and each byte
/// is encoded as `%XX` (uppercase hex).
fn percent_encode(input: &str, safe: fn(char) -> bool) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        if safe(c) {
            out.push(c);
        } else {
            // Encode each UTF-8 byte of the character.
            let mut buf = [0u8; 4];
            let bytes = c.encode_utf8(&mut buf);
            for &b in bytes.as_bytes() {
                out.push('%');
                out.push(hex_digit(b >> 4));
                out.push(hex_digit(b & 0xF));
            }
        }
    }
    out
}

/// Percent-decode `input`.  If `keep_reserved(byte)` returns `true` for the
/// decoded byte, the percent-encoded sequence is left as-is in the output.
///
/// Returns [`StatorError::URIError`] for a malformed sequence.
fn percent_decode(input: &str, keep_reserved: fn(u8) -> bool) -> StatorResult<String> {
    let bytes = input.as_bytes();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'%' {
            out.push(bytes[i] as char);
            i += 1;
            continue;
        }
        // Decode one percent-encoded multi-byte UTF-8 sequence.
        // First, decode the leading byte to determine the sequence length.
        let b0 = decode_hex_byte(bytes, i)?;
        let seq_len = utf8_seq_len(b0)?;
        if seq_len == 1 {
            if keep_reserved(b0) {
                // Leave the encoded sequence verbatim.
                out.push('%');
                out.push(bytes[i + 1] as char);
                out.push(bytes[i + 2] as char);
            } else {
                out.push(b0 as char);
            }
            i += 3;
        } else {
            // Multi-byte sequence: decode all continuation bytes.
            let mut buf = [0u8; 4];
            buf[0] = b0;
            for (k, cell) in buf.iter_mut().enumerate().take(seq_len).skip(1) {
                *cell = decode_hex_byte(bytes, i + k * 3)?;
            }
            let s = std::str::from_utf8(&buf[..seq_len]).map_err(|_| {
                StatorError::URIError("malformed URI: invalid UTF-8 sequence".into())
            })?;
            out.push_str(s);
            i += seq_len * 3;
        }
    }
    Ok(out)
}

/// Decode the two hex digits after `bytes[pos]` (the `%` byte) into a `u8`.
fn decode_hex_byte(bytes: &[u8], pos: usize) -> StatorResult<u8> {
    if pos + 2 >= bytes.len() {
        return Err(StatorError::URIError(
            "malformed URI: incomplete percent-encoded sequence".into(),
        ));
    }
    let hi = from_hex_digit(bytes[pos + 1])?;
    let lo = from_hex_digit(bytes[pos + 2])?;
    Ok((hi << 4) | lo)
}

/// Return the number of UTF-8 bytes in the sequence starting with `b`.
fn utf8_seq_len(b: u8) -> StatorResult<usize> {
    if b & 0x80 == 0 {
        Ok(1)
    } else if b & 0xE0 == 0xC0 {
        Ok(2)
    } else if b & 0xF0 == 0xE0 {
        Ok(3)
    } else if b & 0xF8 == 0xF0 {
        Ok(4)
    } else {
        Err(StatorError::URIError(
            "malformed URI: invalid UTF-8 leading byte".into(),
        ))
    }
}

/// Convert an ASCII hex digit character to its numeric value.
fn from_hex_digit(b: u8) -> StatorResult<u8> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        _ => Err(StatorError::URIError(format!(
            "malformed URI: invalid hex digit '{}'",
            b as char
        ))),
    }
}

/// Convert a nibble (0–15) to an uppercase ASCII hex character.
fn hex_digit(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        _ => (b'A' + n - 10) as char,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Constants ────────────────────────────────────────────────────────────

    #[test]
    fn test_global_undefined_is_undefined() {
        assert_eq!(GLOBAL_UNDEFINED, JsValue::Undefined);
    }

    #[test]
    fn test_global_nan_is_nan() {
        assert!(GLOBAL_NAN.is_nan());
    }

    #[test]
    fn test_global_infinity_is_positive_infinity() {
        assert!(GLOBAL_INFINITY.is_infinite() && GLOBAL_INFINITY > 0.0);
    }

    // ── global_is_nan ────────────────────────────────────────────────────────

    #[test]
    fn test_global_is_nan_with_nan() {
        assert!(global_is_nan(f64::NAN));
    }

    #[test]
    fn test_global_is_nan_with_finite() {
        assert!(!global_is_nan(0.0));
        assert!(!global_is_nan(42.0));
        assert!(!global_is_nan(-1.5));
    }

    #[test]
    fn test_global_is_nan_with_infinity() {
        assert!(!global_is_nan(f64::INFINITY));
        assert!(!global_is_nan(f64::NEG_INFINITY));
    }

    #[test]
    fn test_global_is_nan_coercion_via_parse() {
        // Simulates ToNumber coercion of a non-numeric string → NaN.
        let coerced = "abc".parse::<f64>().unwrap_or(f64::NAN);
        assert!(global_is_nan(coerced));
        // Numeric string → not NaN.
        let coerced_num = "1".parse::<f64>().unwrap_or(f64::NAN);
        assert!(!global_is_nan(coerced_num));
    }

    // ── global_is_finite ─────────────────────────────────────────────────────

    #[test]
    fn test_global_is_finite_normal() {
        assert!(global_is_finite(0.0));
        assert!(global_is_finite(42.0));
        assert!(global_is_finite(-1.5));
    }

    #[test]
    fn test_global_is_finite_special() {
        assert!(!global_is_finite(f64::NAN));
        assert!(!global_is_finite(f64::INFINITY));
        assert!(!global_is_finite(f64::NEG_INFINITY));
    }

    #[test]
    fn test_global_is_finite_coercion_via_parse() {
        // "1" → 1.0 → finite
        let coerced = "1".parse::<f64>().unwrap_or(f64::NAN);
        assert!(global_is_finite(coerced));
        // "abc" → NaN → not finite
        let coerced_nan = "abc".parse::<f64>().unwrap_or(f64::NAN);
        assert!(!global_is_finite(coerced_nan));
    }

    // ── global_parse_int ─────────────────────────────────────────────────────

    #[test]
    fn test_global_parse_int_decimal() {
        assert_eq!(global_parse_int("42", 10), 42.0);
        assert_eq!(global_parse_int("-7", 10), -7.0);
        assert_eq!(global_parse_int("0", 10), 0.0);
    }

    #[test]
    fn test_global_parse_int_hex_prefix() {
        assert_eq!(global_parse_int("0xff", 0), 255.0);
        assert_eq!(global_parse_int("0xFF", 0), 255.0);
    }

    #[test]
    fn test_global_parse_int_radix() {
        assert_eq!(global_parse_int("10", 2), 2.0);
        assert_eq!(global_parse_int("ff", 16), 255.0);
        assert_eq!(global_parse_int("z", 36), 35.0);
    }

    #[test]
    fn test_global_parse_int_invalid() {
        assert!(global_parse_int("xyz", 10).is_nan());
        assert!(global_parse_int("", 10).is_nan());
    }

    #[test]
    fn test_global_parse_int_leading_whitespace() {
        assert_eq!(global_parse_int("  42  ", 10), 42.0);
    }

    #[test]
    fn test_global_parse_int_partial_match() {
        // Stops at first non-digit character.
        assert_eq!(global_parse_int("123abc", 10), 123.0);
    }

    // ── global_parse_float ───────────────────────────────────────────────────

    #[test]
    fn test_global_parse_float_normal() {
        assert_eq!(global_parse_float("3.14"), 3.14);
        assert_eq!(global_parse_float("-2.5"), -2.5);
        assert_eq!(global_parse_float("1e10"), 1e10);
    }

    #[test]
    fn test_global_parse_float_whitespace() {
        assert_eq!(global_parse_float("  1.5  "), 1.5);
    }

    #[test]
    fn test_global_parse_float_invalid() {
        assert!(global_parse_float("abc").is_nan());
        assert!(global_parse_float("").is_nan());
    }

    #[test]
    fn test_global_parse_float_infinity() {
        assert_eq!(global_parse_float("inf"), f64::INFINITY);
        assert_eq!(global_parse_float("-inf"), f64::NEG_INFINITY);
    }

    // ── global_encode_uri ────────────────────────────────────────────────────

    #[test]
    fn test_encode_uri_preserves_unreserved() {
        assert_eq!(global_encode_uri("abcXYZ0-_.!~*'()"), "abcXYZ0-_.!~*'()");
    }

    #[test]
    fn test_encode_uri_preserves_reserved() {
        assert_eq!(global_encode_uri(";,/?:@&=+$#"), ";,/?:@&=+$#");
    }

    #[test]
    fn test_encode_uri_encodes_space() {
        assert_eq!(global_encode_uri("hello world"), "hello%20world");
    }

    #[test]
    fn test_encode_uri_full_url() {
        assert_eq!(
            global_encode_uri("https://example.com/path?q=hello world"),
            "https://example.com/path?q=hello%20world"
        );
    }

    #[test]
    fn test_encode_uri_non_ascii() {
        // 'é' is U+00E9 → UTF-8 bytes 0xC3 0xA9 → %C3%A9
        assert_eq!(global_encode_uri("caf\u{00E9}"), "caf%C3%A9");
    }

    #[test]
    fn test_encode_uri_empty() {
        assert_eq!(global_encode_uri(""), "");
    }

    // ── global_decode_uri ────────────────────────────────────────────────────

    #[test]
    fn test_decode_uri_space() {
        assert_eq!(global_decode_uri("hello%20world").unwrap(), "hello world");
    }

    #[test]
    fn test_decode_uri_preserves_reserved_sequences() {
        // %2F is '/', a URI-reserved char — must not be decoded by decodeURI.
        assert_eq!(global_decode_uri("a%2Fb").unwrap(), "a%2Fb");
    }

    #[test]
    fn test_decode_uri_non_ascii() {
        assert_eq!(global_decode_uri("caf%C3%A9").unwrap(), "caf\u{00E9}");
    }

    #[test]
    fn test_decode_uri_passthrough() {
        assert_eq!(
            global_decode_uri("https://example.com/path").unwrap(),
            "https://example.com/path"
        );
    }

    #[test]
    fn test_decode_uri_malformed_sequence() {
        assert!(global_decode_uri("%GG").is_err());
        assert!(global_decode_uri("%2").is_err());
    }

    // ── global_encode_uri_component ──────────────────────────────────────────

    #[test]
    fn test_encode_uri_component_preserves_unreserved() {
        assert_eq!(
            global_encode_uri_component("abcABC0-_.!~*'()"),
            "abcABC0-_.!~*'()"
        );
    }

    #[test]
    fn test_encode_uri_component_encodes_reserved() {
        assert_eq!(global_encode_uri_component("a=1&b=2"), "a%3D1%26b%3D2");
    }

    #[test]
    fn test_encode_uri_component_encodes_slash() {
        assert_eq!(global_encode_uri_component("foo/bar"), "foo%2Fbar");
    }

    #[test]
    fn test_encode_uri_component_space() {
        assert_eq!(global_encode_uri_component("hello world"), "hello%20world");
    }

    #[test]
    fn test_encode_uri_component_non_ascii() {
        assert_eq!(global_encode_uri_component("caf\u{00E9}"), "caf%C3%A9");
    }

    #[test]
    fn test_encode_uri_component_empty() {
        assert_eq!(global_encode_uri_component(""), "");
    }

    // ── global_decode_uri_component ──────────────────────────────────────────

    #[test]
    fn test_decode_uri_component_space() {
        assert_eq!(
            global_decode_uri_component("hello%20world").unwrap(),
            "hello world"
        );
    }

    #[test]
    fn test_decode_uri_component_decodes_all() {
        assert_eq!(
            global_decode_uri_component("a%3D1%26b%3D2").unwrap(),
            "a=1&b=2"
        );
    }

    #[test]
    fn test_decode_uri_component_slash() {
        // decodeURIComponent should decode %2F (unlike decodeURI).
        assert_eq!(global_decode_uri_component("foo%2Fbar").unwrap(), "foo/bar");
    }

    #[test]
    fn test_decode_uri_component_non_ascii() {
        assert_eq!(
            global_decode_uri_component("caf%C3%A9").unwrap(),
            "caf\u{00E9}"
        );
    }

    #[test]
    fn test_decode_uri_component_malformed() {
        assert!(global_decode_uri_component("%GG").is_err());
        assert!(global_decode_uri_component("%2").is_err());
    }

    // ── global_eval ──────────────────────────────────────────────────────────

    #[test]
    fn test_global_eval_arithmetic() {
        let result = global_eval("1 + 2").unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    #[test]
    fn test_global_eval_number_expression() {
        let result = global_eval("40 + 2").unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn test_global_eval_boolean() {
        let result = global_eval("true").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_global_eval_complex_expression() {
        let result = global_eval("(function() { var x = 10; return x * 2; })()").unwrap();
        assert_eq!(result, JsValue::Smi(20));
    }

    #[test]
    fn test_global_eval_syntax_error() {
        let err = global_eval("(((").unwrap_err();
        assert!(matches!(err, StatorError::SyntaxError(_)));
    }

    #[test]
    fn test_global_eval_returns_undefined_for_empty() {
        // An empty program should return undefined.
        let result = global_eval("").unwrap();
        assert_eq!(result, JsValue::Undefined);
    }
}
