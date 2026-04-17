//! ECMAScript §21.1 `Number` built-in static methods and prototype equivalents.
//!
//! Every function in this module is a direct Rust equivalent of either a static
//! property of the JavaScript `Number` constructor or a method on
//! `Number.prototype`.  They operate on plain `f64` values and have no
//! side-effects beyond the values passed in.
//!
//! # Naming convention
//!
//! Each function is prefixed `number_` to avoid ambiguity with similarly-named
//! standard-library items (e.g. `number_is_nan` vs `f64::is_nan`).
//!
//! # References
//!
//! * ECMAScript 2025 Language Specification §21.1 — *The Number Constructor*

use num_bigint::BigUint;

use crate::error::{StatorError, StatorResult};
use crate::objects::value::number_to_string;

// ── Constants (ECMAScript §21.1.2) ────────────────────────────────────────────

/// The largest safe integer in JavaScript: `2^53 − 1`.
pub const NUMBER_MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0_f64;

/// The smallest safe integer in JavaScript: `-(2^53 − 1)`.
pub const NUMBER_MIN_SAFE_INTEGER: f64 = -9_007_199_254_740_991.0_f64;

/// The largest finite floating-point number representable in JavaScript.
pub const NUMBER_MAX_VALUE: f64 = f64::MAX;

/// The smallest positive floating-point number representable in JavaScript
/// (the smallest subnormal, `5e-324`).
pub const NUMBER_MIN_VALUE: f64 = 5e-324;

/// The difference between 1.0 and the next representable `f64` value.
pub const NUMBER_EPSILON: f64 = f64::EPSILON;

/// Positive infinity.
pub const NUMBER_POSITIVE_INFINITY: f64 = f64::INFINITY;

/// Negative infinity.
pub const NUMBER_NEGATIVE_INFINITY: f64 = f64::NEG_INFINITY;

/// Not-a-Number.
pub const NUMBER_NAN: f64 = f64::NAN;

// ── Number.isFinite ───────────────────────────────────────────────────────────

/// ECMAScript §21.1.2.2 `Number.isFinite(value)`.
///
/// Returns `true` if `value` is a finite number (not `NaN`, `+Infinity`, or
/// `-Infinity`).
///
/// # Examples
///
/// ```
/// use stator_js::builtins::number::number_is_finite;
///
/// assert!(number_is_finite(42.0));
/// assert!(!number_is_finite(f64::INFINITY));
/// assert!(!number_is_finite(f64::NEG_INFINITY));
/// assert!(!number_is_finite(f64::NAN));
/// ```
pub fn number_is_finite(value: f64) -> bool {
    value.is_finite()
}

// ── Number.isInteger ──────────────────────────────────────────────────────────

/// ECMAScript §21.1.2.3 `Number.isInteger(value)`.
///
/// Returns `true` if `value` is a finite number with no fractional part.
///
/// # Examples
///
/// ```
/// use stator_js::builtins::number::number_is_integer;
///
/// assert!(number_is_integer(42.0));
/// assert!(number_is_integer(-7.0));
/// assert!(number_is_integer(0.0));
/// assert!(!number_is_integer(3.14));
/// assert!(!number_is_integer(f64::NAN));
/// assert!(!number_is_integer(f64::INFINITY));
/// ```
pub fn number_is_integer(value: f64) -> bool {
    value.is_finite() && value.fract() == 0.0
}

// ── Number.isNaN ──────────────────────────────────────────────────────────────

/// ECMAScript §21.1.2.4 `Number.isNaN(value)`.
///
/// Returns `true` if `value` is the IEEE 754 `NaN` value.  Unlike the global
/// `isNaN`, this function does **not** coerce its argument.
///
/// # Examples
///
/// ```
/// use stator_js::builtins::number::number_is_nan;
///
/// assert!(number_is_nan(f64::NAN));
/// assert!(!number_is_nan(42.0));
/// assert!(!number_is_nan(f64::INFINITY));
/// ```
pub fn number_is_nan(value: f64) -> bool {
    value.is_nan()
}

// ── Number.isSafeInteger ──────────────────────────────────────────────────────

/// ECMAScript §21.1.2.5 `Number.isSafeInteger(value)`.
///
/// Returns `true` if `value` is an integer in the range
/// `[-(2^53 − 1), 2^53 − 1]` (inclusive).
///
/// # Examples
///
/// ```
/// use stator_js::builtins::number::{number_is_safe_integer, NUMBER_MAX_SAFE_INTEGER};
///
/// assert!(number_is_safe_integer(0.0));
/// assert!(number_is_safe_integer(NUMBER_MAX_SAFE_INTEGER));
/// assert!(!number_is_safe_integer(NUMBER_MAX_SAFE_INTEGER + 1.0));
/// assert!(!number_is_safe_integer(3.14));
/// ```
pub fn number_is_safe_integer(value: f64) -> bool {
    number_is_integer(value) && value.abs() <= NUMBER_MAX_SAFE_INTEGER
}

// ── Number.parseInt / Number.parseFloat ──────────────────────────────────────

/// ECMAScript §21.1.2.6 `Number.parseInt(string, radix)`.
///
/// Parses a string as an integer in the given `radix` (2–36).  A `radix` of
/// `0` or `None` is treated as `10` unless the string starts with `"0x"` /
/// `"0X"`, in which case the radix is 16.  Returns `NaN` if parsing fails.
///
/// # Examples
///
/// ```
/// use stator_js::builtins::number::number_parse_int;
///
/// assert_eq!(number_parse_int("42", 10), 42.0);
/// assert_eq!(number_parse_int("0xff", 0), 255.0);
/// assert_eq!(number_parse_int("10", 2), 2.0);
/// assert!(number_parse_int("xyz", 10).is_nan());
/// ```
pub fn number_parse_int(string: &str, radix: i32) -> f64 {
    let s = string.trim_start();
    if s.is_empty() {
        return f64::NAN;
    }

    // Handle optional leading sign.
    let (negative, s) = if let Some(rest) = s.strip_prefix('-') {
        (true, rest)
    } else if let Some(rest) = s.strip_prefix('+') {
        (false, rest)
    } else {
        (false, s)
    };

    // Determine radix and strip any `0x`/`0X` prefix.
    let (effective_radix, digits) =
        if radix == 16 || radix == 0 && (s.starts_with("0x") || s.starts_with("0X")) {
            let stripped = s
                .strip_prefix("0x")
                .or_else(|| s.strip_prefix("0X"))
                .unwrap_or(s);
            (16u32, stripped)
        } else if radix == 0 || radix == 10 {
            (10u32, s)
        } else if !(2..=36).contains(&radix) {
            return f64::NAN;
        } else {
            (radix as u32, s)
        };

    // Consume the longest valid prefix.
    let valid: String = digits
        .chars()
        .take_while(|c| c.is_digit(effective_radix))
        .collect();

    if valid.is_empty() {
        return f64::NAN;
    }

    // Parse using u64 to cover the full safe-integer range, then convert.
    let parsed = u64::from_str_radix(&valid, effective_radix)
        .map(|n| n as f64)
        .unwrap_or_else(|_| parse_int_digits(&valid, effective_radix));

    if negative { -parsed } else { parsed }
}

/// ECMAScript §21.1.2.7 `Number.parseFloat(string)`.
///
/// Parses the longest valid numeric prefix of `string` as an IEEE 754 double.
/// Returns `NaN` if no valid prefix exists.  Recognises `"Infinity"` and
/// `"-Infinity"` per spec.
///
/// # Examples
///
/// ```
/// use stator_js::builtins::number::number_parse_float;
///
/// assert_eq!(number_parse_float("3.14"), 3.14);
/// assert_eq!(number_parse_float("  -2.5  "), -2.5);
/// assert_eq!(number_parse_float("3.14abc"), 3.14);
/// assert_eq!(number_parse_float("Infinity"), f64::INFINITY);
/// assert!(number_parse_float("abc").is_nan());
/// ```
pub fn number_parse_float(string: &str) -> f64 {
    let s = string.trim_start();
    if s.is_empty() {
        return f64::NAN;
    }

    // Handle optional sign + "Infinity".
    let (sign, rest) = match s.as_bytes().first() {
        Some(b'+') => (1.0_f64, &s[1..]),
        Some(b'-') => (-1.0_f64, &s[1..]),
        _ => (1.0_f64, s),
    };
    if rest.starts_with("Infinity") {
        return sign * f64::INFINITY;
    }

    // Find the longest prefix that forms a valid StrDecimalLiteral.
    let end = float_prefix_end(s);
    if end == 0 {
        return f64::NAN;
    }
    s[..end].parse::<f64>().unwrap_or(f64::NAN)
}

/// Returns the byte-length of the longest valid numeric prefix in `s`.
fn float_prefix_end(s: &str) -> usize {
    let b = s.as_bytes();
    let mut i = 0;

    // Optional sign.
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }

    // Integer part.
    let mut has_digits = false;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
        has_digits = true;
    }

    // Fractional part.
    if i < b.len() && b[i] == b'.' {
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
            has_digits = true;
        }
    }

    if !has_digits {
        return 0;
    }
    let mut end = i;

    // Exponent part.
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        let mut j = i + 1;
        if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
            j += 1;
        }
        if j < b.len() && b[j].is_ascii_digit() {
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            end = j;
        }
    }
    end
}

// ── Number.prototype.toFixed ──────────────────────────────────────────────────

/// ECMAScript §21.1.3.3 `Number.prototype.toFixed(digits)`.
///
/// Returns a string representation of `value` with exactly `digits` digits
/// after the decimal point.  `digits` must be in `[0, 100]`.
///
/// Returns [`StatorError::RangeError`] if `digits` is out of range or if
/// `value` is too large to represent with fixed notation.
///
/// # Examples
///
/// ```
/// use stator_js::builtins::number::number_to_fixed;
///
/// assert_eq!(number_to_fixed(3.14159, 2).unwrap(), "3.14");
/// assert_eq!(number_to_fixed(1.0, 0).unwrap(), "1");
/// assert_eq!(number_to_fixed(0.1 + 0.2, 1).unwrap(), "0.3");
/// ```
pub fn number_to_fixed(value: f64, digits: u32) -> StatorResult<String> {
    if digits > 100 {
        return Err(StatorError::RangeError(
            "toFixed() digits must be between 0 and 100".to_string(),
        ));
    }
    if value.is_nan() {
        return Ok("NaN".to_string());
    }
    if value.is_infinite() {
        return Ok(if value > 0.0 {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        });
    }
    // Per spec §21.1.3.3: values >= 1e21 return ToString(value).
    if value.abs() >= 1e21 {
        return Ok(number_to_string(value));
    }
    let rounded = round_to_fixed_integer(value.abs(), digits);
    let magnitude = format_fixed_digits(rounded, digits);
    if value.is_sign_negative() && value != 0.0 {
        Ok(format!("-{magnitude}"))
    } else {
        Ok(magnitude)
    }
}

fn parse_int_digits(digits: &str, radix: u32) -> f64 {
    let base = f64::from(radix);
    digits
        .chars()
        .try_fold(0.0_f64, |acc, ch| {
            let digit = ch.to_digit(radix)?;
            Some(acc.mul_add(base, f64::from(digit)))
        })
        .unwrap_or(f64::NAN)
}

fn decompose_finite_positive(value: f64) -> (u64, i32) {
    let bits = value.to_bits();
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction_bits = bits & ((1_u64 << 52) - 1);
    if exponent_bits == 0 {
        (fraction_bits, -1074)
    } else {
        ((1_u64 << 52) | fraction_bits, exponent_bits - 1075)
    }
}

fn round_to_decimal_integer(value: f64, decimal_shift: i32) -> BigUint {
    if value == 0.0 {
        return BigUint::default();
    }

    let (mantissa, binary_exponent) = decompose_finite_positive(value);
    let mut numerator = BigUint::from(mantissa);
    let mut denominator = BigUint::from(1_u8);

    if decimal_shift >= 0 {
        numerator *= BigUint::from(5_u8).pow(decimal_shift as u32);
    } else {
        denominator *= BigUint::from(5_u8).pow((-decimal_shift) as u32);
    }

    let binary_shift = binary_exponent + decimal_shift;
    if binary_shift >= 0 {
        numerator <<= binary_shift as usize;
    } else {
        denominator <<= (-binary_shift) as usize;
    }

    let mut rounded = &numerator / &denominator;
    let remainder = numerator % &denominator;
    if (remainder << 1usize) >= denominator {
        rounded += BigUint::from(1_u8);
    }
    rounded
}

fn round_to_fixed_integer(value: f64, digits: u32) -> BigUint {
    round_to_decimal_integer(value, digits as i32)
}

fn decimal_exponent(value: f64) -> i32 {
    let text = number_to_string(value);
    if let Some(pos) = text.find('e') {
        return text[pos + 1..].parse().unwrap_or(0);
    }

    if let Some(pos) = text.find('.') {
        if let Some(fraction) = text.strip_prefix("0.") {
            let leading_zero_count = fraction.bytes().take_while(|digit| *digit == b'0').count();
            return -((leading_zero_count as i32) + 1);
        }
        return pos as i32 - 1;
    }

    text.len() as i32 - 1
}

fn format_exponential_digits(digits: &str, exponent: i32) -> String {
    let mut chars = digits.chars();
    let first = chars.next().unwrap_or('0');
    let rest: String = chars.collect();
    let exponent_sign = if exponent >= 0 { '+' } else { '-' };
    let exponent_magnitude = exponent.unsigned_abs();
    if rest.is_empty() {
        format!("{first}e{exponent_sign}{exponent_magnitude}")
    } else {
        format!("{first}.{rest}e{exponent_sign}{exponent_magnitude}")
    }
}

fn pad_significant_digits(mut digits: String, width: usize) -> String {
    while digits.len() < width {
        digits.insert(0, '0');
    }
    digits
}

fn pow10_biguint(power: usize) -> BigUint {
    BigUint::from(10_u8).pow(power as u32)
}

fn format_precision_digits(digits: &str, exponent: i32, precision: u32) -> String {
    if exponent < -6 || exponent >= precision as i32 {
        return format_exponential_digits(digits, exponent);
    }

    if exponent >= 0 {
        let integer_len = exponent as usize + 1;
        if integer_len >= digits.len() {
            let mut out = digits.to_string();
            out.push_str(&"0".repeat(integer_len - digits.len()));
            out
        } else {
            format!("{}.{}", &digits[..integer_len], &digits[integer_len..])
        }
    } else {
        format!("0.{}{}", "0".repeat((-exponent - 1) as usize), digits)
    }
}

fn format_zero_precision(precision: u32) -> String {
    if precision == 1 {
        "0".to_string()
    } else {
        format!("0.{}", "0".repeat((precision - 1) as usize))
    }
}

fn rounded_significant_digits(value: f64, precision: u32) -> (String, i32) {
    let mut exponent = decimal_exponent(value);
    let mut rounded = round_to_decimal_integer(value, precision as i32 - 1 - exponent);
    if rounded == pow10_biguint(precision as usize) {
        rounded /= 10_u8;
        exponent += 1;
    }

    (
        pad_significant_digits(rounded.to_str_radix(10), precision as usize),
        exponent,
    )
}

fn rounded_exponential_digits(value: f64, fraction_digits: u32) -> (String, i32) {
    let significant_digits = fraction_digits + 1;
    rounded_significant_digits(value, significant_digits)
}

fn format_fixed_digits(integer: BigUint, digits: u32) -> String {
    let mut text = integer.to_str_radix(10);
    if digits == 0 {
        return text;
    }

    let fraction_len = digits as usize;
    if text.len() <= fraction_len {
        let padding = "0".repeat(fraction_len + 1 - text.len());
        text = format!("{padding}{text}");
    }
    let split = text.len() - fraction_len;
    text.insert(split, '.');
    text
}

// ── Number.prototype.toPrecision ──────────────────────────────────────────────

/// ECMAScript §21.1.3.5 `Number.prototype.toPrecision(precision)`.
///
/// Returns a string representation of `value` with `precision` significant
/// digits.  `precision` must be in `[1, 100]`.
///
/// Returns [`StatorError::RangeError`] if `precision` is out of range.
///
/// # Examples
///
/// ```
/// use stator_js::builtins::number::number_to_precision;
///
/// assert_eq!(number_to_precision(123.456, 5).unwrap(), "123.46");
/// assert_eq!(number_to_precision(0.000123, 2).unwrap(), "0.00012");
/// ```
pub fn number_to_precision(value: f64, precision: u32) -> StatorResult<String> {
    if !(1..=100).contains(&precision) {
        return Err(StatorError::RangeError(
            "toPrecision() precision must be between 1 and 100".to_string(),
        ));
    }
    if value.is_nan() {
        return Ok("NaN".to_string());
    }
    if value.is_infinite() {
        return Ok(if value > 0.0 {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        });
    }
    if value == 0.0 {
        return Ok(format_zero_precision(precision));
    }

    let sign = if value.is_sign_negative() { "-" } else { "" };
    let (digits, exponent) = rounded_significant_digits(value.abs(), precision);
    Ok(format!(
        "{sign}{}",
        format_precision_digits(&digits, exponent, precision)
    ))
}

// ── Number.prototype.toExponential ────────────────────────────────────────────

/// ECMAScript §21.1.3.2 `Number.prototype.toExponential(fraction_digits)`.
///
/// Returns a string in exponential notation with `fraction_digits` digits
/// after the decimal point.  `fraction_digits` must be in `[0, 100]`.
///
/// Returns [`StatorError::RangeError`] if `fraction_digits` is out of range.
///
/// # Examples
///
/// ```
/// use stator_js::builtins::number::number_to_exponential;
///
/// assert_eq!(number_to_exponential(123456.0, 3).unwrap(), "1.235e+5");
/// assert_eq!(number_to_exponential(0.0, 2).unwrap(), "0.00e+0");
/// ```
pub fn number_to_exponential(value: f64, fraction_digits: u32) -> StatorResult<String> {
    if fraction_digits > 100 {
        return Err(StatorError::RangeError(
            "toExponential() fractionDigits must be between 0 and 100".to_string(),
        ));
    }
    if value.is_nan() {
        return Ok("NaN".to_string());
    }
    if value.is_infinite() {
        return Ok(if value > 0.0 {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        });
    }
    if value == 0.0 {
        let digits = format!("0{}", "0".repeat(fraction_digits as usize));
        return Ok(format_exponential_digits(&digits, 0));
    }

    let sign = if value.is_sign_negative() { "-" } else { "" };
    let (digits, exponent) = rounded_exponential_digits(value.abs(), fraction_digits);
    Ok(format!(
        "{sign}{}",
        format_exponential_digits(&digits, exponent)
    ))
}

/// Converts Rust's `{:e}` output (e.g. `"1.23e5"`) to ECMAScript exponential
/// notation (e.g. `"1.23e+5"`).  The exponent is always preceded by a sign and
/// has no leading zeros beyond what is necessary.
pub fn number_reformat_exponential(s: &str) -> String {
    if let Some(pos) = s.find('e') {
        let mantissa = &s[..pos];
        let exp_str = &s[pos + 1..];
        let exp: i32 = exp_str.parse().unwrap_or(0);
        if exp >= 0 {
            format!("{mantissa}e+{exp}")
        } else {
            format!("{mantissa}e{exp}")
        }
    } else {
        s.to_string()
    }
}

// ── Number.prototype.toString ─────────────────────────────────────────────────

/// ECMAScript §21.1.3.6 `Number.prototype.toString(radix)`.
///
/// Converts `value` to a string in the given `radix` (2–36).  A `radix` of
/// `10` produces the standard decimal representation.
///
/// Returns [`StatorError::RangeError`] if `radix` is not in `[2, 36]`.
///
/// # Examples
///
/// ```
/// use stator_js::builtins::number::number_to_string_radix;
///
/// assert_eq!(number_to_string_radix(255.0, 16).unwrap(), "ff");
/// assert_eq!(number_to_string_radix(10.0, 2).unwrap(), "1010");
/// assert_eq!(number_to_string_radix(42.0, 10).unwrap(), "42");
/// ```
pub fn number_to_string_radix(value: f64, radix: u32) -> StatorResult<String> {
    if !(2..=36).contains(&radix) {
        return Err(StatorError::RangeError(
            "toString() radix must be between 2 and 36".to_string(),
        ));
    }
    if value.is_nan() {
        return Ok("NaN".to_string());
    }
    if value.is_infinite() {
        return Ok(if value > 0.0 {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        });
    }
    if radix == 10 {
        // Delegate to the standard decimal formatter.
        if value.fract() == 0.0 && value.abs() < 1e15 {
            return Ok(format!("{}", value as i64));
        }
        return Ok(format!("{value}"));
    }

    // For integer values in the safe-integer range, use integer radix conversion.
    if value.fract() == 0.0 && value.abs() <= NUMBER_MAX_SAFE_INTEGER {
        let n = value as i64;
        return Ok(int_to_string_radix(n, radix));
    }

    // For non-integer values outside radix 10, fall back to decimal.
    Ok(format!("{value}"))
}

/// Converts a signed 64-bit integer to a string in the given radix.
fn int_to_string_radix(n: i64, radix: u32) -> String {
    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    if n == 0 {
        return "0".to_string();
    }
    let negative = n < 0;
    // Use u64 arithmetic to handle i64::MIN without overflow.
    let mut val = if negative {
        (-(n as i128)) as u64
    } else {
        n as u64
    };
    let mut buf = Vec::new();
    while val > 0 {
        buf.push(DIGITS[(val % radix as u64) as usize]);
        val /= radix as u64;
    }
    if negative {
        buf.push(b'-');
    }
    buf.reverse();
    String::from_utf8(buf).unwrap_or_default()
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Constants ────────────────────────────────────────────────────────────

    #[test]
    fn test_max_safe_integer_value() {
        assert_eq!(NUMBER_MAX_SAFE_INTEGER, 9_007_199_254_740_991.0);
    }

    #[test]
    fn test_min_safe_integer_value() {
        assert_eq!(NUMBER_MIN_SAFE_INTEGER, -9_007_199_254_740_991.0);
    }

    // ── number_is_finite ─────────────────────────────────────────────────────

    #[test]
    fn test_is_finite_normal() {
        assert!(number_is_finite(42.0));
        assert!(number_is_finite(-1.5));
        assert!(number_is_finite(0.0));
    }

    #[test]
    fn test_is_finite_special() {
        assert!(!number_is_finite(f64::NAN));
        assert!(!number_is_finite(f64::INFINITY));
        assert!(!number_is_finite(f64::NEG_INFINITY));
    }

    // ── number_is_integer ────────────────────────────────────────────────────

    #[test]
    fn test_is_integer_whole_numbers() {
        assert!(number_is_integer(0.0));
        assert!(number_is_integer(1.0));
        assert!(number_is_integer(-42.0));
        assert!(number_is_integer(1e15));
    }

    #[test]
    fn test_is_integer_fractions() {
        assert!(!number_is_integer(3.14));
        assert!(!number_is_integer(-0.5));
    }

    #[test]
    fn test_is_integer_special() {
        assert!(!number_is_integer(f64::NAN));
        assert!(!number_is_integer(f64::INFINITY));
        assert!(!number_is_integer(f64::NEG_INFINITY));
    }

    // ── number_is_nan ────────────────────────────────────────────────────────

    #[test]
    fn test_is_nan_positive() {
        assert!(number_is_nan(f64::NAN));
    }

    #[test]
    fn test_is_nan_negative() {
        assert!(!number_is_nan(0.0));
        assert!(!number_is_nan(42.0));
        assert!(!number_is_nan(f64::INFINITY));
    }

    // ── number_is_safe_integer ───────────────────────────────────────────────

    #[test]
    fn test_is_safe_integer_within_range() {
        assert!(number_is_safe_integer(0.0));
        assert!(number_is_safe_integer(NUMBER_MAX_SAFE_INTEGER));
        assert!(number_is_safe_integer(NUMBER_MIN_SAFE_INTEGER));
    }

    #[test]
    fn test_is_safe_integer_outside_range() {
        assert!(!number_is_safe_integer(NUMBER_MAX_SAFE_INTEGER + 1.0));
        assert!(!number_is_safe_integer(NUMBER_MIN_SAFE_INTEGER - 1.0));
    }

    #[test]
    fn test_is_safe_integer_fraction() {
        assert!(!number_is_safe_integer(1.5));
    }

    // ── number_parse_int ─────────────────────────────────────────────────────

    #[test]
    fn test_parse_int_decimal() {
        assert_eq!(number_parse_int("42", 10), 42.0);
        assert_eq!(number_parse_int("-7", 10), -7.0);
        assert_eq!(number_parse_int("  100  ", 10), 100.0);
    }

    #[test]
    fn test_parse_int_hex_prefix() {
        assert_eq!(number_parse_int("0xff", 0), 255.0);
        assert_eq!(number_parse_int("0XFF", 16), 255.0);
    }

    #[test]
    fn test_parse_int_binary() {
        assert_eq!(number_parse_int("1010", 2), 10.0);
    }

    #[test]
    fn test_parse_int_invalid() {
        assert!(number_parse_int("xyz", 10).is_nan());
        assert!(number_parse_int("", 10).is_nan());
    }

    #[test]
    fn test_parse_int_partial_prefix() {
        // "12abc" → 12 (stops at first non-digit)
        assert_eq!(number_parse_int("12abc", 10), 12.0);
    }

    #[test]
    fn test_parse_int_invalid_radix() {
        assert!(number_parse_int("10", 1).is_nan());
        assert!(number_parse_int("10", 37).is_nan());
    }

    #[test]
    fn test_parse_int_default_radix_cases() {
        assert_eq!(number_parse_int("10", 0), 10.0);
        assert_eq!(number_parse_int("0x10", 0), 16.0);
        assert_eq!(number_parse_int("077", 0), 77.0);
    }

    // ── number_parse_float ───────────────────────────────────────────────────

    #[test]
    fn test_parse_float_normal() {
        assert_eq!(number_parse_float("3.14"), 3.14);
        assert_eq!(number_parse_float("  -2.5  "), -2.5);
        assert_eq!(number_parse_float("1e10"), 1e10);
    }

    #[test]
    fn test_parse_float_invalid() {
        assert!(number_parse_float("abc").is_nan());
        assert!(number_parse_float("").is_nan());
    }

    #[test]
    fn test_parse_float_prefix() {
        // Parses the longest valid numeric prefix.
        assert_eq!(number_parse_float("3.14abc"), 3.14);
        assert_eq!(number_parse_float("123xyz"), 123.0);
    }

    #[test]
    fn test_parse_float_infinity() {
        assert_eq!(number_parse_float("Infinity"), f64::INFINITY);
        assert_eq!(number_parse_float("-Infinity"), f64::NEG_INFINITY);
        assert_eq!(number_parse_float("+Infinity"), f64::INFINITY);
    }

    // ── number_to_fixed ──────────────────────────────────────────────────────

    #[test]
    fn test_to_fixed_basic() {
        assert_eq!(number_to_fixed(3.14159, 2).unwrap(), "3.14");
        assert_eq!(number_to_fixed(1.0, 0).unwrap(), "1");
        assert_eq!(number_to_fixed(1.005, 2).unwrap(), "1.00");
        assert_eq!(number_to_fixed(0.615, 2).unwrap(), "0.61");
    }

    #[test]
    fn test_to_fixed_nan_and_infinity() {
        assert_eq!(number_to_fixed(f64::NAN, 2).unwrap(), "NaN");
        assert_eq!(number_to_fixed(f64::INFINITY, 2).unwrap(), "Infinity");
        assert_eq!(number_to_fixed(f64::NEG_INFINITY, 2).unwrap(), "-Infinity");
    }

    #[test]
    fn test_to_fixed_out_of_range_digits() {
        assert!(matches!(
            number_to_fixed(1.0, 101),
            Err(StatorError::RangeError(_))
        ));
    }

    #[test]
    fn test_to_fixed_large_value() {
        // Per spec, values >= 1e21 return ToString(value), not a RangeError.
        let s = number_to_fixed(1e22, 2).unwrap();
        assert!(!s.is_empty());
    }

    // ── number_to_precision ──────────────────────────────────────────────────

    #[test]
    fn test_to_precision_basic() {
        assert_eq!(number_to_precision(123.456, 5).unwrap(), "123.46");
    }

    #[test]
    fn test_to_precision_nan_infinity() {
        assert_eq!(number_to_precision(f64::NAN, 3).unwrap(), "NaN");
        assert_eq!(number_to_precision(f64::INFINITY, 3).unwrap(), "Infinity");
    }

    #[test]
    fn test_to_precision_out_of_range() {
        assert!(matches!(
            number_to_precision(1.0, 0),
            Err(StatorError::RangeError(_))
        ));
        assert!(matches!(
            number_to_precision(1.0, 101),
            Err(StatorError::RangeError(_))
        ));
    }

    // ── number_to_exponential ────────────────────────────────────────────────

    #[test]
    fn test_to_exponential_basic() {
        assert_eq!(number_to_exponential(123456.0, 3).unwrap(), "1.235e+5");
    }

    #[test]
    fn test_to_exponential_zero() {
        assert_eq!(number_to_exponential(0.0, 2).unwrap(), "0.00e+0");
    }

    // ── Number() type conversion (via string_to_number / to_number) ──────

    #[test]
    fn test_number_conversion_empty_string() {
        // Number("") → 0
        assert_eq!(
            crate::objects::value::JsValue::String("".into())
                .to_number()
                .unwrap(),
            0.0
        );
    }

    #[test]
    fn test_number_conversion_whitespace_string() {
        // Number("  ") → 0
        assert_eq!(
            crate::objects::value::JsValue::String("  ".into())
                .to_number()
                .unwrap(),
            0.0
        );
    }

    #[test]
    fn test_number_conversion_null() {
        // Number(null) → 0
        assert_eq!(
            crate::objects::value::JsValue::Null.to_number().unwrap(),
            0.0
        );
    }

    #[test]
    fn test_number_conversion_undefined() {
        // Number(undefined) → NaN
        assert!(
            crate::objects::value::JsValue::Undefined
                .to_number()
                .unwrap()
                .is_nan()
        );
    }

    #[test]
    fn test_number_conversion_true() {
        // Number(true) → 1
        assert_eq!(
            crate::objects::value::JsValue::Boolean(true)
                .to_number()
                .unwrap(),
            1.0
        );
    }

    #[test]
    fn test_number_conversion_false() {
        // Number(false) → 0
        assert_eq!(
            crate::objects::value::JsValue::Boolean(false)
                .to_number()
                .unwrap(),
            0.0
        );
    }

    #[test]
    fn test_number_conversion_hex() {
        // Number("0x1A") → 26
        assert_eq!(
            crate::objects::value::JsValue::String("0x1A".into())
                .to_number()
                .unwrap(),
            26.0
        );
    }

    #[test]
    fn test_number_conversion_octal() {
        // Number("0o17") → 15
        assert_eq!(
            crate::objects::value::JsValue::String("0o17".into())
                .to_number()
                .unwrap(),
            15.0
        );
    }

    #[test]
    fn test_number_conversion_binary() {
        // Number("0b11") → 3
        assert_eq!(
            crate::objects::value::JsValue::String("0b11".into())
                .to_number()
                .unwrap(),
            3.0
        );
    }

    // ── Number.prototype.toFixed edge cases ─────────────────────────────

    #[test]
    fn test_to_fixed_zero_digits() {
        assert_eq!(number_to_fixed(1.5, 0).unwrap(), "2");
        assert_eq!(number_to_fixed(2.5, 0).unwrap(), "3");
        assert_eq!(number_to_fixed(-2.5, 0).unwrap(), "-3");
    }

    #[test]
    fn test_to_fixed_negative_zero() {
        // (-0).toFixed(1) → "0.0"
        assert_eq!(number_to_fixed(-0.0, 1).unwrap(), "0.0");
    }

    // ── Number.prototype.toPrecision edge cases ─────────────────────────

    #[test]
    fn test_to_precision_small_fraction() {
        assert_eq!(number_to_precision(0.000123, 2).unwrap(), "0.00012");
    }

    #[test]
    fn test_to_precision_integer() {
        assert_eq!(number_to_precision(100.0, 1).unwrap(), "1e+2");
    }

    #[test]
    fn test_to_precision_large_integer() {
        assert_eq!(number_to_precision(1000.0, 4).unwrap(), "1000");
    }

    #[test]
    fn test_to_precision_switches_to_exponential_for_tiny_values() {
        assert_eq!(number_to_precision(1e-7, 1).unwrap(), "1e-7");
    }

    #[test]
    fn test_to_precision_stays_fixed_at_negative_six_exponent() {
        assert_eq!(number_to_precision(1e-6, 1).unwrap(), "0.000001");
    }

    #[test]
    fn test_to_precision_rounds_exact_half_up() {
        assert_eq!(number_to_precision(12.5, 2).unwrap(), "13");
    }

    #[test]
    fn test_to_precision_zero_padding() {
        assert_eq!(number_to_precision(0.0, 4).unwrap(), "0.000");
    }

    // ── Number.prototype.toString(radix) ────────────────────────────────

    #[test]
    fn test_to_string_radix_hex() {
        assert_eq!(number_to_string_radix(255.0, 16).unwrap(), "ff");
    }

    #[test]
    fn test_to_string_radix_binary() {
        assert_eq!(number_to_string_radix(10.0, 2).unwrap(), "1010");
    }

    #[test]
    fn test_to_string_radix_base36() {
        assert_eq!(number_to_string_radix(35.0, 36).unwrap(), "z");
    }

    #[test]
    fn test_to_string_radix_nan() {
        assert_eq!(number_to_string_radix(f64::NAN, 16).unwrap(), "NaN");
    }

    #[test]
    fn test_to_string_radix_infinity() {
        assert_eq!(
            number_to_string_radix(f64::INFINITY, 16).unwrap(),
            "Infinity"
        );
    }

    #[test]
    fn test_to_string_radix_negative_infinity() {
        assert_eq!(
            number_to_string_radix(f64::NEG_INFINITY, 2).unwrap(),
            "-Infinity"
        );
    }

    #[test]
    fn test_to_string_radix_invalid() {
        assert!(number_to_string_radix(1.0, 1).is_err());
        assert!(number_to_string_radix(1.0, 37).is_err());
    }

    // ── Number constants ────────────────────────────────────────────────

    #[test]
    fn test_number_epsilon() {
        assert_eq!(NUMBER_EPSILON, f64::EPSILON);
    }

    #[test]
    fn test_number_max_value() {
        assert_eq!(NUMBER_MAX_VALUE, f64::MAX);
    }

    #[test]
    fn test_number_min_value() {
        assert_eq!(NUMBER_MIN_VALUE, 5e-324);
    }

    #[test]
    fn test_number_positive_infinity() {
        assert_eq!(NUMBER_POSITIVE_INFINITY, f64::INFINITY);
    }

    #[test]
    fn test_number_negative_infinity() {
        assert_eq!(NUMBER_NEGATIVE_INFINITY, f64::NEG_INFINITY);
    }

    #[test]
    fn test_number_nan_is_nan() {
        assert!(NUMBER_NAN.is_nan());
    }

    // ── parseInt edge cases ─────────────────────────────────────────────

    #[test]
    fn test_parse_int_leading_whitespace() {
        assert_eq!(number_parse_int("  42", 10), 42.0);
    }

    #[test]
    fn test_parse_int_plus_sign() {
        assert_eq!(number_parse_int("+10", 10), 10.0);
    }

    #[test]
    fn test_parse_int_hex_0x_prefix_radix_16() {
        assert_eq!(number_parse_int("0xFF", 16), 255.0);
    }

    // ── parseFloat edge cases ───────────────────────────────────────────

    #[test]
    fn test_parse_float_leading_whitespace() {
        assert_eq!(number_parse_float("  3.14"), 3.14);
    }

    #[test]
    fn test_parse_float_plus_sign() {
        assert_eq!(number_parse_float("+2.5"), 2.5);
    }

    #[test]
    fn test_to_exponential_nan_infinity() {
        assert_eq!(number_to_exponential(f64::NAN, 2).unwrap(), "NaN");
        assert_eq!(number_to_exponential(f64::INFINITY, 2).unwrap(), "Infinity");
        assert_eq!(
            number_to_exponential(f64::NEG_INFINITY, 2).unwrap(),
            "-Infinity"
        );
    }

    #[test]
    fn test_to_exponential_out_of_range() {
        assert!(matches!(
            number_to_exponential(1.0, 101),
            Err(StatorError::RangeError(_))
        ));
    }

    #[test]
    fn test_to_exponential_negative() {
        let s = number_to_exponential(0.001, 2).unwrap();
        assert_eq!(s, "1.00e-3");
    }

    #[test]
    fn test_to_exponential_rounds_exact_half_up() {
        assert_eq!(number_to_exponential(1.25, 1).unwrap(), "1.3e+0");
        assert_eq!(number_to_exponential(-1.25, 1).unwrap(), "-1.3e+0");
    }

    #[test]
    fn test_to_exponential_rounding_carries_into_exponent() {
        assert_eq!(number_to_exponential(99.5, 1).unwrap(), "1.0e+2");
    }

    #[test]
    fn test_to_exponential_negative_zero_formats_without_sign() {
        assert_eq!(number_to_exponential(-0.0, 2).unwrap(), "0.00e+0");
    }

    // ── number_to_string_radix ────────────────────────────────────────────────

    #[test]
    fn test_to_string_radix_decimal() {
        assert_eq!(number_to_string_radix(42.0, 10).unwrap(), "42");
    }

    #[test]
    fn test_to_string_radix_hex_v2() {
        assert_eq!(number_to_string_radix(255.0, 16).unwrap(), "ff");
    }

    #[test]
    fn test_to_string_radix_binary_v2() {
        assert_eq!(number_to_string_radix(10.0, 2).unwrap(), "1010");
    }

    #[test]
    fn test_to_string_radix_octal() {
        assert_eq!(number_to_string_radix(8.0, 8).unwrap(), "10");
    }

    #[test]
    fn test_to_string_radix_nan_infinity() {
        assert_eq!(number_to_string_radix(f64::NAN, 16).unwrap(), "NaN");
        assert_eq!(
            number_to_string_radix(f64::INFINITY, 16).unwrap(),
            "Infinity"
        );
    }

    #[test]
    fn test_to_string_radix_invalid_radix() {
        assert!(matches!(
            number_to_string_radix(10.0, 1),
            Err(StatorError::RangeError(_))
        ));
        assert!(matches!(
            number_to_string_radix(10.0, 37),
            Err(StatorError::RangeError(_))
        ));
    }

    #[test]
    fn test_to_string_radix_negative() {
        assert_eq!(number_to_string_radix(-255.0, 16).unwrap(), "-ff");
    }

    // ── IEEE 754 edge cases ───────────────────────────────────────────────────

    #[test]
    fn test_ieee754_negative_zero_is_finite() {
        assert!(number_is_finite(-0.0_f64));
    }

    #[test]
    fn test_ieee754_negative_zero_is_integer() {
        assert!(number_is_integer(-0.0_f64));
    }

    #[test]
    fn test_ieee754_nan_is_not_equal_to_itself() {
        let nan = f64::NAN;
        // number_is_nan detects NaN correctly.
        assert!(number_is_nan(nan));
        // Any arithmetic with NaN produces NaN.
        assert!(number_is_nan(nan + 1.0));
        assert!(number_is_nan(nan - nan));
        // NaN is not equal to itself (the defining IEEE 754 property).
        assert!(nan != nan);
    }

    #[test]
    fn test_ieee754_subnormal_is_finite() {
        let subnormal = f64::MIN_POSITIVE / 2.0;
        assert!(number_is_finite(subnormal));
        assert!(!number_is_integer(subnormal));
    }

    #[test]
    fn test_ieee754_max_safe_integer_plus_one_is_not_safe() {
        // 2^53 is representable but not safe because 2^53 + 1 == 2^53 in f64.
        assert!(!number_is_safe_integer(NUMBER_MAX_SAFE_INTEGER + 1.0));
    }

    #[test]
    fn test_min_value_is_subnormal() {
        // NUMBER_MIN_VALUE must be the smallest positive subnormal (5e-324).
        assert!(NUMBER_MIN_VALUE > 0.0);
        assert!(NUMBER_MIN_VALUE < f64::MIN_POSITIVE);
        assert_eq!(NUMBER_MIN_VALUE, 5e-324);
    }
}
