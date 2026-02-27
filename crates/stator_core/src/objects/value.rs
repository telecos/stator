//! JavaScript value representation.
//!
//! This module provides [`JsValue`], the top-level enum that can hold any
//! ECMAScript value, together with type-checking predicates and the three
//! abstract type-conversion operations defined in ECMAScript §7.1:
//! [`to_boolean`][JsValue::to_boolean] (§7.1.2),
//! [`to_number`][JsValue::to_number] (§7.1.4), and
//! [`to_js_string`][JsValue::to_js_string] (§7.1.17).

use crate::error::{StatorError, StatorResult};
use crate::objects::heap_object::HeapObject;

/// Any ECMAScript value.
///
/// Primitive variants carry their data inline; `Object` holds a raw pointer to
/// a GC-managed [`HeapObject`].
///
/// # Safety – `Object` variant
///
/// The pointer stored in `Object(ptr)` must refer to a live object managed by
/// the engine heap.  It is the caller's responsibility to ensure the object
/// outlives the `JsValue` that wraps it and that no GC compaction has
/// invalidated the pointer.
#[derive(Debug, Clone)]
pub enum JsValue {
    /// The ECMAScript `undefined` primitive.
    Undefined,
    /// The ECMAScript `null` primitive.
    Null,
    /// A JavaScript boolean (`true` or `false`).
    Boolean(bool),
    /// A small (31-bit signed) integer, stored inline without heap allocation.
    Smi(i32),
    /// A double-precision floating-point number stored inline.
    HeapNumber(f64),
    /// A JavaScript string value.
    String(String),
    /// A unique JavaScript symbol, identified by an opaque 64-bit descriptor.
    Symbol(u64),
    /// A pointer to a GC-managed heap object.
    Object(*mut HeapObject),
    /// A JavaScript `BigInt` value (represented as a 128-bit signed integer).
    BigInt(i128),
}

// ──────────────────────────────────────────────────────────────────────────────
// Type-checking predicates
// ──────────────────────────────────────────────────────────────────────────────

impl JsValue {
    /// Returns `true` if this value is `undefined`.
    #[inline]
    pub fn is_undefined(&self) -> bool {
        matches!(self, Self::Undefined)
    }

    /// Returns `true` if this value is `null`.
    #[inline]
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Returns `true` if this value is `null` or `undefined`.
    #[inline]
    pub fn is_nullish(&self) -> bool {
        matches!(self, Self::Null | Self::Undefined)
    }

    /// Returns `true` if this value is a boolean.
    #[inline]
    pub fn is_boolean(&self) -> bool {
        matches!(self, Self::Boolean(_))
    }

    /// Returns `true` if this value is a small integer ([`Smi`][JsValue::Smi]).
    #[inline]
    pub fn is_smi(&self) -> bool {
        matches!(self, Self::Smi(_))
    }

    /// Returns `true` if this value is a heap number ([`HeapNumber`][JsValue::HeapNumber]).
    #[inline]
    pub fn is_heap_number(&self) -> bool {
        matches!(self, Self::HeapNumber(_))
    }

    /// Returns `true` if this value is any numeric type (`Smi` or `HeapNumber`).
    #[inline]
    pub fn is_number(&self) -> bool {
        matches!(self, Self::Smi(_) | Self::HeapNumber(_))
    }

    /// Returns `true` if this value is a string.
    #[inline]
    pub fn is_string(&self) -> bool {
        matches!(self, Self::String(_))
    }

    /// Returns `true` if this value is a symbol.
    #[inline]
    pub fn is_symbol(&self) -> bool {
        matches!(self, Self::Symbol(_))
    }

    /// Returns `true` if this value is an object.
    #[inline]
    pub fn is_object(&self) -> bool {
        matches!(self, Self::Object(_))
    }

    /// Returns `true` if this value is a `BigInt`.
    #[inline]
    pub fn is_bigint(&self) -> bool {
        matches!(self, Self::BigInt(_))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Abstract type-conversion operations (ECMAScript §7.1)
// ──────────────────────────────────────────────────────────────────────────────

impl JsValue {
    /// ECMAScript §7.1.2 **ToBoolean**.
    ///
    /// | Value type | Result |
    /// |---|---|
    /// | `Undefined` | `false` |
    /// | `Null` | `false` |
    /// | `Boolean` | the boolean itself |
    /// | `Smi` | `false` if `0`, otherwise `true` |
    /// | `HeapNumber` | `false` if `+0.0`, `-0.0`, or `NaN`; otherwise `true` |
    /// | `String` | `false` if the string is empty; otherwise `true` |
    /// | `Symbol` | `true` |
    /// | `Object` | `true` |
    /// | `BigInt` | `false` if `0`, otherwise `true` |
    pub fn to_boolean(&self) -> bool {
        match self {
            Self::Undefined | Self::Null => false,
            Self::Boolean(b) => *b,
            Self::Smi(n) => *n != 0,
            Self::HeapNumber(n) => !n.is_nan() && *n != 0.0,
            Self::String(s) => !s.is_empty(),
            Self::Symbol(_) | Self::Object(_) => true,
            Self::BigInt(n) => *n != 0,
        }
    }

    /// ECMAScript §7.1.4 **ToNumber**.
    ///
    /// Returns `Err(StatorError::TypeError)` for `Symbol`, `BigInt`, and
    /// `Object` (the latter would require a `ToPrimitive` call that is not yet
    /// implemented).
    ///
    /// | Value type | Result |
    /// |---|---|
    /// | `Undefined` | `NaN` |
    /// | `Null` | `+0.0` |
    /// | `Boolean` | `0.0` / `1.0` |
    /// | `Smi` | integer cast to `f64` |
    /// | `HeapNumber` | the value itself |
    /// | `String` | parsed numeric value, or `NaN` if not a valid number |
    /// | `Symbol` | `TypeError` |
    /// | `Object` | `TypeError` (ToPrimitive not yet implemented) |
    /// | `BigInt` | `TypeError` |
    pub fn to_number(&self) -> StatorResult<f64> {
        match self {
            Self::Undefined => Ok(f64::NAN),
            Self::Null => Ok(0.0),
            Self::Boolean(b) => Ok(if *b { 1.0 } else { 0.0 }),
            Self::Smi(n) => Ok(f64::from(*n)),
            Self::HeapNumber(n) => Ok(*n),
            Self::String(s) => {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    Ok(0.0)
                } else {
                    // Per ECMAScript, strings that are not valid numeric literals
                    // produce NaN rather than an error.
                    Ok(trimmed.parse::<f64>().unwrap_or(f64::NAN))
                }
            }
            Self::Symbol(_) => Err(StatorError::TypeError(
                "Cannot convert a Symbol value to a number".to_string(),
            )),
            Self::Object(_) => Err(StatorError::TypeError(
                "Cannot convert an Object to a number without ToPrimitive".to_string(),
            )),
            Self::BigInt(_) => Err(StatorError::TypeError(
                "Cannot convert a BigInt value to a number".to_string(),
            )),
        }
    }

    /// ECMAScript §7.1.17 **ToString**.
    ///
    /// Named `to_js_string` to avoid ambiguity with [`ToString::to_string`].
    ///
    /// Returns `Err(StatorError::TypeError)` for `Symbol` and `Object` (the
    /// latter would require a `ToPrimitive` call that is not yet implemented).
    ///
    /// | Value type | Result |
    /// |---|---|
    /// | `Undefined` | `"undefined"` |
    /// | `Null` | `"null"` |
    /// | `Boolean` | `"true"` / `"false"` |
    /// | `Smi` | decimal string |
    /// | `HeapNumber` | number string (`"NaN"`, `"Infinity"`, `"-Infinity"`, or decimal) |
    /// | `String` | the string itself |
    /// | `Symbol` | `TypeError` |
    /// | `Object` | `TypeError` (ToPrimitive not yet implemented) |
    /// | `BigInt` | decimal string |
    pub fn to_js_string(&self) -> StatorResult<String> {
        match self {
            Self::Undefined => Ok("undefined".to_string()),
            Self::Null => Ok("null".to_string()),
            Self::Boolean(b) => Ok(if *b { "true" } else { "false" }.to_string()),
            Self::Smi(n) => Ok(n.to_string()),
            Self::HeapNumber(n) => Ok(number_to_string(*n)),
            Self::String(s) => Ok(s.clone()),
            Self::Symbol(_) => Err(StatorError::TypeError(
                "Cannot convert a Symbol value to a string".to_string(),
            )),
            Self::Object(_) => Err(StatorError::TypeError(
                "Cannot convert an Object to a string without ToPrimitive".to_string(),
            )),
            Self::BigInt(n) => Ok(n.to_string()),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Formats an `f64` as a JavaScript number string (ECMAScript §7.1.12.1).
///
/// Special cases: `NaN → "NaN"`, `+∞ → "Infinity"`, `-∞ → "-Infinity"`,
/// and both `+0.0` and `-0.0` → `"0"`.  All other values use Rust's default
/// `f64` `Display` formatting, which provides a minimal decimal representation
/// compatible with the ECMAScript spec for common values.
fn number_to_string(n: f64) -> String {
    if n.is_nan() {
        "NaN".to_string()
    } else if n.is_infinite() {
        if n > 0.0 {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        }
    } else if n == 0.0 {
        // Both +0.0 and -0.0 produce "0".
        "0".to_string()
    } else {
        format!("{n}")
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::heap_object::HeapObject;

    // ── is_* predicates ──────────────────────────────────────────────────────

    #[test]
    fn test_is_undefined() {
        assert!(JsValue::Undefined.is_undefined());
        assert!(!JsValue::Null.is_undefined());
        assert!(!JsValue::Smi(0).is_undefined());
    }

    #[test]
    fn test_is_null() {
        assert!(JsValue::Null.is_null());
        assert!(!JsValue::Undefined.is_null());
        assert!(!JsValue::Boolean(false).is_null());
    }

    #[test]
    fn test_is_nullish() {
        assert!(JsValue::Undefined.is_nullish());
        assert!(JsValue::Null.is_nullish());
        assert!(!JsValue::Boolean(false).is_nullish());
        assert!(!JsValue::Smi(0).is_nullish());
    }

    #[test]
    fn test_is_boolean() {
        assert!(JsValue::Boolean(true).is_boolean());
        assert!(JsValue::Boolean(false).is_boolean());
        assert!(!JsValue::Smi(0).is_boolean());
    }

    #[test]
    fn test_is_smi() {
        assert!(JsValue::Smi(42).is_smi());
        assert!(!JsValue::HeapNumber(42.0).is_smi());
    }

    #[test]
    fn test_is_heap_number() {
        assert!(JsValue::HeapNumber(3.14).is_heap_number());
        assert!(!JsValue::Smi(3).is_heap_number());
    }

    #[test]
    fn test_is_number() {
        assert!(JsValue::Smi(0).is_number());
        assert!(JsValue::HeapNumber(0.0).is_number());
        assert!(!JsValue::Boolean(false).is_number());
        assert!(!JsValue::Null.is_number());
    }

    #[test]
    fn test_is_string() {
        assert!(JsValue::String("hello".to_string()).is_string());
        assert!(!JsValue::Smi(0).is_string());
    }

    #[test]
    fn test_is_symbol() {
        assert!(JsValue::Symbol(1).is_symbol());
        assert!(!JsValue::String("sym".to_string()).is_symbol());
    }

    #[test]
    fn test_is_object() {
        let mut obj = HeapObject::new_null();
        let ptr = &raw mut obj;
        assert!(JsValue::Object(ptr).is_object());
        assert!(!JsValue::Null.is_object());
    }

    #[test]
    fn test_is_bigint() {
        assert!(JsValue::BigInt(0).is_bigint());
        assert!(!JsValue::Smi(0).is_bigint());
    }

    // ── to_boolean ───────────────────────────────────────────────────────────

    #[test]
    fn test_to_boolean_undefined_is_false() {
        assert!(!JsValue::Undefined.to_boolean());
    }

    #[test]
    fn test_to_boolean_null_is_false() {
        assert!(!JsValue::Null.to_boolean());
    }

    #[test]
    fn test_to_boolean_boolean_passthrough() {
        assert!(JsValue::Boolean(true).to_boolean());
        assert!(!JsValue::Boolean(false).to_boolean());
    }

    #[test]
    fn test_to_boolean_smi_zero_is_false() {
        assert!(!JsValue::Smi(0).to_boolean());
        assert!(JsValue::Smi(1).to_boolean());
        assert!(JsValue::Smi(-1).to_boolean());
    }

    #[test]
    fn test_to_boolean_heap_number_special_cases() {
        assert!(!JsValue::HeapNumber(0.0).to_boolean());
        assert!(!JsValue::HeapNumber(-0.0).to_boolean());
        assert!(!JsValue::HeapNumber(f64::NAN).to_boolean());
        assert!(JsValue::HeapNumber(1.0).to_boolean());
        assert!(JsValue::HeapNumber(-1.0).to_boolean());
        assert!(JsValue::HeapNumber(f64::INFINITY).to_boolean());
    }

    #[test]
    fn test_to_boolean_string() {
        assert!(!JsValue::String(String::new()).to_boolean());
        assert!(JsValue::String("x".to_string()).to_boolean());
        assert!(JsValue::String("false".to_string()).to_boolean());
    }

    #[test]
    fn test_to_boolean_symbol_is_true() {
        assert!(JsValue::Symbol(0).to_boolean());
        assert!(JsValue::Symbol(u64::MAX).to_boolean());
    }

    #[test]
    fn test_to_boolean_object_is_true() {
        let mut obj = HeapObject::new_null();
        let ptr = &raw mut obj;
        assert!(JsValue::Object(ptr).to_boolean());
    }

    #[test]
    fn test_to_boolean_bigint_zero_is_false() {
        assert!(!JsValue::BigInt(0).to_boolean());
        assert!(JsValue::BigInt(1).to_boolean());
        assert!(JsValue::BigInt(-1).to_boolean());
    }

    // ── to_number ────────────────────────────────────────────────────────────

    #[test]
    fn test_to_number_undefined_is_nan() {
        let n = JsValue::Undefined.to_number().unwrap();
        assert!(n.is_nan());
    }

    #[test]
    fn test_to_number_null_is_zero() {
        assert_eq!(JsValue::Null.to_number().unwrap(), 0.0);
    }

    #[test]
    fn test_to_number_boolean() {
        assert_eq!(JsValue::Boolean(true).to_number().unwrap(), 1.0);
        assert_eq!(JsValue::Boolean(false).to_number().unwrap(), 0.0);
    }

    #[test]
    fn test_to_number_smi() {
        assert_eq!(JsValue::Smi(42).to_number().unwrap(), 42.0);
        assert_eq!(JsValue::Smi(-1).to_number().unwrap(), -1.0);
        assert_eq!(JsValue::Smi(0).to_number().unwrap(), 0.0);
    }

    #[test]
    fn test_to_number_heap_number_passthrough() {
        assert_eq!(JsValue::HeapNumber(3.14).to_number().unwrap(), 3.14);
        let n = JsValue::HeapNumber(f64::NAN).to_number().unwrap();
        assert!(n.is_nan());
    }

    #[test]
    fn test_to_number_string_numeric() {
        assert_eq!(JsValue::String("42".to_string()).to_number().unwrap(), 42.0);
        assert_eq!(
            JsValue::String("  3.14  ".to_string()).to_number().unwrap(),
            3.14
        );
        assert_eq!(JsValue::String("".to_string()).to_number().unwrap(), 0.0);
        assert_eq!(JsValue::String("   ".to_string()).to_number().unwrap(), 0.0);
    }

    #[test]
    fn test_to_number_string_non_numeric_gives_nan() {
        let n = JsValue::String("abc".to_string()).to_number().unwrap();
        assert!(n.is_nan());
    }

    #[test]
    fn test_to_number_symbol_is_type_error() {
        assert!(matches!(
            JsValue::Symbol(1).to_number(),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_to_number_object_is_type_error() {
        let mut obj = HeapObject::new_null();
        let ptr = &raw mut obj;
        assert!(matches!(
            JsValue::Object(ptr).to_number(),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_to_number_bigint_is_type_error() {
        assert!(matches!(
            JsValue::BigInt(42).to_number(),
            Err(StatorError::TypeError(_))
        ));
    }

    // ── to_js_string ─────────────────────────────────────────────────────────

    #[test]
    fn test_to_js_string_undefined() {
        assert_eq!(JsValue::Undefined.to_js_string().unwrap(), "undefined");
    }

    #[test]
    fn test_to_js_string_null() {
        assert_eq!(JsValue::Null.to_js_string().unwrap(), "null");
    }

    #[test]
    fn test_to_js_string_boolean() {
        assert_eq!(JsValue::Boolean(true).to_js_string().unwrap(), "true");
        assert_eq!(JsValue::Boolean(false).to_js_string().unwrap(), "false");
    }

    #[test]
    fn test_to_js_string_smi() {
        assert_eq!(JsValue::Smi(0).to_js_string().unwrap(), "0");
        assert_eq!(JsValue::Smi(42).to_js_string().unwrap(), "42");
        assert_eq!(JsValue::Smi(-7).to_js_string().unwrap(), "-7");
    }

    #[test]
    fn test_to_js_string_heap_number_special_cases() {
        assert_eq!(JsValue::HeapNumber(f64::NAN).to_js_string().unwrap(), "NaN");
        assert_eq!(
            JsValue::HeapNumber(f64::INFINITY).to_js_string().unwrap(),
            "Infinity"
        );
        assert_eq!(
            JsValue::HeapNumber(f64::NEG_INFINITY)
                .to_js_string()
                .unwrap(),
            "-Infinity"
        );
        assert_eq!(JsValue::HeapNumber(0.0).to_js_string().unwrap(), "0");
        assert_eq!(JsValue::HeapNumber(-0.0).to_js_string().unwrap(), "0");
    }

    #[test]
    fn test_to_js_string_heap_number_normal() {
        assert_eq!(JsValue::HeapNumber(42.0).to_js_string().unwrap(), "42");
        assert_eq!(JsValue::HeapNumber(3.14).to_js_string().unwrap(), "3.14");
    }

    #[test]
    fn test_to_js_string_string_passthrough() {
        assert_eq!(
            JsValue::String("hello".to_string()).to_js_string().unwrap(),
            "hello"
        );
        assert_eq!(JsValue::String(String::new()).to_js_string().unwrap(), "");
    }

    #[test]
    fn test_to_js_string_symbol_is_type_error() {
        assert!(matches!(
            JsValue::Symbol(1).to_js_string(),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_to_js_string_object_is_type_error() {
        let mut obj = HeapObject::new_null();
        let ptr = &raw mut obj;
        assert!(matches!(
            JsValue::Object(ptr).to_js_string(),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_to_js_string_bigint() {
        assert_eq!(JsValue::BigInt(0).to_js_string().unwrap(), "0");
        assert_eq!(JsValue::BigInt(12345).to_js_string().unwrap(), "12345");
        assert_eq!(JsValue::BigInt(-99).to_js_string().unwrap(), "-99");
    }

    // ── number_to_string helper ───────────────────────────────────────────────

    #[test]
    fn test_number_to_string_nan() {
        assert_eq!(number_to_string(f64::NAN), "NaN");
    }

    #[test]
    fn test_number_to_string_positive_infinity() {
        assert_eq!(number_to_string(f64::INFINITY), "Infinity");
    }

    #[test]
    fn test_number_to_string_negative_infinity() {
        assert_eq!(number_to_string(f64::NEG_INFINITY), "-Infinity");
    }

    #[test]
    fn test_number_to_string_negative_zero() {
        assert_eq!(number_to_string(-0.0_f64), "0");
    }
}
