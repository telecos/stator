//! Shared utility functions used across the `builtins` sub-modules.

use crate::error::{StatorError, StatorResult};
use crate::objects::value::JsValue;

// ── Safe numeric conversions ──────────────────────────────────────────────────

/// Maximum number of elements we allow in a single allocation originating from
/// a JS-visible length/size value.  This is deliberately much smaller than
/// `usize::MAX` so that a corrupt or adversarial `f64` can never cause a
/// multi-TiB allocation attempt.  2^32 − 1 matches the ECMAScript maximum
/// array length and is already more than enough for any single allocation.
pub(crate) const MAX_ALLOCATION_LENGTH: usize = u32::MAX as usize; // 4 294 967 295

/// Safely convert an `f64` to a `usize` suitable for memory allocation sizes
/// (array lengths, buffer sizes, repeat counts, etc.).
///
/// Returns `0` for NaN, negative values, and infinities.  Clamps values above
/// [`MAX_ALLOCATION_LENGTH`] to that limit.
///
/// Use this for **index-like** conversions where out-of-range values should
/// silently clamp (e.g. `charAt`, `substring`, `slice`).  For constructors
/// that must reject bad lengths (e.g. `ArrayBuffer`, `TypedArray`), prefer
/// [`checked_f64_to_length`].
pub(crate) fn clamped_f64_to_usize(n: f64) -> usize {
    if n.is_nan() || n < 0.0 || n.is_infinite() {
        return 0;
    }
    // n is finite and non-negative
    let clamped = n.min(MAX_ALLOCATION_LENGTH as f64);
    clamped as usize
}

/// Convert an `f64` to a `usize` for allocation, returning a `RangeError` if
/// the value is not a valid non-negative integer within
/// [`MAX_ALLOCATION_LENGTH`].
///
/// Use this for constructors and APIs that **must reject** bad lengths
/// (e.g. `new ArrayBuffer(n)`, `new TypedArray(n)`).
pub(crate) fn checked_f64_to_length(n: f64) -> StatorResult<usize> {
    if n.is_nan() || n.is_infinite() || n < 0.0 {
        return Err(StatorError::RangeError(
            "Invalid array/buffer length".to_string(),
        ));
    }
    let truncated = n.floor();
    if truncated > MAX_ALLOCATION_LENGTH as f64 {
        return Err(StatorError::RangeError(
            "Invalid array/buffer length".to_string(),
        ));
    }
    Ok(truncated as usize)
}

// ── SameValueZero ──────────────────────────────────────────────────────────────

/// ECMAScript §7.2.11 `SameValueZero(x, y)`.
///
/// Used by [`Map`][super::map] and [`Set`][super::set] for key/value equality.
/// Identical to `===` except:
/// * `NaN` is considered equal to `NaN`.
/// * `-0` is considered equal to `+0`.
///
/// Delegates to [`JsValue::same_value_zero`].
///
/// # Examples
///
/// ```
/// use stator_core::objects::value::JsValue;
///
/// // NaN equals itself under SameValueZero.
/// // (Tested indirectly through Map/Set APIs.)
/// let a = JsValue::HeapNumber(f64::NAN);
/// let b = JsValue::HeapNumber(f64::NAN);
/// // a != b under PartialEq but equal under SameValueZero.
/// assert!(a != b); // PartialEq for f64 NaN
/// ```
pub(crate) fn same_value_zero(a: &JsValue, b: &JsValue) -> bool {
    a.same_value_zero(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_value_zero_nan_equals_nan() {
        assert!(same_value_zero(
            &JsValue::HeapNumber(f64::NAN),
            &JsValue::HeapNumber(f64::NAN)
        ));
    }

    #[test]
    fn test_same_value_zero_negative_zero_equals_positive_zero() {
        assert!(same_value_zero(
            &JsValue::HeapNumber(0.0_f64),
            &JsValue::HeapNumber(-0.0_f64)
        ));
    }

    #[test]
    fn test_same_value_zero_distinct_numbers() {
        assert!(!same_value_zero(
            &JsValue::HeapNumber(1.0),
            &JsValue::HeapNumber(2.0)
        ));
    }

    #[test]
    fn test_same_value_zero_primitives() {
        assert!(same_value_zero(&JsValue::Smi(42), &JsValue::Smi(42)));
        assert!(!same_value_zero(&JsValue::Smi(1), &JsValue::Smi(2)));
        assert!(same_value_zero(&JsValue::Null, &JsValue::Null));
        assert!(!same_value_zero(&JsValue::Null, &JsValue::Undefined));
    }

    // ── clamped_f64_to_usize ──────────────────────────────────────────────

    #[test]
    fn test_clamped_f64_to_usize_normal() {
        assert_eq!(clamped_f64_to_usize(0.0), 0);
        assert_eq!(clamped_f64_to_usize(42.9), 42);
        assert_eq!(clamped_f64_to_usize(100.0), 100);
    }

    #[test]
    fn test_clamped_f64_to_usize_edge_cases() {
        assert_eq!(clamped_f64_to_usize(f64::NAN), 0);
        assert_eq!(clamped_f64_to_usize(f64::INFINITY), 0);
        assert_eq!(clamped_f64_to_usize(f64::NEG_INFINITY), 0);
        assert_eq!(clamped_f64_to_usize(-1.0), 0);
    }

    #[test]
    fn test_clamped_f64_to_usize_huge_value() {
        // This is the exact crash value from the bug report.
        assert_eq!(
            clamped_f64_to_usize(7_881_299_347_898_368.0),
            MAX_ALLOCATION_LENGTH,
        );
    }

    // ── checked_f64_to_length ─────────────────────────────────────────────

    #[test]
    fn test_checked_f64_to_length_normal() {
        assert_eq!(checked_f64_to_length(0.0).unwrap(), 0);
        assert_eq!(checked_f64_to_length(10.0).unwrap(), 10);
    }

    #[test]
    fn test_checked_f64_to_length_rejects_bad_values() {
        assert!(checked_f64_to_length(f64::NAN).is_err());
        assert!(checked_f64_to_length(f64::INFINITY).is_err());
        assert!(checked_f64_to_length(-1.0).is_err());
        assert!(checked_f64_to_length(5_000_000_000_000.0).is_err());
    }
}
