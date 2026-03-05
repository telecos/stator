//! Shared utility functions used across the `builtins` sub-modules.

use crate::objects::value::JsValue;

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
}
