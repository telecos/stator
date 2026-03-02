//! ECMAScript §21.3 `Math` built-in static methods.
//!
//! Every function in this module is a direct Rust equivalent of a static
//! method on the JavaScript `Math` object.  They operate on plain `f64`
//! values and have no side-effects beyond the values passed in.
//!
//! # Naming convention
//!
//! Each function is prefixed `math_` to avoid ambiguity with similarly-named
//! standard-library items (e.g. `math_abs` vs `f64::abs`).
//!
//! # References
//!
//! * ECMAScript 2025 Language Specification §21.3 — *The Math Object*

// ── Constants (ECMAScript §21.3.1) ────────────────────────────────────────────

/// `Math.E` — Euler's number.
pub const MATH_E: f64 = std::f64::consts::E;

/// `Math.LN2` — natural logarithm of 2.
pub const MATH_LN2: f64 = std::f64::consts::LN_2;

/// `Math.LN10` — natural logarithm of 10.
pub const MATH_LN10: f64 = std::f64::consts::LN_10;

/// `Math.LOG2E` — base-2 logarithm of E.
pub const MATH_LOG2E: f64 = std::f64::consts::LOG2_E;

/// `Math.LOG10E` — base-10 logarithm of E.
pub const MATH_LOG10E: f64 = std::f64::consts::LOG10_E;

/// `Math.PI` — the ratio of a circle's circumference to its diameter.
pub const MATH_PI: f64 = std::f64::consts::PI;

/// `Math.SQRT1_2` — square root of 1/2.
pub const MATH_SQRT1_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// `Math.SQRT2` — square root of 2.
pub const MATH_SQRT2: f64 = std::f64::consts::SQRT_2;

// ── Math.abs ──────────────────────────────────────────────────────────────────

/// ECMAScript §21.3.2.1 `Math.abs(x)`.
///
/// Returns the absolute value of `x`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_abs;
///
/// assert_eq!(math_abs(-5.0), 5.0);
/// assert_eq!(math_abs(3.0), 3.0);
/// assert!(math_abs(f64::NAN).is_nan());
/// ```
pub fn math_abs(x: f64) -> f64 {
    x.abs()
}

// ── Math.ceil ─────────────────────────────────────────────────────────────────

/// ECMAScript §21.3.2.6 `Math.ceil(x)`.
///
/// Returns the smallest integer greater than or equal to `x`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_ceil;
///
/// assert_eq!(math_ceil(1.2), 2.0);
/// assert_eq!(math_ceil(-1.2), -1.0);
/// assert_eq!(math_ceil(2.0), 2.0);
/// ```
pub fn math_ceil(x: f64) -> f64 {
    x.ceil()
}

// ── Math.floor ────────────────────────────────────────────────────────────────

/// ECMAScript §21.3.2.16 `Math.floor(x)`.
///
/// Returns the largest integer less than or equal to `x`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_floor;
///
/// assert_eq!(math_floor(1.9), 1.0);
/// assert_eq!(math_floor(-1.1), -2.0);
/// assert_eq!(math_floor(2.0), 2.0);
/// ```
pub fn math_floor(x: f64) -> f64 {
    x.floor()
}

// ── Math.round ────────────────────────────────────────────────────────────────

/// ECMAScript §21.3.2.28 `Math.round(x)`.
///
/// Returns the integer nearest to `x`, rounding half-way cases toward
/// `+Infinity` (i.e. `0.5` rounds to `1`, `-0.5` rounds to `0`).
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_round;
///
/// assert_eq!(math_round(1.5), 2.0);
/// assert_eq!(math_round(-1.5), -1.0);
/// assert_eq!(math_round(1.4), 1.0);
/// ```
pub fn math_round(x: f64) -> f64 {
    // ECMAScript §21.3.2.28: ties round toward +Infinity, equivalent to
    // floor(x + 0.5).
    (x + 0.5).floor()
}

// ── Math.trunc ────────────────────────────────────────────────────────────────

/// ECMAScript §21.3.2.35 `Math.trunc(x)`.
///
/// Returns the integer part of `x` by removing any fractional digits.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_trunc;
///
/// assert_eq!(math_trunc(1.9), 1.0);
/// assert_eq!(math_trunc(-1.9), -1.0);
/// assert_eq!(math_trunc(0.5), 0.0);
/// ```
pub fn math_trunc(x: f64) -> f64 {
    x.trunc()
}

// ── Math.sign ─────────────────────────────────────────────────────────────────

/// ECMAScript §21.3.2.29 `Math.sign(x)`.
///
/// Returns `1.0` if `x > 0`, `-1.0` if `x < 0`, `+0.0` if `x` is `+0`,
/// `-0.0` if `x` is `-0`, and `NaN` if `x` is `NaN`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_sign;
///
/// assert_eq!(math_sign(5.0), 1.0);
/// assert_eq!(math_sign(-3.0), -1.0);
/// assert_eq!(math_sign(0.0), 0.0);
/// assert!(math_sign(f64::NAN).is_nan());
/// ```
pub fn math_sign(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        // Preserve the sign of zero.
        x
    }
}

// ── Math.max / Math.min ───────────────────────────────────────────────────────

/// ECMAScript §21.3.2.24 `Math.max(...values)`.
///
/// Returns the largest of the supplied values.  Returns `-Infinity` for an
/// empty slice and `NaN` if any value is `NaN`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_max;
///
/// assert_eq!(math_max(&[1.0, 3.0, 2.0]), 3.0);
/// assert_eq!(math_max(&[]), f64::NEG_INFINITY);
/// assert!(math_max(&[1.0, f64::NAN]).is_nan());
/// ```
pub fn math_max(values: &[f64]) -> f64 {
    let mut result = f64::NEG_INFINITY;
    for &v in values {
        if v.is_nan() {
            return f64::NAN;
        }
        if v > result {
            result = v;
        }
    }
    result
}

/// ECMAScript §21.3.2.25 `Math.min(...values)`.
///
/// Returns the smallest of the supplied values.  Returns `+Infinity` for an
/// empty slice and `NaN` if any value is `NaN`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_min;
///
/// assert_eq!(math_min(&[1.0, 3.0, 2.0]), 1.0);
/// assert_eq!(math_min(&[]), f64::INFINITY);
/// assert!(math_min(&[1.0, f64::NAN]).is_nan());
/// ```
pub fn math_min(values: &[f64]) -> f64 {
    let mut result = f64::INFINITY;
    for &v in values {
        if v.is_nan() {
            return f64::NAN;
        }
        if v < result {
            result = v;
        }
    }
    result
}

// ── Math.pow ──────────────────────────────────────────────────────────────────

/// ECMAScript §21.3.2.26 `Math.pow(base, exponent)`.
///
/// Returns `base` raised to the power `exponent`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_pow;
///
/// assert_eq!(math_pow(2.0, 10.0), 1024.0);
/// assert_eq!(math_pow(4.0, 0.5), 2.0);
/// ```
pub fn math_pow(base: f64, exponent: f64) -> f64 {
    base.powf(exponent)
}

// ── Math.sqrt ─────────────────────────────────────────────────────────────────

/// ECMAScript §21.3.2.32 `Math.sqrt(x)`.
///
/// Returns the square root of `x`.  Returns `NaN` for negative inputs.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_sqrt;
///
/// assert_eq!(math_sqrt(4.0), 2.0);
/// assert!(math_sqrt(-1.0).is_nan());
/// ```
pub fn math_sqrt(x: f64) -> f64 {
    x.sqrt()
}

// ── Math.cbrt ─────────────────────────────────────────────────────────────────

/// ECMAScript §21.3.2.5 `Math.cbrt(x)`.
///
/// Returns the cube root of `x`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_cbrt;
///
/// assert_eq!(math_cbrt(27.0), 3.0);
/// assert_eq!(math_cbrt(-8.0), -2.0);
/// ```
pub fn math_cbrt(x: f64) -> f64 {
    x.cbrt()
}

// ── Math.hypot ────────────────────────────────────────────────────────────────

/// ECMAScript §21.3.2.18 `Math.hypot(...values)`.
///
/// Returns the square root of the sum of squares of the supplied values.
/// Returns `0` for an empty slice and `NaN` if any value is `NaN`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_hypot;
///
/// assert_eq!(math_hypot(&[3.0, 4.0]), 5.0);
/// assert_eq!(math_hypot(&[]), 0.0);
/// ```
pub fn math_hypot(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    // Check for any infinities first (per ECMAScript spec).
    for &v in values {
        if v.is_infinite() {
            return f64::INFINITY;
        }
    }
    for &v in values {
        if v.is_nan() {
            return f64::NAN;
        }
    }
    let sum_sq: f64 = values.iter().map(|v| v * v).sum();
    sum_sq.sqrt()
}

// ── Math.log / Math.log2 / Math.log10 ────────────────────────────────────────

/// ECMAScript §21.3.2.20 `Math.log(x)`.
///
/// Returns the natural logarithm (base-*e*) of `x`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_log;
///
/// assert!((math_log(std::f64::consts::E) - 1.0).abs() < 1e-15);
/// assert!(math_log(-1.0).is_nan());
/// assert_eq!(math_log(0.0), f64::NEG_INFINITY);
/// ```
pub fn math_log(x: f64) -> f64 {
    x.ln()
}

/// ECMAScript §21.3.2.22 `Math.log2(x)`.
///
/// Returns the base-2 logarithm of `x`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_log2;
///
/// assert_eq!(math_log2(8.0), 3.0);
/// assert!(math_log2(-1.0).is_nan());
/// ```
pub fn math_log2(x: f64) -> f64 {
    x.log2()
}

/// ECMAScript §21.3.2.21 `Math.log10(x)`.
///
/// Returns the base-10 logarithm of `x`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_log10;
///
/// assert_eq!(math_log10(1000.0), 3.0);
/// assert!(math_log10(-1.0).is_nan());
/// ```
pub fn math_log10(x: f64) -> f64 {
    x.log10()
}

// ── Math.sin / Math.cos / Math.tan ────────────────────────────────────────────

/// ECMAScript §21.3.2.30 `Math.sin(x)`.
///
/// Returns the sine of `x` (in radians).
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_sin;
///
/// assert!((math_sin(0.0) - 0.0).abs() < 1e-15);
/// ```
pub fn math_sin(x: f64) -> f64 {
    x.sin()
}

/// ECMAScript §21.3.2.7 `Math.cos(x)`.
///
/// Returns the cosine of `x` (in radians).
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_cos;
///
/// assert!((math_cos(0.0) - 1.0).abs() < 1e-15);
/// ```
pub fn math_cos(x: f64) -> f64 {
    x.cos()
}

/// ECMAScript §21.3.2.33 `Math.tan(x)`.
///
/// Returns the tangent of `x` (in radians).
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_tan;
///
/// assert!((math_tan(0.0) - 0.0).abs() < 1e-15);
/// ```
pub fn math_tan(x: f64) -> f64 {
    x.tan()
}

// ── Math.asin / Math.acos / Math.atan / Math.atan2 ───────────────────────────

/// ECMAScript §21.3.2.3 `Math.asin(x)`.
///
/// Returns the arcsine of `x` (in radians).
pub fn math_asin(x: f64) -> f64 {
    x.asin()
}

/// ECMAScript §21.3.2.2 `Math.acos(x)`.
///
/// Returns the arccosine of `x` (in radians).
pub fn math_acos(x: f64) -> f64 {
    x.acos()
}

/// ECMAScript §21.3.2.4 `Math.atan(x)`.
///
/// Returns the arctangent of `x` (in radians).
pub fn math_atan(x: f64) -> f64 {
    x.atan()
}

/// ECMAScript §21.3.2.8 `Math.atan2(y, x)`.
///
/// Returns the angle in radians between the positive x-axis and the point
/// `(x, y)`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_atan2;
///
/// assert!((math_atan2(1.0, 1.0) - std::f64::consts::FRAC_PI_4).abs() < 1e-15);
/// ```
pub fn math_atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

// ── Math.random ───────────────────────────────────────────────────────────────

/// ECMAScript §21.3.2.27 `Math.random()`.
///
/// Returns a pseudo-random `f64` in `[0, 1)`.
///
/// This implementation uses the [xorshift64] algorithm seeded from the current
/// system time (nanoseconds since UNIX epoch) so that each call sequence
/// produces different values.  It is **not** cryptographically secure.
///
/// [xorshift64]: https://en.wikipedia.org/wiki/Xorshift
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_random;
///
/// let v = math_random();
/// assert!((0.0..1.0).contains(&v));
/// ```
pub fn math_random() -> f64 {
    use std::cell::Cell;
    use std::time::{SystemTime, UNIX_EPOCH};

    thread_local! {
        static STATE: Cell<u64> = Cell::new({
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.subsec_nanos() as u64 ^ (d.as_secs().wrapping_mul(6_364_136_223_846_793_005)))
                .unwrap_or(12_345_678_901_234_567)
                | 1 // ensure non-zero
        });
    }

    STATE.with(|s| {
        let mut x = s.get();
        // xorshift64
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        // Map to [0, 1) by taking 53 bits and dividing by 2^53.
        (x >> 11) as f64 / (1u64 << 53) as f64
    })
}

// ── Math.clz32 ────────────────────────────────────────────────────────────────

/// ECMAScript §21.3.2.9 `Math.clz32(x)`.
///
/// Converts `x` to a 32-bit unsigned integer and returns the number of
/// leading zero bits.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_clz32;
///
/// assert_eq!(math_clz32(1.0), 31);
/// assert_eq!(math_clz32(1000.0), 22);
/// assert_eq!(math_clz32(0.0), 32);
/// ```
pub fn math_clz32(x: f64) -> u32 {
    let n = x as i64 as i32 as u32;
    n.leading_zeros()
}

// ── Math.imul ─────────────────────────────────────────────────────────────────

/// ECMAScript §21.3.2.18 `Math.imul(a, b)`.
///
/// Returns the result of the C-like 32-bit integer multiplication of `a` and
/// `b`.  Both operands are converted to `i32` before multiplication.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_imul;
///
/// assert_eq!(math_imul(3.0, 4.0), 12);
/// assert_eq!(math_imul(-5.0, 3.0), -15);
/// // Overflow wraps around.
/// assert_eq!(math_imul(0xffffffff_u32 as f64, 5.0), -5);
/// ```
pub fn math_imul(a: f64, b: f64) -> i32 {
    let a = a as i64 as i32;
    let b = b as i64 as i32;
    a.wrapping_mul(b)
}

// ── Math.fround ───────────────────────────────────────────────────────────────

/// ECMAScript §21.3.2.17 `Math.fround(x)`.
///
/// Returns the nearest single-precision (`f32`) floating-point representation
/// of `x`, converted back to `f64`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::math::math_fround;
///
/// assert_eq!(math_fround(1.337), 1.337_f32 as f64);
/// assert!(math_fround(f64::NAN).is_nan());
/// ```
pub fn math_fround(x: f64) -> f64 {
    (x as f32) as f64
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Constants ────────────────────────────────────────────────────────────

    #[test]
    fn test_math_pi_approx() {
        assert!((MATH_PI - 3.141_592_653_589_793).abs() < 1e-15);
    }

    #[test]
    fn test_math_e_approx() {
        assert!((MATH_E - 2.718_281_828_459_045).abs() < 1e-15);
    }

    // ── math_abs ─────────────────────────────────────────────────────────────

    #[test]
    fn test_abs_positive() {
        assert_eq!(math_abs(3.5), 3.5);
    }

    #[test]
    fn test_abs_negative() {
        assert_eq!(math_abs(-3.5), 3.5);
    }

    #[test]
    fn test_abs_zero() {
        assert_eq!(math_abs(0.0), 0.0);
        assert_eq!(math_abs(-0.0), 0.0);
    }

    #[test]
    fn test_abs_nan() {
        assert!(math_abs(f64::NAN).is_nan());
    }

    #[test]
    fn test_abs_infinity() {
        assert_eq!(math_abs(f64::INFINITY), f64::INFINITY);
        assert_eq!(math_abs(f64::NEG_INFINITY), f64::INFINITY);
    }

    // ── math_ceil ────────────────────────────────────────────────────────────

    #[test]
    fn test_ceil_positive_fraction() {
        assert_eq!(math_ceil(1.2), 2.0);
    }

    #[test]
    fn test_ceil_negative_fraction() {
        assert_eq!(math_ceil(-1.2), -1.0);
    }

    #[test]
    fn test_ceil_whole() {
        assert_eq!(math_ceil(3.0), 3.0);
    }

    // ── math_floor ───────────────────────────────────────────────────────────

    #[test]
    fn test_floor_positive_fraction() {
        assert_eq!(math_floor(1.9), 1.0);
    }

    #[test]
    fn test_floor_negative_fraction() {
        assert_eq!(math_floor(-1.1), -2.0);
    }

    #[test]
    fn test_floor_whole() {
        assert_eq!(math_floor(2.0), 2.0);
    }

    // ── math_round ───────────────────────────────────────────────────────────

    #[test]
    fn test_round_half_up() {
        assert_eq!(math_round(1.5), 2.0);
        assert_eq!(math_round(2.5), 3.0);
    }

    #[test]
    fn test_round_negative_half() {
        // ECMAScript: -0.5 rounds to -0 (i.e. 0), -1.5 rounds to -1.
        assert_eq!(math_round(-1.5), -1.0);
        assert_eq!(math_round(-0.5), 0.0);
    }

    #[test]
    fn test_round_below_half() {
        assert_eq!(math_round(1.4), 1.0);
        assert_eq!(math_round(-1.4), -1.0);
    }

    // ── math_trunc ───────────────────────────────────────────────────────────

    #[test]
    fn test_trunc_positive() {
        assert_eq!(math_trunc(1.9), 1.0);
    }

    #[test]
    fn test_trunc_negative() {
        assert_eq!(math_trunc(-1.9), -1.0);
    }

    #[test]
    fn test_trunc_zero() {
        assert_eq!(math_trunc(0.5), 0.0);
        assert_eq!(math_trunc(-0.5), -0.0);
    }

    // ── math_sign ────────────────────────────────────────────────────────────

    #[test]
    fn test_sign_positive() {
        assert_eq!(math_sign(5.0), 1.0);
    }

    #[test]
    fn test_sign_negative() {
        assert_eq!(math_sign(-3.0), -1.0);
    }

    #[test]
    fn test_sign_zero() {
        assert_eq!(math_sign(0.0), 0.0);
        // Negative zero should preserve sign.
        assert!(math_sign(-0.0_f64).is_sign_negative());
    }

    #[test]
    fn test_sign_nan() {
        assert!(math_sign(f64::NAN).is_nan());
    }

    // ── math_max / math_min ──────────────────────────────────────────────────

    #[test]
    fn test_max_basic() {
        assert_eq!(math_max(&[1.0, 3.0, 2.0]), 3.0);
    }

    #[test]
    fn test_max_empty() {
        assert_eq!(math_max(&[]), f64::NEG_INFINITY);
    }

    #[test]
    fn test_max_nan_propagates() {
        assert!(math_max(&[1.0, f64::NAN, 3.0]).is_nan());
    }

    #[test]
    fn test_min_basic() {
        assert_eq!(math_min(&[1.0, 3.0, 2.0]), 1.0);
    }

    #[test]
    fn test_min_empty() {
        assert_eq!(math_min(&[]), f64::INFINITY);
    }

    #[test]
    fn test_min_nan_propagates() {
        assert!(math_min(&[1.0, f64::NAN, 3.0]).is_nan());
    }

    // ── math_pow ─────────────────────────────────────────────────────────────

    #[test]
    fn test_pow_integer() {
        assert_eq!(math_pow(2.0, 10.0), 1024.0);
    }

    #[test]
    fn test_pow_fractional_exponent() {
        assert!((math_pow(4.0, 0.5) - 2.0).abs() < 1e-15);
    }

    #[test]
    fn test_pow_zero_exponent() {
        assert_eq!(math_pow(5.0, 0.0), 1.0);
        assert_eq!(math_pow(0.0, 0.0), 1.0);
    }

    #[test]
    fn test_pow_nan_base() {
        assert!(math_pow(f64::NAN, 2.0).is_nan());
    }

    // ── math_sqrt ────────────────────────────────────────────────────────────

    #[test]
    fn test_sqrt_perfect_square() {
        assert_eq!(math_sqrt(4.0), 2.0);
        assert_eq!(math_sqrt(9.0), 3.0);
    }

    #[test]
    fn test_sqrt_negative() {
        assert!(math_sqrt(-1.0).is_nan());
    }

    #[test]
    fn test_sqrt_zero() {
        assert_eq!(math_sqrt(0.0), 0.0);
    }

    // ── math_cbrt ────────────────────────────────────────────────────────────

    #[test]
    fn test_cbrt_positive() {
        assert!((math_cbrt(27.0) - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_cbrt_negative() {
        assert!((math_cbrt(-8.0) - (-2.0)).abs() < 1e-14);
    }

    // ── math_hypot ───────────────────────────────────────────────────────────

    #[test]
    fn test_hypot_3_4_5() {
        assert!((math_hypot(&[3.0, 4.0]) - 5.0).abs() < 1e-14);
    }

    #[test]
    fn test_hypot_empty() {
        assert_eq!(math_hypot(&[]), 0.0);
    }

    #[test]
    fn test_hypot_infinity() {
        assert_eq!(math_hypot(&[f64::INFINITY, 0.0]), f64::INFINITY);
    }

    #[test]
    fn test_hypot_nan() {
        assert!(math_hypot(&[f64::NAN, 1.0]).is_nan());
    }

    // ── math_log / math_log2 / math_log10 ────────────────────────────────────

    #[test]
    fn test_log_e() {
        assert!((math_log(std::f64::consts::E) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_log_zero() {
        assert_eq!(math_log(0.0), f64::NEG_INFINITY);
    }

    #[test]
    fn test_log_negative() {
        assert!(math_log(-1.0).is_nan());
    }

    #[test]
    fn test_log2_power_of_two() {
        assert_eq!(math_log2(8.0), 3.0);
    }

    #[test]
    fn test_log10_power_of_ten() {
        assert!((math_log10(1000.0) - 3.0).abs() < 1e-14);
    }

    // ── math_sin / math_cos / math_tan ───────────────────────────────────────

    #[test]
    fn test_sin_zero() {
        assert!((math_sin(0.0)).abs() < 1e-15);
    }

    #[test]
    fn test_sin_half_pi() {
        assert!((math_sin(std::f64::consts::FRAC_PI_2) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_cos_zero() {
        assert!((math_cos(0.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_cos_pi() {
        assert!((math_cos(std::f64::consts::PI) + 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_tan_zero() {
        assert!(math_tan(0.0).abs() < 1e-15);
    }

    // ── math_random ──────────────────────────────────────────────────────────

    #[test]
    fn test_random_in_range() {
        for _ in 0..100 {
            let v = math_random();
            assert!((0.0..1.0).contains(&v), "random value {v} not in [0, 1)");
        }
    }

    #[test]
    fn test_random_produces_different_values() {
        let a = math_random();
        let b = math_random();
        // With a 53-bit mantissa the probability of two identical values is
        // ~1 in 2^53 — this test is practically always true.
        assert_ne!(a, b);
    }

    // ── math_clz32 ───────────────────────────────────────────────────────────

    #[test]
    fn test_clz32_one() {
        assert_eq!(math_clz32(1.0), 31);
    }

    #[test]
    fn test_clz32_zero() {
        assert_eq!(math_clz32(0.0), 32);
    }

    #[test]
    fn test_clz32_large() {
        assert_eq!(math_clz32(1000.0), 22);
    }

    // ── math_imul ────────────────────────────────────────────────────────────

    #[test]
    fn test_imul_basic() {
        assert_eq!(math_imul(3.0, 4.0), 12);
    }

    #[test]
    fn test_imul_negative() {
        assert_eq!(math_imul(-5.0, 3.0), -15);
    }

    #[test]
    fn test_imul_overflow_wraps() {
        // 0xffffffff * 5 wraps in 32-bit arithmetic.
        assert_eq!(math_imul(0xffff_ffff_u32 as f64, 5.0), -5);
    }

    // ── math_fround ──────────────────────────────────────────────────────────

    #[test]
    fn test_fround_identity_for_f32_exact() {
        assert_eq!(math_fround(1.0), 1.0);
    }

    #[test]
    fn test_fround_reduces_precision() {
        let exact = 1.337_f32 as f64;
        assert_eq!(math_fround(1.337), exact);
    }

    #[test]
    fn test_fround_nan() {
        assert!(math_fround(f64::NAN).is_nan());
    }

    // ── IEEE 754 edge cases ───────────────────────────────────────────────────

    #[test]
    fn test_ieee754_abs_negative_zero() {
        // Per IEEE 754, abs(-0) == +0.
        assert!(math_abs(-0.0_f64).is_sign_positive());
    }

    #[test]
    fn test_ieee754_sqrt_infinity() {
        assert_eq!(math_sqrt(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn test_ieee754_pow_neg1_infinity() {
        // ECMAScript: (-1)^±Infinity == 1.
        assert_eq!(math_pow(-1.0, f64::INFINITY), 1.0);
        assert_eq!(math_pow(-1.0, f64::NEG_INFINITY), 1.0);
    }

    #[test]
    fn test_ieee754_log_one() {
        assert_eq!(math_log(1.0), 0.0);
    }

    #[test]
    fn test_ieee754_hypot_inf_over_nan() {
        // Per ECMAScript, if any value is Infinity, result is Infinity even
        // if there is a NaN.
        assert_eq!(math_hypot(&[f64::INFINITY, f64::NAN]), f64::INFINITY);
    }
}
