//! Numeric literal conformance tests.
//!
//! Covers numeric separators, BigInt literals, hex/octal/binary parsing,
//! scientific notation, special values (`Infinity`, `NaN`), `Number()`
//! conversions, and BigInt arithmetic / typeof semantics.

#[cfg(test)]
mod tests {
    use crate::builtins::global::global_eval;
    use crate::objects::value::JsValue;

    /// Helper: evaluate `src` and assert the result is boolean `true`.
    fn assert_eval_true(src: &str) {
        let result = global_eval(src).unwrap();
        assert_eq!(result, JsValue::Boolean(true), "expected true for: {src}");
    }

    /// Helper: evaluate `src` and assert it produces an error.
    fn assert_eval_err(src: &str) {
        assert!(global_eval(src).is_err(), "expected error for: {src}");
    }

    // ── 1. Numeric separators ────────────────────────────────────────────────

    #[test]
    fn e2e_numeric_sep_decimal() {
        assert_eval_true("1_000_000 === 1000000");
    }

    #[test]
    fn e2e_numeric_sep_hex() {
        assert_eval_true("0xFF_FF === 0xFFFF");
    }

    #[test]
    fn e2e_numeric_sep_binary() {
        assert_eval_true("0b1010_0001 === 0b10100001");
    }

    #[test]
    fn e2e_numeric_sep_octal() {
        assert_eval_true("0o77_77 === 0o7777");
    }

    #[test]
    fn e2e_numeric_sep_float() {
        assert_eval_true("1_000.000_1 === 1000.0001");
    }

    #[test]
    fn e2e_numeric_sep_exponent() {
        assert_eval_true("1_0e1_0 === 1e11");
    }

    #[test]
    fn e2e_numeric_sep_bigint() {
        assert_eval_true("1_000n === 1000n");
    }

    #[test]
    fn e2e_numeric_sep_invalid_double_underscore() {
        assert_eval_err("1__0");
    }

    #[test]
    fn e2e_numeric_sep_invalid_trailing() {
        assert_eval_err("100_");
    }

    #[test]
    fn e2e_numeric_sep_invalid_leading_after_zero() {
        assert_eval_err("0_1");
    }

    // ── 2. BigInt literals ───────────────────────────────────────────────────

    #[test]
    fn e2e_bigint_literal_basic() {
        assert_eval_true("123n === 123n");
    }

    #[test]
    fn e2e_bigint_zero() {
        assert_eval_true("0n === 0n");
    }

    #[test]
    fn e2e_bigint_typeof() {
        let result = global_eval("typeof 1n").unwrap();
        assert_eq!(result, JsValue::String("bigint".into()));
    }

    #[test]
    fn e2e_bigint_typeof_comparison() {
        assert_eval_true("typeof 1n === 'bigint'");
    }

    #[test]
    fn e2e_bigint_add() {
        assert_eval_true("10n + 20n === 30n");
    }

    #[test]
    fn e2e_bigint_sub() {
        assert_eval_true("100n - 42n === 58n");
    }

    #[test]
    fn e2e_bigint_mul() {
        assert_eval_true("7n * 6n === 42n");
    }

    #[test]
    fn e2e_bigint_div() {
        assert_eval_true("100n / 3n === 33n");
    }

    #[test]
    fn e2e_bigint_mixed_number_add_throws() {
        assert_eval_err("1n + 1");
    }

    #[test]
    fn e2e_bigint_mixed_number_sub_throws() {
        assert_eval_err("1n - 1");
    }

    #[test]
    fn e2e_bigint_mixed_number_mul_throws() {
        assert_eval_err("1n * 2");
    }

    #[test]
    fn e2e_bigint_constructor_from_string() {
        assert_eval_true("BigInt('123') === 123n");
    }

    #[test]
    fn e2e_bigint_constructor_from_number() {
        assert_eval_true("BigInt(42) === 42n");
    }

    #[test]
    fn e2e_bigint_constructor_from_boolean_true() {
        assert_eval_true("BigInt(true) === 1n");
    }

    #[test]
    fn e2e_bigint_constructor_from_boolean_false() {
        assert_eval_true("BigInt(false) === 0n");
    }

    #[test]
    fn e2e_bigint_hex_literal() {
        assert_eval_true("0xFFn === 255n");
    }

    #[test]
    fn e2e_bigint_octal_literal() {
        assert_eval_true("0o77n === 63n");
    }

    #[test]
    fn e2e_bigint_binary_literal() {
        assert_eval_true("0b1010n === 10n");
    }

    // ── 3. Hex / Octal / Binary literals ─────────────────────────────────────

    #[test]
    fn e2e_hex_literal() {
        assert_eval_true("0xFF === 255");
    }

    #[test]
    fn e2e_hex_literal_uppercase() {
        assert_eval_true("0XFF === 255");
    }

    #[test]
    fn e2e_octal_literal() {
        assert_eval_true("0o77 === 63");
    }

    #[test]
    fn e2e_octal_literal_uppercase() {
        assert_eval_true("0O77 === 63");
    }

    #[test]
    fn e2e_binary_literal() {
        assert_eval_true("0b1010 === 10");
    }

    #[test]
    fn e2e_binary_literal_uppercase() {
        assert_eval_true("0B1111 === 15");
    }

    // ── 4. Scientific notation / exponent ────────────────────────────────────

    #[test]
    fn e2e_exponent_lowercase() {
        assert_eval_true("1e10 === 10000000000");
    }

    #[test]
    fn e2e_exponent_uppercase() {
        assert_eval_true("1E10 === 10000000000");
    }

    #[test]
    fn e2e_exponent_positive_sign() {
        assert_eval_true("1e+10 === 10000000000");
    }

    #[test]
    fn e2e_exponent_negative() {
        assert_eval_true("1e-3 === 0.001");
    }

    #[test]
    fn e2e_exponent_fractional_base() {
        assert_eval_true("1.5e2 === 150");
    }

    // ── 5. Special values ────────────────────────────────────────────────────

    #[test]
    fn e2e_global_infinity() {
        let result = global_eval("Infinity").unwrap();
        assert_eq!(result, JsValue::HeapNumber(f64::INFINITY));
    }

    #[test]
    fn e2e_global_neg_infinity() {
        assert_eval_true("-Infinity === -Infinity");
    }

    #[test]
    fn e2e_global_nan_is_nan() {
        // NaN !== NaN per IEEE 754
        assert_eval_true("NaN !== NaN");
    }

    #[test]
    fn e2e_number_max_safe_integer() {
        assert_eval_true("Number.MAX_SAFE_INTEGER === 9007199254740991");
    }

    #[test]
    fn e2e_number_min_safe_integer() {
        assert_eval_true("Number.MIN_SAFE_INTEGER === -9007199254740991");
    }

    #[test]
    fn e2e_number_is_nan_true() {
        assert_eval_true("Number.isNaN(NaN) === true");
    }

    #[test]
    fn e2e_number_is_finite_false_for_infinity() {
        assert_eval_true("Number.isFinite(Infinity) === false");
    }

    // ── 6. Number() conversions ──────────────────────────────────────────────

    #[test]
    fn e2e_number_from_hex_string() {
        assert_eval_true("Number('0x10') === 16");
    }

    #[test]
    fn e2e_number_from_binary_string() {
        assert_eval_true("Number('0b10') === 2");
    }

    #[test]
    fn e2e_number_from_octal_string() {
        assert_eval_true("Number('0o10') === 8");
    }

    #[test]
    fn e2e_number_from_empty_string() {
        assert_eval_true("Number('') === 0");
    }

    #[test]
    fn e2e_number_from_whitespace_string() {
        assert_eval_true("Number('   ') === 0");
    }

    #[test]
    fn e2e_number_from_null() {
        assert_eval_true("Number(null) === 0");
    }

    #[test]
    fn e2e_number_from_undefined_is_nan() {
        assert_eval_true("Number.isNaN(Number(undefined))");
    }

    #[test]
    fn e2e_number_from_true() {
        assert_eval_true("Number(true) === 1");
    }

    #[test]
    fn e2e_number_from_false() {
        assert_eval_true("Number(false) === 0");
    }

    #[test]
    fn e2e_number_from_infinity_string() {
        assert_eval_true("Number('Infinity') === Infinity");
    }

    #[test]
    fn e2e_number_from_neg_infinity_string() {
        assert_eval_true("Number('-Infinity') === -Infinity");
    }
}
