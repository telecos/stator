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

    macro_rules! coercion_e2e_test {
        ($name:ident, $script:expr) => {
            #[test]
            fn $name() {
                assert_eval_true($script);
            }
        };
    }

    // ── 7. Coercion and comparison corner cases ───────────────────────────────

    coercion_e2e_test!(
        e2e_to_primitive_number_hint_prefers_value_of,
        "Number({ valueOf() { return 7; }, toString() { return '9'; } }) === 7"
    );
    coercion_e2e_test!(
        e2e_to_primitive_string_hint_prefers_to_string,
        "String({ toString() { return 'text'; }, valueOf() { return 9; } }) === 'text'"
    );
    coercion_e2e_test!(
        e2e_to_primitive_default_hint_acts_like_number,
        "({ valueOf() { return 5; }, toString() { return '9'; } }) + 1 === 6"
    );
    coercion_e2e_test!(
        e2e_to_primitive_number_hint_falls_back_to_to_string,
        "Number({ valueOf() { return {}; }, toString() { return '13'; } }) === 13"
    );
    coercion_e2e_test!(
        e2e_to_primitive_string_hint_falls_back_to_value_of,
        "String({ toString() { return {}; }, valueOf() { return 8; } }) === '8'"
    );
    coercion_e2e_test!(
        e2e_to_primitive_symbol_method_has_priority_for_number,
        "Number({ [Symbol.toPrimitive]() { return 11; }, valueOf() { return 1; } }) === 11"
    );
    coercion_e2e_test!(
        e2e_to_primitive_symbol_method_has_priority_for_string,
        "String({ [Symbol.toPrimitive]() { return 'sym'; }, toString() { return 'no'; } }) === 'sym'"
    );
    coercion_e2e_test!(
        e2e_to_primitive_symbol_method_receives_default_hint,
        "({ [Symbol.toPrimitive](hint) { return hint; } }) + '' === 'default'"
    );
    coercion_e2e_test!(
        e2e_to_primitive_symbol_method_receives_number_hint,
        "Number({ [Symbol.toPrimitive](hint) { return hint === 'number' ? 2 : 0; } }) === 2"
    );
    coercion_e2e_test!(
        e2e_to_primitive_symbol_method_receives_string_hint,
        "String({ [Symbol.toPrimitive](hint) { return hint === 'string' ? 'ok' : 'bad'; } }) === 'ok'"
    );
    coercion_e2e_test!(
        e2e_to_primitive_symbol_method_non_primitive_throws,
        "try { Number({ [Symbol.toPrimitive]() { return {}; } }); false; } catch (e) { e instanceof TypeError; }"
    );
    coercion_e2e_test!(
        e2e_to_primitive_symbol_method_non_callable_throws,
        "try { Number({ [Symbol.toPrimitive]: 1 }); false; } catch (e) { e instanceof TypeError; }"
    );
    coercion_e2e_test!(
        e2e_to_primitive_both_number_methods_non_primitive_throw,
        "try { Number({ valueOf() { return {}; }, toString() { return {}; } }); false; } catch (e) { e instanceof TypeError; }"
    );
    coercion_e2e_test!(
        e2e_to_primitive_both_string_methods_non_primitive_throw,
        "try { String({ toString() { return {}; }, valueOf() { return {}; } }); false; } catch (e) { e instanceof TypeError; }"
    );

    coercion_e2e_test!(e2e_to_number_null_to_zero, "Number(null) === 0");
    coercion_e2e_test!(
        e2e_to_number_undefined_to_nan,
        "Number.isNaN(Number(undefined))"
    );
    coercion_e2e_test!(e2e_to_number_true_to_one, "Number(true) === 1");
    coercion_e2e_test!(e2e_to_number_false_to_zero, "Number(false) === 0");
    coercion_e2e_test!(e2e_to_number_empty_string_to_zero, "Number('') === 0");
    coercion_e2e_test!(
        e2e_to_number_whitespace_string_to_zero,
        "Number('  \\t\\n  ') === 0"
    );
    coercion_e2e_test!(e2e_to_number_hex_string_to_sixteen, "Number('0x10') === 16");
    coercion_e2e_test!(
        e2e_to_number_trimmed_hex_string_to_sixteen,
        "Number('  0x10  ') === 16"
    );
    coercion_e2e_test!(
        e2e_to_number_invalid_string_to_nan,
        "Number.isNaN(Number('not numeric'))"
    );
    coercion_e2e_test!(
        e2e_to_number_object_uses_number_hint,
        "Number({ valueOf() { return '12'; } }) === 12"
    );
    coercion_e2e_test!(
        e2e_to_number_object_falls_back_to_to_string,
        "Number({ valueOf() { return {}; }, toString() { return '34'; } }) === 34"
    );
    coercion_e2e_test!(e2e_to_number_object_empty_array_to_zero, "Number([]) === 0");
    coercion_e2e_test!(e2e_to_number_object_singleton_array, "Number([7]) === 7");
    coercion_e2e_test!(
        e2e_to_number_object_multi_element_array_is_nan,
        "Number.isNaN(Number([1, 2]))"
    );

    coercion_e2e_test!(e2e_to_string_null_literal, "String(null) === 'null'");
    coercion_e2e_test!(
        e2e_to_string_undefined_literal,
        "String(undefined) === 'undefined'"
    );
    coercion_e2e_test!(e2e_to_string_true_literal, "String(true) === 'true'");
    coercion_e2e_test!(
        e2e_to_string_symbol_via_constructor_function,
        "String(Symbol('x')) === 'Symbol(x)'"
    );
    coercion_e2e_test!(
        e2e_to_string_symbol_throws_in_concatenation,
        "try { '' + Symbol('x'); false; } catch (e) { e instanceof TypeError; }"
    );
    coercion_e2e_test!(
        e2e_to_string_object_prefers_to_string,
        "String({ toString() { return 'ok'; }, valueOf() { return 1; } }) === 'ok'"
    );
    coercion_e2e_test!(
        e2e_to_string_object_falls_back_to_value_of,
        "String({ toString() { return {}; }, valueOf() { return 7; } }) === '7'"
    );
    coercion_e2e_test!(
        e2e_to_string_array_joins_nullish_as_empty_segments,
        "String([1, null, undefined, 4]) === '1,,,4'"
    );

    coercion_e2e_test!(e2e_to_boolean_zero_is_false, "!0 && !(-0)");
    coercion_e2e_test!(e2e_to_boolean_nan_is_false, "!NaN");
    coercion_e2e_test!(e2e_to_boolean_empty_string_is_false, "!''");
    coercion_e2e_test!(
        e2e_to_boolean_nullish_and_false_are_false,
        "!null && !undefined && !false"
    );
    coercion_e2e_test!(
        e2e_to_boolean_non_empty_string_is_true,
        "Boolean('0') === true"
    );
    coercion_e2e_test!(e2e_to_boolean_object_is_true, "Boolean({}) === true");
    coercion_e2e_test!(e2e_to_boolean_array_is_true, "Boolean([]) === true");
    coercion_e2e_test!(
        e2e_to_boolean_non_zero_numbers_are_true,
        "Boolean(1) === true && Boolean(-1) === true && Boolean(Infinity) === true"
    );

    coercion_e2e_test!(
        e2e_abstract_equality_null_equals_undefined_only,
        "null == undefined && undefined == null && !(null == 0)"
    );
    coercion_e2e_test!(
        e2e_abstract_equality_zero_empty_string_and_false,
        "'0' == false && '' == 0 && '' == false"
    );
    coercion_e2e_test!(
        e2e_abstract_equality_boolean_number_pairs,
        "true == 1 && false == 0"
    );
    coercion_e2e_test!(
        e2e_abstract_equality_array_and_number,
        "[1] == 1 && ['1'] == 1"
    );
    coercion_e2e_test!(
        e2e_abstract_equality_object_uses_to_primitive,
        "({ valueOf() { return 5; } }) == '5'"
    );
    coercion_e2e_test!(
        e2e_abstract_equality_object_falls_back_to_string_primitive,
        "({ valueOf() { return {}; }, toString() { return '8'; } }) == 8"
    );
    coercion_e2e_test!(
        e2e_abstract_equality_distinct_symbols_are_not_equal,
        "!(Symbol('x') == Symbol('x'))"
    );
    coercion_e2e_test!(
        e2e_abstract_equality_bigint_number_edges,
        "1n == 1 && !(1n == 1.5)"
    );
    coercion_e2e_test!(
        e2e_abstract_equality_nan_is_never_equal,
        "!(NaN == NaN) && !(Number('foo') == Number('foo'))"
    );

    coercion_e2e_test!(e2e_relational_string_comparison_stays_stringy, "'10' < '9'");
    coercion_e2e_test!(
        e2e_relational_number_and_string_compare_numerically,
        "!(10 < '9') && '5' < 10"
    );
    coercion_e2e_test!(
        e2e_relational_utf16_string_ordering,
        "'\\uD855\\uDE51' < '\\uFF3A'"
    );
    coercion_e2e_test!(
        e2e_relational_nan_makes_all_order_checks_false,
        "!(NaN < 1) && !(NaN > 1) && !(NaN <= 1) && !(NaN >= 1)"
    );
    coercion_e2e_test!(
        e2e_relational_null_and_undefined_numeric_conversion,
        "null < 1 && !(undefined < 1)"
    );
    coercion_e2e_test!(
        e2e_relational_less_equal_and_greater_equal_use_number_hint,
        "1 <= '1' && 1 >= '1'"
    );
    coercion_e2e_test!(
        e2e_relational_object_less_than_number,
        "({ valueOf() { return 3; } }) < 4"
    );
    coercion_e2e_test!(
        e2e_relational_number_greater_than_object,
        "4 > ({ valueOf() { return 3; } })"
    );
    coercion_e2e_test!(
        e2e_relational_object_falls_back_to_string,
        "({ valueOf() { return {}; }, toString() { return '2'; } }) <= 2"
    );
    coercion_e2e_test!(
        e2e_relational_array_and_number_coercion,
        "[5] < 10 && !([5] < [10])"
    );
    coercion_e2e_test!(
        e2e_relational_bigint_and_number,
        "1n < 2 && 2 > 1n && !(1n >= 2)"
    );
    coercion_e2e_test!(
        e2e_relational_type_error_when_number_hint_cannot_produce_primitive,
        "try { ({ valueOf() { return {}; }, toString() { return {}; } }) < 1; false; } catch (e) { e instanceof TypeError; }"
    );
}
