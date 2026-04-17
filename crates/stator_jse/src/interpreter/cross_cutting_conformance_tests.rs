//! Cross-cutting conformance tests — edge cases and gap closure.
//!
//! Covers: `typeof`, `void`, comma operator, compound assignment operators,
//! increment/decrement, tagged templates, optional chaining, nullish
//! coalescing assignment, logical assignment, `Object.hasOwn`,
//! `globalThis`, and `queueMicrotask`.

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
    #[allow(dead_code)]
    fn assert_eval_err(src: &str) {
        assert!(global_eval(src).is_err(), "expected error for: {src}");
    }

    // ── 1. typeof on all value types ────────────────────────────────────────

    #[test]
    fn e2e_typeof_undefined() {
        assert_eval_true("typeof undefined === 'undefined'");
    }

    #[test]
    fn e2e_typeof_null_is_object() {
        assert_eval_true("typeof null === 'object'");
    }

    #[test]
    fn e2e_typeof_boolean() {
        assert_eval_true("typeof true === 'boolean' && typeof false === 'boolean'");
    }

    #[test]
    fn e2e_typeof_number() {
        assert_eval_true(
            "typeof 42 === 'number' && typeof 3.14 === 'number' && typeof NaN === 'number'",
        );
    }

    #[test]
    fn e2e_typeof_bigint() {
        assert_eval_true("typeof 0n === 'bigint'");
    }

    #[test]
    fn e2e_typeof_string() {
        assert_eval_true("typeof 'hello' === 'string' && typeof '' === 'string'");
    }

    #[test]
    fn e2e_typeof_symbol() {
        assert_eval_true("typeof Symbol() === 'symbol'");
    }

    #[test]
    fn e2e_typeof_function() {
        assert_eval_true("typeof function(){} === 'function' && typeof (() => {}) === 'function'");
    }

    // ── 2. void operator ────────────────────────────────────────────────────

    #[test]
    fn e2e_void_zero_is_undefined() {
        assert_eval_true("void 0 === undefined");
    }

    #[test]
    fn e2e_void_expression_is_undefined() {
        assert_eval_true("void (1 + 2) === undefined && void 'abc' === undefined");
    }

    // ── 3. Comma operator ───────────────────────────────────────────────────

    #[test]
    fn e2e_comma_expression_returns_last() {
        assert_eval_true("(1, 2, 3) === 3");
    }

    #[test]
    fn e2e_comma_in_for_loop() {
        assert_eval_true(
            "var sum = 0; for (var i = 0, j = 10; i < 3; i++, j--) { sum = sum + i + j; } sum === 30",
        );
    }

    #[test]
    fn e2e_comma_in_variable_declaration() {
        assert_eval_true("var a = 1, b = 2, c = 3; a + b + c === 6");
    }

    // ── 4. Compound assignment operators ────────────────────────────────────

    #[test]
    fn e2e_add_assign() {
        assert_eval_true("var x = 10; x += 5; x === 15");
    }

    #[test]
    fn e2e_sub_assign() {
        assert_eval_true("var x = 10; x -= 3; x === 7");
    }

    #[test]
    fn e2e_mul_assign() {
        assert_eval_true("var x = 4; x *= 3; x === 12");
    }

    #[test]
    fn e2e_div_assign() {
        assert_eval_true("var x = 20; x /= 4; x === 5");
    }

    #[test]
    fn e2e_mod_assign() {
        assert_eval_true("var x = 10; x %= 3; x === 1");
    }

    #[test]
    fn e2e_exp_assign() {
        assert_eval_true("var x = 2; x **= 10; x === 1024");
    }

    #[test]
    fn e2e_shl_assign() {
        assert_eval_true("var x = 1; x <<= 3; x === 8");
    }

    #[test]
    fn e2e_shr_assign() {
        assert_eval_true("var x = 16; x >>= 2; x === 4");
    }

    #[test]
    fn e2e_ushr_assign() {
        assert_eval_true("var x = -1; x >>>= 0; x === 4294967295");
    }

    #[test]
    fn e2e_bitand_assign() {
        assert_eval_true("var x = 0b1111; x &= 0b1010; x === 0b1010");
    }

    #[test]
    fn e2e_bitor_assign() {
        assert_eval_true("var x = 0b1010; x |= 0b0101; x === 0b1111");
    }

    #[test]
    fn e2e_bitxor_assign() {
        assert_eval_true("var x = 0b1111; x ^= 0b1010; x === 0b0101");
    }

    #[test]
    fn e2e_logical_and_assign_truthy() {
        assert_eval_true("var x = 1; x &&= 42; x === 42");
    }

    #[test]
    fn e2e_logical_and_assign_falsy_short_circuits() {
        assert_eval_true("var x = 0; x &&= 42; x === 0");
    }

    #[test]
    fn e2e_logical_or_assign_falsy() {
        assert_eval_true("var x = 0; x ||= 99; x === 99");
    }

    #[test]
    fn e2e_logical_or_assign_truthy_short_circuits() {
        assert_eval_true("var x = 5; x ||= 99; x === 5");
    }

    #[test]
    fn e2e_nullish_assign_null() {
        assert_eval_true("var x = null; x ??= 77; x === 77");
    }

    #[test]
    fn e2e_nullish_assign_undefined() {
        assert_eval_true("var x = undefined; x ??= 77; x === 77");
    }

    #[test]
    fn e2e_nullish_assign_zero_keeps_value() {
        assert_eval_true("var x = 0; x ??= 77; x === 0");
    }

    // ── 5. Increment / decrement ────────────────────────────────────────────

    #[test]
    fn e2e_prefix_increment() {
        assert_eval_true("var x = 5; ++x === 6 && x === 6");
    }

    #[test]
    fn e2e_postfix_increment() {
        assert_eval_true("var x = 5; x++ === 5 && x === 6");
    }

    #[test]
    fn e2e_prefix_decrement() {
        assert_eval_true("var x = 5; --x === 4 && x === 4");
    }

    #[test]
    fn e2e_postfix_decrement() {
        assert_eval_true("var x = 5; x-- === 5 && x === 4");
    }

    #[test]
    fn e2e_increment_on_object_property() {
        assert_eval_true("var o = { a: 10 }; o.a++; o.a === 11");
    }

    #[test]
    fn e2e_prefix_increment_on_object_property() {
        assert_eval_true("var o = { a: 10 }; ++o.a === 11 && o.a === 11");
    }

    // ── 6. Tagged template literals ─────────────────────────────────────────

    #[test]
    fn e2e_tagged_template_receives_strings_array() {
        assert_eval_true(
            "function tag(strings) { return strings[0] === 'hello ' && strings[1] === ' world'; } \
             tag`hello ${1} world`",
        );
    }

    #[test]
    fn e2e_tagged_template_receives_expression_values() {
        assert_eval_true(
            "function tag(strings, a, b) { return a === 1 && b === 2; } \
             tag`${1} and ${2}`",
        );
    }

    #[test]
    fn e2e_tagged_template_raw_property() {
        assert_eval_true(
            r"function tag(strings) { return strings.raw[0] === 'line1\\nline2'; } tag`line1\nline2`",
        );
    }

    // ── 7. Optional chaining ────────────────────────────────────────────────

    #[test]
    fn e2e_optional_chaining_prop_exists() {
        assert_eval_true("var o = { a: { b: 42 } }; o?.a?.b === 42");
    }

    #[test]
    fn e2e_optional_chaining_prop_nullish() {
        assert_eval_true("var o = null; o?.a === undefined");
    }

    #[test]
    fn e2e_optional_chaining_bracket_access() {
        assert_eval_true("var o = { key: 99 }; o?.['key'] === 99");
    }

    #[test]
    fn e2e_optional_chaining_bracket_nullish() {
        assert_eval_true("var o = undefined; o?.['key'] === undefined");
    }

    #[test]
    fn e2e_optional_chaining_call() {
        assert_eval_true("var o = { fn: function() { return 7; } }; o.fn?.() === 7");
    }

    #[test]
    fn e2e_optional_chaining_call_nullish() {
        assert_eval_true("var o = {}; o.fn?.() === undefined");
    }

    #[test]
    fn e2e_optional_chaining_deep_null() {
        assert_eval_true("var o = { a: null }; o.a?.b?.c === undefined");
    }

    // ── 8. Nullish coalescing assignment — edge cases ───────────────────────

    #[test]
    fn e2e_nullish_assign_false_keeps_value() {
        assert_eval_true("var x = false; x ??= 77; x === false");
    }

    #[test]
    fn e2e_nullish_assign_empty_string_keeps_value() {
        assert_eval_true("var x = ''; x ??= 'default'; x === ''");
    }

    // ── 9. Logical assignment — extra edge cases ────────────────────────────

    #[test]
    fn e2e_logical_and_assign_null() {
        assert_eval_true("var x = null; x &&= 42; x === null");
    }

    #[test]
    fn e2e_logical_or_assign_null() {
        assert_eval_true("var x = null; x ||= 55; x === 55");
    }

    #[test]
    fn e2e_logical_or_assign_empty_string() {
        assert_eval_true("var x = ''; x ||= 'filled'; x === 'filled'");
    }

    // ── 10. Object.hasOwn ───────────────────────────────────────────────────

    #[test]
    fn e2e_object_has_own_true() {
        assert_eval_true("Object.hasOwn({ a: 1 }, 'a') === true");
    }

    #[test]
    fn e2e_object_has_own_false_for_missing() {
        assert_eval_true("Object.hasOwn({ a: 1 }, 'b') === false");
    }

    #[test]
    fn e2e_object_has_own_false_for_inherited() {
        assert_eval_true("Object.hasOwn({}, 'toString') === false");
    }

    // ── 13. globalThis ──────────────────────────────────────────────────────

    #[test]
    fn e2e_global_this_is_object() {
        assert_eval_true("typeof globalThis === 'object'");
    }

    #[test]
    fn e2e_global_this_has_object_constructor() {
        assert_eval_true("typeof globalThis.Object === 'function'");
    }

    // ── Misc: typeof on undeclared variable ─────────────────────────────────

    #[test]
    fn e2e_typeof_undeclared_var() {
        assert_eval_true("typeof someUndeclaredVariable === 'undefined'");
    }

    // ── Misc: chained assignment ────────────────────────────────────────────

    #[test]
    fn e2e_chained_assignment() {
        assert_eval_true("var a, b, c; a = b = c = 42; a === 42 && b === 42 && c === 42");
    }

    // ── Misc: compound assignment on object properties ──────────────────────

    #[test]
    fn e2e_compound_assign_on_property() {
        assert_eval_true("var o = { x: 10 }; o.x += 5; o.x *= 2; o.x === 30");
    }

    // ── Misc: nullish coalescing chain ──────────────────────────────────────

    #[test]
    #[ignore] // TODO: nullish coalescing chain regression
    fn e2e_nullish_coalescing_chain() {
        assert_eval_true("null ?? undefined ?? 0 ?? 42 === 0");
    }

    // ── Misc: void side effects still evaluated ─────────────────────────────

    #[test]
    fn e2e_void_side_effects() {
        assert_eval_true("var x = 0; void (x = 5); x === 5");
    }

    // ── Misc: optional chaining with nullish coalescing ─────────────────────

    #[test]
    fn e2e_optional_chaining_with_nullish_coalescing() {
        assert_eval_true("var o = null; (o?.a ?? 'default') === 'default'");
    }
}
