//! RegExp conformance tests for `/v` and duplicate named capture groups.

#[cfg(test)]
mod tests {
    use crate::builtins::global::global_eval;
    use crate::objects::value::JsValue;

    fn assert_eval_true(src: &str) {
        let result = global_eval(src).unwrap();
        assert_eq!(result, JsValue::Boolean(true), "expected true for: {src}");
    }

    fn assert_eval_err(src: &str) {
        assert!(global_eval(src).is_err(), "expected error for: {src}");
    }

    #[test]
    fn e2e_regexp_v_constructor_accepts_flag() {
        assert_eval_true("new RegExp('a', 'v') instanceof RegExp");
    }

    #[test]
    fn e2e_regexp_v_literal_accepts_flag() {
        assert_eval_true("/a/v instanceof RegExp");
    }

    #[test]
    fn e2e_regexp_v_flags_contains_v() {
        assert_eval_true("new RegExp('a', 'v').flags === 'v'");
    }

    #[test]
    fn e2e_regexp_v_flags_orders_before_y() {
        assert_eval_true("new RegExp('a', 'yvdg').flags === 'dgvy'");
    }

    #[test]
    fn e2e_regexp_v_literal_flags_orders_before_y() {
        assert_eval_true("/a/dgyv.flags === 'dgvy'");
    }

    #[test]
    fn e2e_regexp_v_unicode_sets_getter_is_true() {
        assert_eval_true("new RegExp('a', 'v').unicodeSets === true");
    }

    #[test]
    fn e2e_regexp_v_unicode_getter_is_true() {
        assert_eval_true("new RegExp('a', 'v').unicode === true");
    }

    #[test]
    fn e2e_regexp_v_literal_unicode_getter_is_true() {
        assert_eval_true("/a/v.unicode === true");
    }

    #[test]
    fn e2e_regexp_v_and_u_constructor_is_syntax_error() {
        assert_eval_err("new RegExp('a', 'uv')");
    }

    #[test]
    fn e2e_regexp_v_and_u_literal_is_syntax_error() {
        assert_eval_err("/a/uv");
    }

    #[test]
    fn e2e_regexp_v_string_property_rgi_emoji_matches() {
        assert_eval_true(r"new RegExp('\\p{RGI_Emoji}', 'v').test('👨‍👩‍👧‍👦')");
    }

    #[test]
    fn e2e_regexp_v_string_property_basic_emoji_matches() {
        assert_eval_true(r"new RegExp('\\p{Basic_Emoji}', 'v').test('😀')");
    }

    #[test]
    fn e2e_regexp_v_string_property_literal_matches() {
        assert_eval_true(r"/\p{RGI_Emoji}/v.test('👨‍👩‍👧‍👦')");
    }

    #[test]
    fn e2e_regexp_v_negated_string_property_is_syntax_error() {
        assert_eval_err(r"new RegExp('\\P{RGI_Emoji}', 'v')");
    }

    #[test]
    fn e2e_regexp_v_class_intersection_ascii_letters() {
        assert_eval_true(r"/[\p{ASCII}&&\p{Letter}]+/v.test('abc')");
    }

    #[test]
    fn e2e_regexp_v_class_intersection_rejects_non_letters() {
        assert_eval_true(r"/^[\p{ASCII}&&\p{Letter}]+$/v.test('123') === false");
    }

    #[test]
    fn e2e_regexp_v_class_subtraction_excludes_ascii_digits() {
        assert_eval_true(r"/^[\p{ASCII}--\p{Decimal_Number}]+$/v.test('abcXYZ')");
    }

    #[test]
    fn e2e_regexp_v_class_subtraction_rejects_digits() {
        assert_eval_true(r"/^[\p{ASCII}--\p{Decimal_Number}]+$/v.test('abc123') === false");
    }

    #[test]
    fn e2e_regexp_v_nested_class_set_operations() {
        assert_eval_true(r"/^[[\p{ASCII}&&\p{Letter}]--[AEIOUaeiou]]+$/v.test('bcdf')");
    }

    #[test]
    fn e2e_regexp_v_property_inside_class_set_matches() {
        assert_eval_true(r"/^[\p{RGI_Emoji}]$/v.test('😀')");
    }

    #[test]
    fn e2e_regexp_v_to_string_preserves_flag() {
        assert_eval_true("String(new RegExp('a', 'gv')) === '/a/gv'");
    }

    #[test]
    fn e2e_regexp_v_exec_reports_unicode_sets() {
        assert_eval_true("var re = /a/v; re.unicodeSets && re.unicode && re.flags === 'v'");
    }

    #[test]
    fn e2e_regexp_duplicate_named_groups_first_branch_matches() {
        assert_eval_true(r"/(?<name>a)|(?<name>b)/.exec('a').groups.name === 'a'");
    }

    #[test]
    fn e2e_regexp_duplicate_named_groups_second_branch_matches() {
        assert_eval_true(r"/(?<name>a)|(?<name>b)/.exec('b').groups.name === 'b'");
    }

    #[test]
    fn e2e_regexp_duplicate_named_groups_unmatched_branch_is_ignored() {
        assert_eval_true(
            r"var m = /(?<left>a)|(?<left>b)/.exec('b'); m.groups.left === 'b' && m[1] === undefined && m[2] === 'b'",
        );
    }

    #[test]
    fn e2e_regexp_duplicate_named_groups_with_constructor() {
        assert_eval_true(r"new RegExp('(?<name>a)|(?<name>b)').exec('a').groups.name === 'a'");
    }

    #[test]
    fn e2e_regexp_duplicate_named_groups_with_quantified_alternatives() {
        assert_eval_true(r"/(?:(?<word>foo)|(?<word>bar))/.exec('bar').groups.word === 'bar'");
    }

    #[test]
    fn e2e_regexp_duplicate_named_groups_backref_first_branch() {
        assert_eval_true(r"/(?:(?<name>a)|(?<name>b))\k<name>/.test('aa')");
    }

    #[test]
    fn e2e_regexp_duplicate_named_groups_backref_second_branch() {
        assert_eval_true(r"/(?:(?<name>a)|(?<name>b))\k<name>/.test('bb')");
    }

    #[test]
    fn e2e_regexp_duplicate_named_groups_backref_rejects_cross_branch() {
        assert_eval_true(r"/(?:(?<name>a)|(?<name>b))\k<name>/.test('ab') === false");
    }

    #[test]
    fn e2e_regexp_duplicate_named_groups_backref_constructor() {
        assert_eval_true(r"new RegExp('(?:(?<name>a)|(?<name>b))\\k<name>').test('bb') === true");
    }

    #[test]
    fn e2e_regexp_duplicate_named_groups_replace_uses_matched_branch() {
        assert_eval_true(r"'a b'.replace(/(?<name>a)|(?<name>b)/g, '[$<name>]') === '[a] [b]'");
    }

    #[test]
    fn e2e_regexp_duplicate_named_groups_match_all_uses_matched_branch() {
        assert_eval_true(
            r"var ms = Array.from('ab'.matchAll(/(?<name>a)|(?<name>b)/g)); ms[0].groups.name === 'a' && ms[1].groups.name === 'b'",
        );
    }

    #[test]
    fn e2e_regexp_duplicate_named_groups_indices_use_matched_branch() {
        assert_eval_true(
            r"var m = /(?<name>a)|(?<name>b)/d.exec('b'); m.indices.groups.name[0] === 0 && m.indices.groups.name[1] === 1",
        );
    }

    #[test]
    fn e2e_regexp_duplicate_named_groups_nested_alternation_branch_one() {
        assert_eval_true(r"/(?:(?<x>foo)|(?:(?<x>bar)))/.exec('foo').groups.x === 'foo'");
    }

    #[test]
    fn e2e_regexp_duplicate_named_groups_nested_alternation_branch_two() {
        assert_eval_true(r"/(?:(?<x>foo)|(?:(?<x>bar)))/.exec('bar').groups.x === 'bar'");
    }

    #[test]
    fn e2e_regexp_duplicate_named_groups_literal_flags_stay_canonical() {
        assert_eval_true(r"/(?<name>a)|(?<name>b)/dgv.flags === 'dgv'");
    }

    #[test]
    fn e2e_regexp_duplicate_named_groups_can_coexist_with_v_mode() {
        assert_eval_true(r"/(?:(?<emoji>\p{RGI_Emoji})|(?<emoji>x))\k<emoji>/v.test('xx')");
    }
}
