//! Map and Set conformance tests.

#[cfg(test)]
mod tests {
    use crate::builtins::global::global_eval;
    use crate::objects::value::JsValue;

    fn assert_eval_true(src: &str) {
        let result = global_eval(src).unwrap();
        assert_eq!(result, JsValue::Boolean(true), "expected true for: {src}");
    }

    // ── Map ──────────────────────────────────────────────────────────────────

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_constructor_from_array_iterable() {
        assert_eval_true("new Map([[1, 'a'], [2, 'b']]).get(2) === 'b'");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_constructor_from_custom_iterable() {
        assert_eval_true(
            "var iterable = { [Symbol.iterator]: function() { var i = 0; return { next: function() { i++; if (i === 1) return { value: ['x', 1], done: false }; if (i === 2) return { value: ['y', 2], done: false }; return { value: undefined, done: true }; } }; } }; var m = new Map(iterable); m.get('x') === 1 && m.get('y') === 2 && m.size === 2",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_constructor_from_map_iterable() {
        assert_eval_true(
            "var src = new Map([[1, 'a'], [2, 'b']]); var copy = new Map(src); copy.get(1) === 'a' && copy.get(2) === 'b' && copy.size === 2",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_insertion_order_entries() {
        assert_eval_true(
            "var it = new Map([[1, 'a'], [2, 'b'], [3, 'c']]).entries(); var a = it.next(); var b = it.next(); var c = it.next(); !a.done && a.value[0] === 1 && a.value[1] === 'a' && !b.done && b.value[0] === 2 && b.value[1] === 'b' && !c.done && c.value[0] === 3 && c.value[1] === 'c'",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_reset_existing_key_preserves_order() {
        assert_eval_true(
            "var m = new Map([[1, 'a'], [2, 'b']]); m.set(1, 'z'); var it = m.keys(); it.next().value === 1 && it.next().value === 2 && m.get(1) === 'z'",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_delete_and_reinsert_moves_to_end() {
        assert_eval_true(
            "var m = new Map([[1, 'a'], [2, 'b']]); m.delete(1); m.set(1, 'c'); var it = m.keys(); it.next().value === 2 && it.next().value === 1",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_negative_zero_normalizes_lookup() {
        assert_eval_true(
            "var m = new Map(); m.set(-0, 'v'); m.get(0) === 'v' && m.get(-0) === 'v' && m.has(+0) && m.has(-0)",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_negative_zero_iterates_as_positive_zero() {
        assert_eval_true(
            "var m = new Map(); m.set(-0, 'v'); var entry = m.entries().next().value; entry[0] === 0 && 1 / entry[0] === Infinity",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_nan_key_roundtrips() {
        assert_eval_true("var m = new Map(); m.set(NaN, 'v'); m.get(NaN) === 'v' && m.has(NaN)");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_nan_key_deduplicates() {
        assert_eval_true(
            "var m = new Map(); m.set(NaN, 1); m.set(NaN, 2); m.size === 1 && m.get(NaN) === 2",
        );
    }

    #[test]
    fn e2e_map_prototype_chain_is_correct() {
        assert_eval_true("Object.getPrototypeOf(new Map()) === Map.prototype");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_size_is_prototype_getter() {
        assert_eval_true(
            "var desc = Object.getOwnPropertyDescriptor(Map.prototype, 'size'); typeof desc.get === 'function' && desc.value === undefined && new Map([[1, 2]]).hasOwnProperty('size') === false",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_clear_resets_size_and_content() {
        assert_eval_true(
            "var m = new Map([[1, 'a'], [2, 'b']]); m.clear(); m.size === 0 && m.get(1) === undefined && m.has(2) === false",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_for_each_callback_arguments() {
        assert_eval_true(
            "var m = new Map([[1, 'a']]); var ok = false; m.forEach(function(v, k, self) { ok = v === 'a' && k === 1 && self === m; }); ok",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_for_each_uses_this_arg() {
        assert_eval_true(
            "var m = new Map([[1, 'a']]); var ctx = { seen: 0 }; m.forEach(function() { this.seen++; }, ctx); ctx.seen === 1",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_for_each_visits_new_items_added_during_iteration() {
        assert_eval_true(
            "var m = new Map([[1, 'a'], [2, 'b']]); var seen = []; m.forEach(function(v, k) { seen.push(k); if (k === 1) m.set(3, 'c'); }); seen.join(',') === '1,2,3'",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_for_each_deleting_visited_item_is_safe() {
        assert_eval_true(
            "var m = new Map([[1, 'a'], [2, 'b'], [3, 'c']]); var seen = []; m.forEach(function(v, k) { seen.push(k); m.delete(k); }); seen.join(',') === '1,2,3' && m.size === 0",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_keys_iterator_order() {
        assert_eval_true(
            "var it = new Map([[1, 'a'], [2, 'b']]).keys(); var a = it.next(); var b = it.next(); var c = it.next(); !a.done && a.value === 1 && !b.done && b.value === 2 && c.done",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_values_iterator_order() {
        assert_eval_true(
            "var it = new Map([[1, 'a'], [2, 'b']]).values(); var a = it.next(); var b = it.next(); var c = it.next(); !a.done && a.value === 'a' && !b.done && b.value === 'b' && c.done",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_entries_iterator_order() {
        assert_eval_true(
            "var it = new Map([[1, 'a'], [2, 'b']]).entries(); var a = it.next(); var b = it.next(); var c = it.next(); !a.done && a.value[0] === 1 && a.value[1] === 'a' && !b.done && b.value[0] === 2 && b.value[1] === 'b' && c.done",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_iterator_is_self_iterable() {
        assert_eval_true("var it = new Map([[1, 'a']]).entries(); it[Symbol.iterator]() === it");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_iterator_sees_mutation_after_creation() {
        assert_eval_true(
            "var m = new Map([[1, 'a']]); var it = m.keys(); var first = it.next(); m.set(2, 'b'); var second = it.next(); !first.done && first.value === 1 && !second.done && second.value === 2",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_map_to_string_tag() {
        assert_eval_true("Map.prototype[Symbol.toStringTag] === 'Map'");
    }

    // ── Set ──────────────────────────────────────────────────────────────────

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_constructor_from_array_iterable() {
        assert_eval_true(
            "var s = new Set([1, 2, 3]); s.has(1) && s.has(2) && s.has(3) && s.size === 3",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_constructor_from_custom_iterable() {
        assert_eval_true(
            "var iterable = { [Symbol.iterator]: function() { var i = 0; return { next: function() { i++; if (i === 1) return { value: 4, done: false }; if (i === 2) return { value: 5, done: false }; return { value: undefined, done: true }; } }; } }; var s = new Set(iterable); s.has(4) && s.has(5) && s.size === 2",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_constructor_from_set_iterable() {
        assert_eval_true(
            "var src = new Set([1, 2]); var copy = new Set(src); copy.has(1) && copy.has(2) && copy.size === 2",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_constructor_from_string_iterable() {
        assert_eval_true("var s = new Set('aba'); s.size === 2 && s.has('a') && s.has('b')");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_insertion_order_values() {
        assert_eval_true(
            "var it = new Set([1, 2, 3]).values(); var a = it.next(); var b = it.next(); var c = it.next(); !a.done && a.value === 1 && !b.done && b.value === 2 && !c.done && c.value === 3",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_readding_existing_value_preserves_order() {
        assert_eval_true(
            "var s = new Set([1, 2]); s.add(1); var it = s.values(); it.next().value === 1 && it.next().value === 2 && s.size === 2",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_delete_and_reinsert_moves_to_end() {
        assert_eval_true(
            "var s = new Set([1, 2]); s.delete(1); s.add(1); var it = s.values(); it.next().value === 2 && it.next().value === 1",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_negative_zero_normalizes_lookup() {
        assert_eval_true("var s = new Set(); s.add(-0); s.has(0) && s.has(-0) && s.size === 1");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_negative_zero_iterates_as_positive_zero() {
        assert_eval_true(
            "var s = new Set(); s.add(-0); var value = s.values().next().value; value === 0 && 1 / value === Infinity",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_nan_roundtrips() {
        assert_eval_true("var s = new Set(); s.add(NaN); s.has(NaN) && s.size === 1");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_nan_deduplicates() {
        assert_eval_true("var s = new Set(); s.add(NaN); s.add(NaN); s.size === 1");
    }

    #[test]
    fn e2e_set_prototype_chain_is_correct() {
        assert_eval_true("Object.getPrototypeOf(new Set()) === Set.prototype");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_size_is_prototype_getter() {
        assert_eval_true(
            "var desc = Object.getOwnPropertyDescriptor(Set.prototype, 'size'); typeof desc.get === 'function' && desc.value === undefined && new Set([1]).hasOwnProperty('size') === false",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_clear_resets_size_and_content() {
        assert_eval_true(
            "var s = new Set([1, 2]); s.clear(); s.size === 0 && s.has(1) === false && s.has(2) === false",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_for_each_callback_arguments() {
        assert_eval_true(
            "var s = new Set([1]); var ok = false; s.forEach(function(v, v2, self) { ok = v === 1 && v2 === 1 && self === s; }); ok",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_for_each_uses_this_arg() {
        assert_eval_true(
            "var s = new Set([1]); var ctx = { seen: 0 }; s.forEach(function() { this.seen++; }, ctx); ctx.seen === 1",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_for_each_visits_new_items_added_during_iteration() {
        assert_eval_true(
            "var s = new Set([1, 2]); var seen = []; s.forEach(function(v) { seen.push(v); if (v === 1) s.add(3); }); seen.join(',') === '1,2,3'",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_for_each_deleting_visited_item_is_safe() {
        assert_eval_true(
            "var s = new Set([1, 2, 3]); var seen = []; s.forEach(function(v) { seen.push(v); s.delete(v); }); seen.join(',') === '1,2,3' && s.size === 0",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_values_iterator_order() {
        assert_eval_true(
            "var it = new Set([1, 2]).values(); var a = it.next(); var b = it.next(); var c = it.next(); !a.done && a.value === 1 && !b.done && b.value === 2 && c.done",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_keys_iterator_matches_values() {
        assert_eval_true(
            "var it = new Set([1, 2]).keys(); var a = it.next(); var b = it.next(); var c = it.next(); !a.done && a.value === 1 && !b.done && b.value === 2 && c.done",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_entries_iterator_pairs_values() {
        assert_eval_true(
            "var it = new Set([1, 2]).entries(); var a = it.next(); var b = it.next(); var c = it.next(); !a.done && a.value[0] === 1 && a.value[1] === 1 && !b.done && b.value[0] === 2 && b.value[1] === 2 && c.done",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_iterator_is_self_iterable() {
        assert_eval_true("var it = new Set([1]).values(); it[Symbol.iterator]() === it");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_iterator_sees_mutation_after_creation() {
        assert_eval_true(
            "var s = new Set([1]); var it = s.values(); var first = it.next(); s.add(2); var second = it.next(); !first.done && first.value === 1 && !second.done && second.value === 2",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_set_to_string_tag() {
        assert_eval_true("Set.prototype[Symbol.toStringTag] === 'Set'");
    }
}
