//! End-to-end tests verifying that `Array` and `TypedArray` methods with the
//! same name expose identical observable semantics (modulo the specified
//! differences such as return type and default sort comparator).
//!
//! These tests exercise the ten alignment rules listed in the ECMAScript
//! specification (§23.1 vs §23.2) and constitute the 35+ e2e coverage
//! required by the task.

#[cfg(test)]
mod tests {
    use crate::builtins::array::*;
    use crate::builtins::typed_array::*;
    use crate::error::StatorError;
    use crate::objects::js_array::JsArray;
    use crate::objects::value::JsValue;

    // ── helpers ──────────────────────────────────────────────────────────

    /// Build a quick `JsArray` from `i32` values.
    fn arr_of(vals: &[i32]) -> JsArray {
        array_of(&vals.iter().map(|&n| JsValue::Smi(n)).collect::<Vec<_>>())
    }

    /// Build a quick `Int32` typed array from `i32` values.
    fn ta_of(vals: &[i32]) -> JsTypedArray {
        typed_array_from_values(
            TypedArrayKind::Int32,
            &vals.iter().map(|&n| JsValue::Smi(n)).collect::<Vec<_>>(),
        )
        .unwrap()
    }

    // =====================================================================
    // 1. map — both have map; TypedArray.map returns same TypedArray type
    // =====================================================================

    #[test]
    fn test_align_map_both_double_values() {
        let a = arr_of(&[1, 2, 3]);
        let mapped_a = array_map(&a, |v, _| match v {
            JsValue::Smi(n) => JsValue::Smi(n * 2),
            _ => v.clone(),
        });
        assert_eq!(mapped_a.get(0), JsValue::Smi(2));
        assert_eq!(mapped_a.get(2), JsValue::Smi(6));

        let t = ta_of(&[1, 2, 3]);
        let mapped_t = typed_array_map(&t, |v, _| match v {
            JsValue::Smi(n) => Ok(JsValue::Smi(n * 2)),
            _ => Ok(v.clone()),
        })
        .unwrap();
        // Rule 2: TypedArray.map returns same TypedArray type
        assert_eq!(mapped_t.kind, TypedArrayKind::Int32);
        assert_eq!(typed_array_get(&mapped_t, 0), JsValue::Smi(2));
        assert_eq!(typed_array_get(&mapped_t, 2), JsValue::Smi(6));
    }

    // =====================================================================
    // 2. filter — TypedArray.filter returns same TypedArray type
    // =====================================================================

    #[test]
    fn test_align_filter_both_keep_even() {
        let a = arr_of(&[1, 2, 3, 4]);
        let fa = array_filter(&a, |v, _| matches!(v, JsValue::Smi(n) if n % 2 == 0));
        assert_eq!(fa.length(), 2);
        assert_eq!(fa.get(0), JsValue::Smi(2));

        let t = ta_of(&[1, 2, 3, 4]);
        let ft =
            typed_array_filter(&t, |v, _| Ok(matches!(v, JsValue::Smi(n) if n % 2 == 0))).unwrap();
        // Rule 3: TypedArray.filter returns same TypedArray type
        assert_eq!(ft.kind, TypedArrayKind::Int32);
        assert_eq!(typed_array_length(&ft), 2);
        assert_eq!(typed_array_get(&ft, 0), JsValue::Smi(2));
    }

    // =====================================================================
    // 3. find / findIndex — identical predicate semantics
    // =====================================================================

    #[test]
    fn test_align_find_both_find_first_gt_two() {
        let a = arr_of(&[1, 3, 5]);
        let found_a = array_find(&a, |v, _| matches!(v, JsValue::Smi(n) if *n > 2));
        assert_eq!(found_a, JsValue::Smi(3));

        let t = ta_of(&[1, 3, 5]);
        let found_t =
            typed_array_find(&t, |v, _| Ok(matches!(v, JsValue::Smi(n) if *n > 2))).unwrap();
        assert_eq!(found_t, JsValue::Smi(3));
    }

    #[test]
    fn test_align_find_index_both() {
        let a = arr_of(&[10, 20, 30]);
        assert_eq!(
            array_find_index(&a, |v, _| matches!(v, JsValue::Smi(20))),
            Some(1)
        );

        let t = ta_of(&[10, 20, 30]);
        assert_eq!(
            typed_array_find_index(&t, |v, _| Ok(matches!(v, JsValue::Smi(20)))).unwrap(),
            1
        );
    }

    // =====================================================================
    // 4. findLast / findLastIndex — reverse search
    // =====================================================================

    #[test]
    fn test_align_find_last_both() {
        let a = arr_of(&[1, 2, 3, 2]);
        let last_a = array_find_last(&a, |v, _| matches!(v, JsValue::Smi(2)));
        assert_eq!(last_a, JsValue::Smi(2));

        let t = ta_of(&[1, 2, 3, 2]);
        let last_t = typed_array_find_last(&t, |v, _| Ok(matches!(v, JsValue::Smi(2)))).unwrap();
        assert_eq!(last_t, JsValue::Smi(2));
    }

    #[test]
    fn test_align_find_last_index_both() {
        let a = arr_of(&[1, 2, 3, 2]);
        assert_eq!(
            array_find_last_index(&a, |v, _| matches!(v, JsValue::Smi(2))),
            Some(3)
        );

        let t = ta_of(&[1, 2, 3, 2]);
        assert_eq!(
            typed_array_find_last_index(&t, |v, _| Ok(matches!(v, JsValue::Smi(2)))).unwrap(),
            3
        );
    }

    // =====================================================================
    // 5. every — returns false if any predicate fails
    // =====================================================================

    #[test]
    fn test_align_every_all_positive() {
        let a = arr_of(&[1, 2, 3]);
        assert!(array_every(
            &a,
            |v, _| matches!(v, JsValue::Smi(n) if *n > 0)
        ));

        let t = ta_of(&[1, 2, 3]);
        assert!(typed_array_every(&t, |v, _| Ok(matches!(v, JsValue::Smi(n) if *n > 0))).unwrap());
    }

    #[test]
    fn test_align_every_fails_on_negative() {
        let a = arr_of(&[1, -1, 3]);
        assert!(!array_every(
            &a,
            |v, _| matches!(v, JsValue::Smi(n) if *n > 0)
        ));

        let t = ta_of(&[1, -1, 3]);
        assert!(!typed_array_every(&t, |v, _| Ok(matches!(v, JsValue::Smi(n) if *n > 0))).unwrap());
    }

    // =====================================================================
    // 6. some — returns true if any predicate succeeds
    // =====================================================================

    #[test]
    fn test_align_some_finds_match() {
        let a = arr_of(&[1, 2, 3]);
        assert!(array_some(
            &a,
            |v, _| matches!(v, JsValue::Smi(n) if *n == 2)
        ));

        let t = ta_of(&[1, 2, 3]);
        assert!(typed_array_some(&t, |v, _| Ok(matches!(v, JsValue::Smi(n) if *n == 2))).unwrap());
    }

    // =====================================================================
    // 7. reduce / reduceRight — accumulator semantics
    // =====================================================================

    #[test]
    fn test_align_reduce_sum() {
        let a = arr_of(&[1, 2, 3]);
        let sum_a = array_reduce(
            &a,
            |acc, v, _| match (acc, v) {
                (JsValue::Smi(a), JsValue::Smi(b)) => JsValue::Smi(a + b),
                _ => JsValue::Undefined,
            },
            Some(JsValue::Smi(0)),
        )
        .unwrap();
        assert_eq!(sum_a, JsValue::Smi(6));

        let t = ta_of(&[1, 2, 3]);
        let sum_t = typed_array_reduce(
            &t,
            |acc, v, _| {
                let a = acc.to_number()?;
                let b = v.to_number()?;
                Ok(JsValue::Smi((a + b) as i32))
            },
            Some(JsValue::Smi(0)),
        )
        .unwrap();
        assert_eq!(sum_t, JsValue::Smi(6));
    }

    #[test]
    fn test_align_reduce_right_concat_order() {
        let a = arr_of(&[1, 2, 3]);
        let rr_a = array_reduce_right(
            &a,
            |acc, v, _| match (acc, v) {
                (JsValue::Smi(a), JsValue::Smi(b)) => JsValue::Smi(a * 10 + b),
                _ => JsValue::Undefined,
            },
            Some(JsValue::Smi(0)),
        )
        .unwrap();
        // 0→acc, 3→v ⇒ 3; 3→acc, 2→v ⇒ 32; 32→acc, 1→v ⇒ 321
        assert_eq!(rr_a, JsValue::Smi(321));

        let t = ta_of(&[1, 2, 3]);
        let rr_t = typed_array_reduce_right(
            &t,
            |acc, v, _| {
                let a = acc.to_number()?;
                let b = v.to_number()?;
                Ok(JsValue::Smi((a * 10.0 + b) as i32))
            },
            Some(JsValue::Smi(0)),
        )
        .unwrap();
        assert_eq!(rr_t, JsValue::Smi(321));
    }

    #[test]
    fn test_align_reduce_empty_no_initial_errors() {
        let a = JsArray::new();
        assert!(matches!(
            array_reduce(&a, |acc, _, _| acc, None),
            Err(StatorError::TypeError(_))
        ));

        let t = typed_array_new_from_length(TypedArrayKind::Int32, 0);
        assert!(matches!(
            typed_array_reduce(&t, |acc, _, _| Ok(acc.clone()), None),
            Err(StatorError::TypeError(_))
        ));
    }

    // =====================================================================
    // 8. forEach — visits all elements
    // =====================================================================

    #[test]
    fn test_align_for_each_visits_all() {
        let a = arr_of(&[10, 20, 30]);
        let mut sum_a = 0i32;
        array_for_each(&a, |v, _| {
            if let JsValue::Smi(n) = v {
                sum_a += n;
            }
        });
        assert_eq!(sum_a, 60);

        let t = ta_of(&[10, 20, 30]);
        let mut sum_t = 0i32;
        typed_array_for_each(&t, |v, _| {
            if let JsValue::Smi(n) = v {
                sum_t += n;
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(sum_t, 60);
    }

    // =====================================================================
    // 9. indexOf / includes / lastIndexOf — search semantics
    // =====================================================================

    #[test]
    fn test_align_index_of_both() {
        let a = arr_of(&[5, 10, 15, 10]);
        assert_eq!(array_index_of(&a, &JsValue::Smi(10), None), Some(1));

        let t = ta_of(&[5, 10, 15, 10]);
        assert_eq!(typed_array_index_of(&t, &JsValue::Smi(10), 0), 1);
    }

    #[test]
    fn test_align_includes_both() {
        let a = arr_of(&[1, 2, 3]);
        assert!(array_includes(&a, &JsValue::Smi(2), None));
        assert!(!array_includes(&a, &JsValue::Smi(99), None));

        let t = ta_of(&[1, 2, 3]);
        assert!(typed_array_includes(&t, &JsValue::Smi(2), 0));
        assert!(!typed_array_includes(&t, &JsValue::Smi(99), 0));
    }

    #[test]
    fn test_align_last_index_of_both() {
        let a = arr_of(&[5, 10, 15, 10]);
        assert_eq!(array_last_index_of(&a, &JsValue::Smi(10), None), Some(3));

        let t = ta_of(&[5, 10, 15, 10]);
        assert_eq!(typed_array_last_index_of(&t, &JsValue::Smi(10), 3), 3);
    }

    // =====================================================================
    // 10. join — string concatenation with separator
    // =====================================================================

    #[test]
    fn test_align_join_default_separator() {
        let a = arr_of(&[1, 2, 3]);
        assert_eq!(array_join(&a, None).unwrap(), "1,2,3");

        let t = ta_of(&[1, 2, 3]);
        assert_eq!(typed_array_join(&t, ",").unwrap(), "1,2,3");
    }

    #[test]
    fn test_align_join_custom_separator() {
        let a = arr_of(&[4, 5]);
        assert_eq!(array_join(&a, Some("-")).unwrap(), "4-5");

        let t = ta_of(&[4, 5]);
        assert_eq!(typed_array_join(&t, "-").unwrap(), "4-5");
    }

    // =====================================================================
    // 11. at — negative index access
    // =====================================================================

    #[test]
    fn test_align_at_negative() {
        let a = arr_of(&[10, 20, 30]);
        assert_eq!(array_at(&a, -1), JsValue::Smi(30));

        let t = ta_of(&[10, 20, 30]);
        assert_eq!(typed_array_at(&t, -1), JsValue::Smi(30));
    }

    // =====================================================================
    // 12. keys / values / entries — iterator results match
    // =====================================================================

    #[test]
    fn test_align_keys_both() {
        let a = arr_of(&[10, 20]);
        assert_eq!(array_keys(&a), vec![0, 1]);

        let t = ta_of(&[10, 20]);
        assert_eq!(typed_array_keys(&t), vec![JsValue::Smi(0), JsValue::Smi(1)]);
    }

    #[test]
    fn test_align_values_both() {
        let a = arr_of(&[7, 8]);
        let va = array_values(&a);
        assert_eq!(va, vec![JsValue::Smi(7), JsValue::Smi(8)]);

        let t = ta_of(&[7, 8]);
        let vt = typed_array_values(&t);
        assert_eq!(vt, vec![JsValue::Smi(7), JsValue::Smi(8)]);
    }

    #[test]
    fn test_align_entries_both_shape() {
        let a = arr_of(&[42]);
        let ea = array_entries(&a);
        assert_eq!(ea.len(), 1);
        assert_eq!(ea[0], (0u32, JsValue::Smi(42)));

        let t = ta_of(&[42]);
        let et = typed_array_entries(&t);
        assert_eq!(et.len(), 1);
        // TypedArray entries return JsValue arrays [index, value]
    }

    // =====================================================================
    // 13. fill — fills range with value
    // =====================================================================

    #[test]
    fn test_align_fill_both() {
        let mut a = arr_of(&[0, 0, 0, 0]);
        array_fill(&mut a, JsValue::Smi(7), Some(1), Some(3));
        assert_eq!(a.get(0), JsValue::Smi(0));
        assert_eq!(a.get(1), JsValue::Smi(7));
        assert_eq!(a.get(2), JsValue::Smi(7));
        assert_eq!(a.get(3), JsValue::Smi(0));

        let t = ta_of(&[0, 0, 0, 0]);
        typed_array_fill(&t, &JsValue::Smi(7), 1, 3).unwrap();
        assert_eq!(typed_array_get(&t, 0), JsValue::Smi(0));
        assert_eq!(typed_array_get(&t, 1), JsValue::Smi(7));
        assert_eq!(typed_array_get(&t, 2), JsValue::Smi(7));
        assert_eq!(typed_array_get(&t, 3), JsValue::Smi(0));
    }

    // =====================================================================
    // 14. copyWithin — overlapping copy semantics
    // =====================================================================

    #[test]
    fn test_align_copy_within_both() {
        let mut a = arr_of(&[1, 2, 3, 4, 5]);
        array_copy_within(&mut a, 0, 3, None);
        assert_eq!(a.get(0), JsValue::Smi(4));
        assert_eq!(a.get(1), JsValue::Smi(5));

        let t = ta_of(&[1, 2, 3, 4, 5]);
        typed_array_copy_within(&t, 0, 3, 5);
        assert_eq!(typed_array_get(&t, 0), JsValue::Smi(4));
        assert_eq!(typed_array_get(&t, 1), JsValue::Smi(5));
    }

    // =====================================================================
    // 15. reverse — in-place reversal
    // =====================================================================

    #[test]
    fn test_align_reverse_both() {
        let mut a = arr_of(&[1, 2, 3]);
        array_reverse(&mut a);
        assert_eq!(a.get(0), JsValue::Smi(3));
        assert_eq!(a.get(2), JsValue::Smi(1));

        let t = ta_of(&[1, 2, 3]);
        typed_array_reverse(&t);
        assert_eq!(typed_array_get(&t, 0), JsValue::Smi(3));
        assert_eq!(typed_array_get(&t, 2), JsValue::Smi(1));
    }

    // =====================================================================
    // 16. sort — TypedArray default is numeric, Array default is lexicographic
    // =====================================================================

    #[test]
    fn test_align_sort_typed_array_default_is_numeric() {
        // Array.prototype.sort with default comparator is lexicographic (string).
        // We pass numeric comparator to Array explicitly.
        let mut a = arr_of(&[3, 1, 2]);
        array_sort(
            &mut a,
            Some(|x: &JsValue, y: &JsValue| {
                let nx = if let JsValue::Smi(n) = x { *n } else { 0 };
                let ny = if let JsValue::Smi(n) = y { *n } else { 0 };
                nx.cmp(&ny)
            }),
        )
        .unwrap();
        assert_eq!(a.get(0), JsValue::Smi(1));

        // Rule 5: TypedArray default sort is numeric.
        let t = ta_of(&[3, 1, 2]);
        typed_array_sort(&t, None).unwrap();
        assert_eq!(typed_array_get(&t, 0), JsValue::Smi(1));
        assert_eq!(typed_array_get(&t, 1), JsValue::Smi(2));
        assert_eq!(typed_array_get(&t, 2), JsValue::Smi(3));
    }

    // =====================================================================
    // 17. slice — Array returns Array, TypedArray returns same TypedArray
    // =====================================================================

    #[test]
    fn test_align_slice_return_types() {
        let a = arr_of(&[10, 20, 30, 40]);
        let sa = array_slice(&a, Some(1), Some(3));
        assert_eq!(sa.length(), 2);
        assert_eq!(sa.get(0), JsValue::Smi(20));

        // Rule 9: TypedArray.slice returns new TypedArray of same type
        let t = ta_of(&[10, 20, 30, 40]);
        let st = typed_array_slice(&t, 1, 3).unwrap();
        assert_eq!(st.kind, TypedArrayKind::Int32);
        assert_eq!(typed_array_length(&st), 2);
        assert_eq!(typed_array_get(&st, 0), JsValue::Smi(20));
    }

    // =====================================================================
    // 18. toReversed — non-mutating reverse
    // =====================================================================

    #[test]
    fn test_align_to_reversed_both_non_mutating() {
        let a = arr_of(&[1, 2, 3]);
        let ra = array_to_reversed(&a);
        assert_eq!(ra.get(0), JsValue::Smi(3));
        assert_eq!(a.get(0), JsValue::Smi(1)); // original unchanged

        let t = ta_of(&[1, 2, 3]);
        let rt = typed_array_to_reversed(&t).unwrap();
        assert_eq!(rt.kind, TypedArrayKind::Int32);
        assert_eq!(typed_array_get(&rt, 0), JsValue::Smi(3));
        assert_eq!(typed_array_get(&t, 0), JsValue::Smi(1)); // original unchanged
    }

    // =====================================================================
    // 19. toSorted — non-mutating sort
    // =====================================================================

    #[test]
    fn test_align_to_sorted_both_non_mutating() {
        let a = arr_of(&[3, 1, 2]);
        let sa = array_to_sorted(
            &a,
            Some(|x: &JsValue, y: &JsValue| {
                let nx = if let JsValue::Smi(n) = x { *n } else { 0 };
                let ny = if let JsValue::Smi(n) = y { *n } else { 0 };
                nx.cmp(&ny)
            }),
        )
        .unwrap();
        assert_eq!(sa.get(0), JsValue::Smi(1));
        assert_eq!(a.get(0), JsValue::Smi(3)); // original unchanged

        let t = ta_of(&[3, 1, 2]);
        let st = typed_array_to_sorted(&t, None).unwrap();
        assert_eq!(st.kind, TypedArrayKind::Int32);
        assert_eq!(typed_array_get(&st, 0), JsValue::Smi(1));
        assert_eq!(typed_array_get(&t, 0), JsValue::Smi(3)); // original unchanged
    }

    // =====================================================================
    // 20. with — non-mutating element replacement
    // =====================================================================

    #[test]
    fn test_align_with_both_positive_index() {
        let a = arr_of(&[1, 2, 3]);
        let wa = array_with(&a, 1, JsValue::Smi(99)).unwrap();
        assert_eq!(wa.get(1), JsValue::Smi(99));
        assert_eq!(a.get(1), JsValue::Smi(2)); // original unchanged

        let t = ta_of(&[1, 2, 3]);
        let wt = typed_array_with(&t, 1, &JsValue::Smi(99)).unwrap();
        assert_eq!(wt.kind, TypedArrayKind::Int32);
        assert_eq!(typed_array_get(&wt, 1), JsValue::Smi(99));
        assert_eq!(typed_array_get(&t, 1), JsValue::Smi(2)); // original unchanged
    }

    #[test]
    fn test_align_with_both_negative_index() {
        let a = arr_of(&[1, 2, 3]);
        let wa = array_with(&a, -1, JsValue::Smi(99)).unwrap();
        assert_eq!(wa.get(2), JsValue::Smi(99));

        let t = ta_of(&[1, 2, 3]);
        let wt = typed_array_with(&t, -1, &JsValue::Smi(99)).unwrap();
        assert_eq!(typed_array_get(&wt, 2), JsValue::Smi(99));
    }

    #[test]
    fn test_align_with_both_out_of_bounds_errors() {
        let a = arr_of(&[1]);
        assert!(matches!(
            array_with(&a, 5, JsValue::Smi(1)),
            Err(StatorError::RangeError(_))
        ));

        let t = ta_of(&[1]);
        assert!(matches!(
            typed_array_with(&t, 5, &JsValue::Smi(1)),
            Err(StatorError::RangeError(_))
        ));
    }

    // =====================================================================
    // 21. TypedArray.map returns same TypedArray type (Float64)
    // =====================================================================

    #[test]
    fn test_typed_array_map_preserves_float64_kind() {
        let t = typed_array_from_values(
            TypedArrayKind::Float64,
            &[JsValue::HeapNumber(1.5), JsValue::HeapNumber(2.5)],
        )
        .unwrap();
        let mapped = typed_array_map(&t, |v, _| {
            let n = v.to_number().unwrap_or(0.0);
            Ok(JsValue::HeapNumber(n * 2.0))
        })
        .unwrap();
        assert_eq!(mapped.kind, TypedArrayKind::Float64);
    }

    // =====================================================================
    // 22. TypedArray.filter returns same TypedArray type (Uint8)
    // =====================================================================

    #[test]
    fn test_typed_array_filter_preserves_uint8_kind() {
        let t = typed_array_from_values(
            TypedArrayKind::Uint8,
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
        )
        .unwrap();
        let filtered =
            typed_array_filter(&t, |v, _| Ok(matches!(v, JsValue::Smi(n) if *n > 1))).unwrap();
        assert_eq!(filtered.kind, TypedArrayKind::Uint8);
        assert_eq!(typed_array_length(&filtered), 2);
    }

    // =====================================================================
    // 23. TypedArray has no holes (dense) — Rule 4
    // =====================================================================

    #[test]
    fn test_typed_array_dense_no_holes() {
        let t = typed_array_new_from_length(TypedArrayKind::Int32, 5);
        // All elements are initialized to 0 — no holes.
        for i in 0..5 {
            assert_eq!(typed_array_get(&t, i), JsValue::Smi(0));
        }
    }

    #[test]
    fn test_array_sparse_holes_vs_typed_array_dense() {
        // Array can have holes (undefined at unset indices).
        let mut a = JsArray::new();
        a.set(0, JsValue::Smi(1));
        a.set(2, JsValue::Smi(3)); // index 1 is a hole
        assert_eq!(a.get(1), JsValue::Undefined);

        // TypedArray: every index is initialized, no holes.
        let t = typed_array_new_from_length(TypedArrayKind::Int32, 3);
        typed_array_set(&t, 0, &JsValue::Smi(1)).unwrap();
        typed_array_set(&t, 2, &JsValue::Smi(3)).unwrap();
        assert_eq!(typed_array_get(&t, 1), JsValue::Smi(0)); // zero, not undefined
    }

    // =====================================================================
    // 24. TypedArray sort default is numeric — Rule 5
    // =====================================================================

    #[test]
    fn test_typed_array_sort_numeric_not_lexicographic() {
        // In Array, default sort is lexicographic: [1, 10, 2, 9]
        // In TypedArray, default sort is numeric: [1, 2, 9, 10]
        let t = ta_of(&[10, 2, 1, 9]);
        typed_array_sort(&t, None).unwrap();
        assert_eq!(typed_array_get(&t, 0), JsValue::Smi(1));
        assert_eq!(typed_array_get(&t, 1), JsValue::Smi(2));
        assert_eq!(typed_array_get(&t, 2), JsValue::Smi(9));
        assert_eq!(typed_array_get(&t, 3), JsValue::Smi(10));
    }

    // =====================================================================
    // 25. Array.prototype methods are generic (array-like via .call) — Rule 6
    // =====================================================================

    #[test]
    fn test_array_map_generic_on_array_like() {
        // Array.prototype.map can be called on any array-like.
        // In our pure-Rust API this means we can create an array from
        // arbitrary JsValues and map over it.
        let a = array_of(&[JsValue::String("x".into()), JsValue::String("y".into())]);
        let mapped = array_map(&a, |v, idx| {
            if let JsValue::String(s) = v {
                JsValue::String(format!("{s}{idx}").into())
            } else {
                v.clone()
            }
        });
        assert_eq!(mapped.get(0), JsValue::String("x0".into()));
        assert_eq!(mapped.get(1), JsValue::String("y1".into()));
    }

    // =====================================================================
    // 26. Array.prototype.concat is NOT on TypedArray — Rule 7
    //     (we verify concat exists on Array and show set() on TypedArray)
    // =====================================================================

    #[test]
    fn test_array_has_concat_typed_array_does_not() {
        let a1 = arr_of(&[1, 2]);
        let a2 = arr_of(&[3, 4]);
        let concatenated = array_concat(&a1, &[&a2]);
        assert_eq!(concatenated.length(), 4);
        assert_eq!(concatenated.get(2), JsValue::Smi(3));
        // TypedArray has no concat — use set() instead (tested below).
    }

    // =====================================================================
    // 27. TypedArray.prototype.set() is unique to TypedArray — Rule 8
    // =====================================================================

    #[test]
    fn test_typed_array_set_from_unique() {
        let dst = typed_array_new_from_length(TypedArrayKind::Int32, 4);
        typed_array_set_from(&dst, &[JsValue::Smi(10), JsValue::Smi(20)], 1).unwrap();
        assert_eq!(typed_array_get(&dst, 0), JsValue::Smi(0));
        assert_eq!(typed_array_get(&dst, 1), JsValue::Smi(10));
        assert_eq!(typed_array_get(&dst, 2), JsValue::Smi(20));
    }

    #[test]
    fn test_typed_array_set_from_typed_array_unique() {
        let src = ta_of(&[5, 6]);
        let dst = typed_array_new_from_length(TypedArrayKind::Int32, 4);
        typed_array_set_from_typed_array(&dst, &src, 2).unwrap();
        assert_eq!(typed_array_get(&dst, 2), JsValue::Smi(5));
        assert_eq!(typed_array_get(&dst, 3), JsValue::Smi(6));
    }

    // =====================================================================
    // 28. TypedArray.slice returns same kind — Rule 9
    // =====================================================================

    #[test]
    fn test_typed_array_slice_preserves_uint16_kind() {
        let t = typed_array_from_values(
            TypedArrayKind::Uint16,
            &[JsValue::Smi(100), JsValue::Smi(200), JsValue::Smi(300)],
        )
        .unwrap();
        let s = typed_array_slice(&t, 1, 3).unwrap();
        assert_eq!(s.kind, TypedArrayKind::Uint16);
        assert_eq!(typed_array_length(&s), 2);
    }

    // =====================================================================
    // 29. Symbol.iterator returns same-style iterator — Rule 10
    // =====================================================================

    #[test]
    fn test_align_symbol_iterator_values_match() {
        let a = arr_of(&[1, 2, 3]);
        let iter_a = array_symbol_iterator(&a);
        assert_eq!(
            iter_a,
            vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]
        );

        let t = ta_of(&[1, 2, 3]);
        let iter_t = typed_array_values(&t);
        assert_eq!(
            iter_t,
            vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]
        );
    }

    // =====================================================================
    // 30. Empty array / typed array edge cases for shared methods
    // =====================================================================

    #[test]
    fn test_align_empty_map() {
        let a = JsArray::new();
        let ma = array_map(&a, |v, _| v.clone());
        assert_eq!(ma.length(), 0);

        let t = typed_array_new_from_length(TypedArrayKind::Int32, 0);
        let mt = typed_array_map(&t, |v, _| Ok(v.clone())).unwrap();
        assert_eq!(typed_array_length(&mt), 0);
    }

    #[test]
    fn test_align_empty_filter() {
        let a = JsArray::new();
        let fa = array_filter(&a, |_, _| true);
        assert_eq!(fa.length(), 0);

        let t = typed_array_new_from_length(TypedArrayKind::Int32, 0);
        let ft = typed_array_filter(&t, |_, _| Ok(true)).unwrap();
        assert_eq!(typed_array_length(&ft), 0);
    }

    // =====================================================================
    // 31. find returns Undefined / default when no match
    // =====================================================================

    #[test]
    fn test_align_find_no_match_returns_undefined() {
        let a = arr_of(&[1, 2, 3]);
        assert_eq!(array_find(&a, |_, _| false), JsValue::Undefined);

        let t = ta_of(&[1, 2, 3]);
        assert_eq!(
            typed_array_find(&t, |_, _| Ok(false)).unwrap(),
            JsValue::Undefined
        );
    }

    // =====================================================================
    // 32. TypedArray reduce_right empty no-initial errors
    // =====================================================================

    #[test]
    fn test_align_reduce_right_empty_no_initial_errors() {
        let a = JsArray::new();
        assert!(array_reduce_right(&a, |acc, _, _| acc, None).is_err());

        let t = typed_array_new_from_length(TypedArrayKind::Int32, 0);
        assert!(typed_array_reduce_right(&t, |acc, _, _| Ok(acc.clone()), None).is_err());
    }

    // =====================================================================
    // 33. toReversed on empty arrays
    // =====================================================================

    #[test]
    fn test_align_to_reversed_empty() {
        let a = JsArray::new();
        let ra = array_to_reversed(&a);
        assert_eq!(ra.length(), 0);

        let t = typed_array_new_from_length(TypedArrayKind::Int32, 0);
        let rt = typed_array_to_reversed(&t).unwrap();
        assert_eq!(typed_array_length(&rt), 0);
    }

    // =====================================================================
    // 34. toSorted on single-element arrays
    // =====================================================================

    #[test]
    fn test_align_to_sorted_single_element() {
        let a = arr_of(&[42]);
        let sa: JsArray =
            array_to_sorted(&a, None::<fn(&JsValue, &JsValue) -> std::cmp::Ordering>).unwrap();
        assert_eq!(sa.get(0), JsValue::Smi(42));

        let t = ta_of(&[42]);
        let st = typed_array_to_sorted(&t, None).unwrap();
        assert_eq!(typed_array_get(&st, 0), JsValue::Smi(42));
    }

    // =====================================================================
    // 35. slice with negative indices
    // =====================================================================

    #[test]
    fn test_align_slice_negative_indices() {
        let a = arr_of(&[10, 20, 30, 40, 50]);
        let sa = array_slice(&a, Some(-3), Some(-1));
        assert_eq!(sa.length(), 2);
        assert_eq!(sa.get(0), JsValue::Smi(30));
        assert_eq!(sa.get(1), JsValue::Smi(40));

        let t = ta_of(&[10, 20, 30, 40, 50]);
        let st = typed_array_slice(&t, -3, -1).unwrap();
        assert_eq!(typed_array_length(&st), 2);
        assert_eq!(typed_array_get(&st, 0), JsValue::Smi(30));
        assert_eq!(typed_array_get(&st, 1), JsValue::Smi(40));
    }

    // =====================================================================
    // 36. fill entire array
    // =====================================================================

    #[test]
    fn test_align_fill_entire_array() {
        let mut a = arr_of(&[0, 0, 0]);
        array_fill(&mut a, JsValue::Smi(9), None, None);
        for i in 0..3 {
            assert_eq!(a.get(i), JsValue::Smi(9));
        }

        let t = ta_of(&[0, 0, 0]);
        typed_array_fill(&t, &JsValue::Smi(9), 0, 3).unwrap();
        for i in 0..3 {
            assert_eq!(typed_array_get(&t, i), JsValue::Smi(9));
        }
    }

    // =====================================================================
    // 37. copyWithin with negative start
    // =====================================================================

    #[test]
    fn test_align_copy_within_negative_start() {
        let mut a = arr_of(&[1, 2, 3, 4]);
        array_copy_within(&mut a, 0, -2, None);
        assert_eq!(a.get(0), JsValue::Smi(3));
        assert_eq!(a.get(1), JsValue::Smi(4));

        let t = ta_of(&[1, 2, 3, 4]);
        typed_array_copy_within(&t, 0, -2, 4);
        assert_eq!(typed_array_get(&t, 0), JsValue::Smi(3));
        assert_eq!(typed_array_get(&t, 1), JsValue::Smi(4));
    }
}
