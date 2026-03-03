#![no_main]

use libfuzzer_sys::fuzz_target;
use stator_core::builtins::array::{
    array_concat, array_every, array_filter, array_find, array_find_index, array_for_each,
    array_from, array_includes, array_index_of, array_join, array_map, array_pop, array_push,
    array_reduce, array_reverse, array_shift, array_slice, array_some, array_sort, array_splice,
    array_unshift,
};
use stator_core::objects::js_array::JsArray;
use stator_core::objects::value::JsValue;

/// Build a [`JsValue`] leaf from a single byte selector.
fn make_value(sel: u8) -> JsValue {
    match sel % 6 {
        0 => JsValue::Undefined,
        1 => JsValue::Null,
        2 => JsValue::Boolean(sel & 1 == 0),
        3 => JsValue::Smi(i32::from(sel)),
        4 => JsValue::HeapNumber(f64::from(sel)),
        _ => JsValue::String(format!("s{sel}")),
    }
}

/// Build a [`JsArray`] from a slice of bytes: each byte becomes one element.
fn make_array(bytes: &[u8]) -> JsArray {
    array_from(bytes.iter().copied().map(make_value))
}

fuzz_target!(|data: &[u8]| {
    // Need at least 2 bytes: one operation selector and at least one element byte.
    if data.len() < 2 {
        return;
    }

    let op = data[0];
    let rest = &data[1..];

    // Cap array size to keep execution time bounded.
    let max_elems = 32usize;
    let elem_bytes = &rest[..rest.len().min(max_elems)];
    let mut arr = make_array(elem_bytes);
    let initial_len = arr.length();

    match op % 20 {
        0 => {
            // push: append all remaining bytes as elements
            let vals: Vec<JsValue> = rest.iter().take(8).copied().map(make_value).collect();
            let new_len = array_push(&mut arr, &vals);
            // Invariant: new length >= initial length
            assert!(new_len >= initial_len);
        }
        1 => {
            // pop: removes last element
            let _popped = array_pop(&mut arr);
            // Invariant: length decrements (or stays 0)
            let after = arr.length();
            if initial_len == 0 {
                assert_eq!(after, 0);
            } else {
                assert_eq!(after, initial_len - 1);
            }
        }
        2 => {
            // shift: removes first element
            let _shifted = array_shift(&mut arr);
        }
        3 => {
            // unshift: prepend elements
            if rest.len() >= 2 {
                let vals: Vec<JsValue> =
                    rest.iter().take(4).copied().map(make_value).collect();
                let new_len = array_unshift(&mut arr, &vals);
                assert!(new_len >= initial_len);
            }
        }
        4 => {
            // splice: delete some, insert some
            if rest.len() >= 2 {
                let start = i64::from(rest[0]) - 16;
                let del = Some(u32::from(rest[1] % 4));
                let items: Vec<JsValue> = rest.iter().skip(2).take(4).copied().map(make_value).collect();
                let _deleted = array_splice(&mut arr, start, del, &items);
            }
        }
        5 => {
            // map: transform each element
            let _mapped = array_map(&arr, |v, _| v.clone());
        }
        6 => {
            // filter: keep all
            let _filtered = array_filter(&arr, |_, _| true);
        }
        7 => {
            // reduce: sum Smis
            let _ = array_reduce(
                &arr,
                |acc, v, _| match (acc, v) {
                    (JsValue::Smi(a), JsValue::Smi(b)) => {
                        JsValue::Smi(a.saturating_add(*b))
                    }
                    _ => JsValue::Smi(0),
                },
                Some(JsValue::Smi(0)),
            );
        }
        8 => {
            // forEach: iterate
            array_for_each(&arr, |_, _| {});
        }
        9 => {
            // find / findIndex
            let _found = array_find(&arr, |_, _| false);
            let _idx = array_find_index(&arr, |_, _| false);
        }
        10 => {
            // some / every
            let _s = array_some(&arr, |_, _| false);
            let _e = array_every(&arr, |_, _| true);
        }
        11 => {
            // includes / indexOf
            let needle = JsValue::Smi(42);
            let _inc = array_includes(&arr, &needle, None);
            let _io = array_index_of(&arr, &needle, None);
        }
        12 => {
            // concat with a copy of itself
            let other = make_array(elem_bytes);
            let _c = array_concat(&arr, &[&other]);
        }
        13 => {
            // slice
            if rest.len() >= 2 {
                let start = i64::from(rest[0]) - 16;
                let end = i64::from(rest[1]) - 16;
                let _s = array_slice(&arr, Some(start), Some(end));
            }
        }
        14 => {
            // join
            let _ = array_join(&arr, Some(","));
        }
        15 => {
            // reverse: length must be unchanged
            array_reverse(&mut arr);
            assert_eq!(arr.length(), initial_len);
        }
        16 => {
            // sort (default string comparator)
            let _ = array_sort(&mut arr, None::<fn(&JsValue, &JsValue) -> std::cmp::Ordering>);
            assert_eq!(arr.length(), initial_len);
        }
        17 => {
            // sort with custom comparator (sort by string representation length)
            let _ = array_sort(&mut arr, Some(|a: &JsValue, b: &JsValue| {
                let sa = a.to_js_string().unwrap_or_default();
                let sb = b.to_js_string().unwrap_or_default();
                sa.len().cmp(&sb.len())
            }));
            assert_eq!(arr.length(), initial_len);
        }
        18 => {
            // push then pop: net length must be unchanged
            array_push(&mut arr, &[JsValue::Smi(0)]);
            array_pop(&mut arr);
            assert_eq!(arr.length(), initial_len);
        }
        _ => {
            // unshift then shift: net length must be unchanged
            array_unshift(&mut arr, &[JsValue::Smi(0)]);
            array_shift(&mut arr);
            assert_eq!(arr.length(), initial_len);
        }
    }
});
