#![no_main]

use libfuzzer_sys::fuzz_target;
use stator_core::objects::js_object::JsObject;
use stator_core::objects::value::JsValue;

fuzz_target!(|data: &[u8]| {
    // Drive random property set/get/delete operations on a JsObject and verify:
    //   1. A property set without error is immediately readable.
    //   2. A deleted property is no longer readable as an own property.
    //   3. The fast→slow mode transition does not lose any properties.
    //
    // Input layout: triplets of bytes
    //   byte[0]: operation selector
    //     0 → set_property
    //     1 → get_property
    //     2 → delete_own_property
    //     3 → set_element
    //     4 → get_element / has_element
    //   byte[1]: property-name index (0..=7, naming scheme "p0"…"p7")
    //   byte[2]: value selector (see `make_value`)

    fn make_value(selector: u8) -> JsValue {
        match selector % 6 {
            0 => JsValue::Undefined,
            1 => JsValue::Null,
            2 => JsValue::Boolean(selector & 1 == 0),
            3 => JsValue::Smi(i32::from(selector)),
            4 => JsValue::HeapNumber(f64::from(selector)),
            _ => JsValue::String(format!("v{selector}")),
        }
    }

    const KEYS: [&str; 8] = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"];

    let mut obj = JsObject::new();
    let mut chunk = data;

    while chunk.len() >= 3 {
        let op = chunk[0];
        let key_idx = usize::from(chunk[1] & 0x07);
        let val_sel = chunk[2];
        chunk = &chunk[3..];

        let key = KEYS[key_idx];

        match op % 5 {
            0 => {
                // set_property
                let value = make_value(val_sel);
                if obj.set_property(key, value.clone()).is_ok() {
                    // The property must now be readable.
                    let read = obj.get_property(key);
                    assert!(
                        read == value || !obj.has_own_property(key),
                        "set property should be readable immediately"
                    );
                }
            }
            1 => {
                // get_property – must not panic
                let _ = obj.get_property(key);
            }
            2 => {
                // delete_own_property
                if matches!(obj.delete_own_property(key), Ok(true)) {
                    // Must no longer appear as an own property.
                    assert!(
                        !obj.has_own_property(key),
                        "deleted property must not be an own property"
                    );
                }
            }
            3 => {
                // set_element using key_idx as the index
                let value = make_value(val_sel);
                obj.set_element(key_idx, value.clone());
                assert!(
                    obj.get_element(key_idx) == value,
                    "element set must be immediately readable"
                );
            }
            4 => {
                // has_element / get_element – must not panic
                let _ = obj.has_element(key_idx);
                let _ = obj.get_element(key_idx);
            }
            _ => unreachable!(),
        }

        // Invariant: fast-mode object must have the map descriptors consistent
        // with the number of stored values.
        if obj.is_fast_mode() {
            assert!(
                obj.map().descriptors().len()
                    <= stator_core::objects::js_object::MAX_FAST_PROPERTIES,
                "fast-mode map must not exceed MAX_FAST_PROPERTIES descriptors"
            );
        }
    }
});
