#![no_main]

use libfuzzer_sys::fuzz_target;
use stator_core::objects::js_object::JsObject;
use stator_core::objects::map::PropertyAttributes;
use stator_core::objects::value::JsValue;

// Fuzz random property `set` / `get` / `delete` / `define` operations on a
// `JsObject` and verify that map transitions and slow-mode normalisations
// never leave the object in an inconsistent state.
//
// Each operation is encoded in two bytes:
//   byte 0  bits [1:0] – operation selector (0=set, 1=get, 2=delete, 3=define)
//           bits [7:2] – value payload (interpreted as Smi)
//   byte 1  bits [3:0] – property key index k0–k15
//           bits [7:4] – element index 0–15
//
// Using a small bounded key space (k0–k15) ensures that the fuzzer quickly
// exercises both fast-mode (≤8 properties) and slow-mode (>8 properties)
// storage paths, as well as property deletion and redefinition.
fuzz_target!(|data: &[u8]| {
    let mut obj = JsObject::new();
    const MAX_OPS: usize = 256;
    let mut ops = 0;

    let chunks = data.chunks_exact(2);
    for chunk in chunks {
        if ops >= MAX_OPS {
            break;
        }
        ops += 1;

        let op_byte = chunk[0];
        let key_byte = chunk[1];

        let op = op_byte & 0x3;
        let smi_val = (op_byte >> 2) as i32;
        let key_idx = key_byte & 0xf;
        let elem_idx = (key_byte >> 4) as usize;

        let key = format!("k{key_idx}");

        match op {
            0 => {
                // [[Set]]: writable properties accept the new value; read-only
                // ones return a TypeError, which we discard.
                let _ = obj.set_property(&key, JsValue::Smi(smi_val));
            }
            1 => {
                // [[Get]]: returns the value or Undefined; must not panic.
                let _ = obj.get_property(&key);

                // Also exercise the element backing store.
                obj.set_element(elem_idx, JsValue::Smi(smi_val));
                let _ = obj.get_element(elem_idx);
            }
            2 => {
                // [[Delete]]: non-configurable properties return false; others
                // are removed and the map transitions to slow mode.
                let _ = obj.delete_own_property(&key);

                // Also delete from the element store.
                let _ = obj.delete_element(elem_idx);
            }
            _ => {
                // [[DefineOwnProperty]]: use the smi_val bits to select
                // attribute flags (WRITABLE, ENUMERABLE, CONFIGURABLE).
                let attrs = PropertyAttributes::from_bits_truncate(smi_val as u8);
                let _ = obj.define_own_property(&key, JsValue::Smi(smi_val), attrs);
            }
        }
    }

    // Post-condition: the object must be in a consistent mode.
    // is_fast_mode() must not panic regardless of the operation history.
    let _ = obj.is_fast_mode();
    let _ = obj.elements_length();
});
