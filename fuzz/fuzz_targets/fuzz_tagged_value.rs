#![no_main]

use libfuzzer_sys::fuzz_target;
use stator_core::objects::tagged::TaggedValue;

fuzz_target!(|data: &[u8]| {
    // We need at least 1 byte to drive the test.
    if data.is_empty() {
        return;
    }

    // Use the first byte to choose the operation, and the remaining bytes as
    // a raw u64 payload.
    let op = data[0];
    let mut raw = [0u8; 8];
    let payload = &data[1..];
    let copy_len = payload.len().min(8);
    raw[..copy_len].copy_from_slice(&payload[..copy_len]);
    let bits = u64::from_le_bytes(raw);

    if op & 1 == 0 {
        // Smi round-trip: encode a 32-bit integer and verify the decoded
        // value matches the original.
        let smi_val = bits as i32;
        let tv = TaggedValue::from_smi(smi_val);
        assert!(tv.is_smi(), "expected Smi tag");
        assert!(!tv.is_heap_object(), "Smi must not be a heap-object");
        let recovered = tv.as_smi().expect("Smi decode must succeed");
        assert_eq!(recovered, smi_val, "Smi round-trip mismatch");
    } else {
        // HeapObject pointer encoding: we need a non-null, even address.
        // Manufacture one from the raw bits by aligning to 2 bytes.
        let addr = (bits as usize | 2) & !1usize; // ensure bit 0 == 0, non-zero
        if addr == 0 {
            return;
        }
        // Construct a TaggedValue directly from the raw bits (no heap
        // dereference; we only exercise the tag/mask logic).
        let tv = TaggedValue(addr);
        assert!(!tv.is_smi(), "even address must not be Smi-tagged");
        assert!(tv.is_heap_object(), "even address must be a heap-object");
        assert_eq!(tv.raw(), addr, "raw bits must round-trip");
    }
});
