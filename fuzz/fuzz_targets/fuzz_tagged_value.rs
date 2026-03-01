#![no_main]

use libfuzzer_sys::fuzz_target;
use stator_core::objects::tagged::TaggedValue;

// Fuzz the `TaggedValue` encoding/decoding for both Smi and raw-bit paths.
//
// For the Smi path we verify that `from_smi` followed by `as_smi` never
// panics and that the round-trip is consistent with the documented 31-bit
// encoding contract.  For the raw-bits path we exercise all the predicate
// methods (`is_smi`, `is_heap_object`, `raw`) on arbitrary bit patterns to
// confirm they are free of panics or undefined behaviour.
fuzz_target!(|data: &[u8]| {
    // ── Smi round-trip ────────────────────────────────────────────────────
    if data.len() >= 4 {
        let raw_i32 = i32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let tv = TaggedValue::from_smi(raw_i32);
        assert!(tv.is_smi(), "from_smi must produce a Smi-tagged value");
        assert!(
            !tv.is_heap_object(),
            "a Smi must not also appear as a heap object"
        );
        let decoded = tv.as_smi().expect("is_smi guarantees Some from as_smi");

        // The encoding shifts the i32 left by 1 bit (SMI_SHIFT = 1), so only
        // 31 bits of the original value are preserved.  Verify the round-trip
        // is stable: encoding the decoded value again must reproduce the same
        // TaggedValue.
        let tv2 = TaggedValue::from_smi(decoded);
        assert_eq!(
            tv2, tv,
            "re-encoding the decoded Smi must reproduce the original TaggedValue"
        );
    }

    // ── Raw-bits predicate exercise ───────────────────────────────────────
    if data.len() >= 8 {
        let raw_usize =
            usize::from_le_bytes(data[..8].try_into().expect("slice is exactly 8 bytes"));
        let tv = TaggedValue(raw_usize);

        // These must not panic for any bit pattern.
        let is_smi = tv.is_smi();
        let is_heap = tv.is_heap_object();
        let raw = tv.raw();

        // The two predicates must be exact complements.
        assert_eq!(
            is_smi, !is_heap,
            "is_smi and is_heap_object must be mutually exclusive and exhaustive"
        );
        assert_eq!(raw, raw_usize, "raw() must return the stored bits unchanged");

        // as_smi must be consistent with is_smi.
        match tv.as_smi() {
            Some(_) => assert!(is_smi, "as_smi returned Some but is_smi is false"),
            None => assert!(!is_smi, "as_smi returned None but is_smi is true"),
        }
    }
});
