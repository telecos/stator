#![no_main]

use libfuzzer_sys::fuzz_target;
use std::ptr::NonNull;
use stator_core::gc::handle::HandleScope;

// Fuzz nested `HandleScope` create/destroy patterns.
//
// The fuzz input drives the number of locals registered in the outer scope
// and whether an inner (child) scope is opened and populated.  After each
// pattern the raw-handle counts are verified to confirm that no handles are
// leaked or double-counted across scope boundaries.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    // Number of locals in the outer scope: capped to avoid excessive memory use.
    let outer_count = (data[0] as usize).min(128);
    // Whether to open a child scope (driven by the second byte if present).
    let open_child = data.len() >= 2 && data[1] & 1 == 1;
    // Number of locals in the child scope.
    let inner_count = if data.len() >= 3 {
        (data[2] as usize).min(64)
    } else {
        0
    };

    let mut unit = ();
    let mut outer = HandleScope::new(&mut unit);

    // Allocate `outer_count` u64 values on the stack and register a local for
    // each.  The values are kept alive for the entire scope duration.
    let mut outer_values: Vec<u64> = (0..outer_count as u64).collect();
    for v in &mut outer_values {
        if let Some(ptr) = NonNull::new(v as *mut u64) {
            // SAFETY: `outer_values` is live for the entire duration of `outer`.
            unsafe { outer.create_local(ptr) };
        }
    }

    assert_eq!(
        outer.raw_handles().count(),
        outer_count,
        "outer scope must track exactly the registered handles"
    );

    if open_child {
        // Open a child scope; this mutably borrows `outer` for the duration.
        {
            let mut child = outer.open_child_scope();

            let mut inner_values: Vec<u64> = (0..inner_count as u64).collect();
            for v in &mut inner_values {
                if let Some(ptr) = NonNull::new(v as *mut u64) {
                    // SAFETY: `inner_values` is live for the entire duration of
                    // `child`.
                    unsafe { child.create_local(ptr) };
                }
            }

            assert_eq!(
                child.raw_handles().count(),
                inner_count,
                "child scope must track exactly its own handles"
            );
            // `child` is dropped here; its blocks are freed.
        }

        // After the child scope is dropped the parent's handles are intact.
        assert_eq!(
            outer.raw_handles().count(),
            outer_count,
            "outer scope handle count must be unchanged after child scope drop"
        );
    }

    // `outer` drops here; all its blocks are freed.
});
