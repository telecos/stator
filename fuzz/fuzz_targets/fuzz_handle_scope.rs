#![no_main]

use std::ptr::NonNull;

use libfuzzer_sys::fuzz_target;
use stator_core::gc::handle::HandleScope;

fuzz_target!(|data: &[u8]| {
    // Interpret the fuzz bytes as a sequence of operations on HandleScopes:
    //
    //  byte & 0x03 == 0 → create a Local in the root scope
    //  byte & 0x03 == 1 → open a child scope, add (byte >> 2) locals, then close it
    //  byte & 0x03 == 2 → open a grandchild scope inside a transient child
    //  other            → ignored
    //
    // Stack values are kept alive in `root_value` and short-lived stack
    // variables inside each arm, so all NonNull pointers remain valid for the
    // duration of the scope that uses them.

    const MAX_HANDLES_PER_SCOPE: usize = 32;

    let mut root_value: u32 = 0;
    let mut unit = ();
    let mut root_scope = HandleScope::new(&mut unit);
    let mut root_handle_count: usize = 0;

    for &byte in data {
        match byte & 0x03 {
            0 => {
                // Add a Local to the root scope.
                if root_handle_count < MAX_HANDLES_PER_SCOPE {
                    let ptr = NonNull::new(&mut root_value as *mut u32).unwrap();
                    // SAFETY: `root_value` outlives `root_scope`.
                    unsafe { root_scope.create_local(ptr) };
                    root_handle_count += 1;
                }
            }
            1 => {
                // Open a child scope, add some locals, then close it.
                let mut child_value: u32 = 0;
                let handles_to_add = (usize::from(byte) >> 2).min(MAX_HANDLES_PER_SCOPE);
                let mut child_scope = root_scope.open_child_scope();
                for _ in 0..handles_to_add {
                    let ptr = NonNull::new(&mut child_value as *mut u32).unwrap();
                    // SAFETY: `child_value` outlives `child_scope`.
                    unsafe { child_scope.create_local(ptr) };
                }
                assert_eq!(
                    child_scope.raw_handles().count(),
                    handles_to_add,
                    "child scope must track exactly the requested number of handles"
                );
                // `child_scope` is dropped here, releasing all child handles.
            }
            2 => {
                // Open a child + grandchild scope pair and verify nesting.
                let mut cv: u32 = 0;
                let mut gv: u32 = 0;
                let mut child = root_scope.open_child_scope();
                let ptr_c = NonNull::new(&mut cv as *mut u32).unwrap();
                // SAFETY: `cv` outlives `child`.
                unsafe { child.create_local(ptr_c) };
                {
                    let mut grand = child.open_child_scope();
                    let ptr_g = NonNull::new(&mut gv as *mut u32).unwrap();
                    // SAFETY: `gv` outlives `grand`.
                    unsafe { grand.create_local(ptr_g) };
                    assert_eq!(grand.raw_handles().count(), 1, "grandchild must have 1 handle");
                    // `grand` dropped here.
                }
                assert_eq!(child.raw_handles().count(), 1, "child must still have 1 handle");
                // `child` dropped here.
            }
            _ => {}
        }
    }

    // Root scope must still hold exactly `root_handle_count` handles.
    assert_eq!(
        root_scope.raw_handles().count(),
        root_handle_count,
        "root scope handle count mismatch after all operations"
    );
});

