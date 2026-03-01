#![no_main]

use std::mem::align_of;
use std::mem::size_of;

use libfuzzer_sys::fuzz_target;
use stator_core::gc::heap::{Heap, HeapObject};

fuzz_target!(|data: &[u8]| {
    // Fuzz the GC under stress: allocate objects in configurable burst sizes,
    // then collect repeatedly to verify no crash occurs and the heap cursor
    // invariants hold.
    //
    // Input layout (each byte drives one action):
    //   even byte n → allocate a burst of (n & 0x0F)+1 small objects
    //   odd  byte n → trigger (n & 0x07)+1 minor GC cycles

    const SMALL_SIZE: usize = size_of::<HeapObject>() * 4;

    let mut heap = Heap::new();
    let small_layout =
        std::alloc::Layout::from_size_align(SMALL_SIZE, align_of::<HeapObject>()).expect("layout");

    for (i, &byte) in data.iter().enumerate() {
        if i % 2 == 0 {
            // Burst allocation.
            let burst = usize::from(byte & 0x0F) + 1;
            for _ in 0..burst {
                // We don't dereference the pointer; we only check heap-level
                // invariants.  A null return means the young space is full
                // and the retry after an implicit GC also failed.
                let ptr = heap.allocate(small_layout);
                if !ptr.is_null() {
                    assert_eq!(
                        ptr as usize % align_of::<HeapObject>(),
                        0,
                        "allocation must be HeapObject-aligned"
                    );
                }
                assert!(
                    heap.young_space.used() <= heap.young_space.capacity(),
                    "young-space used must not exceed capacity"
                );
                assert!(
                    heap.old_space.used() <= heap.old_space.capacity(),
                    "old-space used must not exceed capacity"
                );
            }
        } else {
            // GC cycle burst.
            let cycles = usize::from(byte & 0x07) + 1;
            for _ in 0..cycles {
                heap.collect();
                assert_eq!(
                    heap.young_space.used(),
                    0,
                    "young-space must be empty after minor GC"
                );
            }
        }
    }

    // Final GC to confirm the heap is in a consistent state.
    heap.collect();
    assert!(
        heap.young_space.used() <= heap.young_space.capacity(),
        "young-space invariant must hold after final GC"
    );
});
