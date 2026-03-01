#![no_main]

use libfuzzer_sys::fuzz_target;
use std::alloc::Layout;
use stator_core::gc::heap::Heap;

// Fuzz the heap allocator with random allocation sizes and sequences.
//
// Each pair of bytes is interpreted as a (size, align_shift) descriptor.
// After processing all descriptors a minor GC is triggered and the heap's
// invariants are verified to ensure that the allocator and collector remain
// consistent regardless of the allocation sequence.
fuzz_target!(|data: &[u8]| {
    let mut heap = Heap::new();

    let mut cursor = 0;
    while cursor + 1 < data.len() {
        // Map the raw byte to a non-zero allocation size in [1, 256].
        let size = (data[cursor] as usize) + 1;
        // Map the align byte to a valid power-of-two alignment: 1, 2, 4, or 8.
        let align_shift = data[cursor + 1] & 0x3;
        let align = 1usize << align_shift;
        cursor += 2;

        if let Ok(layout) = Layout::from_size_align(size, align) {
            // Allocate; the heap may return null if nursery + GC-retry both
            // fail (size > semi-space capacity), which is a valid outcome.
            let _ = heap.allocate(layout);
        }
    }

    // Trigger a minor collection and verify post-collection invariants.
    heap.collect();

    assert!(
        heap.young_space.used() <= heap.young_space.capacity(),
        "young-space used must never exceed capacity after collection"
    );
    assert!(
        heap.old_space.used() <= heap.old_space.capacity(),
        "old-space used must never exceed capacity"
    );
});
