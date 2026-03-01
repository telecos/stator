#![no_main]

use std::mem::align_of;
use std::mem::size_of;

use libfuzzer_sys::fuzz_target;
use stator_core::gc::heap::{Heap, HeapObject};

fuzz_target!(|data: &[u8]| {
    // Each pair of bytes encodes one allocation request:
    //   byte[0]: high byte of size (0..=255)
    //   byte[1]: low byte of size (0..=255)
    // The effective size is capped to avoid trivially-huge allocations.
    const MAX_ALLOC: usize = 512;
    const ALLOC_HEADER: usize = size_of::<HeapObject>();

    if data.len() < 2 {
        return;
    }

    let mut heap = Heap::new();
    let mut chunk = data;
    let mut alloc_count = 0usize;

    while chunk.len() >= 2 {
        let requested = (((chunk[0] as usize) << 8) | (chunk[1] as usize)).max(ALLOC_HEADER);
        let size = requested.min(MAX_ALLOC);
        chunk = &chunk[2..];

        let layout =
            std::alloc::Layout::from_size_align(size, align_of::<HeapObject>()).expect("layout");
        let ptr = heap.allocate(layout);
        if !ptr.is_null() {
            // Verify pointer alignment invariant.
            assert_eq!(
                ptr as usize % align_of::<HeapObject>(),
                0,
                "allocated pointer must be HeapObject-aligned"
            );
            alloc_count += 1;
        }

        // Every 16 allocations trigger an explicit minor collection so we
        // exercise the GC path rather than just the bump allocator.
        if alloc_count % 16 == 0 {
            heap.collect();
            // After a minor GC the young from-space cursor must be reset.
            assert!(
                heap.young_space.used() <= heap.young_space.capacity(),
                "young-space used must not exceed capacity after GC"
            );
        }
    }

    // Final collection to verify the heap is in a consistent state.
    heap.collect();
    assert_eq!(
        heap.young_space.used(),
        0,
        "young-space must be empty after final collection"
    );
});
