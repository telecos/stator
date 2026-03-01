#![no_main]

use libfuzzer_sys::fuzz_target;
use std::alloc::Layout;
use stator_core::gc::heap::Heap;

// Fuzz the GC under stress: interleave random allocations with minor
// collection cycles and verify heap invariants after every operation.
//
// Each byte of the fuzz input encodes one operation:
//   0 – allocate a small object (8–136 bytes, 8-byte aligned)
//   1 – trigger `heap.collect()` (minor GC)
//   2 – verify heap invariants inline
//   3 – allocate then immediately collect
//
// The upper nibble of allocation bytes selects the size bucket so that the
// fuzzer can explore both tiny and moderate allocations.
fuzz_target!(|data: &[u8]| {
    let mut heap = Heap::new();
    // Cap the number of operations to keep individual runs bounded.
    const MAX_OPS: usize = 256;

    let mut cursor = 0;
    let mut ops = 0;

    while cursor < data.len() && ops < MAX_OPS {
        let byte = data[cursor];
        cursor += 1;
        ops += 1;

        let op = byte & 0x3;
        // Use bits 2–7 to select an object size in [8, 136] bytes.
        let size = ((byte >> 2) as usize) * 2 + 8;

        match op {
            0 => {
                // Allocate; null return (OOM) is acceptable.
                if let Ok(layout) = Layout::from_size_align(size, 8) {
                    let _ = heap.allocate(layout);
                }
            }
            1 => {
                heap.collect();
            }
            2 => {
                // Invariant check: used must not exceed capacity.
                assert!(
                    heap.young_space.used() <= heap.young_space.capacity(),
                    "young-space invariant violated"
                );
                assert!(
                    heap.old_space.used() <= heap.old_space.capacity(),
                    "old-space invariant violated"
                );
            }
            _ => {
                // Allocate then collect.
                if let Ok(layout) = Layout::from_size_align(size, 8) {
                    let _ = heap.allocate(layout);
                }
                heap.collect();
            }
        }
    }

    // Final collection: the young space must be empty afterwards.
    heap.collect();
    assert_eq!(
        heap.young_space.used(),
        0,
        "young space must be empty after a final collection"
    );
    assert!(
        heap.old_space.used() <= heap.old_space.capacity(),
        "old-space invariant must hold after final collection"
    );
});
