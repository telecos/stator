//! Criterion benchmarks for core engine operations.
//!
//! Run with: `cargo bench --package stator_core`

use std::alloc::Layout;
use std::ptr::NonNull;

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

use stator_core::gc::handle::HandleScope;
use stator_core::gc::heap::{Heap, HeapObject};
use stator_core::objects::tagged::TaggedValue;

// ---------------------------------------------------------------------------
// Object allocation throughput
// ---------------------------------------------------------------------------

const BURST_ALLOC_COUNT: usize = 1000;

fn bench_heap_allocate(c: &mut Criterion) {
    let layout = Layout::new::<HeapObject>();
    c.bench_function("heap_allocate_single", |b| {
        // Heap lives across iterations to measure steady-state bump-allocation.
        // When the young space fills up we collect (reset) and keep going.
        let mut heap = Heap::new();
        b.iter(|| {
            let ptr = heap.allocate(black_box(layout));
            if ptr.is_null() {
                heap.collect();
            }
            black_box(ptr);
        });
    });
}

fn bench_heap_allocate_burst(c: &mut Criterion) {
    let layout = Layout::new::<HeapObject>();
    c.bench_function("heap_allocate_burst_1000", |b| {
        b.iter(|| {
            let mut heap = Heap::new();
            for _ in 0..BURST_ALLOC_COUNT {
                let ptr = heap.allocate(black_box(layout));
                black_box(ptr);
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Tagged value operations
// ---------------------------------------------------------------------------

fn bench_tagged_smi_round_trip(c: &mut Criterion) {
    c.bench_function("tagged_smi_round_trip", |b| {
        b.iter(|| {
            let tv = TaggedValue::from_smi(black_box(42));
            black_box(tv.is_smi());
            black_box(tv.as_smi());
        });
    });
}

fn bench_tagged_heap_ptr_round_trip(c: &mut Criterion) {
    let mut obj = HeapObject { meta: 0 };
    let ptr: *mut HeapObject = &mut obj;
    c.bench_function("tagged_heap_ptr_round_trip", |b| {
        b.iter(|| {
            // SAFETY: `ptr` is non-null, aligned, and `obj` is live for the
            // duration of the benchmark.
            let tv = unsafe { TaggedValue::from_heap_ptr(black_box(ptr)) };
            black_box(tv.is_heap_object());
            // SAFETY: `obj` has not been freed.
            black_box(unsafe { tv.as_heap_ptr() });
        });
    });
}

// ---------------------------------------------------------------------------
// HandleScope create / destroy
// ---------------------------------------------------------------------------

const LOCAL_HANDLE_COUNT: usize = 100;

fn bench_handle_scope_create_destroy(c: &mut Criterion) {
    c.bench_function("handle_scope_create_destroy", |b| {
        b.iter(|| {
            let mut isolate = ();
            let scope = HandleScope::new(black_box(&mut isolate));
            black_box(&scope);
            drop(scope);
        });
    });
}

fn bench_handle_scope_create_locals(c: &mut Criterion) {
    c.bench_function("handle_scope_create_100_locals", |b| {
        let mut values: Vec<u64> = (0..LOCAL_HANDLE_COUNT as u64).collect();
        let ptrs: Vec<NonNull<u64>> = values
            .iter_mut()
            .map(|v| NonNull::new(v as *mut u64).unwrap())
            .collect();
        b.iter(|| {
            let mut isolate = ();
            let mut scope = HandleScope::new(&mut isolate);
            for &ptr in &ptrs {
                // SAFETY: each pointer is live for the benchmark duration.
                let local = unsafe { scope.create_local(ptr) };
                black_box(local);
            }
            black_box(scope.raw_handles().count());
        });
    });
}

criterion_group!(
    benches,
    bench_heap_allocate,
    bench_heap_allocate_burst,
    bench_tagged_smi_round_trip,
    bench_tagged_heap_ptr_round_trip,
    bench_handle_scope_create_destroy,
    bench_handle_scope_create_locals,
);
criterion_main!(benches);
