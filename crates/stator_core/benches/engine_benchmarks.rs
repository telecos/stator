use std::alloc::Layout;
use std::ptr::NonNull;

use criterion::{Criterion, criterion_group, criterion_main};
use stator_core::gc::handle::HandleScope;
use stator_core::gc::heap::{Heap, HeapObject};
use stator_core::objects::tagged::TaggedValue;

// ---------------------------------------------------------------------------
// Object allocation throughput
// ---------------------------------------------------------------------------

fn bench_allocate_small_objects(c: &mut Criterion) {
    let layout = Layout::new::<HeapObject>();
    c.bench_function("heap_allocate_small_object", |b| {
        let mut heap = Heap::new();
        b.iter(|| {
            let ptr = heap.allocate(layout);
            if ptr.is_null() {
                heap.collect();
            }
            ptr
        });
    });
}

// ---------------------------------------------------------------------------
// Tagged value operations
// ---------------------------------------------------------------------------

fn bench_tagged_value_operations(c: &mut Criterion) {
    c.bench_function("tagged_value_from_smi", |b| {
        b.iter(|| TaggedValue::from_smi(42));
    });

    c.bench_function("tagged_value_as_smi", |b| {
        let tv = TaggedValue::from_smi(42);
        b.iter(|| tv.as_smi());
    });

    c.bench_function("tagged_value_is_smi", |b| {
        let tv = TaggedValue::from_smi(42);
        b.iter(|| tv.is_smi());
    });

    c.bench_function("tagged_value_heap_ptr_round_trip", |b| {
        let mut heap = Heap::new();
        let layout = Layout::new::<HeapObject>();
        let ptr = heap.allocate(layout);
        assert!(!ptr.is_null());
        b.iter(|| {
            let tv = unsafe { TaggedValue::from_heap_ptr(ptr) };
            unsafe { tv.as_heap_ptr() }
        });
    });
}

// ---------------------------------------------------------------------------
// HandleScope create / destroy
// ---------------------------------------------------------------------------

fn bench_handle_scope_lifecycle(c: &mut Criterion) {
    c.bench_function("handle_scope_create_destroy", |b| {
        b.iter(|| {
            let mut isolate = ();
            let _scope = HandleScope::new(&mut isolate);
        });
    });

    c.bench_function("handle_scope_create_local", |b| {
        let mut isolate = ();
        let mut value: u32 = 99;
        let ptr = NonNull::new(&mut value as *mut u32).unwrap();
        b.iter(|| {
            let mut scope = HandleScope::new(&mut isolate);
            let local = unsafe { scope.create_local(ptr) };
            std::hint::black_box(local.as_ptr());
        });
    });
}

// ---------------------------------------------------------------------------
// Group & main
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_allocate_small_objects,
    bench_tagged_value_operations,
    bench_handle_scope_lifecycle,
);
criterion_main!(benches);
