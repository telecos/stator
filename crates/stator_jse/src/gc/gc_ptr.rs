//! GC-managed typed pointer ([`GcPtr<T>`]) and the [`GcObject`] trait.
//!
//! `GcPtr<T>` is the primary interface for working with objects allocated on
//! the garbage-collected heap.  It wraps a non-null raw pointer to a `T` that
//! was allocated through [`Heap::alloc`][crate::gc::heap::Heap::alloc] and
//! whose first field (by `#[repr(C)]` layout) is a
//! [`HeapObject`][crate::objects::heap_object::HeapObject] header.
//!
//! # Rooting
//!
//! A `GcPtr<T>` does **not** by itself keep the object alive.  Before a GC
//! cycle can occur, the pointer must be *rooted* — either by registering it
//! with a [`HandleScope`][crate::gc::handle::HandleScope] (producing a
//! [`Local`][crate::gc::handle::Local]) or by adding it to
//! [`PersistentRoots`][crate::gc::handle::PersistentRoots].
//!
//! # Safety model
//!
//! The [`GcObject`] trait is `unsafe` because its implementor must guarantee
//! `#[repr(C)]` layout with `HeapObject` as the first field.  All access
//! through `GcPtr` is `unsafe` because the GC may relocate or collect the
//! object if it is not rooted.

use std::fmt;
use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::gc::trace::{Trace, Tracer};
use crate::objects::heap_object::HeapObject;

/// Marker trait for types that can be allocated on the GC heap.
///
/// # Safety
///
/// Implementors **must** satisfy all of the following:
///
/// 1. The type is `#[repr(C)]` with [`HeapObject`] as its **first** field, so
///    that a `*mut Self` can be safely cast to `*mut HeapObject`.
/// 2. The type implements [`Trace`], correctly reporting every outgoing heap
///    pointer to the tracer.
/// 3. The type is `Sized` (DSTs are not supported by the bump allocator).
///
/// These invariants are not (and cannot be) checked by the compiler, which is
/// why the trait is `unsafe`.
pub unsafe trait GcObject: Trace + Sized {
    /// Returns a shared reference to this object's [`HeapObject`] header.
    ///
    /// The default implementation casts `self` to `*const HeapObject`, which
    /// is valid because of the `#[repr(C)]`-first-field invariant.
    fn header(&self) -> &HeapObject {
        // SAFETY: GcObject contract guarantees HeapObject is the first field.
        unsafe { &*(self as *const Self as *const HeapObject) }
    }

    /// Returns a mutable reference to this object's [`HeapObject`] header.
    ///
    /// The default implementation casts `self` to `*mut HeapObject`, which
    /// is valid because of the `#[repr(C)]`-first-field invariant.
    fn header_mut(&mut self) -> &mut HeapObject {
        // SAFETY: GcObject contract guarantees HeapObject is the first field.
        unsafe { &mut *(self as *mut Self as *mut HeapObject) }
    }
}

/// A non-null pointer to a GC-managed heap object of type `T`.
///
/// `GcPtr<T>` is a lightweight, `Copy`able wrapper around a [`NonNull<T>`].
/// It does not participate in reference counting or prevent garbage collection
/// on its own — the caller must ensure the pointer is rooted before a GC
/// cycle can run.
///
/// # Memory layout
///
/// `GcPtr<T>` is `#[repr(transparent)]` over `NonNull<T>`, so it has the same
/// size and alignment as a raw pointer.
///
/// # Thread safety
///
/// `GcPtr` is `!Send` and `!Sync` by default (the GC heap is
/// single-threaded).
#[repr(transparent)]
pub struct GcPtr<T: GcObject> {
    ptr: NonNull<T>,
    _marker: PhantomData<*mut T>,
}

// Manual impls because the derive macro would require T: Copy / T: Clone.
impl<T: GcObject> Copy for GcPtr<T> {}

impl<T: GcObject> Clone for GcPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: GcObject> GcPtr<T> {
    /// Create a `GcPtr` from a [`NonNull<T>`].
    ///
    /// # Safety
    ///
    /// * `ptr` must point to a live, GC-allocated `T` whose `HeapObject`
    ///   header has been initialised (at minimum `alloc_size` must be set).
    /// * The allocation must not have been freed or relocated since `ptr` was
    ///   obtained.
    pub unsafe fn from_raw(ptr: NonNull<T>) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// Return the underlying raw pointer.
    pub fn as_ptr(self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Return a shared reference to the pointed-to value.
    ///
    /// # Safety
    ///
    /// The object must be rooted (i.e. reachable from the GC root set) and
    /// must not have been relocated by a GC cycle since this `GcPtr` was
    /// created or last updated.
    pub unsafe fn as_ref(&self) -> &T {
        // SAFETY: caller guarantees the object is live and un-relocated.
        unsafe { self.ptr.as_ref() }
    }

    /// Return a mutable reference to the pointed-to value.
    ///
    /// # Safety
    ///
    /// In addition to the requirements of [`as_ref`](Self::as_ref), the caller
    /// must ensure no other references (shared or mutable) to this object
    /// exist.
    pub unsafe fn as_mut(&mut self) -> &mut T {
        // SAFETY: caller guarantees exclusive access and liveness.
        unsafe { self.ptr.as_mut() }
    }

    /// Access the [`HeapObject`] header of the pointed-to object.
    ///
    /// # Safety
    ///
    /// Same as [`as_ref`](Self::as_ref).
    pub unsafe fn header(&self) -> &HeapObject {
        // SAFETY: GcObject guarantees HeapObject is the first repr(C) field;
        // caller guarantees liveness.
        unsafe { &*(self.ptr.as_ptr() as *const HeapObject) }
    }

    /// Cast the `GcPtr<T>` to a raw `*mut HeapObject`.
    ///
    /// Useful for passing the pointer into low-level GC routines that operate
    /// on untyped `HeapObject` pointers.
    pub fn as_heap_object_ptr(self) -> *mut HeapObject {
        self.ptr.as_ptr() as *mut HeapObject
    }

    /// Construct a `GcPtr<T>` from a `*mut HeapObject`.
    ///
    /// # Safety
    ///
    /// * `ptr` must be non-null, properly aligned, and point to a live `T`
    ///   (not just a `HeapObject`).
    /// * The `HeapObject` header must have been initialised.
    pub unsafe fn from_heap_object_ptr(ptr: *mut HeapObject) -> Self {
        Self {
            // SAFETY: caller guarantees ptr is non-null.
            ptr: unsafe { NonNull::new_unchecked(ptr as *mut T) },
            _marker: PhantomData,
        }
    }
}

impl<T: GcObject> fmt::Debug for GcPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GcPtr({:p})", self.ptr.as_ptr())
    }
}

impl<T: GcObject> fmt::Pointer for GcPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr.as_ptr(), f)
    }
}

impl<T: GcObject> PartialEq for GcPtr<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl<T: GcObject> Eq for GcPtr<T> {}

impl<T: GcObject> std::hash::Hash for GcPtr<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ptr.hash(state);
    }
}

// ── Trace implementation ──────────────────────────────────────────────────────

impl<T: GcObject> Trace for GcPtr<T> {
    fn trace(&self, tracer: &mut Tracer) {
        // SAFETY: GcPtr is always non-null; the tracer will mark the
        // underlying HeapObject for further scanning.
        unsafe { tracer.mark_raw(self.ptr.as_ptr() as *mut u8) };
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gc::trace::Tracer;
    use crate::objects::heap_object::HeapObject;
    use std::alloc::Layout;
    use std::mem::align_of;

    /// A minimal `#[repr(C)]` GC object for testing.
    #[repr(C)]
    struct TestObj {
        header: HeapObject,
        value: u64,
    }

    impl Trace for TestObj {
        fn trace(&self, _tracer: &mut Tracer) {
            // Leaf object: no outgoing heap references.
        }
    }

    // SAFETY: TestObj is #[repr(C)] with HeapObject as the first field.
    unsafe impl GcObject for TestObj {}

    fn make_test_obj(value: u64) -> TestObj {
        TestObj {
            header: HeapObject::new_null(),
            value,
        }
    }

    #[test]
    fn gc_ptr_round_trip_through_raw() {
        let mut obj = make_test_obj(42);
        let ptr = NonNull::new(&mut obj as *mut TestObj).unwrap();
        // SAFETY: obj is live for the duration of this test.
        let gc = unsafe { GcPtr::from_raw(ptr) };
        // Compare against `ptr.as_ptr()` (already captured above) rather than
        // re-borrowing `&mut obj`, which would create a new Unique retag and
        // invalidate the SharedReadWrite permission under Stacked Borrows.
        assert_eq!(gc.as_ptr(), ptr.as_ptr());
        // SAFETY: obj is live.
        assert_eq!(unsafe { gc.as_ref() }.value, 42);
    }

    #[test]
    fn gc_ptr_as_heap_object_ptr_and_back() {
        let mut obj = make_test_obj(99);
        let ptr = NonNull::new(&mut obj as *mut TestObj).unwrap();
        // SAFETY: obj is live for the duration of this test.
        let gc: GcPtr<TestObj> = unsafe { GcPtr::from_raw(ptr) };
        let heap_ptr = gc.as_heap_object_ptr();
        assert_eq!(heap_ptr as usize, &obj as *const TestObj as usize);
        // SAFETY: heap_ptr is a valid TestObj.
        let gc2: GcPtr<TestObj> = unsafe { GcPtr::from_heap_object_ptr(heap_ptr) };
        assert_eq!(gc, gc2);
    }

    #[test]
    fn gc_ptr_header_access() {
        let mut obj = make_test_obj(7);
        obj.header.init_alloc_size(123);
        let ptr = NonNull::new(&mut obj as *mut TestObj).unwrap();
        // SAFETY: obj is live.
        let gc = unsafe { GcPtr::from_raw(ptr) };
        // SAFETY: obj is live.
        assert_eq!(unsafe { gc.header() }.alloc_size(), 123);
    }

    #[test]
    fn gc_ptr_is_copy() {
        let mut obj = make_test_obj(1);
        let ptr = NonNull::new(&mut obj as *mut TestObj).unwrap();
        // SAFETY: obj is live.
        let gc = unsafe { GcPtr::from_raw(ptr) };
        let gc2 = gc; // copy
        assert_eq!(gc.as_ptr(), gc2.as_ptr());
    }

    #[test]
    fn gc_ptr_equality() {
        let mut obj = make_test_obj(1);
        let ptr = NonNull::new(&mut obj as *mut TestObj).unwrap();
        // SAFETY: obj is live.
        let a = unsafe { GcPtr::from_raw(ptr) };
        let b = unsafe { GcPtr::from_raw(ptr) };
        assert_eq!(a, b);
    }

    #[test]
    fn gc_ptr_trace_marks_object() {
        let mut obj = make_test_obj(5);
        let ptr = NonNull::new(&mut obj as *mut TestObj).unwrap();
        // SAFETY: obj is live.
        let gc = unsafe { GcPtr::from_raw(ptr) };
        let mut tracer = Tracer::new();
        gc.trace(&mut tracer);
        assert_eq!(tracer.gray_stack.len(), 1);
        assert_eq!(tracer.gray_stack[0], ptr.as_ptr() as *mut u8);
    }

    #[test]
    fn gc_object_default_header_accessors() {
        let mut obj = make_test_obj(0);
        obj.header.init_alloc_size(256);
        assert_eq!(obj.header().alloc_size(), 256);

        obj.header_mut().increment_age();
        assert_eq!(obj.header().age(), 1);
    }

    #[test]
    fn gc_ptr_debug_format() {
        let mut obj = make_test_obj(1);
        let ptr = NonNull::new(&mut obj as *mut TestObj).unwrap();
        // SAFETY: obj is live.
        let gc = unsafe { GcPtr::from_raw(ptr) };
        let debug = format!("{gc:?}");
        assert!(debug.starts_with("GcPtr(0x"), "debug output: {debug}");
    }

    #[test]
    fn heap_alloc_typed_returns_gc_ptr() {
        use crate::gc::heap::Heap;
        let mut heap = Heap::new();
        let gc = heap.alloc(make_test_obj(42));
        assert!(gc.is_some(), "typed allocation must succeed");
        let gc = gc.unwrap();
        // SAFETY: gc is a freshly allocated, live object.
        let obj = unsafe { gc.as_ref() };
        assert_eq!(obj.value, 42);
        // The alloc_size must have been initialised by the allocator.
        assert!(
            unsafe { gc.header() }.alloc_size() > 0,
            "alloc_size must be set"
        );
    }

    #[test]
    fn heap_alloc_typed_alloc_size_covers_full_type() {
        use crate::gc::heap::Heap;
        let mut heap = Heap::new();
        let gc = heap.alloc(make_test_obj(0)).unwrap();
        let layout = Layout::new::<TestObj>()
            .align_to(align_of::<HeapObject>())
            .unwrap()
            .pad_to_align();
        assert_eq!(
            unsafe { gc.header() }.alloc_size() as usize,
            layout.size(),
            "alloc_size must match the padded layout size of the type"
        );
    }

    #[test]
    fn heap_alloc_typed_as_mut_modifies_value() {
        use crate::gc::heap::Heap;
        let mut heap = Heap::new();
        let mut gc = heap.alloc(make_test_obj(10)).unwrap();
        // SAFETY: gc is a freshly allocated, live object; exclusive access.
        unsafe { gc.as_mut() }.value = 20;
        assert_eq!(unsafe { gc.as_ref() }.value, 20);
    }
}
