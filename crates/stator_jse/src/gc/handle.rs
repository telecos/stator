use std::marker::PhantomData;
use std::ptr::NonNull;

/// Number of handle pointer slots in each [`HandleBlock`].
const HANDLE_BLOCK_CAPACITY: usize = 1_024;

/// A fixed-capacity slab of raw GC-root pointers used by [`HandleScope`].
///
/// `HandleScope` manages a stack of `HandleBlock`s.  When the active block is
/// full a fresh one is pushed; all blocks are freed when the scope is dropped.
struct HandleBlock {
    slots: Vec<*mut u8>,
}

// SAFETY: Raw pointers here are GC bookkeeping data; the GC is single-threaded
// and all typed access goes through `Local<T>` whose lifetimes are checked by
// the borrow checker.
unsafe impl Send for HandleBlock {}

impl HandleBlock {
    fn new() -> Self {
        Self {
            slots: Vec::with_capacity(HANDLE_BLOCK_CAPACITY),
        }
    }

    fn is_full(&self) -> bool {
        self.slots.len() >= HANDLE_BLOCK_CAPACITY
    }

    fn push(&mut self, ptr: *mut u8) {
        self.slots.push(ptr);
    }
}

/// A stack frame that gives out [`Local`] handles.
///
/// The lifetime `'isolate` is borrowed from the entity (typically an
/// `Isolate`) that owns the root list and the heap.  Every [`Local`] handle
/// created through this scope carries `'isolate` as its lifetime, ensuring
/// that no local handle can outlive the isolate (or the nested scope) that
/// created it.
///
/// Handles are stored in a stack of [`HandleBlock`]s: when the active block
/// fills up a new one is appended; all blocks are freed when the scope drops.
///
/// Nested scopes are supported via [`HandleScope::open_child_scope`]: the
/// child scope borrows from the parent, so the borrow checker prevents the
/// parent from being used directly while the child is live and ensures that
/// child-scope locals cannot escape past the child's lifetime.
pub struct HandleScope<'isolate> {
    /// Stack of fixed-capacity handle blocks.  New handles fill the last block;
    /// a fresh block is appended when the last one is full.
    blocks: Vec<HandleBlock>,
    _isolate: PhantomData<&'isolate mut ()>,
}

// SAFETY: Raw pointers are used only for GC bookkeeping; all access goes
// through typed `Local<T>` references whose lifetimes are checked by the
// borrow checker.
unsafe impl<'isolate> Send for HandleScope<'isolate> {}

impl<'isolate> HandleScope<'isolate> {
    /// Open a new `HandleScope` that borrows from `isolate`.
    ///
    /// The `isolate` parameter is a mutable borrow token: holding it ensures
    /// that no two scopes are open simultaneously on the same isolate (a
    /// constraint the real engine will enforce more rigorously later).
    pub fn new(_isolate: &'isolate mut ()) -> Self {
        Self {
            blocks: Vec::new(),
            _isolate: PhantomData,
        }
    }

    /// Push `ptr` into the active handle block, allocating a fresh block first
    /// if the current one is full (or if no block exists yet).
    fn push_handle(&mut self, ptr: *mut u8) {
        if self.blocks.last().is_none_or(|b| b.is_full()) {
            self.blocks.push(HandleBlock::new());
        }
        // At this point `self.blocks` is guaranteed non-empty: either it
        // already contained a block that was not full, or we just pushed one
        // above.  The `expect` is therefore unreachable.
        self.blocks
            .last_mut()
            .expect("block stack is non-empty; a block was just pushed if needed")
            .push(ptr);
    }

    /// Register `ptr` with this scope and return a [`Local`] handle.
    ///
    /// The returned `Local<'isolate, T>` carries the scope's `'isolate`
    /// lifetime, guaranteeing it cannot outlive the isolate.  Because the
    /// lifetime is `'isolate` rather than the duration of the `&mut self`
    /// borrow, the scope can still be passed to [`open_child_scope`] while
    /// outer-scope locals remain accessible.
    ///
    /// [`open_child_scope`]: HandleScope::open_child_scope
    ///
    /// # Safety
    /// `ptr` must point to a live, properly-aligned `T` that the heap will
    /// not collect for at least as long as this `HandleScope` is alive.
    pub unsafe fn create_local<T>(&mut self, ptr: NonNull<T>) -> Local<'isolate, T> {
        self.push_handle(ptr.as_ptr() as *mut u8);
        Local {
            ptr,
            _scope: PhantomData,
        }
    }

    /// Open a child scope nested inside this scope.
    ///
    /// While the child scope is alive, `self` is mutably borrowed and cannot
    /// be used directly.  [`Local`] handles created in the child scope carry
    /// the child's (shorter) `'parent` lifetime and therefore **cannot escape**
    /// past the closing brace of the child scope.
    ///
    /// When the child scope is dropped all of its handle blocks are freed;
    /// the parent scope and its locals remain unaffected.
    pub fn open_child_scope<'parent>(&'parent mut self) -> HandleScope<'parent>
    where
        'isolate: 'parent,
    {
        HandleScope {
            blocks: Vec::new(),
            _isolate: PhantomData,
        }
    }

    /// Iterate the raw handle pointers registered in this scope.
    ///
    /// Used by the GC to scan local roots without knowing their concrete types.
    pub fn raw_handles(&self) -> impl Iterator<Item = *mut u8> + '_ {
        self.blocks.iter().flat_map(|b| b.slots.iter().copied())
    }
}

/// A handle to a GC-managed value that is valid only within a [`HandleScope`].
///
/// `Local<'scope, T>` is `Copy` so it can be passed freely within `'scope`,
/// mirroring the semantics of V8's `v8::Local<T>`.
#[derive(Copy, Clone)]
pub struct Local<'scope, T> {
    ptr: NonNull<T>,
    _scope: PhantomData<&'scope T>,
}

impl<'scope, T> Local<'scope, T> {
    /// Return a shared reference to the value, bound to `'scope`.
    ///
    /// # Safety
    /// The pointed-to object must not have been moved or freed by a GC cycle
    /// that ran after this handle was created.
    pub unsafe fn as_ref(&self) -> &'scope T {
        // SAFETY: caller guarantees the pointer is valid.
        unsafe { self.ptr.as_ref() }
    }

    /// Return the raw pointer.
    pub fn as_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }
}

/// Registry of GC roots owned by an `Isolate`.
///
/// Every [`Persistent`] handle registers its pointer here so that the GC can
/// treat it as a root even after the [`HandleScope`] that first created the
/// `Local` has been dropped.
pub struct PersistentRoots {
    /// Slot `i` is `Some(ptr)` while the corresponding `Persistent` is live,
    /// and `None` once it has been dropped.
    roots: Vec<Option<*mut u8>>,
}

// SAFETY: Raw pointers are GC roots; access is guarded by Rust ownership.
unsafe impl Send for PersistentRoots {}

impl PersistentRoots {
    /// Create an empty root set.
    pub fn new() -> Self {
        Self { roots: Vec::new() }
    }

    /// Register `ptr` and return its slot index.
    fn register(&mut self, ptr: *mut u8) -> usize {
        // Reuse a freed slot when available.
        if let Some(idx) = self.roots.iter().position(|s| s.is_none()) {
            self.roots[idx] = Some(ptr);
            return idx;
        }
        let idx = self.roots.len();
        self.roots.push(Some(ptr));
        idx
    }

    /// Clear the slot at `index`, indicating the root is no longer live.
    fn unregister(&mut self, index: usize) {
        if let Some(slot) = self.roots.get_mut(index) {
            *slot = None;
        }
    }

    /// Iterate all live (non-`None`) root pointers.
    pub fn iter_roots(&self) -> impl Iterator<Item = *mut u8> + '_ {
        self.roots.iter().filter_map(|s| *s)
    }
}

impl Default for PersistentRoots {
    fn default() -> Self {
        Self::new()
    }
}

/// A GC root handle that keeps a heap object alive across [`HandleScope`] exits.
///
/// While a `Persistent<T>` is alive its pointer is held in [`PersistentRoots`],
/// preventing the GC from collecting the object.  When the `Persistent` is
/// dropped, the root is automatically unregistered.
///
/// The root can also be explicitly cleared before `drop` by calling
/// [`reset`](Self::reset); subsequent drops become no-ops.
pub struct Persistent<T> {
    ptr: NonNull<T>,
    /// Raw pointer to the `PersistentRoots` that owns the slot.
    /// Using a raw pointer avoids borrow-checker entanglement while keeping
    /// `Persistent` as a plain owned type; the safety invariant is documented
    /// on [`Persistent::from_local`].
    roots: *mut PersistentRoots,
    /// `Some(idx)` while the root slot is registered; `None` after [`reset`](Self::reset).
    index: Option<usize>,
    _marker: PhantomData<T>,
}

// SAFETY: `Persistent` owns a GC root slot; it is safe to send across threads
// as long as `T` is `Send`.
unsafe impl<T: Send> Send for Persistent<T> {}

impl<T> Persistent<T> {
    /// Upgrade a `Local` into a `Persistent` root.
    ///
    /// The pointer from `local` is registered in `roots` so it survives the
    /// current `HandleScope`.
    ///
    /// # Safety
    /// `roots` must remain valid (i.e. not be dropped or moved) for the entire
    /// lifetime of this `Persistent`.
    pub unsafe fn from_local<'scope>(local: Local<'scope, T>, roots: &mut PersistentRoots) -> Self {
        let ptr = local.ptr;
        let index = roots.register(ptr.as_ptr() as *mut u8);
        Self {
            ptr,
            roots,
            index: Some(index),
            _marker: PhantomData,
        }
    }

    /// Explicitly unregister this root from [`PersistentRoots`].
    ///
    /// After `reset()` the GC no longer considers the pointed-to object a
    /// root, and the slot in [`PersistentRoots`] is freed for reuse.  The
    /// `Persistent` itself remains valid as an owned struct; dropping it
    /// afterwards is a safe no-op.
    ///
    /// # Safety note
    /// After calling `reset()`, calling [`as_ref`](Self::as_ref) is unsound if
    /// the GC may have collected the object; the pointer is retained in the
    /// struct but is no longer considered live.
    pub fn reset(&mut self) {
        if let Some(idx) = self.index.take() {
            // SAFETY: `self.roots` is valid for our lifetime by the contract of
            // `from_local`.  The GC is single-threaded, so no concurrent
            // mutation of `PersistentRoots` is possible.  `self.index.take()`
            // sets the slot to `None` before dereferencing, making repeated
            // calls idempotent and preventing any double-unregister.
            unsafe { (*self.roots).unregister(idx) };
        }
    }

    /// Returns `true` if this handle has been [`reset`](Self::reset) and no
    /// longer holds an active GC root.
    pub fn is_empty(&self) -> bool {
        self.index.is_none()
    }

    /// Return a shared reference to the underlying value.
    ///
    /// # Safety
    /// The caller must ensure the object has not been freed or moved since
    /// this `Persistent` was created, and that [`reset`](Self::reset) has not
    /// been called (which would allow the GC to collect the object).
    pub unsafe fn as_ref(&self) -> &T {
        // SAFETY: caller upholds the validity invariant.
        unsafe { self.ptr.as_ref() }
    }

    /// Return the raw pointer.
    pub fn as_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<T> Drop for Persistent<T> {
    fn drop(&mut self) {
        // Delegate to `reset` so that the unregister logic lives in one place
        // and a double-free (drop after explicit reset) is a safe no-op.
        self.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dummy isolate token (the real `Isolate` will own this slot).
    fn make_isolate() {}

    #[test]
    fn local_lifetime_tied_to_scope() {
        make_isolate();
        let mut unit = ();
        let mut scope = HandleScope::new(&mut unit);
        let value: u32 = 99;
        let mut boxed = Box::new(value);
        let ptr = NonNull::new(boxed.as_mut() as *mut u32).unwrap();
        // SAFETY: `boxed` outlives `scope`.
        let local = unsafe { scope.create_local(ptr) };
        // SAFETY: `boxed` has not been moved or freed.
        assert_eq!(unsafe { *local.as_ref() }, 99);
    }

    #[test]
    fn handle_scope_uses_blocks_for_storage() {
        let mut unit = ();
        let mut scope = HandleScope::new(&mut unit);
        let mut values: Vec<u32> = (0..10).collect();
        for v in &mut values {
            let ptr = NonNull::new(v as *mut u32).unwrap();
            // SAFETY: `values` outlives `scope`.
            unsafe { scope.create_local(ptr) };
        }
        assert_eq!(scope.raw_handles().count(), 10);
    }

    #[test]
    fn nested_scopes_inner_handles_tracked_separately() {
        let mut unit = ();
        let mut outer_scope = HandleScope::new(&mut unit);

        let mut v_outer: u32 = 42;
        let ptr_outer = NonNull::new(&mut v_outer as *mut u32).unwrap();
        // SAFETY: `v_outer` outlives `outer_scope`.
        let local_outer = unsafe { outer_scope.create_local(ptr_outer) };
        assert_eq!(outer_scope.raw_handles().count(), 1);

        // Open an inner (child) scope.  While the child scope is alive,
        // `outer_scope` is mutably borrowed via `open_child_scope`.
        {
            let mut inner_scope = outer_scope.open_child_scope();
            let mut v_inner: u32 = 7;
            let ptr_inner = NonNull::new(&mut v_inner as *mut u32).unwrap();
            // SAFETY: `v_inner` outlives `inner_scope`.
            let local_inner = unsafe { inner_scope.create_local(ptr_inner) };
            // SAFETY: `v_inner` has not been moved or freed.
            assert_eq!(unsafe { *local_inner.as_ref() }, 7);
            assert_eq!(inner_scope.raw_handles().count(), 1);
            // `inner_scope` and `local_inner` are dropped here; their blocks
            // are freed.
        }

        // Outer scope is accessible again; its handle is still present.
        assert_eq!(outer_scope.raw_handles().count(), 1);
        // SAFETY: `v_outer` has not been moved or freed.
        assert_eq!(unsafe { *local_outer.as_ref() }, 42);
    }

    #[test]
    fn persistent_root_is_tracked() {
        make_isolate();
        let mut roots = PersistentRoots::new();
        let value: u32 = 42;
        let mut boxed = Box::new(value);
        let ptr = NonNull::new(boxed.as_mut() as *mut u32).unwrap();

        let persistent = {
            let mut unit = ();
            let mut scope = HandleScope::new(&mut unit);
            // SAFETY: `boxed` outlives `persistent`.
            let local = unsafe { scope.create_local(ptr) };
            // SAFETY: `roots` outlives `persistent`.
            unsafe { Persistent::from_local(local, &mut roots) }
        };

        // Root must still be tracked after the scope is dropped.
        let live_roots: Vec<_> = roots.iter_roots().collect();
        assert_eq!(live_roots.len(), 1);
        assert_eq!(live_roots[0], boxed.as_mut() as *mut u32 as *mut u8);

        drop(persistent);

        // After the Persistent is dropped the root slot must be freed.
        assert_eq!(roots.iter_roots().count(), 0);
    }

    #[test]
    fn persistent_explicit_reset_clears_root() {
        let mut roots = PersistentRoots::new();
        let mut value: u32 = 55;
        let ptr = NonNull::new(&mut value as *mut u32).unwrap();

        let mut unit = ();
        let mut scope = HandleScope::new(&mut unit);
        // SAFETY: `value` outlives `persistent`.
        let local = unsafe { scope.create_local(ptr) };
        // SAFETY: `roots` outlives `persistent`.
        let mut persistent = unsafe { Persistent::from_local(local, &mut roots) };

        assert!(!persistent.is_empty());
        assert_eq!(roots.iter_roots().count(), 1);

        // Explicitly clear the root before drop.
        persistent.reset();

        assert!(persistent.is_empty());
        assert_eq!(roots.iter_roots().count(), 0);

        // A second reset (and the implicit drop) must be safe no-ops.
        persistent.reset();
        drop(persistent);
        assert_eq!(roots.iter_roots().count(), 0);
    }

    #[test]
    fn persistent_roots_reuses_freed_slots() {
        let mut roots = PersistentRoots::new();
        let mut v1: u8 = 1;
        let mut v2: u8 = 2;
        let p1 = NonNull::new(&mut v1 as *mut u8).unwrap();
        let p2 = NonNull::new(&mut v2 as *mut u8).unwrap();

        let idx1 = roots.register(p1.as_ptr());
        roots.unregister(idx1);
        let idx2 = roots.register(p2.as_ptr());

        // The freed slot should be reused.
        assert_eq!(idx1, idx2);
    }
}
