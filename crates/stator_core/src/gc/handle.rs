use std::marker::PhantomData;
use std::ptr::NonNull;

/// A stack frame that gives out [`Local`] handles.
///
/// The lifetime `'isolate` is borrowed from the entity (typically an
/// `Isolate`) that owns the root list and the heap.  Every [`Local`] handle
/// created through this scope carries a `'scope` lifetime that is strictly
/// shorter than — or equal to — `'isolate`, ensuring that no local handle can
/// outlive the scope that created it.
///
/// When `HandleScope` is dropped, any `Local` borrows tied to it become
/// invalid **at compile time** (the borrow checker rejects code that tries to
/// use them after the scope ends).
pub struct HandleScope<'isolate> {
    /// Raw handle pointers registered in this scope.  These are kept so that
    /// a future nursery-scanning GC can trace live locals without walking the
    /// Rust call stack.
    handles: Vec<*mut u8>,
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
            handles: Vec::new(),
            _isolate: PhantomData,
        }
    }

    /// Register `ptr` with this scope and return a [`Local`] handle.
    ///
    /// The returned `Local<'scope, T>` borrows `'scope` (the lifetime of
    /// *this mutable borrow of `self`*), guaranteeing it cannot outlive the
    /// scope.
    ///
    /// # Safety
    /// `ptr` must point to a live, properly-aligned `T` that the heap will
    /// not collect for at least as long as this `HandleScope` is alive.
    pub unsafe fn create_local<'scope, T>(&'scope mut self, ptr: NonNull<T>) -> Local<'scope, T> {
        self.handles.push(ptr.as_ptr() as *mut u8);
        Local {
            ptr,
            _scope: PhantomData,
        }
    }

    /// Iterate the raw handle pointers registered in this scope.
    ///
    /// Used by the GC to scan local roots without knowing their concrete types.
    pub fn raw_handles(&self) -> impl Iterator<Item = *mut u8> + '_ {
        self.handles.iter().copied()
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
pub struct Persistent<T> {
    ptr: NonNull<T>,
    /// Raw pointer to the `PersistentRoots` that owns the slot.
    /// Using a raw pointer avoids borrow-checker entanglement while keeping
    /// `Persistent` as a plain owned type; the safety invariant is documented
    /// on [`Persistent::from_local`].
    roots: *mut PersistentRoots,
    index: usize,
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
            index,
            _marker: PhantomData,
        }
    }

    /// Return a shared reference to the underlying value.
    ///
    /// # Safety
    /// The caller must ensure the object has not been freed or moved since
    /// this `Persistent` was created.
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
        // SAFETY: `self.roots` is guaranteed valid for our lifetime by the
        // contract of `from_local`.
        unsafe { (*self.roots).unregister(self.index) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dummy isolate token (the real `Isolate` will own this slot).
    fn make_isolate() -> () {}

    #[test]
    fn local_lifetime_tied_to_scope() {
        let mut isolate = make_isolate();
        let mut scope = HandleScope::new(&mut isolate);
        let value: u32 = 99;
        let mut boxed = Box::new(value);
        let ptr = NonNull::new(boxed.as_mut() as *mut u32).unwrap();
        // SAFETY: `boxed` outlives `scope`.
        let local = unsafe { scope.create_local(ptr) };
        // SAFETY: `boxed` has not been moved or freed.
        assert_eq!(unsafe { *local.as_ref() }, 99);
    }

    #[test]
    fn persistent_root_is_tracked() {
        let mut isolate = make_isolate();
        let mut roots = PersistentRoots::new();
        let value: u32 = 42;
        let mut boxed = Box::new(value);
        let ptr = NonNull::new(boxed.as_mut() as *mut u32).unwrap();

        let persistent = {
            let mut scope = HandleScope::new(&mut isolate);
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
