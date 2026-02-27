/// Drives the mark-and-trace traversal during a GC cycle.
///
/// The tracer maintains a grey stack of pointers that have been *marked* as
/// reachable but whose outgoing references have not yet been visited.  The GC
/// loop pops entries from the grey stack and calls `Trace::trace` on them,
/// which in turn pushes their referents onto the grey stack.
pub struct Tracer {
    /// Raw pointers to heap objects that are marked but not yet fully traced.
    pub(crate) gray_stack: Vec<*mut u8>,
}

impl Tracer {
    /// Create a new, empty `Tracer`.
    pub fn new() -> Self {
        Self {
            gray_stack: Vec::new(),
        }
    }

    /// Mark a raw heap pointer as reachable and enqueue it for tracing.
    ///
    /// # Safety
    /// `ptr` must point to a live, properly-aligned heap object that will
    /// remain valid for the duration of the GC cycle.  Passing a null or
    /// dangling pointer is undefined behaviour.
    pub unsafe fn mark_raw(&mut self, ptr: *mut u8) {
        if !ptr.is_null() {
            self.gray_stack.push(ptr);
        }
    }
}

impl Default for Tracer {
    fn default() -> Self {
        Self::new()
    }
}

/// All GC-managed types must implement `Trace` to expose their outgoing
/// heap references to the garbage collector.
///
/// # Contract
/// An implementation **must** call [`Tracer::mark_raw`] (or an equivalent
/// typed helper) for *every* heap pointer it owns.  Any pointer that is not
/// reported will be considered unreachable and may be freed or moved.
///
/// # Safety
/// Implementors must not hold any mutable borrows to GC-managed memory while
/// `trace` is running, as the tracer may inspect the same objects.
pub trait Trace {
    /// Visit all outgoing heap references, marking each via the tracer.
    fn trace(&self, tracer: &mut Tracer);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    struct Leaf;
    impl Trace for Leaf {
        fn trace(&self, _tracer: &mut Tracer) {}
    }

    #[test]
    fn tracer_ignores_null() {
        let mut tracer = Tracer::new();
        // SAFETY: null pointer check is the point of this test.
        unsafe { tracer.mark_raw(std::ptr::null_mut()) };
        assert!(tracer.gray_stack.is_empty());
    }

    #[test]
    fn tracer_enqueues_non_null() {
        let mut x: u8 = 42;
        let mut tracer = Tracer::new();
        // SAFETY: &mut x is a valid, live pointer for this test.
        unsafe { tracer.mark_raw(&mut x as *mut u8) };
        assert_eq!(tracer.gray_stack.len(), 1);
    }
}
