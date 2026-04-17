//! Thread-local allocation buffers (TLABs).
//!
//! A TLAB is a small (default 32 KiB) contiguous block of memory carved out
//! from the GC heap.  Each mutator thread owns one TLAB and performs bump
//! allocation into it without any synchronisation.
//!
//! # Fast path (inline)
//!
//! The fast path is exactly **1 add + 1 compare + 1 store**:
//!
//! ```text
//! new_cursor = cursor + size;
//! if new_cursor > limit { goto slow_path; }
//! result = cursor;
//! cursor = new_cursor;
//! ```
//!
//! # Slow path
//!
//! When the current block is exhausted the TLAB requests a fresh block from
//! the global heap (or a central free-list) and resets its cursor/limit pair.

use std::alloc::{Layout, alloc, dealloc};

/// Default TLAB block size: 32 KiB.
pub const TLAB_SIZE: usize = 32 * 1024;

/// Alignment for TLAB base addresses and internal allocations.
const TLAB_ALIGN: usize = 8;

/// A thread-local allocation buffer.
///
/// Owns a single contiguous block of `capacity` bytes.  Allocations are
/// bump-pointer and never free individual objects — the entire buffer is
/// returned to the heap when the TLAB is retired.
pub struct Tlab {
    /// Start of the owned block.
    base: *mut u8,
    /// Current allocation cursor (next byte to hand out).
    cursor: usize,
    /// One-past-end of the usable region (`base + capacity`).
    limit: usize,
    /// Size of the backing allocation.
    capacity: usize,
}

// SAFETY: A Tlab is only accessed by the thread that owns it (the "thread-
// local" part).  We mark it `Send` so it can be moved to a new thread during
// initialisation, but concurrent access is never performed.
unsafe impl Send for Tlab {}

impl Tlab {
    /// Allocate a new TLAB with the default block size ([`TLAB_SIZE`]).
    pub fn new() -> Self {
        Self::with_capacity(TLAB_SIZE)
    }

    /// Allocate a new TLAB with the given block size.
    ///
    /// # Panics
    /// Panics if `capacity` is zero or if the system allocator fails.
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity > 0, "TLAB capacity must be non-zero");
        // SAFETY: capacity > 0 and TLAB_ALIGN is a valid power of two.
        let layout = Layout::from_size_align(capacity, TLAB_ALIGN).expect("valid TLAB layout");
        let base = unsafe { alloc(layout) };
        assert!(!base.is_null(), "TLAB allocation failed");
        let base_addr = base as usize;
        Self {
            base,
            cursor: base_addr,
            limit: base_addr + capacity,
            capacity,
        }
    }

    /// **Fast-path** bump allocation: 1 add + 1 compare + 1 store.
    ///
    /// Returns a properly-aligned pointer to `size` bytes, or `None` if the
    /// current block is exhausted (caller should invoke the slow path).
    ///
    /// # Safety contract
    /// The returned pointer is valid for writes of `size` bytes.  The caller
    /// must not use it after the TLAB is dropped.
    #[inline(always)]
    pub fn bump_alloc(&mut self, size: usize) -> Option<*mut u8> {
        // Round up cursor to TLAB_ALIGN.
        let aligned_cursor = (self.cursor + (TLAB_ALIGN - 1)) & !(TLAB_ALIGN - 1);
        let new_cursor = aligned_cursor + size;
        if new_cursor > self.limit {
            return None; // exhausted → slow path
        }
        let ptr = aligned_cursor as *mut u8;
        self.cursor = new_cursor;
        Some(ptr)
    }

    /// **Slow path**: retire the current block and allocate a fresh one, then
    /// retry the bump allocation.
    ///
    /// Returns `None` only if the requested `size` exceeds a single TLAB
    /// block (the caller should fall back to a large-object allocator).
    pub fn bump_alloc_slow(&mut self, size: usize) -> Option<*mut u8> {
        // Cannot satisfy an allocation larger than a full block.
        let needed = size + (TLAB_ALIGN - 1); // worst-case alignment padding
        if needed > self.capacity {
            return None;
        }
        self.refill();
        self.bump_alloc(size)
    }

    /// Allocate `size` bytes, using the fast path first and falling back to
    /// the slow path on exhaustion.
    #[inline]
    pub fn allocate(&mut self, size: usize) -> Option<*mut u8> {
        self.bump_alloc(size).or_else(|| self.bump_alloc_slow(size))
    }

    /// Remaining free bytes in the current block.
    pub fn remaining(&self) -> usize {
        self.limit.saturating_sub(self.cursor)
    }

    /// Total capacity of the current block.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of bytes already allocated in the current block.
    pub fn used(&self) -> usize {
        self.cursor - (self.base as usize)
    }

    /// Replace the backing block with a fresh allocation of the same
    /// capacity.
    ///
    /// The old block is freed.  In a production engine this would return the
    /// old block to a central free-list for the GC to scan.
    fn refill(&mut self) {
        // Free old block.
        if !self.base.is_null() {
            // SAFETY: base was allocated with the same layout.
            let layout =
                Layout::from_size_align(self.capacity, TLAB_ALIGN).expect("valid TLAB layout");
            unsafe { dealloc(self.base, layout) };
        }

        // Allocate new block.
        let layout = Layout::from_size_align(self.capacity, TLAB_ALIGN).expect("valid TLAB layout");
        // SAFETY: layout is valid (capacity > 0, alignment is power-of-two).
        let base = unsafe { alloc(layout) };
        assert!(!base.is_null(), "TLAB refill allocation failed");
        let base_addr = base as usize;
        self.base = base;
        self.cursor = base_addr;
        self.limit = base_addr + self.capacity;
    }
}

impl Default for Tlab {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Tlab {
    fn drop(&mut self) {
        if !self.base.is_null() {
            // SAFETY: base was allocated with the matching layout.
            let layout =
                Layout::from_size_align(self.capacity, TLAB_ALIGN).expect("valid TLAB layout");
            unsafe { dealloc(self.base, layout) };
            self.base = std::ptr::null_mut();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tlab_new_has_full_capacity() {
        let tlab = Tlab::new();
        assert_eq!(tlab.capacity(), TLAB_SIZE);
        assert_eq!(tlab.used(), 0);
        assert_eq!(tlab.remaining(), TLAB_SIZE);
    }

    #[test]
    fn test_bump_alloc_fast_path() {
        let mut tlab = Tlab::with_capacity(256);
        let p1 = tlab.bump_alloc(16).expect("first alloc");
        assert!(!p1.is_null());
        assert_eq!(tlab.used(), 16);
        assert_eq!(tlab.remaining(), 256 - 16);

        let p2 = tlab.bump_alloc(32).expect("second alloc");
        assert!(!p2.is_null());
        assert_ne!(p1, p2);
    }

    #[test]
    fn test_bump_alloc_returns_none_on_exhaustion() {
        let mut tlab = Tlab::with_capacity(64);
        // Allocate all 64 bytes.
        assert!(tlab.bump_alloc(64).is_some());
        // Next allocation must fail (fast path returns None).
        assert!(tlab.bump_alloc(1).is_none());
    }

    #[test]
    fn test_slow_path_refills_and_succeeds() {
        let mut tlab = Tlab::with_capacity(64);
        // Exhaust the block.
        assert!(tlab.bump_alloc(64).is_some());
        // Fast path fails, slow path should refill and succeed.
        let p = tlab.bump_alloc_slow(16).expect("slow path must succeed");
        assert!(!p.is_null());
    }

    #[test]
    fn test_slow_path_rejects_oversized_allocation() {
        let mut tlab = Tlab::with_capacity(64);
        // Request more than the block capacity.
        assert!(tlab.bump_alloc_slow(128).is_none());
    }

    #[test]
    fn test_allocate_combines_fast_and_slow() {
        let mut tlab = Tlab::with_capacity(64);
        // Fast path.
        assert!(tlab.allocate(32).is_some());
        assert!(tlab.allocate(32).is_some());
        // Exhausted — should refill via slow path.
        assert!(tlab.allocate(16).is_some());
    }

    #[test]
    fn test_allocations_are_aligned() {
        let mut tlab = Tlab::with_capacity(256);
        for _ in 0..10 {
            let p = tlab.allocate(7).expect("alloc");
            assert_eq!(p as usize % TLAB_ALIGN, 0, "pointer must be aligned");
        }
    }

    #[test]
    fn test_default_tlab_size_is_32kb() {
        assert_eq!(TLAB_SIZE, 32 * 1024);
    }
}
