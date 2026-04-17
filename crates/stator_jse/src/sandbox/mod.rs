//! Memory sandbox for Stator heap isolation.
//!
//! The [`Sandbox`] reserves a contiguous virtual-address range and ensures
//! all [`HeapObject`][crate::objects::heap_object::HeapObject] pointers are
//! confined within it.  External pointers (e.g. native function pointers or
//! embedder data that live outside the sandbox) are stored in the
//! [`ExternalPointerTable`], identified by an opaque [`ExternalPointerHandle`],
//! so that in-sandbox code never holds raw pointers to external memory.
//!
//! # Design
//!
//! On Unix platforms the sandbox reserves virtual address space with
//! `mmap(PROT_NONE)` so that the range is unavailable to other allocators but
//! no physical pages are committed.  On other platforms a committed allocation
//! from the global allocator is used as a fallback.
//!
//! # Example
//!
//! ```no_run
//! use stator_jse::sandbox::Sandbox;
//!
//! let sandbox = Sandbox::new(64 * 1024 * 1024).expect("sandbox creation");
//! let ptr: *const u8 = std::ptr::null();
//! assert!(!sandbox.contains(ptr));
//! ```

use crate::error::{StatorError, StatorResult};

/// Default sandbox size: 1 GiB of reserved virtual address space.
pub const DEFAULT_SANDBOX_SIZE: usize = 1 << 30; // 1 GiB

// ── Sandbox ──────────────────────────────────────────────────────────────────

/// A contiguous virtual-address region that bounds all heap-object pointers.
///
/// All allocations for
/// [`HeapObject`][crate::objects::heap_object::HeapObject] must reside within
/// the sandbox range `[base, base + size)`.  Any pointer that falls outside
/// this range is rejected by [`check_in_bounds`][Sandbox::check_in_bounds].
///
/// External (non-sandbox) pointers such as native function callbacks and
/// embedder data are stored out-of-line in an [`ExternalPointerTable`] and
/// referenced through an [`ExternalPointerHandle`].
pub struct Sandbox {
    base: *mut u8,
    size: usize,
}

// SAFETY: The sandbox owns its virtual-address range exclusively; no other
// code holds a raw alias into it.
unsafe impl Send for Sandbox {}
unsafe impl Sync for Sandbox {}

impl Sandbox {
    /// Reserve `size` bytes of virtual address space for the sandbox.
    ///
    /// On Unix the range is reserved with `mmap(PROT_NONE)` (no physical
    /// pages committed).  On other platforms the memory is committed
    /// immediately via the global allocator.
    ///
    /// # Errors
    ///
    /// Returns [`StatorError::OutOfMemory`] when the OS cannot satisfy the
    /// reservation request.
    ///
    /// # Panics
    ///
    /// Panics if `size` is zero.
    pub fn new(size: usize) -> StatorResult<Self> {
        assert!(size > 0, "sandbox size must be non-zero");
        let base = reserve(size)?;
        Ok(Self { base, size })
    }

    /// Reserve the [`DEFAULT_SANDBOX_SIZE`] virtual-address range.
    ///
    /// # Errors
    ///
    /// Returns [`StatorError::OutOfMemory`] on reservation failure.
    pub fn new_default() -> StatorResult<Self> {
        Self::new(DEFAULT_SANDBOX_SIZE)
    }

    /// Base (lowest) address of the sandbox virtual-address range.
    #[inline]
    pub fn base(&self) -> *mut u8 {
        self.base
    }

    /// Size of the sandbox virtual-address range in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns `true` if `ptr` falls within `[base, base + size)`.
    ///
    /// A null pointer always returns `false`.
    #[inline]
    pub fn contains(&self, ptr: *const u8) -> bool {
        let addr = ptr as usize;
        let base = self.base as usize;
        // Use saturating_add to avoid usize overflow on degenerate inputs.
        addr >= base && addr < base.saturating_add(self.size)
    }

    /// Validates that `ptr` is within the sandbox address range.
    ///
    /// # Errors
    ///
    /// Returns [`StatorError::SandboxViolation`] when `ptr` does not fall in
    /// `[base, base + size)`.
    #[inline]
    pub fn check_in_bounds(&self, ptr: *const u8) -> StatorResult<()> {
        if self.contains(ptr) {
            Ok(())
        } else {
            Err(StatorError::SandboxViolation {
                address: ptr as usize,
                sandbox_base: self.base as usize,
                sandbox_size: self.size,
            })
        }
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        if !self.base.is_null() && self.size > 0 {
            // SAFETY: `base` and `size` were set by `reserve` in `new`.
            unsafe { release(self.base, self.size) };
        }
    }
}

// ── Platform virtual-address reservation ─────────────────────────────────────

/// Reserve `size` bytes of virtual address space and return the base pointer.
///
/// On Unix this uses `mmap(PROT_NONE)` so no physical pages are committed.
/// On other platforms the global allocator is used (memory is committed).
#[cfg(unix)]
fn reserve(size: usize) -> StatorResult<*mut u8> {
    // SAFETY: mmap with PROT_NONE is always safe; the range is inaccessible
    // until individual pages are mapped with a subsequent mmap/mprotect call.
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED || ptr.is_null() {
        Err(StatorError::OutOfMemory)
    } else {
        Ok(ptr as *mut u8)
    }
}

#[cfg(not(unix))]
fn reserve(size: usize) -> StatorResult<*mut u8> {
    use std::alloc::{Layout, alloc};
    let layout = Layout::from_size_align(size, 8).map_err(|_| StatorError::OutOfMemory)?;
    // SAFETY: layout is valid (size > 0, align is power-of-two).
    let ptr = unsafe { alloc(layout) };
    if ptr.is_null() {
        Err(StatorError::OutOfMemory)
    } else {
        Ok(ptr)
    }
}

/// Release virtual address space previously returned by [`reserve`].
///
/// # Safety
///
/// `base` and `size` must exactly match the values passed to the corresponding
/// [`reserve`] call.
#[cfg(unix)]
unsafe fn release(base: *mut u8, size: usize) {
    // SAFETY: `base` was returned by `mmap` with this exact `size`.
    unsafe { libc::munmap(base as *mut libc::c_void, size) };
}

#[cfg(not(unix))]
unsafe fn release(base: *mut u8, size: usize) {
    use std::alloc::{Layout, dealloc};
    let layout = Layout::from_size_align(size, 8).expect("valid layout");
    // SAFETY: `base` was allocated with this layout in `reserve`.
    unsafe { dealloc(base, layout) };
}

// ── ExternalPointerHandle ─────────────────────────────────────────────────────

/// An opaque index into an [`ExternalPointerTable`].
///
/// The value `0` is reserved as a null/invalid sentinel
/// ([`ExternalPointerHandle::NULL`]).  All other values refer to live or
/// previously-removed table entries.  A handle remains valid until
/// [`ExternalPointerTable::remove`] is called.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExternalPointerHandle(u32);

impl ExternalPointerHandle {
    /// The null (invalid) handle.  No valid table entry has this value.
    pub const NULL: Self = Self(0);

    /// Returns `true` if this is the null handle.
    #[inline]
    pub fn is_null(self) -> bool {
        self.0 == 0
    }
}

// ── ExternalPointerTable ──────────────────────────────────────────────────────

/// A lookup table mapping [`ExternalPointerHandle`]s to raw pointers that
/// live **outside** the sandbox.
///
/// In-sandbox code never holds non-sandbox raw pointers directly.  Instead it
/// stores a compact [`ExternalPointerHandle`] and resolves it through this
/// table.  The table is the single choke-point for all external-pointer
/// accesses.
///
/// # Allocation strategy
///
/// Entries are stored in a flat `Vec`.  Removed entries are recycled via a
/// free-list so that handle values can be re-used without growing the table
/// indefinitely.  Index 0 is permanently reserved as the null sentinel.
pub struct ExternalPointerTable {
    entries: Vec<Option<*mut ()>>,
    free_list: Vec<u32>,
}

// SAFETY: The table has exclusive ownership over the stored pointers; the
// caller is responsible for ensuring the pointed-to memory outlives any use
// through this table.
unsafe impl Send for ExternalPointerTable {}

impl ExternalPointerTable {
    /// Create an empty table.
    pub fn new() -> Self {
        Self {
            // Index 0 is always `None` — reserved as the null sentinel.
            entries: vec![None],
            free_list: Vec::new(),
        }
    }

    /// Insert an external pointer and return its [`ExternalPointerHandle`].
    ///
    /// The returned handle remains valid until [`remove`][Self::remove] is
    /// called.  Removed slots are recycled for subsequent insertions.
    ///
    /// # Panics
    ///
    /// Panics if the table already contains [`u32::MAX`] entries and all
    /// recycled slots have been exhausted (extremely unlikely in practice).
    pub fn insert(&mut self, ptr: *mut ()) -> ExternalPointerHandle {
        let idx = if let Some(recycled) = self.free_list.pop() {
            self.entries[recycled as usize] = Some(ptr);
            recycled
        } else {
            let next = self.entries.len();
            assert!(
                next <= u32::MAX as usize,
                "ExternalPointerTable overflow: cannot allocate more than u32::MAX entries"
            );
            self.entries.push(Some(ptr));
            next as u32
        };
        ExternalPointerHandle(idx)
    }

    /// Look up the pointer associated with `handle`.
    ///
    /// Returns `None` when `handle` is [`ExternalPointerHandle::NULL`] or has
    /// already been removed.
    pub fn get(&self, handle: ExternalPointerHandle) -> Option<*mut ()> {
        if handle.is_null() {
            return None;
        }
        self.entries.get(handle.0 as usize).and_then(|slot| *slot)
    }

    /// Remove `handle` from the table and recycle its slot.
    ///
    /// Returns the previously-stored pointer, or `None` if `handle` was null
    /// or had already been removed.
    pub fn remove(&mut self, handle: ExternalPointerHandle) -> Option<*mut ()> {
        if handle.is_null() {
            return None;
        }
        let slot = self.entries.get_mut(handle.0 as usize)?;
        let ptr = slot.take();
        if ptr.is_some() {
            self.free_list.push(handle.0);
        }
        ptr
    }

    /// Number of live (non-removed) entries currently in the table.
    pub fn len(&self) -> usize {
        // Skip index 0 (null sentinel).
        self.entries.iter().skip(1).filter(|s| s.is_some()).count()
    }

    /// Returns `true` if the table contains no live entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for ExternalPointerTable {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Sandbox tests ─────────────────────────────────────────────────────────

    /// A small sandbox size used in tests to avoid reserving gigabytes of VA.
    const TEST_SANDBOX_SIZE: usize = 4 * 1024 * 1024; // 4 MiB

    #[test]
    fn test_sandbox_contains_base_address() {
        let sb = Sandbox::new(TEST_SANDBOX_SIZE).expect("sandbox creation");
        // The base pointer itself must be inside the sandbox.
        assert!(sb.contains(sb.base()));
    }

    #[test]
    fn test_sandbox_contains_last_byte() {
        let sb = Sandbox::new(TEST_SANDBOX_SIZE).expect("sandbox creation");
        // SAFETY: `base + size - 1` is the last byte of the reserved range.
        let last = unsafe { sb.base().add(sb.size() - 1) };
        assert!(sb.contains(last));
    }

    #[test]
    fn test_sandbox_does_not_contain_one_past_end() {
        let sb = Sandbox::new(TEST_SANDBOX_SIZE).expect("sandbox creation");
        // SAFETY: arithmetic only; we never dereference this pointer.
        let one_past = unsafe { sb.base().add(sb.size()) };
        assert!(!sb.contains(one_past));
    }

    #[test]
    fn test_sandbox_does_not_contain_null() {
        let sb = Sandbox::new(TEST_SANDBOX_SIZE).expect("sandbox creation");
        assert!(!sb.contains(std::ptr::null()));
    }

    #[test]
    fn test_check_in_bounds_ok_for_base() {
        let sb = Sandbox::new(TEST_SANDBOX_SIZE).expect("sandbox creation");
        assert!(sb.check_in_bounds(sb.base()).is_ok());
    }

    #[test]
    fn test_check_in_bounds_err_for_out_of_range() {
        let sb = Sandbox::new(TEST_SANDBOX_SIZE).expect("sandbox creation");
        // A null pointer is always outside any real sandbox.
        let result = sb.check_in_bounds(std::ptr::null());
        assert!(result.is_err(), "expected SandboxViolation, got Ok");
        let err = result.unwrap_err();
        // Verify the error is the correct variant.
        assert!(
            matches!(err, StatorError::SandboxViolation { .. }),
            "unexpected error variant: {err:?}"
        );
    }

    #[test]
    fn test_check_in_bounds_err_one_past_end() {
        let sb = Sandbox::new(TEST_SANDBOX_SIZE).expect("sandbox creation");
        // SAFETY: arithmetic only; we never dereference this pointer.
        let one_past = unsafe { sb.base().add(sb.size()) };
        let result = sb.check_in_bounds(one_past);
        assert!(result.is_err(), "one-past-end must be out of bounds");
        assert!(matches!(
            result.unwrap_err(),
            StatorError::SandboxViolation { .. }
        ));
    }

    #[test]
    fn test_sandbox_violation_error_message() {
        let sb = Sandbox::new(TEST_SANDBOX_SIZE).expect("sandbox creation");
        let err = sb.check_in_bounds(std::ptr::null()).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("sandbox violation"),
            "error message should mention 'sandbox violation': {msg}"
        );
    }

    // ── ExternalPointerHandle tests ───────────────────────────────────────────

    #[test]
    fn test_null_handle_is_null() {
        assert!(ExternalPointerHandle::NULL.is_null());
    }

    #[test]
    fn test_non_null_handle_is_not_null() {
        let mut table = ExternalPointerTable::new();
        let handle = table.insert(std::ptr::null_mut());
        assert!(!handle.is_null());
    }

    // ── ExternalPointerTable tests ────────────────────────────────────────────

    #[test]
    fn test_table_insert_and_get() {
        let mut table = ExternalPointerTable::new();
        let value: u64 = 0xDEAD_BEEF;
        let ptr = &raw const value as *mut ();
        let handle = table.insert(ptr);
        assert_eq!(table.get(handle), Some(ptr));
    }

    #[test]
    fn test_table_null_handle_returns_none() {
        let table = ExternalPointerTable::new();
        assert_eq!(table.get(ExternalPointerHandle::NULL), None);
    }

    #[test]
    fn test_table_remove_returns_ptr() {
        let mut table = ExternalPointerTable::new();
        let value: u64 = 42;
        let ptr = &raw const value as *mut ();
        let handle = table.insert(ptr);
        assert_eq!(table.remove(handle), Some(ptr));
    }

    #[test]
    fn test_table_get_after_remove_returns_none() {
        let mut table = ExternalPointerTable::new();
        let value: u64 = 42;
        let ptr = &raw const value as *mut ();
        let handle = table.insert(ptr);
        table.remove(handle);
        assert_eq!(table.get(handle), None);
    }

    #[test]
    fn test_table_remove_twice_returns_none_second_time() {
        let mut table = ExternalPointerTable::new();
        let value: u64 = 1;
        let ptr = &raw const value as *mut ();
        let handle = table.insert(ptr);
        table.remove(handle);
        assert_eq!(table.remove(handle), None);
    }

    #[test]
    fn test_table_len_and_is_empty() {
        let mut table = ExternalPointerTable::new();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);

        let value: u64 = 1;
        let ptr = &raw const value as *mut ();
        let h1 = table.insert(ptr);
        let h2 = table.insert(ptr);
        assert_eq!(table.len(), 2);
        assert!(!table.is_empty());

        table.remove(h1);
        assert_eq!(table.len(), 1);

        table.remove(h2);
        assert!(table.is_empty());
    }

    #[test]
    fn test_table_slot_recycled_after_remove() {
        let mut table = ExternalPointerTable::new();
        let value: u64 = 99;
        let ptr = &raw const value as *mut ();
        let h1 = table.insert(ptr);
        table.remove(h1);

        let value2: u64 = 100;
        let ptr2 = &raw const value2 as *mut ();
        let h2 = table.insert(ptr2);

        // The recycled slot should have been reused.
        assert_eq!(h1, h2, "recycled slot should produce same handle index");
        assert_eq!(table.get(h2), Some(ptr2));
    }

    #[test]
    fn test_table_many_entries() {
        let mut table = ExternalPointerTable::new();
        let values: Vec<u64> = (0..1_000).collect();
        let handles: Vec<ExternalPointerHandle> = values
            .iter()
            .map(|v| table.insert(v as *const u64 as *mut ()))
            .collect();
        assert_eq!(table.len(), 1_000);
        for (i, handle) in handles.iter().enumerate() {
            assert_eq!(
                table.get(*handle),
                Some(&values[i] as *const u64 as *mut ()),
                "entry {i} mismatch"
            );
        }
    }
}
