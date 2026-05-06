//! Cross-platform abstraction for allocating, populating, and releasing the
//! executable memory regions used by Stator's baseline and Maglev JIT tiers.
//!
//! The abstraction follows a strict W^X (write-xor-execute) construction
//! pattern: pages are first mapped read/write so the caller can copy code
//! into them, then re-protected to read/execute before the
//! [`ExecutableMemory`] handle is returned.  Once constructed the region is
//! immutable from the user's point of view; the only way to free it is to
//! drop the handle.
//!
//! Platform support:
//!
//! * Unix (`mmap` / `mprotect` / `munmap`) — production path used by the
//!   existing baseline and Maglev caches.
//! * Windows x86-64 (`VirtualAlloc` / `VirtualProtect` /
//!   `FlushInstructionCache` / `VirtualFree`) — provides parity for the
//!   in-progress Win64 tiering work.  Only the executable memory abstraction
//!   is provided on Windows; the JIT entry-point ABI port lives in a
//!   separate task.
//! * All other targets — [`ExecutableMemory::new`] returns
//!   [`ExecutableMemoryError::Unsupported`] so callers can fall back to the
//!   interpreter without crashing.

use std::fmt;
use std::ptr::NonNull;

/// Failure modes reported by [`ExecutableMemory::new`].
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ExecutableMemoryError {
    /// The caller supplied an empty code buffer.
    ZeroSize,
    /// The operating system refused the underlying page allocation
    /// (`mmap` returned `MAP_FAILED` or `VirtualAlloc` returned null).
    AllocationFailed,
    /// The W^X protection switch (`mprotect` / `VirtualProtect`) failed.
    ProtectionChangeFailed,
    /// The current build target does not have an executable-memory
    /// implementation.
    Unsupported,
}

impl fmt::Display for ExecutableMemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSize => f.write_str("executable memory request was zero bytes"),
            Self::AllocationFailed => {
                f.write_str("operating system rejected the executable memory allocation")
            }
            Self::ProtectionChangeFailed => {
                f.write_str("operating system rejected the W^X protection change")
            }
            Self::Unsupported => {
                f.write_str("executable memory is not supported on this build target")
            }
        }
    }
}

impl std::error::Error for ExecutableMemoryError {}

/// Owning handle to a region of read-only-executable memory containing JIT
/// machine code.
///
/// The region is allocated and freed through the host's page allocator (see
/// the module docs for the per-platform syscall list).  After construction
/// the region is **immutable** — there is no safe way to write to it again.
pub struct ExecutableMemory {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: The owned region is read-only-executable after construction.
// Calls into the code go through caller-supplied function pointers; the
// handle itself never mutates state.  These bounds match the existing
// `JitExecutableCode` / `CachedExecutableCode` wrappers.
unsafe impl Send for ExecutableMemory {}
unsafe impl Sync for ExecutableMemory {}

impl fmt::Debug for ExecutableMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecutableMemory")
            .field("len", &self.len)
            .finish()
    }
}

impl ExecutableMemory {
    /// Allocate a fresh executable region, copy `code` into it, and return
    /// the owning handle with the region marked read/execute.
    ///
    /// Returns [`ExecutableMemoryError::Unsupported`] on build targets that
    /// do not have a backend (currently any non-Unix, non-Windows-x86_64
    /// target).
    pub fn new(code: &[u8]) -> Result<Self, ExecutableMemoryError> {
        if code.is_empty() {
            return Err(ExecutableMemoryError::ZeroSize);
        }
        imp::allocate(code).map(|(ptr, len)| Self { ptr, len })
    }

    /// Returns a read-only pointer to the start of the executable region.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Returns a mutable pointer to the start of the executable region.
    ///
    /// The pointer aliases the same backing memory as [`Self::as_ptr`]; it
    /// is exposed only because the existing JIT call sites transmute it
    /// into an `extern "C"` function pointer for invocation.  Callers must
    /// **not** write through this pointer — the region is mapped
    /// read/execute and writes will fault.
    #[inline]
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Length of the executable region in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True iff the executable region is empty.  Currently always false
    /// because [`Self::new`] rejects empty buffers, but exposed to satisfy
    /// the standard `len`/`is_empty` clippy lint.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// View the executable region as a byte slice.  Useful for callers that
    /// need to read the code back (e.g. to seed a sibling cache).
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `ptr` and `len` were established by `imp::allocate` and
        // remain valid until `Drop`.  The region is mapped read/execute so
        // ordinary loads are well-defined.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for ExecutableMemory {
    fn drop(&mut self) {
        // SAFETY: `ptr` and `len` were established by `imp::allocate` and
        // have not been mutated since.
        unsafe { imp::release(self.ptr, self.len) };
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-platform implementations
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(unix)]
mod imp {
    use super::ExecutableMemoryError;
    use std::ptr::NonNull;

    pub(super) fn allocate(code: &[u8]) -> Result<(NonNull<u8>, usize), ExecutableMemoryError> {
        let len = code.len();
        // SAFETY: arguments are valid; MAP_FAILED is checked before use.
        let mem = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if mem == libc::MAP_FAILED {
            return Err(ExecutableMemoryError::AllocationFailed);
        }
        // SAFETY: `mem` is page-aligned and sized for `len` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), mem.cast::<u8>(), len);
        }
        // Flip to R+X (W^X).
        // SAFETY: `mem` and `len` describe the freshly-mapped region.
        let rc = unsafe { libc::mprotect(mem, len, libc::PROT_READ | libc::PROT_EXEC) };
        if rc != 0 {
            // SAFETY: undo the partial allocation before reporting failure.
            unsafe {
                libc::munmap(mem, len);
            }
            return Err(ExecutableMemoryError::ProtectionChangeFailed);
        }
        // SAFETY: a successful `mmap` never returns null.
        let ptr = unsafe { NonNull::new_unchecked(mem.cast::<u8>()) };
        Ok((ptr, len))
    }

    /// # Safety
    /// `ptr` / `len` must originate from a previous call to [`allocate`].
    pub(super) unsafe fn release(ptr: NonNull<u8>, len: usize) {
        // SAFETY: precondition delegates validity to the caller.
        unsafe {
            libc::munmap(ptr.as_ptr().cast(), len);
        }
    }
}

#[cfg(all(target_arch = "x86_64", windows))]
mod imp {
    use super::ExecutableMemoryError;
    use std::ptr::NonNull;
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_READWRITE, VirtualAlloc,
        VirtualFree, VirtualProtect,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    pub(super) fn allocate(code: &[u8]) -> Result<(NonNull<u8>, usize), ExecutableMemoryError> {
        let len = code.len();
        // SAFETY: arguments are valid; the result is checked before use.
        let mem = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        };
        if mem.is_null() {
            return Err(ExecutableMemoryError::AllocationFailed);
        }
        // SAFETY: `mem` is committed and sized for `len` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), mem.cast::<u8>(), len);
        }
        let mut old_protect: u32 = 0;
        // SAFETY: `mem` / `len` describe the freshly-allocated region;
        // `old_protect` is a valid `&mut u32`.
        let ok = unsafe { VirtualProtect(mem, len, PAGE_EXECUTE_READ, &mut old_protect) };
        if ok == 0 {
            // SAFETY: undo the allocation; ignore the result intentionally.
            unsafe {
                VirtualFree(mem, 0, MEM_RELEASE);
            }
            return Err(ExecutableMemoryError::ProtectionChangeFailed);
        }
        // x86-64 has coherent I/D caches but `FlushInstructionCache` is the
        // documented contract for newly-written code on Windows.
        // SAFETY: `GetCurrentProcess` returns a pseudo-handle that is always
        // valid for the current process; `mem` / `len` describe a committed
        // region.
        unsafe {
            FlushInstructionCache(GetCurrentProcess(), mem, len);
        }
        // SAFETY: successful `VirtualAlloc` never returns null.
        let ptr = unsafe { NonNull::new_unchecked(mem.cast::<u8>()) };
        Ok((ptr, len))
    }

    /// # Safety
    /// `ptr` must originate from a previous call to [`allocate`]; `len` is
    /// accepted for symmetry with the Unix path but is unused here because
    /// `MEM_RELEASE` always frees the entire reservation.
    pub(super) unsafe fn release(ptr: NonNull<u8>, _len: usize) {
        // SAFETY: precondition delegates validity to the caller.
        unsafe {
            VirtualFree(ptr.as_ptr().cast(), 0, MEM_RELEASE);
        }
    }
}

#[cfg(not(any(unix, all(target_arch = "x86_64", windows))))]
mod imp {
    use super::ExecutableMemoryError;
    use std::ptr::NonNull;

    pub(super) fn allocate(_code: &[u8]) -> Result<(NonNull<u8>, usize), ExecutableMemoryError> {
        Err(ExecutableMemoryError::Unsupported)
    }

    /// # Safety
    /// Unreachable on unsupported targets — `allocate` never succeeds.
    pub(super) unsafe fn release(_ptr: NonNull<u8>, _len: usize) {}
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Allocating an empty buffer is rejected on every target.
    #[test]
    fn empty_buffer_is_rejected() {
        assert_eq!(
            ExecutableMemory::new(&[]).unwrap_err(),
            ExecutableMemoryError::ZeroSize
        );
    }

    /// On unsupported targets the abstraction must report `Unsupported`
    /// rather than allocate or crash.
    #[cfg(not(any(unix, all(target_arch = "x86_64", windows))))]
    #[test]
    fn unsupported_targets_report_unsupported() {
        // A non-empty buffer of NOPs (architecture-agnostic placeholder).
        let buf = [0u8; 4];
        assert_eq!(
            ExecutableMemory::new(&buf).unwrap_err(),
            ExecutableMemoryError::Unsupported
        );
    }

    /// Allocate, copy, and read back a tiny payload on supported targets.
    /// This validates the W^X allocation path without invoking the bytes
    /// (so it is safe on every architecture, not just x86-64).
    #[cfg(any(unix, all(target_arch = "x86_64", windows)))]
    #[test]
    fn allocates_and_exposes_code_bytes() {
        // Single `ret` byte on x86-64 (`0xC3`).  On other architectures
        // the value is irrelevant — we only assert the byte round-trips.
        let payload: [u8; 4] = [0xC3, 0x90, 0x90, 0x90];
        let mem = ExecutableMemory::new(&payload).expect("allocation must succeed");
        assert_eq!(mem.len(), payload.len());
        assert!(!mem.is_empty());
        assert_eq!(mem.as_bytes(), &payload);
        assert!(!mem.as_ptr().is_null());
    }

    /// On x86-64 (Unix or Windows) we can additionally verify execution by
    /// emitting a trivial `mov eax, imm32; ret` and calling it through a
    /// transmuted `extern "C"` function pointer.  This is the smoke test
    /// that exercises the full allocate → copy → protect → execute path.
    #[cfg(all(target_arch = "x86_64", any(unix, windows)))]
    #[test]
    fn executes_tiny_x86_64_blob() {
        // mov eax, 0x2A     ; B8 2A 00 00 00
        // ret               ; C3
        let code: [u8; 6] = [0xB8, 0x2A, 0x00, 0x00, 0x00, 0xC3];
        let mem = ExecutableMemory::new(&code).expect("allocation must succeed");
        // SAFETY: `mem` holds a valid x86-64 function with signature
        // `extern "C" fn() -> u32` returning the immediate constant 42.
        let result = unsafe {
            let f: extern "C" fn() -> u32 = std::mem::transmute(mem.as_ptr());
            f()
        };
        assert_eq!(result, 42);
    }
}
