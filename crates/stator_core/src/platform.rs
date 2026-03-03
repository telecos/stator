//! Platform abstraction: exposes engine hooks (task scheduling, timers) for
//! embedder customisation.

/// Host-side platform services consumed by the Stator engine.
///
/// Embedders implement this trait to provide custom threading, timing, and
/// task-scheduling behaviour.  The vtable-backed implementation in
/// `stator_ffi` wraps C function pointers into this interface.
pub trait Platform: Send {
    /// Return the number of worker threads the platform makes available to the
    /// engine for background work (compilation, GC helpers, etc.).
    fn number_of_worker_threads(&self) -> u32;

    /// Schedule `task` for eventual execution on a platform-managed thread.
    ///
    /// Ownership of the raw task pointer is transferred to the platform; the
    /// platform is responsible for ensuring the task is eventually executed or
    /// deallocated.
    ///
    /// # Safety
    /// `task` must be a non-null pointer whose lifetime is managed by the
    /// caller prior to this call.  After this call the caller must not access
    /// `task` again.
    unsafe fn post_task(&self, task: *mut std::ffi::c_void);

    /// Return a monotonically increasing time value in seconds.
    ///
    /// The epoch is unspecified; only differences between two readings are
    /// meaningful.
    fn monotonically_increasing_time(&self) -> f64;

    /// Return the current wall-clock time in milliseconds since the Unix epoch.
    fn current_clock_time_millis(&self) -> f64;
}
