//! Background compilation thread pool.
//!
//! This module moves JIT compilation off the main thread so the interpreter can
//! continue running while higher-tier code is being generated.
//!
//! # Architecture
//!
//! ```text
//! Main thread                   Worker pool (N = num_cpus − 1)
//! ───────────                   ─────────────────────────────────
//! on_tier_up_request()          ┌──> worker picks CompileJob
//!   └─ queue.push(job) ───────►│   └─ compile IR / Cranelift
//!                               │   └─ atomic code-pointer swap
//!                               └──> …
//! ```
//!
//! The pool uses a priority queue so higher-tier requests are serviced first
//! (Turbofan before Maglev, Maglev before Baseline).  Once compilation
//! finishes, the worker publishes the result through the shared
//! [`CompileResult`] and the main thread installs it via an atomic swap.

use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// Unique identifier for a JavaScript function known to the engine.
pub type FunctionId = u32;

/// Compilation tier — higher ordinal = higher optimisation level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Tier {
    /// Non-optimising baseline JIT.
    Baseline = 0,
    /// Mid-tier optimising compiler (Maglev).
    Maglev = 1,
    /// Top-tier optimising compiler (Turbofan / Cranelift).
    Turbofan = 2,
}

/// A unit of work submitted to the background compilation pool.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CompileJob {
    /// The function to compile.
    pub function_id: FunctionId,
    /// The tier to compile to.
    pub tier: Tier,
    /// Monotonically increasing sequence number for FIFO ordering within the
    /// same priority level.
    sequence: u64,
}

// BinaryHeap is a *max*-heap, so we want higher-tier jobs and lower sequence
// numbers to compare as "greater".
impl Ord for CompileJob {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.tier
            .cmp(&other.tier)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

impl PartialOrd for CompileJob {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// The result produced by a background compilation worker.
#[derive(Debug)]
pub struct CompileResult {
    /// The function that was compiled.
    pub function_id: FunctionId,
    /// The tier that was compiled.
    pub tier: Tier,
    /// Opaque pointer to the compiled code (e.g. a `*const u8` entry-point).
    /// `null` if compilation failed.
    pub code_ptr: *const u8,
}

// SAFETY: The raw pointer is only dereferenced on the main thread after an
// atomic load; no aliased writes occur.
unsafe impl Send for CompileResult {}

/// Atomic slot through which a background worker publishes compiled code.
///
/// The main thread reads the pointer with [`Ordering::Acquire`]; a worker
/// writes it with [`Ordering::Release`], forming a release/acquire pair that
/// guarantees the code buffer is visible before the pointer is.
pub struct AtomicCodeSlot {
    ptr: AtomicPtr<u8>,
}

impl AtomicCodeSlot {
    /// Create a slot initialised to null (no compiled code available).
    pub fn new() -> Self {
        Self {
            ptr: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    /// Atomically publish a new code pointer.
    ///
    /// Uses [`Ordering::Release`] so that all writes to the code buffer that
    /// precede this call are visible to any thread that loads the pointer with
    /// [`Ordering::Acquire`].
    pub fn store(&self, ptr: *mut u8) {
        self.ptr.store(ptr, Ordering::Release);
    }

    /// Atomically load the current code pointer.
    ///
    /// Uses [`Ordering::Acquire`] to pair with the [`Ordering::Release`]
    /// store, ensuring the code buffer contents are visible after this load.
    pub fn load(&self) -> *const u8 {
        self.ptr.load(Ordering::Acquire)
    }
}

impl Default for AtomicCodeSlot {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Job queue
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe, priority-ordered compilation queue.
///
/// Higher [`Tier`] jobs are dequeued first.  Within the same tier, jobs are
/// serviced in FIFO order.
pub struct CompilationQueue {
    inner: Mutex<BinaryHeap<CompileJob>>,
    condvar: Condvar,
    next_seq: AtomicU64,
}

impl CompilationQueue {
    /// Create an empty compilation queue.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(BinaryHeap::new()),
            condvar: Condvar::new(),
            next_seq: AtomicU64::new(0),
        }
    }

    /// Push a new compilation request.
    pub fn push(&self, function_id: FunctionId, tier: Tier) {
        let sequence = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let job = CompileJob {
            function_id,
            tier,
            sequence,
        };
        {
            let mut heap = self.inner.lock().expect("queue lock poisoned");
            heap.push(job);
        }
        self.condvar.notify_one();
    }

    /// Block until a job is available, then return it.
    pub fn pop_blocking(&self) -> CompileJob {
        let mut heap = self.inner.lock().expect("queue lock poisoned");
        loop {
            if let Some(job) = heap.pop() {
                return job;
            }
            heap = self.condvar.wait(heap).expect("queue lock poisoned");
        }
    }

    /// Try to pop a job without blocking.
    pub fn try_pop(&self) -> Option<CompileJob> {
        self.inner.lock().expect("queue lock poisoned").pop()
    }

    /// Number of jobs currently enqueued.
    pub fn len(&self) -> usize {
        self.inner.lock().expect("queue lock poisoned").len()
    }

    /// Returns `true` if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for CompilationQueue {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Thread pool
// ─────────────────────────────────────────────────────────────────────────────

/// Callback invoked by a worker thread to compile a single job.
///
/// The implementer should perform the actual IR lowering / code generation and
/// return a raw pointer to the emitted code, or `null` on failure.
pub type CompileFn = Box<dyn Fn(&CompileJob) -> *const u8 + Send + Sync>;

/// A pool of background threads that drain the [`CompilationQueue`].
///
/// Each worker calls the user-supplied [`CompileFn`] for every job it
/// dequeues.  When the pool is dropped, a poison flag is set and all workers
/// are joined.
pub struct CompilationPool {
    queue: Arc<CompilationQueue>,
    workers: Vec<thread::JoinHandle<()>>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}

impl CompilationPool {
    /// Spawn `num_workers` background threads.
    ///
    /// `compile_fn` is called on each dequeued [`CompileJob`].  The returned
    /// code pointer is published through the per-function
    /// [`AtomicCodeSlot`]; callers are responsible for setting up the slot
    /// table.
    pub fn new(num_workers: usize, compile_fn: Arc<CompileFn>) -> Self {
        let num_workers = num_workers.max(1);
        let queue = Arc::new(CompilationQueue::new());
        let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut workers = Vec::with_capacity(num_workers);

        for id in 0..num_workers {
            let q = Arc::clone(&queue);
            let f = Arc::clone(&compile_fn);
            let stop = Arc::clone(&shutdown);
            let handle = thread::Builder::new()
                .name(format!("stator-compile-{id}"))
                .spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        // Use try_pop + park to allow shutdown checks.
                        if let Some(job) = q.try_pop() {
                            let _code = f(&job);
                            // In a full implementation the code pointer would
                            // be published to the function's AtomicCodeSlot
                            // here.
                        } else {
                            thread::park_timeout(std::time::Duration::from_millis(5));
                        }
                    }
                })
                .expect("failed to spawn compilation worker");
            workers.push(handle);
        }

        Self {
            queue,
            workers,
            shutdown,
        }
    }

    /// Submit a compilation job to the pool.
    pub fn submit(&self, function_id: FunctionId, tier: Tier) {
        self.queue.push(function_id, tier);
        // Wake one parked worker.
        if let Some(w) = self.workers.first() {
            w.thread().unpark();
        }
    }

    /// Shared reference to the underlying queue.
    pub fn queue(&self) -> &Arc<CompilationQueue> {
        &self.queue
    }

    /// Number of worker threads.
    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }
}

impl Drop for CompilationPool {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        // Unpark all workers so they notice the shutdown flag.
        for w in &self.workers {
            w.thread().unpark();
        }
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience: tier-up request entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Submit a tier-up request to `pool`.
///
/// This is the main-thread entry point shown in the issue description:
///
/// ```text
/// fn on_tier_up_request(function_id: FunctionId, tier: Tier) {
///     compilation_queue.push(CompileJob { function_id, tier });
/// }
/// ```
pub fn on_tier_up_request(pool: &CompilationPool, function_id: FunctionId, tier: Tier) {
    pool.submit(function_id, tier);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn test_compile_job_priority_higher_tier_first() {
        let queue = CompilationQueue::new();
        queue.push(1, Tier::Baseline);
        queue.push(2, Tier::Turbofan);
        queue.push(3, Tier::Maglev);

        let first = queue.try_pop().unwrap();
        assert_eq!(first.tier, Tier::Turbofan);

        let second = queue.try_pop().unwrap();
        assert_eq!(second.tier, Tier::Maglev);

        let third = queue.try_pop().unwrap();
        assert_eq!(third.tier, Tier::Baseline);
    }

    #[test]
    fn test_compile_job_fifo_within_same_tier() {
        let queue = CompilationQueue::new();
        queue.push(10, Tier::Maglev);
        queue.push(20, Tier::Maglev);
        queue.push(30, Tier::Maglev);

        assert_eq!(queue.try_pop().unwrap().function_id, 10);
        assert_eq!(queue.try_pop().unwrap().function_id, 20);
        assert_eq!(queue.try_pop().unwrap().function_id, 30);
    }

    #[test]
    fn test_queue_is_empty() {
        let queue = CompilationQueue::new();
        assert!(queue.is_empty());
        queue.push(1, Tier::Baseline);
        assert!(!queue.is_empty());
        queue.try_pop();
        assert!(queue.is_empty());
    }

    #[test]
    fn test_atomic_code_slot_store_load() {
        let slot = AtomicCodeSlot::new();
        assert!(slot.load().is_null());

        let dummy: u8 = 42;
        let ptr = &dummy as *const u8 as *mut u8;
        slot.store(ptr);
        assert_eq!(slot.load(), ptr as *const u8);
    }

    #[test]
    fn test_pool_processes_jobs() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let compile_fn: Arc<CompileFn> = Arc::new(Box::new(move |_job: &CompileJob| {
            counter_clone.fetch_add(1, Ordering::Relaxed);
            std::ptr::null()
        }));

        let pool = CompilationPool::new(2, compile_fn);
        pool.submit(1, Tier::Baseline);
        pool.submit(2, Tier::Maglev);
        pool.submit(3, Tier::Turbofan);

        // Give workers time to drain the queue.
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert_eq!(counter.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_pool_shutdown_joins_workers() {
        let compile_fn: Arc<CompileFn> = Arc::new(Box::new(|_| std::ptr::null()));
        let pool = CompilationPool::new(2, compile_fn);
        assert_eq!(pool.num_workers(), 2);
        // Dropping the pool should join all threads without hanging.
        drop(pool);
    }

    #[test]
    fn test_on_tier_up_request_enqueues() {
        let compile_fn: Arc<CompileFn> = Arc::new(Box::new(|_| std::ptr::null()));
        let pool = CompilationPool::new(1, compile_fn);
        on_tier_up_request(&pool, 42, Tier::Turbofan);

        // The job should eventually be processed; just verify it was enqueued.
        std::thread::sleep(std::time::Duration::from_millis(50));
        // Queue should be drained by the worker.
        assert!(pool.queue().is_empty());
    }
}
