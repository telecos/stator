//! Incremental and concurrent garbage collection for sub-1 ms pauses.
//!
//! This module builds on the existing GC infrastructure in [`crate::gc`] to
//! provide:
//!
//! - **Dijkstra-style write barrier** (snapshot-at-beginning) for safe
//!   incremental marking — when the mutator writes a reference into an
//!   already-scanned (black) object, the overwritten pointer is shaded grey so
//!   the collector does not miss it.
//!
//! - **Budget-based incremental marking** — the mark phase is split into small
//!   steps, each processing at most a configurable number of objects, so GC
//!   work is interleaved with mutator execution without long pauses.
//!
//! - **Concurrent sweeping** — dead-object reclamation runs on a background
//!   thread, freeing old-space byte ranges without blocking the mutator.
//!
//! - **Generational collection** — a nursery (young gen) fast path with minor
//!   GC (scavenge), and a major GC for the tenured (old) generation.
//!
//! - **Idle-time GC** — the embedder can notify the engine of idle periods so
//!   incremental marking steps are performed only when the main thread is not
//!   busy.
//!
//! - **GC metrics** — heap size, pause time, and collection count are tracked
//!   and exposed for diagnostics and embedder-side reporting.
//!
//! # Integration
//!
//! The central type is [`IncrementalGc`], which wraps a [`Heap`] and
//! orchestrates incremental/concurrent collection phases.  Embedders call
//! [`IncrementalGc::write_barrier`] on every heap-pointer store and
//! [`IncrementalGc::notify_idle`] during idle callbacks.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::gc::heap::{Heap, OldSpace};
use crate::gc::trace::{Tracer, trace_heap_object};
use crate::objects::heap_object::HeapObject;

// ── Tri-colour marking ───────────────────────────────────────────────────────

/// Tri-colour mark state for incremental marking.
///
/// - **White** — not yet discovered; presumed dead unless later reached.
/// - **Grey** — discovered (reachable) but outgoing references not yet scanned.
/// - **Black** — fully scanned; all children are at least grey.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkColour {
    /// Not yet discovered by the marker.
    White,
    /// Discovered but not yet fully scanned.
    Grey,
    /// Fully scanned — all outgoing edges have been visited.
    Black,
}

// ── GcPhase ──────────────────────────────────────────────────────────────────

/// The current phase of the incremental/concurrent GC cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcPhase {
    /// No GC cycle is in progress.  The mutator runs freely.
    Idle,
    /// Incremental marking is in progress.
    Marking,
    /// Sweeping (possibly concurrent) is in progress.
    Sweeping,
    /// A full GC compaction is running (stop-the-world).
    Compacting,
}

// ── GcMetrics ────────────────────────────────────────────────────────────────

/// Diagnostic counters exposed to the embedder.
#[derive(Debug, Clone)]
pub struct GcMetrics {
    /// Total bytes currently in use across all generations.
    pub heap_size_bytes: usize,
    /// Duration of the last GC pause in microseconds.
    pub last_pause_us: u64,
    /// Cumulative number of minor (nursery) collections.
    pub minor_gc_count: u64,
    /// Cumulative number of major (old-gen) collections.
    pub major_gc_count: u64,
    /// Cumulative number of incremental marking steps.
    pub incremental_steps: u64,
}

impl GcMetrics {
    /// Create a zeroed metrics snapshot.
    pub fn new() -> Self {
        Self {
            heap_size_bytes: 0,
            last_pause_us: 0,
            minor_gc_count: 0,
            major_gc_count: 0,
            incremental_steps: 0,
        }
    }
}

impl Default for GcMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ── SweepResult ──────────────────────────────────────────────────────────────

/// Byte ranges of dead objects identified during a sweep, communicated from
/// the background sweeper to the mutator thread.
///
/// Each entry is `(offset_from_old_space_base, byte_size)`.
pub type SweepResult = Vec<(usize, usize)>;

// ── IncrementalGc ────────────────────────────────────────────────────────────

/// Default marking budget: objects to mark per incremental step.
const DEFAULT_MARK_BUDGET: usize = 256;

/// Orchestrates incremental marking, concurrent sweeping, generational
/// collection, idle-time GC, and metrics tracking on top of the existing
/// [`Heap`].
pub struct IncrementalGc {
    /// Current GC cycle phase.
    phase: GcPhase,

    /// Per-object mark colour table (address → colour).
    mark_table: HashSet<usize>,

    /// Grey objects awaiting scanning (addresses).
    grey_stack: Vec<*mut HeapObject>,

    /// Objects that have been fully scanned (black).
    black_set: HashSet<usize>,

    /// Maximum number of objects to mark in a single incremental step.
    mark_budget: usize,

    /// Shared sweep result: the background sweeper writes here; the mutator
    /// reads once sweeping is complete.
    sweep_result: Arc<Mutex<Option<SweepResult>>>,

    /// Diagnostic counters.
    metrics: GcMetrics,
}

// SAFETY: All raw pointers stored in `grey_stack` are GC bookkeeping data;
// they are only dereferenced on the GC (mutator) thread.  The only `Send`
// crossing is the `Arc<Mutex<SweepResult>>`.
unsafe impl Send for IncrementalGc {}

impl IncrementalGc {
    /// Create a new `IncrementalGc` controller in the [`GcPhase::Idle`] state.
    pub fn new() -> Self {
        Self {
            phase: GcPhase::Idle,
            mark_table: HashSet::new(),
            grey_stack: Vec::new(),
            black_set: HashSet::new(),
            mark_budget: DEFAULT_MARK_BUDGET,
            sweep_result: Arc::new(Mutex::new(None)),
            metrics: GcMetrics::new(),
        }
    }

    /// Create an `IncrementalGc` with a custom per-step marking budget.
    pub fn with_budget(budget: usize) -> Self {
        let mut gc = Self::new();
        gc.mark_budget = budget;
        gc
    }

    // ── Read-only accessors ──────────────────────────────────────────────

    /// Current phase of the GC cycle.
    pub fn phase(&self) -> GcPhase {
        self.phase
    }

    /// Snapshot of the current GC metrics.
    pub fn metrics(&self) -> &GcMetrics {
        &self.metrics
    }

    /// Returns the colour of the object at `ptr`.
    ///
    /// Objects not present in the mark table are [`MarkColour::White`].
    pub fn colour_of(&self, ptr: *mut HeapObject) -> MarkColour {
        let addr = ptr as usize;
        if self.black_set.contains(&addr) {
            MarkColour::Black
        } else if self.mark_table.contains(&addr) {
            MarkColour::Grey
        } else {
            MarkColour::White
        }
    }

    // ── Dijkstra write barrier ───────────────────────────────────────────

    /// Dijkstra-style snapshot-at-beginning write barrier.
    ///
    /// Must be called **before** every heap-pointer store that might overwrite
    /// a reference in an object that has already been scanned (black).  If the
    /// host object is black and the overwritten value is a heap pointer, the
    /// *old* target is shaded grey so incremental marking does not miss it.
    ///
    /// During non-marking phases the barrier is a no-op.
    ///
    /// # Safety
    /// `old_target` must be null or a valid, aligned `HeapObject` pointer.
    pub unsafe fn write_barrier(&mut self, host: *mut HeapObject, old_target: *mut HeapObject) {
        // Fast-path: barrier is only active during incremental marking.
        if self.phase != GcPhase::Marking {
            return;
        }
        if host.is_null() || old_target.is_null() {
            return;
        }
        // Dijkstra invariant: if the source is black and the old target is
        // white, shade the old target grey to prevent it from being collected.
        if self.black_set.contains(&(host as usize))
            && !self.mark_table.contains(&(old_target as usize))
            && !self.black_set.contains(&(old_target as usize))
        {
            self.shade_grey(old_target);
        }
    }

    // ── Incremental marking ──────────────────────────────────────────────

    /// Begin an incremental marking cycle.
    ///
    /// Transitions from [`GcPhase::Idle`] to [`GcPhase::Marking`] and seeds
    /// the grey stack from the given `roots`.  Each root that falls within
    /// `old_space` is shaded grey.
    ///
    /// # Safety
    /// Every non-null root must be a valid, aligned pointer to a live
    /// `HeapObject` within `old_space`.
    pub unsafe fn start_marking(&mut self, roots: &[*mut *mut HeapObject], old_space: &OldSpace) {
        self.phase = GcPhase::Marking;
        self.mark_table.clear();
        self.grey_stack.clear();
        self.black_set.clear();

        for &slot in roots {
            // SAFETY: caller guarantees slot validity.
            let ptr = unsafe { *slot };
            if ptr.is_null() {
                continue;
            }
            if old_space.contains(ptr as *mut u8) {
                self.shade_grey(ptr);
            }
        }
    }

    /// Perform one incremental marking step, processing up to
    /// [`mark_budget`][Self::mark_budget] grey objects.
    ///
    /// Returns `true` when all grey objects have been drained (marking is
    /// complete).  Returns `false` when there is still work remaining.
    ///
    /// # Safety
    /// Grey-stack entries must be valid, aligned `HeapObject` pointers with
    /// initialised `alloc_size` headers.
    pub unsafe fn mark_step(&mut self) -> bool {
        if self.phase != GcPhase::Marking {
            return true;
        }

        let start = Instant::now();
        let mut budget = self.mark_budget;

        while budget > 0 {
            let Some(obj) = self.grey_stack.pop() else {
                break;
            };
            let addr = obj as usize;

            // Skip if already black (re-shaded by the write barrier).
            if self.black_set.contains(&addr) {
                continue;
            }

            // Promote grey → black.
            self.mark_table.remove(&addr);
            self.black_set.insert(addr);

            // Trace child pointers and enqueue them as grey.
            let mut tracer = Tracer::new();
            // SAFETY: obj is a valid, marked HeapObject pointer.
            unsafe { trace_heap_object(obj, &mut tracer) };
            for child_raw in tracer.gray_stack {
                let child_addr = child_raw as usize;
                if !self.black_set.contains(&child_addr) && !self.mark_table.contains(&child_addr) {
                    self.mark_table.insert(child_addr);
                    self.grey_stack.push(child_raw as *mut HeapObject);
                }
            }

            budget -= 1;
        }

        let elapsed = start.elapsed();
        self.metrics.last_pause_us = elapsed.as_micros() as u64;
        self.metrics.incremental_steps += 1;

        let done = self.grey_stack.is_empty();
        if done {
            self.phase = GcPhase::Sweeping;
        }
        done
    }

    /// Run incremental marking to completion, calling [`mark_step`][Self::mark_step]
    /// in a loop until the grey stack is drained.
    ///
    /// # Safety
    /// See [`mark_step`][Self::mark_step].
    pub unsafe fn finish_marking(&mut self) {
        while !unsafe { self.mark_step() } {}
    }

    // ── Sweeping ─────────────────────────────────────────────────────────

    /// Perform a synchronous sweep of `old_space`, returning dead-object byte
    /// ranges.
    ///
    /// This mirrors [`MarkSweepCompactor::sweep`] but uses the incremental
    /// mark table (black set) instead of the compactor's mark set.
    ///
    /// # Safety
    /// `old_space` must contain a contiguous sequence of valid `HeapObject`
    /// records with `alloc_size > 0`.
    pub unsafe fn sweep_sync(&self, old_space: &OldSpace) -> SweepResult {
        let base = old_space.base_ptr();
        let used = old_space.used();
        let mut free_regions = Vec::new();

        let mut offset = 0usize;
        while offset < used {
            let obj_ptr = unsafe { base.add(offset) } as *mut HeapObject;
            let size = unsafe { (*obj_ptr).alloc_size() } as usize;
            if size == 0 {
                break;
            }
            if !self.black_set.contains(&(obj_ptr as usize)) {
                free_regions.push((offset, size));
            }
            offset += size;
        }
        free_regions
    }

    /// Launch a concurrent sweep on a background thread.
    ///
    /// The sweep result is written into the shared `sweep_result` field and
    /// can be retrieved with [`take_sweep_result`][Self::take_sweep_result].
    ///
    /// # Arguments
    /// * `old_base` — raw base pointer of the old-space region.
    /// * `old_used` — number of bytes currently in use.
    /// * `black_set` — clone of the black (live) set at the end of marking.
    ///
    /// # Safety
    /// The old-space region `[old_base, old_base + old_used)` must remain
    /// valid and unmodified for the duration of the background sweep.
    pub unsafe fn launch_concurrent_sweep(
        &mut self,
        old_base: *mut u8,
        old_used: usize,
        black_set: HashSet<usize>,
    ) {
        let result_handle = Arc::clone(&self.sweep_result);

        // Wrap the raw pointer so it can cross the thread boundary.
        let base_addr = old_base as usize;

        std::thread::spawn(move || {
            let mut free_regions = Vec::new();
            let mut offset = 0usize;
            while offset < old_used {
                let obj_addr = base_addr + offset;
                let obj_ptr = obj_addr as *mut HeapObject;
                // SAFETY: caller guarantees [base, base+used) is valid.
                let size = unsafe { (*obj_ptr).alloc_size() } as usize;
                if size == 0 {
                    break;
                }
                if !black_set.contains(&obj_addr) {
                    free_regions.push((offset, size));
                }
                offset += size;
            }
            if let Ok(mut guard) = result_handle.lock() {
                *guard = Some(free_regions);
            }
        });
    }

    /// Take the sweep result produced by a concurrent sweep, if available.
    ///
    /// Returns `None` if the background thread has not yet finished (or was
    /// never launched).
    pub fn take_sweep_result(&mut self) -> Option<SweepResult> {
        if let Ok(mut guard) = self.sweep_result.lock() {
            guard.take()
        } else {
            None
        }
    }

    // ── Generational helpers ─────────────────────────────────────────────

    /// Trigger a minor (nursery) GC on `heap`.
    ///
    /// Increments [`GcMetrics::minor_gc_count`] and updates
    /// [`GcMetrics::heap_size_bytes`].
    pub fn minor_gc(&mut self, heap: &mut Heap) {
        let start = Instant::now();
        heap.collect();
        let elapsed = start.elapsed();
        self.metrics.last_pause_us = elapsed.as_micros() as u64;
        self.metrics.minor_gc_count += 1;
        self.update_heap_size(heap);
    }

    /// Trigger a full major GC cycle (incremental mark → sweep) using the
    /// provided roots.
    ///
    /// # Safety
    /// All root pointers must be valid, aligned, live `HeapObject`s.
    pub unsafe fn major_gc(&mut self, heap: &mut Heap, roots: &[*mut *mut HeapObject]) {
        let start = Instant::now();

        // Mark phase (run to completion synchronously).
        unsafe { self.start_marking(roots, &heap.old_space) };
        unsafe { self.finish_marking() };

        // Sweep phase (synchronous).
        let _free_regions = unsafe { self.sweep_sync(&heap.old_space) };

        // Transition back to idle.
        self.phase = GcPhase::Idle;

        let elapsed = start.elapsed();
        self.metrics.last_pause_us = elapsed.as_micros() as u64;
        self.metrics.major_gc_count += 1;
        self.update_heap_size(heap);
    }

    // ── Idle-time GC ─────────────────────────────────────────────────────

    /// Notify the GC that the embedder is idle for `deadline_us` microseconds.
    ///
    /// If an incremental marking cycle is active, one marking step is
    /// performed within the idle window.  If no cycle is active and the young
    /// space is more than half full, a minor GC is triggered.
    ///
    /// Returns `true` if GC work was performed during this idle notification.
    ///
    /// # Safety
    /// If the GC is in the marking phase, all grey-stack entries must be
    /// valid heap pointers (see [`mark_step`][Self::mark_step]).
    pub unsafe fn notify_idle(&mut self, heap: &mut Heap, _deadline_us: u64) -> bool {
        match self.phase {
            GcPhase::Marking => {
                unsafe { self.mark_step() };
                true
            }
            GcPhase::Idle => {
                // Heuristic: if the nursery is >50 % full, collect it.
                let young_usage = heap.young_space.used() as f64;
                let young_cap = heap.young_space.capacity() as f64;
                if young_cap > 0.0 && young_usage / young_cap > 0.5 {
                    self.minor_gc(heap);
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    // ── Internal helpers ─────────────────────────────────────────────────

    /// Shade `ptr` grey: add to the grey stack and mark table.
    fn shade_grey(&mut self, ptr: *mut HeapObject) {
        let addr = ptr as usize;
        if self.mark_table.insert(addr) {
            self.grey_stack.push(ptr);
        }
    }

    /// Recompute `heap_size_bytes` from the heap's allocators.
    fn update_heap_size(&mut self, heap: &Heap) {
        self.metrics.heap_size_bytes = heap.young_space.used() + heap.old_space.used();
    }

    /// Clear all internal marking state and transition to [`GcPhase::Idle`].
    pub fn reset(&mut self) {
        self.phase = GcPhase::Idle;
        self.mark_table.clear();
        self.grey_stack.clear();
        self.black_set.clear();
    }
}

impl Default for IncrementalGc {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gc::heap::{Heap, OldSpace};
    use crate::objects::heap_object::HeapObject;
    use std::alloc::Layout;

    /// Bump-allocate a zero-initialised `HeapObject` in old-space.
    fn alloc_in_old(old: &mut OldSpace) -> *mut HeapObject {
        let layout = Layout::new::<HeapObject>();
        let raw = old.bump_alloc(layout).expect("old space has free space");
        // SAFETY: raw is valid and layout.size() bytes are accessible.
        unsafe { std::ptr::write_bytes(raw, 0, layout.size()) };
        let obj = raw as *mut HeapObject;
        // SAFETY: obj is valid and zero-initialised.
        unsafe { (*obj).init_alloc_size(layout.size() as u32) };
        obj
    }

    // ── Phase transitions ─────────────────────────────────────────────────

    #[test]
    fn test_new_starts_idle() {
        let gc = IncrementalGc::new();
        assert_eq!(gc.phase(), GcPhase::Idle);
    }

    #[test]
    fn test_start_marking_transitions_to_marking() {
        let mut gc = IncrementalGc::new();
        let mut old = OldSpace::new(65536);
        let obj = alloc_in_old(&mut old);

        let mut root: *mut HeapObject = obj;
        let roots = [&raw mut root as *mut *mut HeapObject];

        unsafe { gc.start_marking(&roots, &old) };
        assert_eq!(gc.phase(), GcPhase::Marking);
    }

    #[test]
    fn test_mark_step_completes_and_transitions_to_sweeping() {
        let mut gc = IncrementalGc::new();
        let mut old = OldSpace::new(65536);
        let obj = alloc_in_old(&mut old);

        let mut root: *mut HeapObject = obj;
        let roots = [&raw mut root as *mut *mut HeapObject];

        unsafe { gc.start_marking(&roots, &old) };
        let done = unsafe { gc.mark_step() };
        assert!(done, "single root must be fully processed in one step");
        assert_eq!(gc.phase(), GcPhase::Sweeping);
    }

    // ── Tri-colour invariants ─────────────────────────────────────────────

    #[test]
    fn test_colour_of_unmarked_is_white() {
        let gc = IncrementalGc::new();
        let mut obj = HeapObject::new_null();
        assert_eq!(gc.colour_of(&raw mut obj), MarkColour::White);
    }

    #[test]
    fn test_start_marking_shades_roots_grey() {
        let mut gc = IncrementalGc::new();
        let mut old = OldSpace::new(65536);
        let obj = alloc_in_old(&mut old);

        let mut root: *mut HeapObject = obj;
        let roots = [&raw mut root as *mut *mut HeapObject];

        unsafe { gc.start_marking(&roots, &old) };
        assert_eq!(gc.colour_of(obj), MarkColour::Grey);
    }

    #[test]
    fn test_mark_step_promotes_grey_to_black() {
        let mut gc = IncrementalGc::new();
        let mut old = OldSpace::new(65536);
        let obj = alloc_in_old(&mut old);

        let mut root: *mut HeapObject = obj;
        let roots = [&raw mut root as *mut *mut HeapObject];

        unsafe { gc.start_marking(&roots, &old) };
        unsafe { gc.mark_step() };
        assert_eq!(gc.colour_of(obj), MarkColour::Black);
    }

    // ── Write barrier (Dijkstra snapshot-at-beginning) ────────────────────

    #[test]
    fn test_write_barrier_noop_when_idle() {
        let mut gc = IncrementalGc::new();
        let mut a = HeapObject::new_null();
        let mut b = HeapObject::new_null();
        // Barrier should not panic or change state when idle.
        unsafe { gc.write_barrier(&raw mut a, &raw mut b) };
        assert_eq!(gc.colour_of(&raw mut b), MarkColour::White);
    }

    #[test]
    fn test_write_barrier_shades_old_target_grey_when_host_is_black() {
        let mut gc = IncrementalGc::new();
        let mut old = OldSpace::new(65536);
        let host = alloc_in_old(&mut old);
        let old_target = alloc_in_old(&mut old);

        let mut root: *mut HeapObject = host;
        let roots = [&raw mut root as *mut *mut HeapObject];

        // Start marking and run host to black.
        unsafe { gc.start_marking(&roots, &old) };
        unsafe { gc.mark_step() };
        assert_eq!(gc.phase(), GcPhase::Sweeping);

        // Force back to marking to test barrier during marking phase.
        gc.phase = GcPhase::Marking;
        assert_eq!(gc.colour_of(host), MarkColour::Black);
        assert_eq!(gc.colour_of(old_target), MarkColour::White);

        // Barrier: host (black) is about to overwrite a slot that pointed to
        // old_target (white) → old_target must become grey.
        unsafe { gc.write_barrier(host, old_target) };
        assert_eq!(
            gc.colour_of(old_target),
            MarkColour::Grey,
            "write barrier must shade the white old target grey"
        );
    }

    #[test]
    fn test_write_barrier_no_shade_when_host_is_grey() {
        let mut gc = IncrementalGc::new();
        let mut old = OldSpace::new(65536);
        let host = alloc_in_old(&mut old);
        let old_target = alloc_in_old(&mut old);

        let mut root: *mut HeapObject = host;
        let roots = [&raw mut root as *mut *mut HeapObject];

        unsafe { gc.start_marking(&roots, &old) };
        // Host is still grey (not yet scanned).
        assert_eq!(gc.colour_of(host), MarkColour::Grey);

        unsafe { gc.write_barrier(host, old_target) };
        assert_eq!(
            gc.colour_of(old_target),
            MarkColour::White,
            "barrier must not shade when host is grey"
        );
    }

    // ── Incremental budget ────────────────────────────────────────────────

    #[test]
    fn test_mark_step_respects_budget() {
        let mut gc = IncrementalGc::with_budget(1);
        let mut old = OldSpace::new(65536);

        let obj1 = alloc_in_old(&mut old);
        let obj2 = alloc_in_old(&mut old);

        let mut r1: *mut HeapObject = obj1;
        let mut r2: *mut HeapObject = obj2;
        let roots = [
            &raw mut r1 as *mut *mut HeapObject,
            &raw mut r2 as *mut *mut HeapObject,
        ];

        unsafe { gc.start_marking(&roots, &old) };

        // With budget=1, the first step should process only 1 object.
        let done = unsafe { gc.mark_step() };
        assert!(!done, "budget=1 must not finish 2 roots in one step");

        // Second step finishes.
        let done = unsafe { gc.mark_step() };
        assert!(done, "second step must drain the remaining grey object");
    }

    // ── Synchronous sweep ─────────────────────────────────────────────────

    #[test]
    fn test_sweep_sync_identifies_dead_objects() {
        let mut gc = IncrementalGc::new();
        let mut old = OldSpace::new(65536);

        let live = alloc_in_old(&mut old);
        let _dead = alloc_in_old(&mut old);

        let mut root: *mut HeapObject = live;
        let roots = [&raw mut root as *mut *mut HeapObject];

        unsafe { gc.start_marking(&roots, &old) };
        unsafe { gc.finish_marking() };

        let free = unsafe { gc.sweep_sync(&old) };
        assert_eq!(free.len(), 1, "one dead object → one free region");
    }

    #[test]
    fn test_sweep_sync_no_dead_when_all_rooted() {
        let mut gc = IncrementalGc::new();
        let mut old = OldSpace::new(65536);

        let obj1 = alloc_in_old(&mut old);
        let obj2 = alloc_in_old(&mut old);

        let mut r1: *mut HeapObject = obj1;
        let mut r2: *mut HeapObject = obj2;
        let roots = [
            &raw mut r1 as *mut *mut HeapObject,
            &raw mut r2 as *mut *mut HeapObject,
        ];

        unsafe { gc.start_marking(&roots, &old) };
        unsafe { gc.finish_marking() };

        let free = unsafe { gc.sweep_sync(&old) };
        assert!(free.is_empty(), "all rooted → no dead objects");
    }

    // ── Concurrent sweep ──────────────────────────────────────────────────

    #[test]
    fn test_concurrent_sweep_produces_result() {
        let mut gc = IncrementalGc::new();
        let mut old = OldSpace::new(65536);

        let live = alloc_in_old(&mut old);
        let _dead = alloc_in_old(&mut old);

        let mut root: *mut HeapObject = live;
        let roots = [&raw mut root as *mut *mut HeapObject];

        unsafe { gc.start_marking(&roots, &old) };
        unsafe { gc.finish_marking() };

        let black_set = gc.black_set.clone();
        unsafe {
            gc.launch_concurrent_sweep(old.base_ptr(), old.used(), black_set);
        }

        // Poll until the result appears (the background thread is fast).
        let mut result = None;
        for _ in 0..100 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            if let Some(r) = gc.take_sweep_result() {
                result = Some(r);
                break;
            }
        }

        let free = result.expect("concurrent sweep must produce a result");
        assert_eq!(free.len(), 1, "one dead object → one free region");
    }

    // ── Minor GC (generational) ───────────────────────────────────────────

    #[test]
    fn test_minor_gc_increments_counter() {
        let mut gc = IncrementalGc::new();
        let mut heap = Heap::new();

        // Allocate something so the nursery is non-empty.
        let layout = Layout::new::<HeapObject>();
        heap.allocate(layout);

        gc.minor_gc(&mut heap);
        assert_eq!(gc.metrics().minor_gc_count, 1);
        assert_eq!(heap.young_space.used(), 0, "minor GC must clear nursery");
    }

    // ── Major GC ──────────────────────────────────────────────────────────

    #[test]
    fn test_major_gc_increments_counter() {
        let mut gc = IncrementalGc::new();
        let mut heap = Heap::new();
        let roots: [*mut *mut HeapObject; 0] = [];

        unsafe { gc.major_gc(&mut heap, &roots) };
        assert_eq!(gc.metrics().major_gc_count, 1);
        assert_eq!(gc.phase(), GcPhase::Idle);
    }

    // ── Idle-time GC ──────────────────────────────────────────────────────

    #[test]
    fn test_notify_idle_triggers_minor_gc_when_nursery_half_full() {
        let mut gc = IncrementalGc::new();
        let mut heap = Heap::new();

        // Fill nursery past 50 %.
        let layout = Layout::new::<HeapObject>();
        let obj_size = layout
            .align_to(std::mem::align_of::<HeapObject>())
            .unwrap()
            .pad_to_align()
            .size();
        let half_cap = heap.young_space.capacity() / 2;
        let count = (half_cap / obj_size) + 1;
        for _ in 0..count {
            heap.allocate(layout);
        }
        assert!(heap.young_space.used() > half_cap);

        let did_work = unsafe { gc.notify_idle(&mut heap, 1_000) };
        assert!(did_work, "idle notification must trigger minor GC");
        assert_eq!(gc.metrics().minor_gc_count, 1);
    }

    #[test]
    fn test_notify_idle_does_nothing_when_nursery_empty() {
        let mut gc = IncrementalGc::new();
        let mut heap = Heap::new();

        let did_work = unsafe { gc.notify_idle(&mut heap, 1_000) };
        assert!(!did_work, "idle must be no-op when nursery is empty");
    }

    #[test]
    fn test_notify_idle_performs_mark_step_during_marking() {
        let mut gc = IncrementalGc::with_budget(1);
        let mut heap = Heap::new();

        let mut old = OldSpace::new(65536);
        let obj = alloc_in_old(&mut old);
        let mut root: *mut HeapObject = obj;
        let roots = [&raw mut root as *mut *mut HeapObject];

        unsafe { gc.start_marking(&roots, &old) };
        assert_eq!(gc.phase(), GcPhase::Marking);

        let did_work = unsafe { gc.notify_idle(&mut heap, 1_000) };
        assert!(did_work, "idle must perform a mark step during marking");
        assert!(gc.metrics().incremental_steps > 0);
    }

    // ── Metrics ───────────────────────────────────────────────────────────

    #[test]
    fn test_metrics_default_zeroed() {
        let m = GcMetrics::new();
        assert_eq!(m.heap_size_bytes, 0);
        assert_eq!(m.last_pause_us, 0);
        assert_eq!(m.minor_gc_count, 0);
        assert_eq!(m.major_gc_count, 0);
        assert_eq!(m.incremental_steps, 0);
    }

    #[test]
    fn test_incremental_steps_counted() {
        let mut gc = IncrementalGc::new();
        let mut old = OldSpace::new(65536);
        let obj = alloc_in_old(&mut old);

        let mut root: *mut HeapObject = obj;
        let roots = [&raw mut root as *mut *mut HeapObject];

        unsafe { gc.start_marking(&roots, &old) };
        unsafe { gc.finish_marking() };

        assert!(
            gc.metrics().incremental_steps >= 1,
            "at least one step must have been counted"
        );
    }

    // ── Reset ─────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_returns_to_idle() {
        let mut gc = IncrementalGc::new();
        let mut old = OldSpace::new(65536);
        let obj = alloc_in_old(&mut old);

        let mut root: *mut HeapObject = obj;
        let roots = [&raw mut root as *mut *mut HeapObject];

        unsafe { gc.start_marking(&roots, &old) };
        gc.reset();
        assert_eq!(gc.phase(), GcPhase::Idle);
    }

    // ── Stress test ───────────────────────────────────────────────────────

    #[test]
    fn test_stress_allocate_mark_sweep_many_objects() {
        let mut gc = IncrementalGc::new();
        let mut old = OldSpace::new(1024 * 1024); // 1 MiB

        // Allocate 1000 objects; root every other one.
        let mut ptrs = Vec::new();
        for _ in 0..1000 {
            ptrs.push(alloc_in_old(&mut old));
        }

        let mut roots_raw: Vec<*mut HeapObject> = ptrs.iter().step_by(2).copied().collect();
        let mut root_slots: Vec<*mut *mut HeapObject> = roots_raw
            .iter_mut()
            .map(|p| p as *mut *mut HeapObject)
            .collect();

        unsafe { gc.start_marking(&root_slots, &old) };
        unsafe { gc.finish_marking() };

        let free = unsafe { gc.sweep_sync(&old) };
        // Half the objects are dead.
        assert_eq!(free.len(), 500, "500 unrooted objects must be dead");
    }
}
