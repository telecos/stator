/// GC-managed typed pointer ([`gc_ptr::GcPtr<T>`]) and the [`gc_ptr::GcObject`] trait.
pub mod gc_ptr;
/// Handle scopes and persistent roots for safe GC-pointer access.
pub mod handle;
/// Heap allocator with generational memory regions.
pub mod heap;
/// Incremental/concurrent GC: budget-based marking, concurrent sweeping,
/// idle-time collection, and GC metrics for sub-1 ms pauses.
pub mod incremental;
/// Mark-Sweep-Compact collector for the old (tenured) generation.
pub mod mark_sweep_compact;
/// Cheney semi-space scavenger (minor GC) and write-barrier remembered set.
pub mod scavenger;
/// Thread-local allocation buffers (TLABs) for fast bump-pointer allocation.
pub mod tlab;
/// Mark-and-trace infrastructure for garbage collection.
pub mod trace;
/// Write barrier for tracking old-generation → young-generation pointer edges.
pub mod write_barrier;
