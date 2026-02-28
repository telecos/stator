/// Handle scopes and persistent roots for safe GC-pointer access.
pub mod handle;
/// Heap allocator with generational memory regions.
pub mod heap;
/// Mark-Sweep-Compact collector for the old (tenured) generation.
pub mod mark_sweep_compact;
/// Cheney semi-space scavenger (minor GC) and write-barrier remembered set.
pub mod scavenger;
/// Mark-and-trace infrastructure for garbage collection.
pub mod trace;
