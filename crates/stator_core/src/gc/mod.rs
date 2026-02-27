/// Handle scopes and persistent roots for safe GC-pointer access.
pub mod handle;
/// Heap allocator with generational memory regions.
pub mod heap;
/// Mark-and-trace infrastructure for garbage collection.
pub mod trace;
