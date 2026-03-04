//! Chrome DevTools Protocol (CDP) inspector interface.
//!
//! This module contains [`cdp`], a minimal WebSocket server that implements
//! enough of the Chrome DevTools Protocol to support `Runtime.evaluate`,
//! `Debugger.enable`, `Profiler.enable`, `Profiler.start`, `Profiler.stop`,
//! `HeapProfiler.enable`, `HeapProfiler.takeHeapSnapshot`,
//! `HeapProfiler.startTrackingHeapObjects`, and
//! `HeapProfiler.stopTrackingHeapObjects`.

/// Breakpoint-based debugger with step-into/over/out, pause-on-exceptions,
/// in-context evaluation, and source-map support.
pub mod debugger;

/// CDP WebSocket server, JSON-RPC message parsing, and domain routing.
pub mod cdp;
/// Heap snapshot builder: walks the JS value graph and emits a
/// CDP-compatible `HeapProfiler.HeapSnapshot` payload.  Also provides
/// allocation tracking via [`heap_snapshot::record_allocation`].
pub mod heap_snapshot;
/// Sampling CPU profiler: SIGPROF/setitimer-based sample collection and CDP
/// `Profiler.Profile` emission.
pub mod profiler;
