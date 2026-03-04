//! Chrome DevTools Protocol (CDP) inspector interface.
//!
//! This module contains [`cdp`], a minimal WebSocket server that implements
//! enough of the Chrome DevTools Protocol to support `Runtime.evaluate`,
//! `Debugger.enable`, `Profiler.enable`, `Profiler.start`, `Profiler.stop`,
//! and `HeapProfiler.enable`.

/// CDP WebSocket server, JSON-RPC message parsing, and domain routing.
pub mod cdp;
/// Sampling CPU profiler: SIGPROF/setitimer-based sample collection and CDP
/// `Profiler.Profile` emission.
pub mod profiler;
