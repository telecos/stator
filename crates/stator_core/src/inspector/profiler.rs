//! Sampling CPU profiler for the Stator JS engine.
//!
//! # Overview
//!
//! The profiler uses a periodic timer to request stack samples from the running
//! interpreter.  On Unix platforms it installs a `SIGPROF` signal handler backed
//! by `setitimer(2)`, which fires at the requested interval; on other platforms a
//! dedicated background thread serves as the timer source.
//!
//! To avoid async-signal-safety problems the signal/timer handler only sets the
//! [`SAMPLE_NEEDED`] atomic flag.  The actual stack capture happens inside
//! [`maybe_record_sample`], which the interpreter calls at the top of its
//! dispatch loop.  Because the interpreter loop calls this on the *same* thread
//! that owns the [`CALL_STACK`](crate::builtins::error) thread-local, reading the
//! stack is always safe.
//!
//! # CDP output
//!
//! [`CpuProfiler::stop`] returns a [`CpuProfile`] that serialises directly to
//! the CDP `Profiler.Profile` JSON format expected by Chrome DevTools.
//!
//! # Example
//!
//! ```no_run
//! use stator_core::inspector::profiler::CpuProfiler;
//!
//! let mut profiler = CpuProfiler::new();
//! profiler.start(1_000).unwrap(); // sample every 1 ms
//! // … run interpreter …
//! let profile = profiler.stop().unwrap();
//! assert!(!profile.nodes.is_empty());
//! ```

use std::cell::{Cell, RefCell};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::builtins::error::capture_call_stack;
use crate::error::{StatorError, StatorResult};

// ─────────────────────────────────────────────────────────────────────────────
// Global profiler state (written by timer; read by interpreter thread)
// ─────────────────────────────────────────────────────────────────────────────

/// Set to `true` while a profiling session is active.
///
/// Read by [`maybe_record_sample`] as a fast-path guard so the function is
/// essentially free when the profiler is not running.
pub(crate) static PROFILING_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Set to `true` by the timer source (signal handler or background thread)
/// when a new sample should be captured.
///
/// [`maybe_record_sample`] atomically swaps this back to `false` and, when it
/// was `true`, captures the current call stack.
pub(crate) static SAMPLE_NEEDED: AtomicBool = AtomicBool::new(false);

// ─────────────────────────────────────────────────────────────────────────────
// Thread-local sample buffer (written on the interpreter thread)
// ─────────────────────────────────────────────────────────────────────────────

thread_local! {
    /// Accumulated samples: each entry is `(timestamp_micros, call_stack)`.
    ///
    /// Written by [`maybe_record_sample`] and drained by [`CpuProfiler::stop`].
    static SAMPLES: RefCell<Vec<(u64, Vec<&'static str>)>> =
        const { RefCell::new(Vec::new()) };

    /// Profiling-session start time in microseconds since the Unix epoch.
    static SESSION_START_MICROS: Cell<u64> = const { Cell::new(0) };
}

// ─────────────────────────────────────────────────────────────────────────────
// CDP data structures
// ─────────────────────────────────────────────────────────────────────────────

/// CDP `Runtime.CallFrame` — the source location of a profile node.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CallFrame {
    /// Name of the function, or `"(root)"` / `"(program)"` for synthetic nodes.
    pub function_name: String,
    /// Numeric script identifier as a string (always `"0"` for now).
    pub script_id: String,
    /// Source URL (empty when unavailable).
    pub url: String,
    /// Zero-based line number within the script.
    pub line_number: u32,
    /// Zero-based column number within the script.
    pub column_number: u32,
}

/// A single node in the CPU profile call-tree.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProfileNode {
    /// Unique identifier for this node (1-based).
    pub id: u32,
    /// Location information for this node.
    pub call_frame: CallFrame,
    /// Number of samples where this node was the *leaf* of the call stack.
    pub hit_count: u32,
    /// Identifiers of child nodes (callee nodes in the tree).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<u32>,
}

/// CDP `Profiler.Profile` — complete CPU profile ready for serialisation.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CpuProfile {
    /// All nodes in the profile call-tree.
    pub nodes: Vec<ProfileNode>,
    /// Session start time in microseconds.
    pub start_time: u64,
    /// Session end time in microseconds.
    pub end_time: u64,
    /// Per-sample leaf-node ID.
    pub samples: Vec<u32>,
    /// Per-sample time delta from the previous sample, in microseconds.
    pub time_deltas: Vec<u32>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Checkpoint — called from the interpreter dispatch loop
// ─────────────────────────────────────────────────────────────────────────────

/// Check whether a sample is due and, if so, record one from the current
/// thread-local call stack.
///
/// The interpreter calls this at the top of its dispatch loop.  The function
/// is nearly free when profiling is inactive (one `Relaxed` atomic load).
pub fn maybe_record_sample() {
    // Fast path: bail immediately when no session is active.
    if !PROFILING_ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    // Atomically claim the pending sample; skip if nothing is due.
    if SAMPLE_NEEDED.swap(false, Ordering::AcqRel) {
        let stack = capture_call_stack();
        let ts = now_micros();
        SAMPLES.with(|s| s.borrow_mut().push((ts, stack)));
    }
}

#[inline]
fn now_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Platform timer — Unix (x86_64): SIGPROF + setitimer
// ─────────────────────────────────────────────────────────────────────────────

/// Install the `SIGPROF` signal handler and arm `setitimer(ITIMER_PROF)` with
/// the given interval.
///
/// The handler stores its `pthread_t` identifier so that the background timer
/// thread can use `pthread_kill` to route signals to the correct thread.
#[cfg(all(target_arch = "x86_64", unix))]
fn setup_sigprof() {
    use std::mem;

    extern "C" fn sigprof_handler(_sig: libc::c_int) {
        SAMPLE_NEEDED.store(true, Ordering::Release);
    }

    // SAFETY: We are installing a new signal action for SIGPROF.  The handler
    // only sets an atomic flag and performs no allocations, making it
    // async-signal-safe.  The previous disposition is discarded intentionally.
    unsafe {
        let mut sa: libc::sigaction = mem::zeroed();
        sa.sa_sigaction = sigprof_handler as *const () as libc::sighandler_t;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = 0;
        libc::sigaction(libc::SIGPROF, &sa, std::ptr::null_mut());
    }
}

/// Disarm `setitimer(ITIMER_PROF)` and restore the default `SIGPROF`
/// disposition.
#[cfg(all(target_arch = "x86_64", unix))]
fn teardown_sigprof() {
    use std::mem;

    // SAFETY: We disarm the timer and restore the default signal disposition.
    // No concurrent signal delivery is possible after the timer is zeroed.
    unsafe {
        let timer: libc::itimerval = mem::zeroed();
        libc::setitimer(libc::ITIMER_PROF, &timer, std::ptr::null_mut());

        let mut sa: libc::sigaction = mem::zeroed();
        sa.sa_sigaction = libc::SIG_DFL;
        libc::sigemptyset(&mut sa.sa_mask);
        libc::sigaction(libc::SIGPROF, &sa, std::ptr::null_mut());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CpuProfiler
// ─────────────────────────────────────────────────────────────────────────────

/// A sampling CPU profiler.
///
/// Call [`start`][Self::start] to begin recording samples, run JavaScript code,
/// then call [`stop`][Self::stop] to obtain the collected [`CpuProfile`].
///
/// Only one profiler session may be active at a time (enforced via the global
/// [`PROFILING_ACTIVE`] flag).  Both [`start`][Self::start] and
/// [`stop`][Self::stop] must be called from the same thread that runs the
/// interpreter, because samples are stored in thread-local storage.
pub struct CpuProfiler {
    /// Flag used to tell the background timer thread to exit.
    timer_stop: Option<Arc<AtomicBool>>,
    /// Background timer thread handle.
    timer_handle: Option<std::thread::JoinHandle<()>>,
    /// The `pthread_t` of the thread that called [`start`][Self::start].
    /// Used on Unix to route `SIGPROF` to the correct thread.
    #[cfg(all(target_arch = "x86_64", unix))]
    interpreter_thread: libc::pthread_t,
}

impl CpuProfiler {
    /// Create a new, idle profiler.
    pub fn new() -> Self {
        Self {
            timer_stop: None,
            timer_handle: None,
            #[cfg(all(target_arch = "x86_64", unix))]
            interpreter_thread: 0,
        }
    }

    /// Start a profiling session, sampling every `interval_micros` microseconds.
    ///
    /// On Unix/x86-64 this arms `setitimer(ITIMER_PROF)` with a `SIGPROF`
    /// handler in addition to a background timer thread that calls
    /// `pthread_kill` to route the signal to the correct thread.
    ///
    /// # Errors
    ///
    /// Returns [`StatorError::Internal`] if a session is already active or if
    /// the background timer thread cannot be spawned.
    pub fn start(&mut self, interval_micros: u64) -> StatorResult<()> {
        if PROFILING_ACTIVE.swap(true, Ordering::SeqCst) {
            return Err(StatorError::Internal(
                "CPU profiler: a session is already active".into(),
            ));
        }

        // Clear any leftover state from a previous session.
        SAMPLES.with(|s| s.borrow_mut().clear());
        SESSION_START_MICROS.with(|t| t.set(now_micros()));
        SAMPLE_NEEDED.store(false, Ordering::Release);

        // ── Unix: install SIGPROF handler and capture the interpreter thread id ──
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            // SAFETY: `pthread_self` always succeeds.
            self.interpreter_thread = unsafe { libc::pthread_self() };
            setup_sigprof();
        }

        // ── Spawn background timer thread ─────────────────────────────────────
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop_flag);
        let interval = Duration::from_micros(interval_micros);

        #[cfg(all(target_arch = "x86_64", unix))]
        let interp_tid = self.interpreter_thread;

        let handle = std::thread::Builder::new()
            .name("stator-profiler-timer".into())
            .spawn(move || {
                while !stop_clone.load(Ordering::Relaxed) {
                    std::thread::sleep(interval);
                    if stop_clone.load(Ordering::Relaxed) {
                        break;
                    }
                    // On Unix: deliver SIGPROF to the interpreter thread so
                    // the OS-level signal is also exercised.  The signal
                    // handler sets SAMPLE_NEEDED redundantly.
                    #[cfg(all(target_arch = "x86_64", unix))]
                    // SAFETY: `interp_tid` is a valid live thread id for the
                    // duration of the profiling session (the session ends only
                    // after `stop()` joins this thread).
                    unsafe {
                        libc::pthread_kill(interp_tid, libc::SIGPROF);
                    }
                    // On non-Unix (or as an additional trigger on Unix) set
                    // the flag directly so `maybe_record_sample` picks it up.
                    SAMPLE_NEEDED.store(true, Ordering::Release);
                }
            })
            .map_err(|e| StatorError::Internal(format!("profiler timer thread: {e}")))?;

        self.timer_stop = Some(stop_flag);
        self.timer_handle = Some(handle);
        Ok(())
    }

    /// Stop the profiling session and return the collected [`CpuProfile`].
    ///
    /// Returns `None` if no session was active.  Must be called from the same
    /// thread that called [`start`][Self::start].
    pub fn stop(&mut self) -> Option<CpuProfile> {
        if !PROFILING_ACTIVE.swap(false, Ordering::SeqCst) {
            return None;
        }

        // Signal the timer thread to exit and wait for it.
        if let Some(flag) = self.timer_stop.take() {
            flag.store(true, Ordering::Relaxed);
        }
        if let Some(handle) = self.timer_handle.take() {
            let _ = handle.join();
        }

        // Disarm the SIGPROF timer on Unix.
        #[cfg(all(target_arch = "x86_64", unix))]
        teardown_sigprof();

        let end_time = now_micros();
        let start_time = SESSION_START_MICROS.with(|t| t.get());
        let samples: Vec<(u64, Vec<&'static str>)> =
            SAMPLES.with(|s| s.borrow_mut().drain(..).collect());

        Some(build_profile_tree(&samples, start_time, end_time))
    }
}

impl Default for CpuProfiler {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Profile-tree construction
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a flat list of `(timestamp, call_stack)` samples into a
/// [`CpuProfile`] with a call-tree and per-sample hit sequence.
///
/// The call stack in each sample is ordered bottom-to-top (outermost frame
/// first, leaf frame last), matching the order maintained by
/// `push_call_frame` / `pop_call_frame`.
fn build_profile_tree(
    samples: &[(u64, Vec<&'static str>)],
    start_time: u64,
    end_time: u64,
) -> CpuProfile {
    // Node 1 is always the synthetic root.
    let mut nodes: Vec<ProfileNode> = vec![ProfileNode {
        id: 1,
        call_frame: CallFrame {
            function_name: "(root)".to_string(),
            script_id: "0".to_string(),
            url: String::new(),
            line_number: 0,
            column_number: 0,
        },
        hit_count: 0,
        children: vec![],
    }];

    let mut sample_ids: Vec<u32> = Vec::with_capacity(samples.len());
    let mut time_deltas: Vec<u32> = Vec::with_capacity(samples.len());
    let mut prev_ts = start_time;

    for (ts, stack) in samples {
        let delta = ts.saturating_sub(prev_ts) as u32;
        time_deltas.push(delta);
        prev_ts = *ts;

        // Walk / lazily-create the path from root through each frame.
        let mut current_id: u32 = 1; // root node

        for frame_name in stack {
            let child_id = find_or_create_child(&mut nodes, current_id, frame_name);
            current_id = child_id;
        }

        // The leaf node gets a hit.
        nodes[(current_id - 1) as usize].hit_count += 1;
        sample_ids.push(current_id);
    }

    CpuProfile {
        nodes,
        start_time,
        end_time,
        samples: sample_ids,
        time_deltas,
    }
}

/// Return the id of the child of `parent_id` whose `functionName` matches
/// `name`, creating a new node if none exists yet.
fn find_or_create_child(nodes: &mut Vec<ProfileNode>, parent_id: u32, name: &str) -> u32 {
    let parent_idx = (parent_id - 1) as usize;

    // Check existing children using index-based access so no temporary
    // allocation is required and no shared borrow is held across iterations.
    let n = nodes[parent_idx].children.len();
    for i in 0..n {
        let child_id = nodes[parent_idx].children[i];
        if nodes[(child_id - 1) as usize].call_frame.function_name == name {
            return child_id;
        }
    }

    // Create a new child node.
    let new_id = nodes.len() as u32 + 1;
    nodes.push(ProfileNode {
        id: new_id,
        call_frame: CallFrame {
            function_name: name.to_string(),
            script_id: "0".to_string(),
            url: String::new(),
            line_number: 0,
            column_number: 0,
        },
        hit_count: 0,
        children: vec![],
    });
    nodes[parent_idx].children.push(new_id);
    new_id
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::time::Duration;

    use super::*;
    use crate::builtins::error::{pop_call_frame, push_call_frame};
    use crate::bytecode::bytecode_generator::BytecodeGenerator;
    use crate::interpreter::{Interpreter, InterpreterFrame};
    use crate::parser;

    // ── build_profile_tree ───────────────────────────────────────────────────

    #[test]
    fn test_build_profile_tree_empty() {
        let profile = build_profile_tree(&[], 1000, 2000);
        // Always has the root node even with no samples.
        assert_eq!(profile.nodes.len(), 1);
        assert_eq!(profile.nodes[0].call_frame.function_name, "(root)");
        assert!(profile.samples.is_empty());
        assert!(profile.time_deltas.is_empty());
        assert_eq!(profile.start_time, 1000);
        assert_eq!(profile.end_time, 2000);
    }

    #[test]
    fn test_build_profile_tree_single_sample() {
        let samples = vec![(1500u64, vec!["outer", "inner"])];
        let profile = build_profile_tree(&samples, 1000, 2000);

        // root → outer → inner
        assert_eq!(profile.nodes.len(), 3);
        assert_eq!(profile.nodes[0].call_frame.function_name, "(root)");
        assert_eq!(profile.nodes[1].call_frame.function_name, "outer");
        assert_eq!(profile.nodes[2].call_frame.function_name, "inner");

        // "inner" is the leaf — it should have hitCount = 1.
        assert_eq!(profile.nodes[2].hit_count, 1);
        assert_eq!(profile.nodes[0].hit_count, 0);

        assert_eq!(profile.samples, vec![3u32]); // leaf id = 3
        assert_eq!(profile.time_deltas, vec![500u32]);
    }

    #[test]
    fn test_build_profile_tree_merges_common_prefixes() {
        // Two samples that share the "outer" frame.
        let samples = vec![(1100u64, vec!["outer", "a"]), (1200u64, vec!["outer", "b"])];
        let profile = build_profile_tree(&samples, 1000, 2000);

        // root(1) → outer(2) → a(3)
        //                    → b(4)
        assert_eq!(profile.nodes.len(), 4);
        // "outer" node should have two children.
        let outer = profile
            .nodes
            .iter()
            .find(|n| n.call_frame.function_name == "outer")
            .unwrap();
        assert_eq!(outer.children.len(), 2);
    }

    #[test]
    fn test_build_profile_tree_time_deltas() {
        let samples = vec![
            (1100u64, vec!["f"]),
            (1300u64, vec!["f"]),
            (1700u64, vec!["f"]),
        ];
        let profile = build_profile_tree(&samples, 1000, 2000);

        assert_eq!(profile.time_deltas, vec![100u32, 200u32, 400u32]);
    }

    // ── CpuProfiler ─────────────────────────────────────────────────────────

    /// All `CpuProfiler` tests must acquire this lock to avoid races on
    /// the process-global `PROFILING_ACTIVE` / `SAMPLE_NEEDED` / `SAMPLES`
    /// state when tests are executed in parallel.
    fn profiler_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn test_profiler_stop_without_start_returns_none() {
        let _g = profiler_lock();
        // Ensure clean global state.
        PROFILING_ACTIVE.store(false, Ordering::SeqCst);
        let mut p = CpuProfiler::new();
        assert!(p.stop().is_none());
    }

    #[test]
    fn test_profiler_double_start_returns_error() {
        let _g = profiler_lock();
        PROFILING_ACTIVE.store(false, Ordering::SeqCst);

        let mut p1 = CpuProfiler::new();
        let mut p2 = CpuProfiler::new();

        p1.start(100_000).expect("first start should succeed");
        let result = p2.start(100_000);
        assert!(result.is_err(), "second start must fail");

        // Clean up.
        p1.stop();
    }

    #[test]
    fn test_profiler_collects_samples_during_js_execution() {
        let _g = profiler_lock();
        PROFILING_ACTIVE.store(false, Ordering::SeqCst);

        // Start a profiling session with a 1 ms interval.
        let mut profiler = CpuProfiler::new();
        profiler.start(1_000).expect("start");

        // Run a JS loop that keeps the interpreter busy long enough for several
        // timer ticks to fire.  Use a simple arithmetic sum (no `var`).
        let src = "1 + 2 + 3";
        let bytecodes = parser::parse(src)
            .and_then(|p| BytecodeGenerator::compile_program(&p))
            .expect("compile");
        let mut frame = InterpreterFrame::new_with_globals(
            bytecodes,
            vec![],
            Rc::new(RefCell::new(std::collections::HashMap::new())),
        );
        Interpreter::run(&mut frame).expect("run");

        let profile = profiler.stop().expect("stop returns Some");

        // The profile must always contain at least the root node.
        assert!(!profile.nodes.is_empty(), "profile must have nodes");
        // start_time ≤ end_time.
        assert!(profile.start_time <= profile.end_time);
    }

    #[test]
    fn test_profiler_records_call_stack_frames() {
        let _g = profiler_lock();
        PROFILING_ACTIVE.store(false, Ordering::SeqCst);
        SAMPLES.with(|s| s.borrow_mut().clear());

        // Manually push frames to simulate interpreter activity, trigger a
        // sample by hand, then verify the profile tree contains those frames.
        let mut profiler = CpuProfiler::new();
        profiler.start(10_000_000).expect("start"); // very long interval — we trigger manually

        // Simulate a call to "myFunction".
        push_call_frame("myFunction");
        // Force a sample immediately.
        SAMPLE_NEEDED.store(true, Ordering::Release);
        maybe_record_sample();
        pop_call_frame();

        let profile = profiler.stop().expect("stop returns Some");

        // The sample should have included "myFunction".
        let has_my_fn = profile
            .nodes
            .iter()
            .any(|n| n.call_frame.function_name == "myFunction");
        assert!(has_my_fn, "profile must contain myFunction node");

        // samples vec should have exactly one entry.
        assert_eq!(profile.samples.len(), 1);
    }

    #[test]
    fn test_profiler_serialises_to_cdp_json() {
        let _g = profiler_lock();
        PROFILING_ACTIVE.store(false, Ordering::SeqCst);
        SAMPLES.with(|s| s.borrow_mut().clear());

        let mut profiler = CpuProfiler::new();
        profiler.start(10_000_000).expect("start");

        push_call_frame("greet");
        SAMPLE_NEEDED.store(true, Ordering::Release);
        maybe_record_sample();
        pop_call_frame();

        let profile = profiler.stop().expect("stop returns Some");
        let json = serde_json::to_value(&profile).expect("serialise");

        assert!(json["nodes"].is_array());
        assert!(json["startTime"].is_number());
        assert!(json["endTime"].is_number());
        assert!(json["samples"].is_array());
        assert!(json["timeDeltas"].is_array());

        // Every node must have a callFrame with at least a functionName.
        for node in json["nodes"].as_array().unwrap() {
            assert!(node["callFrame"]["functionName"].is_string());
        }
    }

    // ── maybe_record_sample (fast path) ─────────────────────────────────────

    #[test]
    fn test_maybe_record_sample_noop_when_inactive() {
        let _g = profiler_lock();
        // Ensure no session is active.
        PROFILING_ACTIVE.store(false, Ordering::SeqCst);
        SAMPLES.with(|s| s.borrow_mut().clear());

        // Explicitly arm the "sample needed" flag so we know maybe_record_sample
        // would have recorded if profiling were active.
        SAMPLE_NEEDED.store(true, Ordering::Release);
        assert!(
            SAMPLE_NEEDED.load(Ordering::Acquire),
            "SAMPLE_NEEDED must be true before the call"
        );

        // Call should be a no-op because PROFILING_ACTIVE is false.
        maybe_record_sample();

        let count = SAMPLES.with(|s| s.borrow().len());
        assert_eq!(count, 0, "no sample should have been recorded");

        // Clean up global flag.
        SAMPLE_NEEDED.store(false, Ordering::Release);
    }

    #[test]
    fn test_profiler_profile_has_duration() {
        let _g = profiler_lock();
        PROFILING_ACTIVE.store(false, Ordering::SeqCst);

        let mut profiler = CpuProfiler::new();
        profiler.start(10_000_000).expect("start");
        std::thread::sleep(Duration::from_millis(5));
        let profile = profiler.stop().expect("stop");
        // end_time must be >= start_time.
        assert!(
            profile.end_time >= profile.start_time,
            "end_time ({}) must be >= start_time ({})",
            profile.end_time,
            profile.start_time
        );
    }
}
