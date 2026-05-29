//! Breakpoint-based debugger for the Stator bytecode interpreter.
//!
//! This module provides a lightweight, single-threaded debugger that can:
//!
//! - Set and remove **breakpoints** at source-level line/column positions or
//!   raw bytecode byte offsets.
//! - **Step** through execution (`step-into`, `step-over`, `step-out`).
//! - **Pause on exceptions** when a `Throw` instruction is about to fire.
//! - **Evaluate** expressions in the paused execution context by reusing the
//!   frame's shared global environment.
//! - Return all **breakpointable source locations** for a compiled function
//!   (source-map support).
//!
//! # How pausing works
//!
//! When execution is suspended, [`crate::interpreter::Interpreter::run`]
//! returns [`crate::error::StatorError::DebuggerPaused`] instead of
//! `Ok(value)`.  The interpreter frame (`InterpreterFrame`) is left in a
//! consistent state: registers, accumulator, and program counter all reflect
//! the moment just **before** the paused instruction would execute (breakpoint
//! and step pauses), or just **after** a `debugger;` statement or `Throw`.
//!
//! After inspecting state the caller should:
//!
//! 1. Decide what to do next (continue / step-into / step-over / step-out).
//! 2. Call [`Debugger::apply_action`] on the attached debugger.
//! 3. Call [`crate::interpreter::Interpreter::run`] on the same frame again.
//!
//! # Example
//!
//! ```no_run
//! use std::rc::Rc;
//! use std::cell::RefCell;
//! use stator_jse::inspector::debugger::{DebugAction, Debugger, PauseReason};
//! use stator_jse::interpreter::{attach_debugger, detach_debugger, Interpreter, InterpreterFrame};
//! use stator_jse::error::StatorError;
//! use stator_jse::bytecode::bytecode_array::BytecodeArray;
//!
//! fn run_with_debugger(bytecodes: BytecodeArray) {
//!     let dbg = Rc::new(RefCell::new(Debugger::new()));
//!     // Set a breakpoint at bytecode offset 0 (line 1, column 1).
//!     dbg.borrow_mut().set_breakpoint_at_offset(0, 1, 1);
//!
//!     attach_debugger(Rc::clone(&dbg));
//!     let mut frame = InterpreterFrame::new(Rc::new(bytecodes), vec![]);
//!
//!     // First run: pauses at the breakpoint.
//!     let r = Interpreter::run(&mut frame);
//!     assert!(matches!(r, Err(StatorError::DebuggerPaused { .. })));
//!
//!     // Inspect: query pause reason.
//!     let reason = dbg.borrow().last_pause_reason().cloned();
//!     assert_eq!(reason, Some(PauseReason::Breakpoint(1)));
//!
//!     // Resume with step-over.
//!     dbg.borrow_mut().apply_action(DebugAction::StepOver);
//!     // (call Interpreter::run again to continue …)
//!
//!     detach_debugger();
//! }
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Condvar, Mutex};

use crate::builtins::error::call_stack_depth;
use crate::bytecode::bytecode_array::BytecodeArray;
use crate::error::StatorError;
use crate::interpreter::InterpreterFrame;
use crate::objects::value::JsValue;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Numeric identifier for an installed breakpoint.
pub type BreakpointId = u32;

/// The reason why the debugger paused execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PauseReason {
    /// A `debugger;` statement was executed.
    DebuggerStatement,
    /// A user-installed breakpoint was reached.  The inner value is the
    /// [`BreakpointId`] of the matching breakpoint.
    Breakpoint(BreakpointId),
    /// Execution was paused by a step command (step-into, step-over, or
    /// step-out).
    Step,
    /// An exception was thrown and `pause_on_exceptions` is enabled.
    Exception,
}

/// A single pause event recorded by the interpreter debugger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PauseEvent {
    /// Reason for the pause.
    pub reason: PauseReason,
    /// Bytecode byte offset where the pause happened.
    pub bytecode_offset: u32,
    /// Best-effort 1-based source line for the pause.
    pub line: u32,
}

/// Snapshot of the top interpreter frame at a pause site.
#[derive(Debug, Clone, PartialEq)]
pub struct PauseFrameSnapshot {
    /// Declared parameter register count.
    pub parameter_count: u32,
    /// Local/temporary register count.
    pub frame_size: u32,
    /// Flat register snapshot: `[params..., locals...]`.
    pub registers: Vec<JsValue>,
    /// Accumulator value at the pause site.
    pub accumulator: JsValue,
}

impl PauseFrameSnapshot {
    /// Capture an interpreter frame using `accumulator`, which may be newer
    /// than `frame.accumulator` in the hot dispatch loop.
    pub fn from_frame(frame: &InterpreterFrame, accumulator: &JsValue) -> Self {
        Self {
            parameter_count: frame.bytecode_array.parameter_count(),
            frame_size: frame.bytecode_array.frame_size(),
            registers: frame.registers.iter().map(JsValue::cheap_clone).collect(),
            accumulator: accumulator.cheap_clone(),
        }
    }
}

/// Cross-thread pause/resume bridge for a debugger attached to an interpreter.
///
/// The interpreter thread publishes a [`PauseEvent`] and blocks in
/// [`DebuggerPauseBridge::publish_and_wait`] until a CDP transport thread calls
/// [`DebuggerPauseBridge::resume`].  This lets a WebSocket session deliver
/// `Debugger.paused` and `Debugger.resume` without moving the interpreter's
/// non-`Send` state across threads.
#[derive(Debug, Clone)]
pub struct DebuggerPauseBridge {
    inner: Arc<DebuggerPauseBridgeInner>,
}

#[derive(Debug)]
struct DebuggerPauseBridgeInner {
    state: Mutex<DebuggerPauseBridgeState>,
    resumed: Condvar,
}

#[derive(Debug, Default)]
struct DebuggerPauseBridgeState {
    events: VecDeque<PauseEvent>,
    paused: bool,
    resume_action: Option<DebugAction>,
    pause_requested: bool,
    disconnected: bool,
}

impl Default for DebuggerPauseBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl DebuggerPauseBridge {
    /// Create a new pause bridge with no pending pause.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(DebuggerPauseBridgeInner {
                state: Mutex::new(DebuggerPauseBridgeState::default()),
                resumed: Condvar::new(),
            }),
        }
    }

    /// Returns `true` while an interpreter thread is blocked in a pause.
    pub fn is_paused(&self) -> bool {
        self.inner
            .state
            .lock()
            .expect("debugger pause bridge mutex poisoned")
            .paused
    }

    /// Drain queued pause events in FIFO order.
    pub fn take_pause_events(&self) -> Vec<PauseEvent> {
        self.inner
            .state
            .lock()
            .expect("debugger pause bridge mutex poisoned")
            .events
            .drain(..)
            .collect()
    }

    /// Resume a currently blocked interpreter pause.
    ///
    /// Returns `false` when there is no active pause, avoiding stale resume
    /// actions being consumed by a future pause.
    pub fn resume(&self, action: DebugAction) -> bool {
        let mut state = self
            .inner
            .state
            .lock()
            .expect("debugger pause bridge mutex poisoned");
        if !state.paused {
            return false;
        }
        state.resume_action = Some(action);
        state.paused = false;
        self.inner.resumed.notify_all();
        true
    }

    /// Request that the interpreter pause at its next debugger poll.
    pub fn request_pause(&self) {
        let mut state = self
            .inner
            .state
            .lock()
            .expect("debugger pause bridge mutex poisoned");
        if !state.disconnected {
            state.pause_requested = true;
        }
    }

    fn take_pause_request(&self) -> bool {
        let mut state = self
            .inner
            .state
            .lock()
            .expect("debugger pause bridge mutex poisoned");
        if state.disconnected {
            return false;
        }
        let requested = state.pause_requested;
        state.pause_requested = false;
        requested
    }

    /// Wake any blocked interpreter and prevent future waits.
    pub fn disconnect(&self) {
        let mut state = self
            .inner
            .state
            .lock()
            .expect("debugger pause bridge mutex poisoned");
        state.disconnected = true;
        state.paused = false;
        state.pause_requested = false;
        state.resume_action = Some(DebugAction::Continue);
        state.events.clear();
        self.inner.resumed.notify_all();
    }

    fn publish_and_wait(&self, event: PauseEvent) -> DebugAction {
        let mut state = self
            .inner
            .state
            .lock()
            .expect("debugger pause bridge mutex poisoned");
        if state.disconnected {
            return DebugAction::Continue;
        }
        state.events.push_back(event);
        state.paused = true;
        state.resume_action = None;
        self.inner.resumed.notify_all();
        while state.paused && !state.disconnected {
            state = self
                .inner
                .resumed
                .wait(state)
                .expect("debugger pause bridge mutex poisoned");
        }
        state.resume_action.take().unwrap_or(DebugAction::Continue)
    }
}

/// What to do after handling a pause.
///
/// Returned by the caller after inspecting the paused state and passed to
/// [`Debugger::apply_action`] before re-entering the interpreter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugAction {
    /// Resume normal execution without pausing again (unless another
    /// breakpoint or step fires).
    Continue,
    /// Execute one statement, stepping *over* any function calls without
    /// descending into them.
    StepOver,
    /// Execute one statement, stepping *into* any function calls.
    StepInto,
    /// Run until the current function returns to its caller, then pause.
    StepOut,
}

/// A single installed breakpoint.
#[derive(Debug, Clone)]
pub struct Breakpoint {
    /// Unique identifier for this breakpoint.
    pub id: BreakpointId,
    /// Bytecode byte offset at which this breakpoint fires.
    pub bytecode_offset: u32,
    /// 1-based source line number.
    pub line: u32,
    /// 1-based source column number.
    pub column: u32,
}

/// A valid breakpoint location derived from a function's source-position
/// table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BreakpointLocation {
    /// Bytecode byte offset of this location.
    pub bytecode_offset: u32,
    /// 1-based source line number.
    pub line: u32,
    /// 1-based source column number.
    pub column: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal types
// ─────────────────────────────────────────────────────────────────────────────

/// Current stepping mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepMode {
    /// No active step — only breakpoints trigger pauses.
    None,
    /// Pause on the very next instruction (step-into semantics).
    Into,
    /// Pause when the call-stack depth returns to ≤ the saved depth
    /// (step-over semantics).
    Over { depth: usize },
    /// Pause when the call-stack depth drops *below* the saved depth
    /// (step-out semantics).
    Out { depth: usize },
}

// ─────────────────────────────────────────────────────────────────────────────
// Debugger
// ─────────────────────────────────────────────────────────────────────────────

/// A lightweight debugger for the Stator bytecode interpreter.
///
/// ## Attach / detach
///
/// Use [`crate::interpreter::attach_debugger`] and
/// [`crate::interpreter::detach_debugger`] to activate the debugger for the
/// current thread.  While attached, the interpreter checks for breakpoints
/// and step conditions before each instruction dispatch.
///
/// ## Resuming execution
///
/// After the interpreter returns
/// [`crate::error::StatorError::DebuggerPaused`], call [`Self::apply_action`]
/// with the desired [`DebugAction`] and then call
/// [`crate::interpreter::Interpreter::run`] on the same frame again.
pub struct Debugger {
    /// Active breakpoints, keyed by bytecode byte offset.
    breakpoints: HashMap<u32, Breakpoint>,
    /// Breakpoints that remove themselves after the first pause.
    one_shot_breakpoints: HashSet<BreakpointId>,
    /// Whether user-installed breakpoints are allowed to pause execution.
    breakpoints_active: bool,
    /// Monotonically increasing breakpoint ID counter.
    next_id: BreakpointId,
    /// When `true`, any `Throw` instruction causes a pause before the
    /// exception propagates.
    pub pause_on_exceptions: bool,
    /// When `true`, all pause sources are ignored until cleared.
    skip_all_pauses: bool,
    /// Current stepping state.
    step_mode: StepMode,
    /// When `true`, the next call to [`Self::check_pause_at`] is skipped
    /// once to prevent immediately re-pausing at the same offset after
    /// resuming from a breakpoint or step pause.
    skip_next: bool,
    /// When `true`, the next `Throw` instruction encountered is a
    /// *resumed* throw after a [`PauseReason::Exception`] pause.  The Throw
    /// is allowed to propagate without re-firing the exception pause.
    exception_resume: bool,
    /// Cached reason from the most recent pause.
    last_pause_reason: Option<PauseReason>,
    /// Cached bytecode offset from the most recent pause.
    last_pause_offset: u32,
    /// FIFO pause events waiting for inspector fan-out.
    pause_events: VecDeque<PauseEvent>,
    /// Optional cross-thread bridge used by transport-backed CDP sessions.
    pause_bridge: Option<DebuggerPauseBridge>,
    /// One-shot request from `Debugger.pause`.
    pause_requested: bool,
    /// Snapshot of the top frame at the most recent pause, when available.
    last_pause_frame: Option<PauseFrameSnapshot>,
}

impl Default for Debugger {
    fn default() -> Self {
        Self::new()
    }
}

impl Debugger {
    /// Create a new `Debugger` with no breakpoints, no step mode, and
    /// pause-on-exceptions disabled.
    pub fn new() -> Self {
        Self {
            breakpoints: HashMap::new(),
            one_shot_breakpoints: HashSet::new(),
            breakpoints_active: true,
            next_id: 1,
            pause_on_exceptions: false,
            skip_all_pauses: false,
            step_mode: StepMode::None,
            skip_next: false,
            exception_resume: false,
            last_pause_reason: None,
            last_pause_offset: 0,
            pause_events: VecDeque::new(),
            pause_bridge: None,
            pause_requested: false,
            last_pause_frame: None,
        }
    }

    // ── Breakpoint management ────────────────────────────────────────────────

    /// Install a breakpoint at the given raw bytecode byte `offset`.
    ///
    /// Returns the new breakpoint's [`BreakpointId`].
    pub fn set_breakpoint_at_offset(
        &mut self,
        offset: u32,
        line: u32,
        column: u32,
    ) -> BreakpointId {
        let id = self.next_id;
        self.next_id += 1;
        self.breakpoints.insert(
            offset,
            Breakpoint {
                id,
                bytecode_offset: offset,
                line,
                column,
            },
        );
        id
    }

    /// Install a breakpoint that removes itself after its first pause.
    ///
    /// Returns the new breakpoint's [`BreakpointId`].
    pub fn set_one_shot_breakpoint_at_offset(
        &mut self,
        offset: u32,
        line: u32,
        column: u32,
    ) -> BreakpointId {
        let id = self.set_breakpoint_at_offset(offset, line, column);
        self.one_shot_breakpoints.insert(id);
        id
    }

    /// Install a breakpoint at the first bytecode position that maps to
    /// `line` in the source-position table of `bytecodes`.
    ///
    /// Returns `Some(id)` on success, or `None` if the table has no entry
    /// for that line.
    pub fn set_breakpoint_at_line(
        &mut self,
        bytecodes: &BytecodeArray,
        line: u32,
    ) -> Option<BreakpointId> {
        let pos = bytecodes
            .source_positions()
            .iter()
            .find(|p| p.line == line)?;
        Some(self.set_breakpoint_at_offset(pos.bytecode_offset, pos.line, pos.column))
    }

    /// Remove the breakpoint with the given `id`.
    ///
    /// Returns `true` if the breakpoint existed and was removed, `false` if
    /// it was not found.
    pub fn remove_breakpoint(&mut self, id: BreakpointId) -> bool {
        let before = self.breakpoints.len();
        self.breakpoints.retain(|_, bp| bp.id != id);
        self.one_shot_breakpoints.remove(&id);
        self.breakpoints.len() < before
    }

    /// Return an iterator over all currently installed [`Breakpoint`]s.
    pub fn breakpoints(&self) -> impl Iterator<Item = &Breakpoint> {
        self.breakpoints.values()
    }

    /// Enable or disable user-installed breakpoints without removing them.
    pub fn set_breakpoints_active(&mut self, active: bool) {
        self.breakpoints_active = active;
    }

    /// Return whether user-installed breakpoints are currently active.
    pub fn breakpoints_active(&self) -> bool {
        self.breakpoints_active
    }

    // ── Source map ───────────────────────────────────────────────────────────

    /// Return all valid breakpoint locations in `bytecodes`, derived from its
    /// source-position table.
    ///
    /// Each [`BreakpointLocation`] can be passed directly to
    /// [`Self::set_breakpoint_at_offset`].
    pub fn breakpoint_locations(bytecodes: &BytecodeArray) -> Vec<BreakpointLocation> {
        bytecodes
            .source_positions()
            .iter()
            .map(|p| BreakpointLocation {
                bytecode_offset: p.bytecode_offset,
                line: p.line,
                column: p.column,
            })
            .collect()
    }

    // ── Pause-on-exceptions ──────────────────────────────────────────────────

    /// Enable or disable pausing when an exception is thrown.
    ///
    /// When enabled, any `Throw` instruction causes
    /// [`crate::error::StatorError::DebuggerPaused`] to be returned *before*
    /// the exception propagates.  Resuming with
    /// [`DebugAction::Continue`] allows the exception to propagate normally.
    pub fn set_pause_on_exceptions(&mut self, enable: bool) {
        self.pause_on_exceptions = enable;
    }

    /// Enable or disable the global "skip all pauses" state.
    pub fn set_skip_all_pauses(&mut self, skip: bool) {
        self.skip_all_pauses = skip;
    }

    /// Return whether every pause source is currently suppressed.
    pub fn skip_all_pauses(&self) -> bool {
        self.skip_all_pauses
    }

    // ── Last pause information ───────────────────────────────────────────────

    /// The reason for the most recent pause, or `None` if execution has not
    /// yet been paused.
    pub fn last_pause_reason(&self) -> Option<&PauseReason> {
        self.last_pause_reason.as_ref()
    }

    /// The bytecode byte offset of the most recent pause point.
    pub fn last_pause_offset(&self) -> u32 {
        self.last_pause_offset
    }

    /// The 1-based source line of the most recent pause point, or `0` if
    /// the pause did not correspond to a known breakpoint location.
    pub fn last_pause_line(&self) -> u32 {
        self.breakpoints
            .get(&self.last_pause_offset)
            .map(|bp| bp.line)
            .unwrap_or(0)
    }

    /// Drain recorded pause events in FIFO order.
    pub fn take_pause_events(&mut self) -> Vec<PauseEvent> {
        self.pause_events.drain(..).collect()
    }

    /// Snapshot captured for the most recent pause, if that pause came from a
    /// path that can observe the interpreter frame.
    pub fn last_pause_frame_snapshot(&self) -> Option<&PauseFrameSnapshot> {
        self.last_pause_frame.as_ref()
    }

    /// Attach an opt-in cross-thread pause bridge.
    ///
    /// When attached, pause sites block until the bridge receives a resume or
    /// step action.  Without a bridge, the historical in-process behaviour is
    /// preserved: pause sites return [`StatorError::DebuggerPaused`] and the
    /// embedder re-enters the interpreter manually.
    pub fn set_pause_bridge(&mut self, bridge: DebuggerPauseBridge) {
        self.pause_bridge = Some(bridge);
    }

    /// Request a pause at the next debugger poll.
    pub fn request_pause(&mut self) {
        self.pause_requested = true;
    }

    // ── Interpreter callbacks ────────────────────────────────────────────────

    /// Called by the interpreter **before** each instruction is fetched and
    /// dispatched (i.e., when `frame.pc` still points at the instruction that
    /// is *about* to execute).
    ///
    /// Returns `Some(StatorError::DebuggerPaused)` when execution should be
    /// suspended, `None` to continue normally.
    ///
    /// This method also maintains the `skip_next` one-shot flag that prevents
    /// re-pausing at the same offset immediately after resuming from a
    /// breakpoint or step pause.
    pub fn check_pause_at(&mut self, offset: u32) -> Option<StatorError> {
        let (reason, hit_breakpoint_id) = self.pause_reason_at(offset)?;
        self.finish_pause(reason, offset, hit_breakpoint_id, None)
    }

    /// Like [`Self::check_pause_at`], but captures a frame snapshot only when
    /// the offset actually pauses.
    pub fn check_pause_at_with_frame<F>(
        &mut self,
        offset: u32,
        frame_snapshot: F,
    ) -> Option<StatorError>
    where
        F: FnOnce() -> PauseFrameSnapshot,
    {
        let (reason, hit_breakpoint_id) = self.pause_reason_at(offset)?;
        self.finish_pause(reason, offset, hit_breakpoint_id, Some(frame_snapshot()))
    }

    fn pause_reason_at(&mut self, offset: u32) -> Option<(PauseReason, Option<BreakpointId>)> {
        if self.skip_all_pauses {
            return None;
        }
        // After a breakpoint or step pause, skip once to avoid re-pausing at
        // the same instruction when the interpreter is re-entered.
        if self.skip_next {
            self.skip_next = false;
            return None;
        }

        if self.pause_requested
            || self
                .pause_bridge
                .as_ref()
                .is_some_and(DebuggerPauseBridge::take_pause_request)
        {
            self.pause_requested = false;
            return Some((PauseReason::Step, None));
        }

        let hit_breakpoint_id = if self.breakpoints_active {
            self.breakpoints.get(&offset).map(|bp| bp.id)
        } else {
            None
        };
        let reason = if let Some(id) = hit_breakpoint_id {
            PauseReason::Breakpoint(id)
        } else {
            let depth = call_stack_depth();
            match self.step_mode {
                StepMode::None => return None,
                StepMode::Into => PauseReason::Step,
                StepMode::Over { depth: saved } if depth <= saved => PauseReason::Step,
                StepMode::Out { depth: saved } if depth < saved => PauseReason::Step,
                _ => return None,
            }
        };
        Some((reason, hit_breakpoint_id))
    }

    fn finish_pause(
        &mut self,
        reason: PauseReason,
        offset: u32,
        hit_breakpoint_id: Option<BreakpointId>,
        frame_snapshot: Option<PauseFrameSnapshot>,
    ) -> Option<StatorError> {
        self.last_pause_frame = frame_snapshot;
        let bridge_action = self.record_pause(reason, offset);
        if let Some(id) = hit_breakpoint_id
            && self.one_shot_breakpoints.remove(&id)
        {
            self.breakpoints.remove(&offset);
        }
        if let Some(action) = bridge_action {
            self.apply_blocking_resume_action(action);
            return None;
        }
        Some(StatorError::DebuggerPaused {
            bytecode_offset: offset,
        })
    }

    /// Called by the interpreter when a `debugger;` statement is executed.
    ///
    /// Returns `Some(StatorError::DebuggerPaused)` for the historical
    /// in-process pause path, or `None` after an attached cross-thread bridge
    /// has synchronously resumed execution.
    pub fn on_debugger_statement(&mut self, offset: u32) -> Option<StatorError> {
        self.on_debugger_statement_with_frame(offset, None)
    }

    /// Like [`Self::on_debugger_statement`], with a captured top-frame snapshot.
    pub fn on_debugger_statement_with_frame(
        &mut self,
        offset: u32,
        frame_snapshot: Option<PauseFrameSnapshot>,
    ) -> Option<StatorError> {
        self.last_pause_frame = frame_snapshot;
        if let Some(action) = self.record_pause(PauseReason::DebuggerStatement, offset) {
            self.apply_blocking_resume_action(action);
            return None;
        }
        // Debugger statements do not need skip_next: the program counter is
        // already past the statement when the pause fires.
        Some(StatorError::DebuggerPaused {
            bytecode_offset: offset,
        })
    }

    /// Called by the interpreter when a `Throw` instruction is about to
    /// execute and `pause_on_exceptions` is `true`.
    ///
    /// Returns `Some(StatorError::DebuggerPaused)` for the historical
    /// in-process pause path, or `None` after an attached cross-thread bridge
    /// has synchronously resumed execution.
    pub fn on_exception(&mut self, offset: u32) -> Option<StatorError> {
        self.on_exception_with_frame(offset, None)
    }

    /// Like [`Self::on_exception`], with a captured top-frame snapshot.
    pub fn on_exception_with_frame(
        &mut self,
        offset: u32,
        frame_snapshot: Option<PauseFrameSnapshot>,
    ) -> Option<StatorError> {
        self.last_pause_frame = frame_snapshot;
        if let Some(action) = self.record_pause(PauseReason::Exception, offset) {
            self.apply_blocking_resume_action(action);
            return None;
        }
        // skip_next is set so that the pre-fetch check skips the Throw
        // instruction when execution is resumed.  exception_resume prevents
        // the Throw handler from re-pausing on the second execution.
        self.skip_next = true;
        self.exception_resume = true;
        Some(StatorError::DebuggerPaused {
            bytecode_offset: offset,
        })
    }

    /// Query whether the current `Throw` execution is a *resume* after an
    /// exception pause (and consume the flag).
    ///
    /// When this returns `true`, the `Throw` handler should skip the
    /// pause-on-exceptions check and let the exception propagate normally.
    pub fn consume_exception_resume(&mut self) -> bool {
        let v = self.exception_resume;
        self.exception_resume = false;
        v
    }

    /// Apply a [`DebugAction`] returned by the caller after a pause.
    ///
    /// This updates the step mode and sets the `skip_next` flag when
    /// necessary (for breakpoint and step pauses where the program counter is
    /// still pointing at the paused instruction).
    pub fn apply_action(&mut self, action: DebugAction) {
        // For breakpoint and step pauses the program counter has NOT been
        // advanced past the paused instruction.  We must skip the pause check
        // for that instruction once on resume.
        if matches!(
            self.last_pause_reason,
            Some(PauseReason::Breakpoint(_) | PauseReason::Step)
        ) {
            self.skip_next = true;
        }
        // For DebuggerStatement and Exception, the PC is already past the
        // paused instruction (and on_exception already set skip_next).

        let depth = call_stack_depth();
        self.step_mode = match action {
            DebugAction::Continue => StepMode::None,
            DebugAction::StepInto => StepMode::Into,
            DebugAction::StepOver => StepMode::Over { depth },
            DebugAction::StepOut => StepMode::Out { depth },
        };
    }

    fn apply_blocking_resume_action(&mut self, action: DebugAction) {
        let depth = call_stack_depth();
        self.step_mode = match action {
            DebugAction::Continue => StepMode::None,
            DebugAction::StepInto => StepMode::Into,
            DebugAction::StepOver => StepMode::Over { depth },
            DebugAction::StepOut => StepMode::Out { depth },
        };
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Record a pause event (updates `last_pause_reason` and
    /// `last_pause_offset`).  Resets the step mode to `None` so that a fresh
    /// [`apply_action`](Self::apply_action) call is required to set the next
    /// step direction.
    fn record_pause(&mut self, reason: PauseReason, offset: u32) -> Option<DebugAction> {
        self.last_pause_reason = Some(reason);
        self.last_pause_offset = offset;
        self.step_mode = StepMode::None;
        let line = self.last_pause_line();
        let reason = self
            .last_pause_reason
            .as_ref()
            .expect("just stored pause reason")
            .clone();
        let event = PauseEvent {
            reason,
            bytecode_offset: offset,
            line,
        };
        if let Some(bridge) = &self.pause_bridge {
            return Some(bridge.publish_and_wait(event));
        }
        self.pause_events.push_back(event);
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::mpsc;
    use std::thread;
    use std::time::{Duration, Instant};

    use crate::bytecode::bytecode_array::{BytecodeArray, SourcePosition};
    use crate::bytecode::bytecodes::{Instruction, Opcode, Operand, encode};
    use crate::bytecode::feedback::FeedbackMetadata;
    use crate::error::StatorError;
    use crate::interpreter::{Interpreter, InterpreterFrame, attach_debugger, detach_debugger};
    use crate::objects::value::JsValue;

    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn wait_until_paused(bridge: &DebuggerPauseBridge) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if bridge.is_paused() {
                return;
            }
            thread::sleep(Duration::from_millis(5));
        }
        bridge.disconnect();
        panic!("debugger pause bridge did not enter paused state");
    }

    /// Build a `BytecodeArray` from a list of instructions, with source
    /// positions mapping instruction index N → line N+1.
    fn make_bytecodes_with_positions(instructions: Vec<Instruction>) -> BytecodeArray {
        let bytes = encode(&instructions);
        // Build source positions: one entry per 2-byte instruction (each
        // instruction is exactly 2 bytes: 1-byte opcode + 1-byte operand).
        // We use a simple heuristic: offset 2*N → line N+1.
        let mut positions = vec![];
        let mut offset: u32 = 0;
        for (i, instr) in instructions.iter().enumerate() {
            // Compute byte size of each instruction.
            let size = 1 + instr.operand_count() as u32; // opcode (1) + operands
            positions.push(SourcePosition::new(offset, (i + 1) as u32, 1));
            offset += size;
        }
        BytecodeArray::new(
            bytes,
            vec![],
            1, // frame_size
            0, // parameter_count
            positions,
            FeedbackMetadata::empty(),
            vec![],
        )
    }

    #[test]
    fn test_pause_bridge_blocks_until_resume() {
        let bridge = DebuggerPauseBridge::new();
        let worker_bridge = bridge.clone();
        let (tx, rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            let mut dbg = Debugger::new();
            dbg.set_pause_bridge(worker_bridge);
            dbg.set_breakpoint_at_offset(0, 7, 1);
            tx.send(()).unwrap();
            assert!(dbg.check_pause_at(0).is_none());
            dbg.last_pause_reason().cloned()
        });

        rx.recv_timeout(Duration::from_secs(1)).unwrap();
        wait_until_paused(&bridge);
        let events = bridge.take_pause_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].reason, PauseReason::Breakpoint(1));
        assert_eq!(events[0].line, 7);
        assert!(bridge.resume(DebugAction::Continue));
        assert_eq!(handle.join().unwrap(), Some(PauseReason::Breakpoint(1)));
    }

    #[test]
    fn test_pause_bridge_resume_without_pause_is_noop() {
        let bridge = DebuggerPauseBridge::new();
        assert!(!bridge.resume(DebugAction::Continue));
    }

    #[test]
    fn test_pause_bridge_disconnect_unblocks_interpreter() {
        let bridge = DebuggerPauseBridge::new();
        let worker_bridge = bridge.clone();
        let (tx, rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            let mut dbg = Debugger::new();
            dbg.set_pause_bridge(worker_bridge);
            dbg.set_breakpoint_at_offset(0, 1, 1);
            tx.send(()).unwrap();
            dbg.check_pause_at(0).is_none()
        });

        rx.recv_timeout(Duration::from_secs(1)).unwrap();
        wait_until_paused(&bridge);
        bridge.disconnect();
        assert!(handle.join().unwrap());
        assert!(!bridge.is_paused());
    }

    #[test]
    fn test_pause_request_pauses_at_next_offset() {
        let mut dbg = Debugger::new();
        dbg.request_pause();
        assert!(
            dbg.check_pause_at_with_frame(0, || PauseFrameSnapshot {
                parameter_count: 0,
                frame_size: 0,
                registers: vec![],
                accumulator: JsValue::Smi(1),
            })
            .is_some()
        );
        assert_eq!(dbg.last_pause_reason(), Some(&PauseReason::Step));
        assert_eq!(dbg.last_pause_offset(), 0);
        assert!(dbg.last_pause_frame_snapshot().is_some());
    }

    #[test]
    fn test_pause_frame_snapshot_captured_on_breakpoint_pause() {
        let ba = Rc::new(make_bytecodes_with_positions(vec![
            Instruction::new_unchecked(Opcode::LdaZero, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]));
        let mut frame = InterpreterFrame::new(Rc::clone(&ba), vec![]);
        frame.registers[0] = JsValue::Smi(41);
        let mut dbg = Debugger::new();
        dbg.set_breakpoint_at_offset(0, 1, 1);

        assert!(
            dbg.check_pause_at_with_frame(0, || {
                PauseFrameSnapshot::from_frame(&frame, &JsValue::Smi(9))
            })
            .is_some()
        );
        let snapshot = dbg.last_pause_frame_snapshot().unwrap();
        assert_eq!(snapshot.parameter_count, 0);
        assert_eq!(snapshot.frame_size, 1);
        assert_eq!(snapshot.registers[0], JsValue::Smi(41));
        assert_eq!(snapshot.accumulator, JsValue::Smi(9));
    }

    // ── Breakpoint management ─────────────────────────────────────────────────

    #[test]
    fn test_set_and_remove_breakpoint() {
        let mut dbg = Debugger::new();
        let id = dbg.set_breakpoint_at_offset(4, 1, 1);
        assert_eq!(id, 1);
        assert!(
            dbg.breakpoints()
                .any(|b| b.id == id && b.bytecode_offset == 4)
        );

        assert!(dbg.remove_breakpoint(id));
        assert!(!dbg.breakpoints().any(|b| b.id == id));

        // Removing an already-removed breakpoint returns false.
        assert!(!dbg.remove_breakpoint(id));
    }

    #[test]
    fn test_set_breakpoint_at_line() {
        let instructions = vec![
            Instruction::new_unchecked(Opcode::LdaZero, vec![]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecodes_with_positions(instructions);

        let mut dbg = Debugger::new();
        // Line 2 → second instruction (offset 1 byte for LdaZero).
        let id = dbg.set_breakpoint_at_line(&ba, 2);
        assert!(id.is_some());
    }

    #[test]
    fn test_set_breakpoint_at_line_not_found() {
        let instructions = vec![Instruction::new_unchecked(Opcode::LdaZero, vec![])];
        let ba = make_bytecodes_with_positions(instructions);

        let mut dbg = Debugger::new();
        assert!(dbg.set_breakpoint_at_line(&ba, 99).is_none());
    }

    #[test]
    fn test_breakpoints_active_suppresses_breakpoint_without_removing_it() {
        let mut dbg = Debugger::new();
        let id = dbg.set_breakpoint_at_offset(0, 1, 1);
        dbg.set_breakpoints_active(false);

        assert!(!dbg.breakpoints_active());
        assert!(dbg.check_pause_at(0).is_none());
        assert!(dbg.breakpoints().any(|bp| bp.id == id));

        dbg.set_breakpoints_active(true);
        assert!(dbg.check_pause_at(0).is_some());
        assert_eq!(dbg.last_pause_reason(), Some(&PauseReason::Breakpoint(id)));
    }

    #[test]
    fn test_one_shot_breakpoint_removes_itself_after_pause() {
        let mut dbg = Debugger::new();
        let id = dbg.set_one_shot_breakpoint_at_offset(0, 1, 1);

        assert!(dbg.check_pause_at(0).is_some());
        assert_eq!(dbg.last_pause_reason(), Some(&PauseReason::Breakpoint(id)));
        assert!(!dbg.breakpoints().any(|bp| bp.id == id));
        assert!(dbg.check_pause_at(0).is_none());
    }

    #[test]
    fn test_pause_events_drain_in_fifo_order() {
        let mut dbg = Debugger::new();
        let id = dbg.set_breakpoint_at_offset(0, 3, 1);

        assert!(dbg.check_pause_at(0).is_some());
        let _ = dbg.on_debugger_statement(9);

        let events = dbg.take_pause_events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].reason, PauseReason::Breakpoint(id));
        assert_eq!(events[0].bytecode_offset, 0);
        assert_eq!(events[0].line, 3);
        assert_eq!(events[1].reason, PauseReason::DebuggerStatement);
        assert!(dbg.take_pause_events().is_empty());
    }

    #[test]
    fn test_skip_all_pauses_suppresses_breakpoints_and_steps() {
        let mut dbg = Debugger::new();
        dbg.set_breakpoint_at_offset(0, 1, 1);
        dbg.apply_action(DebugAction::StepInto);
        dbg.set_skip_all_pauses(true);

        assert!(dbg.skip_all_pauses());
        assert!(dbg.check_pause_at(0).is_none());
        assert!(dbg.check_pause_at(1).is_none());

        dbg.set_skip_all_pauses(false);
        assert!(dbg.check_pause_at(0).is_some());
    }

    // ── Source map ────────────────────────────────────────────────────────────

    #[test]
    fn test_breakpoint_locations() {
        let instructions = vec![
            Instruction::new_unchecked(Opcode::LdaZero, vec![]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecodes_with_positions(instructions);
        let locs = Debugger::breakpoint_locations(&ba);
        assert_eq!(locs.len(), 3);
        assert_eq!(locs[0].line, 1);
        assert_eq!(locs[1].line, 2);
        assert_eq!(locs[2].line, 3);
    }

    // ── Breakpoint pause ──────────────────────────────────────────────────────

    /// Run instructions with the debugger attached and collect all
    /// `DebuggerPaused` stops.
    fn run_collecting_pauses(
        ba: BytecodeArray,
        dbg: Rc<RefCell<Debugger>>,
        max_pauses: usize,
    ) -> Vec<PauseReason> {
        let mut frame = InterpreterFrame::new(Rc::new(ba), vec![]);
        let mut reasons = vec![];
        attach_debugger(Rc::clone(&dbg));

        for _ in 0..=max_pauses {
            match Interpreter::run(&mut frame) {
                Ok(_) => break,
                Err(StatorError::DebuggerPaused { .. }) => {
                    let reason = dbg.borrow().last_pause_reason().cloned().unwrap();
                    reasons.push(reason);
                    dbg.borrow_mut().apply_action(DebugAction::Continue);
                }
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }

        detach_debugger();
        reasons
    }

    #[test]
    fn test_breakpoint_hit_once() {
        // LdaZero (offset 0) | LdaSmi(7) (offset 1) | Return (offset 3)
        let instructions = vec![
            Instruction::new_unchecked(Opcode::LdaZero, vec![]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecodes_with_positions(instructions);

        let dbg = Rc::new(RefCell::new(Debugger::new()));
        // Set breakpoint at offset 1 (the LdaSmi instruction).
        let bp_id = dbg.borrow_mut().set_breakpoint_at_offset(1, 2, 1);

        let reasons = run_collecting_pauses(ba, Rc::clone(&dbg), 5);

        assert_eq!(reasons.len(), 1);
        assert_eq!(reasons[0], PauseReason::Breakpoint(bp_id));
    }

    #[test]
    fn test_breakpoint_inspect_accumulator_before_instruction() {
        // LdaSmi(10) | LdaSmi(20) | Return
        let instructions = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(20)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecodes_with_positions(instructions);

        let dbg = Rc::new(RefCell::new(Debugger::new()));
        // Breakpoint at offset 2 (LdaSmi(20) — immediately after LdaSmi(10)).
        dbg.borrow_mut().set_breakpoint_at_offset(2, 2, 1);

        let mut frame = InterpreterFrame::new(Rc::new(ba), vec![]);
        attach_debugger(Rc::clone(&dbg));

        let r = Interpreter::run(&mut frame);
        assert!(matches!(r, Err(StatorError::DebuggerPaused { .. })));

        // LdaSmi(10) has already executed; accumulator == 10.
        assert_eq!(frame.accumulator, JsValue::Smi(10));

        dbg.borrow_mut().apply_action(DebugAction::Continue);
        let final_val = Interpreter::run(&mut frame).unwrap();
        detach_debugger();

        assert_eq!(final_val, JsValue::Smi(20));
    }

    // ── Step-into ────────────────────────────────────────────────────────────

    #[test]
    fn test_step_into_pauses_each_instruction() {
        let instructions = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecodes_with_positions(instructions);

        let dbg = Rc::new(RefCell::new(Debugger::new()));
        // Set a breakpoint at the first instruction to start stepping.
        dbg.borrow_mut().set_breakpoint_at_offset(0, 1, 1);

        let mut frame = InterpreterFrame::new(Rc::new(ba), vec![]);
        let mut pause_count = 0;
        attach_debugger(Rc::clone(&dbg));

        loop {
            match Interpreter::run(&mut frame) {
                Ok(_) => break,
                Err(StatorError::DebuggerPaused { .. }) => {
                    pause_count += 1;
                    // After the first pause (breakpoint), switch to step-into.
                    dbg.borrow_mut().apply_action(DebugAction::StepInto);
                }
                Err(e) => panic!("unexpected error: {e:?}"),
            }
            if pause_count > 10 {
                panic!("too many pauses");
            }
        }
        detach_debugger();

        // Breakpoint at 0, then step-into fires at each of the remaining 3
        // instructions before they execute (LdaSmi 2, LdaSmi 3, and Return).
        assert_eq!(pause_count, 4);
    }

    // ── Step-over ────────────────────────────────────────────────────────────

    #[test]
    fn test_step_over_pauses_at_next_instruction_in_same_frame() {
        let instructions = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecodes_with_positions(instructions);

        let dbg = Rc::new(RefCell::new(Debugger::new()));
        dbg.borrow_mut().set_breakpoint_at_offset(0, 1, 1);

        let mut frame = InterpreterFrame::new(Rc::new(ba), vec![]);
        let mut pauses = vec![];
        attach_debugger(Rc::clone(&dbg));

        loop {
            match Interpreter::run(&mut frame) {
                Ok(_) => break,
                Err(StatorError::DebuggerPaused { bytecode_offset }) => {
                    pauses.push(bytecode_offset);
                    dbg.borrow_mut().apply_action(DebugAction::StepOver);
                }
                Err(e) => panic!("{e:?}"),
            }
            if pauses.len() > 10 {
                panic!("too many pauses");
            }
        }
        detach_debugger();

        // Pause at offset 0 (breakpoint), step-over → pause at offset 2
        // (LdaSmi(2)), step-over → pause at offset 4 (Return), step-over →
        // Return executes → done.
        assert_eq!(pauses, vec![0, 2, 4]);
    }

    // ── debugger; statement ───────────────────────────────────────────────────

    #[test]
    fn test_debugger_statement_pauses_when_hook_attached() {
        let instructions = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            Instruction::new_unchecked(Opcode::Debugger, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecodes_with_positions(instructions);

        let dbg = Rc::new(RefCell::new(Debugger::new()));
        let mut frame = InterpreterFrame::new(Rc::new(ba), vec![]);
        attach_debugger(Rc::clone(&dbg));

        let r = Interpreter::run(&mut frame);
        assert!(
            matches!(r, Err(StatorError::DebuggerPaused { .. })),
            "expected debugger pause, got {r:?}"
        );
        assert_eq!(
            dbg.borrow().last_pause_reason(),
            Some(&PauseReason::DebuggerStatement)
        );
        // LdaSmi(42) ran before the debugger stmt; accumulator is 42.
        assert_eq!(frame.accumulator, JsValue::Smi(42));

        dbg.borrow_mut().apply_action(DebugAction::Continue);
        let final_val = Interpreter::run(&mut frame).unwrap();
        detach_debugger();
        assert_eq!(final_val, JsValue::Smi(42));
    }

    #[test]
    fn test_debugger_statement_noop_without_hook() {
        // Without a debugger attached, `debugger;` is a no-op.
        let instructions = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Debugger, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let bytes = encode(&instructions);
        let ba = BytecodeArray::new(
            bytes,
            vec![],
            0,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );
        let mut frame = InterpreterFrame::new(Rc::new(ba), vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    // ── pause_on_exceptions ───────────────────────────────────────────────────

    #[test]
    fn test_pause_on_exceptions_fires_before_throw() {
        // LdaSmi(1) | Throw | Return (unreachable)
        let instructions = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::Throw, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let bytes = encode(&instructions);
        let ba = BytecodeArray::new(
            bytes,
            vec![],
            1,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );

        let dbg = Rc::new(RefCell::new(Debugger::new()));
        dbg.borrow_mut().set_pause_on_exceptions(true);

        let mut frame = InterpreterFrame::new(Rc::new(ba), vec![]);
        attach_debugger(Rc::clone(&dbg));

        let r = Interpreter::run(&mut frame);
        assert!(matches!(r, Err(StatorError::DebuggerPaused { .. })));
        assert_eq!(
            dbg.borrow().last_pause_reason(),
            Some(&PauseReason::Exception)
        );
        // accumulator holds the throw value (Smi(1)).
        assert_eq!(frame.accumulator, JsValue::Smi(1));

        // Continue → the exception propagates.
        dbg.borrow_mut().apply_action(DebugAction::Continue);
        let r2 = Interpreter::run(&mut frame);
        detach_debugger();
        assert!(matches!(r2, Err(StatorError::JsException(_))));
    }

    // ── evaluate in paused context ────────────────────────────────────────────

    #[test]
    fn test_evaluate_in_paused_context() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser;

        // Build a script that pauses on `debugger;` then returns 42.
        let source = "debugger; 42;";
        let parsed = parser::parse(source).expect("parse");
        let ba = BytecodeGenerator::compile_program(&parsed).expect("compile");

        let dbg = Rc::new(RefCell::new(Debugger::new()));
        let mut frame = InterpreterFrame::new(Rc::new(ba), vec![]);
        attach_debugger(Rc::clone(&dbg));

        // Run until the debugger; statement pauses execution.
        let r = Interpreter::run(&mut frame);
        assert!(
            matches!(r, Err(StatorError::DebuggerPaused { .. })),
            "expected pause, got {r:?}"
        );
        assert_eq!(
            dbg.borrow().last_pause_reason(),
            Some(&PauseReason::DebuggerStatement)
        );

        // Evaluate "1 + 2" in the paused context by sharing the frame's
        // global environment.  Detach the debugger so the evaluation runs
        // freely without triggering extra pauses.
        detach_debugger();
        let eval_source = "1 + 2";
        let eval_parsed = parser::parse(eval_source).expect("parse eval");
        let eval_ba = BytecodeGenerator::compile_program(&eval_parsed).expect("compile eval");
        let mut eval_frame = InterpreterFrame::new_with_globals(
            Rc::new(eval_ba),
            vec![],
            Rc::clone(&frame.global_env),
        );
        let eval_result = Interpreter::run(&mut eval_frame).expect("eval");
        assert_eq!(eval_result, JsValue::Smi(3));

        // Re-attach and continue the original script to completion.
        attach_debugger(Rc::clone(&dbg));
        dbg.borrow_mut().apply_action(DebugAction::Continue);
        let final_val = Interpreter::run(&mut frame).ok();
        detach_debugger();
        // The script returns 42 after the debugger; statement.
        assert_eq!(final_val, Some(JsValue::Smi(42)));
    }

    // ── Source map: BreakpointLocation ───────────────────────────────────────

    #[test]
    fn test_breakpoint_locations_empty_source_positions() {
        let bytes = encode(&[Instruction::new_unchecked(Opcode::Return, vec![])]);
        let ba = BytecodeArray::new(
            bytes,
            vec![],
            0,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );
        let locs = Debugger::breakpoint_locations(&ba);
        assert!(locs.is_empty());
    }

    #[test]
    fn test_breakpoint_locations_match_source_positions() {
        let bytes = encode(&[
            Instruction::new_unchecked(Opcode::LdaZero, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let positions = vec![SourcePosition::new(0, 1, 1), SourcePosition::new(1, 2, 1)];
        let ba = BytecodeArray::new(
            bytes,
            vec![],
            0,
            0,
            positions,
            FeedbackMetadata::empty(),
            vec![],
        );
        let locs = Debugger::breakpoint_locations(&ba);
        assert_eq!(locs.len(), 2);
        assert_eq!(
            locs[0],
            BreakpointLocation {
                bytecode_offset: 0,
                line: 1,
                column: 1
            }
        );
        assert_eq!(
            locs[1],
            BreakpointLocation {
                bytecode_offset: 1,
                line: 2,
                column: 1
            }
        );
    }
}
