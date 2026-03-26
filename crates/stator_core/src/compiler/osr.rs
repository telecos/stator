//! On-Stack Replacement (OSR): mid-execution tier-up for hot loops.
//!
//! OSR allows the engine to switch from a lower-tier execution mode
//! (interpreter or baseline JIT) to a higher-tier (Maglev, Turbofan)
//! *without* waiting for the current function invocation to return.
//!
//! # Architecture
//!
//! 1. **Loop back-edge counter** — Every loop back-edge in the interpreter
//!    or baseline JIT increments a per-function OSR counter.
//! 2. **Threshold check** — When the counter exceeds the tier-specific
//!    threshold, an OSR compilation request is enqueued.
//! 3. **Frame state capture** — The current register file (interpreter
//!    registers or baseline spill slots) is captured into an
//!    [`OsrFrameState`].
//! 4. **OSR entry** — The compiled code is patched with an entry point
//!    that restores the captured state and jumps into the loop body.
//!
//! # Tier transitions
//!
//! | From          | To        | Trigger threshold       |
//! |---------------|-----------|-------------------------|
//! | Interpreter   | Baseline  | [`OSR_INTERP_TO_BASELINE`] (10) back-edges |
//! | Baseline      | Maglev    | [`OSR_BASELINE_TO_MAGLEV`] (100) back-edges |
//! | Maglev        | Turbofan  | [`OSR_MAGLEV_TO_TURBOFAN`] (1 000) back-edges |

use std::collections::HashMap;

// ── Thresholds ──────────────────────────────────────────────────────────────

/// Loop back-edge count triggering interpreter → baseline OSR.
pub const OSR_INTERP_TO_BASELINE: u32 = 10;

/// Loop back-edge count triggering baseline → Maglev OSR.
pub const OSR_BASELINE_TO_MAGLEV: u32 = 100;

/// Loop back-edge count triggering Maglev → Turbofan OSR.
pub const OSR_MAGLEV_TO_TURBOFAN: u32 = 1_000;

// ── Execution tier ──────────────────────────────────────────────────────────

/// The JIT tier that a function is currently executing in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionTier {
    /// Interpreter (bytecode dispatch).
    Interpreter,
    /// Non-optimising baseline JIT.
    Baseline,
    /// Optimising Maglev JIT.
    Maglev,
    /// Top-tier Turbofan (Cranelift-backed) JIT.
    Turbofan,
}

impl ExecutionTier {
    /// The next higher tier, or `None` if already at the top.
    pub fn next(self) -> Option<Self> {
        match self {
            Self::Interpreter => Some(Self::Baseline),
            Self::Baseline => Some(Self::Maglev),
            Self::Maglev => Some(Self::Turbofan),
            Self::Turbofan => None,
        }
    }

    /// OSR back-edge threshold for transitioning from `self` to the next tier.
    pub fn osr_threshold(self) -> u32 {
        match self {
            Self::Interpreter => OSR_INTERP_TO_BASELINE,
            Self::Baseline => OSR_BASELINE_TO_MAGLEV,
            Self::Maglev => OSR_MAGLEV_TO_TURBOFAN,
            Self::Turbofan => u32::MAX,
        }
    }
}

// ── Frame state ─────────────────────────────────────────────────────────────

/// Captured register / spill-slot state at an OSR entry point.
///
/// When an OSR compilation completes, the runtime reconstructs this state
/// in the new frame before jumping into the loop body.
#[derive(Debug, Clone)]
pub struct OsrFrameState {
    /// Bytecode offset of the loop header where OSR entry will occur.
    pub loop_header_offset: u32,
    /// The function identifier (bytecode array index or pointer key).
    pub function_id: u64,
    /// Values of the interpreter registers (or baseline spill slots) at the
    /// OSR point, encoded as raw `i64` (tagged JsValue representation).
    pub register_values: Vec<i64>,
    /// The current accumulator value.
    pub accumulator: i64,
    /// Number of local variables in scope.
    pub local_count: u32,
    /// The execution tier being *left* (the source tier).
    pub source_tier: ExecutionTier,
}

impl OsrFrameState {
    /// Create a new frame state for an OSR entry.
    pub fn new(
        function_id: u64,
        loop_header_offset: u32,
        register_values: Vec<i64>,
        accumulator: i64,
        local_count: u32,
        source_tier: ExecutionTier,
    ) -> Self {
        Self {
            loop_header_offset,
            function_id,
            register_values,
            accumulator,
            local_count,
            source_tier,
        }
    }
}

// ── OSR request ─────────────────────────────────────────────────────────────

/// A request to compile an OSR entry point.
#[derive(Debug, Clone)]
pub struct OsrRequest {
    /// Captured frame state for the entry point.
    pub frame_state: OsrFrameState,
    /// Target tier for the compiled code.
    pub target_tier: ExecutionTier,
}

// ── Per-function OSR bookkeeping ────────────────────────────────────────────

/// Per-function loop back-edge counters and OSR state.
#[derive(Debug)]
pub struct OsrState {
    /// Back-edge counter indexed by loop-header bytecode offset.
    counters: HashMap<u32, u32>,
    /// Pending OSR request, if any.
    pending_request: Option<OsrRequest>,
    /// Current execution tier.
    pub current_tier: ExecutionTier,
}

impl OsrState {
    /// Create a new OSR state starting at the interpreter tier.
    pub fn new() -> Self {
        Self {
            counters: HashMap::new(),
            pending_request: None,
            current_tier: ExecutionTier::Interpreter,
        }
    }

    /// Increment the back-edge counter for `loop_offset` and return
    /// `Some(OsrRequest)` if the threshold is exceeded.
    ///
    /// The caller must supply the frame state captured at the back-edge.
    pub fn record_back_edge(
        &mut self,
        loop_offset: u32,
        make_frame_state: impl FnOnce() -> OsrFrameState,
    ) -> Option<OsrRequest> {
        let counter = self.counters.entry(loop_offset).or_insert(0);
        *counter += 1;

        let threshold = self.current_tier.osr_threshold();
        if *counter >= threshold
            && let Some(target) = self.current_tier.next()
        {
            let request = OsrRequest {
                frame_state: make_frame_state(),
                target_tier: target,
            };
            self.pending_request = Some(request.clone());
            return Some(request);
        }
        None
    }

    /// Take and clear the pending OSR request, if any.
    pub fn take_request(&mut self) -> Option<OsrRequest> {
        self.pending_request.take()
    }

    /// Reset the back-edge counter for a specific loop.
    pub fn reset_counter(&mut self, loop_offset: u32) {
        self.counters.remove(&loop_offset);
    }

    /// Mark that an OSR transition completed successfully.
    pub fn complete_transition(&mut self, new_tier: ExecutionTier) {
        self.current_tier = new_tier;
        self.counters.clear();
        self.pending_request = None;
    }
}

impl Default for OsrState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_transitions() {
        assert_eq!(
            ExecutionTier::Interpreter.next(),
            Some(ExecutionTier::Baseline)
        );
        assert_eq!(ExecutionTier::Baseline.next(), Some(ExecutionTier::Maglev));
        assert_eq!(ExecutionTier::Maglev.next(), Some(ExecutionTier::Turbofan));
        assert_eq!(ExecutionTier::Turbofan.next(), None);
    }

    #[test]
    fn test_osr_threshold_values() {
        assert_eq!(
            ExecutionTier::Interpreter.osr_threshold(),
            OSR_INTERP_TO_BASELINE
        );
        assert_eq!(
            ExecutionTier::Baseline.osr_threshold(),
            OSR_BASELINE_TO_MAGLEV
        );
        assert_eq!(ExecutionTier::Turbofan.osr_threshold(), u32::MAX);
    }

    #[test]
    fn test_osr_state_no_trigger_below_threshold() {
        let mut state = OsrState::new();
        for _ in 0..OSR_INTERP_TO_BASELINE - 1 {
            let req = state.record_back_edge(0, || {
                OsrFrameState::new(1, 0, vec![], 0, 0, ExecutionTier::Interpreter)
            });
            assert!(req.is_none());
        }
    }

    #[test]
    fn test_osr_state_triggers_at_threshold() {
        let mut state = OsrState::new();
        let mut req = None;
        for _ in 0..OSR_INTERP_TO_BASELINE {
            req = state.record_back_edge(0, || {
                OsrFrameState::new(1, 0, vec![42], 99, 1, ExecutionTier::Interpreter)
            });
        }
        let req = req.expect("OSR must trigger at threshold");
        assert_eq!(req.target_tier, ExecutionTier::Baseline);
        assert_eq!(req.frame_state.accumulator, 99);
        assert_eq!(req.frame_state.register_values, vec![42]);
    }

    #[test]
    fn test_osr_state_complete_transition() {
        let mut state = OsrState::new();
        // Trigger OSR.
        for _ in 0..OSR_INTERP_TO_BASELINE {
            state.record_back_edge(0, || {
                OsrFrameState::new(1, 0, vec![], 0, 0, ExecutionTier::Interpreter)
            });
        }
        state.complete_transition(ExecutionTier::Baseline);
        assert_eq!(state.current_tier, ExecutionTier::Baseline);
        assert!(state.pending_request.is_none());
    }

    #[test]
    fn test_osr_frame_state_fields() {
        let fs = OsrFrameState::new(42, 10, vec![1, 2, 3], -1, 3, ExecutionTier::Baseline);
        assert_eq!(fs.function_id, 42);
        assert_eq!(fs.loop_header_offset, 10);
        assert_eq!(fs.register_values.len(), 3);
        assert_eq!(fs.accumulator, -1);
        assert_eq!(fs.local_count, 3);
        assert_eq!(fs.source_tier, ExecutionTier::Baseline);
    }

    #[test]
    fn test_turbofan_tier_no_osr() {
        let mut state = OsrState {
            current_tier: ExecutionTier::Turbofan,
            ..OsrState::new()
        };
        for _ in 0..100 {
            let req = state.record_back_edge(0, || {
                OsrFrameState::new(1, 0, vec![], 0, 0, ExecutionTier::Turbofan)
            });
            assert!(req.is_none(), "Turbofan is top tier — no further OSR");
        }
    }
}
