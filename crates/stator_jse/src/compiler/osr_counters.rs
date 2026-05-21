//! Release-safe On-Stack Replacement (OSR) diagnostic counters.
//!
//! Stator's current loop tiering path can request/compile higher tiers at hot
//! back-edges, but it does **not** have a safe true mid-frame OSR entry/exit
//! path: the active interpreter dispatch loops deliberately avoid jumping into
//! JIT code mid-loop because re-running from function entry with stale frame
//! state is not real OSR and has caused correctness issues.  This module
//! therefore exposes a stable, zeroed diagnostics schema for Edge proof runs
//! without fabricating OSR events.
//!
//! Future true OSR machinery should record only real mid-execution handoffs via
//! [`record_entry_attempt`], [`record_entry_success`], [`record_entry_failure`],
//! and [`record_exit`].  Ordinary force-tier, background compile, or next-call
//! tier-up paths must not increment these counters.

use std::sync::atomic::{AtomicU64, Ordering};

/// Whether this build has a true, wired mid-frame OSR entry/exit path.
pub const TRUE_OSR_SUPPORTED: bool = false;

/// Number of per-script OSR rows carried by [`OsrCountersSnapshot`].
///
/// A value of zero means this build exposes aggregate process counters only.
pub const PER_SCRIPT_ROW_COUNT: u32 = 0;

/// Execution tiers used by OSR diagnostics.
///
/// The numeric discriminants double as stable ABI indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum OsrTier {
    /// Bytecode interpreter.
    Interpreter = 0,
    /// Non-optimising baseline JIT.
    Baseline = 1,
    /// Maglev mid-tier optimising compiler.
    Maglev = 2,
    /// Cranelift-backed Turbofan top tier.
    Turbofan = 3,
}

impl OsrTier {
    /// Number of tier variants tracked by OSR diagnostics.
    pub const COUNT: usize = 4;

    /// Stable short name suitable for diagnostic dumps and FFI labels.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Interpreter => "interpreter",
            Self::Baseline => "baseline",
            Self::Maglev => "maglev",
            Self::Turbofan => "turbofan",
        }
    }

    /// All tier variants in stable discriminant order.
    #[must_use]
    pub const fn all() -> [Self; Self::COUNT] {
        [
            Self::Interpreter,
            Self::Baseline,
            Self::Maglev,
            Self::Turbofan,
        ]
    }
}

/// Stable OSR exit reasons.
///
/// These rows are zero while [`TRUE_OSR_SUPPORTED`] is false.  Future true OSR
/// exits must append new reasons rather than reordering existing ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum OsrExitReason {
    /// OSR-entered code returned normally.
    NormalReturn = 0,
    /// OSR-entered code exited through a deopt or bailout path.
    Deopt = 1,
    /// OSR-entered code exited by throwing an exception.
    Exception = 2,
    /// OSR-entered code exited due to embedder termination/interrupt polling.
    TerminationInterrupt = 3,
}

impl OsrExitReason {
    /// Number of exit-reason variants tracked by OSR diagnostics.
    pub const COUNT: usize = 4;

    /// Stable short name suitable for diagnostic dumps and FFI labels.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::NormalReturn => "normal_return",
            Self::Deopt => "deopt",
            Self::Exception => "exception",
            Self::TerminationInterrupt => "termination_interrupt",
        }
    }

    /// All exit reasons in stable discriminant order.
    #[must_use]
    pub const fn all() -> [Self; Self::COUNT] {
        [
            Self::NormalReturn,
            Self::Deopt,
            Self::Exception,
            Self::TerminationInterrupt,
        ]
    }
}

struct EntrySlot {
    attempts: AtomicU64,
    successes: AtomicU64,
    failures: AtomicU64,
}

impl EntrySlot {
    const fn new() -> Self {
        Self {
            attempts: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            failures: AtomicU64::new(0),
        }
    }

    fn reset(&self) {
        self.attempts.store(0, Ordering::Relaxed);
        self.successes.store(0, Ordering::Relaxed);
        self.failures.store(0, Ordering::Relaxed);
    }

    fn snapshot(&self, source_tier: OsrTier, target_tier: OsrTier) -> OsrEntryTransitionSnapshot {
        OsrEntryTransitionSnapshot {
            source_tier,
            target_tier,
            attempts: self.attempts.load(Ordering::Relaxed),
            successes: self.successes.load(Ordering::Relaxed),
            failures: self.failures.load(Ordering::Relaxed),
        }
    }
}

static ENTRY_COUNTERS: [[EntrySlot; OsrTier::COUNT]; OsrTier::COUNT] =
    [const { [const { EntrySlot::new() }; OsrTier::COUNT] }; OsrTier::COUNT];

static EXIT_COUNTERS: [[AtomicU64; OsrExitReason::COUNT]; OsrTier::COUNT] =
    [const { [const { AtomicU64::new(0) }; OsrExitReason::COUNT] }; OsrTier::COUNT];

/// Record that true OSR entry was attempted from `source_tier` to `target_tier`.
///
/// Do not call this for hot-loop compile requests, force-tier APIs, or next-call
/// tier-up: those are not mid-frame OSR entries.
#[inline]
pub fn record_entry_attempt(source_tier: OsrTier, target_tier: OsrTier) {
    ENTRY_COUNTERS[source_tier as usize][target_tier as usize]
        .attempts
        .fetch_add(1, Ordering::Relaxed);
}

/// Record that a previously-attempted true OSR entry completed successfully.
#[inline]
pub fn record_entry_success(source_tier: OsrTier, target_tier: OsrTier) {
    ENTRY_COUNTERS[source_tier as usize][target_tier as usize]
        .successes
        .fetch_add(1, Ordering::Relaxed);
}

/// Record that a true OSR entry attempt failed before control transferred.
#[inline]
pub fn record_entry_failure(source_tier: OsrTier, target_tier: OsrTier) {
    ENTRY_COUNTERS[source_tier as usize][target_tier as usize]
        .failures
        .fetch_add(1, Ordering::Relaxed);
}

/// Record an exit from code that was entered through true OSR.
#[inline]
pub fn record_exit(tier: OsrTier, reason: OsrExitReason) {
    EXIT_COUNTERS[tier as usize][reason as usize].fetch_add(1, Ordering::Relaxed);
}

/// Per-source/target OSR entry counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OsrEntryTransitionSnapshot {
    /// Tier executing before the OSR handoff.
    pub source_tier: OsrTier,
    /// Tier targeted by the OSR handoff.
    pub target_tier: OsrTier,
    /// Real mid-frame OSR entry attempts.
    pub attempts: u64,
    /// Attempts that transferred control to the target tier.
    pub successes: u64,
    /// Attempts that failed before control transferred.
    pub failures: u64,
}

impl OsrEntryTransitionSnapshot {
    /// Sum of all entry counters in this source/target cell.
    #[must_use]
    pub const fn total(&self) -> u64 {
        self.attempts + self.successes + self.failures
    }
}

/// Per-exit-reason counters for one tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OsrExitTierSnapshot {
    /// Tier that had been entered through OSR.
    pub tier: OsrTier,
    /// One counter per [`OsrExitReason`] in stable discriminant order.
    pub reasons: [u64; OsrExitReason::COUNT],
}

impl OsrExitTierSnapshot {
    /// Look up an exit-reason counter.
    #[must_use]
    pub fn count(&self, reason: OsrExitReason) -> u64 {
        self.reasons[reason as usize]
    }

    /// Sum of all exit counters for this tier.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.reasons.iter().sum()
    }
}

/// Immutable snapshot of all process-global OSR counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OsrCountersSnapshot {
    /// Whether this build has real mid-frame OSR instrumentation wired in.
    pub true_osr_supported: bool,
    /// Number of per-script rows exposed by this snapshot.  Currently zero.
    pub per_script_row_count: u32,
    /// Entry counters indexed as `[source_tier][target_tier]`.
    pub entries: [[OsrEntryTransitionSnapshot; OsrTier::COUNT]; OsrTier::COUNT],
    /// Exit counters indexed by tier.
    pub exits: [OsrExitTierSnapshot; OsrTier::COUNT],
}

impl OsrCountersSnapshot {
    /// Return entry counters for a source/target tier pair.
    #[must_use]
    pub fn entry(&self, source_tier: OsrTier, target_tier: OsrTier) -> &OsrEntryTransitionSnapshot {
        &self.entries[source_tier as usize][target_tier as usize]
    }

    /// Return exit counters for a tier.
    #[must_use]
    pub fn exits_for_tier(&self, tier: OsrTier) -> &OsrExitTierSnapshot {
        &self.exits[tier as usize]
    }

    /// Sum of all entry counters.
    #[must_use]
    pub fn total_entries(&self) -> u64 {
        self.entries
            .iter()
            .flatten()
            .map(OsrEntryTransitionSnapshot::total)
            .sum()
    }

    /// Sum of all exit counters.
    #[must_use]
    pub fn total_exits(&self) -> u64 {
        self.exits.iter().map(OsrExitTierSnapshot::total).sum()
    }

    /// Sum of every OSR counter.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.total_entries() + self.total_exits()
    }
}

/// Take a snapshot of every OSR diagnostic counter.
#[must_use]
pub fn snapshot() -> OsrCountersSnapshot {
    let entries = std::array::from_fn(|source_idx| {
        std::array::from_fn(|target_idx| {
            ENTRY_COUNTERS[source_idx][target_idx]
                .snapshot(OsrTier::all()[source_idx], OsrTier::all()[target_idx])
        })
    });
    let exits = std::array::from_fn(|tier_idx| {
        let reasons = std::array::from_fn(|reason_idx| {
            EXIT_COUNTERS[tier_idx][reason_idx].load(Ordering::Relaxed)
        });
        OsrExitTierSnapshot {
            tier: OsrTier::all()[tier_idx],
            reasons,
        }
    });
    OsrCountersSnapshot {
        true_osr_supported: TRUE_OSR_SUPPORTED,
        per_script_row_count: PER_SCRIPT_ROW_COUNT,
        entries,
        exits,
    }
}

/// Reset every OSR diagnostic counter to zero.
pub fn reset() {
    for source in &ENTRY_COUNTERS {
        for target in source {
            target.reset();
        }
    }
    for tier in &EXIT_COUNTERS {
        for reason in tier {
            reason.store(0, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::osr::{ExecutionTier, OSR_INTERP_TO_BASELINE, OsrFrameState, OsrState};
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        match TEST_LOCK.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    #[test]
    fn test_osr_counter_names_and_discriminants_are_stable() {
        assert_eq!(OsrTier::Interpreter as usize, 0);
        assert_eq!(OsrTier::Baseline as usize, 1);
        assert_eq!(OsrTier::Maglev as usize, 2);
        assert_eq!(OsrTier::Turbofan as usize, 3);
        assert_eq!(OsrTier::Interpreter.name(), "interpreter");
        assert_eq!(OsrExitReason::NormalReturn as usize, 0);
        assert_eq!(OsrExitReason::TerminationInterrupt as usize, 3);
        assert_eq!(OsrExitReason::Deopt.name(), "deopt");
    }

    #[test]
    fn test_osr_counters_reset_snapshot_and_no_drift() {
        let _g = lock();
        reset();
        let before = snapshot();
        assert!(!before.true_osr_supported);
        assert_eq!(before.per_script_row_count, 0);
        assert_eq!(before.total(), 0);

        record_entry_attempt(OsrTier::Interpreter, OsrTier::Baseline);
        record_entry_success(OsrTier::Interpreter, OsrTier::Baseline);
        record_entry_failure(OsrTier::Baseline, OsrTier::Maglev);
        record_exit(OsrTier::Baseline, OsrExitReason::NormalReturn);

        let mid = snapshot();
        assert_eq!(
            mid.entry(OsrTier::Interpreter, OsrTier::Baseline).attempts,
            1
        );
        assert_eq!(
            mid.entry(OsrTier::Interpreter, OsrTier::Baseline).successes,
            1
        );
        assert_eq!(mid.entry(OsrTier::Baseline, OsrTier::Maglev).failures, 1);
        assert_eq!(
            mid.exits_for_tier(OsrTier::Baseline)
                .count(OsrExitReason::NormalReturn),
            1
        );
        assert_eq!(snapshot(), mid, "snapshot must not mutate counters");

        reset();
        let after = snapshot();
        assert_eq!(after.total(), 0);
        assert_eq!(snapshot(), after, "reset must not drift");
    }

    #[test]
    fn test_osr_state_bookkeeping_does_not_fabricate_counters() {
        let _g = lock();
        reset();
        let mut state = OsrState::new();
        for _ in 0..OSR_INTERP_TO_BASELINE {
            let _ = state.record_back_edge(0, || {
                OsrFrameState::new(7, 0, vec![1], 2, 1, ExecutionTier::Interpreter)
            });
        }
        state.complete_transition(ExecutionTier::Baseline);
        let snap = snapshot();
        assert!(!snap.true_osr_supported);
        assert_eq!(
            snap.total(),
            0,
            "compile-request OSR scaffolding is not true OSR"
        );
    }
}
