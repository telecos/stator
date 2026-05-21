//! Release-safe inline-cache (IC) probe/hit/miss/transition counters
//! broken down by execution tier and IC operation.
//!
//! These counters are intentionally always-on in release builds so Edge
//! performance proof runs can compare IC stability and hot-path behaviour
//! against V8 without relying on debug-only tracing.  Each event is a
//! single relaxed atomic increment.
//!
//! # Schema
//!
//! Counters are organised as a 3-D table:
//!
//! ```text
//! events[tier][op][event]
//! ```
//!
//! - **Tier** — [`IcTier`]: `Interpreter`, `Baseline`, `Maglev`, `Turbofan`.
//! - **Operation** — [`IcOp`]: `NamedLoad`, `NamedStore`, `IndexedLoad`,
//!   `IndexedStore`, `Call`.
//! - **Event** — [`IcEvent`]: `Probe`, `Hit`, `Miss`, `Transition`.
//!
//! Discriminants are stable and double as indices into the underlying
//! atomic array — adding new variants must always append at the end to
//! preserve ABI/FFI compatibility.
//!
//! # Tier limitations
//!
//! Only the **Interpreter** tier currently records events.  The baseline,
//! Maglev, and Turbofan tiers either inline IC handling into JIT-emitted
//! code or are not currently executing (see `docs/edge_diagnostics.md`).
//! Their rows are exposed but remain zero.  Consumers must not interpret
//! zero baseline/Maglev/Turbofan counts as proof of IC stability for those
//! tiers.  No tier may synthesise fake increments.

use std::sync::atomic::{AtomicU64, Ordering};

/// Execution tier an IC event was recorded from.
///
/// Discriminants are stable and double as indices.  Appending new variants
/// is backwards-compatible; reordering or removing variants is a breaking
/// ABI change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum IcTier {
    /// Bytecode interpreter.
    Interpreter = 0,
    /// Non-optimising baseline JIT.
    Baseline = 1,
    /// Maglev mid-tier optimising compiler.
    Maglev = 2,
    /// Cranelift-backed Turbofan optimising compiler.
    Turbofan = 3,
}

impl IcTier {
    /// Number of tier variants tracked by the IC counters.
    pub const COUNT: usize = 4;

    /// Stable short name used in diagnostic dumps and FFI labels.
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

/// IC operation kind the event was recorded against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum IcOp {
    /// Named property load (`obj.x` / `LdaNamedProperty`).
    NamedLoad = 0,
    /// Named property store (`obj.x = v` / `StaNamedProperty`).
    NamedStore = 1,
    /// Indexed/keyed property load (`obj[i]` / `LdaKeyedProperty`).
    IndexedLoad = 2,
    /// Indexed/keyed property store (`obj[i] = v` / `StaKeyedProperty`).
    IndexedStore = 3,
    /// Call site (`CallProperty*`, `CallUndefinedReceiver*`, …).
    Call = 4,
}

impl IcOp {
    /// Number of IC operation variants tracked.
    pub const COUNT: usize = 5;

    /// Stable short name used in diagnostic dumps and FFI labels.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::NamedLoad => "named_load",
            Self::NamedStore => "named_store",
            Self::IndexedLoad => "indexed_load",
            Self::IndexedStore => "indexed_store",
            Self::Call => "call",
        }
    }

    /// All operation variants in stable discriminant order.
    #[must_use]
    pub const fn all() -> [Self; Self::COUNT] {
        [
            Self::NamedLoad,
            Self::NamedStore,
            Self::IndexedLoad,
            Self::IndexedStore,
            Self::Call,
        ]
    }
}

/// IC event kind being recorded.
///
/// * `Probe` is incremented once at the entry of every IC-bearing operation,
///   whether or not a fast-path entry is present.  `Hit + Miss` therefore
///   equals `Probe` for any non-aborting access pattern.
/// * `Hit` records that a cached entry serviced the access without falling
///   through to the runtime path.
/// * `Miss` records that the cache could not serve the access and the
///   runtime path executed.
/// * `Transition` records that the IC state machine for this operation
///   advanced (e.g. a new shape was added, or the feedback slot moved
///   between Uninitialized/Monomorphic/Polymorphic/Megamorphic).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum IcEvent {
    /// IC entry was probed (a lookup was attempted on the cache).
    Probe = 0,
    /// IC fast path serviced the access.
    Hit = 1,
    /// IC missed and the runtime path was taken.
    Miss = 2,
    /// IC state machine advanced (cache populated or state changed).
    Transition = 3,
}

impl IcEvent {
    /// Number of event variants tracked.
    pub const COUNT: usize = 4;

    /// Stable short name used in diagnostic dumps and FFI labels.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Probe => "probe",
            Self::Hit => "hit",
            Self::Miss => "miss",
            Self::Transition => "transition",
        }
    }

    /// All event variants in stable discriminant order.
    #[must_use]
    pub const fn all() -> [Self; Self::COUNT] {
        [Self::Probe, Self::Hit, Self::Miss, Self::Transition]
    }
}

// Process-global counter table.  Indexed as `COUNTERS[tier][op][event]`.
#[allow(clippy::declare_interior_mutable_const)]
static COUNTERS: [[[AtomicU64; IcEvent::COUNT]; IcOp::COUNT]; IcTier::COUNT] =
    [const { [const { [const { AtomicU64::new(0) }; IcEvent::COUNT] }; IcOp::COUNT] };
        IcTier::COUNT];

/// Record an IC event for `(tier, op, event)`.
#[inline]
pub fn record(tier: IcTier, op: IcOp, event: IcEvent) {
    COUNTERS[tier as usize][op as usize][event as usize].fetch_add(1, Ordering::Relaxed);
}

/// Record an IC probe for `(tier, op)`.  Convenience wrapper.
#[inline]
pub fn record_probe(tier: IcTier, op: IcOp) {
    record(tier, op, IcEvent::Probe);
}

/// Record an IC hit for `(tier, op)`.  Convenience wrapper.
#[inline]
pub fn record_hit(tier: IcTier, op: IcOp) {
    record(tier, op, IcEvent::Hit);
}

/// Record an IC miss for `(tier, op)`.  Convenience wrapper.
#[inline]
pub fn record_miss(tier: IcTier, op: IcOp) {
    record(tier, op, IcEvent::Miss);
}

/// Record an IC state transition for `(tier, op)`.  Convenience wrapper.
#[inline]
pub fn record_transition(tier: IcTier, op: IcOp) {
    record(tier, op, IcEvent::Transition);
}

/// Per-operation IC counters for a single tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IcOpSnapshot {
    /// The IC operation these counters describe.
    pub op: IcOp,
    /// One counter per [`IcEvent`] in stable discriminant order.
    pub events: [u64; IcEvent::COUNT],
}

impl IcOpSnapshot {
    /// Return the counter for `event`.
    #[must_use]
    pub fn count(&self, event: IcEvent) -> u64 {
        self.events[event as usize]
    }

    /// Sum of all event counters for this operation.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.events.iter().sum()
    }
}

/// Per-tier IC counters across every [`IcOp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IcTierSnapshot {
    /// The execution tier these counters describe.
    pub tier: IcTier,
    /// One snapshot per [`IcOp`] in stable discriminant order.
    pub ops: [IcOpSnapshot; IcOp::COUNT],
}

impl IcTierSnapshot {
    /// Return the per-event snapshot for `op`.
    #[must_use]
    pub fn for_op(&self, op: IcOp) -> &IcOpSnapshot {
        &self.ops[op as usize]
    }

    /// Sum of every event counter across every operation for this tier.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.ops.iter().map(IcOpSnapshot::total).sum()
    }
}

/// Immutable snapshot of every IC counter at a point in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IcCountersSnapshot {
    /// One snapshot per [`IcTier`] in stable discriminant order.
    pub per_tier: [IcTierSnapshot; IcTier::COUNT],
}

impl IcCountersSnapshot {
    /// Return the snapshot for `tier`.
    #[must_use]
    pub fn for_tier(&self, tier: IcTier) -> &IcTierSnapshot {
        &self.per_tier[tier as usize]
    }

    /// Sum of every event counter across every tier and operation.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.per_tier.iter().map(IcTierSnapshot::total).sum()
    }
}

/// Take a consistent-per-counter snapshot of every IC counter.
///
/// Reads are relaxed atomic loads per cell; the snapshot is not a globally
/// atomic transaction across all cells.  This matches the existing
/// [`compile_counters`](crate::compiler::compile_counters) and
/// [`deopt_counters`](crate::compiler::deopt_counters) diagnostic style.
#[must_use]
pub fn snapshot() -> IcCountersSnapshot {
    let per_tier = std::array::from_fn(|tier_idx| {
        let ops = std::array::from_fn(|op_idx| {
            let events = std::array::from_fn(|event_idx| {
                COUNTERS[tier_idx][op_idx][event_idx].load(Ordering::Relaxed)
            });
            IcOpSnapshot {
                op: IcOp::all()[op_idx],
                events,
            }
        });
        IcTierSnapshot {
            tier: IcTier::all()[tier_idx],
            ops,
        }
    });
    IcCountersSnapshot { per_tier }
}

/// Reset every IC counter to zero.
///
/// Intended for use in tests and benchmark harnesses; production telemetry
/// should prefer diffing successive [`snapshot`]s.
pub fn reset() {
    for tier in &COUNTERS {
        for op in tier {
            for event in op {
                event.store(0, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // The counters are process-global, so tests that mutate them must run
    // serially.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        match TEST_LOCK.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    #[test]
    fn test_ic_counters_stable_names() {
        assert_eq!(IcTier::Interpreter.name(), "interpreter");
        assert_eq!(IcTier::Baseline.name(), "baseline");
        assert_eq!(IcTier::Maglev.name(), "maglev");
        assert_eq!(IcTier::Turbofan.name(), "turbofan");

        assert_eq!(IcOp::NamedLoad.name(), "named_load");
        assert_eq!(IcOp::NamedStore.name(), "named_store");
        assert_eq!(IcOp::IndexedLoad.name(), "indexed_load");
        assert_eq!(IcOp::IndexedStore.name(), "indexed_store");
        assert_eq!(IcOp::Call.name(), "call");

        assert_eq!(IcEvent::Probe.name(), "probe");
        assert_eq!(IcEvent::Hit.name(), "hit");
        assert_eq!(IcEvent::Miss.name(), "miss");
        assert_eq!(IcEvent::Transition.name(), "transition");
    }

    #[test]
    fn test_ic_counters_stable_discriminants() {
        // Discriminants must remain stable for FFI consumers.
        assert_eq!(IcTier::Interpreter as usize, 0);
        assert_eq!(IcTier::Baseline as usize, 1);
        assert_eq!(IcTier::Maglev as usize, 2);
        assert_eq!(IcTier::Turbofan as usize, 3);

        assert_eq!(IcOp::NamedLoad as usize, 0);
        assert_eq!(IcOp::NamedStore as usize, 1);
        assert_eq!(IcOp::IndexedLoad as usize, 2);
        assert_eq!(IcOp::IndexedStore as usize, 3);
        assert_eq!(IcOp::Call as usize, 4);

        assert_eq!(IcEvent::Probe as usize, 0);
        assert_eq!(IcEvent::Hit as usize, 1);
        assert_eq!(IcEvent::Miss as usize, 2);
        assert_eq!(IcEvent::Transition as usize, 3);
    }

    #[test]
    fn test_reset_and_snapshot_round_trip() {
        let _g = lock();
        reset();
        let before = snapshot();
        assert_eq!(before.total(), 0);

        record_probe(IcTier::Interpreter, IcOp::NamedLoad);
        record_hit(IcTier::Interpreter, IcOp::NamedLoad);

        let mid = snapshot();
        assert_eq!(
            mid.for_tier(IcTier::Interpreter)
                .for_op(IcOp::NamedLoad)
                .count(IcEvent::Probe),
            1
        );
        assert_eq!(
            mid.for_tier(IcTier::Interpreter)
                .for_op(IcOp::NamedLoad)
                .count(IcEvent::Hit),
            1
        );

        // Snapshot must not mutate the counters.
        let mid2 = snapshot();
        assert_eq!(mid, mid2, "snapshot must be idempotent");

        reset();
        let after = snapshot();
        assert_eq!(after.total(), 0);
    }

    #[test]
    fn test_tier_and_op_attribution_is_isolated() {
        let _g = lock();
        reset();
        record_hit(IcTier::Interpreter, IcOp::NamedLoad);
        record_miss(IcTier::Interpreter, IcOp::NamedStore);
        record_transition(IcTier::Interpreter, IcOp::Call);

        let snap = snapshot();
        let interp = snap.for_tier(IcTier::Interpreter);
        assert_eq!(interp.for_op(IcOp::NamedLoad).count(IcEvent::Hit), 1);
        assert_eq!(interp.for_op(IcOp::NamedLoad).count(IcEvent::Miss), 0);
        assert_eq!(interp.for_op(IcOp::NamedStore).count(IcEvent::Miss), 1);
        assert_eq!(interp.for_op(IcOp::Call).count(IcEvent::Transition), 1);

        // No other (tier, op, event) cell may have changed.
        for tier in IcTier::all() {
            let tier_snap = snap.for_tier(tier);
            for op in IcOp::all() {
                let op_snap = tier_snap.for_op(op);
                for event in IcEvent::all() {
                    let expected = matches!(
                        (tier, op, event),
                        (IcTier::Interpreter, IcOp::NamedLoad, IcEvent::Hit)
                            | (IcTier::Interpreter, IcOp::NamedStore, IcEvent::Miss)
                            | (IcTier::Interpreter, IcOp::Call, IcEvent::Transition)
                    );
                    assert_eq!(
                        op_snap.count(event),
                        if expected { 1 } else { 0 },
                        "leak at ({:?}, {:?}, {:?})",
                        tier,
                        op,
                        event
                    );
                }
            }
        }
    }

    #[test]
    fn test_unsupported_tiers_stay_zero_when_interpreter_records() {
        let _g = lock();
        reset();
        // Hammer the Interpreter row with mixed events.
        for _ in 0..16 {
            record_probe(IcTier::Interpreter, IcOp::NamedLoad);
            record_hit(IcTier::Interpreter, IcOp::NamedLoad);
            record_probe(IcTier::Interpreter, IcOp::IndexedStore);
            record_miss(IcTier::Interpreter, IcOp::IndexedStore);
        }
        let snap = snapshot();
        assert!(snap.for_tier(IcTier::Interpreter).total() > 0);
        // Baseline / Maglev / Turbofan rows must remain zero in the absence
        // of real IC machinery feeding them.
        assert_eq!(snap.for_tier(IcTier::Baseline).total(), 0);
        assert_eq!(snap.for_tier(IcTier::Maglev).total(), 0);
        assert_eq!(snap.for_tier(IcTier::Turbofan).total(), 0);
    }

    #[test]
    fn test_no_counter_drift_across_snapshots() {
        let _g = lock();
        reset();
        record_probe(IcTier::Interpreter, IcOp::NamedLoad);
        record_hit(IcTier::Interpreter, IcOp::NamedLoad);

        let a = snapshot();
        let b = snapshot();
        let c = snapshot();
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn test_reset_clears_every_cell_then_no_drift() {
        let _g = lock();
        reset();
        for tier in IcTier::all() {
            for op in IcOp::all() {
                for event in IcEvent::all() {
                    record(tier, op, event);
                }
            }
        }
        assert!(snapshot().total() > 0);
        reset();
        let after = snapshot();
        assert_eq!(after.total(), 0);
        for tier in IcTier::all() {
            let tier_snap = after.for_tier(tier);
            for op in IcOp::all() {
                for event in IcEvent::all() {
                    assert_eq!(tier_snap.for_op(op).count(event), 0);
                }
            }
        }
    }
}
