//! Release-safe deoptimisation and bailout histograms by JIT tier.
//!
//! The counters in this module are intentionally small, stable, and always
//! available in release builds so Edge proof runs can explain tier instability
//! without relying on debug-only tracing.  Only real deopt/bailout/fallback
//! sites should call [`record_deopt`].  Tiers without active deopt machinery
//! remain visible with zeroed counters rather than synthesising events.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::compiler::baseline::compiler::{
    JIT_DEOPT, JIT_DEOPT_DIVZERO, JIT_DEOPT_GLOBAL, JIT_DEOPT_OVERFLOW, JIT_DEOPT_STUB,
    JIT_DEOPT_TERMINATED,
};

/// JIT tiers tracked by deopt histograms.
///
/// The numeric discriminants double as indices and must remain stable for FFI
/// consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum DeoptTier {
    /// Non-optimising baseline JIT.
    Baseline = 0,
    /// Maglev mid-tier optimising compiler.
    Maglev = 1,
    /// Cranelift-backed Turbofan optimising compiler.
    Turbofan = 2,
}

impl DeoptTier {
    /// Number of tier variants tracked by the deopt histogram.
    pub const COUNT: usize = 3;

    /// Stable short name suitable for diagnostic dumps and FFI labels.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Maglev => "maglev",
            Self::Turbofan => "turbofan",
        }
    }

    /// All tier variants in stable discriminant order.
    #[must_use]
    pub const fn all() -> [Self; Self::COUNT] {
        [Self::Baseline, Self::Maglev, Self::Turbofan]
    }
}

/// Stable deopt/bailout/fallback reasons observed by Stator today.
///
/// Reasons are deliberately broad enough to be ABI-stable while still mapping
/// one-to-one to currently emitted JIT deopt sentinels.  The numeric
/// discriminants double as indices and must remain stable for FFI consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum DeoptReason {
    /// The tier hit unsupported bytecode/codegen and fell back to a lower tier.
    UnsupportedOpcodeOrRuntimeFallback = 0,
    /// A checked small-integer arithmetic operation overflowed.
    ArithmeticOverflow = 1,
    /// A runtime stub reported it could not satisfy the specialised fast path.
    RuntimeStubFallback = 2,
    /// A promoted global load/store fast path failed.
    GlobalLoadFallback = 3,
    /// Embedder termination or interrupt polling requested execution unwind.
    TerminationInterrupt = 4,
    /// Integer division by zero would need non-fast-path semantics.
    DivisionByZero = 5,
    /// Unknown, internal, or out-of-range deopt sentinel.
    InternalError = 6,
}

impl DeoptReason {
    /// Number of reason variants tracked by the deopt histogram.
    pub const COUNT: usize = 7;

    /// Stable short name suitable for diagnostic dumps and FFI labels.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::UnsupportedOpcodeOrRuntimeFallback => "unsupported_opcode_or_runtime_fallback",
            Self::ArithmeticOverflow => "arithmetic_overflow",
            Self::RuntimeStubFallback => "runtime_stub_fallback",
            Self::GlobalLoadFallback => "global_load_fallback",
            Self::TerminationInterrupt => "termination_interrupt",
            Self::DivisionByZero => "division_by_zero",
            Self::InternalError => "internal_error",
        }
    }

    /// All reason variants in stable discriminant order.
    #[must_use]
    pub const fn all() -> [Self; Self::COUNT] {
        [
            Self::UnsupportedOpcodeOrRuntimeFallback,
            Self::ArithmeticOverflow,
            Self::RuntimeStubFallback,
            Self::GlobalLoadFallback,
            Self::TerminationInterrupt,
            Self::DivisionByZero,
            Self::InternalError,
        ]
    }
}

static COUNTERS: [[AtomicU64; DeoptReason::COUNT]; DeoptTier::COUNT] =
    [const { [const { AtomicU64::new(0) }; DeoptReason::COUNT] }; DeoptTier::COUNT];

/// Record a deopt, bailout, or runtime fallback for `tier` and `reason`.
#[inline]
pub fn record_deopt(tier: DeoptTier, reason: DeoptReason) {
    COUNTERS[tier as usize][reason as usize].fetch_add(1, Ordering::Relaxed);
}

/// Map a JIT return sentinel to a stable [`DeoptReason`].
#[must_use]
pub const fn reason_from_jit_deopt_value(value: i64) -> DeoptReason {
    match value {
        JIT_DEOPT => DeoptReason::UnsupportedOpcodeOrRuntimeFallback,
        JIT_DEOPT_OVERFLOW => DeoptReason::ArithmeticOverflow,
        JIT_DEOPT_STUB => DeoptReason::RuntimeStubFallback,
        JIT_DEOPT_GLOBAL => DeoptReason::GlobalLoadFallback,
        JIT_DEOPT_TERMINATED => DeoptReason::TerminationInterrupt,
        JIT_DEOPT_DIVZERO => DeoptReason::DivisionByZero,
        _ => DeoptReason::InternalError,
    }
}

/// Record a tier deopt returned as one of the JIT sentinel values.
#[inline]
pub fn record_jit_deopt_value(tier: DeoptTier, value: i64) -> DeoptReason {
    let reason = reason_from_jit_deopt_value(value);
    record_deopt(tier, reason);
    reason
}

/// Per-tier portion of a [`DeoptHistogramSnapshot`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeoptTierSnapshot {
    /// The tier these counters describe.
    pub tier: DeoptTier,
    /// One counter per [`DeoptReason`] in stable discriminant order.
    pub counts: [u64; DeoptReason::COUNT],
}

impl DeoptTierSnapshot {
    /// Look up a reason counter.
    #[must_use]
    pub fn count(&self, reason: DeoptReason) -> u64 {
        self.counts[reason as usize]
    }

    /// Total deopts recorded for this tier.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }
}

/// Immutable snapshot of every deopt histogram counter at a point in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeoptHistogramSnapshot {
    /// One snapshot per [`DeoptTier`] in stable discriminant order.
    pub per_tier: [DeoptTierSnapshot; DeoptTier::COUNT],
}

impl DeoptHistogramSnapshot {
    /// Look up the snapshot for a specific tier.
    #[must_use]
    pub fn for_tier(&self, tier: DeoptTier) -> &DeoptTierSnapshot {
        &self.per_tier[tier as usize]
    }

    /// Total deopts recorded across all tiers and reasons.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.per_tier.iter().map(DeoptTierSnapshot::total).sum()
    }
}

/// Take a snapshot of every deopt histogram counter.
#[must_use]
pub fn snapshot() -> DeoptHistogramSnapshot {
    let per_tier = std::array::from_fn(|tier_idx| {
        let counts = std::array::from_fn(|reason_idx| {
            COUNTERS[tier_idx][reason_idx].load(Ordering::Relaxed)
        });
        DeoptTierSnapshot {
            tier: DeoptTier::all()[tier_idx],
            counts,
        }
    });
    DeoptHistogramSnapshot { per_tier }
}

/// Reset every deopt histogram counter to zero.
pub fn reset() {
    for tier in &COUNTERS {
        for counter in tier {
            counter.store(0, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        match TEST_LOCK.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    #[test]
    fn test_deopt_histogram_stable_names() {
        assert_eq!(DeoptTier::Baseline.name(), "baseline");
        assert_eq!(DeoptTier::Maglev.name(), "maglev");
        assert_eq!(DeoptTier::Turbofan.name(), "turbofan");
        assert_eq!(
            DeoptReason::UnsupportedOpcodeOrRuntimeFallback.name(),
            "unsupported_opcode_or_runtime_fallback"
        );
        assert_eq!(DeoptReason::InternalError.name(), "internal_error");
    }

    #[test]
    fn test_deopt_histogram_reset_snapshot_and_no_drift() {
        let _g = lock();
        reset();
        let before = snapshot();
        assert_eq!(before.total(), 0);

        record_deopt(DeoptTier::Maglev, DeoptReason::RuntimeStubFallback);
        record_deopt(DeoptTier::Maglev, DeoptReason::RuntimeStubFallback);
        record_deopt(DeoptTier::Turbofan, DeoptReason::InternalError);

        let snap = snapshot();
        assert_eq!(
            snap.for_tier(DeoptTier::Maglev)
                .count(DeoptReason::RuntimeStubFallback),
            2
        );
        assert_eq!(
            snap.for_tier(DeoptTier::Turbofan)
                .count(DeoptReason::InternalError),
            1
        );
        assert_eq!(snap.for_tier(DeoptTier::Baseline).total(), 0);

        reset();
        let after = snapshot();
        assert_eq!(after.total(), 0);
        assert_eq!(snapshot(), after, "snapshot must not mutate counters");
    }

    #[test]
    fn test_deopt_histogram_maps_real_jit_sentinels() {
        let _g = lock();
        reset();
        assert_eq!(
            record_jit_deopt_value(DeoptTier::Maglev, JIT_DEOPT),
            DeoptReason::UnsupportedOpcodeOrRuntimeFallback
        );
        assert_eq!(
            record_jit_deopt_value(DeoptTier::Maglev, JIT_DEOPT_OVERFLOW),
            DeoptReason::ArithmeticOverflow
        );
        assert_eq!(
            record_jit_deopt_value(DeoptTier::Maglev, JIT_DEOPT_STUB),
            DeoptReason::RuntimeStubFallback
        );
        assert_eq!(
            record_jit_deopt_value(DeoptTier::Maglev, JIT_DEOPT_GLOBAL),
            DeoptReason::GlobalLoadFallback
        );
        assert_eq!(
            record_jit_deopt_value(DeoptTier::Maglev, JIT_DEOPT_TERMINATED),
            DeoptReason::TerminationInterrupt
        );
        assert_eq!(
            record_jit_deopt_value(DeoptTier::Maglev, JIT_DEOPT_DIVZERO),
            DeoptReason::DivisionByZero
        );
        assert_eq!(
            record_jit_deopt_value(DeoptTier::Maglev, i64::MAX),
            DeoptReason::InternalError
        );

        let maglev = snapshot().for_tier(DeoptTier::Maglev).counts;
        assert_eq!(maglev, [1; DeoptReason::COUNT]);
    }

    #[test]
    fn test_deopt_histogram_inactive_tiers_stay_zero_after_maglev_recording() {
        let _g = lock();
        reset();
        record_jit_deopt_value(DeoptTier::Maglev, JIT_DEOPT_STUB);
        let snap = snapshot();
        assert_eq!(snap.for_tier(DeoptTier::Baseline).total(), 0);
        assert_eq!(snap.for_tier(DeoptTier::Turbofan).total(), 0);
    }
}
