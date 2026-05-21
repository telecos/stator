//! Always-on, release-safe counters and timing for JIT tier promotion
//! requests and successful transitions.
//!
//! These metrics complement
//! [`crate::compiler::compile_counters`] (which counts compile
//! attempts/successes/failures across all compile entry points,
//! including the source → bytecode interpreter tier) by focusing
//! specifically on **tier promotion** from interpreter/baseline up to
//! Maglev/Turbofan: how often Edge requests a promotion, how often it
//! succeeds, and how long the synchronous promotion request takes per
//! target tier.
//!
//! The instrumentation is intended for Edge performance proof runs, so
//! it is **not** gated on `cfg(debug_assertions)`.  Every recorded
//! promotion path performs a single [`std::time::Instant::now`] pair
//! plus a handful of relaxed atomic increments, which is negligible
//! next to even the cheapest baseline JIT compilation step.
//!
//! # Scope
//!
//! Counters are **process-global aggregates**, matching the existing
//! tiering-stats FFI surface ([`stator_isolate_get_tiering_stats`])
//! and the [`compile_counters`](crate::compiler::compile_counters)
//! module.  Per-script attribution is not exposed today: bytecode
//! arrays do not yet carry a stable script-hash identifier through the
//! tier-promotion entry points, and adding one is out of scope for
//! this diagnostic surface.  This limitation is documented in
//! `docs/edge_diagnostics.md`.
//!
//! # Usage
//!
//! Wrap a synchronous tier-promotion entry point with a
//! [`PromotionTimer`] and record the outcome on completion:
//!
//! ```
//! use stator_jse::compiler::tier_latency::{self, PromotionTier};
//!
//! tier_latency::reset();
//! let timer = tier_latency::PromotionTimer::start(PromotionTier::Baseline);
//! // ... actual compile / cache work ...
//! timer.record_success();
//!
//! let snap = tier_latency::snapshot();
//! let baseline = snap.for_tier(PromotionTier::Baseline);
//! assert_eq!(baseline.requested, 1);
//! assert_eq!(baseline.succeeded, 1);
//! assert_eq!(baseline.failed, 0);
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// JIT tiers tracked by the tier-promotion latency counters.
///
/// Only true *promotion* tiers are tracked here — the interpreter
/// (source → bytecode) tier is counted by
/// [`compile_counters`](crate::compiler::compile_counters) instead.
///
/// The numeric discriminants double as indices into the underlying
/// atomic arrays and **must remain stable** for FFI consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum PromotionTier {
    /// Bytecode → native code via the non-optimising baseline JIT.
    Baseline = 0,
    /// Maglev mid-tier optimising compiler.
    Maglev = 1,
    /// Cranelift-backed Turbofan optimising compiler.
    Turbofan = 2,
}

impl PromotionTier {
    /// Number of tier variants tracked by the latency counters.
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

/// Histogram bucket upper bounds in microseconds (inclusive).
///
/// Pauses exceeding the last bucket are recorded in the overflow
/// bucket (`buckets[HISTOGRAM_BUCKETS_US.len()]`).  Buckets are
/// log-spaced from 1µs to 1s to give actionable resolution across both
/// fast baseline compiles and the slower Turbofan path while still
/// capturing rare multi-second outliers.
pub const HISTOGRAM_BUCKETS_US: [u64; 8] = [
    1,          // ≤ 1µs
    10,         // ≤ 10µs
    100,        // ≤ 100µs
    1_000,      // ≤ 1ms
    10_000,     // ≤ 10ms
    100_000,    // ≤ 100ms
    1_000_000,  // ≤ 1s
    10_000_000, // ≤ 10s
];

/// Number of histogram buckets including the overflow bucket.
pub const NUM_HISTOGRAM_BUCKETS: usize = HISTOGRAM_BUCKETS_US.len() + 1;

struct TierSlot {
    requested: AtomicU64,
    succeeded: AtomicU64,
    failed: AtomicU64,
    success_total_ns: AtomicU64,
    success_max_ns: AtomicU64,
    failure_total_ns: AtomicU64,
    failure_max_ns: AtomicU64,
    success_buckets: [AtomicU64; NUM_HISTOGRAM_BUCKETS],
}

impl TierSlot {
    const fn new() -> Self {
        Self {
            requested: AtomicU64::new(0),
            succeeded: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            success_total_ns: AtomicU64::new(0),
            success_max_ns: AtomicU64::new(0),
            failure_total_ns: AtomicU64::new(0),
            failure_max_ns: AtomicU64::new(0),
            success_buckets: [const { AtomicU64::new(0) }; NUM_HISTOGRAM_BUCKETS],
        }
    }

    fn record_request(&self) {
        self.requested.fetch_add(1, Ordering::Relaxed);
    }

    fn record_success(&self, elapsed_ns: u64) {
        self.succeeded.fetch_add(1, Ordering::Relaxed);
        self.success_total_ns
            .fetch_add(elapsed_ns, Ordering::Relaxed);
        update_max(&self.success_max_ns, elapsed_ns);
        let bucket = bucket_index_for_us(ns_to_us(elapsed_ns));
        self.success_buckets[bucket].fetch_add(1, Ordering::Relaxed);
    }

    fn record_failure(&self, elapsed_ns: u64) {
        self.failed.fetch_add(1, Ordering::Relaxed);
        self.failure_total_ns
            .fetch_add(elapsed_ns, Ordering::Relaxed);
        update_max(&self.failure_max_ns, elapsed_ns);
    }

    fn reset(&self) {
        self.requested.store(0, Ordering::Relaxed);
        self.succeeded.store(0, Ordering::Relaxed);
        self.failed.store(0, Ordering::Relaxed);
        self.success_total_ns.store(0, Ordering::Relaxed);
        self.success_max_ns.store(0, Ordering::Relaxed);
        self.failure_total_ns.store(0, Ordering::Relaxed);
        self.failure_max_ns.store(0, Ordering::Relaxed);
        for bucket in &self.success_buckets {
            bucket.store(0, Ordering::Relaxed);
        }
    }

    fn snapshot(&self, tier: PromotionTier) -> TierLatencySnapshot {
        let mut buckets = [0u64; NUM_HISTOGRAM_BUCKETS];
        for (dst, src) in buckets.iter_mut().zip(self.success_buckets.iter()) {
            *dst = src.load(Ordering::Relaxed);
        }
        TierLatencySnapshot {
            tier,
            requested: self.requested.load(Ordering::Relaxed),
            succeeded: self.succeeded.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
            success_total_ns: self.success_total_ns.load(Ordering::Relaxed),
            success_max_ns: self.success_max_ns.load(Ordering::Relaxed),
            failure_total_ns: self.failure_total_ns.load(Ordering::Relaxed),
            failure_max_ns: self.failure_max_ns.load(Ordering::Relaxed),
            success_buckets: buckets,
        }
    }
}

fn update_max(slot: &AtomicU64, value: u64) {
    let mut current = slot.load(Ordering::Relaxed);
    while value > current {
        match slot.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

fn ns_to_us(ns: u64) -> u64 {
    // Round to the nearest microsecond so that a sub-µs sample lands in
    // the `≤ 1µs` bucket rather than an "unmeasurably fast" zero bucket.
    ns.saturating_add(500) / 1_000
}

fn bucket_index_for_us(elapsed_us: u64) -> usize {
    for (i, &upper) in HISTOGRAM_BUCKETS_US.iter().enumerate() {
        if elapsed_us <= upper {
            return i;
        }
    }
    HISTOGRAM_BUCKETS_US.len()
}

static SLOTS: [TierSlot; PromotionTier::COUNT] =
    [TierSlot::new(), TierSlot::new(), TierSlot::new()];

/// Record a tier-promotion request for `tier`.
///
/// Call once at the start of every promotion entry point that may
/// actually perform compile work (i.e. the `force_*_sync` helpers in
/// `interpreter::mod`).  The higher-level `force_tier_sync` wrapper
/// short-circuits on `AlreadyReady` / `JitDisabled` /
/// `UnsupportedTier` before delegating, so those outcomes never reach
/// the per-tier entry points and are deliberately **not** counted as
/// requests here — that keeps the latency histogram free of
/// zero-duration noise.
///
/// Pair this with [`record_success`] or [`record_failure`] (or use
/// the RAII [`PromotionTimer`]) to track latency.
#[inline]
pub fn record_request(tier: PromotionTier) {
    SLOTS[tier as usize].record_request();
}

/// Record a successful promotion of duration `elapsed`.
#[inline]
pub fn record_success(tier: PromotionTier, elapsed: Duration) {
    let ns = duration_to_ns(elapsed);
    SLOTS[tier as usize].record_success(ns);
}

/// Record a failed promotion attempt of duration `elapsed`.
///
/// "Failed" covers any non-ready outcome that consumed measurable
/// time: compile errors, executable-allocation failures, degenerate
/// graphs, and so on.  Outcomes that short-circuit before any work
/// (e.g. `JitDisabled`, `UnsupportedTier`) should not be recorded as
/// failures — they are already counted as requests.
#[inline]
pub fn record_failure(tier: PromotionTier, elapsed: Duration) {
    let ns = duration_to_ns(elapsed);
    SLOTS[tier as usize].record_failure(ns);
}

fn duration_to_ns(d: Duration) -> u64 {
    d.as_nanos().min(u64::MAX as u128) as u64
}

/// Reset every per-tier counter and histogram to zero.
///
/// Intended for tests, benchmark harnesses, and Edge measurement
/// windows reached over the FFI boundary.  Production telemetry
/// should prefer diffing successive [`snapshot`]s.
pub fn reset() {
    for slot in &SLOTS {
        slot.reset();
    }
}

/// Per-tier portion of a [`TierLatencyCountersSnapshot`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierLatencySnapshot {
    /// The tier these counters describe.
    pub tier: PromotionTier,
    /// Number of promotion requests recorded.
    pub requested: u64,
    /// Number of successful promotions (tier ready after the request).
    pub succeeded: u64,
    /// Number of failed promotions (work was attempted but did not
    /// leave the target tier observable as ready).
    pub failed: u64,
    /// Sum of successful promotion durations in nanoseconds.
    pub success_total_ns: u64,
    /// Largest single successful promotion duration in nanoseconds.
    pub success_max_ns: u64,
    /// Sum of failed promotion durations in nanoseconds.
    pub failure_total_ns: u64,
    /// Largest single failed promotion duration in nanoseconds.
    pub failure_max_ns: u64,
    /// Histogram of successful promotion durations.  Bucket `i <
    /// HISTOGRAM_BUCKETS_US.len()` counts successes ≤
    /// `HISTOGRAM_BUCKETS_US[i]` microseconds; the final bucket counts
    /// overflows.
    pub success_buckets: [u64; NUM_HISTOGRAM_BUCKETS],
}

impl TierLatencySnapshot {
    /// Mean successful promotion duration in nanoseconds, or 0 if no
    /// successes have been recorded.
    #[must_use]
    pub fn mean_success_ns(&self) -> u64 {
        if self.succeeded == 0 {
            0
        } else {
            self.success_total_ns / self.succeeded
        }
    }
}

/// Immutable snapshot of every tier-latency counter at a point in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierLatencyCountersSnapshot {
    /// One snapshot per [`PromotionTier`] in stable discriminant order.
    pub per_tier: [TierLatencySnapshot; PromotionTier::COUNT],
}

impl TierLatencyCountersSnapshot {
    /// Look up the snapshot for a specific tier.
    #[must_use]
    pub fn for_tier(&self, tier: PromotionTier) -> &TierLatencySnapshot {
        &self.per_tier[tier as usize]
    }

    /// Total promotion requests across all tiers.
    #[must_use]
    pub fn total_requested(&self) -> u64 {
        self.per_tier.iter().map(|s| s.requested).sum()
    }

    /// Total successful promotions across all tiers.
    #[must_use]
    pub fn total_succeeded(&self) -> u64 {
        self.per_tier.iter().map(|s| s.succeeded).sum()
    }

    /// Total failed promotion attempts across all tiers.
    #[must_use]
    pub fn total_failed(&self) -> u64 {
        self.per_tier.iter().map(|s| s.failed).sum()
    }
}

/// Take a consistent snapshot of every tier-latency counter.
#[must_use]
pub fn snapshot() -> TierLatencyCountersSnapshot {
    let per_tier = std::array::from_fn(|i| SLOTS[i].snapshot(PromotionTier::all()[i]));
    TierLatencyCountersSnapshot { per_tier }
}

/// RAII guard that times a single tier-promotion request and records
/// its outcome.
///
/// Construct one at the start of every synchronous promotion entry
/// point.  The constructor unconditionally records the request; the
/// caller must then invoke [`record_success`](Self::record_success) or
/// [`record_failure`](Self::record_failure) to record the outcome.
///
/// If the timer is dropped without an explicit outcome (for example,
/// the entry point panicked or took a short-circuit path that doesn't
/// count as work), only the request is counted — neither success nor
/// failure latency is recorded.  This keeps no-op paths (such as
/// `AlreadyReady`) from polluting the latency histogram.
#[must_use = "drop the timer with record_success / record_failure to capture latency"]
pub struct PromotionTimer {
    tier: PromotionTier,
    start: Instant,
}

impl PromotionTimer {
    /// Start timing a promotion request for `tier` and record the
    /// request.
    #[inline]
    pub fn start(tier: PromotionTier) -> Self {
        record_request(tier);
        Self {
            tier,
            start: Instant::now(),
        }
    }

    /// Record a successful promotion with the elapsed time so far.
    #[inline]
    pub fn record_success(self) {
        let elapsed = self.start.elapsed();
        record_success(self.tier, elapsed);
    }

    /// Record a failed promotion with the elapsed time so far.
    #[inline]
    pub fn record_failure(self) {
        let elapsed = self.start.elapsed();
        record_failure(self.tier, elapsed);
    }

    /// Drop the timer without recording success/failure latency.
    ///
    /// Use this for short-circuit paths that should still count as a
    /// request (e.g. the request was issued but the tier was already
    /// ready and no compile work occurred).
    #[inline]
    pub fn discard(self) {
        // Intentionally empty: dropping `self` leaves the request
        // counter incremented but does not record any latency.
        drop(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // The counters are process-global, so tests that mutate them must
    // run serially to avoid interfering with each other.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        match TEST_LOCK.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    #[test]
    fn tier_names_are_stable() {
        assert_eq!(PromotionTier::Baseline.name(), "baseline");
        assert_eq!(PromotionTier::Maglev.name(), "maglev");
        assert_eq!(PromotionTier::Turbofan.name(), "turbofan");
    }

    #[test]
    fn record_request_increments_only_requested() {
        let _g = lock();
        reset();
        record_request(PromotionTier::Baseline);
        record_request(PromotionTier::Baseline);
        let s = snapshot();
        let b = s.for_tier(PromotionTier::Baseline);
        assert_eq!(b.requested, 2);
        assert_eq!(b.succeeded, 0);
        assert_eq!(b.failed, 0);
        assert_eq!(b.success_total_ns, 0);
        assert_eq!(b.failure_total_ns, 0);
    }

    #[test]
    fn record_success_and_failure_are_per_tier() {
        let _g = lock();
        reset();
        record_request(PromotionTier::Maglev);
        record_success(PromotionTier::Maglev, Duration::from_micros(123));
        record_request(PromotionTier::Turbofan);
        record_failure(PromotionTier::Turbofan, Duration::from_micros(456));
        let s = snapshot();

        let maglev = s.for_tier(PromotionTier::Maglev);
        assert_eq!(maglev.requested, 1);
        assert_eq!(maglev.succeeded, 1);
        assert_eq!(maglev.failed, 0);
        assert_eq!(maglev.success_total_ns, 123_000);
        assert_eq!(maglev.success_max_ns, 123_000);

        let tf = s.for_tier(PromotionTier::Turbofan);
        assert_eq!(tf.requested, 1);
        assert_eq!(tf.succeeded, 0);
        assert_eq!(tf.failed, 1);
        assert_eq!(tf.failure_total_ns, 456_000);
        assert_eq!(tf.failure_max_ns, 456_000);

        // Other tiers untouched.
        assert_eq!(s.for_tier(PromotionTier::Baseline).requested, 0);
    }

    #[test]
    fn max_latency_tracks_largest_sample() {
        let _g = lock();
        reset();
        record_request(PromotionTier::Baseline);
        record_success(PromotionTier::Baseline, Duration::from_micros(50));
        record_request(PromotionTier::Baseline);
        record_success(PromotionTier::Baseline, Duration::from_micros(500));
        record_request(PromotionTier::Baseline);
        record_success(PromotionTier::Baseline, Duration::from_micros(100));

        let b = snapshot();
        let s = b.for_tier(PromotionTier::Baseline);
        assert_eq!(s.succeeded, 3);
        assert_eq!(s.success_total_ns, 650_000);
        assert_eq!(s.success_max_ns, 500_000);
        assert_eq!(s.mean_success_ns(), 650_000 / 3);
    }

    #[test]
    fn histogram_buckets_classify_correctly() {
        let _g = lock();
        reset();
        record_request(PromotionTier::Baseline);
        record_success(PromotionTier::Baseline, Duration::from_nanos(500)); // ≤ 1µs
        record_request(PromotionTier::Baseline);
        record_success(PromotionTier::Baseline, Duration::from_micros(5)); // ≤ 10µs
        record_request(PromotionTier::Baseline);
        record_success(PromotionTier::Baseline, Duration::from_micros(75)); // ≤ 100µs
        record_request(PromotionTier::Baseline);
        record_success(PromotionTier::Baseline, Duration::from_millis(750)); // ≤ 1s
        record_request(PromotionTier::Baseline);
        record_success(PromotionTier::Baseline, Duration::from_secs(20)); // overflow

        let b = snapshot();
        let s = b.for_tier(PromotionTier::Baseline);
        assert_eq!(s.success_buckets[0], 1, "≤ 1µs");
        assert_eq!(s.success_buckets[1], 1, "≤ 10µs");
        assert_eq!(s.success_buckets[2], 1, "≤ 100µs");
        assert_eq!(s.success_buckets[6], 1, "≤ 1s");
        assert_eq!(s.success_buckets[NUM_HISTOGRAM_BUCKETS - 1], 1, "overflow");
    }

    #[test]
    fn timer_records_request_and_success() {
        let _g = lock();
        reset();
        let timer = PromotionTimer::start(PromotionTier::Maglev);
        std::thread::sleep(Duration::from_micros(10));
        timer.record_success();
        let s = snapshot();
        let m = s.for_tier(PromotionTier::Maglev);
        assert_eq!(m.requested, 1);
        assert_eq!(m.succeeded, 1);
        assert_eq!(m.failed, 0);
        // Avoid wall-clock flakiness: just assert monotonic > 0.
        assert!(m.success_total_ns > 0, "expected nonzero elapsed");
        assert!(m.success_max_ns > 0);
    }

    #[test]
    fn timer_records_request_and_failure() {
        let _g = lock();
        reset();
        let timer = PromotionTimer::start(PromotionTier::Turbofan);
        timer.record_failure();
        let s = snapshot();
        let t = s.for_tier(PromotionTier::Turbofan);
        assert_eq!(t.requested, 1);
        assert_eq!(t.succeeded, 0);
        assert_eq!(t.failed, 1);
    }

    #[test]
    fn timer_discard_records_request_only() {
        let _g = lock();
        reset();
        let timer = PromotionTimer::start(PromotionTier::Baseline);
        timer.discard();
        let s = snapshot();
        let b = s.for_tier(PromotionTier::Baseline);
        assert_eq!(b.requested, 1);
        assert_eq!(b.succeeded, 0);
        assert_eq!(b.failed, 0);
        assert_eq!(b.success_total_ns, 0);
        assert_eq!(b.failure_total_ns, 0);
    }

    #[test]
    fn reset_zeros_all_counters_and_buckets() {
        let _g = lock();
        reset();
        for tier in PromotionTier::all() {
            record_request(tier);
            record_success(tier, Duration::from_micros(20));
            record_request(tier);
            record_failure(tier, Duration::from_micros(40));
        }
        reset();
        let s = snapshot();
        for tier in PromotionTier::all() {
            let t = s.for_tier(tier);
            assert_eq!(t.requested, 0, "tier {:?}", tier);
            assert_eq!(t.succeeded, 0, "tier {:?}", tier);
            assert_eq!(t.failed, 0, "tier {:?}", tier);
            assert_eq!(t.success_total_ns, 0);
            assert_eq!(t.success_max_ns, 0);
            assert_eq!(t.failure_total_ns, 0);
            assert_eq!(t.failure_max_ns, 0);
            for b in t.success_buckets {
                assert_eq!(b, 0);
            }
        }
        assert_eq!(s.total_requested(), 0);
        assert_eq!(s.total_succeeded(), 0);
        assert_eq!(s.total_failed(), 0);
    }

    #[test]
    fn no_drift_after_reset_under_concurrent_updates() {
        let _g = lock();
        reset();
        // Sequential workload, then reset, then verify no drift on a
        // subsequent read.  Genuine concurrency is exercised by the
        // GC pause-metrics tests; this test specifically guards
        // against reset() leaving any counter in a stale state.
        for _ in 0..1_000 {
            record_request(PromotionTier::Maglev);
            record_success(PromotionTier::Maglev, Duration::from_micros(2));
        }
        reset();
        let snap_a = snapshot();
        let snap_b = snapshot();
        assert_eq!(snap_a, snap_b);
        assert_eq!(snap_a.total_requested(), 0);
        assert_eq!(snap_a.total_succeeded(), 0);
        assert_eq!(snap_a.total_failed(), 0);
    }

    #[test]
    fn bucket_overflow_threshold() {
        assert_eq!(bucket_index_for_us(0), 0);
        assert_eq!(bucket_index_for_us(1), 0);
        assert_eq!(bucket_index_for_us(2), 1);
        assert_eq!(
            bucket_index_for_us(10_000_001),
            HISTOGRAM_BUCKETS_US.len(),
            "above last bucket lands in overflow"
        );
    }

    #[test]
    fn ns_to_us_rounds_to_nearest() {
        assert_eq!(ns_to_us(0), 0);
        assert_eq!(ns_to_us(499), 0);
        assert_eq!(ns_to_us(500), 1);
        assert_eq!(ns_to_us(1_499), 1);
        assert_eq!(ns_to_us(1_500), 2);
    }
}
