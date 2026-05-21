//! Always-on GC pause-time counters and fixed-bucket histogram.
//!
//! These metrics are designed to be safe to enable in release builds with
//! negligible overhead per collection (a handful of relaxed atomic increments
//! plus one `Instant::now()` pair at the GC entry/exit boundary).  No metrics
//! are recorded on the allocation fast path.
//!
//! # Use
//!
//! Wrap a GC collection entry point with a [`PauseTimer`] so the elapsed
//! pause time is recorded automatically on drop:
//!
//! ```ignore
//! pub fn collect(&mut self) {
//!     let _timer = PauseTimer::start(GcCollectionKind::Minor);
//!     // ... actual collection work ...
//! }
//! ```
//!
//! Embedders read the accumulated counters with [`snapshot`] and may reset
//! them between measurement windows with [`reset`].  Both calls are intended
//! for diagnostics and the future FFI surface and acquire no locks.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Histogram bucket upper bounds in microseconds (inclusive).
///
/// Any pause exceeding the last bucket is recorded in the overflow bucket
/// (`buckets[HISTOGRAM_BUCKETS_US.len()]`).  Buckets are intentionally
/// log-spaced from 10µs to 50ms to give actionable resolution across the
/// typical sub-millisecond pause-time target while still capturing rare
/// outliers in the tens-of-milliseconds range.
pub const HISTOGRAM_BUCKETS_US: [u64; 8] = [10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000];

/// Number of histogram buckets including the overflow bucket.
pub const NUM_HISTOGRAM_BUCKETS: usize = HISTOGRAM_BUCKETS_US.len() + 1;

/// The category of GC pause being recorded.
///
/// Each kind has an independent set of counters so embedders can distinguish
/// minor (nursery) scavenges from major (full) collections and analyse the
/// Immix and Mark-Sweep-Compact paths separately.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum GcCollectionKind {
    /// Minor / young-generation collection (Cheney scavenge or nursery swap).
    Minor,
    /// Major / full-heap collection (covers both nursery + old-gen work).
    Major,
    /// Immix regional collection (mark → evacuate → recycle).
    Immix,
    /// Old-generation Mark-Sweep-Compact phase.
    MarkSweepCompact,
}

impl GcCollectionKind {
    const ALL: [Self; 4] = [
        Self::Minor,
        Self::Major,
        Self::Immix,
        Self::MarkSweepCompact,
    ];

    const fn index(self) -> usize {
        match self {
            Self::Minor => 0,
            Self::Major => 1,
            Self::Immix => 2,
            Self::MarkSweepCompact => 3,
        }
    }

    /// Short stable identifier suitable for diagnostic dumps and FFI labels.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Minor => "minor",
            Self::Major => "major",
            Self::Immix => "immix",
            Self::MarkSweepCompact => "mark_sweep_compact",
        }
    }
}

/// Number of distinct [`GcCollectionKind`] variants.
pub const NUM_GC_KINDS: usize = GcCollectionKind::ALL.len();

/// Lock-free per-kind counters backed by relaxed atomics.
struct KindCounters {
    count: AtomicU64,
    total_pause_us: AtomicU64,
    max_pause_us: AtomicU64,
    buckets: [AtomicU64; NUM_HISTOGRAM_BUCKETS],
}

impl KindCounters {
    const fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_pause_us: AtomicU64::new(0),
            max_pause_us: AtomicU64::new(0),
            buckets: [const { AtomicU64::new(0) }; NUM_HISTOGRAM_BUCKETS],
        }
    }

    fn record(&self, pause_us: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_pause_us.fetch_add(pause_us, Ordering::Relaxed);
        // Lock-free max update via CAS loop.  Pauses are infrequent so
        // contention here is negligible.
        let mut current = self.max_pause_us.load(Ordering::Relaxed);
        while pause_us > current {
            match self.max_pause_us.compare_exchange_weak(
                current,
                pause_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
        let bucket = bucket_index_for(pause_us);
        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
    }

    fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.total_pause_us.store(0, Ordering::Relaxed);
        self.max_pause_us.store(0, Ordering::Relaxed);
        for bucket in &self.buckets {
            bucket.store(0, Ordering::Relaxed);
        }
    }

    fn snapshot(&self, kind: GcCollectionKind) -> GcKindPauseSnapshot {
        let mut buckets = [0u64; NUM_HISTOGRAM_BUCKETS];
        for (dst, src) in buckets.iter_mut().zip(self.buckets.iter()) {
            *dst = src.load(Ordering::Relaxed);
        }
        GcKindPauseSnapshot {
            kind,
            count: self.count.load(Ordering::Relaxed),
            total_pause_us: self.total_pause_us.load(Ordering::Relaxed),
            max_pause_us: self.max_pause_us.load(Ordering::Relaxed),
            buckets,
        }
    }
}

static METRICS: [KindCounters; NUM_GC_KINDS] = [
    KindCounters::new(),
    KindCounters::new(),
    KindCounters::new(),
    KindCounters::new(),
];

fn bucket_index_for(pause_us: u64) -> usize {
    for (i, &upper) in HISTOGRAM_BUCKETS_US.iter().enumerate() {
        if pause_us <= upper {
            return i;
        }
    }
    HISTOGRAM_BUCKETS_US.len()
}

/// Record a single GC pause of `duration` for `kind`.
///
/// Saturates to `u64::MAX` microseconds for pauses larger than ~584,000 years
/// (effectively impossible in practice).  This call is constant-time and
/// performs only relaxed atomic operations.
pub fn record_pause(kind: GcCollectionKind, duration: Duration) {
    let us = duration.as_micros().min(u64::MAX as u128) as u64;
    METRICS[kind.index()].record(us);
}

/// Reset every counter back to zero.
///
/// Useful for tests and for embedder-driven measurement windows over the FFI
/// boundary.
pub fn reset() {
    for counters in &METRICS {
        counters.reset();
    }
}

/// A consistent snapshot of the GC pause counters.
#[derive(Debug, Clone)]
pub struct GcPauseSnapshot {
    /// One snapshot per [`GcCollectionKind`], in [`GcCollectionKind::ALL`]
    /// order.
    pub per_kind: [GcKindPauseSnapshot; NUM_GC_KINDS],
}

impl GcPauseSnapshot {
    /// Look up the snapshot for a specific collection kind.
    pub fn for_kind(&self, kind: GcCollectionKind) -> &GcKindPauseSnapshot {
        &self.per_kind[kind.index()]
    }

    /// Total number of recorded pauses across all kinds.
    pub fn total_count(&self) -> u64 {
        self.per_kind.iter().map(|k| k.count).sum()
    }

    /// Sum of recorded pause durations across all kinds, in microseconds.
    pub fn total_pause_us(&self) -> u64 {
        self.per_kind.iter().map(|k| k.total_pause_us).sum()
    }

    /// Maximum recorded pause across all kinds, in microseconds.
    pub fn max_pause_us(&self) -> u64 {
        self.per_kind
            .iter()
            .map(|k| k.max_pause_us)
            .max()
            .unwrap_or(0)
    }
}

/// Per-kind portion of a [`GcPauseSnapshot`].
#[derive(Debug, Clone)]
pub struct GcKindPauseSnapshot {
    /// The collection kind these counters describe.
    pub kind: GcCollectionKind,
    /// Number of pauses recorded.
    pub count: u64,
    /// Sum of pause durations in microseconds.
    pub total_pause_us: u64,
    /// Largest single pause in microseconds.
    pub max_pause_us: u64,
    /// Histogram of pauses.  Index `i < HISTOGRAM_BUCKETS_US.len()` counts
    /// pauses of at most `HISTOGRAM_BUCKETS_US[i]` µs; the final index counts
    /// overflows.
    pub buckets: [u64; NUM_HISTOGRAM_BUCKETS],
}

/// Capture a consistent snapshot of all GC pause counters.
///
/// Individual fields are read with relaxed ordering, so a snapshot taken
/// concurrently with an in-flight collection may observe a partially-updated
/// bucket but never tears within a single 64-bit field.
pub fn snapshot() -> GcPauseSnapshot {
    let per_kind = std::array::from_fn(|i| METRICS[i].snapshot(GcCollectionKind::ALL[i]));
    GcPauseSnapshot { per_kind }
}

/// RAII guard that records the elapsed time as a GC pause on drop.
///
/// Construct one at the very start of a collection entry point and let
/// destructor order capture the full pause window — including any cleanup
/// that runs before the function returns.
pub struct PauseTimer {
    kind: GcCollectionKind,
    start: Instant,
}

impl PauseTimer {
    /// Start a new timer for `kind`.
    pub fn start(kind: GcCollectionKind) -> Self {
        Self {
            kind,
            start: Instant::now(),
        }
    }
}

impl Drop for PauseTimer {
    fn drop(&mut self) {
        record_pause(self.kind, self.start.elapsed());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Many tests touch the single global counter set.  Serialise them so
    // resets and reads don't interleave.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn bucket_index_lower_bound() {
        assert_eq!(bucket_index_for(0), 0);
        assert_eq!(bucket_index_for(10), 0);
    }

    #[test]
    fn bucket_index_upper_bounds() {
        assert_eq!(bucket_index_for(11), 1);
        assert_eq!(bucket_index_for(50), 1);
        assert_eq!(bucket_index_for(100), 2);
        assert_eq!(bucket_index_for(50_000), 7);
    }

    #[test]
    fn bucket_index_overflow() {
        assert_eq!(bucket_index_for(50_001), HISTOGRAM_BUCKETS_US.len());
        assert_eq!(bucket_index_for(u64::MAX), HISTOGRAM_BUCKETS_US.len());
    }

    #[test]
    fn record_and_snapshot_minor() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        record_pause(GcCollectionKind::Minor, Duration::from_micros(42));
        record_pause(GcCollectionKind::Minor, Duration::from_micros(1));
        record_pause(GcCollectionKind::Minor, Duration::from_micros(200));

        let snap = snapshot();
        let m = snap.for_kind(GcCollectionKind::Minor);
        assert_eq!(m.count, 3);
        assert_eq!(m.total_pause_us, 243);
        assert_eq!(m.max_pause_us, 200);
        // Buckets: [≤10] gets the 1µs, [≤50] gets the 42µs, [≤500] gets 200µs.
        assert_eq!(m.buckets[0], 1);
        assert_eq!(m.buckets[1], 1);
        assert_eq!(m.buckets[3], 1);

        // Other kinds are untouched.
        assert_eq!(snap.for_kind(GcCollectionKind::Major).count, 0);
        assert_eq!(snap.for_kind(GcCollectionKind::Immix).count, 0);
        assert_eq!(snap.for_kind(GcCollectionKind::MarkSweepCompact).count, 0);
    }

    #[test]
    fn reset_zeros_everything() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        for kind in GcCollectionKind::ALL {
            record_pause(kind, Duration::from_micros(123));
        }
        reset();
        let snap = snapshot();
        for kind in GcCollectionKind::ALL {
            let k = snap.for_kind(kind);
            assert_eq!(k.count, 0);
            assert_eq!(k.total_pause_us, 0);
            assert_eq!(k.max_pause_us, 0);
            for b in k.buckets {
                assert_eq!(b, 0);
            }
        }
    }

    #[test]
    fn overflow_bucket_records_large_pauses() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        record_pause(GcCollectionKind::Major, Duration::from_millis(250));
        let snap = snapshot();
        let m = snap.for_kind(GcCollectionKind::Major);
        assert_eq!(m.count, 1);
        assert_eq!(m.max_pause_us, 250_000);
        assert_eq!(m.buckets[NUM_HISTOGRAM_BUCKETS - 1], 1);
    }

    #[test]
    fn pause_timer_records_on_drop() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        {
            let _t = PauseTimer::start(GcCollectionKind::Immix);
            std::thread::sleep(Duration::from_millis(1));
        }
        let snap = snapshot();
        assert_eq!(snap.for_kind(GcCollectionKind::Immix).count, 1);
        assert!(snap.for_kind(GcCollectionKind::Immix).total_pause_us >= 1);
    }

    #[test]
    fn snapshot_aggregates_match() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        record_pause(GcCollectionKind::Minor, Duration::from_micros(5));
        record_pause(GcCollectionKind::Major, Duration::from_micros(2_000));
        record_pause(GcCollectionKind::Immix, Duration::from_micros(150));
        let snap = snapshot();
        assert_eq!(snap.total_count(), 3);
        assert_eq!(snap.total_pause_us(), 5 + 2_000 + 150);
        assert_eq!(snap.max_pause_us(), 2_000);
    }

    #[test]
    fn kind_as_str_is_stable() {
        assert_eq!(GcCollectionKind::Minor.as_str(), "minor");
        assert_eq!(GcCollectionKind::Major.as_str(), "major");
        assert_eq!(GcCollectionKind::Immix.as_str(), "immix");
        assert_eq!(
            GcCollectionKind::MarkSweepCompact.as_str(),
            "mark_sweep_compact"
        );
    }
}
