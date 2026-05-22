//! Release-safe diagnostics for JIT code memory and emitted native bytes.
//!
//! The counters are process-global atomics because executable-code allocation
//! currently happens below the FFI isolate boundary.  The FFI surface therefore
//! reports `isolate_scoped = false` and accepts null isolates, matching the
//! existing tier/deopt/IC diagnostics.  Edge consumers that need workload
//! attribution should reset or diff snapshots around the workload boundary.

use std::sync::atomic::{AtomicU64, Ordering};

/// JIT code-producing tiers tracked by the memory diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum JitMemoryTier {
    /// Non-optimising baseline JIT.
    Baseline = 0,
    /// Maglev mid-tier optimising compiler.
    Maglev = 1,
    /// Turbofan tier name.  The current native backend is Cranelift, so this
    /// row remains zero until a distinct Turbofan emitter exists.
    Turbofan = 2,
    /// Cranelift-backed top-tier code generation.
    Cranelift = 3,
}

impl JitMemoryTier {
    /// Number of tier rows exposed by this diagnostic schema.
    pub const COUNT: usize = 4;

    /// All tiers in stable ABI order.
    #[must_use]
    pub const fn all() -> [Self; Self::COUNT] {
        [
            Self::Baseline,
            Self::Maglev,
            Self::Turbofan,
            Self::Cranelift,
        ]
    }
}

static CODE_BYTES_EMITTED: [AtomicU64; JitMemoryTier::COUNT] = zeroed_atomic_array();
static EXECUTABLE_BYTES_COMMITTED: [AtomicU64; JitMemoryTier::COUNT] = zeroed_atomic_array();
static EXECUTABLE_BYTES_FREED: [AtomicU64; JitMemoryTier::COUNT] = zeroed_atomic_array();
static EXECUTABLE_PAGES_COMMITTED: [AtomicU64; JitMemoryTier::COUNT] = zeroed_atomic_array();
static EXECUTABLE_PAGES_RESERVED: [AtomicU64; JitMemoryTier::COUNT] = zeroed_atomic_array();
static EXECUTABLE_PAGES_FREED: [AtomicU64; JitMemoryTier::COUNT] = zeroed_atomic_array();
static LIVE_CODE_BYTES: [AtomicU64; JitMemoryTier::COUNT] = zeroed_atomic_array();
static CODE_CACHE_ARTIFACT_BYTES: [AtomicU64; JitMemoryTier::COUNT] = zeroed_atomic_array();

const fn zeroed_atomic_array() -> [AtomicU64; JitMemoryTier::COUNT] {
    [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ]
}

const ASSUMED_PAGE_SIZE: u64 = 4096;

fn pages_for(bytes: u64) -> u64 {
    if bytes == 0 {
        0
    } else {
        bytes.div_ceil(ASSUMED_PAGE_SIZE)
    }
}

/// Record native instruction bytes emitted by a compiler tier.
#[inline]
pub fn record_code_emitted(tier: JitMemoryTier, bytes: usize) {
    CODE_BYTES_EMITTED[tier as usize].fetch_add(bytes as u64, Ordering::Relaxed);
}

/// Record successful executable-memory allocation for a tier.
#[inline]
pub fn record_executable_allocated(tier: JitMemoryTier, bytes: usize) {
    let bytes = bytes as u64;
    let pages = pages_for(bytes);
    let i = tier as usize;
    EXECUTABLE_BYTES_COMMITTED[i].fetch_add(bytes, Ordering::Relaxed);
    EXECUTABLE_PAGES_COMMITTED[i].fetch_add(pages, Ordering::Relaxed);
    EXECUTABLE_PAGES_RESERVED[i].fetch_add(pages, Ordering::Relaxed);
    LIVE_CODE_BYTES[i].fetch_add(bytes, Ordering::Relaxed);
}

/// Record executable-memory release for a tier.
#[inline]
pub fn record_executable_freed(tier: JitMemoryTier, bytes: usize) {
    let bytes = bytes as u64;
    let pages = pages_for(bytes);
    let i = tier as usize;
    EXECUTABLE_BYTES_FREED[i].fetch_add(bytes, Ordering::Relaxed);
    EXECUTABLE_PAGES_FREED[i].fetch_add(pages, Ordering::Relaxed);
    LIVE_CODE_BYTES[i].fetch_sub(bytes, Ordering::Relaxed);
}

/// Record bytes written to a native code-cache artifact.
///
/// Current Stator cache APIs persist parser/bytecode artifacts only, so this
/// remains zero in normal builds until native artifact production is added.
#[inline]
pub fn record_code_cache_artifact(tier: JitMemoryTier, bytes: usize) {
    CODE_CACHE_ARTIFACT_BYTES[tier as usize].fetch_add(bytes as u64, Ordering::Relaxed);
}

/// Immutable per-tier memory-counter snapshot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JitMemoryTierSnapshot {
    /// Native instruction bytes emitted by this tier.
    pub code_bytes_emitted: u64,
    /// Executable bytes successfully committed for cached/live code.
    pub executable_bytes_committed: u64,
    /// Executable bytes released by teardown/drop hooks.
    pub executable_bytes_freed: u64,
    /// Executable pages committed by successful allocations.
    pub executable_pages_committed: u64,
    /// Executable pages reserved by successful allocations.
    pub executable_pages_reserved: u64,
    /// Executable pages released by teardown/drop hooks.
    pub executable_pages_freed: u64,
    /// Current live executable-code bytes after observed frees.
    pub live_code_bytes: u64,
    /// Native code-cache artifact bytes produced by this tier.
    pub code_cache_artifact_bytes: u64,
}

/// Immutable aggregate JIT memory diagnostic snapshot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JitMemorySnapshot {
    /// Whether rows are scoped to the requested isolate.
    pub isolate_scoped: bool,
    /// Whether executable page counters are populated on this target.
    pub executable_page_accounting_supported: bool,
    tiers: [JitMemoryTierSnapshot; JitMemoryTier::COUNT],
}

impl JitMemorySnapshot {
    /// Snapshot row for `tier`.
    #[must_use]
    pub fn for_tier(&self, tier: JitMemoryTier) -> &JitMemoryTierSnapshot {
        &self.tiers[tier as usize]
    }
}

/// Take a snapshot of the current process-global JIT memory counters.
#[must_use]
pub fn snapshot() -> JitMemorySnapshot {
    let mut snap = JitMemorySnapshot {
        isolate_scoped: false,
        executable_page_accounting_supported: cfg!(any(unix, all(target_arch = "x86_64", windows))),
        ..JitMemorySnapshot::default()
    };
    for tier in JitMemoryTier::all() {
        let i = tier as usize;
        snap.tiers[i] = JitMemoryTierSnapshot {
            code_bytes_emitted: CODE_BYTES_EMITTED[i].load(Ordering::Relaxed),
            executable_bytes_committed: EXECUTABLE_BYTES_COMMITTED[i].load(Ordering::Relaxed),
            executable_bytes_freed: EXECUTABLE_BYTES_FREED[i].load(Ordering::Relaxed),
            executable_pages_committed: EXECUTABLE_PAGES_COMMITTED[i].load(Ordering::Relaxed),
            executable_pages_reserved: EXECUTABLE_PAGES_RESERVED[i].load(Ordering::Relaxed),
            executable_pages_freed: EXECUTABLE_PAGES_FREED[i].load(Ordering::Relaxed),
            live_code_bytes: LIVE_CODE_BYTES[i].load(Ordering::Relaxed),
            code_cache_artifact_bytes: CODE_CACHE_ARTIFACT_BYTES[i].load(Ordering::Relaxed),
        };
    }
    snap
}

/// Reset all JIT memory counters to zero.
pub fn reset() {
    for i in 0..JitMemoryTier::COUNT {
        CODE_BYTES_EMITTED[i].store(0, Ordering::Relaxed);
        EXECUTABLE_BYTES_COMMITTED[i].store(0, Ordering::Relaxed);
        EXECUTABLE_BYTES_FREED[i].store(0, Ordering::Relaxed);
        EXECUTABLE_PAGES_COMMITTED[i].store(0, Ordering::Relaxed);
        EXECUTABLE_PAGES_RESERVED[i].store(0, Ordering::Relaxed);
        EXECUTABLE_PAGES_FREED[i].store(0, Ordering::Relaxed);
        LIVE_CODE_BYTES[i].store(0, Ordering::Relaxed);
        CODE_CACHE_ARTIFACT_BYTES[i].store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_reset_zeroes_all_tiers_without_drift() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        record_code_emitted(JitMemoryTier::Baseline, 7);
        record_executable_allocated(JitMemoryTier::Maglev, 4097);
        record_code_cache_artifact(JitMemoryTier::Cranelift, 11);
        reset();
        let snap = snapshot();
        for tier in JitMemoryTier::all() {
            assert_eq!(snap.for_tier(tier), &JitMemoryTierSnapshot::default());
        }
        assert!(!snap.isolate_scoped);
    }

    #[test]
    fn test_allocation_and_free_update_live_bytes_and_pages() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        record_executable_allocated(JitMemoryTier::Baseline, 4097);
        let after_alloc = snapshot();
        let base = after_alloc.for_tier(JitMemoryTier::Baseline);
        assert_eq!(base.executable_bytes_committed, 4097);
        assert_eq!(base.executable_pages_committed, 2);
        assert_eq!(base.executable_pages_reserved, 2);
        assert_eq!(base.live_code_bytes, 4097);

        record_executable_freed(JitMemoryTier::Baseline, 4097);
        let after_free = snapshot();
        let base = after_free.for_tier(JitMemoryTier::Baseline);
        assert_eq!(base.executable_bytes_freed, 4097);
        assert_eq!(base.executable_pages_freed, 2);
        assert_eq!(base.live_code_bytes, 0);
    }

    #[test]
    fn test_unsupported_turbofan_row_stays_zero_until_recorded() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        record_code_emitted(JitMemoryTier::Cranelift, 13);
        let snap = snapshot();
        assert_eq!(
            snap.for_tier(JitMemoryTier::Turbofan),
            &JitMemoryTierSnapshot::default()
        );
        assert_eq!(
            snap.for_tier(JitMemoryTier::Cranelift).code_bytes_emitted,
            13
        );
    }
}
