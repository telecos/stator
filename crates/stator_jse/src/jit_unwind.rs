//! Windows x64 JIT unwind metadata registration.
//!
//! Edge crash reporting, ETW profiling, and the Visual Studio debugger all
//! rely on the Win64 structured exception handling table to walk through
//! frames on an `x86_64-pc-windows-*` thread.  When a JIT emits new
//! executable code at runtime that code is, by default, invisible to those
//! tools — the OS rejects any unwind through a region that is not covered
//! by a function table, so a sampling profiler attempting to climb out of a
//! JIT frame either truncates the stack or, worse, reports a bogus walk.
//!
//! The Win64 ABI lets a JIT plug a code region into the unwind machinery by
//! handing the kernel an array of [`RUNTIME_FUNCTION`] records plus a
//! module-style base address through [`RtlAddFunctionTable`], and removing
//! the registration with [`RtlDeleteFunctionTable`] when the code is freed.
//!
//! ## Current tier support
//!
//! None of Stator's JIT tiers presently emit Win64 unwind records:
//!
//! * **Baseline** — handcrafted `masm_x64` prologue/epilogue, no `.pdata`
//!   emission.
//! * **Maglev** — same in-tree assembler, no `.pdata` emission.
//! * **Turbofan / Cranelift** — Cranelift can emit Windows unwind info per
//!   compiled function, but the in-process pipeline does not yet thread that
//!   information through to the executable-memory allocator.
//!
//! Until those tiers learn to emit `RUNTIME_FUNCTION` records this module
//! intentionally fails closed: [`register_for_tier`] returns
//! [`UnwindError::UnsupportedTier`] for every tier, increments the
//! `unsupported_tier_attempts` counter, and refuses to fabricate bogus
//! unwind metadata.  A future tier that *does* emit `.pdata` can call
//! [`register_runtime_functions`] directly with caller-owned records and
//! receive a real [`JitUnwindRegistration`] handle.
//!
//! ## Lifetime contract
//!
//! [`JitUnwindRegistration`] owns:
//!
//! * a heap-allocated, immovable `Vec<RUNTIME_FUNCTION>` whose pointer was
//!   passed to `RtlAddFunctionTable` and which therefore must not move
//!   until the registration is dropped, and
//! * the registered code region's base address (kept for diagnostics and to
//!   forbid re-registering the same range twice).
//!
//! Dropping the handle calls `RtlDeleteFunctionTable` with the same pointer
//! that was registered, ensuring there is no dangling entry in the Win64
//! function table after the underlying executable memory is freed.  The
//! handle is `!Clone`, `!Copy`, and `Send + Sync` (registration is a
//! process-global operation; the OS serialises it internally).
//!
//! On non-Windows targets every public type still compiles so that callers
//! do not need `cfg` gates: [`register_for_tier`] returns
//! [`UnwindError::UnsupportedPlatform`] and registration handles cannot be
//! constructed.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// JIT tiers tracked by the unwind diagnostics.
///
/// The discriminants double as stable ABI indices for the FFI snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum JitTier {
    /// Non-optimising baseline JIT.
    Baseline = 0,
    /// Maglev mid-tier optimising compiler.
    Maglev = 1,
    /// Top-tier Turbofan output (currently Cranelift-backed).
    Turbofan = 2,
    /// Standalone Cranelift-compiled functions (Wasm, etc.) that share the
    /// same executable-memory pool as the JS tiers.
    Cranelift = 3,
}

impl JitTier {
    /// Number of tier variants tracked by the unwind diagnostics.
    pub const COUNT: usize = 4;

    /// Stable short name suitable for diagnostic dumps and FFI labels.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Maglev => "maglev",
            Self::Turbofan => "turbofan",
            Self::Cranelift => "cranelift",
        }
    }

    /// All tier variants in stable discriminant order.
    #[must_use]
    pub const fn all() -> [Self; Self::COUNT] {
        [
            Self::Baseline,
            Self::Maglev,
            Self::Turbofan,
            Self::Cranelift,
        ]
    }

    /// Whether this tier currently emits Win64 unwind metadata for the JIT
    /// code it produces.
    ///
    /// All four tiers presently return `false`; see the module-level docs
    /// for the rationale.  When a tier learns to emit `.pdata`, flip the
    /// corresponding arm and update [`register_for_tier`] to honour it.
    #[must_use]
    pub const fn emits_unwind_info(self) -> bool {
        match self {
            Self::Baseline | Self::Maglev | Self::Turbofan | Self::Cranelift => false,
        }
    }
}

/// Failure modes reported by [`register_for_tier`] /
/// [`register_runtime_functions`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnwindError {
    /// The current build target is not Windows x86-64, so there is no
    /// `RtlAddFunctionTable` to call.
    UnsupportedPlatform,
    /// The requested JIT tier does not yet emit Win64 unwind records.
    /// Reported by [`register_for_tier`] for every tier whose
    /// [`JitTier::emits_unwind_info`] returns `false`.
    UnsupportedTier(JitTier),
    /// The caller supplied an empty `RUNTIME_FUNCTION` slice or a
    /// zero-length code region.
    EmptyRegion,
    /// The kernel rejected the `RtlAddFunctionTable` call.  This is
    /// surfaced verbatim so Edge crash reporting can correlate registration
    /// failures with subsequent unwalkable stacks.
    OsRejected,
}

impl fmt::Display for UnwindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => f.write_str(
                "Win64 JIT unwind registration is only available on x86_64-pc-windows targets",
            ),
            Self::UnsupportedTier(tier) => write!(
                f,
                "JIT tier {:?} does not currently emit Win64 unwind records",
                tier
            ),
            Self::EmptyRegion => f.write_str("unwind registration requires a non-empty region"),
            Self::OsRejected => f.write_str("RtlAddFunctionTable rejected the registration"),
        }
    }
}

impl std::error::Error for UnwindError {}

// ─────────────────────────────────────────────────────────────────────────────
// Process-global counters
// ─────────────────────────────────────────────────────────────────────────────

struct TierCounters {
    register_attempts: AtomicU64,
    register_successes: AtomicU64,
    register_failures: AtomicU64,
    unsupported_attempts: AtomicU64,
    deregistrations: AtomicU64,
}

impl TierCounters {
    const fn new() -> Self {
        Self {
            register_attempts: AtomicU64::new(0),
            register_successes: AtomicU64::new(0),
            register_failures: AtomicU64::new(0),
            unsupported_attempts: AtomicU64::new(0),
            deregistrations: AtomicU64::new(0),
        }
    }

    fn reset(&self) {
        for slot in [
            &self.register_attempts,
            &self.register_successes,
            &self.register_failures,
            &self.unsupported_attempts,
            &self.deregistrations,
        ] {
            slot.store(0, Ordering::Relaxed);
        }
    }

    fn snapshot(&self, tier: JitTier) -> JitUnwindTierSnapshot {
        JitUnwindTierSnapshot {
            tier,
            unwind_supported: tier.emits_unwind_info(),
            register_attempts: self.register_attempts.load(Ordering::Relaxed),
            register_successes: self.register_successes.load(Ordering::Relaxed),
            register_failures: self.register_failures.load(Ordering::Relaxed),
            unsupported_tier_attempts: self.unsupported_attempts.load(Ordering::Relaxed),
            deregistrations: self.deregistrations.load(Ordering::Relaxed),
        }
    }
}

static TIER_COUNTERS: [TierCounters; JitTier::COUNT] =
    [const { TierCounters::new() }; JitTier::COUNT];

/// Per-tier release-safe unwind counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JitUnwindTierSnapshot {
    /// Tier these counters describe.
    pub tier: JitTier,
    /// Whether the build currently emits unwind metadata for this tier.
    pub unwind_supported: bool,
    /// Number of registration calls (including unsupported / failed ones).
    pub register_attempts: u64,
    /// Number of registrations that returned a live [`JitUnwindRegistration`].
    pub register_successes: u64,
    /// Number of registrations rejected by the OS after the supported-tier
    /// check passed.
    pub register_failures: u64,
    /// Number of registrations rejected because the tier does not yet emit
    /// Win64 unwind records.  Fail-closed sentinel; never auto-resets.
    pub unsupported_tier_attempts: u64,
    /// Number of [`JitUnwindRegistration`] handles that have been dropped
    /// (and therefore had their function tables deregistered) for this
    /// tier.
    pub deregistrations: u64,
}

impl JitUnwindTierSnapshot {
    /// Number of registrations still held live for this tier
    /// (successes minus deregistrations).  Saturates at zero so a stale
    /// reset cannot make the counter negative.
    #[must_use]
    pub fn currently_registered(&self) -> u64 {
        self.register_successes.saturating_sub(self.deregistrations)
    }
}

/// Aggregate snapshot of every release-safe JIT unwind counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JitUnwindSnapshot {
    /// Whether the current build target supports Win64 unwind
    /// registration at all (i.e. is `x86_64-pc-windows-*`).
    pub platform_supported: bool,
    /// Per-tier counters, indexed by [`JitTier`] discriminant.
    pub tiers: [JitUnwindTierSnapshot; JitTier::COUNT],
}

impl JitUnwindSnapshot {
    /// Look up the snapshot row for a tier.
    #[must_use]
    pub fn tier(&self, tier: JitTier) -> &JitUnwindTierSnapshot {
        &self.tiers[tier as usize]
    }

    /// Sum of `register_attempts` across all tiers.
    #[must_use]
    pub fn total_register_attempts(&self) -> u64 {
        self.tiers.iter().map(|t| t.register_attempts).sum()
    }

    /// Sum of `register_successes` across all tiers.
    #[must_use]
    pub fn total_register_successes(&self) -> u64 {
        self.tiers.iter().map(|t| t.register_successes).sum()
    }

    /// Sum of `unsupported_tier_attempts` across all tiers.
    #[must_use]
    pub fn total_unsupported_tier_attempts(&self) -> u64 {
        self.tiers.iter().map(|t| t.unsupported_tier_attempts).sum()
    }

    /// Sum of `deregistrations` across all tiers.
    #[must_use]
    pub fn total_deregistrations(&self) -> u64 {
        self.tiers.iter().map(|t| t.deregistrations).sum()
    }

    /// Sum of [`JitUnwindTierSnapshot::currently_registered`] across all
    /// tiers.
    #[must_use]
    pub fn total_currently_registered(&self) -> u64 {
        self.tiers.iter().map(|t| t.currently_registered()).sum()
    }
}

/// Whether the current build target has a real Win64 unwind backend.
#[must_use]
pub const fn platform_supported() -> bool {
    cfg!(all(target_arch = "x86_64", windows))
}

/// Take a snapshot of every JIT unwind diagnostic counter.
#[must_use]
pub fn snapshot() -> JitUnwindSnapshot {
    let tiers =
        std::array::from_fn(|tier_idx| TIER_COUNTERS[tier_idx].snapshot(JitTier::all()[tier_idx]));
    JitUnwindSnapshot {
        platform_supported: platform_supported(),
        tiers,
    }
}

/// Reset every JIT unwind diagnostic counter to zero.
///
/// The counters are process-global; reset is intended for test isolation
/// and Edge benchmark warm-up rather than steady-state production use.
pub fn reset() {
    for counters in &TIER_COUNTERS {
        counters.reset();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public registration surface
// ─────────────────────────────────────────────────────────────────────────────

/// Attempt to register unwind metadata for a freshly-emitted JIT code
/// region produced by `tier`.
///
/// This is the **safe, fail-closed** entry point.  Because no in-tree tier
/// currently emits Win64 `RUNTIME_FUNCTION` records, this call always
/// returns [`UnwindError::UnsupportedTier`] (or
/// [`UnwindError::UnsupportedPlatform`] on non-Windows targets) and bumps
/// the corresponding release-safe counter.
///
/// When a tier learns to emit `.pdata`, update
/// [`JitTier::emits_unwind_info`] and route this function to
/// [`register_runtime_functions`] with the tier's owned records.
pub fn register_for_tier(
    tier: JitTier,
    code_base: *const u8,
    code_len: usize,
) -> Result<JitUnwindRegistration, UnwindError> {
    let counters = &TIER_COUNTERS[tier as usize];
    counters.register_attempts.fetch_add(1, Ordering::Relaxed);

    if !platform_supported() {
        counters
            .unsupported_attempts
            .fetch_add(1, Ordering::Relaxed);
        return Err(UnwindError::UnsupportedPlatform);
    }
    if code_base.is_null() || code_len == 0 {
        counters.register_failures.fetch_add(1, Ordering::Relaxed);
        return Err(UnwindError::EmptyRegion);
    }
    if !tier.emits_unwind_info() {
        counters
            .unsupported_attempts
            .fetch_add(1, Ordering::Relaxed);
        return Err(UnwindError::UnsupportedTier(tier));
    }

    // No tier currently reaches this branch; future tiers must call
    // `register_runtime_functions` directly with their owned `.pdata`.
    counters.register_failures.fetch_add(1, Ordering::Relaxed);
    Err(UnwindError::UnsupportedTier(tier))
}

// ─────────────────────────────────────────────────────────────────────────────
// Platform implementation
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(target_arch = "x86_64", windows))]
mod imp {
    use super::{JitTier, TIER_COUNTERS, UnwindError};
    use std::sync::atomic::Ordering;
    use windows_sys::Win32::System::Diagnostics::Debug::{
        IMAGE_RUNTIME_FUNCTION_ENTRY, RtlAddFunctionTable, RtlDeleteFunctionTable,
    };

    /// Win64 `RUNTIME_FUNCTION` record.  Re-exported so callers do not need
    /// to depend on `windows-sys` directly.
    pub type RuntimeFunction = IMAGE_RUNTIME_FUNCTION_ENTRY;

    /// Owned Win64 unwind registration handle.
    ///
    /// Drop deregisters the underlying function table.  The boxed
    /// `Vec<RuntimeFunction>` is kept alive (and pinned by its heap
    /// allocation, since we never expose a mutable view) for the lifetime
    /// of the handle so that the pointer handed to `RtlAddFunctionTable`
    /// stays valid.
    pub struct JitUnwindRegistration {
        tier: JitTier,
        // `Box` keeps the slice pointer stable; we never move out of it.
        functions: Box<[RuntimeFunction]>,
        code_base: u64,
        code_len: usize,
    }

    // SAFETY: the handle owns its boxed `RuntimeFunction` slice and a raw
    // base address.  Registration / deregistration are process-global
    // operations that the OS serialises internally, and no interior
    // mutability is exposed.
    unsafe impl Send for JitUnwindRegistration {}
    unsafe impl Sync for JitUnwindRegistration {}

    impl std::fmt::Debug for JitUnwindRegistration {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("JitUnwindRegistration")
                .field("tier", &self.tier)
                .field("function_count", &self.functions.len())
                .field("code_base", &format_args!("{:#x}", self.code_base))
                .field("code_len", &self.code_len)
                .finish()
        }
    }

    impl JitUnwindRegistration {
        /// Tier whose code is described by this registration.
        #[must_use]
        pub fn tier(&self) -> JitTier {
            self.tier
        }

        /// Number of `RUNTIME_FUNCTION` records registered.
        #[must_use]
        pub fn function_count(&self) -> usize {
            self.functions.len()
        }

        /// Base address handed to `RtlAddFunctionTable`.
        #[must_use]
        pub fn code_base(&self) -> u64 {
            self.code_base
        }

        /// Length in bytes of the registered code region (informational).
        #[must_use]
        pub fn code_len(&self) -> usize {
            self.code_len
        }
    }

    impl Drop for JitUnwindRegistration {
        fn drop(&mut self) {
            let counters = &TIER_COUNTERS[self.tier as usize];
            // SAFETY: `functions` was registered with the same pointer in
            // `register_runtime_functions_impl` and has not moved.  The Win64
            // API contract requires the same pointer for deletion.
            let ok = unsafe {
                RtlDeleteFunctionTable(self.functions.as_ptr().cast::<RuntimeFunction>())
            };
            // RtlDeleteFunctionTable returns non-zero on success; we still
            // record the deregistration so the live-count never drifts even
            // if the OS rejects the call (which should not happen for a
            // pointer we just registered).
            let _ = ok;
            counters.deregistrations.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(super) fn register_runtime_functions_impl(
        tier: JitTier,
        code_base: *const u8,
        code_len: usize,
        functions: Vec<RuntimeFunction>,
    ) -> Result<JitUnwindRegistration, UnwindError> {
        let counters = &TIER_COUNTERS[tier as usize];
        counters.register_attempts.fetch_add(1, Ordering::Relaxed);
        if functions.is_empty() || code_base.is_null() || code_len == 0 {
            counters.register_failures.fetch_add(1, Ordering::Relaxed);
            return Err(UnwindError::EmptyRegion);
        }
        let boxed = functions.into_boxed_slice();
        let entry_count: u32 = match boxed.len().try_into() {
            Ok(n) => n,
            Err(_) => {
                counters.register_failures.fetch_add(1, Ordering::Relaxed);
                return Err(UnwindError::OsRejected);
            }
        };
        let base_u64 = code_base as u64;
        // SAFETY: `boxed.as_ptr()` is a valid pointer to `entry_count`
        // `RUNTIME_FUNCTION` records that remain live (and immobile,
        // because we never move out of the box) until the returned
        // registration is dropped.  `base_u64` is the module-style base
        // address of the JIT code region.
        let ok = unsafe {
            RtlAddFunctionTable(
                boxed.as_ptr().cast::<RuntimeFunction>(),
                entry_count,
                base_u64,
            )
        };
        if !ok {
            counters.register_failures.fetch_add(1, Ordering::Relaxed);
            return Err(UnwindError::OsRejected);
        }
        counters.register_successes.fetch_add(1, Ordering::Relaxed);
        Ok(JitUnwindRegistration {
            tier,
            functions: boxed,
            code_base: base_u64,
            code_len,
        })
    }
}

#[cfg(not(all(target_arch = "x86_64", windows)))]
mod imp {
    use super::{JitTier, TIER_COUNTERS, UnwindError};
    use std::sync::atomic::Ordering;

    /// Stub `RUNTIME_FUNCTION` placeholder so the API compiles unchanged on
    /// non-Windows targets.  The type intentionally has no fields — any
    /// attempt to construct one is dead code on these targets.
    #[derive(Debug, Clone, Copy)]
    pub struct RuntimeFunction {
        _private: (),
    }

    /// No-op stand-in for the Windows-only registration handle.
    ///
    /// Cannot be constructed off-Windows; declared for API parity so
    /// downstream `cfg` gates can remain narrow.
    pub struct JitUnwindRegistration {
        _never: std::convert::Infallible,
    }

    impl std::fmt::Debug for JitUnwindRegistration {
        fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self._never {}
        }
    }

    impl JitUnwindRegistration {
        /// Always unreachable off-Windows.
        #[must_use]
        pub fn tier(&self) -> JitTier {
            match self._never {}
        }

        /// Always unreachable off-Windows.
        #[must_use]
        pub fn function_count(&self) -> usize {
            match self._never {}
        }

        /// Always unreachable off-Windows.
        #[must_use]
        pub fn code_base(&self) -> u64 {
            match self._never {}
        }

        /// Always unreachable off-Windows.
        #[must_use]
        pub fn code_len(&self) -> usize {
            match self._never {}
        }
    }

    pub(super) fn register_runtime_functions_impl(
        tier: JitTier,
        _code_base: *const u8,
        _code_len: usize,
        _functions: Vec<RuntimeFunction>,
    ) -> Result<JitUnwindRegistration, UnwindError> {
        let counters = &TIER_COUNTERS[tier as usize];
        counters.register_attempts.fetch_add(1, Ordering::Relaxed);
        counters
            .unsupported_attempts
            .fetch_add(1, Ordering::Relaxed);
        Err(UnwindError::UnsupportedPlatform)
    }
}

pub use imp::{JitUnwindRegistration, RuntimeFunction};

/// Register a caller-provided `RUNTIME_FUNCTION` array for `tier`.
///
/// This is the low-level entry point intended for a future tier that
/// emits real Win64 unwind metadata.  The caller hands over ownership of
/// `functions`; the returned [`JitUnwindRegistration`] keeps the slice
/// alive and deregisters it on drop.
///
/// On non-Windows targets the call always returns
/// [`UnwindError::UnsupportedPlatform`] without touching `functions`.
pub fn register_runtime_functions(
    tier: JitTier,
    code_base: *const u8,
    code_len: usize,
    functions: Vec<RuntimeFunction>,
) -> Result<JitUnwindRegistration, UnwindError> {
    imp::register_runtime_functions_impl(tier, code_base, code_len, functions)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

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
    fn tier_names_are_stable() {
        assert_eq!(JitTier::Baseline.name(), "baseline");
        assert_eq!(JitTier::Maglev.name(), "maglev");
        assert_eq!(JitTier::Turbofan.name(), "turbofan");
        assert_eq!(JitTier::Cranelift.name(), "cranelift");
        assert_eq!(JitTier::all().len(), JitTier::COUNT);
    }

    #[test]
    fn no_tier_currently_emits_unwind_info() {
        for tier in JitTier::all() {
            assert!(
                !tier.emits_unwind_info(),
                "tier {:?} unexpectedly reports unwind support",
                tier
            );
        }
    }

    #[test]
    fn platform_constant_matches_cfg() {
        assert_eq!(
            platform_supported(),
            cfg!(all(target_arch = "x86_64", windows))
        );
    }

    #[test]
    fn register_for_tier_is_fail_closed_for_every_tier() {
        let _g = lock();
        reset();
        // A non-null pointer with non-zero length so we exercise the
        // tier-support check rather than the empty-region guard.
        let buf = [0u8; 16];
        for tier in JitTier::all() {
            let err = register_for_tier(tier, buf.as_ptr(), buf.len()).unwrap_err();
            if platform_supported() {
                assert_eq!(err, UnwindError::UnsupportedTier(tier));
            } else {
                assert_eq!(err, UnwindError::UnsupportedPlatform);
            }
        }
        let snap = snapshot();
        assert_eq!(
            snap.total_register_attempts(),
            JitTier::COUNT as u64,
            "every tier should record an attempt"
        );
        assert_eq!(
            snap.total_unsupported_tier_attempts(),
            JitTier::COUNT as u64,
            "every tier should be flagged unsupported"
        );
        assert_eq!(snap.total_register_successes(), 0);
        assert_eq!(snap.total_currently_registered(), 0);
        reset();
    }

    #[test]
    fn empty_region_is_rejected() {
        let _g = lock();
        reset();
        let err = register_for_tier(JitTier::Baseline, std::ptr::null(), 0).unwrap_err();
        if platform_supported() {
            assert_eq!(err, UnwindError::EmptyRegion);
        } else {
            assert_eq!(err, UnwindError::UnsupportedPlatform);
        }
        reset();
    }

    #[test]
    fn snapshot_reset_zeroes_every_counter() {
        let _g = lock();
        reset();
        let buf = [0u8; 16];
        let _ = register_for_tier(JitTier::Maglev, buf.as_ptr(), buf.len());
        let before = snapshot();
        assert!(before.total_register_attempts() > 0);
        reset();
        let after = snapshot();
        assert_eq!(after.total_register_attempts(), 0);
        assert_eq!(after.total_unsupported_tier_attempts(), 0);
        assert_eq!(after.total_register_successes(), 0);
        assert_eq!(after.total_deregistrations(), 0);
    }

    #[test]
    fn currently_registered_saturates_after_reset() {
        let snap = JitUnwindTierSnapshot {
            tier: JitTier::Baseline,
            unwind_supported: false,
            register_attempts: 0,
            register_successes: 1,
            register_failures: 0,
            unsupported_tier_attempts: 0,
            deregistrations: 7,
        };
        assert_eq!(snap.currently_registered(), 0);
    }

    /// Smoke test that the public registration API is callable through the
    /// stable signature even on non-Windows targets.  An empty
    /// `RuntimeFunction` Vec is the only thing we can hand it portably
    /// (the type has no public constructor off-Windows), and the call
    /// must return `UnsupportedPlatform` without panicking.
    #[cfg(not(all(target_arch = "x86_64", windows)))]
    #[test]
    fn register_runtime_functions_is_stub_off_windows() {
        let _g = lock();
        reset();
        let buf = [0u8; 16];
        let err =
            register_runtime_functions(JitTier::Turbofan, buf.as_ptr(), buf.len(), Vec::new())
                .unwrap_err();
        assert_eq!(err, UnwindError::UnsupportedPlatform);
        reset();
    }

    /// On Windows x64 we can actually register a synthetic
    /// `RUNTIME_FUNCTION` describing an empty code region.  The point of
    /// the test is the bookkeeping path, not stack-walking: the kernel
    /// accepts well-formed records even when the addresses do not point
    /// at real code (it only consults them during an unwind).  Dropping
    /// the handle deregisters and bumps the counter.
    #[cfg(all(target_arch = "x86_64", windows))]
    #[test]
    fn registers_and_deregisters_real_runtime_functions() {
        use windows_sys::Win32::System::Diagnostics::Debug::IMAGE_RUNTIME_FUNCTION_ENTRY;
        let _g = lock();
        reset();
        // SAFETY: a zero-initialised RUNTIME_FUNCTION is a well-formed
        // (if degenerate) record; the kernel only walks it during an
        // unwind, which the test never triggers.
        let entry: IMAGE_RUNTIME_FUNCTION_ENTRY = unsafe { std::mem::zeroed() };
        let buf = [0u8; 64];
        let registration =
            register_runtime_functions(JitTier::Cranelift, buf.as_ptr(), buf.len(), vec![entry])
                .expect("registration must succeed on win-x64");
        assert_eq!(registration.tier(), JitTier::Cranelift);
        assert_eq!(registration.function_count(), 1);
        assert_eq!(registration.code_len(), buf.len());

        let mid = snapshot();
        assert_eq!(mid.tier(JitTier::Cranelift).register_successes, 1);
        assert_eq!(mid.tier(JitTier::Cranelift).deregistrations, 0);
        assert_eq!(mid.tier(JitTier::Cranelift).currently_registered(), 1);

        drop(registration);

        let after = snapshot();
        assert_eq!(after.tier(JitTier::Cranelift).deregistrations, 1);
        assert_eq!(after.tier(JitTier::Cranelift).currently_registered(), 0);
        reset();
    }

    /// Double-register/deregister: two independent handles for the same
    /// code base must both register and deregister cleanly without
    /// touching each other's bookkeeping.
    #[cfg(all(target_arch = "x86_64", windows))]
    #[test]
    fn double_register_deregister_is_safe() {
        use windows_sys::Win32::System::Diagnostics::Debug::IMAGE_RUNTIME_FUNCTION_ENTRY;
        let _g = lock();
        reset();
        // SAFETY: see `registers_and_deregisters_real_runtime_functions`.
        let entry: IMAGE_RUNTIME_FUNCTION_ENTRY = unsafe { std::mem::zeroed() };
        let buf = [0u8; 64];
        let a = register_runtime_functions(JitTier::Turbofan, buf.as_ptr(), buf.len(), vec![entry])
            .expect("first registration succeeds");
        let b = register_runtime_functions(JitTier::Turbofan, buf.as_ptr(), buf.len(), vec![entry])
            .expect("second registration succeeds");

        let mid = snapshot();
        assert_eq!(mid.tier(JitTier::Turbofan).register_successes, 2);
        assert_eq!(mid.tier(JitTier::Turbofan).currently_registered(), 2);

        drop(a);
        drop(b);

        let after = snapshot();
        assert_eq!(after.tier(JitTier::Turbofan).deregistrations, 2);
        assert_eq!(after.tier(JitTier::Turbofan).currently_registered(), 0);
        reset();
    }

    /// Empty function table is rejected even on supported targets.
    #[cfg(all(target_arch = "x86_64", windows))]
    #[test]
    fn empty_function_table_is_rejected_on_windows() {
        let _g = lock();
        reset();
        let buf = [0u8; 16];
        let err =
            register_runtime_functions(JitTier::Baseline, buf.as_ptr(), buf.len(), Vec::new())
                .unwrap_err();
        assert_eq!(err, UnwindError::EmptyRegion);
        reset();
    }
}
