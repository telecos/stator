//! Windows Control-Flow Guard (CFG) and Control-flow Enforcement Technology
//! (CET / shadow-stack / IBT) diagnostics for JIT executable pages.
//!
//! Chromium/Edge ship with CFG enabled and rely on CET shadow stacks on
//! capable CPUs.  A JIT that emits executable code at runtime must either:
//!
//! 1. Register every valid call target with the Windows CFG bitmap
//!    (`SetProcessValidCallTargets`), and
//! 2. Make sure the executable pages it produces are compatible with
//!    CET's hardware-enforced shadow stack and Indirect Branch Tracking
//!    (`ENDBR64`) prologues.
//!
//! Stator's current JIT tiers do **not** thread per-region metadata
//! (function entry offsets, IBT-compatible prologues, shadow-stack-safe
//! returns) through to the executable-memory allocator, so we cannot
//! safely call `SetProcessValidCallTargets`: feeding the OS an unverified
//! address list would *claim* mitigation coverage we have not actually
//! produced.  This module therefore exposes a **fail-closed** diagnostic
//! surface:
//!
//! * Per-tier `cfg_supported` / `cet_compatible` flags currently return
//!   `false` for every tier.  They flip on a per-arm basis once a tier
//!   emits the required metadata (see [`JitMitigationsTier`] docs).
//! * Per-tier counters record registration *attempts* (and the
//!   corresponding unsupported-tier / unsupported-platform refusals) so
//!   Edge can detect a stuck registration path without us having to
//!   pretend the targets were registered.
//! * A process-level [`process_cfg_status`] / [`process_cet_status`]
//!   probe reports whether the host process has CFG / CET shadow stack
//!   enabled at all — Edge uses this to distinguish a "no mitigation
//!   coverage because the OS turned it off" outcome from a
//!   "no coverage because Stator has not wired it up" outcome.
//!
//! On non-Windows targets every type still compiles so embedders do not
//! need `cfg` gates; the runtime probes return
//! [`MitigationStatus::UnsupportedPlatform`] and the counters never
//! advance past the unsupported-tier sentinel.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// JIT tiers tracked by the CFG/CET diagnostics.
///
/// The discriminants double as stable ABI indices for the FFI snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum JitMitigationsTier {
    /// Non-optimising baseline JIT.
    Baseline = 0,
    /// Maglev mid-tier optimising compiler.
    Maglev = 1,
    /// Top-tier Turbofan output (currently Cranelift-backed).
    Turbofan = 2,
    /// Standalone Cranelift-compiled functions (Wasm, etc.) sharing the
    /// JIT executable pool.
    Cranelift = 3,
}

impl JitMitigationsTier {
    /// Number of tier variants tracked by the mitigations diagnostics.
    pub const COUNT: usize = 4;

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

    /// Whether this tier currently produces metadata sufficient to
    /// register its emitted call targets with the Windows CFG bitmap.
    ///
    /// All four tiers presently return `false`: the in-tree assemblers
    /// and the Cranelift integration do not yet thread function-entry
    /// offsets through to the executable-memory allocator.  A tier that
    /// learns to emit a per-region target list can flip its arm here and
    /// route registrations through [`record_cfg_registration`].
    #[must_use]
    pub const fn cfg_supported(self) -> bool {
        match self {
            Self::Baseline | Self::Maglev | Self::Turbofan | Self::Cranelift => false,
        }
    }

    /// Whether this tier emits prologues compatible with CET hardware
    /// shadow stacks and Indirect Branch Tracking (`ENDBR64` on every
    /// indirect-call target, no shadow-stack-violating returns).
    ///
    /// All four tiers presently return `false`: none of them emit
    /// `ENDBR64` prologues or have been audited for shadow-stack
    /// compatibility.
    #[must_use]
    pub const fn cet_compatible(self) -> bool {
        match self {
            Self::Baseline | Self::Maglev | Self::Turbofan | Self::Cranelift => false,
        }
    }
}

/// Result of a process-level CFG / CET capability probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MitigationStatus {
    /// The current build target is not Windows, so neither CFG nor CET
    /// applies.
    UnsupportedPlatform,
    /// Windows reported that the mitigation is **disabled** for this
    /// process.  Stator JIT pages will run without that mitigation
    /// regardless of what we register.
    Disabled,
    /// Windows reported that the mitigation is **enabled** for this
    /// process.  JIT-produced pages must comply (register CFG targets;
    /// emit CET-compatible prologues) or they will trip the mitigation.
    Enabled,
    /// The probe failed (e.g. `GetProcessMitigationPolicy` returned an
    /// error).  Treated as "unknown" by callers; we never upgrade an
    /// unknown to enabled.
    Unknown,
}

impl fmt::Display for MitigationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => f.write_str("unsupported-platform"),
            Self::Disabled => f.write_str("disabled"),
            Self::Enabled => f.write_str("enabled"),
            Self::Unknown => f.write_str("unknown"),
        }
    }
}

impl MitigationStatus {
    /// Compact stable label suitable for FFI / telemetry strings.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::UnsupportedPlatform => "unsupported-platform",
            Self::Disabled => "disabled",
            Self::Enabled => "enabled",
            Self::Unknown => "unknown",
        }
    }

    /// Stable u32 encoding for the FFI struct (`StatorJitMitigationsStats`).
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        match self {
            Self::UnsupportedPlatform => 0,
            Self::Disabled => 1,
            Self::Enabled => 2,
            Self::Unknown => 3,
        }
    }
}

/// Failure modes reported by [`record_cfg_registration`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MitigationError {
    /// Current build target is not Windows; CFG/CET APIs are absent.
    UnsupportedPlatform,
    /// The requested JIT tier does not yet produce metadata sufficient to
    /// safely register CFG call targets.
    UnsupportedTier(JitMitigationsTier),
    /// Caller supplied an empty call-target list or a zero-length region.
    EmptyRegion,
}

impl fmt::Display for MitigationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => {
                f.write_str("CFG registration is only available on Windows targets")
            }
            Self::UnsupportedTier(tier) => write!(
                f,
                "JIT tier {:?} does not currently produce CFG call-target metadata",
                tier
            ),
            Self::EmptyRegion => f.write_str("CFG registration requires a non-empty region"),
        }
    }
}

impl std::error::Error for MitigationError {}

// ─────────────────────────────────────────────────────────────────────────────
// Process-global counters
// ─────────────────────────────────────────────────────────────────────────────

struct TierCounters {
    cfg_register_attempts: AtomicU64,
    cfg_register_successes: AtomicU64,
    cfg_register_failures: AtomicU64,
    cfg_unsupported_attempts: AtomicU64,
    cfg_targets_registered: AtomicU64,
    cet_pages_marked_compatible: AtomicU64,
    cet_pages_marked_incompatible: AtomicU64,
}

impl TierCounters {
    const fn new() -> Self {
        Self {
            cfg_register_attempts: AtomicU64::new(0),
            cfg_register_successes: AtomicU64::new(0),
            cfg_register_failures: AtomicU64::new(0),
            cfg_unsupported_attempts: AtomicU64::new(0),
            cfg_targets_registered: AtomicU64::new(0),
            cet_pages_marked_compatible: AtomicU64::new(0),
            cet_pages_marked_incompatible: AtomicU64::new(0),
        }
    }

    fn reset(&self) {
        for slot in [
            &self.cfg_register_attempts,
            &self.cfg_register_successes,
            &self.cfg_register_failures,
            &self.cfg_unsupported_attempts,
            &self.cfg_targets_registered,
            &self.cet_pages_marked_compatible,
            &self.cet_pages_marked_incompatible,
        ] {
            slot.store(0, Ordering::Relaxed);
        }
    }

    fn snapshot(&self, tier: JitMitigationsTier) -> JitMitigationsTierSnapshot {
        JitMitigationsTierSnapshot {
            tier,
            cfg_supported: tier.cfg_supported(),
            cet_compatible: tier.cet_compatible(),
            cfg_register_attempts: self.cfg_register_attempts.load(Ordering::Relaxed),
            cfg_register_successes: self.cfg_register_successes.load(Ordering::Relaxed),
            cfg_register_failures: self.cfg_register_failures.load(Ordering::Relaxed),
            cfg_unsupported_tier_attempts: self.cfg_unsupported_attempts.load(Ordering::Relaxed),
            cfg_targets_registered: self.cfg_targets_registered.load(Ordering::Relaxed),
            cet_pages_marked_compatible: self.cet_pages_marked_compatible.load(Ordering::Relaxed),
            cet_pages_marked_incompatible: self
                .cet_pages_marked_incompatible
                .load(Ordering::Relaxed),
        }
    }
}

static TIER_COUNTERS: [TierCounters; JitMitigationsTier::COUNT] =
    [const { TierCounters::new() }; JitMitigationsTier::COUNT];

/// Per-tier release-safe CFG/CET counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JitMitigationsTierSnapshot {
    /// Tier these counters describe.
    pub tier: JitMitigationsTier,
    /// Whether this build emits CFG-target metadata for the tier.
    pub cfg_supported: bool,
    /// Whether this build's emitted code is CET shadow-stack/IBT
    /// compatible for the tier.
    pub cet_compatible: bool,
    /// Number of CFG-registration calls (including unsupported / failed).
    pub cfg_register_attempts: u64,
    /// Number of CFG-registration calls that successfully delivered the
    /// target list to the OS.
    pub cfg_register_successes: u64,
    /// Number of CFG-registration calls rejected by the OS after the
    /// supported-tier check passed.
    pub cfg_register_failures: u64,
    /// Number of CFG-registration calls rejected because the tier or
    /// platform does not yet produce safe call-target metadata.  Fail-closed
    /// sentinel; never auto-resets.
    pub cfg_unsupported_tier_attempts: u64,
    /// Total individual CFG call targets registered via successful
    /// registrations.
    pub cfg_targets_registered: u64,
    /// Number of JIT pages observed to be CET-compatible.  Stays at zero
    /// while every tier reports `cet_compatible == false`.
    pub cet_pages_marked_compatible: u64,
    /// Number of JIT pages observed to be CET-incompatible.  Bumped by
    /// the fail-closed [`record_cet_page`] path while no tier is
    /// CET-ready.
    pub cet_pages_marked_incompatible: u64,
}

/// Aggregate snapshot of every release-safe CFG/CET diagnostic counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JitMitigationsSnapshot {
    /// Whether the build target is a Windows target (and therefore CFG
    /// and CET APIs structurally apply).
    pub platform_supported: bool,
    /// Process-level CFG status reported by Windows.
    pub process_cfg_status: MitigationStatus,
    /// Process-level CET shadow-stack status reported by Windows.
    pub process_cet_shadow_stack_status: MitigationStatus,
    /// Process-level CET user-mode IBT status reported by Windows.
    pub process_cet_user_shadow_stack_strict_status: MitigationStatus,
    /// Per-tier counters, indexed by [`JitMitigationsTier`] discriminant.
    pub tiers: [JitMitigationsTierSnapshot; JitMitigationsTier::COUNT],
}

impl JitMitigationsSnapshot {
    /// Look up the snapshot row for a tier.
    #[must_use]
    pub fn tier(&self, tier: JitMitigationsTier) -> &JitMitigationsTierSnapshot {
        &self.tiers[tier as usize]
    }

    /// Sum of `cfg_register_attempts` across all tiers.
    #[must_use]
    pub fn total_cfg_register_attempts(&self) -> u64 {
        self.tiers.iter().map(|t| t.cfg_register_attempts).sum()
    }

    /// Sum of `cfg_unsupported_tier_attempts` across all tiers.
    #[must_use]
    pub fn total_cfg_unsupported_tier_attempts(&self) -> u64 {
        self.tiers
            .iter()
            .map(|t| t.cfg_unsupported_tier_attempts)
            .sum()
    }

    /// Sum of `cfg_targets_registered` across all tiers.
    #[must_use]
    pub fn total_cfg_targets_registered(&self) -> u64 {
        self.tiers.iter().map(|t| t.cfg_targets_registered).sum()
    }

    /// Sum of `cet_pages_marked_incompatible` across all tiers.
    #[must_use]
    pub fn total_cet_pages_marked_incompatible(&self) -> u64 {
        self.tiers
            .iter()
            .map(|t| t.cet_pages_marked_incompatible)
            .sum()
    }
}

/// Whether the current build target is a Windows target (CFG/CET are
/// only meaningful there).
#[must_use]
pub const fn platform_supported() -> bool {
    cfg!(windows)
}

// ─────────────────────────────────────────────────────────────────────────────
// Process-level mitigation probes
// ─────────────────────────────────────────────────────────────────────────────

/// Report whether Control-Flow Guard is enabled for *this process*.
///
/// On non-Windows targets always returns
/// [`MitigationStatus::UnsupportedPlatform`].  On Windows, queries
/// `GetProcessMitigationPolicy` for `ProcessControlFlowGuardPolicy`.
#[must_use]
pub fn process_cfg_status() -> MitigationStatus {
    #[cfg(windows)]
    {
        windows_impl::query_cfg_status()
    }
    #[cfg(not(windows))]
    {
        MitigationStatus::UnsupportedPlatform
    }
}

/// Report whether CET user-mode shadow stack is enabled for this
/// process.
#[must_use]
pub fn process_cet_shadow_stack_status() -> MitigationStatus {
    #[cfg(windows)]
    {
        windows_impl::query_cet_shadow_stack_status(false)
    }
    #[cfg(not(windows))]
    {
        MitigationStatus::UnsupportedPlatform
    }
}

/// Report whether CET user-mode shadow stack is enabled in **strict**
/// mode (no compatibility relaxations) for this process.
#[must_use]
pub fn process_cet_user_shadow_stack_strict_status() -> MitigationStatus {
    #[cfg(windows)]
    {
        windows_impl::query_cet_shadow_stack_status(true)
    }
    #[cfg(not(windows))]
    {
        MitigationStatus::UnsupportedPlatform
    }
}

/// Take a snapshot of every CFG/CET diagnostic counter.
#[must_use]
pub fn snapshot() -> JitMitigationsSnapshot {
    let tiers =
        std::array::from_fn(|idx| TIER_COUNTERS[idx].snapshot(JitMitigationsTier::all()[idx]));
    JitMitigationsSnapshot {
        platform_supported: platform_supported(),
        process_cfg_status: process_cfg_status(),
        process_cet_shadow_stack_status: process_cet_shadow_stack_status(),
        process_cet_user_shadow_stack_strict_status: process_cet_user_shadow_stack_strict_status(),
        tiers,
    }
}

/// Reset every CFG/CET diagnostic counter to zero.
///
/// The counters are process-global; reset is intended for test
/// isolation and Edge benchmark warm-up rather than steady-state
/// production use.
pub fn reset() {
    for counters in &TIER_COUNTERS {
        counters.reset();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public recording surface
// ─────────────────────────────────────────────────────────────────────────────

/// Attempt to register a freshly-emitted JIT code region's CFG call
/// targets with the OS.
///
/// This is the **safe, fail-closed** entry point.  Because no in-tree
/// tier currently produces a verified call-target list, this call
/// always returns [`MitigationError::UnsupportedTier`] (or
/// [`MitigationError::UnsupportedPlatform`] on non-Windows targets) and
/// bumps the corresponding release-safe counter.  It deliberately never
/// invokes `SetProcessValidCallTargets` — feeding the OS an unverified
/// list would *claim* mitigation coverage we have not actually
/// produced.
///
/// When a tier learns to emit a verified target list, flip
/// [`JitMitigationsTier::cfg_supported`] for that arm and route the
/// registration to a (future) `register_cfg_targets` helper that does
/// call the OS with the caller-owned list.
pub fn record_cfg_registration(
    tier: JitMitigationsTier,
    _code_base: *const u8,
    code_len: usize,
    target_count: usize,
) -> Result<(), MitigationError> {
    let counters = &TIER_COUNTERS[tier as usize];
    counters
        .cfg_register_attempts
        .fetch_add(1, Ordering::Relaxed);

    if !platform_supported() {
        counters
            .cfg_unsupported_attempts
            .fetch_add(1, Ordering::Relaxed);
        return Err(MitigationError::UnsupportedPlatform);
    }

    if code_len == 0 || target_count == 0 {
        counters
            .cfg_register_failures
            .fetch_add(1, Ordering::Relaxed);
        return Err(MitigationError::EmptyRegion);
    }

    if !tier.cfg_supported() {
        counters
            .cfg_unsupported_attempts
            .fetch_add(1, Ordering::Relaxed);
        return Err(MitigationError::UnsupportedTier(tier));
    }

    // Reserved for the day a tier reports `cfg_supported = true`: the
    // helper that *actually* calls `SetProcessValidCallTargets` will
    // bump `cfg_register_successes` and `cfg_targets_registered` here.
    counters
        .cfg_register_successes
        .fetch_add(1, Ordering::Relaxed);
    counters
        .cfg_targets_registered
        .fetch_add(target_count as u64, Ordering::Relaxed);
    Ok(())
}

/// Record the CET compatibility outcome for one JIT page in `tier`.
///
/// `compatible == true` means the caller has verified the page's
/// indirect-call targets begin with `ENDBR64` and its returns do not
/// violate the shadow stack.  No in-tree tier currently performs that
/// verification, so even when called with `true` the counter only moves
/// if [`JitMitigationsTier::cet_compatible`] is also true for that
/// tier; otherwise the page is fail-closed into the incompatible bucket.
pub fn record_cet_page(tier: JitMitigationsTier, compatible: bool) {
    let counters = &TIER_COUNTERS[tier as usize];
    if compatible && tier.cet_compatible() {
        counters
            .cet_pages_marked_compatible
            .fetch_add(1, Ordering::Relaxed);
    } else {
        counters
            .cet_pages_marked_incompatible
            .fetch_add(1, Ordering::Relaxed);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Windows backend
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(windows)]
mod windows_impl {
    use super::MitigationStatus;
    use std::ffi::c_void;

    // Re-declare the bits of the Win32 ABI we need.  These match the
    // signatures in <processthreadsapi.h> / <winnt.h>; using the raw
    // declarations avoids pulling a full `windows` / `winapi` crate
    // dependency just for two probes.
    //
    // `PROCESS_MITIGATION_POLICY` values:
    //   ProcessControlFlowGuardPolicy           = 7
    //   ProcessUserShadowStackPolicy            = 17
    const PROCESS_CONTROL_FLOW_GUARD_POLICY: u32 = 7;
    const PROCESS_USER_SHADOW_STACK_POLICY: u32 = 17;

    #[repr(C)]
    #[derive(Default)]
    struct ProcessMitigationControlFlowGuardPolicy {
        flags: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct ProcessMitigationUserShadowStackPolicy {
        flags: u32,
    }

    type GetProcessMitigationPolicyFn = unsafe extern "system" fn(
        process: *mut c_void,
        policy: u32,
        buffer: *mut c_void,
        length: usize,
    ) -> i32;

    type GetCurrentProcessFn = unsafe extern "system" fn() -> *mut c_void;

    type GetModuleHandleAFn = unsafe extern "system" fn(name: *const u8) -> *mut c_void;

    type GetProcAddressFn =
        unsafe extern "system" fn(module: *mut c_void, name: *const u8) -> *mut c_void;

    // We resolve `GetProcessMitigationPolicy` dynamically rather than
    // link-time because it is a Win8+ API and we want pre-Win8 hosts to
    // simply report "Unknown" instead of failing to load Stator.
    fn resolve_get_process_mitigation_policy() -> Option<GetProcessMitigationPolicyFn> {
        // SAFETY: the two function-pointer types match the documented
        // signatures of `GetModuleHandleA` / `GetProcAddress` in
        // kernel32.dll, and we pass only NUL-terminated literal byte
        // strings as their `name` arguments.
        unsafe {
            unsafe extern "system" {
                fn GetModuleHandleA(name: *const u8) -> *mut c_void;
                fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
            }
            let _ = (
                GetModuleHandleA as GetModuleHandleAFn,
                GetProcAddress as GetProcAddressFn,
            );
            let module = GetModuleHandleA(c"kernel32.dll".as_ptr().cast::<u8>());
            if module.is_null() {
                return None;
            }
            let ptr = GetProcAddress(module, c"GetProcessMitigationPolicy".as_ptr().cast::<u8>());
            if ptr.is_null() {
                None
            } else {
                Some(std::mem::transmute::<
                    *mut c_void,
                    GetProcessMitigationPolicyFn,
                >(ptr))
            }
        }
    }

    fn current_process() -> *mut c_void {
        // SAFETY: `GetCurrentProcess` is a kernel32 entry point that
        // takes no arguments and returns a pseudo-handle valid for the
        // lifetime of the process.
        unsafe {
            unsafe extern "system" {
                fn GetCurrentProcess() -> *mut c_void;
            }
            let _ = GetCurrentProcess as GetCurrentProcessFn;
            GetCurrentProcess()
        }
    }

    pub(super) fn query_cfg_status() -> MitigationStatus {
        let Some(get_policy) = resolve_get_process_mitigation_policy() else {
            return MitigationStatus::Unknown;
        };
        let mut policy = ProcessMitigationControlFlowGuardPolicy::default();
        // SAFETY: `policy` is a valid, properly aligned writable
        // location of the size we report; `get_policy` is the resolved
        // `GetProcessMitigationPolicy` entry point and accepts the
        // current-process pseudo-handle.
        let ok = unsafe {
            get_policy(
                current_process(),
                PROCESS_CONTROL_FLOW_GUARD_POLICY,
                &mut policy as *mut _ as *mut c_void,
                std::mem::size_of::<ProcessMitigationControlFlowGuardPolicy>(),
            )
        };
        if ok == 0 {
            return MitigationStatus::Unknown;
        }
        // Bit 0 of the policy flags = `EnableControlFlowGuard`.
        if policy.flags & 0x1 != 0 {
            MitigationStatus::Enabled
        } else {
            MitigationStatus::Disabled
        }
    }

    pub(super) fn query_cet_shadow_stack_status(strict: bool) -> MitigationStatus {
        let Some(get_policy) = resolve_get_process_mitigation_policy() else {
            return MitigationStatus::Unknown;
        };
        let mut policy = ProcessMitigationUserShadowStackPolicy::default();
        // SAFETY: same contract as `query_cfg_status` above; we pass a
        // valid writable buffer of the size we declare.
        let ok = unsafe {
            get_policy(
                current_process(),
                PROCESS_USER_SHADOW_STACK_POLICY,
                &mut policy as *mut _ as *mut c_void,
                std::mem::size_of::<ProcessMitigationUserShadowStackPolicy>(),
            )
        };
        if ok == 0 {
            return MitigationStatus::Unknown;
        }
        // Bit layout (Windows SDK `PROCESS_MITIGATION_USER_SHADOW_STACK_POLICY`):
        //   0  EnableUserShadowStack
        //   1  AuditUserShadowStack
        //   2  SetContextIpValidation
        //   3  AuditSetContextIpValidation
        //   4  UserShadowStackStrictMode
        //   5  BlockNonCetBinaries
        //   6  BlockNonCetBinariesNonEhcont
        //   7  AuditBlockNonCetBinaries
        let enabled = policy.flags & 0x1 != 0;
        if !enabled {
            return MitigationStatus::Disabled;
        }
        if strict {
            if policy.flags & 0x10 != 0 {
                MitigationStatus::Enabled
            } else {
                MitigationStatus::Disabled
            }
        } else {
            MitigationStatus::Enabled
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_platform_supported_matches_target_family() {
        assert_eq!(platform_supported(), cfg!(windows));
    }

    #[test]
    fn test_tier_metadata_is_fail_closed() {
        for tier in JitMitigationsTier::all() {
            assert!(
                !tier.cfg_supported(),
                "tier {tier:?} unexpectedly claims CFG support"
            );
            assert!(
                !tier.cet_compatible(),
                "tier {tier:?} unexpectedly claims CET compatibility"
            );
        }
    }

    #[test]
    fn test_status_label_and_encoding_are_stable() {
        assert_eq!(MitigationStatus::UnsupportedPlatform.as_u32(), 0);
        assert_eq!(MitigationStatus::Disabled.as_u32(), 1);
        assert_eq!(MitigationStatus::Enabled.as_u32(), 2);
        assert_eq!(MitigationStatus::Unknown.as_u32(), 3);
        assert_eq!(MitigationStatus::Enabled.label(), "enabled");
    }

    #[test]
    fn test_reset_zeroes_all_counters() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        let _ = record_cfg_registration(JitMitigationsTier::Baseline, std::ptr::null(), 16, 1);
        record_cet_page(JitMitigationsTier::Maglev, false);
        reset();
        let snap = snapshot();
        for tier in JitMitigationsTier::all() {
            let row = snap.tier(tier);
            assert_eq!(row.cfg_register_attempts, 0);
            assert_eq!(row.cfg_unsupported_tier_attempts, 0);
            assert_eq!(row.cfg_targets_registered, 0);
            assert_eq!(row.cet_pages_marked_compatible, 0);
            assert_eq!(row.cet_pages_marked_incompatible, 0);
        }
    }

    #[test]
    fn test_record_cfg_registration_is_fail_closed_today() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        let err = record_cfg_registration(JitMitigationsTier::Cranelift, std::ptr::null(), 32, 4)
            .expect_err("no tier should be cfg_supported yet");
        if platform_supported() {
            assert_eq!(
                err,
                MitigationError::UnsupportedTier(JitMitigationsTier::Cranelift)
            );
        } else {
            assert_eq!(err, MitigationError::UnsupportedPlatform);
        }
        let snap = snapshot();
        let row = snap.tier(JitMitigationsTier::Cranelift);
        assert_eq!(row.cfg_register_attempts, 1);
        assert_eq!(row.cfg_register_successes, 0);
        assert_eq!(row.cfg_unsupported_tier_attempts, 1);
        assert_eq!(row.cfg_targets_registered, 0);
    }

    #[test]
    fn test_empty_region_increments_register_failures_on_windows() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        let err = record_cfg_registration(JitMitigationsTier::Baseline, std::ptr::null(), 0, 0)
            .expect_err("empty region must be rejected");
        if platform_supported() {
            assert_eq!(err, MitigationError::EmptyRegion);
            let snap = snapshot();
            assert_eq!(
                snap.tier(JitMitigationsTier::Baseline)
                    .cfg_register_failures,
                1
            );
        } else {
            // Off Windows we fail closed on platform before validating the region.
            assert_eq!(err, MitigationError::UnsupportedPlatform);
        }
    }

    #[test]
    fn test_record_cet_page_fails_closed_to_incompatible() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        // Caller *claims* compatibility, but no tier is CET-ready yet,
        // so the count must land in the incompatible bucket.
        record_cet_page(JitMitigationsTier::Turbofan, true);
        record_cet_page(JitMitigationsTier::Turbofan, false);
        let snap = snapshot();
        let row = snap.tier(JitMitigationsTier::Turbofan);
        assert_eq!(row.cet_pages_marked_compatible, 0);
        assert_eq!(row.cet_pages_marked_incompatible, 2);
    }

    #[test]
    fn test_snapshot_populates_per_tier_metadata() {
        let snap = snapshot();
        for tier in JitMitigationsTier::all() {
            let row = snap.tier(tier);
            assert_eq!(row.tier, tier);
            assert_eq!(row.cfg_supported, tier.cfg_supported());
            assert_eq!(row.cet_compatible, tier.cet_compatible());
        }
        assert_eq!(snap.platform_supported, cfg!(windows));
    }

    #[cfg(not(windows))]
    #[test]
    fn test_non_windows_probes_return_unsupported_platform() {
        assert_eq!(process_cfg_status(), MitigationStatus::UnsupportedPlatform);
        assert_eq!(
            process_cet_shadow_stack_status(),
            MitigationStatus::UnsupportedPlatform
        );
        assert_eq!(
            process_cet_user_shadow_stack_strict_status(),
            MitigationStatus::UnsupportedPlatform
        );
    }

    #[cfg(windows)]
    #[test]
    fn test_windows_probes_return_known_status() {
        let cfg = process_cfg_status();
        assert!(matches!(
            cfg,
            MitigationStatus::Enabled | MitigationStatus::Disabled | MitigationStatus::Unknown
        ));
        let cet = process_cet_shadow_stack_status();
        assert!(matches!(
            cet,
            MitigationStatus::Enabled | MitigationStatus::Disabled | MitigationStatus::Unknown
        ));
        let cet_strict = process_cet_user_shadow_stack_strict_status();
        assert!(matches!(
            cet_strict,
            MitigationStatus::Enabled | MitigationStatus::Disabled | MitigationStatus::Unknown
        ));
    }
}
