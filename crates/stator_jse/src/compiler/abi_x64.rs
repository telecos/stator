//! x86-64 ABI abstraction for the Stator JIT.
//!
//! Two ABIs are relevant on x86-64:
//!
//! - **System V AMD64** (Linux, macOS, BSDs): integer parameters in
//!   `RDI, RSI, RDX, RCX, R8, R9`; `RDI` and `RSI` are caller-saved; no
//!   shadow space; the stack must be 16-byte aligned at the point of a
//!   `CALL` instruction (so the callee sees `RSP ≡ 8 (mod 16)` on entry).
//! - **Microsoft x64** (Windows): integer parameters in
//!   `RCX, RDX, R8, R9`; `RDI` and `RSI` are *callee-saved* (extra
//!   non-volatile registers); callers must reserve **32 bytes of shadow
//!   space** immediately above the return address; the stack must be
//!   16-byte aligned at the point of a `CALL`.
//!
//! Generated code historically assumed System V.  This module centralises
//! the per-ABI facts so the baseline and Maglev code generators can ask
//! questions like "which register holds the register-file pointer at JIT
//! function entry?" instead of hard-coding `RDI`.
//!
//! The module is `cfg`-free so it compiles and is unit-testable on every
//! platform — including Windows, where the JIT execution paths are still
//! gated to `unix` while the rest of the Windows port lands in stages.

use crate::compiler::baseline::masm_x64::Reg64;

/// Identifier for an x86-64 calling convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AbiX64 {
    /// System V AMD64 (Linux, macOS, BSDs).
    SysV,
    /// Microsoft x64 (Windows).
    Win64,
}

/// The x86-64 ABI used by the host's `extern "C"` calling convention.
///
/// Selected at compile time from `target_os`.  On non-x86-64 hosts the
/// JIT execution paths are disabled, but the constant still resolves so
/// downstream `cfg`-free code compiles cleanly; it defaults to `SysV` in
/// that case because that is the convention the legacy code generator
/// was written against.
pub const NATIVE_ABI: AbiX64 = if cfg!(target_os = "windows") {
    AbiX64::Win64
} else {
    AbiX64::SysV
};

impl AbiX64 {
    /// Register holding the **register-file base pointer** (1st argument
    /// of `extern "C" fn(*mut i64, ...) -> i64`) on entry to a JIT
    /// function.
    pub const fn entry_arg_register_file(self) -> Reg64 {
        match self {
            AbiX64::SysV => Reg64::Rdi,
            AbiX64::Win64 => Reg64::Rcx,
        }
    }

    /// Register holding the **closure-context raw pointer** (2nd
    /// argument) on entry to a JIT function.
    pub const fn entry_arg_closure_context(self) -> Reg64 {
        match self {
            AbiX64::SysV => Reg64::Rsi,
            AbiX64::Win64 => Reg64::Rdx,
        }
    }

    /// Helper-call (runtime stub) integer argument register at index
    /// `i` (0-based).  Only the indices used by current code generators
    /// are defined; out-of-range indices return `None`.
    pub const fn helper_arg_register(self, i: usize) -> Option<Reg64> {
        match (self, i) {
            (AbiX64::SysV, 0) => Some(Reg64::Rdi),
            (AbiX64::SysV, 1) => Some(Reg64::Rsi),
            (AbiX64::SysV, 2) => Some(Reg64::Rdx),
            (AbiX64::SysV, 3) => Some(Reg64::Rcx),
            (AbiX64::SysV, 4) => Some(Reg64::R8),
            (AbiX64::SysV, 5) => Some(Reg64::R9),
            (AbiX64::Win64, 0) => Some(Reg64::Rcx),
            (AbiX64::Win64, 1) => Some(Reg64::Rdx),
            (AbiX64::Win64, 2) => Some(Reg64::R8),
            (AbiX64::Win64, 3) => Some(Reg64::R9),
            _ => None,
        }
    }

    /// Number of integer-argument registers defined by this ABI.
    pub const fn helper_arg_register_count(self) -> usize {
        match self {
            AbiX64::SysV => 6,
            AbiX64::Win64 => 4,
        }
    }

    /// Bytes of *shadow space* the caller must reserve above the
    /// return address before issuing a `CALL`.  Win64 mandates 32; SysV
    /// has none.
    pub const fn shadow_space_bytes(self) -> i32 {
        match self {
            AbiX64::SysV => 0,
            AbiX64::Win64 => 32,
        }
    }

    /// Stack adjustment (in bytes) the caller must add to `RSP`
    /// immediately before a `CALL` to satisfy the ABI's mandatory
    /// shadow-space contract.  Equivalent to [`Self::shadow_space_bytes`]
    /// but expressed as the value that should be passed to a `sub rsp,
    /// imm` emission.  Returns `0` for SysV.
    pub const fn helper_call_pre_stack_adjust(self) -> i32 {
        self.shadow_space_bytes()
    }

    /// Stack adjustment (in bytes) a generated-code caller must reserve
    /// immediately before a direct JIT-entry `CALL`.
    ///
    /// Direct entry calls share the same shadow-space contract as helper
    /// calls, but exposing a distinct fact keeps call-site code explicit
    /// about the callee ABI it is satisfying.
    pub const fn entry_call_pre_stack_adjust(self) -> i32 {
        self.shadow_space_bytes()
    }

    /// Required `RSP` alignment (in bytes) at the point a `CALL`
    /// instruction is executed.  Both SysV and Win64 mandate 16-byte
    /// alignment.
    pub const fn call_site_stack_alignment(self) -> i32 {
        16
    }

    /// Number of helper-call arguments at indices ≥
    /// [`Self::helper_arg_register_count`] for a call of `total_args`
    /// integer/pointer arguments — i.e. the count that must be passed
    /// on the stack rather than in registers.
    ///
    /// Returns `0` when every argument fits in the ABI's register set.
    pub const fn helper_stack_arg_count(self, total_args: usize) -> usize {
        let regs = self.helper_arg_register_count();
        if total_args > regs {
            total_args - regs
        } else {
            0
        }
    }

    /// Stack offset (bytes from `RSP` *after* the matching
    /// [`Self::helper_call_pre_stack_adjust_for`] reservation has been
    /// emitted) at which the `i`-th helper argument is stored.
    ///
    /// `i` must be ≥ [`Self::helper_arg_register_count`]; smaller
    /// indices are passed in registers and have no stack home in the
    /// outgoing-args region.  Win64 places stack args immediately above
    /// the 32-byte shadow region (so the 5th arg lands at `[RSP+32]`);
    /// SysV places them at offset `0` because it has no shadow space.
    pub const fn helper_stack_arg_offset(self, i: usize) -> i32 {
        let regs = self.helper_arg_register_count();
        // Caller responsibility: this is undefined for register-passed
        // arguments.  We compute as if the i-th outgoing slot follows
        // the shadow space so that Win64 lays out arg 4 at +32.
        self.shadow_space_bytes() + ((i - regs) as i32) * 8
    }

    /// Total bytes the caller must reserve below `RSP` immediately
    /// before issuing a runtime helper `CALL` of `total_args`
    /// integer/pointer arguments.
    ///
    /// The returned value covers both the ABI shadow space (32 bytes
    /// on Win64, none on SysV) and 8 bytes per stack-passed argument,
    /// rounded up to [`Self::call_site_stack_alignment`] so that — given
    /// `RSP` is 16-byte aligned at the point of reservation — the
    /// alignment contract still holds at the `CALL`.
    pub const fn helper_call_pre_stack_adjust_for(self, total_args: usize) -> i32 {
        let raw = self.shadow_space_bytes() + (self.helper_stack_arg_count(total_args) as i32) * 8;
        let align = self.call_site_stack_alignment();
        (raw + align - 1) & !(align - 1)
    }

    /// Extra registers that are **callee-saved on this ABI** but not
    /// on SysV — and therefore must be saved/restored by a JIT
    /// function's prologue/epilogue when this ABI is in effect.
    ///
    /// On SysV this is empty.  On Win64 it is `[RDI, RSI]`, which the
    /// SysV-flavoured code generator otherwise treats as scratch.
    pub const fn extra_entry_callee_saved(self) -> &'static [Reg64] {
        match self {
            AbiX64::SysV => &[],
            AbiX64::Win64 => &[Reg64::Rdi, Reg64::Rsi],
        }
    }

    /// `true` when generated code must explicitly preserve `RDI` and
    /// `RSI` across a JIT-function body because the host ABI treats
    /// them as non-volatile.
    pub const fn entry_must_preserve_rdi_rsi(self) -> bool {
        matches!(self, AbiX64::Win64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sysv_entry_args_are_rdi_rsi() {
        assert_eq!(AbiX64::SysV.entry_arg_register_file(), Reg64::Rdi);
        assert_eq!(AbiX64::SysV.entry_arg_closure_context(), Reg64::Rsi);
    }

    #[test]
    fn win64_entry_args_are_rcx_rdx() {
        assert_eq!(AbiX64::Win64.entry_arg_register_file(), Reg64::Rcx);
        assert_eq!(AbiX64::Win64.entry_arg_closure_context(), Reg64::Rdx);
    }

    #[test]
    fn sysv_helper_args_match_amd64_psabi() {
        let abi = AbiX64::SysV;
        assert_eq!(abi.helper_arg_register(0), Some(Reg64::Rdi));
        assert_eq!(abi.helper_arg_register(1), Some(Reg64::Rsi));
        assert_eq!(abi.helper_arg_register(2), Some(Reg64::Rdx));
        assert_eq!(abi.helper_arg_register(3), Some(Reg64::Rcx));
        assert_eq!(abi.helper_arg_register(4), Some(Reg64::R8));
        assert_eq!(abi.helper_arg_register(5), Some(Reg64::R9));
        assert_eq!(abi.helper_arg_register(6), None);
        assert_eq!(abi.helper_arg_register_count(), 6);
    }

    #[test]
    fn win64_helper_args_match_msft_x64() {
        let abi = AbiX64::Win64;
        assert_eq!(abi.helper_arg_register(0), Some(Reg64::Rcx));
        assert_eq!(abi.helper_arg_register(1), Some(Reg64::Rdx));
        assert_eq!(abi.helper_arg_register(2), Some(Reg64::R8));
        assert_eq!(abi.helper_arg_register(3), Some(Reg64::R9));
        assert_eq!(abi.helper_arg_register(4), None);
        assert_eq!(abi.helper_arg_register_count(), 4);
    }

    #[test]
    fn shadow_space_only_on_win64() {
        assert_eq!(AbiX64::SysV.shadow_space_bytes(), 0);
        assert_eq!(AbiX64::Win64.shadow_space_bytes(), 32);
        assert_eq!(
            AbiX64::Win64.helper_call_pre_stack_adjust(),
            AbiX64::Win64.shadow_space_bytes()
        );
        assert_eq!(AbiX64::SysV.entry_call_pre_stack_adjust(), 0);
        assert_eq!(
            AbiX64::Win64.entry_call_pre_stack_adjust(),
            AbiX64::Win64.shadow_space_bytes()
        );
    }

    #[test]
    fn call_sites_are_16_byte_aligned_under_both_abis() {
        assert_eq!(AbiX64::SysV.call_site_stack_alignment(), 16);
        assert_eq!(AbiX64::Win64.call_site_stack_alignment(), 16);
    }

    #[test]
    fn helper_stack_arg_count_matches_register_capacity() {
        // SysV passes 6 in registers; Win64 passes 4.
        for n in 0..=6 {
            assert_eq!(AbiX64::SysV.helper_stack_arg_count(n), 0);
        }
        assert_eq!(AbiX64::SysV.helper_stack_arg_count(7), 1);
        assert_eq!(AbiX64::SysV.helper_stack_arg_count(8), 2);

        for n in 0..=4 {
            assert_eq!(AbiX64::Win64.helper_stack_arg_count(n), 0);
        }
        assert_eq!(AbiX64::Win64.helper_stack_arg_count(5), 1);
        assert_eq!(AbiX64::Win64.helper_stack_arg_count(7), 3);
        assert_eq!(AbiX64::Win64.helper_stack_arg_count(8), 4);
    }

    #[test]
    fn helper_stack_arg_offset_layout() {
        // SysV: stack args start at +0 (no shadow).  arg index 6 is the
        // first stack arg, then +8 per slot.
        assert_eq!(AbiX64::SysV.helper_stack_arg_offset(6), 0);
        assert_eq!(AbiX64::SysV.helper_stack_arg_offset(7), 8);

        // Win64: stack args start immediately above the 32-byte shadow
        // region.  arg index 4 (the 5th arg) lives at +32.
        assert_eq!(AbiX64::Win64.helper_stack_arg_offset(4), 32);
        assert_eq!(AbiX64::Win64.helper_stack_arg_offset(5), 40);
        assert_eq!(AbiX64::Win64.helper_stack_arg_offset(6), 48);
        assert_eq!(AbiX64::Win64.helper_stack_arg_offset(7), 56);
    }

    #[test]
    fn helper_call_pre_stack_adjust_for_total_args() {
        // SysV: zero shadow.  With ≤ 6 args nothing is reserved.
        for n in 0..=6 {
            assert_eq!(AbiX64::SysV.helper_call_pre_stack_adjust_for(n), 0);
        }
        // SysV stack args are 8 bytes each, rounded up to 16.
        assert_eq!(AbiX64::SysV.helper_call_pre_stack_adjust_for(7), 16);
        assert_eq!(AbiX64::SysV.helper_call_pre_stack_adjust_for(8), 16);
        assert_eq!(AbiX64::SysV.helper_call_pre_stack_adjust_for(9), 32);

        // Win64: shadow space is always reserved.
        for n in 0..=4 {
            assert_eq!(
                AbiX64::Win64.helper_call_pre_stack_adjust_for(n),
                AbiX64::Win64.shadow_space_bytes()
            );
        }
        // Shadow (32) + per-stack-arg, rounded up to 16.
        assert_eq!(AbiX64::Win64.helper_call_pre_stack_adjust_for(5), 48); // 32+8 → 48
        assert_eq!(AbiX64::Win64.helper_call_pre_stack_adjust_for(6), 48); // 32+16 → 48
        assert_eq!(AbiX64::Win64.helper_call_pre_stack_adjust_for(7), 64); // 32+24 → 64
        assert_eq!(AbiX64::Win64.helper_call_pre_stack_adjust_for(8), 64); // 32+32 → 64
        assert_eq!(AbiX64::Win64.helper_call_pre_stack_adjust_for(9), 80); // 32+40 → 80
    }

    /// Reservations returned by [`AbiX64::helper_call_pre_stack_adjust_for`]
    /// must always be a multiple of the call-site alignment so that the
    /// caller's prior 16-byte alignment is preserved through the `CALL`.
    #[test]
    fn helper_call_pre_stack_adjust_for_is_16_byte_aligned() {
        for abi in [AbiX64::SysV, AbiX64::Win64] {
            for n in 0..=10 {
                let adj = abi.helper_call_pre_stack_adjust_for(n);
                assert_eq!(
                    adj % abi.call_site_stack_alignment(),
                    0,
                    "pre-call adjust must preserve 16-byte alignment \
                     (abi = {:?}, total_args = {}, adj = {})",
                    abi,
                    n,
                    adj
                );
            }
        }
    }

    /// For ≤ register-arg-count helper calls, the new
    /// `helper_call_pre_stack_adjust_for` must agree with the original
    /// `helper_call_pre_stack_adjust` (= shadow space) — preserving the
    /// behaviour of every previously-migrated ≤4-arg call site.
    #[test]
    fn pre_stack_adjust_for_matches_simple_helper_adjust_when_no_stack_args() {
        for abi in [AbiX64::SysV, AbiX64::Win64] {
            let regs = abi.helper_arg_register_count();
            for n in 0..=regs {
                assert_eq!(
                    abi.helper_call_pre_stack_adjust_for(n),
                    abi.helper_call_pre_stack_adjust(),
                    "no-stack-arg call must reserve only the shadow space \
                     (abi = {:?}, total_args = {})",
                    abi,
                    n
                );
            }
        }
    }

    #[test]
    fn win64_adds_rdi_rsi_to_callee_saved_set() {
        assert!(AbiX64::SysV.extra_entry_callee_saved().is_empty());
        assert_eq!(
            AbiX64::Win64.extra_entry_callee_saved(),
            &[Reg64::Rdi, Reg64::Rsi]
        );
        assert!(!AbiX64::SysV.entry_must_preserve_rdi_rsi());
        assert!(AbiX64::Win64.entry_must_preserve_rdi_rsi());
    }

    /// The baseline and Maglev prologues rely on
    /// `extra_entry_callee_saved()` having an even number of elements so
    /// they can push/pop the set without disturbing the existing 16-byte
    /// stack-alignment math.  Validate that invariant for every ABI.
    #[test]
    fn extra_entry_callee_saved_has_even_cardinality() {
        for abi in [AbiX64::SysV, AbiX64::Win64] {
            let saved = abi.extra_entry_callee_saved();
            assert_eq!(
                saved.len() % 2,
                0,
                "extra_entry_callee_saved() must have even length to preserve \
                 16-byte stack alignment in JIT prologues (abi = {:?}, set = {:?})",
                abi,
                saved
            );
        }
    }

    /// `extra_entry_callee_saved()` must not include any register that the
    /// JIT relies on as a fixed callee-saved (RBP/RBX/R12-R15) — otherwise
    /// the prologue would push the same physical register twice.
    #[test]
    fn extra_entry_callee_saved_disjoint_from_fixed_callee_saved() {
        let fixed = [
            Reg64::Rbp,
            Reg64::Rbx,
            Reg64::R12,
            Reg64::R13,
            Reg64::R14,
            Reg64::R15,
        ];
        for abi in [AbiX64::SysV, AbiX64::Win64] {
            for reg in abi.extra_entry_callee_saved() {
                assert!(
                    !fixed.contains(reg),
                    "{:?} must not appear in extra_entry_callee_saved() for {:?} \
                     (it is already in the fixed callee-saved push block)",
                    reg,
                    abi
                );
            }
        }
    }

    #[test]
    fn native_abi_matches_target_os() {
        if cfg!(target_os = "windows") {
            assert_eq!(NATIVE_ABI, AbiX64::Win64);
        } else {
            assert_eq!(NATIVE_ABI, AbiX64::SysV);
        }
    }

    #[test]
    fn entry_arg_helpers_are_const_evaluable() {
        // Force const evaluation in this scope; ensures the helpers
        // remain usable from `const fn` contexts in code generators.
        const SYSV_RF: Reg64 = AbiX64::SysV.entry_arg_register_file();
        const WIN_RF: Reg64 = AbiX64::Win64.entry_arg_register_file();
        const NATIVE_RF: Reg64 = NATIVE_ABI.entry_arg_register_file();
        assert_eq!(SYSV_RF, Reg64::Rdi);
        assert_eq!(WIN_RF, Reg64::Rcx);
        assert_eq!(
            NATIVE_RF,
            if cfg!(target_os = "windows") {
                Reg64::Rcx
            } else {
                Reg64::Rdi
            }
        );
    }
}
