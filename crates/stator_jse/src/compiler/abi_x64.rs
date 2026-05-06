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

    /// Required `RSP` alignment (in bytes) at the point a `CALL`
    /// instruction is executed.  Both SysV and Win64 mandate 16-byte
    /// alignment.
    pub const fn call_site_stack_alignment(self) -> i32 {
        16
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
    }

    #[test]
    fn call_sites_are_16_byte_aligned_under_both_abis() {
        assert_eq!(AbiX64::SysV.call_site_stack_alignment(), 16);
        assert_eq!(AbiX64::Win64.call_site_stack_alignment(), 16);
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
