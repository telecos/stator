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

    /// Maglev's allocatable physical-register **bank**, in
    /// register-allocator-index order.
    ///
    /// Index `n` returned by the Maglev linear-scan allocator
    /// corresponds to the `n`-th [`Reg64`] in this slice.  The bank is
    /// currently identical for SysV and Win64:
    ///
    /// ```text
    /// 0:RBX 1:RCX 2:RDX 3:RSI 4:R8 5:R9 6:R13 7:R12 8:R15
    /// ```
    ///
    /// `RSI` is a SysV scratch register and a Win64 *callee-saved*
    /// register — leaving it in the bank under Win64 is safe because
    /// the JIT prologue/epilogue (see [`Self::extra_entry_callee_saved`])
    /// already saves and restores it.  Helper-call clobber metadata,
    /// however, *does* differ between ABIs and is exposed via
    /// [`Self::maglev_caller_saved_indices`] /
    /// [`Self::maglev_caller_saved_mask`].
    pub const fn maglev_allocatable_registers(self) -> &'static [Reg64] {
        // The bank is the same under both ABIs today; the only ABI
        // variation lives in the caller-saved subset below.
        const BANK: &[Reg64] = &[
            Reg64::Rbx,
            Reg64::Rcx,
            Reg64::Rdx,
            Reg64::Rsi,
            Reg64::R8,
            Reg64::R9,
            Reg64::R13,
            Reg64::R12,
            Reg64::R15,
        ];
        let _ = self;
        BANK
    }

    /// Number of allocatable physical registers Maglev exposes for a
    /// function compiled under this ABI.
    pub const fn maglev_register_bank_size(self) -> u32 {
        self.maglev_allocatable_registers().len() as u32
    }

    /// Indices into [`Self::maglev_allocatable_registers`] whose
    /// physical register is **caller-clobbered** across an
    /// `extern "C"` helper / runtime call under this ABI.
    ///
    /// These are exactly the bank entries the Maglev codegen must
    /// save around a helper call (in addition to scratch registers
    /// outside the bank such as `RAX` / `R11`).
    ///
    /// - **SysV**: `RCX(1), RDX(2), RSI(3), R8(4), R9(5)`.
    /// - **Win64**: `RCX(1), RDX(2), R8(4), R9(5)` — `RSI(3)` is
    ///   non-volatile and is preserved by the JIT prologue.
    pub const fn maglev_caller_saved_indices(self) -> &'static [u32] {
        match self {
            AbiX64::SysV => &[1, 2, 3, 4, 5],
            AbiX64::Win64 => &[1, 2, 4, 5],
        }
    }

    /// Bitmask form of [`Self::maglev_caller_saved_indices`]: bit `i`
    /// is set iff bank index `i` is caller-clobbered under this ABI.
    ///
    /// The mask is intentionally narrow (`u8`) — current bank
    /// caller-saved indices all fit in the low 8 bits.  An
    /// out-of-range index would be a bug in
    /// [`Self::maglev_caller_saved_indices`]; the unit tests in this
    /// module cross-check the two representations.
    pub const fn maglev_caller_saved_mask(self) -> u8 {
        match self {
            AbiX64::SysV => (1u8 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5),
            AbiX64::Win64 => (1u8 << 1) | (1 << 2) | (1 << 4) | (1 << 5),
        }
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

    /// The Maglev allocatable register bank is the contract between
    /// the (ABI-agnostic) linear-scan allocator and the (ABI-specific)
    /// codegen.  Its layout — including `Rsi` at index 3 — is locked
    /// in for both ABIs.  Win64 keeps `Rsi` allocatable because the
    /// JIT prologue saves it via `extra_entry_callee_saved`.
    #[test]
    fn maglev_bank_layout_is_stable_for_both_abis() {
        let expected: &[Reg64] = &[
            Reg64::Rbx,
            Reg64::Rcx,
            Reg64::Rdx,
            Reg64::Rsi,
            Reg64::R8,
            Reg64::R9,
            Reg64::R13,
            Reg64::R12,
            Reg64::R15,
        ];
        for abi in [AbiX64::SysV, AbiX64::Win64] {
            assert_eq!(
                abi.maglev_allocatable_registers(),
                expected,
                "Maglev bank changed for {:?} — review caller-saved \
                 indices and codegen `phys_reg` mapping together",
                abi
            );
            assert_eq!(abi.maglev_register_bank_size(), expected.len() as u32);
        }
    }

    /// SysV must keep its current Maglev caller-saved set (RCX, RDX,
    /// RSI, R8, R9 → indices 1..=5).  Any drift here would silently
    /// change spill behaviour at every helper-call site.
    #[test]
    fn maglev_caller_saved_sysv_matches_legacy_indices_1_to_5() {
        let abi = AbiX64::SysV;
        assert_eq!(abi.maglev_caller_saved_indices(), &[1, 2, 3, 4, 5]);
        assert_eq!(
            abi.maglev_caller_saved_mask(),
            (1u8 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
        );
    }

    /// Win64 helper calls must **not** clobber `RSI` (it is
    /// non-volatile on Win64 and is preserved by the JIT prologue),
    /// but they do clobber `RCX, RDX, R8, R9`.
    #[test]
    fn maglev_caller_saved_win64_excludes_rsi() {
        let abi = AbiX64::Win64;
        let bank = abi.maglev_allocatable_registers();
        let indices = abi.maglev_caller_saved_indices();
        // RSI lives at bank index 3 — explicitly assert it is absent.
        assert_eq!(bank[3], Reg64::Rsi);
        assert!(
            !indices.contains(&3),
            "Win64 maglev_caller_saved_indices must exclude RSI (index 3): {:?}",
            indices
        );
        assert_eq!(indices, &[1, 2, 4, 5]);

        // Cross-check the mask form.
        let expected_mask = (1u8 << 1) | (1 << 2) | (1 << 4) | (1 << 5);
        assert_eq!(abi.maglev_caller_saved_mask(), expected_mask);
        assert_eq!(abi.maglev_caller_saved_mask() & (1u8 << 3), 0);

        // And cross-check that every named-volatile bank entry is
        // actually a Win64 volatile integer register.
        for &i in indices {
            let r = bank[i as usize];
            assert!(
                matches!(r, Reg64::Rcx | Reg64::Rdx | Reg64::R8 | Reg64::R9),
                "bank[{}] = {:?} is not a Win64 volatile integer reg",
                i,
                r
            );
        }
    }

    /// `maglev_caller_saved_mask` and `maglev_caller_saved_indices`
    /// must agree bit-for-bit on every ABI, and every advertised
    /// caller-saved index must fall inside the allocatable bank.
    #[test]
    fn maglev_caller_saved_mask_matches_indices() {
        for abi in [AbiX64::SysV, AbiX64::Win64] {
            let bank_size = abi.maglev_register_bank_size();
            let mut from_indices: u8 = 0;
            for &i in abi.maglev_caller_saved_indices() {
                assert!(
                    i < bank_size,
                    "caller-saved index {} out of bank ({}) for {:?}",
                    i,
                    bank_size,
                    abi
                );
                assert!(
                    i < 8,
                    "caller-saved index {} would not fit in u8 mask for {:?}",
                    i,
                    abi
                );
                from_indices |= 1u8 << i;
            }
            assert_eq!(
                from_indices,
                abi.maglev_caller_saved_mask(),
                "indices/mask disagree for {:?}",
                abi
            );
        }
    }

    /// The Maglev caller-saved set must be a subset of the registers
    /// the Win64/SysV ABI actually treats as volatile across a C call.
    /// In particular, on Win64 the set must avoid the non-volatile
    /// integer registers that the JIT prologue already preserves.
    #[test]
    fn maglev_caller_saved_does_not_overlap_callee_saved() {
        // Universal integer callee-saves (both ABIs).  RBP is the
        // frame pointer and is intentionally excluded from the bank.
        let universal_callee_saved = [Reg64::Rbx, Reg64::R12, Reg64::R13, Reg64::R14, Reg64::R15];
        for abi in [AbiX64::SysV, AbiX64::Win64] {
            let bank = abi.maglev_allocatable_registers();
            for &i in abi.maglev_caller_saved_indices() {
                let r = bank[i as usize];
                assert!(
                    !universal_callee_saved.contains(&r),
                    "{:?} bank[{}] = {:?} is callee-saved on both ABIs and \
                     must not appear in maglev_caller_saved_indices()",
                    abi,
                    i,
                    r
                );
            }
        }
        // Win64-only: RDI/RSI are also callee-saved and must be
        // absent from the Win64 caller-saved set.
        let bank = AbiX64::Win64.maglev_allocatable_registers();
        for &i in AbiX64::Win64.maglev_caller_saved_indices() {
            let r = bank[i as usize];
            assert!(
                r != Reg64::Rdi && r != Reg64::Rsi,
                "Win64 bank[{}] = {:?} is callee-saved on Win64",
                i,
                r
            );
        }
    }

    /// The bank/mask helpers must remain `const fn`-callable so the
    /// codegen can build static tables from them.
    #[test]
    fn maglev_bank_helpers_are_const_evaluable() {
        const SYSV_BANK_LEN: u32 = AbiX64::SysV.maglev_register_bank_size();
        const WIN_BANK_LEN: u32 = AbiX64::Win64.maglev_register_bank_size();
        const SYSV_MASK: u8 = AbiX64::SysV.maglev_caller_saved_mask();
        const WIN_MASK: u8 = AbiX64::Win64.maglev_caller_saved_mask();
        const NATIVE_MASK: u8 = NATIVE_ABI.maglev_caller_saved_mask();
        assert_eq!(SYSV_BANK_LEN, 9);
        assert_eq!(WIN_BANK_LEN, 9);
        assert_ne!(SYSV_MASK, 0);
        assert_ne!(WIN_MASK, 0);
        assert!(NATIVE_MASK == SYSV_MASK || NATIVE_MASK == WIN_MASK);
    }
}
