//! x86-64 macro-assembler for the Stator baseline compiler.
//!
//! Emits x86-64 machine code into an owned byte buffer.  The assembler
//! supports the instructions required for a baseline JIT compiler tier: data
//! movement, arithmetic, comparisons, control flow, and function
//! call/return conventions.
//!
//! # Calling convention
//!
//! All helpers follow the **System V AMD64 ABI** (Linux/macOS):
//!
//! | Role            | Registers                   |
//! |-----------------|-----------------------------|
//! | Parameters 1–6  | RDI, RSI, RDX, RCX, R8, R9 |
//! | Return value    | RAX                         |
//! | Caller-saved    | RAX, RCX, RDX, RSI, RDI, R8–R11 |
//! | Callee-saved    | RBX, RBP, R12–R15, RSP      |
//!
//! # Example
//!
//! ```
//! use stator_core::compiler::baseline::masm_x64::{MacroAssembler, Reg64, Label};
//!
//! // Emit: fn identity(x: i64) -> i64 { x }
//! // (SysV AMD64: first arg in RDI, return value in RAX)
//! let mut masm = MacroAssembler::new();
//! masm.mov_rr(Reg64::Rax, Reg64::Rdi);
//! masm.ret();
//!
//! // The emitted bytes are: REX.W MOV RAX,RDI + RET
//! assert_eq!(masm.code(), &[0x48, 0x8B, 0xC7, 0xC3]);
//! ```

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// CondCode
// ─────────────────────────────────────────────────────────────────────────────

/// A condition code for [`MacroAssembler::jcc`] and [`MacroAssembler::setcc_al`].
///
/// Includes both **signed** and **unsigned** comparison conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CondCode {
    /// Equal (`ZF = 1`).
    Equal,
    /// Not equal (`ZF = 0`).
    NotEqual,
    /// Less-than, signed (`SF ≠ OF`).
    Less,
    /// Less-than-or-equal, signed (`ZF = 1 or SF ≠ OF`).
    LessEq,
    /// Greater-than, signed (`ZF = 0 and SF = OF`).
    Greater,
    /// Greater-than-or-equal, signed (`SF = OF`).
    GreaterEq,
    /// Overflow (`OF = 1`).
    Overflow,
    /// Above-or-equal, unsigned (`CF = 0`).  Also known as JNB/JNC.
    AboveEq,
    /// Below, unsigned (`CF = 1`).  Also known as JC.
    Below,
}

impl CondCode {
    /// Second opcode byte for the `SETCC` (0F 9x) family.
    pub(crate) fn setcc_byte(self) -> u8 {
        match self {
            Self::Equal => 0x94,
            Self::NotEqual => 0x95,
            Self::Less => 0x9C,
            Self::LessEq => 0x9E,
            Self::Greater => 0x9F,
            Self::GreaterEq => 0x9D,
            Self::Overflow => 0x90,
            Self::AboveEq => 0x93,
            Self::Below => 0x92,
        }
    }

    /// Second opcode byte for the near `Jcc` (0F 8x) family.
    pub(crate) fn jcc_byte(self) -> u8 {
        match self {
            Self::Equal => 0x84,
            Self::NotEqual => 0x85,
            Self::Less => 0x8C,
            Self::LessEq => 0x8E,
            Self::Greater => 0x8F,
            Self::GreaterEq => 0x8D,
            Self::Overflow => 0x80,
            Self::AboveEq => 0x83,
            Self::Below => 0x82,
        }
    }

    /// Return the logically inverted condition code.
    ///
    /// For example, `Less` inverts to `GreaterEq`, `Equal` inverts to
    /// `NotEqual`, and so on.  Used by comparison-branch fusion to reverse
    /// the sense of a conditional jump.
    pub fn invert(self) -> Self {
        match self {
            Self::Equal => Self::NotEqual,
            Self::NotEqual => Self::Equal,
            Self::Less => Self::GreaterEq,
            Self::LessEq => Self::Greater,
            Self::Greater => Self::LessEq,
            Self::GreaterEq => Self::Less,
            Self::Overflow => Self::Overflow, // no logical inverse for OF
            Self::AboveEq => Self::Below,
            Self::Below => Self::AboveEq,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reg64
// ─────────────────────────────────────────────────────────────────────────────

/// An x86-64 general-purpose 64-bit register.
///
/// The `u8` discriminant is the hardware encoding (0–15).  Registers 0–7
/// (RAX through RDI) do not require a REX extension bit; registers 8–15
/// (R8–R15) set either REX.R (when used in the ModRM `reg` field) or REX.B
/// (when used in the ModRM `r/m` field or as the target of a short-form
/// opcode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Reg64 {
    /// RAX — general-purpose / return value register.
    Rax = 0,
    /// RCX — general-purpose / 4th parameter (Windows ABI).
    Rcx = 1,
    /// RDX — general-purpose / 3rd parameter (SysV AMD64).
    Rdx = 2,
    /// RBX — general-purpose, callee-saved.
    Rbx = 3,
    /// RSP — stack pointer.
    Rsp = 4,
    /// RBP — frame pointer, callee-saved.
    Rbp = 5,
    /// RSI — 2nd parameter (SysV AMD64).
    Rsi = 6,
    /// RDI — 1st parameter (SysV AMD64).
    Rdi = 7,
    /// R8 — 5th parameter (SysV AMD64), caller-saved.
    R8 = 8,
    /// R9 — 6th parameter (SysV AMD64), caller-saved.
    R9 = 9,
    /// R10 — caller-saved scratch register.
    R10 = 10,
    /// R11 — caller-saved scratch register.
    R11 = 11,
    /// R12 — callee-saved.
    R12 = 12,
    /// R13 — callee-saved.
    R13 = 13,
    /// R14 — callee-saved.
    R14 = 14,
    /// R15 — callee-saved.
    R15 = 15,
}

impl Reg64 {
    /// The 3-bit register encoding placed in the ModRM/SIB `reg` or `r/m`
    /// field (low 3 bits of the register number).
    #[inline]
    pub(crate) fn enc(self) -> u8 {
        (self as u8) & 0x07
    }

    /// `true` if this register requires the REX extension bit (R8–R15).
    #[inline]
    pub(crate) fn needs_rex(self) -> bool {
        (self as u8) >= 8
    }
}

impl fmt::Display for Reg64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Rax => "rax",
            Self::Rcx => "rcx",
            Self::Rdx => "rdx",
            Self::Rbx => "rbx",
            Self::Rsp => "rsp",
            Self::Rbp => "rbp",
            Self::Rsi => "rsi",
            Self::Rdi => "rdi",
            Self::R8 => "r8",
            Self::R9 => "r9",
            Self::R10 => "r10",
            Self::R11 => "r11",
            Self::R12 => "r12",
            Self::R13 => "r13",
            Self::R14 => "r14",
            Self::R15 => "r15",
        };
        f.write_str(name)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Label
// ─────────────────────────────────────────────────────────────────────────────

/// A position in the code buffer used as a branch target.
///
/// A label can be:
///
/// - **Unbound** (the default) — not yet placed at a specific offset.
///   Forward branches to an unbound label record a *patch site*: the 4-byte
///   `rel32` field is filled with zeros and the offset is saved.
/// - **Bound** — placed at a specific byte offset by
///   [`MacroAssembler::bind_label`].  All recorded patch sites are
///   retroactively filled with the correct signed 32-bit displacement.
///
/// # Example — forward jump
///
/// ```
/// use stator_core::compiler::baseline::masm_x64::{MacroAssembler, Reg64, Label};
///
/// let mut masm = MacroAssembler::new();
/// let mut done = Label::new();
///
/// // Jump unconditionally past the `mov`.
/// masm.jmp(&mut done);
/// masm.mov_ri(Reg64::Rax, 99); // this instruction is skipped
/// masm.bind_label(&mut done);
/// masm.ret();
/// ```
///
/// # Example — backward jump (loop)
///
/// ```
/// use stator_core::compiler::baseline::masm_x64::{MacroAssembler, Reg64, Label};
///
/// let mut masm = MacroAssembler::new();
/// let mut loop_top = Label::new();
///
/// masm.bind_label(&mut loop_top);   // mark loop start
/// masm.sub_ri(Reg64::Rcx, 1);       // dec loop counter
/// masm.jne(&mut loop_top);          // branch back if non-zero
/// masm.ret();
/// ```
#[derive(Debug, Default)]
pub struct Label {
    /// Byte offset in the code buffer, if already bound.
    bound_offset: Option<usize>,
    /// Offsets of the 4-byte `rel32` fields that must be patched when this
    /// label is eventually bound.
    patch_sites: Vec<usize>,
}

impl Label {
    /// Create a new, unbound label.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the bound byte offset, or `None` if the label has not yet been
    /// placed with [`MacroAssembler::bind_label`].
    pub fn bound_offset(&self) -> Option<usize> {
        self.bound_offset
    }

    /// Returns `true` if this label has been placed with
    /// [`MacroAssembler::bind_label`].
    pub fn is_bound(&self) -> bool {
        self.bound_offset.is_some()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MacroAssembler
// ─────────────────────────────────────────────────────────────────────────────

/// x86-64 macro-assembler.
///
/// Appends raw machine code bytes to an internal `Vec<u8>` buffer.  All
/// arithmetic and data-movement instructions operate on **64-bit** operands
/// (REX.W prefix is always emitted for those instructions).
///
/// # Usage
///
/// ```
/// use stator_core::compiler::baseline::masm_x64::{MacroAssembler, Reg64, Label};
///
/// // Emit: fn add(a: i64, b: i64) -> i64 { a + b }
/// // (SysV AMD64: a in RDI, b in RSI, result in RAX)
/// let mut masm = MacroAssembler::new();
/// masm.mov_rr(Reg64::Rax, Reg64::Rdi);
/// masm.add_rr(Reg64::Rax, Reg64::Rsi);
/// masm.ret();
/// assert_eq!(masm.code(), &[0x48, 0x8B, 0xC7, 0x48, 0x03, 0xC6, 0xC3]);
/// ```
#[derive(Debug, Default)]
pub struct MacroAssembler {
    buf: Vec<u8>,
}

impl MacroAssembler {
    /// Create a new, empty assembler.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a slice of the emitted code bytes.
    pub fn code(&self) -> &[u8] {
        &self.buf
    }

    /// Consume the assembler and return the code buffer.
    pub fn into_code(self) -> Vec<u8> {
        self.buf
    }

    /// Current write position (byte offset of the next instruction to be
    /// emitted).
    pub fn position(&self) -> usize {
        self.buf.len()
    }

    /// Emit multi-byte NOP padding to align the current position to
    /// `boundary` bytes.  Uses Intel-recommended NOP sequences (1–9 bytes)
    /// for best decode throughput.
    pub fn align_to(&mut self, boundary: usize) {
        debug_assert!(boundary.is_power_of_two());
        let pos = self.buf.len();
        let remainder = pos & (boundary - 1);
        if remainder == 0 {
            return;
        }
        let pad = boundary - remainder;
        self.emit_nop_sequence(pad);
    }

    /// Emit `n` bytes of NOP padding using Intel-recommended multi-byte
    /// NOP encodings for best decode throughput.
    fn emit_nop_sequence(&mut self, mut n: usize) {
        // Intel multi-byte NOPs (Vol 2, Table 4-12):
        // 1: 90
        // 2: 66 90
        // 3: 0F 1F 00
        // 4: 0F 1F 40 00
        // 5: 0F 1F 44 00 00
        // 6: 66 0F 1F 44 00 00
        // 7: 0F 1F 80 00 00 00 00
        // 8: 0F 1F 84 00 00 00 00 00
        // 9: 66 0F 1F 84 00 00 00 00 00
        static NOPS: [&[u8]; 10] = [
            &[],
            &[0x90],
            &[0x66, 0x90],
            &[0x0F, 0x1F, 0x00],
            &[0x0F, 0x1F, 0x40, 0x00],
            &[0x0F, 0x1F, 0x44, 0x00, 0x00],
            &[0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00],
            &[0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00],
            &[0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
            &[0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
        ];
        while n > 0 {
            let chunk = n.min(9);
            self.buf.extend_from_slice(NOPS[chunk]);
            n -= chunk;
        }
    }

    // ── Label support ────────────────────────────────────────────────────────

    /// Bind `label` to the current position in the code buffer.
    ///
    /// If the label was referenced by forward branches, all pending patch sites
    /// are retroactively filled with the correct signed 32-bit displacement.
    ///
    /// # Panics
    ///
    /// Panics if the label has already been bound — double-binding is a
    /// compiler bug.
    pub fn bind_label(&mut self, label: &mut Label) {
        assert!(label.bound_offset.is_none(), "label already bound");
        let here = self.buf.len();
        for site in &label.patch_sites {
            let rel32 = (here as i32) - ((*site as i32) + 4);
            self.buf[*site..*site + 4].copy_from_slice(&rel32.to_le_bytes());
        }
        label.patch_sites.clear();
        label.bound_offset = Some(here);
    }

    // ── Low-level encoding helpers ───────────────────────────────────────────

    /// Emit a REX byte with REX.W set for a two-register operand.
    ///
    /// REX.R is set when `reg_field` is R8–R15 (extends the ModRM `reg`
    /// field).  REX.B is set when `rm_field` is R8–R15 (extends the ModRM
    /// `r/m` field).  REX.X is never set (no SIB byte).
    fn emit_rex_wrb(&mut self, reg_field: Reg64, rm_field: Reg64) {
        let r_bit = if reg_field.needs_rex() { 0x04 } else { 0 };
        let b_bit = if rm_field.needs_rex() { 0x01 } else { 0 };
        // REX = 0100 W R X B  (W=1 for 64-bit operand size)
        self.buf.push(0x48 | r_bit | b_bit);
    }

    /// Emit a REX prefix **without** the W bit (32-bit operand size).
    ///
    /// Only emits when at least one of `reg_field` / `rm_field` is R8–R15.
    /// For low registers (RAX–RDI) this is a no-op.
    fn emit_rex_rb_if_needed(&mut self, reg_field: Reg64, rm_field: Reg64) {
        let r_bit = if reg_field.needs_rex() { 0x04 } else { 0 };
        let b_bit = if rm_field.needs_rex() { 0x01 } else { 0 };
        if r_bit | b_bit != 0 {
            self.buf.push(0x40 | r_bit | b_bit);
        }
    }

    /// Emit a REX.B-only prefix for single-register short-form instructions
    /// (PUSH, POP).
    ///
    /// Only emits the byte when `reg` requires the REX extension (R8–R15).
    fn emit_rex_b_only(&mut self, reg: Reg64) {
        if reg.needs_rex() {
            self.buf.push(0x41); // REX.B
        }
    }

    /// Emit a ModRM byte with `mod=11` (register-direct), placing `reg_field`
    /// in the `reg` bits and `rm_field` in the `r/m` bits.
    fn emit_modrm_rr(&mut self, reg_field: Reg64, rm_field: Reg64) {
        self.buf
            .push(0xC0 | (reg_field.enc() << 3) | rm_field.enc());
    }

    /// Emit a ModRM byte with `mod=11` and an opcode-extension digit in the
    /// `reg` field (`/digit` form used by immediate instructions).
    fn emit_modrm_digit(&mut self, digit: u8, rm_field: Reg64) {
        self.buf.push(0xC0 | (digit << 3) | rm_field.enc());
    }

    /// `LEA dst, [base + index * scale]` — compute an effective address using
    /// a SIB (Scale-Index-Base) addressing mode, 64-bit result.
    ///
    /// `scale` must be one of 1, 2, 4, or 8.  This is useful for strength-
    /// reducing multiplies by small constants:
    ///
    /// - `x * 3` → `lea dst, [x + x*2]`
    /// - `x * 5` → `lea dst, [x + x*4]`
    /// - `x * 9` → `lea dst, [x + x*8]`
    ///
    /// Encoding: `REX.W 8D /r` with ModRM mod=00, r/m=100 (SIB present),
    /// SIB byte = `SS.index.base`.
    ///
    /// **Note**: `base` must not be RBP/R13 (encoding ambiguity with mod=00).
    /// `index` must not be RSP (index=100 means "no index").
    pub fn lea_scaled(&mut self, dst: Reg64, base: Reg64, index: Reg64, scale: u8) {
        debug_assert!(
            matches!(scale, 1 | 2 | 4 | 8),
            "LEA scale must be 1, 2, 4, or 8"
        );
        let ss = match scale {
            1 => 0b00,
            2 => 0b01,
            4 => 0b10,
            8 => 0b11,
            _ => unreachable!(),
        };
        // REX.W (64-bit) | REX.R (dst ext) | REX.X (index ext) | REX.B (base ext)
        let r_bit = if dst.needs_rex() { 0x04 } else { 0 };
        let x_bit = if index.needs_rex() { 0x02 } else { 0 };
        let b_bit = if base.needs_rex() { 0x01 } else { 0 };
        self.buf.push(0x48 | r_bit | x_bit | b_bit);
        self.buf.push(0x8D); // LEA opcode
        // ModRM: mod=00, reg=dst, r/m=100 (SIB follows)
        self.buf.push((dst.enc() << 3) | 0x04);
        // SIB: scale=ss, index=index, base=base
        self.buf.push((ss << 6) | (index.enc() << 3) | base.enc());
    }

    /// Emit a signed 32-bit integer in little-endian byte order.
    fn emit_i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Emit a signed 64-bit integer in little-endian byte order.
    fn emit_i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Emit the 4-byte `rel32` field for a branch to `label`.
    ///
    /// If the label is already bound the displacement is computed immediately.
    /// Otherwise, four zero bytes are emitted and the offset of this field is
    /// saved in `label.patch_sites` so that [`bind_label`] can fill it in
    /// later.
    ///
    /// [`bind_label`]: MacroAssembler::bind_label
    fn emit_rel32_for_label(&mut self, label: &mut Label) {
        let rel32_start = self.buf.len();
        if let Some(target) = label.bound_offset {
            // Backward reference — displacement is known immediately.
            let rel32 = (target as i32) - ((rel32_start as i32) + 4);
            self.emit_i32(rel32);
        } else {
            // Forward reference — emit placeholder and record the patch site.
            self.buf.extend_from_slice(&[0u8; 4]);
            label.patch_sites.push(rel32_start);
        }
    }

    // ── Data movement ────────────────────────────────────────────────────────

    /// `MOV dst, src` — copy a 64-bit register.
    ///
    /// Encoding: `REX.W 8B /r` (MOV r64, r/m64).
    pub fn mov_rr(&mut self, dst: Reg64, src: Reg64) {
        self.emit_rex_wrb(dst, src);
        self.buf.push(0x8B);
        self.emit_modrm_rr(dst, src);
    }

    /// `MOV dst, imm` — load a 64-bit immediate into a register.
    ///
    /// Uses the 7-byte `REX.W C7 /0 imm32` (sign-extended) form when `imm`
    /// fits in a 32-bit signed integer, and the 10-byte `REX.W B8+rd imm64`
    /// form otherwise.
    pub fn mov_ri(&mut self, dst: Reg64, imm: i64) {
        if (i32::MIN as i64..=i32::MAX as i64).contains(&imm) {
            // REX.W + C7 /0 + ModRM + imm32 (sign-extended to 64 bits).
            // Use Rax as dummy for reg field (enc=0, needs_rex=false).
            self.emit_rex_wrb(Reg64::Rax, dst);
            self.buf.push(0xC7);
            self.emit_modrm_digit(0, dst);
            self.emit_i32(imm as i32);
        } else {
            // REX.W + (B8 + rd) + imm64.
            let b_bit = if dst.needs_rex() { 0x01 } else { 0 };
            self.buf.push(0x48 | b_bit);
            self.buf.push(0xB8 | dst.enc());
            self.emit_i64(imm);
        }
    }

    /// `MOV dst_32, imm32` — load a 32-bit immediate into a register,
    /// zero-extending to 64 bits.
    ///
    /// Shorter than [`mov_ri`] for non-negative values that fit in `u32`:
    /// 5 bytes (classic registers) or 6 bytes (R8–R15) vs 7 bytes for the
    /// sign-extended 64-bit form.
    ///
    /// Encoding: `[REX.B] B8+rd imm32` (no REX.W — 32-bit operand size
    /// implicitly zero-extends the upper 32 bits).
    pub fn mov_ri32(&mut self, dst: Reg64, imm: u32) {
        self.emit_rex_b_only(dst);
        self.buf.push(0xB8 | dst.enc());
        self.buf.extend_from_slice(&imm.to_le_bytes());
    }

    /// `LEA dst, [RIP + offset]` — load a RIP-relative address.
    ///
    /// `offset` is the signed byte displacement from the **end** of this LEA
    /// instruction (i.e. from the address of the next instruction).
    ///
    /// Encoding: `REX.W 8D /r` with `ModRM(mod=00, reg=dst, r/m=101)`.
    /// `r/m=101` with `mod=00` selects RIP-relative addressing in 64-bit mode.
    pub fn lea_rip_rel(&mut self, dst: Reg64, offset: i32) {
        let r_bit = if dst.needs_rex() { 0x04 } else { 0 };
        self.buf.push(0x48 | r_bit); // REX.W [+ REX.R]
        self.buf.push(0x8D);
        // ModRM: mod=00, reg=dst, r/m=101 (RIP-relative).
        self.buf.push((dst.enc() << 3) | 0x05);
        self.emit_i32(offset);
    }

    // ── Arithmetic ───────────────────────────────────────────────────────────

    /// `ADD dst, src` — add two 64-bit registers (`dst += src`).
    ///
    /// Encoding: `REX.W 03 /r` (ADD r64, r/m64).
    pub fn add_rr(&mut self, dst: Reg64, src: Reg64) {
        self.emit_rex_wrb(dst, src);
        self.buf.push(0x03);
        self.emit_modrm_rr(dst, src);
    }

    /// `ADD dst, imm` — add a sign-extended 32-bit immediate to a register.
    ///
    /// Uses the 4-byte `REX.W 83 /0 imm8` form when `imm` fits in a signed
    /// byte, and the 7-byte `REX.W 81 /0 imm32` form otherwise.
    pub fn add_ri(&mut self, dst: Reg64, imm: i32) {
        self.emit_rex_wrb(Reg64::Rax, dst);
        if (i8::MIN as i32..=i8::MAX as i32).contains(&imm) {
            self.buf.push(0x83);
            self.emit_modrm_digit(0, dst);
            self.buf.push(imm as i8 as u8);
        } else {
            self.buf.push(0x81);
            self.emit_modrm_digit(0, dst);
            self.emit_i32(imm);
        }
    }

    /// `SUB dst, src` — subtract two 64-bit registers (`dst -= src`).
    ///
    /// Encoding: `REX.W 2B /r` (SUB r64, r/m64).
    pub fn sub_rr(&mut self, dst: Reg64, src: Reg64) {
        self.emit_rex_wrb(dst, src);
        self.buf.push(0x2B);
        self.emit_modrm_rr(dst, src);
    }

    /// `SUB dst, imm` — subtract a sign-extended 32-bit immediate from a
    /// register.
    ///
    /// Uses the short `REX.W 83 /5 imm8` form when `imm` fits in a signed
    /// byte.
    pub fn sub_ri(&mut self, dst: Reg64, imm: i32) {
        self.emit_rex_wrb(Reg64::Rax, dst);
        if (i8::MIN as i32..=i8::MAX as i32).contains(&imm) {
            self.buf.push(0x83);
            self.emit_modrm_digit(5, dst);
            self.buf.push(imm as i8 as u8);
        } else {
            self.buf.push(0x81);
            self.emit_modrm_digit(5, dst);
            self.emit_i32(imm);
        }
    }

    // ── 32-bit arithmetic (for overflow-checked Smi ops) ────────────────────

    /// `ADD r32, r32` — add two 32-bit registers (`dst += src`), setting OF.
    ///
    /// Encoding: `[REX] 03 /r` (ADD r32, r/m32) — no REX.W bit.
    pub fn add32_rr(&mut self, dst: Reg64, src: Reg64) {
        self.emit_rex_rb_if_needed(dst, src);
        self.buf.push(0x03);
        self.emit_modrm_rr(dst, src);
    }

    /// `ADD r32, imm` — add immediate to a 32-bit register, setting OF.
    pub fn add32_ri(&mut self, dst: Reg64, imm: i32) {
        self.emit_rex_rb_if_needed(Reg64::Rax, dst);
        if (i8::MIN as i32..=i8::MAX as i32).contains(&imm) {
            self.buf.push(0x83);
            self.emit_modrm_digit(0, dst);
            self.buf.push(imm as i8 as u8);
        } else {
            self.buf.push(0x81);
            self.emit_modrm_digit(0, dst);
            self.emit_i32(imm);
        }
    }

    /// `SUB r32, r32` — subtract two 32-bit registers (`dst -= src`), setting
    /// OF.
    ///
    /// Encoding: `[REX] 2B /r` (SUB r32, r/m32) — no REX.W bit.
    pub fn sub32_rr(&mut self, dst: Reg64, src: Reg64) {
        self.emit_rex_rb_if_needed(dst, src);
        self.buf.push(0x2B);
        self.emit_modrm_rr(dst, src);
    }

    /// `SUB r32, imm` — subtract immediate from a 32-bit register, setting
    /// OF.
    pub fn sub32_ri(&mut self, dst: Reg64, imm: i32) {
        self.emit_rex_rb_if_needed(Reg64::Rax, dst);
        if (i8::MIN as i32..=i8::MAX as i32).contains(&imm) {
            self.buf.push(0x83);
            self.emit_modrm_digit(5, dst);
            self.buf.push(imm as i8 as u8);
        } else {
            self.buf.push(0x81);
            self.emit_modrm_digit(5, dst);
            self.emit_i32(imm);
        }
    }

    /// `IMUL r32, r32` — signed 32-bit multiply (`dst *= src`), setting OF.
    ///
    /// Encoding: `[REX] 0F AF /r` (IMUL r32, r/m32) — no REX.W bit.
    pub fn imul32_rr(&mut self, dst: Reg64, src: Reg64) {
        self.emit_rex_rb_if_needed(dst, src);
        self.buf.push(0x0F);
        self.buf.push(0xAF);
        self.emit_modrm_rr(dst, src);
    }

    /// `IMUL r32, r32, imm` — three-operand signed 32-bit multiply, setting
    /// OF.
    pub fn imul32_rri(&mut self, dst: Reg64, src: Reg64, imm: i32) {
        self.emit_rex_rb_if_needed(dst, src);
        if (i8::MIN as i32..=i8::MAX as i32).contains(&imm) {
            self.buf.push(0x6B);
            self.emit_modrm_rr(dst, src);
            self.buf.push(imm as i8 as u8);
        } else {
            self.buf.push(0x69);
            self.emit_modrm_rr(dst, src);
            self.emit_i32(imm);
        }
    }

    /// `MOVSXD dst, src` — sign-extend 32-bit register into 64-bit.
    ///
    /// Needed after 32-bit arithmetic to restore the 64-bit value for
    /// subsequent 64-bit operations.
    pub fn movsxd_sign_extend(&mut self, dst: Reg64, src: Reg64) {
        self.emit_rex_wrb(dst, src);
        self.buf.push(0x63);
        self.emit_modrm_rr(dst, src);
    }

    // ── Comparison ───────────────────────────────────────────────────────────

    /// `CMP lhs, rhs` — set CPU flags for `lhs − rhs` without storing the
    /// result.
    ///
    /// Encoding: `REX.W 3B /r` (CMP r64, r/m64).
    pub fn cmp_rr(&mut self, lhs: Reg64, rhs: Reg64) {
        self.emit_rex_wrb(lhs, rhs);
        self.buf.push(0x3B);
        self.emit_modrm_rr(lhs, rhs);
    }

    /// `CMP r32, r32` — 32-bit compare, setting flags based on the lower
    /// 32 bits only (no REX.W prefix).
    ///
    /// Encoding: `[REX] 3B /r` (CMP r32, r/m32).
    pub fn cmp32_rr(&mut self, lhs: Reg64, rhs: Reg64) {
        self.emit_rex_rb_if_needed(lhs, rhs);
        self.buf.push(0x3B);
        self.emit_modrm_rr(lhs, rhs);
    }

    /// `CMP reg, imm` — compare a register against a sign-extended 32-bit
    /// immediate.
    ///
    /// Uses the short `REX.W 83 /7 imm8` form when `imm` fits in a signed
    /// byte.
    pub fn cmp_ri(&mut self, reg: Reg64, imm: i32) {
        self.emit_rex_wrb(Reg64::Rax, reg);
        if (i8::MIN as i32..=i8::MAX as i32).contains(&imm) {
            self.buf.push(0x83);
            self.emit_modrm_digit(7, reg);
            self.buf.push(imm as i8 as u8);
        } else {
            self.buf.push(0x81);
            self.emit_modrm_digit(7, reg);
            self.emit_i32(imm);
        }
    }

    /// `CMP r32, imm` — 32-bit compare register against immediate (no
    /// REX.W prefix).
    ///
    /// Encoding: `[REX] 83 /7 imm8` or `[REX] 81 /7 imm32`.
    pub fn cmp32_ri(&mut self, reg: Reg64, imm: i32) {
        self.emit_rex_rb_if_needed(Reg64::Rax, reg);
        if (i8::MIN as i32..=i8::MAX as i32).contains(&imm) {
            self.buf.push(0x83);
            self.emit_modrm_digit(7, reg);
            self.buf.push(imm as i8 as u8);
        } else {
            self.buf.push(0x81);
            self.emit_modrm_digit(7, reg);
            self.emit_i32(imm);
        }
    }

    /// `CMP reg, [base + disp32]` — compare a 64-bit register against a
    /// memory operand (sets flags; result is discarded).
    ///
    /// Encoding: `REX.W [REX.R] [REX.B] 3B /r` with ModRM mod=10.
    pub fn cmp_rm(&mut self, reg: Reg64, base: Reg64, disp: i32) {
        self.emit_rex_wrb(reg, base);
        self.buf.push(0x3B);
        self.emit_modrm_base_disp32(reg, base, disp);
    }

    /// `MOVSXD dst, src` — sign-extend the lower 32 bits of `src` into the
    /// 64-bit `dst`.
    ///
    /// Encoding: `REX.W 63 /r` (MOVSXD r64, r/m32).
    pub fn movsxd_rr(&mut self, dst: Reg64, src: Reg64) {
        self.emit_rex_wrb(dst, src);
        self.buf.push(0x63);
        self.emit_modrm_rr(dst, src);
    }

    /// `MOVSXD dst, DWORD PTR [base + disp32]` — load a 32-bit memory value
    /// and sign-extend it to 64 bits.
    ///
    /// Encoding: `REX.W [REX.R] [REX.B] 63 /r` with ModRM mod=10.
    pub fn movsxd_base_disp32(&mut self, dst: Reg64, base: Reg64, disp: i32) {
        self.emit_rex_wrb(dst, base);
        self.buf.push(0x63);
        self.emit_modrm_base_disp32(dst, base, disp);
    }

    // ── Control flow ─────────────────────────────────────────────────────────

    /// `JMP label` — unconditional near jump to a label.
    ///
    /// Encoding: `E9 rel32`.
    pub fn jmp(&mut self, label: &mut Label) {
        self.buf.push(0xE9);
        self.emit_rel32_for_label(label);
    }

    /// `JMP reg` — unconditional near jump through a register.
    ///
    /// Encoding: `[REX.B] FF /4`.
    pub fn jmp_reg(&mut self, reg: Reg64) {
        self.emit_rex_b_only(reg);
        self.buf.push(0xFF);
        self.emit_modrm_digit(4, reg);
    }

    /// `JE label` / `JZ label` — jump if equal (ZF=1).
    ///
    /// Encoding: `0F 84 rel32`.
    pub fn je(&mut self, label: &mut Label) {
        self.buf.push(0x0F);
        self.buf.push(0x84);
        self.emit_rel32_for_label(label);
    }

    /// `JNE label` / `JNZ label` — jump if not equal (ZF=0).
    ///
    /// Encoding: `0F 85 rel32`.
    pub fn jne(&mut self, label: &mut Label) {
        self.buf.push(0x0F);
        self.buf.push(0x85);
        self.emit_rel32_for_label(label);
    }

    /// `JO label` — jump if overflow (OF=1).
    ///
    /// Encoding: `0F 80 rel32`.
    pub fn jo(&mut self, label: &mut Label) {
        self.buf.push(0x0F);
        self.buf.push(0x80);
        self.emit_rel32_for_label(label);
    }

    /// `CALL reg` — call an indirect target through a register.
    ///
    /// Encoding: `[REX.B] FF /2`.
    pub fn call_reg(&mut self, reg: Reg64) {
        self.emit_rex_b_only(reg);
        self.buf.push(0xFF);
        self.emit_modrm_digit(2, reg);
    }

    /// `CALL label` — PC-relative call to a label.
    ///
    /// Encoding: `E8 rel32`.
    pub fn call_rel(&mut self, label: &mut Label) {
        self.buf.push(0xE8);
        self.emit_rel32_for_label(label);
    }

    /// `RET` — near return from procedure.
    ///
    /// Encoding: `C3`.
    pub fn ret(&mut self) {
        self.buf.push(0xC3);
    }

    // ── Stack operations ─────────────────────────────────────────────────────

    /// `PUSH reg` — push a 64-bit register onto the stack.
    ///
    /// 64-bit operand size is the default in 64-bit mode, so no REX.W is
    /// needed.  REX.B is emitted only for R8–R15.
    ///
    /// Encoding: `[REX.B] 50+rd`.
    pub fn push(&mut self, reg: Reg64) {
        self.emit_rex_b_only(reg);
        self.buf.push(0x50 | reg.enc());
    }

    /// `POP reg` — pop a 64-bit value from the stack into a register.
    ///
    /// Encoding: `[REX.B] 58+rd`.
    pub fn pop(&mut self, reg: Reg64) {
        self.emit_rex_b_only(reg);
        self.buf.push(0x58 | reg.enc());
    }

    // ── Extended arithmetic ──────────────────────────────────────────────────

    /// `XOR dst, src` — bitwise exclusive-or of two 64-bit registers.
    ///
    /// Encoding: `REX.W 33 /r` (XOR r64, r/m64).
    pub fn xor_rr(&mut self, dst: Reg64, src: Reg64) {
        self.emit_rex_wrb(dst, src);
        self.buf.push(0x33);
        self.emit_modrm_rr(dst, src);
    }

    /// `XOR dst, imm` — bitwise XOR with a sign-extended 32-bit immediate.
    ///
    /// Uses the 4-byte `REX.W 83 /6 imm8` form when `imm` fits in a signed
    /// byte, and the 7-byte `REX.W 81 /6 imm32` form otherwise.
    pub fn xor_ri(&mut self, dst: Reg64, imm: i32) {
        self.emit_rex_wrb(Reg64::Rax, dst);
        if (i8::MIN as i32..=i8::MAX as i32).contains(&imm) {
            self.buf.push(0x83);
            self.emit_modrm_digit(6, dst);
            self.buf.push(imm as i8 as u8);
        } else {
            self.buf.push(0x81);
            self.emit_modrm_digit(6, dst);
            self.emit_i32(imm);
        }
    }

    /// `OR dst, src` — bitwise OR of two 64-bit registers (`dst |= src`).
    ///
    /// Encoding: `REX.W 0B /r` (OR r64, r/m64).
    pub fn or_rr(&mut self, dst: Reg64, src: Reg64) {
        self.emit_rex_wrb(dst, src);
        self.buf.push(0x0B);
        self.emit_modrm_rr(dst, src);
    }

    /// `OR dst, imm` — bitwise OR with a sign-extended 32-bit immediate.
    ///
    /// Uses the 4-byte `REX.W 83 /1 imm8` form when `imm` fits in a signed
    /// byte, and the 7-byte `REX.W 81 /1 imm32` form otherwise.
    pub fn or_ri(&mut self, dst: Reg64, imm: i32) {
        self.emit_rex_wrb(Reg64::Rax, dst);
        if (i8::MIN as i32..=i8::MAX as i32).contains(&imm) {
            self.buf.push(0x83);
            self.emit_modrm_digit(1, dst);
            self.buf.push(imm as i8 as u8);
        } else {
            self.buf.push(0x81);
            self.emit_modrm_digit(1, dst);
            self.emit_i32(imm);
        }
    }

    /// `AND dst, src` — bitwise AND of two 64-bit registers (`dst &= src`).
    ///
    /// Encoding: `REX.W 23 /r` (AND r64, r/m64).
    pub fn and_rr(&mut self, dst: Reg64, src: Reg64) {
        self.emit_rex_wrb(dst, src);
        self.buf.push(0x23);
        self.emit_modrm_rr(dst, src);
    }

    /// `AND dst, imm` — bitwise AND with a sign-extended 32-bit immediate.
    ///
    /// Uses the 4-byte `REX.W 83 /4 imm8` form when `imm` fits in a signed
    /// byte, and the 7-byte `REX.W 81 /4 imm32` form otherwise.
    pub fn and_ri(&mut self, dst: Reg64, imm: i32) {
        self.emit_rex_wrb(Reg64::Rax, dst);
        if (i8::MIN as i32..=i8::MAX as i32).contains(&imm) {
            self.buf.push(0x83);
            self.emit_modrm_digit(4, dst);
            self.buf.push(imm as i8 as u8);
        } else {
            self.buf.push(0x81);
            self.emit_modrm_digit(4, dst);
            self.emit_i32(imm);
        }
    }

    /// `NOT dst` — bitwise NOT (one's complement) of a 64-bit register.
    ///
    /// Encoding: `REX.W F7 /2`.
    pub fn not_r(&mut self, dst: Reg64) {
        self.emit_rex_wrb(Reg64::Rax, dst);
        self.buf.push(0xF7);
        self.emit_modrm_digit(2, dst);
    }

    /// `IMUL dst, src` — signed multiply, two-operand form (`dst *= src`).
    ///
    /// Encoding: `REX.W 0F AF /r` (IMUL r64, r/m64).
    pub fn imul_rr(&mut self, dst: Reg64, src: Reg64) {
        self.emit_rex_wrb(dst, src);
        self.buf.push(0x0F);
        self.buf.push(0xAF);
        self.emit_modrm_rr(dst, src);
    }

    /// `IMUL dst, src, imm` — three-operand signed multiply with immediate.
    ///
    /// Uses the short `REX.W 6B /r imm8` form when `imm` fits in a signed
    /// byte, and the `REX.W 69 /r imm32` form otherwise.
    pub fn imul_rri(&mut self, dst: Reg64, src: Reg64, imm: i32) {
        self.emit_rex_wrb(dst, src);
        if (i8::MIN as i32..=i8::MAX as i32).contains(&imm) {
            self.buf.push(0x6B);
            self.emit_modrm_rr(dst, src);
            self.buf.push(imm as i8 as u8);
        } else {
            self.buf.push(0x69);
            self.emit_modrm_rr(dst, src);
            self.emit_i32(imm);
        }
    }

    /// `NEG dst` — two's-complement negation of a 64-bit register.
    ///
    /// Encoding: `REX.W F7 /3`.
    pub fn neg_r(&mut self, dst: Reg64) {
        self.emit_rex_wrb(Reg64::Rax, dst);
        self.buf.push(0xF7);
        self.emit_modrm_digit(3, dst);
    }

    /// `TEST lhs, rhs` — set CPU flags for `lhs & rhs` without storing the
    /// result.
    ///
    /// Encoding: `REX.W 85 /r` (TEST r/m64, r64).
    pub fn test_rr(&mut self, lhs: Reg64, rhs: Reg64) {
        self.emit_rex_wrb(rhs, lhs);
        self.buf.push(0x85);
        self.emit_modrm_rr(rhs, lhs);
    }

    // ── Memory access (base + disp32) ────────────────────────────────────────

    /// Emit ModRM (mod=10) plus an optional SIB byte, then a 32-bit
    /// displacement.
    ///
    /// When `base.enc() == 4` (RSP or R12), a SIB byte of `0x24` is required
    /// (scale=0, index=no-index, base=4).
    fn emit_modrm_base_disp32(&mut self, reg_field: Reg64, base: Reg64, disp: i32) {
        if base.enc() == 4 {
            // r/m=4 signals SIB; REX.B extends SIB.base (not ModRM.r/m).
            self.buf.push(0x80 | (reg_field.enc() << 3) | 4);
            self.buf.push(0x24); // SIB: scale=0, index=4(none), base=4
        } else {
            self.buf.push(0x80 | (reg_field.enc() << 3) | base.enc());
        }
        self.emit_i32(disp);
    }

    /// `MOV dst, [base + disp32]` — load a 64-bit value from memory.
    ///
    /// Encoding: `REX.W [REX.R] [REX.B] 8B /r` with ModRM mod=10.
    pub fn mov_load_base_disp32(&mut self, dst: Reg64, base: Reg64, disp: i32) {
        self.emit_rex_wrb(dst, base);
        self.buf.push(0x8B);
        self.emit_modrm_base_disp32(dst, base, disp);
    }

    /// `LEA dst, [base + disp32]` — compute effective address.
    ///
    /// Encoding: `REX.W [REX.R] [REX.B] 8D /r` with ModRM mod=10.
    pub fn lea_base_disp32(&mut self, dst: Reg64, base: Reg64, disp: i32) {
        self.emit_rex_wrb(dst, base);
        self.buf.push(0x8D);
        self.emit_modrm_base_disp32(dst, base, disp);
    }

    /// `MOV [base + disp32], src` — store a 64-bit register to memory.
    ///
    /// Encoding: `REX.W [REX.R] [REX.B] 89 /r` with ModRM mod=10.
    pub fn mov_store_base_disp32(&mut self, base: Reg64, disp: i32, src: Reg64) {
        self.emit_rex_wrb(src, base);
        self.buf.push(0x89);
        self.emit_modrm_base_disp32(src, base, disp);
    }

    // ── Conditional instructions ─────────────────────────────────────────────

    /// `SETCC AL` — set `AL` to `1` if the condition is satisfied, `0`
    /// otherwise.
    ///
    /// Encoding: `0F 9x C0` (no REX needed for `AL`).
    pub fn setcc_al(&mut self, cc: CondCode) {
        self.buf.push(0x0F);
        self.buf.push(cc.setcc_byte());
        // ModRM: mod=11, reg=0 (opcode extension), r/m=0 (AL).
        self.buf.push(0xC0);
    }

    /// `MOVZX dst, AL` — zero-extend `AL` (byte) into a 64-bit register.
    ///
    /// Encoding: `REX.W [REX.R] 0F B6 /r` with `r/m=0` (AL).
    pub fn movzx_r64_al(&mut self, dst: Reg64) {
        let r_bit = if dst.needs_rex() { 0x04 } else { 0 };
        self.buf.push(0x48 | r_bit); // REX.W [+ REX.R]
        self.buf.push(0x0F);
        self.buf.push(0xB6);
        // ModRM: mod=11, reg=dst.enc(), r/m=0 (AL).
        self.buf.push(0xC0 | (dst.enc() << 3));
    }

    /// `MOVZX dst, BYTE PTR [base + disp32]` — zero-extend a memory byte
    /// into a 64-bit register.
    ///
    /// Encoding: `REX.W [REX.R] [REX.B] 0F B6 /r` with ModRM mod=10.
    pub fn movzx_byte_base_disp32(&mut self, dst: Reg64, base: Reg64, disp: i32) {
        self.emit_rex_wrb(dst, base);
        self.buf.push(0x0F);
        self.buf.push(0xB6);
        self.emit_modrm_base_disp32(dst, base, disp);
    }

    /// `Jcc label` — conditional near jump (32-bit displacement).
    ///
    /// Encoding: `0F 8x rel32`.
    pub fn jcc(&mut self, cc: CondCode, label: &mut Label) {
        self.buf.push(0x0F);
        self.buf.push(cc.jcc_byte());
        self.emit_rel32_for_label(label);
    }

    /// `REP STOSQ` — repeat store quadword.
    ///
    /// Zeros memory starting at `[RDI]` for `RCX` qwords (8 bytes each).
    /// Assumes `RAX = 0` and `RCX` contains the count and `RDI` points to destination.
    /// After execution, `RDI` points past the zeroed region and `RCX = 0`.
    ///
    /// Encoding: `F3 48 AB` (REP prefix + REX.W + STOSQ opcode).
    pub fn rep_stosq(&mut self) {
        self.buf.push(0xF3); // REP prefix
        self.buf.push(0x48); // REX.W
        self.buf.push(0xAB); // STOSQ opcode
    }

    // ── Low-level byte access (crate-internal) ────────────────────────────────

    /// Append a single raw byte to the code buffer.
    ///
    /// Used by higher-level instruction encoders that require precise control
    /// over the emitted bytes (e.g., `AND r64, r/m64` and `OR r64, r/m64`
    /// which are not included in the standard instruction set of this
    /// assembler).
    pub(crate) fn emit_byte(&mut self, b: u8) {
        self.buf.push(b);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Byte-level encoding tests ─────────────────────────────────────────────

    #[test]
    fn test_ret_encoding() {
        let mut m = MacroAssembler::new();
        m.ret();
        assert_eq!(m.code(), &[0xC3]);
    }

    #[test]
    fn test_push_classic_registers() {
        // RAX  → 50
        // RDI  → 57  (0x50 + 7)
        let mut m = MacroAssembler::new();
        m.push(Reg64::Rax);
        m.push(Reg64::Rdi);
        assert_eq!(m.code(), &[0x50, 0x57]);
    }

    #[test]
    fn test_push_extended_registers() {
        // R8  → 41 50
        // R15 → 41 57  (0x50 + 7)
        let mut m = MacroAssembler::new();
        m.push(Reg64::R8);
        m.push(Reg64::R15);
        assert_eq!(m.code(), &[0x41, 0x50, 0x41, 0x57]);
    }

    #[test]
    fn test_pop_encoding() {
        // RAX → 58
        // R15 → 41 5F  (0x58 + 7)
        let mut m = MacroAssembler::new();
        m.pop(Reg64::Rax);
        m.pop(Reg64::R15);
        assert_eq!(m.code(), &[0x58, 0x41, 0x5F]);
    }

    #[test]
    fn test_mov_rr_encoding() {
        // mov rax, rdi  →  48 8B C7
        let mut m = MacroAssembler::new();
        m.mov_rr(Reg64::Rax, Reg64::Rdi);
        assert_eq!(m.code(), &[0x48, 0x8B, 0xC7]);
    }

    #[test]
    fn test_mov_rr_extended_src() {
        // mov rax, r9  →  49 8B C1  (REX.B for r9)
        let mut m = MacroAssembler::new();
        m.mov_rr(Reg64::Rax, Reg64::R9);
        assert_eq!(m.code(), &[0x49, 0x8B, 0xC1]);
    }

    #[test]
    fn test_mov_rr_both_extended() {
        // mov r11, r9  →  4D 8B D9  (REX.W|R|B)
        let mut m = MacroAssembler::new();
        m.mov_rr(Reg64::R11, Reg64::R9);
        assert_eq!(m.code(), &[0x4D, 0x8B, 0xD9]);
    }

    #[test]
    fn test_mov_ri_small_imm() {
        // mov rdi, 42  →  48 C7 C7 2A 00 00 00  (C7/0, sign-ext imm32)
        let mut m = MacroAssembler::new();
        m.mov_ri(Reg64::Rdi, 42);
        assert_eq!(m.code(), &[0x48, 0xC7, 0xC7, 0x2A, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_mov_ri_large_imm() {
        // mov rax, 0x1_2345_6789  →  48 B8 89 67 45 23 01 00 00 00
        let mut m = MacroAssembler::new();
        m.mov_ri(Reg64::Rax, 0x1_2345_6789_i64);
        assert_eq!(
            m.code(),
            &[0x48, 0xB8, 0x89, 0x67, 0x45, 0x23, 0x01, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn test_mov_ri_extended_reg_large_imm() {
        // mov r9, 0x1_2345_6789  →  49 B9 89 67 45 23 01 00 00 00
        let mut m = MacroAssembler::new();
        m.mov_ri(Reg64::R9, 0x1_2345_6789_i64);
        assert_eq!(
            m.code(),
            &[0x49, 0xB9, 0x89, 0x67, 0x45, 0x23, 0x01, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn test_add_rr_encoding() {
        // add rax, rsi  →  48 03 C6
        let mut m = MacroAssembler::new();
        m.add_rr(Reg64::Rax, Reg64::Rsi);
        assert_eq!(m.code(), &[0x48, 0x03, 0xC6]);
    }

    #[test]
    fn test_add_ri_short_form() {
        // add rax, 5  →  48 83 C0 05
        let mut m = MacroAssembler::new();
        m.add_ri(Reg64::Rax, 5);
        assert_eq!(m.code(), &[0x48, 0x83, 0xC0, 0x05]);
    }

    #[test]
    fn test_add_ri_long_form() {
        // add rax, 1000  →  48 81 C0 E8 03 00 00
        let mut m = MacroAssembler::new();
        m.add_ri(Reg64::Rax, 1000);
        assert_eq!(m.code(), &[0x48, 0x81, 0xC0, 0xE8, 0x03, 0x00, 0x00]);
    }

    #[test]
    fn test_sub_rr_encoding() {
        // sub rax, rsi  →  48 2B C6
        let mut m = MacroAssembler::new();
        m.sub_rr(Reg64::Rax, Reg64::Rsi);
        assert_eq!(m.code(), &[0x48, 0x2B, 0xC6]);
    }

    #[test]
    fn test_sub_ri_short_form() {
        // sub rsp, 32  →  48 83 EC 20
        let mut m = MacroAssembler::new();
        m.sub_ri(Reg64::Rsp, 32);
        assert_eq!(m.code(), &[0x48, 0x83, 0xEC, 0x20]);
    }

    #[test]
    fn test_cmp_rr_encoding() {
        // cmp rax, rcx  →  48 3B C1
        let mut m = MacroAssembler::new();
        m.cmp_rr(Reg64::Rax, Reg64::Rcx);
        assert_eq!(m.code(), &[0x48, 0x3B, 0xC1]);
    }

    #[test]
    fn test_cmp_ri_short_form() {
        // cmp rcx, 0  →  48 83 F9 00
        let mut m = MacroAssembler::new();
        m.cmp_ri(Reg64::Rcx, 0);
        assert_eq!(m.code(), &[0x48, 0x83, 0xF9, 0x00]);
    }

    #[test]
    fn test_jmp_reg_encoding() {
        // jmp rax  →  FF E0
        // jmp r8   →  41 FF E0
        let mut m = MacroAssembler::new();
        m.jmp_reg(Reg64::Rax);
        m.jmp_reg(Reg64::R8);
        assert_eq!(m.code(), &[0xFF, 0xE0, 0x41, 0xFF, 0xE0]);
    }

    #[test]
    fn test_call_reg_encoding() {
        // call rax  →  FF D0
        let mut m = MacroAssembler::new();
        m.call_reg(Reg64::Rax);
        assert_eq!(m.code(), &[0xFF, 0xD0]);
    }

    #[test]
    fn test_rep_stosq_encoding() {
        // rep stosq  →  F3 48 AB
        let mut m = MacroAssembler::new();
        m.rep_stosq();
        assert_eq!(m.code(), &[0xF3, 0x48, 0xAB]);
    }

    #[test]
    fn test_lea_rip_rel_encoding() {
        // lea rax, [rip+0]  →  48 8D 05 00 00 00 00
        let mut m = MacroAssembler::new();
        m.lea_rip_rel(Reg64::Rax, 0);
        assert_eq!(m.code(), &[0x48, 0x8D, 0x05, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_lea_scaled_times3() {
        // lea rax, [rcx + rcx*2]  →  48 8D 04 49
        // REX.W=0x48, LEA=0x8D, ModRM=00.000.100(SIB), SIB=01.001.001
        let mut m = MacroAssembler::new();
        m.lea_scaled(Reg64::Rax, Reg64::Rcx, Reg64::Rcx, 2);
        assert_eq!(m.code(), &[0x48, 0x8D, 0x04, 0x49]);
    }

    #[test]
    fn test_lea_scaled_times5() {
        // lea rdx, [rsi + rsi*4]  →  48 8D 14 B6
        // REX.W=0x48, LEA=0x8D, ModRM=00.010.100(SIB), SIB=10.110.110
        let mut m = MacroAssembler::new();
        m.lea_scaled(Reg64::Rdx, Reg64::Rsi, Reg64::Rsi, 4);
        assert_eq!(m.code(), &[0x48, 0x8D, 0x14, 0xB6]);
    }

    #[test]
    fn test_lea_scaled_with_extended_regs() {
        // lea r8, [r9 + r9*8]  →  4F 8D 04 C9
        // REX.W+R+X+B=0x4F, LEA=0x8D, ModRM=00.000.100(SIB), SIB=11.001.001
        let mut m = MacroAssembler::new();
        m.lea_scaled(Reg64::R8, Reg64::R9, Reg64::R9, 8);
        assert_eq!(m.code(), &[0x4F, 0x8D, 0x04, 0xC9]);
    }

    // ── Label / branch tests ──────────────────────────────────────────────────

    #[test]
    fn test_forward_jmp_patched_correctly() {
        // Emit:
        //   jmp done        (5 bytes: E9 + rel32)
        //   mov rax, 1      (7 bytes: REX.W C7 /0 + imm32) — skipped
        // done:
        //   ret             (1 byte)
        let mut m = MacroAssembler::new();
        let mut done = Label::new();

        m.jmp(&mut done);
        m.mov_ri(Reg64::Rax, 1);
        m.bind_label(&mut done);
        m.ret();

        // jmp target = offset 12 (5 + 7).
        // rel32 = 12 - (1 + 4) = 7.
        let code = m.code();
        assert_eq!(code[0], 0xE9);
        let rel32 = i32::from_le_bytes([code[1], code[2], code[3], code[4]]);
        assert_eq!(rel32, 7);
        assert!(done.is_bound());
        assert_eq!(done.bound_offset(), Some(12));
    }

    #[test]
    fn test_backward_jne_loop() {
        // Emit:
        //   loop_top:
        //     sub rcx, 1
        //     jne loop_top
        //   ret
        let mut m = MacroAssembler::new();
        let mut loop_top = Label::new();

        m.bind_label(&mut loop_top); // offset 0
        m.sub_ri(Reg64::Rcx, 1); // 4 bytes: 48 83 E9 01
        m.jne(&mut loop_top); // 6 bytes: 0F 85 + rel32

        // After sub_ri (4 bytes) + jne opcode bytes (2: 0F 85), rel32_start = 6.
        // rel32 = 0 - (6 + 4) = -10.
        let code = m.code();
        assert_eq!(&code[0..2], &[0x48, 0x83]); // REX.W + 83
        assert_eq!(code[4], 0x0F);
        assert_eq!(code[5], 0x85);
        let rel32 = i32::from_le_bytes([code[6], code[7], code[8], code[9]]);
        assert_eq!(rel32, -10);
    }

    #[test]
    fn test_forward_je_patched_correctly() {
        // je target (6 bytes: 0F 84 + rel32)
        // nop sequence (1 byte: ret used as placeholder)
        // target:
        let mut m = MacroAssembler::new();
        let mut target = Label::new();

        m.je(&mut target); // 6 bytes
        m.ret(); // 1 byte (placeholder)
        m.bind_label(&mut target);

        // target = offset 7, rel32_start = 2
        // rel32 = 7 - (2 + 4) = 1
        let code = m.code();
        let rel32 = i32::from_le_bytes([code[2], code[3], code[4], code[5]]);
        assert_eq!(rel32, 1);
    }

    #[test]
    fn test_call_rel_forward_patched() {
        // call fn_label (5 bytes: E8 + rel32)
        // ret            (1 byte)
        // fn_label:
        //   ret
        let mut m = MacroAssembler::new();
        let mut fn_label = Label::new();

        m.call_rel(&mut fn_label);
        m.ret();
        m.bind_label(&mut fn_label);
        m.ret();

        // fn_label = offset 6, rel32_start = 1
        // rel32 = 6 - (1 + 4) = 1
        let code = m.code();
        assert_eq!(code[0], 0xE8);
        let rel32 = i32::from_le_bytes([code[1], code[2], code[3], code[4]]);
        assert_eq!(rel32, 1);
    }

    #[test]
    fn test_emit_add_function_bytes() {
        // fn add(a: i64, b: i64) -> i64 { a + b }
        // SysV AMD64: a in RDI, b in RSI, return in RAX
        //   mov rax, rdi  →  48 8B C7
        //   add rax, rsi  →  48 03 C6
        //   ret           →  C3
        let mut m = MacroAssembler::new();
        m.mov_rr(Reg64::Rax, Reg64::Rdi);
        m.add_rr(Reg64::Rax, Reg64::Rsi);
        m.ret();
        assert_eq!(m.code(), &[0x48, 0x8B, 0xC7, 0x48, 0x03, 0xC6, 0xC3]);
    }

    #[test]
    fn test_reg64_display() {
        assert_eq!(Reg64::Rax.to_string(), "rax");
        assert_eq!(Reg64::Rdi.to_string(), "rdi");
        assert_eq!(Reg64::R8.to_string(), "r8");
        assert_eq!(Reg64::R15.to_string(), "r15");
    }

    #[test]
    fn test_position_tracks_offset() {
        let mut m = MacroAssembler::new();
        assert_eq!(m.position(), 0);
        m.ret();
        assert_eq!(m.position(), 1);
        m.push(Reg64::Rax);
        assert_eq!(m.position(), 2);
    }

    // ── FFI execution test (x86-64 + Unix only) ───────────────────────────────

    /// Call emitted machine code via a raw function pointer.
    ///
    /// This test allocates a page of read-write-execute memory with `mmap`,
    /// copies the emitted bytes into it, then invokes the resulting function
    /// pointer through Rust's `extern "C"` FFI.
    #[cfg(all(target_arch = "x86_64", unix))]
    #[test]
    fn test_emit_add_and_call_via_ffi() {
        // Emit: fn add(a: i64, b: i64) -> i64 { a + b }
        let mut masm = MacroAssembler::new();
        masm.mov_rr(Reg64::Rax, Reg64::Rdi);
        masm.add_rr(Reg64::Rax, Reg64::Rsi);
        masm.ret();

        let code = masm.into_code();
        let size = code.len();

        // Allocate a page of RWX memory and copy the code in.
        // SAFETY: We pass valid arguments to mmap and check the return value.
        let mem = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(mem, libc::MAP_FAILED, "mmap failed");

        // SAFETY:
        // - `mem` is a valid, non-null, page-aligned pointer to `size` bytes of
        //   RWX memory returned by mmap.
        // - We copy exactly `size` bytes of correctly-encoded x86-64 machine
        //   code into it.
        // - We transmute the pointer to an `extern "C"` function type whose
        //   signature matches the emitted calling convention (SysV AMD64).
        // - We unmap the memory after the last call.
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), mem.cast::<u8>(), size);
            let f: extern "C" fn(i64, i64) -> i64 = std::mem::transmute(mem);
            assert_eq!(f(3, 4), 7);
            assert_eq!(f(-1, 1), 0);
            assert_eq!(f(100, -50), 50);
            assert_eq!(f(0, 0), 0);
            assert_eq!(f(i64::MAX, 0), i64::MAX);
            libc::munmap(mem, size);
        }
    }
}
