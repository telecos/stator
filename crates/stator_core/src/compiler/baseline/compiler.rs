//! Baseline (non-optimising) JIT compiler.
//!
//! Translates a [`BytecodeArray`] to x86-64 machine code by compiling each
//! bytecode instruction to a fixed sequence of native instructions.
//!
//! # Design
//!
//! ## JIT value representation
//!
//! Inside JIT code all values are `i64`.  The encoding is:
//!
//! | JavaScript value | `i64` representation |
//! |------------------|----------------------|
//! | `Smi(v)`         | `v as i64`           |
//! | `true`           | [`JIT_TRUE`]         |
//! | `false`          | [`JIT_FALSE`]        |
//! | `undefined`      | [`JIT_UNDEFINED`]    |
//! | `null`           | [`JIT_NULL`]         |
//!
//! Smi values occupy `i32::MIN..=i32::MAX` (≈ ±2.1 billion).
//! The special sentinels are all above `i32::MAX` and therefore disjoint.
//!
//! ## Register conventions
//!
//! | Register | Role |
//! |----------|------|
//! | `R12`    | accumulator (callee-saved) |
//! | `R14`    | register-file base pointer (`*mut i64`, callee-saved) |
//! | `R11`    | scratch (caller-saved) |
//!
//! ## JIT function signature
//!
//! ```text
//! extern "C" fn(regs: *mut i64) -> i64
//! ```
//!
//! `regs` points to an array of `parameter_count + frame_size` `i64` slots.
//! The caller pre-initialises parameter slots; the function returns the
//! accumulator value on normal completion or [`JIT_DEOPT`] if it encounters
//! a bytecode it cannot handle.
//!
//! ## Register-file slot layout
//!
//! Slot indices follow the interpreter's convention:
//!
//! ```text
//! [ param[0], param[1], …, local[0], local[1], … ]
//!  ^------ parameter_count ------^^---- frame_size ----^
//! ```
//!
//! For a bytecode operand `v` (encoded as `u32`):
//! - `v as i32 ≥ 0`: flat index = `parameter_count + (v as usize)`.
//! - `v as i32 < 0`: flat index = `-(v as i32 + 1) as usize` (parameter).

use crate::bytecode::bytecode_array::{BytecodeArray, ConstantPoolEntry};
use crate::bytecode::bytecodes::{Instruction, Opcode, Operand, decode_with_byte_offsets};
use crate::compiler::baseline::masm_x64::{CondCode, Label, MacroAssembler, Reg64};
use crate::error::{StatorError, StatorResult};

// ─────────────────────────────────────────────────────────────────────────────
// Table serialization constants
// ─────────────────────────────────────────────────────────────────────────────

/// Magic number written in the last 4 bytes of the code buffer to mark the
/// presence of serialized safepoint and deopt tables.  ASCII `STAT` in
/// little-endian order.
pub const METADATA_MAGIC: u32 = 0x5441_5453;

/// Size in bytes of the fixed metadata footer appended at the very end of the
/// code buffer (3 × `u32` = 12 bytes).
pub const FOOTER_SIZE: usize = 12;

/// Serialized size of a single [`SafepointEntry`] in bytes
/// (`code_offset u32` + `bytecode_index u32` + `gc_map u64` = 16 bytes).
const SAFEPOINT_ENTRY_SIZE: usize = 16;

/// Serialized size of a single [`DeoptEntry`] in bytes
/// (`code_offset u32` + `bytecode_offset u32` + `liveness_map u64` = 16 bytes).
const DEOPT_ENTRY_SIZE: usize = 16;

// ─────────────────────────────────────────────────────────────────────────────
// JIT value sentinels
// ─────────────────────────────────────────────────────────────────────────────

/// JIT representation of `false`.
///
/// Chosen to be just above `i32::MAX` so it cannot be confused with any Smi.
pub const JIT_FALSE: i64 = 0x1_0000_0000_i64;
/// JIT representation of `true` (`JIT_FALSE + 1`).
pub const JIT_TRUE: i64 = 0x1_0000_0001_i64;
/// JIT representation of `undefined`.
pub const JIT_UNDEFINED: i64 = 0x1_0000_0002_i64;
/// JIT representation of `null`.
pub const JIT_NULL: i64 = 0x1_0000_0003_i64;
/// Sentinel returned by the JIT function when it encounters an unsupported
/// bytecode and must fall back to the interpreter.
pub const JIT_DEOPT: i64 = i64::MIN;

/// Convert a JIT `i64` value back to a [`crate::objects::value::JsValue`].
///
/// Returns `None` if `v` is not a recognised sentinel and does not fit in
/// `i32` (i.e. is an out-of-range or deopt value).
pub fn jit_to_jsvalue(v: i64) -> Option<crate::objects::value::JsValue> {
    use crate::objects::value::JsValue;
    if v == JIT_FALSE {
        Some(JsValue::Boolean(false))
    } else if v == JIT_TRUE {
        Some(JsValue::Boolean(true))
    } else if v == JIT_UNDEFINED {
        Some(JsValue::Undefined)
    } else if v == JIT_NULL {
        Some(JsValue::Null)
    } else if v >= i32::MIN as i64 && v <= i32::MAX as i64 {
        Some(JsValue::Smi(v as i32))
    } else {
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Metadata types
// ─────────────────────────────────────────────────────────────────────────────

/// A single entry in the safepoint table.
///
/// A safepoint is any code location at which the garbage collector is allowed
/// to run.  The entry records the byte offset in the JIT code buffer and the
/// corresponding bytecode instruction index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafepointEntry {
    /// Byte offset in the JIT code buffer.
    pub code_offset: u32,
    /// Index of the bytecode instruction this safepoint corresponds to.
    pub bytecode_index: u32,
    /// Bitmask of register-file slots that hold GC-managed heap pointers at
    /// this safepoint.  Bit `i` set means slot `i` holds a pointer that must
    /// be reported to the garbage collector during stack scanning.
    /// Zero for the current Smi-only JIT tier (no heap-object values).
    pub gc_map: u64,
}

/// A single entry in the deoptimization table.
///
/// When a JIT compiled function encounters a value or operation it cannot
/// handle natively, it jumps to the deopt epilogue and returns [`JIT_DEOPT`].
/// The deopt table maps those code offsets back to bytecode offsets so that
/// the interpreter can resume from the correct point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeoptEntry {
    /// Byte offset in the JIT code buffer of the deopt point.
    pub code_offset: u32,
    /// Bytecode byte offset where interpretation should resume.
    pub bytecode_offset: u32,
    /// Bitmask of register-file slots that hold live values at this deopt
    /// point.  Bit `i` set means slot `i` must be preserved when
    /// reconstructing the interpreter frame.  Conservatively set to all slots
    /// live in the current baseline tier.
    pub liveness_map: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// CompiledCode
// ─────────────────────────────────────────────────────────────────────────────

/// The output of the [`BaselineCompiler`]: machine code bytes plus metadata.
pub struct CompiledCode {
    /// Emitted x86-64 machine code followed by the serialized safepoint and
    /// deopt tables and a 12-byte footer.  The footer is the last
    /// [`FOOTER_SIZE`] bytes and encodes the byte offsets of each table within
    /// `code`, plus [`METADATA_MAGIC`] as a sanity check.
    pub code: Vec<u8>,
    /// Number of bytes in `code` that are native machine instructions.
    /// Bytes in `code[native_code_len..]` are metadata tables.
    pub native_code_len: usize,
    /// Number of `i64` slots in the register file
    /// (`parameter_count + frame_size`).
    pub register_file_slots: usize,
    /// Safepoint table (code offset → bytecode instruction index + gc_map).
    pub safepoints: Vec<SafepointEntry>,
    /// Deoptimization table (code offset → bytecode byte offset + liveness).
    pub deopt_entries: Vec<DeoptEntry>,
}

/// Internal representation of the 12-byte metadata footer.
struct MetadataFooter {
    safepoint_table_start: u32,
    deopt_table_start: u32,
}

impl CompiledCode {
    /// Look up the safepoint entry whose `code_offset` equals `offset`.
    ///
    /// Returns `None` if no safepoint at that exact offset is recorded.
    pub fn find_safepoint(&self, offset: u32) -> Option<&SafepointEntry> {
        self.safepoints.iter().find(|e| e.code_offset == offset)
    }

    /// Look up the deopt entry whose `code_offset` equals `offset`.
    ///
    /// Returns `None` if no deopt entry at that exact offset is recorded.
    pub fn find_deopt(&self, offset: u32) -> Option<&DeoptEntry> {
        self.deopt_entries.iter().find(|e| e.code_offset == offset)
    }

    /// Parse the safepoint table from a slice of serialized code bytes.
    ///
    /// Returns `None` if the buffer does not contain a valid footer with
    /// [`METADATA_MAGIC`], or if the encoded table would overflow the buffer.
    pub fn parse_safepoints(code: &[u8]) -> Option<Vec<SafepointEntry>> {
        let footer = Self::read_footer(code)?;
        let start = footer.safepoint_table_start as usize;
        let data = code.get(start..)?;
        let count = u32::from_le_bytes(data.get(..4)?.try_into().ok()?) as usize;
        // Sanity-check: cap allocation to the number of entries that can
        // physically fit in the remaining buffer.
        let max_entries = data.len().saturating_sub(4) / SAFEPOINT_ENTRY_SIZE;
        if count > max_entries {
            return None;
        }
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let off = 4 + i * SAFEPOINT_ENTRY_SIZE;
            let e = data.get(off..off + SAFEPOINT_ENTRY_SIZE)?;
            out.push(SafepointEntry {
                code_offset: u32::from_le_bytes(e[0..4].try_into().ok()?),
                bytecode_index: u32::from_le_bytes(e[4..8].try_into().ok()?),
                gc_map: u64::from_le_bytes(e[8..16].try_into().ok()?),
            });
        }
        Some(out)
    }

    /// Parse the deopt table from a slice of serialized code bytes.
    ///
    /// Returns `None` if the buffer does not contain a valid footer with
    /// [`METADATA_MAGIC`], or if the encoded table would overflow the buffer.
    pub fn parse_deopt_entries(code: &[u8]) -> Option<Vec<DeoptEntry>> {
        let footer = Self::read_footer(code)?;
        let start = footer.deopt_table_start as usize;
        let data = code.get(start..)?;
        let count = u32::from_le_bytes(data.get(..4)?.try_into().ok()?) as usize;
        // Sanity-check: cap allocation to the number of entries that can
        // physically fit in the remaining buffer.
        let max_entries = data.len().saturating_sub(4) / DEOPT_ENTRY_SIZE;
        if count > max_entries {
            return None;
        }
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let off = 4 + i * DEOPT_ENTRY_SIZE;
            let e = data.get(off..off + DEOPT_ENTRY_SIZE)?;
            out.push(DeoptEntry {
                code_offset: u32::from_le_bytes(e[0..4].try_into().ok()?),
                bytecode_offset: u32::from_le_bytes(e[4..8].try_into().ok()?),
                liveness_map: u64::from_le_bytes(e[8..16].try_into().ok()?),
            });
        }
        Some(out)
    }

    /// Read and validate the 12-byte footer at the end of `code`.
    fn read_footer(code: &[u8]) -> Option<MetadataFooter> {
        if code.len() < FOOTER_SIZE {
            return None;
        }
        let f = &code[code.len() - FOOTER_SIZE..];
        let magic = u32::from_le_bytes(f[8..12].try_into().ok()?);
        if magic != METADATA_MAGIC {
            return None;
        }
        Some(MetadataFooter {
            safepoint_table_start: u32::from_le_bytes(f[0..4].try_into().ok()?),
            deopt_table_start: u32::from_le_bytes(f[4..8].try_into().ok()?),
        })
    }

    /// Execute the compiled code on x86-64 Linux/macOS by allocating a page of
    /// read-write-execute memory with `mmap`, copying the code bytes into it,
    /// and invoking the JIT function.
    ///
    /// `args` provides the initial values for the parameter slots.  Missing
    /// arguments are filled with `0` (`Smi(0)`); extra arguments are ignored.
    ///
    /// Returns the accumulator value on success.  Returns
    /// [`StatorError::Internal`] if `mmap` fails.
    ///
    /// If the JIT function returns [`JIT_DEOPT`], the result is
    /// `Err(StatorError::Internal("jit deopt"))`.
    ///
    /// # Safety
    ///
    /// The `code` bytes inside this [`CompiledCode`] must be valid x86-64
    /// machine code emitted by [`BaselineCompiler`].  Executing arbitrary or
    /// malformed bytes via `mmap` and a function pointer is undefined behaviour.
    #[cfg(all(target_arch = "x86_64", unix))]
    pub unsafe fn execute(&self, args: &[i64]) -> StatorResult<i64> {
        use std::ptr;

        let code_size = self.code.len();
        if code_size == 0 {
            return Err(StatorError::Internal("compiled code is empty".into()));
        }

        // Allocate a page of read/write/execute memory.
        //
        // SAFETY: arguments are valid; the return value is checked against
        // MAP_FAILED before use.
        let mem = unsafe {
            libc::mmap(
                ptr::null_mut(),
                code_size,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if mem == libc::MAP_FAILED {
            return Err(StatorError::Internal("mmap failed for JIT code".into()));
        }

        // SAFETY: `mem` is valid, page-aligned, and sized for `code_size` bytes.
        unsafe {
            ptr::copy_nonoverlapping(self.code.as_ptr(), mem.cast::<u8>(), code_size);
        }

        // Build the register file: fill with zeros, then overwrite with args.
        let mut regs = vec![0i64; self.register_file_slots];
        for (i, &v) in args.iter().enumerate().take(regs.len()) {
            regs[i] = v;
        }

        // Transmute and call the JIT function.
        //
        // SAFETY:
        // - `mem` contains correctly-encoded x86-64 machine code emitted by
        //   the baseline compiler.
        // - The function signature matches the JIT calling convention:
        //   `extern "C" fn(*mut i64) -> i64` (SysV AMD64).
        // - `regs.as_mut_ptr()` is valid for the lifetime of the call.
        let result = unsafe {
            let f: extern "C" fn(*mut i64) -> i64 = std::mem::transmute(mem);
            f(regs.as_mut_ptr())
        };

        // SAFETY: `mem` is a valid mapping of `code_size` bytes.
        unsafe {
            libc::munmap(mem, code_size);
        }

        if result == JIT_DEOPT {
            Err(StatorError::Internal("jit deopt".into()))
        } else {
            Ok(result)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BaselineCompiler
// ─────────────────────────────────────────────────────────────────────────────

/// Baseline (non-optimising) JIT compiler.
///
/// Compiles a [`BytecodeArray`] to x86-64 machine code.  Each bytecode
/// instruction is translated to a fixed native-code sequence; complex
/// operations (property access, function calls, generators) emit a jump to
/// the deopt epilogue that returns [`JIT_DEOPT`].
pub struct BaselineCompiler<'a> {
    bytecode: &'a BytecodeArray,
    masm: MacroAssembler,
    param_count: usize,
    safepoints: Vec<SafepointEntry>,
    deopt_entries: Vec<DeoptEntry>,
    /// Per-instruction label (bound at the start of each instruction's code).
    labels: Vec<Label>,
    /// Label for the shared deopt epilogue.
    deopt_label: Label,
}

impl<'a> BaselineCompiler<'a> {
    /// Compile `bytecode` to native x86-64 machine code.
    ///
    /// Returns a [`CompiledCode`] containing the emitted bytes and all
    /// associated metadata.  The safepoint and deopt tables are serialized and
    /// appended to the code buffer after the native instructions; a 12-byte
    /// footer records the table start offsets and [`METADATA_MAGIC`].
    pub fn compile(bytecode: &'a BytecodeArray) -> StatorResult<CompiledCode> {
        let mut c = Self {
            bytecode,
            masm: MacroAssembler::new(),
            param_count: bytecode.parameter_count() as usize,
            safepoints: Vec::new(),
            deopt_entries: Vec::new(),
            labels: Vec::new(),
            deopt_label: Label::new(),
        };
        c.compile_function()?;
        let register_file_slots =
            bytecode.parameter_count() as usize + bytecode.frame_size() as usize;
        let mut code = c.masm.into_code();
        let native_code_len = code.len();

        // ── Serialize safepoint table ────────────────────────────────────────
        let safepoint_table_start = u32::try_from(code.len())
            .map_err(|_| StatorError::Internal("compiled code exceeds 4 GiB limit".into()))?;
        let sp_count = c.safepoints.len() as u32;
        code.extend_from_slice(&sp_count.to_le_bytes());
        for e in &c.safepoints {
            code.extend_from_slice(&e.code_offset.to_le_bytes());
            code.extend_from_slice(&e.bytecode_index.to_le_bytes());
            code.extend_from_slice(&e.gc_map.to_le_bytes());
        }

        // ── Serialize deopt table ────────────────────────────────────────────
        let deopt_table_start = u32::try_from(code.len())
            .map_err(|_| StatorError::Internal("compiled code exceeds 4 GiB limit".into()))?;
        let de_count = c.deopt_entries.len() as u32;
        code.extend_from_slice(&de_count.to_le_bytes());
        for e in &c.deopt_entries {
            code.extend_from_slice(&e.code_offset.to_le_bytes());
            code.extend_from_slice(&e.bytecode_offset.to_le_bytes());
            code.extend_from_slice(&e.liveness_map.to_le_bytes());
        }

        // ── Serialize footer (last FOOTER_SIZE bytes) ────────────────────────
        code.extend_from_slice(&safepoint_table_start.to_le_bytes());
        code.extend_from_slice(&deopt_table_start.to_le_bytes());
        code.extend_from_slice(&METADATA_MAGIC.to_le_bytes());

        Ok(CompiledCode {
            code,
            native_code_len,
            register_file_slots,
            safepoints: c.safepoints,
            deopt_entries: c.deopt_entries,
        })
    }

    // ── Prologue / epilogue ──────────────────────────────────────────────────

    /// Emit the standard function prologue.
    ///
    /// Sets up the call frame and loads the register-file pointer into R14.
    ///
    /// ```text
    /// push rbp
    /// mov  rbp, rsp
    /// push r12        ; callee-saved accumulator
    /// push r14        ; callee-saved register-file pointer
    /// mov  r14, rdi   ; r14 = regs argument
    /// xor  r12, r12   ; accumulator = 0
    /// ```
    fn emit_prologue(&mut self) {
        self.masm.push(Reg64::Rbp);
        self.masm.mov_rr(Reg64::Rbp, Reg64::Rsp);
        self.masm.push(Reg64::R12);
        self.masm.push(Reg64::R14);
        self.masm.mov_rr(Reg64::R14, Reg64::Rdi);
        self.masm.xor_rr(Reg64::R12, Reg64::R12);
    }

    /// Emit the normal function epilogue.
    ///
    /// ```text
    /// mov rax, r12    ; return accumulator
    /// pop r14
    /// pop r12
    /// pop rbp
    /// ret
    /// ```
    fn emit_normal_epilogue(&mut self) {
        self.masm.mov_rr(Reg64::Rax, Reg64::R12);
        self.masm.pop(Reg64::R14);
        self.masm.pop(Reg64::R12);
        self.masm.pop(Reg64::Rbp);
        self.masm.ret();
    }

    /// Emit the deopt epilogue.
    ///
    /// Loads [`JIT_DEOPT`] into the return-value register and then falls
    /// through to the standard register-restore / `ret` sequence.
    ///
    /// ```text
    /// deopt_label:
    ///   mov r12, JIT_DEOPT
    ///   mov rax, r12
    ///   pop r14
    ///   pop r12
    ///   pop rbp
    ///   ret
    /// ```
    fn emit_deopt_epilogue(&mut self) {
        self.masm.bind_label(&mut self.deopt_label);
        self.masm.mov_ri(Reg64::R12, JIT_DEOPT);
        self.emit_normal_epilogue();
    }

    // ── Register-file helpers ────────────────────────────────────────────────

    /// Compute the byte offset from R14 (register-file base) for a bytecode
    /// register operand value `v`.
    fn reg_offset(&self, v: u32) -> i32 {
        let signed = v as i32;
        let flat_index = if signed >= 0 {
            self.param_count + signed as usize
        } else {
            (-(signed + 1)) as usize
        };
        (flat_index * 8) as i32
    }

    /// Emit code to load register `v` into `dst`.
    fn emit_load_reg(&mut self, dst: Reg64, v: u32) {
        let off = self.reg_offset(v);
        self.masm.mov_load_base_disp32(dst, Reg64::R14, off);
    }

    /// Emit code to store `src` into register `v`.
    fn emit_store_reg(&mut self, v: u32, src: Reg64) {
        let off = self.reg_offset(v);
        self.masm.mov_store_base_disp32(Reg64::R14, off, src);
    }

    // ── Comparison helper ────────────────────────────────────────────────────

    /// Emit code that:
    /// 1. Loads the RHS register `v` into R11.
    /// 2. Compares R12 (accumulator) against R11.
    /// 3. Sets AL via `SETCC cc`.
    /// 4. Zero-extends AL into R12.
    /// 5. Converts the 0/1 result in R12 to [`JIT_FALSE`] / [`JIT_TRUE`].
    fn emit_compare_and_set(&mut self, v: u32, cc: CondCode) {
        self.emit_load_reg(Reg64::R11, v);
        self.masm.cmp_rr(Reg64::R12, Reg64::R11);
        self.masm.setcc_al(cc);
        self.masm.movzx_r64_al(Reg64::R12);
        // Convert raw 0/1 → JIT_FALSE/JIT_TRUE by adding JIT_FALSE.
        self.masm.mov_ri(Reg64::R11, JIT_FALSE);
        self.masm.add_rr(Reg64::R12, Reg64::R11);
    }

    /// Emit code that compares R12 against `sentinel` and converts the result
    /// to JIT_FALSE / JIT_TRUE (1 if equal, 0 if not).
    fn emit_test_sentinel(&mut self, sentinel: i64) {
        self.masm.mov_ri(Reg64::R11, sentinel);
        self.masm.cmp_rr(Reg64::R12, Reg64::R11);
        self.masm.setcc_al(CondCode::Equal);
        self.masm.movzx_r64_al(Reg64::R12);
        self.masm.mov_ri(Reg64::R11, JIT_FALSE);
        self.masm.add_rr(Reg64::R12, Reg64::R11);
    }

    // ── Jump helpers ─────────────────────────────────────────────────────────

    /// Find the instruction index whose bytecode byte offset equals
    /// `target_byte`, given the `byte_offsets` table.
    fn resolve_target(
        target_byte: usize,
        byte_offsets: &[usize],
        instr_count: usize,
    ) -> StatorResult<usize> {
        byte_offsets[..instr_count]
            .binary_search(&target_byte)
            .map_err(|_| {
                StatorError::Internal(format!(
                    "jump target {target_byte} is not at an instruction boundary"
                ))
            })
    }

    /// Emit an unconditional JMP to the instruction at `target_idx`.
    fn emit_jump(&mut self, target_idx: usize) {
        self.masm.jmp(&mut self.labels[target_idx]);
    }

    /// Emit a conditional JMP (`cc`) to `target_idx`.
    fn emit_cond_jump(&mut self, cc: CondCode, target_idx: usize) {
        self.masm.jcc(cc, &mut self.labels[target_idx]);
    }

    // ── Deopt helper ─────────────────────────────────────────────────────────

    /// Compute a conservative liveness bitmask covering all register-file
    /// slots for use in deopt entries.
    ///
    /// Sets bit `i` for every slot index `i` in `0..register_file_slots`.
    /// This is a conservative over-approximation: the interpreter will
    /// preserve all slots when reconstructing the frame.
    fn all_slots_live(&self) -> u64 {
        let slots = self.param_count + self.bytecode.frame_size() as usize;
        if slots >= 64 {
            u64::MAX
        } else {
            (1u64 << slots).wrapping_sub(1)
        }
    }

    /// Record a deopt point at the current code position and emit a JMP to
    /// the deopt epilogue.
    fn emit_deopt(&mut self, bytecode_offset: u32) {
        let code_off = self.masm.position() as u32;
        let liveness_map = self.all_slots_live();
        self.deopt_entries.push(DeoptEntry {
            code_offset: code_off,
            bytecode_offset,
            liveness_map,
        });
        self.masm.jmp(&mut self.deopt_label);
    }

    // ── Main compilation pass ────────────────────────────────────────────────

    fn compile_function(&mut self) -> StatorResult<()> {
        let (instructions, byte_offsets) = decode_with_byte_offsets(self.bytecode.bytecodes())?;
        let n = instructions.len();

        // Pre-create one label per instruction.
        self.labels = (0..n).map(|_| Label::new()).collect();

        self.emit_prologue();

        for (idx, instr) in instructions.iter().enumerate() {
            // Bind the label for this instruction to the current code position.
            self.masm.bind_label(&mut self.labels[idx]);

            // Safepoint at every instruction.
            self.safepoints.push(SafepointEntry {
                code_offset: self.masm.position() as u32,
                bytecode_index: idx as u32,
                gc_map: 0,
            });

            self.compile_instruction(
                idx,
                &instructions,
                &byte_offsets,
                instr,
                byte_offsets[idx] as u32,
            )?;
        }

        // Emit the deopt epilogue after the normal instruction stream.
        self.emit_deopt_epilogue();

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_instruction(
        &mut self,
        idx: usize,
        instructions: &[Instruction],
        byte_offsets: &[usize],
        instr: &Instruction,
        bytecode_offset: u32,
    ) -> StatorResult<()> {
        let n = instructions.len();

        match instr.opcode {
            // ── Load immediates ──────────────────────────────────────────────
            Opcode::LdaZero => {
                self.masm.xor_rr(Reg64::R12, Reg64::R12);
            }
            Opcode::LdaSmi => {
                let Operand::Immediate(v) = instr.operands[0] else {
                    return Err(bad_operand("LdaSmi", 0));
                };
                self.masm.mov_ri(Reg64::R12, v as i64);
            }
            Opcode::LdaUndefined => {
                self.masm.mov_ri(Reg64::R12, JIT_UNDEFINED);
            }
            Opcode::LdaNull => {
                self.masm.mov_ri(Reg64::R12, JIT_NULL);
            }
            Opcode::LdaTrue => {
                self.masm.mov_ri(Reg64::R12, JIT_TRUE);
            }
            Opcode::LdaFalse => {
                self.masm.mov_ri(Reg64::R12, JIT_FALSE);
            }
            Opcode::LdaConstant => {
                let Operand::ConstantPoolIdx(idx_cp) = instr.operands[0] else {
                    return Err(bad_operand("LdaConstant", 0));
                };
                match self.bytecode.get_constant(idx_cp) {
                    Some(ConstantPoolEntry::Number(f)) => {
                        let f = *f;
                        if f.fract() == 0.0
                            && (i32::MIN as f64..=i32::MAX as f64).contains(&f)
                            && f.is_finite()
                        {
                            self.masm.mov_ri(Reg64::R12, f as i64);
                        } else {
                            // Non-integer numbers require a HeapNumber; deopt.
                            self.emit_deopt(bytecode_offset);
                        }
                    }
                    Some(ConstantPoolEntry::Boolean(true)) => {
                        self.masm.mov_ri(Reg64::R12, JIT_TRUE);
                    }
                    Some(ConstantPoolEntry::Boolean(false)) => {
                        self.masm.mov_ri(Reg64::R12, JIT_FALSE);
                    }
                    Some(ConstantPoolEntry::Null) => {
                        self.masm.mov_ri(Reg64::R12, JIT_NULL);
                    }
                    Some(ConstantPoolEntry::Undefined) => {
                        self.masm.mov_ri(Reg64::R12, JIT_UNDEFINED);
                    }
                    _ => {
                        // Strings and Function entries require heap allocation; deopt.
                        self.emit_deopt(bytecode_offset);
                    }
                }
            }

            // ── Register moves ───────────────────────────────────────────────
            Opcode::Ldar => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("Ldar", 0));
                };
                self.emit_load_reg(Reg64::R12, v);
            }
            Opcode::Star => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("Star", 0));
                };
                self.emit_store_reg(v, Reg64::R12);
            }
            Opcode::Mov => {
                let Operand::Register(src) = instr.operands[0] else {
                    return Err(bad_operand("Mov", 0));
                };
                let Operand::Register(dst) = instr.operands[1] else {
                    return Err(bad_operand("Mov", 1));
                };
                self.emit_load_reg(Reg64::R11, src);
                self.emit_store_reg(dst, Reg64::R11);
            }

            // ── Arithmetic ───────────────────────────────────────────────────
            Opcode::Add => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("Add", 0));
                };
                self.emit_load_reg(Reg64::R11, v);
                self.masm.add_rr(Reg64::R12, Reg64::R11);
            }
            Opcode::Sub => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("Sub", 0));
                };
                self.emit_load_reg(Reg64::R11, v);
                self.masm.sub_rr(Reg64::R12, Reg64::R11);
            }
            Opcode::Mul => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("Mul", 0));
                };
                self.emit_load_reg(Reg64::R11, v);
                self.masm.imul_rr(Reg64::R12, Reg64::R11);
            }
            Opcode::Inc => {
                self.masm.add_ri(Reg64::R12, 1);
            }
            Opcode::Dec => {
                self.masm.sub_ri(Reg64::R12, 1);
            }
            Opcode::AddSmi => {
                let Operand::Immediate(imm) = instr.operands[0] else {
                    return Err(bad_operand("AddSmi", 0));
                };
                self.masm.add_ri(Reg64::R12, imm);
            }
            Opcode::SubSmi => {
                let Operand::Immediate(imm) = instr.operands[0] else {
                    return Err(bad_operand("SubSmi", 0));
                };
                self.masm.sub_ri(Reg64::R12, imm);
            }
            Opcode::MulSmi => {
                let Operand::Immediate(imm) = instr.operands[0] else {
                    return Err(bad_operand("MulSmi", 0));
                };
                self.masm.mov_ri(Reg64::R11, imm as i64);
                self.masm.imul_rr(Reg64::R12, Reg64::R11);
            }
            Opcode::Negate => {
                self.masm.neg_r(Reg64::R12);
            }

            // ── Comparisons ──────────────────────────────────────────────────
            Opcode::TestLessThan => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("TestLessThan", 0));
                };
                self.emit_compare_and_set(v, CondCode::Less);
            }
            Opcode::TestGreaterThan => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("TestGreaterThan", 0));
                };
                self.emit_compare_and_set(v, CondCode::Greater);
            }
            Opcode::TestLessThanOrEqual => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("TestLessThanOrEqual", 0));
                };
                self.emit_compare_and_set(v, CondCode::LessEq);
            }
            Opcode::TestGreaterThanOrEqual => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("TestGreaterThanOrEqual", 0));
                };
                self.emit_compare_and_set(v, CondCode::GreaterEq);
            }
            Opcode::TestEqual | Opcode::TestEqualStrict => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("TestEqual", 0));
                };
                self.emit_compare_and_set(v, CondCode::Equal);
            }
            Opcode::TestNotEqual => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("TestNotEqual", 0));
                };
                self.emit_compare_and_set(v, CondCode::NotEqual);
            }
            Opcode::TestNull => {
                self.emit_test_sentinel(JIT_NULL);
            }
            Opcode::TestUndefined => {
                self.emit_test_sentinel(JIT_UNDEFINED);
            }

            // ── Logical ──────────────────────────────────────────────────────
            Opcode::LogicalNot | Opcode::ToBooleanLogicalNot => {
                // Flip JIT_FALSE ↔ JIT_TRUE by toggling bit 0.
                // JIT_FALSE = 0x1_0000_0000 (bit 0 = 0) → XOR 1 → JIT_TRUE
                // JIT_TRUE  = 0x1_0000_0001 (bit 0 = 1) → XOR 1 → JIT_FALSE
                self.masm.xor_ri(Reg64::R12, 1);
            }

            // ── Control flow ─────────────────────────────────────────────────
            Opcode::Jump => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("Jump", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                self.emit_jump(target);
            }
            Opcode::JumpLoop => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("JumpLoop", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                self.emit_jump(target);
            }
            Opcode::JumpIfTrue => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("JumpIfTrue", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                // Jump if acc == JIT_TRUE
                self.masm.mov_ri(Reg64::R11, JIT_TRUE);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(CondCode::Equal, target);
            }
            Opcode::JumpIfFalse => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("JumpIfFalse", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                self.masm.mov_ri(Reg64::R11, JIT_FALSE);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(CondCode::Equal, target);
            }
            Opcode::JumpIfToBooleanTrue => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("JumpIfToBooleanTrue", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                // Jump if acc != JIT_FALSE (i.e. truthy).
                self.masm.mov_ri(Reg64::R11, JIT_FALSE);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(CondCode::NotEqual, target);
            }
            Opcode::JumpIfToBooleanFalse => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("JumpIfToBooleanFalse", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                // Jump if acc == JIT_FALSE (i.e. falsy).
                self.masm.mov_ri(Reg64::R11, JIT_FALSE);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(CondCode::Equal, target);
            }
            Opcode::JumpIfNull => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("JumpIfNull", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                self.masm.mov_ri(Reg64::R11, JIT_NULL);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(CondCode::Equal, target);
            }
            Opcode::JumpIfNotNull => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("JumpIfNotNull", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                self.masm.mov_ri(Reg64::R11, JIT_NULL);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(CondCode::NotEqual, target);
            }
            Opcode::JumpIfUndefined => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("JumpIfUndefined", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                self.masm.mov_ri(Reg64::R11, JIT_UNDEFINED);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(CondCode::Equal, target);
            }
            Opcode::JumpIfNotUndefined => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("JumpIfNotUndefined", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                self.masm.mov_ri(Reg64::R11, JIT_UNDEFINED);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(CondCode::NotEqual, target);
            }
            Opcode::JumpIfUndefinedOrNull => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("JumpIfUndefinedOrNull", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                // Jump if acc == JIT_UNDEFINED.
                self.masm.mov_ri(Reg64::R11, JIT_UNDEFINED);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(CondCode::Equal, target);
                // Also jump if acc == JIT_NULL.
                self.masm.mov_ri(Reg64::R11, JIT_NULL);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(CondCode::Equal, target);
            }

            // ── Return ───────────────────────────────────────────────────────
            Opcode::Return => {
                self.emit_normal_epilogue();
            }

            // ── No-ops / metadata ─────────────────────────────────────────────
            Opcode::StackCheck
            | Opcode::SetExpressionPosition
            | Opcode::SetExpressionPositionFromEnd
            | Opcode::CollectTypeProfile => {
                // These carry no runtime semantics; emit nothing.
            }

            // ── IC stubs / property access / calls → deopt ────────────────────
            Opcode::LdaNamedProperty
            | Opcode::LdaNamedPropertyFromSuper
            | Opcode::LdaKeyedProperty
            | Opcode::LdaEnumeratedKeyedProperty
            | Opcode::StaNamedProperty
            | Opcode::StaNamedOwnProperty
            | Opcode::StaKeyedProperty
            | Opcode::DefineNamedOwnProperty
            | Opcode::DefineKeyedOwnProperty
            | Opcode::StaInArrayLiteral
            | Opcode::DefineKeyedOwnPropertyInLiteral
            | Opcode::LdaGlobal
            | Opcode::LdaGlobalInsideTypeof
            | Opcode::StaGlobal
            | Opcode::LdaContextSlot
            | Opcode::LdaImmutableContextSlot
            | Opcode::LdaCurrentContextSlot
            | Opcode::LdaImmutableCurrentContextSlot
            | Opcode::StaContextSlot
            | Opcode::StaCurrentContextSlot
            | Opcode::LdaLookupSlot
            | Opcode::LdaLookupContextSlot
            | Opcode::LdaLookupGlobalSlot
            | Opcode::LdaLookupSlotInsideTypeof
            | Opcode::LdaLookupContextSlotInsideTypeof
            | Opcode::LdaLookupGlobalSlotInsideTypeof
            | Opcode::StaLookupSlot
            | Opcode::CallAnyReceiver
            | Opcode::CallProperty
            | Opcode::CallProperty0
            | Opcode::CallProperty1
            | Opcode::CallProperty2
            | Opcode::CallUndefinedReceiver0
            | Opcode::CallUndefinedReceiver1
            | Opcode::CallUndefinedReceiver2
            | Opcode::CallWithSpread
            | Opcode::CallRuntime
            | Opcode::CallRuntimeForPair
            | Opcode::CallJSRuntime
            | Opcode::InvokeIntrinsic
            | Opcode::Construct
            | Opcode::ConstructWithSpread
            | Opcode::ConstructForwardAllArgs
            | Opcode::CreateClosure
            | Opcode::CreateBlockContext
            | Opcode::CreateCatchContext
            | Opcode::CreateFunctionContext
            | Opcode::CreateEvalContext
            | Opcode::CreateWithContext
            | Opcode::CreateMappedArguments
            | Opcode::CreateUnmappedArguments
            | Opcode::CreateRestParameter
            | Opcode::CreateRegExpLiteral
            | Opcode::CreateArrayLiteral
            | Opcode::CreateArrayFromIterable
            | Opcode::CreateEmptyArrayLiteral
            | Opcode::CreateObjectLiteral
            | Opcode::CreateEmptyObjectLiteral
            | Opcode::CreateObjectFromIterable
            | Opcode::GetIterator
            | Opcode::GetAsyncIterator
            | Opcode::IteratorNext
            | Opcode::PushContext
            | Opcode::PopContext
            | Opcode::ForInEnumerate
            | Opcode::ForInPrepare
            | Opcode::ForInNext
            | Opcode::ForInStep
            | Opcode::JumpIfForInDone
            | Opcode::GetTemplateObject
            | Opcode::SwitchOnGeneratorState
            | Opcode::SuspendGenerator
            | Opcode::ResumeGenerator
            | Opcode::GetGeneratorState
            | Opcode::SetGeneratorState
            | Opcode::ToName
            | Opcode::ToNumber
            | Opcode::ToNumeric
            | Opcode::ToObject
            | Opcode::ToString
            | Opcode::ToBoolean
            | Opcode::TypeOf
            | Opcode::DeletePropertyStrict
            | Opcode::DeletePropertySloppy
            | Opcode::TestReferenceEqual
            | Opcode::TestInstanceOf
            | Opcode::TestIn
            | Opcode::TestUndetectable
            | Opcode::TestTypeOf
            | Opcode::Throw
            | Opcode::ReThrow
            | Opcode::SetPendingMessage
            | Opcode::ThrowReferenceErrorIfHole
            | Opcode::ThrowSuperNotCalledIfHole
            | Opcode::ThrowSuperAlreadyCalledIfNotHole
            | Opcode::Debugger
            | Opcode::JumpConstant
            | Opcode::JumpIfTrueConstant
            | Opcode::JumpIfFalseConstant
            | Opcode::JumpIfToBooleanTrueConstant
            | Opcode::JumpIfToBooleanFalseConstant
            | Opcode::JumpIfNullConstant
            | Opcode::JumpIfNotNullConstant
            | Opcode::JumpIfUndefinedConstant
            | Opcode::JumpIfNotUndefinedConstant
            | Opcode::JumpIfUndefinedOrNullConstant
            | Opcode::JumpIfJSReceiver
            | Opcode::JumpIfJSReceiverConstant
            | Opcode::Div
            | Opcode::Mod
            | Opcode::Exp
            | Opcode::BitwiseOr
            | Opcode::BitwiseXor
            | Opcode::BitwiseAnd
            | Opcode::ShiftLeft
            | Opcode::ShiftRight
            | Opcode::ShiftRightLogical
            | Opcode::DivSmi
            | Opcode::ModSmi
            | Opcode::ExpSmi
            | Opcode::BitwiseOrSmi
            | Opcode::BitwiseXorSmi
            | Opcode::BitwiseAndSmi
            | Opcode::ShiftLeftSmi
            | Opcode::ShiftRightSmi
            | Opcode::ShiftRightLogicalSmi
            | Opcode::BitwiseNot => {
                self.emit_deopt(bytecode_offset);
            }

            // These prefix/trap opcodes should never appear here.
            Opcode::Wide | Opcode::ExtraWide | Opcode::Illegal => {
                return Err(StatorError::Internal(format!(
                    "unexpected opcode in compilation: {:?}",
                    instr.opcode
                )));
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the target byte offset for a jump with the given `delta`.
///
/// `idx` is the instruction index of the jump; `byte_offsets` is the table
/// produced by [`decode_with_byte_offsets`].  The target byte is `delta`
/// bytes past the end of the jump instruction.
fn jump_target_byte(idx: usize, delta: i32, byte_offsets: &[usize]) -> usize {
    // byte_offsets[idx + 1] = byte offset one past the jump instruction.
    let after = byte_offsets[idx + 1];
    (after as i64 + delta as i64) as usize
}

/// Construct a [`StatorError::Internal`] for an unexpected operand kind.
#[cold]
fn bad_operand(opcode: &'static str, i: usize) -> StatorError {
    StatorError::Internal(format!("{opcode}: unexpected operand at index {i}"))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::bytecode_array::BytecodeArray;
    use crate::bytecode::bytecodes::{Instruction, Opcode, Operand, encode};
    use crate::bytecode::feedback::FeedbackMetadata;
    use crate::interpreter::{Interpreter, InterpreterFrame};
    use crate::objects::value::JsValue;

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Build a [`BytecodeArray`] from a list of instructions with no constant
    /// pool, one register and zero parameters.
    fn bytecode(instrs: Vec<Instruction>) -> BytecodeArray {
        let bytes = encode(&instrs);
        BytecodeArray::new(
            bytes,
            vec![],
            4,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        )
    }

    /// Run `ba` through the Stator interpreter and return the accumulator.
    fn interp_run(ba: BytecodeArray) -> JsValue {
        let mut frame = InterpreterFrame::new(ba, vec![]);
        Interpreter::run(&mut frame).expect("interpreter error")
    }

    // ── compile-only tests (no execution, all platforms) ─────────────────────

    #[test]
    fn test_compile_produces_non_empty_code() {
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let cc = BaselineCompiler::compile(&ba).expect("compile failed");
        assert!(!cc.code.is_empty(), "compiled code must be non-empty");
    }

    #[test]
    fn test_compile_builds_safepoint_table() {
        // LdaSmi(7), Return  →  2 safepoint entries
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let cc = BaselineCompiler::compile(&ba).expect("compile failed");
        assert_eq!(cc.safepoints.len(), 2);
        assert_eq!(cc.safepoints[0].bytecode_index, 0);
        assert_eq!(cc.safepoints[1].bytecode_index, 1);
    }

    #[test]
    fn test_compile_deopt_for_unsupported_opcode() {
        // LdaGlobal emits a deopt entry.
        let ba = bytecode(vec![
            Instruction::new_unchecked(
                Opcode::LdaGlobal,
                vec![Operand::ConstantPoolIdx(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let cc = BaselineCompiler::compile(&ba).expect("compile failed");
        assert_eq!(cc.deopt_entries.len(), 1);
    }

    #[test]
    fn test_compile_register_file_slots() {
        let ba = BytecodeArray::new(
            encode(&[Instruction::new_unchecked(Opcode::Return, vec![])]),
            vec![],
            5, // frame_size
            2, // parameter_count
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );
        let cc = BaselineCompiler::compile(&ba).expect("compile failed");
        assert_eq!(cc.register_file_slots, 7); // 2 params + 5 locals
    }

    #[test]
    fn test_compile_empty_bytecode_errors() {
        let ba = BytecodeArray::new(
            vec![],
            vec![],
            0,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );
        // An empty bytecode stream has no Return; compile succeeds but the
        // resulting code (prologue + deopt epilogue only) is still valid bytes.
        let cc = BaselineCompiler::compile(&ba).expect("compile should not error on empty stream");
        assert!(!cc.code.is_empty());
    }

    // ── execution tests (x86-64 + Unix only) ─────────────────────────────────

    /// Compile `ba`, execute the JIT code with `args`, and return the raw i64
    /// accumulator value.
    #[cfg(all(target_arch = "x86_64", unix))]
    fn jit_run(ba: &BytecodeArray, args: &[i64]) -> i64 {
        let cc = BaselineCompiler::compile(ba).expect("compile failed");
        unsafe { cc.execute(args).expect("jit execute failed") }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_lda_smi_return() {
        // LdaSmi(42), Return  →  42
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let result = jit_run(&ba, &[]);
        assert_eq!(result, 42);
        assert_eq!(interp_run(ba), JsValue::Smi(42));
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_lda_zero_return() {
        // LdaZero, Return  →  0
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaZero, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let result = jit_run(&ba, &[]);
        assert_eq!(result, 0);
        assert_eq!(interp_run(ba), JsValue::Smi(0));
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_addition() {
        // LdaSmi(3), Star(r0), LdaSmi(2), Add(r0, _), Return  →  5
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
            Instruction::new_unchecked(
                Opcode::Add,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let jit_result = jit_run(&ba, &[]);
        let interp_result = interp_run(ba);
        assert_eq!(jit_result, 5);
        assert_eq!(interp_result, JsValue::Smi(5));
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_subtraction() {
        // LdaSmi(10) → r0; LdaSmi(4) → acc; Sub(r0): acc = acc − r0 = 4 − 10 = −6
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(4)]),
            Instruction::new_unchecked(
                Opcode::Sub,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let jit_result = jit_run(&ba, &[]);
        let interp_result = interp_run(ba);
        assert_eq!(jit_result, -6);
        assert_eq!(interp_result, JsValue::Smi(-6));
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_multiplication() {
        // LdaSmi(6), Star(r0), LdaSmi(7), Mul(r0, _), Return  →  42
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(6)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
            Instruction::new_unchecked(
                Opcode::Mul,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let jit_result = jit_run(&ba, &[]);
        let interp_result = interp_run(ba);
        assert_eq!(jit_result, 42);
        assert_eq!(interp_result, JsValue::Smi(42));
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_increment() {
        // LdaSmi(99), Inc(_), Return  →  100
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Inc, vec![Operand::FeedbackSlot(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let jit_result = jit_run(&ba, &[]);
        let interp_result = interp_run(ba);
        assert_eq!(jit_result, 100);
        assert_eq!(interp_result, JsValue::Smi(100));
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_negate() {
        // LdaSmi(7), Negate(_), Return  →  -7
        // Note: Negate is not yet implemented in the tree-walking interpreter,
        // so we only compare the JIT result to the expected value.
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
            Instruction::new_unchecked(Opcode::Negate, vec![Operand::FeedbackSlot(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let jit_result = jit_run(&ba, &[]);
        assert_eq!(jit_result, -7);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_comparison_less_than_true() {
        // Store 5 in r0; load 3 into acc; TestLessThan(r0): acc = (3 < 5) = true → JIT_TRUE
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
            Instruction::new_unchecked(
                Opcode::TestLessThan,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let jit_result = jit_run(&ba, &[]);
        let interp_result = interp_run(ba);
        assert_eq!(jit_result, JIT_TRUE);
        assert_eq!(interp_result, JsValue::Boolean(true));
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_comparison_less_than_false() {
        // Store 3 in r0; load 5 into acc; TestLessThan(r0): acc = (5 < 3) = false → JIT_FALSE
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
            Instruction::new_unchecked(
                Opcode::TestLessThan,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let jit_result = jit_run(&ba, &[]);
        let interp_result = interp_run(ba);
        assert_eq!(jit_result, JIT_FALSE);
        assert_eq!(interp_result, JsValue::Boolean(false));
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_unconditional_jump() {
        // LdaSmi(1), Jump(+2 bytes → Return), LdaSmi(99), Return
        // (Jump skips the LdaSmi(99); should return 1)
        //
        // Byte layout (all narrow, each opcode + 1-byte operand = 2 bytes):
        //   0: LdaSmi(1)   2 bytes
        //   2: Jump(+2)    2 bytes  → target = end_of_jump(4) + 2 = 6 = Return
        //   4: LdaSmi(99)  2 bytes  (skipped)
        //   6: Return      1 byte
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::Jump, vec![Operand::JumpOffset(2)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = bytecode(instrs);
        let jit_result = jit_run(&ba, &[]);
        let interp_result = interp_run(ba);
        assert_eq!(jit_result, 1);
        assert_eq!(interp_result, JsValue::Smi(1));
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_sum_loop() {
        // Compute sum = 0; for (let i = 1; i <= 5; i++) sum += i;
        // Equivalent bytecode:
        //   LdaZero                ; acc = 0
        //   Star r0                ; sum = 0
        //   LdaSmi 1               ; acc = 1
        //   Star r1                ; i = 1
        // loop:
        //   LdaSmi 5
        //   Star r2                ; r2 = 5
        //   Ldar r1                ; acc = i
        //   TestLessThanOrEqual r2 ; acc = (i <= 5)
        //   JumpIfToBooleanFalse exit
        //   Ldar r0                ; acc = sum
        //   Add r1                 ; acc = sum + i
        //   Star r0                ; sum = acc
        //   Ldar r1                ; acc = i
        //   Inc                    ; acc = i + 1
        //   Star r1                ; i = acc
        //   Jump loop
        // exit:
        //   Ldar r0                ; acc = sum
        //   Return
        //
        // We need to compute the jump offsets manually based on the encoded size.
        // Let's build and encode step by step.

        // Opcode sizes (narrow encoding): each opcode is 1 byte + operands.
        // LdaZero: 1+0=1
        // Star r0: 1+1=2
        // LdaSmi 1: 1+1=2
        // Star r1: 1+1=2
        // --- loop starts here (offset = 7) ---
        // LdaSmi 5: 2
        // Star r2: 2
        // Ldar r1: 2
        // TestLessThanOrEqual r2: 1+1+1=3
        // JumpIfToBooleanFalse +N: 1+1=2  (need to compute N)
        // Ldar r0: 2
        // Add r1: 1+1+1=3
        // Star r0: 2
        // Ldar r1: 2
        // Inc: 1+1=2
        // Star r1: 2
        // Jump -M: 1+1=2  (need to compute M)
        // --- exit starts here ---
        // Ldar r0: 2
        // Return: 1

        // Let's compute offsets:
        // 0:  LdaZero (1)
        // 1:  Star r0 (2)
        // 3:  LdaSmi 1 (2)
        // 5:  Star r1 (2)
        // 7:  LdaSmi 5 (2)          ← loop top
        // 9:  Star r2 (2)
        // 11: Ldar r1 (2)
        // 13: TestLessThanOrEqual r2,slot (3)
        // 16: JumpIfToBooleanFalse δ (2) ← δ from end of this (byte 18) to exit
        // 18: Ldar r0 (2)
        // 20: Add r1,slot (3)
        // 23: Star r0 (2)
        // 25: Ldar r1 (2)
        // 27: Inc slot (2)
        // 29: Star r1 (2)
        // 31: Jump δ2 (2)            ← δ2 from byte 33 back to 7 = 7-33 = -26
        // 33: Ldar r0 (2)            ← exit
        // 35: Return (1)
        // JumpIfToBooleanFalse: from byte 18 to byte 33 → δ = 33 - 18 = 15
        // Jump: from byte 33 to byte 7 → δ = 7 - 33 = -26

        let instrs = vec![
            // 0: LdaZero
            Instruction::new_unchecked(Opcode::LdaZero, vec![]),
            // 1: Star r0
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            // 3: LdaSmi 1
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            // 5: Star r1
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            // 7: LdaSmi 5   ← loop
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
            // 9: Star r2
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            // 11: Ldar r1
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(1)]),
            // 13: TestLessThanOrEqual r2
            Instruction::new_unchecked(
                Opcode::TestLessThanOrEqual,
                vec![Operand::Register(2), Operand::FeedbackSlot(0)],
            ),
            // 16: JumpIfToBooleanFalse +15
            Instruction::new_unchecked(Opcode::JumpIfToBooleanFalse, vec![Operand::JumpOffset(15)]),
            // 18: Ldar r0
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            // 20: Add r1
            Instruction::new_unchecked(
                Opcode::Add,
                vec![Operand::Register(1), Operand::FeedbackSlot(0)],
            ),
            // 23: Star r0
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            // 25: Ldar r1
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(1)]),
            // 27: Inc
            Instruction::new_unchecked(Opcode::Inc, vec![Operand::FeedbackSlot(0)]),
            // 29: Star r1
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            // 31: Jump -26
            Instruction::new_unchecked(Opcode::Jump, vec![Operand::JumpOffset(-26)]),
            // 33: Ldar r0   ← exit
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            // 35: Return
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let bytes = encode(&instrs);
        let ba = BytecodeArray::new(
            bytes,
            vec![],
            4,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );

        let jit_result = jit_run(&ba, &[]);
        let interp_result = interp_run(ba);
        // sum = 1+2+3+4+5 = 15
        assert_eq!(jit_result, 15);
        assert_eq!(interp_result, JsValue::Smi(15));
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_lda_true_false() {
        let ba_true = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaTrue, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        assert_eq!(jit_run(&ba_true, &[]), JIT_TRUE);

        let ba_false = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaFalse, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        assert_eq!(jit_run(&ba_false, &[]), JIT_FALSE);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_logical_not() {
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaTrue, vec![]),
            Instruction::new_unchecked(Opcode::LogicalNot, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        assert_eq!(jit_run(&ba, &[]), JIT_FALSE);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_deopt_for_global_load() {
        let ba = bytecode(vec![
            Instruction::new_unchecked(
                Opcode::LdaGlobal,
                vec![Operand::ConstantPoolIdx(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let cc = BaselineCompiler::compile(&ba).expect("compile failed");
        // execute() should return Err("jit deopt") since LdaGlobal deopts.
        let result = unsafe { cc.execute(&[]) };
        assert!(
            matches!(result, Err(ref e) if e.to_string().contains("deopt")),
            "expected deopt error, got: {result:?}"
        );
    }

    // ── safepoint / deopt table tests (all platforms) ────────────────────────

    #[test]
    fn test_safepoint_gc_map_is_zero_for_smi_only() {
        // The current JIT tier only handles Smi/boolean/null/undefined values;
        // no slot ever holds a GC-managed heap pointer → gc_map must be zero.
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let cc = BaselineCompiler::compile(&ba).expect("compile failed");
        for sp in &cc.safepoints {
            assert_eq!(sp.gc_map, 0, "no GC pointers in Smi-only JIT code");
        }
    }

    #[test]
    fn test_deopt_liveness_map_covers_all_slots() {
        // 2 params + 3 locals = 5 register-file slots.
        // The deopt liveness_map must have bits 0–4 set (= 0b1_1111 = 31).
        let ba = BytecodeArray::new(
            encode(&[
                Instruction::new_unchecked(
                    Opcode::LdaGlobal,
                    vec![Operand::ConstantPoolIdx(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ]),
            vec![],
            3, // frame_size = 3 locals
            2, // parameter_count = 2 params
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );
        let cc = BaselineCompiler::compile(&ba).expect("compile failed");
        assert_eq!(cc.deopt_entries.len(), 1);
        let expected_liveness = (1u64 << 5) - 1; // bits 0..4 set
        assert_eq!(
            cc.deopt_entries[0].liveness_map, expected_liveness,
            "liveness_map must cover all 5 register-file slots"
        );
    }

    #[test]
    fn test_native_code_len_excludes_tables() {
        // The metadata tables are appended after the native instructions, so
        // native_code_len must be strictly less than the total code length.
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let cc = BaselineCompiler::compile(&ba).expect("compile failed");
        assert!(
            cc.native_code_len < cc.code.len(),
            "metadata tables must be appended after native code"
        );
        // The last FOOTER_SIZE bytes must contain the magic number.
        let footer_off = cc.code.len() - FOOTER_SIZE;
        let magic =
            u32::from_le_bytes(cc.code[footer_off + 8..footer_off + 12].try_into().unwrap());
        assert_eq!(
            magic, METADATA_MAGIC,
            "footer magic must match METADATA_MAGIC"
        );
    }

    #[test]
    fn test_tables_round_trip_through_serialized_code() {
        // The safepoint and deopt tables serialized into code bytes must be
        // parseable back to entries that are identical to the in-memory ones.
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
            Instruction::new_unchecked(
                Opcode::LdaGlobal,
                vec![Operand::ConstantPoolIdx(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let cc = BaselineCompiler::compile(&ba).expect("compile failed");

        // Round-trip safepoint table.
        let parsed_sp =
            CompiledCode::parse_safepoints(&cc.code).expect("safepoint table must be parseable");
        assert_eq!(parsed_sp.len(), cc.safepoints.len());
        for (parsed, orig) in parsed_sp.iter().zip(&cc.safepoints) {
            assert_eq!(parsed.code_offset, orig.code_offset);
            assert_eq!(parsed.bytecode_index, orig.bytecode_index);
            assert_eq!(parsed.gc_map, orig.gc_map);
        }

        // Round-trip deopt table.
        let parsed_de =
            CompiledCode::parse_deopt_entries(&cc.code).expect("deopt table must be parseable");
        assert_eq!(parsed_de.len(), cc.deopt_entries.len());
        for (parsed, orig) in parsed_de.iter().zip(&cc.deopt_entries) {
            assert_eq!(parsed.code_offset, orig.code_offset);
            assert_eq!(parsed.bytecode_offset, orig.bytecode_offset);
            assert_eq!(parsed.liveness_map, orig.liveness_map);
        }
    }

    #[test]
    fn test_simulate_gc_stack_scan() {
        // Simulate what the GC would do at a safepoint:
        //   1. Find the CompiledCode for the currently-executing JIT frame.
        //   2. Use the PC (code_offset) to look up the SafepointEntry.
        //   3. Read gc_map to determine which register-file slots hold GC roots.
        // In the current Smi-only JIT, gc_map must always be zero (no roots).
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let cc = BaselineCompiler::compile(&ba).expect("compile failed");

        // The first safepoint is at the very first instruction.
        let sp0 = &cc.safepoints[0];

        // Simulate GC: look up safepoint by code_offset.
        let found = cc
            .find_safepoint(sp0.code_offset)
            .expect("safepoint lookup must succeed");

        // No GC roots in a Smi-only JIT frame.
        assert_eq!(found.gc_map, 0, "no GC roots in Smi-only JIT frame");
        assert_eq!(found.bytecode_index, 0);

        // Also verify via the parsed tables (as the GC would use them from the
        // serialized code bytes embedded in the Code object).
        let parsed =
            CompiledCode::parse_safepoints(&cc.code).expect("must parse from serialized code");
        let found_parsed = parsed
            .iter()
            .find(|e| e.code_offset == sp0.code_offset)
            .expect("parsed safepoint lookup must succeed");
        assert_eq!(found_parsed.gc_map, 0);
    }
}
