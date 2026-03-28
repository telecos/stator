//! Maglev code generator.
//!
//! # Overview
//!
//! This module walks a register-allocated [`MaglevGraph`] and emits x86-64
//! machine code via the [`MacroAssembler`] from the baseline compiler tier.
//!
//! The code generator consumes a [`MaglevGraph`] together with an
//! [`AllocationResult`] produced by [`crate::compiler::maglev::regalloc`] and
//! produces a [`MaglevCompiledCode`] value containing the emitted bytes and
//! the associated metadata tables.
//!
//! # Calling convention
//!
//! The generated function uses the same calling convention as the baseline JIT:
//!
//! ```text
//! extern "C" fn(regs: *mut i64, ctx: i64) -> i64
//! ```
//!
//! `regs` is a caller-allocated array of `register_file_slots` × `i64` values:
//!
//! ```text
//! [ param[0], param[1], …, spill[0], spill[1], …, promoted_global[0], … ]
//!  ←── parameter_count ───→←──── spill_count ────→←─ promoted_extras ──→
//! ```
//!
//! `ctx` is the closure context raw pointer (0 if no context).  Stored in
//! `R15` by the prologue for use by context-slot load/store stubs.
//!
//! Parameters are loaded from the array at function entry.  Spilled values use
//! slots starting at index `parameter_count`.
//!
//! # Physical register bank
//!
//! The allocator assigns register indices 0 – [`NUM_PHYS_REGS`]-1.  They map
//! to x86-64 registers as follows:
//!
//! | Index | Register | ABI role           |
//! |-------|----------|--------------------|
//! | 0     | RBX      | callee-saved       |
//! | 1     | RCX      | general-purpose    |
//! | 2     | RDX      | general-purpose    |
//! | 3     | RSI      | general-purpose    |
//! | 4     | R8       | general-purpose    |
//! | 5     | R9       | general-purpose    |
//!
//! Reserved registers (not in the allocation pool):
//! - `R10`, `R11`: scratch (clobbered by helper sequences)
//! - `R14`: register-file base pointer (callee-saved; saved/restored)
//! - `RAX`: return value
//! - `RBP`, `RSP`: frame / stack pointers
//!
//! # Metadata tables
//!
//! The [`MaglevCompiledCode`] contains three serialized tables appended after
//! the native code bytes, followed by a 16-byte footer:
//!
//! - **Safepoint table** — one entry per basic block (GC-allowed points).
//! - **Deopt table** — one entry per deoptimisation site.
//! - **Source-position table** — reserved; currently empty.
//!
//! # Example
//!
//! ```
//! use stator_core::compiler::maglev::ir::{
//!     BasicBlock, ControlNode, MaglevGraph, ValueNode,
//! };
//! use stator_core::compiler::maglev::codegen::compile;
//!
//! let mut graph = MaglevGraph::new(0);
//! let mut block = BasicBlock::new(0);
//! let c = block.push_value(ValueNode::SmiConstant { value: 42 });
//! block.set_control(ControlNode::Return { value: c });
//! graph.add_block(block);
//!
//! let compiled = compile(&graph, 0).expect("codegen failed");
//! assert!(!compiled.code.is_empty());
//! ```

#[cfg(all(target_arch = "x86_64", unix))]
use crate::bytecode::bytecodes::Opcode;
#[cfg(all(target_arch = "x86_64", unix))]
use crate::compiler::baseline::compiler::jit_runtime;
use crate::compiler::baseline::compiler::{
    DeoptEntry, JIT_DEOPT, JIT_FALSE, JIT_NULL, JIT_TRUE, JIT_UNDEFINED, METADATA_MAGIC,
    SafepointEntry,
};
use crate::compiler::baseline::masm_x64::{CondCode, Label, MacroAssembler, Reg64};
use crate::compiler::maglev::ir::{ControlNode, MaglevGraph, NodeId, ValueNode};
use crate::compiler::maglev::regalloc::{AllocationResult, Location, allocate};
use crate::error::{StatorError, StatorResult};
use std::collections::HashSet;

// ─────────────────────────────────────────────────────────────────────────────
// Constants & helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Number of physical registers available to the Maglev register allocator.
pub const NUM_PHYS_REGS: u32 = 7;

/// A stub call argument that is either a Maglev IR node or an immediate i64.
#[cfg(all(target_arch = "x86_64", unix))]
enum NodeOrImm {
    Node(NodeId),
    Imm(i64),
}

// ─────────────────────────────────────────────────────────────────────────────
// Output types
// ─────────────────────────────────────────────────────────────────────────────

/// A single entry in the source-position table.
///
/// Maps a native code offset to a bytecode byte offset for source-level
/// debugging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourcePositionEntry {
    /// Byte offset in the JIT code buffer.
    pub code_offset: u32,
    /// Corresponding bytecode byte offset.
    pub bytecode_offset: u32,
}

/// The output of the Maglev code generator.
///
/// Contains the emitted x86-64 machine code followed by serialized metadata
/// tables and a 16-byte footer.  The footer layout (identical to the baseline
/// tier) is:
///
/// ```text
/// [0..4]   safepoint_table_start  (u32 LE)
/// [4..8]   deopt_table_start      (u32 LE)
/// [8..12]  source_pos_table_start (u32 LE)
/// [12..16] METADATA_MAGIC         (u32 LE)
/// ```
pub struct MaglevCompiledCode {
    /// Emitted machine code followed by serialized metadata tables and footer.
    pub code: Vec<u8>,
    /// Number of bytes in `code` that are native instructions.
    /// `code[native_code_len..]` contains the metadata and footer.
    pub native_code_len: usize,
    /// Total number of `i64` slots in the register file
    /// (`parameter_count + spill_count`).
    pub register_file_slots: usize,
    /// Safepoint table (one entry per basic block).
    pub safepoints: Vec<SafepointEntry>,
    /// Deoptimisation table (one entry per deopt site).
    pub deopt_entries: Vec<DeoptEntry>,
    /// Source-position table (currently empty; reserved for future use).
    pub source_positions: Vec<SourcePositionEntry>,
}

impl MaglevCompiledCode {
    /// Look up the safepoint entry whose `code_offset` equals `offset`.
    pub fn find_safepoint(&self, offset: u32) -> Option<&SafepointEntry> {
        self.safepoints.iter().find(|e| e.code_offset == offset)
    }

    /// Look up the deopt entry whose `code_offset` equals `offset`.
    pub fn find_deopt(&self, offset: u32) -> Option<&DeoptEntry> {
        self.deopt_entries.iter().find(|e| e.code_offset == offset)
    }

    /// Execute the compiled Maglev function on x86-64 Linux / macOS.
    ///
    /// Allocates a read-write-execute page with `mmap`, copies the native
    /// code bytes (the first `native_code_len` bytes of `code`), and calls
    /// the JIT function with a register file initialised from `args`.
    ///
    /// Missing argument slots are filled with `0` (`Smi(0)`); extra arguments
    /// are ignored.
    ///
    /// Returns the raw `i64` return value of the JIT function, or
    /// [`StatorError::Internal`] if `mmap` fails or the function returns
    /// [`JIT_DEOPT`].
    ///
    /// # Safety
    ///
    /// The `code` bytes must be valid x86-64 machine code emitted by
    /// [`compile`].  Executing arbitrary bytes via a transmuted function
    /// pointer is undefined behaviour.
    #[cfg(all(target_arch = "x86_64", unix))]
    pub unsafe fn execute(&self, args: &[i64]) -> StatorResult<i64> {
        use std::ptr;

        let code_size = self.native_code_len;
        if code_size == 0 {
            return Err(StatorError::Internal("compiled code is empty".into()));
        }

        // SAFETY: arguments are valid; MAP_FAILED is checked before use.
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

        // SAFETY: `mem` is valid and sized for `code_size` bytes.
        unsafe {
            ptr::copy_nonoverlapping(self.code.as_ptr(), mem.cast::<u8>(), code_size);
        }

        let mut regs = vec![0i64; self.register_file_slots];
        for (i, &v) in args.iter().enumerate().take(regs.len()) {
            regs[i] = v;
        }

        // SAFETY:
        // - `mem` holds valid x86-64 machine code produced by MaglevCodegen.
        // - Signature `extern "C" fn(*mut i64) -> i64` matches the JIT
        //   calling convention (SysV AMD64 ABI).
        // - `regs` remains live for the duration of the call.
        let result = unsafe {
            let f: extern "C" fn(*mut i64) -> i64 = std::mem::transmute(mem);
            f(regs.as_mut_ptr())
        };

        // SAFETY: `mem` is a valid mapping of `code_size` bytes.
        unsafe {
            libc::munmap(mem, code_size);
        }

        if result == JIT_DEOPT {
            Err(StatorError::Internal("maglev deopt".into()))
        } else {
            Ok(result)
        }
    }
}

/// Persistent executable Maglev code cache.
///
/// Unlike [`MaglevCompiledCode::execute`] which does `mmap`/`munmap` on every
/// call, this struct keeps the executable page alive for the lifetime of the
/// cache entry.  A thread-local register file is reused across calls to
/// eliminate per-call heap allocation.
#[cfg(all(target_arch = "x86_64", unix))]
pub struct CachedMaglevCode {
    ptr: *mut u8,
    size: usize,
    /// Total number of `i64` slots in the register file.
    pub register_file_slots: usize,
}

#[cfg(all(target_arch = "x86_64", unix))]
impl std::fmt::Debug for CachedMaglevCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedMaglevCode")
            .field("size", &self.size)
            .field("register_file_slots", &self.register_file_slots)
            .finish()
    }
}

// SAFETY: The mmap'd page is process-global memory.  We only access it
// through an `extern "C"` function-pointer call, which is thread-safe as
// long as the code is read-only (which it is after the initial memcpy).
#[cfg(all(target_arch = "x86_64", unix))]
unsafe impl Send for CachedMaglevCode {}
#[cfg(all(target_arch = "x86_64", unix))]
unsafe impl Sync for CachedMaglevCode {}

#[cfg(all(target_arch = "x86_64", unix))]
impl CachedMaglevCode {
    /// Create a new cached Maglev code entry from a compiled code buffer.
    ///
    /// Returns `None` if the code is empty or `mmap` fails.
    ///
    /// # Safety
    ///
    /// `code` must contain valid x86-64 machine code produced by
    /// [`compile`].
    pub unsafe fn new(code: &[u8], register_file_slots: usize) -> Option<Self> {
        use std::ptr;

        if code.is_empty() {
            return None;
        }

        // SAFETY: arguments are valid; MAP_FAILED is checked before use.
        let mem = unsafe {
            libc::mmap(
                ptr::null_mut(),
                code.len(),
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if mem == libc::MAP_FAILED {
            return None;
        }

        // SAFETY: `mem` is valid for `code.len()` bytes.
        unsafe {
            ptr::copy_nonoverlapping(code.as_ptr(), mem.cast::<u8>(), code.len());
        }

        Some(Self {
            ptr: mem.cast::<u8>(),
            size: code.len(),
            register_file_slots,
        })
    }

    /// Execute the cached Maglev code with the given arguments.
    ///
    /// Uses a thread-local register file pool to avoid per-call allocation.
    ///
    /// # Safety
    ///
    /// The cached code must be valid x86-64 machine code.
    pub unsafe fn execute(&self, args: &[i64]) -> i64 {
        let n = self.register_file_slots;

        // Fast path: stack-allocated register file avoids TLS.
        if n <= 16 {
            let mut regs = [0i64; 16];
            for (i, &v) in args.iter().enumerate().take(n) {
                regs[i] = v;
            }
            // SAFETY: `self.ptr` holds valid x86-64 machine code.
            let f: extern "C" fn(*mut i64) -> i64 = unsafe { std::mem::transmute(self.ptr) };
            return f(regs.as_mut_ptr());
        }

        // Large register files: pooled Vec via TLS.
        thread_local! {
            static MAGLEV_REG_FILE: std::cell::RefCell<Vec<i64>> = const {
                std::cell::RefCell::new(Vec::new())
            };
        }

        MAGLEV_REG_FILE.with(|pool| {
            let mut regs = pool.borrow_mut();
            regs.clear();
            regs.resize(n, 0);
            for (i, &v) in args.iter().enumerate().take(n) {
                regs[i] = v;
            }

            // SAFETY: `self.ptr` holds valid x86-64 machine code.
            let f: extern "C" fn(*mut i64) -> i64 = unsafe { std::mem::transmute(self.ptr) };
            f(regs.as_mut_ptr())
        })
    }
}

#[cfg(all(target_arch = "x86_64", unix))]
impl Drop for CachedMaglevCode {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` is a valid mmap'd region of `self.size` bytes.
        unsafe {
            libc::munmap(self.ptr.cast(), self.size);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Physical register mapping
// ─────────────────────────────────────────────────────────────────────────────

/// Map a register-allocator index (0 – [`NUM_PHYS_REGS`]-1) to the
/// corresponding x86-64 [`Reg64`].
///
/// # Panics
///
/// Panics when `n >= NUM_PHYS_REGS`.
fn phys_reg(n: u32) -> Reg64 {
    match n {
        0 => Reg64::Rbx,
        1 => Reg64::Rcx,
        2 => Reg64::Rdx,
        3 => Reg64::Rsi,
        4 => Reg64::R8,
        5 => Reg64::R9,
        6 => Reg64::R13,
        _ => panic!("phys_reg: index {n} out of range (NUM_PHYS_REGS = {NUM_PHYS_REGS})"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry-point
// ─────────────────────────────────────────────────────────────────────────────

/// Compile a [`MaglevGraph`] to native x86-64 machine code.
///
/// `param_count` is the number of formal parameters for the function.  The
/// caller should pass `graph.parameter_count()` here.
///
/// Register allocation is performed internally using [`NUM_PHYS_REGS`]
/// physical registers.
///
/// # Errors
///
/// Returns [`StatorError::Internal`] when the code buffer would exceed 4 GiB
/// (which cannot occur in practice with well-formed graphs).
pub fn compile(graph: &MaglevGraph, param_count: u32) -> StatorResult<MaglevCompiledCode> {
    let alloc = allocate(graph, NUM_PHYS_REGS);
    MaglevCodegen::new(graph, &alloc, param_count).compile()
}

// ─────────────────────────────────────────────────────────────────────────────
// MaglevCodegen
// ─────────────────────────────────────────────────────────────────────────────

/// Internal code-generation state for a single Maglev function.
struct MaglevCodegen<'a> {
    graph: &'a MaglevGraph,
    alloc: &'a AllocationResult,
    param_count: u32,
    masm: MacroAssembler,
    /// One label per basic block, indexed by block index.
    block_labels: Vec<Label>,
    /// Category-specific deopt labels (forward reference during block emission).
    deopt_label: Label,
    /// Category-specific deopt labels for diagnostics.
    deopt_overflow_label: Label,
    deopt_stub_label: Label,
    deopt_global_label: Label,
    deopt_loop_label: Label,
    deopt_divzero_label: Label,
    /// Common deopt exit (pops + ret, does NOT overwrite RAX).
    deopt_common_label: Label,
    safepoints: Vec<SafepointEntry>,
    deopt_entries: Vec<DeoptEntry>,
    source_positions: Vec<SourcePositionEntry>,
    /// Promoted globals: `(name_idx, slot_index)` where `slot_index` is the
    /// byte offset from R14 divided by 8.  These slots live after the
    /// allocator's spill area in the register file.
    promoted_globals: Vec<(u32, usize)>,
    /// Number of extra register-file slots allocated for promoted globals.
    promoted_extra_slots: usize,
}

impl<'a> MaglevCodegen<'a> {
    fn new(graph: &'a MaglevGraph, alloc: &'a AllocationResult, param_count: u32) -> Self {
        let num_blocks = graph.blocks().len();
        Self {
            graph,
            alloc,
            param_count,
            masm: MacroAssembler::new(),
            block_labels: (0..num_blocks).map(|_| Label::new()).collect(),
            deopt_label: Label::new(),
            deopt_overflow_label: Label::new(),
            deopt_stub_label: Label::new(),
            deopt_global_label: Label::new(),
            deopt_loop_label: Label::new(),
            deopt_divzero_label: Label::new(),
            deopt_common_label: Label::new(),
            safepoints: Vec::new(),
            deopt_entries: Vec::new(),
            source_positions: Vec::new(),
            promoted_globals: Vec::new(),
            promoted_extra_slots: 0,
        }
    }

    // ── Top-level compilation ────────────────────────────────────────────────

    fn compile(mut self) -> StatorResult<MaglevCompiledCode> {
        let spill_count = self.alloc.spill_count();
        let base_slots = (self.param_count + spill_count) as usize;

        // Scan the IR graph and promote every unique global name into an
        // extra register-file slot beyond the allocator's spill area.
        self.scan_and_promote_globals(base_slots);
        let register_file_slots = base_slots + self.promoted_extra_slots;

        self.emit_prologue();
        self.emit_promoted_global_loads();

        for block_idx in 0..self.graph.blocks().len() {
            self.emit_block(block_idx as u32);
        }

        self.emit_deopt_epilogue();

        let mut code = self.masm.into_code();
        let native_code_len = code.len();

        // ── Serialize safepoint table ────────────────────────────────────────
        let safepoint_table_start = u32::try_from(code.len())
            .map_err(|_| StatorError::Internal("compiled code exceeds 4 GiB limit".into()))?;
        let sp_count = self.safepoints.len() as u32;
        code.extend_from_slice(&sp_count.to_le_bytes());
        for e in &self.safepoints {
            code.extend_from_slice(&e.code_offset.to_le_bytes());
            code.extend_from_slice(&e.bytecode_index.to_le_bytes());
            code.extend_from_slice(&e.gc_map.to_le_bytes());
        }

        // ── Serialize deopt table ────────────────────────────────────────────
        let deopt_table_start = u32::try_from(code.len())
            .map_err(|_| StatorError::Internal("compiled code exceeds 4 GiB limit".into()))?;
        let de_count = self.deopt_entries.len() as u32;
        code.extend_from_slice(&de_count.to_le_bytes());
        for e in &self.deopt_entries {
            code.extend_from_slice(&e.code_offset.to_le_bytes());
            code.extend_from_slice(&e.bytecode_offset.to_le_bytes());
            code.extend_from_slice(&e.liveness_map.to_le_bytes());
        }

        // ── Serialize source-position table ─────────────────────────────────
        let source_pos_table_start = u32::try_from(code.len())
            .map_err(|_| StatorError::Internal("compiled code exceeds 4 GiB limit".into()))?;
        let sp_pos_count = self.source_positions.len() as u32;
        code.extend_from_slice(&sp_pos_count.to_le_bytes());
        for e in &self.source_positions {
            code.extend_from_slice(&e.code_offset.to_le_bytes());
            code.extend_from_slice(&e.bytecode_offset.to_le_bytes());
        }

        // ── Serialize 16-byte footer ─────────────────────────────────────────
        code.extend_from_slice(&safepoint_table_start.to_le_bytes());
        code.extend_from_slice(&deopt_table_start.to_le_bytes());
        code.extend_from_slice(&source_pos_table_start.to_le_bytes());
        code.extend_from_slice(&METADATA_MAGIC.to_le_bytes());

        Ok(MaglevCompiledCode {
            code,
            native_code_len,
            register_file_slots,
            safepoints: self.safepoints,
            deopt_entries: self.deopt_entries,
            source_positions: self.source_positions,
        })
    }

    // ── Prologue / epilogue ──────────────────────────────────────────────────

    /// Emit the standard function prologue.
    ///
    /// ```text
    /// push  rbp
    /// mov   rbp, rsp
    /// push  rbx         ; [RBP-8]  callee-saved Register(0) holder
    /// push  r14         ; [RBP-16] callee-saved reg-file pointer
    /// push  r13         ; [RBP-24] callee-saved (preserved for caller)
    /// sub   rsp, 8      ; [RBP-32] stack-based loop safety counter
    /// mov   r14, rdi    ; r14 = regs argument (SysV: first arg in RDI)
    /// mov   r11, LOOP_COUNTER_MAX
    /// mov   [rbp-32], r11
    /// ```
    fn emit_prologue(&mut self) {
        self.masm.push(Reg64::Rbp);
        self.masm.mov_rr(Reg64::Rbp, Reg64::Rsp);
        self.masm.push(Reg64::Rbx);
        self.masm.push(Reg64::R14);
        self.masm.push(Reg64::R13);
        // After 4 pushes (including return-address push by `call`):
        // RSP ≡ 8 mod 16.  Stub calls do 5-push save which aligns
        // RSP to 0 mod 16 before the inner `call`.
        self.masm.mov_rr(Reg64::R14, Reg64::Rdi);
    }

    /// Emit the normal function return sequence.
    ///
    /// `rax` must already hold the return value before calling this.
    ///
    /// ```text
    /// pop  r13
    /// pop  r14
    /// pop  rbx
    /// pop  rbp
    /// ret
    /// ```
    fn emit_normal_epilogue(&mut self) {
        self.masm.pop(Reg64::R13);
        self.masm.pop(Reg64::R14);
        self.masm.pop(Reg64::Rbx);
        self.masm.pop(Reg64::Rbp);
        self.masm.ret();
    }

    /// Emit the shared deopt epilogue with categorised deopt labels.
    ///
    /// Category labels set RAX to their specific constant and jump to
    /// `deopt_common_label`.  The uncategorised `deopt_label` sets
    /// `JIT_DEOPT` and falls through to common.
    fn emit_deopt_epilogue(&mut self) {
        use crate::compiler::baseline::compiler::{
            JIT_DEOPT_DIVZERO, JIT_DEOPT_GLOBAL, JIT_DEOPT_LOOP, JIT_DEOPT_OVERFLOW, JIT_DEOPT_STUB,
        };

        // Category-specific entry points.
        self.masm.bind_label(&mut self.deopt_overflow_label);
        self.masm.mov_ri(Reg64::Rax, JIT_DEOPT_OVERFLOW);
        self.masm.jmp(&mut self.deopt_common_label);

        self.masm.bind_label(&mut self.deopt_stub_label);
        self.masm.mov_ri(Reg64::Rax, JIT_DEOPT_STUB);
        self.masm.jmp(&mut self.deopt_common_label);

        self.masm.bind_label(&mut self.deopt_global_label);
        self.masm.mov_ri(Reg64::Rax, JIT_DEOPT_GLOBAL);
        self.masm.jmp(&mut self.deopt_common_label);

        self.masm.bind_label(&mut self.deopt_loop_label);
        self.masm.mov_ri(Reg64::Rax, JIT_DEOPT_LOOP);
        self.masm.jmp(&mut self.deopt_common_label);

        self.masm.bind_label(&mut self.deopt_divzero_label);
        self.masm.mov_ri(Reg64::Rax, JIT_DEOPT_DIVZERO);
        self.masm.jmp(&mut self.deopt_common_label);

        // Uncategorised fallback (existing deopt paths that still use
        // deopt_label directly).
        self.masm.bind_label(&mut self.deopt_label);
        self.masm.mov_ri(Reg64::Rax, JIT_DEOPT);

        // Common exit — RAX already set, just restore and return.
        self.masm.bind_label(&mut self.deopt_common_label);
        self.masm.pop(Reg64::R13);
        self.masm.pop(Reg64::R14);
        self.masm.pop(Reg64::Rbx);
        self.masm.pop(Reg64::Rbp);
        self.masm.ret();
    }

    // ── Block emission ───────────────────────────────────────────────────────

    fn emit_block(&mut self, block_idx: u32) {
        let block = match self.graph.block(block_idx) {
            Some(b) => b,
            None => return,
        };

        // Bind this block's label to the current code position.
        self.masm
            .bind_label(&mut self.block_labels[block_idx as usize]);

        // Record one safepoint per block (conservative GC point).
        self.safepoints.push(SafepointEntry {
            code_offset: self.masm.position() as u32,
            bytecode_index: block_idx,
            gc_map: 0,
        });

        // Clone to avoid borrow conflicts during emission.
        let nodes: Vec<(NodeId, ValueNode)> = block.nodes.clone();
        let control = block.control.clone();

        for (id, node) in &nodes {
            self.emit_value_node(*id, node);
        }

        if let Some(ctrl) = control {
            self.emit_control_node(block_idx, &ctrl);
        }
    }

    // ── Value node emission ──────────────────────────────────────────────────

    /// Emit machine code for one [`ValueNode`].
    ///
    /// The computed result is written to the node's allocated [`Location`]
    /// (physical register or spill slot).  Inputs are loaded via the `R11` /
    /// `R10` scratch registers.
    #[allow(clippy::too_many_lines)]
    fn emit_value_node(&mut self, id: NodeId, node: &ValueNode) {
        match node {
            // ── Constants ────────────────────────────────────────────────────
            ValueNode::SmiConstant { value } => {
                self.masm.mov_ri(Reg64::R11, *value as i64);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::Int32Constant { value } => {
                self.masm.mov_ri(Reg64::R11, *value as i64);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::Uint32Constant { value } => {
                self.masm.mov_ri(Reg64::R11, *value as i64);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::Float64Constant { value } => {
                // Store the f64 bit pattern as an i64.
                self.masm.mov_ri(Reg64::R11, value.to_bits() as i64);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::TrueConstant => {
                self.masm.mov_ri(Reg64::R11, JIT_TRUE);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::FalseConstant => {
                self.masm.mov_ri(Reg64::R11, JIT_FALSE);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::NullConstant => {
                self.masm.mov_ri(Reg64::R11, JIT_NULL);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::UndefinedConstant => {
                self.masm.mov_ri(Reg64::R11, JIT_UNDEFINED);
                self.emit_store(id, Reg64::R11);
            }

            // ── Parameters ───────────────────────────────────────────────────
            ValueNode::Parameter { index } => {
                let off = (*index * 8) as i32;
                self.masm.mov_load_base_disp32(Reg64::R11, Reg64::R14, off);
                self.emit_store(id, Reg64::R11);
            }

            // ── Int32 binary arithmetic ──────────────────────────────────────
            ValueNode::Int32Add { left, right } => {
                self.emit_load(*left, Reg64::R11);
                self.emit_load(*right, Reg64::R10);
                self.masm.add_rr(Reg64::R11, Reg64::R10);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::Int32Subtract { left, right } => {
                self.emit_load(*left, Reg64::R11);
                self.emit_load(*right, Reg64::R10);
                self.masm.sub_rr(Reg64::R11, Reg64::R10);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::Int32Multiply { left, right } => {
                self.emit_load(*left, Reg64::R11);
                self.emit_load(*right, Reg64::R10);
                self.masm.imul_rr(Reg64::R11, Reg64::R10);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::Int32BitwiseAnd { left, right } => {
                self.emit_load(*left, Reg64::R11);
                self.emit_load(*right, Reg64::R10);
                self.emit_and_rr(Reg64::R11, Reg64::R10);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::Int32BitwiseOr { left, right } => {
                self.emit_load(*left, Reg64::R11);
                self.emit_load(*right, Reg64::R10);
                self.emit_or_rr(Reg64::R11, Reg64::R10);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::Int32BitwiseXor { left, right } => {
                self.emit_load(*left, Reg64::R11);
                self.emit_load(*right, Reg64::R10);
                self.masm.xor_rr(Reg64::R11, Reg64::R10);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::Int32ShiftLeft { left, right } => {
                self.emit_load(*left, Reg64::R11);
                self.emit_load(*right, Reg64::Rcx);
                // SHL r11, cl — REX.B D3 /4 r/m=R11(enc=3)
                // ModRM: mod=11(0xC0) /4(0x20) r/m=R11.enc()=3 → 0xE3
                self.masm.emit_byte(0x49); // REX.B (R11 needs REX.B)
                self.masm.emit_byte(0xD3);
                self.masm.emit_byte(0xE3); // mod=11, /4, r/m=3
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::Int32ShiftRight { left, right } => {
                self.emit_load(*left, Reg64::R11);
                self.emit_load(*right, Reg64::Rcx);
                // SAR r11, cl — REX.B D3 /7 r/m=R11(enc=3)
                // ModRM: mod=11(0xC0) /7(0x38) r/m=3 → 0xFB
                self.masm.emit_byte(0x49);
                self.masm.emit_byte(0xD3);
                self.masm.emit_byte(0xFB); // mod=11, /7, r/m=3
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::Int32ShiftRightLogical { left, right } => {
                self.emit_load(*left, Reg64::R11);
                self.emit_load(*right, Reg64::Rcx);
                // SHR r11, cl — REX.B D3 /5 r/m=R11(enc=3)
                // ModRM: mod=11(0xC0) /5(0x28) r/m=3 → 0xEB
                self.masm.emit_byte(0x49);
                self.masm.emit_byte(0xD3);
                self.masm.emit_byte(0xEB); // mod=11, /5, r/m=3
                self.emit_store(id, Reg64::R11);
            }

            // ── Int32 unary arithmetic ───────────────────────────────────────
            ValueNode::Int32Negate { value } => {
                self.emit_load(*value, Reg64::R11);
                self.masm.neg_r(Reg64::R11);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::Int32Increment { value } => {
                self.emit_load(*value, Reg64::R11);
                self.masm.add_ri(Reg64::R11, 1);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::Int32Decrement { value } => {
                self.emit_load(*value, Reg64::R11);
                self.masm.sub_ri(Reg64::R11, 1);
                self.emit_store(id, Reg64::R11);
            }

            // ── Checked Smi arithmetic (deopt on signed 64-bit overflow) ─────
            //
            // We detect overflow by checking whether the 64-bit result can be
            // faithfully represented in 32 bits: perform the operation, then
            // compare the result's low 32 bits (sign-extended to 64) with the
            // full 64-bit result.  If they differ, the Smi range was exceeded.
            ValueNode::CheckedSmiAdd { left, right } => {
                self.emit_load(*left, Reg64::R11);
                self.emit_load(*right, Reg64::R10);
                self.masm.add_rr(Reg64::R11, Reg64::R10);
                self.emit_deopt_on_smi_overflow(0);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::CheckedSmiSubtract { left, right } => {
                self.emit_load(*left, Reg64::R11);
                self.emit_load(*right, Reg64::R10);
                self.masm.sub_rr(Reg64::R11, Reg64::R10);
                self.emit_deopt_on_smi_overflow(0);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::CheckedSmiMultiply { left, right } => {
                self.emit_load(*left, Reg64::R11);
                self.emit_load(*right, Reg64::R10);
                self.masm.imul_rr(Reg64::R11, Reg64::R10);
                self.emit_deopt_on_smi_overflow(0);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::CheckedSmiIncrement { value } => {
                self.emit_load(*value, Reg64::R11);
                self.masm.add_ri(Reg64::R11, 1);
                self.emit_deopt_on_smi_overflow(0);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::CheckedSmiDecrement { value } => {
                self.emit_load(*value, Reg64::R11);
                self.masm.sub_ri(Reg64::R11, 1);
                self.emit_deopt_on_smi_overflow(0);
                self.emit_store(id, Reg64::R11);
            }

            // ── Int32 comparisons ────────────────────────────────────────────
            ValueNode::Int32Equal { left, right } | ValueNode::Int32StrictEqual { left, right } => {
                self.emit_int32_compare(*left, *right, CondCode::Equal, id);
            }
            ValueNode::Int32LessThan { left, right } => {
                self.emit_int32_compare(*left, *right, CondCode::Less, id);
            }
            ValueNode::Int32LessThanOrEqual { left, right } => {
                self.emit_int32_compare(*left, *right, CondCode::LessEq, id);
            }
            ValueNode::Int32GreaterThan { left, right } => {
                self.emit_int32_compare(*left, *right, CondCode::Greater, id);
            }
            ValueNode::Int32GreaterThanOrEqual { left, right } => {
                self.emit_int32_compare(*left, *right, CondCode::GreaterEq, id);
            }

            // ── Type identity conversions ────────────────────────────────────
            //
            // In the JIT's flat `i64` representation, the bit patterns for
            // unboxed int32 / uint32 and tagged Smi are identical for values
            // within the Smi range.  These conversions are therefore no-ops.
            ValueNode::ChangeInt32ToTagged { input }
            | ValueNode::ChangeUint32ToTagged { input }
            | ValueNode::ChangeInt32ToFloat64 { input }
            | ValueNode::ChangeUint32ToFloat64 { input }
            | ValueNode::ChangeTaggedToInt32 { input }
            | ValueNode::ChangeTaggedToUint32 { input }
            | ValueNode::ChangeTaggedToFloat64 { input } => {
                if self.alloc.location(id) != self.alloc.location(*input) {
                    self.emit_load(*input, Reg64::R11);
                    self.emit_store(id, Reg64::R11);
                }
            }

            // ── Boolean conversion ───────────────────────────────────────────
            //
            // Produce JIT_FALSE when the input is zero (Smi 0), JIT_FALSE, or
            // JIT_NULL / JIT_UNDEFINED.  Produce JIT_TRUE otherwise.
            // Implementation: compare with 0; if non-zero → true, else false.
            ValueNode::ToBoolean { value } => {
                self.emit_load(*value, Reg64::R11);
                self.masm.cmp_ri(Reg64::R11, 0);
                self.masm.setcc_al(CondCode::NotEqual);
                self.masm.movzx_r64_al(Reg64::R11);
                self.masm.mov_ri(Reg64::R10, JIT_FALSE);
                self.masm.add_rr(Reg64::R11, Reg64::R10);
                self.emit_store(id, Reg64::R11);
            }

            // ── Phi ──────────────────────────────────────────────────────────
            //
            // Phi-resolution copies are emitted at predecessor jump/branch
            // sites by `emit_phi_copies_for_successor`.  By the time the JIT
            // code for this block is executed, the correct value is already in
            // the phi's allocated location.  Emit no code here.
            ValueNode::Phi { .. } => {}

            // ── Guards ───────────────────────────────────────────────────
            //
            // CheckSmi verifies the value fits in the Smi range (i32).
            ValueNode::CheckSmi { receiver } | ValueNode::CheckNumber { receiver } => {
                self.emit_load(*receiver, Reg64::R11);
                self.emit_deopt_on_smi_overflow(0);
                self.emit_store(id, Reg64::R11);
            }
            // CheckInt32IsSmi: every i32 is a valid Smi — pass-through.
            ValueNode::CheckInt32IsSmi { input } => {
                // Skip redundant copy if input and output share a location.
                if self.alloc.location(id) != self.alloc.location(*input) {
                    self.emit_load(*input, Reg64::R11);
                    self.emit_store(id, Reg64::R11);
                }
            }
            // CheckUint32IsSmi: valid when the u32 fits in i32 (bit 31 clear).
            ValueNode::CheckUint32IsSmi { input } | ValueNode::CheckHoleyFloat64IsSmi { input } => {
                self.emit_load(*input, Reg64::R11);
                self.emit_deopt_on_smi_overflow(0);
                self.emit_store(id, Reg64::R11);
            }

            // ── Int32 division and modulus ────────────────────────────────
            //
            // x86-64 IDIV uses RAX/RDX, so we save/restore RDX (which is
            // the allocatable register phys_reg(2)).
            ValueNode::Int32Divide { left, right } => {
                self.emit_load(*left, Reg64::R11);
                self.emit_load(*right, Reg64::R10);
                self.emit_div_zero_check(0);
                self.masm.push(Reg64::Rdx);
                self.emit_idiv_sequence();
                self.masm.mov_rr(Reg64::R11, Reg64::Rax);
                self.masm.pop(Reg64::Rdx);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::Int32Modulus { left, right } => {
                self.emit_load(*left, Reg64::R11);
                self.emit_load(*right, Reg64::R10);
                self.emit_div_zero_check(0);
                self.masm.push(Reg64::Rdx);
                self.emit_idiv_sequence();
                self.masm.mov_rr(Reg64::R11, Reg64::Rdx);
                self.masm.pop(Reg64::Rdx);
                self.emit_store(id, Reg64::R11);
            }

            // ── Checked Smi division (deopt on div-by-zero or overflow) ──
            ValueNode::CheckedSmiDivide { left, right } => {
                self.emit_load(*left, Reg64::R11);
                self.emit_load(*right, Reg64::R10);
                self.emit_div_zero_check(0);
                self.masm.push(Reg64::Rdx);
                self.emit_idiv_sequence();
                self.masm.mov_rr(Reg64::R11, Reg64::Rax);
                self.masm.pop(Reg64::Rdx);
                self.emit_deopt_on_smi_overflow(0);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::CheckedSmiModulus { left, right } => {
                self.emit_load(*left, Reg64::R11);
                self.emit_load(*right, Reg64::R10);
                self.emit_div_zero_check(0);
                self.masm.push(Reg64::Rdx);
                self.emit_idiv_sequence();
                self.masm.mov_rr(Reg64::R11, Reg64::Rdx);
                self.masm.pop(Reg64::Rdx);
                self.emit_deopt_on_smi_overflow(0);
                self.emit_store(id, Reg64::R11);
            }

            // ── Uint32 binary arithmetic ─────────────────────────────────
            //
            // At the i64 level, unsigned and signed add/sub/mul produce the
            // same bit pattern; we reuse the existing instructions.
            ValueNode::Uint32Add { left, right } => {
                self.emit_load(*left, Reg64::R11);
                self.emit_load(*right, Reg64::R10);
                self.masm.add_rr(Reg64::R11, Reg64::R10);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::Uint32Subtract { left, right } => {
                self.emit_load(*left, Reg64::R11);
                self.emit_load(*right, Reg64::R10);
                self.masm.sub_rr(Reg64::R11, Reg64::R10);
                self.emit_store(id, Reg64::R11);
            }
            ValueNode::Uint32Multiply { left, right } => {
                self.emit_load(*left, Reg64::R11);
                self.emit_load(*right, Reg64::R10);
                self.masm.imul_rr(Reg64::R11, Reg64::R10);
                self.emit_store(id, Reg64::R11);
            }

            // ── Global variable access via promoted register-file slots ────
            //
            // Each unique global name is pre-loaded into a dedicated
            // register-file slot during the prologue.  LoadGlobal reads from
            // the slot; StoreGlobal writes to it.  The epilogue flushes all
            // promoted slots back to the runtime GlobalEnv.
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::LoadGlobal {
                name,
                feedback_slot,
            } => {
                let _ = feedback_slot;
                if let Some(off) = self.promoted_global_offset(*name) {
                    // Load directly into the allocated register when possible,
                    // avoiding an intermediate R11 copy.
                    match self.alloc.location(id) {
                        Some(Location::Register(n)) => {
                            let dst = phys_reg(n);
                            self.masm.mov_load_base_disp32(dst, Reg64::R14, off);
                        }
                        _ => {
                            self.masm.mov_load_base_disp32(Reg64::R11, Reg64::R14, off);
                            self.emit_store(id, Reg64::R11);
                        }
                    }
                } else {
                    self.emit_save_caller_saved();
                    self.masm.mov_ri(Reg64::Rdi, i64::from(*name));
                    let addr = jit_runtime::jit_runtime_lda_global as *const () as usize as i64;
                    self.masm.mov_ri(Reg64::R11, addr);
                    self.masm.call_reg(Reg64::R11);
                    self.emit_restore_caller_saved();
                    self.emit_deopt_check_rax();
                    self.emit_store(id, Reg64::Rax);
                }
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::StoreGlobal {
                name,
                value,
                feedback_slot,
            } => {
                let _ = feedback_slot;
                if let Some(off) = self.promoted_global_offset(*name) {
                    // Store directly from the allocated register when possible.
                    match self.alloc.location(*value) {
                        Some(Location::Register(n)) => {
                            let src = phys_reg(n);
                            self.masm.mov_store_base_disp32(Reg64::R14, off, src);
                        }
                        _ => {
                            self.emit_load(*value, Reg64::R11);
                            self.masm.mov_store_base_disp32(Reg64::R14, off, Reg64::R11);
                        }
                    }
                } else {
                    self.emit_save_caller_saved();
                    self.masm.mov_ri(Reg64::Rdi, i64::from(*name));
                    self.emit_load(*value, Reg64::Rsi);
                    let addr = jit_runtime::jit_runtime_sta_global as *const () as usize as i64;
                    self.masm.mov_ri(Reg64::R11, addr);
                    self.masm.call_reg(Reg64::R11);
                    self.emit_restore_caller_saved();
                    self.emit_deopt_check_rax();
                    self.emit_store(id, Reg64::Rax);
                }
            }
            #[cfg(not(all(target_arch = "x86_64", unix)))]
            ValueNode::LoadGlobal { .. } | ValueNode::StoreGlobal { .. } => {
                self.emit_store(id, Reg64::R11);
            }

            // ── Property access via runtime stubs ─────────────────────────────
            //
            // These delegate to the baseline JIT's runtime stubs.  We save and
            // restore all caller-saved allocatable registers around the call.
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::LoadNamedGeneric {
                object,
                name,
                feedback_slot,
            } => {
                self.emit_stub_call_3arg(
                    id,
                    *object,
                    i64::from(*name),
                    NodeOrImm::Imm(i64::from(*feedback_slot)),
                    jit_runtime::jit_runtime_lda_named_property as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::StoreNamedGeneric {
                object,
                name,
                value,
                feedback_slot: _,
            } => {
                self.emit_stub_call_3arg(
                    id,
                    *object,
                    i64::from(*name),
                    NodeOrImm::Node(*value),
                    jit_runtime::jit_runtime_sta_named_property as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::LoadKeyedGeneric { object, key, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *object,
                    *key,
                    jit_runtime::jit_runtime_lda_keyed_property as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::StoreKeyedGeneric {
                object, key, value, ..
            } => {
                self.emit_stub_call_3node(
                    id,
                    *object,
                    *key,
                    *value,
                    jit_runtime::jit_runtime_sta_keyed_property as usize,
                );
            }

            // ── Guards ── pass-through when we use generic stubs ──────────────
            ValueNode::CheckMaps { receiver, .. }
            | ValueNode::CheckMapsWithMigration { receiver, .. } => {
                // The generic property stubs handle shape-checking internally.
                // Pass the receiver through unchanged.
                if self.alloc.location(id) != self.alloc.location(*receiver) {
                    self.emit_load(*receiver, Reg64::R11);
                    self.emit_store(id, Reg64::R11);
                }
            }

            // ── Context slot access via runtime stubs ─────────────────────────
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::LoadCurrentContextSlot { slot } => {
                // jit_runtime_lda_context_slot(slot_idx: i64) -> i64
                self.emit_save_caller_saved();
                self.masm.mov_ri(Reg64::Rdi, i64::from(*slot));
                self.masm.mov_ri(
                    Reg64::R11,
                    jit_runtime::jit_runtime_lda_context_slot as usize as i64,
                );
                self.masm.call_reg(Reg64::R11);
                self.emit_restore_caller_saved();
                self.emit_deopt_check_rax();
                self.masm.mov_rr(Reg64::R11, Reg64::Rax);
                self.emit_store(id, Reg64::R11);
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::StoreCurrentContextSlot { slot, value } => {
                // jit_runtime_sta_context_slot(slot_idx: i64, value: i64) -> i64
                self.emit_save_caller_saved();
                self.masm.mov_ri(Reg64::Rdi, i64::from(*slot));
                self.emit_load(*value, Reg64::Rsi);
                self.masm.mov_ri(
                    Reg64::R11,
                    jit_runtime::jit_runtime_sta_context_slot as usize as i64,
                );
                self.masm.call_reg(Reg64::R11);
                self.emit_restore_caller_saved();
                self.emit_deopt_check_rax();
                self.masm.mov_rr(Reg64::R11, Reg64::Rax);
                self.emit_store(id, Reg64::R11);
            }

            // ── Function calls via runtime stubs ──────────────────────────────
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::Call {
                callee,
                receiver,
                args,
                ..
            } => {
                // Check receiver: if it's UndefinedConstant, use the
                // CallUndefinedReceiver stubs; otherwise use CallProperty.
                let recv_is_undef = matches!(
                    self.graph.node(*receiver),
                    Some(ValueNode::UndefinedConstant)
                );
                match (recv_is_undef, args.len()) {
                    // CallUndefinedReceiver0
                    (true, 0) => {
                        self.emit_stub_call_1node(
                            id,
                            *callee,
                            jit_runtime::jit_runtime_call_undefined_receiver0 as usize,
                        );
                    }
                    // CallUndefinedReceiver1
                    (true, 1) => {
                        self.emit_stub_call_2node(
                            id,
                            *callee,
                            args[0],
                            jit_runtime::jit_runtime_call_undefined_receiver1 as usize,
                        );
                    }
                    // CallUndefinedReceiver2
                    (true, 2) => {
                        self.emit_stub_call_3node(
                            id,
                            *callee,
                            args[0],
                            args[1],
                            jit_runtime::jit_runtime_call_undefined_receiver2 as usize,
                        );
                    }
                    // CallProperty0
                    (false, 0) => {
                        self.emit_stub_call_2node(
                            id,
                            *callee,
                            *receiver,
                            jit_runtime::jit_runtime_call_property0 as usize,
                        );
                    }
                    // CallProperty1
                    (false, 1) => {
                        self.emit_stub_call_3node(
                            id,
                            *callee,
                            *receiver,
                            args[0],
                            jit_runtime::jit_runtime_call_property1 as usize,
                        );
                    }
                    // Unsupported arity — deopt.
                    _ => {
                        self.emit_deopt_unconditional(0);
                        self.masm.mov_ri(Reg64::R11, JIT_UNDEFINED);
                        self.emit_store(id, Reg64::R11);
                    }
                }
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::CallKnownFunction {
                callee,
                receiver,
                args,
                ..
            } => {
                let recv_is_undef = matches!(
                    self.graph.node(*receiver),
                    Some(ValueNode::UndefinedConstant)
                );
                match (recv_is_undef, args.len()) {
                    (true, 0) => {
                        self.emit_stub_call_1node(
                            id,
                            *callee,
                            jit_runtime::jit_runtime_call_undefined_receiver0 as usize,
                        );
                    }
                    (true, 1) => {
                        self.emit_stub_call_2node(
                            id,
                            *callee,
                            args[0],
                            jit_runtime::jit_runtime_call_undefined_receiver1 as usize,
                        );
                    }
                    (true, 2) => {
                        self.emit_stub_call_3node(
                            id,
                            *callee,
                            args[0],
                            args[1],
                            jit_runtime::jit_runtime_call_undefined_receiver2 as usize,
                        );
                    }
                    (false, 0) => {
                        self.emit_stub_call_2node(
                            id,
                            *callee,
                            *receiver,
                            jit_runtime::jit_runtime_call_property0 as usize,
                        );
                    }
                    (false, 1) => {
                        self.emit_stub_call_3node(
                            id,
                            *callee,
                            *receiver,
                            args[0],
                            jit_runtime::jit_runtime_call_property1 as usize,
                        );
                    }
                    _ => {
                        self.emit_deopt_unconditional(0);
                        self.masm.mov_ri(Reg64::R11, JIT_UNDEFINED);
                        self.emit_store(id, Reg64::R11);
                    }
                }
            }

            // ── Object / array / closure creation via trampoline ──────────────
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::CreateObjectLiteral {
                feedback_slot,
                flags,
                ..
            } => {
                self.emit_trampoline_call(
                    id,
                    Opcode::CreateObjectLiteral as u8,
                    i64::from(*feedback_slot),
                    i64::from(*flags),
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::CreateShallowObjectLiteral {
                feedback_slot,
                flags,
                ..
            } => {
                self.emit_trampoline_call(
                    id,
                    Opcode::CreateObjectLiteral as u8,
                    i64::from(*feedback_slot),
                    i64::from(*flags),
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::CreateEmptyObjectLiteral => {
                self.emit_trampoline_call(id, Opcode::CreateEmptyObjectLiteral as u8, 0, 0);
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::CreateArrayLiteral { feedback_slot, .. } => {
                self.emit_trampoline_call(
                    id,
                    Opcode::CreateArrayLiteral as u8,
                    i64::from(*feedback_slot),
                    0,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::CreateShallowArrayLiteral { feedback_slot, .. } => {
                self.emit_trampoline_call(
                    id,
                    Opcode::CreateArrayLiteral as u8,
                    i64::from(*feedback_slot),
                    0,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::CreateClosure {
                shared_function_info,
                ..
            } => {
                self.emit_trampoline_call(
                    id,
                    Opcode::CreateClosure as u8,
                    i64::from(*shared_function_info),
                    0,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::FastCreateClosure {
                shared_function_info,
                ..
            } => {
                self.emit_trampoline_call(
                    id,
                    Opcode::CreateClosure as u8,
                    i64::from(*shared_function_info),
                    0,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::CreateFunctionContext { slot_count, .. } => {
                self.emit_trampoline_call(
                    id,
                    Opcode::CreateFunctionContext as u8,
                    i64::from(*slot_count),
                    0,
                );
            }

            // ── Construct via dedicated stub ──────────────────────────────────
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::Construct {
                constructor, args, ..
            } => {
                // Simplified: only handle 0-arg construct.
                if args.is_empty() {
                    self.emit_stub_call_1node(
                        id,
                        *constructor,
                        jit_runtime::jit_runtime_construct0 as usize,
                    );
                } else {
                    self.emit_deopt_unconditional(0);
                    self.masm.mov_ri(Reg64::R11, JIT_UNDEFINED);
                    self.emit_store(id, Reg64::R11);
                }
            }

            // ── Generic arithmetic via dedicated stubs ────────────────────────
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericAdd { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_generic_add as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericSubtract { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_generic_sub as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericMultiply { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_generic_mul as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericDivide { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_generic_div as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericModulus { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_generic_mod as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericBitwiseAnd { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_generic_bitwise_and as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericBitwiseOr { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_generic_bitwise_or as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericBitwiseXor { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_generic_bitwise_xor as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericShiftLeft { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_generic_shift_left as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericShiftRight { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_generic_shift_right as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericShiftRightLogical { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_generic_shift_right_logical as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericNegate { value, .. } => {
                self.emit_stub_call_1node(
                    id,
                    *value,
                    jit_runtime::jit_runtime_generic_negate as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericIncrement { value, .. } => {
                self.emit_stub_call_1node(
                    id,
                    *value,
                    jit_runtime::jit_runtime_generic_increment as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericDecrement { value, .. } => {
                self.emit_stub_call_1node(
                    id,
                    *value,
                    jit_runtime::jit_runtime_generic_decrement as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericBitwiseNot { value, .. } => {
                self.emit_stub_call_1node(
                    id,
                    *value,
                    jit_runtime::jit_runtime_generic_bitwise_not as usize,
                );
            }

            // ── Type checks and conversions via trampoline ────────────────────
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::CheckHeapObject { .. } => {
                // Guard pass-through — no code needed.
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::ToString { value, .. } => {
                self.emit_stub_call_1node(id, *value, jit_runtime::jit_runtime_tostring as usize);
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::ToNumber { value, .. } => {
                self.emit_stub_call_1node(id, *value, jit_runtime::jit_runtime_tonumber as usize);
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::TypeOf { value } => {
                self.emit_stub_call_1node(id, *value, jit_runtime::jit_runtime_typeof as usize);
            }

            // ── Deep context slot access via trampoline ───────────────────────
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::LoadContextSlot { depth, slot, .. } => {
                self.emit_trampoline_call(
                    id,
                    Opcode::LdaContextSlot as u8,
                    i64::from(*slot),
                    i64::from(*depth),
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::StoreContextSlot {
                depth, slot, value, ..
            } => {
                // Store the value into the accumulator (R12) before the
                // trampoline call, since it reads the value from `acc`.
                self.emit_load(*value, Reg64::R12);
                self.emit_trampoline_call(
                    id,
                    Opcode::StaContextSlot as u8,
                    i64::from(*slot),
                    i64::from(*depth),
                );
            }

            // ── Tagged equality ───────────────────────────────────────────────
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::TaggedEqual { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_tagged_equal as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::TaggedNotEqual { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_tagged_not_equal as usize,
                );
            }

            // ── Unsupported nodes → unconditional deopt ───────────────────────
            #[cfg_attr(not(all(target_arch = "x86_64", unix)), allow(unused_variables))]
            other => {
                #[cfg(all(target_arch = "x86_64", unix))]
                eprintln!("MAGLEV_UNSUPPORTED_NODE: {other:?}");
                self.emit_deopt_unconditional(0);
                // Satisfy the allocation invariant: write a placeholder.
                self.masm.mov_ri(Reg64::R11, JIT_UNDEFINED);
                self.emit_store(id, Reg64::R11);
            }
        }
    }

    // ── Control node emission ────────────────────────────────────────────────

    fn emit_control_node(&mut self, block_idx: u32, ctrl: &ControlNode) {
        match ctrl {
            ControlNode::Return { value } => {
                self.emit_load(*value, Reg64::Rax);
                self.emit_promoted_global_stores();
                self.emit_normal_epilogue();
            }
            ControlNode::Jump { target } => {
                let target = *target as usize;
                self.emit_phi_copies_for_successor(block_idx, target as u32);
                self.masm.jmp(&mut self.block_labels[target]);
            }
            ControlNode::Branch {
                condition,
                if_true,
                if_false,
            } => {
                let if_true_idx = *if_true as usize;
                let if_false_idx = *if_false as usize;

                // Try compare-branch fusion: if the condition is an Int32
                // comparison, re-emit CMP and branch directly instead of
                // materializing a JIT_TRUE/JIT_FALSE boolean.
                let fused =
                    self.try_fuse_compare_branch(*condition, block_idx, *if_true, *if_false);
                if !fused {
                    // Fallback: load materialised boolean, compare with JIT_TRUE.
                    self.emit_load(*condition, Reg64::R11);
                    self.masm.mov_ri(Reg64::R10, JIT_TRUE);
                    self.masm.cmp_rr(Reg64::R11, Reg64::R10);

                    let mut false_path = Label::new();
                    self.masm.jcc(CondCode::NotEqual, &mut false_path);
                    self.emit_phi_copies_for_successor(block_idx, *if_true);
                    self.masm.jmp(&mut self.block_labels[if_true_idx]);
                    self.masm.bind_label(&mut false_path);
                    self.emit_phi_copies_for_successor(block_idx, *if_false);
                    self.masm.jmp(&mut self.block_labels[if_false_idx]);
                }
            }
            ControlNode::Deoptimize {
                bytecode_offset,
                reason: _,
            } => {
                self.emit_deopt_unconditional(*bytecode_offset);
            }
        }
    }

    // ── Phi-resolution copy sequences ────────────────────────────────────────

    /// For each [`ValueNode::Phi`] in the successor block, copy the
    /// appropriate input (the one corresponding to `pred_idx`) into the Phi's
    /// allocated location.  This implements out-of-SSA destruction at jump
    /// sites.
    fn emit_phi_copies_for_successor(&mut self, pred_idx: u32, successor_idx: u32) {
        let successor = match self.graph.block(successor_idx) {
            Some(b) => b,
            None => return,
        };

        let pred_pos = match successor.predecessors.iter().position(|&p| p == pred_idx) {
            Some(pos) => pos,
            None => return,
        };

        let phi_ops: Vec<(NodeId, NodeId)> = successor
            .nodes
            .iter()
            .filter_map(|(phi_id, node)| {
                if let ValueNode::Phi { inputs } = node {
                    inputs.get(pred_pos).map(|&src| (*phi_id, src))
                } else {
                    None
                }
            })
            // Skip self-referential Phis (src == phi) and same-location copies.
            .filter(|(phi_id, src_id)| {
                if phi_id == src_id {
                    return false;
                }
                // Also skip when source and destination occupy the same location.
                self.alloc.location(*phi_id) != self.alloc.location(*src_id)
            })
            .collect();

        // Parallel-move: load ALL sources onto the stack first, then
        // pop into destinations in reverse order.  This avoids the
        // classic overwrite-before-read bug when a Phi destination
        // coincides with another Phi's source location.
        if phi_ops.len() <= 1 {
            // Single or no Phi — no conflict possible.
            for (phi_id, src_id) in &phi_ops {
                self.emit_load(*src_id, Reg64::R11);
                self.emit_store(*phi_id, Reg64::R11);
            }
        } else {
            // Push all source values.
            for (_phi_id, src_id) in &phi_ops {
                self.emit_load(*src_id, Reg64::R11);
                self.masm.push(Reg64::R11);
            }
            // Pop into destinations in reverse order.
            for (phi_id, _src_id) in phi_ops.iter().rev() {
                self.masm.pop(Reg64::R11);
                self.emit_store(*phi_id, Reg64::R11);
            }
        }
    }

    // ── Load / store helpers ─────────────────────────────────────────────────

    /// Load the value produced by node `id` from its allocated location into
    /// `dst`.
    fn emit_load(&mut self, id: NodeId, dst: Reg64) {
        match self.alloc.location(id) {
            Some(Location::Register(n)) => {
                let src = phys_reg(n);
                if src != dst {
                    self.masm.mov_rr(dst, src);
                }
            }
            Some(Location::StackSlot(n)) => {
                let off = self.slot_offset(n);
                self.masm.mov_load_base_disp32(dst, Reg64::R14, off);
            }
            None => {
                self.masm.mov_ri(dst, JIT_UNDEFINED);
            }
        }
    }

    /// Store the value in `src` to the allocated location of node `id`.
    fn emit_store(&mut self, id: NodeId, src: Reg64) {
        match self.alloc.location(id) {
            Some(Location::Register(n)) => {
                let dst = phys_reg(n);
                if dst != src {
                    self.masm.mov_rr(dst, src);
                }
            }
            Some(Location::StackSlot(n)) => {
                let off = self.slot_offset(n);
                self.masm.mov_store_base_disp32(Reg64::R14, off, src);
            }
            None => {
                // Node has no allocated location — the register allocator
                // determined its value is unused (dead).  This is expected
                // for side-effect-only stub calls.
            }
        }
    }

    /// Byte offset from R14 (register-file base) for spill slot `n`.
    fn slot_offset(&self, n: u32) -> i32 {
        ((self.param_count + n) * 8) as i32
    }

    // ── Promoted globals ─────────────────────────────────────────────────────

    /// Scan the IR graph for every unique `LoadGlobal` / `StoreGlobal` name
    /// and assign each one a dedicated register-file slot beyond the
    /// allocator's spill area.
    fn scan_and_promote_globals(&mut self, base_slots: usize) {
        let mut seen = HashSet::new();
        for block in self.graph.blocks() {
            for (_, node) in &block.nodes {
                match node {
                    ValueNode::LoadGlobal { name, .. } | ValueNode::StoreGlobal { name, .. } => {
                        seen.insert(*name);
                    }
                    _ => {}
                }
            }
        }
        let mut sorted: Vec<u32> = seen.into_iter().collect();
        sorted.sort_unstable();
        self.promoted_globals = sorted
            .iter()
            .enumerate()
            .map(|(i, &name_idx)| (name_idx, base_slots + i))
            .collect();
        self.promoted_extra_slots = sorted.len();
    }

    /// Byte offset from R14 for the promoted global with `name_idx`.
    #[cfg_attr(not(all(target_arch = "x86_64", unix)), allow(dead_code))]
    fn promoted_global_offset(&self, name_idx: u32) -> Option<i32> {
        self.promoted_globals
            .iter()
            .find(|(n, _)| *n == name_idx)
            .map(|&(_, slot)| (slot * 8) as i32)
    }

    /// Emit code to load all promoted globals from `GlobalEnv` into their
    /// register-file slots via [`jit_runtime_lda_global`].
    ///
    /// Called once after the prologue.  At this point no allocatable registers
    /// hold live values, so we can freely clobber caller-saved registers.
    #[allow(unused_variables)]
    fn emit_promoted_global_loads(&mut self) {
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            if self.promoted_globals.is_empty() {
                return;
            }
            let addr = jit_runtime::jit_runtime_lda_global as *const () as usize as i64;
            // After the prologue (4 pushes + return addr = 5 items), RSP ≡ 8
            // mod 16.  We need RSP ≡ 0 mod 16 before `call` so the callee
            // sees RSP ≡ 8 mod 16 per SysV ABI.  A single dummy push fixes it.
            self.masm.push(Reg64::R11);
            for &(name_idx, slot) in &self.promoted_globals.clone() {
                let off = (slot * 8) as i32;
                // RDI = name_idx (SysV ABI first arg)
                self.masm.mov_ri(Reg64::Rdi, i64::from(name_idx));
                self.masm.mov_ri(Reg64::R11, addr);
                self.masm.call_reg(Reg64::R11);
                // If the stub returned JIT_DEOPT, bail out immediately.
                self.masm.mov_ri(Reg64::R11, JIT_DEOPT);
                self.masm.cmp_rr(Reg64::Rax, Reg64::R11);
                self.masm.jcc(CondCode::Equal, &mut self.deopt_global_label);
                // Store result in promoted slot: [R14 + off] = RAX
                self.masm.mov_store_base_disp32(Reg64::R14, off, Reg64::Rax);
            }
            self.masm.pop(Reg64::R11);
        }
    }

    /// Emit code to flush all promoted globals from their register-file
    /// slots back to `GlobalEnv` via [`jit_runtime_sta_global`].
    ///
    /// Called before each `Return`.  We save/restore RAX around the flush
    /// because it already holds the return value.
    #[allow(unused_variables)]
    fn emit_promoted_global_stores(&mut self) {
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            if self.promoted_globals.is_empty() {
                return;
            }
            let addr = jit_runtime::jit_runtime_sta_global as *const () as usize as i64;
            // Save the return value (already in RAX).
            self.masm.push(Reg64::Rax);
            for &(name_idx, slot) in &self.promoted_globals.clone() {
                let off = (slot * 8) as i32;
                // RDI = name_idx
                self.masm.mov_ri(Reg64::Rdi, i64::from(name_idx));
                // RSI = value from promoted slot
                self.masm.mov_load_base_disp32(Reg64::Rsi, Reg64::R14, off);
                self.masm.mov_ri(Reg64::R11, addr);
                self.masm.call_reg(Reg64::R11);
            }
            // Restore the return value.
            self.masm.pop(Reg64::Rax);
        }
    }

    // ── Runtime stub call helpers ────────────────────────────────────────────
    //
    // These helpers save all caller-saved allocatable registers, set up
    // arguments in SysV ABI order, call the extern "C" stub, check for
    // JIT_DEOPT, and store the result.

    /// Save caller-saved allocatable registers: RCX, RDX, RSI, R8, R9.
    #[cfg(all(target_arch = "x86_64", unix))]
    fn emit_save_caller_saved(&mut self) {
        self.masm.push(Reg64::Rcx);
        self.masm.push(Reg64::Rdx);
        self.masm.push(Reg64::Rsi);
        self.masm.push(Reg64::R8);
        self.masm.push(Reg64::R9);
    }

    /// Restore caller-saved allocatable registers (reverse order).
    #[cfg(all(target_arch = "x86_64", unix))]
    fn emit_restore_caller_saved(&mut self) {
        self.masm.pop(Reg64::R9);
        self.masm.pop(Reg64::R8);
        self.masm.pop(Reg64::Rsi);
        self.masm.pop(Reg64::Rdx);
        self.masm.pop(Reg64::Rcx);
    }

    /// Emit a deopt check: if RAX == JIT_DEOPT, jump to stub-deopt epilogue.
    #[cfg(all(target_arch = "x86_64", unix))]
    fn emit_deopt_check_rax(&mut self) {
        self.masm.mov_ri(Reg64::R11, JIT_DEOPT);
        self.masm.cmp_rr(Reg64::Rax, Reg64::R11);
        self.masm.jcc(CondCode::Equal, &mut self.deopt_stub_label);
    }

    /// Call a 1-arg stub: `stub(node_arg)`.
    #[cfg(all(target_arch = "x86_64", unix))]
    fn emit_stub_call_1node(&mut self, id: NodeId, arg0: NodeId, stub_addr: usize) {
        self.emit_save_caller_saved();
        self.emit_load(arg0, Reg64::Rdi);
        self.masm.mov_ri(Reg64::R11, stub_addr as i64);
        self.masm.call_reg(Reg64::R11);
        self.emit_restore_caller_saved();
        self.emit_deopt_check_rax();
        self.masm.mov_rr(Reg64::R11, Reg64::Rax);
        self.emit_store(id, Reg64::R11);
    }

    /// Call a 2-node-arg stub: `stub(node0, node1)`.
    #[cfg(all(target_arch = "x86_64", unix))]
    fn emit_stub_call_2node(&mut self, id: NodeId, arg0: NodeId, arg1: NodeId, stub_addr: usize) {
        self.emit_save_caller_saved();
        self.emit_load(arg0, Reg64::Rdi);
        self.emit_load(arg1, Reg64::Rsi);
        self.masm.mov_ri(Reg64::R11, stub_addr as i64);
        self.masm.call_reg(Reg64::R11);
        self.emit_restore_caller_saved();
        self.emit_deopt_check_rax();
        self.masm.mov_rr(Reg64::R11, Reg64::Rax);
        self.emit_store(id, Reg64::R11);
    }

    /// Call a 3-node-arg stub: `stub(node0, node1, node2)`.
    #[cfg(all(target_arch = "x86_64", unix))]
    fn emit_stub_call_3node(
        &mut self,
        id: NodeId,
        arg0: NodeId,
        arg1: NodeId,
        arg2: NodeId,
        stub_addr: usize,
    ) {
        self.emit_save_caller_saved();
        self.emit_load(arg0, Reg64::Rdi);
        self.emit_load(arg1, Reg64::Rsi);
        self.emit_load(arg2, Reg64::Rdx);
        self.masm.mov_ri(Reg64::R11, stub_addr as i64);
        self.masm.call_reg(Reg64::R11);
        self.emit_restore_caller_saved();
        self.emit_deopt_check_rax();
        self.masm.mov_rr(Reg64::R11, Reg64::Rax);
        self.emit_store(id, Reg64::R11);
    }

    /// Call a 3-arg stub where arg0 is a node, arg1 is an immediate i64,
    /// and arg2 is either a node or an immediate (see [`NodeOrImm`]).
    #[cfg(all(target_arch = "x86_64", unix))]
    fn emit_stub_call_3arg(
        &mut self,
        id: NodeId,
        arg0_node: NodeId,
        arg1_imm: i64,
        arg2: NodeOrImm,
        stub_addr: usize,
    ) {
        self.emit_save_caller_saved();
        // Load all node arguments BEFORE setting any immediates into
        // allocatable registers (RSI = phys_reg(3)).  The old order
        // clobbered RSI with arg1_imm before loading arg2, which could
        // read from RSI if the allocator placed it there.
        self.emit_load(arg0_node, Reg64::Rdi);
        match arg2 {
            NodeOrImm::Node(n) => self.emit_load(n, Reg64::Rdx),
            NodeOrImm::Imm(v) => self.masm.mov_ri(Reg64::Rdx, v),
        }
        self.masm.mov_ri(Reg64::Rsi, arg1_imm);
        self.masm.mov_ri(Reg64::R11, stub_addr as i64);
        self.masm.call_reg(Reg64::R11);
        self.emit_restore_caller_saved();
        self.emit_deopt_check_rax();
        self.masm.mov_rr(Reg64::R11, Reg64::Rax);
        self.emit_store(id, Reg64::R11);
    }

    /// Call the generic trampoline for opcodes that only need immediate operands.
    ///
    /// The trampoline signature is:
    /// `fn(opcode: u32, regs: *mut i64, acc: i64, operand1: i64, operand2: i64) -> i64`
    ///
    /// We pass R14 (register-file base) as `regs` and R12 (accumulator) as
    /// `acc`, though most creation ops only use `operand1`/`operand2` and TLS.
    #[cfg(all(target_arch = "x86_64", unix))]
    fn emit_trampoline_call(&mut self, id: NodeId, opcode: u8, operand1: i64, operand2: i64) {
        self.emit_save_caller_saved();
        // RDI = opcode
        self.masm.mov_ri(Reg64::Rdi, i64::from(opcode));
        // RSI = register-file base (R14)
        self.masm.mov_rr(Reg64::Rsi, Reg64::R14);
        // RDX = accumulator (R12)
        self.masm.mov_rr(Reg64::Rdx, Reg64::R12);
        // RCX = operand1
        self.masm.mov_ri(Reg64::Rcx, operand1);
        // R8 = operand2
        self.masm.mov_ri(Reg64::R8, operand2);
        let addr = jit_runtime::jit_runtime_trampoline as *const () as usize;
        self.masm.mov_ri(Reg64::R11, addr as i64);
        self.masm.call_reg(Reg64::R11);
        self.emit_restore_caller_saved();
        self.emit_deopt_check_rax();
        self.masm.mov_rr(Reg64::R11, Reg64::Rax);
        self.emit_store(id, Reg64::R11);
    }

    // ── Arithmetic helpers ───────────────────────────────────────────────────

    /// `AND dst, src` — `REX.W 23 /r` (AND r64, r/m64).
    fn emit_and_rr(&mut self, dst: Reg64, src: Reg64) {
        // REX: W=1, R set if dst needs REX, B set if src needs REX.
        let rex: u8 = 0x48
            | (if dst.needs_rex() { 0x04 } else { 0 })
            | (if src.needs_rex() { 0x01 } else { 0 });
        let modrm: u8 = 0xC0 | (dst.enc() << 3) | src.enc();
        self.masm.emit_byte(rex);
        self.masm.emit_byte(0x23);
        self.masm.emit_byte(modrm);
    }

    /// `OR dst, src` — `REX.W 0B /r` (OR r64, r/m64).
    fn emit_or_rr(&mut self, dst: Reg64, src: Reg64) {
        let rex: u8 = 0x48
            | (if dst.needs_rex() { 0x04 } else { 0 })
            | (if src.needs_rex() { 0x01 } else { 0 });
        let modrm: u8 = 0xC0 | (dst.enc() << 3) | src.enc();
        self.masm.emit_byte(rex);
        self.masm.emit_byte(0x0B);
        self.masm.emit_byte(modrm);
    }

    // ── Division helpers ─────────────────────────────────────────────────────

    /// Emit `CQO` — sign-extend `RAX` into `RDX:RAX`.
    fn emit_cqo(&mut self) {
        self.masm.emit_byte(0x48); // REX.W
        self.masm.emit_byte(0x99); // CQO
    }

    /// Emit `IDIV R10` — signed divide `RDX:RAX` by `R10`.
    ///
    /// After execution: quotient in `RAX`, remainder in `RDX`.
    fn emit_idiv_r10(&mut self) {
        // REX.WB (R10 needs REX.B): 0x49
        self.masm.emit_byte(0x49);
        // IDIV r/m64
        self.masm.emit_byte(0xF7);
        // ModRM: mod=11, /7, r/m=R10(enc=2) = 0xC0|0x38|0x02 = 0xFA
        self.masm.emit_byte(0xFA);
    }

    /// Emit a division-by-zero guard: deopt if `R10` is zero.
    fn emit_div_zero_check(&mut self, bytecode_offset: u32) {
        self.masm.test_rr(Reg64::R10, Reg64::R10);
        let code_off = self.masm.position() as u32;
        let liveness_map = self.all_slots_live();
        self.deopt_entries.push(DeoptEntry {
            code_offset: code_off,
            bytecode_offset,
            liveness_map,
        });
        self.masm
            .jcc(CondCode::Equal, &mut self.deopt_divzero_label);
    }

    /// Emit the core 64-bit signed-divide sequence.
    ///
    /// Expects dividend in `R11`, divisor in `R10`.  Clobbers `RAX` and
    /// `RDX`; the caller must save/restore `RDX` if needed.
    ///
    /// After return: quotient in `RAX`, remainder in `RDX`.
    fn emit_idiv_sequence(&mut self) {
        self.masm.mov_rr(Reg64::Rax, Reg64::R11);
        self.emit_cqo();
        self.emit_idiv_r10();
    }

    // ── Comparison helper ────────────────────────────────────────────────────

    /// Emit a 64-bit integer comparison, writing `JIT_FALSE` or `JIT_TRUE`
    /// to `result`'s allocated location.
    fn emit_int32_compare(&mut self, left: NodeId, right: NodeId, cc: CondCode, result: NodeId) {
        self.emit_load(left, Reg64::R11);
        self.emit_load(right, Reg64::R10);
        self.masm.cmp_rr(Reg64::R11, Reg64::R10);
        self.masm.setcc_al(cc);
        self.masm.movzx_r64_al(Reg64::R11);
        self.masm.mov_ri(Reg64::R10, JIT_FALSE);
        self.masm.add_rr(Reg64::R11, Reg64::R10);
        self.emit_store(result, Reg64::R11);
    }

    /// Try to fuse a Branch's condition with a comparison node.
    ///
    /// If `condition` is an `Int32LessThan` (or similar), re-emit `CMP` from
    /// the original operands and branch directly, saving ~10 instructions per
    /// compare-and-branch versus materialising a JIT_TRUE/JIT_FALSE boolean.
    ///
    /// Returns `true` if the fusion was emitted.
    fn try_fuse_compare_branch(
        &mut self,
        condition: NodeId,
        block_idx: u32,
        if_true: u32,
        if_false: u32,
    ) -> bool {
        let cond_node = match self.graph.node(condition) {
            Some(n) => n.clone(),
            None => return false,
        };
        let (left, right, cc) = match &cond_node {
            ValueNode::Int32LessThan { left, right } => (*left, *right, CondCode::Less),
            ValueNode::Int32LessThanOrEqual { left, right } => (*left, *right, CondCode::LessEq),
            ValueNode::Int32GreaterThan { left, right } => (*left, *right, CondCode::Greater),
            ValueNode::Int32GreaterThanOrEqual { left, right } => {
                (*left, *right, CondCode::GreaterEq)
            }
            ValueNode::Int32Equal { left, right } | ValueNode::Int32StrictEqual { left, right } => {
                (*left, *right, CondCode::Equal)
            }
            _ => return false,
        };

        // Re-emit CMP from the comparison's operands.
        self.emit_load(left, Reg64::R11);
        self.emit_load(right, Reg64::R10);
        self.masm.cmp_rr(Reg64::R11, Reg64::R10);

        // Branch directly: condition-false falls through to false phi-copies.
        let if_true_idx = if_true as usize;
        let if_false_idx = if_false as usize;
        let negated = Self::negate_cc(cc);
        let mut false_path = Label::new();
        self.masm.jcc(negated, &mut false_path);
        self.emit_phi_copies_for_successor(block_idx, if_true);
        self.masm.jmp(&mut self.block_labels[if_true_idx]);
        self.masm.bind_label(&mut false_path);
        self.emit_phi_copies_for_successor(block_idx, if_false);
        self.masm.jmp(&mut self.block_labels[if_false_idx]);
        true
    }

    /// Negate a condition code (for compare-branch fusion: jump on the
    /// *opposite* condition to skip the true path).
    const fn negate_cc(cc: CondCode) -> CondCode {
        match cc {
            CondCode::Equal => CondCode::NotEqual,
            CondCode::NotEqual => CondCode::Equal,
            CondCode::Less => CondCode::GreaterEq,
            CondCode::GreaterEq => CondCode::Less,
            CondCode::LessEq => CondCode::Greater,
            CondCode::Greater => CondCode::LessEq,
        }
    }

    // ── Deopt helpers ────────────────────────────────────────────────────────

    /// Emit a Smi-overflow guard after an arithmetic operation whose result is
    /// in `R11`.
    ///
    /// The guard verifies that the 64-bit result fits in a 32-bit signed
    /// integer (i.e. it is a valid Smi).  If not, the function deoptimises.
    ///
    /// Algorithm (all operations on `R11` / `R10`):
    /// 1. Copy `R11` → `R10`.
    /// 2. `SAR R10, 31` — arithmetic-right-shift by 31 fills `R10` with the
    ///    sign extension of bit 31 of the original `R10`.  After this step
    ///    `R10` is `0` when bit 31 of the copy was 0, and `-1` (all ones) when
    ///    bit 31 was 1.
    /// 3. `XOR R10, R11` — if `R11` fits in i32, the upper 33 bits of `R10`
    ///    after step 2 match the upper 33 bits of `R11`, so XOR produces 0 in
    ///    the upper half.  If `R11` overflowed i32 the upper bits differ, so
    ///    XOR is non-zero.
    /// 4. `SAR R10, 32` — shift the upper-half indicator down into the low 32
    ///    bits so that `TEST R10, R10` can detect non-zero.
    /// 5. `TEST R10, R10` — sets ZF=0 when overflow is detected.
    /// 6. `JNE deopt_label` — jump to the deopt epilogue on overflow.
    fn emit_deopt_on_smi_overflow(&mut self, bytecode_offset: u32) {
        // Check if R11 fits in i32 using MOVSXD + CMP (3 instructions vs 6).
        // MOVSXD R10, R11d — sign-extends the lower 32 bits of R11 into R10.
        // If R11 was a valid i32, MOVSXD(R11[31:0]) == R11.
        // REX.W (0x4C for R10 dst with REX.R=1, R11 src with REX.B=1)
        self.masm.emit_byte(0x4D); // REX.W + REX.R (R10) + REX.B (R11)
        self.masm.emit_byte(0x63); // MOVSXD
        self.masm.emit_byte(0xD3); // ModRM: mod=11, reg=R10(enc=2), r/m=R11(enc=3)

        // CMP R10, R11 — if sign-extended version differs, overflow occurred.
        self.masm.cmp_rr(Reg64::R10, Reg64::R11);

        let code_off = self.masm.position() as u32;
        let liveness_map = self.all_slots_live();
        self.deopt_entries.push(DeoptEntry {
            code_offset: code_off,
            bytecode_offset,
            liveness_map,
        });
        // JNE deopt_overflow_label — jump to overflow deopt if R10 != R11.
        self.masm
            .jcc(CondCode::NotEqual, &mut self.deopt_overflow_label);
    }

    /// Emit an unconditional deopt (record entry + jump to deopt epilogue).
    fn emit_deopt_unconditional(&mut self, bytecode_offset: u32) {
        let code_off = self.masm.position() as u32;
        let liveness_map = self.all_slots_live();
        self.deopt_entries.push(DeoptEntry {
            code_offset: code_off,
            bytecode_offset,
            liveness_map,
        });
        self.masm.jmp(&mut self.deopt_label);
    }

    /// Compute a conservative liveness bitmask covering all register-file
    /// slots (`param_count + spill_count`).
    fn all_slots_live(&self) -> u64 {
        let slots = (self.param_count + self.alloc.spill_count()) as usize;
        if slots >= 64 {
            u64::MAX
        } else {
            (1u64 << slots) - 1
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::maglev::ir::{BasicBlock, ControlNode, MaglevGraph, ValueNode};

    // ── Helper ───────────────────────────────────────────────────────────────

    fn do_compile(graph: &MaglevGraph, param_count: u32) -> MaglevCompiledCode {
        compile(graph, param_count).expect("codegen failed")
    }

    // ── Smoke tests (non-execution) ───────────────────────────────────────────

    #[test]
    fn test_compile_return_smi_constant() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c = block.push_value(ValueNode::SmiConstant { value: 42 });
        block.set_control(ControlNode::Return { value: c });
        graph.add_block(block);

        let cc = do_compile(&graph, 0);
        assert!(!cc.code.is_empty());
        assert!(cc.native_code_len > 0);
        assert!(cc.native_code_len <= cc.code.len());
        assert_eq!(cc.safepoints.len(), 1);
    }

    #[test]
    fn test_compile_constants_produce_safepoints() {
        for variant in [
            ValueNode::TrueConstant,
            ValueNode::FalseConstant,
            ValueNode::NullConstant,
            ValueNode::UndefinedConstant,
        ] {
            let mut block = BasicBlock::new(0);
            let c = block.push_value(variant);
            block.set_control(ControlNode::Return { value: c });
            let mut g = MaglevGraph::new(0);
            g.add_block(block);
            let cc = do_compile(&g, 0);
            assert!(
                !cc.code.is_empty(),
                "expected non-empty code for constant variant"
            );
        }
    }

    #[test]
    fn test_compile_return_parameter() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        block.set_control(ControlNode::Return { value: p0 });
        graph.add_block(block);

        let cc = do_compile(&graph, 1);
        assert!(cc.native_code_len > 0);
        assert_eq!(cc.register_file_slots, 1);
    }

    #[test]
    fn test_compile_int32_add_produces_code() {
        let mut graph = MaglevGraph::new(2);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        let p1 = block.push_value(ValueNode::Parameter { index: 1 });
        let sum = block.push_value(ValueNode::Int32Add {
            left: p0,
            right: p1,
        });
        block.set_control(ControlNode::Return { value: sum });
        graph.add_block(block);

        let cc = do_compile(&graph, 2);
        assert!(cc.native_code_len > 0);
    }

    #[test]
    fn test_compile_two_block_jump_safepoints() {
        let mut graph = MaglevGraph::new(1);

        let mut b0 = BasicBlock::new(0);
        let p0 = b0.push_value(ValueNode::Parameter { index: 0 });
        b0.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b0);

        let mut b1 = BasicBlock::new(1);
        b1.set_control(ControlNode::Return { value: p0 });
        graph.add_block(b1);

        let cc = do_compile(&graph, 1);
        assert_eq!(cc.safepoints.len(), 2, "expected one safepoint per block");
    }

    #[test]
    fn test_compile_branch_graph_safepoints() {
        let mut graph = MaglevGraph::new(1);

        let mut b0 = BasicBlock::new(0);
        let p0 = b0.push_value(ValueNode::Parameter { index: 0 });
        let cond = b0.push_value(ValueNode::ToBoolean { value: p0 });
        b0.set_control(ControlNode::Branch {
            condition: cond,
            if_true: 1,
            if_false: 2,
        });
        graph.add_block(b0);

        let mut b1 = BasicBlock::new(1);
        b1.add_predecessor(0);
        let one = b1.push_value(ValueNode::SmiConstant { value: 1 });
        b1.set_control(ControlNode::Return { value: one });
        graph.add_block(b1);

        let mut b2 = BasicBlock::new(2);
        b2.add_predecessor(0);
        let zero = b2.push_value(ValueNode::SmiConstant { value: 0 });
        b2.set_control(ControlNode::Return { value: zero });
        graph.add_block(b2);

        let cc = do_compile(&graph, 1);
        assert_eq!(cc.safepoints.len(), 3, "expected one safepoint per block");
    }

    #[test]
    fn test_compile_deoptimize_records_entry() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        block.set_control(ControlNode::Deoptimize {
            bytecode_offset: 12,
            reason: 1,
        });
        graph.add_block(block);

        let cc = do_compile(&graph, 0);
        assert!(
            !cc.deopt_entries.is_empty(),
            "expected at least one deopt entry"
        );
        assert_eq!(
            cc.deopt_entries[0].bytecode_offset, 12,
            "deopt bytecode offset mismatch"
        );
    }

    #[test]
    fn test_compile_checked_smi_add_records_deopt_entry() {
        let mut graph = MaglevGraph::new(2);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        let p1 = block.push_value(ValueNode::Parameter { index: 1 });
        let sum = block.push_value(ValueNode::CheckedSmiAdd {
            left: p0,
            right: p1,
        });
        block.set_control(ControlNode::Return { value: sum });
        graph.add_block(block);

        let cc = do_compile(&graph, 2);
        assert!(
            !cc.deopt_entries.is_empty(),
            "expected a deopt entry for CheckedSmiAdd overflow guard"
        );
    }

    #[test]
    fn test_source_positions_empty() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c = block.push_value(ValueNode::UndefinedConstant);
        block.set_control(ControlNode::Return { value: c });
        graph.add_block(block);

        let cc = do_compile(&graph, 0);
        assert!(
            cc.source_positions.is_empty(),
            "source position table should be empty (reserved for future use)"
        );
    }

    #[test]
    fn test_register_file_slots_equals_param_plus_spills() {
        let mut graph = MaglevGraph::new(2);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        let p1 = block.push_value(ValueNode::Parameter { index: 1 });
        let sum = block.push_value(ValueNode::Int32Add {
            left: p0,
            right: p1,
        });
        block.set_control(ControlNode::Return { value: sum });
        graph.add_block(block);

        let cc = do_compile(&graph, 2);
        // register_file_slots = param_count + spill_count >= 2
        assert!(
            cc.register_file_slots >= 2,
            "register file must include at least the parameters"
        );
    }

    #[test]
    fn test_compile_check_smi_records_deopt_entry() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        let checked = block.push_value(ValueNode::CheckSmi { receiver: p0 });
        block.set_control(ControlNode::Return { value: checked });
        graph.add_block(block);

        let cc = do_compile(&graph, 1);
        assert!(
            !cc.deopt_entries.is_empty(),
            "expected a deopt entry for CheckSmi overflow guard"
        );
    }

    #[test]
    fn test_compile_int32_divide_produces_code() {
        let mut graph = MaglevGraph::new(2);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        let p1 = block.push_value(ValueNode::Parameter { index: 1 });
        let div = block.push_value(ValueNode::Int32Divide {
            left: p0,
            right: p1,
        });
        block.set_control(ControlNode::Return { value: div });
        graph.add_block(block);

        let cc = do_compile(&graph, 2);
        assert!(cc.native_code_len > 0);
        assert!(
            !cc.deopt_entries.is_empty(),
            "expected a deopt entry for div-by-zero guard"
        );
    }

    #[test]
    fn test_compile_int32_modulus_produces_code() {
        let mut graph = MaglevGraph::new(2);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        let p1 = block.push_value(ValueNode::Parameter { index: 1 });
        let rem = block.push_value(ValueNode::Int32Modulus {
            left: p0,
            right: p1,
        });
        block.set_control(ControlNode::Return { value: rem });
        graph.add_block(block);

        let cc = do_compile(&graph, 2);
        assert!(cc.native_code_len > 0);
    }

    #[test]
    fn test_compile_uint32_add_produces_code() {
        let mut graph = MaglevGraph::new(2);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        let p1 = block.push_value(ValueNode::Parameter { index: 1 });
        let sum = block.push_value(ValueNode::Uint32Add {
            left: p0,
            right: p1,
        });
        block.set_control(ControlNode::Return { value: sum });
        graph.add_block(block);

        let cc = do_compile(&graph, 2);
        assert!(cc.native_code_len > 0);
    }

    // ── Execution tests (x86-64 / unix only) ─────────────────────────────────

    #[cfg(all(target_arch = "x86_64", unix))]
    mod exec {
        use super::*;

        fn run(graph: &MaglevGraph, param_count: u32, args: &[i64]) -> i64 {
            let cc = do_compile(graph, param_count);
            // SAFETY: code was produced by MaglevCodegen from a well-formed graph.
            unsafe { cc.execute(args).expect("execution failed") }
        }

        #[test]
        fn test_execute_return_constant_42() {
            let mut graph = MaglevGraph::new(0);
            let mut block = BasicBlock::new(0);
            let c = block.push_value(ValueNode::SmiConstant { value: 42 });
            block.set_control(ControlNode::Return { value: c });
            graph.add_block(block);

            assert_eq!(run(&graph, 0, &[]), 42);
        }

        #[test]
        fn test_execute_return_negative_constant() {
            let mut graph = MaglevGraph::new(0);
            let mut block = BasicBlock::new(0);
            let c = block.push_value(ValueNode::SmiConstant { value: -7 });
            block.set_control(ControlNode::Return { value: c });
            graph.add_block(block);

            assert_eq!(run(&graph, 0, &[]), -7);
        }

        #[test]
        fn test_execute_return_zero() {
            let mut graph = MaglevGraph::new(0);
            let mut block = BasicBlock::new(0);
            let c = block.push_value(ValueNode::SmiConstant { value: 0 });
            block.set_control(ControlNode::Return { value: c });
            graph.add_block(block);

            assert_eq!(run(&graph, 0, &[]), 0);
        }

        #[test]
        fn test_execute_return_parameter() {
            let mut graph = MaglevGraph::new(1);
            let mut block = BasicBlock::new(0);
            let p0 = block.push_value(ValueNode::Parameter { index: 0 });
            block.set_control(ControlNode::Return { value: p0 });
            graph.add_block(block);

            assert_eq!(run(&graph, 1, &[7]), 7);
            assert_eq!(run(&graph, 1, &[-3]), -3);
            assert_eq!(run(&graph, 1, &[0]), 0);
        }

        #[test]
        fn test_execute_int32_add() {
            let mut graph = MaglevGraph::new(2);
            let mut block = BasicBlock::new(0);
            let p0 = block.push_value(ValueNode::Parameter { index: 0 });
            let p1 = block.push_value(ValueNode::Parameter { index: 1 });
            let sum = block.push_value(ValueNode::Int32Add {
                left: p0,
                right: p1,
            });
            block.set_control(ControlNode::Return { value: sum });
            graph.add_block(block);

            assert_eq!(run(&graph, 2, &[3, 4]), 7);
            assert_eq!(run(&graph, 2, &[100, -50]), 50);
            assert_eq!(run(&graph, 2, &[0, 0]), 0);
        }

        #[test]
        fn test_execute_int32_subtract() {
            let mut graph = MaglevGraph::new(2);
            let mut block = BasicBlock::new(0);
            let p0 = block.push_value(ValueNode::Parameter { index: 0 });
            let p1 = block.push_value(ValueNode::Parameter { index: 1 });
            let diff = block.push_value(ValueNode::Int32Subtract {
                left: p0,
                right: p1,
            });
            block.set_control(ControlNode::Return { value: diff });
            graph.add_block(block);

            assert_eq!(run(&graph, 2, &[10, 4]), 6);
            assert_eq!(run(&graph, 2, &[0, 5]), -5);
        }

        #[test]
        fn test_execute_int32_multiply() {
            let mut graph = MaglevGraph::new(2);
            let mut block = BasicBlock::new(0);
            let p0 = block.push_value(ValueNode::Parameter { index: 0 });
            let p1 = block.push_value(ValueNode::Parameter { index: 1 });
            let prod = block.push_value(ValueNode::Int32Multiply {
                left: p0,
                right: p1,
            });
            block.set_control(ControlNode::Return { value: prod });
            graph.add_block(block);

            assert_eq!(run(&graph, 2, &[3, 7]), 21);
            assert_eq!(run(&graph, 2, &[-2, 5]), -10);
        }

        #[test]
        fn test_execute_int32_negate() {
            let mut graph = MaglevGraph::new(1);
            let mut block = BasicBlock::new(0);
            let p0 = block.push_value(ValueNode::Parameter { index: 0 });
            let neg = block.push_value(ValueNode::Int32Negate { value: p0 });
            block.set_control(ControlNode::Return { value: neg });
            graph.add_block(block);

            assert_eq!(run(&graph, 1, &[5]), -5);
            assert_eq!(run(&graph, 1, &[-3]), 3);
            assert_eq!(run(&graph, 1, &[0]), 0);
        }

        #[test]
        fn test_execute_int32_increment() {
            let mut graph = MaglevGraph::new(1);
            let mut block = BasicBlock::new(0);
            let p0 = block.push_value(ValueNode::Parameter { index: 0 });
            let inc = block.push_value(ValueNode::Int32Increment { value: p0 });
            block.set_control(ControlNode::Return { value: inc });
            graph.add_block(block);

            assert_eq!(run(&graph, 1, &[5]), 6);
            assert_eq!(run(&graph, 1, &[-1]), 0);
        }

        #[test]
        fn test_execute_int32_decrement() {
            let mut graph = MaglevGraph::new(1);
            let mut block = BasicBlock::new(0);
            let p0 = block.push_value(ValueNode::Parameter { index: 0 });
            let dec = block.push_value(ValueNode::Int32Decrement { value: p0 });
            block.set_control(ControlNode::Return { value: dec });
            graph.add_block(block);

            assert_eq!(run(&graph, 1, &[5]), 4);
            assert_eq!(run(&graph, 1, &[0]), -1);
        }

        #[test]
        fn test_execute_checked_smi_add_no_overflow() {
            let mut graph = MaglevGraph::new(2);
            let mut block = BasicBlock::new(0);
            let p0 = block.push_value(ValueNode::Parameter { index: 0 });
            let p1 = block.push_value(ValueNode::Parameter { index: 1 });
            let sum = block.push_value(ValueNode::CheckedSmiAdd {
                left: p0,
                right: p1,
            });
            block.set_control(ControlNode::Return { value: sum });
            graph.add_block(block);

            // These values don't overflow i32; the guard should not trigger.
            assert_eq!(run(&graph, 2, &[10, 20]), 30);
            assert_eq!(run(&graph, 2, &[-5, 5]), 0);
        }

        #[test]
        fn test_execute_to_boolean() {
            let mut graph = MaglevGraph::new(1);
            let mut block = BasicBlock::new(0);
            let p0 = block.push_value(ValueNode::Parameter { index: 0 });
            let b = block.push_value(ValueNode::ToBoolean { value: p0 });
            block.set_control(ControlNode::Return { value: b });
            graph.add_block(block);

            // non-zero → JIT_TRUE (0x1_0000_0001)
            let jit_true = run(&graph, 1, &[42]);
            assert_eq!(jit_true, JIT_TRUE);
            // zero → JIT_FALSE (0x1_0000_0000)
            let jit_false = run(&graph, 1, &[0]);
            assert_eq!(jit_false, JIT_FALSE);
        }

        #[test]
        fn test_execute_two_block_jump_identity() {
            let mut graph = MaglevGraph::new(1);

            let mut b0 = BasicBlock::new(0);
            let p0 = b0.push_value(ValueNode::Parameter { index: 0 });
            b0.set_control(ControlNode::Jump { target: 1 });
            graph.add_block(b0);

            let mut b1 = BasicBlock::new(1);
            b1.set_control(ControlNode::Return { value: p0 });
            graph.add_block(b1);

            assert_eq!(run(&graph, 1, &[99]), 99);
            assert_eq!(run(&graph, 1, &[-1]), -1);
        }

        #[test]
        fn test_execute_branch_truthy_arm() {
            // if (param0) { return 1 } else { return 0 }
            let mut graph = MaglevGraph::new(1);

            let mut b0 = BasicBlock::new(0);
            let p0 = b0.push_value(ValueNode::Parameter { index: 0 });
            let cond = b0.push_value(ValueNode::ToBoolean { value: p0 });
            b0.set_control(ControlNode::Branch {
                condition: cond,
                if_true: 1,
                if_false: 2,
            });
            graph.add_block(b0);

            let mut b1 = BasicBlock::new(1);
            b1.add_predecessor(0);
            let one = b1.push_value(ValueNode::SmiConstant { value: 1 });
            b1.set_control(ControlNode::Return { value: one });
            graph.add_block(b1);

            let mut b2 = BasicBlock::new(2);
            b2.add_predecessor(0);
            let zero = b2.push_value(ValueNode::SmiConstant { value: 0 });
            b2.set_control(ControlNode::Return { value: zero });
            graph.add_block(b2);

            assert_eq!(run(&graph, 1, &[42]), 1, "truthy → 1");
            assert_eq!(run(&graph, 1, &[0]), 0, "falsy → 0");
        }

        #[test]
        fn test_execute_int32_compare_equal() {
            let mut graph = MaglevGraph::new(2);
            let mut block = BasicBlock::new(0);
            let p0 = block.push_value(ValueNode::Parameter { index: 0 });
            let p1 = block.push_value(ValueNode::Parameter { index: 1 });
            let eq = block.push_value(ValueNode::Int32Equal {
                left: p0,
                right: p1,
            });
            block.set_control(ControlNode::Return { value: eq });
            graph.add_block(block);

            let result_eq = run(&graph, 2, &[5, 5]);
            let result_neq = run(&graph, 2, &[5, 6]);
            assert_eq!(result_eq, JIT_TRUE, "5 == 5 should be JIT_TRUE");
            assert_eq!(result_neq, JIT_FALSE, "5 == 6 should be JIT_FALSE");
        }

        #[test]
        fn test_execute_deoptimize_control_returns_error() {
            let mut graph = MaglevGraph::new(0);
            let mut block = BasicBlock::new(0);
            block.set_control(ControlNode::Deoptimize {
                bytecode_offset: 0,
                reason: 0,
            });
            graph.add_block(block);

            let cc = do_compile(&graph, 0);
            // SAFETY: code was produced by MaglevCodegen.
            let result = unsafe { cc.execute(&[]) };
            assert!(
                result.is_err(),
                "Deoptimize control node must cause an error"
            );
        }
    }
}
