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
//! | 1     | RCX      | caller-saved       |
//! | 2     | RDX      | caller-saved       |
//! | 3     | RSI      | caller-saved       |
//! | 4     | R8       | caller-saved       |
//! | 5     | R9       | caller-saved       |
//! | 6     | R13      | callee-saved       |
//! | 7     | R12      | callee-saved       |
//! | 8     | R15      | callee-saved       |
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
use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────────────────────────────────────
// Constants & helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Number of physical registers available to the Maglev register allocator.
pub const NUM_PHYS_REGS: u32 = 9;

/// Bytes per monomorphic call-site cache slot (4 × i64 = 32 bytes).
///
/// ```text
/// [RBP - base - 0]   cached callee i64 (0 = empty)
/// [RBP - base - 8]   cached entry point
/// [RBP - base - 16]  cached context pointer
/// [RBP - base - 24]  cached BA pointer
/// ```
#[cfg(all(target_arch = "x86_64", unix))]
const MONO_CACHE_SLOT_BYTES: i32 = 32;

/// Offset from the start of a mono-cache slot to each field.
#[cfg(all(target_arch = "x86_64", unix))]
const MONO_OFF_CALLEE: i32 = 0;
#[cfg(all(target_arch = "x86_64", unix))]
const MONO_OFF_ENTRY: i32 = 8;
#[cfg(all(target_arch = "x86_64", unix))]
const MONO_OFF_CTX: i32 = 16;
#[cfg(all(target_arch = "x86_64", unix))]
const MONO_OFF_BA: i32 = 24;

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
        use smallvec::{SmallVec, smallvec};
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

        let mut regs: SmallVec<[i64; 32]> = smallvec![0i64; self.register_file_slots];
        for (i, &value) in args.iter().enumerate().take(regs.len()) {
            regs[i] = value;
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

    /// Returns the raw entry-point pointer of the compiled Maglev code.
    ///
    /// This is the `mmap`'d executable page that can be called as
    /// `extern "C" fn(*mut i64) -> i64`.  Used by the mono-cache upgrade
    /// path to patch a cached baseline entry point with the Maglev one.
    pub fn entry_point(&self) -> *const u8 {
        self.ptr
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
        7 => Reg64::R12,
        8 => Reg64::R15,
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
    /// Bitmask of caller-saved allocatable registers that are actually used.
    /// Bit i means register index i is allocated somewhere.  Only the
    /// caller-saved subset (RCX=1, RDX=2, RSI=3, R8=4, R9=5) is relevant
    /// for save/restore around stub calls.
    #[cfg_attr(not(all(target_arch = "x86_64", unix)), allow(dead_code))]
    used_caller_saved: u8,
    /// Set of wrapping Int32 operation results whose consumers ALL only use
    /// the lower 32 bits.  For these nodes the post-operation `movsxd` can be
    /// skipped because the next consumer will operate on 32-bit values anyway.
    narrow_int32: HashSet<NodeId>,
    /// Set of nodes whose values are provably within the i32 range.  Computed
    /// by a forward (producer-driven) analysis.  Used to enable 32-bit
    /// emission of `CheckedSmi*` operations when both inputs are i32-range,
    /// eliminating the 64-bit ALU + MOVSXD pattern in favour of a single
    /// 32-bit ALU instruction + `JO` deopt.
    i32_range: HashSet<NodeId>,
    /// Total stack bytes reserved in the prologue for monomorphic call caches.
    /// Each `CallUndefinedReceiver0` call site gets one 32-byte cache slot.
    /// Zero if the function has no direct-call-0 sites.
    #[cfg_attr(not(all(target_arch = "x86_64", unix)), allow(dead_code))]
    mono_call_cache_bytes: i32,
    /// Counter used during block emission to assign sequential cache-slot
    /// offsets to each `CallUndefinedReceiver0` site.
    #[cfg_attr(not(all(target_arch = "x86_64", unix)), allow(dead_code))]
    next_mono_cache_site: i32,
    /// Deferred cold-path branch targets.  When a fused compare-branch's
    /// true target is the very next block, the false (unlikely) path is
    /// deferred here and emitted after all main blocks.  This lets the
    /// hot loop path fall through without an extra JMP.
    deferred_branches: Vec<DeferredBranch>,
}

/// A deferred cold-path branch emitted out-of-line after all blocks.
struct DeferredBranch {
    /// Forward-reference label jumped to by the conditional branch.
    label: Label,
    /// Phi copies to emit at the cold path (pred → successor).
    pred_idx: u32,
    successor_idx: u32,
    /// Block label to jump to after Phi copies.
    target_block: u32,
}

impl<'a> MaglevCodegen<'a> {
    fn new(graph: &'a MaglevGraph, alloc: &'a AllocationResult, param_count: u32) -> Self {
        let num_blocks = graph.blocks().len();
        // Precompute which caller-saved registers are actually allocated.
        let mut used_caller_saved: u8 = 0;
        // Count direct-call-0 sites for monomorphic cache allocation.
        let mut mono_call_sites: i32 = 0;
        for block in graph.blocks() {
            for (node_id, _) in &block.nodes {
                if let Some(Location::Register(n)) = alloc.location(*node_id) {
                    // Only track caller-saved: RCX(1), RDX(2), RSI(3), R8(4), R9(5)
                    if (1..=5).contains(&n) {
                        used_caller_saved |= 1 << n;
                    }
                }
            }
            for (_nid, node) in &block.nodes {
                if let ValueNode::Call { receiver, args, .. }
                | ValueNode::CallKnownFunction { receiver, args, .. } = node
                {
                    let recv_is_undef =
                        matches!(graph.node(*receiver), Some(ValueNode::UndefinedConstant));
                    if recv_is_undef && args.is_empty() {
                        mono_call_sites += 1;
                    }
                }
            }
        }
        #[cfg(all(target_arch = "x86_64", unix))]
        let mono_call_cache_bytes = mono_call_sites.saturating_mul(MONO_CACHE_SLOT_BYTES);
        #[cfg(not(all(target_arch = "x86_64", unix)))]
        let mono_call_cache_bytes = 0i32;
        // Keep the total a multiple of 16 for stack alignment preservation.
        let mono_call_cache_bytes = (mono_call_cache_bytes + 15) & !15;
        let mono_call_cache_bytes = if mono_call_sites == 0 {
            0
        } else {
            mono_call_cache_bytes
        };
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
            deopt_divzero_label: Label::new(),
            deopt_common_label: Label::new(),
            safepoints: Vec::new(),
            deopt_entries: Vec::new(),
            source_positions: Vec::new(),
            promoted_globals: Vec::new(),
            promoted_extra_slots: 0,
            used_caller_saved,
            narrow_int32: HashSet::new(),
            i32_range: HashSet::new(),
            mono_call_cache_bytes,
            next_mono_cache_site: 0,
            deferred_branches: Vec::new(),
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

        // Narrow-Int32 analysis: identify wrapping Int32 operations whose
        // consumers ALL only read the lower 32 bits.  These can skip the
        // post-operation MOVSXD sign-extension, saving ~1 instruction per op.
        // Spilled values and values feeding Phis/Returns/StoreGlobals are
        // conservatively excluded.
        //
        // i32-range analysis (forward / producer-driven): identify nodes whose
        // values are provably within the i32 range.  When both inputs of a
        // CheckedSmi* operation are i32-range the operation is added as a
        // narrow candidate so it can be emitted with a 32-bit ALU instruction
        // + JO, eliminating the extra MOVSXD.
        self.i32_range = Self::compute_i32_range(self.graph);
        self.narrow_int32 = Self::compute_narrow_int32(self.graph, self.alloc, &self.i32_range);

        self.emit_prologue();
        self.emit_promoted_global_loads();

        for block_idx in 0..self.graph.blocks().len() {
            self.emit_block(block_idx as u32);
        }

        // Emit deferred cold-path branches (loop exit paths placed
        // out-of-line so the hot loop body can fall through).
        self.emit_deferred_branches();

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
    /// push  rbp         ; frame pointer
    /// mov   rbp, rsp
    /// push  rbx         ; callee-saved allocatable Register(0)
    /// push  r14         ; callee-saved reg-file pointer
    /// push  r13         ; callee-saved allocatable Register(6)
    /// push  r12         ; callee-saved allocatable Register(7)
    /// push  r15         ; callee-saved allocatable Register(8)
    /// mov   r14, rdi    ; r14 = regs argument (SysV: first arg in RDI)
    /// ```
    fn emit_prologue(&mut self) {
        self.masm.push(Reg64::Rbp);
        self.masm.mov_rr(Reg64::Rbp, Reg64::Rsp);
        self.masm.push(Reg64::Rbx);
        self.masm.push(Reg64::R14);
        self.masm.push(Reg64::R13);
        self.masm.push(Reg64::R12);
        self.masm.push(Reg64::R15);
        // After 6 pushes (+ return-address by `call` = 7 total):
        // RSP ≡ 8 mod 16.  Stub calls use selective save (with
        // alignment padding when needed) to align RSP to 0 mod 16
        // before the inner `call`.

        // Allocate monomorphic call-cache slots (kept as a multiple of
        // 16, so RSP alignment is preserved: still ≡ 8 mod 16).
        if self.mono_call_cache_bytes > 0 {
            self.masm.sub_ri(Reg64::Rsp, self.mono_call_cache_bytes);
            // Zero all cache slots (callee field = 0 → empty).
            self.masm.xor_rr(Reg64::R11, Reg64::R11);
            let slots = self.mono_call_cache_bytes / 8;
            for i in 0..slots {
                self.masm
                    .mov_store_base_disp32(Reg64::Rsp, i * 8, Reg64::R11);
            }
        }

        self.masm.mov_rr(Reg64::R14, Reg64::Rdi);
    }

    /// Emit the normal function return sequence.
    ///
    /// `rax` must already hold the return value before calling this.
    ///
    /// ```text
    /// pop  r15
    /// pop  r12
    /// pop  r13
    /// pop  r14
    /// pop  rbx
    /// pop  rbp
    /// ret
    /// ```
    fn emit_normal_epilogue(&mut self) {
        if self.mono_call_cache_bytes > 0 {
            self.masm.add_ri(Reg64::Rsp, self.mono_call_cache_bytes);
        }
        self.masm.pop(Reg64::R15);
        self.masm.pop(Reg64::R12);
        self.masm.pop(Reg64::R13);
        self.masm.pop(Reg64::R14);
        self.masm.pop(Reg64::Rbx);
        self.masm.pop(Reg64::Rbp);
        self.masm.ret();
    }

    /// Emit all deferred cold-path branches.  These are loop-exit paths
    /// moved out-of-line so the hot loop body can fall through from the
    /// fused compare-branch without an extra JMP.
    fn emit_deferred_branches(&mut self) {
        let branches = std::mem::take(&mut self.deferred_branches);
        for mut db in branches {
            self.masm.bind_label(&mut db.label);
            self.emit_phi_copies_for_successor(db.pred_idx, db.successor_idx);
            let target = db.target_block as usize;
            self.masm.jmp(&mut self.block_labels[target]);
        }
    }

    /// Emit the shared deopt epilogue with categorised deopt labels.
    ///
    /// Category labels set RAX to their specific constant and jump to
    /// `deopt_common_label`.  The uncategorised `deopt_label` sets
    /// `JIT_DEOPT` and falls through to common.
    fn emit_deopt_epilogue(&mut self) {
        use crate::compiler::baseline::compiler::{
            JIT_DEOPT_DIVZERO, JIT_DEOPT_GLOBAL, JIT_DEOPT_OVERFLOW, JIT_DEOPT_STUB,
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

        self.masm.bind_label(&mut self.deopt_divzero_label);
        self.masm.mov_ri(Reg64::Rax, JIT_DEOPT_DIVZERO);
        self.masm.jmp(&mut self.deopt_common_label);

        // Uncategorised fallback (existing deopt paths that still use
        // deopt_label directly).
        self.masm.bind_label(&mut self.deopt_label);
        self.masm.mov_ri(Reg64::Rax, JIT_DEOPT);

        // Common exit — RAX already set, just restore and return.
        self.masm.bind_label(&mut self.deopt_common_label);
        if self.mono_call_cache_bytes > 0 {
            self.masm.add_ri(Reg64::Rsp, self.mono_call_cache_bytes);
        }
        self.masm.pop(Reg64::R15);
        self.masm.pop(Reg64::R12);
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

        // Align loop headers to 16-byte boundaries for better instruction
        // fetch throughput.  The NOP padding cost is paid once at loop entry;
        // every back-edge iteration benefits from aligned fetch.
        if block.is_loop_header {
            self.masm.align_to(16);
        }

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

        // Dead comparison elimination: when the block ends with a Branch
        // whose condition is a fusible Int32 comparison AND that comparison
        // is the last value node, skip emitting the 6-instruction boolean
        // materialization.  The Branch's try_fuse_compare_branch will emit
        // just CMP + JCC directly.
        let fused_cmp_id = Self::detect_fusible_branch_comparison(&nodes, &control);

        for (id, node) in &nodes {
            if Some(*id) == fused_cmp_id {
                continue;
            }
            self.emit_value_node(*id, node);
        }

        if let Some(ctrl) = control {
            self.emit_control_node(block_idx, &ctrl);
        }
    }

    /// Detect when a block's [`ControlNode::Branch`] condition is a fusible
    /// Int32 comparison that is the **last** value node in the block.
    ///
    /// In SSA order, no node after the comparison can exist (it *is* the last),
    /// and no node before it can reference a forward definition.  Therefore the
    /// comparison's only consumer is the Branch itself, making the 6-instruction
    /// boolean materialisation dead code — the Branch's
    /// [`try_fuse_compare_branch`] will emit just CMP + JCC.
    fn detect_fusible_branch_comparison(
        nodes: &[(NodeId, ValueNode)],
        control: &Option<ControlNode>,
    ) -> Option<NodeId> {
        let condition = match control {
            Some(ControlNode::Branch { condition, .. }) => *condition,
            _ => return None,
        };

        let (last_id, last_node) = nodes.last()?;
        if *last_id != condition {
            return None;
        }

        if matches!(
            last_node,
            ValueNode::Int32LessThan { .. }
                | ValueNode::Int32LessThanOrEqual { .. }
                | ValueNode::Int32GreaterThan { .. }
                | ValueNode::Int32GreaterThanOrEqual { .. }
                | ValueNode::Int32Equal { .. }
                | ValueNode::Int32StrictEqual { .. }
        ) {
            Some(condition)
        } else {
            None
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
            ValueNode::SmiConstant { value } => match self.alloc.location(id) {
                Some(Location::Register(n)) => {
                    self.masm.mov_ri(phys_reg(n), *value as i64);
                }
                _ => {
                    self.masm.mov_ri(Reg64::R11, *value as i64);
                    self.emit_store(id, Reg64::R11);
                }
            },
            ValueNode::Int32Constant { value } => match self.alloc.location(id) {
                Some(Location::Register(n)) => {
                    self.masm.mov_ri(phys_reg(n), *value as i64);
                }
                _ => {
                    self.masm.mov_ri(Reg64::R11, *value as i64);
                    self.emit_store(id, Reg64::R11);
                }
            },
            ValueNode::Uint32Constant { value } => match self.alloc.location(id) {
                Some(Location::Register(n)) => {
                    self.masm.mov_ri(phys_reg(n), *value as i64);
                }
                _ => {
                    self.masm.mov_ri(Reg64::R11, *value as i64);
                    self.emit_store(id, Reg64::R11);
                }
            },
            ValueNode::Float64Constant { value } => {
                // Store the f64 bit pattern as an i64.
                match self.alloc.location(id) {
                    Some(Location::Register(n)) => {
                        self.masm.mov_ri(phys_reg(n), value.to_bits() as i64);
                    }
                    _ => {
                        self.masm.mov_ri(Reg64::R11, value.to_bits() as i64);
                        self.emit_store(id, Reg64::R11);
                    }
                }
            }
            ValueNode::TrueConstant => match self.alloc.location(id) {
                Some(Location::Register(n)) => {
                    self.masm.mov_ri(phys_reg(n), JIT_TRUE);
                }
                _ => {
                    self.masm.mov_ri(Reg64::R11, JIT_TRUE);
                    self.emit_store(id, Reg64::R11);
                }
            },
            ValueNode::FalseConstant => match self.alloc.location(id) {
                Some(Location::Register(n)) => {
                    self.masm.mov_ri(phys_reg(n), JIT_FALSE);
                }
                _ => {
                    self.masm.mov_ri(Reg64::R11, JIT_FALSE);
                    self.emit_store(id, Reg64::R11);
                }
            },
            ValueNode::NullConstant => match self.alloc.location(id) {
                Some(Location::Register(n)) => {
                    self.masm.mov_ri(phys_reg(n), JIT_NULL);
                }
                _ => {
                    self.masm.mov_ri(Reg64::R11, JIT_NULL);
                    self.emit_store(id, Reg64::R11);
                }
            },
            ValueNode::UndefinedConstant => match self.alloc.location(id) {
                Some(Location::Register(n)) => {
                    self.masm.mov_ri(phys_reg(n), JIT_UNDEFINED);
                }
                _ => {
                    self.masm.mov_ri(Reg64::R11, JIT_UNDEFINED);
                    self.emit_store(id, Reg64::R11);
                }
            },

            // ── Parameters ───────────────────────────────────────────────────
            ValueNode::Parameter { index } => {
                let off = (*index * 8) as i32;
                match self.alloc.location(id) {
                    Some(Location::Register(n)) => {
                        self.masm.mov_load_base_disp32(phys_reg(n), Reg64::R14, off);
                    }
                    _ => {
                        self.masm.mov_load_base_disp32(Reg64::R11, Reg64::R14, off);
                        self.emit_store(id, Reg64::R11);
                    }
                }
            }

            // ── Int32 binary arithmetic ──────────────────────────────────────
            ValueNode::Int32Add { left, right } => {
                if let Some(imm) = self.try_get_i32_constant(*right) {
                    self.emit_wrapping_int32_binop_imm(*left, imm, id, MacroAssembler::add32_ri);
                } else if let Some(imm) = self.try_get_i32_constant(*left) {
                    // ADD is commutative
                    self.emit_wrapping_int32_binop_imm(*right, imm, id, MacroAssembler::add32_ri);
                } else {
                    self.emit_wrapping_int32_binop(*left, *right, id, MacroAssembler::add32_rr);
                }
            }
            ValueNode::Int32Subtract { left, right } => {
                if let Some(imm) = self.try_get_i32_constant(*right) {
                    self.emit_wrapping_int32_binop_imm(*left, imm, id, MacroAssembler::sub32_ri);
                } else {
                    self.emit_wrapping_int32_binop(*left, *right, id, MacroAssembler::sub32_rr);
                }
            }
            ValueNode::Int32Multiply { left, right } => {
                if let Some(imm) = self.try_get_i32_constant(*right) {
                    self.emit_imul32_imm(*left, imm, id);
                } else if let Some(imm) = self.try_get_i32_constant(*left) {
                    // MUL is commutative
                    self.emit_imul32_imm(*right, imm, id);
                } else {
                    self.emit_wrapping_int32_binop(*left, *right, id, MacroAssembler::imul32_rr);
                }
            }
            ValueNode::Int32BitwiseAnd { left, right } => {
                if let Some(imm) = self.try_get_i32_constant(*right) {
                    self.emit_int32_binop_imm(*left, imm, id, MacroAssembler::and_ri);
                } else {
                    self.emit_int32_binop(*left, *right, id, MacroAssembler::and_rr);
                }
            }
            ValueNode::Int32BitwiseOr { left, right } => {
                if let Some(imm) = self.try_get_i32_constant(*right) {
                    if imm == 0 {
                        // OR x, 0 is identity — just copy left to result.
                        self.emit_load(*left, Reg64::R11);
                        self.emit_store(id, Reg64::R11);
                    } else {
                        self.emit_int32_binop_imm(*left, imm, id, MacroAssembler::or_ri);
                    }
                } else {
                    self.emit_int32_binop(*left, *right, id, MacroAssembler::or_rr);
                }
            }
            ValueNode::Int32BitwiseXor { left, right } => {
                if let Some(imm) = self.try_get_i32_constant(*right) {
                    self.emit_int32_binop_imm(*left, imm, id, MacroAssembler::xor_ri);
                } else {
                    self.emit_int32_binop(*left, *right, id, MacroAssembler::xor_rr);
                }
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
                let narrow = self.narrow_int32.contains(&id);
                match self.alloc.location(id) {
                    Some(Location::Register(n)) => {
                        let dst = phys_reg(n);
                        self.emit_load(*value, dst);
                        self.masm.neg_r(dst);
                        if !narrow {
                            self.masm.movsxd_sign_extend(dst, dst);
                        }
                    }
                    _ => {
                        self.emit_load(*value, Reg64::R11);
                        self.masm.neg_r(Reg64::R11);
                        if !narrow {
                            self.masm.movsxd_sign_extend(Reg64::R11, Reg64::R11);
                        }
                        self.emit_store(id, Reg64::R11);
                    }
                }
            }
            ValueNode::Int32Increment { value } => {
                let narrow = self.narrow_int32.contains(&id);
                match self.alloc.location(id) {
                    Some(Location::Register(n)) => {
                        let dst = phys_reg(n);
                        self.emit_load(*value, dst);
                        self.masm.add32_ri(dst, 1);
                        if !narrow {
                            self.masm.movsxd_sign_extend(dst, dst);
                        }
                    }
                    _ => {
                        self.emit_load(*value, Reg64::R11);
                        self.masm.add32_ri(Reg64::R11, 1);
                        if !narrow {
                            self.masm.movsxd_sign_extend(Reg64::R11, Reg64::R11);
                        }
                        self.emit_store(id, Reg64::R11);
                    }
                }
            }
            ValueNode::Int32Decrement { value } => {
                let narrow = self.narrow_int32.contains(&id);
                match self.alloc.location(id) {
                    Some(Location::Register(n)) => {
                        let dst = phys_reg(n);
                        self.emit_load(*value, dst);
                        self.masm.sub32_ri(dst, 1);
                        if !narrow {
                            self.masm.movsxd_sign_extend(dst, dst);
                        }
                    }
                    _ => {
                        self.emit_load(*value, Reg64::R11);
                        self.masm.sub32_ri(Reg64::R11, 1);
                        if !narrow {
                            self.masm.movsxd_sign_extend(Reg64::R11, Reg64::R11);
                        }
                        self.emit_store(id, Reg64::R11);
                    }
                }
            }

            // ── Checked Smi arithmetic (deopt on signed overflow) ────────────
            //
            // When both inputs are proven i32-range the operation can use a
            // 32-bit ALU instruction whose OF flag captures i32 overflow,
            // followed by a JO deopt.  The 32-bit result's upper 32 bits
            // are zero-extended by the CPU; downstream narrow consumers
            // only read the lower 32 bits, so no MOVSXD is needed.
            //
            // When either input may exceed i32 the 64-bit path is used
            // (identical to the previous code) to correctly handle the
            // full Smi range.
            ValueNode::CheckedSmiAdd { left, right } => {
                if self.narrow_int32.contains(&id) {
                    self.emit_checked_smi_binop(*left, *right, id, MacroAssembler::add32_rr);
                } else {
                    self.emit_checked_smi_binop(*left, *right, id, MacroAssembler::add_rr);
                }
            }
            ValueNode::CheckedSmiSubtract { left, right } => {
                if self.narrow_int32.contains(&id) {
                    self.emit_checked_smi_binop(*left, *right, id, MacroAssembler::sub32_rr);
                } else {
                    self.emit_checked_smi_binop(*left, *right, id, MacroAssembler::sub_rr);
                }
            }
            ValueNode::CheckedSmiMultiply { left, right } => {
                if self.narrow_int32.contains(&id) {
                    self.emit_checked_smi_binop(*left, *right, id, MacroAssembler::imul32_rr);
                } else {
                    self.emit_checked_smi_binop(*left, *right, id, MacroAssembler::imul_rr);
                }
            }
            ValueNode::CheckedSmiIncrement { value } => {
                let narrow = self.narrow_int32.contains(&id);
                match self.alloc.location(id) {
                    Some(Location::Register(n)) => {
                        let dst = phys_reg(n);
                        self.emit_load(*value, dst);
                        if narrow {
                            self.masm.add32_ri(dst, 1);
                        } else {
                            self.masm.add_ri(dst, 1);
                        }
                        self.emit_deopt_on_i64_overflow(0);
                    }
                    _ => {
                        self.emit_load(*value, Reg64::R11);
                        if narrow {
                            self.masm.add32_ri(Reg64::R11, 1);
                        } else {
                            self.masm.add_ri(Reg64::R11, 1);
                        }
                        self.emit_deopt_on_i64_overflow(0);
                        self.emit_store(id, Reg64::R11);
                    }
                }
            }
            ValueNode::CheckedSmiDecrement { value } => {
                let narrow = self.narrow_int32.contains(&id);
                match self.alloc.location(id) {
                    Some(Location::Register(n)) => {
                        let dst = phys_reg(n);
                        self.emit_load(*value, dst);
                        if narrow {
                            self.masm.sub32_ri(dst, 1);
                        } else {
                            self.masm.sub_ri(dst, 1);
                        }
                        self.emit_deopt_on_i64_overflow(0);
                    }
                    _ => {
                        self.emit_load(*value, Reg64::R11);
                        if narrow {
                            self.masm.sub32_ri(Reg64::R11, 1);
                        } else {
                            self.masm.sub_ri(Reg64::R11, 1);
                        }
                        self.emit_deopt_on_i64_overflow(0);
                        self.emit_store(id, Reg64::R11);
                    }
                }
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
            // CheckSmi / CheckNumber: pass-through for i64 Smis.
            //
            // Stator Smis are full i64 values — there is no i32 constraint.
            // Stubs already return JIT_DEOPT for non-integer values (caught
            // by emit_deopt_check_rax), so CheckSmi is redundant and was
            // previously causing spurious deopts for values > i32 max
            // (e.g. loop accumulators exceeding ~2.1 billion).
            ValueNode::CheckSmi { receiver } | ValueNode::CheckNumber { receiver } => {
                if self.alloc.location(id) != self.alloc.location(*receiver) {
                    self.emit_load(*receiver, Reg64::R11);
                    self.emit_store(id, Reg64::R11);
                }
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
                self.emit_int32_binop(*left, *right, id, MacroAssembler::add_rr);
            }
            ValueNode::Uint32Subtract { left, right } => {
                self.emit_int32_binop(*left, *right, id, MacroAssembler::sub_rr);
            }
            ValueNode::Uint32Multiply { left, right } => {
                self.emit_int32_binop(*left, *right, id, MacroAssembler::imul_rr);
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
                    let saved = self.emit_save_live_regs(id);
                    self.masm.mov_ri(Reg64::Rdi, i64::from(*name));
                    let addr = jit_runtime::jit_runtime_lda_global as *const () as usize as i64;
                    self.masm.mov_ri(Reg64::R11, addr);
                    self.masm.call_reg(Reg64::R11);
                    self.emit_restore_live_regs(saved);
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
                    // Sign-extend narrow values before storing to the
                    // promoted slot — the exit-path sta_global needs full
                    // 64-bit values.  This is on the cold exit path only.
                    match self.alloc.location(*value) {
                        Some(Location::Register(n)) => {
                            let src = phys_reg(n);
                            if self.narrow_int32.contains(value) {
                                self.masm.movsxd_sign_extend(src, src);
                            }
                            self.masm.mov_store_base_disp32(Reg64::R14, off, src);
                        }
                        _ => {
                            self.emit_load(*value, Reg64::R11);
                            if self.narrow_int32.contains(value) {
                                self.masm.movsxd_sign_extend(Reg64::R11, Reg64::R11);
                            }
                            self.masm.mov_store_base_disp32(Reg64::R14, off, Reg64::R11);
                        }
                    }
                } else {
                    let saved = self.emit_save_live_regs(id);
                    self.masm.mov_ri(Reg64::Rdi, i64::from(*name));
                    self.emit_load(*value, Reg64::Rsi);
                    // Sign-extend narrow values for the runtime stub.
                    if self.narrow_int32.contains(value) {
                        self.masm.movsxd_sign_extend(Reg64::Rsi, Reg64::Rsi);
                    }
                    let addr = jit_runtime::jit_runtime_sta_global as *const () as usize as i64;
                    self.masm.mov_ri(Reg64::R11, addr);
                    self.masm.call_reg(Reg64::R11);
                    self.emit_restore_live_regs(saved);
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
                    jit_runtime::jit_runtime_lda_named_property as *const () as usize,
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
                    jit_runtime::jit_runtime_sta_named_property as *const () as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::LoadKeyedGeneric { object, key, .. } => {
                if self.is_known_int32_key(*key) {
                    self.emit_inline_load_keyed_smi(id, *object, *key);
                } else {
                    self.emit_stub_call_2node(
                        id,
                        *object,
                        *key,
                        jit_runtime::jit_runtime_lda_keyed_property as *const () as usize,
                    );
                }
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::StoreKeyedGeneric {
                object, key, value, ..
            } => {
                if self.is_known_int32_key(*key) {
                    self.emit_inline_store_keyed_smi(id, *object, *key, *value);
                } else {
                    self.emit_stub_call_3node(
                        id,
                        *object,
                        *key,
                        *value,
                        jit_runtime::jit_runtime_sta_keyed_property as *const () as usize,
                    );
                }
            }

            // ── FixedArray element access (routed through keyed-property stubs) ──
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::LoadFixedArrayElement { elements, index }
            | ValueNode::LoadFixedDoubleArrayElement { elements, index }
            | ValueNode::LoadHoleyFixedDoubleArrayElement { elements, index } => {
                if self.is_known_int32_key(*index) {
                    self.emit_inline_load_keyed_smi(id, *elements, *index);
                } else {
                    self.emit_stub_call_2node(
                        id,
                        *elements,
                        *index,
                        jit_runtime::jit_runtime_lda_keyed_property as *const () as usize,
                    );
                }
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::StoreFixedArrayElement {
                elements,
                index,
                value,
            }
            | ValueNode::StoreFixedDoubleArrayElement {
                elements,
                index,
                value,
            } => {
                if self.is_known_int32_key(*index) {
                    self.emit_inline_store_keyed_smi(id, *elements, *index, *value);
                } else {
                    self.emit_stub_call_3node(
                        id,
                        *elements,
                        *index,
                        *value,
                        jit_runtime::jit_runtime_sta_keyed_property as *const () as usize,
                    );
                }
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
                let saved = self.emit_save_live_regs(id);
                self.masm.mov_ri(Reg64::Rdi, i64::from(*slot));
                self.masm.mov_ri(
                    Reg64::R11,
                    jit_runtime::jit_runtime_lda_context_slot as *const () as usize as i64,
                );
                self.masm.call_reg(Reg64::R11);
                self.emit_restore_live_regs(saved);
                self.emit_deopt_check_rax();
                self.masm.mov_rr(Reg64::R11, Reg64::Rax);
                self.emit_store(id, Reg64::R11);
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::StoreCurrentContextSlot { slot, value } => {
                // jit_runtime_sta_context_slot(slot_idx: i64, value: i64) -> i64
                let saved = self.emit_save_live_regs(id);
                self.masm.mov_ri(Reg64::Rdi, i64::from(*slot));
                self.emit_load(*value, Reg64::Rsi);
                self.masm.mov_ri(
                    Reg64::R11,
                    jit_runtime::jit_runtime_sta_context_slot as *const () as usize as i64,
                );
                self.masm.call_reg(Reg64::R11);
                self.emit_restore_live_regs(saved);
                self.emit_deopt_check_rax();
                self.masm.mov_rr(Reg64::R11, Reg64::Rax);
                self.emit_store(id, Reg64::R11);
            }

            // ── Function calls — direct JIT-to-JIT with stub fallback ──────
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::Call {
                callee,
                receiver,
                args,
                ..
            } => {
                // Check receiver: if it's UndefinedConstant, use the
                // CallUndefinedReceiver path (with direct-call fast path);
                // otherwise use CallProperty stubs.
                let recv_is_undef = matches!(
                    self.graph.node(*receiver),
                    Some(ValueNode::UndefinedConstant)
                );
                match (recv_is_undef, args.len()) {
                    // CallUndefinedReceiver0 — direct call fast path
                    (true, 0) => {
                        self.emit_direct_call_0(id, *callee);
                    }
                    // CallUndefinedReceiver1 — direct call fast path
                    (true, 1) => {
                        self.emit_direct_call_1(id, *callee, args[0]);
                    }
                    // CallUndefinedReceiver2 — direct call fast path
                    (true, 2) => {
                        self.emit_direct_call_2(id, *callee, args[0], args[1]);
                    }
                    // CallProperty0
                    (false, 0) => {
                        self.emit_stub_call_2node(
                            id,
                            *callee,
                            *receiver,
                            jit_runtime::jit_runtime_call_property0 as *const () as usize,
                        );
                    }
                    // CallProperty1
                    (false, 1) => {
                        self.emit_stub_call_3node(
                            id,
                            *callee,
                            *receiver,
                            args[0],
                            jit_runtime::jit_runtime_call_property1 as *const () as usize,
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
                        self.emit_direct_call_0(id, *callee);
                    }
                    (true, 1) => {
                        self.emit_direct_call_1(id, *callee, args[0]);
                    }
                    (true, 2) => {
                        self.emit_direct_call_2(id, *callee, args[0], args[1]);
                    }
                    (false, 0) => {
                        self.emit_stub_call_2node(
                            id,
                            *callee,
                            *receiver,
                            jit_runtime::jit_runtime_call_property0 as *const () as usize,
                        );
                    }
                    (false, 1) => {
                        self.emit_stub_call_3node(
                            id,
                            *callee,
                            *receiver,
                            args[0],
                            jit_runtime::jit_runtime_call_property1 as *const () as usize,
                        );
                    }
                    _ => {
                        self.emit_deopt_unconditional(0);
                        self.masm.mov_ri(Reg64::R11, JIT_UNDEFINED);
                        self.emit_store(id, Reg64::R11);
                    }
                }
            }

            // ── Object / array / closure creation ─────────────────────────────
            //
            // CreateObjectLiteral and CreateShallowObjectLiteral use a
            // direct stub call (2 immediate args) instead of the generic
            // trampoline to avoid the 5-arg setup and opcode-dispatch
            // overhead.
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::CreateObjectLiteral {
                feedback_slot,
                flags,
                ..
            } => {
                let saved = self.emit_save_live_regs(id);
                self.masm.mov_ri(Reg64::Rdi, i64::from(*feedback_slot));
                self.masm.mov_ri(Reg64::Rsi, i64::from(*flags));
                let addr = jit_runtime::jit_runtime_fast_create_object_literal as *const () as usize
                    as i64;
                self.masm.mov_ri(Reg64::R11, addr);
                self.masm.call_reg(Reg64::R11);
                self.emit_restore_live_regs(saved);
                self.emit_deopt_check_rax();
                self.emit_store(id, Reg64::Rax);
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::CreateShallowObjectLiteral {
                feedback_slot,
                flags,
                ..
            } => {
                let saved = self.emit_save_live_regs(id);
                self.masm.mov_ri(Reg64::Rdi, i64::from(*feedback_slot));
                self.masm.mov_ri(Reg64::Rsi, i64::from(*flags));
                let addr = jit_runtime::jit_runtime_fast_create_object_literal as *const () as usize
                    as i64;
                self.masm.mov_ri(Reg64::R11, addr);
                self.masm.call_reg(Reg64::R11);
                self.emit_restore_live_regs(saved);
                self.emit_deopt_check_rax();
                self.emit_store(id, Reg64::Rax);
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
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::PushContext { context } => {
                self.emit_stub_call_1node(
                    id,
                    *context,
                    jit_runtime::jit_runtime_push_context as *const () as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::PopContext { context } => {
                self.emit_stub_call_1node(
                    id,
                    *context,
                    jit_runtime::jit_runtime_pop_context as *const () as usize,
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
                        jit_runtime::jit_runtime_construct0 as *const () as usize,
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
                    jit_runtime::jit_runtime_generic_add as *const () as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericSubtract { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_generic_sub as *const () as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericMultiply { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_generic_mul as *const () as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericDivide { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_generic_div as *const () as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericModulus { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_generic_mod as *const () as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericBitwiseAnd { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_generic_bitwise_and as *const () as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericBitwiseOr { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_generic_bitwise_or as *const () as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericBitwiseXor { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_generic_bitwise_xor as *const () as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericShiftLeft { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_generic_shift_left as *const () as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericShiftRight { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_generic_shift_right as *const () as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericShiftRightLogical { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_generic_shift_right_logical as *const () as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericNegate { value, .. } => {
                self.emit_stub_call_1node(
                    id,
                    *value,
                    jit_runtime::jit_runtime_generic_negate as *const () as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericIncrement { value, .. } => {
                self.emit_stub_call_1node(
                    id,
                    *value,
                    jit_runtime::jit_runtime_generic_increment as *const () as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericDecrement { value, .. } => {
                self.emit_stub_call_1node(
                    id,
                    *value,
                    jit_runtime::jit_runtime_generic_decrement as *const () as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::GenericBitwiseNot { value, .. } => {
                self.emit_stub_call_1node(
                    id,
                    *value,
                    jit_runtime::jit_runtime_generic_bitwise_not as *const () as usize,
                );
            }

            // ── Type checks and conversions via trampoline ────────────────────
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::CheckHeapObject { .. } => {
                // Guard pass-through — no code needed.
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::ToString { value, .. } => {
                self.emit_stub_call_1node(
                    id,
                    *value,
                    jit_runtime::jit_runtime_tostring as *const () as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::ToNumber { value, .. } => {
                self.emit_stub_call_1node(
                    id,
                    *value,
                    jit_runtime::jit_runtime_tonumber as *const () as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::TypeOf { value } => {
                self.emit_stub_call_1node(
                    id,
                    *value,
                    jit_runtime::jit_runtime_typeof as *const () as usize,
                );
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
                    jit_runtime::jit_runtime_tagged_equal as *const () as usize,
                );
            }
            #[cfg(all(target_arch = "x86_64", unix))]
            ValueNode::TaggedNotEqual { left, right, .. } => {
                self.emit_stub_call_2node(
                    id,
                    *left,
                    *right,
                    jit_runtime::jit_runtime_tagged_not_equal as *const () as usize,
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

                // Back-edge branch inversion: when jumping to a loop header
                // whose control is a fused compare-branch, duplicate the
                // comparison here and emit a conditional back-edge to the
                // body.  This converts:
                //   body: ADD + ADD + JMP header
                //   header: CMP + JGE exit
                // into:
                //   header: CMP + JGE exit   (first iteration only)
                //   body: ADD + ADD + CMP + JL body
                // saving one unconditional JMP per iteration.
                let rotated = self.try_rotate_back_edge(block_idx, target as u32);
                if !rotated {
                    self.masm.jmp(&mut self.block_labels[target]);
                }
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

                    let next_block = block_idx + 1;
                    let num_blocks = self.graph.blocks().len() as u32;
                    if *if_true == next_block && next_block < num_blocks {
                        // Hot path falls through to if_true; cold path deferred.
                        let mut cold_label = Label::new();
                        self.masm.jcc(CondCode::NotEqual, &mut cold_label);
                        self.emit_phi_copies_for_successor(block_idx, *if_true);
                        self.deferred_branches.push(DeferredBranch {
                            label: cold_label,
                            pred_idx: block_idx,
                            successor_idx: *if_false,
                            target_block: *if_false,
                        });
                    } else {
                        let mut false_path = Label::new();
                        self.masm.jcc(CondCode::NotEqual, &mut false_path);
                        self.emit_phi_copies_for_successor(block_idx, *if_true);
                        self.masm.jmp(&mut self.block_labels[if_true_idx]);
                        self.masm.bind_label(&mut false_path);
                        self.emit_phi_copies_for_successor(block_idx, *if_false);
                        self.masm.jmp(&mut self.block_labels[if_false_idx]);
                    }
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
            // Single or no Phi — no conflict possible.  Use direct
            // register-to-register copy when both are in physical regs.
            for (phi_id, src_id) in &phi_ops {
                match (self.alloc.location(*phi_id), self.alloc.location(*src_id)) {
                    (Some(Location::Register(dn)), Some(Location::Register(sn))) => {
                        let dst = phys_reg(dn);
                        let src = phys_reg(sn);
                        if dst != src {
                            self.masm.mov_rr(dst, src);
                        }
                    }
                    _ => {
                        self.emit_load(*src_id, Reg64::R11);
                        self.emit_store(*phi_id, Reg64::R11);
                    }
                }
            }
        } else {
            // Check if any destination conflicts with another source.
            // If no conflicts, we can do direct copies without stack.
            let has_conflict = phi_ops.iter().any(|(phi_id, _)| {
                let phi_loc = self.alloc.location(*phi_id);
                phi_ops
                    .iter()
                    .any(|(_, other_src)| self.alloc.location(*other_src) == phi_loc)
            });
            if !has_conflict {
                // No conflicts — safe to do direct copies.
                for (phi_id, src_id) in &phi_ops {
                    match (self.alloc.location(*phi_id), self.alloc.location(*src_id)) {
                        (Some(Location::Register(dn)), Some(Location::Register(sn))) => {
                            self.masm.mov_rr(phys_reg(dn), phys_reg(sn));
                        }
                        _ => {
                            self.emit_load(*src_id, Reg64::R11);
                            self.emit_store(*phi_id, Reg64::R11);
                        }
                    }
                }
            } else {
                // Parallel-move: push/pop through stack.
                for (_phi_id, src_id) in &phi_ops {
                    self.emit_load(*src_id, Reg64::R11);
                    self.masm.push(Reg64::R11);
                }
                for (phi_id, _src_id) in phi_ops.iter().rev() {
                    self.masm.pop(Reg64::R11);
                    self.emit_store(*phi_id, Reg64::R11);
                }
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

    /// If `id` refers to a `SmiConstant` or `Int32Constant` node whose value
    /// fits in an i32, return that value.  Used to emit `CMP reg, imm` and
    /// `ADD reg, imm` instead of loading the constant into a scratch register.
    fn try_get_i32_constant(&self, id: NodeId) -> Option<i32> {
        match self.graph.node(id)? {
            ValueNode::SmiConstant { value } => {
                let v = *value as i64;
                if v >= i32::MIN as i64 && v <= i32::MAX as i64 {
                    Some(v as i32)
                } else {
                    None
                }
            }
            ValueNode::Int32Constant { value } => Some(*value),
            ValueNode::Uint32Constant { value } => {
                let v = *value;
                if v <= i32::MAX as u32 {
                    Some(v as i32)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Returns `true` when `id` refers to an IR node that is known to
    /// produce an integer (Smi / Int32) value.  Used by the inline
    /// array-element emission to decide between the lean Smi fast-path
    /// and the generic keyed-property runtime call.
    ///
    /// Checks both the static node type and the `i32_range` set which
    /// includes Phi nodes whose inputs are all provably i32 (e.g. loop
    /// counters after range analysis).
    #[cfg(all(target_arch = "x86_64", unix))]
    fn is_known_int32_key(&self, id: NodeId) -> bool {
        // The i32_range set (computed via fixed-point) includes Phi nodes
        // that feed from Int32 producers — essential for promoted loop
        // counters used as array indices.
        if self.i32_range.contains(&id) {
            return true;
        }
        matches!(
            self.graph.node(id),
            Some(
                ValueNode::SmiConstant { .. }
                    | ValueNode::Int32Constant { .. }
                    | ValueNode::Uint32Constant { .. }
                    | ValueNode::Int32Add { .. }
                    | ValueNode::Int32Subtract { .. }
                    | ValueNode::Int32Multiply { .. }
                    | ValueNode::Int32Divide { .. }
                    | ValueNode::Int32Modulus { .. }
                    | ValueNode::Int32Negate { .. }
                    | ValueNode::Int32Increment { .. }
                    | ValueNode::Int32Decrement { .. }
                    | ValueNode::Int32BitwiseAnd { .. }
                    | ValueNode::Int32BitwiseOr { .. }
                    | ValueNode::Int32BitwiseXor { .. }
                    | ValueNode::Int32ShiftLeft { .. }
                    | ValueNode::Int32ShiftRight { .. }
                    | ValueNode::Int32ShiftRightLogical { .. }
                    | ValueNode::CheckedSmiAdd { .. }
                    | ValueNode::CheckedSmiSubtract { .. }
                    | ValueNode::CheckedSmiMultiply { .. }
                    | ValueNode::CheckedSmiIncrement { .. }
                    | ValueNode::CheckedSmiDecrement { .. }
                    | ValueNode::CheckedSmiDivide { .. }
                    | ValueNode::CheckedSmiModulus { .. }
            )
        )
    }

    // ── Direct-register binary operation helpers ────────────────────────────

    /// Emit an integer binary operation (`dst = left OP right`) using the
    /// result's allocated physical register directly, avoiding scratch
    /// register copies.
    ///
    /// When the result is in a physical register, this saves 1–3 MOV
    /// instructions per operation compared to the scratch-register path.
    fn emit_int32_binop(
        &mut self,
        left: NodeId,
        right: NodeId,
        result: NodeId,
        op: fn(&mut MacroAssembler, Reg64, Reg64),
    ) {
        match self.alloc.location(result) {
            Some(Location::Register(n)) => {
                let dst = phys_reg(n);
                // Check if right operand lives in the result register —
                // if so, loading left would clobber it.
                let right_in_dst = left != right
                    && matches!(
                        self.alloc.location(right),
                        Some(Location::Register(rn)) if phys_reg(rn) == dst
                    );
                if right_in_dst {
                    self.emit_load(right, Reg64::R10);
                    self.emit_load(left, dst);
                    op(&mut self.masm, dst, Reg64::R10);
                } else {
                    self.emit_load(left, dst);
                    match self.alloc.location(right) {
                        Some(Location::Register(rn)) if phys_reg(rn) != dst => {
                            op(&mut self.masm, dst, phys_reg(rn));
                        }
                        _ => {
                            self.emit_load(right, Reg64::R10);
                            op(&mut self.masm, dst, Reg64::R10);
                        }
                    }
                }
            }
            _ => {
                self.emit_load(left, Reg64::R11);
                self.emit_load(right, Reg64::R10);
                op(&mut self.masm, Reg64::R11, Reg64::R10);
                self.emit_store(result, Reg64::R11);
            }
        }
    }

    /// Emit `dst = src OP imm` using a register-immediate instruction.
    ///
    /// Saves 1 MOV instruction per operation compared to loading the constant
    /// into a scratch register (e.g., `ADD reg, 3` instead of `MOV R10, 3;
    /// ADD reg, R10`).
    fn emit_int32_binop_imm(
        &mut self,
        src: NodeId,
        imm: i32,
        result: NodeId,
        op: fn(&mut MacroAssembler, Reg64, i32),
    ) {
        match self.alloc.location(result) {
            Some(Location::Register(n)) => {
                let dst = phys_reg(n);
                self.emit_load(src, dst);
                op(&mut self.masm, dst, imm);
            }
            _ => {
                self.emit_load(src, Reg64::R11);
                op(&mut self.masm, Reg64::R11, imm);
                self.emit_store(result, Reg64::R11);
            }
        }
    }

    /// Emit a wrapping 32-bit binary operation: `dst = left OP32 right; movsxd dst, dst`.
    ///
    /// Uses a true 32-bit instruction (no REX.W) so the CPU wraps on overflow,
    /// then sign-extends the 32-bit result back to i64 for the Smi register file.
    /// Used for `Int32Add`/`Int32Subtract`/`Int32Multiply` generated by the
    /// truncation analysis pass (the `(expr) | 0` pattern).
    fn emit_wrapping_int32_binop(
        &mut self,
        left: NodeId,
        right: NodeId,
        result: NodeId,
        op: fn(&mut MacroAssembler, Reg64, Reg64),
    ) {
        let narrow = self.narrow_int32.contains(&result);
        match self.alloc.location(result) {
            Some(Location::Register(n)) => {
                let dst = phys_reg(n);
                let right_in_dst = left != right
                    && matches!(
                        self.alloc.location(right),
                        Some(Location::Register(rn)) if phys_reg(rn) == dst
                    );
                if right_in_dst {
                    self.emit_load(right, Reg64::R10);
                    self.emit_load(left, dst);
                    op(&mut self.masm, dst, Reg64::R10);
                } else {
                    self.emit_load(left, dst);
                    match self.alloc.location(right) {
                        Some(Location::Register(rn)) if phys_reg(rn) != dst => {
                            op(&mut self.masm, dst, phys_reg(rn));
                        }
                        _ => {
                            self.emit_load(right, Reg64::R10);
                            op(&mut self.masm, dst, Reg64::R10);
                        }
                    }
                }
                if !narrow {
                    self.masm.movsxd_sign_extend(dst, dst);
                }
            }
            _ => {
                self.emit_load(left, Reg64::R11);
                self.emit_load(right, Reg64::R10);
                op(&mut self.masm, Reg64::R11, Reg64::R10);
                if !narrow {
                    self.masm.movsxd_sign_extend(Reg64::R11, Reg64::R11);
                }
                self.emit_store(result, Reg64::R11);
            }
        }
    }

    /// Emit a wrapping 32-bit register-immediate operation + sign-extend.
    fn emit_wrapping_int32_binop_imm(
        &mut self,
        src: NodeId,
        imm: i32,
        result: NodeId,
        op: fn(&mut MacroAssembler, Reg64, i32),
    ) {
        let narrow = self.narrow_int32.contains(&result);
        match self.alloc.location(result) {
            Some(Location::Register(n)) => {
                let dst = phys_reg(n);
                self.emit_load(src, dst);
                op(&mut self.masm, dst, imm);
                if !narrow {
                    self.masm.movsxd_sign_extend(dst, dst);
                }
            }
            _ => {
                self.emit_load(src, Reg64::R11);
                op(&mut self.masm, Reg64::R11, imm);
                if !narrow {
                    self.masm.movsxd_sign_extend(Reg64::R11, Reg64::R11);
                }
                self.emit_store(result, Reg64::R11);
            }
        }
    }

    /// Emit `dst = src * imm` using the three-operand 32-bit IMUL + sign-extend.
    /// Emit a 32-bit multiply by an immediate constant, with strength reduction
    /// for small constants.
    ///
    /// - `*3` → `LEA dst, [src + src*2]` + `MOVSXD`  (1c vs 3c latency)
    /// - `*5` → `LEA dst, [src + src*4]` + `MOVSXD`
    /// - `*9` → `LEA dst, [src + src*8]` + `MOVSXD`
    /// - Otherwise → `IMUL32 dst, src, imm` + `MOVSXD`
    fn emit_imul32_imm(&mut self, src: NodeId, imm: i32, result: NodeId) {
        // Strength-reduce multiply by {3, 5, 9} to LEA with SIB.
        // LEA operates on 64-bit values; MOVSXD afterward gives wrapping
        // 32-bit semantics identical to IMUL32.
        let lea_scale: Option<u8> = match imm {
            3 => Some(2),
            5 => Some(4),
            9 => Some(8),
            _ => None,
        };

        let narrow = self.narrow_int32.contains(&result);

        match self.alloc.location(result) {
            Some(Location::Register(n)) => {
                let dst = phys_reg(n);
                let src_reg = match self.alloc.location(src) {
                    Some(Location::Register(sn)) => phys_reg(sn),
                    _ => {
                        self.emit_load(src, Reg64::R10);
                        Reg64::R10
                    }
                };
                // RBP/R13 cannot be LEA base with mod=00 (ambiguous encoding).
                if let Some(scale) = lea_scale {
                    if src_reg != Reg64::Rbp && src_reg != Reg64::R13 {
                        self.masm.lea_scaled(dst, src_reg, src_reg, scale);
                    } else {
                        self.masm.imul32_rri(dst, src_reg, imm);
                    }
                } else {
                    self.masm.imul32_rri(dst, src_reg, imm);
                }
                if !narrow {
                    self.masm.movsxd_sign_extend(dst, dst);
                }
            }
            _ => {
                self.emit_load(src, Reg64::R11);
                if let Some(scale) = lea_scale {
                    self.masm
                        .lea_scaled(Reg64::R11, Reg64::R11, Reg64::R11, scale);
                } else {
                    self.masm.imul32_rri(Reg64::R11, Reg64::R11, imm);
                }
                if !narrow {
                    self.masm.movsxd_sign_extend(Reg64::R11, Reg64::R11);
                }
                self.emit_store(result, Reg64::R11);
            }
        }
    }

    /// Like [`emit_int32_binop`] but emits an i32 overflow guard after the
    /// operation.  Uses **32-bit** arithmetic + `JO` (jump on overflow) + `MOVSXD`
    /// to sign-extend the result, saving one instruction per op versus the old
    /// `MOVSXD` + `CMP` + `JNE` approach.
    fn emit_checked_smi_binop(
        &mut self,
        left: NodeId,
        right: NodeId,
        result: NodeId,
        op: fn(&mut MacroAssembler, Reg64, Reg64),
    ) {
        // CheckedSmi operations work on full-width i64 Smis.  Use the 64-bit
        // op and then check for i64 overflow via the CPU's OF flag (JO).
        // Stator Smis are i64, so the only overflow that matters is 64-bit
        // overflow — NOT i32 range.  The old MOVSXD+CMP+JNE guard was wrong
        // because it rejected values >2^31 (e.g. sum of 0..100000 ≈ 5 billion).
        match self.alloc.location(result) {
            Some(Location::Register(n)) => {
                let dst = phys_reg(n);
                let right_in_dst = left != right
                    && matches!(
                        self.alloc.location(right),
                        Some(Location::Register(rn)) if phys_reg(rn) == dst
                    );
                if right_in_dst {
                    self.emit_load(right, Reg64::R10);
                    self.emit_load(left, dst);
                    op(&mut self.masm, dst, Reg64::R10);
                } else {
                    self.emit_load(left, dst);
                    match self.alloc.location(right) {
                        Some(Location::Register(rn)) if phys_reg(rn) != dst => {
                            op(&mut self.masm, dst, phys_reg(rn));
                        }
                        _ => {
                            self.emit_load(right, Reg64::R10);
                            op(&mut self.masm, dst, Reg64::R10);
                        }
                    }
                }
                // JO deopt — 64-bit overflow from the preceding ADD/SUB/IMUL.
                self.emit_deopt_on_i64_overflow(0);
            }
            _ => {
                self.emit_load(left, Reg64::R11);
                self.emit_load(right, Reg64::R10);
                op(&mut self.masm, Reg64::R11, Reg64::R10);
                // JO deopt — 64-bit overflow from the preceding ADD/SUB/IMUL.
                self.emit_deopt_on_i64_overflow(0);
                self.emit_store(result, Reg64::R11);
            }
        }
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
            // After the prologue (6 pushes + return addr = 7 items), RSP ≡ 8
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
    // These helpers save only the caller-saved allocatable registers that are
    // **live across** the call (determined per-call-site by the register
    // allocator's liveness data), set up arguments in SysV ABI order, call the
    // extern "C" stub, check for JIT_DEOPT, and store the result.

    /// Save only the caller-saved registers that are **live across**
    /// `at_node`.  Returns a bitmask of the saved register indices so that
    /// [`emit_restore_live_regs`] can pop exactly the right set.
    #[cfg(all(target_arch = "x86_64", unix))]
    fn emit_save_live_regs(&mut self, at_node: NodeId) -> u8 {
        // Intersect per-call-site liveness with the function-wide mask
        // as a safety belt.
        let mask = self.alloc.live_caller_saved_at(at_node) & self.used_caller_saved;
        let count = mask.count_ones();
        // Need an odd number of pushes for 16-byte stack alignment
        // before the CALL instruction.  After prologue, RSP ≡ 8 mod 16,
        // so an odd push count brings it to 0 mod 16.
        if count.is_multiple_of(2) {
            self.masm.push(Reg64::R11); // alignment padding
        }
        if mask & (1 << 1) != 0 {
            self.masm.push(Reg64::Rcx);
        }
        if mask & (1 << 2) != 0 {
            self.masm.push(Reg64::Rdx);
        }
        if mask & (1 << 3) != 0 {
            self.masm.push(Reg64::Rsi);
        }
        if mask & (1 << 4) != 0 {
            self.masm.push(Reg64::R8);
        }
        if mask & (1 << 5) != 0 {
            self.masm.push(Reg64::R9);
        }
        mask
    }

    /// Restore caller-saved registers previously saved by
    /// [`emit_save_live_regs`] (reverse push order).
    #[cfg(all(target_arch = "x86_64", unix))]
    fn emit_restore_live_regs(&mut self, mask: u8) {
        if mask & (1 << 5) != 0 {
            self.masm.pop(Reg64::R9);
        }
        if mask & (1 << 4) != 0 {
            self.masm.pop(Reg64::R8);
        }
        if mask & (1 << 3) != 0 {
            self.masm.pop(Reg64::Rsi);
        }
        if mask & (1 << 2) != 0 {
            self.masm.pop(Reg64::Rdx);
        }
        if mask & (1 << 1) != 0 {
            self.masm.pop(Reg64::Rcx);
        }
        let count = mask.count_ones();
        if count.is_multiple_of(2) {
            self.masm.pop(Reg64::R11); // alignment padding
        }
    }

    /// Emit a deopt check: if RAX represents a deopt signal, jump to
    /// stub-deopt epilogue.
    ///
    /// All deopt values are i64::MIN..i64::MIN+5 (< i32::MIN).  Valid
    /// results are always ≥ i32::MIN.  A single `CMP RAX, i32::MIN; JL`
    /// replaces the old 3-instruction `MOV R11, JIT_DEOPT; CMP; JE`.
    #[cfg(all(target_arch = "x86_64", unix))]
    fn emit_deopt_check_rax(&mut self) {
        self.masm.cmp_ri(Reg64::Rax, i32::MIN);
        self.masm.jcc(CondCode::Less, &mut self.deopt_stub_label);
    }

    /// Call a 1-arg stub: `stub(node_arg)`.
    #[cfg(all(target_arch = "x86_64", unix))]
    fn emit_stub_call_1node(&mut self, id: NodeId, arg0: NodeId, stub_addr: usize) {
        let saved = self.emit_save_live_regs(id);
        self.emit_load(arg0, Reg64::Rdi);
        self.masm.mov_ri(Reg64::R11, stub_addr as i64);
        self.masm.call_reg(Reg64::R11);
        self.emit_restore_live_regs(saved);
        self.emit_deopt_check_rax();
        self.emit_store(id, Reg64::Rax);
    }

    /// Call a 2-node-arg stub: `stub(node0, node1)`.
    #[cfg(all(target_arch = "x86_64", unix))]
    fn emit_stub_call_2node(&mut self, id: NodeId, arg0: NodeId, arg1: NodeId, stub_addr: usize) {
        let saved = self.emit_save_live_regs(id);
        self.emit_load(arg0, Reg64::Rdi);
        self.emit_load(arg1, Reg64::Rsi);
        self.masm.mov_ri(Reg64::R11, stub_addr as i64);
        self.masm.call_reg(Reg64::R11);
        self.emit_restore_live_regs(saved);
        self.emit_deopt_check_rax();
        self.emit_store(id, Reg64::Rax);
    }

    /// Emit an inline fast-path for keyed **load** when the key is a
    /// known integer.
    ///
    /// Calls `jit_runtime_inline_load_keyed_smi` which returns
    /// `{value, hit}` in `RAX:RDX`.  On a hit the result is used
    /// directly; on a miss the generic `jit_runtime_lda_keyed_property`
    /// stub is called as a fallback.
    ///
    /// Generated code layout:
    /// ```text
    ///   save live regs
    ///   load obj → RDI, key → RSI
    ///   push RSI; push RDI          // save args for fallback
    ///   call inline_load_keyed_smi  // → RAX = value, RDX = hit
    ///   test RDX, RDX
    ///   je   slow_path
    ///   add  RSP, 16                // discard saved args
    ///   jmp  done
    /// slow_path:
    ///   pop  RDI; pop RSI           // restore args
    ///   call lda_keyed_property
    /// done:
    ///   restore live regs
    ///   deopt check
    ///   store result
    /// ```
    #[cfg(all(target_arch = "x86_64", unix))]
    fn emit_inline_load_keyed_smi(&mut self, id: NodeId, object: NodeId, key: NodeId) {
        let saved = self.emit_save_live_regs(id);

        // Load arguments into ABI registers.
        self.emit_load(object, Reg64::Rdi);
        self.emit_load(key, Reg64::Rsi);

        // Save args for the potential fallback (2 pushes = 16 bytes,
        // preserves 16-byte stack alignment).
        self.masm.push(Reg64::Rsi);
        self.masm.push(Reg64::Rdi);

        // Call the lean inline helper: returns {value, hit} in RAX:RDX.
        let inline_addr = jit_runtime::jit_runtime_inline_load_keyed_smi as *const () as usize;
        self.masm.mov_ri(Reg64::R11, inline_addr as i64);
        self.masm.call_reg(Reg64::R11);

        // Check hit flag (RDX).
        let mut slow_label = Label::new();
        let mut done_label = Label::new();
        self.masm.test_rr(Reg64::Rdx, Reg64::Rdx);
        self.masm.je(&mut slow_label);

        // ── Fast path: inline helper handled it ──
        // Discard the saved args and jump to common exit.
        self.masm.add_ri(Reg64::Rsp, 16);
        self.masm.jmp(&mut done_label);

        // ── Slow path: fall back to the generic stub ──
        self.masm.bind_label(&mut slow_label);
        self.masm.pop(Reg64::Rdi); // restore object
        self.masm.pop(Reg64::Rsi); // restore key
        let stub_addr = jit_runtime::jit_runtime_lda_keyed_property as *const () as usize;
        self.masm.mov_ri(Reg64::R11, stub_addr as i64);
        self.masm.call_reg(Reg64::R11);

        // ── Common exit ──
        self.masm.bind_label(&mut done_label);
        self.emit_restore_live_regs(saved);
        // Deopt check: harmless on fast path (RAX is always valid);
        // catches JIT_DEOPT from the generic stub on the slow path.
        self.emit_deopt_check_rax();
        self.emit_store(id, Reg64::Rax);
    }

    /// Emit an inline fast-path for keyed **store** when the key is a
    /// known integer.
    ///
    /// Calls `jit_runtime_inline_store_keyed_smi` which returns
    /// `{value, hit}` in `RAX:RDX`.  On a hit the result is used
    /// directly; on a miss the generic `jit_runtime_sta_keyed_property`
    /// stub is called as a fallback.
    #[cfg(all(target_arch = "x86_64", unix))]
    fn emit_inline_store_keyed_smi(
        &mut self,
        id: NodeId,
        object: NodeId,
        key: NodeId,
        value: NodeId,
    ) {
        let saved = self.emit_save_live_regs(id);

        // Load all three arguments.
        self.emit_load(object, Reg64::Rdi);
        self.emit_load(key, Reg64::Rsi);
        self.emit_load(value, Reg64::Rdx);

        // Save args for fallback (4 pushes = 32 bytes for alignment).
        self.masm.push(Reg64::Rdx);
        self.masm.push(Reg64::Rsi);
        self.masm.push(Reg64::Rdi);
        self.masm.push(Reg64::R11); // alignment padding

        // Call the lean inline helper.
        let inline_addr = jit_runtime::jit_runtime_inline_store_keyed_smi as *const () as usize;
        self.masm.mov_ri(Reg64::R11, inline_addr as i64);
        self.masm.call_reg(Reg64::R11);

        // Check hit flag (RDX).
        let mut slow_label = Label::new();
        let mut done_label = Label::new();
        self.masm.test_rr(Reg64::Rdx, Reg64::Rdx);
        self.masm.je(&mut slow_label);

        // ── Fast path ──
        self.masm.add_ri(Reg64::Rsp, 32);
        self.masm.jmp(&mut done_label);

        // ── Slow path ──
        self.masm.bind_label(&mut slow_label);
        self.masm.pop(Reg64::R11); // discard padding
        self.masm.pop(Reg64::Rdi); // restore object
        self.masm.pop(Reg64::Rsi); // restore key
        self.masm.pop(Reg64::Rdx); // restore value
        let stub_addr = jit_runtime::jit_runtime_sta_keyed_property as *const () as usize;
        self.masm.mov_ri(Reg64::R11, stub_addr as i64);
        self.masm.call_reg(Reg64::R11);

        // ── Common exit ──
        self.masm.bind_label(&mut done_label);
        self.emit_restore_live_regs(saved);
        self.emit_deopt_check_rax();
        self.emit_store(id, Reg64::Rax);
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
        let saved = self.emit_save_live_regs(id);
        self.emit_load(arg0, Reg64::Rdi);
        self.emit_load(arg1, Reg64::Rsi);
        self.emit_load(arg2, Reg64::Rdx);
        self.masm.mov_ri(Reg64::R11, stub_addr as i64);
        self.masm.call_reg(Reg64::R11);
        self.emit_restore_live_regs(saved);
        self.emit_deopt_check_rax();
        self.emit_store(id, Reg64::Rax);
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
        let saved = self.emit_save_live_regs(id);
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
        self.emit_restore_live_regs(saved);
        self.emit_deopt_check_rax();
        self.emit_store(id, Reg64::Rax);
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
        let saved = self.emit_save_live_regs(id);
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
        self.emit_restore_live_regs(saved);
        self.emit_deopt_check_rax();
        self.emit_store(id, Reg64::Rax);
    }

    // ── Direct JIT-to-JIT call helpers ──────────────────────────────────────
    //
    // When the callee already has baseline JIT code cached, these methods
    // emit a fast path that:
    //   1. Calls `jit_runtime_get_jit_entry` to obtain the entry point.
    //   2. If non-null, allocates a 128-byte register file on the stack,
    //      stores arguments, and calls the entry point directly.
    //   3. Calls `jit_runtime_finish_direct_call` to restore TLS state.
    //   4. Falls back to the full runtime stub on cache miss.
    //
    // `CallUndefinedReceiver0` additionally implements a **monomorphic
    // inline cache** (MIC): after the first successful call through
    // `jit_runtime_get_jit_entry`, the callee identity, entry point,
    // context pointer, and BA pointer are cached in per-site frame slots.
    // On subsequent calls, if the callee identity matches, the much
    // lighter `jit_runtime_mono_call_prepare` is used instead.

    /// Emit a direct-call fast path for `CallUndefinedReceiver0` with a
    /// monomorphic inline cache.
    ///
    /// ## Mono fast path (cache hit)
    ///
    /// ```text
    ///  cmp  callee, [RBP - cache_callee_off]
    ///  jne  cache_miss
    ///  ; call mono_prepare(callee, cached_ba, cached_ctx)
    ///  ; allocate register file, call cached entry_point
    ///  ; call finish_direct_call
    /// ```
    ///
    /// ## Cache miss / first call
    ///
    /// ```text
    ///  call jit_runtime_get_jit_entry
    ///  ; on success: populate cache, call entry_point, finish
    ///  ; on failure: fall back to call_undefined_receiver0 stub
    /// ```
    #[cfg(all(target_arch = "x86_64", unix))]
    fn emit_direct_call_0(&mut self, id: NodeId, callee: NodeId) {
        // Claim a mono-cache slot from the frame.  Offsets are relative
        // to RBP.  After the prologue pushes (40 bytes from RBP) and the
        // cache allocation (sub rsp, mono_call_cache_bytes), the first
        // cache slot starts at [RBP - 48].
        //
        // Each slot is 32 bytes:
        //   +0  callee_i64  (0 = empty)
        //   +8  entry_point
        //   +16 ctx_ptr
        //   +24 ba_ptr
        let site = self.next_mono_cache_site;
        self.next_mono_cache_site += 1;
        // Base offset from RBP to the start of all cache slots.
        const CACHE_BASE: i32 = -48; // first slot starts right after R15 push
        let slot_base = CACHE_BASE - site * MONO_CACHE_SLOT_BYTES;
        let off_callee = slot_base - MONO_OFF_CALLEE;
        let off_entry = slot_base - MONO_OFF_ENTRY;
        let off_ctx = slot_base - MONO_OFF_CTX;
        let off_ba = slot_base - MONO_OFF_BA;

        let saved = self.emit_save_live_regs(id);
        // After save_live_regs: RSP ≡ 0 mod 16.

        // Load callee into R11 (scratch).
        self.emit_load(callee, Reg64::R11);

        // ── Monomorphic cache check ─────────────────────────────────
        let mut cache_miss = Label::new();
        let mut do_direct = Label::new();
        let mut done_label = Label::new();

        self.masm.cmp_rm(Reg64::R11, Reg64::Rbp, off_callee);
        self.masm.jne(&mut cache_miss);

        // ── Mono hit: inline context comparison ──────────────────────
        // Load cached context pointer.
        self.masm
            .mov_load_base_disp32(Reg64::R10, Reg64::Rbp, off_ctx);

        // Get current context pointer from runtime.
        // Push padding to align RSP to 0 mod 16 (already after prologue saves).
        self.masm.push(Reg64::R10);
        self.masm.push(Reg64::R11);
        let get_ctx_addr = jit_runtime::jit_runtime_get_current_ctx_ptr as *const () as usize;
        self.masm.mov_ri(Reg64::R9, get_ctx_addr as i64);
        self.masm.call_reg(Reg64::R9);
        // RAX now contains current context pointer.
        self.masm.pop(Reg64::R11);
        self.masm.pop(Reg64::R10);

        // Compare: cached ctx (R10) vs current ctx (RAX).
        self.masm.cmp_rr(Reg64::R10, Reg64::Rax);
        let mut ctx_mismatch = Label::new();
        self.masm.jne(&mut ctx_mismatch);

        // ── Context match: lightweight prepare (fast path) ────────────
        // R11 = callee_i64, R10 = ba_ptr.
        self.masm.mov_rr(Reg64::Rdi, Reg64::R11); // arg0 = callee
        self.masm
            .mov_load_base_disp32(Reg64::Rsi, Reg64::Rbp, off_ba); // arg1 = ba

        // Push callee + padding (2 pushes → RSP ≡ 0 mod 16).
        self.masm.push(Reg64::Rdi);
        self.masm.push(Reg64::Rdi);

        let same_ctx_addr =
            jit_runtime::jit_runtime_mono_call_prepare_same_ctx as *const () as usize;
        self.masm.mov_ri(Reg64::R11, same_ctx_addr as i64);
        self.masm.call_reg(Reg64::R11);

        // Check success (RAX = 0 → fail, 1 → use cached entry,
        // > 1 → Maglev upgrade: RAX is the new entry point).
        self.masm.test_rr(Reg64::Rax, Reg64::Rax);
        let mut mono_fail = Label::new();
        self.masm.je(&mut mono_fail);

        // RAX > 1 means Maglev entry point upgrade.
        self.masm.cmp_ri(Reg64::Rax, 1);
        let mut same_ctx_no_upgrade = Label::new();
        self.masm.je(&mut same_ctx_no_upgrade);

        // Upgraded: store new entry point into cache slot and use it.
        self.masm
            .mov_store_base_disp32(Reg64::Rbp, off_entry, Reg64::Rax);
        self.masm.mov_rr(Reg64::R11, Reg64::Rax);
        self.masm
            .mov_load_base_disp32(Reg64::R10, Reg64::Rbp, off_ctx);
        self.masm.jmp(&mut do_direct);

        // No upgrade: load cached entry + ctx for the direct call.
        self.masm.bind_label(&mut same_ctx_no_upgrade);
        self.masm
            .mov_load_base_disp32(Reg64::R11, Reg64::Rbp, off_entry);
        self.masm
            .mov_load_base_disp32(Reg64::R10, Reg64::Rbp, off_ctx);
        self.masm.jmp(&mut do_direct);

        // ── Context mismatch: full prepare (slow path) ───────────────
        self.masm.bind_label(&mut ctx_mismatch);
        // R11 = callee_i64, load ba and cached ctx again.
        self.masm.mov_rr(Reg64::Rdi, Reg64::R11); // arg0 = callee
        self.masm
            .mov_load_base_disp32(Reg64::Rsi, Reg64::Rbp, off_ba); // arg1 = ba
        self.masm
            .mov_load_base_disp32(Reg64::Rdx, Reg64::Rbp, off_ctx); // arg2 = ctx

        // Push callee + padding (2 pushes → RSP ≡ 0 mod 16).
        self.masm.push(Reg64::Rdi);
        self.masm.push(Reg64::Rdi);

        let mono_addr = jit_runtime::jit_runtime_mono_call_prepare as *const () as usize;
        self.masm.mov_ri(Reg64::R11, mono_addr as i64);
        self.masm.call_reg(Reg64::R11);

        // Check success (RAX = 0 → fail, 1 → use cached entry,
        // > 1 → Maglev upgrade: RAX is the new entry point).
        self.masm.test_rr(Reg64::Rax, Reg64::Rax);
        self.masm.je(&mut mono_fail);

        // RAX > 1 means Maglev entry point upgrade.
        self.masm.cmp_ri(Reg64::Rax, 1);
        let mut full_no_upgrade = Label::new();
        self.masm.je(&mut full_no_upgrade);

        // Upgraded: store new entry point into cache slot and use it.
        self.masm
            .mov_store_base_disp32(Reg64::Rbp, off_entry, Reg64::Rax);
        self.masm.mov_rr(Reg64::R11, Reg64::Rax);
        self.masm
            .mov_load_base_disp32(Reg64::R10, Reg64::Rbp, off_ctx);
        self.masm.jmp(&mut do_direct);

        // No upgrade: load cached entry + ctx for the direct call.
        self.masm.bind_label(&mut full_no_upgrade);
        self.masm
            .mov_load_base_disp32(Reg64::R11, Reg64::Rbp, off_entry);
        self.masm
            .mov_load_base_disp32(Reg64::R10, Reg64::Rbp, off_ctx);
        self.masm.jmp(&mut do_direct);

        // ── Cache miss: full jit_runtime_get_jit_entry path ─────────
        self.masm.bind_label(&mut cache_miss);

        self.masm.mov_rr(Reg64::Rdi, Reg64::R11); // callee
        // Push callee + padding.
        self.masm.push(Reg64::Rdi);
        self.masm.push(Reg64::Rdi);

        let get_entry_addr = jit_runtime::jit_runtime_get_jit_entry as *const () as usize;
        self.masm.mov_ri(Reg64::R11, get_entry_addr as i64);
        self.masm.call_reg(Reg64::R11);

        // Check entry_point == 0 → stub fallback.
        let mut stub_fallback = Label::new();
        self.masm.test_rr(Reg64::Rax, Reg64::Rax);
        self.masm.je(&mut stub_fallback);

        // RAX = entry_point, RDX = ctx_ptr.
        self.masm.mov_rr(Reg64::R11, Reg64::Rax); // entry
        self.masm.mov_rr(Reg64::R10, Reg64::Rdx); // ctx

        // ── Populate mono cache ─────────────────────────────────────
        // callee is at [RSP] (both pushes have the same value).
        self.masm.mov_load_base_disp32(Reg64::Rax, Reg64::Rsp, 0);
        self.masm
            .mov_store_base_disp32(Reg64::Rbp, off_callee, Reg64::Rax);
        self.masm
            .mov_store_base_disp32(Reg64::Rbp, off_entry, Reg64::R11);
        self.masm
            .mov_store_base_disp32(Reg64::Rbp, off_ctx, Reg64::R10);

        // Read the current BA (set by get_jit_entry) for caching.
        self.masm.push(Reg64::R11);
        self.masm.push(Reg64::R10);
        let ba_addr = jit_runtime::jit_runtime_read_current_ba as *const () as usize;
        self.masm.mov_ri(Reg64::R11, ba_addr as i64);
        self.masm.call_reg(Reg64::R11);
        self.masm
            .mov_store_base_disp32(Reg64::Rbp, off_ba, Reg64::Rax);
        self.masm.pop(Reg64::R10);
        self.masm.pop(Reg64::R11);

        // ── Common direct-call path ─────────────────────────────────
        // R11 = entry point, R10 = ctx_ptr.
        // Stack: [RSP] = padding, [RSP+8] = callee.
        self.masm.bind_label(&mut do_direct);

        // Allocate 128-byte register file.
        self.masm.sub_ri(Reg64::Rsp, 128);

        // Zero register file using rep stosq.
        // Save R10, R11 across the operation.
        self.masm.push(Reg64::R10);
        self.masm.push(Reg64::R11);

        self.masm.xor_rr(Reg64::Rax, Reg64::Rax); // RAX = 0
        self.masm.mov_ri(Reg64::Rcx, 16); // RCX = 16 (qwords)
        // RDI points to the register file at [RSP + 16] (after two pushes)
        self.masm.mov_rr(Reg64::Rdi, Reg64::Rsp);
        self.masm.add_ri(Reg64::Rdi, 16);
        self.masm.rep_stosq(); // Zero 128 bytes (~2-3 cycles)

        self.masm.pop(Reg64::R11);
        self.masm.pop(Reg64::R10);

        // RDI = register file, RSI = ctx_ptr.
        self.masm.mov_rr(Reg64::Rdi, Reg64::Rsp);
        self.masm.mov_rr(Reg64::Rsi, Reg64::R10);

        // CALL entry point.
        self.masm.call_reg(Reg64::R11);

        // Deallocate register file.
        self.masm.add_ri(Reg64::Rsp, 128);

        // Finish direct call (restores BA, context, truncates heap).
        self.masm.mov_rr(Reg64::Rdi, Reg64::Rax);
        let finish_addr = jit_runtime::jit_runtime_finish_direct_call as *const () as usize;
        self.masm.mov_ri(Reg64::R11, finish_addr as i64);
        self.masm.call_reg(Reg64::R11);

        // Pop saved callee + padding.
        self.masm.add_ri(Reg64::Rsp, 16);
        self.masm.jmp(&mut done_label);

        // ── Stub fallback (get_jit_entry returned 0) ────────────────
        self.masm.bind_label(&mut stub_fallback);
        self.masm.pop(Reg64::R11); // discard padding
        self.masm.pop(Reg64::Rdi); // callee
        let stub_addr = jit_runtime::jit_runtime_call_undefined_receiver0 as *const () as usize;
        self.masm.mov_ri(Reg64::R11, stub_addr as i64);
        self.masm.call_reg(Reg64::R11);
        self.masm.jmp(&mut done_label);

        // ── Mono prepare failed — fall back to stub ─────────────────
        self.masm.bind_label(&mut mono_fail);
        // Invalidate cache so next call goes through full lookup.
        self.masm.xor_rr(Reg64::R11, Reg64::R11);
        self.masm
            .mov_store_base_disp32(Reg64::Rbp, off_callee, Reg64::R11);
        self.masm.pop(Reg64::R11); // discard padding
        self.masm.pop(Reg64::Rdi); // callee
        self.masm.mov_ri(
            Reg64::R11,
            jit_runtime::jit_runtime_call_undefined_receiver0 as *const () as usize as i64,
        );
        self.masm.call_reg(Reg64::R11);

        // ── Common exit ─────────────────────────────────────────────
        self.masm.bind_label(&mut done_label);
        self.emit_restore_live_regs(saved);
        self.emit_deopt_check_rax();
        self.emit_store(id, Reg64::Rax);
    }

    /// Emit a direct-call fast path for `CallUndefinedReceiver1`.
    #[cfg(all(target_arch = "x86_64", unix))]
    fn emit_direct_call_1(&mut self, id: NodeId, callee: NodeId, arg0: NodeId) {
        let saved = self.emit_save_live_regs(id);

        // Load callee and arg0 into registers, then save to stack.
        self.emit_load(callee, Reg64::Rdi);
        self.emit_load(arg0, Reg64::Rsi);
        // Push arg0, callee (2 pushes → RSP ≡ 0 mod 16).
        self.masm.push(Reg64::Rsi); // arg0
        self.masm.push(Reg64::Rdi); // callee

        // Call jit_runtime_get_jit_entry(callee_i64).
        // RDI already has callee.
        let get_entry_addr = jit_runtime::jit_runtime_get_jit_entry as *const () as usize;
        self.masm.mov_ri(Reg64::R11, get_entry_addr as i64);
        self.masm.call_reg(Reg64::R11);

        let mut slow_label = Label::new();
        let mut done_label = Label::new();
        self.masm.test_rr(Reg64::Rax, Reg64::Rax);
        self.masm.je(&mut slow_label);

        // ── Fast path ───────────────────────────────────────────────
        self.masm.mov_rr(Reg64::R11, Reg64::Rax); // entry point
        self.masm.mov_rr(Reg64::R10, Reg64::Rdx); // ctx_ptr

        // Allocate register file (128 bytes).
        self.masm.sub_ri(Reg64::Rsp, 128);

        // Zero register file using rep stosq.
        self.masm.push(Reg64::R10);
        self.masm.push(Reg64::R11);
        self.masm.xor_rr(Reg64::Rax, Reg64::Rax); // RAX = 0
        self.masm.mov_ri(Reg64::Rcx, 16); // RCX = 16 (qwords)
        // RDI points to the register file at [RSP + 16] (after two pushes)
        self.masm.mov_rr(Reg64::Rdi, Reg64::Rsp);
        self.masm.add_ri(Reg64::Rdi, 16);
        self.masm.rep_stosq(); // Zero 128 bytes (~2-3 cycles)
        self.masm.pop(Reg64::R11);
        self.masm.pop(Reg64::R10);

        // Store arg0 into register file slot 0.
        // arg0 is on the stack at RSP + 128 + 8 (callee at +128, arg0 at +136).
        self.masm
            .mov_load_base_disp32(Reg64::Rax, Reg64::Rsp, 128 + 8);
        self.masm.mov_store_base_disp32(Reg64::Rsp, 0, Reg64::Rax);

        // Call entry point.
        self.masm.mov_rr(Reg64::Rdi, Reg64::Rsp);
        self.masm.mov_rr(Reg64::Rsi, Reg64::R10);
        self.masm.call_reg(Reg64::R11);

        // Deallocate register file.
        self.masm.add_ri(Reg64::Rsp, 128);

        // Finish direct call.
        self.masm.mov_rr(Reg64::Rdi, Reg64::Rax);
        let finish_addr = jit_runtime::jit_runtime_finish_direct_call as *const () as usize;
        self.masm.mov_ri(Reg64::R11, finish_addr as i64);
        self.masm.call_reg(Reg64::R11);

        // Pop saved values.
        self.masm.add_ri(Reg64::Rsp, 16);
        self.masm.jmp(&mut done_label);

        // ── Slow path ───────────────────────────────────────────────
        self.masm.bind_label(&mut slow_label);
        self.masm.pop(Reg64::Rdi); // callee
        self.masm.pop(Reg64::Rsi); // arg0
        let stub_addr = jit_runtime::jit_runtime_call_undefined_receiver1 as *const () as usize;
        self.masm.mov_ri(Reg64::R11, stub_addr as i64);
        self.masm.call_reg(Reg64::R11);

        // ── Common exit ─────────────────────────────────────────────
        self.masm.bind_label(&mut done_label);
        self.emit_restore_live_regs(saved);
        self.emit_deopt_check_rax();
        self.emit_store(id, Reg64::Rax);
    }

    /// Emit a direct-call fast path for `CallUndefinedReceiver2`.
    #[cfg(all(target_arch = "x86_64", unix))]
    fn emit_direct_call_2(&mut self, id: NodeId, callee: NodeId, arg0: NodeId, arg1: NodeId) {
        let saved = self.emit_save_live_regs(id);

        // Load all three values before pushing (avoids clobbering issues).
        self.emit_load(callee, Reg64::Rdi);
        self.emit_load(arg0, Reg64::Rsi);
        self.emit_load(arg1, Reg64::Rdx);
        // 4 pushes → RSP ≡ 0 mod 16 (even count, but we started at
        // 0 mod 16, so 4×8 = 32 → still 0 mod 16).
        self.masm.push(Reg64::Rdx); // arg1
        self.masm.push(Reg64::Rsi); // arg0
        self.masm.push(Reg64::Rdi); // callee
        self.masm.push(Reg64::Rdi); // padding

        // Call jit_runtime_get_jit_entry(callee_i64).
        let get_entry_addr = jit_runtime::jit_runtime_get_jit_entry as *const () as usize;
        self.masm.mov_ri(Reg64::R11, get_entry_addr as i64);
        self.masm.call_reg(Reg64::R11);

        let mut slow_label = Label::new();
        let mut done_label = Label::new();
        self.masm.test_rr(Reg64::Rax, Reg64::Rax);
        self.masm.je(&mut slow_label);

        // ── Fast path ───────────────────────────────────────────────
        self.masm.mov_rr(Reg64::R11, Reg64::Rax); // entry
        self.masm.mov_rr(Reg64::R10, Reg64::Rdx); // ctx

        self.masm.sub_ri(Reg64::Rsp, 128);

        // Zero register file using rep stosq.
        self.masm.push(Reg64::R10);
        self.masm.push(Reg64::R11);
        self.masm.xor_rr(Reg64::Rax, Reg64::Rax); // RAX = 0
        self.masm.mov_ri(Reg64::Rcx, 16); // RCX = 16 (qwords)
        // RDI points to the register file at [RSP + 16] (after two pushes)
        self.masm.mov_rr(Reg64::Rdi, Reg64::Rsp);
        self.masm.add_ri(Reg64::Rdi, 16);
        self.masm.rep_stosq(); // Zero 128 bytes (~2-3 cycles)
        self.masm.pop(Reg64::R11);
        self.masm.pop(Reg64::R10);

        // Store args into register file.
        // Stack after sub 128: [regfile(128), padding(8), callee(8), arg0(8), arg1(8)]
        // arg0 at RSP + 128 + 16, arg1 at RSP + 128 + 24.
        self.masm
            .mov_load_base_disp32(Reg64::Rax, Reg64::Rsp, 128 + 16);
        self.masm.mov_store_base_disp32(Reg64::Rsp, 0, Reg64::Rax); // slot 0 = arg0
        self.masm
            .mov_load_base_disp32(Reg64::Rax, Reg64::Rsp, 128 + 24);
        self.masm.mov_store_base_disp32(Reg64::Rsp, 8, Reg64::Rax); // slot 1 = arg1

        self.masm.mov_rr(Reg64::Rdi, Reg64::Rsp);
        self.masm.mov_rr(Reg64::Rsi, Reg64::R10);
        self.masm.call_reg(Reg64::R11);

        self.masm.add_ri(Reg64::Rsp, 128);

        self.masm.mov_rr(Reg64::Rdi, Reg64::Rax);
        let finish_addr = jit_runtime::jit_runtime_finish_direct_call as *const () as usize;
        self.masm.mov_ri(Reg64::R11, finish_addr as i64);
        self.masm.call_reg(Reg64::R11);

        self.masm.add_ri(Reg64::Rsp, 32); // pop 4 saved values
        self.masm.jmp(&mut done_label);

        // ── Slow path ───────────────────────────────────────────────
        self.masm.bind_label(&mut slow_label);
        self.masm.pop(Reg64::R11); // padding
        self.masm.pop(Reg64::Rdi); // callee
        self.masm.pop(Reg64::Rsi); // arg0
        self.masm.pop(Reg64::Rdx); // arg1
        let stub_addr = jit_runtime::jit_runtime_call_undefined_receiver2 as *const () as usize;
        self.masm.mov_ri(Reg64::R11, stub_addr as i64);
        self.masm.call_reg(Reg64::R11);

        // ── Common exit ─────────────────────────────────────────────
        self.masm.bind_label(&mut done_label);
        self.emit_restore_live_regs(saved);
        self.emit_deopt_check_rax();
        self.emit_store(id, Reg64::Rax);
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

    /// Emit a 32-bit integer comparison, writing `JIT_FALSE` or `JIT_TRUE`
    /// to `result`'s allocated location.
    fn emit_int32_compare(&mut self, left: NodeId, right: NodeId, cc: CondCode, result: NodeId) {
        self.emit_cmp32(left, right);
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
    /// When `if_true` is the very next block in emission order, the false
    /// (unlikely) path is deferred out-of-line so the hot path can fall
    /// through without an extra JMP — saving one instruction per loop
    /// iteration.
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

        // Re-emit 32-bit CMP from the comparison's operands, using physical
        // registers directly when available and CMP-immediate for constants.
        self.emit_cmp32(left, right);

        let if_true_idx = if_true as usize;
        let if_false_idx = if_false as usize;
        let negated = Self::negate_cc(cc);

        // Optimisation: when the true target is the next emitted block, defer
        // the false (cold) path out-of-line so the hot path falls through
        // without an unconditional JMP.
        let next_block = block_idx + 1;
        let num_blocks = self.graph.blocks().len() as u32;
        if if_true == next_block && next_block < num_blocks {
            let mut cold_label = Label::new();
            self.masm.jcc(negated, &mut cold_label);
            self.emit_phi_copies_for_successor(block_idx, if_true);
            // Fall through to the next block (if_true) — no JMP needed.
            self.deferred_branches.push(DeferredBranch {
                label: cold_label,
                pred_idx: block_idx,
                successor_idx: if_false,
                target_block: if_false,
            });
        } else {
            let mut false_path = Label::new();
            self.masm.jcc(negated, &mut false_path);
            self.emit_phi_copies_for_successor(block_idx, if_true);
            self.masm.jmp(&mut self.block_labels[if_true_idx]);
            self.masm.bind_label(&mut false_path);
            self.emit_phi_copies_for_successor(block_idx, if_false);
            self.masm.jmp(&mut self.block_labels[if_false_idx]);
        }
        true
    }

    /// Try to convert a back-edge `Jump` into a conditional branch that
    /// skips the loop header on subsequent iterations.
    ///
    /// When a body block jumps to a loop header whose control is a fused
    /// compare-branch `(CMP + Jcc)`, we duplicate the comparison at the
    /// back-edge and emit a conditional jump directly to the body's first
    /// block — eliminating the unconditional JMP per iteration.
    ///
    /// Returns `true` if the back-edge was rotated (caller must NOT emit
    /// the unconditional JMP).
    fn try_rotate_back_edge(&mut self, body_end_idx: u32, header_idx: u32) -> bool {
        // Only for loop headers.
        let header = match self.graph.block(header_idx) {
            Some(b) if b.is_loop_header => b,
            _ => return false,
        };

        // The header must end with a Branch.
        let (condition, if_true, if_false) = match &header.control {
            Some(ControlNode::Branch {
                condition,
                if_true,
                if_false,
            }) => (*condition, *if_true, *if_false),
            _ => return false,
        };

        // The condition must be a fusible Int32 comparison.
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

        // Don't rotate if the body end is also the header's true target
        // (single-block loop with no separate body).
        if body_end_idx == if_true {
            return false;
        }

        // Emit the duplicated comparison + conditional back-edge to body.
        self.emit_cmp32(left, right);
        let body_target = if_true as usize;
        self.masm.jcc(cc, &mut self.block_labels[body_target]);

        // Fall through to exit.  If exit is the next block, no JMP needed;
        // otherwise emit an unconditional JMP.
        let next_block = body_end_idx + 1;
        let num_blocks = self.graph.blocks().len() as u32;
        if if_false != next_block || next_block >= num_blocks {
            let exit_target = if_false as usize;
            self.masm.jmp(&mut self.block_labels[exit_target]);
        }
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
            // Overflow is never used in compare-branch fusion.
            CondCode::Overflow => CondCode::Overflow,
        }
    }

    // ── Deopt helpers ────────────────────────────────────────────────────────

    /// Emit a deopt on 64-bit overflow (OF flag) from a preceding ADD/SUB/IMUL.
    ///
    /// After a 64-bit arithmetic instruction the CPU's OF flag is set when the
    /// signed result overflows the i64 range.  This emits a single `JO`
    /// (2 bytes + 4-byte displacement).  For practical JS values this never
    /// fires — it guards against pathological i64 wraparound only.
    fn emit_deopt_on_i64_overflow(&mut self, bytecode_offset: u32) {
        let code_off = self.masm.position() as u32;
        let liveness_map = self.all_slots_live();
        self.deopt_entries.push(DeoptEntry {
            code_offset: code_off,
            bytecode_offset,
            liveness_map,
        });
        // JO deopt_overflow_label — jump if OF=1 (i64 overflow).
        self.masm
            .jcc(CondCode::Overflow, &mut self.deopt_overflow_label);
    }

    /// Emit a Smi-overflow guard after an arithmetic operation whose result is
    /// in `R11`.  Delegates to [`emit_deopt_on_smi_overflow_in`].
    fn emit_deopt_on_smi_overflow(&mut self, bytecode_offset: u32) {
        self.emit_deopt_on_smi_overflow_in(Reg64::R11, bytecode_offset);
    }

    /// Emit an i32 overflow guard on an arbitrary register.
    ///
    /// Uses `MOVSXD R10, src_d` + `CMP R10, src` + `JNE deopt`.
    /// `R10` is always the scratch for the sign-extended copy.
    fn emit_deopt_on_smi_overflow_in(&mut self, src: Reg64, bytecode_offset: u32) {
        // MOVSXD R10, src_d — sign-extend lower 32 bits into R10.
        self.masm.movsxd_rr(Reg64::R10, src);
        // CMP R10, src — if sign-extended version differs, overflow.
        self.masm.cmp_rr(Reg64::R10, src);

        let code_off = self.masm.position() as u32;
        let liveness_map = self.all_slots_live();
        self.deopt_entries.push(DeoptEntry {
            code_offset: code_off,
            bytecode_offset,
            liveness_map,
        });
        // JNE deopt_overflow_label — jump to overflow deopt if R10 != src.
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

    // ── MOVSXD elimination helpers ──────────────────────────────────────────

    /// Returns `true` when `node` is a wrapping 32-bit operation that emits a
    /// `movsxd_sign_extend` in the default codegen path.
    fn is_wrapping_int32_producer(node: &ValueNode) -> bool {
        matches!(
            node,
            ValueNode::Int32Add { .. }
                | ValueNode::Int32Subtract { .. }
                | ValueNode::Int32Multiply { .. }
                | ValueNode::Int32Negate { .. }
                | ValueNode::Int32Increment { .. }
                | ValueNode::Int32Decrement { .. }
        )
    }

    /// Returns `true` when `node` is a `CheckedSmi*` operation whose inputs
    /// are all provably within the i32 range.  Such operations can be emitted
    /// with 32-bit ALU instructions + `JO` and therefore produce a 32-bit
    /// (zero-extended) result — making them candidates for MOVSXD elision.
    fn is_i32_range_checked_producer(node: &ValueNode, i32_range: &HashSet<NodeId>) -> bool {
        match node {
            ValueNode::CheckedSmiAdd { left, right }
            | ValueNode::CheckedSmiSubtract { left, right }
            | ValueNode::CheckedSmiMultiply { left, right } => {
                i32_range.contains(left) && i32_range.contains(right)
            }
            ValueNode::CheckedSmiIncrement { value } | ValueNode::CheckedSmiDecrement { value } => {
                i32_range.contains(value)
            }
            _ => false,
        }
    }

    /// Returns `true` when `node` only reads the lower 32 bits of its
    /// `NodeId` operands, meaning upstream producers can skip sign-extension.
    ///
    /// Only wrapping Int32 arithmetic and Int32 comparisons qualify.  Other
    /// Int32 operations (divide, modulus, bitwise, shifts) are deliberately
    /// excluded because they use 64-bit ALU instructions (`REX.W`); garbage
    /// upper bits in their operands would produce incorrect results.
    fn is_narrow_int32_consumer(node: &ValueNode) -> bool {
        matches!(
            node,
            // Wrapping Int32 arithmetic — uses 32-bit ALU ops.
            ValueNode::Int32Add { .. }
                | ValueNode::Int32Subtract { .. }
                | ValueNode::Int32Multiply { .. }
                | ValueNode::Int32Negate { .. }
                | ValueNode::Int32Increment { .. }
                | ValueNode::Int32Decrement { .. }
                // Int32 comparisons — uses 32-bit CMP.
                | ValueNode::Int32Equal { .. }
                | ValueNode::Int32StrictEqual { .. }
                | ValueNode::Int32LessThan { .. }
                | ValueNode::Int32LessThanOrEqual { .. }
                | ValueNode::Int32GreaterThan { .. }
                | ValueNode::Int32GreaterThanOrEqual { .. }
        )
    }

    /// Collect all `NodeId` operands referenced by `node` into `out`.
    fn collect_node_inputs(node: &ValueNode, out: &mut HashSet<NodeId>) {
        match node {
            // ── No NodeId inputs ────────────────────────────────────────
            ValueNode::SmiConstant { .. }
            | ValueNode::Float64Constant { .. }
            | ValueNode::Int32Constant { .. }
            | ValueNode::Uint32Constant { .. }
            | ValueNode::BigIntConstant { .. }
            | ValueNode::TrueConstant
            | ValueNode::FalseConstant
            | ValueNode::NullConstant
            | ValueNode::UndefinedConstant
            | ValueNode::RootConstant { .. }
            | ValueNode::ExternalConstant { .. }
            | ValueNode::StringConstant { .. }
            | ValueNode::ConstantPoolEntry { .. }
            | ValueNode::Parameter { .. }
            | ValueNode::RegisterInput { .. }
            | ValueNode::ArgumentsLength
            | ValueNode::RestLength
            | ValueNode::CreateObjectLiteral { .. }
            | ValueNode::CreateArrayLiteral { .. }
            | ValueNode::CreateShallowObjectLiteral { .. }
            | ValueNode::CreateShallowArrayLiteral { .. }
            | ValueNode::CreateFunctionContext { .. }
            | ValueNode::CreateBlockContext { .. }
            | ValueNode::CreateClosure { .. }
            | ValueNode::FastCreateClosure { .. }
            | ValueNode::CreateEmptyObjectLiteral
            | ValueNode::CreateRegExpLiteral { .. }
            | ValueNode::ArgumentsElements { .. }
            | ValueNode::RestElements { .. }
            | ValueNode::VirtualObject { .. }
            | ValueNode::GetTemplateObject { .. }
            | ValueNode::LoadGlobal { .. }
            | ValueNode::LoadCurrentContextSlot { .. }
            | ValueNode::Debugger
            | ValueNode::Abort { .. } => {}

            // ── Single-input nodes (field name: value) ──────────────────
            ValueNode::Int32Negate { value }
            | ValueNode::Int32Increment { value }
            | ValueNode::Int32Decrement { value }
            | ValueNode::Float64Negate { value }
            | ValueNode::TestUndetectable { value }
            | ValueNode::ToBoolean { value }
            | ValueNode::TypeOf { value }
            | ValueNode::CheckedSmiIncrement { value }
            | ValueNode::CheckedSmiDecrement { value } => {
                out.insert(*value);
            }

            ValueNode::Float64Ieee754Unary { value, .. }
            | ValueNode::GenericBitwiseNot { value, .. }
            | ValueNode::GenericNegate { value, .. }
            | ValueNode::GenericIncrement { value, .. }
            | ValueNode::GenericDecrement { value, .. }
            | ValueNode::ToString { value, .. }
            | ValueNode::ToObject { value, .. }
            | ValueNode::ToName { value, .. }
            | ValueNode::ToNumber { value, .. }
            | ValueNode::ToNumberOrNumeric { value, .. }
            | ValueNode::TestTypeOf { value, .. }
            | ValueNode::NumberToString { value, .. }
            | ValueNode::StoreGlobal { value, .. }
            | ValueNode::StoreCurrentContextSlot { value, .. } => {
                out.insert(*value);
            }

            // ── Single-input nodes (field name: input) ──────────────────
            ValueNode::ChangeInt32ToFloat64 { input }
            | ValueNode::ChangeUint32ToFloat64 { input }
            | ValueNode::ChangeFloat64ToInt32 { input }
            | ValueNode::CheckedFloat64ToInt32 { input }
            | ValueNode::ChangeInt32ToTagged { input }
            | ValueNode::ChangeUint32ToTagged { input }
            | ValueNode::ChangeFloat64ToTagged { input }
            | ValueNode::ChangeTaggedToInt32 { input }
            | ValueNode::ChangeTaggedToUint32 { input }
            | ValueNode::ChangeTaggedToFloat64 { input }
            | ValueNode::CheckedTaggedToInt32 { input }
            | ValueNode::CheckedTaggedToFloat64 { input }
            | ValueNode::CheckInt32IsSmi { input }
            | ValueNode::CheckUint32IsSmi { input }
            | ValueNode::CheckHoleyFloat64IsSmi { input }
            | ValueNode::CheckFloat64IsNan { input } => {
                out.insert(*input);
            }

            // ── Single-input nodes (field name: receiver) ───────────────
            ValueNode::CheckSmi { receiver }
            | ValueNode::CheckNumber { receiver }
            | ValueNode::CheckHeapObject { receiver }
            | ValueNode::CheckSymbol { receiver }
            | ValueNode::CheckString { receiver }
            | ValueNode::CheckStringOrStringWrapper { receiver }
            | ValueNode::CheckSeqOneByteString { receiver } => {
                out.insert(*receiver);
            }
            ValueNode::CheckMaps { receiver, .. }
            | ValueNode::CheckMapsWithMigration { receiver, .. }
            | ValueNode::CheckValue { receiver, .. } => {
                out.insert(*receiver);
            }

            // ── Single-input nodes (other field names) ──────────────────
            ValueNode::GetArgument { index } => {
                out.insert(*index);
            }
            ValueNode::LoadField { object, .. }
            | ValueNode::LoadTaggedField { object, .. }
            | ValueNode::LoadDoubleField { object, .. }
            | ValueNode::LoadNamedGeneric { object, .. } => {
                out.insert(*object);
            }
            ValueNode::LoadContextSlot { context, .. } => {
                out.insert(*context);
            }
            ValueNode::ForInPrepare { enumerator, .. } => {
                out.insert(*enumerator);
            }
            ValueNode::LoadEnumCacheLength { map } => {
                out.insert(*map);
            }
            ValueNode::StringLength { string } => {
                out.insert(*string);
            }
            ValueNode::CreateCatchContext { exception, .. } => {
                out.insert(*exception);
            }
            ValueNode::CreateWithContext { object, .. } => {
                out.insert(*object);
            }
            ValueNode::PushContext { context } | ValueNode::PopContext { context } => {
                out.insert(*context);
            }

            // ── Two-input nodes (left, right) ───────────────────────────
            ValueNode::Int32Add { left, right }
            | ValueNode::Int32Subtract { left, right }
            | ValueNode::Int32Multiply { left, right }
            | ValueNode::Int32Divide { left, right }
            | ValueNode::Int32Modulus { left, right }
            | ValueNode::Int32BitwiseAnd { left, right }
            | ValueNode::Int32BitwiseOr { left, right }
            | ValueNode::Int32BitwiseXor { left, right }
            | ValueNode::Int32ShiftLeft { left, right }
            | ValueNode::Int32ShiftRight { left, right }
            | ValueNode::Int32ShiftRightLogical { left, right }
            | ValueNode::Int32Equal { left, right }
            | ValueNode::Int32StrictEqual { left, right }
            | ValueNode::Int32LessThan { left, right }
            | ValueNode::Int32LessThanOrEqual { left, right }
            | ValueNode::Int32GreaterThan { left, right }
            | ValueNode::Int32GreaterThanOrEqual { left, right }
            | ValueNode::Uint32Add { left, right }
            | ValueNode::Uint32Subtract { left, right }
            | ValueNode::Uint32Multiply { left, right }
            | ValueNode::Uint32Divide { left, right }
            | ValueNode::Uint32Modulus { left, right }
            | ValueNode::Float64Add { left, right }
            | ValueNode::Float64Subtract { left, right }
            | ValueNode::Float64Multiply { left, right }
            | ValueNode::Float64Divide { left, right }
            | ValueNode::Float64Modulus { left, right }
            | ValueNode::Float64Exponentiate { left, right }
            | ValueNode::Float64Equal { left, right }
            | ValueNode::Float64LessThan { left, right }
            | ValueNode::Float64LessThanOrEqual { left, right }
            | ValueNode::Float64GreaterThan { left, right }
            | ValueNode::Float64GreaterThanOrEqual { left, right }
            | ValueNode::CheckedSmiAdd { left, right }
            | ValueNode::CheckedSmiSubtract { left, right }
            | ValueNode::CheckedSmiMultiply { left, right }
            | ValueNode::CheckedSmiDivide { left, right }
            | ValueNode::CheckedSmiModulus { left, right }
            | ValueNode::StringConcat { left, right }
            | ValueNode::StringEqual { left, right } => {
                out.insert(*left);
                out.insert(*right);
            }

            ValueNode::GenericAdd { left, right, .. }
            | ValueNode::GenericSubtract { left, right, .. }
            | ValueNode::GenericMultiply { left, right, .. }
            | ValueNode::GenericDivide { left, right, .. }
            | ValueNode::GenericModulus { left, right, .. }
            | ValueNode::GenericExponentiate { left, right, .. }
            | ValueNode::GenericBitwiseAnd { left, right, .. }
            | ValueNode::GenericBitwiseOr { left, right, .. }
            | ValueNode::GenericBitwiseXor { left, right, .. }
            | ValueNode::GenericShiftLeft { left, right, .. }
            | ValueNode::GenericShiftRight { left, right, .. }
            | ValueNode::GenericShiftRightLogical { left, right, .. }
            | ValueNode::TaggedEqual { left, right, .. }
            | ValueNode::TaggedNotEqual { left, right, .. } => {
                out.insert(*left);
                out.insert(*right);
            }

            // ── Two-input nodes (various field names) ───────────────────
            ValueNode::CheckDynamicValue { receiver, expected } => {
                out.insert(*receiver);
                out.insert(*expected);
            }
            ValueNode::CheckInt32Condition { left, right, .. } => {
                out.insert(*left);
                out.insert(*right);
            }
            ValueNode::CheckCacheIndicesNotCleared { receiver, indices } => {
                out.insert(*receiver);
                out.insert(*indices);
            }
            ValueNode::StoreField { object, value, .. } => {
                out.insert(*object);
                out.insert(*value);
            }
            ValueNode::StoreNamedGeneric { object, value, .. } => {
                out.insert(*object);
                out.insert(*value);
            }
            ValueNode::StoreContextSlot { context, value, .. } => {
                out.insert(*context);
                out.insert(*value);
            }
            ValueNode::LoadFixedArrayElement { elements, index }
            | ValueNode::LoadFixedDoubleArrayElement { elements, index }
            | ValueNode::LoadHoleyFixedDoubleArrayElement { elements, index } => {
                out.insert(*elements);
                out.insert(*index);
            }
            ValueNode::StringAt { string, index } => {
                out.insert(*string);
                out.insert(*index);
            }
            ValueNode::HasInPrototypeChain { object, prototype } => {
                out.insert(*object);
                out.insert(*prototype);
            }
            ValueNode::DeleteProperty { object, key, .. } => {
                out.insert(*object);
                out.insert(*key);
            }
            ValueNode::TestInstanceOf {
                object, callable, ..
            } => {
                out.insert(*object);
                out.insert(*callable);
            }
            ValueNode::TestIn { key, object, .. } => {
                out.insert(*key);
                out.insert(*object);
            }
            ValueNode::LoadKeyedGeneric { object, key, .. } => {
                out.insert(*object);
                out.insert(*key);
            }

            // ── Three-input nodes ───────────────────────────────────────
            ValueNode::StoreFixedArrayElement {
                elements,
                index,
                value,
            }
            | ValueNode::StoreFixedDoubleArrayElement {
                elements,
                index,
                value,
            } => {
                out.insert(*elements);
                out.insert(*index);
                out.insert(*value);
            }
            ValueNode::StoreKeyedGeneric {
                object, key, value, ..
            } => {
                out.insert(*object);
                out.insert(*key);
                out.insert(*value);
            }
            ValueNode::ForInNext {
                receiver,
                cache_index,
                cache_array,
                ..
            } => {
                out.insert(*receiver);
                out.insert(*cache_index);
                out.insert(*cache_array);
            }

            // ── Variable-input nodes ────────────────────────────────────
            ValueNode::Call {
                callee,
                receiver,
                args,
                ..
            }
            | ValueNode::CallWithSpread {
                callee,
                receiver,
                args,
                ..
            } => {
                out.insert(*callee);
                out.insert(*receiver);
                for a in args {
                    out.insert(*a);
                }
            }
            ValueNode::CallKnownFunction {
                callee,
                receiver,
                args,
            } => {
                out.insert(*callee);
                out.insert(*receiver);
                for a in args {
                    out.insert(*a);
                }
            }
            ValueNode::CallBuiltin { args, .. } | ValueNode::CallRuntime { args, .. } => {
                for a in args {
                    out.insert(*a);
                }
            }
            ValueNode::Construct {
                constructor, args, ..
            }
            | ValueNode::ConstructWithSpread {
                constructor, args, ..
            } => {
                out.insert(*constructor);
                for a in args {
                    out.insert(*a);
                }
            }
            ValueNode::Phi { inputs } => {
                for i in inputs {
                    out.insert(*i);
                }
            }
        }
    }

    /// Compute the set of nodes whose values are provably within the i32
    /// range.  This is a forward (producer-driven) analysis: a node is
    /// i32-range if it produces a value that fits in `[-2^31, 2^31)` on
    /// every non-deopt path.
    ///
    /// **Intrinsically i32-range producers:**
    ///   - `SmiConstant` (its `value` field is `i32`)
    ///   - `Int32Constant`
    ///   - All wrapping Int32 arithmetic / bitwise / shift operations (they
    ///     operate on 32 bits by definition)
    ///
    /// **Derived i32-range nodes (fixed-point iteration):**
    ///   - `Phi` whose *all* inputs are i32-range
    ///   - `CheckedSmiAdd / Sub / Mul` where *both* operands are i32-range
    ///     (the result either fits in i32 or the operation deopts)
    ///   - `CheckedSmiIncrement / Decrement` where the input is i32-range
    fn compute_i32_range(graph: &MaglevGraph) -> HashSet<NodeId> {
        let mut i32_range: HashSet<NodeId> = HashSet::new();

        // Phase 1: mark intrinsically i32-range producers.
        for block in graph.blocks() {
            for (id, node) in &block.nodes {
                let is_i32 = matches!(
                    node,
                    ValueNode::SmiConstant { .. }
                        | ValueNode::Int32Constant { .. }
                        | ValueNode::Int32Add { .. }
                        | ValueNode::Int32Subtract { .. }
                        | ValueNode::Int32Multiply { .. }
                        | ValueNode::Int32Negate { .. }
                        | ValueNode::Int32Increment { .. }
                        | ValueNode::Int32Decrement { .. }
                        | ValueNode::Int32BitwiseAnd { .. }
                        | ValueNode::Int32BitwiseOr { .. }
                        | ValueNode::Int32BitwiseXor { .. }
                        | ValueNode::Int32ShiftLeft { .. }
                        | ValueNode::Int32ShiftRight { .. }
                        | ValueNode::Int32ShiftRightLogical { .. }
                        | ValueNode::Int32Divide { .. }
                        | ValueNode::Int32Modulus { .. }
                );
                if is_i32 {
                    i32_range.insert(*id);
                }
            }
        }

        // Phase 2: fixed-point iteration for Phi and CheckedSmi nodes.
        loop {
            let mut changed = false;
            for block in graph.blocks() {
                for (id, node) in &block.nodes {
                    if i32_range.contains(id) {
                        continue;
                    }
                    let derived = match node {
                        ValueNode::Phi { inputs } => {
                            inputs.iter().all(|inp| i32_range.contains(inp))
                        }
                        ValueNode::CheckedSmiAdd { left, right }
                        | ValueNode::CheckedSmiSubtract { left, right }
                        | ValueNode::CheckedSmiMultiply { left, right } => {
                            i32_range.contains(left) && i32_range.contains(right)
                        }
                        ValueNode::CheckedSmiIncrement { value }
                        | ValueNode::CheckedSmiDecrement { value } => i32_range.contains(value),
                        _ => false,
                    };
                    if derived {
                        i32_range.insert(*id);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        i32_range
    }

    /// Precompute the set of wrapping Int32 nodes whose consumers all operate
    /// on 32-bit values, allowing the post-operation `movsxd` to be elided.
    ///
    /// The analysis is deliberately conservative: candidates are excluded when
    /// they feed into a non-32-bit consumer such as `StoreGlobal`, `Return`,
    /// or any type check.  Spilled values are also excluded because
    /// spill/reload uses 64-bit MOVs and the upper 32 bits must be
    /// well-defined.
    ///
    /// **Phi-aware narrowing**: a `Phi` node whose consumers are *all* narrow
    /// (32-bit ALU / CMP) or other narrow-transparent Phis is treated as
    /// transparent — it merely copies 64 bits between registers/slots, so
    /// garbage upper bits in its inputs are invisible to the downstream
    /// 32-bit consumers.  This lets loop-carried values (e.g. the `i++`
    /// increment feeding a loop-counter Phi) skip `MOVSXD`.
    ///
    /// **i32-range extension**: `CheckedSmi*` operations whose inputs are all
    /// provably within the i32 range (provided by [`compute_i32_range`]) are
    /// also treated as candidates.  When emitted as 32-bit ALU + `JO`, their
    /// upper 32 bits become zero-extended — downstream narrow consumers
    /// only read the lower 32 bits, so the `MOVSXD` can be elided.
    fn compute_narrow_int32(
        graph: &MaglevGraph,
        alloc: &AllocationResult,
        i32_range: &HashSet<NodeId>,
    ) -> HashSet<NodeId> {
        // Step 1: collect all wrapping Int32 producer IDs AND i32-range
        // CheckedSmi operations (they produce 32-bit results when narrowed).
        let mut candidates: HashSet<NodeId> = HashSet::new();
        for block in graph.blocks() {
            for (id, node) in &block.nodes {
                if Self::is_wrapping_int32_producer(node)
                    || Self::is_i32_range_checked_producer(node, i32_range)
                {
                    candidates.insert(*id);
                }
            }
        }

        if candidates.is_empty() {
            return candidates;
        }

        // Step 2: build a reverse-consumer map (value → consuming nodes)
        // and identify Phi nodes.
        let mut consumers_of: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut node_lookup: HashMap<NodeId, &ValueNode> = HashMap::new();
        let mut ctrl_consumed: HashSet<NodeId> = HashSet::new();

        for block in graph.blocks() {
            for (id, node) in &block.nodes {
                node_lookup.insert(*id, node);
                let mut inputs = HashSet::new();
                Self::collect_node_inputs(node, &mut inputs);
                for inp in inputs {
                    consumers_of.entry(inp).or_default().push(*id);
                }
            }
            if let Some(ctrl) = &block.control {
                match ctrl {
                    ControlNode::Return { value } => {
                        ctrl_consumed.insert(*value);
                    }
                    ControlNode::Branch { condition, .. } => {
                        ctrl_consumed.insert(*condition);
                    }
                    ControlNode::Jump { .. } | ControlNode::Deoptimize { .. } => {}
                }
            }
        }

        // Step 3: determine narrow-transparent Phis.
        //
        // Start optimistic (all Phis not consumed by control flow are
        // narrow-transparent) and iterate: if any Phi has a consumer that
        // is neither a narrow consumer nor a narrow-transparent Phi, remove
        // it.  Converges in O(#Phi_chains) iterations (typically 1-2).
        let mut narrow_phi: HashSet<NodeId> = HashSet::new();
        for block in graph.blocks() {
            for (id, node) in &block.nodes {
                if matches!(node, ValueNode::Phi { .. }) && !ctrl_consumed.contains(id) {
                    narrow_phi.insert(*id);
                }
            }
        }

        loop {
            let mut changed = false;
            for phi_id in narrow_phi.clone() {
                let ok = consumers_of.get(&phi_id).is_none_or(|cs| {
                    cs.iter().all(|c| {
                        node_lookup
                            .get(c)
                            .is_some_and(|n| Self::is_narrow_int32_consumer(n))
                            || node_lookup
                                .get(c)
                                .is_some_and(|n| Self::is_i32_range_checked_producer(n, i32_range))
                            || narrow_phi.contains(c)
                            // StoreGlobal is narrow-compatible for Phis:
                            // codegen sign-extends on the cold exit path.
                            || node_lookup
                                .get(c)
                                .is_some_and(|n| matches!(n, ValueNode::StoreGlobal { .. }))
                    })
                });
                if !ok {
                    narrow_phi.remove(&phi_id);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // Step 4: build the non-narrow set.  Narrow consumers, i32-range
        // checked ops, and narrow-transparent Phis are skipped — their
        // inputs can have garbage upper bits without affecting correctness.
        // StoreGlobal whose value is a narrow-transparent Phi is also
        // skipped: codegen sign-extends on the cold exit path.
        let mut non_narrow: HashSet<NodeId> = HashSet::new();
        for block in graph.blocks() {
            for (id, node) in &block.nodes {
                let skip = Self::is_narrow_int32_consumer(node)
                    || Self::is_i32_range_checked_producer(node, i32_range)
                    || narrow_phi.contains(id)
                    || matches!(node, ValueNode::StoreGlobal { value, .. }
                        if narrow_phi.contains(value));
                if !skip {
                    Self::collect_node_inputs(node, &mut non_narrow);
                }
            }
            if let Some(ctrl) = &block.control {
                match ctrl {
                    ControlNode::Return { value } => {
                        non_narrow.insert(*value);
                    }
                    ControlNode::Branch { condition, .. } => {
                        non_narrow.insert(*condition);
                    }
                    ControlNode::Jump { .. } | ControlNode::Deoptimize { .. } => {}
                }
            }
        }

        // Step 5: exclude spilled values — the spill/reload path uses
        // 64-bit MOVs so the upper 32 bits must be well-defined.
        candidates.retain(|id| {
            !non_narrow.contains(id) && !matches!(alloc.location(*id), Some(Location::StackSlot(_)))
        });

        // Step 6: add narrow-transparent Phis to the result set.
        // These Phis pass through narrow values (garbage upper 32 bits)
        // so consumers like StoreGlobal can detect them and sign-extend.
        for id in &narrow_phi {
            if !matches!(alloc.location(*id), Some(Location::StackSlot(_))) {
                candidates.insert(*id);
            }
        }

        candidates
    }

    /// Emit a 32-bit `CMP` between `left` and `right`, using the most
    /// efficient encoding.  Emits the operand-size-32 variant (no REX.W) so
    /// that only the lower 32 bits matter.
    fn emit_cmp32(&mut self, left: NodeId, right: NodeId) {
        if let Some(imm) = self.try_get_i32_constant(right) {
            match self.alloc.location(left) {
                Some(Location::Register(ln)) => {
                    self.masm.cmp32_ri(phys_reg(ln), imm);
                    return;
                }
                _ => {
                    self.emit_load(left, Reg64::R11);
                    self.masm.cmp32_ri(Reg64::R11, imm);
                    return;
                }
            }
        }
        match (self.alloc.location(left), self.alloc.location(right)) {
            (Some(Location::Register(ln)), Some(Location::Register(rn))) => {
                self.masm.cmp32_rr(phys_reg(ln), phys_reg(rn));
            }
            (Some(Location::Register(ln)), _) => {
                self.emit_load(right, Reg64::R10);
                self.masm.cmp32_rr(phys_reg(ln), Reg64::R10);
            }
            (_, Some(Location::Register(rn))) => {
                self.emit_load(left, Reg64::R11);
                self.masm.cmp32_rr(Reg64::R11, phys_reg(rn));
            }
            _ => {
                self.emit_load(left, Reg64::R11);
                self.emit_load(right, Reg64::R10);
                self.masm.cmp32_rr(Reg64::R11, Reg64::R10);
            }
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
    fn test_compile_check_smi_is_passthrough() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        let checked = block.push_value(ValueNode::CheckSmi { receiver: p0 });
        block.set_control(ControlNode::Return { value: checked });
        graph.add_block(block);

        let cc = do_compile(&graph, 1);
        // CheckSmi is now a pass-through for i64 Smis — no deopt guard.
        assert!(
            cc.deopt_entries.is_empty(),
            "CheckSmi should not emit a deopt entry (i64 Smis have no i32 constraint)"
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

    // ── Narrow-Int32 analysis tests ─────────────────────────────────────────

    /// Helper: build a graph + run regalloc, then return `compute_narrow_int32`.
    fn narrow_set(graph: &MaglevGraph) -> HashSet<NodeId> {
        let alloc = allocate(graph, NUM_PHYS_REGS);
        let i32_range = MaglevCodegen::compute_i32_range(graph);
        MaglevCodegen::compute_narrow_int32(graph, &alloc, &i32_range)
    }

    #[test]
    fn test_narrow_int32_add_chain_all_narrow() {
        // p0, p1 -> Int32Add(A) -> Int32Add(B) -> Int32Add(C) -> Return
        // A and B are candidates.  C feeds Return (non-narrow), so C is
        // excluded.  A feeds B (Int32Add = narrow consumer), so A stays
        // narrow.  B feeds C (Int32Add = narrow consumer), so B stays
        // narrow.  Only C is excluded because it feeds Return directly.
        let mut graph = MaglevGraph::new(2);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        let p1 = block.push_value(ValueNode::Parameter { index: 1 });
        let a = block.push_value(ValueNode::Int32Add {
            left: p0,
            right: p1,
        });
        let b = block.push_value(ValueNode::Int32Add { left: a, right: p0 });
        let c = block.push_value(ValueNode::Int32Add { left: b, right: p1 });
        block.set_control(ControlNode::Return { value: c });
        graph.add_block(block);

        let set = narrow_set(&graph);
        // C feeds Return -> excluded.
        assert!(!set.contains(&c), "C feeds Return and must not be narrow");
        // A and B only feed Int32Add (narrow consumers), so they stay narrow
        // and skip MOVSXD.
        assert!(
            set.contains(&a),
            "A only feeds narrow consumer B, should be narrow"
        );
        assert!(
            set.contains(&b),
            "B only feeds narrow consumer C, should be narrow"
        );
    }

    #[test]
    fn test_narrow_int32_phi_excludes_producers() {
        // Block 0: p0 -> Int32Add(A) -> Jump(1)
        // Block 1: Phi(A, A) -> Return
        // A feeds a Phi.  The Phi feeds Return (non-narrow), so the Phi is
        // NOT narrow-transparent.  Therefore A must be excluded.
        let mut graph = MaglevGraph::new(1);
        let mut b0 = BasicBlock::new(0);
        let p0 = b0.push_value(ValueNode::Parameter { index: 0 });
        let a = b0.push_value(ValueNode::Int32Add {
            left: p0,
            right: p0,
        });
        b0.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b0);

        let mut b1 = BasicBlock::new(1);
        b1.add_predecessor(0);
        let phi = b1.push_value(ValueNode::Phi { inputs: vec![a, a] });
        b1.set_control(ControlNode::Return { value: phi });
        graph.add_block(b1);

        let set = narrow_set(&graph);
        assert!(
            !set.contains(&a),
            "A feeds Phi→Return and must not be narrow"
        );
    }

    #[test]
    fn test_narrow_int32_phi_transparent_when_all_consumers_narrow() {
        // Block 0: p0, p1 -> Int32Add(A) -> Jump(1)
        // Block 1: Phi(A, A) -> Int32Add(B, p1) -> Return
        // Phi's ONLY consumer is Int32Add(B) which is narrow.
        // So Phi is narrow-transparent, and A (feeding Phi) is narrow.
        // B feeds Return (non-narrow) → B is NOT narrow.
        let mut graph = MaglevGraph::new(2);

        // Create empty blocks and add them to the graph first so that
        // `add_value_node` can assign graph-global unique IDs.
        let mut b0 = BasicBlock::new(0);
        let mut b1 = BasicBlock::new(1);
        b1.add_predecessor(0);
        graph.add_block(b0);
        graph.add_block(b1);

        let p0 = graph
            .add_value_node(0, ValueNode::Parameter { index: 0 })
            .unwrap();
        let p1 = graph
            .add_value_node(0, ValueNode::Parameter { index: 1 })
            .unwrap();
        let a = graph
            .add_value_node(
                0,
                ValueNode::Int32Add {
                    left: p0,
                    right: p1,
                },
            )
            .unwrap();
        graph.blocks_mut()[0].set_control(ControlNode::Jump { target: 1 });

        let phi = graph
            .add_value_node(1, ValueNode::Phi { inputs: vec![a, a] })
            .unwrap();
        let b = graph
            .add_value_node(
                1,
                ValueNode::Int32Add {
                    left: phi,
                    right: p1,
                },
            )
            .unwrap();
        graph.blocks_mut()[1].set_control(ControlNode::Return { value: b });

        let set = narrow_set(&graph);
        assert!(
            set.contains(&a),
            "A feeds narrow-transparent Phi, should be narrow"
        );
        assert!(!set.contains(&b), "B feeds Return and must not be narrow");
    }

    #[test]
    fn test_narrow_int32_only_narrow_consumers_is_narrow() {
        // p0, p1 -> Int32Add(A) -> Int32Add(B) -> Int32Equal(C) -> Return
        // A feeds B (narrow), and B feeds C (narrow comparison).
        // C feeds Return (non-narrow) → C is not a candidate (it's a compare).
        // B feeds C (narrow) — no non-narrow consumer of B.
        // A feeds B (narrow) — no non-narrow consumer of A.
        // Both A and B should be narrow.
        let mut graph = MaglevGraph::new(2);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        let p1 = block.push_value(ValueNode::Parameter { index: 1 });
        let a = block.push_value(ValueNode::Int32Add {
            left: p0,
            right: p1,
        });
        let b = block.push_value(ValueNode::Int32Add { left: a, right: p0 });
        let c = block.push_value(ValueNode::Int32Equal { left: b, right: p1 });
        block.set_control(ControlNode::Return { value: c });
        graph.add_block(block);

        let set = narrow_set(&graph);
        // Int32Equal is a narrow consumer of B, and Return consumes the
        // comparison result (not a candidate).  A and B have only narrow
        // consumers.
        assert!(
            set.contains(&a),
            "A should be narrow (only consumed by narrow B)"
        );
        assert!(
            set.contains(&b),
            "B should be narrow (only consumed by narrow Int32Equal)"
        );
    }

    #[test]
    fn test_narrow_int32_non_int32_consumer_excludes() {
        // p0, p1 -> Int32Add(A) -> StoreGlobal
        // A feeds StoreGlobal (non-narrow) → A excluded.
        let mut graph = MaglevGraph::new(2);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        let p1 = block.push_value(ValueNode::Parameter { index: 1 });
        let a = block.push_value(ValueNode::Int32Add {
            left: p0,
            right: p1,
        });
        let _sg = block.push_value(ValueNode::StoreGlobal {
            name: 0,
            value: a,
            feedback_slot: 0,
        });
        let c = block.push_value(ValueNode::UndefinedConstant);
        block.set_control(ControlNode::Return { value: c });
        graph.add_block(block);

        let set = narrow_set(&graph);
        assert!(
            !set.contains(&a),
            "A feeds StoreGlobal and must not be narrow"
        );
    }

    #[test]
    fn test_narrow_int32_empty_for_no_candidates() {
        // A graph with no wrapping Int32 producers returns empty.
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c = block.push_value(ValueNode::SmiConstant { value: 1 });
        block.set_control(ControlNode::Return { value: c });
        graph.add_block(block);

        let set = narrow_set(&graph);
        assert!(set.is_empty());
    }

    // ── i32-range analysis tests ─────────────────────────────────────────────

    /// Helper: build a graph and return the `compute_i32_range` set.
    fn i32_range_set(graph: &MaglevGraph) -> HashSet<NodeId> {
        MaglevCodegen::compute_i32_range(graph)
    }

    #[test]
    fn test_i32_range_analysis() {
        // Test 1: SmiConstant and Int32Constant are i32-range.
        {
            let mut graph = MaglevGraph::new(0);
            let mut block = BasicBlock::new(0);
            let smi = block.push_value(ValueNode::SmiConstant { value: 42 });
            let i32c = block.push_value(ValueNode::Int32Constant { value: -1 });
            block.set_control(ControlNode::Return { value: smi });
            graph.add_block(block);

            let set = i32_range_set(&graph);
            assert!(set.contains(&smi), "SmiConstant must be i32-range");
            assert!(set.contains(&i32c), "Int32Constant must be i32-range");
        }

        // Test 2: Int32Add is intrinsically i32-range.
        {
            let mut graph = MaglevGraph::new(2);
            let mut block = BasicBlock::new(0);
            let p0 = block.push_value(ValueNode::Parameter { index: 0 });
            let p1 = block.push_value(ValueNode::Parameter { index: 1 });
            let add = block.push_value(ValueNode::Int32Add {
                left: p0,
                right: p1,
            });
            block.set_control(ControlNode::Return { value: add });
            graph.add_block(block);

            let set = i32_range_set(&graph);
            assert!(set.contains(&add), "Int32Add must be i32-range");
            assert!(!set.contains(&p0), "Parameter must NOT be i32-range");
        }

        // Test 3: CheckedSmiAdd with i32-range inputs becomes i32-range.
        {
            let mut graph = MaglevGraph::new(0);
            let mut block = BasicBlock::new(0);
            let a = block.push_value(ValueNode::SmiConstant { value: 10 });
            let b = block.push_value(ValueNode::SmiConstant { value: 20 });
            let add = block.push_value(ValueNode::CheckedSmiAdd { left: a, right: b });
            block.set_control(ControlNode::Return { value: add });
            graph.add_block(block);

            let set = i32_range_set(&graph);
            assert!(
                set.contains(&add),
                "CheckedSmiAdd with i32-range inputs must be i32-range"
            );
        }

        // Test 4: CheckedSmiAdd with non-i32-range input is NOT i32-range.
        {
            let mut graph = MaglevGraph::new(2);
            let mut block = BasicBlock::new(0);
            let p0 = block.push_value(ValueNode::Parameter { index: 0 });
            let p1 = block.push_value(ValueNode::Parameter { index: 1 });
            let add = block.push_value(ValueNode::CheckedSmiAdd {
                left: p0,
                right: p1,
            });
            block.set_control(ControlNode::Return { value: add });
            graph.add_block(block);

            let set = i32_range_set(&graph);
            assert!(
                !set.contains(&add),
                "CheckedSmiAdd with Parameter inputs must NOT be i32-range"
            );
        }

        // Test 5: CheckedSmiIncrement with i32-range input is i32-range.
        {
            let mut graph = MaglevGraph::new(0);
            let mut block = BasicBlock::new(0);
            let c = block.push_value(ValueNode::SmiConstant { value: 5 });
            let inc = block.push_value(ValueNode::CheckedSmiIncrement { value: c });
            block.set_control(ControlNode::Return { value: inc });
            graph.add_block(block);

            let set = i32_range_set(&graph);
            assert!(
                set.contains(&inc),
                "CheckedSmiIncrement with i32-range input must be i32-range"
            );
        }

        // Test 6: Phi with all i32-range inputs is i32-range.
        {
            let mut graph = MaglevGraph::new(0);
            let mut b0 = BasicBlock::new(0);
            let a = b0.push_value(ValueNode::SmiConstant { value: 1 });
            b0.set_control(ControlNode::Jump { target: 1 });
            graph.add_block(b0);

            let mut b1 = BasicBlock::new(1);
            b1.add_predecessor(0);
            let phi = b1.push_value(ValueNode::Phi { inputs: vec![a, a] });
            b1.set_control(ControlNode::Return { value: phi });
            graph.add_block(b1);

            let set = i32_range_set(&graph);
            assert!(
                set.contains(&phi),
                "Phi with all i32-range inputs must be i32-range"
            );
        }

        // Test 7: Chained CheckedSmiAdd — result of first feeds second.
        // Both should be i32-range because inputs propagate.
        {
            let mut graph = MaglevGraph::new(0);
            let mut block = BasicBlock::new(0);
            let a = block.push_value(ValueNode::SmiConstant { value: 3 });
            let b = block.push_value(ValueNode::SmiConstant { value: 7 });
            let sum1 = block.push_value(ValueNode::CheckedSmiAdd { left: a, right: b });
            let sum2 = block.push_value(ValueNode::CheckedSmiAdd {
                left: sum1,
                right: a,
            });
            block.set_control(ControlNode::Return { value: sum2 });
            graph.add_block(block);

            let set = i32_range_set(&graph);
            assert!(set.contains(&sum1), "sum1 must be i32-range");
            assert!(set.contains(&sum2), "sum2 (chained) must be i32-range");
        }

        // Test 8: i32-range CheckedSmiAdd feeding narrow consumer becomes
        // a narrow candidate (MOVSXD elision).
        {
            let mut graph = MaglevGraph::new(0);
            let mut block = BasicBlock::new(0);
            let a = block.push_value(ValueNode::SmiConstant { value: 1 });
            let b = block.push_value(ValueNode::SmiConstant { value: 2 });
            let sum = block.push_value(ValueNode::CheckedSmiAdd { left: a, right: b });
            let cmp = block.push_value(ValueNode::Int32LessThan {
                left: sum,
                right: a,
            });
            block.set_control(ControlNode::Return { value: cmp });
            graph.add_block(block);

            let set = narrow_set(&graph);
            assert!(
                set.contains(&sum),
                "i32-range CheckedSmiAdd consumed only by narrow \
                 Int32LessThan must be in narrow_int32"
            );
        }

        // Test 9: i32-range CheckedSmiAdd feeding Return is NOT narrow.
        {
            let mut graph = MaglevGraph::new(0);
            let mut block = BasicBlock::new(0);
            let a = block.push_value(ValueNode::SmiConstant { value: 1 });
            let b = block.push_value(ValueNode::SmiConstant { value: 2 });
            let sum = block.push_value(ValueNode::CheckedSmiAdd { left: a, right: b });
            block.set_control(ControlNode::Return { value: sum });
            graph.add_block(block);

            let set = narrow_set(&graph);
            assert!(
                !set.contains(&sum),
                "i32-range CheckedSmiAdd feeding Return must NOT be narrow"
            );
        }
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
