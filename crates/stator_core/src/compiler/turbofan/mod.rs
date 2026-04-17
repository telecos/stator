//! Turbofan: Cranelift-backed optimising JIT backend.
//!
//! # Overview
//!
//! This module provides a bridge from the Maglev IR ([`MaglevGraph`]) to
//! Cranelift's CLIF (Cranelift Intermediate Form), enabling advanced JIT
//! compilation via the [`cranelift-jit`] crate.
//!
//! The compilation pipeline is:
//!
//! ```text
//! BytecodeArray  →  MaglevGraph  →  [pre-CLIF passes]  →  Cranelift CLIF  →  native machine code
//! (interpreter)    (graph_builder)   (specialize module)   (this module)       (cranelift-jit)
//! ```
//!
//! The optional pre-CLIF specialisation passes ([`specialize`]) are applied
//! before CLIF lowering when a [`FeedbackVector`] is available.  They perform
//! type narrowing, hot call-site specialisation, load/store elimination, and
//! escape-analysis-based allocation sinking.
//!
//! Use [`compile_with_feedback`] to compile a graph with pre-CLIF
//! specialisation enabled, or [`compile`] to skip it.
//!
//! # Type mapping
//!
//! JavaScript value types are mapped to Cranelift types as follows:
//!
//! | JS / Maglev type   | Cranelift type | Notes                          |
//! |--------------------|----------------|--------------------------------|
//! | Tagged (Smi/ref)   | `I64`          | Pointer-sized tagged value     |
//! | Int32 (unboxed)    | `I32`          | 32-bit signed integer          |
//! | Uint32 (unboxed)   | `I32`          | 32-bit unsigned (same bits)    |
//! | Float64 (unboxed)  | `F64`          | IEEE 754 double-precision      |
//! | Boolean            | `I8`           | 0 = false, 1 = true            |
//!
//! # Calling convention
//!
//! Generated functions use the same register-file calling convention as the
//! baseline JIT:
//!
//! ```text
//! extern "C" fn(regs: *mut i64) -> i64
//! ```
//!
//! `regs` is a caller-allocated array of `register_file_slots` × `i64`
//! values.  Parameters occupy the first `param_count` slots.  The return
//! value is the raw `i64` result.  On deoptimisation the special sentinel
//! [`JIT_DEOPT`][crate::compiler::baseline::compiler::JIT_DEOPT] is returned.
//!
//! # Deoptimisation
//!
//! Checked operations (e.g. [`ValueNode::CheckedSmiAdd`]) emit an overflow
//! check.  When overflow is detected the generated code branches to a shared
//! *deopt epilogue* that returns [`JIT_DEOPT`] to the caller, which then
//! re-invokes the interpreter via the deoptimiser.  Each deopt site is
//! recorded in [`TurbofanCompiledCode::deopt_points`] with its
//! [`DeoptPoint`] metadata.
//!
//! # Example
//!
//! ```
//! use stator_js::compiler::maglev::ir::{
//!     BasicBlock, ControlNode, MaglevGraph, ValueNode,
//! };
//! use stator_js::compiler::turbofan::compile;
//!
//! // Build a trivial graph: return the Int32 constant 42.
//! let mut graph = MaglevGraph::new(0);
//! let mut block = BasicBlock::new(0);
//! let c = block.push_value(ValueNode::Int32Constant { value: 42 });
//! block.set_control(ControlNode::Return { value: c });
//! graph.add_block(block);
//!
//! let compiled = compile(&graph, 0).expect("turbofan compile failed");
//! // SAFETY: compiled code is produced by cranelift-jit from a well-formed graph.
//! let result = unsafe { compiled.execute(&[]) }.expect("execute failed");
//! assert_eq!(result, 42);
//! ```

use std::collections::HashMap;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types::{F64, I8, I32, I64};
use cranelift_codegen::ir::{AbiParam, BlockArg, Function, InstBuilder, MemFlags, Signature};
use cranelift_codegen::ir::{Block, Value};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use crate::bytecode::feedback::FeedbackVector;
use crate::compiler::baseline::compiler::JIT_DEOPT;
use crate::compiler::maglev::ir::{ControlNode, MaglevGraph, NodeId, ValueNode};
use crate::error::{StatorError, StatorResult};

/// Pre-CLIF type-specialisation passes (type narrowing, call-site
/// specialisation, load/store elimination, escape analysis, GVN,
/// loop unrolling, register coalescing, scalar replacement).
pub mod specialize;

/// Cranelift deoptimiser: frame reconstruction and interpreter resume.
pub mod deopt;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Byte size of a single register-file slot (`i64`).
const REGISTER_SLOT_BYTES: i32 = 8;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// The JS value type for a Maglev IR node, mapped to a Cranelift type.
///
/// Used during lowering to select the correct Cranelift type for each
/// value-producing node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsType {
    /// A 64-bit tagged value (Smi pointer or heap-object reference).
    Tagged,
    /// An unboxed 32-bit signed integer.
    Int32,
    /// An unboxed 32-bit unsigned integer (same bit width as [`JsType::Int32`]).
    Uint32,
    /// An unboxed 64-bit IEEE 754 floating-point number.
    Float64,
    /// A boolean value stored as an 8-bit integer (0 = false, 1 = true).
    Bool,
}

impl JsType {
    /// Convert to the corresponding Cranelift IR type.
    pub fn to_cranelift(self) -> cranelift_codegen::ir::Type {
        match self {
            JsType::Tagged => I64,
            JsType::Int32 | JsType::Uint32 => I32,
            JsType::Float64 => F64,
            JsType::Bool => I8,
        }
    }
}

/// Metadata for a single deoptimisation point emitted by the Turbofan backend.
///
/// When a speculative check fails at runtime the generated code returns
/// [`JIT_DEOPT`] to the caller.  The caller can use this metadata together
/// with the interpreter's deoptimiser to reconstruct the interpreter frame
/// and resume execution.
#[derive(Debug, Clone)]
pub struct DeoptPoint {
    /// Zero-based index of the deopt point within the compiled function.
    pub index: u32,
    /// Byte offset in the original bytecode array at which to resume
    /// interpretation.
    pub bytecode_offset: u32,
    /// Human-readable description of the speculation that failed.
    pub reason: &'static str,
}

/// The result of compiling a [`MaglevGraph`] through the Turbofan backend.
///
/// Owns the JIT module (and therefore the compiled code) for the lifetime of
/// this value.  The code is no longer executable after this value is dropped.
pub struct TurbofanCompiledCode {
    /// Cranelift JIT module that owns the compiled machine code.
    module: JITModule,
    /// Cranelift function ID for the compiled function.
    func_id: FuncId,
    /// Number of `i64` slots in the register file
    /// (`param_count` slots for parameters).
    pub register_file_slots: usize,
    /// Deoptimisation sites recorded during compilation.
    pub deopt_points: Vec<DeoptPoint>,
    /// Approximate size of the emitted machine code in bytes.
    ///
    /// Captured from Cranelift's [`CompiledCode::code_buffer`] before the
    /// context is cleared.  Used only for statistics reporting.
    pub code_size: usize,
}

// SAFETY: `TurbofanCompiledCode` is transferred from a background compilation
// thread to the main interpreter thread via `Arc<Mutex<...>>`.  After
// `finalize_definitions()` the `JITModule`'s internal `RefCell<symbols>` is
// no longer mutated; the only post-transfer operation performed on the module
// is `get_finalized_function()`, which accesses `compiled_functions` (a plain
// `PrimaryMap`, no `RefCell`) and does NOT borrow `symbols`.  This has been
// verified against cranelift-jit 0.129.1 (backend.rs lines 265–274).  We
// guarantee that at most one thread calls `execute()` at a time (enforced by
// the `Mutex` in `TurbofanJitCodeCache`).
unsafe impl Send for TurbofanCompiledCode {}

impl std::fmt::Debug for TurbofanCompiledCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TurbofanCompiledCode")
            .field("register_file_slots", &self.register_file_slots)
            .field("deopt_points", &self.deopt_points.len())
            .field("code_size", &self.code_size)
            .finish_non_exhaustive()
    }
}

impl TurbofanCompiledCode {
    /// Execute the compiled function with the given register-file arguments.
    ///
    /// Uses a stack-allocated buffer ([`SmallVec`]) for the register file when
    /// the slot count is small enough, avoiding a heap allocation on the hot
    /// path.
    ///
    /// # Safety
    ///
    /// The code was produced by [`compile`] from a well-formed [`MaglevGraph`].
    /// Callers must not use this value after the owning
    /// [`TurbofanCompiledCode`] is dropped.
    pub unsafe fn execute(&self, args: &[i64]) -> StatorResult<i64> {
        use smallvec::{SmallVec, smallvec};

        let fn_ptr = self.module.get_finalized_function(self.func_id);

        // Allocate register_file_slots + 1 entries: the extra trailing slot
        // receives the deopt-site index when the JIT triggers a deopt.
        // Use SmallVec to keep the register file on the stack for typical
        // functions (≤ 32 slots).
        let total = self.register_file_slots + 1;
        let mut regs: SmallVec<[i64; 32]> = smallvec![0i64; total];
        for (i, &v) in args.iter().enumerate().take(self.register_file_slots) {
            regs[i] = v;
        }

        // SAFETY:
        // - fn_ptr is the address of a cranelift-jit-compiled function.
        // - The signature `extern "C" fn(*mut i64) -> i64` matches the
        //   register-file calling convention used in the code generator.
        // - regs is live for the duration of the call.
        let result = unsafe {
            let f: extern "C" fn(*mut i64) -> i64 = std::mem::transmute(fn_ptr);
            f(regs.as_mut_ptr())
        };

        if result == JIT_DEOPT {
            Err(StatorError::Internal("turbofan deopt".into()))
        } else {
            Ok(result)
        }
    }

    /// Execute the compiled function, falling back to the interpreter on deopt.
    ///
    /// If the JIT function triggers a deoptimisation (returns [`JIT_DEOPT`]),
    /// the register-file snapshot and the deopt-site index (written into the
    /// trailing slot by the JIT) are captured and forwarded to
    /// [`deopt::deoptimize_turbofan`], which reconstructs an
    /// [`InterpreterFrame`][crate::interpreter::InterpreterFrame] and resumes
    /// bytecode execution.
    ///
    /// On a successful (non-deopt) return the raw `i64` result is decoded to a
    /// [`JsValue`][crate::objects::value::JsValue] via
    /// [`jit_to_jsvalue`][crate::compiler::baseline::compiler::jit_to_jsvalue].
    ///
    /// Uses a stack-allocated buffer ([`SmallVec`]) for the register file when
    /// the slot count is small enough, avoiding a heap allocation on the hot
    /// path.
    ///
    /// # Safety
    ///
    /// The code was produced by [`compile`] from a well-formed [`MaglevGraph`].
    /// Callers must not use this value after the owning
    /// [`TurbofanCompiledCode`] is dropped.
    pub unsafe fn execute_with_deopt(
        &self,
        bytecode_array: crate::bytecode::bytecode_array::BytecodeArray,
        feedback: &mut FeedbackVector,
        global_env: std::rc::Rc<std::cell::RefCell<crate::interpreter::GlobalEnv>>,
    ) -> StatorResult<crate::objects::value::JsValue> {
        use smallvec::{SmallVec, smallvec};

        let fn_ptr = self.module.get_finalized_function(self.func_id);

        // +1 trailing slot for the deopt-site index written by the JIT.
        // Use SmallVec to keep the register file on the stack for typical
        // functions (≤ 32 slots).
        let total = self.register_file_slots + 1;
        let mut regs: SmallVec<[i64; 32]> = smallvec![0i64; total];

        // SAFETY: same invariants as execute().
        let result = unsafe {
            let f: extern "C" fn(*mut i64) -> i64 = std::mem::transmute(fn_ptr);
            f(regs.as_mut_ptr())
        };

        if result == JIT_DEOPT {
            // The JIT stored the deopt-site index into the trailing slot.
            let deopt_index = regs[self.register_file_slots] as u32;
            let frame = deopt::TurbofanFrameState {
                registers: regs[..self.register_file_slots].to_vec(),
                deopt_index,
            };
            deopt::deoptimize_turbofan(
                bytecode_array,
                feedback,
                &self.deopt_points,
                frame,
                global_env,
            )
        } else {
            use crate::compiler::baseline::compiler::jit_to_jsvalue;
            jit_to_jsvalue(result).ok_or_else(|| {
                StatorError::Internal(format!("turbofan: unrecognised return value {result}"))
            })
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry-point
// ─────────────────────────────────────────────────────────────────────────────

/// Compile a [`MaglevGraph`] to native machine code via Cranelift.
///
/// `param_count` is the number of formal parameters for the function; it
/// should equal `graph.parameter_count()`.
///
/// # Errors
///
/// Returns [`StatorError::Internal`] if Cranelift code generation fails or
/// the ISA cannot be initialised.
pub fn compile(graph: &MaglevGraph, param_count: u32) -> StatorResult<TurbofanCompiledCode> {
    TurbofanCodegen::new(param_count)?.compile(graph)
}

/// Compile a [`MaglevGraph`] to native machine code via Cranelift, running the
/// pre-CLIF specialisation passes first.
///
/// If `fv` is `Some`, [`specialize::run_pre_clif_passes`] is applied to a
/// clone of `graph` before CLIF lowering.  This enables:
/// - **type narrowing** from feedback (generic → checked-Smi / Int32 ops),
/// - **hot call-site specialisation** (call → known-function),
/// - **load/store elimination** (redundant field loads CSE),
/// - **escape analysis / allocation sinking** (non-escaping allocs →
///   virtual objects).
///
/// When `fv` is `None` the behaviour is identical to [`compile`].
///
/// # Errors
///
/// Returns [`StatorError::Internal`] if Cranelift code generation fails or
/// the ISA cannot be initialised.
pub fn compile_with_feedback(
    graph: &MaglevGraph,
    param_count: u32,
    fv: Option<&FeedbackVector>,
) -> StatorResult<TurbofanCompiledCode> {
    if fv.is_some() {
        let mut specialised = graph.clone();
        specialize::run_pre_clif_passes(&mut specialised, fv);
        TurbofanCodegen::new(param_count)?.compile(&specialised)
    } else {
        TurbofanCodegen::new(param_count)?.compile(graph)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal code-generation state
// ─────────────────────────────────────────────────────────────────────────────

/// Internal code-generation state for a single Turbofan-compiled function.
struct TurbofanCodegen {
    /// Cranelift JIT module.
    module: JITModule,
    /// Number of function parameters.
    param_count: u32,
}

impl TurbofanCodegen {
    /// Create a new code generator backed by a fresh [`JITModule`].
    fn new(param_count: u32) -> StatorResult<Self> {
        let module = JITModule::new(
            JITBuilder::with_flags(
                &[("opt_level", "speed")],
                cranelift_module::default_libcall_names(),
            )
            .map_err(|e| StatorError::Internal(format!("cranelift JITBuilder: {e}")))?,
        );
        Ok(Self {
            module,
            param_count,
        })
    }

    /// Compile the given graph and return the finalised [`TurbofanCompiledCode`].
    fn compile(mut self, graph: &MaglevGraph) -> StatorResult<TurbofanCompiledCode> {
        // Build the Cranelift function signature:
        //   extern "C" fn(regs: *mut i64) -> i64
        let pointer_type = self.module.target_config().pointer_type();
        let call_conv = self.module.isa().default_call_conv();
        let mut sig = Signature::new(call_conv);
        sig.params.push(AbiParam::new(pointer_type));
        sig.returns.push(AbiParam::new(I64));

        let func_id = self
            .module
            .declare_function("turbofan_fn", Linkage::Local, &sig)
            .map_err(|e| StatorError::Internal(format!("declare_function: {e}")))?;

        let mut ctx = self.module.make_context();
        ctx.func = Function::with_name_signature(
            cranelift_codegen::ir::UserFuncName::user(0, func_id.as_u32()),
            sig,
        );

        let mut fb_ctx = FunctionBuilderContext::new();
        let mut deopt_points: Vec<DeoptPoint> = Vec::new();

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
            let lowering = Lowering::new(
                &mut builder,
                graph,
                self.param_count,
                &mut deopt_points,
                pointer_type,
            );
            lowering.lower()?;
            builder.finalize();
        }

        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| StatorError::Internal(format!("define_function: {e}")))?;

        // Capture the compiled code size before clearing the context.
        let code_size = ctx
            .compiled_code()
            .map(|c| c.code_buffer().len())
            .unwrap_or(0);

        self.module.clear_context(&mut ctx);
        self.module
            .finalize_definitions()
            .map_err(|e| StatorError::Internal(format!("finalize_definitions: {e}")))?;

        Ok(TurbofanCompiledCode {
            module: self.module,
            func_id,
            register_file_slots: self.param_count as usize,
            deopt_points,
            code_size,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lowering: MaglevGraph → Cranelift CLIF
// ─────────────────────────────────────────────────────────────────────────────

/// Lowers a [`MaglevGraph`] into Cranelift CLIF IR via a [`FunctionBuilder`].
struct Lowering<'a, 'b> {
    builder: &'b mut FunctionBuilder<'a>,
    graph: &'a MaglevGraph,
    deopt_points: &'b mut Vec<DeoptPoint>,
    /// Map from graph [`NodeId`] to the corresponding Cranelift [`Value`].
    value_map: HashMap<NodeId, Value>,
    /// Map from block index to Cranelift [`Block`].
    block_map: HashMap<u32, Block>,
    /// Cranelift SSA variable for the register-file pointer argument.
    regs_var: cranelift_frontend::Variable,
    /// Deopt epilogue block (lazily created).
    ///
    /// This block accepts a single `I32` parameter: the deopt-site index.
    /// The body stores the index into the trailing register-file slot (at
    /// byte offset `register_file_slots × 8`) and returns [`JIT_DEOPT`].
    deopt_block: Option<Block>,
    /// Next deopt-point index.
    deopt_index: u32,
    /// Number of user-visible register-file slots (= `param_count`).
    ///
    /// The caller allocates one extra slot beyond this count for the
    /// deopt-site index.  The deopt epilogue writes its `I32` parameter
    /// (the site index) into that slot.
    register_file_slots: u32,
}

impl<'a, 'b> Lowering<'a, 'b> {
    fn new(
        builder: &'b mut FunctionBuilder<'a>,
        graph: &'a MaglevGraph,
        param_count: u32,
        deopt_points: &'b mut Vec<DeoptPoint>,
        pointer_type: cranelift_codegen::ir::Type,
    ) -> Self {
        // declare_var(ty) allocates a new SSA variable and returns its ID.
        let regs_var = builder.declare_var(pointer_type);
        Self {
            builder,
            graph,
            deopt_points,
            value_map: HashMap::new(),
            block_map: HashMap::new(),
            regs_var,
            deopt_block: None,
            deopt_index: 0,
            register_file_slots: param_count,
        }
    }

    /// Lower the entire graph.
    fn lower(mut self) -> StatorResult<()> {
        // Extract the graph reference as a Copy value so that subsequent
        // mutable borrows of `self` do not conflict with graph accesses.
        let graph = self.graph;

        // Pre-allocate one Cranelift block per Maglev basic block.
        for block in graph.blocks() {
            let clif_block = self.builder.create_block();
            self.block_map.insert(block.id, clif_block);
        }

        // Entry block: append function params, store the register-file pointer
        // in the SSA variable so it can be read in any successor block.
        let entry_clif = self.block_map[&0];
        self.builder.switch_to_block(entry_clif);
        self.builder
            .append_block_params_for_function_params(entry_clif);
        let regs_ptr = self.builder.block_params(entry_clif)[0];
        self.builder.def_var(self.regs_var, regs_ptr);

        // Lower each Maglev basic block in order.
        for block_idx in 0..graph.blocks().len() {
            let maglev_block = &graph.blocks()[block_idx];
            let clif_block = self.block_map[&maglev_block.id];

            if block_idx > 0 {
                self.builder.switch_to_block(clif_block);
            }

            // Lower value nodes.  `node` borrows from `graph` (lifetime 'a)
            // which is independent of the `&mut self` needed by lower_value_node.
            for (node_id, node) in &maglev_block.nodes {
                self.lower_value_node(*node_id, node)?;
            }

            // Lower control node.
            if let Some(ctrl) = &maglev_block.control {
                self.lower_control_node(ctrl)?;
            }

            // Seal once fully emitted (all predecessors are known).
            self.builder.seal_block(clif_block);
        }

        // Emit the deopt epilogue if it was created.
        if let Some(deopt_block) = self.deopt_block {
            self.builder.switch_to_block(deopt_block);
            self.builder.seal_block(deopt_block);
            // The block accepts a single I32 parameter: the deopt-site index.
            // Store it (sign-extended to I64) into the trailing register-file
            // slot so the caller can identify which guard fired.
            let idx_param = self.builder.block_params(deopt_block)[0];
            let regs = self.builder.use_var(self.regs_var);
            let slot_offset = (self.register_file_slots as i32) * REGISTER_SLOT_BYTES;
            let idx_i64 = self.builder.ins().sextend(I64, idx_param);
            self.builder
                .ins()
                .store(MemFlags::new(), idx_i64, regs, slot_offset);
            let sentinel = self.builder.ins().iconst(I64, JIT_DEOPT);
            self.builder.ins().return_(&[sentinel]);
        }

        Ok(())
    }

    /// Lower a single [`ValueNode`] and record its Cranelift [`Value`] in
    /// `value_map`.
    fn lower_value_node(&mut self, id: NodeId, node: &ValueNode) -> StatorResult<()> {
        let val = match node {
            // ── Constants ──────────────────────────────────────────────────
            ValueNode::SmiConstant { value } => {
                // Smis are tagged: value is the raw i32 shifted left by 1
                // (Smi tag bit = 0).  We widen to i64 tagged.
                let tagged = (*value as i64) << 1;
                self.builder.ins().iconst(I64, tagged)
            }
            ValueNode::Int32Constant { value } => self.builder.ins().iconst(I32, *value as i64),
            ValueNode::Uint32Constant { value } => self.builder.ins().iconst(I32, *value as i64),
            ValueNode::Float64Constant { value } => self.builder.ins().f64const(*value),
            ValueNode::TrueConstant => self.builder.ins().iconst(I64, 1),
            ValueNode::FalseConstant => self.builder.ins().iconst(I64, 0),
            ValueNode::NullConstant => self.builder.ins().iconst(I64, 0),
            ValueNode::UndefinedConstant => self.builder.ins().iconst(I64, 0),

            // ── Parameters ─────────────────────────────────────────────────
            ValueNode::Parameter { index } => {
                // Parameters are stored as i64 slots in the register file.
                // Load from regs[index].
                let regs = self.builder.use_var(self.regs_var);
                let byte_offset = (*index as i32) * REGISTER_SLOT_BYTES;
                self.builder
                    .ins()
                    .load(I64, MemFlags::new(), regs, byte_offset)
            }

            // ── Int32 arithmetic ───────────────────────────────────────────
            ValueNode::Int32Add { left, right } => {
                let l = self.use_i32(*left)?;
                let r = self.use_i32(*right)?;
                self.builder.ins().iadd(l, r)
            }
            ValueNode::Int32Subtract { left, right } => {
                let l = self.use_i32(*left)?;
                let r = self.use_i32(*right)?;
                self.builder.ins().isub(l, r)
            }
            ValueNode::Int32Multiply { left, right } => {
                let l = self.use_i32(*left)?;
                let r = self.use_i32(*right)?;
                self.builder.ins().imul(l, r)
            }
            ValueNode::Int32Divide { left, right } => {
                let l = self.use_i32(*left)?;
                let r = self.use_i32(*right)?;
                self.builder.ins().sdiv(l, r)
            }
            ValueNode::Int32Modulus { left, right } => {
                let l = self.use_i32(*left)?;
                let r = self.use_i32(*right)?;
                self.builder.ins().srem(l, r)
            }
            ValueNode::Int32Negate { value } => {
                let v = self.use_i32(*value)?;
                self.builder.ins().ineg(v)
            }
            ValueNode::Int32Increment { value } => {
                let v = self.use_i32(*value)?;
                let one = self.builder.ins().iconst(I32, 1);
                self.builder.ins().iadd(v, one)
            }
            ValueNode::Int32Decrement { value } => {
                let v = self.use_i32(*value)?;
                let one = self.builder.ins().iconst(I32, 1);
                self.builder.ins().isub(v, one)
            }

            // ── Float64 arithmetic ─────────────────────────────────────────
            ValueNode::Float64Add { left, right } => {
                let l = self.use_f64(*left)?;
                let r = self.use_f64(*right)?;
                self.builder.ins().fadd(l, r)
            }
            ValueNode::Float64Subtract { left, right } => {
                let l = self.use_f64(*left)?;
                let r = self.use_f64(*right)?;
                self.builder.ins().fsub(l, r)
            }
            ValueNode::Float64Multiply { left, right } => {
                let l = self.use_f64(*left)?;
                let r = self.use_f64(*right)?;
                self.builder.ins().fmul(l, r)
            }
            ValueNode::Float64Divide { left, right } => {
                let l = self.use_f64(*left)?;
                let r = self.use_f64(*right)?;
                self.builder.ins().fdiv(l, r)
            }
            ValueNode::Float64Negate { value } => {
                let v = self.use_f64(*value)?;
                self.builder.ins().fneg(v)
            }

            // ── CheckedSmi arithmetic (with deopt on overflow) ─────────────
            ValueNode::CheckedSmiAdd { left, right } => {
                self.lower_checked_smi_add(*left, *right)?
            }
            ValueNode::CheckedSmiSubtract { left, right } => {
                self.lower_checked_smi_sub(*left, *right)?
            }
            ValueNode::CheckedSmiMultiply { left, right } => {
                self.lower_checked_smi_mul(*left, *right)?
            }

            // ── Type conversions ───────────────────────────────────────────
            ValueNode::ChangeInt32ToFloat64 { input } => {
                let v = self.use_i32(*input)?;
                self.builder.ins().fcvt_from_sint(F64, v)
            }
            ValueNode::ChangeFloat64ToInt32 { input } => {
                let v = self.use_f64(*input)?;
                self.builder.ins().fcvt_to_sint_sat(I32, v)
            }
            ValueNode::ChangeInt32ToTagged { input } => {
                // Tag an int32 as a Smi: shift left 1, widen to i64.
                let v = self.use_i32(*input)?;
                let shifted = self.builder.ins().ishl_imm(v, 1);
                self.builder.ins().sextend(I64, shifted)
            }
            ValueNode::ChangeTaggedToInt32 { input } => {
                // Untag a Smi: narrow to i64, shift right 1, truncate to i32.
                let v = self.use_i64(*input)?;
                let shifted = self.builder.ins().sshr_imm(v, 1);
                self.builder.ins().ireduce(I32, shifted)
            }
            ValueNode::ChangeFloat64ToTagged { input } => {
                // Re-interpret f64 bits as i64 (heap-number encoding stub).
                let v = self.use_f64(*input)?;
                self.builder.ins().bitcast(I64, MemFlags::new(), v)
            }
            ValueNode::ChangeTaggedToFloat64 { input } => {
                // Re-interpret i64 bits as f64.
                let v = self.use_i64(*input)?;
                self.builder.ins().bitcast(F64, MemFlags::new(), v)
            }

            // ── Phi nodes ─────────────────────────────────────────────────
            ValueNode::Phi { .. } => {
                // Phi nodes are handled during block param setup; emit a
                // placeholder value (block param) for the first input type.
                // For now, create a block parameter of type I64 so that the
                // graph still compiles; a more complete implementation would
                // resolve types from predecessor values.
                let current_block = self
                    .builder
                    .current_block()
                    .ok_or_else(|| StatorError::Internal("phi outside block".into()))?;
                self.builder.append_block_param(current_block, I64)
            }

            // ── Unsupported nodes: trigger deoptimisation ──────────────────
            _ => {
                // Unsupported node kinds (e.g. GenericAdd, string operations,
                // object creation) produce the JIT_DEOPT sentinel so the
                // caller falls back to the next lower tier.  This is
                // intentional: complex or untyped operations are best handled
                // by the Maglev or interpreter tier.
                self.builder.ins().iconst(I64, JIT_DEOPT)
            }
        };

        self.value_map.insert(id, val);
        Ok(())
    }

    /// Lower a [`ControlNode`].
    fn lower_control_node(&mut self, ctrl: &ControlNode) -> StatorResult<()> {
        match ctrl {
            ControlNode::Return { value } => {
                let v = self.use_value(*value)?;
                // Widen to i64 if necessary so that the return type matches
                // the declared signature.
                let ret_val = self.coerce_to_i64(v)?;
                self.builder.ins().return_(&[ret_val]);
            }
            ControlNode::Jump { target } => {
                let target_block = self.block_for(*target)?;
                self.builder.ins().jump(target_block, &[]);
            }
            ControlNode::Branch {
                condition,
                if_true,
                if_false,
            } => {
                let cond_val = self.use_value(*condition)?;
                // Normalise condition to i8 for brnz/brz.
                let cond8 = self.coerce_to_bool(cond_val)?;
                let true_block = self.block_for(*if_true)?;
                let false_block = self.block_for(*if_false)?;
                self.builder
                    .ins()
                    .brif(cond8, true_block, &[], false_block, &[]);
            }
            ControlNode::Deoptimize {
                bytecode_offset,
                reason: _,
            } => {
                let site_index = self.deopt_index;
                self.deopt_points.push(DeoptPoint {
                    index: site_index,
                    bytecode_offset: *bytecode_offset,
                    reason: "Deoptimize",
                });
                self.deopt_index += 1;
                let deopt_block = self.get_or_create_deopt_block();
                let idx_val = self.builder.ins().iconst(I32, site_index as i64);
                self.builder
                    .ins()
                    .jump(deopt_block, &[BlockArg::from(idx_val)]);
            }
        }
        Ok(())
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Get the Cranelift block for a Maglev block index.
    fn block_for(&self, idx: u32) -> StatorResult<Block> {
        self.block_map
            .get(&idx)
            .copied()
            .ok_or_else(|| StatorError::Internal(format!("unknown block index {idx}")))
    }

    /// Look up the [`Value`] for a [`NodeId`].
    fn use_value(&self, id: NodeId) -> StatorResult<Value> {
        self.value_map
            .get(&id)
            .copied()
            .ok_or_else(|| StatorError::Internal(format!("undefined node {:?}", id)))
    }

    /// Look up a value and coerce it to `I32`.
    fn use_i32(&mut self, id: NodeId) -> StatorResult<Value> {
        let v = self.use_value(id)?;
        let ty = self.builder.func.dfg.value_type(v);
        if ty == I64 {
            Ok(self.builder.ins().ireduce(I32, v))
        } else {
            Ok(v)
        }
    }

    /// Look up a value and coerce it to `I64`.
    fn use_i64(&mut self, id: NodeId) -> StatorResult<Value> {
        let v = self.use_value(id)?;
        let ty = self.builder.func.dfg.value_type(v);
        if ty == I32 {
            Ok(self.builder.ins().sextend(I64, v))
        } else {
            Ok(v)
        }
    }

    /// Look up a value and coerce it to `F64`.
    fn use_f64(&mut self, id: NodeId) -> StatorResult<Value> {
        let v = self.use_value(id)?;
        let ty = self.builder.func.dfg.value_type(v);
        if ty == I64 {
            Ok(self.builder.ins().bitcast(F64, MemFlags::new(), v))
        } else {
            Ok(v)
        }
    }

    /// Coerce any integer value to `I64` for use as a return value.
    fn coerce_to_i64(&mut self, v: Value) -> StatorResult<Value> {
        let ty = self.builder.func.dfg.value_type(v);
        if ty == I32 {
            Ok(self.builder.ins().sextend(I64, v))
        } else if ty == I8 {
            Ok(self.builder.ins().uextend(I64, v))
        } else if ty == F64 {
            Ok(self.builder.ins().bitcast(I64, MemFlags::new(), v))
        } else {
            Ok(v)
        }
    }

    /// Coerce a value to a single-bit `I8` for use as a branch condition.
    fn coerce_to_bool(&mut self, v: Value) -> StatorResult<Value> {
        let ty = self.builder.func.dfg.value_type(v);
        if ty == I8 {
            Ok(v)
        } else if ty == I32 || ty == I64 {
            let zero = self.builder.ins().iconst(ty, 0);
            Ok(self.builder.ins().icmp(IntCC::NotEqual, v, zero))
        } else if ty == F64 {
            let zero = self.builder.ins().f64const(0.0);
            Ok(self.builder.ins().fcmp(FloatCC::NotEqual, v, zero))
        } else {
            Ok(v)
        }
    }

    /// Get-or-create the shared deopt epilogue block.
    ///
    /// The block has a single `I32` block parameter that holds the deopt-site
    /// index.  Every branch to this block must pass the index as an argument.
    fn get_or_create_deopt_block(&mut self) -> Block {
        if let Some(b) = self.deopt_block {
            return b;
        }
        let b = self.builder.create_block();
        // Deopt-site index parameter (I32).
        self.builder.append_block_param(b, I32);
        self.deopt_block = Some(b);
        b
    }

    /// Emit a checked Smi add with overflow deopt.
    fn lower_checked_smi_add(&mut self, left: NodeId, right: NodeId) -> StatorResult<Value> {
        let (l_i32, r_i32) = self.untag_smi_pair(left, right)?;
        let (result_i32, overflow) = self.builder.ins().sadd_overflow(l_i32, r_i32);
        self.emit_overflow_deopt(overflow, "CheckedSmiAdd overflow")?;
        Ok(self.retag_smi(result_i32))
    }

    /// Emit a checked Smi subtract with overflow deopt.
    fn lower_checked_smi_sub(&mut self, left: NodeId, right: NodeId) -> StatorResult<Value> {
        let (l_i32, r_i32) = self.untag_smi_pair(left, right)?;
        let (result_i32, overflow) = self.builder.ins().ssub_overflow(l_i32, r_i32);
        self.emit_overflow_deopt(overflow, "CheckedSmiSubtract overflow")?;
        Ok(self.retag_smi(result_i32))
    }

    /// Emit a checked Smi multiply with overflow deopt.
    fn lower_checked_smi_mul(&mut self, left: NodeId, right: NodeId) -> StatorResult<Value> {
        let (l_i32, r_i32) = self.untag_smi_pair(left, right)?;
        let (result_i32, overflow) = self.builder.ins().smul_overflow(l_i32, r_i32);
        self.emit_overflow_deopt(overflow, "CheckedSmiMultiply overflow")?;
        Ok(self.retag_smi(result_i32))
    }

    /// Untag two Smi-tagged I64 values to unboxed I32 pairs.
    fn untag_smi_pair(&mut self, left: NodeId, right: NodeId) -> StatorResult<(Value, Value)> {
        let l_i64 = self.use_i64(left)?;
        let r_i64 = self.use_i64(right)?;
        let l_i32 = {
            let shifted = self.builder.ins().sshr_imm(l_i64, 1);
            self.builder.ins().ireduce(I32, shifted)
        };
        let r_i32 = {
            let shifted = self.builder.ins().sshr_imm(r_i64, 1);
            self.builder.ins().ireduce(I32, shifted)
        };
        Ok((l_i32, r_i32))
    }

    /// Branch to deopt on non-zero overflow flag, fall through to a new ok block.
    fn emit_overflow_deopt(&mut self, overflow: Value, reason: &'static str) -> StatorResult<()> {
        let ok_block = self.builder.create_block();
        let deopt_block = self.get_or_create_deopt_block();
        let site_index = self.deopt_index;
        let idx_val = self.builder.ins().iconst(I32, site_index as i64);
        self.builder.ins().brif(
            overflow,
            deopt_block,
            &[BlockArg::from(idx_val)],
            ok_block,
            &[],
        );
        self.deopt_points.push(DeoptPoint {
            index: site_index,
            bytecode_offset: 0,
            reason,
        });
        self.deopt_index += 1;
        self.builder.switch_to_block(ok_block);
        self.builder.seal_block(ok_block);
        Ok(())
    }

    /// Re-tag an unboxed I32 result as a Smi-tagged I64.
    fn retag_smi(&mut self, v: Value) -> Value {
        let shifted = self.builder.ins().ishl_imm(v, 1);
        self.builder.ins().sextend(I64, shifted)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::maglev::ir::{BasicBlock, ControlNode, MaglevGraph, ValueNode};

    fn run(graph: &MaglevGraph, param_count: u32, args: &[i64]) -> i64 {
        let compiled = compile(graph, param_count).expect("turbofan compile failed");
        // SAFETY: compiled code is produced by cranelift-jit from a
        // well-formed MaglevGraph constructed in a unit test.
        unsafe { compiled.execute(args) }.expect("execute failed")
    }

    // ── Basic constant returns ────────────────────────────────────────────────

    #[test]
    fn test_return_int32_constant() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c = block.push_value(ValueNode::Int32Constant { value: 42 });
        block.set_control(ControlNode::Return { value: c });
        graph.add_block(block);

        assert_eq!(run(&graph, 0, &[]), 42);
    }

    #[test]
    fn test_return_smi_constant() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        // SmiConstant { 7 } → tagged: 7 << 1 = 14
        let c = block.push_value(ValueNode::SmiConstant { value: 7 });
        block.set_control(ControlNode::Return { value: c });
        graph.add_block(block);

        assert_eq!(run(&graph, 0, &[]), 14); // tagged representation
    }

    #[test]
    fn test_return_float64_constant() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c = block.push_value(ValueNode::Float64Constant { value: 3.14 });
        // Re-interpret bits as i64 for tagged return.
        let tagged = block.push_value(ValueNode::ChangeFloat64ToTagged { input: c });
        block.set_control(ControlNode::Return { value: tagged });
        graph.add_block(block);

        let raw = run(&graph, 0, &[]);
        // SAFETY: bits were stored as f64, now read back as f64.
        let f: f64 = f64::from_bits(raw as u64);
        assert!((f - 3.14).abs() < 1e-10);
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    #[test]
    fn test_int32_add() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let a = block.push_value(ValueNode::Int32Constant { value: 10 });
        let b = block.push_value(ValueNode::Int32Constant { value: 32 });
        let sum = block.push_value(ValueNode::Int32Add { left: a, right: b });
        block.set_control(ControlNode::Return { value: sum });
        graph.add_block(block);

        assert_eq!(run(&graph, 0, &[]), 42);
    }

    #[test]
    fn test_int32_subtract() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let a = block.push_value(ValueNode::Int32Constant { value: 100 });
        let b = block.push_value(ValueNode::Int32Constant { value: 58 });
        let diff = block.push_value(ValueNode::Int32Subtract { left: a, right: b });
        block.set_control(ControlNode::Return { value: diff });
        graph.add_block(block);

        assert_eq!(run(&graph, 0, &[]), 42);
    }

    #[test]
    fn test_int32_multiply() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let a = block.push_value(ValueNode::Int32Constant { value: 6 });
        let b = block.push_value(ValueNode::Int32Constant { value: 7 });
        let prod = block.push_value(ValueNode::Int32Multiply { left: a, right: b });
        block.set_control(ControlNode::Return { value: prod });
        graph.add_block(block);

        assert_eq!(run(&graph, 0, &[]), 42);
    }

    // ── Parameters ────────────────────────────────────────────────────────────

    #[test]
    fn test_return_parameter() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        block.set_control(ControlNode::Return { value: p });
        graph.add_block(block);

        assert_eq!(run(&graph, 1, &[99]), 99);
    }

    #[test]
    fn test_add_two_parameters() {
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

        // Pass tagged i64 parameters, read back as i32 (truncated).
        assert_eq!(run(&graph, 2, &[20, 22]) as i32, 42);
    }

    // ── Float64 arithmetic ────────────────────────────────────────────────────

    #[test]
    fn test_float64_add() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let a = block.push_value(ValueNode::Float64Constant { value: 1.5 });
        let b = block.push_value(ValueNode::Float64Constant { value: 2.5 });
        let sum = block.push_value(ValueNode::Float64Add { left: a, right: b });
        let tagged = block.push_value(ValueNode::ChangeFloat64ToTagged { input: sum });
        block.set_control(ControlNode::Return { value: tagged });
        graph.add_block(block);

        let raw = run(&graph, 0, &[]);
        let f: f64 = f64::from_bits(raw as u64);
        assert!((f - 4.0).abs() < 1e-10);
    }

    // ── Type conversions ──────────────────────────────────────────────────────

    #[test]
    fn test_int32_to_float64() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c = block.push_value(ValueNode::Int32Constant { value: 10 });
        let f = block.push_value(ValueNode::ChangeInt32ToFloat64 { input: c });
        let tagged = block.push_value(ValueNode::ChangeFloat64ToTagged { input: f });
        block.set_control(ControlNode::Return { value: tagged });
        graph.add_block(block);

        let raw = run(&graph, 0, &[]);
        let fv: f64 = f64::from_bits(raw as u64);
        assert!((fv - 10.0).abs() < 1e-10);
    }

    // ── Checked Smi operations (deopt path) ───────────────────────────────────

    #[test]
    fn test_checked_smi_add_no_overflow() {
        // Build: CheckedSmiAdd(SmiConstant(3), SmiConstant(4))
        // SmiConstant(v) → tagged = v << 1, so inputs are 6 and 8.
        // The operation untags (shift right 1), adds (3+4=7), re-tags (14).
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let a = block.push_value(ValueNode::SmiConstant { value: 3 });
        let b = block.push_value(ValueNode::SmiConstant { value: 4 });
        let sum = block.push_value(ValueNode::CheckedSmiAdd { left: a, right: b });
        block.set_control(ControlNode::Return { value: sum });
        graph.add_block(block);

        assert_eq!(run(&graph, 0, &[]), 14); // (3+4) << 1 = 14
    }

    #[test]
    fn test_checked_smi_add_overflow_deopt() {
        // Overflow path: i32::MAX + 1 should trigger deopt.
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let a = block.push_value(ValueNode::SmiConstant { value: i32::MAX });
        let b = block.push_value(ValueNode::SmiConstant { value: 1 });
        let sum = block.push_value(ValueNode::CheckedSmiAdd { left: a, right: b });
        block.set_control(ControlNode::Return { value: sum });
        graph.add_block(block);

        let compiled = compile(&graph, 0).expect("compile ok");
        // SAFETY: compiled from a well-formed graph in a unit test.
        let result = unsafe { compiled.execute(&[]) };
        assert!(result.is_err(), "expected deopt error, got: {result:?}");
    }

    // ── Deopt node ────────────────────────────────────────────────────────────

    #[test]
    fn test_deoptimize_node_returns_error() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        // Any value node to avoid empty block
        let _ = block.push_value(ValueNode::UndefinedConstant);
        block.set_control(ControlNode::Deoptimize {
            bytecode_offset: 4,
            reason: 1,
        });
        graph.add_block(block);

        let compiled = compile(&graph, 0).expect("compile ok");
        assert_eq!(compiled.deopt_points.len(), 1);
        assert_eq!(compiled.deopt_points[0].bytecode_offset, 4);
        // SAFETY: compiled from a well-formed graph in a unit test.
        let result = unsafe { compiled.execute(&[]) };
        assert!(result.is_err());
    }

    // ── JsType helper ─────────────────────────────────────────────────────────

    #[test]
    fn test_js_type_to_cranelift() {
        assert_eq!(JsType::Tagged.to_cranelift(), I64);
        assert_eq!(JsType::Int32.to_cranelift(), I32);
        assert_eq!(JsType::Uint32.to_cranelift(), I32);
        assert_eq!(JsType::Float64.to_cranelift(), F64);
        assert_eq!(JsType::Bool.to_cranelift(), I8);
    }
}
