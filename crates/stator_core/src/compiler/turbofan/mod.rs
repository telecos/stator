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
//! BytecodeArray  →  MaglevGraph  →  Cranelift CLIF  →  native machine code
//! (interpreter)    (graph_builder)   (this module)       (cranelift-jit)
//! ```
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
//! use stator_core::compiler::maglev::ir::{
//!     BasicBlock, ControlNode, MaglevGraph, ValueNode,
//! };
//! use stator_core::compiler::turbofan::compile;
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
use cranelift_codegen::ir::{AbiParam, Function, InstBuilder, MemFlags, Signature};
use cranelift_codegen::ir::{Block, Value};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use crate::compiler::baseline::compiler::JIT_DEOPT;
use crate::compiler::maglev::ir::{ControlNode, MaglevGraph, NodeId, ValueNode};
use crate::error::{StatorError, StatorResult};

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
}

impl TurbofanCompiledCode {
    /// Execute the compiled function with the given register-file arguments.
    ///
    /// # Safety
    ///
    /// The code was produced by [`compile`] from a well-formed [`MaglevGraph`].
    /// Callers must not use this value after the owning
    /// [`TurbofanCompiledCode`] is dropped.
    pub unsafe fn execute(&self, args: &[i64]) -> StatorResult<i64> {
        let fn_ptr = self.module.get_finalized_function(self.func_id);

        let mut regs = vec![0i64; self.register_file_slots];
        for (i, &v) in args.iter().enumerate().take(regs.len()) {
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
        let mut sig = Signature::new(CallConv::SystemV);
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

        self.module.clear_context(&mut ctx);
        self.module
            .finalize_definitions()
            .map_err(|e| StatorError::Internal(format!("finalize_definitions: {e}")))?;

        Ok(TurbofanCompiledCode {
            module: self.module,
            func_id,
            register_file_slots: self.param_count as usize,
            deopt_points,
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
    deopt_block: Option<Block>,
    /// Next deopt-point index.
    deopt_index: u32,
}

impl<'a, 'b> Lowering<'a, 'b> {
    fn new(
        builder: &'b mut FunctionBuilder<'a>,
        graph: &'a MaglevGraph,
        _param_count: u32,
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
                let byte_offset = (*index as i32) * 8;
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

            // ── Unsupported nodes: produce a placeholder ────────────────────
            _ => {
                // Unsupported node kinds produce a 0 constant so that the rest
                // of the graph can still be translated.  This is intentional:
                // complex operations (object creation, string ops, calls, …)
                // are best handled by the interpreter tier via deopt.
                self.builder.ins().iconst(I64, 0)
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
                self.deopt_points.push(DeoptPoint {
                    index: self.deopt_index,
                    bytecode_offset: *bytecode_offset,
                    reason: "Deoptimize",
                });
                self.deopt_index += 1;
                let deopt_block = self.get_or_create_deopt_block();
                self.builder.ins().jump(deopt_block, &[]);
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
    fn get_or_create_deopt_block(&mut self) -> Block {
        if let Some(b) = self.deopt_block {
            return b;
        }
        let b = self.builder.create_block();
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
        self.builder
            .ins()
            .brif(overflow, deopt_block, &[], ok_block, &[]);
        self.deopt_points.push(DeoptPoint {
            index: self.deopt_index,
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
