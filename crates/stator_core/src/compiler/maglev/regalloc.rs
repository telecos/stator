//! Maglev linear-scan register allocator.
//!
//! # Overview
//!
//! This module implements a linear-scan register allocator over a
//! [`MaglevGraph`].  The allocator:
//!
//! 1. **Linearises** the graph — blocks are visited in index order; within
//!    each block value nodes are visited in their stored order, followed by the
//!    block's control node.  Each position receives a unique *program point*
//!    (a monotonically increasing `u32`).
//!
//! 2. **Computes live ranges** — for every [`NodeId`] the allocator records
//!    the program point at which the value is *defined* (the node that
//!    produces it) and the *last* program point at which it is *used* as an
//!    input.
//!
//! 3. **Runs linear scan** — live intervals are sorted by their start point.
//!    The allocator greedily assigns one of `num_regs` physical registers.
//!    When all registers are occupied it *spills* the live interval with the
//!    farthest endpoint, freeing a register for the new interval (Poletto &
//!    Sarkar, 1999).
//!
//! # Output
//!
//! The allocator produces an [`AllocationResult`] that maps each [`NodeId`]
//! to a [`Location`]:
//!
//! - [`Location::Register(u32)`] — the value lives in physical register `n`.
//! - [`Location::StackSlot(u32)`] — the value has been spilled to stack slot
//!   `n`.
//!
//! # Example
//!
//! ```
//! use stator_core::compiler::maglev::ir::{
//!     BasicBlock, ControlNode, MaglevGraph, ValueNode,
//! };
//! use stator_core::compiler::maglev::regalloc::{allocate, Location};
//!
//! let mut graph = MaglevGraph::new(1);
//! let mut block = BasicBlock::new(0);
//! let p0 = block.push_value(ValueNode::Parameter { index: 0 });
//! block.set_control(ControlNode::Return { value: p0 });
//! graph.add_block(block);
//!
//! let result = allocate(&graph, 8);
//! // With 8 registers and a single value, no spill occurs.
//! assert!(matches!(result.location(p0), Some(Location::Register(_))));
//! assert_eq!(result.spill_count(), 0);
//! ```

use std::collections::HashMap;

use crate::compiler::maglev::ir::{ControlNode, MaglevGraph, NodeId, ValueNode};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// The physical location assigned to a [`NodeId`] by the register allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Location {
    /// The value lives in physical register `n`.
    Register(u32),
    /// The value has been spilled to stack slot `n`.
    StackSlot(u32),
}

/// The result of register allocation for a [`MaglevGraph`].
///
/// Obtained by calling [`allocate`].
#[derive(Debug, Clone)]
pub struct AllocationResult {
    /// Map from [`NodeId`] to the assigned [`Location`].
    assignments: HashMap<NodeId, Location>,
    /// Total number of stack slots allocated (i.e. the required stack-frame
    /// size in slots).
    spill_count: u32,
    /// Per-node bitmask of caller-saved allocatable registers (indices 1–5:
    /// RCX, RDX, RSI, R8, R9) that hold live values **across** that node.
    /// A register is included only when its value is still needed *after* the
    /// node, so stub-call arguments consumed at the call site are excluded.
    live_caller_saved: HashMap<NodeId, u8>,
}

impl AllocationResult {
    /// Look up the [`Location`] assigned to `id`.  Returns `None` for nodes
    /// that produce no value (this should not happen for a well-formed graph).
    pub fn location(&self, id: NodeId) -> Option<Location> {
        self.assignments.get(&id).copied()
    }

    /// The number of stack slots required by the function's frame.
    pub fn spill_count(&self) -> u32 {
        self.spill_count
    }

    /// Returns a bitmask of caller-saved allocatable registers (bits 1–5)
    /// that hold values live *across* the given node.  Returns `0` when no
    /// caller-saved register needs saving at that point.
    pub fn live_caller_saved_at(&self, id: NodeId) -> u8 {
        self.live_caller_saved.get(&id).copied().unwrap_or(0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Live-range analysis
// ─────────────────────────────────────────────────────────────────────────────

/// A half-open live interval `[start, end)` for a single [`NodeId`].
///
/// `start` is the program point at which the value is *defined*; `end` is one
/// past the last program point at which the value is *used* as an input.
#[derive(Debug, Clone, Copy)]
struct LiveInterval {
    id: NodeId,
    start: u32,
    end: u32,
}

/// Compute a live interval for every value-producing node in `graph`.
///
/// Program points are assigned by visiting blocks in index order; within each
/// block value nodes are visited in their stored order and the control node
/// occupies the next program point after the last value node.
fn compute_live_intervals(graph: &MaglevGraph) -> Vec<LiveInterval> {
    // Pass 1: assign a definition program point to every node.
    let mut def_point: HashMap<NodeId, u32> = HashMap::new();
    let mut pp: u32 = 0;
    for block in graph.blocks() {
        for &(id, _) in &block.nodes {
            def_point.insert(id, pp);
            pp += 1;
        }
        if block.control.is_some() {
            pp += 1; // control node occupies one slot
        }
    }

    // Pass 2: for each use of a NodeId record the program point and track the
    // farthest (last) use.
    let mut end_point: HashMap<NodeId, u32> = HashMap::new();

    let mut record_use = |id: NodeId, at: u32| {
        let e = end_point.entry(id).or_insert(at);
        if at > *e {
            *e = at;
        }
    };

    let mut pp: u32 = 0;
    for block in graph.blocks() {
        for (_, node) in &block.nodes {
            let node_pp = pp;
            collect_inputs(node, &mut |inp| record_use(inp, node_pp));
            pp += 1;
        }
        if let Some(ctrl) = &block.control {
            let ctrl_pp = pp;
            collect_control_inputs(ctrl, &mut |inp| record_use(inp, ctrl_pp));
            pp += 1;
        }
    }

    // Build the interval list from the definition map.
    def_point
        .into_iter()
        .map(|(id, start)| {
            // end is one past the last use; if never used, [start, start+1).
            let end = end_point.get(&id).map_or(start + 1, |&e| e + 1);
            LiveInterval { id, start, end }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Input-collection helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Call `f` for each [`NodeId`] operand of `node`.
#[allow(clippy::too_many_lines)]
fn collect_inputs(node: &ValueNode, f: &mut impl FnMut(NodeId)) {
    match node {
        // ── Nodes with no inputs ─────────────────────────────────────────────
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
        | ValueNode::LoadGlobal { .. }
        | ValueNode::LoadCurrentContextSlot { .. }
        | ValueNode::ArgumentsElements { .. }
        | ValueNode::RestElements { .. }
        | ValueNode::VirtualObject { .. }
        | ValueNode::CreateFunctionContext { .. }
        | ValueNode::CreateBlockContext { .. }
        | ValueNode::CreateShallowObjectLiteral { .. }
        | ValueNode::CreateShallowArrayLiteral { .. }
        | ValueNode::CreateObjectLiteral { .. }
        | ValueNode::CreateArrayLiteral { .. }
        | ValueNode::CreateEmptyObjectLiteral
        | ValueNode::CreateRegExpLiteral { .. }
        | ValueNode::CreateClosure { .. }
        | ValueNode::FastCreateClosure { .. }
        | ValueNode::GetTemplateObject { .. }
        | ValueNode::Debugger
        | ValueNode::Abort { .. } => {}

        // ── Single-input nodes ────────────────────────────────────────────────
        ValueNode::GetArgument { index } => f(*index),

        ValueNode::CheckedSmiIncrement { value }
        | ValueNode::CheckedSmiDecrement { value }
        | ValueNode::Int32Negate { value }
        | ValueNode::Int32Increment { value }
        | ValueNode::Int32Decrement { value }
        | ValueNode::Float64Negate { value }
        | ValueNode::Float64Ieee754Unary { value, .. }
        | ValueNode::GenericBitwiseNot { value, .. }
        | ValueNode::GenericNegate { value, .. }
        | ValueNode::GenericIncrement { value, .. }
        | ValueNode::GenericDecrement { value, .. }
        | ValueNode::ChangeInt32ToFloat64 { input: value }
        | ValueNode::ChangeUint32ToFloat64 { input: value }
        | ValueNode::ChangeFloat64ToInt32 { input: value }
        | ValueNode::CheckedFloat64ToInt32 { input: value }
        | ValueNode::ChangeInt32ToTagged { input: value }
        | ValueNode::ChangeUint32ToTagged { input: value }
        | ValueNode::ChangeFloat64ToTagged { input: value }
        | ValueNode::ChangeTaggedToInt32 { input: value }
        | ValueNode::ChangeTaggedToUint32 { input: value }
        | ValueNode::ChangeTaggedToFloat64 { input: value }
        | ValueNode::CheckedTaggedToInt32 { input: value }
        | ValueNode::CheckedTaggedToFloat64 { input: value }
        | ValueNode::ToBoolean { value }
        | ValueNode::ToString { value, .. }
        | ValueNode::ToObject { value, .. }
        | ValueNode::ToName { value, .. }
        | ValueNode::ToNumber { value, .. }
        | ValueNode::ToNumberOrNumeric { value, .. }
        | ValueNode::TypeOf { value }
        | ValueNode::NumberToString { value, .. }
        | ValueNode::TestUndetectable { value }
        | ValueNode::TestTypeOf { value, .. } => f(*value),

        ValueNode::CheckSmi { receiver }
        | ValueNode::CheckNumber { receiver }
        | ValueNode::CheckHeapObject { receiver }
        | ValueNode::CheckSymbol { receiver }
        | ValueNode::CheckString { receiver }
        | ValueNode::CheckStringOrStringWrapper { receiver }
        | ValueNode::CheckSeqOneByteString { receiver }
        | ValueNode::CheckMaps { receiver, .. }
        | ValueNode::CheckMapsWithMigration { receiver, .. }
        | ValueNode::CheckValue { receiver, .. } => f(*receiver),

        ValueNode::CheckDynamicValue { receiver, expected } => {
            f(*receiver);
            f(*expected);
        }

        ValueNode::CheckInt32IsSmi { input }
        | ValueNode::CheckUint32IsSmi { input }
        | ValueNode::CheckHoleyFloat64IsSmi { input }
        | ValueNode::CheckFloat64IsNan { input } => f(*input),

        ValueNode::LoadField { object, .. }
        | ValueNode::LoadTaggedField { object, .. }
        | ValueNode::LoadDoubleField { object, .. }
        | ValueNode::LoadNamedGeneric { object, .. }
        | ValueNode::ForInPrepare {
            enumerator: object, ..
        }
        | ValueNode::StringLength { string: object } => f(*object),

        ValueNode::LoadEnumCacheLength { map } => f(*map),

        ValueNode::LoadKeyedGeneric { object, key, .. } => {
            f(*object);
            f(*key);
        }

        ValueNode::HasInPrototypeChain { object, prototype } => {
            f(*object);
            f(*prototype);
        }

        ValueNode::StoreField { object, value, .. } => {
            f(*object);
            f(*value);
        }

        ValueNode::StoreCurrentContextSlot { value, .. } | ValueNode::StoreGlobal { value, .. } => {
            f(*value);
        }

        ValueNode::LoadContextSlot { context, .. } => f(*context),
        ValueNode::StoreContextSlot { context, value, .. } => {
            f(*context);
            f(*value);
        }

        ValueNode::PushContext { context } | ValueNode::PopContext { context } => f(*context),

        ValueNode::LoadFixedArrayElement { elements, index }
        | ValueNode::LoadFixedDoubleArrayElement { elements, index }
        | ValueNode::LoadHoleyFixedDoubleArrayElement { elements, index } => {
            f(*elements);
            f(*index);
        }

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
            f(*elements);
            f(*index);
            f(*value);
        }

        ValueNode::StoreNamedGeneric { object, value, .. } => {
            f(*object);
            f(*value);
        }

        ValueNode::StoreKeyedGeneric {
            object, key, value, ..
        } => {
            f(*object);
            f(*key);
            f(*value);
        }

        // ── Binary nodes ──────────────────────────────────────────────────────
        ValueNode::CheckedSmiAdd { left, right }
        | ValueNode::CheckedSmiSubtract { left, right }
        | ValueNode::CheckedSmiMultiply { left, right }
        | ValueNode::CheckedSmiDivide { left, right }
        | ValueNode::CheckedSmiModulus { left, right }
        | ValueNode::Int32Add { left, right }
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
        | ValueNode::Int32Equal { left, right }
        | ValueNode::Int32StrictEqual { left, right }
        | ValueNode::Int32LessThan { left, right }
        | ValueNode::Int32LessThanOrEqual { left, right }
        | ValueNode::Int32GreaterThan { left, right }
        | ValueNode::Int32GreaterThanOrEqual { left, right }
        | ValueNode::Float64Equal { left, right }
        | ValueNode::Float64LessThan { left, right }
        | ValueNode::Float64LessThanOrEqual { left, right }
        | ValueNode::Float64GreaterThan { left, right }
        | ValueNode::Float64GreaterThanOrEqual { left, right }
        | ValueNode::StringConcat { left, right }
        | ValueNode::StringEqual { left, right }
        | ValueNode::GenericAdd { left, right, .. }
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
            f(*left);
            f(*right);
        }

        ValueNode::CheckInt32Condition { left, right, .. } => {
            f(*left);
            f(*right);
        }

        ValueNode::CheckCacheIndicesNotCleared { receiver, indices } => {
            f(*receiver);
            f(*indices);
        }

        ValueNode::TestInstanceOf {
            object, callable, ..
        } => {
            f(*object);
            f(*callable);
        }
        ValueNode::TestIn { key, object, .. } => {
            f(*key);
            f(*object);
        }

        ValueNode::StringAt { string, index } => {
            f(*string);
            f(*index);
        }

        ValueNode::ForInNext {
            receiver,
            cache_index,
            cache_array,
            ..
        } => {
            f(*receiver);
            f(*cache_index);
            f(*cache_array);
        }

        ValueNode::DeleteProperty { object, key, .. } => {
            f(*object);
            f(*key);
        }

        ValueNode::CreateCatchContext { exception, .. } => f(*exception),
        ValueNode::CreateWithContext { object, .. } => f(*object),

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
        }
        | ValueNode::CallKnownFunction {
            callee,
            receiver,
            args,
        } => {
            f(*callee);
            f(*receiver);
            for &a in args {
                f(a);
            }
        }

        ValueNode::CallBuiltin { args, .. } | ValueNode::CallRuntime { args, .. } => {
            for &a in args {
                f(a);
            }
        }

        ValueNode::Construct {
            constructor, args, ..
        }
        | ValueNode::ConstructWithSpread {
            constructor, args, ..
        } => {
            f(*constructor);
            for &a in args {
                f(a);
            }
        }

        ValueNode::Phi { inputs } => {
            for &inp in inputs {
                f(inp);
            }
        }
    }
}

/// Call `f` for each [`NodeId`] operand of `ctrl`.
fn collect_control_inputs(ctrl: &ControlNode, f: &mut impl FnMut(NodeId)) {
    match ctrl {
        ControlNode::Return { value } => f(*value),
        ControlNode::Branch { condition, .. } => f(*condition),
        ControlNode::Jump { .. } | ControlNode::Deoptimize { .. } => {}
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear-scan register allocation
// ─────────────────────────────────────────────────────────────────────────────

/// Run linear-scan register allocation over `graph` using `num_regs` physical
/// registers.
///
/// Values whose live intervals cannot be assigned a register are *spilled* to
/// numbered stack slots.  The algorithm follows Poletto & Sarkar (1999):
/// intervals are processed in start-point order; when all registers are in use
/// the active interval with the farthest endpoint is spilled if it ends later
/// than the current interval, otherwise the current interval is spilled.
///
/// # Loop awareness
///
/// The current implementation does **not** weight spill decisions by loop
/// depth.  Values live across a loop back-edge naturally have long intervals
/// and are therefore *more* likely to be spilled — the opposite of what we
/// want.  A future improvement would detect loop back-edges, compute loop
/// nesting depth for each interval, and multiply the spill cost by the depth
/// so that loop-live values are preferentially kept in registers.
///
/// # Panics
///
/// Panics if `num_regs` is `0`.
pub fn allocate(graph: &MaglevGraph, num_regs: u32) -> AllocationResult {
    assert!(num_regs > 0, "allocate: num_regs must be > 0");

    let mut intervals = compute_live_intervals(graph);
    // Sort by start point; break ties by NodeId for determinism.
    intervals.sort_by_key(|iv| (iv.start, iv.id.0));

    let mut assignments: HashMap<NodeId, Location> = HashMap::new();
    // Active intervals: (register, interval).
    let mut active: Vec<(u32, LiveInterval)> = Vec::new();
    let mut next_spill: u32 = 0;
    // Free-register pool (lower-numbered registers are preferred).
    let mut free_regs: Vec<u32> = (0..num_regs).rev().collect();

    for iv in &intervals {
        // Expire intervals whose live range ends at or before this start point,
        // reclaiming their registers.
        let mut still_active: Vec<(u32, LiveInterval)> = Vec::new();
        for (reg, active_iv) in active.drain(..) {
            if active_iv.end <= iv.start {
                free_regs.push(reg);
            } else {
                still_active.push((reg, active_iv));
            }
        }
        active = still_active;

        if let Some(reg) = free_regs.pop() {
            // A free register is available — assign it.
            assignments.insert(iv.id, Location::Register(reg));
            active.push((reg, *iv));
        } else {
            // All registers occupied — spill the interval with the farthest
            // endpoint.  If that interval ends later than the current one,
            // evict it and give its register to the current interval; otherwise
            // spill the current interval.
            let spill_idx = active
                .iter()
                .enumerate()
                .max_by_key(|(_, (_, a))| a.end)
                .map(|(i, _)| i);

            // active is non-empty because free_regs is empty and the
            // invariant active.len() + free_regs.len() == num_regs holds.
            let idx = spill_idx.expect("active must be non-empty when no free regs");
            if active[idx].1.end > iv.end {
                // Evict the farthest-ending active interval.
                let (reg, spilled_iv) = active.remove(idx);
                let slot = next_spill;
                next_spill += 1;
                assignments.insert(spilled_iv.id, Location::StackSlot(slot));
                assignments.insert(iv.id, Location::Register(reg));
                active.push((reg, *iv));
            } else {
                // Spill the current interval.
                let slot = next_spill;
                next_spill += 1;
                assignments.insert(iv.id, Location::StackSlot(slot));
            }
        }
    }

    // ── Per-node caller-saved liveness ──────────────────────────────────────
    //
    // For each node at program point P, compute a bitmask of caller-saved
    // register indices (1–5) whose live intervals span *across* P, i.e. the
    // value is defined before P and still needed after P.  This lets codegen
    // save only the registers that are truly live at each stub-call site.

    // Collect intervals that ended up in caller-saved registers.
    let cs_intervals: Vec<(u32, u32, u8)> = intervals
        .iter()
        .filter_map(|iv| {
            if let Some(Location::Register(n)) = assignments.get(&iv.id)
                && (1..=5).contains(n)
            {
                return Some((iv.start, iv.end, 1u8 << n));
            }
            None
        })
        .collect();

    let mut live_caller_saved: HashMap<NodeId, u8> = HashMap::new();

    if !cs_intervals.is_empty() {
        // Rebuild the program-point map (same iteration order as
        // compute_live_intervals).
        let mut pp: u32 = 0;
        for block in graph.blocks() {
            for &(nid, _) in &block.nodes {
                let mut mask: u8 = 0;
                for &(start, end, bit) in &cs_intervals {
                    // The value is live *across* this node when it was
                    // defined strictly before and its last use is strictly
                    // after this program point.
                    if start < pp && end > pp + 1 {
                        mask |= bit;
                    }
                }
                if mask != 0 {
                    live_caller_saved.insert(nid, mask);
                }
                pp += 1;
            }
            if block.control.is_some() {
                pp += 1;
            }
        }
    }

    AllocationResult {
        assignments,
        spill_count: next_spill,
        live_caller_saved,
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

    /// Assert that no two overlapping live intervals are assigned the same
    /// physical register.
    fn assert_no_conflicts(graph: &MaglevGraph, result: &AllocationResult, num_regs: u32) {
        let intervals = compute_live_intervals(graph);
        for i in 0..intervals.len() {
            for j in (i + 1)..intervals.len() {
                let a = &intervals[i];
                let b = &intervals[j];
                // Intervals overlap when a.start < b.end && b.start < a.end.
                if a.start < b.end && b.start < a.end {
                    if let (Some(Location::Register(ra)), Some(Location::Register(rb))) =
                        (result.location(a.id), result.location(b.id))
                    {
                        assert_ne!(
                            ra, rb,
                            "register conflict: {:?} and {:?} both in register {} \
                             (intervals [{},{}) and [{},{}) with {} regs)",
                            a.id, b.id, ra, a.start, a.end, b.start, b.end, num_regs
                        );
                    }
                }
            }
        }
    }

    // ── Test: single parameter gets a register ────────────────────────────────

    #[test]
    fn test_single_parameter_gets_register() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        block.set_control(ControlNode::Return { value: p0 });
        graph.add_block(block);

        let result = allocate(&graph, 8);
        assert!(matches!(result.location(p0), Some(Location::Register(_))));
        assert_eq!(result.spill_count(), 0);
        assert_no_conflicts(&graph, &result, 8);
    }

    // ── Test: two independent constants, both get registers ───────────────────

    #[test]
    fn test_two_independent_values_no_spill() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c1 = block.push_value(ValueNode::Int32Constant { value: 1 });
        let c2 = block.push_value(ValueNode::Int32Constant { value: 2 });
        let add = block.push_value(ValueNode::Int32Add {
            left: c1,
            right: c2,
        });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        let result = allocate(&graph, 8);
        assert_no_conflicts(&graph, &result, 8);
        assert_eq!(result.spill_count(), 0);
    }

    // ── Test: forced spill with only 1 register available ────────────────────

    #[test]
    fn test_spill_with_one_register() {
        // p0 and p1 are both live at the Int32Add node; with 1 register one
        // of them must be spilled.
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

        let result = allocate(&graph, 1);
        assert_no_conflicts(&graph, &result, 1);
        assert!(result.spill_count() > 0, "expected at least one spill");
    }

    // ── Test: two-block graph with unconditional jump ─────────────────────────

    #[test]
    fn test_two_block_graph() {
        let mut graph = MaglevGraph::new(1);

        // block 0: define parameter, jump to block 1
        let mut b0 = BasicBlock::new(0);
        let p0 = b0.push_value(ValueNode::Parameter { index: 0 });
        b0.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b0);

        // block 1: return the parameter defined in block 0
        let mut b1 = BasicBlock::new(1);
        b1.set_control(ControlNode::Return { value: p0 });
        graph.add_block(b1);

        let result = allocate(&graph, 4);
        assert_no_conflicts(&graph, &result, 4);
        assert!(matches!(result.location(p0), Some(Location::Register(_))));
    }

    // ── Test: branch graph ────────────────────────────────────────────────────

    #[test]
    fn test_branch_graph() {
        let mut graph = MaglevGraph::new(1);

        // block 0: condition based on parameter; branch to block 1 or block 2
        let mut b0 = BasicBlock::new(0);
        let p0 = b0.push_value(ValueNode::Parameter { index: 0 });
        let cond = b0.push_value(ValueNode::ToBoolean { value: p0 });
        b0.set_control(ControlNode::Branch {
            condition: cond,
            if_true: 1,
            if_false: 2,
        });
        graph.add_block(b0);

        // block 1: return TrueConstant
        let mut b1 = BasicBlock::new(1);
        let t = b1.push_value(ValueNode::TrueConstant);
        b1.set_control(ControlNode::Return { value: t });
        graph.add_block(b1);

        // block 2: return FalseConstant
        let mut b2 = BasicBlock::new(2);
        let f = b2.push_value(ValueNode::FalseConstant);
        b2.set_control(ControlNode::Return { value: f });
        graph.add_block(b2);

        let result = allocate(&graph, 4);
        assert_no_conflicts(&graph, &result, 4);
    }

    // ── Test: chain of unary ops — at most 2 values live at once ─────────────

    #[test]
    fn test_no_spills_with_sufficient_registers() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c0 = block.push_value(ValueNode::Int32Constant { value: 42 });
        let neg0 = block.push_value(ValueNode::Int32Negate { value: c0 });
        let neg1 = block.push_value(ValueNode::Int32Negate { value: neg0 });
        let neg2 = block.push_value(ValueNode::Int32Negate { value: neg1 });
        block.set_control(ControlNode::Return { value: neg2 });
        graph.add_block(block);

        // At most 2 values are live simultaneously along this chain.
        let result = allocate(&graph, 4);
        assert_eq!(result.spill_count(), 0);
        assert_no_conflicts(&graph, &result, 4);
    }

    // ── Test: every node receives a location ─────────────────────────────────

    #[test]
    fn test_all_nodes_get_location() {
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

        let result = allocate(&graph, 2);
        assert!(result.location(p0).is_some(), "p0 missing location");
        assert!(result.location(p1).is_some(), "p1 missing location");
        assert!(result.location(add).is_some(), "add missing location");
        assert_no_conflicts(&graph, &result, 2);
    }

    // ── Test: Phi node — inputs collected across blocks ───────────────────────

    #[test]
    fn test_phi_node_allocation() {
        let mut graph = MaglevGraph::new(1);

        // block 0: branch on parameter
        let mut b0 = BasicBlock::new(0);
        let p0 = b0.push_value(ValueNode::Parameter { index: 0 });
        let cond = b0.push_value(ValueNode::ToBoolean { value: p0 });
        b0.set_control(ControlNode::Branch {
            condition: cond,
            if_true: 1,
            if_false: 2,
        });
        graph.add_block(b0);

        // block 1: produce value 1
        let mut b1 = BasicBlock::new(1);
        let v1 = b1.push_value(ValueNode::Int32Constant { value: 1 });
        b1.set_control(ControlNode::Jump { target: 3 });
        graph.add_block(b1);

        // block 2: produce value 2
        let mut b2 = BasicBlock::new(2);
        let v2 = b2.push_value(ValueNode::Int32Constant { value: 2 });
        b2.set_control(ControlNode::Jump { target: 3 });
        graph.add_block(b2);

        // block 3: phi merging v1 and v2, then return
        let mut b3 = BasicBlock::new(3);
        let phi = b3.push_value(ValueNode::Phi {
            inputs: vec![v1, v2],
        });
        b3.set_control(ControlNode::Return { value: phi });
        graph.add_block(b3);

        let result = allocate(&graph, 4);
        assert_no_conflicts(&graph, &result, 4);
        assert!(result.location(phi).is_some(), "phi missing location");
    }
}
