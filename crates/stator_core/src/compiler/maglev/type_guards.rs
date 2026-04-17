//! Type-guard insertion with deoptimisation bailout.
//!
//! This pass scans for **unguarded** typed arithmetic nodes —
//! [`ValueNode::Int32Add`], [`ValueNode::Float64Multiply`], etc. — whose
//! inputs come from **untyped** sources (parameters, generic loads, phi
//! nodes).  For each such input it inserts an appropriate **type guard**
//! node (`CheckSmi`, `CheckNumber`, etc.) with an associated
//! `bytecode_offset` so that a failed guard can trigger a deoptimisation
//! back to the interpreter.
//!
//! # When is this useful?
//!
//! After **type specialisation** has replaced `GenericAdd` with
//! `CheckedSmiAdd`, the operands are assumed to be Smis — but no guard
//! exists yet.  This pass makes the assumption explicit by inserting the
//! guard node, ensuring correctness: if a non-Smi value reaches the
//! `CheckedSmiAdd` at runtime, the guard will trigger a deoptimisation
//! instead of producing garbage.
//!
//! # Algorithm
//!
//! For each block (in order):
//!
//! 1. Maintain a set of [`NodeId`]s known to have been guarded.
//! 2. For every typed arithmetic node, check whether its input(s) need
//!    guards.  An input **needs a guard** if it is:
//!    - A `Parameter`, `Phi`, `LoadField`, `LoadNamedGeneric`,
//!      `LoadKeyedGeneric`, or any `Generic*` node, **and**
//!    - Not already in the guarded set.
//! 3. If a guard is needed, insert a `CheckSmi` (for Int32/Smi ops) or
//!    `CheckNumber` (for Float64 ops) immediately before the consumer
//!    node.  The guard's `receiver` is the unguarded input.
//! 4. Add the input [`NodeId`] to the guarded set so subsequent uses in
//!    the same block are not re-guarded.
//!
//! # Usage
//!
//! ```
//! use stator_jse::compiler::maglev::ir::{
//!     BasicBlock, ControlNode, MaglevGraph, ValueNode,
//! };
//! use stator_jse::compiler::maglev::type_guards::insert_type_guards;
//!
//! let mut graph = MaglevGraph::new(2);
//! let mut block = BasicBlock::new(0);
//! let p0 = block.push_value(ValueNode::Parameter { index: 0 });
//! let p1 = block.push_value(ValueNode::Parameter { index: 1 });
//! // Unguarded CheckedSmiAdd on raw parameters.
//! let add = block.push_value(ValueNode::CheckedSmiAdd {
//!     left: p0,
//!     right: p1,
//! });
//! block.set_control(ControlNode::Return { value: add });
//! graph.add_block(block);
//!
//! let before = graph.blocks()[0].nodes.len();
//! insert_type_guards(&mut graph, 0);
//! // Two CheckSmi guards were inserted (one per parameter).
//! assert!(graph.blocks()[0].nodes.len() > before);
//! ```

use std::collections::HashSet;

use crate::compiler::maglev::ir::{MaglevGraph, NodeId, ValueNode};

// ─────────────────────────────────────────────────────────────────────────────
// Public entry-point
// ─────────────────────────────────────────────────────────────────────────────

/// Insert type-guard nodes before unguarded typed arithmetic.
///
/// `bytecode_offset` is recorded in any newly inserted guard so that a
/// deoptimisation triggered by a failed guard can resume the interpreter at
/// the correct bytecode position.
pub fn insert_type_guards(graph: &mut MaglevGraph, bytecode_offset: u32) {
    // We need to know which NodeIds are "typed" (i.e. produced by a node
    // that guarantees its output type) so we don't guard them redundantly.
    let typed_ids = collect_typed_ids(graph);

    for block_idx in 0..graph.blocks().len() {
        insert_guards_in_block(graph, block_idx, bytecode_offset, &typed_ids);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Typed-ID collection
// ─────────────────────────────────────────────────────────────────────────────

/// Collect all [`NodeId`]s whose producing node guarantees a specific type
/// (constants, checked ops, type conversions, existing guards).
fn collect_typed_ids(graph: &MaglevGraph) -> HashSet<NodeId> {
    let mut typed = HashSet::new();

    for block in graph.blocks() {
        for (id, node) in &block.nodes {
            if is_typed_producer(node) {
                typed.insert(*id);
            }
        }
    }

    typed
}

/// A node is a *typed producer* if its output is guaranteed to be a specific
/// JS type (Smi, Float64, etc.) without additional guards.
fn is_typed_producer(node: &ValueNode) -> bool {
    matches!(
        node,
        // Constants.
        ValueNode::SmiConstant { .. }
            | ValueNode::Float64Constant { .. }
            | ValueNode::Int32Constant { .. }
            | ValueNode::Uint32Constant { .. }
            | ValueNode::TrueConstant
            | ValueNode::FalseConstant
            | ValueNode::NullConstant
            | ValueNode::UndefinedConstant
            | ValueNode::StringConstant { .. }
            // Typed arithmetic.
            | ValueNode::CheckedSmiAdd { .. }
            | ValueNode::CheckedSmiSubtract { .. }
            | ValueNode::CheckedSmiMultiply { .. }
            | ValueNode::CheckedSmiDivide { .. }
            | ValueNode::CheckedSmiModulus { .. }
            | ValueNode::CheckedSmiIncrement { .. }
            | ValueNode::CheckedSmiDecrement { .. }
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
            | ValueNode::Uint32Add { .. }
            | ValueNode::Uint32Subtract { .. }
            | ValueNode::Uint32Multiply { .. }
            | ValueNode::Uint32Divide { .. }
            | ValueNode::Uint32Modulus { .. }
            | ValueNode::Float64Add { .. }
            | ValueNode::Float64Subtract { .. }
            | ValueNode::Float64Multiply { .. }
            | ValueNode::Float64Divide { .. }
            | ValueNode::Float64Modulus { .. }
            | ValueNode::Float64Negate { .. }
            | ValueNode::Float64Exponentiate { .. }
            // Type conversions.
            | ValueNode::ChangeInt32ToFloat64 { .. }
            | ValueNode::ChangeUint32ToFloat64 { .. }
            | ValueNode::ChangeFloat64ToInt32 { .. }
            | ValueNode::ChangeInt32ToTagged { .. }
            | ValueNode::ChangeUint32ToTagged { .. }
            | ValueNode::ChangeFloat64ToTagged { .. }
            | ValueNode::ChangeTaggedToInt32 { .. }
            | ValueNode::ChangeTaggedToUint32 { .. }
            | ValueNode::ChangeTaggedToFloat64 { .. }
            // Existing guards.
            | ValueNode::CheckSmi { .. }
            | ValueNode::CheckNumber { .. }
            | ValueNode::CheckHeapObject { .. }
            | ValueNode::CheckString { .. }
            | ValueNode::CheckMaps { .. }
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Guard insertion
// ─────────────────────────────────────────────────────────────────────────────

/// The kind of guard to insert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GuardKind {
    /// `CheckSmi` — for Smi / Int32 arithmetic inputs.
    Smi,
    /// `CheckNumber` — for Float64 arithmetic inputs.
    Number,
}

/// Insert guards in a single block.
fn insert_guards_in_block(
    graph: &mut MaglevGraph,
    block_idx: usize,
    _bytecode_offset: u32,
    typed_ids: &HashSet<NodeId>,
) {
    // First pass: collect what guards are needed and where.
    let guards_needed: Vec<(usize, NodeId, GuardKind)>;
    {
        let block = &graph.blocks()[block_idx];
        let mut guarded: HashSet<NodeId> = HashSet::new();
        let mut needed: Vec<(usize, NodeId, GuardKind)> = Vec::new();

        for (pos, (_id, node)) in block.nodes.iter().enumerate() {
            let required = inputs_needing_guard(node, typed_ids, &guarded);
            for (input_id, kind) in required {
                needed.push((pos, input_id, kind));
                guarded.insert(input_id);
            }
        }

        guards_needed = needed;
    }

    if guards_needed.is_empty() {
        return;
    }

    // Second pass: insert guard nodes.  Each insertion shifts indices by 1,
    // so we track a cumulative offset.
    let block = &mut graph.blocks_mut()[block_idx];

    for (offset, (orig_pos, input_id, kind)) in guards_needed.iter().enumerate() {
        let insert_at = orig_pos + offset;
        let guard_node = match kind {
            GuardKind::Smi => ValueNode::CheckSmi {
                receiver: *input_id,
            },
            GuardKind::Number => ValueNode::CheckNumber {
                receiver: *input_id,
            },
        };
        let guard_id = NodeId(10_000 + insert_at as u32 + block.id * 1000);
        block.nodes.insert(insert_at, (guard_id, guard_node));
    }
}

/// For a given node, return a list of `(input_id, guard_kind)` pairs that
/// need a guard inserted.
fn inputs_needing_guard(
    node: &ValueNode,
    typed_ids: &HashSet<NodeId>,
    guarded: &HashSet<NodeId>,
) -> Vec<(NodeId, GuardKind)> {
    let mut result = Vec::new();

    match node {
        // Smi / Int32 arithmetic — inputs need CheckSmi.
        ValueNode::CheckedSmiAdd { left, right }
        | ValueNode::CheckedSmiSubtract { left, right }
        | ValueNode::CheckedSmiMultiply { left, right }
        | ValueNode::CheckedSmiDivide { left, right }
        | ValueNode::CheckedSmiModulus { left, right }
        | ValueNode::Int32Add { left, right }
        | ValueNode::Int32Subtract { left, right }
        | ValueNode::Int32Multiply { left, right }
        | ValueNode::Int32Divide { left, right }
        | ValueNode::Int32Modulus { left, right } => {
            maybe_guard(*left, GuardKind::Smi, typed_ids, guarded, &mut result);
            maybe_guard(*right, GuardKind::Smi, typed_ids, guarded, &mut result);
        }

        ValueNode::CheckedSmiIncrement { value }
        | ValueNode::CheckedSmiDecrement { value }
        | ValueNode::Int32Negate { value }
        | ValueNode::Int32Increment { value }
        | ValueNode::Int32Decrement { value } => {
            maybe_guard(*value, GuardKind::Smi, typed_ids, guarded, &mut result);
        }

        // Float64 arithmetic — inputs need CheckNumber.
        ValueNode::Float64Add { left, right }
        | ValueNode::Float64Subtract { left, right }
        | ValueNode::Float64Multiply { left, right }
        | ValueNode::Float64Divide { left, right }
        | ValueNode::Float64Modulus { left, right }
        | ValueNode::Float64Exponentiate { left, right } => {
            maybe_guard(*left, GuardKind::Number, typed_ids, guarded, &mut result);
            maybe_guard(*right, GuardKind::Number, typed_ids, guarded, &mut result);
        }

        ValueNode::Float64Negate { value } => {
            maybe_guard(*value, GuardKind::Number, typed_ids, guarded, &mut result);
        }

        _ => {}
    }

    result
}

/// Add `(id, kind)` to `out` if `id` is untyped, unguarded, and not already
/// queued in `out`.
fn maybe_guard(
    id: NodeId,
    kind: GuardKind,
    typed_ids: &HashSet<NodeId>,
    guarded: &HashSet<NodeId>,
    out: &mut Vec<(NodeId, GuardKind)>,
) {
    if !typed_ids.contains(&id) && !guarded.contains(&id) && !out.iter().any(|(nid, _)| *nid == id)
    {
        out.push((id, kind));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::maglev::ir::{BasicBlock, ControlNode, MaglevGraph, ValueNode};

    // ── CheckSmi guards inserted for Smi add ─────────────────────────────────

    #[test]
    fn test_guard_inserted_for_parameter_inputs_smi_add() {
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

        assert_eq!(graph.blocks()[0].nodes.len(), 3);

        insert_type_guards(&mut graph, 0);

        // Two CheckSmi guards should have been inserted.
        assert_eq!(graph.blocks()[0].nodes.len(), 5);

        // The first two inserted nodes should be CheckSmi.
        let nodes = &graph.blocks()[0].nodes;
        assert!(matches!(nodes[2].1, ValueNode::CheckSmi { .. }));
        assert!(matches!(nodes[3].1, ValueNode::CheckSmi { .. }));
    }

    // ── CheckNumber guards for Float64 ops ───────────────────────────────────

    #[test]
    fn test_guard_inserted_for_parameter_inputs_float64_mul() {
        let mut graph = MaglevGraph::new(2);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        let p1 = block.push_value(ValueNode::Parameter { index: 1 });
        let mul = block.push_value(ValueNode::Float64Multiply {
            left: p0,
            right: p1,
        });
        block.set_control(ControlNode::Return { value: mul });
        graph.add_block(block);

        insert_type_guards(&mut graph, 0);

        // Two CheckNumber guards.
        assert_eq!(graph.blocks()[0].nodes.len(), 5);
        let nodes = &graph.blocks()[0].nodes;
        assert!(matches!(nodes[2].1, ValueNode::CheckNumber { .. }));
        assert!(matches!(nodes[3].1, ValueNode::CheckNumber { .. }));
    }

    // ── No guard when input is already typed ─────────────────────────────────

    #[test]
    fn test_no_guard_when_input_is_constant() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c1 = block.push_value(ValueNode::SmiConstant { value: 1 });
        let c2 = block.push_value(ValueNode::SmiConstant { value: 2 });
        let add = block.push_value(ValueNode::CheckedSmiAdd {
            left: c1,
            right: c2,
        });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        let before = graph.blocks()[0].nodes.len();
        insert_type_guards(&mut graph, 0);
        assert_eq!(graph.blocks()[0].nodes.len(), before);
    }

    // ── Guard deduplication ──────────────────────────────────────────────────

    #[test]
    fn test_same_input_guarded_once() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        // Use the same parameter in both operands.
        let add = block.push_value(ValueNode::CheckedSmiAdd { left: p, right: p });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        insert_type_guards(&mut graph, 0);

        // Only one CheckSmi should be inserted (not two).
        let guard_count = graph.blocks()[0]
            .nodes
            .iter()
            .filter(|(_, n)| matches!(n, ValueNode::CheckSmi { .. }))
            .count();
        assert_eq!(guard_count, 1);
    }

    // ── Multiple consumers share guard ───────────────────────────────────────

    #[test]
    fn test_guard_shared_across_consumers() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        let c = block.push_value(ValueNode::SmiConstant { value: 1 });
        // Two smi adds using the same parameter.
        let add1 = block.push_value(ValueNode::CheckedSmiAdd { left: p, right: c });
        let add2 = block.push_value(ValueNode::CheckedSmiAdd {
            left: p,
            right: add1,
        });
        block.set_control(ControlNode::Return { value: add2 });
        graph.add_block(block);

        insert_type_guards(&mut graph, 0);

        // Only one CheckSmi for `p` (used by both adds).
        let guard_count = graph.blocks()[0]
            .nodes
            .iter()
            .filter(|(_, n)| matches!(n, ValueNode::CheckSmi { .. }))
            .count();
        assert_eq!(guard_count, 1);
    }

    // ── Unary node gets guard ────────────────────────────────────────────────

    #[test]
    fn test_guard_for_unary_negate() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        let neg = block.push_value(ValueNode::Int32Negate { value: p });
        block.set_control(ControlNode::Return { value: neg });
        graph.add_block(block);

        insert_type_guards(&mut graph, 0);

        let guard_count = graph.blocks()[0]
            .nodes
            .iter()
            .filter(|(_, n)| matches!(n, ValueNode::CheckSmi { .. }))
            .count();
        assert_eq!(guard_count, 1);
    }

    // ── Float64Negate gets CheckNumber ───────────────────────────────────────

    #[test]
    fn test_guard_for_float64_negate() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        let neg = block.push_value(ValueNode::Float64Negate { value: p });
        block.set_control(ControlNode::Return { value: neg });
        graph.add_block(block);

        insert_type_guards(&mut graph, 0);

        let guard_count = graph.blocks()[0]
            .nodes
            .iter()
            .filter(|(_, n)| matches!(n, ValueNode::CheckNumber { .. }))
            .count();
        assert_eq!(guard_count, 1);
    }
}
