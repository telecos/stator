//! Type specialisation from inline-cache (IC) feedback.
//!
//! When the interpreter collects runtime type information via inline caches,
//! the feedback vector records which types were observed for each feedback
//! slot.  This pass walks the graph and replaces **generic** (slow-path)
//! arithmetic nodes — [`ValueNode::GenericAdd`], [`ValueNode::GenericSubtract`],
//! etc. — with their **typed** (fast-path) equivalents when the feedback
//! indicates a monomorphic or narrow polymorphic type profile.
//!
//! # Algorithm
//!
//! For each generic arithmetic node whose `feedback_slot` is present in the
//! supplied [`FeedbackMap`]:
//!
//! 1. Look up the observed [`SpeculatedType`] for that slot.
//! 2. If the type is [`SpeculatedType::Smi`], replace the generic node with
//!    the corresponding `CheckedSmi*` variant (which will deopt on overflow).
//! 3. If the type is [`SpeculatedType::Float64`], replace with the unboxed
//!    `Float64*` variant.
//! 4. If the type is [`SpeculatedType::String`], replace binary `GenericAdd`
//!    with [`ValueNode::StringConcat`].
//!
//! Unary generics (`GenericNegate`, `GenericIncrement`, `GenericDecrement`)
//! are similarly specialised.
//!
//! # Usage
//!
//! ```
//! use stator_js::compiler::maglev::ir::{
//!     BasicBlock, ControlNode, MaglevGraph, NodeId, ValueNode,
//! };
//! use stator_js::compiler::maglev::type_specialization::{
//!     specialize_types, FeedbackMap, SpeculatedType,
//! };
//!
//! let mut graph = MaglevGraph::new(1);
//! let mut block = BasicBlock::new(0);
//! let p0 = block.push_value(ValueNode::Parameter { index: 0 });
//! let p1 = block.push_value(ValueNode::Parameter { index: 1 });
//! let add = block.push_value(ValueNode::GenericAdd {
//!     left: p0,
//!     right: p1,
//!     feedback_slot: 42,
//! });
//! block.set_control(ControlNode::Return { value: add });
//! graph.add_block(block);
//!
//! let mut feedback = FeedbackMap::new();
//! feedback.insert(42, SpeculatedType::Smi);
//!
//! specialize_types(&mut graph, &feedback);
//!
//! // The GenericAdd has been replaced with a CheckedSmiAdd.
//! let node = &graph.blocks()[0].nodes[2].1;
//! assert!(matches!(node, ValueNode::CheckedSmiAdd { .. }));
//! ```

use std::collections::HashMap;

use crate::compiler::maglev::ir::{MaglevGraph, ValueNode};

// ─────────────────────────────────────────────────────────────────────────────
// Feedback types
// ─────────────────────────────────────────────────────────────────────────────

/// The speculated runtime type observed at a particular IC feedback slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpeculatedType {
    /// The value was always a tagged small integer (Smi).
    Smi,
    /// The value was always a heap number (Float64).
    Float64,
    /// The value was always a string.
    String,
}

/// Maps a feedback-slot index to the observed [`SpeculatedType`].
pub type FeedbackMap = HashMap<u32, SpeculatedType>;

// ─────────────────────────────────────────────────────────────────────────────
// Public entry-point
// ─────────────────────────────────────────────────────────────────────────────

/// Replace generic arithmetic nodes with typed specialisations where IC
/// feedback is available.
///
/// Nodes whose `feedback_slot` does not appear in `feedback` are left
/// untouched (they remain on the generic slow-path).
pub fn specialize_types(graph: &mut MaglevGraph, feedback: &FeedbackMap) {
    if feedback.is_empty() {
        return;
    }
    for block in graph.blocks_mut() {
        for (_id, node) in &mut block.nodes {
            specialize_node(node, feedback);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-node specialisation
// ─────────────────────────────────────────────────────────────────────────────

/// Attempt to specialise a single generic node.
fn specialize_node(node: &mut ValueNode, feedback: &FeedbackMap) {
    let replacement = match node {
        // ── Binary generic arithmetic ────────────────────────────────────
        ValueNode::GenericAdd {
            left,
            right,
            feedback_slot,
        } => specialize_binary_add(*left, *right, *feedback_slot, feedback),

        ValueNode::GenericSubtract {
            left,
            right,
            feedback_slot,
        } => specialize_binary(*left, *right, *feedback_slot, feedback, BinOp::Sub),

        ValueNode::GenericMultiply {
            left,
            right,
            feedback_slot,
        } => specialize_binary(*left, *right, *feedback_slot, feedback, BinOp::Mul),

        ValueNode::GenericDivide {
            left,
            right,
            feedback_slot,
        } => specialize_binary(*left, *right, *feedback_slot, feedback, BinOp::Div),

        ValueNode::GenericModulus {
            left,
            right,
            feedback_slot,
        } => specialize_binary(*left, *right, *feedback_slot, feedback, BinOp::Mod),

        ValueNode::GenericExponentiate {
            left,
            right,
            feedback_slot,
        } => specialize_binary_f64_only(*left, *right, *feedback_slot, feedback, BinF64Op::Exp),

        ValueNode::GenericBitwiseAnd {
            left,
            right,
            feedback_slot,
        } => specialize_bitwise(*left, *right, *feedback_slot, feedback, BitwiseOp::And),

        ValueNode::GenericBitwiseOr {
            left,
            right,
            feedback_slot,
        } => specialize_bitwise(*left, *right, *feedback_slot, feedback, BitwiseOp::Or),

        ValueNode::GenericBitwiseXor {
            left,
            right,
            feedback_slot,
        } => specialize_bitwise(*left, *right, *feedback_slot, feedback, BitwiseOp::Xor),

        ValueNode::GenericShiftLeft {
            left,
            right,
            feedback_slot,
        } => specialize_bitwise(*left, *right, *feedback_slot, feedback, BitwiseOp::Shl),

        ValueNode::GenericShiftRight {
            left,
            right,
            feedback_slot,
        } => specialize_bitwise(*left, *right, *feedback_slot, feedback, BitwiseOp::Shr),

        ValueNode::GenericShiftRightLogical {
            left,
            right,
            feedback_slot,
        } => specialize_bitwise(*left, *right, *feedback_slot, feedback, BitwiseOp::Ushr),

        // ── Unary generic arithmetic ─────────────────────────────────────
        ValueNode::GenericNegate {
            value,
            feedback_slot,
        } => specialize_unary_negate(*value, *feedback_slot, feedback),

        ValueNode::GenericIncrement {
            value,
            feedback_slot,
        } => specialize_unary_inc_dec(*value, *feedback_slot, feedback, true),

        ValueNode::GenericDecrement {
            value,
            feedback_slot,
        } => specialize_unary_inc_dec(*value, *feedback_slot, feedback, false),

        _ => None,
    };

    if let Some(r) = replacement {
        *node = r;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

use crate::compiler::maglev::ir::NodeId;

/// Arithmetic binary operations that have both Smi and Float64 specialisations.
enum BinOp {
    Sub,
    Mul,
    Div,
    Mod,
}

/// Binary operations that only have a Float64 specialisation.
enum BinF64Op {
    Exp,
}

/// Bitwise binary operations (always specialise to Int32).
enum BitwiseOp {
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Ushr,
}

/// Specialise `GenericAdd`.  Strings get `StringConcat`; Smi/Float64 follow
/// the normal numeric path.
fn specialize_binary_add(
    left: NodeId,
    right: NodeId,
    slot: u32,
    feedback: &FeedbackMap,
) -> Option<ValueNode> {
    match feedback.get(&slot)? {
        SpeculatedType::Smi => Some(ValueNode::CheckedSmiAdd { left, right }),
        SpeculatedType::Float64 => Some(ValueNode::Float64Add { left, right }),
        SpeculatedType::String => Some(ValueNode::StringConcat { left, right }),
    }
}

/// Specialise a numeric binary operation (Sub / Mul / Div / Mod).
fn specialize_binary(
    left: NodeId,
    right: NodeId,
    slot: u32,
    feedback: &FeedbackMap,
    op: BinOp,
) -> Option<ValueNode> {
    match feedback.get(&slot)? {
        SpeculatedType::Smi => Some(match op {
            BinOp::Sub => ValueNode::CheckedSmiSubtract { left, right },
            BinOp::Mul => ValueNode::CheckedSmiMultiply { left, right },
            BinOp::Div => ValueNode::CheckedSmiDivide { left, right },
            BinOp::Mod => ValueNode::CheckedSmiModulus { left, right },
        }),
        SpeculatedType::Float64 => Some(match op {
            BinOp::Sub => ValueNode::Float64Subtract { left, right },
            BinOp::Mul => ValueNode::Float64Multiply { left, right },
            BinOp::Div => ValueNode::Float64Divide { left, right },
            BinOp::Mod => ValueNode::Float64Modulus { left, right },
        }),
        SpeculatedType::String => None,
    }
}

/// Specialise a binary operation that only has a Float64 fast-path (e.g.
/// exponentiation).
fn specialize_binary_f64_only(
    left: NodeId,
    right: NodeId,
    slot: u32,
    feedback: &FeedbackMap,
    op: BinF64Op,
) -> Option<ValueNode> {
    match feedback.get(&slot)? {
        SpeculatedType::Float64 => Some(match op {
            BinF64Op::Exp => ValueNode::Float64Exponentiate { left, right },
        }),
        _ => None,
    }
}

/// Specialise bitwise binary operations (always map to Int32 variants when
/// feedback says Smi, since bitwise ops coerce to i32).
fn specialize_bitwise(
    left: NodeId,
    right: NodeId,
    slot: u32,
    feedback: &FeedbackMap,
    op: BitwiseOp,
) -> Option<ValueNode> {
    match feedback.get(&slot)? {
        SpeculatedType::Smi => Some(match op {
            BitwiseOp::And => ValueNode::Int32BitwiseAnd { left, right },
            BitwiseOp::Or => ValueNode::Int32BitwiseOr { left, right },
            BitwiseOp::Xor => ValueNode::Int32BitwiseXor { left, right },
            BitwiseOp::Shl => ValueNode::Int32ShiftLeft { left, right },
            BitwiseOp::Shr => ValueNode::Int32ShiftRight { left, right },
            BitwiseOp::Ushr => ValueNode::Int32ShiftRightLogical { left, right },
        }),
        _ => None,
    }
}

/// Specialise `GenericNegate`.
fn specialize_unary_negate(value: NodeId, slot: u32, feedback: &FeedbackMap) -> Option<ValueNode> {
    match feedback.get(&slot)? {
        SpeculatedType::Smi => Some(ValueNode::Int32Negate { value }),
        SpeculatedType::Float64 => Some(ValueNode::Float64Negate { value }),
        SpeculatedType::String => None,
    }
}

/// Specialise `GenericIncrement` / `GenericDecrement`.
fn specialize_unary_inc_dec(
    value: NodeId,
    slot: u32,
    feedback: &FeedbackMap,
    is_increment: bool,
) -> Option<ValueNode> {
    match feedback.get(&slot)? {
        SpeculatedType::Smi => Some(if is_increment {
            ValueNode::CheckedSmiIncrement { value }
        } else {
            ValueNode::CheckedSmiDecrement { value }
        }),
        SpeculatedType::Float64 => {
            // No dedicated Float64 inc/dec — leave generic.
            None
        }
        SpeculatedType::String => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::maglev::ir::{BasicBlock, ControlNode, MaglevGraph, ValueNode};

    /// Build a two-parameter graph with a single generic binary node.
    fn binary_graph(node_fn: impl FnOnce(NodeId, NodeId) -> ValueNode) -> MaglevGraph {
        let mut graph = MaglevGraph::new(2);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        let p1 = block.push_value(ValueNode::Parameter { index: 1 });
        let op = block.push_value(node_fn(p0, p1));
        block.set_control(ControlNode::Return { value: op });
        graph.add_block(block);
        graph
    }

    // ── GenericAdd → CheckedSmiAdd ───────────────────────────────────────────

    #[test]
    fn test_specialize_generic_add_to_smi() {
        let mut graph = binary_graph(|l, r| ValueNode::GenericAdd {
            left: l,
            right: r,
            feedback_slot: 1,
        });
        let mut fb = FeedbackMap::new();
        fb.insert(1, SpeculatedType::Smi);

        specialize_types(&mut graph, &fb);

        assert!(matches!(
            &graph.blocks()[0].nodes[2].1,
            ValueNode::CheckedSmiAdd { .. }
        ));
    }

    // ── GenericAdd → Float64Add ──────────────────────────────────────────────

    #[test]
    fn test_specialize_generic_add_to_float64() {
        let mut graph = binary_graph(|l, r| ValueNode::GenericAdd {
            left: l,
            right: r,
            feedback_slot: 1,
        });
        let mut fb = FeedbackMap::new();
        fb.insert(1, SpeculatedType::Float64);

        specialize_types(&mut graph, &fb);

        assert!(matches!(
            &graph.blocks()[0].nodes[2].1,
            ValueNode::Float64Add { .. }
        ));
    }

    // ── GenericAdd → StringConcat ────────────────────────────────────────────

    #[test]
    fn test_specialize_generic_add_to_string_concat() {
        let mut graph = binary_graph(|l, r| ValueNode::GenericAdd {
            left: l,
            right: r,
            feedback_slot: 1,
        });
        let mut fb = FeedbackMap::new();
        fb.insert(1, SpeculatedType::String);

        specialize_types(&mut graph, &fb);

        assert!(matches!(
            &graph.blocks()[0].nodes[2].1,
            ValueNode::StringConcat { .. }
        ));
    }

    // ── GenericSubtract → CheckedSmiSubtract ─────────────────────────────────

    #[test]
    fn test_specialize_generic_subtract_to_smi() {
        let mut graph = binary_graph(|l, r| ValueNode::GenericSubtract {
            left: l,
            right: r,
            feedback_slot: 5,
        });
        let mut fb = FeedbackMap::new();
        fb.insert(5, SpeculatedType::Smi);

        specialize_types(&mut graph, &fb);

        assert!(matches!(
            &graph.blocks()[0].nodes[2].1,
            ValueNode::CheckedSmiSubtract { .. }
        ));
    }

    // ── GenericMultiply → Float64Multiply ────────────────────────────────────

    #[test]
    fn test_specialize_generic_multiply_to_float64() {
        let mut graph = binary_graph(|l, r| ValueNode::GenericMultiply {
            left: l,
            right: r,
            feedback_slot: 7,
        });
        let mut fb = FeedbackMap::new();
        fb.insert(7, SpeculatedType::Float64);

        specialize_types(&mut graph, &fb);

        assert!(matches!(
            &graph.blocks()[0].nodes[2].1,
            ValueNode::Float64Multiply { .. }
        ));
    }

    // ── GenericBitwiseAnd → Int32BitwiseAnd ──────────────────────────────────

    #[test]
    fn test_specialize_generic_bitwise_and_to_int32() {
        let mut graph = binary_graph(|l, r| ValueNode::GenericBitwiseAnd {
            left: l,
            right: r,
            feedback_slot: 10,
        });
        let mut fb = FeedbackMap::new();
        fb.insert(10, SpeculatedType::Smi);

        specialize_types(&mut graph, &fb);

        assert!(matches!(
            &graph.blocks()[0].nodes[2].1,
            ValueNode::Int32BitwiseAnd { .. }
        ));
    }

    // ── GenericNegate → Int32Negate ───────────────────────────────────────────

    #[test]
    fn test_specialize_generic_negate_to_int32() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        let neg = block.push_value(ValueNode::GenericNegate {
            value: p,
            feedback_slot: 3,
        });
        block.set_control(ControlNode::Return { value: neg });
        graph.add_block(block);

        let mut fb = FeedbackMap::new();
        fb.insert(3, SpeculatedType::Smi);

        specialize_types(&mut graph, &fb);

        assert!(matches!(
            &graph.blocks()[0].nodes[1].1,
            ValueNode::Int32Negate { .. }
        ));
    }

    // ── GenericIncrement → CheckedSmiIncrement ───────────────────────────────

    #[test]
    fn test_specialize_generic_increment_to_smi() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        let inc = block.push_value(ValueNode::GenericIncrement {
            value: p,
            feedback_slot: 4,
        });
        block.set_control(ControlNode::Return { value: inc });
        graph.add_block(block);

        let mut fb = FeedbackMap::new();
        fb.insert(4, SpeculatedType::Smi);

        specialize_types(&mut graph, &fb);

        assert!(matches!(
            &graph.blocks()[0].nodes[1].1,
            ValueNode::CheckedSmiIncrement { .. }
        ));
    }

    // ── No feedback → no change ──────────────────────────────────────────────

    #[test]
    fn test_no_feedback_leaves_node_unchanged() {
        let mut graph = binary_graph(|l, r| ValueNode::GenericAdd {
            left: l,
            right: r,
            feedback_slot: 99,
        });

        specialize_types(&mut graph, &FeedbackMap::new());

        assert!(matches!(
            &graph.blocks()[0].nodes[2].1,
            ValueNode::GenericAdd { .. }
        ));
    }

    // ── GenericExponentiate → Float64Exponentiate ────────────────────────────

    #[test]
    fn test_specialize_generic_exponentiate_to_float64() {
        let mut graph = binary_graph(|l, r| ValueNode::GenericExponentiate {
            left: l,
            right: r,
            feedback_slot: 20,
        });
        let mut fb = FeedbackMap::new();
        fb.insert(20, SpeculatedType::Float64);

        specialize_types(&mut graph, &fb);

        assert!(matches!(
            &graph.blocks()[0].nodes[2].1,
            ValueNode::Float64Exponentiate { .. }
        ));
    }

    // ── String feedback on non-add → no change ───────────────────────────────

    #[test]
    fn test_string_feedback_on_subtract_leaves_unchanged() {
        let mut graph = binary_graph(|l, r| ValueNode::GenericSubtract {
            left: l,
            right: r,
            feedback_slot: 1,
        });
        let mut fb = FeedbackMap::new();
        fb.insert(1, SpeculatedType::String);

        specialize_types(&mut graph, &fb);

        assert!(matches!(
            &graph.blocks()[0].nodes[2].1,
            ValueNode::GenericSubtract { .. }
        ));
    }
}
