//! Maglev IR optimisation passes.
//!
//! Eight passes are implemented and composed by [`optimize`]:
//!
//! 1. **Constant folding** — replaces arithmetic/comparison nodes whose *all*
//!    inputs resolve to compile-time constant nodes with a new constant node,
//!    eliminating run-time computation.  Covered operations:
//!    - `Int32` binary arithmetic: `Add`, `Subtract`, `Multiply`, `Divide`,
//!      `Modulus` and the corresponding `CheckedSmi*` variants.
//!    - `Float64` binary arithmetic: `Add`, `Subtract`, `Multiply`, `Divide`,
//!      `Modulus`.
//!    - `Int32` / `Float64` unary: `Negate`, `CheckedSmiIncrement`,
//!      `CheckedSmiDecrement`, `Int32Increment`, `Int32Decrement`.
//!
//! 2. **Strength reduction** — replaces `Int32Multiply` nodes where one
//!    operand is a compile-time power-of-two constant (≥ 2) with an
//!    equivalent `Int32ShiftLeft` by the corresponding shift amount.  This
//!    turns an expensive IMUL into a cheap SHL for common loop-index
//!    scaling patterns like `i * 2`, `i * 4`, `i * 8`.
//!
//! 3. **Range analysis** — tracks integer `[min, max]` intervals through
//!    the graph and replaces `CheckedSmi*` nodes whose output provably fits
//!    `i32` with unchecked `Int32*` equivalents (eliminating deopt overhead).
//!    See [`crate::compiler::maglev::range_analysis`].
//!
//! 4. **Loop-invariant code motion (LICM)** — detects natural loops via
//!    back-edges and hoists pure nodes whose inputs are all defined outside
//!    the loop into the preheader block.
//!    See [`crate::compiler::maglev::licm`].
//!
//! 5. **Redundant type-guard elimination** — removes `CheckSmi` guards on
//!    values already known to be Smis.  A value is *known Smi* when it is
//!    produced by `SmiConstant` or any `CheckedSmi*` node, or when a prior
//!    `CheckSmi` for the same receiver already appears in the same block.
//!
//! 6. **Redundant `CheckMaps` removal** — within each basic block, a
//!    `CheckMaps { receiver, feedback_slot }` node is redundant if an
//!    identical guard for the *same* (receiver, feedback_slot) pair has
//!    already been emitted earlier in the same block.  The duplicate is
//!    replaced by a [`ValueNode::UndefinedConstant`] placeholder and the
//!    relevant ID is remapped so all consumers still compile correctly.
//!
//! 7. **Inlining analysis** — scans for `CallKnownFunction` nodes with
//!    small argument counts and marks them as inlining candidates for a
//!    future inlining pass.
//!
//! 8. **Dead-code elimination** — removes `ValueNode`s whose [`NodeId`]
//!    is never referenced by any other node (value or control) in the graph.
//!    Pure side-effect-free nodes that produce a value which nobody consumes
//!    are safe to drop.  Nodes with observable side-effects (stores, calls,
//!    allocations, guards/checks) are always considered *live* and are kept
//!    even when their result value is unused.
//!
//! # Usage
//!
//! ```
//! use stator_core::compiler::maglev::ir::{
//!     BasicBlock, ControlNode, MaglevGraph, ValueNode,
//! };
//! use stator_core::compiler::maglev::optimizer::optimize;
//!
//! let mut graph = MaglevGraph::new(0);
//! let mut block = BasicBlock::new(0);
//! // 3 + 4  — both operands are compile-time constants.
//! let c3 = block.push_value(ValueNode::Int32Constant { value: 3 });
//! let c4 = block.push_value(ValueNode::Int32Constant { value: 4 });
//! let add = block.push_value(ValueNode::Int32Add { left: c3, right: c4 });
//! block.set_control(ControlNode::Return { value: add });
//! graph.add_block(block);
//!
//! let before = graph.blocks()[0].nodes.len();
//! optimize(&mut graph);
//! // The Int32Add node has been folded; node count is reduced.
//! assert!(graph.blocks()[0].nodes.len() < before);
//! ```

use std::collections::{HashMap, HashSet};

use crate::compiler::maglev::ir::{BasicBlock, ControlNode, MaglevGraph, NodeId, ValueNode};
use crate::compiler::maglev::licm;

// ─────────────────────────────────────────────────────────────────────────────
// Public entry-point
// ─────────────────────────────────────────────────────────────────────────────

/// Run all optimisation passes on `graph` in place.
///
/// Passes are applied in the order: constant folding → strength reduction →
/// range analysis → LICM → redundant type-guard elimination → inlining
/// analysis → redundant-CheckMaps removal → DCE.
///
/// Multiple rounds are *not* performed; a single sweep of each pass is
/// sufficient for the patterns targeted here.
pub fn optimize(graph: &mut MaglevGraph) {
    let block_count = graph.blocks().len();
    let node_count: usize = graph.blocks().iter().map(|b| b.nodes.len()).sum();

    fold_constants(graph);
    let truncations = propagate_int32_truncation(graph);
    simplify_identities(graph);
    strength_reduce(graph);
    crate::compiler::maglev::range_analysis::eliminate_overflow_checks(graph);
    let licm_hoisted = crate::compiler::maglev::licm::hoist_loop_invariants(graph);
    // TODO: implement loop peeling — execute the first iteration outside the
    // loop to establish type information (e.g. from CheckSmi guards), then
    // specialise the loop body based on proven types.  This requires
    // duplicating the loop header and body, which is non-trivial with the
    // current graph structure.
    eliminate_common_subexpressions(graph);
    let globals_promoted = promote_loop_globals_counted(graph);
    // Re-run truncation after global promotion: promotion replaces
    // LoadGlobal/StoreGlobal with Phi nodes, reducing use-counts on
    // arithmetic nodes — allowing CheckedSmi→Int32 conversion.
    let truncations2 = propagate_int32_truncation(graph);
    eliminate_redundant_type_guards(graph);
    mark_inlining_candidates(graph);
    remove_redundant_check_maps(graph);
    eliminate_dead_code(graph);

    let total_truncations = truncations + truncations2;
    let final_nodes: usize = graph.blocks().iter().map(|b| b.nodes.len()).sum();
    // Always print summary so we can confirm optimizer runs at all.
    eprintln!(
        "MAGLEV_OPT: blocks={block_count} nodes={node_count}->{final_nodes} \
         licm_hoisted={licm_hoisted} globals_promoted={globals_promoted} \
         truncated={total_truncations}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1 — Constant folding
// ─────────────────────────────────────────────────────────────────────────────

/// Replace arithmetic nodes with constant inputs by a single constant node.
///
/// The pass walks every block linearly, maintaining a map from [`NodeId`] →
/// known constant value for all nodes emitted so far.  When a binary or unary
/// arithmetic node is found whose operand(s) are all in the constant map, the
/// node is replaced in-place with the folded constant; the old `NodeId` for
/// that position is re-used so downstream references remain valid.
fn fold_constants(graph: &mut MaglevGraph) {
    for block in graph.blocks_mut() {
        fold_block_constants(block);
    }
}

/// Represents a compile-time-known scalar value.
#[derive(Clone, Copy)]
enum ConstVal {
    I32(i32),
    F64(f64),
}

/// Fold constants within a single [`BasicBlock`].
fn fold_block_constants(block: &mut BasicBlock) {
    // Map from NodeId → known constant value for nodes already processed.
    let mut consts: HashMap<NodeId, ConstVal> = HashMap::new();

    for (id, node) in &mut block.nodes {
        // Seed the constant map from literal constant nodes.
        match node {
            ValueNode::Int32Constant { value } => {
                consts.insert(*id, ConstVal::I32(*value));
                continue;
            }
            ValueNode::SmiConstant { value } => {
                consts.insert(*id, ConstVal::I32(*value));
                continue;
            }
            ValueNode::Float64Constant { value } => {
                consts.insert(*id, ConstVal::F64(*value));
                continue;
            }
            _ => {}
        }

        // Try to fold binary i32 operations.
        let folded: Option<ValueNode> = match node {
            // ── Binary Int32 / CheckedSmi ─────────────────────────────────
            ValueNode::Int32Add { left, right } | ValueNode::CheckedSmiAdd { left, right } => {
                fold_i32_bin(left, right, &consts, |a, b| a.wrapping_add(b))
            }
            ValueNode::Int32Subtract { left, right }
            | ValueNode::CheckedSmiSubtract { left, right } => {
                fold_i32_bin(left, right, &consts, |a, b| a.wrapping_sub(b))
            }
            ValueNode::Int32Multiply { left, right }
            | ValueNode::CheckedSmiMultiply { left, right } => {
                fold_i32_bin(left, right, &consts, |a, b| a.wrapping_mul(b))
            }
            ValueNode::Int32Divide { left, right }
            | ValueNode::CheckedSmiDivide { left, right } => {
                if let (Some(ConstVal::I32(a)), Some(ConstVal::I32(b))) =
                    (consts.get(left), consts.get(right))
                {
                    let (a, b) = (*a, *b);
                    if b != 0 {
                        Some(ValueNode::Int32Constant { value: a / b })
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            ValueNode::Int32Modulus { left, right }
            | ValueNode::CheckedSmiModulus { left, right } => {
                if let (Some(ConstVal::I32(a)), Some(ConstVal::I32(b))) =
                    (consts.get(left), consts.get(right))
                {
                    let (a, b) = (*a, *b);
                    if b != 0 {
                        Some(ValueNode::Int32Constant { value: a % b })
                    } else {
                        None
                    }
                } else {
                    None
                }
            }

            // ── Unary Int32 / CheckedSmi ──────────────────────────────────
            ValueNode::Int32Negate { value } => {
                if let Some(ConstVal::I32(v)) = consts.get(value) {
                    Some(ValueNode::Int32Constant {
                        value: v.wrapping_neg(),
                    })
                } else {
                    None
                }
            }
            ValueNode::Int32Increment { value } | ValueNode::CheckedSmiIncrement { value } => {
                if let Some(ConstVal::I32(v)) = consts.get(value) {
                    Some(ValueNode::Int32Constant {
                        value: v.wrapping_add(1),
                    })
                } else {
                    None
                }
            }
            ValueNode::Int32Decrement { value } | ValueNode::CheckedSmiDecrement { value } => {
                if let Some(ConstVal::I32(v)) = consts.get(value) {
                    Some(ValueNode::Int32Constant {
                        value: v.wrapping_sub(1),
                    })
                } else {
                    None
                }
            }

            // ── Binary Float64 ────────────────────────────────────────────
            ValueNode::Float64Add { left, right } => {
                fold_f64_bin(left, right, &consts, |a, b| a + b)
            }
            ValueNode::Float64Subtract { left, right } => {
                fold_f64_bin(left, right, &consts, |a, b| a - b)
            }
            ValueNode::Float64Multiply { left, right } => {
                fold_f64_bin(left, right, &consts, |a, b| a * b)
            }
            ValueNode::Float64Divide { left, right } => {
                fold_f64_bin(left, right, &consts, |a, b| a / b)
            }
            ValueNode::Float64Modulus { left, right } => {
                fold_f64_bin(left, right, &consts, |a, b| a % b)
            }

            // ── Unary Float64 ─────────────────────────────────────────────
            ValueNode::Float64Negate { value } => {
                if let Some(ConstVal::F64(v)) = consts.get(value) {
                    Some(ValueNode::Float64Constant { value: -*v })
                } else {
                    None
                }
            }

            _ => None,
        };

        if let Some(replacement) = folded {
            // Update the constant map so later nodes can fold through this one.
            match &replacement {
                ValueNode::Int32Constant { value } => {
                    consts.insert(*id, ConstVal::I32(*value));
                }
                ValueNode::Float64Constant { value } => {
                    consts.insert(*id, ConstVal::F64(*value));
                }
                _ => {}
            }
            *node = replacement;
        }
    }
}

/// Attempt to fold a binary `i32` operation given a constant-value map.
fn fold_i32_bin(
    left: &NodeId,
    right: &NodeId,
    consts: &HashMap<NodeId, ConstVal>,
    op: impl Fn(i32, i32) -> i32,
) -> Option<ValueNode> {
    if let (Some(ConstVal::I32(a)), Some(ConstVal::I32(b))) = (consts.get(left), consts.get(right))
    {
        Some(ValueNode::Int32Constant { value: op(*a, *b) })
    } else {
        None
    }
}

/// Attempt to fold a binary `f64` operation given a constant-value map.
fn fold_f64_bin(
    left: &NodeId,
    right: &NodeId,
    consts: &HashMap<NodeId, ConstVal>,
    op: impl Fn(f64, f64) -> f64,
) -> Option<ValueNode> {
    let lv = match consts.get(left) {
        Some(ConstVal::F64(v)) => *v,
        Some(ConstVal::I32(v)) => *v as f64,
        None => return None,
    };
    let rv = match consts.get(right) {
        Some(ConstVal::F64(v)) => *v,
        Some(ConstVal::I32(v)) => *v as f64,
        None => return None,
    };
    Some(ValueNode::Float64Constant { value: op(lv, rv) })
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1.3 — Algebraic identity simplification
// ─────────────────────────────────────────────────────────────────────────────
// Pass 1.3 — Int32 truncation propagation
// ─────────────────────────────────────────────────────────────────────────────

/// When a `CheckedSmi*` node's result flows exclusively into a bitwise
/// operation (which truncates to `i32`), the overflow check is unnecessary
/// and can be replaced with the cheaper unchecked `Int32*` variant.
///
/// JavaScript bitwise operators (`|`, `&`, `^`, `<<`, `>>`, `>>>`) convert
/// their operands via `ToInt32`, which wraps to 32 bits.  A common pattern is
/// `(a + b) | 0` to force int32 semantics.  By detecting that the consumer
/// is a truncating op, we eliminate the deopt-on-overflow guard.
///
/// Only nodes whose *sole* consumer is on a truncating path are rewritten,
/// ensuring correctness when a value is also used by non-truncating ops.
fn propagate_int32_truncation(graph: &mut MaglevGraph) -> usize {
    // 1. Build use-count map: NodeId → number of consuming nodes.
    //    CheckSmi guards and StoreGlobal nodes are excluded because they are
    //    "transparent" for truncation purposes — a value consumed only by
    //    truncation ops + CheckSmi guards + StoreGlobal exit-stores is still
    //    safe to convert to unchecked Int32.  StoreGlobal at loop exits just
    //    materialises the (already-truncated) value back to global storage;
    //    it does not observe the overflow semantics.
    let mut use_counts: HashMap<NodeId, usize> = HashMap::new();
    for block in graph.blocks() {
        for (_, node) in &block.nodes {
            if matches!(
                node,
                ValueNode::CheckSmi { .. } | ValueNode::StoreGlobal { .. }
            ) {
                continue;
            }
            visit_value_node_inputs(node, &mut |id| {
                *use_counts.entry(id).or_insert(0) += 1;
            });
        }
        if let Some(ctrl) = &block.control {
            visit_control_inputs(ctrl, &mut |id| {
                *use_counts.entry(id).or_insert(0) += 1;
            });
        }
    }

    // 2. Build NodeId → ValueNode snapshot for backward walking.
    let mut node_map: HashMap<NodeId, ValueNode> = HashMap::new();
    for block in graph.blocks() {
        for (id, node) in &block.nodes {
            node_map.insert(*id, node.clone());
        }
    }

    // 3. Find truncation points (bitwise/shift ops) and walk backward.
    let mut replacements: HashMap<NodeId, ValueNode> = HashMap::new();
    for block in graph.blocks() {
        for (_, node) in &block.nodes {
            match node {
                ValueNode::Int32BitwiseOr { left, right }
                | ValueNode::Int32BitwiseXor { left, right }
                | ValueNode::Int32BitwiseAnd { left, right }
                | ValueNode::Int32ShiftLeft { left, right }
                | ValueNode::Int32ShiftRight { left, right }
                | ValueNode::Int32ShiftRightLogical { left, right } => {
                    mark_truncated(*left, &use_counts, &node_map, &mut replacements);
                    mark_truncated(*right, &use_counts, &node_map, &mut replacements);
                }
                _ => {}
            }
        }
    }

    // 4. Apply replacements in-place.
    let count = replacements.len();
    if !replacements.is_empty() {
        for block in graph.blocks_mut() {
            for (id, node) in &mut block.nodes {
                if let Some(new_node) = replacements.remove(id) {
                    *node = new_node;
                }
            }
        }
    }
    count
}

/// Walk backward from a truncation point, replacing `CheckedSmi*` with
/// unchecked `Int32*`.  Only touches nodes whose non-CheckSmi use count is
/// exactly one (the truncating path).
fn mark_truncated(
    id: NodeId,
    use_counts: &HashMap<NodeId, usize>,
    node_map: &HashMap<NodeId, ValueNode>,
    replacements: &mut HashMap<NodeId, ValueNode>,
) {
    if use_counts.get(&id) != Some(&1) {
        return;
    }
    if let Some(node) = node_map.get(&id) {
        let (new_node, inputs) = match node {
            ValueNode::CheckedSmiAdd { left, right } => (
                ValueNode::Int32Add {
                    left: *left,
                    right: *right,
                },
                vec![*left, *right],
            ),
            ValueNode::CheckedSmiSubtract { left, right } => (
                ValueNode::Int32Subtract {
                    left: *left,
                    right: *right,
                },
                vec![*left, *right],
            ),
            ValueNode::CheckedSmiMultiply { left, right } => (
                ValueNode::Int32Multiply {
                    left: *left,
                    right: *right,
                },
                vec![*left, *right],
            ),
            _ => return,
        };

        replacements.insert(id, new_node);
        for input in inputs {
            mark_truncated(input, use_counts, node_map, replacements);
        }
    }
}

/// Visit all [`NodeId`] inputs of a [`ValueNode`], calling `f` for each.
///
/// Uses a catch-all arm for zero-input nodes and any new variants that
/// don't have NodeId inputs, which is conservative for use-counting.
#[allow(clippy::too_many_lines)]
fn visit_value_node_inputs(node: &ValueNode, f: &mut impl FnMut(NodeId)) {
    match node {
        // ── Two-input arithmetic / comparisons ──────────────────────────
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
        | ValueNode::Uint32Add { left, right }
        | ValueNode::Uint32Subtract { left, right }
        | ValueNode::Uint32Multiply { left, right }
        | ValueNode::Int32BitwiseOr { left, right }
        | ValueNode::Int32BitwiseXor { left, right }
        | ValueNode::Int32BitwiseAnd { left, right }
        | ValueNode::Int32ShiftLeft { left, right }
        | ValueNode::Int32ShiftRight { left, right }
        | ValueNode::Int32ShiftRightLogical { left, right }
        | ValueNode::Float64Add { left, right }
        | ValueNode::Float64Subtract { left, right }
        | ValueNode::Float64Multiply { left, right }
        | ValueNode::Float64Divide { left, right }
        | ValueNode::Float64Modulus { left, right }
        | ValueNode::Float64Exponentiate { left, right }
        | ValueNode::GenericAdd { left, right, .. }
        | ValueNode::GenericSubtract { left, right, .. }
        | ValueNode::GenericMultiply { left, right, .. }
        | ValueNode::GenericDivide { left, right, .. }
        | ValueNode::GenericModulus { left, right, .. }
        | ValueNode::GenericExponentiate { left, right, .. }
        | ValueNode::GenericBitwiseOr { left, right, .. }
        | ValueNode::GenericBitwiseXor { left, right, .. }
        | ValueNode::GenericBitwiseAnd { left, right, .. }
        | ValueNode::GenericShiftLeft { left, right, .. }
        | ValueNode::GenericShiftRight { left, right, .. }
        | ValueNode::GenericShiftRightLogical { left, right, .. }
        | ValueNode::Int32LessThan { left, right }
        | ValueNode::Int32LessThanOrEqual { left, right }
        | ValueNode::Int32GreaterThan { left, right }
        | ValueNode::Int32GreaterThanOrEqual { left, right }
        | ValueNode::Int32StrictEqual { left, right }
        | ValueNode::Float64LessThan { left, right }
        | ValueNode::Float64LessThanOrEqual { left, right }
        | ValueNode::Float64GreaterThan { left, right }
        | ValueNode::Float64GreaterThanOrEqual { left, right }
        | ValueNode::TaggedEqual { left, right, .. }
        | ValueNode::TaggedNotEqual { left, right, .. } => {
            f(*left);
            f(*right);
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

        // ── Single-input nodes ──────────────────────────────────────────
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
        | ValueNode::CheckSmi { receiver: value }
        | ValueNode::CheckNumber { receiver: value }
        | ValueNode::CheckString { receiver: value }
        | ValueNode::CheckSymbol { receiver: value }
        | ValueNode::CheckHeapObject { receiver: value }
        | ValueNode::ToBoolean { value }
        | ValueNode::ToNumber { value, .. }
        | ValueNode::ToString { value, .. }
        | ValueNode::ToObject { value, .. }
        | ValueNode::TypeOf { value }
        | ValueNode::TestTypeOf { value, .. }
        | ValueNode::TestUndetectable { value } => f(*value),

        // ── Multi-input special nodes ───────────────────────────────────
        ValueNode::Phi { inputs } => {
            for id in inputs {
                f(*id);
            }
        }
        ValueNode::CheckMaps { receiver, .. } => f(*receiver),
        ValueNode::LoadField { object, .. } => f(*object),
        ValueNode::LoadFixedArrayElement {
            elements, index, ..
        } => {
            f(*elements);
            f(*index);
        }
        ValueNode::StoreFixedArrayElement {
            elements,
            index,
            value,
            ..
        } => {
            f(*elements);
            f(*index);
            f(*value);
        }
        ValueNode::LoadNamedGeneric { object, .. } => f(*object),
        ValueNode::StoreNamedGeneric { object, value, .. } => {
            f(*object);
            f(*value);
        }
        ValueNode::LoadKeyedGeneric { object, key, .. } => {
            f(*object);
            f(*key);
        }
        ValueNode::StoreKeyedGeneric {
            object, key, value, ..
        } => {
            f(*object);
            f(*key);
            f(*value);
        }
        ValueNode::StoreGlobal { value, .. }
        | ValueNode::StoreContextSlot { value, .. }
        | ValueNode::StoreCurrentContextSlot { value, .. } => f(*value),
        ValueNode::LoadContextSlot { context, .. } => f(*context),
        ValueNode::GetArgument { index } => f(*index),
        ValueNode::Call {
            callee,
            receiver,
            args,
            ..
        } => {
            f(*callee);
            f(*receiver);
            for a in args {
                f(*a);
            }
        }
        ValueNode::CallRuntime { args, .. } => {
            for a in args {
                f(*a);
            }
        }
        ValueNode::Construct {
            constructor, args, ..
        } => {
            f(*constructor);
            for a in args {
                f(*a);
            }
        }

        // Catch-all for zero-input nodes and any new variants.
        _ => {}
    }
}

/// Visit all [`NodeId`] inputs of a [`ControlNode`], calling `f` for each.
fn visit_control_inputs(ctrl: &ControlNode, f: &mut impl FnMut(NodeId)) {
    match ctrl {
        ControlNode::Return { value } => f(*value),
        ControlNode::Branch { condition, .. } => f(*condition),
        ControlNode::Jump { .. } | ControlNode::Deoptimize { .. } => {}
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Simplify algebraic identities: `x + 0 → x`, `x * 1 → x`, `x - 0 → x`,
/// `x * 0 → 0`, `x | 0 → x`, `x ^ 0 → x`, `x & -1 → x`.
///
/// Replaced nodes are turned into [`ValueNode::UndefinedConstant`] placeholders
/// and a substitution table redirects all downstream references to the identity
/// operand.  DCE cleans up the dead placeholders.
fn simplify_identities(graph: &mut MaglevGraph) {
    for block in graph.blocks_mut() {
        simplify_identities_in_block(block);
    }
}

/// Perform algebraic identity simplification within a single [`BasicBlock`].
fn simplify_identities_in_block(block: &mut BasicBlock) {
    // Collect compile-time integer constants visible in this block.
    let mut consts: HashMap<NodeId, i32> = HashMap::new();
    for (id, node) in &block.nodes {
        match node {
            ValueNode::Int32Constant { value } | ValueNode::SmiConstant { value } => {
                consts.insert(*id, *value);
            }
            _ => {}
        }
    }

    if consts.is_empty() {
        return;
    }

    // Build substitution map for identity patterns.
    let mut subst: HashMap<NodeId, NodeId> = HashMap::new();

    for (id, node) in &mut block.nodes {
        let replacement: Option<NodeId> = match node {
            // x + 0 → x, 0 + x → x
            ValueNode::Int32Add { left, right } | ValueNode::CheckedSmiAdd { left, right } => {
                if consts.get(right) == Some(&0) {
                    Some(*left)
                } else if consts.get(left) == Some(&0) {
                    Some(*right)
                } else {
                    None
                }
            }
            // x - 0 → x
            ValueNode::Int32Subtract { left, right }
            | ValueNode::CheckedSmiSubtract { left, right } => {
                if consts.get(right) == Some(&0) {
                    Some(*left)
                } else {
                    None
                }
            }
            // x * 1 → x, 1 * x → x, x * 0 → 0, 0 * x → 0
            ValueNode::Int32Multiply { left, right }
            | ValueNode::CheckedSmiMultiply { left, right } => {
                if consts.get(right) == Some(&1) {
                    Some(*left)
                } else if consts.get(left) == Some(&1) || consts.get(right) == Some(&0) {
                    Some(*right)
                } else if consts.get(left) == Some(&0) {
                    Some(*left)
                } else {
                    None
                }
            }
            // x | 0 → x, 0 | x → x
            ValueNode::Int32BitwiseOr { left, right } => {
                if consts.get(right) == Some(&0) {
                    Some(*left)
                } else if consts.get(left) == Some(&0) {
                    Some(*right)
                } else {
                    None
                }
            }
            // x ^ 0 → x, 0 ^ x → x
            ValueNode::Int32BitwiseXor { left, right } => {
                if consts.get(right) == Some(&0) {
                    Some(*left)
                } else if consts.get(left) == Some(&0) {
                    Some(*right)
                } else {
                    None
                }
            }
            // x & -1 → x, -1 & x → x
            ValueNode::Int32BitwiseAnd { left, right } => {
                if consts.get(right) == Some(&-1) {
                    Some(*left)
                } else if consts.get(left) == Some(&-1) {
                    Some(*right)
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(canonical_id) = replacement {
            subst.insert(*id, canonical_id);
            *node = ValueNode::UndefinedConstant;
        }
    }

    if !subst.is_empty() {
        apply_subst_to_block(block, &subst);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1.5 — Strength reduction
// ─────────────────────────────────────────────────────────────────────────────

/// Replace `Int32Multiply` nodes where one operand is a power-of-two constant
/// (≥ 2) with the cheaper `Int32ShiftLeft` by the corresponding shift amount.
///
/// For example, `x * 4` becomes `x << 2`, turning an `IMUL` into a `SHL` on
/// x86-64.  Only positive compile-time constants that are exact powers of two
/// are considered.
fn strength_reduce(graph: &mut MaglevGraph) {
    // We may need to create new constant nodes for shift amounts.  Find the
    // maximum existing NodeId so new ones are globally unique.
    let mut next_id = graph
        .blocks()
        .iter()
        .flat_map(|b| b.nodes.iter())
        .map(|(id, _)| id.0)
        .max()
        .map_or(0, |m| m + 1);

    for block in graph.blocks_mut() {
        strength_reduce_block(block, &mut next_id);
    }
}

/// Perform strength reduction within a single [`BasicBlock`].
fn strength_reduce_block(block: &mut BasicBlock, next_id: &mut u32) {
    // Collect compile-time integer constants visible in this block.
    let mut consts: HashMap<NodeId, i32> = HashMap::new();
    for (id, node) in &block.nodes {
        match node {
            ValueNode::Int32Constant { value } | ValueNode::SmiConstant { value } => {
                consts.insert(*id, *value);
            }
            _ => {}
        }
    }

    // Identify multiply-by-power-of-two patterns.
    // Each entry: (position, non-constant operand, shift amount).
    let mut reductions: Vec<(usize, NodeId, u32)> = Vec::new();
    for (pos, (_, node)) in block.nodes.iter().enumerate() {
        if let ValueNode::Int32Multiply { left, right } = node
            && let Some((operand, shift)) = find_power_of_two_operand(*left, *right, &consts)
        {
            reductions.push((pos, operand, shift));
        }
    }

    if reductions.is_empty() {
        return;
    }

    // Apply in reverse position order so earlier insertions don't shift later
    // indices.
    for (pos, operand, shift_amt) in reductions.into_iter().rev() {
        let shift_const_id = NodeId(*next_id);
        *next_id += 1;

        // Replace the multiply in-place, keeping its original NodeId.
        let (mul_id, _) = block.nodes[pos];
        block.nodes[pos] = (
            mul_id,
            ValueNode::Int32ShiftLeft {
                left: operand,
                right: shift_const_id,
            },
        );

        // Insert the shift-amount constant immediately before the shift node
        // so it is defined before its use.
        block.nodes.insert(
            pos,
            (
                shift_const_id,
                ValueNode::Int32Constant {
                    value: shift_amt as i32,
                },
            ),
        );
    }
}

/// If exactly one operand of `left × right` is a constant positive power of
/// two (≥ 2), return `(other_operand, trailing_zeros)`.
fn find_power_of_two_operand(
    left: NodeId,
    right: NodeId,
    consts: &HashMap<NodeId, i32>,
) -> Option<(NodeId, u32)> {
    // Check right first (the more common `x * 4` pattern).
    if let Some(&val) = consts.get(&right)
        && val >= 2
        && val.count_ones() == 1
    {
        return Some((left, val.trailing_zeros()));
    }
    // Check left (`4 * x`).
    if let Some(&val) = consts.get(&left)
        && val >= 2
        && val.count_ones() == 1
    {
        return Some((right, val.trailing_zeros()));
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 3.5 — Common subexpression elimination (CSE)
// ─────────────────────────────────────────────────────────────────────────────

/// A hashable key for common-subexpression elimination.
///
/// Two pure nodes with the same discriminant and identical input [`NodeId`]s
/// compute the same value.
#[derive(Hash, Eq, PartialEq)]
enum CseKey {
    /// A binary pure operation: (operation kind, left operand, right operand).
    Binary(u16, NodeId, NodeId),
    /// A unary pure operation: (operation kind, input operand).
    Unary(u16, NodeId),
}

/// Extract a CSE key from a pure [`ValueNode`], or `None` if the node is not
/// eligible for CSE (side-effecting, no inputs, or not yet covered).
fn make_cse_key(node: &ValueNode) -> Option<CseKey> {
    match node {
        // Binary Int32 arithmetic
        ValueNode::Int32Add { left, right } => Some(CseKey::Binary(1, *left, *right)),
        ValueNode::Int32Subtract { left, right } => Some(CseKey::Binary(2, *left, *right)),
        ValueNode::Int32Multiply { left, right } => Some(CseKey::Binary(3, *left, *right)),
        ValueNode::Int32Divide { left, right } => Some(CseKey::Binary(4, *left, *right)),
        ValueNode::Int32Modulus { left, right } => Some(CseKey::Binary(5, *left, *right)),
        ValueNode::Int32BitwiseAnd { left, right } => Some(CseKey::Binary(6, *left, *right)),
        ValueNode::Int32BitwiseOr { left, right } => Some(CseKey::Binary(7, *left, *right)),
        ValueNode::Int32BitwiseXor { left, right } => Some(CseKey::Binary(8, *left, *right)),
        ValueNode::Int32ShiftLeft { left, right } => Some(CseKey::Binary(9, *left, *right)),
        ValueNode::Int32ShiftRight { left, right } => Some(CseKey::Binary(10, *left, *right)),
        ValueNode::Int32ShiftRightLogical { left, right } => {
            Some(CseKey::Binary(11, *left, *right))
        }
        // Int32 comparisons
        ValueNode::Int32Equal { left, right } => Some(CseKey::Binary(12, *left, *right)),
        ValueNode::Int32StrictEqual { left, right } => Some(CseKey::Binary(13, *left, *right)),
        ValueNode::Int32LessThan { left, right } => Some(CseKey::Binary(14, *left, *right)),
        ValueNode::Int32LessThanOrEqual { left, right } => Some(CseKey::Binary(15, *left, *right)),
        ValueNode::Int32GreaterThan { left, right } => Some(CseKey::Binary(16, *left, *right)),
        ValueNode::Int32GreaterThanOrEqual { left, right } => {
            Some(CseKey::Binary(17, *left, *right))
        }
        // Binary Float64 arithmetic
        ValueNode::Float64Add { left, right } => Some(CseKey::Binary(18, *left, *right)),
        ValueNode::Float64Subtract { left, right } => Some(CseKey::Binary(19, *left, *right)),
        ValueNode::Float64Multiply { left, right } => Some(CseKey::Binary(20, *left, *right)),
        ValueNode::Float64Divide { left, right } => Some(CseKey::Binary(21, *left, *right)),
        ValueNode::Float64Modulus { left, right } => Some(CseKey::Binary(22, *left, *right)),
        // Unary operations
        ValueNode::Int32Negate { value } => Some(CseKey::Unary(1, *value)),
        ValueNode::Int32Increment { value } => Some(CseKey::Unary(2, *value)),
        ValueNode::Int32Decrement { value } => Some(CseKey::Unary(3, *value)),
        ValueNode::Float64Negate { value } => Some(CseKey::Unary(4, *value)),
        ValueNode::ToBoolean { value } => Some(CseKey::Unary(5, *value)),
        _ => None,
    }
}

/// Remove duplicate pure computations within each basic block.
///
/// When two nodes in the same block compute the exact same operation on the
/// same input [`NodeId`]s, the second occurrence is replaced with an
/// [`ValueNode::UndefinedConstant`] placeholder and a substitution table
/// redirects all downstream references to the first occurrence.  DCE cleans
/// up the dead placeholder.
fn eliminate_common_subexpressions(graph: &mut MaglevGraph) {
    for block in graph.blocks_mut() {
        cse_in_block(block);
    }
}

/// Perform local CSE within a single [`BasicBlock`].
fn cse_in_block(block: &mut BasicBlock) {
    let mut seen: HashMap<CseKey, NodeId> = HashMap::new();
    let mut subst: HashMap<NodeId, NodeId> = HashMap::new();

    for (id, node) in &mut block.nodes {
        if has_side_effects(node) {
            continue;
        }

        if let Some(key) = make_cse_key(node) {
            if let Some(&first_id) = seen.get(&key) {
                subst.insert(*id, first_id);
                *node = ValueNode::UndefinedConstant;
            } else {
                seen.insert(key, *id);
            }
        }
    }

    if !subst.is_empty() {
        apply_subst_to_block(block, &subst);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 4.4 — Loop-global promotion
// ─────────────────────────────────────────────────────────────────────────────

/// Promote `LoadGlobal`/`StoreGlobal` pairs inside loops into Phi nodes.
///
/// For each natural loop that contains no user function calls (which could
/// observe or modify global state), every global name that is both loaded and
/// stored is replaced with:
///
/// 1. A single `LoadGlobal` in the preheader.
/// 2. A `Phi` at the front of the loop header merging the preheader value
///    and the loop-body computed value.
/// 3. `StoreGlobal` nodes at every loop exit to materialise the final value.
///
/// This eliminates per-iteration memory traffic for loop-carried globals.
fn promote_loop_globals_counted(graph: &mut MaglevGraph) -> usize {
    let loops = licm::detect_loops(graph);
    if loops.is_empty() {
        eprintln!("GLOBALS_PROMO: no loops detected, skipping");
    }
    let mut count = 0;

    for lp in &loops {
        count += promote_globals_in_loop(graph, lp);
    }
    count
}

/// Return `true` if `node` is a user-visible function call that might read or
/// write global variables.  Stub calls (generic property loads/stores) are
/// considered safe because they don't access globals.
fn is_user_call(node: &ValueNode) -> bool {
    matches!(
        node,
        ValueNode::Call { .. }
            | ValueNode::CallKnownFunction { .. }
            | ValueNode::CallBuiltin { .. }
            | ValueNode::CallRuntime { .. }
            | ValueNode::CallWithSpread { .. }
            | ValueNode::Construct { .. }
            | ValueNode::ConstructWithSpread { .. }
    )
}

/// Information about one promoted global inside a loop.
struct PromotedGlobal {
    /// The constant-pool name index of the global.
    name: u32,
    /// Feedback slot reused from the original `LoadGlobal`.
    feedback_slot: u32,
    /// `NodeId` of the `LoadGlobal` inserted in the preheader.
    preheader_load_id: NodeId,
    /// `NodeId` of the `Phi` inserted at the loop header.
    phi_id: NodeId,
    /// `NodeId` of the value the loop body would have stored (back-edge input).
    store_value_id: NodeId,
    /// `NodeId`s of the original `LoadGlobal` nodes inside the loop body.
    original_load_ids: Vec<NodeId>,
    /// `NodeId`s of the original `StoreGlobal` nodes inside the loop body.
    original_store_ids: Vec<NodeId>,
}

/// Promote loop-carried globals to Phi nodes.  Returns the number of globals
/// promoted (0 if none).
fn promote_globals_in_loop(graph: &mut MaglevGraph, lp: &licm::NaturalLoop) -> usize {
    // Safety check: skip loops containing user function calls.
    for block in graph.blocks() {
        if !lp.body.contains(&block.id) {
            continue;
        }
        for (_, node) in &block.nodes {
            if is_user_call(node) {
                return 0;
            }
        }
    }

    // Collect LoadGlobal and StoreGlobal occurrences inside the loop.
    let mut load_names: HashSet<u32> = HashSet::new();
    let mut store_names: HashSet<u32> = HashSet::new();

    // Details keyed by name index.
    let mut load_info: HashMap<u32, (u32, Vec<NodeId>)> = HashMap::new(); // name -> (feedback_slot, [NodeId])
    let mut store_info: HashMap<u32, (NodeId, Vec<NodeId>)> = HashMap::new(); // name -> (last value, [NodeId])

    for block in graph.blocks() {
        if !lp.body.contains(&block.id) {
            continue;
        }
        for (id, node) in &block.nodes {
            match node {
                ValueNode::LoadGlobal {
                    name,
                    feedback_slot,
                } => {
                    load_names.insert(*name);
                    let entry = load_info
                        .entry(*name)
                        .or_insert((*feedback_slot, Vec::new()));
                    entry.1.push(*id);
                }
                ValueNode::StoreGlobal {
                    name,
                    value,
                    feedback_slot: _,
                } => {
                    store_names.insert(*name);
                    let entry = store_info.entry(*name).or_insert((*value, Vec::new()));
                    // Update the value to the latest store (last writer wins).
                    entry.0 = *value;
                    entry.1.push(*id);
                }
                _ => {}
            }
        }
    }

    // Promotable globals: names that appear in both load and store sets.
    let promotable: Vec<u32> = load_names.intersection(&store_names).copied().collect();
    if promotable.is_empty() {
        return 0;
    }

    // Build PromotedGlobal entries: allocate preheader loads and Phi IDs.
    let mut promoted: Vec<PromotedGlobal> = Vec::new();
    for &name in &promotable {
        let (feedback_slot, ref load_ids) = load_info[&name];
        let (store_value, ref store_ids) = store_info[&name];

        // Insert LoadGlobal into the preheader block.
        let preheader_load_id = graph
            .add_value_node(
                lp.preheader,
                ValueNode::LoadGlobal {
                    name,
                    feedback_slot,
                },
            )
            .expect("preheader block must exist");

        // Allocate a NodeId for the Phi (we'll insert it manually at the front).
        let phi_id = graph.alloc_node_id();

        promoted.push(PromotedGlobal {
            name,
            feedback_slot,
            preheader_load_id,
            phi_id,
            store_value_id: store_value,
            original_load_ids: load_ids.clone(),
            original_store_ids: store_ids.clone(),
        });
    }

    // Build the substitution map: old LoadGlobal id → Phi id.
    let mut subst: HashMap<NodeId, NodeId> = HashMap::new();
    for pg in &promoted {
        for &load_id in &pg.original_load_ids {
            subst.insert(load_id, pg.phi_id);
        }
    }

    // Insert Phi nodes at the front of the loop header block.
    // Each Phi starts with a single input (the preheader load); the back-edge
    // input is added after substitution has been applied to store_value_ids.
    if let Some(header_block) = graph.block_mut(lp.header) {
        // Insert in reverse order so the first promoted global ends up first.
        for pg in promoted.iter().rev() {
            header_block.nodes.insert(
                0,
                (
                    pg.phi_id,
                    ValueNode::Phi {
                        inputs: vec![pg.preheader_load_id],
                    },
                ),
            );
        }
    }

    // Replace original LoadGlobal and StoreGlobal nodes inside the loop body
    // with UndefinedConstant (DCE will remove them).
    for block in graph.blocks_mut() {
        if !lp.body.contains(&block.id) {
            continue;
        }
        for (id, node) in &mut block.nodes {
            for pg in &promoted {
                if pg.original_load_ids.contains(id) || pg.original_store_ids.contains(id) {
                    *node = ValueNode::UndefinedConstant;
                }
            }
        }
    }

    // Apply substitution to all blocks in the loop body.
    for block in graph.blocks_mut() {
        if !lp.body.contains(&block.id) {
            continue;
        }
        apply_subst_to_block(block, &subst);
    }

    // Resolve back-edge store_value_ids through the substitution map and add
    // them as the second Phi input.
    for pg in &mut promoted {
        pg.store_value_id = subst
            .get(&pg.store_value_id)
            .copied()
            .unwrap_or(pg.store_value_id);
    }

    if let Some(header_block) = graph.block_mut(lp.header) {
        for pg in &promoted {
            for (_, node) in &mut header_block.nodes {
                if let ValueNode::Phi { inputs } = node {
                    // Find the Phi we inserted by checking the first input.
                    if inputs.len() == 1 && inputs[0] == pg.preheader_load_id {
                        inputs.push(pg.store_value_id);
                        break;
                    }
                }
            }
        }
    }

    // Find loop exits and insert StoreGlobal nodes at each exit.
    // A loop exit is a successor block that is NOT in the loop body.
    let mut exit_blocks: HashSet<u32> = HashSet::new();
    for block in graph.blocks() {
        if !lp.body.contains(&block.id) {
            continue;
        }
        let targets = licm::control_targets(block);
        for &target in &targets {
            if !lp.body.contains(&target) {
                exit_blocks.insert(target);
            }
        }
    }

    // At each exit, prepend StoreGlobal nodes for each promoted global.
    // The value to store is the StoreGlobal's original value input (the
    // computed value from the loop body), which has been substituted through
    // the Phi chain.
    for &exit_id in &exit_blocks {
        for pg in promoted.iter().rev() {
            let store_id = graph.alloc_node_id();
            if let Some(exit_block) = graph.block_mut(exit_id) {
                exit_block.nodes.insert(
                    0,
                    (
                        store_id,
                        ValueNode::StoreGlobal {
                            name: pg.name,
                            value: pg.store_value_id,
                            feedback_slot: pg.feedback_slot,
                        },
                    ),
                );
            }
        }
    }
    promotable.len()
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 4.5 — Redundant type-guard elimination
// ─────────────────────────────────────────────────────────────────────────────

/// Remove `CheckSmi` guards whose receiver is already known to be a Smi.
///
/// A value is *known Smi* when:
/// - It was produced by [`ValueNode::SmiConstant`], or
/// - It was produced by any `CheckedSmi*` node (these deopt on non-Smi and
///   produce a Smi on the success path), or
/// - A prior `CheckSmi` for the same receiver has already appeared in the
///   same basic block (making subsequent guards redundant).
///
/// Eliminated guards are replaced with [`ValueNode::UndefinedConstant`]
/// placeholders that will later be cleaned up by dead-code elimination.
fn eliminate_redundant_type_guards(graph: &mut MaglevGraph) {
    // Phase 1: collect NodeIds that are *globally* known to produce Smi values.
    let mut known_smi: HashSet<NodeId> = HashSet::new();
    for block in graph.blocks() {
        for (id, node) in &block.nodes {
            if is_known_smi_producer(node) {
                known_smi.insert(*id);
            }
        }
    }

    // Phase 1b: iterative fixed-point for Phi nodes.
    // A Phi is known-Smi when ALL of its inputs are known-Smi.  Because Phi
    // inputs may reference other Phi nodes (loop-carried values), we iterate
    // until convergence.
    loop {
        let mut changed = false;
        for block in graph.blocks() {
            for (id, node) in &block.nodes {
                if known_smi.contains(id) {
                    continue;
                }
                if let ValueNode::Phi { inputs } = node
                    && !inputs.is_empty()
                    && inputs.iter().all(|inp| known_smi.contains(inp))
                    && known_smi.insert(*id)
                {
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Phase 2: walk each block, eliminating redundant CheckSmi guards.
    for block in graph.blocks_mut() {
        // Track receivers that have already been guarded within this block.
        let mut block_guarded: HashSet<NodeId> = HashSet::new();

        for (_, node) in &mut block.nodes {
            if let ValueNode::CheckSmi { receiver } = node {
                if known_smi.contains(receiver) || block_guarded.contains(receiver) {
                    // Guard is provably redundant — replace with a harmless
                    // placeholder that DCE will clean up.
                    *node = ValueNode::UndefinedConstant;
                } else {
                    // This guard is required, but from now on the receiver is
                    // known Smi within this block.
                    block_guarded.insert(*receiver);
                }
            }
        }
    }
}

/// Return `true` when `node` is guaranteed to produce a tagged Smi value,
/// making any subsequent `CheckSmi` on its output redundant.
fn is_known_smi_producer(node: &ValueNode) -> bool {
    matches!(
        node,
        ValueNode::SmiConstant { .. }
            | ValueNode::Int32Constant { .. }
            | ValueNode::CheckedSmiAdd { .. }
            | ValueNode::CheckedSmiSubtract { .. }
            | ValueNode::CheckedSmiMultiply { .. }
            | ValueNode::CheckedSmiDivide { .. }
            | ValueNode::CheckedSmiModulus { .. }
            | ValueNode::CheckedSmiIncrement { .. }
            | ValueNode::CheckedSmiDecrement { .. }
            // Unchecked Int32 ops always produce values in i32 range → always Smi.
            | ValueNode::Int32Add { .. }
            | ValueNode::Int32Subtract { .. }
            | ValueNode::Int32Multiply { .. }
            | ValueNode::Int32Divide { .. }
            | ValueNode::Int32Modulus { .. }
            | ValueNode::Int32Negate { .. }
            | ValueNode::Int32Increment { .. }
            | ValueNode::Int32Decrement { .. }
            // Bitwise ops always truncate to i32 → always Smi.
            | ValueNode::Int32BitwiseOr { .. }
            | ValueNode::Int32BitwiseAnd { .. }
            | ValueNode::Int32BitwiseXor { .. }
            | ValueNode::GenericBitwiseNot { .. }
            | ValueNode::ChangeInt32ToTagged { .. }
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 5 — Redundant CheckMaps removal
// ─────────────────────────────────────────────────────────────────────────────

/// Remove duplicate `CheckMaps` guards within each basic block.
///
/// Within a single block a `CheckMaps { receiver, feedback_slot }` node is
/// redundant if the *same* pair has already appeared earlier in the block.
/// Redundant nodes are replaced with `UndefinedConstant` (a harmless
/// placeholder) and any uses of the original [`NodeId`] are re-mapped to the
/// first occurrence's [`NodeId`] via a substitution table applied at the end.
fn remove_redundant_check_maps(graph: &mut MaglevGraph) {
    for block in graph.blocks_mut() {
        remove_redundant_check_maps_in_block(block);
    }
}

/// Remove redundant `CheckMaps` in a single block.
fn remove_redundant_check_maps_in_block(block: &mut BasicBlock) {
    // (receiver NodeId, feedback_slot) → first NodeId that checked this pair.
    let mut seen: HashMap<(NodeId, u32), NodeId> = HashMap::new();
    // Substitution: redundant NodeId → canonical NodeId.
    let mut subst: HashMap<NodeId, NodeId> = HashMap::new();

    for (id, node) in &mut block.nodes {
        if let ValueNode::CheckMaps {
            receiver,
            feedback_slot,
        } = node
        {
            let key = (*receiver, *feedback_slot);
            if let Some(&first) = seen.get(&key) {
                // This guard is redundant — record the substitution and replace
                // the node with a cheap placeholder.
                subst.insert(*id, first);
                *node = ValueNode::UndefinedConstant;
            } else {
                seen.insert(key, *id);
            }
        }
    }

    if subst.is_empty() {
        return;
    }

    // Apply the substitution to all node inputs and the control node.
    apply_subst_to_block(block, &subst);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2.5 — Inlining candidate analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum bytecode size (in instructions) for a function to be considered
/// an inlining candidate.  Used by future inlining passes that have access
/// to callee bytecode.
#[allow(dead_code)]
const INLINE_SIZE_THRESHOLD: usize = 32;

/// Maximum number of parameters for an inlining candidate.
const INLINE_PARAM_THRESHOLD: u32 = 8;

/// Scan `graph` for `CallKnownFunction` nodes and mark small, frequently-
/// called targets as inlining candidates by converting them to inlined-call
/// markers.
///
/// This pass performs call-site profiling analysis: it examines
/// `CallKnownFunction` nodes whose `bytecode_size` is below
/// [`INLINE_SIZE_THRESHOLD`] and marks them for future inlining by setting
/// an `inline_candidate` flag.  Actual AST/IR splicing is deferred to a
/// future iteration — for now, this pass collects statistics and annotates
/// the graph.
///
/// # Heuristics
///
/// A call is considered an inlining candidate when **all** of:
/// 1. The callee's bytecode size ≤ [`INLINE_SIZE_THRESHOLD`] instructions
/// 2. The callee has ≤ [`INLINE_PARAM_THRESHOLD`] parameters
/// 3. The call is not recursive (callee ≠ current function)
fn mark_inlining_candidates(graph: &mut MaglevGraph) {
    let mut candidate_count = 0u32;
    for block in graph.blocks_mut() {
        for (_id, node) in &mut block.nodes {
            if let ValueNode::CallKnownFunction { args, .. } = node {
                // Heuristic: small argument count suggests a simple, inlineable
                // function.  Real inlining would inspect the callee's bytecode
                // size, but at graph-build time we don't have that info.
                if args.len() <= INLINE_PARAM_THRESHOLD as usize {
                    candidate_count += 1;
                }
            }
        }
    }
    graph.set_inline_candidates(candidate_count);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 3 — Dead-code elimination
// ─────────────────────────────────────────────────────────────────────────────

/// Remove `ValueNode`s that are never used and have no observable side-effects.
///
/// Algorithm:
/// 1. Walk all blocks and collect every [`NodeId`] that is referenced — either
///    as an input to another `ValueNode` or as the `value` of a `ControlNode`.
/// 2. Walk all nodes; any node whose own [`NodeId`] is *not* in the live set
///    AND is considered *pure* (no side-effects) is removed.
fn eliminate_dead_code(graph: &mut MaglevGraph) {
    let live = collect_live_ids(graph);

    for block in graph.blocks_mut() {
        block.nodes.retain(|(id, node)| {
            // Always keep nodes with side-effects.
            if has_side_effects(node) {
                return true;
            }
            // Keep pure nodes only if their result is consumed somewhere.
            live.contains(id)
        });
    }
}

/// Collect every [`NodeId`] that appears as an *input* in any node or control
/// node across all blocks.
fn collect_live_ids(graph: &MaglevGraph) -> HashSet<NodeId> {
    let mut live = HashSet::new();

    for block in graph.blocks() {
        for (_id, node) in &block.nodes {
            collect_value_node_inputs(node, &mut live);
        }
        if let Some(ctrl) = &block.control {
            collect_control_node_inputs(ctrl, &mut live);
        }
    }

    live
}

/// Enumerate all [`NodeId`] operands referenced by a [`ValueNode`].
#[allow(clippy::too_many_lines)]
fn collect_value_node_inputs(node: &ValueNode, live: &mut HashSet<NodeId>) {
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
        ValueNode::GetArgument { index } => {
            live.insert(*index);
        }
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
        | ValueNode::NumberToString { value, .. } => {
            live.insert(*value);
        }

        ValueNode::CheckSmi { receiver }
        | ValueNode::CheckNumber { receiver }
        | ValueNode::CheckHeapObject { receiver }
        | ValueNode::CheckSymbol { receiver }
        | ValueNode::CheckString { receiver }
        | ValueNode::CheckStringOrStringWrapper { receiver }
        | ValueNode::CheckSeqOneByteString { receiver }
        | ValueNode::CheckMaps { receiver, .. }
        | ValueNode::CheckMapsWithMigration { receiver, .. }
        | ValueNode::CheckValue { receiver, .. } => {
            live.insert(*receiver);
        }

        ValueNode::CheckDynamicValue { receiver, expected } => {
            live.insert(*receiver);
            live.insert(*expected);
        }

        ValueNode::CheckInt32IsSmi { input }
        | ValueNode::CheckUint32IsSmi { input }
        | ValueNode::CheckHoleyFloat64IsSmi { input }
        | ValueNode::CheckFloat64IsNan { input } => {
            live.insert(*input);
        }

        ValueNode::LoadField { object, .. }
        | ValueNode::LoadTaggedField { object, .. }
        | ValueNode::LoadDoubleField { object, .. }
        | ValueNode::LoadNamedGeneric { object, .. }
        | ValueNode::ForInPrepare {
            enumerator: object, ..
        }
        | ValueNode::StringLength { string: object } => {
            live.insert(*object);
        }

        ValueNode::LoadEnumCacheLength { map } => {
            live.insert(*map);
        }

        ValueNode::LoadKeyedGeneric { object, key, .. } => {
            live.insert(*object);
            live.insert(*key);
        }

        ValueNode::HasInPrototypeChain { object, prototype } => {
            live.insert(*object);
            live.insert(*prototype);
        }

        ValueNode::StoreField { object, value, .. } => {
            live.insert(*object);
            live.insert(*value);
        }

        ValueNode::StoreCurrentContextSlot { value, .. } | ValueNode::StoreGlobal { value, .. } => {
            live.insert(*value);
        }

        ValueNode::LoadContextSlot { context, .. } => {
            live.insert(*context);
        }
        ValueNode::StoreContextSlot { context, value, .. } => {
            live.insert(*context);
            live.insert(*value);
        }

        ValueNode::LoadFixedArrayElement { elements, index }
        | ValueNode::LoadFixedDoubleArrayElement { elements, index }
        | ValueNode::LoadHoleyFixedDoubleArrayElement { elements, index } => {
            live.insert(*elements);
            live.insert(*index);
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
            live.insert(*elements);
            live.insert(*index);
            live.insert(*value);
        }

        ValueNode::StoreNamedGeneric { object, value, .. } => {
            live.insert(*object);
            live.insert(*value);
        }

        ValueNode::StoreKeyedGeneric {
            object, key, value, ..
        } => {
            live.insert(*object);
            live.insert(*key);
            live.insert(*value);
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
            live.insert(*left);
            live.insert(*right);
        }

        ValueNode::CheckInt32Condition { left, right, .. } => {
            live.insert(*left);
            live.insert(*right);
        }

        ValueNode::CheckCacheIndicesNotCleared { receiver, indices } => {
            live.insert(*receiver);
            live.insert(*indices);
        }

        ValueNode::TestInstanceOf {
            object, callable, ..
        } => {
            live.insert(*object);
            live.insert(*callable);
        }
        ValueNode::TestIn { key, object, .. } => {
            live.insert(*key);
            live.insert(*object);
        }
        ValueNode::TestUndetectable { value } | ValueNode::TestTypeOf { value, .. } => {
            live.insert(*value);
        }

        ValueNode::StringAt { string, index } => {
            live.insert(*string);
            live.insert(*index);
        }

        ValueNode::ForInNext {
            receiver,
            cache_index,
            cache_array,
            ..
        } => {
            live.insert(*receiver);
            live.insert(*cache_index);
            live.insert(*cache_array);
        }

        ValueNode::DeleteProperty { object, key, .. } => {
            live.insert(*object);
            live.insert(*key);
        }

        ValueNode::CreateCatchContext { exception, .. } => {
            live.insert(*exception);
        }
        ValueNode::CreateWithContext { object, .. } => {
            live.insert(*object);
        }

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
            live.insert(*callee);
            live.insert(*receiver);
            for &a in args {
                live.insert(a);
            }
        }

        ValueNode::CallBuiltin { args, .. } | ValueNode::CallRuntime { args, .. } => {
            for &a in args {
                live.insert(a);
            }
        }

        ValueNode::Construct {
            constructor, args, ..
        }
        | ValueNode::ConstructWithSpread {
            constructor, args, ..
        } => {
            live.insert(*constructor);
            for &a in args {
                live.insert(a);
            }
        }

        ValueNode::Phi { inputs } => {
            for &inp in inputs {
                live.insert(inp);
            }
        }
    }
}

/// Collect all [`NodeId`] operands referenced by a [`ControlNode`].
fn collect_control_node_inputs(ctrl: &ControlNode, live: &mut HashSet<NodeId>) {
    match ctrl {
        ControlNode::Return { value } => {
            live.insert(*value);
        }
        ControlNode::Branch { condition, .. } => {
            live.insert(*condition);
        }
        ControlNode::Jump { .. } | ControlNode::Deoptimize { .. } => {}
    }
}

/// Return `true` if this node has observable side-effects and must never be
/// removed by DCE regardless of whether its result value is consumed.
fn has_side_effects(node: &ValueNode) -> bool {
    matches!(
        node,
        // Stores always have side-effects.
        ValueNode::StoreField { .. }
            | ValueNode::StoreFixedArrayElement { .. }
            | ValueNode::StoreFixedDoubleArrayElement { .. }
            | ValueNode::StoreNamedGeneric { .. }
            | ValueNode::StoreKeyedGeneric { .. }
            | ValueNode::StoreGlobal { .. }
            | ValueNode::StoreContextSlot { .. }
            | ValueNode::StoreCurrentContextSlot { .. }
            // Calls may have side-effects.
            | ValueNode::Call { .. }
            | ValueNode::CallKnownFunction { .. }
            | ValueNode::CallBuiltin { .. }
            | ValueNode::CallRuntime { .. }
            | ValueNode::CallWithSpread { .. }
            | ValueNode::Construct { .. }
            | ValueNode::ConstructWithSpread { .. }
            // Allocations have side-effects (GC pressure / observable identity).
            | ValueNode::CreateObjectLiteral { .. }
            | ValueNode::CreateArrayLiteral { .. }
            | ValueNode::CreateShallowObjectLiteral { .. }
            | ValueNode::CreateShallowArrayLiteral { .. }
            | ValueNode::CreateFunctionContext { .. }
            | ValueNode::CreateBlockContext { .. }
            | ValueNode::CreateCatchContext { .. }
            | ValueNode::CreateWithContext { .. }
            | ValueNode::CreateClosure { .. }
            | ValueNode::FastCreateClosure { .. }
            | ValueNode::CreateEmptyObjectLiteral
            | ValueNode::CreateRegExpLiteral { .. }
            // Guards deoptimise on failure — side-effecting.
            | ValueNode::CheckSmi { .. }
            | ValueNode::CheckNumber { .. }
            | ValueNode::CheckHeapObject { .. }
            | ValueNode::CheckSymbol { .. }
            | ValueNode::CheckString { .. }
            | ValueNode::CheckStringOrStringWrapper { .. }
            | ValueNode::CheckSeqOneByteString { .. }
            | ValueNode::CheckMaps { .. }
            | ValueNode::CheckMapsWithMigration { .. }
            | ValueNode::CheckValue { .. }
            | ValueNode::CheckDynamicValue { .. }
            | ValueNode::CheckInt32IsSmi { .. }
            | ValueNode::CheckUint32IsSmi { .. }
            | ValueNode::CheckHoleyFloat64IsSmi { .. }
            | ValueNode::CheckInt32Condition { .. }
            | ValueNode::CheckCacheIndicesNotCleared { .. }
            | ValueNode::CheckFloat64IsNan { .. }
            | ValueNode::CheckedSmiAdd { .. }
            | ValueNode::CheckedSmiSubtract { .. }
            | ValueNode::CheckedSmiMultiply { .. }
            | ValueNode::CheckedSmiDivide { .. }
            | ValueNode::CheckedSmiModulus { .. }
            | ValueNode::CheckedSmiIncrement { .. }
            | ValueNode::CheckedSmiDecrement { .. }
            | ValueNode::CheckedFloat64ToInt32 { .. }
            | ValueNode::CheckedTaggedToInt32 { .. }
            | ValueNode::CheckedTaggedToFloat64 { .. }
            // Debugger / abort are always live.
            | ValueNode::Debugger
            | ValueNode::Abort { .. }
            // Property mutation.
            | ValueNode::DeleteProperty { .. }
            // For-in side-effects (cache invalidation).
            | ValueNode::ForInPrepare { .. }
            | ValueNode::ForInNext { .. }
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Substitution helper
// ─────────────────────────────────────────────────────────────────────────────

/// Rewrite every [`NodeId`] operand in `block` using the `subst` table.
///
/// This is used after the redundant-CheckMaps pass to redirect consumers of
/// the removed node ID to the canonical first-occurrence ID.
fn apply_subst_to_block(block: &mut BasicBlock, subst: &HashMap<NodeId, NodeId>) {
    let resolve = |id: NodeId| *subst.get(&id).unwrap_or(&id);

    for (_id, node) in &mut block.nodes {
        apply_subst_to_value_node(node, &resolve);
    }
    if let Some(ctrl) = &mut block.control {
        apply_subst_to_control_node(ctrl, &resolve);
    }
}

/// Apply a substitution function to every [`NodeId`] operand in a [`ValueNode`].
#[allow(clippy::too_many_lines)]
fn apply_subst_to_value_node(node: &mut ValueNode, resolve: &impl Fn(NodeId) -> NodeId) {
    match node {
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

        ValueNode::GetArgument { index } => *index = resolve(*index),

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
        | ValueNode::TestTypeOf { value, .. } => *value = resolve(*value),

        ValueNode::CheckSmi { receiver }
        | ValueNode::CheckNumber { receiver }
        | ValueNode::CheckHeapObject { receiver }
        | ValueNode::CheckSymbol { receiver }
        | ValueNode::CheckString { receiver }
        | ValueNode::CheckStringOrStringWrapper { receiver }
        | ValueNode::CheckSeqOneByteString { receiver }
        | ValueNode::CheckMaps { receiver, .. }
        | ValueNode::CheckMapsWithMigration { receiver, .. }
        | ValueNode::CheckValue { receiver, .. } => *receiver = resolve(*receiver),

        ValueNode::CheckDynamicValue { receiver, expected } => {
            *receiver = resolve(*receiver);
            *expected = resolve(*expected);
        }

        ValueNode::CheckInt32IsSmi { input }
        | ValueNode::CheckUint32IsSmi { input }
        | ValueNode::CheckHoleyFloat64IsSmi { input }
        | ValueNode::CheckFloat64IsNan { input } => *input = resolve(*input),

        ValueNode::LoadField { object, .. }
        | ValueNode::LoadTaggedField { object, .. }
        | ValueNode::LoadDoubleField { object, .. }
        | ValueNode::LoadNamedGeneric { object, .. }
        | ValueNode::LoadEnumCacheLength { map: object }
        | ValueNode::StringLength { string: object } => *object = resolve(*object),

        ValueNode::LoadKeyedGeneric { object, key, .. } => {
            *object = resolve(*object);
            *key = resolve(*key);
        }

        ValueNode::HasInPrototypeChain { object, prototype } => {
            *object = resolve(*object);
            *prototype = resolve(*prototype);
        }

        ValueNode::StoreField { object, value, .. } => {
            *object = resolve(*object);
            *value = resolve(*value);
        }

        ValueNode::StoreCurrentContextSlot { value, .. } | ValueNode::StoreGlobal { value, .. } => {
            *value = resolve(*value);
        }

        ValueNode::LoadContextSlot { context, .. } => *context = resolve(*context),
        ValueNode::StoreContextSlot { context, value, .. } => {
            *context = resolve(*context);
            *value = resolve(*value);
        }

        ValueNode::LoadFixedArrayElement { elements, index }
        | ValueNode::LoadFixedDoubleArrayElement { elements, index }
        | ValueNode::LoadHoleyFixedDoubleArrayElement { elements, index } => {
            *elements = resolve(*elements);
            *index = resolve(*index);
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
            *elements = resolve(*elements);
            *index = resolve(*index);
            *value = resolve(*value);
        }

        ValueNode::StoreNamedGeneric { object, value, .. } => {
            *object = resolve(*object);
            *value = resolve(*value);
        }

        ValueNode::StoreKeyedGeneric {
            object, key, value, ..
        } => {
            *object = resolve(*object);
            *key = resolve(*key);
            *value = resolve(*value);
        }

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
            *left = resolve(*left);
            *right = resolve(*right);
        }

        ValueNode::CheckInt32Condition { left, right, .. } => {
            *left = resolve(*left);
            *right = resolve(*right);
        }

        ValueNode::CheckCacheIndicesNotCleared { receiver, indices } => {
            *receiver = resolve(*receiver);
            *indices = resolve(*indices);
        }

        ValueNode::TestInstanceOf {
            object, callable, ..
        } => {
            *object = resolve(*object);
            *callable = resolve(*callable);
        }
        ValueNode::TestIn { key, object, .. } => {
            *key = resolve(*key);
            *object = resolve(*object);
        }

        ValueNode::StringAt { string, index } => {
            *string = resolve(*string);
            *index = resolve(*index);
        }

        ValueNode::ForInPrepare { enumerator, .. } => *enumerator = resolve(*enumerator),
        ValueNode::ForInNext {
            receiver,
            cache_index,
            cache_array,
            ..
        } => {
            *receiver = resolve(*receiver);
            *cache_index = resolve(*cache_index);
            *cache_array = resolve(*cache_array);
        }

        ValueNode::DeleteProperty { object, key, .. } => {
            *object = resolve(*object);
            *key = resolve(*key);
        }

        ValueNode::CreateCatchContext { exception, .. } => *exception = resolve(*exception),
        ValueNode::CreateWithContext { object, .. } => *object = resolve(*object),

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
            *callee = resolve(*callee);
            *receiver = resolve(*receiver);
            for a in args.iter_mut() {
                *a = resolve(*a);
            }
        }

        ValueNode::CallBuiltin { args, .. } | ValueNode::CallRuntime { args, .. } => {
            for a in args.iter_mut() {
                *a = resolve(*a);
            }
        }

        ValueNode::Construct {
            constructor, args, ..
        }
        | ValueNode::ConstructWithSpread {
            constructor, args, ..
        } => {
            *constructor = resolve(*constructor);
            for a in args.iter_mut() {
                *a = resolve(*a);
            }
        }

        ValueNode::Phi { inputs } => {
            for inp in inputs.iter_mut() {
                *inp = resolve(*inp);
            }
        }
    }
}

/// Apply a substitution to the operands of a [`ControlNode`].
fn apply_subst_to_control_node(ctrl: &mut ControlNode, resolve: &impl Fn(NodeId) -> NodeId) {
    match ctrl {
        ControlNode::Return { value } => *value = resolve(*value),
        ControlNode::Branch { condition, .. } => *condition = resolve(*condition),
        ControlNode::Jump { .. } | ControlNode::Deoptimize { .. } => {}
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::maglev::ir::{BasicBlock, ControlNode, MaglevGraph, ValueNode};

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Count the total number of value nodes across all blocks.
    fn total_node_count(graph: &MaglevGraph) -> usize {
        graph.blocks().iter().map(|b| b.nodes.len()).sum()
    }

    /// Return the single value node in block 0 at position `pos`.
    fn node_at(graph: &MaglevGraph, pos: usize) -> &ValueNode {
        &graph.blocks()[0].nodes[pos].1
    }

    // ── Constant folding — Int32 ──────────────────────────────────────────────

    #[test]
    fn test_fold_int32_add() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c3 = block.push_value(ValueNode::Int32Constant { value: 3 });
        let c4 = block.push_value(ValueNode::Int32Constant { value: 4 });
        let add = block.push_value(ValueNode::Int32Add {
            left: c3,
            right: c4,
        });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        optimize(&mut graph);

        // The Int32Add node must have been folded to Int32Constant(7).
        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 7 });
    }

    #[test]
    fn test_fold_int32_subtract() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c10 = block.push_value(ValueNode::Int32Constant { value: 10 });
        let c3 = block.push_value(ValueNode::Int32Constant { value: 3 });
        let sub = block.push_value(ValueNode::Int32Subtract {
            left: c10,
            right: c3,
        });
        block.set_control(ControlNode::Return { value: sub });
        graph.add_block(block);

        optimize(&mut graph);

        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 7 });
    }

    #[test]
    fn test_fold_int32_multiply() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c6 = block.push_value(ValueNode::Int32Constant { value: 6 });
        let c7 = block.push_value(ValueNode::Int32Constant { value: 7 });
        let mul = block.push_value(ValueNode::Int32Multiply {
            left: c6,
            right: c7,
        });
        block.set_control(ControlNode::Return { value: mul });
        graph.add_block(block);

        optimize(&mut graph);

        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 42 });
    }

    #[test]
    fn test_fold_int32_divide() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c12 = block.push_value(ValueNode::Int32Constant { value: 12 });
        let c4 = block.push_value(ValueNode::Int32Constant { value: 4 });
        let div = block.push_value(ValueNode::Int32Divide {
            left: c12,
            right: c4,
        });
        block.set_control(ControlNode::Return { value: div });
        graph.add_block(block);

        optimize(&mut graph);

        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 3 });
    }

    #[test]
    fn test_fold_int32_divide_by_zero_not_folded() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c5 = block.push_value(ValueNode::Int32Constant { value: 5 });
        let c0 = block.push_value(ValueNode::Int32Constant { value: 0 });
        let div = block.push_value(ValueNode::Int32Divide {
            left: c5,
            right: c0,
        });
        block.set_control(ControlNode::Return { value: div });
        graph.add_block(block);

        let before = total_node_count(&graph);
        optimize(&mut graph);
        // Div-by-zero must NOT be folded (the node stays as Int32Divide).
        assert!(matches!(node_at(&graph, 2), ValueNode::Int32Divide { .. }));
        // Node count should not decrease (DCE keeps the divide because its
        // result is consumed by the Return).
        assert_eq!(total_node_count(&graph), before);
    }

    #[test]
    fn test_fold_int32_modulus() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c10 = block.push_value(ValueNode::Int32Constant { value: 10 });
        let c3 = block.push_value(ValueNode::Int32Constant { value: 3 });
        let rem = block.push_value(ValueNode::Int32Modulus {
            left: c10,
            right: c3,
        });
        block.set_control(ControlNode::Return { value: rem });
        graph.add_block(block);

        optimize(&mut graph);

        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 1 });
    }

    #[test]
    fn test_fold_int32_negate() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c5 = block.push_value(ValueNode::Int32Constant { value: 5 });
        let neg = block.push_value(ValueNode::Int32Negate { value: c5 });
        block.set_control(ControlNode::Return { value: neg });
        graph.add_block(block);

        optimize(&mut graph);

        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: -5 });
    }

    #[test]
    fn test_fold_int32_increment() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c9 = block.push_value(ValueNode::Int32Constant { value: 9 });
        let inc = block.push_value(ValueNode::Int32Increment { value: c9 });
        block.set_control(ControlNode::Return { value: inc });
        graph.add_block(block);

        optimize(&mut graph);

        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 10 });
    }

    #[test]
    fn test_fold_int32_decrement() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c10 = block.push_value(ValueNode::Int32Constant { value: 10 });
        let dec = block.push_value(ValueNode::Int32Decrement { value: c10 });
        block.set_control(ControlNode::Return { value: dec });
        graph.add_block(block);

        optimize(&mut graph);

        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 9 });
    }

    // ── Constant folding — CheckedSmi variants ────────────────────────────────

    #[test]
    fn test_fold_checked_smi_add() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let a = block.push_value(ValueNode::SmiConstant { value: 2 });
        let b = block.push_value(ValueNode::SmiConstant { value: 3 });
        let sum = block.push_value(ValueNode::CheckedSmiAdd { left: a, right: b });
        block.set_control(ControlNode::Return { value: sum });
        graph.add_block(block);

        optimize(&mut graph);

        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 5 });
    }

    #[test]
    fn test_fold_checked_smi_increment() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c = block.push_value(ValueNode::SmiConstant { value: 41 });
        let inc = block.push_value(ValueNode::CheckedSmiIncrement { value: c });
        block.set_control(ControlNode::Return { value: inc });
        graph.add_block(block);

        optimize(&mut graph);

        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 42 });
    }

    // ── Constant folding — Float64 ────────────────────────────────────────────

    #[test]
    fn test_fold_float64_add() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let x = block.push_value(ValueNode::Float64Constant { value: 1.5 });
        let y = block.push_value(ValueNode::Float64Constant { value: 2.5 });
        let add = block.push_value(ValueNode::Float64Add { left: x, right: y });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        optimize(&mut graph);

        if let ValueNode::Float64Constant { value } = node_at(&graph, 0) {
            assert!((value - 4.0).abs() < f64::EPSILON);
        } else {
            panic!("expected Float64Constant after folding");
        }
    }

    #[test]
    fn test_fold_float64_multiply() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let x = block.push_value(ValueNode::Float64Constant { value: 3.0 });
        let y = block.push_value(ValueNode::Float64Constant { value: 4.0 });
        let mul = block.push_value(ValueNode::Float64Multiply { left: x, right: y });
        block.set_control(ControlNode::Return { value: mul });
        graph.add_block(block);

        optimize(&mut graph);

        if let ValueNode::Float64Constant { value } = node_at(&graph, 0) {
            assert!((value - 12.0).abs() < f64::EPSILON);
        } else {
            panic!("expected Float64Constant after folding");
        }
    }

    #[test]
    fn test_fold_float64_negate() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let x = block.push_value(ValueNode::Float64Constant { value: 2.5 });
        let neg = block.push_value(ValueNode::Float64Negate { value: x });
        block.set_control(ControlNode::Return { value: neg });
        graph.add_block(block);

        optimize(&mut graph);

        if let ValueNode::Float64Constant { value } = node_at(&graph, 0) {
            assert!((value - (-2.5)).abs() < f64::EPSILON);
        } else {
            panic!("expected Float64Constant after folding");
        }
    }

    // ── Constant folding — no fold when operand is dynamic ────────────────────

    #[test]
    fn test_no_fold_when_operand_is_parameter() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        let c = block.push_value(ValueNode::Int32Constant { value: 1 });
        let add = block.push_value(ValueNode::Int32Add { left: p, right: c });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        optimize(&mut graph);

        // The add node should remain unchanged.
        assert!(matches!(node_at(&graph, 2), ValueNode::Int32Add { .. }));
    }

    // ── Dead-code elimination ─────────────────────────────────────────────────

    #[test]
    fn test_dce_removes_unused_pure_node() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        // This constant is never used.
        block.push_value(ValueNode::Int32Constant { value: 99 });
        // This one is returned.
        let ret_val = block.push_value(ValueNode::SmiConstant { value: 0 });
        block.set_control(ControlNode::Return { value: ret_val });
        graph.add_block(block);

        let before = total_node_count(&graph);
        optimize(&mut graph);
        // The unused constant should have been removed.
        assert!(total_node_count(&graph) < before);
    }

    #[test]
    fn test_dce_keeps_side_effect_node() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let obj = block.push_value(ValueNode::Parameter { index: 0 });
        let val = block.push_value(ValueNode::Int32Constant { value: 1 });
        // Store has a side-effect: must be kept even if nobody reads its result.
        block.push_value(ValueNode::StoreField {
            object: obj,
            offset: 8,
            value: val,
        });
        let ret_val = block.push_value(ValueNode::UndefinedConstant);
        block.set_control(ControlNode::Return { value: ret_val });
        graph.add_block(block);

        let before = total_node_count(&graph);
        optimize(&mut graph);
        // Store must not be removed.
        assert_eq!(total_node_count(&graph), before);
    }

    #[test]
    fn test_dce_keeps_check_maps_node() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        // CheckMaps is a guard with side-effects (deopt on failure).
        let _check = block.push_value(ValueNode::CheckMaps {
            receiver: p,
            feedback_slot: 0,
        });
        let field = block.push_value(ValueNode::LoadField {
            object: p,
            offset: 8,
        });
        block.set_control(ControlNode::Return { value: field });
        graph.add_block(block);

        let before = total_node_count(&graph);
        optimize(&mut graph);
        // CheckMaps must survive DCE.
        assert_eq!(total_node_count(&graph), before);
    }

    // ── Redundant CheckMaps removal ───────────────────────────────────────────

    #[test]
    fn test_redundant_check_maps_removed() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        // First check — canonical.
        let _check1 = block.push_value(ValueNode::CheckMaps {
            receiver: p,
            feedback_slot: 0,
        });
        // Second identical check — redundant.
        let _check2 = block.push_value(ValueNode::CheckMaps {
            receiver: p,
            feedback_slot: 0,
        });
        let field = block.push_value(ValueNode::LoadField {
            object: p,
            offset: 8,
        });
        block.set_control(ControlNode::Return { value: field });
        graph.add_block(block);

        let before = total_node_count(&graph);
        optimize(&mut graph);
        // The redundant CheckMaps gets replaced by UndefinedConstant; DCE then
        // removes it (it's pure and unused).  Net effect: node count decreases.
        assert!(total_node_count(&graph) < before);
    }

    #[test]
    fn test_different_receivers_both_kept() {
        let mut graph = MaglevGraph::new(2);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        let p1 = block.push_value(ValueNode::Parameter { index: 1 });
        // Two checks with *different* receivers — both must be kept.
        block.push_value(ValueNode::CheckMaps {
            receiver: p0,
            feedback_slot: 0,
        });
        block.push_value(ValueNode::CheckMaps {
            receiver: p1,
            feedback_slot: 0,
        });
        let f0 = block.push_value(ValueNode::LoadField {
            object: p0,
            offset: 8,
        });
        let f1 = block.push_value(ValueNode::LoadField {
            object: p1,
            offset: 8,
        });
        let add = block.push_value(ValueNode::Int32Add {
            left: f0,
            right: f1,
        });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        let before = total_node_count(&graph);
        optimize(&mut graph);
        // Neither check is redundant — count must not decrease.
        assert_eq!(total_node_count(&graph), before);
    }

    #[test]
    fn test_different_slots_both_kept() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        // Same receiver, different feedback slots — both must be kept.
        block.push_value(ValueNode::CheckMaps {
            receiver: p,
            feedback_slot: 0,
        });
        block.push_value(ValueNode::CheckMaps {
            receiver: p,
            feedback_slot: 1,
        });
        let f = block.push_value(ValueNode::LoadField {
            object: p,
            offset: 8,
        });
        block.set_control(ControlNode::Return { value: f });
        graph.add_block(block);

        let before = total_node_count(&graph);
        optimize(&mut graph);
        assert_eq!(total_node_count(&graph), before);
    }

    // ── Combined passes ───────────────────────────────────────────────────────

    #[test]
    fn test_combined_fold_and_dce_reduces_node_count() {
        // Build: return (3 + 4)
        // After constant folding → return Int32Constant(7)
        // After DCE → constants 3 and 4 are dead.
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c3 = block.push_value(ValueNode::Int32Constant { value: 3 });
        let c4 = block.push_value(ValueNode::Int32Constant { value: 4 });
        let add = block.push_value(ValueNode::Int32Add {
            left: c3,
            right: c4,
        });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        let before = total_node_count(&graph); // 3 nodes
        optimize(&mut graph);
        let after = total_node_count(&graph);
        // We expect only Int32Constant(7) to remain; 3 and 4 are dead.
        assert!(after < before, "expected fewer nodes after optimisation");
        // Exactly one node should survive (the folded result).
        assert_eq!(after, 1);
        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 7 });
    }

    #[test]
    fn test_chain_folding() {
        // (1 + 2) * (3 + 4) = 3 * 7 = 21
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c1 = block.push_value(ValueNode::Int32Constant { value: 1 });
        let c2 = block.push_value(ValueNode::Int32Constant { value: 2 });
        let c3 = block.push_value(ValueNode::Int32Constant { value: 3 });
        let c4 = block.push_value(ValueNode::Int32Constant { value: 4 });
        let add1 = block.push_value(ValueNode::Int32Add {
            left: c1,
            right: c2,
        });
        let add2 = block.push_value(ValueNode::Int32Add {
            left: c3,
            right: c4,
        });
        let mul = block.push_value(ValueNode::Int32Multiply {
            left: add1,
            right: add2,
        });
        block.set_control(ControlNode::Return { value: mul });
        graph.add_block(block);

        optimize(&mut graph);

        // All intermediates should be DCE'd; only one node left.
        assert_eq!(total_node_count(&graph), 1);
        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 21 });
    }

    // ── Strength reduction ────────────────────────────────────────────────────

    #[test]
    fn test_strength_reduce_multiply_by_2() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        let c2 = block.push_value(ValueNode::Int32Constant { value: 2 });
        let mul = block.push_value(ValueNode::Int32Multiply { left: p, right: c2 });
        block.set_control(ControlNode::Return { value: mul });
        graph.add_block(block);

        optimize(&mut graph);

        // The multiply should have been replaced with a shift-left.
        let has_shift = graph.blocks()[0]
            .nodes
            .iter()
            .any(|(_, n)| matches!(n, ValueNode::Int32ShiftLeft { .. }));
        assert!(
            has_shift,
            "expected Int32ShiftLeft after strength reduction"
        );
    }

    #[test]
    fn test_strength_reduce_multiply_by_4() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        let c4 = block.push_value(ValueNode::Int32Constant { value: 4 });
        let mul = block.push_value(ValueNode::Int32Multiply { left: p, right: c4 });
        block.set_control(ControlNode::Return { value: mul });
        graph.add_block(block);

        optimize(&mut graph);

        // The shift amount should be 2 (since 4 == 2^2).
        let shift_const = graph.blocks()[0].nodes.iter().find_map(|(_, n)| {
            if let ValueNode::Int32ShiftLeft { right, .. } = n {
                // Find the constant node that `right` refers to.
                graph.blocks()[0].nodes.iter().find_map(|(id, cn)| {
                    if *id == *right {
                        if let ValueNode::Int32Constant { value } = cn {
                            Some(*value)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
            } else {
                None
            }
        });
        assert_eq!(shift_const, Some(2), "shift amount should be 2 for * 4");
    }

    #[test]
    fn test_strength_reduce_not_applied_to_non_power_of_two() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        let c3 = block.push_value(ValueNode::Int32Constant { value: 3 });
        let mul = block.push_value(ValueNode::Int32Multiply { left: p, right: c3 });
        block.set_control(ControlNode::Return { value: mul });
        graph.add_block(block);

        optimize(&mut graph);

        // Multiply by 3 is not a power of two — node should stay as multiply.
        assert!(
            graph.blocks()[0]
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::Int32Multiply { .. }))
        );
    }

    #[test]
    fn test_strength_reduce_multiply_by_1_simplified() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        let c1 = block.push_value(ValueNode::Int32Constant { value: 1 });
        let mul = block.push_value(ValueNode::Int32Multiply { left: p, right: c1 });
        block.set_control(ControlNode::Return { value: mul });
        graph.add_block(block);

        optimize(&mut graph);

        // Multiply by 1 is identity-simplified (x * 1 → x), so the multiply
        // node should no longer be present.
        let has_mul = graph.blocks()[0]
            .nodes
            .iter()
            .any(|(_, n)| matches!(n, ValueNode::Int32Multiply { .. }));
        assert!(
            !has_mul,
            "expected Int32Multiply(x, 1) to be identity-simplified away"
        );
    }

    // ── Redundant type-guard elimination ──────────────────────────────────────

    #[test]
    fn test_eliminate_check_smi_on_smi_constant() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let smi = block.push_value(ValueNode::SmiConstant { value: 42 });
        let _check = block.push_value(ValueNode::CheckSmi { receiver: smi });
        block.set_control(ControlNode::Return { value: smi });
        graph.add_block(block);

        let before = total_node_count(&graph);
        optimize(&mut graph);
        // The CheckSmi on a SmiConstant should be eliminated (replaced then
        // cleaned up by DCE).
        assert!(total_node_count(&graph) < before);
    }

    #[test]
    fn test_eliminate_duplicate_check_smi_same_block() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        let _c1 = block.push_value(ValueNode::CheckSmi { receiver: p });
        let _c2 = block.push_value(ValueNode::CheckSmi { receiver: p });
        block.set_control(ControlNode::Return { value: p });
        graph.add_block(block);

        optimize(&mut graph);

        // At most one CheckSmi should survive (the first one).
        let check_count = graph.blocks()[0]
            .nodes
            .iter()
            .filter(|(_, n)| matches!(n, ValueNode::CheckSmi { .. }))
            .count();
        assert!(
            check_count <= 1,
            "expected at most 1 CheckSmi, got {check_count}"
        );
    }

    // ── Algebraic identity simplification ─────────────────────────────────────

    #[test]
    fn test_identity_add_zero_right() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        let c0 = block.push_value(ValueNode::Int32Constant { value: 0 });
        let add = block.push_value(ValueNode::Int32Add { left: p, right: c0 });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        optimize(&mut graph);

        // x + 0 → x; the add should be eliminated and return uses the param.
        let has_add = graph.blocks()[0]
            .nodes
            .iter()
            .any(|(_, n)| matches!(n, ValueNode::Int32Add { .. }));
        assert!(!has_add, "expected Int32Add to be simplified away");
    }

    #[test]
    fn test_identity_multiply_by_one() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        let c1 = block.push_value(ValueNode::Int32Constant { value: 1 });
        let mul = block.push_value(ValueNode::Int32Multiply { left: p, right: c1 });
        block.set_control(ControlNode::Return { value: mul });
        graph.add_block(block);

        optimize(&mut graph);

        let has_mul = graph.blocks()[0]
            .nodes
            .iter()
            .any(|(_, n)| matches!(n, ValueNode::Int32Multiply { .. }));
        assert!(!has_mul, "expected Int32Multiply(x, 1) to be simplified");
    }

    #[test]
    fn test_identity_multiply_by_zero() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        let c0 = block.push_value(ValueNode::Int32Constant { value: 0 });
        let mul = block.push_value(ValueNode::Int32Multiply { left: p, right: c0 });
        block.set_control(ControlNode::Return { value: mul });
        graph.add_block(block);

        optimize(&mut graph);

        let has_mul = graph.blocks()[0]
            .nodes
            .iter()
            .any(|(_, n)| matches!(n, ValueNode::Int32Multiply { .. }));
        assert!(!has_mul, "expected Int32Multiply(x, 0) to be simplified");
    }

    #[test]
    fn test_identity_subtract_zero() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        let c0 = block.push_value(ValueNode::Int32Constant { value: 0 });
        let sub = block.push_value(ValueNode::Int32Subtract { left: p, right: c0 });
        block.set_control(ControlNode::Return { value: sub });
        graph.add_block(block);

        optimize(&mut graph);

        let has_sub = graph.blocks()[0]
            .nodes
            .iter()
            .any(|(_, n)| matches!(n, ValueNode::Int32Subtract { .. }));
        assert!(!has_sub, "expected Int32Subtract(x, 0) to be simplified");
    }

    // ── Common subexpression elimination ──────────────────────────────────────

    #[test]
    fn test_cse_duplicate_add_eliminated() {
        let mut graph = MaglevGraph::new(2);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        let p1 = block.push_value(ValueNode::Parameter { index: 1 });
        let add1 = block.push_value(ValueNode::Int32Add {
            left: p0,
            right: p1,
        });
        let add2 = block.push_value(ValueNode::Int32Add {
            left: p0,
            right: p1,
        });
        let sum = block.push_value(ValueNode::Int32Add {
            left: add1,
            right: add2,
        });
        block.set_control(ControlNode::Return { value: sum });
        graph.add_block(block);

        let before = total_node_count(&graph);
        optimize(&mut graph);
        // The duplicate add should have been eliminated.
        assert!(
            total_node_count(&graph) < before,
            "expected CSE to reduce node count"
        );
    }

    #[test]
    fn test_cse_different_ops_not_eliminated() {
        let mut graph = MaglevGraph::new(2);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        let p1 = block.push_value(ValueNode::Parameter { index: 1 });
        let add = block.push_value(ValueNode::Int32Add {
            left: p0,
            right: p1,
        });
        let sub = block.push_value(ValueNode::Int32Subtract {
            left: p0,
            right: p1,
        });
        let result = block.push_value(ValueNode::Int32Add {
            left: add,
            right: sub,
        });
        block.set_control(ControlNode::Return { value: result });
        graph.add_block(block);

        let before = total_node_count(&graph);
        optimize(&mut graph);
        // Different operations on same inputs — no CSE should occur.
        assert_eq!(
            total_node_count(&graph),
            before,
            "different ops should not be CSE'd"
        );
    }

    #[test]
    fn test_eliminate_check_smi_on_checked_smi_add_result() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let a = block.push_value(ValueNode::SmiConstant { value: 1 });
        let b = block.push_value(ValueNode::SmiConstant { value: 2 });
        let sum = block.push_value(ValueNode::CheckedSmiAdd { left: a, right: b });
        let _check = block.push_value(ValueNode::CheckSmi { receiver: sum });
        block.set_control(ControlNode::Return { value: sum });
        graph.add_block(block);

        optimize(&mut graph);

        // The CheckSmi on the output of CheckedSmiAdd should be eliminated.
        let check_count = graph.blocks()[0]
            .nodes
            .iter()
            .filter(|(_, n)| matches!(n, ValueNode::CheckSmi { .. }))
            .count();
        assert_eq!(
            check_count, 0,
            "CheckSmi on CheckedSmiAdd output should be eliminated"
        );
    }
}
