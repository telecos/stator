//! Maglev IR optimisation passes.
//!
//! Nine passes are implemented and composed by [`optimize`]:
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
//! 4. **Trivial Phi elimination** — replaces Phi nodes whose non-self
//!    inputs are all the same value with that value.  This is essential for
//!    enabling LICM: loop-header Phis like `Phi(X, self)` for variables
//!    that are never modified in the loop body are replaced by `X`, moving
//!    their definition out of the loop and unblocking hoisting.
//!
//! 5. **Loop-invariant code motion (LICM)** — detects natural loops via
//!    back-edges and hoists pure nodes whose inputs are all defined outside
//!    the loop into the preheader block.
//!    See [`crate::compiler::maglev::licm`].
//!
//! 6. **Redundant type-guard elimination** — removes `CheckSmi` guards on
//!    values already known to be Smis.  A value is *known Smi* when it is
//!    produced by `SmiConstant` or any `CheckedSmi*` node, or when a prior
//!    `CheckSmi` for the same receiver already appears in the same block.
//!
//! 7. **Redundant `CheckMaps` removal** — within each basic block, a
//!    `CheckMaps { receiver, feedback_slot }` node is redundant if an
//!    identical guard for the *same* (receiver, feedback_slot) pair has
//!    already been emitted earlier in the same block.  The duplicate is
//!    replaced by a [`ValueNode::UndefinedConstant`] placeholder and the
//!    relevant ID is remapped so all consumers still compile correctly.
//!
//! 8. **Inlining analysis** — scans for `CallKnownFunction` nodes with
//!    small argument counts and marks them as inlining candidates for a
//!    future inlining pass.
//!
//! 9. **Dead-code elimination** — removes `ValueNode`s whose [`NodeId`]
//!    is never referenced by any other node (value or control) in the graph.
//!    Pure side-effect-free nodes that produce a value which nobody consumes
//!    are safe to drop.  Nodes with observable side-effects (stores, calls,
//!    allocations, guards/checks) are always considered *live* and are kept
//!    even when their result value is unused.
//!
//! # Usage
//!
//! ```
//! use stator_jse::compiler::maglev::ir::{
//!     BasicBlock, ControlNode, MaglevGraph, ValueNode,
//! };
//! use stator_jse::compiler::maglev::optimizer::optimize;
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
use std::sync::atomic::{AtomicU32, Ordering};

use crate::compiler::maglev::ir::{BasicBlock, ControlNode, MaglevGraph, NodeId, ValueNode};
use crate::compiler::maglev::licm;

// ── Global promotion diagnostics ─────────────────────────────────────────────

/// Total globals promoted by the optimizer (LoadGlobal/StoreGlobal → Phi).
static OPT_GLOBALS_PROMOTED: AtomicU32 = AtomicU32::new(0);
/// Loops where promotion was skipped because of user calls.
static OPT_GLOBALS_SKIPPED: AtomicU32 = AtomicU32::new(0);

/// Return (promoted, skipped) diagnostic counters for global promotion.
pub fn globals_promoted_diagnostics() -> (u32, u32) {
    (
        OPT_GLOBALS_PROMOTED.load(Ordering::Relaxed),
        OPT_GLOBALS_SKIPPED.load(Ordering::Relaxed),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry-point
// ─────────────────────────────────────────────────────────────────────────────

/// Run all optimisation passes on `graph` in place.
///
/// Passes are applied in the order: constant folding → strength reduction →
/// range analysis → strength reduction → trivial Phi elimination → LICM →
/// redundant type-guard elimination → inlining analysis →
/// redundant-CheckMaps removal → DCE.  Strength reduction runs twice: once
/// before range analysis for existing Int32 nodes, and once after for newly
/// lowered Int32 nodes (e.g. `GenericMultiply` → `Int32Multiply`).
///
/// Multiple rounds are *not* performed; a single sweep of each pass is
/// sufficient for the patterns targeted here.
pub fn optimize(graph: &mut MaglevGraph) {
    let _block_count = graph.blocks().len();
    let _node_count: usize = graph.blocks().iter().map(|b| b.nodes.len()).sum();

    fold_constants(graph);

    let _truncations = propagate_int32_truncation(graph);
    simplify_identities(graph);
    reassociate_arithmetic(graph);
    strength_reduce(graph);
    crate::compiler::maglev::range_analysis::eliminate_overflow_checks(graph);
    // Re-run strength reduction: range analysis may have lowered
    // GenericMultiply→Int32Multiply, exposing power-of-two patterns.
    strength_reduce(graph);
    // Re-run reassociation: range analysis lowered Generic→Int32 ops,
    // exposing patterns like (a + x*K1) - x*K2 → a + x*(K1-K2) that
    // the first reassociation pass could not see.
    reassociate_arithmetic(graph);
    eliminate_trivial_phis(graph);
    let _licm_hoisted = crate::compiler::maglev::licm::hoist_loop_invariants(graph);
    // TODO: implement loop peeling — execute the first iteration outside the
    // loop to establish type information (e.g. from CheckSmi guards), then
    // specialise the loop body based on proven types.  This requires
    // duplicating the loop header and body, which is non-trivial with the
    // current graph structure.
    eliminate_common_subexpressions(graph);
    let _globals_promoted = promote_loop_globals_counted(graph);
    eliminate_trivial_phis(graph);
    let _licm_hoisted2 = crate::compiler::maglev::licm::hoist_loop_invariants(graph);
    // Re-run range analysis after global promotion: promotion replaces
    // LoadGlobal/StoreGlobal with Phi nodes, exposing induction variable
    // and accumulator patterns that enable CheckedSmi→Int32 conversion.
    crate::compiler::maglev::range_analysis::eliminate_overflow_checks(graph);
    // Re-run strength reduction after second range analysis pass.
    strength_reduce(graph);
    // Re-run reassociation after second range analysis pass.
    reassociate_arithmetic(graph);
    // Fold invariant addition chains: after LICM has hoisted property loads
    // and range analysis has lowered Generic→Int32, loop bodies may contain
    // chains of Int32Add with loop-invariant operands (e.g. hoisted loads).
    // This groups them into a single pre-computed sum in the preheader,
    // reducing per-iteration additions from N to 1.
    let _inv_chains = crate::compiler::maglev::licm::fold_invariant_addition_chains(graph);
    // Re-run truncation after global promotion: promotion reduces
    // use-counts on arithmetic nodes — allowing CheckedSmi→Int32.
    let _truncations2 = propagate_int32_truncation(graph);
    eliminate_identity_operations_global(graph);
    eliminate_redundant_type_guards(graph);
    specialize_closure_calls(graph);
    mark_inlining_candidates(graph);
    remove_redundant_check_maps(graph);
    fuse_object_literal_stores(graph);
    eliminate_store_to_load(graph);
    eliminate_dead_object_stores(graph);
    eliminate_dead_allocations(graph);
    replace_dead_arguments(graph);
    unroll_simple_loops(graph);
    eliminate_dead_code(graph);
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

            // ── Int32 bitwise ────────────────────────────────────────────
            ValueNode::Int32BitwiseOr { left, right } => {
                fold_i32_bin(left, right, &consts, |a, b| a | b)
            }
            ValueNode::Int32BitwiseAnd { left, right } => {
                fold_i32_bin(left, right, &consts, |a, b| a & b)
            }
            ValueNode::Int32BitwiseXor { left, right } => {
                fold_i32_bin(left, right, &consts, |a, b| a ^ b)
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
    //    Both Int32 and Generic bitwise/shift ops truncate to i32 —
    //    Generic variants still call ToInt32 on their operands, which
    //    wraps to 32 bits.  Recognizing Generic truncation points lets
    //    the pass convert `(a + b) | 0` patterns where the graph
    //    builder emitted GenericAdd + GenericBitwiseOr.
    let mut replacements: HashMap<NodeId, ValueNode> = HashMap::new();
    for block in graph.blocks() {
        for (_node_id, node) in &block.nodes {
            match node {
                ValueNode::Int32BitwiseOr { left, right }
                | ValueNode::Int32BitwiseXor { left, right }
                | ValueNode::Int32BitwiseAnd { left, right }
                | ValueNode::Int32ShiftLeft { left, right }
                | ValueNode::Int32ShiftRight { left, right }
                | ValueNode::Int32ShiftRightLogical { left, right }
                | ValueNode::GenericBitwiseOr { left, right, .. }
                | ValueNode::GenericBitwiseXor { left, right, .. }
                | ValueNode::GenericBitwiseAnd { left, right, .. }
                | ValueNode::GenericShiftLeft { left, right, .. }
                | ValueNode::GenericShiftRight { left, right, .. }
                | ValueNode::GenericShiftRightLogical { left, right, .. } => {
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

/// Walk backward from a truncation point, replacing `CheckedSmi*` and
/// `Generic*` arithmetic with unchecked `Int32*`.  Only touches nodes
/// whose non-transparent use count is exactly one (the truncating path).
///
/// `Generic*` conversions are guarded: both inputs must be provably
/// integer-valued (Smi constants, Int32 ops, bitwise ops that always
/// produce i32, Phi nodes in smi loops, or nodes already converted by
/// this pass).  This prevents miscompilation when a Generic op's input
/// could be a string or object at runtime.
fn mark_truncated(
    id: NodeId,
    use_counts: &HashMap<NodeId, usize>,
    node_map: &HashMap<NodeId, ValueNode>,
    replacements: &mut HashMap<NodeId, ValueNode>,
) {
    if replacements.contains_key(&id) {
        return;
    }
    let uc = use_counts.get(&id);
    if uc != Some(&1) {
        return;
    }
    let Some(node) = node_map.get(&id) else {
        return;
    };

    // CheckedSmi* → Int32*: always safe (inputs already Smi-checked).
    // Convert immediately and recurse into inputs.
    match node {
        ValueNode::CheckedSmiAdd { left, right } => {
            replacements.insert(
                id,
                ValueNode::Int32Add {
                    left: *left,
                    right: *right,
                },
            );
            mark_truncated(*left, use_counts, node_map, replacements);
            mark_truncated(*right, use_counts, node_map, replacements);
            return;
        }
        ValueNode::CheckedSmiSubtract { left, right } => {
            replacements.insert(
                id,
                ValueNode::Int32Subtract {
                    left: *left,
                    right: *right,
                },
            );
            mark_truncated(*left, use_counts, node_map, replacements);
            mark_truncated(*right, use_counts, node_map, replacements);
            return;
        }
        ValueNode::CheckedSmiMultiply { left, right } => {
            replacements.insert(
                id,
                ValueNode::Int32Multiply {
                    left: *left,
                    right: *right,
                },
            );
            mark_truncated(*left, use_counts, node_map, replacements);
            mark_truncated(*right, use_counts, node_map, replacements);
            return;
        }
        ValueNode::CheckedSmiIncrement { value } => {
            replacements.insert(id, ValueNode::Int32Increment { value: *value });
            mark_truncated(*value, use_counts, node_map, replacements);
            return;
        }
        ValueNode::CheckedSmiDecrement { value } => {
            replacements.insert(id, ValueNode::Int32Decrement { value: *value });
            mark_truncated(*value, use_counts, node_map, replacements);
            return;
        }
        _ => {}
    }

    // Generic* → Int32*: recurse into inputs FIRST so they get converted
    // before we check is_provably_i32.  This allows chains like
    //   GenericBitwiseOr → GenericSub → GenericAdd → GenericMul → Phi
    // to be fully converted in a single backward walk.
    match node {
        ValueNode::GenericAdd { left, right, .. }
        | ValueNode::GenericSubtract { left, right, .. }
        | ValueNode::GenericMultiply { left, right, .. } => {
            mark_truncated(*left, use_counts, node_map, replacements);
            mark_truncated(*right, use_counts, node_map, replacements);
            if !is_provably_i32(*left, replacements, node_map)
                || !is_provably_i32(*right, replacements, node_map)
            {
                return;
            }
            let new_node = match node {
                ValueNode::GenericAdd { left, right, .. } => ValueNode::Int32Add {
                    left: *left,
                    right: *right,
                },
                ValueNode::GenericSubtract { left, right, .. } => ValueNode::Int32Subtract {
                    left: *left,
                    right: *right,
                },
                ValueNode::GenericMultiply { left, right, .. } => ValueNode::Int32Multiply {
                    left: *left,
                    right: *right,
                },
                _ => unreachable!(),
            };
            replacements.insert(id, new_node);
        }
        ValueNode::GenericIncrement { value, .. } => {
            mark_truncated(*value, use_counts, node_map, replacements);
            if !is_provably_i32(*value, replacements, node_map) {
                return;
            }
            replacements.insert(id, ValueNode::Int32Increment { value: *value });
        }
        ValueNode::GenericDecrement { value, .. } => {
            mark_truncated(*value, use_counts, node_map, replacements);
            if !is_provably_i32(*value, replacements, node_map) {
                return;
            }
            replacements.insert(id, ValueNode::Int32Decrement { value: *value });
        }
        _ => {}
    }
}

/// Returns `true` when `id` is known to produce an i32 value, making it
/// safe to feed into a wrapping `Int32*` operation.
fn is_provably_i32(
    id: NodeId,
    replacements: &HashMap<NodeId, ValueNode>,
    node_map: &HashMap<NodeId, ValueNode>,
) -> bool {
    // Already converted to Int32 by this pass.
    if replacements.contains_key(&id) {
        return true;
    }
    let Some(node) = node_map.get(&id) else {
        return false;
    };
    matches!(
        node,
        // Constants are always i32.
        ValueNode::SmiConstant { .. }
            | ValueNode::Int32Constant { .. }
            // Int32 operations always produce i32.
            | ValueNode::Int32Add { .. }
            | ValueNode::Int32Subtract { .. }
            | ValueNode::Int32Multiply { .. }
            | ValueNode::Int32Negate { .. }
            | ValueNode::Int32Increment { .. }
            | ValueNode::Int32Decrement { .. }
            | ValueNode::Int32BitwiseOr { .. }
            | ValueNode::Int32BitwiseAnd { .. }
            | ValueNode::Int32BitwiseXor { .. }
            | ValueNode::Int32ShiftLeft { .. }
            | ValueNode::Int32ShiftRight { .. }
            | ValueNode::Int32ShiftRightLogical { .. }
            // JS bitwise ops always produce i32 (ToInt32 per spec).
            | ValueNode::GenericBitwiseOr { .. }
            | ValueNode::GenericBitwiseAnd { .. }
            | ValueNode::GenericBitwiseXor { .. }
            | ValueNode::GenericBitwiseNot { .. }
            // Phi in smi loops: preheader is SmiConstant, back-edge is
            // i32 result (BitwiseOr or smi-guarded arithmetic).  Safe
            // because the smi-guarded load point already deopts on
            // non-integer values.
            | ValueNode::Phi { .. }
    )
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
        | ValueNode::Int32Equal { left, right }
        | ValueNode::Int32StrictEqual { left, right }
        | ValueNode::Float64Equal { left, right }
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
        ValueNode::CreateObjectLiteralWithProperties { values, .. } => {
            for &v in values {
                f(v);
            }
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
        ValueNode::PushContext { context } | ValueNode::PopContext { context } => f(*context),
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
// Trivial Phi elimination
// ─────────────────────────────────────────────────────────────────────────────

/// Eliminate trivial Phi nodes whose value is statically determined.
///
/// A Phi node is *trivial* when every non-self input refers to the same
/// [`NodeId`] `X`.  Such a Phi always evaluates to `X` regardless of which
/// predecessor is taken, so all uses of the Phi can be replaced by `X`.
/// This commonly arises in loop headers for variables that are never
/// modified inside the loop body (e.g. `Phi(root, self)` for an object
/// reference that is only read).
///
/// Eliminating trivial Phis is a prerequisite for effective LICM: a
/// [`ValueNode::LoadNamedGeneric`] whose receiver is a trivial Phi in the
/// loop header appears to depend on a loop-body definition.  Once the Phi
/// is replaced, the load's input lives in the preheader and becomes
/// eligible for hoisting.
///
/// The pass iterates to a fixed point so that cascading trivial Phis
/// (where one Phi's sole external input is another trivial Phi) are fully
/// resolved.
fn eliminate_trivial_phis(graph: &mut MaglevGraph) {
    // Iterate to a fixed point for cascading trivial Phis.
    loop {
        // Step 1: Scan all blocks for trivial Phi nodes.
        let mut subst: HashMap<NodeId, NodeId> = HashMap::new();
        for block in graph.blocks() {
            for &(id, ref node) in &block.nodes {
                if let ValueNode::Phi { inputs } = node {
                    // Collect unique non-self inputs.
                    let mut unique_external = None;
                    let mut is_trivial = true;
                    for &inp in inputs {
                        if inp == id {
                            continue; // skip self-references
                        }
                        match unique_external {
                            None => unique_external = Some(inp),
                            Some(prev) if prev == inp => {} // same as before
                            Some(_) => {
                                is_trivial = false;
                                break;
                            }
                        }
                    }
                    if is_trivial && let Some(replacement) = unique_external {
                        subst.insert(id, replacement);
                    }
                    // If unique_external is None, all inputs are self-refs
                    // (unreachable Phi) — leave it for DCE.
                }
            }
        }

        if subst.is_empty() {
            break;
        }

        // Transitively resolve substitution chains: if A→B and B→C, then
        // A→C. Limit iterations to avoid infinite loops from malformed IR.
        for _ in 0..subst.len() {
            let mut changed = false;
            let keys: Vec<NodeId> = subst.keys().copied().collect();
            for key in keys {
                if let Some(&further) = subst.get(&subst[&key])
                    && subst[&key] != further
                {
                    subst.insert(key, further);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // Step 2: Replace trivial Phi nodes with dead placeholders.
        for block in graph.blocks_mut() {
            for (id, node) in &mut block.nodes {
                if subst.contains_key(id) {
                    *node = ValueNode::UndefinedConstant;
                }
            }
        }

        // Step 3: Apply substitution to all node inputs and control nodes.
        for block in graph.blocks_mut() {
            apply_subst_to_block(block, &subst);
        }
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
            // x | 0 → x, 0 | x → x  (Int32 only)
            //
            // GenericBitwiseOr is intentionally NOT folded here: it serves
            // as a ToInt32 truncation point for `propagate_int32_truncation`.
            // The first truncation pass may not be able to convert the chain
            // (e.g. when LoadGlobal is not provably i32).  After
            // `promote_loop_globals_counted` replaces LoadGlobal with Phi
            // nodes, the *second* truncation pass needs the BitwiseOr marker
            // still present to walk backward and convert GenericAdd/Sub/Mul
            // to their Int32 equivalents.  `eliminate_identity_operations_global`
            // cleans up any remaining `GenericBitwiseOr(x, 0)` afterward.
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
            ValueNode::Int32BitwiseXor { left, right }
            | ValueNode::GenericBitwiseXor { left, right, .. } => {
                if consts.get(right) == Some(&0) {
                    Some(*left)
                } else if consts.get(left) == Some(&0) {
                    Some(*right)
                } else {
                    None
                }
            }
            // x & -1 → x, -1 & x → x
            ValueNode::Int32BitwiseAnd { left, right }
            | ValueNode::GenericBitwiseAnd { left, right, .. } => {
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
// Pass 1.25b — Cross-block identity elimination
// ─────────────────────────────────────────────────────────────────────────────

/// Eliminate identity operations whose constant operand lives in a different
/// block than the arithmetic node.
///
/// The block-local [`simplify_identities`] pass only sees constants defined
/// within the same basic block.  This global pass collects *all* constants
/// across the entire graph and then replaces identity patterns such as
/// `x | 0`, `x & -1`, `x + 0`, `x - 0`, and `x * 1` regardless of which
/// block the constant was defined in.  Remaining dead placeholders are
/// cleaned up by the later DCE pass.
fn eliminate_identity_operations_global(graph: &mut MaglevGraph) {
    // Step 1: Collect all integer constants across every block.
    let mut consts: HashMap<NodeId, i32> = HashMap::new();
    for block in graph.blocks() {
        for (id, node) in &block.nodes {
            match node {
                ValueNode::Int32Constant { value } | ValueNode::SmiConstant { value } => {
                    consts.insert(*id, *value);
                }
                _ => {}
            }
        }
    }

    if consts.is_empty() {
        return;
    }

    // Step 2: Scan all blocks for identity patterns, building a global
    // substitution map and replacing matched nodes with dead placeholders.
    let mut subst: HashMap<NodeId, NodeId> = HashMap::new();
    for block in graph.blocks_mut() {
        for (id, node) in &mut block.nodes {
            let replacement: Option<NodeId> = match node {
                // x | 0 → x, 0 | x → x
                ValueNode::Int32BitwiseOr { left, right }
                | ValueNode::GenericBitwiseOr { left, right, .. } => {
                    if consts.get(right) == Some(&0) {
                        Some(*left)
                    } else if consts.get(left) == Some(&0) {
                        Some(*right)
                    } else {
                        None
                    }
                }
                // x & -1 → x, -1 & x → x
                ValueNode::Int32BitwiseAnd { left, right }
                | ValueNode::GenericBitwiseAnd { left, right, .. } => {
                    if consts.get(right) == Some(&-1) {
                        Some(*left)
                    } else if consts.get(left) == Some(&-1) {
                        Some(*right)
                    } else {
                        None
                    }
                }
                // x + 0 → x, 0 + x → x
                ValueNode::Int32Add { left, right }
                | ValueNode::CheckedSmiAdd { left, right }
                | ValueNode::GenericAdd { left, right, .. } => {
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
                | ValueNode::CheckedSmiSubtract { left, right }
                | ValueNode::GenericSubtract { left, right, .. } => {
                    if consts.get(right) == Some(&0) {
                        Some(*left)
                    } else {
                        None
                    }
                }
                // x * 1 → x, 1 * x → x
                ValueNode::Int32Multiply { left, right }
                | ValueNode::CheckedSmiMultiply { left, right }
                | ValueNode::GenericMultiply { left, right, .. } => {
                    if consts.get(right) == Some(&1) {
                        Some(*left)
                    } else if consts.get(left) == Some(&1) {
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
    }

    if subst.is_empty() {
        return;
    }

    // Step 3: Apply substitutions across all blocks so every reference to
    // an eliminated node is redirected to its non-constant operand.
    for block in graph.blocks_mut() {
        apply_subst_to_block(block, &subst);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1.35 — Arithmetic reassociation
// ─────────────────────────────────────────────────────────────────────────────

/// Combine multiply-add/sub patterns into simpler multiplies.
///
/// Recognized patterns (all for `Int32` variants):
///
/// - `(x * K) + x` → `x * (K + 1)`
/// - `x + (x * K)` → `x * (K + 1)`
/// - `(x * K) - x` → `x * (K - 1)` when K > 1
/// - `(a - x) + (x * K)` → `a + x * (K - 1)` when K > 1
/// - `(a + (x * K1)) + x` → `a + x * (K1 + 1)`
/// - `x + (a + (x * K1))` → `a + x * (K1 + 1)`
///
/// The pass runs iteratively until convergence so that chained patterns
/// (e.g. `i*3 - i + i*2` → `i*4`) are fully simplified.
fn reassociate_arithmetic(graph: &mut MaglevGraph) {
    loop {
        let changed = reassociate_arithmetic_once(graph);
        if changed == 0 {
            break;
        }
    }
}

fn reassociate_arithmetic_once(graph: &mut MaglevGraph) -> usize {
    let mut next_id = graph
        .blocks()
        .iter()
        .flat_map(|b| b.nodes.iter())
        .map(|(id, _)| id.0)
        .max()
        .map_or(0, |m| m + 1);

    let mut count = 0;
    for block in graph.blocks_mut() {
        count += reassociate_block(block, &mut next_id);
    }
    count
}

/// Describes a single reassociation replacement.
struct Reassoc {
    pos: usize,
    new_node: ValueNode,
    prefix: Vec<(NodeId, ValueNode)>,
}

/// Build helper maps and apply one reassociation pattern within one block.
fn reassociate_block(block: &mut BasicBlock, next_id: &mut u32) -> usize {
    let mut consts: HashMap<NodeId, i32> = HashMap::new();
    let mut mul_info: HashMap<NodeId, (NodeId, i32)> = HashMap::new();
    let mut node_defs: HashMap<NodeId, ValueNode> = HashMap::new();

    for (id, node) in &block.nodes {
        match node {
            ValueNode::Int32Constant { value } | ValueNode::SmiConstant { value } => {
                consts.insert(*id, *value);
            }
            _ => {}
        }
        if let ValueNode::Int32Multiply { left, right } = node {
            if let Some(&k) = consts.get(right) {
                mul_info.insert(*id, (*left, k));
            } else if let Some(&k) = consts.get(left) {
                mul_info.insert(*id, (*right, k));
            }
        }
        node_defs.insert(*id, node.clone());
    }

    for (pos, (_id, node)) in block.nodes.iter().enumerate() {
        if let Some(r) = try_reassociate(node, pos, &consts, &mul_info, &node_defs, next_id) {
            return apply_reassoc(block, r);
        }
    }
    0
}

/// Try to match a reassociation pattern for the node at `pos`.
fn try_reassociate(
    node: &ValueNode,
    pos: usize,
    consts: &HashMap<NodeId, i32>,
    mul_info: &HashMap<NodeId, (NodeId, i32)>,
    node_defs: &HashMap<NodeId, ValueNode>,
    next_id: &mut u32,
) -> Option<Reassoc> {
    match node {
        ValueNode::Int32Add { left, right } => {
            // (x * K) + x  ->  x * (K+1)
            if let Some(&(base, k)) = mul_info.get(left)
                && base == *right
            {
                return build_multiply_reassoc(pos, base, k + 1, consts, next_id);
            }
            // x + (x * K)  ->  x * (K+1)
            if let Some(&(base, k)) = mul_info.get(right)
                && base == *left
            {
                return build_multiply_reassoc(pos, base, k + 1, consts, next_id);
            }
            // (a - x) + (x * K)  ->  a + x * (K-1)  when K > 1
            if let Some(ValueNode::Int32Subtract {
                left: sub_a,
                right: sub_x,
            }) = node_defs.get(left)
                && let Some(&(base, k)) = mul_info.get(right)
                && base == *sub_x
                && k > 1
            {
                return build_add_mul_reassoc(pos, *sub_a, base, k - 1, consts, next_id);
            }
            // (a + x*K) + x  ->  a + x*(K+1)
            if let Some(ValueNode::Int32Add {
                left: add_a,
                right: add_b,
            }) = node_defs.get(left)
            {
                if let Some(&(base, k)) = mul_info.get(add_b)
                    && base == *right
                {
                    return build_add_mul_reassoc(pos, *add_a, base, k + 1, consts, next_id);
                }
                if let Some(&(base, k)) = mul_info.get(add_a)
                    && base == *right
                {
                    return build_add_mul_reassoc(pos, *add_b, base, k + 1, consts, next_id);
                }
            }
            // x + (a + x*K)  ->  a + x*(K+1)
            if let Some(ValueNode::Int32Add {
                left: add_a,
                right: add_b,
            }) = node_defs.get(right)
            {
                if let Some(&(base, k)) = mul_info.get(add_b)
                    && base == *left
                {
                    return build_add_mul_reassoc(pos, *add_a, base, k + 1, consts, next_id);
                }
                if let Some(&(base, k)) = mul_info.get(add_a)
                    && base == *left
                {
                    return build_add_mul_reassoc(pos, *add_b, base, k + 1, consts, next_id);
                }
            }
            None
        }
        ValueNode::Int32Subtract { left, right } => {
            // (x * K) - x  ->  x * (K-1)  when K > 1
            if let Some(&(base, k)) = mul_info.get(left)
                && base == *right
                && k > 1
            {
                return build_multiply_reassoc(pos, base, k - 1, consts, next_id);
            }
            // (x * K1) - (x * K2)  ->  x * (K1-K2)  when same base, K1 > K2
            if let Some(&(base_l, k_l)) = mul_info.get(left)
                && let Some(&(base_r, k_r)) = mul_info.get(right)
                && base_l == base_r
                && k_l > k_r
            {
                return build_multiply_reassoc(pos, base_l, k_l - k_r, consts, next_id);
            }
            // (a + x * K1) - (x * K2)  ->  a + x * (K1-K2)  when same base
            if let Some(&(base_r, k_r)) = mul_info.get(right)
                && let Some(ValueNode::Int32Add {
                    left: add_a,
                    right: add_b,
                }) = node_defs.get(left)
            {
                if let Some(&(base_l, k_l)) = mul_info.get(add_b)
                    && base_l == base_r
                    && k_l >= k_r
                {
                    return build_add_mul_reassoc(pos, *add_a, base_r, k_l - k_r, consts, next_id);
                }
                if let Some(&(base_l, k_l)) = mul_info.get(add_a)
                    && base_l == base_r
                    && k_l >= k_r
                {
                    return build_add_mul_reassoc(pos, *add_b, base_r, k_l - k_r, consts, next_id);
                }
            }
            None
        }
        _ => None,
    }
}

/// Build a `Reassoc` that replaces the node at `pos` with `base * new_k`.
fn build_multiply_reassoc(
    pos: usize,
    base: NodeId,
    new_k: i32,
    consts: &HashMap<NodeId, i32>,
    next_id: &mut u32,
) -> Option<Reassoc> {
    if new_k <= 0 {
        return None;
    }
    let (const_node_id, prefix) = find_or_create_const(new_k, consts, next_id);
    Some(Reassoc {
        pos,
        new_node: ValueNode::Int32Multiply {
            left: base,
            right: const_node_id,
        },
        prefix,
    })
}

/// Build a `Reassoc` replacing the node at `pos` with `addend + base * new_k`.
fn build_add_mul_reassoc(
    pos: usize,
    addend: NodeId,
    base: NodeId,
    new_k: i32,
    consts: &HashMap<NodeId, i32>,
    next_id: &mut u32,
) -> Option<Reassoc> {
    if new_k <= 0 {
        return None;
    }
    if new_k == 1 {
        return Some(Reassoc {
            pos,
            new_node: ValueNode::Int32Add {
                left: addend,
                right: base,
            },
            prefix: Vec::new(),
        });
    }
    let (const_node_id, mut prefix) = find_or_create_const(new_k, consts, next_id);
    let mul_id = NodeId(*next_id);
    *next_id += 1;
    prefix.push((
        mul_id,
        ValueNode::Int32Multiply {
            left: base,
            right: const_node_id,
        },
    ));
    Some(Reassoc {
        pos,
        new_node: ValueNode::Int32Add {
            left: addend,
            right: mul_id,
        },
        prefix,
    })
}

/// Find an existing constant with value `k`, or allocate a new one.
fn find_or_create_const(
    k: i32,
    consts: &HashMap<NodeId, i32>,
    next_id: &mut u32,
) -> (NodeId, Vec<(NodeId, ValueNode)>) {
    if let Some((&id, _)) = consts.iter().find(|&(_, &v)| v == k) {
        (id, Vec::new())
    } else {
        let id = NodeId(*next_id);
        *next_id += 1;
        (id, vec![(id, ValueNode::Int32Constant { value: k })])
    }
}

/// Apply a single `Reassoc` to the block.
fn apply_reassoc(block: &mut BasicBlock, r: Reassoc) -> usize {
    let (node_id, _) = block.nodes[r.pos];
    let prefix_len = r.prefix.len();
    for (i, entry) in r.prefix.into_iter().enumerate() {
        block.nodes.insert(r.pos + i, entry);
    }
    block.nodes[r.pos + prefix_len] = (node_id, r.new_node);
    1
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
    /// A nullary operation keyed only by kind and a u32 index (e.g. LoadGlobal
    /// keyed by its name index).
    Nullary(u16, u32),
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
        // Nullary: keyed only by name index.
        ValueNode::LoadGlobal { name, .. } => Some(CseKey::Nullary(1, *name)),
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
        return 0;
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
                OPT_GLOBALS_SKIPPED.fetch_add(1, Ordering::Relaxed);
                return 0;
            }
        }
    }

    // Collect LoadGlobal and StoreGlobal occurrences inside the loop.
    let mut load_names: HashSet<u32> = HashSet::new();
    let mut store_names: HashSet<u32> = HashSet::new();

    // Details keyed by name index.
    let mut load_info: HashMap<u32, (u32, Vec<NodeId>)> = HashMap::new(); // name -> (feedback_slot, [NodeId])
    let mut store_info: HashMap<u32, (NodeId, Vec<NodeId>, u32)> = HashMap::new(); // name -> (last value, [NodeId], feedback_slot)

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
                    feedback_slot,
                } => {
                    store_names.insert(*name);
                    let entry =
                        store_info
                            .entry(*name)
                            .or_insert((*value, Vec::new(), *feedback_slot));
                    // Update the value to the latest store (last writer wins).
                    entry.0 = *value;
                    entry.1.push(*id);
                }
                _ => {}
            }
        }
    }

    // Promotable globals: any global stored inside the loop (read-write OR
    // write-only).  Read-only globals are already handled by LICM, so we only
    // need to promote those that have at least one StoreGlobal in the body.
    let promotable: Vec<u32> = store_names.iter().copied().collect();
    if promotable.is_empty() {
        return 0;
    }

    // Scan the preheader for StoreGlobal nodes that set known initial values.
    // When a promotable global was stored in the preheader (e.g. `var i = 0`),
    // we can use the stored value directly as the Phi's entry input instead of
    // inserting a LoadGlobal.  This gives exact range information (e.g. [0,0]
    // for a SmiConstant) which enables overflow-check elimination downstream.
    let preheader_stores: HashMap<u32, NodeId> = {
        let mut stores = HashMap::new();
        if let Some(pb) = graph.blocks().iter().find(|b| b.id == lp.preheader) {
            for (_, node) in &pb.nodes {
                if let ValueNode::StoreGlobal { name, value, .. } = node {
                    // Last writer wins (insert overwrites earlier entries).
                    stores.insert(*name, *value);
                }
            }
        }
        stores
    };

    // Build PromotedGlobal entries: allocate preheader loads and Phi IDs.
    let mut promoted: Vec<PromotedGlobal> = Vec::new();
    for &name in &promotable {
        let (store_value, ref store_ids, store_feedback_slot) = store_info[&name];

        // For read-write globals, use feedback_slot from LoadGlobal; for
        // write-only globals, fall back to the StoreGlobal's feedback_slot.
        let (feedback_slot, load_ids) = match load_info.get(&name) {
            Some((fs, ids)) => (*fs, ids.clone()),
            None => (store_feedback_slot, Vec::new()),
        };

        // If the preheader stores a known value to this global, use that
        // value directly (gives exact range).  Otherwise fall back to a
        // LoadGlobal which will be seeded with I32_FULL.
        let preheader_load_id = if let Some(&init_value) = preheader_stores.get(&name) {
            init_value
        } else {
            graph
                .add_value_node(
                    lp.preheader,
                    ValueNode::LoadGlobal {
                        name,
                        feedback_slot,
                    },
                )
                .expect("preheader block must exist")
        };

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
    OPT_GLOBALS_PROMOTED.fetch_add(promotable.len() as u32, Ordering::Relaxed);
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
// Pass 2.4 — Monomorphic closure call specialization
// ─────────────────────────────────────────────────────────────────────────────

/// Convert `Call` nodes whose callee provably originates from a single
/// `FastCreateClosure` or `CreateClosure` into `CallKnownFunction`, enabling
/// the code generator to emit a direct JIT-to-JIT call instead of going
/// through IC dispatch.
///
/// The pass works in two phases:
///
/// 1. **Collect** — scan all blocks and record every `NodeId` that is produced
///    by `FastCreateClosure` or `CreateClosure`.  Additionally, resolve `Phi`
///    nodes whose non-self inputs all trace back to the *same* closure origin
///    (monomorphic phi).
///
/// 2. **Rewrite** — for every `Call` node whose `callee` is in the closure
///    set, replace the node in-place with `CallKnownFunction`.
fn specialize_closure_calls(graph: &mut MaglevGraph) {
    // Phase 1: collect all node IDs that produce a closure value.
    let mut closure_nodes: HashSet<NodeId> = HashSet::new();

    for block in graph.blocks() {
        for &(id, ref node) in &block.nodes {
            if matches!(
                node,
                ValueNode::FastCreateClosure { .. } | ValueNode::CreateClosure { .. }
            ) {
                closure_nodes.insert(id);
            }
        }
    }

    if closure_nodes.is_empty() {
        return;
    }

    // Resolve Phi nodes that funnel a single closure definition.  We iterate
    // to a fixed point because Phis can reference other Phis.
    let mut phi_map: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for block in graph.blocks() {
        for &(id, ref node) in &block.nodes {
            if let ValueNode::Phi { inputs } = node {
                phi_map.insert(id, inputs.clone());
            }
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for (&phi_id, inputs) in &phi_map {
            if closure_nodes.contains(&phi_id) {
                continue;
            }
            // Check if every non-self input is either a closure node or
            // another phi already proven to be a closure.
            let all_closure = inputs
                .iter()
                .all(|&inp| inp == phi_id || closure_nodes.contains(&inp));
            if all_closure && !inputs.is_empty() {
                // Ensure at least one input is not self-referencing.
                let has_non_self = inputs.iter().any(|&inp| inp != phi_id);
                if has_non_self {
                    closure_nodes.insert(phi_id);
                    changed = true;
                }
            }
        }
    }

    // Phase 2: rewrite Call → CallKnownFunction for monomorphic closure calls.
    for block in graph.blocks_mut() {
        for (_id, node) in &mut block.nodes {
            if let ValueNode::Call {
                callee,
                receiver,
                args,
                ..
            } = node
                && closure_nodes.contains(callee)
            {
                *node = ValueNode::CallKnownFunction {
                    callee: *callee,
                    receiver: *receiver,
                    args: args.clone(),
                };
            }
        }
    }
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
// Object-literal + StoreNamedGeneric fusion
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum number of property stores that can be fused into a single
/// [`CreateObjectLiteralWithProperties`] node.  The SysV AMD64 calling
/// convention provides 6 register arguments; we use RDI for
/// `feedback_slot`, RSI for packed `names` (12-bit encoding fits 5
/// indices), RDX/RCX/R8/R9 for 4 values, and the stack for the 5th.
///
/// Name indices must fit in 12 bits (≤ 4095); properties with larger
/// constant-pool indices are not fused.
pub const MAX_FUSED_OBJECT_PROPS: usize = 5;

/// Fuse `CreateObjectLiteral` + consecutive `StoreNamedGeneric` into a
/// single [`CreateObjectLiteralWithProperties`] node.
///
/// The pattern detected is:
///
/// ```text
/// %obj = CreateObjectLiteral { feedback_slot, flags, .. }
/// StoreNamedGeneric { object: %obj, name: n0, value: v0, .. }
/// StoreNamedGeneric { object: %obj, name: n1, value: v1, .. }
/// …
/// ```
///
/// The fused stores are removed and the `CreateObjectLiteral` is replaced
/// in-place with a `CreateObjectLiteralWithProperties` that carries the
/// property names and values.  The runtime stub creates the object and
/// fills all properties in a single call, eliminating per-store TLS
/// accesses and function-call overhead.
fn fuse_object_literal_stores(graph: &mut MaglevGraph) {
    for block in graph.blocks_mut() {
        fuse_object_literal_stores_in_block(block);
    }
}

/// Per-block implementation of object-literal store fusion.
fn fuse_object_literal_stores_in_block(block: &mut BasicBlock) {
    // Indices of StoreNamedGeneric nodes consumed by fusion (to remove).
    let mut remove_indices: Vec<usize> = Vec::new();
    // Replacements: (index, new_node).
    let mut replacements: Vec<(usize, ValueNode)> = Vec::new();

    let nodes = &block.nodes;
    let len = nodes.len();
    let mut i = 0;
    while i < len {
        let (create_id, create_node) = &nodes[i];

        let (feedback_slot, flags) = match create_node {
            ValueNode::CreateObjectLiteral {
                feedback_slot,
                flags,
                ..
            } => (*feedback_slot, *flags),
            // Extend fusion to CreateEmptyObjectLiteral.  Use u32::MAX as
            // a sentinel so the codegen emits feedback_slot = -1, which
            // tells the runtime stub to skip the template cache entirely.
            // Without this, every fused CreateEmptyObjectLiteral shares
            // feedback_slot 0, causing template-cache collisions when
            // multiple empty-object literals appear in the same function
            // (e.g. nested object initializers like `{ a: { b: ... } }`).
            ValueNode::CreateEmptyObjectLiteral => (u32::MAX, 0),
            _ => {
                i += 1;
                continue;
            }
        };

        // Scan consecutive StoreNamedGeneric nodes that target this object.
        let create_node_id = *create_id;
        let mut names: Vec<u32> = Vec::new();
        let mut values: Vec<NodeId> = Vec::new();
        let mut j = i + 1;

        // Track the actual indices of the StoreNamedGeneric nodes we consume,
        // since pure-value nodes (SmiConstant, etc.) may be interleaved.
        let mut store_indices: Vec<usize> = Vec::new();

        while j < len && names.len() < MAX_FUSED_OBJECT_PROPS {
            let node = &nodes[j].1;
            match node {
                ValueNode::StoreNamedGeneric {
                    object,
                    name,
                    value,
                    ..
                } if *object == create_node_id => {
                    // Name index must fit in the 12-bit packed encoding.
                    if *name > 0xFFF {
                        break;
                    }
                    names.push(*name);
                    values.push(*value);
                    store_indices.push(j);
                    j += 1;
                }
                // Skip any interleaved node that does NOT reference the
                // newly created object.  The previous code only skipped
                // constant nodes, which broke fusion when value
                // computations (Add, Mul, …) appeared between stores —
                // e.g. `{ x: i, y: i + 1, z: i * 2 }`.  Because the
                // object is freshly created and only reachable via the
                // stores being fused, any node that doesn't mention it
                // cannot observe or mutate it, so it is safe to skip.
                _ => {
                    let mut refs_object = false;
                    visit_value_node_inputs(node, &mut |id| {
                        if id == create_node_id {
                            refs_object = true;
                        }
                    });
                    if refs_object {
                        // Some non-store node uses the object — stop.
                        break;
                    }
                    j += 1;
                }
            }
        }

        if names.is_empty() {
            i += 1;
            continue;
        }

        // Record indices of consumed StoreNamedGeneric nodes for removal.
        remove_indices.extend_from_slice(&store_indices);

        // Replace CreateObjectLiteral with the fused node.
        replacements.push((
            i,
            ValueNode::CreateObjectLiteralWithProperties {
                feedback_slot,
                flags,
                names,
                values,
            },
        ));

        // Skip past the fused stores.
        i = j;
    }

    // Apply replacements (non-overlapping with removals).
    for (idx, new_node) in replacements {
        block.nodes[idx].1 = new_node;
    }
    // Remove consumed stores in reverse order to keep indices valid.
    remove_indices.sort_unstable();
    for &idx in remove_indices.iter().rev() {
        block.nodes.remove(idx);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2b — Store-to-load forwarding
// ─────────────────────────────────────────────────────────────────────────────

/// Forward values from `StoreNamedGeneric` (and `CreateObjectLiteralWithProperties`)
/// to subsequent `LoadNamedGeneric` on the same object and property name,
/// eliminating redundant FFI round-trips.
///
/// Within a single basic block, we track the most recent value stored to each
/// `(object, name)` pair.  When a `LoadNamedGeneric` is encountered whose
/// `(object, name)` is in the map, the load is replaced with an
/// `UndefinedConstant` placeholder and all references to its result are
/// redirected to the stored value.
///
/// The map is conservatively invalidated on any side-effecting node that is
/// not itself a `StoreNamedGeneric` or `CreateObjectLiteralWithProperties`,
/// since calls, other stores, etc. may mutate arbitrary object properties.
fn eliminate_store_to_load(graph: &mut MaglevGraph) {
    let mut global_subst: HashMap<NodeId, NodeId> = HashMap::new();

    // Propagate store maps, global maps, and alias maps across blocks for
    // cross-block forwarding.  For single-predecessor blocks we inherit
    // the predecessor's final maps, enabling stores in the entry block to
    // be forwarded to loads in the preheader/loop body.  For merge points
    // (multiple predecessors) we conservatively start empty.
    let mut block_store_maps: HashMap<u32, HashMap<(NodeId, u32), NodeId>> = HashMap::new();
    let mut block_global_maps: HashMap<u32, HashMap<u32, NodeId>> = HashMap::new();
    let mut block_alias_maps: HashMap<u32, HashMap<NodeId, NodeId>> = HashMap::new();

    // Phase 1: compute substitutions (read-only access to blocks).
    for block in graph.blocks() {
        let single_pred = if block.predecessors.len() == 1 {
            Some(block.predecessors[0])
        } else {
            None
        };
        let mut store_map: HashMap<(NodeId, u32), NodeId> = single_pred
            .and_then(|p| block_store_maps.get(&p).cloned())
            .unwrap_or_default();
        let mut global_map: HashMap<u32, NodeId> = single_pred
            .and_then(|p| block_global_maps.get(&p).cloned())
            .unwrap_or_default();
        let mut alias_map: HashMap<NodeId, NodeId> = single_pred
            .and_then(|p| block_alias_maps.get(&p).cloned())
            .unwrap_or_default();

        let subst = compute_store_to_load_subst_with_map(
            block,
            &mut store_map,
            &mut global_map,
            &mut alias_map,
        );
        block_store_maps.insert(block.id, store_map);
        block_global_maps.insert(block.id, global_map);
        block_alias_maps.insert(block.id, alias_map);
        global_subst.extend(subst);
    }

    if global_subst.is_empty() {
        return;
    }

    // Phase 2: replace forwarded loads with dead placeholders and apply
    // substitutions to ALL blocks so cross-block references are updated.
    for block in graph.blocks_mut() {
        for (id, node) in &mut block.nodes {
            if global_subst.contains_key(id) {
                *node = ValueNode::UndefinedConstant;
            }
        }
        apply_subst_to_block(block, &global_subst);
    }
}

/// Compute store-to-load forwarding substitutions for a block, using the
/// supplied `store_map` (which may be pre-seeded with entries from dominating
/// blocks).  The map is updated in place so callers can propagate it to
/// successor blocks.
fn compute_store_to_load_subst_with_map(
    block: &BasicBlock,
    store_map: &mut HashMap<(NodeId, u32), NodeId>,
    global_map: &mut HashMap<u32, NodeId>,
    alias_map: &mut HashMap<NodeId, NodeId>,
) -> HashMap<NodeId, NodeId> {
    let mut subst: HashMap<NodeId, NodeId> = HashMap::new();

    for (id, node) in &block.nodes {
        match node {
            // Seed the map from fused object creation.
            ValueNode::CreateObjectLiteralWithProperties { names, values, .. } => {
                let obj_id = *id;
                for (name, value) in names.iter().zip(values.iter()) {
                    store_map.insert((obj_id, *name), *value);
                }
            }

            // Track global stores for global forwarding.
            ValueNode::StoreGlobal { name, value, .. } => {
                global_map.insert(*name, *value);
            }

            // Forward global loads.
            ValueNode::LoadGlobal { name, .. } => {
                if let Some(&stored_value) = global_map.get(name) {
                    subst.insert(*id, stored_value);
                    alias_map.insert(*id, stored_value);
                }
            }

            // Track named property stores.
            ValueNode::StoreNamedGeneric {
                object,
                name,
                value,
                ..
            } => {
                // Resolve aliases: if object was a forwarded LoadGlobal,
                // use the original value for the store_map key.
                let resolved_obj = alias_map.get(object).copied().unwrap_or(*object);
                store_map.insert((resolved_obj, *name), *value);
            }

            // Forward named property loads.
            ValueNode::LoadNamedGeneric { object, name, .. } => {
                // Resolve aliases: if object was a forwarded LoadGlobal or
                // a prior forwarded LoadNamedGeneric, look up properties
                // on the original object.
                let resolved_obj = alias_map.get(object).copied().unwrap_or(*object);
                if let Some(&stored_value) = store_map.get(&(resolved_obj, *name)) {
                    subst.insert(*id, stored_value);
                    // Record the alias so chained property accesses can
                    // resolve through this load.  E.g. for `root.a.b`,
                    // after forwarding `root.a` → inner_obj, the load of
                    // `.b` needs to resolve `root.a` → inner_obj so it
                    // can look up (inner_obj, "b") in the store_map.
                    alias_map.insert(*id, stored_value);
                }
            }

            // Nodes that can mutate named properties invalidate the store map.
            // Nodes with any side effect invalidate the global map (globals
            // can change through deopts or calls).
            other => {
                if can_invalidate_named_stores(other) {
                    store_map.clear();
                    global_map.clear();
                } else if has_side_effects(other) {
                    global_map.clear();
                }
                // Keep alias_map — aliases are structural, not invalidated
                // by side effects.
            }
        }
    }

    subst
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2c — Dead allocation elimination (escape analysis lite)
// ─────────────────────────────────────────────────────────────────────────────

/// Eliminate stores to non-escaping objects.
///
/// After store-to-load forwarding eliminates property loads, an allocation may
/// only be referenced by `StoreNamedGeneric` nodes.  These stores are dead
/// because the object never escapes.  This pass replaces such stores with
/// `UndefinedConstant` placeholders so the subsequent dead-allocation pass can
/// remove the allocation itself.
fn eliminate_dead_object_stores(graph: &mut MaglevGraph) {
    // Step 1: Find all allocation NodeIds.
    let mut alloc_ids: HashSet<NodeId> = HashSet::new();
    for block in graph.blocks() {
        for (id, node) in &block.nodes {
            if matches!(
                node,
                ValueNode::CreateObjectLiteral { .. }
                    | ValueNode::CreateObjectLiteralWithProperties { .. }
                    | ValueNode::CreateEmptyObjectLiteral
                    | ValueNode::CreateArrayLiteral { .. }
                    | ValueNode::CreateEmptyArrayLiteral { .. }
                    | ValueNode::CreateShallowObjectLiteral { .. }
                    | ValueNode::CreateShallowArrayLiteral { .. }
            ) {
                alloc_ids.insert(*id);
            }
        }
    }
    if alloc_ids.is_empty() {
        return;
    }

    // Step 2: For each allocation, collect which nodes reference it and
    // determine if it "escapes" (referenced by anything other than
    // StoreNamedGeneric targeting it as the object).
    let mut escapes: HashSet<NodeId> = HashSet::new();
    let mut stores_per_alloc: HashMap<NodeId, Vec<(u32, usize)>> = HashMap::new();

    for block in graph.blocks() {
        for (idx, (_id, node)) in block.nodes.iter().enumerate() {
            match node {
                ValueNode::StoreNamedGeneric { object, .. } if alloc_ids.contains(object) => {
                    stores_per_alloc
                        .entry(*object)
                        .or_default()
                        .push((block.id, idx));
                    // The store's value operand doesn't count as an escape
                    // of the allocation (it's the value being stored, not
                    // the object).  But check if *id* itself is an alloc
                    // used elsewhere — handled by other branches.
                }
                _ => {
                    // Check if this node references any allocation in a
                    // non-store capacity (load, call arg, return, etc.).
                    visit_value_node_inputs(node, &mut |input| {
                        if alloc_ids.contains(&input) {
                            // If this node is a StoreNamedGeneric targeting
                            // this alloc, it was handled above.  Otherwise
                            // the alloc escapes.
                            escapes.insert(input);
                        }
                    });
                    // If the node IS a StoreNamedGeneric but its object is
                    // not an allocation, inputs were already visited above.
                }
            }
        }
        // Also check control nodes for escaping allocations.
        if let Some(ctrl) = &block.control {
            collect_control_node_inputs(ctrl, &mut escapes);
        }
    }

    // Step 3: For non-escaping allocations, replace their stores with
    // UndefinedConstant.
    for (alloc_id, store_locs) in &stores_per_alloc {
        if escapes.contains(alloc_id) {
            continue;
        }
        // All references to this allocation are StoreNamedGeneric — safe to
        // eliminate both the stores and (later) the allocation.
        for &(block_id, node_idx) in store_locs {
            if let Some(block) = graph.blocks_mut().iter_mut().find(|b| b.id == block_id) {
                block.nodes[node_idx].1 = ValueNode::UndefinedConstant;
            }
        }
    }
}

/// Eliminate object/array literal allocations whose results are never used.
///
/// After store-to-load forwarding, property loads from short-lived objects are
/// replaced with the stored values directly.  If the allocation's [`NodeId`]
/// is no longer referenced by any live node, the object never "escapes" and
/// the allocation is dead.  This pass replaces such allocations with cheap
/// [`ValueNode::UndefinedConstant`] placeholders (in-place, to preserve graph
/// structure for the register allocator).
fn eliminate_dead_allocations(graph: &mut MaglevGraph) {
    let live = collect_live_ids(graph);

    for block in graph.blocks_mut() {
        for (id, node) in &mut block.nodes {
            let is_dead_alloc = matches!(
                node,
                ValueNode::CreateObjectLiteral { .. }
                    | ValueNode::CreateObjectLiteralWithProperties { .. }
                    | ValueNode::CreateEmptyObjectLiteral
                    | ValueNode::CreateArrayLiteral { .. }
                    | ValueNode::CreateEmptyArrayLiteral { .. }
                    | ValueNode::CreateShallowObjectLiteral { .. }
                    | ValueNode::CreateShallowArrayLiteral { .. }
            ) && !live.contains(id);

            if is_dead_alloc {
                *node = ValueNode::UndefinedConstant;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 3 — Dead-code elimination
// ─────────────────────────────────────────────────────────────────────────────

/// Replace dead `CreateMapped/UnmappedArguments` and `CreateRestParameter`
/// nodes with cheap [`ValueNode::UndefinedConstant`] placeholders.
///
/// Unlike full DCE (which **removes** nodes), this pass **replaces** the node
/// in-place, preserving graph structure (node count, positions, register
/// allocation).  This avoids a known issue where node removal changes
/// interference patterns in the register allocator, triggering latent bugs
/// on Linux in release mode.
fn replace_dead_arguments(graph: &mut MaglevGraph) {
    let live = collect_live_ids(graph);

    for block in graph.blocks_mut() {
        for (id, node) in &mut block.nodes {
            if matches!(
                node,
                ValueNode::CreateMappedArguments
                    | ValueNode::CreateUnmappedArguments
                    | ValueNode::CreateRestParameter
            ) && !live.contains(id)
            {
                *node = ValueNode::UndefinedConstant;
            }
        }
    }
}

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
        | ValueNode::CreateEmptyArrayLiteral { .. }
        | ValueNode::CreateMappedArguments
        | ValueNode::CreateUnmappedArguments
        | ValueNode::CreateRestParameter
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
        | ValueNode::TestNullOrUndefined { value }
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

        ValueNode::PushContext { context } | ValueNode::PopContext { context } => {
            live.insert(*context);
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

        ValueNode::CreateObjectLiteralWithProperties { values, .. } => {
            for &v in values {
                live.insert(v);
            }
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
            | ValueNode::CreateObjectLiteralWithProperties { .. }
            | ValueNode::CreateArrayLiteral { .. }
            | ValueNode::CreateShallowObjectLiteral { .. }
            | ValueNode::CreateShallowArrayLiteral { .. }
            | ValueNode::CreateFunctionContext { .. }
            | ValueNode::CreateBlockContext { .. }
            | ValueNode::CreateCatchContext { .. }
            | ValueNode::CreateWithContext { .. }
            | ValueNode::PushContext { .. }
            | ValueNode::PopContext { .. }
            | ValueNode::CreateClosure { .. }
            | ValueNode::FastCreateClosure { .. }
            | ValueNode::CreateEmptyObjectLiteral
            | ValueNode::CreateEmptyArrayLiteral { .. }
            // CreateMappedArguments / CreateUnmappedArguments / CreateRestParameter
            // MUST be treated as side-effecting.  Removing them via DCE changes
            // the Maglev graph shape, which alters register allocation and
            // exposes latent regalloc bugs on Linux release builds (SIGSEGV /
            // wrong results).  Codegen routes them through the runtime
            // trampoline which creates the arguments object successfully.
            | ValueNode::CreateMappedArguments
            | ValueNode::CreateUnmappedArguments
            | ValueNode::CreateRestParameter
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

/// Return `true` for nodes that may mutate named properties of heap objects,
/// requiring the store-to-load forwarding map to be invalidated.
///
/// This is more targeted than [`has_side_effects`]: guards and checked
/// arithmetic can deopt but never mutate object properties, so they do NOT
/// invalidate the store/global maps.  Only memory-writing operations (stores,
/// calls, constructs, property deletion) require invalidation.
fn can_invalidate_named_stores(node: &ValueNode) -> bool {
    matches!(
        node,
        // Stores write to heap memory.
        ValueNode::StoreField { .. }
            | ValueNode::StoreFixedArrayElement { .. }
            | ValueNode::StoreFixedDoubleArrayElement { .. }
            | ValueNode::StoreNamedGeneric { .. }
            | ValueNode::StoreKeyedGeneric { .. }
            | ValueNode::StoreGlobal { .. }
            | ValueNode::StoreContextSlot { .. }
            | ValueNode::StoreCurrentContextSlot { .. }
            // Calls may execute arbitrary JS that mutates properties.
            | ValueNode::Call { .. }
            | ValueNode::CallKnownFunction { .. }
            | ValueNode::CallBuiltin { .. }
            | ValueNode::CallRuntime { .. }
            | ValueNode::CallWithSpread { .. }
            | ValueNode::Construct { .. }
            | ValueNode::ConstructWithSpread { .. }
            // Property deletion mutates objects.
            | ValueNode::DeleteProperty { .. }
            // For-in can invalidate caches.
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
        | ValueNode::CreateEmptyArrayLiteral { .. }
        | ValueNode::CreateMappedArguments
        | ValueNode::CreateUnmappedArguments
        | ValueNode::CreateRestParameter
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
        | ValueNode::TestNullOrUndefined { value }
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
        ValueNode::PushContext { context } | ValueNode::PopContext { context } => {
            *context = resolve(*context);
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

        ValueNode::CreateObjectLiteralWithProperties { values, .. } => {
            for v in values {
                *v = resolve(*v);
            }
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
// Pass 10 — Simple counted-loop unrolling (2×)
// ─────────────────────────────────────────────────────────────────────────────

/// Unroll simple counted for-loops by a factor of 2.
///
/// The pass targets loops with the following shape:
///
/// ```text
///   header:  Phi_i, Phi_n, …  |  cmp Phi_i < LIMIT  |  branch(body, exit)
///   body:    … new_i = Phi_i + 1 …  |  jump(header)
/// ```
///
/// When the trip count (`LIMIT − init`) is even, the body block is duplicated
/// inline: the second copy uses the first copy's outputs as inputs, halving
/// the number of back-edge comparisons.
fn unroll_simple_loops(graph: &mut MaglevGraph) {
    let loops = licm::detect_loops(graph);
    for lp in &loops {
        try_unroll_counted_loop(graph, lp, 2);
    }
}

/// Attempt to unroll a single natural loop by the given `factor`.
///
/// Returns `true` if the loop was successfully unrolled.
fn try_unroll_counted_loop(graph: &mut MaglevGraph, lp: &licm::NaturalLoop, factor: u32) -> bool {
    if factor < 2 {
        return false;
    }

    let header_idx = lp.header;
    let header = match graph.block(header_idx) {
        Some(b) => b,
        None => return false,
    };

    // ── 1. The header must end with a Branch on an Int32 comparison ──────
    let (condition_id, body_block_idx, _exit_block_idx) = match &header.control {
        Some(ControlNode::Branch {
            condition,
            if_true,
            if_false,
        }) => (*condition, *if_true, *if_false),
        _ => return false,
    };

    // The body block must be inside the loop.
    if !lp.body.contains(&body_block_idx) {
        return false;
    }

    // ── 2. Simple loop: body is a single block jumping back to header ────
    if lp.body.len() != 2 || !lp.body.contains(&header_idx) {
        return false;
    }

    let body = match graph.block(body_block_idx) {
        Some(b) => b,
        None => return false,
    };

    // Body must jump back to the header (back-edge).
    match &body.control {
        Some(ControlNode::Jump { target }) if *target == header_idx => {}
        _ => return false,
    }

    // ── 3. Find the comparison and its operands ─────────────────────────
    let (cmp_left, cmp_right) = {
        let mut found = None;
        for (nid, node) in &header.nodes {
            if *nid == condition_id {
                found = Some(node.clone());
                break;
            }
        }
        match found {
            Some(ValueNode::Int32LessThan { left, right }) => (left, right),
            _ => return false,
        }
    };

    // ── 4. Identify Phi nodes, induction variable, and constants ─────────
    let back_edge_pred_pos = {
        let h = graph.block(header_idx).unwrap();
        match h.predecessors.iter().position(|&p| p == body_block_idx) {
            Some(pos) => pos,
            None => return false,
        }
    };

    let preheader_pred_pos = {
        let h = graph.block(header_idx).unwrap();
        match h.predecessors.iter().position(|&p| p == lp.preheader) {
            Some(pos) => pos,
            None => return false,
        }
    };

    // Check that cmp_left is a Phi in the header (the induction variable).
    let induction_phi_inputs = {
        let h = graph.block(header_idx).unwrap();
        let mut found = None;
        for (nid, node) in &h.nodes {
            if *nid == cmp_left {
                if let ValueNode::Phi { inputs } = node {
                    found = Some(inputs.clone());
                }
                break;
            }
        }
        match found {
            Some(inputs) => inputs,
            None => return false,
        }
    };

    // Get the initial value and the limit as constants.
    let init_id = induction_phi_inputs[preheader_pred_pos];
    let limit_id = cmp_right;

    let init_value = find_i32_constant(graph, init_id);
    let limit_value = find_i32_constant(graph, limit_id);

    let (init_val, limit_val) = match (init_value, limit_value) {
        (Some(i), Some(l)) => (i, l),
        _ => return false,
    };

    // ── 5. Find the induction variable increment in the body ────────────
    let back_edge_input = induction_phi_inputs[back_edge_pred_pos];
    let step = find_increment_step(graph, back_edge_input, cmp_left);
    let step_val = match step {
        Some(s) if s > 0 => s,
        _ => return false,
    };

    // ── 6. Check trip count divisibility ─────────────────────────────────
    let trip_count = (limit_val - init_val) as i64;
    if trip_count <= 0 {
        return false;
    }
    let effective_step = (step_val as i64) * (factor as i64);
    if trip_count % effective_step != 0 {
        return false;
    }

    // Don't unroll very small loops (< 2*factor iterations).
    if trip_count < effective_step * 2 {
        return false;
    }

    // ── 7. Collect Phi → back-edge-input mappings ────────────────────────
    let header_phis: Vec<(NodeId, Vec<NodeId>)> = {
        let h = graph.block(header_idx).unwrap();
        h.nodes
            .iter()
            .filter_map(|(nid, node)| {
                if let ValueNode::Phi { inputs } = node {
                    Some((*nid, inputs.clone()))
                } else {
                    None
                }
            })
            .collect()
    };

    let phi_to_back_input: HashMap<NodeId, NodeId> = header_phis
        .iter()
        .filter_map(|(phi_id, inputs)| {
            inputs
                .get(back_edge_pred_pos)
                .map(|&back_input| (*phi_id, back_input))
        })
        .collect();

    // ── 8. Clone the body nodes for additional unrolled copies ───────────
    let body_nodes: Vec<(NodeId, ValueNode)> = graph.block(body_block_idx).unwrap().nodes.clone();

    let mut prev_phi_to_output = phi_to_back_input.clone();

    for _copy in 1..factor {
        let mut subst: HashMap<NodeId, NodeId> = HashMap::new();
        for (phi_id, output_id) in &prev_phi_to_output {
            subst.insert(*phi_id, *output_id);
        }

        let mut old_to_new: HashMap<NodeId, NodeId> = HashMap::new();
        for (old_id, _node) in &body_nodes {
            let new_id = graph.alloc_node_id();
            old_to_new.insert(*old_id, new_id);
        }

        let resolve = |id: NodeId| -> NodeId {
            if let Some(&new_id) = old_to_new.get(&id) {
                new_id
            } else if let Some(&output_id) = subst.get(&id) {
                output_id
            } else {
                id
            }
        };

        let mut cloned_nodes: Vec<(NodeId, ValueNode)> = Vec::new();
        for (old_id, node) in &body_nodes {
            let new_id = old_to_new[old_id];
            let mut cloned = node.clone();
            apply_subst_to_value_node(&mut cloned, &resolve);
            cloned_nodes.push((new_id, cloned));
        }

        let body_block = graph.block_mut(body_block_idx).unwrap();
        for (id, node) in &cloned_nodes {
            body_block.push_with_id(*id, node.clone());
        }

        let mut new_phi_to_output: HashMap<NodeId, NodeId> = HashMap::new();
        for (phi_id, back_input) in &phi_to_back_input {
            if let Some(&new_id) = old_to_new.get(back_input) {
                new_phi_to_output.insert(*phi_id, new_id);
            } else {
                new_phi_to_output.insert(*phi_id, *back_input);
            }
        }
        prev_phi_to_output = new_phi_to_output;
    }

    // ── 9. Update header Phi back-edge inputs to last copy's outputs ─────
    let header_block = graph.block_mut(header_idx).unwrap();
    for (nid, node) in &mut header_block.nodes {
        if let ValueNode::Phi { inputs } = node
            && let Some(&new_output) = prev_phi_to_output.get(nid)
            && let Some(back_input) = inputs.get_mut(back_edge_pred_pos)
        {
            *back_input = new_output;
        }
    }

    true
}

/// Find the compile-time i32 constant value for a [`NodeId`], if any.
fn find_i32_constant(graph: &MaglevGraph, id: NodeId) -> Option<i32> {
    for block in graph.blocks() {
        for (nid, node) in &block.nodes {
            if *nid == id {
                return match node {
                    ValueNode::Int32Constant { value } | ValueNode::SmiConstant { value } => {
                        Some(*value)
                    }
                    _ => None,
                };
            }
        }
    }
    None
}

/// Determine the increment step of an induction variable.
///
/// Returns `Some(step)` if `result_id` is computed as `base_id + step`.
fn find_increment_step(graph: &MaglevGraph, result_id: NodeId, base_id: NodeId) -> Option<i32> {
    let node = graph.node(result_id)?;
    match node {
        ValueNode::Int32Increment { value }
        | ValueNode::CheckedSmiIncrement { value }
        | ValueNode::GenericIncrement { value, .. } => {
            if *value == base_id {
                Some(1)
            } else {
                None
            }
        }
        ValueNode::Int32Add { left, right }
        | ValueNode::CheckedSmiAdd { left, right }
        | ValueNode::GenericAdd { left, right, .. } => {
            if *left == base_id {
                find_i32_constant(graph, *right)
            } else if *right == base_id {
                find_i32_constant(graph, *left)
            } else {
                None
            }
        }
        _ => None,
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

    // ── Trivial Phi elimination ──────────────────────────────────────────────

    #[test]
    fn test_trivial_phi_self_referencing_eliminated() {
        // Simulate a loop-header identity Phi: Phi(X, self) → should become X.
        //
        //   Block 0 (preheader):
        //     NodeId(0) = CreateEmptyObjectLiteral   ← the "root" object
        //     → Jump to block 1
        //
        //   Block 1 (loop header):
        //     NodeId(1) = Phi([NodeId(0), NodeId(1)])  ← trivial!
        //     NodeId(2) = LoadNamedGeneric(NodeId(1), "a")
        //     → Return NodeId(2)
        //
        // After trivial-Phi elimination, NodeId(2) should reference NodeId(0)
        // directly, and the Phi should be dead.
        let mut graph = MaglevGraph::new(0);

        // Block 0: preheader
        let mut b0 = BasicBlock::new(0);
        let root = graph.alloc_node_id(); // NodeId(0)
        b0.push_with_id(root, ValueNode::CreateEmptyObjectLiteral);
        b0.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b0);

        // Block 1: loop header with identity Phi
        let mut b1 = BasicBlock::new(1);
        b1.predecessors = vec![0, 1]; // entry + back-edge
        b1.is_loop_header = true;
        let phi_id = graph.alloc_node_id(); // NodeId(1)
        b1.push_with_id(
            phi_id,
            ValueNode::Phi {
                inputs: vec![root, phi_id],
            },
        );
        let load = graph.alloc_node_id(); // NodeId(2)
        b1.push_with_id(
            load,
            ValueNode::LoadNamedGeneric {
                object: phi_id,
                name: 0,
                feedback_slot: 0,
            },
        );
        b1.set_control(ControlNode::Return { value: load });
        graph.add_block(b1);

        // Run the full optimizer (which includes trivial Phi elimination).
        optimize(&mut graph);

        // The LoadNamedGeneric should now reference the preheader's root
        // (NodeId(0)) directly, not the Phi (NodeId(1)).
        let load_node = graph.blocks()[1]
            .nodes
            .iter()
            .find(|(_, n)| matches!(n, ValueNode::LoadNamedGeneric { .. }));
        if let Some((_, ValueNode::LoadNamedGeneric { object, .. })) = load_node {
            assert_eq!(
                *object, root,
                "LoadNamedGeneric should reference preheader root after Phi elimination"
            );
        } else {
            // The load might have been moved to block 0 by LICM (even better!)
            let load_in_b0 = graph.blocks()[0]
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::LoadNamedGeneric { .. }));
            assert!(
                load_in_b0,
                "LoadNamedGeneric should be hoisted to preheader by LICM"
            );
        }
    }

    #[test]
    fn test_trivial_phi_non_trivial_preserved() {
        // A real Phi with two distinct non-self inputs should NOT be eliminated.
        //
        //   Block 0: NodeId(0) = SmiConstant(0)  → Jump to block 1
        //   Block 1: NodeId(2) = Phi([NodeId(0), NodeId(1)])
        //            NodeId(1) = GenericIncrement(NodeId(2))
        //            → Return NodeId(2)
        let mut graph = MaglevGraph::new(0);

        let mut b0 = BasicBlock::new(0);
        let zero = graph.alloc_node_id();
        b0.push_with_id(zero, ValueNode::SmiConstant { value: 0 });
        b0.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b0);

        let mut b1 = BasicBlock::new(1);
        b1.predecessors = vec![0, 1];
        b1.is_loop_header = true;
        let phi_id = graph.alloc_node_id();
        let inc_id = graph.alloc_node_id();
        b1.push_with_id(
            phi_id,
            ValueNode::Phi {
                inputs: vec![zero, inc_id],
            },
        );
        b1.push_with_id(
            inc_id,
            ValueNode::GenericIncrement {
                value: phi_id,
                feedback_slot: 0,
            },
        );
        b1.set_control(ControlNode::Return { value: phi_id });
        graph.add_block(b1);

        optimize(&mut graph);

        // The Phi should still exist (it's not trivial — two distinct inputs).
        let phi_exists = graph
            .blocks()
            .iter()
            .flat_map(|b| b.nodes.iter())
            .any(|(_, n)| matches!(n, ValueNode::Phi { .. }));
        assert!(phi_exists, "Non-trivial Phi should be preserved");
    }

    // ── Store-to-load forwarding ──────────────────────────────────────────────

    #[test]
    fn test_store_to_load_forwarding_basic() {
        // StoreNamedGeneric(obj, "x", val) followed by LoadNamedGeneric(obj, "x")
        // should forward val, eliminating the load.
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let obj = block.push_value(ValueNode::Parameter { index: 0 });
        let val = block.push_value(ValueNode::SmiConstant { value: 42 });
        let _store = block.push_value(ValueNode::StoreNamedGeneric {
            object: obj,
            name: 5,
            value: val,
            feedback_slot: 0,
        });
        let load = block.push_value(ValueNode::LoadNamedGeneric {
            object: obj,
            name: 5,
            feedback_slot: 1,
        });
        block.set_control(ControlNode::Return { value: load });
        graph.add_block(block);

        eliminate_store_to_load(&mut graph);

        // The return should now reference val directly, not the load.
        match &graph.blocks()[0].control {
            Some(ControlNode::Return { value }) => {
                assert_eq!(*value, val, "Load should be forwarded to stored value");
            }
            other => panic!("Expected Return, got {:?}", other),
        }
    }

    #[test]
    fn test_store_to_load_forwarding_fused_create() {
        // CreateObjectLiteralWithProperties seeds the map so subsequent loads
        // are forwarded.
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let val_x = block.push_value(ValueNode::SmiConstant { value: 10 });
        let val_y = block.push_value(ValueNode::SmiConstant { value: 20 });
        let obj = block.push_value(ValueNode::CreateObjectLiteralWithProperties {
            feedback_slot: 0,
            flags: 0,
            names: vec![1, 2],
            values: vec![val_x, val_y],
        });
        let load_x = block.push_value(ValueNode::LoadNamedGeneric {
            object: obj,
            name: 1,
            feedback_slot: 1,
        });
        let load_y = block.push_value(ValueNode::LoadNamedGeneric {
            object: obj,
            name: 2,
            feedback_slot: 2,
        });
        let sum = block.push_value(ValueNode::Int32Add {
            left: load_x,
            right: load_y,
        });
        block.set_control(ControlNode::Return { value: sum });
        graph.add_block(block);

        eliminate_store_to_load(&mut graph);

        // The Int32Add should now reference val_x and val_y directly.
        let add_node = graph.blocks()[0]
            .nodes
            .iter()
            .find(|(_, n)| matches!(n, ValueNode::Int32Add { .. }));
        match add_node {
            Some((_, ValueNode::Int32Add { left, right })) => {
                assert_eq!(*left, val_x, "load_x should be forwarded to val_x");
                assert_eq!(*right, val_y, "load_y should be forwarded to val_y");
            }
            other => panic!("Expected Int32Add, got {:?}", other),
        }
    }

    #[test]
    fn test_store_to_load_invalidated_by_call() {
        // A Call between store and load should prevent forwarding.
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let obj = block.push_value(ValueNode::Parameter { index: 0 });
        let val = block.push_value(ValueNode::SmiConstant { value: 42 });
        let _store = block.push_value(ValueNode::StoreNamedGeneric {
            object: obj,
            name: 5,
            value: val,
            feedback_slot: 0,
        });
        // A call could mutate obj's properties.
        let _call = block.push_value(ValueNode::Call {
            callee: obj,
            args: vec![],
            receiver: obj,
            feedback_slot: 0,
        });
        let load = block.push_value(ValueNode::LoadNamedGeneric {
            object: obj,
            name: 5,
            feedback_slot: 1,
        });
        block.set_control(ControlNode::Return { value: load });
        graph.add_block(block);

        eliminate_store_to_load(&mut graph);

        // The return should still reference the load (not forwarded).
        match &graph.blocks()[0].control {
            Some(ControlNode::Return { value }) => {
                assert_eq!(*value, load, "Load should NOT be forwarded past a call");
            }
            other => panic!("Expected Return, got {:?}", other),
        }
    }

    #[test]
    fn test_dead_allocation_elimination_after_forwarding() {
        // After store-to-load forwarding eliminates all loads from an object,
        // the allocation itself becomes dead and should be replaced.
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let val_x = block.push_value(ValueNode::SmiConstant { value: 10 });
        let val_y = block.push_value(ValueNode::SmiConstant { value: 20 });
        let obj = block.push_value(ValueNode::CreateObjectLiteralWithProperties {
            feedback_slot: 0,
            flags: 0,
            names: vec![1, 2],
            values: vec![val_x, val_y],
        });
        let load_x = block.push_value(ValueNode::LoadNamedGeneric {
            object: obj,
            name: 1,
            feedback_slot: 1,
        });
        let load_y = block.push_value(ValueNode::LoadNamedGeneric {
            object: obj,
            name: 2,
            feedback_slot: 2,
        });
        let sum = block.push_value(ValueNode::Int32Add {
            left: load_x,
            right: load_y,
        });
        block.set_control(ControlNode::Return { value: sum });
        graph.add_block(block);

        // Forward loads first.
        eliminate_store_to_load(&mut graph);
        // Now the allocation should be dead (only loads referenced it).
        eliminate_dead_allocations(&mut graph);

        // The CreateObjectLiteralWithProperties should be replaced with
        // UndefinedConstant since nothing references it anymore.
        let obj_node = graph.blocks()[0].nodes.iter().find(|(id, _)| *id == obj);
        match obj_node {
            Some((_, ValueNode::UndefinedConstant)) => {} // expected
            other => panic!(
                "Expected dead allocation to be replaced with UndefinedConstant, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_live_allocation_preserved() {
        // If the object escapes (e.g., returned), the allocation must stay.
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let val = block.push_value(ValueNode::SmiConstant { value: 42 });
        let obj = block.push_value(ValueNode::CreateObjectLiteralWithProperties {
            feedback_slot: 0,
            flags: 0,
            names: vec![1],
            values: vec![val],
        });
        // Object escapes via return.
        block.set_control(ControlNode::Return { value: obj });
        graph.add_block(block);

        eliminate_dead_allocations(&mut graph);

        // Allocation should NOT be replaced.
        let obj_node = graph.blocks()[0].nodes.iter().find(|(id, _)| *id == obj);
        assert!(
            matches!(
                obj_node,
                Some((_, ValueNode::CreateObjectLiteralWithProperties { .. }))
            ),
            "Live allocation should be preserved"
        );
    }

    #[test]
    fn test_store_to_load_cross_block_forwarding() {
        // Store in block 0 (entry), load in block 1 (single predecessor).
        // Cross-block forwarding should forward the load to the stored value.
        //
        //   Block 0:
        //     obj = Parameter(0)
        //     val = SmiConstant(42)
        //     StoreNamedGeneric(obj, "x", val)
        //     → Jump to block 1
        //
        //   Block 1 (predecessors: [0]):
        //     load = LoadNamedGeneric(obj, "x")
        //     → Return load
        //
        // After forwarding, the return should reference val directly.
        let mut graph = MaglevGraph::new(0);

        let mut b0 = BasicBlock::new(0);
        let obj = graph.alloc_node_id();
        b0.push_with_id(obj, ValueNode::Parameter { index: 0 });
        let val = graph.alloc_node_id();
        b0.push_with_id(val, ValueNode::SmiConstant { value: 42 });
        let store = graph.alloc_node_id();
        b0.push_with_id(
            store,
            ValueNode::StoreNamedGeneric {
                object: obj,
                name: 5,
                value: val,
                feedback_slot: 0,
            },
        );
        b0.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b0);

        let mut b1 = BasicBlock::new(1);
        b1.predecessors = vec![0];
        let load = graph.alloc_node_id();
        b1.push_with_id(
            load,
            ValueNode::LoadNamedGeneric {
                object: obj,
                name: 5,
                feedback_slot: 1,
            },
        );
        b1.set_control(ControlNode::Return { value: load });
        graph.add_block(b1);

        eliminate_store_to_load(&mut graph);

        // The return in block 1 should now reference val directly.
        match &graph.blocks()[1].control {
            Some(ControlNode::Return { value }) => {
                assert_eq!(
                    *value, val,
                    "Cross-block load should be forwarded to stored value"
                );
            }
            other => panic!("Expected Return, got {:?}", other),
        }
    }

    #[test]
    fn test_store_to_load_cross_block_merge_conservative() {
        // Stores in two predecessors should NOT be forwarded at a merge point.
        //
        //   Block 0: Store(obj, "x", val_a) → Jump 2
        //   Block 1: Store(obj, "x", val_b) → Jump 2
        //   Block 2 (predecessors: [0, 1]): Load(obj, "x") → Return
        //
        // At the merge point we cannot know which predecessor ran, so the
        // load must NOT be forwarded.
        let mut graph = MaglevGraph::new(0);

        let mut b0 = BasicBlock::new(0);
        let obj = graph.alloc_node_id();
        b0.push_with_id(obj, ValueNode::Parameter { index: 0 });
        let val_a = graph.alloc_node_id();
        b0.push_with_id(val_a, ValueNode::SmiConstant { value: 1 });
        let store_a = graph.alloc_node_id();
        b0.push_with_id(
            store_a,
            ValueNode::StoreNamedGeneric {
                object: obj,
                name: 5,
                value: val_a,
                feedback_slot: 0,
            },
        );
        b0.set_control(ControlNode::Jump { target: 2 });
        graph.add_block(b0);

        let mut b1 = BasicBlock::new(1);
        let val_b = graph.alloc_node_id();
        b1.push_with_id(val_b, ValueNode::SmiConstant { value: 2 });
        let store_b = graph.alloc_node_id();
        b1.push_with_id(
            store_b,
            ValueNode::StoreNamedGeneric {
                object: obj,
                name: 5,
                value: val_b,
                feedback_slot: 0,
            },
        );
        b1.set_control(ControlNode::Jump { target: 2 });
        graph.add_block(b1);

        let mut b2 = BasicBlock::new(2);
        b2.predecessors = vec![0, 1];
        let load = graph.alloc_node_id();
        b2.push_with_id(
            load,
            ValueNode::LoadNamedGeneric {
                object: obj,
                name: 5,
                feedback_slot: 1,
            },
        );
        b2.set_control(ControlNode::Return { value: load });
        graph.add_block(b2);

        eliminate_store_to_load(&mut graph);

        // The return in block 2 should still reference the load (not forwarded).
        match &graph.blocks()[2].control {
            Some(ControlNode::Return { value }) => {
                assert_eq!(*value, load, "Load at merge point should NOT be forwarded");
            }
            other => panic!("Expected Return, got {:?}", other),
        }
    }

    // ── Loop unrolling ────────────────────────────────────────────────────────

    #[test]
    fn test_loop_unroll_even_trip_count() {
        // for (var i = 0; i < 10; i++) { n = n + i; }
        // Use graph-global NodeIds to avoid block-local ID collisions.
        let mut graph = MaglevGraph::new(0);

        // block 0 (preheader)
        graph.add_block(BasicBlock::new(0));
        let init_i = graph
            .add_value_node(0, ValueNode::Int32Constant { value: 0 })
            .unwrap();
        let init_n = graph
            .add_value_node(0, ValueNode::Int32Constant { value: 0 })
            .unwrap();
        let limit = graph
            .add_value_node(0, ValueNode::Int32Constant { value: 10 })
            .unwrap();
        graph
            .block_mut(0)
            .unwrap()
            .set_control(ControlNode::Jump { target: 1 });

        // block 1 (header) — Phi back-edge inputs patched below
        let mut b1 = BasicBlock::new(1);
        b1.is_loop_header = true;
        b1.predecessors = vec![0, 2];
        graph.add_block(b1);
        let phi_i = graph
            .add_value_node(
                1,
                ValueNode::Phi {
                    inputs: vec![init_i, NodeId(999)],
                },
            )
            .unwrap();
        let phi_n = graph
            .add_value_node(
                1,
                ValueNode::Phi {
                    inputs: vec![init_n, NodeId(999)],
                },
            )
            .unwrap();
        let cmp = graph
            .add_value_node(
                1,
                ValueNode::Int32LessThan {
                    left: phi_i,
                    right: limit,
                },
            )
            .unwrap();
        graph
            .block_mut(1)
            .unwrap()
            .set_control(ControlNode::Branch {
                condition: cmp,
                if_true: 2,
                if_false: 3,
            });

        // block 2 (body)
        let mut b2 = BasicBlock::new(2);
        b2.predecessors = vec![1];
        graph.add_block(b2);
        let new_n = graph
            .add_value_node(
                2,
                ValueNode::Int32Add {
                    left: phi_n,
                    right: phi_i,
                },
            )
            .unwrap();
        let new_i = graph
            .add_value_node(2, ValueNode::Int32Increment { value: phi_i })
            .unwrap();
        graph
            .block_mut(2)
            .unwrap()
            .set_control(ControlNode::Jump { target: 1 });

        // block 3 (exit)
        let mut b3 = BasicBlock::new(3);
        b3.predecessors = vec![1];
        graph.add_block(b3);
        graph
            .block_mut(3)
            .unwrap()
            .set_control(ControlNode::Return { value: phi_n });

        // Patch Phi back-edge inputs.
        if let ValueNode::Phi { inputs } = &mut graph.block_mut(1).unwrap().nodes[0].1 {
            inputs[1] = new_i;
        }
        if let ValueNode::Phi { inputs } = &mut graph.block_mut(1).unwrap().nodes[1].1 {
            inputs[1] = new_n;
        }

        let body_nodes_before = graph.block(2).unwrap().nodes.len();
        unroll_simple_loops(&mut graph);
        let body_nodes_after = graph.block(2).unwrap().nodes.len();
        assert_eq!(body_nodes_after, body_nodes_before * 2);

        // Phi back-edge inputs should point to cloned nodes.
        if let ValueNode::Phi { inputs } = &graph.block(1).unwrap().nodes[0].1 {
            assert_ne!(inputs[1], new_i);
        }
        if let ValueNode::Phi { inputs } = &graph.block(1).unwrap().nodes[1].1 {
            assert_ne!(inputs[1], new_n);
        }
    }

    #[test]
    fn test_loop_unroll_skips_odd_trip() {
        // Trip count 7 — not divisible by 2, should skip unrolling.
        let mut graph = MaglevGraph::new(0);

        graph.add_block(BasicBlock::new(0));
        let init_i = graph
            .add_value_node(0, ValueNode::Int32Constant { value: 0 })
            .unwrap();
        let init_n = graph
            .add_value_node(0, ValueNode::Int32Constant { value: 0 })
            .unwrap();
        let limit = graph
            .add_value_node(0, ValueNode::Int32Constant { value: 7 })
            .unwrap();
        graph
            .block_mut(0)
            .unwrap()
            .set_control(ControlNode::Jump { target: 1 });

        let mut b1 = BasicBlock::new(1);
        b1.is_loop_header = true;
        b1.predecessors = vec![0, 2];
        graph.add_block(b1);
        let phi_i = graph
            .add_value_node(
                1,
                ValueNode::Phi {
                    inputs: vec![init_i, NodeId(999)],
                },
            )
            .unwrap();
        let phi_n = graph
            .add_value_node(
                1,
                ValueNode::Phi {
                    inputs: vec![init_n, NodeId(999)],
                },
            )
            .unwrap();
        let cmp = graph
            .add_value_node(
                1,
                ValueNode::Int32LessThan {
                    left: phi_i,
                    right: limit,
                },
            )
            .unwrap();
        graph
            .block_mut(1)
            .unwrap()
            .set_control(ControlNode::Branch {
                condition: cmp,
                if_true: 2,
                if_false: 3,
            });

        let mut b2 = BasicBlock::new(2);
        b2.predecessors = vec![1];
        graph.add_block(b2);
        let new_n = graph
            .add_value_node(
                2,
                ValueNode::Int32Add {
                    left: phi_n,
                    right: phi_i,
                },
            )
            .unwrap();
        let new_i = graph
            .add_value_node(2, ValueNode::Int32Increment { value: phi_i })
            .unwrap();
        graph
            .block_mut(2)
            .unwrap()
            .set_control(ControlNode::Jump { target: 1 });

        let mut b3 = BasicBlock::new(3);
        b3.predecessors = vec![1];
        graph.add_block(b3);
        graph
            .block_mut(3)
            .unwrap()
            .set_control(ControlNode::Return { value: phi_n });

        if let ValueNode::Phi { inputs } = &mut graph.block_mut(1).unwrap().nodes[0].1 {
            inputs[1] = new_i;
        }
        if let ValueNode::Phi { inputs } = &mut graph.block_mut(1).unwrap().nodes[1].1 {
            inputs[1] = new_n;
        }

        let body_before = graph.block(2).unwrap().nodes.len();
        unroll_simple_loops(&mut graph);
        assert_eq!(graph.block(2).unwrap().nodes.len(), body_before);
    }
}
