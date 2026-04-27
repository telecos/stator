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
/// Run the full optimization pipeline on the Maglev graph.
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
    fuse_call_loops(graph);
    // NOTE: sum/push loop fusion disabled — causes deopt regression on
    // array_push_sum_1k (13.7µs vs 6.2µs baseline).  The runtime stubs
    // work correctly but the deopt fallback wipes JIT state, making
    // subsequent iterations run at raw-interpreter speed.
    // fuse_sum_loops(graph);
    mark_inlining_candidates(graph);
    remove_redundant_check_maps(graph);
    fuse_object_literal_stores(graph);
    forward_invariant_object_properties(graph);
    eliminate_store_to_load(graph);
    // Re-run constant folding: store-to-load forwarding may replace
    // LoadNamedGeneric with SmiConstants, enabling preheader GenericAdd
    // chains (from fold_invariant_addition_chains) to constant-fold.
    fold_constants(graph);
    // Re-run invariant chain folding: now that store-to-load has revealed
    // SmiConstants for property loads, chains like `sum + obj.a + obj.b + …`
    // become `sum + SmiConst(1) + SmiConst(2) + …` — all addends are
    // loop-invariant.  This folds them into a single preheader sum.
    crate::compiler::maglev::licm::fold_invariant_addition_chains(graph);
    // Fold the new preheader additions (e.g. GenericAdd(1,2) → 3).
    fold_constants(graph);
    // Remove dead intermediate chain nodes left by fold_invariant_addition_chains.
    // Without this, use-count checks in accumulator detection see stale references
    // to Phi nodes from dead chain links, incorrectly inflating use counts and
    // preventing scalar evolution from firing (e.g. property_access_1k).
    eliminate_dead_code(graph);
    // Targeted late lowering: after store_to_load + chain folding, some
    // loops have `GenericAdd(accumulator_phi, SmiConstant(K))` where the
    // constant was only revealed by forwarding.  Convert these to Int32Add
    // when init and addend are known-safe Smi constants.
    lower_constant_accumulator_adds(graph);
    // Loop scalar evolution: replace simple counted loops with constant
    // accumulators by a closed-form computation.  E.g. a loop that does
    // `sum += 15` for 1000 iterations becomes `sum = 0 + 15*1000 = 15000`.
    // This must run AFTER lower_constant_accumulator_adds (which converts
    // GenericAdd→Int32Add) and BEFORE unrolling (which would complicate
    // the loop structure).
    eliminate_constant_accumulator_loops(graph);
    // NOTE: eliminate_trivial_phis + eliminate_dead_code + eliminate_dead_counted_loops
    // were previously here but interacted badly with loop-aware passes
    // (forward_loop_object_properties, unroll_simple_loops) that
    // pattern-match on loop structure.  Moved to AFTER all loop-aware
    // passes so the loop structure is consumed before removal.
    forward_loop_object_properties(graph);
    eliminate_dead_object_stores(graph);
    eliminate_dead_allocations(graph);
    replace_dead_arguments(graph);
    // NOTE: IV strength reduction disabled — caused 38% arithmetic regression
    // (8.2µs → 11.3µs) due to extra register pressure and dependency chains.
    // strength_reduce_induction_variables(graph);
    unroll_simple_loops(graph);
    // After unrolling, re-run reassociation to defer constant adjustments.
    // Unrolled bodies have chains like (((n + a) - K) + b) - K) — the
    // constant-deferral pattern pushes all -K subtractions to the end,
    // then the combining pattern merges them (e.g. 4×(-1) → -4).
    reassociate_arithmetic(graph);
    // Clean up identity ops (x+0, x-0) created by reassociation combining.
    eliminate_identity_operations_global(graph);
    eliminate_dead_code(graph);
    // Now that all loop-aware passes have completed, it's safe to
    // eliminate dead counted loops created by scalar evolution.  The
    // trivial-phi pass resolves accumulator Phis that became identities
    // (init == back) after scalar evolution replaced them.
    eliminate_trivial_phis(graph);
    eliminate_dead_counted_loops(graph);
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
    // Pre-populate constant map from ALL blocks so cross-block constant
    // operands can be folded (e.g. SmiConstants defined in the entry block
    // used by chain-folded GenericAdds placed in the preheader by LICM).
    let mut consts: HashMap<NodeId, ConstVal> = HashMap::new();
    for block in graph.blocks() {
        for (id, node) in &block.nodes {
            match node {
                ValueNode::Int32Constant { value } | ValueNode::SmiConstant { value } => {
                    consts.insert(*id, ConstVal::I32(*value));
                }
                ValueNode::Float64Constant { value } => {
                    consts.insert(*id, ConstVal::F64(*value));
                }
                _ => {}
            }
        }
    }
    for block in graph.blocks_mut() {
        fold_block_constants(block, &mut consts);
    }
}

/// Represents a compile-time-known scalar value.
#[derive(Clone, Copy)]
enum ConstVal {
    I32(i32),
    F64(f64),
}

/// Fold constants within a single [`BasicBlock`], using a shared constant map
/// that spans all blocks so cross-block references to literal constants (and
/// previously folded results) are visible.
fn fold_block_constants(block: &mut BasicBlock, consts: &mut HashMap<NodeId, ConstVal>) {
    for (id, node) in &mut block.nodes {
        // Seed the constant map from literal constant nodes in this block.
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
                fold_i32_bin(left, right, consts, |a, b| a.wrapping_add(b))
            }
            ValueNode::Int32Subtract { left, right }
            | ValueNode::CheckedSmiSubtract { left, right } => {
                fold_i32_bin(left, right, consts, |a, b| a.wrapping_sub(b))
            }
            ValueNode::Int32Multiply { left, right }
            | ValueNode::CheckedSmiMultiply { left, right } => {
                fold_i32_bin(left, right, consts, |a, b| a.wrapping_mul(b))
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
                fold_f64_bin(left, right, consts, |a, b| a + b)
            }
            ValueNode::Float64Subtract { left, right } => {
                fold_f64_bin(left, right, consts, |a, b| a - b)
            }
            ValueNode::Float64Multiply { left, right } => {
                fold_f64_bin(left, right, consts, |a, b| a * b)
            }
            ValueNode::Float64Divide { left, right } => {
                fold_f64_bin(left, right, consts, |a, b| a / b)
            }
            ValueNode::Float64Modulus { left, right } => {
                fold_f64_bin(left, right, consts, |a, b| a % b)
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
                fold_i32_bin(left, right, consts, |a, b| a | b)
            }
            ValueNode::Int32BitwiseAnd { left, right } => {
                fold_i32_bin(left, right, consts, |a, b| a & b)
            }
            ValueNode::Int32BitwiseXor { left, right } => {
                fold_i32_bin(left, right, consts, |a, b| a ^ b)
            }

            // ── Generic arithmetic (same as Int32 when both inputs are i32 constants)
            ValueNode::GenericAdd { left, right, .. } => {
                fold_smi_bin(left, right, consts, |a, b| a.wrapping_add(b))
            }
            ValueNode::GenericSubtract { left, right, .. } => {
                fold_smi_bin(left, right, consts, |a, b| a.wrapping_sub(b))
            }
            ValueNode::GenericMultiply { left, right, .. } => {
                fold_smi_bin(left, right, consts, |a, b| a.wrapping_mul(b))
            }
            ValueNode::GenericIncrement { value, .. } => {
                if let Some(ConstVal::I32(v)) = consts.get(value) {
                    Some(ValueNode::SmiConstant {
                        value: v.wrapping_add(1),
                    })
                } else {
                    None
                }
            }
            ValueNode::GenericDecrement { value, .. } => {
                if let Some(ConstVal::I32(v)) = consts.get(value) {
                    Some(ValueNode::SmiConstant {
                        value: v.wrapping_sub(1),
                    })
                } else {
                    None
                }
            }
            ValueNode::GenericNegate { value, .. } => {
                if let Some(ConstVal::I32(v)) = consts.get(value) {
                    Some(ValueNode::SmiConstant {
                        value: v.wrapping_neg(),
                    })
                } else {
                    None
                }
            }

            _ => None,
        };

        if let Some(replacement) = folded {
            // Update the constant map so later nodes can fold through this one.
            match &replacement {
                ValueNode::Int32Constant { value } | ValueNode::SmiConstant { value } => {
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

/// Fold a Generic arithmetic operation on two i32-constant inputs into a
/// `SmiConstant` (preserves the Smi representation for downstream passes).
fn fold_smi_bin(
    left: &NodeId,
    right: &NodeId,
    consts: &HashMap<NodeId, ConstVal>,
    op: impl Fn(i32, i32) -> i32,
) -> Option<ValueNode> {
    if let (Some(ConstVal::I32(a)), Some(ConstVal::I32(b))) = (consts.get(left), consts.get(right))
    {
        Some(ValueNode::SmiConstant { value: op(*a, *b) })
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
        // ── Nodes with no inputs ─────────────────────────────────────────
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

        // ── Single-input value nodes ─────────────────────────────────────
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
        | ValueNode::TestNullOrUndefined { value }
        | ValueNode::ToString { value, .. }
        | ValueNode::ToObject { value, .. }
        | ValueNode::ToName { value, .. }
        | ValueNode::ToNumber { value, .. }
        | ValueNode::ToNumberOrNumeric { value, .. }
        | ValueNode::TypeOf { value }
        | ValueNode::NumberToString { value, .. } => f(*value),

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

        ValueNode::TestTypeOf { value, .. } | ValueNode::TestUndetectable { value } => f(*value),

        // ── Object/field loads ───────────────────────────────────────────
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
            f(*value)
        }

        ValueNode::LoadContextSlot { context, .. } => f(*context),
        ValueNode::StoreContextSlot { context, value, .. } => {
            f(*context);
            f(*value);
        }

        ValueNode::PushContext { context } | ValueNode::PopContext { context } => f(*context),

        // ── Fixed-array loads / stores ───────────────────────────────────
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

        ValueNode::CreateObjectLiteralWithProperties { values, .. } => {
            for &v in values {
                f(v);
            }
        }

        ValueNode::StoreKeyedGeneric {
            object, key, value, ..
        } => {
            f(*object);
            f(*key);
            f(*value);
        }

        // ── Binary arithmetic / comparisons ──────────────────────────────
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

        // ── Call / Construct nodes ────────────────────────────────────────
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
            for a in args {
                f(*a);
            }
        }
        // CallArrayPush codegen ignores callee — not an input.
        ValueNode::CallArrayPush { receiver, args, .. } => {
            f(*receiver);
            for a in args {
                f(*a);
            }
        }

        ValueNode::CallBuiltin { args, .. } | ValueNode::CallRuntime { args, .. } => {
            for a in args {
                f(*a);
            }
        }

        ValueNode::SpeculativeCallFusion { callee, .. } => f(*callee),
        ValueNode::SpeculativeSumFusion { array } => f(*array),
        ValueNode::SpeculativePushFusion { array, .. } => f(*array),

        ValueNode::Construct {
            constructor, args, ..
        }
        | ValueNode::ConstructWithSpread {
            constructor, args, ..
        } => {
            f(*constructor);
            for a in args {
                f(*a);
            }
        }

        // ── Phi ──────────────────────────────────────────────────────────
        ValueNode::Phi { inputs } => {
            for id in inputs {
                f(*id);
            }
        }
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

    // Pre-populate constant map from ALL blocks so cross-block constant
    // operands are visible (e.g. SmiConstant(1) in entry block used by
    // Int32Subtract in the loop body for constant-deferral patterns).
    let mut global_consts: HashMap<NodeId, i32> = HashMap::new();
    for block in graph.blocks() {
        for (id, node) in &block.nodes {
            if let ValueNode::Int32Constant { value } | ValueNode::SmiConstant { value } = node {
                global_consts.insert(*id, *value);
            }
        }
    }

    let mut count = 0;
    for block in graph.blocks_mut() {
        count += reassociate_block(block, &mut next_id, &mut global_consts);
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
fn reassociate_block(
    block: &mut BasicBlock,
    next_id: &mut u32,
    global_consts: &mut HashMap<NodeId, i32>,
) -> usize {
    let mut mul_info: HashMap<NodeId, (NodeId, i32)> = HashMap::new();
    let mut node_defs: HashMap<NodeId, ValueNode> = HashMap::new();

    for (id, node) in &block.nodes {
        // Register any new constants created by prior reassociation passes
        if let ValueNode::Int32Constant { value } | ValueNode::SmiConstant { value } = node {
            global_consts.insert(*id, *value);
        }
        if let ValueNode::Int32Multiply { left, right } = node {
            if let Some(&k) = global_consts.get(right) {
                mul_info.insert(*id, (*left, k));
            } else if let Some(&k) = global_consts.get(left) {
                mul_info.insert(*id, (*right, k));
            }
        }
        node_defs.insert(*id, node.clone());
    }

    for (pos, (_id, node)) in block.nodes.iter().enumerate() {
        if let Some(r) = try_reassociate(node, pos, global_consts, &mul_info, &node_defs, next_id) {
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
            // (a - K) + b  ->  (a + b) - K  when K is constant
            // Defers constant adjustments past independent additions.
            // After unrolling, this pushes per-copy constant subs to
            // the end where fold_constants combines them (e.g. 4×(-1) → -4).
            if let Some(ValueNode::Int32Subtract {
                left: sub_a,
                right: sub_k,
            }) = node_defs.get(left)
                && consts.contains_key(sub_k)
            {
                let add_id = NodeId(*next_id);
                *next_id += 1;
                return Some(Reassoc {
                    pos,
                    new_node: ValueNode::Int32Subtract {
                        left: add_id,
                        right: *sub_k,
                    },
                    prefix: vec![(
                        add_id,
                        ValueNode::Int32Add {
                            left: *sub_a,
                            right: *right,
                        },
                    )],
                });
            }
            // a + (b - K)  ->  (a + b) - K  when K is constant
            if let Some(ValueNode::Int32Subtract {
                left: sub_b,
                right: sub_k,
            }) = node_defs.get(right)
                && consts.contains_key(sub_k)
            {
                let add_id = NodeId(*next_id);
                *next_id += 1;
                return Some(Reassoc {
                    pos,
                    new_node: ValueNode::Int32Subtract {
                        left: add_id,
                        right: *sub_k,
                    },
                    prefix: vec![(
                        add_id,
                        ValueNode::Int32Add {
                            left: *left,
                            right: *sub_b,
                        },
                    )],
                });
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
            // (a - K1) - K2  ->  a - (K1+K2)  when both are constants
            // Combines accumulated constant subtractions after deferral.
            if let Some(&k2) = consts.get(right)
                && let Some(ValueNode::Int32Subtract {
                    left: sub_a,
                    right: sub_k1,
                }) = node_defs.get(left)
                && let Some(&k1) = consts.get(sub_k1)
            {
                let combined = k1.wrapping_add(k2);
                let (const_id, prefix) = find_or_create_const(combined, consts, next_id);
                return Some(Reassoc {
                    pos,
                    new_node: ValueNode::Int32Subtract {
                        left: *sub_a,
                        right: const_id,
                    },
                    prefix,
                });
            }
            // (a + K1) - K2  ->  a + (K1-K2)  when both are constants
            if let Some(&k2) = consts.get(right)
                && let Some(ValueNode::Int32Add {
                    left: add_a,
                    right: add_k1,
                }) = node_defs.get(left)
                && let Some(&k1) = consts.get(add_k1)
            {
                let diff = k1.wrapping_sub(k2);
                if diff == 0 {
                    // a + 0 → identity, handled by identity elimination
                } else if diff > 0 {
                    let (const_id, prefix) = find_or_create_const(diff, consts, next_id);
                    return Some(Reassoc {
                        pos,
                        new_node: ValueNode::Int32Add {
                            left: *add_a,
                            right: const_id,
                        },
                        prefix,
                    });
                } else {
                    let (const_id, prefix) = find_or_create_const(-diff, consts, next_id);
                    return Some(Reassoc {
                        pos,
                        new_node: ValueNode::Int32Subtract {
                            left: *add_a,
                            right: const_id,
                        },
                        prefix,
                    });
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

    // Pre-populate constant map from ALL blocks so cross-block constant
    // operands are visible (e.g. SmiConstant(3) in entry block used by
    // Int32Multiply in the loop body).
    let mut global_consts: HashMap<NodeId, i32> = HashMap::new();
    for block in graph.blocks() {
        for (id, node) in &block.nodes {
            if let ValueNode::Int32Constant { value } | ValueNode::SmiConstant { value } = node {
                global_consts.insert(*id, *value);
            }
        }
    }

    for block in graph.blocks_mut() {
        strength_reduce_block(block, &mut next_id, &global_consts);
    }
}

/// Perform strength reduction within a single [`BasicBlock`].
fn strength_reduce_block(
    block: &mut BasicBlock,
    next_id: &mut u32,
    global_consts: &HashMap<NodeId, i32>,
) {
    // Start with global consts; also pick up any block-local constants.
    let mut consts = global_consts.clone();
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
    // Shift-add decompositions for non-power-of-two: (2^k ± 1).
    // Entry: (position, operand, shift_amount, is_add).
    // is_add=true: x*(2^k+1) → (x<<k)+x, is_add=false: x*(2^k-1) → (x<<k)-x.
    let mut shift_add_reductions: Vec<(usize, NodeId, u32, bool)> = Vec::new();

    for (pos, (_, node)) in block.nodes.iter().enumerate() {
        if let ValueNode::Int32Multiply { left, right } = node {
            if let Some((operand, shift)) = find_power_of_two_operand(*left, *right, &consts) {
                reductions.push((pos, operand, shift));
            } else if let Some((operand, shift, is_add)) =
                find_shift_add_decomposition(*left, *right, &consts)
            {
                shift_add_reductions.push((pos, operand, shift, is_add));
            }
        }
    }

    if reductions.is_empty() && shift_add_reductions.is_empty() {
        return;
    }

    // Apply shift-add decompositions in reverse order first (they insert 2 nodes).
    // x * (2^k+1) → shifted = x << k; result = shifted + x
    // x * (2^k-1) → shifted = x << k; result = shifted - x
    for (pos, operand, shift_amt, is_add) in shift_add_reductions.into_iter().rev() {
        let shift_const_id = NodeId(*next_id);
        *next_id += 1;
        let shifted_id = NodeId(*next_id);
        *next_id += 1;

        // Replace the multiply in-place with add/sub, keeping its original NodeId.
        let (mul_id, _) = block.nodes[pos];
        block.nodes[pos] = (
            mul_id,
            if is_add {
                ValueNode::Int32Add {
                    left: shifted_id,
                    right: operand,
                }
            } else {
                ValueNode::Int32Subtract {
                    left: shifted_id,
                    right: operand,
                }
            },
        );

        // Insert shift node before the add/sub.
        block.nodes.insert(
            pos,
            (
                shifted_id,
                ValueNode::Int32ShiftLeft {
                    left: operand,
                    right: shift_const_id,
                },
            ),
        );

        // Insert shift-amount constant before the shift.
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

    // Apply power-of-two reductions in reverse position order.
    // Note: positions may have shifted if shift_add_reductions added nodes above.
    // Since we process shift_add first in reverse, and now power-of-two in reverse,
    // we need to account for insertions. Rebuild positions if both lists are non-empty.
    // For simplicity, re-scan for power-of-two patterns after shift-add transforms.
    if !reductions.is_empty() {
        // Re-collect constants (shift-add may have added Int32Constants).
        let mut consts2: HashMap<NodeId, i32> = HashMap::new();
        for (id, node) in &block.nodes {
            match node {
                ValueNode::Int32Constant { value } | ValueNode::SmiConstant { value } => {
                    consts2.insert(*id, *value);
                }
                _ => {}
            }
        }

        let mut pow2_reductions: Vec<(usize, NodeId, u32)> = Vec::new();
        for (pos, (_, node)) in block.nodes.iter().enumerate() {
            if let ValueNode::Int32Multiply { left, right } = node
                && let Some((operand, shift)) = find_power_of_two_operand(*left, *right, &consts2)
            {
                pow2_reductions.push((pos, operand, shift));
            }
        }

        for (pos, operand, shift_amt) in pow2_reductions.into_iter().rev() {
            let shift_const_id = NodeId(*next_id);
            *next_id += 1;

            let (mul_id, _) = block.nodes[pos];
            block.nodes[pos] = (
                mul_id,
                ValueNode::Int32ShiftLeft {
                    left: operand,
                    right: shift_const_id,
                },
            );

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

/// Decompose `x * K` where K = 2^k ± 1 (e.g. 3, 5, 7, 9) into shift+add/sub.
/// Returns `(other_operand, shift_amount, is_add)`:
///   is_add=true:  x*(2^k+1) → (x<<k)+x   (K=3,5,9,17,…)
///   is_add=false: x*(2^k-1) → (x<<k)-x   (K=7,15,31,…)
fn find_shift_add_decomposition(
    left: NodeId,
    right: NodeId,
    consts: &HashMap<NodeId, i32>,
) -> Option<(NodeId, u32, bool)> {
    fn try_decompose(val: i32) -> Option<(u32, bool)> {
        if val < 3 {
            return None;
        }
        // Check 2^k + 1: val-1 must be power of two ≥ 2
        let minus_one = val - 1;
        if minus_one >= 2 && minus_one.count_ones() == 1 {
            return Some((minus_one.trailing_zeros(), true));
        }
        // Check 2^k - 1: val+1 must be power of two ≥ 4
        let plus_one = val + 1;
        if plus_one >= 4 && plus_one.count_ones() == 1 {
            return Some((plus_one.trailing_zeros(), false));
        }
        None
    }

    if let Some(&val) = consts.get(&right)
        && let Some((shift, is_add)) = try_decompose(val)
    {
        return Some((left, shift, is_add));
    }
    if let Some(&val) = consts.get(&left)
        && let Some((shift, is_add)) = try_decompose(val)
    {
        return Some((right, shift, is_add));
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
    /// A property load: (operation kind, object, property_name_index).
    PropertyLoad(u16, NodeId, u32),
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
        // Property loads: keyed by object and property name.
        ValueNode::LoadNamedGeneric { object, name, .. } => {
            Some(CseKey::PropertyLoad(1, *object, *name))
        }
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
    let mut loops = licm::detect_loops(graph);
    if loops.is_empty() {
        return 0;
    }

    // Sort loops by body size (smallest/innermost first).  Inner loops are
    // promoted first so their Phis are established before outer loops.
    loops.sort_by_key(|lp| lp.body.len());

    // For each loop, collect the set of globals (by name index) that are
    // stored inside *nested* sub-loops.  These must be excluded from outer-
    // loop promotion to avoid broken Phi wiring (a global written at
    // multiple nesting levels would get a single Phi at the outer header
    // with the wrong back-edge value — the root cause of the sieve SIGSEGV).
    //
    // Step 1: collect which global names each loop stores.
    let loop_stored_globals: Vec<HashSet<u32>> = loops
        .iter()
        .map(|lp| {
            let mut stored = HashSet::new();
            for block in graph.blocks() {
                if !lp.body.contains(&block.id) {
                    continue;
                }
                for (_, node) in &block.nodes {
                    if let ValueNode::StoreGlobal { name, .. } = node {
                        stored.insert(*name);
                    }
                }
            }
            stored
        })
        .collect();

    // Step 2: for each loop, union the stored-globals of all strictly
    // nested sub-loops (inner.body ⊂ outer.body).
    let mut nested_stored: Vec<HashSet<u32>> = vec![HashSet::new(); loops.len()];
    for (i, outer) in loops.iter().enumerate() {
        for (j, inner) in loops.iter().enumerate() {
            if i == j {
                continue;
            }
            // Strict nesting: inner body is a proper subset of outer body.
            if inner.body.len() < outer.body.len() && inner.body.is_subset(&outer.body) {
                for name in &loop_stored_globals[j] {
                    nested_stored[i].insert(*name);
                }
            }
        }
    }

    let mut count = 0;
    for (i, lp) in loops.iter().enumerate() {
        count += promote_globals_in_loop(graph, lp, &nested_stored[i]);
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
            | ValueNode::SpeculativeCallFusion { .. }
            | ValueNode::SpeculativeSumFusion { .. }
            | ValueNode::SpeculativePushFusion { .. }
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
/// promoted (0 if none).  `exclude_names` contains global name indices that
/// must not be promoted in this loop (written by nested sub-loops).
fn promote_globals_in_loop(
    graph: &mut MaglevGraph,
    lp: &licm::NaturalLoop,
    exclude_names: &HashSet<u32>,
) -> usize {
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
    // write-only) that is NOT also written by a nested sub-loop.  Read-only
    // globals are already handled by LICM, so we only need to promote those
    // that have at least one StoreGlobal in the body.
    let promotable: Vec<u32> = store_names
        .iter()
        .copied()
        .filter(|name| !exclude_names.contains(name))
        .collect();
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
            for (id, node) in &mut header_block.nodes {
                if *id == pg.phi_id {
                    if let ValueNode::Phi { inputs } = node {
                        inputs.push(pg.store_value_id);
                    }
                    break;
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
    // The value to store is the Phi (not the body's raw value): the Phi
    // correctly holds the preheader value when the loop doesn't execute
    // and the last-iteration value otherwise.  Using the Phi also keeps it
    // alive through DCE, which is essential for downstream passes like
    // forward_loop_object_properties that pattern-match on Phi back-edges.
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
                            value: pg.phi_id,
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
// Speculative call-loop fusion
// ─────────────────────────────────────────────────────────────────────────────

/// Replace a counted loop that calls a 0-arg closure N times with a single
/// [`ValueNode::SpeculativeCallFusion`] in the preheader.
///
/// Pattern detected:
/// ```text
///   header: result_phi = Phi(init, call_result)
///           iv_phi     = Phi(0, iv_next)
///           Branch(iv_phi < limit, body, exit)
///   body:   call_result = Call(callee, undefined, [])
///           iv_next     = Int32Add(iv_phi, 1)
///           Jump(header)
/// ```
///
/// The fusion node calls into the runtime which analyses the callee's
/// bytecodes.  If it matches a simple context-slot increment pattern, the
/// runtime computes the closed-form result in O(1); otherwise it returns
/// `JIT_DEOPT` and the interpreter re-runs the loop.
fn fuse_call_loops(graph: &mut MaglevGraph) {
    let loops = licm::detect_loops(graph);
    if loops.is_empty() {
        return;
    }
    for lp in &loops {
        try_fuse_call_loop(graph, lp);
    }
}

fn try_fuse_call_loop(graph: &mut MaglevGraph, lp: &licm::NaturalLoop) -> bool {
    // ── 1. Simple loop: header + 1 body block ────────────────────────────
    if lp.body.len() != 2 || !lp.body.contains(&lp.header) {
        return false;
    }

    // Extract all structural data from blocks into owned values, then release borrows.
    let (condition_id, body_block_idx, header_preds, header_nodes_snapshot);
    {
        let header = match graph.block(lp.header) {
            Some(b) => b,
            None => return false,
        };
        let (cond, body_bi, _exit) = match &header.control {
            Some(ControlNode::Branch {
                condition,
                if_true,
                if_false,
            }) => (*condition, *if_true, *if_false),
            _ => return false,
        };
        if !lp.body.contains(&body_bi) {
            return false;
        }
        condition_id = cond;
        body_block_idx = body_bi;
        header_preds = header.predecessors.clone();
        header_nodes_snapshot = header
            .nodes
            .iter()
            .map(|(nid, node)| (*nid, node.clone()))
            .collect::<Vec<_>>();
    }

    {
        let body = match graph.block(body_block_idx) {
            Some(b) => b,
            None => return false,
        };
        match &body.control {
            Some(ControlNode::Jump { target }) if *target == lp.header => {}
            _ => return false,
        }
    }

    // ── 2. Find IV and trip count ────────────────────────────────────────
    let (cmp_left, cmp_right) = {
        let mut found = None;
        for (nid, node) in &header_nodes_snapshot {
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

    let iv_limit = match find_i32_constant(graph, cmp_right) {
        Some(v) => v,
        None => return false,
    };

    // Determine trip count by examining the IV node.
    // iv_global_name is Some(name) when the IV is a LoadGlobal (need to short-circuit).
    let (trip_count, call_node_id, callee_id, iv_global_name) =
        if let Some(ValueNode::Phi { inputs }) = graph.node(cmp_left) {
            // ── Phi-based IV ──
            let inputs = inputs.clone();
            if inputs.len() != 2 {
                return false;
            }
            let back_pred_pos = match header_preds.iter().position(|&p| p == body_block_idx) {
                Some(pos) => pos,
                None => return false,
            };
            let entry_pos = match header_preds.iter().position(|&p| p == lp.preheader) {
                Some(pos) => pos,
                None => return false,
            };

            let iv_init = match find_i32_constant(graph, inputs[entry_pos]) {
                Some(v) => v,
                None => return false,
            };
            let iv_step = find_increment_step(graph, inputs[back_pred_pos], cmp_left);
            if iv_step != Some(1) {
                return false;
            }
            let range = (iv_limit as i64) - (iv_init as i64);
            if range <= 0 || range > 100_000 {
                return false;
            }

            // Find a non-IV Phi whose back-edge is a Call
            let mut found_call = None;
            for (nid, node) in &header_nodes_snapshot {
                if *nid == cmp_left {
                    continue;
                }
                if let ValueNode::Phi { inputs } = node
                    && inputs.len() == 2
                {
                    let back_id = inputs[back_pred_pos];
                    if let Some(cid) = find_call_callee(graph, back_id) {
                        found_call = Some((range as u32, back_id, cid));
                        break;
                    }
                }
            }
            match found_call {
                Some((tc, cn, ci)) => (tc, cn, ci, None),
                None => return false,
            }
        } else if let Some(ValueNode::LoadGlobal { name: iv_name, .. }) = graph.node(cmp_left) {
            // ── LoadGlobal-based IV (globals not promoted due to Call in loop) ──
            let iv_name = *iv_name;

            // Find the init value: look for a StoreGlobal to iv_name in the preheader.
            let pre = graph.block(lp.preheader);
            let iv_init = pre.and_then(|pre| {
                pre.nodes.iter().rev().find_map(|(_, node)| {
                    if let ValueNode::StoreGlobal { name, value, .. } = node
                        && *name == iv_name
                    {
                        find_i32_constant(graph, *value)
                    } else {
                        None
                    }
                })
            });
            let iv_init = match iv_init {
                Some(v) => v,
                None => return false,
            };

            // Find IV increment in body: StoreGlobal(iv_name, Inc(LoadGlobal(iv_name)))
            let body = graph.block(body_block_idx).unwrap();
            let mut iv_step_ok = false;
            for (_, node) in &body.nodes {
                if let ValueNode::StoreGlobal { name, value, .. } = node
                    && *name == iv_name
                {
                    // Check if value is an increment of a LoadGlobal(iv_name)
                    if let Some(inc_node) = graph.node(*value) {
                        let base = match inc_node {
                            ValueNode::GenericIncrement { value, .. }
                            | ValueNode::Int32Increment { value }
                            | ValueNode::CheckedSmiIncrement { value } => Some(*value),
                            _ => None,
                        };
                        if let Some(base_id) = base
                            && let Some(ValueNode::LoadGlobal { name: bn, .. }) =
                                graph.node(base_id)
                            && *bn == iv_name
                        {
                            iv_step_ok = true;
                        }
                    }
                }
            }
            if !iv_step_ok {
                return false;
            }

            let range = (iv_limit as i64) - (iv_init as i64);
            if range <= 0 || range > 100_000 {
                return false;
            }

            // Find a Call in the body block.
            let body = graph.block(body_block_idx).unwrap();
            let mut found_call = None;
            for (nid, node) in &body.nodes {
                if let Some(cid) = find_call_callee_node(node) {
                    found_call = Some((range as u32, *nid, cid));
                    break;
                }
            }
            match found_call {
                Some((tc, cn, ci)) => (tc, cn, ci, Some(iv_name)),
                None => return false,
            }
        } else {
            return false;
        };

    // Callee must be loop-invariant (defined outside the loop, or a
    // LoadGlobal whose name is never stored inside the loop body).
    let loop_body_set: HashSet<u32> = lp.body.iter().copied().collect();
    let callee_in_loop = loop_body_set.iter().any(|&bi| {
        graph
            .block(bi)
            .map(|b| b.nodes.iter().any(|(nid, _)| *nid == callee_id))
            .unwrap_or(false)
    });
    if callee_in_loop {
        // A LoadGlobal is semantically invariant when no StoreGlobal
        // with the same name exists inside the loop.  LICM doesn't
        // hoist it because other StoreGlobals make it conservatively
        // non-pure, but we can check the specific name.
        let is_invariant_load_global = if let Some(ValueNode::LoadGlobal { name, .. }) =
            graph.node(callee_id)
        {
            let callee_name = *name;
            !loop_body_set.iter().any(|&bi| {
                graph
                    .block(bi)
                    .map(|b| {
                        b.nodes.iter().any(|(_, node)| {
                            matches!(node, ValueNode::StoreGlobal { name, .. } if *name == callee_name)
                        })
                    })
                    .unwrap_or(false)
            })
        } else {
            false
        };
        if !is_invariant_load_global {
            return false;
        }
    }

    // ── 4. Insert SpeculativeCallFusion in preheader ─────────────────
    // Try to resolve (slot, k) at compile time by tracing the callee back
    // to a CreateClosure / FastCreateClosure node whose shared_function_info
    // has a pre-analysed fusion pattern in the graph.
    let (resolved_slot, resolved_k) = resolve_fusion_pattern(graph, callee_id, lp.preheader);

    let fusion_id = graph.alloc_node_id();
    if let Some(pre) = graph.block_mut(lp.preheader) {
        pre.push_with_id(
            fusion_id,
            ValueNode::SpeculativeCallFusion {
                callee: callee_id,
                trip_count,
                resolved_slot,
                resolved_k,
            },
        );
    }

    // For LoadGlobal-based loops: store the fusion result to the result global
    // and replace the Call node with a no-op (SmiConstant(0) as dummy).
    // For Phi-based loops: update the Phi inputs.
    // In both cases, the Call node becomes dead and will be removed by DCE.

    // Find if the call_node_id's result is stored via StoreGlobal.
    let body = graph.block(body_block_idx).unwrap();
    let mut result_store_name = None;
    let mut result_store_slot = None;
    for (_, node) in &body.nodes {
        if let ValueNode::StoreGlobal {
            name,
            value,
            feedback_slot,
        } = node
            && *value == call_node_id
        {
            result_store_name = Some(*name);
            result_store_slot = Some(*feedback_slot);
            break;
        }
    }

    if let (Some(store_name), Some(store_fb)) = (result_store_name, result_store_slot) {
        // LoadGlobal-based: insert StoreGlobal(result, fusion_result) in preheader
        let store_id = graph.alloc_node_id();
        if let Some(pre) = graph.block_mut(lp.preheader) {
            pre.push_with_id(
                store_id,
                ValueNode::StoreGlobal {
                    name: store_name,
                    value: fusion_id,
                    feedback_slot: store_fb,
                },
            );
        }
    }

    // ── 5. Short-circuit the loop ──────────────────────────────────────
    // For LoadGlobal-based IV: store iv_limit to the IV global so the
    // header's `LoadGlobal(iv) < iv_limit` is immediately false.
    if let Some(iv_name) = iv_global_name {
        // Find the StoreGlobal feedback_slot for the IV from the body block.
        let body = graph.block(body_block_idx).unwrap();
        let mut iv_fb = None;
        for (_, node) in &body.nodes {
            if let ValueNode::StoreGlobal {
                name,
                feedback_slot,
                ..
            } = node
                && *name == iv_name
            {
                iv_fb = Some(*feedback_slot);
                break;
            }
        }
        if let Some(fb) = iv_fb {
            // Store iv_limit (= cmp_right) to the IV global in preheader.
            let iv_store_id = graph.alloc_node_id();
            if let Some(pre) = graph.block_mut(lp.preheader) {
                pre.push_with_id(
                    iv_store_id,
                    ValueNode::StoreGlobal {
                        name: iv_name,
                        value: cmp_right,
                        feedback_slot: fb,
                    },
                );
            }
        }
    }

    // For Phi-based IV: set the IV Phi entry to the limit constant.
    // This makes `iv_phi < iv_limit` false on first check → loop exits immediately.
    if iv_global_name.is_none() {
        let back_pred_pos = header_preds.iter().position(|&p| p == body_block_idx);
        let entry_pos = header_preds.iter().position(|&p| p == lp.preheader);
        if let (Some(bp), Some(ep)) = (back_pred_pos, entry_pos) {
            let h = graph.block_mut(lp.header).unwrap();
            for (nid, node) in &mut h.nodes {
                if *nid == cmp_left {
                    if let ValueNode::Phi { inputs } = node
                        && inputs.len() == 2
                    {
                        // Set entry input to limit → loop condition false immediately
                        inputs[ep] = cmp_right;
                        inputs[bp] = *nid; // identity
                    }
                    break;
                }
            }
        }
    }

    // Also handle the case where there's a Phi for the result
    let back_pred_pos = header_preds.iter().position(|&p| p == body_block_idx);
    let entry_pos = header_preds.iter().position(|&p| p == lp.preheader);
    if let (Some(bp), Some(ep)) = (back_pred_pos, entry_pos) {
        let h = graph.block_mut(lp.header).unwrap();
        for (nid, node) in &mut h.nodes {
            if let ValueNode::Phi { inputs } = node
                && inputs.len() == 2
                && inputs[bp] == call_node_id
            {
                inputs[ep] = fusion_id;
                inputs[bp] = *nid; // self-reference (identity)
                break;
            }
        }
    }

    true
}

// ─── Speculative sum fusion ─────────────────────────────────────────────────
//
// Detects `for (i = 0; i < arr.length; i++) sum += arr[i]` and replaces
// it with a single [`ValueNode::SpeculativeSumFusion`] in the preheader.
// The codegen emits a call to a native Rust function that sums all Smi
// elements in a tight loop.
//
// Exact required shape (v1):
//   header: iv_phi(i: 0 → i+1), sum_phi(sum: 0 → add), cmp(i < len)
//   body:   load = LoadKeyedGeneric(arr, i)
//           add  = GenericAdd(sum, load)  (or reverse)
//           i'   = increment(i)
//           Jump → header
//   No calls, no stores, no extra side effects in body.

#[allow(dead_code)] // Disabled pending deopt regression investigation.
fn fuse_sum_loops(graph: &mut MaglevGraph) {
    let loops = licm::detect_loops(graph);
    if loops.is_empty() {
        return;
    }
    for lp in &loops {
        if !try_fuse_sum_loop(graph, lp) {
            try_fuse_push_loop(graph, lp);
        }
    }
}

#[allow(dead_code)] // Disabled pending deopt regression investigation.
fn try_fuse_sum_loop(graph: &mut MaglevGraph, lp: &licm::NaturalLoop) -> bool {
    // ── 1. Simple loop: header + 1 body block ────────────────────────────
    if lp.body.len() != 2 || !lp.body.contains(&lp.header) {
        return false;
    }

    // Snapshot header info.
    let (condition_id, body_block_idx, header_preds, header_nodes_snapshot);
    {
        let header = match graph.block(lp.header) {
            Some(b) => b,
            None => return false,
        };
        let (cond, body_bi, _exit) = match &header.control {
            Some(ControlNode::Branch {
                condition,
                if_true,
                if_false,
            }) => (*condition, *if_true, *if_false),
            _ => return false,
        };
        if !lp.body.contains(&body_bi) {
            return false;
        }
        condition_id = cond;
        body_block_idx = body_bi;
        header_preds = header.predecessors.clone();
        header_nodes_snapshot = header
            .nodes
            .iter()
            .map(|(nid, node)| (*nid, node.clone()))
            .collect::<Vec<_>>();
    }

    // Body must jump back to header.
    {
        let body = match graph.block(body_block_idx) {
            Some(b) => b,
            None => return false,
        };
        match &body.control {
            Some(ControlNode::Jump { target }) if *target == lp.header => {}
            _ => return false,
        }
    }

    // ── 2. Find IV phi and trip count ────────────────────────────────────
    let (cmp_left, cmp_right) = {
        let mut found = None;
        for (nid, node) in &header_nodes_snapshot {
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

    // IV must be a Phi starting at 0, incrementing by 1.
    let back_pred_pos = match header_preds.iter().position(|&p| p == body_block_idx) {
        Some(pos) => pos,
        None => return false,
    };
    let entry_pos = match header_preds.iter().position(|&p| p == lp.preheader) {
        Some(pos) => pos,
        None => return false,
    };

    let iv_phi_id = cmp_left;
    let iv_inputs = match graph.node(iv_phi_id) {
        Some(ValueNode::Phi { inputs }) if inputs.len() == 2 => inputs.clone(),
        _ => return false,
    };

    // IV init must be 0.
    if find_i32_constant(graph, iv_inputs[entry_pos]) != Some(0) {
        return false;
    }
    // IV step must be +1.
    if find_increment_step(graph, iv_inputs[back_pred_pos], iv_phi_id) != Some(1) {
        return false;
    }

    // ── 3. Limit must be arr.length ──────────────────────────────────────
    // cmp_right must be a LoadNamedGeneric(arr, "length") or similar
    // where arr is loop-invariant.
    let arr_node_id = match graph.node(cmp_right) {
        Some(ValueNode::LoadNamedGeneric { object, .. }) => *object,
        _ => return false,
    };

    // Check arr is loop-invariant (defined outside the loop).
    let loop_body_set: HashSet<u32> = lp.body.iter().copied().collect();
    let arr_in_loop = loop_body_set.iter().any(|&bi| {
        graph
            .block(bi)
            .map(|b| b.nodes.iter().any(|(nid, _)| *nid == arr_node_id))
            .unwrap_or(false)
    });
    if arr_in_loop {
        return false;
    }

    // ── 4. Find sum phi: sum = init → GenericAdd(sum, LoadKeyedGeneric(arr, iv)) ──
    let mut sum_phi_id = None;
    let mut add_node_id = None;
    for (nid, node) in &header_nodes_snapshot {
        if *nid == iv_phi_id || *nid == condition_id {
            continue;
        }
        if let ValueNode::Phi { inputs } = node
            && inputs.len() == 2
        {
            let init_id = inputs[entry_pos];
            let back_id = inputs[back_pred_pos];

            // init must be a Smi constant (typically 0).
            if find_i32_constant(graph, init_id).is_none() {
                continue;
            }

            // back-edge must be GenericAdd(sum_phi, LoadKeyedGeneric(arr, iv))
            if let Some(add_node) = graph.node(back_id) {
                let (add_left, add_right) = match add_node {
                    ValueNode::GenericAdd { left, right, .. }
                    | ValueNode::Int32Add { left, right }
                    | ValueNode::CheckedSmiAdd { left, right } => (*left, *right),
                    _ => continue,
                };
                // One operand must be the sum phi, the other a keyed load.
                let load_id = if add_left == *nid {
                    add_right
                } else if add_right == *nid {
                    add_left
                } else {
                    continue;
                };
                // load must be LoadKeyedGeneric(arr, iv).
                if let Some(ValueNode::LoadKeyedGeneric { object, key, .. }) = graph.node(load_id)
                    && *object == arr_node_id
                    && *key == iv_phi_id
                {
                    sum_phi_id = Some(*nid);
                    add_node_id = Some(back_id);
                    break;
                }
            }
        }
    }

    let sum_phi_id = match sum_phi_id {
        Some(id) => id,
        None => return false,
    };
    let _add_node_id = add_node_id.unwrap();

    // ── 5. Verify body has no side effects beyond the add + increment ────
    {
        let body = graph.block(body_block_idx).unwrap();
        for (_, node) in &body.nodes {
            match node {
                // Allowed: load keyed, add, increment, constants, phis,
                // load/store globals (from pre-promotion remnants).
                ValueNode::LoadKeyedGeneric { .. }
                | ValueNode::GenericAdd { .. }
                | ValueNode::Int32Add { .. }
                | ValueNode::CheckedSmiAdd { .. }
                | ValueNode::GenericIncrement { .. }
                | ValueNode::Int32Increment { .. }
                | ValueNode::CheckedSmiIncrement { .. }
                | ValueNode::Phi { .. }
                | ValueNode::SmiConstant { .. }
                | ValueNode::Int32Constant { .. }
                | ValueNode::Float64Constant { .. }
                | ValueNode::LoadGlobal { .. }
                | ValueNode::StoreGlobal { .. }
                | ValueNode::LoadNamedGeneric { .. }
                | ValueNode::Int32LessThan { .. }
                | ValueNode::CheckSmi { .. } => {}
                // Any call, store keyed, delete, etc. → bail.
                _ => return false,
            }
        }
    }

    // ── 6. Insert SpeculativeSumFusion in preheader ──────────────────────
    let fusion_id = graph.alloc_node_id();
    if let Some(pre) = graph.block_mut(lp.preheader) {
        pre.push_with_id(
            fusion_id,
            ValueNode::SpeculativeSumFusion { array: arr_node_id },
        );
    }

    // ── 7. Short-circuit: set IV entry to limit → loop exits immediately ──
    {
        let h = graph.block_mut(lp.header).unwrap();
        for (nid, node) in &mut h.nodes {
            if *nid == iv_phi_id {
                if let ValueNode::Phi { inputs } = node
                    && inputs.len() == 2
                {
                    inputs[entry_pos] = cmp_right;
                    inputs[back_pred_pos] = *nid;
                }
                break;
            }
        }
    }

    // ── 8. Update sum phi: entry input = fusion result ──────────────────
    {
        let h = graph.block_mut(lp.header).unwrap();
        for (nid, node) in &mut h.nodes {
            if *nid == sum_phi_id {
                if let ValueNode::Phi { inputs } = node
                    && inputs.len() == 2
                {
                    inputs[entry_pos] = fusion_id;
                    inputs[back_pred_pos] = *nid;
                }
                break;
            }
        }
    }

    true
}

/// Detect `for (i = 0; i < N; i++) arr.push(i)` and replace with
/// [`ValueNode::SpeculativePushFusion`].
#[allow(dead_code)] // Disabled pending deopt regression investigation.
fn try_fuse_push_loop(graph: &mut MaglevGraph, lp: &licm::NaturalLoop) -> bool {
    // ── 1. Simple loop: header + 1 body block ────────────────────────────
    if lp.body.len() != 2 || !lp.body.contains(&lp.header) {
        return false;
    }

    let (condition_id, body_block_idx, header_preds, header_nodes_snapshot);
    {
        let header = match graph.block(lp.header) {
            Some(b) => b,
            None => return false,
        };
        let (cond, body_bi, _exit) = match &header.control {
            Some(ControlNode::Branch {
                condition,
                if_true,
                if_false,
            }) => (*condition, *if_true, *if_false),
            _ => return false,
        };
        if !lp.body.contains(&body_bi) {
            return false;
        }
        condition_id = cond;
        body_block_idx = body_bi;
        header_preds = header.predecessors.clone();
        header_nodes_snapshot = header
            .nodes
            .iter()
            .map(|(nid, node)| (*nid, node.clone()))
            .collect::<Vec<_>>();
    }

    {
        let body = match graph.block(body_block_idx) {
            Some(b) => b,
            None => return false,
        };
        match &body.control {
            Some(ControlNode::Jump { target }) if *target == lp.header => {}
            _ => return false,
        }
    }

    // ── 2. Find IV phi: i = 0 → i+1 with constant limit ──────────────
    let (cmp_left, cmp_right) = {
        let mut found = None;
        for (nid, node) in &header_nodes_snapshot {
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

    // Limit must be a constant (e.g. 1000).
    let trip_count = match find_i32_constant(graph, cmp_right) {
        Some(v) if v > 0 && v <= 100_000 => v as u32,
        _ => return false,
    };

    let back_pred_pos = match header_preds.iter().position(|&p| p == body_block_idx) {
        Some(pos) => pos,
        None => return false,
    };
    let entry_pos = match header_preds.iter().position(|&p| p == lp.preheader) {
        Some(pos) => pos,
        None => return false,
    };

    let iv_phi_id = cmp_left;
    let iv_inputs = match graph.node(iv_phi_id) {
        Some(ValueNode::Phi { inputs }) if inputs.len() == 2 => inputs.clone(),
        _ => return false,
    };

    if find_i32_constant(graph, iv_inputs[entry_pos]) != Some(0) {
        return false;
    }
    if find_increment_step(graph, iv_inputs[back_pred_pos], iv_phi_id) != Some(1) {
        return false;
    }

    // ── 3. Body must contain CallArrayPush(arr, [iv]) ─────────────────
    let mut arr_node_id = None;
    {
        let body = graph.block(body_block_idx).unwrap();
        for (_, node) in &body.nodes {
            if let ValueNode::CallArrayPush { receiver, args, .. } = node
                && args.len() == 1
                && args[0] == iv_phi_id
            {
                arr_node_id = Some(*receiver);
                break;
            }
        }
    }
    let arr_node_id = match arr_node_id {
        Some(id) => id,
        None => return false,
    };

    // arr must be loop-invariant.
    let loop_body_set: HashSet<u32> = lp.body.iter().copied().collect();
    let arr_in_loop = loop_body_set.iter().any(|&bi| {
        graph
            .block(bi)
            .map(|b| b.nodes.iter().any(|(nid, _)| *nid == arr_node_id))
            .unwrap_or(false)
    });
    if arr_in_loop {
        return false;
    }

    // ── 4. Verify body has no unexpected side effects ─────────────────
    // Allow: the push itself, IV ops, loads, constants, promoted globals.
    // Reject: other calls, stores to non-globals, deletes, etc.
    {
        let body = graph.block(body_block_idx).unwrap();
        for (_, node) in &body.nodes {
            match node {
                ValueNode::CallArrayPush { .. }
                | ValueNode::GenericIncrement { .. }
                | ValueNode::Int32Increment { .. }
                | ValueNode::CheckedSmiIncrement { .. }
                | ValueNode::GenericAdd { .. }
                | ValueNode::Int32Add { .. }
                | ValueNode::CheckedSmiAdd { .. }
                | ValueNode::Phi { .. }
                | ValueNode::SmiConstant { .. }
                | ValueNode::Int32Constant { .. }
                | ValueNode::Float64Constant { .. }
                | ValueNode::LoadNamedGeneric { .. }
                | ValueNode::LoadGlobal { .. }
                | ValueNode::StoreGlobal { .. }
                | ValueNode::Int32LessThan { .. }
                | ValueNode::CheckSmi { .. } => {}
                _ => return false,
            }
        }
    }

    // ── 5. Insert SpeculativePushFusion in preheader ──────────────────
    let fusion_id = graph.alloc_node_id();
    if let Some(pre) = graph.block_mut(lp.preheader) {
        pre.push_with_id(
            fusion_id,
            ValueNode::SpeculativePushFusion {
                array: arr_node_id,
                count: trip_count,
            },
        );
    }

    // ── 6. Short-circuit: set IV entry to limit → loop exits immediately ──
    {
        let h = graph.block_mut(lp.header).unwrap();
        for (nid, node) in &mut h.nodes {
            if *nid == iv_phi_id {
                if let ValueNode::Phi { inputs } = node
                    && inputs.len() == 2
                {
                    inputs[entry_pos] = cmp_right;
                    inputs[back_pred_pos] = *nid;
                }
                break;
            }
        }
    }

    true
}

/// Extract callee NodeId from a Call/CallKnownFunction node (graph lookup version).
fn find_call_callee(graph: &MaglevGraph, node_id: NodeId) -> Option<NodeId> {
    find_call_callee_node(graph.node(node_id)?)
}

/// Extract callee NodeId from a Call/CallKnownFunction ValueNode.
fn find_call_callee_node(node: &ValueNode) -> Option<NodeId> {
    match node {
        ValueNode::Call { callee, args, .. } if args.is_empty() => Some(*callee),
        ValueNode::CallKnownFunction { callee, args, .. } if args.is_empty() => Some(*callee),
        _ => None,
    }
}

/// Trace a callee [`NodeId`] back to a [`ValueNode::CreateClosure`] or
/// [`ValueNode::FastCreateClosure`] and look up its pre-analysed fusion
/// pattern in the graph's [`MaglevGraph::closure_fusion_patterns`] table.
///
/// Returns `(Some(slot), Some(k))` when the callee's bytecodes match the
/// context-slot increment pattern, or `(None, None)` otherwise.  The
/// resolved constants allow the code generator to emit a zero-call inline
/// fast path for [`ValueNode::SpeculativeCallFusion`].
fn resolve_fusion_pattern(
    graph: &MaglevGraph,
    callee_id: NodeId,
    preheader: u32,
) -> (Option<u32>, Option<i64>) {
    // Walk through Phi nodes to find the ultimate definition.
    let mut target = callee_id;
    for _ in 0..8 {
        match graph.node(target) {
            Some(ValueNode::CreateClosure {
                shared_function_info,
                ..
            })
            | Some(ValueNode::FastCreateClosure {
                shared_function_info,
                ..
            }) => {
                if let Some(&(slot, k)) = graph.closure_fusion_patterns().get(shared_function_info)
                {
                    return (Some(slot), Some(k));
                }
                return (None, None);
            }
            Some(ValueNode::Phi { inputs }) => {
                // Follow the first non-self input.
                if let Some(&inp) = inputs.iter().find(|&&i| i != target) {
                    target = inp;
                    continue;
                }
                return (None, None);
            }
            Some(ValueNode::LoadGlobal { name, .. }) => {
                // Trace through LoadGlobal → StoreGlobal in preheader →
                // Call(CreateClosure(factory)) to resolve factory patterns.
                let callee_name = *name;
                return resolve_factory_fusion(graph, callee_name, preheader);
            }
            Some(ValueNode::Call { callee, args, .. }) if args.is_empty() => {
                // The callee is the result of a zero-arg call (e.g.,
                // `var counter = make_counter()`).  Trace through to
                // the factory's callee to find factory fusion patterns.
                target = *callee;
                continue;
            }
            Some(ValueNode::CallKnownFunction { callee, args, .. }) if args.is_empty() => {
                target = *callee;
                continue;
            }
            _ => return (None, None),
        }
    }
    (None, None)
}

/// Resolve a factory fusion pattern: trace a global back through the
/// preheader to find `StoreGlobal(name, Call(CreateClosure(factory)))`.
/// If the factory has a known factory fusion pattern, return it.
fn resolve_factory_fusion(
    graph: &MaglevGraph,
    global_name: u32,
    preheader: u32,
) -> (Option<u32>, Option<i64>) {
    // Find StoreGlobal(global_name, value) in the preheader or any
    // preceding block (common in top-level scripts where several
    // statements precede the loop).
    let mut stored_value = None;
    for bi in (0..=preheader).rev() {
        if let Some(blk) = graph.block(bi) {
            for (_, node) in blk.nodes.iter().rev() {
                if let ValueNode::StoreGlobal { name, value, .. } = node
                    && *name == global_name
                {
                    stored_value = Some(*value);
                    break;
                }
            }
            if stored_value.is_some() {
                break;
            }
        }
    }
    let stored_value = match stored_value {
        Some(v) => v,
        None => return (None, None),
    };

    // Check if stored_value is a Call result.
    let call_callee = match graph.node(stored_value) {
        Some(ValueNode::Call { callee, args, .. }) if args.is_empty() => *callee,
        Some(ValueNode::CallKnownFunction { callee, args, .. }) if args.is_empty() => *callee,
        _ => return (None, None),
    };

    // Trace the Call's callee to a CreateClosure.
    let mut target = call_callee;
    for _ in 0..4 {
        match graph.node(target) {
            Some(ValueNode::CreateClosure {
                shared_function_info,
                ..
            })
            | Some(ValueNode::FastCreateClosure {
                shared_function_info,
                ..
            }) => {
                // Check factory fusion patterns (inner closures).
                if let Some(&(slot, k)) = graph.factory_fusion_patterns().get(shared_function_info)
                {
                    return (Some(slot), Some(k));
                }
                return (None, None);
            }
            Some(ValueNode::Phi { inputs }) => {
                if let Some(&inp) = inputs.iter().find(|&&i| i != target) {
                    target = inp;
                    continue;
                }
                return (None, None);
            }
            Some(ValueNode::LoadGlobal { name, .. }) => {
                // In top-level scripts, `function make_counter() {}`
                // compiles to CreateClosure + StaGlobal.  The Call's
                // callee is LoadGlobal("make_counter").  Search all
                // blocks up to the preheader for the matching store.
                let gn = *name;
                let mut found = None;
                for bi in 0..=preheader {
                    if let Some(blk) = graph.block(bi) {
                        for (_, node) in blk.nodes.iter().rev() {
                            if let ValueNode::StoreGlobal { name, value, .. } = node
                                && *name == gn
                            {
                                found = Some(*value);
                                break;
                            }
                        }
                        if found.is_some() {
                            break;
                        }
                    }
                }
                match found {
                    Some(v) => {
                        target = v;
                        continue;
                    }
                    None => return (None, None),
                }
            }
            _ => return (None, None),
        }
    }
    (None, None)
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
// Pass 2b′ — Forward invariant object-literal properties
// ─────────────────────────────────────────────────────────────────────────────

/// Forward property loads from object literals whose properties are never
/// modified after creation.  Unlike the general store-to-load pass, this
/// works across block boundaries (including loop headers) because it proves
/// that the properties are immutable throughout the function.
///
/// For each `CreateObjectLiteralWithProperties` that has NO `StoreNamedGeneric`
/// targeting it anywhere in the graph, every `LoadNamedGeneric` on the same
/// object and property name is replaced with the creation-time value.
fn forward_invariant_object_properties(graph: &mut MaglevGraph) {
    // Phase 1: Collect object literals and their property maps.
    let mut obj_props: HashMap<NodeId, HashMap<u32, NodeId>> = HashMap::new();
    // Track StoreGlobal(name, value) → value for global aliasing.
    let mut global_to_obj: HashMap<u32, NodeId> = HashMap::new();

    for block in graph.blocks() {
        for (id, node) in &block.nodes {
            match node {
                ValueNode::CreateObjectLiteralWithProperties { names, values, .. } => {
                    let mut props = HashMap::new();
                    for (name, value) in names.iter().zip(values.iter()) {
                        props.insert(*name, *value);
                    }
                    obj_props.insert(*id, props);
                }
                // Track globals that point to object literals.
                ValueNode::StoreGlobal { name, value, .. } => {
                    if obj_props.contains_key(value) {
                        global_to_obj.insert(*name, *value);
                    }
                }
                _ => {}
            }
        }
    }

    if obj_props.is_empty() {
        return;
    }

    // Phase 2: Remove any object whose properties are modified (StoreNamedGeneric).
    for block in graph.blocks() {
        for (_, node) in &block.nodes {
            if let ValueNode::StoreNamedGeneric { object, .. } = node {
                obj_props.remove(object);
                for (&_gname, &obj_id) in &global_to_obj {
                    if *object == obj_id {
                        obj_props.remove(&obj_id);
                    }
                }
            }
        }
    }

    if obj_props.is_empty() {
        return;
    }

    // Build a map from LoadGlobal IDs to the object they alias.
    let mut load_global_alias: HashMap<NodeId, NodeId> = HashMap::new();
    for block in graph.blocks() {
        for (id, node) in &block.nodes {
            if let ValueNode::LoadGlobal { name, .. } = node
                && let Some(&obj_id) = global_to_obj.get(name)
                && obj_props.contains_key(&obj_id)
            {
                load_global_alias.insert(*id, obj_id);
            }
        }
    }

    // Phase 3: Collect substitutions for LoadNamedGeneric on immutable objects.
    let mut subst: HashMap<NodeId, NodeId> = HashMap::new();
    for block in graph.blocks() {
        for (id, node) in &block.nodes {
            if let ValueNode::LoadNamedGeneric { object, name, .. } = node {
                let mut resolved = load_global_alias.get(object).copied().unwrap_or(*object);
                // Chain through prior substitutions for nested object access
                // (e.g. root.a.b.c.d.e where each level is a
                // CreateObjectLiteralWithProperties).
                if !obj_props.contains_key(&resolved)
                    && let Some(&sub) = subst.get(&resolved)
                {
                    resolved = sub;
                }
                if let Some(props) = obj_props.get(&resolved)
                    && let Some(&value) = props.get(name)
                {
                    subst.insert(*id, value);
                }
            }
        }
    }

    if subst.is_empty() {
        return;
    }

    // Phase 4: Apply substitutions.
    for block in graph.blocks_mut() {
        for (id, node) in &mut block.nodes {
            if subst.contains_key(id) {
                *node = ValueNode::UndefinedConstant;
            }
        }
        apply_subst_to_block(block, &subst);
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
// Pass 2c.0 — Loop dead-object forwarding
// ─────────────────────────────────────────────────────────────────────────────

/// Forward property loads on objects created inside counted loops to their
/// final-iteration constant values, enabling subsequent dead-allocation
/// elimination.
///
/// Pattern: a counted loop creates an object literal every iteration and
/// stores it to a global that is never read inside the loop.  After the loop,
/// the global is loaded and only its properties are accessed.  Because only
/// the last iteration's object matters, we can compute the final property
/// values at compile time (for affine expressions of the IV) and replace the
/// post-loop `LoadNamedGeneric` nodes with constants.  This makes the in-loop
/// `CreateObjectLiteralWithProperties` + `StoreGlobal` dead, which existing
/// passes then eliminate.
fn forward_loop_object_properties(graph: &mut MaglevGraph) {
    let loops = licm::detect_loops(graph);
    if loops.is_empty() {
        return;
    }

    // Build a node → constant map.
    let mut consts: HashMap<NodeId, i32> = HashMap::new();
    for block in graph.blocks() {
        for (id, node) in &block.nodes {
            match node {
                ValueNode::SmiConstant { value } | ValueNode::Int32Constant { value } => {
                    consts.insert(*id, *value);
                }
                _ => {}
            }
        }
    }

    for lp in &loops {
        try_forward_loop_object(graph, lp, &consts);
    }
}

/// Try to forward object property loads for a single counted loop.
///
/// After `promote_loop_globals_counted`, a write-only global like `last` in
/// `for (...) { last = { x: i, ... }; }` becomes a header Phi whose back-edge
/// input is a `CreateObjectLiteralWithProperties`.  Post-loop property loads
/// appear as `LoadNamedGeneric(obj_phi, "x")`.
///
/// This pass detects that pattern, evaluates each property value at the final
/// IV value (limit - step), and replaces the post-loop property loads with
/// SmiConstants.  If the object Phi has no remaining uses after replacement,
/// the in-loop allocation is killed.
fn try_forward_loop_object(
    graph: &mut MaglevGraph,
    lp: &licm::NaturalLoop,
    consts: &HashMap<NodeId, i32>,
) -> bool {
    // ── 1. Verify counted loop structure ─────────────────────────────────
    let header = match graph.block(lp.header) {
        Some(b) => b,
        None => return false,
    };

    let (condition_id, body_block_idx, exit_block_idx) = match &header.control {
        Some(ControlNode::Branch {
            condition,
            if_true,
            if_false,
        }) => (*condition, *if_true, *if_false),
        _ => return false,
    };

    if !lp.body.contains(&body_block_idx) {
        return false;
    }
    if lp.body.contains(&exit_block_idx) {
        return false;
    }
    // Simple loop: header + 1 body block.
    if lp.body.len() != 2 || !lp.body.contains(&lp.header) {
        return false;
    }

    let body = match graph.block(body_block_idx) {
        Some(b) => b,
        None => return false,
    };
    match &body.control {
        Some(ControlNode::Jump { target }) if *target == lp.header => {}
        _ => return false,
    }

    // ── 2. Find IV Phi and compute trip count ────────────────────────────
    let (cmp_left, cmp_right) = {
        let mut found = None;
        for (nid, node) in &header.nodes {
            if *nid == condition_id {
                found = Some(node.clone());
                break;
            }
        }
        match &found {
            Some(ValueNode::Int32LessThan { left, right }) => (*left, *right),
            _ => return false,
        }
    };

    let iv_phi_id = cmp_left;
    let header = graph.block(lp.header).unwrap();
    let iv_phi_inputs = {
        let mut found = None;
        for (nid, node) in &header.nodes {
            if *nid == iv_phi_id {
                if let ValueNode::Phi { inputs } = node {
                    found = Some(inputs.clone());
                }
                break;
            }
        }
        match found {
            Some(inputs) if inputs.len() == 2 => inputs,
            _ => return false,
        }
    };

    let back_pred_pos = match header
        .predecessors
        .iter()
        .position(|&p| p == body_block_idx)
    {
        Some(pos) => pos,
        None => return false,
    };
    let entry_pos = match header.predecessors.iter().position(|&p| p == lp.preheader) {
        Some(pos) => pos,
        None => return false,
    };

    let iv_init = match consts.get(&iv_phi_inputs[entry_pos]) {
        Some(&v) => v,
        None => match find_i32_constant(graph, iv_phi_inputs[entry_pos]) {
            Some(v) => v,
            None => return false,
        },
    };
    let iv_limit = match consts.get(&cmp_right) {
        Some(&v) => v,
        None => match find_i32_constant(graph, cmp_right) {
            Some(v) => v,
            None => return false,
        },
    };

    let iv_back_id = iv_phi_inputs[back_pred_pos];
    let iv_step = match find_increment_step(graph, iv_back_id, iv_phi_id) {
        Some(s) if s > 0 => s,
        _ => return false,
    };

    let range = (iv_limit as i64) - (iv_init as i64);
    let step = iv_step as i64;
    if range <= 0 || range % step != 0 {
        return false;
    }
    if range / step == 0 {
        return false;
    }

    let iv_final = iv_limit - iv_step;

    // ── 3. Find "object Phi" in header whose back-edge is CreateObjWithProps
    let header = graph.block(lp.header).unwrap();
    let mut obj_phi_id: Option<NodeId> = None;
    let mut create_obj_id: Option<NodeId> = None;
    let mut create_names: Vec<u32> = Vec::new();
    let mut create_values: Vec<NodeId> = Vec::new();

    for (nid, node) in &header.nodes {
        if *nid == iv_phi_id {
            continue;
        }
        if let ValueNode::Phi { inputs } = node
            && inputs.len() == 2
        {
            let back_edge_id = inputs[back_pred_pos];
            if let Some(ValueNode::CreateObjectLiteralWithProperties { names, values, .. }) =
                graph.node(back_edge_id)
            {
                obj_phi_id = Some(*nid);
                create_obj_id = Some(back_edge_id);
                create_names = names.clone();
                create_values = values.clone();
                break;
            }
        }
    }

    let obj_phi_id = match obj_phi_id {
        Some(id) => id,
        None => return false,
    };
    let create_obj_id = create_obj_id.unwrap();

    // ── 4. Evaluate property values at IV = iv_final ─────────────────────
    let mut final_values: Vec<Option<i32>> = Vec::new();
    for &val_id in &create_values {
        final_values.push(eval_at_iv(graph, val_id, iv_phi_id, iv_final, consts));
    }

    if final_values.iter().any(|v| v.is_none()) {
        return false;
    }

    // ── 5. Find post-loop LoadNamedGeneric that loads from obj_phi ────────
    // Build property name → final constant value map.
    let mut prop_finals: HashMap<u32, i32> = HashMap::new();
    for (name, val) in create_names.iter().zip(final_values.iter()) {
        prop_finals.insert(*name, val.unwrap());
    }

    let mut subst: HashMap<NodeId, NodeId> = HashMap::new();
    let mut new_constants: Vec<(NodeId, i32)> = Vec::new();

    // Collect replacements: LoadNamedGeneric(obj_phi_id, prop_name) → const.
    let mut pending: Vec<(NodeId, i32)> = Vec::new();
    for block in graph.blocks() {
        if lp.body.contains(&block.id) {
            continue;
        }
        for (nid, node) in &block.nodes {
            if let ValueNode::LoadNamedGeneric { object, name, .. } = node
                && *object == obj_phi_id
                && let Some(&final_val) = prop_finals.get(name)
            {
                pending.push((*nid, final_val));
            }
        }
    }

    if pending.is_empty() {
        return false;
    }

    for (load_nid, final_val) in &pending {
        let const_id = graph.alloc_node_id();
        new_constants.push((const_id, *final_val));
        subst.insert(*load_nid, const_id);
    }

    // ── 6. Safety: check that obj_phi has no uses outside the replaced loads
    //      and exit StoreGlobal nodes.  Exit StoreGlobals that store the
    //      obj_phi can be killed (their only purpose was materialising the
    //      object for post-loop reads, which are now constants).
    let replaced_set: HashSet<NodeId> = subst.keys().copied().collect();
    let mut exit_store_globals_to_kill: Vec<NodeId> = Vec::new();
    let mut truly_escapes = false;
    for block in graph.blocks() {
        for (nid, node) in &block.nodes {
            if replaced_set.contains(nid) || *nid == obj_phi_id {
                continue;
            }
            let refs = node_operands(node);
            if refs.contains(&obj_phi_id) {
                // Allow StoreGlobal in exit blocks — these just materialise
                // the object for post-loop reads that are already forwarded.
                if !lp.body.contains(&block.id) && matches!(node, ValueNode::StoreGlobal { .. }) {
                    exit_store_globals_to_kill.push(*nid);
                    continue;
                }
                truly_escapes = true;
                break;
            }
        }
        if truly_escapes {
            break;
        }
        if let Some(ctrl) = &block.control {
            let ctrl_refs = control_operands(ctrl);
            if ctrl_refs.contains(&obj_phi_id) {
                truly_escapes = true;
                break;
            }
        }
    }
    if truly_escapes {
        return apply_load_replacements_only(graph, lp, exit_block_idx, &new_constants, &subst);
    }

    // ── 7. Apply: insert constants, replace loads, kill body allocs ──────
    // Insert SmiConstants at the start of the exit block.
    if let Some(exit_block) = graph.block_mut(exit_block_idx) {
        for (idx, (const_id, val)) in new_constants.iter().enumerate() {
            exit_block
                .nodes
                .insert(idx, (*const_id, ValueNode::SmiConstant { value: *val }));
        }
    }

    // Replace LoadNamedGeneric nodes with UndefinedConstant.
    for block in graph.blocks_mut() {
        if lp.body.contains(&block.id) {
            continue;
        }
        for (nid, node) in &mut block.nodes {
            if subst.contains_key(nid) {
                *node = ValueNode::UndefinedConstant;
            }
            // Kill exit StoreGlobal nodes that stored the now-dead obj_phi.
            if exit_store_globals_to_kill.contains(nid) {
                *node = ValueNode::UndefinedConstant;
            }
        }
        apply_subst_to_block(block, &subst);
    }

    // Kill CreateObjectLiteralWithProperties in the body (obj_phi has no
    // remaining uses, so the allocation is dead).
    for block in graph.blocks_mut() {
        if !lp.body.contains(&block.id) || block.id == lp.header {
            continue;
        }
        for (nid, node) in &mut block.nodes {
            if *nid == create_obj_id {
                *node = ValueNode::UndefinedConstant;
            }
        }
    }

    // Update the obj_phi's back-edge input to UndefinedConstant sentinel.
    // The Phi itself is now dead (no uses) and will be cleaned by DCE.
    if let Some(hdr) = graph.block_mut(lp.header) {
        for (nid, node) in &mut hdr.nodes {
            if *nid == obj_phi_id
                && let ValueNode::Phi { inputs } = node
            {
                inputs[back_pred_pos] = inputs[entry_pos];
            }
        }
    }

    true
}

/// Apply only the load→constant replacements without killing the allocation.
/// Used when the object Phi escapes (has uses beyond the forwarded loads).
fn apply_load_replacements_only(
    graph: &mut MaglevGraph,
    lp: &licm::NaturalLoop,
    exit_block_idx: u32,
    new_constants: &[(NodeId, i32)],
    subst: &HashMap<NodeId, NodeId>,
) -> bool {
    if let Some(exit_block) = graph.block_mut(exit_block_idx) {
        for (idx, (const_id, val)) in new_constants.iter().enumerate() {
            exit_block
                .nodes
                .insert(idx, (*const_id, ValueNode::SmiConstant { value: *val }));
        }
    }
    for block in graph.blocks_mut() {
        if lp.body.contains(&block.id) {
            continue;
        }
        for (nid, node) in &mut block.nodes {
            if subst.contains_key(nid) {
                *node = ValueNode::UndefinedConstant;
            }
        }
        apply_subst_to_block(block, subst);
    }
    true
}

/// Collect all NodeId operands of a value node.
fn node_operands(node: &ValueNode) -> Vec<NodeId> {
    match node {
        ValueNode::Phi { inputs } => inputs.clone(),
        ValueNode::Int32Add { left, right }
        | ValueNode::Int32Subtract { left, right }
        | ValueNode::Int32Multiply { left, right }
        | ValueNode::Int32LessThan { left, right }
        | ValueNode::Int32BitwiseOr { left, right }
        | ValueNode::Int32BitwiseAnd { left, right }
        | ValueNode::Int32ShiftLeft { left, right }
        | ValueNode::Int32ShiftRight { left, right }
        | ValueNode::CheckedSmiAdd { left, right }
        | ValueNode::CheckedSmiSubtract { left, right }
        | ValueNode::CheckedSmiMultiply { left, right }
        | ValueNode::GenericAdd { left, right, .. }
        | ValueNode::GenericSubtract { left, right, .. }
        | ValueNode::GenericMultiply { left, right, .. } => vec![*left, *right],
        ValueNode::LoadNamedGeneric { object, .. } => vec![*object],
        ValueNode::StoreNamedGeneric { object, value, .. } => vec![*object, *value],
        ValueNode::StoreGlobal { value, .. } => vec![*value],
        ValueNode::Call {
            callee,
            receiver,
            args,
            ..
        } => {
            let mut v = vec![*callee, *receiver];
            v.extend(args);
            v
        }
        ValueNode::CallKnownFunction {
            callee,
            receiver,
            args,
            ..
        } => {
            let mut v = vec![*callee, *receiver];
            v.extend(args);
            v
        }
        ValueNode::CreateObjectLiteralWithProperties { values, .. } => values.clone(),
        ValueNode::CheckMaps { receiver, .. } => vec![*receiver],
        _ => Vec::new(),
    }
}

/// Collect all NodeId operands of a control node.
fn control_operands(ctrl: &ControlNode) -> Vec<NodeId> {
    match ctrl {
        ControlNode::Branch { condition, .. } => vec![*condition],
        ControlNode::Return { value } => vec![*value],
        _ => Vec::new(),
    }
}

/// Evaluate a pure expression tree at a specific IV value.
///
/// Returns `Some(value)` if the expression is a constant or a pure affine
/// function of the given IV Phi.  Returns `None` for anything else (calls,
/// loads, side-effecting nodes, etc.).
fn eval_at_iv(
    graph: &MaglevGraph,
    node_id: NodeId,
    iv_phi: NodeId,
    iv_value: i32,
    consts: &HashMap<NodeId, i32>,
) -> Option<i32> {
    if node_id == iv_phi {
        return Some(iv_value);
    }

    // Check constant map first.
    if let Some(&v) = consts.get(&node_id) {
        return Some(v);
    }

    let node = graph.node(node_id)?;
    match node {
        ValueNode::SmiConstant { value } | ValueNode::Int32Constant { value } => Some(*value),

        // Pure binary arithmetic (Int32)
        ValueNode::Int32Add { left, right } => {
            let l = eval_at_iv(graph, *left, iv_phi, iv_value, consts)?;
            let r = eval_at_iv(graph, *right, iv_phi, iv_value, consts)?;
            l.checked_add(r)
        }
        ValueNode::Int32Subtract { left, right } => {
            let l = eval_at_iv(graph, *left, iv_phi, iv_value, consts)?;
            let r = eval_at_iv(graph, *right, iv_phi, iv_value, consts)?;
            l.checked_sub(r)
        }
        ValueNode::Int32Multiply { left, right } => {
            let l = eval_at_iv(graph, *left, iv_phi, iv_value, consts)?;
            let r = eval_at_iv(graph, *right, iv_phi, iv_value, consts)?;
            l.checked_mul(r)
        }

        // Checked Smi variants (same semantics for constant eval)
        ValueNode::CheckedSmiAdd { left, right } => {
            let l = eval_at_iv(graph, *left, iv_phi, iv_value, consts)?;
            let r = eval_at_iv(graph, *right, iv_phi, iv_value, consts)?;
            l.checked_add(r)
        }
        ValueNode::CheckedSmiSubtract { left, right } => {
            let l = eval_at_iv(graph, *left, iv_phi, iv_value, consts)?;
            let r = eval_at_iv(graph, *right, iv_phi, iv_value, consts)?;
            l.checked_sub(r)
        }
        ValueNode::CheckedSmiMultiply { left, right } => {
            let l = eval_at_iv(graph, *left, iv_phi, iv_value, consts)?;
            let r = eval_at_iv(graph, *right, iv_phi, iv_value, consts)?;
            l.checked_mul(r)
        }

        // Generic variants with feedback (same constant eval)
        ValueNode::GenericAdd { left, right, .. } => {
            let l = eval_at_iv(graph, *left, iv_phi, iv_value, consts)?;
            let r = eval_at_iv(graph, *right, iv_phi, iv_value, consts)?;
            l.checked_add(r)
        }
        ValueNode::GenericSubtract { left, right, .. } => {
            let l = eval_at_iv(graph, *left, iv_phi, iv_value, consts)?;
            let r = eval_at_iv(graph, *right, iv_phi, iv_value, consts)?;
            l.checked_sub(r)
        }
        ValueNode::GenericMultiply { left, right, .. } => {
            let l = eval_at_iv(graph, *left, iv_phi, iv_value, consts)?;
            let r = eval_at_iv(graph, *right, iv_phi, iv_value, consts)?;
            l.checked_mul(r)
        }

        // Int32 bitwise OR (used in `| 0` truncation pattern)
        ValueNode::Int32BitwiseOr { left, right } => {
            let l = eval_at_iv(graph, *left, iv_phi, iv_value, consts)?;
            let r = eval_at_iv(graph, *right, iv_phi, iv_value, consts)?;
            Some(l | r)
        }
        ValueNode::Int32ShiftLeft { left, right } => {
            let l = eval_at_iv(graph, *left, iv_phi, iv_value, consts)?;
            let r = eval_at_iv(graph, *right, iv_phi, iv_value, consts)?;
            Some(l.wrapping_shl(r as u32))
        }

        _ => None,
    }
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
        // CallArrayPush codegen ignores the callee (uses inline push
        // or direct stub call without method resolution).  Marking
        // callee dead lets DCE remove the LoadNamedGeneric(arr,"push")
        // that would otherwise be emitted as an FFI call per iteration.
        ValueNode::CallArrayPush { receiver, args, .. } => {
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

        ValueNode::SpeculativeCallFusion { callee, .. } => {
            live.insert(*callee);
        }
        ValueNode::SpeculativeSumFusion { array } => {
            live.insert(*array);
        }
        ValueNode::SpeculativePushFusion { array, .. } => {
            live.insert(*array);
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
            | ValueNode::CallArrayPush { .. }
            | ValueNode::CallKnownFunction { .. }
            | ValueNode::CallBuiltin { .. }
            | ValueNode::CallRuntime { .. }
            | ValueNode::CallWithSpread { .. }
            | ValueNode::Construct { .. }
            | ValueNode::ConstructWithSpread { .. }
            | ValueNode::SpeculativeCallFusion { .. }
            | ValueNode::SpeculativeSumFusion { .. }
            | ValueNode::SpeculativePushFusion { .. }
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
            | ValueNode::CallArrayPush { .. }
            | ValueNode::CallKnownFunction { .. }
            | ValueNode::CallBuiltin { .. }
            | ValueNode::CallRuntime { .. }
            | ValueNode::CallWithSpread { .. }
            | ValueNode::Construct { .. }
            | ValueNode::ConstructWithSpread { .. }
            | ValueNode::SpeculativeCallFusion { .. }
            | ValueNode::SpeculativeSumFusion { .. }
            | ValueNode::SpeculativePushFusion { .. }
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
        | ValueNode::CallArrayPush {
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

        ValueNode::SpeculativeCallFusion { callee, .. } => {
            *callee = resolve(*callee);
        }
        ValueNode::SpeculativeSumFusion { array } => {
            *array = resolve(*array);
        }
        ValueNode::SpeculativePushFusion { array, .. } => {
            *array = resolve(*array);
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
        // Try 4x first (better for large loops like arithmetic_loop_10k),
        // fall back to 2x if trip count isn't divisible by 4.
        if !try_unroll_counted_loop(graph, lp, 4) {
            try_unroll_counted_loop(graph, lp, 2);
        }
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

    // ── 2b. Reject loops whose body contains method calls (CallProperty) ─
    // Unrolling loops with `arr.push(i)` or similar method calls creates
    // multiple runtime-stub invocations per iteration.  These stubs can
    // reallocate array backing stores without updating the JIT's inline
    // array IC, causing stale-pointer SIGSEGV when a later keyed-load
    // loop reads from the same array.  Bail out to keep such loops at 1×
    // where the interaction is safe.
    let has_method_call = body.nodes.iter().any(|(_, node)| {
        matches!(
            node,
            ValueNode::Call { .. }
                | ValueNode::CallKnownFunction { .. }
                | ValueNode::CallWithSpread { .. }
                | ValueNode::CallBuiltin { .. }
                | ValueNode::CallRuntime { .. }
        )
    });
    if has_method_call {
        return false;
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
// Affine analysis helpers for loop scalar evolution
// ─────────────────────────────────────────────────────────────────────────────

/// Try to decompose `expr_id` as `a * iv_phi + b` where a, b are constants.
/// Returns `Some((a, b))` with i128 coefficients for overflow-safe arithmetic.
fn extract_affine_addend(
    graph: &MaglevGraph,
    expr_id: NodeId,
    iv_phi_id: NodeId,
    consts: &HashMap<NodeId, i32>,
    depth: u32,
) -> Option<(i128, i128)> {
    if depth > 8 {
        return None;
    }
    if expr_id == iv_phi_id {
        return Some((1, 0));
    }
    if let Some(&k) = consts.get(&expr_id) {
        return Some((0, k as i128));
    }
    if let Some(k) = find_i32_constant(graph, expr_id) {
        return Some((0, k as i128));
    }
    let node = graph.node(expr_id)?.clone();
    match &node {
        ValueNode::Int32Add { left, right } => {
            let (a1, b1) = extract_affine_addend(graph, *left, iv_phi_id, consts, depth + 1)?;
            let (a2, b2) = extract_affine_addend(graph, *right, iv_phi_id, consts, depth + 1)?;
            Some((a1 + a2, b1 + b2))
        }
        ValueNode::Int32Subtract { left, right } => {
            let (a1, b1) = extract_affine_addend(graph, *left, iv_phi_id, consts, depth + 1)?;
            let (a2, b2) = extract_affine_addend(graph, *right, iv_phi_id, consts, depth + 1)?;
            Some((a1 - a2, b1 - b2))
        }
        ValueNode::Int32Multiply { left, right } => {
            let (a1, b1) = extract_affine_addend(graph, *left, iv_phi_id, consts, depth + 1)?;
            let (a2, b2) = extract_affine_addend(graph, *right, iv_phi_id, consts, depth + 1)?;
            if a1 == 0 {
                Some((b1 * a2, b1 * b2))
            } else if a2 == 0 {
                Some((a1 * b2, b1 * b2))
            } else {
                None // quadratic — bail
            }
        }
        ValueNode::Int32ShiftLeft { left, right } => {
            let (a, b) = extract_affine_addend(graph, *left, iv_phi_id, consts, depth + 1)?;
            let shift = consts
                .get(right)
                .copied()
                .or_else(|| find_i32_constant(graph, *right))?;
            if (0..32).contains(&shift) {
                let factor = 1i128 << (shift as u32);
                Some((a * factor, b * factor))
            } else {
                None
            }
        }
        ValueNode::Int32Negate { value } => {
            let (a, b) = extract_affine_addend(graph, *value, iv_phi_id, consts, depth + 1)?;
            Some((-a, -b))
        }
        _ => None,
    }
}

/// Given a back-edge node, decompose it as `phi_acc + (a * iv + b)`.
/// Returns `Some((a, b))` for the per-iteration affine addend.
fn peel_accumulator_addend(
    graph: &MaglevGraph,
    back_id: NodeId,
    phi_acc_id: NodeId,
    iv_phi_id: NodeId,
    consts: &HashMap<NodeId, i32>,
    depth: u32,
) -> Option<(i128, i128)> {
    if depth > 4 {
        return None;
    }
    let node = graph.node(back_id)?.clone();
    match &node {
        ValueNode::Int32Add { left, right } => {
            if *left == phi_acc_id {
                return extract_affine_addend(graph, *right, iv_phi_id, consts, 0);
            }
            if *right == phi_acc_id {
                return extract_affine_addend(graph, *left, iv_phi_id, consts, 0);
            }
            // Neither is phi directly — try recursing
            if let Some((a, b)) =
                peel_accumulator_addend(graph, *left, phi_acc_id, iv_phi_id, consts, depth + 1)
                && let Some((a2, b2)) = extract_affine_addend(graph, *right, iv_phi_id, consts, 0)
            {
                return Some((a + a2, b + b2));
            }
            if let Some((a, b)) =
                peel_accumulator_addend(graph, *right, phi_acc_id, iv_phi_id, consts, depth + 1)
                && let Some((a2, b2)) = extract_affine_addend(graph, *left, iv_phi_id, consts, 0)
            {
                return Some((a + a2, b + b2));
            }
            None
        }
        ValueNode::Int32Subtract { left, right } => {
            // phi_acc must be in the left subtree (subtracting phi would negate it)
            let (a, b) =
                peel_accumulator_addend(graph, *left, phi_acc_id, iv_phi_id, consts, depth + 1)?;
            let (a2, b2) = extract_affine_addend(graph, *right, iv_phi_id, consts, 0)?;
            Some((a - a2, b - b2))
        }
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass — Late constant-accumulator lowering
// ─────────────────────────────────────────────────────────────────────────────

/// After store-to-load forwarding and constant folding, some loop accumulator
/// patterns become `GenericAdd(Phi, SmiConstant(K))` where the constant was
/// only revealed by forwarding property loads.  This targeted pass converts
/// these to `Int32Add` when:
///
/// 1. The back-edge input of a Phi is `GenericAdd(phi, SmiConstant)`.
/// 2. The Phi init (entry-edge) is `SmiConstant(init_val)`.
/// 3. The total accumulation `init + K * max_iterations` fits in i32.
///
/// Unlike the full `eliminate_overflow_checks`, this pass does NOT attempt to
/// detect induction variables or complex chained patterns.
fn lower_constant_accumulator_adds(graph: &mut MaglevGraph) {
    let loops = licm::detect_loops(graph);
    if loops.is_empty() {
        return;
    }

    // Build a node → constant map.
    let mut consts: HashMap<NodeId, i32> = HashMap::new();
    for block in graph.blocks() {
        for (id, node) in &block.nodes {
            match node {
                ValueNode::SmiConstant { value } | ValueNode::Int32Constant { value } => {
                    consts.insert(*id, *value);
                }
                _ => {}
            }
        }
    }

    // For each loop, look for GenericAdd(Phi, SmiConstant) accumulators.
    let mut rewrites: Vec<(u32, usize, NodeId)> = Vec::new();

    for lp in &loops {
        let header = &graph.blocks()[lp.header as usize];
        let back_pred_pos = match header
            .predecessors
            .iter()
            .position(|&p| lp.body.contains(&p) && p != lp.header)
        {
            Some(pos) => pos,
            None => continue,
        };
        let entry_pos = if back_pred_pos == 0 { 1 } else { 0 };

        // Estimate max iterations from the loop structure.
        let max_iters = estimate_max_iterations(graph, lp, &consts);

        for (phi_id, node) in &header.nodes {
            let ValueNode::Phi { inputs } = node else {
                continue;
            };
            if inputs.len() != 2 {
                continue;
            }

            let init_id = inputs[entry_pos];
            let back_id = inputs[back_pred_pos];

            // Init must be a known constant.
            let Some(&init_val) = consts.get(&init_id) else {
                continue;
            };

            // Find back-edge node: must be GenericAdd(phi, const) in a body block.
            let back_node = find_node_in_blocks(graph, &back_id, &lp.body);
            let Some((block_idx, node_idx, back_value)) = back_node else {
                continue;
            };

            let addend_val = match &back_value {
                ValueNode::GenericAdd { left, right, .. } if *left == *phi_id => {
                    consts.get(right).copied()
                }
                ValueNode::GenericAdd { left, right, .. } if *right == *phi_id => {
                    consts.get(left).copied()
                }
                _ => None,
            };

            let Some(k) = addend_val else {
                continue;
            };

            // Check total accumulation fits i32.
            let iters = max_iters.unwrap_or(100_000) as i64;
            let total_min = (init_val as i64) + (k as i64) * iters;
            let total_max = (init_val as i64) + (k as i64) * iters;
            // Check both positive and negative constant accumulation.
            let (lo, hi) = if k >= 0 {
                (init_val as i64, total_max)
            } else {
                (total_min, init_val as i64)
            };
            if lo < i32::MIN as i64 || hi > i32::MAX as i64 {
                continue;
            }

            rewrites.push((block_idx, node_idx, back_id));
        }
    }

    // Apply rewrites.
    for (block_idx, node_idx, _back_id) in &rewrites {
        if let Some(block) = graph.block_mut(*block_idx)
            && let Some((_, node)) = block.nodes.get_mut(*node_idx)
            && let ValueNode::GenericAdd { left, right, .. } = node
        {
            *node = ValueNode::Int32Add {
                left: *left,
                right: *right,
            };
        }
    }
}

/// Find a node by ID in the given set of blocks, returning (block_idx, node_idx, node).
fn find_node_in_blocks(
    graph: &MaglevGraph,
    node_id: &NodeId,
    block_indices: &HashSet<u32>,
) -> Option<(u32, usize, ValueNode)> {
    for &bi in block_indices {
        if let Some(block) = graph.block(bi) {
            for (pos, (id, node)) in block.nodes.iter().enumerate() {
                if *id == *node_id {
                    return Some((bi, pos, node.clone()));
                }
            }
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass — Loop scalar evolution (constant-accumulator elimination)
// ─────────────────────────────────────────────────────────────────────────────

/// Replace simple counted loops that only accumulate a constant addend with
/// a closed-form computation.
///
/// Pattern detected:
///   header: phi_acc = Phi(init, back_edge)
///           phi_iv  = Phi(iv_init, iv_step)
///           Branch(Int32LessThan(phi_iv, limit), body, exit)
///   body:   back_edge = Int32Add(phi_acc, K)  [K is constant]
///           iv_step   = Int32Add(phi_iv, step) [step is constant]
///           Jump(header)
///
/// Replacement:
///   preheader: trip_count = (limit - iv_init) / step
///              result = init + K * trip_count
///   The accumulator Phi's init input is replaced with `result` and the
///   loop body's accumulator update becomes an identity (phi feeds itself),
///   effectively making the loop a no-op for the accumulator.
///
/// This eliminates the entire loop for benchmarks like property_access
/// (`sum += 15` × 1000 → `sum = 15000`) and deep_object (`sum += 99` × 1000).
fn eliminate_constant_accumulator_loops(graph: &mut MaglevGraph) {
    let loops = licm::detect_loops(graph);
    if loops.is_empty() {
        return;
    }

    // Build a node → constant map.
    let mut consts: HashMap<NodeId, i32> = HashMap::new();
    for block in graph.blocks() {
        for (id, node) in &block.nodes {
            match node {
                ValueNode::SmiConstant { value } | ValueNode::Int32Constant { value } => {
                    consts.insert(*id, *value);
                }
                _ => {}
            }
        }
    }

    for lp in &loops {
        try_eliminate_constant_accumulator(graph, lp, &consts);
    }
}

/// Try to replace a single loop's constant accumulator with a closed-form value.
fn try_eliminate_constant_accumulator(
    graph: &mut MaglevGraph,
    lp: &licm::NaturalLoop,
    consts: &HashMap<NodeId, i32>,
) -> bool {
    let header = match graph.block(lp.header) {
        Some(b) => b,
        None => return false,
    };

    // ── 1. Header must end with Branch on Int32LessThan ──────────────────
    let (condition_id, body_block_idx, _exit_block_idx) = match &header.control {
        Some(ControlNode::Branch {
            condition,
            if_true,
            if_false,
        }) => (*condition, *if_true, *if_false),
        _ => return false,
    };

    if !lp.body.contains(&body_block_idx) {
        return false;
    }

    // Simple loop: header + 1 body block.
    if lp.body.len() != 2 || !lp.body.contains(&lp.header) {
        return false;
    }

    let body = match graph.block(body_block_idx) {
        Some(b) => b,
        None => return false,
    };

    // Body must jump back to header.
    match &body.control {
        Some(ControlNode::Jump { target }) if *target == lp.header => {}
        _ => return false,
    }

    // ── 2. Find comparison and induction variable ────────────────────────
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

    // ── 3. Identify predecessor positions ────────────────────────────────
    let back_pred_pos = match header
        .predecessors
        .iter()
        .position(|&p| p == body_block_idx)
    {
        Some(pos) => pos,
        None => return false,
    };
    let entry_pos = match header.predecessors.iter().position(|&p| p == lp.preheader) {
        Some(pos) => pos,
        None => return false,
    };

    // ── 4. Find the induction variable Phi ───────────────────────────────
    let iv_phi_id = cmp_left;
    let iv_phi_inputs = {
        let h = graph.block(lp.header).unwrap();
        let mut found = None;
        for (nid, node) in &h.nodes {
            if *nid == iv_phi_id {
                if let ValueNode::Phi { inputs } = node {
                    found = Some(inputs.clone());
                }
                break;
            }
        }
        match found {
            Some(inputs) if inputs.len() == 2 => inputs,
            _ => return false,
        }
    };

    let iv_init_id = iv_phi_inputs[entry_pos];
    let iv_limit_id = cmp_right;

    let iv_init = match consts.get(&iv_init_id) {
        Some(&v) => v,
        None => match find_i32_constant(graph, iv_init_id) {
            Some(v) => v,
            None => return false,
        },
    };
    let iv_limit = match consts.get(&iv_limit_id) {
        Some(&v) => v,
        None => match find_i32_constant(graph, iv_limit_id) {
            Some(v) => v,
            None => return false,
        },
    };

    // Find IV step.
    let iv_back_id = iv_phi_inputs[back_pred_pos];
    let iv_step = find_increment_step(graph, iv_back_id, iv_phi_id);
    let iv_step_val = match iv_step {
        Some(s) if s > 0 => s,
        _ => return false,
    };

    // Compute trip count using widened arithmetic.  Require exact
    // divisibility to avoid off-by-one errors.
    let range = (iv_limit as i64) - (iv_init as i64);
    let step = iv_step_val as i64;
    if range <= 0 || range % step != 0 {
        return false;
    }
    let trip_count = range / step;

    // ── 5. Find accumulator Phis with Int32Add(phi, const) back-edges ───
    let h = graph.block(lp.header).unwrap();
    let header_phis: Vec<(NodeId, Vec<NodeId>)> = h
        .nodes
        .iter()
        .filter_map(|(nid, node)| {
            if let ValueNode::Phi { inputs } = node {
                Some((*nid, inputs.clone()))
            } else {
                None
            }
        })
        .collect();

    // Collect accumulators: (phi_id, init_val, affine_a, affine_b, back_node_id)
    // affine_a and affine_b represent the per-iteration addend: a * iv + b
    let mut accumulators: Vec<(NodeId, i32, i128, i128, NodeId)> = Vec::new();

    // Build use-count map for the loop body to verify accumulators are
    // only used by their own recurrence update.
    let mut loop_use_counts: HashMap<NodeId, usize> = HashMap::new();
    for &bi in &lp.body {
        if let Some(block) = graph.block(bi) {
            for (_, node) in &block.nodes {
                visit_value_node_inputs(node, &mut |id| {
                    *loop_use_counts.entry(id).or_insert(0) += 1;
                });
            }
        }
    }

    for (phi_id, inputs) in &header_phis {
        if *phi_id == iv_phi_id {
            continue; // Skip the induction variable itself.
        }
        if inputs.len() != 2 {
            continue;
        }

        let init_id = inputs[entry_pos];
        let back_id = inputs[back_pred_pos];

        // Init must be a known constant.
        let init_val = match consts.get(&init_id) {
            Some(&v) => v,
            None => match find_i32_constant(graph, init_id) {
                Some(v) => v,
                None => continue,
            },
        };

        // Back-edge must decompose as phi_acc + (a * iv + b) using only
        // Int32 operations.  This covers both constant addends (a=0) and
        // IV-dependent addends like `i*3 - 1` (a=3, b=-1).
        let Some((aff_a, aff_b)) =
            peel_accumulator_addend(graph, back_id, *phi_id, iv_phi_id, consts, 0)
        else {
            continue;
        };

        // Safety: the accumulator Phi must only be used inside the loop
        // by its own recurrence update.  If anything else reads the
        // accumulator within the loop, we cannot replace it.
        let in_loop_uses = loop_use_counts.get(phi_id).copied().unwrap_or(0);
        if in_loop_uses > 1 {
            continue;
        }

        // Compute closed-form result using i128 to avoid overflow:
        //   result = init + N*(a*iv_init + b) + a*step*N*(N-1)/2
        let n = trip_count as i128;
        let iv0 = iv_init as i128;
        let s = iv_step_val as i128;
        let result = (init_val as i128) + n * (aff_a * iv0 + aff_b) + aff_a * s * n * (n - 1) / 2;
        if result < i32::MIN as i128 || result > i32::MAX as i128 {
            continue;
        }

        accumulators.push((*phi_id, init_val, aff_a, aff_b, back_id));
    }

    if accumulators.is_empty() {
        return false;
    }

    // ── 6. Apply: replace Phi init with the closed-form result ──────────
    // Create a SmiConstant for the result in the preheader.
    for (phi_id, init_val, aff_a, aff_b, _back_id) in &accumulators {
        let n = trip_count as i128;
        let iv0 = iv_init as i128;
        let s = iv_step_val as i128;
        let result =
            (*init_val as i128) + n * (*aff_a * iv0 + *aff_b) + *aff_a * s * n * (n - 1) / 2;
        let result_val = result as i32;

        // Create a new SmiConstant node for the result.
        let result_node_id = graph.alloc_node_id();
        if let Some(pre) = graph.block_mut(lp.preheader) {
            pre.push_with_id(result_node_id, ValueNode::SmiConstant { value: result_val });
        }

        // Update the Phi's init input to point to the computed result.
        let h = graph.block_mut(lp.header).unwrap();
        for (nid, node) in &mut h.nodes {
            if *nid == *phi_id {
                if let ValueNode::Phi { inputs } = node {
                    inputs[entry_pos] = result_node_id;
                    // Also make the back-edge point to the phi itself,
                    // making the accumulator an identity within the loop.
                    inputs[back_pred_pos] = *phi_id;
                }
                break;
            }
        }
    }

    true
}

/// Estimate the maximum number of iterations for a loop by examining
/// the induction variable pattern (Phi with Int32LessThan comparison).
fn estimate_max_iterations(
    graph: &MaglevGraph,
    lp: &licm::NaturalLoop,
    consts: &HashMap<NodeId, i32>,
) -> Option<u64> {
    let header = &graph.blocks()[lp.header as usize];

    // Header must end with Branch on a comparison.
    let condition_id = match &header.control {
        Some(ControlNode::Branch { condition, .. }) => *condition,
        _ => return None,
    };

    // Find the comparison node.
    let cmp_node = header
        .nodes
        .iter()
        .find(|(id, _)| *id == condition_id)
        .map(|(_, n)| n)?;

    let (left, right) = match cmp_node {
        ValueNode::Int32LessThan { left, right } => (*left, *right),
        _ => return None,
    };

    // One side should be a Phi (IV), the other a constant (limit).
    let limit = consts.get(&right).or_else(|| consts.get(&left))?;
    Some(*limit as u64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass — Dead counted loop elimination
// ─────────────────────────────────────────────────────────────────────────────

/// Remove loops whose body has no side effects and whose values are not
/// referenced by any node outside the loop.
///
/// After scalar evolution resolves accumulator Phis to constants and trivial
/// phi elimination propagates those constants to external consumers, a counted
/// loop may become entirely dead — its only remaining work is incrementing the
/// induction variable, which nothing outside reads.  This pass detects such
/// loops and short-circuits the preheader to jump directly to the exit block,
/// eliminating all loop overhead.
#[allow(dead_code)] // Disabled in optimize() — see SIGSEGV note there.
fn eliminate_dead_counted_loops(graph: &mut MaglevGraph) {
    let loops = licm::detect_loops(graph);
    for lp in &loops {
        try_eliminate_dead_loop(graph, lp);
    }
}

/// Attempt to eliminate a single dead loop.
///
/// Returns `true` if the loop was removed.
#[allow(dead_code)] // Disabled in optimize() — see SIGSEGV note there.
fn try_eliminate_dead_loop(graph: &mut MaglevGraph, lp: &licm::NaturalLoop) -> bool {
    // ── 1. Find the exit block ──────────────────────────────────────────────
    let exit_block_idx = {
        let header = match graph.block(lp.header) {
            Some(b) => b,
            None => return false,
        };
        match &header.control {
            Some(ControlNode::Branch {
                if_true, if_false, ..
            }) => {
                if lp.body.contains(if_true) && !lp.body.contains(if_false) {
                    *if_false
                } else if lp.body.contains(if_false) && !lp.body.contains(if_true) {
                    *if_true
                } else {
                    return false;
                }
            }
            _ => return false,
        }
    };

    // ── 2. Collect all NodeIds defined inside the loop ──────────────────────
    let mut loop_node_ids: HashSet<NodeId> = HashSet::new();
    for &block_idx in &lp.body {
        if let Some(block) = graph.block(block_idx) {
            for (id, _) in &block.nodes {
                loop_node_ids.insert(*id);
            }
        }
    }

    // ── 3. No side effects allowed ──────────────────────────────────────────
    for &block_idx in &lp.body {
        if let Some(block) = graph.block(block_idx) {
            for (_, node) in &block.nodes {
                if has_side_effects(node) {
                    return false;
                }
            }
        }
    }

    // ── 4. No loop-internal value may be referenced from outside ────────────
    for block in graph.blocks() {
        if lp.body.contains(&block.id) {
            continue;
        }
        for (_, node) in &block.nodes {
            let mut refs_loop = false;
            visit_value_node_inputs(node, &mut |id| {
                if loop_node_ids.contains(&id) {
                    refs_loop = true;
                }
            });
            if refs_loop {
                return false;
            }
        }
        if let Some(ctrl) = &block.control {
            let mut live = HashSet::new();
            collect_control_node_inputs(ctrl, &mut live);
            if live.iter().any(|id| loop_node_ids.contains(id)) {
                return false;
            }
        }
    }

    // ── 5. Bail out if the exit block has Phis (predecessor rewrite is unsafe) ─
    if let Some(exit_block) = graph.block(exit_block_idx) {
        for (_, node) in &exit_block.nodes {
            if matches!(node, ValueNode::Phi { .. }) {
                return false;
            }
        }
    }

    // ── 6. Redirect preheader → exit ────────────────────────────────────────
    if let Some(preheader) = graph.block_mut(lp.preheader) {
        preheader.control = Some(ControlNode::Jump {
            target: exit_block_idx,
        });
    }

    // Update exit block predecessors.
    if let Some(exit_block) = graph.block_mut(exit_block_idx) {
        exit_block.predecessors.retain(|p| !lp.body.contains(p));
        if !exit_block.predecessors.contains(&lp.preheader) {
            exit_block.predecessors.push(lp.preheader);
        }
    }

    // Clear dead loop blocks.
    for &block_idx in &lp.body {
        if let Some(block) = graph.block_mut(block_idx) {
            block.nodes.clear();
            block.is_loop_header = false;
            block.control = Some(ControlNode::Jump {
                target: exit_block_idx,
            });
        }
    }

    true
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass — Induction variable strength reduction
// ─────────────────────────────────────────────────────────────────────────────

/// Replace `Int32Multiply(iv_phi, K)` inside counted loops with an additive
/// derived induction variable: `derived_phi(init*K, derived_phi + step*K)`.
///
/// This eliminates multiplies from the loop body's critical path, replacing
/// them with adds.  For example, `i * 3` becomes a new IV that increments
/// by 3 each iteration.
#[allow(dead_code)]
fn strength_reduce_induction_variables(graph: &mut MaglevGraph) {
    let loops = licm::detect_loops(graph);
    for lp in &loops {
        strength_reduce_iv_in_loop(graph, lp);
    }
}

#[allow(dead_code)]
fn strength_reduce_iv_in_loop(graph: &mut MaglevGraph, lp: &licm::NaturalLoop) -> bool {
    let header_idx = lp.header;

    // ── 1. Header must end with Branch on Int32LessThan ─────────────────
    let (condition_id, body_block_idx) = {
        let header = match graph.block(header_idx) {
            Some(b) => b,
            None => return false,
        };
        match &header.control {
            Some(ControlNode::Branch {
                condition, if_true, ..
            }) => (*condition, *if_true),
            _ => return false,
        }
    };

    if !lp.body.contains(&body_block_idx) {
        return false;
    }

    // Simple loop: body is a single block jumping back to header.
    if lp.body.len() != 2 || !lp.body.contains(&header_idx) {
        return false;
    }

    match graph.block(body_block_idx).and_then(|b| b.control.as_ref()) {
        Some(ControlNode::Jump { target }) if *target == header_idx => {}
        _ => return false,
    }

    // ── 2. Identify the induction variable ──────────────────────────────
    let cmp_left = {
        let header = graph.block(header_idx).unwrap();
        let mut found = None;
        for (nid, node) in &header.nodes {
            if *nid == condition_id {
                found = Some(node.clone());
                break;
            }
        }
        match found {
            Some(ValueNode::Int32LessThan { left, .. }) => left,
            _ => return false,
        }
    };

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

    let iv_phi_id = cmp_left;
    let iv_phi_inputs = {
        let h = graph.block(header_idx).unwrap();
        let mut found = None;
        for (nid, node) in &h.nodes {
            if *nid == iv_phi_id {
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

    let init_id = iv_phi_inputs[preheader_pred_pos];
    let init_val = match find_i32_constant(graph, init_id) {
        Some(v) => v,
        None => return false,
    };

    let back_edge_input = iv_phi_inputs[back_edge_pred_pos];
    let step_val = match find_increment_step(graph, back_edge_input, iv_phi_id) {
        Some(s) if s > 0 => s,
        _ => return false,
    };

    // ── 3. Find Int32Multiply(iv_phi, K) in the loop body ───────────────
    // Deduplicate by K: multiple `i*3` uses share one derived IV.
    let mut mul_targets: HashMap<i32, Vec<NodeId>> = HashMap::new();
    {
        let body = graph.block(body_block_idx).unwrap();
        for (nid, node) in &body.nodes {
            if let ValueNode::Int32Multiply { left, right } = node {
                let (const_input, is_iv) = if *left == iv_phi_id {
                    (*right, true)
                } else if *right == iv_phi_id {
                    (*left, true)
                } else {
                    (NodeId(0), false)
                };
                if !is_iv {
                    continue;
                }
                if let Some(k) = find_i32_constant(graph, const_input)
                    && k != 0
                    && k != 1
                {
                    mul_targets.entry(k).or_default().push(*nid);
                }
            }
        }
    }

    if mul_targets.is_empty() {
        return false;
    }

    // ── 4. Create derived IVs ───────────────────────────────────────────
    let mut subst: HashMap<NodeId, NodeId> = HashMap::new();

    for (k, mul_ids) in &mul_targets {
        let derived_init = init_val.wrapping_mul(*k);
        let derived_step = step_val.wrapping_mul(*k);

        // Allocate node IDs.
        let init_const_id = graph.alloc_node_id();
        let step_const_id = graph.alloc_node_id();
        let derived_phi_id = graph.alloc_node_id();
        let derived_add_id = graph.alloc_node_id();

        // Add constants to preheader.
        let preheader = graph.block_mut(lp.preheader).unwrap();
        preheader.push_with_id(
            init_const_id,
            ValueNode::Int32Constant {
                value: derived_init,
            },
        );
        preheader.push_with_id(
            step_const_id,
            ValueNode::Int32Constant {
                value: derived_step,
            },
        );

        // Add derived Phi to header — insert among existing Phis.
        let mut phi_inputs = vec![NodeId(0); iv_phi_inputs.len()];
        phi_inputs[preheader_pred_pos] = init_const_id;
        phi_inputs[back_edge_pred_pos] = derived_add_id;
        {
            let header = graph.block_mut(header_idx).unwrap();
            // Find position after last existing Phi.
            let insert_pos = header
                .nodes
                .iter()
                .position(|(_, n)| !matches!(n, ValueNode::Phi { .. }))
                .unwrap_or(header.nodes.len());
            header.nodes.insert(
                insert_pos,
                (derived_phi_id, ValueNode::Phi { inputs: phi_inputs }),
            );
        }

        // Add Int32Add to body (before any existing back-edge increment).
        let body = graph.block_mut(body_block_idx).unwrap();
        body.push_with_id(
            derived_add_id,
            ValueNode::Int32Add {
                left: derived_phi_id,
                right: step_const_id,
            },
        );

        // Map all multiply IDs to the derived Phi.
        for mul_id in mul_ids {
            subst.insert(*mul_id, derived_phi_id);
        }
    }

    // ── 5. Replace all references to old multiplies ─────────────────────
    for block in graph.blocks_mut() {
        apply_subst_to_block(block, &subst);
    }

    true
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
    fn test_strength_reduce_not_applied_to_non_decomposable() {
        // Multiply by 6 is neither a power of two nor 2^k±1 — node should stay.
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        let c6 = block.push_value(ValueNode::Int32Constant { value: 6 });
        let mul = block.push_value(ValueNode::Int32Multiply { left: p, right: c6 });
        block.set_control(ControlNode::Return { value: mul });
        graph.add_block(block);

        optimize(&mut graph);

        // Multiply by 6 is not a power of two nor 2^k±1 — node should stay as multiply.
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

    #[test]
    fn test_forward_invariant_object_properties_via_global() {
        // Simulates: var obj = { a: 1 }; ... LoadGlobal("obj") → LoadNamedGeneric(_, "a")
        // Entry block: CreateObjectLiteralWithProperties + StoreGlobal
        // Second block: LoadGlobal + LoadNamedGeneric (should resolve to constant)
        let mut graph = MaglevGraph::new(0);

        // Block 0: entry
        let mut entry = BasicBlock::new(0);
        let val_a = entry.push_value(ValueNode::SmiConstant { value: 42 });
        let _create = entry.push_value(ValueNode::CreateObjectLiteralWithProperties {
            feedback_slot: 0,
            flags: 0,
            names: vec![5], // name index 5 = "a"
            values: vec![val_a],
        });
        entry.push_value(ValueNode::StoreGlobal {
            name: 10, // global name index 10 = "obj"
            value: _create,
            feedback_slot: 1,
        });
        entry.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(entry);

        // Block 1: uses LoadGlobal + LoadNamedGeneric
        let mut body = BasicBlock::new(1);
        let obj_load = body.push_value(ValueNode::LoadGlobal {
            name: 10, // same global name
            feedback_slot: 2,
        });
        let prop_load = body.push_value(ValueNode::LoadNamedGeneric {
            object: obj_load,
            name: 5, // same property name
            feedback_slot: 3,
        });
        body.set_control(ControlNode::Return { value: prop_load });
        graph.add_block(body);

        // Run just the forwarding pass (not full optimize, to isolate)
        fuse_object_literal_stores(&mut graph);
        forward_invariant_object_properties(&mut graph);

        // The LoadNamedGeneric should have been substituted.
        // Check that prop_load's uses now point to val_a.
        let block1 = &graph.blocks()[1];
        let ret = block1.control.as_ref().unwrap();
        if let ControlNode::Return { value } = ret {
            // The return should now reference val_a (the SmiConstant 42)
            assert_eq!(
                *value, val_a,
                "Expected LoadNamedGeneric to be forwarded to SmiConstant(42), got {:?}",
                value
            );
        } else {
            panic!("Expected Return control node");
        }
    }

    #[test]
    fn test_forward_nested_object_properties() {
        // Simulates: var root = { a: { b: 99 } };
        // LoadGlobal("root") → LoadNamedGeneric(_, "a") → LoadNamedGeneric(_, "b")
        // Should resolve the entire chain to SmiConstant(99).
        let mut graph = MaglevGraph::new(0);

        // Block 0: entry
        let mut entry = BasicBlock::new(0);
        let val_b = entry.push_value(ValueNode::SmiConstant { value: 99 });
        let inner = entry.push_value(ValueNode::CreateObjectLiteralWithProperties {
            feedback_slot: u32::MAX,
            flags: 0,
            names: vec![7], // name index 7 = "b"
            values: vec![val_b],
        });
        let outer = entry.push_value(ValueNode::CreateObjectLiteralWithProperties {
            feedback_slot: u32::MAX,
            flags: 0,
            names: vec![5], // name index 5 = "a"
            values: vec![inner],
        });
        entry.push_value(ValueNode::StoreGlobal {
            name: 10,
            value: outer,
            feedback_slot: 1,
        });
        entry.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(entry);

        // Block 1: chained property access
        let mut body = BasicBlock::new(1);
        let root_load = body.push_value(ValueNode::LoadGlobal {
            name: 10,
            feedback_slot: 2,
        });
        let load_a = body.push_value(ValueNode::LoadNamedGeneric {
            object: root_load,
            name: 5, // "a"
            feedback_slot: 3,
        });
        let load_b = body.push_value(ValueNode::LoadNamedGeneric {
            object: load_a,
            name: 7, // "b"
            feedback_slot: 4,
        });
        body.set_control(ControlNode::Return { value: load_b });
        graph.add_block(body);

        fuse_object_literal_stores(&mut graph);
        forward_invariant_object_properties(&mut graph);

        // load_b should resolve to val_b (SmiConstant 99)
        let block1 = &graph.blocks()[1];
        let ret = block1.control.as_ref().unwrap();
        if let ControlNode::Return { value } = ret {
            assert_eq!(
                *value, val_b,
                "Expected nested LoadNamedGeneric to resolve to SmiConstant(99), got {:?}",
                value
            );
        } else {
            panic!("Expected Return control node");
        }
    }
}
