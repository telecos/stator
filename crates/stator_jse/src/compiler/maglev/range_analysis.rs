//! Integer range analysis for overflow-check elimination.
//!
//! This pass computes a conservative [`Range`] — a `[min, max]` interval —
//! for every [`NodeId`] in the graph using `i64` arithmetic to detect
//! potential 32-bit overflow.  When the range of both operands to a
//! **checked** Smi operation proves that the result cannot overflow `i32`,
//! the node is replaced with the cheaper **unchecked** `Int32*` variant,
//! eliminating the deoptimisation bailout.
//!
//! # Algorithm
//!
//! 1. **Seed** (Phase 0) — assign exact ranges to constant nodes
//!    (`SmiConstant`, `Int32Constant`).  Parameters and other dynamic nodes
//!    receive the full `[i32::MIN, i32::MAX]` interval.
//! 2. **Loop induction** (Phase 1) — detect `for (i = init; i < bound; i++)`
//!    patterns and rewrite the checked step node to its unchecked variant
//!    when both the Phi's full range and the step's body range fit `i32`.
//! 3. **Propagate** (Phase 2) — walk each block linearly and, for every
//!    arithmetic node whose inputs already have ranges, compute the output
//!    range via interval arithmetic and rewrite checked variants.
//!
//! # Limitations
//!
//! - Only a single forward pass is performed (no fixed-point iteration).
//! - Phi nodes not involved in a recognised induction variable pattern
//!   receive the union of their input ranges when all inputs have known
//!   ranges; otherwise they are skipped.
//! - Only positive step deltas are handled for loop induction.
//! - Only `i32` ranges are tracked; `f64` and `u32` are left untouched.
//!
//! # Usage
//!
//! ```
//! use stator_jse::compiler::maglev::ir::{
//!     BasicBlock, ControlNode, MaglevGraph, ValueNode,
//! };
//! use stator_jse::compiler::maglev::range_analysis::eliminate_overflow_checks;
//!
//! let mut graph = MaglevGraph::new(0);
//! let mut block = BasicBlock::new(0);
//! let c1 = block.push_value(ValueNode::SmiConstant { value: 10 });
//! let c2 = block.push_value(ValueNode::SmiConstant { value: 20 });
//! let add = block.push_value(ValueNode::CheckedSmiAdd {
//!     left: c1,
//!     right: c2,
//! });
//! block.set_control(ControlNode::Return { value: add });
//! graph.add_block(block);
//!
//! eliminate_overflow_checks(&mut graph);
//!
//! // The CheckedSmiAdd has been replaced with an unchecked Int32Add.
//! assert!(matches!(
//!     &graph.blocks()[0].nodes[2].1,
//!     ValueNode::Int32Add { .. }
//! ));
//! ```

use std::collections::HashMap;

use crate::compiler::maglev::ir::{ControlNode, MaglevGraph, NodeId, ValueNode};

// ─────────────────────────────────────────────────────────────────────────────
// Range type
// ─────────────────────────────────────────────────────────────────────────────

/// A conservative integer interval `[min, max]` using `i64` to detect 32-bit
/// overflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Range {
    /// Lower bound (inclusive).
    pub min: i64,
    /// Upper bound (inclusive).
    pub max: i64,
}

impl Range {
    /// The full 32-bit signed range.
    pub const I32_FULL: Self = Self {
        min: i32::MIN as i64,
        max: i32::MAX as i64,
    };

    /// An exact constant range.
    pub const fn exact(v: i64) -> Self {
        Self { min: v, max: v }
    }

    /// Returns `true` when every value in this range fits in a signed 32-bit
    /// integer.
    pub fn fits_i32(self) -> bool {
        self.min >= i32::MIN as i64 && self.max <= i32::MAX as i64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry-point
// ─────────────────────────────────────────────────────────────────────────────

/// Run range analysis over every block and replace checked Smi operations
/// whose output provably fits `i32` with unchecked `Int32*` equivalents.
///
/// The analysis is split into three phases:
///
/// 1. **Phase 0 — Seed**: assign exact ranges to constants and full-i32 ranges
///    to parameters so that loop-bound ranges are available before Phase 1.
/// 2. **Phase 1 — Loop induction**: detect `for (i = init; i < bound; i++)`
///    patterns and directly rewrite the step node when both the Phi's full
///    range and the step's body range provably fit `i32`.
/// 3. **Phase 2 — Forward propagation**: a single forward pass that
///    propagates ranges through arithmetic and rewrites remaining checked
///    operations.
pub fn eliminate_overflow_checks(graph: &mut MaglevGraph) {
    let mut ranges: HashMap<NodeId, Range> = HashMap::new();

    // Phase 0 — seed constants and parameters.
    for block in graph.blocks() {
        for (id, node) in &block.nodes {
            if let Some(r) = seed_range(node) {
                ranges.insert(*id, r);
            }
        }
    }

    // Phase 0.5 — seed CheckSmi receivers with I32_FULL.
    //
    // A `CheckSmi` guard guarantees its receiver is a valid Smi (i32) at
    // runtime (deoptimising otherwise).  This lets us assign the full i32
    // range to the guarded value, unlocking downstream propagation for
    // nodes like `LoadGlobal` that have no inherent range.
    for block in graph.blocks() {
        for (_id, node) in &block.nodes {
            if let ValueNode::CheckSmi { receiver } = node {
                ranges.entry(*receiver).or_insert(Range::I32_FULL);
            }
        }
    }

    // Phase 1 — detect loop induction variables and rewrite step nodes.
    rewrite_loop_induction_steps(graph, &mut ranges);

    // Phase 2 — forward propagation and rewriting.
    let mut rewrites = 0u32;
    for block in graph.blocks_mut() {
        for (id, node) in &mut block.nodes {
            // Seed any nodes missed in Phase 0 (shouldn't happen, but
            // defensive).
            if !ranges.contains_key(id)
                && let Some(r) = seed_range(node)
            {
                ranges.insert(*id, r);
            }

            if let Some((out_range, replacement)) = try_rewrite(node, *id, &ranges) {
                ranges.insert(*id, out_range);
                if let Some(new_node) = replacement {
                    *node = new_node;
                    rewrites += 1;
                }
            }
        }
    }
    let _ = rewrites;
}

// ─────────────────────────────────────────────────────────────────────────────
// Range seeding
// ─────────────────────────────────────────────────────────────────────────────

/// Return a range for constant / parameter nodes.
fn seed_range(node: &ValueNode) -> Option<Range> {
    match node {
        ValueNode::SmiConstant { value } | ValueNode::Int32Constant { value } => {
            Some(Range::exact(*value as i64))
        }
        ValueNode::Uint32Constant { value } => Some(Range::exact(*value as i64)),
        // Parameters and LoadGlobal have the full i32 range.
        // LoadGlobal returns a runtime Smi which fits i32; if the global
        // isn't a Smi, any downstream CheckedSmi operation will deopt.
        ValueNode::Parameter { .. } | ValueNode::LoadGlobal { .. } => Some(Range::I32_FULL),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Range propagation & rewriting
// ─────────────────────────────────────────────────────────────────────────────

/// For a node, compute the output range and — if the node is a checked
/// variant that can be lowered — return the replacement node.
///
/// Returns `None` if the node is not an arithmetic node we track.
fn try_rewrite(
    node: &ValueNode,
    node_id: NodeId,
    ranges: &HashMap<NodeId, Range>,
) -> Option<(Range, Option<ValueNode>)> {
    match node {
        // ── Phi — propagate as union of input ranges ─────────────────────
        // If Phase 1 already assigned a range (induction / accumulator),
        // keep it — it is more precise.
        ValueNode::Phi { inputs } => {
            if ranges.contains_key(&node_id) {
                return None;
            }
            let mut min = i64::MAX;
            let mut max = i64::MIN;
            let mut any = false;
            for input in inputs {
                if *input == node_id {
                    continue;
                }
                let r = ranges.get(input)?;
                min = min.min(r.min);
                max = max.max(r.max);
                any = true;
            }
            if !any {
                return None;
            }
            Some((Range { min, max }, None))
        }

        // ── CheckedSmiAdd ────────────────────────────────────────────────
        ValueNode::CheckedSmiAdd { left, right } => {
            let (lr, rr) = (ranges.get(left)?, ranges.get(right)?);
            let out = Range {
                min: lr.min + rr.min,
                max: lr.max + rr.max,
            };
            let repl = if out.fits_i32() {
                Some(ValueNode::Int32Add {
                    left: *left,
                    right: *right,
                })
            } else {
                None
            };
            Some((out, repl))
        }

        // ── CheckedSmiSubtract ───────────────────────────────────────────
        ValueNode::CheckedSmiSubtract { left, right } => {
            let (lr, rr) = (ranges.get(left)?, ranges.get(right)?);
            let out = Range {
                min: lr.min - rr.max,
                max: lr.max - rr.min,
            };
            let repl = if out.fits_i32() {
                Some(ValueNode::Int32Subtract {
                    left: *left,
                    right: *right,
                })
            } else {
                None
            };
            Some((out, repl))
        }

        // ── CheckedSmiMultiply ───────────────────────────────────────────
        ValueNode::CheckedSmiMultiply { left, right } => {
            let (lr, rr) = (ranges.get(left)?, ranges.get(right)?);
            let candidates = [
                lr.min * rr.min,
                lr.min * rr.max,
                lr.max * rr.min,
                lr.max * rr.max,
            ];
            let out = Range {
                min: candidates.iter().copied().min().unwrap(),
                max: candidates.iter().copied().max().unwrap(),
            };
            let repl = if out.fits_i32() {
                Some(ValueNode::Int32Multiply {
                    left: *left,
                    right: *right,
                })
            } else {
                None
            };
            Some((out, repl))
        }

        // ── CheckedSmiIncrement ──────────────────────────────────────────
        ValueNode::CheckedSmiIncrement { value } => {
            let vr = ranges.get(value)?;
            let out = Range {
                min: vr.min + 1,
                max: vr.max + 1,
            };
            let repl = if out.fits_i32() {
                Some(ValueNode::Int32Increment { value: *value })
            } else {
                None
            };
            Some((out, repl))
        }

        // ── CheckedSmiDecrement ──────────────────────────────────────────
        ValueNode::CheckedSmiDecrement { value } => {
            let vr = ranges.get(value)?;
            let out = Range {
                min: vr.min - 1,
                max: vr.max - 1,
            };
            let repl = if out.fits_i32() {
                Some(ValueNode::Int32Decrement { value: *value })
            } else {
                None
            };
            Some((out, repl))
        }

        // ── Unchecked Int32 ops — propagate ranges but no rewrite ────────
        ValueNode::Int32Add { left, right } => {
            let (lr, rr) = (ranges.get(left)?, ranges.get(right)?);
            Some((
                Range {
                    min: lr.min + rr.min,
                    max: lr.max + rr.max,
                },
                None,
            ))
        }
        ValueNode::Int32Subtract { left, right } => {
            let (lr, rr) = (ranges.get(left)?, ranges.get(right)?);
            Some((
                Range {
                    min: lr.min - rr.max,
                    max: lr.max - rr.min,
                },
                None,
            ))
        }

        // ── Generic arithmetic — lower to Int32 when range fits ──────────
        ValueNode::GenericAdd { left, right, .. } => {
            let (lr, rr) = (ranges.get(left)?, ranges.get(right)?);
            let out = Range {
                min: lr.min + rr.min,
                max: lr.max + rr.max,
            };
            let repl = if out.fits_i32() {
                Some(ValueNode::Int32Add {
                    left: *left,
                    right: *right,
                })
            } else {
                None
            };
            Some((out, repl))
        }
        ValueNode::GenericSubtract { left, right, .. } => {
            let (lr, rr) = (ranges.get(left)?, ranges.get(right)?);
            let out = Range {
                min: lr.min - rr.max,
                max: lr.max - rr.min,
            };
            let repl = if out.fits_i32() {
                Some(ValueNode::Int32Subtract {
                    left: *left,
                    right: *right,
                })
            } else {
                None
            };
            Some((out, repl))
        }
        ValueNode::GenericMultiply { left, right, .. } => {
            let (lr, rr) = (ranges.get(left)?, ranges.get(right)?);
            let candidates = [
                lr.min * rr.min,
                lr.min * rr.max,
                lr.max * rr.min,
                lr.max * rr.max,
            ];
            let out = Range {
                min: candidates.iter().copied().min().unwrap(),
                max: candidates.iter().copied().max().unwrap(),
            };
            let repl = if out.fits_i32() {
                Some(ValueNode::Int32Multiply {
                    left: *left,
                    right: *right,
                })
            } else {
                None
            };
            Some((out, repl))
        }
        ValueNode::GenericIncrement { value, .. } => {
            let vr = ranges.get(value)?;
            let out = Range {
                min: vr.min + 1,
                max: vr.max + 1,
            };
            let repl = if out.fits_i32() {
                Some(ValueNode::Int32Increment { value: *value })
            } else {
                None
            };
            Some((out, repl))
        }
        ValueNode::GenericDecrement { value, .. } => {
            let vr = ranges.get(value)?;
            let out = Range {
                min: vr.min - 1,
                max: vr.max - 1,
            };
            let repl = if out.fits_i32() {
                Some(ValueNode::Int32Decrement { value: *value })
            } else {
                None
            };
            Some((out, repl))
        }

        // ── Bitwise OR — truncates to i32 ───────────────────────────────
        // `x | y` always calls `ToInt32` per spec; result fits i32.
        // Propagating the range enables downstream Phi nodes (whose
        // back-edge goes through `| 0`) to get a range, unlocking
        // lowering of checked arithmetic on accumulators.
        ValueNode::GenericBitwiseOr { left, right, .. }
        | ValueNode::Int32BitwiseOr { left, right } => {
            // Conservative: result is full i32 (bitwise OR can set any bit).
            let _lr = ranges.get(left);
            let _rr = ranges.get(right);
            // We don't try to narrow the range; just declare i32.
            Some((Range::I32_FULL, None))
        }

        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1 — Loop induction variable detection
// ─────────────────────────────────────────────────────────────────────────────

/// The upper bound extracted from a loop header comparison.
#[derive(Debug)]
enum LoopBound {
    /// `phi < bound` — body executes for phi in `[init, bound - 1]`.
    LessThan(Range),
    /// `phi <= bound` — body executes for phi in `[init, bound]`.
    LessThanOrEqual(Range),
}

/// Per-loop metadata collected during induction variable analysis, used by
/// the subsequent accumulator analysis.
struct LoopMeta {
    header_idx: u32,
    back_pred_pos: usize,
    entry_pos: usize,
    max_iterations: i64,
    induction_phi_ids: Vec<NodeId>,
}

/// Scan the graph for loop induction variable patterns and, where safe,
/// rewrite checked step nodes to unchecked variants.
///
/// Pattern: a Phi with two inputs (entry value, back-edge step), where the
/// step is `CheckedSmiIncrement { value: phi }` and the loop header has a
/// comparison `phi < bound` or `phi <= bound`.
///
/// Unlike the old implementation, this iterates over **all** Phi nodes in
/// each loop header so that multiple induction variables are recognised
/// even when the first Phi in program order is not an induction variable
/// (e.g. an accumulator Phi such as `sum`).
///
/// After lowering induction variable step nodes, a second sub-pass detects
/// additive *accumulator* Phis (e.g. `sum += i`) and rewrites their
/// `CheckedSmiAdd` back-edge when the total accumulation fits `i32`.
fn rewrite_loop_induction_steps(graph: &mut MaglevGraph, ranges: &mut HashMap<NodeId, Range>) {
    // Build a node → (block_idx, value_node) map for fast lookup.
    let node_map: HashMap<NodeId, (u32, ValueNode)> = graph
        .blocks()
        .iter()
        .flat_map(|b| b.nodes.iter().map(|(id, n)| (*id, (b.id, n.clone()))))
        .collect();

    // Detect back-edges: block B with Jump { target: H } where H <= B.
    let back_edges: Vec<(u32, u32)> = graph
        .blocks()
        .iter()
        .filter_map(|b| {
            if let Some(ControlNode::Jump { target }) = &b.control
                && *target <= b.id
            {
                Some((b.id, *target))
            } else {
                None
            }
        })
        .collect();

    // ── Phase 1a: detect induction variable Phis ─────────────────────────
    let mut step_rewrites: Vec<(u32, usize, ValueNode, NodeId, Range)> = Vec::new();
    let mut phi_ranges: Vec<(NodeId, Range)> = Vec::new();
    let mut loop_metas: Vec<LoopMeta> = Vec::new();

    for &(back_src, header_idx) in &back_edges {
        let header = &graph.blocks()[header_idx as usize];

        let Some(back_pred_pos) = header.predecessors.iter().position(|&p| p == back_src) else {
            continue;
        };
        let entry_pos = if back_pred_pos == 0 { 1 } else { 0 };

        // Collect all Phis with exactly 2 inputs.
        let phis: Vec<(NodeId, Vec<NodeId>)> = header
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                if let ValueNode::Phi { inputs } = node
                    && inputs.len() == 2
                {
                    Some((*id, inputs.clone()))
                } else {
                    None
                }
            })
            .collect();

        let mut induction_phi_ids = Vec::new();
        let mut max_iterations: Option<i64> = None;

        // Try each Phi as a potential induction variable.
        for (phi_id, phi_inputs) in &phis {
            let init_id = phi_inputs[entry_pos];
            let step_id = phi_inputs[back_pred_pos];

            let Some(init_range) = ranges.get(&init_id) else {
                continue;
            };
            let Some(delta) = find_step_delta(&step_id, phi_id, &node_map, ranges) else {
                continue;
            };
            let Some(bound) = find_comparison_bound(header, phi_id, &node_map, ranges) else {
                continue;
            };
            let Some((phi_range, step_range)) = compute_induction_ranges(init_range, delta, &bound)
            else {
                continue;
            };

            if !phi_range.fits_i32() || !step_range.fits_i32() {
                continue;
            }

            // Find the step node's position in its block.
            let Some((step_block_idx, step_node)) = node_map.get(&step_id) else {
                continue;
            };
            let step_block = &graph.blocks()[*step_block_idx as usize];
            let Some(node_idx) = step_block.nodes.iter().position(|(id, _)| *id == step_id) else {
                continue;
            };
            let Some(replacement) = lower_checked_to_unchecked(step_node) else {
                continue;
            };

            step_rewrites.push((*step_block_idx, node_idx, replacement, step_id, step_range));
            phi_ranges.push((*phi_id, phi_range));
            induction_phi_ids.push(*phi_id);

            // Compute max iterations for the accumulator analysis.
            let iters = match &bound {
                LoopBound::LessThan(br) => br.max - init_range.min,
                LoopBound::LessThanOrEqual(br) => br.max - init_range.min + 1,
            };
            max_iterations = Some(max_iterations.map_or(iters, |prev: i64| prev.max(iters)));
        }

        if let Some(max_iters) = max_iterations {
            loop_metas.push(LoopMeta {
                header_idx,
                back_pred_pos,
                entry_pos,
                max_iterations: max_iters,
                induction_phi_ids,
            });
        }
    }

    // Apply step node rewrites and record their ranges.
    for (block_idx, node_idx, new_node, step_id, step_range) in step_rewrites {
        if let Some(block) = graph.block_mut(block_idx) {
            block.nodes[node_idx].1 = new_node;
        }
        ranges.insert(step_id, step_range);
    }
    // Record induction-variable Phi ranges for downstream use.
    for (phi_id, phi_range) in phi_ranges {
        ranges.insert(phi_id, phi_range);
    }

    // ── Phase 1b: detect additive accumulator Phis ───────────────────────
    rewrite_accumulator_phis(graph, ranges, &loop_metas, &node_map);
}

/// Determine the step delta for an induction variable.
///
/// Supports `CheckedSmiIncrement { value: phi }` (delta = 1) and
/// `CheckedSmiAdd { left: phi, right: const }` (delta = const value).
fn find_step_delta(
    step_id: &NodeId,
    phi_id: &NodeId,
    node_map: &HashMap<NodeId, (u32, ValueNode)>,
    ranges: &HashMap<NodeId, Range>,
) -> Option<i64> {
    let (_, step_node) = node_map.get(step_id)?;
    match step_node {
        ValueNode::CheckedSmiIncrement { value } | ValueNode::GenericIncrement { value, .. }
            if value == phi_id =>
        {
            Some(1)
        }
        ValueNode::CheckedSmiDecrement { value } | ValueNode::GenericDecrement { value, .. }
            if value == phi_id =>
        {
            Some(-1)
        }
        ValueNode::CheckedSmiAdd { left, right } | ValueNode::GenericAdd { left, right, .. }
            if left == phi_id =>
        {
            let r = ranges.get(right)?;
            if r.min == r.max { Some(r.min) } else { None }
        }
        _ => None,
    }
}

/// Extract the loop bound from the header's branch condition.
///
/// Supports `Int32LessThan { left: phi, right: bound }` and
/// `Int32LessThanOrEqual { left: phi, right: bound }`, plus the
/// `GreaterThan` / `GreaterThanOrEqual` mirrors.
fn find_comparison_bound(
    header: &crate::compiler::maglev::ir::BasicBlock,
    phi_id: &NodeId,
    node_map: &HashMap<NodeId, (u32, ValueNode)>,
    ranges: &HashMap<NodeId, Range>,
) -> Option<LoopBound> {
    let cmp_id = match &header.control {
        Some(ControlNode::Branch { condition, .. }) => condition,
        _ => return None,
    };
    let (_, cmp_node) = node_map.get(cmp_id)?;

    match cmp_node {
        // phi < bound
        ValueNode::Int32LessThan { left, right } if left == phi_id => {
            Some(LoopBound::LessThan(*ranges.get(right)?))
        }
        // phi <= bound
        ValueNode::Int32LessThanOrEqual { left, right } if left == phi_id => {
            Some(LoopBound::LessThanOrEqual(*ranges.get(right)?))
        }
        // bound > phi  ⟹  phi < bound
        ValueNode::Int32GreaterThan { left, right } if right == phi_id => {
            Some(LoopBound::LessThan(*ranges.get(left)?))
        }
        // bound >= phi  ⟹  phi <= bound
        ValueNode::Int32GreaterThanOrEqual { left, right } if right == phi_id => {
            Some(LoopBound::LessThanOrEqual(*ranges.get(left)?))
        }
        _ => None,
    }
}

/// Compute the Phi's full range and the step's body range for an induction
/// variable with given `init`, `delta`, and `bound`.
///
/// For `for (i = init; i < bound; i += delta)` with delta > 0:
///   - `phi_range = [init.min, bound.max]`
///   - `step_range = [init.min + delta, bound.max - 1 + delta]`
///
/// For `<=`:
///   - `phi_range = [init.min, bound.max + delta]`
///   - `step_range = [init.min + delta, bound.max + delta]`
fn compute_induction_ranges(
    init_range: &Range,
    delta: i64,
    bound: &LoopBound,
) -> Option<(Range, Range)> {
    // Only handle positive deltas for now.
    if delta <= 0 {
        return None;
    }

    match bound {
        LoopBound::LessThan(bound_range) => {
            let phi_range = Range {
                min: init_range.min,
                max: bound_range.max,
            };
            // step fires when phi < bound ⟹ phi ∈ [init.min, bound.max - 1]
            let step_range = Range {
                min: init_range.min + delta,
                max: bound_range.max - 1 + delta,
            };
            Some((phi_range, step_range))
        }
        LoopBound::LessThanOrEqual(bound_range) => {
            let phi_range = Range {
                min: init_range.min,
                max: bound_range.max + delta,
            };
            // step fires when phi <= bound ⟹ phi ∈ [init.min, bound.max]
            let step_range = Range {
                min: init_range.min + delta,
                max: bound_range.max + delta,
            };
            Some((phi_range, step_range))
        }
    }
}

/// Convert a checked Smi or Generic step node to its unchecked `Int32*`
/// equivalent.  Generic variants are safe to lower here because the range
/// analysis has already proven that the operands stay within `i32` across
/// all loop iterations.
fn lower_checked_to_unchecked(node: &ValueNode) -> Option<ValueNode> {
    match node {
        ValueNode::CheckedSmiIncrement { value } | ValueNode::GenericIncrement { value, .. } => {
            Some(ValueNode::Int32Increment { value: *value })
        }
        ValueNode::CheckedSmiDecrement { value } | ValueNode::GenericDecrement { value, .. } => {
            Some(ValueNode::Int32Decrement { value: *value })
        }
        ValueNode::CheckedSmiAdd { left, right } | ValueNode::GenericAdd { left, right, .. } => {
            Some(ValueNode::Int32Add {
                left: *left,
                right: *right,
            })
        }
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1b — Additive accumulator detection
// ─────────────────────────────────────────────────────────────────────────────

/// Detect additive accumulator Phis and rewrite their back-edge
/// `CheckedSmiAdd` to `Int32Add` when the total accumulation provably fits
/// `i32`.
///
/// An accumulator pattern is a Phi whose back-edge input is
/// `CheckedSmiAdd { phi, addend }` where `addend` has a known range (e.g.
/// from a recognised induction variable) and the loop's max iteration count
/// bounds the total.
fn rewrite_accumulator_phis(
    graph: &mut MaglevGraph,
    ranges: &mut HashMap<NodeId, Range>,
    loop_metas: &[LoopMeta],
    node_map: &HashMap<NodeId, (u32, ValueNode)>,
) {
    // (block_idx, node_idx, back_id, back_range, replacement, phi_id, acc_range)
    let mut rewrites: Vec<(u32, usize, NodeId, Range, ValueNode, NodeId, Range)> = Vec::new();

    for meta in loop_metas {
        let header = &graph.blocks()[meta.header_idx as usize];

        for (phi_id, node) in &header.nodes {
            // Skip Phis already handled as induction variables.
            if meta.induction_phi_ids.contains(phi_id) {
                continue;
            }

            let ValueNode::Phi { inputs } = node else {
                continue;
            };
            if inputs.len() != 2 {
                continue;
            }

            let init_id = inputs[meta.entry_pos];
            let back_id = inputs[meta.back_pred_pos];

            let Some(init_range) = ranges.get(&init_id) else {
                continue;
            };
            let Some((_, back_node)) = node_map.get(&back_id) else {
                continue;
            };

            // Match additive accumulator: back = CheckedSmiAdd/GenericAdd { phi, addend }.
            let addend_range = match back_node {
                ValueNode::CheckedSmiAdd { left, right }
                | ValueNode::GenericAdd { left, right, .. }
                    if *left == *phi_id =>
                {
                    ranges.get(right).copied()
                }
                ValueNode::CheckedSmiAdd { left, right }
                | ValueNode::GenericAdd { left, right, .. }
                    if *right == *phi_id =>
                {
                    ranges.get(left).copied()
                }
                _ => None,
            };

            // Fallback: try chained pattern like `(sum + i*2) + 1`.
            let addend_range =
                addend_range.or_else(|| find_chained_addend(&back_id, phi_id, node_map, ranges));

            let Some(addend_range) = addend_range else {
                continue;
            };

            let acc_range =
                match compute_accumulator_range(init_range, &addend_range, meta.max_iterations) {
                    Some(r) => r,
                    None => continue,
                };
            if !acc_range.fits_i32() {
                continue;
            }

            // Back-edge result range: acc + addend.
            let back_min = match acc_range.min.checked_add(addend_range.min) {
                Some(v) => v,
                None => continue,
            };
            let back_max = match acc_range.max.checked_add(addend_range.max) {
                Some(v) => v,
                None => continue,
            };
            let back_range = Range {
                min: back_min,
                max: back_max,
            };
            if !back_range.fits_i32() {
                continue;
            }

            // Find the back-edge node in its block for rewriting.
            let Some((step_block_idx, _)) = node_map.get(&back_id) else {
                continue;
            };
            let step_block = &graph.blocks()[*step_block_idx as usize];
            let Some(node_idx) = step_block.nodes.iter().position(|(id, _)| *id == back_id) else {
                continue;
            };

            let replacement = match back_node {
                ValueNode::CheckedSmiAdd { left, right }
                | ValueNode::GenericAdd { left, right, .. } => ValueNode::Int32Add {
                    left: *left,
                    right: *right,
                },
                _ => continue,
            };

            rewrites.push((
                *step_block_idx,
                node_idx,
                back_id,
                back_range,
                replacement,
                *phi_id,
                acc_range,
            ));
        }
    }

    for (block_idx, node_idx, back_id, back_range, replacement, phi_id, acc_range) in rewrites {
        if let Some(block) = graph.block_mut(block_idx) {
            block.nodes[node_idx].1 = replacement;
        }
        ranges.insert(back_id, back_range);
        ranges.insert(phi_id, acc_range);
    }
}

/// Eagerly compute the range of a node that Phase 2 has not yet processed.
///
/// This is used during accumulator detection (Phase 1b) to compute addend
/// ranges for nodes like `GenericMultiply(i, 2)` whose ranges are not yet
/// in the `ranges` map.  The `phi_id` parameter is the accumulator Phi
/// being analysed — any subtree referencing it is rejected to avoid
/// circular reasoning.
fn eager_range(
    node_id: &NodeId,
    phi_id: &NodeId,
    ranges: &HashMap<NodeId, Range>,
    node_map: &HashMap<NodeId, (u32, ValueNode)>,
    depth: u32,
) -> Option<Range> {
    if let Some(r) = ranges.get(node_id) {
        return Some(*r);
    }
    // Reject cycles through the accumulator Phi.
    if *node_id == *phi_id {
        return None;
    }
    if depth > 8 {
        return None;
    }
    let (_, node) = node_map.get(node_id)?;
    match node {
        ValueNode::GenericAdd { left, right, .. } | ValueNode::CheckedSmiAdd { left, right } => {
            let lr = eager_range(left, phi_id, ranges, node_map, depth + 1)?;
            let rr = eager_range(right, phi_id, ranges, node_map, depth + 1)?;
            Some(Range {
                min: lr.min.checked_add(rr.min)?,
                max: lr.max.checked_add(rr.max)?,
            })
        }
        ValueNode::GenericSubtract { left, right, .. }
        | ValueNode::CheckedSmiSubtract { left, right } => {
            let lr = eager_range(left, phi_id, ranges, node_map, depth + 1)?;
            let rr = eager_range(right, phi_id, ranges, node_map, depth + 1)?;
            Some(Range {
                min: lr.min.checked_sub(rr.max)?,
                max: lr.max.checked_sub(rr.min)?,
            })
        }
        ValueNode::GenericMultiply { left, right, .. }
        | ValueNode::CheckedSmiMultiply { left, right } => {
            let lr = eager_range(left, phi_id, ranges, node_map, depth + 1)?;
            let rr = eager_range(right, phi_id, ranges, node_map, depth + 1)?;
            let candidates = [
                lr.min.checked_mul(rr.min)?,
                lr.min.checked_mul(rr.max)?,
                lr.max.checked_mul(rr.min)?,
                lr.max.checked_mul(rr.max)?,
            ];
            Some(Range {
                min: candidates.iter().copied().min().unwrap(),
                max: candidates.iter().copied().max().unwrap(),
            })
        }
        ValueNode::GenericIncrement { value, .. } | ValueNode::CheckedSmiIncrement { value } => {
            let vr = eager_range(value, phi_id, ranges, node_map, depth + 1)?;
            Some(Range {
                min: vr.min.checked_add(1)?,
                max: vr.max.checked_add(1)?,
            })
        }
        // Already-lowered Int32 ops (from Phase 1a induction rewriting).
        ValueNode::Int32Add { left, right } => {
            let lr = eager_range(left, phi_id, ranges, node_map, depth + 1)?;
            let rr = eager_range(right, phi_id, ranges, node_map, depth + 1)?;
            Some(Range {
                min: lr.min.checked_add(rr.min)?,
                max: lr.max.checked_add(rr.max)?,
            })
        }
        ValueNode::Int32Multiply { left, right } => {
            let lr = eager_range(left, phi_id, ranges, node_map, depth + 1)?;
            let rr = eager_range(right, phi_id, ranges, node_map, depth + 1)?;
            let candidates = [
                lr.min.checked_mul(rr.min)?,
                lr.min.checked_mul(rr.max)?,
                lr.max.checked_mul(rr.min)?,
                lr.max.checked_mul(rr.max)?,
            ];
            Some(Range {
                min: candidates.iter().copied().min().unwrap(),
                max: candidates.iter().copied().max().unwrap(),
            })
        }
        ValueNode::Int32Increment { value } => {
            let vr = eager_range(value, phi_id, ranges, node_map, depth + 1)?;
            Some(Range {
                min: vr.min.checked_add(1)?,
                max: vr.max.checked_add(1)?,
            })
        }
        ValueNode::Int32ShiftLeft { left, right } => {
            let lr = eager_range(left, phi_id, ranges, node_map, depth + 1)?;
            let rr = eager_range(right, phi_id, ranges, node_map, depth + 1)?;
            if rr.min < 0 || rr.max > 31 {
                return None;
            }
            Some(Range {
                min: lr.min.checked_shl(rr.min as u32)?,
                max: lr.max.checked_shl(rr.max as u32)?,
            })
        }
        // Bitwise OR truncates to i32 (ToInt32 per spec).
        ValueNode::GenericBitwiseOr { .. } | ValueNode::Int32BitwiseOr { .. } => {
            Some(Range::I32_FULL)
        }
        _ => None,
    }
}

/// Walk a chain of `GenericAdd`/`CheckedSmiAdd` from the back-edge towards
/// the accumulator Phi, summing the per-iteration addend contribution at
/// each link.
///
/// For `sum = sum + (i * 2) + 1` the back-edge chain is:
///   `GenericAdd(GenericAdd(sum_phi, i*2), 1)`
/// This function walks the chain and returns the total addend range
/// `[min(i*2+1), max(i*2+1)]`.
///
/// Returns `None` if the chain does not reach `phi_id` or if any addend
/// range cannot be computed.
fn find_chained_addend(
    back_id: &NodeId,
    phi_id: &NodeId,
    node_map: &HashMap<NodeId, (u32, ValueNode)>,
    ranges: &HashMap<NodeId, Range>,
) -> Option<Range> {
    let mut current = *back_id;
    let mut total_min: i64 = 0;
    let mut total_max: i64 = 0;
    let mut depth: u32 = 0;

    loop {
        if depth > 8 {
            return None;
        }
        depth += 1;

        let (_, node) = node_map.get(&current)?;
        let (left, right) = match node {
            ValueNode::GenericAdd { left, right, .. }
            | ValueNode::CheckedSmiAdd { left, right } => (*left, *right),
            _ => return None,
        };

        // Check if either operand is the accumulator Phi.
        if left == *phi_id {
            let addend = eager_range(&right, phi_id, ranges, node_map, 0)?;
            total_min = total_min.checked_add(addend.min)?;
            total_max = total_max.checked_add(addend.max)?;
            return Some(Range {
                min: total_min,
                max: total_max,
            });
        }
        if right == *phi_id {
            let addend = eager_range(&left, phi_id, ranges, node_map, 0)?;
            total_min = total_min.checked_add(addend.min)?;
            total_max = total_max.checked_add(addend.max)?;
            return Some(Range {
                min: total_min,
                max: total_max,
            });
        }

        // Neither operand is the Phi — one is an addend, the other
        // continues the chain.  Try right as addend first (more common
        // in `(sum + expr) + const` patterns).
        let right_range = eager_range(&right, phi_id, ranges, node_map, 0);
        let left_range = eager_range(&left, phi_id, ranges, node_map, 0);

        if let Some(addend) = right_range {
            total_min = total_min.checked_add(addend.min)?;
            total_max = total_max.checked_add(addend.max)?;
            current = left;
        } else if let Some(addend) = left_range {
            total_min = total_min.checked_add(addend.min)?;
            total_max = total_max.checked_add(addend.max)?;
            current = right;
        } else {
            return None;
        }
    }
}

/// Compute a conservative range for an additive accumulator.
///
/// Given an init range, per-iteration addend range, and max iteration count,
/// returns the possible range of the accumulator after all iterations.
/// Returns `None` if the computation overflows `i64`.
fn compute_accumulator_range(init: &Range, addend: &Range, max_iters: i64) -> Option<Range> {
    let min = if addend.min < 0 {
        let product = max_iters.checked_mul(addend.min)?;
        init.min.checked_add(product)?
    } else {
        init.min
    };
    let max = if addend.max > 0 {
        let product = max_iters.checked_mul(addend.max)?;
        init.max.checked_add(product)?
    } else {
        init.max
    };
    Some(Range { min, max })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::maglev::ir::{BasicBlock, ControlNode, MaglevGraph, ValueNode};

    // ── Range type ───────────────────────────────────────────────────────────

    #[test]
    fn test_range_exact_fits_i32() {
        assert!(Range::exact(0).fits_i32());
        assert!(Range::exact(i32::MAX as i64).fits_i32());
        assert!(Range::exact(i32::MIN as i64).fits_i32());
    }

    #[test]
    fn test_range_overflow_does_not_fit_i32() {
        assert!(!Range::exact(i32::MAX as i64 + 1).fits_i32());
        assert!(!Range::exact(i32::MIN as i64 - 1).fits_i32());
    }

    // ── CheckedSmiAdd → Int32Add when safe ───────────────────────────────────

    #[test]
    fn test_checked_add_small_constants_lowered() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c10 = block.push_value(ValueNode::SmiConstant { value: 10 });
        let c20 = block.push_value(ValueNode::SmiConstant { value: 20 });
        let add = block.push_value(ValueNode::CheckedSmiAdd {
            left: c10,
            right: c20,
        });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        eliminate_overflow_checks(&mut graph);

        assert!(matches!(
            &graph.blocks()[0].nodes[2].1,
            ValueNode::Int32Add { .. }
        ));
    }

    #[test]
    fn test_checked_add_near_max_not_lowered() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let big = block.push_value(ValueNode::Int32Constant {
            value: i32::MAX - 1,
        });
        let one = block.push_value(ValueNode::Int32Constant { value: 2 });
        let add = block.push_value(ValueNode::CheckedSmiAdd {
            left: big,
            right: one,
        });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        eliminate_overflow_checks(&mut graph);

        // Sum = i32::MAX + 1 overflows → keep checked.
        assert!(matches!(
            &graph.blocks()[0].nodes[2].1,
            ValueNode::CheckedSmiAdd { .. }
        ));
    }

    // ── CheckedSmiSubtract → Int32Subtract when safe ─────────────────────────

    #[test]
    fn test_checked_subtract_small_constants_lowered() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c30 = block.push_value(ValueNode::SmiConstant { value: 30 });
        let c10 = block.push_value(ValueNode::SmiConstant { value: 10 });
        let sub = block.push_value(ValueNode::CheckedSmiSubtract {
            left: c30,
            right: c10,
        });
        block.set_control(ControlNode::Return { value: sub });
        graph.add_block(block);

        eliminate_overflow_checks(&mut graph);

        assert!(matches!(
            &graph.blocks()[0].nodes[2].1,
            ValueNode::Int32Subtract { .. }
        ));
    }

    // ── CheckedSmiMultiply → Int32Multiply when safe ─────────────────────────

    #[test]
    fn test_checked_multiply_small_constants_lowered() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c5 = block.push_value(ValueNode::SmiConstant { value: 5 });
        let c7 = block.push_value(ValueNode::SmiConstant { value: 7 });
        let mul = block.push_value(ValueNode::CheckedSmiMultiply {
            left: c5,
            right: c7,
        });
        block.set_control(ControlNode::Return { value: mul });
        graph.add_block(block);

        eliminate_overflow_checks(&mut graph);

        assert!(matches!(
            &graph.blocks()[0].nodes[2].1,
            ValueNode::Int32Multiply { .. }
        ));
    }

    #[test]
    fn test_checked_multiply_overflow_not_lowered() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let big = block.push_value(ValueNode::Int32Constant {
            value: i32::MAX / 2 + 1,
        });
        let two = block.push_value(ValueNode::Int32Constant { value: 2 });
        let mul = block.push_value(ValueNode::CheckedSmiMultiply {
            left: big,
            right: two,
        });
        block.set_control(ControlNode::Return { value: mul });
        graph.add_block(block);

        eliminate_overflow_checks(&mut graph);

        assert!(matches!(
            &graph.blocks()[0].nodes[2].1,
            ValueNode::CheckedSmiMultiply { .. }
        ));
    }

    // ── CheckedSmiIncrement / Decrement ──────────────────────────────────────

    #[test]
    fn test_checked_increment_small_constant_lowered() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c = block.push_value(ValueNode::SmiConstant { value: 99 });
        let inc = block.push_value(ValueNode::CheckedSmiIncrement { value: c });
        block.set_control(ControlNode::Return { value: inc });
        graph.add_block(block);

        eliminate_overflow_checks(&mut graph);

        assert!(matches!(
            &graph.blocks()[0].nodes[1].1,
            ValueNode::Int32Increment { .. }
        ));
    }

    #[test]
    fn test_checked_increment_at_max_not_lowered() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let big = block.push_value(ValueNode::Int32Constant { value: i32::MAX });
        let inc = block.push_value(ValueNode::CheckedSmiIncrement { value: big });
        block.set_control(ControlNode::Return { value: inc });
        graph.add_block(block);

        eliminate_overflow_checks(&mut graph);

        assert!(matches!(
            &graph.blocks()[0].nodes[1].1,
            ValueNode::CheckedSmiIncrement { .. }
        ));
    }

    #[test]
    fn test_checked_decrement_small_constant_lowered() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c = block.push_value(ValueNode::SmiConstant { value: 5 });
        let dec = block.push_value(ValueNode::CheckedSmiDecrement { value: c });
        block.set_control(ControlNode::Return { value: dec });
        graph.add_block(block);

        eliminate_overflow_checks(&mut graph);

        assert!(matches!(
            &graph.blocks()[0].nodes[1].1,
            ValueNode::Int32Decrement { .. }
        ));
    }

    // ── Parameter ranges ─────────────────────────────────────────────────────

    #[test]
    fn test_parameter_add_not_lowered_full_range() {
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

        eliminate_overflow_checks(&mut graph);

        // i32::MAX + i32::MAX overflows → keep checked.
        assert!(matches!(
            &graph.blocks()[0].nodes[2].1,
            ValueNode::CheckedSmiAdd { .. }
        ));
    }

    // ── Chained narrowing ────────────────────────────────────────────────────

    #[test]
    fn test_chained_add_constants_both_lowered() {
        // (10 + 20) + 30 — both adds should be lowered.
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c10 = block.push_value(ValueNode::SmiConstant { value: 10 });
        let c20 = block.push_value(ValueNode::SmiConstant { value: 20 });
        let add1 = block.push_value(ValueNode::CheckedSmiAdd {
            left: c10,
            right: c20,
        });
        let c30 = block.push_value(ValueNode::SmiConstant { value: 30 });
        let add2 = block.push_value(ValueNode::CheckedSmiAdd {
            left: add1,
            right: c30,
        });
        block.set_control(ControlNode::Return { value: add2 });
        graph.add_block(block);

        eliminate_overflow_checks(&mut graph);

        assert!(matches!(
            &graph.blocks()[0].nodes[2].1,
            ValueNode::Int32Add { .. }
        ));
        assert!(matches!(
            &graph.blocks()[0].nodes[4].1,
            ValueNode::Int32Add { .. }
        ));
    }

    // ── Loop induction variable detection ────────────────────────────────────

    /// Build a minimal loop graph: `for (i = 0; i < bound; i++)`.
    /// Returns the graph and the [`NodeId`] of the `CheckedSmiIncrement` step.
    ///
    /// Uses explicit graph-global [`NodeId`]s to avoid block-local collisions.
    fn build_loop_graph(bound_value: i32) -> (MaglevGraph, NodeId) {
        let mut graph = MaglevGraph::new(0);

        let init = NodeId(0);
        let phi = NodeId(1);
        let bound = NodeId(2);
        let cmp = NodeId(3);
        let step = NodeId(4);

        // Block 0 (entry): init = SmiConstant(0), jump → header.
        let mut b0 = BasicBlock::new(0);
        b0.push_with_id(init, ValueNode::SmiConstant { value: 0 });
        b0.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b0);

        // Block 1 (header): phi, comparison, branch.
        let mut b1 = BasicBlock::new(1);
        b1.add_predecessor(0);
        b1.add_predecessor(2);
        b1.push_with_id(
            phi,
            ValueNode::Phi {
                inputs: vec![init, step],
            },
        );
        b1.push_with_id(bound, ValueNode::SmiConstant { value: bound_value });
        b1.push_with_id(
            cmp,
            ValueNode::Int32LessThan {
                left: phi,
                right: bound,
            },
        );
        b1.set_control(ControlNode::Branch {
            condition: cmp,
            if_true: 2,
            if_false: 3,
        });
        graph.add_block(b1);

        // Block 2 (body): increment, back-edge.
        let mut b2 = BasicBlock::new(2);
        b2.add_predecessor(1);
        b2.push_with_id(step, ValueNode::CheckedSmiIncrement { value: phi });
        b2.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b2);

        // Block 3 (exit): return phi.
        let mut b3 = BasicBlock::new(3);
        b3.add_predecessor(1);
        b3.set_control(ControlNode::Return { value: phi });
        graph.add_block(b3);

        (graph, step)
    }

    #[test]
    fn test_loop_counter_increment_lowered_constant_bound() {
        let (mut graph, step) = build_loop_graph(40);
        eliminate_overflow_checks(&mut graph);

        let step_node = &graph.blocks()[2].nodes[0];
        assert_eq!(step_node.0, step);
        assert!(
            matches!(step_node.1, ValueNode::Int32Increment { .. }),
            "expected Int32Increment, got {:?}",
            step_node.1
        );
    }

    #[test]
    fn test_loop_counter_increment_lowered_param_bound() {
        let mut graph = MaglevGraph::new(1);

        let param = NodeId(0);
        let init = NodeId(1);
        let phi = NodeId(2);
        let cmp = NodeId(3);
        let step = NodeId(4);

        let mut b0 = BasicBlock::new(0);
        b0.push_with_id(param, ValueNode::Parameter { index: 0 });
        b0.push_with_id(init, ValueNode::SmiConstant { value: 0 });
        b0.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b0);

        let mut b1 = BasicBlock::new(1);
        b1.add_predecessor(0);
        b1.add_predecessor(2);
        b1.push_with_id(
            phi,
            ValueNode::Phi {
                inputs: vec![init, step],
            },
        );
        b1.push_with_id(
            cmp,
            ValueNode::Int32LessThan {
                left: phi,
                right: param,
            },
        );
        b1.set_control(ControlNode::Branch {
            condition: cmp,
            if_true: 2,
            if_false: 3,
        });
        graph.add_block(b1);

        let mut b2 = BasicBlock::new(2);
        b2.add_predecessor(1);
        b2.push_with_id(step, ValueNode::CheckedSmiIncrement { value: phi });
        b2.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b2);

        let mut b3 = BasicBlock::new(3);
        b3.add_predecessor(1);
        b3.set_control(ControlNode::Return { value: phi });
        graph.add_block(b3);

        eliminate_overflow_checks(&mut graph);

        // param bound has range [i32::MIN, i32::MAX].  For `i < param` with
        // delta = 1: step ∈ [1, i32::MAX].  Fits i32 → lowered.
        assert!(
            matches!(
                graph.blocks()[2].nodes[0].1,
                ValueNode::Int32Increment { .. }
            ),
            "expected Int32Increment with parameter bound"
        );
    }

    #[test]
    fn test_loop_counter_not_lowered_when_unsafe() {
        let mut graph = MaglevGraph::new(1);

        let param = NodeId(0);
        let init = NodeId(1);
        let phi = NodeId(2);
        let cmp = NodeId(3);
        let step = NodeId(4);

        let mut b0 = BasicBlock::new(0);
        b0.push_with_id(param, ValueNode::Parameter { index: 0 });
        b0.push_with_id(init, ValueNode::SmiConstant { value: 0 });
        b0.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b0);

        let mut b1 = BasicBlock::new(1);
        b1.add_predecessor(0);
        b1.add_predecessor(2);
        b1.push_with_id(
            phi,
            ValueNode::Phi {
                inputs: vec![init, step],
            },
        );
        b1.push_with_id(
            cmp,
            ValueNode::Int32LessThanOrEqual {
                left: phi,
                right: param,
            },
        );
        b1.set_control(ControlNode::Branch {
            condition: cmp,
            if_true: 2,
            if_false: 3,
        });
        graph.add_block(b1);

        let mut b2 = BasicBlock::new(2);
        b2.add_predecessor(1);
        b2.push_with_id(step, ValueNode::CheckedSmiIncrement { value: phi });
        b2.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b2);

        let mut b3 = BasicBlock::new(3);
        b3.add_predecessor(1);
        b3.set_control(ControlNode::Return { value: phi });
        graph.add_block(b3);

        eliminate_overflow_checks(&mut graph);

        // With `<=` and parameter bound: step could reach i32::MAX + 1 → NOT lowered.
        assert!(
            matches!(
                graph.blocks()[2].nodes[0].1,
                ValueNode::CheckedSmiIncrement { .. }
            ),
            "expected CheckedSmiIncrement to stay checked for unsafe <= bound"
        );
    }

    // ── Multi-Phi loop (induction variable not first) ────────────────────

    /// Build a loop graph for `for (i = 0; i < bound; i++) { sum += i; }`
    /// with the `sum` Phi placed **before** the `i` Phi in the header, so
    /// the old `find_map` approach would fail to detect `i`.
    fn build_sum_loop_graph(bound_value: i32) -> (MaglevGraph, NodeId, NodeId, NodeId) {
        let mut graph = MaglevGraph::new(0);

        let init_i = NodeId(0);
        let init_sum = NodeId(1);
        let phi_sum = NodeId(2); // placed first in header
        let phi_i = NodeId(3);
        let bound = NodeId(4);
        let cmp = NodeId(5);
        let add = NodeId(6); // sum + i
        let step = NodeId(7); // i++

        // Block 0 (entry)
        let mut b0 = BasicBlock::new(0);
        b0.push_with_id(init_i, ValueNode::SmiConstant { value: 0 });
        b0.push_with_id(init_sum, ValueNode::SmiConstant { value: 0 });
        b0.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b0);

        // Block 1 (header): sum Phi comes first, then i Phi.
        let mut b1 = BasicBlock::new(1);
        b1.add_predecessor(0);
        b1.add_predecessor(2);
        b1.push_with_id(
            phi_sum,
            ValueNode::Phi {
                inputs: vec![init_sum, add],
            },
        );
        b1.push_with_id(
            phi_i,
            ValueNode::Phi {
                inputs: vec![init_i, step],
            },
        );
        b1.push_with_id(bound, ValueNode::SmiConstant { value: bound_value });
        b1.push_with_id(
            cmp,
            ValueNode::Int32LessThan {
                left: phi_i,
                right: bound,
            },
        );
        b1.set_control(ControlNode::Branch {
            condition: cmp,
            if_true: 2,
            if_false: 3,
        });
        graph.add_block(b1);

        // Block 2 (body): sum += i; i++
        let mut b2 = BasicBlock::new(2);
        b2.add_predecessor(1);
        b2.push_with_id(
            add,
            ValueNode::CheckedSmiAdd {
                left: phi_sum,
                right: phi_i,
            },
        );
        b2.push_with_id(step, ValueNode::CheckedSmiIncrement { value: phi_i });
        b2.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b2);

        // Block 3 (exit)
        let mut b3 = BasicBlock::new(3);
        b3.add_predecessor(1);
        b3.set_control(ControlNode::Return { value: phi_sum });
        graph.add_block(b3);

        (graph, step, add, phi_i)
    }

    #[test]
    fn test_loop_counter_lowered_when_not_first_phi() {
        let (mut graph, step, _add, _phi_i) = build_sum_loop_graph(10000);
        eliminate_overflow_checks(&mut graph);

        // The induction variable `i++` should still be lowered even though
        // the `sum` Phi appears first in the header.
        let step_node = &graph.blocks()[2].nodes[1];
        assert_eq!(step_node.0, step);
        assert!(
            matches!(step_node.1, ValueNode::Int32Increment { .. }),
            "expected Int32Increment, got {:?}",
            step_node.1
        );
    }

    #[test]
    fn test_accumulator_add_lowered_small_bound() {
        let (mut graph, _step, add, _phi_i) = build_sum_loop_graph(10000);
        eliminate_overflow_checks(&mut graph);

        // sum += i with bound 10000: acc ∈ [0, 100_000_000] fits i32.
        // The CheckedSmiAdd should be lowered to Int32Add.
        let add_node = &graph.blocks()[2].nodes[0];
        assert_eq!(add_node.0, add);
        assert!(
            matches!(add_node.1, ValueNode::Int32Add { .. }),
            "expected Int32Add for accumulator, got {:?}",
            add_node.1
        );
    }

    #[test]
    fn test_accumulator_add_not_lowered_huge_bound() {
        // bound = 50_000: max_iters = 50_000, addend_max = 50_000
        // acc_max = 50_000 * 50_000 = 2_500_000_000 > i32::MAX
        let (mut graph, _step, add, _phi_i) = build_sum_loop_graph(50_000);
        eliminate_overflow_checks(&mut graph);

        let add_node = &graph.blocks()[2].nodes[0];
        assert_eq!(add_node.0, add);
        assert!(
            matches!(add_node.1, ValueNode::CheckedSmiAdd { .. }),
            "expected CheckedSmiAdd to stay checked for large accumulator, got {:?}",
            add_node.1
        );
    }

    // ── Chained accumulator pattern: sum = sum + i*2 + 1 ────────────────

    /// Build a loop graph for `for (i = 0; i < bound; i++) sum = sum + i*2 + 1;`
    /// This creates a chained accumulator pattern where the back-edge goes
    /// through two GenericAdd nodes: GenericAdd(GenericAdd(sum, i*2), 1).
    fn build_chained_sum_loop_graph(
        bound_value: i32,
    ) -> (MaglevGraph, NodeId, NodeId, NodeId, NodeId) {
        let mut graph = MaglevGraph::new(0);

        let init_i = NodeId(0);
        let init_sum = NodeId(1);
        let const2 = NodeId(2);
        let const1 = NodeId(3);
        let phi_sum = NodeId(10);
        let phi_i = NodeId(11);
        let bound = NodeId(12);
        let cmp = NodeId(13);
        let mul = NodeId(20); // i * 2
        let add1 = NodeId(21); // sum + i*2
        let add2 = NodeId(22); // (sum + i*2) + 1
        let step = NodeId(23); // i++

        // Block 0 (entry)
        let mut b0 = BasicBlock::new(0);
        b0.push_with_id(init_i, ValueNode::SmiConstant { value: 0 });
        b0.push_with_id(init_sum, ValueNode::SmiConstant { value: 0 });
        b0.push_with_id(const2, ValueNode::SmiConstant { value: 2 });
        b0.push_with_id(const1, ValueNode::SmiConstant { value: 1 });
        b0.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b0);

        // Block 1 (header)
        let mut b1 = BasicBlock::new(1);
        b1.add_predecessor(0);
        b1.add_predecessor(2);
        b1.push_with_id(
            phi_sum,
            ValueNode::Phi {
                inputs: vec![init_sum, add2],
            },
        );
        b1.push_with_id(
            phi_i,
            ValueNode::Phi {
                inputs: vec![init_i, step],
            },
        );
        b1.push_with_id(bound, ValueNode::SmiConstant { value: bound_value });
        b1.push_with_id(
            cmp,
            ValueNode::Int32LessThan {
                left: phi_i,
                right: bound,
            },
        );
        b1.set_control(ControlNode::Branch {
            condition: cmp,
            if_true: 2,
            if_false: 3,
        });
        graph.add_block(b1);

        // Block 2 (body): t1 = i*2; t2 = sum + t1; t3 = t2 + 1; i++
        let mut b2 = BasicBlock::new(2);
        b2.add_predecessor(1);
        b2.push_with_id(
            mul,
            ValueNode::GenericMultiply {
                left: phi_i,
                right: const2,
                feedback_slot: 0,
            },
        );
        b2.push_with_id(
            add1,
            ValueNode::GenericAdd {
                left: phi_sum,
                right: mul,
                feedback_slot: 0,
            },
        );
        b2.push_with_id(
            add2,
            ValueNode::GenericAdd {
                left: add1,
                right: const1,
                feedback_slot: 0,
            },
        );
        b2.push_with_id(step, ValueNode::CheckedSmiIncrement { value: phi_i });
        b2.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b2);

        // Block 3 (exit)
        let mut b3 = BasicBlock::new(3);
        b3.add_predecessor(1);
        b3.set_control(ControlNode::Return { value: phi_sum });
        graph.add_block(b3);

        (graph, add1, add2, mul, step)
    }

    #[test]
    fn test_chained_accumulator_lowered() {
        // sum = sum + i*2 + 1, bound = 10000
        // Per-iteration addend range: i*2 ∈ [0, 19998], plus 1 → [1, 19999]
        // Accumulator range: [0, 10000*19999] = [0, 199990000] fits i32
        let (mut graph, _add1, add2, _mul, _step) = build_chained_sum_loop_graph(10_000);
        eliminate_overflow_checks(&mut graph);

        // The back-edge node (add2: (sum+i*2)+1) should be lowered to
        // Int32Add since the accumulator range fits i32.
        let back_node = graph.blocks()[2]
            .nodes
            .iter()
            .find(|(id, _)| *id == add2)
            .unwrap();
        assert!(
            matches!(back_node.1, ValueNode::Int32Add { .. }),
            "expected Int32Add for chained accumulator back-edge, got {:?}",
            back_node.1
        );
    }

    #[test]
    fn test_chained_accumulator_intermediate_lowered() {
        // After chained detection assigns the Phi range, Phase 2 should
        // also lower the intermediate GenericAdd (sum + i*2) → Int32Add.
        let (mut graph, add1, _add2, _mul, _step) = build_chained_sum_loop_graph(10_000);
        eliminate_overflow_checks(&mut graph);

        let mid_node = graph.blocks()[2]
            .nodes
            .iter()
            .find(|(id, _)| *id == add1)
            .unwrap();
        assert!(
            matches!(mid_node.1, ValueNode::Int32Add { .. }),
            "expected Int32Add for intermediate accumulator add, got {:?}",
            mid_node.1
        );
    }

    #[test]
    fn test_chained_accumulator_multiply_lowered() {
        // The GenericMultiply(i, 2) in the chain should also be lowered
        // to Int32Multiply by Phase 2.
        let (mut graph, _add1, _add2, mul, _step) = build_chained_sum_loop_graph(10_000);
        eliminate_overflow_checks(&mut graph);

        let mul_node = graph.blocks()[2]
            .nodes
            .iter()
            .find(|(id, _)| *id == mul)
            .unwrap();
        assert!(
            matches!(mul_node.1, ValueNode::Int32Multiply { .. }),
            "expected Int32Multiply for i*2, got {:?}",
            mul_node.1
        );
    }

    #[test]
    fn test_chained_accumulator_not_lowered_huge_bound() {
        // bound = 50000: per-iter max addend = 2*49999+1 = 99999
        // acc_max = 50000 * 99999 = 4_999_950_000 > i32::MAX (~2.1B)
        let (mut graph, _add1, add2, _mul, _step) = build_chained_sum_loop_graph(50_000);
        eliminate_overflow_checks(&mut graph);

        let back_node = graph.blocks()[2]
            .nodes
            .iter()
            .find(|(id, _)| *id == add2)
            .unwrap();
        assert!(
            matches!(back_node.1, ValueNode::GenericAdd { .. }),
            "expected GenericAdd to stay generic for huge accumulator, got {:?}",
            back_node.1
        );
    }
}
