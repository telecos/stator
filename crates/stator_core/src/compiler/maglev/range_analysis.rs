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
//! - Phi nodes not involved in a recognised induction variable pattern are
//!   not assigned a range (they are skipped).
//! - Only positive step deltas are handled for loop induction.
//! - Only `i32` ranges are tracked; `f64` and `u32` are left untouched.
//!
//! # Usage
//!
//! ```
//! use stator_core::compiler::maglev::ir::{
//!     BasicBlock, ControlNode, MaglevGraph, ValueNode,
//! };
//! use stator_core::compiler::maglev::range_analysis::eliminate_overflow_checks;
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

    // Phase 1 — detect loop induction variables and rewrite step nodes.
    rewrite_loop_induction_steps(graph, &mut ranges);

    // Phase 2 — forward propagation and rewriting.
    for block in graph.blocks_mut() {
        for (id, node) in &mut block.nodes {
            // Seed any nodes missed in Phase 0 (shouldn't happen, but
            // defensive).
            if !ranges.contains_key(id)
                && let Some(r) = seed_range(node)
            {
                ranges.insert(*id, r);
            }

            if let Some((out_range, replacement)) = try_rewrite(node, &ranges) {
                ranges.insert(*id, out_range);
                if let Some(new_node) = replacement {
                    *node = new_node;
                }
            }
        }
    }
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
        // Parameters have the full i32 range.
        ValueNode::Parameter { .. } => Some(Range::I32_FULL),
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
    ranges: &HashMap<NodeId, Range>,
) -> Option<(Range, Option<ValueNode>)> {
    match node {
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

/// Scan the graph for loop induction variable patterns and, where safe,
/// rewrite checked step nodes to unchecked variants.
///
/// Pattern: a Phi with two inputs (entry value, back-edge step), where the
/// step is `CheckedSmiIncrement { value: phi }` and the loop header has a
/// comparison `phi < bound` or `phi <= bound`.
fn rewrite_loop_induction_steps(graph: &mut MaglevGraph, ranges: &mut HashMap<NodeId, Range>) {
    // Build a node → (block_idx, value_node) map for fast lookup.
    let mut node_map: HashMap<NodeId, (u32, ValueNode)> = HashMap::new();
    for block in graph.blocks() {
        for (id, node) in &block.nodes {
            node_map.insert(*id, (block.id, node.clone()));
        }
    }

    // Detect back-edges: block B with Jump { target: H } where H <= B.
    let mut back_edges: Vec<(u32, u32)> = Vec::new();
    for block in graph.blocks() {
        if let Some(ControlNode::Jump { target }) = &block.control
            && *target <= block.id
        {
            back_edges.push((block.id, *target));
        }
    }

    // For each back-edge, analyze the header for induction variable patterns.
    let mut rewrites: Vec<(u32, usize, ValueNode, NodeId, Range)> = Vec::new();

    for &(back_src, header_idx) in &back_edges {
        if let Some(rw) =
            analyze_header_for_induction(graph, header_idx, back_src, &node_map, ranges)
        {
            rewrites.push(rw);
        }
    }

    // Apply rewrites: replace step nodes and record their ranges.
    for (block_idx, node_idx, new_node, step_id, step_range) in rewrites {
        if let Some(block) = graph.block_mut(block_idx) {
            block.nodes[node_idx].1 = new_node;
        }
        ranges.insert(step_id, step_range);
    }
}

/// Analyze a loop header for an induction variable pattern.
///
/// Returns `Some((block_idx, node_idx, replacement_node, step_id, step_range))`
/// if the pattern matches and the step can be safely lowered.
fn analyze_header_for_induction(
    graph: &MaglevGraph,
    header_idx: u32,
    back_src: u32,
    node_map: &HashMap<NodeId, (u32, ValueNode)>,
    ranges: &HashMap<NodeId, Range>,
) -> Option<(u32, usize, ValueNode, NodeId, Range)> {
    let header = &graph.blocks()[header_idx as usize];

    // Find the back-edge predecessor position.
    let back_pred_pos = header.predecessors.iter().position(|&p| p == back_src)?;
    let entry_pos = if back_pred_pos == 0 { 1 } else { 0 };

    // Find a Phi with exactly 2 inputs in the header.
    let (phi_id, phi_inputs) = header.nodes.iter().find_map(|(id, node)| {
        if let ValueNode::Phi { inputs } = node
            && inputs.len() == 2
        {
            return Some((*id, inputs.clone()));
        }
        None
    })?;

    let init_id = phi_inputs[entry_pos];
    let step_id = phi_inputs[back_pred_pos];
    let init_range = ranges.get(&init_id)?;

    // Check that the step is an increment/add of the phi.
    let delta = find_step_delta(&step_id, &phi_id, node_map, ranges)?;

    // Find the comparison bound in the header's branch condition.
    let bound = find_comparison_bound(header, &phi_id, node_map, ranges)?;

    // Compute the phi's full range and the step's body range.
    let (phi_range, step_range) = compute_induction_ranges(init_range, delta, &bound)?;

    // Both must fit i32 for safe lowering.
    if !phi_range.fits_i32() || !step_range.fits_i32() {
        return None;
    }

    // Find the step node's position in its block.
    let (step_block_idx, step_node) = node_map.get(&step_id)?;
    let step_block = &graph.blocks()[*step_block_idx as usize];
    let node_idx = step_block.nodes.iter().position(|(id, _)| *id == step_id)?;

    // Build the unchecked replacement.
    let replacement = lower_checked_to_unchecked(step_node)?;

    Some((*step_block_idx, node_idx, replacement, step_id, step_range))
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
        ValueNode::CheckedSmiIncrement { value } if value == phi_id => Some(1),
        ValueNode::CheckedSmiDecrement { value } if value == phi_id => Some(-1),
        ValueNode::CheckedSmiAdd { left, right } if left == phi_id => {
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

/// Convert a checked Smi step node to its unchecked `Int32*` equivalent.
fn lower_checked_to_unchecked(node: &ValueNode) -> Option<ValueNode> {
    match node {
        ValueNode::CheckedSmiIncrement { value } => {
            Some(ValueNode::Int32Increment { value: *value })
        }
        ValueNode::CheckedSmiDecrement { value } => {
            Some(ValueNode::Int32Decrement { value: *value })
        }
        ValueNode::CheckedSmiAdd { left, right } => Some(ValueNode::Int32Add {
            left: *left,
            right: *right,
        }),
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
}
