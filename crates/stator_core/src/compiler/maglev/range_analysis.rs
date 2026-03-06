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
//! 1. **Seed** — assign exact ranges to constant nodes (`SmiConstant`,
//!    `Int32Constant`).  Parameters and other dynamic nodes receive the
//!    full `[i32::MIN, i32::MAX]` interval.
//! 2. **Propagate** — walk each block linearly and, for every arithmetic
//!    node whose inputs already have ranges, compute the output range via
//!    interval arithmetic.
//! 3. **Rewrite** — for `CheckedSmi*` nodes whose computed output range is
//!    within `[i32::MIN, i32::MAX]`, replace the node with the unchecked
//!    `Int32*` equivalent.
//!
//! # Limitations
//!
//! - Only a single forward pass is performed (no fixed-point iteration).
//! - Phi nodes are conservatively widened to `[i32::MIN, i32::MAX]`.
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

use crate::compiler::maglev::ir::{MaglevGraph, NodeId, ValueNode};

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
pub fn eliminate_overflow_checks(graph: &mut MaglevGraph) {
    let mut ranges: HashMap<NodeId, Range> = HashMap::new();

    for block in graph.blocks_mut() {
        for (id, node) in &mut block.nodes {
            // 1. Seed constants.
            if let Some(r) = seed_range(node) {
                ranges.insert(*id, r);
            }

            // 2. Propagate ranges through arithmetic and try rewriting.
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
}
