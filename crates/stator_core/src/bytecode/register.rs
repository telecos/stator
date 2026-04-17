//! Virtual register types and register allocator for the Stator bytecode
//! compiler.
//!
//! # Register layout
//!
//! | Index range | Meaning                                             |
//! |-------------|-----------------------------------------------------|
//! | `i32::MIN`  | **Accumulator** — the implicit operand register     |
//! | `< 0`       | **Parameter** registers (`-1` = param\[0\], …)      |
//! | `>= 0`      | **Local** and **temporary** registers               |
//!
//! Most arithmetic and load/store bytecodes use the accumulator as an
//! implicit source or destination.  Parameter registers are pre-allocated
//! before a function's body is compiled.  Local registers are assigned
//! sequentially by the compiler for named bindings, and temporaries are
//! stacked above locals for short-lived scratch values.
//!
//! # Example
//!
//! ```
//! use stator_js::bytecode::register::{Register, RegisterAllocator};
//!
//! let mut alloc = RegisterAllocator::new(2); // 2 formal parameters
//!
//! // Parameters are pre-indexed — no allocator state is mutated.
//! let p0 = alloc.new_parameter(0).unwrap();
//! let p1 = alloc.new_parameter(1).unwrap();
//! assert!(p0.is_parameter());
//! assert_eq!(p0.to_string(), "a0");
//!
//! // Named locals are assigned sequentially.
//! let x = alloc.new_local();
//! let y = alloc.new_local();
//! assert!(x.is_local());
//! assert_eq!(x.to_string(), "r0");
//! assert_eq!(y.to_string(), "r1");
//!
//! // Temporaries are stacked above locals.
//! let t0 = alloc.allocate_temporary();
//! assert_eq!(alloc.temporary_count(), 1);
//! alloc.release_temporary(t0).unwrap();
//! assert_eq!(alloc.temporary_count(), 0);
//!
//! // frame_size records the high-water mark (2 locals + 1 temp peak).
//! assert_eq!(alloc.frame_size(), 3);
//! ```

use std::fmt;

use crate::error::{StatorError, StatorResult};

// ─────────────────────────────────────────────────────────────────────────────
// Register
// ─────────────────────────────────────────────────────────────────────────────

/// A virtual register in the Stator bytecode VM.
///
/// The `i32` index encodes the register's role:
///
/// - `i32::MIN` — the implicit **accumulator** register
///   ([`Register::ACCUMULATOR`]).  Most arithmetic and load/store instructions
///   use it without encoding it as an explicit bytecode operand.
/// - Negative (`< 0`) — a **parameter** register.  Index `-1` corresponds to
///   parameter 0, `-2` to parameter 1, and so on.
/// - Non-negative (`>= 0`) — a **local** or **temporary** register.  The
///   first `n` non-negative indices are assigned to named locals; temporaries
///   are allocated above them by [`RegisterAllocator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Register(pub i32);

impl Register {
    /// The implicit accumulator register.
    ///
    /// Encoded as `i32::MIN` so that it is always distinct from any parameter
    /// or local register index.
    pub const ACCUMULATOR: Self = Self(i32::MIN);

    /// Construct a **parameter** register for the given zero-based `index`.
    ///
    /// Parameter 0 maps to register `-1`, parameter 1 to `-2`, and so on.
    ///
    /// # Example
    ///
    /// ```
    /// use stator_js::bytecode::register::Register;
    /// assert_eq!(Register::parameter(0), Register(-1));
    /// assert_eq!(Register::parameter(2), Register(-3));
    /// ```
    pub fn parameter(index: u32) -> Self {
        Self(-(index as i32) - 1)
    }

    /// Construct a **local** (or temporary) register with the given
    /// non-negative `index`.
    ///
    /// # Example
    ///
    /// ```
    /// use stator_js::bytecode::register::Register;
    /// assert_eq!(Register::local(0), Register(0));
    /// assert_eq!(Register::local(3), Register(3));
    /// ```
    pub fn local(index: u32) -> Self {
        Self(index as i32)
    }

    /// Returns `true` if this is the implicit accumulator register.
    pub fn is_accumulator(self) -> bool {
        self == Self::ACCUMULATOR
    }

    /// Returns `true` if this is a **parameter** register (index `< 0`,
    /// excluding the accumulator sentinel).
    pub fn is_parameter(self) -> bool {
        self.0 < 0 && self != Self::ACCUMULATOR
    }

    /// Returns `true` if this is a **local** or **temporary** register
    /// (index `>= 0`).
    pub fn is_local(self) -> bool {
        self.0 >= 0
    }

    /// Returns the zero-based parameter index, or `None` if this is not a
    /// parameter register.
    ///
    /// # Example
    ///
    /// ```
    /// use stator_js::bytecode::register::Register;
    /// assert_eq!(Register::parameter(1).parameter_index(), Some(1));
    /// assert_eq!(Register::local(0).parameter_index(), None);
    /// ```
    pub fn parameter_index(self) -> Option<u32> {
        if self.is_parameter() {
            Some((-(self.0 + 1)) as u32)
        } else {
            None
        }
    }

    /// Returns the non-negative local/temporary index, or `None` if this is
    /// not a local register.
    ///
    /// # Example
    ///
    /// ```
    /// use stator_js::bytecode::register::Register;
    /// assert_eq!(Register::local(5).local_index(), Some(5));
    /// assert_eq!(Register::parameter(0).local_index(), None);
    /// ```
    pub fn local_index(self) -> Option<u32> {
        if self.is_local() {
            Some(self.0 as u32)
        } else {
            None
        }
    }
}

impl fmt::Display for Register {
    /// Formats the register for diagnostic output.
    ///
    /// - Accumulator → `"acc"`
    /// - Parameter `n` → `"a{n}"`
    /// - Local/temporary `n` → `"r{n}"`
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_accumulator() {
            write!(f, "acc")
        } else if let Some(idx) = self.parameter_index() {
            write!(f, "a{idx}")
        } else {
            write!(f, "r{}", self.local_index().unwrap())
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RegisterAllocator
// ─────────────────────────────────────────────────────────────────────────────

/// Allocates virtual registers for a single function during bytecode
/// compilation.
///
/// The allocator manages three classes of registers:
///
/// - **Parameter** registers — one per formal parameter, accessed via
///   [`new_parameter`].  These correspond to negative register indices and do
///   not affect [`frame_size`].
/// - **Local** registers — one per named binding in the function scope,
///   assigned in call order by [`new_local`].  These occupy the low end of
///   the non-negative index space.
/// - **Temporary** registers — short-lived scratch space allocated above
///   locals by [`allocate_temporary`] and released in LIFO order by
///   [`release_temporary`].
///
/// [`frame_size`] returns the high-water mark of `local_count +
/// temporary_count` observed during compilation.  This is the number of
/// register slots the VM must reserve for each activation of the function.
///
/// [`new_parameter`]: RegisterAllocator::new_parameter
/// [`new_local`]: RegisterAllocator::new_local
/// [`allocate_temporary`]: RegisterAllocator::allocate_temporary
/// [`release_temporary`]: RegisterAllocator::release_temporary
/// [`frame_size`]: RegisterAllocator::frame_size
#[derive(Debug)]
pub struct RegisterAllocator {
    /// Number of formal parameters declared by the function.
    parameter_count: u32,
    /// Next available local register index (grows monotonically).
    local_count: u32,
    /// Number of currently live temporary registers.
    temporary_count: u32,
    /// High-water mark: max(`local_count` + `temporary_count`) ever seen.
    frame_size: u32,
}

impl RegisterAllocator {
    /// Create a new allocator for a function with `parameter_count` formal
    /// parameters.
    pub fn new(parameter_count: u32) -> Self {
        Self {
            parameter_count,
            local_count: 0,
            temporary_count: 0,
            frame_size: 0,
        }
    }

    /// Return the pre-indexed parameter register for the given zero-based
    /// `index`.
    ///
    /// Returns [`StatorError::Internal`] if `index >= parameter_count`.
    pub fn new_parameter(&self, index: u32) -> StatorResult<Register> {
        if index < self.parameter_count {
            Ok(Register::parameter(index))
        } else {
            Err(StatorError::Internal(format!(
                "parameter index {index} out of range (count = {})",
                self.parameter_count
            )))
        }
    }

    /// Assign the next available **local** register and return it.
    ///
    /// Each call increments the internal local counter and updates
    /// [`frame_size`] if necessary.
    ///
    /// [`frame_size`]: RegisterAllocator::frame_size
    pub fn new_local(&mut self) -> Register {
        let reg = Register::local(self.local_count);
        self.local_count += 1;
        self.update_frame_size();
        reg
    }

    /// Allocate the next **temporary** register (stacked above locals) and
    /// return it.
    ///
    /// Temporaries are numbered `local_count`, `local_count + 1`, … and must
    /// be released in LIFO order.  The internal high-water mark is updated
    /// automatically.
    pub fn allocate_temporary(&mut self) -> Register {
        let reg = Register::local(self.local_count + self.temporary_count);
        self.temporary_count += 1;
        self.update_frame_size();
        reg
    }

    /// Release a temporary register previously returned by
    /// [`allocate_temporary`].
    ///
    /// Temporaries must be released in **LIFO** order.  Releasing the wrong
    /// register (i.e. not the most-recently-allocated one) or calling this
    /// when no temporaries are live returns [`StatorError::Internal`].
    pub fn release_temporary(&mut self, reg: Register) -> StatorResult<()> {
        if self.temporary_count == 0 {
            return Err(StatorError::Internal(
                "release_temporary called with no live temporaries".into(),
            ));
        }
        let expected_index = self.local_count + self.temporary_count - 1;
        match reg.local_index() {
            Some(idx) if idx == expected_index => {
                self.temporary_count -= 1;
                Ok(())
            }
            _ => Err(StatorError::Internal(format!(
                "release_temporary: expected r{expected_index}, got {reg}"
            ))),
        }
    }

    /// Number of formal parameters for this function.
    pub fn parameter_count(&self) -> u32 {
        self.parameter_count
    }

    /// Number of named local registers assigned so far.
    pub fn local_count(&self) -> u32 {
        self.local_count
    }

    /// Number of currently live temporary registers.
    pub fn temporary_count(&self) -> u32 {
        self.temporary_count
    }

    /// The minimum VM frame size required: the high-water mark of
    /// `local_count + temporary_count` observed during compilation.
    pub fn frame_size(&self) -> u32 {
        self.frame_size
    }

    /// Update the high-water mark after a new local or temporary is allocated.
    fn update_frame_size(&mut self) {
        let current = self.local_count + self.temporary_count;
        if current > self.frame_size {
            self.frame_size = current;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Register type ────────────────────────────────────────────────────────

    #[test]
    fn test_register_accumulator() {
        let acc = Register::ACCUMULATOR;
        assert!(acc.is_accumulator());
        assert!(!acc.is_parameter());
        assert!(!acc.is_local());
        assert_eq!(acc.parameter_index(), None);
        assert_eq!(acc.local_index(), None);
        assert_eq!(acc.to_string(), "acc");
    }

    #[test]
    fn test_register_parameter() {
        let p0 = Register::parameter(0);
        assert_eq!(p0, Register(-1));
        assert!(p0.is_parameter());
        assert!(!p0.is_accumulator());
        assert!(!p0.is_local());
        assert_eq!(p0.parameter_index(), Some(0));
        assert_eq!(p0.local_index(), None);
        assert_eq!(p0.to_string(), "a0");

        let p3 = Register::parameter(3);
        assert_eq!(p3, Register(-4));
        assert_eq!(p3.parameter_index(), Some(3));
        assert_eq!(p3.to_string(), "a3");
    }

    #[test]
    fn test_register_local() {
        let r0 = Register::local(0);
        assert_eq!(r0, Register(0));
        assert!(r0.is_local());
        assert!(!r0.is_accumulator());
        assert!(!r0.is_parameter());
        assert_eq!(r0.local_index(), Some(0));
        assert_eq!(r0.parameter_index(), None);
        assert_eq!(r0.to_string(), "r0");

        let r5 = Register::local(5);
        assert_eq!(r5.local_index(), Some(5));
        assert_eq!(r5.to_string(), "r5");
    }

    // ── RegisterAllocator ────────────────────────────────────────────────────

    #[test]
    fn test_allocator_parameter_range() {
        let alloc = RegisterAllocator::new(3);
        assert_eq!(alloc.parameter_count(), 3);

        assert_eq!(alloc.new_parameter(0).unwrap(), Register::parameter(0));
        assert_eq!(alloc.new_parameter(2).unwrap(), Register::parameter(2));
        assert!(alloc.new_parameter(3).is_err());
    }

    #[test]
    fn test_allocator_locals_sequential() {
        let mut alloc = RegisterAllocator::new(0);

        let r0 = alloc.new_local();
        let r1 = alloc.new_local();
        let r2 = alloc.new_local();

        assert_eq!(r0, Register::local(0));
        assert_eq!(r1, Register::local(1));
        assert_eq!(r2, Register::local(2));
        assert_eq!(alloc.local_count(), 3);
    }

    #[test]
    fn test_allocate_release_temporary() {
        let mut alloc = RegisterAllocator::new(0);
        let _x = alloc.new_local(); // r0 is a named local

        let t0 = alloc.allocate_temporary();
        assert_eq!(t0, Register::local(1)); // stacked above r0
        assert_eq!(alloc.temporary_count(), 1);

        let t1 = alloc.allocate_temporary();
        assert_eq!(t1, Register::local(2));
        assert_eq!(alloc.temporary_count(), 2);

        // LIFO release
        alloc.release_temporary(t1).unwrap();
        assert_eq!(alloc.temporary_count(), 1);

        alloc.release_temporary(t0).unwrap();
        assert_eq!(alloc.temporary_count(), 0);
    }

    #[test]
    fn test_release_out_of_order_is_error() {
        let mut alloc = RegisterAllocator::new(0);
        let t0 = alloc.allocate_temporary();
        let _t1 = alloc.allocate_temporary();

        // Releasing t0 before t1 is out of order.
        assert!(alloc.release_temporary(t0).is_err());
    }

    #[test]
    fn test_release_when_none_live_is_error() {
        let mut alloc = RegisterAllocator::new(0);
        let fake = Register::local(0);
        assert!(alloc.release_temporary(fake).is_err());
    }

    #[test]
    fn test_frame_size_tracks_high_water_mark() {
        let mut alloc = RegisterAllocator::new(1);

        // 2 locals → frame_size = 2
        let _a = alloc.new_local();
        let _b = alloc.new_local();
        assert_eq!(alloc.frame_size(), 2);

        // 1 temp → frame_size = 3
        let t0 = alloc.allocate_temporary();
        assert_eq!(alloc.frame_size(), 3);

        // 1 more temp → frame_size = 4
        let t1 = alloc.allocate_temporary();
        assert_eq!(alloc.frame_size(), 4);

        // Release both temporaries; frame_size stays at 4
        alloc.release_temporary(t1).unwrap();
        alloc.release_temporary(t0).unwrap();
        assert_eq!(alloc.frame_size(), 4);

        // Allocating another temp reuses the same slot; frame_size unchanged.
        let t2 = alloc.allocate_temporary();
        assert_eq!(alloc.frame_size(), 4);
        alloc.release_temporary(t2).unwrap();
        assert_eq!(alloc.frame_size(), 4);
    }

    #[test]
    fn test_frame_size_zero_with_no_locals() {
        let alloc = RegisterAllocator::new(5);
        // Parameters do not contribute to frame_size.
        assert_eq!(alloc.frame_size(), 0);
    }

    #[test]
    fn test_accumulator_distinct_from_all_registers() {
        // Make sure the accumulator sentinel can never alias a real register.
        let acc = Register::ACCUMULATOR;
        for i in 0_u32..100 {
            assert_ne!(acc, Register::parameter(i));
            assert_ne!(acc, Register::local(i));
        }
    }
}
