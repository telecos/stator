//! Feedback vectors and inline-cache state for the Stator VM.
//!
//! Every compiled function carries a [`FeedbackMetadata`] that describes the
//! type and count of feedback slots allocated by the compiler.  At runtime a
//! [`FeedbackVector`] is created from that metadata; the interpreter reads and
//! writes [`InlineCacheState`] values into the vector to guide adaptive
//! optimisations such as type specialisation and inline caching.
//!
//! # Design
//!
//! ```text
//! Compile time:
//!   FunctionCompiler → alloc_slot(kind) → FeedbackMetadata
//!                                               │
//!                                        stored in BytecodeArray
//!
//! Runtime:
//!   FeedbackMetadata → FeedbackVector::new(metadata)
//!   interpreter reads/writes InlineCacheState per slot index
//! ```
//!
//! # Example
//!
//! ```
//! use stator_jse::bytecode::feedback::{
//!     FeedbackMetadata, FeedbackSlotKind, FeedbackVector, InlineCacheState,
//! };
//!
//! // Compiler allocates slots.
//! let metadata = FeedbackMetadata::new(vec![
//!     FeedbackSlotKind::Call,
//!     FeedbackSlotKind::LoadProperty,
//! ]);
//! assert_eq!(metadata.slot_count(), 2);
//!
//! // Runtime creates a vector from that metadata.
//! let mut vector = FeedbackVector::new(&metadata);
//! assert_eq!(vector.get_state(0), Some(InlineCacheState::Uninitialized));
//!
//! // Interpreter updates state after the first execution.
//! vector.set_state(0, InlineCacheState::Monomorphic);
//! assert_eq!(vector.get_state(0), Some(InlineCacheState::Monomorphic));
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// FeedbackSlotKind
// ─────────────────────────────────────────────────────────────────────────────

/// The semantic purpose of a single feedback vector slot.
///
/// Each slot kind corresponds to one (or more) bytecode instructions that
/// perform a speculative or adaptive operation.  The kind is fixed at compile
/// time and recorded in [`FeedbackMetadata`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedbackSlotKind {
    /// A function call site (`CallAnyReceiver`, `CallProperty`, etc.).
    Call,
    /// A named property load (`LdaNamedProperty`).
    LoadProperty,
    /// A named property store (`StaNamedProperty`, `DefineNamedOwnProperty`).
    StoreProperty,
    /// A keyed (computed) property load (`LdaKeyedProperty`).
    KeyedLoadProperty,
    /// A keyed (computed) property store (`StaKeyedProperty`).
    KeyedStoreProperty,
    /// A binary arithmetic or bitwise operation (`Add`, `Sub`, `Mul`, …).
    BinaryOp,
    /// A comparison operation (`TestEqual`, `TestLessThan`, …).
    Compare,
    /// A `for-in` enumeration helper (`ForInPrepare`, `ForInNext`).
    ForIn,
    /// The `typeof` unary operator.
    TypeOf,
    /// A closure creation site (`CreateClosure`).
    CreateClosure,
    /// A global variable load (`LdaGlobal`).
    LoadGlobal,
    /// A global variable store (`StaGlobal`).
    StoreGlobal,
    /// An `instanceof` operator.
    InstanceOf,
    /// An increment or decrement operation (`Inc`, `Dec`).
    BinaryOpInc,
    /// A unary negate or bitwise-not operation (`Negate`, `BitwiseNot`).
    UnaryOp,
    /// An object, array, or regexp literal creation site
    /// (`CreateObjectLiteral`, `CreateArrayLiteral`, `CreateRegExpLiteral`).
    Literal,
    /// A getter or setter definition site
    /// (`DefineGetterProperty`, `DefineSetterProperty`, etc.).
    DefineAccessor,
}

// ─────────────────────────────────────────────────────────────────────────────
// InlineCacheState
// ─────────────────────────────────────────────────────────────────────────────

/// The inline-cache state of a single feedback slot.
///
/// The state starts at [`Uninitialized`](InlineCacheState::Uninitialized) and
/// transitions forward as the interpreter observes type diversity at the
/// corresponding instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum InlineCacheState {
    /// The slot has never been executed.
    Uninitialized,
    /// The slot has observed exactly one receiver/operand type (fast path).
    Monomorphic,
    /// The slot has observed 2–4 distinct types (moderate polymorphism).
    Polymorphic,
    /// The slot has observed 5 or more types; all speculation is abandoned.
    Megamorphic,
}

// ─────────────────────────────────────────────────────────────────────────────
// FeedbackMetadata
// ─────────────────────────────────────────────────────────────────────────────

/// Compile-time description of all feedback slots for a single function.
///
/// A `FeedbackMetadata` is produced by the bytecode compiler and stored inside
/// [`crate::bytecode::bytecode_array::BytecodeArray`].  At runtime a
/// [`FeedbackVector`] is created from it to hold the mutable
/// [`InlineCacheState`] values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeedbackMetadata {
    /// Ordered list of slot kinds, one entry per allocated slot.
    slot_kinds: Vec<FeedbackSlotKind>,
}

impl FeedbackMetadata {
    /// Create a `FeedbackMetadata` from a pre-built list of slot kinds.
    pub fn new(slot_kinds: Vec<FeedbackSlotKind>) -> Self {
        Self { slot_kinds }
    }

    /// Create an empty `FeedbackMetadata` (no feedback slots).
    pub fn empty() -> Self {
        Self {
            slot_kinds: Vec::new(),
        }
    }

    /// The total number of feedback slots.
    pub fn slot_count(&self) -> u32 {
        self.slot_kinds.len() as u32
    }

    /// Return the [`FeedbackSlotKind`] for the given zero-based slot index, or
    /// `None` if the index is out of range.
    pub fn kind_of(&self, slot: u32) -> Option<FeedbackSlotKind> {
        self.slot_kinds.get(slot as usize).copied()
    }

    /// The full ordered slice of slot kinds.
    pub fn slot_kinds(&self) -> &[FeedbackSlotKind] {
        &self.slot_kinds
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FeedbackVector
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime feedback vector for a single function activation.
///
/// A `FeedbackVector` is created from a [`FeedbackMetadata`] at runtime (once
/// per function object, not once per call).  The interpreter reads and writes
/// [`InlineCacheState`] entries to record what types it has seen at each
/// instruction site.
///
/// # Slot transitions
///
/// Slots start at [`InlineCacheState::Uninitialized`] and can only move
/// *forward* through the ordering `Uninitialized → Monomorphic → Polymorphic →
/// Megamorphic`.  Use [`FeedbackVector::transition`] to advance a slot while
/// respecting this invariant.
#[derive(Debug, Clone)]
pub struct FeedbackVector {
    /// The static kind of each slot (mirrors the metadata).
    slot_kinds: Vec<FeedbackSlotKind>,
    /// The mutable IC state of each slot.
    states: Vec<InlineCacheState>,
}

impl FeedbackVector {
    /// Create a new `FeedbackVector` from compile-time metadata.
    ///
    /// All slots start at [`InlineCacheState::Uninitialized`].
    pub fn new(metadata: &FeedbackMetadata) -> Self {
        let count = metadata.slot_kinds.len();
        Self {
            slot_kinds: metadata.slot_kinds.clone(),
            states: vec![InlineCacheState::Uninitialized; count],
        }
    }

    /// The total number of slots in this vector.
    pub fn slot_count(&self) -> u32 {
        self.states.len() as u32
    }

    /// Return the [`FeedbackSlotKind`] for `slot`, or `None` if out of range.
    pub fn kind_of(&self, slot: u32) -> Option<FeedbackSlotKind> {
        self.slot_kinds.get(slot as usize).copied()
    }

    /// Return the current [`InlineCacheState`] for `slot`, or `None` if out of
    /// range.
    pub fn get_state(&self, slot: u32) -> Option<InlineCacheState> {
        self.states.get(slot as usize).copied()
    }

    /// Unconditionally set the [`InlineCacheState`] for `slot`.
    ///
    /// Returns `true` if the slot existed (state was updated), `false`
    /// otherwise.
    pub fn set_state(&mut self, slot: u32, state: InlineCacheState) -> bool {
        match self.states.get_mut(slot as usize) {
            Some(s) => {
                *s = state;
                true
            }
            None => false,
        }
    }

    /// Advance the slot to `new_state` only if `new_state` is strictly higher
    /// in the ordering than the current state.
    ///
    /// Returns `true` if the state was advanced, `false` otherwise (including
    /// when the slot index is out of range).
    pub fn transition(&mut self, slot: u32, new_state: InlineCacheState) -> bool {
        match self.states.get_mut(slot as usize) {
            Some(s) if new_state > *s => {
                *s = new_state;
                true
            }
            _ => false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metadata() -> FeedbackMetadata {
        FeedbackMetadata::new(vec![
            FeedbackSlotKind::Call,
            FeedbackSlotKind::LoadProperty,
            FeedbackSlotKind::BinaryOp,
        ])
    }

    // ── FeedbackMetadata ────────────────────────────────────────────────────

    #[test]
    fn test_metadata_slot_count() {
        let m = make_metadata();
        assert_eq!(m.slot_count(), 3);
    }

    #[test]
    fn test_metadata_kind_of() {
        let m = make_metadata();
        assert_eq!(m.kind_of(0), Some(FeedbackSlotKind::Call));
        assert_eq!(m.kind_of(1), Some(FeedbackSlotKind::LoadProperty));
        assert_eq!(m.kind_of(2), Some(FeedbackSlotKind::BinaryOp));
        assert_eq!(m.kind_of(3), None);
    }

    #[test]
    fn test_metadata_empty() {
        let m = FeedbackMetadata::empty();
        assert_eq!(m.slot_count(), 0);
        assert_eq!(m.kind_of(0), None);
        assert!(m.slot_kinds().is_empty());
    }

    // ── FeedbackVector ──────────────────────────────────────────────────────

    #[test]
    fn test_vector_initial_state_uninitialized() {
        let m = make_metadata();
        let v = FeedbackVector::new(&m);
        assert_eq!(v.slot_count(), 3);
        for i in 0..3 {
            assert_eq!(v.get_state(i), Some(InlineCacheState::Uninitialized));
        }
    }

    #[test]
    fn test_vector_out_of_range() {
        let m = make_metadata();
        let v = FeedbackVector::new(&m);
        assert_eq!(v.get_state(3), None);
        assert_eq!(v.kind_of(3), None);
    }

    #[test]
    fn test_vector_set_state() {
        let m = make_metadata();
        let mut v = FeedbackVector::new(&m);
        assert!(v.set_state(0, InlineCacheState::Monomorphic));
        assert_eq!(v.get_state(0), Some(InlineCacheState::Monomorphic));
        // Out-of-range returns false.
        assert!(!v.set_state(99, InlineCacheState::Megamorphic));
    }

    #[test]
    fn test_vector_transition_advances_forward() {
        let m = FeedbackMetadata::new(vec![FeedbackSlotKind::BinaryOp]);
        let mut v = FeedbackVector::new(&m);

        assert!(v.transition(0, InlineCacheState::Monomorphic));
        assert_eq!(v.get_state(0), Some(InlineCacheState::Monomorphic));

        assert!(v.transition(0, InlineCacheState::Polymorphic));
        assert_eq!(v.get_state(0), Some(InlineCacheState::Polymorphic));

        assert!(v.transition(0, InlineCacheState::Megamorphic));
        assert_eq!(v.get_state(0), Some(InlineCacheState::Megamorphic));
    }

    #[test]
    fn test_vector_transition_does_not_go_backward() {
        let m = FeedbackMetadata::new(vec![FeedbackSlotKind::Compare]);
        let mut v = FeedbackVector::new(&m);
        v.set_state(0, InlineCacheState::Megamorphic);

        // Attempting to "transition" to a lower state is ignored.
        assert!(!v.transition(0, InlineCacheState::Monomorphic));
        assert_eq!(v.get_state(0), Some(InlineCacheState::Megamorphic));
    }

    #[test]
    fn test_vector_kind_of() {
        let m = make_metadata();
        let v = FeedbackVector::new(&m);
        assert_eq!(v.kind_of(0), Some(FeedbackSlotKind::Call));
        assert_eq!(v.kind_of(1), Some(FeedbackSlotKind::LoadProperty));
        assert_eq!(v.kind_of(2), Some(FeedbackSlotKind::BinaryOp));
    }

    #[test]
    fn test_vector_empty_metadata() {
        let m = FeedbackMetadata::empty();
        let v = FeedbackVector::new(&m);
        assert_eq!(v.slot_count(), 0);
        assert_eq!(v.get_state(0), None);
    }

    #[test]
    fn test_inline_cache_state_ordering() {
        use InlineCacheState::*;
        assert!(Uninitialized < Monomorphic);
        assert!(Monomorphic < Polymorphic);
        assert!(Polymorphic < Megamorphic);
    }

    // ── FeedbackSlotKind coverage ───────────────────────────────────────────

    #[test]
    fn test_all_slot_kind_variants_round_trip() {
        // Every FeedbackSlotKind variant should survive a metadata round-trip.
        let all_kinds = vec![
            FeedbackSlotKind::Call,
            FeedbackSlotKind::LoadProperty,
            FeedbackSlotKind::StoreProperty,
            FeedbackSlotKind::KeyedLoadProperty,
            FeedbackSlotKind::KeyedStoreProperty,
            FeedbackSlotKind::BinaryOp,
            FeedbackSlotKind::Compare,
            FeedbackSlotKind::ForIn,
            FeedbackSlotKind::TypeOf,
            FeedbackSlotKind::CreateClosure,
            FeedbackSlotKind::LoadGlobal,
            FeedbackSlotKind::StoreGlobal,
            FeedbackSlotKind::InstanceOf,
            FeedbackSlotKind::BinaryOpInc,
            FeedbackSlotKind::UnaryOp,
            FeedbackSlotKind::Literal,
            FeedbackSlotKind::DefineAccessor,
        ];
        let metadata = FeedbackMetadata::new(all_kinds.clone());
        assert_eq!(metadata.slot_count(), 17);
        for (i, &expected) in all_kinds.iter().enumerate() {
            assert_eq!(metadata.kind_of(i as u32), Some(expected));
        }
        // Beyond the end is None.
        assert_eq!(metadata.kind_of(17), None);
    }

    #[test]
    fn test_metadata_slot_kinds_nonempty() {
        let m = make_metadata();
        let kinds = m.slot_kinds();
        assert_eq!(kinds.len(), 3);
        assert_eq!(kinds[0], FeedbackSlotKind::Call);
        assert_eq!(kinds[1], FeedbackSlotKind::LoadProperty);
        assert_eq!(kinds[2], FeedbackSlotKind::BinaryOp);
    }

    #[test]
    fn test_metadata_equality() {
        let a = FeedbackMetadata::new(vec![FeedbackSlotKind::Call, FeedbackSlotKind::Compare]);
        let b = FeedbackMetadata::new(vec![FeedbackSlotKind::Call, FeedbackSlotKind::Compare]);
        let c = FeedbackMetadata::new(vec![FeedbackSlotKind::Call]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ── Transition edge cases ───────────────────────────────────────────────

    #[test]
    fn test_transition_same_state_returns_false() {
        let m = FeedbackMetadata::new(vec![FeedbackSlotKind::BinaryOp]);
        let mut v = FeedbackVector::new(&m);

        // Transition to the current state (Uninitialized → Uninitialized) is
        // a no-op and returns false.
        assert!(!v.transition(0, InlineCacheState::Uninitialized));
        assert_eq!(v.get_state(0), Some(InlineCacheState::Uninitialized));

        v.set_state(0, InlineCacheState::Polymorphic);
        // Transition to the same state returns false.
        assert!(!v.transition(0, InlineCacheState::Polymorphic));
        assert_eq!(v.get_state(0), Some(InlineCacheState::Polymorphic));
    }

    #[test]
    fn test_transition_out_of_range_returns_false() {
        let m = FeedbackMetadata::new(vec![FeedbackSlotKind::Call]);
        let mut v = FeedbackVector::new(&m);
        assert!(!v.transition(99, InlineCacheState::Megamorphic));
    }

    #[test]
    fn test_set_state_allows_downgrade() {
        // Unlike `transition`, `set_state` permits arbitrary state changes.
        let m = FeedbackMetadata::new(vec![FeedbackSlotKind::Compare]);
        let mut v = FeedbackVector::new(&m);
        v.set_state(0, InlineCacheState::Megamorphic);
        assert!(v.set_state(0, InlineCacheState::Uninitialized));
        assert_eq!(v.get_state(0), Some(InlineCacheState::Uninitialized));
    }

    #[test]
    fn test_full_transition_chain() {
        // Verify the canonical Uninitialized → Mono → Poly → Mega path.
        let m = FeedbackMetadata::new(vec![FeedbackSlotKind::LoadProperty]);
        let mut v = FeedbackVector::new(&m);

        assert_eq!(v.get_state(0), Some(InlineCacheState::Uninitialized));

        assert!(v.transition(0, InlineCacheState::Monomorphic));
        assert_eq!(v.get_state(0), Some(InlineCacheState::Monomorphic));

        assert!(v.transition(0, InlineCacheState::Polymorphic));
        assert_eq!(v.get_state(0), Some(InlineCacheState::Polymorphic));

        assert!(v.transition(0, InlineCacheState::Megamorphic));
        assert_eq!(v.get_state(0), Some(InlineCacheState::Megamorphic));

        // Already at Megamorphic — any transition should return false.
        assert!(!v.transition(0, InlineCacheState::Megamorphic));
        assert!(!v.transition(0, InlineCacheState::Polymorphic));
    }

    #[test]
    fn test_vector_slot_count_matches_metadata() {
        for n in [0usize, 1, 5, 100] {
            let kinds = vec![FeedbackSlotKind::BinaryOp; n];
            let m = FeedbackMetadata::new(kinds);
            let v = FeedbackVector::new(&m);
            assert_eq!(v.slot_count(), n as u32);
            assert_eq!(v.slot_count(), m.slot_count());
        }
    }

    #[test]
    fn test_vector_set_state_all_slots() {
        // Verify every slot can be independently set.
        let m = FeedbackMetadata::new(vec![
            FeedbackSlotKind::Call,
            FeedbackSlotKind::LoadProperty,
            FeedbackSlotKind::Compare,
        ]);
        let mut v = FeedbackVector::new(&m);
        v.set_state(0, InlineCacheState::Monomorphic);
        v.set_state(1, InlineCacheState::Polymorphic);
        v.set_state(2, InlineCacheState::Megamorphic);

        assert_eq!(v.get_state(0), Some(InlineCacheState::Monomorphic));
        assert_eq!(v.get_state(1), Some(InlineCacheState::Polymorphic));
        assert_eq!(v.get_state(2), Some(InlineCacheState::Megamorphic));
    }
}
