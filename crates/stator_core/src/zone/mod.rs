//! Zone allocator for compiler temporaries.
//!
//! A [`Zone`] is a bump-pointer region allocator backed by [`bumpalo`].
//! Objects allocated inside a zone are valid for the lifetime of the zone;
//! all memory is freed in bulk when the [`Zone`] is dropped.
//!
//! # Example
//!
//! ```
//! use stator_core::zone::Zone;
//!
//! let zone = Zone::new();
//! let x: &u64 = zone.alloc(42_u64);
//! assert_eq!(*x, 42);
//! ```

use bumpalo::Bump;

/// A bump-pointer region allocator for compiler temporaries.
///
/// All allocations made through [`Zone::alloc`] are tied to the lifetime of
/// the `Zone`. When the `Zone` is dropped, every allocation is freed at once
/// without running individual destructors (the same semantics as `bumpalo`).
///
/// This makes zones ideal for short-lived compiler passes where you want
/// arena-speed allocation and zero per-object deallocation overhead.
pub struct Zone {
    bump: Bump,
}

impl Zone {
    /// Create a new, empty `Zone`.
    pub fn new() -> Self {
        Self { bump: Bump::new() }
    }

    /// Allocate `value` inside the zone and return a reference to it.
    ///
    /// The returned reference is valid for the lifetime of the `Zone`.
    /// Destructors are **not** run when the zone is dropped; if `T` has a
    /// non-trivial `Drop` impl that you must not skip, do not allocate it
    /// here.
    pub fn alloc<T>(&self, value: T) -> &T {
        self.bump.alloc(value)
    }
}

impl Default for Zone {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::Zone;

    #[test]
    fn test_alloc_single_value() {
        let zone = Zone::new();
        let r: &u32 = zone.alloc(99_u32);
        assert_eq!(*r, 99);
    }

    #[test]
    fn test_alloc_many_objects() {
        let zone = Zone::new();
        let count = 10_000_usize;
        let refs: Vec<&usize> = (0..count).map(|i| zone.alloc(i)).collect();
        for (i, r) in refs.iter().enumerate() {
            assert_eq!(**r, i, "value at index {i} was corrupted");
        }
    }

    #[test]
    fn test_alloc_struct() {
        #[derive(Debug, PartialEq)]
        struct Point {
            x: f64,
            y: f64,
        }

        let zone = Zone::new();
        let p = zone.alloc(Point { x: 1.5, y: -2.5 });
        assert_eq!(p.x, 1.5);
        assert_eq!(p.y, -2.5);
    }

    #[test]
    fn test_alloc_string_slices() {
        let zone = Zone::new();
        let a = zone.alloc("hello");
        let b = zone.alloc("world");
        assert_eq!(*a, "hello");
        assert_eq!(*b, "world");
    }

    #[test]
    fn test_default_is_new() {
        let zone: Zone = Zone::default();
        let v = zone.alloc(42_i32);
        assert_eq!(*v, 42);
    }

    #[test]
    fn test_alloc_references_remain_valid() {
        let zone = Zone::new();
        let r1 = zone.alloc(1_u64);
        let r2 = zone.alloc(2_u64);
        let r3 = zone.alloc(3_u64);
        // Ensure allocating more objects doesn't invalidate earlier refs.
        assert_eq!(*r1, 1);
        assert_eq!(*r2, 2);
        assert_eq!(*r3, 3);
    }
}
