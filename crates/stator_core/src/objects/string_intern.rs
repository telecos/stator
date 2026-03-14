//! Thread-local string interning pool for JavaScript property names.
//!
//! Interning ensures that identical property-name strings share a single
//! [`Rc<str>`] allocation, which reduces memory usage and enables O(1)
//! pointer-based equality checks via [`interned_eq`].

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

thread_local! {
    static INTERN_POOL: RefCell<HashSet<Rc<str>>> = RefCell::new(HashSet::new());
}

/// Intern a string, returning a shared `Rc<str>` reference.
/// If the string is already interned, returns the existing reference.
/// This allows O(1) pointer comparison for string equality.
pub fn intern(s: &str) -> Rc<str> {
    INTERN_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if let Some(existing) = pool.get(s) {
            Rc::clone(existing)
        } else {
            let rc: Rc<str> = s.into();
            pool.insert(Rc::clone(&rc));
            rc
        }
    })
}

/// Clear the thread-local interning pool.
///
/// Used by test harnesses to reclaim memory between test runs.
pub fn clear_intern_pool() {
    INTERN_POOL.with(|pool| pool.borrow_mut().clear());
}

/// Check if two interned strings are the same by pointer comparison.
#[inline]
pub fn interned_eq(a: &Rc<str>, b: &Rc<str>) -> bool {
    Rc::ptr_eq(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intern_returns_same_rc() {
        let a = intern("hello");
        let b = intern("hello");
        assert!(Rc::ptr_eq(&a, &b));
    }

    #[test]
    fn test_intern_different_strings() {
        let a = intern("foo");
        let b = intern("bar");
        assert!(!Rc::ptr_eq(&a, &b));
    }

    #[test]
    fn test_interned_eq() {
        let a = intern("test");
        let b = intern("test");
        let c = intern("other");
        assert!(interned_eq(&a, &b));
        assert!(!interned_eq(&a, &c));
    }
}
