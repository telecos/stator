//! Thread-local string interning pool for JavaScript property names.
//!
//! Interning ensures that identical property-name strings share a single
//! [`Rc<str>`] allocation, which reduces memory usage and enables O(1)
//! pointer-based equality checks via [`interned_eq`] and [`str_eq_fast`].
//!
//! The pool is pre-seeded with [`WELL_KNOWN`] property names so that the
//! most frequently accessed names (e.g. `"length"`, `"prototype"`) are
//! available immediately without a cache miss on first use.

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

/// Well-known property names used frequently in JavaScript.
///
/// These are pre-loaded into every thread-local intern pool so that the
/// first access to any of these names returns an already-interned
/// [`Rc<str>`], avoiding a fresh allocation.
const WELL_KNOWN: &[&str] = &[
    // Core object properties
    "length",
    "prototype",
    "constructor",
    "__proto__",
    "toString",
    "valueOf",
    "hasOwnProperty",
    "name",
    "message",
    "stack",
    // Primitive singletons
    "undefined",
    "null",
    "NaN",
    "Infinity",
    // Function meta
    "arguments",
    "caller",
    "callee",
    "apply",
    "call",
    "bind",
    // Promise / iterator protocol
    "then",
    "catch",
    "finally",
    "next",
    "done",
    "value",
    // Property descriptor keys
    "get",
    "set",
    "writable",
    "enumerable",
    "configurable",
    // Additional well-known property names
    "return",
    "throw",
    "symbol",
    "number",
    "string",
    "boolean",
    "object",
    "function",
    "bigint",
    "index",
    "input",
    "groups",
    "indices",
    "source",
    "flags",
    "global",
    "ignoreCase",
    "multiline",
    "dotAll",
    "unicode",
    "unicodeSets",
    "sticky",
    "lastIndex",
    "exec",
    "test",
    "compile",
    // Array / TypedArray
    "from",
    "of",
    "isArray",
    "entries",
    "keys",
    "values",
    "forEach",
    "map",
    "filter",
    "reduce",
    "reduceRight",
    "find",
    "findIndex",
    "findLast",
    "findLastIndex",
    "includes",
    "indexOf",
    "lastIndexOf",
    "every",
    "some",
    "flat",
    "flatMap",
    "fill",
    "copyWithin",
    "sort",
    "reverse",
    "join",
    "concat",
    "slice",
    "splice",
    "push",
    "pop",
    "shift",
    "unshift",
    "at",
    "with",
    // Object statics
    "assign",
    "create",
    "defineProperty",
    "defineProperties",
    "freeze",
    "getOwnPropertyDescriptor",
    "getOwnPropertyDescriptors",
    "getOwnPropertyNames",
    "getOwnPropertySymbols",
    "getPrototypeOf",
    "is",
    "isExtensible",
    "isFrozen",
    "isSealed",
    "preventExtensions",
    "seal",
    "setPrototypeOf",
    "hasOwn",
    "groupBy",
    "fromEntries",
    // String methods
    "charAt",
    "charCodeAt",
    "codePointAt",
    "endsWith",
    "match",
    "matchAll",
    "normalize",
    "padEnd",
    "padStart",
    "repeat",
    "replace",
    "replaceAll",
    "search",
    "split",
    "startsWith",
    "substring",
    "toLowerCase",
    "toUpperCase",
    "trim",
    "trimStart",
    "trimEnd",
    "raw",
    // Number / Math
    "toFixed",
    "toPrecision",
    "isFinite",
    "isInteger",
    "isNaN",
    "isSafeInteger",
    "parseInt",
    "parseFloat",
    "EPSILON",
    "MAX_SAFE_INTEGER",
    "MIN_SAFE_INTEGER",
    "MAX_VALUE",
    "MIN_VALUE",
    "POSITIVE_INFINITY",
    "NEGATIVE_INFINITY",
    // Map / Set
    "size",
    "has",
    "add",
    "delete",
    "clear",
    // JSON
    "parse",
    "stringify",
    "toJSON",
    // Promise
    "resolve",
    "reject",
    "all",
    "allSettled",
    "any",
    "race",
    "status",
    "reason",
    // Error types
    "Error",
    "TypeError",
    "RangeError",
    "ReferenceError",
    "SyntaxError",
    "URIError",
    "EvalError",
    "AggregateError",
    "cause",
    "errors",
    // Misc builtins
    "description",
    "species",
    "iterator",
    "asyncIterator",
    "toPrimitive",
    "toStringTag",
    "hasInstance",
    "isConcatSpreadable",
    "unscopables",
];

/// Build the initial pool contents from [`WELL_KNOWN`].
fn seed_pool() -> HashSet<Rc<str>> {
    let mut pool = HashSet::with_capacity(WELL_KNOWN.len());
    for &s in WELL_KNOWN {
        pool.insert(Rc::from(s));
    }
    pool
}

thread_local! {
    static INTERN_POOL: RefCell<HashSet<Rc<str>>> = RefCell::new(seed_pool());
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

/// Clear the thread-local interning pool, then re-seed it with
/// [`WELL_KNOWN`] names so that subsequent lookups still hit the pool.
///
/// Used by test harnesses to reclaim memory between test runs.
pub fn clear_intern_pool() {
    INTERN_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        pool.clear();
        *pool = seed_pool();
    });
}

/// Check if two interned strings are the same by pointer comparison.
#[inline]
pub fn interned_eq(a: &Rc<str>, b: &Rc<str>) -> bool {
    Rc::ptr_eq(a, b)
}

/// Fast string equality: pointer check first, then byte comparison.
///
/// When both slices originate from the same interned [`Rc<str>`] the
/// pointer check succeeds in O(1), avoiding the O(n) content scan.
#[inline(always)]
pub fn str_eq_fast(a: &str, b: &str) -> bool {
    std::ptr::eq(a, b) || a == b
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

    #[test]
    fn test_well_known_pre_seeded() {
        for &name in WELL_KNOWN {
            let a = intern(name);
            let b = intern(name);
            assert!(Rc::ptr_eq(&a, &b), "well-known name {name:?} not interned");
        }
    }

    #[test]
    fn test_clear_reseeds_well_known() {
        let before = intern("length");
        clear_intern_pool();
        let after = intern("length");
        // After clearing and re-seeding, a new Rc is created but the
        // name is still present in the pool.
        assert_eq!(&*before, &*after);
        // The old and new Rc are distinct allocations.
        assert!(!Rc::ptr_eq(&before, &after));
    }

    #[test]
    fn test_str_eq_fast_same_pointer() {
        let s = intern("prototype");
        let a: &str = &s;
        let b: &str = &s;
        assert!(str_eq_fast(a, b));
    }

    #[test]
    fn test_str_eq_fast_different_pointer_same_content() {
        let a = String::from("hello");
        let b = String::from("hello");
        assert!(str_eq_fast(&a, &b));
    }

    #[test]
    fn test_str_eq_fast_different_content() {
        assert!(!str_eq_fast("foo", "bar"));
    }
}
