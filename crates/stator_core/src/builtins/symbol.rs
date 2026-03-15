//! ECMAScript §20.4 `Symbol` built-in.
//!
//! Provides the global symbol registry, well-known symbol constants, and
//! helper functions used by [`super::install_globals`] to expose the `Symbol`
//! constructor and its static methods to JavaScript code.
//!
//! # Architecture
//!
//! [`JsValue::Symbol(u64)`][crate::objects::value::JsValue::Symbol] stores
//! only an opaque 64-bit identifier.  Descriptions and the
//! `Symbol.for()` / `Symbol.keyFor()` global registry live in a
//! thread-local [`SymbolRegistry`] so that the `JsValue` enum stays small
//! and `Copy`-friendly.
//!
//! Well-known symbols (e.g. `Symbol.iterator`) are assigned fixed IDs in the
//! range `1..=20` and their descriptions are pre-registered the first time
//! any symbol API is called.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// ─────────────────────────────────────────────────────────────────────────────
// Well-known symbol IDs (fixed, never change)
// ─────────────────────────────────────────────────────────────────────────────

/// `Symbol.iterator` — used by `for..of`, spread, destructuring, etc.
pub const SYMBOL_ITERATOR: u64 = 1;
/// `Symbol.toPrimitive` — customises `ToPrimitive` conversion.
pub const SYMBOL_TO_PRIMITIVE: u64 = 2;
/// `Symbol.hasInstance` — customises `instanceof`.
pub const SYMBOL_HAS_INSTANCE: u64 = 3;
/// `Symbol.toStringTag` — customises `Object.prototype.toString`.
pub const SYMBOL_TO_STRING_TAG: u64 = 4;
/// `Symbol.isConcatSpreadable` — customises `Array.prototype.concat`.
pub const SYMBOL_IS_CONCAT_SPREADABLE: u64 = 5;
/// `Symbol.species` — customises the constructor for derived objects.
pub const SYMBOL_SPECIES: u64 = 6;
/// `Symbol.match` — customises `String.prototype.match`.
pub const SYMBOL_MATCH: u64 = 7;
/// `Symbol.replace` — customises `String.prototype.replace`.
pub const SYMBOL_REPLACE: u64 = 8;
/// `Symbol.search` — customises `String.prototype.search`.
pub const SYMBOL_SEARCH: u64 = 9;
/// `Symbol.split` — customises `String.prototype.split`.
pub const SYMBOL_SPLIT: u64 = 10;
/// `Symbol.unscopables` — customises `with` environment bindings.
pub const SYMBOL_UNSCOPABLES: u64 = 11;
/// `Symbol.asyncIterator` — used by `for await..of`.
pub const SYMBOL_ASYNC_ITERATOR: u64 = 12;
/// `Symbol.matchAll` — customises `String.prototype.matchAll`.
pub const SYMBOL_MATCH_ALL: u64 = 13;
/// Well-known `Symbol.dispose` — sync disposal protocol.
pub const SYMBOL_DISPOSE: u64 = 14;
/// Well-known `Symbol.asyncDispose` — async disposal protocol.
pub const SYMBOL_ASYNC_DISPOSE: u64 = 15;

/// First ID available for user-created symbols (everything below is reserved).
const FIRST_USER_SYMBOL_ID: u64 = 100;

/// Global atomic counter for generating unique symbol IDs.
static NEXT_SYMBOL_ID: AtomicU64 = AtomicU64::new(FIRST_USER_SYMBOL_ID);

// ─────────────────────────────────────────────────────────────────────────────
// Thread-local registry
// ─────────────────────────────────────────────────────────────────────────────

/// Per-thread symbol metadata store.
///
/// Maps symbol IDs to their optional description, and maintains the
/// two-way mapping for the `Symbol.for()` / `Symbol.keyFor()` global
/// registry.
struct SymbolRegistry {
    /// `id → description` for every symbol created in this thread.
    descriptions: HashMap<u64, Option<String>>,
    /// `key → id` for the global `Symbol.for()` registry.
    for_registry: HashMap<String, u64>,
    /// `id → key` (reverse of `for_registry`).
    for_reverse: HashMap<u64, String>,
}

impl SymbolRegistry {
    fn new() -> Self {
        let mut reg = Self {
            descriptions: HashMap::new(),
            for_registry: HashMap::new(),
            for_reverse: HashMap::new(),
        };
        // Pre-register well-known symbols with their canonical descriptions.
        reg.descriptions
            .insert(SYMBOL_ITERATOR, Some("Symbol.iterator".into()));
        reg.descriptions
            .insert(SYMBOL_TO_PRIMITIVE, Some("Symbol.toPrimitive".into()));
        reg.descriptions
            .insert(SYMBOL_HAS_INSTANCE, Some("Symbol.hasInstance".into()));
        reg.descriptions
            .insert(SYMBOL_TO_STRING_TAG, Some("Symbol.toStringTag".into()));
        reg.descriptions.insert(
            SYMBOL_IS_CONCAT_SPREADABLE,
            Some("Symbol.isConcatSpreadable".into()),
        );
        reg.descriptions
            .insert(SYMBOL_SPECIES, Some("Symbol.species".into()));
        reg.descriptions
            .insert(SYMBOL_MATCH, Some("Symbol.match".into()));
        reg.descriptions
            .insert(SYMBOL_REPLACE, Some("Symbol.replace".into()));
        reg.descriptions
            .insert(SYMBOL_SEARCH, Some("Symbol.search".into()));
        reg.descriptions
            .insert(SYMBOL_SPLIT, Some("Symbol.split".into()));
        reg.descriptions
            .insert(SYMBOL_UNSCOPABLES, Some("Symbol.unscopables".into()));
        reg.descriptions
            .insert(SYMBOL_ASYNC_ITERATOR, Some("Symbol.asyncIterator".into()));
        reg.descriptions
            .insert(SYMBOL_MATCH_ALL, Some("Symbol.matchAll".into()));
        reg.descriptions
            .insert(SYMBOL_DISPOSE, Some("Symbol.dispose".into()));
        reg.descriptions
            .insert(SYMBOL_ASYNC_DISPOSE, Some("Symbol.asyncDispose".into()));
        reg
    }
}

thread_local! {
    static REGISTRY: RefCell<SymbolRegistry> = RefCell::new(SymbolRegistry::new());
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Create a brand-new unique symbol, optionally with a description.
///
/// Each call to this function returns a distinct ID, even when the same
/// description is passed.  This implements `Symbol()` / `Symbol("desc")`.
pub fn symbol_create(description: Option<String>) -> u64 {
    let id = NEXT_SYMBOL_ID.fetch_add(1, Ordering::Relaxed);
    REGISTRY.with(|r| {
        r.borrow_mut().descriptions.insert(id, description);
    });
    id
}

/// ECMAScript §20.4.2.2 `Symbol.for(key)`.
///
/// Returns the symbol associated with `key` in the global symbol registry,
/// creating one if it does not yet exist.
pub fn symbol_for(key: &str) -> u64 {
    REGISTRY.with(|r| {
        let mut reg = r.borrow_mut();
        if let Some(&id) = reg.for_registry.get(key) {
            return id;
        }
        let id = NEXT_SYMBOL_ID.fetch_add(1, Ordering::Relaxed);
        reg.descriptions.insert(id, Some(key.to_owned()));
        reg.for_registry.insert(key.to_owned(), id);
        reg.for_reverse.insert(id, key.to_owned());
        id
    })
}

/// ECMAScript §20.4.2.6 `Symbol.keyFor(sym)`.
///
/// Returns the key string if `sym` was created via `Symbol.for()`, or
/// `None` otherwise.
pub fn symbol_key_for(id: u64) -> Option<String> {
    REGISTRY.with(|r| r.borrow().for_reverse.get(&id).cloned())
}

/// Retrieve the description of a symbol (if it has one).
///
/// Used to implement `Symbol.prototype.description` and the
/// `Symbol.prototype.toString()` representation.
pub fn symbol_description(id: u64) -> Option<String> {
    REGISTRY.with(|r| r.borrow().descriptions.get(&id).cloned().flatten())
}

/// Map a well-known symbol ID to its internal `@@name` property key.
///
/// Returns `None` for user-created symbols that have no canonical `@@`
/// property key.
pub fn well_known_symbol_to_key(id: u64) -> Option<&'static str> {
    match id {
        SYMBOL_ITERATOR => Some("@@iterator"),
        SYMBOL_TO_PRIMITIVE => Some("@@toPrimitive"),
        SYMBOL_HAS_INSTANCE => Some("@@hasInstance"),
        SYMBOL_TO_STRING_TAG => Some("@@toStringTag"),
        SYMBOL_IS_CONCAT_SPREADABLE => Some("@@isConcatSpreadable"),
        SYMBOL_SPECIES => Some("@@species"),
        SYMBOL_MATCH => Some("@@match"),
        SYMBOL_REPLACE => Some("@@replace"),
        SYMBOL_SEARCH => Some("@@search"),
        SYMBOL_SPLIT => Some("@@split"),
        SYMBOL_UNSCOPABLES => Some("@@unscopables"),
        SYMBOL_ASYNC_ITERATOR => Some("@@asyncIterator"),
        SYMBOL_MATCH_ALL => Some("@@matchAll"),
        SYMBOL_DISPOSE => Some("@@dispose"),
        SYMBOL_ASYNC_DISPOSE => Some("@@asyncDispose"),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_unique_ids() {
        let a = symbol_create(None);
        let b = symbol_create(None);
        assert_ne!(a, b);
    }

    #[test]
    fn create_with_description() {
        let id = symbol_create(Some("mySymbol".into()));
        assert_eq!(symbol_description(id), Some("mySymbol".into()));
    }

    #[test]
    fn create_without_description() {
        let id = symbol_create(None);
        assert_eq!(symbol_description(id), None);
    }

    #[test]
    fn symbol_for_returns_same_id() {
        let a = symbol_for("shared");
        let b = symbol_for("shared");
        assert_eq!(a, b);
    }

    #[test]
    fn symbol_for_different_keys() {
        let a = symbol_for("key_a");
        let b = symbol_for("key_b");
        assert_ne!(a, b);
    }

    #[test]
    fn key_for_returns_key() {
        let id = symbol_for("hello");
        assert_eq!(symbol_key_for(id), Some("hello".into()));
    }

    #[test]
    fn key_for_returns_none_for_non_registry_symbol() {
        let id = symbol_create(Some("not in registry".into()));
        assert_eq!(symbol_key_for(id), None);
    }

    #[test]
    fn well_known_symbol_descriptions() {
        assert_eq!(
            symbol_description(SYMBOL_ITERATOR),
            Some("Symbol.iterator".into())
        );
        assert_eq!(
            symbol_description(SYMBOL_TO_PRIMITIVE),
            Some("Symbol.toPrimitive".into())
        );
        assert_eq!(
            symbol_description(SYMBOL_HAS_INSTANCE),
            Some("Symbol.hasInstance".into())
        );
        assert_eq!(
            symbol_description(SYMBOL_TO_STRING_TAG),
            Some("Symbol.toStringTag".into())
        );
        assert_eq!(
            symbol_description(SYMBOL_ASYNC_ITERATOR),
            Some("Symbol.asyncIterator".into())
        );
    }

    #[test]
    fn well_known_symbols_not_in_for_registry() {
        // Well-known symbols should NOT be in the Symbol.for() registry.
        assert_eq!(symbol_key_for(SYMBOL_ITERATOR), None);
        assert_eq!(symbol_key_for(SYMBOL_TO_PRIMITIVE), None);
    }

    #[test]
    fn all_well_known_ids_are_distinct() {
        let ids = [
            SYMBOL_ITERATOR,
            SYMBOL_TO_PRIMITIVE,
            SYMBOL_HAS_INSTANCE,
            SYMBOL_TO_STRING_TAG,
            SYMBOL_IS_CONCAT_SPREADABLE,
            SYMBOL_SPECIES,
            SYMBOL_MATCH,
            SYMBOL_REPLACE,
            SYMBOL_SEARCH,
            SYMBOL_SPLIT,
            SYMBOL_UNSCOPABLES,
            SYMBOL_ASYNC_ITERATOR,
            SYMBOL_MATCH_ALL,
        ];
        let mut sorted = ids.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(ids.len(), sorted.len());
    }

    #[test]
    fn all_well_known_descriptions() {
        let cases: &[(u64, &str)] = &[
            (SYMBOL_ITERATOR, "Symbol.iterator"),
            (SYMBOL_TO_PRIMITIVE, "Symbol.toPrimitive"),
            (SYMBOL_HAS_INSTANCE, "Symbol.hasInstance"),
            (SYMBOL_TO_STRING_TAG, "Symbol.toStringTag"),
            (SYMBOL_IS_CONCAT_SPREADABLE, "Symbol.isConcatSpreadable"),
            (SYMBOL_SPECIES, "Symbol.species"),
            (SYMBOL_MATCH, "Symbol.match"),
            (SYMBOL_REPLACE, "Symbol.replace"),
            (SYMBOL_SEARCH, "Symbol.search"),
            (SYMBOL_SPLIT, "Symbol.split"),
            (SYMBOL_UNSCOPABLES, "Symbol.unscopables"),
            (SYMBOL_ASYNC_ITERATOR, "Symbol.asyncIterator"),
            (SYMBOL_MATCH_ALL, "Symbol.matchAll"),
        ];
        for &(id, expected_desc) in cases {
            assert_eq!(
                symbol_description(id),
                Some(expected_desc.to_string()),
                "description mismatch for id={id}"
            );
        }
    }

    #[test]
    fn user_symbols_do_not_overlap_well_known() {
        let user = symbol_create(Some("user".into()));
        assert!(user >= FIRST_USER_SYMBOL_ID);
    }

    #[test]
    fn symbol_for_description_is_key() {
        let id = symbol_for("globalKey");
        assert_eq!(symbol_description(id), Some("globalKey".into()));
    }

    #[test]
    fn symbol_for_empty_key() {
        let id = symbol_for("");
        assert_eq!(symbol_key_for(id), Some("".into()));
        assert_eq!(symbol_description(id), Some("".into()));
    }

    #[test]
    fn symbol_create_empty_description() {
        let id = symbol_create(Some("".into()));
        assert_eq!(symbol_description(id), Some("".into()));
    }
}
