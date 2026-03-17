//! JavaScript `RegExp` object.
//!
//! This module provides [`JsRegExp`], which wraps the [`regress`] ECMAScript
//! regular expression engine and exposes the core built-in operations defined
//! by the ECMAScript specification:
//!
//! * [`JsRegExp::test`]   — ECMAScript `RegExp.prototype.test`
//! * [`JsRegExp::exec`]   — ECMAScript `RegExp.prototype.exec`
//! * [`JsRegExp::to_string`] — ECMAScript `RegExp.prototype.toString`
//! * [`JsRegExp::symbol_match`]   — ECMAScript `RegExp.prototype[Symbol.match]`
//! * [`JsRegExp::symbol_replace`] — ECMAScript `RegExp.prototype[Symbol.replace]`
//! * [`JsRegExp::symbol_search`]  — ECMAScript `RegExp.prototype[Symbol.search]`
//! * [`JsRegExp::symbol_split`]   — ECMAScript `RegExp.prototype[Symbol.split]`
//!
//! # Supported flags
//!
//! | Flag | Meaning |
//! |------|---------|
//! | `g` | Global — find all matches, advance `lastIndex` |
//! | `i` | Ignore case |
//! | `m` | Multiline — `^`/`$` match line boundaries |
//! | `s` | DotAll — `.` matches `\n` |
//! | `u` | Unicode — full Unicode matching |
//! | `v` | UnicodeSets — extended Unicode set operations |
//! | `y` | Sticky — match only from `lastIndex` |
//! | `d` | HasIndices — record match indices |
//!
//! # Named captures, lookbehind, and Unicode property escapes
//!
//! Named capture groups (`(?<name>...)`), lookbehind assertions (`(?<=...)`,
//! `(?<!...)`), and Unicode property escapes (`\p{...}`, `\P{...}`) are all
//! delegated to the `regress` crate, which implements ECMAScript 2018+ syntax.

use std::cell::Cell;
use std::collections::HashMap;

use bitflags::bitflags;
use regress::{Flags as RegressFlags, Regex};

use crate::error::{StatorError, StatorResult};

// ──────────────────────────────────────────────────────────────────────────────
// RegExpFlags
// ──────────────────────────────────────────────────────────────────────────────

bitflags! {
    /// ECMAScript `RegExp` flags.
    ///
    /// The engine tracks the `g`, `y`, and `d` flags itself because they
    /// affect iteration strategy rather than the underlying pattern compiler.
    /// The remaining flags (`i`, `m`, `s`, `u`, `v`) are forwarded to the
    /// `regress` backend.
    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    pub struct RegExpFlags: u8 {
        /// `g` — search for all matches; advance `lastIndex` between calls.
        const GLOBAL       = 0b0000_0001;
        /// `i` — ignore case.
        const IGNORE_CASE  = 0b0000_0010;
        /// `m` — multiline; `^`/`$` match at line boundaries.
        const MULTILINE    = 0b0000_0100;
        /// `s` — dotAll; `.` matches line terminators.
        const DOT_ALL      = 0b0000_1000;
        /// `u` — Unicode mode.
        const UNICODE      = 0b0001_0000;
        /// `v` — UnicodeSets mode (extends `u`).
        const UNICODE_SETS = 0b0010_0000;
        /// `y` — sticky; match only at `lastIndex`.
        const STICKY       = 0b0100_0000;
        /// `d` — hasIndices; expose match indices.
        const HAS_INDICES  = 0b1000_0000;
    }
}

impl RegExpFlags {
    /// Parses a flags string (e.g., `"gim"`) into [`RegExpFlags`].
    ///
    /// Returns `Err(StatorError::SyntaxError)` on duplicate or unknown flags,
    /// or if `u` and `v` are both present.
    pub fn parse(flags: &str) -> StatorResult<Self> {
        let mut result = Self::empty();
        for ch in flags.chars() {
            let bit = match ch {
                'g' => Self::GLOBAL,
                'i' => Self::IGNORE_CASE,
                'm' => Self::MULTILINE,
                's' => Self::DOT_ALL,
                'u' => Self::UNICODE,
                'v' => Self::UNICODE_SETS,
                'y' => Self::STICKY,
                'd' => Self::HAS_INDICES,
                _ => {
                    return Err(StatorError::SyntaxError(format!(
                        "Invalid regular expression flags: '{ch}'"
                    )));
                }
            };
            if result.contains(bit) {
                return Err(StatorError::SyntaxError(format!(
                    "Duplicate regular expression flag: '{ch}'"
                )));
            }
            result |= bit;
        }
        if result.contains(Self::UNICODE) && result.contains(Self::UNICODE_SETS) {
            return Err(StatorError::SyntaxError(
                "Regular expression flags 'u' and 'v' cannot be combined".to_string(),
            ));
        }
        Ok(result)
    }

    /// Returns the canonical flags string, sorted in ECMAScript order:
    /// `d`, `g`, `i`, `m`, `s`, `u`, `v`, `y`.
    pub fn to_flags_string(self) -> String {
        let mut s = String::with_capacity(8);
        if self.contains(Self::HAS_INDICES) {
            s.push('d');
        }
        if self.contains(Self::GLOBAL) {
            s.push('g');
        }
        if self.contains(Self::IGNORE_CASE) {
            s.push('i');
        }
        if self.contains(Self::MULTILINE) {
            s.push('m');
        }
        if self.contains(Self::DOT_ALL) {
            s.push('s');
        }
        if self.contains(Self::UNICODE) {
            s.push('u');
        }
        if self.contains(Self::UNICODE_SETS) {
            s.push('v');
        }
        if self.contains(Self::STICKY) {
            s.push('y');
        }
        s
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// RegExpMatch
// ──────────────────────────────────────────────────────────────────────────────

/// The result of a successful [`JsRegExp::exec`] call.
///
/// Mirrors the array-like result object produced by ECMAScript
/// `RegExp.prototype.exec`:
///
/// * Index `0` is the full match string.
/// * Indices `1..` are the numbered capture groups (`None` means the group did
///   not participate in the match).
/// * `index` is the byte offset in the input string where the full match
///   starts.
/// * `input` is the original input string.
/// * `named_groups` contains any named capture group values keyed by name.
/// * `indices` is populated when the `d` (hasIndices) flag is set.
#[derive(Debug, Clone, PartialEq)]
pub struct RegExpMatch {
    /// The full matched substring.
    pub matched: String,
    /// Numbered capture groups (1-indexed; the outer `Vec` is 0-indexed).
    pub captures: Vec<Option<String>>,
    /// Named capture groups (`name → value`).
    ///
    /// All named groups defined in the pattern are present.  Groups that did
    /// not participate in the match have a `None` value (surfaced as
    /// `undefined` in JavaScript).
    pub named_groups: HashMap<String, Option<String>>,
    /// Byte offset of the match start in the input string.
    pub index: usize,
    /// The input string against which the pattern was matched.
    pub input: String,
    /// Match indices for the `d` flag (`[start, end]` pairs).
    ///
    /// The first element is the full match range; subsequent elements are
    /// the capture group ranges (`None` for groups that did not participate).
    /// Only populated when [`RegExpFlags::HAS_INDICES`] is set.
    pub indices: Option<MatchIndices>,
}

/// Byte-offset index pairs for the `/d` (hasIndices) flag.
///
/// Each element is `(start, end)` as a byte-offset pair into the input string.
/// The first element corresponds to the full match; subsequent elements are the
/// capture groups (with `None` for groups that did not participate).
#[derive(Debug, Clone, PartialEq)]
pub struct MatchIndices {
    /// `[start, end]` pairs: index 0 = full match, 1.. = capture groups.
    pub pairs: Vec<Option<(usize, usize)>>,
    /// Named group indices keyed by group name.
    pub groups: HashMap<String, (usize, usize)>,
}

// ──────────────────────────────────────────────────────────────────────────────
// JsRegExp
// ──────────────────────────────────────────────────────────────────────────────

/// A JavaScript `RegExp` object.
///
/// `JsRegExp` combines a raw `pattern` string with [`RegExpFlags`] and a
/// compiled [`Regex`] from the `regress` engine.  The `lastIndex` property
/// required by the `g` and `y` semantics is stored as a [`Cell`] so that
/// `test` and `exec` can update it even through a shared reference.
///
/// # Example
///
/// ```rust
/// use stator_core::objects::regexp::JsRegExp;
///
/// let re = JsRegExp::new(r"\d+", "g").unwrap();
/// assert!(re.test("foo 42 bar"));
/// assert_eq!(re.to_string(), r"/\d+/g");
/// ```
pub struct JsRegExp {
    /// The raw pattern string (as supplied to the constructor).
    pattern: String,
    /// Parsed flags.
    flags: RegExpFlags,
    /// Compiled regex from the `regress` backend.
    compiled: Regex,
    /// Current `lastIndex` for global / sticky iteration.
    ///
    /// Uses [`Cell`] so that stateful methods (`test`, `exec`) can advance
    /// the index without requiring `&mut self`.
    last_index: Cell<usize>,
}

impl std::fmt::Debug for JsRegExp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsRegExp")
            .field("pattern", &self.pattern)
            .field("flags", &self.flags)
            .field("last_index", &self.last_index.get())
            .finish()
    }
}

impl JsRegExp {
    // ── Constructor ──────────────────────────────────────────────────────────

    /// Creates a new `JsRegExp` from a `pattern` string and a `flags` string.
    ///
    /// Returns `Err(StatorError::SyntaxError)` if:
    /// * `flags` contains unknown or duplicate characters, or
    /// * `pattern` is not valid ECMAScript regular expression syntax.
    ///
    /// # Example
    ///
    /// ```rust
    /// use stator_core::objects::regexp::JsRegExp;
    ///
    /// let re = JsRegExp::new(r"(?<year>\d{4})-(?<month>\d{2})", "u").unwrap();
    /// assert!(re.test("2024-07"));
    /// ```
    pub fn new(pattern: &str, flags: &str) -> StatorResult<Self> {
        let flags = RegExpFlags::parse(flags)?;
        let regress_flags = build_regress_flags(flags);
        // Regex compilation in `regress` can recurse deeply for complex
        // alternation / back-reference patterns.
        let compiled = stacker::maybe_grow(256 * 1024, 4 * 1024 * 1024, || {
            Regex::with_flags(pattern, regress_flags)
        })
        .map_err(|e| {
            StatorError::SyntaxError(format!("Invalid regular expression: /{pattern}/: {e}"))
        })?;
        Ok(Self {
            pattern: pattern.to_string(),
            flags,
            compiled,
            last_index: Cell::new(0),
        })
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    /// Returns the raw source pattern string.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Returns the ECMAScript `source` text for this regexp.
    ///
    /// This preserves the original pattern text, substitutes `(?:)` for the
    /// empty pattern, and escapes `/` so the result can appear inside
    /// `/source/flags`.
    pub fn source_text(&self) -> String {
        if self.pattern.is_empty() {
            return "(?:)".to_string();
        }

        let mut source = String::with_capacity(self.pattern.len());
        for ch in self.pattern.chars() {
            if ch == '/' {
                source.push('\\');
            }
            source.push(ch);
        }
        source
    }

    /// Returns the [`RegExpFlags`] for this regexp.
    pub fn flags(&self) -> RegExpFlags {
        self.flags
    }

    /// Returns the current `lastIndex` value.
    ///
    /// This is only meaningful when the `g` (global) or `y` (sticky) flag is
    /// set.
    pub fn last_index(&self) -> usize {
        self.last_index.get()
    }

    /// Sets the `lastIndex` value.
    pub fn set_last_index(&self, index: usize) {
        self.last_index.set(index);
    }

    // ── Core operations ──────────────────────────────────────────────────────

    /// ECMAScript `RegExp.prototype.test(string)`.
    ///
    /// Returns `true` if the pattern matches anywhere in `input` (or at
    /// `lastIndex` when the `g`/`y` flag is set).
    ///
    /// When the `g` or `y` flag is present, `lastIndex` is advanced past the
    /// match on success and reset to `0` on failure, matching ECMAScript
    /// semantics.
    pub fn test(&self, input: &str) -> bool {
        self.exec(input).is_some()
    }

    /// ECMAScript `RegExp.prototype.exec(string)`.
    ///
    /// Returns the first match as a [`RegExpMatch`], or `None` if the pattern
    /// does not match.
    ///
    /// For global (`g`) and sticky (`y`) regexps the search starts at
    /// `lastIndex`; after a successful match `lastIndex` is updated to the
    /// byte position immediately after the match.  After a failed match
    /// `lastIndex` is reset to `0`.
    pub fn exec(&self, input: &str) -> Option<RegExpMatch> {
        // Regex matching in the `regress` crate can recurse deeply for
        // pathological patterns.  Ensure we have ample stack.
        stacker::maybe_grow(256 * 1024, 4 * 1024 * 1024, || self.exec_inner(input))
    }

    fn exec_inner(&self, input: &str) -> Option<RegExpMatch> {
        let is_stateful = self
            .flags
            .intersects(RegExpFlags::GLOBAL | RegExpFlags::STICKY);
        let start = if is_stateful {
            self.last_index.get()
        } else {
            0
        };

        // Guard: if start is beyond the input length, reset and return None.
        if start > input.len() {
            if is_stateful {
                self.last_index.set(0);
            }
            return None;
        }

        let m = if self.flags.contains(RegExpFlags::STICKY) {
            // Sticky: only match at exactly `start`.
            self.compiled
                .find_from(input, start)
                .next()
                .filter(|m| m.start() == start)
        } else {
            self.compiled.find_from(input, start).next()
        };

        match m {
            None => {
                if is_stateful {
                    self.last_index.set(0);
                }
                None
            }
            Some(mat) => {
                if is_stateful {
                    self.last_index.set(mat.end());
                }
                Some(build_match(
                    input,
                    &mat,
                    self.flags.contains(RegExpFlags::HAS_INDICES),
                ))
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Display — ECMAScript RegExp.prototype.toString()
// ──────────────────────────────────────────────────────────────────────────────

impl std::fmt::Display for JsRegExp {
    /// Returns the ECMAScript `RegExp.prototype.toString()` representation:
    /// `/pattern/flags`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "/{}/{}",
            self.source_text(),
            self.flags.to_flags_string()
        )
    }
}

impl JsRegExp {
    // ── Symbol methods ───────────────────────────────────────────────────────

    /// ECMAScript `RegExp.prototype[Symbol.match](string)`.
    ///
    /// * **Non-global**: returns `Some` with the first [`RegExpMatch`], or
    ///   `None`.
    /// * **Global**: resets `lastIndex` to `0`, then collects every
    ///   non-overlapping matched string into a `Vec<String>`.  Returns `None`
    ///   if there are no matches. `lastIndex` is reset to `0` before returning.
    pub fn symbol_match(&self, input: &str) -> Option<SymbolMatchResult> {
        if !self.flags.contains(RegExpFlags::GLOBAL) {
            // Non-global: behave like exec, which also preserves sticky
            // semantics when /y is present.
            self.exec(input).map(SymbolMatchResult::Single)
        } else {
            // Global: collect all matches and reset lastIndex when finished.
            self.last_index.set(0);
            let mut matches: Vec<String> = Vec::new();
            loop {
                let start = self.last_index.get();
                if start > input.len() {
                    self.last_index.set(0);
                    break;
                }
                let m = if self.flags.contains(RegExpFlags::STICKY) {
                    self.compiled
                        .find_from(input, start)
                        .next()
                        .filter(|m| m.start() == start)
                } else {
                    self.compiled.find_from(input, start).next()
                };
                match m {
                    None => {
                        self.last_index.set(0);
                        break;
                    }
                    Some(mat) => {
                        let end = mat.end();
                        matches.push(input[mat.range()].to_string());
                        self.last_index.set(advance_after_match(input, start, end));
                    }
                }
            }
            if matches.is_empty() {
                None
            } else {
                Some(SymbolMatchResult::All(matches))
            }
        }
    }

    /// ECMAScript `RegExp.prototype[Symbol.replace](string, replacement)`.
    ///
    /// Replaces matches in `input` with `replacement`.  The replacement
    /// supports the following substitution patterns:
    ///
    /// | Pattern | Replacement |
    /// |---------|-------------|
    /// | `$$`    | Literal `$` |
    /// | `$&`    | Entire match |
    /// | `` $` `` | Portion of string before the match |
    /// | `$'`    | Portion of string after the match |
    /// | `$n`    | `n`-th capture group (1-indexed) |
    /// | `$<name>` | Named capture group |
    ///
    /// When the `g` flag is set all matches are replaced; otherwise only the
    /// first match is replaced.
    pub fn symbol_replace(&self, input: &str, replacement: &str) -> String {
        let global = self.flags.contains(RegExpFlags::GLOBAL);
        if global {
            self.last_index.set(0);
            let mut result = String::new();
            let mut last_end = 0_usize;

            loop {
                let start = self.last_index.get();
                if start > input.len() {
                    self.last_index.set(0);
                    break;
                }
                let m = if self.flags.contains(RegExpFlags::STICKY) {
                    self.compiled
                        .find_from(input, start)
                        .next()
                        .filter(|m| m.start() == start)
                } else {
                    self.compiled.find_from(input, start).next()
                };
                match m {
                    None => {
                        self.last_index.set(0);
                        break;
                    }
                    Some(mat) => {
                        let rm = build_match(input, &mat, false);
                        result.push_str(&input[last_end..rm.index]);
                        result.push_str(&apply_replacement(replacement, &rm, input));
                        let end = mat.end();
                        last_end = end;
                        self.last_index.set(advance_after_match(input, start, end));
                    }
                }
            }

            result.push_str(&input[last_end..]);
            return result;
        }

        if let Some(matched) = self.exec(input) {
            let end = matched.index + matched.matched.len();
            let mut result = String::new();
            result.push_str(&input[..matched.index]);
            result.push_str(&apply_replacement(replacement, &matched, input));
            result.push_str(&input[end..]);
            result
        } else {
            input.to_string()
        }
    }

    /// ECMAScript `RegExp.prototype[Symbol.search](string)`.
    ///
    /// Returns the byte index of the first match, or `-1` if no match is
    /// found.  `lastIndex` is always reset to `0` before and after the search.
    pub fn symbol_search(&self, input: &str) -> i64 {
        let saved = self.last_index.get();
        self.last_index.set(0);
        let result = self.exec(input).map_or(-1, |m| m.index as i64);
        self.last_index.set(saved);
        result
    }

    /// ECMAScript `RegExp.prototype[Symbol.split](string[, limit])`.
    ///
    /// Splits `input` around each match and returns the parts as a
    /// `Vec<Option<String>>`.  Capture groups are included in the result
    /// between the surrounding parts, matching ECMAScript semantics.
    /// Non-participating capture groups are represented as `None` (which
    /// should be surfaced as `undefined` in JavaScript).
    ///
    /// If `limit` is `Some(0)` an empty `Vec` is returned immediately.
    pub fn symbol_split(&self, input: &str, limit: Option<usize>) -> Vec<Option<String>> {
        let lim = limit.unwrap_or(usize::MAX);
        if lim == 0 {
            return Vec::new();
        }

        if input.is_empty() {
            let empty_match = self
                .compiled
                .find(input)
                .is_some_and(|m| m.start() == 0 && m.end() == 0);
            return if empty_match {
                Vec::new()
            } else {
                vec![Some(String::new())]
            };
        }

        let mut parts: Vec<Option<String>> = Vec::new();
        let mut last_end = 0usize;
        let mut search_index = 0usize;

        // Per ECMAScript, @@split behaves like a sticky walk over the input:
        // a match is only consumed when it starts exactly at the current
        // search position. Otherwise we advance by one string element and try
        // again. This matters for zero-width matches and for regexps without
        // an explicit `y` flag.
        while search_index < input.len() {
            let matched = self
                .compiled
                .find_from(input, search_index)
                .next()
                .filter(|mat| mat.start() == search_index);

            let Some(mat) = matched else {
                search_index = advance_string_index(input, search_index);
                continue;
            };

            let match_end = mat.end();
            if match_end == last_end {
                search_index = advance_string_index(input, search_index);
                continue;
            }

            parts.push(Some(input[last_end..search_index].to_string()));
            if parts.len() >= lim {
                return parts;
            }
            for cap in &mat.captures {
                parts.push(cap.as_ref().map(|r| input[r.clone()].to_string()));
                if parts.len() >= lim {
                    return parts;
                }
            }
            last_end = match_end;
            search_index = last_end;
        }

        if parts.len() < lim {
            parts.push(Some(input[last_end..].to_string()));
        }
        parts
    }

    /// ECMAScript `RegExp.prototype[Symbol.matchAll](string)`.
    ///
    /// Returns all successive matches as a `Vec<RegExpMatch>`, producing the
    /// same kind of result objects that [`Self::exec`] would for each match.
    /// This is the underlying implementation for `String.prototype.matchAll`.
    ///
    /// The search starts from `self.last_index` so callers can set it before
    /// invoking this method to honour an existing `lastIndex` on a cloned
    /// regexp.
    pub fn symbol_match_all(&self, input: &str) -> Vec<RegExpMatch> {
        let has_indices = self.flags.contains(RegExpFlags::HAS_INDICES);
        let mut results = Vec::new();
        let mut start = self.last_index.get();
        loop {
            if start > input.len() {
                break;
            }
            let m = if self.flags.contains(RegExpFlags::STICKY) {
                self.compiled
                    .find_from(input, start)
                    .next()
                    .filter(|m| m.start() == start)
            } else {
                self.compiled.find_from(input, start).next()
            };
            match m {
                None => break,
                Some(mat) => {
                    let end = mat.end();
                    results.push(build_match(input, &mat, has_indices));
                    start = advance_after_match(input, start, end);
                }
            }
        }
        results
    }

    /// Create a clone of this regexp for use by `[Symbol.matchAll]`.
    ///
    /// Per §22.2.6.9 the clone has the same pattern and flags, and its
    /// `lastIndex` is set to `last_index`.  This avoids mutating the
    /// original regexp's state during iteration.
    pub fn clone_for_match_all(&self, last_index: usize) -> StatorResult<Self> {
        let flags_str = self.flags.to_flags_string();
        let cloned = Self::new(&self.pattern, &flags_str)?;
        cloned.last_index.set(last_index);
        Ok(cloned)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SymbolMatchResult
// ──────────────────────────────────────────────────────────────────────────────

/// The result type returned by [`JsRegExp::symbol_match`].
///
/// * [`Single`][SymbolMatchResult::Single] — non-global match, returns the
///   full exec result.
/// * [`All`][SymbolMatchResult::All] — global match, returns the list
///   of matched strings.
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolMatchResult {
    /// Result of a non-global `Symbol.match` call.
    Single(RegExpMatch),
    /// Result of a global `Symbol.match` call: all matched strings.
    All(Vec<String>),
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Builds the [`RegressFlags`] value that corresponds to a [`RegExpFlags`].
///
/// Note: `g`, `y`, and `d` flags are handled by the `JsRegExp` wrapper itself
/// and are not forwarded to `regress`.
fn build_regress_flags(f: RegExpFlags) -> RegressFlags {
    RegressFlags {
        icase: f.contains(RegExpFlags::IGNORE_CASE),
        multiline: f.contains(RegExpFlags::MULTILINE),
        dot_all: f.contains(RegExpFlags::DOT_ALL),
        unicode: f.contains(RegExpFlags::UNICODE) || f.contains(RegExpFlags::UNICODE_SETS),
        unicode_sets: f.contains(RegExpFlags::UNICODE_SETS),
        no_opt: false,
    }
}

/// Converts a [`regress::Match`] into a [`RegExpMatch`].
///
/// When `has_indices` is `true` the [`MatchIndices`] field is populated with
/// byte-offset pairs for the full match and each capture group.
fn build_match(input: &str, mat: &regress::Match, has_indices: bool) -> RegExpMatch {
    let matched = input[mat.range()].to_string();
    let index = mat.start();

    let captures: Vec<Option<String>> = mat
        .captures
        .iter()
        .map(|cap| cap.as_ref().map(|r| input[r.clone()].to_string()))
        .collect();

    let named_groups: HashMap<String, Option<String>> = mat
        .named_groups()
        .map(|(name, range)| (name.to_string(), range.map(|r| input[r].to_string())))
        .collect();

    let indices = if has_indices {
        let mut pairs = Vec::with_capacity(1 + mat.captures.len());
        pairs.push(Some((mat.start(), mat.end())));
        for cap in &mat.captures {
            pairs.push(cap.as_ref().map(|r| (r.start, r.end)));
        }
        let groups: HashMap<String, (usize, usize)> = mat
            .named_groups()
            .filter_map(|(name, range)| range.map(|r| (name.to_string(), (r.start, r.end))))
            .collect();
        Some(MatchIndices { pairs, groups })
    } else {
        None
    };

    RegExpMatch {
        matched,
        captures,
        named_groups,
        index,
        input: input.to_string(),
        indices,
    }
}

/// Applies ECMAScript replacement pattern substitutions.
///
/// Handles `$$`, `$&`, `` $` ``, `$'`, `$n` (1–99), and `$<name>`.
fn apply_replacement(replacement: &str, m: &RegExpMatch, input: &str) -> String {
    let mut out = String::new();
    let bytes = replacement.as_bytes();
    let mut i = 0;
    while i < replacement.len() {
        if bytes[i] == b'$' && i + 1 < bytes.len() {
            match bytes[i + 1] {
                b'$' => {
                    out.push('$');
                    i += 2;
                }
                b'&' => {
                    out.push_str(&m.matched);
                    i += 2;
                }
                b'`' => {
                    out.push_str(&input[..m.index]);
                    i += 2;
                }
                b'\'' => {
                    let after_start = m.index + m.matched.len();
                    if after_start <= input.len() {
                        out.push_str(&input[after_start..]);
                    }
                    i += 2;
                }
                b'<' => {
                    // Named capture: $<name>
                    if let Some(end) = replacement[i + 2..].find('>') {
                        let name = &replacement[i + 2..i + 2 + end];
                        if m.named_groups.is_empty() {
                            out.push_str(&replacement[i..i + 3 + end]);
                        } else if let Some(Some(val)) = m.named_groups.get(name) {
                            out.push_str(val);
                        }
                        i += 2 + end + 1; // skip $<name>
                    } else {
                        out.push('$');
                        i += 1;
                    }
                }
                b'0'..=b'9' => {
                    // Numbered capture: $n or $nn (up to 99).
                    let mut num = (bytes[i + 1] - b'0') as usize;
                    let mut consumed = 2;
                    if i + 2 < bytes.len()
                        && let Some(d) = (bytes[i + 2] as char).to_digit(10)
                    {
                        let two_digit = num * 10 + d as usize;
                        // Only use two digits if the two-digit index is valid.
                        if two_digit > 0 && two_digit <= m.captures.len() {
                            num = two_digit;
                            consumed = 3;
                        }
                    }
                    if num > 0 {
                        if let Some(Some(cap)) = m.captures.get(num - 1) {
                            out.push_str(cap);
                        }
                    } else {
                        out.push('$');
                        // bytes[i+1] is an ASCII digit, safe to cast.
                        out.push(bytes[i + 1] as char);
                    }
                    i += consumed;
                }
                _ => {
                    out.push('$');
                    i += 1;
                }
            }
        } else {
            // Copy the next Unicode scalar value from the replacement string.
            // Using char iteration avoids incorrect casts of multi-byte UTF-8
            // sequences to `char`.
            if let Some(ch) = replacement[i..].chars().next() {
                out.push(ch);
                i += ch.len_utf8();
            } else {
                break;
            }
        }
    }
    out
}

fn advance_after_match(input: &str, start: usize, end: usize) -> usize {
    if end > start {
        end
    } else {
        advance_string_index(input, start)
    }
}

fn advance_string_index(input: &str, index: usize) -> usize {
    if index >= input.len() {
        index.saturating_add(1)
    } else {
        input[index..]
            .chars()
            .next()
            .map_or(index.saturating_add(1), |ch| index + ch.len_utf8())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RegExpFlags::parse ────────────────────────────────────────────────────

    #[test]
    fn test_parse_empty_flags() {
        let f = RegExpFlags::parse("").unwrap();
        assert_eq!(f, RegExpFlags::empty());
    }

    #[test]
    fn test_parse_all_flags() {
        let f = RegExpFlags::parse("gimsyd").unwrap();
        assert!(f.contains(RegExpFlags::GLOBAL));
        assert!(f.contains(RegExpFlags::IGNORE_CASE));
        assert!(f.contains(RegExpFlags::MULTILINE));
        assert!(f.contains(RegExpFlags::DOT_ALL));
        assert!(f.contains(RegExpFlags::STICKY));
        assert!(f.contains(RegExpFlags::HAS_INDICES));
    }

    #[test]
    fn test_parse_unknown_flag_errors() {
        let err = RegExpFlags::parse("x").unwrap_err();
        assert!(matches!(err, StatorError::SyntaxError(_)));
    }

    #[test]
    fn test_parse_duplicate_flag_errors() {
        let err = RegExpFlags::parse("gg").unwrap_err();
        assert!(matches!(err, StatorError::SyntaxError(_)));
    }

    #[test]
    fn test_parse_uv_combined_errors() {
        let err = RegExpFlags::parse("uv").unwrap_err();
        assert!(matches!(err, StatorError::SyntaxError(_)));
    }

    // ── RegExpFlags::to_flags_string ──────────────────────────────────────────

    #[test]
    fn test_flags_string_order() {
        // ECMAScript canonical order: d g i m s u v y
        let f = RegExpFlags::parse("ymisgd").unwrap();
        assert_eq!(f.to_flags_string(), "dgimsy");
    }

    #[test]
    fn test_flags_string_single() {
        assert_eq!(RegExpFlags::parse("g").unwrap().to_flags_string(), "g");
        assert_eq!(RegExpFlags::parse("i").unwrap().to_flags_string(), "i");
        assert_eq!(RegExpFlags::parse("m").unwrap().to_flags_string(), "m");
        assert_eq!(RegExpFlags::parse("s").unwrap().to_flags_string(), "s");
        assert_eq!(RegExpFlags::parse("u").unwrap().to_flags_string(), "u");
        assert_eq!(RegExpFlags::parse("v").unwrap().to_flags_string(), "v");
        assert_eq!(RegExpFlags::parse("y").unwrap().to_flags_string(), "y");
        assert_eq!(RegExpFlags::parse("d").unwrap().to_flags_string(), "d");
    }

    // ── JsRegExp::new ─────────────────────────────────────────────────────────

    #[test]
    fn test_new_valid_pattern() {
        let re = JsRegExp::new(r"\d+", "g").unwrap();
        assert_eq!(re.pattern(), r"\d+");
        assert!(re.flags().contains(RegExpFlags::GLOBAL));
    }

    #[test]
    fn test_new_invalid_pattern_errors() {
        let err = JsRegExp::new("[invalid", "").unwrap_err();
        assert!(matches!(err, StatorError::SyntaxError(_)));
    }

    #[test]
    fn test_new_invalid_flag_errors() {
        let err = JsRegExp::new("a", "z").unwrap_err();
        assert!(matches!(err, StatorError::SyntaxError(_)));
    }

    // ── JsRegExp::to_string ───────────────────────────────────────────────────

    #[test]
    fn test_to_string_no_flags() {
        let re = JsRegExp::new("hello", "").unwrap();
        assert_eq!(re.to_string(), "/hello/");
    }

    #[test]
    fn test_to_string_with_flags() {
        let re = JsRegExp::new("foo", "gi").unwrap();
        assert_eq!(re.to_string(), "/foo/gi");
    }

    // ── JsRegExp::test ────────────────────────────────────────────────────────

    #[test]
    fn test_test_match() {
        let re = JsRegExp::new(r"\d+", "").unwrap();
        assert!(re.test("foo 42 bar"));
        assert!(!re.test("no numbers here"));
    }

    #[test]
    fn test_test_case_insensitive() {
        let re = JsRegExp::new("hello", "i").unwrap();
        assert!(re.test("Say HELLO World"));
        assert!(!re.test("say goodbye world"));
    }

    #[test]
    fn test_test_multiline() {
        let re = JsRegExp::new("^start", "m").unwrap();
        assert!(re.test("first line\nstart of second"));
        assert!(!re.test("first line\n  start with space"));
    }

    #[test]
    fn test_test_dot_all() {
        let re = JsRegExp::new("a.b", "s").unwrap();
        assert!(re.test("a\nb"));
        let re_no_s = JsRegExp::new("a.b", "").unwrap();
        assert!(!re_no_s.test("a\nb"));
    }

    // ── JsRegExp::exec ────────────────────────────────────────────────────────

    #[test]
    fn test_exec_no_match_returns_none() {
        let re = JsRegExp::new("xyz", "").unwrap();
        assert!(re.exec("hello world").is_none());
    }

    #[test]
    fn test_exec_simple_match() {
        let re = JsRegExp::new(r"(\d+)", "").unwrap();
        let m = re.exec("price 42 dollars").unwrap();
        assert_eq!(m.matched, "42");
        assert_eq!(m.index, 6);
        assert_eq!(m.captures, vec![Some("42".to_string())]);
    }

    #[test]
    fn test_exec_named_captures() {
        let re = JsRegExp::new(r"(?<year>\d{4})-(?<month>\d{2})-(?<day>\d{2})", "u").unwrap();
        let m = re.exec("today is 2024-07-15 ok").unwrap();
        assert_eq!(m.matched, "2024-07-15");
        assert_eq!(
            m.named_groups.get("year").and_then(|v| v.as_deref()),
            Some("2024")
        );
        assert_eq!(
            m.named_groups.get("month").and_then(|v| v.as_deref()),
            Some("07")
        );
        assert_eq!(
            m.named_groups.get("day").and_then(|v| v.as_deref()),
            Some("15")
        );
    }

    #[test]
    fn test_exec_lookbehind() {
        let re = JsRegExp::new(r"(?<=\$)\d+", "").unwrap();
        let m = re.exec("price $100").unwrap();
        assert_eq!(m.matched, "100");
        // Should NOT match when there's no $ before the digits.
        assert!(re.exec("price 100").is_none());
    }

    #[test]
    fn test_exec_negative_lookbehind() {
        let re = JsRegExp::new(r"(?<!\$)\d+", "").unwrap();
        let m = re.exec("100 dollars").unwrap();
        assert_eq!(m.matched, "100");
    }

    #[test]
    fn test_exec_unicode_flag() {
        let re = JsRegExp::new(r"\p{L}+", "u").unwrap();
        let m = re.exec("hello 42").unwrap();
        assert_eq!(m.matched, "hello");
    }

    // ── Global flag / lastIndex ───────────────────────────────────────────────

    #[test]
    fn test_global_last_index_advances() {
        let re = JsRegExp::new(r"\d+", "g").unwrap();
        assert_eq!(re.last_index(), 0);
        let m1 = re.exec("a1 b2 c3").unwrap();
        assert_eq!(m1.matched, "1");
        let m2 = re.exec("a1 b2 c3").unwrap();
        assert_eq!(m2.matched, "2");
        let m3 = re.exec("a1 b2 c3").unwrap();
        assert_eq!(m3.matched, "3");
        // After the last match there are no more matches; lastIndex resets.
        let m4 = re.exec("a1 b2 c3");
        assert!(m4.is_none());
        assert_eq!(re.last_index(), 0);
    }

    #[test]
    fn test_sticky_only_matches_at_last_index() {
        let re = JsRegExp::new(r"\d+", "y").unwrap();
        // "5" is at index 0.
        let m = re.exec("5 apples").unwrap();
        assert_eq!(m.matched, "5");
        // lastIndex is now 1; the next char is ' ', so no match.
        assert!(re.exec("5 apples").is_none());
        assert_eq!(re.last_index(), 0);
    }

    // ── Symbol.match ─────────────────────────────────────────────────────────

    #[test]
    fn test_symbol_match_non_global() {
        let re = JsRegExp::new(r"\d+", "").unwrap();
        let result = re.symbol_match("price 42 and 7").unwrap();
        if let SymbolMatchResult::Single(m) = result {
            assert_eq!(m.matched, "42");
        } else {
            panic!("expected Single");
        }
    }

    #[test]
    fn test_symbol_match_global_all() {
        let re = JsRegExp::new(r"\d+", "g").unwrap();
        let result = re.symbol_match("a1 b22 c333").unwrap();
        if let SymbolMatchResult::All(v) = result {
            assert_eq!(v, vec!["1", "22", "333"]);
        } else {
            panic!("expected All");
        }
    }

    #[test]
    fn test_symbol_match_no_match_returns_none() {
        let re = JsRegExp::new(r"\d+", "g").unwrap();
        assert!(re.symbol_match("no numbers").is_none());
    }

    // ── Symbol.replace ───────────────────────────────────────────────────────

    #[test]
    fn test_symbol_replace_first_match() {
        let re = JsRegExp::new(r"\d+", "").unwrap();
        assert_eq!(re.symbol_replace("foo 42 bar 7", "NUM"), "foo NUM bar 7");
    }

    #[test]
    fn test_symbol_replace_global() {
        let re = JsRegExp::new(r"\d+", "g").unwrap();
        assert_eq!(re.symbol_replace("a1 b2 c3", "N"), "aN bN cN");
    }

    #[test]
    fn test_symbol_replace_dollar_amp() {
        let re = JsRegExp::new(r"\d+", "").unwrap();
        assert_eq!(re.symbol_replace("price 42", "[$&]"), "price [42]");
    }

    #[test]
    fn test_symbol_replace_dollar_dollar() {
        let re = JsRegExp::new("x", "").unwrap();
        assert_eq!(re.symbol_replace("axb", "$$"), "a$b");
    }

    #[test]
    fn test_symbol_replace_capture_group() {
        let re = JsRegExp::new(r"(\d+)-(\d+)", "").unwrap();
        assert_eq!(re.symbol_replace("2024-07", "$2/$1"), "07/2024");
    }

    #[test]
    fn test_symbol_replace_named_capture() {
        let re = JsRegExp::new(r"(?<y>\d{4})-(?<m>\d{2})", "u").unwrap();
        assert_eq!(
            re.symbol_replace("date 2024-07 end", "$<m>/$<y>"),
            "date 07/2024 end"
        );
    }

    #[test]
    fn test_symbol_replace_before_after() {
        let re = JsRegExp::new("b", "").unwrap();
        // $` = before match ("a"), $' = after match ("c")
        // result = "a" (before) + "a|c" (replacement) + "c" (trailing) = "aa|cc"
        assert_eq!(re.symbol_replace("abc", "$`|$'"), "aa|cc");
    }

    // ── Symbol.search ─────────────────────────────────────────────────────────

    #[test]
    fn test_symbol_search_found() {
        let re = JsRegExp::new(r"\d+", "").unwrap();
        assert_eq!(re.symbol_search("foo 42 bar"), 4);
    }

    #[test]
    fn test_symbol_search_not_found() {
        let re = JsRegExp::new(r"\d+", "").unwrap();
        assert_eq!(re.symbol_search("no numbers"), -1);
    }

    #[test]
    fn test_symbol_search_resets_last_index() {
        let re = JsRegExp::new(r"\d+", "g").unwrap();
        re.set_last_index(5);
        let _ = re.symbol_search("foo 42 bar");
        assert_eq!(re.last_index(), 5); // restored
    }

    // ── Symbol.split ─────────────────────────────────────────────────────────

    #[test]
    fn test_symbol_split_basic() {
        let re = JsRegExp::new(",", "").unwrap();
        assert_eq!(
            re.symbol_split("a,b,c", None),
            vec![Some("a".into()), Some("b".into()), Some("c".into())]
        );
    }

    #[test]
    fn test_symbol_split_with_limit() {
        let re = JsRegExp::new(",", "").unwrap();
        assert_eq!(
            re.symbol_split("a,b,c,d", Some(2)),
            vec![Some("a".into()), Some("b".into())]
        );
    }

    #[test]
    fn test_symbol_split_zero_limit() {
        let re = JsRegExp::new(",", "").unwrap();
        assert_eq!(
            re.symbol_split("a,b,c", Some(0)),
            Vec::<Option<String>>::new()
        );
    }

    #[test]
    fn test_symbol_split_captures_included() {
        // ES spec: capture groups are included in the result.
        let re = JsRegExp::new(r"(\d+)", "").unwrap();
        assert_eq!(
            re.symbol_split("a1b2c", None),
            vec![
                Some("a".into()),
                Some("1".into()),
                Some("b".into()),
                Some("2".into()),
                Some("c".into())
            ]
        );
    }

    #[test]
    fn test_symbol_split_no_match_returns_whole_string() {
        let re = JsRegExp::new(r"\d+", "").unwrap();
        assert_eq!(re.symbol_split("abc", None), vec![Some("abc".into())]);
    }

    #[test]
    fn test_symbol_split_nonparticipating_capture_is_none() {
        let re = JsRegExp::new("-(x)?", "").unwrap();
        assert_eq!(
            re.symbol_split("a-b", None),
            vec![Some("a".into()), None, Some("b".into())]
        );
    }

    #[test]
    fn test_symbol_split_zero_width_returns_characters() {
        let re = JsRegExp::new(r"(?:)", "").unwrap();
        assert_eq!(
            re.symbol_split("ab", None),
            vec![Some("a".into()), Some("b".into())]
        );
    }

    #[test]
    fn test_symbol_split_scans_forward_sticky_style() {
        let re = JsRegExp::new("a", "").unwrap();
        assert_eq!(
            re.symbol_split("baab", None),
            vec![Some("b".into()), Some(String::new()), Some("b".into())]
        );
    }

    #[test]
    fn test_apply_replacement_named_capture_is_literal_without_named_groups() {
        let m = RegExpMatch {
            matched: "a".into(),
            captures: vec![],
            named_groups: HashMap::new(),
            index: 0,
            input: "a".into(),
            indices: None,
        };
        assert_eq!(apply_replacement("$<x>", &m, "a"), "$<x>");
    }

    // ── Unicode property escapes ──────────────────────────────────────────────

    #[test]
    fn test_unicode_property_escape_letter() {
        let re = JsRegExp::new(r"\p{L}", "u").unwrap();
        assert!(re.test("hello"));
        assert!(!re.test("123"));
    }

    #[test]
    fn test_unicode_property_escape_digit() {
        let re = JsRegExp::new(r"\p{N}", "u").unwrap();
        assert!(re.test("42"));
        assert!(!re.test("abc"));
    }

    // ── Named capture groups ─────────────────────────────────────────────

    #[test]
    fn test_named_capture_single_group() {
        let re = JsRegExp::new(r"(?<word>\w+)", "").unwrap();
        let m = re.exec("hello world").unwrap();
        assert_eq!(m.matched, "hello");
        assert_eq!(
            m.named_groups.get("word").and_then(|v| v.as_deref()),
            Some("hello")
        );
    }

    #[test]
    fn test_named_capture_nonparticipating_group() {
        let re = JsRegExp::new(r"(?<a>x)?(?<b>\d+)", "").unwrap();
        let m = re.exec("42").unwrap();
        assert_eq!(m.matched, "42");
        assert_eq!(m.named_groups.get("a").and_then(|v| v.as_deref()), None);
        assert_eq!(
            m.named_groups.get("b").and_then(|v| v.as_deref()),
            Some("42")
        );
        // The key "a" should still be present (with None value).
        assert!(m.named_groups.contains_key("a"));
    }

    #[test]
    fn test_named_capture_multiple_groups() {
        let re = JsRegExp::new(r"(?<first>\w+)\s(?<last>\w+)", "").unwrap();
        let m = re.exec("John Doe").unwrap();
        assert_eq!(
            m.named_groups.get("first").and_then(|v| v.as_deref()),
            Some("John")
        );
        assert_eq!(
            m.named_groups.get("last").and_then(|v| v.as_deref()),
            Some("Doe")
        );
    }

    #[test]
    fn test_named_and_numbered_captures_coexist() {
        let re = JsRegExp::new(r"(\d+)-(?<name>\w+)", "").unwrap();
        let m = re.exec("42-foo").unwrap();
        assert_eq!(m.captures[0], Some("42".to_string()));
        assert_eq!(m.captures[1], Some("foo".to_string()));
        assert_eq!(
            m.named_groups.get("name").and_then(|v| v.as_deref()),
            Some("foo")
        );
    }

    // ── Named backreferences \k<name> ────────────────────────────────────

    #[test]
    fn test_named_backreference_basic() {
        let re = JsRegExp::new(r"(?<tag>\w+)=\k<tag>", "").unwrap();
        let m = re.exec("abc=abc").unwrap();
        assert_eq!(m.matched, "abc=abc");
    }

    #[test]
    fn test_named_backreference_no_match_when_different() {
        let re = JsRegExp::new(r"(?<tag>\w+)=\k<tag>", "").unwrap();
        assert!(re.exec("abc=def").is_none());
    }

    #[test]
    fn test_named_backreference_with_flag_u() {
        let re = JsRegExp::new(r"(?<char>.)\k<char>", "u").unwrap();
        let m = re.exec("aabbcc").unwrap();
        assert_eq!(m.matched, "aa");
    }

    #[test]
    fn test_named_backreference_html_tag() {
        let re = JsRegExp::new(r"<(?<tag>\w+)>.*?</\k<tag>>", "").unwrap();
        let m = re.exec("<div>hello</div>").unwrap();
        assert_eq!(m.matched, "<div>hello</div>");
        assert!(re.exec("<div>hello</span>").is_none());
    }

    // ── String replace with named groups ─────────────────────────────────

    #[test]
    fn test_replace_named_capture_global() {
        let re = JsRegExp::new(r"(?<d>\d+)", "gu").unwrap();
        assert_eq!(re.symbol_replace("a1 b2 c3", "[$<d>]"), "a[1] b[2] c[3]");
    }

    #[test]
    fn test_replace_named_capture_missing_name() {
        let re = JsRegExp::new(r"(?<a>\d+)", "u").unwrap();
        // $<b> doesn't match any group — should produce empty string
        assert_eq!(re.symbol_replace("42", "$<b>"), "");
    }

    #[test]
    fn test_replace_named_capture_nonparticipating() {
        let re = JsRegExp::new(r"(?<a>x)?(?<b>\d+)", "").unwrap();
        // Group "a" didn't participate — $<a> produces empty string
        assert_eq!(re.symbol_replace("42", "$<a>-$<b>"), "-42");
    }

    #[test]
    fn test_replace_named_capture_unclosed_angle() {
        let re = JsRegExp::new(r"(?<a>\d+)", "").unwrap();
        // $< without closing > — literal $
        assert_eq!(re.symbol_replace("42", "$<a"), "$<a");
    }

    // ── /d flag (hasIndices) ─────────────────────────────────────────────

    #[test]
    fn test_has_indices_flag_parsed() {
        let re = JsRegExp::new("a", "d").unwrap();
        assert!(re.flags().contains(RegExpFlags::HAS_INDICES));
    }

    #[test]
    fn test_has_indices_basic() {
        let re = JsRegExp::new(r"\d+", "d").unwrap();
        let m = re.exec("abc 42 end").unwrap();
        let idx = m.indices.as_ref().unwrap();
        assert_eq!(idx.pairs[0], Some((4, 6)));
    }

    #[test]
    fn test_has_indices_capture_groups() {
        let re = JsRegExp::new(r"(\d+)-(\d+)", "d").unwrap();
        let m = re.exec("abc 12-34 end").unwrap();
        let idx = m.indices.as_ref().unwrap();
        assert_eq!(idx.pairs[0], Some((4, 9)));
        assert_eq!(idx.pairs[1], Some((4, 6)));
        assert_eq!(idx.pairs[2], Some((7, 9)));
    }

    #[test]
    fn test_has_indices_nonparticipating_group() {
        let re = JsRegExp::new(r"(x)?(\d+)", "d").unwrap();
        let m = re.exec("42").unwrap();
        let idx = m.indices.as_ref().unwrap();
        assert_eq!(idx.pairs[0], Some((0, 2)));
        assert_eq!(idx.pairs[1], None); // (x)? didn't participate
        assert_eq!(idx.pairs[2], Some((0, 2)));
    }

    #[test]
    fn test_has_indices_named_groups() {
        let re = JsRegExp::new(r"(?<year>\d{4})-(?<month>\d{2})", "du").unwrap();
        let m = re.exec("2024-07").unwrap();
        let idx = m.indices.as_ref().unwrap();
        assert_eq!(idx.groups.get("year"), Some(&(0, 4)));
        assert_eq!(idx.groups.get("month"), Some(&(5, 7)));
    }

    #[test]
    fn test_no_indices_without_d_flag() {
        let re = JsRegExp::new(r"\d+", "").unwrap();
        let m = re.exec("42").unwrap();
        assert!(m.indices.is_none());
    }

    // ── /v flag stub ─────────────────────────────────────────────────────

    #[test]
    fn test_v_flag_parsed() {
        let re = JsRegExp::new("a", "v").unwrap();
        assert!(re.flags().contains(RegExpFlags::UNICODE_SETS));
    }

    #[test]
    fn test_v_flag_in_flags_string() {
        let re = JsRegExp::new("a", "v").unwrap();
        assert_eq!(re.flags().to_flags_string(), "v");
    }

    #[test]
    fn test_v_flag_enables_unicode_matching() {
        // /v implies unicode mode — \p{L} should work
        let re = JsRegExp::new(r"\p{L}+", "v").unwrap();
        assert!(re.test("hello"));
        assert!(!re.test("123"));
    }

    #[test]
    fn test_v_and_u_cannot_combine() {
        let err = RegExpFlags::parse("uv").unwrap_err();
        assert!(matches!(err, StatorError::SyntaxError(_)));
    }

    #[test]
    fn test_v_flag_in_to_string() {
        let re = JsRegExp::new("abc", "gv").unwrap();
        assert_eq!(re.to_string(), "/abc/gv");
    }

    // ── RegExp.prototype.flags ordering ──────────────────────────────────

    #[test]
    fn test_flags_canonical_order_all() {
        let f = RegExpFlags::parse("ysmigd").unwrap();
        assert_eq!(f.to_flags_string(), "dgimsy");
    }

    #[test]
    fn test_flags_canonical_order_with_v() {
        let f = RegExpFlags::parse("yvgd").unwrap();
        assert_eq!(f.to_flags_string(), "dgvy");
    }

    #[test]
    fn test_flags_empty() {
        let f = RegExpFlags::parse("").unwrap();
        assert_eq!(f.to_flags_string(), "");
    }

    // ── matchAll with named groups ───────────────────────────────────────

    #[test]
    fn test_match_all_named_groups() {
        let re = JsRegExp::new(r"(?<num>\d+)", "g").unwrap();
        let results = re.symbol_match_all("a1 b22 c333");
        assert_eq!(results.len(), 3);
        assert_eq!(
            results[0]
                .named_groups
                .get("num")
                .and_then(|v| v.as_deref()),
            Some("1")
        );
        assert_eq!(
            results[2]
                .named_groups
                .get("num")
                .and_then(|v| v.as_deref()),
            Some("333")
        );
    }

    #[test]
    fn test_match_all_with_indices() {
        let re = JsRegExp::new(r"\d+", "gd").unwrap();
        let results = re.symbol_match_all("a1 b22");
        assert_eq!(results.len(), 2);
        let idx0 = results[0].indices.as_ref().unwrap();
        assert_eq!(idx0.pairs[0], Some((1, 2)));
        let idx1 = results[1].indices.as_ref().unwrap();
        assert_eq!(idx1.pairs[0], Some((4, 6)));
    }

    // ── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn test_named_group_empty_match() {
        let re = JsRegExp::new(r"(?<empty>)", "").unwrap();
        let m = re.exec("abc").unwrap();
        assert_eq!(
            m.named_groups.get("empty").and_then(|v| v.as_deref()),
            Some("")
        );
    }

    #[test]
    fn test_exec_no_named_groups_empty_map() {
        let re = JsRegExp::new(r"\d+", "").unwrap();
        let m = re.exec("42").unwrap();
        assert!(m.named_groups.is_empty());
    }

    #[test]
    fn test_symbol_replace_dollar_n_and_named_combined() {
        let re = JsRegExp::new(r"(\d+)-(?<w>\w+)", "").unwrap();
        assert_eq!(re.symbol_replace("42-foo", "$1=$<w>"), "42=foo");
    }

    #[test]
    fn test_source_text_empty_pattern() {
        let re = JsRegExp::new("", "").unwrap();
        assert_eq!(re.source_text(), "(?:)");
    }

    #[test]
    fn test_source_text_escapes_slash() {
        let re = JsRegExp::new("a/b", "").unwrap();
        assert_eq!(re.source_text(), r"a\/b");
    }

    #[test]
    fn test_global_empty_pattern_advances() {
        let re = JsRegExp::new("", "g").unwrap();
        let results = re.symbol_match_all("ab");
        // Empty pattern matches at every position: "", "", ""
        assert_eq!(results.len(), 3);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Advanced features — lookbehind, dotAll, Unicode property escapes,
    //                      named backreferences, flag combinations
    // ══════════════════════════════════════════════════════════════════════

    // ── Positive lookbehind (?<=...) ─────────────────────────────────────

    #[test]
    fn test_lookbehind_positive_dollar_sign() {
        let re = JsRegExp::new(r"(?<=\$)\d+", "").unwrap();
        let m = re.exec("price $100 and €200").unwrap();
        assert_eq!(m.matched, "100");
    }

    #[test]
    fn test_lookbehind_positive_word_boundary() {
        let re = JsRegExp::new(r"(?<=\bfoo)\w+", "").unwrap();
        let m = re.exec("foobar baz").unwrap();
        assert_eq!(m.matched, "bar");
    }

    #[test]
    fn test_lookbehind_positive_global_all() {
        let re = JsRegExp::new(r"(?<=@)\w+", "g").unwrap();
        let results = re.symbol_match_all("@alice and @bob");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].matched, "alice");
        assert_eq!(results[1].matched, "bob");
    }

    // ── Negative lookbehind (?<!...) ─────────────────────────────────────

    #[test]
    fn test_lookbehind_negative_no_dollar() {
        let re = JsRegExp::new(r"(?<!\$)\d+", "").unwrap();
        let m = re.exec("free 42 items").unwrap();
        assert_eq!(m.matched, "42");
    }

    #[test]
    fn test_lookbehind_negative_skips_prefixed() {
        let re = JsRegExp::new(r"(?<!un)happy", "").unwrap();
        assert!(re.test("happy day"));
        assert!(!re.test("unhappy day"));
    }

    #[test]
    fn test_lookbehind_negative_global() {
        let re = JsRegExp::new(r"(?<!#)\b\w+", "g").unwrap();
        let results = re.symbol_match_all("hello #world foo");
        // "hello" and "foo" should match; "world" is preceded by #
        let matched: Vec<&str> = results.iter().map(|m| m.matched.as_str()).collect();
        assert!(matched.contains(&"hello"));
        assert!(matched.contains(&"foo"));
    }

    // ── Lookbehind with captures ─────────────────────────────────────────

    #[test]
    fn test_lookbehind_with_capture_group() {
        let re = JsRegExp::new(r"(?<=(\d+)\s)\w+", "").unwrap();
        let m = re.exec("42 apples").unwrap();
        assert_eq!(m.matched, "apples");
        // The capture inside the lookbehind should be recorded
        assert_eq!(m.captures[0], Some("42".to_string()));
    }

    #[test]
    fn test_lookbehind_capture_numbering() {
        // Capture groups inside lookbehind are numbered before those outside
        let re = JsRegExp::new(r"(?<=(a)(b))cd", "").unwrap();
        let m = re.exec("abcd").unwrap();
        assert_eq!(m.matched, "cd");
        assert_eq!(m.captures[0], Some("a".to_string()));
        assert_eq!(m.captures[1], Some("b".to_string()));
    }

    // ── Lookbehind with quantifiers (variable-length) ────────────────────

    #[test]
    fn test_lookbehind_variable_length() {
        // regress supports variable-length lookbehind
        let re = JsRegExp::new(r"(?<=\d+)\s\w+", "").unwrap();
        let m = re.exec("123 abc").unwrap();
        assert_eq!(m.matched, " abc");
    }

    #[test]
    fn test_lookbehind_alternation_variable_length() {
        let re = JsRegExp::new(r"(?<=cat|hello)\s\w+", "").unwrap();
        let m = re.exec("hello world").unwrap();
        assert_eq!(m.matched, " world");
    }

    // ── dotAll flag (s) — dot matches newlines ───────────────────────────

    #[test]
    fn test_dotall_dot_matches_newline() {
        let re = JsRegExp::new("a.b", "s").unwrap();
        assert!(re.test("a\nb"));
        assert!(re.test("a\rb"));
        assert!(re.test("axb"));
    }

    #[test]
    fn test_dotall_off_dot_rejects_newline() {
        let re = JsRegExp::new("a.b", "").unwrap();
        assert!(!re.test("a\nb"));
        assert!(re.test("axb"));
    }

    #[test]
    fn test_dotall_multiline_interaction() {
        // /s and /m can coexist: dot matches newline, ^/$ match line boundaries
        let re = JsRegExp::new(r"^.+$", "sm").unwrap();
        let m = re.exec("line1\nline2").unwrap();
        // With /s, . matches \n, so ^.+$ can match across lines
        assert!(m.matched.contains('\n'));
    }

    #[test]
    fn test_dotall_flag_accessor() {
        let re = JsRegExp::new("a", "s").unwrap();
        assert!(re.flags().contains(RegExpFlags::DOT_ALL));
    }

    #[test]
    fn test_dotall_in_flags_string() {
        let re = JsRegExp::new("a", "gs").unwrap();
        assert_eq!(re.flags().to_flags_string(), "gs");
    }

    #[test]
    fn test_dotall_global_replaces_across_newlines() {
        let re = JsRegExp::new(".+", "gs").unwrap();
        let result = re.symbol_replace("a\nb", "x");
        assert_eq!(result, "x");
    }

    // ── Unicode property escapes ─────────────────────────────────────────

    #[test]
    fn test_unicode_property_letter_match() {
        let re = JsRegExp::new(r"\p{Letter}+", "u").unwrap();
        let m = re.exec("hello123").unwrap();
        assert_eq!(m.matched, "hello");
    }

    #[test]
    fn test_unicode_property_number_match() {
        let re = JsRegExp::new(r"\p{Number}+", "u").unwrap();
        let m = re.exec("abc42def").unwrap();
        assert_eq!(m.matched, "42");
    }

    #[test]
    fn test_unicode_property_negated_number() {
        let re = JsRegExp::new(r"\P{Number}+", "u").unwrap();
        let m = re.exec("42abc99").unwrap();
        assert_eq!(m.matched, "abc");
    }

    #[test]
    fn test_unicode_property_script_greek() {
        let re = JsRegExp::new(r"\p{Script=Greek}+", "u").unwrap();
        assert!(re.test("αβγ"));
        assert!(!re.test("abc"));
    }

    #[test]
    fn test_unicode_property_script_latin() {
        let re = JsRegExp::new(r"\p{Script=Latin}+", "u").unwrap();
        let m = re.exec("hello世界").unwrap();
        assert_eq!(m.matched, "hello");
    }

    #[test]
    fn test_unicode_property_general_category_uppercase() {
        let re = JsRegExp::new(r"\p{General_Category=Uppercase_Letter}+", "u").unwrap();
        let m = re.exec("helloWORLD").unwrap();
        assert_eq!(m.matched, "WORLD");
    }

    #[test]
    fn test_unicode_property_emoji_like() {
        // \p{L} should match CJK characters
        let re = JsRegExp::new(r"\p{L}+", "u").unwrap();
        assert!(re.test("你好"));
    }

    #[test]
    fn test_unicode_property_global_all_letters() {
        let re = JsRegExp::new(r"\p{L}+", "gu").unwrap();
        let results = re.symbol_match_all("hello 42 world");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].matched, "hello");
        assert_eq!(results[1].matched, "world");
    }

    // ── Named capture backreferences \k<name> ────────────────────────────

    #[test]
    fn test_backreference_named_repeat() {
        let re = JsRegExp::new(r"(?<ch>.)\k<ch>", "").unwrap();
        let m = re.exec("aabbcc").unwrap();
        assert_eq!(m.matched, "aa");
    }

    #[test]
    fn test_backreference_named_no_match_different() {
        let re = JsRegExp::new(r"(?<ch>.)\k<ch>", "").unwrap();
        assert!(re.exec("abcd").is_none());
    }

    #[test]
    fn test_backreference_named_html_tags() {
        let re = JsRegExp::new(r"<(?<tag>\w+)>[^<]*</\k<tag>>", "").unwrap();
        let m = re.exec("<b>bold</b>").unwrap();
        assert_eq!(m.matched, "<b>bold</b>");
        assert!(re.exec("<b>bold</i>").is_none());
    }

    #[test]
    fn test_backreference_named_with_unicode() {
        let re = JsRegExp::new(r"(?<w>\w+)\s\k<w>", "u").unwrap();
        let m = re.exec("the the dog").unwrap();
        assert_eq!(m.matched, "the the");
    }

    #[test]
    fn test_backreference_named_global_replace() {
        let re = JsRegExp::new(r"(?<w>\w+)\s\k<w>", "g").unwrap();
        let result = re.symbol_replace("the the is is ok", "[$<w>]");
        assert_eq!(result, "[the] [is] ok");
    }

    // ── Flag combinations ────────────────────────────────────────────────

    #[test]
    fn test_flags_gimus() {
        let re = JsRegExp::new(".", "gimus").unwrap();
        assert_eq!(re.flags().to_flags_string(), "gimsu");
        assert!(re.flags().contains(RegExpFlags::GLOBAL));
        assert!(re.flags().contains(RegExpFlags::IGNORE_CASE));
        assert!(re.flags().contains(RegExpFlags::MULTILINE));
        assert!(re.flags().contains(RegExpFlags::UNICODE));
        assert!(re.flags().contains(RegExpFlags::DOT_ALL));
    }

    #[test]
    fn test_flags_gimsuy() {
        let re = JsRegExp::new(".", "gimsuy").unwrap();
        assert_eq!(re.flags().to_flags_string(), "gimsuy");
        assert!(re.flags().contains(RegExpFlags::GLOBAL));
        assert!(re.flags().contains(RegExpFlags::IGNORE_CASE));
        assert!(re.flags().contains(RegExpFlags::MULTILINE));
        assert!(re.flags().contains(RegExpFlags::DOT_ALL));
        assert!(re.flags().contains(RegExpFlags::UNICODE));
        assert!(re.flags().contains(RegExpFlags::STICKY));
    }

    #[test]
    fn test_flags_du_combined() {
        let re = JsRegExp::new(r"\p{L}+", "du").unwrap();
        assert_eq!(re.flags().to_flags_string(), "du");
        let m = re.exec("hello").unwrap();
        assert!(m.indices.is_some());
    }

    #[test]
    fn test_flags_dgs_combined() {
        let re = JsRegExp::new(".+", "dgs").unwrap();
        let results = re.symbol_match_all("a\nb");
        // With /s, . matches newlines, so we get one match "a\nb"
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].matched, "a\nb");
        assert!(results[0].indices.is_some());
    }

    // ── dotAll + lookbehind combined ─────────────────────────────────────

    #[test]
    fn test_dotall_with_lookbehind() {
        let re = JsRegExp::new(r"(?<=\n)\w+", "s").unwrap();
        let m = re.exec("line1\nline2").unwrap();
        assert_eq!(m.matched, "line2");
    }

    // ── Unicode property with lookbehind ─────────────────────────────────

    #[test]
    fn test_unicode_property_with_lookbehind() {
        let re = JsRegExp::new(r"(?<=\p{L})\d+", "u").unwrap();
        let m = re.exec("abc42").unwrap();
        assert_eq!(m.matched, "42");
        assert!(re.exec("  42").is_none());
    }

    // ── Unicode + dotAll combined ────────────────────────────────────────

    #[test]
    fn test_unicode_dotall_combined() {
        let re = JsRegExp::new(r"\p{L}.+\p{L}", "su").unwrap();
        let m = re.exec("a\nb").unwrap();
        assert_eq!(m.matched, "a\nb");
    }

    // ── Named captures with indices ──────────────────────────────────────

    #[test]
    fn test_named_capture_with_indices() {
        let re = JsRegExp::new(r"(?<word>\w+)", "du").unwrap();
        let m = re.exec("hello").unwrap();
        let idx = m.indices.as_ref().unwrap();
        assert_eq!(idx.pairs[0], Some((0, 5)));
        assert_eq!(idx.groups.get("word"), Some(&(0, 5)));
    }

    // ── Lookbehind in replace ────────────────────────────────────────────

    #[test]
    fn test_lookbehind_in_global_replace() {
        let re = JsRegExp::new(r"(?<=\$)\d+", "g").unwrap();
        let result = re.symbol_replace("$100 and $200", "XXX");
        assert_eq!(result, "$XXX and $XXX");
    }

    // ── Edge: lookbehind at start of string ──────────────────────────────

    #[test]
    fn test_lookbehind_at_start_no_match() {
        let re = JsRegExp::new(r"(?<=x)\d+", "").unwrap();
        assert!(re.exec("42").is_none());
    }

    // ── Lookahead + lookbehind combined ──────────────────────────────────

    #[test]
    fn test_lookahead_and_lookbehind_combined() {
        let re = JsRegExp::new(r"(?<=\$)\d+(?=\s)", "").unwrap();
        let m = re.exec("$100 dollars").unwrap();
        assert_eq!(m.matched, "100");
        // Must not match: no space after digits
        assert!(re.exec("$100dollars").is_none());
    }
}
