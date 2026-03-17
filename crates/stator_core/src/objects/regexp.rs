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
    /// * **Global / sticky**: resets `lastIndex` to `0`, then collects every
    ///   non-overlapping matched string into a `Vec<String>`.  Returns `None`
    ///   if there are no matches.
    pub fn symbol_match(&self, input: &str) -> Option<SymbolMatchResult> {
        if !self.flags.contains(RegExpFlags::GLOBAL) && !self.flags.contains(RegExpFlags::STICKY) {
            // Non-global: behave like exec.
            self.exec(input).map(SymbolMatchResult::Single)
        } else {
            // Global/sticky: collect all matches.
            self.last_index.set(0);
            let mut matches: Vec<String> = Vec::new();
            loop {
                let start = self.last_index.get();
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
                        matches.push(input[mat.range()].to_string());
                        // Advance by at least 1 to avoid infinite loops on
                        // zero-length matches.
                        self.last_index
                            .set(if end > start { end } else { start + 1 });
                    }
                }
            }
            self.last_index.set(0);
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
        let global =
            self.flags.contains(RegExpFlags::GLOBAL) || self.flags.contains(RegExpFlags::STICKY);
        if global {
            self.last_index.set(0);
        }
        let mut result = String::new();
        let mut last_end = 0_usize;

        loop {
            let start = if global { self.last_index.get() } else { 0 };
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
                    let rm = build_match(input, &mat, false);
                    // Append the portion of input before this match.
                    result.push_str(&input[last_end..rm.index]);
                    // Apply replacement.
                    result.push_str(&apply_replacement(replacement, &rm, input));
                    let end = mat.end();
                    last_end = end;
                    if global {
                        self.last_index
                            .set(if end > start { end } else { start + 1 });
                    } else {
                        break;
                    }
                }
            }
        }
        // Append the remainder of the input.
        result.push_str(&input[last_end..]);
        if global {
            self.last_index.set(0);
        }
        result
    }

    /// ECMAScript `RegExp.prototype[Symbol.search](string)`.
    ///
    /// Returns the byte index of the first match, or `-1` if no match is
    /// found.  `lastIndex` is always reset to `0` before and after the search.
    pub fn symbol_search(&self, input: &str) -> i64 {
        let saved = self.last_index.get();
        self.last_index.set(0);
        let result = self
            .compiled
            .find(input)
            .map_or(-1, |m| input[..m.start()].encode_utf16().count() as i64);
        self.last_index.set(saved);
        result
    }

    /// ECMAScript `RegExp.prototype[Symbol.split](string[, limit])`.
    ///
    /// Splits `input` around each match and returns the parts as a
    /// `Vec<String>`.  Capture groups are included in the result between the
    /// surrounding parts, matching ECMAScript semantics.
    ///
    /// If `limit` is `Some(0)` an empty `Vec` is returned immediately.
    pub fn symbol_split(&self, input: &str, limit: Option<usize>) -> Vec<String> {
        let lim = limit.unwrap_or(usize::MAX);
        if lim == 0 {
            return Vec::new();
        }
        let mut parts: Vec<String> = Vec::new();
        let mut last_end = 0_usize;

        for mat in self.compiled.find_iter(input) {
            if parts.len() >= lim {
                break;
            }
            // Push the part before this match.
            parts.push(input[last_end..mat.start()].to_string());
            // Push each capture group (None → empty string, per ES spec).
            for cap in &mat.captures {
                if parts.len() >= lim {
                    break;
                }
                parts.push(
                    cap.as_ref()
                        .map_or(String::new(), |r| input[r.clone()].to_string()),
                );
            }
            last_end = mat.end();
        }

        // Push the trailing portion (only if we haven't hit the limit).
        if parts.len() < lim {
            parts.push(input[last_end..].to_string());
        }
        parts
    }

    /// ECMAScript `RegExp.prototype[Symbol.matchAll](string)`.
    ///
    /// Returns all successive matches as a `Vec<RegExpMatch>`, producing the
    /// same kind of result objects that [`Self::exec`] would for each match.
    /// This is the underlying implementation for `String.prototype.matchAll`.
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
                    start = if end > start { end } else { start + 1 };
                }
            }
        }
        results
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SymbolMatchResult
// ──────────────────────────────────────────────────────────────────────────────

/// The result type returned by [`JsRegExp::symbol_match`].
///
/// * [`Single`][SymbolMatchResult::Single] — non-global match, returns the
///   full exec result.
/// * [`All`][SymbolMatchResult::All] — global/sticky match, returns the list
///   of matched strings.
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolMatchResult {
    /// Result of a non-global `Symbol.match` call.
    Single(RegExpMatch),
    /// Result of a global/sticky `Symbol.match` call: all matched strings.
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
                        if let Some(Some(val)) = m.named_groups.get(name) {
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
        assert_eq!(re.symbol_split("a,b,c", None), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_symbol_split_with_limit() {
        let re = JsRegExp::new(",", "").unwrap();
        assert_eq!(re.symbol_split("a,b,c,d", Some(2)), vec!["a", "b"]);
    }

    #[test]
    fn test_symbol_split_zero_limit() {
        let re = JsRegExp::new(",", "").unwrap();
        assert_eq!(re.symbol_split("a,b,c", Some(0)), Vec::<String>::new());
    }

    #[test]
    fn test_symbol_split_captures_included() {
        // ES spec: capture groups are included in the result.
        let re = JsRegExp::new(r"(\d+)", "").unwrap();
        assert_eq!(
            re.symbol_split("a1b2c", None),
            vec!["a", "1", "b", "2", "c"]
        );
    }

    #[test]
    fn test_symbol_split_no_match_returns_whole_string() {
        let re = JsRegExp::new(r"\d+", "").unwrap();
        assert_eq!(re.symbol_split("abc", None), vec!["abc"]);
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
}
