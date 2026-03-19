//! ECMAScript §25.5 `JSON` built-in — `JSON.parse` and `JSON.stringify`.
//!
//! This module provides pure-Rust implementations of the two static methods
//! on the JavaScript `JSON` object:
//!
//! - [`json_parse`] — Deserialises a JSON text per RFC 8259, returning a
//!   [`JsonValue`].  An optional `reviver` closure is applied bottom-up to
//!   each parsed value.
//!
//! - [`json_stringify`] — Serialises a [`JsonValue`] to a JSON string, with
//!   optional [`JsonReplacer`] filtering/transformation and [`JsonSpace`]
//!   indentation.  Circular references (objects or arrays that appear as their
//!   own ancestors in the value tree) are detected and reported as a
//!   [`StatorError::TypeError`].
//!
//! # BigInt
//!
//! [`json_stringify_js_value`] converts [`crate::objects::value::JsValue`]
//! variants to JSON.  Encountering a `JsValue::BigInt` raises
//! `TypeError: Do not know how to serialize a BigInt` per the ECMAScript spec.
//!
//! # `toJSON()` methods
//!
//! The optional `to_json` callback accepted by [`json_stringify`] mirrors the
//! ECMAScript `toJSON(key)` hook.  Pass a closure that receives the property
//! key and the current [`JsonValue`] and returns `Some(replacement)` to
//! substitute the serialised value, or `None` to keep the original.
//!
//! # Naming convention
//!
//! Each public function is prefixed `json_` to avoid ambiguity with
//! similarly-named standard-library items.
//!
//! # References
//!
//! * ECMAScript 2025 Language Specification §25.5 — *The JSON Object*
//! * [RFC 8259](https://datatracker.ietf.org/doc/html/rfc8259) — *The JavaScript Object Notation (JSON) Data Interchange Format*

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use crate::builtins::proxy::{proxy_get, proxy_get_own_property_descriptor, proxy_own_keys};
use crate::error::{StatorError, StatorResult};
use crate::objects::value::JsValue;

// ── Type aliases for complex closure signatures ───────────────────────────────

/// A reviver closure: `(key, value) → Option<JsonValue>`.
///
/// Used by [`json_parse`] to transform each parsed value bottom-up.
/// Returning `None` deletes the property from its parent object; for
/// array elements, `None` is replaced with [`JsonValue::Null`].
pub type ReviverFn<'a> = &'a dyn Fn(&str, JsonValue) -> StatorResult<Option<JsonValue>>;

/// A `toJSON` hook closure: `(key, value) → Option<JsonValue>`.
///
/// Used by [`json_stringify`] to let callers intercept each value before it
/// is serialised (mirrors the ECMAScript `toJSON(key)` method on objects).
pub type ToJsonFn<'a> = &'a dyn Fn(&str, &JsonValue) -> Option<JsonValue>;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum number of characters / spaces per indent level for
/// [`JsonSpace::Count`] and [`JsonSpace::Str`].
///
/// ECMAScript §25.5.2 specifies that the `space` argument is clamped to 10.
const MAX_INDENT_SIZE: usize = 10;

/// Numbers whose absolute value is below this threshold are emitted as
/// integers (without a decimal point) when they have no fractional part.
///
/// `1e15` is chosen because `f64` can represent all integers up to `2^53`
/// (≈ `9.007e15`) exactly, and formatting values larger than `1e15` as
/// integers would lose precision on the representational boundary.
const INTEGER_THRESHOLD: f64 = 1e15;

// ─────────────────────────────────────────────────────────────────────────────
// JsonValue
// ─────────────────────────────────────────────────────────────────────────────

/// A JSON value per RFC 8259.
///
/// Arrays and objects are stored behind `Rc<RefCell<…>>` so that:
/// 1. Values can be cheaply shared and cloned.
/// 2. Circular structures can be constructed (in tests and by callers) to
///    exercise the cycle-detection logic in [`json_stringify`].
///
/// # Examples
///
/// ```
/// use stator_core::builtins::json::{json_parse, json_stringify, JsonValue, JsonSpace};
///
/// let v = json_parse(r#"{"x": 1}"#, None).unwrap();
/// let s = json_stringify(&v, None, None, None).unwrap().unwrap();
/// assert_eq!(s, r#"{"x":1}"#);
/// ```
#[derive(Debug, Clone)]
pub enum JsonValue {
    /// The JSON `null` literal.
    Null,
    /// A JSON boolean (`true` or `false`).
    Bool(bool),
    /// A JSON number (always stored as `f64`).
    Number(f64),
    /// A JSON string.
    Str(String),
    /// A JSON array — shared and mutable behind `Rc<RefCell<…>>`.
    Array(Rc<RefCell<Vec<JsonValue>>>),
    /// A JSON object — an ordered list of `(key, value)` pairs, shared and
    /// mutable behind `Rc<RefCell<…>>`.
    Object(Rc<RefCell<Vec<(String, JsonValue)>>>),
}

impl JsonValue {
    /// Creates a new, empty JSON array.
    pub fn new_array() -> Self {
        Self::Array(Rc::new(RefCell::new(Vec::new())))
    }

    /// Creates a new, empty JSON object.
    pub fn new_object() -> Self {
        Self::Object(Rc::new(RefCell::new(Vec::new())))
    }

    /// Returns `true` if this is the JSON `null` value.
    #[inline]
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Returns `true` if this is a JSON boolean.
    #[inline]
    pub fn is_bool(&self) -> bool {
        matches!(self, Self::Bool(_))
    }

    /// Returns `true` if this is a JSON number.
    #[inline]
    pub fn is_number(&self) -> bool {
        matches!(self, Self::Number(_))
    }

    /// Returns `true` if this is a JSON string.
    #[inline]
    pub fn is_string(&self) -> bool {
        matches!(self, Self::Str(_))
    }

    /// Returns `true` if this is a JSON array.
    #[inline]
    pub fn is_array(&self) -> bool {
        matches!(self, Self::Array(_))
    }

    /// Returns `true` if this is a JSON object.
    #[inline]
    pub fn is_object(&self) -> bool {
        matches!(self, Self::Object(_))
    }
}

impl PartialEq for JsonValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Number(a), Self::Number(b)) => {
                // NaN ≠ NaN per IEEE 754; two JSON Numbers are equal iff their
                // bit patterns are identical (same as Rust f64::total_cmp for
                // our purposes — we just use the f64 PartialEq here which
                // treats NaN as not equal to anything including itself).
                a == b
            }
            (Self::Str(a), Self::Str(b)) => a == b,
            (Self::Array(a), Self::Array(b)) => {
                // Structural equality of the arrays' contents.
                *a.borrow() == *b.borrow()
            }
            (Self::Object(a), Self::Object(b)) => {
                // Structural equality of the objects' entry lists.
                *a.borrow() == *b.borrow()
            }
            _ => false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JsonReplacer
// ─────────────────────────────────────────────────────────────────────────────

/// Controls which properties / array elements are included during
/// [`json_stringify`].
///
/// This mirrors the second argument of `JSON.stringify(value, replacer, space)`
/// in JavaScript:
///
/// * `Function` — called for every `(key, value)` pair; return `Some` to
///   include a (possibly transformed) value, `None` to omit it.
/// * `Array` — an allow-list of property names; only own properties whose key
///   appears in the list are included in object output.  Array elements are
///   unaffected.
pub enum JsonReplacer<'a> {
    /// A callable replacer: `(key, value) → Option<value>`.
    Function(&'a dyn Fn(&str, &JsonValue) -> StatorResult<Option<JsonValue>>),
    /// A property allow-list for objects.
    Array(Vec<String>),
}

// ─────────────────────────────────────────────────────────────────────────────
// JsonSpace
// ─────────────────────────────────────────────────────────────────────────────

/// Indentation specifier for [`json_stringify`].
///
/// Mirrors the third argument of `JSON.stringify(value, replacer, space)` in
/// JavaScript:
///
/// * `Count(n)` — indent with `n` space characters per level (clamped to
///   0 – 10).
/// * `Str(s)` — indent with the string `s` per level (truncated to 10
///   characters).
pub enum JsonSpace {
    /// Indent with up to 10 space characters per nesting level.
    Count(u32),
    /// Indent with a custom string per nesting level (max 10 chars).
    Str(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// json_parse — public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// ECMAScript §25.5.1 `JSON.parse(text, reviver)`.
///
/// Parses `text` as a JSON document per RFC 8259 and returns the resulting
/// [`JsonValue`].
///
/// If `reviver` is `Some`, it is called **bottom-up** for every
/// `(key, value)` pair: the final top-level value is produced by calling
/// `reviver("", top_level_value)`.  Returning `Some(value)` from the
/// reviver replaces the parsed value; returning `None` deletes the
/// property from its parent object (or replaces the element with
/// [`JsonValue::Null`] in arrays).  If the root-level reviver returns
/// `None`, `json_parse` returns [`JsonValue::Null`] (the closest
/// approximation, since [`JsonValue`] has no `Undefined` variant).
///
/// # Errors
///
/// Returns [`StatorError::SyntaxError`] for any malformed JSON input.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::json::{json_parse, JsonValue};
///
/// let v = json_parse("[1, true, null]", None).unwrap();
/// if let JsonValue::Array(arr) = &v {
///     assert_eq!(arr.borrow()[0], JsonValue::Number(1.0));
///     assert_eq!(arr.borrow()[1], JsonValue::Bool(true));
///     assert_eq!(arr.borrow()[2], JsonValue::Null);
/// } else {
///     panic!("expected array");
/// }
/// ```
pub fn json_parse(text: &str, reviver: Option<ReviverFn<'_>>) -> StatorResult<JsonValue> {
    let chars: Vec<char> = text.chars().collect();
    let mut parser = Parser {
        src: &chars,
        pos: 0,
        depth: 0,
    };
    parser.skip_ws();
    let value = parser.parse_value()?;
    parser.skip_ws();
    if parser.pos != parser.src.len() {
        return Err(StatorError::SyntaxError(format!(
            "Unexpected token at position {}",
            parser.pos
        )));
    }
    if let Some(rev) = reviver {
        // §25.5.1: if the top-level reviver returns undefined (None),
        // the closest JSON equivalent is null.
        Ok(apply_reviver(value, "", rev)?.unwrap_or(JsonValue::Null))
    } else {
        Ok(value)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// json_stringify — public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// ECMAScript §25.5.2 `JSON.stringify(value, replacer, space)`.
///
/// Serialises `value` to a JSON string and returns it wrapped in `Some`.
/// Returns `Ok(None)` when the top-level value would be omitted (i.e. when the
/// replacer filters out the root).
///
/// # Parameters
///
/// * `value` — The [`JsonValue`] to serialise.
/// * `replacer` — An optional [`JsonReplacer`] that filters or transforms each
///   value as it is serialised.
/// * `space` — An optional [`JsonSpace`] that adds newlines and indentation to
///   the output for human readability.
/// * `to_json` — An optional `toJSON(key)` hook.  When `Some`, it is called
///   for every value before serialisation; returning `Some(replacement)` uses
///   the replacement value, returning `None` uses the original.
///
/// # Errors
///
/// Returns [`StatorError::TypeError`] when a circular reference is detected
/// (i.e. an array or object appears as its own ancestor in the value tree).
///
/// # Examples
///
/// ```
/// use stator_core::builtins::json::{json_stringify, json_parse, JsonValue, JsonSpace};
///
/// let v = json_parse(r#"{"a":1,"b":[2,3]}"#, None).unwrap();
/// let s = json_stringify(&v, None, None, None).unwrap().unwrap();
/// assert_eq!(s, r#"{"a":1,"b":[2,3]}"#);
/// ```
pub fn json_stringify(
    value: &JsonValue,
    replacer: Option<&JsonReplacer<'_>>,
    space: Option<&JsonSpace>,
    to_json: Option<ToJsonFn<'_>>,
) -> StatorResult<Option<String>> {
    let indent = resolve_indent(space);
    let mut in_progress: HashSet<usize> = HashSet::new();
    stringify_value(value, "", replacer, &indent, 0, &mut in_progress, to_json)
}

/// Converts a [`JsValue`] to JSON, returning `Ok(None)` when the value has no
/// JSON representation (e.g. `undefined`, `Symbol`, `Function`).
///
/// # Errors
///
/// Returns [`StatorError::TypeError`] when `value` is a `BigInt`, which cannot
/// be serialised per the ECMAScript specification.
///
/// Returns [`StatorError::TypeError`] when a circular reference is detected
/// inside a `JsValue::Array`.
///
/// # Supported variants
///
/// | `JsValue` variant | JSON output |
/// |---|---|
/// | `Undefined` | `None` (omitted) |
/// | `Null` | `null` |
/// | `Boolean(b)` | `true` / `false` |
/// | `Smi(n)` | integer |
/// | `HeapNumber(n)` | number (`NaN`/`Infinity` → `null`) |
/// | `String(s)` | quoted string |
/// | `BigInt` | **TypeError** |
/// | `Symbol` | `None` (omitted) |
/// | `Function` | `None` (omitted) |
/// | `Array(items)` | JSON array |
/// | `Generator` / `Iterator` | `None` (omitted) |
/// | `Object` | `{}` (properties inaccessible without GC context) |
pub fn json_stringify_js_value(
    value: &JsValue,
    replacer: Option<&JsonReplacer<'_>>,
    space: Option<&JsonSpace>,
) -> StatorResult<Option<String>> {
    let json = js_value_to_json(value)?;
    match json {
        Some(jv) => json_stringify(&jv, replacer, space, None),
        None => Ok(None),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: RFC 8259 parser
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum nesting depth for JSON values (arrays/objects).
const JSON_MAX_DEPTH: usize = 512;

struct Parser<'a> {
    src: &'a [char],
    pos: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    /// Advance past any JSON whitespace (space, tab, CR, LF).
    fn skip_ws(&mut self) {
        while self.pos < self.src.len() && matches!(self.src[self.pos], ' ' | '\t' | '\r' | '\n') {
            self.pos += 1;
        }
    }

    /// Return the current character without advancing.
    fn peek(&self) -> Option<char> {
        self.src.get(self.pos).copied()
    }

    /// Consume the current character and advance.
    fn advance(&mut self) -> Option<char> {
        let ch = self.src.get(self.pos).copied();
        if ch.is_some() {
            self.pos += 1;
        }
        ch
    }

    /// Expect `expected` as the next character, returning an error otherwise.
    fn expect(&mut self, expected: char) -> StatorResult<()> {
        match self.advance() {
            Some(c) if c == expected => Ok(()),
            Some(c) => Err(StatorError::SyntaxError(format!(
                "Expected '{expected}', got '{c}' at position {}",
                self.pos - 1
            ))),
            None => Err(StatorError::SyntaxError(format!(
                "Expected '{expected}' but reached end of input"
            ))),
        }
    }

    /// Consume a known literal string (`true`, `false`, `null`).
    fn expect_literal(&mut self, lit: &str) -> StatorResult<()> {
        for expected in lit.chars() {
            match self.advance() {
                Some(c) if c == expected => {}
                Some(c) => {
                    return Err(StatorError::SyntaxError(format!(
                        "Expected '{lit}', found unexpected '{c}' at position {}",
                        self.pos - 1
                    )));
                }
                None => {
                    return Err(StatorError::SyntaxError(format!(
                        "Expected '{lit}', reached end of input"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Parse any JSON value.
    fn parse_value(&mut self) -> StatorResult<JsonValue> {
        self.skip_ws();
        match self.peek() {
            Some('"') => self.parse_string().map(JsonValue::Str),
            Some('{') | Some('[') => {
                self.depth += 1;
                if self.depth > JSON_MAX_DEPTH {
                    return Err(StatorError::SyntaxError(
                        "JSON nesting depth exceeded".to_string(),
                    ));
                }
                let result = if self.peek() == Some('{') {
                    self.parse_object()
                } else {
                    self.parse_array()
                };
                self.depth -= 1;
                result
            }
            Some('t') => {
                self.expect_literal("true")?;
                Ok(JsonValue::Bool(true))
            }
            Some('f') => {
                self.expect_literal("false")?;
                Ok(JsonValue::Bool(false))
            }
            Some('n') => {
                self.expect_literal("null")?;
                Ok(JsonValue::Null)
            }
            Some(c) if c == '-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(StatorError::SyntaxError(format!(
                "Unexpected character '{c}' at position {}",
                self.pos
            ))),
            None => Err(StatorError::SyntaxError(
                "Unexpected end of JSON input".to_string(),
            )),
        }
    }

    /// Parse a JSON string (RFC 8259 §7).
    fn parse_string(&mut self) -> StatorResult<String> {
        self.expect('"')?;
        let mut s = String::new();
        loop {
            match self.advance() {
                None => return Err(StatorError::SyntaxError("Unterminated string".to_string())),
                Some('"') => break,
                Some('\\') => {
                    let escaped = self.parse_escape()?;
                    s.push_str(&escaped);
                }
                Some(c) if (c as u32) < 0x20 => {
                    return Err(StatorError::SyntaxError(format!(
                        "Unescaped control character U+{:04X} in string",
                        c as u32
                    )));
                }
                Some(c) => s.push(c),
            }
        }
        Ok(s)
    }

    /// Parse a JSON escape sequence after the leading `\`.
    fn parse_escape(&mut self) -> StatorResult<String> {
        match self.advance() {
            Some('"') => Ok("\"".to_string()),
            Some('\\') => Ok("\\".to_string()),
            Some('/') => Ok("/".to_string()),
            Some('b') => Ok("\x08".to_string()),
            Some('f') => Ok("\x0C".to_string()),
            Some('n') => Ok("\n".to_string()),
            Some('r') => Ok("\r".to_string()),
            Some('t') => Ok("\t".to_string()),
            Some('u') => {
                let cp = self.parse_hex4()?;
                // Check for a surrogate pair.
                if (0xD800..=0xDBFF).contains(&cp) {
                    // Expect a following \uDC00–\uDFFF low surrogate.
                    if self.peek() == Some('\\') {
                        self.advance(); // consume '\'
                        if self.peek() == Some('u') {
                            self.advance(); // consume 'u'
                            let low = self.parse_hex4()?;
                            if (0xDC00..=0xDFFF).contains(&low) {
                                let scalar =
                                    0x10000 + ((cp as u32 - 0xD800) << 10) + (low as u32 - 0xDC00);
                                let ch = char::from_u32(scalar).ok_or_else(|| {
                                    StatorError::SyntaxError(format!(
                                        "Invalid surrogate pair U+{cp:04X} U+{low:04X}"
                                    ))
                                })?;
                                return Ok(ch.to_string());
                            }
                            // `low` is not a low surrogate; treat the high
                            // surrogate as a lone surrogate — emit the
                            // replacement character for the high surrogate and
                            // then the decoded `low` codepoint.
                            let high_ch =
                                char::from_u32(cp as u32).unwrap_or(char::REPLACEMENT_CHARACTER);
                            let low_ch =
                                char::from_u32(low as u32).unwrap_or(char::REPLACEMENT_CHARACTER);
                            return Ok(format!("{high_ch}{low_ch}"));
                        }
                        // Not a `\u` sequence after `\` — put back the `\`
                        // by rewinding one position and emit the lone
                        // surrogate as the replacement character.
                        self.pos -= 1;
                    }
                    // Lone high surrogate: emit replacement character.
                    return Ok(char::REPLACEMENT_CHARACTER.to_string());
                }
                // Regular BMP codepoint (or lone low surrogate).
                let ch = char::from_u32(cp as u32).unwrap_or(char::REPLACEMENT_CHARACTER);
                Ok(ch.to_string())
            }
            Some(c) => Err(StatorError::SyntaxError(format!(
                "Invalid escape sequence '\\{c}'"
            ))),
            None => Err(StatorError::SyntaxError(
                "Unexpected end of input in escape sequence".to_string(),
            )),
        }
    }

    /// Parse exactly 4 hex digits and return the resulting u16.
    fn parse_hex4(&mut self) -> StatorResult<u16> {
        let mut val: u32 = 0;
        for _ in 0..4 {
            match self.advance() {
                Some(c) if c.is_ascii_hexdigit() => {
                    val = val * 16 + c.to_digit(16).unwrap();
                }
                Some(c) => {
                    return Err(StatorError::SyntaxError(format!(
                        "Invalid hex digit '{c}' in \\uXXXX escape"
                    )));
                }
                None => {
                    return Err(StatorError::SyntaxError(
                        "Unexpected end of input in \\uXXXX escape".to_string(),
                    ));
                }
            }
        }
        Ok(val as u16)
    }

    /// Parse a JSON number (RFC 8259 §6).
    fn parse_number(&mut self) -> StatorResult<JsonValue> {
        let start = self.pos;
        // Optional leading minus.
        if self.peek() == Some('-') {
            self.advance();
        }
        // Integer part.
        match self.peek() {
            Some('0') => {
                self.advance();
            }
            Some(c) if c.is_ascii_digit() => {
                while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    self.advance();
                }
            }
            _ => {
                return Err(StatorError::SyntaxError(format!(
                    "Invalid number at position {start}"
                )));
            }
        }
        // Optional fractional part.
        if self.peek() == Some('.') {
            self.advance();
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(StatorError::SyntaxError(
                    "Expected digit after decimal point".to_string(),
                ));
            }
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.advance();
            }
        }
        // Optional exponent.
        if matches!(self.peek(), Some('e') | Some('E')) {
            self.advance();
            if matches!(self.peek(), Some('+') | Some('-')) {
                self.advance();
            }
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(StatorError::SyntaxError(
                    "Expected digit in exponent".to_string(),
                ));
            }
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.advance();
            }
        }
        let slice: String = self.src[start..self.pos].iter().collect();
        let n: f64 = slice
            .parse()
            .map_err(|_| StatorError::SyntaxError(format!("Invalid number literal '{slice}'")))?;
        Ok(JsonValue::Number(n))
    }

    /// Parse a JSON array.
    fn parse_array(&mut self) -> StatorResult<JsonValue> {
        self.expect('[')?;
        let arr = Rc::new(RefCell::new(Vec::new()));
        self.skip_ws();
        if self.peek() == Some(']') {
            self.advance();
            return Ok(JsonValue::Array(arr));
        }
        loop {
            self.skip_ws();
            let v = self.parse_value()?;
            arr.borrow_mut().push(v);
            self.skip_ws();
            match self.peek() {
                Some(',') => {
                    self.advance();
                }
                Some(']') => {
                    self.advance();
                    break;
                }
                Some(c) => {
                    return Err(StatorError::SyntaxError(format!(
                        "Expected ',' or ']', got '{c}' at position {}",
                        self.pos
                    )));
                }
                None => {
                    return Err(StatorError::SyntaxError("Unterminated array".to_string()));
                }
            }
        }
        Ok(JsonValue::Array(arr))
    }

    /// Parse a JSON object.
    fn parse_object(&mut self) -> StatorResult<JsonValue> {
        self.expect('{')?;
        let obj = Rc::new(RefCell::new(Vec::new()));
        self.skip_ws();
        if self.peek() == Some('}') {
            self.advance();
            return Ok(JsonValue::Object(obj));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some('"') {
                return Err(StatorError::SyntaxError(format!(
                    "Expected string key at position {}",
                    self.pos
                )));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(':')?;
            self.skip_ws();
            let val = self.parse_value()?;
            // §25.5.1: duplicate keys — last value wins.
            let mut entries = obj.borrow_mut();
            if let Some(existing) = entries.iter_mut().find(|(k, _)| k == &key) {
                existing.1 = val;
            } else {
                entries.push((key, val));
            }
            drop(entries);
            self.skip_ws();
            match self.peek() {
                Some(',') => {
                    self.advance();
                }
                Some('}') => {
                    self.advance();
                    break;
                }
                Some(c) => {
                    return Err(StatorError::SyntaxError(format!(
                        "Expected ',' or '}}', got '{c}' at position {}",
                        self.pos
                    )));
                }
                None => {
                    return Err(StatorError::SyntaxError("Unterminated object".to_string()));
                }
            }
        }
        Ok(JsonValue::Object(obj))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: reviver application (bottom-up walk)
// ─────────────────────────────────────────────────────────────────────────────

/// Recursively applies `reviver` to all nodes in `value`, bottom-up.
///
/// Returns `None` when the reviver returns `None` for the current node,
/// signalling that the caller should delete the property (objects) or
/// substitute [`JsonValue::Null`] (arrays).
fn apply_reviver(
    value: JsonValue,
    key: &str,
    reviver: &dyn Fn(&str, JsonValue) -> StatorResult<Option<JsonValue>>,
) -> StatorResult<Option<JsonValue>> {
    let transformed = match value {
        JsonValue::Array(ref arr) => {
            let items: Vec<JsonValue> = {
                let borrow = arr.borrow();
                borrow.clone()
            };
            let mut new_items = Vec::with_capacity(items.len());
            for (i, item) in items.into_iter().enumerate() {
                let idx_str = i.to_string();
                let revived = apply_reviver(item, &idx_str, reviver)?;
                // §25.5.1.1: undefined → null for array elements.
                new_items.push(revived.unwrap_or(JsonValue::Null));
            }
            *arr.borrow_mut() = new_items;
            value
        }
        JsonValue::Object(ref obj) => {
            let pairs: Vec<(String, JsonValue)> = {
                let borrow = obj.borrow();
                borrow.clone()
            };
            let mut new_pairs = Vec::with_capacity(pairs.len());
            for (k, v) in pairs {
                let revived = apply_reviver(v, &k, reviver)?;
                // §25.5.1.1: undefined → delete the property.
                if let Some(rv) = revived {
                    new_pairs.push((k, rv));
                }
            }
            *obj.borrow_mut() = new_pairs;
            value
        }
        other => other,
    };
    reviver(key, transformed)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: stringify helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve the `JsonSpace` option to the indent string used between levels.
fn resolve_indent(space: Option<&JsonSpace>) -> String {
    match space {
        None => String::new(),
        Some(JsonSpace::Count(n)) => {
            let count = (*n).min(MAX_INDENT_SIZE as u32) as usize;
            " ".repeat(count)
        }
        Some(JsonSpace::Str(s)) => {
            // Clamp to the first MAX_INDENT_SIZE characters.
            s.chars().take(MAX_INDENT_SIZE).collect()
        }
    }
}

/// Recursively stringify `value`.
///
/// `key`         — the property key under which `value` appears (empty string
///                 for the root value).
/// `in_progress` — the set of `Rc` pointer addresses currently being
///                 serialised; used for circular-reference detection.
/// `depth`       — current nesting depth for indentation.
#[allow(clippy::too_many_arguments)]
fn stringify_value(
    value: &JsonValue,
    key: &str,
    replacer: Option<&JsonReplacer<'_>>,
    indent: &str,
    depth: usize,
    in_progress: &mut HashSet<usize>,
    to_json: Option<ToJsonFn<'_>>,
) -> StatorResult<Option<String>> {
    // Apply the toJSON hook (if provided).
    let to_json_owned;
    let value = if let Some(hook) = to_json {
        if let Some(replacement) = hook(key, value) {
            to_json_owned = replacement;
            &to_json_owned
        } else {
            value
        }
    } else {
        value
    };

    // Apply the function replacer: transform (or omit) the value.
    // After the replacer runs, its returned value is serialised; the replacer
    // is still active for the returned value's child properties/elements.
    let fn_owned;
    let value = if let Some(JsonReplacer::Function(f)) = replacer {
        match f(key, value)? {
            Some(replaced) => {
                fn_owned = replaced;
                &fn_owned
            }
            None => return Ok(None),
        }
    } else {
        value
    };

    match value {
        JsonValue::Null => Ok(Some("null".to_string())),
        JsonValue::Bool(b) => Ok(Some(if *b { "true" } else { "false" }.to_string())),
        JsonValue::Number(n) => {
            if n.is_nan() || n.is_infinite() {
                // ECMAScript: NaN and Infinity are serialised as "null".
                Ok(Some("null".to_string()))
            } else {
                // Emit an integer representation if the value is a whole number.
                if n.fract() == 0.0 && n.abs() < INTEGER_THRESHOLD {
                    Ok(Some(format!("{}", *n as i64)))
                } else {
                    Ok(Some(format!("{n}")))
                }
            }
        }
        JsonValue::Str(s) => Ok(Some(stringify_string(s))),
        JsonValue::Array(arr) => {
            let ptr = Rc::as_ptr(arr) as usize;
            if in_progress.contains(&ptr) {
                return Err(StatorError::TypeError(
                    "Converting circular structure to JSON".to_string(),
                ));
            }
            in_progress.insert(ptr);
            let result = stringify_array(arr, replacer, indent, depth, in_progress, to_json);
            in_progress.remove(&ptr);
            result
        }
        JsonValue::Object(obj) => {
            let ptr = Rc::as_ptr(obj) as usize;
            if in_progress.contains(&ptr) {
                return Err(StatorError::TypeError(
                    "Converting circular structure to JSON".to_string(),
                ));
            }
            in_progress.insert(ptr);
            let result = stringify_object(obj, replacer, indent, depth, in_progress, to_json);
            in_progress.remove(&ptr);
            result
        }
    }
}

/// Produce the JSON representation of a string, escaping per RFC 8259 §7.
fn stringify_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\x08' => out.push_str("\\b"),
            '\x0C' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04X}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Stringify a JSON array.
fn stringify_array(
    arr: &Rc<RefCell<Vec<JsonValue>>>,
    replacer: Option<&JsonReplacer<'_>>,
    indent: &str,
    depth: usize,
    in_progress: &mut HashSet<usize>,
    to_json: Option<ToJsonFn<'_>>,
) -> StatorResult<Option<String>> {
    let borrow = arr.borrow();
    if borrow.is_empty() {
        return Ok(Some("[]".to_string()));
    }
    let use_indent = !indent.is_empty();
    let inner_indent = indent.repeat(depth + 1);
    let outer_indent = indent.repeat(depth);

    let mut parts = Vec::with_capacity(borrow.len());
    for (i, item) in borrow.iter().enumerate() {
        let idx_str = i.to_string();
        let serialised = stringify_value(
            item,
            &idx_str,
            replacer,
            indent,
            depth + 1,
            in_progress,
            to_json,
        )?;
        // Array holes: omitted (None) items serialise as "null".
        parts.push(serialised.unwrap_or_else(|| "null".to_string()));
    }

    if use_indent {
        let joined = parts.join(&format!(",\n{inner_indent}"));
        Ok(Some(format!("[\n{inner_indent}{joined}\n{outer_indent}]")))
    } else {
        Ok(Some(format!("[{}]", parts.join(","))))
    }
}

/// Stringify a JSON object.
fn stringify_object(
    obj: &Rc<RefCell<Vec<(String, JsonValue)>>>,
    replacer: Option<&JsonReplacer<'_>>,
    indent: &str,
    depth: usize,
    in_progress: &mut HashSet<usize>,
    to_json: Option<ToJsonFn<'_>>,
) -> StatorResult<Option<String>> {
    let borrow = obj.borrow();
    let use_indent = !indent.is_empty();
    let inner_indent = indent.repeat(depth + 1);
    let outer_indent = indent.repeat(depth);

    let mut parts: Vec<String> = Vec::new();
    for (k, v) in borrow.iter() {
        // Array replacer: skip properties not in the allow-list.
        if let Some(JsonReplacer::Array(allowed)) = replacer
            && !allowed.iter().any(|a| a == k)
        {
            continue;
        }
        let serialised = stringify_value(v, k, replacer, indent, depth + 1, in_progress, to_json)?;
        // If the replacer or toJSON returns None (omit), skip this property.
        if let Some(s) = serialised {
            let key_str = stringify_string(k);
            if use_indent {
                parts.push(format!("{key_str}: {s}"));
            } else {
                parts.push(format!("{key_str}:{s}"));
            }
        }
    }

    if parts.is_empty() {
        return Ok(Some("{}".to_string()));
    }

    if use_indent {
        let joined = parts.join(&format!(",\n{inner_indent}"));
        Ok(Some(format!(
            "{{\n{inner_indent}{joined}\n{outer_indent}}}"
        )))
    } else {
        Ok(Some(format!("{{{}}}", parts.join(","))))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: JsValue → JsonValue conversion
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a `JsValue` to a `JsonValue`.
///
/// Returns `None` for values that have no JSON representation (`Undefined`,
/// `Symbol`, `Function`, `Generator`, `Iterator`).
///
/// # Errors
///
/// Returns `TypeError` for `BigInt` and for circular `Array` values.
pub fn js_value_to_json(value: &JsValue) -> StatorResult<Option<JsonValue>> {
    js_value_to_json_inner(value, &mut HashSet::new())
}

fn js_value_to_json_inner(
    value: &JsValue,
    seen: &mut HashSet<usize>,
) -> StatorResult<Option<JsonValue>> {
    match value {
        JsValue::Undefined
        | JsValue::TheHole
        | JsValue::Symbol(_)
        | JsValue::Function(_)
        | JsValue::Generator(_)
        | JsValue::Iterator(_)
        | JsValue::Error(_)
        | JsValue::Promise(_)
        | JsValue::Context(_)
        | JsValue::ArrayBuffer(_)
        | JsValue::TypedArray(_)
        | JsValue::DataView(_) => Ok(None),
        JsValue::Null => Ok(Some(JsonValue::Null)),
        JsValue::Boolean(b) => Ok(Some(JsonValue::Bool(*b))),
        JsValue::Smi(n) => Ok(Some(JsonValue::Number(f64::from(*n)))),
        JsValue::HeapNumber(n) => Ok(Some(JsonValue::Number(*n))),
        JsValue::String(s) => Ok(Some(JsonValue::Str(s.to_string()))),
        JsValue::BigInt(_) => Err(StatorError::TypeError(
            "Do not know how to serialize a BigInt".to_string(),
        )),
        JsValue::Array(items) => {
            let ptr = Rc::as_ptr(items) as usize;
            if seen.contains(&ptr) {
                return Err(StatorError::TypeError(
                    "Converting circular structure to JSON".to_string(),
                ));
            }
            seen.insert(ptr);
            let mut arr: Vec<JsonValue> = Vec::with_capacity(items.borrow().len());
            for item in items.borrow().iter() {
                let json_item = js_value_to_json_inner(item, seen)?;
                arr.push(json_item.unwrap_or(JsonValue::Null));
            }
            seen.remove(&ptr);
            Ok(Some(JsonValue::Array(Rc::new(RefCell::new(arr)))))
        }
        // Object values are inaccessible without a GC context; emit an empty
        // object to indicate the value is object-shaped.
        JsValue::Object(_) => Ok(Some(JsonValue::Object(Rc::new(RefCell::new(Vec::new()))))),
        // NativeFunction and PlainObject are not JSON-serializable.
        JsValue::NativeFunction(_) => Ok(None),
        JsValue::PlainObject(map) => {
            // §25.5.2 step 2: if the object has a callable `toJSON` property,
            // invoke it and serialise the return value instead.
            let to_json_fn = map.borrow().get("toJSON").and_then(|v| {
                if let JsValue::NativeFunction(f) = v {
                    Some(f.clone())
                } else {
                    None
                }
            });
            if let Some(f) = to_json_fn {
                let result = f(vec![JsValue::String(String::new().into())])?;
                return js_value_to_json_inner(&result, seen);
            }

            let ptr = Rc::as_ptr(map) as usize;
            if seen.contains(&ptr) {
                return Err(StatorError::TypeError(
                    "Converting circular structure to JSON".to_string(),
                ));
            }
            seen.insert(ptr);
            let mut entries: Vec<(String, JsonValue)> = Vec::new();
            // §25.5.2 step 6: only enumerable own properties are serialised.
            for (k, v) in map.borrow().enumerable_iter() {
                if let Some(jv) = js_value_to_json_inner(v, seen)? {
                    entries.push((k.to_string(), jv));
                }
            }
            seen.remove(&ptr);
            Ok(Some(JsonValue::Object(Rc::new(RefCell::new(entries)))))
        }
        // §25.5.2: Proxy objects are serialized through their traps.
        JsValue::Proxy(proxy) => {
            let ptr = Rc::as_ptr(proxy) as usize;
            if seen.contains(&ptr) {
                return Err(StatorError::TypeError(
                    "Converting circular structure to JSON".to_string(),
                ));
            }
            seen.insert(ptr);
            let keys = proxy_own_keys(&proxy.borrow())?;
            let mut entries: Vec<(String, JsonValue)> = Vec::new();
            for key in keys {
                let key_str = match &key {
                    JsValue::String(s) => s.to_string(),
                    _ => continue,
                };
                let Some((_, attrs)) =
                    proxy_get_own_property_descriptor(&proxy.borrow(), &key_str)?
                else {
                    continue;
                };
                if !attrs.contains(crate::objects::map::PropertyAttributes::ENUMERABLE) {
                    continue;
                }
                let val = proxy_get(&proxy.borrow(), &key_str)?;
                if let Some(jv) = js_value_to_json_inner(&val, seen)? {
                    entries.push((key_str, jv));
                }
            }
            seen.remove(&ptr);
            Ok(Some(JsonValue::Object(Rc::new(RefCell::new(entries)))))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── json_parse — primitives ───────────────────────────────────────────────

    #[test]
    fn test_parse_null() {
        assert_eq!(json_parse("null", None).unwrap(), JsonValue::Null);
    }

    #[test]
    fn test_parse_true() {
        assert_eq!(json_parse("true", None).unwrap(), JsonValue::Bool(true));
    }

    #[test]
    fn test_parse_false() {
        assert_eq!(json_parse("false", None).unwrap(), JsonValue::Bool(false));
    }

    #[test]
    fn test_parse_integer() {
        assert_eq!(json_parse("42", None).unwrap(), JsonValue::Number(42.0));
    }

    #[test]
    fn test_parse_negative_integer() {
        assert_eq!(json_parse("-7", None).unwrap(), JsonValue::Number(-7.0));
    }

    #[test]
    fn test_parse_float() {
        assert_eq!(json_parse("3.14", None).unwrap(), JsonValue::Number(3.14));
    }

    #[test]
    fn test_parse_exponent() {
        assert_eq!(json_parse("1e2", None).unwrap(), JsonValue::Number(100.0));
    }

    #[test]
    fn test_parse_string() {
        assert_eq!(
            json_parse(r#""hello""#, None).unwrap(),
            JsonValue::Str("hello".to_string())
        );
    }

    #[test]
    fn test_parse_string_escape_sequences() {
        let v = json_parse(r#""tab\there""#, None).unwrap();
        assert_eq!(v, JsonValue::Str("tab\there".to_string()));
    }

    #[test]
    fn test_parse_string_newline_escape() {
        let v = json_parse(r#""line1\nline2""#, None).unwrap();
        assert_eq!(v, JsonValue::Str("line1\nline2".to_string()));
    }

    // ── json_parse — Unicode escapes ──────────────────────────────────────────

    #[test]
    fn test_parse_unicode_escape_basic() {
        // \u0041 = 'A'
        let v = json_parse(r#""\u0041""#, None).unwrap();
        assert_eq!(v, JsonValue::Str("A".to_string()));
    }

    #[test]
    fn test_parse_unicode_escape_non_ascii() {
        // \u00E9 = 'é'
        let v = json_parse(r#""\u00E9""#, None).unwrap();
        assert_eq!(v, JsonValue::Str("é".to_string()));
    }

    #[test]
    fn test_parse_unicode_surrogate_pair() {
        // U+1F600 (😀) encoded as surrogate pair \uD83D\uDE00
        let v = json_parse(r#""\uD83D\uDE00""#, None).unwrap();
        assert_eq!(v, JsonValue::Str("😀".to_string()));
    }

    // ── json_parse — arrays ───────────────────────────────────────────────────

    #[test]
    fn test_parse_empty_array() {
        let v = json_parse("[]", None).unwrap();
        if let JsonValue::Array(arr) = &v {
            assert!(arr.borrow().is_empty());
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn test_parse_flat_array() {
        let v = json_parse("[1, 2, 3]", None).unwrap();
        if let JsonValue::Array(arr) = &v {
            let b = arr.borrow();
            assert_eq!(b[0], JsonValue::Number(1.0));
            assert_eq!(b[1], JsonValue::Number(2.0));
            assert_eq!(b[2], JsonValue::Number(3.0));
        } else {
            panic!("expected array");
        }
    }

    // ── json_parse — objects ──────────────────────────────────────────────────

    #[test]
    fn test_parse_empty_object() {
        let v = json_parse("{}", None).unwrap();
        if let JsonValue::Object(obj) = &v {
            assert!(obj.borrow().is_empty());
        } else {
            panic!("expected object");
        }
    }

    #[test]
    fn test_parse_flat_object() {
        let v = json_parse(r#"{"a":1,"b":true}"#, None).unwrap();
        if let JsonValue::Object(obj) = &v {
            let b = obj.borrow();
            assert_eq!(b[0], ("a".to_string(), JsonValue::Number(1.0)));
            assert_eq!(b[1], ("b".to_string(), JsonValue::Bool(true)));
        } else {
            panic!("expected object");
        }
    }

    // ── json_parse — nested objects ───────────────────────────────────────────

    #[test]
    fn test_parse_nested_objects() {
        let v = json_parse(r#"{"outer":{"inner":42}}"#, None).unwrap();
        if let JsonValue::Object(obj) = &v {
            let b = obj.borrow();
            assert_eq!(b[0].0, "outer");
            if let JsonValue::Object(inner) = &b[0].1 {
                let ib = inner.borrow();
                assert_eq!(ib[0], ("inner".to_string(), JsonValue::Number(42.0)));
            } else {
                panic!("expected inner object");
            }
        } else {
            panic!("expected outer object");
        }
    }

    #[test]
    fn test_parse_nested_array_of_objects() {
        let v = json_parse(r#"[{"x":1},{"x":2}]"#, None).unwrap();
        if let JsonValue::Array(arr) = &v {
            let b = arr.borrow();
            assert_eq!(b.len(), 2);
            for (i, item) in b.iter().enumerate() {
                if let JsonValue::Object(obj) = item {
                    let ob = obj.borrow();
                    assert_eq!(ob[0].1, JsonValue::Number((i + 1) as f64));
                } else {
                    panic!("expected object at index {i}");
                }
            }
        } else {
            panic!("expected array");
        }
    }

    // ── json_parse — reviver ──────────────────────────────────────────────────

    #[test]
    fn test_parse_reviver_doubles_numbers() {
        let v = json_parse(
            "[1, 2, 3]",
            Some(&|_key, val| {
                Ok(Some(match val {
                    JsonValue::Number(n) => JsonValue::Number(n * 2.0),
                    other => other,
                }))
            }),
        )
        .unwrap();
        if let JsonValue::Array(arr) = &v {
            let b = arr.borrow();
            assert_eq!(b[0], JsonValue::Number(2.0));
            assert_eq!(b[1], JsonValue::Number(4.0));
            assert_eq!(b[2], JsonValue::Number(6.0));
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn test_parse_reviver_deletes_object_property() {
        let v = json_parse(
            r#"{"a":1,"b":2,"c":3}"#,
            Some(&|key, val| {
                if key == "b" { Ok(None) } else { Ok(Some(val)) }
            }),
        )
        .unwrap();
        if let JsonValue::Object(obj) = &v {
            let b = obj.borrow();
            assert_eq!(b.len(), 2);
            assert_eq!(b[0], ("a".to_string(), JsonValue::Number(1.0)));
            assert_eq!(b[1], ("c".to_string(), JsonValue::Number(3.0)));
        } else {
            panic!("expected object");
        }
    }

    #[test]
    fn test_parse_reviver_array_undefined_becomes_null() {
        let v = json_parse(
            "[1, 2, 3]",
            Some(&|key, val| {
                if key == "1" { Ok(None) } else { Ok(Some(val)) }
            }),
        )
        .unwrap();
        if let JsonValue::Array(arr) = &v {
            let b = arr.borrow();
            assert_eq!(b[0], JsonValue::Number(1.0));
            assert_eq!(b[1], JsonValue::Null);
            assert_eq!(b[2], JsonValue::Number(3.0));
        } else {
            panic!("expected array");
        }
    }

    // ── json_parse — error handling ───────────────────────────────────────────

    #[test]
    fn test_parse_trailing_garbage() {
        assert!(json_parse("1 trailing", None).is_err());
    }

    #[test]
    fn test_parse_invalid_number_no_digits_after_dot() {
        assert!(json_parse("1.", None).is_err());
    }

    #[test]
    fn test_parse_unterminated_string() {
        assert!(json_parse(r#""no closing quote"#, None).is_err());
    }

    #[test]
    fn test_parse_invalid_escape() {
        assert!(json_parse(r#""\q""#, None).is_err());
    }

    #[test]
    fn test_parse_empty_input() {
        assert!(json_parse("", None).is_err());
    }

    // ── json_stringify — primitives ───────────────────────────────────────────

    #[test]
    fn test_stringify_null() {
        let s = json_stringify(&JsonValue::Null, None, None, None)
            .unwrap()
            .unwrap();
        assert_eq!(s, "null");
    }

    #[test]
    fn test_stringify_bool_true() {
        let s = json_stringify(&JsonValue::Bool(true), None, None, None)
            .unwrap()
            .unwrap();
        assert_eq!(s, "true");
    }

    #[test]
    fn test_stringify_bool_false() {
        let s = json_stringify(&JsonValue::Bool(false), None, None, None)
            .unwrap()
            .unwrap();
        assert_eq!(s, "false");
    }

    #[test]
    fn test_stringify_integer() {
        let s = json_stringify(&JsonValue::Number(42.0), None, None, None)
            .unwrap()
            .unwrap();
        assert_eq!(s, "42");
    }

    #[test]
    fn test_stringify_float() {
        let s = json_stringify(&JsonValue::Number(3.14), None, None, None)
            .unwrap()
            .unwrap();
        assert_eq!(s, "3.14");
    }

    #[test]
    fn test_stringify_nan_becomes_null() {
        let s = json_stringify(&JsonValue::Number(f64::NAN), None, None, None)
            .unwrap()
            .unwrap();
        assert_eq!(s, "null");
    }

    #[test]
    fn test_stringify_infinity_becomes_null() {
        let s = json_stringify(&JsonValue::Number(f64::INFINITY), None, None, None)
            .unwrap()
            .unwrap();
        assert_eq!(s, "null");
    }

    #[test]
    fn test_stringify_neg_infinity_becomes_null() {
        let s = json_stringify(&JsonValue::Number(f64::NEG_INFINITY), None, None, None)
            .unwrap()
            .unwrap();
        assert_eq!(s, "null");
    }

    #[test]
    fn test_stringify_string() {
        let s = json_stringify(&JsonValue::Str("hello".to_string()), None, None, None)
            .unwrap()
            .unwrap();
        assert_eq!(s, r#""hello""#);
    }

    #[test]
    fn test_stringify_string_with_special_chars() {
        let s = json_stringify(
            &JsonValue::Str("line1\nline2\ttab".to_string()),
            None,
            None,
            None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(s, r#""line1\nline2\ttab""#);
    }

    #[test]
    fn test_stringify_string_with_quotes() {
        let s = json_stringify(
            &JsonValue::Str(r#"say "hello""#.to_string()),
            None,
            None,
            None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(s, r#""say \"hello\"""#);
    }

    // ── json_stringify — Unicode ──────────────────────────────────────────────

    #[test]
    fn test_stringify_unicode_passthrough() {
        // Code points ≥ U+0020 are passed through unchanged (they are valid
        // JSON characters).
        let s = json_stringify(&JsonValue::Str("café 日本語".to_string()), None, None, None)
            .unwrap()
            .unwrap();
        assert_eq!(s, r#""café 日本語""#);
    }

    #[test]
    fn test_stringify_control_char_escaped() {
        // Control characters U+0000–U+001F must be escaped as \uXXXX.
        let s = json_stringify(&JsonValue::Str("\x01\x1F".to_string()), None, None, None)
            .unwrap()
            .unwrap();
        assert_eq!(s, r#""\u0001\u001F""#);
    }

    // ── json_stringify — arrays ───────────────────────────────────────────────

    #[test]
    fn test_stringify_empty_array() {
        let arr = JsonValue::new_array();
        let s = json_stringify(&arr, None, None, None).unwrap().unwrap();
        assert_eq!(s, "[]");
    }

    #[test]
    fn test_stringify_flat_array() {
        let arr = JsonValue::Array(Rc::new(RefCell::new(vec![
            JsonValue::Number(1.0),
            JsonValue::Number(2.0),
            JsonValue::Number(3.0),
        ])));
        let s = json_stringify(&arr, None, None, None).unwrap().unwrap();
        assert_eq!(s, "[1,2,3]");
    }

    // ── json_stringify — objects ──────────────────────────────────────────────

    #[test]
    fn test_stringify_empty_object() {
        let obj = JsonValue::new_object();
        let s = json_stringify(&obj, None, None, None).unwrap().unwrap();
        assert_eq!(s, "{}");
    }

    #[test]
    fn test_stringify_flat_object() {
        let obj = JsonValue::Object(Rc::new(RefCell::new(vec![
            ("a".to_string(), JsonValue::Number(1.0)),
            ("b".to_string(), JsonValue::Bool(true)),
        ])));
        let s = json_stringify(&obj, None, None, None).unwrap().unwrap();
        assert_eq!(s, r#"{"a":1,"b":true}"#);
    }

    // ── json_stringify — nested objects ───────────────────────────────────────

    #[test]
    fn test_stringify_nested_objects() {
        let inner = JsonValue::Object(Rc::new(RefCell::new(vec![(
            "inner".to_string(),
            JsonValue::Number(42.0),
        )])));
        let outer = JsonValue::Object(Rc::new(RefCell::new(vec![("outer".to_string(), inner)])));
        let s = json_stringify(&outer, None, None, None).unwrap().unwrap();
        assert_eq!(s, r#"{"outer":{"inner":42}}"#);
    }

    // ── json_stringify — indentation (space) ─────────────────────────────────

    #[test]
    fn test_stringify_with_count_indent() {
        let obj = JsonValue::Object(Rc::new(RefCell::new(vec![(
            "x".to_string(),
            JsonValue::Number(1.0),
        )])));
        let s = json_stringify(&obj, None, Some(&JsonSpace::Count(2)), None)
            .unwrap()
            .unwrap();
        assert_eq!(s, "{\n  \"x\": 1\n}");
    }

    #[test]
    fn test_stringify_with_string_indent() {
        let obj = JsonValue::Object(Rc::new(RefCell::new(vec![(
            "x".to_string(),
            JsonValue::Number(1.0),
        )])));
        let s = json_stringify(&obj, None, Some(&JsonSpace::Str("\t".to_string())), None)
            .unwrap()
            .unwrap();
        assert_eq!(s, "{\n\t\"x\": 1\n}");
    }

    #[test]
    fn test_stringify_indent_clamped_to_10() {
        let obj = JsonValue::Object(Rc::new(RefCell::new(vec![(
            "x".to_string(),
            JsonValue::Number(1.0),
        )])));
        // 20 spaces requested; should be clamped to 10.
        let s = json_stringify(&obj, None, Some(&JsonSpace::Count(20)), None)
            .unwrap()
            .unwrap();
        let expected = "{\n          \"x\": 1\n}";
        assert_eq!(s, expected);
    }

    // ── json_stringify — replacer (array) ────────────────────────────────────

    #[test]
    fn test_stringify_replacer_array_filters_properties() {
        let obj = JsonValue::Object(Rc::new(RefCell::new(vec![
            ("a".to_string(), JsonValue::Number(1.0)),
            ("b".to_string(), JsonValue::Number(2.0)),
            ("c".to_string(), JsonValue::Number(3.0)),
        ])));
        let replacer = JsonReplacer::Array(vec!["a".to_string(), "c".to_string()]);
        let s = json_stringify(&obj, Some(&replacer), None, None)
            .unwrap()
            .unwrap();
        assert_eq!(s, r#"{"a":1,"c":3}"#);
    }

    // ── json_stringify — replacer (function) ─────────────────────────────────

    #[test]
    fn test_stringify_replacer_function_transforms_values() {
        let obj = JsonValue::Object(Rc::new(RefCell::new(vec![
            ("a".to_string(), JsonValue::Number(1.0)),
            ("b".to_string(), JsonValue::Number(2.0)),
        ])));
        let replacer = JsonReplacer::Function(&|_key, val| {
            Ok(match val {
                JsonValue::Number(n) => Some(JsonValue::Number(n * 10.0)),
                other => Some(other.clone()),
            })
        });
        let s = json_stringify(&obj, Some(&replacer), None, None)
            .unwrap()
            .unwrap();
        assert_eq!(s, r#"{"a":10,"b":20}"#);
    }

    #[test]
    fn test_stringify_replacer_function_omits_value() {
        let obj = JsonValue::Object(Rc::new(RefCell::new(vec![
            ("keep".to_string(), JsonValue::Number(1.0)),
            ("drop".to_string(), JsonValue::Number(2.0)),
        ])));
        let replacer = JsonReplacer::Function(&|key, val| {
            if key == "drop" {
                Ok(None)
            } else {
                Ok(Some(val.clone()))
            }
        });
        let s = json_stringify(&obj, Some(&replacer), None, None)
            .unwrap()
            .unwrap();
        assert_eq!(s, r#"{"keep":1}"#);
    }

    // ── json_stringify — toJSON hook ──────────────────────────────────────────

    #[test]
    fn test_stringify_to_json_hook() {
        let obj = JsonValue::Object(Rc::new(RefCell::new(vec![(
            "val".to_string(),
            JsonValue::Number(1.0),
        )])));
        // The to_json hook replaces any object with the string "custom".
        let hook: &dyn Fn(&str, &JsonValue) -> Option<JsonValue> = &|_key, v| {
            if v.is_object() {
                Some(JsonValue::Str("custom".to_string()))
            } else {
                None
            }
        };
        let s = json_stringify(&obj, None, None, Some(hook))
            .unwrap()
            .unwrap();
        assert_eq!(s, r#""custom""#);
    }

    // ── json_stringify — circular reference detection ─────────────────────────

    #[test]
    fn test_stringify_circular_array_detected() {
        // Build: outer = [inner], inner = [outer]  (circular).
        let inner: Rc<RefCell<Vec<JsonValue>>> = Rc::new(RefCell::new(vec![]));
        let outer = JsonValue::Array(inner.clone());
        inner.borrow_mut().push(outer.clone());

        let result = json_stringify(&outer, None, None, None);
        assert!(
            matches!(&result, Err(StatorError::TypeError(msg)) if msg.contains("circular")),
            "expected circular structure TypeError, got {result:?}"
        );
    }

    #[test]
    fn test_stringify_circular_object_detected() {
        // Build: obj.self = obj  (circular).
        let obj: Rc<RefCell<Vec<(String, JsonValue)>>> = Rc::new(RefCell::new(vec![]));
        let self_ref = JsonValue::Object(obj.clone());
        obj.borrow_mut()
            .push(("self".to_string(), self_ref.clone()));

        let result = json_stringify(&self_ref, None, None, None);
        assert!(
            matches!(&result, Err(StatorError::TypeError(msg)) if msg.contains("circular")),
            "expected circular structure TypeError, got {result:?}"
        );
    }

    // ── round-trip ────────────────────────────────────────────────────────────

    #[test]
    fn test_round_trip_primitives() {
        for text in &["null", "true", "false", "42", "3.14", r#""hello""#] {
            let parsed = json_parse(text, None).unwrap();
            let stringified = json_stringify(&parsed, None, None, None).unwrap().unwrap();
            assert_eq!(&stringified, text, "round-trip failed for {text}");
        }
    }

    #[test]
    fn test_round_trip_nested_object() {
        let original = r#"{"a":1,"b":[2,3],"c":{"d":null}}"#;
        let parsed = json_parse(original, None).unwrap();
        let stringified = json_stringify(&parsed, None, None, None).unwrap().unwrap();
        assert_eq!(stringified, original);
    }

    #[test]
    fn test_round_trip_unicode_string() {
        let original = r#""café 日本語 😀""#;
        let parsed = json_parse(original, None).unwrap();
        let stringified = json_stringify(&parsed, None, None, None).unwrap().unwrap();
        assert_eq!(stringified, original);
    }

    // ── json_stringify_js_value ───────────────────────────────────────────────

    #[test]
    fn test_js_value_stringify_bigint_throws_type_error() {
        let result = json_stringify_js_value(&JsValue::BigInt(42), None, None);
        assert!(
            matches!(&result, Err(StatorError::TypeError(msg)) if msg.contains("BigInt")),
            "expected BigInt TypeError, got {result:?}"
        );
    }

    #[test]
    fn test_js_value_stringify_undefined_returns_none() {
        let result = json_stringify_js_value(&JsValue::Undefined, None, None).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_js_value_stringify_null() {
        let result = json_stringify_js_value(&JsValue::Null, None, None)
            .unwrap()
            .unwrap();
        assert_eq!(result, "null");
    }

    #[test]
    fn test_js_value_stringify_smi() {
        let result = json_stringify_js_value(&JsValue::Smi(7), None, None)
            .unwrap()
            .unwrap();
        assert_eq!(result, "7");
    }

    #[test]
    fn test_js_value_stringify_boolean() {
        let result = json_stringify_js_value(&JsValue::Boolean(true), None, None)
            .unwrap()
            .unwrap();
        assert_eq!(result, "true");
    }

    #[test]
    fn test_js_value_stringify_heap_number_nan_null() {
        let result = json_stringify_js_value(&JsValue::HeapNumber(f64::NAN), None, None)
            .unwrap()
            .unwrap();
        assert_eq!(result, "null");
    }

    #[test]
    fn test_js_value_stringify_array() {
        let arr = JsValue::new_array(vec![
            JsValue::Smi(1),
            JsValue::Boolean(false),
            JsValue::Null,
        ]);
        let result = json_stringify_js_value(&arr, None, None).unwrap().unwrap();
        assert_eq!(result, "[1,false,null]");
    }

    // ── js_value_to_json: toJSON method ──────────────────────────────────────

    #[test]
    fn test_js_value_to_json_with_to_json_method() {
        use crate::objects::property_map::PropertyMap;

        let mut inner = PropertyMap::new();
        inner.insert("value".into(), JsValue::Smi(42));
        inner.insert(
            "toJSON".into(),
            JsValue::NativeFunction(Rc::new(|_args| Ok(JsValue::String("replaced".into())))),
        );
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(inner)));

        let json = js_value_to_json(&obj).unwrap().unwrap();
        assert_eq!(json, JsonValue::Str("replaced".to_string()));
    }

    #[test]
    fn test_js_value_stringify_plain_object_with_replacer() {
        use crate::objects::property_map::PropertyMap;

        let mut map = PropertyMap::new();
        map.insert("a".into(), JsValue::Smi(1));
        map.insert("b".into(), JsValue::Smi(2));
        map.insert("c".into(), JsValue::Smi(3));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));

        let replacer = JsonReplacer::Array(vec!["a".to_string(), "c".to_string()]);
        let s = json_stringify_js_value(&obj, Some(&replacer), None)
            .unwrap()
            .unwrap();
        assert!(s.contains("\"a\""), "should contain a: {s}");
        assert!(s.contains("\"c\""), "should contain c: {s}");
        assert!(!s.contains("\"b\""), "should not contain b: {s}");
    }

    #[test]
    fn test_js_value_stringify_with_space() {
        let s = json_stringify_js_value(
            &JsValue::new_array(vec![JsValue::Smi(1), JsValue::Smi(2)]),
            None,
            Some(&JsonSpace::Count(2)),
        )
        .unwrap()
        .unwrap();
        assert_eq!(s, "[\n  1,\n  2\n]");
    }

    // ── JSON.stringify property enumeration order conformance ─────────────

    #[test]
    fn test_stringify_plain_object_integer_keys_first() {
        use crate::objects::property_map::PropertyMap;
        let mut map = PropertyMap::new();
        map.insert("b".to_string(), JsValue::Smi(1));
        map.insert("1".to_string(), JsValue::Smi(2));
        map.insert("a".to_string(), JsValue::Smi(3));
        map.insert("0".to_string(), JsValue::Smi(4));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        let json = js_value_to_json(&obj).unwrap().unwrap();
        let s = json_stringify(&json, None, None, None).unwrap().unwrap();
        // Keys should be: "0", "1", "b", "a"
        assert_eq!(s, r#"{"0":4,"1":2,"b":1,"a":3}"#);
    }

    #[test]
    fn test_stringify_plain_object_sparse_indices() {
        use crate::objects::property_map::PropertyMap;
        let mut map = PropertyMap::new();
        map.insert("2".to_string(), JsValue::String("c".into()));
        map.insert("0".to_string(), JsValue::String("a".into()));
        map.insert("1".to_string(), JsValue::String("b".into()));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        let json = js_value_to_json(&obj).unwrap().unwrap();
        let s = json_stringify(&json, None, None, None).unwrap().unwrap();
        assert_eq!(s, r#"{"0":"a","1":"b","2":"c"}"#);
    }
}
