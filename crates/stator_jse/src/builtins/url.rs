//! WHATWG-aligned `URL` and `URLSearchParams` data model and parser.
//!
//! This module implements a focused subset of the WHATWG URL Standard that is
//! sufficient for the page-script compatibility goals of Stator's runtime.
//! It is intentionally *not* a full URL parser: features that require IDN
//! tables, Unicode IDNA processing, or browser-internal context (such as
//! `blob:` object-URL lifetimes) are explicitly rejected so that no
//! "fake success" parse results are produced.
//!
//! Supported scope:
//!
//! * Special schemes: `http`, `https`, `ws`, `wss`, `ftp`, `file`.
//!   These parse authority (`userinfo@host:port`), path, query, and
//!   fragment, and serialize back with default-port collapsing and
//!   dot-segment removal.
//! * Non-special schemes (any other `ALPHA (ALPHA / DIGIT / + / - / .)*`
//!   scheme): treated as *opaque-path* URLs (`scheme:opaque[?query][#hash]`).
//!   The opaque path is preserved as a single string and no authority is
//!   parsed.
//! * Relative resolution against an absolute base URL per RFC 3986 §5.
//! * `URLSearchParams` as a standalone class with `append`, `set`, `get`,
//!   `getAll`, `has`, `delete`, `sort`, `toString`, and iteration.
//!
//! Out-of-scope and intentionally rejected (parser returns
//! [`UrlParseError`], the caller throws `TypeError`):
//!
//! * Non-ASCII host names (no IDN/punycode conversion).
//! * `URL.createObjectURL` / `URL.revokeObjectURL` (host-integrated).
//! * Live `url.searchParams` identity (the engine exposes standalone
//!   `URLSearchParams` instead; `url.searchParams` is absent).

use std::cell::RefCell;
use std::rc::Rc;

/// Parse error returned by [`parse_url`].
#[derive(Debug, Clone)]
pub struct UrlParseError(pub String);

impl std::fmt::Display for UrlParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Parsed URL state.
///
/// Fields are stored in already-serialized/encoded form so that the
/// component getters can return them directly without re-encoding.
#[derive(Debug, Clone)]
pub struct UrlData {
    /// Lowercase scheme without the trailing colon, e.g. `"https"`.
    pub scheme: String,
    /// `true` if the URL has a `//`-introduced authority section. Always
    /// true for special schemes; false for opaque-path non-special URLs.
    pub has_authority: bool,
    /// Percent-encoded username (may be empty).
    pub username: String,
    /// Percent-encoded password (may be empty).
    pub password: String,
    /// Serialized host. Empty string means "no host".
    /// Bracketed for IPv6 literals (`[::1]`).
    pub host: String,
    /// Optional port number, or `None` if absent / equal to the default
    /// port for the scheme.
    pub port: Option<u16>,
    /// Path. For special schemes this is the absolute path starting with
    /// `/` (or empty for `file://host` with no path). For non-special
    /// opaque URLs this is the raw opaque payload after `scheme:`.
    pub path: String,
    /// Query, shared with any linked `URLSearchParams` instance.
    pub query: Rc<RefCell<QueryPairs>>,
    /// Optional fragment without the leading `#`.
    pub fragment: Option<String>,
}

/// Query pair storage shared between `URL` and `URLSearchParams`.
pub type QueryPairs = Vec<(String, String)>;

/// Returns `true` for the WHATWG "special scheme" set.
pub fn is_special_scheme(scheme: &str) -> bool {
    matches!(scheme, "http" | "https" | "ws" | "wss" | "ftp" | "file")
}

/// Default port for special schemes (`None` for `file` and non-special).
pub fn default_port(scheme: &str) -> Option<u16> {
    match scheme {
        "http" | "ws" => Some(80),
        "https" | "wss" => Some(443),
        "ftp" => Some(21),
        _ => None,
    }
}

// ── Percent-encoding sets (WHATWG §1.3) ──────────────────────────────────

/// Encode set for the URL fragment.
fn in_fragment_encode_set(b: u8) -> bool {
    in_c0_control_or_space(b) || matches!(b, b' ' | b'"' | b'<' | b'>' | b'`')
}

/// Encode set for the URL path (special-scheme).
fn in_path_encode_set(b: u8) -> bool {
    in_fragment_encode_set(b) || matches!(b, b'?' | b'`' | b'{' | b'}' | b'#')
}

/// Encode set for the userinfo (username/password).
fn in_userinfo_encode_set(b: u8) -> bool {
    in_path_encode_set(b)
        || matches!(
            b,
            b'/' | b':' | b';' | b'=' | b'@' | b'[' | b'\\' | b']' | b'^' | b'|'
        )
}

fn in_c0_control_or_space(b: u8) -> bool {
    b <= 0x20 || b == 0x7f
}

/// Percent-encode a byte as `%XX` (uppercase hex).
fn pct_encode_byte(out: &mut String, b: u8) {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    out.push('%');
    out.push(HEX[(b >> 4) as usize] as char);
    out.push(HEX[(b & 0xf) as usize] as char);
}

/// Percent-encode all bytes of `input` whose byte value matches
/// `in_set`. Bytes outside the ASCII range (>= 0x80) are always encoded
/// per their UTF-8 representation.
fn percent_encode(input: &str, in_set: fn(u8) -> bool) -> String {
    let mut out = String::with_capacity(input.len());
    for &b in input.as_bytes() {
        if b >= 0x80 || in_set(b) {
            pct_encode_byte(&mut out, b);
        } else {
            out.push(b as char);
        }
    }
    out
}

/// Application/x-www-form-urlencoded byte serializer (WHATWG §5.2).
fn form_urlencoded_byte(out: &mut String, b: u8) {
    match b {
        0x2A | 0x2D | 0x2E | 0x30..=0x39 | 0x41..=0x5A | 0x5F | 0x61..=0x7A => {
            out.push(b as char);
        }
        0x20 => out.push('+'),
        _ => pct_encode_byte(out, b),
    }
}

/// Serialize a string per application/x-www-form-urlencoded.
pub fn form_urlencoded_serialize(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for &b in input.as_bytes() {
        form_urlencoded_byte(&mut out, b);
    }
    out
}

/// Decode a `+`-and-`%`-encoded `application/x-www-form-urlencoded` byte
/// string into a UTF-8 `String`, replacing invalid sequences with U+FFFD
/// per the WHATWG `UTF-8 decode without BOM or fail` algorithm step.
pub fn form_urlencoded_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'+' {
            out.push(b' ');
            i += 1;
        } else if b == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2])) {
                out.push((h << 4) | l);
                i += 3;
            } else {
                out.push(b);
                i += 1;
            }
        } else {
            out.push(b);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Parse an `application/x-www-form-urlencoded` byte string into pairs.
pub fn parse_query_pairs(input: &str) -> QueryPairs {
    if input.is_empty() {
        return Vec::new();
    }
    let mut pairs = Vec::new();
    for chunk in input.split('&') {
        if chunk.is_empty() {
            continue;
        }
        let (k, v) = match chunk.split_once('=') {
            Some((k, v)) => (k, v),
            None => (chunk, ""),
        };
        pairs.push((form_urlencoded_decode(k), form_urlencoded_decode(v)));
    }
    pairs
}

/// Serialize a query-pair list per the WHATWG `application/x-www-form-urlencoded`
/// serializer.
pub fn serialize_query_pairs(pairs: &QueryPairs) -> String {
    let mut out = String::new();
    for (i, (k, v)) in pairs.iter().enumerate() {
        if i > 0 {
            out.push('&');
        }
        out.push_str(&form_urlencoded_serialize(k));
        out.push('=');
        out.push_str(&form_urlencoded_serialize(v));
    }
    out
}

// ── Parser ───────────────────────────────────────────────────────────────

fn trim_c0_and_space(s: &str) -> &str {
    let bytes = s.as_bytes();
    let mut start = 0;
    let mut end = bytes.len();
    while start < end && bytes[start] <= 0x20 {
        start += 1;
    }
    while end > start && bytes[end - 1] <= 0x20 {
        end -= 1;
    }
    &s[start..end]
}

fn strip_ascii_tab_or_newline(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(*c, '\t' | '\n' | '\r'))
        .collect()
}

/// Try to extract a scheme prefix (`ALPHA (ALPHA/DIGIT/+/-/.)* :`).
/// Returns `(lowercased_scheme, rest_after_colon)`.
fn split_scheme(input: &str) -> Option<(String, &str)> {
    let bytes = input.as_bytes();
    if bytes.is_empty() || !bytes[0].is_ascii_alphabetic() {
        return None;
    }
    for (i, &b) in bytes.iter().enumerate() {
        if b == b':' && i > 0 {
            let scheme = input[..i].to_ascii_lowercase();
            return Some((scheme, &input[i + 1..]));
        }
        let ok = b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.');
        if !ok {
            return None;
        }
    }
    None
}

/// Validate an ASCII-only host for special schemes.
///
/// Returns the canonical (lower-cased for DNS-like) host or an error.
/// Rejects empty hosts (except when `allow_empty` is true) and any
/// non-ASCII characters (since IDN/punycode conversion is intentionally
/// not implemented).
fn parse_host_ascii(input: &str, allow_empty: bool) -> Result<String, UrlParseError> {
    if input.is_empty() {
        if allow_empty {
            return Ok(String::new());
        }
        return Err(UrlParseError("URL host is empty".into()));
    }
    if input.starts_with('[') {
        let end = input
            .find(']')
            .ok_or_else(|| UrlParseError("URL IPv6 host missing closing bracket".into()))?;
        if end + 1 != input.len() {
            return Err(UrlParseError("URL IPv6 host has trailing garbage".into()));
        }
        let inner = &input[1..end];
        if inner.is_empty()
            || !inner
                .bytes()
                .all(|b| b.is_ascii_hexdigit() || matches!(b, b':' | b'.'))
        {
            return Err(UrlParseError("URL IPv6 host is invalid".into()));
        }
        return Ok(format!("[{}]", inner.to_ascii_lowercase()));
    }
    for b in input.bytes() {
        if !b.is_ascii() {
            return Err(UrlParseError(
                "URL host contains non-ASCII characters (IDN not supported)".into(),
            ));
        }
        let forbidden = matches!(
            b,
            0x00..=0x20
                | b'#'
                | b'/'
                | b':'
                | b'<'
                | b'>'
                | b'?'
                | b'@'
                | b'['
                | b'\\'
                | b']'
                | b'^'
                | b'|'
                | 0x7f
        );
        if forbidden {
            return Err(UrlParseError(format!(
                "URL host contains forbidden character {:?}",
                b as char
            )));
        }
    }
    Ok(input.to_ascii_lowercase())
}

fn parse_port(input: &str) -> Result<Option<u16>, UrlParseError> {
    if input.is_empty() {
        return Ok(None);
    }
    if !input.bytes().all(|b| b.is_ascii_digit()) {
        return Err(UrlParseError(format!("URL port {input:?} is not numeric")));
    }
    let n: u32 = input
        .parse()
        .map_err(|_| UrlParseError(format!("URL port {input:?} is out of range")))?;
    if n > 65535 {
        return Err(UrlParseError(format!("URL port {input:?} is out of range")));
    }
    Ok(Some(n as u16))
}

/// RFC 3986 §5.2.4 "remove_dot_segments".
fn remove_dot_segments(path: &str) -> String {
    let mut output: Vec<&str> = Vec::new();
    let leading_slash = path.starts_with('/');
    let trailing_slash = path.ends_with('/') && path.len() > 1;
    for seg in path.split('/').filter(|s| !s.is_empty()) {
        match seg {
            "." => {}
            ".." => {
                output.pop();
            }
            other => output.push(other),
        }
    }
    let mut out = String::new();
    if leading_slash {
        out.push('/');
    }
    out.push_str(&output.join("/"));
    if trailing_slash && !out.ends_with('/') {
        out.push('/');
    }
    if out.is_empty() && (leading_slash || path == "/") {
        out.push('/');
    }
    if (path == "/." || path == "/.." || path.ends_with("/.") || path.ends_with("/.."))
        && !out.ends_with('/')
    {
        out.push('/');
    }
    out
}

/// Parse an absolute URL, or a relative URL with respect to `base`.
pub fn parse_url(input: &str, base: Option<&UrlData>) -> Result<UrlData, UrlParseError> {
    let cleaned = strip_ascii_tab_or_newline(trim_c0_and_space(input));
    parse_url_inner(&cleaned, base)
}

fn parse_url_inner(input: &str, base: Option<&UrlData>) -> Result<UrlData, UrlParseError> {
    if let Some((scheme, rest_after_colon)) = split_scheme(input) {
        return parse_absolute(&scheme, rest_after_colon, base);
    }

    // Relative parsing requires a base.
    let base = base.ok_or_else(|| {
        UrlParseError(format!(
            "URL constructor: {input:?} is not a valid absolute URL"
        ))
    })?;
    parse_relative(input, base)
}

fn parse_absolute(
    scheme: &str,
    after_colon: &str,
    base: Option<&UrlData>,
) -> Result<UrlData, UrlParseError> {
    if !is_special_scheme(scheme) {
        // Non-special, opaque-path URL.
        return parse_opaque(scheme, after_colon);
    }

    let special_uses_base_authority = matches!(scheme, "file")
        && base.is_some()
        && base.unwrap().scheme == scheme
        && !after_colon.starts_with("//")
        && after_colon.starts_with('/');

    // Special schemes: the input after `scheme:` MUST start with `//` to
    // introduce an authority — except for the relative cases handled by
    // `parse_relative`. We accept `\\` as `//` for compatibility (per
    // WHATWG step that translates backslashes for special schemes).
    let normalized: String = after_colon
        .chars()
        .map(|c| if c == '\\' { '/' } else { c })
        .collect();

    if !normalized.starts_with("//") {
        if special_uses_base_authority {
            // file: with single-slash absolute path inherits base host
            let b = base.unwrap();
            let mut data = UrlData {
                scheme: scheme.to_string(),
                has_authority: true,
                username: b.username.clone(),
                password: b.password.clone(),
                host: b.host.clone(),
                port: b.port,
                path: String::new(),
                query: Rc::new(RefCell::new(Vec::new())),
                fragment: None,
            };
            populate_path_query_fragment(&mut data, &normalized, scheme)?;
            return Ok(data);
        }
        return Err(UrlParseError(format!(
            "URL constructor: missing authority for special scheme {scheme:?}"
        )));
    }

    let rest = &normalized[2..];

    // Split rest into authority + path-query-fragment.
    let auth_end = rest
        .find(['/', '?', '#'])
        .unwrap_or(rest.len());
    let authority = &rest[..auth_end];
    let tail = &rest[auth_end..];

    let mut data = UrlData {
        scheme: scheme.to_string(),
        has_authority: true,
        username: String::new(),
        password: String::new(),
        host: String::new(),
        port: None,
        path: String::new(),
        query: Rc::new(RefCell::new(Vec::new())),
        fragment: None,
    };

    parse_authority(&mut data, authority, scheme)?;
    populate_path_query_fragment(&mut data, tail, scheme)?;

    Ok(data)
}

fn parse_authority(data: &mut UrlData, authority: &str, scheme: &str) -> Result<(), UrlParseError> {
    let (userinfo, hostport) = match authority.rfind('@') {
        Some(idx) => (Some(&authority[..idx]), &authority[idx + 1..]),
        None => (None, authority),
    };

    if let Some(ui) = userinfo {
        let (user, pass) = match ui.split_once(':') {
            Some((u, p)) => (u, Some(p)),
            None => (ui, None),
        };
        data.username = percent_encode(user, in_userinfo_encode_set);
        if let Some(p) = pass {
            data.password = percent_encode(p, in_userinfo_encode_set);
        }
    }

    // Split host and port. IPv6 hosts contain ':' inside brackets, so
    // search for the rightmost ':' after a possible closing ']'.
    let host_end = if hostport.starts_with('[') {
        let bracket = hostport
            .find(']')
            .ok_or_else(|| UrlParseError("URL IPv6 host missing closing bracket".into()))?;
        bracket + 1
    } else {
        hostport.find(':').unwrap_or(hostport.len())
    };

    let host_part = &hostport[..host_end];
    let allow_empty_host = scheme == "file";
    data.host = parse_host_ascii(host_part, allow_empty_host)?;

    if host_end < hostport.len() {
        if !hostport[host_end..].starts_with(':') {
            return Err(UrlParseError("URL has invalid port separator".into()));
        }
        let port_str = &hostport[host_end + 1..];
        let port = parse_port(port_str)?;
        // Collapse default port.
        if port != default_port(scheme) {
            data.port = port;
        }
    }

    Ok(())
}

fn populate_path_query_fragment(
    data: &mut UrlData,
    tail: &str,
    scheme: &str,
) -> Result<(), UrlParseError> {
    let mut s = tail;

    // Fragment.
    if let Some(idx) = s.find('#') {
        let frag = &s[idx + 1..];
        data.fragment = Some(percent_encode(frag, in_fragment_encode_set));
        s = &s[..idx];
    }

    // Query.
    if let Some(idx) = s.find('?') {
        let q = &s[idx + 1..];
        *data.query.borrow_mut() = parse_query_pairs(q);
        s = &s[..idx];
    }

    // Path.
    let normalized: String = if is_special_scheme(scheme) {
        s.chars().map(|c| if c == '\\' { '/' } else { c }).collect()
    } else {
        s.to_string()
    };

    let path = if normalized.is_empty() {
        if is_special_scheme(scheme) {
            "/".to_string()
        } else {
            String::new()
        }
    } else if normalized.starts_with('/') {
        normalized
    } else {
        format!("/{normalized}")
    };

    let encoded = percent_encode(&path, in_path_encode_set);
    data.path = if is_special_scheme(scheme) {
        remove_dot_segments(&encoded)
    } else {
        encoded
    };

    Ok(())
}

fn parse_opaque(scheme: &str, after_colon: &str) -> Result<UrlData, UrlParseError> {
    let mut data = UrlData {
        scheme: scheme.to_string(),
        has_authority: false,
        username: String::new(),
        password: String::new(),
        host: String::new(),
        port: None,
        path: String::new(),
        query: Rc::new(RefCell::new(Vec::new())),
        fragment: None,
    };

    let mut s = after_colon;
    if let Some(idx) = s.find('#') {
        let frag = &s[idx + 1..];
        data.fragment = Some(percent_encode(frag, in_fragment_encode_set));
        s = &s[..idx];
    }
    if let Some(idx) = s.find('?') {
        let q = &s[idx + 1..];
        *data.query.borrow_mut() = parse_query_pairs(q);
        s = &s[..idx];
    }

    data.path = percent_encode(s, in_path_encode_set_opaque);
    Ok(data)
}

/// Encode set for opaque-path URLs: slightly more permissive (does not
/// encode `?`, `#`, `{`, `}` since those would have been split off
/// already during opaque parsing).
fn in_path_encode_set_opaque(b: u8) -> bool {
    in_c0_control_or_space(b) || matches!(b, b'"' | b'<' | b'>' | b'`')
}

fn parse_relative(input: &str, base: &UrlData) -> Result<UrlData, UrlParseError> {
    // Disallow relative resolution against opaque-path bases (per WHATWG
    // "cannot-be-a-base-URL"). The only exception is bare fragment.
    if !base.has_authority && !input.starts_with('#') {
        return Err(UrlParseError(format!(
            "URL constructor: cannot resolve {input:?} against opaque base"
        )));
    }

    if let Some(rest) = input.strip_prefix('#') {
        let mut data = base.clone();
        data.query = Rc::new(RefCell::new(base.query.borrow().clone()));
        data.fragment = Some(percent_encode(rest, in_fragment_encode_set));
        return Ok(data);
    }
    if let Some(rest) = input.strip_prefix('?') {
        let mut data = base.clone();
        data.query = Rc::new(RefCell::new(parse_query_pairs(rest)));
        data.fragment = None;
        // Drop the original `?...#...` from the path: parse_relative does
        // not include the base's query/fragment in this case.
        return Ok(data);
    }

    if input.starts_with("//") {
        // Network-path reference: re-parse as `scheme://...`.
        let synthetic = format!("{}:{}", base.scheme, input);
        return parse_url_inner(&synthetic, Some(base));
    }

    if input.starts_with('/') {
        // Absolute path reference.
        let mut data = base.clone();
        data.query = Rc::new(RefCell::new(Vec::new()));
        data.fragment = None;
        populate_path_query_fragment(&mut data, input, &base.scheme)?;
        return Ok(data);
    }

    // Relative path reference.
    let mut data = base.clone();
    data.query = Rc::new(RefCell::new(Vec::new()));
    data.fragment = None;

    // Merge per RFC 3986 §5.2.3.
    let merged = if base.has_authority && base.path.is_empty() {
        format!("/{input}")
    } else {
        let base_path = &base.path;
        let prefix_end = base_path.rfind('/').map(|i| i + 1).unwrap_or(0);
        format!("{}{}", &base_path[..prefix_end], input)
    };

    populate_path_query_fragment(&mut data, &merged, &base.scheme)?;
    Ok(data)
}

// ── Serialization ────────────────────────────────────────────────────────

impl UrlData {
    /// Serialize the URL to its `href` form.
    pub fn href(&self) -> String {
        let mut out = String::new();
        out.push_str(&self.scheme);
        out.push(':');
        if self.has_authority {
            out.push_str("//");
            if !self.username.is_empty() || !self.password.is_empty() {
                out.push_str(&self.username);
                if !self.password.is_empty() {
                    out.push(':');
                    out.push_str(&self.password);
                }
                out.push('@');
            }
            out.push_str(&self.host);
            if let Some(p) = self.port {
                out.push(':');
                out.push_str(&p.to_string());
            }
            out.push_str(&self.path);
        } else {
            out.push_str(&self.path);
        }
        let query = self.query.borrow();
        if !query.is_empty() {
            out.push('?');
            out.push_str(&serialize_query_pairs(&query));
        }
        if let Some(ref f) = self.fragment {
            out.push('#');
            out.push_str(f);
        }
        out
    }

    /// Origin per WHATWG §6: `"scheme://host[:port]"` for special schemes
    /// with a host; `"null"` for opaque-path URLs and host-less URLs.
    pub fn origin(&self) -> String {
        if !self.has_authority || self.host.is_empty() || self.scheme == "file" {
            return "null".to_string();
        }
        if !matches!(
            self.scheme.as_str(),
            "http" | "https" | "ws" | "wss" | "ftp"
        ) {
            return "null".to_string();
        }
        let mut out = format!("{}://{}", self.scheme, self.host);
        if let Some(p) = self.port {
            out.push(':');
            out.push_str(&p.to_string());
        }
        out
    }

    /// `"host:port"` if a non-default port is set, else just `host`.
    pub fn host_with_port(&self) -> String {
        if let Some(p) = self.port {
            format!("{}:{}", self.host, p)
        } else {
            self.host.clone()
        }
    }

    /// Returns the serialized search (`""` if empty, else `"?..."`).
    pub fn search_string(&self) -> String {
        let q = self.query.borrow();
        if q.is_empty() {
            String::new()
        } else {
            format!("?{}", serialize_query_pairs(&q))
        }
    }

    /// Returns the serialized hash (`""` if absent, else `"#..."`).
    pub fn hash_string(&self) -> String {
        match &self.fragment {
            Some(f) if !f.is_empty() => format!("#{f}"),
            _ => String::new(),
        }
    }

    /// Replace the query from a possibly-`?`-prefixed string.
    pub fn set_search(&mut self, input: &str) {
        let s = input.strip_prefix('?').unwrap_or(input);
        if s.is_empty() {
            self.query.borrow_mut().clear();
        } else {
            *self.query.borrow_mut() = parse_query_pairs(s);
        }
    }

    /// Replace the fragment from a possibly-`#`-prefixed string. Setting
    /// to empty removes the fragment.
    pub fn set_hash(&mut self, input: &str) {
        let s = input.strip_prefix('#').unwrap_or(input);
        if s.is_empty() {
            self.fragment = None;
        } else {
            self.fragment = Some(percent_encode(s, in_fragment_encode_set));
        }
    }

    /// Replace the pathname. For special schemes the leading `/` is
    /// enforced and dot segments removed; for opaque-path URLs this is a
    /// no-op.
    pub fn set_pathname(&mut self, input: &str) -> Result<(), UrlParseError> {
        if !self.has_authority {
            return Ok(());
        }
        let mut data = UrlData {
            path: String::new(),
            ..self.clone_shallow_for_setter()
        };
        // Reuse path normalization by piping through populate.
        populate_path_query_fragment(&mut data, input, &self.scheme)?;
        // populate_path_query_fragment also touched query/fragment if they
        // were embedded in the input; only adopt the path itself.
        self.path = data.path;
        Ok(())
    }

    /// Replace the hostname.
    pub fn set_hostname(&mut self, input: &str) -> Result<(), UrlParseError> {
        if !self.has_authority {
            return Ok(());
        }
        let allow_empty = self.scheme == "file";
        self.host = parse_host_ascii(input, allow_empty)?;
        Ok(())
    }

    /// Replace the port. Empty input clears the port.
    pub fn set_port(&mut self, input: &str) -> Result<(), UrlParseError> {
        if !self.has_authority {
            return Ok(());
        }
        if input.is_empty() {
            self.port = None;
            return Ok(());
        }
        let port = parse_port(input)?;
        if port == default_port(&self.scheme) {
            self.port = None;
        } else {
            self.port = port;
        }
        Ok(())
    }

    /// Replace the username (percent-encoded).
    pub fn set_username(&mut self, input: &str) {
        if !self.has_authority {
            return;
        }
        self.username = percent_encode(input, in_userinfo_encode_set);
    }

    /// Replace the password (percent-encoded).
    pub fn set_password(&mut self, input: &str) {
        if !self.has_authority {
            return;
        }
        self.password = percent_encode(input, in_userinfo_encode_set);
    }

    /// Replace the protocol/scheme. Only allows swapping within the same
    /// "special / non-special" class to avoid losing or fabricating
    /// authority components.
    pub fn set_protocol(&mut self, input: &str) -> Result<(), UrlParseError> {
        let trimmed = input.strip_suffix(':').unwrap_or(input);
        if trimmed.is_empty() {
            return Err(UrlParseError("URL protocol cannot be empty".into()));
        }
        let new_scheme = trimmed.to_ascii_lowercase();
        if split_scheme(&format!("{new_scheme}:")).is_none() {
            return Err(UrlParseError(format!(
                "URL protocol {input:?} is not a valid scheme"
            )));
        }
        let same_class = is_special_scheme(&new_scheme) == is_special_scheme(&self.scheme);
        if !same_class {
            // Per spec, this is silently ignored; we surface that as a
            // no-op (no error) to match browser behavior.
            return Ok(());
        }
        // file<->non-file special schemes have different default port
        // semantics; collapse default ports for the new scheme.
        if self.port == default_port(&new_scheme) {
            self.port = None;
        }
        self.scheme = new_scheme;
        Ok(())
    }

    /// Replace the host:port portion.
    pub fn set_host(&mut self, input: &str) -> Result<(), UrlParseError> {
        if !self.has_authority {
            return Ok(());
        }
        // Find port boundary like in parse_authority.
        let host_end = if input.starts_with('[') {
            let bracket = input
                .find(']')
                .ok_or_else(|| UrlParseError("URL IPv6 host missing closing bracket".into()))?;
            bracket + 1
        } else {
            input.find(':').unwrap_or(input.len())
        };
        let host_part = &input[..host_end];
        let allow_empty = self.scheme == "file";
        let new_host = parse_host_ascii(host_part, allow_empty)?;
        let new_port = if host_end < input.len() {
            if !input[host_end..].starts_with(':') {
                return Err(UrlParseError("URL has invalid port separator".into()));
            }
            let p = parse_port(&input[host_end + 1..])?;
            if p == default_port(&self.scheme) {
                None
            } else {
                p
            }
        } else {
            self.port
        };
        self.host = new_host;
        self.port = new_port;
        Ok(())
    }

    fn clone_shallow_for_setter(&self) -> UrlData {
        UrlData {
            scheme: self.scheme.clone(),
            has_authority: self.has_authority,
            username: self.username.clone(),
            password: self.password.clone(),
            host: self.host.clone(),
            port: self.port,
            path: self.path.clone(),
            query: Rc::clone(&self.query),
            fragment: self.fragment.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(input: &str) -> UrlData {
        parse_url(input, None).expect("parse")
    }

    #[test]
    fn test_parse_url_basic_https() {
        let u = parse("https://user:pw@example.com:8443/a/b?x=1&y=2#frag");
        assert_eq!(u.scheme, "https");
        assert_eq!(u.username, "user");
        assert_eq!(u.password, "pw");
        assert_eq!(u.host, "example.com");
        assert_eq!(u.port, Some(8443));
        assert_eq!(u.path, "/a/b");
        assert_eq!(u.query.borrow().len(), 2);
        assert_eq!(u.fragment.as_deref(), Some("frag"));
    }

    #[test]
    fn test_parse_url_default_port_collapsed() {
        let u = parse("http://example.com:80/x");
        assert_eq!(u.port, None);
        assert_eq!(u.href(), "http://example.com/x");
    }

    #[test]
    fn test_parse_url_dot_segments_removed() {
        let u = parse("https://a.test/x/./y/../z");
        assert_eq!(u.path, "/x/z");
    }

    #[test]
    fn test_parse_url_invalid_scheme() {
        assert!(parse_url("not a url", None).is_err());
    }

    #[test]
    fn test_parse_url_relative_against_base() {
        let base = parse("https://example.com/a/b/c?q=1#f");
        let u = parse_url("../x", Some(&base)).unwrap();
        assert_eq!(u.href(), "https://example.com/a/x");
    }

    #[test]
    fn test_parse_url_relative_absolute_path() {
        let base = parse("https://example.com/a/b");
        let u = parse_url("/z", Some(&base)).unwrap();
        assert_eq!(u.href(), "https://example.com/z");
    }

    #[test]
    fn test_parse_url_fragment_only_relative() {
        let base = parse("https://example.com/a?q=1");
        let u = parse_url("#new", Some(&base)).unwrap();
        assert_eq!(u.href(), "https://example.com/a?q=1#new");
    }

    #[test]
    fn test_parse_url_opaque() {
        let u = parse("mailto:user@example.com?subject=Hi");
        assert_eq!(u.scheme, "mailto");
        assert!(!u.has_authority);
        assert_eq!(u.path, "user@example.com");
        assert_eq!(u.query.borrow().len(), 1);
        assert_eq!(u.origin(), "null");
    }

    #[test]
    fn test_parse_url_non_ascii_host_rejected() {
        assert!(parse_url("https://exämple.com/", None).is_err());
    }

    #[test]
    fn test_parse_url_ipv6() {
        let u = parse("https://[::1]:8443/x");
        assert_eq!(u.host, "[::1]");
        assert_eq!(u.port, Some(8443));
        assert_eq!(u.href(), "https://[::1]:8443/x");
    }

    #[test]
    fn test_serialize_query_pairs_form_encoding() {
        let pairs = vec![
            ("a b".to_string(), "1+2".to_string()),
            ("x".to_string(), "&=".to_string()),
        ];
        assert_eq!(serialize_query_pairs(&pairs), "a+b=1%2B2&x=%26%3D");
    }

    #[test]
    fn test_parse_query_pairs_basic() {
        let pairs = parse_query_pairs("a=1&b=&c");
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0], ("a".to_string(), "1".to_string()));
        assert_eq!(pairs[1], ("b".to_string(), "".to_string()));
        assert_eq!(pairs[2], ("c".to_string(), "".to_string()));
    }

    #[test]
    fn test_set_search_replaces_query() {
        let mut u = parse("https://a.test/?old=1");
        u.set_search("new=2&also=3");
        assert_eq!(u.search_string(), "?new=2&also=3");
    }

    #[test]
    fn test_set_hash_clears_when_empty() {
        let mut u = parse("https://a.test/#frag");
        u.set_hash("");
        assert_eq!(u.hash_string(), "");
    }

    #[test]
    fn test_set_protocol_same_class_only() {
        let mut u = parse("https://a.test/");
        u.set_protocol("http").unwrap();
        assert_eq!(u.scheme, "http");
        // Cross-class change is silently ignored.
        u.set_protocol("mailto").unwrap();
        assert_eq!(u.scheme, "http");
    }
}
