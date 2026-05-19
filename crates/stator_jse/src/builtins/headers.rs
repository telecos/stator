//! WHATWG Fetch `Headers` data model (validation, normalization, storage).
//!
//! This module implements only the data-structure semantics of the `Headers`
//! interface — name/value validation, ASCII-lowercase name normalization,
//! HTTP-whitespace value trimming, list mutation, lookup, and the spec's
//! "sort and combine" iteration order. It does **not** implement fetch,
//! request/response bodies, streams, or any forbidden-header policy: those
//! require a host network stack and request/response context which this
//! standalone engine intentionally does not provide.
//!
//! Supported scope:
//!
//! * Construction from no init, another `Headers` value (any list-shaped
//!   value with the same internal storage), a sequence of `[name, value]`
//!   pairs, or a record-like object.
//! * `append`, `delete`, `get`, `getSetCookie`, `has`, `set`, `forEach`,
//!   and iteration via `keys` / `values` / `entries` / `@@iterator`.
//! * Comma-joining of duplicate values for non-`Set-Cookie` headers on
//!   read (`get`) and iteration; `Set-Cookie` is preserved as separate
//!   entries on iteration and exposed via `getSetCookie`.
//!
//! Out-of-scope and intentionally rejected:
//!
//! * Forbidden request-header / response-header / no-cors policy — these
//!   require fetch integration which is fail-closed in this engine.
//! * Live coupling between a `Request`/`Response` and its `Headers` — both
//!   `Request` and `Response` are fail-closed constructors here.

/// Header list entry: `(lowercased_name, value)` preserving insertion order.
pub type HeaderList = Vec<(String, String)>;

/// Returns `true` when `b` is a valid HTTP token byte (RFC 7230 token).
#[inline]
fn is_http_token_byte(b: u8) -> bool {
    matches!(
        b,
        b'!' | b'#'
            | b'$'
            | b'%'
            | b'&'
            | b'\''
            | b'*'
            | b'+'
            | b'-'
            | b'.'
            | b'^'
            | b'_'
            | b'`'
            | b'|'
            | b'~'
            | b'0'..=b'9'
            | b'A'..=b'Z'
            | b'a'..=b'z'
    )
}

/// Returns `true` when `b` is HTTP whitespace (HTAB, LF, CR, SP).
#[inline]
fn is_http_whitespace(b: u8) -> bool {
    matches!(b, 0x09 | 0x0A | 0x0D | 0x20)
}

/// Validate a header name per the Fetch spec.
///
/// A valid name is a non-empty sequence of HTTP token bytes (ASCII only).
/// Returns the ASCII-lowercased name on success.
pub fn validate_and_normalize_name(name: &str) -> Result<String, String> {
    if name.is_empty() {
        return Err("Headers: invalid header name (empty)".into());
    }
    if !name.bytes().all(is_http_token_byte) {
        return Err(format!("Headers: invalid header name {name:?}"));
    }
    Ok(name.to_ascii_lowercase())
}

/// Validate a header value per the Fetch spec.
///
/// A valid value must not contain NUL, CR, or LF bytes. Note that the
/// caller is responsible for trimming HTTP whitespace from the start and
/// end of the value *before* validation, per the Fetch spec "normalize a
/// byte sequence" algorithm.
pub fn validate_value(value: &str) -> Result<(), String> {
    if value.bytes().any(|b| matches!(b, 0x00 | 0x0A | 0x0D)) {
        return Err("Headers: invalid header value (contains NUL/CR/LF)".into());
    }
    Ok(())
}

/// Trim leading and trailing HTTP whitespace from `value`.
pub fn normalize_value(value: &str) -> String {
    let bytes = value.as_bytes();
    let start = bytes
        .iter()
        .position(|b| !is_http_whitespace(*b))
        .unwrap_or(bytes.len());
    let end = bytes
        .iter()
        .rposition(|b| !is_http_whitespace(*b))
        .map(|i| i + 1)
        .unwrap_or(start);
    // SAFETY-equivalent: HTTP whitespace bytes are all ASCII, so slicing on
    // those boundaries preserves UTF-8 validity.
    value[start..end].to_string()
}

/// Append `(name, value)` to the list. Both inputs must already be
/// normalized/validated by the caller.
pub fn list_append(list: &mut HeaderList, name: String, value: String) {
    list.push((name, value));
}

/// Delete every entry whose name equals `name` (already lowercased).
pub fn list_delete(list: &mut HeaderList, name: &str) {
    list.retain(|(k, _)| k != name);
}

/// Return `true` if any entry has name `name` (already lowercased).
pub fn list_has(list: &HeaderList, name: &str) -> bool {
    list.iter().any(|(k, _)| k == name)
}

/// Per Fetch spec "get a structured field value": combine all values for
/// `name` with `, ` separator (including `Set-Cookie`). Returns `None`
/// when the header is absent.
pub fn list_get_combined(list: &HeaderList, name: &str) -> Option<String> {
    let mut found = false;
    let mut out = String::new();
    for (k, v) in list {
        if k == name {
            if found {
                out.push_str(", ");
            }
            out.push_str(v);
            found = true;
        }
    }
    if found { Some(out) } else { None }
}

/// Per Fetch spec, replace: remove all entries whose name equals `name`
/// and append `(name, value)` exactly once.
pub fn list_set(list: &mut HeaderList, name: String, value: String) {
    list.retain(|(k, _)| k != &name);
    list.push((name, value));
}

/// Return all `Set-Cookie` values in insertion order.
pub fn list_get_set_cookie(list: &HeaderList) -> Vec<String> {
    list.iter()
        .filter(|(k, _)| k == "set-cookie")
        .map(|(_, v)| v.clone())
        .collect()
}

/// Produce the spec's "sort and combine" iteration view: names sorted by
/// byte ordering, with non-`Set-Cookie` duplicates merged via `, ` and
/// `Set-Cookie` entries kept as separate items in their original order.
pub fn sort_and_combine(list: &HeaderList) -> Vec<(String, String)> {
    let mut names: Vec<&str> = list.iter().map(|(k, _)| k.as_str()).collect();
    names.sort();
    names.dedup();
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        if name == "set-cookie" {
            for (k, v) in list {
                if k == name {
                    out.push((name.to_string(), v.clone()));
                }
            }
        } else {
            let combined = list_get_combined(list, name).unwrap_or_default();
            out.push((name.to_string(), combined));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_and_normalize_name_accepts_token_chars() {
        assert_eq!(
            validate_and_normalize_name("Content-Type").unwrap(),
            "content-type"
        );
        assert_eq!(validate_and_normalize_name("X-Foo_9").unwrap(), "x-foo_9");
    }

    #[test]
    fn test_validate_and_normalize_name_rejects_invalid() {
        assert!(validate_and_normalize_name("").is_err());
        assert!(validate_and_normalize_name("a b").is_err());
        assert!(validate_and_normalize_name("a:b").is_err());
        assert!(validate_and_normalize_name("a\n").is_err());
        assert!(validate_and_normalize_name("á").is_err());
    }

    #[test]
    fn test_validate_value_rejects_nul_cr_lf() {
        assert!(validate_value("ok").is_ok());
        assert!(validate_value("hello world").is_ok());
        assert!(validate_value("bad\0").is_err());
        assert!(validate_value("bad\r").is_err());
        assert!(validate_value("bad\n").is_err());
    }

    #[test]
    fn test_normalize_value_trims_http_whitespace() {
        assert_eq!(normalize_value("  hello  "), "hello");
        assert_eq!(normalize_value("\t hi \r\n"), "hi");
        assert_eq!(normalize_value("   "), "");
        assert_eq!(normalize_value("a  b"), "a  b");
    }

    #[test]
    fn test_list_append_and_get_combined() {
        let mut list: HeaderList = Vec::new();
        list_append(&mut list, "x-a".into(), "1".into());
        list_append(&mut list, "x-a".into(), "2".into());
        assert_eq!(list_get_combined(&list, "x-a"), Some("1, 2".into()));
        assert_eq!(list_get_combined(&list, "x-missing"), None);
    }

    #[test]
    fn test_list_set_removes_other_and_appends() {
        let mut list: HeaderList = Vec::new();
        list_append(&mut list, "a".into(), "1".into());
        list_append(&mut list, "b".into(), "2".into());
        list_append(&mut list, "a".into(), "3".into());
        list_set(&mut list, "a".into(), "X".into());
        assert_eq!(
            list,
            vec![("b".to_string(), "2".to_string()), ("a".into(), "X".into())]
        );
    }

    #[test]
    fn test_list_delete_and_has() {
        let mut list: HeaderList = Vec::new();
        list_append(&mut list, "a".into(), "1".into());
        list_append(&mut list, "a".into(), "2".into());
        list_append(&mut list, "b".into(), "3".into());
        assert!(list_has(&list, "a"));
        list_delete(&mut list, "a");
        assert!(!list_has(&list, "a"));
        assert!(list_has(&list, "b"));
    }

    #[test]
    fn test_sort_and_combine_with_set_cookie() {
        let mut list: HeaderList = Vec::new();
        list_append(&mut list, "x-b".into(), "2".into());
        list_append(&mut list, "x-a".into(), "1".into());
        list_append(&mut list, "set-cookie".into(), "k=1".into());
        list_append(&mut list, "x-a".into(), "3".into());
        list_append(&mut list, "set-cookie".into(), "k=2".into());
        let view = sort_and_combine(&list);
        assert_eq!(
            view,
            vec![
                ("set-cookie".into(), "k=1".into()),
                ("set-cookie".into(), "k=2".into()),
                ("x-a".into(), "1, 3".into()),
                ("x-b".into(), "2".into()),
            ]
        );
    }

    #[test]
    fn test_get_set_cookie() {
        let mut list: HeaderList = Vec::new();
        list_append(&mut list, "set-cookie".into(), "a=1".into());
        list_append(&mut list, "x-other".into(), "y".into());
        list_append(&mut list, "set-cookie".into(), "b=2".into());
        assert_eq!(list_get_set_cookie(&list), vec!["a=1", "b=2"]);
    }
}
