//! `URLPattern` pattern parser and matcher.
//!
//! Implements a conservative subset of the WHATWG `URLPattern` grammar that
//! covers the common cases used in real product scripts without ever
//! "fake matching" syntax it does not understand:
//!
//! Supported pattern tokens per component:
//!
//! * Literal characters (e.g. `/users`, `example.com`).
//! * `*` — zero-or-more wildcard. Captured as an anonymous numeric group
//!   `"0"`, `"1"`, ... in declaration order.
//! * `:name` — named group. Matches at least one character. `name` must be
//!   an ASCII identifier (`[A-Za-z_][A-Za-z0-9_]*`).
//! * `\<c>` — backslash escape for any single character, so users can
//!   embed literal `*`, `:`, `\`, `(`, `{`, etc.
//!
//! Delimiter semantics (matches the WHATWG component delimiter set):
//!
//! * In a [`PatternKind::Pathname`] pattern, `:name` does not cross `/`.
//! * In a [`PatternKind::Hostname`] pattern, `:name` does not cross `.`.
//! * In [`PatternKind::Other`] (protocol, username, password, port,
//!   search, hash), `:name` is unconstrained.
//! * `*` always consumes greedily across any character.
//!
//! Unsupported syntax (rejected by [`compile`] with a [`PatternParseError`],
//! which the host turns into a `TypeError`):
//!
//! * Grouping `{ ... }`, optional `?`, repeat `+`.
//! * Inline regex `( ... )`.
//! * Character classes `[ ... ]`.
//! * Any other regex-style metacharacter.

use std::collections::BTreeMap;

/// A single token in a compiled URL-pattern component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternToken {
    /// Literal text that must match exactly.
    Literal(String),
    /// A capture group. `allow_empty` is `true` for `*`, `false` for `:name`.
    /// `delimiter` is the component delimiter character that the group must
    /// not consume (e.g. `'/'` in pathname `:name`). `None` means the group
    /// may consume any character.
    Group {
        /// Group name. Anonymous `*` groups are numbered `"0"`, `"1"`, ...
        name: String,
        /// Delimiter character the group must not consume, if any.
        delimiter: Option<char>,
        /// Whether the group may match an empty span (`true` for `*`).
        allow_empty: bool,
    },
}

/// Compiled URL-pattern for a single component.
#[derive(Debug, Clone)]
pub struct CompiledPattern {
    /// The original pattern string (returned by component accessors).
    pub raw: String,
    /// The compiled token sequence.
    pub tokens: Vec<PatternToken>,
    /// Names of capture groups in declaration order.
    pub group_names: Vec<String>,
}

/// Pattern-parse error returned by [`compile`].
#[derive(Debug, Clone)]
pub struct PatternParseError(pub String);

impl std::fmt::Display for PatternParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Component kind, controlling the delimiter used by `:name` groups.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PatternKind {
    /// Pathname — `:name` does not cross `/`.
    Pathname,
    /// Hostname — `:name` does not cross `.`.
    Hostname,
    /// Anything else — `:name` may consume any character.
    Other,
}

impl PatternKind {
    fn named_delimiter(self) -> Option<char> {
        match self {
            Self::Pathname => Some('/'),
            Self::Hostname => Some('.'),
            Self::Other => None,
        }
    }
}

/// Compile a raw URL-pattern string into a [`CompiledPattern`].
pub fn compile(raw: &str, kind: PatternKind) -> Result<CompiledPattern, PatternParseError> {
    let chars: Vec<char> = raw.chars().collect();
    let mut tokens: Vec<PatternToken> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut literal = String::new();
    let mut anon: usize = 0;
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        match c {
            '\\' => {
                if i + 1 >= chars.len() {
                    return Err(PatternParseError(
                        "URLPattern: pattern ends with a dangling backslash escape".into(),
                    ));
                }
                literal.push(chars[i + 1]);
                i += 2;
            }
            '*' => {
                if !literal.is_empty() {
                    tokens.push(PatternToken::Literal(std::mem::take(&mut literal)));
                }
                let name = anon.to_string();
                anon += 1;
                tokens.push(PatternToken::Group {
                    name: name.clone(),
                    delimiter: None,
                    allow_empty: true,
                });
                names.push(name);
                i += 1;
            }
            ':' => {
                let start = i + 1;
                let mut end = start;
                while end < chars.len() {
                    let cc = chars[end];
                    let is_first = end == start;
                    let ok = if is_first {
                        cc.is_ascii_alphabetic() || cc == '_'
                    } else {
                        cc.is_ascii_alphanumeric() || cc == '_'
                    };
                    if !ok {
                        break;
                    }
                    end += 1;
                }
                if end == start {
                    return Err(PatternParseError(
                        "URLPattern: ':' must be followed by an ASCII identifier".into(),
                    ));
                }
                let name: String = chars[start..end].iter().collect();
                if !literal.is_empty() {
                    tokens.push(PatternToken::Literal(std::mem::take(&mut literal)));
                }
                tokens.push(PatternToken::Group {
                    name: name.clone(),
                    delimiter: kind.named_delimiter(),
                    allow_empty: false,
                });
                names.push(name);
                i = end;
            }
            '{' | '}' | '(' | ')' | '+' | '?' | '[' | ']' => {
                return Err(PatternParseError(format!(
                    "URLPattern: unsupported pattern character {c:?} (escape with `\\\\{c}` for literal use)"
                )));
            }
            _ => {
                literal.push(c);
                i += 1;
            }
        }
    }

    if !literal.is_empty() {
        tokens.push(PatternToken::Literal(literal));
    }

    Ok(CompiledPattern {
        raw: raw.to_string(),
        tokens,
        group_names: names,
    })
}

/// Attempt to match `input` against `pattern`. On success, return the
/// captured group map (group name → captured substring). Capture order
/// follows declaration order.
pub fn match_input(pattern: &CompiledPattern, input: &str) -> Option<BTreeMap<String, String>> {
    let chars: Vec<char> = input.chars().collect();
    let mut groups: BTreeMap<String, String> = BTreeMap::new();
    // Seed every declared group with an empty default so that callers
    // always see every name in the result map (matching browser shape).
    for name in &pattern.group_names {
        groups.insert(name.clone(), String::new());
    }
    if match_at(&pattern.tokens, 0, &chars, 0, &mut groups) {
        Some(groups)
    } else {
        None
    }
}

fn match_at(
    tokens: &[PatternToken],
    ti: usize,
    input: &[char],
    ii: usize,
    groups: &mut BTreeMap<String, String>,
) -> bool {
    if ti == tokens.len() {
        return ii == input.len();
    }
    match &tokens[ti] {
        PatternToken::Literal(s) => {
            let lc: Vec<char> = s.chars().collect();
            if ii + lc.len() > input.len() {
                return false;
            }
            for (k, ch) in lc.iter().enumerate() {
                if input[ii + k] != *ch {
                    return false;
                }
            }
            match_at(tokens, ti + 1, input, ii + lc.len(), groups)
        }
        PatternToken::Group {
            name,
            delimiter,
            allow_empty,
        } => {
            let min: usize = if *allow_empty { 0 } else { 1 };
            // Determine the largest span the group may consume, stopping
            // at the component delimiter character (if any).
            let mut max = 0usize;
            while ii + max < input.len() {
                let c = input[ii + max];
                if let Some(d) = delimiter
                    && c == *d
                {
                    break;
                }
                max += 1;
            }
            if max < min {
                return false;
            }
            // Greedy with backtracking: try longest match first, shrink down to `min`.
            let prev = groups.get(name).cloned();
            let mut len = max;
            loop {
                let consumed: String = input[ii..ii + len].iter().collect();
                groups.insert(name.clone(), consumed);
                if match_at(tokens, ti + 1, input, ii + len, groups) {
                    return true;
                }
                if len == min {
                    break;
                }
                len -= 1;
            }
            match prev {
                Some(p) => groups.insert(name.clone(), p),
                None => groups.remove(name),
            };
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compile_path(s: &str) -> CompiledPattern {
        compile(s, PatternKind::Pathname).expect("pattern should compile")
    }

    #[test]
    fn test_compile_literal_only() {
        let p = compile_path("/users/list");
        assert_eq!(
            p.tokens,
            vec![PatternToken::Literal("/users/list".to_string())]
        );
        assert!(p.group_names.is_empty());
    }

    #[test]
    fn test_compile_named_group() {
        let p = compile_path("/users/:id");
        assert_eq!(p.group_names, vec!["id".to_string()]);
        assert_eq!(
            p.tokens,
            vec![
                PatternToken::Literal("/users/".to_string()),
                PatternToken::Group {
                    name: "id".to_string(),
                    delimiter: Some('/'),
                    allow_empty: false,
                },
            ]
        );
    }

    #[test]
    fn test_compile_anonymous_wildcard_numbers_groups() {
        let p = compile_path("/*/x/*");
        assert_eq!(p.group_names, vec!["0".to_string(), "1".to_string()]);
    }

    #[test]
    fn test_compile_escape_special_chars() {
        let p = compile_path(r"/foo\*bar\:baz");
        assert_eq!(
            p.tokens,
            vec![PatternToken::Literal("/foo*bar:baz".to_string())]
        );
    }

    #[test]
    fn test_compile_rejects_braces() {
        assert!(compile("{foo}", PatternKind::Other).is_err());
    }

    #[test]
    fn test_compile_rejects_optional_modifier() {
        assert!(compile("/foo?", PatternKind::Pathname).is_err());
    }

    #[test]
    fn test_compile_rejects_paren_group() {
        assert!(compile("/(abc)", PatternKind::Pathname).is_err());
    }

    #[test]
    fn test_compile_rejects_dangling_backslash() {
        assert!(compile("/foo\\", PatternKind::Pathname).is_err());
    }

    #[test]
    fn test_compile_rejects_colon_without_name() {
        assert!(compile("/:", PatternKind::Pathname).is_err());
    }

    #[test]
    fn test_match_literal_exact() {
        let p = compile_path("/users/list");
        assert!(match_input(&p, "/users/list").is_some());
        assert!(match_input(&p, "/users/lists").is_none());
        assert!(match_input(&p, "/users").is_none());
    }

    #[test]
    fn test_match_named_group_in_pathname_does_not_cross_slash() {
        let p = compile_path("/users/:id");
        let g = match_input(&p, "/users/42").expect("should match");
        assert_eq!(g.get("id").map(String::as_str), Some("42"));
        // Slash inside :id is not allowed since pathname delimiter is `/`.
        assert!(match_input(&p, "/users/42/posts").is_none());
    }

    #[test]
    fn test_match_wildcard_crosses_slash() {
        let p = compile_path("/users/*");
        let g = match_input(&p, "/users/42/posts").expect("should match");
        assert_eq!(g.get("0").map(String::as_str), Some("42/posts"));
    }

    #[test]
    fn test_match_wildcard_allows_empty() {
        let p = compile_path("/users/*");
        let g = match_input(&p, "/users/").expect("should match");
        assert_eq!(g.get("0").map(String::as_str), Some(""));
    }

    #[test]
    fn test_match_named_group_in_hostname_does_not_cross_dot() {
        let p = compile("api.:domain", PatternKind::Hostname).unwrap();
        let g = match_input(&p, "api.example").expect("should match");
        assert_eq!(g.get("domain").map(String::as_str), Some("example"));
        assert!(match_input(&p, "api.example.com").is_none());
    }

    #[test]
    fn test_match_named_group_other_allows_anything() {
        let p = compile(":scheme", PatternKind::Other).unwrap();
        let g = match_input(&p, "https").expect("should match");
        assert_eq!(g.get("scheme").map(String::as_str), Some("https"));
    }

    #[test]
    fn test_match_full_anchored() {
        let p = compile_path("/x");
        assert!(match_input(&p, "/x/y").is_none());
        assert!(match_input(&p, " /x").is_none());
    }

    #[test]
    fn test_match_backtracking_two_groups() {
        let p = compile_path("/:a/:b");
        let g = match_input(&p, "/foo/bar").expect("should match");
        assert_eq!(g.get("a").map(String::as_str), Some("foo"));
        assert_eq!(g.get("b").map(String::as_str), Some("bar"));
    }

    #[test]
    fn test_match_no_match_returns_none() {
        let p = compile_path("/users/:id");
        assert!(match_input(&p, "/posts/42").is_none());
    }

    #[test]
    fn test_groups_seeded_with_empty_when_no_groups() {
        let p = compile_path("/users/list");
        let g = match_input(&p, "/users/list").unwrap();
        assert!(g.is_empty());
    }
}
