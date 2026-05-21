//! Shared helpers for parser and bytecode fuzz targets.
//!
//! This library is only compiled as part of the `stator-fuzz` package and must
//! not be used from the main `stator_jse` crate.

use stator_jse::parser::ast::{
    BinaryExpr, BinaryOp, BoolLit, Expr, ExprStmt, Ident, IfStmt, NullLit, NumLit, Pat, Program,
    ProgramItem, ReturnStmt, SourceType, Stmt, StringLit, VarDecl, VarDeclarator, VarKind,
};
use stator_jse::parser::scanner::{Position, Span};

/// Returns a zero-offset dummy [`Span`] suitable for all AST node `loc` fields
/// in synthesised programs.
pub fn dummy_span() -> Span {
    let p = Position {
        offset: 0,
        line: 1,
        column: 1,
    };
    Span { start: p, end: p }
}

/// Build a leaf [`Expr`] driven by a single byte selector.
pub fn leaf_expr(sel: u8) -> Expr {
    match sel % 6 {
        0 => Expr::Null(NullLit { loc: dummy_span() }),
        1 => Expr::Bool(BoolLit {
            loc: dummy_span(),
            value: sel & 1 == 0,
        }),
        2 => {
            let v = f64::from(sel);
            Expr::Num(NumLit {
                loc: dummy_span(),
                value: v,
                raw: v.to_string(),
            })
        }
        3 => Expr::Str(StringLit {
            loc: dummy_span(),
            value: format!("s{sel}"),
        }),
        4 => Expr::Ident(Ident {
            loc: dummy_span(),
            name: format!("x{}", sel % 4),
        }),
        _ => Expr::Num(NumLit {
            loc: dummy_span(),
            value: 0.0,
            raw: "0".to_owned(),
        }),
    }
}

/// Build a simple [`Stmt`] driven by two bytes: `op` selects the statement
/// kind and `val` seeds the expression content.
pub fn make_stmt(op: u8, val: u8) -> Stmt {
    match op % 5 {
        0 => {
            // `var xN = <leaf>`
            Stmt::VarDecl(VarDecl {
                loc: dummy_span(),
                kind: VarKind::Var,
                declarators: vec![VarDeclarator {
                    loc: dummy_span(),
                    id: Pat::Ident(Ident {
                        loc: dummy_span(),
                        name: format!("x{}", val % 4),
                    }),
                    init: Some(Box::new(leaf_expr(val))),
                }],
            })
        }
        1 => {
            // `return <leaf>`
            Stmt::Return(ReturnStmt {
                loc: dummy_span(),
                argument: Some(Box::new(leaf_expr(val))),
            })
        }
        2 => {
            // `<leaf> + <leaf>`
            let lhs = leaf_expr(val);
            let rhs = leaf_expr(val.wrapping_add(1));
            Stmt::Expr(ExprStmt {
                loc: dummy_span(),
                expr: Box::new(Expr::Binary(Box::new(BinaryExpr {
                    loc: dummy_span(),
                    op: BinaryOp::Add,
                    left: Box::new(lhs),
                    right: Box::new(rhs),
                }))),
            })
        }
        3 => {
            // `if (<leaf>) { return <leaf>; }`
            Stmt::If(IfStmt {
                loc: dummy_span(),
                test: Box::new(leaf_expr(val)),
                consequent: Box::new(Stmt::Return(ReturnStmt {
                    loc: dummy_span(),
                    argument: Some(Box::new(leaf_expr(val.wrapping_add(2)))),
                })),
                alternate: None,
            })
        }
        _ => {
            // `null` expression statement
            Stmt::Expr(ExprStmt {
                loc: dummy_span(),
                expr: Box::new(Expr::Null(NullLit { loc: dummy_span() })),
            })
        }
    }
}

/// Build a [`Program`] from fuzz input bytes.
///
/// Bytes are consumed two at a time; each pair drives one statement via
/// [`make_stmt`].  The number of statements is capped at `max_stmts` to keep
/// compilation time bounded.
pub fn program_from_bytes(data: &[u8], max_stmts: usize) -> Program {
    let stmts: Vec<Stmt> = data
        .chunks(2)
        .take(max_stmts)
        .filter_map(|ch| {
            if ch.len() < 2 {
                None
            } else {
                Some(make_stmt(ch[0], ch[1]))
            }
        })
        .collect();

    Program {
        loc: dummy_span(),
        source_type: SourceType::Script,
        body: stmts.into_iter().map(ProgramItem::Stmt).collect(),
        is_strict: false,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Snapshot fuzz helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot envelope variants exercised by the `snapshot_load` fuzz target.
///
/// The variants intentionally mirror the on-disk magic prefixes recognised by
/// [`stator_jse::snapshot`]: `STSS` (legacy startup), `STSM` (manifest-aware
/// startup), and `STWC` (warm-context).  `Unknown` is included so the fuzzer
/// can also exercise the rejection paths for unrecognised inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotEnvelope {
    /// `STSS` legacy startup snapshot.
    Stss,
    /// `STSM` manifest-aware startup snapshot.
    Stsm,
    /// `STWC` warm-context snapshot.
    Stwc,
    /// No recognised magic prefix.
    Unknown,
}

impl SnapshotEnvelope {
    /// Magic prefix associated with this envelope, if any.
    pub fn magic(self) -> Option<&'static [u8; 4]> {
        match self {
            Self::Stss => Some(b"STSS"),
            Self::Stsm => Some(b"STSM"),
            Self::Stwc => Some(b"STWC"),
            Self::Unknown => None,
        }
    }
}

/// Classify a snapshot blob solely by its leading four bytes.
///
/// Returns the matching [`SnapshotEnvelope`] for `STSS` / `STSM` / `STWC`,
/// otherwise [`SnapshotEnvelope::Unknown`].  This helper does **not** perform
/// any structural validation — it is intentionally cheap so the fuzz target
/// can use it to pick which loader to drive without rejecting candidate inputs
/// that downstream code is expected to handle as failures.
pub fn classify_snapshot_magic(bytes: &[u8]) -> SnapshotEnvelope {
    if bytes.len() < 4 {
        return SnapshotEnvelope::Unknown;
    }
    match &bytes[..4] {
        b"STSS" => SnapshotEnvelope::Stss,
        b"STSM" => SnapshotEnvelope::Stsm,
        b"STWC" => SnapshotEnvelope::Stwc,
        _ => SnapshotEnvelope::Unknown,
    }
}

/// Reshape raw fuzz input into a candidate snapshot blob.
///
/// The first input byte is consumed as a *selector* that biases the resulting
/// blob toward one of four shapes:
///
/// | selector mod 5 | shape                                                   |
/// |----------------|---------------------------------------------------------|
/// | `0`            | tail bytes verbatim (no magic stamping)                 |
/// | `1`            | stamp `STSS` magic over the first 4 bytes of the tail   |
/// | `2`            | stamp `STSM` magic over the first 4 bytes of the tail   |
/// | `3`            | stamp `STWC` magic over the first 4 bytes of the tail   |
/// | `4`            | tail bytes verbatim (mirrors selector `0`)              |
///
/// When stamping is requested but the tail is shorter than 4 bytes, the magic
/// is *prepended* so the loader still sees the requested envelope kind.  This
/// keeps the fuzzer's coverage focused on each loader's framing and validation
/// logic rather than on the magic-prefix mismatch path alone.
///
/// Returns the resulting blob together with the envelope kind it is expected
/// to classify as.  The blob may still be malformed in every other respect —
/// that is the whole point of the fuzz target.
pub fn prepare_snapshot_bytes(data: &[u8]) -> (Vec<u8>, SnapshotEnvelope) {
    if data.is_empty() {
        return (Vec::new(), SnapshotEnvelope::Unknown);
    }
    let selector = data[0];
    let tail = &data[1..];
    let envelope = match selector % 5 {
        1 => SnapshotEnvelope::Stss,
        2 => SnapshotEnvelope::Stsm,
        3 => SnapshotEnvelope::Stwc,
        _ => return (tail.to_vec(), classify_snapshot_magic(tail)),
    };
    let magic = envelope.magic().expect("non-unknown envelope has magic");
    let mut out = Vec::with_capacity(tail.len().max(4));
    if tail.len() >= 4 {
        out.extend_from_slice(magic);
        out.extend_from_slice(&tail[4..]);
    } else {
        out.extend_from_slice(magic);
        out.extend_from_slice(tail);
    }
    (out, envelope)
}

#[cfg(test)]
mod snapshot_helper_tests {
    use super::*;

    #[test]
    fn test_classify_snapshot_magic_recognises_known_envelopes() {
        assert_eq!(classify_snapshot_magic(b"STSS...."), SnapshotEnvelope::Stss);
        assert_eq!(classify_snapshot_magic(b"STSM...."), SnapshotEnvelope::Stsm);
        assert_eq!(classify_snapshot_magic(b"STWC...."), SnapshotEnvelope::Stwc);
    }

    #[test]
    fn test_classify_snapshot_magic_handles_short_and_unknown_inputs() {
        assert_eq!(classify_snapshot_magic(b""), SnapshotEnvelope::Unknown);
        assert_eq!(classify_snapshot_magic(b"ST"), SnapshotEnvelope::Unknown);
        assert_eq!(classify_snapshot_magic(b"XXXX...."), SnapshotEnvelope::Unknown);
    }

    #[test]
    fn test_prepare_snapshot_bytes_empty_input_yields_unknown() {
        let (out, env) = prepare_snapshot_bytes(b"");
        assert!(out.is_empty());
        assert_eq!(env, SnapshotEnvelope::Unknown);
    }

    #[test]
    fn test_prepare_snapshot_bytes_passthrough_preserves_tail() {
        // selector 0 -> mod 5 == 0 -> no stamping.
        let (out, env) = prepare_snapshot_bytes(&[0u8, 1, 2, 3, 4]);
        assert_eq!(out, vec![1, 2, 3, 4]);
        assert_eq!(env, SnapshotEnvelope::Unknown);
    }

    #[test]
    fn test_prepare_snapshot_bytes_stamps_each_known_envelope() {
        for (selector, expected) in [
            (1u8, SnapshotEnvelope::Stss),
            (2, SnapshotEnvelope::Stsm),
            (3, SnapshotEnvelope::Stwc),
        ] {
            let mut input = vec![selector];
            input.extend_from_slice(&[0xaa, 0xbb, 0xcc, 0xdd, 0xee]);
            let (out, env) = prepare_snapshot_bytes(&input);
            assert_eq!(env, expected);
            assert_eq!(
                &out[..4],
                expected.magic().expect("known envelope has magic")
            );
            // Bytes after the 4-byte magic must be preserved verbatim.
            assert_eq!(&out[4..], &[0xee]);
        }
    }

    #[test]
    fn test_prepare_snapshot_bytes_stamps_magic_when_tail_is_short() {
        // selector 2 -> STSM, but tail is only 2 bytes.  The magic is
        // prepended so the loader still classifies the blob as STSM.
        let (out, env) = prepare_snapshot_bytes(&[2u8, 0x01, 0x02]);
        assert_eq!(env, SnapshotEnvelope::Stsm);
        assert_eq!(&out[..4], b"STSM");
        assert_eq!(&out[4..], &[0x01, 0x02]);
    }

    #[test]
    fn test_prepare_snapshot_bytes_selector_four_is_passthrough() {
        // selector 4 mod 5 == 4 -> passthrough.
        let (out, env) = prepare_snapshot_bytes(&[4u8, b'S', b'T', b'W', b'C', 0]);
        assert_eq!(out, vec![b'S', b'T', b'W', b'C', 0]);
        assert_eq!(env, SnapshotEnvelope::Stwc);
    }
}
