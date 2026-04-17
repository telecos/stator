//! Shared helpers for parser and bytecode fuzz targets.
//!
//! This library is only compiled as part of the `stator-fuzz` package and must
//! not be used from the main `stator_js` crate.

use stator_js::parser::ast::{
    BinaryExpr, BinaryOp, BoolLit, Expr, ExprStmt, Ident, IfStmt, NullLit, NumLit, Pat, Program,
    ProgramItem, ReturnStmt, SourceType, Stmt, StringLit, VarDecl, VarDeclarator, VarKind,
};
use stator_js::parser::scanner::{Position, Span};

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
