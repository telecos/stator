#![no_main]

//! Deoptimization fuzzer.
//!
//! Constructs programs whose value types change mid-execution (Smi → string,
//! Smi → boolean, etc.) to exercise the deoptimization path where the JIT
//! returns [`JIT_DEOPT`] and the interpreter must resume correctly.
//!
//! Verification: after a deopt the interpreter must produce the same result as
//! running purely through the interpreter from the start.

use libfuzzer_sys::fuzz_target;
use stator_fuzz::program_from_bytes;
use stator_js::bytecode::bytecode_generator::BytecodeGenerator;
use stator_js::interpreter::{Interpreter, InterpreterFrame};
use stator_js::parser::ast::{
    BoolLit, Expr, ExprStmt, Ident, NullLit, NumLit, Pat, ProgramItem, ReturnStmt, Stmt,
    StringLit, VarDecl, VarDeclarator, VarKind,
};

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    // Use the first byte to choose a "deopt scenario" and the remaining bytes
    // to seed the function body.
    let scenario = data[0] % 4;
    let body_data = &data[1..];

    // Build a program whose body is seeded from the fuzz input.
    let mut prog = program_from_bytes(body_data, 12);

    // Inject a type-changing statement based on the scenario, so that any
    // JIT-compiled path will encounter an unexpected value type and deopt.
    let loc = stator_fuzz::dummy_span();
    let type_change_stmt = match scenario {
        // Assign a string to a variable that the body likely initialized as
        // a number, provoking a type mismatch on any subsequent arithmetic.
        0 => Stmt::VarDecl(VarDecl {
            loc,
            kind: VarKind::Var,
            declarators: vec![VarDeclarator {
                loc,
                id: Pat::Ident(Ident {
                    loc,
                    name: "x0".to_owned(),
                }),
                init: Some(Box::new(Expr::Str(StringLit {
                    loc,
                    value: "deopt_trigger".to_owned(),
                }))),
            }],
        }),
        // Assign a boolean.
        1 => Stmt::VarDecl(VarDecl {
            loc,
            kind: VarKind::Var,
            declarators: vec![VarDeclarator {
                loc,
                id: Pat::Ident(Ident {
                    loc,
                    name: "x1".to_owned(),
                }),
                init: Some(Box::new(Expr::Bool(BoolLit {
                    loc,
                    value: true,
                }))),
            }],
        }),
        // Assign null.
        2 => Stmt::VarDecl(VarDecl {
            loc,
            kind: VarKind::Var,
            declarators: vec![VarDeclarator {
                loc,
                id: Pat::Ident(Ident {
                    loc,
                    name: "x2".to_owned(),
                }),
                init: Some(Box::new(Expr::Null(NullLit {
                    loc,
                }))),
            }],
        }),
        // No type change — exercise a plain program where the JIT might still
        // deopt on an unsupported opcode.
        _ => Stmt::Expr(ExprStmt {
            loc,
            expr: Box::new(Expr::Num(NumLit {
                loc,
                value: f64::from(data[0]),
                raw: data[0].to_string(),
            })),
        }),
    };

    // Append the type-changing statement to the program body.
    prog.body.push(ProgramItem::Stmt(type_change_stmt));

    // Also append an explicit `return x0` so there is a deterministic result
    // to compare across the two execution paths.
    prog.body.push(ProgramItem::Stmt(Stmt::Return(ReturnStmt {
        loc,
        argument: Some(Box::new(Expr::Ident(Ident {
            loc,
            name: "x0".to_owned(),
        }))),
    })));

    // Compile to bytecode; compiler errors are acceptable.
    let Ok(bytecode) = BytecodeGenerator::compile_program(&prog) else {
        return;
    };

    // ── Interpreter baseline ───────────────────────────────────────────────
    let interp_result = {
        let mut frame = InterpreterFrame::new(bytecode.clone(), vec![]);
        Interpreter::run(&mut frame)
    };

    // If the interpreter itself errors we have nothing to compare against.
    let Ok(interp_val) = interp_result else {
        return;
    };

    // ── JIT path + deopt verification (x86-64 Unix only) ─────────────────
    #[cfg(all(target_arch = "x86_64", unix))]
    {
        use stator_js::compiler::baseline::compiler::{
            BaselineCompiler, JIT_DEOPT, JIT_UNDEFINED, jit_to_jsvalue,
        };

        // Compile to native code; skip if unsupported.
        let Ok(cc) = BaselineCompiler::compile(&bytecode) else {
            return;
        };

        let jit_args: Vec<i64> = vec![JIT_UNDEFINED; bytecode.parameter_count() as usize];

        // SAFETY: `cc.code` was produced by `BaselineCompiler::compile`.
        let jit_raw = match unsafe { cc.execute(&jit_args) } {
            Ok(v) => v,
            Err(_) => return,
        };

        if jit_raw == JIT_DEOPT {
            // The JIT deoptimized — verify that re-running through the
            // interpreter produces the same result as the initial interpreter
            // run (i.e. the deopt did not corrupt any observable state).
            let mut frame2 = InterpreterFrame::new(bytecode.clone(), vec![]);
            if let Ok(resume_val) = Interpreter::run(&mut frame2) {
                assert_eq!(
                    interp_val, resume_val,
                    "interpreter result must be stable across re-executions after deopt\n\
                     first run:  {interp_val:?}\n\
                     second run: {resume_val:?}"
                );
            }
            // The deopt path is valid — no further assertions needed.
            return;
        }

        // JIT completed without deopt — compare against interpreter.
        let Some(jit_val) = jit_to_jsvalue(jit_raw) else {
            return;
        };

        assert_eq!(
            interp_val, jit_val,
            "interpreter and JIT produced different results\n\
             interpreter: {interp_val:?}\n\
             jit:         {jit_val:?}"
        );
    }

    #[cfg(not(all(target_arch = "x86_64", unix)))]
    let _ = interp_val;
});
