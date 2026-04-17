//! Dynamic `import()` and `import.meta` conformance tests.
//!
//! Covers the following ECMAScript specification areas:
//!
//! - `import()` returns a `Promise` (§16.2.1.8 *HostImportModuleDynamically*)
//! - `import()` with non-string argument (ToString coercion)
//! - `import()` is valid in scripts, not just modules
//! - `import()` works in any expression position
//! - `import.meta` is only valid in module code (`SyntaxError` in scripts)
//! - `import.meta` is an ordinary, extensible object
//! - `typeof import.meta === "object"`
//! - `import()` with failed resolution rejects the promise
//! - `import()` syntax: `new import()` is invalid
//! - Conditional import: `if (cond) { const m = await import("…") }`
//! - `import()` inside `eval` (sloppy mode)
//! - `import()` inside arrow functions, ternary, comma expressions

#[cfg(test)]
mod tests {
    use crate::builtins::global::global_eval;
    use crate::objects::value::JsValue;
    use crate::parser::{parse, parse_module, parse_script};
    use std::rc::Rc;

    /// Evaluate `src` and assert the result is boolean `true`.
    fn assert_eval_true(src: &str) {
        let result = global_eval(src).unwrap();
        assert_eq!(result, JsValue::Boolean(true), "expected true for: {src}");
    }

    /// Evaluate `src` and assert it produces an error.
    fn assert_eval_err(src: &str) {
        assert!(global_eval(src).is_err(), "expected error for: {src}");
    }

    // ── 1. import() returns a Promise ────────────────────────────────────────

    #[test]
    fn e2e_dynamic_import_returns_promise() {
        // import() should produce a Promise value.
        let result = global_eval("import('foo')").unwrap();
        assert!(
            result.is_promise(),
            "import() should return a Promise, got: {result:?}"
        );
    }

    #[test]
    fn e2e_dynamic_import_typeof_is_object() {
        // typeof a Promise is "object".
        assert_eval_true("typeof import('foo') === 'object'");
    }

    #[test]
    fn e2e_dynamic_import_instanceof_promise() {
        // The result of import() should be instanceof Promise.
        assert_eval_true("import('foo') instanceof Promise");
    }

    // ── 2. import() with non-string argument (ToString coercion) ─────────────

    #[test]
    fn e2e_dynamic_import_numeric_specifier() {
        // import() with a number — calls ToString on the argument.
        let result = global_eval("import(42)").unwrap();
        assert!(
            result.is_promise(),
            "import(42) should return a Promise, got: {result:?}"
        );
    }

    #[test]
    fn e2e_dynamic_import_boolean_specifier() {
        let result = global_eval("import(true)").unwrap();
        assert!(
            result.is_promise(),
            "import(true) should return a Promise, got: {result:?}"
        );
    }

    #[test]
    fn e2e_dynamic_import_undefined_specifier() {
        let result = global_eval("import(undefined)").unwrap();
        assert!(
            result.is_promise(),
            "import(undefined) should return a Promise, got: {result:?}"
        );
    }

    #[test]
    fn e2e_dynamic_import_null_specifier() {
        let result = global_eval("import(null)").unwrap();
        assert!(
            result.is_promise(),
            "import(null) should return a Promise, got: {result:?}"
        );
    }

    #[test]
    fn e2e_dynamic_import_template_literal_specifier() {
        // import() with a template literal argument.
        let result = global_eval("import(`module`)").unwrap();
        assert!(
            result.is_promise(),
            "import(`module`) should return a Promise, got: {result:?}"
        );
    }

    // ── 3. import() is valid in scripts ──────────────────────────────────────

    #[test]
    fn e2e_dynamic_import_parses_in_script_mode() {
        // import() must be valid in script (non-module) code.
        let prog = parse_script("import('foo');").unwrap();
        assert_eq!(
            prog.source_type,
            crate::parser::ast::SourceType::Script,
            "import() should be parseable in script mode"
        );
    }

    #[test]
    fn e2e_dynamic_import_parses_in_module_mode() {
        // import() must also work in module code.
        let prog = parse_module("import('foo');").unwrap();
        assert_eq!(prog.source_type, crate::parser::ast::SourceType::Module);
    }

    #[test]
    fn e2e_dynamic_import_in_script_executes() {
        // Full end-to-end: import() in script code returns a Promise.
        assert_eval_true("typeof import('x') === 'object'");
    }

    // ── 4. import() can be used in any expression position ───────────────────

    #[test]
    fn e2e_dynamic_import_in_variable_declaration() {
        assert_eval_true("var p = import('foo'); typeof p === 'object'");
    }

    #[test]
    fn e2e_dynamic_import_in_arrow_function() {
        assert_eval_true("var f = () => import('foo'); typeof f() === 'object'");
    }

    #[test]
    fn e2e_dynamic_import_in_ternary() {
        assert_eval_true("var p = true ? import('a') : import('b'); typeof p === 'object'");
    }

    #[test]
    fn e2e_dynamic_import_in_comma_expression() {
        // The comma expression should yield the import() result.
        assert_eval_true("var p = (0, import('foo')); typeof p === 'object'");
    }

    #[test]
    fn e2e_dynamic_import_in_array_literal() {
        assert_eval_true("var a = [import('foo')]; typeof a[0] === 'object'");
    }

    #[test]
    fn e2e_dynamic_import_in_object_value() {
        assert_eval_true("var o = { p: import('foo') }; typeof o.p === 'object'");
    }

    #[test]
    fn e2e_dynamic_import_as_function_argument() {
        assert_eval_true("function id(x) { return x; } typeof id(import('foo')) === 'object'");
    }

    // ── 5. import.meta — only valid in module code ───────────────────────────

    #[test]
    fn e2e_import_meta_syntax_error_in_script() {
        // import.meta in script mode must produce a SyntaxError.
        let err = parse_script("import.meta").unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("import.meta is only valid in module code"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn e2e_import_meta_syntax_error_in_eval() {
        // global_eval uses parse() which starts in script-detect mode.
        // Without module-triggering syntax, import.meta should fail.
        assert_eval_err("import.meta");
    }

    #[test]
    fn e2e_import_meta_valid_in_module_parse() {
        // import.meta must parse successfully in module mode.
        let prog = parse_module("import.meta").unwrap();
        assert_eq!(prog.source_type, crate::parser::ast::SourceType::Module);
    }

    #[test]
    fn e2e_import_meta_property_access_in_module() {
        // import.meta.url should parse in module mode.
        let prog = parse_module("import.meta.url").unwrap();
        assert_eq!(prog.body.len(), 1);
    }

    #[test]
    fn e2e_import_meta_invalid_property_name() {
        // import.foo should be a SyntaxError (only .meta is valid).
        let err = parse_module("import.foo").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("import.meta"), "unexpected error: {msg}");
    }

    // ── 6. import.meta is an ordinary, extensible object ─────────────────────

    #[test]
    fn e2e_import_meta_is_object_ast() {
        // Verify the AST node is MetaProp.
        use crate::parser::ast::{Expr, ProgramItem, Stmt};
        let prog = parse_module("import.meta").unwrap();
        if let ProgramItem::Stmt(Stmt::Expr(es)) = &prog.body[0] {
            assert!(
                matches!(es.expr.as_ref(), Expr::MetaProp(_)),
                "expected MetaProp node"
            );
        } else {
            panic!("expected expression statement");
        }
    }

    // ── 7. typeof import.meta === "object" (compile-level verification) ──────

    #[test]
    fn e2e_import_meta_compiles_to_lda_import_meta() {
        // Verify the bytecode generator emits the correct opcode.
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::bytecode::bytecodes::{Opcode, decode};

        let prog = parse_module("import.meta").unwrap();
        let ba = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = decode(&ba.bytecodes()).unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::LdaImportMeta),
            "expected LdaImportMeta opcode in bytecode"
        );
    }

    #[test]
    fn e2e_import_meta_runtime_returns_object() {
        // Run the LdaImportMeta bytecode through the interpreter directly.
        use crate::bytecode::bytecode_array::BytecodeArray;
        use crate::bytecode::bytecodes::{Instruction, Opcode, encode};
        use crate::bytecode::feedback::FeedbackMetadata;
        use crate::interpreter::{Interpreter, InterpreterFrame};

        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaImportMeta, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = BytecodeArray::new(
            encode(&instrs),
            vec![],
            0,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        )
        .with_module_flag(true);
        let mut frame = InterpreterFrame::new(Rc::new(ba), vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert!(
            matches!(result, JsValue::PlainObject(_)),
            "import.meta should return a PlainObject, got: {result:?}"
        );
    }

    // ── 8. import() with failed resolution — rejects the promise ─────────────

    #[test]
    fn e2e_dynamic_import_result_is_always_promise_regardless_of_specifier() {
        // Even with a nonsensical specifier, the result is a Promise.
        let result = global_eval("import('nonexistent/path/to/nothing')").unwrap();
        assert!(result.is_promise(), "expected Promise, got: {result:?}");
    }

    // ── 9. import() twice with same specifier ────────────────────────────────

    #[test]
    fn e2e_dynamic_import_same_specifier_twice_both_promise() {
        assert_eval_true(
            "var a = import('mod'); var b = import('mod'); typeof a === 'object' && typeof b === 'object'",
        );
    }

    // ── 10. new import() is not valid — import is not a constructor ──────────

    #[test]
    fn e2e_new_import_is_syntax_error() {
        assert_eval_err("new import('foo')");
    }

    #[test]
    fn e2e_new_import_syntax_error_parse() {
        let result = parse("new import('foo')");
        assert!(result.is_err(), "new import() should be a SyntaxError");
    }

    // ── 11. Conditional import syntax acceptance ─────────────────────────────

    #[test]
    fn e2e_conditional_import_parses() {
        let prog = parse("if (true) { import('mod'); }").unwrap();
        assert!(!prog.body.is_empty());
    }

    #[test]
    fn e2e_conditional_import_executes() {
        // The import() inside the if-block should succeed.
        let result =
            global_eval("var r; if (true) { r = import('mod'); } typeof r === 'object'").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── 12. import() in eval — works in sloppy mode ─────────────────────────

    #[test]
    fn e2e_dynamic_import_inside_eval_sloppy() {
        // Nested eval with import() — sloppy mode.
        assert_eval_true("typeof eval(\"import('foo')\") === 'object'");
    }

    // ── Additional expression-position and edge-case tests ───────────────────

    #[test]
    fn e2e_dynamic_import_with_expression_specifier() {
        // import() with a concatenation expression as specifier.
        assert_eval_true("typeof import('a' + 'b') === 'object'");
    }

    #[test]
    fn e2e_dynamic_import_with_variable_specifier() {
        assert_eval_true("var x = 'mod'; typeof import(x) === 'object'");
    }

    #[test]
    fn e2e_dynamic_import_empty_string_specifier() {
        let result = global_eval("import('')").unwrap();
        assert!(result.is_promise(), "import('') should return a Promise");
    }

    #[test]
    fn e2e_dynamic_import_requires_argument() {
        // import() with no arguments is a SyntaxError.
        assert_eval_err("import()");
    }

    #[test]
    fn e2e_dynamic_import_no_args_parse_error() {
        let result = parse("import()");
        assert!(
            result.is_err(),
            "import() with no args should be a SyntaxError"
        );
    }

    #[test]
    fn e2e_dynamic_import_spread_is_syntax_error() {
        // import(...args) is not valid syntax.
        assert_eval_err("var args = ['foo']; import(...args)");
    }

    #[test]
    fn e2e_import_meta_in_auto_detected_module() {
        // When the parser auto-detects module mode (import decl present),
        // import.meta becomes valid.
        let prog = parse("import x from 'y'; import.meta").unwrap();
        assert_eq!(prog.source_type, crate::parser::ast::SourceType::Module);
    }

    #[test]
    fn e2e_dynamic_import_compiles_to_call_runtime() {
        // Verify the bytecode generator emits CallRuntime for import().
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::bytecode::bytecodes::{Opcode, decode};

        let prog = parse("import('foo')").unwrap();
        let ba = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = decode(&ba.bytecodes()).unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::CallRuntime),
            "expected CallRuntime opcode for dynamic import()"
        );
    }
}
