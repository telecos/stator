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
        assert!(
            msg.contains("expected 'meta' after 'import.'"),
            "unexpected error: {msg}"
        );
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

    // ── Host-routed dynamic import and import.meta ─────────────────────────────

    use crate::host::{
        HostDynamicImportRequest, HostImportAttribute, HostImportMeta, HostModuleLoader, HostScope,
    };
    use std::cell::RefCell;
    use std::rc::Rc as StdRc;

    struct RecordingLoader {
        calls: RefCell<Vec<(String, Option<String>, Vec<HostImportAttribute>)>>,
        result: JsValue,
    }

    impl HostModuleLoader for RecordingLoader {
        fn dynamic_import(
            &self,
            request: HostDynamicImportRequest,
        ) -> Result<(), crate::builtins::error::JsError> {
            self.calls.borrow_mut().push((
                request.specifier().to_string(),
                request.referrer().map(str::to_string),
                request.attributes().to_vec(),
            ));
            request.resolve(self.result.clone());
            Ok(())
        }

        fn resolve(
            &self,
            specifier: &str,
            referrer: Option<&str>,
        ) -> Result<String, crate::builtins::error::JsError> {
            self.calls.borrow_mut().push((
                specifier.to_string(),
                referrer.map(str::to_string),
                Vec::new(),
            ));
            Ok(format!("resolved:{specifier}"))
        }
    }

    #[test]
    fn e2e_dynamic_import_routes_to_host_loader_and_fulfills_promise() {
        use crate::builtins::promise::PromiseState;

        let loader = StdRc::new(RecordingLoader {
            calls: RefCell::new(Vec::new()),
            result: JsValue::Smi(123),
        });
        let _scope = HostScope::install(
            Some(loader.clone() as StdRc<dyn HostModuleLoader>),
            Some("https://example/m.js"),
        );

        let result = global_eval("import('./dep.js')").unwrap();
        let JsValue::Promise(promise) = result else {
            panic!("expected promise");
        };
        assert!(matches!(
            promise.state(),
            PromiseState::Fulfilled(JsValue::Smi(123))
        ));
        let calls = loader.calls.borrow();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "./dep.js");
        assert_eq!(calls[0].1.as_deref(), Some("https://example/m.js"));
        assert!(calls[0].2.is_empty());
    }

    #[test]
    fn e2e_dynamic_import_can_settle_asynchronously_and_propagates_attributes() {
        use crate::builtins::promise::PromiseState;

        struct DeferredLoader {
            requests: RefCell<Vec<HostDynamicImportRequest>>,
        }

        impl HostModuleLoader for DeferredLoader {
            fn dynamic_import(
                &self,
                request: HostDynamicImportRequest,
            ) -> Result<(), crate::builtins::error::JsError> {
                self.requests.borrow_mut().push(request);
                Ok(())
            }

            fn resolve(
                &self,
                _specifier: &str,
                _referrer: Option<&str>,
            ) -> Result<String, crate::builtins::error::JsError> {
                unreachable!()
            }
        }

        let loader = StdRc::new(DeferredLoader {
            requests: RefCell::new(Vec::new()),
        });
        let _scope = HostScope::install(
            Some(loader.clone() as StdRc<dyn HostModuleLoader>),
            Some("https://example/referrer.js"),
        );

        let result = global_eval("import('./data.json', { with: { type: 'json' } })").unwrap();
        let JsValue::Promise(promise) = result else {
            panic!("expected promise");
        };
        assert!(matches!(promise.state(), PromiseState::Pending));
        let request = loader.requests.borrow()[0].clone();
        assert_eq!(request.specifier(), "./data.json");
        assert_eq!(request.referrer(), Some("https://example/referrer.js"));
        assert_eq!(
            request.attributes(),
            &[HostImportAttribute {
                key: "type".to_string(),
                value: "json".to_string(),
            }]
        );
        request.resolve(JsValue::Smi(7));
        assert!(matches!(
            promise.state(),
            PromiseState::Fulfilled(JsValue::Smi(7))
        ));
    }

    #[test]
    fn e2e_dynamic_import_host_rejection_yields_rejected_promise() {
        use crate::builtins::error::{ErrorKind, JsError};
        use crate::builtins::promise::PromiseState;

        struct RejectingLoader;
        impl HostModuleLoader for RejectingLoader {
            fn dynamic_import(&self, _request: HostDynamicImportRequest) -> Result<(), JsError> {
                Err(JsError::new(ErrorKind::TypeError, "nope".into()))
            }
            fn resolve(
                &self,
                _specifier: &str,
                _referrer: Option<&str>,
            ) -> Result<String, JsError> {
                unreachable!()
            }
        }

        let _scope = HostScope::install(
            Some(StdRc::new(RejectingLoader) as StdRc<dyn HostModuleLoader>),
            None,
        );
        let result = global_eval("import('./bad.js')").unwrap();
        let JsValue::Promise(promise) = result else {
            panic!("expected promise");
        };
        let PromiseState::Rejected(JsValue::Error(err)) = promise.state() else {
            panic!("expected rejected error promise");
        };
        assert_eq!(err.kind, ErrorKind::TypeError);
        assert_eq!(err.message, "nope");
    }

    #[test]
    fn e2e_dynamic_import_without_loader_still_rejects_with_typeerror() {
        use crate::builtins::error::ErrorKind;
        use crate::builtins::promise::PromiseState;

        // No HostScope installed: behaviour must match the historical
        // host-less rejection.
        let result = global_eval("import('foo')").unwrap();
        let JsValue::Promise(promise) = result else {
            panic!("expected promise");
        };
        let PromiseState::Rejected(JsValue::Error(err)) = promise.state() else {
            panic!("expected rejected error promise");
        };
        assert_eq!(err.kind, ErrorKind::TypeError);
        assert_eq!(err.message, "dynamic import is not supported by this host");
    }

    #[test]
    fn e2e_import_meta_url_reflects_published_module_url() {
        use crate::bytecode::bytecode_array::BytecodeArray;
        use crate::bytecode::bytecodes::{Instruction, Opcode, encode};
        use crate::bytecode::feedback::FeedbackMetadata;
        use crate::interpreter::{Interpreter, InterpreterFrame};
        use crate::objects::value::JsValue as Val;

        let _scope = HostScope::install(None, Some("https://host/page.js"));
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
        let Val::PlainObject(map) = result else {
            panic!("expected PlainObject")
        };
        let url = map.borrow().get("url").cloned();
        assert_eq!(url, Some(Val::String("https://host/page.js".into())));
    }

    #[test]
    fn e2e_import_meta_host_population_can_override_url_and_metadata() {
        struct MetaLoader;
        impl HostModuleLoader for MetaLoader {
            fn dynamic_import(
                &self,
                request: HostDynamicImportRequest,
            ) -> Result<(), crate::builtins::error::JsError> {
                request.resolve(JsValue::Undefined);
                Ok(())
            }

            fn resolve(
                &self,
                _specifier: &str,
                _referrer: Option<&str>,
            ) -> Result<String, crate::builtins::error::JsError> {
                Ok("unused".to_string())
            }

            fn populate_import_meta(
                &self,
                mut defaults: HostImportMeta,
            ) -> Result<HostImportMeta, crate::builtins::error::JsError> {
                defaults.url = "https://host/overridden.js".to_string();
                defaults.origin = Some("https://host/".to_string());
                defaults.referrer_policy = Some("strict-origin".to_string());
                Ok(defaults)
            }
        }

        use crate::bytecode::bytecode_array::BytecodeArray;
        use crate::bytecode::bytecodes::{Instruction, Opcode, encode};
        use crate::bytecode::feedback::FeedbackMetadata;
        use crate::interpreter::{Interpreter, InterpreterFrame};
        use crate::objects::value::JsValue as Val;

        let _scope = HostScope::install(
            Some(StdRc::new(MetaLoader) as StdRc<dyn HostModuleLoader>),
            Some("https://host/page.js"),
        );
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
        let Val::PlainObject(map) = result else {
            panic!("expected PlainObject")
        };
        let map = map.borrow();
        assert_eq!(
            map.get("url"),
            Some(&Val::String("https://host/overridden.js".into()))
        );
        assert_eq!(
            map.get("origin"),
            Some(&Val::String("https://host/".into()))
        );
        assert_eq!(
            map.get("referrerPolicy"),
            Some(&Val::String("strict-origin".into()))
        );
    }

    #[test]
    fn e2e_import_meta_resolve_routes_to_host_loader() {
        let loader = StdRc::new(RecordingLoader {
            calls: RefCell::new(Vec::new()),
            result: JsValue::Undefined,
        });
        let _scope = HostScope::install(
            Some(loader.clone() as StdRc<dyn HostModuleLoader>),
            Some("https://host/m.js"),
        );
        // import.meta is only valid in module mode; build a tiny module-flagged
        // bytecode: load import.meta, fetch the resolve method, invoke with a
        // string literal, return the result.
        use crate::bytecode::bytecode_array::{BytecodeArray, ConstantPoolEntry};
        use crate::bytecode::bytecodes::{Instruction, Opcode, Operand, encode};
        use crate::bytecode::feedback::FeedbackMetadata;
        use crate::interpreter::{Interpreter, InterpreterFrame};

        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaImportMeta, vec![]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::LdaNamedProperty,
                vec![
                    Operand::Register(0),
                    Operand::ConstantPoolIdx(0),
                    Operand::FeedbackSlot(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(1)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            Instruction::new_unchecked(
                Opcode::CallProperty1,
                vec![
                    Operand::Register(1),
                    Operand::Register(0),
                    Operand::Register(2),
                    Operand::FeedbackSlot(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = BytecodeArray::new(
            encode(&instrs),
            vec![
                ConstantPoolEntry::String("resolve".into()),
                ConstantPoolEntry::String("./dep".into()),
            ],
            3,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        )
        .with_module_flag(true);
        let mut frame = InterpreterFrame::new(Rc::new(ba), vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::String("resolved:./dep".into()));
        let calls = loader.calls.borrow();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "./dep");
        assert_eq!(calls[0].1.as_deref(), Some("https://host/m.js"));
    }

    #[test]
    fn e2e_import_meta_resolve_runtime_throws_without_host_loader() {
        use crate::bytecode::bytecode_array::{BytecodeArray, ConstantPoolEntry};
        use crate::bytecode::bytecodes::{Instruction, Opcode, Operand, encode};
        use crate::bytecode::feedback::FeedbackMetadata;
        use crate::interpreter::{Interpreter, InterpreterFrame};

        // Ensure no scope is active.
        assert!(crate::host::current_loader().is_none());

        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaImportMeta, vec![]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::LdaNamedProperty,
                vec![
                    Operand::Register(0),
                    Operand::ConstantPoolIdx(0),
                    Operand::FeedbackSlot(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(1)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            Instruction::new_unchecked(
                Opcode::CallProperty1,
                vec![
                    Operand::Register(1),
                    Operand::Register(0),
                    Operand::Register(2),
                    Operand::FeedbackSlot(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = BytecodeArray::new(
            encode(&instrs),
            vec![
                ConstantPoolEntry::String("resolve".into()),
                ConstantPoolEntry::String("./dep".into()),
            ],
            3,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        )
        .with_module_flag(true);
        let mut frame = InterpreterFrame::new(Rc::new(ba), vec![]);
        let err = Interpreter::run(&mut frame).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("import.meta.resolve is not supported by this host"),
            "unexpected error: {msg}"
        );
    }
}
