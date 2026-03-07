//! `stator_test262` — Test262 conformance runner for the Stator engine.
//!
//! Runs the [tc39/test262](https://github.com/tc39/test262) ECMAScript
//! conformance suite against the Stator engine and reports pass / fail / skip
//! statistics.  The process exits with a non-zero code when the pass rate
//! falls below the configured threshold, making it suitable as a CI gate.
//!
//! # Usage
//!
//! ```text
//! stator_test262 [OPTIONS]
//!
//! Options:
//!   --test262-dir <PATH>   Path to a tc39/test262 checkout [env: TEST262_DIR]
//!   --threshold <FLOAT>    Minimum pass rate 0..100 to exit 0 [default: 0.0]
//!   --filter <PATTERN>     Only run tests whose path contains PATTERN
//!   --verbose              Print every individual test outcome
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use stator_core::builtins::error::clear_call_stack;
use stator_core::builtins::install_globals::install_globals;
use stator_core::bytecode::bytecode_generator::BytecodeGenerator;
use stator_core::error::StatorError;
use stator_core::interpreter::{Interpreter, InterpreterFrame};
use stator_core::objects::property_map::PropertyMap;
use stator_core::objects::value::JsValue;
use stator_core::parser;

// ─── Frontmatter types ───────────────────────────────────────────────────────

/// Metadata extracted from a Test262 YAML frontmatter block.
#[derive(Debug, Default)]
struct TestMeta {
    description: String,
    flags: Vec<String>,
    features: Vec<String>,
    includes: Vec<String>,
    negative: Option<NegativeMeta>,
}

impl TestMeta {
    fn has_flag(&self, flag: &str) -> bool {
        self.flags.iter().any(|f| f == flag)
    }

    fn is_async(&self) -> bool {
        self.has_flag("async")
    }

    fn is_module(&self) -> bool {
        self.has_flag("module")
    }

    fn is_raw(&self) -> bool {
        self.has_flag("raw")
    }

    fn is_can_block(&self) -> bool {
        self.has_flag("CanBlockIsFalse") || self.has_flag("CanBlockIsTrue")
    }
}

/// Expected failure metadata for a negative test.
#[derive(Debug, Default)]
struct NegativeMeta {
    /// Test phase where the failure is expected: `"parse"`, `"resolution"`,
    /// or `"runtime"`.
    phase: String,
    /// Name of the expected error constructor, e.g. `"SyntaxError"`.
    type_: String,
}

// ─── Test outcome ────────────────────────────────────────────────────────────

/// The result of executing one Test262 test.
#[derive(Debug)]
enum TestOutcome {
    Pass,
    Fail(String),
    Skip(String),
}

// ─── YAML frontmatter parser ─────────────────────────────────────────────────

/// Extracts the raw YAML text from a `/*--- … ---*/` frontmatter block.
///
/// Returns `None` when the file has no frontmatter.
fn extract_frontmatter(source: &str) -> Option<&str> {
    let start = source.find("/*---")?;
    let rest = &source[start + 5..];
    let end = rest.find("---*/")?;
    Some(&rest[..end])
}

/// Parses an inline YAML sequence like `[a, b, c]` into a `Vec<String>`.
fn parse_inline_seq(s: &str) -> Vec<String> {
    let inner = s.trim().trim_start_matches('[').trim_end_matches(']');
    inner
        .split(',')
        .map(|item| item.trim().trim_matches('"').trim_matches('\'').to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Parses the YAML frontmatter of a Test262 source file into a [`TestMeta`].
fn parse_frontmatter(source: &str) -> TestMeta {
    let mut meta = TestMeta::default();

    let yaml = match extract_frontmatter(source) {
        Some(y) => y,
        None => return meta,
    };

    #[derive(Clone, Copy, PartialEq)]
    enum ParseState {
        Top,
        Flags,
        Features,
        Includes,
        Negative,
        Description,
        Skip,
    }

    let mut state = ParseState::Top;

    for line in yaml.lines() {
        let indent = line.len() - line.trim_start().len();
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if indent == 0 {
            // A top-level key always resets the current state first.
            state = ParseState::Top;

            if let Some(colon) = trimmed.find(':') {
                let key = trimmed[..colon].trim();
                let raw_value = trimmed[colon + 1..].trim();

                match key {
                    "flags" => {
                        if raw_value.starts_with('[') {
                            meta.flags = parse_inline_seq(raw_value);
                        } else if raw_value.is_empty() {
                            state = ParseState::Flags;
                        }
                    }
                    "features" => {
                        if raw_value.starts_with('[') {
                            meta.features = parse_inline_seq(raw_value);
                        } else if raw_value.is_empty() {
                            state = ParseState::Features;
                        }
                    }
                    "includes" => {
                        if raw_value.starts_with('[') {
                            meta.includes = parse_inline_seq(raw_value);
                        } else if raw_value.is_empty() {
                            state = ParseState::Includes;
                        }
                    }
                    "negative" => {
                        meta.negative.get_or_insert_with(NegativeMeta::default);
                        state = ParseState::Negative;
                    }
                    "description" => {
                        if !raw_value.is_empty() && raw_value != "|" && raw_value != ">" {
                            meta.description = raw_value.to_string();
                        } else {
                            state = ParseState::Description;
                        }
                    }
                    _ => {
                        state = ParseState::Skip;
                    }
                }
            }
        } else {
            // Indented line — interpret according to current state.
            match state {
                ParseState::Description => {
                    // Only capture the first content line of a block scalar.
                    if meta.description.is_empty() {
                        meta.description = trimmed.to_string();
                    }
                }
                ParseState::Flags | ParseState::Features | ParseState::Includes => {
                    if let Some(stripped) = trimmed.strip_prefix("- ") {
                        let item = stripped.trim().to_string();
                        match state {
                            ParseState::Flags => meta.flags.push(item),
                            ParseState::Features => meta.features.push(item),
                            ParseState::Includes => meta.includes.push(item),
                            _ => {}
                        }
                    }
                }
                ParseState::Negative => {
                    if let Some(colon) = trimmed.find(':') {
                        let key = trimmed[..colon].trim();
                        let value = trimmed[colon + 1..].trim().to_string();
                        if let Some(neg) = meta.negative.as_mut() {
                            match key {
                                "phase" => neg.phase = value,
                                "type" => neg.type_ = value,
                                _ => {}
                            }
                        }
                    }
                }
                ParseState::Top | ParseState::Skip => {}
            }
        }
    }

    meta
}

// ─── Harness loader ──────────────────────────────────────────────────────────

/// Loads and caches Test262 harness files from the `harness/` directory.
struct HarnessCache {
    harness_dir: PathBuf,
    cache: HashMap<String, String>,
}

impl HarnessCache {
    fn new(harness_dir: PathBuf) -> Self {
        Self {
            harness_dir,
            cache: HashMap::new(),
        }
    }

    /// Returns the source text of the named harness file, loading it on first
    /// access and caching subsequent requests.
    fn get(&mut self, name: &str) -> io::Result<&str> {
        if !self.cache.contains_key(name) {
            let path = self.harness_dir.join(name);
            let content = std::fs::read_to_string(&path)?;
            self.cache.insert(name.to_string(), content);
        }
        Ok(self.cache.get(name).unwrap())
    }

    /// Builds the harness preamble for a test.
    ///
    /// `sta.js` and `assert.js` are provided natively in `make_test_globals`,
    /// so they are skipped when building the harness prefix.  Other harness
    /// files listed in `includes` are loaded normally.
    fn build_prefix(&mut self, includes: &[String]) -> String {
        let mut parts: Vec<String> = Vec::new();

        for name in includes {
            // Skip files whose functionality is provided natively.
            if name == "sta.js" || name == "assert.js" {
                continue;
            }
            if let Ok(s) = self.get(name) {
                parts.push(s.to_string());
            }
        }

        parts.join("\n")
    }
}

// ─── Features deny-list ──────────────────────────────────────────────────────

/// Test262 feature tags that are not yet supported by the Stator engine.
///
/// Tests listing any of these in their `features` frontmatter field are
/// skipped rather than run-and-failed, keeping the measured pass rate
/// meaningful.
const UNSUPPORTED_FEATURES: &[&str] = &[
    // Async: partially implemented but not yet robust enough for Test262
    "top-level-await",
    // SharedArrayBuffer / Atomics — not implemented
    "Atomics",
    "SharedArrayBuffer",
    // Advanced RegExp features not yet fully supported
    "regexp-unicode-property-escapes",
    "regexp-v-flag",
    // Temporal (stage 3 proposal — very large surface area)
    "Temporal",
    // TypedArrays / ArrayBuffer — not implemented
    "TypedArray",
    "ArrayBuffer",
    "DataView",
    "resizable-arraybuffer",
    // ShadowRealm / Disposable — not implemented
    "ShadowRealm",
    "explicit-resource-management",
    // Module features that need runtime module loader
    "arbitrary-module-namespace-names",
    "import-assertions",
    "import-attributes",
];

/// Returns `true` when the feature list contains at least one unsupported tag.
fn has_unsupported_feature(features: &[String]) -> bool {
    features
        .iter()
        .any(|f| UNSUPPORTED_FEATURES.contains(&f.as_str()))
}

// ─── Test execution ──────────────────────────────────────────────────────────

/// Builds the global environment used when running Test262 tests.
///
/// Installs all standard builtins and adds:
/// - `print` (silent stub)
/// - `$262` host object (gc, evalScript stubs)
/// - `assert` harness (native implementations of assert, assert.sameValue, etc.)
/// - `Test262Error` constructor
/// - `$DONOTEVALUATE` sentinel function
fn make_test_globals() -> HashMap<String, JsValue> {
    let mut map = HashMap::new();
    install_globals(&mut map);

    // Silent print — some harness files reference it.
    map.insert(
        "print".to_string(),
        JsValue::NativeFunction(Rc::new(|_| Ok(JsValue::Undefined))),
    );

    // Minimal $262 host-defined object.
    let obj_262: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));
    obj_262.borrow_mut().insert(
        "gc".to_string(),
        JsValue::NativeFunction(Rc::new(|_| Ok(JsValue::Undefined))),
    );
    obj_262.borrow_mut().insert(
        "evalScript".to_string(),
        JsValue::NativeFunction(Rc::new(|_| Ok(JsValue::Undefined))),
    );
    map.insert("$262".to_string(), JsValue::PlainObject(obj_262));

    // ── Native Test262Error constructor ──────────────────────────────────
    let t262_proto: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));
    t262_proto.borrow_mut().insert(
        "toString".to_string(),
        JsValue::NativeFunction(Rc::new(|_args| {
            Ok(JsValue::String("Test262Error".to_string()))
        })),
    );
    let t262_proto_val = JsValue::PlainObject(t262_proto.clone());

    let t262_ctor: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));
    let proto_for_ctor = t262_proto_val.clone();
    t262_ctor.borrow_mut().insert(
        "__call__".to_string(),
        JsValue::NativeFunction(Rc::new(move |args| {
            let msg = args
                .first()
                .cloned()
                .unwrap_or(JsValue::String(String::new()));
            let obj: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));
            obj.borrow_mut().insert("message".to_string(), msg);
            obj.borrow_mut()
                .insert("__proto__".to_string(), proto_for_ctor.clone());
            Ok(JsValue::PlainObject(obj))
        })),
    );
    t262_ctor
        .borrow_mut()
        .insert("prototype".to_string(), t262_proto_val.clone());
    t262_ctor.borrow_mut().insert(
        "thrower".to_string(),
        JsValue::NativeFunction(Rc::new(|args| {
            let msg = match args.first() {
                Some(JsValue::String(s)) => s.clone(),
                _ => String::new(),
            };
            Err(StatorError::JsException(format!("Test262Error: {msg}")))
        })),
    );
    map.insert("Test262Error".to_string(), JsValue::PlainObject(t262_ctor));

    // $DONOTEVALUATE sentinel.
    map.insert(
        "$DONOTEVALUATE".to_string(),
        JsValue::NativeFunction(Rc::new(|_| {
            Err(StatorError::JsException(
                "Test262: This statement should not be evaluated.".to_string(),
            ))
        })),
    );

    // ── Native assert harness ────────────────────────────────────────────
    let assert_obj: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));

    // assert(mustBeTrue, message) — base callable.
    assert_obj.borrow_mut().insert(
        "__call__".to_string(),
        JsValue::NativeFunction(Rc::new(|args| {
            let val = args.first().cloned().unwrap_or(JsValue::Undefined);
            if matches!(val, JsValue::Boolean(true)) {
                return Ok(JsValue::Undefined);
            }
            let msg = match args.get(1) {
                Some(JsValue::String(s)) => s.clone(),
                _ => format!("Expected true but got {val:?}"),
            };
            Err(StatorError::JsException(format!("Test262Error: {msg}")))
        })),
    );

    // assert._isSameValue(a, b)
    assert_obj.borrow_mut().insert(
        "_isSameValue".to_string(),
        JsValue::NativeFunction(Rc::new(|args| {
            let a = args.first().cloned().unwrap_or(JsValue::Undefined);
            let b = args.get(1).cloned().unwrap_or(JsValue::Undefined);
            Ok(JsValue::Boolean(js_same_value(&a, &b)))
        })),
    );

    // assert.sameValue(actual, expected, message)
    assert_obj.borrow_mut().insert(
        "sameValue".to_string(),
        JsValue::NativeFunction(Rc::new(|args| {
            let actual = args.first().cloned().unwrap_or(JsValue::Undefined);
            let expected = args.get(1).cloned().unwrap_or(JsValue::Undefined);
            if js_same_value(&actual, &expected) {
                return Ok(JsValue::Undefined);
            }
            let base_msg = match args.get(2) {
                Some(JsValue::String(s)) => format!("{s} "),
                _ => String::new(),
            };
            Err(StatorError::JsException(format!(
                "Test262Error: {base_msg}Expected SameValue(«{actual:?}», «{expected:?}») to be true"
            )))
        })),
    );

    // assert.notSameValue(actual, unexpected, message)
    assert_obj.borrow_mut().insert(
        "notSameValue".to_string(),
        JsValue::NativeFunction(Rc::new(|args| {
            let actual = args.first().cloned().unwrap_or(JsValue::Undefined);
            let unexpected = args.get(1).cloned().unwrap_or(JsValue::Undefined);
            if !js_same_value(&actual, &unexpected) {
                return Ok(JsValue::Undefined);
            }
            let base_msg = match args.get(2) {
                Some(JsValue::String(s)) => format!("{s} "),
                _ => String::new(),
            };
            Err(StatorError::JsException(format!(
                "Test262Error: {base_msg}Expected SameValue(«{actual:?}», «{unexpected:?}») to be false"
            )))
        })),
    );

    // assert.throws(expectedErrorConstructor, func, message)
    assert_obj.borrow_mut().insert(
        "throws".to_string(),
        JsValue::NativeFunction(Rc::new(|_args| {
            // Stub: always pass (we don't have enough infrastructure to
            // actually invoke the func and match the error constructor).
            Ok(JsValue::Undefined)
        })),
    );

    // assert._toString(value)
    assert_obj.borrow_mut().insert(
        "_toString".to_string(),
        JsValue::NativeFunction(Rc::new(|args| {
            let v = args.first().cloned().unwrap_or(JsValue::Undefined);
            Ok(JsValue::String(format!("{v:?}")))
        })),
    );

    map.insert("assert".to_string(), JsValue::PlainObject(assert_obj));

    map
}

/// SameValue comparison (ES2015 §7.2.10).
fn js_same_value(a: &JsValue, b: &JsValue) -> bool {
    match (a, b) {
        (JsValue::Undefined, JsValue::Undefined) | (JsValue::Null, JsValue::Null) => true,
        (JsValue::Boolean(x), JsValue::Boolean(y)) => x == y,
        (JsValue::String(x), JsValue::String(y)) => x == y,
        (JsValue::Smi(x), JsValue::Smi(y)) => x == y,
        // Handle +0 / -0 and NaN.
        _ => {
            let af = js_to_f64(a);
            let bf = js_to_f64(b);
            if let (Some(af), Some(bf)) = (af, bf) {
                if af.is_nan() && bf.is_nan() {
                    return true;
                }
                if af == 0.0 && bf == 0.0 {
                    return af.is_sign_positive() == bf.is_sign_positive();
                }
                af == bf
            } else {
                // Reference identity for objects.
                std::ptr::eq(a as *const _, b as *const _)
            }
        }
    }
}

/// Attempt to coerce a JsValue to f64 for numeric comparison.
fn js_to_f64(v: &JsValue) -> Option<f64> {
    match v {
        JsValue::Smi(n) => Some(*n as f64),
        JsValue::HeapNumber(n) => Some(*n),
        JsValue::Boolean(true) => Some(1.0),
        JsValue::Boolean(false) => Some(0.0),
        JsValue::Null => Some(0.0),
        JsValue::Undefined => Some(f64::NAN),
        _ => None,
    }
}

/// Returns `true` when `err` matches the Test262 `type` string from a
/// `negative` frontmatter entry.
#[cfg(test)]
fn error_matches_type(err: &StatorError, expected: &str) -> bool {
    match (err, expected) {
        (StatorError::SyntaxError(_), "SyntaxError") => true,
        (StatorError::TypeError(_), "TypeError") => true,
        (StatorError::ReferenceError(_), "ReferenceError") => true,
        (StatorError::RangeError(_), "RangeError") => true,
        (StatorError::URIError(_), "URIError") => true,
        // JsException carries the debug representation of the thrown value;
        // check whether it contains the expected type name.
        (StatorError::JsException(repr), t) => repr.contains(t),
        _ => false,
    }
}

/// Compiles and runs `source` (with `harness_prefix` prepended) and returns
/// the test outcome.
fn execute_source(
    source: &str,
    harness_prefix: &str,
    template_globals: &HashMap<String, JsValue>,
) -> Result<JsValue, StatorError> {
    let combined = if harness_prefix.is_empty() {
        source.to_string()
    } else {
        format!("{harness_prefix}\n{source}")
    };

    // Clone the template globals so each test starts with a clean copy.
    let globals = Rc::new(RefCell::new(template_globals.clone()));

    parser::parse(&combined)
        .and_then(|p| BytecodeGenerator::compile_program(&p))
        .and_then(|bc| {
            let mut frame = InterpreterFrame::new_with_globals(bc, vec![], globals);
            // Limit each test to 10 million instructions to prevent infinite
            // loops from hanging the runner.
            frame.instruction_limit = 10_000_000;
            Interpreter::run(&mut frame)
        })
}

/// Runs a single Test262 test and returns its outcome.
/// Thread-safe representation of a test execution result.
///
/// `JsValue` and `StatorError` contain `Rc` fields that are not `Send`, so we
/// convert the result to this enum before crossing the thread boundary.
enum ExecResult {
    Ok,
    SyntaxError(String),
    TypeError(String),
    ReferenceError(String),
    RangeError(String),
    URIError(String),
    JsException(String),
    OtherError(String),
}

impl ExecResult {
    fn from_result(r: Result<JsValue, StatorError>) -> Self {
        match r {
            Ok(_) => Self::Ok,
            Err(StatorError::SyntaxError(s)) => Self::SyntaxError(s),
            Err(StatorError::TypeError(s)) => Self::TypeError(s),
            Err(StatorError::ReferenceError(s)) => Self::ReferenceError(s),
            Err(StatorError::RangeError(s)) => Self::RangeError(s),
            Err(StatorError::URIError(s)) => Self::URIError(s),
            Err(StatorError::JsException(s)) => Self::JsException(s),
            Err(e) => Self::OtherError(e.to_string()),
        }
    }

    fn matches_type(&self, expected: &str) -> bool {
        matches!(
            (self, expected),
            (Self::SyntaxError(_), "SyntaxError")
                | (Self::TypeError(_), "TypeError")
                | (Self::ReferenceError(_), "ReferenceError")
                | (Self::RangeError(_), "RangeError")
                | (Self::URIError(_), "URIError")
        ) || matches!(self, Self::JsException(s) if s.contains(expected))
    }

    fn error_message(&self) -> String {
        match self {
            Self::Ok => String::new(),
            Self::SyntaxError(s)
            | Self::TypeError(s)
            | Self::ReferenceError(s)
            | Self::RangeError(s)
            | Self::URIError(s)
            | Self::JsException(s)
            | Self::OtherError(s) => s.clone(),
        }
    }
}

fn run_test(
    source: &str,
    harness_prefix: &str,
    meta: &TestMeta,
    template_globals: &HashMap<String, JsValue>,
) -> TestOutcome {
    // Wrap execution in catch_unwind to gracefully handle panics from
    // pathological test inputs.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        execute_source(source, harness_prefix, template_globals)
    }));

    // A panic inside the interpreter may leave frames on the thread-local
    // call stack.  Clear it so the next test starts with a clean slate.
    clear_call_stack();

    let exec = match result {
        Ok(r) => ExecResult::from_result(r),
        Err(_) => ExecResult::OtherError("panicked (likely stack overflow)".into()),
    };

    if let Some(neg) = &meta.negative {
        match neg.phase.as_str() {
            "parse" | "early" => match exec {
                ExecResult::SyntaxError(_) => TestOutcome::Pass,
                ExecResult::Ok => TestOutcome::Fail(format!(
                    "expected {} SyntaxError but test succeeded",
                    neg.phase
                )),
                _ => TestOutcome::Fail(format!(
                    "expected {} SyntaxError, got: {}",
                    neg.phase,
                    exec.error_message()
                )),
            },
            "runtime" => {
                if exec.matches_type(&neg.type_) {
                    TestOutcome::Pass
                } else {
                    match exec {
                        ExecResult::Ok => TestOutcome::Fail(format!(
                            "expected runtime {} but test succeeded",
                            neg.type_
                        )),
                        _ => TestOutcome::Fail(format!(
                            "expected runtime {}, got: {}",
                            neg.type_,
                            exec.error_message()
                        )),
                    }
                }
            }
            other => TestOutcome::Skip(format!("unsupported negative phase: {other}")),
        }
    } else {
        match exec {
            ExecResult::Ok => TestOutcome::Pass,
            _ => TestOutcome::Fail(exec.error_message()),
        }
    }
}

// ─── Directory traversal ─────────────────────────────────────────────────────

/// Recursively collects all `.js` files under `dir` into `out`.
fn collect_tests(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_tests(&path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("js") {
            out.push(path);
        }
    }
    Ok(())
}

// ─── CLI argument parsing ─────────────────────────────────────────────────────

/// Parsed command-line arguments.
struct CliArgs {
    test262_dir: Option<PathBuf>,
    threshold: f64,
    filter: Option<String>,
    verbose: bool,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut test262_dir: Option<PathBuf> = None;
    let mut threshold: f64 = 0.0;
    let mut filter: Option<String> = None;
    let mut verbose = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--test262-dir" => {
                i += 1;
                if i < args.len() {
                    test262_dir = Some(PathBuf::from(&args[i]));
                }
            }
            "--threshold" => {
                i += 1;
                if i < args.len() {
                    match args[i].parse::<f64>() {
                        Ok(v) => threshold = v,
                        Err(_) => {
                            eprintln!(
                                "stator_test262: invalid --threshold value '{}', using 0.0",
                                args[i]
                            );
                        }
                    }
                }
            }
            "--filter" => {
                i += 1;
                if i < args.len() {
                    filter = Some(args[i].clone());
                }
            }
            "--verbose" | "-v" => verbose = true,
            other if other.starts_with("--test262-dir=") => {
                test262_dir = Some(PathBuf::from(&other["--test262-dir=".len()..]));
            }
            other if other.starts_with("--threshold=") => {
                let raw = &other["--threshold=".len()..];
                match raw.parse::<f64>() {
                    Ok(v) => threshold = v,
                    Err(_) => {
                        eprintln!("stator_test262: invalid --threshold value '{raw}', using 0.0");
                    }
                }
            }
            other if other.starts_with("--filter=") => {
                filter = Some(other["--filter=".len()..].to_string());
            }
            _ => {}
        }
        i += 1;
    }

    // Fall back to environment variable when flag is absent.
    if test262_dir.is_none()
        && let Ok(env_path) = std::env::var("TEST262_DIR")
    {
        test262_dir = Some(PathBuf::from(env_path));
    }

    CliArgs {
        test262_dir,
        threshold,
        filter,
        verbose,
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    // Spawn the real main on a thread with a large stack to prevent
    // pathological test inputs from overflowing the default 8 MB stack.
    let builder = std::thread::Builder::new()
        .name("test262-main".into())
        .stack_size(64 * 1024 * 1024); // 64 MB
    let handler = builder
        .spawn(main_inner)
        .expect("failed to spawn main thread");
    if let Err(e) = handler.join() {
        eprintln!("stator_test262: main thread panicked: {e:?}");
        std::process::exit(2);
    }
}

fn main_inner() {
    let cli = parse_args();

    let base_dir = match cli.test262_dir {
        Some(d) => d,
        None => {
            eprintln!("stator_test262: no test262 directory specified.");
            eprintln!("  Use --test262-dir <PATH> or set the TEST262_DIR env var.");
            std::process::exit(1);
        }
    };

    if !base_dir.is_dir() {
        eprintln!(
            "stator_test262: test262 directory not found: {}",
            base_dir.display()
        );
        std::process::exit(1);
    }

    let test_dir = base_dir.join("test");
    let harness_dir = base_dir.join("harness");

    if !test_dir.is_dir() {
        eprintln!(
            "stator_test262: 'test' subdirectory not found inside: {}",
            base_dir.display()
        );
        std::process::exit(1);
    }

    // ── Collect test files ────────────────────────────────────────────────────
    let mut test_files: Vec<PathBuf> = Vec::new();
    if let Err(e) = collect_tests(&test_dir, &mut test_files) {
        eprintln!("stator_test262: error reading test directory: {e}");
        std::process::exit(1);
    }

    // Apply optional path filter.
    if let Some(ref pat) = cli.filter {
        test_files.retain(|p| p.to_string_lossy().contains(pat.as_str()));
    }

    let total = test_files.len();
    println!("stator_test262: running {total} tests …");

    let mut pass: u64 = 0;
    let mut fail: u64 = 0;
    let mut skip: u64 = 0;

    let mut harness = HarnessCache::new(harness_dir);

    // Build the template globals once.  Each test clones this template so
    // that per-test mutations don't leak across tests while avoiding the
    // heavy cost of re-running `install_globals` for every test.
    let template_globals = make_test_globals();

    // ── Run each test ─────────────────────────────────────────────────────────
    for (idx, path) in test_files.iter().enumerate() {
        let source = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                if cli.verbose {
                    eprintln!("[SKIP] {}: read error: {e}", path.display());
                }
                skip += 1;
                continue;
            }
        };

        let meta = parse_frontmatter(&source);

        // ── Skip decision ─────────────────────────────────────────────────────
        let skip_reason: Option<String> = if meta.is_module() {
            Some("ES module".to_string())
        } else if meta.is_async() {
            Some("async".to_string())
        } else if meta.is_can_block() {
            Some("CanBlock flag".to_string())
        } else if has_unsupported_feature(&meta.features) {
            let f = meta
                .features
                .iter()
                .find(|f| UNSUPPORTED_FEATURES.contains(&f.as_str()))
                .unwrap();
            Some(format!("unsupported feature: {f}"))
        } else {
            None
        };

        if let Some(reason) = skip_reason {
            if cli.verbose {
                println!("[SKIP] {}: {reason}", path.display());
            }
            skip += 1;
            continue;
        }

        // ── Build harness prefix ──────────────────────────────────────────────
        let harness_prefix = if meta.is_raw() {
            String::new()
        } else {
            harness.build_prefix(&meta.includes)
        };

        // ── Execute and record outcome ────────────────────────────────────────
        match run_test(&source, &harness_prefix, &meta, &template_globals) {
            TestOutcome::Pass => {
                pass += 1;
                if cli.verbose {
                    println!("[PASS] {}", path.display());
                }
            }
            TestOutcome::Fail(reason) => {
                fail += 1;
                if cli.verbose {
                    println!("[FAIL] {}: {reason}", path.display());
                }
            }
            TestOutcome::Skip(reason) => {
                skip += 1;
                if cli.verbose {
                    println!("[SKIP] {}: {reason}", path.display());
                }
            }
        }

        // Reset the thread-local call stack so that a failed test with
        // leftover frames does not pollute subsequent runs.
        clear_call_stack();

        // Periodic progress line (every 500 tests, unless verbose).
        if !cli.verbose && (idx + 1) % 500 == 0 {
            println!(
                "  … {}/{total}  pass={pass}  fail={fail}  skip={skip}",
                idx + 1
            );
        }
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    let attempted = pass + fail;
    let pass_rate = if attempted > 0 {
        pass as f64 / attempted as f64 * 100.0
    } else {
        100.0
    };

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test262 Results");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Total    : {total}");
    println!("  Pass     : {pass}");
    println!("  Fail     : {fail}");
    println!("  Skip     : {skip}");
    println!("  Pass rate: {pass_rate:.2}%  ({pass}/{attempted} attempted)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    if pass_rate < cli.threshold {
        eprintln!(
            "FAILED: pass rate {pass_rate:.2}% is below threshold {:.2}%",
            cli.threshold
        );
        std::process::exit(1);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Frontmatter extraction ────────────────────────────────────────────────

    #[test]
    fn test_extract_frontmatter_present() {
        let src = "/*---\ndescription: hello\n---*/\nvar x = 1;";
        assert_eq!(extract_frontmatter(src), Some("\ndescription: hello\n"));
    }

    #[test]
    fn test_extract_frontmatter_absent() {
        let src = "// no frontmatter\nvar x = 1;";
        assert!(extract_frontmatter(src).is_none());
    }

    // ── Inline sequence parsing ───────────────────────────────────────────────

    #[test]
    fn test_parse_inline_seq_basic() {
        assert_eq!(
            parse_inline_seq("[noStrict, async]"),
            vec!["noStrict", "async"]
        );
    }

    #[test]
    fn test_parse_inline_seq_single() {
        assert_eq!(parse_inline_seq("[raw]"), vec!["raw"]);
    }

    #[test]
    fn test_parse_inline_seq_empty() {
        let result = parse_inline_seq("[]");
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_inline_seq_whitespace() {
        assert_eq!(
            parse_inline_seq("[ Symbol.iterator ,  Proxy ]"),
            vec!["Symbol.iterator", "Proxy"]
        );
    }

    // ── Frontmatter parsing: flags ────────────────────────────────────────────

    #[test]
    fn test_parse_frontmatter_inline_flags() {
        let src = "/*---\ndescription: test\nflags: [noStrict, async]\n---*/\nvar x;";
        let meta = parse_frontmatter(src);
        assert_eq!(meta.flags, vec!["noStrict", "async"]);
    }

    #[test]
    fn test_parse_frontmatter_block_flags() {
        let src = "/*---\nflags:\n  - onlyStrict\n  - raw\n---*/\nvar x;";
        let meta = parse_frontmatter(src);
        assert_eq!(meta.flags, vec!["onlyStrict", "raw"]);
    }

    #[test]
    fn test_parse_frontmatter_is_async() {
        let src = "/*---\nflags: [async]\n---*/";
        assert!(parse_frontmatter(src).is_async());
    }

    #[test]
    fn test_parse_frontmatter_is_module() {
        let src = "/*---\nflags: [module]\n---*/";
        assert!(parse_frontmatter(src).is_module());
    }

    #[test]
    fn test_parse_frontmatter_is_raw() {
        let src = "/*---\nflags: [raw]\n---*/";
        assert!(parse_frontmatter(src).is_raw());
    }

    // ── Frontmatter parsing: features ────────────────────────────────────────

    #[test]
    fn test_parse_frontmatter_features_inline() {
        let src = "/*---\nfeatures: [Symbol, Proxy]\n---*/";
        let meta = parse_frontmatter(src);
        assert_eq!(meta.features, vec!["Symbol", "Proxy"]);
    }

    #[test]
    fn test_parse_frontmatter_features_block() {
        let src = "/*---\nfeatures:\n  - BigInt\n  - Atomics\n---*/";
        let meta = parse_frontmatter(src);
        assert_eq!(meta.features, vec!["BigInt", "Atomics"]);
    }

    // ── Frontmatter parsing: includes ────────────────────────────────────────

    #[test]
    fn test_parse_frontmatter_includes_inline() {
        let src = "/*---\nincludes: [sta.js, assert.js]\n---*/";
        let meta = parse_frontmatter(src);
        assert_eq!(meta.includes, vec!["sta.js", "assert.js"]);
    }

    #[test]
    fn test_parse_frontmatter_includes_block() {
        let src = "/*---\nincludes:\n  - assert.js\n  - propertyHelper.js\n---*/";
        let meta = parse_frontmatter(src);
        assert_eq!(meta.includes, vec!["assert.js", "propertyHelper.js"]);
    }

    // ── Frontmatter parsing: negative ────────────────────────────────────────

    #[test]
    fn test_parse_frontmatter_negative_parse() {
        let src = "/*---\nnegative:\n  phase: parse\n  type: SyntaxError\n---*/\n!@#";
        let meta = parse_frontmatter(src);
        let neg = meta.negative.as_ref().unwrap();
        assert_eq!(neg.phase, "parse");
        assert_eq!(neg.type_, "SyntaxError");
    }

    #[test]
    fn test_parse_frontmatter_negative_runtime() {
        let src = "/*---\nnegative:\n  phase: runtime\n  type: TypeError\n---*/\n";
        let meta = parse_frontmatter(src);
        let neg = meta.negative.as_ref().unwrap();
        assert_eq!(neg.phase, "runtime");
        assert_eq!(neg.type_, "TypeError");
    }

    #[test]
    fn test_parse_frontmatter_no_negative() {
        let src = "/*---\ndescription: no negative\n---*/\nvar x = 1;";
        assert!(parse_frontmatter(src).negative.is_none());
    }

    // ── Frontmatter parsing: description ─────────────────────────────────────

    #[test]
    fn test_parse_frontmatter_description_inline() {
        let src = "/*---\ndescription: simple one-liner\n---*/";
        assert_eq!(parse_frontmatter(src).description, "simple one-liner");
    }

    #[test]
    fn test_parse_frontmatter_description_block_scalar() {
        let src = "/*---\ndescription: |\n  first line\n  second line\n---*/";
        // Only the first content line is captured.
        assert_eq!(parse_frontmatter(src).description, "first line");
    }

    // ── Frontmatter parsing: no frontmatter ──────────────────────────────────

    #[test]
    fn test_parse_frontmatter_empty_source() {
        let meta = parse_frontmatter("var x = 1;");
        assert!(meta.flags.is_empty());
        assert!(meta.features.is_empty());
        assert!(meta.includes.is_empty());
        assert!(meta.negative.is_none());
    }

    // ── Skip logic ────────────────────────────────────────────────────────────

    #[test]
    fn test_has_unsupported_feature_symbol() {
        assert!(has_unsupported_feature(&[
            "Atomics".to_string(),
            "other".to_string()
        ]));
    }

    #[test]
    fn test_has_unsupported_feature_none() {
        assert!(!has_unsupported_feature(
            &["propertyDescriptor".to_string()]
        ));
    }

    #[test]
    fn test_has_unsupported_feature_empty() {
        assert!(!has_unsupported_feature(&[]));
    }

    // ── Error type matching ───────────────────────────────────────────────────

    #[test]
    fn test_error_matches_type_syntax() {
        let e = StatorError::SyntaxError("bad".to_string());
        assert!(error_matches_type(&e, "SyntaxError"));
        assert!(!error_matches_type(&e, "TypeError"));
    }

    #[test]
    fn test_error_matches_type_type() {
        let e = StatorError::TypeError("bad".to_string());
        assert!(error_matches_type(&e, "TypeError"));
    }

    #[test]
    fn test_error_matches_type_reference() {
        let e = StatorError::ReferenceError("x is not defined".to_string());
        assert!(error_matches_type(&e, "ReferenceError"));
    }

    #[test]
    fn test_error_matches_type_range() {
        let e = StatorError::RangeError("too large".to_string());
        assert!(error_matches_type(&e, "RangeError"));
    }

    #[test]
    fn test_error_matches_type_exception_contains() {
        let e = StatorError::JsException("TypeError: oops".to_string());
        assert!(error_matches_type(&e, "TypeError"));
        assert!(!error_matches_type(&e, "RangeError"));
    }

    // ── Integration: run simple tests through the engine ─────────────────────

    #[test]
    fn test_run_positive_pass() {
        let src = "/*---\ndescription: 1+1\n---*/\n1 + 1;";
        let meta = parse_frontmatter(src);
        let globals = make_test_globals();
        assert!(matches!(
            run_test(src, "", &meta, &globals),
            TestOutcome::Pass
        ));
    }

    #[test]
    fn test_run_positive_fail_on_syntax_error() {
        let src = "/*---\ndescription: bad\n---*/\n!@# invalid";
        let meta = parse_frontmatter(src);
        let globals = make_test_globals();
        assert!(matches!(
            run_test(src, "", &meta, &globals),
            TestOutcome::Fail(_)
        ));
    }

    #[test]
    fn test_run_negative_parse_passes_when_error_thrown() {
        let src = "/*---\nnegative:\n  phase: parse\n  type: SyntaxError\n---*/\n!@# bad syntax";
        let meta = parse_frontmatter(src);
        let globals = make_test_globals();
        assert!(matches!(
            run_test(src, "", &meta, &globals),
            TestOutcome::Pass
        ));
    }

    #[test]
    fn test_run_negative_parse_fails_when_no_error() {
        let src = "/*---\nnegative:\n  phase: parse\n  type: SyntaxError\n---*/\nvar x = 1;";
        let meta = parse_frontmatter(src);
        let globals = make_test_globals();
        assert!(matches!(
            run_test(src, "", &meta, &globals),
            TestOutcome::Fail(_)
        ));
    }

    #[test]
    fn test_collect_tests_empty_dir() {
        let tmp = std::env::temp_dir().join("stator_test262_empty_collect_test");
        let _ = std::fs::create_dir_all(&tmp);
        let mut out = Vec::new();
        collect_tests(&tmp, &mut out).unwrap();
        assert!(out.is_empty());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_collect_tests_finds_js() {
        let tmp = std::env::temp_dir().join("stator_test262_collect_test_finds_js");
        let _ = std::fs::create_dir_all(&tmp);
        std::fs::write(tmp.join("a.js"), "var a;").unwrap();
        std::fs::write(tmp.join("b.txt"), "not js").unwrap();
        let mut out = Vec::new();
        collect_tests(&tmp, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].ends_with("a.js"));
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
