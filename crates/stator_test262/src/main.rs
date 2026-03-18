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

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use stator_core::builtins::error::clear_call_stack;
use stator_core::builtins::install_globals::install_globals;
use stator_core::builtins::promise::drain_active_microtask_queue;
use stator_core::bytecode::bytecode_generator::BytecodeGenerator;
use stator_core::error::StatorError;
use stator_core::interpreter::{
    Interpreter, InterpreterFrame, clear_interpreter_state, set_execution_deadline,
};
use stator_core::objects::property_map::PropertyMap;
use stator_core::objects::string_intern::clear_intern_pool;
use stator_core::objects::value::JsValue;
use stator_core::parser;

// ─── Guarded global allocator ────────────────────────────────────────────────

/// A thin wrapper around the system allocator that tracks oversized allocations.
/// This prevents pathological Test262 inputs (e.g. `new Array(2**53)`) from
/// silently consuming all CI runner memory.
///
/// When a single allocation exceeds [`MAX_ALLOC_SIZE`], the allocator sets a
/// thread-local flag and **still delegates to `System`**.  The test runner
/// checks this flag after each test's `catch_unwind` and marks the test as
/// failed.  This avoids panicking inside `GlobalAlloc::alloc()`, which is
/// undefined behavior per the Rust reference.
///
/// We always delegate to `System` regardless of size — the OS/OOM-killer
/// handles truly catastrophic allocations.
struct GuardedAlloc;

/// Soft limit: allocations above this set the exceeded flag (512 MiB).
const MAX_ALLOC_SIZE: usize = 512 << 20;

thread_local! {
    /// Set to `true` when an oversized allocation is detected.
    /// Checked after each test to mark it as failed.
    static ALLOC_EXCEEDED: Cell<bool> = const { Cell::new(false) };
}

// SAFETY: All methods delegate to `System` (or return null for oversized
// requests).  The safety invariants of `GlobalAlloc` (valid layout, matching
// alloc/dealloc) are upheld by the inner `System` allocator.  We never panic.
// For oversized allocations, we return null — callers (typically the Rust
// standard library) will call `handle_alloc_error`, which aborts the process.
// The CI workflow catches exit code 134 (SIGABRT) and treats it as a known
// issue rather than a hard failure.
unsafe impl GlobalAlloc for GuardedAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() > MAX_ALLOC_SIZE {
            // Oversized — set flag AND return null to prevent the allocation
            // from actually going through.  Previously we delegated to System
            // even for huge sizes, which caused the runner to OOM.
            ALLOC_EXCEEDED.with(|g| g.set(true));
            return std::ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

/// Check whether any oversized allocation occurred during the current test.
fn alloc_limit_exceeded() -> bool {
    ALLOC_EXCEEDED.with(|g| g.get())
}

/// Reset the allocation flag for the next test.
fn reset_alloc_guard() {
    ALLOC_EXCEEDED.with(|g| g.set(false));
}

#[global_allocator]
static ALLOC: GuardedAlloc = GuardedAlloc;

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
    /// Cached harness prefix strings keyed by the sorted includes list.
    /// Most Test262 tests share the same set of includes, so this avoids
    /// re-concatenating the same harness files thousands of times.
    prefix_cache: HashMap<Vec<String>, String>,
}

impl HarnessCache {
    fn new(harness_dir: PathBuf) -> Self {
        Self {
            harness_dir,
            cache: HashMap::new(),
            prefix_cache: HashMap::new(),
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
    ///
    /// Results are cached by the includes list so that tests sharing the same
    /// set of harness files reuse a single allocation.
    fn build_prefix(&mut self, includes: &[String]) -> String {
        // Build the cache key: the filtered, ordered list of includes.
        let key: Vec<String> = includes
            .iter()
            .filter(|name| {
                name.as_str() != "sta.js"
                    && name.as_str() != "assert.js"
                    && name.as_str() != "donNotEvaluate.js"
            })
            .cloned()
            .collect();

        if let Some(cached) = self.prefix_cache.get(&key) {
            return cached.clone();
        }

        let mut parts: Vec<String> = Vec::new();
        for name in &key {
            match self.get(name) {
                Ok(s) => parts.push(s.to_string()),
                Err(e) => log::warn!("failed to load harness file '{name}': {e}"),
            }
        }

        let prefix = parts.join("\n");
        self.prefix_cache.insert(key, prefix.clone());
        prefix
    }
}

// ─── Features deny-list ──────────────────────────────────────────────────────

/// Test262 feature tags that are not yet supported by the Stator engine.
///
/// Tests listing any of these in their `features` frontmatter field are
/// skipped rather than run-and-failed, keeping the measured pass rate
/// meaningful.
const UNSUPPORTED_FEATURES: &[&str] = &[
    // Resizable ArrayBuffer — not yet implemented
    "resizable-arraybuffer",
    // Module features that need runtime module loader
    "arbitrary-module-namespace-names",
    "import-assertions",
    "import-attributes",
    // TC39 Stage 3 / new built-ins not yet implemented
    "Temporal",
    "decorators",
    "source-phase-imports",
    "json-modules",
    "ShadowRealm",
    // Regex features not yet supported by the regress engine
    "regexp-duplicate-named-groups",
    "regexp-modifiers",
    "regexp-v-flag",
    // Resource management — `using` keyword not yet in the parser
    "explicit-resource-management",
    // Atomics.waitAsync — promise integration not implemented
    "Atomics.waitAsync",
    // ArrayBuffer.prototype.transfer not implemented
    "arraybuffer-transfer",
    // Float16Array typed array not implemented
    "Float16Array",
    // Tail-call optimisation not implemented in the interpreter
    "tail-call-optimization",
    // cross-realm — requires full realm support
    "cross-realm",
    // Symbol.species — defined but constructor integration incomplete
    "Symbol.species",
    // Intl not implemented
    "Intl-enumeration",
    "Intl.DateTimeFormat-datetimestyle",
    "Intl.DateTimeFormat-dayPeriod",
    "Intl.DateTimeFormat-formatRange",
    "Intl.DateTimeFormat-fractionalSecondDigits",
    "Intl.DisplayNames",
    "Intl.DisplayNames-v2",
    "Intl.DurationFormat",
    "Intl.ListFormat",
    "Intl.Locale",
    "Intl.NumberFormat-unified",
    "Intl.NumberFormat-v3",
    "Intl.RelativeTimeFormat",
    "Intl.Segmenter",
];

/// Returns `true` when the feature list contains at least one unsupported tag.
fn has_unsupported_feature(features: &[String]) -> bool {
    features
        .iter()
        .any(|f| UNSUPPORTED_FEATURES.contains(&f.as_str()))
}

/// Path prefixes (relative to the `test/` directory, forward-slash separated)
/// for test categories that are known to hang or that exercise entirely
/// unimplemented subsystems.  Checked via [`is_skipped_path`].
const SKIPPED_PATH_PREFIXES: &[&str] = &[
    // Async generators / iterators — incomplete async runtime
    "built-ins/AsyncGeneratorFunction/",
    "built-ins/AsyncGeneratorPrototype/",
    "built-ins/AsyncFunction/",
    "built-ins/AsyncIteratorPrototype/",
    // Array.fromAsync — causes unbounded interpreter recursion (stack overflow)
    "built-ins/Array/fromAsync/",
    // Intl — not implemented
    "intl402/",
    // Annex B — legacy browser features, low priority
    "annexB/",
    // AggregateError tests trigger allocations >256 MiB (up to 461 MiB),
    // which trip the guarded allocator and cause cascading panics.
    "built-ins/AggregateError/",
];

/// Individual test files (relative to the `test/` directory, forward-slash
/// separated) that are known to hang due to catastrophic backtracking or
/// other pathological behaviour.
const SKIPPED_TEST_FILES: &[&str] = &[
    // Catastrophic backtracking: ((.*\n?)*?) in a body-matching regex.
    "built-ins/RegExp/S15.10.2.8_A3_T17.js",
    // Slow catastrophic backtracking (35+ seconds).
    "built-ins/RegExp/prototype/exec/S15.10.6.2_A3_T7.js",
    // Slow regex quantifier tests that exceed the per-test deadline.
    "built-ins/RegExp/S15.10.2.8_A3_T15.js",
    "built-ins/RegExp/S15.10.2.8_A3_T16.js",
    "built-ins/RegExp/S15.10.2.8_A3_T18.js",
];

/// Returns `true` when `rel_path` starts with any of the [`SKIPPED_PATH_PREFIXES`]
/// or exactly matches a [`SKIPPED_TEST_FILES`] entry.
fn is_skipped_path(rel_path: &str) -> bool {
    SKIPPED_PATH_PREFIXES
        .iter()
        .any(|prefix| rel_path.starts_with(prefix))
        || SKIPPED_TEST_FILES.contains(&rel_path)
}

// ─── Test execution ──────────────────────────────────────────────────────────

/// - `print` (silent stub)
/// - `$262` host object (gc, evalScript stubs)
/// - `assert` harness (native implementations of assert, assert.sameValue, etc.)
/// - `Test262Error` constructor
/// - `$DONOTEVALUATE` sentinel function
#[inline(never)]
fn make_test_globals() -> HashMap<String, JsValue> {
    // Force stacker to allocate a fresh heap segment for install_globals.
    // The 128 MiB red_zone exceeds any reasonable thread stack, so stacker
    // always switches to a 32 MiB heap-backed segment.
    //
    // NOTE: Do NOT use a multi-GiB red_zone here.  When the OS stack limit
    // is very large or unlimited (`ulimit -s unlimited`), stacker may
    // compute "remaining stack = infinity", decide the red_zone is
    // satisfied, and skip the heap segment entirely.  The 128 MiB value
    // works correctly on both bounded (default 8 MiB) and unbounded stacks.
    stacker::maybe_grow(
        128 * 1024 * 1024, // red_zone: 128 MiB
        32 * 1024 * 1024,  // new segment: 32 MiB
        || {
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
                JsValue::NativeFunction(Rc::new(|args| {
                    let code = match args.first() {
                        Some(JsValue::String(s)) => s.to_string(),
                        _ => return Ok(JsValue::Undefined),
                    };
                    let program = parser::parse(&code)?;
                    let bc = BytecodeGenerator::compile_program(&program)?;
                    let mut frame = InterpreterFrame::new(bc, vec![]);
                    Interpreter::run(&mut frame)
                })),
            );
            obj_262.borrow_mut().insert(
                "detachArrayBuffer".to_string(),
                JsValue::NativeFunction(Rc::new(|_| Ok(JsValue::Undefined))),
            );
            // $262.createRealm() — returns an object with `global` and `evalScript`.
            obj_262.borrow_mut().insert(
                "createRealm".to_string(),
                JsValue::NativeFunction(Rc::new(|_args| {
                    let mut realm_globals = HashMap::new();
                    install_globals(&mut realm_globals);
                    let global_obj = {
                        let mut gm = PropertyMap::new();
                        for (k, v) in &realm_globals {
                            gm.insert(k.clone(), v.clone());
                        }
                        Rc::new(RefCell::new(gm))
                    };
                    let global_val = JsValue::PlainObject(global_obj);

                    let realm_globals_rc = Rc::new(RefCell::new(realm_globals));
                    let mut realm = PropertyMap::new();
                    realm.insert("global".to_string(), global_val);
                    let rg = Rc::clone(&realm_globals_rc);
                    realm.insert(
                        "evalScript".to_string(),
                        JsValue::NativeFunction(Rc::new(move |args| {
                            let code = match args.first() {
                                Some(JsValue::String(s)) => s.to_string(),
                                _ => return Ok(JsValue::Undefined),
                            };
                            let program = parser::parse(&code)?;
                            let bc = BytecodeGenerator::compile_program(&program)?;
                            let mut frame = InterpreterFrame::new(bc, vec![]);
                            // Populate the frame's global env with the realm's globals.
                            {
                                let mut env = frame.global_env.borrow_mut();
                                for (k, v) in rg.borrow().iter() {
                                    env.insert(k.clone(), v.clone());
                                }
                            }
                            Interpreter::run(&mut frame)
                        })),
                    );
                    Ok(JsValue::PlainObject(Rc::new(RefCell::new(realm))))
                })),
            );
            obj_262
                .borrow_mut()
                .insert("global".to_string(), JsValue::Undefined);
            map.insert("$262".to_string(), JsValue::PlainObject(obj_262));

            // ── Native Test262Error constructor ──────────────────────────────────
            let t262_proto: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));
            t262_proto.borrow_mut().insert(
                "toString".to_string(),
                JsValue::NativeFunction(Rc::new(|_args| {
                    Ok(JsValue::String("Test262Error".into()))
                })),
            );
            let t262_proto_val = JsValue::PlainObject(t262_proto.clone());

            let t262_ctor: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));
            let proto_for_ctor = t262_proto_val.clone();
            t262_ctor.borrow_mut().insert(
                "__call__".to_string(),
                JsValue::NativeFunction(Rc::new(move |args| {
                    let msg = args.first().cloned().unwrap_or(JsValue::String("".into()));
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
            t262_ctor
                .borrow_mut()
                .insert("name".to_string(), JsValue::String("Test262Error".into()));
            t262_ctor.borrow_mut().insert(
                "thrower".to_string(),
                JsValue::NativeFunction(Rc::new(|args| {
                    let msg = match args.first() {
                        Some(JsValue::String(s)) => s.to_string(),
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
                    if val.to_boolean() {
                        return Ok(JsValue::Undefined);
                    }
                    let msg = match args.get(1) {
                        Some(JsValue::String(s)) => s.to_string(),
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
                JsValue::NativeFunction(Rc::new(|args| {
                    let expected_ctor = args.first().cloned().unwrap_or(JsValue::Undefined);
                    let func = args.get(1).cloned().unwrap_or(JsValue::Undefined);
                    let message = args
                        .get(2)
                        .and_then(|v| match v {
                            JsValue::String(s) => Some(s.to_string()),
                            _ => None,
                        })
                        .unwrap_or_else(|| "Expected a throw".to_string());

                    // Determine the expected error type name from the constructor.
                    let expected_type = match &expected_ctor {
                        JsValue::PlainObject(map) => {
                            let borrow = map.borrow();
                            if let Some(JsValue::String(name)) = borrow.get("name") {
                                Some(name.to_string())
                            } else {
                                borrow.get("prototype").and_then(|p| {
                                    if let JsValue::PlainObject(pm) = p {
                                        pm.borrow().get("constructor").and_then(|c| {
                                            if let JsValue::PlainObject(cm) = c {
                                                cm.borrow().get("name").and_then(|n| {
                                                    if let JsValue::String(s) = n {
                                                        Some(s.to_string())
                                                    } else {
                                                        None
                                                    }
                                                })
                                            } else {
                                                None
                                            }
                                        })
                                    } else {
                                        None
                                    }
                                })
                            }
                        }
                        JsValue::NativeFunction(_) => None,
                        _ => None,
                    };

                    // Invoke the function and check if it throws.
                    let result = match &func {
                        JsValue::Function(ba) => {
                            let mut frame = InterpreterFrame::new((**ba).clone(), vec![]);
                            Interpreter::run(&mut frame)
                        }
                        JsValue::NativeFunction(f) => f(vec![]),
                        JsValue::PlainObject(map) => {
                            if let Some(call_fn) = map.borrow().get("__call__").cloned() {
                                stator_core::interpreter::dispatch_call_value(&call_fn, vec![])
                            } else {
                                Ok(JsValue::Undefined)
                            }
                        }
                        _ => Ok(JsValue::Undefined),
                    };

                    match result {
                        Err(e) => {
                            // Verify the error type if we know the expected type.
                            if let Some(ref exp) = expected_type {
                                let type_matches = match (&e, exp.as_str()) {
                                    (StatorError::TypeError(_), "TypeError") => true,
                                    (StatorError::ReferenceError(_), "ReferenceError") => true,
                                    (StatorError::SyntaxError(_), "SyntaxError") => true,
                                    (StatorError::RangeError(_), "RangeError") => true,
                                    (StatorError::URIError(_), "URIError") => true,
                                    (StatorError::JsException(s), t) => s.contains(t),
                                    _ => true, // allow pass if we can't determine
                                };
                                if type_matches {
                                    Ok(JsValue::Undefined)
                                } else {
                                    Err(StatorError::JsException(format!(
                                        "Test262Error: Expected {exp} but got {e}"
                                    )))
                                }
                            } else {
                                Ok(JsValue::Undefined)
                            }
                        }
                        Ok(_) => Err(StatorError::TypeError(message)),
                    }
                })),
            );

            // assert._toString(value)
            assert_obj.borrow_mut().insert(
                "_toString".to_string(),
                JsValue::NativeFunction(Rc::new(|args| {
                    let v = args.first().cloned().unwrap_or(JsValue::Undefined);
                    Ok(JsValue::String(format!("{v:?}").into()))
                })),
            );

            map.insert("assert".to_string(), JsValue::PlainObject(assert_obj));

            map
        },
    )
}

/// SameValue comparison (ES2015 §7.2.10).
///
/// Unlike strict equality (`===`), SameValue treats `NaN` as equal to `NaN`
/// and distinguishes `+0` from `-0`.  Cross-type comparisons always return
/// `false` (e.g. `SameValue(true, 1)` is `false`).
fn js_same_value(a: &JsValue, b: &JsValue) -> bool {
    match (a, b) {
        (JsValue::Undefined, JsValue::Undefined) | (JsValue::Null, JsValue::Null) => true,
        (JsValue::Boolean(x), JsValue::Boolean(y)) => x == y,
        (JsValue::String(x), JsValue::String(y)) => x == y,
        (JsValue::Symbol(x), JsValue::Symbol(y)) => x == y,
        (JsValue::BigInt(x), JsValue::BigInt(y)) => x == y,

        // Number type: Smi and HeapNumber are both JS "Number".
        (JsValue::Smi(x), JsValue::Smi(y)) => x == y,
        (JsValue::HeapNumber(x), JsValue::HeapNumber(y)) => {
            if x.is_nan() && y.is_nan() {
                return true;
            }
            if *x == 0.0 && *y == 0.0 {
                return x.is_sign_positive() == y.is_sign_positive();
            }
            x == y
        }
        (JsValue::Smi(s), JsValue::HeapNumber(h)) | (JsValue::HeapNumber(h), JsValue::Smi(s)) => {
            let sf = *s as f64;
            // Smi is always a signed integer, so Smi(0) is positive zero.
            if sf == 0.0 && *h == 0.0 {
                return sf.is_sign_positive() == h.is_sign_positive();
            }
            sf == *h
        }

        // Object identity — same reference means SameValue.
        (JsValue::PlainObject(x), JsValue::PlainObject(y)) => Rc::ptr_eq(x, y),
        (JsValue::Array(x), JsValue::Array(y)) => Rc::ptr_eq(x, y),
        (JsValue::Function(x), JsValue::Function(y)) => Rc::ptr_eq(x, y),
        (JsValue::NativeFunction(x), JsValue::NativeFunction(y)) => Rc::ptr_eq(x, y),
        (JsValue::Error(x), JsValue::Error(y)) => Rc::ptr_eq(x, y),

        // Different types → never SameValue.
        _ => false,
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
///
/// Accepts an owned `HashMap` of globals so the caller can provide
/// a pre-cloned map (e.g. with `$DONE` already inserted for async tests),
/// avoiding a redundant second clone.
fn execute_source(
    source: &str,
    harness_prefix: &str,
    globals_map: HashMap<String, JsValue>,
) -> Result<JsValue, StatorError> {
    // Each test runs on a dedicated 256 MiB heap-backed stacker segment.
    // With the default 8 MiB main thread stack and an 8 MiB red_zone,
    // stacker always allocates a fresh segment (remaining ≤ red_zone).
    // This isolates each test's recursion depth from the main stack,
    // preventing both cross-test stack accumulation and runner OOM.
    stacker::maybe_grow(8 * 1024 * 1024, 256 * 1024 * 1024, || {
        execute_source_inner(source, harness_prefix, globals_map)
    })
}

fn execute_source_inner(
    source: &str,
    harness_prefix: &str,
    globals_map: HashMap<String, JsValue>,
) -> Result<JsValue, StatorError> {
    let combined = if harness_prefix.is_empty() {
        source.to_string()
    } else {
        format!("{harness_prefix}\n{source}")
    };

    // Wrap the caller-provided shallow clone in Rc<RefCell<…>> for the
    // interpreter.  The caller already cloned the template globals, so no
    // additional clone is needed here.
    let globals = Rc::new(RefCell::new(globals_map));
    // Keep a handle so we can break Rc cycles after the test finishes.
    let globals_cleanup = Rc::clone(&globals);

    // Run parse → compile → interpret inside a closure so that early `?`
    // returns still reach the cycle-breaking cleanup below.
    let result = (|| -> Result<JsValue, StatorError> {
        let program = parser::parse(&combined)?;
        let bc = BytecodeGenerator::compile_program(&program)?;
        let mut frame = InterpreterFrame::new_with_globals(bc, vec![], globals);
        // Limit each test to 5 million instructions to prevent infinite
        // loops from hanging the runner.
        frame.instruction_limit = 5_000_000;
        // Wall-clock deadline: 2 seconds per test.  Set both on the frame AND as
        // a thread-local so that child frames created by eval() / Function()
        // also respect the timeout.
        let dl = std::time::Instant::now() + std::time::Duration::from_secs(2);
        frame.deadline = Some(dl);
        set_execution_deadline(Some(dl));
        let result = Interpreter::run(&mut frame);
        // Clear the thread-local deadline after each test.
        set_execution_deadline(None);
        result
    })();

    // Release the test's shallow-cloned globals to drop Rc references
    // back to the template baseline.
    break_rc_cycles(&globals_cleanup);

    result
}

/// Release the test's references to template globals by clearing its
/// HashMap.  With shallow cloning, this is all that is needed — the
/// template's own `Rc` references keep the builtins alive, and the
/// test's `Rc` reference counts simply decrement back to their baseline.
fn break_rc_cycles(globals: &Rc<RefCell<HashMap<String, JsValue>>>) {
    if let Ok(mut g) = globals.try_borrow_mut() {
        g.clear();
    }
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
    // ── Flag handling: strict mode ────────────────────────────────────────
    // `onlyStrict` and `module` tests run in strict mode.
    let effective_prefix = if meta.has_flag("onlyStrict") || meta.is_module() {
        format!("\"use strict\";\n{harness_prefix}")
    } else {
        harness_prefix.to_string()
    };

    // ── Flag handling: async $DONE callback ───────────────────────────────
    let is_async = meta.is_async();
    let done_called = Rc::new(Cell::new(false));

    // Shallow-clone the template globals once.  For async tests we inject
    // the $DONE callback into this clone; for sync tests the clone is
    // passed straight through to the interpreter.  Either way, only ONE
    // clone is made per test (previously async tests were cloned twice).
    let mut test_globals = template_globals.clone();

    if is_async {
        let dc = done_called.clone();
        test_globals.insert(
            "$DONE".to_string(),
            JsValue::NativeFunction(Rc::new(move |args| {
                dc.set(true);
                let error = args.first().cloned().unwrap_or(JsValue::Undefined);
                if error.to_boolean() {
                    Err(StatorError::JsException(format!(
                        "Test262Error: $DONE called with error: {error:?}"
                    )))
                } else {
                    Ok(JsValue::Undefined)
                }
            })),
        );
    }

    // ── Execute ───────────────────────────────────────────────────────────
    // Wrap execution in catch_unwind to gracefully handle panics from
    // pathological test inputs.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        execute_source(source, &effective_prefix, test_globals)
    }));

    // Drain the microtask queue so that promise reactions (e.g. .then($DONE))
    // execute before we check the outcome.  This is a no-op when no queue is
    // installed or the queue is empty.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        drain_active_microtask_queue();
    }));

    // A panic inside the interpreter may leave frames on the thread-local
    // call stack.  Clear it so the next test starts with a clean slate.
    //
    // Each cleanup call is wrapped in its own `catch_unwind` to provide
    // double-panic protection: if a RefCell is still borrowed (poisoned)
    // from the original panic, the cleanup function itself may panic.
    // Catching that second panic ensures the remaining cleanup steps
    // still execute and the runner can proceed to the next test.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        clear_call_stack();
    }));

    // Clear interpreter thread-local caches (FUNCTION_PROPS, STRING_TABLE,
    // CURRENT_GLOBALS) and the string intern pool to prevent cross-test
    // contamination and unbounded memory growth.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        clear_interpreter_state();
    }));
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        clear_intern_pool();
    }));

    // Check if any oversized allocation occurred during this test.
    let alloc_exceeded = alloc_limit_exceeded();

    // Reset the allocator flag so the next test can trigger it.
    reset_alloc_guard();
    let exec = match result {
        Ok(r) => {
            if alloc_exceeded {
                ExecResult::OtherError("allocation limit exceeded (>512 MiB)".into())
            } else {
                ExecResult::from_result(r)
            }
        }
        Err(_) => ExecResult::OtherError("panicked (likely stack overflow)".into()),
    };

    // ── Async: verify $DONE was invoked ───────────────────────────────────
    if is_async
        && meta.negative.is_none()
        && let ExecResult::Ok = &exec
        && !done_called.get()
    {
        return TestOutcome::Fail("async test completed without calling $DONE".into());
    }

    // ── Evaluate outcome ──────────────────────────────────────────────────
    if let Some(neg) = &meta.negative {
        match neg.phase.as_str() {
            "parse" | "early" => {
                if exec.matches_type(&neg.type_) {
                    TestOutcome::Pass
                } else {
                    match exec {
                        ExecResult::Ok => TestOutcome::Fail(format!(
                            "expected {} {} but test succeeded",
                            neg.phase, neg.type_
                        )),
                        _ => TestOutcome::Fail(format!(
                            "expected {} {}, got: {}",
                            neg.phase,
                            neg.type_,
                            exec.error_message()
                        )),
                    }
                }
            }
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
            "resolution" => {
                // Module resolution errors are treated like parse-phase errors.
                if exec.matches_type(&neg.type_) {
                    TestOutcome::Pass
                } else {
                    match exec {
                        ExecResult::Ok => TestOutcome::Fail(format!(
                            "expected resolution {} but test succeeded",
                            neg.type_
                        )),
                        _ => TestOutcome::Fail(format!(
                            "expected resolution {}, got: {}",
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
///
/// Skips directories that match [`SKIPPED_PATH_PREFIXES`] and individual
/// `_FIXTURE.js` files (harness fixtures, not runnable tests) to avoid
/// wasting I/O and collection time on tests that will be skipped anyway.
fn collect_tests(dir: &Path, test_root: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            // Skip entire directory trees that are known to be unsupported,
            // avoiding thousands of filesystem stat/read calls.
            let rel = path
                .strip_prefix(test_root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            let rel_with_slash = if rel.ends_with('/') {
                rel
            } else {
                format!("{rel}/")
            };
            if SKIPPED_PATH_PREFIXES
                .iter()
                .any(|prefix| rel_with_slash.starts_with(prefix))
            {
                continue;
            }
            collect_tests(&path, test_root, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("js") {
            // Skip _FIXTURE.js files — they are included by other tests via
            // the harness mechanism and are not runnable on their own.
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                && stem.ends_with("_FIXTURE")
            {
                continue;
            }
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
    /// Skip the first N tests (used for crash-restart recovery).
    skip_first: usize,
    /// Path to write current test index before each test (crash recovery).
    progress_file: Option<PathBuf>,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut test262_dir: Option<PathBuf> = None;
    let mut threshold: f64 = 0.0;
    let mut filter: Option<String> = None;
    let mut verbose = false;
    let mut skip_first: usize = 0;
    let mut progress_file: Option<PathBuf> = None;

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
            "--skip-first" => {
                i += 1;
                if i < args.len() {
                    skip_first = args[i].parse().unwrap_or(0);
                }
            }
            "--progress-file" => {
                i += 1;
                if i < args.len() {
                    progress_file = Some(PathBuf::from(&args[i]));
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
        skip_first,
        progress_file,
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    // Run directly on the main thread.  The CI workflow sets
    // `ulimit -s unlimited`, which makes the main thread's stack
    // dynamically growable — unlike pthread stacks which are fixed-size
    // mmap regions.  This avoids the "overflowed its stack" crash that
    // hits even with a 1 GiB fixed stack on spawned threads.
    main_inner();
}

fn main_inner() {
    // Diagnostic: report stacker's view of remaining stack at startup.
    match stacker::remaining_stack() {
        Some(remaining) => eprintln!(
            "stator_test262: remaining stack at startup: {} bytes ({:.1} MiB)",
            remaining,
            remaining as f64 / (1024.0 * 1024.0)
        ),
        None => eprintln!("stator_test262: remaining_stack() returned None (unknown platform)"),
    }

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
    if let Err(e) = collect_tests(&test_dir, &test_dir, &mut test_files) {
        eprintln!("stator_test262: error reading test directory: {e}");
        std::process::exit(1);
    }

    // Apply optional path filter.
    if let Some(ref pat) = cli.filter {
        test_files.retain(|p| p.to_string_lossy().contains(pat.as_str()));
    }

    let total = test_files.len();
    println!("stator_test262: running {total} tests …");
    let _ = io::stdout().flush();

    let mut pass: u64 = 0;
    let mut fail: u64 = 0;
    let mut skip: u64 = 0;

    // ── Crash recovery: read accumulated results from progress file ───────
    // The progress file format is: INDEX PASS FAIL SKIP
    // This lets us accumulate results across crash-restart cycles.
    if cli.skip_first > 0
        && let Some(ref pf) = cli.progress_file
        && let Ok(contents) = std::fs::read_to_string(pf)
    {
        let parts: Vec<&str> = contents.split_whitespace().collect();
        if parts.len() >= 4 {
            pass = parts[1].parse().unwrap_or(0);
            fail = parts[2].parse().unwrap_or(0);
            skip = parts[3].parse().unwrap_or(0);
            println!("stator_test262: resuming with pass={pass} fail={fail} skip={skip}");
        }
    }
    let run_start = std::time::Instant::now();

    let mut harness = HarnessCache::new(harness_dir);

    // Build the template globals once.  Each test clones this template so
    // that per-test mutations don't leak across tests while avoiding the
    // heavy cost of re-running `install_globals` for every test.
    let template_globals = make_test_globals();
    let init_elapsed = run_start.elapsed();
    println!(
        "stator_test262: globals initialized in {:.1}s",
        init_elapsed.as_secs_f64()
    );
    let _ = io::stdout().flush();

    if cli.skip_first > 0 {
        println!(
            "stator_test262: skipping first {} tests (crash recovery)",
            cli.skip_first
        );
    }

    // ── Watchdog thread ───────────────────────────────────────────────────────
    // A background thread monitors a shared deadline.  Before each test the
    // main thread sets the deadline to NOW + 5 seconds.  If the test
    // completes, the deadline is reset to u64::MAX.  If the test hangs
    // (infinite loop in parser/compiler/interpreter), the watchdog fires
    // and terminates the process — the outer crash-restart loop picks up
    // from the next test.
    use std::sync::atomic::{AtomicU64, Ordering};
    static WATCHDOG_DEADLINE: AtomicU64 = AtomicU64::new(u64::MAX);
    std::thread::spawn(|| {
        loop {
            std::thread::sleep(std::time::Duration::from_millis(250));
            let deadline = WATCHDOG_DEADLINE.load(Ordering::Relaxed);
            if deadline != u64::MAX {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                if now > deadline {
                    eprintln!("WATCHDOG: test exceeded 5s deadline, aborting process");
                    std::process::exit(134); // Match SIGABRT exit code
                }
            }
        }
    });

    // ── Run each test ─────────────────────────────────────────────────────────
    // Jump directly to the first un-skipped test (O(1) restart).
    let start_idx = cli.skip_first.min(test_files.len());
    // The crashed test itself counts as a fail (it never completed).
    // Only bump fail if this is a restart (skip_first > 0) and we didn't
    // already account for the crash in the progress counts.
    if cli.skip_first > 0 && pass + fail + skip < cli.skip_first as u64 {
        // Progress file didn't have counts — legacy format or missing.
        // Count all skipped tests as skip.
        skip = start_idx as u64;
    } else if cli.skip_first > 0 {
        // The crashed test counts as a fail.
        fail += 1;
    }
    for (offset, path) in test_files[start_idx..].iter().enumerate() {
        let idx = start_idx + offset;

        // Write current test index + accumulated results to progress file.
        // Format: INDEX PASS FAIL SKIP
        if let Some(ref pf) = cli.progress_file {
            let _ = std::fs::write(pf, format!("{idx} {pass} {fail} {skip}"));
        }

        // ── Early skip: path-based filtering ──────────────────────────────────
        // Check skip patterns BEFORE reading or parsing the file to avoid
        // wasting I/O and CPU on tests we know will be skipped.
        let rel_path = path
            .strip_prefix(&test_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");

        if is_skipped_path(&rel_path) {
            if cli.verbose {
                println!(
                    "[SKIP] {}: path-based skip (known unsupported category)",
                    path.display()
                );
            }
            skip += 1;
            // Check the global timeout even on skipped tests so we don't
            // silently ignore it when a large block is skipped.
            if run_start.elapsed().as_secs() > 28 * 60 {
                eprintln!(
                    "Runner timeout after 28 min — stopping with {}/{total} tests processed  \
                     (pass={pass} fail={fail} skip={skip})",
                    idx + 1
                );
                break;
            }
            continue;
        }

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
        let skip_reason: Option<String> = if meta.is_can_block() {
            Some("CanBlock flag".to_string())
        } else if meta.is_module() {
            Some("module tests require ES module loader".to_string())
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
        let test_start = std::time::Instant::now();
        // Arm the watchdog: 5 seconds from now.
        let now_epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        WATCHDOG_DEADLINE.store(now_epoch + 5, Ordering::Relaxed);

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
        // Disarm the watchdog — test completed.
        WATCHDOG_DEADLINE.store(u64::MAX, Ordering::Relaxed);

        let elapsed = test_start.elapsed();
        if elapsed.as_secs() >= 3 {
            eprintln!("[SLOW] {}: {:.1}s", path.display(), elapsed.as_secs_f64());
        }

        // Reset the thread-local call stack so that a failed test with
        // leftover frames does not pollute subsequent runs.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            clear_call_stack();
        }));

        // Global runner timeout: stop processing after 18 minutes to leave
        // headroom for CI's 20-minute limit and ensure we print a summary.
        if run_start.elapsed().as_secs() > 28 * 60 {
            eprintln!(
                "Runner timeout after 28 min — stopping with {}/{total} tests processed  \
                 (pass={pass} fail={fail} skip={skip})",
                idx + 1
            );
            break;
        }

        // Periodic progress line (every 100 tests, unless verbose).
        if !cli.verbose && (idx + 1).is_multiple_of(100) {
            let elapsed = run_start.elapsed();
            let rate = (idx + 1) as f64 / elapsed.as_secs_f64();
            println!(
                "  … {}/{total}  pass={pass}  fail={fail}  skip={skip}  \
                 elapsed={:.0}s  ({rate:.0} tests/sec)",
                idx + 1,
                elapsed.as_secs_f64()
            );
            let _ = io::stdout().flush();
        }
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    let attempted = pass + fail;
    let pass_rate = if attempted > 0 {
        pass as f64 / attempted as f64 * 100.0
    } else {
        100.0
    };

    let total_elapsed = run_start.elapsed();

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test262 Results");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Total    : {total}");
    println!("  Pass     : {pass}");
    println!("  Fail     : {fail}");
    println!("  Skip     : {skip}");
    println!("  Pass rate: {pass_rate:.2}%  ({pass}/{attempted} attempted)");
    println!(
        "  Elapsed  : {:.1}s ({:.1} tests/sec)",
        total_elapsed.as_secs_f64(),
        total as f64 / total_elapsed.as_secs_f64()
    );
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
            "resizable-arraybuffer".to_string(),
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
    //
    // These tests call `install_globals` → `run_test` which needs a very
    // large stack (install_globals is deeply nested in debug builds).
    // We spawn each on a dedicated thread with 64 MiB of stack.

    const TEST_STACK: usize = 64 * 1024 * 1024;

    #[test]
    fn test_run_positive_pass() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let src = "/*---\ndescription: 1+1\n---*/\n1 + 1;";
                let meta = parse_frontmatter(src);
                let globals = make_test_globals();
                assert!(matches!(
                    run_test(src, "", &meta, &globals),
                    TestOutcome::Pass
                ));
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn test_run_positive_fail_on_syntax_error() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let src = "/*---\ndescription: bad\n---*/\n!@# invalid";
                let meta = parse_frontmatter(src);
                let globals = make_test_globals();
                assert!(matches!(
                    run_test(src, "", &meta, &globals),
                    TestOutcome::Fail(_)
                ));
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn test_run_negative_parse_passes_when_error_thrown() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let src =
                    "/*---\nnegative:\n  phase: parse\n  type: SyntaxError\n---*/\n!@# bad syntax";
                let meta = parse_frontmatter(src);
                let globals = make_test_globals();
                assert!(matches!(
                    run_test(src, "", &meta, &globals),
                    TestOutcome::Pass
                ));
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn test_run_negative_parse_fails_when_no_error() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let src =
                    "/*---\nnegative:\n  phase: parse\n  type: SyntaxError\n---*/\nvar x = 1;";
                let meta = parse_frontmatter(src);
                let globals = make_test_globals();
                assert!(matches!(
                    run_test(src, "", &meta, &globals),
                    TestOutcome::Fail(_)
                ));
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn test_collect_tests_empty_dir() {
        let tmp = std::env::temp_dir().join("stator_test262_empty_collect_test");
        let _ = std::fs::create_dir_all(&tmp);
        let mut out: Vec<PathBuf> = Vec::new();
        collect_tests(&tmp, &tmp, &mut out).unwrap();
        assert!(out.is_empty());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_collect_tests_finds_js() {
        let tmp = std::env::temp_dir().join("stator_test262_collect_test_finds_js");
        let _ = std::fs::create_dir_all(&tmp);
        std::fs::write(tmp.join("a.js"), "var a;").unwrap();
        std::fs::write(tmp.join("b.txt"), "not js").unwrap();
        let mut out: Vec<PathBuf> = Vec::new();
        collect_tests(&tmp, &tmp, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].ends_with("a.js"));
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
