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

use stator_core::bytecode::bytecode_generator::BytecodeGenerator;
use stator_core::error::StatorError;
use stator_core::interpreter::{Interpreter, InterpreterFrame};
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
    /// `sta.js` is always prepended (it defines `$ERROR` and `Test262Error`).
    /// The files listed in `includes` follow in order; `sta.js` is not
    /// duplicated if listed there too.
    fn build_prefix(&mut self, includes: &[String]) -> String {
        let mut parts: Vec<String> = Vec::new();

        // sta.js is always first.
        if let Ok(s) = self.get("sta.js") {
            parts.push(s.to_string());
        }

        for name in includes {
            if name == "sta.js" {
                continue; // Already added above.
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
    // Async / generators
    "async-functions",
    "async-iteration",
    "generators",
    "top-level-await",
    // Classes
    "class",
    "class-fields-private",
    "class-fields-public",
    "class-methods-private",
    "class-static-block",
    "class-static-fields-private",
    "class-static-fields-public",
    "class-static-methods-private",
    // Symbols
    "Symbol",
    "Symbol.asyncIterator",
    "Symbol.hasInstance",
    "Symbol.isConcatSpreadable",
    "Symbol.iterator",
    "Symbol.match",
    "Symbol.matchAll",
    "Symbol.replace",
    "Symbol.search",
    "Symbol.species",
    "Symbol.split",
    "Symbol.toPrimitive",
    "Symbol.toStringTag",
    "Symbol.unscopables",
    // Proxy / Reflect
    "Proxy",
    "Reflect",
    "Reflect.construct",
    // BigInt
    "BigInt",
    // Modules
    "arbitrary-module-namespace-names",
    "dynamic-import",
    "export-star-as-namespace-from-module",
    "import-assertions",
    "import-attributes",
    "import.meta",
    // SharedArrayBuffer / Atomics
    "Atomics",
    "SharedArrayBuffer",
    // WeakRef / FinalizationRegistry
    "FinalizationRegistry",
    "WeakRef",
    // Advanced RegExp
    "regexp-dotall",
    "regexp-lookbehind",
    "regexp-match-indices",
    "regexp-named-groups",
    "regexp-unicode-property-escapes",
    "regexp-v-flag",
    // Internationalisation
    "Intl",
    // Miscellaneous
    "globalThis",
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
/// Provides a silent `print` stub (some harness files call it) and a minimal
/// `$262` object with `gc()` and `evalScript()` stubs.
fn make_test_globals() -> Rc<RefCell<HashMap<String, JsValue>>> {
    let globals: Rc<RefCell<HashMap<String, JsValue>>> = Rc::new(RefCell::new(HashMap::new()));

    // Silent print — some harness files reference it.
    globals.borrow_mut().insert(
        "print".to_string(),
        JsValue::NativeFunction(Rc::new(|_| Ok(JsValue::Undefined))),
    );

    // Minimal $262 host-defined object.
    let obj_262: Rc<RefCell<HashMap<String, JsValue>>> = Rc::new(RefCell::new(HashMap::new()));
    obj_262.borrow_mut().insert(
        "gc".to_string(),
        JsValue::NativeFunction(Rc::new(|_| Ok(JsValue::Undefined))),
    );
    obj_262.borrow_mut().insert(
        "evalScript".to_string(),
        JsValue::NativeFunction(Rc::new(|_| Ok(JsValue::Undefined))),
    );
    globals
        .borrow_mut()
        .insert("$262".to_string(), JsValue::PlainObject(obj_262));

    globals
}

/// Returns `true` when `err` matches the Test262 `type` string from a
/// `negative` frontmatter entry.
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
fn execute_source(source: &str, harness_prefix: &str) -> Result<JsValue, StatorError> {
    let combined = if harness_prefix.is_empty() {
        source.to_string()
    } else {
        format!("{harness_prefix}\n{source}")
    };

    let globals = make_test_globals();

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
fn run_test(source: &str, harness_prefix: &str, meta: &TestMeta) -> TestOutcome {
    // Wrap execution in catch_unwind to gracefully handle stack overflows
    // or other panics from pathological test inputs.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        execute_source(source, harness_prefix)
    }));

    let result = match result {
        Ok(r) => r,
        Err(_) => return TestOutcome::Fail("panicked (likely stack overflow)".into()),
    };

    if let Some(neg) = &meta.negative {
        match neg.phase.as_str() {
            // "parse" errors occur during parsing (invalid token sequences).
            // "early" errors are static-semantics violations caught at
            // compile time (e.g. `return` outside a function).  Both phases
            // surface as `StatorError::SyntaxError` in our engine because we
            // combine parsing and static-semantic checking into one pass.
            "parse" | "early" => match result {
                Err(StatorError::SyntaxError(_)) => TestOutcome::Pass,
                Err(e) => {
                    TestOutcome::Fail(format!("expected {} SyntaxError, got: {e}", neg.phase))
                }
                Ok(_) => TestOutcome::Fail(format!(
                    "expected {} SyntaxError but test succeeded",
                    neg.phase
                )),
            },
            "runtime" => match result {
                Err(ref e) if error_matches_type(e, &neg.type_) => TestOutcome::Pass,
                Err(ref e) => {
                    TestOutcome::Fail(format!("expected runtime {}, got: {e}", neg.type_))
                }
                Ok(_) => {
                    TestOutcome::Fail(format!("expected runtime {} but test succeeded", neg.type_))
                }
            },
            other => TestOutcome::Skip(format!("unsupported negative phase: {other}")),
        }
    } else {
        match result {
            Ok(_) => TestOutcome::Pass,
            Err(e) => TestOutcome::Fail(e.to_string()),
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
        match run_test(&source, &harness_prefix, &meta) {
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
            "Symbol".to_string(),
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
        assert!(matches!(run_test(src, "", &meta), TestOutcome::Pass));
    }

    #[test]
    fn test_run_positive_fail_on_syntax_error() {
        let src = "/*---\ndescription: bad\n---*/\n!@# invalid";
        let meta = parse_frontmatter(src);
        assert!(matches!(run_test(src, "", &meta), TestOutcome::Fail(_)));
    }

    #[test]
    fn test_run_negative_parse_passes_when_error_thrown() {
        let src = "/*---\nnegative:\n  phase: parse\n  type: SyntaxError\n---*/\n!@# bad syntax";
        let meta = parse_frontmatter(src);
        assert!(matches!(run_test(src, "", &meta), TestOutcome::Pass));
    }

    #[test]
    fn test_run_negative_parse_fails_when_no_error() {
        let src = "/*---\nnegative:\n  phase: parse\n  type: SyntaxError\n---*/\nvar x = 1;";
        let meta = parse_frontmatter(src);
        assert!(matches!(run_test(src, "", &meta), TestOutcome::Fail(_)));
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
