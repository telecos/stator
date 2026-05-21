//! Always-on, low-overhead diagnostic counters for compile attempts,
//! successes, and failures broken down by compilation tier.
//!
//! These counters are intended for Edge performance proofs and other
//! release-mode diagnostics, so they are **not** gated on
//! `cfg(debug_assertions)`.  Updates are single relaxed atomic
//! increments per public compile entry point, which is negligible
//! compared to the cost of any actual compilation step.
//!
//! # Usage
//!
//! ```
//! use stator_jse::compiler::compile_counters::{self, CompileTier};
//!
//! compile_counters::reset();
//! compile_counters::record_attempt(CompileTier::Baseline);
//! compile_counters::record_success(CompileTier::Baseline);
//! let snap = compile_counters::snapshot();
//! assert_eq!(snap.attempts(CompileTier::Baseline), 1);
//! assert_eq!(snap.successes(CompileTier::Baseline), 1);
//! assert_eq!(snap.failures(CompileTier::Baseline), 0);
//! ```
//!
//! The counters are process-global atomics, suitable for reading from
//! any thread and across isolates.  Future FFI exposure can wrap
//! [`snapshot`] and [`reset`] directly.

use std::sync::atomic::{AtomicU64, Ordering};

/// Compilation tiers tracked by the counters module.
///
/// The numeric discriminants double as indices into the underlying
/// atomic arrays and **must remain stable** for any future FFI
/// consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum CompileTier {
    /// Source-AST → bytecode compilation (interpreter tier).
    Interpreter = 0,
    /// Bytecode → native code via the non-optimising baseline JIT.
    Baseline = 1,
    /// Maglev mid-tier optimising compiler.
    Maglev = 2,
    /// Cranelift-backed Turbofan optimising compiler.
    Turbofan = 3,
}

impl CompileTier {
    /// Number of tier variants tracked by the counters.
    pub const COUNT: usize = 4;

    /// Stable short name used for diagnostic output.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Interpreter => "interpreter",
            Self::Baseline => "baseline",
            Self::Maglev => "maglev",
            Self::Turbofan => "turbofan",
        }
    }

    /// All tier variants in stable discriminant order.
    #[must_use]
    pub const fn all() -> [Self; Self::COUNT] {
        [
            Self::Interpreter,
            Self::Baseline,
            Self::Maglev,
            Self::Turbofan,
        ]
    }
}

// One AtomicU64 per (tier, kind).  Indexed as `[tier as usize]`.
static ATTEMPTS: [AtomicU64; CompileTier::COUNT] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static SUCCESSES: [AtomicU64; CompileTier::COUNT] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static FAILURES: [AtomicU64; CompileTier::COUNT] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Record that a compile of `tier` has been attempted.
///
/// Call once at the start of every public compile entry point for
/// the given tier, regardless of outcome.
#[inline]
pub fn record_attempt(tier: CompileTier) {
    ATTEMPTS[tier as usize].fetch_add(1, Ordering::Relaxed);
}

/// Record that a compile of `tier` completed successfully.
#[inline]
pub fn record_success(tier: CompileTier) {
    SUCCESSES[tier as usize].fetch_add(1, Ordering::Relaxed);
}

/// Record that a compile of `tier` failed (returned an `Err`).
#[inline]
pub fn record_failure(tier: CompileTier) {
    FAILURES[tier as usize].fetch_add(1, Ordering::Relaxed);
}

/// Convenience wrapper: record an attempt, run `f`, then record the
/// resulting success or failure and return its `Result`.
///
/// Use this in compile entry points to keep instrumentation in one
/// place and avoid forgetting to record the outcome on early returns.
#[inline]
pub fn record<T, E>(tier: CompileTier, f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    record_attempt(tier);
    match f() {
        Ok(v) => {
            record_success(tier);
            Ok(v)
        }
        Err(e) => {
            record_failure(tier);
            Err(e)
        }
    }
}

/// Immutable snapshot of the compile counters at a point in time.
///
/// Returned by [`snapshot`]; reading from a snapshot is consistent
/// per-tier per-kind but is not a globally atomic transaction
/// across all tiers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompileCountersSnapshot {
    attempts: [u64; CompileTier::COUNT],
    successes: [u64; CompileTier::COUNT],
    failures: [u64; CompileTier::COUNT],
}

impl CompileCountersSnapshot {
    /// Number of compile attempts recorded for `tier`.
    #[must_use]
    pub fn attempts(&self, tier: CompileTier) -> u64 {
        self.attempts[tier as usize]
    }

    /// Number of successful compiles recorded for `tier`.
    #[must_use]
    pub fn successes(&self, tier: CompileTier) -> u64 {
        self.successes[tier as usize]
    }

    /// Number of failed compiles recorded for `tier`.
    #[must_use]
    pub fn failures(&self, tier: CompileTier) -> u64 {
        self.failures[tier as usize]
    }

    /// Total attempts across all tiers.
    #[must_use]
    pub fn total_attempts(&self) -> u64 {
        self.attempts.iter().sum()
    }

    /// Total successes across all tiers.
    #[must_use]
    pub fn total_successes(&self) -> u64 {
        self.successes.iter().sum()
    }

    /// Total failures across all tiers.
    #[must_use]
    pub fn total_failures(&self) -> u64 {
        self.failures.iter().sum()
    }
}

/// Take a snapshot of the current global compile counters.
#[must_use]
pub fn snapshot() -> CompileCountersSnapshot {
    let mut snap = CompileCountersSnapshot::default();
    for tier in CompileTier::all() {
        let i = tier as usize;
        snap.attempts[i] = ATTEMPTS[i].load(Ordering::Relaxed);
        snap.successes[i] = SUCCESSES[i].load(Ordering::Relaxed);
        snap.failures[i] = FAILURES[i].load(Ordering::Relaxed);
    }
    snap
}

/// Reset every counter to zero.
///
/// Intended for use in tests and benchmark harnesses; production
/// telemetry should prefer taking and diffing [`snapshot`]s instead.
pub fn reset() {
    for i in 0..CompileTier::COUNT {
        ATTEMPTS[i].store(0, Ordering::Relaxed);
        SUCCESSES[i].store(0, Ordering::Relaxed);
        FAILURES[i].store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // The counters are process-global, so tests that mutate them must
    // run serially to avoid interfering with each other.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_record_attempt_increments_only_attempts() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        record_attempt(CompileTier::Baseline);
        record_attempt(CompileTier::Baseline);
        let s = snapshot();
        assert_eq!(s.attempts(CompileTier::Baseline), 2);
        assert_eq!(s.successes(CompileTier::Baseline), 0);
        assert_eq!(s.failures(CompileTier::Baseline), 0);
    }

    #[test]
    fn test_record_success_and_failure_are_per_tier() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        record_success(CompileTier::Interpreter);
        record_failure(CompileTier::Turbofan);
        let s = snapshot();
        assert_eq!(s.successes(CompileTier::Interpreter), 1);
        assert_eq!(s.failures(CompileTier::Interpreter), 0);
        assert_eq!(s.failures(CompileTier::Turbofan), 1);
        assert_eq!(s.successes(CompileTier::Turbofan), 0);
        assert_eq!(s.successes(CompileTier::Baseline), 0);
        assert_eq!(s.successes(CompileTier::Maglev), 0);
    }

    #[test]
    fn test_record_wrapper_counts_ok() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        let r: Result<i32, &str> = record(CompileTier::Maglev, || Ok(7));
        assert_eq!(r, Ok(7));
        let s = snapshot();
        assert_eq!(s.attempts(CompileTier::Maglev), 1);
        assert_eq!(s.successes(CompileTier::Maglev), 1);
        assert_eq!(s.failures(CompileTier::Maglev), 0);
    }

    #[test]
    fn test_record_wrapper_counts_err() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        let r: Result<(), &str> = record(CompileTier::Turbofan, || Err("boom"));
        assert_eq!(r, Err("boom"));
        let s = snapshot();
        assert_eq!(s.attempts(CompileTier::Turbofan), 1);
        assert_eq!(s.successes(CompileTier::Turbofan), 0);
        assert_eq!(s.failures(CompileTier::Turbofan), 1);
    }

    #[test]
    fn test_reset_zeroes_all_tiers_and_kinds() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        for tier in CompileTier::all() {
            record_attempt(tier);
            record_success(tier);
            record_failure(tier);
        }
        reset();
        let s = snapshot();
        for tier in CompileTier::all() {
            assert_eq!(s.attempts(tier), 0, "tier {:?}", tier);
            assert_eq!(s.successes(tier), 0, "tier {:?}", tier);
            assert_eq!(s.failures(tier), 0, "tier {:?}", tier);
        }
        assert_eq!(s.total_attempts(), 0);
        assert_eq!(s.total_successes(), 0);
        assert_eq!(s.total_failures(), 0);
    }

    #[test]
    fn test_totals_sum_across_tiers() {
        let _g = TEST_LOCK.lock().unwrap();
        reset();
        record_attempt(CompileTier::Interpreter);
        record_attempt(CompileTier::Baseline);
        record_success(CompileTier::Interpreter);
        record_failure(CompileTier::Baseline);
        let s = snapshot();
        assert_eq!(s.total_attempts(), 2);
        assert_eq!(s.total_successes(), 1);
        assert_eq!(s.total_failures(), 1);
    }

    #[test]
    fn test_tier_names_are_stable() {
        assert_eq!(CompileTier::Interpreter.name(), "interpreter");
        assert_eq!(CompileTier::Baseline.name(), "baseline");
        assert_eq!(CompileTier::Maglev.name(), "maglev");
        assert_eq!(CompileTier::Turbofan.name(), "turbofan");
    }

    /// Real-compiler integration: bytecode (interpreter tier)
    /// compilation increments interpreter attempts and successes
    /// on success.
    #[test]
    fn test_real_bytecode_compile_increments_interpreter_success() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::recursive_descent::parse_script;

        let _g = TEST_LOCK.lock().unwrap();
        let before = snapshot();
        let prog = parse_script("var x = 1 + 2;").expect("parse");
        BytecodeGenerator::compile_program(&prog).expect("compile");
        let after = snapshot();
        assert_eq!(
            after.attempts(CompileTier::Interpreter) - before.attempts(CompileTier::Interpreter),
            1,
        );
        assert_eq!(
            after.successes(CompileTier::Interpreter) - before.successes(CompileTier::Interpreter),
            1,
        );
        assert_eq!(
            after.failures(CompileTier::Interpreter) - before.failures(CompileTier::Interpreter),
            0,
        );
    }

    /// Real-compiler integration: bytecode (interpreter tier)
    /// compilation of a script that contains module declarations
    /// returns Err and increments the interpreter failure counter.
    #[test]
    fn test_real_bytecode_compile_increments_interpreter_failure() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::ast::{Program, ProgramItem, SourceType};
        use crate::parser::recursive_descent::parse_module;
        use crate::parser::scanner::{Position, Span};

        let _g = TEST_LOCK.lock().unwrap();
        let before = snapshot();

        // Construct a *script*-typed Program whose body contains a
        // ModuleDecl.  `compile_program_with_source` rejects this with
        // a SyntaxError, exercising the failure path of the wrapped
        // entry point.
        let module = parse_module("export const x = 1;").expect("parse module");
        let module_decls: Vec<ProgramItem> = module
            .body
            .into_iter()
            .filter(|item| matches!(item, ProgramItem::ModuleDecl(_)))
            .collect();
        assert!(!module_decls.is_empty(), "expected at least one ModuleDecl");
        let zero = Position {
            offset: 0,
            line: 1,
            column: 1,
        };
        let bogus_script = Program {
            loc: Span {
                start: zero,
                end: zero,
            },
            source_type: SourceType::Script,
            body: module_decls,
            is_strict: false,
        };
        let err = BytecodeGenerator::compile_program(&bogus_script);
        assert!(err.is_err(), "expected compile failure, got {:?}", err);

        let after = snapshot();
        assert_eq!(
            after.attempts(CompileTier::Interpreter) - before.attempts(CompileTier::Interpreter),
            1,
        );
        assert_eq!(
            after.successes(CompileTier::Interpreter) - before.successes(CompileTier::Interpreter),
            0,
        );
        assert_eq!(
            after.failures(CompileTier::Interpreter) - before.failures(CompileTier::Interpreter),
            1,
        );
    }
}
