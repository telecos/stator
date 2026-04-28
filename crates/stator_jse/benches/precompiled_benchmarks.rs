//! Pre-compiled V8 comparison benchmarks.
//!
//! These skip parse+compile and measure pure interpreter/JIT speed, giving a
//! fairer comparison against V8's JIT-compiled hot code.  JS sources match
//! `benchmarks/v8_comparison/benchmarks.js` exactly.
//!
//! This is a **separate binary** from `engine_benchmarks` so that Maglev warmup
//! (which may SIGSEGV on certain patterns) does not crash the V8-comparison
//! benchmark process.

use std::cell::RefCell;
use std::hint::black_box;
use std::rc::Rc;

use criterion::{Criterion, criterion_group, criterion_main};

use stator_jse::bytecode::bytecode_generator::BytecodeGenerator;
use stator_jse::compiler::baseline::compiler::{
    STUB_DEOPT_SLOTS, STUB_NAMES, first_deopt_counts, reset_first_deopt_counts,
    reset_stub_call_counts, reset_stub_deopt_counts, stub_call_counts, stub_deopt_counts,
};
use stator_jse::interpreter::{
    GlobalEnv, Interpreter, InterpreterFrame, dispatch_entry_diagnostics,
    globals_promotion_diagnostics, jit_entry_diagnostics, licm_diagnostics,
    maglev_deopt_categories, maglev_diagnostics,
};
use stator_jse::parser::recursive_descent;

/// Install a SIGSEGV handler that prints diagnostic info before aborting.
#[cfg(unix)]
fn install_sigsegv_handler() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        extern "C" fn handler(
            sig: libc::c_int,
            info: *mut libc::siginfo_t,
            _ctx: *mut libc::c_void,
        ) {
            // SAFETY: reading si_addr from a valid siginfo_t in signal context.
            let fault_addr = if info.is_null() {
                0usize
            } else {
                unsafe { (*info).si_addr() as usize }
            };
            // Print to stderr — CI captures this.
            let msg = format!(
                "\n=== SIGSEGV HANDLER ===\nsignal={sig} fault_addr=0x{fault_addr:016x}\n\
                 backtrace:\n{}\n=== END SIGSEGV ===\n",
                std::backtrace::Backtrace::force_capture()
            );
            // SAFETY: write(2) is async-signal-safe; _exit terminates immediately.
            unsafe {
                libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len());
                libc::_exit(11);
            }
        }
        // SAFETY: installing a signal handler with valid sigaction struct.
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = handler as usize;
            sa.sa_flags = libc::SA_SIGINFO | libc::SA_RESETHAND;
            libc::sigemptyset(&mut sa.sa_mask);
            libc::sigaction(libc::SIGSEGV, &sa, std::ptr::null_mut());
        }
    });
}

#[cfg(not(unix))]
fn install_sigsegv_handler() {}

/// Check whether the current benchmark name matches the CLI filter passed
/// via `cargo bench -- "^name$"`.  Criterion calls every registered benchmark
/// function regardless of the filter, so we need this guard to skip warmup
/// for non-targeted benchmarks (preventing a crashing benchmark from killing
/// the whole process).
fn matches_bench_filter(bench_name: &str) -> bool {
    for arg in std::env::args().skip(1) {
        if arg.starts_with('-') {
            continue;
        }
        // First non-flag positional argument is the filter regex.
        let pat = arg.trim_matches('^').trim_matches('$');
        return bench_name == pat || bench_name.contains(pat) || pat.contains(bench_name);
    }
    true // no filter — run everything
}

/// CI-friendly Criterion configuration with reduced samples to avoid timeouts.
fn ci_config() -> Criterion {
    Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(1))
        .sample_size(10)
}

/// Create a shared [`GlobalEnv`] with builtins pre-installed.
fn make_global_env() -> Rc<RefCell<GlobalEnv>> {
    let mut env = GlobalEnv::new();
    stator_jse::builtins::install_globals::install_globals(&mut env.vars);
    env.rebuild_slots();
    env.globals_installed = true;
    Rc::new(RefCell::new(env))
}

/// Print deopt state after reset — helps diagnose why Maglev may be blocked.
fn print_deopt_state(
    name: &str,
    ba: &std::rc::Rc<stator_jse::bytecode::bytecode_array::BytecodeArray>,
) {
    eprintln!(
        "DEOPT_STATE[{name}]: has_deopted={} count={} next_try={} inv={} cache_populated={}",
        ba.jit_maglev_has_deopted(),
        ba.maglev_deopt_count(),
        ba.maglev_next_try_at(),
        ba.invocation_count(),
        ba.has_maglev_executable_cached(),
    );
}

/// Two-phase warmup: (1) trigger Maglev compilation with warm ICs, then
/// (2) run a validation phase with fresh deopt counter so Maglev's own
/// inline ICs warm up.  Without the second phase, Maglev may deopt on
/// the first JIT execution (cold JIT ICs) and exponential backoff blocks
/// it permanently during Criterion measurement.
fn warmup_with_maglev(
    ba: &Rc<stator_jse::bytecode::bytecode_array::BytecodeArray>,
    env: &Rc<RefCell<GlobalEnv>>,
    name: &str,
) {
    install_sigsegv_handler();
    // Phase 1: 100 interpreter iterations to warm ICs + trigger Maglev.
    for _ in 0..100 {
        let mut frame = InterpreterFrame::new_with_globals(Rc::clone(ba), vec![], Rc::clone(env));
        let _ = Interpreter::run(&mut frame);
    }
    // Wait for background Maglev compilation to complete.
    let start = std::time::Instant::now();
    while !ba.has_all_maglev_jit_code()
        && !ba.has_turbofan_jit_code()
        && start.elapsed() < std::time::Duration::from_millis(2000)
    {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    eprintln!(
        "BENCH_JIT[{name}]: has_maglev={} has_turbofan={} deopt_count={} inv_count={}",
        ba.has_maglev_jit_code(),
        ba.has_turbofan_jit_code(),
        ba.maglev_deopt_count(),
        ba.invocation_count(),
    );
    // Phase 2: Reset deopt counter and run validation iterations.
    // Maglev retries with compiled code + warm interpreter ICs.
    // The JIT's own inline ICs populate during these iterations.
    ba.reset_maglev_deopt_count();
    for _ in 0..100 {
        let mut frame = InterpreterFrame::new_with_globals(Rc::clone(ba), vec![], Rc::clone(env));
        let _ = Interpreter::run(&mut frame);
    }
    // Phase 3: Final reset before Criterion measurement.
    ba.reset_maglev_deopt_count();
    reset_stub_deopt_counts();
    reset_first_deopt_counts();
    print_deopt_state(name, ba);
}

/// Interpreter-only warmup: warms ICs without triggering Maglev JIT.
/// Used for benchmarks where Maglev produces incorrect code (e.g. sieve's
/// nested loops) but the interpreter is already fast enough.
fn warmup_interpreter_only(
    ba: &Rc<stator_jse::bytecode::bytecode_array::BytecodeArray>,
    env: &Rc<RefCell<GlobalEnv>>,
    name: &str,
) {
    install_sigsegv_handler();
    // Block Maglev permanently for this BytecodeArray by setting the
    // next-try threshold to u32::MAX so jit_maglev_has_deopted() returns
    // true on every check.
    ba.set_maglev_next_try_at(u32::MAX);
    for _ in 0..200 {
        let mut frame = InterpreterFrame::new_with_globals(Rc::clone(ba), vec![], Rc::clone(env));
        let _ = Interpreter::run(&mut frame);
    }
    eprintln!("BENCH_INTERP[{name}]: inv_count={}", ba.invocation_count(),);
    reset_stub_deopt_counts();
    reset_first_deopt_counts();
}

// ---------------------------------------------------------------------------
// Precompiled benchmark functions
// ---------------------------------------------------------------------------

fn bench_fib_40_iterative_precompiled(c: &mut Criterion) {
    let source = r#"
        var a = 0, b = 1;
        for (var i = 0; i < 40; i++) {
            var t = a + b;
            a = b;
            b = t;
        }
        b;
    "#;
    let program = recursive_descent::parse(source).unwrap();
    let bytecode = BytecodeGenerator::compile_program(&program).unwrap();
    let ba = Rc::new(bytecode);
    eprintln!(
        "BENCH_SETUP[fib_40]: ba_ptr={:p} bc_len={}",
        &*ba as *const _,
        ba.bytecodes().len()
    );
    let env = make_global_env();
    if matches_bench_filter("fib_40_iterative_precompiled") {
        warmup_with_maglev(&ba, &env, "fib_40");
    }
    let diag_before = maglev_diagnostics();
    let cats_before = maglev_deopt_categories();
    let stubs_before = stub_deopt_counts();
    let first_deopts_before = first_deopt_counts();
    let jit_before = jit_entry_diagnostics();
    let dispatch_before = dispatch_entry_diagnostics();
    c.bench_function("fib_40_iterative_precompiled", |b| {
        b.iter(|| black_box(Interpreter::run_fast(&ba, &[], &env).unwrap()));
    });
    print_maglev_diag(
        "fib_40_iterative",
        &diag_before,
        &cats_before,
        &stubs_before,
        &first_deopts_before,
        &jit_before,
        &dispatch_before,
    );
}

fn bench_js_arithmetic_precompiled(c: &mut Criterion) {
    let source = r#"
        var n = 0;
        for (var i = 0; i < 10000; i++) {
            n = (n + i * 3 - 1) | 0;
        }
        n;
    "#;
    let program = recursive_descent::parse(source).unwrap();
    let bytecode = BytecodeGenerator::compile_program(&program).unwrap();
    let ba = Rc::new(bytecode);
    eprintln!(
        "BENCH_SETUP[arithmetic_loop]: ba_ptr={:p} bc_len={}",
        &*ba as *const _,
        ba.bytecodes().len()
    );
    let env = make_global_env();
    if matches_bench_filter("arithmetic_loop_10k_precompiled") {
        warmup_with_maglev(&ba, &env, "arithmetic_loop");
    }
    let diag_before = maglev_diagnostics();
    let cats_before = maglev_deopt_categories();
    let stubs_before = stub_deopt_counts();
    let first_deopts_before = first_deopt_counts();
    let jit_before = jit_entry_diagnostics();
    let dispatch_before = dispatch_entry_diagnostics();
    c.bench_function("arithmetic_loop_10k_precompiled", |b| {
        b.iter(|| {
            black_box(Interpreter::run_fast(&ba, &[], &env).unwrap());
        });
    });
    print_maglev_diag(
        "arithmetic_loop_10k",
        &diag_before,
        &cats_before,
        &stubs_before,
        &first_deopts_before,
        &jit_before,
        &dispatch_before,
    );
}

fn bench_property_access_1k_precompiled(c: &mut Criterion) {
    let source = r#"
        var obj = { a: 1, b: 2, c: 3, d: 4, e: 5 };
        var sum = 0;
        for (var i = 0; i < 1000; i++) {
            sum = sum + obj.a + obj.b + obj.c + obj.d + obj.e;
        }
        sum;
    "#;
    let program = recursive_descent::parse(source).unwrap();
    let bytecode = BytecodeGenerator::compile_program(&program).unwrap();
    let ba = Rc::new(bytecode);
    eprintln!(
        "BENCH_SETUP[property_access]: ba_ptr={:p} bc_len={}",
        &*ba as *const _,
        ba.bytecodes().len()
    );
    let env = make_global_env();
    if matches_bench_filter("property_access_1k_precompiled") {
        warmup_with_maglev(&ba, &env, "property_access");
    }
    let diag_before = maglev_diagnostics();
    let cats_before = maglev_deopt_categories();
    let stubs_before = stub_deopt_counts();
    let first_deopts_before = first_deopt_counts();
    let jit_before = jit_entry_diagnostics();
    let dispatch_before = dispatch_entry_diagnostics();
    c.bench_function("property_access_1k_precompiled", |b| {
        b.iter(|| black_box(Interpreter::run_fast(&ba, &[], &env).unwrap()));
    });
    print_maglev_diag(
        "property_access_1k",
        &diag_before,
        &cats_before,
        &stubs_before,
        &first_deopts_before,
        &jit_before,
        &dispatch_before,
    );
    let (loops, hoisted, named_hoisted, blocked) = licm_diagnostics();
    eprintln!(
        "LICM_DIAG[property_access_1k]: loops={loops} hoisted={hoisted} named_generic_hoisted={named_hoisted} blocked_by_side_effects={blocked}"
    );
    let (opt_promoted, opt_skipped, cg_promoted) = globals_promotion_diagnostics();
    eprintln!(
        "GLOBALS_DIAG[property_access_1k]: opt_promoted={opt_promoted} opt_skipped={opt_skipped} codegen_promoted={cg_promoted}"
    );
}

fn bench_object_creation_1k_precompiled(c: &mut Criterion) {
    let source = r#"
        var last;
        for (var i = 0; i < 1000; i++) {
            last = { x: i, y: i + 1, z: i * 2 };
        }
        last.x + last.y + last.z;
    "#;
    let program = recursive_descent::parse(source).unwrap();
    let bytecode = BytecodeGenerator::compile_program(&program).unwrap();
    let ba = Rc::new(bytecode);
    eprintln!(
        "BENCH_SETUP[object_creation]: ba_ptr={:p} bc_len={}",
        &*ba as *const _,
        ba.bytecodes().len()
    );
    let env = make_global_env();
    if matches_bench_filter("object_creation_1k_precompiled") {
        warmup_with_maglev(&ba, &env, "object_creation");
    }
    let diag_before = maglev_diagnostics();
    let cats_before = maglev_deopt_categories();
    let stubs_before = stub_deopt_counts();
    let first_deopts_before = first_deopt_counts();
    let jit_before = jit_entry_diagnostics();
    let dispatch_before = dispatch_entry_diagnostics();
    c.bench_function("object_creation_1k_precompiled", |b| {
        b.iter(|| black_box(Interpreter::run_fast(&ba, &[], &env).unwrap()));
    });
    print_maglev_diag(
        "object_creation_1k",
        &diag_before,
        &cats_before,
        &stubs_before,
        &first_deopts_before,
        &jit_before,
        &dispatch_before,
    );
}

fn bench_array_push_sum_1k_precompiled(c: &mut Criterion) {
    let source = r#"
        var arr = [];
        for (var i = 0; i < 1000; i++) {
            arr.push(i);
        }
        var sum = 0;
        for (var i = 0; i < arr.length; i++) {
            sum = sum + arr[i];
        }
        sum;
    "#;
    let program = recursive_descent::parse(source).unwrap();
    let bytecode = BytecodeGenerator::compile_program(&program).unwrap();
    let ba = Rc::new(bytecode);
    eprintln!(
        "BENCH_SETUP[array_push]: ba_ptr={:p} bc_len={}",
        &*ba as *const _,
        ba.bytecodes().len()
    );
    let env = make_global_env();
    if matches_bench_filter("array_push_sum_1k_precompiled") {
        warmup_with_maglev(&ba, &env, "array_push_sum");
    }
    let diag_before = maglev_diagnostics();
    let cats_before = maglev_deopt_categories();
    let stubs_before = stub_deopt_counts();
    let first_deopts_before = first_deopt_counts();
    let jit_before = jit_entry_diagnostics();
    let dispatch_before = dispatch_entry_diagnostics();
    reset_stub_call_counts();
    c.bench_function("array_push_sum_1k_precompiled", |b| {
        b.iter(|| black_box(Interpreter::run_fast(&ba, &[], &env).unwrap()));
    });
    print_maglev_diag(
        "array_push_sum_1k",
        &diag_before,
        &cats_before,
        &stubs_before,
        &first_deopts_before,
        &jit_before,
        &dispatch_before,
    );
    let (loops, hoisted, named_hoisted, blocked) = licm_diagnostics();
    eprintln!(
        "LICM_DIAG[array_push_sum_1k]: loops={loops} hoisted={hoisted} named_generic_hoisted={named_hoisted} blocked_by_side_effects={blocked}"
    );
    let (opt_promoted, opt_skipped, cg_promoted) = globals_promotion_diagnostics();
    eprintln!(
        "GLOBALS_DIAG[array_push_sum_1k]: opt_promoted={opt_promoted} opt_skipped={opt_skipped} codegen_promoted={cg_promoted}"
    );
    // Print per-stub FFI call counts to diagnose inline fast path usage.
    let calls = stub_call_counts();
    eprint!("STUB_CALLS[array_push_sum_1k]:");
    for i in 0..STUB_DEOPT_SLOTS {
        if calls[i] > 0 {
            eprint!(" {}={}", STUB_NAMES[i], calls[i]);
        }
    }
    eprintln!();
}

fn bench_closure_counter_1k_precompiled(c: &mut Criterion) {
    let source = r#"
        function make_counter() {
            var count = 0;
            return function() { count = count + 1; return count; };
        }
        var counter = make_counter();
        var result = 0;
        for (var i = 0; i < 1000; i++) {
            result = counter();
        }
        result;
    "#;
    let program = recursive_descent::parse(source).unwrap();
    let bytecode = BytecodeGenerator::compile_program(&program).unwrap();
    let ba = Rc::new(bytecode);
    eprintln!(
        "BENCH_SETUP[closure_counter]: ba_ptr={:p} bc_len={}",
        &*ba as *const _,
        ba.bytecodes().len()
    );
    let env = make_global_env();
    if matches_bench_filter("closure_counter_1k_precompiled") {
        warmup_with_maglev(&ba, &env, "closure_counter");
    }
    let diag_before = maglev_diagnostics();
    let cats_before = maglev_deopt_categories();
    let stubs_before = stub_deopt_counts();
    let first_deopts_before = first_deopt_counts();
    let jit_before = jit_entry_diagnostics();
    let dispatch_before = dispatch_entry_diagnostics();
    c.bench_function("closure_counter_1k_precompiled", |b| {
        b.iter(|| black_box(Interpreter::run_fast(&ba, &[], &env).unwrap()));
    });
    print_maglev_diag(
        "closure_counter_1k",
        &diag_before,
        &cats_before,
        &stubs_before,
        &first_deopts_before,
        &jit_before,
        &dispatch_before,
    );
}

fn bench_prototype_chain_1k_precompiled(c: &mut Criterion) {
    let source = r#"
        function Base() {}
        Base.prototype.x = 42;
        function Mid() {}
        Mid.prototype = new Base();
        function Leaf() {}
        Leaf.prototype = new Mid();
        var obj = new Leaf();
        var sum = 0;
        for (var i = 0; i < 1000; i++) {
            sum = sum + obj.x;
        }
        sum;
    "#;
    let program = recursive_descent::parse(source).unwrap();
    let bytecode = BytecodeGenerator::compile_program(&program).unwrap();
    let ba = Rc::new(bytecode);
    eprintln!(
        "BENCH_SETUP[prototype_chain]: ba_ptr={:p} bc_len={}",
        &*ba as *const _,
        ba.bytecodes().len()
    );
    let env = make_global_env();
    if matches_bench_filter("prototype_chain_1k_precompiled") {
        warmup_with_maglev(&ba, &env, "prototype_chain");
    }
    let diag_before = maglev_diagnostics();
    let cats_before = maglev_deopt_categories();
    let stubs_before = stub_deopt_counts();
    let first_deopts_before = first_deopt_counts();
    let jit_before = jit_entry_diagnostics();
    let dispatch_before = dispatch_entry_diagnostics();
    c.bench_function("prototype_chain_1k_precompiled", |b| {
        b.iter(|| black_box(Interpreter::run_fast(&ba, &[], &env).unwrap()));
    });
    print_maglev_diag(
        "prototype_chain_1k",
        &diag_before,
        &cats_before,
        &stubs_before,
        &first_deopts_before,
        &jit_before,
        &dispatch_before,
    );
}

fn bench_sieve_primes_1k_precompiled(c: &mut Criterion) {
    let source = r#"
        var n = 1000;
        var sieve = [];
        for (var i = 0; i <= n; i++) sieve[i] = true;
        sieve[0] = false;
        sieve[1] = false;
        for (var i = 2; i * i <= n; i++) {
            if (sieve[i]) {
                for (var j = i * i; j <= n; j = j + i) {
                    sieve[j] = false;
                }
            }
        }
        var count = 0;
        for (var i = 0; i <= n; i++) {
            if (sieve[i]) count = count + 1;
        }
        count;
    "#;
    let program = recursive_descent::parse(source).unwrap();
    let bytecode = BytecodeGenerator::compile_program(&program).unwrap();
    let ba = Rc::new(bytecode);
    eprintln!(
        "BENCH_SETUP[sieve_primes]: ba_ptr={:p} bc_len={}",
        &*ba as *const _,
        ba.bytecodes().len()
    );
    let env = make_global_env();
    if matches_bench_filter("sieve_primes_1k_precompiled") {
        // Maglev works for sieve via the eval_js path (engine_benchmarks
        // show 3.7M JIT hits with 0 deopts).  Use standard Maglev warmup.
        warmup_with_maglev(&ba, &env, "sieve_primes");
    }
    let diag_before = maglev_diagnostics();
    let cats_before = maglev_deopt_categories();
    let stubs_before = stub_deopt_counts();
    let first_deopts_before = first_deopt_counts();
    let jit_before = jit_entry_diagnostics();
    let dispatch_before = dispatch_entry_diagnostics();
    c.bench_function("sieve_primes_1k_precompiled", |b| {
        b.iter(|| black_box(Interpreter::run_fast(&ba, &[], &env).unwrap()));
    });
    print_maglev_diag(
        "sieve_primes_1k",
        &diag_before,
        &cats_before,
        &stubs_before,
        &first_deopts_before,
        &jit_before,
        &dispatch_before,
    );
}

fn bench_deep_object_access_1k_precompiled(c: &mut Criterion) {
    let source = r#"
        var root = { a: { b: { c: { d: { e: 99 } } } } };
        var sum = 0;
        for (var i = 0; i < 1000; i++) {
            sum = sum + root.a.b.c.d.e;
        }
        sum;
    "#;
    let program = recursive_descent::parse(source).unwrap();
    let bytecode = BytecodeGenerator::compile_program(&program).unwrap();
    let ba = Rc::new(bytecode);
    eprintln!(
        "BENCH_SETUP[deep_object]: ba_ptr={:p} bc_len={}",
        &*ba as *const _,
        ba.bytecodes().len()
    );
    let env = make_global_env();
    if matches_bench_filter("deep_object_access_1k_precompiled") {
        warmup_with_maglev(&ba, &env, "deep_object");
    }
    let diag_before = maglev_diagnostics();
    let cats_before = maglev_deopt_categories();
    let stubs_before = stub_deopt_counts();
    let first_deopts_before = first_deopt_counts();
    let jit_before = jit_entry_diagnostics();
    let dispatch_before = dispatch_entry_diagnostics();
    c.bench_function("deep_object_access_1k_precompiled", |b| {
        b.iter(|| black_box(Interpreter::run_fast(&ba, &[], &env).unwrap()));
    });
    print_maglev_diag(
        "deep_object_access_1k",
        &diag_before,
        &cats_before,
        &stubs_before,
        &first_deopts_before,
        &jit_before,
        &dispatch_before,
    );
}

// ---------------------------------------------------------------------------
// Diagnostic helper
// ---------------------------------------------------------------------------

fn print_maglev_diag(
    name: &str,
    diag_before: &(u64, u64, u64, u64, u32, usize, u32, u32, u32, u64, u64, u64),
    cats_before: &[u64; 6],
    stubs_before: &[u64; STUB_DEOPT_SLOTS],
    first_deopts_before: &[u64; STUB_DEOPT_SLOTS],
    jit_before: &(u64, u64, u64),
    dispatch_before: &(u64, u64),
) {
    let (
        tried,
        executed,
        deopted,
        not_ready,
        compilations,
        code_bytes,
        started,
        failed,
        panicked,
        blocked,
        cache_empty,
        turbofan_hit,
    ) = maglev_diagnostics();
    let cats_after = maglev_deopt_categories();
    let stubs_after = stub_deopt_counts();
    let first_deopts_after = first_deopt_counts();
    eprintln!(
        "MAGLEV_DIAG[{name}]: tried={} executed={} deopted={} not_ready={} blocked={} cache_empty={} turbofan_hit={} compilations={} code_bytes={} started={} failed={} panicked={}",
        tried - diag_before.0,
        executed - diag_before.1,
        deopted - diag_before.2,
        not_ready - diag_before.3,
        blocked - diag_before.9,
        cache_empty - diag_before.10,
        turbofan_hit - diag_before.11,
        compilations - diag_before.4,
        code_bytes - diag_before.5,
        started - diag_before.6,
        failed - diag_before.7,
        panicked - diag_before.8
    );
    eprintln!(
        "  deopt_cats_delta: generic={} overflow={} stub={} global={} divzero={} unknown={}",
        cats_after[0] - cats_before[0],
        cats_after[1] - cats_before[1],
        cats_after[2] - cats_before[2],
        cats_after[3] - cats_before[3],
        cats_after[4] - cats_before[4],
        cats_after[5] - cats_before[5]
    );
    // Per-stub deopt breakdown — only print non-zero deltas.
    let mut parts = Vec::new();
    for i in 0..STUB_DEOPT_SLOTS {
        let delta = stubs_after[i] - stubs_before[i];
        if delta > 0 {
            parts.push(format!("{}={}", STUB_NAMES[i], delta));
        }
    }
    if parts.is_empty() {
        eprintln!("  stub_deopts: (none)");
    } else {
        eprintln!("  stub_deopts: {}", parts.join(" "));
    }
    // Per-stub first-deopt-per-invocation breakdown.
    let mut first_parts = Vec::new();
    for i in 0..STUB_DEOPT_SLOTS {
        let delta = first_deopts_after[i] - first_deopts_before[i];
        if delta > 0 {
            first_parts.push(format!("{}={}", STUB_NAMES[i], delta));
        }
    }
    if first_parts.is_empty() {
        eprintln!("  first_deopt_stubs: (none)");
    } else {
        eprintln!("  first_deopt_stubs: {}", first_parts.join(" "));
    }
    // Atomic (non-TLS) diagnostics for cross-checking.
    let (entered, hit, miss) = jit_entry_diagnostics();
    let (inner_now, dispatch_now) = dispatch_entry_diagnostics();
    eprintln!(
        "  ATOMIC_JIT[{name}]: entered={} maglev_hit={} maglev_miss={} (entered_delta={})",
        entered,
        hit,
        miss,
        entered - jit_before.0
    );
    eprintln!(
        "  DISPATCH[{name}]: run_inner={} run_dispatch={} (inner_delta={} dispatch_delta={})",
        inner_now,
        dispatch_now,
        inner_now - dispatch_before.0,
        dispatch_now - dispatch_before.1
    );
}

// ---------------------------------------------------------------------------
// Criterion groups and main
// ---------------------------------------------------------------------------

criterion_group! {
    name = v8_precompiled_benches;
    config = ci_config();
    targets =
        bench_fib_40_iterative_precompiled,
        bench_js_arithmetic_precompiled,
        bench_property_access_1k_precompiled,
        bench_object_creation_1k_precompiled,
        bench_array_push_sum_1k_precompiled,
        bench_closure_counter_1k_precompiled,
        bench_prototype_chain_1k_precompiled,
        bench_sieve_primes_1k_precompiled,
        bench_deep_object_access_1k_precompiled,
}

criterion_main!(v8_precompiled_benches);
