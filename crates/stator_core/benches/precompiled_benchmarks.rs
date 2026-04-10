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

use stator_core::bytecode::bytecode_generator::BytecodeGenerator;
use stator_core::interpreter::{
    GlobalEnv, Interpreter, InterpreterFrame, globals_promotion_diagnostics, licm_diagnostics,
    maglev_deopt_categories, maglev_diagnostics,
};
use stator_core::parser::recursive_descent;

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
    stator_core::builtins::install_globals::install_globals(&mut env.vars);
    env.rebuild_slots();
    env.globals_installed = true;
    Rc::new(RefCell::new(env))
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
    // Warmup: trigger Maglev compilation
    for _ in 0..100 {
        let mut frame = InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
        let _ = Interpreter::run(&mut frame);
    }
    // Wait for background Maglev compilation to complete
    let warmup_start = std::time::Instant::now();
    while !ba.has_all_maglev_jit_code()
        && !ba.has_turbofan_jit_code()
        && warmup_start.elapsed() < std::time::Duration::from_millis(2000)
    {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    eprintln!(
        "BENCH_JIT[fib_40]: has_maglev={} has_turbofan={} deopt_count={} inv_count={}",
        ba.has_maglev_jit_code(),
        ba.has_turbofan_jit_code(),
        ba.maglev_deopt_count(),
        ba.invocation_count(),
    );
    let diag_before = maglev_diagnostics();
    let cats_before = maglev_deopt_categories();
    c.bench_function("fib_40_iterative_precompiled", |b| {
        b.iter(|| {
            let mut frame =
                InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
            black_box(Interpreter::run(black_box(&mut frame)).unwrap())
        });
    });
    print_maglev_diag("fib_40_iterative", &diag_before, &cats_before);
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
    for _ in 0..100 {
        let mut frame = InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
        let _ = Interpreter::run(&mut frame);
    }
    let warmup_start = std::time::Instant::now();
    while !ba.has_all_maglev_jit_code()
        && !ba.has_turbofan_jit_code()
        && warmup_start.elapsed() < std::time::Duration::from_millis(2000)
    {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    eprintln!(
        "BENCH_JIT[arithmetic]: has_maglev={} has_turbofan={} deopt_count={} inv_count={}",
        ba.has_maglev_jit_code(),
        ba.has_turbofan_jit_code(),
        ba.maglev_deopt_count(),
        ba.invocation_count(),
    );
    let diag_before = maglev_diagnostics();
    let cats_before = maglev_deopt_categories();
    c.bench_function("arithmetic_loop_10k_precompiled", |b| {
        b.iter(|| {
            let mut frame =
                InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
            black_box(Interpreter::run(black_box(&mut frame)).unwrap());
        });
    });
    print_maglev_diag("arithmetic_loop_10k", &diag_before, &cats_before);
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
    for _ in 0..100 {
        let mut frame = InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
        let _ = Interpreter::run(&mut frame);
    }
    let warmup_start = std::time::Instant::now();
    while !ba.has_all_maglev_jit_code()
        && !ba.has_turbofan_jit_code()
        && warmup_start.elapsed() < std::time::Duration::from_millis(2000)
    {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    eprintln!(
        "BENCH_JIT[property_access]: has_maglev={} has_turbofan={} deopt_count={} inv_count={}",
        ba.has_maglev_jit_code(),
        ba.has_turbofan_jit_code(),
        ba.maglev_deopt_count(),
        ba.invocation_count(),
    );
    let diag_before = maglev_diagnostics();
    let cats_before = maglev_deopt_categories();
    c.bench_function("property_access_1k_precompiled", |b| {
        b.iter(|| {
            let mut frame =
                InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
            black_box(Interpreter::run(black_box(&mut frame)).unwrap())
        });
    });
    print_maglev_diag("property_access_1k", &diag_before, &cats_before);
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
    for _ in 0..100 {
        let mut frame = InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
        let _ = Interpreter::run(&mut frame);
    }
    let warmup_start = std::time::Instant::now();
    while !ba.has_all_maglev_jit_code()
        && !ba.has_turbofan_jit_code()
        && warmup_start.elapsed() < std::time::Duration::from_millis(2000)
    {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    eprintln!(
        "BENCH_JIT[object_creation]: has_maglev={} has_turbofan={} deopt_count={} inv_count={}",
        ba.has_maglev_jit_code(),
        ba.has_turbofan_jit_code(),
        ba.maglev_deopt_count(),
        ba.invocation_count(),
    );
    let diag_before = maglev_diagnostics();
    let cats_before = maglev_deopt_categories();
    c.bench_function("object_creation_1k_precompiled", |b| {
        b.iter(|| {
            let mut frame =
                InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
            black_box(Interpreter::run(black_box(&mut frame)).unwrap())
        });
    });
    print_maglev_diag("object_creation_1k", &diag_before, &cats_before);
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
    for _ in 0..100 {
        let mut frame = InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
        let _ = Interpreter::run(&mut frame);
    }
    let warmup_start = std::time::Instant::now();
    while !ba.has_all_maglev_jit_code()
        && !ba.has_turbofan_jit_code()
        && warmup_start.elapsed() < std::time::Duration::from_millis(2000)
    {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    eprintln!(
        "BENCH_JIT[array_push_sum]: has_maglev={} has_turbofan={} deopt_count={} inv_count={}",
        ba.has_maglev_jit_code(),
        ba.has_turbofan_jit_code(),
        ba.maglev_deopt_count(),
        ba.invocation_count(),
    );
    let diag_before = maglev_diagnostics();
    let cats_before = maglev_deopt_categories();
    c.bench_function("array_push_sum_1k_precompiled", |b| {
        b.iter(|| {
            let mut frame =
                InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
            black_box(Interpreter::run(black_box(&mut frame)).unwrap())
        });
    });
    print_maglev_diag("array_push_sum_1k", &diag_before, &cats_before);
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
    for _ in 0..100 {
        let mut frame = InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
        let _ = Interpreter::run(&mut frame);
    }
    let warmup_start = std::time::Instant::now();
    while !ba.has_all_maglev_jit_code()
        && !ba.has_turbofan_jit_code()
        && warmup_start.elapsed() < std::time::Duration::from_millis(2000)
    {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    eprintln!(
        "BENCH_JIT[closure_counter]: has_maglev={} has_turbofan={} deopt_count={} inv_count={}",
        ba.has_maglev_jit_code(),
        ba.has_turbofan_jit_code(),
        ba.maglev_deopt_count(),
        ba.invocation_count(),
    );
    let diag_before = maglev_diagnostics();
    let cats_before = maglev_deopt_categories();
    c.bench_function("closure_counter_1k_precompiled", |b| {
        b.iter(|| {
            let mut frame =
                InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
            black_box(Interpreter::run(black_box(&mut frame)).unwrap())
        });
    });
    print_maglev_diag("closure_counter_1k", &diag_before, &cats_before);
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
    for _ in 0..100 {
        let mut frame = InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
        let _ = Interpreter::run(&mut frame);
    }
    let warmup_start = std::time::Instant::now();
    while !ba.has_all_maglev_jit_code()
        && !ba.has_turbofan_jit_code()
        && warmup_start.elapsed() < std::time::Duration::from_millis(2000)
    {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    eprintln!(
        "BENCH_JIT[prototype_chain]: has_maglev={} has_turbofan={} deopt_count={} inv_count={}",
        ba.has_maglev_jit_code(),
        ba.has_turbofan_jit_code(),
        ba.maglev_deopt_count(),
        ba.invocation_count(),
    );
    let diag_before = maglev_diagnostics();
    let cats_before = maglev_deopt_categories();
    c.bench_function("prototype_chain_1k_precompiled", |b| {
        b.iter(|| {
            let mut frame =
                InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
            black_box(Interpreter::run(black_box(&mut frame)).unwrap())
        });
    });
    print_maglev_diag("prototype_chain_1k", &diag_before, &cats_before);
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
    for _ in 0..100 {
        let mut frame = InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
        let _ = Interpreter::run(&mut frame);
    }
    let warmup_start = std::time::Instant::now();
    while !ba.has_all_maglev_jit_code()
        && !ba.has_turbofan_jit_code()
        && warmup_start.elapsed() < std::time::Duration::from_millis(2000)
    {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    eprintln!(
        "BENCH_JIT[sieve_primes]: has_maglev={} has_turbofan={} deopt_count={} inv_count={}",
        ba.has_maglev_jit_code(),
        ba.has_turbofan_jit_code(),
        ba.maglev_deopt_count(),
        ba.invocation_count(),
    );
    let diag_before = maglev_diagnostics();
    let cats_before = maglev_deopt_categories();
    c.bench_function("sieve_primes_1k_precompiled", |b| {
        b.iter(|| {
            let mut frame =
                InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
            black_box(Interpreter::run(black_box(&mut frame)).unwrap())
        });
    });
    print_maglev_diag("sieve_primes_1k", &diag_before, &cats_before);
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
    for _ in 0..100 {
        let mut frame = InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
        let _ = Interpreter::run(&mut frame);
    }
    let warmup_start = std::time::Instant::now();
    while !ba.has_all_maglev_jit_code()
        && !ba.has_turbofan_jit_code()
        && warmup_start.elapsed() < std::time::Duration::from_millis(2000)
    {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    eprintln!(
        "BENCH_JIT[deep_object]: has_maglev={} has_turbofan={} deopt_count={} inv_count={}",
        ba.has_maglev_jit_code(),
        ba.has_turbofan_jit_code(),
        ba.maglev_deopt_count(),
        ba.invocation_count(),
    );
    let diag_before = maglev_diagnostics();
    let cats_before = maglev_deopt_categories();
    c.bench_function("deep_object_access_1k_precompiled", |b| {
        b.iter(|| {
            let mut frame =
                InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
            black_box(Interpreter::run(black_box(&mut frame)).unwrap())
        });
    });
    print_maglev_diag("deep_object_access_1k", &diag_before, &cats_before);
}

// ---------------------------------------------------------------------------
// Diagnostic helper
// ---------------------------------------------------------------------------

fn print_maglev_diag(
    name: &str,
    diag_before: &(u64, u64, u64, u64, u32, usize, u32, u32, u32),
    cats_before: &[u64; 6],
) {
    let (tried, executed, deopted, not_ready, compilations, code_bytes, started, failed, panicked) =
        maglev_diagnostics();
    let cats_after = maglev_deopt_categories();
    eprintln!(
        "MAGLEV_DIAG[{name}]: tried={} executed={} deopted={} not_ready={} compilations={} code_bytes={} started={} failed={} panicked={}",
        tried - diag_before.0,
        executed - diag_before.1,
        deopted - diag_before.2,
        not_ready - diag_before.3,
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
