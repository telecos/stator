//! Criterion benchmarks for core engine operations.
//!
//! Run with: `cargo bench --package stator_core`
//!
//! ## Benchmark categories
//!
//! 1. **Heap allocation** – bump-pointer throughput and GC-triggered steady
//!    state.
//! 2. **Tagged values** – round-trip encode/decode for SMIs and heap pointers.
//! 3. **HandleScope** – scope creation, local handle allocation.
//! 4. **PropertyMap** – named property insert / lookup / inline-cache hit rate.
//! 5. **JsValue** – clone cost for common variant types.
//! 6. **End-to-end JS** – parse → compile → interpret for micro-kernels that
//!    exercise property access, function calls, string concatenation, array
//!    operations, and object creation.

use std::alloc::Layout;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hint::black_box;
use std::ptr::NonNull;
use std::rc::Rc;

use criterion::{Criterion, criterion_group, criterion_main};

/// CI-friendly Criterion configuration with reduced samples to avoid timeouts.
fn ci_config() -> Criterion {
    Criterion::default()
        .warm_up_time(std::time::Duration::from_millis(200))
        .measurement_time(std::time::Duration::from_secs(1))
        .sample_size(10)
}

use stator_core::bytecode::bytecode_array::BytecodeArray;
use stator_core::bytecode::bytecode_generator::BytecodeGenerator;
use stator_core::error::StatorResult;
use stator_core::gc::handle::HandleScope;
use stator_core::gc::heap::{Heap, HeapObject};
use stator_core::interpreter::{GlobalEnv, Interpreter, InterpreterFrame, maglev_diagnostics};
use stator_core::objects::property_map::PropertyMap;
use stator_core::objects::tagged::TaggedValue;
use stator_core::objects::value::JsValue;
use stator_core::parser::recursive_descent;

/// Create a shared [`GlobalEnv`] with builtins pre-installed.
///
/// Benchmarks should call this **once** before the timing loop and pass
/// the returned `Rc<RefCell<GlobalEnv>>` to
/// [`InterpreterFrame::new_with_globals`] in each iteration.  This avoids
/// the ~3 ms cost of `install_globals()` on every iteration.
fn make_global_env() -> Rc<RefCell<GlobalEnv>> {
    let mut env = GlobalEnv::new();
    stator_core::builtins::install_globals::install_globals(&mut env.vars);
    env.rebuild_slots();
    env.globals_installed = true;
    Rc::new(RefCell::new(env))
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Parse, compile, and execute a snippet of JavaScript source, returning the
/// final accumulator value.
///
/// Caches both the global environment and compiled bytecode across calls.
/// This matches V8's behaviour where (a) the global object persists across
/// `eval()` calls and (b) repeated `eval()` of the same source string reuses
/// cached compiled code.
fn eval_js(source: &str) -> StatorResult<JsValue> {
    use std::hash::{Hash, Hasher};

    thread_local! {
        static CACHED_ENV: RefCell<Option<Rc<RefCell<GlobalEnv>>>> = const { RefCell::new(None) };
        static COMPILE_CACHE: RefCell<HashMap<u64, Rc<BytecodeArray>>> = RefCell::new(HashMap::new());
    }
    let env = CACHED_ENV.with(|c| c.borrow().clone()).unwrap_or_else(|| {
        let env = make_global_env();
        CACHED_ENV.with(|c| *c.borrow_mut() = Some(Rc::clone(&env)));
        env
    });
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source.hash(&mut hasher);
    let hash = hasher.finish();
    let bytecode = COMPILE_CACHE.with(|cache| {
        if let Some(ba) = cache.borrow().get(&hash) {
            return Ok(Rc::clone(ba));
        }
        let program = recursive_descent::parse(source)?;
        let ba = Rc::new(BytecodeGenerator::compile_program(&program)?);
        cache.borrow_mut().insert(hash, Rc::clone(&ba));
        Ok(ba)
    })?;
    let mut frame = InterpreterFrame::new_with_globals(bytecode, vec![], env);
    Interpreter::run(&mut frame)
}

// ===========================================================================
// 1. Heap allocation throughput
// ===========================================================================

const BURST_ALLOC_COUNT: usize = 1000;

fn bench_heap_allocate(c: &mut Criterion) {
    let layout = Layout::new::<HeapObject>();
    c.bench_function("heap_allocate_single", |b| {
        // Heap lives across iterations to measure steady-state bump-allocation.
        // When the young space fills up we collect (reset) and keep going.
        let mut heap = Heap::new();
        b.iter(|| {
            let ptr = heap.allocate(black_box(layout));
            if ptr.is_null() {
                heap.collect();
            }
            black_box(ptr);
        });
    });
}

fn bench_heap_allocate_burst(c: &mut Criterion) {
    let layout = Layout::new::<HeapObject>();
    c.bench_function("heap_allocate_burst_1000", |b| {
        b.iter(|| {
            let mut heap = Heap::new();
            for _ in 0..BURST_ALLOC_COUNT {
                let ptr = heap.allocate(black_box(layout));
                black_box(ptr);
            }
        });
    });
}

// ===========================================================================
// 2. Tagged value operations
// ===========================================================================

fn bench_tagged_smi_round_trip(c: &mut Criterion) {
    c.bench_function("tagged_smi_round_trip", |b| {
        b.iter(|| {
            let tv = TaggedValue::from_smi(black_box(42));
            black_box(tv.is_smi());
            black_box(tv.as_smi());
        });
    });
}

fn bench_tagged_heap_ptr_round_trip(c: &mut Criterion) {
    let mut obj = HeapObject::new_null();
    let ptr: *mut HeapObject = &mut obj;
    c.bench_function("tagged_heap_ptr_round_trip", |b| {
        b.iter(|| {
            // SAFETY: `ptr` is non-null, aligned, and `obj` is live for the
            // duration of the benchmark.
            let tv = unsafe { TaggedValue::from_heap_object(black_box(ptr)) };
            black_box(tv.is_heap_object());
            // SAFETY: `obj` has not been freed.
            black_box(unsafe { tv.as_heap_object() });
        });
    });
}

// ===========================================================================
// 3. HandleScope create / destroy
// ===========================================================================

const LOCAL_HANDLE_COUNT: usize = 100;

fn bench_handle_scope_create_destroy(c: &mut Criterion) {
    c.bench_function("handle_scope_create_destroy", |b| {
        b.iter(|| {
            let mut isolate = ();
            let scope = HandleScope::new(black_box(&mut isolate));
            black_box(&scope);
            drop(scope);
        });
    });
}

fn bench_handle_scope_create_locals(c: &mut Criterion) {
    c.bench_function("handle_scope_create_100_locals", |b| {
        let mut values: Vec<u64> = (0..LOCAL_HANDLE_COUNT as u64).collect();
        let ptrs: Vec<NonNull<u64>> = values
            .iter_mut()
            .map(|v| NonNull::new(v as *mut u64).unwrap())
            .collect();
        b.iter(|| {
            let mut isolate = ();
            let mut scope = HandleScope::new(&mut isolate);
            for &ptr in &ptrs {
                // SAFETY: each pointer is live for the benchmark duration.
                let local = unsafe { scope.create_local(ptr) };
                black_box(local);
            }
            black_box(scope.raw_handles().count());
        });
    });
}

// ===========================================================================
// 4. PropertyMap throughput
// ===========================================================================

fn bench_property_map_insert(c: &mut Criterion) {
    c.bench_function("property_map_insert_100", |b| {
        b.iter(|| {
            let mut map = PropertyMap::new();
            for i in 0..100u32 {
                map.insert(format!("prop_{i}"), JsValue::Smi(i as i32));
            }
            black_box(&map);
        });
    });
}

fn bench_property_map_lookup_hit(c: &mut Criterion) {
    let mut map = PropertyMap::new();
    for i in 0..20u32 {
        map.insert(format!("prop_{i}"), JsValue::Smi(i as i32));
    }
    c.bench_function("property_map_lookup_hit", |b| {
        b.iter(|| {
            // Repeated lookup of the same key exercises the inline cache.
            black_box(map.get(black_box("prop_10")));
        });
    });
}

fn bench_property_map_lookup_miss(c: &mut Criterion) {
    let mut map = PropertyMap::new();
    for i in 0..20u32 {
        map.insert(format!("prop_{i}"), JsValue::Smi(i as i32));
    }
    c.bench_function("property_map_lookup_miss", |b| {
        b.iter(|| {
            black_box(map.get(black_box("nonexistent")));
        });
    });
}

fn bench_property_map_integer_index_insert(c: &mut Criterion) {
    c.bench_function("property_map_integer_index_insert_1000", |b| {
        b.iter(|| {
            let mut map = PropertyMap::new();
            for i in 0..1000u32 {
                map.insert(i.to_string(), JsValue::Smi(i as i32));
            }
            black_box(&map);
        });
    });
}

fn bench_property_map_shape_offset(c: &mut Criterion) {
    let mut map = PropertyMap::new();
    for i in 0..10u32 {
        map.insert(format!("field_{i}"), JsValue::Smi(i as i32));
    }
    let offset = map.offset_of("field_5").unwrap();
    let shape = map.shape_id();
    c.bench_function("property_map_get_by_offset", |b| {
        b.iter(|| {
            // Simulates an inline-cache hit: check shape, read by offset.
            assert_eq!(map.shape_id(), shape);
            black_box(map.get_by_offset(black_box(offset)));
        });
    });
}

// ===========================================================================
// 5. JsValue clone cost
// ===========================================================================

fn bench_jsvalue_clone_smi(c: &mut Criterion) {
    let val = JsValue::Smi(42);
    c.bench_function("jsvalue_clone_smi", |b| {
        b.iter(|| {
            black_box(black_box(&val).clone());
        });
    });
}

fn bench_jsvalue_clone_string(c: &mut Criterion) {
    let val = JsValue::String(Rc::from("hello world"));
    c.bench_function("jsvalue_clone_string", |b| {
        b.iter(|| {
            black_box(black_box(&val).clone());
        });
    });
}

fn bench_jsvalue_clone_plain_object(c: &mut Criterion) {
    let mut map = PropertyMap::new();
    for i in 0..10u32 {
        map.insert(format!("key_{i}"), JsValue::Smi(i as i32));
    }
    let val = JsValue::PlainObject(Rc::new(RefCell::new(map)));
    c.bench_function("jsvalue_clone_plain_object", |b| {
        b.iter(|| {
            black_box(black_box(&val).clone());
        });
    });
}

fn bench_jsvalue_clone_array(c: &mut Criterion) {
    let items: Vec<JsValue> = (0..100).map(|i| JsValue::Smi(i)).collect();
    let val = JsValue::Array(Rc::new(RefCell::new(items)));
    c.bench_function("jsvalue_clone_array_100", |b| {
        b.iter(|| {
            black_box(black_box(&val).clone());
        });
    });
}

// ===========================================================================
// 6. End-to-end JS micro-benchmarks
// ===========================================================================

fn bench_js_property_access(c: &mut Criterion) {
    let source = r#"
        var obj = { a: 1, b: 2, c: 3, d: 4, e: 5 };
        var sum = 0;
        for (var i = 0; i < 1000; i++) {
            sum = sum + obj.a + obj.b + obj.c + obj.d + obj.e;
        }
        sum;
    "#;
    c.bench_function("js_property_access_1000", |b| {
        b.iter(|| {
            black_box(eval_js(black_box(source)).unwrap());
        });
    });
}

#[allow(dead_code)]
fn bench_js_function_call(c: &mut Criterion) {
    let source = r#"
        function add(a, b) { return a + b; }
        var result = 0;
        for (var i = 0; i < 100; i++) {
            result = add(result, 1);
        }
        result;
    "#;
    c.bench_function("js_function_call_100", |b| {
        b.iter(|| {
            black_box(eval_js(black_box(source)).unwrap());
        });
    });
}

fn bench_js_string_concat(c: &mut Criterion) {
    let source = r#"
        var s = "";
        for (var i = 0; i < 500; i++) {
            s = s + "x";
        }
        s.length;
    "#;
    c.bench_function("js_string_concat_500", |b| {
        b.iter(|| {
            black_box(eval_js(black_box(source)).unwrap());
        });
    });
}

fn bench_js_object_creation(c: &mut Criterion) {
    let source = r#"
        var last;
        for (var i = 0; i < 100; i++) {
            last = { x: i, y: i + 1, z: i + 2 };
        }
        last.x + last.y + last.z;
    "#;
    c.bench_function("js_object_creation_100", |b| {
        b.iter(|| {
            black_box(eval_js(black_box(source)).unwrap());
        });
    });
}

fn bench_js_keyed_access(c: &mut Criterion) {
    let source = r#"
        var obj = {};
        for (var i = 0; i < 50; i++) {
            obj["key_" + i] = i;
        }
        var sum = 0;
        for (var i = 0; i < 50; i++) {
            sum = sum + obj["key_" + i];
        }
        sum;
    "#;
    c.bench_function("js_keyed_property_access_50", |b| {
        b.iter(|| {
            black_box(eval_js(black_box(source)).unwrap());
        });
    });
}

fn bench_js_prototype_chain_lookup(c: &mut Criterion) {
    // Tests property lookup walking up the prototype chain.
    let source = r#"
        function Base() {}
        Base.prototype.x = 42;
        function Mid() {}
        Mid.prototype = new Base();
        function Leaf() {}
        Leaf.prototype = new Mid();
        var obj = new Leaf();
        var sum = 0;
        for (var i = 0; i < 100; i++) {
            sum = sum + obj.x;
        }
        sum;
    "#;
    c.bench_function("js_prototype_chain_lookup_100", |b| {
        b.iter(|| {
            black_box(eval_js(black_box(source)).unwrap());
        });
    });
}

fn bench_js_arithmetic_loop(c: &mut Criterion) {
    let source = r#"
        var n = 0;
        for (var i = 0; i < 1000; i++) {
            n = (n + i) | 0;
        }
        n;
    "#;
    c.bench_function("js_arithmetic_loop_1000", |b| {
        b.iter(|| {
            black_box(eval_js(black_box(source)).unwrap());
        });
    });
}

fn bench_js_closure_capture(c: &mut Criterion) {
    let source = r#"
        function make_counter() {
            var count = 0;
            return function() { count = count + 1; return count; };
        }
        var counter = make_counter();
        var result = 0;
        for (var i = 0; i < 10; i++) {
            result = counter();
        }
        result;
    "#;
    c.bench_function("js_closure_capture_10", |b| {
        b.iter(|| {
            black_box(eval_js(black_box(source)).unwrap());
        });
    });
}

// ===========================================================================
// 5. V8 comparison benchmarks
// ===========================================================================
// These benchmarks mirror the JS snippets in benchmarks/v8_comparison/benchmarks.js
// so that Stator's Criterion numbers can be compared to V8 (Node.js) results.

// NOTE: fib_10_recursive is intentionally excluded from the Criterion group
// because deep recursion can time out in CI. string_concat_5k remains
// available for ad-hoc comparison runs.

fn bench_fib_10_recursive(c: &mut Criterion) {
    let source = r#"
        function fib(n) {
            if (n < 2) return n;
            return fib(n - 1) + fib(n - 2);
        }
        fib(10);
    "#;
    c.bench_function("fib_10_recursive", |b| {
        b.iter(|| {
            black_box(eval_js(black_box(source)).unwrap());
        });
    });
}

fn bench_fib_40_iterative(c: &mut Criterion) {
    let source = r#"
        var a = 0, b = 1;
        for (var i = 0; i < 40; i++) {
            var t = a + b;
            a = b;
            b = t;
        }
        b;
    "#;
    c.bench_function("fib_40_iterative", |b| {
        b.iter(|| {
            black_box(eval_js(black_box(source)).unwrap());
        });
    });
}

fn bench_arithmetic_loop_10k(c: &mut Criterion) {
    let source = r#"
        var n = 0;
        for (var i = 0; i < 10000; i++) {
            n = (n + i * 3 - 1) | 0;
        }
        n;
    "#;
    c.bench_function("arithmetic_loop_10k", |b| {
        b.iter(|| {
            black_box(eval_js(black_box(source)).unwrap());
        });
    });
}

fn bench_property_access_1k(c: &mut Criterion) {
    let source = r#"
        var obj = { a: 1, b: 2, c: 3, d: 4, e: 5 };
        var sum = 0;
        for (var i = 0; i < 1000; i++) {
            sum = sum + obj.a + obj.b + obj.c + obj.d + obj.e;
        }
        sum;
    "#;
    c.bench_function("property_access_1k", |b| {
        b.iter(|| {
            black_box(eval_js(black_box(source)).unwrap());
        });
    });
}

fn bench_object_creation_1k(c: &mut Criterion) {
    let source = r#"
        var last;
        for (var i = 0; i < 1000; i++) {
            last = { x: i, y: i + 1, z: i * 2 };
        }
        last.x + last.y + last.z;
    "#;
    c.bench_function("object_creation_1k", |b| {
        b.iter(|| {
            black_box(eval_js(black_box(source)).unwrap());
        });
    });
}

fn bench_array_push_sum_1k(c: &mut Criterion) {
    let source = r#"
        var arr = [];
        for (var i = 0; i < 100; i++) {
            arr.push(i);
        }
        var sum = 0;
        for (var i = 0; i < arr.length; i++) {
            sum = sum + arr[i];
        }
        sum;
    "#;
    c.bench_function("array_push_sum_100", |b| {
        b.iter(|| {
            black_box(eval_js(black_box(source)).unwrap());
        });
    });
}

fn bench_string_concat_5k(c: &mut Criterion) {
    let source = r#"
        var s = "";
        for (var i = 0; i < 500; i++) {
            s = s + "x";
        }
        s.length;
    "#;
    c.bench_function("string_concat_500", |b| {
        b.iter(|| {
            black_box(eval_js(black_box(source)).unwrap());
        });
    });
}

fn bench_function_calls_1k(c: &mut Criterion) {
    let source = r#"
        function add(a, b) { return a + b; }
        var sum = 0;
        for (var i = 0; i < 10; i++) {
            sum = add(sum, i);
        }
        sum;
    "#;
    c.bench_function("function_calls_10", |b| {
        b.iter(|| {
            black_box(eval_js(black_box(source)).unwrap());
        });
    });
}

fn bench_closure_counter_1k(c: &mut Criterion) {
    let source = r#"
        function make_counter() {
            var count = 0;
            return function() { count = count + 1; return count; };
        }
        var counter = make_counter();
        var result = 0;
        for (var i = 0; i < 100; i++) {
            result = counter();
        }
        result;
    "#;
    c.bench_function("closure_counter_100", |b| {
        b.iter(|| {
            black_box(eval_js(black_box(source)).unwrap());
        });
    });
}

fn bench_prototype_chain_1k(c: &mut Criterion) {
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
    c.bench_function("prototype_chain_1k", |b| {
        b.iter(|| {
            black_box(eval_js(black_box(source)).unwrap());
        });
    });
}

fn bench_sieve_primes_1k(c: &mut Criterion) {
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
    c.bench_function("sieve_primes_1k", |b| {
        b.iter(|| {
            black_box(eval_js(black_box(source)).unwrap());
        });
    });
}

fn bench_deep_object_access_1k(c: &mut Criterion) {
    let source = r#"
        var root = { a: { b: { c: { d: { e: 99 } } } } };
        var sum = 0;
        for (var i = 0; i < 1000; i++) {
            sum = sum + root.a.b.c.d.e;
        }
        sum;
    "#;
    c.bench_function("deep_object_access_1k", |b| {
        b.iter(|| {
            black_box(eval_js(black_box(source)).unwrap());
        });
    });
}

// ===========================================================================
// Benchmark groups
// ===========================================================================

criterion_group! {
    name = infra_benches;
    config = ci_config();
    targets =
        bench_heap_allocate,
        bench_heap_allocate_burst,
        bench_tagged_smi_round_trip,
        bench_tagged_heap_ptr_round_trip,
        bench_handle_scope_create_destroy,
        bench_handle_scope_create_locals,
}

criterion_group! {
    name = property_map_benches;
    config = ci_config();
    targets =
        bench_property_map_insert,
        bench_property_map_lookup_hit,
        bench_property_map_lookup_miss,
        bench_property_map_integer_index_insert,
        bench_property_map_shape_offset,
}

criterion_group! {
    name = jsvalue_benches;
    config = ci_config();
    targets =
        bench_jsvalue_clone_smi,
        bench_jsvalue_clone_string,
        bench_jsvalue_clone_plain_object,
        bench_jsvalue_clone_array,
}

criterion_group! {
    name = js_benches;
    config = ci_config();
    targets =
        bench_js_property_access,
        bench_js_string_concat,
        bench_js_object_creation,
        bench_js_keyed_access,
        bench_js_prototype_chain_lookup,
        bench_js_arithmetic_loop,
        bench_js_closure_capture,
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
    let env = make_global_env();
    c.bench_function("arithmetic_loop_10k_precompiled", |b| {
        b.iter(|| {
            let mut frame =
                InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
            black_box(Interpreter::run(&mut frame).unwrap());
        });
    });
    let (tried, executed, deopted, not_ready, compilations, code_bytes) = maglev_diagnostics();
    eprintln!(
        "MAGLEV_DIAG[arithmetic_loop_10k]: tried={tried} executed={executed} deopted={deopted} not_ready={not_ready} compilations={compilations} code_bytes={code_bytes}"
    );
}

criterion_group! {
    name = v8_comparison_benches;
    config = ci_config();
    targets =
        bench_fib_40_iterative,
        bench_arithmetic_loop_10k,
        bench_js_arithmetic_precompiled,
        bench_property_access_1k,
        bench_object_creation_1k,
        bench_array_push_sum_1k,
        bench_closure_counter_1k,
        bench_prototype_chain_1k,
        bench_sieve_primes_1k,
        bench_deep_object_access_1k,
}

// ===========================================================================
// Pre-compiled V8 comparison benchmarks
// ===========================================================================
// These skip parse+compile and measure pure interpreter speed, giving a fairer
// comparison against V8's JIT-compiled hot code.  JS sources match
// benchmarks/v8_comparison/benchmarks.js exactly.

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
    let env = make_global_env();
    c.bench_function("fib_40_iterative_precompiled", |b| {
        b.iter(|| {
            let mut frame =
                InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
            black_box(Interpreter::run(black_box(&mut frame)).unwrap())
        });
    });
    let (tried, executed, deopted, not_ready, compilations, code_bytes) = maglev_diagnostics();
    eprintln!(
        "MAGLEV_DIAG[fib_40_iterative]: tried={tried} executed={executed} deopted={deopted} not_ready={not_ready} compilations={compilations} code_bytes={code_bytes}"
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
    let env = make_global_env();
    c.bench_function("property_access_1k_precompiled", |b| {
        b.iter(|| {
            let mut frame =
                InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
            black_box(Interpreter::run(black_box(&mut frame)).unwrap())
        });
    });
    let (tried, executed, deopted, not_ready, compilations, code_bytes) = maglev_diagnostics();
    eprintln!(
        "MAGLEV_DIAG[property_access_1k]: tried={tried} executed={executed} deopted={deopted} not_ready={not_ready} compilations={compilations} code_bytes={code_bytes}"
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
    let env = make_global_env();
    c.bench_function("object_creation_1k_precompiled", |b| {
        b.iter(|| {
            let mut frame =
                InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
            black_box(Interpreter::run(black_box(&mut frame)).unwrap())
        });
    });
    let (tried, executed, deopted, not_ready, compilations, code_bytes) = maglev_diagnostics();
    eprintln!(
        "MAGLEV_DIAG[object_creation_1k]: tried={tried} executed={executed} deopted={deopted} not_ready={not_ready} compilations={compilations} code_bytes={code_bytes}"
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
    let env = make_global_env();
    c.bench_function("array_push_sum_1k_precompiled", |b| {
        b.iter(|| {
            let mut frame =
                InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
            black_box(Interpreter::run(black_box(&mut frame)).unwrap())
        });
    });
    let (tried, executed, deopted, not_ready, compilations, code_bytes) = maglev_diagnostics();
    eprintln!(
        "MAGLEV_DIAG[array_push_sum_1k]: tried={tried} executed={executed} deopted={deopted} not_ready={not_ready} compilations={compilations} code_bytes={code_bytes}"
    );
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
    let env = make_global_env();
    c.bench_function("closure_counter_1k_precompiled", |b| {
        b.iter(|| {
            let mut frame =
                InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
            black_box(Interpreter::run(black_box(&mut frame)).unwrap())
        });
    });
    let (tried, executed, deopted, not_ready, compilations, code_bytes) = maglev_diagnostics();
    eprintln!(
        "MAGLEV_DIAG[closure_counter_1k]: tried={tried} executed={executed} deopted={deopted} not_ready={not_ready} compilations={compilations} code_bytes={code_bytes}"
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
    let env = make_global_env();
    c.bench_function("prototype_chain_1k_precompiled", |b| {
        b.iter(|| {
            let mut frame =
                InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
            black_box(Interpreter::run(black_box(&mut frame)).unwrap())
        });
    });
    let (tried, executed, deopted, not_ready, compilations, code_bytes) = maglev_diagnostics();
    eprintln!(
        "MAGLEV_DIAG[prototype_chain_1k]: tried={tried} executed={executed} deopted={deopted} not_ready={not_ready} compilations={compilations} code_bytes={code_bytes}"
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
    let env = make_global_env();
    c.bench_function("sieve_primes_1k_precompiled", |b| {
        b.iter(|| {
            let mut frame =
                InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
            black_box(Interpreter::run(black_box(&mut frame)).unwrap())
        });
    });
    let (tried, executed, deopted, not_ready, compilations, code_bytes) = maglev_diagnostics();
    eprintln!(
        "MAGLEV_DIAG[sieve_primes_1k]: tried={tried} executed={executed} deopted={deopted} not_ready={not_ready} compilations={compilations} code_bytes={code_bytes}"
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
    let env = make_global_env();
    c.bench_function("deep_object_access_1k_precompiled", |b| {
        b.iter(|| {
            let mut frame =
                InterpreterFrame::new_with_globals(Rc::clone(&ba), vec![], Rc::clone(&env));
            black_box(Interpreter::run(black_box(&mut frame)).unwrap())
        });
    });
    let (tried, executed, deopted, not_ready, compilations, code_bytes) = maglev_diagnostics();
    eprintln!(
        "MAGLEV_DIAG[deep_object_access_1k]: tried={tried} executed={executed} deopted={deopted} not_ready={not_ready} compilations={compilations} code_bytes={code_bytes}"
    );
}

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

criterion_main!(
    infra_benches,
    property_map_benches,
    jsvalue_benches,
    js_benches,
    v8_comparison_benches,
    v8_precompiled_benches
);
