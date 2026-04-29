/**
 * chromium_bench — Benchmark Stator through the C FFI layer.
 *
 * Proves that the engine speedup over V8 holds when called through the
 * same stator.h API that Chromium embedders would use.  Each benchmark
 * compiles a JS snippet once, then times repeated stator_script_run()
 * calls.  Output is JSON-compatible with benchmarks/v8_comparison/benchmarks.js
 * so results can be compared directly.
 *
 * Usage:
 *   chromium_bench              # default: 200 iterations, human-readable
 *   chromium_bench --json       # JSON output (same format as benchmarks.js)
 *   chromium_bench -n 500       # custom iteration count
 *   chromium_bench --filter sieve_primes_1k
 */

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "stator.h"

// ─── Benchmark snippets ──────────────────────────────────────────────────
// Identical to benchmarks/v8_comparison/benchmarks.js and the Criterion
// suite in crates/stator_jse/benches/engine_benchmarks.rs.

struct BenchSpec {
    const char *name;
    const char *source;
    int iterations;
};

static const BenchSpec benchmarks[] = {
    {"fib_40_iterative",
     "var a = 0, b = 1;\n"
     "for (var i = 0; i < 40; i++) { var t = a + b; a = b; b = t; }\n"
     "b;\n",
     200},

    {"arithmetic_loop_10k",
     "var n = 0;\n"
     "for (var i = 0; i < 10000; i++) { n = (n + i * 3 - 1) | 0; }\n"
     "n;\n",
     200},

    {"property_access_1k",
     "var obj = { a: 1, b: 2, c: 3, d: 4, e: 5 };\n"
     "var sum = 0;\n"
     "for (var i = 0; i < 1000; i++) {\n"
     "  sum = sum + obj.a + obj.b + obj.c + obj.d + obj.e;\n"
     "}\n"
     "sum;\n",
     200},

    {"object_creation_1k",
     "var last;\n"
     "for (var i = 0; i < 1000; i++) {\n"
     "  last = { x: i, y: i + 1, z: i * 2 };\n"
     "}\n"
     "last.x + last.y + last.z;\n",
     200},

    {"array_push_sum_1k",
     "var arr = [];\n"
     "for (var i = 0; i < 1000; i++) { arr.push(i); }\n"
     "var sum = 0;\n"
     "for (var i = 0; i < arr.length; i++) { sum = sum + arr[i]; }\n"
     "sum;\n",
     200},

    {"closure_counter_1k",
     "var count = 0;\n"
     "var counter = function() { return count++; };\n"
     "var sum = 0;\n"
     "for (var i = 0; i < 1000; i++) { sum = sum + counter(); }\n"
     "sum;\n",
     200},

    {"prototype_chain_1k",
     "function Base() {}\n"
     "Base.prototype.x = 42;\n"
     "function Mid() {}\n"
     "Mid.prototype = new Base();\n"
     "function Leaf() {}\n"
     "Leaf.prototype = new Mid();\n"
     "var obj = new Leaf();\n"
     "var sum = 0;\n"
     "for (var i = 0; i < 1000; i++) { sum = sum + obj.x; }\n"
     "sum;\n",
     200},

    {"sieve_primes_1k",
     "var n = 1000;\n"
     "var sieve = [];\n"
     "for (var i = 0; i <= n; i++) sieve[i] = true;\n"
     "sieve[0] = false; sieve[1] = false;\n"
     "for (var i = 2; i * i <= n; i++) {\n"
     "  if (sieve[i]) {\n"
     "    for (var j = i * i; j <= n; j = j + i) { sieve[j] = false; }\n"
     "  }\n"
     "}\n"
     "var count = 0;\n"
     "for (var i = 0; i <= n; i++) { if (sieve[i]) count = count + 1; }\n"
     "count;\n",
     200},

    {"deep_object_access_1k",
     "var root = { a: { b: { c: { d: { e: 99 } } } } };\n"
     "var sum = 0;\n"
     "for (var i = 0; i < 1000; i++) { sum = sum + root.a.b.c.d.e; }\n"
     "sum;\n",
     200},
};

static constexpr int NUM_BENCHMARKS =
    static_cast<int>(sizeof(benchmarks) / sizeof(benchmarks[0]));

// ─── Measurement ─────────────────────────────────────────────────────────

struct BenchResult {
    const char *name;
    int64_t median_ns;
    int64_t mean_ns;
    int64_t min_ns;
    int iterations;
};

static BenchResult run_bench(StatorIsolate *isolate, const BenchSpec &spec) {
    std::vector<int64_t> times;
    times.reserve(spec.iterations);

    StatorContext *ctx = stator_context_new(isolate);
    StatorScript *script =
        stator_script_compile(ctx, spec.source, std::strlen(spec.source));
    if (stator_script_get_error(script)) {
        std::fprintf(stderr, "ERROR compiling '%s': %s\n", spec.name,
                     stator_script_get_error(script));
        stator_script_free(script);
        stator_context_destroy(ctx);
        return {spec.name, -1, -1, -1, 0};
    }

    auto run_once = [&]() -> bool {
        StatorValue *val = stator_script_run(script, ctx);
        if (!val) return false;
        stator_value_destroy(val);
        return true;
    };

    // Warm the interpreter ICs and give background JIT compilation time to
    // complete, matching the Rust precompiled benchmark warmup pattern.
    for (int w = 0; w < 100; w++) {
        if (!run_once()) {
            std::fprintf(stderr, "ERROR running warmup for '%s'\n", spec.name);
            stator_script_free(script);
            stator_context_destroy(ctx);
            return {spec.name, -1, -1, -1, 0};
        }
    }
    std::this_thread::sleep_for(std::chrono::seconds(2));
    for (int w = 0; w < 100; w++) {
        if (!run_once()) {
            std::fprintf(stderr, "ERROR running warmup for '%s'\n", spec.name);
            stator_script_free(script);
            stator_context_destroy(ctx);
            return {spec.name, -1, -1, -1, 0};
        }
    }

    // Timed iterations: the script is already compiled and warmed, so this
    // measures the Chromium embedder path for repeated script execution.
    for (int i = 0; i < spec.iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        bool ok = run_once();
        auto end = std::chrono::high_resolution_clock::now();

        if (!ok) {
            std::fprintf(stderr, "ERROR running timed iteration for '%s'\n",
                         spec.name);
            stator_script_free(script);
            stator_context_destroy(ctx);
            return {spec.name, -1, -1, -1, 0};
        }

        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        times.push_back(ns);
    }

    stator_script_free(script);
    stator_context_destroy(ctx);

    std::sort(times.begin(), times.end());
    int64_t median = times[times.size() / 2];
    int64_t sum = 0;
    for (auto t : times) sum += t;
    int64_t mean = sum / static_cast<int64_t>(times.size());
    int64_t min_val = times[0];

    return {spec.name, median, mean, min_val, spec.iterations};
}

// ─── Output ──────────────────────────────────────────────────────────────

static void print_json(const std::vector<BenchResult> &results) {
    std::printf("STATOR_FFI_BENCHMARK_RESULTS_JSON=[");
    for (size_t i = 0; i < results.size(); i++) {
        const auto &r = results[i];
        std::printf(
            "{\"name\":\"%s\",\"median_ns\":%" PRId64
            ",\"mean_ns\":%" PRId64 ",\"min_ns\":%" PRId64
            ",\"iterations\":%d}",
            r.name, r.median_ns, r.mean_ns, r.min_ns, r.iterations);
        if (i + 1 < results.size()) std::printf(",");
    }
    std::printf("]\n");
}

static void print_table(const std::vector<BenchResult> &results) {
    std::printf("\n=== Stator FFI Benchmarks (C++ embedder path) ===\n\n");
    std::printf("%-30s %15s %15s %15s\n", "Benchmark", "Median (us)",
                "Mean (us)", "Min (us)");
    std::printf("----------------------------------------------------------------------"
                "---------\n");
    for (const auto &r : results) {
        if (r.median_ns < 0) {
            std::printf("%-30s %15s\n", r.name, "(compile error)");
        } else {
            std::printf("%-30s %14.1f %14.1f %14.1f\n", r.name,
                        r.median_ns / 1000.0, r.mean_ns / 1000.0,
                        r.min_ns / 1000.0);
        }
    }
    std::printf("\n");
}

// ─── Main ────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    bool json_mode = false;
    int custom_iters = 0;
    const char *filter = nullptr;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--json") == 0) {
            json_mode = true;
        } else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            custom_iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
            filter = argv[++i];
        }
    }

    StatorIsolate *isolate = stator_isolate_create();
    if (!isolate) {
        std::fprintf(stderr, "Failed to create Stator isolate\n");
        return 1;
    }

    std::vector<BenchResult> results;
    results.reserve(NUM_BENCHMARKS);

    for (int b = 0; b < NUM_BENCHMARKS; b++) {
        BenchSpec spec = benchmarks[b];
        if (filter && std::strcmp(spec.name, filter) != 0) {
            continue;
        }
        if (custom_iters > 0) spec.iterations = custom_iters;

        if (!json_mode) {
            std::fprintf(stderr, "Running %-30s (%d iterations)...\n",
                         spec.name, spec.iterations);
        }

        results.push_back(run_bench(isolate, spec));
    }

    stator_isolate_destroy(isolate);

    if (results.empty()) {
        std::fprintf(stderr, "No benchmark matched filter '%s'\n",
                     filter ? filter : "");
        return 1;
    }

    if (json_mode) {
        print_json(results);
    }
    print_table(results);

    return 0;
}
