# Stator vs V8 Performance Comparison

Micro-benchmarks comparing Stator's bytecode interpreter to V8 (via Node.js).

## Benchmarks

| # | Name | Description |
|---|------|-------------|
| 1 | `fib_30_recursive` | Recursive Fibonacci(30) — call overhead & recursion |
| 2 | `fib_40_iterative` | Iterative Fibonacci(40) — tight loop performance |
| 3 | `arithmetic_loop_100k` | 100k arithmetic ops with bitwise OR |
| 4 | `property_access_10k` | Named property reads on a 5-key object |
| 5 | `object_creation_10k` | Create 10k objects with 3 properties each |
| 6 | `array_push_sum_10k` | Push 10k elements, then sum them |
| 7 | `string_concat_5k` | Concatenate 5k single-char strings |
| 8 | `function_calls_10k` | 10k calls to a simple `add(a,b)` function |
| 9 | `closure_counter_10k` | 10k calls to a closure-captured counter |
| 10 | `prototype_chain_10k` | 10k lookups through 3-level prototype chain |
| 11 | `sieve_primes_10k` | Sieve of Eratosthenes to 10,000 |
| 12 | `deep_object_access_5k` | 5k reads through 5-level nested object |

## Running

### Quick comparison (requires Node.js + Rust toolchain)

```bash
pwsh benchmarks/v8_comparison/compare.ps1
```

### V8 only

```bash
node benchmarks/v8_comparison/benchmarks.js
```

### Stator only (via Criterion)

```bash
cargo bench --package stator_js --bench engine_benchmarks
```

## How it works

The same JavaScript snippets are run on both engines:
- **V8**: Executed via `node` with `process.hrtime.bigint()` timing
- **Stator**: Executed via Criterion's `bench_function` with `eval_js()` helper

Results are compared by median execution time. The ratio column shows
`Stator/V8` — values below 1.0 mean Stator is faster.

## Notes

- V8 includes a multi-tier JIT compiler (Ignition → Sparkplug → Maglev → TurboFan)
- Stator currently runs as a bytecode interpreter with inline caches
- For cold-start / single-execution scenarios, Stator can outperform V8 due to
  lower startup overhead and no JIT compilation cost
- For hot loops, V8's JIT will typically win by 10-100x on compute-heavy benchmarks
