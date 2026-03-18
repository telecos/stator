// V8 (Node.js) comparison benchmarks
// Each benchmark is a self-contained JS snippet that returns a numeric result.
// The same snippets are embedded in Stator's Criterion benchmarks.
//
// Usage: node benchmarks.js

"use strict";

function measure(name, fn, iterations) {
  // Warmup
  for (let i = 0; i < 3; i++) fn();

  const times = [];
  for (let i = 0; i < iterations; i++) {
    const start = process.hrtime.bigint();
    fn();
    const end = process.hrtime.bigint();
    times.push(Number(end - start));
  }

  times.sort((a, b) => a - b);
  const median = times[Math.floor(times.length / 2)];
  const mean = times.reduce((a, b) => a + b, 0) / times.length;
  const min = times[0];

  return { name, median_ns: median, mean_ns: Math.round(mean), min_ns: min, iterations };
}

const results = [];

// ─── 1. Fibonacci (recursive) ────────────────────────────────────────────
results.push(measure("fib_30_recursive", () => {
  function fib(n) {
    if (n < 2) return n;
    return fib(n - 1) + fib(n - 2);
  }
  return fib(30);
}, 50));

// ─── 2. Fibonacci (iterative) ────────────────────────────────────────────
results.push(measure("fib_40_iterative", () => {
  var a = 0, b = 1;
  for (var i = 0; i < 40; i++) {
    var t = a + b;
    a = b;
    b = t;
  }
  return b;
}, 200));

// ─── 3. Arithmetic loop ─────────────────────────────────────────────────
results.push(measure("arithmetic_loop_100k", () => {
  var n = 0;
  for (var i = 0; i < 100000; i++) {
    n = (n + i * 3 - 1) | 0;
  }
  return n;
}, 100));

// ─── 4. Property access ─────────────────────────────────────────────────
results.push(measure("property_access_10k", () => {
  var obj = { a: 1, b: 2, c: 3, d: 4, e: 5 };
  var sum = 0;
  for (var i = 0; i < 10000; i++) {
    sum = sum + obj.a + obj.b + obj.c + obj.d + obj.e;
  }
  return sum;
}, 100));

// ─── 5. Object creation ─────────────────────────────────────────────────
results.push(measure("object_creation_10k", () => {
  var last;
  for (var i = 0; i < 10000; i++) {
    last = { x: i, y: i + 1, z: i * 2 };
  }
  return last.x + last.y + last.z;
}, 50));

// ─── 6. Array push + sum ────────────────────────────────────────────────
results.push(measure("array_push_sum_10k", () => {
  var arr = [];
  for (var i = 0; i < 10000; i++) {
    arr.push(i);
  }
  var sum = 0;
  for (var i = 0; i < arr.length; i++) {
    sum = sum + arr[i];
  }
  return sum;
}, 50));

// ─── 7. String concatenation ────────────────────────────────────────────
results.push(measure("string_concat_5k", () => {
  var s = "";
  for (var i = 0; i < 5000; i++) {
    s = s + "x";
  }
  return s.length;
}, 30));

// ─── 8. Function calls (nested) ─────────────────────────────────────────
results.push(measure("function_calls_10k", () => {
  function add(a, b) { return a + b; }
  var sum = 0;
  for (var i = 0; i < 10000; i++) {
    sum = add(sum, i);
  }
  return sum;
}, 100));

// ─── 9. Closure capture ─────────────────────────────────────────────────
results.push(measure("closure_counter_10k", () => {
  function make_counter() {
    var count = 0;
    return function() { count = count + 1; return count; };
  }
  var counter = make_counter();
  var result = 0;
  for (var i = 0; i < 10000; i++) {
    result = counter();
  }
  return result;
}, 100));

// ─── 10. Prototype chain lookup ──────────────────────────────────────────
results.push(measure("prototype_chain_10k", () => {
  function Base() {}
  Base.prototype.x = 42;
  function Mid() {}
  Mid.prototype = new Base();
  function Leaf() {}
  Leaf.prototype = new Mid();
  var obj = new Leaf();
  var sum = 0;
  for (var i = 0; i < 10000; i++) {
    sum = sum + obj.x;
  }
  return sum;
}, 100));

// ─── 11. Sieve of Eratosthenes ──────────────────────────────────────────
results.push(measure("sieve_primes_10k", () => {
  var n = 10000;
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
  return count;
}, 50));

// ─── 12. JSON-like deep object access ────────────────────────────────────
results.push(measure("deep_object_access_5k", () => {
  var root = { a: { b: { c: { d: { e: 99 } } } } };
  var sum = 0;
  for (var i = 0; i < 5000; i++) {
    sum = sum + root.a.b.c.d.e;
  }
  return sum;
}, 100));

// ─── Output ──────────────────────────────────────────────────────────────
console.log("V8_BENCHMARK_RESULTS_JSON=" + JSON.stringify(results));

// Human-readable table
console.log("\n=== V8 (Node.js) Benchmark Results ===\n");
console.log("Benchmark".padEnd(30) + "Median (µs)".padStart(15) + "Mean (µs)".padStart(15) + "Min (µs)".padStart(15));
console.log("-".repeat(75));
for (const r of results) {
  console.log(
    r.name.padEnd(30) +
    (r.median_ns / 1000).toFixed(1).padStart(15) +
    (r.mean_ns / 1000).toFixed(1).padStart(15) +
    (r.min_ns / 1000).toFixed(1).padStart(15)
  );
}
