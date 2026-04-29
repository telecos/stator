#!/usr/bin/env bash
# compare.sh — Run both V8 (Node.js) and Stator (via C FFI) benchmarks
# and print a side-by-side comparison table.
#
# Prerequisites:
#   - Node.js on PATH
#   - chromium_bench built (see examples/chromium_bench/README)
#
# Usage:
#   ./compare.sh                     # auto-detect paths
#   ./compare.sh /path/to/chromium_bench

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CHROMIUM_BENCH="${1:-$REPO_ROOT/examples/chromium_bench/build/chromium_bench}"

if [ ! -x "$CHROMIUM_BENCH" ]; then
    echo "ERROR: chromium_bench not found at $CHROMIUM_BENCH"
    echo "Build it first:"
    echo "  cd $REPO_ROOT && cargo build --release"
    echo "  cd examples/chromium_bench && cmake -B build && cmake --build build"
    exit 1
fi

if ! command -v node &>/dev/null; then
    echo "ERROR: node not found on PATH"
    exit 1
fi

echo "=== Running V8 (Node.js) benchmarks ==="
v8_output=$(node "$REPO_ROOT/benchmarks/v8_comparison/benchmarks.js" 2>&1)
v8_json=$(echo "$v8_output" | grep '^V8_BENCHMARK_RESULTS_JSON=' | sed 's/^V8_BENCHMARK_RESULTS_JSON=//')

shared=(
    "fib_40_iterative"
    "arithmetic_loop_10k"
    "property_access_1k"
    "object_creation_1k"
    "array_push_sum_1k"
    "closure_counter_1k"
    "prototype_chain_1k"
    "sieve_primes_1k"
    "deep_object_access_1k"
)

echo "=== Running Stator FFI benchmarks ==="
stator_json="["
first=1
for bench in "${shared[@]}"; do
    echo "--- $bench ---"
    stator_output=$("$CHROMIUM_BENCH" --json --filter "$bench" 2>&1)
    bench_json=$(echo "$stator_output" | grep '^STATOR_FFI_BENCHMARK_RESULTS_JSON=' | sed 's/^STATOR_FFI_BENCHMARK_RESULTS_JSON=//')
    bench_item=$(node -e 'const rows = JSON.parse(process.argv[1]); process.stdout.write(JSON.stringify(rows[0]));' "$bench_json")
    if [ "$first" -eq 0 ]; then
        stator_json+=","
    fi
    stator_json+="$bench_item"
    first=0
done
stator_json+="]"

echo ""
echo "=== V8 vs Stator-FFI Comparison ==="
echo ""

# Use node to merge and display results.
node - "$v8_json" "$stator_json" <<'NODE'
const v8Results = JSON.parse(process.argv[2]);
const statorResults = JSON.parse(process.argv[3]);

// Build lookup by benchmark name.
const v8Map = {};
for (const r of v8Results) v8Map[r.name] = r;

const shared = [
    "fib_40_iterative",
    "arithmetic_loop_10k",
    "property_access_1k",
    "object_creation_1k",
    "array_push_sum_1k",
    "closure_counter_1k",
    "prototype_chain_1k",
    "sieve_primes_1k",
    "deep_object_access_1k",
];

const hdr = [
    "Benchmark".padEnd(28),
    "V8 Med(µs)".padStart(12),
    "V8 Min(µs)".padStart(12),
    "Stator(µs)".padStart(12),
    "Speedup".padStart(10),
    "Result".padStart(8),
].join("");
console.log(hdr);
console.log("-".repeat(82));

let wins = 0;
for (const name of shared) {
    const v8 = v8Map[name];
    const st = statorResults.find(r => r.name === name);
    if (!v8 || !st) continue;

    const v8Med = v8.median_ns / 1000;
    const v8Min = v8.min_ns / 1000;
    const stMed = st.median_ns / 1000;
    const ratio = v8Min / stMed;
    const result = ratio >= 1.0 ? "🏆 BEAT" : "❌ LOSE";
    if (ratio >= 1.0) wins++;

    console.log(
        name.padEnd(28) +
        v8Med.toFixed(1).padStart(12) +
        v8Min.toFixed(1).padStart(12) +
        stMed.toFixed(1).padStart(12) +
        (ratio.toFixed(2) + "x").padStart(10) +
        result.padStart(10)
    );
}

console.log("-".repeat(82));
console.log(`Score: ${wins}/${shared.length} benchmarks beat V8 (via FFI embedding layer)`);
console.log("");
NODE
