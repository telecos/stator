#!/usr/bin/env node

import fs from "node:fs";

const TARGETS = [
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

function argValue(name, fallback) {
  const index = process.argv.indexOf(name);
  return index === -1 ? fallback : process.argv[index + 1];
}

function readResults(path, label) {
  let rows;
  try {
    rows = JSON.parse(fs.readFileSync(path, "utf8"));
  } catch (error) {
    throw new Error(`failed to read ${label} results from ${path}: ${error.message}`);
  }
  if (!Array.isArray(rows)) {
    throw new Error(`${label} results in ${path} must be a JSON array`);
  }
  return new Map(rows.map((row) => [row.name, row]));
}

function fmtUs(ns) {
  return (ns / 1000).toFixed(2);
}

function compareOne(name, label, statorMap, v8Map, maxRatio) {
  const v8 = v8Map.get(name);
  const stator = statorMap.get(name);
  if (!v8 || !stator) {
    return {
      name,
      label,
      status: "FAIL",
      reason: !v8 ? "missing V8 result" : `missing ${label} result`,
    };
  }
  const ratio = stator.median_ns / v8.median_ns;
  const speedup = v8.median_ns / stator.median_ns;
  return {
    name,
    label,
    status: ratio <= maxRatio ? "PASS" : "FAIL",
    v8MedianNs: v8.median_ns,
    statorMedianNs: stator.median_ns,
    ratio,
    speedup,
  };
}

const v8Path = argValue("--v8", "v8.json");
const precompiledPath = argValue("--precompiled", "stator_precompiled.json");
const ffiPath = argValue("--ffi", "stator_ffi.json");
const maxRatio = Number(argValue("--max-ratio", process.env.STATOR_V8_MAX_RATIO ?? "1.0"));

if (!Number.isFinite(maxRatio) || maxRatio <= 0) {
  throw new Error(`invalid max ratio: ${maxRatio}`);
}

const v8Map = readResults(v8Path, "V8");
const precompiledMap = readResults(precompiledPath, "Stator precompiled");
const ffiMap = readResults(ffiPath, "Stator FFI");

const rows = [];
for (const name of TARGETS) {
  rows.push(compareOne(name, "precompiled", precompiledMap, v8Map, maxRatio));
  rows.push(compareOne(name, "ffi", ffiMap, v8Map, maxRatio));
}

const lines = [];
lines.push("| Benchmark | Path | V8 median (us) | Stator median (us) | Speedup | Ratio | Result |");
lines.push("|---|---:|---:|---:|---:|---:|---|");

let failures = 0;
for (const row of rows) {
  if (row.status !== "PASS") {
    failures += 1;
  }
  if (row.reason) {
    lines.push(`| ${row.name} | ${row.label} | - | - | - | - | FAIL: ${row.reason} |`);
    continue;
  }
  lines.push(
    `| ${row.name} | ${row.label} | ${fmtUs(row.v8MedianNs)} | ${fmtUs(row.statorMedianNs)} | ${row.speedup.toFixed(2)}x | ${row.ratio.toFixed(3)} | ${row.status} |`,
  );
}

const output = [
  `Stator/V8 performance gate (median-vs-median, required ratio <= ${maxRatio})`,
  "",
  ...lines,
  "",
  failures === 0 ? "All targeted Stator benchmark paths beat V8." : `${failures} targeted benchmark path(s) did not beat V8.`,
].join("\n");

console.log(output);

if (process.env.GITHUB_STEP_SUMMARY) {
  fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY, `${output}\n`);
}

if (failures > 0) {
  process.exit(1);
}
