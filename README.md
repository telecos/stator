# Stator

Stator is an experimental JavaScript engine written in Rust, featuring a
Maglev-inspired JIT compiler that generates native x86-64 machine code.
It is designed to be embedded in browser-like environments such as Chromium via
a stable C FFI layer.

[![CI](https://github.com/telecos/stator/actions/workflows/ci.yml/badge.svg)](https://github.com/telecos/stator/actions/workflows/ci.yml)
[![Test262](https://github.com/telecos/stator/actions/workflows/test262.yml/badge.svg)](https://github.com/telecos/stator/actions/workflows/test262.yml)
[![Benchmarks](https://github.com/telecos/stator/actions/workflows/bench.yml/badge.svg)](https://github.com/telecos/stator/actions/workflows/bench.yml)
[![Coverage](https://codecov.io/gh/telecos/stator/branch/main/graph/badge.svg)](https://codecov.io/gh/telecos/stator)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Highlights

- **100% Test262 conformance** — full ECMAScript specification compliance
- **Maglev JIT compiler** — generates optimized x86-64 machine code with inline
  caches, loop-invariant code motion (LICM), and range analysis
- **WebAssembly support** — full WebAssembly JS API via Cranelift/Wasmtime
- **Chrome DevTools Protocol** — attach the Chrome inspector to debug running
  JavaScript
- **Embeddable via C FFI** — stable C ABI for integration with Chromium, Node.js
  or any C/C++ application
- **Memory-safe** — written entirely in safe Rust (with targeted `unsafe` only
  for JIT code emission and FFI boundaries)

## Performance

Stator's JIT compiler produces competitive native code.  On two micro-benchmarks
it already **beats V8 (Node.js)** while maintaining full spec compliance:

| Benchmark | V8 (µs) | Stator (µs) | Ratio | Status |
|---|---|---|---|---|
| fib_40_iterative | 1.4 | 0.7 | **0.5x** | ✅ Beats V8 |
| prototype_chain_1k | 17.9 | 5.4 | **0.3x** | ✅ Beats V8 |
| arithmetic_loop_10k | 6.3 | 12.9 | 2.0x | 🔧 Optimizing |
| deep_object_access_1k | 1.6 | 3.4 | 2.1x | 🔧 Optimizing |

> Measured on GitHub Actions `ubuntu-latest` runners.  See the
> [Benchmarks workflow](https://github.com/telecos/stator/actions/workflows/bench.yml)
> for the latest numbers.

## Architecture

```
┌──────────────────────────────────────────┐
│           Chromium / Embedder            │
│   (C++ content layer, Node.js, etc.)     │
└───────────────────┬──────────────────────┘
                    │  C ABI  (stator.h)
┌───────────────────▼──────────────────────┐
│              stator_jse_ffi                  │
│  cdylib / staticlib — opaque C handles   │
│  stator_isolate_create / _gc / _destroy  │
└───────────────────┬──────────────────────┘
                    │  Rust rlib
┌───────────────────▼──────────────────────┐
│             stator_jse                  │
│  Parser · Bytecode · Maglev JIT · GC     │
│  Interpreter · IC · Objects · Heap       │
└──────────────────────────────────────────┘
```

### Crates

| Crate              | Type               | Purpose                                          |
|--------------------|--------------------|--------------------------------------------------|
| `stator_jse`      | rlib               | Engine internals — parser, bytecode compiler, Maglev JIT, interpreter, GC, heap, objects |
| `stator_jse_ffi`       | cdylib + staticlib | Stable C API for embedders                       |
| `st8`              | bin                | Interactive JavaScript shell (like V8's `d8`)    |
| `stator_jse_test262`   | bin                | ECMA-262 Test262 conformance harness             |

### Execution pipeline

```
Source → Parser → AST → Bytecode Compiler → Bytecode
                                               │
                              Interpreter ◄────┘
                                  │
                          (hot loop detected)
                                  │
                          Maglev JIT Compiler
                           │          │
                     Graph Builder   Optimizer
                           │      (LICM, Range Analysis,
                           │       Global Promotion)
                           │          │
                         Register Allocator
                                  │
                          x86-64 Code Emission
                          (Inline Caches, Deopts)
```

## Requirements

| Tool | Minimum version |
|------|-----------------|
| Rust (stable) | 1.85 (edition 2024) |
| Cargo | bundled with Rust |

**Platform note:** The Maglev JIT compiler targets x86-64 Linux.  The
interpreter and all other components work on all platforms Rust supports.

## Build

```sh
# Build all crates
cargo build

# Build in release mode (recommended for benchmarks)
cargo build --release
```

The C FFI artifacts are written to:

- `target/release/libstator_jse_ffi.a` — static library (Linux / macOS)
- `target/release/libstator_jse_ffi.so` — shared library (Linux)
- `target/release/stator_jse_ffi.lib` / `stator_jse_ffi.dll` — Windows equivalents

## Test

```sh
# Run unit tests
cargo test

# Run ECMA-262 Test262 conformance suite
cargo run --release --bin stator_jse_test262
```

## Benchmarks

```sh
# Run the benchmark suite (requires release mode)
cargo bench -p stator_jse
```

Benchmarks compare Stator's JIT output against V8 (Node.js) on the same
workloads.  The CI runs benchmarks on every push to `main`:
[Benchmark results →](https://github.com/telecos/stator/actions/workflows/bench.yml)

## `st8` CLI

`st8` is the Stator command-line shell, analogous to V8's `d8`.  It executes
JavaScript files, evaluates inline snippets, and supports Chrome DevTools
Protocol (CDP) inspection.

```sh
# Build the shell
cargo build --bin st8

# Execute a JavaScript file
cargo run --bin st8 -- file.js

# Evaluate an inline expression
cargo run --bin st8 -- -e '1 + 2'

# Run with Chrome DevTools inspector on port 9229
cargo run --bin st8 -- --inspect file.js

# Run and pause before the first statement
cargo run --bin st8 -- --inspect-brk file.js
```

Example session:

```
$ echo 'print(6 * 7)' > hello.js
$ cargo run --bin st8 -- hello.js
42
```

### Built-in globals

| Global | Description |
|---|---|
| `print(...args)` | Prints arguments joined by a space, followed by a newline |
| `console.log(...args)` | Alias for `print` |
| `WebAssembly` | Full WebAssembly JS API namespace |

### CLI options

| Flag | Description |
|---|---|
| `-e '<code>'` | Evaluate an inline JavaScript expression |
| `--inspect[=port]` | Start CDP inspector (default port: 9229) |
| `--inspect-brk[=port]` | Start inspector and pause before first statement |
| `--emit-snapshot=<path>` | Serialize built-in globals to a snapshot file |
| `--snapshot=<path>` | Load globals from a snapshot for faster startup |
| `--jit-stats` | Print JIT compilation statistics after execution |

## Embedding

See [`examples/mini_browser`](examples/mini_browser/README.md) for a minimal
C++ application that demonstrates how Chromium's content layer would create and
destroy isolates via the C FFI API.

## CI

| Workflow | Purpose |
|---|---|
| [CI](https://github.com/telecos/stator/actions/workflows/ci.yml) | Format, clippy, unit tests (debug + release), mini_browser, differential testing |
| [Test262](https://github.com/telecos/stator/actions/workflows/test262.yml) | Full ECMA-262 conformance suite |
| [Benchmarks](https://github.com/telecos/stator/actions/workflows/bench.yml) | Performance regression tracking vs V8 |
| [Coverage](https://codecov.io/gh/telecos/stator) | Code coverage via codecov |
| [Fuzz](https://github.com/telecos/stator/actions/workflows/fuzz.yml) | Fuzz testing for parser and interpreter |
| [ASAN / TSAN / Miri](https://github.com/telecos/stator/actions/workflows/asan.yml) | Memory and thread safety sanitizers |

## Roadmap

See the [open issues](https://github.com/telecos/stator/issues) and project
board for the current roadmap and planned milestones.

## License

Stator is distributed under the [MIT License](LICENSE).
