# Stator

Stator is an experimental JavaScript engine written in Rust, designed to be
embedded in browser-like environments such as Chromium via a stable C FFI layer.

[![CI](https://github.com/telecos/stator/actions/workflows/ci.yml/badge.svg)](https://github.com/telecos/stator/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Architecture

```
┌──────────────────────────────────────────┐
│           Chromium / Embedder            │
│   (C++ content layer, Node.js, etc.)     │
└───────────────────┬──────────────────────┘
                    │  C ABI  (stator.h)
┌───────────────────▼──────────────────────┐
│              stator_ffi                  │
│  cdylib / staticlib — opaque C handles   │
│  stator_isolate_create / _gc / _destroy  │
└───────────────────┬──────────────────────┘
                    │  Rust rlib
┌───────────────────▼──────────────────────┐
│             stator_core                  │
│  Heap · GC · Objects · Parser · IR · VM  │
└──────────────────────────────────────────┘
```

Additional crates:

| Crate              | Type        | Purpose                                          |
|--------------------|-------------|--------------------------------------------------|
| `stator_core`      | rlib        | Engine internals (heap, GC, IR, VM)              |
| `stator_ffi`       | cdylib + staticlib | Stable C API for embedders              |
| `st8`              | bin         | Interactive JavaScript shell (like V8's `d8`)    |
| `stator_test262`   | bin         | Test262 conformance harness runner               |

## Requirements

| Tool | Minimum version |
|------|-----------------|
| Rust (stable) | 1.85 (edition 2024) |
| Cargo | bundled with Rust |

## Build

```sh
# Build all crates
cargo build

# Build in release mode
cargo build --release
```

The C FFI artifacts are written to:

- `target/release/libstator_ffi.a` — static library (Linux / macOS)
- `target/release/libstator_ffi.so` — shared library (Linux)
- `target/release/stator_ffi.lib` / `stator_ffi.dll` — Windows equivalents

## Test

```sh
cargo test
```

## `st8` CLI

`st8` is the Stator command-line shell, analogous to V8's `d8`.  It provides
script execution, a REPL, and debugging utilities once the interpreter is
functional.

```sh
# Build the shell
cargo build --bin st8

# Run the shell (placeholder output until interpreter is wired up)
cargo run --bin st8
```

Expected output:

```
st8: Stator JavaScript shell (not yet implemented)
```

## Embedding example

See [`examples/mini_browser`](examples/mini_browser/README.md) for a minimal
C++ application that demonstrates how Chromium's content layer would create and
destroy isolates via the C FFI API.

## Roadmap

See the [open issues](https://github.com/telecos/stator/issues) and project
board for the current roadmap and planned milestones.

## License

Stator is distributed under the [MIT License](LICENSE).
