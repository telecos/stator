# stator_jse

Core Rust library for the Stator JavaScript engine.

This crate contains the parser, bytecode compiler, interpreter, garbage
collector, WebAssembly integration, inspector support, and Maglev-inspired JIT
infrastructure used by Stator. Most embedders should prefer the
`stator_jse_ffi` crate, which exposes a stable C ABI and generated C/C++
headers.

## Platform notes

- The interpreter and core runtime are portable Rust.
- The native Maglev-style JIT targets x86-64 Linux.
- The crate uses Rust edition 2024.

## Package contents

The published crate intentionally includes only:

- `src/`: engine source.
- `benches/`: benchmark targets referenced by the manifest.
- `examples/`: small example targets referenced by the manifest.
- `README.md` and package metadata.

Conformance harnesses, fuzz targets, Chromium integration shims, CI scripts,
and repository-level benchmark artifacts are not part of this crate package.

## Related crates

- `stator_jse_ffi`: stable C ABI for browser and C/C++ embedders.
- `st8`: command-line JavaScript shell built on top of `stator_jse`.
