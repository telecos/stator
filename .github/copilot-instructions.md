# Copilot Agent Instructions for Stator

## Mandatory Pre-Commit Checks

Before committing ANY code changes, you MUST run the following checks and fix all issues they report. Do NOT push code that fails any of these:

```bash
# 1. Format all Rust code (auto-fixes formatting)
cargo fmt --all

# 2. Run clippy with warnings as errors — fix ALL warnings
cargo clippy --workspace -- -D warnings

# 3. Build the workspace (must succeed)
cargo build --workspace

# 4. Run all tests (must pass)
cargo test --workspace
```

**Run these in order.** If any step fails, fix the issue before proceeding to the next step. After fixing, re-run from step 1 to make sure the fix didn't introduce new problems.

### Common Clippy Issues to Avoid

- Unused imports: remove them, don't just suppress with `#[allow(unused)]`
- Unused variables: prefix with `_` or remove if truly unneeded
- Missing documentation on public items: add `///` doc comments
- Needless `return`: use expression-based returns (idiomatic Rust)
- `clone()` on Copy types: just copy, don't clone
- Single-match patterns: use `if let` instead of `match`
- Use `Self` instead of repeating the type name in impl blocks

### Formatting Rules

- Do NOT manually format code — let `cargo fmt` handle it
- Do NOT add `#[rustfmt::skip]` unless there is a very specific reason documented in a comment
- Use edition 2024 style (configured in `rust-toolchain.toml`)

## Code Style

- All `unsafe` blocks MUST have a `// SAFETY:` comment explaining why it's safe
- All public types and functions MUST have `///` doc comments
- Use `thiserror` for error types (shared dependency in workspace)
- Prefer `Result<T, StatorError>` (aliased as `StatorResult<T>`) for fallible operations
- Tests go in `#[cfg(test)] mod tests` at the bottom of each file
- Test names follow `test_<what>_<scenario>` convention

## Project Structure

- Engine internals: `crates/stator_core/src/`
- C-ABI FFI surface: `crates/stator_ffi/src/`
- Test262 harness: `crates/stator_test262/src/`
- CLI shell: `crates/st8/src/`
- Demo app (C++): `examples/mini_browser/`
- Fuzz targets: `fuzz/`

## FFI Conventions

- All FFI functions use `extern "C"` and `#[no_mangle]`
- FFI function names are prefixed with `stator_`
- Opaque types use `#[repr(C)]` with `_opaque: [u8; 0]`
- Use `cbindgen` to auto-generate the C header — don't hand-write `stator.h`

## When Implementing an Issue

1. Read the issue description fully before starting
2. Create the files/modules as specified in the issue
3. Write tests for every public function
4. Run the 4 mandatory checks above
5. Verify your changes compile with `--release` too: `cargo build --workspace --release`
6. If the issue mentions updating `examples/mini_browser/`, make sure it still compiles:
   ```bash
   cd examples/mini_browser && mkdir -p build && cd build
   cmake .. -DSTATOR_LIB_DIR=$PWD/../../../target/release
   make
   ```
