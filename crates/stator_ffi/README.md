# stator_jse_ffi

Stable C ABI layer for embedding the Stator JavaScript engine.

This crate builds `stator_jse` as a C-compatible library and ships generated
headers for C and C++ embedders. It is the recommended crate for Chromium, Edge,
and other non-Rust hosts.

## Outputs

The library is built as:

- `cdylib`
- `staticlib`
- `rlib`

The package includes:

- `include/stator.h`: raw C ABI.
- `include/v8_compat.h`: lightweight C++ compatibility shim for selected V8-like
  embedding APIs.
- `src/lib.rs`: FFI implementation.
- `build.rs` and `cbindgen.toml`: header generation pipeline.

Repository-level tests, benchmarks, Chromium proof-of-concept files, and CI
configuration are intentionally excluded from the crate package.

## Minimal C embedding flow

```c
#include "stator.h"

StatorIsolate *isolate = stator_isolate_create();
StatorContext *context = stator_context_new(isolate);

const char *source = "1 + 2";
StatorScript *script = stator_script_compile(context, source, 5);
StatorValue *value = stator_script_run(script, context);

stator_value_destroy(value);
stator_script_free(script);
stator_context_destroy(context);
stator_isolate_destroy(isolate);
```

Always pair handles returned by the API with the matching destroy/free function
documented in `stator.h`.

## ABI version contract

The C ABI exposes an explicit, packed version marker so embedders can detect a
header/library skew before the first FFI call:

- `STATOR_FFI_ABI_VERSION_MAJOR` / `STATOR_FFI_ABI_VERSION_MINOR` /
  `STATOR_FFI_ABI_VERSION_PATCH` — individual components, emitted as `#define`s
  in `include/stator.h`.
- `STATOR_FFI_ABI_VERSION` — packed `(major << 16) | (minor << 8) | patch`.
- `uint32_t stator_ffi_abi_version(void)` — exported function returning the
  packed value compiled into the library. Embedders should compare it against
  the header constant at startup and refuse to proceed on mismatch.

The major component is bumped on any breaking change to existing exported
types or function signatures, the minor component on additive changes (new
functions or trailing enum variants), and the patch component for non-ABI
fixes.

The integration test `tests/abi_contract.rs` enforces this contract on every
`cargo test` run:

- It calls `stator_ffi_abi_version()` and asserts equality with the
  `STATOR_FFI_ABI_VERSION` Rust constant.
- It asserts that the generated `include/stator.h` still contains all
  `STATOR_FFI_ABI_VERSION*` markers and `stator_ffi_abi_version*` function
  declarations.
- It diffs the set of `stator_*` function names parsed from the generated
  header against the checked-in baseline at `tests/abi_symbols.baseline.txt`.
  Any added or removed export fails the test, forcing an intentional baseline
  refresh and a deliberate ABI version bump.

After an intentional ABI change, regenerate the baseline:

```pwsh
$env:STATOR_FFI_ABI_UPDATE_BASELINE = "1"
cargo test -p stator_jse_ffi --test abi_contract
Remove-Item Env:STATOR_FFI_ABI_UPDATE_BASELINE
```

Commit the updated `tests/abi_symbols.baseline.txt` alongside the matching
bump to `STATOR_FFI_ABI_VERSION_MAJOR`/`MINOR`/`PATCH` in `src/lib.rs`.
