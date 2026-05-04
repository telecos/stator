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
