# mini_browser — Stator Embedding Example

A minimal C++ application that demonstrates how an embedder (e.g. a browser's
content layer) would use the Stator JavaScript engine via its C FFI API.

## What it does

The sample exercises the Phase 1 GC and object model to simulate a browser tab
lifecycle:

1. **Create isolate and context** — one engine instance and execution context.
2. **Allocate values** — a number, a string, and an object with named
   properties are created via the FFI.
3. **Inspect values** — types and contents are read back through the API.
4. **GC with live handles** — a minor collection is triggered; objects survive
   because handles are still held.
5. **Release handles, GC again** — handles are destroyed and a second GC
   confirms all objects are reclaimed.
6. **Print heap stats** — final bytes-used and capacity are reported.

## Prerequisites

| Tool | Minimum version |
|---|---|
| Rust (stable) | 1.85 (edition 2024) |
| CMake | 3.16 |
| C++17-capable compiler | GCC 8, Clang 7, or MSVC 2019 |

## Build

### 1 — Build the Stator FFI library

From the **workspace root**:

```sh
cargo build --release
```

This produces `target/release/libstator_js_ffi.a` (Linux/macOS) or
`target/release/stator_js_ffi.lib` (Windows).

### 2 — Build `mini_browser`

```sh
cd examples/mini_browser
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

If the Rust workspace root is not two directory levels above
`examples/mini_browser`, pass the library location explicitly:

```sh
cmake -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DSTATOR_LIB_DIR=/absolute/path/to/target/release
cmake --build build
```

### 3 — Run

```sh
./build/mini_browser
```

Expected output:

```
[tab] created context
[tab] allocated: number(42), string("hello"), object{x: 1, y: 2}
[tab] GC: 3 objects survived (held by handles)
[tab] released handles, GC: 0 objects (all reclaimed)
[tab] heap: 0 bytes used / 8388608 bytes capacity
```

## File layout

```
examples/mini_browser/
├── CMakeLists.txt        # CMake build definition
├── README.md             # This file
├── include/
│   └── stator.h          # C header mirroring crates/stator_js_ffi exports
└── src/
    └── main.cpp          # Minimalistic browser simulation
```

## API used

| Function | Description |
|---|---|
| `stator_isolate_create()` | Allocate a new engine instance |
| `stator_context_new(isolate)` | Create an execution context |
| `stator_value_new_number(isolate, val)` | Create a number value handle |
| `stator_value_new_string(isolate, data, len)` | Create a string value handle |
| `stator_object_new(isolate)` | Create an empty object handle |
| `stator_object_set(obj, key, val)` | Set a named property |
| `stator_object_get(obj, key)` | Get a named property as a new value handle |
| `stator_value_type(val)` | Return the type name (`"number"` or `"string"`) |
| `stator_value_as_number(val)` | Extract the numeric value |
| `stator_value_as_string(val)` | Extract the string contents |
| `stator_gc_collect(isolate)` | Trigger a minor GC |
| `stator_live_object_count(isolate)` | Count live embedder-held handles |
| `stator_heap_used(isolate)` | Young-generation bytes currently in use |
| `stator_heap_capacity(isolate)` | Total young-generation capacity |
| `stator_value_destroy(val)` | Release a value handle |
| `stator_object_destroy(obj)` | Release an object handle |
| `stator_context_destroy(ctx)` | Release a context |
| `stator_isolate_destroy(isolate)` | Release all isolate resources |
