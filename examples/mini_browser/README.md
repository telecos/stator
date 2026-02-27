# mini_browser — Stator Embedding Example

A minimal C++ application that demonstrates how an embedder (e.g. a browser's
content layer) would use the Stator JavaScript engine via its C FFI API.

## What it does

The sample simulates the lifecycle of two browser tabs, each owning its own
Stator *isolate*:

1. **Create isolate** — one engine instance per tab.
2. **Parse HTML** — a trivial `<script>` extractor pulls inline scripts out of
   a raw HTML string.
3. **Execute scripts** — the extracted script source is passed to the engine
   *(stubbed until the interpreter is implemented)*.
4. **GC** — a minor collection is triggered after the page load to reclaim
   short-lived heap objects.
5. **Destroy isolate** — all resources are released when the tab closes.

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

This produces `target/release/libstator_ffi.a` (Linux/macOS) or
`target/release/stator_ffi.lib` (Windows).

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

Expected output (once the interpreter is wired up the `(not yet implemented)`
lines will be replaced with actual results):

```
=== Stator mini-browser (placeholder) ===

[tab] opening   https://example.com/page-a
[tab] loading   https://example.com/page-a
[tab] parsed    HTML document
[tab] found     1 inline script(s)
[tab] executing script[0]: console.log('page A loaded');
[tab] (script execution not yet implemented)
[tab] gc        collecting nursery
[tab] loaded    https://example.com/page-a
[tab] closing   https://example.com/page-a

[tab] opening   https://example.com/page-b
[tab] loading   https://example.com/page-b
[tab] parsed    HTML document
[tab] found     2 inline script(s)
[tab] executing script[0]: const x = 1 + 2;
[tab] (script execution not yet implemented)
[tab] executing script[1]: console.log('x =', x);
[tab] (script execution not yet implemented)
[tab] gc        collecting nursery
[tab] loaded    https://example.com/page-b
[tab] closing   https://example.com/page-b

Done.
```

## File layout

```
examples/mini_browser/
├── CMakeLists.txt        # CMake build definition
├── README.md             # This file
├── include/
│   └── stator.h          # C header mirroring crates/stator_ffi exports
└── src/
    └── main.cpp          # Minimalistic browser simulation
```

## API used

| Function | Description |
|---|---|
| `stator_isolate_create()` | Allocate a new engine instance |
| `stator_isolate_gc()` | Trigger a minor GC on the isolate heap |
| `stator_isolate_destroy()` | Release all isolate resources |

Future API additions (planned, not yet implemented):

| Function | Description |
|---|---|
| `stator_context_create(isolate)` | Create a JS execution context |
| `stator_context_eval(ctx, src, len)` | Execute a JavaScript source string |
| `stator_context_destroy(ctx)` | Release context resources |
