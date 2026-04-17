// stator_bridge.cc — Compilation unit for the stator_bridge GN source_set.
//
// The public v8:: API surface is provided entirely by the header-only shim
// crates/stator_js_ffi/include/v8_compat.h.  This file exists so that GN has a
// .cc translation unit to compile (source_set requires at least one source)
// and to verify that the shim header compiles cleanly as a standalone C++17
// translation unit.
//
// Chromium embedders should #include "v8_compat.h" (or "stator.h" for the
// raw C API) directly from their own source files; they do not need to
// include this file.

#include "stator.h"
#include "v8_compat.h"
