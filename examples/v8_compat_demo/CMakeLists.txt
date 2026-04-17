cmake_minimum_required(VERSION 3.16)
project(v8_compat_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Locate the pre-built Stator FFI library.
#
# Expected layout (produced by `cargo build --release` in the workspace root):
#   <workspace>/target/release/libstator_js_ffi.a   (Linux/macOS static)
#
# Pass -DSTATOR_LIB_DIR=<path> to override the default, e.g.:
#   cmake -DSTATOR_LIB_DIR=/path/to/target/release ..
# ---------------------------------------------------------------------------
if(NOT DEFINED STATOR_LIB_DIR)
    set(STATOR_LIB_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../../target/release"
        CACHE PATH "Directory containing libstator_js_ffi.{a,so,dylib,lib,dll}")
endif()

if(WIN32)
    set(_stator_lib_name "stator_js_ffi.lib")
else()
    set(_stator_lib_name "libstator_js_ffi.a")
endif()

add_library(stator_js_ffi STATIC IMPORTED)
set_target_properties(stator_js_ffi PROPERTIES
    IMPORTED_LOCATION "${STATOR_LIB_DIR}/${_stator_lib_name}"
)

# ---------------------------------------------------------------------------
# Locate the v8_compat.h header.
#
# The canonical copy lives next to stator.h in:
#   crates/stator_js_ffi/include/
# From this CMakeLists.txt that is two levels up.
# ---------------------------------------------------------------------------
set(STATOR_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../crates/stator_js_ffi/include"
    CACHE PATH "Directory containing stator.h and v8_compat.h")

# ---------------------------------------------------------------------------
# v8_compat_demo executable
# ---------------------------------------------------------------------------
add_executable(v8_compat_demo src/main.cpp)

target_include_directories(v8_compat_demo PRIVATE "${STATOR_INCLUDE_DIR}")

target_link_libraries(v8_compat_demo PRIVATE stator_js_ffi)

# On Linux the static Rust library needs the C runtime and pthread.
if(UNIX AND NOT APPLE)
    target_link_libraries(v8_compat_demo PRIVATE pthread dl m)
endif()
