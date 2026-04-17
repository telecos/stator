cmake_minimum_required(VERSION 3.16)
project(mini_browser CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Locate the pre-built Stator FFI library.
#
# Expected layout (produced by `cargo build --release` in the workspace root):
#
#   <workspace>/target/release/libstator_jse_ffi.a      (Linux / macOS static)
#   <workspace>/target/release/libstator_jse_ffi.so     (Linux dynamic)
#   <workspace>/target/release/libstator_jse_ffi.dylib  (macOS dynamic)
#   <workspace>/target/release/stator_jse_ffi.dll       (Windows dynamic)
#   <workspace>/target/release/stator_jse_ffi.lib       (Windows import lib)
#
# Pass -DSTATOR_LIB_DIR=<path> to CMake to override the default, e.g.:
#   cmake -DSTATOR_LIB_DIR=/path/to/target/release ..
# ---------------------------------------------------------------------------
if(NOT DEFINED STATOR_LIB_DIR)
    # Default: two levels up from this CMakeLists.txt → workspace root,
    # then into target/release.
    set(STATOR_LIB_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../../target/release"
        CACHE PATH "Directory containing libstator_jse_ffi.{a,so,dylib,lib,dll}")
endif()

# Prefer the static library so the sample can run without a dynamic linker
# path adjustment.
if(WIN32)
    set(_stator_lib_name "stator_jse_ffi.lib")
else()
    set(_stator_lib_name "libstator_jse_ffi.a")
endif()

add_library(stator_jse_ffi STATIC IMPORTED)
set_target_properties(stator_jse_ffi PROPERTIES
    IMPORTED_LOCATION "${STATOR_LIB_DIR}/${_stator_lib_name}"
)

# ---------------------------------------------------------------------------
# mini_browser executable
# ---------------------------------------------------------------------------
add_executable(mini_browser src/main.cpp)

target_include_directories(mini_browser PRIVATE include)

target_link_libraries(mini_browser PRIVATE stator_jse_ffi)

# On Linux the static Rust library needs the C runtime and pthread.
if(UNIX AND NOT APPLE)
    target_link_libraries(mini_browser PRIVATE pthread dl m)
endif()
