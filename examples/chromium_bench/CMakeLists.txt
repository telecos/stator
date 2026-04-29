cmake_minimum_required(VERSION 3.16)
project(chromium_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Locate the pre-built Stator FFI library.
# Pass -DSTATOR_LIB_DIR=<path> to override the default.
# ---------------------------------------------------------------------------
if(NOT DEFINED STATOR_LIB_DIR)
    set(STATOR_LIB_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../../target/release"
        CACHE PATH "Directory containing libstator_jse_ffi.{a,so,dylib,lib,dll}")
endif()

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
# chromium_bench executable
# ---------------------------------------------------------------------------
add_executable(chromium_bench src/main.cpp)

# Prefer the generated FFI header; keep mini_browser include/ available for
# compatibility shims used by other embedder examples.
target_include_directories(chromium_bench PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../crates/stator_ffi/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../mini_browser/include"
)

target_link_libraries(chromium_bench PRIVATE stator_jse_ffi)

if(UNIX AND NOT APPLE)
    target_link_libraries(chromium_bench PRIVATE pthread dl m)
endif()
