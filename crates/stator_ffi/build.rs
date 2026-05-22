//! build.rs — cbindgen pipeline for `stator_jse_ffi`.
//!
//! Generates `include/stator.h` from the `extern "C"` surface in
//! `src/lib.rs`.  The header is written to the crate root (not `$OUT_DIR`)
//! so it can be committed and used by embedders without a Rust toolchain.
//!
//! Also re-emits the `stator_maglev_jit_x86_64` cfg so that the FFI crate's
//! `cfg(stator_maglev_jit_x86_64)` branches are recognised on the same set of
//! targets the underlying `stator_jse` crate considers Maglev-capable.

use std::path::PathBuf;

fn main() {
    // Mirror the Maglev JIT support cfg from `stator_jse`'s build.rs so the
    // FFI crate can gate on the same predicate without recomputing it at
    // every call site.
    println!("cargo::rustc-check-cfg=cfg(stator_baseline_jit_x86_64)");
    println!("cargo::rustc-check-cfg=cfg(stator_maglev_jit_x86_64)");
    println!("cargo::rustc-check-cfg=cfg(stator_baseline_jit_x86_64)");
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_family = std::env::var("CARGO_CFG_TARGET_FAMILY").unwrap_or_default();
    let families: Vec<&str> = target_family.split(',').collect();
    let maglev_supported =
        target_arch == "x86_64" && (families.contains(&"unix") || families.contains(&"windows"));
    if maglev_supported {
        println!("cargo::rustc-cfg=stator_maglev_jit_x86_64");
    }

    let crate_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));

    let stator_jse_manifest = crate_dir.join("..").join("stator_jse").join("Cargo.toml");
    let stator_jse_manifest =
        std::fs::read_to_string(stator_jse_manifest).expect("failed to read stator_jse Cargo.toml");
    let stator_jse_version = stator_jse_manifest
        .lines()
        .find_map(|line| line.strip_prefix("version = \"")?.strip_suffix('"'))
        .expect("failed to find stator_jse package version");
    println!("cargo::rustc-env=STATOR_JSE_CRATE_VERSION={stator_jse_version}");

    let include_dir = crate_dir.join("include");
    std::fs::create_dir_all(&include_dir).expect("failed to create include/ directory");

    let header_path = include_dir.join("stator.h");

    let config = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml"))
        .expect("cbindgen.toml not found");

    cbindgen::generate_with_config(&crate_dir, config)
        .expect("cbindgen failed to generate header")
        .write_to_file(&header_path);
}
