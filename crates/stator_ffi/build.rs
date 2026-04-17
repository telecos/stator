//! build.rs — cbindgen pipeline for `stator_jse_ffi`.
//!
//! Generates `include/stator.h` from the `extern "C"` surface in
//! `src/lib.rs`.  The header is written to the crate root (not `$OUT_DIR`)
//! so it can be committed and used by embedders without a Rust toolchain.

use std::path::PathBuf;

fn main() {
    let crate_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));

    let include_dir = crate_dir.join("include");
    std::fs::create_dir_all(&include_dir).expect("failed to create include/ directory");

    let header_path = include_dir.join("stator.h");

    let config = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml"))
        .expect("cbindgen.toml not found");

    cbindgen::generate_with_config(&crate_dir, config)
        .expect("cbindgen failed to generate header")
        .write_to_file(&header_path);
}
