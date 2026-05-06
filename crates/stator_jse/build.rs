//! Build script for `stator_jse`.
//!
//! Emits two internal cfgs:
//!
//! - `stator_baseline_jit_x86_64` — set when the target supports the Stator
//!   baseline JIT.
//! - `stator_maglev_jit_x86_64` — set when the target supports the Stator
//!   Maglev JIT.
//!
//! Both tiers are currently enabled on `target_arch = "x86_64"` for Unix and
//! Windows.  The Turbofan tier continues to gate on
//! `cfg(all(target_arch = "x86_64", unix))` independently.
//!
//! Centralising the predicates here keeps the source tree readable: instead
//! of repeating `cfg(all(target_arch = "x86_64", any(unix, windows)))` in
//! hundreds of call sites, the source uses `cfg(stator_baseline_jit_x86_64)`
//! / `cfg(stator_maglev_jit_x86_64)`.

fn main() {
    // Tell rustc the cfgs are expected (avoids `unexpected-cfgs` warnings).
    println!("cargo::rustc-check-cfg=cfg(stator_baseline_jit_x86_64)");
    println!("cargo::rustc-check-cfg=cfg(stator_maglev_jit_x86_64)");

    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_family = std::env::var("CARGO_CFG_TARGET_FAMILY").unwrap_or_default();
    // CARGO_CFG_TARGET_FAMILY is comma-separated for some triples
    // (e.g. "unix,wasm").
    let families: Vec<&str> = target_family.split(',').collect();

    let baseline_supported =
        target_arch == "x86_64" && (families.contains(&"unix") || families.contains(&"windows"));

    if baseline_supported {
        println!("cargo::rustc-cfg=stator_baseline_jit_x86_64");
    }

    // Maglev currently shares the same supported-target set as baseline.
    let maglev_supported = baseline_supported;
    if maglev_supported {
        println!("cargo::rustc-cfg=stator_maglev_jit_x86_64");
    }

    // Re-run when the target changes.
    println!("cargo:rerun-if-changed=build.rs");
}
