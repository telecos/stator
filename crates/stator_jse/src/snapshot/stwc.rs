//! Warm-context snapshot binary format (`STWC` v1).
//!
//! `STWC` is the third snapshot envelope shipped by the engine, sitting
//! alongside the legacy globals-only `STSS` format and the manifest-aware
//! `STSM` format.  It binds a snapshot to the exact engine build,
//! embedder ABI surface, target environment, callback manifest, and
//! (optionally) Edge vendored release metadata that produced it.  Every
//! compatibility key in the header is validated *before* the loader
//! decodes any payload bytes; any mismatch is a fail-closed
//! [`StatorError::SnapshotCompatibilityMismatch`] with the offending
//! field name surfaced through the C ABI for embedder telemetry.
//!
//! Design source: `docs/snapshot.md` §6 ("Strict deterministic format"),
//! §7 ("Compatibility / version metadata"), §8 ("Proposed FFI / API
//! shape").  The implementation reuses the existing strict
//! deterministic serializer (see
//! [`super::serialize_globals_with_manifest`] / [`super::write_jsvalue_with_manifest`])
//! for the value payload; only the outer envelope, build binding, and
//! integrity digest are new.
//!
//! # Binary layout (`STWC` v1)
//!
//! ```text
//! Header (variable length, fields written in declared order):
//!   magic                : [u8; 4]   = b"STWC"
//!   snapshot_format_ver  : u32 LE    = STWC_FORMAT_VERSION (1)
//!   bytecode_format_ver  : u32 LE
//!   engine_crate_ver     : str32     (UTF-8, u32 LE length prefix)
//!   ffi_crate_ver        : str32
//!   commit_id            : str32
//!   build_id             : [u8; 32]
//!   ffi_abi_version      : u32 LE
//!   target_triple        : str32
//!   os                   : str32
//!   arch                 : str32
//!   pointer_width        : u8
//!   endianness           : u8        (1 = little, 2 = big)
//!   cargo_profile        : str32
//!   build_features_hash  : [u8; 32]
//!   jit_tiering_hash     : [u8; 32]
//!   cpu_features_hash    : [u8; 32]
//!   manifest_hash        : [u8; 32]  (mirrors SnapshotCallbackManifest::digest)
//!   edge_release_hash    : [u8; 32]  (all-zero = unused)
//!   payload_len          : u64 LE
//!
//! Payload (payload_len bytes):
//!   manifest_id_count    : u32 LE
//!   manifest_ids         : str32 * manifest_id_count  (sorted lex, unique)
//!   global_count         : u32 LE
//!   entries              : (str32 key, JsValue value) * global_count
//!                          (keys sorted lexicographically; values use the
//!                           strict deterministic JsValue encoding shared
//!                           with `STSM`)
//!
//! Footer:
//!   digest               : [u8; 32]  (STWC integrity digest over the
//!                                     concatenation of header + payload
//!                                     bytes; computed by `stwc_hash32`,
//!                                     a four-fold domain-separated
//!                                     FNV-1a-64 hash)
//! ```
//!
//! # Compatibility / fail-closed loader
//!
//! [`load_globals_stwc`] verifies every header field against the
//! load-time [`StwcBuildBinding`].  Any mismatch is reported as
//! [`StatorError::SnapshotCompatibilityMismatch`] with the offending
//! `field` name.  The footer digest is verified before any payload is
//! decoded; a tampered or truncated blob is rejected with
//! [`StatorError::SnapshotDigestMismatch`] (or
//! [`StatorError::Internal`] for length / UTF-8 framing errors).
//! The callback manifest is verified by digest *and* sorted id list;
//! any missing or extra id returns
//! [`StatorError::SnapshotManifestMismatch`], identical to the
//! `STSM` loader.
//!
//! Legacy `STSS` and `STSM` blobs are entirely untouched: their
//! magic bytes are distinct, and neither loader accepts an `STWC`
//! blob (and vice versa).
//!
//! # Determinism
//!
//! Given the same `globals`, manifest, and build binding, the emitted
//! bytes are byte-for-byte deterministic.  The payload reuses
//! [`super::write_jsvalue_with_manifest`], which sorts every
//! `PlainObject` entry, every globals key, and every manifest id
//! lexicographically and canonicalizes any `f64` NaN payload.
//!
//! # Caps
//!
//! - `STWC_MAX_PAYLOAD_LEN`: 256 MiB (rejected with
//!   [`StatorError::SnapshotCompatibilityMismatch`] on `payload_len`).
//! - `STWC_MAX_METADATA_STR_LEN`: 4 KiB cap on individual header string
//!   fields (`engine_crate_ver`, `commit_id`, …); oversize values are
//!   rejected before allocation.

use std::collections::BTreeSet;
use std::collections::HashMap;

use crate::error::{StatorError, StatorResult};
use crate::objects::value::JsValue;

use super::manifest::manifest_digest_from_ids;
use super::{
    DeserContext, SerContext, SnapshotCallbackManifest, StartupSnapshot, hex_lower, need,
    read_jsvalue_with_manifest, read_str32, read_u8, read_u32, read_u64,
    write_jsvalue_with_manifest, write_str32, write_u8, write_u32, write_u64,
};

/// Magic bytes that identify a Stator Warm-Context (`STWC`) snapshot.
pub const STWC_MAGIC: [u8; 4] = *b"STWC";

/// Format version of the warm-context snapshot envelope.
pub const STWC_FORMAT_VERSION: u32 = 1;

/// Current engine bytecode encoding version.  Bumped whenever
/// [`crate::bytecode::bytecode_array::BytecodeArray`] gains or removes
/// an encoded field, a constant pool tag, or otherwise alters the byte
/// representation of compiled functions.  Mirrored into every `STWC`
/// header for fail-closed loading.
pub const STWC_BYTECODE_FORMAT_VERSION: u32 = 1;

/// Length in bytes of the warm-context integrity digest.
pub const STWC_DIGEST_LEN: usize = 32;

/// Length in bytes of every fixed-size hash field in the header
/// (`build_id`, `build_features_hash`, `jit_tiering_hash`,
/// `cpu_features_hash`, `manifest_hash`, `edge_release_hash`).
pub const STWC_HASH_LEN: usize = 32;

/// Maximum total payload size accepted by the loader (256 MiB).
pub const STWC_MAX_PAYLOAD_LEN: u64 = 256 * 1024 * 1024;

/// Maximum byte length of any individual header metadata string.
pub const STWC_MAX_METADATA_STR_LEN: u32 = 4096;

/// All-zero sentinel value for the optional `edge_release_hash`
/// header field, meaning "no Edge release binding".
pub const STWC_EDGE_RELEASE_UNSET: [u8; STWC_HASH_LEN] = [0u8; STWC_HASH_LEN];

/// Compatibility / build binding metadata embedded in every `STWC`
/// header.
///
/// The struct holds both the values the engine itself knows (crate
/// version, target triple, pointer width, endianness, build profile,
/// snapshot/bytecode format versions) and the values an embedder must
/// supply (FFI ABI version, FFI crate version, commit id, build id,
/// build/JIT/CPU feature hashes, optional Edge release hash).
///
/// Use [`StwcBuildBinding::current_engine_defaults`] to obtain a
/// binding pre-filled with the engine-known values and zeroed/empty
/// embedder-supplied fields, then override the fields the embedder is
/// authoritative about.  Both snapshot create-time and load-time
/// callers MUST construct equivalent bindings — any field difference
/// is a fatal load error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StwcBuildBinding {
    /// Warm-context envelope format version.  Defaults to
    /// [`STWC_FORMAT_VERSION`].  Embedders should not override this
    /// unless they intentionally produce a blob for a future loader.
    pub snapshot_format_ver: u32,
    /// Engine bytecode encoding version.  Defaults to
    /// [`STWC_BYTECODE_FORMAT_VERSION`].
    pub bytecode_format_ver: u32,
    /// `stator_jse` Cargo package version (e.g. `"0.1.0"`).
    pub engine_crate_ver: String,
    /// `stator_ffi` / `stator_jse_ffi` Cargo package version.  Empty
    /// string is allowed for embedders that do not consume the FFI
    /// crate, but every snapshot's create-time value MUST equal its
    /// load-time value.
    pub ffi_crate_ver: String,
    /// Git commit id (or `"unknown"`) of the engine build.
    pub commit_id: String,
    /// 32-byte fingerprint of the engine build; the canonical mismatch
    /// gate for "this snapshot was produced by a different binary".
    pub build_id: [u8; STWC_HASH_LEN],
    /// Packed `STATOR_FFI_ABI_VERSION` value
    /// (`(major << 16) | (minor << 8) | patch`).
    pub ffi_abi_version: u32,
    /// Rust target triple (e.g. `"x86_64-pc-windows-gnu"`).
    pub target_triple: String,
    /// Target OS component (e.g. `"windows"`, `"linux"`).
    pub os: String,
    /// Target arch component (e.g. `"x86_64"`, `"aarch64"`).
    pub arch: String,
    /// Pointer width in bytes (`4` or `8`).
    pub pointer_width: u8,
    /// Endianness marker: `1` = little, `2` = big.
    pub endianness: u8,
    /// Cargo build profile (`"debug"`, `"release"`, or a custom name).
    pub cargo_profile: String,
    /// 32-byte fingerprint of enabled Cargo features and relevant
    /// `cfg` values.
    pub build_features_hash: [u8; STWC_HASH_LEN],
    /// 32-byte fingerprint of the configured JIT / tiering modes.
    pub jit_tiering_hash: [u8; STWC_HASH_LEN],
    /// 32-byte fingerprint of the required CPU feature set.
    pub cpu_features_hash: [u8; STWC_HASH_LEN],
    /// 32-byte fingerprint of the Edge vendored release metadata.
    /// All-zero ([`STWC_EDGE_RELEASE_UNSET`]) means "not bound to an
    /// Edge release".
    pub edge_release_hash: [u8; STWC_HASH_LEN],
}

impl StwcBuildBinding {
    /// Construct a binding pre-populated with the values the engine
    /// can derive at compile time, leaving embedder-supplied fields
    /// at safe defaults.
    ///
    /// Pre-filled fields:
    /// - `snapshot_format_ver` = [`STWC_FORMAT_VERSION`]
    /// - `bytecode_format_ver` = [`STWC_BYTECODE_FORMAT_VERSION`]
    /// - `engine_crate_ver` = `env!("CARGO_PKG_VERSION")`
    /// - `target_triple` / `os` / `arch` from `std::env::consts`
    /// - `pointer_width` from `usize` size
    /// - `endianness` from `cfg!(target_endian = "little")`
    /// - `cargo_profile` from `cfg!(debug_assertions)`
    ///
    /// Embedder-supplied fields (`ffi_crate_ver`, `commit_id`,
    /// `build_id`, `ffi_abi_version`, hashes) default to empty
    /// strings / zero arrays.  An embedder that ships a real
    /// warm-context snapshot SHOULD override every one of them.
    pub fn current_engine_defaults() -> Self {
        Self {
            snapshot_format_ver: STWC_FORMAT_VERSION,
            bytecode_format_ver: STWC_BYTECODE_FORMAT_VERSION,
            engine_crate_ver: env!("CARGO_PKG_VERSION").to_owned(),
            ffi_crate_ver: String::new(),
            commit_id: String::new(),
            build_id: [0u8; STWC_HASH_LEN],
            ffi_abi_version: 0,
            target_triple: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
            os: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            pointer_width: std::mem::size_of::<usize>() as u8,
            endianness: if cfg!(target_endian = "little") { 1 } else { 2 },
            cargo_profile: if cfg!(debug_assertions) {
                "debug".to_owned()
            } else {
                "release".to_owned()
            },
            build_features_hash: [0u8; STWC_HASH_LEN],
            jit_tiering_hash: [0u8; STWC_HASH_LEN],
            cpu_features_hash: [0u8; STWC_HASH_LEN],
            edge_release_hash: STWC_EDGE_RELEASE_UNSET,
        }
    }
}

/// Compute the 32-byte warm-context integrity digest over arbitrary
/// bytes.
///
/// The digest is a domain-separated four-fold FNV-1a-64 hash, packed
/// little-endian.  It is **non-cryptographic** and matches the
/// in-process strength contract documented in `docs/snapshot.md` §6:
/// it detects accidental corruption and casual tampering but is not a
/// substitute for the BLAKE3 signature applied to the outer
/// release-bundle in §10.
pub fn stwc_hash32(data: &[u8]) -> [u8; STWC_DIGEST_LEN] {
    const DOMAINS: [&[u8]; 4] = [b"STWC1", b"STWC2", b"STWC3", b"STWC4"];
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hashes: [u64; 4] = [FNV_OFFSET; 4];
    for (slot, domain) in hashes.iter_mut().zip(DOMAINS) {
        for &b in domain {
            *slot ^= u64::from(b);
            *slot = slot.wrapping_mul(FNV_PRIME);
        }
        *slot ^= 0xff;
        *slot = slot.wrapping_mul(FNV_PRIME);
    }
    let len_bytes = (data.len() as u64).to_le_bytes();
    for slot in &mut hashes {
        for &b in &len_bytes {
            *slot ^= u64::from(b);
            *slot = slot.wrapping_mul(FNV_PRIME);
        }
    }
    for &b in data {
        for slot in &mut hashes {
            *slot ^= u64::from(b);
            *slot = slot.wrapping_mul(FNV_PRIME);
        }
    }
    let mut out = [0u8; STWC_DIGEST_LEN];
    for (i, h) in hashes.iter().enumerate() {
        out[i * 8..(i + 1) * 8].copy_from_slice(&h.to_le_bytes());
    }
    out
}

/// Serialize `globals` into a warm-context (`STWC` v1) snapshot.
///
/// The emitted blob binds the snapshot to `binding` (build / target /
/// ABI metadata) and `manifest` (sorted native callback id list and
/// its in-process digest).  Every reachable
/// [`JsValue::NativeFunction`] must be registered in `manifest`; any
/// unregistered callback or otherwise unsupported value class (see
/// [`super::serialize_globals_strict`]) is rejected with
/// [`StatorError::SnapshotUnsupportedValue`] before any bytes are
/// committed.
///
/// # Errors
///
/// - [`StatorError::SnapshotUnsupportedValue`] for any value class
///   the strict serializer cannot capture.
/// - [`StatorError::Internal`] if the resulting payload would exceed
///   [`STWC_MAX_PAYLOAD_LEN`].
pub fn serialize_globals_stwc(
    globals: &HashMap<String, JsValue>,
    manifest: &SnapshotCallbackManifest,
    binding: &StwcBuildBinding,
) -> StatorResult<StartupSnapshot> {
    // Build the payload first so we can stamp `payload_len` into the
    // header before computing the footer digest.
    let mut payload = Vec::new();
    let ids = manifest.sorted_ids();
    write_u32(&mut payload, ids.len() as u32);
    for id in &ids {
        write_str32(&mut payload, id);
    }
    write_u32(&mut payload, globals.len() as u32);
    let mut keys: Vec<&String> = globals.keys().collect();
    keys.sort();
    let mut ctx = SerContext::new();
    for key in keys {
        let value = &globals[key];
        write_str32(&mut payload, key);
        let path = format!("globals.{key}");
        write_jsvalue_with_manifest(&mut payload, value, &mut ctx, manifest, &path)?;
    }

    let payload_len = payload.len() as u64;
    if payload_len > STWC_MAX_PAYLOAD_LEN {
        return Err(StatorError::Internal(format!(
            "snapshot: STWC payload too large ({payload_len} bytes > cap {STWC_MAX_PAYLOAD_LEN})"
        )));
    }

    let mut buf = Vec::with_capacity(payload.len() + 512);
    write_header(&mut buf, binding, &manifest.digest(), payload_len);
    buf.extend_from_slice(&payload);
    let digest = stwc_hash32(&buf);
    buf.extend_from_slice(&digest);
    Ok(StartupSnapshot::from_bytes(buf))
}

/// Strict, fail-closed loader for warm-context (`STWC` v1) snapshots.
///
/// Steps, performed in order, each before any subsequent allocation:
///
/// 1. Validate magic and `snapshot_format_ver`.
/// 2. Decode every header field with bounded length checks.
/// 3. Compare every compatibility field against `binding`.  Any
///    mismatch returns
///    [`StatorError::SnapshotCompatibilityMismatch`].
/// 4. Verify the footer digest covers `header || payload` exactly.
///    A mismatch returns [`StatorError::SnapshotDigestMismatch`].
/// 5. Compare the embedded `manifest_hash` and sorted id list with
///    `manifest`; any difference returns
///    [`StatorError::SnapshotManifestMismatch`].
/// 6. Decode the payload using the strict manifest-aware value
///    decoder, reinstalling each `NativeFunction` by id.
///
/// No JS code is executed and no callback is invoked during load.
///
/// # Errors
///
/// See the step-by-step list above; framing / UTF-8 / unknown-tag
/// errors are surfaced as [`StatorError::Internal`] like the other
/// snapshot loaders.
pub fn load_globals_stwc(
    bytes: &[u8],
    manifest: &SnapshotCallbackManifest,
    binding: &StwcBuildBinding,
) -> StatorResult<HashMap<String, JsValue>> {
    if bytes.len() < STWC_DIGEST_LEN + 8 {
        return Err(StatorError::Internal(
            "snapshot: STWC blob too small for header + footer".into(),
        ));
    }

    let mut cursor = 0usize;

    // ── Magic + format version ────────────────────────────────────────────
    need(bytes, cursor, 4)?;
    let magic: [u8; 4] = bytes[cursor..cursor + 4].try_into().expect("4-byte slice");
    if magic != STWC_MAGIC {
        return Err(StatorError::SnapshotCompatibilityMismatch {
            field: "magic",
            found: render_magic(&magic),
            expected: render_magic(&STWC_MAGIC),
        });
    }
    cursor += 4;

    let snapshot_format_ver = read_u32(bytes, &mut cursor)?;
    if snapshot_format_ver != binding.snapshot_format_ver {
        return Err(StatorError::SnapshotCompatibilityMismatch {
            field: "snapshot_format_ver",
            found: snapshot_format_ver.to_string(),
            expected: binding.snapshot_format_ver.to_string(),
        });
    }

    let bytecode_format_ver = read_u32(bytes, &mut cursor)?;
    if bytecode_format_ver != binding.bytecode_format_ver {
        return Err(StatorError::SnapshotCompatibilityMismatch {
            field: "bytecode_format_ver",
            found: bytecode_format_ver.to_string(),
            expected: binding.bytecode_format_ver.to_string(),
        });
    }

    let engine_crate_ver = read_bounded_str(bytes, &mut cursor, "engine_crate_ver")?;
    expect_string_eq(
        "engine_crate_ver",
        &engine_crate_ver,
        &binding.engine_crate_ver,
    )?;

    let ffi_crate_ver = read_bounded_str(bytes, &mut cursor, "ffi_crate_ver")?;
    expect_string_eq("ffi_crate_ver", &ffi_crate_ver, &binding.ffi_crate_ver)?;

    let commit_id = read_bounded_str(bytes, &mut cursor, "commit_id")?;
    expect_string_eq("commit_id", &commit_id, &binding.commit_id)?;

    let mut build_id = [0u8; STWC_HASH_LEN];
    read_hash_into(bytes, &mut cursor, &mut build_id)?;
    expect_hash_eq("build_id", &build_id, &binding.build_id)?;

    let ffi_abi_version = read_u32(bytes, &mut cursor)?;
    if ffi_abi_version != binding.ffi_abi_version {
        return Err(StatorError::SnapshotCompatibilityMismatch {
            field: "ffi_abi_version",
            found: format!("{ffi_abi_version:#x}"),
            expected: format!("{:#x}", binding.ffi_abi_version),
        });
    }

    let target_triple = read_bounded_str(bytes, &mut cursor, "target_triple")?;
    expect_string_eq("target_triple", &target_triple, &binding.target_triple)?;

    let os = read_bounded_str(bytes, &mut cursor, "os")?;
    expect_string_eq("os", &os, &binding.os)?;

    let arch = read_bounded_str(bytes, &mut cursor, "arch")?;
    expect_string_eq("arch", &arch, &binding.arch)?;

    let pointer_width = read_u8(bytes, &mut cursor)?;
    if pointer_width != binding.pointer_width {
        return Err(StatorError::SnapshotCompatibilityMismatch {
            field: "pointer_width",
            found: pointer_width.to_string(),
            expected: binding.pointer_width.to_string(),
        });
    }

    let endianness = read_u8(bytes, &mut cursor)?;
    if endianness != binding.endianness {
        return Err(StatorError::SnapshotCompatibilityMismatch {
            field: "endianness",
            found: endianness.to_string(),
            expected: binding.endianness.to_string(),
        });
    }

    let cargo_profile = read_bounded_str(bytes, &mut cursor, "cargo_profile")?;
    expect_string_eq("cargo_profile", &cargo_profile, &binding.cargo_profile)?;

    let mut build_features_hash = [0u8; STWC_HASH_LEN];
    read_hash_into(bytes, &mut cursor, &mut build_features_hash)?;
    expect_hash_eq(
        "build_features_hash",
        &build_features_hash,
        &binding.build_features_hash,
    )?;

    let mut jit_tiering_hash = [0u8; STWC_HASH_LEN];
    read_hash_into(bytes, &mut cursor, &mut jit_tiering_hash)?;
    expect_hash_eq(
        "jit_tiering_hash",
        &jit_tiering_hash,
        &binding.jit_tiering_hash,
    )?;

    let mut cpu_features_hash = [0u8; STWC_HASH_LEN];
    read_hash_into(bytes, &mut cursor, &mut cpu_features_hash)?;
    expect_hash_eq(
        "cpu_features_hash",
        &cpu_features_hash,
        &binding.cpu_features_hash,
    )?;

    let mut manifest_hash = [0u8; STWC_HASH_LEN];
    read_hash_into(bytes, &mut cursor, &mut manifest_hash)?;

    let mut edge_release_hash = [0u8; STWC_HASH_LEN];
    read_hash_into(bytes, &mut cursor, &mut edge_release_hash)?;
    expect_hash_eq(
        "edge_release_hash",
        &edge_release_hash,
        &binding.edge_release_hash,
    )?;

    let payload_len = read_u64(bytes, &mut cursor)?;
    if payload_len > STWC_MAX_PAYLOAD_LEN {
        return Err(StatorError::SnapshotCompatibilityMismatch {
            field: "payload_len",
            found: payload_len.to_string(),
            expected: format!("<= {STWC_MAX_PAYLOAD_LEN}"),
        });
    }
    let payload_len = payload_len as usize;
    // Header ends at `cursor`; footer is the last 32 bytes.
    let header_end = cursor;
    if header_end + payload_len + STWC_DIGEST_LEN != bytes.len() {
        return Err(StatorError::SnapshotCompatibilityMismatch {
            field: "payload_len",
            found: payload_len.to_string(),
            expected: bytes
                .len()
                .saturating_sub(header_end + STWC_DIGEST_LEN)
                .to_string(),
        });
    }

    // ── Footer digest verification ────────────────────────────────────────
    let footer_start = bytes.len() - STWC_DIGEST_LEN;
    let stored_digest: [u8; STWC_DIGEST_LEN] =
        bytes[footer_start..].try_into().expect("32-byte slice");
    let computed_digest = stwc_hash32(&bytes[..footer_start]);
    if stored_digest != computed_digest {
        return Err(StatorError::SnapshotDigestMismatch {
            expected: hex_lower(&stored_digest),
            found: hex_lower(&computed_digest),
        });
    }

    // ── Payload decode ────────────────────────────────────────────────────
    let payload_end = footer_start;
    let payload_view = &bytes[..payload_end]; // header + payload, no footer
    let mut p_cursor = header_end;

    let id_count = read_u32(payload_view, &mut p_cursor)? as usize;
    let mut stored_ids: Vec<String> = Vec::with_capacity(id_count.min(4096));
    for _ in 0..id_count {
        stored_ids.push(read_str32(payload_view, &mut p_cursor)?);
    }
    if !stored_ids.windows(2).all(|w| w[0] < w[1]) {
        return Err(StatorError::Internal(
            "snapshot: STWC manifest id table is not sorted/unique".into(),
        ));
    }

    let recomputed_manifest_hash = manifest_digest_from_ids(stored_ids.iter().map(String::as_str));
    if recomputed_manifest_hash != manifest_hash {
        return Err(StatorError::Internal(format!(
            "snapshot: STWC stored manifest_hash ({}) does not match the digest of the \
             serialized id list ({}); snapshot blob is corrupt",
            hex_lower(&manifest_hash),
            hex_lower(&recomputed_manifest_hash),
        )));
    }

    let load_digest = manifest.digest();
    let load_ids: Vec<String> = manifest
        .sorted_ids()
        .into_iter()
        .map(str::to_owned)
        .collect();
    if load_digest != manifest_hash || load_ids != stored_ids {
        let stored_set: BTreeSet<&str> = stored_ids.iter().map(String::as_str).collect();
        let load_set: BTreeSet<&str> = load_ids.iter().map(String::as_str).collect();
        let missing_ids: Vec<String> = stored_set
            .difference(&load_set)
            .map(|s| (*s).to_owned())
            .collect();
        let extra_ids: Vec<String> = load_set
            .difference(&stored_set)
            .map(|s| (*s).to_owned())
            .collect();
        return Err(StatorError::SnapshotManifestMismatch {
            expected: hex_lower(&manifest_hash),
            found: hex_lower(&load_digest),
            missing_ids,
            extra_ids,
        });
    }

    let mut ctx = DeserContext::new();
    let count = read_u32(payload_view, &mut p_cursor)? as usize;
    let mut globals = HashMap::with_capacity(count.min(4096));
    for _ in 0..count {
        let key = read_str32(payload_view, &mut p_cursor)?;
        let value = read_jsvalue_with_manifest(payload_view, &mut p_cursor, &mut ctx, manifest)?;
        globals.insert(key, value);
    }
    if p_cursor != payload_end {
        return Err(StatorError::Internal(format!(
            "snapshot: STWC payload has {} trailing bytes after decode",
            payload_end - p_cursor
        )));
    }
    Ok(globals)
}

// ─────────────────────────────────────────────────────────────────────────────
// Header writing / reading helpers
// ─────────────────────────────────────────────────────────────────────────────

fn write_header(
    buf: &mut Vec<u8>,
    b: &StwcBuildBinding,
    manifest_hash: &[u8; STWC_HASH_LEN],
    payload_len: u64,
) {
    buf.extend_from_slice(&STWC_MAGIC);
    write_u32(buf, b.snapshot_format_ver);
    write_u32(buf, b.bytecode_format_ver);
    write_str32(buf, &b.engine_crate_ver);
    write_str32(buf, &b.ffi_crate_ver);
    write_str32(buf, &b.commit_id);
    buf.extend_from_slice(&b.build_id);
    write_u32(buf, b.ffi_abi_version);
    write_str32(buf, &b.target_triple);
    write_str32(buf, &b.os);
    write_str32(buf, &b.arch);
    write_u8(buf, b.pointer_width);
    write_u8(buf, b.endianness);
    write_str32(buf, &b.cargo_profile);
    buf.extend_from_slice(&b.build_features_hash);
    buf.extend_from_slice(&b.jit_tiering_hash);
    buf.extend_from_slice(&b.cpu_features_hash);
    buf.extend_from_slice(manifest_hash);
    buf.extend_from_slice(&b.edge_release_hash);
    write_u64(buf, payload_len);
}

fn read_bounded_str(bytes: &[u8], cursor: &mut usize, field: &'static str) -> StatorResult<String> {
    let len_at = *cursor;
    let len = read_u32(bytes, cursor)?;
    if len > STWC_MAX_METADATA_STR_LEN {
        return Err(StatorError::SnapshotCompatibilityMismatch {
            field,
            found: format!("string length {len} at offset {len_at}"),
            expected: format!("<= {STWC_MAX_METADATA_STR_LEN}"),
        });
    }
    let len = len as usize;
    need(bytes, *cursor, len)?;
    let s = std::str::from_utf8(&bytes[*cursor..*cursor + len])
        .map_err(|e| {
            StatorError::Internal(format!(
                "snapshot: STWC header `{field}` is not valid UTF-8: {e}"
            ))
        })?
        .to_owned();
    *cursor += len;
    Ok(s)
}

fn read_hash_into(
    bytes: &[u8],
    cursor: &mut usize,
    out: &mut [u8; STWC_HASH_LEN],
) -> StatorResult<()> {
    need(bytes, *cursor, STWC_HASH_LEN)?;
    out.copy_from_slice(&bytes[*cursor..*cursor + STWC_HASH_LEN]);
    *cursor += STWC_HASH_LEN;
    Ok(())
}

fn expect_string_eq(field: &'static str, found: &str, expected: &str) -> StatorResult<()> {
    if found == expected {
        Ok(())
    } else {
        Err(StatorError::SnapshotCompatibilityMismatch {
            field,
            found: found.to_owned(),
            expected: expected.to_owned(),
        })
    }
}

fn expect_hash_eq(
    field: &'static str,
    found: &[u8; STWC_HASH_LEN],
    expected: &[u8; STWC_HASH_LEN],
) -> StatorResult<()> {
    if found == expected {
        Ok(())
    } else {
        Err(StatorError::SnapshotCompatibilityMismatch {
            field,
            found: hex_lower(found),
            expected: hex_lower(expected),
        })
    }
}

fn render_magic(m: &[u8; 4]) -> String {
    match std::str::from_utf8(m) {
        Ok(s) => format!("\"{s}\""),
        Err(_) => format!("{m:?}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::value::{JsValue, NativeFn};
    use std::rc::Rc;

    fn cb(tag: i32) -> NativeFn {
        Rc::new(move |_| Ok(JsValue::Smi(tag)))
    }

    fn empty_globals() -> HashMap<String, JsValue> {
        HashMap::new()
    }

    fn fixed_binding() -> StwcBuildBinding {
        // Use a fully-deterministic binding so create / load are exact
        // mirrors regardless of the host environment.
        StwcBuildBinding {
            snapshot_format_ver: STWC_FORMAT_VERSION,
            bytecode_format_ver: STWC_BYTECODE_FORMAT_VERSION,
            engine_crate_ver: "0.1.0".to_owned(),
            ffi_crate_ver: "0.1.0".to_owned(),
            commit_id: "deadbeefcafef00d".to_owned(),
            build_id: [0x11; STWC_HASH_LEN],
            ffi_abi_version: 0x0001_0500,
            target_triple: "x86_64-unknown-linux-gnu".to_owned(),
            os: "linux".to_owned(),
            arch: "x86_64".to_owned(),
            pointer_width: 8,
            endianness: 1,
            cargo_profile: "release".to_owned(),
            build_features_hash: [0x22; STWC_HASH_LEN],
            jit_tiering_hash: [0x33; STWC_HASH_LEN],
            cpu_features_hash: [0x44; STWC_HASH_LEN],
            edge_release_hash: [0x55; STWC_HASH_LEN],
        }
    }

    #[test]
    fn test_stwc_magic_bytes() {
        let binding = fixed_binding();
        let manifest = SnapshotCallbackManifest::new();
        let snap =
            serialize_globals_stwc(&empty_globals(), &manifest, &binding).expect("serialize");
        assert_eq!(&snap.as_bytes()[0..4], b"STWC");
    }

    #[test]
    fn test_stwc_format_version_field() {
        let binding = fixed_binding();
        let manifest = SnapshotCallbackManifest::new();
        let snap = serialize_globals_stwc(&empty_globals(), &manifest, &binding).unwrap();
        let bytes = snap.as_bytes();
        let ver = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(ver, STWC_FORMAT_VERSION);
        let bc_ver = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        assert_eq!(bc_ver, STWC_BYTECODE_FORMAT_VERSION);
    }

    #[test]
    fn test_stwc_deterministic_output() {
        let binding = fixed_binding();
        let cb1 = cb(1);
        let mut manifest = SnapshotCallbackManifest::new();
        manifest.register("edge.alpha", cb1.clone()).unwrap();
        let mut g: HashMap<String, JsValue> = HashMap::new();
        g.insert("alpha".into(), JsValue::NativeFunction(cb1));
        g.insert("zeta".into(), JsValue::Smi(7));
        g.insert("beta".into(), JsValue::String("hi".to_string().into()));
        let s1 = serialize_globals_stwc(&g, &manifest, &binding).unwrap();
        let s2 = serialize_globals_stwc(&g, &manifest, &binding).unwrap();
        assert_eq!(s1.as_bytes(), s2.as_bytes(), "STWC must be deterministic");
    }

    #[test]
    fn test_stwc_round_trip_supported_values() {
        let binding = fixed_binding();
        let cb1 = cb(42);
        let mut manifest = SnapshotCallbackManifest::new();
        manifest.register("edge.cb", cb1.clone()).unwrap();
        let mut g = HashMap::new();
        g.insert("n".into(), JsValue::Smi(3));
        g.insert("s".into(), JsValue::String("hi".to_string().into()));
        g.insert("cb".into(), JsValue::NativeFunction(cb1.clone()));

        let snap = serialize_globals_stwc(&g, &manifest, &binding).unwrap();
        let restored = load_globals_stwc(snap.as_bytes(), &manifest, &binding).unwrap();
        assert_eq!(restored.get("n"), Some(&JsValue::Smi(3)));
        assert_eq!(
            restored.get("s"),
            Some(&JsValue::String("hi".to_string().into())),
        );
        match restored.get("cb") {
            Some(JsValue::NativeFunction(rc)) => assert!(Rc::ptr_eq(rc, &cb1)),
            other => panic!("expected NativeFunction, got {other:?}"),
        }
    }

    #[test]
    fn test_stwc_rejects_unsupported_value() {
        let binding = fixed_binding();
        let manifest = SnapshotCallbackManifest::new();
        let mut g = HashMap::new();
        g.insert("h".into(), JsValue::TheHole);
        let err = serialize_globals_stwc(&g, &manifest, &binding).unwrap_err();
        match err {
            StatorError::SnapshotUnsupportedValue { class, path, .. } => {
                assert_eq!(class, "TheHole");
                assert_eq!(path, "globals.h");
            }
            other => panic!("expected SnapshotUnsupportedValue, got {other:?}"),
        }
    }

    #[test]
    fn test_stwc_rejects_unregistered_native_function() {
        let binding = fixed_binding();
        let manifest = SnapshotCallbackManifest::new();
        let mut g = HashMap::new();
        g.insert("cb".into(), JsValue::NativeFunction(cb(0)));
        let err = serialize_globals_stwc(&g, &manifest, &binding).unwrap_err();
        assert!(matches!(
            err,
            StatorError::SnapshotUnsupportedValue {
                class: "NativeFunction",
                ..
            }
        ));
    }

    #[test]
    fn test_stwc_compat_mismatch_engine_crate_ver() {
        let create = fixed_binding();
        let mut load = create.clone();
        load.engine_crate_ver = "9.9.9".into();
        let manifest = SnapshotCallbackManifest::new();
        let snap = serialize_globals_stwc(&empty_globals(), &manifest, &create).unwrap();
        let err = load_globals_stwc(snap.as_bytes(), &manifest, &load).unwrap_err();
        match err {
            StatorError::SnapshotCompatibilityMismatch {
                field,
                found,
                expected,
            } => {
                assert_eq!(field, "engine_crate_ver");
                assert_eq!(found, "0.1.0");
                assert_eq!(expected, "9.9.9");
            }
            other => panic!("expected SnapshotCompatibilityMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_stwc_compat_mismatch_build_id() {
        let create = fixed_binding();
        let mut load = create.clone();
        load.build_id = [0x99; STWC_HASH_LEN];
        let manifest = SnapshotCallbackManifest::new();
        let snap = serialize_globals_stwc(&empty_globals(), &manifest, &create).unwrap();
        let err = load_globals_stwc(snap.as_bytes(), &manifest, &load).unwrap_err();
        match err {
            StatorError::SnapshotCompatibilityMismatch { field, .. } => {
                assert_eq!(field, "build_id");
            }
            other => panic!("expected SnapshotCompatibilityMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_stwc_compat_mismatch_ffi_abi_version() {
        let create = fixed_binding();
        let mut load = create.clone();
        load.ffi_abi_version = 0xdead_beef;
        let manifest = SnapshotCallbackManifest::new();
        let snap = serialize_globals_stwc(&empty_globals(), &manifest, &create).unwrap();
        let err = load_globals_stwc(snap.as_bytes(), &manifest, &load).unwrap_err();
        assert!(matches!(
            err,
            StatorError::SnapshotCompatibilityMismatch {
                field: "ffi_abi_version",
                ..
            }
        ));
    }

    #[test]
    fn test_stwc_compat_mismatch_edge_release_hash() {
        let create = fixed_binding();
        let mut load = create.clone();
        load.edge_release_hash = STWC_EDGE_RELEASE_UNSET;
        let manifest = SnapshotCallbackManifest::new();
        let snap = serialize_globals_stwc(&empty_globals(), &manifest, &create).unwrap();
        let err = load_globals_stwc(snap.as_bytes(), &manifest, &load).unwrap_err();
        assert!(matches!(
            err,
            StatorError::SnapshotCompatibilityMismatch {
                field: "edge_release_hash",
                ..
            }
        ));
    }

    #[test]
    fn test_stwc_rejects_tampered_digest() {
        let binding = fixed_binding();
        let manifest = SnapshotCallbackManifest::new();
        let snap = serialize_globals_stwc(&empty_globals(), &manifest, &binding).unwrap();
        let mut tampered = snap.into_bytes();
        let n = tampered.len();
        // Flip a bit inside the footer digest.
        tampered[n - 1] ^= 0x01;
        let err = load_globals_stwc(&tampered, &manifest, &binding).unwrap_err();
        assert!(matches!(err, StatorError::SnapshotDigestMismatch { .. }));
    }

    #[test]
    fn test_stwc_rejects_tampered_payload_body() {
        let binding = fixed_binding();
        let manifest = SnapshotCallbackManifest::new();
        let mut g = HashMap::new();
        g.insert("k".into(), JsValue::Smi(1));
        let snap = serialize_globals_stwc(&g, &manifest, &binding).unwrap();
        let mut tampered = snap.into_bytes();
        let n = tampered.len();
        // Flip a bit in the middle (likely inside the payload).
        tampered[n / 2] ^= 0x01;
        let err = load_globals_stwc(&tampered, &manifest, &binding).unwrap_err();
        // Could be digest mismatch (most likely) or a framing error if the
        // flipped byte fell inside a length prefix.  Either is acceptable
        // fail-closed behavior.
        assert!(
            matches!(
                err,
                StatorError::SnapshotDigestMismatch { .. }
                    | StatorError::Internal(_)
                    | StatorError::SnapshotCompatibilityMismatch { .. }
            ),
            "expected digest / framing failure, got: {err:?}"
        );
    }

    #[test]
    fn test_stwc_rejects_legacy_stss_blob() {
        // A legacy `STSS` snapshot must not be accepted by the STWC loader.
        let stss = super::super::serialize_globals(&empty_globals());
        let binding = fixed_binding();
        let manifest = SnapshotCallbackManifest::new();
        let err = load_globals_stwc(stss.as_bytes(), &manifest, &binding).unwrap_err();
        assert!(matches!(
            err,
            StatorError::SnapshotCompatibilityMismatch { field: "magic", .. }
                | StatorError::Internal(_)
        ));
    }

    #[test]
    fn test_stwc_blob_rejected_by_legacy_loaders() {
        // The reverse must also hold: legacy loaders must refuse STWC blobs.
        let binding = fixed_binding();
        let manifest = SnapshotCallbackManifest::new();
        let snap = serialize_globals_stwc(&empty_globals(), &manifest, &binding).unwrap();
        assert!(super::super::deserialize_globals(snap.as_bytes()).is_err());
        assert!(super::super::reinstall_globals_with_manifest(snap.as_bytes(), &manifest).is_err());
    }

    #[test]
    fn test_stwc_too_small_blob_rejected() {
        let binding = fixed_binding();
        let manifest = SnapshotCallbackManifest::new();
        let err = load_globals_stwc(&[0u8; 10], &manifest, &binding).unwrap_err();
        assert!(matches!(err, StatorError::Internal(_)));
    }

    #[test]
    fn test_stwc_manifest_id_set_mismatch_rejected() {
        let binding = fixed_binding();
        let cb1 = cb(1);
        let mut create_manifest = SnapshotCallbackManifest::new();
        create_manifest.register("edge.foo", cb1.clone()).unwrap();
        create_manifest.register("edge.bar", cb(2)).unwrap();
        let mut g = HashMap::new();
        g.insert("foo".into(), JsValue::NativeFunction(cb1));
        let snap = serialize_globals_stwc(&g, &create_manifest, &binding).unwrap();

        let mut load_manifest = SnapshotCallbackManifest::new();
        load_manifest.register("edge.foo", cb(99)).unwrap();
        let err = load_globals_stwc(snap.as_bytes(), &load_manifest, &binding).unwrap_err();
        match err {
            StatorError::SnapshotManifestMismatch {
                missing_ids,
                extra_ids,
                ..
            } => {
                assert_eq!(missing_ids, vec!["edge.bar".to_string()]);
                assert!(extra_ids.is_empty());
            }
            other => panic!("expected SnapshotManifestMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_current_engine_defaults_populates_engine_fields() {
        let b = StwcBuildBinding::current_engine_defaults();
        assert_eq!(b.snapshot_format_ver, STWC_FORMAT_VERSION);
        assert_eq!(b.bytecode_format_ver, STWC_BYTECODE_FORMAT_VERSION);
        assert_eq!(b.engine_crate_ver, env!("CARGO_PKG_VERSION"));
        assert!(b.pointer_width == 4 || b.pointer_width == 8);
        assert!(b.endianness == 1 || b.endianness == 2);
        assert!(!b.os.is_empty());
        assert!(!b.arch.is_empty());
        // Defaults match across two invocations within the same process.
        let b2 = StwcBuildBinding::current_engine_defaults();
        assert_eq!(b, b2);
    }

    #[test]
    fn test_stwc_hash_is_deterministic_and_sensitive() {
        let a = stwc_hash32(b"hello");
        let b = stwc_hash32(b"hello");
        let c = stwc_hash32(b"hello!");
        assert_eq!(a, b);
        assert_ne!(a, c);
        // Empty input still produces a non-zero digest (so a wholly-zero
        // footer is detectable as wrong).
        assert_ne!(stwc_hash32(&[]), [0u8; STWC_DIGEST_LEN]);
    }

    #[test]
    fn test_stwc_does_not_perturb_legacy_paths() {
        // Smoke: legacy serialize_globals / deserialize_globals round-trip
        // still works after we link in the STWC module.
        let mut g = HashMap::new();
        g.insert("x".into(), JsValue::Smi(7));
        let snap = super::super::serialize_globals(&g);
        let restored = super::super::deserialize_globals(snap.as_bytes()).unwrap();
        assert_eq!(restored.get("x"), Some(&JsValue::Smi(7)));
        assert_eq!(&snap.as_bytes()[0..4], b"STSS");
    }
}
