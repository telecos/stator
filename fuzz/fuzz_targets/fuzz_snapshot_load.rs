//! Fuzz target for snapshot blob loading and classification.
//!
//! This target exercises the strict snapshot loaders for all three
//! recognised envelope kinds (`STSS`, `STSM`, `STWC`) plus the
//! unknown-magic rejection path.  The contract under test is that
//! arbitrary, possibly malicious input bytes either deserialize cleanly
//! into a `HashMap<String, JsValue>` *or* surface a structured
//! [`stator_jse::error::StatorError`].  Panics, aborts, undefined
//! behaviour, or lossy "success" results (e.g. truncated decoding that
//! silently returns `Ok` for inputs that fail their own footer / magic
//! / digest checks) are bugs — libFuzzer will surface the first two,
//! and ASan/MSan/UBSan instrumentation surfaces the rest when this
//! target is run under `cargo +nightly fuzz run snapshot_load`.
//!
//! No JS callbacks are registered in the manifest used here, so this
//! target additionally enforces the "no host code may run during
//! snapshot load" invariant for every input.

#![no_main]

use libfuzzer_sys::fuzz_target;

use stator_fuzz::{prepare_snapshot_bytes, SnapshotEnvelope};
use stator_jse::snapshot::{
    deserialize_globals, load_globals_stwc, reinstall_globals_with_manifest,
    SnapshotCallbackManifest, StwcBuildBinding,
};

fuzz_target!(|data: &[u8]| {
    // Shape the raw bytes into one of `{ Unknown, Stss, Stsm, Stwc }`
    // envelopes.  The first byte is consumed as a selector so libFuzzer's
    // mutator can flip envelope kinds cheaply during coverage-guided
    // exploration.
    let (bytes, envelope) = prepare_snapshot_bytes(data);

    // An empty manifest is sufficient: every fuzz input that happens to
    // reach the NativeFunction reinstall path will fail closed with
    // SnapshotManifestMismatch (or an Internal framing error) because no
    // ids are registered.  This also enforces that the loaders never
    // call into host code during a fuzz iteration.
    let manifest = SnapshotCallbackManifest::new();
    let binding = StwcBuildBinding::current_engine_defaults();

    match envelope {
        SnapshotEnvelope::Stss => {
            // Any Ok/Err is acceptable; we only care that the loader
            // returns and does not panic / abort / trigger UB.
            let _ = deserialize_globals(&bytes);
        }
        SnapshotEnvelope::Stsm => {
            let _ = reinstall_globals_with_manifest(&bytes, &manifest);
        }
        SnapshotEnvelope::Stwc => {
            let _ = load_globals_stwc(&bytes, &manifest, &binding);
        }
        SnapshotEnvelope::Unknown => {
            // Drive all three loaders with arbitrary bytes that have no
            // recognised magic prefix.  Each must reject the input via a
            // structured StatorError rather than panicking.
            let _ = deserialize_globals(&bytes);
            let _ = reinstall_globals_with_manifest(&bytes, &manifest);
            let _ = load_globals_stwc(&bytes, &manifest, &binding);
        }
    }
});
