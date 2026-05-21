//! Embedder-supplied registry mapping stable string ids to native host
//! callbacks, used by the warm-context snapshot pipeline to serialize
//! and reinstall [`JsValue::NativeFunction`] values without serializing
//! raw function pointers.
//!
//! See `docs/snapshot.md` §4 ("Native callback / template reinstall
//! plan") for the design contract.  This module provides the v1 in-tree
//! implementation of `SnapshotCallbackManifest` plus a deterministic
//! 32-byte validation digest suitable for embedding in Edge release
//! metadata.
//!
//! ## Contract summary
//!
//! - At snapshot **create** time the embedder builds a
//!   [`SnapshotCallbackManifest`] and registers every native callback
//!   that may be reachable from globals.  Each registered callback gets
//!   a stable string id (e.g. `"edge.console.log"`).  Strict
//!   serialization writes the id (`str32`) — never the raw closure — and
//!   fails closed with [`StatorError::SnapshotUnsupportedValue`] if it
//!   encounters a `NativeFunction` whose `Rc` allocation is not present
//!   in the manifest.
//! - At snapshot **load** time the embedder constructs a matching
//!   manifest.  The loader compares the load-time digest with the
//!   on-disk digest, refuses to proceed on any mismatch (no
//!   `allow_extra` mode in v1), and reinstalls each id with the
//!   load-time callback `Rc`.  Missing ids are a fatal load error.
//!
//! ## Manifest digest
//!
//! [`SnapshotCallbackManifest::digest`] returns 32 bytes derived
//! deterministically from the sorted id list using four
//! domain-separated FNV-1a-64 folds.  This is a **non-cryptographic**
//! digest sufficient for the in-process validation contract; the Edge
//! release-bundle format (`manifest.json` / `STWC` header) layers a
//! BLAKE3 digest on top of the bundle bytes — see `docs/snapshot.md`
//! §10 — and that outer hash is still authoritative for signing.
//!
//! ## Pointer identity
//!
//! Callbacks are matched at serialization time by the data-pointer of
//! the `Rc<dyn Fn(...)>` (the same `Rc` that was registered must be
//! installed in globals).  Two distinct `Rc::new(|_| …)` allocations
//! containing logically equivalent closures are treated as different
//! callbacks; the embedder is expected to centralise creation so that
//! every global-table entry references the same `Rc` clone.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::rc::Rc;

use crate::error::{StatorError, StatorResult};
use crate::objects::value::{JsValue, NativeFn};

/// Length in bytes of the in-process manifest digest.
pub const MANIFEST_DIGEST_LEN: usize = 32;

/// 32-byte fingerprint of a `SnapshotCallbackManifest`.
///
/// Computed by [`SnapshotCallbackManifest::digest`] over the sorted id
/// list with four domain-separated FNV-1a-64 folds.  The digest is
/// deterministic for a given id set and changes whenever any id is
/// added, removed, or renamed.
pub type ManifestDigest = [u8; MANIFEST_DIGEST_LEN];

/// Maximum permitted byte length of a callback id.  Keeps the on-disk
/// `str32` envelope from being abused by oversized ids.
pub const MAX_CALLBACK_ID_LEN: usize = 256;

/// Registry of native host callbacks addressable by stable string id.
///
/// See the [module documentation](self) for the snapshot contract and
/// usage notes.
#[derive(Clone, Default)]
pub struct SnapshotCallbackManifest {
    /// id → callback.  Stored in a `BTreeMap` so id enumeration is
    /// always lexicographically sorted, which is the canonical order
    /// for the digest and the on-disk id table.
    by_id: BTreeMap<String, NativeFn>,
    /// Reverse index from the `Rc` data-pointer address to the
    /// registered id, used by the strict serializer to translate a
    /// `JsValue::NativeFunction(rc)` back into its manifest id.
    by_ptr: HashMap<usize, String>,
}

impl std::fmt::Debug for SnapshotCallbackManifest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SnapshotCallbackManifest")
            .field("ids", &self.by_id.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl SnapshotCallbackManifest {
    /// Construct an empty manifest.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `cb` under stable string `id`.
    ///
    /// # Errors
    ///
    /// - [`StatorError::Internal`] if `id` is empty, exceeds
    ///   [`MAX_CALLBACK_ID_LEN`], contains non-printable ASCII, or is
    ///   already registered with a different callback `Rc`.  Registering
    ///   the same id with the exact same `Rc` clone is a no-op.
    pub fn register(&mut self, id: &str, cb: NativeFn) -> StatorResult<()> {
        validate_id(id)?;
        let ptr = native_fn_ptr(&cb);
        if let Some(existing) = self.by_id.get(id) {
            if Rc::ptr_eq(existing, &cb) {
                return Ok(());
            }
            return Err(StatorError::Internal(format!(
                "snapshot manifest: id `{id}` is already registered with a different callback"
            )));
        }
        if let Some(other_id) = self.by_ptr.get(&ptr) {
            return Err(StatorError::Internal(format!(
                "snapshot manifest: callback already registered under id `{other_id}`, \
                 cannot also register as `{id}`"
            )));
        }
        self.by_id.insert(id.to_owned(), cb);
        self.by_ptr.insert(ptr, id.to_owned());
        Ok(())
    }

    /// Number of registered callbacks.
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    /// `true` if no callbacks have been registered.
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// `true` if `id` is registered.
    pub fn contains(&self, id: &str) -> bool {
        self.by_id.contains_key(id)
    }

    /// Look up the callback registered under `id`.
    pub fn callback(&self, id: &str) -> Option<&NativeFn> {
        self.by_id.get(id)
    }

    /// Look up the id under which the given [`JsValue::NativeFunction`]
    /// is registered (matched by `Rc` data-pointer identity), or
    /// `None` if the value is not a native function or not registered.
    pub fn id_for_value(&self, value: &JsValue) -> Option<&str> {
        match value {
            JsValue::NativeFunction(rc) => self
                .by_ptr
                .get(&native_fn_ptr(rc))
                .map(std::string::String::as_str),
            _ => None,
        }
    }

    /// Return the sorted (lexicographic) list of registered ids.
    ///
    /// This is the canonical order used by both [`Self::digest`] and
    /// the on-disk id table of a manifest-aware snapshot.
    pub fn sorted_ids(&self) -> Vec<&str> {
        self.by_id.keys().map(std::string::String::as_str).collect()
    }

    /// Compute the deterministic 32-byte manifest digest over the
    /// sorted id list.
    ///
    /// The digest is **non-cryptographic** (four domain-separated
    /// FNV-1a-64 folds) and is intended for the in-process integrity
    /// check between snapshot create and load.  Bundle-level signing
    /// uses a separate BLAKE3 digest layered on top.
    pub fn digest(&self) -> ManifestDigest {
        manifest_digest_from_ids(self.by_id.keys().map(std::string::String::as_str))
    }
}

/// Validate that `id` is a well-formed manifest id.
fn validate_id(id: &str) -> StatorResult<()> {
    if id.is_empty() {
        return Err(StatorError::Internal(
            "snapshot manifest: callback id must not be empty".into(),
        ));
    }
    if id.len() > MAX_CALLBACK_ID_LEN {
        return Err(StatorError::Internal(format!(
            "snapshot manifest: callback id `{id}` exceeds {MAX_CALLBACK_ID_LEN} bytes"
        )));
    }
    for (i, b) in id.bytes().enumerate() {
        if !(0x20..0x7F).contains(&b) {
            return Err(StatorError::Internal(format!(
                "snapshot manifest: callback id contains non-printable byte {b:#04x} at index {i}"
            )));
        }
    }
    Ok(())
}

/// Return the data-pointer address (as `usize`) of a `Rc<dyn Fn(...)>`,
/// stripping the trailing vtable pointer from the fat pointer.
pub(crate) fn native_fn_ptr(cb: &NativeFn) -> usize {
    Rc::as_ptr(cb) as *const () as usize
}

/// Compute the manifest digest from any iterator of sorted ids.
///
/// Exposed at crate scope so the snapshot deserializer can recompute
/// the digest from the on-disk id table without reconstructing a
/// `SnapshotCallbackManifest`.
pub(crate) fn manifest_digest_from_ids<'a, I>(ids: I) -> ManifestDigest
where
    I: IntoIterator<Item = &'a str>,
{
    // Four FNV-1a-64 hashes, each seeded with a 5-byte domain
    // separator, packed little-endian to give 32 bytes.
    const DOMAINS: [&[u8]; 4] = [b"STMD1", b"STMD2", b"STMD3", b"STMD4"];
    let mut hashes: [u64; 4] = [0; 4];
    for (slot, domain) in hashes.iter_mut().zip(DOMAINS) {
        *slot = 0xcbf2_9ce4_8422_2325_u64;
        for &b in domain {
            *slot ^= u64::from(b);
            *slot = slot.wrapping_mul(0x0000_0100_0000_01b3);
        }
        // Separator after domain prefix.
        *slot ^= 0xff;
        *slot = slot.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let ids: Vec<&str> = ids.into_iter().collect();
    let count = ids.len() as u32;
    let count_bytes = count.to_le_bytes();
    for slot in &mut hashes {
        for &b in &count_bytes {
            *slot ^= u64::from(b);
            *slot = slot.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    for id in ids {
        let len_bytes = (id.len() as u32).to_le_bytes();
        for slot in &mut hashes {
            for &b in &len_bytes {
                *slot ^= u64::from(b);
                *slot = slot.wrapping_mul(0x0000_0100_0000_01b3);
            }
            for &b in id.as_bytes() {
                *slot ^= u64::from(b);
                *slot = slot.wrapping_mul(0x0000_0100_0000_01b3);
            }
            // Per-id record separator so `["ab", "c"]` and `["a", "bc"]`
            // (which would already differ via the length prefix) also
            // differ if any future length-prefix change is made.
            *slot ^= 0x1f;
            *slot = slot.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    let mut out = [0u8; MANIFEST_DIGEST_LEN];
    for (i, h) in hashes.iter().enumerate() {
        out[i * 8..(i + 1) * 8].copy_from_slice(&h.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cb() -> NativeFn {
        Rc::new(|_args| Ok(JsValue::Undefined))
    }

    #[test]
    fn test_register_and_lookup() {
        let mut m = SnapshotCallbackManifest::new();
        let c = cb();
        m.register("edge.foo", c.clone()).unwrap();
        assert!(m.contains("edge.foo"));
        assert!(!m.contains("edge.bar"));
        assert_eq!(m.len(), 1);
        assert!(!m.is_empty());
        assert!(Rc::ptr_eq(m.callback("edge.foo").unwrap(), &c));
        let val = JsValue::NativeFunction(c);
        assert_eq!(m.id_for_value(&val), Some("edge.foo"));
    }

    #[test]
    fn test_id_for_value_unregistered() {
        let m = SnapshotCallbackManifest::new();
        let v = JsValue::NativeFunction(cb());
        assert_eq!(m.id_for_value(&v), None);
        assert_eq!(m.id_for_value(&JsValue::Undefined), None);
    }

    #[test]
    fn test_register_idempotent_same_rc() {
        let mut m = SnapshotCallbackManifest::new();
        let c = cb();
        m.register("x", c.clone()).unwrap();
        m.register("x", c).unwrap(); // same Rc → ok
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_register_duplicate_id_different_rc_errors() {
        let mut m = SnapshotCallbackManifest::new();
        m.register("x", cb()).unwrap();
        let err = m.register("x", cb()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("already registered"), "got: {msg}");
    }

    #[test]
    fn test_register_same_callback_under_two_ids_errors() {
        let mut m = SnapshotCallbackManifest::new();
        let c = cb();
        m.register("a", c.clone()).unwrap();
        let err = m.register("b", c).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("already registered under id `a`"),
            "got: {msg}"
        );
    }

    #[test]
    fn test_register_rejects_empty_id() {
        let mut m = SnapshotCallbackManifest::new();
        let err = m.register("", cb()).unwrap_err();
        assert!(format!("{err}").contains("must not be empty"));
    }

    #[test]
    fn test_register_rejects_non_printable_id() {
        let mut m = SnapshotCallbackManifest::new();
        let err = m.register("bad\tid", cb()).unwrap_err();
        assert!(format!("{err}").contains("non-printable"));
    }

    #[test]
    fn test_register_rejects_oversized_id() {
        let mut m = SnapshotCallbackManifest::new();
        let long = "a".repeat(MAX_CALLBACK_ID_LEN + 1);
        let err = m.register(&long, cb()).unwrap_err();
        assert!(format!("{err}").contains("exceeds"));
    }

    #[test]
    fn test_sorted_ids() {
        let mut m = SnapshotCallbackManifest::new();
        m.register("zeta", cb()).unwrap();
        m.register("alpha", cb()).unwrap();
        m.register("middle", cb()).unwrap();
        assert_eq!(m.sorted_ids(), vec!["alpha", "middle", "zeta"]);
    }

    #[test]
    fn test_digest_deterministic_independent_of_insertion_order() {
        let mut a = SnapshotCallbackManifest::new();
        a.register("edge.fetch", cb()).unwrap();
        a.register("edge.console.log", cb()).unwrap();
        a.register("edge.crypto.random", cb()).unwrap();

        let mut b = SnapshotCallbackManifest::new();
        b.register("edge.crypto.random", cb()).unwrap();
        b.register("edge.console.log", cb()).unwrap();
        b.register("edge.fetch", cb()).unwrap();

        assert_eq!(a.digest(), b.digest());
    }

    #[test]
    fn test_digest_changes_with_added_id() {
        let mut a = SnapshotCallbackManifest::new();
        a.register("edge.foo", cb()).unwrap();
        let d0 = a.digest();
        a.register("edge.bar", cb()).unwrap();
        assert_ne!(d0, a.digest());
    }

    #[test]
    fn test_digest_changes_with_renamed_id() {
        let mut a = SnapshotCallbackManifest::new();
        a.register("edge.foo", cb()).unwrap();
        let mut b = SnapshotCallbackManifest::new();
        b.register("edge.foo2", cb()).unwrap();
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn test_empty_manifest_digest_is_stable() {
        let a = SnapshotCallbackManifest::new();
        let b = SnapshotCallbackManifest::new();
        assert_eq!(a.digest(), b.digest());
        // Sanity: empty digest is not all-zero (so a corrupt all-zero
        // header is detectably wrong).
        assert_ne!(a.digest(), [0u8; MANIFEST_DIGEST_LEN]);
    }
}
