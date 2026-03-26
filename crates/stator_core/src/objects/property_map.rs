//! A property map that stores ECMAScript property values alongside their
//! attribute flags (`[[Writable]]`, `[[Enumerable]]`, `[[Configurable]]`).
//!
//! [`PropertyMap`] is the backing store for [`JsValue::PlainObject`] and
//! provides a HashMap-like API that transparently attaches
//! [`PropertyAttributes`] to every entry.  Newly inserted properties default
//! to `WRITABLE | ENUMERABLE | CONFIGURABLE` (the ECMAScript default for
//! ordinary user-created data properties).
//!
//! ## ECMAScript enumeration order (§10.1.11)
//!
//! Properties are stored in **ECMAScript enumeration order**: integer-indexed
//! keys (array indices `0 ..= 2^32 − 2`) occupy the front of the storage in
//! ascending numeric order, followed by non-index string keys in the order
//! they were first inserted.  All iteration methods (`keys`, `iter`,
//! `enumerable_keys`, `iter_with_attrs`) naturally produce this order.
//!
//! Internally, property values are stored in a flat `Vec<JsValue>` for
//! cache-friendly iteration and O(1) slot-based access. Small objects keep
//! name lookup in a compact linear-scan mode that searches the key vector
//! directly, while larger objects promote to a secondary `HashMap<Rc<str>,
//! usize>` for O(1) slot lookup.
//!
//! An [`INLINE_CACHE_CAP`]-entry inline cache of recently accessed property
//! names sits in front of the lookup path to avoid repeated scans or
//! hash-table probes for hot property names.

use std::array;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use crate::builtins::symbol::{is_symbol_property_key, property_key_to_symbol};
use crate::objects::map::PropertyAttributes;
use crate::objects::value::JsValue;

/// Internal storage key used for an object's prototype link.
pub const INTERNAL_PROTO_PROPERTY_KEY: &str = "\0stator.internal.proto";
/// User-visible `__proto__` key used for prototype chain lookup.
const USER_VISIBLE_PROTO_PROPERTY_KEY: &str = "__proto__";

thread_local! {
    /// Monotonically-increasing counter used to assign unique shape
    /// identifiers to each [`PropertyMap`] structural configuration.
    static NEXT_SHAPE_ID: Cell<u64> = const { Cell::new(1) };
    /// Epoch counter incremented on every prototype-relevant mutation.
    /// Used to cheaply detect when a cached `proto_generation` may be stale.
    static GLOBAL_PROTO_MUTATION_EPOCH: Cell<u64> = const { Cell::new(0) };
}

/// Default attributes for properties created by ordinary JS assignment:
/// writable, enumerable, and configurable.
const DEFAULT_ATTRS: PropertyAttributes = PropertyAttributes::from_bits_truncate(
    PropertyAttributes::WRITABLE.bits()
        | PropertyAttributes::ENUMERABLE.bits()
        | PropertyAttributes::CONFIGURABLE.bits(),
);

/// Attributes for built-in methods per ES spec §10.4.7: writable,
/// non-enumerable, configurable.
const BUILTIN_ATTRS: PropertyAttributes = PropertyAttributes::from_bits_truncate(
    PropertyAttributes::WRITABLE.bits() | PropertyAttributes::CONFIGURABLE.bits(),
);

/// Maximum number of entries in the inline property-name cache.
///
/// Sixteen entries provide better coverage for moderately polymorphic property
/// access patterns while keeping probes short and cache-friendly.
const INLINE_CACHE_CAP: usize = 16;

/// Small objects keep name lookup in compact linear-scan mode until they grow
/// beyond this many properties.
const SMALL_PROPERTY_LINEAR_SCAN_CAP: usize = 6;

/// Number of reusable property-storage buffers kept per thread.
const PROPERTY_STORAGE_POOL_CAP: usize = 64;

/// Avoid retaining very large backing allocations in the pool.
const MAX_POOLED_PROPERTY_CAPACITY: usize = SMALL_PROPERTY_LINEAR_SCAN_CAP * 4;

thread_local! {
    static PROPERTY_STORAGE_POOL: RefCell<Vec<PropertyStorageBuffers>> =
        RefCell::new(Vec::with_capacity(PROPERTY_STORAGE_POOL_CAP));
}

#[derive(Debug)]
struct PropertyStorageBuffers {
    keys: Vec<Rc<str>>,
    values: Vec<JsValue>,
    attrs: Vec<PropertyAttributes>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
enum PropertyIndex {
    #[default]
    Inline,
    Map(HashMap<Rc<str>, usize>),
}

#[derive(Debug)]
struct InlinePropertyCache {
    hashes: [Cell<u64>; INLINE_CACHE_CAP],
    slots: [Cell<u32>; INLINE_CACHE_CAP],
}

impl InlinePropertyCache {
    fn new() -> Self {
        Self {
            hashes: Default::default(),
            slots: array::from_fn(|_| Cell::new(u32::MAX)),
        }
    }
}

impl Clone for InlinePropertyCache {
    fn clone(&self) -> Self {
        Self {
            hashes: array::from_fn(|i| Cell::new(self.hashes[i].get())),
            slots: array::from_fn(|i| Cell::new(self.slots[i].get())),
        }
    }
}

#[inline]
fn next_shape_id() -> u64 {
    NEXT_SHAPE_ID.with(|next| {
        let shape_id = next.get();
        next.set(shape_id.wrapping_add(1));
        shape_id
    })
}

#[inline]
fn current_global_proto_mutation_epoch() -> u64 {
    GLOBAL_PROTO_MUTATION_EPOCH.with(Cell::get)
}

#[inline]
fn bump_global_proto_mutation_epoch() {
    GLOBAL_PROTO_MUTATION_EPOCH.with(|epoch| {
        epoch.set(epoch.get().wrapping_add(1));
    });
}

fn acquire_storage_buffers(capacity: usize) -> PropertyStorageBuffers {
    PROPERTY_STORAGE_POOL
        .try_with(|pool| {
            let mut pool = pool.borrow_mut();
            if let Some(index) = pool.iter().position(|buffers| {
                buffers.keys.capacity() >= capacity
                    && buffers.values.capacity() >= capacity
                    && buffers.attrs.capacity() >= capacity
            }) {
                let mut buffers = pool.swap_remove(index);
                buffers.keys.clear();
                buffers.values.clear();
                buffers.attrs.clear();
                buffers
            } else {
                PropertyStorageBuffers {
                    keys: Vec::with_capacity(capacity),
                    values: Vec::with_capacity(capacity),
                    attrs: Vec::with_capacity(capacity),
                }
            }
        })
        .unwrap_or_else(|_| PropertyStorageBuffers {
            keys: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
            attrs: Vec::with_capacity(capacity),
        })
}

fn release_storage_buffers(mut buffers: PropertyStorageBuffers) {
    if buffers.keys.capacity() > MAX_POOLED_PROPERTY_CAPACITY
        || buffers.values.capacity() > MAX_POOLED_PROPERTY_CAPACITY
        || buffers.attrs.capacity() > MAX_POOLED_PROPERTY_CAPACITY
    {
        return;
    }

    buffers.keys.clear();
    buffers.values.clear();
    buffers.attrs.clear();

    let _ = PROPERTY_STORAGE_POOL.try_with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < PROPERTY_STORAGE_POOL_CAP {
            pool.push(buffers);
        }
    });
}

/// Returns `Some(n)` if `key` is a valid ECMAScript array index — a canonical
/// decimal string representing an integer in `0 ..= 2^32 − 2`.
///
/// Per ECMA-262 §6.1.7, an integer index is a String value that is a canonical
/// numeric string (no leading zeros except `"0"` itself) whose numeric value
/// *i* satisfies `0 ≤ i < 2^32 − 1`.
#[inline]
fn parse_integer_index(key: &str) -> Option<u32> {
    if key.is_empty() || (key.len() > 1 && key.as_bytes()[0] == b'0') {
        return None;
    }
    let n: u32 = key.parse().ok()?;
    // Array indices are 0 ..= 2^32 − 2 (u32::MAX is *not* an index).
    if n < u32::MAX { Some(n) } else { None }
}

/// Cheap 64-bit FNV-1a hash for inline-cache probing.
///
/// This is intentionally *not* `SipHash`: it trades collision resistance for
/// raw speed on the short property names typical of JavaScript.
#[inline]
fn name_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Deterministically hashes a property layout so structurally identical maps
/// share the same layout identifier.
#[inline]
fn layout_hash(keys: &[Rc<str>], attrs: &[PropertyAttributes]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (key, attr) in keys.iter().zip(attrs.iter()) {
        for b in key.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x0100_0000_01b3);
        for b in attr.bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h ^= keys.len() as u64;
    h.wrapping_mul(0x0100_0000_01b3)
}

/// A map of named properties with ECMAScript attribute flags.
///
/// Property values are stored in a flat `Vec` in ECMAScript enumeration
/// order (integer indices ascending, then string keys in insertion order)
/// for cache-friendly, spec-compliant iteration. Small objects use compact
/// linear scans over the key vector for lookups; larger objects promote to a
/// `HashMap` of name-to-slot offsets.
///
/// An [`INLINE_CACHE_CAP`]-entry inline cache of recently accessed property
/// name hashes sits in front of the lookup path to avoid repeated scans or
/// hash-table probes on hot property names. The cache uses
/// [`Cell`]-based interior mutability so that read-only (`&self`) lookups
/// can still populate the cache.
#[derive(Debug)]
pub struct PropertyMap {
    /// Property names in ECMAScript enumeration order.
    keys: Vec<Rc<str>>,
    /// Property values, one per key.
    values: Vec<JsValue>,
    /// Property attributes, one per key.
    attrs: Vec<PropertyAttributes>,
    /// Number of integer-indexed keys stored at the front of `keys`.
    integer_key_count: usize,
    /// Name → slot-index mapping, kept inline for small objects and promoted
    /// to a hash map once the object grows beyond
    /// [`SMALL_PROPERTY_LINEAR_SCAN_CAP`] properties.
    index: PropertyIndex,
    /// Optional inline cache of recently accessed property names.
    ///
    /// Small objects stay cache-free because their linear scans are already
    /// cheap and the cache state would dominate the object footprint.
    inline_cache: Option<Box<InlinePropertyCache>>,
    /// Shape identifier — a monotonically-increasing stamp that changes on
    /// every structural mutation (property add/remove or attribute change).
    shape_id: u64,
    /// Deterministic layout identifier shared by maps with the same property
    /// names, insertion order, and attributes.
    layout_id: u64,
    /// Cached prototype-chain generation for this map.
    proto_generation: Cell<u32>,
    /// Local mutation generation that contributes to `proto_generation`.
    proto_epoch: Cell<u32>,
    /// Global prototype-mutation epoch used to validate `proto_generation`.
    proto_global_epoch: Cell<u64>,
    /// Whether new properties may be added to this object (§10.1 `[[Extensible]]`).
    pub extensible: bool,
    /// Whether this map contains any accessor properties (`__get_*__` or `__set_*__` keys).
    /// Once set to `true` it is never cleared, so it is a conservative over-approximation.
    pub has_accessors: bool,
    /// Offset of the next property expected to be filled when this map was
    /// created via [`clone_shape`](Self::clone_shape),
    /// [`clone_template`](Self::clone_template), or
    /// [`from_boilerplate`](Self::from_boilerplate).
    ///
    /// When a `PropertyMap` is created from a cached object-literal
    /// template, all keys and attributes are pre-populated but values are
    /// [`JsValue::Undefined`].  This counter tracks the sequential
    /// fill position so that [`try_template_fill`](Self::try_template_fill)
    /// can write values in O(1) without hash lookups.
    ///
    /// A value of `usize::MAX` means this map is **not** in template-fill
    /// mode (the default for non-template-created maps).
    template_next_slot: usize,
    /// Original requested capacity for this map, preserved so shape clones
    /// can pre-size their secondary index consistently.
    capacity_hint: usize,
}

/// Cached object-literal layout used to instantiate repeated literals without
/// re-observing the first instance's values.
#[derive(Debug, Clone)]
pub(crate) struct ObjectLiteralTemplate {
    keys: Box<[Rc<str>]>,
    attrs: Box<[PropertyAttributes]>,
    integer_key_count: usize,
    layout_id: u64,
    extensible: bool,
    has_accessors: bool,
    capacity_hint: usize,
}

#[derive(Debug, Clone, Copy)]
struct ShapeMetadata {
    integer_key_count: usize,
    layout_id: u64,
    extensible: bool,
    has_accessors: bool,
    capacity_hint: usize,
}

impl ObjectLiteralTemplate {
    pub(crate) fn capture(map: &PropertyMap) -> Option<Self> {
        (!map.is_empty()).then(|| Self {
            keys: map.keys.clone().into_boxed_slice(),
            attrs: map.attrs.clone().into_boxed_slice(),
            integer_key_count: map.integer_key_count,
            layout_id: map.layout_id,
            extensible: map.extensible,
            has_accessors: map.has_accessors,
            capacity_hint: map.capacity_hint,
        })
    }

    pub(crate) fn instantiate(&self) -> PropertyMap {
        PropertyMap::from_shape_parts(
            &self.keys,
            &self.attrs,
            ShapeMetadata {
                integer_key_count: self.integer_key_count,
                layout_id: self.layout_id,
                extensible: self.extensible,
                has_accessors: self.has_accessors,
                capacity_hint: self.capacity_hint,
            },
            0,
        )
    }
}

impl PartialEq for PropertyMap {
    fn eq(&self, other: &Self) -> bool {
        self.keys == other.keys
            && self.values == other.values
            && self.attrs == other.attrs
            && self.integer_key_count == other.integer_key_count
            && self.index == other.index
            && self.extensible == other.extensible
            && self.has_accessors == other.has_accessors
    }
}

impl Clone for PropertyMap {
    fn clone(&self) -> Self {
        Self {
            keys: self.keys.iter().map(Rc::clone).collect(),
            values: self.values.clone(),
            attrs: self.attrs.clone(),
            integer_key_count: self.integer_key_count,
            index: self.index.clone(),
            inline_cache: self
                .inline_cache
                .as_ref()
                .map(|cache| Box::new((**cache).clone())),
            shape_id: self.shape_id,
            layout_id: self.layout_id,
            proto_generation: Cell::new(self.proto_generation.get()),
            proto_epoch: Cell::new(self.proto_epoch.get()),
            proto_global_epoch: Cell::new(self.proto_global_epoch.get()),
            extensible: self.extensible,
            has_accessors: self.has_accessors,
            template_next_slot: self.template_next_slot,
            capacity_hint: self.capacity_hint,
        }
    }
}

impl PropertyMap {
    /// Creates an empty property map.
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    fn from_buffers(capacity: usize, buffers: PropertyStorageBuffers) -> Self {
        let index = Self::build_index_from_keys(&[], capacity);

        Self {
            keys: buffers.keys,
            values: buffers.values,
            attrs: buffers.attrs,
            integer_key_count: 0,
            index,
            inline_cache: None,
            shape_id: next_shape_id(),
            layout_id: layout_hash(&[], &[]),
            proto_generation: Cell::new(0),
            proto_epoch: Cell::new(0),
            proto_global_epoch: Cell::new(0),
            extensible: true,
            has_accessors: false,
            template_next_slot: usize::MAX,
            capacity_hint: capacity,
        }
    }

    /// Creates a property map pre-allocated for `capacity` entries.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::from_buffers(capacity, acquire_storage_buffers(capacity))
    }

    /// Creates a property map pre-populated with the given boilerplate
    /// keys and attribute flags.  All values are initialised to
    /// [`JsValue::Undefined`] and are expected to be overwritten by the
    /// constructor body. The returned map starts in template-fill mode.
    pub fn from_boilerplate(
        keys: &[Rc<str>],
        attrs: &[crate::objects::map::PropertyAttributes],
    ) -> Self {
        debug_assert_eq!(keys.len(), attrs.len());
        Self::from_shape_parts(
            keys,
            attrs,
            ShapeMetadata {
                integer_key_count: keys
                    .iter()
                    .filter(|key| parse_integer_index(key).is_some())
                    .count(),
                layout_id: layout_hash(keys, attrs),
                extensible: true,
                has_accessors: keys
                    .iter()
                    .any(|key| key.starts_with("__get_") || key.starts_with("__set_")),
                capacity_hint: keys.len(),
            },
            0,
        )
    }
    /// Returns a snapshot of the property keys and their attribute flags,
    /// suitable for caching as a constructor boilerplate.
    pub fn boilerplate_snapshot(
        &self,
    ) -> (Vec<Rc<str>>, Vec<crate::objects::map::PropertyAttributes>) {
        (self.keys.clone(), self.attrs.clone())
    }

    /// Clones only the property-map shape for object/constructor templates.
    ///
    /// The returned map has the same key names, insertion order, and
    /// attribute flags as `self`, but every value slot is reset to
    /// [`JsValue::Undefined`] and a fresh shape identifier is assigned.
    /// The map is placed in *template-fill mode* so that subsequent calls
    /// to [`try_template_fill`](Self::try_template_fill) can populate
    /// values in O(1) without hash lookups.
    pub fn clone_shape(&self) -> Self {
        Self::from_shape_parts(
            &self.keys,
            &self.attrs,
            ShapeMetadata {
                integer_key_count: self.integer_key_count,
                layout_id: self.layout_id,
                extensible: self.extensible,
                has_accessors: self.has_accessors,
                capacity_hint: self.capacity_hint,
            },
            0,
        )
    }

    /// Backwards-compatible alias for shape-only template cloning.
    #[inline]
    pub fn clone_template(&self) -> Self {
        self.clone_shape()
    }

    /// Attempts to write `value` into the next expected template slot.
    ///
    /// When a `PropertyMap` is in template-fill mode (created via
    /// [`clone_shape`](Self::clone_shape),
    /// [`clone_template`](Self::clone_template), or
    /// [`from_boilerplate`](Self::from_boilerplate)), this method checks
    /// whether `key` matches the property name at the next expected
    /// sequential position.  If so, the value is written directly by
    /// offset in O(1) — no hash lookup, no shape mutation, no prototype
    /// generation bump.
    ///
    /// Returns `Ok(offset)` if the fast path succeeded and `value` was
    /// consumed.  Returns `Err(value)` if the key did not match; the
    /// unconsumed value is returned to the caller for fallback via
    /// [`insert`](Self::insert).  On mismatch the template-fill mode is
    /// permanently disabled for this map.
    #[inline]
    pub fn try_template_fill(&mut self, key: &str, value: JsValue) -> Result<usize, JsValue> {
        let slot = self.template_next_slot;
        if slot < self.keys.len() && self.keys[slot].as_ref() == key {
            self.values[slot] = value;
            self.template_next_slot = slot + 1;
            Ok(slot)
        } else {
            self.template_next_slot = usize::MAX;
            Err(value)
        }
    }

    #[inline]
    fn build_index_from_keys(keys: &[Rc<str>], capacity_hint: usize) -> PropertyIndex {
        let capacity = capacity_hint.max(keys.len());
        if capacity > SMALL_PROPERTY_LINEAR_SCAN_CAP {
            let mut map = HashMap::with_capacity(capacity);
            for (slot, key) in keys.iter().enumerate() {
                map.insert(key.clone(), slot);
            }
            PropertyIndex::Map(map)
        } else {
            PropertyIndex::Inline
        }
    }

    fn from_shape_parts(
        keys: &[Rc<str>],
        attrs: &[PropertyAttributes],
        metadata: ShapeMetadata,
        template_next_slot: usize,
    ) -> Self {
        let cap = keys.len();
        let mut buffers = acquire_storage_buffers(cap);
        buffers.keys.extend_from_slice(keys);
        buffers.values.resize(cap, JsValue::Undefined);
        buffers.attrs.extend_from_slice(attrs);

        let capacity_hint = metadata.capacity_hint.max(cap);
        let index = Self::build_index_from_keys(&buffers.keys, capacity_hint);

        Self {
            keys: buffers.keys,
            values: buffers.values,
            attrs: buffers.attrs,
            integer_key_count: metadata.integer_key_count,
            index,
            inline_cache: (cap > SMALL_PROPERTY_LINEAR_SCAN_CAP)
                .then(|| Box::new(InlinePropertyCache::new())),
            shape_id: next_shape_id(),
            layout_id: metadata.layout_id,
            proto_generation: Cell::new(0),
            proto_epoch: Cell::new(0),
            proto_global_epoch: Cell::new(0),
            extensible: metadata.extensible,
            has_accessors: metadata.has_accessors,
            template_next_slot,
            capacity_hint,
        }
    }

    // ── Inline cache helpers ──────────────────────────────────────────────

    /// Probes the inline cache for `key`, returning its slot index on hit.
    #[inline]
    fn cache_probe(&self, key: &str) -> Option<usize> {
        let cache = self.inline_cache.as_ref()?;
        let h = name_hash(key);
        let idx = (h as usize) & (INLINE_CACHE_CAP - 1);
        if cache.hashes[idx].get() != h {
            return None;
        }
        let slot = cache.slots[idx].get() as usize;
        (slot < self.keys.len() && self.keys[slot].as_ref() == key).then_some(slot)
    }

    /// Records a `(key, slot)` pair in the inline cache.
    #[inline]
    fn cache_record(&self, key: &str, slot: usize) {
        let Some(cache) = self.inline_cache.as_ref() else {
            return;
        };
        let h = name_hash(key);
        let idx = (h as usize) & (INLINE_CACHE_CAP - 1);
        cache.hashes[idx].set(h);
        cache.slots[idx].set(slot as u32);
    }

    /// Invalidates all inline cache entries.
    #[inline]
    fn cache_invalidate(&self) {
        let Some(cache) = self.inline_cache.as_ref() else {
            return;
        };
        for entry in &cache.hashes {
            entry.set(0);
        }
        for slot in &cache.slots {
            slot.set(u32::MAX);
        }
    }

    /// Assigns a fresh shape identifier, signalling that the structural layout
    /// (set of property names or their attribute flags) has changed.
    #[inline]
    fn bump_shape_id(&mut self) {
        self.shape_id = next_shape_id();
        self.layout_id = layout_hash(&self.keys, &self.attrs);
    }

    /// Bumps the local prototype mutation epoch, invalidating any cached
    /// `proto_generation` values that depend on this map's state.
    #[inline]
    fn touch_proto_generation(&self) {
        self.proto_epoch.set(self.proto_epoch.get().wrapping_add(1));
        bump_global_proto_mutation_epoch();
        self.proto_global_epoch.set(u64::MAX);
    }

    // ── Shape / offset API ───────────────────────────────────────────────

    /// Returns the current shape identifier.
    ///
    /// The value changes on every structural mutation (property add, remove,
    /// or attribute change) but is stable across value-only updates.
    #[inline]
    pub fn shape_id(&self) -> u64 {
        self.shape_id
    }

    /// Returns the deterministic property-layout identifier shared by maps
    /// with the same keys, key order, and attributes.
    #[inline]
    pub fn layout_id(&self) -> u64 {
        self.layout_id
    }

    /// Returns the cached generation for this map's prototype chain.
    ///
    /// The generation is a composite of the local mutation epoch and the
    /// inherited generation from the prototype.  The value is cached and
    /// recomputed only when the global prototype-mutation epoch changes.
    #[inline]
    pub fn proto_generation(&self) -> u32 {
        let global_epoch = current_global_proto_mutation_epoch();
        if self.proto_global_epoch.get() == global_epoch {
            return self.proto_generation.get();
        }

        let inherited_generation = self
            .get(INTERNAL_PROTO_PROPERTY_KEY)
            .or_else(|| self.get(USER_VISIBLE_PROTO_PROPERTY_KEY))
            .and_then(|value| match value {
                JsValue::PlainObject(proto) => Some(proto.borrow().proto_generation()),
                _ => None,
            })
            .unwrap_or(0);
        let generation = self.proto_epoch.get().wrapping_add(inherited_generation);
        self.proto_generation.set(generation);
        self.proto_global_epoch.set(global_epoch);
        generation
    }

    /// Returns the global epoch used to invalidate prototype-dependent caches.
    #[inline]
    pub fn global_proto_mutation_epoch() -> u64 {
        current_global_proto_mutation_epoch()
    }

    /// Returns the slot index (offset) for `key`, or `None` if absent.
    ///
    /// The offset is valid as long as `shape_id()` does not change.
    #[inline]
    pub fn offset_of(&self, key: &str) -> Option<usize> {
        self.lookup_slot(key)
    }

    #[inline]
    fn lookup_slot(&self, key: &str) -> Option<usize> {
        match &self.index {
            PropertyIndex::Inline => self
                .keys
                .iter()
                .position(|candidate| candidate.as_ref() == key),
            PropertyIndex::Map(index) => index.get(key).copied(),
        }
    }

    #[inline]
    fn lookup_slot_cached(&self, key: &str) -> Option<usize> {
        if let Some(slot) = self.cache_probe(key) {
            return Some(slot);
        }
        let slot = self.lookup_slot(key)?;
        self.cache_record(key, slot);
        Some(slot)
    }

    fn promote_index(&mut self) {
        if matches!(self.index, PropertyIndex::Map(_)) {
            return;
        }
        let mut index = HashMap::with_capacity(self.capacity_hint.max(self.keys.len()));
        for (slot, key) in self.keys.iter().enumerate() {
            index.insert(key.clone(), slot);
        }
        self.index = PropertyIndex::Map(index);
        if self.inline_cache.is_none() {
            self.inline_cache = Some(Box::new(InlinePropertyCache::new()));
        }
    }

    fn refresh_index_mode(&mut self) {
        if self.keys.len() <= SMALL_PROPERTY_LINEAR_SCAN_CAP {
            self.index = PropertyIndex::Inline;
            self.inline_cache = None;
        } else {
            self.promote_index();
        }
    }

    /// Returns the value at a raw slot offset.
    ///
    /// # Safety contract (logical)
    ///
    /// The caller must ensure that `offset` was obtained from
    /// [`offset_of`](Self::offset_of) while `shape_id()` has not changed
    /// since.
    #[inline]
    pub fn get_by_offset(&self, offset: usize) -> Option<&JsValue> {
        self.values.get(offset)
    }

    /// Returns `true` when `offset` still names `key` in the current layout.
    #[inline]
    pub fn matches_key_at_offset(&self, offset: usize, key: &str) -> bool {
        self.keys
            .get(offset)
            .is_some_and(|candidate| candidate.as_ref() == key)
    }

    /// Overwrites the value at a raw slot offset, returning `true` on
    /// success.
    ///
    /// The same validity constraint as [`get_by_offset`](Self::get_by_offset)
    /// applies.
    #[inline]
    pub fn set_by_offset(&mut self, offset: usize, value: JsValue) -> bool {
        if let Some(slot) = self.values.get_mut(offset) {
            *slot = value;
            self.touch_proto_generation();
            true
        } else {
            false
        }
    }

    /// Returns `true` if the property at `offset` has the `WRITABLE` flag.
    #[inline]
    pub fn is_writable_by_offset(&self, offset: usize) -> bool {
        self.attrs
            .get(offset)
            .is_some_and(|a| a.contains(PropertyAttributes::WRITABLE))
    }

    // ── ECMAScript enumeration-order helpers ──────────────────────────────

    /// Returns the position at which `key` should be inserted to maintain
    /// ECMAScript §10.1.11 enumeration order: integer indices (ascending)
    /// first, then non-symbol string keys in insertion order, then symbol
    /// keys in insertion order.
    fn spec_insert_pos(&self, key: &str) -> usize {
        if let Some(n) = parse_integer_index(key) {
            self.keys[..self.integer_key_count]
                .binary_search_by(|existing| {
                    parse_integer_index(existing).unwrap_or(u32::MAX).cmp(&n)
                })
                .unwrap_or_else(|pos| pos)
        } else if is_symbol_property_key(key) {
            // Symbol keys go at the very end, after all string keys.
            self.keys.len()
        } else {
            // Non-symbol string key: insert before the first symbol key
            // (which, if any, occupies the tail of the keys vec).
            let mut pos = self.keys.len();
            while pos > 0 && is_symbol_property_key(&self.keys[pos - 1]) {
                pos -= 1;
            }
            pos
        }
    }

    /// Inserts a new property at `pos`, updating indices and integer-key
    /// bookkeeping as needed.
    fn insert_new(&mut self, key: Rc<str>, value: JsValue, attrs: PropertyAttributes, pos: usize) {
        if key.starts_with("__get_") || key.starts_with("__set_") {
            self.has_accessors = true;
        }
        let is_integer_key = parse_integer_index(&key).is_some();
        self.index_shift_right(pos);
        if let PropertyIndex::Map(index) = &mut self.index {
            index.insert(key.clone(), pos);
        }
        self.keys.insert(pos, key);
        self.values.insert(pos, value);
        self.attrs.insert(pos, attrs);
        self.capacity_hint = self.capacity_hint.max(self.keys.len());
        if is_integer_key {
            self.integer_key_count += 1;
        }
        if self.keys.len() > SMALL_PROPERTY_LINEAR_SCAN_CAP {
            self.promote_index();
        }
    }

    /// Increments all HashMap index values `>= pos` by one, preparing for
    /// an element insertion at `pos`.
    fn index_shift_right(&mut self, pos: usize) {
        if let PropertyIndex::Map(index) = &mut self.index {
            for idx in index.values_mut() {
                if *idx >= pos {
                    *idx += 1;
                }
            }
        }
    }

    // ── HashMap-compatible API ────────────────────────────────────────────

    /// Returns the value for `key`, ignoring attributes.
    pub fn get(&self, key: &str) -> Option<&JsValue> {
        if let Some(slot) = self.cache_probe(key) {
            return Some(&self.values[slot]);
        }
        let slot = self.lookup_slot(key)?;
        self.cache_record(key, slot);
        Some(&self.values[slot])
    }

    /// Returns the value for an interned key by trying pointer equality before
    /// falling back to the regular string lookup path.
    pub fn get_by_rc(&self, key: &Rc<str>) -> Option<&JsValue> {
        for (slot, candidate) in self.keys.iter().enumerate() {
            if Rc::ptr_eq(candidate, key) {
                self.cache_record(key.as_ref(), slot);
                return Some(&self.values[slot]);
            }
        }
        self.get(key.as_ref())
    }

    /// Returns a clone of the value for `key`, ignoring attributes.
    pub fn get_cloned(&self, key: &str) -> Option<JsValue> {
        self.lookup_slot_cached(key)
            .map(|slot| self.values[slot].clone())
    }

    /// Returns `true` if the map contains an entry for `key`.
    pub fn contains_key(&self, key: &str) -> bool {
        self.lookup_slot_cached(key).is_some()
    }

    /// Returns `true` if an accessor getter `__get_{key}__` exists in this
    /// map.  Uses a stack-allocated buffer to avoid heap `format!()`.
    #[inline]
    pub fn has_getter_for(&self, key: &str) -> bool {
        if !self.has_accessors {
            return false;
        }
        // Build "__get_{key}__" on the stack (max 128 bytes, fall back to
        // heap allocation for very long property names).
        let prefix = "__get_";
        let suffix = "__";
        let total = prefix.len() + key.len() + suffix.len();
        if total <= 128 {
            let mut buf = [0u8; 128];
            buf[..prefix.len()].copy_from_slice(prefix.as_bytes());
            buf[prefix.len()..prefix.len() + key.len()].copy_from_slice(key.as_bytes());
            buf[prefix.len() + key.len()..total].copy_from_slice(suffix.as_bytes());
            // SAFETY: both prefix, key, and suffix are valid UTF-8.
            let getter_key = unsafe { std::str::from_utf8_unchecked(&buf[..total]) };
            self.contains_key(getter_key)
        } else {
            self.contains_key(&format!("__get_{key}__"))
        }
    }

    /// Returns `true` if an accessor setter `__set_{key}__` exists in this
    /// map.  Uses a stack-allocated buffer to avoid heap `format!()`.
    #[inline]
    pub fn has_setter_for(&self, key: &str) -> bool {
        if !self.has_accessors {
            return false;
        }
        let prefix = "__set_";
        let suffix = "__";
        let total = prefix.len() + key.len() + suffix.len();
        if total <= 128 {
            let mut buf = [0u8; 128];
            buf[..prefix.len()].copy_from_slice(prefix.as_bytes());
            buf[prefix.len()..prefix.len() + key.len()].copy_from_slice(key.as_bytes());
            buf[prefix.len() + key.len()..total].copy_from_slice(suffix.as_bytes());
            // SAFETY: both prefix, key, and suffix are valid UTF-8.
            let setter_key = unsafe { std::str::from_utf8_unchecked(&buf[..total]) };
            self.contains_key(setter_key)
        } else {
            self.contains_key(&format!("__set_{key}__"))
        }
    }

    /// Returns the getter function for `key` (`__get_{key}__`) if one
    /// exists.  Stack-allocated key avoids heap `format!()`.
    #[inline]
    pub fn get_getter_for(&self, key: &str) -> Option<&JsValue> {
        if !self.has_accessors {
            return None;
        }
        let prefix = "__get_";
        let suffix = "__";
        let total = prefix.len() + key.len() + suffix.len();
        if total <= 128 {
            let mut buf = [0u8; 128];
            buf[..prefix.len()].copy_from_slice(prefix.as_bytes());
            buf[prefix.len()..prefix.len() + key.len()].copy_from_slice(key.as_bytes());
            buf[prefix.len() + key.len()..total].copy_from_slice(suffix.as_bytes());
            // SAFETY: both prefix, key, and suffix are valid UTF-8.
            let getter_key = unsafe { std::str::from_utf8_unchecked(&buf[..total]) };
            self.get(getter_key)
        } else {
            self.get(&format!("__get_{key}__"))
        }
    }

    /// Returns the setter function for `key` (`__set_{key}__`) if one
    /// exists.  Stack-allocated key avoids heap `format!()`.
    #[inline]
    pub fn get_setter_for(&self, key: &str) -> Option<&JsValue> {
        if !self.has_accessors {
            return None;
        }
        let prefix = "__set_";
        let suffix = "__";
        let total = prefix.len() + key.len() + suffix.len();
        if total <= 128 {
            let mut buf = [0u8; 128];
            buf[..prefix.len()].copy_from_slice(prefix.as_bytes());
            buf[prefix.len()..prefix.len() + key.len()].copy_from_slice(key.as_bytes());
            buf[prefix.len() + key.len()..total].copy_from_slice(suffix.as_bytes());
            // SAFETY: both prefix, key, and suffix are valid UTF-8.
            let setter_key = unsafe { std::str::from_utf8_unchecked(&buf[..total]) };
            self.get(setter_key)
        } else {
            self.get(&format!("__set_{key}__"))
        }
    }

    /// Inserts a property with default attributes (writable, enumerable,
    /// configurable).  If the key already exists the value is replaced but
    /// the existing attributes are preserved.
    ///
    /// New properties are placed according to ECMAScript enumeration order:
    /// integer-indexed keys occupy the front of the storage (sorted
    /// numerically), followed by string keys in insertion order.
    pub fn insert(&mut self, key: String, value: JsValue) {
        self.insert_rc(key.into(), value);
    }

    /// Inserts a property using an already-interned key.
    pub fn insert_rc(&mut self, key: Rc<str>, value: JsValue) {
        if let Some(i) = self.lookup_slot_cached(&key) {
            self.values[i] = value;
            self.touch_proto_generation();
        } else {
            // Non-extensible objects reject new properties (except internal __dunder__ keys).
            if !self.extensible && !key.starts_with("__") {
                return;
            }
            // The internal prototype link must never be enumerable.
            let attrs = if key.as_ref() == INTERNAL_PROTO_PROPERTY_KEY {
                BUILTIN_ATTRS
            } else {
                DEFAULT_ATTRS
            };
            let pos = self.spec_insert_pos(&key);
            self.insert_new(key, value, attrs, pos);
            self.bump_shape_id();
            if pos != self.keys.len() - 1 {
                self.cache_invalidate();
            }
            self.touch_proto_generation();
        }
    }

    /// Insert a built-in method or constructor property (writable,
    /// non-enumerable, configurable — per ES spec).
    pub fn insert_builtin(&mut self, key: String, value: JsValue) {
        self.insert_builtin_rc(key.into(), value);
    }

    /// Inserts a built-in property using an already-interned key.
    pub fn insert_builtin_rc(&mut self, key: Rc<str>, value: JsValue) {
        if let Some(i) = self.lookup_slot_cached(&key) {
            self.values[i] = value;
            self.attrs[i] = BUILTIN_ATTRS;
            self.touch_proto_generation();
        } else {
            let pos = self.spec_insert_pos(&key);
            self.insert_new(key, value, BUILTIN_ATTRS, pos);
            self.bump_shape_id();
            if pos != self.keys.len() - 1 {
                self.cache_invalidate();
            }
            self.touch_proto_generation();
        }
    }

    /// Set all existing properties to non-enumerable.
    ///
    /// Called after populating built-in prototype objects whose methods
    /// should not appear in `for…in` or `Object.keys()`.
    pub fn make_all_non_enumerable(&mut self) {
        for attr in &mut self.attrs {
            attr.remove(PropertyAttributes::ENUMERABLE);
        }
        if !self.attrs.is_empty() {
            self.bump_shape_id();
            self.touch_proto_generation();
        }
    }

    /// Removes the entry for `key`, returning the old value (if any).
    ///
    /// Uses an order-preserving shift-remove so that the ECMAScript
    /// enumeration order of the remaining properties is maintained.
    pub fn remove(&mut self, key: &str) -> Option<JsValue> {
        if let Some(i) = self.lookup_slot(key) {
            if let PropertyIndex::Map(index) = &mut self.index {
                index.remove(key);
            }
            if parse_integer_index(&self.keys[i]).is_some() {
                self.integer_key_count -= 1;
            }
            let val = self.values[i].clone();
            self.keys.remove(i);
            self.values.remove(i);
            self.attrs.remove(i);
            // Decrement indices for every slot that shifted left.
            if let PropertyIndex::Map(index) = &mut self.index {
                for idx in index.values_mut() {
                    if *idx > i {
                        *idx -= 1;
                    }
                }
            }
            self.refresh_index_mode();
            self.bump_shape_id();
            self.cache_invalidate();
            self.touch_proto_generation();
            Some(val)
        } else {
            None
        }
    }

    /// Returns an iterator over the property keys.
    pub fn keys(&self) -> impl Iterator<Item = &Rc<str>> {
        self.keys.iter()
    }

    /// Returns an iterator over `(key, value)` pairs, ignoring attributes.
    pub fn iter(&self) -> impl Iterator<Item = (&Rc<str>, &JsValue)> {
        self.keys.iter().zip(self.values.iter())
    }

    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Returns `true` if the map contains no entries.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    // ── Attribute-aware API ───────────────────────────────────────────────

    /// Returns the value and attribute flags for `key`.
    pub fn get_with_attrs(&self, key: &str) -> Option<(&JsValue, PropertyAttributes)> {
        self.lookup_slot_cached(key)
            .map(|slot| (&self.values[slot], self.attrs[slot]))
    }

    /// Inserts a property with explicit attribute flags.
    ///
    /// New properties are placed according to ECMAScript enumeration order
    /// (see [`insert`][Self::insert]).
    pub fn insert_with_attrs(&mut self, key: String, value: JsValue, attrs: PropertyAttributes) {
        self.insert_with_attrs_rc(key.into(), value, attrs);
    }

    /// Inserts a property with explicit attributes using an already-interned key.
    pub fn insert_with_attrs_rc(
        &mut self,
        key: Rc<str>,
        value: JsValue,
        attrs: PropertyAttributes,
    ) {
        if let Some(i) = self.lookup_slot_cached(&key) {
            self.values[i] = value;
            self.attrs[i] = attrs;
            self.touch_proto_generation();
        } else {
            let pos = self.spec_insert_pos(&key);
            self.insert_new(key, value, attrs, pos);
            self.bump_shape_id();
            if pos != self.keys.len() - 1 {
                self.cache_invalidate();
            }
            self.touch_proto_generation();
        }
    }

    /// Updates the attribute flags for an existing property.
    /// Returns `true` if the property existed and was updated.
    pub fn set_attrs(&mut self, key: &str, attrs: PropertyAttributes) -> bool {
        if let Some(i) = self.lookup_slot_cached(key) {
            self.attrs[i] = attrs;
            self.bump_shape_id();
            self.touch_proto_generation();
            true
        } else {
            false
        }
    }

    /// Returns the attribute flags for `key`, or `None` if absent.
    pub fn attrs(&self, key: &str) -> Option<PropertyAttributes> {
        self.lookup_slot_cached(key).map(|slot| self.attrs[slot])
    }

    /// Returns `true` if the property exists and is writable.
    pub fn is_writable(&self, key: &str) -> bool {
        self.lookup_slot_cached(key)
            .map(|i| self.attrs[i].contains(PropertyAttributes::WRITABLE))
            .unwrap_or(false)
    }

    /// Returns `true` if the property exists and is configurable.
    pub fn is_configurable(&self, key: &str) -> bool {
        self.lookup_slot_cached(key)
            .map(|i| self.attrs[i].contains(PropertyAttributes::CONFIGURABLE))
            .unwrap_or(false)
    }

    /// Returns `true` if the property exists and is enumerable.
    pub fn is_enumerable(&self, key: &str) -> bool {
        self.lookup_slot_cached(key)
            .map(|i| self.attrs[i].contains(PropertyAttributes::ENUMERABLE))
            .unwrap_or(false)
    }

    /// Set or clear the `WRITABLE` flag for an existing property.
    pub fn set_writable(&mut self, key: &str, writable: bool) {
        if let Some(i) = self.lookup_slot_cached(key) {
            if writable {
                self.attrs[i].insert(PropertyAttributes::WRITABLE);
            } else {
                self.attrs[i].remove(PropertyAttributes::WRITABLE);
            }
            self.bump_shape_id();
            self.touch_proto_generation();
        }
    }

    /// Set or clear the `ENUMERABLE` flag for an existing property.
    pub fn set_enumerable(&mut self, key: &str, enumerable: bool) {
        if let Some(i) = self.lookup_slot_cached(key) {
            if enumerable {
                self.attrs[i].insert(PropertyAttributes::ENUMERABLE);
            } else {
                self.attrs[i].remove(PropertyAttributes::ENUMERABLE);
            }
            self.bump_shape_id();
            self.touch_proto_generation();
        }
    }

    /// Set or clear the `CONFIGURABLE` flag for an existing property.
    pub fn set_configurable(&mut self, key: &str, configurable: bool) {
        if let Some(i) = self.lookup_slot_cached(key) {
            if configurable {
                self.attrs[i].insert(PropertyAttributes::CONFIGURABLE);
            } else {
                self.attrs[i].remove(PropertyAttributes::CONFIGURABLE);
            }
            self.bump_shape_id();
            self.touch_proto_generation();
        }
    }

    /// Returns an iterator over only the enumerable property keys.
    pub fn enumerable_keys(&self) -> impl Iterator<Item = &Rc<str>> {
        self.keys
            .iter()
            .zip(self.attrs.iter())
            .filter(|(k, a)| {
                a.contains(PropertyAttributes::ENUMERABLE) && !is_symbol_property_key(k)
            })
            .map(|(k, _)| k)
    }

    /// Returns an iterator over `(key, value)` pairs for only enumerable
    /// properties — the set that ES `EnumerableOwnProperties` would return.
    pub fn enumerable_iter(&self) -> impl Iterator<Item = (&Rc<str>, &JsValue)> {
        self.keys
            .iter()
            .zip(self.values.iter())
            .zip(self.attrs.iter())
            .filter(|((k, _), a)| {
                a.contains(PropertyAttributes::ENUMERABLE) && !is_symbol_property_key(k)
            })
            .map(|((k, v), _)| (k, v))
    }

    /// Returns an iterator over `(key, value, attrs)` triples.
    pub fn iter_with_attrs(
        &self,
    ) -> impl Iterator<Item = (&Rc<str>, &JsValue, PropertyAttributes)> {
        self.keys
            .iter()
            .zip(self.values.iter())
            .zip(self.attrs.iter())
            .map(|((k, v), a)| (k, v, *a))
    }

    /// Returns the own symbol-keyed property identifiers in insertion order.
    pub fn own_symbol_keys(&self) -> Vec<u64> {
        self.keys
            .iter()
            .filter_map(|key| property_key_to_symbol(key))
            .collect()
    }

    /// Mark all properties as non-writable and non-configurable, and prevent
    /// new properties from being added (ES §20.1.2.6).
    pub fn freeze(&mut self) {
        for a in &mut self.attrs {
            a.remove(PropertyAttributes::WRITABLE);
            a.remove(PropertyAttributes::CONFIGURABLE);
        }
        self.extensible = false;
        self.bump_shape_id();
        self.touch_proto_generation();
    }

    /// Returns `true` if the object is frozen: non-extensible with all
    /// properties non-writable and non-configurable (ES §20.1.2.15).
    pub fn is_frozen(&self) -> bool {
        if self.extensible {
            return false;
        }
        self.attrs.iter().all(|a| {
            !a.contains(PropertyAttributes::WRITABLE)
                && !a.contains(PropertyAttributes::CONFIGURABLE)
        })
    }

    /// Mark all properties as non-configurable and prevent new properties
    /// from being added (ES §20.1.2.20).
    pub fn seal(&mut self) {
        for a in &mut self.attrs {
            a.remove(PropertyAttributes::CONFIGURABLE);
        }
        self.extensible = false;
        self.bump_shape_id();
        self.touch_proto_generation();
    }

    /// Returns `true` if the object is sealed: non-extensible with all
    /// properties non-configurable (ES §20.1.2.16).
    pub fn is_sealed(&self) -> bool {
        if self.extensible {
            return false;
        }
        self.attrs
            .iter()
            .all(|a| !a.contains(PropertyAttributes::CONFIGURABLE))
    }
}

impl Default for PropertyMap {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for PropertyMap {
    fn drop(&mut self) {
        let buffers = PropertyStorageBuffers {
            keys: std::mem::take(&mut self.keys),
            values: std::mem::take(&mut self.values),
            attrs: std::mem::take(&mut self.attrs),
        };
        release_storage_buffers(buffers);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_get_default_attrs() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(42));
        assert_eq!(pm.get("x"), Some(&JsValue::Smi(42)));
        let attrs = pm.attrs("x").unwrap();
        assert!(attrs.contains(PropertyAttributes::WRITABLE));
        assert!(attrs.contains(PropertyAttributes::ENUMERABLE));
        assert!(attrs.contains(PropertyAttributes::CONFIGURABLE));
    }

    #[test]
    fn test_insert_with_attrs() {
        let mut pm = PropertyMap::new();
        pm.insert_with_attrs(
            "ro".to_string(),
            JsValue::Smi(1),
            PropertyAttributes::empty(),
        );
        assert!(!pm.is_writable("ro"));
        assert!(!pm.is_enumerable("ro"));
        assert!(!pm.is_configurable("ro"));
    }

    #[test]
    fn test_insert_preserves_existing_attrs() {
        let mut pm = PropertyMap::new();
        pm.insert_with_attrs(
            "p".to_string(),
            JsValue::Smi(1),
            PropertyAttributes::ENUMERABLE,
        );
        // Re-insert with default insert — should preserve ENUMERABLE-only.
        pm.insert("p".to_string(), JsValue::Smi(2));
        assert_eq!(pm.get("p"), Some(&JsValue::Smi(2)));
        let attrs = pm.attrs("p").unwrap();
        assert_eq!(attrs, PropertyAttributes::ENUMERABLE);
    }

    #[test]
    fn test_remove() {
        let mut pm = PropertyMap::new();
        pm.insert("a".to_string(), JsValue::Boolean(true));
        assert!(pm.contains_key("a"));
        let removed = pm.remove("a");
        assert_eq!(removed, Some(JsValue::Boolean(true)));
        assert!(!pm.contains_key("a"));
    }

    #[test]
    fn test_enumerable_keys() {
        let mut pm = PropertyMap::new();
        pm.insert("visible".to_string(), JsValue::Smi(1));
        pm.insert_with_attrs(
            "hidden".to_string(),
            JsValue::Smi(2),
            PropertyAttributes::WRITABLE,
        );
        let enum_keys: Vec<&str> = pm.enumerable_keys().map(|k| &**k).collect();
        assert!(enum_keys.contains(&"visible"));
        assert!(!enum_keys.contains(&"hidden"));
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut pm = PropertyMap::new();
        assert!(pm.is_empty());
        assert_eq!(pm.len(), 0);
        pm.insert("k".to_string(), JsValue::Null);
        assert!(!pm.is_empty());
        assert_eq!(pm.len(), 1);
    }

    #[test]
    fn test_set_attrs() {
        let mut pm = PropertyMap::new();
        pm.insert("p".to_string(), JsValue::Smi(1));
        assert!(pm.is_writable("p"));
        pm.set_attrs("p", PropertyAttributes::ENUMERABLE);
        assert!(!pm.is_writable("p"));
        assert!(pm.is_enumerable("p"));
        assert!(!pm.is_configurable("p"));
    }

    #[test]
    fn test_get_with_attrs() {
        let mut pm = PropertyMap::new();
        pm.insert_with_attrs(
            "x".to_string(),
            JsValue::Smi(5),
            PropertyAttributes::WRITABLE | PropertyAttributes::ENUMERABLE,
        );
        let (val, attrs) = pm.get_with_attrs("x").unwrap();
        assert_eq!(val, &JsValue::Smi(5));
        assert!(attrs.contains(PropertyAttributes::WRITABLE));
        assert!(attrs.contains(PropertyAttributes::ENUMERABLE));
        assert!(!attrs.contains(PropertyAttributes::CONFIGURABLE));
    }

    #[test]
    fn test_iter_with_attrs() {
        let mut pm = PropertyMap::new();
        pm.insert_with_attrs(
            "a".to_string(),
            JsValue::Smi(1),
            PropertyAttributes::WRITABLE,
        );
        let entries: Vec<_> = pm.iter_with_attrs().collect();
        assert_eq!(entries.len(), 1);
        assert_eq!(&**entries[0].0, "a");
        assert_eq!(entries[0].1, &JsValue::Smi(1));
        assert_eq!(entries[0].2, PropertyAttributes::WRITABLE);
    }

    #[test]
    fn test_enumerable_keys_skip_symbol_keys() {
        let mut pm = PropertyMap::new();
        pm.insert("visible".to_string(), JsValue::Smi(1));
        pm.insert(
            crate::builtins::symbol::symbol_to_property_key(123),
            JsValue::Smi(2),
        );
        let keys: Vec<&str> = pm.enumerable_keys().map(|s| &**s).collect();
        assert_eq!(keys, vec!["visible"]);
    }

    #[test]
    fn test_own_symbol_keys_returns_symbols() {
        let mut pm = PropertyMap::new();
        pm.insert(
            crate::builtins::symbol::symbol_to_property_key(321),
            JsValue::Boolean(true),
        );
        assert_eq!(pm.own_symbol_keys(), vec![321]);
    }

    #[test]
    fn test_missing_key_attr_queries() {
        let pm = PropertyMap::new();
        assert!(!pm.is_writable("nope"));
        assert!(!pm.is_enumerable("nope"));
        assert!(!pm.is_configurable("nope"));
        assert!(pm.attrs("nope").is_none());
    }

    #[test]
    fn test_with_capacity() {
        let pm = PropertyMap::with_capacity(16);
        assert!(pm.is_empty());
    }

    #[test]
    fn test_small_maps_skip_inline_cache() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(1));
        pm.insert("y".to_string(), JsValue::Smi(2));

        assert_eq!(pm.get("x"), Some(&JsValue::Smi(1)));
        assert!(pm.inline_cache.is_none());
    }

    #[test]
    fn test_clone_shape_resets_values_and_preserves_layout() {
        let mut pm = PropertyMap::with_capacity(SMALL_PROPERTY_LINEAR_SCAN_CAP + 4);
        for idx in 0..(SMALL_PROPERTY_LINEAR_SCAN_CAP + 1) {
            pm.insert(format!("k{idx}"), JsValue::Smi(idx as i32));
        }
        let shape_id = pm.shape_id();
        let layout_id = pm.layout_id();

        let cloned = pm.clone_shape();

        assert_eq!(cloned.keys, pm.keys);
        assert!(
            cloned
                .values
                .iter()
                .all(|value| matches!(value, JsValue::Undefined))
        );
        assert_eq!(cloned.attrs, pm.attrs);
        assert_ne!(cloned.shape_id(), shape_id);
        assert_eq!(cloned.layout_id(), layout_id);
        assert!(matches!(cloned.index, PropertyIndex::Map(_)));
        assert_eq!(cloned.template_next_slot, 0);
        assert_eq!(cloned.capacity_hint, pm.capacity_hint);
    }

    #[test]
    fn test_try_template_fill_returns_offset_and_disables_on_miss() {
        let mut pm = PropertyMap::from_boilerplate(
            &[Rc::from("first"), Rc::from("second")],
            &[DEFAULT_ATTRS, DEFAULT_ATTRS],
        );

        assert_eq!(pm.try_template_fill("first", JsValue::Smi(1)), Ok(0));
        assert_eq!(pm.try_template_fill("second", JsValue::Smi(2)), Ok(1));
        assert_eq!(pm.get("first"), Some(&JsValue::Smi(1)));
        assert_eq!(pm.get("second"), Some(&JsValue::Smi(2)));

        let err = pm.try_template_fill("third", JsValue::Smi(3));
        assert_eq!(err, Err(JsValue::Smi(3)));
        assert_eq!(pm.template_next_slot, usize::MAX);
    }

    // ── Inline cache tests ───────────────────────────────────────────────

    #[test]
    fn test_cache_populated_on_get() {
        let mut pm = PropertyMap::new();
        for i in 0..=SMALL_PROPERTY_LINEAR_SCAN_CAP {
            pm.insert(format!("k{i}"), JsValue::Smi(i as i32));
        }

        // First access populates the cache.
        assert_eq!(pm.get("k0"), Some(&JsValue::Smi(0)));

        // Second access of same key hits the cache.
        assert_eq!(pm.get("k0"), Some(&JsValue::Smi(0)));

        // Different key adds another cache entry.
        assert_eq!(pm.get("k1"), Some(&JsValue::Smi(1)));
        assert!(pm.inline_cache.is_some());
    }

    #[test]
    fn test_cache_invalidated_on_remove() {
        let mut pm = PropertyMap::new();
        for i in 0..=SMALL_PROPERTY_LINEAR_SCAN_CAP {
            pm.insert(format!("k{i}"), JsValue::Smi(i as i32));
        }

        // Populate cache.
        assert_eq!(pm.get("k0"), Some(&JsValue::Smi(0)));

        // Remove invalidates cache.
        pm.remove("k0");
    }

    #[test]
    fn test_cache_wraps_around() {
        let mut pm = PropertyMap::new();
        for i in 0..(INLINE_CACHE_CAP as i32 + 2) {
            pm.insert(format!("k{i}"), JsValue::Smi(i));
        }
        // Access more keys than fit in the cache so insertion wraps via the cursor.
        for i in 0..(INLINE_CACHE_CAP as i32 + 2) {
            assert_eq!(pm.get(&format!("k{i}")), Some(&JsValue::Smi(i)));
        }
        // All lookups should still work (cache or HashMap fallback).
        for i in 0..(INLINE_CACHE_CAP as i32 + 2) {
            assert_eq!(pm.get(&format!("k{i}")), Some(&JsValue::Smi(i)));
        }
    }

    #[test]
    fn test_cache_hit_moves_entry_to_front() {
        let mut pm = PropertyMap::new();
        for i in 0..=SMALL_PROPERTY_LINEAR_SCAN_CAP {
            pm.insert(format!("k{i}"), JsValue::Smi(i as i32));
        }

        assert_eq!(pm.get("k0"), Some(&JsValue::Smi(0)));
        assert_eq!(pm.get("k1"), Some(&JsValue::Smi(1)));

        // With a 16-entry direct-mapped hash cache, each key hashes to a
        // specific slot.  Verify that both are cached at their hash slots.
        let hash_a = name_hash("k0");
        let hash_b = name_hash("k1");
        let idx_a = (hash_a as usize) & (INLINE_CACHE_CAP - 1);
        let idx_b = (hash_b as usize) & (INLINE_CACHE_CAP - 1);
        let cache = pm
            .inline_cache
            .as_ref()
            .expect("large map should allocate cache");
        assert_eq!(cache.hashes[idx_a].get(), hash_a);
        assert_eq!(cache.hashes[idx_b].get(), hash_b);

        // Re-access "k1" — still at the same hash slot.
        assert_eq!(pm.get("k1"), Some(&JsValue::Smi(1)));
        assert_eq!(cache.hashes[idx_b].get(), hash_b);
    }

    #[test]
    fn test_cache_contains_key_fast_path() {
        let mut pm = PropertyMap::new();
        for i in 0..=SMALL_PROPERTY_LINEAR_SCAN_CAP {
            pm.insert(format!("k{i}"), JsValue::Smi(i as i32));
        }
        // Populate cache via get.
        let _ = pm.get("k0");
        // contains_key should hit the cache.
        assert!(pm.contains_key("k0"));
        assert!(!pm.contains_key("missing"));
    }

    #[test]
    fn test_cache_get_with_attrs_fast_path() {
        let mut pm = PropertyMap::new();
        for i in 0..SMALL_PROPERTY_LINEAR_SCAN_CAP {
            pm.insert(format!("k{i}"), JsValue::Smi(i as i32));
        }
        pm.insert_with_attrs(
            format!("k{SMALL_PROPERTY_LINEAR_SCAN_CAP}"),
            JsValue::Smi(42),
            PropertyAttributes::WRITABLE | PropertyAttributes::ENUMERABLE,
        );
        // First call: populates cache.
        let (val, attrs) = pm
            .get_with_attrs(&format!("k{SMALL_PROPERTY_LINEAR_SCAN_CAP}"))
            .unwrap();
        assert_eq!(val, &JsValue::Smi(42));
        assert!(attrs.contains(PropertyAttributes::WRITABLE));
        // Second call: should hit cache.
        let (val2, attrs2) = pm
            .get_with_attrs(&format!("k{SMALL_PROPERTY_LINEAR_SCAN_CAP}"))
            .unwrap();
        assert_eq!(val2, &JsValue::Smi(42));
        assert_eq!(attrs, attrs2);
    }

    #[test]
    fn test_cache_equality_ignores_cache_state() {
        let mut pm1 = PropertyMap::new();
        let mut pm2 = PropertyMap::new();
        pm1.insert("x".to_string(), JsValue::Smi(1));
        pm2.insert("x".to_string(), JsValue::Smi(1));
        // pm1 has a populated cache, pm2 does not.
        let _ = pm1.get("x");
        // They should still be equal.
        assert_eq!(pm1, pm2);
    }

    // ── ECMAScript enumeration-order tests ───────────────────────────────

    #[test]
    fn test_parse_integer_index() {
        assert_eq!(parse_integer_index("0"), Some(0));
        assert_eq!(parse_integer_index("1"), Some(1));
        assert_eq!(parse_integer_index("42"), Some(42));
        assert_eq!(parse_integer_index("4294967294"), Some(u32::MAX - 1));
        // u32::MAX is NOT a valid array index.
        assert_eq!(parse_integer_index("4294967295"), None);
        // Leading zeros are not canonical.
        assert_eq!(parse_integer_index("01"), None);
        assert_eq!(parse_integer_index("007"), None);
        // Non-numeric strings.
        assert_eq!(parse_integer_index(""), None);
        assert_eq!(parse_integer_index("abc"), None);
        assert_eq!(parse_integer_index("-1"), None);
        assert_eq!(parse_integer_index("1.5"), None);
    }

    #[test]
    fn test_integer_indices_sorted_before_strings() {
        let mut pm = PropertyMap::new();
        // Insert in non-spec order: strings first, then integers.
        pm.insert("b".to_string(), JsValue::Smi(1));
        pm.insert("2".to_string(), JsValue::Smi(2));
        pm.insert("a".to_string(), JsValue::Smi(3));
        pm.insert("0".to_string(), JsValue::Smi(4));
        // Expected spec order: 0, 2, b, a
        let keys: Vec<&str> = pm.keys().map(|k| &**k).collect();
        assert_eq!(keys, vec!["0", "2", "b", "a"]);
    }

    #[test]
    fn test_integer_indices_ascending_numeric_order() {
        let mut pm = PropertyMap::new();
        pm.insert("10".to_string(), JsValue::Smi(10));
        pm.insert("2".to_string(), JsValue::Smi(2));
        pm.insert("1".to_string(), JsValue::Smi(1));
        pm.insert("20".to_string(), JsValue::Smi(20));
        let keys: Vec<&str> = pm.keys().map(|s| &**s).collect();
        assert_eq!(keys, vec!["1", "2", "10", "20"]);
        assert_eq!(pm.integer_key_count, 4);
    }

    #[test]
    fn test_string_and_symbol_inserts_do_not_change_integer_key_count() {
        use crate::builtins::symbol::{symbol_create, symbol_to_property_key};

        let mut pm = PropertyMap::new();
        pm.insert("2".to_string(), JsValue::Smi(2));
        pm.insert("name".to_string(), JsValue::Smi(1));
        pm.insert(symbol_to_property_key(symbol_create(None)), JsValue::Smi(3));

        assert_eq!(pm.integer_key_count, 1);
    }

    #[test]
    fn test_string_keys_preserve_insertion_order() {
        let mut pm = PropertyMap::new();
        pm.insert("z".to_string(), JsValue::Smi(1));
        pm.insert("a".to_string(), JsValue::Smi(2));
        pm.insert("m".to_string(), JsValue::Smi(3));
        let keys: Vec<&str> = pm.keys().map(|s| &**s).collect();
        assert_eq!(keys, vec!["z", "a", "m"]);
    }

    #[test]
    fn test_mixed_integer_and_string_order() {
        let mut pm = PropertyMap::new();
        // Simulate: obj.z = 1; obj[5] = 2; obj.a = 3; obj[1] = 4; obj.m = 5; obj[3] = 6;
        pm.insert("z".to_string(), JsValue::Smi(1));
        pm.insert("5".to_string(), JsValue::Smi(2));
        pm.insert("a".to_string(), JsValue::Smi(3));
        pm.insert("1".to_string(), JsValue::Smi(4));
        pm.insert("m".to_string(), JsValue::Smi(5));
        pm.insert("3".to_string(), JsValue::Smi(6));
        // Spec: integer indices ascending, then strings in insertion order.
        let keys: Vec<&str> = pm.keys().map(|s| &**s).collect();
        assert_eq!(keys, vec!["1", "3", "5", "z", "a", "m"]);
    }

    #[test]
    fn test_remove_preserves_order() {
        let mut pm = PropertyMap::new();
        pm.insert("a".to_string(), JsValue::Smi(1));
        pm.insert("b".to_string(), JsValue::Smi(2));
        pm.insert("c".to_string(), JsValue::Smi(3));
        pm.remove("b");
        let keys: Vec<&str> = pm.keys().map(|s| &**s).collect();
        assert_eq!(keys, vec!["a", "c"]);
        // Remaining entries still accessible.
        assert_eq!(pm.get("a"), Some(&JsValue::Smi(1)));
        assert_eq!(pm.get("c"), Some(&JsValue::Smi(3)));
    }

    #[test]
    fn test_remove_preserves_spec_order() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(1));
        pm.insert("3".to_string(), JsValue::Smi(2));
        pm.insert("y".to_string(), JsValue::Smi(3));
        pm.insert("1".to_string(), JsValue::Smi(4));
        // Before remove: 1, 3, x, y
        pm.remove("3");
        let keys: Vec<&str> = pm.keys().map(|s| &**s).collect();
        assert_eq!(keys, vec!["1", "x", "y"]);
        assert_eq!(pm.integer_key_count, 1);
    }

    #[test]
    fn test_enumerable_keys_spec_order() {
        let mut pm = PropertyMap::new();
        pm.insert("b".to_string(), JsValue::Smi(1));
        pm.insert("2".to_string(), JsValue::Smi(2));
        pm.insert_with_attrs(
            "hidden".to_string(),
            JsValue::Smi(99),
            PropertyAttributes::WRITABLE, // not enumerable
        );
        pm.insert("0".to_string(), JsValue::Smi(3));
        // Enumerable keys should follow spec order, excluding "hidden".
        let keys: Vec<&str> = pm.enumerable_keys().map(|s| &**s).collect();
        assert_eq!(keys, vec!["0", "2", "b"]);
    }

    #[test]
    fn test_iter_values_match_spec_ordered_keys() {
        let mut pm = PropertyMap::new();
        pm.insert("b".to_string(), JsValue::Smi(10));
        pm.insert("1".to_string(), JsValue::Smi(20));
        pm.insert("0".to_string(), JsValue::Smi(30));
        // Spec order: 0, 1, b — values should follow.
        let pairs: Vec<(&str, &JsValue)> = pm.iter().map(|(k, v)| (&**k, v)).collect();
        assert_eq!(
            pairs,
            vec![
                ("0", &JsValue::Smi(30)),
                ("1", &JsValue::Smi(20)),
                ("b", &JsValue::Smi(10)),
            ]
        );
    }

    #[test]
    fn test_insert_existing_integer_key_no_reorder() {
        let mut pm = PropertyMap::new();
        pm.insert("1".to_string(), JsValue::Smi(10));
        pm.insert("a".to_string(), JsValue::Smi(20));
        // Re-insert "1" — should update value, not move it.
        pm.insert("1".to_string(), JsValue::Smi(99));
        let keys: Vec<&str> = pm.keys().map(|s| &**s).collect();
        assert_eq!(keys, vec!["1", "a"]);
        assert_eq!(pm.get("1"), Some(&JsValue::Smi(99)));
    }

    // ── Shape ID / offset API tests ──────────────────────────────────────

    #[test]
    fn test_shape_id_stable_on_value_update() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(1));
        let id_after_insert = pm.shape_id();
        // Updating an existing value does not change the shape.
        pm.insert("x".to_string(), JsValue::Smi(2));
        assert_eq!(pm.shape_id(), id_after_insert);
    }

    #[test]
    fn test_shape_id_changes_on_new_property() {
        let mut pm = PropertyMap::new();
        let id0 = pm.shape_id();
        pm.insert("x".to_string(), JsValue::Smi(1));
        assert_ne!(pm.shape_id(), id0);
    }

    #[test]
    fn test_shape_id_changes_on_remove() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(1));
        let id1 = pm.shape_id();
        pm.remove("x");
        assert_ne!(pm.shape_id(), id1);
    }

    #[test]
    fn test_shape_id_changes_on_attr_change() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(1));
        let id1 = pm.shape_id();
        pm.set_writable("x", false);
        assert_ne!(pm.shape_id(), id1);
    }

    #[test]
    fn test_layout_id_shared_for_identical_property_sequences() {
        let mut first = PropertyMap::new();
        first.insert("x".to_string(), JsValue::Smi(1));
        first.insert("y".to_string(), JsValue::Smi(2));

        let mut second = PropertyMap::new();
        second.insert("x".to_string(), JsValue::Smi(10));
        second.insert("y".to_string(), JsValue::Smi(20));

        assert_eq!(first.layout_id(), second.layout_id());
    }

    #[test]
    fn test_layout_id_changes_for_different_layouts() {
        let mut attrs = PropertyMap::new();
        attrs.insert("x".to_string(), JsValue::Smi(1));
        attrs.set_writable("x", false);

        let mut order = PropertyMap::new();
        order.insert("y".to_string(), JsValue::Smi(1));
        order.insert("x".to_string(), JsValue::Smi(2));

        let mut baseline = PropertyMap::new();
        baseline.insert("x".to_string(), JsValue::Smi(3));

        assert_ne!(baseline.layout_id(), attrs.layout_id());
        assert_ne!(baseline.layout_id(), order.layout_id());
    }

    #[test]
    fn test_offset_of_and_get_by_offset() {
        let mut pm = PropertyMap::new();
        pm.insert("a".to_string(), JsValue::Smi(10));
        pm.insert("b".to_string(), JsValue::Smi(20));
        let off_a = pm.offset_of("a").unwrap();
        let off_b = pm.offset_of("b").unwrap();
        assert_eq!(pm.get_by_offset(off_a), Some(&JsValue::Smi(10)));
        assert_eq!(pm.get_by_offset(off_b), Some(&JsValue::Smi(20)));
        assert!(pm.offset_of("missing").is_none());
        assert!(pm.get_by_offset(999).is_none());
    }

    #[test]
    fn test_set_by_offset() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(1));
        let off = pm.offset_of("x").unwrap();
        assert!(pm.set_by_offset(off, JsValue::Smi(42)));
        assert_eq!(pm.get("x"), Some(&JsValue::Smi(42)));
        assert!(!pm.set_by_offset(999, JsValue::Null));
    }

    #[test]
    fn test_is_writable_by_offset() {
        let mut pm = PropertyMap::new();
        pm.insert("w".to_string(), JsValue::Smi(1));
        pm.insert_with_attrs(
            "ro".to_string(),
            JsValue::Smi(2),
            PropertyAttributes::ENUMERABLE,
        );
        let off_w = pm.offset_of("w").unwrap();
        let off_ro = pm.offset_of("ro").unwrap();
        assert!(pm.is_writable_by_offset(off_w));
        assert!(!pm.is_writable_by_offset(off_ro));
        assert!(!pm.is_writable_by_offset(999));
    }

    #[test]
    fn test_unique_shape_ids_across_maps() {
        let pm1 = PropertyMap::new();
        let pm2 = PropertyMap::new();
        assert_ne!(pm1.shape_id(), pm2.shape_id());
    }

    // ── freeze / seal / is_frozen / is_sealed ────────────────────────────

    #[test]
    fn test_freeze_makes_all_non_writable_non_configurable() {
        let mut pm = PropertyMap::new();
        pm.insert("a".to_string(), JsValue::Smi(1));
        pm.insert("b".to_string(), JsValue::Smi(2));
        pm.freeze();
        assert!(!pm.is_writable("a"));
        assert!(!pm.is_configurable("a"));
        assert!(!pm.is_writable("b"));
        assert!(!pm.is_configurable("b"));
        assert!(!pm.extensible);
    }

    #[test]
    fn test_seal_preserves_writable_removes_configurable() {
        let mut pm = PropertyMap::new();
        pm.insert("a".to_string(), JsValue::Smi(1));
        pm.seal();
        assert!(pm.is_writable("a"), "seal should preserve writable");
        assert!(!pm.is_configurable("a"), "seal should remove configurable");
        assert!(!pm.extensible);
    }

    #[test]
    fn test_is_frozen_empty_non_extensible() {
        let mut pm = PropertyMap::new();
        pm.extensible = false;
        assert!(pm.is_frozen(), "empty non-extensible map should be frozen");
    }

    #[test]
    fn test_is_frozen_with_writable_property() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(1));
        pm.extensible = false;
        assert!(
            !pm.is_frozen(),
            "non-extensible map with writable prop is not frozen"
        );
    }

    #[test]
    fn test_is_sealed_with_configurable_property() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(1));
        pm.extensible = false;
        assert!(
            !pm.is_sealed(),
            "non-extensible map with configurable prop is not sealed"
        );
    }

    #[test]
    fn test_non_extensible_insert_rejected() {
        let mut pm = PropertyMap::new();
        pm.extensible = false;
        pm.insert("newkey".to_string(), JsValue::Smi(42));
        assert!(
            !pm.contains_key("newkey"),
            "non-extensible map should reject new property"
        );
    }

    #[test]
    fn test_non_extensible_allows_existing_update() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(1));
        pm.extensible = false;
        pm.insert("x".to_string(), JsValue::Smi(2));
        assert_eq!(
            pm.get("x"),
            Some(&JsValue::Smi(2)),
            "updating existing property should succeed on non-extensible"
        );
    }

    // ── enumerable_keys / enumerable_iter ────────────────────────────────

    #[test]
    fn test_enumerable_keys_skips_non_enumerable() {
        let mut pm = PropertyMap::new();
        pm.insert("a".to_string(), JsValue::Smi(1)); // default = enumerable
        pm.insert_with_attrs(
            "b".to_string(),
            JsValue::Smi(2),
            PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE,
        ); // not enumerable
        pm.insert("c".to_string(), JsValue::Smi(3)); // default = enumerable
        let keys: Vec<&str> = pm.enumerable_keys().map(|k| &**k).collect();
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0], "a");
        assert_eq!(keys[1], "c");
    }

    #[test]
    fn test_enumerable_iter_returns_only_enumerable_pairs() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(10));
        pm.insert_with_attrs(
            "hidden".to_string(),
            JsValue::Smi(99),
            PropertyAttributes::empty(),
        );
        pm.insert("y".to_string(), JsValue::Smi(20));
        let pairs: Vec<(&str, &JsValue)> = pm.enumerable_iter().map(|(k, v)| (&**k, v)).collect();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, "x");
        assert_eq!(pairs[1].0, "y");
    }

    // ── iter_with_attrs ──────────────────────────────────────────────────

    #[test]
    fn test_iter_with_attrs_returns_all_triples() {
        let mut pm = PropertyMap::new();
        pm.insert("a".to_string(), JsValue::Smi(1));
        pm.insert_with_attrs(
            "b".to_string(),
            JsValue::Smi(2),
            PropertyAttributes::WRITABLE,
        );
        let triples: Vec<(&str, &JsValue, PropertyAttributes)> =
            pm.iter_with_attrs().map(|(k, v, a)| (&**k, v, a)).collect();
        assert_eq!(triples.len(), 2);
        assert!(triples[0].2.contains(PropertyAttributes::ENUMERABLE));
        assert!(!triples[1].2.contains(PropertyAttributes::ENUMERABLE));
    }

    // ── freeze / seal ────────────────────────────────────────────────────

    #[test]
    fn test_freeze_makes_non_writable_non_configurable() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(1));
        pm.freeze();
        assert!(!pm.is_writable("x"));
        assert!(!pm.is_configurable("x"));
        assert!(!pm.extensible);
        assert!(pm.is_frozen());
    }

    #[test]
    fn test_seal_makes_non_configurable() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(1));
        pm.seal();
        assert!(pm.is_writable("x")); // seal doesn't remove writable
        assert!(!pm.is_configurable("x"));
        assert!(!pm.extensible);
        assert!(pm.is_sealed());
    }

    // ── __proto__ key is non-enumerable ──────────────────────────────────

    #[test]
    fn test_proto_key_non_enumerable() {
        let mut pm = PropertyMap::new();
        pm.insert(INTERNAL_PROTO_PROPERTY_KEY.to_string(), JsValue::Null);
        pm.insert("visible".to_string(), JsValue::Smi(1));
        let enum_keys: Vec<&str> = pm.enumerable_keys().map(|k| &**k).collect();
        assert_eq!(enum_keys.len(), 1);
        assert_eq!(enum_keys[0], "visible");
    }

    // ── make_all_non_enumerable ──────────────────────────────────────────

    #[test]
    fn test_make_all_non_enumerable() {
        let mut pm = PropertyMap::new();
        pm.insert("a".to_string(), JsValue::Smi(1));
        pm.insert("b".to_string(), JsValue::Smi(2));
        assert!(pm.is_enumerable("a"));
        pm.make_all_non_enumerable();
        assert!(!pm.is_enumerable("a"));
        assert!(!pm.is_enumerable("b"));
    }

    // ── proto_generation ─────────────────────────────────────────────────

    #[test]
    fn test_proto_generation_tracks_prototype_mutations() {
        use std::cell::RefCell;
        use std::rc::Rc;

        let proto = Rc::new(RefCell::new(PropertyMap::new()));
        proto.borrow_mut().insert("x".to_string(), JsValue::Smi(1));

        let child = Rc::new(RefCell::new(PropertyMap::new()));
        child.borrow_mut().insert(
            "__proto__".to_string(),
            JsValue::PlainObject(Rc::clone(&proto)),
        );

        let before = child.borrow().proto_generation();
        proto.borrow_mut().insert("x".to_string(), JsValue::Smi(2));
        let after = child.borrow().proto_generation();

        assert_ne!(after, before);
    }

    // ── set_writable / set_enumerable / set_configurable ─────────────────

    #[test]
    fn test_set_writable_toggle() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(1));
        assert!(pm.is_writable("x"));
        pm.set_writable("x", false);
        assert!(!pm.is_writable("x"));
        pm.set_writable("x", true);
        assert!(pm.is_writable("x"));
    }

    #[test]
    fn test_set_enumerable_toggle() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(1));
        assert!(pm.is_enumerable("x"));
        pm.set_enumerable("x", false);
        assert!(!pm.is_enumerable("x"));
    }

    #[test]
    fn test_set_configurable_toggle() {
        let mut pm = PropertyMap::new();
        pm.insert("x".to_string(), JsValue::Smi(1));
        assert!(pm.is_configurable("x"));
        pm.set_configurable("x", false);
        assert!(!pm.is_configurable("x"));
    }

    // ── Property enumeration order conformance ────────────────────────────

    #[test]
    fn test_enum_order_integer_indices_sorted_ascending() {
        let mut pm = PropertyMap::new();
        pm.insert("2".to_string(), JsValue::Smi(2));
        pm.insert("0".to_string(), JsValue::Smi(0));
        pm.insert("1".to_string(), JsValue::Smi(1));
        let keys: Vec<&str> = pm.keys().map(|k| &**k).collect();
        assert_eq!(keys, &["0", "1", "2"]);
    }

    #[test]
    fn test_enum_order_strings_after_integers() {
        let mut pm = PropertyMap::new();
        pm.insert("b".to_string(), JsValue::Smi(1));
        pm.insert("1".to_string(), JsValue::Smi(2));
        pm.insert("a".to_string(), JsValue::Smi(3));
        pm.insert("0".to_string(), JsValue::Smi(4));
        let keys: Vec<&str> = pm.keys().map(|k| &**k).collect();
        assert_eq!(keys, &["0", "1", "b", "a"]);
    }

    #[test]
    fn test_enum_order_symbols_after_strings() {
        use crate::builtins::symbol::{symbol_create, symbol_to_property_key};
        let mut pm = PropertyMap::new();
        pm.insert("a".to_string(), JsValue::Smi(1));
        let sym = symbol_create(Some("s".into()));
        let sym_key = symbol_to_property_key(sym);
        pm.insert(sym_key.clone(), JsValue::Smi(2));
        pm.insert("b".to_string(), JsValue::Smi(3));
        let keys: Vec<&str> = pm.keys().map(|k| &**k).collect();
        // "a", "b" (strings in insertion order), then symbol
        assert_eq!(keys, &["a", "b", sym_key.as_str()]);
    }

    #[test]
    fn test_enum_order_integers_strings_symbols_combined() {
        use crate::builtins::symbol::{symbol_create, symbol_to_property_key};
        let mut pm = PropertyMap::new();
        let sym = symbol_create(Some("sym".into()));
        let sym_key = symbol_to_property_key(sym);
        pm.insert("z".to_string(), JsValue::Smi(1));
        pm.insert(sym_key.clone(), JsValue::Smi(2));
        pm.insert("5".to_string(), JsValue::Smi(3));
        pm.insert("a".to_string(), JsValue::Smi(4));
        pm.insert("0".to_string(), JsValue::Smi(5));
        let keys: Vec<&str> = pm.keys().map(|k| &**k).collect();
        assert_eq!(keys, &["0", "5", "z", "a", sym_key.as_str()]);
    }

    #[test]
    fn test_enum_order_sparse_array_indices() {
        let mut pm = PropertyMap::new();
        pm.insert("2".to_string(), JsValue::String("c".into()));
        pm.insert("0".to_string(), JsValue::String("a".into()));
        pm.insert("1".to_string(), JsValue::String("b".into()));
        let keys: Vec<&str> = pm.keys().map(|k| &**k).collect();
        assert_eq!(keys, &["0", "1", "2"]);
    }

    #[test]
    fn test_enum_order_large_integer_indices() {
        let mut pm = PropertyMap::new();
        pm.insert("100".to_string(), JsValue::Smi(1));
        pm.insert("5".to_string(), JsValue::Smi(2));
        pm.insert("42".to_string(), JsValue::Smi(3));
        pm.insert("3".to_string(), JsValue::Smi(4));
        let keys: Vec<&str> = pm.keys().map(|k| &**k).collect();
        assert_eq!(keys, &["3", "5", "42", "100"]);
    }

    #[test]
    fn test_enum_order_u32_max_not_array_index() {
        let mut pm = PropertyMap::new();
        let max_str = u32::MAX.to_string();
        pm.insert("0".to_string(), JsValue::Smi(1));
        pm.insert(max_str.clone(), JsValue::Smi(2));
        pm.insert("a".to_string(), JsValue::Smi(3));
        let keys: Vec<&str> = pm.keys().map(|k| &**k).collect();
        // u32::MAX is NOT an array index, so treated as string
        assert_eq!(keys, &["0", max_str.as_str(), "a"]);
    }

    #[test]
    fn test_enum_order_leading_zero_not_array_index() {
        let mut pm = PropertyMap::new();
        pm.insert("01".to_string(), JsValue::Smi(1));
        pm.insert("0".to_string(), JsValue::Smi(2));
        pm.insert("1".to_string(), JsValue::Smi(3));
        let keys: Vec<&str> = pm.keys().map(|k| &**k).collect();
        // "01" is not a valid array index, so it's a string key
        assert_eq!(keys, &["0", "1", "01"]);
    }

    #[test]
    fn test_enum_order_string_insertion_order_preserved() {
        let mut pm = PropertyMap::new();
        pm.insert("c".to_string(), JsValue::Smi(1));
        pm.insert("a".to_string(), JsValue::Smi(2));
        pm.insert("b".to_string(), JsValue::Smi(3));
        let keys: Vec<&str> = pm.keys().map(|k| &**k).collect();
        assert_eq!(keys, &["c", "a", "b"]);
    }

    #[test]
    fn test_enum_order_multiple_symbols_insertion_order() {
        use crate::builtins::symbol::{symbol_create, symbol_to_property_key};
        let mut pm = PropertyMap::new();
        let s1 = symbol_create(Some("s1".into()));
        let s2 = symbol_create(Some("s2".into()));
        let k1 = symbol_to_property_key(s1);
        let k2 = symbol_to_property_key(s2);
        pm.insert(k1.clone(), JsValue::Smi(1));
        pm.insert("a".to_string(), JsValue::Smi(2));
        pm.insert(k2.clone(), JsValue::Smi(3));
        let keys: Vec<&str> = pm.keys().map(|k| &**k).collect();
        // Strings first, then symbols in insertion order
        assert_eq!(keys, &["a", k1.as_str(), k2.as_str()]);
    }

    #[test]
    fn test_enumerable_keys_skips_symbols() {
        use crate::builtins::symbol::{symbol_create, symbol_to_property_key};
        let mut pm = PropertyMap::new();
        pm.insert("a".to_string(), JsValue::Smi(1));
        let sym = symbol_create(Some("hidden".into()));
        let sym_key = symbol_to_property_key(sym);
        pm.insert(sym_key, JsValue::Smi(2));
        pm.insert("b".to_string(), JsValue::Smi(3));
        let enumerable: Vec<&str> = pm.enumerable_keys().map(|k| &**k).collect();
        assert_eq!(enumerable, &["a", "b"]);
    }

    #[test]
    fn test_enumerable_keys_skips_non_enumerable_v2() {
        let mut pm = PropertyMap::new();
        pm.insert("visible".to_string(), JsValue::Smi(1));
        pm.insert_with_attrs(
            "hidden".to_string(),
            JsValue::Smi(2),
            PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE,
        );
        let enumerable: Vec<&str> = pm.enumerable_keys().map(|k| &**k).collect();
        assert_eq!(enumerable, &["visible"]);
    }

    #[test]
    fn test_enumerable_iter_follows_spec_order() {
        let mut pm = PropertyMap::new();
        pm.insert("z".to_string(), JsValue::Smi(1));
        pm.insert("3".to_string(), JsValue::Smi(2));
        pm.insert("1".to_string(), JsValue::Smi(3));
        pm.insert("a".to_string(), JsValue::Smi(4));
        let pairs: Vec<(&str, &JsValue)> = pm.enumerable_iter().map(|(k, v)| (&**k, v)).collect();
        let keys: Vec<&str> = pairs.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, &["1", "3", "z", "a"]);
    }

    #[test]
    fn test_own_symbol_keys_returns_symbols_only() {
        use crate::builtins::symbol::{symbol_create, symbol_to_property_key};
        let mut pm = PropertyMap::new();
        pm.insert("a".to_string(), JsValue::Smi(1));
        let s1 = symbol_create(None);
        let s2 = symbol_create(None);
        let k1 = symbol_to_property_key(s1);
        let k2 = symbol_to_property_key(s2);
        pm.insert(k1, JsValue::Smi(2));
        pm.insert(k2, JsValue::Smi(3));
        pm.insert("b".to_string(), JsValue::Smi(4));
        let syms = pm.own_symbol_keys();
        assert_eq!(syms.len(), 2);
        assert_eq!(syms[0], s1);
        assert_eq!(syms[1], s2);
    }

    #[test]
    fn test_remove_preserves_spec_order_v2() {
        let mut pm = PropertyMap::new();
        pm.insert("1".to_string(), JsValue::Smi(1));
        pm.insert("0".to_string(), JsValue::Smi(2));
        pm.insert("a".to_string(), JsValue::Smi(3));
        pm.insert("b".to_string(), JsValue::Smi(4));
        pm.remove("a");
        let keys: Vec<&str> = pm.keys().map(|k| &**k).collect();
        assert_eq!(keys, &["0", "1", "b"]);
    }

    #[test]
    fn test_iter_with_attrs_follows_spec_order() {
        let mut pm = PropertyMap::new();
        pm.insert("b".to_string(), JsValue::Smi(1));
        pm.insert("10".to_string(), JsValue::Smi(2));
        pm.insert("2".to_string(), JsValue::Smi(3));
        pm.insert("a".to_string(), JsValue::Smi(4));
        let keys: Vec<&str> = pm.iter_with_attrs().map(|(k, _, _)| &**k).collect();
        assert_eq!(keys, &["2", "10", "b", "a"]);
    }

    #[test]
    fn test_insert_with_attrs_follows_spec_order() {
        let mut pm = PropertyMap::new();
        pm.insert_with_attrs(
            "b".to_string(),
            JsValue::Smi(1),
            PropertyAttributes::ENUMERABLE,
        );
        pm.insert_with_attrs(
            "3".to_string(),
            JsValue::Smi(2),
            PropertyAttributes::ENUMERABLE,
        );
        pm.insert_with_attrs(
            "1".to_string(),
            JsValue::Smi(3),
            PropertyAttributes::ENUMERABLE,
        );
        let keys: Vec<&str> = pm.keys().map(|k| &**k).collect();
        assert_eq!(keys, &["1", "3", "b"]);
    }

    #[test]
    fn test_small_maps_stay_inline_until_threshold() {
        let mut pm = PropertyMap::with_capacity(SMALL_PROPERTY_LINEAR_SCAN_CAP);
        for idx in 0..SMALL_PROPERTY_LINEAR_SCAN_CAP {
            pm.insert(format!("k{idx}"), JsValue::Smi(idx as i32));
        }
        assert!(matches!(pm.index, PropertyIndex::Inline));
        let last_key = format!("k{}", SMALL_PROPERTY_LINEAR_SCAN_CAP - 1);
        assert_eq!(
            pm.get(&last_key),
            Some(&JsValue::Smi((SMALL_PROPERTY_LINEAR_SCAN_CAP - 1) as i32))
        );

        pm.insert(
            format!("k{SMALL_PROPERTY_LINEAR_SCAN_CAP}"),
            JsValue::Smi(SMALL_PROPERTY_LINEAR_SCAN_CAP as i32),
        );
        assert!(matches!(pm.index, PropertyIndex::Map(_)));
        assert_eq!(
            pm.get(&format!("k{SMALL_PROPERTY_LINEAR_SCAN_CAP}")),
            Some(&JsValue::Smi(SMALL_PROPERTY_LINEAR_SCAN_CAP as i32))
        );
    }

    #[test]
    fn test_small_map_remove_demotes_to_inline() {
        let mut pm = PropertyMap::with_capacity(SMALL_PROPERTY_LINEAR_SCAN_CAP + 1);
        for idx in 0..=SMALL_PROPERTY_LINEAR_SCAN_CAP {
            pm.insert(format!("k{idx}"), JsValue::Smi(idx as i32));
        }
        assert!(matches!(pm.index, PropertyIndex::Map(_)));

        assert_eq!(
            pm.remove(&format!("k{SMALL_PROPERTY_LINEAR_SCAN_CAP}")),
            Some(JsValue::Smi(SMALL_PROPERTY_LINEAR_SCAN_CAP as i32))
        );
        assert!(matches!(pm.index, PropertyIndex::Inline));
        assert_eq!(pm.get("k0"), Some(&JsValue::Smi(0)));
    }
}
