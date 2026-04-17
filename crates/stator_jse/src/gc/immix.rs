//! Immix-style regional garbage collector for the young generation.
//!
//! # Architecture
//!
//! The Immix collector organises the young generation into fixed-size
//! **blocks** (32 KiB each), subdivided into **lines** (128 bytes each).
//! Allocation is a fast bump-pointer within the current block.  When a
//! block is exhausted, the allocator obtains a fresh (or recycled) block
//! from the [`ImmixSpace`].
//!
//! ## Line-Granularity Marking
//!
//! During collection the marker sets per-line bits instead of per-object
//! bits.  A line is marked if it contains any part of a live object.
//! After marking, unmarked lines are immediately reclaimable and the
//! block can be **recycled** (partially reused) rather than requiring a
//! full evacuation.
//!
//! ## Opportunistic Evacuation
//!
//! Blocks whose live-line ratio falls below [`EVACUATION_THRESHOLD_PCT`]
//! are selected for **evacuation**: live objects in those blocks are
//! copied out to a fresh block and the fragmented block is released.
//! This keeps fragmentation bounded without a full copying collector.
//!
//! ## Thread-Local Allocation Buffers (TLABs)
//!
//! Each mutator thread holds a [`Tlab`] that bump-allocates into the
//! thread's current block.  TLAB exhaustion triggers a block request
//! from the shared [`ImmixSpace`], amortising synchronisation costs.
//!
//! ## Concurrent Marking (Old Generation)
//!
//! The [`ConcurrentMarker`] performs snapshot-at-the-beginning (Dijkstra)
//! marking on the old generation, communicating results through an
//! `Arc<Mutex<…>>` channel.
//!
//! # Integration
//!
//! This module reuses the existing [`Tracer`][crate::gc::trace::Tracer] /
//! [`Trace`][crate::gc::trace::Trace] infrastructure and the
//! [`WriteBarrier`][crate::gc::write_barrier::WriteBarrier] mechanism.

use std::alloc::{Layout, alloc, dealloc};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use crate::gc::heap::OldSpace;
use crate::gc::trace::{Tracer, trace_heap_object};
use crate::objects::heap_object::HeapObject;

// ── Constants ────────────────────────────────────────────────────────────────

/// Size of a single Immix block in bytes (32 KiB).
pub const BLOCK_SIZE: usize = 32 * 1024;

/// Size of a single line within a block in bytes (128 B).
pub const LINE_SIZE: usize = 128;

/// Number of lines per block (`BLOCK_SIZE / LINE_SIZE` = 256).
pub const LINES_PER_BLOCK: usize = BLOCK_SIZE / LINE_SIZE;

/// Blocks with fewer than this percentage of live lines are selected for
/// opportunistic evacuation during collection.
pub const EVACUATION_THRESHOLD_PCT: u8 = 50;

/// Default number of blocks in the Immix young-generation space.
///
/// 128 blocks × 32 KiB = 4 MiB, matching the existing semi-space nursery.
const DEFAULT_BLOCK_COUNT: usize = 128;

// ── LineMap ──────────────────────────────────────────────────────────────────

/// Per-block bitmap tracking which 128-byte lines contain live data.
///
/// Each bit corresponds to one line.  A set bit means the line is
/// **marked** (contains at least part of a live object).
#[derive(Clone)]
pub struct LineMap {
    /// One bit per line.  `LINES_PER_BLOCK` = 256, so 256 / 8 = 32 bytes.
    bits: [u8; LINES_PER_BLOCK / 8],
}

impl LineMap {
    /// Create a new line map with all lines unmarked.
    pub fn new() -> Self {
        Self {
            bits: [0u8; LINES_PER_BLOCK / 8],
        }
    }

    /// Mark the line at `index` as containing live data.
    ///
    /// # Panics
    ///
    /// Panics if `index >= LINES_PER_BLOCK`.
    pub fn mark(&mut self, index: usize) {
        assert!(index < LINES_PER_BLOCK, "line index out of range");
        self.bits[index / 8] |= 1 << (index % 8);
    }

    /// Returns `true` if the line at `index` is marked.
    ///
    /// # Panics
    ///
    /// Panics if `index >= LINES_PER_BLOCK`.
    pub fn is_marked(&self, index: usize) -> bool {
        assert!(index < LINES_PER_BLOCK, "line index out of range");
        (self.bits[index / 8] & (1 << (index % 8))) != 0
    }

    /// Clear all line marks.
    pub fn clear(&mut self) {
        self.bits = [0u8; LINES_PER_BLOCK / 8];
    }

    /// Count the number of marked (live) lines.
    pub fn live_line_count(&self) -> usize {
        self.bits.iter().map(|b| b.count_ones() as usize).sum()
    }

    /// Returns the percentage (0–100) of lines that are marked.
    pub fn occupancy_percent(&self) -> u8 {
        let live = self.live_line_count();
        ((live * 100) / LINES_PER_BLOCK) as u8
    }
}

impl Default for LineMap {
    fn default() -> Self {
        Self::new()
    }
}

// ── ImmixBlock ───────────────────────────────────────────────────────────────

/// A single 32 KiB Immix block with line-granularity metadata.
///
/// Allocation within a block is a simple bump pointer.  The block also
/// carries a [`LineMap`] used during collection to identify live lines
/// and decide whether the block should be evacuated.
pub struct ImmixBlock {
    /// Base pointer of the 32 KiB allocation.
    base: *mut u8,
    /// Byte offset of the next free position (bump cursor).
    cursor: usize,
    /// Per-line mark bitmap.
    line_map: LineMap,
    /// `true` if this block has been selected for evacuation.
    evacuate: bool,
}

// SAFETY: `ImmixBlock` owns its backing allocation exclusively.
unsafe impl Send for ImmixBlock {}

impl ImmixBlock {
    /// Allocate a new, empty Immix block.
    ///
    /// # Panics
    ///
    /// Panics if the system allocator fails.
    pub fn new() -> Self {
        // SAFETY: BLOCK_SIZE > 0 and alignment 8 is a valid power of two.
        let layout = Layout::from_size_align(BLOCK_SIZE, 8).expect("valid layout");
        let base = unsafe { alloc(layout) };
        assert!(!base.is_null(), "Immix block allocation failed");
        Self {
            base,
            cursor: 0,
            line_map: LineMap::new(),
            evacuate: false,
        }
    }

    /// Bump-allocate `layout.size()` bytes from this block.
    ///
    /// Returns `None` if the remaining space in the block is insufficient.
    pub fn bump_alloc(&mut self, layout: Layout) -> Option<*mut u8> {
        let align = layout.align();
        let size = layout.size();
        let current = self.base as usize + self.cursor;
        let aligned = current.checked_add(align - 1)? & !(align - 1);
        let end = aligned.checked_add(size)?;
        let new_cursor = end - self.base as usize;
        if new_cursor > BLOCK_SIZE {
            return None;
        }
        self.cursor = new_cursor;
        Some(aligned as *mut u8)
    }

    /// Returns `true` if `ptr` falls within this block's address range.
    pub fn contains(&self, ptr: *const u8) -> bool {
        let addr = ptr as usize;
        let base = self.base as usize;
        addr >= base && addr < base + BLOCK_SIZE
    }

    /// Compute the line index for a pointer within this block.
    ///
    /// # Panics
    ///
    /// Panics if `ptr` is not within this block.
    pub fn line_index_of(&self, ptr: *const u8) -> usize {
        let offset = ptr as usize - self.base as usize;
        assert!(offset < BLOCK_SIZE, "pointer not within this block");
        offset / LINE_SIZE
    }

    /// Mark the line(s) spanned by an object at `ptr` with `size` bytes.
    pub fn mark_lines_for_object(&mut self, ptr: *const u8, size: usize) {
        let start = self.line_index_of(ptr);
        let end_byte = (ptr as usize + size).saturating_sub(1);
        let end = (end_byte - self.base as usize) / LINE_SIZE;
        for i in start..=end.min(LINES_PER_BLOCK - 1) {
            self.line_map.mark(i);
        }
    }

    /// Returns `true` if this block's live-line ratio is below the
    /// evacuation threshold.
    pub fn should_evacuate(&self) -> bool {
        self.line_map.occupancy_percent() < EVACUATION_THRESHOLD_PCT
    }

    /// Access the line map.
    pub fn line_map(&self) -> &LineMap {
        &self.line_map
    }

    /// Mutable access to the line map.
    pub fn line_map_mut(&mut self) -> &mut LineMap {
        &mut self.line_map
    }

    /// Raw base pointer.
    pub fn base_ptr(&self) -> *mut u8 {
        self.base
    }

    /// Current bump cursor offset.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Reset the block for reuse: clear the cursor and line map.
    pub fn reset(&mut self) {
        self.cursor = 0;
        self.line_map.clear();
        self.evacuate = false;
    }

    /// Whether this block is flagged for evacuation.
    pub fn is_evacuate_candidate(&self) -> bool {
        self.evacuate
    }

    /// Flag (or unflag) this block for evacuation.
    pub fn set_evacuate(&mut self, flag: bool) {
        self.evacuate = flag;
    }

    /// Number of bytes currently in use.
    pub fn used(&self) -> usize {
        self.cursor
    }

    /// Number of free bytes remaining.
    pub fn remaining(&self) -> usize {
        BLOCK_SIZE - self.cursor
    }
}

impl Default for ImmixBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ImmixBlock {
    fn drop(&mut self) {
        if !self.base.is_null() {
            // SAFETY: `base` was allocated with layout (BLOCK_SIZE, 8) in `new`.
            let layout = Layout::from_size_align(BLOCK_SIZE, 8).expect("valid layout");
            unsafe { dealloc(self.base, layout) };
        }
    }
}

// ── ImmixSpace ───────────────────────────────────────────────────────────────

/// Immix-organised young-generation space.
///
/// Manages a pool of [`ImmixBlock`]s, providing fresh blocks for
/// allocation and recycling blocks after collection.
pub struct ImmixSpace {
    /// Blocks that are full (no bump space remaining).
    full_blocks: Vec<ImmixBlock>,
    /// Blocks available for allocation (have free space or recycled).
    free_blocks: Vec<ImmixBlock>,
    /// Maximum number of blocks the space may contain.
    max_blocks: usize,
}

impl ImmixSpace {
    /// Create a new Immix space with the given maximum block count.
    pub fn new(max_blocks: usize) -> Self {
        Self {
            full_blocks: Vec::new(),
            free_blocks: Vec::new(),
            max_blocks,
        }
    }

    /// Create an Immix space with the default block count (128 × 32 KiB).
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_BLOCK_COUNT)
    }

    /// Obtain a block for allocation.
    ///
    /// Returns a recycled free block if available, otherwise allocates a
    /// new block (up to the maximum).  Returns `None` when the space is
    /// exhausted.
    pub fn obtain_block(&mut self) -> Option<ImmixBlock> {
        if let Some(block) = self.free_blocks.pop() {
            return Some(block);
        }
        let total = self.full_blocks.len() + self.free_blocks.len() + 1;
        if total <= self.max_blocks {
            return Some(ImmixBlock::new());
        }
        None
    }

    /// Return a full block to the space for future collection.
    pub fn return_full_block(&mut self, block: ImmixBlock) {
        self.full_blocks.push(block);
    }

    /// Return a recycled (empty) block to the free list.
    pub fn return_free_block(&mut self, block: ImmixBlock) {
        self.free_blocks.push(block);
    }

    /// Total number of bytes currently in use across all full blocks.
    pub fn used(&self) -> usize {
        self.full_blocks.iter().map(|b| b.used()).sum()
    }

    /// Total capacity of the space in bytes.
    pub fn capacity(&self) -> usize {
        self.max_blocks * BLOCK_SIZE
    }

    /// Number of full blocks.
    pub fn full_block_count(&self) -> usize {
        self.full_blocks.len()
    }

    /// Number of free (available) blocks.
    pub fn free_block_count(&self) -> usize {
        self.free_blocks.len()
    }

    /// Returns `true` if `ptr` falls within any full block in this space.
    pub fn contains(&self, ptr: *const u8) -> bool {
        self.full_blocks.iter().any(|b| b.contains(ptr))
    }

    /// Recycle blocks after a collection cycle.
    ///
    /// Blocks whose live-line ratio is below the evacuation threshold
    /// are reset and moved to the free list.  Remaining blocks have
    /// their line maps cleared and stay in the full list.
    ///
    /// Returns the number of blocks recycled.
    pub fn recycle_blocks(&mut self) -> usize {
        let mut keep = Vec::new();
        let mut recycled = 0usize;

        for mut block in self.full_blocks.drain(..) {
            if block.should_evacuate() {
                block.reset();
                self.free_blocks.push(block);
                recycled += 1;
            } else {
                block.line_map_mut().clear();
                keep.push(block);
            }
        }
        self.full_blocks = keep;
        recycled
    }

    /// Drain all full blocks for collection processing.
    pub(crate) fn drain_full_blocks(&mut self) -> Vec<ImmixBlock> {
        self.full_blocks.drain(..).collect()
    }

    /// Re-insert blocks after collection.
    pub(crate) fn return_blocks(&mut self, full: Vec<ImmixBlock>, free: Vec<ImmixBlock>) {
        self.full_blocks.extend(full);
        self.free_blocks.extend(free);
    }
}

impl Default for ImmixSpace {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ── Tlab ─────────────────────────────────────────────────────────────────────

/// Thread-Local Allocation Buffer for fast per-thread bump allocation.
///
/// Each mutator thread owns a `Tlab` that holds the current block.
/// Allocation is a fast pointer-bump; when the block is exhausted the
/// TLAB requests a new block from the shared [`ImmixSpace`].
pub struct Tlab {
    /// The block currently being allocated into.
    current_block: Option<ImmixBlock>,
    /// Total bytes allocated through this TLAB (lifetime counter).
    bytes_allocated: usize,
}

impl Tlab {
    /// Create a new, empty TLAB with no backing block.
    pub fn new() -> Self {
        Self {
            current_block: None,
            bytes_allocated: 0,
        }
    }

    /// Bump-allocate `layout` bytes, requesting new blocks from `space`
    /// when the current one is exhausted.
    ///
    /// Returns a pointer to the allocation, or `None` if the space cannot
    /// provide a new block.
    pub fn allocate(&mut self, layout: Layout, space: &mut ImmixSpace) -> Option<*mut u8> {
        // Fast path: try the current block.
        if let Some(ref mut block) = self.current_block
            && let Some(ptr) = block.bump_alloc(layout)
        {
            self.bytes_allocated += layout.size();
            return Some(ptr);
        }

        // Retire the current block.
        if let Some(block) = self.current_block.take() {
            space.return_full_block(block);
        }

        // Obtain a new block and retry.
        let mut new_block = space.obtain_block()?;
        let ptr = new_block.bump_alloc(layout)?;
        self.bytes_allocated += layout.size();
        self.current_block = Some(new_block);
        Some(ptr)
    }

    /// Flush the current block back to `space` (e.g. before a GC cycle).
    pub fn flush(&mut self, space: &mut ImmixSpace) {
        if let Some(block) = self.current_block.take() {
            space.return_full_block(block);
        }
    }

    /// Total bytes allocated through this TLAB (lifetime).
    pub fn bytes_allocated(&self) -> usize {
        self.bytes_allocated
    }

    /// Returns `true` if the TLAB currently holds a block.
    pub fn has_block(&self) -> bool {
        self.current_block.is_some()
    }

    /// Remaining bytes in the current block, or 0 if no block is held.
    pub fn remaining(&self) -> usize {
        self.current_block.as_ref().map_or(0, |b| b.remaining())
    }
}

impl Default for Tlab {
    fn default() -> Self {
        Self::new()
    }
}

// ── ImmixCollector ───────────────────────────────────────────────────────────

/// Immix collector: line-granularity marking with opportunistic evacuation.
///
/// # Collection Algorithm
///
/// 1. **Mark** — Starting from roots, mark the lines spanned by every
///    reachable object using a grey-stack BFS.
/// 2. **Identify evacuation candidates** — Blocks whose live-line ratio
///    is below [`EVACUATION_THRESHOLD_PCT`] are flagged for evacuation.
/// 3. **Evacuate** — Live objects in candidate blocks are copied to
///    fresh blocks; root slots are updated to the new addresses.
/// 4. **Recycle** — Evacuated blocks are returned to the free list;
///    surviving blocks have their line maps cleared.
pub struct ImmixCollector {
    /// Addresses (as `usize`) of objects marked reachable in this cycle.
    mark_set: HashSet<usize>,
}

impl ImmixCollector {
    /// Create a new, empty collector.
    pub fn new() -> Self {
        Self {
            mark_set: HashSet::new(),
        }
    }

    /// Mark an object as reachable.
    fn mark_object(&mut self, ptr: *mut HeapObject) {
        self.mark_set.insert(ptr as usize);
    }

    /// Returns `true` if the object at `ptr` has been marked.
    pub fn is_marked(&self, ptr: *mut HeapObject) -> bool {
        self.mark_set.contains(&(ptr as usize))
    }

    /// **Mark phase**: mark all objects reachable from `roots`.
    ///
    /// For each root the corresponding line(s) in the containing block
    /// are marked.  A grey-stack BFS discovers transitive references
    /// (child traversal via `Trace` dispatch will be wired in once the
    /// full object model is available).
    ///
    /// # Safety
    ///
    /// Every non-null root must point to a valid, live `HeapObject` that
    /// resides within one of the `blocks`.
    pub unsafe fn mark(&mut self, roots: &[*mut *mut HeapObject], blocks: &mut [ImmixBlock]) {
        let mut grey_stack: Vec<*mut HeapObject> = Vec::new();

        for &slot in roots {
            // SAFETY: caller guarantees slot validity.
            let ptr = unsafe { *slot };
            if ptr.is_null() {
                continue;
            }
            if !self.is_marked(ptr) {
                self.mark_object(ptr);
                grey_stack.push(ptr);

                // Mark the line(s) in the containing block.
                // SAFETY: ptr is a valid HeapObject; alloc_size was
                // initialised by the allocator.
                let obj_size = unsafe { (*ptr).alloc_size() } as usize;
                for block in blocks.iter_mut() {
                    if block.contains(ptr as *const u8) {
                        block.mark_lines_for_object(ptr as *const u8, obj_size);
                        break;
                    }
                }
            }
        }

        // BFS drain — trace each object's children via the Trace dispatch.
        while let Some(obj) = grey_stack.pop() {
            let mut tracer = Tracer::new();
            // SAFETY: obj is a valid, marked HeapObject pointer.
            unsafe { trace_heap_object(obj, &mut tracer) };

            // Enqueue newly discovered children.
            for child_raw in tracer.gray_stack {
                let child = child_raw as *mut HeapObject;
                if child.is_null() || self.is_marked(child) {
                    continue;
                }
                self.mark_object(child);
                grey_stack.push(child);

                // Mark lines for the child in its containing block.
                // SAFETY: child is a valid HeapObject with initialised alloc_size.
                let child_size = unsafe { (*child).alloc_size() } as usize;
                for block in blocks.iter_mut() {
                    if block.contains(child as *const u8) {
                        block.mark_lines_for_object(child as *const u8, child_size);
                        break;
                    }
                }
            }
        }
    }

    /// **Identify evacuation candidates**: flag blocks whose live-line
    /// ratio is below the threshold.
    ///
    /// Returns the number of blocks selected for evacuation.
    pub fn select_evacuation_candidates(&self, blocks: &mut [ImmixBlock]) -> usize {
        let mut count = 0;
        for block in blocks.iter_mut() {
            if block.should_evacuate() && block.used() > 0 {
                block.set_evacuate(true);
                count += 1;
            }
        }
        count
    }

    /// **Evacuate**: copy live objects from candidate blocks into fresh
    /// blocks provided by `space`, updating root slots.
    ///
    /// Returns the list of evacuated (now-empty) blocks.
    ///
    /// # Safety
    ///
    /// Every root pointing into an evacuation candidate must be a valid,
    /// marked, live `HeapObject`.
    pub unsafe fn evacuate(
        &self,
        blocks: &mut Vec<ImmixBlock>,
        roots: &mut [*mut *mut HeapObject],
        space: &mut ImmixSpace,
    ) -> Vec<ImmixBlock> {
        let mut evacuated: Vec<ImmixBlock> = Vec::new();
        let mut target_block: Option<ImmixBlock> = None;
        let mut forwarding: Vec<(usize, usize)> = Vec::new();

        let mut kept = Vec::new();
        for block in blocks.drain(..) {
            if block.is_evacuate_candidate() {
                let base = block.base_ptr();
                let used = block.cursor();
                let mut offset = 0usize;
                while offset < used {
                    // SAFETY: offset is within [0, used); the header was
                    // initialised by the allocator.
                    let old_ptr = unsafe { base.add(offset) } as *mut HeapObject;
                    let size = unsafe { (*old_ptr).alloc_size() } as usize;
                    if size == 0 {
                        break;
                    }
                    if self.is_marked(old_ptr) {
                        let layout = Layout::from_size_align(size, 8).expect("valid layout");
                        let dest = loop {
                            if let Some(ref mut tb) = target_block {
                                if let Some(ptr) = tb.bump_alloc(layout) {
                                    break ptr;
                                }
                                // Target block full — retire it.
                                kept.push(target_block.take().unwrap());
                            }
                            target_block = space.obtain_block();
                            if target_block.is_none() {
                                break std::ptr::null_mut();
                            }
                        };
                        if dest.is_null() {
                            break;
                        }
                        // SAFETY: old_ptr and dest are valid,
                        // non-overlapping (different blocks).
                        unsafe {
                            std::ptr::copy_nonoverlapping(old_ptr as *const u8, dest, size);
                        }
                        forwarding.push((old_ptr as usize, dest as usize));
                    }
                    offset += size;
                }
                evacuated.push(block);
            } else {
                kept.push(block);
            }
        }

        // Retire the last target block.
        if let Some(tb) = target_block {
            if tb.used() > 0 {
                kept.push(tb);
            } else {
                space.return_free_block(tb);
            }
        }

        *blocks = kept;

        // Update root slots that were forwarded.
        for slot in roots.iter_mut() {
            // SAFETY: slot is a valid root pointer slot.
            let old_addr = unsafe { *(*slot) } as usize;
            if let Some(&(_, new_addr)) = forwarding.iter().find(|(o, _)| *o == old_addr) {
                unsafe { **slot = new_addr as *mut HeapObject };
            }
        }

        evacuated
    }

    /// Run a complete Immix collection cycle.
    ///
    /// Performs mark → select candidates → evacuate → recycle.
    ///
    /// # Safety
    ///
    /// See [`mark`][Self::mark] and [`evacuate`][Self::evacuate] for the
    /// full safety contracts.
    pub unsafe fn collect(&mut self, roots: &mut [*mut *mut HeapObject], space: &mut ImmixSpace) {
        let mut blocks = space.drain_full_blocks();

        // Phase 1: Mark.
        // SAFETY: caller upholds mark's preconditions.
        unsafe { self.mark(roots, &mut blocks) };

        // Phase 2: Select evacuation candidates.
        let _candidates = self.select_evacuation_candidates(&mut blocks);

        // Phase 3: Evacuate.
        // SAFETY: caller upholds evacuate's preconditions.
        let evacuated = unsafe { self.evacuate(&mut blocks, roots, space) };

        // Phase 4: Recycle evacuated blocks.
        let mut free = Vec::new();
        for mut block in evacuated {
            block.reset();
            free.push(block);
        }

        // Clear line maps on surviving blocks.
        for block in &mut blocks {
            block.line_map_mut().clear();
            block.set_evacuate(false);
        }

        space.return_blocks(blocks, free);
        self.mark_set.clear();
    }
}

impl Default for ImmixCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ── ConcurrentMarker ─────────────────────────────────────────────────────────

/// Result of a concurrent marking phase on the old generation.
#[derive(Debug, Clone)]
pub struct ConcurrentMarkResult {
    /// Addresses of objects identified as live by the concurrent marker.
    pub live_set: HashSet<usize>,
    /// Number of objects marked.
    pub marked_count: usize,
}

/// Concurrent marker for the old generation using Dijkstra-style
/// snapshot-at-the-beginning barriers.
///
/// The marker communicates results through a shared
/// `Arc<Mutex<Option<ConcurrentMarkResult>>>`.
pub struct ConcurrentMarker {
    /// Shared channel for the marking result.
    result: Arc<Mutex<Option<ConcurrentMarkResult>>>,
}

impl ConcurrentMarker {
    /// Create a new concurrent marker.
    pub fn new() -> Self {
        Self {
            result: Arc::new(Mutex::new(None)),
        }
    }

    /// Perform a concurrent marking pass over the old generation.
    ///
    /// Snapshots the root set, then marks all reachable objects in
    /// `old_space`.  A Dijkstra-style write barrier must be active for
    /// the duration of this pass to maintain the snapshot invariant.
    ///
    /// # Safety
    ///
    /// Every root must point to a valid, live `HeapObject` within
    /// `old_space`.
    pub unsafe fn mark_old_generation(&self, roots: &[*const HeapObject], old_space: &OldSpace) {
        let mut live_set = HashSet::new();
        let mut grey_stack: Vec<*const HeapObject> = Vec::new();

        for &ptr in roots {
            if ptr.is_null() {
                continue;
            }
            if old_space.contains(ptr as *mut u8) && !live_set.contains(&(ptr as usize)) {
                live_set.insert(ptr as usize);
                grey_stack.push(ptr);
            }
        }

        // BFS — trace each object's children via the Trace dispatch.
        while let Some(obj) = grey_stack.pop() {
            let mut tracer = Tracer::new();
            // SAFETY: obj is a valid, marked HeapObject pointer.
            unsafe { trace_heap_object(obj as *mut HeapObject, &mut tracer) };

            for child_raw in tracer.gray_stack {
                let child = child_raw as *const HeapObject;
                if child.is_null() {
                    continue;
                }
                let addr = child as usize;
                if old_space.contains(child as *mut u8) && !live_set.contains(&addr) {
                    live_set.insert(addr);
                    grey_stack.push(child);
                }
            }
        }

        let marked_count = live_set.len();
        let result = ConcurrentMarkResult {
            live_set,
            marked_count,
        };

        if let Ok(mut guard) = self.result.lock() {
            *guard = Some(result);
        }
    }

    /// Retrieve the result of the most recent concurrent marking pass.
    ///
    /// Returns `None` if no marking has completed or the result has
    /// already been consumed.
    pub fn take_result(&self) -> Option<ConcurrentMarkResult> {
        self.result.lock().ok().and_then(|mut guard| guard.take())
    }

    /// Clone the shared result channel for use by a background thread.
    pub fn result_channel(&self) -> Arc<Mutex<Option<ConcurrentMarkResult>>> {
        Arc::clone(&self.result)
    }
}

impl Default for ConcurrentMarker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::Layout;

    // ── LineMap ────────────────────────────────────────────────────────────

    #[test]
    fn test_line_map_mark_and_query() {
        let mut lm = LineMap::new();
        assert!(!lm.is_marked(0));
        lm.mark(0);
        assert!(lm.is_marked(0));
        assert!(!lm.is_marked(1));
    }

    #[test]
    fn test_line_map_clear_resets_all() {
        let mut lm = LineMap::new();
        lm.mark(0);
        lm.mark(100);
        lm.mark(255);
        assert_eq!(lm.live_line_count(), 3);
        lm.clear();
        assert_eq!(lm.live_line_count(), 0);
    }

    #[test]
    fn test_line_map_occupancy() {
        let mut lm = LineMap::new();
        for i in 0..LINES_PER_BLOCK / 2 {
            lm.mark(i);
        }
        assert_eq!(lm.occupancy_percent(), 50);
    }

    #[test]
    fn test_line_map_full_occupancy() {
        let mut lm = LineMap::new();
        for i in 0..LINES_PER_BLOCK {
            lm.mark(i);
        }
        assert_eq!(lm.live_line_count(), LINES_PER_BLOCK);
        assert_eq!(lm.occupancy_percent(), 100);
    }

    // ── ImmixBlock ────────────────────────────────────────────────────────

    #[test]
    fn test_block_bump_alloc_and_contains() {
        let mut block = ImmixBlock::new();
        let layout = Layout::from_size_align(64, 8).unwrap();
        let ptr = block.bump_alloc(layout).expect("block has space");
        assert!(block.contains(ptr));
        assert_eq!(block.used(), 64);
    }

    #[test]
    fn test_block_exhaustion() {
        let mut block = ImmixBlock::new();
        let layout = Layout::from_size_align(BLOCK_SIZE, 8).unwrap();
        assert!(block.bump_alloc(layout).is_some());
        let small = Layout::from_size_align(1, 1).unwrap();
        assert!(block.bump_alloc(small).is_none());
    }

    #[test]
    fn test_block_reset() {
        let mut block = ImmixBlock::new();
        let layout = Layout::from_size_align(128, 8).unwrap();
        block.bump_alloc(layout).unwrap();
        block.line_map_mut().mark(0);
        block.reset();
        assert_eq!(block.used(), 0);
        assert_eq!(block.line_map().live_line_count(), 0);
    }

    #[test]
    fn test_block_line_marking() {
        let mut block = ImmixBlock::new();
        let layout = Layout::from_size_align(256, 8).unwrap();
        let ptr = block.bump_alloc(layout).unwrap();
        block.mark_lines_for_object(ptr, 256);
        // 256 bytes spans lines 0 and 1 (each 128 bytes).
        assert!(block.line_map().is_marked(0));
        assert!(block.line_map().is_marked(1));
        assert!(!block.line_map().is_marked(2));
    }

    #[test]
    fn test_block_evacuation_threshold() {
        let mut block = ImmixBlock::new();
        // No lines marked → 0% occupancy → should evacuate.
        assert!(block.should_evacuate());
        // Mark 50% of lines → at threshold, not below.
        for i in 0..LINES_PER_BLOCK / 2 {
            block.line_map_mut().mark(i);
        }
        assert!(!block.should_evacuate());
    }

    #[test]
    fn test_block_remaining() {
        let mut block = ImmixBlock::new();
        assert_eq!(block.remaining(), BLOCK_SIZE);
        let layout = Layout::from_size_align(100, 8).unwrap();
        block.bump_alloc(layout).unwrap();
        assert_eq!(block.remaining(), BLOCK_SIZE - 100);
    }

    // ── ImmixSpace ────────────────────────────────────────────────────────

    #[test]
    fn test_space_obtain_and_return() {
        let mut space = ImmixSpace::new(4);
        let block = space.obtain_block().expect("should get a block");
        assert_eq!(block.used(), 0);
        space.return_full_block(block);
        assert_eq!(space.full_block_count(), 1);
    }

    #[test]
    fn test_space_exhaustion() {
        let mut space = ImmixSpace::new(2);
        let b1 = space.obtain_block().expect("first block");
        space.return_full_block(b1);
        let b2 = space.obtain_block().expect("second block");
        space.return_full_block(b2);
        assert!(space.obtain_block().is_none());
    }

    #[test]
    fn test_space_recycle() {
        let mut space = ImmixSpace::new(4);
        let mut block = ImmixBlock::new();
        let layout = Layout::from_size_align(64, 8).unwrap();
        block.bump_alloc(layout).unwrap();
        // Mark only 1 line (< 50%).
        block.line_map_mut().mark(0);
        space.return_full_block(block);

        let recycled = space.recycle_blocks();
        assert_eq!(recycled, 1);
        assert_eq!(space.full_block_count(), 0);
        assert_eq!(space.free_block_count(), 1);
    }

    #[test]
    fn test_space_capacity() {
        let space = ImmixSpace::new(4);
        assert_eq!(space.capacity(), 4 * BLOCK_SIZE);
    }

    // ── Tlab ──────────────────────────────────────────────────────────────

    #[test]
    fn test_tlab_allocate_within_block() {
        let mut space = ImmixSpace::new(4);
        let mut tlab = Tlab::new();
        let layout = Layout::from_size_align(64, 8).unwrap();

        let ptr = tlab.allocate(layout, &mut space);
        assert!(ptr.is_some());
        assert!(tlab.has_block());
        assert_eq!(tlab.bytes_allocated(), 64);
    }

    #[test]
    fn test_tlab_block_transition() {
        let mut space = ImmixSpace::new(4);
        let mut tlab = Tlab::new();
        let layout = Layout::from_size_align(BLOCK_SIZE, 8).unwrap();
        tlab.allocate(layout, &mut space).expect("first block");

        let small = Layout::from_size_align(64, 8).unwrap();
        let ptr = tlab.allocate(small, &mut space);
        assert!(ptr.is_some());
        assert_eq!(space.full_block_count(), 1);
    }

    #[test]
    fn test_tlab_flush() {
        let mut space = ImmixSpace::new(4);
        let mut tlab = Tlab::new();
        let layout = Layout::from_size_align(64, 8).unwrap();
        tlab.allocate(layout, &mut space).unwrap();
        assert!(tlab.has_block());

        tlab.flush(&mut space);
        assert!(!tlab.has_block());
        assert_eq!(space.full_block_count(), 1);
    }

    #[test]
    fn test_tlab_remaining() {
        let mut space = ImmixSpace::new(4);
        let mut tlab = Tlab::new();
        assert_eq!(tlab.remaining(), 0);
        let layout = Layout::from_size_align(64, 8).unwrap();
        tlab.allocate(layout, &mut space).unwrap();
        assert_eq!(tlab.remaining(), BLOCK_SIZE - 64);
    }

    // ── ImmixCollector ────────────────────────────────────────────────────

    /// Allocate a `HeapObject` in an Immix block.
    fn alloc_in_block(block: &mut ImmixBlock) -> *mut HeapObject {
        let layout = Layout::new::<HeapObject>();
        let raw = block.bump_alloc(layout).expect("block has space");
        // SAFETY: raw is valid and layout.size() bytes are accessible.
        unsafe { std::ptr::write_bytes(raw, 0, layout.size()) };
        let obj = raw as *mut HeapObject;
        // SAFETY: obj is valid, zero-initialised.
        unsafe { (*obj).init_alloc_size(layout.size() as u32) };
        obj
    }

    #[test]
    fn test_collector_mark_roots() {
        let mut block = ImmixBlock::new();
        let obj = alloc_in_block(&mut block);

        let mut root: *mut HeapObject = obj;
        let roots = [&raw mut root as *mut *mut HeapObject];
        let mut collector = ImmixCollector::new();
        let mut blocks = [block];

        // SAFETY: root is a valid heap object within the block.
        unsafe { collector.mark(&roots, &mut blocks) };

        assert!(collector.is_marked(obj));
        assert!(blocks[0].line_map().is_marked(0));
    }

    #[test]
    fn test_collector_skips_null_roots() {
        let mut collector = ImmixCollector::new();
        let mut root: *mut HeapObject = std::ptr::null_mut();
        let roots = [&raw mut root as *mut *mut HeapObject];
        let mut blocks: [ImmixBlock; 0] = [];

        // SAFETY: null root should be skipped.
        unsafe { collector.mark(&roots, &mut blocks) };

        assert!(collector.mark_set.is_empty());
    }

    #[test]
    fn test_collector_evacuation_candidates() {
        let mut block = ImmixBlock::new();
        let _obj = alloc_in_block(&mut block);
        // Only 1 line marked → below 50% → evacuation candidate.
        block.line_map_mut().mark(0);

        let collector = ImmixCollector::new();
        let mut blocks = vec![block];
        let count = collector.select_evacuation_candidates(&mut blocks);

        assert_eq!(count, 1);
        assert!(blocks[0].is_evacuate_candidate());
    }

    #[test]
    fn test_collector_full_cycle() {
        let mut space = ImmixSpace::new(8);
        let mut tlab = Tlab::new();

        let layout = Layout::new::<HeapObject>();
        let raw = tlab.allocate(layout, &mut space).expect("alloc succeeds");
        // SAFETY: raw is valid and layout.size() bytes are accessible.
        unsafe { std::ptr::write_bytes(raw, 0, layout.size()) };
        let obj = raw as *mut HeapObject;
        // SAFETY: obj is valid, zero-initialised.
        unsafe { (*obj).init_alloc_size(layout.size() as u32) };

        let mut root: *mut HeapObject = obj;
        let mut roots = [&raw mut root as *mut *mut HeapObject];

        tlab.flush(&mut space);

        let mut collector = ImmixCollector::new();
        // SAFETY: root is a valid heap object.
        unsafe { collector.collect(&mut roots, &mut space) };

        assert!(!root.is_null(), "root must survive collection");
    }

    #[test]
    fn test_collector_unreachable_block_recycled() {
        let mut space = ImmixSpace::new(8);

        // Create a block with an object but do NOT root it.
        let mut block = ImmixBlock::new();
        let _dead = alloc_in_block(&mut block);
        space.return_full_block(block);

        let mut roots: [*mut *mut HeapObject; 0] = [];
        let mut collector = ImmixCollector::new();

        // SAFETY: no roots → all objects are dead.
        unsafe { collector.collect(&mut roots, &mut space) };

        // The block should have been recycled.
        assert_eq!(space.full_block_count(), 0);
        assert!(space.free_block_count() > 0);
    }

    // ── ConcurrentMarker ──────────────────────────────────────────────────

    #[test]
    fn test_concurrent_marker_marks_roots() {
        let mut old = OldSpace::new(65536);
        let layout = Layout::new::<HeapObject>();
        let raw = old.bump_alloc(layout).expect("old space has room");
        // SAFETY: raw is valid.
        unsafe { std::ptr::write_bytes(raw, 0, layout.size()) };
        let obj = raw as *mut HeapObject;
        // SAFETY: obj is valid.
        unsafe { (*obj).init_alloc_size(layout.size() as u32) };

        let marker = ConcurrentMarker::new();
        let roots = [obj as *const HeapObject];

        // SAFETY: obj is a valid old-space object.
        unsafe { marker.mark_old_generation(&roots, &old) };

        let result = marker.take_result().expect("result should be available");
        assert_eq!(result.marked_count, 1);
        assert!(result.live_set.contains(&(obj as usize)));
    }

    #[test]
    fn test_concurrent_marker_no_roots() {
        let old = OldSpace::new(65536);
        let marker = ConcurrentMarker::new();
        let roots: &[*const HeapObject] = &[];

        // SAFETY: no roots to check.
        unsafe { marker.mark_old_generation(roots, &old) };

        let result = marker.take_result().expect("result should be available");
        assert_eq!(result.marked_count, 0);
    }

    #[test]
    fn test_concurrent_marker_take_result_is_consume() {
        let old = OldSpace::new(65536);
        let marker = ConcurrentMarker::new();
        let roots: &[*const HeapObject] = &[];

        // SAFETY: no roots.
        unsafe { marker.mark_old_generation(roots, &old) };

        assert!(marker.take_result().is_some());
        assert!(
            marker.take_result().is_none(),
            "second take must return None"
        );
    }

    #[test]
    fn test_concurrent_marker_result_channel() {
        let marker = ConcurrentMarker::new();
        let channel = marker.result_channel();

        // Channel should be empty initially.
        let guard = channel.lock().unwrap();
        assert!(guard.is_none());
    }
}
