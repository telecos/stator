#!/usr/bin/env python3
"""Apply all global IC optimizations to interpreter/mod.rs in one pass."""
import sys

FILE = r'C:\Work\stator\crates\stator_core\src\interpreter\mod.rs'

with open(FILE, 'r', encoding='utf-8') as f:
    content = f.read()

original = content  # Save for comparison

# ── 1. Replace type alias with constants ──────────────────────────────
content = content.replace(
    'type GlobalIcCache = HashMap<u32, (usize, u64)>;',
    '''/// Number of slots in the direct-mapped global variable inline cache.
const GLOBAL_IC_SLOTS: usize = 32;
/// Bitmask for mapping a constant-pool index to an IC slot.
const GLOBAL_IC_MASK: u32 = (GLOBAL_IC_SLOTS as u32) - 1;
/// Sentinel tag indicating an empty IC slot.
const GLOBAL_IC_EMPTY: u32 = u32::MAX;
/// A single IC entry: `(name_idx tag, slot_index, generation)`.
type GlobalIcEntry = (u32, usize, u64);
/// Fixed-size direct-mapped global variable IC array.
type GlobalIcArray = [GlobalIcEntry; GLOBAL_IC_SLOTS];
/// Empty IC array constant used to initialise / reset the cache.
const GLOBAL_IC_INIT: GlobalIcArray = [(GLOBAL_IC_EMPTY, 0, 0); GLOBAL_IC_SLOTS];'''
)

# ── 2. Replace field definition ───────────────────────────────────────
content = content.replace(
    '''    /// Global variable inline cache: `constant_pool_idx \u2192 (slot_index, generation)`.
    /// Maps bytecode constant-pool indices to indexed `GlobalEnv` slots for
    /// O(1) global variable access.  Lazily allocated on first IC miss.
    pub global_ic: Option<Box<GlobalIcCache>>,''',
    '''    /// Direct-mapped global variable IC: `[(name_idx, slot_index, generation); 32]`.
    /// Indexed by `name_idx & 31`.  Tag `GLOBAL_IC_EMPTY` (`u32::MAX`) = vacant.
    /// Always present \u2014 no `Option` / `Box` indirection on the hot path.
    pub global_ic: GlobalIcArray,
    /// `true` once the IC has been seeded from the shared `BytecodeArray` cache.
    global_ic_seeded: bool,'''
)

# ── 3. Replace constructor initializations ────────────────────────────
content = content.replace(
    '            global_ic: None,\n            global_cache: None,',
    '            global_ic: GLOBAL_IC_INIT,\n            global_ic_seeded: false,\n            global_cache: None,'
)

# ── 4. Replace all frame.global_ic.is_some() ─────────────────────────
content = content.replace('frame.global_ic.is_some()', 'frame.global_ic_seeded')

# ── 5. Replace global_ic_mut method ──────────────────────────────────
content = content.replace(
    '''    #[inline]
    fn global_ic_mut(&mut self) -> &mut GlobalIcCache {
        if self.global_ic.is_none() {
            self.global_ic = self
                .bytecode_array
                .shared_global_ic()
                .or_else(|| Some(Box::default()));
        }
        self.global_ic.as_mut().unwrap()
    }''',
    '''    /// O(1) lookup in the direct-mapped global IC.
    #[inline(always)]
    pub(super) fn global_ic_get(&self, name_idx: u32) -> Option<(usize, u64)> {
        let entry = &self.global_ic[(name_idx & GLOBAL_IC_MASK) as usize];
        if entry.0 == name_idx {
            Some((entry.1, entry.2))
        } else {
            None
        }
    }

    /// Insert / overwrite an entry in the direct-mapped global IC.
    /// On first write, seeds from the shared `BytecodeArray` cache so that
    /// subsequent calls to the same function start warm.
    #[inline(always)]
    pub(super) fn global_ic_put(&mut self, name_idx: u32, slot_idx: usize, generation_val: u64) {
        if !self.global_ic_seeded {
            self.global_ic_seeded = true;
            if let Some(shared) = self.bytecode_array.shared_global_ic() {
                for (&k, &(si, g)) in shared.iter() {
                    self.global_ic[(k & GLOBAL_IC_MASK) as usize] = (k, si, g);
                }
            }
        }
        self.global_ic[(name_idx & GLOBAL_IC_MASK) as usize] =
            (name_idx, slot_idx, generation_val);
    }

    /// Reset the direct-mapped global IC (e.g. on tail-call reuse).
    #[inline(always)]
    pub(super) fn global_ic_reset(&mut self) {
        self.global_ic = GLOBAL_IC_INIT;
        self.global_ic_seeded = false;
    }'''
)

# ── 6. Replace store_global ───────────────────────────────────────────
content = content.replace(
    '''        // IC fast path: known slot, generation matches.
        if let Some(ref ic) = self.global_ic
            && let Some(&(slot_idx, cached_gen)) = ic.get(&name_idx)
        {
            let current_gen = self.global_env.borrow().generation();
            if current_gen == cached_gen {
                set_function_name_if_missing(&value, name);
                self.global_env
                    .borrow_mut()
                    .store_by_index_sync(slot_idx, name, value.clone());
                self.global_cache_put(name, value);
                return;
            }
        }
        // Slow path: full insert + IC population.
        set_function_name_if_missing(&value, name);
        let mut env = self.global_env.borrow_mut();
        env.insert(name.to_string(), value.clone());
        let slot_gen = env.slot_index_for(name).map(|idx| (idx, env.generation()));
        drop(env);
        if let Some((slot_idx, cached_gen)) = slot_gen {
            self.global_ic_mut()
                .insert(name_idx, (slot_idx, cached_gen));
        }
        self.global_cache_put(name, value);''',
    '''        // IC fast path: known slot, generation matches.
        // Single borrow_mut avoids the double RefCell overhead of
        // borrow() for generation + borrow_mut() for store.
        if let Some((slot_idx, cached_gen)) = self.global_ic_get(name_idx) {
            let mut env = self.global_env.borrow_mut();
            if env.generation() == cached_gen {
                set_function_name_if_missing(&value, name);
                env.store_by_index_sync(slot_idx, name, value.clone());
                drop(env);
                self.global_cache_put(name, value);
                return;
            }
        }
        // Slow path: full insert + IC population.
        set_function_name_if_missing(&value, name);
        let mut env = self.global_env.borrow_mut();
        env.insert(name.to_string(), value.clone());
        let slot_gen = env.slot_index_for(name).map(|idx| (idx, env.generation()));
        drop(env);
        if let Some((slot_idx, generation_val)) = slot_gen {
            self.global_ic_put(name_idx, slot_idx, generation_val);
        }
        self.global_cache_put(name, value);'''
)

# ── 7. Replace teardown code ─────────────────────────────────────────
content = content.replace(
    '''        if let Some(ic) = frame.global_ic.take() {
            frame.bytecode_array.set_shared_global_ic(ic);
        }''',
    '''        // Write back the direct-mapped global IC to the shared HashMap cache
        // on the BytecodeArray so subsequent invocations start warm.
        if frame.global_ic_seeded {
            let mut map: HashMap<u32, (usize, u64)> = HashMap::new();
            for &(tag, slot_idx, generation_val) in &frame.global_ic {
                if tag != GLOBAL_IC_EMPTY {
                    map.insert(tag, (slot_idx, generation_val));
                }
            }
            if !map.is_empty() {
                frame.bytecode_array.set_shared_global_ic(Box::new(map));
            }
        }'''
)

# ── 8. Replace SMI-mode LdaGlobal IC hit ─────────────────────────────
content = content.replace(
    '''                                // IC fast path: read directly from the GlobalEnv
                                // slot, avoiding get_string_constant + load_global.
                                let ic_hit = frame
                                    .global_ic
                                    .as_ref()
                                    .and_then(|ic| ic.get(&name_idx).copied());
                                if let Some((slot_idx, cached_gen)) = ic_hit {''',
    '''                                // IC fast path: read directly from the GlobalEnv
                                // slot, avoiding get_string_constant + load_global.
                                let ic_hit = frame.global_ic_get(name_idx);
                                if let Some((slot_idx, cached_gen)) = ic_hit {'''
)

# ── 9. Replace SMI-mode StaGlobal IC hit ─────────────────────────────
content = content.replace(
    '''                                // IC fast path: inline the store and refresh the
                                // IC entry so subsequent iterations stay fast.
                                let ic_hit = frame
                                    .global_ic
                                    .as_ref()
                                    .and_then(|ic| ic.get(&name_idx).copied());
                                if let Some((slot_idx, cached_gen)) = ic_hit {
                                    let cur_gen = frame.global_env.borrow().generation();
                                    if cur_gen == cached_gen {
                                        frame.pc = pc;
                                        frame.accumulator = val.cheap_clone();
                                        let name = frame.get_string_constant(name_idx)?;
                                        frame.global_env.borrow_mut().store_by_index_sync(
                                            slot_idx,
                                            &name,
                                            val.cheap_clone(),
                                        );
                                        let new_gen = frame.global_env.borrow().generation();
                                        frame.global_ic_mut().insert(name_idx, (slot_idx, new_gen));
                                        frame.global_cache_put(&name, val);
                                        continue 'smi;
                                    }
                                }''',
    '''                                // IC fast path: inline the store and refresh the
                                // IC entry so subsequent iterations stay fast.
                                let ic_hit = frame.global_ic_get(name_idx);
                                if let Some((slot_idx, cached_gen)) = ic_hit {
                                    let cur_gen = frame.global_env.borrow().generation();
                                    if cur_gen == cached_gen {
                                        frame.pc = pc;
                                        frame.accumulator = val.cheap_clone();
                                        let name = frame.get_string_constant(name_idx)?;
                                        frame.global_env.borrow_mut().store_by_index_sync(
                                            slot_idx,
                                            &name,
                                            val.cheap_clone(),
                                        );
                                        let new_gen = frame.global_env.borrow().generation();
                                        frame.global_ic_put(name_idx, slot_idx, new_gen);
                                        frame.global_cache_put(&name, val);
                                        continue 'smi;
                                    }
                                }'''
)

# ── 10. Replace dispatch-loop inlined LdaGlobal ──────────────────────
content = content.replace(
    '''                            if let Some(ref ic) = frame.global_ic
                                && let Some(&(slot_idx, cached_gen)) = ic.get(&name_idx)
                            {
                                let env = frame.global_env.borrow();
                                if env.generation() == cached_gen {
                                    let value = env.get_by_index(slot_idx).clone();
                                    drop(env);
                                    if value != JsValue::TheHole {
                                        acc = value;
                                        continue 'dispatch;
                                    }
                                }
                            }''',
    '''                            if let Some((slot_idx, cached_gen)) =
                                frame.global_ic_get(name_idx)
                            {
                                let env = frame.global_env.borrow();
                                if env.generation() == cached_gen {
                                    let value = env.get_by_index(slot_idx).cheap_clone();
                                    drop(env);
                                    if value != JsValue::TheHole {
                                        acc = value;
                                        continue 'dispatch;
                                    }
                                }
                            }'''
)

# ── Verify ────────────────────────────────────────────────────────────
checks = {
    'global_ic_mut': content.count('global_ic_mut'),
    'global_ic.is_some()': content.count('.global_ic.is_some()'),
    'global_ic.take()': content.count('global_ic.take()'),
    'Some(ref ic) = frame.global_ic': content.count('Some(ref ic) = frame.global_ic'),
    'global_ic.as_ref()': content.count('.global_ic.as_ref()'),
    'GLOBAL_IC_INIT': content.count('GLOBAL_IC_INIT'),
    'global_ic_get': content.count('global_ic_get'),
    'global_ic_put': content.count('global_ic_put'),
    'global_ic_reset': content.count('global_ic_reset'),
    'global_ic_seeded': content.count('global_ic_seeded'),
}

all_good = True
for name, count in checks.items():
    print(f'  {name}: {count}')
    if name in ('global_ic_mut', 'global_ic.is_some()', 'global_ic.take()', 
                'Some(ref ic) = frame.global_ic', 'global_ic.as_ref()'):
        if count > 0:
            all_good = False
            print(f'    *** ERROR: should be 0!')

if not all_good:
    print('\nERROR: Some old references remain!')
    sys.exit(1)

if content == original:
    print('\nERROR: No changes made!')
    sys.exit(1)

with open(FILE, 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)
print(f'\nSuccessfully wrote {len(content)} bytes to {FILE}')
