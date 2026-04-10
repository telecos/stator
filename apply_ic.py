#!/usr/bin/env python3
"""Apply IC fast path optimization to SMI loop. Single-pass, all fixes included."""
import sys, os

FILE = 'crates/stator_core/src/interpreter/mod.rs'

with open(FILE, 'rb') as f:
    original = f.read()

lines = list(original.split(b'\n'))
print(f"Read {len(lines)} lines ({len(original)} bytes)")

def find(pattern, start=0, end=None):
    if end is None: end = len(lines)
    for i in range(start, end):
        if pattern in lines[i]:
            return i
    raise ValueError(f"Not found: {pattern!r} [{start}..{end})")

def ins_before(idx, block):
    global lines
    lines[idx:idx] = block

def ins_after(idx, block):
    global lines
    lines[idx+1:idx+1] = block

I = b'                                '  # 32 spaces
J = b'                            '      # 28 spaces

SYNC_I = [
    I + b'if smi_vars_dirty {',
    I + b'    frame.global_env.borrow_mut().sync_slots_to_vars();',
    I + b'    smi_vars_dirty = false;',
    I + b'}',
]
SYNC_J = [
    J + b'    if smi_vars_dirty {',
    J + b'        frame.global_env.borrow_mut().sync_slots_to_vars();',
    J + b'    }',
]

# 1. sync_slots_to_vars
sc = find(b'pub fn slot_count')
fc = find(b'}', sc + 1)
ic = find(b'}', fc + 1)
ins_before(ic, [
    b'',
    b'    /// Copy every slot value back into the `vars` HashMap.',
    b'    pub fn sync_slots_to_vars(&mut self) {',
    b'        for (name, &idx) in &self.name_to_index {',
    b'            if idx < self.slots.len() && let Some(v) = self.vars.get_mut(name) {',
    b'                *v = self.slots[idx].cheap_clone();',
    b'            }',
    b'        }',
    b'    }',
])
print("[1] sync_slots_to_vars")

# 2. env_ptr + smi_vars_dirty
smi = find(b"'smi: loop {")
ins_before(smi, [
    J + b'    // SAFETY: GlobalEnv outlives the SMI loop; no other',
    J + b'    // RefCell borrow is active while we access slots.',
    J + b'    let env_ptr = frame.global_env.as_ptr();',
    J + b'    let mut smi_vars_dirty = false;',
])
print("[2] env_ptr + smi_vars_dirty")

# 3. LdaGlobal IC fast path
smi = find(b"'smi: loop {")
lda = find(b'Opcode::LdaGlobal =>', smi)
sc = find(b'// Sync state before fallible frame methods.', lda)
ins_before(sc, [
    I + b'// IC fast path: read directly from slots via raw pointer.',
    I + b'if let Some(&(slot_idx, _cached_gen)) =',
    I + b'    frame.global_ic.as_ref().and_then(|ic| ic.get(&name_idx))',
    I + b'{',
    I + b'    // SAFETY: env_ptr valid, slot_idx from prior IC population.',
    I + b'    let val = unsafe { &(&(*env_ptr).slots)[slot_idx] };',
    I + b'    if let JsValue::Smi(v) = val {',
    I + b'        sa = *v;',
    I + b'        smi_acc_bool = false;',
    I + b'        smi_acc_spilled = false;',
    I + b'        hot_acc = Some(NanBoxedValue::from_smi(*v));',
    I + b"        continue 'smi;",
    I + b'    } else {',
    I + b'        acc = val.cheap_clone();',
    I + b'        smi_acc_spilled = true;',
    I + b'        smi_acc_bool = false;',
    I + b'        hot_acc = None;',
    I + b"        continue 'smi;",
    I + b'    }',
    I + b'}',
    I + b'// IC miss: sync vars before slow path.',
] + SYNC_I)
print("[3] LdaGlobal IC fast path")

# 3b. LdaGlobal IC populate
smi = find(b"'smi: loop {")
lda = find(b'Opcode::LdaGlobal =>', smi)
sta = find(b'Opcode::StaGlobal =>', lda)
hc = sta - 1
while lines[hc].strip() != b'}': hc -= 1
ins_before(hc, [
    I + b'// Populate IC for next iteration.',
    I + b'{',
    I + b'    let env_borrow = frame.global_env.borrow();',
    I + b'    let ic_info = env_borrow',
    I + b'        .slot_index_for(&name)',
    I + b'        .map(|si| (si, env_borrow.generation));',
    I + b'    drop(env_borrow);',
    I + b'    if let Some((slot_idx, cur_gen)) = ic_info {',
    I + b'        frame.global_ic_put(name_idx, slot_idx, cur_gen);',
    I + b'    }',
    I + b'}',
])
print("[3b] LdaGlobal IC populate")

# 4. StaGlobal IC fast path
smi = find(b"'smi: loop {")
sta = find(b'Opcode::StaGlobal =>', smi)
sm = find(b'acc = materialize_acc!();', sta)
ins_before(sm, [
    I + b'// IC fast path: write directly to slot via raw pointer.',
    I + b'{',
    I + b'    let val = materialize_acc!();',
    I + b'    if let Some(&(slot_idx, _cached_gen)) =',
    I + b'        frame.global_ic.as_ref().and_then(|ic| ic.get(&name_idx))',
    I + b'    {',
    I + b'        // SAFETY: env_ptr valid, slot_idx from prior IC population.',
    I + b'        unsafe {',
    I + b'            (&mut (*env_ptr).slots)[slot_idx] = val.cheap_clone();',
    I + b'            (*env_ptr).generation = (*env_ptr).generation.wrapping_add(1);',
    I + b'        }',
    I + b'        smi_vars_dirty = true;',
    I + b"        continue 'smi;",
    I + b'    }',
    I + b'}',
    I + b'// IC miss: sync vars before slow path.',
] + SYNC_I)
print("[4] StaGlobal IC fast path")

# 4b. StaGlobal IC populate
smi = find(b"'smi: loop {")
sta = find(b'Opcode::StaGlobal =>', smi)
sg = find(b'frame.store_global(', sta)
no = find(b'Opcode::', sg + 1)
sc2 = no - 1
while lines[sc2].strip() != b'}': sc2 -= 1
ins_before(sc2, [
    I + b'// Populate IC for next iteration.',
    I + b'{',
    I + b'    let env_borrow = frame.global_env.borrow();',
    I + b'    let ic_info = env_borrow',
    I + b'        .slot_index_for(&name)',
    I + b'        .map(|si| (si, env_borrow.generation));',
    I + b'    drop(env_borrow);',
    I + b'    if let Some((slot_idx, cur_gen)) = ic_info {',
    I + b'        frame.global_ic_put(name_idx, slot_idx, cur_gen);',
    I + b'    }',
    I + b'}',
])
print("[4b] StaGlobal IC populate")

# 5. LdaGlobalStar sync
smi = find(b"'smi: loop {")
lgs = find(b'Opcode::LdaGlobalStar =>', smi)
lm = find(b'acc = materialize_acc!();', lgs)
ins_before(lm, [
    I + b'// Flush deferred slot writes so vars is up-to-date.',
] + SYNC_I)
print("[5] LdaGlobalStar sync")

# 6-9. Sync before function calls
for v in [b'CallUndefinedReceiver0', b'CallUndefinedReceiver1',
          b'CallUndefinedReceiver2', b'CallAnyReceiver']:
    smi = find(b"'smi: loop {")
    tag = b'Opcode::' + v + b' =>'
    try:
        cl = find(tag, smi)
        bl = cl
        while b'{' not in lines[bl]: bl += 1
        ins_after(bl, [
            I + b'// Flush deferred slot writes before function call.',
        ] + SYNC_I)
        print(f"[6-9] sync before {v.decode()}")
    except ValueError:
        print(f"  WARN: {v.decode()} not found")

# 10a. Wildcard exit sync
smi = find(b"'smi: loop {")
wc = find(b'// Unsupported opcode', smi)
wm = find(b'acc = materialize_acc!();', wc)
ins_before(wm, [
    b'                                // Flush deferred slot writes.',
    b'                                if smi_vars_dirty {',
    b'                                    frame.global_env.borrow_mut().sync_slots_to_vars();',
    b'                                }',
])
print("[10a] wildcard exit sync")

# 10b. loop_end_pc exit sync
smi = find(b"'smi: loop {")
le = find(b'pc >= frame.loop_end_pc', smi)
lm2 = find(b'acc = materialize_acc!();', le)
ins_before(lm2, [
    b'                            // Flush deferred slot writes.',
    b'                            if smi_vars_dirty {',
    b'                                frame.global_env.borrow_mut().sync_slots_to_vars();',
    b'                            }',
])
print("[10b] loop_end_pc exit sync")

# Write
result = b'\n'.join(lines).replace(b'\r\n', b'\n')
assert b'env_ptr' in result
assert b'sync_slots_to_vars' in result
assert b'global_ic_put' in result

with open(FILE, 'wb') as f:
    f.write(result)

# Verify
with open(FILE, 'rb') as f:
    v = f.read()
assert b'env_ptr' in v, "WRITE FAILED"
print(f"\nDone! {len(lines)} lines, {len(result)} bytes. Verified.")
