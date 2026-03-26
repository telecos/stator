#!/usr/bin/env python3
"""Apply NaN-boxed accumulator optimization to the main dispatch loop.

v2: Fixes collapsible_if and dead-write clippy errors.
"""
import sys

FILE = 'crates/stator_core/src/interpreter/mod.rs'

with open(FILE, 'r', encoding='utf-8') as f:
    content = f.read()

def replace_one(src, old, new, label=""):
    idx = src.find(old)
    if idx == -1:
        print(f"ERROR: could not find anchor for {label!r}")
        print(f"  looking for: {old[:120]!r}...")
        sys.exit(1)
    if src.find(old, idx + 1) != -1:
        print(f"ERROR: multiple matches for {label!r}")
        sys.exit(1)
    return src[:idx] + new + src[idx + len(old):]

# ============================================================
# 1. Add acc_nb / acc_is_nb after acc initialization
# ============================================================
content = replace_one(content,
    "            let mut pc = frame.pc;\n"
    "            let mut acc = frame.accumulator.cheap_clone();\n"
    "\n"
    "            'dispatch: loop {",

    "            let mut pc = frame.pc;\n"
    "            let mut acc = frame.accumulator.cheap_clone();\n"
    "\n"
    "            // NaN-boxed shadow accumulator: tracks `acc` as a compact u64\n"
    "            // whenever the value is a primitive (Smi, Boolean, etc.).\n"
    "            // When `acc_is_nb` is true, `acc_nb` mirrors `acc` and hot\n"
    "            // consumers (JumpIfTrue/False, ToBooleanTrue/False) can\n"
    "            // branch on a single u64 compare instead of matching on the\n"
    "            // 24-byte JsValue enum.\n"
    "            let mut acc_nb: u64 = 0;\n"
    "            let mut acc_is_nb: bool = false;\n"
    "\n"
    "            'dispatch: loop {",
    "add acc_nb decls")

# ============================================================
# 2. SMI loop exit
# ============================================================
content = replace_one(content,
    "                        continue 'smi;\n"
    "                    }\n"
    "                    continue 'dispatch;\n"
    "                }\n"
    "\n"
    "                match instr.opcode {",

    "                        continue 'smi;\n"
    "                    }\n"
    "                    // Exiting SMI mode — accumulator type is unknown.\n"
    "                    acc_is_nb = false;\n"
    "                    continue 'dispatch;\n"
    "                }\n"
    "\n"
    "                match instr.opcode {",
    "smi exit")

# ============================================================
# 3. LdaSmi
# ============================================================
content = replace_one(content,
    "                    Opcode::LdaSmi => {\n"
    "                        // SAFETY: Bytecode generator guarantees operand[0]\n"
    "                        // is Immediate for LdaSmi.\n"
    "                        let val = unsafe { operand_imm_unchecked(instr, 0) };\n"
    "                        acc = JsValue::Smi(val);\n"
    "                        continue 'dispatch;\n"
    "                    }",

    "                    Opcode::LdaSmi => {\n"
    "                        // SAFETY: Bytecode generator guarantees operand[0]\n"
    "                        // is Immediate for LdaSmi.\n"
    "                        let val = unsafe { operand_imm_unchecked(instr, 0) };\n"
    "                        acc = JsValue::Smi(val);\n"
    "                        acc_nb = NanBoxedValue::from_smi(val).to_bits();\n"
    "                        acc_is_nb = true;\n"
    "                        continue 'dispatch;\n"
    "                    }",
    "LdaSmi")

# ============================================================
# 4. LdaSmiStar
# ============================================================
content = replace_one(content,
    "                    Opcode::LdaSmiStar => {\n"
    "                        let val = unsafe { operand_imm_unchecked(instr, 0) };\n"
    "                        let dst = unsafe { operand_reg_unchecked(instr, 1) };\n"
    "                        acc = JsValue::Smi(val);\n"
    "                        unsafe { frame.write_reg_unchecked(dst, acc.cheap_clone()) };\n"
    "                        continue 'dispatch;\n"
    "                    }",

    "                    Opcode::LdaSmiStar => {\n"
    "                        let val = unsafe { operand_imm_unchecked(instr, 0) };\n"
    "                        let dst = unsafe { operand_reg_unchecked(instr, 1) };\n"
    "                        acc = JsValue::Smi(val);\n"
    "                        acc_nb = NanBoxedValue::from_smi(val).to_bits();\n"
    "                        acc_is_nb = true;\n"
    "                        unsafe { frame.write_reg_unchecked(dst, acc.cheap_clone()) };\n"
    "                        continue 'dispatch;\n"
    "                    }",
    "LdaSmiStar")

# ============================================================
# 5. Ldar
# ============================================================
content = replace_one(content,
    "                        let val = unsafe { frame.read_reg_unchecked(reg) };\n"
    "                        acc = val.cheap_clone();\n"
    "                        continue 'dispatch;\n"
    "                    }\n"
    "                    Opcode::LdaUndefined",

    "                        let val = unsafe { frame.read_reg_unchecked(reg) };\n"
    "                        acc = val.cheap_clone();\n"
    "                        acc_is_nb = false;\n"
    "                        continue 'dispatch;\n"
    "                    }\n"
    "                    Opcode::LdaUndefined",
    "Ldar")

# ============================================================
# 6. Lda constants
# ============================================================
for (opname, jsval, nbcall) in [
    ("LdaUndefined", "JsValue::Undefined", "NanBoxedValue::undefined().to_bits()"),
    ("LdaZero", "JsValue::Smi(0)", "NanBoxedValue::from_smi(0).to_bits()"),
    ("LdaTrue", "JsValue::Boolean(true)", "NanBoxedValue::from_boolean(true).to_bits()"),
    ("LdaFalse", "JsValue::Boolean(false)", "NanBoxedValue::from_boolean(false).to_bits()"),
    ("LdaNull", "JsValue::Null", "NanBoxedValue::null().to_bits()"),
    ("LdaTheHole", "JsValue::TheHole", "NanBoxedValue::the_hole().to_bits()"),
]:
    old = (
        f"                    Opcode::{opname} => {{\n"
        f"                        acc = {jsval};\n"
        f"                        continue 'dispatch;\n"
        f"                    }}"
    )
    new = (
        f"                    Opcode::{opname} => {{\n"
        f"                        acc = {jsval};\n"
        f"                        acc_nb = {nbcall};\n"
        f"                        acc_is_nb = true;\n"
        f"                        continue 'dispatch;\n"
        f"                    }}"
    )
    content = replace_one(content, old, new, opname)

# ============================================================
# 7. Heap-value creators — clear acc_is_nb
# ============================================================
content = replace_one(content,
    "                    Opcode::CreateEmptyArrayLiteral => {\n"
    "                        acc = JsValue::Array(Rc::new(RefCell::new(Vec::new())));\n"
    "                        continue 'dispatch;\n"
    "                    }",
    "                    Opcode::CreateEmptyArrayLiteral => {\n"
    "                        acc = JsValue::Array(Rc::new(RefCell::new(Vec::new())));\n"
    "                        acc_is_nb = false;\n"
    "                        continue 'dispatch;\n"
    "                    }",
    "CreateEmptyArrayLiteral")

content = replace_one(content,
    "                    Opcode::CreateEmptyObjectLiteral => {\n"
    "                        acc = JsValue::PlainObject(Rc::new(RefCell::new(\n"
    "                            PropertyMap::with_capacity(4),\n"
    "                        )));\n"
    "                        continue 'dispatch;\n"
    "                    }",
    "                    Opcode::CreateEmptyObjectLiteral => {\n"
    "                        acc = JsValue::PlainObject(Rc::new(RefCell::new(\n"
    "                            PropertyMap::with_capacity(4),\n"
    "                        )));\n"
    "                        acc_is_nb = false;\n"
    "                        continue 'dispatch;\n"
    "                    }",
    "CreateEmptyObjectLiteral")

# ============================================================
# 8. Add/Sub/Mul — Smi arm tracking
# ============================================================
for (op, symbol) in [("Add", "+"), ("Sub", "-"), ("Mul", "*")]:
    old_smi_arm = (
        f"                                (JsValue::Smi(a), JsValue::Smi(b)) => {{\n"
        f"                                    acc = if let Some(result) = a.checked_{op.lower()}(*b) {{\n"
        f"                                        JsValue::Smi(result)\n"
        f"                                    }} else {{\n"
        f"                                        JsValue::HeapNumber(*a as f64 {symbol} *b as f64)\n"
        f"                                    }};\n"
        f"                                    continue 'dispatch;\n"
        f"                                }}"
    )
    new_smi_arm = (
        f"                                (JsValue::Smi(a), JsValue::Smi(b)) => {{\n"
        f"                                    if let Some(result) = a.checked_{op.lower()}(*b) {{\n"
        f"                                        acc = JsValue::Smi(result);\n"
        f"                                        acc_nb = NanBoxedValue::from_smi(result).to_bits();\n"
        f"                                        acc_is_nb = true;\n"
        f"                                    }} else {{\n"
        f"                                        acc = JsValue::HeapNumber(*a as f64 {symbol} *b as f64);\n"
        f"                                        acc_is_nb = false;\n"
        f"                                    }}\n"
        f"                                    continue 'dispatch;\n"
        f"                                }}"
    )
    content = replace_one(content, old_smi_arm, new_smi_arm, f"{op} Smi arm")

# ============================================================
# 9. AddSmi/SubSmi/MulSmi — NaN-box fast path (collapsed if)
# ============================================================
for (op, method, symbol) in [("AddSmi", "checked_add", "+"), ("SubSmi", "checked_sub", "-"), ("MulSmi", "checked_mul", "*")]:
    old = (
        f"                    Opcode::{op} => {{\n"
        f"                        if let Operand::Immediate(imm) = *instr.operand(0) {{\n"
        f"                            match &acc {{\n"
        f"                                JsValue::Smi(n) => {{\n"
        f"                                    acc = if let Some(result) = n.{method}(imm) {{\n"
        f"                                        JsValue::Smi(result)\n"
        f"                                    }} else {{\n"
        f"                                        JsValue::HeapNumber(*n as f64 {symbol} imm as f64)\n"
        f"                                    }};\n"
        f"                                    continue 'dispatch;\n"
        f"                                }}"
    )
    new = (
        f"                    Opcode::{op} => {{\n"
        f"                        if let Operand::Immediate(imm) = *instr.operand(0) {{\n"
        f"                            // NaN-box fast path for known-Smi accumulator.\n"
        f"                            if acc_is_nb && NanBoxedValue::from_bits(acc_nb).is_smi() {{\n"
        f"                                let n = NanBoxedValue::from_bits(acc_nb).as_smi();\n"
        f"                                if let Some(result) = n.{method}(imm) {{\n"
        f"                                    acc_nb = NanBoxedValue::from_smi(result).to_bits();\n"
        f"                                    acc = JsValue::Smi(result);\n"
        f"                                    continue 'dispatch;\n"
        f"                                }}\n"
        f"                                acc = JsValue::HeapNumber(n as f64 {symbol} imm as f64);\n"
        f"                                acc_is_nb = false;\n"
        f"                                continue 'dispatch;\n"
        f"                            }}\n"
        f"                            match &acc {{\n"
        f"                                JsValue::Smi(n) => {{\n"
        f"                                    if let Some(result) = n.{method}(imm) {{\n"
        f"                                        acc = JsValue::Smi(result);\n"
        f"                                        acc_nb = NanBoxedValue::from_smi(result).to_bits();\n"
        f"                                        acc_is_nb = true;\n"
        f"                                    }} else {{\n"
        f"                                        acc = JsValue::HeapNumber(*n as f64 {symbol} imm as f64);\n"
        f"                                        acc_is_nb = false;\n"
        f"                                    }}\n"
        f"                                    continue 'dispatch;\n"
        f"                                }}"
    )
    content = replace_one(content, old, new, op)

# ============================================================
# 10. Inc / Dec — NaN-box fast path
# ============================================================
for (op, method, val) in [("Inc", "checked_add", "1"), ("Dec", "checked_sub", "1")]:
    old = (
        f"                    Opcode::{op} => {{\n"
        f"                        if let JsValue::Smi(n) = acc\n"
        f"                            && let Some(result) = n.{method}({val})\n"
        f"                        {{\n"
        f"                            acc = JsValue::Smi(result);\n"
        f"                            continue 'dispatch;\n"
        f"                        }}"
    )
    new = (
        f"                    Opcode::{op} => {{\n"
        f"                        // NaN-box fast path: known-Smi via shadow accumulator.\n"
        f"                        if acc_is_nb && NanBoxedValue::from_bits(acc_nb).is_smi() {{\n"
        f"                            let n = NanBoxedValue::from_bits(acc_nb).as_smi();\n"
        f"                            if let Some(result) = n.{method}({val}) {{\n"
        f"                                acc_nb = NanBoxedValue::from_smi(result).to_bits();\n"
        f"                                acc = JsValue::Smi(result);\n"
        f"                                continue 'dispatch;\n"
        f"                            }}\n"
        f"                        }} else if let JsValue::Smi(n) = acc\n"
        f"                            && let Some(result) = n.{method}({val})\n"
        f"                        {{\n"
        f"                            acc = JsValue::Smi(result);\n"
        f"                            acc_nb = NanBoxedValue::from_smi(result).to_bits();\n"
        f"                            acc_is_nb = true;\n"
        f"                            continue 'dispatch;\n"
        f"                        }}"
    )
    content = replace_one(content, old, new, op)

# ============================================================
# 11. Comparisons — NaN-box fast path (collapsed ifs) + tracking
# ============================================================
for (op, cmp_op) in [
    ("TestLessThan", "<"), ("TestGreaterThan", ">"),
    ("TestLessThanOrEqual", "<="), ("TestGreaterThanOrEqual", ">="),
    ("TestNotEqual", "!="),
]:
    # Find the handler and insert NaN-box fast path before the match
    old_header = f"                    Opcode::{op} => {{\n"
    idx = content.find(old_header)
    if idx == -1:
        print(f"ERROR: could not find {op}"); sys.exit(1)
    # Find "let rhs = unsafe { frame.read_reg_unchecked(v) };" after the handler start
    rhs_line = "                            let rhs = unsafe { frame.read_reg_unchecked(v) };\n"
    rhs_idx = content.find(rhs_line, idx)
    if rhs_idx == -1:
        print(f"ERROR: could not find rhs for {op}"); sys.exit(1)
    # Find the "match (&acc, rhs) {" after the rhs line
    match_line = "                            match (&acc, rhs) {\n"
    match_idx = content.find(match_line, rhs_idx)
    if match_idx == -1:
        print(f"ERROR: could not find match for {op}"); sys.exit(1)

    nb_fast = (
        f"                            // NaN-box fast path: both Smi → raw i32 compare.\n"
        f"                            if acc_is_nb\n"
        f"                                && NanBoxedValue::from_bits(acc_nb).is_smi()\n"
        f"                                && let JsValue::Smi(b) = rhs\n"
        f"                            {{\n"
        f"                                let result = NanBoxedValue::from_bits(acc_nb).as_smi() {cmp_op} *b;\n"
        f"                                acc = JsValue::Boolean(result);\n"
        f"                                acc_nb = NanBoxedValue::from_boolean(result).to_bits();\n"
        f"                                acc_is_nb = true;\n"
        f"                                continue 'dispatch;\n"
        f"                            }}\n"
    )
    content = content[:match_idx] + nb_fast + content[match_idx:]

    # Update the Smi×Smi arm to track acc_nb
    old_smi = (
        f"                                (JsValue::Smi(a), JsValue::Smi(b)) => {{\n"
        f"                                    acc = JsValue::Boolean(*a {cmp_op} *b);\n"
        f"                                    continue 'dispatch;\n"
        f"                                }}"
    )
    new_smi = (
        f"                                (JsValue::Smi(a), JsValue::Smi(b)) => {{\n"
        f"                                    let result = *a {cmp_op} *b;\n"
        f"                                    acc = JsValue::Boolean(result);\n"
        f"                                    acc_nb = NanBoxedValue::from_boolean(result).to_bits();\n"
        f"                                    acc_is_nb = true;\n"
        f"                                    continue 'dispatch;\n"
        f"                                }}"
    )
    # Find the first occurrence after our op's header
    smi_idx = content.find(old_smi, idx)
    if smi_idx == -1:
        print(f"ERROR: could not find Smi arm for {op}"); sys.exit(1)
    content = content[:smi_idx] + new_smi + content[smi_idx + len(old_smi):]

# TestEqual and TestEqualStrict (== comparison, same pattern)
for (op, eq_fn) in [("TestEqual", "abstract_eq"), ("TestEqualStrict", "strict_eq")]:
    old_header = f"                    Opcode::{op} => {{\n"
    idx = content.find(old_header)
    if idx == -1:
        print(f"ERROR: could not find {op}"); sys.exit(1)
    rhs_line = "                            let rhs = unsafe { frame.read_reg_unchecked(v) };\n"
    rhs_idx = content.find(rhs_line, idx)
    match_line = "                            match (&acc, rhs) {\n"
    match_idx = content.find(match_line, rhs_idx)

    nb_fast = (
        f"                            // NaN-box fast path: both Smi.\n"
        f"                            if acc_is_nb\n"
        f"                                && NanBoxedValue::from_bits(acc_nb).is_smi()\n"
        f"                                && let JsValue::Smi(b) = rhs\n"
        f"                            {{\n"
        f"                                let result = NanBoxedValue::from_bits(acc_nb).as_smi() == *b;\n"
        f"                                acc = JsValue::Boolean(result);\n"
        f"                                acc_nb = NanBoxedValue::from_boolean(result).to_bits();\n"
        f"                                acc_is_nb = true;\n"
        f"                                continue 'dispatch;\n"
        f"                            }}\n"
    )
    content = content[:match_idx] + nb_fast + content[match_idx:]

    # Update Smi arm
    old_smi = (
        f"                                (JsValue::Smi(a), JsValue::Smi(b)) => {{\n"
        f"                                    acc = JsValue::Boolean(*a == *b);\n"
        f"                                    continue 'dispatch;\n"
        f"                                }}"
    )
    new_smi = (
        f"                                (JsValue::Smi(a), JsValue::Smi(b)) => {{\n"
        f"                                    let result = *a == *b;\n"
        f"                                    acc = JsValue::Boolean(result);\n"
        f"                                    acc_nb = NanBoxedValue::from_boolean(result).to_bits();\n"
        f"                                    acc_is_nb = true;\n"
        f"                                    continue 'dispatch;\n"
        f"                                }}"
    )
    smi_idx = content.find(old_smi, idx)
    if smi_idx == -1:
        print(f"ERROR: could not find Smi arm for {op}"); sys.exit(1)
    content = content[:smi_idx] + new_smi + content[smi_idx + len(old_smi):]

    # Track the eq_fn fallback
    old_fb = f"                            let result = {eq_fn}(&acc, rhs);\n"
    fb_idx = content.find(old_fb, idx)
    if fb_idx != -1:
        if op == "TestNotEqual":
            old_block = f"                            let result = {eq_fn}(&acc, rhs);\n                            acc = JsValue::Boolean(!result);\n                            continue 'dispatch;"
            new_block = f"                            let result = {eq_fn}(&acc, rhs);\n                            acc = JsValue::Boolean(!result);\n                            acc_nb = NanBoxedValue::from_boolean(!result).to_bits();\n                            acc_is_nb = true;\n                            continue 'dispatch;"
        else:
            old_block = f"                            let result = {eq_fn}(&acc, rhs);\n                            acc = JsValue::Boolean(result);\n                            continue 'dispatch;"
            new_block = f"                            let result = {eq_fn}(&acc, rhs);\n                            acc = JsValue::Boolean(result);\n                            acc_nb = NanBoxedValue::from_boolean(result).to_bits();\n                            acc_is_nb = true;\n                            continue 'dispatch;"
        actual_idx = content.find(old_block, idx)
        if actual_idx != -1:
            content = content[:actual_idx] + new_block + content[actual_idx + len(old_block):]

# TestNotEqual fallback
op_idx = content.find("Opcode::TestNotEqual =>")
if op_idx != -1:
    old_ne_fb = "                            let result = abstract_eq(&acc, rhs);\n                            acc = JsValue::Boolean(!result);\n                            continue 'dispatch;"
    ne_fb_idx = content.find(old_ne_fb, op_idx)
    if ne_fb_idx != -1:
        new_ne_fb = "                            let result = abstract_eq(&acc, rhs);\n                            acc = JsValue::Boolean(!result);\n                            acc_nb = NanBoxedValue::from_boolean(!result).to_bits();\n                            acc_is_nb = true;\n                            continue 'dispatch;"
        content = content[:ne_fb_idx] + new_ne_fb + content[ne_fb_idx + len(old_ne_fb):]

# ============================================================
# 12. JumpIfTrue / JumpIfFalse
# ============================================================
content = replace_one(content,
    "                    Opcode::JumpIfTrue => {\n"
    "                        if matches!(acc, JsValue::Boolean(true)) {\n"
    "                            // SAFETY: Bytecode compiler guarantees all\n"
    "                            // JumpIfTrue instructions have pre-computed targets.\n"
    "                            pc = unsafe { resolve_jump_unchecked(pc, jump_targets) };\n"
    "                        }\n"
    "                        continue 'dispatch;\n"
    "                    }",

    "                    Opcode::JumpIfTrue => {\n"
    "                        // Fast path: u64 compare avoids matching 24-byte enum.\n"
    "                        let is_true = if acc_is_nb {\n"
    "                            acc_nb == NanBoxedValue::from_boolean(true).to_bits()\n"
    "                        } else {\n"
    "                            matches!(acc, JsValue::Boolean(true))\n"
    "                        };\n"
    "                        if is_true {\n"
    "                            // SAFETY: Bytecode compiler guarantees all\n"
    "                            // JumpIfTrue instructions have pre-computed targets.\n"
    "                            pc = unsafe { resolve_jump_unchecked(pc, jump_targets) };\n"
    "                        }\n"
    "                        continue 'dispatch;\n"
    "                    }",
    "JumpIfTrue")

content = replace_one(content,
    "                    Opcode::JumpIfFalse => {\n"
    "                        if matches!(acc, JsValue::Boolean(false)) {\n"
    "                            // SAFETY: Bytecode compiler guarantees all\n"
    "                            // JumpIfFalse instructions have pre-computed targets.\n"
    "                            pc = unsafe { resolve_jump_unchecked(pc, jump_targets) };\n"
    "                        }\n"
    "                        continue 'dispatch;\n"
    "                    }",

    "                    Opcode::JumpIfFalse => {\n"
    "                        let is_false = if acc_is_nb {\n"
    "                            acc_nb == NanBoxedValue::from_boolean(false).to_bits()\n"
    "                        } else {\n"
    "                            matches!(acc, JsValue::Boolean(false))\n"
    "                        };\n"
    "                        if is_false {\n"
    "                            // SAFETY: Bytecode compiler guarantees all\n"
    "                            // JumpIfFalse instructions have pre-computed targets.\n"
    "                            pc = unsafe { resolve_jump_unchecked(pc, jump_targets) };\n"
    "                        }\n"
    "                        continue 'dispatch;\n"
    "                    }",
    "JumpIfFalse")

# ============================================================
# 13. JumpIfToBooleanTrue/False
# ============================================================
for (op, jump_cond) in [("JumpIfToBooleanTrue", "truthy"), ("JumpIfToBooleanFalse", "!truthy")]:
    old = (
        f"                    Opcode::{op} => {{\n"
        f"                        let truthy = match &acc {{\n"
        f"                            JsValue::Boolean(b) => *b,\n"
        f"                            JsValue::Smi(n) => *n != 0,\n"
        f"                            JsValue::Undefined | JsValue::Null | JsValue::TheHole => false,\n"
        f"                            JsValue::HeapNumber(n) => !n.is_nan() && *n != 0.0,\n"
        f"                            JsValue::String(s) => !s.is_empty(),\n"
        f"                            JsValue::BigInt(n) => **n != 0,\n"
    )
    idx = content.find(old)
    if idx == -1:
        print(f"ERROR: could not find {op}"); sys.exit(1)
    after = content[idx + len(old):]
    if after.lstrip().startswith("// Objects"):
        trailer = "                            // Objects, functions, arrays, etc. are always truthy.\n                            _ => true,\n"
    else:
        trailer = "                            _ => true,\n"

    old_full = old + trailer + (
        f"                        }};\n"
        f"                        if {jump_cond} {{\n"
        f"                            pc = unsafe {{ resolve_jump_unchecked(pc, jump_targets) }};\n"
        f"                        }}\n"
        f"                        continue 'dispatch;\n"
        f"                    }}"
    )

    new_full = (
        f"                    Opcode::{op} => {{\n"
        f"                        let truthy = if acc_is_nb {{\n"
        f"                            let nb = NanBoxedValue::from_bits(acc_nb);\n"
        f"                            if nb.is_boolean() {{\n"
        f"                                nb.as_boolean()\n"
        f"                            }} else if nb.is_smi() {{\n"
        f"                                nb.as_smi() != 0\n"
        f"                            }} else {{\n"
        f"                                // undefined, null, the_hole are all falsy\n"
        f"                                false\n"
        f"                            }}\n"
        f"                        }} else {{\n"
        f"                            match &acc {{\n"
        f"                                JsValue::Boolean(b) => *b,\n"
        f"                                JsValue::Smi(n) => *n != 0,\n"
        f"                                JsValue::Undefined | JsValue::Null | JsValue::TheHole => false,\n"
        f"                                JsValue::HeapNumber(n) => !n.is_nan() && *n != 0.0,\n"
        f"                                JsValue::String(s) => !s.is_empty(),\n"
        f"                                JsValue::BigInt(n) => **n != 0,\n"
        f"                                _ => true,\n"
        f"                            }}\n"
        f"                        }};\n"
        f"                        if {jump_cond} {{\n"
        f"                            pc = unsafe {{ resolve_jump_unchecked(pc, jump_targets) }};\n"
        f"                        }}\n"
        f"                        continue 'dispatch;\n"
        f"                    }}"
    )
    content = replace_one(content, old_full, new_full, op)

# ============================================================
# 14. LdaGlobal IC hit — clear acc_is_nb
# ============================================================
content = replace_one(content,
    "                                    if value != JsValue::TheHole {\n"
    "                                        acc = value;\n"
    "                                        continue 'dispatch;",
    "                                    if value != JsValue::TheHole {\n"
    "                                        acc = value;\n"
    "                                        acc_is_nb = false;\n"
    "                                        continue 'dispatch;",
    "LdaGlobal IC hit")

# ============================================================
# 15. Clear acc_is_nb for untracked acc-modifying handlers
#     Strategy: insert right before each "continue 'dispatch" that's
#     inside a fast-path of these handlers. Also clear in the
#     dispatch_via_table Continue arm.
#     To avoid dead-write warnings, we ONLY set acc_is_nb=false
#     right before "continue 'dispatch" (which guarantees it IS read
#     on the next iteration).
# ============================================================
# For the remaining inline handlers that modify acc, we need to clear
# acc_is_nb on their fast paths. Instead of modifying each individual
# handler (too many), we'll use the catch-all approach: clear acc_is_nb
# in the catch-all `_ =>` dispatch table handler.
# The only handlers that need explicit clearing are those with inline
# fast paths that `continue 'dispatch` without going through the table.
# These are: BitwiseOr, BitwiseOrSmi, Div, BitwiseAnd, Mod, etc.
# But since the user's optimization focuses on the hottest opcodes
# (LdaSmi, Add, comparisons, jumps), and the bitwise/div/mod ops are
# much rarer, the stale acc_is_nb from a previous instruction won't
# cause correctness issues — it will only cause the NaN-box fast path
# to be tried and then fall through (since the acc_nb value won't match).
#
# WAIT: Actually there IS a correctness issue. If LdaSmi sets acc_is_nb=true
# and acc_nb to some Smi, then BitwiseOr changes acc to a different Smi
# without updating acc_nb, then TestLessThan checks acc_nb and gets the
# WRONG value. That's a bug.
#
# So we MUST clear acc_is_nb in these handlers. But we need to avoid
# the dead-write warning. The solution: set acc_is_nb=false ONLY on
# paths that do `continue 'dispatch` (not before dispatch_via_table).

# For the "continue 'dispatch" in fast-path arms of untracked handlers,
# we insert acc_is_nb=false right before. We find these by looking for
# specific patterns in each handler.

# BitwiseOr fast path: acc = JsValue::Smi(*a | *b);
for pat in [
    "                                acc = JsValue::Smi(*a | *b);",
    "                                    acc = JsValue::Smi(n | imm);",  # BitwiseOrSmi
    "                                acc = JsValue::Smi(*a & *b);",  # BitwiseAnd
    "                                    acc = JsValue::Smi(n & imm);",  # BitwiseAndSmi
]:
    idx = content.find(pat)
    if idx != -1:
        # Find the next "continue 'dispatch;" after this line
        cont = content.find("continue 'dispatch;", idx)
        if cont != -1 and cont - idx < 300:
            # Insert acc_is_nb = false before the continue
            nl = content.rfind("\n", idx, cont) + 1
            indent = " " * (cont - nl)
            insert_str = indent + "acc_is_nb = false;\n"
            content = content[:cont] + "acc_is_nb = false;\n" + " " * 32 + "continue 'dispatch;" + content[cont + len("continue 'dispatch;"):]

# For Div, Mod, ShiftRightSmi, ShiftLeftSmi — these all have Smi fast paths
# that produce acc values. Find and patch them.
for pat_label, acc_pat in [
    ("Div", "                                    acc = JsValue::Smi(result);"),
    ("Mod Smi", "                                    acc = JsValue::Smi(a_val % b_val);"),
]:
    idx = content.find(acc_pat)
    if idx != -1:
        cont = content.find("continue 'dispatch;", idx)
        if cont != -1 and cont - idx < 200:
            content = content[:cont] + "acc_is_nb = false;\n                                    continue 'dispatch;" + content[cont + len("continue 'dispatch;"):]

# For shift ops: find the specific patterns
for pat in [
    "                                    acc = JsValue::Smi(n >> shift);",
    "                                    acc = JsValue::Smi(n << shift);",
]:
    idx = content.find(pat)
    if idx != -1:
        cont = content.find("continue 'dispatch;", idx)
        if cont != -1 and cont - idx < 200:
            content = content[:cont] + "acc_is_nb = false;\n                                    continue 'dispatch;" + content[cont + len("continue 'dispatch;"):]

# For HeapNumber coercion paths in BitwiseOr etc:
for pat in [
    "                                    acc = JsValue::Smi(a_i32 | b_i32);",
    "                                    acc = JsValue::Smi(a_i32 & b_i32);",
    "                                    acc = number_to_jsvalue(a_f | b_f);",
]:
    idx = content.find(pat)
    while idx != -1:
        cont = content.find("continue 'dispatch;", idx)
        if cont != -1 and cont - idx < 200:
            # Check if acc_is_nb = false is already there
            between = content[idx:cont]
            if "acc_is_nb" not in between:
                content = content[:cont] + "acc_is_nb = false;\n                                    continue 'dispatch;" + content[cont + len("continue 'dispatch;"):]
        idx = content.find(pat, idx + 1)

# ============================================================
# 16. Catch-all — clear acc_is_nb
# ============================================================
content = replace_one(content,
    "                    // ── All other opcodes: dispatch table ────\n"
    "                    _ => {\n"
    "                        // Write back locals before cold-path dispatch.\n"
    "                        frame.pc = pc;\n"
    "                        frame.accumulator = acc;",

    "                    // ── All other opcodes: dispatch table ────\n"
    "                    _ => {\n"
    "                        acc_is_nb = false;\n"
    "                        // Write back locals before cold-path dispatch.\n"
    "                        frame.pc = pc;\n"
    "                        frame.accumulator = acc;",
    "catch-all")

# ============================================================
# 17. For all dispatch_via_table Continue arms, clear acc_is_nb
#     after reloading acc. This handles untracked handlers that
#     fall through to the table.
# ============================================================
# Pattern: acc = std::mem::replace(&mut frame.accumulator, JsValue::Undefined);
#          continue 'dispatch;
# We add acc_is_nb = false between them.
old_continue_pat = "                                acc = std::mem::replace(&mut frame.accumulator, JsValue::Undefined);\n                                continue 'dispatch;"
new_continue_pat = "                                acc = std::mem::replace(&mut frame.accumulator, JsValue::Undefined);\n                                acc_is_nb = false;\n                                continue 'dispatch;"
# Replace ALL occurrences (there are many dispatch_via_table fallbacks)
count = content.count(old_continue_pat)
content = content.replace(old_continue_pat, new_continue_pat)
print(f"Patched {count} dispatch_via_table Continue arms")

# Also handle the error recovery paths
old_err_continue = "                                    acc = std::mem::replace(\n                                        &mut frame.accumulator,\n                                        JsValue::Undefined,\n                                    );\n                                    continue 'dispatch;"
new_err_continue = "                                    acc = std::mem::replace(\n                                        &mut frame.accumulator,\n                                        JsValue::Undefined,\n                                    );\n                                    acc_is_nb = false;\n                                    continue 'dispatch;"
count2 = content.count(old_err_continue)
content = content.replace(old_err_continue, new_err_continue)
print(f"Patched {count2} dispatch_via_table error-recovery arms")

# Also handle the acc reload in JumpLoop's SMI mode entry
old_jl = "                                acc = std::mem::replace(&mut frame.accumulator, JsValue::Undefined);"
# This was already handled by the global replace above

# ============================================================
# Write
# ============================================================
with open(FILE, 'w', encoding='utf-8') as f:
    f.write(content)

print("SUCCESS: All patches applied")
print(f"File written: {len(content)} chars, {content.count(chr(10))} lines")
