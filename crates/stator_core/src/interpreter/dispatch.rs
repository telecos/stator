//! Computed-goto–style dispatch table for the interpreter.
//
//! Each [`Opcode`] is mapped to a handler function via
//! [`DISPATCH_TABLE`].  The main interpreter loop indexes
//! the table by opcode discriminant and calls the handler,
//! replacing the former exhaustive `match`.

#![allow(clippy::too_many_lines)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::objects::map::PropertyAttributes;
use crate::objects::property_map::PropertyMap;

use super::{
    ACTIVE_DEBUGGER, Interpreter, InterpreterFrame, MAGLEV_OSR_LOOP_THRESHOLD, OSR_LOOP_THRESHOLD,
    PropertyIc, TURBOFAN_OSR_LOOP_THRESHOLD, abstract_eq, bigint_pow, collect_args, concat_rc_strs,
    constant_pool_jump_delta, constant_to_value, decode_string_constant, dispatch_call_property,
    dispatch_call_with_this, dispatch_setter, err_bad_operand, error_message_from_value,
    extract_context, find_handler, fn_props_set, is_js_receiver, js_add, js_less_than, keyed_load,
    keyed_store, maybe_compile_baseline, maybe_compile_maglev, maybe_compile_turbofan,
    number_to_jsvalue, plain_object_to_array_items, proto_lookup, resolve_jump,
    restore_closure_context, set_pending_exception, strict_eq, to_array_index, to_bigint,
    to_property_key, try_execute_best_jit, walk_context_chain, wire_construct_prototype,
};
use crate::builtins::error::{ErrorKind, pop_call_frame, push_call_frame};
use crate::builtins::proxy::{proxy_delete_property, proxy_has, proxy_set};
use crate::bytecode::bytecode_array::{
    ConstantPoolEntry, HandlerTableEntry, MAGLEV_TIERING_THRESHOLD, TIERING_THRESHOLD,
    TURBOFAN_TIERING_THRESHOLD,
};
use crate::bytecode::bytecodes::{Instruction, Opcode, Operand};
use crate::error::{StatorError, StatorResult};
use crate::objects::value::{
    GeneratorResumeMode, GeneratorState, GeneratorStatus, GeneratorStep, JsContext, JsValue,
    NativeIterator,
};

/// Result of executing a single opcode handler.
pub(super) enum DispatchAction {
    /// Continue to the next instruction.
    Continue,
    /// Return from the interpreter with the given value.
    Return(JsValue),
    /// Restart the tail-call trampoline loop.
    TailCall,
}

/// Mutable execution context passed to every opcode handler.
pub(super) struct DispatchContext<'a> {
    /// The current interpreter frame.
    pub frame: &'a mut InterpreterFrame,
    /// Pre-decoded instruction stream.
    pub instructions: &'a [Instruction],
    /// Byte-offset table parallel to `instructions`.
    pub byte_offsets: &'a [usize],
    /// Exception handler table for the current function.
    #[allow(dead_code)]
    pub handler_table: &'a [HandlerTableEntry],
}

/// Signature of a single opcode handler function.
pub(super) type OpcodeHandler =
    fn(&mut DispatchContext, &Instruction) -> StatorResult<DispatchAction>;

/// Number of opcode variants (= `Opcode::Illegal as usize + 1`).
const OPCODE_COUNT: usize = 192;

#[inline]
fn handle_lda_zero(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    ctx.frame.accumulator = JsValue::Smi(0);
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_lda_smi(ctx: &mut DispatchContext, instr: &Instruction) -> StatorResult<DispatchAction> {
    let Operand::Immediate(v) = instr.operands[0] else {
        return Err(err_bad_operand("LdaSmi", 0));
    };
    ctx.frame.accumulator = JsValue::Smi(v);
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_lda_undefined(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    ctx.frame.accumulator = JsValue::Undefined;
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_lda_the_hole(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    ctx.frame.accumulator = JsValue::TheHole;
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_lda_null(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    ctx.frame.accumulator = JsValue::Null;
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_lda_true(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    ctx.frame.accumulator = JsValue::Boolean(true);
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_lda_false(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    ctx.frame.accumulator = JsValue::Boolean(false);
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_lda_constant(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(idx) = instr.operands[0] else {
        return Err(err_bad_operand("LdaConstant", 0));
    };
    let entry =
        ctx.frame.bytecode_array.get_constant(idx).ok_or_else(|| {
            StatorError::Internal(format!("constant pool index {idx} out of bounds"))
        })?;
    ctx.frame.accumulator = constant_to_value(entry);
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_ldar(ctx: &mut DispatchContext, instr: &Instruction) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("Ldar", 0));
    };
    ctx.frame.accumulator = ctx.frame.read_reg(v)?.cheap_clone();
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_star(ctx: &mut DispatchContext, instr: &Instruction) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("Star", 0));
    };
    let val = ctx.frame.accumulator.cheap_clone();
    ctx.frame.write_reg(v, val)?;
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_mov(ctx: &mut DispatchContext, instr: &Instruction) -> StatorResult<DispatchAction> {
    let Operand::Register(src) = instr.operands[0] else {
        return Err(err_bad_operand("Mov", 0));
    };
    let Operand::Register(dst) = instr.operands[1] else {
        return Err(err_bad_operand("Mov", 1));
    };
    ctx.frame.copy_reg(src, dst)?;
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_add(ctx: &mut DispatchContext, instr: &Instruction) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("Add", 0));
    };
    let rhs = ctx.frame.read_reg(v)?;
    // Fast path: Smi + Smi → Smi (no allocation, overflow → HeapNumber)
    if let JsValue::Smi(a) = ctx.frame.accumulator
        && let JsValue::Smi(b) = rhs
    {
        ctx.frame.accumulator = match a.checked_add(*b) {
            Some(r) => JsValue::Smi(r),
            None => JsValue::HeapNumber(a as f64 + *b as f64),
        };
        return Ok(DispatchAction::Continue);
    }
    // Fast path: String + String – skip to_js_string conversion.
    if let JsValue::String(ref a) = ctx.frame.accumulator
        && let JsValue::String(b) = rhs
    {
        let total = a.len().saturating_add(b.len());
        if total > crate::builtins::string::MAX_STRING_LEN {
            return Err(crate::error::StatorError::RangeError(
                "Invalid string length".into(),
            ));
        }
        ctx.frame.accumulator = concat_rc_strs(a, b);
        return Ok(DispatchAction::Continue);
    }
    let rhs = rhs.cheap_clone();
    ctx.frame.accumulator = js_add(&ctx.frame.accumulator, &rhs)?;
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_sub(ctx: &mut DispatchContext, instr: &Instruction) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("Sub", 0));
    };
    let rhs = ctx.frame.read_reg(v)?;
    // Fast path: Smi - Smi → Smi
    if let JsValue::Smi(a) = ctx.frame.accumulator
        && let JsValue::Smi(b) = rhs
    {
        ctx.frame.accumulator = match a.checked_sub(*b) {
            Some(r) => JsValue::Smi(r),
            None => JsValue::HeapNumber(a as f64 - *b as f64),
        };
        return Ok(DispatchAction::Continue);
    }
    let rhs = rhs.cheap_clone();
    if ctx.frame.accumulator.is_bigint() || rhs.is_bigint() {
        let l = to_bigint(&ctx.frame.accumulator)?;
        let r = to_bigint(&rhs)?;
        ctx.frame.accumulator = JsValue::BigInt(l.wrapping_sub(r));
    } else {
        let lhs_n = ctx.frame.accumulator.to_number()?;
        let rhs_n = rhs.to_number()?;
        ctx.frame.accumulator = number_to_jsvalue(lhs_n - rhs_n);
    }
    Ok(DispatchAction::Continue)
}

fn handle_mul(ctx: &mut DispatchContext, instr: &Instruction) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("Mul", 0));
    };
    let rhs = ctx.frame.read_reg(v)?;
    // Fast path: Smi * Smi → Smi
    if let JsValue::Smi(a) = ctx.frame.accumulator
        && let JsValue::Smi(b) = rhs
    {
        ctx.frame.accumulator = match a.checked_mul(*b) {
            Some(r) => JsValue::Smi(r),
            None => JsValue::HeapNumber(a as f64 * *b as f64),
        };
        return Ok(DispatchAction::Continue);
    }
    let rhs = rhs.cheap_clone();
    if ctx.frame.accumulator.is_bigint() || rhs.is_bigint() {
        let l = to_bigint(&ctx.frame.accumulator)?;
        let r = to_bigint(&rhs)?;
        ctx.frame.accumulator = JsValue::BigInt(l.wrapping_mul(r));
    } else {
        let lhs_n = ctx.frame.accumulator.to_number()?;
        let rhs_n = rhs.to_number()?;
        ctx.frame.accumulator = number_to_jsvalue(lhs_n * rhs_n);
    }
    Ok(DispatchAction::Continue)
}

fn handle_div(ctx: &mut DispatchContext, instr: &Instruction) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("Div", 0));
    };
    let rhs = ctx.frame.read_reg(v)?;
    // Fast path: Smi / Smi → HeapNumber (JS division always yields float)
    if let JsValue::Smi(a) = ctx.frame.accumulator
        && let JsValue::Smi(b) = rhs
    {
        ctx.frame.accumulator = number_to_jsvalue(a as f64 / *b as f64);
        return Ok(DispatchAction::Continue);
    }
    let rhs = rhs.cheap_clone();
    if ctx.frame.accumulator.is_bigint() || rhs.is_bigint() {
        let l = to_bigint(&ctx.frame.accumulator)?;
        let r = to_bigint(&rhs)?;
        if r == 0 {
            return Err(StatorError::RangeError("Division by zero".to_string()));
        }
        ctx.frame.accumulator = JsValue::BigInt(l / r);
    } else {
        let lhs_n = ctx.frame.accumulator.to_number()?;
        let rhs_n = rhs.to_number()?;
        ctx.frame.accumulator = number_to_jsvalue(lhs_n / rhs_n);
    }
    Ok(DispatchAction::Continue)
}

fn handle_mod(ctx: &mut DispatchContext, instr: &Instruction) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("Mod", 0));
    };
    let rhs = ctx.frame.read_reg(v)?;
    // Fast path: Smi % Smi
    if let JsValue::Smi(a) = ctx.frame.accumulator
        && let JsValue::Smi(b) = rhs
    {
        ctx.frame.accumulator = number_to_jsvalue(a as f64 % *b as f64);
        return Ok(DispatchAction::Continue);
    }
    let rhs = rhs.cheap_clone();
    if ctx.frame.accumulator.is_bigint() || rhs.is_bigint() {
        let l = to_bigint(&ctx.frame.accumulator)?;
        let r = to_bigint(&rhs)?;
        if r == 0 {
            return Err(StatorError::RangeError("Division by zero".to_string()));
        }
        ctx.frame.accumulator = JsValue::BigInt(l % r);
    } else {
        let lhs_n = ctx.frame.accumulator.to_number()?;
        let rhs_n = rhs.to_number()?;
        ctx.frame.accumulator = number_to_jsvalue(lhs_n % rhs_n);
    }
    Ok(DispatchAction::Continue)
}

fn handle_exp(ctx: &mut DispatchContext, instr: &Instruction) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("Exp", 0));
    };
    let rhs = ctx.frame.read_reg(v)?;
    // Fast path: Smi ** Smi
    if let JsValue::Smi(a) = ctx.frame.accumulator
        && let JsValue::Smi(b) = rhs
    {
        ctx.frame.accumulator = number_to_jsvalue((a as f64).powf(*b as f64));
        return Ok(DispatchAction::Continue);
    }
    let rhs = rhs.cheap_clone();
    if ctx.frame.accumulator.is_bigint() || rhs.is_bigint() {
        let l = to_bigint(&ctx.frame.accumulator)?;
        let r = to_bigint(&rhs)?;
        if r < 0 {
            return Err(StatorError::RangeError(
                "Exponent must be positive".to_string(),
            ));
        }
        ctx.frame.accumulator = JsValue::BigInt(bigint_pow(l, r as u32));
    } else {
        let lhs_n = ctx.frame.accumulator.to_number()?;
        let rhs_n = rhs.to_number()?;
        ctx.frame.accumulator = number_to_jsvalue(lhs_n.powf(rhs_n));
    }
    Ok(DispatchAction::Continue)
}

fn handle_bitwise_or(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("BitwiseOr", 0));
    };
    let rhs = ctx.frame.read_reg(v)?;
    // Fast path: Smi | Smi (no clone needed)
    if let JsValue::Smi(a) = ctx.frame.accumulator
        && let JsValue::Smi(b) = rhs
    {
        ctx.frame.accumulator = JsValue::Smi(a | *b);
        return Ok(DispatchAction::Continue);
    }
    let rhs = rhs.cheap_clone();
    if ctx.frame.accumulator.is_bigint() || rhs.is_bigint() {
        let l = to_bigint(&ctx.frame.accumulator)?;
        let r = to_bigint(&rhs)?;
        ctx.frame.accumulator = JsValue::BigInt(l | r);
    } else {
        let lhs = ctx.frame.accumulator.to_number()? as i32;
        let rhs_i = rhs.to_number()? as i32;
        ctx.frame.accumulator = JsValue::Smi(lhs | rhs_i);
    }
    Ok(DispatchAction::Continue)
}

fn handle_bitwise_xor(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("BitwiseXor", 0));
    };
    let rhs = ctx.frame.read_reg(v)?;
    // Fast path: Smi ^ Smi (no clone needed)
    if let JsValue::Smi(a) = ctx.frame.accumulator
        && let JsValue::Smi(b) = rhs
    {
        ctx.frame.accumulator = JsValue::Smi(a ^ *b);
        return Ok(DispatchAction::Continue);
    }
    let rhs = rhs.cheap_clone();
    if ctx.frame.accumulator.is_bigint() || rhs.is_bigint() {
        let l = to_bigint(&ctx.frame.accumulator)?;
        let r = to_bigint(&rhs)?;
        ctx.frame.accumulator = JsValue::BigInt(l ^ r);
    } else {
        let lhs = ctx.frame.accumulator.to_number()? as i32;
        let rhs_i = rhs.to_number()? as i32;
        ctx.frame.accumulator = JsValue::Smi(lhs ^ rhs_i);
    }
    Ok(DispatchAction::Continue)
}

fn handle_bitwise_and(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("BitwiseAnd", 0));
    };
    let rhs = ctx.frame.read_reg(v)?;
    // Fast path: Smi & Smi (no clone needed)
    if let JsValue::Smi(a) = ctx.frame.accumulator
        && let JsValue::Smi(b) = rhs
    {
        ctx.frame.accumulator = JsValue::Smi(a & *b);
        return Ok(DispatchAction::Continue);
    }
    let rhs = rhs.cheap_clone();
    if ctx.frame.accumulator.is_bigint() || rhs.is_bigint() {
        let l = to_bigint(&ctx.frame.accumulator)?;
        let r = to_bigint(&rhs)?;
        ctx.frame.accumulator = JsValue::BigInt(l & r);
    } else {
        let lhs = ctx.frame.accumulator.to_number()? as i32;
        let rhs_i = rhs.to_number()? as i32;
        ctx.frame.accumulator = JsValue::Smi(lhs & rhs_i);
    }
    Ok(DispatchAction::Continue)
}

fn handle_shift_left(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("ShiftLeft", 0));
    };
    let rhs = ctx.frame.read_reg(v)?;
    // Fast path: Smi << Smi (no clone needed)
    if let JsValue::Smi(a) = ctx.frame.accumulator
        && let JsValue::Smi(b) = rhs
    {
        let shift = (*b as u32) & 0x1f;
        ctx.frame.accumulator = JsValue::Smi(a << shift);
        return Ok(DispatchAction::Continue);
    }
    let rhs = rhs.cheap_clone();
    if ctx.frame.accumulator.is_bigint() || rhs.is_bigint() {
        let l = to_bigint(&ctx.frame.accumulator)?;
        let r = to_bigint(&rhs)?;
        ctx.frame.accumulator = JsValue::BigInt(l.wrapping_shl(r as u32));
    } else {
        let lhs = ctx.frame.accumulator.to_number()? as i32;
        let shift = (rhs.to_number()? as u32) & 0x1f;
        ctx.frame.accumulator = JsValue::Smi(lhs << shift);
    }
    Ok(DispatchAction::Continue)
}

fn handle_shift_right(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("ShiftRight", 0));
    };
    let rhs = ctx.frame.read_reg(v)?;
    // Fast path: Smi >> Smi (no clone needed)
    if let JsValue::Smi(a) = ctx.frame.accumulator
        && let JsValue::Smi(b) = rhs
    {
        let shift = (*b as u32) & 0x1f;
        ctx.frame.accumulator = JsValue::Smi(a >> shift);
        return Ok(DispatchAction::Continue);
    }
    let rhs = rhs.cheap_clone();
    if ctx.frame.accumulator.is_bigint() || rhs.is_bigint() {
        let l = to_bigint(&ctx.frame.accumulator)?;
        let r = to_bigint(&rhs)?;
        ctx.frame.accumulator = JsValue::BigInt(l.wrapping_shr(r as u32));
    } else {
        let lhs = ctx.frame.accumulator.to_number()? as i32;
        let shift = (rhs.to_number()? as u32) & 0x1f;
        ctx.frame.accumulator = JsValue::Smi(lhs >> shift);
    }
    Ok(DispatchAction::Continue)
}

fn handle_shift_right_logical(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("ShiftRightLogical", 0));
    };
    let rhs = ctx.frame.read_reg(v)?;
    // Fast path: Smi >>> Smi (no clone needed)
    if let JsValue::Smi(a) = ctx.frame.accumulator
        && let JsValue::Smi(b) = rhs
    {
        let lhs = a as u32;
        let shift = (*b as u32) & 0x1f;
        let result = lhs >> shift;
        ctx.frame.accumulator = number_to_jsvalue(result as f64);
        return Ok(DispatchAction::Continue);
    }
    let rhs = rhs.cheap_clone();
    let lhs = ctx.frame.accumulator.to_number()? as i32 as u32;
    let shift = (rhs.to_number()? as u32) & 0x1f;
    let result = lhs >> shift;
    ctx.frame.accumulator = number_to_jsvalue(result as f64);
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_add_smi(ctx: &mut DispatchContext, instr: &Instruction) -> StatorResult<DispatchAction> {
    let Operand::Immediate(imm) = instr.operands[0] else {
        return Err(err_bad_operand("AddSmi", 0));
    };
    // Fast path: Smi + immediate
    if let JsValue::Smi(n) = ctx.frame.accumulator {
        ctx.frame.accumulator = match n.checked_add(imm) {
            Some(r) => JsValue::Smi(r),
            None => JsValue::HeapNumber(n as f64 + imm as f64),
        };
        return Ok(DispatchAction::Continue);
    }
    if let JsValue::BigInt(n) = &ctx.frame.accumulator {
        ctx.frame.accumulator = JsValue::BigInt(n.wrapping_add(i128::from(imm)));
    } else {
        let lhs_n = ctx.frame.accumulator.to_number()?;
        ctx.frame.accumulator = number_to_jsvalue(lhs_n + imm as f64);
    }
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_sub_smi(ctx: &mut DispatchContext, instr: &Instruction) -> StatorResult<DispatchAction> {
    let Operand::Immediate(imm) = instr.operands[0] else {
        return Err(err_bad_operand("SubSmi", 0));
    };
    // Fast path: Smi - immediate
    if let JsValue::Smi(n) = ctx.frame.accumulator {
        ctx.frame.accumulator = match n.checked_sub(imm) {
            Some(r) => JsValue::Smi(r),
            None => JsValue::HeapNumber(n as f64 - imm as f64),
        };
        return Ok(DispatchAction::Continue);
    }
    if let JsValue::BigInt(n) = &ctx.frame.accumulator {
        ctx.frame.accumulator = JsValue::BigInt(n.wrapping_sub(i128::from(imm)));
    } else {
        let lhs_n = ctx.frame.accumulator.to_number()?;
        ctx.frame.accumulator = number_to_jsvalue(lhs_n - imm as f64);
    }
    Ok(DispatchAction::Continue)
}

fn handle_mul_smi(ctx: &mut DispatchContext, instr: &Instruction) -> StatorResult<DispatchAction> {
    let Operand::Immediate(imm) = instr.operands[0] else {
        return Err(err_bad_operand("MulSmi", 0));
    };
    // Fast path: Smi * immediate
    if let JsValue::Smi(n) = ctx.frame.accumulator {
        ctx.frame.accumulator = match n.checked_mul(imm) {
            Some(r) => JsValue::Smi(r),
            None => JsValue::HeapNumber(n as f64 * imm as f64),
        };
        return Ok(DispatchAction::Continue);
    }
    if let JsValue::BigInt(n) = &ctx.frame.accumulator {
        ctx.frame.accumulator = JsValue::BigInt(n.wrapping_mul(i128::from(imm)));
    } else {
        let lhs_n = ctx.frame.accumulator.to_number()?;
        ctx.frame.accumulator = number_to_jsvalue(lhs_n * imm as f64);
    }
    Ok(DispatchAction::Continue)
}

fn handle_div_smi(ctx: &mut DispatchContext, instr: &Instruction) -> StatorResult<DispatchAction> {
    let Operand::Immediate(imm) = instr.operands[0] else {
        return Err(err_bad_operand("DivSmi", 0));
    };
    if let JsValue::BigInt(n) = &ctx.frame.accumulator {
        if imm == 0 {
            return Err(StatorError::RangeError("Division by zero".to_string()));
        }
        ctx.frame.accumulator = JsValue::BigInt(n / i128::from(imm));
    } else {
        let lhs_n = ctx.frame.accumulator.to_number()?;
        ctx.frame.accumulator = number_to_jsvalue(lhs_n / imm as f64);
    }
    Ok(DispatchAction::Continue)
}

fn handle_mod_smi(ctx: &mut DispatchContext, instr: &Instruction) -> StatorResult<DispatchAction> {
    let Operand::Immediate(imm) = instr.operands[0] else {
        return Err(err_bad_operand("ModSmi", 0));
    };
    if let JsValue::BigInt(n) = &ctx.frame.accumulator {
        if imm == 0 {
            return Err(StatorError::RangeError("Division by zero".to_string()));
        }
        ctx.frame.accumulator = JsValue::BigInt(n % i128::from(imm));
    } else {
        let lhs_n = ctx.frame.accumulator.to_number()?;
        ctx.frame.accumulator = number_to_jsvalue(lhs_n % imm as f64);
    }
    Ok(DispatchAction::Continue)
}

fn handle_exp_smi(ctx: &mut DispatchContext, instr: &Instruction) -> StatorResult<DispatchAction> {
    let Operand::Immediate(imm) = instr.operands[0] else {
        return Err(err_bad_operand("ExpSmi", 0));
    };
    if let JsValue::BigInt(n) = &ctx.frame.accumulator {
        if imm < 0 {
            return Err(StatorError::RangeError(
                "Exponent must be positive".to_string(),
            ));
        }
        ctx.frame.accumulator = JsValue::BigInt(bigint_pow(*n, imm as u32));
    } else {
        let lhs_n = ctx.frame.accumulator.to_number()?;
        ctx.frame.accumulator = number_to_jsvalue(lhs_n.powf(imm as f64));
    }
    Ok(DispatchAction::Continue)
}

fn handle_bitwise_or_smi(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Immediate(imm) = instr.operands[0] else {
        return Err(err_bad_operand("BitwiseOrSmi", 0));
    };
    if let JsValue::BigInt(n) = &ctx.frame.accumulator {
        ctx.frame.accumulator = JsValue::BigInt(n | i128::from(imm));
    } else {
        let lhs = ctx.frame.accumulator.to_number()? as i32;
        ctx.frame.accumulator = JsValue::Smi(lhs | imm);
    }
    Ok(DispatchAction::Continue)
}

fn handle_bitwise_xor_smi(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Immediate(imm) = instr.operands[0] else {
        return Err(err_bad_operand("BitwiseXorSmi", 0));
    };
    if let JsValue::BigInt(n) = &ctx.frame.accumulator {
        ctx.frame.accumulator = JsValue::BigInt(n ^ i128::from(imm));
    } else {
        let lhs = ctx.frame.accumulator.to_number()? as i32;
        ctx.frame.accumulator = JsValue::Smi(lhs ^ imm);
    }
    Ok(DispatchAction::Continue)
}

fn handle_bitwise_and_smi(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Immediate(imm) = instr.operands[0] else {
        return Err(err_bad_operand("BitwiseAndSmi", 0));
    };
    if let JsValue::BigInt(n) = &ctx.frame.accumulator {
        ctx.frame.accumulator = JsValue::BigInt(n & i128::from(imm));
    } else {
        let lhs = ctx.frame.accumulator.to_number()? as i32;
        ctx.frame.accumulator = JsValue::Smi(lhs & imm);
    }
    Ok(DispatchAction::Continue)
}

fn handle_shift_left_smi(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Immediate(imm) = instr.operands[0] else {
        return Err(err_bad_operand("ShiftLeftSmi", 0));
    };
    if let JsValue::BigInt(n) = &ctx.frame.accumulator {
        ctx.frame.accumulator = JsValue::BigInt(n.wrapping_shl(imm as u32));
    } else {
        let lhs = ctx.frame.accumulator.to_number()? as i32;
        let shift = (imm as u32) & 0x1f;
        ctx.frame.accumulator = JsValue::Smi(lhs << shift);
    }
    Ok(DispatchAction::Continue)
}

fn handle_shift_right_smi(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Immediate(imm) = instr.operands[0] else {
        return Err(err_bad_operand("ShiftRightSmi", 0));
    };
    if let JsValue::BigInt(n) = &ctx.frame.accumulator {
        ctx.frame.accumulator = JsValue::BigInt(n.wrapping_shr(imm as u32));
    } else {
        let lhs = ctx.frame.accumulator.to_number()? as i32;
        let shift = (imm as u32) & 0x1f;
        ctx.frame.accumulator = JsValue::Smi(lhs >> shift);
    }
    Ok(DispatchAction::Continue)
}

fn handle_shift_right_logical_smi(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Immediate(imm) = instr.operands[0] else {
        return Err(err_bad_operand("ShiftRightLogicalSmi", 0));
    };
    let lhs = ctx.frame.accumulator.to_number()? as i32 as u32;
    let shift = (imm as u32) & 0x1f;
    let result = lhs >> shift;
    ctx.frame.accumulator = number_to_jsvalue(result as f64);
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_inc(ctx: &mut DispatchContext, _instr: &Instruction) -> StatorResult<DispatchAction> {
    // operands[0] is a FeedbackSlot, ignored at runtime.
    if let JsValue::BigInt(n) = &ctx.frame.accumulator {
        ctx.frame.accumulator = JsValue::BigInt(n.wrapping_add(1));
    } else {
        let n = ctx.frame.accumulator.to_number()?;
        ctx.frame.accumulator = number_to_jsvalue(n + 1.0);
    }
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_dec(ctx: &mut DispatchContext, _instr: &Instruction) -> StatorResult<DispatchAction> {
    // operands[0] is a FeedbackSlot, ignored at runtime.
    if let JsValue::BigInt(n) = &ctx.frame.accumulator {
        ctx.frame.accumulator = JsValue::BigInt(n.wrapping_sub(1));
    } else {
        let n = ctx.frame.accumulator.to_number()?;
        ctx.frame.accumulator = number_to_jsvalue(n - 1.0);
    }
    Ok(DispatchAction::Continue)
}

fn handle_test_equal(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("TestEqual", 0));
    };
    let rhs = ctx.frame.read_reg(v)?.cheap_clone();
    let result = abstract_eq(&ctx.frame.accumulator, &rhs);
    ctx.frame.accumulator = JsValue::Boolean(result);
    Ok(DispatchAction::Continue)
}

fn handle_test_not_equal(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("TestNotEqual", 0));
    };
    let rhs = ctx.frame.read_reg(v)?.cheap_clone();
    let result = !abstract_eq(&ctx.frame.accumulator, &rhs);
    ctx.frame.accumulator = JsValue::Boolean(result);
    Ok(DispatchAction::Continue)
}

fn handle_test_equal_strict(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("TestEqualStrict", 0));
    };
    let rhs = ctx.frame.read_reg(v)?;
    // Fast path: Smi === Smi
    if let JsValue::Smi(a) = ctx.frame.accumulator
        && let JsValue::Smi(b) = rhs
    {
        ctx.frame.accumulator = JsValue::Boolean(a == *b);
        return Ok(DispatchAction::Continue);
    }
    let rhs = rhs.clone();
    let result = strict_eq(&ctx.frame.accumulator, &rhs);
    ctx.frame.accumulator = JsValue::Boolean(result);
    Ok(DispatchAction::Continue)
}

fn handle_test_less_than(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("TestLessThan", 0));
    };
    let rhs = ctx.frame.read_reg(v)?;
    // Fast path: Smi < Smi
    if let JsValue::Smi(a) = ctx.frame.accumulator
        && let JsValue::Smi(b) = rhs
    {
        ctx.frame.accumulator = JsValue::Boolean(a < *b);
        return Ok(DispatchAction::Continue);
    }
    let rhs = rhs.clone();
    let result = js_less_than(&ctx.frame.accumulator, &rhs)?;
    ctx.frame.accumulator = JsValue::Boolean(result);
    Ok(DispatchAction::Continue)
}

fn handle_test_greater_than(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("TestGreaterThan", 0));
    };
    let rhs = ctx.frame.read_reg(v)?;
    // Fast path: Smi > Smi
    if let JsValue::Smi(a) = ctx.frame.accumulator
        && let JsValue::Smi(b) = rhs
    {
        ctx.frame.accumulator = JsValue::Boolean(a > *b);
        return Ok(DispatchAction::Continue);
    }
    let rhs = rhs.clone();
    // a > b  ≡  b < a
    let result = js_less_than(&rhs, &ctx.frame.accumulator)?;
    ctx.frame.accumulator = JsValue::Boolean(result);
    Ok(DispatchAction::Continue)
}

fn handle_test_less_than_or_equal(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("TestLessThanOrEqual", 0));
    };
    let rhs = ctx.frame.read_reg(v)?;
    // Fast path: Smi <= Smi
    if let JsValue::Smi(a) = ctx.frame.accumulator
        && let JsValue::Smi(b) = rhs
    {
        ctx.frame.accumulator = JsValue::Boolean(a <= *b);
        return Ok(DispatchAction::Continue);
    }
    let rhs = rhs.clone();
    // a <= b  ≡  !(b < a)
    let result = !js_less_than(&rhs, &ctx.frame.accumulator)?;
    ctx.frame.accumulator = JsValue::Boolean(result);
    Ok(DispatchAction::Continue)
}

fn handle_test_greater_than_or_equal(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("TestGreaterThanOrEqual", 0));
    };
    let rhs = ctx.frame.read_reg(v)?;
    // Fast path: Smi >= Smi
    if let JsValue::Smi(a) = ctx.frame.accumulator
        && let JsValue::Smi(b) = rhs
    {
        ctx.frame.accumulator = JsValue::Boolean(a >= *b);
        return Ok(DispatchAction::Continue);
    }
    let rhs = rhs.clone();
    // a >= b  ≡  !(a < b)
    let result = !js_less_than(&ctx.frame.accumulator, &rhs)?;
    ctx.frame.accumulator = JsValue::Boolean(result);
    Ok(DispatchAction::Continue)
}

fn handle_test_null(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    ctx.frame.accumulator = JsValue::Boolean(ctx.frame.accumulator.is_null());
    Ok(DispatchAction::Continue)
}

fn handle_test_undefined(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    ctx.frame.accumulator = JsValue::Boolean(ctx.frame.accumulator.is_undefined());
    Ok(DispatchAction::Continue)
}

fn handle_logical_not(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    // `!acc` — the compiler emits this when acc is already a
    // boolean.  We coerce via ToBoolean for safety.
    ctx.frame.accumulator = JsValue::Boolean(!ctx.frame.accumulator.to_boolean());
    Ok(DispatchAction::Continue)
}

fn handle_to_boolean_logical_not(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    // `!ToBoolean(acc)` — explicit coercion before negation.
    ctx.frame.accumulator = JsValue::Boolean(!ctx.frame.accumulator.to_boolean());
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_jump(ctx: &mut DispatchContext, instr: &Instruction) -> StatorResult<DispatchAction> {
    let Operand::JumpOffset(delta) = instr.operands[0] else {
        return Err(err_bad_operand("Jump", 0));
    };
    ctx.frame.pc = resolve_jump(
        ctx.frame.pc,
        delta,
        ctx.byte_offsets,
        ctx.instructions.len(),
    )?;
    Ok(DispatchAction::Continue)
}

fn handle_jump_loop(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::JumpOffset(delta) = instr.operands[0] else {
        return Err(err_bad_operand("JumpLoop", 0));
    };
    ctx.frame.pc = resolve_jump(
        ctx.frame.pc,
        delta,
        ctx.byte_offsets,
        ctx.instructions.len(),
    )?;
    ctx.frame.osr_loop_count = ctx.frame.osr_loop_count.saturating_add(1);
    if ctx.frame.osr_loop_count >= OSR_LOOP_THRESHOLD
        && ctx.frame.bytecode_array.try_get_jit_code().is_none()
    {
        maybe_compile_baseline(&ctx.frame.bytecode_array);
    }
    if ctx.frame.osr_loop_count >= MAGLEV_OSR_LOOP_THRESHOLD {
        maybe_compile_maglev(&ctx.frame.bytecode_array);
    }
    if ctx.frame.osr_loop_count >= TURBOFAN_OSR_LOOP_THRESHOLD {
        maybe_compile_turbofan(&ctx.frame.bytecode_array);
    }
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_jump_if_true(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::JumpOffset(delta) = instr.operands[0] else {
        return Err(err_bad_operand("JumpIfTrue", 0));
    };
    if matches!(ctx.frame.accumulator, JsValue::Boolean(true)) {
        ctx.frame.pc = resolve_jump(
            ctx.frame.pc,
            delta,
            ctx.byte_offsets,
            ctx.instructions.len(),
        )?;
    }
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_jump_if_false(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::JumpOffset(delta) = instr.operands[0] else {
        return Err(err_bad_operand("JumpIfFalse", 0));
    };
    if matches!(ctx.frame.accumulator, JsValue::Boolean(false)) {
        ctx.frame.pc = resolve_jump(
            ctx.frame.pc,
            delta,
            ctx.byte_offsets,
            ctx.instructions.len(),
        )?;
    }
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_jump_if_to_boolean_true(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::JumpOffset(delta) = instr.operands[0] else {
        return Err(err_bad_operand("JumpIfToBooleanTrue", 0));
    };
    if ctx.frame.accumulator.to_boolean() {
        ctx.frame.pc = resolve_jump(
            ctx.frame.pc,
            delta,
            ctx.byte_offsets,
            ctx.instructions.len(),
        )?;
    }
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_jump_if_to_boolean_false(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::JumpOffset(delta) = instr.operands[0] else {
        return Err(err_bad_operand("JumpIfToBooleanFalse", 0));
    };
    if !ctx.frame.accumulator.to_boolean() {
        ctx.frame.pc = resolve_jump(
            ctx.frame.pc,
            delta,
            ctx.byte_offsets,
            ctx.instructions.len(),
        )?;
    }
    Ok(DispatchAction::Continue)
}

fn handle_jump_if_null(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::JumpOffset(delta) = instr.operands[0] else {
        return Err(err_bad_operand("JumpIfNull", 0));
    };
    if ctx.frame.accumulator.is_null() {
        ctx.frame.pc = resolve_jump(
            ctx.frame.pc,
            delta,
            ctx.byte_offsets,
            ctx.instructions.len(),
        )?;
    }
    Ok(DispatchAction::Continue)
}

fn handle_jump_if_not_null(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::JumpOffset(delta) = instr.operands[0] else {
        return Err(err_bad_operand("JumpIfNotNull", 0));
    };
    if !ctx.frame.accumulator.is_null() {
        ctx.frame.pc = resolve_jump(
            ctx.frame.pc,
            delta,
            ctx.byte_offsets,
            ctx.instructions.len(),
        )?;
    }
    Ok(DispatchAction::Continue)
}

fn handle_jump_if_undefined(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::JumpOffset(delta) = instr.operands[0] else {
        return Err(err_bad_operand("JumpIfUndefined", 0));
    };
    if ctx.frame.accumulator.is_undefined() {
        ctx.frame.pc = resolve_jump(
            ctx.frame.pc,
            delta,
            ctx.byte_offsets,
            ctx.instructions.len(),
        )?;
    }
    Ok(DispatchAction::Continue)
}

fn handle_jump_if_not_undefined(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::JumpOffset(delta) = instr.operands[0] else {
        return Err(err_bad_operand("JumpIfNotUndefined", 0));
    };
    if !ctx.frame.accumulator.is_undefined() {
        ctx.frame.pc = resolve_jump(
            ctx.frame.pc,
            delta,
            ctx.byte_offsets,
            ctx.instructions.len(),
        )?;
    }
    Ok(DispatchAction::Continue)
}

fn handle_jump_if_undefined_or_null(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::JumpOffset(delta) = instr.operands[0] else {
        return Err(err_bad_operand("JumpIfUndefinedOrNull", 0));
    };
    if ctx.frame.accumulator.is_nullish() {
        ctx.frame.pc = resolve_jump(
            ctx.frame.pc,
            delta,
            ctx.byte_offsets,
            ctx.instructions.len(),
        )?;
    }
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_return(ctx: &mut DispatchContext, _instr: &Instruction) -> StatorResult<DispatchAction> {
    Ok(DispatchAction::Return(ctx.frame.accumulator.clone()))
}

fn handle_create_closure(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(idx) = instr.operands[0] else {
        return Err(err_bad_operand("CreateClosure", 0));
    };
    let entry = ctx.frame.bytecode_array.get_constant(idx).ok_or_else(|| {
        StatorError::Internal(format!(
            "CreateClosure: constant pool index {idx} out of bounds"
        ))
    })?;
    let ConstantPoolEntry::Function(ba) = entry else {
        return Err(StatorError::Internal(
            "CreateClosure: constant pool entry is not a Function".into(),
        ));
    };
    let mut closure_ba = (**ba).clone();
    // Capture the enclosing context so the closure can walk the scope chain.
    if let Some(JsValue::Context(c)) = &ctx.frame.context {
        closure_ba.set_closure_context(Rc::clone(c));
    }
    ctx.frame.accumulator = JsValue::Function(Rc::new(closure_ba));
    Ok(DispatchAction::Continue)
}

fn handle_call_any_receiver(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(callee_v) = instr.operands[0] else {
        return Err(err_bad_operand("CallAnyReceiver", 0));
    };
    let Operand::Register(args_start_v) = instr.operands[1] else {
        return Err(err_bad_operand("CallAnyReceiver", 1));
    };
    let Operand::RegisterCount(arg_count) = instr.operands[2] else {
        return Err(err_bad_operand("CallAnyReceiver", 2));
    };
    let callee = ctx.frame.read_reg(callee_v)?.clone();
    match callee {
        JsValue::Function(ba) => {
            if ba.is_generator() {
                ctx.frame.accumulator = JsValue::Generator(GeneratorState::new((*ba).clone()));
            } else if ba.is_async() {
                let args = collect_args(ctx.frame, args_start_v, arg_count)?;
                ctx.frame.accumulator = Interpreter::run_async_function((*ba).clone(), args)?;
            } else {
                let args = collect_args(ctx.frame, args_start_v, arg_count)?;
                // ── Tiering ──────────────────────────────────
                let count = ba.increment_invocation_count();
                if count >= TIERING_THRESHOLD && ba.try_get_jit_code().is_none() {
                    maybe_compile_baseline(&ba);
                }
                if count >= MAGLEV_TIERING_THRESHOLD {
                    maybe_compile_maglev(&ba);
                }
                if count >= TURBOFAN_TIERING_THRESHOLD {
                    maybe_compile_turbofan(&ba);
                }
                let mut tried_jit = false;
                if let Some(jit_result) = try_execute_best_jit(&ba, &args) {
                    ctx.frame.accumulator = jit_result?;
                    tried_jit = true;
                }
                if !tried_jit {
                    let mut callee_frame = InterpreterFrame::new_with_globals(
                        (*ba).clone(),
                        args,
                        Rc::clone(&ctx.frame.global_env),
                    );
                    restore_closure_context(&mut callee_frame, &ba);
                    push_call_frame("<anonymous>")?;
                    let result = Interpreter::run(&mut callee_frame);
                    pop_call_frame();
                    ctx.frame.accumulator = result?;
                }
            }
        }
        JsValue::NativeFunction(f) => {
            let args = collect_args(ctx.frame, args_start_v, arg_count)?;
            ctx.frame.accumulator = f(args)?;
        }
        JsValue::PlainObject(ref map) => {
            let call_val = map.borrow().get("__call__").cloned();
            match call_val {
                Some(JsValue::NativeFunction(f)) => {
                    let args = collect_args(ctx.frame, args_start_v, arg_count)?;
                    ctx.frame.accumulator = f(args)?;
                }
                Some(JsValue::Function(ba)) => {
                    let args = collect_args(ctx.frame, args_start_v, arg_count)?;
                    call_plain_object_function(ctx, &ba, args)?;
                }
                _ => {
                    return Err(StatorError::TypeError(
                        "CallAnyReceiver: callee is not a function (got PlainObject)".to_string(),
                    ));
                }
            }
        }
        other => {
            return Err(StatorError::TypeError(format!(
                "CallAnyReceiver: callee is not a function (got {other:?})"
            )));
        }
    }
    Ok(DispatchAction::Continue)
}

fn handle_tail_call(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(callee_v) = instr.operands[0] else {
        return Err(err_bad_operand("TailCall", 0));
    };
    let Operand::Register(args_start_v) = instr.operands[1] else {
        return Err(err_bad_operand("TailCall", 1));
    };
    let Operand::RegisterCount(arg_count) = instr.operands[2] else {
        return Err(err_bad_operand("TailCall", 2));
    };
    let callee = ctx.frame.read_reg(callee_v)?.clone();
    match callee {
        JsValue::Function(ba) => {
            if ba.is_generator() || ba.is_async() {
                // Generators/async cannot be tail-called;
                // fall back to a normal call.
                let args = collect_args(ctx.frame, args_start_v, arg_count)?;
                if ba.is_generator() {
                    ctx.frame.accumulator = JsValue::Generator(GeneratorState::new((*ba).clone()));
                } else {
                    ctx.frame.accumulator = Interpreter::run_async_function((*ba).clone(), args)?;
                }
            } else {
                let args = collect_args(ctx.frame, args_start_v, arg_count)?;
                // ── Tiering ──────────────────────────────
                let count = ba.increment_invocation_count();
                if count >= TIERING_THRESHOLD && ba.try_get_jit_code().is_none() {
                    maybe_compile_baseline(&ba);
                }
                if count >= MAGLEV_TIERING_THRESHOLD {
                    maybe_compile_maglev(&ba);
                }
                if count >= TURBOFAN_TIERING_THRESHOLD {
                    maybe_compile_turbofan(&ba);
                }
                let mut tried_jit = false;
                if let Some(jit_result) = try_execute_best_jit(&ba, &args) {
                    ctx.frame.accumulator = jit_result?;
                    tried_jit = true;
                }
                if !tried_jit {
                    // ── Proper tail call: reuse the frame ─
                    let new_ba = (*ba).clone();
                    let param_count = new_ba.parameter_count() as usize;
                    let frame_size = new_ba.frame_size() as usize;
                    let total_regs = param_count + frame_size;
                    ctx.frame.bytecode_array = new_ba;
                    ctx.frame.registers.clear();
                    ctx.frame.registers.resize(total_regs, JsValue::Undefined);
                    for (i, arg) in args.into_iter().enumerate().take(param_count) {
                        ctx.frame.registers[i] = arg;
                    }
                    ctx.frame.accumulator = JsValue::Undefined;
                    ctx.frame.pc = 0;
                    ctx.frame.context =
                        ba.closure_context().map(|c| JsValue::Context(Rc::clone(c)));
                    ctx.frame.string_cache.clear();
                    ctx.frame.mono_load_cache.clear();
                    ctx.frame.poly_load_cache.clear();
                    ctx.frame.shape_load_ic.clear();
                    ctx.frame.shape_store_ic.clear();
                    return Ok(DispatchAction::TailCall);
                }
            }
        }
        JsValue::NativeFunction(f) => {
            let args = collect_args(ctx.frame, args_start_v, arg_count)?;
            ctx.frame.accumulator = f(args)?;
        }
        JsValue::PlainObject(ref map) => {
            if let Some(JsValue::NativeFunction(f)) = map.borrow().get("__call__").cloned() {
                let args = collect_args(ctx.frame, args_start_v, arg_count)?;
                ctx.frame.accumulator = f(args)?;
            } else {
                return Err(StatorError::TypeError(
                    "TailCall: callee is not a function (got PlainObject)".to_string(),
                ));
            }
        }
        other => {
            return Err(StatorError::TypeError(format!(
                "TailCall: callee is not a function (got {other:?})"
            )));
        }
    }
    Ok(DispatchAction::Continue)
}

fn handle_call_undefined_receiver0(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(callee_v) = instr.operands[0] else {
        return Err(err_bad_operand("CallUndefinedReceiver0", 0));
    };
    let callee = ctx.frame.read_reg(callee_v)?.clone();
    match callee {
        JsValue::Function(ba) => {
            if ba.is_generator() {
                ctx.frame.accumulator = JsValue::Generator(GeneratorState::new((*ba).clone()));
            } else if ba.is_async() {
                ctx.frame.accumulator = Interpreter::run_async_function((*ba).clone(), vec![])?;
            } else {
                let args: Vec<JsValue> = vec![];
                let count = ba.increment_invocation_count();
                if count >= TIERING_THRESHOLD && ba.try_get_jit_code().is_none() {
                    maybe_compile_baseline(&ba);
                }
                if count >= MAGLEV_TIERING_THRESHOLD {
                    maybe_compile_maglev(&ba);
                }
                if count >= TURBOFAN_TIERING_THRESHOLD {
                    maybe_compile_turbofan(&ba);
                }
                let mut tried_jit = false;
                if let Some(jit_result) = try_execute_best_jit(&ba, &args) {
                    ctx.frame.accumulator = jit_result?;
                    tried_jit = true;
                }
                if !tried_jit {
                    // Strict mode: `this` is undefined for free function calls.
                    let saved_this = if ba.is_strict() {
                        let old = ctx.frame.global_env.borrow().get("this").cloned();
                        ctx.frame
                            .global_env
                            .borrow_mut()
                            .insert("this".to_string(), JsValue::Undefined);
                        old
                    } else {
                        None
                    };
                    let mut callee_frame = InterpreterFrame::new_with_globals(
                        (*ba).clone(),
                        args,
                        Rc::clone(&ctx.frame.global_env),
                    );
                    restore_closure_context(&mut callee_frame, &ba);
                    push_call_frame("<anonymous>")?;
                    let result = Interpreter::run(&mut callee_frame);
                    pop_call_frame();
                    if ba.is_strict() {
                        match saved_this {
                            Some(v) => {
                                ctx.frame
                                    .global_env
                                    .borrow_mut()
                                    .insert("this".to_string(), v);
                            }
                            None => {
                                ctx.frame.global_env.borrow_mut().remove("this");
                            }
                        }
                    }
                    ctx.frame.accumulator = result?;
                }
            }
        }
        JsValue::NativeFunction(f) => {
            ctx.frame.accumulator = f(vec![])?;
        }
        JsValue::PlainObject(ref map) => {
            if let Some(JsValue::NativeFunction(f)) = map.borrow().get("__call__").cloned() {
                ctx.frame.accumulator = f(vec![])?;
            } else {
                return Err(StatorError::TypeError(
                    "CallUndefinedReceiver0: callee is not a function (got PlainObject)"
                        .to_string(),
                ));
            }
        }
        other => {
            return Err(StatorError::TypeError(format!(
                "CallUndefinedReceiver0: callee is not a function (got {other:?})"
            )));
        }
    }
    Ok(DispatchAction::Continue)
}

fn handle_call_undefined_receiver1(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(callee_v) = instr.operands[0] else {
        return Err(err_bad_operand("CallUndefinedReceiver1", 0));
    };
    let Operand::Register(arg1_v) = instr.operands[1] else {
        return Err(err_bad_operand("CallUndefinedReceiver1", 1));
    };
    let callee = ctx.frame.read_reg(callee_v)?.clone();
    match callee {
        JsValue::Function(ba) => {
            if ba.is_generator() {
                ctx.frame.accumulator = JsValue::Generator(GeneratorState::new((*ba).clone()));
            } else if ba.is_async() {
                let arg1 = ctx.frame.read_reg(arg1_v)?.clone();
                ctx.frame.accumulator = Interpreter::run_async_function((*ba).clone(), vec![arg1])?;
            } else {
                let arg1 = ctx.frame.read_reg(arg1_v)?.clone();
                let args = vec![arg1];
                let count = ba.increment_invocation_count();
                if count >= TIERING_THRESHOLD && ba.try_get_jit_code().is_none() {
                    maybe_compile_baseline(&ba);
                }
                if count >= MAGLEV_TIERING_THRESHOLD {
                    maybe_compile_maglev(&ba);
                }
                if count >= TURBOFAN_TIERING_THRESHOLD {
                    maybe_compile_turbofan(&ba);
                }
                let mut tried_jit = false;
                if let Some(jit_result) = try_execute_best_jit(&ba, &args) {
                    ctx.frame.accumulator = jit_result?;
                    tried_jit = true;
                }
                if !tried_jit {
                    // Strict mode: `this` is undefined for free function calls.
                    let saved_this = if ba.is_strict() {
                        let old = ctx.frame.global_env.borrow().get("this").cloned();
                        ctx.frame
                            .global_env
                            .borrow_mut()
                            .insert("this".to_string(), JsValue::Undefined);
                        old
                    } else {
                        None
                    };
                    let mut callee_frame = InterpreterFrame::new_with_globals(
                        (*ba).clone(),
                        args,
                        Rc::clone(&ctx.frame.global_env),
                    );
                    restore_closure_context(&mut callee_frame, &ba);
                    push_call_frame("<anonymous>")?;
                    let result = Interpreter::run(&mut callee_frame);
                    pop_call_frame();
                    if ba.is_strict() {
                        match saved_this {
                            Some(v) => {
                                ctx.frame
                                    .global_env
                                    .borrow_mut()
                                    .insert("this".to_string(), v);
                            }
                            None => {
                                ctx.frame.global_env.borrow_mut().remove("this");
                            }
                        }
                    }
                    ctx.frame.accumulator = result?;
                }
            }
        }
        JsValue::NativeFunction(f) => {
            let arg1 = ctx.frame.read_reg(arg1_v)?.clone();
            ctx.frame.accumulator = f(vec![arg1])?;
        }
        JsValue::PlainObject(ref map) => {
            if let Some(JsValue::NativeFunction(f)) = map.borrow().get("__call__").cloned() {
                let arg1 = ctx.frame.read_reg(arg1_v)?.clone();
                ctx.frame.accumulator = f(vec![arg1])?;
            } else {
                return Err(StatorError::TypeError(
                    "CallUndefinedReceiver1: callee is not a function (got PlainObject)"
                        .to_string(),
                ));
            }
        }
        other => {
            return Err(StatorError::TypeError(format!(
                "CallUndefinedReceiver1: callee is not a function (got {other:?})"
            )));
        }
    }
    Ok(DispatchAction::Continue)
}

fn handle_call_undefined_receiver2(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(callee_v) = instr.operands[0] else {
        return Err(err_bad_operand("CallUndefinedReceiver2", 0));
    };
    let Operand::Register(arg1_v) = instr.operands[1] else {
        return Err(err_bad_operand("CallUndefinedReceiver2", 1));
    };
    let Operand::Register(arg2_v) = instr.operands[2] else {
        return Err(err_bad_operand("CallUndefinedReceiver2", 2));
    };
    let callee = ctx.frame.read_reg(callee_v)?.clone();
    match callee {
        JsValue::Function(ba) => {
            if ba.is_generator() {
                ctx.frame.accumulator = JsValue::Generator(GeneratorState::new((*ba).clone()));
            } else if ba.is_async() {
                let arg1 = ctx.frame.read_reg(arg1_v)?.clone();
                let arg2 = ctx.frame.read_reg(arg2_v)?.clone();
                ctx.frame.accumulator =
                    Interpreter::run_async_function((*ba).clone(), vec![arg1, arg2])?;
            } else {
                let arg1 = ctx.frame.read_reg(arg1_v)?.clone();
                let arg2 = ctx.frame.read_reg(arg2_v)?.clone();
                let args = vec![arg1, arg2];
                // ── Tiering ──────────────────────────────────
                let count = ba.increment_invocation_count();
                if count >= TIERING_THRESHOLD && ba.try_get_jit_code().is_none() {
                    maybe_compile_baseline(&ba);
                }
                if count >= MAGLEV_TIERING_THRESHOLD {
                    maybe_compile_maglev(&ba);
                }
                if count >= TURBOFAN_TIERING_THRESHOLD {
                    maybe_compile_turbofan(&ba);
                }
                let mut tried_jit = false;
                if let Some(jit_result) = try_execute_best_jit(&ba, &args) {
                    ctx.frame.accumulator = jit_result?;
                    tried_jit = true;
                }
                if !tried_jit {
                    // Strict mode: `this` is undefined for free function calls.
                    let saved_this = if ba.is_strict() {
                        let old = ctx.frame.global_env.borrow().get("this").cloned();
                        ctx.frame
                            .global_env
                            .borrow_mut()
                            .insert("this".to_string(), JsValue::Undefined);
                        old
                    } else {
                        None
                    };
                    let mut callee_frame = InterpreterFrame::new_with_globals(
                        (*ba).clone(),
                        args,
                        Rc::clone(&ctx.frame.global_env),
                    );
                    restore_closure_context(&mut callee_frame, &ba);
                    push_call_frame("<anonymous>")?;
                    let result = Interpreter::run(&mut callee_frame);
                    pop_call_frame();
                    if ba.is_strict() {
                        match saved_this {
                            Some(v) => {
                                ctx.frame
                                    .global_env
                                    .borrow_mut()
                                    .insert("this".to_string(), v);
                            }
                            None => {
                                ctx.frame.global_env.borrow_mut().remove("this");
                            }
                        }
                    }
                    ctx.frame.accumulator = result?;
                }
            }
        }
        JsValue::NativeFunction(f) => {
            let arg1 = ctx.frame.read_reg(arg1_v)?.clone();
            let arg2 = ctx.frame.read_reg(arg2_v)?.clone();
            ctx.frame.accumulator = f(vec![arg1, arg2])?;
        }
        JsValue::PlainObject(ref map) => {
            if let Some(JsValue::NativeFunction(f)) = map.borrow().get("__call__").cloned() {
                let arg1 = ctx.frame.read_reg(arg1_v)?.clone();
                let arg2 = ctx.frame.read_reg(arg2_v)?.clone();
                ctx.frame.accumulator = f(vec![arg1, arg2])?;
            } else {
                return Err(StatorError::TypeError(
                    "CallUndefinedReceiver2: callee is not a function (got PlainObject)"
                        .to_string(),
                ));
            }
        }
        other => {
            return Err(StatorError::TypeError(format!(
                "CallUndefinedReceiver2: callee is not a function (got {other:?})"
            )));
        }
    }
    Ok(DispatchAction::Continue)
}

fn handle_call_property(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(callee_v) = instr.operands[0] else {
        return Err(err_bad_operand("CallProperty", 0));
    };
    let Operand::Register(recv_v) = instr.operands[1] else {
        return Err(err_bad_operand("CallProperty", 1));
    };
    let Operand::RegisterCount(arg_count) = instr.operands[2] else {
        return Err(err_bad_operand("CallProperty", 2));
    };
    let callee = ctx.frame.read_reg(callee_v)?.clone();
    // Arguments reside in the registers immediately following
    // the callee register in the flat register file.
    let callee_flat = ctx.frame.reg_index(callee_v)?;
    let args = (0..arg_count as usize)
        .map(|i| ctx.frame.registers[callee_flat + 1 + i].clone())
        .collect::<Vec<_>>();
    match callee {
        JsValue::Function(ba) => {
            let this_val = ctx.frame.read_reg(recv_v)?.clone();
            // ── Tiering ──────────────────────────────────────
            let count = ba.increment_invocation_count();
            if count >= TIERING_THRESHOLD && ba.try_get_jit_code().is_none() {
                maybe_compile_baseline(&ba);
            }
            if count >= MAGLEV_TIERING_THRESHOLD {
                maybe_compile_maglev(&ba);
            }
            if count >= TURBOFAN_TIERING_THRESHOLD {
                maybe_compile_turbofan(&ba);
            }
            let mut tried_jit = false;
            if let Some(jit_result) = try_execute_best_jit(&ba, &args) {
                ctx.frame.accumulator = jit_result?;
                tried_jit = true;
            }
            if !tried_jit {
                let mut callee_frame = InterpreterFrame::new_with_globals(
                    (*ba).clone(),
                    args,
                    Rc::clone(&ctx.frame.global_env),
                );
                restore_closure_context(&mut callee_frame, &ba);
                callee_frame
                    .global_env
                    .borrow_mut()
                    .insert("this".to_string(), this_val);
                push_call_frame("<anonymous>")?;
                let result = Interpreter::run(&mut callee_frame);
                pop_call_frame();
                ctx.frame.accumulator = result?;
            }
        }
        JsValue::NativeFunction(f) => {
            ctx.frame.accumulator = f(args)?;
        }
        JsValue::PlainObject(ref map) => {
            let call_val = map.borrow().get("__call__").cloned();
            match call_val {
                Some(JsValue::NativeFunction(f)) => {
                    ctx.frame.accumulator = f(args)?;
                }
                Some(JsValue::Function(ba)) => {
                    call_plain_object_function(ctx, &ba, args)?;
                }
                _ => {
                    return Err(StatorError::TypeError(
                        "CallProperty: callee is not a function (got PlainObject)".to_string(),
                    ));
                }
            }
        }
        other => {
            return Err(StatorError::TypeError(format!(
                "CallProperty: callee is not a function (got {other:?})"
            )));
        }
    }
    Ok(DispatchAction::Continue)
}

fn handle_call_with_spread(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(callee_v) = instr.operands[0] else {
        return Err(err_bad_operand("CallWithSpread", 0));
    };
    let Operand::Register(args_start_v) = instr.operands[1] else {
        return Err(err_bad_operand("CallWithSpread", 1));
    };
    let Operand::RegisterCount(arg_count) = instr.operands[2] else {
        return Err(err_bad_operand("CallWithSpread", 2));
    };
    let callee = ctx.frame.read_reg(callee_v)?.clone();
    let raw_args = collect_args(ctx.frame, args_start_v, arg_count)?;
    // If the bytecode generator packed all arguments into a single array,
    // expand it into the real argument list.
    let args = if raw_args.len() == 1 {
        if let JsValue::Array(ref items) = raw_args[0] {
            items.borrow().clone()
        } else {
            raw_args
        }
    } else {
        raw_args
    };
    match callee {
        JsValue::Function(ba) => {
            // ── Tiering ──────────────────────────────────────
            let count = ba.increment_invocation_count();
            if count >= TIERING_THRESHOLD && ba.try_get_jit_code().is_none() {
                maybe_compile_baseline(&ba);
            }
            if count >= MAGLEV_TIERING_THRESHOLD {
                maybe_compile_maglev(&ba);
            }
            if count >= TURBOFAN_TIERING_THRESHOLD {
                maybe_compile_turbofan(&ba);
            }
            let mut tried_jit = false;
            if let Some(jit_result) = try_execute_best_jit(&ba, &args) {
                ctx.frame.accumulator = jit_result?;
                tried_jit = true;
            }
            if !tried_jit {
                let mut callee_frame = InterpreterFrame::new_with_globals(
                    (*ba).clone(),
                    args,
                    Rc::clone(&ctx.frame.global_env),
                );
                restore_closure_context(&mut callee_frame, &ba);
                push_call_frame("<anonymous>")?;
                let result = Interpreter::run(&mut callee_frame);
                pop_call_frame();
                ctx.frame.accumulator = result?;
            }
        }
        JsValue::NativeFunction(f) => {
            ctx.frame.accumulator = f(args)?;
        }
        JsValue::PlainObject(ref map) => {
            if let Some(JsValue::NativeFunction(f)) = map.borrow().get("__call__").cloned() {
                ctx.frame.accumulator = f(args)?;
            } else {
                return Err(StatorError::TypeError(
                    "CallWithSpread: callee is not a function (got PlainObject)".to_string(),
                ));
            }
        }
        other => {
            return Err(StatorError::TypeError(format!(
                "CallWithSpread: callee is not a function (got {other:?})"
            )));
        }
    }
    Ok(DispatchAction::Continue)
}

/// Call a `PlainObject`-wrapped class constructor as a regular function.
///
/// Used when `super()` appears in a derived constructor — the parent
/// constructor is called (not `[[Construct]]`-ed).  The shared global
/// environment already has `"this"` set by the enclosing `[[Construct]]`
/// handler, so the parent body can see and mutate the same object.
fn call_plain_object_function(
    ctx: &mut DispatchContext,
    ba: &Rc<crate::bytecode::bytecode_array::BytecodeArray>,
    args: Vec<JsValue>,
) -> StatorResult<()> {
    let mut callee_frame =
        InterpreterFrame::new_with_globals((**ba).clone(), args, Rc::clone(&ctx.frame.global_env));
    restore_closure_context(&mut callee_frame, ba);
    push_call_frame("<anonymous>")?;
    let result = Interpreter::run(&mut callee_frame);
    pop_call_frame();
    ctx.frame.accumulator = result?;
    Ok(())
}

/// Shared helper for `[[Construct]]` on a class created by `CreateClass`.
///
/// The class constructor is a `PlainObject` whose `__call__` holds the
/// constructor bytecode (`JsValue::Function`).  This function:
///   1. Creates a fresh `this` object with `__proto__` = constructor's
///      `prototype`.
///   2. Sets `"this"` and (when the class `extends`) `"super"` in the
///      global environment so the constructor body can access them.
///   3. Runs the constructor bytecode with `new_target` set.
///   4. Invokes the hidden `.class_field_initializer` closure, if present.
///   5. Returns the `this` object (or an explicit object returned by the
///      constructor).
fn construct_class_from_plain_object(
    ctx: &mut DispatchContext,
    ba: &Rc<crate::bytecode::bytecode_array::BytecodeArray>,
    class_map: &Rc<RefCell<PropertyMap>>,
    ctor_proto: &JsValue,
    args: Vec<JsValue>,
) -> StatorResult<()> {
    // 1. Create `this`.
    let this_obj: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));
    if !matches!(ctor_proto, JsValue::Undefined) {
        this_obj
            .borrow_mut()
            .insert("__proto__".to_string(), ctor_proto.clone());
    }
    let this_val = JsValue::PlainObject(Rc::clone(&this_obj));

    // 2. Set up callee frame.
    let mut callee_frame =
        InterpreterFrame::new_with_globals((**ba).clone(), args, Rc::clone(&ctx.frame.global_env));
    restore_closure_context(&mut callee_frame, ba);
    callee_frame.new_target = JsValue::PlainObject(Rc::clone(class_map));
    callee_frame
        .global_env
        .borrow_mut()
        .insert("this".to_string(), this_val.clone());

    // 3. Expose parent constructor as "super" for `super()` calls.
    if let Some(parent) = class_map.borrow().get("__proto__").cloned()
        && !matches!(parent, JsValue::Undefined | JsValue::Null)
    {
        callee_frame
            .global_env
            .borrow_mut()
            .insert("super".to_string(), parent);
    }

    // 4. Run constructor body.
    push_call_frame("<anonymous>")?;
    let result = Interpreter::run(&mut callee_frame);
    pop_call_frame();
    let val = result?;

    // 5. Run field initializer if present.
    if let Some(JsValue::Function(init_ba)) =
        class_map.borrow().get(".class_field_initializer").cloned()
    {
        let mut init_frame = InterpreterFrame::new_with_globals(
            (*init_ba).clone(),
            vec![this_val.clone()],
            Rc::clone(&ctx.frame.global_env),
        );
        restore_closure_context(&mut init_frame, &init_ba);
        push_call_frame("<field_init>")?;
        let _ = Interpreter::run(&mut init_frame);
        pop_call_frame();
    }

    // 6. If constructor returns an object, use it; else use `this`.
    ctx.frame.accumulator = match val {
        JsValue::PlainObject(_) | JsValue::Object(_) => val,
        _ => this_val,
    };
    Ok(())
}

fn handle_construct(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(ctor_v) = instr.operands[0] else {
        return Err(err_bad_operand("Construct", 0));
    };
    let Operand::Register(args_start_v) = instr.operands[1] else {
        return Err(err_bad_operand("Construct", 1));
    };
    let Operand::RegisterCount(arg_count) = instr.operands[2] else {
        return Err(err_bad_operand("Construct", 2));
    };
    let ctor = ctx.frame.read_reg(ctor_v)?.clone();
    // Resolve constructor's "prototype" for [[Prototype]] wiring.
    let ctor_proto = proto_lookup(&ctor, "prototype");
    match ctor {
        JsValue::Function(ba) => {
            let args = collect_args(ctx.frame, args_start_v, arg_count)?;
            // [[Construct]]: create a fresh object for `this`,
            // wire its __proto__ to the constructor's prototype,
            // then run the constructor body with `this` bound.
            let this_obj: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));
            if !matches!(ctor_proto, JsValue::Undefined) {
                this_obj
                    .borrow_mut()
                    .insert("__proto__".to_string(), ctor_proto.clone());
            }
            let this_val = JsValue::PlainObject(this_obj);
            let mut callee_frame = InterpreterFrame::new_with_globals(
                (*ba).clone(),
                args,
                Rc::clone(&ctx.frame.global_env),
            );
            restore_closure_context(&mut callee_frame, &ba);
            callee_frame.new_target = JsValue::Function(Rc::clone(&ba));
            callee_frame
                .global_env
                .borrow_mut()
                .insert("this".to_string(), this_val.clone());
            push_call_frame("<anonymous>")?;
            let result = Interpreter::run(&mut callee_frame);
            pop_call_frame();
            let val = result?;
            // If the constructor explicitly returns an object,
            // use it; otherwise return the `this` object.
            ctx.frame.accumulator = match val {
                JsValue::PlainObject(_) | JsValue::Object(_) => val,
                _ => this_val,
            };
        }
        JsValue::NativeFunction(f) => {
            let args = collect_args(ctx.frame, args_start_v, arg_count)?;
            ctx.frame.accumulator = f(args)?;
        }
        JsValue::PlainObject(ref map) => {
            let call_val = map.borrow().get("__call__").cloned();
            match call_val {
                Some(JsValue::NativeFunction(f)) => {
                    let args = collect_args(ctx.frame, args_start_v, arg_count)?;
                    let val = f(args)?;
                    ctx.frame.accumulator = wire_construct_prototype(val, &ctor_proto);
                }
                Some(JsValue::Function(ba)) => {
                    let args = collect_args(ctx.frame, args_start_v, arg_count)?;
                    construct_class_from_plain_object(ctx, &ba, map, &ctor_proto, args)?;
                }
                _ => {
                    return Err(StatorError::TypeError(format!(
                        "Construct: constructor is not a function (got {other:?})",
                        other = JsValue::PlainObject(Rc::clone(map))
                    )));
                }
            }
        }
        other => {
            return Err(StatorError::TypeError(format!(
                "Construct: constructor is not a function (got {other:?})"
            )));
        }
    }
    Ok(DispatchAction::Continue)
}

/// Like `handle_construct` but expands a single Array argument into the
/// actual constructor arguments (used when `new Foo(...args)` is compiled).
fn handle_construct_with_spread(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(ctor_v) = instr.operands[0] else {
        return Err(err_bad_operand("ConstructWithSpread", 0));
    };
    let Operand::Register(args_start_v) = instr.operands[1] else {
        return Err(err_bad_operand("ConstructWithSpread", 1));
    };
    let Operand::RegisterCount(arg_count) = instr.operands[2] else {
        return Err(err_bad_operand("ConstructWithSpread", 2));
    };
    let ctor = ctx.frame.read_reg(ctor_v)?.clone();
    let ctor_proto = proto_lookup(&ctor, "prototype");
    let raw_args = collect_args(ctx.frame, args_start_v, arg_count)?;
    let args = if raw_args.len() == 1 {
        if let JsValue::Array(ref items) = raw_args[0] {
            items.borrow().clone()
        } else {
            raw_args
        }
    } else {
        raw_args
    };
    match ctor {
        JsValue::Function(ba) => {
            let this_obj: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));
            if !matches!(ctor_proto, JsValue::Undefined) {
                this_obj
                    .borrow_mut()
                    .insert("__proto__".to_string(), ctor_proto.clone());
            }
            let this_val = JsValue::PlainObject(this_obj);
            let mut callee_frame = InterpreterFrame::new_with_globals(
                (*ba).clone(),
                args,
                Rc::clone(&ctx.frame.global_env),
            );
            restore_closure_context(&mut callee_frame, &ba);
            callee_frame.new_target = JsValue::Function(Rc::clone(&ba));
            callee_frame
                .global_env
                .borrow_mut()
                .insert("this".to_string(), this_val.clone());
            push_call_frame("<anonymous>")?;
            let result = Interpreter::run(&mut callee_frame);
            pop_call_frame();
            let val = result?;
            ctx.frame.accumulator = match val {
                JsValue::PlainObject(_) | JsValue::Object(_) => val,
                _ => this_val,
            };
        }
        JsValue::NativeFunction(f) => {
            ctx.frame.accumulator = f(args)?;
        }
        JsValue::PlainObject(ref map) => {
            let call_val = map.borrow().get("__call__").cloned();
            match call_val {
                Some(JsValue::NativeFunction(f)) => {
                    let val = f(args)?;
                    ctx.frame.accumulator = wire_construct_prototype(val, &ctor_proto);
                }
                Some(JsValue::Function(ba)) => {
                    construct_class_from_plain_object(ctx, &ba, map, &ctor_proto, args)?;
                }
                _ => {
                    return Err(StatorError::TypeError(
                        "ConstructWithSpread: constructor is not a function".to_string(),
                    ));
                }
            }
        }
        other => {
            return Err(StatorError::TypeError(format!(
                "ConstructWithSpread: constructor is not a function (got {other:?})"
            )));
        }
    }
    Ok(DispatchAction::Continue)
}

fn handle_push_context(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("PushContext", 0));
    };
    // Encode None as Undefined so it can be stored in a register.
    let old_ctx = ctx.frame.context.take().unwrap_or(JsValue::Undefined);
    ctx.frame.write_reg(v, old_ctx)?;
    ctx.frame.context = Some(ctx.frame.accumulator.clone());
    Ok(DispatchAction::Continue)
}

fn handle_pop_context(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("PopContext", 0));
    };
    let saved = ctx.frame.read_reg(v)?.clone();
    ctx.frame.context = if saved.is_undefined() {
        None
    } else {
        Some(saved)
    };
    Ok(DispatchAction::Continue)
}

fn handle_lda_context_slot(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(ctx_v) = instr.operands[0] else {
        return Err(err_bad_operand("LdaContextSlot", 0));
    };
    let Operand::ConstantPoolIdx(slot_idx) = instr.operands[1] else {
        return Err(err_bad_operand("LdaContextSlot", 1));
    };
    let Operand::Immediate(depth) = instr.operands[2] else {
        return Err(err_bad_operand("LdaContextSlot", 2));
    };
    let ctx_val = ctx.frame.read_reg(ctx_v)?.clone();
    let js_ctx = extract_context(&ctx_val, "LdaContextSlot")?;
    let target = walk_context_chain(&js_ctx, depth as u32, "LdaContextSlot")?;
    let borrowed = target.borrow();
    let slot = slot_idx as usize;
    ctx.frame.accumulator = borrowed
        .slots
        .get(slot)
        .cloned()
        .unwrap_or(JsValue::Undefined);
    Ok(DispatchAction::Continue)
}

fn handle_lda_current_context_slot(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(slot_idx) = instr.operands[0] else {
        return Err(err_bad_operand("LdaCurrentContextSlot", 0));
    };
    let ctx_val = ctx
        .frame
        .context
        .as_ref()
        .ok_or_else(|| StatorError::Internal("LdaCurrentContextSlot: no active context".into()))?
        .clone();
    let js_ctx = extract_context(&ctx_val, "LdaCurrentContextSlot")?;
    let borrowed = js_ctx.borrow();
    let slot = slot_idx as usize;
    ctx.frame.accumulator = borrowed
        .slots
        .get(slot)
        .cloned()
        .unwrap_or(JsValue::Undefined);
    Ok(DispatchAction::Continue)
}

fn handle_sta_context_slot(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(ctx_v) = instr.operands[0] else {
        return Err(err_bad_operand("StaContextSlot", 0));
    };
    let Operand::ConstantPoolIdx(slot_idx) = instr.operands[1] else {
        return Err(err_bad_operand("StaContextSlot", 1));
    };
    let Operand::Immediate(depth) = instr.operands[2] else {
        return Err(err_bad_operand("StaContextSlot", 2));
    };
    let ctx_val = ctx.frame.read_reg(ctx_v)?.clone();
    let js_ctx = extract_context(&ctx_val, "StaContextSlot")?;
    let target = walk_context_chain(&js_ctx, depth as u32, "StaContextSlot")?;
    let mut borrowed = target.borrow_mut();
    let slot = slot_idx as usize;
    if slot >= borrowed.slots.len() {
        borrowed.slots.resize(slot + 1, JsValue::Undefined);
    }
    borrowed.slots[slot] = ctx.frame.accumulator.clone();
    Ok(DispatchAction::Continue)
}

fn handle_sta_current_context_slot(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(slot_idx) = instr.operands[0] else {
        return Err(err_bad_operand("StaCurrentContextSlot", 0));
    };
    let ctx_val = ctx
        .frame
        .context
        .as_ref()
        .ok_or_else(|| StatorError::Internal("StaCurrentContextSlot: no active context".into()))?
        .clone();
    let js_ctx = extract_context(&ctx_val, "StaCurrentContextSlot")?;
    let mut borrowed = js_ctx.borrow_mut();
    let slot = slot_idx as usize;
    if slot >= borrowed.slots.len() {
        borrowed.slots.resize(slot + 1, JsValue::Undefined);
    }
    borrowed.slots[slot] = ctx.frame.accumulator.clone();
    Ok(DispatchAction::Continue)
}

fn handle_create_function_context(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Immediate(slot_count) = instr.operands[1] else {
        return Err(err_bad_operand("CreateFunctionContext", 1));
    };
    let parent = match &ctx.frame.context {
        Some(JsValue::Context(c)) => Some(Rc::clone(c)),
        _ => None,
    };
    let js_ctx = JsContext::new(slot_count as usize, parent);
    ctx.frame.accumulator = JsValue::Context(js_ctx);
    Ok(DispatchAction::Continue)
}

fn handle_create_block_context(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let parent = match &ctx.frame.context {
        Some(JsValue::Context(c)) => Some(Rc::clone(c)),
        _ => None,
    };
    let js_ctx = JsContext::new(0, parent);
    ctx.frame.accumulator = JsValue::Context(js_ctx);
    Ok(DispatchAction::Continue)
}

fn handle_create_eval_context(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Immediate(slot_count) = instr.operands[1] else {
        return Err(err_bad_operand("CreateEvalContext", 1));
    };
    let parent = match &ctx.frame.context {
        Some(JsValue::Context(c)) => Some(Rc::clone(c)),
        _ => None,
    };
    let js_ctx = JsContext::new(slot_count as usize, parent);
    ctx.frame.accumulator = JsValue::Context(js_ctx);
    Ok(DispatchAction::Continue)
}

fn handle_create_catch_context(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(exc_v) = instr.operands[0] else {
        return Err(err_bad_operand("CreateCatchContext", 0));
    };
    let exception = ctx.frame.read_reg(exc_v)?.clone();
    let parent = match &ctx.frame.context {
        Some(JsValue::Context(c)) => Some(Rc::clone(c)),
        _ => None,
    };
    let js_ctx = JsContext::new(1, parent);
    js_ctx.borrow_mut().slots[0] = exception;
    ctx.frame.accumulator = JsValue::Context(js_ctx);
    Ok(DispatchAction::Continue)
}

fn handle_create_with_context(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("CreateWithContext", 0));
    };
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    let parent = match &ctx.frame.context {
        Some(JsValue::Context(c)) => Some(Rc::clone(c)),
        _ => None,
    };
    let js_ctx = JsContext::new(1, parent);
    js_ctx.borrow_mut().slots[0] = obj;
    ctx.frame.accumulator = JsValue::Context(js_ctx);
    Ok(DispatchAction::Continue)
}

fn handle_throw(ctx: &mut DispatchContext, _instr: &Instruction) -> StatorResult<DispatchAction> {
    let thrown = ctx.frame.accumulator.clone();
    let throw_idx = (ctx.frame.pc - 1) as u32;
    if let Some(handler_pc) = find_handler(throw_idx, ctx.handler_table) {
        ctx.frame.accumulator = thrown;
        ctx.frame.pc = handler_pc;
        return Ok(DispatchAction::Continue);
    }
    // ── pause-on-exceptions ───────────────────────────────
    // When a debugger is attached with pause_on_exceptions
    // enabled, suspend execution *before* the exception
    // propagates.  Back up the program counter to the Throw
    // instruction so that resuming re-executes it and lets the
    // exception propagate normally (skip_next prevents a
    // double-pause on the re-execution).
    let throw_offset = ctx.byte_offsets[throw_idx as usize] as u32;
    if let Some(pause_err) = ACTIVE_DEBUGGER.with(|d| {
        let opt = d.borrow();
        opt.as_ref().and_then(|rc| {
            let mut dbg = rc.borrow_mut();
            // consume_exception_resume returns true on a
            // resume re-execution — skip the pause in that
            // case so the exception can propagate.
            if dbg.pause_on_exceptions && !dbg.consume_exception_resume() {
                ctx.frame.pc = throw_idx as usize; // back up
                Some(dbg.on_exception(throw_offset))
            } else {
                None
            }
        })
    }) {
        return Err(pause_err);
    }
    // No handler in this frame — store the original thrown value so
    // an outer frame's catch/finally can recover it, then propagate
    // as `StatorError::JsException`.
    set_pending_exception(thrown.clone());
    let msg = error_message_from_value(&thrown);
    Err(StatorError::JsException(msg))
}

fn handle_stack_check(
    _ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    Ok(DispatchAction::Continue)
}

fn handle_lda_global(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[0] else {
        return Err(err_bad_operand("LdaGlobal", 0));
    };
    let name = ctx.frame.get_string_constant(name_idx)?;
    ctx.frame.accumulator = ctx
        .frame
        .global_env
        .borrow()
        .get(name.as_ref())
        .cloned()
        .unwrap_or(JsValue::Undefined);
    Ok(DispatchAction::Continue)
}

fn handle_sta_global(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[0] else {
        return Err(err_bad_operand("StaGlobal", 0));
    };
    let name = match ctx.frame.bytecode_array.get_constant(name_idx) {
        Some(ConstantPoolEntry::String(s)) => s.clone(),
        _ => {
            return Err(StatorError::Internal(
                "StaGlobal: constant is not a string".into(),
            ));
        }
    };
    let val = ctx.frame.accumulator.clone();
    let mut env = ctx.frame.global_env.borrow_mut();
    // Strict mode: assigning to an undeclared variable is a ReferenceError.
    if ctx.frame.bytecode_array.is_strict() && !env.contains_key(&name) {
        return Err(StatorError::ReferenceError(format!(
            "{name} is not defined"
        )));
    }
    env.insert(name, val);
    Ok(DispatchAction::Continue)
}

fn handle_lda_named_property(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("LdaNamedProperty", 0));
    };
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[1] else {
        return Err(err_bad_operand("LdaNamedProperty", 1));
    };
    let slot = if let Operand::FeedbackSlot(s) = instr.operands[2] {
        s
    } else {
        u32::MAX
    };
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    // ── Shape IC fast path: O(1) own-property access via cached offset ──
    if slot != u32::MAX
        && let JsValue::PlainObject(ref map) = obj
        && let Some(ic) = ctx.frame.shape_load_ic.get(&slot)
    {
        let pm = map.borrow();
        if pm.shape_id() == ic.cached_shape
            && let Some(val) = pm.get_by_offset(ic.cached_offset)
        {
            let v = val.clone();
            drop(pm);
            ctx.frame.accumulator = v;
            return Ok(DispatchAction::Continue);
        }
    }
    // Polymorphic cache: check if any cached entry matches by pointer identity.
    if slot != u32::MAX {
        let obj_ptr = match &obj {
            JsValue::PlainObject(map) => Some(Rc::as_ptr(map) as usize),
            JsValue::Array(arr) => Some(Rc::as_ptr(arr) as usize),
            JsValue::Function(ba) => Some(Rc::as_ptr(ba) as usize),
            _ => None,
        };
        if let Some(ptr) = obj_ptr
            && let Some(entries) = ctx.frame.poly_load_cache.get(&slot)
        {
            for &(cached_ptr, ref cached_val) in entries {
                if cached_ptr == ptr {
                    ctx.frame.accumulator = cached_val.clone();
                    return Ok(DispatchAction::Continue);
                }
            }
        }
    }
    let prop_name = ctx.frame.get_string_constant(name_idx)?;
    // TypeError for property access on null or undefined (ES §13.10.3).
    if matches!(obj, JsValue::Null | JsValue::Undefined) {
        return Err(StatorError::TypeError(format!(
            "Cannot read properties of {} (reading '{prop_name}')",
            if matches!(obj, JsValue::Null) {
                "null"
            } else {
                "undefined"
            }
        )));
    }
    let result = proto_lookup(&obj, &prop_name);
    // ── Populate shape IC for own-property hits on PlainObject ───────────
    if slot != u32::MAX
        && let JsValue::PlainObject(ref map) = obj
    {
        let pm = map.borrow();
        if let Some(offset) = pm.offset_of(&prop_name) {
            ctx.frame.shape_load_ic.insert(
                slot,
                PropertyIc {
                    cached_shape: pm.shape_id(),
                    cached_offset: offset,
                },
            );
        }
    }
    // Update polymorphic cache (up to 4 entries per slot).
    if slot != u32::MAX {
        let obj_ptr = match &obj {
            JsValue::PlainObject(map) => Some(Rc::as_ptr(map) as usize),
            JsValue::Array(arr) => Some(Rc::as_ptr(arr) as usize),
            JsValue::Function(ba) => Some(Rc::as_ptr(ba) as usize),
            _ => None,
        };
        if let Some(ptr) = obj_ptr {
            let entries = ctx.frame.poly_load_cache.entry(slot).or_default();
            // Check if we already have this pointer; update in place.
            let mut found = false;
            for entry in entries.iter_mut() {
                if entry.0 == ptr {
                    entry.1 = result.clone();
                    found = true;
                    break;
                }
            }
            if !found && entries.len() < 4 {
                entries.push((ptr, result.clone()));
            }
        }
    }
    ctx.frame.accumulator = result;
    Ok(DispatchAction::Continue)
}

fn handle_sta_named_property(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("StaNamedProperty", 0));
    };
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[1] else {
        return Err(err_bad_operand("StaNamedProperty", 1));
    };
    let slot = if let Operand::FeedbackSlot(s) = instr.operands[2] {
        s
    } else {
        u32::MAX
    };
    let prop_name = ctx.frame.get_string_constant(name_idx)?;
    let val = ctx.frame.accumulator.clone();
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    // TypeError for property store on null or undefined.
    if matches!(obj, JsValue::Null | JsValue::Undefined) {
        return Err(StatorError::TypeError(format!(
            "Cannot set properties of {} (setting '{prop_name}')",
            if matches!(obj, JsValue::Null) {
                "null"
            } else {
                "undefined"
            }
        )));
    }
    match obj {
        JsValue::Proxy(ref p) => {
            let _ = proxy_set(&mut p.borrow_mut(), &prop_name, val)?;
        }
        JsValue::PlainObject(ref map) => {
            // ── Shape IC fast path for store: existing writable property ─
            if slot != u32::MAX
                && let Some(ic) = ctx.frame.shape_store_ic.get(&slot)
            {
                let pm = map.borrow();
                if pm.shape_id() == ic.cached_shape {
                    if pm.is_writable_by_offset(ic.cached_offset) {
                        drop(pm);
                        map.borrow_mut().set_by_offset(ic.cached_offset, val);
                        // Invalidate value-based caches for this object.
                        let map_ptr = Rc::as_ptr(map) as usize;
                        ctx.frame
                            .mono_load_cache
                            .retain(|_, (ptr, _)| *ptr != map_ptr);
                        ctx.frame.poly_load_cache.retain(|_, entries| {
                            entries.retain(|(ptr, _)| *ptr != map_ptr);
                            !entries.is_empty()
                        });
                        return Ok(DispatchAction::Continue);
                    }
                    // Non-writable: TypeError in strict mode, silently ignore in sloppy.
                    if ctx.frame.bytecode_array.is_strict() {
                        return Err(StatorError::TypeError(format!(
                            "Cannot assign to read only property '{prop_name}'"
                        )));
                    }
                    return Ok(DispatchAction::Continue);
                }
            }
            // Check for setter accessor first.
            let setter_key = format!("__set_{prop_name}__");
            let setter = map.borrow().get(&setter_key).cloned();
            if let Some(setter_fn) = setter {
                dispatch_setter(&setter_fn, &obj, val)?;
                return Ok(DispatchAction::Continue);
            }
            let pm = map.borrow();
            // Existing non-writable property: TypeError in strict mode, silently ignore in sloppy.
            if pm.contains_key(prop_name.as_ref()) && !pm.is_writable(prop_name.as_ref()) {
                if ctx.frame.bytecode_array.is_strict() {
                    return Err(StatorError::TypeError(format!(
                        "Cannot assign to read only property '{prop_name}'"
                    )));
                }
                return Ok(DispatchAction::Continue);
            }
            drop(pm);
            // Non-extensible: TypeError for new property additions in strict mode.
            {
                let pm = map.borrow();
                if !pm.extensible && !pm.contains_key(prop_name.as_ref()) {
                    if ctx.frame.bytecode_array.is_strict() {
                        return Err(StatorError::TypeError(format!(
                            "Cannot add property {prop_name}, object is not extensible"
                        )));
                    }
                    return Ok(DispatchAction::Continue);
                }
            }
            map.borrow_mut().insert(prop_name.to_string(), val);
            // Populate shape store IC for future fast-path stores.
            if slot != u32::MAX {
                let pm = map.borrow();
                if let Some(offset) = pm.offset_of(&prop_name) {
                    ctx.frame.shape_store_ic.insert(
                        slot,
                        PropertyIc {
                            cached_shape: pm.shape_id(),
                            cached_offset: offset,
                        },
                    );
                }
            }
            // Invalidate value-based caches for this object.
            let map_ptr = Rc::as_ptr(map) as usize;
            ctx.frame
                .mono_load_cache
                .retain(|_, (ptr, _)| *ptr != map_ptr);
            ctx.frame.poly_load_cache.retain(|_, entries| {
                entries.retain(|(ptr, _)| *ptr != map_ptr);
                !entries.is_empty()
            });
        }
        JsValue::Function(ref ba) => {
            fn_props_set(ba, prop_name.to_string(), val);
        }
        JsValue::Array(ref arr) => {
            if prop_name.as_ref() == "length" {
                let new_len = val.to_number()?;
                let new_len_u32 = new_len as u32;
                if (new_len_u32 as f64) != new_len || new_len < 0.0 || !new_len.is_finite() {
                    return Err(StatorError::RangeError("Invalid array length".to_string()));
                }
                let mut v = arr.borrow_mut();
                let current_len = v.len();
                if (new_len_u32 as usize) < current_len {
                    v.truncate(new_len_u32 as usize);
                } else {
                    v.resize(new_len_u32 as usize, JsValue::Undefined);
                }
            }
        }
        JsValue::Error(ref e) => {
            e.props.borrow_mut().insert(prop_name.to_string(), val);
        }
        _ => {}
    }
    // Accumulator stays unchanged: the assignment's completion
    // value is the stored value (already in the accumulator).
    Ok(DispatchAction::Continue)
}

fn handle_lda_keyed_property(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("LdaKeyedProperty", 0));
    };
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    let key = ctx.frame.accumulator.clone();
    // TypeError for property access on null or undefined (ES §13.10.3).
    if matches!(obj, JsValue::Null | JsValue::Undefined) {
        let key_str = key.to_js_string().unwrap_or_default();
        return Err(StatorError::TypeError(format!(
            "Cannot read properties of {} (reading '{key_str}')",
            if matches!(obj, JsValue::Null) {
                "null"
            } else {
                "undefined"
            }
        )));
    }
    ctx.frame.accumulator = keyed_load(&obj, &key)?;
    Ok(DispatchAction::Continue)
}

fn handle_sta_keyed_property(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("StaKeyedProperty", 0));
    };
    let Operand::Register(key_v) = instr.operands[1] else {
        return Err(err_bad_operand("StaKeyedProperty", 1));
    };
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    let key = ctx.frame.read_reg(key_v)?.clone();
    let val = ctx.frame.accumulator.clone();
    // TypeError for property store on null or undefined.
    if matches!(obj, JsValue::Null | JsValue::Undefined) {
        let key_str = key.to_js_string().unwrap_or_default();
        return Err(StatorError::TypeError(format!(
            "Cannot set properties of {} (setting '{key_str}')",
            if matches!(obj, JsValue::Null) {
                "null"
            } else {
                "undefined"
            }
        )));
    }
    // PlainObject: check for setter accessor, writable, and extensibility
    // before falling through to keyed_store.
    if let JsValue::PlainObject(ref map) = obj {
        let key_str = to_property_key(&key)?;
        // Check for setter accessor first (__set_<key>__).
        let setter_key = format!("__set_{key_str}__");
        let setter = map.borrow().get(&setter_key).cloned();
        if let Some(setter_fn) = setter {
            dispatch_setter(&setter_fn, &obj, val)?;
            return Ok(DispatchAction::Continue);
        }
        let pm = map.borrow();
        // Existing non-writable property: TypeError in strict mode.
        if pm.contains_key(&key_str) && !pm.is_writable(&key_str) {
            if ctx.frame.bytecode_array.is_strict() {
                return Err(StatorError::TypeError(format!(
                    "Cannot assign to read only property '{key_str}'"
                )));
            }
            return Ok(DispatchAction::Continue);
        }
        // Non-extensible object: TypeError for new property in strict mode.
        if !pm.extensible && !pm.contains_key(&key_str) {
            if ctx.frame.bytecode_array.is_strict() {
                return Err(StatorError::TypeError(format!(
                    "Cannot add property {key_str}, object is not extensible"
                )));
            }
            return Ok(DispatchAction::Continue);
        }
        drop(pm);
    }
    keyed_store(&obj, &key, val)?;
    // Accumulator stays unchanged.
    Ok(DispatchAction::Continue)
}

fn handle_get_iterator(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(iter_v) = instr.operands[0] else {
        return Err(err_bad_operand("GetIterator", 0));
    };
    let iterable = ctx.frame.read_reg(iter_v)?.clone();
    ctx.frame.accumulator = match iterable {
        JsValue::Array(items) => {
            let items_vec: Vec<JsValue> = items.borrow().clone();
            JsValue::Iterator(NativeIterator::from_items(items_vec))
        }
        JsValue::String(ref s) => JsValue::Iterator(NativeIterator::from_string(s)),
        // Generators and existing iterators pass through unchanged.
        JsValue::Generator(_) | JsValue::Iterator(_) => iterable,
        // PlainObject with @@iterator or Symbol.iterator → call it.
        JsValue::PlainObject(ref map) => {
            let borrow = map.borrow();
            // Check for both internal @@iterator and user-visible Symbol(SYMBOL_ITERATOR)
            let iter_fn = borrow.get("@@iterator").cloned().or_else(|| {
                let sym_key = format!("Symbol({})", crate::builtins::symbol::SYMBOL_ITERATOR);
                borrow.get(&sym_key).cloned()
            });
            drop(borrow);
            match iter_fn {
                Some(ref f @ (JsValue::NativeFunction(_) | JsValue::Function(_))) => {
                    dispatch_call_with_this(f, iterable.clone(), vec![])?
                }
                Some(JsValue::PlainObject(ref call_obj))
                    if call_obj.borrow().contains_key("__call__") =>
                {
                    dispatch_call_with_this(
                        &JsValue::PlainObject(call_obj.clone()),
                        iterable.clone(),
                        vec![],
                    )?
                }
                Some(_) => {
                    return Err(StatorError::TypeError(
                        "GetIterator: @@iterator is not a function".into(),
                    ));
                }
                None => {
                    // Fallback: PlainObject with "length" → array-like
                    if map.borrow().contains_key("length") {
                        let items = plain_object_to_array_items(map);
                        JsValue::Iterator(NativeIterator::from_items(items))
                    } else {
                        return Err(StatorError::TypeError(format!(
                            "GetIterator: value is not iterable (got {iterable:?})"
                        )));
                    }
                }
            }
        }
        other => {
            return Err(StatorError::TypeError(format!(
                "GetIterator: value is not iterable (got {other:?})"
            )));
        }
    };
    Ok(DispatchAction::Continue)
}

fn handle_get_async_iterator(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(iter_v) = instr.operands[0] else {
        return Err(err_bad_operand("GetAsyncIterator", 0));
    };
    let iterable = ctx.frame.read_reg(iter_v)?.clone();
    ctx.frame.accumulator = match iterable {
        JsValue::Array(items) => {
            let items_vec: Vec<JsValue> = items.borrow().clone();
            JsValue::Iterator(NativeIterator::from_items(items_vec))
        }
        JsValue::String(ref s) => JsValue::Iterator(NativeIterator::from_string(s)),
        JsValue::Generator(_) | JsValue::Iterator(_) => iterable,
        // PlainObject with @@asyncIterator → call it first (§27.1.4.2).
        JsValue::PlainObject(ref map) if map.borrow().contains_key("@@asyncIterator") => {
            let iter_fn = map.borrow().get("@@asyncIterator").cloned();
            match iter_fn {
                Some(ref f @ (JsValue::NativeFunction(_) | JsValue::Function(_))) => {
                    dispatch_call_with_this(f, iterable.clone(), vec![])?
                }
                _ => {
                    return Err(StatorError::TypeError(
                        "GetAsyncIterator: @@asyncIterator is not a function".into(),
                    ));
                }
            }
        }
        // Fall back to @@iterator (sync iterator wrapped for async).
        JsValue::PlainObject(ref map) if map.borrow().contains_key("@@iterator") => {
            let iter_fn = map.borrow().get("@@iterator").cloned();
            match iter_fn {
                Some(ref f @ (JsValue::NativeFunction(_) | JsValue::Function(_))) => {
                    dispatch_call_with_this(f, iterable.clone(), vec![])?
                }
                _ => {
                    return Err(StatorError::TypeError(
                        "GetAsyncIterator: @@iterator is not a function".into(),
                    ));
                }
            }
        }
        JsValue::PlainObject(ref map) if map.borrow().contains_key("length") => {
            let items = plain_object_to_array_items(map);
            JsValue::Iterator(NativeIterator::from_items(items))
        }
        other => {
            return Err(StatorError::TypeError(format!(
                "GetAsyncIterator: value is not iterable (got {other:?})"
            )));
        }
    };
    Ok(DispatchAction::Continue)
}

fn handle_iterator_next(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(iter_v) = instr.operands[0] else {
        return Err(err_bad_operand("IteratorNext", 0));
    };
    let Operand::Register(value_out_v) = instr.operands[1] else {
        return Err(err_bad_operand("IteratorNext", 1));
    };
    let iter = ctx.frame.read_reg(iter_v)?.clone();
    let (value, done) = match iter {
        JsValue::Iterator(ni) => match ni.borrow_mut().next_item() {
            Some(v) => (v, false),
            None => (JsValue::Undefined, true),
        },
        JsValue::Generator(ref gs) => {
            match Interpreter::run_generator_step(gs, JsValue::Undefined)? {
                GeneratorStep::Yield(v) => (v, false),
                GeneratorStep::Return(v) => (v, true),
            }
        }
        JsValue::PlainObject(ref map) if map.borrow().contains_key("next") => {
            let next_fn = map.borrow().get("next").cloned();
            match next_fn {
                Some(ref f @ (JsValue::NativeFunction(_) | JsValue::Function(_))) => {
                    let result = dispatch_call_with_this(f, iter.clone(), vec![])?;
                    match result {
                        JsValue::PlainObject(ref res_map) => {
                            let done = res_map.borrow().get("done").is_some_and(|d| d.to_boolean());
                            let value = res_map
                                .borrow()
                                .get("value")
                                .cloned()
                                .unwrap_or(JsValue::Undefined);
                            (value, done)
                        }
                        _ => (JsValue::Undefined, true),
                    }
                }
                _ => {
                    return Err(StatorError::TypeError(
                        "IteratorNext: next is not a function".into(),
                    ));
                }
            }
        }
        other => {
            return Err(StatorError::TypeError(format!(
                "IteratorNext: value is not an iterator (got {other:?})"
            )));
        }
    };
    ctx.frame.write_reg(value_out_v, value)?;
    ctx.frame.accumulator = JsValue::Boolean(done);
    Ok(DispatchAction::Continue)
}

/// `CopyDataProperties <target_reg> <source_reg>`
///
/// Copies all own enumerable properties from the source object to the target.
fn handle_copy_data_properties(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(target_v) = instr.operands[0] else {
        return Err(err_bad_operand("CopyDataProperties", 0));
    };
    let Operand::Register(source_v) = instr.operands[1] else {
        return Err(err_bad_operand("CopyDataProperties", 1));
    };
    let target = ctx.frame.read_reg(target_v)?.clone();
    let source = ctx.frame.read_reg(source_v)?.clone();

    if let (JsValue::PlainObject(t), JsValue::PlainObject(s)) = (&target, &source) {
        let entries: Vec<(String, JsValue)> = s
            .borrow()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        for (k, v) in entries {
            t.borrow_mut().insert(k, v);
        }
    }
    Ok(DispatchAction::Continue)
}

fn handle_suspend_generator(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let yield_val = ctx.frame.accumulator.clone();

    // Save state into the attached GeneratorState (if any).
    if let Some(gs_rc) = ctx.frame.generator_state.as_ref() {
        let mut gs = gs_rc.borrow_mut();
        // Save the full register file so that ResumeGenerator
        // can restore it on the next step.
        gs.registers.clone_from(&ctx.frame.registers);
        // ctx.frame.pc was already advanced past this instruction.
        gs.resume_pc = ctx.frame.pc;
        gs.status = GeneratorStatus::SuspendedAtYield;
    }

    ctx.frame.suspend_result = Some(yield_val.clone());
    Ok(DispatchAction::Return(yield_val))
}

fn handle_resume_generator(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let mut throw_val: Option<JsValue> = None;
    if let Some(gs_rc) = ctx.frame.generator_state.as_ref() {
        let mut gs = gs_rc.borrow_mut();
        // Restore the saved registers into the ctx.frame.
        // The saved register file has the same length as the frame
        // (set by SuspendGenerator from ctx.frame.registers.clone()), so
        // the min() only protects against a fresh generator where
        // gs.registers is empty (resume_pc == 0).
        let count = gs.registers.len().min(ctx.frame.registers.len());
        ctx.frame.registers[..count].clone_from_slice(&gs.registers[..count]);
        gs.status = GeneratorStatus::Executing;
        // Check if we need to throw at the yield point.
        if let GeneratorResumeMode::Throw(val) =
            std::mem::replace(&mut gs.resume_mode, GeneratorResumeMode::Normal)
        {
            throw_val = Some(val);
        }
    }
    if let Some(val) = throw_val {
        // Trigger exception propagation — the interpreter's error handling
        // will consult the handler table and jump to any active catch block.
        ctx.frame.accumulator = val.clone();
        set_pending_exception(val.clone());
        let msg = error_message_from_value(&val);
        return Err(StatorError::JsException(msg));
    }
    // Accumulator keeps the resume value supplied by run_generator_step.
    Ok(DispatchAction::Continue)
}

fn handle_get_generator_state(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    ctx.frame.accumulator = if let Some(gs_rc) = ctx.frame.generator_state.as_ref() {
        JsValue::Smi(gs_rc.borrow().status.to_smi())
    } else {
        JsValue::Smi(GeneratorStatus::Completed.to_smi())
    };
    Ok(DispatchAction::Continue)
}

fn handle_set_generator_state(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    if let Some(gs_rc) = ctx.frame.generator_state.as_ref()
        && let JsValue::Smi(n) = ctx.frame.accumulator
    {
        gs_rc.borrow_mut().status = GeneratorStatus::from_smi(n);
    }
    Ok(DispatchAction::Continue)
}

fn handle_switch_on_generator_state(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    if let Some(gs_rc) = ctx.frame.generator_state.as_ref() {
        let resume_pc = gs_rc.borrow().resume_pc;
        if resume_pc > 0 {
            ctx.frame.pc = resume_pc;
        }
    }
    Ok(DispatchAction::Continue)
}

fn handle_debugger(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let stmt_offset = ctx.byte_offsets[ctx.frame.pc - 1] as u32;
    if let Some(pause_err) = ACTIVE_DEBUGGER.with(|d| {
        let opt = d.borrow();
        opt.as_ref()
            .map(|rc| rc.borrow_mut().on_debugger_statement(stmt_offset))
    }) {
        return Err(pause_err);
    }
    // No debugger attached: debugger; is a no-op.
    Ok(DispatchAction::Continue)
}

fn handle_type_of(ctx: &mut DispatchContext, _instr: &Instruction) -> StatorResult<DispatchAction> {
    let type_str = match &ctx.frame.accumulator {
        JsValue::Undefined | JsValue::TheHole => "undefined",
        JsValue::Null => "object",
        JsValue::Boolean(_) => "boolean",
        JsValue::Smi(_) | JsValue::HeapNumber(_) => "number",
        JsValue::String(_) => "string",
        JsValue::Symbol(_) => "symbol",
        JsValue::BigInt(_) => "bigint",
        JsValue::Function(_) | JsValue::NativeFunction(_) => "function",
        JsValue::Object(_) | JsValue::Array(_) | JsValue::Error(_) => "object",
        JsValue::PlainObject(map) => {
            if map.borrow().get("__call__").is_some() {
                "function"
            } else {
                "object"
            }
        }
        JsValue::Generator(_) => "object",
        JsValue::Iterator(_) => "object",
        JsValue::Promise(_) => "object",
        JsValue::Context(_) => "object",
        JsValue::Proxy(p) => {
            let proxy = p.borrow();
            if proxy.is_callable() {
                "function"
            } else {
                "object"
            }
        }
        JsValue::ArrayBuffer(_) | JsValue::TypedArray(_) | JsValue::DataView(_) => "object",
    };
    ctx.frame.accumulator = JsValue::String(type_str.to_owned().into());
    Ok(DispatchAction::Continue)
}

fn handle_test_type_of(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Flag(flag) = instr.operands[0] else {
        return Err(err_bad_operand("TestTypeOf", 0));
    };
    let matches_type = match flag {
        0 => matches!(
            ctx.frame.accumulator,
            JsValue::Smi(_) | JsValue::HeapNumber(_)
        ),
        1 => matches!(ctx.frame.accumulator, JsValue::String(_)),
        2 => matches!(ctx.frame.accumulator, JsValue::Symbol(_)),
        3 => matches!(ctx.frame.accumulator, JsValue::Boolean(_)),
        4 => matches!(ctx.frame.accumulator, JsValue::BigInt(_)),
        5 => matches!(ctx.frame.accumulator, JsValue::Undefined),
        6 => match &ctx.frame.accumulator {
            JsValue::Function(_) | JsValue::NativeFunction(_) => true,
            JsValue::PlainObject(map) => map.borrow().get("__call__").is_some(),
            JsValue::Proxy(p) => p.borrow().is_callable(),
            _ => false,
        },
        7 => match &ctx.frame.accumulator {
            JsValue::Null
            | JsValue::Object(_)
            | JsValue::Array(_)
            | JsValue::Error(_)
            | JsValue::Generator(_)
            | JsValue::Iterator(_)
            | JsValue::Promise(_)
            | JsValue::ArrayBuffer(_)
            | JsValue::TypedArray(_)
            | JsValue::DataView(_)
            | JsValue::Context(_) => true,
            JsValue::PlainObject(map) => map.borrow().get("__call__").is_none(),
            JsValue::Proxy(p) => !p.borrow().is_callable(),
            _ => false,
        },
        _ => false,
    };
    ctx.frame.accumulator = JsValue::Boolean(matches_type);
    Ok(DispatchAction::Continue)
}

fn handle_to_number(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    // operands[0] is a FeedbackSlot, ignored at runtime.
    let n = ctx.frame.accumulator.to_number()?;
    ctx.frame.accumulator = number_to_jsvalue(n);
    Ok(DispatchAction::Continue)
}

fn handle_to_string(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let s = ctx.frame.accumulator.to_js_string()?;
    ctx.frame.accumulator = JsValue::String(s.into());
    Ok(DispatchAction::Continue)
}

fn handle_to_boolean(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    // operands[0] is a FeedbackSlot, ignored at runtime.
    let b = ctx.frame.accumulator.to_boolean();
    ctx.frame.accumulator = JsValue::Boolean(b);
    Ok(DispatchAction::Continue)
}

fn handle_to_object(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    // operands[0] is a Register destination.
    let Operand::Register(dst) = instr.operands[0] else {
        return Err(err_bad_operand("ToObject", 0));
    };
    let wrapped = match &ctx.frame.accumulator {
        JsValue::Null | JsValue::Undefined | JsValue::TheHole => {
            return Err(StatorError::TypeError(
                "Cannot convert undefined or null to object".to_string(),
            ));
        }
        // Objects, arrays, and other reference types stay as-is.
        JsValue::PlainObject(_)
        | JsValue::Array(_)
        | JsValue::Function(_)
        | JsValue::NativeFunction(_)
        | JsValue::Promise(_)
        | JsValue::Generator(_)
        | JsValue::Error(_)
        | JsValue::Proxy(_)
        | JsValue::Object(_)
        | JsValue::Context(_)
        | JsValue::Iterator(_)
        | JsValue::ArrayBuffer(_)
        | JsValue::TypedArray(_)
        | JsValue::DataView(_) => ctx.frame.accumulator.clone(),
        // ECMAScript §7.1.18 – Boolean wrapper object.
        JsValue::Boolean(b) => {
            let b_val = *b;
            let mut map = PropertyMap::new();
            map.insert("__wrapped__".into(), JsValue::Boolean(b_val));
            map.insert(
                "valueOf".into(),
                JsValue::NativeFunction(Rc::new(move |_| Ok(JsValue::Boolean(b_val)))),
            );
            map.insert(
                "toString".into(),
                JsValue::NativeFunction(Rc::new(move |_| {
                    Ok(JsValue::String(if b_val { "true" } else { "false" }.into()))
                })),
            );
            JsValue::PlainObject(Rc::new(RefCell::new(map)))
        }
        // ECMAScript §7.1.18 – Number wrapper object (Smi).
        JsValue::Smi(n) => {
            let n_val = *n;
            let mut map = PropertyMap::new();
            map.insert("__wrapped__".into(), JsValue::Smi(n_val));
            map.insert(
                "valueOf".into(),
                JsValue::NativeFunction(Rc::new(move |_| Ok(JsValue::Smi(n_val)))),
            );
            map.insert(
                "toString".into(),
                JsValue::NativeFunction(Rc::new(move |_| {
                    Ok(JsValue::String(n_val.to_string().into()))
                })),
            );
            JsValue::PlainObject(Rc::new(RefCell::new(map)))
        }
        // ECMAScript §7.1.18 – Number wrapper object (HeapNumber).
        JsValue::HeapNumber(n) => {
            let n_val = *n;
            let mut map = PropertyMap::new();
            map.insert("__wrapped__".into(), JsValue::HeapNumber(n_val));
            map.insert(
                "valueOf".into(),
                JsValue::NativeFunction(Rc::new(move |_| Ok(JsValue::HeapNumber(n_val)))),
            );
            map.insert(
                "toString".into(),
                JsValue::NativeFunction(Rc::new(move |_| {
                    Ok(JsValue::String(n_val.to_string().into()))
                })),
            );
            JsValue::PlainObject(Rc::new(RefCell::new(map)))
        }
        // ECMAScript §7.1.18 – String wrapper object with length and indexed access.
        JsValue::String(s) => {
            let s_val = s.clone();
            let mut map = PropertyMap::new();
            map.insert("__wrapped__".into(), JsValue::String(s_val.clone()));
            map.insert("length".into(), JsValue::Smi(s_val.len() as i32));
            // Indexed character access ("0", "1", …).
            for (i, ch) in s_val.chars().enumerate() {
                map.insert(i.to_string(), JsValue::String(ch.to_string().into()));
            }
            let s_vo = s_val.clone();
            map.insert(
                "valueOf".into(),
                JsValue::NativeFunction(Rc::new(move |_| Ok(JsValue::String(s_vo.clone())))),
            );
            let s_ts = s_val;
            map.insert(
                "toString".into(),
                JsValue::NativeFunction(Rc::new(move |_| Ok(JsValue::String(s_ts.clone())))),
            );
            JsValue::PlainObject(Rc::new(RefCell::new(map)))
        }
        // ECMAScript §7.1.18 – Symbol wrapper object.
        JsValue::Symbol(sym) => {
            let sym_val = *sym;
            let mut map = PropertyMap::new();
            map.insert("__wrapped__".into(), JsValue::Symbol(sym_val));
            map.insert(
                "valueOf".into(),
                JsValue::NativeFunction(Rc::new(move |_| Ok(JsValue::Symbol(sym_val)))),
            );
            map.insert(
                "toString".into(),
                JsValue::NativeFunction(Rc::new(move |_| {
                    Ok(JsValue::String(format!("Symbol({})", sym_val).into()))
                })),
            );
            JsValue::PlainObject(Rc::new(RefCell::new(map)))
        }
        // ECMAScript §7.1.18 – BigInt wrapper object.
        JsValue::BigInt(n) => {
            let n_val = *n;
            let mut map = PropertyMap::new();
            map.insert("__wrapped__".into(), JsValue::BigInt(n_val));
            map.insert(
                "valueOf".into(),
                JsValue::NativeFunction(Rc::new(move |_| Ok(JsValue::BigInt(n_val)))),
            );
            map.insert(
                "toString".into(),
                JsValue::NativeFunction(Rc::new(move |_| {
                    Ok(JsValue::String(n_val.to_string().into()))
                })),
            );
            JsValue::PlainObject(Rc::new(RefCell::new(map)))
        }
    };
    ctx.frame.write_reg(dst, wrapped)?;
    Ok(DispatchAction::Continue)
}

fn handle_to_name(ctx: &mut DispatchContext, instr: &Instruction) -> StatorResult<DispatchAction> {
    // operands[0] is a Register destination.
    // Convert accumulator to a property key (string or symbol).
    let Operand::Register(dst) = instr.operands[0] else {
        return Err(err_bad_operand("ToName", 0));
    };
    let key = match &ctx.frame.accumulator {
        JsValue::String(_) | JsValue::Symbol(_) => ctx.frame.accumulator.clone(),
        other => JsValue::String(other.to_js_string()?.into()),
    };
    ctx.frame.write_reg(dst, key)?;
    Ok(DispatchAction::Continue)
}

fn handle_negate(ctx: &mut DispatchContext, _instr: &Instruction) -> StatorResult<DispatchAction> {
    // operands[0] is a FeedbackSlot, ignored at runtime.
    if let JsValue::BigInt(n) = &ctx.frame.accumulator {
        ctx.frame.accumulator = JsValue::BigInt(n.wrapping_neg());
    } else {
        let n = ctx.frame.accumulator.to_number()?;
        ctx.frame.accumulator = number_to_jsvalue(-n);
    }
    Ok(DispatchAction::Continue)
}

fn handle_bitwise_not(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    // operands[0] is a FeedbackSlot, ignored at runtime.
    if let JsValue::BigInt(n) = &ctx.frame.accumulator {
        ctx.frame.accumulator = JsValue::BigInt(!n);
    } else {
        let n = ctx.frame.accumulator.to_number()? as i32;
        ctx.frame.accumulator = JsValue::Smi(!n);
    }
    Ok(DispatchAction::Continue)
}

fn handle_create_empty_object_literal(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    ctx.frame.accumulator = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
    Ok(DispatchAction::Continue)
}

fn handle_create_empty_array_literal(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    // operands[0] is a FeedbackSlot, ignored at runtime.
    let mut map = PropertyMap::new();
    map.insert("length".to_string(), JsValue::Smi(0));
    map.insert("__is_array__".to_string(), JsValue::Boolean(true));
    ctx.frame.accumulator = JsValue::PlainObject(Rc::new(RefCell::new(map)));
    Ok(DispatchAction::Continue)
}

fn handle_create_array_literal(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    // operands: [ConstantPoolIdx, FeedbackSlot, Flag]
    let mut map = PropertyMap::new();
    map.insert("length".to_string(), JsValue::Smi(0));
    map.insert("__is_array__".to_string(), JsValue::Boolean(true));
    ctx.frame.accumulator = JsValue::PlainObject(Rc::new(RefCell::new(map)));
    Ok(DispatchAction::Continue)
}

fn handle_create_array_from_iterable(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let iterable = ctx.frame.accumulator.clone();
    let items: Vec<JsValue> = match &iterable {
        JsValue::Array(arr) => arr.borrow().clone(),
        JsValue::Iterator(iter) => {
            let mut out = Vec::new();
            loop {
                let mut it = iter.borrow_mut();
                match it.next_item() {
                    Some(v) => out.push(v),
                    None => break,
                }
            }
            out
        }
        JsValue::String(s) => s
            .chars()
            .map(|c| JsValue::String(c.to_string().into()))
            .collect(),
        JsValue::Generator(gs) => {
            let mut out = Vec::new();
            loop {
                match Interpreter::run_generator_step(gs, JsValue::Undefined)? {
                    GeneratorStep::Yield(v) => out.push(v),
                    GeneratorStep::Return(v) => {
                        if !matches!(v, JsValue::Undefined) {
                            out.push(v);
                        }
                        break;
                    }
                }
            }
            out
        }
        JsValue::PlainObject(map) if map.borrow().contains_key("next") => {
            let mut out = Vec::new();
            loop {
                let next_fn = map.borrow().get("next").cloned();
                match next_fn {
                    Some(ref f @ (JsValue::NativeFunction(_) | JsValue::Function(_))) => {
                        let result = dispatch_call_with_this(f, iterable.clone(), vec![])?;
                        match result {
                            JsValue::PlainObject(ref res_map) => {
                                let done =
                                    res_map.borrow().get("done").is_some_and(|d| d.to_boolean());
                                if done {
                                    break;
                                }
                                let value = res_map
                                    .borrow()
                                    .get("value")
                                    .cloned()
                                    .unwrap_or(JsValue::Undefined);
                                out.push(value);
                            }
                            _ => break,
                        }
                    }
                    _ => break,
                }
            }
            out
        }
        _ => vec![],
    };
    let mut map = PropertyMap::new();
    for (i, v) in items.iter().enumerate() {
        map.insert(i.to_string(), v.clone());
    }
    map.insert("length".to_string(), JsValue::Smi(items.len() as i32));
    ctx.frame.accumulator = JsValue::PlainObject(Rc::new(RefCell::new(map)));
    Ok(DispatchAction::Continue)
}

fn handle_create_object_literal(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    // operands: [ConstantPoolIdx, FeedbackSlot, Flag]
    ctx.frame.accumulator = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
    Ok(DispatchAction::Continue)
}

fn handle_create_reg_exp_literal(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(pattern_idx) = instr.operands[0] else {
        return Err(err_bad_operand("CreateRegExpLiteral", 0));
    };
    // operands[1] = FeedbackSlot (ignored)
    let Operand::Flag(flags_val) = instr.operands[2] else {
        return Err(err_bad_operand("CreateRegExpLiteral", 2));
    };
    let pattern = match ctx.frame.bytecode_array.get_constant(pattern_idx) {
        Some(ConstantPoolEntry::String(s)) => decode_string_constant(s),
        _ => String::new(),
    };
    // Decode flag bits back to a flag string.
    let mut flags_str = String::new();
    if flags_val & 0x01 != 0 {
        flags_str.push('g');
    }
    if flags_val & 0x02 != 0 {
        flags_str.push('i');
    }
    if flags_val & 0x04 != 0 {
        flags_str.push('m');
    }
    if flags_val & 0x08 != 0 {
        flags_str.push('s');
    }
    if flags_val & 0x10 != 0 {
        flags_str.push('u');
    }
    if flags_val & 0x20 != 0 {
        flags_str.push('y');
    }
    // Build a proper RegExp object backed by JsRegExp.
    let re = crate::objects::regexp::JsRegExp::new(&pattern, &flags_str)?;
    let re_rc = Rc::new(RefCell::new(re));
    let mut map = PropertyMap::new();
    map.insert("__is_regexp__".to_string(), JsValue::Boolean(true));
    map.insert(
        "source".to_string(),
        JsValue::String(pattern.clone().into()),
    );
    map.insert(
        "flags".to_string(),
        JsValue::String(flags_str.clone().into()),
    );
    map.insert(
        "global".to_string(),
        JsValue::Boolean(flags_str.contains('g')),
    );
    map.insert(
        "ignoreCase".to_string(),
        JsValue::Boolean(flags_str.contains('i')),
    );
    map.insert(
        "multiline".to_string(),
        JsValue::Boolean(flags_str.contains('m')),
    );
    map.insert(
        "dotAll".to_string(),
        JsValue::Boolean(flags_str.contains('s')),
    );
    map.insert(
        "unicode".to_string(),
        JsValue::Boolean(flags_str.contains('u')),
    );
    map.insert(
        "sticky".to_string(),
        JsValue::Boolean(flags_str.contains('y')),
    );
    // test(str)
    {
        let r = Rc::clone(&re_rc);
        map.insert(
            "test".to_string(),
            JsValue::NativeFunction(Rc::new(move |args| {
                let input = match args.first() {
                    Some(v) => v.to_js_string()?,
                    None => String::new(),
                };
                Ok(JsValue::Boolean(r.borrow().test(&input)))
            })),
        );
    }
    // exec(str)
    {
        let r = Rc::clone(&re_rc);
        map.insert(
            "exec".to_string(),
            JsValue::NativeFunction(Rc::new(move |args| {
                let input = match args.first() {
                    Some(v) => v.to_js_string()?,
                    None => String::new(),
                };
                match r.borrow().exec(&input) {
                    Some(m) => {
                        let mut props = PropertyMap::new();
                        props.insert("0".to_string(), JsValue::String(m.matched.clone().into()));
                        for (i, g) in m.captures.iter().enumerate() {
                            props.insert(
                                (i + 1).to_string(),
                                match g {
                                    Some(s) => JsValue::String(s.clone().into()),
                                    None => JsValue::Undefined,
                                },
                            );
                        }
                        props.insert(
                            "length".to_string(),
                            JsValue::Smi((1 + m.captures.len()) as i32),
                        );
                        props.insert("index".to_string(), JsValue::Smi(m.index as i32));
                        props.insert("input".to_string(), JsValue::String(m.input.clone().into()));
                        if m.named_groups.is_empty() {
                            props.insert("groups".to_string(), JsValue::Undefined);
                        } else {
                            let mut groups = PropertyMap::new();
                            for (k, v) in &m.named_groups {
                                groups.insert(k.clone(), JsValue::String(v.clone().into()));
                            }
                            props.insert(
                                "groups".to_string(),
                                JsValue::PlainObject(Rc::new(RefCell::new(groups))),
                            );
                        }
                        props.insert("__is_array__".to_string(), JsValue::Boolean(true));
                        Ok(JsValue::PlainObject(Rc::new(RefCell::new(props))))
                    }
                    None => Ok(JsValue::Null),
                }
            })),
        );
    }
    // toString()
    {
        let p = pattern.clone();
        let f = flags_str.clone();
        map.insert(
            "toString".to_string(),
            JsValue::NativeFunction(Rc::new(move |_args| {
                Ok(JsValue::String(format!("/{p}/{f}").into()))
            })),
        );
    }
    ctx.frame.accumulator = JsValue::PlainObject(Rc::new(RefCell::new(map)));
    Ok(DispatchAction::Continue)
}

fn handle_sta_in_array_literal(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(arr_v) = instr.operands[0] else {
        return Err(err_bad_operand("StaInArrayLiteral", 0));
    };
    let Operand::Register(idx_v) = instr.operands[1] else {
        return Err(err_bad_operand("StaInArrayLiteral", 1));
    };
    // operands[2] is a FeedbackSlot, ignored at runtime.
    let arr = ctx.frame.read_reg(arr_v)?.clone();
    let key = ctx.frame.read_reg(idx_v)?.clone();
    let val = ctx.frame.accumulator.clone();
    if let JsValue::PlainObject(ref map) = arr {
        let idx_str = to_property_key(&key)?;
        map.borrow_mut().insert(idx_str, val);
        // Update length: max(current_length, index + 1).
        if let Some(idx) = to_array_index(&key) {
            let new_len = (idx + 1) as i32;
            let cur_len = match map.borrow().get("length") {
                Some(JsValue::Smi(n)) => *n,
                _ => 0,
            };
            if new_len > cur_len {
                map.borrow_mut()
                    .insert("length".to_string(), JsValue::Smi(new_len));
            }
        }
    }
    // Accumulator stays unchanged.
    Ok(DispatchAction::Continue)
}

fn handle_define_named_own_property(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("DefineNamedOwnProperty", 0));
    };
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[1] else {
        return Err(err_bad_operand("DefineNamedOwnProperty", 1));
    };
    // operands[2] is a FeedbackSlot, ignored at runtime.
    let prop_name = match ctx.frame.bytecode_array.get_constant(name_idx) {
        Some(ConstantPoolEntry::String(s)) => s.clone(),
        _ => {
            return Err(StatorError::Internal(
                "DefineNamedOwnProperty: property name is not a string".into(),
            ));
        }
    };
    let val = ctx.frame.accumulator.clone();
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    if let JsValue::PlainObject(ref map) = obj {
        map.borrow_mut().insert(prop_name, val);
    }
    // Accumulator stays unchanged.
    Ok(DispatchAction::Continue)
}

fn handle_define_keyed_own_property(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("DefineKeyedOwnProperty", 0));
    };
    let Operand::Register(key_v) = instr.operands[1] else {
        return Err(err_bad_operand("DefineKeyedOwnProperty", 1));
    };
    // operands[2] = Flag (ignored), operands[3] = FeedbackSlot (ignored).
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    let key = ctx.frame.read_reg(key_v)?.clone();
    let val = ctx.frame.accumulator.clone();
    if let JsValue::PlainObject(ref map) = obj {
        let prop_name = to_property_key(&key)?;
        map.borrow_mut().insert(prop_name, val);
    }
    // Accumulator stays unchanged.
    Ok(DispatchAction::Continue)
}

fn handle_define_keyed_own_property_in_literal(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("DefineKeyedOwnPropertyInLiteral", 0));
    };
    let Operand::Register(key_v) = instr.operands[1] else {
        return Err(err_bad_operand("DefineKeyedOwnPropertyInLiteral", 1));
    };
    // operands[2] = Flag (ignored), operands[3] = FeedbackSlot (ignored).
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    let key = ctx.frame.read_reg(key_v)?.clone();
    let val = ctx.frame.accumulator.clone();
    if let JsValue::PlainObject(ref map) = obj {
        let prop_name = to_property_key(&key)?;
        map.borrow_mut().insert(prop_name, val);
    }
    // Accumulator stays unchanged.
    Ok(DispatchAction::Continue)
}

fn handle_test_instance_of(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("TestInstanceOf", 0));
    };
    let constructor = ctx.frame.read_reg(v)?.clone();

    // §7.3.21 OrdinaryHasInstance — first check @@hasInstance
    if let JsValue::PlainObject(map) = &constructor
        && let Some(JsValue::NativeFunction(f)) = map.borrow().get("@@hasInstance").cloned()
    {
        let result = f(vec![ctx.frame.accumulator.clone()])?;
        ctx.frame.accumulator = JsValue::Boolean(result.to_boolean());
        return Ok(DispatchAction::Continue);
    }

    // ── Built-in type checks via constructor identity ──────────────
    // NativeFunction constructors (Error, Array, Function, etc.) don't have
    // .prototype properties. Identify them by pointer-comparing with global
    // scope entries and do direct type checking.
    if let JsValue::NativeFunction(ref ctor_fn) = constructor {
        let acc = &ctx.frame.accumulator;
        let global = ctx.frame.global_env.borrow();

        type BuiltinCheck = (&'static str, fn(&JsValue) -> bool);
        let builtin_checks: &[BuiltinCheck] = &[
            ("Array", |v| {
                matches!(v, JsValue::Array(_))
                    || matches!(v, JsValue::PlainObject(m) if m.borrow().get("__is_array__").is_some())
            }),
            ("Function", |v| {
                matches!(v, JsValue::Function(_) | JsValue::NativeFunction(_))
            }),
            ("Promise", |v| matches!(v, JsValue::Promise(_))),
            // Error hierarchy: instanceof Error matches ALL error kinds
            ("Error", |v| matches!(v, JsValue::Error(_))),
        ];
        for &(name, predicate) in builtin_checks {
            if let Some(JsValue::NativeFunction(global_fn)) = global.get(name)
                && Rc::ptr_eq(ctor_fn, global_fn)
            {
                let result = predicate(acc);
                drop(global);
                ctx.frame.accumulator = JsValue::Boolean(result);
                return Ok(DispatchAction::Continue);
            }
        }
        // Specific error sub-types: TypeError, RangeError, etc.
        // These inherit from Error, so `e instanceof TypeError` should check e.kind
        let error_kinds: &[(&str, ErrorKind)] = &[
            ("TypeError", ErrorKind::TypeError),
            ("RangeError", ErrorKind::RangeError),
            ("ReferenceError", ErrorKind::ReferenceError),
            ("SyntaxError", ErrorKind::SyntaxError),
            ("URIError", ErrorKind::URIError),
            ("EvalError", ErrorKind::EvalError),
            ("AggregateError", ErrorKind::AggregateError),
        ];
        for &(name, kind) in error_kinds {
            if let Some(JsValue::NativeFunction(global_fn)) = global.get(name)
                && Rc::ptr_eq(ctor_fn, global_fn)
            {
                let is_match = matches!(acc, JsValue::Error(e) if e.kind == kind);
                drop(global);
                ctx.frame.accumulator = JsValue::Boolean(is_match);
                return Ok(DispatchAction::Continue);
            }
        }
        drop(global);
    }

    // Obtain the constructor's "prototype" property.
    let ctor_proto = match &constructor {
        JsValue::PlainObject(map) => map.borrow().get("prototype").cloned(),
        _ => None,
    };

    let result = if let Some(proto_val) = ctor_proto {
        // Walk the __proto__ chain of the accumulator object.
        let mut current = ctx.frame.accumulator.clone();
        let mut found = false;
        for _ in 0..256 {
            match &current {
                JsValue::PlainObject(map) => {
                    if let JsValue::PlainObject(p) = &proto_val
                        && Rc::ptr_eq(map, p)
                    {
                        found = true;
                        break;
                    }
                    let next = map.borrow().get("__proto__").cloned();
                    match next {
                        Some(v) => current = v,
                        None => break,
                    }
                }
                _ => break,
            }
        }
        found
    } else {
        false
    };

    ctx.frame.accumulator = JsValue::Boolean(result);
    Ok(DispatchAction::Continue)
}

fn handle_test_in(ctx: &mut DispatchContext, instr: &Instruction) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("TestIn", 0));
    };
    let object = ctx.frame.read_reg(v)?.clone();
    let key = &ctx.frame.accumulator;

    let result = match &object {
        JsValue::Proxy(p) => {
            let prop = to_property_key(key)?;
            proxy_has(&p.borrow(), &prop).unwrap_or(false)
        }
        JsValue::PlainObject(_) => {
            let prop = to_property_key(key)?;
            // Walk the prototype chain for `in` operator.
            !matches!(proto_lookup(&object, &prop), JsValue::Undefined)
        }
        JsValue::Array(items) => {
            // "length" is always present on arrays.
            if let JsValue::String(s) = key
                && &**s == "length"
            {
                true
            } else if let Some(idx) = to_array_index(key) {
                idx < items.borrow().len()
            } else {
                // Check Array.prototype / Object.prototype methods.
                let prop = to_property_key(key)?;
                !matches!(proto_lookup(&object, &prop), JsValue::Undefined)
            }
        }
        JsValue::TypedArray(ta) => {
            // TypedArray: numeric indices check bounds, "length" is always present.
            if let JsValue::String(s) = key
                && &**s == "length"
            {
                true
            } else if let Some(idx) = to_array_index(key) {
                idx < ta.borrow().length
            } else {
                let prop = to_property_key(key)?;
                !matches!(proto_lookup(&object, &prop), JsValue::Undefined)
            }
        }
        JsValue::Function(_)
        | JsValue::Error(_)
        | JsValue::Promise(_)
        | JsValue::Generator(_)
        | JsValue::Iterator(_)
        | JsValue::NativeFunction(_)
        | JsValue::ArrayBuffer(_)
        | JsValue::DataView(_)
        | JsValue::Object(_) => {
            // Object-like types without own properties — fall back to proto.
            let prop = to_property_key(key)?;
            !matches!(proto_lookup(&object, &prop), JsValue::Undefined)
        }
        other => {
            return Err(StatorError::TypeError(format!(
                "Cannot use 'in' operator to search for '{}' in {}",
                key.to_display_string(),
                other.to_display_string()
            )));
        }
    };

    ctx.frame.accumulator = JsValue::Boolean(result);
    Ok(DispatchAction::Continue)
}

fn handle_for_in_enumerate(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("ForInEnumerate", 0));
    };
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    let keys: Vec<JsValue> = match &obj {
        JsValue::PlainObject(map) => {
            let mut all_keys = Vec::new();
            let mut seen = std::collections::HashSet::new();
            // Walk the prototype chain collecting enumerable keys.
            // ALL own keys (including non-enumerable) are added to `seen` so
            // that non-enumerable own properties correctly shadow inherited
            // enumerable ones (ES §14.7.5.9).
            let mut current_map = Some(Rc::clone(map));
            for _ in 0..256 {
                let Some(m) = current_map.take() else { break };
                let borrow = m.borrow();
                for (k, _val, attrs) in borrow.iter_with_attrs() {
                    let is_enumerable = attrs.contains(PropertyAttributes::ENUMERABLE);
                    // Translate accessor convention keys (__get_X__ / __set_X__)
                    // to the actual property name X.
                    if let Some(prop) = k
                        .strip_prefix("__get_")
                        .and_then(|s| s.strip_suffix("__"))
                    {
                        seen.insert(k.clone());
                        if seen.insert(prop.to_string()) && is_enumerable {
                            all_keys.push(JsValue::String(prop.to_string().into()));
                        }
                        continue;
                    }
                    if let Some(prop) = k
                        .strip_prefix("__set_")
                        .and_then(|s| s.strip_suffix("__"))
                    {
                        seen.insert(k.clone());
                        if seen.insert(prop.to_string()) && is_enumerable {
                            all_keys.push(JsValue::String(prop.to_string().into()));
                        }
                        continue;
                    }
                    // Regular property: add to seen for shadowing, push if
                    // enumerable and not already seen at a higher level.
                    if seen.insert(k.clone()) && is_enumerable {
                        all_keys.push(JsValue::String(k.clone().into()));
                    }
                }
                current_map = borrow.get("__proto__").and_then(|v| {
                    if let JsValue::PlainObject(proto) = v {
                        Some(Rc::clone(proto))
                    } else {
                        None
                    }
                });
            }
            all_keys
        }
        JsValue::Array(items) => (0..items.borrow().len())
            .map(|i| JsValue::String(i.to_string().into()))
            .collect(),
        JsValue::Null | JsValue::Undefined => vec![],
        _ => vec![],
    };
    ctx.frame.accumulator = JsValue::new_array(keys);
    Ok(DispatchAction::Continue)
}

fn handle_for_in_prepare(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(keys_v) = instr.operands[0] else {
        return Err(err_bad_operand("ForInPrepare", 0));
    };
    // operands[1] is a FeedbackSlot, ignored at runtime.
    let keys = ctx.frame.read_reg(keys_v)?.clone();
    let len = match &keys {
        JsValue::Array(items) => items.borrow().len() as i32,
        _ => 0,
    };
    ctx.frame.accumulator = JsValue::Smi(len);
    Ok(DispatchAction::Continue)
}

fn handle_for_in_next(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(_receiver_v) = instr.operands[0] else {
        return Err(err_bad_operand("ForInNext", 0));
    };
    let Operand::Register(idx_v) = instr.operands[1] else {
        return Err(err_bad_operand("ForInNext", 1));
    };
    let Operand::Register(keys_v) = instr.operands[2] else {
        return Err(err_bad_operand("ForInNext", 2));
    };
    // operands[3] is a FeedbackSlot, ignored at runtime.
    let idx = match ctx.frame.read_reg(idx_v)? {
        JsValue::Smi(n) => (*n).max(0) as usize,
        _ => 0,
    };
    let keys = ctx.frame.read_reg(keys_v)?.clone();
    let key = match &keys {
        JsValue::Array(items) => items
            .borrow()
            .get(idx)
            .cloned()
            .unwrap_or(JsValue::Undefined),
        _ => JsValue::Undefined,
    };
    ctx.frame.accumulator = key;
    Ok(DispatchAction::Continue)
}

fn handle_for_in_step(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(idx_v) = instr.operands[0] else {
        return Err(err_bad_operand("ForInStep", 0));
    };
    let idx = match ctx.frame.read_reg(idx_v)? {
        JsValue::Smi(n) => *n,
        _ => 0,
    };
    ctx.frame.accumulator = JsValue::Smi(idx + 1);
    Ok(DispatchAction::Continue)
}

fn handle_jump_if_for_in_done(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::JumpOffset(delta) = instr.operands[0] else {
        return Err(err_bad_operand("JumpIfForInDone", 0));
    };
    let Operand::Register(idx_v) = instr.operands[1] else {
        return Err(err_bad_operand("JumpIfForInDone", 1));
    };
    let Operand::Register(len_v) = instr.operands[2] else {
        return Err(err_bad_operand("JumpIfForInDone", 2));
    };
    let idx = match ctx.frame.read_reg(idx_v)? {
        JsValue::Smi(n) => *n,
        _ => 0,
    };
    let len = match ctx.frame.read_reg(len_v)? {
        JsValue::Smi(n) => *n,
        _ => 0,
    };
    if idx >= len {
        ctx.frame.pc = resolve_jump(
            ctx.frame.pc,
            delta,
            ctx.byte_offsets,
            ctx.instructions.len(),
        )?;
    }
    Ok(DispatchAction::Continue)
}

fn handle_delete_property_sloppy(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("DeletePropertySloppy", 0));
    };
    let key = to_property_key(&ctx.frame.accumulator)?;
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    let removed = if let JsValue::Proxy(ref p) = obj {
        proxy_delete_property(&mut p.borrow_mut(), &key).unwrap_or(false)
    } else if let JsValue::PlainObject(ref map) = obj {
        let pm = map.borrow();
        if pm.contains_key(&key) {
            if !pm.is_configurable(&key) {
                // Non-configurable: silently fail in sloppy mode.
                false
            } else {
                drop(pm);
                map.borrow_mut().remove(&key);
                true
            }
        } else {
            // Non-existent property: return true per spec.
            true
        }
    } else if let JsValue::Array(ref items) = obj {
        if key == "length" {
            // "length" is non-configurable on arrays.
            false
        } else if let Ok(idx) = key.parse::<usize>() {
            let mut arr = items.borrow_mut();
            if idx < arr.len() {
                arr[idx] = JsValue::Undefined;
            }
            true
        } else {
            true
        }
    } else {
        // Primitives and other object types: delete returns true.
        true
    };
    ctx.frame.accumulator = JsValue::Boolean(removed);
    Ok(DispatchAction::Continue)
}

fn handle_delete_property_strict(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("DeletePropertyStrict", 0));
    };
    let key = to_property_key(&ctx.frame.accumulator)?;
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    if let JsValue::Proxy(ref p) = obj {
        proxy_delete_property(&mut p.borrow_mut(), &key)?;
    } else if let JsValue::PlainObject(ref map) = obj {
        let pm = map.borrow();
        if pm.contains_key(&key) && !pm.is_configurable(&key) {
            return Err(StatorError::TypeError(format!(
                "Cannot delete property '{key}' of object"
            )));
        }
        drop(pm);
        map.borrow_mut().remove(&key);
    } else if let JsValue::Array(ref items) = obj {
        if key == "length" {
            return Err(StatorError::TypeError(
                "Cannot delete property 'length' of array".to_string(),
            ));
        }
        if let Ok(idx) = key.parse::<usize>() {
            let mut arr = items.borrow_mut();
            if idx < arr.len() {
                arr[idx] = JsValue::Undefined;
            }
        }
    }
    ctx.frame.accumulator = JsValue::Boolean(true);
    Ok(DispatchAction::Continue)
}

fn handle_create_rest_parameter(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let param_count = ctx.frame.bytecode_array.parameter_count() as usize;
    let rest: Vec<JsValue> = if ctx.frame.registers.len() > param_count {
        ctx.frame.registers[param_count..].to_vec()
    } else {
        vec![]
    };
    ctx.frame.accumulator = JsValue::new_array(rest);
    Ok(DispatchAction::Continue)
}

fn handle_create_mapped_arguments(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let param_count = ctx.frame.bytecode_array.parameter_count() as usize;
    let args: Vec<JsValue> = ctx
        .frame
        .registers
        .get(..param_count)
        .unwrap_or(&[])
        .to_vec();
    let mut map = PropertyMap::new();
    for (i, v) in args.iter().enumerate() {
        map.insert(i.to_string(), v.clone());
    }
    map.insert("length".to_string(), JsValue::Smi(args.len() as i32));
    // callee: reference to the executing function (sloppy mode only)
    map.insert(
        "callee".to_string(),
        JsValue::Function(Rc::new(ctx.frame.bytecode_array.clone())),
    );
    // @@iterator: array-like iteration support via NativeIterator
    let args_for_iter = args.clone();
    map.insert(
        "@@iterator".to_string(),
        JsValue::NativeFunction(Rc::new(move |_args: Vec<JsValue>| {
            Ok(JsValue::Iterator(NativeIterator::from_items(
                args_for_iter.clone(),
            )))
        })),
    );
    ctx.frame.accumulator = JsValue::PlainObject(Rc::new(RefCell::new(map)));
    Ok(DispatchAction::Continue)
}

fn handle_create_unmapped_arguments(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let param_count = ctx.frame.bytecode_array.parameter_count() as usize;
    let args: Vec<JsValue> = ctx
        .frame
        .registers
        .get(..param_count)
        .unwrap_or(&[])
        .to_vec();
    let mut map = PropertyMap::new();
    for (i, v) in args.iter().enumerate() {
        map.insert(i.to_string(), v.clone());
    }
    map.insert("length".to_string(), JsValue::Smi(args.len() as i32));
    // callee: throws TypeError in strict mode
    map.insert(
        "callee".to_string(),
        JsValue::NativeFunction(Rc::new(|_args: Vec<JsValue>| {
            Err(StatorError::TypeError(
                "'caller', 'callee', and 'arguments' properties may not be accessed on strict mode functions or the arguments objects for calls to them".into(),
            ))
        })),
    );
    // @@iterator: array-like iteration support via NativeIterator
    let args_for_iter = args.clone();
    map.insert(
        "@@iterator".to_string(),
        JsValue::NativeFunction(Rc::new(move |_args: Vec<JsValue>| {
            Ok(JsValue::Iterator(NativeIterator::from_items(
                args_for_iter.clone(),
            )))
        })),
    );
    ctx.frame.accumulator = JsValue::PlainObject(Rc::new(RefCell::new(map)));
    Ok(DispatchAction::Continue)
}

fn handle_throw_reference_error_if_hole(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[0] else {
        return Err(err_bad_operand("ThrowReferenceErrorIfHole", 0));
    };
    if ctx.frame.accumulator == JsValue::TheHole {
        let name = match ctx.frame.bytecode_array.get_constant(name_idx) {
            Some(ConstantPoolEntry::String(s)) => s.clone(),
            _ => "<unknown>".to_string(),
        };
        return Err(StatorError::ReferenceError(format!(
            "Cannot access '{name}' before initialization"
        )));
    }
    Ok(DispatchAction::Continue)
}

fn handle_throw_super_not_called_if_hole(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    if ctx.frame.accumulator == JsValue::TheHole {
        return Err(StatorError::ReferenceError(
            "Must call super constructor in derived class \
         before accessing 'this' or returning from \
         derived constructor"
                .to_string(),
        ));
    }
    Ok(DispatchAction::Continue)
}

fn handle_throw_super_already_called_if_not_hole(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    if ctx.frame.accumulator != JsValue::TheHole {
        return Err(StatorError::ReferenceError(
            "Super constructor may only be called once".to_string(),
        ));
    }
    Ok(DispatchAction::Continue)
}

fn handle_call_property0(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(callee_v) = instr.operands[0] else {
        return Err(err_bad_operand("CallProperty0", 0));
    };
    let Operand::Register(recv_v) = instr.operands[1] else {
        return Err(err_bad_operand("CallProperty0", 1));
    };
    let callee = ctx.frame.read_reg(callee_v)?.clone();
    let this_val = ctx.frame.read_reg(recv_v)?.clone();
    dispatch_call_property(ctx.frame, &callee, this_val, vec![])?;
    Ok(DispatchAction::Continue)
}

fn handle_call_property1(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(callee_v) = instr.operands[0] else {
        return Err(err_bad_operand("CallProperty1", 0));
    };
    let Operand::Register(recv_v) = instr.operands[1] else {
        return Err(err_bad_operand("CallProperty1", 1));
    };
    let Operand::Register(arg0_v) = instr.operands[2] else {
        return Err(err_bad_operand("CallProperty1", 2));
    };
    let callee = ctx.frame.read_reg(callee_v)?.clone();
    let this_val = ctx.frame.read_reg(recv_v)?.clone();
    let arg0 = ctx.frame.read_reg(arg0_v)?.clone();
    dispatch_call_property(ctx.frame, &callee, this_val, vec![arg0])?;
    Ok(DispatchAction::Continue)
}

fn handle_call_property2(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(callee_v) = instr.operands[0] else {
        return Err(err_bad_operand("CallProperty2", 0));
    };
    let Operand::Register(recv_v) = instr.operands[1] else {
        return Err(err_bad_operand("CallProperty2", 1));
    };
    let Operand::Register(arg0_v) = instr.operands[2] else {
        return Err(err_bad_operand("CallProperty2", 2));
    };
    let Operand::Register(arg1_v) = instr.operands[3] else {
        return Err(err_bad_operand("CallProperty2", 3));
    };
    let callee = ctx.frame.read_reg(callee_v)?.clone();
    let this_val = ctx.frame.read_reg(recv_v)?.clone();
    let arg0 = ctx.frame.read_reg(arg0_v)?.clone();
    let arg1 = ctx.frame.read_reg(arg1_v)?.clone();
    dispatch_call_property(ctx.frame, &callee, this_val, vec![arg0, arg1])?;
    Ok(DispatchAction::Continue)
}

fn handle_call_runtime(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::RuntimeId(runtime_id) = instr.operands[0] else {
        return Err(err_bad_operand("CallRuntime", 0));
    };
    let Operand::Register(args_start_v) = instr.operands[1] else {
        return Err(err_bad_operand("CallRuntime", 1));
    };
    let Operand::RegisterCount(arg_count) = instr.operands[2] else {
        return Err(err_bad_operand("CallRuntime", 2));
    };

    if runtime_id == crate::bytecode::bytecode_generator::RUNTIME_DYNAMIC_IMPORT {
        use crate::builtins::promise::{MicrotaskQueue, promise_resolve};

        let args = collect_args(ctx.frame, args_start_v, arg_count)?;
        let specifier = args.first().cloned().unwrap_or(JsValue::Undefined);

        // Build a namespace object with a default export
        // equal to the specifier string (stub for now — a
        // full implementation would resolve a module).
        let ns = PropertyMap::new();
        let ns_val = JsValue::PlainObject(Rc::new(RefCell::new(ns)));

        let queue = MicrotaskQueue::new();
        let p = promise_resolve(ns_val, &queue);
        queue.drain();
        ctx.frame.accumulator = JsValue::Promise(p);

        let _ = specifier; // consumed for future use
    }
    // Unrecognised runtime IDs are no-ops.
    Ok(DispatchAction::Continue)
}

fn handle_sta_named_own_property(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("StaNamedOwnProperty", 0));
    };
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[1] else {
        return Err(err_bad_operand("StaNamedOwnProperty", 1));
    };
    let prop_name = match ctx.frame.bytecode_array.get_constant(name_idx) {
        Some(ConstantPoolEntry::String(s)) => s.clone(),
        _ => {
            return Err(StatorError::Internal(
                "StaNamedOwnProperty: property name is \
                 not a string"
                    .into(),
            ));
        }
    };
    let val = ctx.frame.accumulator.clone();
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    if let JsValue::PlainObject(ref map) = obj {
        map.borrow_mut().insert(prop_name, val);
    }
    Ok(DispatchAction::Continue)
}

fn handle_sta_lookup_slot(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[0] else {
        return Err(err_bad_operand("StaLookupSlot", 0));
    };
    let name = match ctx.frame.bytecode_array.get_constant(name_idx) {
        Some(ConstantPoolEntry::String(s)) => s.clone(),
        _ => {
            return Err(StatorError::Internal(
                "StaLookupSlot: slot name is not a string".into(),
            ));
        }
    };
    let val = ctx.frame.accumulator.clone();
    let mut env = ctx.frame.global_env.borrow_mut();
    // Strict mode: assigning to an undeclared variable is a ReferenceError.
    if ctx.frame.bytecode_array.is_strict() && !env.contains_key(&name) {
        return Err(StatorError::ReferenceError(format!(
            "{name} is not defined"
        )));
    }
    env.insert(name, val);
    Ok(DispatchAction::Continue)
}

fn handle_lda_lookup_slot(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[0] else {
        return Err(err_bad_operand("LdaLookupSlot", 0));
    };
    let name = match ctx.frame.bytecode_array.get_constant(name_idx) {
        Some(ConstantPoolEntry::String(s)) => s.clone(),
        _ => {
            return Err(StatorError::Internal(
                "LdaLookupSlot: slot name is not a string".into(),
            ));
        }
    };
    ctx.frame.accumulator = match ctx.frame.global_env.borrow().get(&name) {
        Some(v) => v.clone(),
        None => {
            return Err(StatorError::ReferenceError(format!(
                "{name} is not defined"
            )));
        }
    };
    Ok(DispatchAction::Continue)
}

fn handle_lda_lookup_slot_inside_typeof(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[0] else {
        return Err(err_bad_operand("LdaLookupSlotInsideTypeof", 0));
    };
    let name = match ctx.frame.bytecode_array.get_constant(name_idx) {
        Some(ConstantPoolEntry::String(s)) => s.clone(),
        _ => {
            return Err(StatorError::Internal(
                "LdaLookupSlotInsideTypeof: slot name is not a string".into(),
            ));
        }
    };
    ctx.frame.accumulator = ctx
        .frame
        .global_env
        .borrow()
        .get(&name)
        .cloned()
        .unwrap_or(JsValue::Undefined);
    Ok(DispatchAction::Continue)
}

fn handle_lda_lookup_context_slot(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[0] else {
        return Err(err_bad_operand("LdaLookupContextSlot", 0));
    };
    let Operand::ConstantPoolIdx(slot_idx) = instr.operands[1] else {
        return Err(err_bad_operand("LdaLookupContextSlot", 1));
    };
    let Operand::Immediate(depth) = instr.operands[2] else {
        return Err(err_bad_operand("LdaLookupContextSlot", 2));
    };

    // Fast path: read from the context chain at the statically known slot/depth.
    if let Some(ctx_val) = &ctx.frame.context
        && let Ok(js_ctx) = extract_context(ctx_val, "LdaLookupContextSlot")
        && let Ok(target) = walk_context_chain(&js_ctx, depth as u32, "LdaLookupContextSlot")
    {
        let borrowed = target.borrow();
        let slot = slot_idx as usize;
        if let Some(val) = borrowed.slots.get(slot) {
            ctx.frame.accumulator = val.clone();
            return Ok(DispatchAction::Continue);
        }
    }

    // Slow path: fall back to global name-based lookup.
    let name = match ctx.frame.bytecode_array.get_constant(name_idx) {
        Some(ConstantPoolEntry::String(s)) => s.clone(),
        _ => {
            return Err(StatorError::Internal(
                "LdaLookupContextSlot: slot name is not a string".into(),
            ));
        }
    };
    ctx.frame.accumulator = match ctx.frame.global_env.borrow().get(&name) {
        Some(v) => v.clone(),
        None => {
            return Err(StatorError::ReferenceError(format!(
                "{name} is not defined"
            )));
        }
    };
    Ok(DispatchAction::Continue)
}

fn handle_lda_lookup_context_slot_inside_typeof(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[0] else {
        return Err(err_bad_operand("LdaLookupContextSlotInsideTypeof", 0));
    };
    let Operand::ConstantPoolIdx(slot_idx) = instr.operands[1] else {
        return Err(err_bad_operand("LdaLookupContextSlotInsideTypeof", 1));
    };
    let Operand::Immediate(depth) = instr.operands[2] else {
        return Err(err_bad_operand("LdaLookupContextSlotInsideTypeof", 2));
    };

    // Fast path: read from the context chain at the statically known slot/depth.
    if let Some(ctx_val) = &ctx.frame.context
        && let Ok(js_ctx) = extract_context(ctx_val, "LdaLookupContextSlotInsideTypeof")
        && let Ok(target) =
            walk_context_chain(&js_ctx, depth as u32, "LdaLookupContextSlotInsideTypeof")
    {
        let borrowed = target.borrow();
        let slot = slot_idx as usize;
        if let Some(val) = borrowed.slots.get(slot) {
            ctx.frame.accumulator = val.clone();
            return Ok(DispatchAction::Continue);
        }
    }

    // Slow path: fall back to global name-based lookup (undefined if missing).
    let name = match ctx.frame.bytecode_array.get_constant(name_idx) {
        Some(ConstantPoolEntry::String(s)) => s.clone(),
        _ => {
            return Err(StatorError::Internal(
                "LdaLookupContextSlotInsideTypeof: slot name is not a string".into(),
            ));
        }
    };
    ctx.frame.accumulator = ctx
        .frame
        .global_env
        .borrow()
        .get(&name)
        .cloned()
        .unwrap_or(JsValue::Undefined);
    Ok(DispatchAction::Continue)
}

fn handle_lda_lookup_global_slot(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[0] else {
        return Err(err_bad_operand("LdaLookupGlobalSlot", 0));
    };
    let name = match ctx.frame.bytecode_array.get_constant(name_idx) {
        Some(ConstantPoolEntry::String(s)) => s.clone(),
        _ => {
            return Err(StatorError::Internal(
                "LdaLookupGlobalSlot: slot name is not a string".into(),
            ));
        }
    };
    ctx.frame.accumulator = match ctx.frame.global_env.borrow().get(&name) {
        Some(v) => v.clone(),
        None => {
            return Err(StatorError::ReferenceError(format!(
                "{name} is not defined"
            )));
        }
    };
    Ok(DispatchAction::Continue)
}

fn handle_lda_lookup_global_slot_inside_typeof(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[0] else {
        return Err(err_bad_operand("LdaLookupGlobalSlotInsideTypeof", 0));
    };
    let name = match ctx.frame.bytecode_array.get_constant(name_idx) {
        Some(ConstantPoolEntry::String(s)) => s.clone(),
        _ => {
            return Err(StatorError::Internal(
                "LdaLookupGlobalSlotInsideTypeof: slot name is not a string".into(),
            ));
        }
    };
    ctx.frame.accumulator = ctx
        .frame
        .global_env
        .borrow()
        .get(&name)
        .cloned()
        .unwrap_or(JsValue::Undefined);
    Ok(DispatchAction::Continue)
}

fn handle_lda_named_property_from_super(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("LdaNamedPropertyFromSuper", 0));
    };
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[1] else {
        return Err(err_bad_operand("LdaNamedPropertyFromSuper", 1));
    };
    let prop_name = match ctx.frame.bytecode_array.get_constant(name_idx) {
        Some(ConstantPoolEntry::String(s)) => s.clone(),
        _ => {
            return Err(StatorError::Internal(
                "LdaNamedPropertyFromSuper: property name is not a string".into(),
            ));
        }
    };
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    ctx.frame.accumulator = proto_lookup(&obj, &prop_name);
    Ok(DispatchAction::Continue)
}

fn handle_get_template_object(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(tpl_idx) = instr.operands[0] else {
        return Err(err_bad_operand("GetTemplateObject", 0));
    };
    let cache_key = ctx.byte_offsets[ctx.frame.pc - 1] as u32;
    if let Some(cached) = ctx.frame.template_cache.get(&cache_key) {
        ctx.frame.accumulator = cached.clone();
    } else {
        let entry = ctx
            .frame
            .bytecode_array
            .get_constant(tpl_idx)
            .ok_or_else(|| {
                StatorError::Internal(format!(
                    "GetTemplateObject: constant pool index {tpl_idx} out of bounds"
                ))
            })?;
        let tpl_val = constant_to_value(entry);
        ctx.frame.template_cache.insert(cache_key, tpl_val.clone());
        ctx.frame.accumulator = tpl_val;
    }
    Ok(DispatchAction::Continue)
}

fn handle_set_pending_message(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    std::mem::swap(&mut ctx.frame.accumulator, &mut ctx.frame.pending_message);
    Ok(DispatchAction::Continue)
}

fn handle_test_reference_equal(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("TestReferenceEqual", 0));
    };
    let rhs = ctx.frame.read_reg(v)?.clone();
    let result = strict_eq(&ctx.frame.accumulator, &rhs);
    ctx.frame.accumulator = JsValue::Boolean(result);
    Ok(DispatchAction::Continue)
}

fn handle_test_undetectable(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let result = matches!(ctx.frame.accumulator, JsValue::Null | JsValue::Undefined);
    ctx.frame.accumulator = JsValue::Boolean(result);
    Ok(DispatchAction::Continue)
}

fn handle_jump_if_js_receiver(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::JumpOffset(delta) = instr.operands[0] else {
        return Err(err_bad_operand("JumpIfJSReceiver", 0));
    };
    if is_js_receiver(&ctx.frame.accumulator) {
        ctx.frame.pc = resolve_jump(
            ctx.frame.pc,
            delta,
            ctx.byte_offsets,
            ctx.instructions.len(),
        )?;
    }
    Ok(DispatchAction::Continue)
}

fn handle_jump_if_js_receiver_constant(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(idx) = instr.operands[0] else {
        return Err(err_bad_operand("JumpIfJSReceiverConstant", 0));
    };
    if is_js_receiver(&ctx.frame.accumulator) {
        let delta = constant_pool_jump_delta(ctx.frame, idx, "JumpIfJSReceiverConstant")?;
        ctx.frame.pc = resolve_jump(
            ctx.frame.pc,
            delta,
            ctx.byte_offsets,
            ctx.instructions.len(),
        )?;
    }
    Ok(DispatchAction::Continue)
}

fn handle_to_numeric(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    // operands[0] is a FeedbackSlot, ignored at runtime.
    if !matches!(ctx.frame.accumulator, JsValue::BigInt(_)) {
        let n = ctx.frame.accumulator.to_number()?;
        ctx.frame.accumulator = number_to_jsvalue(n);
    }
    Ok(DispatchAction::Continue)
}

fn handle_wide(_ctx: &mut DispatchContext, _instr: &Instruction) -> StatorResult<DispatchAction> {
    Err(StatorError::Internal(
        "Wide/ExtraWide prefix should not appear as a decoded opcode".into(),
    ))
}

fn handle_jump_constant(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(idx) = instr.operands[0] else {
        return Err(err_bad_operand("JumpConstant", 0));
    };
    let delta = constant_pool_jump_delta(ctx.frame, idx, "JumpConstant")?;
    ctx.frame.pc = resolve_jump(
        ctx.frame.pc,
        delta,
        ctx.byte_offsets,
        ctx.instructions.len(),
    )?;
    Ok(DispatchAction::Continue)
}

fn handle_jump_if_true_constant(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(idx) = instr.operands[0] else {
        return Err(err_bad_operand("JumpIfTrueConstant", 0));
    };
    if matches!(ctx.frame.accumulator, JsValue::Boolean(true)) {
        let delta = constant_pool_jump_delta(ctx.frame, idx, "JumpIfTrueConstant")?;
        ctx.frame.pc = resolve_jump(
            ctx.frame.pc,
            delta,
            ctx.byte_offsets,
            ctx.instructions.len(),
        )?;
    }
    Ok(DispatchAction::Continue)
}

fn handle_jump_if_false_constant(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(idx) = instr.operands[0] else {
        return Err(err_bad_operand("JumpIfFalseConstant", 0));
    };
    if matches!(ctx.frame.accumulator, JsValue::Boolean(false)) {
        let delta = constant_pool_jump_delta(ctx.frame, idx, "JumpIfFalseConstant")?;
        ctx.frame.pc = resolve_jump(
            ctx.frame.pc,
            delta,
            ctx.byte_offsets,
            ctx.instructions.len(),
        )?;
    }
    Ok(DispatchAction::Continue)
}

fn handle_jump_if_to_boolean_true_constant(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(idx) = instr.operands[0] else {
        return Err(err_bad_operand("JumpIfToBooleanTrueConstant", 0));
    };
    if ctx.frame.accumulator.to_boolean() {
        let delta = constant_pool_jump_delta(ctx.frame, idx, "JumpIfToBooleanTrueConstant")?;
        ctx.frame.pc = resolve_jump(
            ctx.frame.pc,
            delta,
            ctx.byte_offsets,
            ctx.instructions.len(),
        )?;
    }
    Ok(DispatchAction::Continue)
}

fn handle_jump_if_to_boolean_false_constant(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(idx) = instr.operands[0] else {
        return Err(err_bad_operand("JumpIfToBooleanFalseConstant", 0));
    };
    if !ctx.frame.accumulator.to_boolean() {
        let delta = constant_pool_jump_delta(ctx.frame, idx, "JumpIfToBooleanFalseConstant")?;
        ctx.frame.pc = resolve_jump(
            ctx.frame.pc,
            delta,
            ctx.byte_offsets,
            ctx.instructions.len(),
        )?;
    }
    Ok(DispatchAction::Continue)
}

fn handle_jump_if_null_constant(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(idx) = instr.operands[0] else {
        return Err(err_bad_operand("JumpIfNullConstant", 0));
    };
    if ctx.frame.accumulator.is_null() {
        let delta = constant_pool_jump_delta(ctx.frame, idx, "JumpIfNullConstant")?;
        ctx.frame.pc = resolve_jump(
            ctx.frame.pc,
            delta,
            ctx.byte_offsets,
            ctx.instructions.len(),
        )?;
    }
    Ok(DispatchAction::Continue)
}

fn handle_jump_if_not_null_constant(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(idx) = instr.operands[0] else {
        return Err(err_bad_operand("JumpIfNotNullConstant", 0));
    };
    if !ctx.frame.accumulator.is_null() {
        let delta = constant_pool_jump_delta(ctx.frame, idx, "JumpIfNotNullConstant")?;
        ctx.frame.pc = resolve_jump(
            ctx.frame.pc,
            delta,
            ctx.byte_offsets,
            ctx.instructions.len(),
        )?;
    }
    Ok(DispatchAction::Continue)
}

fn handle_jump_if_undefined_constant(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(idx) = instr.operands[0] else {
        return Err(err_bad_operand("JumpIfUndefinedConstant", 0));
    };
    if ctx.frame.accumulator.is_undefined() {
        let delta = constant_pool_jump_delta(ctx.frame, idx, "JumpIfUndefinedConstant")?;
        ctx.frame.pc = resolve_jump(
            ctx.frame.pc,
            delta,
            ctx.byte_offsets,
            ctx.instructions.len(),
        )?;
    }
    Ok(DispatchAction::Continue)
}

fn handle_jump_if_not_undefined_constant(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(idx) = instr.operands[0] else {
        return Err(err_bad_operand("JumpIfNotUndefinedConstant", 0));
    };
    if !ctx.frame.accumulator.is_undefined() {
        let delta = constant_pool_jump_delta(ctx.frame, idx, "JumpIfNotUndefinedConstant")?;
        ctx.frame.pc = resolve_jump(
            ctx.frame.pc,
            delta,
            ctx.byte_offsets,
            ctx.instructions.len(),
        )?;
    }
    Ok(DispatchAction::Continue)
}

fn handle_jump_if_undefined_or_null_constant(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(idx) = instr.operands[0] else {
        return Err(err_bad_operand("JumpIfUndefinedOrNullConstant", 0));
    };
    if ctx.frame.accumulator.is_nullish() {
        let delta = constant_pool_jump_delta(ctx.frame, idx, "JumpIfUndefinedOrNullConstant")?;
        ctx.frame.pc = resolve_jump(
            ctx.frame.pc,
            delta,
            ctx.byte_offsets,
            ctx.instructions.len(),
        )?;
    }
    Ok(DispatchAction::Continue)
}

fn handle_call_js_runtime(
    _ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(_ctx_idx) = instr.operands[0] else {
        return Err(err_bad_operand("CallJSRuntime", 0));
    };
    // No-op: accumulator is left unchanged.
    Ok(DispatchAction::Continue)
}

fn handle_invoke_intrinsic(
    _ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::RuntimeId(_runtime_id) = instr.operands[0] else {
        return Err(err_bad_operand("InvokeIntrinsic", 0));
    };
    // No-op: accumulator is left unchanged.
    Ok(DispatchAction::Continue)
}

fn handle_call_runtime_for_pair(
    _ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::RuntimeId(_runtime_id) = instr.operands[0] else {
        return Err(err_bad_operand("CallRuntimeForPair", 0));
    };
    // No-op: accumulator is left unchanged.
    Ok(DispatchAction::Continue)
}

fn handle_construct_forward_all_args(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(ctor_v) = instr.operands[0] else {
        return Err(err_bad_operand("ConstructForwardAllArgs", 0));
    };
    // operands[1] is a FeedbackSlot, ignored at runtime.
    let ctor = ctx.frame.read_reg(ctor_v)?.clone();
    let ctor_proto = proto_lookup(&ctor, "prototype");
    let param_count = ctx.frame.bytecode_array.parameter_count() as usize;
    let args: Vec<JsValue> = ctx
        .frame
        .registers
        .get(..param_count)
        .unwrap_or(&[])
        .to_vec();
    match ctor {
        JsValue::Function(ba) => {
            let this_obj: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));
            if !matches!(ctor_proto, JsValue::Undefined) {
                this_obj
                    .borrow_mut()
                    .insert("__proto__".to_string(), ctor_proto.clone());
            }
            let this_val = JsValue::PlainObject(this_obj);
            let mut callee_frame = InterpreterFrame::new_with_globals(
                (*ba).clone(),
                args,
                Rc::clone(&ctx.frame.global_env),
            );
            restore_closure_context(&mut callee_frame, &ba);
            callee_frame.new_target = JsValue::Function(Rc::clone(&ba));
            callee_frame
                .global_env
                .borrow_mut()
                .insert("this".to_string(), this_val.clone());
            push_call_frame("<anonymous>")?;
            let result = Interpreter::run(&mut callee_frame);
            pop_call_frame();
            let val = result?;
            ctx.frame.accumulator = match val {
                JsValue::PlainObject(_) | JsValue::Object(_) => val,
                _ => this_val,
            };
        }
        JsValue::NativeFunction(f) => {
            ctx.frame.accumulator = f(args)?;
        }
        JsValue::PlainObject(ref map) => {
            let call_val = map.borrow().get("__call__").cloned();
            match call_val {
                Some(JsValue::NativeFunction(f)) => {
                    let val = f(args)?;
                    ctx.frame.accumulator = wire_construct_prototype(val, &ctor_proto);
                }
                Some(JsValue::Function(ba)) => {
                    construct_class_from_plain_object(ctx, &ba, map, &ctor_proto, args)?;
                }
                _ => {
                    return Err(StatorError::TypeError(format!(
                        "ConstructForwardAllArgs: constructor is not a function (got {other:?})",
                        other = JsValue::PlainObject(Rc::clone(map))
                    )));
                }
            }
        }
        other => {
            return Err(StatorError::TypeError(format!(
                "ConstructForwardAllArgs: constructor is not a function (got {other:?})"
            )));
        }
    }
    Ok(DispatchAction::Continue)
}

fn handle_collect_type_profile(
    _ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    // No-op: operands[0] is an Immediate (position), ignored.
    Ok(DispatchAction::Continue)
}

fn handle_create_object_from_iterable(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let iterable = ctx.frame.accumulator.clone();
    let map: PropertyMap = match &iterable {
        JsValue::PlainObject(obj) => obj.borrow().clone(),
        JsValue::Array(arr) => {
            let mut m = PropertyMap::new();
            for (i, v) in arr.borrow().iter().enumerate() {
                m.insert(i.to_string(), v.clone());
            }
            m.insert(
                "length".to_string(),
                JsValue::Smi(arr.borrow().len() as i32),
            );
            m
        }
        JsValue::Iterator(iter) => {
            let mut m = PropertyMap::new();
            let mut idx = 0usize;
            loop {
                let mut it = iter.borrow_mut();
                match it.next_item() {
                    Some(v) => {
                        m.insert(idx.to_string(), v);
                        idx += 1;
                    }
                    None => break,
                }
            }
            m.insert("length".to_string(), JsValue::Smi(idx as i32));
            m
        }
        _ => PropertyMap::new(),
    };
    ctx.frame.accumulator = JsValue::PlainObject(Rc::new(RefCell::new(map)));
    Ok(DispatchAction::Continue)
}

fn handle_call_direct_eval(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(callee_v) = instr.operands[0] else {
        return Err(err_bad_operand("CallDirectEval", 0));
    };
    let Operand::Register(args_start_v) = instr.operands[1] else {
        return Err(err_bad_operand("CallDirectEval", 1));
    };
    let Operand::RegisterCount(arg_count) = instr.operands[2] else {
        return Err(err_bad_operand("CallDirectEval", 2));
    };
    let callee = ctx.frame.read_reg(callee_v)?.clone();
    let args = collect_args(ctx.frame, args_start_v, arg_count)?;

    // Check whether the callee is the original built-in eval
    // by comparing the Rc pointer with the one stored in the
    // global environment under "eval".
    let is_builtin = if let JsValue::NativeFunction(ref callee_fn) = callee {
        if let Some(JsValue::NativeFunction(ref global_fn)) =
            ctx.frame.global_env.borrow().get("eval").cloned()
        {
            Rc::ptr_eq(callee_fn, global_fn)
        } else {
            false
        }
    } else {
        false
    };

    if is_builtin {
        // Direct eval semantics (ECMAScript §19.2.1.1).
        // Non-string arg → return as-is; no arg → undefined.
        let source = match args.first() {
            Some(JsValue::String(s)) => s.clone(),
            Some(other) => {
                ctx.frame.accumulator = other.clone();
                return Ok(DispatchAction::Continue);
            }
            None => {
                ctx.frame.accumulator = JsValue::Undefined;
                return Ok(DispatchAction::Continue);
            }
        };
        ctx.frame.accumulator =
            crate::builtins::global::global_eval_direct(&source, Rc::clone(&ctx.frame.global_env))?;
    } else {
        // Callee was reassigned — fall through to normal call.
        match callee {
            JsValue::Function(ba) => {
                if ba.is_generator() {
                    ctx.frame.accumulator = JsValue::Generator(GeneratorState::new((*ba).clone()));
                } else {
                    let mut callee_frame = InterpreterFrame::new_with_globals(
                        (*ba).clone(),
                        args,
                        Rc::clone(&ctx.frame.global_env),
                    );
                    restore_closure_context(&mut callee_frame, &ba);
                    let _ = push_call_frame("<eval-fallback>");
                    let result = Interpreter::run(&mut callee_frame);
                    pop_call_frame();
                    ctx.frame.accumulator = result?;
                }
            }
            JsValue::NativeFunction(f) => {
                ctx.frame.accumulator = f(args)?;
            }
            JsValue::PlainObject(ref map) => {
                if let Some(JsValue::NativeFunction(f)) = map.borrow().get("__call__").cloned() {
                    ctx.frame.accumulator = f(args)?;
                } else {
                    return Err(StatorError::TypeError(
                        "CallDirectEval: callee is not a function (got PlainObject)".to_string(),
                    ));
                }
            }
            other => {
                return Err(StatorError::TypeError(format!(
                    "CallDirectEval: callee is not a function (got {other:?})"
                )));
            }
        }
    }
    Ok(DispatchAction::Continue)
}

/// Load `new.target` into the accumulator.
fn handle_lda_new_target(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    ctx.frame.accumulator = ctx.frame.new_target.clone();
    Ok(DispatchAction::Continue)
}

// ── CreateClass ──────────────────────────────────────────────────────────────

/// `CreateClass <ctor_idx> <super_reg> <slot>`
///
/// Creates a class constructor from the bytecode in constant pool entry
/// `ctor_idx`, wires its prototype object, and (optionally) sets up the
/// `extends` relationship with the superclass loaded from `super_reg`.
fn handle_create_class(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(ctor_idx) = instr.operands[0] else {
        return Err(err_bad_operand("CreateClass", 0));
    };
    let Operand::Register(super_v) = instr.operands[1] else {
        return Err(err_bad_operand("CreateClass", 1));
    };
    // operands[2] is a FeedbackSlot, ignored at runtime.

    // 1. Read constructor bytecode from constant pool.
    let entry = ctx
        .frame
        .bytecode_array
        .get_constant(ctor_idx)
        .ok_or_else(|| {
            StatorError::Internal(format!(
                "CreateClass: constant pool index {ctor_idx} out of bounds"
            ))
        })?;
    let ConstantPoolEntry::Function(ctor_ba) = entry else {
        return Err(StatorError::Internal(
            "CreateClass: constant pool entry is not a Function".into(),
        ));
    };

    // 2. Read the superclass.
    let super_val = ctx.frame.read_reg(super_v)?.clone();

    // 3. Create the class constructor as a PlainObject wrapping the
    //    bytecode in __call__ so it can be invoked via both `new` and
    //    direct calls.
    let ctor_ba_rc = Rc::new((**ctor_ba).clone());
    let class_obj: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));
    class_obj.borrow_mut().insert(
        "__call__".to_string(),
        JsValue::Function(Rc::clone(&ctor_ba_rc)),
    );

    // 4. Create the prototype object.
    let proto: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));

    // 5. Wire `extends` — set up prototype chain.
    if !matches!(super_val, JsValue::Undefined | JsValue::Null) {
        // class Foo extends Bar {}
        // proto.__proto__ = super_val.prototype
        let super_proto = proto_lookup(&super_val, "prototype");
        if !matches!(super_proto, JsValue::Undefined) {
            proto
                .borrow_mut()
                .insert("__proto__".to_string(), super_proto);
        }
        // class.__proto__ = super_val (static inheritance)
        class_obj
            .borrow_mut()
            .insert("__proto__".to_string(), super_val);
    }

    // 6. Wire constructor ↔ prototype.
    let proto_val = JsValue::PlainObject(Rc::clone(&proto));
    proto.borrow_mut().insert(
        "constructor".to_string(),
        JsValue::PlainObject(Rc::clone(&class_obj)),
    );
    class_obj
        .borrow_mut()
        .insert("prototype".to_string(), proto_val);

    // 7. Result in accumulator.
    ctx.frame.accumulator = JsValue::PlainObject(class_obj);
    Ok(DispatchAction::Continue)
}

// ── DefineGetterProperty / DefineSetterProperty ──────────────────────────────

/// `DefineGetterProperty <obj_reg> <name_idx> <slot>`
///
/// Defines a getter accessor on the object in `obj_reg`.  The getter
/// function is in the accumulator.  The property name is a string at
/// constant pool index `name_idx`.
fn handle_define_getter_property(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("DefineGetterProperty", 0));
    };
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[1] else {
        return Err(err_bad_operand("DefineGetterProperty", 1));
    };
    let prop_name = match ctx.frame.bytecode_array.get_constant(name_idx) {
        Some(ConstantPoolEntry::String(s)) => s.clone(),
        _ => {
            return Err(StatorError::Internal(
                "DefineGetterProperty: property name is not a string".into(),
            ));
        }
    };
    let getter = ctx.frame.accumulator.clone();
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    if let JsValue::PlainObject(ref map) = obj {
        // Store getter as __get_<name>__ — the property access handler
        // checks for this convention when loading.
        map.borrow_mut()
            .insert(format!("__get_{prop_name}__"), getter);
    }
    Ok(DispatchAction::Continue)
}

/// `DefineSetterProperty <obj_reg> <name_idx> <slot>`
fn handle_define_setter_property(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("DefineSetterProperty", 0));
    };
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[1] else {
        return Err(err_bad_operand("DefineSetterProperty", 1));
    };
    let prop_name = match ctx.frame.bytecode_array.get_constant(name_idx) {
        Some(ConstantPoolEntry::String(s)) => s.clone(),
        _ => {
            return Err(StatorError::Internal(
                "DefineSetterProperty: property name is not a string".into(),
            ));
        }
    };
    let setter = ctx.frame.accumulator.clone();
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    if let JsValue::PlainObject(ref map) = obj {
        map.borrow_mut()
            .insert(format!("__set_{prop_name}__"), setter);
    }
    Ok(DispatchAction::Continue)
}

/// `DefineKeyedGetterProperty <obj_reg> <key_reg> <val_reg> <slot>`
fn handle_define_keyed_getter_property(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("DefineKeyedGetterProperty", 0));
    };
    let Operand::Register(key_v) = instr.operands[1] else {
        return Err(err_bad_operand("DefineKeyedGetterProperty", 1));
    };
    let getter = ctx.frame.accumulator.clone();
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    let key = ctx.frame.read_reg(key_v)?.clone();
    let key_str = to_property_key(&key)?;
    if let JsValue::PlainObject(ref map) = obj {
        map.borrow_mut()
            .insert(format!("__get_{key_str}__"), getter);
    }
    Ok(DispatchAction::Continue)
}

/// `DefineKeyedSetterProperty <obj_reg> <key_reg> <val_reg> <slot>`
fn handle_define_keyed_setter_property(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("DefineKeyedSetterProperty", 0));
    };
    let Operand::Register(key_v) = instr.operands[1] else {
        return Err(err_bad_operand("DefineKeyedSetterProperty", 1));
    };
    let setter = ctx.frame.accumulator.clone();
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    let key = ctx.frame.read_reg(key_v)?.clone();
    let key_str = to_property_key(&key)?;
    if let JsValue::PlainObject(ref map) = obj {
        map.borrow_mut()
            .insert(format!("__set_{key_str}__"), setter);
    }
    Ok(DispatchAction::Continue)
}

/// `LdaEnumeratedKeyedProperty <obj_reg> <key_reg> <slot>`
///
/// Like `LdaKeyedProperty` but for enumerated (for-in) keys.
fn handle_lda_enumerated_keyed_property(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("LdaEnumeratedKeyedProperty", 0));
    };
    let Operand::Register(key_v) = instr.operands[1] else {
        return Err(err_bad_operand("LdaEnumeratedKeyedProperty", 1));
    };
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    let key = ctx.frame.read_reg(key_v)?.clone();
    ctx.frame.accumulator = keyed_load(&obj, &key)?;
    Ok(DispatchAction::Continue)
}

/// `TestPrivateBrand <obj_reg>`
///
/// Checks whether the object in `obj_reg` has the private brand of the
/// current class.  For now, we always succeed (brand checking is a
/// runtime validation for private fields).
fn handle_test_private_brand(
    _ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    // TODO: full brand checking once private field storage is available.
    // For now, pass — this allows class code to proceed without errors.
    Ok(DispatchAction::Continue)
}

/// `DefinePrivateBrand <obj_reg>`
///
/// Brands the object so that subsequent `TestPrivateBrand` calls will
/// succeed.  Currently a no-op stub.
fn handle_define_private_brand(
    _ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    Ok(DispatchAction::Continue)
}

#[cold]
fn handle_unimplemented(
    _ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    Err(StatorError::Internal(format!(
        "unimplemented opcode: {:?}",
        instr.opcode
    )))
}

/// Function-pointer dispatch table indexed by `Opcode as usize`.
///
/// At compile time each slot is filled with the corresponding handler
/// function, giving the CPU branch predictor a direct-call target.
pub(super) static DISPATCH_TABLE: [OpcodeHandler; OPCODE_COUNT] = {
    let mut table: [OpcodeHandler; OPCODE_COUNT] = [handle_unimplemented; OPCODE_COUNT];
    table[Opcode::LdaZero as usize] = handle_lda_zero;
    table[Opcode::LdaSmi as usize] = handle_lda_smi;
    table[Opcode::LdaUndefined as usize] = handle_lda_undefined;
    table[Opcode::LdaTheHole as usize] = handle_lda_the_hole;
    table[Opcode::LdaNull as usize] = handle_lda_null;
    table[Opcode::LdaTrue as usize] = handle_lda_true;
    table[Opcode::LdaFalse as usize] = handle_lda_false;
    table[Opcode::LdaConstant as usize] = handle_lda_constant;
    table[Opcode::LdaGlobal as usize] = handle_lda_global;
    table[Opcode::LdaGlobalInsideTypeof as usize] = handle_lda_global;
    table[Opcode::StaGlobal as usize] = handle_sta_global;
    table[Opcode::LdaContextSlot as usize] = handle_lda_context_slot;
    table[Opcode::LdaImmutableContextSlot as usize] = handle_lda_context_slot;
    table[Opcode::LdaCurrentContextSlot as usize] = handle_lda_current_context_slot;
    table[Opcode::LdaImmutableCurrentContextSlot as usize] = handle_lda_current_context_slot;
    table[Opcode::StaContextSlot as usize] = handle_sta_context_slot;
    table[Opcode::StaCurrentContextSlot as usize] = handle_sta_current_context_slot;
    table[Opcode::LdaLookupSlot as usize] = handle_lda_lookup_slot;
    table[Opcode::LdaLookupContextSlot as usize] = handle_lda_lookup_context_slot;
    table[Opcode::LdaLookupGlobalSlot as usize] = handle_lda_lookup_global_slot;
    table[Opcode::LdaLookupSlotInsideTypeof as usize] = handle_lda_lookup_slot_inside_typeof;
    table[Opcode::LdaLookupContextSlotInsideTypeof as usize] =
        handle_lda_lookup_context_slot_inside_typeof;
    table[Opcode::LdaLookupGlobalSlotInsideTypeof as usize] =
        handle_lda_lookup_global_slot_inside_typeof;
    table[Opcode::StaLookupSlot as usize] = handle_sta_lookup_slot;
    table[Opcode::Ldar as usize] = handle_ldar;
    table[Opcode::Star as usize] = handle_star;
    table[Opcode::Mov as usize] = handle_mov;
    table[Opcode::LdaNamedProperty as usize] = handle_lda_named_property;
    table[Opcode::LdaNamedPropertyFromSuper as usize] = handle_lda_named_property_from_super;
    table[Opcode::LdaKeyedProperty as usize] = handle_lda_keyed_property;
    table[Opcode::LdaEnumeratedKeyedProperty as usize] = handle_lda_enumerated_keyed_property;
    table[Opcode::StaNamedProperty as usize] = handle_sta_named_property;
    table[Opcode::StaNamedOwnProperty as usize] = handle_sta_named_own_property;
    table[Opcode::StaKeyedProperty as usize] = handle_sta_keyed_property;
    table[Opcode::DefineNamedOwnProperty as usize] = handle_define_named_own_property;
    table[Opcode::DefineKeyedOwnProperty as usize] = handle_define_keyed_own_property;
    table[Opcode::StaInArrayLiteral as usize] = handle_sta_in_array_literal;
    table[Opcode::DefineKeyedOwnPropertyInLiteral as usize] =
        handle_define_keyed_own_property_in_literal;
    table[Opcode::DefineGetterProperty as usize] = handle_define_getter_property;
    table[Opcode::DefineSetterProperty as usize] = handle_define_setter_property;
    table[Opcode::DefineKeyedGetterProperty as usize] = handle_define_keyed_getter_property;
    table[Opcode::DefineKeyedSetterProperty as usize] = handle_define_keyed_setter_property;
    table[Opcode::CollectTypeProfile as usize] = handle_collect_type_profile;
    table[Opcode::Add as usize] = handle_add;
    table[Opcode::Sub as usize] = handle_sub;
    table[Opcode::Mul as usize] = handle_mul;
    table[Opcode::Div as usize] = handle_div;
    table[Opcode::Mod as usize] = handle_mod;
    table[Opcode::Exp as usize] = handle_exp;
    table[Opcode::BitwiseOr as usize] = handle_bitwise_or;
    table[Opcode::BitwiseXor as usize] = handle_bitwise_xor;
    table[Opcode::BitwiseAnd as usize] = handle_bitwise_and;
    table[Opcode::ShiftLeft as usize] = handle_shift_left;
    table[Opcode::ShiftRight as usize] = handle_shift_right;
    table[Opcode::ShiftRightLogical as usize] = handle_shift_right_logical;
    table[Opcode::AddSmi as usize] = handle_add_smi;
    table[Opcode::SubSmi as usize] = handle_sub_smi;
    table[Opcode::MulSmi as usize] = handle_mul_smi;
    table[Opcode::DivSmi as usize] = handle_div_smi;
    table[Opcode::ModSmi as usize] = handle_mod_smi;
    table[Opcode::ExpSmi as usize] = handle_exp_smi;
    table[Opcode::BitwiseOrSmi as usize] = handle_bitwise_or_smi;
    table[Opcode::BitwiseXorSmi as usize] = handle_bitwise_xor_smi;
    table[Opcode::BitwiseAndSmi as usize] = handle_bitwise_and_smi;
    table[Opcode::ShiftLeftSmi as usize] = handle_shift_left_smi;
    table[Opcode::ShiftRightSmi as usize] = handle_shift_right_smi;
    table[Opcode::ShiftRightLogicalSmi as usize] = handle_shift_right_logical_smi;
    table[Opcode::Inc as usize] = handle_inc;
    table[Opcode::Dec as usize] = handle_dec;
    table[Opcode::Negate as usize] = handle_negate;
    table[Opcode::BitwiseNot as usize] = handle_bitwise_not;
    table[Opcode::ToBooleanLogicalNot as usize] = handle_to_boolean_logical_not;
    table[Opcode::LogicalNot as usize] = handle_logical_not;
    table[Opcode::TypeOf as usize] = handle_type_of;
    table[Opcode::DeletePropertyStrict as usize] = handle_delete_property_strict;
    table[Opcode::DeletePropertySloppy as usize] = handle_delete_property_sloppy;
    table[Opcode::TestEqual as usize] = handle_test_equal;
    table[Opcode::TestNotEqual as usize] = handle_test_not_equal;
    table[Opcode::TestEqualStrict as usize] = handle_test_equal_strict;
    table[Opcode::TestLessThan as usize] = handle_test_less_than;
    table[Opcode::TestGreaterThan as usize] = handle_test_greater_than;
    table[Opcode::TestLessThanOrEqual as usize] = handle_test_less_than_or_equal;
    table[Opcode::TestGreaterThanOrEqual as usize] = handle_test_greater_than_or_equal;
    table[Opcode::TestReferenceEqual as usize] = handle_test_reference_equal;
    table[Opcode::TestInstanceOf as usize] = handle_test_instance_of;
    table[Opcode::TestIn as usize] = handle_test_in;
    table[Opcode::TestUndetectable as usize] = handle_test_undetectable;
    table[Opcode::TestNull as usize] = handle_test_null;
    table[Opcode::TestUndefined as usize] = handle_test_undefined;
    table[Opcode::TestTypeOf as usize] = handle_test_type_of;
    table[Opcode::ToName as usize] = handle_to_name;
    table[Opcode::ToNumber as usize] = handle_to_number;
    table[Opcode::ToNumeric as usize] = handle_to_numeric;
    table[Opcode::ToObject as usize] = handle_to_object;
    table[Opcode::ToString as usize] = handle_to_string;
    table[Opcode::ToBoolean as usize] = handle_to_boolean;
    table[Opcode::CallAnyReceiver as usize] = handle_call_any_receiver;
    table[Opcode::CallProperty as usize] = handle_call_property;
    table[Opcode::CallProperty0 as usize] = handle_call_property0;
    table[Opcode::CallProperty1 as usize] = handle_call_property1;
    table[Opcode::CallProperty2 as usize] = handle_call_property2;
    table[Opcode::CallUndefinedReceiver0 as usize] = handle_call_undefined_receiver0;
    table[Opcode::CallUndefinedReceiver1 as usize] = handle_call_undefined_receiver1;
    table[Opcode::CallUndefinedReceiver2 as usize] = handle_call_undefined_receiver2;
    table[Opcode::CallWithSpread as usize] = handle_call_with_spread;
    table[Opcode::CallRuntime as usize] = handle_call_runtime;
    table[Opcode::CallRuntimeForPair as usize] = handle_call_runtime_for_pair;
    table[Opcode::CallJSRuntime as usize] = handle_call_js_runtime;
    table[Opcode::InvokeIntrinsic as usize] = handle_invoke_intrinsic;
    table[Opcode::CallDirectEval as usize] = handle_call_direct_eval;
    table[Opcode::TailCall as usize] = handle_tail_call;
    table[Opcode::Construct as usize] = handle_construct;
    table[Opcode::ConstructWithSpread as usize] = handle_construct_with_spread;
    table[Opcode::ConstructForwardAllArgs as usize] = handle_construct_forward_all_args;
    table[Opcode::GetIterator as usize] = handle_get_iterator;
    table[Opcode::GetAsyncIterator as usize] = handle_get_async_iterator;
    table[Opcode::IteratorNext as usize] = handle_iterator_next;
    table[Opcode::CopyDataProperties as usize] = handle_copy_data_properties;
    table[Opcode::JumpLoop as usize] = handle_jump_loop;
    table[Opcode::Jump as usize] = handle_jump;
    table[Opcode::JumpConstant as usize] = handle_jump_constant;
    table[Opcode::JumpIfTrue as usize] = handle_jump_if_true;
    table[Opcode::JumpIfTrueConstant as usize] = handle_jump_if_true_constant;
    table[Opcode::JumpIfFalse as usize] = handle_jump_if_false;
    table[Opcode::JumpIfFalseConstant as usize] = handle_jump_if_false_constant;
    table[Opcode::JumpIfNull as usize] = handle_jump_if_null;
    table[Opcode::JumpIfNotNull as usize] = handle_jump_if_not_null;
    table[Opcode::JumpIfUndefined as usize] = handle_jump_if_undefined;
    table[Opcode::JumpIfNotUndefined as usize] = handle_jump_if_not_undefined;
    table[Opcode::JumpIfUndefinedOrNull as usize] = handle_jump_if_undefined_or_null;
    table[Opcode::JumpIfJSReceiver as usize] = handle_jump_if_js_receiver;
    table[Opcode::JumpIfForInDone as usize] = handle_jump_if_for_in_done;
    table[Opcode::JumpIfToBooleanTrue as usize] = handle_jump_if_to_boolean_true;
    table[Opcode::JumpIfToBooleanFalse as usize] = handle_jump_if_to_boolean_false;
    table[Opcode::JumpIfToBooleanTrueConstant as usize] = handle_jump_if_to_boolean_true_constant;
    table[Opcode::JumpIfToBooleanFalseConstant as usize] = handle_jump_if_to_boolean_false_constant;
    table[Opcode::JumpIfNullConstant as usize] = handle_jump_if_null_constant;
    table[Opcode::JumpIfNotNullConstant as usize] = handle_jump_if_not_null_constant;
    table[Opcode::JumpIfUndefinedConstant as usize] = handle_jump_if_undefined_constant;
    table[Opcode::JumpIfNotUndefinedConstant as usize] = handle_jump_if_not_undefined_constant;
    table[Opcode::JumpIfUndefinedOrNullConstant as usize] =
        handle_jump_if_undefined_or_null_constant;
    table[Opcode::JumpIfJSReceiverConstant as usize] = handle_jump_if_js_receiver_constant;
    table[Opcode::Return as usize] = handle_return;
    table[Opcode::ThrowReferenceErrorIfHole as usize] = handle_throw_reference_error_if_hole;
    table[Opcode::ThrowSuperNotCalledIfHole as usize] = handle_throw_super_not_called_if_hole;
    table[Opcode::ThrowSuperAlreadyCalledIfNotHole as usize] =
        handle_throw_super_already_called_if_not_hole;
    table[Opcode::Throw as usize] = handle_throw;
    table[Opcode::ReThrow as usize] = handle_throw;
    table[Opcode::SetPendingMessage as usize] = handle_set_pending_message;
    table[Opcode::Debugger as usize] = handle_debugger;
    table[Opcode::CreateClosure as usize] = handle_create_closure;
    table[Opcode::CreateBlockContext as usize] = handle_create_block_context;
    table[Opcode::CreateCatchContext as usize] = handle_create_catch_context;
    table[Opcode::CreateFunctionContext as usize] = handle_create_function_context;
    table[Opcode::CreateEvalContext as usize] = handle_create_eval_context;
    table[Opcode::CreateWithContext as usize] = handle_create_with_context;
    table[Opcode::CreateMappedArguments as usize] = handle_create_mapped_arguments;
    table[Opcode::CreateUnmappedArguments as usize] = handle_create_unmapped_arguments;
    table[Opcode::CreateRestParameter as usize] = handle_create_rest_parameter;
    table[Opcode::CreateRegExpLiteral as usize] = handle_create_reg_exp_literal;
    table[Opcode::CreateArrayLiteral as usize] = handle_create_array_literal;
    table[Opcode::CreateArrayFromIterable as usize] = handle_create_array_from_iterable;
    table[Opcode::CreateEmptyArrayLiteral as usize] = handle_create_empty_array_literal;
    table[Opcode::CreateObjectLiteral as usize] = handle_create_object_literal;
    table[Opcode::CreateEmptyObjectLiteral as usize] = handle_create_empty_object_literal;
    table[Opcode::CreateObjectFromIterable as usize] = handle_create_object_from_iterable;
    table[Opcode::PushContext as usize] = handle_push_context;
    table[Opcode::PopContext as usize] = handle_pop_context;
    table[Opcode::ForInEnumerate as usize] = handle_for_in_enumerate;
    table[Opcode::ForInPrepare as usize] = handle_for_in_prepare;
    table[Opcode::ForInNext as usize] = handle_for_in_next;
    table[Opcode::ForInStep as usize] = handle_for_in_step;
    table[Opcode::GetTemplateObject as usize] = handle_get_template_object;
    table[Opcode::StackCheck as usize] = handle_stack_check;
    table[Opcode::SetExpressionPosition as usize] = handle_stack_check;
    table[Opcode::SetExpressionPositionFromEnd as usize] = handle_stack_check;
    table[Opcode::ResumeGenerator as usize] = handle_resume_generator;
    table[Opcode::GetGeneratorState as usize] = handle_get_generator_state;
    table[Opcode::SuspendGenerator as usize] = handle_suspend_generator;
    table[Opcode::SetGeneratorState as usize] = handle_set_generator_state;
    table[Opcode::SwitchOnGeneratorState as usize] = handle_switch_on_generator_state;
    table[Opcode::CreateClass as usize] = handle_create_class;
    table[Opcode::TestPrivateBrand as usize] = handle_test_private_brand;
    table[Opcode::DefinePrivateBrand as usize] = handle_define_private_brand;
    table[Opcode::LdaModuleVariable as usize] = handle_unimplemented;
    table[Opcode::StaModuleVariable as usize] = handle_unimplemented;
    table[Opcode::LdaImportMeta as usize] = handle_unimplemented;
    table[Opcode::LdaNewTarget as usize] = handle_lda_new_target;
    table[Opcode::GetModuleNamespace as usize] = handle_unimplemented;
    table[Opcode::Wide as usize] = handle_wide;
    table[Opcode::ExtraWide as usize] = handle_wide;
    table[Opcode::Illegal as usize] = handle_unimplemented;
    table
};
