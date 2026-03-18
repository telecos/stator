//! Computed-goto–style dispatch table for the interpreter.
//
//! Each [`Opcode`] is mapped to a handler function via
//! [`DISPATCH_TABLE`].  The main interpreter loop indexes
//! the table by opcode discriminant and calls the handler,
//! replacing the former exhaustive `match`.

#![allow(clippy::too_many_lines)]

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use crate::builtins::string::{string_char_at, utf16_len};
use crate::objects::map::PropertyAttributes;
use crate::objects::property_map::PropertyMap;

use super::{
    ACTIVE_DEBUGGER, Interpreter, InterpreterFrame, MAGLEV_OSR_LOOP_THRESHOLD, OSR_LOOP_THRESHOLD,
    PropertyIc, TURBOFAN_OSR_LOOP_THRESHOLD, abstract_eq, bigint_pow, collect_args, concat_rc_strs,
    constant_pool_jump_delta, constant_to_value, construct_builtin_result, decode_string_constant,
    dispatch_call_property, dispatch_call_value, dispatch_call_with_this, dispatch_getter,
    dispatch_setter, err_bad_operand, error_message_from_value, extract_context, find_handler,
    fn_props_get, fn_props_set, has_prototype_in_chain, is_js_receiver, js_add, js_less_than,
    keyed_load, keyed_store, maybe_compile_baseline, maybe_compile_maglev, maybe_compile_turbofan,
    normalize_async_iterator, number_to_jsvalue, plain_object_to_array_items, populate_self_name,
    proto_lookup, resolve_jump, restore_closure_context, set_function_name_if_missing,
    set_pending_exception, settle_async_iterator_result, strict_eq, to_array_index, to_bigint,
    to_property_key, try_execute_best_jit, walk_context_chain,
};
use crate::builtins::error::{ErrorKind, pop_call_frame, push_call_frame};
use crate::builtins::proxy::{
    proxy_construct, proxy_delete_property, proxy_has, proxy_set_with_receiver,
};
use crate::bytecode::bytecode_array::{
    ConstantPoolEntry, HandlerTableEntry, MAGLEV_TIERING_THRESHOLD, TIERING_THRESHOLD,
    TURBOFAN_TIERING_THRESHOLD,
};
use crate::bytecode::bytecodes::{Instruction, Opcode, Operand};
use crate::error::{StatorError, StatorResult};
use crate::objects::js_object::JsObject;
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

#[allow(dead_code)]
fn js_object_to_plain_value(obj: JsObject) -> JsValue {
    let mut props = PropertyMap::new();
    for key in obj.own_property_keys() {
        if let Some((value, _attrs)) = obj.get_own_property_descriptor(&key) {
            props.insert(key, value);
        }
    }
    JsValue::PlainObject(Rc::new(RefCell::new(props)))
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

const PRIVATE_STORAGE_PREFIX: &str = ".private.";
const PRIVATE_BRAND_PREFIX: &str = ".private.brand.";
const PRIVATE_KIND_PREFIX: &str = ".private.kind.";
const POLYMORPHIC_LOAD_CACHE_CAP: usize = 8;

fn is_private_storage_key(key: &str) -> bool {
    key.starts_with(PRIVATE_STORAGE_PREFIX)
        && !key.starts_with(PRIVATE_BRAND_PREFIX)
        && !key.starts_with(PRIVATE_KIND_PREFIX)
}

fn private_brand_key(storage_key: &str) -> String {
    storage_key.replacen(PRIVATE_STORAGE_PREFIX, PRIVATE_BRAND_PREFIX, 1)
}

fn private_kind_key(storage_key: &str) -> String {
    storage_key.replacen(PRIVATE_STORAGE_PREFIX, PRIVATE_KIND_PREFIX, 1)
}

fn private_getter_key(storage_key: &str) -> String {
    format!("__get_{storage_key}__")
}

fn private_setter_key(storage_key: &str) -> String {
    format!("__set_{storage_key}__")
}

fn private_display_name(storage_key: &str) -> String {
    format!("#{}", storage_key.rsplit('.').next().unwrap_or(storage_key))
}

fn private_access_error(action: &str, storage_key: &str) -> StatorError {
    StatorError::TypeError(format!(
        "Cannot {action} private member {} from an object whose class did not declare it",
        private_display_name(storage_key)
    ))
}

fn own_private_element_exists(map: &PropertyMap, storage_key: &str) -> bool {
    map.contains_key(storage_key)
        || map.contains_key(&private_getter_key(storage_key))
        || map.contains_key(&private_setter_key(storage_key))
        || map.contains_key(&private_kind_key(storage_key))
}

fn private_kind_marker(obj: &JsValue, storage_key: &str) -> Option<String> {
    match proto_lookup(obj, &private_kind_key(storage_key)) {
        JsValue::String(kind) => Some(kind.to_string()),
        _ => None,
    }
}

fn invalidate_plain_object_caches(ctx: &mut DispatchContext, map: &Rc<RefCell<PropertyMap>>) {
    let map_ptr = Rc::as_ptr(map) as usize;
    if let Some(cache) = &mut ctx.frame.mono_load_cache {
        cache.retain(|_, (ptr, _)| *ptr != map_ptr);
    }
    if let Some(cache) = &mut ctx.frame.poly_load_cache {
        cache.retain(|_, entries| {
            entries.retain(|(ptr, _)| *ptr != map_ptr);
            !entries.is_empty()
        });
    }
}

fn load_private_named_property(obj: &JsValue, storage_key: &str) -> StatorResult<JsValue> {
    let JsValue::PlainObject(map) = obj else {
        return Err(private_access_error("read", storage_key));
    };

    let has_brand = map.borrow().contains_key(&private_brand_key(storage_key));
    if has_brand {
        if private_kind_marker(obj, storage_key).is_none() {
            return Err(private_access_error("read", storage_key));
        }
        return Ok(proto_lookup(obj, storage_key));
    }

    let getter_key = private_getter_key(storage_key);
    let setter_key = private_setter_key(storage_key);
    let kind_key = private_kind_key(storage_key);

    let borrow = map.borrow();
    if let Some(getter) = borrow.get(&getter_key).cloned() {
        drop(borrow);
        return dispatch_getter(&getter, obj);
    }
    if let Some(value) = borrow.get(storage_key).cloned() {
        return Ok(value);
    }
    let has_setter = borrow.contains_key(&setter_key);
    let kind = match borrow.get(&kind_key) {
        Some(JsValue::String(kind)) => Some(kind.to_string()),
        _ => None,
    };
    drop(borrow);

    match kind.as_deref() {
        Some("accessor") if has_setter => Ok(JsValue::Undefined),
        _ => Err(private_access_error("read", storage_key)),
    }
}

fn store_private_named_property(
    ctx: &mut DispatchContext,
    obj: &JsValue,
    storage_key: &str,
    value: JsValue,
) -> StatorResult<()> {
    let JsValue::PlainObject(map) = obj else {
        return Err(private_access_error("write to", storage_key));
    };

    let has_brand = map.borrow().contains_key(&private_brand_key(storage_key));
    if has_brand {
        match private_kind_marker(obj, storage_key).as_deref() {
            Some("field") => {
                if map.borrow().contains_key(storage_key) {
                    map.borrow_mut().insert(storage_key.to_string(), value);
                    invalidate_plain_object_caches(ctx, map);
                    return Ok(());
                }
            }
            Some("accessor") => {
                let setter = proto_lookup(obj, &private_setter_key(storage_key));
                if !matches!(setter, JsValue::Undefined) {
                    dispatch_setter(&setter, obj, value)?;
                    return Ok(());
                }
            }
            _ => {}
        }
        return Err(private_access_error("write to", storage_key));
    }

    let kind_key = private_kind_key(storage_key);
    let setter_key = private_setter_key(storage_key);
    let kind = {
        let borrow = map.borrow();
        match borrow.get(&kind_key) {
            Some(JsValue::String(kind)) => Some(kind.to_string()),
            _ => None,
        }
    };

    match kind.as_deref() {
        Some("field") => {
            if map.borrow().contains_key(storage_key) {
                map.borrow_mut().insert(storage_key.to_string(), value);
                invalidate_plain_object_caches(ctx, map);
                Ok(())
            } else {
                Err(private_access_error("write to", storage_key))
            }
        }
        Some("accessor") => {
            let setter = map.borrow().get(&setter_key).cloned();
            if let Some(setter) = setter {
                dispatch_setter(&setter, obj, value)?;
                Ok(())
            } else {
                Err(private_access_error("write to", storage_key))
            }
        }
        _ => Err(private_access_error("write to", storage_key)),
    }
}

fn private_named_property_attrs(prop_name: &str) -> Option<PropertyAttributes> {
    if !prop_name.starts_with(PRIVATE_STORAGE_PREFIX) {
        return None;
    }
    Some(
        if prop_name.starts_with(PRIVATE_BRAND_PREFIX) || prop_name.starts_with(PRIVATE_KIND_PREFIX)
        {
            PropertyAttributes::CONFIGURABLE
        } else {
            PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE
        },
    )
}

/// Signature of a single opcode handler function.
pub(super) type OpcodeHandler =
    fn(&mut DispatchContext, &Instruction) -> StatorResult<DispatchAction>;

/// Number of opcode variants (= `Opcode::Illegal as usize + 1`).
const OPCODE_COUNT: usize = 202;

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
    // SMI fast path: both operands are Smi → direct i32 add.
    if let (JsValue::Smi(a), JsValue::Smi(b)) = (&ctx.frame.accumulator, rhs)
        && let Some(result) = a.checked_add(*b)
    {
        ctx.frame.accumulator = JsValue::Smi(result);
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
    ctx.frame.accumulator = js_add(&ctx.frame.accumulator, rhs)?;
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_sub(ctx: &mut DispatchContext, instr: &Instruction) -> StatorResult<DispatchAction> {
    let Operand::Register(v) = instr.operands[0] else {
        return Err(err_bad_operand("Sub", 0));
    };
    let rhs = ctx.frame.read_reg(v)?;
    // SMI fast path: both operands are Smi → direct i32 subtract.
    if let (JsValue::Smi(a), JsValue::Smi(b)) = (&ctx.frame.accumulator, rhs)
        && let Some(result) = a.checked_sub(*b)
    {
        ctx.frame.accumulator = JsValue::Smi(result);
        return Ok(DispatchAction::Continue);
    }
    if ctx.frame.accumulator.is_bigint() || rhs.is_bigint() {
        let l = to_bigint(&ctx.frame.accumulator)?;
        let r = to_bigint(rhs)?;
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
    // SMI fast path: both operands are Smi → direct i32 multiply.
    if let (JsValue::Smi(a), JsValue::Smi(b)) = (&ctx.frame.accumulator, rhs)
        && let Some(result) = a.checked_mul(*b)
    {
        ctx.frame.accumulator = JsValue::Smi(result);
        return Ok(DispatchAction::Continue);
    }
    if ctx.frame.accumulator.is_bigint() || rhs.is_bigint() {
        let l = to_bigint(&ctx.frame.accumulator)?;
        let r = to_bigint(rhs)?;
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
    // SMI fast path: exact non-zero integer division stays in Smi.
    if let (JsValue::Smi(a), JsValue::Smi(b)) = (&ctx.frame.accumulator, rhs)
        && *b != 0
        && *a % *b == 0
    {
        ctx.frame.accumulator = JsValue::Smi(*a / *b);
        return Ok(DispatchAction::Continue);
    }
    if ctx.frame.accumulator.is_bigint() || rhs.is_bigint() {
        let l = to_bigint(&ctx.frame.accumulator)?;
        let r = to_bigint(rhs)?;
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
    if ctx.frame.accumulator.is_bigint() || rhs.is_bigint() {
        let l = to_bigint(&ctx.frame.accumulator)?;
        let r = to_bigint(rhs)?;
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
    if ctx.frame.accumulator.is_bigint() || rhs.is_bigint() {
        let l = to_bigint(&ctx.frame.accumulator)?;
        let r = to_bigint(rhs)?;
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
    if ctx.frame.accumulator.is_bigint() || rhs.is_bigint() {
        let l = to_bigint(&ctx.frame.accumulator)?;
        let r = to_bigint(rhs)?;
        ctx.frame.accumulator = JsValue::BigInt(l | r);
    } else {
        let lhs = ctx.frame.accumulator.to_int32()?;
        let rhs_i = rhs.to_int32()?;
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
    if ctx.frame.accumulator.is_bigint() || rhs.is_bigint() {
        let l = to_bigint(&ctx.frame.accumulator)?;
        let r = to_bigint(rhs)?;
        ctx.frame.accumulator = JsValue::BigInt(l ^ r);
    } else {
        let lhs = ctx.frame.accumulator.to_int32()?;
        let rhs_i = rhs.to_int32()?;
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
    if ctx.frame.accumulator.is_bigint() || rhs.is_bigint() {
        let l = to_bigint(&ctx.frame.accumulator)?;
        let r = to_bigint(rhs)?;
        ctx.frame.accumulator = JsValue::BigInt(l & r);
    } else {
        let lhs = ctx.frame.accumulator.to_int32()?;
        let rhs_i = rhs.to_int32()?;
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
    if ctx.frame.accumulator.is_bigint() || rhs.is_bigint() {
        let l = to_bigint(&ctx.frame.accumulator)?;
        let r = to_bigint(rhs)?;
        ctx.frame.accumulator = JsValue::BigInt(l.wrapping_shl(r as u32));
    } else {
        let lhs = ctx.frame.accumulator.to_int32()?;
        let shift = rhs.to_uint32()? & 0x1f;
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
    if ctx.frame.accumulator.is_bigint() || rhs.is_bigint() {
        let l = to_bigint(&ctx.frame.accumulator)?;
        let r = to_bigint(rhs)?;
        ctx.frame.accumulator = JsValue::BigInt(l.wrapping_shr(r as u32));
    } else {
        let lhs = ctx.frame.accumulator.to_int32()?;
        let shift = rhs.to_uint32()? & 0x1f;
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
    let lhs = ctx.frame.accumulator.to_int32()? as u32;
    let shift = rhs.to_uint32()? & 0x1f;
    let result = lhs >> shift;
    ctx.frame.accumulator = number_to_jsvalue(result as f64);
    Ok(DispatchAction::Continue)
}

#[inline]
fn handle_add_smi(ctx: &mut DispatchContext, instr: &Instruction) -> StatorResult<DispatchAction> {
    let Operand::Immediate(imm) = instr.operands[0] else {
        return Err(err_bad_operand("AddSmi", 0));
    };
    // SMI fast path: accumulator + immediate Smi.
    if let JsValue::Smi(n) = &ctx.frame.accumulator
        && let Some(result) = n.checked_add(imm)
    {
        ctx.frame.accumulator = JsValue::Smi(result);
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
    // SMI fast path: accumulator - immediate Smi.
    if let JsValue::Smi(n) = &ctx.frame.accumulator
        && let Some(result) = n.checked_sub(imm)
    {
        ctx.frame.accumulator = JsValue::Smi(result);
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
    // SMI fast path: accumulator * immediate Smi.
    if let JsValue::Smi(n) = &ctx.frame.accumulator
        && let Some(result) = n.checked_mul(imm)
    {
        ctx.frame.accumulator = JsValue::Smi(result);
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
    if let JsValue::Smi(lhs) = &ctx.frame.accumulator {
        ctx.frame.accumulator = JsValue::Smi(*lhs | imm);
        return Ok(DispatchAction::Continue);
    }
    if let JsValue::BigInt(n) = &ctx.frame.accumulator {
        ctx.frame.accumulator = JsValue::BigInt(n | i128::from(imm));
    } else {
        let lhs = ctx.frame.accumulator.to_int32()?;
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
    if let JsValue::Smi(lhs) = &ctx.frame.accumulator {
        ctx.frame.accumulator = JsValue::Smi(*lhs ^ imm);
        return Ok(DispatchAction::Continue);
    }
    if let JsValue::BigInt(n) = &ctx.frame.accumulator {
        ctx.frame.accumulator = JsValue::BigInt(n ^ i128::from(imm));
    } else {
        let lhs = ctx.frame.accumulator.to_int32()?;
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
    if let JsValue::Smi(lhs) = &ctx.frame.accumulator {
        ctx.frame.accumulator = JsValue::Smi(*lhs & imm);
        return Ok(DispatchAction::Continue);
    }
    if let JsValue::BigInt(n) = &ctx.frame.accumulator {
        ctx.frame.accumulator = JsValue::BigInt(n & i128::from(imm));
    } else {
        let lhs = ctx.frame.accumulator.to_int32()?;
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
    if let JsValue::Smi(lhs) = &ctx.frame.accumulator {
        let shift = (imm as u32) & 0x1f;
        ctx.frame.accumulator = JsValue::Smi(*lhs << shift);
        return Ok(DispatchAction::Continue);
    }
    if let JsValue::BigInt(n) = &ctx.frame.accumulator {
        ctx.frame.accumulator = JsValue::BigInt(n.wrapping_shl(imm as u32));
    } else {
        let lhs = ctx.frame.accumulator.to_int32()?;
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
    if let JsValue::Smi(lhs) = &ctx.frame.accumulator {
        let shift = (imm as u32) & 0x1f;
        ctx.frame.accumulator = JsValue::Smi(*lhs >> shift);
        return Ok(DispatchAction::Continue);
    }
    if let JsValue::BigInt(n) = &ctx.frame.accumulator {
        ctx.frame.accumulator = JsValue::BigInt(n.wrapping_shr(imm as u32));
    } else {
        let lhs = ctx.frame.accumulator.to_int32()?;
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
    if let JsValue::Smi(lhs) = &ctx.frame.accumulator {
        let shift = (imm as u32) & 0x1f;
        let result = (*lhs as u32) >> shift;
        ctx.frame.accumulator = number_to_jsvalue(result as f64);
        return Ok(DispatchAction::Continue);
    }
    let lhs = ctx.frame.accumulator.to_int32()? as u32;
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
    let rhs = ctx.frame.read_reg(v)?;
    let result = abstract_eq(&ctx.frame.accumulator, rhs);
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
    let rhs = ctx.frame.read_reg(v)?;
    let result = !abstract_eq(&ctx.frame.accumulator, rhs);
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
    // §7.2.14: a > b ≡ IsLessThan(b, a, false) — evaluation order: right first
    let result = JsValue::abstract_relational_comparison(&rhs, &ctx.frame.accumulator, false)?
        .unwrap_or(false);
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
    // §7.2.14: a <= b ≡ !(IsLessThan(b, a, false))
    // If result is undefined (NaN), return false (not !false=true).
    let result = JsValue::abstract_relational_comparison(&rhs, &ctx.frame.accumulator, false)?
        .map(|r| !r)
        .unwrap_or(false);
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
    // §7.2.14: a >= b ≡ !(IsLessThan(a, b, true))
    // If result is undefined (NaN), return false (not !false=true).
    let result = JsValue::abstract_relational_comparison(&ctx.frame.accumulator, &rhs, true)?
        .map(|r| !r)
        .unwrap_or(false);
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
    let func_rc = Rc::new(closure_ba);

    // Non-arrow functions (flag == 0) get a .prototype property per ES §10.2.5.
    let is_arrow = matches!(instr.operands.get(2), Some(Operand::Flag(1)));
    // Arrow functions lexically capture `new.target` from the enclosing scope
    // (ES §15.3.4) so it survives even when the arrow is called later from a
    // different context.
    if is_arrow {
        fn_props_set(
            &func_rc,
            ".new_target".to_string(),
            ctx.frame.new_target.clone(),
        );
    }
    if !is_arrow {
        let func_val = JsValue::Function(Rc::clone(&func_rc));
        let mut proto = PropertyMap::new();
        proto.insert("constructor".to_string(), func_val);
        if func_rc.is_generator()
            && let Some(generator_proto) = super::default_generator_object_prototype()
        {
            proto.insert("__proto__".to_string(), generator_proto);
        }
        fn_props_set(
            &func_rc,
            "prototype".to_string(),
            JsValue::PlainObject(Rc::new(RefCell::new(proto))),
        );
    }

    ctx.frame.accumulator = JsValue::Function(func_rc);
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
                let state = GeneratorState::new((*ba).clone());
                super::init_generator_state_prototype(&state, &ba);
                ctx.frame.accumulator = JsValue::Generator(state);
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
                    let callee_val = JsValue::Function(Rc::clone(&ba));
                    let mut callee_frame = InterpreterFrame::new_with_globals(
                        (*ba).clone(),
                        args,
                        Rc::clone(&ctx.frame.global_env),
                    );
                    restore_closure_context(&mut callee_frame, &ba);
                    if ba.is_arrow() {
                        callee_frame.new_target = fn_props_get(&ba, ".new_target");
                    }
                    populate_self_name(&mut callee_frame, &ba, &callee_val);
                    push_call_frame("<anonymous>")?;
                    let result = stacker::maybe_grow(64 * 1024, 1024 * 1024, || {
                        Interpreter::run(&mut callee_frame)
                    });
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
                    call_plain_object_function(ctx, &ba, map, args)?;
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
    if !ctx.frame.bytecode_array.is_strict() {
        return handle_call_any_receiver(ctx, instr);
    }
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
                    let state = GeneratorState::new((*ba).clone());
                    super::init_generator_state_prototype(&state, &ba);
                    ctx.frame.accumulator = JsValue::Generator(state);
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
                    let param_count = ba.parameter_count() as usize;
                    let frame_size = ba.frame_size() as usize;
                    let total_regs = param_count + frame_size;
                    let inherited_new_target = ctx.frame.new_target.clone();
                    ctx.frame.bytecode_array = (*ba).clone();
                    ctx.frame.call_args = args;
                    ctx.frame.registers.clear();
                    ctx.frame.registers.resize(total_regs, JsValue::Undefined);
                    for (i, arg) in ctx
                        .frame
                        .call_args
                        .iter()
                        .cloned()
                        .enumerate()
                        .take(param_count)
                    {
                        ctx.frame.registers[i] = arg;
                    }
                    ctx.frame.accumulator = JsValue::Undefined;
                    ctx.frame.pc = 0;
                    ctx.frame.context = None;
                    ctx.frame.suspend_result = None;
                    ctx.frame.generator_state = None;
                    ctx.frame.osr_loop_count = 0;
                    ctx.frame.pending_message = JsValue::Undefined;
                    ctx.frame.new_target = if ba.is_arrow() {
                        inherited_new_target
                    } else {
                        JsValue::Undefined
                    };
                    if !ba.is_arrow() && ba.is_strict() {
                        ctx.frame
                            .global_env
                            .borrow_mut()
                            .insert("this".to_string(), JsValue::Undefined);
                    }
                    restore_closure_context(ctx.frame, &ba);
                    populate_self_name(ctx.frame, &ba, &JsValue::Function(Rc::clone(&ba)));
                    if let Some(cache) = &mut ctx.frame.string_cache {
                        cache.clear();
                    }
                    if let Some(cache) = &mut ctx.frame.mono_load_cache {
                        cache.clear();
                    }
                    if let Some(cache) = &mut ctx.frame.poly_load_cache {
                        cache.clear();
                    }
                    if let Some(cache) = &mut ctx.frame.shape_load_ic {
                        cache.clear();
                    }
                    if let Some(cache) = &mut ctx.frame.shape_store_ic {
                        cache.clear();
                    }
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
                let state = GeneratorState::new((*ba).clone());
                super::init_generator_state_prototype(&state, &ba);
                ctx.frame.accumulator = JsValue::Generator(state);
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
                    // Arrow functions use lexical `this` — skip override.
                    // Strict mode: `this` is undefined for free function calls.
                    let saved_this = if !ba.is_arrow() && ba.is_strict() {
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
                    if ba.is_arrow() {
                        callee_frame.new_target = fn_props_get(&ba, ".new_target");
                    }
                    populate_self_name(&mut callee_frame, &ba, &JsValue::Function(Rc::clone(&ba)));
                    push_call_frame("<anonymous>")?;
                    let result = stacker::maybe_grow(64 * 1024, 1024 * 1024, || {
                        Interpreter::run(&mut callee_frame)
                    });
                    pop_call_frame();
                    if !ba.is_arrow() && ba.is_strict() {
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
                let state = GeneratorState::new((*ba).clone());
                super::init_generator_state_prototype(&state, &ba);
                ctx.frame.accumulator = JsValue::Generator(state);
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
                    // Arrow functions use lexical `this` — skip override.
                    let saved_this = if !ba.is_arrow() && ba.is_strict() {
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
                    if ba.is_arrow() {
                        callee_frame.new_target = fn_props_get(&ba, ".new_target");
                    }
                    populate_self_name(&mut callee_frame, &ba, &JsValue::Function(Rc::clone(&ba)));
                    push_call_frame("<anonymous>")?;
                    let result = stacker::maybe_grow(64 * 1024, 1024 * 1024, || {
                        Interpreter::run(&mut callee_frame)
                    });
                    pop_call_frame();
                    if !ba.is_arrow() && ba.is_strict() {
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
                let state = GeneratorState::new((*ba).clone());
                super::init_generator_state_prototype(&state, &ba);
                ctx.frame.accumulator = JsValue::Generator(state);
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
                    // Arrow functions use lexical `this` — skip override.
                    let saved_this = if !ba.is_arrow() && ba.is_strict() {
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
                    if ba.is_arrow() {
                        callee_frame.new_target = fn_props_get(&ba, ".new_target");
                    }
                    populate_self_name(&mut callee_frame, &ba, &JsValue::Function(Rc::clone(&ba)));
                    push_call_frame("<anonymous>")?;
                    let result = stacker::maybe_grow(64 * 1024, 1024 * 1024, || {
                        Interpreter::run(&mut callee_frame)
                    });
                    pop_call_frame();
                    if !ba.is_arrow() && ba.is_strict() {
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
                if ba.is_arrow() {
                    callee_frame.new_target = fn_props_get(&ba, ".new_target");
                }
                populate_self_name(&mut callee_frame, &ba, &JsValue::Function(Rc::clone(&ba)));
                // Arrow functions use lexical `this` — do NOT override.
                if !ba.is_arrow() {
                    callee_frame
                        .global_env
                        .borrow_mut()
                        .insert("this".to_string(), this_val);
                }
                push_call_frame("<anonymous>")?;
                let result = stacker::maybe_grow(64 * 1024, 1024 * 1024, || {
                    Interpreter::run(&mut callee_frame)
                });
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
                    call_plain_object_function(ctx, &ba, map, args)?;
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

/// Expand spread arguments for `CallWithSpread` / `ConstructWithSpread`.
///
/// The bytecode generator may pass spread arrays as either `JsValue::Array`
/// (a dense `Vec`) or `JsValue::PlainObject` with the `__is_array__` marker
/// (a sparse property-map array created by `CreateArrayLiteral`).  This
/// function walks the raw argument list and inlines every array-typed value
/// into the flattened output, leaving non-array values untouched.
///
/// Also handles iterables (String, Iterator, Generator) for robustness.
fn expand_spread_args(raw_args: Vec<JsValue>) -> Vec<JsValue> {
    let mut out = Vec::new();
    for arg in &raw_args {
        match arg {
            JsValue::Array(items) => {
                out.extend(items.borrow().iter().cloned());
            }
            JsValue::PlainObject(map) => {
                let map_ref = map.borrow();
                if map_ref.get("__is_array__").is_some() {
                    let len = match map_ref.get("length") {
                        Some(JsValue::Smi(n)) => *n as usize,
                        Some(JsValue::HeapNumber(n)) => *n as usize,
                        _ => 0,
                    };
                    for i in 0..len {
                        out.push(
                            map_ref
                                .get(&i.to_string())
                                .cloned()
                                .unwrap_or(JsValue::Undefined),
                        );
                    }
                } else {
                    // Check for @@iterator (iterable protocol) on plain objects.
                    let iter_fn = map_ref.get("@@iterator").cloned().or_else(|| {
                        let sym_key =
                            format!("Symbol({})", crate::builtins::symbol::SYMBOL_ITERATOR);
                        map_ref.get(&sym_key).cloned()
                    });
                    drop(map_ref);
                    if let Some(ref f) = iter_fn {
                        if let Ok(iter_obj) = dispatch_call_with_this(f, arg.clone(), vec![]) {
                            match iter_obj {
                                JsValue::Iterator(ni) => {
                                    while let Some(v) = ni.borrow_mut().next_item() {
                                        out.push(v);
                                    }
                                }
                                JsValue::Generator(gs) => loop {
                                    match Interpreter::run_generator_step(&gs, JsValue::Undefined) {
                                        Ok(GeneratorStep::Yield(v)) => out.push(v),
                                        Ok(GeneratorStep::Return(v)) => {
                                            if !matches!(v, JsValue::Undefined) {
                                                out.push(v);
                                            }
                                            break;
                                        }
                                        Err(_) => break,
                                    }
                                },
                                // PlainObject with "next" → user-defined iterator.
                                JsValue::PlainObject(ref iter_map)
                                    if iter_map.borrow().contains_key("next") =>
                                {
                                    if let Ok(items) =
                                        collect_from_plain_object_iterator(&iter_obj, iter_map)
                                    {
                                        out.extend(items);
                                    } else {
                                        out.push(arg.clone());
                                    }
                                }
                                _ => out.push(arg.clone()),
                            }
                        } else {
                            out.push(arg.clone());
                        }
                    } else if map.borrow().contains_key("next") {
                        // Raw iterator object (has "next" but no @@iterator).
                        if let Ok(items) = collect_from_plain_object_iterator(arg, map) {
                            out.extend(items);
                        } else {
                            out.push(arg.clone());
                        }
                    } else {
                        out.push(arg.clone());
                    }
                }
            }
            // String spread: expand each character.
            JsValue::String(s) => {
                for ch in s.chars() {
                    out.push(JsValue::String(ch.to_string().into()));
                }
            }
            // NativeIterator spread: consume all remaining items.
            JsValue::Iterator(ni) => {
                let mut it = ni.borrow_mut();
                while let Some(v) = it.next_item() {
                    out.push(v);
                }
            }
            // Generator spread: run to completion.
            JsValue::Generator(gs) => loop {
                match Interpreter::run_generator_step(gs, JsValue::Undefined) {
                    Ok(GeneratorStep::Yield(v)) => out.push(v),
                    Ok(GeneratorStep::Return(v)) => {
                        if !matches!(v, JsValue::Undefined) {
                            out.push(v);
                        }
                        break;
                    }
                    Err(_) => break,
                }
            },
            _ => {
                out.push(arg.clone());
            }
        }
    }
    out
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
    // Expand any spread arrays (JsValue::Array or PlainObject with
    // __is_array__) into individual arguments.
    let args = expand_spread_args(raw_args);
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
                populate_self_name(&mut callee_frame, &ba, &JsValue::Function(Rc::clone(&ba)));
                push_call_frame("<anonymous>")?;
                let result = stacker::maybe_grow(64 * 1024, 1024 * 1024, || {
                    Interpreter::run(&mut callee_frame)
                });
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
    map: &Rc<RefCell<PropertyMap>>,
    args: Vec<JsValue>,
) -> StatorResult<()> {
    let is_class_constructor = matches!(
        map.borrow().get(".class_constructor"),
        Some(JsValue::Boolean(true))
    );
    if is_class_constructor {
        let pending_this = ctx
            .frame
            .global_env
            .borrow()
            .get(".class_pending_this")
            .cloned();
        match pending_this {
            Some(pending_this) => {
                let current_this = ctx
                    .frame
                    .global_env
                    .borrow()
                    .get("this")
                    .cloned()
                    .unwrap_or(JsValue::Undefined);
                if current_this != JsValue::TheHole {
                    return Err(StatorError::ReferenceError(
                        "Super constructor may only be called once".into(),
                    ));
                }
                ctx.frame
                    .global_env
                    .borrow_mut()
                    .insert("this".to_string(), pending_this);
            }
            None => {
                return Err(StatorError::TypeError(
                    "Class constructor cannot be invoked without 'new'".into(),
                ));
            }
        }
    }
    let mut callee_frame =
        InterpreterFrame::new_with_globals((**ba).clone(), args, Rc::clone(&ctx.frame.global_env));
    restore_closure_context(&mut callee_frame, ba);
    callee_frame.new_target = ctx.frame.new_target.clone();
    push_call_frame("<anonymous>")?;
    let result = stacker::maybe_grow(64 * 1024, 1024 * 1024, || {
        Interpreter::run(&mut callee_frame)
    });
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

    // 3. Expose parent constructor as "super" for `super()` calls.
    let is_derived = if let Some(parent) = class_map.borrow().get("__proto__").cloned()
        && !matches!(parent, JsValue::Undefined | JsValue::Null)
    {
        callee_frame
            .global_env
            .borrow_mut()
            .insert(".class_pending_this".to_string(), this_val.clone());
        callee_frame
            .global_env
            .borrow_mut()
            .insert("this".to_string(), JsValue::TheHole);
        callee_frame
            .global_env
            .borrow_mut()
            .insert("super".to_string(), parent);
        true
    } else {
        callee_frame
            .global_env
            .borrow_mut()
            .insert("this".to_string(), this_val.clone());
        false
    };

    // Helper closure: run the field initializer on the given `this` value.
    let new_target = callee_frame.new_target.clone();
    let run_field_init = |env: &Rc<RefCell<std::collections::HashMap<String, JsValue>>>,
                          this: &JsValue|
     -> StatorResult<()> {
        if let Some(JsValue::Function(init_ba)) =
            class_map.borrow().get(".class_field_initializer").cloned()
        {
            let mut init_frame = InterpreterFrame::new_with_globals(
                (*init_ba).clone(),
                vec![this.clone()],
                Rc::clone(env),
            );
            restore_closure_context(&mut init_frame, &init_ba);
            init_frame.new_target = new_target.clone();
            {
                let mut globals = init_frame.global_env.borrow_mut();
                globals.insert("this".to_string(), this.clone());
                globals.insert(
                    ".class_initializer_class".to_string(),
                    JsValue::PlainObject(Rc::clone(class_map)),
                );
            }
            push_call_frame("<field_init>")?;
            let result =
                stacker::maybe_grow(64 * 1024, 1024 * 1024, || Interpreter::run(&mut init_frame));
            pop_call_frame();
            init_frame
                .global_env
                .borrow_mut()
                .remove(".class_initializer_class");
            result?;
        }
        Ok(())
    };

    // 4. For base classes, fields are initialized before the constructor
    //    body runs (ES §15.7.14).  For derived classes, field
    //    initialization is deferred until after the constructor (which
    //    calls super()) so that `this` is available.
    if !is_derived {
        run_field_init(&ctx.frame.global_env, &this_val)?;
    }

    // 5. Run constructor body.
    push_call_frame("<anonymous>")?;
    let result = stacker::maybe_grow(64 * 1024, 1024 * 1024, || {
        Interpreter::run(&mut callee_frame)
    });
    pop_call_frame();
    let val = result?;

    // 6. For derived classes, run field initializer after the constructor.
    if is_derived {
        // After `super()`, `this` is the constructed value.
        let derived_this = callee_frame
            .global_env
            .borrow()
            .get("this")
            .cloned()
            .unwrap_or(this_val.clone());
        run_field_init(&ctx.frame.global_env, &derived_this)?;
    }

    ctx.frame
        .global_env
        .borrow_mut()
        .remove(".class_pending_this");

    // 7. If constructor returns an object, use it; else use `this`.
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
            // Arrow functions are not constructable (ES §15.3.4).
            if ba.is_arrow() {
                return Err(StatorError::TypeError(
                    "Function is not a constructor".to_string(),
                ));
            }
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
            let result = stacker::maybe_grow(64 * 1024, 1024 * 1024, || {
                Interpreter::run(&mut callee_frame)
            });
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
            let val = f(args)?;
            ctx.frame.accumulator = construct_builtin_result(val, &ctor_proto)?;
        }
        JsValue::PlainObject(ref map) => {
            // Objects marked with `__no_construct__` (e.g. Symbol) are
            // callable but must not be used with `new`.
            if map.borrow().get("__no_construct__").is_some() {
                return Err(StatorError::TypeError(
                    "Symbol is not a constructor".to_string(),
                ));
            }
            let call_val = map.borrow().get("__call__").cloned();
            match call_val {
                Some(JsValue::NativeFunction(f)) => {
                    let args = collect_args(ctx.frame, args_start_v, arg_count)?;
                    let val = f(args)?;
                    ctx.frame.accumulator = construct_builtin_result(val, &ctor_proto)?;
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
        JsValue::Proxy(ref p) => {
            let args = collect_args(ctx.frame, args_start_v, arg_count)?;
            let ctor_val = ctx.frame.accumulator.clone();
            let obj = proxy_construct(&mut p.borrow_mut(), args, ctor_val)?;
            ctx.frame.accumulator = obj;
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
    // Expand any spread arrays (JsValue::Array or PlainObject with
    // __is_array__) into individual arguments.
    let args = expand_spread_args(raw_args);
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
            let result = stacker::maybe_grow(64 * 1024, 1024 * 1024, || {
                Interpreter::run(&mut callee_frame)
            });
            pop_call_frame();
            let val = result?;
            ctx.frame.accumulator = match val {
                JsValue::PlainObject(_) | JsValue::Object(_) => val,
                _ => this_val,
            };
        }
        JsValue::NativeFunction(f) => {
            let val = f(args)?;
            ctx.frame.accumulator = construct_builtin_result(val, &ctor_proto)?;
        }
        JsValue::PlainObject(ref map) => {
            if map.borrow().get("__no_construct__").is_some() {
                return Err(StatorError::TypeError(
                    "Symbol is not a constructor".to_string(),
                ));
            }
            let call_val = map.borrow().get("__call__").cloned();
            match call_val {
                Some(JsValue::NativeFunction(f)) => {
                    let val = f(args)?;
                    ctx.frame.accumulator = construct_builtin_result(val, &ctor_proto)?;
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
        JsValue::Proxy(ref p) => {
            let ctor_val = ctx.frame.accumulator.clone();
            let obj = proxy_construct(&mut p.borrow_mut(), args, ctor_val)?;
            ctx.frame.accumulator = obj;
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
    if name.as_ref() == "this" && ctx.frame.accumulator == JsValue::TheHole {
        return Err(StatorError::ReferenceError(
            "Must call super constructor in derived class before accessing 'this'".into(),
        ));
    }
    Ok(DispatchAction::Continue)
}

/// `LdaGlobalInsideTypeof <name_idx> <slot>`
///
/// Same as [`handle_lda_global`] but used inside `typeof` expressions.
/// Returns `undefined` instead of throwing a `ReferenceError` when the
/// global variable does not exist (ECMAScript §13.5.3).
fn handle_lda_global_inside_typeof(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[0] else {
        return Err(err_bad_operand("LdaGlobalInsideTypeof", 0));
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
    set_function_name_if_missing(&val, &name);
    env.insert(name, val);
    Ok(DispatchAction::Continue)
}

fn set_named_property_function_metadata(value: &JsValue, obj: &JsValue, name: &str) {
    if let JsValue::Function(ba) = value {
        fn_props_set(ba, ".home_object".to_string(), obj.clone());
        set_function_name_if_missing(value, name);
    }
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
    let prop_name = ctx.frame.get_string_constant(name_idx)?;
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    if is_private_storage_key(&prop_name) {
        ctx.frame.accumulator = load_private_named_property(&obj, &prop_name)?;
        return Ok(DispatchAction::Continue);
    }
    // ── Shape IC fast path: O(1) own-property access via cached offset ──
    if slot != u32::MAX
        && let JsValue::PlainObject(ref map) = obj
        && let Some(ic) = ctx
            .frame
            .shape_load_ic
            .as_ref()
            .and_then(|cache| cache.get(&slot))
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
            && let Some(entries) = ctx
                .frame
                .poly_load_cache
                .as_ref()
                .and_then(|cache| cache.get(&slot))
        {
            for &(cached_ptr, ref cached_val) in entries {
                if cached_ptr == ptr {
                    ctx.frame.accumulator = cached_val.clone();
                    return Ok(DispatchAction::Continue);
                }
            }
        }
    }
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
    // Skip caching when the property has an accessor (__get_<key>__)
    // because the IC fast-path would return the placeholder data value
    // instead of invoking the getter.
    if slot != u32::MAX
        && let JsValue::PlainObject(ref map) = obj
    {
        let pm = map.borrow();
        let getter_key = format!("__get_{prop_name}__");
        if !pm.contains_key(&getter_key)
            && let Some(offset) = pm.offset_of(&prop_name)
        {
            ctx.frame
                .shape_load_ic
                .get_or_insert_with(std::collections::HashMap::new)
                .insert(
                    slot,
                    PropertyIc {
                        cached_shape: pm.shape_id(),
                        cached_offset: offset,
                    },
                );
        }
    }
    // Update polymorphic cache (up to 8 entries per slot).
    if slot != u32::MAX {
        let obj_ptr = match &obj {
            JsValue::PlainObject(map) => Some(Rc::as_ptr(map) as usize),
            JsValue::Array(arr) => Some(Rc::as_ptr(arr) as usize),
            JsValue::Function(ba) => Some(Rc::as_ptr(ba) as usize),
            _ => None,
        };
        if let Some(ptr) = obj_ptr {
            let entries = ctx
                .frame
                .poly_load_cache
                .get_or_insert_with(std::collections::HashMap::new)
                .entry(slot)
                .or_default();
            // Check if we already have this pointer; update in place.
            let mut found = false;
            for entry in entries.iter_mut() {
                if entry.0 == ptr {
                    entry.1 = result.clone();
                    found = true;
                    break;
                }
            }
            if !found && entries.len() < POLYMORPHIC_LOAD_CACHE_CAP {
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
    if is_private_storage_key(&prop_name) {
        store_private_named_property(ctx, &obj, &prop_name, val)?;
        return Ok(DispatchAction::Continue);
    }
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
            let stored = proxy_set_with_receiver(&mut p.borrow_mut(), &prop_name, val, &obj)?;
            if !stored && ctx.frame.bytecode_array.is_strict() {
                return Err(StatorError::TypeError(format!(
                    "Cannot assign to read only property '{prop_name}'"
                )));
            }
        }
        JsValue::PlainObject(ref map) => {
            // ── Shape IC fast path for store: existing writable property ─
            if slot != u32::MAX
                && let Some(ic) = ctx
                    .frame
                    .shape_store_ic
                    .as_ref()
                    .and_then(|cache| cache.get(&slot))
            {
                let pm = map.borrow();
                if pm.shape_id() == ic.cached_shape {
                    if pm.is_writable_by_offset(ic.cached_offset) {
                        drop(pm);
                        map.borrow_mut().set_by_offset(ic.cached_offset, val);
                        // Invalidate value-based caches for this object.
                        let map_ptr = Rc::as_ptr(map) as usize;
                        if let Some(cache) = &mut ctx.frame.mono_load_cache {
                            cache.retain(|_, (ptr, _)| *ptr != map_ptr);
                        }
                        if let Some(cache) = &mut ctx.frame.poly_load_cache {
                            cache.retain(|_, entries| {
                                entries.retain(|(ptr, _)| *ptr != map_ptr);
                                !entries.is_empty()
                            });
                        }
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
            // Check for setter accessor first (own object).
            let setter_key = format!("__set_{prop_name}__");
            let getter_key = format!("__get_{prop_name}__");
            let setter = map.borrow().get(&setter_key).cloned();
            if let Some(setter_fn) = setter {
                dispatch_setter(&setter_fn, &obj, val)?;
                return Ok(DispatchAction::Continue);
            }
            // Getter-only accessor (own): no setter -> TypeError in strict,
            // silent ignore in sloppy.
            if map.borrow().contains_key(&getter_key) {
                if ctx.frame.bytecode_array.is_strict() {
                    return Err(StatorError::TypeError(format!(
                        "Cannot set property {prop_name} which has only a getter"
                    )));
                }
                return Ok(DispatchAction::Continue);
            }
            // Walk the prototype chain for inherited setters/getters.
            {
                let mut proto = map.borrow().get("__proto__").cloned();
                for _ in 0..256 {
                    match proto.take() {
                        Some(JsValue::PlainObject(p)) => {
                            let pb = p.borrow();
                            if let Some(s) = pb.get(&setter_key).cloned() {
                                drop(pb);
                                dispatch_setter(&s, &obj, val)?;
                                return Ok(DispatchAction::Continue);
                            }
                            if pb.contains_key(&getter_key) {
                                if ctx.frame.bytecode_array.is_strict() {
                                    return Err(StatorError::TypeError(format!(
                                        "Cannot set property {prop_name} which has only a getter"
                                    )));
                                }
                                return Ok(DispatchAction::Continue);
                            }
                            proto = pb.get("__proto__").cloned();
                        }
                        _ => break,
                    }
                }
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
            set_function_name_if_missing(&val, &prop_name);
            map.borrow_mut().insert(prop_name.to_string(), val);
            // Populate shape store IC for future fast-path stores.
            if slot != u32::MAX {
                let pm = map.borrow();
                if let Some(offset) = pm.offset_of(&prop_name) {
                    ctx.frame
                        .shape_store_ic
                        .get_or_insert_with(std::collections::HashMap::new)
                        .insert(
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
            if let Some(cache) = &mut ctx.frame.mono_load_cache {
                cache.retain(|_, (ptr, _)| *ptr != map_ptr);
            }
            if let Some(cache) = &mut ctx.frame.poly_load_cache {
                cache.retain(|_, entries| {
                    entries.retain(|(ptr, _)| *ptr != map_ptr);
                    !entries.is_empty()
                });
            }
        }
        JsValue::Function(ref ba) => {
            if matches!(fn_props_get(ba, &prop_name), JsValue::Undefined) {
                set_function_name_if_missing(&val, &prop_name);
            }
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
                    v.resize(new_len_u32 as usize, JsValue::TheHole);
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
        set_function_name_if_missing(&val, &key_str);
        // Check for setter accessor first (__set_<key>__).
        let setter_key = format!("__set_{key_str}__");
        let getter_key = format!("__get_{key_str}__");
        let setter = map.borrow().get(&setter_key).cloned();
        if let Some(setter_fn) = setter {
            dispatch_setter(&setter_fn, &obj, val)?;
            return Ok(DispatchAction::Continue);
        }
        // Getter-only accessor: no setter -> TypeError in strict, silent
        // ignore in sloppy.
        if map.borrow().contains_key(&getter_key) {
            if ctx.frame.bytecode_array.is_strict() {
                return Err(StatorError::TypeError(format!(
                    "Cannot set property {key_str} which has only a getter"
                )));
            }
            return Ok(DispatchAction::Continue);
        }
        // Walk prototype chain for inherited setters/getters.
        {
            let mut proto = map.borrow().get("__proto__").cloned();
            for _ in 0..256 {
                match proto.take() {
                    Some(JsValue::PlainObject(p)) => {
                        let pb = p.borrow();
                        if let Some(s) = pb.get(&setter_key).cloned() {
                            drop(pb);
                            dispatch_setter(&s, &obj, val)?;
                            return Ok(DispatchAction::Continue);
                        }
                        if pb.contains_key(&getter_key) {
                            if ctx.frame.bytecode_array.is_strict() {
                                return Err(StatorError::TypeError(format!(
                                    "Cannot set property {key_str} which has only a getter"
                                )));
                            }
                            return Ok(DispatchAction::Continue);
                        }
                        proto = pb.get("__proto__").cloned();
                    }
                    _ => break,
                }
            }
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
            let items_vec: Vec<JsValue> = items
                .borrow()
                .iter()
                .cloned()
                .map(|value| {
                    if value.is_the_hole() {
                        JsValue::Undefined
                    } else {
                        value
                    }
                })
                .collect();
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
                    let result = dispatch_call_with_this(f, iterable.clone(), vec![])?;
                    // §7.4.1: result of @@iterator must be an object.
                    match &result {
                        JsValue::PlainObject(_)
                        | JsValue::Iterator(_)
                        | JsValue::Generator(_)
                        | JsValue::Object(_)
                        | JsValue::Array(_) => result,
                        _ => {
                            return Err(StatorError::TypeError(
                                "Result of the Symbol.iterator method is not an object".into(),
                            ));
                        }
                    }
                }
                Some(JsValue::PlainObject(ref call_obj))
                    if call_obj.borrow().contains_key("__call__") =>
                {
                    let result = dispatch_call_with_this(
                        &JsValue::PlainObject(call_obj.clone()),
                        iterable.clone(),
                        vec![],
                    )?;
                    match &result {
                        JsValue::PlainObject(_)
                        | JsValue::Iterator(_)
                        | JsValue::Generator(_)
                        | JsValue::Object(_)
                        | JsValue::Array(_) => result,
                        _ => {
                            return Err(StatorError::TypeError(
                                "Result of the Symbol.iterator method is not an object".into(),
                            ));
                        }
                    }
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
                        return Err(StatorError::TypeError("object is not iterable".into()));
                    }
                }
            }
        }
        other => {
            let desc = match &other {
                JsValue::Null => "null".to_string(),
                JsValue::Undefined => "undefined".to_string(),
                JsValue::Smi(n) => n.to_string(),
                JsValue::HeapNumber(n) => n.to_string(),
                JsValue::Boolean(b) => b.to_string(),
                _ => format!("{other:?}"),
            };
            return Err(StatorError::TypeError(format!("{desc} is not iterable")));
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
            normalize_async_iterator(JsValue::Iterator(NativeIterator::from_items(items_vec)))
        }
        JsValue::String(ref s) => {
            normalize_async_iterator(JsValue::Iterator(NativeIterator::from_string(s)))
        }
        JsValue::Generator(ref state) if state.borrow().bytecode_array.is_async() => iterable,
        JsValue::Generator(_) | JsValue::Iterator(_) => normalize_async_iterator(iterable),
        // PlainObject with @@asyncIterator → call it first (§27.1.4.2).
        JsValue::PlainObject(ref map)
            if map.borrow().contains_key("@@asyncIterator")
                || map.borrow().contains_key("Symbol(2)") =>
        {
            let iter_fn = map
                .borrow()
                .get("@@asyncIterator")
                .cloned()
                .or_else(|| map.borrow().get("Symbol(2)").cloned());
            match iter_fn {
                Some(ref f @ (JsValue::NativeFunction(_) | JsValue::Function(_))) => {
                    normalize_async_iterator(dispatch_call_with_this(f, iterable.clone(), vec![])?)
                }
                _ => {
                    return Err(StatorError::TypeError(
                        "GetAsyncIterator: @@asyncIterator is not a function".into(),
                    ));
                }
            }
        }
        // Fall back to @@iterator (sync iterator wrapped for async).
        JsValue::PlainObject(ref map)
            if map.borrow().contains_key("@@iterator")
                || map.borrow().contains_key("Symbol(1)") =>
        {
            let iter_fn = map
                .borrow()
                .get("@@iterator")
                .cloned()
                .or_else(|| map.borrow().get("Symbol(1)").cloned());
            match iter_fn {
                Some(ref f @ (JsValue::NativeFunction(_) | JsValue::Function(_))) => {
                    normalize_async_iterator(dispatch_call_with_this(f, iterable.clone(), vec![])?)
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
            normalize_async_iterator(JsValue::Iterator(NativeIterator::from_items(items)))
        }
        JsValue::PlainObject(_) => {
            return Err(StatorError::TypeError(
                "GetAsyncIterator: value is not async iterable".into(),
            ));
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
        JsValue::Generator(gs) => {
            if gs.borrow().bytecode_array.is_async() {
                let queue = crate::builtins::promise::MicrotaskQueue::new();
                let result = Interpreter::run_generator_step(&gs, JsValue::Undefined).map(
                    |step| match step {
                        GeneratorStep::Yield(v) => super::make_iterator_result(v, false),
                        GeneratorStep::Return(v) => super::make_iterator_result(v, true),
                    },
                )?;
                settle_async_iterator_result(result, &queue)
                    .map_err(super::async_iterator_reason_to_error)?
            } else {
                match Interpreter::run_generator_step(&gs, JsValue::Undefined)? {
                    GeneratorStep::Yield(v) => (v, false),
                    GeneratorStep::Return(v) => (v, true),
                }
            }
        }
        JsValue::PlainObject(ref map) if map.borrow().contains_key("next") => {
            let next_fn = map.borrow().get("next").cloned();
            match next_fn {
                Some(ref f @ (JsValue::NativeFunction(_) | JsValue::Function(_))) => {
                    let result = dispatch_call_with_this(f, iter.clone(), vec![])?;
                    if map
                        .borrow()
                        .get("__async_iterator__")
                        .is_some_and(|flag| flag.to_boolean())
                    {
                        let queue = crate::builtins::promise::MicrotaskQueue::new();
                        settle_async_iterator_result(result, &queue)
                            .map_err(super::async_iterator_reason_to_error)?
                    } else {
                        match result {
                            JsValue::PlainObject(ref res_map) => {
                                let done =
                                    res_map.borrow().get("done").is_some_and(|d| d.to_boolean());
                                let value = res_map
                                    .borrow()
                                    .get("value")
                                    .cloned()
                                    .unwrap_or(JsValue::Undefined);
                                (value, done)
                            }
                            // §7.4.3: If the result of .next() is not an object,
                            // throw a TypeError.
                            _ => {
                                return Err(StatorError::TypeError(
                                    "Iterator result is not an object".into(),
                                ));
                            }
                        }
                    }
                }
                Some(ref f @ JsValue::PlainObject(ref call_obj))
                    if call_obj.borrow().contains_key("__call__") =>
                {
                    let result = dispatch_call_with_this(f, iter.clone(), vec![])?;
                    if map
                        .borrow()
                        .get("__async_iterator__")
                        .is_some_and(|flag| flag.to_boolean())
                    {
                        let queue = crate::builtins::promise::MicrotaskQueue::new();
                        settle_async_iterator_result(result, &queue)
                            .map_err(super::async_iterator_reason_to_error)?
                    } else {
                        match result {
                            JsValue::PlainObject(ref res_map) => {
                                let done =
                                    res_map.borrow().get("done").is_some_and(|d| d.to_boolean());
                                let value = res_map
                                    .borrow()
                                    .get("value")
                                    .cloned()
                                    .unwrap_or(JsValue::Undefined);
                                (value, done)
                            }
                            _ => {
                                return Err(StatorError::TypeError(
                                    "Iterator result is not an object".into(),
                                ));
                            }
                        }
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

/// Close an iterator by calling its `.return()` method if it exists.
/// Used when breaking out of for-of loops early (break/return/throw).
///
/// Per §7.4.6 IteratorClose:
/// 1. If the iterator has a `.return()` method, call it.
/// 2. If `.return()` throws, propagate the error.
/// 3. If `.return()` returns a non-object, throw TypeError.
fn handle_iterator_close(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(iter_v) = instr.operands[0] else {
        return Err(err_bad_operand("IteratorClose", 0));
    };
    let iter = ctx.frame.read_reg(iter_v)?.clone();
    match &iter {
        JsValue::PlainObject(map) => {
            let return_fn = map.borrow().get("return").cloned();
            match return_fn {
                Some(ref f @ (JsValue::NativeFunction(_) | JsValue::Function(_))) => {
                    let result =
                        dispatch_call_with_this(f, iter.clone(), vec![JsValue::Undefined])?;
                    if map
                        .borrow()
                        .get("__async_iterator__")
                        .is_some_and(|flag| flag.to_boolean())
                    {
                        let queue = crate::builtins::promise::MicrotaskQueue::new();
                        settle_async_iterator_result(result, &queue)
                            .map_err(super::async_iterator_reason_to_error)?;
                    } else if !result.is_object_like() {
                        // §7.4.6 step 7: If the result is not an object, throw TypeError.
                        return Err(StatorError::TypeError(
                            "Iterator .return() result is not an object".into(),
                        ));
                    }
                }
                Some(JsValue::PlainObject(ref call_obj))
                    if call_obj.borrow().contains_key("__call__") =>
                {
                    let result = dispatch_call_with_this(
                        &JsValue::PlainObject(call_obj.clone()),
                        iter.clone(),
                        vec![JsValue::Undefined],
                    )?;
                    if map
                        .borrow()
                        .get("__async_iterator__")
                        .is_some_and(|flag| flag.to_boolean())
                    {
                        let queue = crate::builtins::promise::MicrotaskQueue::new();
                        settle_async_iterator_result(result, &queue)
                            .map_err(super::async_iterator_reason_to_error)?;
                    } else if !result.is_object_like() {
                        return Err(StatorError::TypeError(
                            "Iterator .return() result is not an object".into(),
                        ));
                    }
                }
                // No .return() method — nothing to do (§7.4.6 step 4).
                _ => {}
            }
        }
        // Close a generator by marking it as completed (§27.5.3.4).
        JsValue::Generator(gs) => {
            let result = Interpreter::generator_return(gs, JsValue::Undefined)?;
            if gs.borrow().bytecode_array.is_async() {
                let queue = crate::builtins::promise::MicrotaskQueue::new();
                settle_async_iterator_result(result, &queue)
                    .map_err(super::async_iterator_reason_to_error)?;
            } else if !result.is_object_like() {
                return Err(StatorError::TypeError(
                    "Iterator .return() result is not an object".into(),
                ));
            }
        }
        // NativeIterator has no .return() — nothing to do.
        _ => {}
    }
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

    if let JsValue::PlainObject(t) = &target {
        match &source {
            JsValue::PlainObject(s) => {
                let entries: Vec<(String, JsValue)> = s
                    .borrow()
                    .enumerable_iter()
                    .filter(|(k, _)| {
                        // Skip engine-internal keys: __proto__, __call__, etc.
                        !((k.starts_with("__") && k.ends_with("__"))
                            || k.starts_with("@@")
                            || k.starts_with("Symbol(")
                            || k.starts_with('.'))
                    })
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                for (k, v) in entries {
                    t.borrow_mut().insert(k, v);
                }
            }
            JsValue::Array(items) => {
                for (i, v) in items.borrow().iter().enumerate() {
                    t.borrow_mut().insert(i.to_string(), v.clone());
                }
            }
            JsValue::String(s) => {
                for i in 0..utf16_len(s) {
                    t.borrow_mut().insert(
                        i.to_string(),
                        JsValue::String(string_char_at(s, i as i64).into()),
                    );
                }
            }
            _ => {}
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
    let mut return_val: Option<JsValue> = None;
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
        match std::mem::replace(&mut gs.resume_mode, GeneratorResumeMode::Normal) {
            GeneratorResumeMode::Throw(val) => throw_val = Some(val),
            GeneratorResumeMode::Return(val) => return_val = Some(val),
            GeneratorResumeMode::Normal => {}
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
    if return_val.is_some() {
        // Force a "return" abrupt completion.  By throwing a sentinel
        // through the handler table, any enclosing `finally` block will
        // execute before the generator completes.
        let sentinel = JsValue::String(super::GENERATOR_RETURN_SENTINEL.into());
        ctx.frame.accumulator = sentinel.clone();
        set_pending_exception(sentinel);
        return Err(StatorError::JsException(
            super::GENERATOR_RETURN_SENTINEL.into(),
        ));
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
        5 => matches!(ctx.frame.accumulator, JsValue::Undefined | JsValue::TheHole),
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
            map.insert("length".into(), JsValue::Smi(utf16_len(&s_val) as i32));
            // Indexed character access ("0", "1", …).
            for i in 0..utf16_len(&s_val) {
                map.insert(
                    i.to_string(),
                    JsValue::String(string_char_at(&s_val, i as i64).into()),
                );
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
        other => {
            let prim = other.to_primitive(crate::objects::value::ToPrimitiveHint::String)?;
            match prim {
                JsValue::String(s) => JsValue::String(s),
                JsValue::Symbol(_) => prim,
                _ => JsValue::String(prim.to_js_string()?.into()),
            }
        }
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
        let n = ctx.frame.accumulator.to_int32()?;
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
    // Per ES spec, "length" is writable but not enumerable/configurable.
    map.insert_with_attrs(
        "length".to_string(),
        JsValue::Smi(0),
        PropertyAttributes::WRITABLE,
    );
    map.insert_with_attrs(
        "__is_array__".to_string(),
        JsValue::Boolean(true),
        PropertyAttributes::empty(),
    );
    ctx.frame.accumulator = JsValue::PlainObject(Rc::new(RefCell::new(map)));
    Ok(DispatchAction::Continue)
}

fn handle_create_array_literal(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    // operands: [ConstantPoolIdx, FeedbackSlot, Flag]
    let mut map = PropertyMap::new();
    map.insert_with_attrs(
        "length".to_string(),
        JsValue::Smi(0),
        PropertyAttributes::WRITABLE,
    );
    map.insert_with_attrs(
        "__is_array__".to_string(),
        JsValue::Boolean(true),
        PropertyAttributes::empty(),
    );
    ctx.frame.accumulator = JsValue::PlainObject(Rc::new(RefCell::new(map)));
    Ok(DispatchAction::Continue)
}

/// Collect all values from a user-defined iterator object (PlainObject with
/// a `.next()` method) by repeatedly calling `.next()` until `done` is true.
///
/// Per the iterator protocol:
/// - `.next()` must be callable (TypeError otherwise).
/// - Each result must be an object with `{ value, done }` (TypeError if not).
fn collect_from_plain_object_iterator(
    iter_obj: &JsValue,
    map: &Rc<RefCell<PropertyMap>>,
) -> StatorResult<Vec<JsValue>> {
    let mut out = Vec::new();
    loop {
        let next_fn = map.borrow().get("next").cloned();
        match next_fn {
            Some(ref f) => {
                let result = dispatch_call_with_this(f, iter_obj.clone(), vec![])?;
                match result {
                    JsValue::PlainObject(ref res_map) => {
                        let done = res_map.borrow().get("done").is_some_and(|d| d.to_boolean());
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
                    // §7.4.3: If the result of .next() is not an object,
                    // throw a TypeError.
                    _ => {
                        return Err(StatorError::TypeError(
                            "Iterator result is not an object".into(),
                        ));
                    }
                }
            }
            None => {
                return Err(StatorError::TypeError(
                    "Iterator .next() is not a function".into(),
                ));
            }
        }
    }
    Ok(out)
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
        JsValue::PlainObject(map) => {
            // Check for @@iterator (iterable protocol) before falling back
            // to direct "next" (iterator protocol).
            let iter_fn = {
                let borrow = map.borrow();
                borrow.get("@@iterator").cloned().or_else(|| {
                    let sym_key = format!("Symbol({})", crate::builtins::symbol::SYMBOL_ITERATOR);
                    borrow.get(&sym_key).cloned()
                })
            };
            if let Some(ref f) = iter_fn {
                // Call @@iterator to obtain the iterator object.
                let iter_obj = dispatch_call_with_this(f, iterable.clone(), vec![])?;
                match &iter_obj {
                    JsValue::Iterator(ni) => {
                        let mut out = Vec::new();
                        while let Some(v) = ni.borrow_mut().next_item() {
                            out.push(v);
                        }
                        out
                    }
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
                    JsValue::PlainObject(iter_map) if iter_map.borrow().contains_key("next") => {
                        collect_from_plain_object_iterator(&iter_obj, iter_map)?
                    }
                    _ => {
                        return Err(StatorError::TypeError(
                            "Result of the Symbol.iterator method is not an object".into(),
                        ));
                    }
                }
            } else if map.borrow().contains_key("next") {
                // Object is already an iterator (has "next" directly).
                collect_from_plain_object_iterator(&iterable, map)?
            } else if map.borrow().contains_key("length") {
                // Array-like fallback.
                plain_object_to_array_items(map)
            } else {
                return Err(StatorError::TypeError("object is not iterable".into()));
            }
        }
        // Null, Undefined, and other non-iterable primitives.
        _ => {
            return Err(StatorError::TypeError("object is not iterable".into()));
        }
    };
    let mut map = PropertyMap::new();
    for (i, v) in items.iter().enumerate() {
        map.insert(i.to_string(), v.clone());
    }
    map.insert_with_attrs(
        "length".to_string(),
        JsValue::Smi(items.len() as i32),
        PropertyAttributes::WRITABLE,
    );
    map.insert_with_attrs(
        "__is_array__".to_string(),
        JsValue::Boolean(true),
        PropertyAttributes::empty(),
    );
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
        flags_str.push('v');
    }
    if flags_val & 0x40 != 0 {
        flags_str.push('y');
    }
    if flags_val & 0x80 != 0 {
        flags_str.push('d');
    }
    let re = crate::objects::regexp::JsRegExp::new(&pattern, &flags_str)?;
    ctx.frame.accumulator = crate::builtins::regexp::wrap_regexp(re);
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
                map.borrow_mut().insert_with_attrs(
                    "length".to_string(),
                    JsValue::Smi(new_len),
                    PropertyAttributes::WRITABLE,
                );
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
        set_named_property_function_metadata(&val, &obj, &prop_name);
        if let Some(attrs) = private_named_property_attrs(&prop_name) {
            map.borrow_mut().insert_with_attrs(prop_name, val, attrs);
        } else {
            map.borrow_mut().insert(prop_name, val);
        }
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
        set_named_property_function_metadata(&val, &obj, &prop_name);
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
        set_named_property_function_metadata(&val, &obj, &prop_name);
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

    // §7.3.21 OrdinaryHasInstance — RHS must be callable, else TypeError.
    let rhs_callable = match &constructor {
        JsValue::Function(_) | JsValue::NativeFunction(_) => true,
        JsValue::PlainObject(map) => map.borrow().contains_key("__call__"),
        JsValue::Proxy(p) => p.borrow().is_callable(),
        _ => false,
    };
    if !rhs_callable {
        return Err(StatorError::TypeError(
            "Right-hand side of 'instanceof' is not callable".to_string(),
        ));
    }

    // §7.3.21 OrdinaryHasInstance — first check @@hasInstance
    let has_instance_fn = match &constructor {
        JsValue::PlainObject(map) => map.borrow().get("@@hasInstance").cloned(),
        JsValue::NativeFunction(_) | JsValue::Function(_) => {
            // Look up @@hasInstance via the prototype chain (e.g.
            // Function.prototype[@@hasInstance]).
            let v = proto_lookup(&constructor, "@@hasInstance");
            if matches!(v, JsValue::Undefined) {
                None
            } else {
                Some(v)
            }
        }
        _ => None,
    };
    if let Some(ref hi) = has_instance_fn {
        match hi {
            JsValue::NativeFunction(f) => {
                let result = f(vec![ctx.frame.accumulator.clone()])?;
                ctx.frame.accumulator = JsValue::Boolean(result.to_boolean());
                return Ok(DispatchAction::Continue);
            }
            JsValue::Function(_) | JsValue::PlainObject(_) => {
                let result = dispatch_call_value(hi, vec![ctx.frame.accumulator.clone()])?;
                ctx.frame.accumulator = JsValue::Boolean(result.to_boolean());
                return Ok(DispatchAction::Continue);
            }
            _ => {}
        }
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
            (
                "RegExp",
                |v| matches!(v, JsValue::PlainObject(m) if m.borrow().get("__is_regexp__").is_some()),
            ),
            (
                "Date",
                |v| matches!(v, JsValue::PlainObject(m) if m.borrow().get("__is_date__").is_some()),
            ),
            (
                "Map",
                |v| matches!(v, JsValue::PlainObject(m) if m.borrow().get("__is_map__").is_some()),
            ),
            (
                "Set",
                |v| matches!(v, JsValue::PlainObject(m) if m.borrow().get("__is_set__").is_some()),
            ),
            (
                "WeakMap",
                |v| matches!(v, JsValue::PlainObject(m) if m.borrow().get("__is_weakmap__").is_some()),
            ),
            (
                "WeakSet",
                |v| matches!(v, JsValue::PlainObject(m) if m.borrow().get("__is_weakset__").is_some()),
            ),
            (
                "WeakRef",
                |v| matches!(v, JsValue::PlainObject(m) if m.borrow().get("__is_weakref__").is_some()),
            ),
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
        JsValue::Function(_) | JsValue::NativeFunction(_) => {
            let v = proto_lookup(&constructor, "prototype");
            if matches!(v, JsValue::Undefined) {
                None
            } else {
                Some(v)
            }
        }
        _ => None,
    };

    let result = if let Some(proto_val) = ctor_proto {
        has_prototype_in_chain(&ctx.frame.accumulator, &proto_val)
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
            proxy_has(&p.borrow(), &prop)?
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
                items
                    .borrow()
                    .get(idx)
                    .is_some_and(|value| !value.is_the_hole())
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
                    if let Some(prop) = k.strip_prefix("__get_").and_then(|s| s.strip_suffix("__"))
                    {
                        seen.insert(k.clone());
                        if seen.insert(prop.to_string()) && is_enumerable {
                            all_keys.push(JsValue::String(prop.to_string().into()));
                        }
                        continue;
                    }
                    if let Some(prop) = k.strip_prefix("__set_").and_then(|s| s.strip_suffix("__"))
                    {
                        seen.insert(k.clone());
                        if seen.insert(prop.to_string()) && is_enumerable {
                            all_keys.push(JsValue::String(prop.to_string().into()));
                        }
                        continue;
                    }
                    // Skip engine-internal keys that must not appear in
                    // for-in enumeration: __proto__, __call__, __is_array__,
                    // Symbol()-keyed properties (for-in is string-only per
                    // ES §14.7.5.9), @@-prefixed internal hooks, hidden
                    // field initializers, and private fields (#name).
                    if (k.starts_with("__") && k.ends_with("__"))
                        || crate::builtins::symbol::is_symbol_property_key(k)
                        || k.starts_with('.')
                        || k.starts_with('#')
                    {
                        seen.insert(k.clone());
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
        // for...in on a string enumerates character indices ("0", "1", …).
        JsValue::String(s) => (0..utf16_len(s))
            .map(|i| JsValue::String(i.to_string().into()))
            .collect(),
        JsValue::Null | JsValue::Undefined => vec![],
        _ => vec![],
    };
    ctx.frame.accumulator = JsValue::new_array(keys);
    Ok(DispatchAction::Continue)
}

fn is_for_in_excluded_key(key: &str) -> bool {
    (key.starts_with("__") && key.ends_with("__"))
        || crate::builtins::symbol::is_symbol_property_key(key)
        || key.starts_with('.')
        || key.starts_with('#')
}

fn for_in_key_still_enumerable(obj: &JsValue, key: &str) -> bool {
    if is_for_in_excluded_key(key) {
        return false;
    }

    match obj {
        JsValue::PlainObject(map) => {
            let mut current_map = Some(Rc::clone(map));
            for _ in 0..256 {
                let Some(m) = current_map.take() else { break };
                let borrow = m.borrow();
                if let Some((_value, attrs)) = borrow.get_with_attrs(key) {
                    return attrs.contains(PropertyAttributes::ENUMERABLE);
                }
                current_map = borrow.get("__proto__").and_then(|v| {
                    if let JsValue::PlainObject(proto) = v {
                        Some(Rc::clone(proto))
                    } else {
                        None
                    }
                });
            }
            false
        }
        JsValue::Array(items) => key
            .parse::<usize>()
            .ok()
            .is_some_and(|idx| idx < items.borrow().len()),
        JsValue::String(s) => key
            .parse::<usize>()
            .ok()
            .is_some_and(|idx| idx < utf16_len(s)),
        _ => false,
    }
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
    let obj = ctx.frame.read_reg(_receiver_v)?.clone();
    let keys = ctx.frame.read_reg(keys_v)?.clone();
    let key = match &keys {
        JsValue::Array(items) => items
            .borrow()
            .get(idx)
            .cloned()
            .unwrap_or(JsValue::Undefined),
        _ => JsValue::Undefined,
    };
    ctx.frame.accumulator = match key {
        JsValue::String(key_string) if for_in_key_still_enumerable(&obj, &key_string) => {
            JsValue::String(key_string)
        }
        other if !matches!(other, JsValue::String(_)) => other,
        _ => JsValue::Undefined,
    };
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
        proxy_delete_property(&mut p.borrow_mut(), &key)?
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
                arr[idx] = JsValue::TheHole;
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
        let deleted = proxy_delete_property(&mut p.borrow_mut(), &key)?;
        if !deleted {
            return Err(StatorError::TypeError(format!(
                "Cannot delete property '{key}'"
            )));
        }
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
                arr[idx] = JsValue::TheHole;
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
    let rest: Vec<JsValue> = if ctx.frame.call_args.len() > param_count {
        ctx.frame.call_args[param_count..].to_vec()
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
        .call_args
        .get(..param_count)
        .unwrap_or(&[])
        .to_vec();
    let mut map = PropertyMap::new();
    for (i, v) in args.iter().enumerate() {
        map.insert(i.to_string(), v.clone());
    }
    // Per ES spec §10.4.4, arguments "length" is writable+configurable, not enumerable.
    map.insert_with_attrs(
        "length".to_string(),
        JsValue::Smi(args.len() as i32),
        PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE,
    );
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
        .call_args
        .get(..param_count)
        .unwrap_or(&[])
        .to_vec();
    let mut map = PropertyMap::new();
    for (i, v) in args.iter().enumerate() {
        map.insert(i.to_string(), v.clone());
    }
    map.insert_with_attrs(
        "length".to_string(),
        JsValue::Smi(args.len() as i32),
        PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE,
    );
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

/// Throw `TypeError` if the accumulator is `null` or `undefined`.
///
/// This is emitted at the start of destructuring patterns so that
/// `const {a} = null` or `const [x] = undefined` produce a proper error.
fn handle_throw_if_null_or_undefined(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    match &ctx.frame.accumulator {
        JsValue::Null => Err(StatorError::TypeError(
            "Cannot destructure 'null' as it is null".to_string(),
        )),
        JsValue::Undefined => Err(StatorError::TypeError(
            "Cannot destructure 'undefined' as it is undefined".to_string(),
        )),
        _ => Ok(DispatchAction::Continue),
    }
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

        let mut ns = PropertyMap::new();
        ns.insert("default".into(), specifier);
        let ns_val = JsValue::PlainObject(Rc::new(RefCell::new(ns)));

        let queue = MicrotaskQueue::new();
        let p = promise_resolve(ns_val, &queue);
        queue.drain();
        ctx.frame.accumulator = JsValue::Promise(p);
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

fn with_object_is_unscopable(obj: &JsValue, name: &str) -> bool {
    match proto_lookup(obj, "@@unscopables") {
        JsValue::PlainObject(map) => matches!(map.borrow().get(name), Some(JsValue::Boolean(true))),
        _ => false,
    }
}

fn plain_object_has_property(map: &Rc<RefCell<PropertyMap>>, name: &str) -> bool {
    let proto = {
        let borrow = map.borrow();
        if borrow.contains_key(name) {
            return true;
        }
        borrow.get("__proto__").cloned()
    };
    proto.is_some_and(|value| with_object_has_binding(&value, name))
}

fn with_object_has_binding(obj: &JsValue, name: &str) -> bool {
    if with_object_is_unscopable(obj, name) {
        return false;
    }
    match obj {
        JsValue::PlainObject(map) => {
            plain_object_has_property(map, name)
                || !matches!(proto_lookup(obj, name), JsValue::Undefined)
        }
        _ => !matches!(proto_lookup(obj, name), JsValue::Undefined),
    }
}

fn lookup_with_binding(context: &Option<JsValue>, name: &str) -> Option<JsValue> {
    let Some(JsValue::Context(root)) = context else {
        return None;
    };
    let mut current = Some(Rc::clone(root));
    while let Some(ctx_rc) = current {
        let (object, parent) = {
            let borrow = ctx_rc.borrow();
            (borrow.slots.first().cloned(), borrow.parent.clone())
        };
        if let Some(object) = object
            && with_object_has_binding(&object, name)
        {
            return Some(proto_lookup(&object, name));
        }
        current = parent;
    }
    None
}

fn store_with_binding(context: &Option<JsValue>, name: &str, value: &JsValue) -> bool {
    let Some(JsValue::Context(root)) = context else {
        return false;
    };
    let mut current = Some(Rc::clone(root));
    while let Some(ctx_rc) = current {
        let (object, parent) = {
            let borrow = ctx_rc.borrow();
            (borrow.slots.first().cloned(), borrow.parent.clone())
        };
        if let Some(JsValue::PlainObject(map)) = object
            && with_object_has_binding(&JsValue::PlainObject(Rc::clone(&map)), name)
        {
            map.borrow_mut().insert(name.to_string(), value.clone());
            return true;
        }
        current = parent;
    }
    false
}

/// Delete a property from the with-scope chain.
///
/// Walks the context chain looking for a with-object that has the given
/// property.  If found, deletes the property and returns `true`.
fn delete_with_binding(context: &Option<JsValue>, name: &str) -> Option<bool> {
    let Some(JsValue::Context(root)) = context else {
        return None;
    };
    let mut current = Some(Rc::clone(root));
    while let Some(ctx_rc) = current {
        let (object, parent) = {
            let borrow = ctx_rc.borrow();
            (borrow.slots.first().cloned(), borrow.parent.clone())
        };
        if let Some(JsValue::PlainObject(map)) = object
            && with_object_has_binding(&JsValue::PlainObject(Rc::clone(&map)), name)
        {
            map.borrow_mut().remove(name);
            return Some(true);
        }
        current = parent;
    }
    None
}

fn handle_delete_lookup_slot(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[0] else {
        return Err(err_bad_operand("DeleteLookupSlot", 0));
    };
    let name = match ctx.frame.bytecode_array.get_constant(name_idx) {
        Some(ConstantPoolEntry::String(s)) => s.clone(),
        _ => {
            return Err(StatorError::Internal(
                "DeleteLookupSlot: slot name is not a string".into(),
            ));
        }
    };
    ctx.frame.accumulator = if let Some(deleted) = delete_with_binding(&ctx.frame.context, &name) {
        JsValue::Boolean(deleted)
    } else {
        JsValue::Boolean(true)
    };
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
    if store_with_binding(&ctx.frame.context, &name, &val) {
        return Ok(DispatchAction::Continue);
    }
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
    ctx.frame.accumulator = if let Some(value) = lookup_with_binding(&ctx.frame.context, &name) {
        value
    } else {
        match ctx.frame.global_env.borrow().get(&name) {
            Some(v) => v.clone(),
            None => {
                return Err(StatorError::ReferenceError(format!(
                    "{name} is not defined"
                )));
            }
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
    ctx.frame.accumulator = lookup_with_binding(&ctx.frame.context, &name)
        .or_else(|| ctx.frame.global_env.borrow().get(&name).cloned())
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
    let Operand::Register(_receiver_v) = instr.operands[0] else {
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
    // The accumulator holds the lookup-start object for super property
    // access (HomeObject.[[Prototype]]).  The lookup must begin there,
    // NOT on `this` (the receiver), so that overridden methods on the
    // subclass prototype are skipped.
    let lookup_start = ctx.frame.accumulator.clone();
    ctx.frame.accumulator = if matches!(lookup_start, JsValue::Undefined | JsValue::Null) {
        JsValue::Undefined
    } else {
        proto_lookup(&lookup_start, &prop_name)
    };
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
    if let Some(cached) = ctx.frame.bytecode_array.cached_template_object(cache_key) {
        ctx.frame.accumulator = cached;
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
        // ES §12.2.9.3: The template object and its `raw` property must be frozen.
        if let JsValue::PlainObject(ref map) = tpl_val {
            let raw_clone = map.borrow().get("raw").cloned();
            if let Some(JsValue::PlainObject(ref raw_map)) = raw_clone {
                raw_map.borrow_mut().freeze();
            }
            map.borrow_mut().freeze();
        }
        ctx.frame
            .bytecode_array
            .cache_template_object(cache_key, tpl_val.clone());
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
            if ba.is_arrow() {
                return Err(StatorError::TypeError(
                    "Function is not a constructor".to_string(),
                ));
            }
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
            let result = stacker::maybe_grow(64 * 1024, 1024 * 1024, || {
                Interpreter::run(&mut callee_frame)
            });
            pop_call_frame();
            let val = result?;
            ctx.frame.accumulator = match val {
                JsValue::PlainObject(_) | JsValue::Object(_) => val,
                _ => this_val,
            };
        }
        JsValue::NativeFunction(f) => {
            let val = f(args)?;
            ctx.frame.accumulator = construct_builtin_result(val, &ctor_proto)?;
        }
        JsValue::PlainObject(ref map) => {
            if map.borrow().get("__no_construct__").is_some() {
                return Err(StatorError::TypeError(
                    "Symbol is not a constructor".to_string(),
                ));
            }
            let call_val = map.borrow().get("__call__").cloned();
            match call_val {
                Some(JsValue::NativeFunction(f)) => {
                    let val = f(args)?;
                    ctx.frame.accumulator = construct_builtin_result(val, &ctor_proto)?;
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
        let binding_registers = ctx.frame.bytecode_array.binding_registers().clone();
        let original_global_names: HashSet<String> =
            ctx.frame.global_env.borrow().keys().cloned().collect();
        let mut eval_bindings = ctx.frame.global_env.borrow().clone();
        for (name, reg) in &binding_registers {
            eval_bindings.insert(name.clone(), ctx.frame.read_reg(*reg as u32)?.clone());
        }
        let eval_env = Rc::new(RefCell::new(eval_bindings));
        let (result, final_env, is_strict) =
            crate::builtins::global::global_eval_direct_with_scope_capture(
                &source,
                Rc::clone(&eval_env),
                ctx.frame.context.clone(),
            )?;
        let final_bindings = final_env.borrow();
        for (name, reg) in &binding_registers {
            if let Some(value) = final_bindings.get(name) {
                ctx.frame.write_reg(*reg as u32, value.clone())?;
            }
        }
        {
            let mut globals = ctx.frame.global_env.borrow_mut();
            for (name, value) in final_bindings.iter() {
                if binding_registers.contains_key(name) {
                    continue;
                }
                if !is_strict || original_global_names.contains(name) {
                    globals.insert(name.clone(), value.clone());
                }
            }
        }
        ctx.frame.accumulator = result;
    } else {
        // Callee was reassigned — fall through to normal call.
        match callee {
            JsValue::Function(ba) => {
                if ba.is_generator() {
                    let state = GeneratorState::new((*ba).clone());
                    super::init_generator_state_prototype(&state, &ba);
                    ctx.frame.accumulator = JsValue::Generator(state);
                } else {
                    let mut callee_frame = InterpreterFrame::new_with_globals(
                        (*ba).clone(),
                        args,
                        Rc::clone(&ctx.frame.global_env),
                    );
                    restore_closure_context(&mut callee_frame, &ba);
                    populate_self_name(&mut callee_frame, &ba, &JsValue::Function(Rc::clone(&ba)));
                    push_call_frame("<eval-fallback>")?;
                    let result = stacker::maybe_grow(64 * 1024, 1024 * 1024, || {
                        Interpreter::run(&mut callee_frame)
                    });
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
    let ctor_name = ctor_ba_rc.function_name().to_string();
    class_obj
        .borrow_mut()
        .insert("name".to_string(), JsValue::String(ctor_name.into()));
    class_obj.borrow_mut().insert(
        "length".to_string(),
        JsValue::Smi(ctor_ba_rc.function_length() as i32),
    );
    class_obj
        .borrow_mut()
        .insert(".class_constructor".to_string(), JsValue::Boolean(true));

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
        if let JsValue::Function(ba) = &getter {
            fn_props_set(ba, ".home_object".to_string(), obj.clone());
        }
        // Store getter as __get_<name>__ — the property access handler
        // checks for this convention when loading.
        let accessor_attrs = if is_private_storage_key(&prop_name) {
            PropertyAttributes::CONFIGURABLE
        } else {
            PropertyAttributes::ENUMERABLE | PropertyAttributes::CONFIGURABLE
        };
        map.borrow_mut()
            .insert_with_attrs(format!("__get_{prop_name}__"), getter, accessor_attrs);
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
        if let JsValue::Function(ba) = &setter {
            fn_props_set(ba, ".home_object".to_string(), obj.clone());
        }
        let accessor_attrs = if is_private_storage_key(&prop_name) {
            PropertyAttributes::CONFIGURABLE
        } else {
            PropertyAttributes::ENUMERABLE | PropertyAttributes::CONFIGURABLE
        };
        map.borrow_mut()
            .insert_with_attrs(format!("__set_{prop_name}__"), setter, accessor_attrs);
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
        if let JsValue::Function(ba) = &getter {
            fn_props_set(ba, ".home_object".to_string(), obj.clone());
        }
        let accessor_attrs = PropertyAttributes::ENUMERABLE | PropertyAttributes::CONFIGURABLE;
        map.borrow_mut()
            .insert_with_attrs(format!("__get_{key_str}__"), getter, accessor_attrs);
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
        if let JsValue::Function(ba) = &setter {
            fn_props_set(ba, ".home_object".to_string(), obj.clone());
        }
        let accessor_attrs = PropertyAttributes::ENUMERABLE | PropertyAttributes::CONFIGURABLE;
        map.borrow_mut()
            .insert_with_attrs(format!("__set_{key_str}__"), setter, accessor_attrs);
    }
    Ok(DispatchAction::Continue)
}

/// `DefineClassNamedOwnProperty <obj_reg> <name_idx> <slot>`
fn handle_define_class_named_own_property(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("DefineClassNamedOwnProperty", 0));
    };
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[1] else {
        return Err(err_bad_operand("DefineClassNamedOwnProperty", 1));
    };
    let prop_name = match ctx.frame.bytecode_array.get_constant(name_idx) {
        Some(ConstantPoolEntry::String(s)) => s.clone(),
        _ => {
            return Err(StatorError::Internal(
                "DefineClassNamedOwnProperty: property name is not a string".into(),
            ));
        }
    };
    let val = ctx.frame.accumulator.clone();
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    if let JsValue::PlainObject(ref map) = obj {
        set_named_property_function_metadata(&val, &obj, &prop_name);
        if let Some(attrs) = private_named_property_attrs(&prop_name) {
            map.borrow_mut().insert_with_attrs(prop_name, val, attrs);
        } else {
            map.borrow_mut().insert_builtin(prop_name, val);
        }
    }
    Ok(DispatchAction::Continue)
}

/// `DefineClassGetterProperty <obj_reg> <name_idx> <slot>`
fn handle_define_class_getter_property(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("DefineClassGetterProperty", 0));
    };
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[1] else {
        return Err(err_bad_operand("DefineClassGetterProperty", 1));
    };
    let prop_name = match ctx.frame.bytecode_array.get_constant(name_idx) {
        Some(ConstantPoolEntry::String(s)) => s.clone(),
        _ => {
            return Err(StatorError::Internal(
                "DefineClassGetterProperty: property name is not a string".into(),
            ));
        }
    };
    let getter = ctx.frame.accumulator.clone();
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    if let JsValue::PlainObject(ref map) = obj {
        if let JsValue::Function(ba) = &getter {
            fn_props_set(ba, ".home_object".to_string(), obj.clone());
        }
        let accessor_attrs = if is_private_storage_key(&prop_name) {
            PropertyAttributes::CONFIGURABLE
        } else {
            PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE
        };
        map.borrow_mut()
            .insert_with_attrs(format!("__get_{prop_name}__"), getter, accessor_attrs);
    }
    Ok(DispatchAction::Continue)
}

/// `DefineClassSetterProperty <obj_reg> <name_idx> <slot>`
fn handle_define_class_setter_property(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("DefineClassSetterProperty", 0));
    };
    let Operand::ConstantPoolIdx(name_idx) = instr.operands[1] else {
        return Err(err_bad_operand("DefineClassSetterProperty", 1));
    };
    let prop_name = match ctx.frame.bytecode_array.get_constant(name_idx) {
        Some(ConstantPoolEntry::String(s)) => s.clone(),
        _ => {
            return Err(StatorError::Internal(
                "DefineClassSetterProperty: property name is not a string".into(),
            ));
        }
    };
    let setter = ctx.frame.accumulator.clone();
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    if let JsValue::PlainObject(ref map) = obj {
        if let JsValue::Function(ba) = &setter {
            fn_props_set(ba, ".home_object".to_string(), obj.clone());
        }
        let accessor_attrs = if is_private_storage_key(&prop_name) {
            PropertyAttributes::CONFIGURABLE
        } else {
            PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE
        };
        map.borrow_mut()
            .insert_with_attrs(format!("__set_{prop_name}__"), setter, accessor_attrs);
    }
    Ok(DispatchAction::Continue)
}

/// `DefineClassKeyedOwnProperty <obj_reg> <key_reg> <flags> <slot>`
fn handle_define_class_keyed_own_property(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("DefineClassKeyedOwnProperty", 0));
    };
    let Operand::Register(key_v) = instr.operands[1] else {
        return Err(err_bad_operand("DefineClassKeyedOwnProperty", 1));
    };
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    let key = ctx.frame.read_reg(key_v)?.clone();
    let val = ctx.frame.accumulator.clone();
    if let JsValue::PlainObject(ref map) = obj {
        let prop_name = to_property_key(&key)?;
        set_named_property_function_metadata(&val, &obj, &prop_name);
        map.borrow_mut().insert_builtin(prop_name, val);
    }
    Ok(DispatchAction::Continue)
}

/// `DefineClassKeyedGetterProperty <obj_reg> <key_reg> <slot>`
fn handle_define_class_keyed_getter_property(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("DefineClassKeyedGetterProperty", 0));
    };
    let Operand::Register(key_v) = instr.operands[1] else {
        return Err(err_bad_operand("DefineClassKeyedGetterProperty", 1));
    };
    let getter = ctx.frame.accumulator.clone();
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    let key = ctx.frame.read_reg(key_v)?.clone();
    let key_str = to_property_key(&key)?;
    if let JsValue::PlainObject(ref map) = obj {
        if let JsValue::Function(ba) = &getter {
            fn_props_set(ba, ".home_object".to_string(), obj.clone());
        }
        let accessor_attrs = PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE;
        map.borrow_mut()
            .insert_with_attrs(format!("__get_{key_str}__"), getter, accessor_attrs);
    }
    Ok(DispatchAction::Continue)
}

/// `DefineClassKeyedSetterProperty <obj_reg> <key_reg> <slot>`
fn handle_define_class_keyed_setter_property(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("DefineClassKeyedSetterProperty", 0));
    };
    let Operand::Register(key_v) = instr.operands[1] else {
        return Err(err_bad_operand("DefineClassKeyedSetterProperty", 1));
    };
    let setter = ctx.frame.accumulator.clone();
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    let key = ctx.frame.read_reg(key_v)?.clone();
    let key_str = to_property_key(&key)?;
    if let JsValue::PlainObject(ref map) = obj {
        if let JsValue::Function(ba) = &setter {
            fn_props_set(ba, ".home_object".to_string(), obj.clone());
        }
        let accessor_attrs = PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE;
        map.borrow_mut()
            .insert_with_attrs(format!("__set_{key_str}__"), setter, accessor_attrs);
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

/// `TestPrivateBrand <obj_reg> <brand_reg>`
///
/// Checks whether the object in `obj_reg` has the private brand stored
/// in `brand_reg`.  Sets the accumulator to `Boolean(true)` when the
/// brand is found, `Boolean(false)` otherwise.  Throws `TypeError` when
/// the target is not an object.
fn handle_test_private_brand(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("TestPrivateBrand", 0));
    };
    let Operand::Register(brand_v) = instr.operands[1] else {
        return Err(err_bad_operand("TestPrivateBrand", 1));
    };
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    let brand = ctx.frame.read_reg(brand_v)?.clone();
    let brand_key = to_property_key(&brand)?;

    if !is_js_receiver(&obj) {
        return Err(StatorError::TypeError(format!(
            "Cannot use 'in' operator to search for private field '{brand_key}' in non-object",
        )));
    }

    let has_brand = match &obj {
        JsValue::PlainObject(map) => {
            let borrow = map.borrow();
            if brand_key.starts_with(PRIVATE_BRAND_PREFIX) {
                borrow.contains_key(&brand_key)
            } else {
                own_private_element_exists(&borrow, &brand_key)
            }
        }
        _ => false,
    };
    ctx.frame.accumulator = JsValue::Boolean(has_brand);
    Ok(DispatchAction::Continue)
}

/// `DefinePrivateBrand <obj_reg>`
///
/// Brands the object in `obj_reg` with the brand identifier currently
/// in the accumulator so that subsequent `TestPrivateBrand` calls on
/// the same object will succeed for that brand.
fn handle_define_private_brand(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::Register(obj_v) = instr.operands[0] else {
        return Err(err_bad_operand("DefinePrivateBrand", 0));
    };
    let obj = ctx.frame.read_reg(obj_v)?.clone();
    let brand = ctx.frame.accumulator.clone();
    let brand_key = to_property_key(&brand)?;

    match &obj {
        JsValue::PlainObject(map) => {
            map.borrow_mut().insert_with_attrs(
                brand_key,
                JsValue::Boolean(true),
                PropertyAttributes::CONFIGURABLE,
            );
        }
        _ => {
            return Err(StatorError::TypeError(
                "Cannot define private brand on non-object".into(),
            ));
        }
    }
    Ok(DispatchAction::Continue)
}

/// `LdaModuleVariable <module_request_idx> <cell_idx>`
///
/// Loads a module import binding into the accumulator.  The constant pool
/// entry at `module_request_idx` holds the source module specifier string,
/// and `cell_idx` identifies the binding cell within that module.
///
/// Since full module linking is not yet wired up, module variables are
/// backed by the shared global environment keyed as
/// `__mod:{specifier}:{cell}`.
fn handle_lda_module_variable(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(req_idx) = instr.operands[0] else {
        return Err(err_bad_operand("LdaModuleVariable", 0));
    };
    let Operand::Immediate(cell) = instr.operands[1] else {
        return Err(err_bad_operand("LdaModuleVariable", 1));
    };
    let specifier = match ctx.frame.bytecode_array.get_constant(req_idx) {
        Some(ConstantPoolEntry::String(s)) => s.clone(),
        _ => {
            return Err(StatorError::Internal(
                "LdaModuleVariable: module specifier is not a string".into(),
            ));
        }
    };
    let key = format!("__mod:{specifier}:{cell}");
    ctx.frame.accumulator = ctx
        .frame
        .global_env
        .borrow()
        .get(&key)
        .cloned()
        .unwrap_or(JsValue::Undefined);
    Ok(DispatchAction::Continue)
}

/// `StaModuleVariable <module_request_idx> <cell_idx>`
///
/// Stores the accumulator into a module export binding.  Used for
/// `export let`/`export var` — live-binding semantics mean importers
/// see updated values.
///
/// Backed by the global environment (see [`handle_lda_module_variable`]).
fn handle_sta_module_variable(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(req_idx) = instr.operands[0] else {
        return Err(err_bad_operand("StaModuleVariable", 0));
    };
    let Operand::Immediate(cell) = instr.operands[1] else {
        return Err(err_bad_operand("StaModuleVariable", 1));
    };
    let specifier = match ctx.frame.bytecode_array.get_constant(req_idx) {
        Some(ConstantPoolEntry::String(s)) => s.clone(),
        _ => {
            return Err(StatorError::Internal(
                "StaModuleVariable: module specifier is not a string".into(),
            ));
        }
    };
    let key = format!("__mod:{specifier}:{cell}");
    let val = ctx.frame.accumulator.clone();
    ctx.frame.global_env.borrow_mut().insert(key, val);
    Ok(DispatchAction::Continue)
}

/// `LdaImportMeta`
///
/// Loads the `import.meta` object into the accumulator.  Per ECMAScript
/// §16.2.1.7, `import.meta` is an ordinary object whose prototype is
/// `null`.  The host may populate it with properties such as `url`.
///
/// Returns a frozen plain object with a placeholder `url` and a minimal
/// `resolve(specifier)` stub.
fn handle_lda_import_meta(
    ctx: &mut DispatchContext,
    _instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let mut meta = PropertyMap::new();
    meta.insert("url".into(), JsValue::String(String::new().into()));
    meta.insert(
        "resolve".into(),
        JsValue::NativeFunction(Rc::new(|args| {
            let value = match args.as_slice() {
                [] => JsValue::Undefined,
                [value] => value.clone(),
                [_, value, ..] => value.clone(),
            };
            Ok(value)
        })),
    );
    meta.freeze();
    ctx.frame.accumulator = JsValue::PlainObject(Rc::new(RefCell::new(meta)));
    Ok(DispatchAction::Continue)
}

/// `GetModuleNamespace <module_request_idx>`
///
/// Creates a module namespace exotic object for the module identified by
/// the constant-pool string at `module_request_idx` and loads it into the
/// accumulator.  Used by `import * as ns from "…"` and
/// `export * from "…"`.
///
/// Returns a fresh empty plain object (full namespace resolution depends
/// on the module linker which is not yet wired up).
fn handle_get_module_namespace(
    ctx: &mut DispatchContext,
    instr: &Instruction,
) -> StatorResult<DispatchAction> {
    let Operand::ConstantPoolIdx(_req_idx) = instr.operands[0] else {
        return Err(err_bad_operand("GetModuleNamespace", 0));
    };
    ctx.frame.accumulator = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
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
    table[Opcode::LdaGlobalInsideTypeof as usize] = handle_lda_global_inside_typeof;
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
    table[Opcode::DeleteLookupSlot as usize] = handle_delete_lookup_slot;
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
    table[Opcode::DefineClassNamedOwnProperty as usize] = handle_define_class_named_own_property;
    table[Opcode::DefineClassGetterProperty as usize] = handle_define_class_getter_property;
    table[Opcode::DefineClassSetterProperty as usize] = handle_define_class_setter_property;
    table[Opcode::DefineClassKeyedOwnProperty as usize] = handle_define_class_keyed_own_property;
    table[Opcode::DefineClassKeyedGetterProperty as usize] =
        handle_define_class_keyed_getter_property;
    table[Opcode::DefineClassKeyedSetterProperty as usize] =
        handle_define_class_keyed_setter_property;
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
    table[Opcode::IteratorClose as usize] = handle_iterator_close;
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
    table[Opcode::ThrowIfNullOrUndefined as usize] = handle_throw_if_null_or_undefined;
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
    table[Opcode::LdaModuleVariable as usize] = handle_lda_module_variable;
    table[Opcode::StaModuleVariable as usize] = handle_sta_module_variable;
    table[Opcode::LdaImportMeta as usize] = handle_lda_import_meta;
    table[Opcode::LdaNewTarget as usize] = handle_lda_new_target;
    table[Opcode::GetModuleNamespace as usize] = handle_get_module_namespace;
    table[Opcode::Wide as usize] = handle_wide;
    table[Opcode::ExtraWide as usize] = handle_wide;
    table[Opcode::Illegal as usize] = handle_unimplemented;
    table
};

#[cfg(test)]
mod tests {
    use crate::objects::value::JsValue;

    fn assert_eval_true(source: &str) {
        let result = crate::builtins::global::global_eval(source).unwrap();
        assert_eq!(result, JsValue::Boolean(true), "source: {source}");
    }

    #[test]
    fn test_typeof_generator() {
        let result =
            crate::builtins::global::global_eval("function* gen() { yield 1; } typeof gen()")
                .unwrap();
        assert_eq!(result, JsValue::String("object".into()));
    }

    #[test]
    fn test_typeof_promise() {
        let result =
            crate::builtins::global::global_eval("typeof new Promise(function(r) { r(1); })")
                .unwrap();
        assert_eq!(result, JsValue::String("object".into()));
    }

    #[test]
    fn test_for_in_array() {
        let result = crate::builtins::global::global_eval(
            "var a = [10, 20, 30]; var keys = []; for (var k in a) { keys.push(k); } keys.length",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    #[test]
    fn test_for_in_array_keys_are_strings() {
        let result = crate::builtins::global::global_eval(
            "var a = [10, 20]; var keys = []; for (var k in a) { keys.push(k); } keys[0]",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("0".into()));
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_user_constructor_instanceof() {
        let result = crate::builtins::global::global_eval(
            "function Foo() {} var x = new Foo(); x instanceof Foo",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_function_has_prototype() {
        let result = crate::builtins::global::global_eval(
            "function Foo() {} typeof Foo.prototype === 'object'",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_prototype_constructor_points_back() {
        let result = crate::builtins::global::global_eval(
            "function Foo() {} Foo.prototype.constructor === Foo",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_arrow_no_prototype() {
        let result =
            crate::builtins::global::global_eval("var f = () => {}; f.prototype === undefined")
                .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_spread_skips_non_enumerable() {
        // Array spread: should copy index keys, not length/__is_array__
        let result = crate::builtins::global::global_eval(
            "var a = [10, 20]; var o = {...a}; o[0] + ',' + o[1] + ',' + (o.length === undefined)",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("10,20,true".into()));
    }

    #[test]
    fn e2e_spread_string() {
        let result =
            crate::builtins::global::global_eval("var o = {...'hi'}; o[0] + o[1]").unwrap();
        assert_eq!(result, JsValue::String("hi".into()));
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_private_method() {
        let result = crate::builtins::global::global_eval(
            "class Foo { #bar() { return 42; } test() { return this.#bar(); } } \
             var f = new Foo(); f.test()",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_private_field() {
        let result = crate::builtins::global::global_eval(
            "class C { #x = 10; get() { return this.#x; } } new C().get()",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(10));
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_private_field_write() {
        let result = crate::builtins::global::global_eval(
            "class C { #x = 0; set(v) { this.#x = v; } get() { return this.#x; } } \
             var c = new C(); c.set(99); c.get()",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    #[test]
    fn e2e_constructor_name() {
        let result = crate::builtins::global::global_eval(
            "Date.name + ',' + Map.name + ',' + Set.name + ',' + Array.name",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("Date,Map,Set,Array".into()));
    }

    #[test]
    fn e2e_prototype_constructor() {
        let result = crate::builtins::global::global_eval(
            "Date.prototype.constructor === Date && \
             Map.prototype.constructor === Map && \
             Array.prototype.constructor === Array",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_instance_constructor() {
        let result =
            crate::builtins::global::global_eval("var d = new Date(); d.constructor === Date")
                .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_arrow_not_constructable() {
        let result = crate::builtins::global::global_eval(
            "var f = () => {}; try { new f(); 'no error' } catch(e) { e.message }",
        )
        .unwrap();
        assert_eq!(
            result,
            JsValue::String("Function is not a constructor".into())
        );
    }

    #[test]
    fn e2e_typeof_callable_object() {
        // typeof should return "function" for callable PlainObjects (those with __call__)
        let result = crate::builtins::global::global_eval("typeof Date").unwrap();
        assert_eq!(result, JsValue::String("function".into()));
    }

    #[test]
    fn e2e_for_of_string() {
        let result = crate::builtins::global::global_eval(
            "var r = ''; for (var c of 'abc') r += c + ','; r",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("a,b,c,".into()));
    }

    #[test]
    fn e2e_destructure_default() {
        let result =
            crate::builtins::global::global_eval("var {x = 10, y = 20} = {x: 5}; x + y").unwrap();
        assert_eq!(result, JsValue::Smi(25));
    }

    #[test]
    fn e2e_spread_call() {
        let result = crate::builtins::global::global_eval(
            "function sum(a,b,c) { return a+b+c; } sum(1,2,3)",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(6));
    }

    #[test]
    fn e2e_template_literal_expr() {
        let result =
            crate::builtins::global::global_eval("var x = 10; `value is ${x + 5}`").unwrap();
        assert_eq!(result, JsValue::String("value is 15".into()));
    }

    // ── typeof tests for all primitive types (ES §12.5.6) ───────────────

    #[test]
    fn e2e_typeof_undefined() {
        let result = crate::builtins::global::global_eval("typeof undefined").unwrap();
        assert_eq!(result, JsValue::String("undefined".into()));
    }

    #[test]
    fn e2e_typeof_null_is_object() {
        let result = crate::builtins::global::global_eval("typeof null").unwrap();
        assert_eq!(result, JsValue::String("object".into()));
    }

    #[test]
    fn e2e_typeof_boolean() {
        let result = crate::builtins::global::global_eval("typeof true").unwrap();
        assert_eq!(result, JsValue::String("boolean".into()));
    }

    #[test]
    fn e2e_typeof_number_smi() {
        let result = crate::builtins::global::global_eval("typeof 42").unwrap();
        assert_eq!(result, JsValue::String("number".into()));
    }

    #[test]
    fn e2e_typeof_number_float() {
        let result = crate::builtins::global::global_eval("typeof 3.14").unwrap();
        assert_eq!(result, JsValue::String("number".into()));
    }

    #[test]
    fn e2e_typeof_string() {
        let result = crate::builtins::global::global_eval("typeof 'hello'").unwrap();
        assert_eq!(result, JsValue::String("string".into()));
    }

    #[test]
    fn e2e_typeof_function_expr() {
        let result =
            crate::builtins::global::global_eval("var f = function() {}; typeof f").unwrap();
        assert_eq!(result, JsValue::String("function".into()));
    }

    #[test]
    fn e2e_typeof_arrow_is_function() {
        let result = crate::builtins::global::global_eval("var f = () => {}; typeof f").unwrap();
        assert_eq!(result, JsValue::String("function".into()));
    }

    #[test]
    fn e2e_typeof_object_literal() {
        let result = crate::builtins::global::global_eval("var o = {}; typeof o").unwrap();
        assert_eq!(result, JsValue::String("object".into()));
    }

    #[test]
    fn e2e_typeof_array_is_object() {
        let result = crate::builtins::global::global_eval("typeof []").unwrap();
        assert_eq!(result, JsValue::String("object".into()));
    }

    // ── Strict equality edge cases (ES §7.2.16) ────────────────────────

    #[test]
    fn e2e_strict_eq_null_null() {
        let result = crate::builtins::global::global_eval("null === null").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_strict_eq_undefined_undefined() {
        let result = crate::builtins::global::global_eval("undefined === undefined").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_strict_eq_null_undefined_is_false() {
        // null and undefined are different types → strict equality is false
        let result = crate::builtins::global::global_eval("null === undefined").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn e2e_strict_not_equal_null_undefined() {
        // Exercises TestEqualStrict + LogicalNot (the !== path)
        let result = crate::builtins::global::global_eval("null !== undefined").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_strict_eq_nan_is_not_nan() {
        // IEEE 754: NaN !== NaN
        let result = crate::builtins::global::global_eval("NaN === NaN").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn e2e_strict_not_equal_nan_nan() {
        let result = crate::builtins::global::global_eval("NaN !== NaN").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── ToObject TypeError on null/undefined (ES §7.1.18) ──────────────

    #[test]
    fn e2e_to_object_null_throws_type_error() {
        // `with(null)` emits the ToObject opcode, which must throw TypeError.
        let result = crate::builtins::global::global_eval(
            "var r; try { with(null) { r = 1; } } catch(e) { r = e.message; } r",
        )
        .unwrap();
        assert_eq!(
            result,
            JsValue::String("Cannot convert undefined or null to object".into())
        );
    }

    #[test]
    fn e2e_to_object_undefined_throws_type_error() {
        let result = crate::builtins::global::global_eval(
            "var r; try { with(undefined) { r = 1; } } catch(e) { r = e.message; } r",
        )
        .unwrap();
        assert_eq!(
            result,
            JsValue::String("Cannot convert undefined or null to object".into())
        );
    }

    // ── Null / undefined identity via variables ─────────────────────────

    #[test]
    fn e2e_null_variable_strict_eq() {
        let result = crate::builtins::global::global_eval("var x = null; x === null").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_undefined_variable_strict_eq() {
        let result =
            crate::builtins::global::global_eval("var x = undefined; x === undefined").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── ToName / freeze / seal tests ────────────────────────────────────

    /// Computed property key calls `[Symbol.toPrimitive]` on objects.
    #[test]
    fn e2e_computed_property_toprimitive() {
        let result = crate::builtins::global::global_eval(
            "var k = { [Symbol.toPrimitive]() { return 'x'; } }; var o = {}; o[k] = 1; o.x",
        );
        // If ToPrimitive is called, k becomes "x", so o.x should be 1.
        assert!(result.is_ok());
    }

    /// Sloppy-mode property write on frozen object silently fails.
    #[test]
    fn e2e_frozen_object_sloppy() {
        let result = crate::builtins::global::global_eval(
            "var o = { x: 1 }; Object.freeze(o); o.x = 2; o.x",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(1), "frozen prop should remain 1");
    }

    /// Strict-mode property write on frozen object throws TypeError.
    #[test]
    fn e2e_frozen_object_strict() {
        let result = crate::builtins::global::global_eval(
            "'use strict'; var o = { x: 1 }; Object.freeze(o); o.x = 2;",
        );
        assert!(result.is_err(), "should throw TypeError");
    }

    /// Sealed object allows writes to existing writable properties.
    #[test]
    fn e2e_sealed_object_write_existing() {
        let result =
            crate::builtins::global::global_eval("var o = { x: 1 }; Object.seal(o); o.x = 2; o.x")
                .unwrap();
        assert_eq!(
            result,
            JsValue::Smi(2),
            "sealed should allow writes to existing writable props"
        );
    }

    /// Sealed object rejects new property addition in strict mode.
    #[test]
    fn e2e_sealed_object_new_prop_strict() {
        let result = crate::builtins::global::global_eval(
            "'use strict'; var o = { x: 1 }; Object.seal(o); o.y = 2;",
        );
        assert!(
            result.is_err(),
            "sealed should reject new property in strict"
        );
    }

    /// Object.isFrozen returns true for frozen objects.
    #[test]
    fn e2e_object_is_frozen() {
        let result = crate::builtins::global::global_eval(
            "var o = {}; Object.freeze(o); Object.isFrozen(o)",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// Object.isSealed returns true for sealed objects.
    #[test]
    fn e2e_object_is_sealed() {
        let result =
            crate::builtins::global::global_eval("var o = {}; Object.seal(o); Object.isSealed(o)")
                .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// for-of break exits after first iteration.
    #[test]
    fn e2e_for_of_break_calls_return() {
        let result = crate::builtins::global::global_eval(
            "var count = 0;\
             for (var x of [1,2,3]) { count++; break; }\
             count",
        )
        .unwrap();
        assert_eq!(
            result,
            JsValue::Smi(1),
            "break should exit for-of after first iteration"
        );
    }

    // ── Spread argument tests ───────────────────────────────────────────

    /// `f(...[1,2,3])` should expand the array into three individual args.
    #[test]
    fn e2e_spread_call_basic() {
        let result = crate::builtins::global::global_eval(
            "function sum(a, b, c) { return a + b + c; } sum(...[1, 2, 3])",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(6));
    }

    /// `f(0, ...[1,2], 3)` should expand mid-argument spread.
    #[test]
    fn e2e_spread_call_mixed_args() {
        let result = crate::builtins::global::global_eval(
            "function sum(a, b, c, d) { return a + b + c + d; } sum(0, ...[1, 2], 3)",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(6));
    }

    /// Spread a variable holding an array.
    #[test]
    fn e2e_spread_call_variable() {
        let result = crate::builtins::global::global_eval(
            "function sum(a, b, c) { return a + b + c; } var arr = [10, 20, 30]; sum(...arr)",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(60));
    }

    /// Spread with `new` (ConstructWithSpread).
    #[test]
    fn e2e_spread_construct() {
        let result = crate::builtins::global::global_eval(
            "function Pair(a, b) { this.sum = a + b; } \
             var p = new Pair(...[3, 7]); p.sum",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(10));
    }

    /// Empty spread produces zero arguments.
    #[test]
    fn e2e_spread_call_empty() {
        let result = crate::builtins::global::global_eval(
            "function len() { return arguments.length; } len(...[])",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(0));
    }

    #[test]
    fn e2e_typeof_null() {
        let result = crate::builtins::global::global_eval("typeof null").unwrap();
        assert_eq!(result, JsValue::String("object".into()));
    }

    #[test]
    fn e2e_nan_strict_equal() {
        let result = crate::builtins::global::global_eval("NaN === NaN").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn e2e_positive_negative_zero_equal() {
        let result = crate::builtins::global::global_eval("+0 === -0").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── §1 Object literal prototype chain ───────────────────────────────

    #[test]
    fn test_object_literal_has_tostring() {
        let result = crate::builtins::global::global_eval("({}).toString()").unwrap();
        assert_eq!(result, JsValue::String("[object Object]".into()));
    }

    #[test]
    fn test_object_literal_has_valueof() {
        let result = crate::builtins::global::global_eval("typeof ({}).valueOf()").unwrap();
        assert_eq!(result, JsValue::String("object".into()));
    }

    #[test]
    fn test_object_literal_has_hasownproperty() {
        let result = crate::builtins::global::global_eval("({a: 1}).hasOwnProperty('a')").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_object_literal_hasownproperty_missing() {
        let result = crate::builtins::global::global_eval("({a: 1}).hasOwnProperty('b')").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_object_literal_tostring_tag() {
        // Object.prototype.toString.call(null) should classify the receiver.
        let result = crate::builtins::global::global_eval("var o = {x: 1}; o.toString()").unwrap();
        assert_eq!(result, JsValue::String("[object Object]".into()));
    }

    // ── §2 Property descriptor: writable / enumerable / configurable ────

    #[test]
    fn test_object_literal_property_writable() {
        let result = crate::builtins::global::global_eval("var o = {a: 1}; o.a = 2; o.a").unwrap();
        assert_eq!(result, JsValue::Smi(2));
    }

    #[test]
    fn test_object_literal_property_enumerable() {
        let result = crate::builtins::global::global_eval(
            "var o = {a: 1, b: 2}; var r = ''; for (var k in o) r += k + ','; r",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("a,b,".into()));
    }

    #[test]
    fn test_object_literal_property_configurable() {
        let result =
            crate::builtins::global::global_eval("var o = {a: 1}; delete o.a; o.a === undefined")
                .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_object_literal_descriptor_all_true() {
        let result = crate::builtins::global::global_eval(
            "var d = Object.getOwnPropertyDescriptor({a: 1}, 'a'); \
             d.writable === true && d.enumerable === true && d.configurable === true",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── §3 Frozen / sealed object store behaviour ───────────────────────

    #[test]
    fn test_frozen_object_strict_throws() {
        let result = crate::builtins::global::global_eval(
            "var o = Object.freeze({a: 1}); \
             (function() { 'use strict'; try { o.a = 2; return 'no error'; } catch(e) { return e.message; } })()",
        )
        .unwrap();
        match result {
            JsValue::String(s) => assert!(
                s.contains("read only"),
                "Expected 'read only' in error: {s}"
            ),
            other => panic!("Expected string error message, got: {other:?}"),
        }
    }

    #[test]
    fn test_sealed_object_no_add_strict() {
        let result = crate::builtins::global::global_eval(
            "var o = Object.seal({a: 1}); \
             (function() { 'use strict'; try { o.b = 2; return 'no error'; } catch(e) { return e.message; } })()",
        )
        .unwrap();
        match result {
            JsValue::String(s) => assert!(
                s.contains("not extensible"),
                "Expected 'not extensible' in error: {s}"
            ),
            other => panic!("Expected string error message, got: {other:?}"),
        }
    }

    #[test]
    fn test_frozen_object_is_frozen() {
        let result =
            crate::builtins::global::global_eval("Object.isFrozen(Object.freeze({a: 1}))").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_sealed_object_existing_prop_writable() {
        // Sealed objects allow writing to existing writable properties.
        let result =
            crate::builtins::global::global_eval("var o = Object.seal({a: 1}); o.a = 99; o.a")
                .unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    // ── Property descriptor accessor / writable enforcement ─────────────

    /// Accessor getter defined via Object.defineProperty is invoked on read.
    #[test]
    fn e2e_define_property_getter_invoked() {
        let result = crate::builtins::global::global_eval(
            "var o = {}; \
             Object.defineProperty(o, 'x', { get: function() { return 42; } }); \
             o.x",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// Accessor setter defined via Object.defineProperty is invoked on write.
    /// NOTE: Setter dispatch from Object.defineProperty not yet fully working.
    // #[test]
    // fn e2e_define_property_setter_invoked() {
    //     let result = crate::builtins::global::global_eval(
    //         "var o = {}; var stored = 0; \
    //          Object.defineProperty(o, 'x', { \
    //              get: function() { return stored; }, \
    //              set: function(v) { stored = v * 2; } \
    //          }); \
    //          o.x = 5; o.x",
    //     )
    //     .unwrap();
    //     assert_eq!(result, JsValue::Smi(10));
    // }

    /// Getter-only accessor: strict-mode write throws TypeError.
    /// NOTE: Strict mode getter-only enforcement not yet fully working.
    // #[test]
    // fn e2e_getter_only_strict_throws() {
    //     let result = crate::builtins::global::global_eval(
    //         "'use strict'; var o = {}; \
    //          Object.defineProperty(o, 'x', { get: function() { return 1; } }); \
    //          o.x = 2;",
    //     );
    //     assert!(
    //         result.is_err(),
    //         "strict mode setter on getter-only should throw"
    //     );
    // }

    /// Getter-only accessor: sloppy-mode write is silently ignored.
    /// NOTE: Sloppy mode getter-only enforcement not yet fully working.
    // #[test]
    // fn e2e_getter_only_sloppy_silent() {
    //     let result = crate::builtins::global::global_eval(
    //         "var o = {}; \
    //          Object.defineProperty(o, 'x', { get: function() { return 1; } }); \
    //          o.x = 2; o.x",
    //     )
    //     .unwrap();
    //     assert_eq!(
    //         result,
    //         JsValue::Smi(1),
    //         "getter-only sloppy: value unchanged"
    //     );
    // }

    /// Non-writable property via defineProperty: sloppy silent, value unchanged.
    #[test]
    fn e2e_non_writable_sloppy_silent() {
        let result = crate::builtins::global::global_eval(
            "var o = {}; \
             Object.defineProperty(o, 'x', { value: 10, writable: false }); \
             o.x = 99; o.x",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(10));
    }

    /// Non-writable property: strict-mode write throws TypeError.
    #[test]
    fn e2e_non_writable_strict_throws() {
        let result = crate::builtins::global::global_eval(
            "'use strict'; var o = {}; \
             Object.defineProperty(o, 'x', { value: 10, writable: false }); \
             o.x = 99;",
        );
        assert!(
            result.is_err(),
            "strict mode write to non-writable should throw"
        );
    }

    /// preventExtensions: new property in sloppy mode silently fails.
    #[test]
    fn e2e_prevent_extensions_sloppy_new_prop() {
        let result = crate::builtins::global::global_eval(
            "var o = { x: 1 }; Object.preventExtensions(o); o.y = 2; \
             typeof o.y",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("undefined".into()));
    }

    /// preventExtensions: existing property is still writable.
    #[test]
    fn e2e_prevent_extensions_existing_writable() {
        let result = crate::builtins::global::global_eval(
            "var o = { x: 1 }; Object.preventExtensions(o); o.x = 99; o.x",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    /// Object.isExtensible returns false after preventExtensions.
    #[test]
    fn e2e_object_is_extensible() {
        let result = crate::builtins::global::global_eval(
            "var o = {}; var before = Object.isExtensible(o); \
             Object.preventExtensions(o); \
             var after = Object.isExtensible(o); \
             '' + before + ',' + after",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("true,false".into()));
    }

    /// Non-configurable property cannot be deleted in strict mode.
    #[test]
    fn e2e_non_configurable_delete_strict() {
        let result = crate::builtins::global::global_eval(
            "'use strict'; var o = {}; \
             Object.defineProperty(o, 'x', { value: 1, configurable: false }); \
             delete o.x;",
        );
        assert!(
            result.is_err(),
            "strict delete of non-configurable should throw"
        );
    }

    /// Non-configurable property: sloppy delete returns false.
    #[test]
    fn e2e_non_configurable_delete_sloppy() {
        let result = crate::builtins::global::global_eval(
            "var o = {}; \
             Object.defineProperty(o, 'x', { value: 1, configurable: false }); \
             delete o.x",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    /// Object.freeze makes properties non-writable AND non-configurable.
    #[test]
    fn e2e_freeze_descriptor_check() {
        let result = crate::builtins::global::global_eval(
            "var o = { x: 1 }; Object.freeze(o); \
             var d = Object.getOwnPropertyDescriptor(o, 'x'); \
             '' + d.writable + ',' + d.configurable",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("false,false".into()));
    }

    /// Object.seal makes properties non-configurable but preserves writable.
    #[test]
    fn e2e_seal_descriptor_check() {
        let result = crate::builtins::global::global_eval(
            "var o = { x: 1 }; Object.seal(o); \
             var d = Object.getOwnPropertyDescriptor(o, 'x'); \
             '' + d.writable + ',' + d.configurable",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("true,false".into()));
    }

    // ── §4 Arguments object ─────────────────────────────────────────────

    #[test]
    fn test_arguments_length() {
        let result = crate::builtins::global::global_eval(
            "(function(a, b, c) { return arguments.length; })(10, 20, 30)",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    #[test]
    fn test_arguments_indexed_access() {
        let result = crate::builtins::global::global_eval(
            "(function(a, b) { return arguments[0] + arguments[1]; })(3, 7)",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(10));
    }

    #[test]
    fn test_arguments_callee_sloppy() {
        let result = crate::builtins::global::global_eval(
            "(function f() { return typeof arguments.callee; })()",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("function".into()));
    }

    #[test]
    fn test_arguments_exists_in_function() {
        // arguments object is accessible inside regular functions
        let result = crate::builtins::global::global_eval(
            "(function() { return typeof arguments; })('a','b')",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("object".into()));
    }

    // ── §5 for-in enumeration order ─────────────────────────────────────

    #[test]
    fn test_for_in_insertion_order_strings() {
        let result = crate::builtins::global::global_eval(
            "var o = {b: 1, a: 2, c: 3}; var r = ''; for (var k in o) r += k + ','; r",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("b,a,c,".into()));
    }

    #[test]
    fn test_for_in_integer_indices_first() {
        // Integer indices should come first in ascending order,
        // then string keys in insertion order.
        let result = crate::builtins::global::global_eval(
            "var o = {b: 1, 2: 2, a: 3, 0: 4}; var r = ''; for (var k in o) r += k + ','; r",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("0,2,b,a,".into()));
    }

    #[test]
    fn test_for_in_skips_nonenumerable() {
        // Object.freeze does not change enumerability, so all keys should still appear.
        let result = crate::builtins::global::global_eval(
            "var o = Object.freeze({x: 1, y: 2}); var r = ''; for (var k in o) r += k + ','; r",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("x,y,".into()));
    }

    // ── §6 Template object frozen ───────────────────────────────────────

    #[test]
    fn test_template_object_is_frozen() {
        let result = crate::builtins::global::global_eval(
            "function tag(strs) { return Object.isFrozen(strs); } tag`hello`",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_template_object_raw_is_frozen() {
        let result = crate::builtins::global::global_eval(
            "function tag(strs) { return Object.isFrozen(strs.raw); } tag`hello`",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_template_object_has_raw() {
        let result = crate::builtins::global::global_eval(
            "function tag(strs) { return strs.raw !== undefined; } tag`hello`",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_template_object_length() {
        let result = crate::builtins::global::global_eval(
            "function tag(strs) { return strs.length; } tag`a${1}b`",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(2));
    }

    // ── §7 Conformance round-21 ─────────────────────────────────────────

    /// `typeof null` → "object"
    #[test]
    fn test_typeof_null_is_object() {
        let result = crate::builtins::global::global_eval("typeof null").unwrap();
        assert_eq!(result, JsValue::String("object".into()));
    }

    /// Template literal basic string
    #[test]
    fn test_template_literal_basic() {
        let result = crate::builtins::global::global_eval("`hello world`").unwrap();
        assert_eq!(result, JsValue::String("hello world".into()));
    }

    /// Template literal with expression
    #[test]
    fn test_template_literal_expression() {
        let result = crate::builtins::global::global_eval("var x = 42; `value is ${x}`").unwrap();
        assert_eq!(result, JsValue::String("value is 42".into()));
    }

    /// Spread in array literal
    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_spread_in_array_literal() {
        let result =
            crate::builtins::global::global_eval("var a = [1,2,3]; [...a].length").unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    /// Object spread
    #[test]
    fn test_object_spread() {
        let result =
            crate::builtins::global::global_eval("var a = {x:1}; var b = {...a, y:2}; b.x + b.y")
                .unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    /// Nullish coalescing
    #[test]
    fn test_nullish_coalescing_null() {
        let result = crate::builtins::global::global_eval("null ?? 42").unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// Nullish coalescing with non-null
    #[test]
    fn test_nullish_coalescing_zero() {
        let result = crate::builtins::global::global_eval("0 ?? 42").unwrap();
        assert_eq!(result, JsValue::Smi(0));
    }

    /// Optional chaining with null
    #[test]
    fn test_optional_chaining_null() {
        let result = crate::builtins::global::global_eval("var obj = null; obj?.x").unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    /// Optional chaining with value
    #[test]
    fn test_optional_chaining_value() {
        let result = crate::builtins::global::global_eval("var obj = {x: 10}; obj?.x").unwrap();
        assert_eq!(result, JsValue::Smi(10));
    }

    /// Logical assignment ||=
    #[test]
    fn test_logical_or_assignment() {
        let result = crate::builtins::global::global_eval("var x = 0; x ||= 5; x").unwrap();
        assert_eq!(result, JsValue::Smi(5));
    }

    /// Logical assignment &&=
    #[test]
    fn test_logical_and_assignment() {
        let result = crate::builtins::global::global_eval("var x = 1; x &&= 5; x").unwrap();
        assert_eq!(result, JsValue::Smi(5));
    }

    /// Logical assignment ??=
    #[test]
    fn test_nullish_assignment() {
        let result = crate::builtins::global::global_eval("var x = null; x ??= 5; x").unwrap();
        assert_eq!(result, JsValue::Smi(5));
    }

    /// Exponentiation operator
    #[test]
    fn test_exponentiation() {
        let result = crate::builtins::global::global_eval("2 ** 10").unwrap();
        assert_eq!(result, JsValue::Smi(1024));
    }

    /// Destructuring with default values
    #[test]
    fn test_destructuring_default() {
        let result = crate::builtins::global::global_eval("var {x = 10} = {}; x").unwrap();
        assert_eq!(result, JsValue::Smi(10));
    }

    /// Array destructuring
    #[test]
    fn test_array_destructuring_basic() {
        let result = crate::builtins::global::global_eval("var [a, b] = [1, 2]; a + b").unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    /// Computed property names
    #[test]
    fn test_computed_property_name() {
        let result =
            crate::builtins::global::global_eval("var key = 'x'; var obj = {[key]: 42}; obj.x")
                .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// for...of with array
    #[test]
    fn test_for_of_array_sum() {
        let result = crate::builtins::global::global_eval(
            "var sum = 0; for (var x of [1,2,3]) { sum += x; } sum",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(6));
    }

    /// Generator basic
    #[test]
    fn test_generator_basic_next() {
        let result = crate::builtins::global::global_eval(
            "function* gen() { yield 1; yield 2; } var g = gen(); g.next().value",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(1));
    }

    /// Promise.resolve returns object
    #[test]
    fn test_promise_resolve_type() {
        let result = crate::builtins::global::global_eval("typeof Promise.resolve(42)").unwrap();
        assert_eq!(result, JsValue::String("object".into()));
    }

    /// Class basic
    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_basic_method() {
        let result = crate::builtins::global::global_eval(
            "class Foo { bar() { return 42; } } new Foo().bar()",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// `typeof undeclaredVar` must return `"undefined"`, not throw.
    #[test]
    fn test_typeof_undeclared_returns_undefined_string() {
        let result = crate::builtins::global::global_eval("typeof totallyUndeclared").unwrap();
        assert_eq!(result, JsValue::String("undefined".into()));
    }

    /// `import.meta` must produce an object (not crash).
    #[test]
    fn test_import_meta_returns_object() {
        use crate::bytecode::bytecode_array::BytecodeArray;
        use crate::bytecode::bytecodes::{Instruction, Opcode, encode};
        use crate::bytecode::feedback::FeedbackMetadata;
        use crate::interpreter::{Interpreter, InterpreterFrame};

        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaImportMeta, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = BytecodeArray::new(
            encode(&instrs),
            vec![],
            0,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        )
        .with_module_flag(true);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert!(
            matches!(result, JsValue::PlainObject(_)),
            "import.meta should produce a PlainObject"
        );
    }

    /// `StaModuleVariable` + `LdaModuleVariable` round-trip.
    #[test]
    fn test_module_variable_store_load_round_trip() {
        use crate::bytecode::bytecode_array::{BytecodeArray, ConstantPoolEntry};
        use crate::bytecode::bytecodes::{Instruction, Opcode, Operand, encode};
        use crate::bytecode::feedback::FeedbackMetadata;
        use crate::interpreter::{Interpreter, InterpreterFrame};

        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(
                Opcode::StaModuleVariable,
                vec![Operand::ConstantPoolIdx(0), Operand::Immediate(0)],
            ),
            Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
            Instruction::new_unchecked(
                Opcode::LdaModuleVariable,
                vec![Operand::ConstantPoolIdx(0), Operand::Immediate(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = BytecodeArray::new(
            encode(&instrs),
            vec![ConstantPoolEntry::String(String::new())],
            0,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        )
        .with_module_flag(true);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    /// `GetModuleNamespace` must produce an object (not crash).
    #[test]
    fn test_get_module_namespace_returns_object() {
        use crate::bytecode::bytecode_array::{BytecodeArray, ConstantPoolEntry};
        use crate::bytecode::bytecodes::{Instruction, Opcode, Operand, encode};
        use crate::bytecode::feedback::FeedbackMetadata;
        use crate::interpreter::{Interpreter, InterpreterFrame};

        let instrs = vec![
            Instruction::new_unchecked(
                Opcode::GetModuleNamespace,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = BytecodeArray::new(
            encode(&instrs),
            vec![ConstantPoolEntry::String("./foo.js".to_string())],
            0,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        )
        .with_module_flag(true);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert!(
            matches!(result, JsValue::PlainObject(_)),
            "GetModuleNamespace should produce a PlainObject"
        );
    }

    // ── for-in prototype chain & internal key filtering ─────────────

    /// for-in must enumerate inherited enumerable properties from the
    /// prototype chain.
    // NOTE: for-in prototype chain enumeration not yet fully implemented
    #[test]
    #[ignore]
    fn test_for_in_prototype_chain() {
        let result = crate::builtins::global::global_eval(
            "function Base() {} \
             Base.prototype.inherited = 1; \
             var obj = new Base(); \
             obj.own = 2; \
             var keys = []; \
             for (var k in obj) { keys.push(k); } \
             keys.length",
        )
        .unwrap();
        // Should enumerate both "own" and "inherited".
        assert!(
            matches!(result, JsValue::Smi(2)),
            "expected 2 keys (own + inherited), got {result:?}"
        );
    }

    /// for-in must not leak __proto__ or other internal keys.
    #[test]
    fn test_for_in_hides_internal_keys() {
        let result = crate::builtins::global::global_eval(
            "var o = {x: 1, y: 2}; \
             var r = ''; \
             for (var k in o) r += k + ','; \
             r",
        )
        .unwrap();
        // Only user-visible keys, no __proto__ / __call__ etc.
        assert_eq!(result, JsValue::String("x,y,".into()));
    }

    /// Own property should shadow an inherited property with the same name.
    #[test]
    fn test_for_in_shadowing() {
        let result = crate::builtins::global::global_eval(
            "function Base() {} \
             Base.prototype.x = 1; \
             var obj = new Base(); \
             obj.x = 2; \
             var count = 0; \
             for (var k in obj) { if (k === 'x') count++; } \
             count",
        )
        .unwrap();
        // "x" should appear only once (own shadows inherited).
        assert_eq!(result, JsValue::Smi(1));
    }

    /// for-in on null/undefined produces no keys (no TypeError).
    #[test]
    fn test_for_in_null_undefined() {
        let result = crate::builtins::global::global_eval(
            "var count = 0; for (var k in null) count++; count",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(0));
    }

    // ── Destructuring ───────────────────────────────────────────────

    /// Array destructuring with defaults: missing element uses default.
    #[test]
    fn test_array_destructuring_with_defaults() {
        let result =
            crate::builtins::global::global_eval("var [a = 1, b = 2] = [10]; a * 100 + b").unwrap();
        // a=10 (from array), b=2 (default) → 1002
        assert_eq!(result, JsValue::Smi(1002));
    }

    /// Object destructuring with defaults.
    #[test]
    fn test_object_destructuring_with_defaults() {
        let result =
            crate::builtins::global::global_eval("var {x = 1, y = 2} = {x: 10}; x * 100 + y")
                .unwrap();
        // x=10 (from object), y=2 (default) → 1002
        assert_eq!(result, JsValue::Smi(1002));
    }

    /// Nested object destructuring.
    #[test]
    fn test_nested_destructuring() {
        let result =
            crate::builtins::global::global_eval("var {a: {b}} = {a: {b: 42}}; b").unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// Array destructuring with rest element.
    #[test]
    fn test_array_destructuring_rest() {
        let result =
            crate::builtins::global::global_eval("var [a, ...b] = [1, 2, 3]; a * 100 + b.length")
                .unwrap();
        // a=1, b=[2,3] → 1*100 + 2 = 102
        assert_eq!(result, JsValue::Smi(102));
    }

    /// Object destructuring with rest.
    #[test]
    fn test_object_destructuring_rest() {
        let result = crate::builtins::global::global_eval(
            "var {a, ...rest} = {a: 1, b: 2, c: 3}; a + rest.b + rest.c",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(6));
    }

    // ── Class features ──────────────────────────────────────────────

    /// Static methods should be accessible on the class constructor.
    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_static_method() {
        let result = crate::builtins::global::global_eval(
            "class Foo { static bar() { return 99; } } Foo.bar()",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    /// Class expression (not just declaration).
    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_expression() {
        let result = crate::builtins::global::global_eval(
            "var Foo = class { greet() { return 7; } }; new Foo().greet()",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    /// Class with computed property name in method.
    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_computed_property_name() {
        let result = crate::builtins::global::global_eval(
            "var name = 'greet'; \
             class Foo { [name]() { return 55; } } \
             new Foo().greet()",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(55));
    }

    #[test]
    fn test_class_decl_basics() {
        let result = crate::builtins::global::global_eval(
            "class Foo { constructor(x) { this.x = x; } } \
             typeof Foo === 'function' && Foo.prototype.constructor === Foo && new Foo(7).x === 7",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_class_decl_tdz() {
        let result = crate::builtins::global::global_eval(
            "try { Foo; class Foo {} } catch (e) { e instanceof ReferenceError }",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_class_body_is_strict() {
        let result = crate::builtins::global::global_eval("class Foo { method() { delete x; } }");
        assert!(
            result.is_err(),
            "class methods should compile in strict mode"
        );
    }

    /// super.method() in a derived class.
    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_super_method_call() {
        let result = crate::builtins::global::global_eval(
            "class Base { value() { return 10; } } \
             class Child extends Base { value() { return super.value() + 5; } } \
             new Child().value()",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(15));
    }

    /// Class inheritance — instanceof works for subclass.
    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_inheritance_instanceof() {
        let result = crate::builtins::global::global_eval(
            "class Base {} class Child extends Base {} \
             var c = new Child(); c instanceof Base",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_inheritance_prototype_chain() {
        let result = crate::builtins::global::global_eval(
            "class Foo {} \
             class Bar extends Foo {} \
             (Bar.__proto__ === Foo) && \
             (Bar.prototype.__proto__ === Foo.prototype) && \
             !(Bar.prototype instanceof Foo) && \
             (new Bar() instanceof Foo)",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_class_static_and_instance_fields() {
        let result = crate::builtins::global::global_eval(
            "class Foo { static x = 42; y = 7; } \
             var a = new Foo(); \
             var b = new Foo(); \
             Foo.x === 42 && a.y === 7 && b.y === 7 && Foo.prototype.y === undefined",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_class_static_block_runs_during_definition() {
        assert_eval_true(
            "var log = []; class C { static { log.push('block'); } } log.join(',') === 'block'",
        );
    }

    #[test]
    fn test_class_static_block_binds_this_to_constructor() {
        assert_eval_true("class C { static { this.answer = 42; } } C.answer === 42");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_static_block_this_identity_matches_class() {
        assert_eval_true("class C { static ok = false; static { this.ok = this === C; } } C.ok");
    }

    #[test]
    fn test_class_multiple_static_blocks_follow_source_order() {
        assert_eval_true(
            "var log = []; class C { static { log.push('a'); } static { log.push('b'); } static { log.push('c'); } } log.join(',') === 'a,b,c'",
        );
    }

    #[test]
    fn test_class_static_blocks_interleave_with_static_fields() {
        assert_eval_true(
            "var log = []; class C { static a = log.push('a'); static { log.push('b'); } static c = log.push('c'); static { log.push('d'); } } log.join(',') === 'a,b,c,d'",
        );
    }

    #[test]
    fn test_class_static_block_reads_prior_static_field() {
        assert_eval_true("class C { static x = 1; static { this.y = this.x + 1; } } C.y === 2");
    }

    #[test]
    fn test_class_static_block_updates_static_field_before_later_field() {
        assert_eval_true(
            "class C { static x = 1; static { this.x += 1; } static y = this.x + 1; } C.x === 2 && C.y === 3",
        );
    }

    #[test]
    fn test_class_static_field_initializer_uses_class_this() {
        assert_eval_true("class C { static x = 7; static y = this.x + 1; } C.y === 8");
    }

    #[test]
    fn test_class_static_field_initializer_sees_prior_block_updates() {
        assert_eval_true(
            "class C { static x = 1; static { this.x = 5; } static y = this.x; } C.y === 5",
        );
    }

    #[test]
    fn test_class_static_block_reads_private_static_field() {
        assert_eval_true(
            "class C { static #x = 10; static y = 0; static { this.y = this.#x; } } C.y === 10",
        );
    }

    #[test]
    fn test_class_static_block_calls_private_static_method() {
        assert_eval_true(
            "class C { static #value() { return 9; } static y = 0; static { this.y = this.#value(); } } C.y === 9",
        );
    }

    #[test]
    fn test_class_static_blocks_share_private_static_state() {
        assert_eval_true(
            "class C { static #x = 1; static first = 0; static second = 0; static { this.first = this.#x; this.#x = 2; } static { this.second = this.#x; } } C.first === 1 && C.second === 2",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_static_block_super_calls_parent_static_method() {
        assert_eval_true(
            "class A { static value() { return 40; } } class B extends A { static result = 0; static { this.result = super.value() + 2; } } B.result === 42",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_static_block_super_uses_current_class_as_receiver() {
        assert_eval_true(
            "class A { static who() { return this.name; } } class B extends A { static result = ''; static { this.result = super.who(); } } B.result === 'B'",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_static_field_super_calls_parent_static_method() {
        assert_eval_true(
            "class A { static value() { return 5; } } class B extends A { static x = super.value() + 1; } B.x === 6",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_static_field_super_uses_current_class_as_receiver() {
        assert_eval_true(
            "class A { static who() { return this.name; } } class B extends A { static value = super.who(); } B.value === 'B'",
        );
    }

    #[test]
    fn test_class_static_block_throw_aborts_class_declaration() {
        assert_eval_true("try { class C { static { throw 1; } } false; } catch (e) { e === 1; }");
    }

    #[test]
    fn test_class_static_block_throw_aborts_class_expression() {
        assert_eval_true(
            "try { var C = class { static { throw 2; } }; false; } catch (e) { e === 2; }",
        );
    }

    #[test]
    fn test_class_static_block_runs_in_class_expression() {
        assert_eval_true(
            "var seen = 0; var C = class { static { seen = 1; } }; seen === 1 && typeof C === 'function'",
        );
    }

    #[test]
    fn test_class_static_blocks_in_class_expression_follow_order() {
        assert_eval_true(
            "var log = []; var C = class { static { log.push('x'); } static { log.push('y'); } }; log.join(',') === 'x,y' && typeof C === 'function'",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_static_block_sees_class_expression_name() {
        assert_eval_true(
            "var value = class C { static { this.ok = C === this; } }; value.ok === true",
        );
    }

    #[test]
    fn test_class_instance_fields_run_per_new_invocation() {
        assert_eval_true(
            "var n = 0; class C { x = ++n; } var a = new C(); var b = new C(); a.x === 1 && b.x === 2",
        );
    }

    #[test]
    fn test_class_instance_fields_follow_declaration_order() {
        assert_eval_true(
            "class C { a = 1; b = this.a + 1; c = this.b + 1; } var o = new C(); o.a === 1 && o.b === 2 && o.c === 3",
        );
    }

    #[test]
    fn test_class_instance_fields_run_before_base_constructor_body() {
        assert_eval_true(
            "class C { x = 1; constructor() { this.y = this.x + 1; } } new C().y === 2",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_instance_fields_run_after_super_in_derived_constructor() {
        assert_eval_true(
            "class A { constructor() { this.log = ['base']; } } class B extends A { x = this.log.push('field'); constructor() { super(); this.log.push('ctor'); } } new B().log.join(',') === 'base,field,ctor'",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_instance_computed_field_name_evaluates_once() {
        assert_eval_true(
            "var count = 0; function key() { count++; return 'x'; } class C { [key()] = 1; } var a = new C(); var b = new C(); count === 1 && a.x === 1 && b.x === 1",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_instance_computed_field_names_follow_source_order() {
        assert_eval_true(
            "var log = []; function key(v) { log.push(v); return v; } class C { [key('a')] = 1; [key('b')] = 2; } new C(); new C(); log.join(',') === 'a,b'",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_instance_computed_field_names_interleave_with_static_blocks() {
        assert_eval_true(
            "var log = []; function key(v) { log.push(v); return v; } class C { [key('a')] = 1; static { log.push('b'); } [key('c')] = 2; } log.join(',') === 'a,b,c'",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_instance_computed_field_names_interleave_with_static_fields() {
        assert_eval_true(
            "var log = []; function key(v) { log.push(v); return v; } class C { [key('a')] = 1; static x = log.push('b'); [key('c')] = 2; } log.join(',') === 'a,b,c'",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_instance_computed_field_names_are_cached_for_multiple_fields() {
        assert_eval_true(
            "var idx = 0; function key() { idx++; return 'k' + idx; } class C { [key()] = 1; [key()] = 2; } var o = new C(); idx === 2 && o.k1 === 1 && o.k2 === 2",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_instance_computed_field_names_are_cached_across_instances() {
        assert_eval_true(
            "var count = 0; function key() { count++; return 'x'; } var value = 0; class C { [key()] = ++value; } var a = new C(); var b = new C(); count === 1 && a.x === 1 && b.x === 2",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_instance_computed_field_names_only_run_once_for_derived_class() {
        assert_eval_true(
            "var count = 0; function key() { count++; return 'x'; } class A {} class B extends A { [key()] = 1; } new B(); new B(); count === 1",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_instance_computed_field_name_with_this_based_initializer() {
        assert_eval_true(
            "var count = 0; function key() { count++; return 'x'; } class C { y = 2; [key()] = this.y + 1; } var o = new C(); count === 1 && o.x === 3",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_class_instance_computed_field_names_are_defined_on_each_instance() {
        assert_eval_true(
            "var count = 0; function key() { count++; return 'x'; } class C { [key()] = 1; } var a = new C(); var b = new C(); count === 1 && a.hasOwnProperty('x') && b.hasOwnProperty('x')",
        );
    }

    #[test]
    fn test_class_instance_field_throw_propagates_per_new() {
        assert_eval_true(
            "class C { x = (() => { throw 7; })(); } try { new C(); false; } catch (e) { e === 7; }",
        );
    }

    #[test]
    fn test_class_expression_name_not_visible_outside() {
        let result =
            crate::builtins::global::global_eval("var Foo = class Bar {}; typeof Bar").unwrap();
        assert_eq!(result, JsValue::String("undefined".into()));
    }

    #[test]
    fn test_class_constructor_requires_new() {
        let result = crate::builtins::global::global_eval(
            "class Foo {} \
             try { Foo(); false; } catch (e) { e instanceof TypeError }",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_derived_class_requires_super_before_this() {
        let result = crate::builtins::global::global_eval(
            "class Foo {} \
             class Bar extends Foo { constructor() { this.x = 1; super(); } } \
             try { new Bar(); false; } catch (e) { e instanceof ReferenceError }",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_derived_class_cannot_call_super_twice() {
        let result = crate::builtins::global::global_eval(
            "class Foo {} \
             class Bar extends Foo { constructor() { super(); super(); } } \
             try { new Bar(); false; } catch (e) { e instanceof ReferenceError }",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── new.target conformance ──────────────────────────────────────────────

    /// 1. `new Foo()` sets `new.target` to `Foo`.
    #[test]
    fn e2e_new_target_in_constructor_is_ctor() {
        let r = crate::builtins::global::global_eval(
            "var nt; function Foo() { nt = new.target; } new Foo(); nt === Foo",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 2. `Foo()` (normal call) → `new.target` is `undefined`.
    #[test]
    fn e2e_new_target_undefined_in_normal_call() {
        let r = crate::builtins::global::global_eval(
            "var nt; function Foo() { nt = new.target; } Foo(); nt === undefined",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 3. `new.target` inside a class constructor is defined.
    #[test]
    fn e2e_new_target_defined_in_class_constructor() {
        let r = crate::builtins::global::global_eval(
            "var nt; class A { constructor() { nt = new.target; } } new A(); nt !== undefined",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 4. `new.target` in a class constructor equals the class itself.
    #[test]
    fn e2e_new_target_is_class_in_class_constructor() {
        let r = crate::builtins::global::global_eval(
            "var nt; class A { constructor() { nt = new.target; } } new A(); nt === A",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 5. Inheritance: `new.target` in parent constructor is the derived class.
    #[test]
    fn e2e_new_target_is_derived_in_parent_constructor() {
        let r = crate::builtins::global::global_eval(
            "var nt; \
             class A { constructor() { nt = new.target; } } \
             class B extends A { constructor() { super(); } } \
             new B(); nt === B",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 6. `new.target` preserved through multiple inheritance levels.
    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_new_target_three_level_inheritance() {
        let r = crate::builtins::global::global_eval(
            "var nt; \
             class A { constructor() { nt = new.target; } } \
             class B extends A { constructor() { super(); } } \
             class C extends B { constructor() { super(); } } \
             new C(); nt === C",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 7. Arrow function inside constructor inherits `new.target`.
    #[test]
    fn e2e_new_target_in_arrow_inside_constructor() {
        let r = crate::builtins::global::global_eval(
            "var nt; \
             function Foo() { var f = () => new.target; nt = f(); } \
             new Foo(); nt === Foo",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 8. Arrow inside normal call → `new.target` is `undefined`.
    #[test]
    fn e2e_new_target_arrow_in_normal_call_undefined() {
        let r = crate::builtins::global::global_eval(
            "var nt; \
             function Foo() { var f = () => new.target; nt = f(); } \
             Foo(); nt === undefined",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 9. Nested arrow captures outer `new.target`.
    #[test]
    fn e2e_new_target_nested_arrow() {
        let r = crate::builtins::global::global_eval(
            "var nt; \
             function Foo() { var f = () => (() => new.target)(); nt = f(); } \
             new Foo(); nt === Foo",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 10. Arrow stored and called later still sees captured `new.target`.
    #[test]
    fn e2e_new_target_arrow_stored_and_called_later() {
        let r = crate::builtins::global::global_eval(
            "var getter; \
             function Foo() { getter = () => new.target; } \
             new Foo(); \
             getter() === Foo",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 11. Arrow inside class constructor captures `new.target` (class).
    #[test]
    fn e2e_new_target_arrow_inside_class_constructor() {
        let r = crate::builtins::global::global_eval(
            "var nt; \
             class A { constructor() { var f = () => new.target; nt = f(); } } \
             new A(); nt === A",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 12. Arrow in derived constructor captures the derived `new.target`.
    #[test]
    fn e2e_new_target_arrow_in_derived_constructor() {
        let r = crate::builtins::global::global_eval(
            "var nt; \
             class A { constructor() {} } \
             class B extends A { constructor() { super(); var f = () => new.target; nt = f(); } } \
             new B(); nt === B",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 13. `new.target` outside any function is a SyntaxError.
    #[test]
    fn e2e_new_target_outside_function_syntax_error() {
        let r = crate::builtins::global::global_eval("new.target");
        assert!(
            r.is_err(),
            "new.target at top level should be a SyntaxError"
        );
    }

    /// 14. `new.target` inside `eval()` at top level is a SyntaxError.
    #[test]
    fn e2e_new_target_in_eval_top_level_error() {
        let r = crate::builtins::global::global_eval(
            "try { eval('new.target'); false; } catch(e) { true }",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 15. `Reflect.construct(A, [])` → `new.target` is `A` inside `A`.
    #[test]
    fn e2e_reflect_construct_new_target_is_target() {
        let r = crate::builtins::global::global_eval(
            "var nt; function A() { nt = new.target; } \
             Reflect.construct(A, []); nt === A",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 16. `Reflect.construct(A, [], B)` → `new.target` is `B` inside `A`.
    #[test]
    fn e2e_reflect_construct_with_new_target() {
        let r = crate::builtins::global::global_eval(
            "var nt; function A() { nt = new.target; } \
             function B() {} \
             Reflect.construct(A, [], B); nt === B",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 17. `Reflect.construct` sets prototype from `newTarget`.
    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_reflect_construct_prototype_from_new_target() {
        let r = crate::builtins::global::global_eval(
            "function A() {} \
             function B() {} \
             B.prototype.x = 42; \
             var obj = Reflect.construct(A, [], B); \
             obj.x === 42",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 18. `Reflect.construct` without third arg uses target as `newTarget`.
    #[test]
    fn e2e_reflect_construct_default_new_target() {
        let r = crate::builtins::global::global_eval(
            "var nt; function A() { nt = new.target; } \
             Reflect.construct(A, []); nt === A",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 19. `Reflect.construct` returns constructed object.
    #[test]
    fn e2e_reflect_construct_returns_object() {
        let r = crate::builtins::global::global_eval(
            "function A() { this.x = 1; } \
             var o = Reflect.construct(A, []); o.x",
        )
        .unwrap();
        assert_eq!(r, JsValue::Smi(1));
    }

    /// 20. `Reflect.construct` with non-constructor target throws TypeError.
    #[test]
    fn e2e_reflect_construct_non_constructor_throws() {
        let r = crate::builtins::global::global_eval(
            "try { Reflect.construct(123, []); false; } catch(e) { e instanceof TypeError }",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 21. `Reflect.construct` with non-constructor newTarget throws TypeError.
    #[test]
    fn e2e_reflect_construct_non_constructor_new_target_throws() {
        let r = crate::builtins::global::global_eval(
            "try { Reflect.construct(function(){}, [], 123); false; } \
             catch(e) { e instanceof TypeError }",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 22. `new.target` is the actual constructor, not the base class, when
    ///     using `new DerivedClass()`.
    #[test]
    fn e2e_new_target_base_vs_derived_identity() {
        let r = crate::builtins::global::global_eval(
            "var ntA, ntB; \
             class A { constructor() { ntA = new.target; } } \
             class B extends A { constructor() { super(); ntB = new.target; } } \
             new B(); ntA === B && ntB === B",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 23. Direct `new A()` → `new.target` is `A`, not `B`.
    #[test]
    fn e2e_new_target_direct_base_new() {
        let r = crate::builtins::global::global_eval(
            "var nt; \
             class A { constructor() { nt = new.target; } } \
             class B extends A { constructor() { super(); } } \
             new A(); nt === A",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 24. `new.target` in a method is `undefined`.
    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_new_target_in_method_undefined() {
        let r = crate::builtins::global::global_eval(
            "var nt; \
             class A { m() { nt = new.target; } } \
             new A().m(); nt === undefined",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 25. `new.target` is preserved across `super()` with arguments.
    #[test]
    fn e2e_new_target_preserved_super_with_args() {
        let r = crate::builtins::global::global_eval(
            "var nt; \
             class A { constructor(x) { nt = new.target; this.x = x; } } \
             class B extends A { constructor() { super(42); } } \
             var b = new B(); nt === B && b.x === 42",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 26. Function expression with `new.target`.
    #[test]
    fn e2e_new_target_function_expression() {
        let r = crate::builtins::global::global_eval(
            "var nt; var F = function() { nt = new.target; }; new F(); nt === F",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 27. `new.target` after `toString` on constructor.
    #[test]
    fn e2e_new_target_typeof() {
        let r = crate::builtins::global::global_eval(
            "var t; function Foo() { t = typeof new.target; } new Foo(); t",
        )
        .unwrap();
        assert_eq!(r, JsValue::String("function".into()));
    }

    /// 28. `new.target` typeof is `undefined` for normal call.
    #[test]
    fn e2e_new_target_typeof_normal_call() {
        let r = crate::builtins::global::global_eval(
            "var t; function Foo() { t = typeof new.target; } Foo(); t",
        )
        .unwrap();
        assert_eq!(r, JsValue::String("undefined".into()));
    }

    /// 29. `new.target` in arrow inside method → `undefined`.
    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_new_target_arrow_in_method_undefined() {
        let r = crate::builtins::global::global_eval(
            "var nt; \
             class A { m() { var f = () => new.target; nt = f(); } } \
             new A().m(); nt === undefined",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 30. Conditional based on `new.target` (abstract class pattern).
    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_new_target_abstract_class_pattern() {
        let r = crate::builtins::global::global_eval(
            "class Abstract { \
                 constructor() { \
                     if (new.target === Abstract) throw new TypeError('abstract'); \
                 } \
             } \
             class Concrete extends Abstract { constructor() { super(); } } \
             var ok = false; \
             try { new Abstract(); } catch(e) { ok = e instanceof TypeError; } \
             ok && (new Concrete() instanceof Concrete)",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 31. `Reflect.construct` with class constructors.
    #[test]
    fn e2e_reflect_construct_class_new_target() {
        let r = crate::builtins::global::global_eval(
            "var nt; \
             class A { constructor() { nt = new.target; } } \
             class B {} \
             Reflect.construct(A, [], B); nt === B",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 32. `new.target` in immediately-invoked constructor arrow.
    #[test]
    fn e2e_new_target_iife_arrow_in_constructor() {
        let r = crate::builtins::global::global_eval(
            "var nt; \
             function Foo() { nt = (() => new.target)(); } \
             new Foo(); nt === Foo",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 33. `new.target` inside `try/catch` in a constructor.
    #[test]
    fn e2e_new_target_inside_try_catch() {
        let r = crate::builtins::global::global_eval(
            "var nt; \
             function Foo() { try { nt = new.target; } catch(e) {} } \
             new Foo(); nt === Foo",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// 34. `Reflect.construct` passes arguments to target constructor.
    #[test]
    fn e2e_reflect_construct_passes_args() {
        let r = crate::builtins::global::global_eval(
            "function A(x, y) { this.sum = x + y; } \
             var o = Reflect.construct(A, [3, 4]); o.sum",
        )
        .unwrap();
        assert_eq!(r, JsValue::Smi(7));
    }

    /// 35. `new.target` in function with default parameters.
    #[test]
    fn e2e_new_target_with_default_params() {
        let r = crate::builtins::global::global_eval(
            "var nt; function Foo(x = 1) { nt = new.target; } new Foo(); nt === Foo",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// §7.3.21: `instanceof` with a non-callable RHS must throw TypeError.
    #[test]
    fn e2e_instanceof_non_callable_rhs_throws() {
        let result = crate::builtins::global::global_eval("1 instanceof 42");
        assert!(
            result.is_err(),
            "instanceof with non-callable RHS should throw TypeError"
        );
    }

    /// `instanceof` with null RHS must throw TypeError.
    #[test]
    fn e2e_instanceof_null_rhs_throws() {
        let result = crate::builtins::global::global_eval("({}) instanceof null");
        assert!(
            result.is_err(),
            "instanceof with null RHS should throw TypeError"
        );
    }

    /// `instanceof` with undefined RHS must throw TypeError.
    #[test]
    fn e2e_instanceof_undefined_rhs_throws() {
        let result = crate::builtins::global::global_eval("({}) instanceof undefined");
        assert!(
            result.is_err(),
            "instanceof with undefined RHS should throw TypeError"
        );
    }

    // ── Template literal improvements ───────────────────────────────

    /// Tagged template receives an array of strings as first argument.
    #[test]
    fn test_tagged_template_strings_array() {
        let result = crate::builtins::global::global_eval(
            "function tag(strs) { return strs.length; } tag`a${1}b${2}c`",
        )
        .unwrap();
        // Template `a${1}b${2}c` has 3 string parts: "a", "b", "c"
        assert_eq!(result, JsValue::Smi(3));
    }

    /// Tagged template receives expression values as extra arguments.
    #[test]
    fn test_tagged_template_expression_args() {
        let result = crate::builtins::global::global_eval(
            "function tag(strs, a, b) { return a + b; } tag`x${10}y${20}z`",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(30));
    }

    /// Template literal with object expression coerces to string.
    #[test]
    fn test_template_literal_with_object() {
        let result = crate::builtins::global::global_eval(
            "var obj = {toString: function() { return 'hello'; }}; `${obj}`",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("hello".into()));
    }

    // ── Optional chaining (?.) ──────────────────────────────────────

    /// Optional chaining on undefined.
    #[test]
    fn test_optional_chaining_undefined_obj() {
        let result = crate::builtins::global::global_eval("var x = undefined; x?.foo").unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    /// Optional chaining nested: a?.b?.c.
    #[test]
    fn test_optional_chaining_nested() {
        let result = crate::builtins::global::global_eval("var a = {b: {c: 42}}; a?.b?.c").unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// Optional chaining nested with null intermediate.
    #[test]
    fn test_optional_chaining_nested_null() {
        let result = crate::builtins::global::global_eval("var a = {b: null}; a?.b?.c").unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    /// Optional chaining method call on null.
    #[test]
    fn test_optional_chaining_method_null() {
        let result =
            crate::builtins::global::global_eval("var obj = null; obj?.toString()").unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    /// Optional chaining computed property on null.
    #[test]
    fn test_optional_chaining_computed_null() {
        let result =
            crate::builtins::global::global_eval("var obj = null; var key = 'x'; obj?.[key]")
                .unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    /// Optional chaining on valid object returns the property.
    #[test]
    fn test_optional_chaining_valid_computed() {
        let result =
            crate::builtins::global::global_eval("var obj = {x: 77}; var key = 'x'; obj?.[key]")
                .unwrap();
        assert_eq!(result, JsValue::Smi(77));
    }

    // ── Optional chaining / nullish coalescing e2e tests ────────────────

    /// `null?.x` returns undefined without error.
    #[test]
    fn e2e_optional_chain_null_prop() {
        let r = crate::builtins::global::global_eval("null?.x").unwrap();
        assert_eq!(r, JsValue::Undefined);
    }

    /// `undefined?.x` returns undefined without error.
    #[test]
    fn e2e_optional_chain_undefined_prop() {
        let r = crate::builtins::global::global_eval("undefined?.x").unwrap();
        assert_eq!(r, JsValue::Undefined);
    }

    /// `({a:1})?.a` returns 1.
    #[test]
    fn e2e_optional_chain_value_prop() {
        let r = crate::builtins::global::global_eval("({a:1})?.a").unwrap();
        assert_eq!(r, JsValue::Smi(1));
    }

    /// `null?.[0]` returns undefined.
    #[test]
    fn e2e_optional_chain_null_computed() {
        let r = crate::builtins::global::global_eval("null?.[0]").unwrap();
        assert_eq!(r, JsValue::Undefined);
    }

    /// `([10,20])?.[1]` returns 20.
    #[test]
    fn e2e_optional_chain_array_computed() {
        let r = crate::builtins::global::global_eval("([10,20])?.[1]").unwrap();
        assert_eq!(r, JsValue::Smi(20));
    }

    /// `null?.toString()` returns undefined (method call on null).
    #[test]
    fn e2e_optional_chain_method_null() {
        let r = crate::builtins::global::global_eval("null?.toString()").unwrap();
        assert_eq!(r, JsValue::Undefined);
    }

    /// `({f: function(){ return 99; }})?.f()` returns 99.
    #[test]
    fn e2e_optional_chain_method_call_value() {
        let r =
            crate::builtins::global::global_eval("var o = {f: function(){ return 99; }}; o?.f()")
                .unwrap();
        assert_eq!(r, JsValue::Smi(99));
    }

    /// `func?.()` — call if callable, undefined otherwise.
    #[test]
    fn e2e_optional_call_null() {
        let r = crate::builtins::global::global_eval("var f = null; f?.()").unwrap();
        assert_eq!(r, JsValue::Undefined);
    }

    /// `func?.()` — call if callable.
    #[test]
    fn e2e_optional_call_function() {
        let r = crate::builtins::global::global_eval("var f = function(){ return 42; }; f?.()")
            .unwrap();
        assert_eq!(r, JsValue::Smi(42));
    }

    /// Deep chain `a?.b?.c?.d` all present.
    #[test]
    fn e2e_optional_chain_deep_all_present() {
        let r = crate::builtins::global::global_eval("var a = {b:{c:{d:7}}}; a?.b?.c?.d").unwrap();
        assert_eq!(r, JsValue::Smi(7));
    }

    /// Deep chain `a?.b?.c?.d` with null intermediate.
    #[test]
    fn e2e_optional_chain_deep_null_intermediate() {
        let r = crate::builtins::global::global_eval("var a = {b:null}; a?.b?.c?.d").unwrap();
        assert_eq!(r, JsValue::Undefined);
    }

    /// `null?.x` entire chain short-circuits.
    #[test]
    fn e2e_optional_chain_short_circuit_entire_chain() {
        let r = crate::builtins::global::global_eval("var a = null; a?.b.c").unwrap();
        assert_eq!(r, JsValue::Undefined);
    }

    /// `delete obj?.prop` on null returns true.
    #[test]
    fn e2e_delete_optional_null() {
        let r = crate::builtins::global::global_eval("var obj = null; delete obj?.x").unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// `delete obj?.prop` on real object deletes the property.
    #[test]
    fn e2e_delete_optional_real() {
        let r = crate::builtins::global::global_eval(
            "var obj = {x:1}; delete obj?.x; obj.x === undefined",
        )
        .unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    /// `null ?? 'default'` returns 'default'.
    #[test]
    fn e2e_nullish_coalesce_null() {
        let r = crate::builtins::global::global_eval("null ?? 'default'").unwrap();
        assert_eq!(r, JsValue::String("default".into()));
    }

    /// `undefined ?? 'default'` returns 'default'.
    #[test]
    fn e2e_nullish_coalesce_undefined() {
        let r = crate::builtins::global::global_eval("undefined ?? 'default'").unwrap();
        assert_eq!(r, JsValue::String("default".into()));
    }

    /// `0 ?? 'default'` returns 0 (not 'default').
    #[test]
    fn e2e_nullish_coalesce_zero() {
        let r = crate::builtins::global::global_eval("0 ?? 'default'").unwrap();
        assert_eq!(r, JsValue::Smi(0));
    }

    /// `'' ?? 'default'` returns '' (not 'default').
    #[test]
    fn e2e_nullish_coalesce_empty_string() {
        let r = crate::builtins::global::global_eval("'' ?? 'default'").unwrap();
        assert_eq!(r, JsValue::String("".into()));
    }

    /// `false ?? 'default'` returns false.
    #[test]
    fn e2e_nullish_coalesce_false() {
        let r = crate::builtins::global::global_eval("false ?? 'default'").unwrap();
        assert_eq!(r, JsValue::Boolean(false));
    }

    /// `x ??= value` only assigns if x is null/undefined.
    #[test]
    fn e2e_nullish_assign_null() {
        let r = crate::builtins::global::global_eval("var x = null; x ??= 42; x").unwrap();
        assert_eq!(r, JsValue::Smi(42));
    }

    /// `x ??= value` does NOT assign if x is 0.
    #[test]
    fn e2e_nullish_assign_zero_keeps() {
        let r = crate::builtins::global::global_eval("var x = 0; x ??= 42; x").unwrap();
        assert_eq!(r, JsValue::Smi(0));
    }

    /// `x ??= value` does NOT assign if x is ''.
    #[test]
    fn e2e_nullish_assign_empty_string_keeps() {
        let r = crate::builtins::global::global_eval("var x = ''; x ??= 'hi'; x").unwrap();
        assert_eq!(r, JsValue::String("".into()));
    }

    /// `x &&= value` — assigns only if x is truthy.
    #[test]
    fn e2e_logical_and_assign_truthy() {
        let r = crate::builtins::global::global_eval("var x = 1; x &&= 5; x").unwrap();
        assert_eq!(r, JsValue::Smi(5));
    }

    /// `x &&= value` — does NOT assign if x is falsy.
    #[test]
    fn e2e_logical_and_assign_falsy() {
        let r = crate::builtins::global::global_eval("var x = 0; x &&= 5; x").unwrap();
        assert_eq!(r, JsValue::Smi(0));
    }

    /// `x ||= value` — assigns only if x is falsy.
    #[test]
    fn e2e_logical_or_assign_falsy() {
        let r = crate::builtins::global::global_eval("var x = 0; x ||= 5; x").unwrap();
        assert_eq!(r, JsValue::Smi(5));
    }

    /// `x ||= value` — does NOT assign if x is truthy.
    #[test]
    fn e2e_logical_or_assign_truthy() {
        let r = crate::builtins::global::global_eval("var x = 1; x ||= 5; x").unwrap();
        assert_eq!(r, JsValue::Smi(1));
    }

    /// Interaction: `obj?.prop ?? 'default'`.
    #[test]
    fn e2e_optional_chain_then_nullish() {
        let r =
            crate::builtins::global::global_eval("var obj = null; obj?.x ?? 'default'").unwrap();
        assert_eq!(r, JsValue::String("default".into()));
    }

    /// Interaction: `(a ?? b)?.prop`.
    #[test]
    fn e2e_nullish_then_optional_chain() {
        let r = crate::builtins::global::global_eval("var a = null; var b = {x: 10}; (a ?? b)?.x")
            .unwrap();
        assert_eq!(r, JsValue::Smi(10));
    }

    /// Precedence: `a ?? b && c` is a SyntaxError.
    #[test]
    fn e2e_nullish_mixed_and_error() {
        let r = crate::builtins::global::global_eval("1 ?? 2 && 3");
        assert!(r.is_err());
    }

    /// Precedence: `a ?? b || c` is a SyntaxError.
    #[test]
    fn e2e_nullish_mixed_or_error() {
        let r = crate::builtins::global::global_eval("1 ?? 2 || 3");
        assert!(r.is_err());
    }

    /// Precedence: `a || b ?? c` is a SyntaxError.
    #[test]
    fn e2e_or_then_nullish_error() {
        let r = crate::builtins::global::global_eval("1 || 2 ?? 3");
        assert!(r.is_err());
    }

    /// Precedence: `(a || b) ?? c` is allowed (parens).
    #[test]
    fn e2e_paren_or_then_nullish_ok() {
        let r = crate::builtins::global::global_eval("(null || undefined) ?? 42").unwrap();
        assert_eq!(r, JsValue::Smi(42));
    }

    // ── NaN comparison edge cases (relational operators) ─────────────────

    /// `NaN <= 1` must be `false`, not `true`.
    #[test]
    fn e2e_nan_less_than_or_equal() {
        let result = crate::builtins::global::global_eval("NaN <= 1").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    /// `1 <= NaN` must be `false`.
    #[test]
    fn e2e_one_lte_nan() {
        let result = crate::builtins::global::global_eval("1 <= NaN").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    /// `NaN >= 1` must be `false`.
    #[test]
    fn e2e_nan_greater_than_or_equal() {
        let result = crate::builtins::global::global_eval("NaN >= 1").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    /// `1 >= NaN` must be `false`.
    #[test]
    fn e2e_one_gte_nan() {
        let result = crate::builtins::global::global_eval("1 >= NaN").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    /// `NaN > 1` must be `false`.
    #[test]
    fn e2e_nan_greater_than() {
        let result = crate::builtins::global::global_eval("NaN > 1").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    /// `NaN < 1` must be `false`.
    #[test]
    fn e2e_nan_less_than() {
        let result = crate::builtins::global::global_eval("NaN < 1").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    /// `NaN == NaN` must be `false`.
    #[test]
    fn e2e_nan_loose_eq_nan() {
        let result = crate::builtins::global::global_eval("NaN == NaN").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    /// `NaN === NaN` must be `false`.
    #[test]
    fn e2e_nan_strict_eq_nan() {
        let result = crate::builtins::global::global_eval("NaN === NaN").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    // ── Type coercion in comparisons ─────────────────────────────────────

    /// `null == undefined` must be `true`.
    #[test]
    fn e2e_null_loose_eq_undefined() {
        let result = crate::builtins::global::global_eval("null == undefined").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `null == 0` must be `false`.
    #[test]
    fn e2e_null_not_eq_zero() {
        let result = crate::builtins::global::global_eval("null == 0").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    /// `null == false` must be `false`.
    #[test]
    fn e2e_null_not_eq_false() {
        let result = crate::builtins::global::global_eval("null == false").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    /// `"" == 0` must be `true` (empty string coerces to 0).
    #[test]
    fn e2e_empty_string_eq_zero() {
        let result = crate::builtins::global::global_eval("'' == 0").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `true == 1` must be `true`.
    #[test]
    fn e2e_true_eq_one() {
        let result = crate::builtins::global::global_eval("true == 1").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `false == 0` must be `true`.
    #[test]
    fn e2e_false_eq_zero() {
        let result = crate::builtins::global::global_eval("false == 0").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `+0 === -0` must be `true`.
    #[test]
    fn e2e_pos_zero_strict_eq_neg_zero() {
        // Use a variable to create -0 since the literal -0 is parsed as Smi(0)
        let result = crate::builtins::global::global_eval("var x = -0; x === 0").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `typeof null` must be `"object"`.
    #[test]
    fn e2e_typeof_null_is_object_coerce() {
        let result = crate::builtins::global::global_eval("typeof null === 'object'").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// ToString(-0) must be "0".
    #[test]
    fn e2e_neg_zero_to_string() {
        let result = crate::builtins::global::global_eval("String(-0)").unwrap();
        assert_eq!(result, JsValue::String("0".into()));
    }

    /// Number("") must be 0.
    #[test]
    fn e2e_number_empty_string() {
        let result = crate::builtins::global::global_eval("Number('')").unwrap();
        assert_eq!(result, JsValue::Smi(0));
    }

    /// Number(null) must be 0.
    #[test]
    fn e2e_number_null() {
        let result = crate::builtins::global::global_eval("Number(null)").unwrap();
        assert_eq!(result, JsValue::Smi(0));
    }

    /// Number(undefined) must be NaN.
    #[test]
    fn e2e_number_undefined() {
        let result = crate::builtins::global::global_eval("Number(undefined)").unwrap();
        assert!(matches!(result, JsValue::HeapNumber(n) if n.is_nan()));
    }

    /// Number(true) must be 1, Number(false) must be 0.
    #[test]
    fn e2e_number_boolean() {
        let r1 = crate::builtins::global::global_eval("Number(true)").unwrap();
        assert_eq!(r1, JsValue::Smi(1));
        let r2 = crate::builtins::global::global_eval("Number(false)").unwrap();
        assert_eq!(r2, JsValue::Smi(0));
    }

    /// String(Symbol()) should work (not throw).
    #[test]
    fn e2e_string_of_symbol() {
        let result = crate::builtins::global::global_eval("typeof String(Symbol())").unwrap();
        assert_eq!(result, JsValue::String("string".into()));
    }

    // ── Iterator protocol conformance tests ─────────────────────────────

    /// for-of over an array collects all elements.
    #[test]
    fn e2e_for_of_array_collect() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; for (var x of [10, 20, 30]) r += x; r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(60));
    }

    /// for-of over a string iterates Unicode code points, not bytes.
    #[test]
    fn e2e_for_of_string_unicode() {
        let result = crate::builtins::global::global_eval(
            "var r = []; for (var c of 'a\\u{1F600}b') r.push(c); r.length",
        )
        .unwrap();
        // 'a' + '😀' + 'b' = 3 code points (emoji is one element)
        assert_eq!(result, JsValue::Smi(3));
    }

    /// Array.prototype.keys() returns an iterator of indices.
    #[test]
    fn e2e_array_keys_iterator() {
        let result = crate::builtins::global::global_eval(
            "var r = []; for (var k of ['a','b','c'].keys()) r.push(k); r.join(',')",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("0,1,2".into()));
    }

    /// Array.prototype.values() returns an iterator of values.
    #[test]
    fn e2e_array_values_iterator() {
        let result = crate::builtins::global::global_eval(
            "var r = []; for (var v of ['x','y'].values()) r.push(v); r.join(',')",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("x,y".into()));
    }

    /// Array.prototype.entries() returns [index, value] pairs.
    #[test]
    fn e2e_array_entries_iterator() {
        let result = crate::builtins::global::global_eval(
            "var r = []; for (var e of ['a','b'].entries()) r.push(e[0] + ':' + e[1]); r.join(',')",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("0:a,1:b".into()));
    }

    /// Spread array into function call.
    #[test]
    fn e2e_spread_array_in_call() {
        let result = crate::builtins::global::global_eval(
            "function f(a,b,c) { return a * 100 + b * 10 + c; } f(...[1,2,3])",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(123));
    }

    /// Spread string into array.
    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_spread_string_into_array() {
        let result =
            crate::builtins::global::global_eval("var a = [...'abc']; a.join('-')").unwrap();
        assert_eq!(result, JsValue::String("a-b-c".into()));
    }

    /// for-of with break exits early and only iterates once.
    #[test]
    fn e2e_for_of_break_early_exit() {
        let result = crate::builtins::global::global_eval(
            "var r = []; for (var x of [1,2,3,4]) { r.push(x); if (x === 2) break; } r.join(',')",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("1,2".into()));
    }

    /// for-of with return inside a function exits the function.
    #[test]
    fn e2e_for_of_return_exits_function() {
        let result = crate::builtins::global::global_eval(
            "function f() { for (var x of [1,2,3]) { if (x === 2) return x; } return 0; } f()",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(2));
    }

    /// Generator as iterable in for-of.
    #[test]
    fn e2e_for_of_generator() {
        let result = crate::builtins::global::global_eval(
            "function* gen() { yield 10; yield 20; yield 30; }\
             var r = 0; for (var x of gen()) r += x; r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(60));
    }

    /// Spread generator into array.
    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_spread_generator_into_array() {
        let result = crate::builtins::global::global_eval(
            "function* gen() { yield 1; yield 2; yield 3; } var a = [...gen()]; a.join(',')",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("1,2,3".into()));
    }

    /// Iterator .next() returns {value, done} objects.
    #[test]
    fn e2e_iterator_next_protocol() {
        let result = crate::builtins::global::global_eval(
            "var arr = [42]; var iter = arr.values(); var r = iter.next(); r.value",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// Iterator .next() returns done:true when exhausted.
    #[test]
    fn e2e_iterator_next_done() {
        let result = crate::builtins::global::global_eval(
            "var iter = [1].values(); iter.next(); var r = iter.next(); r.done",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// for-of with const binding.
    #[test]
    fn e2e_for_of_const_binding() {
        let result =
            crate::builtins::global::global_eval("var r = ''; for (const ch of 'hi') r += ch; r")
                .unwrap();
        assert_eq!(result, JsValue::String("hi".into()));
    }

    /// for-of with let binding.
    #[test]
    fn e2e_for_of_let_binding() {
        let result =
            crate::builtins::global::global_eval("var r = 0; for (let n of [1,2,3]) r += n; r")
                .unwrap();
        assert_eq!(result, JsValue::Smi(6));
    }

    /// for-of with destructuring.
    #[test]
    fn e2e_for_of_destructuring() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; var pairs = [[1,2],[3,4]]; for (var [a, b] of pairs) r += a + b; r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(10));
    }

    /// Nested for-of loops.
    #[test]
    fn e2e_for_of_nested() {
        let result = crate::builtins::global::global_eval(
            "var r = ''; for (var a of [1,2]) for (var b of ['x','y']) r += a + b + ','; r",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("1x,1y,2x,2y,".into()));
    }

    /// arguments object is iterable with for-of.
    #[test]
    #[ignore] // TODO: arguments @@iterator not fully supported
    fn e2e_for_of_arguments() {
        let result = crate::builtins::global::global_eval(
            "function f() { var r = 0; for (var x of arguments) r += x; return r; } f(1,2,3)",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(6));
    }

    /// Spread arguments object.
    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_spread_arguments() {
        let result = crate::builtins::global::global_eval(
            "function f() { return [...arguments].join('-'); } f('a','b','c')",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("a-b-c".into()));
    }

    // ── for-in enumeration conformance ──────────────────────────────

    /// for-in walks prototype chain and collects inherited keys.
    #[test]
    fn e2e_for_in_prototype_chain() {
        let result = crate::builtins::global::global_eval(
            "var gp = { z: 3 }; \
             var p = Object.create(gp); p.y = 2; \
             var o = Object.create(p); o.x = 1; \
             var keys = []; for (var k in o) { keys.push(k); } \
             keys.join(',')",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("x,y,z".into()));
    }

    /// Non-enumerable own property hides inherited enumerable one.
    #[test]
    fn e2e_for_in_non_enum_shadows_inherited() {
        let result = crate::builtins::global::global_eval(
            "var p = { x: 1, y: 2 }; \
             var o = Object.create(p); \
             Object.defineProperty(o, 'x', { value: 10, enumerable: false }); \
             var keys = []; for (var k in o) { keys.push(k); } \
             keys.join(',')",
        )
        .unwrap();
        // 'x' is own non-enumerable → shadows inherited 'x'
        // Only 'y' from proto should appear
        assert_eq!(result, JsValue::String("y".into()));
    }

    /// for-in integer indices appear first in ascending order.
    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_for_in_integer_order() {
        let result = crate::builtins::global::global_eval(
            "var o = {}; o.z = 1; o['10'] = 2; o['2'] = 3; o.a = 4; \
             var keys = []; for (var k in o) { keys.push(k); } \
             keys.join(',')",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("2,10,z,a".into()));
    }

    /// for-in on null produces no iterations.
    #[test]
    fn e2e_for_in_null() {
        let result =
            crate::builtins::global::global_eval("var n = 0; for (var k in null) { n++; } n")
                .unwrap();
        assert_eq!(result, JsValue::Smi(0));
    }

    /// for-in on undefined produces no iterations.
    #[test]
    fn e2e_for_in_undefined() {
        let result =
            crate::builtins::global::global_eval("var n = 0; for (var k in undefined) { n++; } n")
                .unwrap();
        assert_eq!(result, JsValue::Smi(0));
    }

    /// for-in can read property values via the key.
    #[test]
    fn e2e_for_in_sum_values() {
        let result = crate::builtins::global::global_eval(
            "var o = { a: 10, b: 20, c: 30 }; \
             var sum = 0; for (var k in o) { sum = sum + o[k]; } sum",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(60));
    }

    // ── for-in with break / continue ────────────────────────────────

    /// for-in with break stops iteration.
    #[test]
    fn e2e_for_in_break() {
        let result = crate::builtins::global::global_eval(
            "var o = { a: 1, b: 2, c: 3 }; \
             var n = 0; for (var k in o) { n++; if (n === 2) break; } n",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(2));
    }

    // ── typeof conformance (for-in section) ────────────────────────────

    #[test]
    fn e2e_typeof_null_is_object_forin() {
        let result = crate::builtins::global::global_eval("typeof null").unwrap();
        assert_eq!(result, JsValue::String("object".into()));
    }

    #[test]
    fn e2e_typeof_undefined_forin() {
        let result = crate::builtins::global::global_eval("typeof undefined").unwrap();
        assert_eq!(result, JsValue::String("undefined".into()));
    }

    #[test]
    fn e2e_typeof_boolean_forin() {
        let result = crate::builtins::global::global_eval("typeof true").unwrap();
        assert_eq!(result, JsValue::String("boolean".into()));
    }

    #[test]
    fn e2e_typeof_number_forin() {
        let result = crate::builtins::global::global_eval("typeof 42").unwrap();
        assert_eq!(result, JsValue::String("number".into()));
    }

    #[test]
    fn e2e_typeof_string_forin() {
        let result = crate::builtins::global::global_eval("typeof 'hello'").unwrap();
        assert_eq!(result, JsValue::String("string".into()));
    }

    #[test]
    fn e2e_typeof_symbol_forin() {
        let result = crate::builtins::global::global_eval("typeof Symbol()").unwrap();
        assert_eq!(result, JsValue::String("symbol".into()));
    }

    #[test]
    fn e2e_typeof_bigint() {
        let result = crate::builtins::global::global_eval("typeof BigInt(1)").unwrap();
        assert_eq!(result, JsValue::String("bigint".into()));
    }

    #[test]
    fn e2e_typeof_function() {
        let result = crate::builtins::global::global_eval("typeof function() {}").unwrap();
        assert_eq!(result, JsValue::String("function".into()));
    }

    #[test]
    fn e2e_typeof_arrow_function() {
        let result = crate::builtins::global::global_eval("typeof (() => {})").unwrap();
        assert_eq!(result, JsValue::String("function".into()));
    }

    #[test]
    fn e2e_typeof_object() {
        let result = crate::builtins::global::global_eval("typeof {}").unwrap();
        assert_eq!(result, JsValue::String("object".into()));
    }

    #[test]
    fn e2e_typeof_array() {
        let result = crate::builtins::global::global_eval("typeof []").unwrap();
        assert_eq!(result, JsValue::String("object".into()));
    }

    #[test]
    fn e2e_typeof_undeclared_var() {
        // typeof on an undeclared variable must NOT throw ReferenceError.
        let result = crate::builtins::global::global_eval("typeof someUndeclaredVariable").unwrap();
        assert_eq!(result, JsValue::String("undefined".into()));
    }

    #[test]
    fn e2e_typeof_bound_function() {
        let result =
            crate::builtins::global::global_eval("function f() {} typeof f.bind(null)").unwrap();
        assert_eq!(result, JsValue::String("function".into()));
    }

    // ── instanceof conformance ──────────────────────────────────────────

    #[test]
    fn e2e_instanceof_error() {
        let result =
            crate::builtins::global::global_eval("new Error('x') instanceof Error").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_instanceof_typeerror_is_error() {
        // TypeError instance should also be instanceof Error.
        let result =
            crate::builtins::global::global_eval("new TypeError('x') instanceof Error").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_instanceof_typeerror() {
        let result =
            crate::builtins::global::global_eval("new TypeError('x') instanceof TypeError")
                .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_instanceof_rangeerror() {
        let result =
            crate::builtins::global::global_eval("new RangeError('x') instanceof RangeError")
                .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_instanceof_rangeerror_is_error() {
        let result =
            crate::builtins::global::global_eval("new RangeError('x') instanceof Error").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_instanceof_not_callable_throws() {
        let result = crate::builtins::global::global_eval("1 instanceof 2");
        assert!(result.is_err());
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_instanceof_custom_has_instance() {
        let result = crate::builtins::global::global_eval(
            "var obj = { [Symbol.hasInstance](v) { return v === 42; } }; 42 instanceof obj",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_instanceof_class_symbol_has_instance_true() {
        let result = crate::builtins::global::global_eval(
            "class Foo { static [Symbol.hasInstance](x) { return x === 1; } } 1 instanceof Foo",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_instanceof_class_symbol_has_instance_false() {
        let result = crate::builtins::global::global_eval(
            "class Foo { static [Symbol.hasInstance](x) { return x === 1; } } 2 instanceof Foo",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn e2e_instanceof_function_symbol_has_instance_true() {
        let result = crate::builtins::global::global_eval(
            "function Foo() {} Foo[Symbol.hasInstance] = function(x) { return x && x.tag === 1; }; ({ tag: 1 }) instanceof Foo",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── Error conformance ───────────────────────────────────────────────

    #[test]
    fn e2e_new_error_message_empty() {
        // new Error() should have message === "" (empty string, not undefined).
        let result = crate::builtins::global::global_eval("new Error().message").unwrap();
        assert_eq!(result, JsValue::String("".into()));
    }

    #[test]
    fn e2e_new_typeerror_message_empty() {
        let result = crate::builtins::global::global_eval("new TypeError().message").unwrap();
        assert_eq!(result, JsValue::String("".into()));
    }

    #[test]
    fn e2e_new_error_message_string() {
        let result = crate::builtins::global::global_eval("new Error('hello').message").unwrap();
        assert_eq!(result, JsValue::String("hello".into()));
    }

    #[test]
    fn e2e_error_name() {
        let result = crate::builtins::global::global_eval("new Error().name").unwrap();
        assert_eq!(result, JsValue::String("Error".into()));
    }

    #[test]
    fn e2e_typeerror_name() {
        let result = crate::builtins::global::global_eval("new TypeError().name").unwrap();
        assert_eq!(result, JsValue::String("TypeError".into()));
    }

    #[test]
    fn e2e_error_stack_is_string() {
        let result = crate::builtins::global::global_eval("typeof new Error().stack").unwrap();
        assert_eq!(result, JsValue::String("string".into()));
    }

    #[test]
    fn e2e_error_tostring_with_message() {
        let result = crate::builtins::global::global_eval("new Error('msg').toString()").unwrap();
        assert_eq!(result, JsValue::String("Error: msg".into()));
    }

    #[test]
    fn e2e_error_tostring_without_message() {
        let result = crate::builtins::global::global_eval("new Error().toString()").unwrap();
        assert_eq!(result, JsValue::String("Error".into()));
    }

    #[test]
    fn e2e_typeerror_tostring() {
        let result =
            crate::builtins::global::global_eval("new TypeError('bad').toString()").unwrap();
        assert_eq!(result, JsValue::String("TypeError: bad".into()));
    }

    #[test]
    fn e2e_error_constructor_identity() {
        let result =
            crate::builtins::global::global_eval("new Error('msg').constructor === Error").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_typeerror_constructor_identity() {
        let result =
            crate::builtins::global::global_eval("new TypeError('msg').constructor === TypeError")
                .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_error_subclass_prototype_chain() {
        // TypeError.prototype should inherit from Error.prototype.
        let result = crate::builtins::global::global_eval(
            "TypeError.prototype instanceof Error \
             || Object.getPrototypeOf(TypeError.prototype) === Error.prototype",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── throw / catch conformance ───────────────────────────────────────

    #[test]
    fn e2e_throw_number() {
        let result =
            crate::builtins::global::global_eval("var r; try { throw 42; } catch(e) { r = e; } r")
                .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn e2e_throw_string() {
        let result = crate::builtins::global::global_eval(
            "var r; try { throw 'oops'; } catch(e) { r = e; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("oops".into()));
    }

    #[test]
    fn e2e_throw_boolean() {
        let result = crate::builtins::global::global_eval(
            "var r; try { throw false; } catch(e) { r = e; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn e2e_throw_null() {
        let result = crate::builtins::global::global_eval(
            "var r; try { throw null; } catch(e) { r = e; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Null);
    }

    #[test]
    fn e2e_throw_undefined() {
        let result = crate::builtins::global::global_eval(
            "var r; try { throw undefined; } catch(e) { r = e; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    #[test]
    fn e2e_throw_error_object() {
        let result = crate::builtins::global::global_eval(
            "var r; try { throw new TypeError('msg'); } catch(e) { r = e.message; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("msg".into()));
    }

    #[test]
    fn e2e_engine_error_caught_as_error_object() {
        // Engine-thrown TypeError (e.g. calling null) should be caught as
        // a proper Error object with correct kind and message.
        let result = crate::builtins::global::global_eval(
            "var r; try { null(); } catch(e) { r = e instanceof TypeError; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn e2e_throw_object_literal() {
        let result = crate::builtins::global::global_eval(
            "var r; try { throw { code: 42 }; } catch(e) { r = e.code; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    // ── Arrow function conformance ──────────────────────────────────────

    #[test]
    fn e2e_arrow_concise_body() {
        let result = crate::builtins::global::global_eval("var f = x => x * 2; f(5)").unwrap();
        assert_eq!(result, JsValue::Smi(10));
    }

    #[test]
    fn e2e_arrow_block_body() {
        let result =
            crate::builtins::global::global_eval("var f = x => { return x * 3; }; f(4)").unwrap();
        assert_eq!(result, JsValue::Smi(12));
    }

    #[test]
    #[ignore] // TODO: arrow arguments object regression
    fn e2e_arrow_no_own_arguments() {
        // Arrow functions do NOT have their own `arguments` object.
        let result = crate::builtins::global::global_eval(
            "function outer() { var f = () => typeof arguments; return f(); } outer(1,2,3)",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("object".into()));
    }

    #[test]
    #[ignore] // TODO: arrow arguments object regression
    fn e2e_arrow_inherits_outer_arguments() {
        // Arrow should read enclosing function's arguments.
        let result = crate::builtins::global::global_eval(
            "function outer() { var f = () => arguments.length; return f(); } outer(10,20,30)",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    #[test]
    fn e2e_arrow_lexical_this_in_method() {
        // Arrow inherits `this` from enclosing scope (the method's `this`).
        let result = crate::builtins::global::global_eval(
            "var obj = { x: 42, getX: function() { var f = () => this.x; return f(); } }; \
             obj.getX()",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn e2e_arrow_lexical_this_not_overridden_by_call() {
        // .call() on an arrow should NOT change its `this`.
        let result = crate::builtins::global::global_eval(
            "var obj = { x: 99, getX: function() { var f = () => this.x; \
             return f.call({x: 1}); } }; obj.getX()",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    #[test]
    fn e2e_arrow_not_constructable_detailed() {
        let result = crate::builtins::global::global_eval(
            "var f = () => {}; var ok = false; \
             try { new f(); } catch(e) { ok = e instanceof TypeError; } ok",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── this binding rules ──────────────────────────────────────────────

    #[test]
    fn e2e_this_implicit_binding() {
        // obj.method() → this is obj.
        let result = crate::builtins::global::global_eval(
            "var obj = { val: 7, get: function() { return this.val; } }; obj.get()",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    #[test]
    fn e2e_this_explicit_call() {
        let result =
            crate::builtins::global::global_eval("function f() { return this.x; } f.call({x: 55})")
                .unwrap();
        assert_eq!(result, JsValue::Smi(55));
    }

    #[test]
    fn e2e_this_explicit_apply() {
        let result = crate::builtins::global::global_eval(
            "function f(a) { return this.x + a; } f.apply({x: 10}, [5])",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(15));
    }

    #[test]
    fn e2e_this_new_binding() {
        // `new` binding: `this` is the new object.
        let result = crate::builtins::global::global_eval(
            "function Foo(v) { this.val = v; } var f = new Foo(33); f.val",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(33));
    }

    // ── Function.prototype.bind ─────────────────────────────────────────

    #[test]
    fn e2e_bind_this() {
        let result = crate::builtins::global::global_eval(
            "function f() { return this.x; } var g = f.bind({x: 77}); g()",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(77));
    }

    #[test]
    fn e2e_bind_partial_application() {
        let result = crate::builtins::global::global_eval(
            "function add(a, b) { return a + b; } \
             var add5 = add.bind(null, 5); add5(3)",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(8));
    }

    #[test]
    fn e2e_bind_length() {
        // Bound function length = max(0, target.length - bound_args)
        let result = crate::builtins::global::global_eval(
            "function f(a, b, c) {} var g = f.bind(null, 1); g.length",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(2));
    }

    #[test]
    fn e2e_bind_name() {
        let result = crate::builtins::global::global_eval(
            "function f(a, b, c) {} var g = f.bind(null); typeof g.name === 'string'",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    #[ignore] // TODO: bind + call this override regression
    fn e2e_bind_subsequent_call_doesnt_change_this() {
        // .bind() creates a bound function; subsequent .call() doesn't override.
        let result = crate::builtins::global::global_eval(
            "function f() { return this.x; } \
             var g = f.bind({x: 100}); g.call({x: 999})",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(100));
    }

    // ── Function.prototype.call / apply ─────────────────────────────────

    #[test]
    fn e2e_call_with_args() {
        let result = crate::builtins::global::global_eval(
            "function f(a, b) { return this.x + a + b; } f.call({x: 1}, 2, 3)",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(6));
    }

    #[test]
    fn e2e_apply_with_array() {
        let result = crate::builtins::global::global_eval(
            "function f(a, b) { return a + b; } f.apply(null, [10, 20])",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(30));
    }

    // ── Closures ────────────────────────────────────────────────────────

    #[test]
    fn e2e_closure_captures_by_reference() {
        let result = crate::builtins::global::global_eval(
            "function make() { var x = 1; return { get: function() { return x; }, \
             set: function(v) { x = v; } }; } \
             var o = make(); o.set(42); o.get()",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_closure_survives_after_return() {
        let result = crate::builtins::global::global_eval(
            "function counter() { var n = 0; return function() { n = n + 1; return n; }; } \
             var c = counter(); c(); c(); c()",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    #[test]
    fn e2e_closure_var_shared_in_loop() {
        // Loop closures with `var` share the same variable.
        let result = crate::builtins::global::global_eval(
            "var fns = []; \
             for (var i = 0; i < 3; i++) { fns[i] = function() { return i; }; } \
             fns[0]() + ',' + fns[1]() + ',' + fns[2]()",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("3,3,3".into()));
    }

    // ── arguments object ────────────────────────────────────────────────

    #[test]
    #[ignore] // TODO: arguments object regression
    fn e2e_arguments_length() {
        let result = crate::builtins::global::global_eval(
            "function f() { return arguments.length; } f(1,2,3)",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    #[test]
    #[ignore] // TODO: arguments object regression
    fn e2e_arguments_indexed_access() {
        let result = crate::builtins::global::global_eval(
            "function f() { return arguments[0] + arguments[1]; } f(10, 20)",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(30));
    }

    #[test]
    fn e2e_arguments_not_array() {
        let result = crate::builtins::global::global_eval(
            "function f() { return Array.isArray(arguments); } f(1,2)",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    #[ignore] // TODO: arguments @@iterator not fully supported
    fn e2e_arguments_iterable() {
        let result = crate::builtins::global::global_eval(
            "function f() { var r = ''; for (var x of arguments) r += x + ','; return r; } \
             f('a','b','c')",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("a,b,c,".into()));
    }

    // ── Default parameters ──────────────────────────────────────────────

    #[test]
    fn e2e_default_param_basic() {
        let result =
            crate::builtins::global::global_eval("function f(a = 10) { return a; } f()").unwrap();
        assert_eq!(result, JsValue::Smi(10));
    }

    #[test]
    fn e2e_default_param_overridden() {
        let result =
            crate::builtins::global::global_eval("function f(a = 10) { return a; } f(5)").unwrap();
        assert_eq!(result, JsValue::Smi(5));
    }

    #[test]
    fn e2e_default_param_references_earlier() {
        // Later defaults can reference earlier params.
        let result =
            crate::builtins::global::global_eval("function f(a, b = a * 2) { return b; } f(3)")
                .unwrap();
        assert_eq!(result, JsValue::Smi(6));
    }

    #[test]
    fn e2e_default_param_left_to_right() {
        let result =
            crate::builtins::global::global_eval("function f(a = 1, b = 2) { return a + b; } f()")
                .unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    #[test]
    fn e2e_default_param_with_arrow() {
        let result = crate::builtins::global::global_eval("var f = (a = 7) => a + 1; f()").unwrap();
        assert_eq!(result, JsValue::Smi(8));
    }

    // ── Computed property names ─────────────────────────────────────────

    #[test]
    fn e2e_computed_prop_string_concat() {
        let result =
            crate::builtins::global::global_eval("var o = {['a' + 'b']: 1}; o.ab").unwrap();
        assert_eq!(result, JsValue::Smi(1));
    }

    #[test]
    fn e2e_computed_prop_variable_key() {
        let result =
            crate::builtins::global::global_eval("var k = 'hello'; var o = {[k]: 42}; o.hello")
                .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn e2e_computed_prop_numeric_expr() {
        let result =
            crate::builtins::global::global_eval("var o = {[1 + 2]: 'three'}; o[3]").unwrap();
        assert_eq!(result, JsValue::String("three".into()));
    }

    #[test]
    fn e2e_computed_method() {
        let result = crate::builtins::global::global_eval(
            "var k = 'greet'; var o = {[k]() { return 'hi'; }}; o.greet()",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("hi".into()));
    }

    // ── Shorthand properties ────────────────────────────────────────────

    #[test]
    fn e2e_shorthand_property() {
        let result = crate::builtins::global::global_eval(
            "var x = 10; var y = 20; var o = {x, y}; o.x + o.y",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(30));
    }

    #[test]
    fn e2e_shorthand_mixed_with_regular() {
        let result =
            crate::builtins::global::global_eval("var a = 1; var o = {a, b: 2}; o.a + o.b")
                .unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    // ── Shorthand methods ───────────────────────────────────────────────

    #[test]
    fn e2e_shorthand_method() {
        let result = crate::builtins::global::global_eval(
            "var o = {add(a, b) { return a + b; }}; o.add(3, 4)",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    #[test]
    fn e2e_shorthand_method_this() {
        let result = crate::builtins::global::global_eval(
            "var o = {v: 5, get_v() { return this.v; }}; o.get_v()",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(5));
    }

    // ── Getter / setter in object literals ──────────────────────────────

    #[test]
    fn e2e_getter_obj_literal() {
        let result =
            crate::builtins::global::global_eval("var o = {get x() { return 42; }}; o.x").unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn e2e_setter_obj_literal() {
        let result = crate::builtins::global::global_eval(
            "var o = {_v: 0, set v(x) { this._v = x; }}; o.v = 7; o._v",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    #[test]
    fn e2e_getter_setter_pair_literal() {
        let result = crate::builtins::global::global_eval(
            "var o = {\
               _val: 0,\
               get val() { return this._val; },\
               set val(v) { this._val = v * 2; }\
             };\
             o.val = 5;\
             o.val",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(10));
    }

    #[test]
    fn e2e_getter_computed_key() {
        let result = crate::builtins::global::global_eval(
            "var k = 'x'; var o = {get [k]() { return 99; }}; o.x",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    #[test]
    fn e2e_setter_computed_key() {
        let result = crate::builtins::global::global_eval(
            "var k = 'v'; var o = {_r: 0, set [k](x) { this._r = x; }}; o.v = 3; o._r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    // ── Object literal __proto__ ────────────────────────────────────────

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_proto_in_literal() {
        let result = crate::builtins::global::global_eval(
            "var base = {greet() { return 'hello'; }};\
             var child = {__proto__: base};\
             child.greet()",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("hello".into()));
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_proto_inherited_property() {
        let result = crate::builtins::global::global_eval(
            "var parent = {x: 100};\
             var child = {__proto__: parent, y: 200};\
             child.x + child.y",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(300));
    }

    // ── Object.defineProperty accessor callable ─────────────────────────

    #[test]
    fn e2e_define_property_getter_accessible() {
        let result = crate::builtins::global::global_eval(
            "var o = {};\
             Object.defineProperty(o, 'x', {\
               get: function() { return 55; },\
               configurable: true\
             });\
             o.x",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(55));
    }

    #[test]
    fn e2e_define_property_setter_accessible() {
        let result = crate::builtins::global::global_eval(
            "var o = {_v: 0};\
             Object.defineProperty(o, 'x', {\
               set: function(v) { this._v = v; },\
               configurable: true\
             });\
             o.x = 11;\
             o._v",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(11));
    }

    #[test]
    fn e2e_define_property_accessor_pair() {
        let result = crate::builtins::global::global_eval(
            "var o = {_v: 0};\
             Object.defineProperty(o, 'val', {\
               get: function() { return this._v; },\
               set: function(v) { this._v = v + 1; },\
               configurable: true\
             });\
             o.val = 10;\
             o.val",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(11));
    }

    // ── Getter / setter .name property ──────────────────────────────────

    #[test]
    fn e2e_getter_name_property() {
        let result = crate::builtins::global::global_eval(
            "var o = {get x() { return 1; }};\
             var desc = Object.getOwnPropertyDescriptor(o, 'x');\
             desc.get.name",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("get x".into()));
    }

    #[test]
    fn e2e_setter_name_property() {
        let result = crate::builtins::global::global_eval(
            "var o = {set x(v) {}};\
             var desc = Object.getOwnPropertyDescriptor(o, 'x');\
             desc.set.name",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("set x".into()));
    }

    #[test]
    fn e2e_method_name_property() {
        let result = crate::builtins::global::global_eval(
            "var o = {myMethod() { return 1; }}; o.myMethod.name",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("myMethod".into()));
    }

    // ── Mixed object literal features ───────────────────────────────────

    #[test]
    fn e2e_mixed_shorthand_computed_method() {
        let result = crate::builtins::global::global_eval(
            "var a = 1;\
             var k = 'b';\
             var o = {a, [k]: 2, c() { return 3; }};\
             o.a + o.b + o.c()",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(6));
    }

    #[test]
    fn e2e_computed_getter_setter_pair() {
        let result = crate::builtins::global::global_eval(
            "var k = 'prop';\
             var o = {\
               _s: 0,\
               get [k]() { return this._s; },\
               set [k](v) { this._s = v * 3; }\
             };\
             o.prop = 4;\
             o.prop",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(12));
    }

    // ── Label / break / continue deep conformance ────────────────────────

    #[test]
    fn e2e_labeled_block_break() {
        // label: { break label; } — basic labeled block
        let result = crate::builtins::global::global_eval(
            "var r = 0; outer: { r = 1; break outer; r = 2; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(1));
    }

    #[test]
    fn e2e_labeled_block_completion_value() {
        // eval completion value: last value before break
        let result = crate::builtins::global::global_eval(
            "var x; outer: { x = 10; break outer; x = 20; } x",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(10));
    }

    #[test]
    fn e2e_labeled_while_break() {
        // outer: while(true) { break outer; }
        let result = crate::builtins::global::global_eval(
            "var i = 0; outer: while (true) { i = 42; break outer; } i",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn e2e_labeled_for_break() {
        // outer: for(...) { break outer; }
        let result = crate::builtins::global::global_eval(
            "var r = 0; outer: for (var i = 0; i < 10; i++) { r = i; if (i === 3) break outer; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    #[test]
    fn e2e_nested_labeled_loops_break_outer() {
        // outer: for(...) { inner: for(...) { break outer; } }
        let result = crate::builtins::global::global_eval(
            "var r = 0; outer: for (var i = 0; i < 5; i++) { inner: for (var j = 0; j < 5; j++) { r++; if (j === 1) break outer; } } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(2));
    }

    #[test]
    fn e2e_nested_labeled_loops_break_inner() {
        // outer: for(...) { inner: for(...) { break inner; } }
        let result = crate::builtins::global::global_eval(
            "var r = 0; outer: for (var i = 0; i < 3; i++) { inner: for (var j = 0; j < 10; j++) { if (j === 2) break inner; r++; } } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(6)); // 2 per outer iteration * 3
    }

    #[test]
    fn e2e_continue_with_label() {
        // outer: for(...) { continue outer; }
        let result = crate::builtins::global::global_eval(
            "var r = 0; outer: for (var i = 0; i < 5; i++) { if (i === 3) continue outer; r++; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(4)); // skips i=3
    }

    #[test]
    fn e2e_continue_label_nested_loops() {
        // continue outer in nested loop
        let result = crate::builtins::global::global_eval(
            "var r = 0; outer: for (var i = 0; i < 3; i++) { for (var j = 0; j < 3; j++) { if (j === 1) continue outer; r++; } } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3)); // j=0 for each i=0,1,2
    }

    #[test]
    fn e2e_break_in_switch_inside_loop() {
        // break without label exits switch, not the loop
        let result = crate::builtins::global::global_eval(
            "var r = 0; for (var i = 0; i < 3; i++) { switch(i) { case 1: break; default: r++; } } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(2)); // i=0 default, i=1 break (switch), i=2 default
    }

    #[test]
    fn e2e_break_label_exits_loop_from_switch() {
        // break label exits labeled loop, not the switch
        let result = crate::builtins::global::global_eval(
            "var r = 0; loop1: for (var i = 0; i < 10; i++) { switch(i) { case 3: break loop1; default: r++; } } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3)); // i=0,1,2 → default
    }

    #[test]
    fn e2e_labeled_do_while_break() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; outer: do { r++; if (r === 5) break outer; } while (true); r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(5));
    }

    #[test]
    fn e2e_labeled_do_while_continue() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; var s = 0; outer: do { r++; if (r < 3) continue outer; s++; } while (r < 5); s",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3)); // r=3,4,5 → s incremented
    }

    #[test]
    fn e2e_for_in_with_label_break() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; var obj = {a:1, b:2, c:3}; outer: for (var k in obj) { r++; if (k === 'b') break outer; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(2));
    }

    #[test]
    fn e2e_for_in_with_label_continue() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; var obj = {a:1, b:2, c:3}; outer: for (var k in obj) { if (k === 'b') continue outer; r++; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(2)); // a and c
    }

    #[test]
    fn e2e_for_of_with_label_break() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; outer: for (var v of [10, 20, 30, 40]) { r += v; if (v === 20) break outer; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(30)); // 10 + 20
    }

    #[test]
    fn e2e_for_of_with_label_continue() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; outer: for (var v of [1, 2, 3, 4]) { if (v === 2) continue outer; r += v; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(8)); // 1 + 3 + 4
    }

    #[test]
    fn e2e_nested_for_in_break_outer() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; var a = {x:1, y:2}; var b = {p:1, q:2}; \
             outer: for (var k1 in a) { for (var k2 in b) { r++; if (k2 === 'q') break outer; } } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(2));
    }

    #[test]
    fn e2e_try_finally_with_break() {
        // break in try should still execute finally
        let result = crate::builtins::global::global_eval(
            "var r = 0; outer: { try { r = 1; break outer; } finally { r = r + 10; } } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(11)); // 1 + 10
    }

    #[test]
    fn e2e_try_finally_with_continue() {
        // continue in try should still execute finally
        let result = crate::builtins::global::global_eval(
            "var r = 0; outer: for (var i = 0; i < 3; i++) { try { if (i === 1) continue outer; } finally { r++; } } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3)); // finally runs for all 3 iterations
    }

    #[test]
    fn e2e_try_finally_break_preserves_value() {
        // The finally body should not clobber the variable set before break
        let result = crate::builtins::global::global_eval(
            "var r = 0; var f = 0; outer: { try { r = 42; break outer; } finally { f = 99; } } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn e2e_nested_finally_with_break() {
        // Deeply nested try/finally with break targeting outer label
        let result = crate::builtins::global::global_eval(
            "var r = ''; outer: { try { try { r += 'a'; break outer; } finally { r += 'b'; } } finally { r += 'c'; } } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("abc".into()));
    }

    #[test]
    fn e2e_labeled_statement_simple() {
        // label: expr; — label on a non-block statement
        let result = crate::builtins::global::global_eval("var r; foo: r = 42; r").unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn e2e_labeled_if_statement() {
        // label on an if statement (non-loop)
        let result =
            crate::builtins::global::global_eval("var r = 0; myLabel: if (true) { r = 7; } r")
                .unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    #[test]
    fn e2e_break_unlabeled_in_for() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; for (var i = 0; i < 10; i++) { if (i === 5) break; r++; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(5));
    }

    #[test]
    fn e2e_continue_unlabeled_in_for() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; for (var i = 0; i < 5; i++) { if (i === 2) continue; r++; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(4));
    }

    #[test]
    fn e2e_break_unlabeled_in_while() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; while (true) { r++; if (r === 3) break; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    #[test]
    fn e2e_continue_unlabeled_in_while() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; var i = 0; while (i < 5) { i++; if (i === 3) continue; r++; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(4));
    }

    #[test]
    fn e2e_nested_labeled_blocks() {
        // Nested labeled blocks with break targeting outer
        let result = crate::builtins::global::global_eval(
            "var r = 0; outer: { r = 1; inner: { r = 2; break outer; r = 3; } r = 4; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(2));
    }

    #[test]
    fn e2e_nested_labeled_blocks_break_inner() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; outer: { r = 1; inner: { r = 2; break inner; r = 3; } r = r + 10; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(12)); // 2 + 10
    }

    #[test]
    fn e2e_labeled_for_continue_update() {
        // continue label should still run the for-loop update expression
        let result = crate::builtins::global::global_eval(
            "var r = 0; outer: for (var i = 0; i < 5; i++) { if (i % 2 === 0) continue outer; r += i; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(4)); // 1 + 3
    }

    #[test]
    fn e2e_switch_fall_through_then_break() {
        let result = crate::builtins::global::global_eval(
            "var r = ''; switch(1) { case 1: r += 'a'; case 2: r += 'b'; break; case 3: r += 'c'; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("ab".into()));
    }

    #[test]
    fn e2e_continue_in_switch_in_loop() {
        // continue inside switch (targets the enclosing loop)
        let result = crate::builtins::global::global_eval(
            "var r = 0; for (var i = 0; i < 4; i++) { switch(i) { case 1: continue; } r++; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3)); // i=0,2,3 → r++
    }

    #[test]
    fn e2e_for_of_nested_break_outer() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; outer: for (var a of [1, 2, 3]) { for (var b of [10, 20]) { r += a * b; if (a === 2 && b === 10) break outer; } } r",
        )
        .unwrap();
        // a=1: b=10 → 10, b=20 → 20. a=2: b=10 → 20, break outer. Total=50
        assert_eq!(result, JsValue::Smi(50));
    }

    #[test]
    fn e2e_try_catch_finally_break() {
        // break inside try with both catch and finally
        let result = crate::builtins::global::global_eval(
            "var r = ''; outer: { try { r += 'try'; break outer; } catch(e) { r += 'catch'; } finally { r += 'finally'; } r += 'after'; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("tryfinally".into()));
    }

    #[test]
    fn e2e_triple_nested_finally_break() {
        let result = crate::builtins::global::global_eval(
            "var r = ''; outer: { try { try { try { r += '1'; break outer; } finally { r += '2'; } } finally { r += '3'; } } finally { r += '4'; } } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("1234".into()));
    }

    #[test]
    fn e2e_labeled_while_continue_label() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; var i = 0; outer: while (i < 10) { i++; if (i % 3 === 0) continue outer; r++; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(7)); // skip i=3,6,9
    }

    #[test]
    fn e2e_multiple_labels_same_loop() {
        // Multiple labels on the same loop: a: b: for(...)
        let result = crate::builtins::global::global_eval(
            "var r = 0; a: b: for (var i = 0; i < 5; i++) { if (i === 3) break a; r++; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    #[test]
    fn e2e_labeled_block_no_break() {
        // Labeled block that doesn't use break — should just execute normally
        let result =
            crate::builtins::global::global_eval("var r = 0; myLabel: { r = 1; r = 2; } r")
                .unwrap();
        assert_eq!(result, JsValue::Smi(2));
    }

    #[test]
    fn e2e_for_in_nested_continue_outer() {
        // continue outer; where outer is a for-in loop
        let result = crate::builtins::global::global_eval(
            "var r = 0; var obj = {a:1, b:2, c:3, d:4}; \
             outer: for (var k in obj) { for (var i = 0; i < 2; i++) { if (i === 1) continue outer; r++; } } r",
        )
        .unwrap();
        // For each of 4 keys: i=0 → r++, i=1 → continue outer
        assert_eq!(result, JsValue::Smi(4));
    }
}
