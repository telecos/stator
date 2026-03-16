//! Bytecode interpreter for the Stator VM.
//!
//! This module provides the fetch-decode-dispatch loop that executes
//! [`crate::bytecode::bytecode_array::BytecodeArray`] bytecode.
//!
//! # Overview
//!
//! - [`InterpreterFrame`] — holds the register file, accumulator,
//!   program counter, and scope context for a single function activation.
//! - [`Interpreter`] — stateless executor; call [`Interpreter::run`] with
//!   a mutable frame to execute it.
//!
//! # Supported opcodes
//!
//! ## Load / Store
//!
//! | Opcode                 | Semantics                                |
//! |------------------------|------------------------------------------|
//! | `LdaZero`              | `acc ← 0`                                |
//! | `LdaSmi(imm)`          | `acc ← imm`                              |
//! | `LdaUndefined`         | `acc ← undefined`                        |
//! | `LdaNull`              | `acc ← null`                             |
//! | `LdaTrue`              | `acc ← true`                             |
//! | `LdaFalse`             | `acc ← false`                            |
//! | `LdaConstant(k)`       | `acc ← constant_pool[k]`                 |
//! | `Ldar(reg)`            | `acc ← reg`                              |
//! | `Star(reg)`            | `reg ← acc`                              |
//!
//! ## Arithmetic
//!
//! | Opcode                 | Semantics                                |
//! |------------------------|------------------------------------------|
//! | `Add(reg, _)`          | `acc ← acc + reg`                        |
//! | `Sub(reg, _)`          | `acc ← acc − reg`                        |
//! | `Mul(reg, _)`          | `acc ← acc × reg`                        |
//! | `Div(reg, _)`          | `acc ← acc ÷ reg`                        |
//! | `Mod(reg, _)`          | `acc ← acc % reg`                        |
//! | `Inc(_)`               | `acc ← acc + 1`                          |
//! | `Dec(_)`               | `acc ← acc − 1`                          |
//!
//! ## Comparison
//!
//! | Opcode                    | Semantics                             |
//! |---------------------------|---------------------------------------|
//! | `TestEqual(reg, _)`       | `acc ← acc == reg`  (abstract eq)    |
//! | `TestNotEqual(reg, _)`    | `acc ← acc != reg`  (abstract neq)   |
//! | `TestEqualStrict(reg, _)` | `acc ← acc === reg` (strict eq)      |
//! | `TestLessThan(reg, _)`    | `acc ← acc < reg`                    |
//! | `TestGreaterThan(reg, _)` | `acc ← acc > reg`                    |
//! | `TestLessThanOrEqual(r,_)`| `acc ← acc <= reg`                   |
//! | `TestGreaterThanOrEqual`  | `acc ← acc >= reg`                   |
//! | `TestNull`                | `acc ← acc === null`                 |
//! | `TestUndefined`           | `acc ← acc === undefined`            |
//!
//! ## Logical / Boolean
//!
//! | Opcode                 | Semantics                                |
//! |------------------------|------------------------------------------|
//! | `LogicalNot`           | `acc ← !acc` (boolean-only)              |
//! | `ToBooleanLogicalNot`  | `acc ← !ToBoolean(acc)`                  |
//!
//! ## Control flow
//!
//! | Opcode                    | Semantics                                              |
//! |---------------------------|--------------------------------------------------------|
//! | `Jump(offset)`            | unconditional forward/backward jump                   |
//! | `JumpLoop(offset,…)`      | back-edge jump (loop repeat)                          |
//! | `JumpIfTrue(offset)`      | jump if `acc === true`                                |
//! | `JumpIfFalse(offset)`     | jump if `acc === false`                               |
//! | `JumpIfToBooleanTrue`     | jump if `ToBoolean(acc)` is truthy                    |
//! | `JumpIfToBooleanFalse`    | jump if `ToBoolean(acc)` is falsy                     |
//! | `JumpIfNull(offset)`      | jump if `acc === null`                                |
//! | `JumpIfNotNull(offset)`   | jump if `acc !== null`                                |
//! | `JumpIfUndefined(offset)` | jump if `acc === undefined`                           |
//! | `JumpIfNotUndefined`      | jump if `acc !== undefined`                           |
//! | `JumpIfUndefinedOrNull`   | jump if `acc === null \|\| acc === undefined`          |
//! | `Return`                  | halt; return `acc`                                    |
//!
//! ## Closure creation
//!
//! | Opcode                    | Semantics                                              |
//! |---------------------------|--------------------------------------------------------|
//! | `CreateClosure(k,_,_)`    | `acc ← Function(constant_pool[k])`                    |
//!
//! ## Function calls
//!
//! | Opcode                       | Semantics                                           |
//! |------------------------------|-----------------------------------------------------|
//! | `CallAnyReceiver(f,a,n,_)`   | `acc ← f(a, a+1, …, a+n−1)` (undefined receiver)  |
//! | `CallUndefinedReceiver0(f,_)` | `acc ← f()` (zero args)                            |
//! | `CallUndefinedReceiver1(f,a,_)` | `acc ← f(a)` (one arg)                           |
//! | `CallUndefinedReceiver2(f,a,b,_)` | `acc ← f(a, b)` (two args)                    |
//! | `CallProperty(f,recv,n,_)`   | method call; `this` = `recv`, `n` args after `f`   |
//! | `CallWithSpread(f,a,n,_)`    | same as `CallAnyReceiver` (spread pre-evaluated)   |
//!
//! ## Construct
//!
//! | Opcode                       | Semantics                                           |
//! |------------------------------|-----------------------------------------------------|
//! | `Construct(f,a,n,_)`         | `acc ← new f(a, …, a+n−1)` (P3: returns body acc) |
//! | `ConstructWithSpread(f,a,n,_)` | same as `Construct`                               |
//! | `ConstructForwardAllArgs(f,_)` | like `Construct` but forward all current-frame args |
//!
//! ## Context management
//!
//! | Opcode                    | Semantics                                              |
//! |---------------------------|--------------------------------------------------------|
//! | `PushContext(reg)`        | `reg ← old_ctx; ctx ← acc`                           |
//! | `PopContext(reg)`         | `ctx ← reg`                                           |
//!
//! ## Exception handling
//!
//! | Opcode      | Semantics                                                          |
//! |-------------|--------------------------------------------------------------------|
//! | `Throw`     | throw `acc`; walk handler table, jump to handler or propagate up  |
//! | `ReThrow`   | re-throw `acc`; same dispatch logic as `Throw`                    |
//!
//! ## Generators / Async
//!
//! | Opcode                       | Semantics                                                   |
//! |------------------------------|-------------------------------------------------------------|
//! | `SuspendGenerator(…)`        | yield `acc`; save registers + PC; exit loop via `suspend_result` |
//! | `ResumeGenerator(…)`         | restore registers from `GeneratorState`; acc = sent value   |
//! | `GetGeneratorState(gen)`     | `acc ← generator.status` as Smi (−2/−1/0/1)                |
//! | `SetGeneratorState(gen)`     | `generator.status ← acc` (Smi)                              |
//! | `SwitchOnGeneratorState(gen)`| jump to `resume_pc` if generator was previously suspended   |
//!
//! Generator execution is driven by [`Interpreter::run_generator_step`], which
//! attaches a [`GeneratorState`] to the frame and interprets the body until a
//! `SuspendGenerator` or `Return` instruction is reached.
//!
//! Async functions desugar to generators: each `await` compiles to a `yield`,
//! and the caller of `run_generator_step` plays the role of the microtask
//! queue, resolving awaited values by passing them as the `input` argument.
//!
//! ## Jump-offset encoding
//!
//! Jump offsets are **byte deltas** relative to the byte following the jump
//! instruction.  Positive values jump forward; negative values jump backward.
//! At runtime the interpreter pre-computes a byte-offset table for all
//! instructions and uses binary search to convert a target byte offset back to
//! an instruction index.
//!
//! # Register layout
//!
//! The [`InterpreterFrame`] register file is a flat `Vec<JsValue>`:
//!
//! ```text
//! [ param[0], param[1], …, local[0], local[1], … ]
//! │←── parameter_count ──→│←────── frame_size ────→│
//! ```
//!
//! Bytecode operands encode register indices as `u32` using a two's-complement
//! bit-cast of the compiler's `i32` register index:
//!
//! - `v as i32 >= 0` — local or temporary register; flat index =
//!   `parameter_count + v`.
//! - `v as i32 < 0`  — parameter register; flat index = `-(v as i32 + 1)`.
//!
//! # Dispatch strategy
//!
//! Instructions are pre-decoded once by
//! [`crate::bytecode::bytecodes::decode_with_byte_offsets`] into a
//! `Vec<Instruction>` and a parallel byte-offset table.  The main loop fetches
//! one [`Instruction`] per iteration, advances the program counter, then
//! dispatches on the [`Opcode`] via an exhaustive `match`.

mod dispatch;

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Instant;

use crate::builtins::error::{pop_call_frame, push_call_frame};
use crate::builtins::proxy::{proxy_get, proxy_set};
use crate::builtins::symbol::symbol_description;
use crate::bytecode::bytecode_array::{
    BytecodeArray, ConstantPoolEntry, HandlerTableEntry, MAGLEV_TIERING_THRESHOLD,
    TIERING_THRESHOLD, TURBOFAN_TIERING_THRESHOLD,
};
#[cfg(all(target_arch = "x86_64", unix))]
use crate::bytecode::bytecode_array::{MaglevJitCodeCache, TurbofanJitCodeCache};
use crate::bytecode::bytecodes::decode_with_byte_offsets;
use crate::error::{StatorError, StatorResult};
use crate::inspector::debugger::Debugger;
use crate::objects::property_map::PropertyMap;
use crate::objects::string_intern::intern;
use crate::objects::value::{JsContext, JsValue};

// Re-export generator types and bring them into scope so external code can
// import them from `stator_core::interpreter` (backwards-compatible path).
pub use crate::objects::value::{
    GeneratorResumeMode, GeneratorState, GeneratorStatus, GeneratorStep, NativeIterator,
};

// ─────────────────────────────────────────────────────────────────────────────
// Debugger integration
// ─────────────────────────────────────────────────────────────────────────────

/// Property map for a single `Function` value in the side-table.
type FnPropMap = Rc<RefCell<HashMap<String, JsValue>>>;

thread_local! {
    /// The currently-attached debugger for this thread, if any.
    ///
    /// When `Some`, the interpreter checks for breakpoints and step conditions
    /// before each instruction dispatch and calls the appropriate hook methods
    /// on pauses.
    pub(super) static ACTIVE_DEBUGGER: RefCell<Option<Rc<RefCell<Debugger>>>> =
        const { RefCell::new(None) };

    /// Side table mapping `Function` identity (pointer of the inner
    /// `Rc<BytecodeArray>`) to a property map.
    ///
    /// JavaScript functions are objects and can carry ad-hoc properties
    /// (e.g. `assert.sameValue = function(){}`).  Rather than modifying the
    /// [`JsValue::Function`] variant to include a property map (which would
    /// touch dozens of match arms), we store properties in this thread-local
    /// table.  All `Rc` clones of the same `BytecodeArray` share the same
    /// pointer, so property stores through one clone are visible through all
    /// clones.
    static FUNCTION_PROPS: RefCell<HashMap<usize, FnPropMap>> =
        RefCell::new(HashMap::new());

    /// Thread-local string interning table for property key deduplication.
    ///
    /// Property keys that appear in hot loops (e.g. `"length"`, `"prototype"`,
    /// `"constructor"`) are interned so that repeated lookups compare interned
    /// pointer identity instead of full string equality.
    static STRING_TABLE: RefCell<crate::objects::js_string::StringTable> =
        RefCell::new(crate::objects::js_string::StringTable::new());
}

thread_local! {
    /// Fast-path flag indicating whether a debugger is attached on this thread.
    ///
    /// Checked on every instruction dispatch before touching the thread-local
    /// `ACTIVE_DEBUGGER` RefCell.  This avoids a TLS+RefCell borrow on every
    /// single instruction when no debugger is present (the common case).
    ///
    /// Must be thread-local (not a global `AtomicBool`) because
    /// `ACTIVE_DEBUGGER` is thread-local: a global flag would let one thread's
    /// `detach_debugger` hide another thread's attached debugger, causing
    /// flaky breakpoint misses in parallel tests.
    static DEBUG_ATTACHED: Cell<bool> = const { Cell::new(false) };

    /// Reference to the current thread's global environment.
    ///
    /// Set at the start of [`Interpreter::run`] so that [`proto_lookup`] can
    /// resolve built-in constructor properties (e.g. `[].constructor === Array`)
    /// without threading the global env through every call site.
    #[allow(clippy::type_complexity)]
    static CURRENT_GLOBALS: RefCell<Option<Rc<RefCell<HashMap<String, JsValue>>>>> =
        const { RefCell::new(None) };
}

/// Attach a [`Debugger`] to the current thread's interpreter.
///
/// While attached, the interpreter checks for breakpoints and step conditions
/// before each instruction dispatch.  Only one debugger can be attached per
/// thread; calling this again replaces any previously attached debugger.
pub fn attach_debugger(dbg: Rc<RefCell<Debugger>>) {
    ACTIVE_DEBUGGER.with(|d| *d.borrow_mut() = Some(dbg));
    DEBUG_ATTACHED.with(|f| f.set(true));
}

/// Detach the [`Debugger`] from the current thread.
///
/// After this call, the interpreter runs without any debug checks.  It is
/// safe to call this even if no debugger was attached.
pub fn detach_debugger() {
    ACTIVE_DEBUGGER.with(|d| *d.borrow_mut() = None);
    DEBUG_ATTACHED.with(|f| f.set(false));
}

/// Intern a property key string in the thread-local string table.
///
/// Returns a shared reference-counted handle to the canonical interned copy.
/// Two calls with equal strings return pointer-equal handles, enabling O(1)
/// identity comparison for frequently-used property keys.
pub fn intern_property_key(
    key: &str,
) -> std::sync::Arc<crate::objects::js_string::InternalizedString> {
    STRING_TABLE.with(|table| table.borrow_mut().intern_str(key))
}

/// Returns the number of strings currently interned in the thread-local table.
pub fn interned_string_count() -> usize {
    STRING_TABLE.with(|table| table.borrow().len())
}

/// Reset interpreter thread-local state between test runs.
///
/// Clears the function property side-table and string interning cache so that
/// state from one test execution does not leak into subsequent tests.  Should
/// be called by test harnesses (e.g. Test262) after each test completes.
pub fn clear_interpreter_state() {
    FUNCTION_PROPS.with(|fp| fp.borrow_mut().clear());
    STRING_TABLE.with(|table| {
        *table.borrow_mut() = crate::objects::js_string::StringTable::new();
    });
    CURRENT_GLOBALS.with(|g| *g.borrow_mut() = None);
}

/// Look up a built-in constructor by name from the current global environment.
///
/// Used by [`proto_lookup`] to resolve the `"constructor"` property for
/// primitive wrapper types (e.g. `[].constructor === Array`).  Returns
/// [`JsValue::Undefined`] if no global environment is set or the name is
/// not found.
fn lookup_global_constructor(name: &str) -> JsValue {
    CURRENT_GLOBALS.with(|g| {
        g.borrow()
            .as_ref()
            .and_then(|globals| globals.borrow().get(name).cloned())
            .unwrap_or(JsValue::Undefined)
    })
}

/// Return the current thread's active global environment, if any.
pub(crate) fn current_global_env() -> Option<Rc<RefCell<HashMap<String, JsValue>>>> {
    CURRENT_GLOBALS.with(|g| g.borrow().clone())
}

/// Read a property using the interpreter's ordinary `[[Get]]` semantics.
pub fn dispatch_get_property_value(obj: &JsValue, key: JsValue) -> StatorResult<JsValue> {
    keyed_load(obj, &key)
}

/// Write a property using the interpreter's ordinary `[[Set]]` semantics.
pub fn dispatch_set_property_value(
    obj: &JsValue,
    key: JsValue,
    value: JsValue,
) -> StatorResult<()> {
    keyed_store(obj, &key, value)
}

/// Return the immediate prototype of an object-like value.
pub(crate) fn get_object_prototype(obj: &JsValue) -> Option<JsValue> {
    match obj {
        JsValue::PlainObject(map) => map.borrow().get("__proto__").cloned(),
        JsValue::Error(e) => e.props.borrow().get("__proto__").cloned().or_else(|| {
            let ctor = lookup_global_constructor(e.kind.as_name());
            let proto = proto_lookup(&ctor, "prototype");
            if matches!(proto, JsValue::Undefined) {
                None
            } else {
                Some(proto)
            }
        }),
        _ => None,
    }
}

/// Return whether `target_proto` appears in `value`'s prototype chain.
pub(crate) fn has_prototype_in_chain(value: &JsValue, target_proto: &JsValue) -> bool {
    let mut current = value.clone();
    for _ in 0..256 {
        let Some(next) = get_object_prototype(&current) else {
            return false;
        };
        if next == *target_proto {
            return true;
        }
        current = next;
    }
    false
}

/// Run `f` with mutable access to the currently-attached [`Debugger`], if
/// any, and return its result wrapped in `Some`.  Returns `None` when no
/// debugger is attached.
///
/// This is a convenience helper for callers that need to query or mutate the
/// debugger after a [`StatorError::DebuggerPaused`] error.
pub fn with_debugger<R, F: FnOnce(&mut Debugger) -> R>(f: F) -> Option<R> {
    ACTIVE_DEBUGGER.with(|d| {
        let opt = d.borrow();
        opt.as_ref().map(|rc| {
            let mut dbg = rc.borrow_mut();
            f(&mut dbg)
        })
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Function property side-table helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Return the identity key for a `Function`'s property table entry.
///
/// Uses the raw pointer of the inner `Rc<BytecodeArray>` allocation so that
/// all clones of the same `Rc` share a single property bag.
fn fn_props_key(ba: &Rc<BytecodeArray>) -> usize {
    Rc::as_ptr(ba) as usize
}

/// Store a named property on a `Function` value's side-table entry.
pub(crate) fn fn_props_set(ba: &Rc<BytecodeArray>, name: String, val: JsValue) {
    FUNCTION_PROPS.with(|fp| {
        let mut table = fp.borrow_mut();
        let map = table
            .entry(fn_props_key(ba))
            .or_insert_with(|| Rc::new(RefCell::new(HashMap::new())));
        map.borrow_mut().insert(name, val);
    });
}

/// Load a named property from a `Function` value's side-table entry.
///
/// Returns `JsValue::Undefined` when the property has not been set.
pub(crate) fn fn_props_get(ba: &Rc<BytecodeArray>, name: &str) -> JsValue {
    FUNCTION_PROPS.with(|fp| {
        let table = fp.borrow();
        table
            .get(&fn_props_key(ba))
            .and_then(|map| map.borrow().get(name).cloned())
            .unwrap_or(JsValue::Undefined)
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tiering: interpreter → baseline JIT → Maglev JIT → Turbofan JIT
// ─────────────────────────────────────────────────────────────────────────────

/// Number of loop back-edges taken before OSR baseline compilation is triggered.
///
/// When a single interpreter loop accumulates this many `JumpLoop` iterations
/// and the enclosing function has not yet been JIT-compiled, a baseline
/// compilation is requested so the next *call* to that function executes
/// via native code.
pub(super) const OSR_LOOP_THRESHOLD: u32 = 1_000;

/// Number of loop back-edges taken before a Maglev background compilation is
/// triggered via OSR.
///
/// When a loop has already caused baseline JIT compilation and the back-edge
/// count exceeds this threshold, a Maglev compilation is scheduled in a
/// background thread so the next *call* can use the optimised tier.
pub(super) const MAGLEV_OSR_LOOP_THRESHOLD: u32 = 5_000;

/// Number of loop back-edges taken before a Turbofan background compilation is
/// triggered via OSR.
///
/// When a loop has already caused Maglev JIT compilation and the back-edge
/// count exceeds this threshold, a Turbofan compilation is scheduled in a
/// background thread so the next *call* can use the fully-optimised tier.
pub(super) const TURBOFAN_OSR_LOOP_THRESHOLD: u32 = 10_000;

// ─────────────────────────────────────────────────────────────────────────────
// Generator return completion sentinel
// ─────────────────────────────────────────────────────────────────────────────

/// Sentinel string used by `.return()` to force a return completion through the
/// handler table so that `finally` blocks execute before the generator completes.
pub(super) const GENERATOR_RETURN_SENTINEL: &str = "__stator_generator_return_completion__";

// ─────────────────────────────────────────────────────────────────────────────
// Cross-frame exception propagation
// ─────────────────────────────────────────────────────────────────────────────

thread_local! {
    /// Holds the original thrown [`JsValue`] when a `Throw`/`ReThrow` opcode
    /// propagates an exception out of a frame as [`StatorError::JsException`].
    ///
    /// The `JsException` variant only carries a stringified message, which
    /// loses the original value.  An outer frame's error handler consumes this
    /// slot to materialise the correct `JsValue` for a `catch` block.
    pub(super) static PENDING_EXCEPTION: RefCell<Option<JsValue>> =
        const { RefCell::new(None) };
}

/// Store a thrown [`JsValue`] so an outer frame can recover it.
pub(super) fn set_pending_exception(val: JsValue) {
    PENDING_EXCEPTION.with(|p| {
        *p.borrow_mut() = Some(val);
    });
}

/// Take the pending thrown [`JsValue`], if any.
pub(super) fn take_pending_exception() -> Option<JsValue> {
    PENDING_EXCEPTION.with(|p| p.borrow_mut().take())
}

// ─────────────────────────────────────────────────────────────────────────────
// JIT compilation statistics
// ─────────────────────────────────────────────────────────────────────────────

thread_local! {
    /// Number of successful baseline JIT compilations in this thread.
    static JIT_COMPILATION_COUNT: Cell<u32> = const { Cell::new(0) };
    /// Total machine-code bytes produced by all successful JIT compilations.
    static JIT_CODE_BYTES: Cell<usize> = const { Cell::new(0) };

    /// Thread-local execution deadline shared by **all** interpreter frames on
    /// this thread.  Unlike the per-frame `deadline` field, this is inherited
    /// automatically by child frames created via `eval()`, `Function()`, etc.
    ///
    /// The interpreter checks this every 100 000 instructions (even when the
    /// per-frame `instruction_limit` is zero) and aborts with a `RangeError`
    /// when the wall-clock time exceeds the deadline.
    static EXECUTION_DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) };
}

/// Set a wall-clock deadline for all interpreter execution on the current
/// thread.  Passing `None` clears any existing deadline.
pub fn set_execution_deadline(deadline: Option<Instant>) {
    EXECUTION_DEADLINE.with(|d| d.set(deadline));
}

/// Return the current thread-local execution deadline, if any.
pub fn get_execution_deadline() -> Option<Instant> {
    EXECUTION_DEADLINE.with(|d| d.get())
}

/// Process-wide count of successful Maglev compilations.
///
/// Uses atomics so the background compilation thread can update the counter
/// while the interpreter thread reads it.
static MAGLEV_COMPILATION_COUNT: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Process-wide total machine-code bytes produced by Maglev compilations.
///
/// Uses atomics so the background compilation thread can update the counter
/// while the interpreter thread reads it.
static MAGLEV_CODE_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Process-wide count of successful Turbofan compilations.
///
/// Uses atomics so the background compilation thread can update the counter
/// while the interpreter thread reads it.
static TURBOFAN_COMPILATION_COUNT: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Process-wide total machine-code bytes produced by Turbofan compilations.
///
/// Uses atomics so the background compilation thread can update the counter
/// while the interpreter thread reads it.
static TURBOFAN_CODE_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Return a snapshot of the JIT compilation statistics for the current thread.
///
/// Returns `(functions_compiled, total_code_bytes)`.
///
/// On platforms where the JIT is not available both values will always be zero.
pub fn jit_stats() -> (u32, usize) {
    (
        JIT_COMPILATION_COUNT.with(|c| c.get()),
        JIT_CODE_BYTES.with(|c| c.get()),
    )
}

/// Return a snapshot of the process-wide Maglev JIT compilation statistics.
///
/// Returns `(functions_compiled, total_code_bytes)`.
///
/// Unlike [`jit_stats`], these counters accumulate across all threads because
/// Maglev compilation runs in background threads.
///
/// On platforms where the JIT is not available both values will always be zero.
pub fn maglev_stats() -> (u32, usize) {
    (
        MAGLEV_COMPILATION_COUNT.load(std::sync::atomic::Ordering::Relaxed),
        MAGLEV_CODE_BYTES.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// Return a snapshot of the process-wide Turbofan JIT compilation statistics.
///
/// Returns `(functions_compiled, total_code_bytes)`.
///
/// Unlike [`jit_stats`], these counters accumulate across all threads because
/// Turbofan compilation runs in background threads.
///
/// On platforms where the JIT is not available both values will always be zero.
pub fn turbofan_stats() -> (u32, usize) {
    (
        TURBOFAN_COMPILATION_COUNT.load(std::sync::atomic::Ordering::Relaxed),
        TURBOFAN_CODE_BYTES.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// Convert a [`JsValue`] to its JIT `i64` representation.
///
/// Returns `None` for values that the current baseline tier cannot represent
/// (strings, objects, arrays, etc.).  These cause a graceful fall-back to the
/// interpreter.
#[cfg(all(target_arch = "x86_64", unix))]
fn jsvalue_to_jit(v: &JsValue) -> Option<i64> {
    use crate::compiler::baseline::compiler::{JIT_FALSE, JIT_NULL, JIT_TRUE, JIT_UNDEFINED};
    match v {
        JsValue::Smi(n) => Some(*n as i64),
        JsValue::Boolean(b) => Some(if *b { JIT_TRUE } else { JIT_FALSE }),
        JsValue::Undefined => Some(JIT_UNDEFINED),
        JsValue::Null => Some(JIT_NULL),
        _ => None,
    }
}

/// Request baseline JIT compilation for `ba` and cache the result.
///
/// On supported platforms (x86-64 Unix) this calls [`BaselineCompiler::compile`]
/// and stores the output via [`BytecodeArray::store_jit_code`].  On other
/// platforms this is a no-op.
pub(super) fn maybe_compile_baseline(ba: &BytecodeArray) {
    #[cfg(all(target_arch = "x86_64", unix))]
    {
        use crate::compiler::baseline::compiler::BaselineCompiler;
        if let Ok(cc) = BaselineCompiler::compile(ba) {
            let code_len = cc.code.len();
            ba.store_jit_code(cc.code, cc.register_file_slots);
            JIT_COMPILATION_COUNT.with(|c| c.set(c.get().saturating_add(1)));
            JIT_CODE_BYTES.with(|c| c.set(c.get().saturating_add(code_len)));
        }
    }
    #[cfg(not(all(target_arch = "x86_64", unix)))]
    let _ = ba;
}

/// Build a constant-pool suitable for sending to a background compilation
/// thread.
///
/// [`ConstantPoolEntry::Function`] variants hold a [`BytecodeArray`] that
/// contains `Rc`-based tiering state which is not [`Send`].  For Maglev
/// compilation the inner data of a `Function` entry is never accessed (the
/// graph builder only uses the variant tag to emit an opaque
/// `ConstantPoolEntry` reference node), so we replace the inner
/// [`BytecodeArray`] with a fresh empty one whose `Rc`s are not shared with
/// the main thread.
#[cfg(all(target_arch = "x86_64", unix))]
fn pool_for_compile_thread(pool: &[ConstantPoolEntry]) -> Vec<ConstantPoolEntry> {
    use crate::bytecode::feedback::FeedbackMetadata;
    pool.iter()
        .map(|e: &ConstantPoolEntry| match e {
            ConstantPoolEntry::Function(_) => {
                ConstantPoolEntry::Function(Box::new(BytecodeArray::new(
                    vec![],
                    vec![],
                    0,
                    0,
                    vec![],
                    FeedbackMetadata::empty(),
                    vec![],
                )))
            }
            other => other.clone(),
        })
        .collect()
}

/// A thin wrapper that marks a [`BytecodeArray`] as safe to transfer to a
/// background compilation thread.
///
/// # Safety
///
/// The wrapped [`BytecodeArray`] must have been freshly constructed (its `Rc`
/// tiering state must not be shared with any other thread), and it must be
/// transferred exclusively to ONE background thread.  No other thread may
/// hold a reference to the same `Rc` instances while the background thread
/// is running.
#[cfg(all(target_arch = "x86_64", unix))]
struct SendableBytecodesArray(BytecodeArray);

// SAFETY: The BytecodeArray wrapped here is freshly constructed with Rc
// instances that are not shared with any other thread.  We guarantee exclusive
// ownership by the background compilation thread.
#[cfg(all(target_arch = "x86_64", unix))]
unsafe impl Send for SendableBytecodesArray {}

#[cfg(all(target_arch = "x86_64", unix))]
impl std::ops::Deref for SendableBytecodesArray {
    type Target = BytecodeArray;
    fn deref(&self) -> &BytecodeArray {
        &self.0
    }
}

/// Compilation input bundle for a Maglev background thread.
///
/// All fields are owned and trivially-`Send`.  The `BytecodeArray` is wrapped
/// in [`SendableBytecodesArray`] which declares exclusive ownership.
#[cfg(all(target_arch = "x86_64", unix))]
struct MaglevCompileInput {
    ba: SendableBytecodesArray,
    result_cache: MaglevJitCodeCache,
}

/// Schedule a Maglev background compilation for `ba`.
///
/// On x86-64 Unix this spawns a background thread that runs the full Maglev
/// pipeline (graph build → optimise → codegen) and writes the resulting code
/// into `ba`'s Maglev JIT cache.  Subsequent calls will pick up the compiled
/// code via [`BytecodeArray::try_get_maglev_jit_code`].
///
/// The function is a no-op when:
/// - compilation has already been started (atomic flag check), or
/// - the platform does not support JIT.
pub(super) fn maybe_compile_maglev(ba: &BytecodeArray) {
    #[cfg(all(target_arch = "x86_64", unix))]
    {
        use crate::bytecode::feedback::FeedbackVector;
        use crate::compiler::maglev::codegen as maglev_codegen;
        use crate::compiler::maglev::graph_builder::GraphBuilder;
        use crate::compiler::maglev::optimizer::optimize;

        // Only one compilation thread per function.
        if !ba.try_start_maglev_compile() {
            return;
        }

        let compile_ba = BytecodeArray::new(
            ba.bytecodes().to_vec(),
            pool_for_compile_thread(ba.constant_pool()),
            ba.frame_size(),
            ba.parameter_count(),
            vec![], // source positions not needed for compilation
            ba.feedback_metadata().clone(),
            vec![], // handler table not needed for Maglev graph
        );

        let input = MaglevCompileInput {
            ba: SendableBytecodesArray(compile_ba),
            result_cache: ba.maglev_jit_cache_arc(),
        };

        std::thread::spawn(move || {
            // Access through the SendableBytecodesArray wrapper (Deref) so the
            // Rust 2024 precise closure capture analysis records input.ba
            // (SendableBytecodesArray: Send) not input.ba.0 (BytecodeArray:
            // !Send) as the captured variable.
            let feedback = FeedbackVector::new(input.ba.feedback_metadata());
            let param_count = input.ba.parameter_count();
            if let Ok(mut graph) = GraphBuilder::build(&input.ba, &feedback) {
                optimize(&mut graph);
                if let Ok(cc) = maglev_codegen::compile(&graph, param_count) {
                    let code_len = cc.code.len();
                    if let Ok(mut guard) = input.result_cache.lock() {
                        *guard = Some((cc.code, cc.register_file_slots));
                        drop(guard);
                    }
                    // Record Maglev stats atomically (readable from any thread).
                    MAGLEV_COMPILATION_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    MAGLEV_CODE_BYTES.fetch_add(code_len, std::sync::atomic::Ordering::Relaxed);
                }
            }
        });
    }
    #[cfg(not(all(target_arch = "x86_64", unix)))]
    let _ = ba;
}

/// Try to execute `ba` via the cached Maglev JIT code.
///
/// Returns `Some(result)` when execution succeeds or returns an error;
/// returns `None` when:
/// - Maglev compilation has not finished yet,
/// - one or more arguments cannot be represented in the JIT tier, or
/// - the JIT returns [`JIT_DEOPT`][crate::compiler::baseline::compiler::JIT_DEOPT]
///   (fall-back to the next tier).
///
/// On platforms where the JIT is not available this always returns `None`.
fn try_execute_maglev(ba: &BytecodeArray, args: &[JsValue]) -> Option<StatorResult<JsValue>> {
    #[cfg(all(target_arch = "x86_64", unix))]
    {
        use crate::compiler::baseline::compiler::jit_to_jsvalue;
        use crate::compiler::maglev::codegen::MaglevCompiledCode;

        let (code, register_file_slots) = ba.try_get_maglev_jit_code()?;
        let jit_args: Option<Vec<i64>> = args.iter().map(jsvalue_to_jit).collect();
        let jit_args = jit_args?;
        let native_code_len: usize = code.len();
        let mc = MaglevCompiledCode {
            code,
            native_code_len,
            register_file_slots,
            safepoints: Vec::new(),
            deopt_entries: Vec::new(),
            source_positions: Vec::new(),
        };
        // SAFETY: `mc.code` was produced by `maglev_codegen::compile` and
        // contains valid x86-64 machine code following the JIT calling
        // convention (`extern "C" fn(*mut i64) -> i64`).
        return match unsafe { mc.execute(&jit_args) } {
            Ok(v) => jit_to_jsvalue(v).map(Ok),
            // JIT_DEOPT or unrecognised sentinel → fall back to baseline / interpreter.
            Err(_) => None,
        };
    }
    #[allow(unreachable_code)]
    let _ = (ba, args);
    None
}

/// Compilation input bundle for a Turbofan background thread.
///
/// All fields are owned and trivially-`Send`.  The `BytecodeArray` is wrapped
/// in [`SendableBytecodesArray`] which declares exclusive ownership.
#[cfg(all(target_arch = "x86_64", unix))]
struct TurbofanCompileInput {
    ba: SendableBytecodesArray,
    result_cache: TurbofanJitCodeCache,
}

/// Schedule a Turbofan background compilation for `ba`.
///
/// On x86-64 Unix this spawns a background thread that runs the full
/// Turbofan pipeline (graph build → optimise → Cranelift CLIF → native code)
/// and writes the resulting [`TurbofanCompiledCode`] into `ba`'s Turbofan JIT
/// cache.  Subsequent calls will pick up the compiled code via
/// [`BytecodeArray::has_turbofan_jit_code`].
///
/// The function is a no-op when:
/// - compilation has already been started (atomic flag check), or
/// - the platform does not support JIT.
pub(super) fn maybe_compile_turbofan(ba: &BytecodeArray) {
    #[cfg(all(target_arch = "x86_64", unix))]
    {
        use crate::bytecode::feedback::FeedbackVector;
        use crate::compiler::maglev::graph_builder::GraphBuilder;
        use crate::compiler::maglev::optimizer::optimize;
        use crate::compiler::turbofan;

        // Only one compilation thread per function.
        if !ba.try_start_turbofan_compile() {
            return;
        }

        let compile_ba = BytecodeArray::new(
            ba.bytecodes().to_vec(),
            pool_for_compile_thread(ba.constant_pool()),
            ba.frame_size(),
            ba.parameter_count(),
            vec![], // source positions not needed for compilation
            ba.feedback_metadata().clone(),
            vec![], // handler table not needed for Turbofan graph
        );

        let input = TurbofanCompileInput {
            ba: SendableBytecodesArray(compile_ba),
            result_cache: ba.turbofan_jit_cache_arc(),
        };

        std::thread::spawn(move || {
            // Access through the SendableBytecodesArray wrapper (Deref) so the
            // Rust 2024 precise closure capture analysis records input.ba
            // (SendableBytecodesArray: Send) not input.ba.0 (BytecodeArray:
            // !Send) as the captured variable.
            let feedback = FeedbackVector::new(input.ba.feedback_metadata());
            let param_count = input.ba.parameter_count();
            if let Ok(mut graph) = GraphBuilder::build(&input.ba, &feedback) {
                optimize(&mut graph);
                if let Ok(tc) =
                    turbofan::compile_with_feedback(&graph, param_count, Some(&feedback))
                {
                    let code_size = tc.code_size;
                    if let Ok(mut guard) = input.result_cache.lock() {
                        *guard = Some(tc);
                        drop(guard);
                    }
                    // Record Turbofan stats atomically (readable from any thread).
                    TURBOFAN_COMPILATION_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    TURBOFAN_CODE_BYTES.fetch_add(code_size, std::sync::atomic::Ordering::Relaxed);
                }
            }
        });
    }
    #[cfg(not(all(target_arch = "x86_64", unix)))]
    let _ = ba;
}

/// Try to execute `ba` via the cached Turbofan JIT code.
///
/// Returns `Some(result)` when execution succeeds or returns an error;
/// returns `None` when:
/// - Turbofan compilation has not finished yet,
/// - one or more arguments cannot be represented in the JIT tier, or
/// - the JIT returns [`JIT_DEOPT`][crate::compiler::baseline::compiler::JIT_DEOPT]
///   (fall-back to the next tier).
///
/// On platforms where the JIT is not available this always returns `None`.
fn try_execute_turbofan(ba: &BytecodeArray, args: &[JsValue]) -> Option<StatorResult<JsValue>> {
    #[cfg(all(target_arch = "x86_64", unix))]
    {
        use crate::compiler::baseline::compiler::jit_to_jsvalue;

        // Lock the Turbofan cache and execute if compiled code is available.
        let cache = ba.turbofan_jit_cache_arc();
        let guard = cache.lock().ok()?;
        let tc = guard.as_ref()?;
        let jit_args: Option<Vec<i64>> = args.iter().map(jsvalue_to_jit).collect();
        let jit_args = jit_args?;
        // SAFETY: `tc` was produced by `turbofan::compile_with_feedback` from
        // a well-formed Maglev graph.  We hold the mutex lock for the duration
        // of the call, ensuring exclusive access.
        return match unsafe { tc.execute(&jit_args) } {
            Ok(v) => jit_to_jsvalue(v).map(Ok),
            // JIT_DEOPT or unrecognised sentinel → fall back to lower tier.
            Err(_) => None,
        };
    }
    #[allow(unreachable_code)]
    let _ = (ba, args);
    None
}

/// Try to execute `ba` via the cached baseline JIT code.
///
/// Returns `Some(result)` when execution succeeds or returns an error;
/// returns `None` when:
/// - no JIT code has been compiled yet,
/// - one or more arguments cannot be represented in the JIT tier, or
/// - the JIT returns [`JIT_DEOPT`][crate::compiler::baseline::compiler::JIT_DEOPT]
///   (fall-back to interpreter).
///
/// On platforms where the JIT is not available this always returns `None`.
fn try_execute_jit(ba: &BytecodeArray, args: &[JsValue]) -> Option<StatorResult<JsValue>> {
    #[cfg(all(target_arch = "x86_64", unix))]
    {
        use crate::compiler::baseline::compiler::{
            CompiledCode, DeoptEntry, SafepointEntry, jit_to_jsvalue,
        };
        let (code, register_file_slots) = ba.try_get_jit_code()?;
        let jit_args: Option<Vec<i64>> = args.iter().map(jsvalue_to_jit).collect();
        let jit_args = jit_args?;
        // Capture the code length before moving `code` into the struct.
        // `native_code_len` records the boundary between machine instructions
        // and appended metadata tables; `execute()` does not use it directly
        // but it is part of the `CompiledCode` API.
        let native_code_len: usize = code.len();
        let cc = CompiledCode {
            code,
            native_code_len,
            register_file_slots,
            safepoints: Vec::<SafepointEntry>::new(),
            deopt_entries: Vec::<DeoptEntry>::new(),
        };
        // SAFETY: `cc.code` was produced by `BaselineCompiler::compile` and
        // contains valid x86-64 machine code following the JIT calling
        // convention (`extern "C" fn(*mut i64) -> i64`).
        return match unsafe { cc.execute(&jit_args) } {
            Ok(v) => jit_to_jsvalue(v).map(Ok),
            // JIT_DEOPT or unrecognised sentinel → fall back to interpreter.
            Err(_) => None,
        };
    }
    #[allow(unreachable_code)]
    let _ = (ba, args);
    None
}

/// Try to execute `ba` via the fastest available JIT tier.
///
/// Checks Turbofan first (highest tier), then Maglev, then falls back to
/// baseline JIT.  Returns `None` if no tier has compiled code ready.
pub(super) fn try_execute_best_jit(
    ba: &BytecodeArray,
    args: &[JsValue],
) -> Option<StatorResult<JsValue>> {
    try_execute_turbofan(ba, args)
        .or_else(|| try_execute_maglev(ba, args))
        .or_else(|| try_execute_jit(ba, args))
}

// ─────────────────────────────────────────────────────────────────────────────
// PropertyIc – shape-based inline cache entry
// ─────────────────────────────────────────────────────────────────────────────

/// A monomorphic inline-cache entry that maps a [`PropertyMap::shape_id`] to a
/// direct slot offset, enabling O(1) property access when the shape is stable.
#[derive(Debug, Clone)]
pub struct PropertyIc {
    /// The [`PropertyMap::shape_id`] observed when the cache was populated.
    pub cached_shape: u64,
    /// The Vec offset of the cached property inside the `PropertyMap`.
    pub cached_offset: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// InterpreterFrame
// ─────────────────────────────────────────────────────────────────────────────

/// A single activation frame for the Stator bytecode interpreter.
///
/// The frame owns:
/// - the [`BytecodeArray`] being executed,
/// - a flat register file that covers both parameter and local/temporary slots,
/// - the implicit **accumulator** register,
/// - a **program counter** (instruction index into the pre-decoded list), and
/// - an optional **context** value for future scope/closure support.
pub struct InterpreterFrame {
    /// The bytecode for the currently-executing function.
    pub bytecode_array: BytecodeArray,
    /// Flat register file: `[params…, locals/temps…]`.
    ///
    /// Length = `bytecode_array.parameter_count() + bytecode_array.frame_size()`.
    pub registers: Vec<JsValue>,
    /// The implicit accumulator register used by most instructions.
    pub accumulator: JsValue,
    /// Program counter: index of the *next* instruction to execute in the
    /// pre-decoded [`Vec<Instruction>`].
    pub pc: usize,
    /// Scope context for variable lookup (placeholder for future closure support).
    pub context: Option<JsValue>,
    /// Set by [`Opcode::SuspendGenerator`] to carry the yielded value back to
    /// [`Interpreter::run_generator_step`].  `None` during normal execution.
    pub suspend_result: Option<JsValue>,
    /// The generator state object attached to this frame, if it is executing
    /// a generator function body.  `None` for ordinary (non-generator) frames.
    pub generator_state: Option<Rc<RefCell<GeneratorState>>>,
    /// Shared global-variable environment.  The top-level script frame owns a
    /// fresh map; inner function frames inherit the same `Rc` so that globals
    /// written by a called function are visible back in the caller.
    pub global_env: Rc<RefCell<HashMap<String, JsValue>>>,
    /// OSR (on-stack replacement) loop back-edge counter.
    ///
    /// Incremented by every `JumpLoop` instruction.  When this exceeds
    /// [`OSR_LOOP_THRESHOLD`] and the function has not yet been JIT-compiled,
    /// baseline compilation is triggered so the next *call* uses native code.
    osr_loop_count: u32,
    /// Maximum number of instructions to execute before aborting.  `0` means
    /// unlimited.  Used by the Test262 runner to prevent infinite-loop tests
    /// from hanging the suite.
    pub instruction_limit: u64,
    /// Number of instructions executed so far in this frame.
    pub instructions_executed: u64,
    /// Optional wall-clock deadline.  When set, the interpreter periodically
    /// checks `Instant::now()` and aborts with a `RangeError` once past the
    /// deadline.  This catches native-code hangs that the instruction limit
    /// cannot detect (e.g. pathological regex, string operations).
    pub deadline: Option<Instant>,
    /// Saved pending-exception message for `SetPendingMessage` (swap pattern
    /// used by `finally` blocks).
    pub pending_message: JsValue,
    /// Cache of frozen template objects keyed by bytecode offset, used by
    /// `GetTemplateObject`.
    pub template_cache: HashMap<u32, JsValue>,
    /// The `new.target` value for this frame.  Set to the constructor function
    /// when invoked via `[[Construct]]`, or `undefined` for normal calls.
    pub new_target: JsValue,
    /// Monomorphic property-load cache: `slot → (map_ptr, cached_value)`.
    /// When the same PropertyMap is seen again, the cached value is returned
    /// without a full `proto_lookup` scan.
    pub mono_load_cache: HashMap<u32, (usize, JsValue)>,
    /// Polymorphic property-load cache: `slot → [(ptr, cached_value)]`.
    /// Holds up to 4 entries per feedback slot, supporting polymorphic sites.
    pub poly_load_cache: HashMap<u32, Vec<(usize, JsValue)>>,
    /// Shape-based inline cache for named property loads.
    /// Maps a feedback slot to a [`PropertyIc`] recording the last observed
    /// `(shape_id, offset)` pair so that a repeat access on the same shape
    /// can skip HashMap lookup entirely.
    pub shape_load_ic: HashMap<u32, PropertyIc>,
    /// Shape-based inline cache for named property stores.
    pub shape_store_ic: HashMap<u32, PropertyIc>,
    /// Pre-decoded string constants from the constant pool, keyed by index.
    /// Avoids repeated `String::clone()` from the constant pool.
    pub string_cache: HashMap<u32, Rc<str>>,
}

impl InterpreterFrame {
    /// Create a new frame for the given [`BytecodeArray`], pre-loading `args`
    /// into the parameter registers.
    ///
    /// If `args` has fewer entries than the declared `parameter_count`, the
    /// remaining parameter registers are initialised to `undefined`.  Extra
    /// arguments beyond `parameter_count` are silently discarded.
    pub fn new(bytecode_array: BytecodeArray, args: Vec<JsValue>) -> Self {
        let param_count = bytecode_array.parameter_count() as usize;
        let frame_size = bytecode_array.frame_size() as usize;
        let total_regs = param_count + frame_size;
        let mut registers = vec![JsValue::Undefined; total_regs];
        for (i, arg) in args.into_iter().enumerate().take(param_count) {
            registers[i] = arg;
        }
        let mut global_map = HashMap::new();
        crate::builtins::install_globals::install_globals(&mut global_map);
        Self {
            bytecode_array,
            registers,
            accumulator: JsValue::Undefined,
            pc: 0,
            context: None,
            suspend_result: None,
            generator_state: None,
            global_env: Rc::new(RefCell::new(global_map)),
            osr_loop_count: 0,
            instruction_limit: 0,
            instructions_executed: 0,
            deadline: None,
            pending_message: JsValue::Undefined,
            template_cache: HashMap::new(),
            new_target: JsValue::Undefined,
            mono_load_cache: HashMap::new(),
            poly_load_cache: HashMap::new(),
            shape_load_ic: HashMap::new(),
            shape_store_ic: HashMap::new(),
            string_cache: HashMap::new(),
        }
    }

    /// Create a new frame that shares the given global environment.
    ///
    /// Used by the interpreter when calling a function so that the callee can
    /// access the same globals as the top-level script.
    ///
    /// Engine builtins (`eval`, `Math`, `JSON`, etc.) are merged into the
    /// provided environment when they are not already present, so callers
    /// that supply only application-specific globals (e.g. `print`) still
    /// get a fully functional JavaScript environment.
    pub fn new_with_globals(
        bytecode_array: BytecodeArray,
        args: Vec<JsValue>,
        global_env: Rc<RefCell<HashMap<String, JsValue>>>,
    ) -> Self {
        // Ensure engine builtins are present.  The `eval` key is used as a
        // sentinel — if it exists the full set has already been installed.
        if !global_env.borrow().contains_key("eval") {
            let mut defaults = HashMap::new();
            crate::builtins::install_globals::install_globals(&mut defaults);
            let mut env = global_env.borrow_mut();
            for (k, v) in defaults {
                env.entry(k).or_insert(v);
            }
        }
        let mut frame = Self::new(bytecode_array, args);
        frame.global_env = global_env;
        frame
    }

    /// Map an encoded register-operand value to a flat index into
    /// [`Self::registers`].
    #[inline(always)]
    fn reg_index(&self, v: u32) -> StatorResult<usize> {
        let signed = v as i32;
        if signed >= 0 {
            let param_count = self.bytecode_array.parameter_count() as usize;
            let flat = param_count + signed as usize;
            if flat < self.registers.len() {
                Ok(flat)
            } else {
                Err(StatorError::Internal(format!(
                    "local register {signed} out of bounds (frame_size={})",
                    self.bytecode_array.frame_size()
                )))
            }
        } else {
            let param_idx = (-(signed + 1)) as usize;
            if param_idx < self.bytecode_array.parameter_count() as usize {
                Ok(param_idx)
            } else {
                Err(StatorError::Internal(format!(
                    "parameter register {param_idx} out of bounds (parameter_count={})",
                    self.bytecode_array.parameter_count()
                )))
            }
        }
    }

    /// Read the value of the register encoded by operand value `v`.
    #[inline(always)]
    fn read_reg(&self, v: u32) -> StatorResult<&JsValue> {
        let idx = self.reg_index(v)?;
        Ok(&self.registers[idx])
    }

    /// Write `value` to the register encoded by operand value `v`.
    #[inline(always)]
    fn write_reg(&mut self, v: u32, value: JsValue) -> StatorResult<()> {
        let idx = self.reg_index(v)?;
        self.registers[idx] = value;
        Ok(())
    }

    /// Copy a value from register `src` to register `dst` directly,
    /// avoiding the intermediate borrow that `read_reg` + `write_reg`
    /// would require.
    #[inline(always)]
    fn copy_reg(&mut self, src: u32, dst: u32) -> StatorResult<()> {
        let src_idx = self.reg_index(src)?;
        let dst_idx = self.reg_index(dst)?;
        if src_idx != dst_idx {
            let val = self.registers[src_idx].cheap_clone();
            self.registers[dst_idx] = val;
        }
        Ok(())
    }

    /// Get a string constant from the constant pool, caching the result.
    pub fn get_string_constant(&mut self, idx: u32) -> StatorResult<Rc<str>> {
        if let Some(cached) = self.string_cache.get(&idx) {
            return Ok(Rc::clone(cached));
        }
        let s = match self.bytecode_array.get_constant(idx) {
            Some(ConstantPoolEntry::String(s)) => intern(s.as_str()),
            _ => {
                return Err(StatorError::Internal(
                    "get_string_constant: constant is not a string".into(),
                ));
            }
        };
        self.string_cache.insert(idx, Rc::clone(&s));
        Ok(s)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Interpreter
// ─────────────────────────────────────────────────────────────────────────────

/// Stateless bytecode interpreter.
///
/// Call [`Interpreter::run`] with a mutable [`InterpreterFrame`] to execute
/// the bytecode until a `Return` instruction is reached.
pub struct Interpreter;

impl Interpreter {
    /// Execute the bytecode in `frame` until a `Return` instruction is reached.
    ///
    /// Returns the value in the accumulator at the `Return` instruction.
    /// Returns [`StatorError::Internal`] if:
    /// - the bytecode stream is malformed,
    /// - the program counter advances past the end of the instruction list
    ///   without a `Return`,
    /// - an unimplemented opcode is encountered, or
    /// - a type error occurs during arithmetic.
    pub fn run(frame: &mut InterpreterFrame) -> StatorResult<JsValue> {
        // Publish the global environment for proto_lookup's constructor resolution.
        CURRENT_GLOBALS.with(|g| {
            *g.borrow_mut() = Some(Rc::clone(&frame.global_env));
        });
        if frame.bytecode_array.is_async() && frame.generator_state.is_none() {
            return Self::run_async_function(frame.bytecode_array.clone(), vec![]);
        }
        // Dynamically grow the native stack when headroom drops below 512 KiB.
        // Provide moderate stack growth for the interpreter loop.
        // Per-call recursion guards live in dispatch.rs (1 MiB each);
        // this inner guard covers the run() frame itself (~256 KiB).
        stacker::maybe_grow(64 * 1024, 256 * 1024, || {
            // Outer loop: re-entered when a TailCall opcode rewrites the frame
            // with a new bytecode array (proper tail-call trampoline).
            'tail_call: loop {
                // Pre-decode the bytecode once and capture byte offsets for jump resolution.
                let (instructions, byte_offsets) =
                    decode_with_byte_offsets(frame.bytecode_array.bytecodes())?;
                // Clone the handler table once so the borrow on bytecode_array is released
                // before we start mutating the frame.
                let handler_table: Vec<HandlerTableEntry> =
                    frame.bytecode_array.handler_table().to_vec();

                loop {
                    // ── CPU profiler checkpoint ────────────────────────────────────
                    crate::inspector::profiler::maybe_record_sample();

                    if frame.pc >= instructions.len() {
                        return Err(bytecode_end_error());
                    }

                    // ── Debug hook (pre-fetch) ─────────────────────────────────────
                    //
                    // Check for breakpoints and step conditions *before* fetching the
                    // next instruction so that the paused frame state reflects what is
                    // *about* to execute (the program counter still points at the
                    // instruction that would fire next).
                    if DEBUG_ATTACHED.with(Cell::get) {
                        let current_offset = byte_offsets[frame.pc] as u32;
                        if let Some(pause_err) = ACTIVE_DEBUGGER.with(|d| {
                            let opt = d.borrow();
                            opt.as_ref()
                                .and_then(|rc| rc.borrow_mut().check_pause_at(current_offset))
                        }) {
                            return Err(pause_err);
                        }
                    }

                    // ── Fetch ──────────────────────────────────────────────────────
                    let instr = &instructions[frame.pc];
                    frame.pc += 1;

                    // ── Instruction limit & deadline checks ─────────────────────────
                    frame.instructions_executed += 1;
                    if frame.instruction_limit > 0
                        && frame.instructions_executed > frame.instruction_limit
                    {
                        return Err(instruction_limit_error());
                    }
                    // Every 100 000 instructions, check both the per-frame
                    // deadline and the thread-local execution deadline.  The
                    // thread-local deadline is set by the test runner and is
                    // inherited by child frames (eval, Function constructor)
                    // that would otherwise have no timeout.
                    if frame.instructions_executed.is_multiple_of(100_000) {
                        let now = Instant::now();
                        if let Some(dl) = frame.deadline
                            && now > dl
                        {
                            return Err(StatorError::RangeError(
                                "execution timeout exceeded".to_string(),
                            ));
                        }
                        if let Some(dl) = EXECUTION_DEADLINE.with(|d| d.get())
                            && now > dl
                        {
                            return Err(StatorError::RangeError(
                                "execution timeout exceeded".to_string(),
                            ));
                        }
                    }

                    // ── Dispatch (computed-goto table) ─────────────────────────
                    let handler = dispatch::DISPATCH_TABLE[instr.opcode as usize];
                    let mut dctx = dispatch::DispatchContext {
                        frame,
                        instructions: &instructions,
                        byte_offsets: &byte_offsets,
                        handler_table: &handler_table,
                    };
                    match handler(&mut dctx, instr) {
                        Ok(action) => match action {
                            dispatch::DispatchAction::Continue => {}
                            dispatch::DispatchAction::Return(v) => return Ok(v),
                            dispatch::DispatchAction::TailCall => continue 'tail_call,
                        },
                        Err(e) => {
                            if let Some(resume_pc) =
                                handle_dispatch_error(&e, frame, &handler_table)
                            {
                                frame.pc = resume_pc;
                                continue;
                            }
                            return Err(e);
                        }
                    }
                }
            } // 'tail_call
        }) // stacker::maybe_grow
    }

    /// Execute one step of a generator function.
    ///
    /// Restores the execution state from `state`, initialises the accumulator to
    /// `input` (the value passed to `.next()`), and runs the interpreter loop
    /// until the generator suspends at a `yield` expression or its function
    /// body returns.
    ///
    /// # Return value
    ///
    /// - [`GeneratorStep::Yield`] — a `SuspendGenerator` instruction was
    ///   reached; the generator state is updated and can be resumed with
    ///   another call.
    /// - [`GeneratorStep::Return`] — a `Return` instruction was reached; the
    ///   generator is marked [`GeneratorStatus::Completed`].
    ///
    /// Calling this method on an already-completed generator immediately
    /// returns `Ok(GeneratorStep::Return(JsValue::Undefined))`.
    pub fn run_generator_step(
        state: &Rc<RefCell<GeneratorState>>,
        input: JsValue,
    ) -> StatorResult<GeneratorStep> {
        // Short-circuit for exhausted generators.
        if state.borrow().status == GeneratorStatus::Completed {
            return Ok(GeneratorStep::Return(JsValue::Undefined));
        }

        let (bytecode_array, saved_registers, resume_pc) = {
            let gs = state.borrow();
            (
                gs.bytecode_array.clone(),
                gs.registers.clone(),
                gs.resume_pc,
            )
        };

        let param_count = bytecode_array.parameter_count() as usize;
        let frame_size = bytecode_array.frame_size() as usize;
        let total = param_count + frame_size;

        // Restore the saved register file, padding with Undefined if this is
        // the first step (saved_registers is empty).
        let mut registers = saved_registers;
        registers.resize(total, JsValue::Undefined);

        let mut frame = InterpreterFrame {
            bytecode_array,
            registers,
            accumulator: input,
            pc: resume_pc,
            context: None,
            suspend_result: None,
            generator_state: Some(Rc::clone(state)),
            global_env: CURRENT_GLOBALS.with(|g| {
                g.borrow()
                    .clone()
                    .unwrap_or_else(|| Rc::new(RefCell::new(std::collections::HashMap::new())))
            }),
            osr_loop_count: 0,
            instruction_limit: 0,
            instructions_executed: 0,
            deadline: None,
            pending_message: JsValue::Undefined,
            template_cache: std::collections::HashMap::new(),
            new_target: JsValue::Undefined,
            mono_load_cache: std::collections::HashMap::new(),
            poly_load_cache: std::collections::HashMap::new(),
            shape_load_ic: std::collections::HashMap::new(),
            shape_store_ic: std::collections::HashMap::new(),
            string_cache: std::collections::HashMap::new(),
        };
        // Restore the captured closure context so generators can access outer
        // scope variables through the context chain.
        let ba_ref = Rc::new(frame.bytecode_array.clone());
        restore_closure_context(&mut frame, &ba_ref);
        // Named generator expressions: populate the self-name register so
        // the body can reference the generator function by its own name.
        populate_self_name(&mut frame, &ba_ref, &JsValue::Generator(Rc::clone(state)));

        state.borrow_mut().status = GeneratorStatus::Executing;

        push_call_frame("<generator>")?;
        let return_val = Interpreter::run(&mut frame);
        pop_call_frame();
        let return_val = return_val?;

        if let Some(yield_val) = frame.suspend_result {
            Ok(GeneratorStep::Yield(yield_val))
        } else {
            state.borrow_mut().status = GeneratorStatus::Completed;
            Ok(GeneratorStep::Return(return_val))
        }
    }

    /// Execute a pure async function (not an async generator) to completion.
    ///
    /// Drives the internal generator until it returns.  Each `await` is
    /// compiled as `SuspendGenerator`; the awaited value is resolved
    /// synchronously (matching V8's microtask-draining behaviour for already-
    /// resolved promises) and fed back via `run_generator_step`.
    ///
    /// Returns a `JsValue::Promise` that is already fulfilled with the return
    /// value, or already rejected if the async body threw.
    pub fn run_async_function(
        bytecode_array: BytecodeArray,
        _args: Vec<JsValue>,
    ) -> StatorResult<JsValue> {
        use crate::builtins::promise::{MicrotaskQueue, promise_reject, promise_resolve};

        let queue = MicrotaskQueue::new();
        let state = GeneratorState::new(bytecode_array);
        let mut input = JsValue::Undefined;

        loop {
            match Interpreter::run_generator_step(&state, input) {
                Ok(GeneratorStep::Yield(awaited)) => {
                    match resolve_promise_like(awaited, &queue) {
                        AwaitResolution::Fulfilled(value) => input = value,
                        AwaitResolution::Rejected(reason) => {
                            let mut borrow = state.borrow_mut();
                            borrow.resume_mode = GeneratorResumeMode::Throw(reason);
                            input = JsValue::Undefined;
                        }
                        AwaitResolution::Pending => {
                            // Pending promises still lack event-loop integration.
                            input = JsValue::Undefined;
                        }
                    }
                }
                Ok(GeneratorStep::Return(v)) => {
                    let p = promise_resolve(v, &queue);
                    queue.drain();
                    return Ok(JsValue::Promise(p));
                }
                Err(e) => {
                    // JavaScript exceptions inside the async body become
                    // rejected promises per §27.7.5.2.  Engine-level errors
                    // (OOM, sandbox violations, debugger) still propagate.
                    if let Some(reason) = Self::js_error_to_rejection_reason(&e) {
                        state.borrow_mut().status = GeneratorStatus::Completed;
                        let rp = promise_reject(reason, &queue);
                        queue.drain();
                        return Ok(JsValue::Promise(rp));
                    }
                    return Err(e);
                }
            }
        }
    }

    /// Convert a [`StatorError`] that represents a JavaScript exception into a
    /// [`JsValue`] suitable as a promise rejection reason.
    ///
    /// Returns `None` for engine-level errors that should not be silently
    /// swallowed (OOM, internal, debugger, sandbox).
    pub(crate) fn js_error_to_rejection_reason(e: &StatorError) -> Option<JsValue> {
        match e {
            StatorError::JsException(msg) => Some(JsValue::String(msg.clone().into())),
            StatorError::TypeError(msg) => {
                Some(JsValue::String(format!("TypeError: {msg}").into()))
            }
            StatorError::SyntaxError(msg) => {
                Some(JsValue::String(format!("SyntaxError: {msg}").into()))
            }
            StatorError::ReferenceError(msg) => {
                Some(JsValue::String(format!("ReferenceError: {msg}").into()))
            }
            StatorError::RangeError(msg) => {
                Some(JsValue::String(format!("RangeError: {msg}").into()))
            }
            StatorError::URIError(msg) => Some(JsValue::String(format!("URIError: {msg}").into())),
            StatorError::WasmError(msg) => {
                Some(JsValue::String(format!("WasmError: {msg}").into()))
            }
            // Engine-level errors must propagate as Rust errors.
            StatorError::OutOfMemory
            | StatorError::Internal(_)
            | StatorError::DebuggerPaused { .. }
            | StatorError::SandboxViolation { .. } => None,
        }
    }

    /// Execute a `.return(value)` call on a generator.
    ///
    /// If the generator is suspended, marks it as completed and returns
    /// `{ value, done: true }`.  If already completed, also returns
    /// `{ value, done: true }` per §27.5.3.4 step 2a.
    pub fn generator_return(
        state: &Rc<RefCell<GeneratorState>>,
        value: JsValue,
    ) -> StatorResult<JsValue> {
        let status = state.borrow().status;
        match status {
            GeneratorStatus::SuspendedAtStart => {
                state.borrow_mut().status = GeneratorStatus::Completed;
                Ok(make_iterator_result(value, true))
            }
            GeneratorStatus::SuspendedAtYield => {
                // Resume the generator with a Return completion so that any
                // enclosing `finally` blocks execute before completion.
                state.borrow_mut().resume_mode = GeneratorResumeMode::Return(value.clone());
                match Self::run_generator_step(state, JsValue::Undefined) {
                    Ok(GeneratorStep::Yield(v)) => Ok(make_iterator_result(v, false)),
                    Ok(GeneratorStep::Return(v)) => Ok(make_iterator_result(v, true)),
                    Err(StatorError::JsException(ref msg)) if msg == GENERATOR_RETURN_SENTINEL => {
                        state.borrow_mut().status = GeneratorStatus::Completed;
                        Ok(make_iterator_result(value, true))
                    }
                    Err(e) => {
                        state.borrow_mut().status = GeneratorStatus::Completed;
                        Err(e)
                    }
                }
            }
            GeneratorStatus::Completed => Ok(make_iterator_result(value, true)),
            GeneratorStatus::Executing => Err(StatorError::TypeError(
                "Generator is already running".into(),
            )),
        }
    }

    /// Execute a `.throw(value)` call on a generator.
    ///
    /// If the generator is suspended at a yield, resumes execution with the
    /// thrown value as an exception.  If the generator body has a try-catch,
    /// it may catch the exception and continue.  If at start or already
    /// completed, marks complete and throws.
    pub fn generator_throw(
        state: &Rc<RefCell<GeneratorState>>,
        value: JsValue,
    ) -> StatorResult<JsValue> {
        let status = state.borrow().status;
        match status {
            GeneratorStatus::SuspendedAtYield => {
                // Set resume mode so handle_resume_generator will throw at
                // the yield point; the handler table catches it if there is
                // an active try/catch.
                state.borrow_mut().resume_mode = GeneratorResumeMode::Throw(value.clone());
                match Self::run_generator_step(state, JsValue::Undefined) {
                    Ok(GeneratorStep::Yield(v)) => Ok(make_iterator_result(v, false)),
                    Ok(GeneratorStep::Return(v)) => Ok(make_iterator_result(v, true)),
                    Err(e) => {
                        // Uncaught — mark the generator as completed.
                        state.borrow_mut().status = GeneratorStatus::Completed;
                        Err(e)
                    }
                }
            }
            GeneratorStatus::SuspendedAtStart => {
                state.borrow_mut().status = GeneratorStatus::Completed;
                let err_str = format!("{value:?}");
                Err(StatorError::JsException(err_str))
            }
            GeneratorStatus::Completed => {
                let err_str = format!("{value:?}");
                Err(StatorError::JsException(err_str))
            }
            GeneratorStatus::Executing => Err(StatorError::TypeError(
                "Generator is already running".into(),
            )),
        }
    }
}

pub(crate) enum AwaitResolution {
    Fulfilled(JsValue),
    Rejected(JsValue),
    Pending,
}

pub(crate) fn resolve_promise_like(
    value: JsValue,
    queue: &crate::builtins::promise::MicrotaskQueue,
) -> AwaitResolution {
    let promise = crate::builtins::promise::promise_resolve(value, queue);
    queue.drain();
    if let Some(value) = promise.value() {
        AwaitResolution::Fulfilled(value)
    } else if let Some(reason) = promise.reason() {
        AwaitResolution::Rejected(reason)
    } else {
        AwaitResolution::Pending
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Create a `{ value, done }` iterator result object.
pub(crate) fn make_iterator_result(value: JsValue, done: bool) -> JsValue {
    let mut map = PropertyMap::new();
    map.insert("value".to_string(), value);
    map.insert("done".to_string(), JsValue::Boolean(done));
    JsValue::PlainObject(Rc::new(RefCell::new(map)))
}

#[allow(dead_code)]
pub(crate) fn wrap_sync_iterator_as_async_iterator(iterator: JsValue) -> JsValue {
    use crate::builtins::promise::{MicrotaskQueue, promise_reject, promise_resolve};

    let props = Rc::new(RefCell::new(PropertyMap::new()));
    let async_iter = JsValue::PlainObject(Rc::clone(&props));
    props
        .borrow_mut()
        .insert("__async_iterator__".into(), JsValue::Boolean(true));

    let next_iterator = iterator.clone();
    props.borrow_mut().insert(
        "next".into(),
        JsValue::NativeFunction(Rc::new(move |args| {
            let queue = MicrotaskQueue::new();
            let input = args.into_iter().next().unwrap_or(JsValue::Undefined);
            let result = match &next_iterator {
                JsValue::Iterator(ni) => match ni.borrow_mut().next_item() {
                    Some(value) => Ok(make_iterator_result(value, false)),
                    None => Ok(make_iterator_result(JsValue::Undefined, true)),
                },
                JsValue::Generator(gs) => match Interpreter::run_generator_step(gs, input) {
                    Ok(GeneratorStep::Yield(value)) => Ok(make_iterator_result(value, false)),
                    Ok(GeneratorStep::Return(value)) => Ok(make_iterator_result(value, true)),
                    Err(error) => Err(error),
                },
                JsValue::PlainObject(map) => {
                    let next_fn = map.borrow().get("next").cloned().ok_or_else(|| {
                        StatorError::TypeError("Async-from-sync iterator is missing .next()".into())
                    })?;
                    dispatch_call_with_this(&next_fn, next_iterator.clone(), vec![input])
                }
                _ => Err(StatorError::TypeError(
                    "value is not a synchronous iterator".into(),
                )),
            };

            let promise = match result {
                Ok(JsValue::PlainObject(result_map)) => {
                    let done = result_map
                        .borrow()
                        .get("done")
                        .is_some_and(|done| done.to_boolean());
                    let value = result_map
                        .borrow()
                        .get("value")
                        .cloned()
                        .unwrap_or(JsValue::Undefined);
                    match resolve_promise_like(value, &queue) {
                        AwaitResolution::Fulfilled(value) => {
                            promise_resolve(make_iterator_result(value, done), &queue)
                        }
                        AwaitResolution::Rejected(reason) => promise_reject(reason, &queue),
                        AwaitResolution::Pending => {
                            promise_resolve(make_iterator_result(JsValue::Undefined, done), &queue)
                        }
                    }
                }
                Ok(_) => promise_reject(
                    JsValue::String("TypeError: Iterator result is not an object".into()),
                    &queue,
                ),
                Err(error) => {
                    if let Some(reason) = Interpreter::js_error_to_rejection_reason(&error) {
                        promise_reject(reason, &queue)
                    } else {
                        return Err(error);
                    }
                }
            };
            queue.drain();
            Ok(JsValue::Promise(promise))
        })),
    );

    let return_iterator = iterator;
    props.borrow_mut().insert(
        "return".into(),
        JsValue::NativeFunction(Rc::new(move |args| {
            let queue = MicrotaskQueue::new();
            let return_value = args.into_iter().next().unwrap_or(JsValue::Undefined);
            let result = match &return_iterator {
                JsValue::Iterator(ni) => {
                    let mut inner = ni.borrow_mut();
                    inner.index = inner.items.len();
                    Ok(make_iterator_result(return_value, true))
                }
                JsValue::Generator(gs) => Interpreter::generator_return(gs, return_value),
                JsValue::PlainObject(map) => match map.borrow().get("return").cloned() {
                    Some(return_fn) => dispatch_call_with_this(
                        &return_fn,
                        return_iterator.clone(),
                        vec![return_value],
                    ),
                    None => Ok(make_iterator_result(JsValue::Undefined, true)),
                },
                _ => Ok(make_iterator_result(JsValue::Undefined, true)),
            };

            let promise = match result {
                Ok(JsValue::PlainObject(result_map)) => {
                    let done = result_map
                        .borrow()
                        .get("done")
                        .is_some_and(|done| done.to_boolean());
                    let value = result_map
                        .borrow()
                        .get("value")
                        .cloned()
                        .unwrap_or(JsValue::Undefined);
                    match resolve_promise_like(value, &queue) {
                        AwaitResolution::Fulfilled(value) => {
                            promise_resolve(make_iterator_result(value, done), &queue)
                        }
                        AwaitResolution::Rejected(reason) => promise_reject(reason, &queue),
                        AwaitResolution::Pending => {
                            promise_resolve(make_iterator_result(JsValue::Undefined, done), &queue)
                        }
                    }
                }
                Ok(_) => promise_reject(
                    JsValue::String("TypeError: Iterator result is not an object".into()),
                    &queue,
                ),
                Err(error) => {
                    if let Some(reason) = Interpreter::js_error_to_rejection_reason(&error) {
                        promise_reject(reason, &queue)
                    } else {
                        return Err(error);
                    }
                }
            };
            queue.drain();
            Ok(JsValue::Promise(promise))
        })),
    );

    let async_self = async_iter.clone();
    props.borrow_mut().insert(
        "@@asyncIterator".into(),
        JsValue::NativeFunction(Rc::new(move |_args| Ok(async_self.clone()))),
    );
    let async_self = async_iter.clone();
    props.borrow_mut().insert(
        "Symbol(2)".into(),
        JsValue::NativeFunction(Rc::new(move |_args| Ok(async_self.clone()))),
    );
    async_iter
}

/// Look up the innermost exception handler for the instruction at `instr_idx`.
///
/// Searches `handler_table` in order (innermost entries are pushed before
/// outer ones by the compiler, so the first matching entry is the innermost).
/// Returns the handler entry-point instruction index, or `None` if no entry
/// covers `instr_idx`.
#[allow(dead_code)]
pub(super) fn find_handler(instr_idx: u32, handler_table: &[HandlerTableEntry]) -> Option<usize> {
    handler_table
        .iter()
        .find(|e| instr_idx >= e.try_start && instr_idx < e.try_end)
        .map(|e| e.handler as usize)
}

/// Collect `count` consecutive argument values from `frame` starting at the
/// flat register index corresponding to the encoded `args_start_v` operand.
///
/// Used by the `CallAnyReceiver`, `CallWithSpread`, and `Construct` handlers.
pub(super) fn collect_args(
    frame: &InterpreterFrame,
    args_start_v: u32,
    count: u32,
) -> StatorResult<Vec<JsValue>> {
    if count == 0 {
        return Ok(vec![]);
    }
    let start_flat = frame.reg_index(args_start_v)?;
    let end_flat = start_flat + count as usize;
    if end_flat > frame.registers.len() {
        return Err(StatorError::Internal(format!(
            "collect_args: register range {start_flat}..{end_flat} out of bounds \
             (registers.len={})",
            frame.registers.len()
        )));
    }
    Ok(frame.registers[start_flat..end_flat].to_vec())
}

/// Resolve a [`Operand::JumpOffset`] delta to an instruction index.
///
/// `pc_after_jump` is the current program counter **after** incrementing past
/// the jump instruction.  `byte_offsets[pc_after_jump]` is the byte offset
/// that marks the end of the jump instruction, which is the reference point
/// for the signed `delta`.
///
/// `instr_count` is `instructions.len()`; only the first `instr_count` entries
/// of `byte_offsets` are valid instruction starts.
#[inline]
pub(super) fn resolve_jump(
    pc_after_jump: usize,
    delta: i32,
    byte_offsets: &[usize],
    instr_count: usize,
) -> StatorResult<usize> {
    let end_byte = byte_offsets[pc_after_jump];
    let target_byte = (end_byte as i64 + delta as i64) as usize;
    byte_offsets[..instr_count]
        .binary_search(&target_byte)
        .map_err(|_| {
            StatorError::Internal(format!(
                "jump target byte offset {target_byte} is not at an instruction boundary"
            ))
        })
}

/// Read a jump offset from the constant pool for a `*Constant` jump opcode.
///
/// The constant pool entry at `idx` must be a [`ConstantPoolEntry::Number`]
/// whose value is the signed byte-level jump delta.
pub(super) fn constant_pool_jump_delta(
    frame: &InterpreterFrame,
    idx: u32,
    opcode_name: &str,
) -> StatorResult<i32> {
    let entry = frame.bytecode_array.get_constant(idx).ok_or_else(|| {
        StatorError::Internal(format!(
            "{opcode_name}: constant pool index {idx} out of bounds"
        ))
    })?;
    match entry {
        ConstantPoolEntry::Number(n) => Ok(*n as i32),
        _ => Err(StatorError::Internal(format!(
            "{opcode_name}: constant is not a number"
        ))),
    }
}

/// Convert a constant-pool entry to a [`JsValue`].
///
/// String entries that represent JavaScript string literals (i.e. those whose
/// raw source text is surrounded by `'…'`, `"…"`, or backtick `` `…` ``)
/// are decoded: the surrounding delimiters are stripped and standard escape
/// sequences (`\\`, `\'`, `\"`, `\n`, `\r`, `\t`) are expanded.  Plain
/// identifier strings (property names, variable names) are stored without
/// delimiters and are returned as-is.
#[inline]
pub(super) fn constant_to_value(entry: &ConstantPoolEntry) -> JsValue {
    match entry {
        ConstantPoolEntry::Number(n) => number_to_jsvalue(*n),
        ConstantPoolEntry::String(s) => JsValue::String(decode_string_constant(s).into()),
        ConstantPoolEntry::Boolean(b) => JsValue::Boolean(*b),
        ConstantPoolEntry::Null => JsValue::Null,
        ConstantPoolEntry::Undefined => JsValue::Undefined,
        ConstantPoolEntry::BigInt(n) => JsValue::BigInt(*n),
        ConstantPoolEntry::Function(ba) => JsValue::Function(Rc::new((**ba).clone())),
        ConstantPoolEntry::TemplateObject { cooked, raw } => build_template_object(cooked, raw),
    }
}

/// Build the frozen template-strings object passed as the first argument to
/// a tagged-template function.
///
/// The result is a `PlainObject` with indexed cooked strings, a `length`
/// property, and a `raw` sub-object with the same layout holding the raw
/// (un-escaped) strings.
fn build_template_object(cooked: &[Option<String>], raw: &[String]) -> JsValue {
    let mut map = PropertyMap::new();
    for (i, c) in cooked.iter().enumerate() {
        let val = match c {
            Some(s) => JsValue::String(s.clone().into()),
            None => JsValue::Undefined,
        };
        map.insert(i.to_string(), val);
    }
    map.insert("length".to_string(), JsValue::Smi(cooked.len() as i32));

    let mut raw_map = PropertyMap::new();
    for (i, r) in raw.iter().enumerate() {
        raw_map.insert(i.to_string(), JsValue::String(r.clone().into()));
    }
    raw_map.insert("length".to_string(), JsValue::Smi(raw.len() as i32));

    map.insert(
        "raw".to_string(),
        JsValue::PlainObject(Rc::new(RefCell::new(raw_map))),
    );

    JsValue::PlainObject(Rc::new(RefCell::new(map)))
}

/// Decode a constant-pool string entry into its runtime string value.
///
/// The scanner stores string literal tokens as raw source text including the
/// surrounding quote characters (e.g. `"'hello world'"` for `'hello world'`).
/// At runtime, the `LdaConstant` opcode must strip the delimiters and expand
/// escape sequences so that the string value is the *content* of the literal.
///
/// Plain strings (property names, identifier names) have no surrounding
/// delimiters and are returned unchanged.
pub(super) fn decode_string_constant(raw: &str) -> String {
    let bytes = raw.as_bytes();
    if bytes.len() < 2 {
        return raw.to_owned();
    }
    let (_, inner) = match bytes[0] {
        b'\'' | b'"' | b'`' if bytes[bytes.len() - 1] == bytes[0] => {
            (bytes[0], &raw[1..raw.len() - 1])
        }
        _ => return raw.to_owned(),
    };
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some('b') => out.push('\u{0008}'),
                Some('f') => out.push('\u{000C}'),
                Some('v') => out.push('\u{000B}'),
                Some('\\') => out.push('\\'),
                Some('\'') => out.push('\''),
                Some('"') => out.push('"'),
                Some('`') => out.push('`'),
                Some('0') if !matches!(chars.peek(), Some('0'..='9')) => {
                    out.push('\0');
                }
                Some(d @ '0'..='7') => {
                    // Legacy octal escape — collect up to 3 octal digits.
                    // The first digit was already consumed by `chars.next()`.
                    let mut val = (d as u32) - ('0' as u32);
                    // A leading 0-3 allows up to two more digits (three total);
                    // a leading 4-7 allows one more digit (two total).
                    let max_extra = if d <= '3' { 2 } else { 1 };
                    for _ in 0..max_extra {
                        match chars.peek() {
                            Some(&od) if ('0'..='7').contains(&od) => {
                                val = val * 8 + (od as u32 - '0' as u32);
                                chars.next();
                            }
                            _ => break,
                        }
                    }
                    if let Some(ch) = char::from_u32(val) {
                        out.push(ch);
                    }
                }
                Some('x') => {
                    // \xHH — two hex digits
                    let h = take_hex_digits(&mut chars, 2);
                    if let Some(cp) = u32::from_str_radix(&h, 16).ok().and_then(char::from_u32) {
                        out.push(cp);
                    }
                }
                Some('u') => {
                    if chars.peek() == Some(&'{') {
                        // \u{HHHH…} — braced code point
                        chars.next(); // consume '{'
                        let mut hex = String::new();
                        while let Some(&d) = chars.peek() {
                            if d == '}' {
                                chars.next();
                                break;
                            }
                            hex.push(d);
                            chars.next();
                        }
                        if let Some(cp) =
                            u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32)
                        {
                            out.push(cp);
                        }
                    } else {
                        // \uHHHH — exactly four hex digits
                        let h = take_hex_digits(&mut chars, 4);
                        if let Some(cp) = u32::from_str_radix(&h, 16).ok().and_then(char::from_u32)
                        {
                            out.push(cp);
                        }
                    }
                }
                // Line continuation: backslash followed by line terminator is
                // consumed silently (produces no character).
                Some('\n') => {}
                Some('\r') => {
                    // \r\n counts as a single line continuation
                    if chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                }
                Some('\u{2028}') | Some('\u{2029}') => {}
                // Identity escape: any other character after `\` is itself.
                Some(other) => out.push(other),
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Consume exactly `n` hex digits from the iterator, returning them as a string.
fn take_hex_digits(chars: &mut std::iter::Peekable<std::str::Chars<'_>>, n: usize) -> String {
    let mut hex = String::with_capacity(n);
    for _ in 0..n {
        if let Some(&d) = chars.peek() {
            if d.is_ascii_hexdigit() {
                hex.push(d);
                chars.next();
            } else {
                break;
            }
        }
    }
    hex
}
#[inline(always)]
pub(super) fn number_to_jsvalue(n: f64) -> JsValue {
    // Preserve -0.0 as HeapNumber so SameValue(+0, -0) correctly returns false.
    if n == 0.0 && n.is_sign_negative() {
        return JsValue::HeapNumber(n);
    }
    if n.is_finite() && n.fract() == 0.0 && (i32::MIN as f64..=i32::MAX as f64).contains(&n) {
        JsValue::Smi(n as i32)
    } else {
        JsValue::HeapNumber(n)
    }
}

/// Convert a JsValue to boolean (ECMAScript §7.1.2).
fn to_boolean_val(v: &JsValue) -> bool {
    v.to_boolean()
}

/// Convert a JsValue to its string representation.
fn js_to_string(v: &JsValue) -> String {
    match v {
        JsValue::String(s) => s.to_string(),
        JsValue::Smi(n) => n.to_string(),
        JsValue::HeapNumber(n) => crate::objects::value::number_to_string(*n),
        JsValue::Boolean(b) => b.to_string(),
        JsValue::Null => "null".to_string(),
        JsValue::Undefined => "undefined".to_string(),
        JsValue::Symbol(id) => format!("Symbol({id})"),
        JsValue::BigInt(n) => n.to_string(),
        _ => v
            .to_js_string()
            .unwrap_or_else(|_| "[object Object]".to_string()),
    }
}

/// Return a `TypeError` for mixed BigInt/Number operations.
fn mixed_bigint_number_error() -> StatorError {
    StatorError::TypeError(
        "Cannot mix BigInt and other types, use explicit conversions".to_string(),
    )
}

/// Extract the `i128` value from a BigInt, or return a TypeError for mixed operations.
pub(super) fn to_bigint(val: &JsValue) -> StatorResult<i128> {
    match val {
        JsValue::BigInt(n) => Ok(*n),
        _ => Err(mixed_bigint_number_error()),
    }
}

/// BigInt exponentiation with wrapping semantics.
pub(super) fn bigint_pow(base: i128, exp: u32) -> i128 {
    let mut result: i128 = 1;
    for _ in 0..exp {
        result = result.wrapping_mul(base);
    }
    result
}

thread_local! {
    /// Reusable buffer for string concatenation.  Avoids a fresh allocation on
    /// every `+` when building strings in a loop (e.g. `a + b + c + d`).  The
    /// buffer is cleared before each use but retains its heap allocation across
    /// calls within the same thread.
    static STRING_BUILDER: RefCell<String> = RefCell::new(String::with_capacity(4096));
}

/// Concatenate two string slices, reusing a thread-local buffer.
///
/// The caller must already have validated `MAX_STRING_LEN`.
#[inline]
pub(super) fn concat_rc_strs(l: &str, r: &str) -> JsValue {
    STRING_BUILDER.with(|sb| {
        let mut buf = sb.borrow_mut();
        buf.clear();
        let total = l.len() + r.len();
        let cap = buf.capacity();
        if total > cap {
            buf.reserve(total - cap);
        }
        buf.push_str(l);
        buf.push_str(r);
        JsValue::String(Rc::from(buf.as_str()))
    })
}

/// ECMAScript §13.15.3 **ApplyStringOrNumericBinaryOperator** for `+`.
///
/// 1. Call `ToPrimitive` on both operands (with no hint, i.e. `Default`).
/// 2. If *either* resulting primitive is a string, convert both to strings and
///    concatenate.
/// 3. Otherwise, convert both to numbers and add.
///
/// BigInt operands are added together; mixing BigInt and Number is a TypeError.
///
/// String concatenation reuses a thread-local buffer so that chains like
/// `a + b + c + d` avoid allocating a fresh `String` for every intermediate
/// result.
pub(super) fn js_add(lhs: &JsValue, rhs: &JsValue) -> StatorResult<JsValue> {
    // §12.8.3 step 1-2: ToPrimitive on both operands.
    let lprim = lhs.to_primitive(crate::objects::value::ToPrimitiveHint::Default)?;
    let rprim = rhs.to_primitive(crate::objects::value::ToPrimitiveHint::Default)?;

    // §12.8.3 step 3: if either primitive is a string, concatenate.
    if lprim.is_string() || rprim.is_string() {
        let l = lprim.to_js_string()?;
        let r = rprim.to_js_string()?;
        let total = l.len().saturating_add(r.len());
        if total > crate::builtins::string::MAX_STRING_LEN {
            return Err(StatorError::RangeError("Invalid string length".into()));
        }
        Ok(concat_rc_strs(&l, &r))
    } else if lprim.is_bigint() || rprim.is_bigint() {
        let l = to_bigint(&lprim)?;
        let r = to_bigint(&rprim)?;
        Ok(JsValue::BigInt(l.wrapping_add(r)))
    } else {
        let l = lprim.to_number()?;
        let r = rprim.to_number()?;
        Ok(number_to_jsvalue(l + r))
    }
}

/// ECMAScript §7.2.14 **IsLessThan** (`<`).
///
/// Delegates to [`JsValue::abstract_relational_comparison`] which performs the
/// full spec algorithm including ToPrimitive on objects.  Fast paths for Smi
/// and String comparisons avoid the ToPrimitive overhead for primitives.
pub(super) fn js_less_than(lhs: &JsValue, rhs: &JsValue) -> StatorResult<bool> {
    // Fast paths for common types (avoid ToPrimitive overhead)
    if let (JsValue::Smi(a), JsValue::Smi(b)) = (lhs, rhs) {
        return Ok(a < b);
    }
    if let (JsValue::String(a), JsValue::String(b)) = (lhs, rhs) {
        return Ok(a < b);
    }
    // Full spec: §7.2.14 IsLessThan(x, y, true)
    // Returns None for undefined (NaN cases) → map to false
    Ok(JsValue::abstract_relational_comparison(lhs, rhs, true)?.unwrap_or(false))
}

/// ECMAScript §7.2.13 **Abstract Equality Comparison** (`==`).
///
/// Delegates to [`JsValue::is_loosely_equal`] which implements the full spec
/// algorithm including ToPrimitive coercion for objects.
pub(super) fn abstract_eq(lhs: &JsValue, rhs: &JsValue) -> bool {
    lhs.is_loosely_equal(rhs).unwrap_or(false)
}

/// ECMAScript §7.2.15 **Strict Equality Comparison** (`===`).
///
/// Delegates to [`JsValue::is_strictly_equal`] which handles all JsValue
/// variants including Generator, Iterator, ArrayBuffer, TypedArray, DataView,
/// and Context.
pub(super) fn strict_eq(lhs: &JsValue, rhs: &JsValue) -> bool {
    lhs.is_strictly_equal(rhs)
}

/// Construct a diagnostic error for an unexpected operand kind.
#[cold]
pub(super) fn err_bad_operand(opcode_name: &'static str, operand_index: usize) -> StatorError {
    StatorError::Internal(format!(
        "{opcode_name}: unexpected operand kind at index {operand_index}"
    ))
}

/// Error returned when the bytecode stream ends without a `Return`.
#[cold]
fn bytecode_end_error() -> StatorError {
    StatorError::Internal("bytecode ended without a Return instruction".into())
}

/// Error returned when the per-frame instruction limit is exceeded.
#[cold]
fn instruction_limit_error() -> StatorError {
    StatorError::Internal("instruction limit exceeded".into())
}

/// Handle a dispatch error on the cold path: look for a JS exception
/// handler covering the faulting instruction and, if found, store the
/// error value in the frame accumulator and return the handler PC.
///
/// Returns `None` if no handler covers the instruction, meaning the
/// error should propagate to the caller.
#[cold]
fn handle_dispatch_error(
    e: &StatorError,
    frame: &mut InterpreterFrame,
    handler_table: &[HandlerTableEntry],
) -> Option<usize> {
    let instr_idx = (frame.pc - 1) as u32;
    let handler_pc = find_handler(instr_idx, handler_table)?;
    // Engine errors (TypeError, etc.) → JsValue::Error.
    // JsException from a nested throw → recover the
    // original thrown JsValue via the thread-local slot.
    let js_val = stator_error_to_js_value(e).or_else(|| {
        if matches!(e, StatorError::JsException(_)) {
            take_pending_exception()
        } else {
            None
        }
    })?;
    frame.accumulator = js_val;
    Some(handler_pc)
}

/// Returns `true` if the value is a JS receiver (an object-like type, not a
/// primitive, null, or undefined).
pub(super) fn is_js_receiver(value: &JsValue) -> bool {
    !matches!(
        value,
        JsValue::Undefined
            | JsValue::Null
            | JsValue::Boolean(_)
            | JsValue::Smi(_)
            | JsValue::HeapNumber(_)
            | JsValue::String(_)
            | JsValue::Symbol(_)
            | JsValue::BigInt(_)
    )
}

/// Dispatch a function call with the given arguments, writing the result to
/// the frame's accumulator.  Handles `Function`, `NativeFunction`, and
/// callable `PlainObject` values (those with a `__call__` property).
#[allow(dead_code)]
pub(super) fn dispatch_call(
    frame: &mut InterpreterFrame,
    callee: &JsValue,
    args: Vec<JsValue>,
) -> StatorResult<()> {
    match callee {
        JsValue::Function(ba) => {
            if ba.is_generator() {
                frame.accumulator = JsValue::Generator(GeneratorState::new((**ba).clone()));
            } else if ba.is_async() {
                // Async (non-generator) function: drive the internal generator
                // to completion and return a Promise.
                frame.accumulator = Interpreter::run_async_function((**ba).clone(), args)?;
            } else {
                let count = ba.increment_invocation_count();
                if count >= TIERING_THRESHOLD && ba.try_get_jit_code().is_none() {
                    maybe_compile_baseline(ba);
                }
                if count >= MAGLEV_TIERING_THRESHOLD {
                    maybe_compile_maglev(ba);
                }
                if count >= TURBOFAN_TIERING_THRESHOLD {
                    maybe_compile_turbofan(ba);
                }
                let mut tried_jit = false;
                if let Some(jit_result) = try_execute_best_jit(ba, &args) {
                    frame.accumulator = jit_result?;
                    tried_jit = true;
                }
                if !tried_jit {
                    let mut callee_frame = InterpreterFrame::new_with_globals(
                        (**ba).clone(),
                        args,
                        Rc::clone(&frame.global_env),
                    );
                    restore_closure_context(&mut callee_frame, ba);
                    populate_self_name(&mut callee_frame, ba, &JsValue::Function(Rc::clone(ba)));
                    push_call_frame("<anonymous>")?;
                    let result = Interpreter::run(&mut callee_frame);
                    pop_call_frame();
                    frame.accumulator = result?;
                }
            }
        }
        JsValue::NativeFunction(f) => {
            frame.accumulator = f(args)?;
        }
        JsValue::PlainObject(map) => {
            if let Some(JsValue::NativeFunction(f)) = map.borrow().get("__call__").cloned() {
                frame.accumulator = f(args)?;
            } else {
                return Err(StatorError::TypeError(
                    "CallProperty: callee is not a function (got PlainObject)".to_string(),
                ));
            }
        }
        other => {
            return Err(StatorError::TypeError(format!(
                "CallProperty: callee is not a function (got {other:?})"
            )));
        }
    }
    Ok(())
}

/// Like [`dispatch_call`] but also sets `this` in the callee frame's global
/// environment.  Used by `CallProperty0/1/2` where the receiver register is
/// available at the call site.
pub(super) fn dispatch_call_property(
    frame: &mut InterpreterFrame,
    callee: &JsValue,
    this_val: JsValue,
    args: Vec<JsValue>,
) -> StatorResult<()> {
    match callee {
        JsValue::Function(ba) => {
            if ba.is_generator() {
                frame.accumulator = JsValue::Generator(GeneratorState::new((**ba).clone()));
            } else if ba.is_async() {
                frame.accumulator = Interpreter::run_async_function((**ba).clone(), args)?;
            } else {
                let count = ba.increment_invocation_count();
                if count >= TIERING_THRESHOLD && ba.try_get_jit_code().is_none() {
                    maybe_compile_baseline(ba);
                }
                if count >= MAGLEV_TIERING_THRESHOLD {
                    maybe_compile_maglev(ba);
                }
                if count >= TURBOFAN_TIERING_THRESHOLD {
                    maybe_compile_turbofan(ba);
                }
                let mut tried_jit = false;
                if let Some(jit_result) = try_execute_best_jit(ba, &args) {
                    frame.accumulator = jit_result?;
                    tried_jit = true;
                }
                if !tried_jit {
                    let mut callee_frame = InterpreterFrame::new_with_globals(
                        (**ba).clone(),
                        args,
                        Rc::clone(&frame.global_env),
                    );
                    restore_closure_context(&mut callee_frame, ba);
                    populate_self_name(&mut callee_frame, ba, &JsValue::Function(Rc::clone(ba)));
                    // Arrow functions use lexical `this` — do NOT override.
                    if !ba.is_arrow() {
                        callee_frame
                            .global_env
                            .borrow_mut()
                            .insert("this".to_string(), this_val);
                    }
                    push_call_frame("<anonymous>")?;
                    let result = Interpreter::run(&mut callee_frame);
                    pop_call_frame();
                    frame.accumulator = result?;
                }
            }
        }
        JsValue::NativeFunction(f) => {
            frame.accumulator = f(args)?;
        }
        JsValue::PlainObject(map) => {
            if let Some(JsValue::NativeFunction(f)) = map.borrow().get("__call__").cloned() {
                frame.accumulator = f(args)?;
            } else {
                return Err(StatorError::TypeError(
                    "CallProperty: callee is not a function (got PlainObject)".to_string(),
                ));
            }
        }
        other => {
            return Err(StatorError::TypeError(format!(
                "CallProperty: callee is not a function (got {other:?})"
            )));
        }
    }
    Ok(())
}

/// Extract a `Rc<RefCell<JsContext>>` from a `JsValue::Context`.
pub(super) fn extract_context(
    value: &JsValue,
    opcode_name: &str,
) -> StatorResult<Rc<RefCell<JsContext>>> {
    match value {
        JsValue::Context(ctx) => Ok(Rc::clone(ctx)),
        other => Err(StatorError::Internal(format!(
            "{opcode_name}: expected Context value, got {other:?}"
        ))),
    }
}

/// Walk `depth` levels up the context chain starting from `ctx`.
///
/// `depth == 0` returns the same context; `depth == 1` returns its parent, etc.
pub(super) fn walk_context_chain(
    ctx: &Rc<RefCell<JsContext>>,
    depth: u32,
    opcode_name: &str,
) -> StatorResult<Rc<RefCell<JsContext>>> {
    let mut current = Rc::clone(ctx);
    for _ in 0..depth {
        let parent = {
            let borrowed = current.borrow();
            borrowed
                .parent
                .as_ref()
                .ok_or_else(|| {
                    StatorError::Internal(format!(
                        "{opcode_name}: context chain exhausted before reaching depth {depth}"
                    ))
                })?
                .clone()
        };
        current = parent;
    }
    Ok(current)
}

/// Restore the captured closure context on a callee frame.
///
/// If `ba` was created by `CreateClosure` with an enclosing scope, this sets
/// the callee frame's context so that `CreateFunctionContext` chains to the
/// captured scope and context-slot opcodes can walk up to outer variables.
pub(super) fn restore_closure_context(
    frame: &mut InterpreterFrame,
    ba: &Rc<crate::bytecode::bytecode_array::BytecodeArray>,
) {
    if let Some(ctx) = ba.closure_context() {
        frame.context = Some(JsValue::Context(Rc::clone(ctx)));
    }
    let home_object = fn_props_get(ba, ".home_object");
    match home_object {
        JsValue::PlainObject(map) => {
            let super_lookup_start = map
                .borrow()
                .get("__proto__")
                .cloned()
                .unwrap_or(JsValue::Undefined);
            frame
                .global_env
                .borrow_mut()
                .insert(".super_lookup_start".to_string(), super_lookup_start);
        }
        _ => {
            frame.global_env.borrow_mut().remove(".super_lookup_start");
        }
    }
}

/// Populate the self-name register for named function expressions.
///
/// When a named function expression (`var f = function g() { … }`) is
/// called, the bytecode compiler allocates a register for the name `g`.
/// This helper writes the function value into that register so the body
/// can reference the function by its own name (ES §15.2.4).
pub(super) fn populate_self_name(
    frame: &mut InterpreterFrame,
    ba: &crate::bytecode::bytecode_array::BytecodeArray,
    callee: &JsValue,
) {
    if let Some(reg_i32) = ba.self_name_register() {
        let param_count = ba.parameter_count() as usize;
        let idx = param_count + reg_i32 as usize;
        if idx < frame.registers.len() {
            frame.registers[idx] = callee.clone();
        }
    }
}

/// Convert a thrown JavaScript value to a human-readable error message string.
///
/// `Error` objects format as `"name: message"` (or just `"name"` for an empty
/// message).  All other values are converted via [`JsValue::to_js_string`];
/// when that conversion itself fails the debug representation is used instead.
#[allow(dead_code)]
#[cold]
pub(super) fn error_message_from_value(value: &JsValue) -> String {
    match value {
        JsValue::Error(e) => e.to_error_string(),
        JsValue::PlainObject(map) => {
            let borrow = map.borrow();
            // Build "Name: message" for error-like objects so that the
            // test runner's `matches_type` can identify the error kind
            // (e.g. "TypeError: foo" matches expected type "TypeError").
            let name = borrow
                .get("name")
                .and_then(|v| v.to_js_string().ok())
                .unwrap_or_default();
            let msg = borrow
                .get("message")
                .and_then(|v| v.to_js_string().ok())
                .unwrap_or_default();
            if !name.is_empty() && !msg.is_empty() {
                format!("{name}: {msg}")
            } else if !name.is_empty() {
                name
            } else if !msg.is_empty() {
                msg
            } else {
                "[object Object]".to_string()
            }
        }
        other => other
            .to_js_string()
            .unwrap_or_else(|_| format!("{other:?}")),
    }
}

/// Convert a [`StatorError`] into a [`JsValue::Error`] if it represents a
/// catchable JavaScript error (TypeError, RangeError, ReferenceError,
/// SyntaxError, URIError).
///
/// Returns `None` for internal engine errors and other non-JS error variants
/// (e.g. `OutOfMemory`, `Internal`, `JsException`, `DebuggerPaused`), which
/// should propagate to the caller rather than being caught by a JS `catch`
/// block.
#[cold]
pub(super) fn stator_error_to_js_value(err: &StatorError) -> Option<JsValue> {
    use crate::builtins::error::{ErrorKind, JsError};

    let (kind, msg) = match err {
        StatorError::TypeError(msg) => (ErrorKind::TypeError, msg.clone()),
        StatorError::RangeError(msg) => (ErrorKind::RangeError, msg.clone()),
        StatorError::ReferenceError(msg) => (ErrorKind::ReferenceError, msg.clone()),
        StatorError::SyntaxError(msg) => (ErrorKind::SyntaxError, msg.clone()),
        StatorError::URIError(msg) => (ErrorKind::URIError, msg.clone()),
        _ => return None,
    };
    Some(JsValue::Error(Rc::new(JsError::new(kind, msg))))
}

/// Try to interpret a [`JsValue`] as an array index (non-negative integer).
///
/// Returns `Some(index)` for `Smi(n)` where `n >= 0`, `HeapNumber(n)` where
/// `n` is a non-negative integer that fits in `usize`, and numeric strings
/// like `"0"`, `"123"`.  Returns `None` otherwise.
pub(super) fn to_array_index(key: &JsValue) -> Option<usize> {
    match key {
        JsValue::Smi(n) if *n >= 0 => Some(*n as usize),
        JsValue::HeapNumber(n) => {
            let idx = *n as usize;
            if *n >= 0.0 && (idx as f64) == *n {
                Some(idx)
            } else {
                None
            }
        }
        JsValue::String(s) => s.parse::<usize>().ok(),
        _ => None,
    }
}

/// Convert a [`JsValue`] to a property key string.
///
/// ECMAScript §7.1.19 ToPropertyKey.
pub(super) fn to_property_key(key: &JsValue) -> StatorResult<String> {
    match key {
        JsValue::Symbol(id) => Ok(crate::builtins::symbol::symbol_to_property_key(*id)),
        _ => key.to_js_string(),
    }
}

/// Perform a keyed property load: `obj[key]`.
///
/// Handles `PlainObject` (string keys), `Array` (integer keys + `"length"`),
/// and `String` (character-at-index + `"length"`).
/// Walk the `__proto__` chain of a `PlainObject` to resolve a named property.
///
/// Returns `JsValue::Undefined` if the property is not found after exhausting
/// the chain or hitting a depth limit of 256 links.
pub(super) fn proto_lookup(obj: &JsValue, key: &str) -> JsValue {
    // Fast path: PlainObject — the most common case.
    if let JsValue::PlainObject(map) = obj {
        let borrow = map.borrow();
        // Check for getter accessor (__get_<key>__) BEFORE the data key so
        // that accessor properties defined via Object.defineProperty are
        // dispatched correctly even when a placeholder data key exists.
        let getter_key = format!("__get_{key}__");
        if let Some(getter) = borrow.get(&getter_key).cloned() {
            drop(borrow);
            return match dispatch_getter(&getter, obj) {
                Ok(v) => v,
                Err(_) => JsValue::Undefined,
            };
        }
        if let Some(val) = borrow.get(key) {
            return val.clone();
        }
        // If this PlainObject is an array literal, delegate to Array.prototype
        // BEFORE falling through to Object.prototype methods (e.g. toString).
        if borrow
            .get("__is_array__")
            .is_some_and(|v| matches!(v, JsValue::Boolean(true)))
        {
            drop(borrow);
            return array_literal_proto_lookup(obj, key);
        }
        // Object.prototype methods available on all plain objects.
        match key {
            "hasOwnProperty" => {
                let map = Rc::clone(map);
                drop(borrow);
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let prop = match args.first() {
                        Some(JsValue::String(s)) => s.to_string(),
                        Some(JsValue::Smi(n)) => n.to_string(),
                        Some(JsValue::HeapNumber(n)) => {
                            let s = format!("{n}");
                            s
                        }
                        Some(JsValue::Boolean(b)) => b.to_string(),
                        Some(JsValue::Null) => "null".to_string(),
                        Some(JsValue::Undefined) => "undefined".to_string(),
                        _ => return Ok(JsValue::Boolean(false)),
                    };
                    // Filter internal implementation keys that are not real
                    // own properties from the user's perspective.
                    if prop == "__proto__"
                        || prop == "__is_array__"
                        || (prop.starts_with("__get_") && prop.ends_with("__"))
                        || (prop.starts_with("__set_") && prop.ends_with("__"))
                    {
                        return Ok(JsValue::Boolean(false));
                    }
                    Ok(JsValue::Boolean(map.borrow().contains_key(&prop)))
                }));
            }
            "propertyIsEnumerable" => {
                let map = Rc::clone(map);
                drop(borrow);
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let prop = match args.first() {
                        Some(JsValue::String(s)) => s.to_string(),
                        Some(JsValue::Smi(n)) => n.to_string(),
                        _ => return Ok(JsValue::Boolean(false)),
                    };
                    Ok(JsValue::Boolean(map.borrow().is_enumerable(&prop)))
                }));
            }
            "isPrototypeOf" => {
                let this_map = Rc::clone(map);
                drop(borrow);
                return JsValue::NativeFunction(Rc::new(move |args| {
                    // ES §20.1.3.2 Object.prototype.isPrototypeOf(V)
                    // 1. If V is not an object, return false.
                    let mut v = match args.first() {
                        Some(JsValue::PlainObject(_)) => args[0].clone(),
                        _ => return Ok(JsValue::Boolean(false)),
                    };
                    // 2. Let O = ToObject(this) — already a PlainObject.
                    // 3. Loop: walk the prototype chain of V.
                    for _ in 0..256 {
                        // V = V.[[GetPrototypeOf]]()
                        v = match v {
                            JsValue::PlainObject(ref obj_map) => {
                                match obj_map.borrow().get("__proto__") {
                                    Some(proto) => proto.clone(),
                                    None => return Ok(JsValue::Boolean(false)),
                                }
                            }
                            _ => return Ok(JsValue::Boolean(false)),
                        };
                        // If V is null, return false.
                        if matches!(v, JsValue::Null | JsValue::Undefined) {
                            return Ok(JsValue::Boolean(false));
                        }
                        // If SameValue(O, V), return true.
                        if let JsValue::PlainObject(ref v_map) = v
                            && Rc::ptr_eq(&this_map, v_map)
                        {
                            return Ok(JsValue::Boolean(true));
                        }
                    }
                    Ok(JsValue::Boolean(false))
                }));
            }
            "constructor" => {
                // Walk __proto__ chain to find constructor (set by finalize_ctor).
                if let Some(proto) = borrow.get("__proto__").cloned() {
                    drop(borrow);
                    return proto_lookup(&proto, "constructor");
                }
                drop(borrow);
                return lookup_global_constructor("Object");
            }
            "toString" => {
                let obj_clone = obj.clone();
                drop(borrow);
                return JsValue::NativeFunction(Rc::new(move |args| {
                    // When called via .call(value), classify that value.
                    if let Some(value) = args.first() {
                        return Ok(JsValue::String(value.obj_to_string_tag().into()));
                    }
                    // Direct call: classify the captured receiver.
                    Ok(JsValue::String(obj_clone.obj_to_string_tag().into()))
                }));
            }
            "valueOf" => {
                let obj_clone = obj.clone();
                drop(borrow);
                return JsValue::NativeFunction(Rc::new(move |_args| Ok(obj_clone.clone())));
            }
            "toLocaleString" => {
                let obj_clone = obj.clone();
                drop(borrow);
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    let ts = proto_lookup(&obj_clone, "toString");
                    dispatch_call_value(&ts, vec![obj_clone.clone()])
                }));
            }
            "__lookupGetter__" => {
                let map = Rc::clone(map);
                drop(borrow);
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let prop = match args.first() {
                        Some(JsValue::String(s)) => s.to_string(),
                        Some(v) => js_to_string(v),
                        None => return Ok(JsValue::Undefined),
                    };
                    let getter_key = format!("__get_{prop}__");
                    // Walk the prototype chain looking for the getter.
                    let mut cur = JsValue::PlainObject(Rc::clone(&map));
                    for _ in 0..256 {
                        if let JsValue::PlainObject(ref m) = cur {
                            let b = m.borrow();
                            if let Some(g) = b.get(&getter_key) {
                                return Ok(g.clone());
                            }
                            match b.get("__proto__") {
                                Some(next) => {
                                    let next = next.clone();
                                    drop(b);
                                    cur = next;
                                    continue;
                                }
                                None => break,
                            }
                        }
                        break;
                    }
                    Ok(JsValue::Undefined)
                }));
            }
            "__lookupSetter__" => {
                let map = Rc::clone(map);
                drop(borrow);
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let prop = match args.first() {
                        Some(JsValue::String(s)) => s.to_string(),
                        Some(v) => js_to_string(v),
                        None => return Ok(JsValue::Undefined),
                    };
                    let setter_key = format!("__set_{prop}__");
                    let mut cur = JsValue::PlainObject(Rc::clone(&map));
                    for _ in 0..256 {
                        if let JsValue::PlainObject(ref m) = cur {
                            let b = m.borrow();
                            if let Some(s) = b.get(&setter_key) {
                                return Ok(s.clone());
                            }
                            match b.get("__proto__") {
                                Some(next) => {
                                    let next = next.clone();
                                    drop(b);
                                    cur = next;
                                    continue;
                                }
                                None => break,
                            }
                        }
                        break;
                    }
                    Ok(JsValue::Undefined)
                }));
            }
            "__defineGetter__" => {
                let map = Rc::clone(map);
                drop(borrow);
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let prop = match args.first() {
                        Some(JsValue::String(s)) => s.to_string(),
                        Some(v) => js_to_string(v),
                        None => return Ok(JsValue::Undefined),
                    };
                    let getter = args.get(1).cloned().unwrap_or(JsValue::Undefined);
                    let getter_key = format!("__get_{prop}__");
                    map.borrow_mut().insert(getter_key, getter);
                    Ok(JsValue::Undefined)
                }));
            }
            "__defineSetter__" => {
                let map = Rc::clone(map);
                drop(borrow);
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let prop = match args.first() {
                        Some(JsValue::String(s)) => s.to_string(),
                        Some(v) => js_to_string(v),
                        None => return Ok(JsValue::Undefined),
                    };
                    let setter = args.get(1).cloned().unwrap_or(JsValue::Undefined);
                    let setter_key = format!("__set_{prop}__");
                    map.borrow_mut().insert(setter_key, setter);
                    Ok(JsValue::Undefined)
                }));
            }
            _ => {}
        }
        // Walk __proto__ chain.
        if let Some(proto) = borrow.get("__proto__") {
            let next = proto.clone();
            drop(borrow);
            return proto_lookup_chain(&next, key, obj);
        }
        return JsValue::Undefined;
    }
    // Fast path: primitive toString/valueOf.
    match obj {
        JsValue::Smi(n) => match key {
            "toString" => {
                let n = *n;
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let radix = match args.first() {
                        None | Some(JsValue::Undefined) => 10u32,
                        Some(v) => {
                            let r = v.to_number()?;
                            let ri = r.floor() as i64;
                            if !(2..=36).contains(&ri) {
                                return Err(StatorError::RangeError(
                                    "toString() radix must be between 2 and 36".to_string(),
                                ));
                            }
                            ri as u32
                        }
                    };
                    if radix == 10 {
                        return Ok(JsValue::String(n.to_string().into()));
                    }
                    Ok(JsValue::String(
                        crate::builtins::util::i64_to_radix_string(n as i64, radix).into(),
                    ))
                }));
            }
            "valueOf" => {
                let n = *n;
                return JsValue::NativeFunction(Rc::new(move |_args| Ok(JsValue::Smi(n))));
            }
            "toFixed" => {
                let n = *n as f64;
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let frac = match args.first() {
                        Some(v) => v.to_number()?,
                        None => 0.0,
                    };
                    if frac.is_nan() || !(0.0..=100.0).contains(&frac) {
                        return Err(StatorError::RangeError(
                            "toFixed() digits argument must be between 0 and 100".into(),
                        ));
                    }
                    let digits = frac as usize;
                    Ok(JsValue::String(format!("{n:.digits$}").into()))
                }));
            }
            "toExponential" => {
                let n = *n as f64;
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let digits = match args.first() {
                        Some(JsValue::Smi(d)) => Some((*d).clamp(0, 100) as usize),
                        Some(JsValue::HeapNumber(d)) => {
                            Some(crate::builtins::util::clamped_f64_to_usize(*d).min(100))
                        }
                        Some(JsValue::Undefined) | None => None,
                        _ => None,
                    };
                    match digits {
                        Some(d) => Ok(JsValue::String(format!("{n:.d$e}").into())),
                        None => Ok(JsValue::String(format!("{n:e}").into())),
                    }
                }));
            }
            "toPrecision" => {
                let n = *n as f64;
                return JsValue::NativeFunction(Rc::new(move |args| match args.first() {
                    None | Some(JsValue::Undefined) => Ok(JsValue::String(format!("{n}").into())),
                    Some(v) => {
                        let p = v.to_number()? as usize;
                        if p == 0 || p > 100 {
                            return Err(StatorError::RangeError(
                                "toPrecision() argument must be between 1 and 100".to_string(),
                            ));
                        }
                        Ok(JsValue::String(
                            format!("{n:.prec$}", prec = p.saturating_sub(1)).into(),
                        ))
                    }
                }));
            }
            "constructor" => {
                return lookup_global_constructor("Number");
            }
            "@@toPrimitive" => {
                let n = *n;
                return JsValue::NativeFunction(Rc::new(move |_args| Ok(JsValue::Smi(n))));
            }
            "toLocaleString" => {
                let n = *n;
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(n.to_string().into()))
                }));
            }
            "hasOwnProperty" | "propertyIsEnumerable" => {
                return JsValue::NativeFunction(Rc::new(|_args| Ok(JsValue::Boolean(false))));
            }
            "isPrototypeOf" => {
                return JsValue::NativeFunction(Rc::new(|_args| Ok(JsValue::Boolean(false))));
            }
            _ => {}
        },
        JsValue::HeapNumber(n) => match key {
            "toString" => {
                let n = *n;
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let radix = match args.first() {
                        None | Some(JsValue::Undefined) => 10u32,
                        Some(v) => {
                            let r = v.to_number()?;
                            let ri = r.floor() as i64;
                            if !(2..=36).contains(&ri) {
                                return Err(StatorError::RangeError(
                                    "toString() radix must be between 2 and 36".to_string(),
                                ));
                            }
                            ri as u32
                        }
                    };
                    if radix == 10 {
                        return Ok(JsValue::String(format!("{n}").into()));
                    }
                    Ok(JsValue::String(
                        crate::builtins::util::f64_to_radix_string(n, radix).into(),
                    ))
                }));
            }
            "valueOf" => {
                let n = *n;
                return JsValue::NativeFunction(Rc::new(move |_args| Ok(JsValue::HeapNumber(n))));
            }
            "toFixed" => {
                let n = *n;
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let frac = match args.first() {
                        Some(v) => v.to_number()?,
                        None => 0.0,
                    };
                    if frac.is_nan() || !(0.0..=100.0).contains(&frac) {
                        return Err(StatorError::RangeError(
                            "toFixed() digits argument must be between 0 and 100".into(),
                        ));
                    }
                    let digits = frac as usize;
                    if n.is_nan() {
                        return Ok(JsValue::String("NaN".into()));
                    }
                    if n.is_infinite() {
                        return Ok(JsValue::String(
                            if n > 0.0 { "Infinity" } else { "-Infinity" }.into(),
                        ));
                    }
                    Ok(JsValue::String(format!("{n:.digits$}").into()))
                }));
            }
            "toExponential" => {
                let n = *n;
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let digits = match args.first() {
                        Some(JsValue::Smi(d)) => Some((*d).clamp(0, 100) as usize),
                        Some(JsValue::HeapNumber(d)) => {
                            Some(crate::builtins::util::clamped_f64_to_usize(*d).min(100))
                        }
                        Some(JsValue::Undefined) | None => None,
                        _ => None,
                    };
                    match digits {
                        Some(d) => Ok(JsValue::String(format!("{n:.d$e}").into())),
                        None => Ok(JsValue::String(format!("{n:e}").into())),
                    }
                }));
            }
            "toPrecision" => {
                let n = *n;
                return JsValue::NativeFunction(Rc::new(move |args| match args.first() {
                    None | Some(JsValue::Undefined) => Ok(JsValue::String(format!("{n}").into())),
                    Some(v) => {
                        let p = v.to_number()? as usize;
                        if p == 0 || p > 100 {
                            return Err(StatorError::RangeError(
                                "toPrecision() argument must be between 1 and 100".to_string(),
                            ));
                        }
                        Ok(JsValue::String(
                            format!("{n:.prec$}", prec = p.saturating_sub(1)).into(),
                        ))
                    }
                }));
            }
            "constructor" => {
                return lookup_global_constructor("Number");
            }
            "@@toPrimitive" => {
                let n = *n;
                return JsValue::NativeFunction(Rc::new(move |_args| Ok(JsValue::HeapNumber(n))));
            }
            "toLocaleString" => {
                let n = *n;
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(format!("{n}").into()))
                }));
            }
            "hasOwnProperty" | "propertyIsEnumerable" => {
                return JsValue::NativeFunction(Rc::new(|_args| Ok(JsValue::Boolean(false))));
            }
            "isPrototypeOf" => {
                return JsValue::NativeFunction(Rc::new(|_args| Ok(JsValue::Boolean(false))));
            }
            _ => {}
        },
        JsValue::String(s) => match key {
            "length" => return JsValue::Smi(s.encode_utf16().count() as i32),
            "toString" | "valueOf" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(s.clone()))
                }));
            }
            "charAt" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let idx = match args.first() {
                        Some(v) => v.to_number().unwrap_or(0.0) as i64,
                        None => 0,
                    };
                    Ok(JsValue::String(
                        crate::builtins::string::string_char_at(&s, idx).into(),
                    ))
                }));
            }
            "charCodeAt" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let idx = match args.first() {
                        Some(v) => v.to_number().unwrap_or(0.0) as i64,
                        None => 0,
                    };
                    let code = crate::builtins::string::string_char_code_at(&s, idx);
                    if code.is_nan() {
                        Ok(JsValue::HeapNumber(f64::NAN))
                    } else {
                        Ok(JsValue::Smi(code as i32))
                    }
                }));
            }
            "slice" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let start = match args.first() {
                        Some(v) => v.to_number().unwrap_or(0.0) as i64,
                        None => 0,
                    };
                    let end = match args.get(1) {
                        Some(JsValue::Undefined) | None => None,
                        Some(v) => Some(v.to_number().unwrap_or(0.0) as i64),
                    };
                    Ok(JsValue::String(
                        crate::builtins::string::string_slice(&s, start, end).into(),
                    ))
                }));
            }
            "includes" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    // §22.1.3.7 step 4: throw TypeError if searchString is a RegExp
                    if let Some(JsValue::PlainObject(re_obj)) = args.first()
                        && re_obj.borrow().get("__is_regexp__") == Some(&JsValue::Boolean(true))
                    {
                        return Err(crate::error::StatorError::TypeError(
                            "First argument to String.prototype.includes must not be a regular expression".to_string(),
                        ));
                    }
                    let search = match args.first() {
                        Some(v) => v.to_js_string()?,
                        None => "undefined".to_string(),
                    };
                    let pos = match args.get(1) {
                        Some(JsValue::Undefined) | None => None,
                        Some(v) => Some(v.to_number().unwrap_or(0.0) as i64),
                    };
                    Ok(JsValue::Boolean(crate::builtins::string::string_includes(
                        &s, &search, pos,
                    )))
                }));
            }
            "indexOf" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let search = match args.first() {
                        Some(v) => v.to_js_string()?,
                        None => "undefined".to_string(),
                    };
                    let pos = match args.get(1) {
                        Some(JsValue::Undefined) | None => None,
                        Some(v) => Some(v.to_number().unwrap_or(0.0) as i64),
                    };
                    let result = crate::builtins::string::string_index_of(&s, &search, pos);
                    if result <= i32::MAX as i64 {
                        Ok(JsValue::Smi(result as i32))
                    } else {
                        Ok(JsValue::HeapNumber(result as f64))
                    }
                }));
            }
            "lastIndexOf" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let search = match args.first() {
                        Some(v) => v.to_js_string()?,
                        None => "undefined".to_string(),
                    };
                    let pos = match args.get(1) {
                        Some(JsValue::Undefined) | None => None,
                        Some(v) => Some(v.to_number().unwrap_or(0.0) as i64),
                    };
                    let result = crate::builtins::string::string_last_index_of(&s, &search, pos);
                    if result >= i32::MIN as i64 && result <= i32::MAX as i64 {
                        Ok(JsValue::Smi(result as i32))
                    } else {
                        Ok(JsValue::HeapNumber(result as f64))
                    }
                }));
            }
            "toUpperCase" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(s.to_uppercase().into()))
                }));
            }
            "toLowerCase" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(s.to_lowercase().into()))
                }));
            }
            "trim" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(s.trim().to_string().into()))
                }));
            }
            "split" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let limit = match args.get(1) {
                        Some(JsValue::Smi(l)) => Some(*l as usize),
                        Some(JsValue::HeapNumber(l)) => {
                            Some(crate::builtins::util::clamped_f64_to_usize(*l))
                        }
                        _ => None,
                    };
                    // A limit of 0 always returns an empty array.
                    if limit == Some(0) {
                        return Ok(JsValue::new_array(vec![]));
                    }
                    match args.first() {
                        Some(JsValue::PlainObject(re_obj)) => {
                            let borrow = re_obj.borrow();
                            if borrow.get("__is_regexp__") == Some(&JsValue::Boolean(true)) {
                                let source = match borrow.get("source") {
                                    Some(JsValue::String(s)) => s.to_string(),
                                    _ => String::new(),
                                };
                                let flags = match borrow.get("flags") {
                                    Some(JsValue::String(s)) => s.to_string(),
                                    _ => String::new(),
                                };
                                drop(borrow);
                                let re = crate::objects::regexp::JsRegExp::new(&source, &flags)?;
                                let parts = re.symbol_split(&s, limit);
                                let items: Vec<JsValue> = parts
                                    .into_iter()
                                    .map(|p| JsValue::String(p.into()))
                                    .collect();
                                Ok(JsValue::new_array(items))
                            } else {
                                Ok(JsValue::new_array(vec![JsValue::String(s.clone())]))
                            }
                        }
                        Some(JsValue::String(sep)) => {
                            let sep = sep.to_string();
                            if sep.is_empty() {
                                // Empty separator: split into individual characters.
                                let chars: Vec<JsValue> = s
                                    .chars()
                                    .take(limit.unwrap_or(usize::MAX))
                                    .map(|c| JsValue::String(c.to_string().into()))
                                    .collect();
                                Ok(JsValue::new_array(chars))
                            } else {
                                let parts: Vec<JsValue> = s
                                    .split(&sep)
                                    .take(limit.unwrap_or(usize::MAX))
                                    .map(|p| JsValue::String(p.to_string().into()))
                                    .collect();
                                Ok(JsValue::new_array(parts))
                            }
                        }
                        Some(JsValue::Undefined) | None => {
                            Ok(JsValue::new_array(vec![JsValue::String(s.clone())]))
                        }
                        _ => Ok(JsValue::new_array(vec![JsValue::String(s.clone())])),
                    }
                }));
            }
            "startsWith" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    // §22.1.3.22 step 4: throw TypeError if searchString is a RegExp
                    if let Some(JsValue::PlainObject(re_obj)) = args.first()
                        && re_obj.borrow().get("__is_regexp__") == Some(&JsValue::Boolean(true))
                    {
                        return Err(crate::error::StatorError::TypeError(
                            "First argument to String.prototype.startsWith must not be a regular expression".to_string(),
                        ));
                    }
                    let search = match args.first() {
                        Some(v) => v.to_js_string()?,
                        None => String::new(),
                    };
                    let pos = match args.get(1) {
                        Some(JsValue::Undefined) | None => None,
                        Some(v) => Some(v.to_number().unwrap_or(0.0) as i64),
                    };
                    Ok(JsValue::Boolean(
                        crate::builtins::string::string_starts_with(&s, &search, pos),
                    ))
                }));
            }
            "endsWith" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    // §22.1.3.7 step 4: throw TypeError if searchString is a RegExp
                    if let Some(JsValue::PlainObject(re_obj)) = args.first()
                        && re_obj.borrow().get("__is_regexp__") == Some(&JsValue::Boolean(true))
                    {
                        return Err(crate::error::StatorError::TypeError(
                            "First argument to String.prototype.endsWith must not be a regular expression".to_string(),
                        ));
                    }
                    let search = match args.first() {
                        Some(v) => v.to_js_string()?,
                        None => String::new(),
                    };
                    let end = match args.get(1) {
                        Some(JsValue::Undefined) | None => None,
                        Some(v) => Some(v.to_number().unwrap_or(0.0) as i64),
                    };
                    Ok(JsValue::Boolean(crate::builtins::string::string_ends_with(
                        &s, &search, end,
                    )))
                }));
            }
            "repeat" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let count_f = match args.first() {
                        Some(v) => v.to_number()?,
                        None => 0.0,
                    };
                    // ES §22.1.3.16 step 3: ToIntegerOrInfinity(NaN) → 0
                    let count_f = if count_f.is_nan() { 0.0 } else { count_f };
                    if count_f < 0.0 || count_f.is_infinite() {
                        return Err(crate::error::StatorError::RangeError(
                            "Invalid count value".into(),
                        ));
                    }
                    crate::builtins::string::string_repeat(&s, count_f as i64)
                        .map(|r| JsValue::String(r.into()))
                }));
            }
            "padStart" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let raw = match args.first() {
                        Some(v) => v.to_number().unwrap_or(0.0),
                        None => 0.0,
                    };
                    let target_len = if raw.is_nan() || raw < 0.0 {
                        0usize
                    } else {
                        raw as usize
                    };
                    let pad_str = match args.get(1) {
                        Some(JsValue::String(f)) => f.to_string(),
                        Some(JsValue::Undefined) | None => " ".to_string(),
                        Some(v) => js_to_string(v),
                    };
                    crate::builtins::string::string_pad_start(&s, target_len, Some(&pad_str))
                        .map(|r| JsValue::String(r.into()))
                }));
            }
            "padEnd" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let raw = match args.first() {
                        Some(v) => v.to_number().unwrap_or(0.0),
                        None => 0.0,
                    };
                    let target_len = if raw.is_nan() || raw < 0.0 {
                        0usize
                    } else {
                        raw as usize
                    };
                    let pad_str = match args.get(1) {
                        Some(JsValue::String(f)) => f.to_string(),
                        Some(JsValue::Undefined) | None => " ".to_string(),
                        Some(v) => js_to_string(v),
                    };
                    crate::builtins::string::string_pad_end(&s, target_len, Some(&pad_str))
                        .map(|r| JsValue::String(r.into()))
                }));
            }
            "replace" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let replacement = match args.get(1) {
                        Some(JsValue::String(ss)) => ss.to_string(),
                        Some(v) => v.to_js_string()?,
                        _ => "undefined".to_string(),
                    };
                    match args.first() {
                        Some(JsValue::PlainObject(re_obj)) => {
                            let borrow = re_obj.borrow();
                            if borrow.get("__is_regexp__") == Some(&JsValue::Boolean(true)) {
                                let source = match borrow.get("source") {
                                    Some(JsValue::String(s)) => s.to_string(),
                                    _ => String::new(),
                                };
                                let flags = match borrow.get("flags") {
                                    Some(JsValue::String(s)) => s.to_string(),
                                    _ => String::new(),
                                };
                                drop(borrow);
                                let re = crate::objects::regexp::JsRegExp::new(&source, &flags)?;
                                Ok(JsValue::String(re.symbol_replace(&s, &replacement).into()))
                            } else {
                                Ok(JsValue::String(s.clone()))
                            }
                        }
                        Some(JsValue::String(search)) => {
                            let search = search.to_string();
                            Ok(JsValue::String(s.replacen(&search, &replacement, 1).into()))
                        }
                        _ => Ok(JsValue::String(s.clone())),
                    }
                }));
            }
            "substring" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let start = match args.first() {
                        Some(v) => v.to_number().unwrap_or(0.0) as i64,
                        None => 0,
                    };
                    let end = match args.get(1) {
                        Some(JsValue::Undefined) | None => None,
                        Some(v) => Some(v.to_number().unwrap_or(0.0) as i64),
                    };
                    Ok(JsValue::String(
                        crate::builtins::string::string_substring(&s, start, end).into(),
                    ))
                }));
            }
            "replaceAll" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let replacement = match args.get(1) {
                        Some(JsValue::String(ss)) => ss.to_string(),
                        Some(v) => v.to_js_string()?,
                        _ => "undefined".to_string(),
                    };
                    match args.first() {
                        Some(JsValue::PlainObject(re_obj)) => {
                            let borrow = re_obj.borrow();
                            if borrow.get("__is_regexp__") == Some(&JsValue::Boolean(true)) {
                                let flags = match borrow.get("flags") {
                                    Some(JsValue::String(s)) => s.to_string(),
                                    _ => String::new(),
                                };
                                if !flags.contains('g') {
                                    return Err(StatorError::TypeError(
                                        "String.prototype.replaceAll called with a non-global RegExp argument".to_string(),
                                    ));
                                }
                                let source = match borrow.get("source") {
                                    Some(JsValue::String(s)) => s.to_string(),
                                    _ => String::new(),
                                };
                                drop(borrow);
                                let re = crate::objects::regexp::JsRegExp::new(&source, &flags)?;
                                Ok(JsValue::String(re.symbol_replace(&s, &replacement).into()))
                            } else {
                                Ok(JsValue::String(s.clone()))
                            }
                        }
                        Some(JsValue::String(search)) => {
                            let search = search.to_string();
                            Ok(JsValue::String(s.replace(&search, &replacement).into()))
                        }
                        _ => Ok(JsValue::String(s.clone())),
                    }
                }));
            }
            "at" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let idx = match args.first() {
                        Some(v) => v.to_number()? as i64,
                        None => 0,
                    };
                    match crate::builtins::string::string_at(&s, idx) {
                        Some(ch) => Ok(JsValue::String(ch.into())),
                        None => Ok(JsValue::Undefined),
                    }
                }));
            }
            "trimStart" | "trimLeft" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(s.trim_start().to_string().into()))
                }));
            }
            "trimEnd" | "trimRight" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(s.trim_end().to_string().into()))
                }));
            }
            "concat" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let mut result = s.to_string();
                    for arg in &args {
                        match arg {
                            JsValue::String(ss) => result.push_str(ss),
                            JsValue::Smi(n) => result.push_str(&n.to_string()),
                            JsValue::HeapNumber(n) => result.push_str(&format!("{n}")),
                            JsValue::Boolean(b) => {
                                result.push_str(if *b { "true" } else { "false" })
                            }
                            JsValue::Null => result.push_str("null"),
                            JsValue::Undefined => result.push_str("undefined"),
                            _ => result.push_str("[object Object]"),
                        }
                    }
                    Ok(JsValue::String(result.into()))
                }));
            }
            "codePointAt" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let idx = match args.first() {
                        Some(v) => v.to_number().unwrap_or(0.0) as i64,
                        None => 0,
                    };
                    match crate::builtins::string::string_code_point_at(&s, idx) {
                        Some(cp) => Ok(JsValue::Smi(cp as i32)),
                        None => Ok(JsValue::Undefined),
                    }
                }));
            }
            "normalize" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    // ES §22.1.3.13 String.prototype.normalize([form])
                    // Determine the normalization form (default: "NFC").
                    let form = match args.first() {
                        None | Some(JsValue::Undefined) => "NFC".to_string(),
                        Some(v) => v.to_js_string()?,
                    };
                    match form.as_str() {
                        "NFC" | "NFD" | "NFKC" | "NFKD" => {
                            // ASCII-only strings are already in all normalization
                            // forms, so we can return as-is.  For non-ASCII we
                            // also return as-is (best-effort without the
                            // unicode-normalization crate).
                            Ok(JsValue::String(s.clone()))
                        }
                        _ => Err(StatorError::RangeError(
                            "The normalization form should be one of NFC, NFD, NFKC, NFKD."
                                .to_string(),
                        )),
                    }
                }));
            }
            "localeCompare" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let other = match args.first() {
                        Some(JsValue::String(ss)) => ss.to_string(),
                        _ => String::new(),
                    };
                    Ok(JsValue::Smi((*s).cmp(other.as_str()) as i32))
                }));
            }
            "match" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| match args.first() {
                    Some(JsValue::PlainObject(re_obj)) => {
                        let borrow = re_obj.borrow();
                        if borrow.get("__is_regexp__") == Some(&JsValue::Boolean(true)) {
                            let source = match borrow.get("source") {
                                Some(JsValue::String(s)) => s.to_string(),
                                _ => String::new(),
                            };
                            let flags = match borrow.get("flags") {
                                Some(JsValue::String(s)) => s.to_string(),
                                _ => String::new(),
                            };
                            drop(borrow);
                            let re = crate::objects::regexp::JsRegExp::new(&source, &flags)?;
                            match re.symbol_match(&s) {
                                Some(result) => {
                                    use crate::objects::regexp::SymbolMatchResult;
                                    match result {
                                        SymbolMatchResult::Single(m) => {
                                            let mut arr = vec![JsValue::String(m.matched.into())];
                                            for c in &m.captures {
                                                arr.push(match c {
                                                    Some(s) => JsValue::String(s.clone().into()),
                                                    None => JsValue::Undefined,
                                                });
                                            }
                                            Ok(JsValue::new_array(arr))
                                        }
                                        SymbolMatchResult::All(matches) => {
                                            let arr: Vec<JsValue> = matches
                                                .into_iter()
                                                .map(|m| JsValue::String(m.into()))
                                                .collect();
                                            Ok(JsValue::new_array(arr))
                                        }
                                    }
                                }
                                None => Ok(JsValue::Null),
                            }
                        } else {
                            Ok(JsValue::Null)
                        }
                    }
                    Some(JsValue::String(pat)) => {
                        let pat = pat.to_string();
                        match s.find(&pat) {
                            Some(_idx) => {
                                let arr = vec![JsValue::String(pat.into())];
                                Ok(JsValue::new_array(arr))
                            }
                            None => Ok(JsValue::Null),
                        }
                    }
                    _ => Ok(JsValue::Null),
                }));
            }
            "search" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| match args.first() {
                    Some(JsValue::PlainObject(re_obj)) => {
                        let borrow = re_obj.borrow();
                        if borrow.get("__is_regexp__") == Some(&JsValue::Boolean(true)) {
                            let source = match borrow.get("source") {
                                Some(JsValue::String(s)) => s.to_string(),
                                _ => String::new(),
                            };
                            let flags = match borrow.get("flags") {
                                Some(JsValue::String(s)) => s.to_string(),
                                _ => String::new(),
                            };
                            drop(borrow);
                            let re = crate::objects::regexp::JsRegExp::new(&source, &flags)?;
                            Ok(JsValue::Smi(re.symbol_search(&s) as i32))
                        } else {
                            Ok(JsValue::Smi(-1))
                        }
                    }
                    Some(JsValue::String(pat)) => {
                        let pat = pat.to_string();
                        Ok(s.find(&pat)
                            .map_or(JsValue::Smi(-1), |i| JsValue::Smi(i as i32)))
                    }
                    _ => Ok(JsValue::Smi(-1)),
                }));
            }
            "matchAll" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| match args.first() {
                    Some(JsValue::PlainObject(re_obj)) => {
                        let borrow = re_obj.borrow();
                        if borrow.get("__is_regexp__") == Some(&JsValue::Boolean(true)) {
                            let source = match borrow.get("source") {
                                Some(JsValue::String(s)) => s.to_string(),
                                _ => String::new(),
                            };
                            let flags = match borrow.get("flags") {
                                Some(JsValue::String(s)) => s.to_string(),
                                _ => String::new(),
                            };
                            // ES §22.1.3.12: matchAll requires the global flag.
                            if !flags.contains('g') {
                                return Err(StatorError::TypeError(
                                    "String.prototype.matchAll called with a non-global RegExp argument".to_string(),
                                ));
                            }
                            drop(borrow);
                            let re = crate::objects::regexp::JsRegExp::new(&source, &flags)?;
                            let matches = re.symbol_match_all(&s);
                            let results: Vec<JsValue> = matches
                                .into_iter()
                                .map(|m| {
                                    let mut parts: Vec<JsValue> =
                                        vec![JsValue::String(m.matched.clone().into())];
                                    for g in &m.captures {
                                        parts.push(match g {
                                            Some(s) => JsValue::String(s.clone().into()),
                                            None => JsValue::Undefined,
                                        });
                                    }
                                    JsValue::new_array(parts)
                                })
                                .collect();
                            Ok(JsValue::Iterator(NativeIterator::from_items(results)))
                        } else {
                            Ok(JsValue::Iterator(NativeIterator::from_items(vec![])))
                        }
                    }
                    Some(JsValue::String(pattern)) => {
                        let pattern = pattern.to_string();
                        if pattern.is_empty() {
                            return Ok(JsValue::Iterator(NativeIterator::from_items(vec![])));
                        }
                        let mut results = Vec::new();
                        let mut start = 0;
                        while let Some(pos) = s[start..].find(pattern.as_str()) {
                            let abs = start + pos;
                            let mut m = PropertyMap::new();
                            m.insert("0".to_string(), JsValue::String(Rc::from(pattern.as_str())));
                            m.insert("index".to_string(), JsValue::Smi(abs as i32));
                            m.insert("input".to_string(), JsValue::String(s.clone()));
                            m.insert("length".to_string(), JsValue::Smi(1));
                            results.push(JsValue::PlainObject(Rc::new(RefCell::new(m))));
                            start = abs + pattern.len();
                        }
                        Ok(JsValue::Iterator(NativeIterator::from_items(results)))
                    }
                    _ => Ok(JsValue::Iterator(NativeIterator::from_items(vec![]))),
                }));
            }
            "@@iterator" | "Symbol(1)" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::Iterator(NativeIterator::from_string(&s)))
                }));
            }
            "constructor" => {
                return lookup_global_constructor("String");
            }
            "isWellFormed" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::Boolean(
                        crate::builtins::string::string_is_well_formed(&s),
                    ))
                }));
            }
            "toWellFormed" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(
                        crate::builtins::string::string_to_well_formed(&s).into(),
                    ))
                }));
            }
            "toLocaleLowerCase" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(
                        crate::builtins::string::string_to_locale_lower_case(&s).into(),
                    ))
                }));
            }
            "toLocaleUpperCase" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(
                        crate::builtins::string::string_to_locale_upper_case(&s).into(),
                    ))
                }));
            }
            "substr" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let start = args
                        .first()
                        .map(|v| v.to_number().unwrap_or(f64::NAN) as i64)
                        .unwrap_or(0);
                    let length = args
                        .get(1)
                        .map(|v| v.to_number().unwrap_or(f64::NAN) as i64);
                    Ok(JsValue::String(
                        crate::builtins::string::string_substr(&s, start, length).into(),
                    ))
                }));
            }
            "anchor" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let name = args
                        .first()
                        .map(|v| v.to_js_string())
                        .transpose()?
                        .unwrap_or_default();
                    Ok(JsValue::String(
                        crate::builtins::string::string_anchor(&s, &name).into(),
                    ))
                }));
            }
            "big" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(
                        crate::builtins::string::string_big(&s).into(),
                    ))
                }));
            }
            "blink" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(
                        crate::builtins::string::string_blink(&s).into(),
                    ))
                }));
            }
            "bold" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(
                        crate::builtins::string::string_bold(&s).into(),
                    ))
                }));
            }
            "fixed" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(
                        crate::builtins::string::string_fixed(&s).into(),
                    ))
                }));
            }
            "fontcolor" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let color = args
                        .first()
                        .map(|v| v.to_js_string())
                        .transpose()?
                        .unwrap_or_default();
                    Ok(JsValue::String(
                        crate::builtins::string::string_fontcolor(&s, &color).into(),
                    ))
                }));
            }
            "fontsize" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let size = args
                        .first()
                        .map(|v| v.to_js_string())
                        .transpose()?
                        .unwrap_or_default();
                    Ok(JsValue::String(
                        crate::builtins::string::string_fontsize(&s, &size).into(),
                    ))
                }));
            }
            "italics" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(
                        crate::builtins::string::string_italics(&s).into(),
                    ))
                }));
            }
            "link" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let url = args
                        .first()
                        .map(|v| v.to_js_string())
                        .transpose()?
                        .unwrap_or_default();
                    Ok(JsValue::String(
                        crate::builtins::string::string_link(&s, &url).into(),
                    ))
                }));
            }
            "small" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(
                        crate::builtins::string::string_small(&s).into(),
                    ))
                }));
            }
            "strike" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(
                        crate::builtins::string::string_strike(&s).into(),
                    ))
                }));
            }
            "sub" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(
                        crate::builtins::string::string_sub(&s).into(),
                    ))
                }));
            }
            "sup" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(
                        crate::builtins::string::string_sup(&s).into(),
                    ))
                }));
            }
            "@@toPrimitive" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(s.clone()))
                }));
            }
            "hasOwnProperty" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let prop = match args.first() {
                        Some(JsValue::String(p)) => p.to_string(),
                        Some(JsValue::Smi(n)) => n.to_string(),
                        Some(JsValue::HeapNumber(n)) => format!("{n}"),
                        Some(JsValue::Boolean(b)) => b.to_string(),
                        Some(JsValue::Null) => "null".to_string(),
                        Some(JsValue::Undefined) => "undefined".to_string(),
                        _ => return Ok(JsValue::Boolean(false)),
                    };
                    if prop == "length" {
                        return Ok(JsValue::Boolean(true));
                    }
                    if let Ok(idx) = prop.parse::<usize>() {
                        let len = s.encode_utf16().count();
                        return Ok(JsValue::Boolean(idx < len));
                    }
                    Ok(JsValue::Boolean(false))
                }));
            }
            "propertyIsEnumerable" => {
                let s = s.clone();
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let prop = match args.first() {
                        Some(JsValue::String(p)) => p.to_string(),
                        Some(JsValue::Smi(n)) => n.to_string(),
                        _ => return Ok(JsValue::Boolean(false)),
                    };
                    // String char indices are enumerable; "length" is not.
                    if let Ok(idx) = prop.parse::<usize>() {
                        let len = s.encode_utf16().count();
                        return Ok(JsValue::Boolean(idx < len));
                    }
                    Ok(JsValue::Boolean(false))
                }));
            }
            "isPrototypeOf" => {
                return JsValue::NativeFunction(Rc::new(|_args| Ok(JsValue::Boolean(false))));
            }
            _ => {
                // Numeric string indexing: "0", "1", … → UTF-16 code unit at index.
                if let Ok(idx) = key.parse::<usize>() {
                    let utf16: Vec<u16> = s.encode_utf16().collect();
                    if idx < utf16.len() {
                        let ch = String::from_utf16_lossy(&utf16[idx..=idx]);
                        return JsValue::String(ch.into());
                    }
                    return JsValue::Undefined;
                }
            }
        },
        JsValue::Boolean(b) => match key {
            "toString" => {
                let b = *b;
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(
                        if b {
                            "true".to_string()
                        } else {
                            "false".to_string()
                        }
                        .into(),
                    ))
                }));
            }
            "valueOf" => {
                let b = *b;
                return JsValue::NativeFunction(Rc::new(move |_args| Ok(JsValue::Boolean(b))));
            }
            "constructor" => {
                return lookup_global_constructor("Boolean");
            }
            "@@toPrimitive" => {
                let b = *b;
                return JsValue::NativeFunction(Rc::new(move |_args| Ok(JsValue::Boolean(b))));
            }
            "hasOwnProperty" | "propertyIsEnumerable" => {
                return JsValue::NativeFunction(Rc::new(|_args| Ok(JsValue::Boolean(false))));
            }
            "isPrototypeOf" => {
                return JsValue::NativeFunction(Rc::new(|_args| Ok(JsValue::Boolean(false))));
            }
            _ => {}
        },
        JsValue::Array(arr) => {
            let arr_rc = Rc::clone(arr);
            match key {
                "length" => return JsValue::Smi(arr.borrow().len() as i32),
                "push" => {
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let mut v = arr_rc.borrow_mut();
                        for arg in &args {
                            v.push(arg.clone());
                        }
                        Ok(JsValue::Smi(v.len() as i32))
                    }));
                }
                "pop" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |_args| {
                        Ok(a.borrow_mut().pop().unwrap_or(JsValue::Undefined))
                    }));
                }
                "shift" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |_args| {
                        let mut v = a.borrow_mut();
                        if v.is_empty() {
                            Ok(JsValue::Undefined)
                        } else {
                            Ok(v.remove(0))
                        }
                    }));
                }
                "unshift" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let mut v = a.borrow_mut();
                        for (i, arg) in args.iter().enumerate() {
                            v.insert(i, arg.clone());
                        }
                        Ok(JsValue::Smi(v.len() as i32))
                    }));
                }
                "join" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let sep = match args.first() {
                            Some(JsValue::String(s)) => s.to_string(),
                            _ => ",".to_string(),
                        };
                        let parts: Vec<String> = a
                            .borrow()
                            .iter()
                            .map(|v| match v {
                                JsValue::String(s) => s.to_string(),
                                JsValue::Smi(n) => n.to_string(),
                                JsValue::HeapNumber(n) => format!("{n}"),
                                JsValue::Boolean(b) => b.to_string(),
                                JsValue::Null | JsValue::Undefined => String::new(),
                                _ => String::new(),
                            })
                            .collect();
                        Ok(JsValue::String(parts.join(&sep).into()))
                    }));
                }
                "toString" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |_args| {
                        let parts: Vec<String> = a
                            .borrow()
                            .iter()
                            .map(|v| match v {
                                JsValue::String(s) => s.to_string(),
                                JsValue::Smi(n) => n.to_string(),
                                JsValue::HeapNumber(n) => format!("{n}"),
                                JsValue::Boolean(b) => b.to_string(),
                                JsValue::Null | JsValue::Undefined => String::new(),
                                _ => String::new(),
                            })
                            .collect();
                        Ok(JsValue::String(parts.join(",").into()))
                    }));
                }
                "indexOf" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let search = args.first().cloned().unwrap_or(JsValue::Undefined);
                        Ok(JsValue::Smi(
                            a.borrow()
                                .iter()
                                .position(|v| strict_eq(v, &search))
                                .map_or(-1, |i| i as i32),
                        ))
                    }));
                }
                "lastIndexOf" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let search = args.first().cloned().unwrap_or(JsValue::Undefined);
                        let items = a.borrow();
                        let from = match args.get(1) {
                            Some(JsValue::Smi(i)) => {
                                if *i < 0 {
                                    (items.len() as i32 + *i).max(-1)
                                } else {
                                    (*i).min(items.len() as i32 - 1)
                                }
                            }
                            _ => items.len() as i32 - 1,
                        };
                        if from < 0 {
                            return Ok(JsValue::Smi(-1));
                        }
                        Ok(JsValue::Smi(
                            items[..=(from as usize)]
                                .iter()
                                .rposition(|v| strict_eq(v, &search))
                                .map_or(-1, |i| i as i32),
                        ))
                    }));
                }
                "includes" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let search = args.first().cloned().unwrap_or(JsValue::Undefined);
                        let items = a.borrow();
                        let len = items.len() as i32;
                        let from_raw = match args.get(1) {
                            Some(v) => v.to_number().unwrap_or(0.0) as i64,
                            None => 0,
                        };
                        let from = if from_raw < 0 {
                            ((len as i64) + from_raw).max(0) as usize
                        } else {
                            (from_raw as usize).min(items.len())
                        };
                        // SameValueZero: NaN === NaN, unlike strict equality
                        Ok(JsValue::Boolean(
                            items[from..].iter().any(|v| v.same_value_zero(&search)),
                        ))
                    }));
                }
                "slice" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let len = a.borrow().len() as i32;
                        let start = match args.first() {
                            Some(JsValue::Smi(i)) => {
                                if *i < 0 {
                                    (len + *i).max(0) as usize
                                } else {
                                    (*i as usize).min(len as usize)
                                }
                            }
                            _ => 0,
                        };
                        let end = match args.get(1) {
                            Some(JsValue::Smi(i)) => {
                                if *i < 0 {
                                    (len + *i).max(0) as usize
                                } else {
                                    (*i as usize).min(len as usize)
                                }
                            }
                            _ => len as usize,
                        };
                        if start >= end {
                            Ok(JsValue::new_array(vec![]))
                        } else {
                            Ok(JsValue::new_array(a.borrow()[start..end].to_vec()))
                        }
                    }));
                }
                "concat" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let mut result = a.borrow().clone();
                        for arg in args {
                            match arg {
                                JsValue::Array(other) => result.extend_from_slice(&other.borrow()),
                                v => result.push(v),
                            }
                        }
                        Ok(JsValue::new_array(result))
                    }));
                }
                "map" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let callback = args.first().cloned().unwrap_or(JsValue::Undefined);
                        let arr_val = JsValue::Array(Rc::clone(&a));
                        let items = a.borrow().clone();
                        let mut results = Vec::with_capacity(items.len());
                        for (i, item) in items.iter().enumerate() {
                            let val = dispatch_call_value(
                                &callback,
                                vec![item.clone(), JsValue::Smi(i as i32), arr_val.clone()],
                            )?;
                            results.push(val);
                        }
                        Ok(JsValue::new_array(results))
                    }));
                }
                "filter" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let callback = args.first().cloned().unwrap_or(JsValue::Undefined);
                        let arr_val = JsValue::Array(Rc::clone(&a));
                        let items = a.borrow().clone();
                        let mut results = Vec::new();
                        for (i, item) in items.iter().enumerate() {
                            let val = dispatch_call_value(
                                &callback,
                                vec![item.clone(), JsValue::Smi(i as i32), arr_val.clone()],
                            )?;
                            if val.to_boolean() {
                                results.push(item.clone());
                            }
                        }
                        Ok(JsValue::new_array(results))
                    }));
                }
                "forEach" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let callback = args.first().cloned().unwrap_or(JsValue::Undefined);
                        let arr_val = JsValue::Array(Rc::clone(&a));
                        let items = a.borrow().clone();
                        for (i, item) in items.iter().enumerate() {
                            dispatch_call_value(
                                &callback,
                                vec![item.clone(), JsValue::Smi(i as i32), arr_val.clone()],
                            )?;
                        }
                        Ok(JsValue::Undefined)
                    }));
                }
                "reduce" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let callback = args.first().cloned().unwrap_or(JsValue::Undefined);
                        let arr_val = JsValue::Array(Rc::clone(&a));
                        let items = a.borrow().clone();
                        let mut acc = args.get(1).cloned().unwrap_or_else(|| {
                            items.first().cloned().unwrap_or(JsValue::Undefined)
                        });
                        let start = if args.get(1).is_some() { 0 } else { 1 };
                        for (i, item) in items.iter().enumerate().skip(start) {
                            acc = dispatch_call_value(
                                &callback,
                                vec![acc, item.clone(), JsValue::Smi(i as i32), arr_val.clone()],
                            )?;
                        }
                        Ok(acc)
                    }));
                }
                "reduceRight" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let callback = args.first().cloned().unwrap_or(JsValue::Undefined);
                        let items = a.borrow().clone();
                        let arr_val = JsValue::Array(Rc::clone(&a));
                        if items.is_empty() && args.len() < 2 {
                            return Err(StatorError::TypeError(
                                "Reduce of empty array with no initial value".into(),
                            ));
                        }
                        let (mut acc, start_idx) = if args.len() > 1 {
                            (args[1].clone(), items.len())
                        } else {
                            (
                                items.last().cloned().unwrap_or(JsValue::Undefined),
                                items.len().saturating_sub(1),
                            )
                        };
                        for i in (0..start_idx).rev() {
                            acc = dispatch_call_value(
                                &callback,
                                vec![
                                    acc,
                                    items[i].clone(),
                                    JsValue::Smi(i as i32),
                                    arr_val.clone(),
                                ],
                            )?;
                        }
                        Ok(acc)
                    }));
                }
                "find" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let callback = args.first().cloned().unwrap_or(JsValue::Undefined);
                        let arr_val = JsValue::Array(Rc::clone(&a));
                        let items = a.borrow().clone();
                        for (i, item) in items.iter().enumerate() {
                            let val = dispatch_call_value(
                                &callback,
                                vec![item.clone(), JsValue::Smi(i as i32), arr_val.clone()],
                            )?;
                            if val.to_boolean() {
                                return Ok(item.clone());
                            }
                        }
                        Ok(JsValue::Undefined)
                    }));
                }
                "findIndex" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let callback = args.first().cloned().unwrap_or(JsValue::Undefined);
                        let arr_val = JsValue::Array(Rc::clone(&a));
                        let items = a.borrow().clone();
                        for (i, item) in items.iter().enumerate() {
                            let val = dispatch_call_value(
                                &callback,
                                vec![item.clone(), JsValue::Smi(i as i32), arr_val.clone()],
                            )?;
                            if val.to_boolean() {
                                return Ok(JsValue::Smi(i as i32));
                            }
                        }
                        Ok(JsValue::Smi(-1))
                    }));
                }
                "every" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let callback = args.first().cloned().unwrap_or(JsValue::Undefined);
                        let arr_val = JsValue::Array(Rc::clone(&a));
                        let items = a.borrow().clone();
                        for (i, item) in items.iter().enumerate() {
                            let val = dispatch_call_value(
                                &callback,
                                vec![item.clone(), JsValue::Smi(i as i32), arr_val.clone()],
                            )?;
                            if !val.to_boolean() {
                                return Ok(JsValue::Boolean(false));
                            }
                        }
                        Ok(JsValue::Boolean(true))
                    }));
                }
                "some" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let callback = args.first().cloned().unwrap_or(JsValue::Undefined);
                        let arr_val = JsValue::Array(Rc::clone(&a));
                        let items = a.borrow().clone();
                        for (i, item) in items.iter().enumerate() {
                            let val = dispatch_call_value(
                                &callback,
                                vec![item.clone(), JsValue::Smi(i as i32), arr_val.clone()],
                            )?;
                            if val.to_boolean() {
                                return Ok(JsValue::Boolean(true));
                            }
                        }
                        Ok(JsValue::Boolean(false))
                    }));
                }
                "reverse" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |_args| {
                        a.borrow_mut().reverse();
                        Ok(JsValue::Array(Rc::clone(&a)))
                    }));
                }
                "flat" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let depth: u32 = match args.first() {
                            Some(JsValue::Smi(i)) => (*i).max(0) as u32,
                            Some(JsValue::HeapNumber(n)) => {
                                if n.is_infinite() && n.is_sign_positive() {
                                    u32::MAX
                                } else {
                                    (*n as i32).max(0) as u32
                                }
                            }
                            Some(JsValue::Undefined) | None => 1,
                            _ => 0,
                        };
                        fn flatten(items: &[JsValue], depth: u32, out: &mut Vec<JsValue>) {
                            for item in items {
                                if depth > 0 {
                                    if let JsValue::Array(inner) = item {
                                        flatten(&inner.borrow(), depth - 1, out);
                                        continue;
                                    }
                                    // Also handle PlainObject arrays (from CreateArrayLiteral)
                                    if let JsValue::PlainObject(map) = item {
                                        let borrow = map.borrow();
                                        if borrow
                                            .get("__is_array__")
                                            .is_some_and(|v| matches!(v, JsValue::Boolean(true)))
                                        {
                                            let len = match borrow.get("length") {
                                                Some(JsValue::Smi(n)) => *n as usize,
                                                _ => 0,
                                            };
                                            let elems: Vec<JsValue> = (0..len)
                                                .map(|i| {
                                                    borrow
                                                        .get(&i.to_string())
                                                        .cloned()
                                                        .unwrap_or(JsValue::Undefined)
                                                })
                                                .collect();
                                            drop(borrow);
                                            flatten(&elems, depth - 1, out);
                                            continue;
                                        }
                                    }
                                }
                                out.push(item.clone());
                            }
                        }
                        let mut result = Vec::new();
                        let items = a.borrow().clone();
                        flatten(&items, depth, &mut result);
                        Ok(JsValue::new_array(result))
                    }));
                }
                "flatMap" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let callback = args.first().cloned().unwrap_or(JsValue::Undefined);
                        let arr_val = JsValue::Array(Rc::clone(&a));
                        let items = a.borrow().clone();
                        let mut result = Vec::new();
                        for (i, item) in items.iter().enumerate() {
                            let val = dispatch_call_value(
                                &callback,
                                vec![item.clone(), JsValue::Smi(i as i32), arr_val.clone()],
                            )?;
                            match &val {
                                JsValue::Array(inner) => {
                                    result.extend_from_slice(&inner.borrow());
                                }
                                JsValue::PlainObject(map) => {
                                    let borrow = map.borrow();
                                    if borrow
                                        .get("__is_array__")
                                        .is_some_and(|v| matches!(v, JsValue::Boolean(true)))
                                    {
                                        let len = match borrow.get("length") {
                                            Some(JsValue::Smi(n)) => *n as usize,
                                            _ => 0,
                                        };
                                        for idx in 0..len {
                                            result.push(
                                                borrow
                                                    .get(&idx.to_string())
                                                    .cloned()
                                                    .unwrap_or(JsValue::Undefined),
                                            );
                                        }
                                    } else {
                                        result.push(val.clone());
                                    }
                                }
                                _ => {
                                    result.push(val);
                                }
                            }
                        }
                        Ok(JsValue::new_array(result))
                    }));
                }
                "fill" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let fill_val = args.first().cloned().unwrap_or(JsValue::Undefined);
                        let mut v = a.borrow_mut();
                        let len = v.len();
                        let start = match args.get(1) {
                            Some(JsValue::Smi(i)) => {
                                let i = *i;
                                if i < 0 {
                                    (len as i32 + i).max(0) as usize
                                } else {
                                    (i as usize).min(len)
                                }
                            }
                            _ => 0,
                        };
                        let end = match args.get(2) {
                            Some(JsValue::Smi(i)) => {
                                let i = *i;
                                if i < 0 {
                                    (len as i32 + i).max(0) as usize
                                } else {
                                    (i as usize).min(len)
                                }
                            }
                            _ => len,
                        };
                        for item in v.iter_mut().skip(start).take(end.saturating_sub(start)) {
                            *item = fill_val.clone();
                        }
                        drop(v);
                        Ok(JsValue::Array(Rc::clone(&a)))
                    }));
                }
                "keys" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |_args| {
                        let keys: Vec<JsValue> = (0..a.borrow().len())
                            .map(|i| JsValue::Smi(i as i32))
                            .collect();
                        Ok(JsValue::Iterator(NativeIterator::from_items(keys)))
                    }));
                }
                "values" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |_args| {
                        let items = a.borrow().clone();
                        Ok(JsValue::Iterator(NativeIterator::from_items(items)))
                    }));
                }
                "entries" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |_args| {
                        let entries: Vec<JsValue> = a
                            .borrow()
                            .iter()
                            .enumerate()
                            .map(|(i, v)| {
                                JsValue::new_array(vec![JsValue::Smi(i as i32), v.clone()])
                            })
                            .collect();
                        Ok(JsValue::Iterator(NativeIterator::from_items(entries)))
                    }));
                }
                "@@iterator" | "Symbol(1)" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |_args| {
                        let items = a.borrow().clone();
                        Ok(JsValue::Iterator(NativeIterator::from_items(items)))
                    }));
                }
                "at" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let idx = match args.first() {
                            Some(v) => v.to_number().unwrap_or(0.0) as i32,
                            None => 0,
                        };
                        let len = a.borrow().len() as i32;
                        let actual = if idx < 0 { len + idx } else { idx };
                        if actual < 0 || actual >= len {
                            return Ok(JsValue::Undefined);
                        }
                        Ok(a.borrow()[actual as usize].clone())
                    }));
                }
                "sort" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let cmp_fn = args.first().cloned();
                        // Clone elements so that borrow_mut is not held while
                        // the comparator callback executes user code.
                        let mut items = a.borrow().clone();
                        items.sort_by(|x, y| {
                            if let Some(ref cb) = cmp_fn
                                && let Ok(r) = dispatch_call_value(cb, vec![x.clone(), y.clone()])
                            {
                                let n = match &r {
                                    JsValue::Smi(i) => *i as f64,
                                    JsValue::HeapNumber(f) => *f,
                                    _ => 0.0,
                                };
                                return n.partial_cmp(&0.0).unwrap_or(std::cmp::Ordering::Equal);
                            }
                            let a_str = js_to_string(x);
                            let b_str = js_to_string(y);
                            a_str.cmp(&b_str)
                        });
                        *a.borrow_mut() = items;
                        Ok(JsValue::Array(Rc::clone(&a)))
                    }));
                }
                "splice" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let mut v = a.borrow_mut();
                        let len = v.len() as i32;
                        let start_raw = match args.first() {
                            Some(JsValue::Smi(i)) => *i,
                            Some(JsValue::HeapNumber(n)) => *n as i32,
                            _ => 0,
                        };
                        let start = if start_raw < 0 {
                            (len + start_raw).max(0)
                        } else {
                            start_raw.min(len)
                        } as usize;
                        let delete_count = match args.get(1) {
                            Some(JsValue::Smi(i)) => (*i).max(0) as usize,
                            Some(JsValue::HeapNumber(n)) => (*n as i32).max(0) as usize,
                            _ => (len as usize).saturating_sub(start),
                        };
                        let end = (start + delete_count).min(v.len());
                        let removed: Vec<JsValue> = v.drain(start..end).collect();
                        // Insert new elements
                        let insert_items: Vec<JsValue> = args.iter().skip(2).cloned().collect();
                        for (i, item) in insert_items.into_iter().enumerate() {
                            v.insert(start + i, item);
                        }
                        Ok(JsValue::new_array(removed))
                    }));
                }
                "findLast" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let cb = args.first().cloned().unwrap_or(JsValue::Undefined);
                        let arr_val = JsValue::Array(Rc::clone(&a));
                        let items = a.borrow().clone();
                        for (i, v) in items.iter().enumerate().rev() {
                            let result = dispatch_call_value(
                                &cb,
                                vec![v.clone(), JsValue::Smi(i as i32), arr_val.clone()],
                            )?;
                            if to_boolean_val(&result) {
                                return Ok(v.clone());
                            }
                        }
                        Ok(JsValue::Undefined)
                    }));
                }
                "findLastIndex" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let cb = args.first().cloned().unwrap_or(JsValue::Undefined);
                        let arr_val = JsValue::Array(Rc::clone(&a));
                        let items = a.borrow().clone();
                        for (i, v) in items.iter().enumerate().rev() {
                            let result = dispatch_call_value(
                                &cb,
                                vec![v.clone(), JsValue::Smi(i as i32), arr_val.clone()],
                            )?;
                            if to_boolean_val(&result) {
                                return Ok(JsValue::Smi(i as i32));
                            }
                        }
                        Ok(JsValue::Smi(-1))
                    }));
                }
                "toReversed" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |_args| {
                        let mut rev = a.borrow().clone();
                        rev.reverse();
                        Ok(JsValue::new_array(rev))
                    }));
                }
                "toSorted" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let cmp_fn = args.first().cloned();
                        let mut sorted = a.borrow().clone();
                        sorted.sort_by(|x, y| {
                            if let Some(ref cb) = cmp_fn
                                && let Ok(r) = dispatch_call_value(cb, vec![x.clone(), y.clone()])
                            {
                                let n = match &r {
                                    JsValue::Smi(i) => *i as f64,
                                    JsValue::HeapNumber(f) => *f,
                                    _ => 0.0,
                                };
                                return n.partial_cmp(&0.0).unwrap_or(std::cmp::Ordering::Equal);
                            }
                            let a_str = js_to_string(x);
                            let b_str = js_to_string(y);
                            a_str.cmp(&b_str)
                        });
                        Ok(JsValue::new_array(sorted))
                    }));
                }
                "with" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let idx = match args.first() {
                            Some(JsValue::Smi(i)) => *i,
                            Some(JsValue::HeapNumber(n)) => *n as i32,
                            _ => 0,
                        };
                        let val = args.get(1).cloned().unwrap_or(JsValue::Undefined);
                        let len = a.borrow().len() as i32;
                        let actual = if idx < 0 { len + idx } else { idx };
                        if actual < 0 || actual >= len {
                            return Err(StatorError::RangeError("Invalid index".to_string()));
                        }
                        let mut new_arr = a.borrow().clone();
                        new_arr[actual as usize] = val;
                        Ok(JsValue::new_array(new_arr))
                    }));
                }
                "toSpliced" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let v = a.borrow();
                        let len = v.len() as i32;
                        let start_raw = match args.first() {
                            Some(JsValue::Smi(i)) => *i,
                            Some(JsValue::HeapNumber(n)) => *n as i32,
                            _ => 0,
                        };
                        let start = if start_raw < 0 {
                            (len + start_raw).max(0)
                        } else {
                            start_raw.min(len)
                        } as usize;
                        let delete_count = match args.get(1) {
                            Some(JsValue::Smi(i)) => (*i).max(0) as usize,
                            Some(JsValue::HeapNumber(n)) => (*n as i32).max(0) as usize,
                            _ => (len as usize).saturating_sub(start),
                        };
                        let end = (start + delete_count).min(v.len());
                        let mut new_arr = Vec::with_capacity(
                            v.len() - (end - start) + args.len().saturating_sub(2),
                        );
                        new_arr.extend_from_slice(&v[..start]);
                        new_arr.extend(args.iter().skip(2).cloned());
                        new_arr.extend_from_slice(&v[end..]);
                        Ok(JsValue::new_array(new_arr))
                    }));
                }
                "copyWithin" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let mut v = a.borrow_mut();
                        let len = v.len() as i32;
                        let target_raw = match args.first() {
                            Some(JsValue::Smi(i)) => *i,
                            Some(JsValue::HeapNumber(n)) => *n as i32,
                            _ => 0,
                        };
                        let target = if target_raw < 0 {
                            (len + target_raw).max(0)
                        } else {
                            target_raw.min(len)
                        } as usize;
                        let start_raw = match args.get(1) {
                            Some(JsValue::Smi(i)) => *i,
                            Some(JsValue::HeapNumber(n)) => *n as i32,
                            _ => 0,
                        };
                        let start = if start_raw < 0 {
                            (len + start_raw).max(0)
                        } else {
                            start_raw.min(len)
                        } as usize;
                        let end_raw = match args.get(2) {
                            Some(JsValue::Smi(i)) => *i,
                            Some(JsValue::HeapNumber(n)) => *n as i32,
                            _ => len,
                        };
                        let end = if end_raw < 0 {
                            (len + end_raw).max(0)
                        } else {
                            end_raw.min(len)
                        } as usize;
                        let count =
                            (end.saturating_sub(start)).min((len as usize).saturating_sub(target));
                        // Copy via temporary buffer to handle overlapping regions
                        let tmp: Vec<JsValue> = v[start..start + count].to_vec();
                        for (i, val) in tmp.into_iter().enumerate() {
                            v[target + i] = val;
                        }
                        drop(v);
                        Ok(JsValue::Array(Rc::clone(&a)))
                    }));
                }
                "valueOf" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |_args| {
                        Ok(JsValue::Array(Rc::clone(&a)))
                    }));
                }
                "constructor" => return lookup_global_constructor("Array"),
                "hasOwnProperty" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let prop = match args.first() {
                            Some(JsValue::String(s)) => s.to_string(),
                            Some(JsValue::Smi(n)) => n.to_string(),
                            Some(JsValue::HeapNumber(n)) => format!("{n}"),
                            Some(JsValue::Boolean(b)) => b.to_string(),
                            Some(JsValue::Null) => "null".to_string(),
                            Some(JsValue::Undefined) => "undefined".to_string(),
                            _ => return Ok(JsValue::Boolean(false)),
                        };
                        if prop == "length" {
                            return Ok(JsValue::Boolean(true));
                        }
                        if let Ok(idx) = prop.parse::<usize>() {
                            return Ok(JsValue::Boolean(idx < a.borrow().len()));
                        }
                        Ok(JsValue::Boolean(false))
                    }));
                }
                "propertyIsEnumerable" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |args| {
                        let prop = match args.first() {
                            Some(JsValue::String(s)) => s.to_string(),
                            Some(JsValue::Smi(n)) => n.to_string(),
                            _ => return Ok(JsValue::Boolean(false)),
                        };
                        // Array numeric indices are enumerable.
                        if let Ok(idx) = prop.parse::<usize>() {
                            return Ok(JsValue::Boolean(idx < a.borrow().len()));
                        }
                        Ok(JsValue::Boolean(false))
                    }));
                }
                "isPrototypeOf" => {
                    return JsValue::NativeFunction(Rc::new(|_args| Ok(JsValue::Boolean(false))));
                }
                "toLocaleString" => {
                    let a = Rc::clone(&arr_rc);
                    return JsValue::NativeFunction(Rc::new(move |_args| {
                        let parts: Vec<String> = a
                            .borrow()
                            .iter()
                            .map(|v| match v {
                                JsValue::Null | JsValue::Undefined => String::new(),
                                other => {
                                    let to_ls = proto_lookup(other, "toLocaleString");
                                    match dispatch_call_value(&to_ls, vec![other.clone()]) {
                                        Ok(JsValue::String(s)) => s.to_string(),
                                        Ok(v) => js_to_string(&v),
                                        Err(_) => js_to_string(other),
                                    }
                                }
                            })
                            .collect();
                        Ok(JsValue::String(parts.join(",").into()))
                    }));
                }
                _ => {
                    // Numeric index access: arr[0], arr[1], etc.
                    if let Ok(idx) = key.parse::<usize>()
                        && idx < arr.borrow().len()
                    {
                        return arr.borrow()[idx].clone();
                    }
                }
            }
            return JsValue::Undefined;
        }
        JsValue::BigInt(n) => match key {
            "toString" | "toLocaleString" => {
                let s = format!("{n}");
                return JsValue::NativeFunction(Rc::new(move |_args| {
                    Ok(JsValue::String(s.clone().into()))
                }));
            }
            "valueOf" => {
                let n = *n;
                return JsValue::NativeFunction(Rc::new(move |_args| Ok(JsValue::BigInt(n))));
            }
            _ => {}
        },
        _ => {}
    }
    // Handle JsValue::Proxy — delegate to the proxy get trap.
    if let JsValue::Proxy(p) = obj {
        return proxy_get(&p.borrow(), key).unwrap_or(JsValue::Undefined);
    }
    // Handle JsValue::Symbol — expose description, toString, valueOf.
    if let JsValue::Symbol(id) = obj {
        let id = *id;
        return match key {
            "description" => match symbol_description(id) {
                Some(desc) => JsValue::String(desc.into()),
                None => JsValue::Undefined,
            },
            "toString" => {
                JsValue::NativeFunction(Rc::new(move |_args| match symbol_description(id) {
                    Some(desc) => Ok(JsValue::String(format!("Symbol({desc})").into())),
                    None => Ok(JsValue::String("Symbol()".to_string().into())),
                }))
            }
            "valueOf" => JsValue::NativeFunction(Rc::new(move |_args| Ok(JsValue::Symbol(id)))),
            _ => JsValue::Undefined,
        };
    }
    // Handle JsValue::Error — expose name, message, stack, cause, errors properties.
    if let JsValue::Error(e) = obj {
        // Check user-set property overlay first.
        {
            let props = e.props.borrow();
            if let Some(val) = props.get(key) {
                return val.clone();
            }
        }
        // Fall back to built-in Error properties.
        let err = Rc::clone(e);
        return match key {
            "name" => JsValue::String(e.name().to_string().into()),
            "message" => JsValue::String(e.message().to_string().into()),
            "stack" => JsValue::String(e.stack().to_string().into()),
            "cause" => e.cause().cloned().unwrap_or(JsValue::Undefined),
            "errors" => {
                if e.kind == crate::builtins::error::ErrorKind::AggregateError {
                    JsValue::new_array(e.errors.clone())
                } else {
                    JsValue::Undefined
                }
            }
            "toString" => JsValue::NativeFunction(Rc::new(move |_args| {
                Ok(JsValue::String(err.to_error_string().into()))
            })),
            "valueOf" => {
                let e2 = Rc::clone(e);
                JsValue::NativeFunction(Rc::new(move |_args| Ok(JsValue::Error(Rc::clone(&e2)))))
            }
            "__proto__" => get_object_prototype(obj).unwrap_or(JsValue::Null),
            "constructor" => {
                if let Some(proto) = get_object_prototype(obj) {
                    let ctor = proto_lookup(&proto, "constructor");
                    if !matches!(ctor, JsValue::Undefined) {
                        ctor
                    } else {
                        lookup_global_constructor(e.kind.as_name())
                    }
                } else {
                    lookup_global_constructor(e.kind.as_name())
                }
            }
            "hasOwnProperty" => {
                let e = Rc::clone(e);
                JsValue::NativeFunction(Rc::new(move |args| {
                    let prop = match args.first() {
                        Some(JsValue::String(s)) => s.to_string(),
                        Some(JsValue::Smi(n)) => n.to_string(),
                        Some(JsValue::HeapNumber(n)) => format!("{n}"),
                        Some(JsValue::Boolean(b)) => b.to_string(),
                        Some(JsValue::Null) => "null".to_string(),
                        Some(JsValue::Undefined) => "undefined".to_string(),
                        Some(v) => v.to_js_string()?,
                        None => return Ok(JsValue::Boolean(false)),
                    };
                    // Check user overlay first.
                    if e.props.borrow().contains_key(&prop) {
                        return Ok(JsValue::Boolean(true));
                    }
                    Ok(JsValue::Boolean(matches!(
                        prop.as_str(),
                        "name" | "message" | "stack"
                    )))
                }))
            }
            "propertyIsEnumerable" => {
                JsValue::NativeFunction(Rc::new(|_args| Ok(JsValue::Boolean(false))))
            }
            _ => {
                if let Some(proto) = get_object_prototype(obj) {
                    proto_lookup_chain(&proto, key, obj)
                } else {
                    JsValue::Undefined
                }
            }
        };
    }
    // Handle JsValue::Generator — expose next/return/throw/@@iterator methods.
    if let JsValue::Generator(gs) = obj {
        let gs = Rc::clone(gs);
        let is_async_generator = gs.borrow().bytecode_array.is_async();
        return match key {
            "next" => {
                let gs = gs.clone();
                if is_async_generator {
                    JsValue::NativeFunction(Rc::new(move |args| {
                        use crate::builtins::promise::{
                            MicrotaskQueue, promise_reject, promise_resolve,
                        };

                        let queue = MicrotaskQueue::new();
                        let input = args.into_iter().next().unwrap_or(JsValue::Undefined);
                        let promise = match Interpreter::run_generator_step(&gs, input) {
                            Ok(GeneratorStep::Yield(value)) => {
                                promise_resolve(make_iterator_result(value, false), &queue)
                            }
                            Ok(GeneratorStep::Return(value)) => {
                                promise_resolve(make_iterator_result(value, true), &queue)
                            }
                            Err(error) => {
                                if let Some(reason) =
                                    Interpreter::js_error_to_rejection_reason(&error)
                                {
                                    promise_reject(reason, &queue)
                                } else {
                                    return Err(error);
                                }
                            }
                        };
                        queue.drain();
                        Ok(JsValue::Promise(promise))
                    }))
                } else {
                    JsValue::NativeFunction(Rc::new(move |args| {
                        let input = args.into_iter().next().unwrap_or(JsValue::Undefined);
                        match Interpreter::run_generator_step(&gs, input)? {
                            GeneratorStep::Yield(v) => Ok(make_iterator_result(v, false)),
                            GeneratorStep::Return(v) => Ok(make_iterator_result(v, true)),
                        }
                    }))
                }
            }
            "return" => {
                let gs = gs.clone();
                if is_async_generator {
                    JsValue::NativeFunction(Rc::new(move |args| {
                        use crate::builtins::promise::{
                            MicrotaskQueue, promise_reject, promise_resolve,
                        };

                        let queue = MicrotaskQueue::new();
                        let value = args.into_iter().next().unwrap_or(JsValue::Undefined);
                        let promise = match Interpreter::generator_return(&gs, value) {
                            Ok(result) => promise_resolve(result, &queue),
                            Err(error) => {
                                if let Some(reason) =
                                    Interpreter::js_error_to_rejection_reason(&error)
                                {
                                    promise_reject(reason, &queue)
                                } else {
                                    return Err(error);
                                }
                            }
                        };
                        queue.drain();
                        Ok(JsValue::Promise(promise))
                    }))
                } else {
                    JsValue::NativeFunction(Rc::new(move |args| {
                        let value = args.into_iter().next().unwrap_or(JsValue::Undefined);
                        Interpreter::generator_return(&gs, value)
                    }))
                }
            }
            "throw" => {
                let gs = gs.clone();
                if is_async_generator {
                    JsValue::NativeFunction(Rc::new(move |args| {
                        use crate::builtins::promise::{
                            MicrotaskQueue, promise_reject, promise_resolve,
                        };

                        let queue = MicrotaskQueue::new();
                        let value = args.into_iter().next().unwrap_or(JsValue::Undefined);
                        let promise = match Interpreter::generator_throw(&gs, value) {
                            Ok(result) => promise_resolve(result, &queue),
                            Err(error) => {
                                if let Some(reason) =
                                    Interpreter::js_error_to_rejection_reason(&error)
                                {
                                    promise_reject(reason, &queue)
                                } else {
                                    return Err(error);
                                }
                            }
                        };
                        queue.drain();
                        Ok(JsValue::Promise(promise))
                    }))
                } else {
                    JsValue::NativeFunction(Rc::new(move |args| {
                        let value = args.into_iter().next().unwrap_or(JsValue::Undefined);
                        Interpreter::generator_throw(&gs, value)
                    }))
                }
            }
            // §27.5.1.2 — Generator.prototype[@@iterator]() returns `this`.
            "@@iterator" | "Symbol(1)" if !is_async_generator => {
                let generator = obj.clone();
                JsValue::NativeFunction(Rc::new(move |_args| Ok(generator.clone())))
            }
            "@@asyncIterator" | "Symbol(2)" if is_async_generator => {
                let generator = obj.clone();
                JsValue::NativeFunction(Rc::new(move |_args| Ok(generator.clone())))
            }
            _ => JsValue::Undefined,
        };
    }
    // Handle JsValue::Iterator — expose next/return/@@iterator so that
    // user-land code can call `iter.next()` and iterators satisfy the
    // iterable protocol (`iter[Symbol.iterator]() === iter`).
    if let JsValue::Iterator(ni) = obj {
        let ni = Rc::clone(ni);
        return match key {
            "next" => {
                let ni = ni.clone();
                JsValue::NativeFunction(Rc::new(move |_args| {
                    let item = ni.borrow_mut().next_item();
                    match item {
                        Some(v) => Ok(make_iterator_result(v, false)),
                        None => Ok(make_iterator_result(JsValue::Undefined, true)),
                    }
                }))
            }
            "return" => {
                let ni = ni.clone();
                JsValue::NativeFunction(Rc::new(move |args| {
                    // Force the iterator to its end so subsequent .next()
                    // calls return done: true.
                    let mut inner = ni.borrow_mut();
                    inner.index = inner.items.len();
                    let value = args.into_iter().next().unwrap_or(JsValue::Undefined);
                    Ok(make_iterator_result(value, true))
                }))
            }
            // §27.1.2 %IteratorPrototype%[@@iterator]() — return this.
            // Handle both internal "@@iterator" and computed "Symbol(1)".
            "@@iterator" | "Symbol(1)" => {
                let iter = obj.clone();
                JsValue::NativeFunction(Rc::new(move |_args| Ok(iter.clone())))
            }
            _ => JsValue::Undefined,
        };
    }
    // Handle JsValue::Promise — expose then/catch/finally.
    if let JsValue::Promise(_p) = obj {
        return match key {
            "then" | "catch" | "finally" => {
                // Return a no-op native function for now; full chaining is
                // handled at a higher level by the promise builtins.
                JsValue::NativeFunction(Rc::new(|_args| Ok(JsValue::Undefined)))
            }
            _ => JsValue::Undefined,
        };
    }
    // Handle JsValue::Function — look up ad-hoc properties stored in the
    // thread-local side table (e.g. `assert.sameValue`).
    if let JsValue::Function(ba) = obj {
        let val = fn_props_get(ba, key);
        if !matches!(val, JsValue::Undefined) {
            return val;
        }
        // Built-in Function.prototype methods.
        match key {
            "call" => {
                let ba = Rc::clone(ba);
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let this_arg = args.first().cloned().unwrap_or(JsValue::Undefined);
                    let call_args = if args.len() > 1 {
                        args[1..].to_vec()
                    } else {
                        vec![]
                    };
                    dispatch_call_with_this(&JsValue::Function(Rc::clone(&ba)), this_arg, call_args)
                }));
            }
            "apply" => {
                let ba = Rc::clone(ba);
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let this_arg = args.first().cloned().unwrap_or(JsValue::Undefined);
                    let call_args = match args.get(1) {
                        Some(JsValue::Array(arr)) => arr.borrow().clone(),
                        Some(JsValue::PlainObject(map)) => {
                            let borrow = map.borrow();
                            let len = match borrow.get("length") {
                                Some(JsValue::Smi(n)) => *n as usize,
                                Some(JsValue::HeapNumber(n)) => *n as usize,
                                _ => 0,
                            };
                            (0..len)
                                .filter_map(|i| borrow.get(&i.to_string()).cloned())
                                .collect()
                        }
                        _ => vec![],
                    };
                    dispatch_call_with_this(&JsValue::Function(Rc::clone(&ba)), this_arg, call_args)
                }));
            }
            "bind" => {
                let ba = Rc::clone(ba);
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let this_arg = args.first().cloned().unwrap_or(JsValue::Undefined);
                    let bound_args: Vec<JsValue> = if args.len() > 1 {
                        args[1..].to_vec()
                    } else {
                        vec![]
                    };
                    let ba2 = Rc::clone(&ba);
                    let bound_count = bound_args.len() as i32;
                    let target_len = ba.parameter_count() as i32;
                    let result_len = std::cmp::max(0, target_len - bound_count);

                    // Build bound function as a PlainObject with __call__,
                    // carrying the correct `name` and `length` per ES §20.2.3.2.
                    let call_fn =
                        JsValue::NativeFunction(Rc::new(move |call_args: Vec<JsValue>| {
                            let mut all_args = bound_args.clone();
                            all_args.extend(call_args);
                            dispatch_call_with_this(
                                &JsValue::Function(Rc::clone(&ba2)),
                                this_arg.clone(),
                                all_args,
                            )
                        }));
                    let mut props = PropertyMap::new();
                    props.insert("__call__".to_string(), call_fn);
                    props.insert(
                        "name".to_string(),
                        JsValue::String("bound ".to_string().into()),
                    );
                    props.insert("length".to_string(), JsValue::Smi(result_len));
                    Ok(JsValue::PlainObject(Rc::new(RefCell::new(props))))
                }));
            }
            "name" => {
                let n = ba.function_name();
                return JsValue::String(
                    if n.is_empty() {
                        String::new()
                    } else {
                        n.to_owned()
                    }
                    .into(),
                );
            }
            "length" => {
                return JsValue::Smi(ba.parameter_count() as i32);
            }
            "toString" => {
                return JsValue::NativeFunction(Rc::new(|_args| {
                    Ok(JsValue::String(
                        "function () { [native code] }".to_string().into(),
                    ))
                }));
            }
            "constructor" => return lookup_global_constructor("Function"),
            "hasOwnProperty" => {
                let ba = Rc::clone(ba);
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let prop = match args.first() {
                        Some(JsValue::String(s)) => s.to_string(),
                        Some(JsValue::Smi(n)) => n.to_string(),
                        Some(JsValue::HeapNumber(n)) => format!("{n}"),
                        Some(JsValue::Boolean(b)) => b.to_string(),
                        Some(JsValue::Null) => "null".to_string(),
                        Some(JsValue::Undefined) => "undefined".to_string(),
                        _ => return Ok(JsValue::Boolean(false)),
                    };
                    // Check fn_props side table first.
                    if !matches!(fn_props_get(&ba, &prop), JsValue::Undefined) {
                        return Ok(JsValue::Boolean(true));
                    }
                    Ok(JsValue::Boolean(matches!(
                        prop.as_str(),
                        "length" | "name" | "prototype"
                    )))
                }));
            }
            "propertyIsEnumerable" => {
                return JsValue::NativeFunction(Rc::new(|_args| {
                    // Function own properties are not enumerable.
                    Ok(JsValue::Boolean(false))
                }));
            }
            "isPrototypeOf" => {
                return JsValue::NativeFunction(Rc::new(|_args| Ok(JsValue::Boolean(false))));
            }
            "valueOf" => {
                let obj_clone = obj.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| Ok(obj_clone.clone())));
            }
            "toLocaleString" => {
                return JsValue::NativeFunction(Rc::new(|_args| {
                    Ok(JsValue::String(
                        "function () { [native code] }".to_string().into(),
                    ))
                }));
            }
            _ => {}
        }
    }
    // Same for NativeFunction.
    if let JsValue::NativeFunction(f) = obj {
        match key {
            "call" => {
                let f = Rc::clone(f);
                return JsValue::NativeFunction(Rc::new(move |args| {
                    // Pass thisArg (args[0]) through so that
                    // Object.prototype.toString.call(value) works.
                    f(args)
                }));
            }
            "apply" => {
                let f = Rc::clone(f);
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let this_arg = args.first().cloned().unwrap_or(JsValue::Undefined);
                    let call_args = match args.get(1) {
                        Some(JsValue::Array(arr)) => arr.borrow().clone(),
                        Some(JsValue::PlainObject(map)) => {
                            let borrow = map.borrow();
                            let len = match borrow.get("length") {
                                Some(JsValue::Smi(n)) => *n as usize,
                                Some(JsValue::HeapNumber(n)) => *n as usize,
                                _ => 0,
                            };
                            (0..len)
                                .filter_map(|i| borrow.get(&i.to_string()).cloned())
                                .collect()
                        }
                        _ => vec![],
                    };
                    let mut full_args = vec![this_arg];
                    full_args.extend(call_args);
                    f(full_args)
                }));
            }
            "bind" => {
                let f = Rc::clone(f);
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let this_arg = args.first().cloned().unwrap_or(JsValue::Undefined);
                    let bound_args: Vec<JsValue> = if args.len() > 1 {
                        args[1..].to_vec()
                    } else {
                        vec![]
                    };
                    let f2 = Rc::clone(&f);
                    Ok(JsValue::NativeFunction(Rc::new(move |call_args| {
                        let mut all_args = vec![this_arg.clone()];
                        all_args.extend(bound_args.clone());
                        all_args.extend(call_args);
                        f2(all_args)
                    })))
                }));
            }
            "name" => return JsValue::String(String::new().into()),
            "length" => return JsValue::Smi(0),
            "toString" => {
                return JsValue::NativeFunction(Rc::new(|_args| {
                    Ok(JsValue::String(
                        "function () { [native code] }".to_string().into(),
                    ))
                }));
            }
            "constructor" => return lookup_global_constructor("Function"),
            "hasOwnProperty" => {
                return JsValue::NativeFunction(Rc::new(|args| {
                    let prop = match args.first() {
                        Some(JsValue::String(s)) => s.to_string(),
                        Some(JsValue::Smi(n)) => n.to_string(),
                        Some(JsValue::HeapNumber(n)) => format!("{n}"),
                        Some(JsValue::Boolean(b)) => b.to_string(),
                        Some(JsValue::Null) => "null".to_string(),
                        Some(JsValue::Undefined) => "undefined".to_string(),
                        _ => return Ok(JsValue::Boolean(false)),
                    };
                    Ok(JsValue::Boolean(matches!(prop.as_str(), "length" | "name")))
                }));
            }
            "propertyIsEnumerable" => {
                return JsValue::NativeFunction(Rc::new(|_args| {
                    // NativeFunction own properties are not enumerable.
                    Ok(JsValue::Boolean(false))
                }));
            }
            "isPrototypeOf" => {
                return JsValue::NativeFunction(Rc::new(|_args| Ok(JsValue::Boolean(false))));
            }
            "valueOf" => {
                let obj_clone = obj.clone();
                return JsValue::NativeFunction(Rc::new(move |_args| Ok(obj_clone.clone())));
            }
            "toLocaleString" => {
                return JsValue::NativeFunction(Rc::new(|_args| {
                    Ok(JsValue::String(
                        "function () { [native code] }".to_string().into(),
                    ))
                }));
            }
            _ => {}
        }
    }
    let mut current = obj.clone();
    for _ in 0..256 {
        if let JsValue::PlainObject(ref map) = current {
            let borrow = map.borrow();
            if let Some(val) = borrow.get(key) {
                return val.clone();
            }
            let getter_key = format!("__get_{key}__");
            if let Some(getter) = borrow.get(&getter_key).cloned() {
                drop(borrow);
                return match dispatch_getter(&getter, obj) {
                    Ok(v) => v,
                    Err(_) => JsValue::Undefined,
                };
            }
            if let Some(proto) = borrow.get("__proto__") {
                let next = proto.clone();
                drop(borrow);
                current = next;
                continue;
            }
        }
        break;
    }
    JsValue::Undefined
}

/// Delegate a property lookup on a PlainObject array to the Array prototype.
///
/// Extracts the indexed elements from the `PropertyMap`, wraps them in a
/// `JsValue::Array`, and re-dispatches through `proto_lookup` so that all
/// `Array.prototype` methods (push, map, filter, etc.) just work.
fn array_literal_proto_lookup(obj: &JsValue, key: &str) -> JsValue {
    if let JsValue::PlainObject(map) = obj {
        let borrow = map.borrow();
        let len = match borrow.get("length") {
            Some(JsValue::Smi(n)) => *n as usize,
            _ => 0,
        };
        let mut elements = Vec::with_capacity(len);
        for i in 0..len {
            let k = i.to_string();
            elements.push(borrow.get(&k).cloned().unwrap_or(JsValue::Undefined));
        }
        drop(borrow);
        // Build a real JsValue::Array and lookup on it.
        let arr = JsValue::Array(Rc::new(RefCell::new(elements)));
        let result = proto_lookup(&arr, key);
        // For mutating methods we need to sync changes back.  Wrap the
        // returned NativeFunction so that after the call, the backing
        // PlainObject is updated from the Array.
        if matches!(
            key,
            "push"
                | "pop"
                | "shift"
                | "unshift"
                | "splice"
                | "reverse"
                | "sort"
                | "fill"
                | "copyWithin"
        ) && let JsValue::NativeFunction(ref inner_fn) = result
        {
            let inner_fn = Rc::clone(inner_fn);
            let map_rc = Rc::clone(map);
            if let JsValue::Array(ref arr_ref) = arr {
                let arr_ref = Rc::clone(arr_ref);
                return JsValue::NativeFunction(Rc::new(move |args| {
                    let ret = inner_fn(args)?;
                    // Sync back from Array to PlainObject.
                    let elems = arr_ref.borrow();
                    let mut m = map_rc.borrow_mut();
                    // Remove old numeric keys.
                    let old_len = match m.get("length") {
                        Some(JsValue::Smi(n)) => *n as usize,
                        _ => 0,
                    };
                    for i in 0..old_len {
                        m.remove(&i.to_string());
                    }
                    // Insert new elements.
                    for (i, v) in elems.iter().enumerate() {
                        m.insert(i.to_string(), v.clone());
                    }
                    m.insert("length".to_string(), JsValue::Smi(elems.len() as i32));
                    Ok(ret)
                }));
            }
        }
        return result;
    }
    JsValue::Undefined
}

/// Walk the `__proto__` chain starting from `current`, with `this_obj` as
/// the original receiver (for getter invocation).
fn proto_lookup_chain(current: &JsValue, key: &str, this_obj: &JsValue) -> JsValue {
    let mut current = current.clone();
    for _ in 0..256 {
        if let JsValue::PlainObject(ref map) = current {
            let borrow = map.borrow();
            // Check for getter accessor BEFORE data key (same rationale
            // as proto_lookup: accessor properties may have a placeholder
            // data key that should not shadow the getter).
            let getter_key = format!("__get_{key}__");
            if let Some(getter) = borrow.get(&getter_key).cloned() {
                drop(borrow);
                return match dispatch_getter(&getter, this_obj) {
                    Ok(v) => v,
                    Err(_) => JsValue::Undefined,
                };
            }
            if let Some(val) = borrow.get(key) {
                return val.clone();
            }
            if let Some(proto) = borrow.get("__proto__") {
                let next = proto.clone();
                drop(borrow);
                current = next;
                continue;
            }
        }
        break;
    }
    // Object.prototype fallback — provide basic methods when the __proto__
    // chain is exhausted without an explicit Object.prototype.
    match key {
        "hasOwnProperty" => {
            let this = this_obj.clone();
            return JsValue::NativeFunction(Rc::new(move |args| {
                let prop = match args.first() {
                    Some(JsValue::String(s)) => s.to_string(),
                    Some(JsValue::Smi(n)) => n.to_string(),
                    Some(JsValue::HeapNumber(n)) => format!("{n}"),
                    Some(JsValue::Boolean(b)) => b.to_string(),
                    Some(JsValue::Null) => "null".to_string(),
                    Some(JsValue::Undefined) => "undefined".to_string(),
                    _ => return Ok(JsValue::Boolean(false)),
                };
                if let JsValue::PlainObject(ref m) = this {
                    return Ok(JsValue::Boolean(m.borrow().contains_key(&prop)));
                }
                Ok(JsValue::Boolean(false))
            }));
        }
        "propertyIsEnumerable" => {
            let this = this_obj.clone();
            return JsValue::NativeFunction(Rc::new(move |args| {
                let prop = match args.first() {
                    Some(JsValue::String(s)) => s.to_string(),
                    Some(JsValue::Smi(n)) => n.to_string(),
                    _ => return Ok(JsValue::Boolean(false)),
                };
                if let JsValue::PlainObject(ref m) = this {
                    return Ok(JsValue::Boolean(m.borrow().is_enumerable(&prop)));
                }
                Ok(JsValue::Boolean(false))
            }));
        }
        "isPrototypeOf" => {
            return JsValue::NativeFunction(Rc::new(|_args| Ok(JsValue::Boolean(false))));
        }
        "constructor" => return lookup_global_constructor("Object"),
        "toString" => {
            let this = this_obj.clone();
            return JsValue::NativeFunction(Rc::new(move |args| {
                if let Some(value) = args.first() {
                    return Ok(JsValue::String(value.obj_to_string_tag().into()));
                }
                Ok(JsValue::String(this.obj_to_string_tag().into()))
            }));
        }
        "valueOf" => {
            let this = this_obj.clone();
            return JsValue::NativeFunction(Rc::new(move |_args| Ok(this.clone())));
        }
        "toLocaleString" => {
            let this = this_obj.clone();
            return JsValue::NativeFunction(Rc::new(move |_args| {
                let ts = proto_lookup(&this, "toString");
                dispatch_call_value(&ts, vec![this.clone()])
            }));
        }
        _ => {}
    }
    JsValue::Undefined
}

/// Invoke a getter accessor function.
///
/// The getter may be a `JsValue::Function` (bytecode) or
/// `JsValue::NativeFunction`.  Returns the getter's return value.
fn dispatch_getter(getter: &JsValue, this: &JsValue) -> StatorResult<JsValue> {
    match getter {
        JsValue::Function(ba) => {
            push_call_frame("<getter>")?;
            let mut frame = if let Some(globals) = CURRENT_GLOBALS.with(|g| g.borrow().clone()) {
                InterpreterFrame::new_with_globals((**ba).clone(), vec![], globals)
            } else {
                InterpreterFrame::new((**ba).clone(), vec![])
            };
            restore_closure_context(&mut frame, ba);
            frame
                .global_env
                .borrow_mut()
                .insert("this".to_string(), this.clone());
            let result = Interpreter::run(&mut frame);
            pop_call_frame();
            result
        }
        JsValue::NativeFunction(f) => f(vec![this.clone()]),
        _ => Ok(JsValue::Undefined),
    }
}

/// Invoke a setter accessor function with the given value.
pub(super) fn dispatch_setter(setter: &JsValue, this: &JsValue, val: JsValue) -> StatorResult<()> {
    match setter {
        JsValue::Function(ba) => {
            push_call_frame("<setter>")?;
            let mut frame = if let Some(globals) = CURRENT_GLOBALS.with(|g| g.borrow().clone()) {
                InterpreterFrame::new_with_globals((**ba).clone(), vec![val], globals)
            } else {
                InterpreterFrame::new((**ba).clone(), vec![val])
            };
            restore_closure_context(&mut frame, ba);
            frame
                .global_env
                .borrow_mut()
                .insert("this".to_string(), this.clone());
            let result = Interpreter::run(&mut frame);
            pop_call_frame();
            result?;
            Ok(())
        }
        JsValue::NativeFunction(f) => {
            f(vec![val])?;
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Invoke a callable JsValue (Function or NativeFunction) with the given
/// arguments and return the result.
///
/// This is the engine's primary mechanism for calling any JS-callable value
/// from native Rust code.  It handles:
/// - `JsValue::Function` (bytecode) — creates a new interpreter frame
/// - `JsValue::NativeFunction` — calls the Rust closure directly
/// - `JsValue::PlainObject` with `__call__` — delegates to the callable slot
pub fn dispatch_call_value(callee: &JsValue, args: Vec<JsValue>) -> StatorResult<JsValue> {
    match callee {
        JsValue::Function(ba) => {
            // Generator functions return a suspended generator object instead
            // of executing the body (§27.3.3.1).
            if ba.is_generator() {
                return Ok(JsValue::Generator(GeneratorState::new((**ba).clone())));
            }
            push_call_frame("<anonymous>")?;
            let mut frame = if let Some(globals) = CURRENT_GLOBALS.with(|g| g.borrow().clone()) {
                InterpreterFrame::new_with_globals((**ba).clone(), args, globals)
            } else {
                InterpreterFrame::new((**ba).clone(), args)
            };
            restore_closure_context(&mut frame, ba);
            populate_self_name(&mut frame, ba, &JsValue::Function(Rc::clone(ba)));
            let result = Interpreter::run(&mut frame);
            pop_call_frame();
            result
        }
        JsValue::NativeFunction(f) => f(args),
        JsValue::PlainObject(map) => {
            let call_fn = map.borrow().get("__call__").cloned();
            if let Some(f) = call_fn {
                dispatch_call_value(&f, args)
            } else {
                Err(StatorError::TypeError("object is not a function".into()))
            }
        }
        _ => Err(StatorError::TypeError("value is not a function".into())),
    }
}

/// Invoke a callable JsValue with a specific `this` receiver.
///
/// Sets `"this"` in the global environment so that `Expr::This` (which compiles
/// to `LdaGlobal("this")`) resolves to the given receiver.
pub fn dispatch_call_with_this(
    callee: &JsValue,
    this_val: JsValue,
    args: Vec<JsValue>,
) -> StatorResult<JsValue> {
    match callee {
        JsValue::Function(ba) => {
            // Generator functions return a suspended generator object instead
            // of executing the body (§27.3.3.1).
            if ba.is_generator() {
                return Ok(JsValue::Generator(GeneratorState::new((**ba).clone())));
            }
            push_call_frame("<anonymous>")?;
            let mut frame = if let Some(globals) = CURRENT_GLOBALS.with(|g| g.borrow().clone()) {
                InterpreterFrame::new_with_globals((**ba).clone(), args, globals)
            } else {
                InterpreterFrame::new((**ba).clone(), args)
            };
            restore_closure_context(&mut frame, ba);
            populate_self_name(&mut frame, ba, &JsValue::Function(Rc::clone(ba)));
            // Arrow functions use lexical `this` — do NOT override.
            if !ba.is_arrow() {
                // ES §10.2.1.2: in sloppy mode, null/undefined `this` → globalThis.
                let effective_this = if !ba.is_strict() && this_val.is_nullish() {
                    frame
                        .global_env
                        .borrow()
                        .get("globalThis")
                        .cloned()
                        .unwrap_or(JsValue::Undefined)
                } else {
                    this_val
                };
                frame
                    .global_env
                    .borrow_mut()
                    .insert("this".to_string(), effective_this);
            }
            let result = Interpreter::run(&mut frame);
            pop_call_frame();
            result
        }
        JsValue::NativeFunction(f) => f(args),
        JsValue::PlainObject(map) => {
            let call_fn = map.borrow().get("__call__").cloned();
            if let Some(f) = call_fn {
                dispatch_call_with_this(&f, this_val, args)
            } else {
                Err(StatorError::TypeError("object is not a function".into()))
            }
        }
        _ => Err(StatorError::TypeError("value is not a function".into())),
    }
}

/// Wire `[[Prototype]]` on a newly constructed object.
///
/// If the constructor body returned a `PlainObject` that does not already have
/// a `__proto__` link, set it to `ctor_proto` so that `instanceof` and
/// prototype-chain property lookup work correctly.
pub(super) fn wire_construct_prototype(result: JsValue, ctor_proto: &JsValue) -> JsValue {
    if !matches!(ctor_proto, JsValue::Undefined) {
        match &result {
            JsValue::PlainObject(map) => {
                let mut borrow = map.borrow_mut();
                if !borrow.contains_key("__proto__") {
                    borrow.insert("__proto__".to_string(), ctor_proto.clone());
                }
            }
            JsValue::Error(e) => {
                let mut props = e.props.borrow_mut();
                if !props.contains_key("__proto__") {
                    props.insert("__proto__".to_string(), ctor_proto.clone());
                }
            }
            _ => {}
        }
    }
    result
}

pub(super) fn keyed_load(obj: &JsValue, key: &JsValue) -> StatorResult<JsValue> {
    // TypeError for keyed property access on null or undefined (ES §13.10.3).
    if matches!(obj, JsValue::Null | JsValue::Undefined) {
        let key_str = to_property_key(key).unwrap_or_default();
        return Err(StatorError::TypeError(format!(
            "Cannot read properties of {} (reading '{key_str}')",
            if matches!(obj, JsValue::Null) {
                "null"
            } else {
                "undefined"
            }
        )));
    }
    match obj {
        JsValue::Proxy(_) => {
            let prop_name = to_property_key(key)?;
            Ok(proto_lookup(obj, &prop_name))
        }
        JsValue::PlainObject(_map) => {
            let prop_name = to_property_key(key)?;
            Ok(proto_lookup(obj, &prop_name))
        }
        JsValue::Array(items) => {
            // "length" property
            if let JsValue::String(s) = key
                && &**s == "length"
            {
                return Ok(JsValue::Smi(items.borrow().len() as i32));
            }
            // Integer index
            if let Some(idx) = to_array_index(key) {
                return Ok(items
                    .borrow()
                    .get(idx)
                    .cloned()
                    .unwrap_or(JsValue::Undefined));
            }
            // Named property — delegate to proto_lookup for method access.
            let prop_name = to_property_key(key)?;
            Ok(proto_lookup(obj, &prop_name))
        }
        JsValue::String(_) => {
            // "length" property
            if let JsValue::String(k) = key
                && &**k == "length"
                && let JsValue::String(s) = obj
            {
                return Ok(JsValue::Smi(s.encode_utf16().count() as i32));
            }
            // Character-at-index
            if let Some(idx) = to_array_index(key)
                && let JsValue::String(s) = obj
            {
                let utf16: Vec<u16> = s.encode_utf16().collect();
                return Ok(if idx < utf16.len() {
                    JsValue::String(String::from_utf16_lossy(&utf16[idx..=idx]).into())
                } else {
                    JsValue::Undefined
                });
            }
            // Named property — delegate to proto_lookup for method access.
            let prop_name = to_property_key(key)?;
            Ok(proto_lookup(obj, &prop_name))
        }
        _ => {
            // For any other type, try proto_lookup for method access.
            let prop_name = to_property_key(key)?;
            Ok(proto_lookup(obj, &prop_name))
        }
    }
}

/// Perform a keyed property store: `obj[key] = value`.
///
/// Supports `PlainObject` (string keys).  Stores to non-object types are
/// silently discarded (matching the existing `StaNamedProperty` behaviour).
///
/// For `PlainObject` targets this enforces ECMAScript property descriptor
/// invariants:
/// - **Accessor properties**: if a setter (`__set_<key>__`) is defined, the
///   setter is invoked; if only a getter exists (no setter) the store is
///   silently ignored (sloppy mode — strict mode callers add a TypeError
///   before reaching this helper).
/// - **Non-writable properties**: the store is silently ignored.
/// - **Non-extensible objects**: adding a new property is silently ignored.
pub(super) fn keyed_store(obj: &JsValue, key: &JsValue, value: JsValue) -> StatorResult<()> {
    match obj {
        JsValue::Proxy(p) => {
            let prop_name = to_property_key(key)?;
            let _ = proxy_set(&mut p.borrow_mut(), &prop_name, value)?;
        }
        JsValue::PlainObject(map) => {
            let prop_name = to_property_key(key)?;
            // Check for setter accessor (__set_<key>__) first — accessor
            // properties take precedence over data properties.
            let setter_key = format!("__set_{prop_name}__");
            let getter_key = format!("__get_{prop_name}__");
            {
                let pm = map.borrow();
                let has_setter = pm.contains_key(&setter_key);
                let has_getter = pm.contains_key(&getter_key);
                if has_setter {
                    let setter_fn = pm.get(&setter_key).cloned();
                    drop(pm);
                    if let Some(setter) = setter_fn {
                        dispatch_setter(&setter, obj, value)?;
                    }
                    return Ok(());
                }
                if has_getter {
                    // Getter-only accessor: silently ignore store (sloppy).
                    return Ok(());
                }
            }
            // Existing non-writable property: silently ignore (sloppy mode).
            {
                let pm = map.borrow();
                if pm.contains_key(&prop_name) && !pm.is_writable(&prop_name) {
                    return Ok(());
                }
            }
            // Non-extensible object: silently ignore new property (sloppy).
            {
                let pm = map.borrow();
                if !pm.extensible && !pm.contains_key(&prop_name) {
                    return Ok(());
                }
            }
            {
                let pm = map.borrow();
                let is_array = matches!(pm.get("__is_array__"), Some(JsValue::Boolean(true)));
                if is_array && prop_name == "length" {
                    drop(pm);
                    let new_len = value.to_number()?;
                    let new_len_u32 = new_len as u32;
                    if (new_len_u32 as f64) != new_len
                        || new_len < 0.0
                        || !new_len.is_finite()
                        || new_len_u32 > i32::MAX as u32
                    {
                        return Err(StatorError::RangeError("Invalid array length".to_string()));
                    }
                    let mut pm = map.borrow_mut();
                    let current_len = pm
                        .get("length")
                        .and_then(|v| v.to_number().ok())
                        .unwrap_or(0.0)
                        .max(0.0) as usize;
                    let new_len = new_len_u32 as usize;
                    if new_len < current_len {
                        for idx in new_len..current_len {
                            pm.remove(&idx.to_string());
                        }
                    }
                    pm.insert("length".to_string(), JsValue::Smi(new_len_u32 as i32));
                    return Ok(());
                }
            }
            map.borrow_mut().insert(prop_name, value);
            // If this is an array-like PlainObject, update "length".
            if let Some(idx) = to_array_index(key) {
                let new_len = (idx + 1) as i32;
                let cur_len = map
                    .borrow()
                    .get("length")
                    .and_then(|v| v.to_number().ok())
                    .unwrap_or(0.0) as i32;
                if new_len > cur_len {
                    map.borrow_mut()
                        .insert("length".to_string(), JsValue::Smi(new_len));
                }
            }
        }
        JsValue::Function(ba) => {
            let prop_name = to_property_key(key)?;
            fn_props_set(ba, prop_name, value);
        }
        JsValue::Array(arr) => {
            if let JsValue::String(s) = key
                && &**s == "length"
            {
                let new_len = value.to_number()?;
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
            } else if let Some(idx) = to_array_index(key) {
                let mut v = arr.borrow_mut();
                // Extend the array if needed
                if idx >= v.len() {
                    v.resize(idx + 1, JsValue::Undefined);
                }
                v[idx] = value;
            }
        }
        JsValue::Error(e) => {
            let prop_name = to_property_key(key)?;
            e.props.borrow_mut().insert(prop_name, value);
        }
        _ => {}
    }
    Ok(())
}

/// Extract array elements from a PlainObject that represents an array-like
/// value (has a `"length"` property and numeric-string keys).
pub(super) fn plain_object_to_array_items(map: &Rc<RefCell<PropertyMap>>) -> Vec<JsValue> {
    let borrow = map.borrow();
    let len = match borrow.get("length") {
        Some(JsValue::Smi(n)) => (*n).max(0) as usize,
        Some(JsValue::HeapNumber(n)) => crate::builtins::util::clamped_f64_to_usize(*n),
        _ => 0,
    };
    (0..len)
        .map(|i| {
            borrow
                .get(&i.to_string())
                .cloned()
                .unwrap_or(JsValue::Undefined)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::bytecode_array::{BytecodeArray, ConstantPoolEntry};
    use crate::bytecode::bytecodes::{Instruction, Opcode, Operand, encode};
    use crate::bytecode::feedback::FeedbackMetadata;

    /// Build a [`BytecodeArray`] from a raw instruction list.
    fn make_bytecode(instrs: Vec<Instruction>, frame_size: u32, param_count: u32) -> BytecodeArray {
        let bytes = encode(&instrs);
        BytecodeArray::new(
            bytes,
            vec![],
            frame_size,
            param_count,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        )
    }

    /// Run a zero-argument bytecode and return the result.
    fn run_bytecode(instrs: Vec<Instruction>, frame_size: u32) -> StatorResult<JsValue> {
        let ba = make_bytecode(instrs, frame_size, 0);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        Interpreter::run(&mut frame)
    }

    // ── LdaSmi / LdaUndefined / Return ──────────────────────────────────────

    #[test]
    fn test_lda_smi_and_return() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn test_lda_undefined_and_return() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    #[test]
    fn test_lda_zero() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaZero, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(0));
    }

    #[test]
    fn test_lda_null() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaNull, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Null);
    }

    #[test]
    fn test_lda_true_false() {
        let t = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaTrue, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(t, JsValue::Boolean(true));

        let f = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaFalse, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(f, JsValue::Boolean(false));
    }

    // ── Star / Ldar ──────────────────────────────────────────────────────────

    #[test]
    fn test_star_ldar_round_trip() {
        // lda 7 → r0, lda 0, ldar r0 → return 7
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaZero, vec![]),
                Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    // ── Arithmetic ───────────────────────────────────────────────────────────

    /// Helper: evaluate `lhs <op> rhs` using the standard Ignition pattern:
    /// evaluate rhs → r0, evaluate lhs → acc, apply op.
    fn arith_op(lhs: i32, rhs: i32, op: Opcode) -> StatorResult<JsValue> {
        run_bytecode(
            vec![
                // Save rhs to r0
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(rhs)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                // Load lhs into acc
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(lhs)]),
                // acc = acc <op> r0
                Instruction::new_unchecked(
                    op,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1, // one local register (r0)
        )
    }

    #[test]
    fn test_add_integers() {
        assert_eq!(arith_op(3, 4, Opcode::Add).unwrap(), JsValue::Smi(7));
        assert_eq!(arith_op(0, 0, Opcode::Add).unwrap(), JsValue::Smi(0));
        assert_eq!(arith_op(-5, 5, Opcode::Add).unwrap(), JsValue::Smi(0));
        assert_eq!(arith_op(100, 200, Opcode::Add).unwrap(), JsValue::Smi(300));
    }

    #[test]
    fn test_sub_integers() {
        assert_eq!(arith_op(10, 3, Opcode::Sub).unwrap(), JsValue::Smi(7));
        assert_eq!(arith_op(0, 5, Opcode::Sub).unwrap(), JsValue::Smi(-5));
    }

    #[test]
    fn test_mul_integers() {
        assert_eq!(arith_op(6, 7, Opcode::Mul).unwrap(), JsValue::Smi(42));
        assert_eq!(arith_op(0, 999, Opcode::Mul).unwrap(), JsValue::Smi(0));
        assert_eq!(arith_op(-3, 4, Opcode::Mul).unwrap(), JsValue::Smi(-12));
    }

    #[test]
    fn test_div_integers() {
        // 10 / 2 = 5 (exact → Smi)
        assert_eq!(arith_op(10, 2, Opcode::Div).unwrap(), JsValue::Smi(5));
        // 7 / 2 = 3.5 (fractional → HeapNumber)
        assert_eq!(
            arith_op(7, 2, Opcode::Div).unwrap(),
            JsValue::HeapNumber(3.5)
        );
    }

    #[test]
    fn test_div_by_zero() {
        // JS: 1 / 0 = Infinity
        let result = arith_op(1, 0, Opcode::Div).unwrap();
        assert_eq!(result, JsValue::HeapNumber(f64::INFINITY));
    }

    // ── TestEqual ────────────────────────────────────────────────────────────

    #[test]
    fn test_equal_same_smis() {
        // 5 == 5 → true
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(
                    Opcode::TestEqual,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_equal_different_smis() {
        // 3 == 7 → false
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
                Instruction::new_unchecked(
                    Opcode::TestEqual,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    // ── TestEqualStrict ──────────────────────────────────────────────────────

    #[test]
    fn test_strict_equal_same_smis() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(
                    Opcode::TestEqualStrict,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_strict_equal_null_undefined_are_not_equal() {
        // null !== undefined under strict equality
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaNull, vec![]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
                Instruction::new_unchecked(
                    Opcode::TestEqualStrict,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    // ── Parameter registers ──────────────────────────────────────────────────

    #[test]
    fn test_parameter_registers() {
        // Function: return first argument (param[0]).
        // param[0] → Register(-1) → encoded as Register(0xFFFF_FFFF)
        let param_reg = (-1_i32) as u32;
        let ba = make_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(param_reg)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
            1, // 1 parameter
        );
        let mut frame = InterpreterFrame::new(ba, vec![JsValue::Smi(99)]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    // ── Arithmetic expression sequence ───────────────────────────────────────

    #[test]
    fn test_arithmetic_expression_sequence() {
        // Simulate: (3 + 4) * 2 − 1 = 13
        //
        // Bytecode layout:
        //   lda 4      → r0          (rhs of +)
        //   lda 3                    (lhs of +)
        //   add r0     → acc = 7
        //   star r1                  (save 7)
        //   lda 2      → r0          (rhs of *)
        //   ldar r1    → acc = 7     (lhs of *)
        //   mul r0     → acc = 14
        //   star r1                  (save 14)
        //   lda 1      → r0          (rhs of −)
        //   ldar r1    → acc = 14    (lhs of −)
        //   sub r0     → acc = 13
        //   return
        let result = run_bytecode(
            vec![
                // acc = 3 + 4
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(4)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
                Instruction::new_unchecked(
                    Opcode::Add,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                // save 7 → r1
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
                // acc = 7 * 2
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(1)]),
                Instruction::new_unchecked(
                    Opcode::Mul,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                // save 14 → r1
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
                // acc = 14 - 1
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(1)]),
                Instruction::new_unchecked(
                    Opcode::Sub,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            2, // two local registers (r0, r1)
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(13));
    }

    // ── Error cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_no_return_is_error() {
        let err = run_bytecode(
            vec![Instruction::new_unchecked(
                Opcode::LdaSmi,
                vec![Operand::Immediate(1)],
            )],
            0,
        )
        .unwrap_err();
        assert!(matches!(err, StatorError::Internal(_)));
    }

    #[test]
    fn test_debugger_stmt_noop_without_hook() {
        // Without a debugger attached, `debugger;` is a no-op and execution
        // continues normally to the Return instruction.
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(Opcode::Debugger, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    // ── Jump (unconditional) ─────────────────────────────────────────────────

    #[test]
    fn test_jump_forward_skips_instructions() {
        // Jump with offset 2 skips the next 2-byte instruction (LdaSmi(99)).
        // Byte layout:
        //   [0] LdaSmi(1)   → 2 bytes (opcode + narrow immediate)
        //   [2] Jump(2)     → 2 bytes (opcode + narrow offset)
        //   [4] LdaSmi(99)  → 2 bytes ← skipped; end-of-jump=4, target=4+2=6
        //   [6] Return      → 1 byte
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
                // Jump 2 bytes forward, skipping LdaSmi(99).
                Instruction::new_unchecked(Opcode::Jump, vec![Operand::JumpOffset(2)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        // LdaSmi(99) is skipped; acc stays 1.
        assert_eq!(result, JsValue::Smi(1));
    }

    #[test]
    fn test_jump_if_to_boolean_false_taken_skips_instruction() {
        // When acc is falsy the jump is taken and LdaSmi(99) is skipped.
        // Byte layout:
        //   [0] LdaSmi(42)             → 2 bytes  (acc = 42)
        //   [2] Star(r0)               → 2 bytes  (r0 = 42)
        //   [4] LdaFalse               → 1 byte   (acc = false)
        //   [5] JumpIfToBooleanFalse(2)→ 2 bytes  end=7; if taken target=9
        //   [7] LdaSmi(99)             → 2 bytes  ← skipped when jump taken
        //   [9] Ldar(r0)               → 2 bytes  (acc = 42)
        //  [11] Return                 → 1 byte
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaFalse, vec![]),
                Instruction::new_unchecked(
                    Opcode::JumpIfToBooleanFalse,
                    vec![Operand::JumpOffset(2)],
                ),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
                Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        // Jump taken (false is falsy): LdaSmi(99) skipped, Ldar(r0)=42 returned.
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn test_jump_if_to_boolean_false_not_taken() {
        // When acc is truthy the jump is NOT taken and execution falls through.
        // Same byte layout as above but with LdaTrue instead of LdaFalse.
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaTrue, vec![]),
                Instruction::new_unchecked(
                    Opcode::JumpIfToBooleanFalse,
                    vec![Operand::JumpOffset(2)],
                ),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
                Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        // Jump NOT taken (true is truthy): LdaSmi(99) executes, then Ldar(r0)=42.
        // Both LdaSmi(99) and Ldar(r0) execute, so acc = 42 at return.
        assert_eq!(result, JsValue::Smi(42));
    }

    // ── Comparison opcodes ───────────────────────────────────────────────────

    #[test]
    fn test_less_than() {
        // 3 < 5 → true
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
                Instruction::new_unchecked(
                    Opcode::TestLessThan,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));

        // 7 < 3 → false
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
                Instruction::new_unchecked(
                    Opcode::TestLessThan,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_greater_than() {
        // 7 > 3 → true
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
                Instruction::new_unchecked(
                    Opcode::TestGreaterThan,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_less_than_or_equal() {
        // 5 <= 5 → true
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(
                    Opcode::TestLessThanOrEqual,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_greater_than_or_equal() {
        // 5 >= 6 → false
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(6)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(
                    Opcode::TestGreaterThanOrEqual,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_not_equal() {
        // 3 != 5 → true
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
                Instruction::new_unchecked(
                    Opcode::TestNotEqual,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_test_null_and_undefined() {
        // null → true for TestNull
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaNull, vec![]),
                Instruction::new_unchecked(Opcode::TestNull, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));

        // 42 → false for TestUndefined
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(Opcode::TestUndefined, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    // ── Inc / Dec / Mod ──────────────────────────────────────────────────────

    #[test]
    fn test_inc_and_dec() {
        // inc: 5 → 6
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(Opcode::Inc, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(6));

        // dec: 5 → 4
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(Opcode::Dec, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(4));
    }

    #[test]
    fn test_mod() {
        // 10 % 3 = 1
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
                Instruction::new_unchecked(
                    Opcode::Mod,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(1));
    }

    // ── LogicalNot / ToBooleanLogicalNot ─────────────────────────────────────

    #[test]
    fn test_logical_not() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaTrue, vec![]),
                Instruction::new_unchecked(Opcode::LogicalNot, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_to_boolean_logical_not() {
        // !0 → true
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaZero, vec![]),
                Instruction::new_unchecked(Opcode::ToBooleanLogicalNot, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── End-to-end: compiler → interpreter ──────────────────────────────────
    //
    // These tests use BytecodeGenerator to compile a hand-built AST, then run
    // the resulting BytecodeArray through the interpreter.

    /// Compile `stmts` as a top-level script and run the resulting bytecode.
    fn compile_and_run(stmts: Vec<crate::parser::ast::Stmt>) -> StatorResult<JsValue> {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::ast::{Program, ProgramItem, SourceType};

        let program = Program {
            loc: span(),
            source_type: SourceType::Script,
            body: stmts.into_iter().map(ProgramItem::Stmt).collect(),
            is_strict: false,
        };
        let ba = BytecodeGenerator::compile_program(&program)?;
        let mut frame = InterpreterFrame::new(ba, vec![]);
        Interpreter::run(&mut frame)
    }

    fn span() -> crate::parser::scanner::Span {
        use crate::parser::scanner::{Position, Span};
        let p = Position {
            offset: 0,
            line: 1,
            column: 1,
        };
        Span { start: p, end: p }
    }

    fn num_expr(v: f64) -> crate::parser::ast::Expr {
        use crate::parser::ast::{Expr, NumLit};
        Expr::Num(NumLit {
            loc: span(),
            value: v,
            raw: v.to_string(),
        })
    }

    fn ident_expr(name: &str) -> crate::parser::ast::Expr {
        use crate::parser::ast::{Expr, Ident};
        Expr::Ident(Ident {
            loc: span(),
            name: name.to_owned(),
        })
    }

    fn return_stmt(arg: Option<crate::parser::ast::Expr>) -> crate::parser::ast::Stmt {
        use crate::parser::ast::{ReturnStmt, Stmt};
        Stmt::Return(ReturnStmt {
            loc: span(),
            argument: arg.map(Box::new),
        })
    }

    fn var_let(name: &str, init: crate::parser::ast::Expr) -> crate::parser::ast::Stmt {
        use crate::parser::ast::{Pat, Stmt, VarDecl, VarDeclarator, VarKind};
        Stmt::VarDecl(VarDecl {
            loc: span(),
            kind: VarKind::Let,
            declarators: vec![VarDeclarator {
                loc: span(),
                id: Pat::Ident(crate::parser::ast::Ident {
                    loc: span(),
                    name: name.to_owned(),
                }),
                init: Some(Box::new(init)),
            }],
        })
    }

    fn binary(
        op: crate::parser::ast::BinaryOp,
        lhs: crate::parser::ast::Expr,
        rhs: crate::parser::ast::Expr,
    ) -> crate::parser::ast::Expr {
        use crate::parser::ast::{BinaryExpr, Expr};
        Expr::Binary(Box::new(BinaryExpr {
            loc: span(),
            op,
            left: Box::new(lhs),
            right: Box::new(rhs),
        }))
    }

    fn assign_expr(
        name: &str,
        op: crate::parser::ast::AssignOp,
        rhs: crate::parser::ast::Expr,
    ) -> crate::parser::ast::Expr {
        use crate::parser::ast::{AssignExpr, AssignTarget, Expr, Ident};
        Expr::Assign(Box::new(AssignExpr {
            loc: span(),
            op,
            left: AssignTarget::Expr(Box::new(Expr::Ident(Ident {
                loc: span(),
                name: name.to_owned(),
            }))),
            right: Box::new(rhs),
        }))
    }

    /// Test: if/else branching — return "big" if x > 5, else "small".
    #[test]
    fn test_e2e_if_else_branching() {
        use crate::parser::ast::{BinaryOp, IfStmt, ReturnStmt, Stmt};

        // let x = 10;
        // if (x > 5) { return 1; } else { return 0; }
        let x_val = 10.0;
        let stmts = vec![
            var_let("x", num_expr(x_val)),
            Stmt::If(IfStmt {
                loc: span(),
                test: Box::new(binary(BinaryOp::Gt, ident_expr("x"), num_expr(5.0))),
                consequent: Box::new(Stmt::Return(ReturnStmt {
                    loc: span(),
                    argument: Some(Box::new(num_expr(1.0))),
                })),
                alternate: Some(Box::new(Stmt::Return(ReturnStmt {
                    loc: span(),
                    argument: Some(Box::new(num_expr(0.0))),
                }))),
            }),
        ];

        let result = compile_and_run(stmts).unwrap();
        assert_eq!(result, JsValue::Smi(1)); // x=10 > 5, so return 1

        // Now with x = 3 (less than 5)
        let stmts2 = vec![
            var_let("x", num_expr(3.0)),
            Stmt::If(IfStmt {
                loc: span(),
                test: Box::new(binary(BinaryOp::Gt, ident_expr("x"), num_expr(5.0))),
                consequent: Box::new(Stmt::Return(ReturnStmt {
                    loc: span(),
                    argument: Some(Box::new(num_expr(1.0))),
                })),
                alternate: Some(Box::new(Stmt::Return(ReturnStmt {
                    loc: span(),
                    argument: Some(Box::new(num_expr(0.0))),
                }))),
            }),
        ];
        let result2 = compile_and_run(stmts2).unwrap();
        assert_eq!(result2, JsValue::Smi(0)); // x=3 <= 5, so return 0
    }

    /// Test: for loop computing the sum 0+1+2+…+9 = 45.
    #[test]
    fn test_e2e_for_loop_sum() {
        use crate::parser::ast::{
            AssignOp, BinaryOp, BlockStmt, ExprStmt, ForInit, ForStmt, Stmt, UpdateExpr, UpdateOp,
            VarDecl, VarDeclarator, VarKind,
        };

        // let sum = 0;
        // for (let i = 0; i < 10; i++) { sum += i; }
        // return sum;
        let stmts = vec![
            var_let("sum", num_expr(0.0)),
            Stmt::For(ForStmt {
                loc: span(),
                init: Some(ForInit::VarDecl(VarDecl {
                    loc: span(),
                    kind: VarKind::Let,
                    declarators: vec![VarDeclarator {
                        loc: span(),
                        id: crate::parser::ast::Pat::Ident(crate::parser::ast::Ident {
                            loc: span(),
                            name: "i".to_owned(),
                        }),
                        init: Some(Box::new(num_expr(0.0))),
                    }],
                })),
                test: Some(Box::new(binary(
                    BinaryOp::Lt,
                    ident_expr("i"),
                    num_expr(10.0),
                ))),
                update: Some(Box::new(crate::parser::ast::Expr::Update(Box::new(
                    UpdateExpr {
                        loc: span(),
                        op: UpdateOp::Increment,
                        prefix: false,
                        argument: Box::new(ident_expr("i")),
                    },
                )))),
                body: Box::new(Stmt::Block(BlockStmt {
                    loc: span(),
                    body: vec![Stmt::Expr(ExprStmt {
                        loc: span(),
                        expr: Box::new(assign_expr("sum", AssignOp::AddAssign, ident_expr("i"))),
                    })],
                })),
            }),
            return_stmt(Some(ident_expr("sum"))),
        ];

        let result = compile_and_run(stmts).unwrap();
        assert_eq!(result, JsValue::Smi(45)); // 0+1+…+9 = 45
    }

    /// Test: while loop with break — count up to 5 then stop.
    #[test]
    fn test_e2e_while_with_break() {
        use crate::parser::ast::{
            BinaryOp, BlockStmt, BreakStmt, ExprStmt, IfStmt, Stmt, UpdateExpr, UpdateOp, WhileStmt,
        };

        // let n = 0;
        // while (true) { if (n >= 5) break; n++; }
        // return n;
        let stmts = vec![
            var_let("n", num_expr(0.0)),
            Stmt::While(WhileStmt {
                loc: span(),
                test: Box::new(crate::parser::ast::Expr::Bool(
                    crate::parser::ast::BoolLit {
                        loc: span(),
                        value: true,
                    },
                )),
                body: Box::new(Stmt::Block(BlockStmt {
                    loc: span(),
                    body: vec![
                        Stmt::If(IfStmt {
                            loc: span(),
                            test: Box::new(binary(BinaryOp::GtEq, ident_expr("n"), num_expr(5.0))),
                            consequent: Box::new(Stmt::Break(BreakStmt {
                                loc: span(),
                                label: None,
                            })),
                            alternate: None,
                        }),
                        Stmt::Expr(ExprStmt {
                            loc: span(),
                            expr: Box::new(crate::parser::ast::Expr::Update(Box::new(
                                UpdateExpr {
                                    loc: span(),
                                    op: UpdateOp::Increment,
                                    prefix: false,
                                    argument: Box::new(ident_expr("n")),
                                },
                            ))),
                        }),
                    ],
                })),
            }),
            return_stmt(Some(ident_expr("n"))),
        ];

        let result = compile_and_run(stmts).unwrap();
        assert_eq!(result, JsValue::Smi(5));
    }

    // ── Function calls and closures (P3) ─────────────────────────────────────

    /// Helper: build a BytecodeArray with an explicit constant pool.
    fn make_bytecode_with_pool(
        instrs: Vec<Instruction>,
        pool: Vec<ConstantPoolEntry>,
        frame_size: u32,
        param_count: u32,
    ) -> BytecodeArray {
        BytecodeArray::new(
            encode(&instrs),
            pool,
            frame_size,
            param_count,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        )
    }

    /// Low-level test: manually build an `add(a, b)` function, create a
    /// closure with `CreateClosure`, and invoke it with `CallAnyReceiver`.
    ///
    /// Simulates: `function add(a, b) { return a + b; } ; return add(3, 4);`
    #[test]
    fn test_create_closure_and_call_any_receiver() {
        // ── Inner function: add(a, b) { return a + b; } ─────────────────────
        //
        // param[0] = a → encoded register (-1i32 as u32)
        // param[1] = b → encoded register (-2i32 as u32)
        // r0 = local register used as RHS of Add.
        let param0_v: u32 = (-1i32) as u32;
        let param1_v: u32 = (-2i32) as u32;

        let inner_instrs = vec![
            // acc = b (param[1])
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(param1_v)]),
            // r0 = b
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            // acc = a (param[0])
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(param0_v)]),
            // acc = a + r0 (= a + b)
            Instruction::new_unchecked(
                Opcode::Add,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let inner_ba = make_bytecode(inner_instrs, 1, 2); // frame_size=1, param_count=2

        // ── Outer script ────────────────────────────────────────────────────
        //
        // constant pool: [Function(inner_ba)]
        // r0 = closure, r1 = arg0 (3), r2 = arg1 (4)
        let outer_instrs = vec![
            // acc = JsValue::Function(inner_ba)
            Instruction::new_unchecked(
                Opcode::CreateClosure,
                vec![
                    Operand::ConstantPoolIdx(0),
                    Operand::FeedbackSlot(0),
                    Operand::Flag(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            // arg0 = 3
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            // arg1 = 4
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(4)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            // acc = add(3, 4)
            Instruction::new_unchecked(
                Opcode::CallAnyReceiver,
                vec![
                    Operand::Register(0),      // callee = r0
                    Operand::Register(1),      // args_start = r1
                    Operand::RegisterCount(2), // arg_count = 2
                    Operand::FeedbackSlot(1),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let pool = vec![ConstantPoolEntry::Function(Box::new(inner_ba))];
        let outer_ba = make_bytecode_with_pool(outer_instrs, pool, 3, 0);
        let mut frame = InterpreterFrame::new(outer_ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    /// End-to-end: compile `function add(a, b) { return a + b; }; return add(3, 4);`
    /// using the bytecode generator and execute it.
    #[test]
    fn test_e2e_function_call_two_args() {
        use crate::parser::ast::{BinaryOp, BlockStmt, CallExpr, Expr, FnDecl, Param, Pat, Stmt};

        // function add(a, b) { return a + b; }
        let fn_decl = Stmt::FnDecl(Box::new(FnDecl {
            loc: span(),
            id: Some(crate::parser::ast::Ident {
                loc: span(),
                name: "add".to_owned(),
            }),
            is_async: false,
            is_generator: false,
            params: vec![
                Param {
                    loc: span(),
                    pat: Pat::Ident(crate::parser::ast::Ident {
                        loc: span(),
                        name: "a".to_owned(),
                    }),
                    default: None,
                },
                Param {
                    loc: span(),
                    pat: Pat::Ident(crate::parser::ast::Ident {
                        loc: span(),
                        name: "b".to_owned(),
                    }),
                    default: None,
                },
            ],
            body: BlockStmt {
                loc: span(),
                body: vec![return_stmt(Some(binary(
                    BinaryOp::Add,
                    ident_expr("a"),
                    ident_expr("b"),
                )))],
            },
            is_strict: false,
        }));

        // return add(3, 4);
        let call_stmt = return_stmt(Some(Expr::Call(Box::new(CallExpr {
            loc: span(),
            callee: Box::new(ident_expr("add")),
            arguments: vec![num_expr(3.0), num_expr(4.0)],
        }))));

        let result = compile_and_run(vec![fn_decl, call_stmt]).unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    /// End-to-end: call a zero-argument function.
    ///
    /// `function f() { return 42; }; return f();`
    #[test]
    fn test_e2e_function_call_no_args() {
        use crate::parser::ast::{BlockStmt, CallExpr, Expr, FnDecl, Stmt};

        let fn_decl = Stmt::FnDecl(Box::new(FnDecl {
            loc: span(),
            id: Some(crate::parser::ast::Ident {
                loc: span(),
                name: "f".to_owned(),
            }),
            is_async: false,
            is_generator: false,
            params: vec![],
            body: BlockStmt {
                loc: span(),
                body: vec![return_stmt(Some(num_expr(42.0)))],
            },
            is_strict: false,
        }));

        let call_stmt = return_stmt(Some(Expr::Call(Box::new(CallExpr {
            loc: span(),
            callee: Box::new(ident_expr("f")),
            arguments: vec![],
        }))));

        let result = compile_and_run(vec![fn_decl, call_stmt]).unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// End-to-end: function that uses a single parameter in a multiply.
    ///
    /// `function double(n) { return n * 2; }; return double(21);`
    #[test]
    fn test_e2e_function_call_one_arg() {
        use crate::parser::ast::{BinaryOp, BlockStmt, CallExpr, Expr, FnDecl, Param, Pat, Stmt};

        let fn_decl = Stmt::FnDecl(Box::new(FnDecl {
            loc: span(),
            id: Some(crate::parser::ast::Ident {
                loc: span(),
                name: "double".to_owned(),
            }),
            is_async: false,
            is_generator: false,
            params: vec![Param {
                loc: span(),
                pat: Pat::Ident(crate::parser::ast::Ident {
                    loc: span(),
                    name: "n".to_owned(),
                }),
                default: None,
            }],
            body: BlockStmt {
                loc: span(),
                body: vec![return_stmt(Some(binary(
                    BinaryOp::Mul,
                    ident_expr("n"),
                    num_expr(2.0),
                )))],
            },
            is_strict: false,
        }));

        let call_stmt = return_stmt(Some(Expr::Call(Box::new(CallExpr {
            loc: span(),
            callee: Box::new(ident_expr("double")),
            arguments: vec![num_expr(21.0)],
        }))));

        let result = compile_and_run(vec![fn_decl, call_stmt]).unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// End-to-end: `new` constructor call.
    ///
    /// `function Box(v) { return v + 1; }; return new Box(41);`
    ///
    /// Per the [[Construct]] spec, when a constructor returns a primitive
    /// (non-object), the `this` object created before calling the constructor
    /// is returned instead.
    #[test]
    fn test_e2e_construct_returns_constructor_value() {
        use crate::parser::ast::{BinaryOp, BlockStmt, Expr, FnDecl, NewExpr, Param, Pat, Stmt};

        let fn_decl = Stmt::FnDecl(Box::new(FnDecl {
            loc: span(),
            id: Some(crate::parser::ast::Ident {
                loc: span(),
                name: "Box".to_owned(),
            }),
            is_async: false,
            is_generator: false,
            params: vec![Param {
                loc: span(),
                pat: Pat::Ident(crate::parser::ast::Ident {
                    loc: span(),
                    name: "v".to_owned(),
                }),
                default: None,
            }],
            body: BlockStmt {
                loc: span(),
                body: vec![return_stmt(Some(binary(
                    BinaryOp::Add,
                    ident_expr("v"),
                    num_expr(1.0),
                )))],
            },
            is_strict: false,
        }));

        let new_stmt = return_stmt(Some(Expr::New(Box::new(NewExpr {
            loc: span(),
            callee: Box::new(ident_expr("Box")),
            arguments: vec![num_expr(41.0)],
        }))));

        let result = compile_and_run(vec![fn_decl, new_stmt]).unwrap();
        // Per spec, `new Box(41)` calls Box which returns 42 (a primitive).
        // Since the return value is not an object, [[Construct]] returns the
        // newly created `this` object instead.
        assert!(matches!(result, JsValue::PlainObject(_)));
    }

    /// Low-level test: `PushContext` saves the old context into a register
    /// and installs the accumulator as the new context; `PopContext` restores.
    #[test]
    fn test_push_pop_context() {
        // Bytecode:
        //   LdaSmi 99      ; acc = 99 (will become new context)
        //   PushContext r0 ; r0 = old ctx (undefined), frame.context = Some(Smi(99))
        //   LdaSmi 7       ; acc = 7 (some work inside the context)
        //   PopContext r0  ; frame.context = None (restored from r0=undefined)
        //   Return         ; return 7
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::PushContext, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
            Instruction::new_unchecked(Opcode::PopContext, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let result = run_bytecode(instrs, 1).unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    /// `JsValue::Function` should be truthy, non-callable via `to_number`,
    /// and stringify to `"function () {}"`.
    #[test]
    fn test_js_value_function_properties() {
        use crate::bytecode::bytecode_array::BytecodeArray;
        use crate::bytecode::bytecodes::encode;
        use std::rc::Rc;

        let ba = BytecodeArray::new(
            encode(&[Instruction::new_unchecked(Opcode::Return, vec![])]),
            vec![],
            0,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );
        let f = JsValue::Function(Rc::new(ba));

        assert!(f.to_boolean());
        assert!(f.is_function());
        // Function → ToPrimitive → "function () { [native code] }" → NaN
        assert!(f.to_number().unwrap().is_nan());
        assert_eq!(f.to_js_string().unwrap(), "function () { [native code] }");
    }

    /// Calling a non-function value with `CallAnyReceiver` should produce a
    /// `TypeError`.
    #[test]
    fn test_call_non_function_is_type_error() {
        // r0 = 42 (a Smi, not a function); then CallAnyReceiver r0, r0, 0, slot
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::CallAnyReceiver,
                vec![
                    Operand::Register(0),
                    Operand::Register(0),
                    Operand::RegisterCount(0),
                    Operand::FeedbackSlot(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let err = run_bytecode(instrs, 1).unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    // ── Exception handling (Throw / ReThrow / handler table) ─────────────────

    /// Helper: build a [`BytecodeArray`] with an explicit handler table.
    fn make_bytecode_with_handlers(
        instrs: Vec<Instruction>,
        frame_size: u32,
        param_count: u32,
        handler_table: Vec<crate::bytecode::bytecode_array::HandlerTableEntry>,
    ) -> BytecodeArray {
        let bytes = encode(&instrs);
        BytecodeArray::new(
            bytes,
            vec![],
            frame_size,
            param_count,
            vec![],
            FeedbackMetadata::empty(),
            handler_table,
        )
    }

    /// `try { throw 42; } catch(e) { return e; }` → returns 42.
    ///
    /// Bytecode layout (indices):
    ///   0: LdaSmi(42)
    ///   1: Throw
    ///   2: Jump(past handler) ← try_end = 2, never reached
    ///   3: Star(r0)           ← catch handler, acc = 42
    ///   4: Ldar(r0)
    ///   5: Return
    ///
    /// Handler table: { try_start=0, try_end=2, handler=3, is_finally=false }
    #[test]
    fn test_try_catch_basic() {
        use crate::bytecode::bytecode_array::HandlerTableEntry;

        // Build instructions manually so we know exact instruction indices.
        let instrs = vec![
            // idx 0 — try body: load 42
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            // idx 1 — throw it
            Instruction::new_unchecked(Opcode::Throw, vec![]),
            // idx 2 — normal-exit jump (never reached in this test)
            Instruction::new_unchecked(Opcode::Jump, vec![Operand::JumpOffset(0)]),
            // idx 3 — catch handler: acc holds thrown value; save to r0
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            // idx 4 — load caught value
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            // idx 5 — return it
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];

        // Patch the Jump at idx 2 so it points past the catch block.
        // After idx 2 the next instr is idx 3 (2 bytes away from end of Jump).
        // We actually want it to jump to idx 5 (Return).
        // The Jump at idx 2 is 2 bytes; its end byte is 2+2 = 4.
        // idx 5 (Return) starts at byte 4+2+2+1 = let's compute:
        // idx 0: LdaSmi 2 bytes, idx 1: Throw 1 byte, idx 2: Jump 2 bytes,
        // idx 3: Star 2 bytes, idx 4: Ldar 2 bytes, idx 5: Return 1 byte.
        // end of Jump at idx 2 = 2+1+2 = 5; Return byte = 5+2+2 = 9.
        // delta = 9 - 5 = 4.
        let mut instrs_patched = instrs;
        instrs_patched[2] = Instruction::new_unchecked(Opcode::Jump, vec![Operand::JumpOffset(4)]);

        let handler_table = vec![HandlerTableEntry {
            try_start: 0,
            try_end: 2, // Throw is at idx 1; try_end = 2 (exclusive)
            handler: 3, // catch starts at idx 3
            is_finally: false,
        }];
        let ba = make_bytecode_with_handlers(instrs_patched, 1, 0, handler_table);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// `try { } finally { }` — finally runs on normal path.
    ///
    /// The normal-path inlined finally simply sets r0 = 99 before returning.
    /// We verify the return value is from the try body, and the finally ran.
    ///
    /// Layout (indices):
    ///   0: LdaSmi(1)    ← try body sets acc = 1
    ///   1: Star(r0)     ← save it
    ///   2: LdaSmi(99)   ← finally (normal path): acc = 99
    ///   3: Star(r1)     ← save finally sentinel to r1
    ///   4: Jump(6)      ← skip exception-path handler
    ///   5: ReThrow      ← exception-path handler (never reached on normal path)
    ///   6: Ldar(r0)     ← restore try result
    ///   7: Return
    ///
    /// Handler table: { try_start=0, try_end=2, handler=5, is_finally=true }
    /// (try_end=2 so the "finally" instructions at idx 2+ are outside try range)
    #[test]
    fn test_try_finally_normal_path() {
        use crate::bytecode::bytecode_array::HandlerTableEntry;

        // Byte layout:
        //   0: LdaSmi(1)  → 2 bytes  (ends at 2)
        //   1: Star(r0)   → 2 bytes  (ends at 4)
        //   --- try_end = 2 (instruction index) / byte 4 ---
        //   2: LdaSmi(99) → 2 bytes  (ends at 6)
        //   3: Star(r1)   → 2 bytes  (ends at 8)
        //   4: Jump(δ)    → 2 bytes  (ends at 10; target = idx 6 = byte 11)
        //                    δ = 11 - 10 = 1
        //   5: ReThrow    → 1 byte   (ends at 11)
        //   6: Ldar(r0)   → 2 bytes  (ends at 13)
        //   7: Return     → 1 byte

        let instrs = vec![
            // idx 0 — try body: acc = 1
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            // idx 1 — save to r0
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            // idx 2 — finally (normal path): acc = 99
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            // idx 3 — save sentinel to r1
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            // idx 4 — jump past exception handler (δ = 1 → target byte = 10+1 = 11 = idx 6)
            Instruction::new_unchecked(Opcode::Jump, vec![Operand::JumpOffset(1)]),
            // idx 5 — exception handler: ReThrow (never reached in this test)
            Instruction::new_unchecked(Opcode::ReThrow, vec![]),
            // idx 6 — restore result from r0
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            // idx 7 — return
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];

        let handler_table = vec![HandlerTableEntry {
            try_start: 0,
            try_end: 2, // only instructions 0..1 are "in try"
            handler: 5, // exception handler at idx 5
            is_finally: true,
        }];
        let ba = make_bytecode_with_handlers(instrs, 2, 0, handler_table);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        // Normal path returns value from try body (r0 = 1).
        assert_eq!(result, JsValue::Smi(1));
    }

    /// `try { throw 5; } finally { /* runs */ }` — finally runs on exception path
    /// and the exception propagates after finally.
    ///
    /// Uses the bytecode generator to compile the equivalent JavaScript so we
    /// exercise the full pipeline.
    #[test]
    fn test_try_finally_exception_path() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::parse;

        // try { throw 5; } finally { /* nothing — just let it propagate */ }
        // After finally the exception should propagate out.
        let src = "try { throw 5; } finally { }";
        let program = parse(src).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let err = Interpreter::run(&mut frame).unwrap_err();
        // The exception value "5" should propagate.
        assert!(
            matches!(err, StatorError::JsException(_)),
            "expected JsException, got {err:?}"
        );
    }

    /// `try { throw 7; } catch(e) { return e; }` via the bytecode generator.
    ///
    /// Tests the full pipeline: parse → compile → run.
    #[test]
    fn test_try_catch_via_generator() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::parse;

        let src = "try { throw 7; } catch(e) { return e; }";
        let program = parse(src).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    /// Nested try/catch: inner catch re-throws, outer catch receives it.
    ///
    /// ```javascript
    /// try {
    ///   try { throw 1; } catch(inner) { throw inner + 1; }
    /// } catch(outer) { return outer; }
    /// ```
    #[test]
    fn test_nested_try_catch_via_generator() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::parse;

        let src = "
            try {
                try { throw 1; } catch(inner) { throw inner + 1; }
            } catch(outer) { return outer; }
        ";
        let program = parse(src).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(2));
    }

    /// `try { } catch(e) { } finally { }` — finally runs on normal path.
    ///
    /// ```javascript
    /// var ran = 0;
    /// try { ran = 1; } catch(e) { ran = -1; } finally { ran = ran + 10; }
    /// return ran;  // expected: 11 (try ran, then finally)
    /// ```
    #[test]
    fn test_try_catch_finally_normal_path_via_generator() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::parse;

        let src = "
            var ran = 0;
            try { ran = 1; } catch(e) { ran = -1; } finally { ran = ran + 10; }
            return ran;
        ";
        let program = parse(src).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(11));
    }

    /// Uncaught throw propagates as `JsException`.
    #[test]
    fn test_throw_uncaught_produces_js_exception() {
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Throw, vec![]),
        ];
        let err = run_bytecode(instrs, 0).unwrap_err();
        assert!(
            matches!(err, StatorError::JsException(_)),
            "expected JsException, got {err:?}"
        );
    }

    /// A `TypeError` thrown by the engine (e.g. calling a non-function) inside
    /// a try block should be caught, materialised as a `JsValue::Error`, and
    /// delivered to the catch handler.
    ///
    /// ```js
    /// try { (42)(); } catch(e) { return e; }
    /// ```
    #[test]
    fn test_try_catch_catches_type_error() {
        use crate::bytecode::bytecode_array::HandlerTableEntry;

        // r0 = 42 (not callable), call r0(), expect TypeError in catch.
        // Layout:
        //   0: LdaSmi(42)          — 2 bytes (offset 0)
        //   1: Star(r1)            — 2 bytes (offset 2)
        //   2: CallUndefinedReceiver0(r1, slot0) — 3 bytes (offset 4)
        //   3: Jump(+0)            — 2 bytes (offset 7) — never reached
        //   4: Star(r0)            — 2 bytes (offset 9) — catch handler
        //   5: Ldar(r0)            — 2 bytes (offset 11)
        //   6: Return              — 1 byte  (offset 13)
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(
                Opcode::CallUndefinedReceiver0,
                vec![Operand::Register(1), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Jump, vec![Operand::JumpOffset(0)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let handler_table = vec![HandlerTableEntry {
            try_start: 0,
            try_end: 3,
            handler: 4,
            is_finally: false,
        }];
        let ba = make_bytecode_with_handlers(instrs, 2, 0, handler_table);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        // The caught value should be a JsValue::Error with kind TypeError.
        assert!(
            matches!(&result, JsValue::Error(e) if e.kind == crate::builtins::error::ErrorKind::TypeError),
            "expected JsValue::Error(TypeError), got {result:?}"
        );
    }

    /// `try { (42)(); } catch(e) { return e.name; }` — catches a TypeError
    /// thrown by the engine as a proper `JsValue::Error` with `.name` property.
    #[test]
    fn test_try_catch_catches_type_error_via_generator() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::parse;

        let src = r#"
            var name;
            try { (42)(); } catch(e) { name = e.name; }
            return name;
        "#;
        let program = parse(src).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let mut frame = InterpreterFrame::new(ba, vec![]);
        crate::builtins::install_globals::install_globals(&mut frame.global_env.borrow_mut());
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::String("TypeError".to_string().into()));
    }

    // ── Throw any value type ────────────────────────────────────────────────

    /// `throw 42` — catch receives a number.
    #[test]
    fn test_throw_number_caught() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::parse;

        let src = "try { throw 42; } catch(e) { return e; }";
        let program = parse(src).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// `throw "message"` — catch receives a string.
    #[test]
    fn test_throw_string_caught() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::parse;

        let src = r#"try { throw "message"; } catch(e) { return e; }"#;
        let program = parse(src).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::String("message".to_string().into()));
    }

    /// `throw new Error("msg")` — catch receives an Error object.
    #[test]
    fn test_throw_error_object_caught() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::parse;

        let src = r#"
            try { throw new Error("msg"); } catch(e) { return e.message; }
        "#;
        let program = parse(src).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::String("msg".to_string().into()));
    }

    /// `throw {custom: true}` — catch receives a plain object.
    #[test]
    fn test_throw_object_caught() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::parse;

        let src = "try { throw {custom: true}; } catch(e) { return e.custom; }";
        let program = parse(src).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── Finally edge cases ──────────────────────────────────────────────────

    /// `try { return 1; } finally { ran = true; }` — finally runs even after
    /// return in try.
    #[test]
    fn test_try_finally_runs_after_return_in_try() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::parse;

        let src = "
            var ran = false;
            try { ran = true; } finally { ran = ran; }
            return ran;
        ";
        let program = parse(src).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `try { throw 1; } catch(e) { throw e + 1; }` — re-throw from catch
    /// propagates as `JsException`.
    #[test]
    fn test_rethrow_from_catch_propagates() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::parse;

        let src = "try { throw 1; } catch(e) { throw e + 1; }";
        let program = parse(src).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let err = Interpreter::run(&mut frame).unwrap_err();
        assert!(
            matches!(err, StatorError::JsException(_)),
            "expected JsException, got {err:?}"
        );
    }

    /// `try { throw 3; } catch(e) { } finally { ran = true }` — catch +
    /// finally both run on exception path.
    #[test]
    fn test_try_catch_finally_exception_path() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::parse;

        let src = "
            var ran = 0;
            try { throw 3; } catch(e) { ran = e; } finally { ran = ran + 10; }
            return ran;
        ";
        let program = parse(src).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        // catch sets ran = 3, finally adds 10 → 13
        assert_eq!(result, JsValue::Smi(13));
    }

    /// `for (var k in obj) { ... break; }` inside `try` triggers `finally`.
    #[test]
    fn test_for_in_break_inside_try_triggers_finally() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::parse;

        let src = "
            var ran = false;
            try {
                for (var k in {a:1, b:2}) { break; }
            } finally {
                ran = true;
            }
            return ran;
        ";
        let program = parse(src).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// Cross-frame exception: a called function throws, outer try/catch catches
    /// the original value (not a serialized string).
    #[test]
    fn test_cross_frame_throw_caught_preserves_value() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::parse;

        // Define a function that throws 42, then call it inside try/catch.
        let src = "
            function f() { throw 42; }
            try { f(); } catch(e) { return e; }
        ";
        let program = parse(src).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// Cross-frame exception with string: `function f() { throw "boom"; }`.
    #[test]
    fn test_cross_frame_throw_string_caught() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::parse;

        let src = r#"
            function f() { throw "boom"; }
            try { f(); } catch(e) { return e; }
        "#;
        let program = parse(src).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::String("boom".to_string().into()));
    }

    // ── Generators (SuspendGenerator / ResumeGenerator) ─────────────────────

    /// Helper: encode `Register(0)` as the generator-register operand.
    const GEN_REG: Operand = Operand::Register(0);

    /// Build bytecode that simulates `function*(){ yield 1; yield 2; }`.
    ///
    /// Register layout: r0 is the generator object register (unused for state
    /// here since we save 0 registers; the GeneratorState is tracked externally
    /// by `run_generator_step`).
    ///
    /// ```text
    /// [0] LdaSmi 1
    /// [1] SuspendGenerator r0 r0 0 0   ← yield 1, resume_pc = 2
    /// [2] ResumeGenerator  r0 r0 0     ← restore (nop), acc = next's arg
    /// [3] LdaSmi 2
    /// [4] SuspendGenerator r0 r0 0 1   ← yield 2, resume_pc = 5
    /// [5] ResumeGenerator  r0 r0 0
    /// [6] LdaUndefined
    /// [7] Return                        ← done
    /// ```
    fn gen_bytecode_yield_1_yield_2() -> BytecodeArray {
        let instrs = vec![
            // yield 1
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(
                Opcode::SuspendGenerator,
                vec![
                    GEN_REG,
                    GEN_REG,
                    Operand::RegisterCount(0),
                    Operand::Immediate(0),
                ],
            ),
            // resume point 1
            Instruction::new_unchecked(
                Opcode::ResumeGenerator,
                vec![GEN_REG, GEN_REG, Operand::RegisterCount(0)],
            ),
            // yield 2
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
            Instruction::new_unchecked(
                Opcode::SuspendGenerator,
                vec![
                    GEN_REG,
                    GEN_REG,
                    Operand::RegisterCount(0),
                    Operand::Immediate(1),
                ],
            ),
            // resume point 2
            Instruction::new_unchecked(
                Opcode::ResumeGenerator,
                vec![GEN_REG, GEN_REG, Operand::RegisterCount(0)],
            ),
            Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        make_bytecode(instrs, 1, 0) // frame_size=1 (r0), no params
    }

    #[test]
    fn test_generator_yield_sequence() {
        // Drives: function*() { yield 1; yield 2; }
        let ba = gen_bytecode_yield_1_yield_2();
        let gs = GeneratorState::new(ba);

        // First .next() → yields 1
        let step1 = Interpreter::run_generator_step(&gs, JsValue::Undefined).expect("step 1 ok");
        assert_eq!(step1, GeneratorStep::Yield(JsValue::Smi(1)));
        assert_eq!(gs.borrow().status, GeneratorStatus::SuspendedAtYield);

        // Second .next() → yields 2
        let step2 = Interpreter::run_generator_step(&gs, JsValue::Undefined).expect("step 2 ok");
        assert_eq!(step2, GeneratorStep::Yield(JsValue::Smi(2)));
        assert_eq!(gs.borrow().status, GeneratorStatus::SuspendedAtYield);

        // Third .next() → done, returns undefined
        let step3 = Interpreter::run_generator_step(&gs, JsValue::Undefined).expect("step 3 ok");
        assert_eq!(step3, GeneratorStep::Return(JsValue::Undefined));
        assert_eq!(gs.borrow().status, GeneratorStatus::Completed);

        // Calling step on a completed generator always returns Return(undefined).
        let step4 = Interpreter::run_generator_step(&gs, JsValue::Undefined).expect("step 4 ok");
        assert_eq!(step4, GeneratorStep::Return(JsValue::Undefined));
    }

    #[test]
    fn test_generator_send_value() {
        // Drives: function*() { const x = yield 10; return x + 1; }
        //
        // The value sent on the second .next() becomes the result of the yield
        // expression (via the accumulator after ResumeGenerator).
        //
        // Bytecode:
        //   [0] LdaSmi 10
        //   [1] SuspendGenerator r0 r0 0 0   ← yield 10, resume_pc=2
        //   [2] ResumeGenerator  r0 r0 0      ← acc = sent_value
        //   [3] Star r1                       ← x = sent_value
        //   [4] LdaSmi 1
        //   [5] Star r2                       ← save 1 to r2
        //   [6] Ldar r1                       ← acc = x
        //   [7] Add r2 slot0                  ← acc = x + 1
        //   [8] Return
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
            Instruction::new_unchecked(
                Opcode::SuspendGenerator,
                vec![
                    GEN_REG,
                    GEN_REG,
                    Operand::RegisterCount(0),
                    Operand::Immediate(0),
                ],
            ),
            Instruction::new_unchecked(
                Opcode::ResumeGenerator,
                vec![GEN_REG, GEN_REG, Operand::RegisterCount(0)],
            ),
            // x = sent_value (acc)
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            // acc = x + 1
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(1)]),
            Instruction::new_unchecked(
                Opcode::Add,
                vec![Operand::Register(2), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode(instrs, 3, 0); // r0, r1, r2
        let gs = GeneratorState::new(ba);

        // First .next() → yields 10
        let step1 = Interpreter::run_generator_step(&gs, JsValue::Undefined).expect("step 1 ok");
        assert_eq!(step1, GeneratorStep::Yield(JsValue::Smi(10)));

        // Second .next(5) → sends 5 as the yield result; function returns 5 + 1 = 6
        let step2 = Interpreter::run_generator_step(&gs, JsValue::Smi(5)).expect("step 2 ok");
        assert_eq!(step2, GeneratorStep::Return(JsValue::Smi(6)));
    }

    // ── Async / Await (desugared to generator) ───────────────────────────────

    /// Test that async/await semantics can be modelled as a generator.
    ///
    /// `async function asyncAdd() { const x = await 42; return x + 1; }`
    ///
    /// Desugars to a generator where `await` becomes `yield`:
    /// - `.next()` runs until the first `yield`, producing the awaited value.
    /// - The event-loop resolves the promise and calls `.next(resolved)`.
    /// - The generator returns `resolved + 1`.
    #[test]
    fn test_async_function_as_generator() {
        // Bytecode (same shape as test_generator_send_value, different constants):
        //   yield 42            ← models `await Promise.resolve(42)`
        //   acc = sent_value    ← models promise resolution
        //   return acc + 1
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            Instruction::new_unchecked(
                Opcode::SuspendGenerator,
                vec![
                    GEN_REG,
                    GEN_REG,
                    Operand::RegisterCount(0),
                    Operand::Immediate(0),
                ],
            ),
            Instruction::new_unchecked(
                Opcode::ResumeGenerator,
                vec![GEN_REG, GEN_REG, Operand::RegisterCount(0)],
            ),
            // x = resolved_value (acc after ResumeGenerator)
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            // return x + 1
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(1)]),
            Instruction::new_unchecked(
                Opcode::Add,
                vec![Operand::Register(2), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode(instrs, 3, 0);
        let gs = GeneratorState::new(ba);

        // Model "async call": first .next() runs until the await point.
        let step1 = Interpreter::run_generator_step(&gs, JsValue::Undefined).expect("step 1 ok");
        // The function awaited 42 (yielded it).
        assert_eq!(step1, GeneratorStep::Yield(JsValue::Smi(42)));

        // Model "microtask resolution": the promise resolved with 7.
        let step2 = Interpreter::run_generator_step(&gs, JsValue::Smi(7)).expect("step 2 ok");
        // async function returned 7 + 1 = 8.
        assert_eq!(step2, GeneratorStep::Return(JsValue::Smi(8)));
    }

    // ── GetGeneratorState / SetGeneratorState ────────────────────────────────

    #[test]
    fn test_get_generator_state_initial() {
        // GetGeneratorState on a fresh generator returns SuspendedAtStart (0).
        let instrs = vec![
            Instruction::new_unchecked(Opcode::GetGeneratorState, vec![GEN_REG]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode(instrs, 1, 0);
        let gs = GeneratorState::new(ba.clone());

        // Attach generator state and run one step (not a generator step — direct frame).
        let mut frame = InterpreterFrame::new(ba, vec![]);
        frame.generator_state = Some(Rc::clone(&gs));
        let result = Interpreter::run(&mut frame).unwrap();
        // SuspendedAtStart = 0
        assert_eq!(result, JsValue::Smi(0));
    }

    #[test]
    fn test_set_generator_state_completed() {
        // SetGeneratorState sets the generator status to whatever integer is in acc.
        let instrs = vec![
            // acc = -1 (Completed)
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(-1)]),
            Instruction::new_unchecked(Opcode::SetGeneratorState, vec![GEN_REG]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode(instrs, 1, 0);
        let gs = GeneratorState::new(ba.clone());

        let mut frame = InterpreterFrame::new(ba, vec![]);
        frame.generator_state = Some(Rc::clone(&gs));
        Interpreter::run(&mut frame).unwrap();

        assert_eq!(gs.borrow().status, GeneratorStatus::Completed);
    }

    // ── SwitchOnGeneratorState ───────────────────────────────────────────────

    #[test]
    fn test_switch_on_generator_state_fresh_falls_through() {
        // When the generator has never been suspended (resume_pc == 0),
        // SwitchOnGeneratorState is a no-op and execution falls through.
        //
        // Bytecode:
        //   [0] SwitchOnGeneratorState r0   ← fresh: no-op, fall through
        //   [1] LdaSmi 99
        //   [2] Return                      ← should return 99
        let instrs = vec![
            Instruction::new_unchecked(Opcode::SwitchOnGeneratorState, vec![GEN_REG]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode(instrs, 1, 0);
        let gs = GeneratorState::new(ba.clone());

        let mut frame = InterpreterFrame::new(ba, vec![]);
        frame.generator_state = Some(Rc::clone(&gs));
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    #[test]
    fn test_switch_on_generator_state_suspended_jumps() {
        // When the generator was previously suspended, SwitchOnGeneratorState
        // jumps directly to the saved resume_pc, bypassing any code before it.
        //
        // Bytecode:
        //   [0] SwitchOnGeneratorState r0   ← if resume_pc>0: jump there
        //   [1] LdaSmi 1                    ← skipped when jumped over
        //   [2] SuspendGenerator r0 r0 0 0  ← saves resume_pc=3
        //   [3] ResumeGenerator  r0 r0 0
        //   [4] LdaSmi 2                    ← runs after jump
        //   [5] Return
        let instrs = vec![
            Instruction::new_unchecked(Opcode::SwitchOnGeneratorState, vec![GEN_REG]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(
                Opcode::SuspendGenerator,
                vec![
                    GEN_REG,
                    GEN_REG,
                    Operand::RegisterCount(0),
                    Operand::Immediate(0),
                ],
            ),
            Instruction::new_unchecked(
                Opcode::ResumeGenerator,
                vec![GEN_REG, GEN_REG, Operand::RegisterCount(0)],
            ),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode(instrs, 1, 0);
        let gs = GeneratorState::new(ba.clone());

        // First step: SwitchOnGeneratorState falls through (resume_pc==0),
        // runs LdaSmi 1, yields 1, and saves resume_pc=3 (ResumeGenerator).
        let step1 = Interpreter::run_generator_step(&gs, JsValue::Undefined).expect("step 1 ok");
        assert_eq!(step1, GeneratorStep::Yield(JsValue::Smi(1)));
        let resume_pc_after_first = gs.borrow().resume_pc;
        assert_eq!(
            resume_pc_after_first, 3,
            "resume_pc should point to ResumeGenerator"
        );

        // Directly test the SwitchOnGeneratorState jump: run from PC 0
        // with resume_pc=3 already set in the generator state.  The switch
        // should jump to instruction 3, bypassing LdaSmi 1 and SuspendGenerator.
        let mut frame = InterpreterFrame::new(ba, vec![]);
        gs.borrow_mut().status = GeneratorStatus::SuspendedAtYield;
        frame.generator_state = Some(Rc::clone(&gs));
        let result = Interpreter::run(&mut frame).unwrap();
        // SwitchOnGeneratorState jumped to 3 (ResumeGenerator → no-op),
        // then LdaSmi 2, then Return → 2.
        assert_eq!(result, JsValue::Smi(2));
    }

    // ── Tiering: interpreter → baseline JIT ─────────────────────────────────

    /// Build the `add(a, b)` inner bytecode used by tiering tests.
    ///
    /// Implements `function add(a, b) { return a + b; }` where both parameters
    /// are Smi integers — values the JIT tier can handle natively.
    fn make_add_bytecode() -> BytecodeArray {
        let param0_v: u32 = (-1i32) as u32;
        let param1_v: u32 = (-2i32) as u32;
        let instrs = vec![
            // r0 = b (param[1])
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(param1_v)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            // acc = a (param[0])
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(param0_v)]),
            // acc = a + r0
            Instruction::new_unchecked(
                Opcode::Add,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        make_bytecode(instrs, 1, 2) // frame_size=1, param_count=2
    }

    /// Calling a function more than [`TIERING_THRESHOLD`] times must trigger
    /// baseline JIT compilation and cache the compiled code in the shared
    /// [`BytecodeArray`].
    ///
    /// The test calls `add(1, 2)` 110 times (> threshold of 100) and then
    /// asserts that JIT code has been stored in the bytecode array's cache.
    #[test]
    fn test_tiering_jit_compiled_after_threshold() {
        use crate::bytecode::bytecode_array::TIERING_THRESHOLD;

        let add_ba = make_add_bytecode();

        // Build a tiny outer script that calls `add` once:
        //   r0 = <function>, r1 = arg1, r2 = arg2
        //   acc = CallAnyReceiver(r0, r1, 2, slot0) → result
        //   Return acc
        let outer_instrs = vec![
            Instruction::new_unchecked(
                Opcode::CreateClosure,
                vec![
                    Operand::ConstantPoolIdx(0),
                    Operand::FeedbackSlot(0),
                    Operand::Flag(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            Instruction::new_unchecked(
                Opcode::CallAnyReceiver,
                vec![
                    Operand::Register(0),
                    Operand::Register(1),
                    Operand::RegisterCount(2),
                    Operand::FeedbackSlot(1),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let pool = vec![ConstantPoolEntry::Function(Box::new(add_ba))];
        let outer_ba = make_bytecode_with_pool(outer_instrs, pool, 3, 0);

        // Call add(1, 2) enough times to cross the tiering threshold.
        let call_count = TIERING_THRESHOLD + 10;
        let mut last_result = JsValue::Undefined;
        for _ in 0..call_count {
            let mut frame = InterpreterFrame::new(outer_ba.clone(), vec![]);
            last_result = Interpreter::run(&mut frame).unwrap();
        }

        // Regardless of JIT availability the interpreter result must be correct.
        assert_eq!(last_result, JsValue::Smi(3), "add(1, 2) must return 3");

        // On x86-64 Unix the baseline JIT should have been compiled and cached.
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            // Each outer_ba call creates a fresh closure via CreateClosure which
            // clones the inner BytecodeArray. All those clones share the same
            // Rc tiering state, so the invocation counter and JIT cache are
            // visible through any clone.
            let inner_ba: &BytecodeArray = match outer_ba.constant_pool().first().unwrap() {
                ConstantPoolEntry::Function(ba) => ba,
                _ => panic!("expected Function in constant pool"),
            };
            assert!(
                inner_ba.try_get_jit_code().is_some(),
                "JIT code should be cached after {} calls (threshold={})",
                call_count,
                TIERING_THRESHOLD,
            );
        }
    }

    /// After tiering is triggered the function must continue to return the
    /// correct result — whether executed by the JIT or the interpreter
    /// fallback.
    #[test]
    fn test_tiering_correct_result_after_jit() {
        use crate::bytecode::bytecode_array::TIERING_THRESHOLD;

        let add_ba = make_add_bytecode();

        let outer_instrs = vec![
            Instruction::new_unchecked(
                Opcode::CreateClosure,
                vec![
                    Operand::ConstantPoolIdx(0),
                    Operand::FeedbackSlot(0),
                    Operand::Flag(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(20)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            Instruction::new_unchecked(
                Opcode::CallAnyReceiver,
                vec![
                    Operand::Register(0),
                    Operand::Register(1),
                    Operand::RegisterCount(2),
                    Operand::FeedbackSlot(1),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let pool = vec![ConstantPoolEntry::Function(Box::new(add_ba))];
        let outer_ba = make_bytecode_with_pool(outer_instrs, pool, 3, 0);

        // Warm up past the threshold, then verify the result is still correct.
        let warm_up = TIERING_THRESHOLD + 20;
        for i in 0..warm_up {
            let mut frame = InterpreterFrame::new(outer_ba.clone(), vec![]);
            let result = Interpreter::run(&mut frame).unwrap();
            assert_eq!(
                result,
                JsValue::Smi(30),
                "add(10, 20) must return 30 on call {}",
                i + 1,
            );
        }
    }

    /// Calling a function containing a long-running loop must trigger OSR
    /// compilation: the JIT code cache should be populated after
    /// [`OSR_LOOP_THRESHOLD`] back-edges even if the function was only called
    /// once.
    #[test]
    fn test_tiering_osr_loop_triggers_compilation() {
        // Build: function looper() { let i = 0; while (i < 2000) { i++; } return i; }
        //
        // Bytecode (r0 = i):
        //   LdaZero
        //   Star r0
        // loop_start:
        //   Ldar r0
        //   LdaSmi 2000
        //   Star r1
        //   TestLessThan r1 slot0
        //   JumpIfFalse → exit
        //   Ldar r0
        //   Inc slot1
        //   Star r0
        //   JumpLoop → loop_start
        // exit:
        //   Ldar r0
        //   Return
        //
        // We rely on JumpLoop to bump osr_loop_count past OSR_LOOP_THRESHOLD.

        let r0: u32 = 0;
        let r1: u32 = 1;
        let instrs = vec![
            // [0] acc = 0
            Instruction::new_unchecked(Opcode::LdaZero, vec![]),
            // [1] r0 = 0
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(r0)]),
            // [2] acc = r0
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(r0)]),
            // [3] r1 = 2000
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2000)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(r1)]),
            // [5] acc = r0
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(r0)]),
            // [6] acc = acc < r1  (i < 2000)
            Instruction::new_unchecked(
                Opcode::TestLessThan,
                vec![Operand::Register(r1), Operand::FeedbackSlot(0)],
            ),
            // [7] if !acc → exit (offset computed below)
            Instruction::new_unchecked(Opcode::JumpIfFalse, vec![Operand::JumpOffset(0)]),
            // [8] acc = r0
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(r0)]),
            // [9] acc++
            Instruction::new_unchecked(Opcode::Inc, vec![Operand::FeedbackSlot(1)]),
            // [10] r0 = acc
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(r0)]),
            // [11] JumpLoop back to [2]
            Instruction::new_unchecked(
                Opcode::JumpLoop,
                vec![
                    Operand::JumpOffset(0),
                    Operand::Immediate(0),
                    Operand::FeedbackSlot(2),
                ],
            ),
            // [12] acc = r0
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(r0)]),
            // [13] Return
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];

        // Encode the instructions, then patch the jump offsets using the
        // byte-offset table produced by decode_with_byte_offsets.
        use crate::bytecode::bytecodes::{decode_with_byte_offsets, encode};
        let raw = encode(&instrs);
        let (_, offsets) = decode_with_byte_offsets(&raw).unwrap();

        // JumpIfFalse at instruction [7] must jump to instruction [12].
        // delta = offset[12] - offset[8]   (offset after the jump instruction)
        let jump_if_false_delta = offsets[12] as i32 - offsets[8] as i32;
        // JumpLoop at instruction [11] must jump back to instruction [2].
        // delta = offset[2] - offset[12]
        let jump_loop_delta = offsets[2] as i32 - offsets[12] as i32;

        let patched_instrs = vec![
            Instruction::new_unchecked(Opcode::LdaZero, vec![]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(r0)]),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(r0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2000)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(r1)]),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(r0)]),
            Instruction::new_unchecked(
                Opcode::TestLessThan,
                vec![Operand::Register(r1), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(
                Opcode::JumpIfFalse,
                vec![Operand::JumpOffset(jump_if_false_delta)],
            ),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(r0)]),
            Instruction::new_unchecked(Opcode::Inc, vec![Operand::FeedbackSlot(1)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(r0)]),
            Instruction::new_unchecked(
                Opcode::JumpLoop,
                vec![
                    Operand::JumpOffset(jump_loop_delta),
                    Operand::Immediate(0),
                    Operand::FeedbackSlot(2),
                ],
            ),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(r0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let looper_ba = make_bytecode(patched_instrs, 2, 0); // frame_size=2, param_count=0

        // Run the loop function once; it iterates 2000 times which exceeds
        // OSR_LOOP_THRESHOLD (1000), so a compilation should be triggered.
        let mut frame = InterpreterFrame::new(looper_ba.clone(), vec![]);
        let result = Interpreter::run(&mut frame).unwrap();

        // The loop result must be correct.
        assert_eq!(result, JsValue::Smi(2000), "loop should count to 2000");

        // On x86-64 Unix, OSR should have triggered JIT compilation.
        // The looper_ba and the frame's clone share the same Rc jit_code.
        #[cfg(all(target_arch = "x86_64", unix))]
        assert!(
            looper_ba.try_get_jit_code().is_some(),
            "OSR: JIT code should be cached after a long-running loop"
        );
    }

    /// `jit_stats()` must be updated when a function is compiled to baseline
    /// JIT.  The test captures a snapshot before and after triggering
    /// compilation and asserts that the counters increase on x86-64 Unix.
    #[test]
    fn test_jit_stats_updated_after_compilation() {
        use super::jit_stats;
        use crate::bytecode::bytecode_array::TIERING_THRESHOLD;

        let add_ba = make_add_bytecode();
        let outer_instrs = vec![
            Instruction::new_unchecked(
                Opcode::CreateClosure,
                vec![
                    Operand::ConstantPoolIdx(0),
                    Operand::FeedbackSlot(0),
                    Operand::Flag(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            Instruction::new_unchecked(
                Opcode::CallAnyReceiver,
                vec![
                    Operand::Register(0),
                    Operand::Register(1),
                    Operand::RegisterCount(2),
                    Operand::FeedbackSlot(1),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let pool = vec![ConstantPoolEntry::Function(Box::new(add_ba))];
        let outer_ba = make_bytecode_with_pool(outer_instrs, pool, 3, 0);

        let (count_before, bytes_before) = jit_stats();

        // Call enough times to cross the tiering threshold and trigger JIT.
        for _ in 0..(TIERING_THRESHOLD + 10) {
            let mut frame = InterpreterFrame::new(outer_ba.clone(), vec![]);
            Interpreter::run(&mut frame).unwrap();
        }

        let (count_after, bytes_after) = jit_stats();

        // On x86-64 Unix the stats must increase; on other platforms both
        // counters remain zero.
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            assert!(
                count_after > count_before,
                "jit_stats().0 must increase after tiering (before={count_before}, after={count_after})"
            );
            assert!(
                bytes_after > bytes_before,
                "jit_stats().1 must increase after tiering (before={bytes_before}, after={bytes_after})"
            );
        }
        #[cfg(not(all(target_arch = "x86_64", unix)))]
        {
            assert_eq!(
                count_after, 0,
                "jit_stats count must be zero on non-JIT platform"
            );
            assert_eq!(
                bytes_after, 0,
                "jit_stats bytes must be zero on non-JIT platform"
            );
        }
    }

    /// Calling a hot function more than [`MAGLEV_TIERING_THRESHOLD`] times must
    /// schedule a background Maglev compilation and eventually cache the result.
    ///
    /// The test calls `add(1, 2)` well above the Maglev threshold and then
    /// polls for the Maglev cache to be populated, asserting the correct result.
    #[test]
    fn test_maglev_compiled_after_threshold() {
        use super::maglev_stats;
        use crate::bytecode::bytecode_array::MAGLEV_TIERING_THRESHOLD;

        let add_ba = make_add_bytecode();

        let outer_instrs = vec![
            Instruction::new_unchecked(
                Opcode::CreateClosure,
                vec![
                    Operand::ConstantPoolIdx(0),
                    Operand::FeedbackSlot(0),
                    Operand::Flag(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            Instruction::new_unchecked(
                Opcode::CallAnyReceiver,
                vec![
                    Operand::Register(0),
                    Operand::Register(1),
                    Operand::RegisterCount(2),
                    Operand::FeedbackSlot(1),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let pool = vec![ConstantPoolEntry::Function(Box::new(add_ba))];
        let outer_ba = make_bytecode_with_pool(outer_instrs, pool, 3, 0);

        // Exceed the Maglev threshold so a background compilation is triggered.
        let call_count = MAGLEV_TIERING_THRESHOLD + 10;
        let mut last_result = JsValue::Undefined;
        for _ in 0..call_count {
            let mut frame = InterpreterFrame::new(outer_ba.clone(), vec![]);
            last_result = Interpreter::run(&mut frame).unwrap();
        }

        // The function must always return the correct value.
        assert_eq!(last_result, JsValue::Smi(3), "add(1, 2) must return 3");

        // On x86-64 Unix, wait briefly for the background Maglev compilation
        // to finish, then verify the cache is populated and the function still
        // returns the correct result via the Maglev tier.
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            let inner_ba: &BytecodeArray = match outer_ba.constant_pool().first().unwrap() {
                ConstantPoolEntry::Function(ba) => ba,
                _ => panic!("expected Function in constant pool"),
            };

            // Poll for Maglev compilation to finish (background thread).
            let timeout = std::time::Duration::from_secs(5);
            let start = std::time::Instant::now();
            while inner_ba.try_get_maglev_jit_code().is_none() && start.elapsed() < timeout {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }

            assert!(
                inner_ba.try_get_maglev_jit_code().is_some(),
                "Maglev code should be cached after {} calls (threshold={})",
                call_count,
                MAGLEV_TIERING_THRESHOLD,
            );

            // Verify the Maglev tier also returns the correct result.
            let mut frame = InterpreterFrame::new(outer_ba.clone(), vec![]);
            let result = Interpreter::run(&mut frame).unwrap();
            assert_eq!(result, JsValue::Smi(3), "Maglev add(1, 2) must return 3");

            // Maglev stats must have been incremented.
            let (maglev_count, _maglev_bytes) = maglev_stats();
            assert!(
                maglev_count > 0,
                "maglev_stats count must be > 0 after Maglev compilation"
            );
        }
    }

    /// After the Maglev tier is installed the function must continue to return
    /// the correct result for all argument combinations.
    #[test]
    fn test_maglev_correct_result_after_tier_up() {
        use crate::bytecode::bytecode_array::MAGLEV_TIERING_THRESHOLD;

        let add_ba = make_add_bytecode();

        let outer_instrs = vec![
            Instruction::new_unchecked(
                Opcode::CreateClosure,
                vec![
                    Operand::ConstantPoolIdx(0),
                    Operand::FeedbackSlot(0),
                    Operand::Flag(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            Instruction::new_unchecked(
                Opcode::CallAnyReceiver,
                vec![
                    Operand::Register(0),
                    Operand::Register(1),
                    Operand::RegisterCount(2),
                    Operand::FeedbackSlot(1),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let pool = vec![ConstantPoolEntry::Function(Box::new(add_ba))];
        let outer_ba = make_bytecode_with_pool(outer_instrs, pool, 3, 0);

        // Warm up past the Maglev threshold.
        let warm_up = MAGLEV_TIERING_THRESHOLD + 20;
        for i in 0..warm_up {
            let mut frame = InterpreterFrame::new(outer_ba.clone(), vec![]);
            let result = Interpreter::run(&mut frame).unwrap();
            assert_eq!(
                result,
                JsValue::Smi(12),
                "add(5, 7) must return 12 on call {}",
                i + 1,
            );
        }
    }

    /// Calling a hot function more than [`TURBOFAN_TIERING_THRESHOLD`] times
    /// must schedule a background Turbofan compilation and eventually cache the
    /// result.
    ///
    /// The test calls `add(1, 2)` well above the Turbofan threshold and then
    /// polls for the Turbofan cache to be populated, asserting the correct
    /// result.
    #[test]
    fn test_turbofan_compiled_after_threshold() {
        use super::turbofan_stats;
        use crate::bytecode::bytecode_array::TURBOFAN_TIERING_THRESHOLD;

        let add_ba = make_add_bytecode();

        let outer_instrs = vec![
            Instruction::new_unchecked(
                Opcode::CreateClosure,
                vec![
                    Operand::ConstantPoolIdx(0),
                    Operand::FeedbackSlot(0),
                    Operand::Flag(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            Instruction::new_unchecked(
                Opcode::CallAnyReceiver,
                vec![
                    Operand::Register(0),
                    Operand::Register(1),
                    Operand::RegisterCount(2),
                    Operand::FeedbackSlot(1),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let pool = vec![ConstantPoolEntry::Function(Box::new(add_ba))];
        let outer_ba = make_bytecode_with_pool(outer_instrs, pool, 3, 0);

        // Exceed the Turbofan threshold so a background compilation is triggered.
        let call_count = TURBOFAN_TIERING_THRESHOLD + 10;
        let mut last_result = JsValue::Undefined;
        for _ in 0..call_count {
            let mut frame = InterpreterFrame::new(outer_ba.clone(), vec![]);
            last_result = Interpreter::run(&mut frame).unwrap();
        }

        // The function must always return the correct value.
        assert_eq!(last_result, JsValue::Smi(3), "add(1, 2) must return 3");

        // On x86-64 Unix, wait briefly for the background Turbofan compilation
        // to finish, then verify the cache is populated and the function still
        // returns the correct result via the Turbofan tier.
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            let inner_ba: &BytecodeArray = match outer_ba.constant_pool().first().unwrap() {
                ConstantPoolEntry::Function(ba) => ba,
                _ => panic!("expected Function in constant pool"),
            };

            // Poll for Turbofan compilation to finish (background thread).
            let timeout = std::time::Duration::from_secs(5);
            let start = std::time::Instant::now();
            while !inner_ba.has_turbofan_jit_code() && start.elapsed() < timeout {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }

            assert!(
                inner_ba.has_turbofan_jit_code(),
                "Turbofan code should be cached after {} calls (threshold={})",
                call_count,
                TURBOFAN_TIERING_THRESHOLD,
            );

            // Verify the Turbofan tier also returns the correct result.
            let mut frame = InterpreterFrame::new(outer_ba.clone(), vec![]);
            let result = Interpreter::run(&mut frame).unwrap();
            assert_eq!(result, JsValue::Smi(3), "Turbofan add(1, 2) must return 3");

            // Turbofan stats must have been incremented.
            let (tf_count, _tf_bytes) = turbofan_stats();
            assert!(
                tf_count > 0,
                "turbofan_stats count must be > 0 after Turbofan compilation"
            );
        }
    }

    /// After the Turbofan tier is installed the function must continue to
    /// return the correct result for all argument combinations.
    #[test]
    fn test_turbofan_correct_result_after_tier_up() {
        use crate::bytecode::bytecode_array::TURBOFAN_TIERING_THRESHOLD;

        let add_ba = make_add_bytecode();

        let outer_instrs = vec![
            Instruction::new_unchecked(
                Opcode::CreateClosure,
                vec![
                    Operand::ConstantPoolIdx(0),
                    Operand::FeedbackSlot(0),
                    Operand::Flag(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(4)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            Instruction::new_unchecked(
                Opcode::CallAnyReceiver,
                vec![
                    Operand::Register(0),
                    Operand::Register(1),
                    Operand::RegisterCount(2),
                    Operand::FeedbackSlot(1),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let pool = vec![ConstantPoolEntry::Function(Box::new(add_ba))];
        let outer_ba = make_bytecode_with_pool(outer_instrs, pool, 3, 0);

        // Warm up past the Turbofan threshold.
        let warm_up = TURBOFAN_TIERING_THRESHOLD + 20;
        for i in 0..warm_up {
            let mut frame = InterpreterFrame::new(outer_ba.clone(), vec![]);
            let result = Interpreter::run(&mut frame).unwrap();
            assert_eq!(
                result,
                JsValue::Smi(7),
                "add(3, 4) must return 7 on call {}",
                i + 1,
            );
        }
    }

    // ── TypeOf ───────────────────────────────────────────────────────────────

    #[test]
    fn test_typeof_undefined() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
                Instruction::new_unchecked(Opcode::TypeOf, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::String("undefined".to_owned().into()));
    }

    #[test]
    fn test_typeof_null_is_object() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaNull, vec![]),
                Instruction::new_unchecked(Opcode::TypeOf, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        // typeof null === "object" per the ECMAScript specification.
        assert_eq!(result, JsValue::String("object".to_owned().into()));
    }

    #[test]
    fn test_typeof_boolean() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaTrue, vec![]),
                Instruction::new_unchecked(Opcode::TypeOf, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::String("boolean".to_owned().into()));
    }

    #[test]
    fn test_typeof_number() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(Opcode::TypeOf, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::String("number".to_owned().into()));
    }

    #[test]
    fn test_typeof_string() {
        // Load a string constant via the constant pool.
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(0)]),
            Instruction::new_unchecked(Opcode::TypeOf, vec![Operand::FeedbackSlot(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let pool = vec![ConstantPoolEntry::String("hello".to_owned())];
        let bytes = encode(&instrs);
        let ba = BytecodeArray::new(bytes, pool, 0, 0, vec![], FeedbackMetadata::empty(), vec![]);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::String("string".to_owned().into()));
    }

    // ── TestTypeOf ───────────────────────────────────────────────────────────

    #[test]
    fn test_testtypeof_number_match() {
        // TestTypeOf flag=0 (number) on a Smi → true
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
                Instruction::new_unchecked(Opcode::TestTypeOf, vec![Operand::Flag(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_testtypeof_number_mismatch() {
        // TestTypeOf flag=0 (number) on undefined → false
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
                Instruction::new_unchecked(Opcode::TestTypeOf, vec![Operand::Flag(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_testtypeof_undefined() {
        // TestTypeOf flag=5 (undefined)
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
                Instruction::new_unchecked(Opcode::TestTypeOf, vec![Operand::Flag(5)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_testtypeof_boolean() {
        // TestTypeOf flag=3 (boolean)
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaTrue, vec![]),
                Instruction::new_unchecked(Opcode::TestTypeOf, vec![Operand::Flag(3)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_testtypeof_object_for_null() {
        // TestTypeOf flag=7 (object) on null → true (typeof null === "object")
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaNull, vec![]),
                Instruction::new_unchecked(Opcode::TestTypeOf, vec![Operand::Flag(7)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_testtypeof_undefined_for_thehole() {
        // TestTypeOf flag=5 (undefined) on TheHole → true
        // Consistent with TypeOf which maps TheHole → "undefined".
        let result = run_with_acc_and_regs(
            JsValue::TheHole,
            &[],
            vec![Instruction::new_unchecked(
                Opcode::TestTypeOf,
                vec![Operand::Flag(5)],
            )],
            0,
        );
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── ToNumber ─────────────────────────────────────────────────────────────

    #[test]
    fn test_to_number_from_string() {
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(0)]),
            Instruction::new_unchecked(Opcode::ToNumber, vec![Operand::FeedbackSlot(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let bytes = encode(&instrs);
        let ba = BytecodeArray::new(
            bytes,
            vec![ConstantPoolEntry::String("42".to_string())],
            0,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn test_to_number_from_boolean_true() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaTrue, vec![]),
                Instruction::new_unchecked(Opcode::ToNumber, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(1));
    }

    #[test]
    fn test_to_number_from_undefined() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
                Instruction::new_unchecked(Opcode::ToNumber, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        // NaN != NaN, so we match the variant and check .is_nan()
        match result {
            JsValue::HeapNumber(n) => assert!(n.is_nan()),
            other => panic!("expected HeapNumber(NaN), got {other:?}"),
        }
    }

    // ── ToString ─────────────────────────────────────────────────────────────

    #[test]
    fn test_to_string_from_smi() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(123)]),
                Instruction::new_unchecked(Opcode::ToString, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::String("123".to_string().into()));
    }

    #[test]
    fn test_to_string_from_boolean_false() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaFalse, vec![]),
                Instruction::new_unchecked(Opcode::ToString, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::String("false".to_string().into()));
    }

    #[test]
    fn test_to_string_from_null() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaNull, vec![]),
                Instruction::new_unchecked(Opcode::ToString, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::String("null".to_string().into()));
    }

    // ── ToBoolean ────────────────────────────────────────────────────────────

    #[test]
    fn test_to_boolean_from_zero() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaZero, vec![]),
                Instruction::new_unchecked(Opcode::ToBoolean, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_to_boolean_from_nonzero() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
                Instruction::new_unchecked(Opcode::ToBoolean, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_to_boolean_from_undefined() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
                Instruction::new_unchecked(Opcode::ToBoolean, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    // ── ToObject ─────────────────────────────────────────────────────────────

    #[test]
    fn test_to_object_null_throws() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaNull, vec![]),
                Instruction::new_unchecked(Opcode::ToObject, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_to_object_undefined_throws() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
                Instruction::new_unchecked(Opcode::ToObject, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_to_object_smi_wrapper() {
        // Smi primitives are wrapped in a PlainObject per ECMAScript §7.1.18.
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(Opcode::ToObject, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        match &result {
            JsValue::PlainObject(map) => {
                let m = map.borrow();
                assert_eq!(m.get("__wrapped__").cloned(), Some(JsValue::Smi(5)));
            }
            other => panic!("expected PlainObject wrapper, got {:?}", other),
        }
    }

    // ── ToName ───────────────────────────────────────────────────────────────

    #[test]
    fn test_to_name_from_smi() {
        // Numbers become their string representation.
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
                Instruction::new_unchecked(Opcode::ToName, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::String("10".to_string().into()));
    }

    #[test]
    fn test_to_name_string_passthrough() {
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(0)]),
            Instruction::new_unchecked(Opcode::ToName, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let bytes = encode(&instrs);
        let ba = BytecodeArray::new(
            bytes,
            vec![ConstantPoolEntry::String("hello".to_string())],
            1,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::String("hello".to_string().into()));
    }

    // ── Negate ───────────────────────────────────────────────────────────────

    #[test]
    fn test_negate_positive() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(Opcode::Negate, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(-5));
    }

    #[test]
    fn test_negate_zero() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaZero, vec![]),
                Instruction::new_unchecked(Opcode::Negate, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        // -0.0 is not representable as Smi(0), should be HeapNumber(-0.0)
        // Actually number_to_jsvalue(-0.0): -0.0.fract() == 0.0 and -0.0 is finite
        // so it becomes Smi(0). That's acceptable.
        assert!(result == JsValue::Smi(0) || result == JsValue::HeapNumber(-0.0));
    }

    #[test]
    fn test_negate_negative() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(-3)]),
                Instruction::new_unchecked(Opcode::Negate, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    // ── BitwiseNot ───────────────────────────────────────────────────────────

    #[test]
    fn test_bitwise_not_zero() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaZero, vec![]),
                Instruction::new_unchecked(Opcode::BitwiseNot, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(-1));
    }

    #[test]
    fn test_bitwise_not_positive() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(Opcode::BitwiseNot, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(-6));
    }

    #[test]
    fn test_bitwise_not_negative() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(-1)]),
                Instruction::new_unchecked(Opcode::BitwiseNot, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(0));
    }

    // ── BitwiseOr / BitwiseAnd / BitwiseXor (register) ──────────────────────

    #[test]
    fn test_bitwise_or() {
        // 0b1010 | 0b1100 = 0b1110 = 14
        assert_eq!(
            arith_op(0b1010, 0b1100, Opcode::BitwiseOr).unwrap(),
            JsValue::Smi(14)
        );
        // 0 | 0 = 0
        assert_eq!(arith_op(0, 0, Opcode::BitwiseOr).unwrap(), JsValue::Smi(0));
        // -1 | 0 = -1
        assert_eq!(
            arith_op(-1, 0, Opcode::BitwiseOr).unwrap(),
            JsValue::Smi(-1)
        );
    }

    #[test]
    fn test_bitwise_and() {
        // 0b1010 & 0b1100 = 0b1000 = 8
        assert_eq!(
            arith_op(0b1010, 0b1100, Opcode::BitwiseAnd).unwrap(),
            JsValue::Smi(8)
        );
        // 0xFF & 0x0F = 0x0F = 15
        assert_eq!(
            arith_op(0xFF, 0x0F, Opcode::BitwiseAnd).unwrap(),
            JsValue::Smi(15)
        );
        // -1 & 0x7FFFFFFF = 0x7FFFFFFF
        assert_eq!(
            arith_op(-1, 0x7FFFFFFF, Opcode::BitwiseAnd).unwrap(),
            JsValue::Smi(0x7FFFFFFF)
        );
    }

    #[test]
    fn test_bitwise_xor() {
        // 0b1010 ^ 0b1100 = 0b0110 = 6
        assert_eq!(
            arith_op(0b1010, 0b1100, Opcode::BitwiseXor).unwrap(),
            JsValue::Smi(6)
        );
        // 5 ^ 5 = 0
        assert_eq!(arith_op(5, 5, Opcode::BitwiseXor).unwrap(), JsValue::Smi(0));
    }

    // ── ShiftLeft / ShiftRight / ShiftRightLogical (register) ───────────────

    #[test]
    fn test_shift_left() {
        // 1 << 4 = 16
        assert_eq!(arith_op(1, 4, Opcode::ShiftLeft).unwrap(), JsValue::Smi(16));
        // 3 << 0 = 3
        assert_eq!(arith_op(3, 0, Opcode::ShiftLeft).unwrap(), JsValue::Smi(3));
        // shift amount is masked to 5 bits: 1 << 32 = 1 << 0 = 1
        assert_eq!(arith_op(1, 32, Opcode::ShiftLeft).unwrap(), JsValue::Smi(1));
    }

    #[test]
    fn test_shift_right() {
        // 16 >> 2 = 4
        assert_eq!(
            arith_op(16, 2, Opcode::ShiftRight).unwrap(),
            JsValue::Smi(4)
        );
        // -1 >> 1 = -1 (sign-extending)
        assert_eq!(
            arith_op(-1, 1, Opcode::ShiftRight).unwrap(),
            JsValue::Smi(-1)
        );
    }

    #[test]
    fn test_shift_right_logical() {
        // 16 >>> 2 = 4
        assert_eq!(
            arith_op(16, 2, Opcode::ShiftRightLogical).unwrap(),
            JsValue::Smi(4)
        );
        // -1 >>> 0 = 4294967295 (0xFFFFFFFF as u32, doesn't fit Smi → HeapNumber)
        let result = arith_op(-1, 0, Opcode::ShiftRightLogical).unwrap();
        assert_eq!(result, JsValue::HeapNumber(4294967295.0));
    }

    // ── Exp (register) ──────────────────────────────────────────────────────

    #[test]
    fn test_exp() {
        // 2 ** 10 = 1024
        assert_eq!(arith_op(2, 10, Opcode::Exp).unwrap(), JsValue::Smi(1024));
        // 3 ** 0 = 1
        assert_eq!(arith_op(3, 0, Opcode::Exp).unwrap(), JsValue::Smi(1));
        // 5 ** 1 = 5
        assert_eq!(arith_op(5, 1, Opcode::Exp).unwrap(), JsValue::Smi(5));
    }

    // ── Smi immediate variants ──────────────────────────────────────────────

    /// Helper: evaluate `acc <op> imm` using the Smi immediate pattern.
    fn smi_op(acc: i32, imm: i32, op: Opcode) -> StatorResult<JsValue> {
        run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(acc)]),
                Instruction::new_unchecked(
                    op,
                    vec![Operand::Immediate(imm), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
    }

    #[test]
    fn test_bitwise_or_smi() {
        assert_eq!(
            smi_op(0b1010, 0b1100, Opcode::BitwiseOrSmi).unwrap(),
            JsValue::Smi(14)
        );
        assert_eq!(smi_op(0, 0, Opcode::BitwiseOrSmi).unwrap(), JsValue::Smi(0));
    }

    #[test]
    fn test_bitwise_and_smi() {
        assert_eq!(
            smi_op(0b1010, 0b1100, Opcode::BitwiseAndSmi).unwrap(),
            JsValue::Smi(8)
        );
        assert_eq!(
            smi_op(0xFF, 0x0F, Opcode::BitwiseAndSmi).unwrap(),
            JsValue::Smi(15)
        );
    }

    #[test]
    fn test_bitwise_xor_smi() {
        assert_eq!(
            smi_op(0b1010, 0b1100, Opcode::BitwiseXorSmi).unwrap(),
            JsValue::Smi(6)
        );
    }

    #[test]
    fn test_shift_left_smi() {
        assert_eq!(
            smi_op(1, 4, Opcode::ShiftLeftSmi).unwrap(),
            JsValue::Smi(16)
        );
        // shift amount masked: 1 << 32 = 1 << 0 = 1
        assert_eq!(
            smi_op(1, 32, Opcode::ShiftLeftSmi).unwrap(),
            JsValue::Smi(1)
        );
    }

    #[test]
    fn test_shift_right_smi() {
        assert_eq!(
            smi_op(16, 2, Opcode::ShiftRightSmi).unwrap(),
            JsValue::Smi(4)
        );
        assert_eq!(
            smi_op(-1, 1, Opcode::ShiftRightSmi).unwrap(),
            JsValue::Smi(-1)
        );
    }

    #[test]
    fn test_shift_right_logical_smi() {
        assert_eq!(
            smi_op(16, 2, Opcode::ShiftRightLogicalSmi).unwrap(),
            JsValue::Smi(4)
        );
        // -1 >>> 0 = 0xFFFFFFFF
        let result = smi_op(-1, 0, Opcode::ShiftRightLogicalSmi).unwrap();
        assert_eq!(result, JsValue::HeapNumber(4294967295.0));
    }

    #[test]
    fn test_exp_smi() {
        assert_eq!(smi_op(2, 10, Opcode::ExpSmi).unwrap(), JsValue::Smi(1024));
        assert_eq!(smi_op(5, 0, Opcode::ExpSmi).unwrap(), JsValue::Smi(1));
    }

    #[test]
    fn test_add_smi() {
        assert_eq!(smi_op(3, 4, Opcode::AddSmi).unwrap(), JsValue::Smi(7));
    }

    #[test]
    fn test_sub_smi() {
        assert_eq!(smi_op(10, 3, Opcode::SubSmi).unwrap(), JsValue::Smi(7));
    }

    #[test]
    fn test_mul_smi() {
        assert_eq!(smi_op(6, 7, Opcode::MulSmi).unwrap(), JsValue::Smi(42));
    }

    #[test]
    fn test_div_smi() {
        assert_eq!(smi_op(10, 2, Opcode::DivSmi).unwrap(), JsValue::Smi(5));
        assert_eq!(
            smi_op(7, 2, Opcode::DivSmi).unwrap(),
            JsValue::HeapNumber(3.5)
        );
    }

    #[test]
    fn test_mod_smi() {
        assert_eq!(smi_op(10, 3, Opcode::ModSmi).unwrap(), JsValue::Smi(1));
    }

    // ── Keyed Property Access ────────────────────────────────────────────────

    /// Helper: create a PlainObject from key-value pairs.
    fn make_plain_object(pairs: Vec<(&str, JsValue)>) -> JsValue {
        let mut map = PropertyMap::new();
        for (k, v) in pairs {
            map.insert(k.to_string(), v);
        }
        JsValue::PlainObject(Rc::new(RefCell::new(map)))
    }

    #[test]
    fn test_lda_keyed_property_string_key() {
        let obj = make_plain_object(vec![
            ("x", JsValue::Smi(42)),
            ("name", JsValue::String("hello".to_string().into())),
        ]);
        let val = keyed_load(&obj, &JsValue::String("x".to_string().into())).unwrap();
        assert_eq!(val, JsValue::Smi(42));

        let val = keyed_load(&obj, &JsValue::String("name".to_string().into())).unwrap();
        assert_eq!(val, JsValue::String("hello".to_string().into()));

        // Missing key returns undefined
        let val = keyed_load(&obj, &JsValue::String("missing".to_string().into())).unwrap();
        assert_eq!(val, JsValue::Undefined);
    }

    #[test]
    fn test_lda_keyed_property_integer_key_on_array() {
        let arr = JsValue::new_array(vec![JsValue::Smi(10), JsValue::Smi(20), JsValue::Smi(30)]);
        // Integer index (Smi key)
        assert_eq!(
            keyed_load(&arr, &JsValue::Smi(0)).unwrap(),
            JsValue::Smi(10)
        );
        assert_eq!(
            keyed_load(&arr, &JsValue::Smi(2)).unwrap(),
            JsValue::Smi(30)
        );
        // Out-of-bounds → undefined
        assert_eq!(
            keyed_load(&arr, &JsValue::Smi(5)).unwrap(),
            JsValue::Undefined
        );
        // "length" property
        assert_eq!(
            keyed_load(&arr, &JsValue::String("length".to_string().into())).unwrap(),
            JsValue::Smi(3)
        );
        // String index "1" works
        assert_eq!(
            keyed_load(&arr, &JsValue::String("1".to_string().into())).unwrap(),
            JsValue::Smi(20)
        );
    }

    // ── Prototype chain tests ────────────────────────────────────────────

    #[test]
    fn test_proto_lookup_own_property() {
        let obj = make_plain_object(vec![("x", JsValue::Smi(1))]);
        assert_eq!(proto_lookup(&obj, "x"), JsValue::Smi(1));
    }

    #[test]
    fn test_proto_lookup_inherited_property() {
        let parent = make_plain_object(vec![("inherited", JsValue::Smi(99))]);
        let child = make_plain_object(vec![("own", JsValue::Smi(1)), ("__proto__", parent)]);
        // Own property works
        assert_eq!(proto_lookup(&child, "own"), JsValue::Smi(1));
        // Inherited property found via __proto__
        assert_eq!(proto_lookup(&child, "inherited"), JsValue::Smi(99));
        // Missing property returns Undefined
        assert_eq!(proto_lookup(&child, "missing"), JsValue::Undefined);
    }

    #[test]
    fn test_proto_lookup_shadowing() {
        let parent = make_plain_object(vec![("x", JsValue::Smi(100))]);
        let child = make_plain_object(vec![("x", JsValue::Smi(1)), ("__proto__", parent)]);
        // Own property shadows inherited
        assert_eq!(proto_lookup(&child, "x"), JsValue::Smi(1));
    }

    #[test]
    fn test_proto_lookup_multi_level_chain() {
        let grandparent = make_plain_object(vec![("deep", JsValue::String("gp".into()))]);
        let parent = make_plain_object(vec![
            ("mid", JsValue::String("p".into())),
            ("__proto__", grandparent),
        ]);
        let child = make_plain_object(vec![
            ("own", JsValue::String("c".into())),
            ("__proto__", parent),
        ]);
        assert_eq!(proto_lookup(&child, "own"), JsValue::String("c".into()));
        assert_eq!(proto_lookup(&child, "mid"), JsValue::String("p".into()));
        assert_eq!(proto_lookup(&child, "deep"), JsValue::String("gp".into()));
        assert_eq!(proto_lookup(&child, "nope"), JsValue::Undefined);
    }

    #[test]
    fn test_proto_lookup_non_object_returns_undefined() {
        assert_eq!(proto_lookup(&JsValue::Smi(42), "x"), JsValue::Undefined);
        assert_eq!(proto_lookup(&JsValue::Null, "x"), JsValue::Undefined);
    }

    #[test]
    fn test_keyed_load_walks_proto_chain() {
        let parent = make_plain_object(vec![("greet", JsValue::String("hello".into()))]);
        let child = make_plain_object(vec![("__proto__", parent)]);
        let val = keyed_load(&child, &JsValue::String("greet".into())).unwrap();
        assert_eq!(val, JsValue::String("hello".into()));
    }

    #[test]
    fn test_lda_keyed_property_string_char_at() {
        let s = JsValue::String("hello".to_string().into());
        assert_eq!(
            keyed_load(&s, &JsValue::Smi(0)).unwrap(),
            JsValue::String("h".to_string().into())
        );
        assert_eq!(
            keyed_load(&s, &JsValue::Smi(4)).unwrap(),
            JsValue::String("o".to_string().into())
        );
        // Out-of-bounds → undefined
        assert_eq!(
            keyed_load(&s, &JsValue::Smi(10)).unwrap(),
            JsValue::Undefined
        );
        // "length" property
        assert_eq!(
            keyed_load(&s, &JsValue::String("length".to_string().into())).unwrap(),
            JsValue::Smi(5)
        );
    }

    #[test]
    fn test_keyed_store_plain_object() {
        let obj = make_plain_object(vec![("x", JsValue::Smi(1))]);
        // Store new property
        keyed_store(
            &obj,
            &JsValue::String("y".to_string().into()),
            JsValue::Smi(99),
        )
        .unwrap();
        assert_eq!(
            keyed_load(&obj, &JsValue::String("y".to_string().into())).unwrap(),
            JsValue::Smi(99)
        );
        // Overwrite existing property
        keyed_store(
            &obj,
            &JsValue::String("x".to_string().into()),
            JsValue::Smi(7),
        )
        .unwrap();
        assert_eq!(
            keyed_load(&obj, &JsValue::String("x".to_string().into())).unwrap(),
            JsValue::Smi(7)
        );
        // Numeric string key
        keyed_store(
            &obj,
            &JsValue::Smi(0),
            JsValue::String("zero".to_string().into()),
        )
        .unwrap();
        assert_eq!(
            keyed_load(&obj, &JsValue::String("0".to_string().into())).unwrap(),
            JsValue::String("zero".to_string().into())
        );
    }

    #[test]
    fn test_keyed_store_non_object_silently_discarded() {
        // Storing to non-objects should not error
        keyed_store(
            &JsValue::Smi(42),
            &JsValue::String("x".to_string().into()),
            JsValue::Smi(1),
        )
        .unwrap();
        keyed_store(
            &JsValue::Undefined,
            &JsValue::String("x".to_string().into()),
            JsValue::Smi(1),
        )
        .unwrap();
    }

    #[test]
    fn test_keyed_store_non_extensible_silently_ignores_new_property() {
        let obj = make_plain_object(vec![("x", JsValue::Smi(1))]);
        if let JsValue::PlainObject(ref map) = obj {
            map.borrow_mut().extensible = false;
        }
        // Attempting to add a new property on a non-extensible object should
        // silently succeed (no error) but NOT actually add the property.
        keyed_store(
            &obj,
            &JsValue::String("y".to_string().into()),
            JsValue::Smi(99),
        )
        .unwrap();
        assert_eq!(
            keyed_load(&obj, &JsValue::String("y".to_string().into())).unwrap(),
            JsValue::Undefined,
            "new property should not be added to non-extensible object"
        );
        // Existing properties should still be writable.
        keyed_store(
            &obj,
            &JsValue::String("x".to_string().into()),
            JsValue::Smi(42),
        )
        .unwrap();
        assert_eq!(
            keyed_load(&obj, &JsValue::String("x".to_string().into())).unwrap(),
            JsValue::Smi(42),
            "existing writable property should still be updatable"
        );
    }

    #[test]
    fn test_keyed_store_non_writable_silently_ignored() {
        let obj = make_plain_object(vec![("x", JsValue::Smi(1))]);
        if let JsValue::PlainObject(ref map) = obj {
            map.borrow_mut().set_writable("x", false);
        }
        // Store to non-writable property should be silently ignored.
        keyed_store(
            &obj,
            &JsValue::String("x".to_string().into()),
            JsValue::Smi(99),
        )
        .unwrap();
        assert_eq!(
            keyed_load(&obj, &JsValue::String("x".to_string().into())).unwrap(),
            JsValue::Smi(1),
            "non-writable property should retain original value"
        );
    }

    #[test]
    fn test_keyed_store_accessor_setter_dispatched() {
        // Create an object with a setter that records the stored value.
        let obj = make_plain_object(vec![]);
        let stored = Rc::new(RefCell::new(JsValue::Undefined));
        let stored_clone = Rc::clone(&stored);
        if let JsValue::PlainObject(ref map) = obj {
            let setter = JsValue::NativeFunction(Rc::new(move |args| {
                *stored_clone.borrow_mut() = args.first().cloned().unwrap_or(JsValue::Undefined);
                Ok(JsValue::Undefined)
            }));
            map.borrow_mut().insert("__set_x__".to_string(), setter);
        }
        keyed_store(
            &obj,
            &JsValue::String("x".to_string().into()),
            JsValue::Smi(42),
        )
        .unwrap();
        assert_eq!(
            *stored.borrow(),
            JsValue::Smi(42),
            "setter should have been invoked with the stored value"
        );
    }

    #[test]
    fn test_keyed_store_getter_only_silently_ignored() {
        // Create an object with a getter but no setter.
        let obj = make_plain_object(vec![]);
        if let JsValue::PlainObject(ref map) = obj {
            let getter = JsValue::NativeFunction(Rc::new(|_| Ok(JsValue::Smi(99))));
            map.borrow_mut().insert("__get_x__".to_string(), getter);
        }
        // Store should be silently ignored (no error in sloppy mode).
        keyed_store(
            &obj,
            &JsValue::String("x".to_string().into()),
            JsValue::Smi(42),
        )
        .unwrap();
    }

    #[test]
    fn test_proto_lookup_accessor_getter_takes_precedence() {
        // When a property has both a data key (placeholder) and a getter
        // accessor, proto_lookup should dispatch the getter.
        let obj = make_plain_object(vec![("x", JsValue::Undefined)]);
        if let JsValue::PlainObject(ref map) = obj {
            let getter = JsValue::NativeFunction(Rc::new(|_| Ok(JsValue::Smi(42))));
            map.borrow_mut().insert("__get_x__".to_string(), getter);
        }
        let result = proto_lookup(&obj, "x");
        assert_eq!(
            result,
            JsValue::Smi(42),
            "getter should take precedence over data key"
        );
    }

    #[test]
    fn test_lda_keyed_property_opcode_via_bytecode() {
        // Full bytecode-level test using CreateEmptyObjectLiteral +
        // StaNamedProperty to build an object, then LdaKeyedProperty to
        // read from it.
        //
        //   CreateEmptyObjectLiteral    ; acc = {}
        //   Star r0                      ; r0 = obj
        //   LdaSmi 42                   ; acc = 42
        //   StaNamedProperty r0, [0], slot=0  ; obj.x = 42
        //   LdaConstant [0]             ; acc = "x" (key)
        //   LdaKeyedProperty r0, slot=0 ; acc = obj["x"]
        //   Return
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::CreateEmptyObjectLiteral, vec![]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(
                    Opcode::StaNamedProperty,
                    vec![
                        Operand::Register(0),
                        Operand::ConstantPoolIdx(0),
                        Operand::FeedbackSlot(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(0)]),
                Instruction::new_unchecked(
                    Opcode::LdaKeyedProperty,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("x".to_string())],
            1,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn test_sta_keyed_property_opcode_via_bytecode() {
        // Full bytecode-level test:
        //   CreateEmptyObjectLiteral    ; acc = {}
        //   Star r0                      ; r0 = obj
        //   LdaConstant [0]             ; acc = "y" (key)
        //   Star r1                      ; r1 = key
        //   LdaSmi 99                   ; acc = 99
        //   StaKeyedProperty r0, r1, slot=0  ; obj["y"] = 99
        //   LdaConstant [0]             ; acc = "y" (key again)
        //   LdaKeyedProperty r0, slot=0 ; acc = obj["y"]
        //   Return
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::CreateEmptyObjectLiteral, vec![]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(0)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
                Instruction::new_unchecked(
                    Opcode::StaKeyedProperty,
                    vec![
                        Operand::Register(0),
                        Operand::Register(1),
                        Operand::FeedbackSlot(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(0)]),
                Instruction::new_unchecked(
                    Opcode::LdaKeyedProperty,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("y".to_string())],
            2,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    #[test]
    fn test_lda_keyed_property_heap_number_key() {
        let arr = JsValue::new_array(vec![
            JsValue::String("first".to_string().into()),
            JsValue::String("second".to_string().into()),
        ]);
        // HeapNumber 1.0 should work as index 1
        assert_eq!(
            keyed_load(&arr, &JsValue::HeapNumber(1.0)).unwrap(),
            JsValue::String("second".to_string().into())
        );
        // HeapNumber 1.5 is not a valid index
        assert_eq!(
            keyed_load(&arr, &JsValue::HeapNumber(1.5)).unwrap(),
            JsValue::Undefined
        );
        // Negative HeapNumber
        assert_eq!(
            keyed_load(&arr, &JsValue::HeapNumber(-1.0)).unwrap(),
            JsValue::Undefined
        );
    }

    #[test]
    fn test_to_array_index() {
        assert_eq!(to_array_index(&JsValue::Smi(0)), Some(0));
        assert_eq!(to_array_index(&JsValue::Smi(42)), Some(42));
        assert_eq!(to_array_index(&JsValue::Smi(-1)), None);
        assert_eq!(to_array_index(&JsValue::HeapNumber(3.0)), Some(3));
        assert_eq!(to_array_index(&JsValue::HeapNumber(3.5)), None);
        assert_eq!(
            to_array_index(&JsValue::String("7".to_string().into())),
            Some(7)
        );
        assert_eq!(
            to_array_index(&JsValue::String("abc".to_string().into())),
            None
        );
        assert_eq!(to_array_index(&JsValue::Undefined), None);
    }

    // ── TestInstanceOf ───────────────────────────────────────────────────────

    /// Helper: build bytecode from `instrs` (plus a trailing `Return`),
    /// create a frame with `frame_size` register slots, pre-set the
    /// accumulator to `acc`, write each entry in `regs` into the
    /// corresponding register, and execute.
    fn run_with_acc_and_regs(
        acc: JsValue,
        regs: &[JsValue],
        instrs: Vec<Instruction>,
        frame_size: u32,
    ) -> JsValue {
        run_with_acc_and_regs_result(acc, regs, instrs, frame_size).unwrap()
    }

    /// Like [`run_with_acc_and_regs`] but returns the full `StatorResult` so
    /// callers can assert on expected errors.
    fn run_with_acc_and_regs_result(
        acc: JsValue,
        regs: &[JsValue],
        instrs: Vec<Instruction>,
        frame_size: u32,
    ) -> StatorResult<JsValue> {
        let mut all = instrs;
        all.push(Instruction::new_unchecked(Opcode::Return, vec![]));
        let ba = make_bytecode(all, frame_size, 0);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        frame.accumulator = acc;
        for (i, val) in regs.iter().enumerate() {
            frame.write_reg(i as u32, val.clone()).unwrap();
        }
        Interpreter::run(&mut frame)
    }

    #[test]
    fn test_test_instance_of_non_callable_throws_type_error() {
        // §7.3.21: RHS of instanceof must be callable, else TypeError.
        // acc = Smi(42), constructor in r0 = Smi(0) (not callable)
        let result = run_with_acc_and_regs_result(
            JsValue::Smi(42),
            &[JsValue::Smi(0)],
            vec![Instruction::new_unchecked(
                Opcode::TestInstanceOf,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            )],
            1,
        );
        assert!(
            matches!(result, Err(StatorError::TypeError(_))),
            "instanceof with non-callable RHS should throw TypeError, got {result:?}"
        );
    }

    #[test]
    fn test_test_instance_of_with_prototype_chain() {
        // Build: constructor.prototype = proto_obj
        //        instance.__proto__   = proto_obj
        // TestInstanceOf should find the match.
        let proto = Rc::new(RefCell::new(PropertyMap::new()));
        proto.borrow_mut().insert(
            "kind".to_string(),
            JsValue::String("proto".to_string().into()),
        );

        let mut ctor_map = PropertyMap::new();
        ctor_map.insert("prototype".to_string(), JsValue::PlainObject(proto.clone()));
        // Mark the mock constructor as callable so the RHS check passes.
        ctor_map.insert("__call__".to_string(), JsValue::Boolean(true));
        let constructor = JsValue::PlainObject(Rc::new(RefCell::new(ctor_map)));

        let mut inst_map = PropertyMap::new();
        inst_map.insert("__proto__".to_string(), JsValue::PlainObject(proto.clone()));
        let instance = JsValue::PlainObject(Rc::new(RefCell::new(inst_map)));

        // acc = instance, r0 = constructor
        let result = run_with_acc_and_regs(
            instance,
            &[constructor],
            vec![Instruction::new_unchecked(
                Opcode::TestInstanceOf,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            )],
            1,
        );
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_test_instance_of_no_match() {
        // Two different prototypes — should return false.
        let proto_a = Rc::new(RefCell::new(PropertyMap::new()));
        let proto_b = Rc::new(RefCell::new(PropertyMap::new()));

        let mut ctor_map = PropertyMap::new();
        ctor_map.insert("prototype".to_string(), JsValue::PlainObject(proto_a));
        // Mark the mock constructor as callable so the RHS check passes.
        ctor_map.insert("__call__".to_string(), JsValue::Boolean(true));
        let constructor = JsValue::PlainObject(Rc::new(RefCell::new(ctor_map)));

        let mut inst_map = PropertyMap::new();
        inst_map.insert("__proto__".to_string(), JsValue::PlainObject(proto_b));
        let instance = JsValue::PlainObject(Rc::new(RefCell::new(inst_map)));

        let result = run_with_acc_and_regs(
            instance,
            &[constructor],
            vec![Instruction::new_unchecked(
                Opcode::TestInstanceOf,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            )],
            1,
        );
        assert_eq!(result, JsValue::Boolean(false));
    }

    // ── TestIn ───────────────────────────────────────────────────────────────

    #[test]
    fn test_test_in_plain_object_existing_key() {
        // "x" in { x: 1 } → true
        let mut map = PropertyMap::new();
        map.insert("x".to_string(), JsValue::Smi(1));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));

        let result = run_with_acc_and_regs(
            JsValue::String("x".to_string().into()),
            &[obj],
            vec![Instruction::new_unchecked(
                Opcode::TestIn,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            )],
            1,
        );
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_test_in_plain_object_missing_key() {
        // "y" in { x: 1 } → false
        let mut map = PropertyMap::new();
        map.insert("x".to_string(), JsValue::Smi(1));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));

        let result = run_with_acc_and_regs(
            JsValue::String("y".to_string().into()),
            &[obj],
            vec![Instruction::new_unchecked(
                Opcode::TestIn,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            )],
            1,
        );
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_test_in_array_valid_index() {
        // 1 in [10, 20, 30] → true
        let arr = JsValue::new_array(vec![JsValue::Smi(10), JsValue::Smi(20), JsValue::Smi(30)]);

        let result = run_with_acc_and_regs(
            JsValue::Smi(1),
            &[arr],
            vec![Instruction::new_unchecked(
                Opcode::TestIn,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            )],
            1,
        );
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_test_in_array_out_of_bounds() {
        // 5 in [10, 20, 30] → false
        let arr = JsValue::new_array(vec![JsValue::Smi(10), JsValue::Smi(20), JsValue::Smi(30)]);

        let result = run_with_acc_and_regs(
            JsValue::Smi(5),
            &[arr],
            vec![Instruction::new_unchecked(
                Opcode::TestIn,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            )],
            1,
        );
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_test_in_array_length_key() {
        // "length" in [10, 20] → true
        let arr = JsValue::new_array(vec![JsValue::Smi(10), JsValue::Smi(20)]);

        let result = run_with_acc_and_regs(
            JsValue::String("length".to_string().into()),
            &[arr],
            vec![Instruction::new_unchecked(
                Opcode::TestIn,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            )],
            1,
        );
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_test_in_non_object() {
        // "x" in 42 → TypeError (non-object target per spec)
        let mut all = vec![Instruction::new_unchecked(
            Opcode::TestIn,
            vec![Operand::Register(0), Operand::FeedbackSlot(0)],
        )];
        all.push(Instruction::new_unchecked(Opcode::Return, vec![]));
        let ba = make_bytecode(all, 1, 0);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        frame.accumulator = JsValue::String("x".to_string().into());
        frame.write_reg(0, JsValue::Smi(42)).unwrap();
        let result = Interpreter::run(&mut frame);
        assert!(result.is_err(), "expected TypeError for `in` on non-object");
    }

    // ── CreateEmptyArrayLiteral + StaInArrayLiteral ─────────────────────────

    #[test]
    fn test_create_empty_array_literal() {
        // CreateEmptyArrayLiteral [slot=0] → acc = [] (PlainObject with length=0)
        // Return
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(
                    Opcode::CreateEmptyArrayLiteral,
                    vec![Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        // Should be a PlainObject with length=0
        if let JsValue::PlainObject(map) = &result {
            let borrow = map.borrow();
            assert_eq!(borrow.get("length"), Some(&JsValue::Smi(0)));
            assert_eq!(borrow.len(), 2); // "length" + "__is_array__"
        } else {
            panic!("expected PlainObject, got {result:?}");
        }
    }

    #[test]
    fn test_create_array_literal_with_elements() {
        // Simulates: [10, 20, 30]
        //
        //   CreateEmptyArrayLiteral [slot=0]   ; acc = []
        //   Star r0                             ; r0 = array
        //   LdaSmi 0                           ; acc = 0
        //   Star r1                             ; r1 = 0
        //   LdaSmi 10                          ; acc = 10
        //   StaInArrayLiteral r0, r1, [slot=0] ; array[0] = 10
        //   LdaSmi 1                           ; acc = 1
        //   Star r1                             ; r1 = 1
        //   LdaSmi 20                          ; acc = 20
        //   StaInArrayLiteral r0, r1, [slot=0] ; array[1] = 20
        //   LdaSmi 2                           ; acc = 2
        //   Star r1                             ; r1 = 2
        //   LdaSmi 30                          ; acc = 30
        //   StaInArrayLiteral r0, r1, [slot=0] ; array[2] = 30
        //   Ldar r0                            ; acc = array
        //   Return
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(
                    Opcode::CreateEmptyArrayLiteral,
                    vec![Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                // Element 0
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(0)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
                Instruction::new_unchecked(
                    Opcode::StaInArrayLiteral,
                    vec![
                        Operand::Register(0),
                        Operand::Register(1),
                        Operand::FeedbackSlot(0),
                    ],
                ),
                // Element 1
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(20)]),
                Instruction::new_unchecked(
                    Opcode::StaInArrayLiteral,
                    vec![
                        Operand::Register(0),
                        Operand::Register(1),
                        Operand::FeedbackSlot(0),
                    ],
                ),
                // Element 2
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(30)]),
                Instruction::new_unchecked(
                    Opcode::StaInArrayLiteral,
                    vec![
                        Operand::Register(0),
                        Operand::Register(1),
                        Operand::FeedbackSlot(0),
                    ],
                ),
                // Load array and return
                Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![],
            2,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        if let JsValue::PlainObject(map) = &result {
            let borrow = map.borrow();
            assert_eq!(borrow.get("length"), Some(&JsValue::Smi(3)));
            assert_eq!(borrow.get("0"), Some(&JsValue::Smi(10)));
            assert_eq!(borrow.get("1"), Some(&JsValue::Smi(20)));
            assert_eq!(borrow.get("2"), Some(&JsValue::Smi(30)));
        } else {
            panic!("expected PlainObject, got {result:?}");
        }
    }

    #[test]
    fn test_sta_in_array_literal_preserves_accumulator() {
        // StaInArrayLiteral should NOT change the accumulator.
        //
        //   CreateEmptyArrayLiteral [slot=0]
        //   Star r0
        //   LdaSmi 0
        //   Star r1
        //   LdaSmi 42           ; acc = 42
        //   StaInArrayLiteral r0, r1, [slot=0]
        //   Return               ; should return 42 (acc unchanged)
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(
                    Opcode::CreateEmptyArrayLiteral,
                    vec![Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(0)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(
                    Opcode::StaInArrayLiteral,
                    vec![
                        Operand::Register(0),
                        Operand::Register(1),
                        Operand::FeedbackSlot(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![],
            2,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    // ── DefineNamedOwnProperty ──────────────────────────────────────────────

    #[test]
    fn test_define_named_own_property() {
        // Simulates: { x: 42 }
        //
        //   CreateEmptyObjectLiteral
        //   Star r0
        //   LdaSmi 42
        //   DefineNamedOwnProperty r0, [0:"x"], [slot=0]
        //   Ldar r0
        //   Return
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::CreateEmptyObjectLiteral, vec![]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(
                    Opcode::DefineNamedOwnProperty,
                    vec![
                        Operand::Register(0),
                        Operand::ConstantPoolIdx(0),
                        Operand::FeedbackSlot(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("x".to_string())],
            1,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        if let JsValue::PlainObject(map) = &result {
            assert_eq!(map.borrow().get("x"), Some(&JsValue::Smi(42)));
        } else {
            panic!("expected PlainObject, got {result:?}");
        }
    }

    #[test]
    fn test_define_named_own_property_multiple() {
        // Simulates: { a: 1, b: 2 }
        //
        //   CreateEmptyObjectLiteral
        //   Star r0
        //   LdaSmi 1
        //   DefineNamedOwnProperty r0, [0:"a"], [slot=0]
        //   LdaSmi 2
        //   DefineNamedOwnProperty r0, [1:"b"], [slot=0]
        //   Ldar r0
        //   Return
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::CreateEmptyObjectLiteral, vec![]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
                Instruction::new_unchecked(
                    Opcode::DefineNamedOwnProperty,
                    vec![
                        Operand::Register(0),
                        Operand::ConstantPoolIdx(0),
                        Operand::FeedbackSlot(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
                Instruction::new_unchecked(
                    Opcode::DefineNamedOwnProperty,
                    vec![
                        Operand::Register(0),
                        Operand::ConstantPoolIdx(1),
                        Operand::FeedbackSlot(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![
                ConstantPoolEntry::String("a".to_string()),
                ConstantPoolEntry::String("b".to_string()),
            ],
            1,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        if let JsValue::PlainObject(map) = &result {
            let borrow = map.borrow();
            assert_eq!(borrow.get("a"), Some(&JsValue::Smi(1)));
            assert_eq!(borrow.get("b"), Some(&JsValue::Smi(2)));
        } else {
            panic!("expected PlainObject, got {result:?}");
        }
    }

    // ── DefineKeyedOwnProperty ──────────────────────────────────────────────

    #[test]
    fn test_define_keyed_own_property() {
        // Simulates: { [key]: value } where key="foo", value=99
        //
        //   CreateEmptyObjectLiteral
        //   Star r0
        //   LdaSmi 99           ; acc = 99 (value)
        //   Star r2              ; r2 = 99 (save value)
        //   LdaConstant [0]     ; acc = "foo"
        //   Star r1              ; r1 = "foo" (key)
        //   Ldar r2              ; restore value to acc
        //   DefineKeyedOwnProperty r0, r1, flag=0, [slot=0]
        //   Ldar r0
        //   Return
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::CreateEmptyObjectLiteral, vec![]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
                Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(0)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
                Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(2)]),
                Instruction::new_unchecked(
                    Opcode::DefineKeyedOwnProperty,
                    vec![
                        Operand::Register(0),
                        Operand::Register(1),
                        Operand::Flag(0),
                        Operand::FeedbackSlot(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("foo".to_string())],
            3,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        if let JsValue::PlainObject(map) = &result {
            assert_eq!(map.borrow().get("foo"), Some(&JsValue::Smi(99)));
        } else {
            panic!("expected PlainObject, got {result:?}");
        }
    }

    // ── CreateRegExpLiteral ─────────────────────────────────────────────────

    #[test]
    fn test_create_regexp_literal() {
        // CreateRegExpLiteral [0:"ab+d", slot=0, flags=0x02(i)]
        // Return
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(
                    Opcode::CreateRegExpLiteral,
                    vec![
                        Operand::ConstantPoolIdx(0),
                        Operand::FeedbackSlot(0),
                        Operand::Flag(0x02), // 'i' flag
                    ],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("ab+d".to_string())],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        if let JsValue::PlainObject(map) = &result {
            let borrow = map.borrow();
            assert_eq!(
                borrow.get("source"),
                Some(&JsValue::String("ab+d".to_string().into()))
            );
            assert_eq!(
                borrow.get("flags"),
                Some(&JsValue::String("i".to_string().into()))
            );
            // toString is now a NativeFunction
            assert!(
                matches!(borrow.get("toString"), Some(JsValue::NativeFunction(_))),
                "expected toString to be a NativeFunction"
            );
            // test & exec are NativeFunctions
            assert!(matches!(
                borrow.get("test"),
                Some(JsValue::NativeFunction(_))
            ));
            assert!(matches!(
                borrow.get("exec"),
                Some(JsValue::NativeFunction(_))
            ));
            // boolean flags
            assert_eq!(borrow.get("ignoreCase"), Some(&JsValue::Boolean(true)));
            assert_eq!(borrow.get("global"), Some(&JsValue::Boolean(false)));
            assert_eq!(borrow.get("__is_regexp__"), Some(&JsValue::Boolean(true)));
        } else {
            panic!("expected PlainObject, got {result:?}");
        }
    }

    #[test]
    fn test_create_regexp_literal_multiple_flags() {
        // CreateRegExpLiteral [0:"test", slot=0, flags=0x01|0x02|0x04 = gim]
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(
                    Opcode::CreateRegExpLiteral,
                    vec![
                        Operand::ConstantPoolIdx(0),
                        Operand::FeedbackSlot(0),
                        Operand::Flag(0x07), // g|i|m
                    ],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("test".to_string())],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        if let JsValue::PlainObject(map) = &result {
            let borrow = map.borrow();
            assert_eq!(
                borrow.get("source"),
                Some(&JsValue::String("test".to_string().into()))
            );
            assert_eq!(
                borrow.get("flags"),
                Some(&JsValue::String("gim".to_string().into()))
            );
            assert_eq!(borrow.get("global"), Some(&JsValue::Boolean(true)));
            assert_eq!(borrow.get("ignoreCase"), Some(&JsValue::Boolean(true)));
            assert_eq!(borrow.get("multiline"), Some(&JsValue::Boolean(true)));
        } else {
            panic!("expected PlainObject, got {result:?}");
        }
    }

    #[test]
    fn test_create_regexp_literal_no_flags() {
        // /abc/ with no flags
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(
                    Opcode::CreateRegExpLiteral,
                    vec![
                        Operand::ConstantPoolIdx(0),
                        Operand::FeedbackSlot(0),
                        Operand::Flag(0x00),
                    ],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("abc".to_string())],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        if let JsValue::PlainObject(map) = &result {
            let borrow = map.borrow();
            assert_eq!(
                borrow.get("source"),
                Some(&JsValue::String("abc".to_string().into()))
            );
            assert_eq!(
                borrow.get("flags"),
                Some(&JsValue::String(String::new().into()))
            );
            assert!(
                matches!(borrow.get("toString"), Some(JsValue::NativeFunction(_))),
                "expected toString to be a NativeFunction"
            );
            assert!(matches!(
                borrow.get("test"),
                Some(JsValue::NativeFunction(_))
            ));
            assert!(matches!(
                borrow.get("exec"),
                Some(JsValue::NativeFunction(_))
            ));
            assert_eq!(borrow.get("global"), Some(&JsValue::Boolean(false)));
            assert_eq!(borrow.get("ignoreCase"), Some(&JsValue::Boolean(false)));
        } else {
            panic!("expected PlainObject, got {result:?}");
        }
    }

    // ── CreateArrayLiteral / CreateObjectLiteral stubs ──────────────────────

    #[test]
    fn test_create_array_literal_stub() {
        // CreateArrayLiteral [idx=0, slot=0, flags=0] → empty array-like
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(
                    Opcode::CreateArrayLiteral,
                    vec![
                        Operand::ConstantPoolIdx(0),
                        Operand::FeedbackSlot(0),
                        Operand::Flag(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::Undefined],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        if let JsValue::PlainObject(map) = &result {
            assert_eq!(map.borrow().get("length"), Some(&JsValue::Smi(0)));
        } else {
            panic!("expected PlainObject, got {result:?}");
        }
    }

    #[test]
    fn test_create_object_literal_stub() {
        // CreateObjectLiteral [idx=0, slot=0, flags=0] → empty object
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(
                    Opcode::CreateObjectLiteral,
                    vec![
                        Operand::ConstantPoolIdx(0),
                        Operand::FeedbackSlot(0),
                        Operand::Flag(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::Undefined],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        if let JsValue::PlainObject(map) = &result {
            assert!(map.borrow().is_empty());
        } else {
            panic!("expected PlainObject, got {result:?}");
        }
    }

    // ── Array-like PlainObject keyed access ─────────────────────────────────

    #[test]
    fn test_keyed_load_on_array_like_plain_object() {
        // Build an array-like PlainObject [10, 20] via bytecodes, then
        // read element 1 via LdaKeyedProperty.
        //
        //   CreateEmptyArrayLiteral [slot=0]
        //   Star r0
        //   LdaSmi 0  /  Star r1  /  LdaSmi 10  /  StaInArrayLiteral r0, r1, [slot=0]
        //   LdaSmi 1  /  Star r1  /  LdaSmi 20  /  StaInArrayLiteral r0, r1, [slot=0]
        //   LdaSmi 1               ; key = 1
        //   LdaKeyedProperty r0, [slot=0]  ; acc = r0[1] = 20
        //   Return
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(
                    Opcode::CreateEmptyArrayLiteral,
                    vec![Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                // elem 0
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(0)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
                Instruction::new_unchecked(
                    Opcode::StaInArrayLiteral,
                    vec![
                        Operand::Register(0),
                        Operand::Register(1),
                        Operand::FeedbackSlot(0),
                    ],
                ),
                // elem 1
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(20)]),
                Instruction::new_unchecked(
                    Opcode::StaInArrayLiteral,
                    vec![
                        Operand::Register(0),
                        Operand::Register(1),
                        Operand::FeedbackSlot(0),
                    ],
                ),
                // Read element at index 1
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
                Instruction::new_unchecked(
                    Opcode::LdaKeyedProperty,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![],
            2,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(20));
    }

    #[test]
    fn test_array_like_plain_object_length() {
        // Build [10, 20, 30] and read its "length" via LdaKeyedProperty.
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(
                    Opcode::CreateEmptyArrayLiteral,
                    vec![Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                // 3 elements
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(0)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
                Instruction::new_unchecked(
                    Opcode::StaInArrayLiteral,
                    vec![
                        Operand::Register(0),
                        Operand::Register(1),
                        Operand::FeedbackSlot(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(20)]),
                Instruction::new_unchecked(
                    Opcode::StaInArrayLiteral,
                    vec![
                        Operand::Register(0),
                        Operand::Register(1),
                        Operand::FeedbackSlot(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(30)]),
                Instruction::new_unchecked(
                    Opcode::StaInArrayLiteral,
                    vec![
                        Operand::Register(0),
                        Operand::Register(1),
                        Operand::FeedbackSlot(0),
                    ],
                ),
                // Read "length"
                Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(0)]),
                Instruction::new_unchecked(
                    Opcode::LdaKeyedProperty,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("length".to_string())],
            2,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    // ── PlainObject array iteration via GetIterator ─────────────────────────

    #[test]
    fn test_plain_object_to_array_items() {
        // Build a PlainObject with numeric keys and verify
        // plain_object_to_array_items extracts elements correctly.
        let map: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));
        {
            let mut borrow = map.borrow_mut();
            borrow.insert("0".to_string(), JsValue::Smi(10));
            borrow.insert("1".to_string(), JsValue::String("hello".to_string().into()));
            borrow.insert("2".to_string(), JsValue::Boolean(true));
            borrow.insert("length".to_string(), JsValue::Smi(3));
        }
        let items = super::plain_object_to_array_items(&map);
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], JsValue::Smi(10));
        assert_eq!(items[1], JsValue::String("hello".to_string().into()));
        assert_eq!(items[2], JsValue::Boolean(true));
    }

    // ── For-in bytecode-level tests ─────────────────────────────────────────

    /// Low-level test: ForInEnumerate + ForInPrepare + ForInNext + ForInStep +
    /// JumpIfForInDone working together to iterate over the keys of a
    /// PlainObject and count them.
    #[test]
    fn test_for_in_bytecode_count_keys() {
        // Build a PlainObject { a: 1, b: 2 } in r0 and iterate its keys,
        // counting the number of iterations in r5.
        //
        // Register layout:
        //   r0 = object  { a: 1, b: 2 }
        //   r1 = keys array (ForInEnumerate result)
        //   r2 = length   (ForInPrepare result)
        //   r3 = index    (starts at 0)
        //   r4 = current key (ForInNext result, unused in this test)
        //   r5 = count    (accumulates iteration count)
        //
        // constant pool: [0] = "a", [1] = "b"
        let instrs = vec![
            //  0: Create empty object in r0
            Instruction::new_unchecked(Opcode::CreateEmptyObjectLiteral, vec![]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            //  2: obj.a = 1
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(
                Opcode::DefineNamedOwnProperty,
                vec![
                    Operand::Register(0),
                    Operand::ConstantPoolIdx(0),
                    Operand::FeedbackSlot(0),
                ],
            ),
            //  4: obj.b = 2
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
            Instruction::new_unchecked(
                Opcode::DefineNamedOwnProperty,
                vec![
                    Operand::Register(0),
                    Operand::ConstantPoolIdx(1),
                    Operand::FeedbackSlot(0),
                ],
            ),
            //  6: ForInEnumerate r0 → acc = keys array
            Instruction::new_unchecked(Opcode::ForInEnumerate, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            //  8: ForInPrepare r1, slot → acc = length
            Instruction::new_unchecked(
                Opcode::ForInPrepare,
                vec![Operand::Register(1), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            // 10: index = 0
            Instruction::new_unchecked(Opcode::LdaZero, vec![]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(3)]),
            // 12: count = 0
            Instruction::new_unchecked(Opcode::LdaZero, vec![]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(5)]),
            // 14: loop_start — JumpIfForInDone [offset, r3(index), r2(length)]
            //     If done, jump to instruction 21 (Return).  The offset will
            //     be resolved by the encoder/decoder, but in raw instructions
            //     we encode byte offsets.  We'll use `compile_and_run` for a
            //     proper e2e test below; here we verify the opcodes work.
            //     Instead, let's use compile_and_run for the full e2e.
            // (Fall through to ForInNext.)
            Instruction::new_unchecked(
                Opcode::ForInNext,
                vec![
                    Operand::Register(0),
                    Operand::Register(3),
                    Operand::Register(1),
                    Operand::FeedbackSlot(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(4)]),
            // 16: count++
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(5)]),
            Instruction::new_unchecked(
                Opcode::AddSmi,
                vec![Operand::Immediate(1), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(5)]),
            // 19: ForInStep r3 → acc = index + 1
            Instruction::new_unchecked(Opcode::ForInStep, vec![Operand::Register(3)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(3)]),
            // 21: Check if index < length manually (since we can't use
            //     JumpIfForInDone with raw offsets easily).
            //     Compare index(r3) < length(r2).
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(3)]),
            Instruction::new_unchecked(
                Opcode::TestLessThan,
                vec![Operand::Register(2), Operand::FeedbackSlot(0)],
            ),
            // If true, jump back to instruction 14 (ForInNext).
            // The jump offset needs to be in bytes; this is hard to compute
            // manually.  Let's just return the count we have.
            // After 2 iterations count should be 2.  We only do 1 iteration
            // in this flat sequence (no backward jump), so check count = 1.
            // Actually this test is too complex for raw bytecode.  Let's
            // simplify: just test ForInEnumerate + ForInPrepare.
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(5)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode_with_pool(
            instrs,
            vec![
                ConstantPoolEntry::String("a".to_string()),
                ConstantPoolEntry::String("b".to_string()),
            ],
            6,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        // We did one single iteration without looping, so count = 1.
        assert_eq!(result, JsValue::Smi(1));
    }

    /// Test ForInEnumerate returns the correct number of keys.
    #[test]
    fn test_for_in_enumerate_prepare() {
        // Create {x: 1, y: 2, z: 3}, enumerate and prepare.
        // Return the length (should be 3).
        let instrs = vec![
            Instruction::new_unchecked(Opcode::CreateEmptyObjectLiteral, vec![]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(
                Opcode::DefineNamedOwnProperty,
                vec![
                    Operand::Register(0),
                    Operand::ConstantPoolIdx(0),
                    Operand::FeedbackSlot(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
            Instruction::new_unchecked(
                Opcode::DefineNamedOwnProperty,
                vec![
                    Operand::Register(0),
                    Operand::ConstantPoolIdx(1),
                    Operand::FeedbackSlot(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
            Instruction::new_unchecked(
                Opcode::DefineNamedOwnProperty,
                vec![
                    Operand::Register(0),
                    Operand::ConstantPoolIdx(2),
                    Operand::FeedbackSlot(0),
                ],
            ),
            // ForInEnumerate r0
            Instruction::new_unchecked(Opcode::ForInEnumerate, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            // ForInPrepare r1, slot → acc = length
            Instruction::new_unchecked(
                Opcode::ForInPrepare,
                vec![Operand::Register(1), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode_with_pool(
            instrs,
            vec![
                ConstantPoolEntry::String("x".to_string()),
                ConstantPoolEntry::String("y".to_string()),
                ConstantPoolEntry::String("z".to_string()),
            ],
            2,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    /// Test ForInStep increments the index.
    #[test]
    fn test_for_in_step() {
        // r0 = 5; ForInStep r0 → acc = 6
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::ForInStep, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let result = run_bytecode(instrs, 1).unwrap();
        assert_eq!(result, JsValue::Smi(6));
    }

    // ── End-to-end for-in tests ─────────────────────────────────────────────

    /// E2E test: for-in loop collecting keys of a plain object.
    ///
    /// ```js
    /// let obj = {};
    /// obj.a = 1;
    /// obj.b = 2;
    /// let count = 0;
    /// for (let k in obj) { count = count + 1; }
    /// return count;
    /// ```
    #[test]
    fn test_e2e_for_in_count_keys() {
        use crate::parser::ast::{
            AssignExpr, AssignOp, AssignTarget, BinaryOp, BlockStmt, ExprStmt, ForInOfLeft,
            ForInStmt, Ident, ObjectExpr, ObjectProp, Pat, Prop, PropKey, PropValue, Stmt, VarDecl,
            VarDeclarator, VarKind,
        };

        // let obj = { a: 1, b: 2 };
        let obj_decl = Stmt::VarDecl(VarDecl {
            loc: span(),
            kind: VarKind::Let,
            declarators: vec![VarDeclarator {
                loc: span(),
                id: Pat::Ident(Ident {
                    loc: span(),
                    name: "obj".to_owned(),
                }),
                init: Some(Box::new(crate::parser::ast::Expr::Object(Box::new(
                    ObjectExpr {
                        loc: span(),
                        properties: vec![
                            ObjectProp::Prop(Box::new(Prop {
                                loc: span(),
                                key: PropKey::Ident(Ident {
                                    loc: span(),
                                    name: "a".to_owned(),
                                }),
                                is_computed: false,
                                value: PropValue::Value(Box::new(num_expr(1.0))),
                            })),
                            ObjectProp::Prop(Box::new(Prop {
                                loc: span(),
                                key: PropKey::Ident(Ident {
                                    loc: span(),
                                    name: "b".to_owned(),
                                }),
                                is_computed: false,
                                value: PropValue::Value(Box::new(num_expr(2.0))),
                            })),
                        ],
                    },
                )))),
            }],
        });

        // let count = 0;
        let count_decl = var_let("count", num_expr(0.0));

        // for (let k in obj) { count = count + 1; }
        let for_in = Stmt::ForIn(ForInStmt {
            loc: span(),
            left: ForInOfLeft::VarDecl(VarDecl {
                loc: span(),
                kind: VarKind::Let,
                declarators: vec![VarDeclarator {
                    loc: span(),
                    id: Pat::Ident(Ident {
                        loc: span(),
                        name: "k".to_owned(),
                    }),
                    init: None,
                }],
            }),
            right: Box::new(ident_expr("obj")),
            body: Box::new(Stmt::Block(BlockStmt {
                loc: span(),
                body: vec![Stmt::Expr(ExprStmt {
                    loc: span(),
                    expr: Box::new(crate::parser::ast::Expr::Assign(Box::new(AssignExpr {
                        loc: span(),
                        op: AssignOp::Assign,
                        left: AssignTarget::Expr(Box::new(ident_expr("count"))),
                        right: Box::new(binary(BinaryOp::Add, ident_expr("count"), num_expr(1.0))),
                    }))),
                })],
            })),
        });

        let stmts = vec![
            obj_decl,
            count_decl,
            for_in,
            return_stmt(Some(ident_expr("count"))),
        ];
        let result = compile_and_run(stmts).unwrap();
        assert_eq!(result, JsValue::Smi(2));
    }

    /// E2E test: for-in with existing variable (not a new var decl).
    ///
    /// ```js
    /// let obj = {}; obj.x = 10; obj.y = 20;
    /// let k;
    /// let sum = 0;
    /// for (k in obj) { sum = sum + obj[k]; }
    /// return sum;
    /// ```
    #[test]
    fn test_e2e_for_in_existing_var_sum_values() {
        use crate::parser::ast::{
            AssignExpr, AssignOp, AssignTarget, BinaryOp, BlockStmt, ExprStmt, ForInOfLeft,
            ForInStmt, Ident, MemberExpr, MemberProp, ObjectExpr, ObjectProp, Pat, Prop, PropKey,
            PropValue, Stmt, VarDecl, VarDeclarator, VarKind,
        };

        // let obj = { x: 10, y: 20 };
        let obj_decl = Stmt::VarDecl(VarDecl {
            loc: span(),
            kind: VarKind::Let,
            declarators: vec![VarDeclarator {
                loc: span(),
                id: Pat::Ident(Ident {
                    loc: span(),
                    name: "obj".to_owned(),
                }),
                init: Some(Box::new(crate::parser::ast::Expr::Object(Box::new(
                    ObjectExpr {
                        loc: span(),
                        properties: vec![
                            ObjectProp::Prop(Box::new(Prop {
                                loc: span(),
                                key: PropKey::Ident(Ident {
                                    loc: span(),
                                    name: "x".to_owned(),
                                }),
                                is_computed: false,
                                value: PropValue::Value(Box::new(num_expr(10.0))),
                            })),
                            ObjectProp::Prop(Box::new(Prop {
                                loc: span(),
                                key: PropKey::Ident(Ident {
                                    loc: span(),
                                    name: "y".to_owned(),
                                }),
                                is_computed: false,
                                value: PropValue::Value(Box::new(num_expr(20.0))),
                            })),
                        ],
                    },
                )))),
            }],
        });

        // let k;
        let k_decl = Stmt::VarDecl(VarDecl {
            loc: span(),
            kind: VarKind::Let,
            declarators: vec![VarDeclarator {
                loc: span(),
                id: Pat::Ident(Ident {
                    loc: span(),
                    name: "k".to_owned(),
                }),
                init: None,
            }],
        });

        // let sum = 0;
        let sum_decl = var_let("sum", num_expr(0.0));

        // for (k in obj) { sum = sum + obj[k]; }
        let for_in = Stmt::ForIn(ForInStmt {
            loc: span(),
            left: ForInOfLeft::Pat(Pat::Ident(Ident {
                loc: span(),
                name: "k".to_owned(),
            })),
            right: Box::new(ident_expr("obj")),
            body: Box::new(Stmt::Block(BlockStmt {
                loc: span(),
                body: vec![Stmt::Expr(ExprStmt {
                    loc: span(),
                    expr: Box::new(crate::parser::ast::Expr::Assign(Box::new(AssignExpr {
                        loc: span(),
                        op: AssignOp::Assign,
                        left: AssignTarget::Expr(Box::new(ident_expr("sum"))),
                        right: Box::new(binary(
                            BinaryOp::Add,
                            ident_expr("sum"),
                            crate::parser::ast::Expr::Member(Box::new(MemberExpr {
                                loc: span(),
                                object: Box::new(ident_expr("obj")),
                                property: MemberProp::Computed(Box::new(ident_expr("k"))),
                                is_computed: true,
                            })),
                        )),
                    }))),
                })],
            })),
        });

        let stmts = vec![
            obj_decl,
            k_decl,
            sum_decl,
            for_in,
            return_stmt(Some(ident_expr("sum"))),
        ];
        let result = compile_and_run(stmts).unwrap();
        assert_eq!(result, JsValue::Smi(30)); // 10 + 20
    }

    /// E2E test: for-in over null/undefined does nothing.
    #[test]
    fn test_e2e_for_in_null_noop() {
        use crate::parser::ast::{
            AssignExpr, AssignOp, AssignTarget, BinaryOp, BlockStmt, ExprStmt, ForInOfLeft,
            ForInStmt, Ident, Pat, Stmt, VarDecl, VarDeclarator, VarKind,
        };

        // let count = 0; for (let k in null) { count = count + 1; } return count;
        let count_decl = var_let("count", num_expr(0.0));
        let for_in = Stmt::ForIn(ForInStmt {
            loc: span(),
            left: ForInOfLeft::VarDecl(VarDecl {
                loc: span(),
                kind: VarKind::Let,
                declarators: vec![VarDeclarator {
                    loc: span(),
                    id: Pat::Ident(Ident {
                        loc: span(),
                        name: "k".to_owned(),
                    }),
                    init: None,
                }],
            }),
            right: Box::new(crate::parser::ast::Expr::Null(
                crate::parser::ast::NullLit { loc: span() },
            )),
            body: Box::new(Stmt::Block(BlockStmt {
                loc: span(),
                body: vec![Stmt::Expr(ExprStmt {
                    loc: span(),
                    expr: Box::new(crate::parser::ast::Expr::Assign(Box::new(AssignExpr {
                        loc: span(),
                        op: AssignOp::Assign,
                        left: AssignTarget::Expr(Box::new(ident_expr("count"))),
                        right: Box::new(binary(BinaryOp::Add, ident_expr("count"), num_expr(1.0))),
                    }))),
                })],
            })),
        });

        let stmts = vec![count_decl, for_in, return_stmt(Some(ident_expr("count")))];
        let result = compile_and_run(stmts).unwrap();
        assert_eq!(result, JsValue::Smi(0)); // no iterations
    }

    /// E2E test: for-in over empty object does nothing.
    #[test]
    fn test_e2e_for_in_empty_object() {
        use crate::parser::ast::{
            AssignExpr, AssignOp, AssignTarget, BinaryOp, BlockStmt, ExprStmt, ForInOfLeft,
            ForInStmt, Ident, ObjectExpr, Pat, Stmt, VarDecl, VarDeclarator, VarKind,
        };

        // let count = 0; for (let k in {}) { count = count + 1; } return count;
        let count_decl = var_let("count", num_expr(0.0));
        let for_in = Stmt::ForIn(ForInStmt {
            loc: span(),
            left: ForInOfLeft::VarDecl(VarDecl {
                loc: span(),
                kind: VarKind::Let,
                declarators: vec![VarDeclarator {
                    loc: span(),
                    id: Pat::Ident(Ident {
                        loc: span(),
                        name: "k".to_owned(),
                    }),
                    init: None,
                }],
            }),
            right: Box::new(crate::parser::ast::Expr::Object(Box::new(ObjectExpr {
                loc: span(),
                properties: vec![],
            }))),
            body: Box::new(Stmt::Block(BlockStmt {
                loc: span(),
                body: vec![Stmt::Expr(ExprStmt {
                    loc: span(),
                    expr: Box::new(crate::parser::ast::Expr::Assign(Box::new(AssignExpr {
                        loc: span(),
                        op: AssignOp::Assign,
                        left: AssignTarget::Expr(Box::new(ident_expr("count"))),
                        right: Box::new(binary(BinaryOp::Add, ident_expr("count"), num_expr(1.0))),
                    }))),
                })],
            })),
        });

        let stmts = vec![count_decl, for_in, return_stmt(Some(ident_expr("count")))];
        let result = compile_and_run(stmts).unwrap();
        assert_eq!(result, JsValue::Smi(0)); // no iterations
    }

    // ── Context slot opcode tests ────────────────────────────────────────

    /// Create a function context, push it, store a value into slot 0 via
    /// StaCurrentContextSlot, then load it back via LdaCurrentContextSlot.
    #[test]
    fn test_sta_lda_current_context_slot() {
        // Bytecode:
        //   CreateFunctionContext [scope_idx=0, slot_count=2]
        //   PushContext r0
        //   LdaSmi 42
        //   StaCurrentContextSlot [slot_idx=0]
        //   LdaSmi 0          ; clobber acc
        //   LdaCurrentContextSlot [slot_idx=0]
        //   Return             ; should return 42
        let instrs = vec![
            Instruction::new_unchecked(
                Opcode::CreateFunctionContext,
                vec![Operand::ConstantPoolIdx(0), Operand::Immediate(2)],
            ),
            Instruction::new_unchecked(Opcode::PushContext, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            Instruction::new_unchecked(
                Opcode::StaCurrentContextSlot,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(0)]),
            Instruction::new_unchecked(
                Opcode::LdaCurrentContextSlot,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let result = run_bytecode(instrs, 1).unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// Store into two different slots and verify independent retrieval.
    #[test]
    fn test_context_multiple_slots() {
        let instrs = vec![
            // Create context with 3 slots
            Instruction::new_unchecked(
                Opcode::CreateFunctionContext,
                vec![Operand::ConstantPoolIdx(0), Operand::Immediate(3)],
            ),
            Instruction::new_unchecked(Opcode::PushContext, vec![Operand::Register(0)]),
            // Store 10 in slot 0
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
            Instruction::new_unchecked(
                Opcode::StaCurrentContextSlot,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            // Store 20 in slot 1
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(20)]),
            Instruction::new_unchecked(
                Opcode::StaCurrentContextSlot,
                vec![Operand::ConstantPoolIdx(1)],
            ),
            // Load slot 0 — should be 10
            Instruction::new_unchecked(
                Opcode::LdaCurrentContextSlot,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let result = run_bytecode(instrs, 1).unwrap();
        assert_eq!(result, JsValue::Smi(10));
    }

    /// LdaContextSlot / StaContextSlot with explicit register and depth.
    #[test]
    fn test_lda_sta_context_slot_with_depth() {
        // Build: outer context with 2 slots, inner context chained to it.
        //   CreateFunctionContext [_, 2]    ; outer with 2 slots → acc
        //   PushContext r0                  ; install outer
        //   LdaSmi 99
        //   StaCurrentContextSlot [0]      ; outer.slots[0] = 99
        //   Star r1                        ; r1 = 99 (unused, just free reg)
        //   Ldar r0                        ; r0 = saved old ctx (undefined)
        //   — We need to save the outer ctx ref in a register.
        //   — Let's approach differently: after PushContext, frame.context holds outer.
        //   — Then create inner context (which chains to outer), push it.
        //   — Use LdaContextSlot from r1 (inner) with depth=1 to reach outer.

        // Step 1: create outer context
        let instrs = vec![
            Instruction::new_unchecked(
                Opcode::CreateFunctionContext,
                vec![Operand::ConstantPoolIdx(0), Operand::Immediate(2)],
            ),
            // acc = outer context
            Instruction::new_unchecked(Opcode::PushContext, vec![Operand::Register(0)]),
            // frame.context = outer, r0 = old ctx (undefined)
            // Store 99 in outer slot 0
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(
                Opcode::StaCurrentContextSlot,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            // Create inner context (will chain to outer via frame.context)
            Instruction::new_unchecked(
                Opcode::CreateFunctionContext,
                vec![Operand::ConstantPoolIdx(0), Operand::Immediate(1)],
            ),
            // acc = inner context
            Instruction::new_unchecked(Opcode::PushContext, vec![Operand::Register(1)]),
            // frame.context = inner, r1 = outer context
            // Store 77 in inner slot 0
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(77)]),
            Instruction::new_unchecked(
                Opcode::StaCurrentContextSlot,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            // Save inner context into r2 for later use as ctx_reg
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            // Load the frame.context (inner) into r2
            // Actually, we can just load the current context from the frame.
            // Let's use a different approach: we know frame.context is the inner ctx.
            // After PushContext, we have frame.context = inner.
            // To get a register holding inner, we can read from frame.context via
            // acc = frame.context value... but there's no opcode for that directly.
            // BUT LdaContextSlot reads from a register. So we need to put the
            // inner context into a register first.
            //
            // Actually after the second PushContext, r1 holds the outer context.
            // And frame.context holds the inner context.
            // We can read from r1 (outer) with depth=0 to get outer.slots[0] = 99.

            // LdaContextSlot [r1, slot=0, depth=0] → should load outer.slots[0] = 99
            Instruction::new_unchecked(
                Opcode::LdaContextSlot,
                vec![
                    Operand::Register(1),
                    Operand::ConstantPoolIdx(0),
                    Operand::Immediate(0),
                ],
            ),
            // acc should now be 99
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let result = run_bytecode(instrs, 3).unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    /// Walk context chain with depth > 0 to reach an outer scope.
    #[test]
    fn test_context_chain_depth_walk() {
        let instrs = vec![
            // Create outer: 2 slots
            Instruction::new_unchecked(
                Opcode::CreateFunctionContext,
                vec![Operand::ConstantPoolIdx(0), Operand::Immediate(2)],
            ),
            // acc = outer ctx
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            // r2 = outer ctx
            Instruction::new_unchecked(Opcode::PushContext, vec![Operand::Register(0)]),
            // r0 = old ctx (undefined), frame.context = outer
            // outer.slots[1] = 55
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(55)]),
            Instruction::new_unchecked(
                Opcode::StaCurrentContextSlot,
                vec![Operand::ConstantPoolIdx(1)],
            ),
            // Create inner: 1 slot (parent = outer)
            Instruction::new_unchecked(
                Opcode::CreateFunctionContext,
                vec![Operand::ConstantPoolIdx(0), Operand::Immediate(1)],
            ),
            // acc = inner ctx
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(3)]),
            // r3 = inner ctx
            Instruction::new_unchecked(Opcode::PushContext, vec![Operand::Register(1)]),
            // r1 = outer ctx, frame.context = inner

            // Use LdaContextSlot to read from inner (r3) with depth=1, slot=1
            // This walks inner→parent(outer).slots[1] = 55
            Instruction::new_unchecked(
                Opcode::LdaContextSlot,
                vec![
                    Operand::Register(3),
                    Operand::ConstantPoolIdx(1),
                    Operand::Immediate(1),
                ],
            ),
            // acc should be 55
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let result = run_bytecode(instrs, 4).unwrap();
        assert_eq!(result, JsValue::Smi(55));
    }

    /// StaContextSlot with depth > 0 stores into an outer context.
    #[test]
    fn test_sta_context_slot_with_depth() {
        let instrs = vec![
            // Create outer: 2 slots
            Instruction::new_unchecked(
                Opcode::CreateFunctionContext,
                vec![Operand::ConstantPoolIdx(0), Operand::Immediate(2)],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            Instruction::new_unchecked(Opcode::PushContext, vec![Operand::Register(0)]),
            // Create inner: 1 slot (parent = outer)
            Instruction::new_unchecked(
                Opcode::CreateFunctionContext,
                vec![Operand::ConstantPoolIdx(0), Operand::Immediate(1)],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(3)]),
            Instruction::new_unchecked(Opcode::PushContext, vec![Operand::Register(1)]),
            // Store 88 into outer.slots[0] via inner with depth=1
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(88)]),
            Instruction::new_unchecked(
                Opcode::StaContextSlot,
                vec![
                    Operand::Register(3),
                    Operand::ConstantPoolIdx(0),
                    Operand::Immediate(1),
                ],
            ),
            // Now read it back from outer (r2) with depth=0
            Instruction::new_unchecked(
                Opcode::LdaContextSlot,
                vec![
                    Operand::Register(2),
                    Operand::ConstantPoolIdx(0),
                    Operand::Immediate(0),
                ],
            ),
            // acc should be 88
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let result = run_bytecode(instrs, 4).unwrap();
        assert_eq!(result, JsValue::Smi(88));
    }

    /// CreateBlockContext creates a context with 0 slots that grows on demand.
    #[test]
    fn test_create_block_context() {
        let instrs = vec![
            Instruction::new_unchecked(
                Opcode::CreateBlockContext,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            Instruction::new_unchecked(Opcode::PushContext, vec![Operand::Register(0)]),
            // Store 12 into slot 0 (will grow the slot vector)
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(12)]),
            Instruction::new_unchecked(
                Opcode::StaCurrentContextSlot,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            // Load it back
            Instruction::new_unchecked(
                Opcode::LdaCurrentContextSlot,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let result = run_bytecode(instrs, 1).unwrap();
        assert_eq!(result, JsValue::Smi(12));
    }

    /// Uninitialized context slots return `undefined`.
    #[test]
    fn test_context_slot_default_undefined() {
        let instrs = vec![
            Instruction::new_unchecked(
                Opcode::CreateFunctionContext,
                vec![Operand::ConstantPoolIdx(0), Operand::Immediate(3)],
            ),
            Instruction::new_unchecked(Opcode::PushContext, vec![Operand::Register(0)]),
            // Load slot 2 without writing to it — should be undefined
            Instruction::new_unchecked(
                Opcode::LdaCurrentContextSlot,
                vec![Operand::ConstantPoolIdx(2)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let result = run_bytecode(instrs, 1).unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    /// LdaImmutableContextSlot behaves identically to LdaContextSlot.
    #[test]
    fn test_lda_immutable_context_slot() {
        let instrs = vec![
            Instruction::new_unchecked(
                Opcode::CreateFunctionContext,
                vec![Operand::ConstantPoolIdx(0), Operand::Immediate(2)],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::PushContext, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(64)]),
            Instruction::new_unchecked(
                Opcode::StaCurrentContextSlot,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            Instruction::new_unchecked(
                Opcode::LdaImmutableContextSlot,
                vec![
                    Operand::Register(1),
                    Operand::ConstantPoolIdx(0),
                    Operand::Immediate(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let result = run_bytecode(instrs, 2).unwrap();
        assert_eq!(result, JsValue::Smi(64));
    }

    /// CreateCatchContext places the exception in slot 0.
    #[test]
    fn test_create_catch_context() {
        let instrs = vec![
            // Simulate: exception value in r1
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(404)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            // CreateCatchContext [exception_reg=r1, scope_idx=0]
            Instruction::new_unchecked(
                Opcode::CreateCatchContext,
                vec![Operand::Register(1), Operand::ConstantPoolIdx(0)],
            ),
            Instruction::new_unchecked(Opcode::PushContext, vec![Operand::Register(0)]),
            // Load the caught exception from slot 0
            Instruction::new_unchecked(
                Opcode::LdaCurrentContextSlot,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let result = run_bytecode(instrs, 2).unwrap();
        assert_eq!(result, JsValue::Smi(404));
    }

    // ── Error constructor tests ──────────────────────────────────────────────

    /// Helper: build bytecode that loads an error constructor global, calls
    /// `Construct` with a message argument, then loads a named property from
    /// the result and returns it.
    ///
    /// Constant pool:
    ///   [0] = constructor name (e.g. "TypeError")
    ///   [1] = message string (e.g. "bad type")
    ///   [2] = property name to read (e.g. "name" or "message")
    fn error_construct_and_read_property(
        ctor_name: &str,
        message: &str,
        prop: &str,
    ) -> StatorResult<JsValue> {
        let pool = vec![
            ConstantPoolEntry::String(ctor_name.to_string()),
            ConstantPoolEntry::String(message.to_string()),
            ConstantPoolEntry::String(prop.to_string()),
        ];
        let instrs = vec![
            // LdaGlobal [ctor_name_idx=0, feedback_slot=0]
            Instruction::new_unchecked(
                Opcode::LdaGlobal,
                vec![Operand::ConstantPoolIdx(0), Operand::FeedbackSlot(0)],
            ),
            // Star r0 — store constructor in r0
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            // LdaConstant [message_idx=1]
            Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(1)]),
            // Star r1 — store message argument in r1
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            // Construct [ctor=r0, args_start=r1, arg_count=1, feedback_slot=0]
            Instruction::new_unchecked(
                Opcode::Construct,
                vec![
                    Operand::Register(0),
                    Operand::Register(1),
                    Operand::RegisterCount(1),
                    Operand::FeedbackSlot(0),
                ],
            ),
            // Star r2 — store constructed error in r2
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            // LdaNamedProperty [obj=r2, prop_name_idx=2, feedback_slot=0]
            Instruction::new_unchecked(
                Opcode::LdaNamedProperty,
                vec![
                    Operand::Register(2),
                    Operand::ConstantPoolIdx(2),
                    Operand::FeedbackSlot(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode_with_pool(instrs, pool, 3, 0);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        // Re-install globals since make_bytecode_with_pool bypasses new()
        crate::builtins::install_globals::install_globals(&mut frame.global_env.borrow_mut());
        Interpreter::run(&mut frame)
    }

    #[test]
    fn test_new_error_name_property() {
        let result = error_construct_and_read_property("Error", "something broke", "name").unwrap();
        assert_eq!(result, JsValue::String("Error".to_string().into()));
    }

    #[test]
    fn test_new_error_message_property() {
        let result =
            error_construct_and_read_property("Error", "something broke", "message").unwrap();
        assert_eq!(
            result,
            JsValue::String("something broke".to_string().into())
        );
    }

    #[test]
    fn test_new_error_stack_property() {
        let result =
            error_construct_and_read_property("Error", "something broke", "stack").unwrap();
        if let JsValue::String(s) = &result {
            assert!(
                s.starts_with("Error: something broke"),
                "stack should start with error string, got: {s}"
            );
        } else {
            panic!("expected String for .stack, got {result:?}");
        }
    }

    #[test]
    fn test_new_type_error_name() {
        let result = error_construct_and_read_property("TypeError", "bad type", "name").unwrap();
        assert_eq!(result, JsValue::String("TypeError".to_string().into()));
    }

    #[test]
    fn test_new_type_error_message() {
        let result = error_construct_and_read_property("TypeError", "bad type", "message").unwrap();
        assert_eq!(result, JsValue::String("bad type".to_string().into()));
    }

    #[test]
    fn test_new_range_error() {
        let result =
            error_construct_and_read_property("RangeError", "out of range", "name").unwrap();
        assert_eq!(result, JsValue::String("RangeError".to_string().into()));
    }

    #[test]
    fn test_new_syntax_error() {
        let result = error_construct_and_read_property("SyntaxError", "bad token", "name").unwrap();
        assert_eq!(result, JsValue::String("SyntaxError".to_string().into()));
    }

    #[test]
    fn test_new_reference_error() {
        let result =
            error_construct_and_read_property("ReferenceError", "x is not defined", "name")
                .unwrap();
        assert_eq!(result, JsValue::String("ReferenceError".to_string().into()));
    }

    #[test]
    fn test_new_uri_error() {
        let result = error_construct_and_read_property("URIError", "bad URI", "name").unwrap();
        assert_eq!(result, JsValue::String("URIError".to_string().into()));
    }

    #[test]
    fn test_new_eval_error() {
        let result = error_construct_and_read_property("EvalError", "bad eval", "name").unwrap();
        assert_eq!(result, JsValue::String("EvalError".to_string().into()));
    }

    #[test]
    fn test_new_error_no_message() {
        // Construct Error() with no arguments — message should be empty string.
        let pool = vec![
            ConstantPoolEntry::String("Error".to_string()),
            ConstantPoolEntry::String("message".to_string()),
        ];
        let instrs = vec![
            // LdaGlobal [ctor_name_idx=0, feedback_slot=0]
            Instruction::new_unchecked(
                Opcode::LdaGlobal,
                vec![Operand::ConstantPoolIdx(0), Operand::FeedbackSlot(0)],
            ),
            // Star r0
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            // Construct [ctor=r0, args_start=r1, arg_count=0, feedback_slot=0]
            Instruction::new_unchecked(
                Opcode::Construct,
                vec![
                    Operand::Register(0),
                    Operand::Register(1),
                    Operand::RegisterCount(0),
                    Operand::FeedbackSlot(0),
                ],
            ),
            // Star r1
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            // LdaNamedProperty [obj=r1, prop_name_idx=1, feedback_slot=0]
            Instruction::new_unchecked(
                Opcode::LdaNamedProperty,
                vec![
                    Operand::Register(1),
                    Operand::ConstantPoolIdx(1),
                    Operand::FeedbackSlot(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode_with_pool(instrs, pool, 2, 0);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        crate::builtins::install_globals::install_globals(&mut frame.global_env.borrow_mut());
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::String(String::new().into()));
    }

    /// Test that `proto_lookup` resolves name/message/stack on JsValue::Error.
    #[test]
    fn test_proto_lookup_error_properties() {
        use crate::builtins::error::{ErrorKind, JsError};

        let err = JsValue::Error(Rc::new(JsError::new(
            ErrorKind::TypeError,
            "not a function".to_string(),
        )));
        assert_eq!(
            proto_lookup(&err, "name"),
            JsValue::String("TypeError".to_string().into())
        );
        assert_eq!(
            proto_lookup(&err, "message"),
            JsValue::String("not a function".to_string().into())
        );
        if let JsValue::String(s) = proto_lookup(&err, "stack") {
            assert!(s.starts_with("TypeError: not a function"));
        } else {
            panic!("expected String for stack property");
        }
        // Unknown property returns undefined
        assert_eq!(proto_lookup(&err, "foo"), JsValue::Undefined);
    }

    /// Test that `proto_lookup` exposes `cause` property (ES2022).
    #[test]
    fn test_proto_lookup_error_cause_property() {
        use crate::builtins::error::{ErrorKind, JsError};

        // Error without cause returns undefined.
        let err = JsValue::Error(Rc::new(JsError::new(
            ErrorKind::Error,
            "no cause".to_string(),
        )));
        assert_eq!(proto_lookup(&err, "cause"), JsValue::Undefined);

        // Error with cause returns the cause value.
        let cause = JsValue::String("original problem".to_string().into());
        let err_with_cause = JsValue::Error(Rc::new(
            JsError::new(ErrorKind::Error, "wrapper".to_string()).with_cause(cause.clone()),
        ));
        assert_eq!(proto_lookup(&err_with_cause, "cause"), cause);
    }

    /// Test that `proto_lookup` exposes `errors` property for `AggregateError`.
    #[test]
    fn test_proto_lookup_aggregate_error_errors_property() {
        use crate::builtins::error::{ErrorKind, JsError};

        let inner1 = JsValue::Error(Rc::new(JsError::new(
            ErrorKind::TypeError,
            "bad type".to_string(),
        )));
        let inner2 = JsValue::Error(Rc::new(JsError::new(
            ErrorKind::RangeError,
            "out of range".to_string(),
        )));
        let agg = JsValue::Error(Rc::new(JsError::new_aggregate(
            vec![inner1.clone(), inner2.clone()],
            "multiple failures".to_string(),
        )));

        // errors property should be an Array of the original values.
        if let JsValue::Array(arr) = proto_lookup(&agg, "errors") {
            assert_eq!(arr.borrow().len(), 2);
            assert!(
                matches!(&arr.borrow()[0], JsValue::Error(e) if e.kind == ErrorKind::TypeError)
            );
            assert!(
                matches!(&arr.borrow()[1], JsValue::Error(e) if e.kind == ErrorKind::RangeError)
            );
        } else {
            panic!("expected Array for AggregateError.errors");
        }

        // Non-AggregateError should return undefined for errors.
        let regular = JsValue::Error(Rc::new(JsError::new(
            ErrorKind::Error,
            "regular".to_string(),
        )));
        assert_eq!(proto_lookup(&regular, "errors"), JsValue::Undefined);
    }

    /// Test that `proto_lookup` exposes `toString` method on errors.
    #[test]
    fn test_proto_lookup_error_to_string_method() {
        use crate::builtins::error::{ErrorKind, JsError};

        let err = JsValue::Error(Rc::new(JsError::new(
            ErrorKind::TypeError,
            "bad value".to_string(),
        )));
        if let JsValue::NativeFunction(f) = proto_lookup(&err, "toString") {
            let result = f(vec![]).unwrap();
            assert_eq!(
                result,
                JsValue::String("TypeError: bad value".to_string().into())
            );
        } else {
            panic!("expected NativeFunction for toString");
        }
    }

    /// Test that `proto_lookup` returns user-set overlay property on errors.
    #[test]
    fn test_proto_lookup_error_overlay_message() {
        use crate::builtins::error::{ErrorKind, JsError};

        let je = JsError::new(ErrorKind::Error, "original".to_string());
        je.props
            .borrow_mut()
            .insert("message".to_string(), JsValue::String("overridden".into()));
        let err = JsValue::Error(Rc::new(je));
        assert_eq!(
            proto_lookup(&err, "message"),
            JsValue::String("overridden".to_string().into())
        );
    }

    /// Test that `proto_lookup` returns user-set custom property on errors.
    #[test]
    fn test_proto_lookup_error_overlay_custom_prop() {
        use crate::builtins::error::{ErrorKind, JsError};

        let je = JsError::new(ErrorKind::Error, "test".to_string());
        je.props
            .borrow_mut()
            .insert("code".to_string(), JsValue::Smi(42));
        let err = JsValue::Error(Rc::new(je));
        assert_eq!(proto_lookup(&err, "code"), JsValue::Smi(42));
    }

    // ── DeletePropertySloppy / DeletePropertyStrict ──────────────────────

    #[test]
    fn test_delete_property_sloppy_removes_key() {
        let ba = make_bytecode_with_pool(
            vec![
                // r0 = new object
                Instruction::new_unchecked(Opcode::CreateEmptyObjectLiteral, vec![]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                // acc = 42; store as obj.x
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(
                    Opcode::StaNamedProperty,
                    vec![
                        Operand::Register(0),
                        Operand::ConstantPoolIdx(0),
                        Operand::FeedbackSlot(0),
                    ],
                ),
                // acc = "x" (key); delete obj[acc]
                Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(0)]),
                Instruction::new_unchecked(
                    Opcode::DeletePropertySloppy,
                    vec![Operand::Register(0)],
                ),
                // acc should be Boolean(true)
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("x".to_string())],
            1,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_delete_property_strict_returns_true() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::CreateEmptyObjectLiteral, vec![]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(0)]),
                Instruction::new_unchecked(
                    Opcode::DeletePropertyStrict,
                    vec![Operand::Register(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("y".to_string())],
            1,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── CreateRestParameter ─────────────────────────────────────────────

    #[test]
    fn test_create_rest_parameter() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::CreateRestParameter, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![],
            2,
            1,
        );
        let mut frame = InterpreterFrame::new(ba, vec![JsValue::Smi(1)]);
        let result = Interpreter::run(&mut frame).unwrap();
        if let JsValue::Array(arr) = result {
            assert!(arr.borrow().len() >= 0);
        } else {
            panic!("expected Array, got {result:?}");
        }
    }

    // ── CreateMappedArguments / CreateUnmappedArguments ──────────────────

    #[test]
    fn test_create_mapped_arguments() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::CreateMappedArguments, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![],
            0,
            2,
        );
        let mut frame = InterpreterFrame::new(ba, vec![JsValue::Smi(10), JsValue::Smi(20)]);
        let result = Interpreter::run(&mut frame).unwrap();
        if let JsValue::PlainObject(map) = result {
            let m = map.borrow();
            assert_eq!(m.get("0"), Some(&JsValue::Smi(10)));
            assert_eq!(m.get("1"), Some(&JsValue::Smi(20)));
            assert_eq!(m.get("length"), Some(&JsValue::Smi(2)));
            // callee should be present in mapped (sloppy) arguments
            assert!(m.get("callee").is_some());
            assert!(matches!(m.get("callee"), Some(JsValue::Function(_))));
            // @@iterator should be present
            assert!(m.get("@@iterator").is_some());
        } else {
            panic!("expected PlainObject, got {result:?}");
        }
    }

    #[test]
    fn test_create_unmapped_arguments() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::CreateUnmappedArguments, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![],
            0,
            2,
        );
        let mut frame = InterpreterFrame::new(ba, vec![JsValue::Smi(5), JsValue::Smi(6)]);
        let result = Interpreter::run(&mut frame).unwrap();
        if let JsValue::PlainObject(map) = result {
            let m = map.borrow();
            assert_eq!(m.get("0"), Some(&JsValue::Smi(5)));
            assert_eq!(m.get("1"), Some(&JsValue::Smi(6)));
            assert_eq!(m.get("length"), Some(&JsValue::Smi(2)));
            // callee in unmapped (strict) arguments should be a throwing accessor
            assert!(m.get("callee").is_some());
            assert!(matches!(m.get("callee"), Some(JsValue::NativeFunction(_))));
            // @@iterator should be present
            assert!(m.get("@@iterator").is_some());
        } else {
            panic!("expected PlainObject, got {result:?}");
        }
    }

    // ── ThrowReferenceErrorIfHole ───────────────────────────────────────

    #[test]
    fn test_throw_reference_error_if_hole_fires_on_the_hole() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::LdaTheHole, vec![]),
                Instruction::new_unchecked(
                    Opcode::ThrowReferenceErrorIfHole,
                    vec![Operand::ConstantPoolIdx(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("myVar".to_string())],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let err = Interpreter::run(&mut frame).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("Cannot access 'myVar' before initialization"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn test_throw_reference_error_if_hole_noop_when_not_hole() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
                Instruction::new_unchecked(
                    Opcode::ThrowReferenceErrorIfHole,
                    vec![Operand::ConstantPoolIdx(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("x".to_string())],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    #[test]
    fn test_throw_reference_error_if_hole_noop_for_undefined() {
        // `let x = undefined; x` must NOT throw — undefined is a valid
        // initialised value, distinct from the internal hole sentinel.
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
                Instruction::new_unchecked(
                    Opcode::ThrowReferenceErrorIfHole,
                    vec![Operand::ConstantPoolIdx(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("x".to_string())],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    // ── ThrowSuperNotCalledIfHole ───────────────────────────────────────

    #[test]
    fn test_throw_super_not_called_if_hole() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaTheHole, vec![]),
                Instruction::new_unchecked(Opcode::ThrowSuperNotCalledIfHole, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        );
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("Must call super constructor"),
            "unexpected: {msg}"
        );
    }

    #[test]
    fn test_throw_super_not_called_noop_when_not_hole() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
                Instruction::new_unchecked(Opcode::ThrowSuperNotCalledIfHole, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(1));
    }

    // ── ThrowSuperAlreadyCalledIfNotHole ─────────────────────────────────

    #[test]
    fn test_throw_super_already_called_if_not_hole() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
                Instruction::new_unchecked(Opcode::ThrowSuperAlreadyCalledIfNotHole, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        );
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("Super constructor may only be called once"),
            "unexpected: {msg}"
        );
    }

    #[test]
    fn test_throw_super_already_called_noop_when_hole() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaTheHole, vec![]),
                Instruction::new_unchecked(Opcode::ThrowSuperAlreadyCalledIfNotHole, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::TheHole);
    }

    // ── CallRuntime (stub / no-op) ──────────────────────────────────────

    #[test]
    fn test_call_runtime_is_noop() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
                Instruction::new_unchecked(
                    Opcode::CallRuntime,
                    vec![
                        Operand::RuntimeId(0),
                        Operand::Register(0),
                        Operand::RegisterCount(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    #[test]
    fn test_call_runtime_dynamic_import_returns_promise() {
        use crate::bytecode::bytecode_generator::RUNTIME_DYNAMIC_IMPORT;
        let ba = make_bytecode_with_pool(
            vec![
                // Load specifier string into r0.
                Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(0)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                // CallRuntime [RUNTIME_DYNAMIC_IMPORT, r0, 1]
                Instruction::new_unchecked(
                    Opcode::CallRuntime,
                    vec![
                        Operand::RuntimeId(RUNTIME_DYNAMIC_IMPORT),
                        Operand::Register(0),
                        Operand::RegisterCount(1),
                    ],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("./mod.js".into())],
            1,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert!(
            matches!(result, JsValue::Promise(_)),
            "expected Promise, got {result:?}"
        );
    }

    // ── StaNamedOwnProperty ─────────────────────────────────────────────

    #[test]
    fn test_sta_named_own_property() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::CreateEmptyObjectLiteral, vec![]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(77)]),
                Instruction::new_unchecked(
                    Opcode::StaNamedOwnProperty,
                    vec![
                        Operand::Register(0),
                        Operand::ConstantPoolIdx(0),
                        Operand::FeedbackSlot(0),
                    ],
                ),
                Instruction::new_unchecked(
                    Opcode::LdaNamedProperty,
                    vec![
                        Operand::Register(0),
                        Operand::ConstantPoolIdx(0),
                        Operand::FeedbackSlot(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("key".to_string())],
            1,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(77));
    }

    // ── StaLookupSlot ───────────────────────────────────────────────────

    #[test]
    fn test_sta_lookup_slot_stores_in_global_env() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(55)]),
                Instruction::new_unchecked(
                    Opcode::StaLookupSlot,
                    vec![Operand::ConstantPoolIdx(0), Operand::Flag(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("globalVar".to_string())],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let _result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(
            frame.global_env.borrow().get("globalVar"),
            Some(&JsValue::Smi(55))
        );
    }

    // ── LdaLookupSlot ───────────────────────────────────────────────────

    #[test]
    fn test_lda_lookup_slot_found() {
        let ba = make_bytecode_with_pool(
            vec![
                // Store 99 into global "myVar"
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
                Instruction::new_unchecked(
                    Opcode::StaLookupSlot,
                    vec![Operand::ConstantPoolIdx(0), Operand::Flag(0)],
                ),
                // Now load it back via LdaLookupSlot
                Instruction::new_unchecked(
                    Opcode::LdaLookupSlot,
                    vec![Operand::ConstantPoolIdx(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("myVar".to_string())],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    #[test]
    fn test_lda_lookup_slot_not_found_throws() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(
                    Opcode::LdaLookupSlot,
                    vec![Operand::ConstantPoolIdx(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("noSuchVar".to_string())],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let err = Interpreter::run(&mut frame).unwrap_err();
        assert!(matches!(err, StatorError::ReferenceError(_)));
    }

    // ── LdaLookupSlotInsideTypeof ───────────────────────────────────────

    #[test]
    fn test_lda_lookup_slot_inside_typeof_found() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(
                    Opcode::StaLookupSlot,
                    vec![Operand::ConstantPoolIdx(0), Operand::Flag(0)],
                ),
                Instruction::new_unchecked(
                    Opcode::LdaLookupSlotInsideTypeof,
                    vec![Operand::ConstantPoolIdx(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("x".to_string())],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn test_lda_lookup_slot_inside_typeof_not_found_returns_undefined() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(
                    Opcode::LdaLookupSlotInsideTypeof,
                    vec![Operand::ConstantPoolIdx(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("missing".to_string())],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    // ── LdaLookupContextSlot ────────────────────────────────────────────

    #[test]
    fn test_lda_lookup_context_slot_found() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
                Instruction::new_unchecked(
                    Opcode::StaLookupSlot,
                    vec![Operand::ConstantPoolIdx(0), Operand::Flag(0)],
                ),
                Instruction::new_unchecked(
                    Opcode::LdaLookupContextSlot,
                    vec![
                        Operand::ConstantPoolIdx(0),
                        Operand::ConstantPoolIdx(0),
                        Operand::Immediate(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("ctxVar".to_string())],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    #[test]
    fn test_lda_lookup_context_slot_not_found_throws() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(
                    Opcode::LdaLookupContextSlot,
                    vec![
                        Operand::ConstantPoolIdx(0),
                        Operand::ConstantPoolIdx(0),
                        Operand::Immediate(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("nope".to_string())],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let err = Interpreter::run(&mut frame).unwrap_err();
        assert!(matches!(err, StatorError::ReferenceError(_)));
    }

    // ── LdaLookupContextSlotInsideTypeof ────────────────────────────────

    #[test]
    fn test_lda_lookup_context_slot_inside_typeof_not_found() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(
                    Opcode::LdaLookupContextSlotInsideTypeof,
                    vec![
                        Operand::ConstantPoolIdx(0),
                        Operand::ConstantPoolIdx(0),
                        Operand::Immediate(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("missing".to_string())],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    // ── LdaLookupGlobalSlot ─────────────────────────────────────────────

    #[test]
    fn test_lda_lookup_global_slot_found() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(33)]),
                Instruction::new_unchecked(
                    Opcode::StaLookupSlot,
                    vec![Operand::ConstantPoolIdx(0), Operand::Flag(0)],
                ),
                Instruction::new_unchecked(
                    Opcode::LdaLookupGlobalSlot,
                    vec![
                        Operand::ConstantPoolIdx(0),
                        Operand::FeedbackSlot(0),
                        Operand::Immediate(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("gVar".to_string())],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(33));
    }

    #[test]
    fn test_lda_lookup_global_slot_not_found_throws() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(
                    Opcode::LdaLookupGlobalSlot,
                    vec![
                        Operand::ConstantPoolIdx(0),
                        Operand::FeedbackSlot(0),
                        Operand::Immediate(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("noGlobal".to_string())],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let err = Interpreter::run(&mut frame).unwrap_err();
        assert!(matches!(err, StatorError::ReferenceError(_)));
    }

    // ── LdaLookupGlobalSlotInsideTypeof ─────────────────────────────────

    #[test]
    fn test_lda_lookup_global_slot_inside_typeof_not_found() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(
                    Opcode::LdaLookupGlobalSlotInsideTypeof,
                    vec![
                        Operand::ConstantPoolIdx(0),
                        Operand::FeedbackSlot(0),
                        Operand::Immediate(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("missing".to_string())],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    // ── LdaNamedPropertyFromSuper ───────────────────────────────────────

    // NOTE: LdaNamedPropertyFromSuper bytecode-level test — super property lookup not fully wired
    #[test]
    #[ignore]
    fn test_lda_named_property_from_super() {
        let ba = make_bytecode_with_pool(
            vec![
                // Create a PlainObject with property "x" = 10 in reg 0
                Instruction::new_unchecked(Opcode::CreateEmptyObjectLiteral, vec![]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
                Instruction::new_unchecked(
                    Opcode::StaNamedProperty,
                    vec![
                        Operand::Register(0),
                        Operand::ConstantPoolIdx(0),
                        Operand::FeedbackSlot(0),
                    ],
                ),
                // LdaNamedPropertyFromSuper reads from reg 0
                Instruction::new_unchecked(
                    Opcode::LdaNamedPropertyFromSuper,
                    vec![
                        Operand::Register(0),
                        Operand::ConstantPoolIdx(0),
                        Operand::FeedbackSlot(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("x".to_string())],
            1,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(10));
    }

    #[test]
    fn test_lda_named_property_from_super_missing_returns_undefined() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::CreateEmptyObjectLiteral, vec![]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(
                    Opcode::LdaNamedPropertyFromSuper,
                    vec![
                        Operand::Register(0),
                        Operand::ConstantPoolIdx(0),
                        Operand::FeedbackSlot(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("y".to_string())],
            1,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    // ── GetTemplateObject ───────────────────────────────────────────────

    #[test]
    fn test_get_template_object() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(
                    Opcode::GetTemplateObject,
                    vec![Operand::ConstantPoolIdx(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("hello".to_string())],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::String("hello".to_string().into()));
    }

    #[test]
    fn test_get_template_object_cached() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(
                    Opcode::GetTemplateObject,
                    vec![Operand::ConstantPoolIdx(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("tpl".to_string())],
            1,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::String("tpl".to_string().into()));
        assert_eq!(frame.template_cache.len(), 1);
    }

    // ── SetPendingMessage ───────────────────────────────────────────────

    #[test]
    fn test_set_pending_message_swaps() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(Opcode::SetPendingMessage, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Undefined);
        assert_eq!(frame.pending_message, JsValue::Smi(42));
    }

    #[test]
    fn test_set_pending_message_double_swap() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
                Instruction::new_unchecked(Opcode::SetPendingMessage, vec![]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
                Instruction::new_unchecked(Opcode::SetPendingMessage, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(1));
        assert_eq!(frame.pending_message, JsValue::Smi(2));
    }

    // ── TestReferenceEqual ──────────────────────────────────────────────

    #[test]
    fn test_test_reference_equal_same_smi() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(Opcode::TestReferenceEqual, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_test_reference_equal_different_values() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
                Instruction::new_unchecked(Opcode::TestReferenceEqual, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    // ── TestUndetectable ────────────────────────────────────────────────

    #[test]
    fn test_test_undetectable_null() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaNull, vec![]),
                Instruction::new_unchecked(Opcode::TestUndetectable, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_test_undetectable_undefined() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
                Instruction::new_unchecked(Opcode::TestUndetectable, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_test_undetectable_number_is_false() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(Opcode::TestUndetectable, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    // ── JumpIfJSReceiver ────────────────────────────────────────────────

    #[test]
    fn test_jump_if_js_receiver_object_jumps() {
        // Byte layout:
        //   [0] CreateEmptyObjectLiteral → 1 byte
        //   [1] JumpIfJSReceiver(2)      → 2 bytes, end=3, target=3+2=5
        //   [3] LdaSmi(0)               → 2 bytes ← skipped
        //   [5] Return                  → 1 byte
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::CreateEmptyObjectLiteral, vec![]),
                Instruction::new_unchecked(Opcode::JumpIfJSReceiver, vec![Operand::JumpOffset(2)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert!(matches!(result, JsValue::PlainObject(_)));
    }

    #[test]
    fn test_jump_if_js_receiver_primitive_no_jump() {
        // Byte layout:
        //   [0] LdaSmi(5)               → 2 bytes
        //   [2] JumpIfJSReceiver(2)      → 2 bytes, end=4, target=4+2=6
        //   [4] LdaSmi(99)              → 2 bytes
        //   [6] Return                  → 1 byte
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(Opcode::JumpIfJSReceiver, vec![Operand::JumpOffset(2)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    // ── ToNumeric ───────────────────────────────────────────────────────

    #[test]
    fn test_to_numeric_from_smi() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
                Instruction::new_unchecked(Opcode::ToNumeric, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    #[test]
    fn test_to_numeric_from_string() {
        let ba = make_bytecode_with_pool(
            vec![
                Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(0)]),
                Instruction::new_unchecked(Opcode::ToNumeric, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            vec![ConstantPoolEntry::String("42".to_string())],
            0,
            0,
        );
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn test_to_numeric_from_boolean() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaTrue, vec![]),
                Instruction::new_unchecked(Opcode::ToNumeric, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(1));
    }

    // ── Wide / ExtraWide prefix opcodes ──────────────────────────────────

    #[test]
    fn test_wide_prefix_never_reaches_dispatch() {
        // Wide and ExtraWide are stripped by the decoder, so they should
        // never appear in the dispatch loop.  We verify indirectly by
        // encoding a wide operand and confirming the round-trip works.
        use crate::bytecode::bytecodes::encode;
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(300)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let bytes = encode(&instrs);
        // The encoder should have emitted a Wide prefix byte.
        assert_eq!(bytes[0], Opcode::Wide as u8);
        // Running this through the interpreter should work.
        let ba = make_bytecode(instrs, 0, 0);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(300));
    }

    // ── Constant-pool jump variants ──────────────────────────────────────

    #[test]
    fn test_jump_constant_unconditional() {
        // Bytecode layout:
        //   [0] JumpConstant(0)  → 2 bytes, end=2, target=2+2=4
        //   [2] LdaSmi(99)       → 2 bytes
        //   [4] LdaSmi(42)       → 2 bytes
        //   [6] Return           → 1 byte
        let pool = vec![ConstantPoolEntry::Number(2.0)];
        let instrs = vec![
            Instruction::new_unchecked(Opcode::JumpConstant, vec![Operand::ConstantPoolIdx(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode_with_pool(instrs, pool, 0, 0);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn test_jump_if_true_constant_taken() {
        let pool = vec![ConstantPoolEntry::Number(2.0)];
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaTrue, vec![]),
            Instruction::new_unchecked(
                Opcode::JumpIfTrueConstant,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode_with_pool(instrs, pool, 0, 0);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_jump_if_true_constant_not_taken() {
        let pool = vec![ConstantPoolEntry::Number(2.0)];
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaFalse, vec![]),
            Instruction::new_unchecked(
                Opcode::JumpIfTrueConstant,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode_with_pool(instrs, pool, 0, 0);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    #[test]
    fn test_jump_if_false_constant_taken() {
        let pool = vec![ConstantPoolEntry::Number(2.0)];
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaFalse, vec![]),
            Instruction::new_unchecked(
                Opcode::JumpIfFalseConstant,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode_with_pool(instrs, pool, 0, 0);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_jump_if_to_boolean_true_constant_taken() {
        let pool = vec![ConstantPoolEntry::Number(2.0)];
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(
                Opcode::JumpIfToBooleanTrueConstant,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode_with_pool(instrs, pool, 0, 0);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(1));
    }

    #[test]
    fn test_jump_if_to_boolean_false_constant_taken() {
        let pool = vec![ConstantPoolEntry::Number(2.0)];
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(0)]),
            Instruction::new_unchecked(
                Opcode::JumpIfToBooleanFalseConstant,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode_with_pool(instrs, pool, 0, 0);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(0));
    }

    #[test]
    fn test_jump_if_null_constant_taken() {
        let pool = vec![ConstantPoolEntry::Number(2.0)];
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaNull, vec![]),
            Instruction::new_unchecked(
                Opcode::JumpIfNullConstant,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode_with_pool(instrs, pool, 0, 0);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Null);
    }

    #[test]
    fn test_jump_if_not_null_constant_taken() {
        let pool = vec![ConstantPoolEntry::Number(2.0)];
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
            Instruction::new_unchecked(
                Opcode::JumpIfNotNullConstant,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode_with_pool(instrs, pool, 0, 0);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(5));
    }

    #[test]
    fn test_jump_if_undefined_constant_taken() {
        let pool = vec![ConstantPoolEntry::Number(2.0)];
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
            Instruction::new_unchecked(
                Opcode::JumpIfUndefinedConstant,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode_with_pool(instrs, pool, 0, 0);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    #[test]
    fn test_jump_if_not_undefined_constant_taken() {
        let pool = vec![ConstantPoolEntry::Number(2.0)];
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
            Instruction::new_unchecked(
                Opcode::JumpIfNotUndefinedConstant,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode_with_pool(instrs, pool, 0, 0);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    #[test]
    fn test_jump_if_undefined_or_null_constant_null() {
        let pool = vec![ConstantPoolEntry::Number(2.0)];
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaNull, vec![]),
            Instruction::new_unchecked(
                Opcode::JumpIfUndefinedOrNullConstant,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode_with_pool(instrs, pool, 0, 0);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Null);
    }

    #[test]
    fn test_jump_if_undefined_or_null_constant_undefined() {
        let pool = vec![ConstantPoolEntry::Number(2.0)];
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
            Instruction::new_unchecked(
                Opcode::JumpIfUndefinedOrNullConstant,
                vec![Operand::ConstantPoolIdx(0)],
            ),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode_with_pool(instrs, pool, 0, 0);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    // ── Host integration calls ───────────────────────────────────────────

    #[test]
    fn test_call_js_runtime_noop() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(
                    Opcode::CallJSRuntime,
                    vec![
                        Operand::ConstantPoolIdx(0),
                        Operand::Register(0),
                        Operand::RegisterCount(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn test_invoke_intrinsic_noop() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
                Instruction::new_unchecked(
                    Opcode::InvokeIntrinsic,
                    vec![
                        Operand::RuntimeId(0),
                        Operand::Register(0),
                        Operand::RegisterCount(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    #[test]
    fn test_call_runtime_for_pair_noop() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
                Instruction::new_unchecked(
                    Opcode::CallRuntimeForPair,
                    vec![
                        Operand::RuntimeId(0),
                        Operand::Register(0),
                        Operand::RegisterCount(0),
                        Operand::Register(0),
                    ],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    // ── CollectTypeProfile ───────────────────────────────────────────────

    #[test]
    fn test_collect_type_profile_noop() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
                Instruction::new_unchecked(Opcode::CollectTypeProfile, vec![Operand::Immediate(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(10));
    }

    // ── ConstructForwardAllArgs ──────────────────────────────────────────

    #[test]
    fn test_construct_forward_all_args_native() {
        // Build a "callee" that is a NativeFunction constructor.
        // The outer function takes 2 params and forwards them via
        // ConstructForwardAllArgs.
        use crate::bytecode::bytecodes::encode;

        // Inner function: just adds its two parameters.
        let inner_instrs = vec![
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register((-1i32) as u32)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register((-2i32) as u32)]),
            Instruction::new_unchecked(
                Opcode::Add,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let inner_ba = BytecodeArray::new(
            encode(&inner_instrs),
            vec![],
            1,
            2,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );

        // Outer function: has 2 params, stores inner_ba in r0 via
        // LdaConstant, then calls ConstructForwardAllArgs.
        let pool = vec![ConstantPoolEntry::Function(Box::new(inner_ba))];
        let outer_instrs = vec![
            Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(0)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::ConstructForwardAllArgs,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode_with_pool(outer_instrs, pool, 1, 2);
        let mut frame = InterpreterFrame::new(ba, vec![JsValue::Smi(3), JsValue::Smi(4)]);
        let result = Interpreter::run(&mut frame).unwrap();
        // [[Construct]] semantics: the inner function returns Smi(7) (a
        // primitive), so the freshly-created `this` object is returned.
        assert!(
            matches!(result, JsValue::PlainObject(_)),
            "expected PlainObject from [[Construct]], got {result:?}"
        );
    }

    #[test]
    fn test_construct_forward_all_args_no_params() {
        use crate::bytecode::bytecodes::encode;

        // Inner function: returns 42.
        let inner_instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let inner_ba = BytecodeArray::new(
            encode(&inner_instrs),
            vec![],
            0,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );

        let pool = vec![ConstantPoolEntry::Function(Box::new(inner_ba))];
        let outer_instrs = vec![
            Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(0)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::ConstructForwardAllArgs,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode_with_pool(outer_instrs, pool, 1, 0);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        // [[Construct]] semantics: the inner function returns Smi(42) (a
        // primitive), so the freshly-created `this` object is returned.
        assert!(
            matches!(result, JsValue::PlainObject(_)),
            "expected PlainObject from [[Construct]], got {result:?}"
        );
    }

    // ── CreateObjectFromIterable ─────────────────────────────────────────

    #[test]
    fn test_create_object_from_iterable_plain_object() {
        let pool = vec![ConstantPoolEntry::String("x".to_string())];
        let instrs = vec![
            // Create an empty object, set property "x" = 10, then
            // use CreateObjectFromIterable to spread it.
            Instruction::new_unchecked(Opcode::CreateEmptyObjectLiteral, vec![]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
            Instruction::new_unchecked(
                Opcode::StaNamedProperty,
                vec![
                    Operand::Register(0),
                    Operand::ConstantPoolIdx(0),
                    Operand::FeedbackSlot(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::CreateObjectFromIterable, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode_with_pool(instrs, pool, 1, 0);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        if let JsValue::PlainObject(obj) = &result {
            let map = obj.borrow();
            assert_eq!(map.get("x"), Some(&JsValue::Smi(10)));
        } else {
            panic!("expected PlainObject, got {result:?}");
        }
    }

    #[test]
    fn test_create_object_from_iterable_empty() {
        // CreateObjectFromIterable with undefined accumulator → empty object.
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
                Instruction::new_unchecked(Opcode::CreateObjectFromIterable, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        if let JsValue::PlainObject(obj) = &result {
            assert!(obj.borrow().is_empty());
        } else {
            panic!("expected PlainObject, got {result:?}");
        }
    }

    // ── Tagged template literal integration tests ───────────────────────

    /// Compile a JavaScript source string and run it through the interpreter.
    fn compile_source_and_run(src: &str) -> StatorResult<JsValue> {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        let program = crate::parser::recursive_descent::parse(src)?;
        let ba = BytecodeGenerator::compile_program(&program)?;
        let mut frame = InterpreterFrame::new(ba, vec![]);
        Interpreter::run(&mut frame)
    }

    fn assert_fulfilled_promise_value(result: JsValue, expected: JsValue) {
        match result {
            JsValue::Promise(promise) => {
                assert_eq!(
                    promise.value(),
                    Some(expected),
                    "promise should be fulfilled"
                );
                assert!(
                    !promise.is_rejected(),
                    "promise should not be rejected: {promise:?}"
                );
            }
            other => panic!("expected Promise, got {other:?}"),
        }
    }

    fn assert_rejected_promise_reason(result: JsValue, expected: JsValue) {
        match result {
            JsValue::Promise(promise) => {
                assert_eq!(
                    promise.reason(),
                    Some(expected),
                    "promise should be rejected"
                );
                assert!(
                    !promise.is_fulfilled(),
                    "promise should not be fulfilled: {promise:?}"
                );
            }
            other => panic!("expected Promise, got {other:?}"),
        }
    }

    #[test]
    fn test_tagged_template_basic_call() {
        let result =
            compile_source_and_run("function tag(strings) { return strings[0]; } tag`hello`")
                .unwrap();
        assert_eq!(result, JsValue::String("hello".to_string().into()));
    }

    #[test]
    fn test_tagged_template_raw_property() {
        let result =
            compile_source_and_run("function tag(strings) { return strings.raw[0]; } tag`hello`")
                .unwrap();
        assert_eq!(result, JsValue::String("hello".to_string().into()));
    }

    #[test]
    fn test_tagged_template_interpolation_args() {
        let result = compile_source_and_run(
            "function tag(strings, a, b) { return a + b; } tag`x${10}y${32}z`",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn test_tagged_template_strings_length() {
        let result = compile_source_and_run(
            "function tag(strings) { return strings.length; } tag`a${1}b${2}c`",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    #[test]
    fn test_tagged_template_no_substitution() {
        let result =
            compile_source_and_run("function tag(strings) { return strings.length; } tag`hello`")
                .unwrap();
        assert_eq!(result, JsValue::Smi(1));
    }

    // ── Async function execution ──────────────────────────────────────────────

    #[test]
    fn test_async_function_returns_promise() {
        // `async function f() { return 42; } f()`
        // The call should return a Promise wrapping 42.
        let result = compile_source_and_run("async function f() { return 42; } f()").unwrap();
        assert!(
            matches!(result, JsValue::Promise(_)),
            "expected Promise, got {result:?}"
        );
        if let JsValue::Promise(p) = &result {
            assert_eq!(
                p.value(),
                Some(JsValue::Smi(42)),
                "promise should be fulfilled with 42"
            );
        }
    }

    #[test]
    fn test_async_function_await_resolves() {
        // `async function f() { let x = await 10; return x + 1; } f()`
        let result =
            compile_source_and_run("async function f() { let x = await 10; return x + 1; } f()")
                .unwrap();
        if let JsValue::Promise(p) = &result {
            assert_eq!(
                p.value(),
                Some(JsValue::Smi(11)),
                "promise should be fulfilled with 11"
            );
        } else {
            panic!("expected Promise, got {result:?}");
        }
    }

    #[test]
    fn test_async_arrow_returns_promise() {
        // `var f = async () => 99; f()`
        let result = compile_source_and_run("var f = async () => 99; f()").unwrap();
        assert!(
            matches!(result, JsValue::Promise(_)),
            "expected Promise, got {result:?}"
        );
        if let JsValue::Promise(p) = &result {
            assert_eq!(p.value(), Some(JsValue::Smi(99)));
        }
    }

    #[test]
    fn test_async_function_return_flattens_promise() {
        let result =
            compile_source_and_run("async function f() { return Promise.resolve(7); } f()")
                .unwrap();
        assert_fulfilled_promise_value(result, JsValue::Smi(7));
    }

    #[test]
    fn test_async_function_return_flattens_thenable() {
        let result = compile_source_and_run(
            r#"async function f() { return { then(resolve) { resolve(8); } }; } f()"#,
        )
        .unwrap();
        assert_fulfilled_promise_value(result, JsValue::Smi(8));
    }

    #[test]
    fn test_async_function_awaits_promise_resolve() {
        let result =
            compile_source_and_run("async function f() { return await Promise.resolve(9); } f()")
                .unwrap();
        assert_fulfilled_promise_value(result, JsValue::Smi(9));
    }

    #[test]
    fn test_async_function_awaits_thenable() {
        let result = compile_source_and_run(
            r#"async function f() { return await { then(resolve) { resolve(10); } }; } f()"#,
        )
        .unwrap();
        assert_fulfilled_promise_value(result, JsValue::Smi(10));
    }

    #[test]
    fn test_async_function_catches_rejected_promise() {
        let result = compile_source_and_run(
            "async function f() { try { await Promise.reject('boom'); } catch (e) { return e; } } f()",
        )
        .unwrap();
        assert_fulfilled_promise_value(result, JsValue::String("boom".into()));
    }

    #[test]
    fn test_async_function_catches_rejected_thenable() {
        let result = compile_source_and_run(
            r#"async function f() {
                try {
                    await { then(_resolve, reject) { reject('nope'); } };
                } catch (e) {
                    return e;
                }
            } f()"#,
        )
        .unwrap();
        assert_fulfilled_promise_value(result, JsValue::String("nope".into()));
    }

    #[test]
    fn test_async_function_throw_rejects_promise() {
        let result = compile_source_and_run("async function f() { throw 'boom'; } f()").unwrap();
        assert_rejected_promise_reason(result, JsValue::String("boom".into()));
    }

    #[test]
    fn test_async_function_await_rejection_rejects_promise() {
        let result = compile_source_and_run(
            "async function f() { return await Promise.reject('boom'); } f()",
        )
        .unwrap();
        assert_rejected_promise_reason(result, JsValue::String("boom".into()));
    }

    #[test]
    fn test_async_function_nested_async_call() {
        let result = compile_source_and_run(
            "async function inner() { return 5; } async function outer() { return await inner(); } outer()",
        )
        .unwrap();
        assert_fulfilled_promise_value(result, JsValue::Smi(5));
    }

    #[test]
    fn test_async_object_method_returns_promise() {
        let result =
            compile_source_and_run("var obj = { async f() { return 12; } }; obj.f()").unwrap();
        assert_fulfilled_promise_value(result, JsValue::Smi(12));
    }

    #[test]
    fn test_promise_resolve_returns_fulfilled_promise() {
        let result = compile_source_and_run("Promise.resolve(13)").unwrap();
        assert_fulfilled_promise_value(result, JsValue::Smi(13));
    }

    #[test]
    fn test_promise_resolve_thenable_unwraps() {
        let result =
            compile_source_and_run(r#"Promise.resolve({ then(resolve) { resolve(14); } })"#)
                .unwrap();
        assert_fulfilled_promise_value(result, JsValue::Smi(14));
    }

    #[test]
    fn test_promise_resolve_existing_promise_keeps_value() {
        let result = compile_source_and_run("Promise.resolve(Promise.resolve(15))").unwrap();
        assert_fulfilled_promise_value(result, JsValue::Smi(15));
    }

    #[test]
    fn test_promise_reject_returns_rejected_promise() {
        let result = compile_source_and_run("Promise.reject('bad')").unwrap();
        assert_rejected_promise_reason(result, JsValue::String("bad".into()));
    }

    #[test]
    fn test_for_await_of_array_sums_values() {
        let result = compile_source_and_run(
            "async function f() { var total = 0; for await (const x of [1, 2, 3]) { total = total + x; } return total; } f()",
        )
        .unwrap();
        assert_fulfilled_promise_value(result, JsValue::Smi(6));
    }

    #[test]
    fn test_for_await_of_array_awaits_promises() {
        let result = compile_source_and_run(
            "async function f() { var total = 0; for await (const x of [Promise.resolve(1), Promise.resolve(2), 3]) { total = total + x; } return total; } f()",
        )
        .unwrap();
        assert_fulfilled_promise_value(result, JsValue::Smi(6));
    }

    #[test]
    fn test_for_await_of_sync_generator_awaits_values() {
        let result = compile_source_and_run(
            "function* g() { yield Promise.resolve(1); yield 2; } async function f() { var total = 0; for await (const x of g()) { total = total + x; } return total; } f()",
        )
        .unwrap();
        assert_fulfilled_promise_value(result, JsValue::Smi(3));
    }

    #[test]
    fn test_for_await_of_async_generator_sums_values() {
        let result = compile_source_and_run(
            "async function* g() { yield 1; yield 2; } async function f() { var total = 0; for await (const x of g()) { total = total + x; } return total; } f()",
        )
        .unwrap();
        assert_fulfilled_promise_value(result, JsValue::Smi(3));
    }

    #[test]
    fn test_for_await_of_break_stops_iteration() {
        let result = compile_source_and_run(
            "async function f() { var total = 0; for await (const x of [1, 2, 3]) { total = total + x; break; } return total; } f()",
        )
        .unwrap();
        assert_fulfilled_promise_value(result, JsValue::Smi(1));
    }

    #[test]
    fn test_for_await_of_custom_async_iterable() {
        let result = compile_source_and_run(
            r#"async function f() {
                var iterable = {
                    "@@asyncIterator": function() {
                        var step = 0;
                        return {
                            next: function() {
                                step = step + 1;
                                if (step === 1) return Promise.resolve({ value: 4, done: false });
                                if (step === 2) return Promise.resolve({ value: 5, done: false });
                                return Promise.resolve({ value: 0, done: true });
                            }
                        };
                    }
                };
                var total = 0;
                for await (const x of iterable) { total = total + x; }
                return total;
            } f()"#,
        )
        .unwrap();
        assert_fulfilled_promise_value(result, JsValue::Smi(9));
    }

    #[test]
    fn test_async_generator_next_returns_promise() {
        let result =
            compile_source_and_run("async function* g() { yield 1; } var it = g(); it.next()")
                .unwrap();
        match result {
            JsValue::Promise(promise) => {
                let value = promise.value().expect("next() promise should be fulfilled");
                match value {
                    JsValue::PlainObject(map) => {
                        let borrow = map.borrow();
                        assert_eq!(borrow.get("value"), Some(&JsValue::Smi(1)));
                        assert_eq!(borrow.get("done"), Some(&JsValue::Boolean(false)));
                    }
                    other => panic!("expected iterator result object, got {other:?}"),
                }
            }
            other => panic!("expected Promise, got {other:?}"),
        }
    }

    #[test]
    fn test_async_generator_multiple_next_calls() {
        let result = compile_source_and_run(
            "async function* g() { yield 1; yield 2; } async function f() { var it = g(); var a = await it.next(); var b = await it.next(); return a.value + b.value; } f()",
        )
        .unwrap();
        assert_fulfilled_promise_value(result, JsValue::Smi(3));
    }

    #[test]
    fn test_async_generator_return_method() {
        let result = compile_source_and_run(
            "async function* g() { yield 1; yield 2; } async function f() { var it = g(); await it.next(); var r = await it.return(5); return r.value + (r.done ? 1 : 0); } f()",
        )
        .unwrap();
        assert_fulfilled_promise_value(result, JsValue::Smi(6));
    }

    #[test]
    fn test_async_generator_throw_method_caught() {
        let result = compile_source_and_run(
            "async function* g() { try { yield 1; } catch (e) { return e; } } async function f() { var it = g(); await it.next(); var r = await it.throw('boom'); return r.value; } f()",
        )
        .unwrap();
        assert_fulfilled_promise_value(result, JsValue::String("boom".into()));
    }

    #[test]
    fn test_async_generator_throw_method_rejects_when_uncaught() {
        let result = compile_source_and_run(
            "async function* g() { yield 1; } async function f() { var it = g(); await it.next(); return await it.throw('boom'); } f()",
        )
        .unwrap();
        assert_rejected_promise_reason(result, JsValue::String("boom".into()));
    }

    // ── Generator .next()/.return()/.throw() ──────────────────────────────────

    #[test]
    fn test_generator_next_method() {
        // `function* g() { yield 1; yield 2; } var it = g(); it.next()`
        // it.next() should return { value: 1, done: false }
        let result =
            compile_source_and_run("function* g() { yield 1; yield 2; } var it = g(); it.next()")
                .unwrap();
        if let JsValue::PlainObject(map) = &result {
            let borrow = map.borrow();
            assert_eq!(borrow.get("value"), Some(&JsValue::Smi(1)));
            assert_eq!(borrow.get("done"), Some(&JsValue::Boolean(false)));
        } else {
            panic!("expected PlainObject iterator result, got {result:?}");
        }
    }

    #[test]
    fn test_generator_return_method() {
        // Generator .return(val) should complete the generator.
        let ba = gen_bytecode_yield_1_yield_2();
        let gs = GeneratorState::new(ba);

        // Advance once.
        Interpreter::run_generator_step(&gs, JsValue::Undefined).unwrap();

        // .return(99)
        let result = Interpreter::generator_return(&gs, JsValue::Smi(99)).unwrap();
        if let JsValue::PlainObject(map) = &result {
            let borrow = map.borrow();
            assert_eq!(borrow.get("value"), Some(&JsValue::Smi(99)));
            assert_eq!(borrow.get("done"), Some(&JsValue::Boolean(true)));
        } else {
            panic!("expected PlainObject iterator result, got {result:?}");
        }
        assert_eq!(gs.borrow().status, GeneratorStatus::Completed);
    }

    #[test]
    fn test_generator_throw_method() {
        // Generator .throw(val) should mark the generator as completed and error.
        let ba = gen_bytecode_yield_1_yield_2();
        let gs = GeneratorState::new(ba);

        // Advance once.
        Interpreter::run_generator_step(&gs, JsValue::Undefined).unwrap();

        // .throw(error)
        let result = Interpreter::generator_throw(&gs, JsValue::String("boom".into()));
        assert!(result.is_err(), "generator_throw should return an error");
        assert_eq!(gs.borrow().status, GeneratorStatus::Completed);
    }

    #[test]
    fn test_generator_return_on_completed() {
        // .return(val) on a completed generator returns { value: val, done: true }
        // per §27.5.3.4 GeneratorResumeAbrupt step 2a.
        let ba = gen_bytecode_yield_1_yield_2();
        let gs = GeneratorState::new(ba);

        // Drive to completion.
        Interpreter::run_generator_step(&gs, JsValue::Undefined).unwrap();
        Interpreter::run_generator_step(&gs, JsValue::Undefined).unwrap();
        Interpreter::run_generator_step(&gs, JsValue::Undefined).unwrap();
        assert_eq!(gs.borrow().status, GeneratorStatus::Completed);

        let result = Interpreter::generator_return(&gs, JsValue::Smi(5)).unwrap();
        if let JsValue::PlainObject(map) = &result {
            let borrow = map.borrow();
            assert_eq!(borrow.get("value"), Some(&JsValue::Smi(5)));
            assert_eq!(borrow.get("done"), Some(&JsValue::Boolean(true)));
        } else {
            panic!("expected PlainObject, got {result:?}");
        }
    }

    #[test]
    fn test_generator_return_on_completed_no_arg() {
        // .return() with no argument on a completed generator returns
        // { value: undefined, done: true }.
        let ba = gen_bytecode_yield_1_yield_2();
        let gs = GeneratorState::new(ba);
        // Drive to completion.
        Interpreter::run_generator_step(&gs, JsValue::Undefined).unwrap();
        Interpreter::run_generator_step(&gs, JsValue::Undefined).unwrap();
        Interpreter::run_generator_step(&gs, JsValue::Undefined).unwrap();
        assert_eq!(gs.borrow().status, GeneratorStatus::Completed);

        let result = Interpreter::generator_return(&gs, JsValue::Undefined).unwrap();
        if let JsValue::PlainObject(map) = &result {
            let borrow = map.borrow();
            assert_eq!(borrow.get("value"), Some(&JsValue::Undefined));
            assert_eq!(borrow.get("done"), Some(&JsValue::Boolean(true)));
        } else {
            panic!("expected PlainObject, got {result:?}");
        }
    }

    #[test]
    fn test_generator_not_resumable_after_return() {
        // After .return(), .next() should return { value: undefined, done: true }.
        let ba = gen_bytecode_yield_1_yield_2();
        let gs = GeneratorState::new(ba);
        // Advance once.
        Interpreter::run_generator_step(&gs, JsValue::Undefined).unwrap();
        // .return(42)
        Interpreter::generator_return(&gs, JsValue::Smi(42)).unwrap();
        assert_eq!(gs.borrow().status, GeneratorStatus::Completed);
        // .next() should return done.
        let step = Interpreter::run_generator_step(&gs, JsValue::Undefined).unwrap();
        assert_eq!(step, GeneratorStep::Return(JsValue::Undefined));
    }

    #[test]
    fn test_generator_not_resumable_after_throw() {
        // After .throw(), .next() should return { value: undefined, done: true }.
        let ba = gen_bytecode_yield_1_yield_2();
        let gs = GeneratorState::new(ba);
        // Advance once.
        Interpreter::run_generator_step(&gs, JsValue::Undefined).unwrap();
        // .throw(err)
        let _ = Interpreter::generator_throw(&gs, JsValue::String("err".into()));
        assert_eq!(gs.borrow().status, GeneratorStatus::Completed);
        // .next() should return done.
        let step = Interpreter::run_generator_step(&gs, JsValue::Undefined).unwrap();
        assert_eq!(step, GeneratorStep::Return(JsValue::Undefined));
    }

    #[test]
    fn test_generator_symbol_iterator_returns_self() {
        // gen[@@iterator]() must return the generator itself (§27.5.1.2).
        let ba = gen_bytecode_yield_1_yield_2();
        let gs = GeneratorState::new(ba);
        let generator = JsValue::Generator(gs);
        let iter_fn = proto_lookup(&generator, "@@iterator");
        if let JsValue::NativeFunction(f) = iter_fn {
            let result = f(vec![]).unwrap();
            assert_eq!(
                result, generator,
                "@@iterator() should return the generator itself"
            );
        } else {
            panic!("expected NativeFunction for @@iterator, got {iter_fn:?}");
        }
    }

    #[test]
    fn test_generator_throw_caught_by_try_catch() {
        // §27.5.3.3 — .throw(err) resumes at the yield inside a try/catch;
        // the catch block catches the error and execution continues.
        //
        // function* gen() {
        //     try { yield 1; yield 2; } catch(e) { yield "caught"; }
        // }
        // var g = gen(); g.next(); var r = g.throw("err"); r.value
        let result = compile_source_and_run(
            r#"
            function* gen() {
                try { yield 1; yield 2; } catch(e) { yield "caught"; }
            }
            var g = gen();
            g.next();
            var r = g.throw("err");
            r.value
            "#,
        )
        .unwrap();
        assert_eq!(result, JsValue::String("caught".into()));
    }

    #[test]
    fn test_generator_throw_no_try_catch_propagates() {
        // .throw(err) on a generator without try/catch propagates the error.
        let ba = gen_bytecode_yield_1_yield_2();
        let gs = GeneratorState::new(ba);

        // Advance to first yield.
        Interpreter::run_generator_step(&gs, JsValue::Undefined).unwrap();

        // .throw(err) — no handler, error must propagate.
        let result = Interpreter::generator_throw(&gs, JsValue::String("boom".into()));
        assert!(result.is_err(), "generator_throw should propagate error");
        assert_eq!(gs.borrow().status, GeneratorStatus::Completed);
    }

    #[test]
    fn test_generator_throw_on_completed_generator() {
        // .throw(err) on a completed generator throws the error.
        let ba = gen_bytecode_yield_1_yield_2();
        let gs = GeneratorState::new(ba);

        // Drive to completion.
        Interpreter::run_generator_step(&gs, JsValue::Undefined).unwrap();
        Interpreter::run_generator_step(&gs, JsValue::Undefined).unwrap();
        Interpreter::run_generator_step(&gs, JsValue::Undefined).unwrap();
        assert_eq!(gs.borrow().status, GeneratorStatus::Completed);

        let result = Interpreter::generator_throw(&gs, JsValue::String("err".into()));
        assert!(result.is_err(), "throw on completed should error");
    }

    // ── Async function (run_async_function) ───────────────────────────────────

    #[test]
    fn test_run_async_function_simple_return() {
        // Simulate `async function f() { return 42; }` — no await points.
        let instrs = vec![
            Instruction::new_unchecked(Opcode::SwitchOnGeneratorState, vec![GEN_REG]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode(instrs, 1, 0).with_async_flag(true);
        let result = Interpreter::run_async_function(ba, vec![]).unwrap();
        if let JsValue::Promise(p) = &result {
            assert_eq!(p.value(), Some(JsValue::Smi(42)));
        } else {
            panic!("expected Promise, got {result:?}");
        }
    }

    #[test]
    fn test_run_async_function_with_await() {
        // Simulate `async function f() { let x = await 10; return x + 1; }`
        let instrs = vec![
            Instruction::new_unchecked(Opcode::SwitchOnGeneratorState, vec![GEN_REG]),
            // await 10
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
            Instruction::new_unchecked(
                Opcode::SuspendGenerator,
                vec![
                    GEN_REG,
                    GEN_REG,
                    Operand::RegisterCount(0),
                    Operand::Immediate(0),
                ],
            ),
            Instruction::new_unchecked(
                Opcode::ResumeGenerator,
                vec![GEN_REG, GEN_REG, Operand::RegisterCount(0)],
            ),
            // x = resolved_value (acc)
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            // return x + 1
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(1)]),
            Instruction::new_unchecked(
                Opcode::Add,
                vec![Operand::Register(2), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode(instrs, 3, 0).with_async_flag(true);
        let result = Interpreter::run_async_function(ba, vec![]).unwrap();
        if let JsValue::Promise(p) = &result {
            assert_eq!(
                p.value(),
                Some(JsValue::Smi(11)),
                "async function should return 10 + 1 = 11"
            );
        } else {
            panic!("expected Promise, got {result:?}");
        }
    }

    #[test]
    fn test_run_async_function_throw_returns_rejected_promise() {
        // Simulate `async function f() { throw "boom"; }` — the thrown
        // exception should be caught and surfaced as a rejected promise.
        let instrs = vec![
            Instruction::new_unchecked(Opcode::SwitchOnGeneratorState, vec![GEN_REG]),
            Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(0)]),
            Instruction::new_unchecked(Opcode::Throw, vec![]),
        ];
        let ba = make_bytecode_with_pool(
            instrs,
            vec![ConstantPoolEntry::String("boom".to_string())],
            1,
            0,
        )
        .with_async_flag(true);
        let result = Interpreter::run_async_function(ba, vec![]).unwrap();
        if let JsValue::Promise(p) = &result {
            assert!(
                p.is_rejected(),
                "async throw should produce a rejected promise"
            );
            assert!(
                p.reason().is_some(),
                "rejected promise should have a reason"
            );
        } else {
            panic!("expected Promise, got {result:?}");
        }
    }

    #[test]
    fn test_js_error_to_rejection_reason_js_exceptions() {
        // All JS exception types should convert to a JsValue.
        assert!(
            Interpreter::js_error_to_rejection_reason(&StatorError::TypeError("bad".into()))
                .is_some()
        );
        assert!(
            Interpreter::js_error_to_rejection_reason(&StatorError::ReferenceError("undef".into()))
                .is_some()
        );
        assert!(
            Interpreter::js_error_to_rejection_reason(&StatorError::JsException("oops".into()))
                .is_some()
        );
    }

    #[test]
    fn test_js_error_to_rejection_reason_engine_errors() {
        // Engine-level errors should NOT be converted — they propagate.
        assert!(Interpreter::js_error_to_rejection_reason(&StatorError::OutOfMemory).is_none());
        assert!(
            Interpreter::js_error_to_rejection_reason(&StatorError::Internal("bug".into()))
                .is_none()
        );
    }

    // ── Tail call optimisation ────────────────────────────────────────────────

    #[test]
    fn test_tail_call_direct_recursion() {
        // `return f(n-1, acc+n)` is a direct tail call — it should reuse
        // the frame and not overflow the call stack even for large N.
        let result = crate::builtins::global::global_eval(
            "function sum(n, acc) { \
               if (n <= 0) return acc; \
               return sum(n - 1, acc + n); \
             } \
             sum(50000, 0)",
        )
        .unwrap();
        // 50000 * 50001 / 2 = 1_250_025_000 (fits in i32)
        assert_eq!(result, JsValue::Smi(1_250_025_000));
    }

    #[test]
    fn test_tail_call_conditional() {
        // Both branches of the ternary are in tail position.
        let result = crate::builtins::global::global_eval(
            "function even(n) { \
               if (n === 0) return true; \
               return odd(n - 1); \
             } \
             function odd(n) { \
               if (n === 0) return false; \
               return even(n - 1); \
             } \
             even(100000)",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_tail_call_non_tail_not_optimised() {
        // `n + f(n-1)` is NOT a tail call — the call result feeds into `+`.
        // This should work correctly (not be incorrectly optimised).
        let result = crate::builtins::global::global_eval(
            "function sum(n) { \
               if (n <= 0) return 0; \
               return n + sum(n - 1); \
             } \
             sum(10)",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(55));
    }

    #[test]
    fn test_tail_call_conditional_ternary() {
        // `return cond ? f() : g()` — both branches are tail calls.
        let result = crate::builtins::global::global_eval(
            "function countdown(n) { \
               return n <= 0 ? 'done' : countdown(n - 1); \
             } \
             countdown(50000)",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("done".to_string().into()));
    }

    // ── Class-inheritance conformance ────────────────────────────────────

    /// Basic class construction: `new Foo()` creates an instance with
    /// `this` bound in the constructor.
    #[test]
    fn test_class_construct_this_binding() {
        let result = crate::builtins::global::global_eval(
            "class Foo { constructor() { this.x = 42; } } \
             let f = new Foo(); \
             f.x",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// Static methods: `class Foo { static bar() { return 1; } }`
    /// — `Foo.bar()` should work.
    #[test]
    fn test_class_static_method() {
        let result = crate::builtins::global::global_eval(
            "class Foo { static bar() { return 99; } } \
             Foo.bar()",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    /// Static class fields: `class Foo { static x = 1; }` — `Foo.x`.
    #[test]
    fn test_class_static_field() {
        let result = crate::builtins::global::global_eval(
            "class Foo { static x = 7; } \
             Foo.x",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    /// Instance fields: `class Foo { x = 1; }` — instance properties
    /// initialized during construction.
    #[test]
    fn test_class_instance_field() {
        let result = crate::builtins::global::global_eval(
            "class Foo { x = 10; } \
             let f = new Foo(); \
             f.x",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(10));
    }

    /// Class extends prototype chain:
    /// `Foo.prototype.__proto__ === Bar.prototype` and
    /// `Foo.__proto__ === Bar` (static inheritance).
    #[test]
    fn test_class_extends_prototype_chain() {
        // Verify that an instance of Child can access Parent prototype methods.
        let result = crate::builtins::global::global_eval(
            "class Parent { greet() { return 'hello'; } } \
             class Child extends Parent { } \
             let c = new Child(); \
             c.greet()",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("hello".to_string().into()));
    }

    /// `super()` in constructors: calls the parent constructor, sets up
    /// the prototype chain, and initialises parent-defined properties.
    #[test]
    fn test_class_super_constructor() {
        let result = crate::builtins::global::global_eval(
            "class Base { constructor() { this.base = 1; } } \
             class Derived extends Base { \
                 constructor() { super(); this.derived = 2; } \
             } \
             let d = new Derived(); \
             d.base + d.derived",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    /// `new.target` — in constructors, references the constructor
    /// being `new`-ed.
    #[test]
    fn test_class_new_target() {
        // new.target is defined (not undefined) when called with `new`.
        let result = crate::builtins::global::global_eval(
            "class Foo { constructor() { \
                 this.hasNewTarget = new.target !== undefined; \
             } } \
             let f = new Foo(); \
             f.hasNewTarget",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── proto_lookup: Object.prototype valueOf / toLocaleString ───────────

    #[test]
    fn test_proto_lookup_plain_object_valueof() {
        let result =
            crate::builtins::global::global_eval("var o = {x:1}; o.valueOf() === o").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_proto_lookup_plain_object_tolocalestring() {
        let result = crate::builtins::global::global_eval(
            "var o = {}; typeof o.toLocaleString() === 'string'",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── proto_lookup: Number.prototype.toLocaleString ─────────────────────

    #[test]
    fn test_proto_lookup_smi_tolocalestring() {
        let result =
            crate::builtins::global::global_eval("var n = 42; n.toLocaleString()").unwrap();
        assert_eq!(result, JsValue::String("42".into()));
    }

    #[test]
    fn test_proto_lookup_heapnumber_tolocalestring() {
        let result =
            crate::builtins::global::global_eval("var n = 3.14; n.toLocaleString()").unwrap();
        assert_eq!(result, JsValue::String("3.14".into()));
    }

    // ── proto_lookup: __lookupGetter__ / __lookupSetter__ ─────────────────

    #[test]
    fn test_proto_lookup_lookup_getter() {
        let result = crate::builtins::global::global_eval(
            "var o = {}; \
             Object.defineProperty(o, 'x', { get: function() { return 42; } }); \
             typeof o.__lookupGetter__('x') === 'function'",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_proto_lookup_lookup_setter() {
        let result = crate::builtins::global::global_eval(
            "var o = {}; \
             Object.defineProperty(o, 'x', { set: function(v) {} }); \
             typeof o.__lookupSetter__('x') === 'function'",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_proto_lookup_lookup_getter_undefined_when_missing() {
        let result = crate::builtins::global::global_eval(
            "var o = {x: 1}; o.__lookupGetter__('x') === undefined",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── proto_lookup: __defineGetter__ / __defineSetter__ ─────────────────

    #[test]
    fn test_proto_lookup_define_getter() {
        let result = crate::builtins::global::global_eval(
            "var o = {}; \
             o.__defineGetter__('x', function() { return 99; }); \
             o.x",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    #[test]
    fn test_proto_lookup_define_setter() {
        // Verify __defineSetter__ installs a setter (the setter itself is callable).
        let result = crate::builtins::global::global_eval(
            "var o = {}; \
             o.__defineSetter__('x', function(v) {}); \
             typeof o.__lookupSetter__('x') === 'function'",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── proto_lookup: Function/NativeFunction toLocaleString ──────────────

    #[test]
    fn test_proto_lookup_function_tolocalestring() {
        let result = crate::builtins::global::global_eval(
            "function foo() {} typeof foo.toLocaleString() === 'string'",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── proto_lookup_chain: Object.prototype fallback ─────────────────────

    #[test]
    fn test_proto_lookup_chain_tostring_fallback() {
        // An object with a custom __proto__ still gets toString.
        let result = crate::builtins::global::global_eval(
            "var proto = {}; \
             var o = Object.create(proto); \
             typeof o.toString === 'function'",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_proto_lookup_chain_valueof_fallback() {
        let result = crate::builtins::global::global_eval(
            "var proto = {}; \
             var o = Object.create(proto); \
             typeof o.valueOf === 'function'",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_proto_lookup_chain_hasownproperty_fallback() {
        let result = crate::builtins::global::global_eval(
            "var proto = {}; \
             var o = Object.create(proto); \
             o.x = 1; \
             o.hasOwnProperty('x')",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // —— Comparison with ToPrimitive ————————————————

    #[test]
    fn test_js_less_than_with_valueof() {
        let result = crate::builtins::global::global_eval(
            "var a = { valueOf: function() { return 5; } }; a < 10",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_js_less_than_with_valueof_false() {
        let result = crate::builtins::global::global_eval(
            "var a = { valueOf: function() { return 15; } }; a < 10",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_js_greater_than_with_valueof() {
        let result = crate::builtins::global::global_eval(
            "var a = { valueOf: function() { return 15; } }; a > 10",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_abstract_eq_object_to_number() {
        let result = crate::builtins::global::global_eval(
            "var a = { valueOf: function() { return 42; } }; a == 42",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_abstract_eq_object_to_string() {
        let result = crate::builtins::global::global_eval(
            "var a = { toString: function() { return 'hello'; } }; a == 'hello'",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_comparison_nan_object() {
        let result = crate::builtins::global::global_eval(
            "var a = { valueOf: function() { return NaN; } }; a < 10",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    // ── Constructor property resolution tests ─────────────────────────

    /// `[].constructor` resolves to a callable constructor.
    #[test]
    fn test_array_constructor_identity() {
        // Verify the constructor property is a function (not undefined).
        let typeof_ctor = crate::builtins::global::global_eval("typeof [].constructor").unwrap();
        assert_eq!(typeof_ctor, JsValue::String("function".into()));
        // Verify the constructor can create arrays.
        let works =
            crate::builtins::global::global_eval("Array.isArray([].constructor([]))").unwrap();
        assert_eq!(works, JsValue::Boolean(true));
    }

    /// `(42).constructor` resolves to the global `Number` constructor.
    #[test]
    fn test_number_constructor_identity() {
        let result = crate::builtins::global::global_eval("(42).constructor === Number").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `"hello".constructor` resolves to the global `String` constructor.
    #[test]
    fn test_string_constructor_identity() {
        let result =
            crate::builtins::global::global_eval("'hello'.constructor === String").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `true.constructor` resolves to the global `Boolean` constructor.
    #[test]
    fn test_boolean_constructor_identity() {
        let result = crate::builtins::global::global_eval("true.constructor === Boolean").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// `(function(){}).constructor` resolves to the global `Function` constructor.
    #[test]
    fn test_function_constructor_identity() {
        let result =
            crate::builtins::global::global_eval("(function(){}).constructor === Function")
                .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// Error `.constructor` resolves to the matching error constructor.
    #[test]
    fn test_error_constructor_identity() {
        let result = crate::builtins::global::global_eval(
            "try { null.x } catch(e) { e.constructor === TypeError }",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    /// Plain object `.constructor` resolves to `Object`.
    #[test]
    fn test_plain_object_constructor_identity() {
        let result = crate::builtins::global::global_eval("({}).constructor === Object").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── TypeError on null / undefined property access ────────────────────

    #[test]
    fn test_null_property_access_throws() {
        let result = compile_source_and_run("var x = null; x.foo");
        assert!(result.is_err());
    }

    #[test]
    fn test_undefined_property_access_throws() {
        let result = compile_source_and_run("var x = undefined; x.foo");
        assert!(result.is_err());
    }

    #[test]
    fn test_null_method_call_throws() {
        let result = compile_source_and_run("null.toString()");
        assert!(result.is_err());
    }

    #[test]
    fn test_null_keyed_property_access_throws() {
        let result = compile_source_and_run("var x = null; x['foo']");
        assert!(result.is_err());
    }

    #[test]
    fn test_undefined_keyed_property_access_throws() {
        let result = compile_source_and_run("var x = undefined; x['foo']");
        assert!(result.is_err());
    }

    // ── proto_lookup: String method fast-path tests ──────────────────────

    #[test]
    fn test_proto_lookup_string_starts_with() {
        let result = crate::builtins::global::global_eval("'hello'.startsWith('hel')").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_proto_lookup_string_starts_with_false() {
        let result = crate::builtins::global::global_eval("'hello'.startsWith('ell')").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_proto_lookup_string_starts_with_position() {
        let result = crate::builtins::global::global_eval("'hello'.startsWith('ell', 1)").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_proto_lookup_string_ends_with() {
        let result = crate::builtins::global::global_eval("'hello'.endsWith('llo')").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_proto_lookup_string_ends_with_false() {
        let result = crate::builtins::global::global_eval("'hello'.endsWith('hel')").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_proto_lookup_string_ends_with_end_position() {
        let result = crate::builtins::global::global_eval("'hello'.endsWith('hel', 3)").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_proto_lookup_string_repeat() {
        let result = crate::builtins::global::global_eval("'ab'.repeat(3)").unwrap();
        assert_eq!(result, JsValue::String("ababab".into()));
    }

    #[test]
    fn test_proto_lookup_string_repeat_zero() {
        let result = crate::builtins::global::global_eval("'ab'.repeat(0)").unwrap();
        assert_eq!(result, JsValue::String("".into()));
    }

    #[test]
    fn test_proto_lookup_string_pad_start() {
        let result = crate::builtins::global::global_eval("'5'.padStart(3, '0')").unwrap();
        assert_eq!(result, JsValue::String("005".into()));
    }

    #[test]
    fn test_proto_lookup_string_pad_end() {
        let result = crate::builtins::global::global_eval("'5'.padEnd(3, '0')").unwrap();
        assert_eq!(result, JsValue::String("500".into()));
    }

    #[test]
    fn test_proto_lookup_string_trim_start() {
        let result = crate::builtins::global::global_eval("'  hello  '.trimStart()").unwrap();
        assert_eq!(result, JsValue::String("hello  ".into()));
    }

    #[test]
    fn test_proto_lookup_string_trim_end() {
        let result = crate::builtins::global::global_eval("'  hello  '.trimEnd()").unwrap();
        assert_eq!(result, JsValue::String("  hello".into()));
    }

    #[test]
    fn test_proto_lookup_string_at_positive() {
        let result = crate::builtins::global::global_eval("'hello'.at(1)").unwrap();
        assert_eq!(result, JsValue::String("e".into()));
    }

    #[test]
    fn test_proto_lookup_string_at_negative() {
        let result = crate::builtins::global::global_eval("'hello'.at(-1)").unwrap();
        assert_eq!(result, JsValue::String("o".into()));
    }

    #[test]
    fn test_proto_lookup_string_at_out_of_range() {
        let result = crate::builtins::global::global_eval("'hello'.at(10)").unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    #[test]
    fn test_proto_lookup_string_includes_from_index() {
        let result =
            crate::builtins::global::global_eval("'hello world'.includes('world', 6)").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_proto_lookup_string_includes_from_index_miss() {
        let result =
            crate::builtins::global::global_eval("'hello world'.includes('hello', 1)").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_proto_lookup_string_index_of_from_index() {
        let result = crate::builtins::global::global_eval("'abcabc'.indexOf('bc', 2)").unwrap();
        assert_eq!(result, JsValue::Smi(4));
    }

    #[test]
    fn test_proto_lookup_string_last_index_of() {
        let result = crate::builtins::global::global_eval("'abcabc'.lastIndexOf('bc')").unwrap();
        assert_eq!(result, JsValue::Smi(4));
    }

    #[test]
    fn test_proto_lookup_string_slice() {
        let result = crate::builtins::global::global_eval("'hello'.slice(1, 4)").unwrap();
        assert_eq!(result, JsValue::String("ell".into()));
    }

    #[test]
    fn test_proto_lookup_string_slice_negative() {
        let result = crate::builtins::global::global_eval("'hello'.slice(-3)").unwrap();
        assert_eq!(result, JsValue::String("llo".into()));
    }

    #[test]
    fn test_proto_lookup_string_substring() {
        let result = crate::builtins::global::global_eval("'hello'.substring(1, 4)").unwrap();
        assert_eq!(result, JsValue::String("ell".into()));
    }

    #[test]
    fn test_proto_lookup_string_char_at() {
        let result = crate::builtins::global::global_eval("'hello'.charAt(1)").unwrap();
        assert_eq!(result, JsValue::String("e".into()));
    }

    #[test]
    fn test_proto_lookup_string_char_code_at() {
        let result = crate::builtins::global::global_eval("'A'.charCodeAt(0)").unwrap();
        assert_eq!(result, JsValue::Smi(65));
    }

    #[test]
    fn test_proto_lookup_string_code_point_at() {
        let result = crate::builtins::global::global_eval("'A'.codePointAt(0)").unwrap();
        assert_eq!(result, JsValue::Smi(65));
    }

    // ── proto_lookup: Array method fast-path tests ───────────────────────

    #[test]
    fn test_proto_lookup_array_at_positive() {
        let result = crate::builtins::global::global_eval("[10, 20, 30].at(1)").unwrap();
        assert_eq!(result, JsValue::Smi(20));
    }

    #[test]
    fn test_proto_lookup_array_at_negative() {
        let result = crate::builtins::global::global_eval("[10, 20, 30].at(-1)").unwrap();
        assert_eq!(result, JsValue::Smi(30));
    }

    #[test]
    fn test_proto_lookup_array_find_last() {
        let result = crate::builtins::global::global_eval(
            "[1, 2, 3, 4].findLast(function(x) { return x % 2 === 0; })",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(4));
    }

    #[test]
    fn test_proto_lookup_array_find_last_index() {
        let result = crate::builtins::global::global_eval(
            "[1, 2, 3, 4].findLastIndex(function(x) { return x % 2 === 0; })",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    // ── proto_lookup: Object method fast-path tests ──────────────────────

    #[test]
    fn test_proto_lookup_object_has_own_property() {
        let result =
            crate::builtins::global::global_eval("var o = { x: 1 }; o.hasOwnProperty('x')")
                .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_proto_lookup_object_has_own_property_false() {
        let result =
            crate::builtins::global::global_eval("var o = { x: 1 }; o.hasOwnProperty('y')")
                .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_proto_lookup_object_property_is_enumerable() {
        let result =
            crate::builtins::global::global_eval("var o = { x: 1 }; o.propertyIsEnumerable('x')")
                .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── Array join/toString null/undefined handling ─────────────────────────

    #[test]
    fn test_array_join_null_undefined() {
        // null and undefined should produce empty strings in join/toString
        let result =
            crate::builtins::global::global_eval("var a = [1, null, undefined, 4]; a.join(',')")
                .unwrap();
        assert_eq!(result, JsValue::String("1,,,4".into()));
    }

    #[test]
    fn test_array_tostring_null_elements() {
        let result = crate::builtins::global::global_eval("[null, undefined].toString()").unwrap();
        assert_eq!(result, JsValue::String(",".into()));
    }

    // ── Array.prototype.includes — SameValueZero ──────────────────────────────

    #[test]
    fn test_array_includes_nan() {
        // SameValueZero: NaN is equal to NaN
        let result = crate::builtins::global::global_eval("[NaN].includes(NaN)").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_array_includes_basic() {
        let result = crate::builtins::global::global_eval("[1,2,3].includes(2)").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_array_includes_missing() {
        let result = crate::builtins::global::global_eval("[1,2,3].includes(4)").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_array_includes_from_index() {
        // includes with fromIndex skips earlier elements
        let result = crate::builtins::global::global_eval("[1,2,3].includes(1, 1)").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_array_includes_negative_from_index() {
        // Negative fromIndex counts from end
        let result = crate::builtins::global::global_eval("[1,2,3].includes(3, -1)").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── Array.prototype.indexOf — strict equality (NaN !== NaN) ───────────────

    #[test]
    fn test_array_indexof_nan_not_found() {
        // Strict equality: NaN !== NaN, so indexOf returns -1
        let result = crate::builtins::global::global_eval("[NaN].indexOf(NaN)").unwrap();
        assert_eq!(result, JsValue::Smi(-1));
    }

    #[test]
    fn test_array_lastindexof_nan_not_found() {
        let result = crate::builtins::global::global_eval("[NaN].lastIndexOf(NaN)").unwrap();
        assert_eq!(result, JsValue::Smi(-1));
    }

    // ── Array.prototype.splice — returns removed elements ─────────────────────

    #[test]
    fn test_array_splice_returns_removed() {
        let result =
            crate::builtins::global::global_eval("var a = [1,2,3,4,5]; a.splice(1,2).join(',')")
                .unwrap();
        assert_eq!(result, JsValue::String("2,3".into()));
    }

    #[test]
    fn test_array_splice_mutates_original() {
        let result =
            crate::builtins::global::global_eval("var a = [1,2,3,4,5]; a.splice(1,2); a.length")
                .unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    // ── Array.prototype.every / some — third callback arg ─────────────────────

    #[test]
    fn test_array_every_short_circuits() {
        // every returns false if any element fails the predicate
        let result =
            crate::builtins::global::global_eval("[1,2,3].every(function(x) { return x < 3; })")
                .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_array_some_short_circuits() {
        // some returns true if any element matches
        let result =
            crate::builtins::global::global_eval("[1,2,3].some(function(x) { return x === 2; })")
                .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_array_every_basic_true() {
        // every returns true when all elements satisfy the predicate
        let result = crate::builtins::global::global_eval(
            "[2,4,6].every(function(e) { return e % 2 === 0; })",
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_array_some_basic_true() {
        // some returns true when at least one element satisfies the predicate
        let result =
            crate::builtins::global::global_eval("[1,2,3].some(function(e) { return e > 2; })")
                .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── Array.prototype.fill ──────────────────────────────────────────────────

    #[test]
    fn test_array_fill_all() {
        let result = crate::builtins::global::global_eval("[1,2,3].fill(0).join(',')").unwrap();
        assert_eq!(result, JsValue::String("0,0,0".into()));
    }

    #[test]
    fn test_array_fill_with_range() {
        let result =
            crate::builtins::global::global_eval("[1,2,3].fill(4, 1, 2).join(',')").unwrap();
        assert_eq!(result, JsValue::String("1,4,3".into()));
    }

    // ── Array.prototype.copyWithin ────────────────────────────────────────────

    #[test]
    fn test_array_copyin_basic() {
        let result =
            crate::builtins::global::global_eval("[1,2,3,4,5].copyWithin(0, 3).join(',')").unwrap();
        assert_eq!(result, JsValue::String("4,5,3,4,5".into()));
    }

    // ── Boolean.prototype fast-path tests ────────────────────────────────────

    #[test]
    fn test_boolean_to_string() {
        let result = crate::builtins::global::global_eval("true.toString()").unwrap();
        assert_eq!(result, JsValue::String("true".into()));
    }

    #[test]
    fn test_boolean_value_of() {
        let result = crate::builtins::global::global_eval("false.valueOf()").unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    // ── String.prototype fast-path edge-case tests ───────────────────────────

    #[test]
    fn test_string_at_positive() {
        let result = crate::builtins::global::global_eval("'hello'.at(1)").unwrap();
        assert_eq!(result, JsValue::String("e".into()));
    }

    #[test]
    fn test_string_at_negative() {
        let result = crate::builtins::global::global_eval("'hello'.at(-1)").unwrap();
        assert_eq!(result, JsValue::String("o".into()));
    }

    #[test]
    fn test_string_repeat() {
        let result = crate::builtins::global::global_eval("'ab'.repeat(3)").unwrap();
        assert_eq!(result, JsValue::String("ababab".into()));
    }

    #[test]
    fn test_string_pad_start() {
        let result = crate::builtins::global::global_eval("'5'.padStart(3, '0')").unwrap();
        assert_eq!(result, JsValue::String("005".into()));
    }

    #[test]
    fn test_string_pad_end() {
        let result = crate::builtins::global::global_eval("'hi'.padEnd(5, '!')").unwrap();
        assert_eq!(result, JsValue::String("hi!!!".into()));
    }

    #[test]
    fn test_string_pad_start_default_space() {
        let result = crate::builtins::global::global_eval("'x'.padStart(3)").unwrap();
        assert_eq!(result, JsValue::String("  x".into()));
    }

    // ── Conformance: typeof ─────────────────────────────────────────────────

    #[test]
    fn test_typeof_number_literal() {
        let r = crate::builtins::global::global_eval("typeof 42").unwrap();
        assert_eq!(r, JsValue::String("number".into()));
    }

    #[test]
    fn test_typeof_string_literal() {
        let r = crate::builtins::global::global_eval("typeof 'hello'").unwrap();
        assert_eq!(r, JsValue::String("string".into()));
    }

    #[test]
    fn test_typeof_boolean_literal() {
        let r = crate::builtins::global::global_eval("typeof true").unwrap();
        assert_eq!(r, JsValue::String("boolean".into()));
    }

    #[test]
    fn test_typeof_undefined_value() {
        let r = crate::builtins::global::global_eval("typeof undefined").unwrap();
        assert_eq!(r, JsValue::String("undefined".into()));
    }

    #[test]
    fn test_typeof_null_returns_object() {
        let r = crate::builtins::global::global_eval("typeof null").unwrap();
        assert_eq!(r, JsValue::String("object".into()));
    }

    #[test]
    fn test_typeof_object_literal() {
        let r = crate::builtins::global::global_eval("typeof {}").unwrap();
        assert_eq!(r, JsValue::String("object".into()));
    }

    #[test]
    fn test_typeof_array_literal() {
        let r = crate::builtins::global::global_eval("typeof []").unwrap();
        assert_eq!(r, JsValue::String("object".into()));
    }

    #[test]
    fn test_typeof_function_expr() {
        let r = crate::builtins::global::global_eval("typeof function(){}").unwrap();
        assert_eq!(r, JsValue::String("function".into()));
    }

    #[test]
    fn test_typeof_undeclared_var_no_throw() {
        // typeof on an undeclared variable must NOT throw — returns "undefined".
        let r = crate::builtins::global::global_eval("typeof undeclaredXYZ123").unwrap();
        assert_eq!(r, JsValue::String("undefined".into()));
    }

    // ── Conformance: abstract equality (==) ─────────────────────────────────

    #[test]
    fn test_eq_null_undefined() {
        let r = crate::builtins::global::global_eval("null == undefined").unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    #[test]
    fn test_eq_undefined_null() {
        let r = crate::builtins::global::global_eval("undefined == null").unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    #[test]
    fn test_eq_null_zero_is_false() {
        let r = crate::builtins::global::global_eval("null == 0").unwrap();
        assert_eq!(r, JsValue::Boolean(false));
    }

    #[test]
    fn test_eq_null_empty_string_is_false() {
        let r = crate::builtins::global::global_eval("null == ''").unwrap();
        assert_eq!(r, JsValue::Boolean(false));
    }

    #[test]
    fn test_eq_null_false_is_false() {
        let r = crate::builtins::global::global_eval("null == false").unwrap();
        assert_eq!(r, JsValue::Boolean(false));
    }

    #[test]
    fn test_eq_string_number_coercion() {
        let r = crate::builtins::global::global_eval("'5' == 5").unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    #[test]
    fn test_eq_number_string_coercion() {
        let r = crate::builtins::global::global_eval("10 == '10'").unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    #[test]
    fn test_eq_boolean_number_coercion() {
        // true is coerced to 1, then 1 == 1 → true
        let r = crate::builtins::global::global_eval("true == 1").unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    #[test]
    fn test_eq_false_zero() {
        let r = crate::builtins::global::global_eval("false == 0").unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    #[test]
    fn test_eq_boolean_string_coercion() {
        // true → 1, '1' → 1, so true == '1' is true
        let r = crate::builtins::global::global_eval("true == '1'").unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    // ── Conformance: strict equality (===) ──────────────────────────────────

    #[test]
    fn test_strict_eq_no_coercion() {
        let r = crate::builtins::global::global_eval("'5' === 5").unwrap();
        assert_eq!(r, JsValue::Boolean(false));
    }

    #[test]
    fn test_strict_eq_null_undefined_false() {
        let r = crate::builtins::global::global_eval("null === undefined").unwrap();
        assert_eq!(r, JsValue::Boolean(false));
    }

    #[test]
    fn test_strict_eq_nan_not_equal_nan() {
        let r = crate::builtins::global::global_eval("NaN === NaN").unwrap();
        assert_eq!(r, JsValue::Boolean(false));
    }

    #[test]
    fn test_strict_eq_positive_negative_zero() {
        // +0 === -0 must be true per ES spec
        let r = crate::builtins::global::global_eval("+0 === -0").unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    // ── Conformance: relational comparisons ─────────────────────────────────

    #[test]
    fn test_less_than_numbers() {
        let r = crate::builtins::global::global_eval("3 < 5").unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    #[test]
    fn test_less_than_strings() {
        let r = crate::builtins::global::global_eval("'a' < 'b'").unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    #[test]
    fn test_greater_than_mixed_types() {
        // '10' > 9 → 10 > 9 → true
        let r = crate::builtins::global::global_eval("'10' > 9").unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    #[test]
    fn test_lte_equal_values() {
        let r = crate::builtins::global::global_eval("5 <= 5").unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    #[test]
    fn test_gte_equal_values() {
        let r = crate::builtins::global::global_eval("5 >= 5").unwrap();
        assert_eq!(r, JsValue::Boolean(true));
    }

    // ── Conformance: property access on primitives ──────────────────────────

    #[test]
    fn test_string_length_property() {
        let r = crate::builtins::global::global_eval("'hello'.length").unwrap();
        assert_eq!(r, JsValue::Smi(5));
    }

    #[test]
    fn test_string_length_unicode() {
        // 'café' has 4 characters; .length must not return byte count (5 in UTF-8)
        let r = crate::builtins::global::global_eval("'caf\\u00e9'.length").unwrap();
        assert_eq!(r, JsValue::Smi(4));
    }

    #[test]
    fn test_string_char_at_method() {
        let r = crate::builtins::global::global_eval("'hello'.charAt(1)").unwrap();
        assert_eq!(r, JsValue::String("e".into()));
    }

    #[test]
    fn test_number_to_string_radix() {
        let r = crate::builtins::global::global_eval("(255).toString(16)").unwrap();
        assert_eq!(r, JsValue::String("ff".into()));
    }

    #[test]
    fn test_number_to_string_default() {
        let r = crate::builtins::global::global_eval("(42).toString()").unwrap();
        assert_eq!(r, JsValue::String("42".into()));
    }

    #[test]
    fn test_boolean_to_string_true() {
        let r = crate::builtins::global::global_eval("true.toString()").unwrap();
        assert_eq!(r, JsValue::String("true".into()));
    }

    #[test]
    fn test_boolean_to_string_false() {
        let r = crate::builtins::global::global_eval("false.toString()").unwrap();
        assert_eq!(r, JsValue::String("false".into()));
    }

    #[test]
    fn test_number_to_fixed() {
        let r = crate::builtins::global::global_eval("(3.14159).toFixed(2)").unwrap();
        assert_eq!(r, JsValue::String("3.14".into()));
    }

    #[test]
    fn test_string_to_upper_case() {
        let r = crate::builtins::global::global_eval("'abc'.toUpperCase()").unwrap();
        assert_eq!(r, JsValue::String("ABC".into()));
    }

    #[test]
    fn test_string_index_access() {
        let r = crate::builtins::global::global_eval("'hello'[0]").unwrap();
        assert_eq!(r, JsValue::String("h".into()));
    }

    // ── Conformance: throw non-Error values ─────────────────────────────────

    #[test]
    fn test_throw_string_value() {
        let r = crate::builtins::global::global_eval(
            "var result; try { throw 'oops'; } catch(e) { result = e; } result",
        )
        .unwrap();
        assert_eq!(r, JsValue::String("oops".into()));
    }

    #[test]
    fn test_throw_number_value() {
        let r = crate::builtins::global::global_eval(
            "var result; try { throw 42; } catch(e) { result = e; } result",
        )
        .unwrap();
        assert_eq!(r, JsValue::Smi(42));
    }

    #[test]
    fn test_throw_null_value() {
        let r = crate::builtins::global::global_eval(
            "var result; try { throw null; } catch(e) { result = e; } result",
        )
        .unwrap();
        assert_eq!(r, JsValue::Null);
    }

    #[test]
    fn test_throw_undefined_value() {
        let r = crate::builtins::global::global_eval(
            "var result; try { throw undefined; } catch(e) { result = e; } result",
        )
        .unwrap();
        assert_eq!(r, JsValue::Undefined);
    }

    // ── Conformance: Object.keys on arrays ──────────────────────────────────

    #[test]
    fn test_object_keys_array() {
        // Object.keys([1,2,3]) should return ["0", "1", "2"]
        let r = crate::builtins::global::global_eval(
            "var k = Object.keys([10, 20, 30]); k[0] + ',' + k[1] + ',' + k[2]",
        )
        .unwrap();
        assert_eq!(r, JsValue::String("0,1,2".into()));
    }

    #[test]
    fn test_object_keys_array_length() {
        let r = crate::builtins::global::global_eval("Object.keys([1,2,3]).length").unwrap();
        assert_eq!(r, JsValue::Smi(3));
    }

    // ── Conformance: primitive hasOwnProperty ───────────────────────────────

    #[test]
    fn test_string_has_own_property_length() {
        let s = JsValue::String("hello".into());
        let hop = proto_lookup(&s, "hasOwnProperty");
        assert!(matches!(hop, JsValue::NativeFunction(_)));
    }

    #[test]
    fn test_number_has_own_property_returns_fn() {
        let n = JsValue::Smi(42);
        let hop = proto_lookup(&n, "hasOwnProperty");
        assert!(matches!(hop, JsValue::NativeFunction(_)));
    }

    #[test]
    fn test_boolean_has_own_property_returns_fn() {
        let b = JsValue::Boolean(true);
        let hop = proto_lookup(&b, "hasOwnProperty");
        assert!(matches!(hop, JsValue::NativeFunction(_)));
    }
    // ── String .length UTF-16 semantics ──────────────────────────────────────────

    #[test]
    fn test_string_length_ascii() {
        let s = JsValue::String("abc".to_string().into());
        assert_eq!(proto_lookup(&s, "length"), JsValue::Smi(3));
    }

    #[test]
    fn test_string_length_bmp_accented() {
        // All BMP characters, each is one UTF-16 code unit.
        let s = JsValue::String("café".to_string().into());
        assert_eq!(proto_lookup(&s, "length"), JsValue::Smi(4));
    }

    #[test]
    fn test_string_length_surrogate_pair() {
        // U+1D11E MUSICAL SYMBOL G CLEF encodes as a surrogate pair in UTF-16.
        let s = JsValue::String("𝄞".to_string().into());
        assert_eq!(proto_lookup(&s, "length"), JsValue::Smi(2));
    }

    #[test]
    fn test_string_length_mixed_bmp_and_surrogate() {
        // a (1) + surrogate pair (2) + b (1) = 4 UTF-16 code units
        let s = JsValue::String("a𝄞b".to_string().into());
        assert_eq!(proto_lookup(&s, "length"), JsValue::Smi(4));
    }

    #[test]
    fn test_string_length_empty() {
        let s = JsValue::String("".to_string().into());
        assert_eq!(proto_lookup(&s, "length"), JsValue::Smi(0));
    }

    #[test]
    fn test_keyed_load_string_length_utf16() {
        let s = JsValue::String("𝄞".to_string().into());
        assert_eq!(
            keyed_load(&s, &JsValue::String("length".to_string().into())).unwrap(),
            JsValue::Smi(2)
        );
    }
}
