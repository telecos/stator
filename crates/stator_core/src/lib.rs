#![allow(clippy::collapsible_match)]
#![allow(clippy::let_and_return)]
#![allow(clippy::needless_else)]
#![allow(clippy::unnecessary_sort_by)]
//! `stator_js` — the foundational engine library for the Stator JavaScript
//! engine.
//!
//! # Crate layout
//!
//! - [`error`] — Engine error types and `StatorResult` alias.
//! - [`gc`] — Garbage collector infrastructure (heap, tracing, handles).
//! - [`objects`] — JavaScript value representation and heap object types.
//! - [`sandbox`] — Memory sandbox: virtual-address range reservation, pointer
//!   bounds checking, and external pointer table.
//! - [`zone`] — Bump-pointer region allocator for compiler temporaries.
//! - [`parser`] — Lexer ([`parser::scanner`]), AST ([`parser::ast`]), and
//!   scope analysis ([`parser::scope`]), and lazy parsing
//!   ([`parser::preparser`])
//! - [`interpreter`] — Bytecode interpreter ([`interpreter::Interpreter`]):
//!   fetch-decode-dispatch loop, [`interpreter::InterpreterFrame`] activation
//!   frame, and arithmetic/comparison opcode handlers.
//! - [`bytecode`] — Bytecode instruction set ([`bytecode::bytecodes`]):
//!   ~174 opcodes matching V8 Ignition semantics, operand encoding, and
//!   encode/decode utilities.  [`bytecode::bytecode_array`] provides the
//!   compact [`bytecode::bytecode_array::BytecodeArray`] type with constant
//!   pool and source-position table.  [`bytecode::register`] provides the
//!   [`bytecode::register::Register`] type and
//!   [`bytecode::register::RegisterAllocator`] for register assignment during
//!   compilation.  [`bytecode::bytecode_generator`] provides the
//!   [`bytecode::bytecode_generator::BytecodeGenerator`] that compiles a
//!   JavaScript AST into a [`bytecode::bytecode_array::BytecodeArray`].
//!   [`bytecode::feedback`] provides [`bytecode::feedback::FeedbackVector`],
//!   [`bytecode::feedback::FeedbackMetadata`],
//!   [`bytecode::feedback::FeedbackSlotKind`], and
//!   [`bytecode::feedback::InlineCacheState`] for inline-cache feedback.

/// Built-in JavaScript object static methods (`Object`, `Promise`, …).
pub mod builtins;
/// Bytecode instruction set and encode/decode utilities.
pub mod bytecode;
/// JIT compiler infrastructure: baseline code generator and macro-assembler.
pub mod compiler;
/// DOM integration layer: object wrapping, property interceptors, internal
/// fields, and weak references for bridging the JS engine with web platform
/// APIs.
pub mod dom;
/// Engine error types and [`StatorResult`] alias.
pub mod error;
/// Event loop integration: macrotask scheduling, timer management, and
/// microtask queue draining for embedder coordination.
pub mod event_loop;
/// V8-compatible FFI wrapper types (`V8Object`, `V8Array`, `V8Number`, etc.).
pub mod ffi;
/// Garbage collector infrastructure: heap, tracing, and handle scopes.
pub mod gc;
/// Inline-cache runtime: property load/store fast paths and call-site tracking.
pub mod ic;
/// Chrome DevTools Protocol (CDP) WebSocket inspector server and sampling CPU
/// profiler.
pub mod inspector;
/// Bytecode interpreter: fetch-decode-dispatch loop and activation frame.
pub mod interpreter;
/// JavaScript value representation and heap object types.
pub mod objects;
/// JavaScript parser infrastructure (lexer and AST node definitions).
pub mod parser;
/// Platform abstraction for embedder-provided task scheduling and timing.
pub mod platform;
/// Memory sandbox: virtual-address range reservation, pointer bounds checking,
/// and external pointer table for non-sandbox memory.
pub mod sandbox;
/// Startup snapshot: binary serialization and deserialization of heap state.
pub mod snapshot;
/// WebAssembly backend: engine, module, and instance wrappers (Wasmtime).
pub mod wasm;
/// Bump-pointer region allocator for compiler temporaries.
pub mod zone;
