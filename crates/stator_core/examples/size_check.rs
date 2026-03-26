use stator_core::bytecode::bytecodes::{Instruction, Operand};
use stator_core::interpreter::InterpreterFrame;
use stator_core::objects::value::JsValue;
use std::mem::size_of;

fn main() {
    println!("Instruction size: {} bytes", size_of::<Instruction>());
    println!("Operand size: {} bytes", size_of::<Operand>());
    println!("JsValue size: {} bytes", size_of::<JsValue>());
    println!(
        "InterpreterFrame size: {} bytes",
        size_of::<InterpreterFrame>()
    );
}
