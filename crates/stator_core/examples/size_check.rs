use stator_js::bytecode::bytecodes::{Instruction, Operand};
use stator_js::interpreter::InterpreterFrame;
use stator_js::objects::value::JsValue;
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
