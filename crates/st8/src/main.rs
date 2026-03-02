//! `st8` — Stator JavaScript shell.
//!
//! `st8` is the interactive CLI shell for the Stator engine, analogous to
//! V8's `d8`.  It will provide a REPL, script execution, and debugging
//! utilities once the interpreter is functional.  For now it prints a
//! placeholder message so the workspace compiles end-to-end.

fn main() {
    println!("st8: Stator JavaScript shell (not yet implemented)");
}

#[cfg(test)]
mod tests {
    use stator_core::parser::scanner::{Scanner, TokenKind};

    #[test]
    fn test_shell_scanner_tokenises_number_literal() {
        let mut s = Scanner::new("42");
        let tok = s.next_token().unwrap();
        assert_eq!(tok.kind, TokenKind::NumericLiteral);
    }

    #[test]
    fn test_shell_scanner_tokenises_identifier() {
        let mut s = Scanner::new("foo");
        let tok = s.next_token().unwrap();
        assert_eq!(tok.kind, TokenKind::Identifier);
    }
}
