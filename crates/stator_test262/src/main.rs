//! `stator_test262` — Test262 conformance harness for the Stator engine.
//!
//! This binary will drive the ECMAScript Test262 test suite against Stator
//! once the interpreter is functional.  For now it prints a placeholder
//! message so the workspace compiles end-to-end.

fn main() {
    println!("stator_test262: Test262 conformance harness (not yet implemented)");
}

#[cfg(test)]
mod tests {
    use stator_core::parser::scanner::{Scanner, TokenKind};

    #[test]
    fn test_harness_scanner_tokenises_string_literal() {
        let mut s = Scanner::new("\"hello\"");
        let tok = s.next_token().unwrap();
        assert_eq!(tok.kind, TokenKind::StringLiteral);
    }

    #[test]
    fn test_harness_scanner_eof_after_input() {
        let mut s = Scanner::new("1");
        s.next_token().unwrap(); // consume `1`
        let eof = s.next_token().unwrap();
        assert_eq!(eof.kind, TokenKind::Eof);
    }
}
