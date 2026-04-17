#![no_main]

use libfuzzer_sys::fuzz_target;
use stator_js::parser::scanner::Scanner;

fuzz_target!(|data: &[u8]| {
    // Convert raw bytes to a string — invalid UTF-8 is silently replaced so
    // the scanner always receives a valid &str.
    let source = String::from_utf8_lossy(data);

    // Drive the scanner token-by-token until EOF or an error.  We must not
    // panic regardless of the input.
    let mut scanner = Scanner::new(&source);
    while let Ok(tok) = scanner.next_token() {
        use stator_js::parser::scanner::TokenKind;
        if tok.kind == TokenKind::Eof {
            break;
        }
    }
});
