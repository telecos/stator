//! Console message buffer for CDP `Console` domain forwarding.
//!
//! JavaScript `console.log`, `console.warn`, `console.error`, etc. calls push
//! [`ConsoleMessage`]s into a thread-local ring buffer.  When the CDP
//! `Console.enable` method is invoked, the [`CdpSession`](super::cdp::CdpSession)
//! drains the buffer and emits `Console.messageAdded` events to the DevTools
//! frontend.
//!
//! # Example
//!
//! ```
//! use stator_core::inspector::console::{push_console_message, drain_messages, ConsoleMessage, MessageLevel};
//!
//! push_console_message(ConsoleMessage {
//!     level: MessageLevel::Log,
//!     text: "hello from JS".to_string(),
//! });
//!
//! let msgs = drain_messages();
//! assert_eq!(msgs.len(), 1);
//! assert_eq!(msgs[0].text, "hello from JS");
//! ```

use std::cell::RefCell;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// Severity level for a console message, matching CDP `Console.ConsoleMessage.level`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageLevel {
    /// Informational message (`console.log`).
    Log,
    /// Warning message (`console.warn`).
    Warning,
    /// Error message (`console.error`).
    Error,
    /// Debug message (`console.debug`).
    Debug,
    /// Informational message (`console.info`).
    Info,
}

impl MessageLevel {
    /// CDP string representation used in `Console.messageAdded` events.
    pub fn as_cdp_str(self) -> &'static str {
        match self {
            Self::Log => "log",
            Self::Warning => "warning",
            Self::Error => "error",
            Self::Debug => "debug",
            Self::Info => "info",
        }
    }
}

/// A single console message produced by JavaScript code.
#[derive(Debug, Clone)]
pub struct ConsoleMessage {
    /// Severity level.
    pub level: MessageLevel,
    /// Text content of the message.
    pub text: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Thread-local message buffer
// ─────────────────────────────────────────────────────────────────────────────

thread_local! {
    /// Buffered console messages waiting to be forwarded to a CDP client.
    static MESSAGES: RefCell<Vec<ConsoleMessage>> = const { RefCell::new(Vec::new()) };
}

/// Push a console message into the thread-local buffer.
///
/// Called by `console.log` / `console.warn` / `console.error` built-in
/// implementations.
pub fn push_console_message(msg: ConsoleMessage) {
    MESSAGES.with(|m| m.borrow_mut().push(msg));
}

/// Drain all buffered console messages, returning them in insertion order.
///
/// The internal buffer is left empty after this call.
pub fn drain_messages() -> Vec<ConsoleMessage> {
    MESSAGES.with(|m| m.borrow_mut().drain(..).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_drain_messages() {
        // Clear any leftover state.
        let _ = drain_messages();

        push_console_message(ConsoleMessage {
            level: MessageLevel::Log,
            text: "one".to_string(),
        });
        push_console_message(ConsoleMessage {
            level: MessageLevel::Error,
            text: "two".to_string(),
        });

        let msgs = drain_messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].text, "one");
        assert_eq!(msgs[0].level, MessageLevel::Log);
        assert_eq!(msgs[1].text, "two");
        assert_eq!(msgs[1].level, MessageLevel::Error);
    }

    #[test]
    fn test_drain_empties_buffer() {
        let _ = drain_messages();

        push_console_message(ConsoleMessage {
            level: MessageLevel::Warning,
            text: "warn".to_string(),
        });

        let first = drain_messages();
        assert_eq!(first.len(), 1);

        let second = drain_messages();
        assert!(second.is_empty());
    }

    #[test]
    fn test_drain_empty_returns_empty() {
        let _ = drain_messages();
        let msgs = drain_messages();
        assert!(msgs.is_empty());
    }

    #[test]
    fn test_message_level_as_cdp_str() {
        assert_eq!(MessageLevel::Log.as_cdp_str(), "log");
        assert_eq!(MessageLevel::Warning.as_cdp_str(), "warning");
        assert_eq!(MessageLevel::Error.as_cdp_str(), "error");
        assert_eq!(MessageLevel::Debug.as_cdp_str(), "debug");
        assert_eq!(MessageLevel::Info.as_cdp_str(), "info");
    }
}
