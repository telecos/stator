//! Tests for the deterministic execution-step budget exposed by
//! [`regress::Regex::find_from_with_budget`].
//!
//! These tests are deliberately small and do not depend on wall-clock
//! timing — they verify that:
//!
//! * A normal, well-behaved pattern returns `Ok(Some(_))` well below any
//!   reasonable budget.
//! * A classic catastrophic-backtracking pattern aborts with
//!   `Err(BudgetExhausted)` when given a tiny budget, deterministically.
//! * A non-matching short pattern returns `Ok(None)` when the budget is
//!   ample.
//! * Re-running the same `(pattern, input, budget)` triple yields the
//!   same outcome (determinism).

use regress::{BudgetExhausted, Regex};

#[test]
fn budget_allows_short_pattern_to_succeed() {
    let re = Regex::new(r"\d+").unwrap();
    let m = re
        .find_from_with_budget("price 42", 0, 1_000_000)
        .expect("budget should not be exhausted for trivial pattern");
    let m = m.expect("pattern should match");
    assert_eq!(m.range(), 6..8);
}

#[test]
fn budget_allows_short_pattern_to_report_no_match() {
    let re = Regex::new(r"\d+").unwrap();
    let result = re
        .find_from_with_budget("no digits here", 0, 1_000_000)
        .expect("budget should not be exhausted for trivial pattern");
    assert!(result.is_none(), "pattern should not match");
}

#[test]
fn catastrophic_backtracking_pattern_exhausts_tiny_budget() {
    // `^(a+)+b$` has exponential backtracking behaviour on a long run of
    // 'a' characters with no trailing 'b'. A small budget must catch this
    // well before it spends real time.
    let re = Regex::new(r"^(a+)+b$").unwrap();
    let input: String = std::iter::repeat('a').take(32).collect();
    assert!(matches!(
        re.find_from_with_budget(&input, 0, 1_000),
        Err(BudgetExhausted)
    ));
}

#[test]
fn budget_exhaustion_is_deterministic() {
    // Re-running the same call must always produce the same outcome.
    let re = Regex::new(r"^(a+)+b$").unwrap();
    let input: String = std::iter::repeat('a').take(28).collect();
    let r1 = re.find_from_with_budget(&input, 0, 500);
    let r2 = re.find_from_with_budget(&input, 0, 500);
    assert!(matches!(r1, Err(BudgetExhausted)));
    assert!(matches!(r2, Err(BudgetExhausted)));
}

#[test]
fn generous_budget_allows_modestly_complex_pattern() {
    // Same pattern as above but the input contains the trailing `b`, so
    // the optimised path finds the match quickly and never approaches
    // the budget cap.
    let re = Regex::new(r"^(a+)+b$").unwrap();
    let input = format!("{}b", "a".repeat(8));
    let m = re
        .find_from_with_budget(&input, 0, 1_000_000)
        .expect("generous budget should not be exhausted");
    let m = m.expect("pattern should match");
    assert_eq!(m.range(), 0..input.len());
}

#[test]
fn budget_does_not_affect_non_budgeted_apis() {
    // Calling `find_from` after a budgeted call must still work and
    // must not inherit any budget state (each call constructs a fresh
    // executor with no budget cap).
    let re = Regex::new(r"\w+").unwrap();
    let _ = re.find_from_with_budget("hello", 0, 1);
    let m = re.find("hello world").unwrap();
    assert_eq!(m.range(), 0..5);
}
