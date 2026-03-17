//! ECMA-402 `Intl` namespace — basic stub implementations.
//!
//! Provides the `Intl` global namespace with constructor stubs for:
//! - `Intl.NumberFormat`
//! - `Intl.DateTimeFormat`
//! - `Intl.Collator`
//! - `Intl.PluralRules`
//! - `Intl.ListFormat`
//! - `Intl.RelativeTimeFormat`
//! - `Intl.Segmenter`
//! - `Intl.DisplayNames`
//! - `Intl.Locale`
//!
//! This initial implementation uses Rust standard library formatting and
//! returns reasonable en-US results.  Full ICU (icu4x) support is deferred.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::rc::Rc;

use crate::error::StatorResult;
use crate::objects::property_map::PropertyMap;
use crate::objects::value::JsValue;

// ── NumberFormat ──────────────────────────────────────────────────────────────

/// Format a number as a decimal string.
///
/// This stub always uses the default Rust `Display` formatter which produces
/// output comparable to `en-US` locale formatting without grouping separators.
pub fn number_format(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_positive() {
            "∞".to_string()
        } else {
            "-∞".to_string()
        };
    }
    // Remove trailing ".0" for integral values to match JS behaviour.
    if value.fract() == 0.0 && value.abs() < (i64::MAX as f64) {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

// ── DateTimeFormat ───────────────────────────────────────────────────────────

/// Format a millisecond-since-epoch timestamp as a human-readable date string.
///
/// Returns a basic ISO-8601-like representation.  Full locale-aware formatting
/// will be provided once icu4x integration lands.
pub fn date_time_format(ms_epoch: f64) -> String {
    if ms_epoch.is_nan() || ms_epoch.is_infinite() {
        return "Invalid Date".to_string();
    }
    let secs = (ms_epoch / 1000.0).trunc() as i64;
    let (year, month, day, hour, min, sec) = epoch_to_components(secs);
    format!("{month}/{day}/{year}, {hour}:{min:02}:{sec:02} AM")
}

/// Convert Unix epoch seconds to calendar components (UTC).
fn epoch_to_components(epoch_secs: i64) -> (i64, u32, u32, u32, u32, u32) {
    // Civil date from epoch using the algorithm from
    // <https://howardhinnant.github.io/date_algorithms.html>.
    let z = epoch_secs.div_euclid(86400) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097) as u64; // day of era
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };

    let day_secs = epoch_secs.rem_euclid(86400) as u32;
    let h = day_secs / 3600;
    let min = (day_secs % 3600) / 60;
    let sec = day_secs % 60;

    (y, m, d, h, min, sec)
}

// ── Collator ─────────────────────────────────────────────────────────────────

/// Compare two strings using byte-level (code-unit) ordering.
///
/// Returns `−1`, `0`, or `1` matching the `Intl.Collator.prototype.compare`
/// contract.  Locale-sensitive collation will be added with icu4x.
pub fn collator_compare(a: &str, b: &str) -> i32 {
    match a.cmp(b) {
        Ordering::Less => -1,
        Ordering::Equal => 0,
        Ordering::Greater => 1,
    }
}

// ── PluralRules ──────────────────────────────────────────────────────────────

/// Select the CLDR plural category for a number (en-US rules).
///
/// Returns `"one"` for 1, `"other"` for everything else.  Full CLDR plural
/// rules for additional locales will come with icu4x.
pub fn plural_rules_select(n: f64) -> &'static str {
    if n == 1.0 { "one" } else { "other" }
}

// ── ListFormat ───────────────────────────────────────────────────────────────

/// Join a list of strings using English conjunction rules.
///
/// `list_type` accepts `"conjunction"` (default, "and") or `"disjunction"` ("or").
pub fn list_format(items: &[String], list_type: &str) -> String {
    let conjunction = if list_type == "disjunction" {
        "or"
    } else {
        "and"
    };
    match items.len() {
        0 => String::new(),
        1 => items[0].clone(),
        2 => format!("{} {conjunction} {}", items[0], items[1]),
        _ => {
            let last = items.len() - 1;
            let head = items[..last].join(", ");
            format!("{head}, {conjunction} {}", items[last])
        }
    }
}

// ── RelativeTimeFormat ───────────────────────────────────────────────────────

/// Format a relative time value (en-US, long style).
///
/// `unit` must be one of `"second"`, `"minute"`, `"hour"`, `"day"`, `"week"`,
/// `"month"`, `"quarter"`, or `"year"`.
pub fn relative_time_format(value: f64, unit: &str) -> String {
    let abs = value.abs();
    let plural = if abs == 1.0 {
        unit
    } else {
        &format!("{unit}s")
    };
    if value < 0.0 {
        format!("{abs} {plural} ago")
    } else if value > 0.0 {
        format!("in {abs} {plural}")
    } else {
        format!("in 0 {plural}")
    }
}

// ── Segmenter ────────────────────────────────────────────────────────────────

/// Segment a string into grapheme clusters (stub: splits on UTF-8 char boundaries).
pub fn segmenter_segment(input: &str) -> Vec<String> {
    input.chars().map(|c| c.to_string()).collect()
}

// ── DisplayNames ─────────────────────────────────────────────────────────────

/// Return a display name for a code.  Stub returns the code itself.
pub fn display_names_of(code: &str) -> String {
    code.to_string()
}

// ── Locale ───────────────────────────────────────────────────────────────────

/// Parse a BCP-47 locale tag and return the language subtag.
pub fn locale_language(tag: &str) -> String {
    tag.split('-').next().unwrap_or("und").to_string()
}

/// Return the full canonicalised tag (stub: returns input unchanged).
pub fn locale_base_name(tag: &str) -> String {
    tag.to_string()
}

// ── formatToParts helpers ────────────────────────────────────────────────────

/// Create a `{type, value}` part as a `JsValue::PlainObject`.
fn make_part(part_type: &str, value: &str) -> JsValue {
    let mut part = PropertyMap::new();
    part.insert("type".into(), JsValue::String(part_type.to_string().into()));
    part.insert("value".into(), JsValue::String(value.to_string().into()));
    JsValue::PlainObject(Rc::new(RefCell::new(part)))
}

/// Split an already-formatted number string into integer/decimal/fraction parts.
fn integer_fraction_parts(formatted: &str) -> Vec<JsValue> {
    if let Some(dot_pos) = formatted.find('.') {
        vec![
            make_part("integer", &formatted[..dot_pos]),
            make_part("decimal", "."),
            make_part("fraction", &formatted[dot_pos + 1..]),
        ]
    } else {
        vec![make_part("integer", formatted)]
    }
}

// ── JsValue helpers ──────────────────────────────────────────────────────────

/// Convert a numeric `JsValue` to the formatted string produced by
/// `Intl.NumberFormat.prototype.format`.
pub fn number_format_js(args: &[JsValue]) -> StatorResult<JsValue> {
    let n = args.first().unwrap_or(&JsValue::Undefined).to_number()?;
    Ok(JsValue::String(number_format(n).into()))
}

/// Convert a timestamp `JsValue` to the formatted string produced by
/// `Intl.DateTimeFormat.prototype.format`.
pub fn date_time_format_js(args: &[JsValue]) -> StatorResult<JsValue> {
    let ms = args.first().unwrap_or(&JsValue::Undefined).to_number()?;
    Ok(JsValue::String(date_time_format(ms).into()))
}

/// Compare two string `JsValue`s using `Intl.Collator`.
pub fn collator_compare_js(args: &[JsValue]) -> StatorResult<JsValue> {
    let a = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
    let b = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
    Ok(JsValue::Smi(collator_compare(&a, &b)))
}

/// Select the plural category for a number.
pub fn plural_rules_select_js(args: &[JsValue]) -> StatorResult<JsValue> {
    let n = args.first().unwrap_or(&JsValue::Undefined).to_number()?;
    Ok(JsValue::String(plural_rules_select(n).to_string().into()))
}

/// Format a JS array of strings using `Intl.ListFormat`.
pub fn list_format_js(args: &[JsValue], list_type: &str) -> StatorResult<JsValue> {
    let items: Vec<String> = match args.first() {
        Some(JsValue::Array(arr)) => arr
            .borrow()
            .iter()
            .map(|v| v.to_js_string())
            .collect::<StatorResult<Vec<_>>>()?,
        _ => Vec::new(),
    };
    Ok(JsValue::String(list_format(&items, list_type).into()))
}

/// Format a relative time.
pub fn relative_time_format_js(args: &[JsValue]) -> StatorResult<JsValue> {
    let value = args.first().unwrap_or(&JsValue::Undefined).to_number()?;
    let unit = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
    Ok(JsValue::String(relative_time_format(value, &unit).into()))
}

/// `Intl.NumberFormat.prototype.formatToParts` — returns an array of `{type, value}` parts.
pub fn number_format_to_parts_js(args: &[JsValue]) -> StatorResult<JsValue> {
    let n = args.first().unwrap_or(&JsValue::Undefined).to_number()?;
    let parts = if n.is_nan() {
        vec![make_part("nan", "NaN")]
    } else if n.is_infinite() {
        if n.is_sign_positive() {
            vec![make_part("infinity", "∞")]
        } else {
            vec![make_part("minusSign", "-"), make_part("infinity", "∞")]
        }
    } else if n < 0.0 {
        let abs = number_format(n.abs());
        let mut p = vec![make_part("minusSign", "-")];
        p.extend(integer_fraction_parts(&abs));
        p
    } else {
        integer_fraction_parts(&number_format(n))
    };
    Ok(JsValue::new_array(parts))
}

/// `Intl.DateTimeFormat.prototype.formatToParts` — returns an array of `{type, value}` parts.
pub fn date_time_format_to_parts_js(args: &[JsValue]) -> StatorResult<JsValue> {
    let ms = args.first().unwrap_or(&JsValue::Undefined).to_number()?;
    if ms.is_nan() || ms.is_infinite() {
        return Ok(JsValue::new_array(vec![]));
    }
    let secs = (ms / 1000.0).trunc() as i64;
    let (year, month, day, hour, min, sec) = epoch_to_components(secs);
    let parts = vec![
        make_part("month", &format!("{month}")),
        make_part("literal", "/"),
        make_part("day", &format!("{day}")),
        make_part("literal", "/"),
        make_part("year", &format!("{year}")),
        make_part("literal", ", "),
        make_part("hour", &format!("{hour}")),
        make_part("literal", ":"),
        make_part("minute", &format!("{min:02}")),
        make_part("literal", ":"),
        make_part("second", &format!("{sec:02}")),
        make_part("literal", " "),
        make_part("dayPeriod", "AM"),
    ];
    Ok(JsValue::new_array(parts))
}

// ── NumberFormat.formatRange ─────────────────────────────────────────────────

/// Format a numeric range as `"start – end"` (en-US stub).
pub fn number_format_range(start: f64, end: f64) -> String {
    format!("{} – {}", number_format(start), number_format(end))
}

/// `Intl.NumberFormat.prototype.formatRange(start, end)`.
pub fn number_format_range_js(args: &[JsValue]) -> StatorResult<JsValue> {
    let start = args.first().unwrap_or(&JsValue::Undefined).to_number()?;
    let end = args.get(1).unwrap_or(&JsValue::Undefined).to_number()?;
    Ok(JsValue::String(number_format_range(start, end).into()))
}

/// `Intl.NumberFormat.prototype.formatRangeToParts(start, end)`.
pub fn number_format_range_to_parts_js(args: &[JsValue]) -> StatorResult<JsValue> {
    let start = args.first().unwrap_or(&JsValue::Undefined).to_number()?;
    let end = args.get(1).unwrap_or(&JsValue::Undefined).to_number()?;
    let mut parts = vec![make_part("startRange", &number_format(start))];
    parts.push(make_part("shared", " – "));
    parts.push(make_part("endRange", &number_format(end)));
    Ok(JsValue::new_array(parts))
}

// ── PluralRules.selectRange ─────────────────────────────────────────────────

/// Select the plural category for a range (en-US: always `"other"`).
pub fn plural_rules_select_range(start: f64, end: f64) -> &'static str {
    // CLDR: the range plural for en is always based on the end value,
    // but the spec says "other" for ranges in en.
    let _ = start;
    if end == 1.0 { "one" } else { "other" }
}

/// `Intl.PluralRules.prototype.selectRange(start, end)`.
pub fn plural_rules_select_range_js(args: &[JsValue]) -> StatorResult<JsValue> {
    let start = args.first().unwrap_or(&JsValue::Undefined).to_number()?;
    let end = args.get(1).unwrap_or(&JsValue::Undefined).to_number()?;
    Ok(JsValue::String(
        plural_rules_select_range(start, end).to_string().into(),
    ))
}

// ── Locale extended ─────────────────────────────────────────────────────────

/// Extract the region subtag from a BCP-47 tag (e.g. `"en-US"` → `"US"`).
pub fn locale_region(tag: &str) -> String {
    let parts: Vec<&str> = tag.split('-').collect();
    // Region is a 2-letter UPPER subtag or 3-digit subtag.
    for part in &parts[1..] {
        if (part.len() == 2 && part.chars().all(|c| c.is_ascii_uppercase()))
            || (part.len() == 3 && part.chars().all(|c| c.is_ascii_digit()))
        {
            return part.to_string();
        }
    }
    String::new()
}

/// Extract the script subtag (e.g. `"zh-Hans-CN"` → `"Hans"`).
pub fn locale_script(tag: &str) -> String {
    let parts: Vec<&str> = tag.split('-').collect();
    // Script is a 4-letter subtag with leading uppercase.
    for part in &parts[1..] {
        if part.len() == 4 && part.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
            return part.to_string();
        }
    }
    String::new()
}

/// Stub `maximize()` — adds likely subtags (en → en-Latn-US).
pub fn locale_maximize(tag: &str) -> String {
    let lang = locale_language(tag);
    let script = locale_script(tag);
    let region = locale_region(tag);
    let script = if script.is_empty() {
        match lang.as_str() {
            "zh" => "Hans",
            "ja" => "Jpan",
            "ko" => "Kore",
            _ => "Latn",
        }
    } else {
        &script
    };
    let region = if region.is_empty() {
        match lang.as_str() {
            "en" => "US",
            "zh" => "CN",
            "ja" => "JP",
            "ko" => "KR",
            "fr" => "FR",
            "de" => "DE",
            "es" => "ES",
            _ => "001",
        }
    } else {
        &region
    };
    format!("{lang}-{script}-{region}")
}

/// Stub `minimize()` — removes likely subtags.
pub fn locale_minimize(tag: &str) -> String {
    locale_language(tag)
}

// ── ListFormat.formatToParts ────────────────────────────────────────────────

/// `Intl.ListFormat.prototype.formatToParts(list)` — returns `[{type, value}]`.
pub fn list_format_to_parts_js(args: &[JsValue], list_type: &str) -> StatorResult<JsValue> {
    let items: Vec<String> = match args.first() {
        Some(JsValue::Array(arr)) => arr
            .borrow()
            .iter()
            .map(|v| v.to_js_string())
            .collect::<StatorResult<Vec<_>>>()?,
        _ => Vec::new(),
    };
    let conjunction = if list_type == "disjunction" {
        "or"
    } else {
        "and"
    };
    let mut parts: Vec<JsValue> = Vec::new();
    match items.len() {
        0 => {}
        1 => {
            parts.push(make_part("element", &items[0]));
        }
        2 => {
            parts.push(make_part("element", &items[0]));
            parts.push(make_part("literal", &format!(" {conjunction} ")));
            parts.push(make_part("element", &items[1]));
        }
        _ => {
            let last = items.len() - 1;
            for (i, item) in items.iter().enumerate() {
                if i > 0 && i < last {
                    parts.push(make_part("literal", ", "));
                } else if i == last {
                    parts.push(make_part("literal", &format!(", {conjunction} ")));
                }
                parts.push(make_part("element", item));
            }
        }
    }
    Ok(JsValue::new_array(parts))
}

// ── RelativeTimeFormat.formatToParts ────────────────────────────────────────

/// `Intl.RelativeTimeFormat.prototype.formatToParts(value, unit)`.
pub fn relative_time_format_to_parts_js(args: &[JsValue]) -> StatorResult<JsValue> {
    let value = args.first().unwrap_or(&JsValue::Undefined).to_number()?;
    let unit = args.get(1).unwrap_or(&JsValue::Undefined).to_js_string()?;
    let abs = value.abs();
    let plural_unit = if abs == 1.0 {
        unit.to_string()
    } else {
        format!("{unit}s")
    };
    let mut parts: Vec<JsValue> = Vec::new();
    if value < 0.0 {
        parts.push(make_part("integer", &format!("{abs}")));
        parts.push(make_part("literal", " "));
        parts.push(make_part("unit", &plural_unit));
        parts.push(make_part("literal", " ago"));
    } else {
        parts.push(make_part("literal", "in "));
        parts.push(make_part("integer", &format!("{abs}")));
        parts.push(make_part("literal", " "));
        parts.push(make_part("unit", &plural_unit));
    }
    Ok(JsValue::new_array(parts))
}

// ── Segmenter.segment with containing() ─────────────────────────────────────

/// Build a segments object with indexed segment entries and a `containing(idx)`
/// method, conforming to the `Intl.Segmenter.prototype.segment()` contract.
pub fn segmenter_segment_objects(input_str: &str) -> StatorResult<JsValue> {
    let chars: Vec<String> = input_str.chars().map(|c| c.to_string()).collect();
    let input_owned = input_str.to_string();
    let segments: Vec<JsValue> = chars
        .iter()
        .enumerate()
        .map(|(i, ch)| {
            let mut seg = PropertyMap::new();
            seg.insert("segment".into(), JsValue::String(ch.clone().into()));
            seg.insert("index".into(), JsValue::Smi(i as i32));
            seg.insert("input".into(), JsValue::String(input_owned.clone().into()));
            JsValue::PlainObject(Rc::new(RefCell::new(seg)))
        })
        .collect();

    let segments_clone = segments.clone();
    let mut result = PropertyMap::new();
    for (i, seg) in segments.iter().enumerate() {
        result.insert(i.to_string(), seg.clone());
    }
    result.insert("length".into(), JsValue::Smi(segments.len() as i32));

    // containing(index) method
    result.insert(
        "containing".into(),
        JsValue::NativeFunction(Rc::new(move |args: Vec<JsValue>| {
            let idx = args.first().unwrap_or(&JsValue::Undefined).to_number()? as usize;
            Ok(segments_clone
                .get(idx)
                .cloned()
                .unwrap_or(JsValue::Undefined))
        })),
    );

    Ok(JsValue::PlainObject(Rc::new(RefCell::new(result))))
}

// ── DisplayNames with type ──────────────────────────────────────────────────

/// Return a display name for a code, considering the type.
///
/// For `"language"` codes, returns common English names for well-known codes.
/// Falls back to returning the code itself.
pub fn display_names_of_typed(code: &str, dn_type: &str) -> String {
    match dn_type {
        "language" => match code {
            "en" => "English".to_string(),
            "fr" => "French".to_string(),
            "de" => "German".to_string(),
            "es" => "Spanish".to_string(),
            "zh" => "Chinese".to_string(),
            "ja" => "Japanese".to_string(),
            "ko" => "Korean".to_string(),
            _ => code.to_string(),
        },
        "region" => match code {
            "US" => "United States".to_string(),
            "GB" => "United Kingdom".to_string(),
            "FR" => "France".to_string(),
            "DE" => "Germany".to_string(),
            "JP" => "Japan".to_string(),
            "CN" => "China".to_string(),
            _ => code.to_string(),
        },
        "script" => match code {
            "Latn" => "Latin".to_string(),
            "Hans" => "Simplified Han".to_string(),
            "Hant" => "Traditional Han".to_string(),
            "Cyrl" => "Cyrillic".to_string(),
            _ => code.to_string(),
        },
        "currency" => match code {
            "USD" => "US Dollar".to_string(),
            "EUR" => "Euro".to_string(),
            "GBP" => "British Pound".to_string(),
            "JPY" => "Japanese Yen".to_string(),
            _ => code.to_string(),
        },
        _ => code.to_string(),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_number_format_integer() {
        assert_eq!(number_format(42.0), "42");
    }

    #[test]
    fn test_number_format_decimal() {
        assert_eq!(number_format(3.14), "3.14");
    }

    #[test]
    fn test_number_format_nan() {
        assert_eq!(number_format(f64::NAN), "NaN");
    }

    #[test]
    fn test_number_format_infinity() {
        assert_eq!(number_format(f64::INFINITY), "∞");
        assert_eq!(number_format(f64::NEG_INFINITY), "-∞");
    }

    #[test]
    fn test_date_time_format_epoch_zero() {
        let s = date_time_format(0.0);
        assert!(s.contains("1970"), "expected 1970 in '{s}'");
    }

    #[test]
    fn test_date_time_format_invalid() {
        assert_eq!(date_time_format(f64::NAN), "Invalid Date");
    }

    #[test]
    fn test_collator_compare_equal() {
        assert_eq!(collator_compare("abc", "abc"), 0);
    }

    #[test]
    fn test_collator_compare_less() {
        assert_eq!(collator_compare("abc", "def"), -1);
    }

    #[test]
    fn test_collator_compare_greater() {
        assert_eq!(collator_compare("def", "abc"), 1);
    }

    #[test]
    fn test_plural_rules_one() {
        assert_eq!(plural_rules_select(1.0), "one");
    }

    #[test]
    fn test_plural_rules_other() {
        assert_eq!(plural_rules_select(0.0), "other");
        assert_eq!(plural_rules_select(2.0), "other");
    }

    #[test]
    fn test_list_format_empty() {
        assert_eq!(list_format(&[], "conjunction"), "");
    }

    #[test]
    fn test_list_format_single() {
        assert_eq!(list_format(&["one".to_string()], "conjunction"), "one");
    }

    #[test]
    fn test_list_format_two() {
        let items = vec!["A".to_string(), "B".to_string()];
        assert_eq!(list_format(&items, "conjunction"), "A and B");
        assert_eq!(list_format(&items, "disjunction"), "A or B");
    }

    #[test]
    fn test_list_format_three() {
        let items = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        assert_eq!(list_format(&items, "conjunction"), "A, B, and C");
    }

    #[test]
    fn test_relative_time_format_past() {
        assert_eq!(relative_time_format(-3.0, "day"), "3 days ago");
    }

    #[test]
    fn test_relative_time_format_future() {
        assert_eq!(relative_time_format(1.0, "hour"), "in 1 hour");
    }

    #[test]
    fn test_relative_time_format_zero() {
        assert_eq!(relative_time_format(0.0, "second"), "in 0 seconds");
    }

    #[test]
    fn test_segmenter_segment_ascii() {
        let segs = segmenter_segment("abc");
        assert_eq!(segs, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_display_names_of_passthrough() {
        assert_eq!(display_names_of("US"), "US");
    }

    #[test]
    fn test_locale_language() {
        assert_eq!(locale_language("en-US"), "en");
        assert_eq!(locale_language("fr"), "fr");
    }

    #[test]
    fn test_locale_base_name() {
        assert_eq!(locale_base_name("en-US"), "en-US");
    }

    #[test]
    fn test_number_format_js_value() {
        let result = number_format_js(&[JsValue::HeapNumber(42.5)]).unwrap();
        assert_eq!(result, JsValue::String("42.5".into()));
    }

    #[test]
    fn test_collator_compare_js_value() {
        let result =
            collator_compare_js(&[JsValue::String("a".into()), JsValue::String("b".into())])
                .unwrap();
        assert_eq!(result, JsValue::Smi(-1));
    }

    #[test]
    fn test_plural_rules_select_js_value() {
        let result = plural_rules_select_js(&[JsValue::Smi(1)]).unwrap();
        assert_eq!(result, JsValue::String("one".into()));
    }

    #[test]
    fn test_number_format_range() {
        assert_eq!(number_format_range(1.0, 10.0), "1 – 10");
    }

    #[test]
    fn test_number_format_range_decimals() {
        assert_eq!(number_format_range(1.5, 2.5), "1.5 – 2.5");
    }

    #[test]
    fn test_plural_rules_select_range_other() {
        assert_eq!(plural_rules_select_range(1.0, 5.0), "other");
    }

    #[test]
    fn test_plural_rules_select_range_one() {
        assert_eq!(plural_rules_select_range(0.0, 1.0), "one");
    }

    #[test]
    fn test_locale_region_us() {
        assert_eq!(locale_region("en-US"), "US");
    }

    #[test]
    fn test_locale_region_empty() {
        assert_eq!(locale_region("en"), "");
    }

    #[test]
    fn test_locale_script_hans() {
        assert_eq!(locale_script("zh-Hans-CN"), "Hans");
    }

    #[test]
    fn test_locale_script_empty() {
        assert_eq!(locale_script("en-US"), "");
    }

    #[test]
    fn test_locale_maximize_en() {
        assert_eq!(locale_maximize("en"), "en-Latn-US");
    }

    #[test]
    fn test_locale_minimize_en_us() {
        assert_eq!(locale_minimize("en-US"), "en");
    }

    #[test]
    fn test_display_names_of_typed_language() {
        assert_eq!(display_names_of_typed("en", "language"), "English");
    }

    #[test]
    fn test_display_names_of_typed_region() {
        assert_eq!(display_names_of_typed("US", "region"), "United States");
    }

    #[test]
    fn test_display_names_of_typed_script() {
        assert_eq!(display_names_of_typed("Latn", "script"), "Latin");
    }

    #[test]
    fn test_display_names_of_typed_currency() {
        assert_eq!(display_names_of_typed("USD", "currency"), "US Dollar");
    }
}
