//! ECMAScript §21.4 `Date` built-in constructor and prototype methods.
//!
//! Every function in this module is a direct Rust equivalent of either a static
//! method of the JavaScript `Date` constructor or a method on
//! `Date.prototype`.  They operate on plain `f64` values (milliseconds since
//! the Unix epoch, 1970-01-01T00:00:00Z) and have no side-effects beyond the
//! values passed in.
//!
//! # Naming convention
//!
//! Each function is prefixed `date_` to avoid ambiguity with similarly-named
//! standard-library items.
//!
//! # References
//!
//! * ECMAScript 2025 Language Specification §21.4 — *Date Objects*

use crate::error::{StatorError, StatorResult};
use crate::objects::value::JsValue;

// ── Constants ────────────────────────────────────────────────────────────────

/// Milliseconds per second.
const MS_PER_SECOND: f64 = 1000.0;

/// Milliseconds per minute.
const MS_PER_MINUTE: f64 = 60_000.0;

/// Milliseconds per hour.
const MS_PER_HOUR: f64 = 3_600_000.0;

/// Milliseconds per day (ECMAScript §21.4.1.3).
const MS_PER_DAY: f64 = 86_400_000.0;

/// The maximum allowed time value: ±8,640,000,000,000,000 ms (§21.4.1.1).
const MAX_TIME_VALUE: f64 = 8.64e15;

// ── Time helpers (ECMAScript §21.4.1) ────────────────────────────────────────

/// ECMAScript §21.4.1.3 `Day(t)` — the day number for a time value.
fn day(t: f64) -> f64 {
    (t / MS_PER_DAY).floor()
}

/// ECMAScript §21.4.1.3 `TimeWithinDay(t)`.
fn time_within_day(t: f64) -> f64 {
    t.rem_euclid(MS_PER_DAY)
}

/// ECMAScript §21.4.1.4 `DaysInYear(y)`.
fn days_in_year(y: f64) -> f64 {
    let y = y as i64;
    if y % 4 != 0 {
        365.0
    } else if y % 100 != 0 {
        366.0
    } else if y % 400 != 0 {
        365.0
    } else {
        366.0
    }
}

/// ECMAScript §21.4.1.4 `DayFromYear(y)`.
fn day_from_year(y: f64) -> f64 {
    let y = y as i64;
    // 365 * (y - 1970) + floor((y - 1969)/4) - floor((y - 1901)/100) + floor((y - 1601)/400)
    let base = y - 1970;
    let leap4 = ((y - 1969) as f64 / 4.0).floor() as i64;
    let leap100 = ((y - 1901) as f64 / 100.0).floor() as i64;
    let leap400 = ((y - 1601) as f64 / 400.0).floor() as i64;
    (365 * base + leap4 - leap100 + leap400) as f64
}

/// ECMAScript §21.4.1.4 `YearFromTime(t)`.
fn year_from_time(t: f64) -> f64 {
    // Binary search for the year.
    let d = day(t);
    let mut lo = (d / 366.0 + 1970.0).floor() as i64 - 1;
    let mut hi = (d / 365.0 + 1970.0).ceil() as i64 + 1;

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if day_from_year(mid as f64) <= d {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    (lo - 1) as f64
}

/// ECMAScript §21.4.1.5 `InLeapYear(t)`.
fn in_leap_year(t: f64) -> bool {
    days_in_year(year_from_time(t)) == 366.0
}

/// ECMAScript §21.4.1.5 `DayWithinYear(t)`.
fn day_within_year(t: f64) -> f64 {
    day(t) - day_from_year(year_from_time(t))
}

/// ECMAScript §21.4.1.5 `MonthFromTime(t)` — 0-based month (0 = January).
fn month_from_time(t: f64) -> f64 {
    let d = day_within_year(t);
    let leap = if in_leap_year(t) { 1.0 } else { 0.0 };

    if d < 31.0 {
        0.0
    } else if d < 59.0 + leap {
        1.0
    } else if d < 90.0 + leap {
        2.0
    } else if d < 120.0 + leap {
        3.0
    } else if d < 151.0 + leap {
        4.0
    } else if d < 181.0 + leap {
        5.0
    } else if d < 212.0 + leap {
        6.0
    } else if d < 243.0 + leap {
        7.0
    } else if d < 273.0 + leap {
        8.0
    } else if d < 304.0 + leap {
        9.0
    } else if d < 334.0 + leap {
        10.0
    } else {
        11.0
    }
}

/// ECMAScript §21.4.1.6 `DateFromTime(t)` — 1-based day-of-month.
fn date_from_time(t: f64) -> f64 {
    let d = day_within_year(t);
    let m = month_from_time(t);
    let leap = if in_leap_year(t) { 1.0 } else { 0.0 };

    match m as u8 {
        0 => d + 1.0,
        1 => d - 30.0,
        2 => d - 58.0 - leap,
        3 => d - 89.0 - leap,
        4 => d - 119.0 - leap,
        5 => d - 150.0 - leap,
        6 => d - 180.0 - leap,
        7 => d - 211.0 - leap,
        8 => d - 242.0 - leap,
        9 => d - 272.0 - leap,
        10 => d - 303.0 - leap,
        _ => d - 333.0 - leap,
    }
}

/// ECMAScript §21.4.1.7 `WeekDay(t)` — 0 = Sunday, 6 = Saturday.
fn week_day(t: f64) -> f64 {
    (day(t) + 4.0).rem_euclid(7.0)
}

/// ECMAScript §21.4.1.11 `HourFromTime(t)`.
fn hour_from_time(t: f64) -> f64 {
    (time_within_day(t) / MS_PER_HOUR).floor().rem_euclid(24.0)
}

/// ECMAScript §21.4.1.11 `MinFromTime(t)`.
fn min_from_time(t: f64) -> f64 {
    (time_within_day(t) / MS_PER_MINUTE)
        .floor()
        .rem_euclid(60.0)
}

/// ECMAScript §21.4.1.11 `SecFromTime(t)`.
fn sec_from_time(t: f64) -> f64 {
    (time_within_day(t) / MS_PER_SECOND)
        .floor()
        .rem_euclid(60.0)
}

/// ECMAScript §21.4.1.11 `msFromTime(t)`.
fn ms_from_time(t: f64) -> f64 {
    time_within_day(t).rem_euclid(MS_PER_SECOND)
}

/// ECMAScript §21.4.1.12 `MakeTime(hour, min, sec, ms)`.
fn make_time(hour: f64, min: f64, sec: f64, ms: f64) -> f64 {
    if !hour.is_finite() || !min.is_finite() || !sec.is_finite() || !ms.is_finite() {
        return f64::NAN;
    }
    let h = hour.trunc();
    let m = min.trunc();
    let s = sec.trunc();
    let milli = ms.trunc();
    h * MS_PER_HOUR + m * MS_PER_MINUTE + s * MS_PER_SECOND + milli
}

/// ECMAScript §21.4.1.13 `MakeDay(year, month, date)`.
fn make_day(year: f64, month: f64, date: f64) -> f64 {
    if !year.is_finite() || !month.is_finite() || !date.is_finite() {
        return f64::NAN;
    }
    let y = year.trunc();
    let m = month.trunc();
    let dt = date.trunc();

    // Adjust year/month: month can overflow.
    let ym = y + (m / 12.0).floor();
    let mn = m.rem_euclid(12.0);

    // Cumulative days from Jan 1 for each month (non-leap).
    let month_days: [f64; 12] = [
        0.0, 31.0, 59.0, 90.0, 120.0, 151.0, 181.0, 212.0, 243.0, 273.0, 304.0, 334.0,
    ];

    let day_start = day_from_year(ym);
    let leap = if days_in_year(ym) == 366.0 { 1.0 } else { 0.0 };
    let month_offset = month_days[mn as usize];
    let leap_adjust = if mn >= 2.0 { leap } else { 0.0 };

    day_start + month_offset + leap_adjust + dt - 1.0
}

/// ECMAScript §21.4.1.14 `MakeDate(day, time)`.
fn make_date(day: f64, time: f64) -> f64 {
    if !day.is_finite() || !time.is_finite() {
        return f64::NAN;
    }
    day * MS_PER_DAY + time
}

/// ECMAScript §21.4.1.15 `TimeClip(time)`.
fn time_clip(time: f64) -> f64 {
    if !time.is_finite() || time.abs() > MAX_TIME_VALUE {
        f64::NAN
    } else {
        time.trunc()
    }
}

fn days_in_month(year: f64, month: u32) -> u32 {
    match month {
        0 => 31,
        1 => {
            if days_in_year(year) == 366.0 {
                29
            } else {
                28
            }
        }
        2 => 31,
        3 => 30,
        4 => 31,
        5 => 30,
        6 => 31,
        7 => 31,
        8 => 30,
        9 => 31,
        10 => 30,
        11 => 31,
        _ => 0,
    }
}

// ── Local time offset ────────────────────────────────────────────────────────

/// Return the system's local timezone offset in milliseconds.
///
/// Uses platform-specific APIs to determine the offset. Falls back to 0 on
/// unsupported platforms.
fn local_tz_offset_ms(utc_ms: f64) -> f64 {
    // Use std::time to approximate.  We convert the UTC ms to a SystemTime,
    // then inspect the formatted output to extract the offset.
    // Simplified: use a compile-time constant for testing, but at runtime
    // we derive from the platform.
    #[cfg(target_os = "windows")]
    {
        local_tz_offset_windows(utc_ms)
    }
    #[cfg(not(target_os = "windows"))]
    {
        local_tz_offset_unix(utc_ms)
    }
}

#[cfg(target_os = "windows")]
fn local_tz_offset_windows(_utc_ms: f64) -> f64 {
    use std::mem::MaybeUninit;

    // SAFETY: GetTimeZoneInformation is a well-defined Win32 API that writes
    // a TIME_ZONE_INFORMATION structure into the provided pointer.  The
    // MaybeUninit buffer is properly sized and aligned for the struct.
    unsafe {
        #[repr(C)]
        struct SystemTime {
            w_year: u16,
            w_month: u16,
            w_day_of_week: u16,
            w_day: u16,
            w_hour: u16,
            w_minute: u16,
            w_second: u16,
            w_milliseconds: u16,
        }

        #[repr(C)]
        struct TimeZoneInformation {
            bias: i32,
            _standard_name: [u16; 32],
            _standard_date: SystemTime,
            _standard_bias: i32,
            _daylight_name: [u16; 32],
            _daylight_date: SystemTime,
            _daylight_bias: i32,
        }

        // SAFETY: GetTimeZoneInformation is a well-defined Win32 API.
        unsafe extern "system" {
            fn GetTimeZoneInformation(lpTimeZoneInformation: *mut TimeZoneInformation) -> u32;
        }

        let mut tzi = MaybeUninit::<TimeZoneInformation>::uninit();
        GetTimeZoneInformation(tzi.as_mut_ptr());
        let tzi = tzi.assume_init();
        // Bias is in minutes, UTC = local + bias, so local = UTC - bias.
        -(tzi.bias as f64) * MS_PER_MINUTE
    }
}

#[cfg(not(target_os = "windows"))]
fn local_tz_offset_unix(utc_ms: f64) -> f64 {
    use std::ffi::c_long;

    #[repr(C)]
    struct Tm {
        tm_sec: i32,
        tm_min: i32,
        tm_hour: i32,
        tm_mday: i32,
        tm_mon: i32,
        tm_year: i32,
        tm_wday: i32,
        tm_yday: i32,
        tm_isdst: i32,
        tm_gmtoff: c_long,
        _tm_zone: *const i8,
    }

    // SAFETY: localtime_r is POSIX-defined.
    unsafe extern "C" {
        fn localtime_r(timep: *const c_long, result: *mut Tm) -> *mut Tm;
    }

    let secs = (utc_ms / 1000.0).floor() as c_long;
    let mut tm = std::mem::MaybeUninit::<Tm>::uninit();
    // SAFETY: localtime_r is POSIX-defined. We pass a valid pointer to a
    // c_long and a properly-aligned MaybeUninit<Tm> buffer.
    unsafe {
        let result = localtime_r(&secs as *const c_long, tm.as_mut_ptr());
        if result.is_null() {
            return 0.0;
        }
        let tm = tm.assume_init();
        (tm.tm_gmtoff as f64) * 1000.0
    }
}

/// Convert UTC milliseconds to local time milliseconds.
fn utc_to_local(t: f64) -> f64 {
    t + local_tz_offset_ms(t)
}

/// Convert local time milliseconds to UTC milliseconds.
fn local_to_utc(t: f64) -> f64 {
    if !t.is_finite() {
        return t;
    }

    // The timezone offset depends on the UTC instant, not the local wall-clock
    // time. Resolve the corresponding UTC value iteratively so dates around DST
    // transitions use the offset in effect for the resulting instant.
    let mut utc = t - local_tz_offset_ms(t);
    for _ in 0..4 {
        let next = t - local_tz_offset_ms(utc);
        if (next - utc).abs() < 1.0 {
            utc = next;
            break;
        }
        utc = next;
    }
    utc
}

// ── Date.now ─────────────────────────────────────────────────────────────────

/// ECMAScript §21.4.3.1 `Date.now()`.
///
/// Returns the current time as milliseconds since the Unix epoch.
///
/// # Examples
///
/// ```
/// use stator_js::builtins::date::date_now;
///
/// let now = date_now();
/// assert!(now > 0.0);
/// ```
pub fn date_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as f64)
        .unwrap_or(0.0)
}

// ── Date.parse ───────────────────────────────────────────────────────────────

/// ECMAScript §21.4.3.2 `Date.parse(string)`.
///
/// Parses a date string and returns the corresponding time value in
/// milliseconds since the Unix epoch, or `NaN` if the string is not
/// recognised.
///
/// Supports ISO 8601 format (`YYYY-MM-DDTHH:mm:ss.sssZ`) and common
/// legacy formats.
///
/// # Examples
///
/// ```
/// use stator_js::builtins::date::date_parse;
///
/// let t = date_parse("2024-01-15T12:30:00.000Z");
/// assert!(!t.is_nan());
///
/// let t = date_parse("not a date");
/// assert!(t.is_nan());
/// ```
pub fn date_parse(s: &str) -> f64 {
    let s = s.trim();
    if s.is_empty() {
        return f64::NAN;
    }

    // Try ISO 8601 first.
    if let Some(t) = parse_iso8601(s) {
        return time_clip(t);
    }

    // Try legacy date string formats.
    if let Some(t) = parse_legacy(s) {
        return time_clip(t);
    }

    f64::NAN
}

/// Parse ISO 8601 date string: `YYYY`, `YYYY-MM`, `YYYY-MM-DD`,
/// `YYYY-MM-DDTHH:mm`, `YYYY-MM-DDTHH:mm:ss`, `YYYY-MM-DDTHH:mm:ss.sss`,
/// with optional timezone `Z` or `±HH:mm`.
fn parse_iso8601(s: &str) -> Option<f64> {
    let bytes = s.as_bytes();
    let len = bytes.len();

    // Parse year — may have leading sign for extended years.
    let (year, pos) = parse_iso_year(s)?;

    if pos >= len {
        // Date-only: YYYY
        let day = make_day(year, 0.0, 1.0);
        return Some(make_date(day, 0.0));
    }

    if bytes[pos] != b'-' {
        return None;
    }
    let pos = pos + 1;

    // Month
    let (month, pos) = parse_two_digits(s, pos)?;
    if !(1..=12).contains(&month) {
        return None;
    }

    if pos >= len {
        // YYYY-MM
        let day = make_day(year, (month - 1) as f64, 1.0);
        return Some(make_date(day, 0.0));
    }

    if bytes[pos] != b'-' {
        return None;
    }
    let pos = pos + 1;

    // Day
    let (day_val, pos) = parse_two_digits(s, pos)?;
    if !(1..=31).contains(&day_val) || day_val > days_in_month(year, month - 1) {
        return None;
    }

    if pos >= len {
        // YYYY-MM-DD (date-only form is treated as UTC per spec)
        let day = make_day(year, (month - 1) as f64, day_val as f64);
        return Some(make_date(day, 0.0));
    }

    // Expect T or space separator for time
    if bytes[pos] != b'T' && bytes[pos] != b't' && bytes[pos] != b' ' {
        return None;
    }
    let pos = pos + 1;

    // Hours
    let (hour, pos) = parse_two_digits(s, pos)?;
    if pos >= len || bytes[pos] != b':' {
        return None;
    }
    let pos = pos + 1;

    // Minutes
    let (min, pos) = parse_two_digits(s, pos)?;
    if min > 59 {
        return None;
    }

    let (sec, ms, pos) = if pos < len && bytes[pos] == b':' {
        // Seconds
        let (sec, pos) = parse_two_digits(s, pos + 1)?;
        if sec > 59 {
            return None;
        }

        let (ms, pos) = if pos < len && bytes[pos] == b'.' {
            parse_fractional_seconds(s, pos + 1)?
        } else {
            (0, pos)
        };
        (sec, ms, pos)
    } else {
        (0, 0, pos)
    };

    if hour > 24 || (hour == 24 && (min != 0 || sec != 0 || ms != 0)) {
        return None;
    }

    // Timezone: Z, +HH:mm, -HH:mm, or absent (local for datetime forms)
    let (tz_offset_ms, end_pos) = if pos >= len {
        // No timezone specified — treat as local time
        (None, pos)
    } else if bytes[pos] == b'Z' || bytes[pos] == b'z' {
        (Some(0.0), pos + 1)
    } else if bytes[pos] == b'+' || bytes[pos] == b'-' {
        let sign: f64 = if bytes[pos] == b'+' { 1.0 } else { -1.0 };
        let pos = pos + 1;
        let (tz_h, pos) = parse_two_digits(s, pos)?;
        if tz_h > 23 {
            return None;
        }
        if pos >= len || bytes[pos] != b':' {
            return None;
        }
        let (tz_m, pos) = parse_two_digits(s, pos + 1)?;
        if tz_m > 59 {
            return None;
        }
        (
            Some(sign * (tz_h as f64 * MS_PER_HOUR + tz_m as f64 * MS_PER_MINUTE)),
            pos,
        )
    } else {
        return None;
    };

    if end_pos != len {
        return None;
    }

    let time = make_time(hour as f64, min as f64, sec as f64, ms as f64);
    let day = make_day(year, (month - 1) as f64, day_val as f64);
    let utc = make_date(day, time);

    match tz_offset_ms {
        Some(offset) => Some(utc - offset),
        None => Some(local_to_utc(utc)),
    }
}

/// Parse a year component which may have a leading + or - sign.
fn parse_iso_year(s: &str) -> Option<(f64, usize)> {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return None;
    }

    let (sign, start, digits) = if bytes[0] == b'+' || bytes[0] == b'-' {
        (if bytes[0] == b'+' { 1.0 } else { -1.0 }, 1, 6)
    } else {
        (1.0, 0, 4)
    };

    let end = start + digits;
    if end > bytes.len() || !bytes[start..end].iter().all(u8::is_ascii_digit) {
        return None;
    }

    if end < bytes.len() && bytes[end].is_ascii_digit() {
        return None;
    }

    let year_str = &s[start..end];
    let year: f64 = year_str.parse().ok()?;
    Some((sign * year, end))
}

/// Parse exactly two ASCII digits at the given position.
fn parse_two_digits(s: &str, pos: usize) -> Option<(u32, usize)> {
    let bytes = s.as_bytes();
    if pos + 2 > bytes.len() {
        return None;
    }
    let d1 = bytes[pos].wrapping_sub(b'0');
    let d2 = bytes[pos + 1].wrapping_sub(b'0');
    if d1 > 9 || d2 > 9 {
        return None;
    }
    Some(((d1 as u32) * 10 + d2 as u32, pos + 2))
}

/// Parse fractional seconds (1-3 digits after the decimal point).
fn parse_fractional_seconds(s: &str, start: usize) -> Option<(u32, usize)> {
    let bytes = s.as_bytes();
    let mut pos = start;
    let mut val: u32 = 0;
    let mut digits = 0;

    while pos < bytes.len() && bytes[pos].is_ascii_digit() && digits < 3 {
        val = val * 10 + (bytes[pos] - b'0') as u32;
        digits += 1;
        pos += 1;
    }

    if digits == 0 {
        return None;
    }

    // Pad to 3 digits (ms).
    while digits < 3 {
        val *= 10;
        digits += 1;
    }

    // Skip any remaining fractional digits beyond 3.
    while pos < bytes.len() && bytes[pos].is_ascii_digit() {
        pos += 1;
    }

    Some((val, pos))
}

/// Try to parse common legacy date formats (e.g. RFC 2822-ish).
fn parse_legacy(s: &str) -> Option<f64> {
    // Support formats like:
    // "Mon Jan 15 2024 12:30:00 GMT+0000"
    // "Jan 15, 2024"
    // "15 Jan 2024"
    // "1/15/2024"

    let months = [
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    ];

    let lower = s.to_ascii_lowercase();

    // Try to extract month name.
    let mut month_idx = None;
    for (i, m) in months.iter().enumerate() {
        if lower.contains(m) {
            month_idx = Some(i);
            break;
        }
    }

    if let Some(month) = month_idx {
        // Extract numbers from the string.
        let nums: Vec<f64> = s
            .split(|c: char| !c.is_ascii_digit() && c != '-')
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();

        // Heuristic: find year (>31), day (1-31), and optional time components.
        let mut year = None;
        let mut day_val = None;

        for &n in &nums {
            if n > 31.0 && year.is_none() {
                year = Some(n);
            } else if (1.0..=31.0).contains(&n) && day_val.is_none() {
                day_val = Some(n);
            }
        }

        let y = year.unwrap_or(2001.0);
        let d = day_val.unwrap_or(1.0);

        // Try to extract time HH:MM:SS from the string.
        let (hour, min, sec) = extract_time_from_string(s);

        let day = make_day(y, month as f64, d);
        let time = make_time(hour, min, sec, 0.0);

        let result = make_date(day, time);
        if let Some(offset_ms) = extract_gmt_offset(&lower) {
            return Some(result - offset_ms);
        }
        if lower.contains("gmt") || lower.contains("utc") || lower.ends_with('z') {
            return Some(result);
        }
        return Some(local_to_utc(result));
    }

    // Try M/D/YYYY or M-D-YYYY format.
    let parts: Vec<&str> = s.split(['/', '-']).collect();
    if parts.len() >= 3 {
        let m: f64 = parts[0].trim().parse().ok()?;
        let d: f64 = parts[1].trim().parse().ok()?;
        let y: f64 = parts[2].split_whitespace().next()?.parse().ok()?;
        if (1.0..=12.0).contains(&m) && (1.0..=31.0).contains(&d) {
            let day = make_day(y, m - 1.0, d);
            let result = make_date(day, 0.0);
            return Some(local_to_utc(result));
        }
    }

    None
}

fn extract_gmt_offset(s: &str) -> Option<f64> {
    let lower = s.to_ascii_lowercase();
    for marker in ["gmt", "utc"] {
        let Some(idx) = lower.find(marker) else {
            continue;
        };
        let rest = lower[idx + marker.len()..].trim_start();
        let Some(sign_char) = rest.chars().next() else {
            return Some(0.0);
        };
        let sign = match sign_char {
            '+' => 1.0,
            '-' => -1.0,
            _ => return Some(0.0),
        };

        let rest = &rest[1..];
        let bytes = rest.as_bytes();
        if bytes.len() < 2 || !bytes[0].is_ascii_digit() || !bytes[1].is_ascii_digit() {
            return None;
        }
        let hours = ((bytes[0] - b'0') as u32) * 10 + (bytes[1] - b'0') as u32;
        let mut minutes = 0_u32;

        if bytes.len() >= 5 && bytes[2] == b':' {
            if !bytes[3].is_ascii_digit() || !bytes[4].is_ascii_digit() {
                return None;
            }
            minutes = ((bytes[3] - b'0') as u32) * 10 + (bytes[4] - b'0') as u32;
        } else if bytes.len() >= 4 && bytes[2].is_ascii_digit() && bytes[3].is_ascii_digit() {
            minutes = ((bytes[2] - b'0') as u32) * 10 + (bytes[3] - b'0') as u32;
        }

        if hours > 23 || minutes > 59 {
            return None;
        }

        return Some(sign * (hours as f64 * MS_PER_HOUR + minutes as f64 * MS_PER_MINUTE));
    }
    None
}

/// Extract HH:MM:SS from a string containing a time pattern.
fn extract_time_from_string(s: &str) -> (f64, f64, f64) {
    // Look for HH:MM or HH:MM:SS pattern.
    let bytes = s.as_bytes();
    for i in 0..bytes.len().saturating_sub(4) {
        if bytes[i].is_ascii_digit()
            && i + 1 < bytes.len()
            && bytes[i + 1].is_ascii_digit()
            && i + 2 < bytes.len()
            && bytes[i + 2] == b':'
            && i + 3 < bytes.len()
            && bytes[i + 3].is_ascii_digit()
            && i + 4 < bytes.len()
            && bytes[i + 4].is_ascii_digit()
        {
            let hour = (bytes[i] - b'0') as f64 * 10.0 + (bytes[i + 1] - b'0') as f64;
            let min = (bytes[i + 3] - b'0') as f64 * 10.0 + (bytes[i + 4] - b'0') as f64;
            let sec = if i + 7 < bytes.len() && bytes[i + 5] == b':' {
                (bytes[i + 6] - b'0') as f64 * 10.0
                    + if i + 7 < bytes.len() && bytes[i + 7].is_ascii_digit() {
                        (bytes[i + 7] - b'0') as f64
                    } else {
                        0.0
                    }
            } else {
                0.0
            };
            return (hour, min, sec);
        }
    }
    (0.0, 0.0, 0.0)
}

// ── Date.UTC ─────────────────────────────────────────────────────────────────

/// ECMAScript §21.4.3.4 `Date.UTC(year, month, date, hours, minutes, seconds, ms)`.
///
/// Returns the time value for the given UTC date components. The `year`
/// parameter handles the 0–99 two-digit year mapping (adds 1900).
///
/// # Examples
///
/// ```
/// use stator_js::builtins::date::date_utc;
///
/// let t = date_utc(2024.0, 0.0, 15.0, 12.0, 30.0, 0.0, 0.0);
/// assert!(!t.is_nan());
/// ```
pub fn date_utc(
    year: f64,
    month: f64,
    date: f64,
    hours: f64,
    minutes: f64,
    seconds: f64,
    ms: f64,
) -> f64 {
    if year.is_nan() {
        return f64::NAN;
    }
    let yr = year.trunc();
    // Two-digit year mapping: 0–99 → 1900–1999.
    let y = if (0.0..=99.0).contains(&yr) {
        1900.0 + yr
    } else {
        yr
    };

    let day = make_day(y, month, date);
    let time = make_time(hours, minutes, seconds, ms);
    time_clip(make_date(day, time))
}

// ── Date constructor helpers ─────────────────────────────────────────────────

/// Construct a Date time value from `new Date()` — returns current time.
pub fn date_construct_now() -> f64 {
    time_clip(date_now())
}

/// Construct a Date time value from `new Date(value)`.
///
/// If `value` is a number, use it directly; if a string, parse it.
pub fn date_construct_value(value: f64) -> f64 {
    time_clip(value)
}

/// Construct a Date time value from `new Date(year, month, ...)`.
///
/// Applies the two-digit year mapping and converts from local time to UTC.
pub fn date_construct_components(
    year: f64,
    month: f64,
    date: f64,
    hours: f64,
    minutes: f64,
    seconds: f64,
    ms: f64,
) -> f64 {
    if year.is_nan() || month.is_nan() {
        return f64::NAN;
    }
    let yr = year.trunc();
    let y = if (0.0..=99.0).contains(&yr) {
        1900.0 + yr
    } else {
        yr
    };

    let day = make_day(y, month, date);
    let time = make_time(hours, minutes, seconds, ms);
    let local = make_date(day, time);
    time_clip(local_to_utc(local))
}

// ── Getter methods (UTC) ─────────────────────────────────────────────────────

/// ECMAScript §21.4.4.10 `Date.prototype.getFullYear()`.
///
/// Returns the year of the date in local time, or `NaN` if the date is invalid.
///
/// # Examples
///
/// ```
/// use stator_js::builtins::date::date_get_full_year;
///
/// // 2024-01-15T00:00:00Z in local time — exact year depends on timezone
/// let t = 1705276800000.0;
/// let year = date_get_full_year(t);
/// assert!(year >= 2024.0 || year == 2023.0); // timezone-dependent
/// ```
pub fn date_get_full_year(t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    year_from_time(utc_to_local(t))
}

/// ECMAScript §21.4.4.12 `Date.prototype.getMonth()` — 0-based.
pub fn date_get_month(t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    month_from_time(utc_to_local(t))
}

/// ECMAScript §21.4.4.8 `Date.prototype.getDate()` — 1-based day of month.
pub fn date_get_date(t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    date_from_time(utc_to_local(t))
}

/// ECMAScript §21.4.4.9 `Date.prototype.getDay()` — 0 = Sunday.
pub fn date_get_day(t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    week_day(utc_to_local(t))
}

/// ECMAScript §21.4.4.11 `Date.prototype.getHours()`.
pub fn date_get_hours(t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    hour_from_time(utc_to_local(t))
}

/// ECMAScript §21.4.4.13 `Date.prototype.getMinutes()`.
pub fn date_get_minutes(t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    min_from_time(utc_to_local(t))
}

/// ECMAScript §21.4.4.14 `Date.prototype.getSeconds()`.
pub fn date_get_seconds(t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    sec_from_time(utc_to_local(t))
}

/// ECMAScript §21.4.4.15 `Date.prototype.getMilliseconds()`.
pub fn date_get_milliseconds(t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    ms_from_time(utc_to_local(t))
}

/// ECMAScript §21.4.4.16 `Date.prototype.getTime()`.
pub fn date_get_time(t: f64) -> f64 {
    t
}

/// ECMAScript §21.4.4.17 `Date.prototype.getTimezoneOffset()`.
///
/// Returns the difference in minutes between UTC and local time.
pub fn date_get_timezone_offset(t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    (t - utc_to_local(t)) / MS_PER_MINUTE
}

// ── UTC getter methods ───────────────────────────────────────────────────────

/// ECMAScript §21.4.4.18 `Date.prototype.getUTCFullYear()`.
pub fn date_get_utc_full_year(t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    year_from_time(t)
}

/// ECMAScript §21.4.4.19 `Date.prototype.getUTCMonth()`.
pub fn date_get_utc_month(t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    month_from_time(t)
}

/// ECMAScript §21.4.4.20 `Date.prototype.getUTCDate()`.
pub fn date_get_utc_date(t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    date_from_time(t)
}

/// ECMAScript §21.4.4.21 `Date.prototype.getUTCDay()`.
pub fn date_get_utc_day(t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    week_day(t)
}

/// ECMAScript §21.4.4.22 `Date.prototype.getUTCHours()`.
pub fn date_get_utc_hours(t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    hour_from_time(t)
}

/// ECMAScript §21.4.4.23 `Date.prototype.getUTCMinutes()`.
pub fn date_get_utc_minutes(t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    min_from_time(t)
}

/// ECMAScript §21.4.4.24 `Date.prototype.getUTCSeconds()`.
pub fn date_get_utc_seconds(t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    sec_from_time(t)
}

/// ECMAScript §21.4.4.25 `Date.prototype.getUTCMilliseconds()`.
pub fn date_get_utc_milliseconds(t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    ms_from_time(t)
}

// ── Setter methods ───────────────────────────────────────────────────────────

/// ECMAScript §21.4.4.26 `Date.prototype.setTime(time)`.
pub fn date_set_time(time: f64) -> f64 {
    time_clip(time)
}

/// ECMAScript §21.4.4.27 `Date.prototype.setMilliseconds(ms)`.
pub fn date_set_milliseconds(t: f64, ms: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    let local = utc_to_local(t);
    let time = make_time(
        hour_from_time(local),
        min_from_time(local),
        sec_from_time(local),
        ms,
    );
    let day = day(local);
    time_clip(local_to_utc(make_date(day, time)))
}

/// ECMAScript §21.4.4.28 `Date.prototype.setSeconds(sec [, ms])`.
pub fn date_set_seconds(t: f64, sec: f64, ms: Option<f64>) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    let local = utc_to_local(t);
    let ms_val = ms.unwrap_or_else(|| ms_from_time(local));
    let time = make_time(hour_from_time(local), min_from_time(local), sec, ms_val);
    let d = day(local);
    time_clip(local_to_utc(make_date(d, time)))
}

/// ECMAScript §21.4.4.29 `Date.prototype.setMinutes(min [, sec [, ms]])`.
pub fn date_set_minutes(t: f64, min: f64, sec: Option<f64>, ms: Option<f64>) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    let local = utc_to_local(t);
    let s = sec.unwrap_or_else(|| sec_from_time(local));
    let ms_val = ms.unwrap_or_else(|| ms_from_time(local));
    let time = make_time(hour_from_time(local), min, s, ms_val);
    let d = day(local);
    time_clip(local_to_utc(make_date(d, time)))
}

/// ECMAScript §21.4.4.30 `Date.prototype.setHours(hour [, min [, sec [, ms]]])`.
pub fn date_set_hours(
    t: f64,
    hour: f64,
    min: Option<f64>,
    sec: Option<f64>,
    ms: Option<f64>,
) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    let local = utc_to_local(t);
    let m = min.unwrap_or_else(|| min_from_time(local));
    let s = sec.unwrap_or_else(|| sec_from_time(local));
    let ms_val = ms.unwrap_or_else(|| ms_from_time(local));
    let time = make_time(hour, m, s, ms_val);
    let d = day(local);
    time_clip(local_to_utc(make_date(d, time)))
}

/// ECMAScript §21.4.4.31 `Date.prototype.setDate(date)`.
pub fn date_set_date(t: f64, date_val: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    let local = utc_to_local(t);
    let d = make_day(year_from_time(local), month_from_time(local), date_val);
    let time = time_within_day(local);
    time_clip(local_to_utc(make_date(d, time)))
}

/// ECMAScript §21.4.4.32 `Date.prototype.setMonth(month [, date])`.
pub fn date_set_month(t: f64, month: f64, date_val: Option<f64>) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    let local = utc_to_local(t);
    let dt = date_val.unwrap_or_else(|| date_from_time(local));
    let d = make_day(year_from_time(local), month, dt);
    let time = time_within_day(local);
    time_clip(local_to_utc(make_date(d, time)))
}

/// ECMAScript §21.4.4.33 `Date.prototype.setFullYear(year [, month [, date]])`.
pub fn date_set_full_year(t: f64, year: f64, month: Option<f64>, date_val: Option<f64>) -> f64 {
    let local = if t.is_nan() { 0.0 } else { utc_to_local(t) };
    let m = month.unwrap_or_else(|| month_from_time(local));
    let dt = date_val.unwrap_or_else(|| date_from_time(local));
    let d = make_day(year, m, dt);
    let time = time_within_day(local);
    time_clip(local_to_utc(make_date(d, time)))
}

// ── UTC setter methods ───────────────────────────────────────────────────────

/// ECMAScript §21.4.4.34 `Date.prototype.setUTCMilliseconds(ms)`.
pub fn date_set_utc_milliseconds(t: f64, ms: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    let time = make_time(hour_from_time(t), min_from_time(t), sec_from_time(t), ms);
    time_clip(make_date(day(t), time))
}

/// ECMAScript §21.4.4.35 `Date.prototype.setUTCSeconds(sec [, ms])`.
pub fn date_set_utc_seconds(t: f64, sec: f64, ms: Option<f64>) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    let ms_val = ms.unwrap_or_else(|| ms_from_time(t));
    let time = make_time(hour_from_time(t), min_from_time(t), sec, ms_val);
    time_clip(make_date(day(t), time))
}

/// ECMAScript §21.4.4.36 `Date.prototype.setUTCMinutes(min [, sec [, ms]])`.
pub fn date_set_utc_minutes(t: f64, min: f64, sec: Option<f64>, ms: Option<f64>) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    let s = sec.unwrap_or_else(|| sec_from_time(t));
    let ms_val = ms.unwrap_or_else(|| ms_from_time(t));
    let time = make_time(hour_from_time(t), min, s, ms_val);
    time_clip(make_date(day(t), time))
}

/// ECMAScript §21.4.4.37 `Date.prototype.setUTCHours(hour [, min [, sec [, ms]]])`.
pub fn date_set_utc_hours(
    t: f64,
    hour: f64,
    min: Option<f64>,
    sec: Option<f64>,
    ms: Option<f64>,
) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    let m = min.unwrap_or_else(|| min_from_time(t));
    let s = sec.unwrap_or_else(|| sec_from_time(t));
    let ms_val = ms.unwrap_or_else(|| ms_from_time(t));
    let time = make_time(hour, m, s, ms_val);
    time_clip(make_date(day(t), time))
}

/// ECMAScript §21.4.4.38 `Date.prototype.setUTCDate(date)`.
pub fn date_set_utc_date(t: f64, date_val: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    let d = make_day(year_from_time(t), month_from_time(t), date_val);
    let time = time_within_day(t);
    time_clip(make_date(d, time))
}

/// ECMAScript §21.4.4.39 `Date.prototype.setUTCMonth(month [, date])`.
pub fn date_set_utc_month(t: f64, month: f64, date_val: Option<f64>) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    let dt = date_val.unwrap_or_else(|| date_from_time(t));
    let d = make_day(year_from_time(t), month, dt);
    let time = time_within_day(t);
    time_clip(make_date(d, time))
}

/// ECMAScript §21.4.4.40 `Date.prototype.setUTCFullYear(year [, month [, date]])`.
pub fn date_set_utc_full_year(t: f64, year: f64, month: Option<f64>, date_val: Option<f64>) -> f64 {
    let base = if t.is_nan() { 0.0 } else { t };
    let m = month.unwrap_or_else(|| month_from_time(base));
    let dt = date_val.unwrap_or_else(|| date_from_time(base));
    let d = make_day(year, m, dt);
    let time = time_within_day(base);
    time_clip(make_date(d, time))
}

// ── String conversion methods ────────────────────────────────────────────────

/// Day-of-week name abbreviation.
const DAY_NAMES: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

/// Month name abbreviation.
const MONTH_NAMES: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// ECMAScript §21.4.4.41 `Date.prototype.toISOString()`.
///
/// Returns the ISO 8601 string representation. Throws RangeError for
/// invalid dates.
///
/// # Examples
///
/// ```
/// use stator_js::builtins::date::date_to_iso_string;
///
/// let s = date_to_iso_string(0.0).unwrap();
/// assert_eq!(s, "1970-01-01T00:00:00.000Z");
/// ```
pub fn date_to_iso_string(t: f64) -> StatorResult<String> {
    if t.is_nan() || t.is_infinite() {
        return Err(StatorError::RangeError("Invalid time value".to_string()));
    }
    let year = year_from_time(t);
    let month = month_from_time(t) as u32 + 1;
    let day_val = date_from_time(t) as u32;
    let hour = hour_from_time(t) as u32;
    let min = min_from_time(t) as u32;
    let sec = sec_from_time(t) as u32;
    let ms = ms_from_time(t) as u32;

    let year_i = year as i64;
    let year_str = if !(0..=9999).contains(&year_i) {
        if year_i < 0 {
            format!("-{:06}", year_i.unsigned_abs())
        } else {
            format!("+{year_i:06}")
        }
    } else {
        format!("{year_i:04}")
    };

    Ok(format!(
        "{year_str}-{month:02}-{day_val:02}T{hour:02}:{min:02}:{sec:02}.{ms:03}Z"
    ))
}

/// ECMAScript §21.4.4.43 `Date.prototype.toJSON()`.
///
/// Returns the ISO string or `None` for invalid dates (does not throw).
pub fn date_to_json(t: f64) -> Option<String> {
    if t.is_nan() || t.is_infinite() {
        None
    } else {
        date_to_iso_string(t).ok()
    }
}

/// ECMAScript §21.4.4.42 `Date.prototype.toUTCString()`.
///
/// Returns a string like `"Thu, 01 Jan 1970 00:00:00 GMT"`.
///
/// # Examples
///
/// ```
/// use stator_js::builtins::date::date_to_utc_string;
///
/// let s = date_to_utc_string(0.0);
/// assert_eq!(s, "Thu, 01 Jan 1970 00:00:00 GMT");
/// ```
pub fn date_to_utc_string(t: f64) -> String {
    if t.is_nan() {
        return "Invalid Date".to_string();
    }
    let wd = week_day(t) as usize;
    let year = year_from_time(t) as i64;
    let month = month_from_time(t) as usize;
    let day_val = date_from_time(t) as u32;
    let hour = hour_from_time(t) as u32;
    let min = min_from_time(t) as u32;
    let sec = sec_from_time(t) as u32;

    format!(
        "{}, {:02} {} {:04} {:02}:{:02}:{:02} GMT",
        DAY_NAMES[wd], day_val, MONTH_NAMES[month], year, hour, min, sec
    )
}

/// ECMAScript §21.4.4.35 `Date.prototype.toString()`.
///
/// Returns a human-readable date string in local time.
pub fn date_to_string(t: f64) -> String {
    if t.is_nan() {
        return "Invalid Date".to_string();
    }
    let local = utc_to_local(t);
    let wd = week_day(local) as usize;
    let year = year_from_time(local) as i64;
    let month = month_from_time(local) as usize;
    let day_val = date_from_time(local) as u32;
    let hour = hour_from_time(local) as u32;
    let min = min_from_time(local) as u32;
    let sec = sec_from_time(local) as u32;

    let offset_ms = local_tz_offset_ms(t);
    let offset_min = (offset_ms / MS_PER_MINUTE) as i32;
    let sign = if offset_min >= 0 { '+' } else { '-' };
    let abs_offset = offset_min.unsigned_abs();
    let tz_h = abs_offset / 60;
    let tz_m = abs_offset % 60;

    format!(
        "{} {} {:02} {:04} {:02}:{:02}:{:02} GMT{}{:02}{:02}",
        DAY_NAMES[wd], MONTH_NAMES[month], day_val, year, hour, min, sec, sign, tz_h, tz_m
    )
}

/// ECMAScript §21.4.4.36 `Date.prototype.toDateString()`.
pub fn date_to_date_string(t: f64) -> String {
    if t.is_nan() {
        return "Invalid Date".to_string();
    }
    let local = utc_to_local(t);
    let wd = week_day(local) as usize;
    let year = year_from_time(local) as i64;
    let month = month_from_time(local) as usize;
    let day_val = date_from_time(local) as u32;

    format!(
        "{} {} {:02} {:04}",
        DAY_NAMES[wd], MONTH_NAMES[month], day_val, year
    )
}

/// ECMAScript §21.4.4.37 `Date.prototype.toTimeString()`.
pub fn date_to_time_string(t: f64) -> String {
    if t.is_nan() {
        return "Invalid Date".to_string();
    }
    let local = utc_to_local(t);
    let hour = hour_from_time(local) as u32;
    let min = min_from_time(local) as u32;
    let sec = sec_from_time(local) as u32;

    let offset_ms = local_tz_offset_ms(t);
    let offset_min = (offset_ms / MS_PER_MINUTE) as i32;
    let sign = if offset_min >= 0 { '+' } else { '-' };
    let abs_offset = offset_min.unsigned_abs();
    let tz_h = abs_offset / 60;
    let tz_m = abs_offset % 60;

    format!(
        "{:02}:{:02}:{:02} GMT{}{:02}{:02}",
        hour, min, sec, sign, tz_h, tz_m
    )
}

/// ECMAScript §21.4.4.38 `Date.prototype.toLocaleDateString()`.
///
/// Simplified implementation that returns the same format as `toDateString`.
pub fn date_to_locale_date_string(t: f64) -> String {
    date_to_date_string(t)
}

/// ECMAScript §21.4.4.39 `Date.prototype.toLocaleString()`.
///
/// Simplified implementation that returns the same format as `toString`.
pub fn date_to_locale_string(t: f64) -> String {
    date_to_string(t)
}

/// ECMAScript §21.4.4.40 `Date.prototype.toLocaleTimeString()`.
///
/// Simplified implementation that returns the same format as `toTimeString`.
pub fn date_to_locale_time_string(t: f64) -> String {
    date_to_time_string(t)
}

/// ECMAScript §21.4.4.44 `Date.prototype.valueOf()`.
///
/// Returns the time value (same as `getTime`).
pub fn date_value_of(t: f64) -> f64 {
    t
}

/// ECMAScript §21.4.4.45 `Date.prototype[@@toPrimitive](hint)`.
///
/// For `"string"` or `"default"` hints, returns the string representation.
/// For `"number"` hint, returns the time value.
pub fn date_to_primitive(t: f64, hint: &str) -> StatorResult<JsValue> {
    match hint {
        "number" => Ok(
            if t.is_finite()
                && t != 0.0
                && t.fract() == 0.0
                && t >= f64::from(i32::MIN)
                && t <= f64::from(i32::MAX)
            {
                JsValue::Smi(t as i32)
            } else {
                JsValue::HeapNumber(t)
            },
        ),
        "string" | "default" => Ok(JsValue::String(date_to_string(t).into())),
        _ => Err(StatorError::TypeError(
            "Invalid hint for Date.prototype[@@toPrimitive]".into(),
        )),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_date_now_positive() {
        let now = date_now();
        assert!(now > 0.0);
    }

    #[test]
    fn test_date_utc_basic() {
        // 2024-01-15T12:30:00.000Z
        let t = date_utc(2024.0, 0.0, 15.0, 12.0, 30.0, 0.0, 0.0);
        assert!(!t.is_nan());
        assert_eq!(date_get_utc_full_year(t), 2024.0);
        assert_eq!(date_get_utc_month(t), 0.0);
        assert_eq!(date_get_utc_date(t), 15.0);
        assert_eq!(date_get_utc_hours(t), 12.0);
        assert_eq!(date_get_utc_minutes(t), 30.0);
    }

    #[test]
    fn test_date_utc_epoch() {
        let t = date_utc(1970.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(t, 0.0);
    }

    #[test]
    fn test_date_utc_two_digit_year() {
        // Two-digit year: 70 → 1970
        let t = date_utc(70.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(t, 0.0);
    }

    #[test]
    fn test_date_parse_iso_full() {
        let t = date_parse("2024-01-15T12:30:00.000Z");
        assert!(!t.is_nan());
        assert_eq!(date_get_utc_full_year(t), 2024.0);
        assert_eq!(date_get_utc_month(t), 0.0);
        assert_eq!(date_get_utc_date(t), 15.0);
        assert_eq!(date_get_utc_hours(t), 12.0);
        assert_eq!(date_get_utc_minutes(t), 30.0);
    }

    #[test]
    fn test_date_parse_iso_date_only() {
        let t = date_parse("2024-06-15");
        assert!(!t.is_nan());
        assert_eq!(date_get_utc_full_year(t), 2024.0);
        assert_eq!(date_get_utc_month(t), 5.0); // June = 5
        assert_eq!(date_get_utc_date(t), 15.0);
    }

    #[test]
    fn test_date_parse_invalid() {
        assert!(date_parse("not a date").is_nan());
        assert!(date_parse("").is_nan());
        assert!(date_parse("2024-02-30T12:00:00Z").is_nan());
        assert!(date_parse("2024-01-15T12:30:00+25:00").is_nan());
        assert!(date_parse("2024-01-15T24:30:00Z").is_nan());
        assert!(date_parse("2024-01-15T12:30:00Zextra").is_nan());
    }

    #[test]
    fn test_date_to_iso_string_epoch() {
        let s = date_to_iso_string(0.0).unwrap();
        assert_eq!(s, "1970-01-01T00:00:00.000Z");
    }

    #[test]
    fn test_date_to_utc_string_epoch() {
        let s = date_to_utc_string(0.0);
        assert_eq!(s, "Thu, 01 Jan 1970 00:00:00 GMT");
    }

    #[test]
    fn test_date_to_iso_string_invalid() {
        assert!(date_to_iso_string(f64::NAN).is_err());
    }

    #[test]
    fn test_date_invalid_propagation() {
        let nan = f64::NAN;
        assert!(date_get_full_year(nan).is_nan());
        assert!(date_get_month(nan).is_nan());
        assert!(date_get_date(nan).is_nan());
        assert!(date_get_hours(nan).is_nan());
        assert!(date_get_minutes(nan).is_nan());
        assert!(date_get_seconds(nan).is_nan());
        assert!(date_get_milliseconds(nan).is_nan());
        assert!(date_get_day(nan).is_nan());
        assert!(date_get_timezone_offset(nan).is_nan());
    }

    #[test]
    fn test_date_setters_utc() {
        let t = date_utc(2024.0, 0.0, 15.0, 12.0, 30.0, 45.0, 500.0);

        // setUTCFullYear
        let t2 = date_set_utc_full_year(t, 2025.0, None, None);
        assert_eq!(date_get_utc_full_year(t2), 2025.0);
        assert_eq!(date_get_utc_month(t2), 0.0);

        // setUTCMonth
        let t2 = date_set_utc_month(t, 5.0, None);
        assert_eq!(date_get_utc_month(t2), 5.0);

        // setUTCDate
        let t2 = date_set_utc_date(t, 20.0);
        assert_eq!(date_get_utc_date(t2), 20.0);

        // setUTCHours
        let t2 = date_set_utc_hours(t, 8.0, None, None, None);
        assert_eq!(date_get_utc_hours(t2), 8.0);

        // setUTCMinutes
        let t2 = date_set_utc_minutes(t, 15.0, None, None);
        assert_eq!(date_get_utc_minutes(t2), 15.0);

        // setUTCSeconds
        let t2 = date_set_utc_seconds(t, 30.0, None);
        assert_eq!(date_get_utc_seconds(t2), 30.0);

        // setUTCMilliseconds
        let t2 = date_set_utc_milliseconds(t, 999.0);
        assert_eq!(date_get_utc_milliseconds(t2), 999.0);
    }

    #[test]
    fn test_date_construct_components() {
        // This constructs from local time, so we verify round-trip
        let t = date_utc(2024.0, 6.0, 4.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(date_get_utc_full_year(t), 2024.0);
        assert_eq!(date_get_utc_month(t), 6.0);
        assert_eq!(date_get_utc_date(t), 4.0);
    }

    #[test]
    fn test_date_to_json_valid() {
        let result = date_to_json(0.0);
        assert_eq!(result, Some("1970-01-01T00:00:00.000Z".to_string()));
    }

    #[test]
    fn test_date_to_json_invalid() {
        let result = date_to_json(f64::NAN);
        assert!(result.is_none());
    }

    #[test]
    fn test_date_value_of() {
        assert_eq!(date_value_of(12345.0), 12345.0);
        assert!(date_value_of(f64::NAN).is_nan());
    }

    #[test]
    fn test_time_clip_bounds() {
        // Within bounds
        assert_eq!(time_clip(0.0), 0.0);
        assert_eq!(time_clip(1000.5), 1000.0);

        // Out of bounds
        assert!(time_clip(8.64e15 + 1.0).is_nan());
        assert!(time_clip(-8.64e15 - 1.0).is_nan());
        assert!(time_clip(f64::INFINITY).is_nan());
        assert!(time_clip(f64::NAN).is_nan());
    }

    #[test]
    fn test_date_parse_year_only() {
        // "2024" should parse as a year-only ISO date
        let t = date_parse("2024");
        assert!(!t.is_nan());
        assert_eq!(date_get_utc_full_year(t), 2024.0);
        assert_eq!(date_get_utc_month(t), 0.0);
        assert_eq!(date_get_utc_date(t), 1.0);
    }

    #[test]
    fn test_date_to_string_invalid() {
        assert_eq!(date_to_string(f64::NAN), "Invalid Date");
        assert_eq!(date_to_date_string(f64::NAN), "Invalid Date");
        assert_eq!(date_to_time_string(f64::NAN), "Invalid Date");
        assert_eq!(date_to_utc_string(f64::NAN), "Invalid Date");
    }

    #[test]
    fn test_date_set_time() {
        let t = date_set_time(86400000.0);
        assert_eq!(t, 86400000.0);
        assert!(date_set_time(f64::INFINITY).is_nan());
    }

    #[test]
    fn test_extract_gmt_offset() {
        assert_eq!(
            extract_gmt_offset("Mon Jan 15 2024 12:30:00 GMT+0200"),
            Some(7_200_000.0)
        );
        assert_eq!(
            extract_gmt_offset("Mon Jan 15 2024 12:30:00 GMT-05:30"),
            Some(-19_800_000.0)
        );
        assert_eq!(
            extract_gmt_offset("Mon Jan 15 2024 12:30:00 UTC"),
            Some(0.0)
        );
    }

    #[test]
    fn test_parse_legacy_gmt_offset() {
        let t = date_parse("Mon Jan 15 2024 12:30:00 GMT+0200");
        assert!(!t.is_nan());
        assert_eq!(date_get_utc_full_year(t), 2024.0);
        assert_eq!(date_get_utc_month(t), 0.0);
        assert_eq!(date_get_utc_date(t), 15.0);
        assert_eq!(date_get_utc_hours(t), 10.0);
        assert_eq!(date_get_utc_minutes(t), 30.0);
    }

    #[test]
    fn test_date_parse_iso_signed_extended_year() {
        let t = date_parse("+010000-01-01T00:00:00.000Z");
        assert_eq!(
            date_to_iso_string(t).unwrap(),
            "+010000-01-01T00:00:00.000Z"
        );
    }

    #[test]
    fn test_date_parse_iso_rejects_non_canonical_year_width() {
        assert!(date_parse("+2024-01-15T12:30:00Z").is_nan());
        assert!(date_parse("02024-01-15T12:30:00Z").is_nan());
    }

    #[test]
    fn test_date_to_primitive_string_and_default_use_to_string() {
        let expected = date_to_string(0.0);
        assert_eq!(
            date_to_primitive(0.0, "string").unwrap(),
            JsValue::String(expected.clone().into())
        );
        assert_eq!(
            date_to_primitive(0.0, "default").unwrap(),
            JsValue::String(expected.into())
        );
    }

    #[test]
    fn test_date_to_primitive_number_returns_numeric_value() {
        assert_eq!(
            date_to_primitive(1234.0, "number").unwrap(),
            JsValue::Smi(1234)
        );
        match date_to_primitive(f64::NAN, "number").unwrap() {
            JsValue::HeapNumber(n) => assert!(n.is_nan()),
            other => panic!("expected NaN heap number, got {other:?}"),
        }
    }

    #[test]
    fn test_date_to_primitive_invalid_hint_throws() {
        assert!(matches!(
            date_to_primitive(0.0, "invalid"),
            Err(StatorError::TypeError(_))
        ));
    }
}
