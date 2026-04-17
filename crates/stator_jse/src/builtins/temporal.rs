//! TC39 Stage 3 Temporal API — modern date/time types.
//!
//! Implements the core types from the [Temporal proposal][spec]:
//!
//! - [`PlainDate`] — calendar date without time or time zone.
//! - [`PlainTime`] — wall-clock time without date or time zone.
//! - [`PlainDateTime`] — calendar date + wall-clock time (no time zone).
//! - [`Instant`] — exact point on the UTC timeline (nanoseconds since epoch).
//! - [`Duration`] — difference between two Temporal values.
//! - [`PlainYearMonth`] / [`PlainMonthDay`] — partial calendar dates.
//! - [`Now`] — static methods returning the current instant / date / time.
//!
//! # ISO 8601 Calendar
//!
//! All types use the ISO 8601 calendar unless explicitly overridden by a
//! custom calendar protocol object.  Calendar-dependent operations
//! (e.g. month-day arithmetic) delegate to the calendar.
//!
//! [spec]: https://tc39.es/proposal-temporal/

use std::fmt;

// ── ISO date / time components ──────────────────────────────────────────────

/// Year–month–day triple in the ISO 8601 calendar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IsoDate {
    /// ISO year (negative = BCE).
    pub year: i32,
    /// Month 1–12.
    pub month: u8,
    /// Day 1–31.
    pub day: u8,
}

impl IsoDate {
    /// Create a new ISO date, returning `None` if the components are invalid.
    pub fn new(year: i32, month: u8, day: u8) -> Option<Self> {
        if !(1..=12).contains(&month) {
            return None;
        }
        let max_day = days_in_month(year, month);
        if day == 0 || day > max_day {
            return None;
        }
        Some(Self { year, month, day })
    }
}

impl fmt::Display for IsoDate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.year >= 0 && self.year <= 9999 {
            write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
        } else {
            // Extended year format with sign.
            write!(f, "{:+07}-{:02}-{:02}", self.year, self.month, self.day)
        }
    }
}

/// Hour–minute–second–nanosecond time-of-day.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IsoTime {
    /// Hour 0–23.
    pub hour: u8,
    /// Minute 0–59.
    pub minute: u8,
    /// Second 0–59.
    pub second: u8,
    /// Millisecond 0–999.
    pub millisecond: u16,
    /// Microsecond 0–999.
    pub microsecond: u16,
    /// Nanosecond 0–999.
    pub nanosecond: u16,
}

impl IsoTime {
    /// Create a new ISO time, returning `None` if the components are invalid.
    pub fn new(
        hour: u8,
        minute: u8,
        second: u8,
        millisecond: u16,
        microsecond: u16,
        nanosecond: u16,
    ) -> Option<Self> {
        if hour > 23 || minute > 59 || second > 59 {
            return None;
        }
        if millisecond > 999 || microsecond > 999 || nanosecond > 999 {
            return None;
        }
        Some(Self {
            hour,
            minute,
            second,
            millisecond,
            microsecond,
            nanosecond,
        })
    }

    /// Midnight.
    pub fn midnight() -> Self {
        Self {
            hour: 0,
            minute: 0,
            second: 0,
            millisecond: 0,
            microsecond: 0,
            nanosecond: 0,
        }
    }

    /// Total nanoseconds since midnight.
    pub fn to_nanoseconds(&self) -> i64 {
        let h = self.hour as i64;
        let m = self.minute as i64;
        let s = self.second as i64;
        let ms = self.millisecond as i64;
        let us = self.microsecond as i64;
        let ns = self.nanosecond as i64;
        ((((h * 60 + m) * 60 + s) * 1_000 + ms) * 1_000 + us) * 1_000 + ns
    }
}

impl fmt::Display for IsoTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:02}:{:02}:{:02}", self.hour, self.minute, self.second)?;
        if self.nanosecond != 0 {
            write!(
                f,
                ".{:03}{:03}{:03}",
                self.millisecond, self.microsecond, self.nanosecond
            )
        } else if self.microsecond != 0 {
            write!(f, ".{:03}{:03}", self.millisecond, self.microsecond)
        } else if self.millisecond != 0 {
            write!(f, ".{:03}", self.millisecond)
        } else {
            Ok(())
        }
    }
}

// ── PlainDate ───────────────────────────────────────────────────────────────

/// A calendar date without time or time zone (`Temporal.PlainDate`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlainDate {
    /// The underlying ISO date.
    pub iso: IsoDate,
}

impl PlainDate {
    /// Create a `PlainDate` from ISO components.
    pub fn new(year: i32, month: u8, day: u8) -> Option<Self> {
        IsoDate::new(year, month, day).map(|iso| Self { iso })
    }

    /// Year component.
    pub fn year(&self) -> i32 {
        self.iso.year
    }

    /// Month component (1–12).
    pub fn month(&self) -> u8 {
        self.iso.month
    }

    /// Day component (1–31).
    pub fn day(&self) -> u8 {
        self.iso.day
    }

    /// Day of week (1 = Monday .. 7 = Sunday) per ISO 8601.
    pub fn day_of_week(&self) -> u8 {
        iso_day_of_week(self.iso.year, self.iso.month, self.iso.day)
    }

    /// Number of days in the month containing this date.
    pub fn days_in_month(&self) -> u8 {
        days_in_month(self.iso.year, self.iso.month)
    }

    /// Number of days in the year containing this date.
    pub fn days_in_year(&self) -> u16 {
        if is_leap_year(self.iso.year) {
            366
        } else {
            365
        }
    }

    /// Whether the year is a leap year.
    pub fn in_leap_year(&self) -> bool {
        is_leap_year(self.iso.year)
    }
}

impl fmt::Display for PlainDate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.iso)
    }
}

// ── PlainTime ───────────────────────────────────────────────────────────────

/// A wall-clock time without date or time zone (`Temporal.PlainTime`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlainTime {
    /// The underlying ISO time.
    pub iso: IsoTime,
}

impl PlainTime {
    /// Create a `PlainTime` from components.
    pub fn new(
        hour: u8,
        minute: u8,
        second: u8,
        millisecond: u16,
        microsecond: u16,
        nanosecond: u16,
    ) -> Option<Self> {
        IsoTime::new(hour, minute, second, millisecond, microsecond, nanosecond)
            .map(|iso| Self { iso })
    }

    /// Hour component (0–23).
    pub fn hour(&self) -> u8 {
        self.iso.hour
    }

    /// Minute component (0–59).
    pub fn minute(&self) -> u8 {
        self.iso.minute
    }

    /// Second component (0–59).
    pub fn second(&self) -> u8 {
        self.iso.second
    }
}

impl fmt::Display for PlainTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.iso)
    }
}

// ── PlainDateTime ───────────────────────────────────────────────────────────

/// A date + time without time zone (`Temporal.PlainDateTime`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlainDateTime {
    /// The date component.
    pub date: IsoDate,
    /// The time component.
    pub time: IsoTime,
}

impl PlainDateTime {
    /// Create a `PlainDateTime` from date and time components.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        year: i32,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: u8,
        millisecond: u16,
        microsecond: u16,
        nanosecond: u16,
    ) -> Option<Self> {
        let date = IsoDate::new(year, month, day)?;
        let time = IsoTime::new(hour, minute, second, millisecond, microsecond, nanosecond)?;
        Some(Self { date, time })
    }

    /// Create from an existing `PlainDate` and `PlainTime`.
    pub fn from_parts(date: PlainDate, time: PlainTime) -> Self {
        Self {
            date: date.iso,
            time: time.iso,
        }
    }
}

impl fmt::Display for PlainDateTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}T{}", self.date, self.time)
    }
}

// ── Instant ─────────────────────────────────────────────────────────────────

/// An exact point on the UTC timeline (`Temporal.Instant`).
///
/// Stored as nanoseconds since the Unix epoch (1970-01-01T00:00:00Z).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Instant {
    /// Nanoseconds since the Unix epoch.
    pub epoch_nanoseconds: i128,
}

/// Minimum representable instant (−100,000,000 days from epoch).
pub const INSTANT_MIN_NS: i128 = -100_000_000 * 86_400 * 1_000_000_000;
/// Maximum representable instant (+100,000,000 days from epoch).
pub const INSTANT_MAX_NS: i128 = 100_000_000 * 86_400 * 1_000_000_000;

impl Instant {
    /// Create an `Instant` from nanoseconds since the epoch.
    pub fn from_epoch_nanoseconds(ns: i128) -> Option<Self> {
        if !(INSTANT_MIN_NS..=INSTANT_MAX_NS).contains(&ns) {
            return None;
        }
        Some(Self {
            epoch_nanoseconds: ns,
        })
    }

    /// Epoch milliseconds (for compatibility with `Date.now()`).
    pub fn epoch_milliseconds(&self) -> i64 {
        (self.epoch_nanoseconds / 1_000_000) as i64
    }

    /// Epoch seconds.
    pub fn epoch_seconds(&self) -> i64 {
        (self.epoch_nanoseconds / 1_000_000_000) as i64
    }
}

impl fmt::Display for Instant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Simplified: just show epoch nanoseconds.
        write!(f, "{}Z", self.epoch_nanoseconds)
    }
}

// ── Duration ────────────────────────────────────────────────────────────────

/// A duration between two Temporal values (`Temporal.Duration`).
///
/// All fields may be negative (the sign must be uniform across all
/// non-zero fields per the Temporal spec).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Duration {
    /// Years.
    pub years: i32,
    /// Months.
    pub months: i32,
    /// Weeks.
    pub weeks: i32,
    /// Days.
    pub days: i32,
    /// Hours.
    pub hours: i64,
    /// Minutes.
    pub minutes: i64,
    /// Seconds.
    pub seconds: i64,
    /// Milliseconds.
    pub milliseconds: i64,
    /// Microseconds.
    pub microseconds: i64,
    /// Nanoseconds.
    pub nanoseconds: i64,
}

impl Duration {
    /// Create a new duration.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        years: i32,
        months: i32,
        weeks: i32,
        days: i32,
        hours: i64,
        minutes: i64,
        seconds: i64,
        milliseconds: i64,
        microseconds: i64,
        nanoseconds: i64,
    ) -> Option<Self> {
        let d = Self {
            years,
            months,
            weeks,
            days,
            hours,
            minutes,
            seconds,
            milliseconds,
            microseconds,
            nanoseconds,
        };
        // All non-zero fields must have the same sign.
        if !d.is_sign_consistent() {
            return None;
        }
        Some(d)
    }

    /// Returns `true` if all non-zero fields share the same sign.
    pub fn is_sign_consistent(&self) -> bool {
        let fields: [i64; 10] = [
            self.years as i64,
            self.months as i64,
            self.weeks as i64,
            self.days as i64,
            self.hours,
            self.minutes,
            self.seconds,
            self.milliseconds,
            self.microseconds,
            self.nanoseconds,
        ];
        let mut has_pos = false;
        let mut has_neg = false;
        for &f in &fields {
            if f > 0 {
                has_pos = true;
            }
            if f < 0 {
                has_neg = true;
            }
        }
        !(has_pos && has_neg)
    }

    /// The sign of this duration: 1 (positive), −1 (negative), or 0 (zero).
    pub fn sign(&self) -> i8 {
        let fields: [i64; 10] = [
            self.years as i64,
            self.months as i64,
            self.weeks as i64,
            self.days as i64,
            self.hours,
            self.minutes,
            self.seconds,
            self.milliseconds,
            self.microseconds,
            self.nanoseconds,
        ];
        for &f in &fields {
            if f > 0 {
                return 1;
            }
            if f < 0 {
                return -1;
            }
        }
        0
    }

    /// Negate all fields.
    pub fn negated(&self) -> Self {
        Self {
            years: -self.years,
            months: -self.months,
            weeks: -self.weeks,
            days: -self.days,
            hours: -self.hours,
            minutes: -self.minutes,
            seconds: -self.seconds,
            milliseconds: -self.milliseconds,
            microseconds: -self.microseconds,
            nanoseconds: -self.nanoseconds,
        }
    }

    /// Returns `true` if all fields are zero.
    pub fn is_zero(&self) -> bool {
        self.sign() == 0
    }

    /// Total nanoseconds (time portion only — date portions are calendar-dependent).
    pub fn total_nanoseconds(&self) -> i128 {
        let d = self.days as i128;
        let h = self.hours as i128;
        let m = self.minutes as i128;
        let s = self.seconds as i128;
        let ms = self.milliseconds as i128;
        let us = self.microseconds as i128;
        let ns = self.nanoseconds as i128;
        ((((d * 24 + h) * 60 + m) * 60 + s) * 1_000 + ms) * 1_000_000 + us * 1_000 + ns
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = if self.sign() < 0 { "-" } else { "" };
        write!(f, "{sign}P")?;
        let y = self.years.unsigned_abs();
        let mo = self.months.unsigned_abs();
        let w = self.weeks.unsigned_abs();
        let d = self.days.unsigned_abs();
        if y != 0 {
            write!(f, "{y}Y")?;
        }
        if mo != 0 {
            write!(f, "{mo}M")?;
        }
        if w != 0 {
            write!(f, "{w}W")?;
        }
        if d != 0 {
            write!(f, "{d}D")?;
        }
        let h = self.hours.unsigned_abs();
        let mi = self.minutes.unsigned_abs();
        let s = self.seconds.unsigned_abs();
        if h != 0 || mi != 0 || s != 0 {
            write!(f, "T")?;
            if h != 0 {
                write!(f, "{h}H")?;
            }
            if mi != 0 {
                write!(f, "{mi}M")?;
            }
            if s != 0 {
                write!(f, "{s}S")?;
            }
        }
        Ok(())
    }
}

// ── PlainYearMonth ──────────────────────────────────────────────────────────

/// A year and month without a specific day (`Temporal.PlainYearMonth`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlainYearMonth {
    /// ISO year.
    pub year: i32,
    /// Month 1–12.
    pub month: u8,
}

impl PlainYearMonth {
    /// Create a `PlainYearMonth`.
    pub fn new(year: i32, month: u8) -> Option<Self> {
        if !(1..=12).contains(&month) {
            return None;
        }
        Some(Self { year, month })
    }

    /// Number of days in this month.
    pub fn days_in_month(&self) -> u8 {
        days_in_month(self.year, self.month)
    }
}

impl fmt::Display for PlainYearMonth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04}-{:02}", self.year, self.month)
    }
}

// ── PlainMonthDay ───────────────────────────────────────────────────────────

/// A month and day without a year (`Temporal.PlainMonthDay`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlainMonthDay {
    /// Month 1–12.
    pub month: u8,
    /// Day 1–31.
    pub day: u8,
}

impl PlainMonthDay {
    /// Create a `PlainMonthDay`.
    pub fn new(month: u8, day: u8) -> Option<Self> {
        if !(1..=12).contains(&month) || day == 0 || day > 31 {
            return None;
        }
        Some(Self { month, day })
    }
}

impl fmt::Display for PlainMonthDay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:02}-{:02}", self.month, self.day)
    }
}

// ── Temporal.Now ────────────────────────────────────────────────────────────

/// Static methods for `Temporal.Now`.
pub struct Now;

impl Now {
    /// Returns the current instant as nanoseconds since the Unix epoch.
    pub fn instant() -> Instant {
        use std::time::SystemTime;
        let ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos() as i128)
            .unwrap_or(0);
        // Clamp to the valid range.
        Instant::from_epoch_nanoseconds(ns.clamp(INSTANT_MIN_NS, INSTANT_MAX_NS)).unwrap_or(
            Instant {
                epoch_nanoseconds: 0,
            },
        )
    }

    /// Returns the current epoch milliseconds (like `Date.now()`).
    pub fn epoch_milliseconds() -> i64 {
        Self::instant().epoch_milliseconds()
    }
}

// ── Calendar helpers ────────────────────────────────────────────────────────

/// Returns `true` if `year` is a leap year in the ISO 8601 calendar.
pub fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Number of days in the given month of the given year (ISO 8601).
pub fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Day of week for an ISO date (1 = Monday .. 7 = Sunday).
///
/// Uses Tomohiko Sakamoto's algorithm.
fn iso_day_of_week(year: i32, month: u8, day: u8) -> u8 {
    static T: [u8; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let mut y = year;
    if month < 3 {
        y -= 1;
    }
    let dow = (y + y / 4 - y / 100 + y / 400 + T[(month - 1) as usize] as i32 + day as i32) % 7;
    // Convert from Sunday=0 to Monday=1..Sunday=7.
    if dow == 0 { 7 } else { dow as u8 }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── IsoDate ──────────────────────────────────────────────────────────

    #[test]
    fn test_iso_date_valid() {
        assert!(IsoDate::new(2024, 2, 29).is_some()); // leap year
        assert!(IsoDate::new(2023, 2, 28).is_some());
        assert!(IsoDate::new(2023, 2, 29).is_none()); // not leap year
        assert!(IsoDate::new(2024, 13, 1).is_none()); // bad month
        assert!(IsoDate::new(2024, 0, 1).is_none()); // bad month
        assert!(IsoDate::new(2024, 1, 0).is_none()); // bad day
        assert!(IsoDate::new(2024, 1, 32).is_none()); // bad day
    }

    #[test]
    fn test_iso_date_display() {
        let d = IsoDate::new(2024, 3, 15).unwrap();
        assert_eq!(d.to_string(), "2024-03-15");
    }

    // ── IsoTime ──────────────────────────────────────────────────────────

    #[test]
    fn test_iso_time_valid() {
        assert!(IsoTime::new(0, 0, 0, 0, 0, 0).is_some());
        assert!(IsoTime::new(23, 59, 59, 999, 999, 999).is_some());
        assert!(IsoTime::new(24, 0, 0, 0, 0, 0).is_none());
        assert!(IsoTime::new(0, 60, 0, 0, 0, 0).is_none());
    }

    #[test]
    fn test_iso_time_display() {
        let t = IsoTime::new(14, 30, 0, 0, 0, 0).unwrap();
        assert_eq!(t.to_string(), "14:30:00");
        let t2 = IsoTime::new(14, 30, 0, 500, 0, 0).unwrap();
        assert_eq!(t2.to_string(), "14:30:00.500");
    }

    #[test]
    fn test_iso_time_nanoseconds() {
        let t = IsoTime::new(1, 0, 0, 0, 0, 0).unwrap();
        assert_eq!(t.to_nanoseconds(), 3_600_000_000_000);
    }

    // ── PlainDate ────────────────────────────────────────────────────────

    #[test]
    fn test_plain_date_accessors() {
        let d = PlainDate::new(2024, 2, 29).unwrap();
        assert_eq!(d.year(), 2024);
        assert_eq!(d.month(), 2);
        assert_eq!(d.day(), 29);
        assert!(d.in_leap_year());
        assert_eq!(d.days_in_month(), 29);
        assert_eq!(d.days_in_year(), 366);
    }

    #[test]
    fn test_plain_date_day_of_week() {
        // 2024-01-01 is Monday.
        let d = PlainDate::new(2024, 1, 1).unwrap();
        assert_eq!(d.day_of_week(), 1);
        // 2024-01-07 is Sunday.
        let d2 = PlainDate::new(2024, 1, 7).unwrap();
        assert_eq!(d2.day_of_week(), 7);
    }

    // ── PlainTime ────────────────────────────────────────────────────────

    #[test]
    fn test_plain_time_accessors() {
        let t = PlainTime::new(14, 30, 45, 0, 0, 0).unwrap();
        assert_eq!(t.hour(), 14);
        assert_eq!(t.minute(), 30);
        assert_eq!(t.second(), 45);
    }

    // ── PlainDateTime ────────────────────────────────────────────────────

    #[test]
    fn test_plain_date_time_display() {
        let dt = PlainDateTime::new(2024, 3, 15, 14, 30, 0, 0, 0, 0).unwrap();
        assert_eq!(dt.to_string(), "2024-03-15T14:30:00");
    }

    // ── Instant ──────────────────────────────────────────────────────────

    #[test]
    fn test_instant_epoch() {
        let i = Instant::from_epoch_nanoseconds(0).unwrap();
        assert_eq!(i.epoch_milliseconds(), 0);
        assert_eq!(i.epoch_seconds(), 0);
    }

    #[test]
    fn test_instant_range() {
        assert!(Instant::from_epoch_nanoseconds(INSTANT_MAX_NS).is_some());
        assert!(Instant::from_epoch_nanoseconds(INSTANT_MAX_NS + 1).is_none());
        assert!(Instant::from_epoch_nanoseconds(INSTANT_MIN_NS).is_some());
        assert!(Instant::from_epoch_nanoseconds(INSTANT_MIN_NS - 1).is_none());
    }

    // ── Duration ─────────────────────────────────────────────────────────

    #[test]
    fn test_duration_sign_consistency() {
        assert!(Duration::new(1, 0, 0, 0, 0, 0, 0, 0, 0, 0).is_some());
        assert!(Duration::new(-1, 0, 0, 0, 0, 0, 0, 0, 0, 0).is_some());
        assert!(Duration::new(1, -1, 0, 0, 0, 0, 0, 0, 0, 0).is_none()); // mixed signs
    }

    #[test]
    fn test_duration_negated() {
        let d = Duration::new(1, 2, 0, 3, 0, 0, 0, 0, 0, 0).unwrap();
        let neg = d.negated();
        assert_eq!(neg.years, -1);
        assert_eq!(neg.months, -2);
        assert_eq!(neg.days, -3);
    }

    #[test]
    fn test_duration_display() {
        let d = Duration::new(1, 2, 0, 3, 4, 30, 0, 0, 0, 0).unwrap();
        assert_eq!(d.to_string(), "P1Y2M3DT4H30M");
    }

    #[test]
    fn test_duration_zero() {
        let d = Duration::default();
        assert!(d.is_zero());
        assert_eq!(d.sign(), 0);
    }

    // ── PlainYearMonth / PlainMonthDay ───────────────────────────────────

    #[test]
    fn test_plain_year_month() {
        let ym = PlainYearMonth::new(2024, 2).unwrap();
        assert_eq!(ym.days_in_month(), 29);
        assert_eq!(ym.to_string(), "2024-02");
    }

    #[test]
    fn test_plain_month_day() {
        let md = PlainMonthDay::new(12, 25).unwrap();
        assert_eq!(md.to_string(), "12-25");
        assert!(PlainMonthDay::new(0, 1).is_none());
        assert!(PlainMonthDay::new(1, 0).is_none());
    }

    // ── Now ──────────────────────────────────────────────────────────────

    #[test]
    fn test_now_instant() {
        let i = Now::instant();
        // Should be a reasonable epoch time (after 2020).
        assert!(i.epoch_seconds() > 1_577_836_800);
    }

    #[test]
    fn test_now_epoch_milliseconds() {
        let ms = Now::epoch_milliseconds();
        assert!(ms > 1_577_836_800_000);
    }

    // ── Calendar helpers ─────────────────────────────────────────────────

    #[test]
    fn test_is_leap_year() {
        assert!(is_leap_year(2000));
        assert!(is_leap_year(2024));
        assert!(!is_leap_year(1900));
        assert!(!is_leap_year(2023));
    }

    #[test]
    fn test_days_in_month_all() {
        assert_eq!(days_in_month(2024, 1), 31);
        assert_eq!(days_in_month(2024, 2), 29);
        assert_eq!(days_in_month(2023, 2), 28);
        assert_eq!(days_in_month(2024, 4), 30);
        assert_eq!(days_in_month(2024, 12), 31);
    }
}
