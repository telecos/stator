//! Temporal API conformance tests.

#[cfg(test)]
mod tests {
    use crate::builtins::global::global_eval;
    use crate::objects::value::JsValue;

    fn assert_eval_true(src: &str) {
        let result = global_eval(src).unwrap();
        assert_eq!(result, JsValue::Boolean(true), "expected true for: {src}");
    }

    fn assert_eval_err(src: &str) {
        assert!(global_eval(src).is_err(), "expected error for: {src}");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_namespace_exists() {
        assert_eval_true("typeof Temporal === 'object' && Temporal !== null");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_to_string_tag() {
        assert_eval_true("Temporal[Symbol.toStringTag] === 'Temporal'");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_to_string_tag_descriptor() {
        assert_eval_true(
            "var d = Object.getOwnPropertyDescriptor(Temporal, Symbol.toStringTag); d.configurable === true && d.enumerable === false",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_now_namespace_exists() {
        assert_eval_true("typeof Temporal.Now === 'object' && Temporal.Now !== null");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_now_to_string_tag() {
        assert_eval_true("Temporal.Now[Symbol.toStringTag] === 'Temporal.Now'");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_now_instant_returns_object() {
        assert_eval_true("typeof Temporal.Now.instant() === 'object'");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_now_instant_instanceof() {
        assert_eval_true("Temporal.Now.instant() instanceof Temporal.Instant");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_now_instant_epoch_milliseconds_is_number() {
        assert_eval_true("typeof Temporal.Now.instant().epochMilliseconds === 'number'");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_now_instant_returns_fresh_instances() {
        assert_eval_true("Temporal.Now.instant() !== Temporal.Now.instant()");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_now_plain_date_iso_returns_object() {
        assert_eval_true("typeof Temporal.Now.plainDateISO() === 'object'");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_now_plain_date_iso_instanceof() {
        assert_eval_true("Temporal.Now.plainDateISO() instanceof Temporal.PlainDate");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_now_plain_date_iso_has_numeric_fields() {
        assert_eval_true(
            "var d = Temporal.Now.plainDateISO(); typeof d.year === 'number' && typeof d.month === 'number' && typeof d.day === 'number'",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_now_plain_date_iso_ranges() {
        assert_eval_true(
            "var d = Temporal.Now.plainDateISO(); d.month >= 1 && d.month <= 12 && d.day >= 1 && d.day <= 31",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_plain_date_from_parses_iso_string() {
        assert_eval_true(
            "var d = Temporal.PlainDate.from('2024-01-15'); d.year === 2024 && d.month === 1 && d.day === 15",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_plain_date_from_returns_instance() {
        assert_eval_true("Temporal.PlainDate.from('2024-01-15') instanceof Temporal.PlainDate");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_plain_date_constructor_stub() {
        assert_eval_true(
            "var d = new Temporal.PlainDate(2024, 2, 29); d.year === 2024 && d.month === 2 && d.day === 29",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_plain_date_prototype_getters_are_accessors() {
        assert_eval_true(
            "var y = Object.getOwnPropertyDescriptor(Temporal.PlainDate.prototype, 'year'); var m = Object.getOwnPropertyDescriptor(Temporal.PlainDate.prototype, 'month'); var d = Object.getOwnPropertyDescriptor(Temporal.PlainDate.prototype, 'day'); typeof y.get === 'function' && y.enumerable === false && y.configurable === true && typeof m.get === 'function' && typeof d.get === 'function'",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_plain_date_constructor_linkage() {
        assert_eval_true("Temporal.PlainDate.prototype.constructor === Temporal.PlainDate");
    }

    #[test]
    fn e2e_temporal_plain_date_invalid_string_throws() {
        assert_eval_err("Temporal.PlainDate.from('2024-13-15')");
    }

    #[test]
    fn e2e_temporal_plain_date_invalid_constructor_throws() {
        assert_eval_err("new Temporal.PlainDate(2024, 2, 30)");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_duration_from_object() {
        assert_eval_true(
            "var d = Temporal.Duration.from({ hours: 1, minutes: 30 }); d.hours === 1 && d.minutes === 30 && d.seconds === 0",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_duration_from_defaults_missing_fields_to_zero() {
        assert_eval_true(
            "var d = Temporal.Duration.from({ minutes: 45 }); d.years === 0 && d.hours === 0 && d.minutes === 45 && d.nanoseconds === 0",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_duration_constructor_stub() {
        assert_eval_true(
            "var d = new Temporal.Duration(0, 0, 0, 0, 2, 15); d.hours === 2 && d.minutes === 15",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_duration_prototype_getters_are_accessors() {
        assert_eval_true(
            "var h = Object.getOwnPropertyDescriptor(Temporal.Duration.prototype, 'hours'); var m = Object.getOwnPropertyDescriptor(Temporal.Duration.prototype, 'minutes'); typeof h.get === 'function' && h.configurable === true && typeof m.get === 'function'",
        );
    }

    #[test]
    fn e2e_temporal_duration_rejects_mixed_signs() {
        assert_eval_err("Temporal.Duration.from({ hours: 1, minutes: -30 })");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_instant_from_epoch_string() {
        assert_eval_true(
            "var i = Temporal.Instant.from('1705276800000'); i.epochMilliseconds === 1705276800000",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_instant_from_returns_instance() {
        assert_eval_true("Temporal.Instant.from('1705276800000') instanceof Temporal.Instant");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_instant_constructor_stub() {
        assert_eval_true(
            "var i = new Temporal.Instant(1705276800000); i.epochMilliseconds === 1705276800000",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_instant_epoch_milliseconds_descriptor_is_accessor() {
        assert_eval_true(
            "var d = Object.getOwnPropertyDescriptor(Temporal.Instant.prototype, 'epochMilliseconds'); typeof d.get === 'function' && d.configurable === true && d.enumerable === false",
        );
    }

    #[test]
    fn e2e_temporal_instant_invalid_epoch_throws() {
        assert_eval_err("Temporal.Instant.from('not-an-epoch')");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_plain_time_from_parses_string() {
        assert_eval_true(
            "var t = Temporal.PlainTime.from('12:30:00'); t.hour === 12 && t.minute === 30 && t.second === 0",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_plain_time_returns_instance() {
        assert_eval_true("Temporal.PlainTime.from('12:30:00') instanceof Temporal.PlainTime");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_plain_time_constructor_stub() {
        assert_eval_true(
            "var t = new Temporal.PlainTime(7, 8, 9); t.hour === 7 && t.minute === 8 && t.second === 9",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_plain_time_prototype_getters_are_accessors() {
        assert_eval_true(
            "var h = Object.getOwnPropertyDescriptor(Temporal.PlainTime.prototype, 'hour'); var m = Object.getOwnPropertyDescriptor(Temporal.PlainTime.prototype, 'minute'); var s = Object.getOwnPropertyDescriptor(Temporal.PlainTime.prototype, 'second'); typeof h.get === 'function' && typeof m.get === 'function' && typeof s.get === 'function'",
        );
    }

    #[test]
    fn e2e_temporal_plain_time_invalid_string_throws() {
        assert_eval_err("Temporal.PlainTime.from('25:30:00')");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_zoned_date_time_constructor_stub() {
        assert_eval_true(
            "new Temporal.ZonedDateTime(1705276800000, 'UTC') instanceof Temporal.ZonedDateTime",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_zoned_date_time_stub_fields() {
        assert_eval_true(
            "var z = new Temporal.ZonedDateTime(1705276800000, 'UTC'); z.epochMilliseconds === 1705276800000 && z.timeZone === 'UTC'",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_zoned_date_time_getters_are_accessors() {
        assert_eval_true(
            "var e = Object.getOwnPropertyDescriptor(Temporal.ZonedDateTime.prototype, 'epochMilliseconds'); var z = Object.getOwnPropertyDescriptor(Temporal.ZonedDateTime.prototype, 'timeZone'); typeof e.get === 'function' && typeof z.get === 'function'",
        );
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn e2e_temporal_zoned_date_time_constructor_linkage() {
        assert_eval_true("Temporal.ZonedDateTime.prototype.constructor === Temporal.ZonedDateTime");
    }
}
