import pathlib
t = pathlib.Path('crates/stator_core/src/builtins/date.rs').read_text('utf-8')
print('extract_gmt_offset:', 'fn extract_gmt_offset' in t)
print('offset_ms:', 'extract_gmt_offset(s)' in t)
print('test_extract_gmt_offset:', 'test_extract_gmt_offset' in t)
print('test_parse_legacy_gmt:', 'test_parse_legacy_gmt' in t)
