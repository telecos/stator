"""Apply json.rs changes via Python to avoid edit tool sync issues."""
import pathlib

p = pathlib.Path('crates/stator_core/src/builtins/json.rs')
t = p.read_text(encoding='utf-8')

# 1. Replace INTEGER_THRESHOLD constant
old1 = (
    '/// `1e15` is chosen because `f64` can represent all integers up to `2^53`\n'
    '/// (≈ `9.007e15`) exactly, and formatting values larger than `1e15` as\n'
    '/// integers would lose precision on the representational boundary.\n'
    'const INTEGER_THRESHOLD: f64 = 1e15;'
)
new1 = (
    '/// ECMAScript uses exponential notation for numbers whose magnitude is ≥ 10²¹.\n'
    'const EXPONENTIAL_THRESHOLD: f64 = 1e21;'
)
assert old1 in t, 'constant not found'
t = t.replace(old1, new1, 1)
print('1. Replaced constant')

# 2. Replace stringify_value number branch
old2 = (
    '        JsonValue::Number(n) => {\n'
    '            if n.is_nan() || n.is_infinite() {\n'
    '                // ECMAScript: NaN and Infinity are serialised as "null".\n'
    '                Ok(Some("null".to_string()))\n'
    '            } else {\n'
    '                // Emit an integer representation if the value is a whole number.\n'
    '                if n.fract() == 0.0 && n.abs() < INTEGER_THRESHOLD {\n'
    '                    Ok(Some(format!("{}", *n as i64)))\n'
    '                } else {\n'
    '                    Ok(Some(format!("{n}")))\n'
    '                }\n'
    '            }\n'
    '        }'
)
new2 = (
    '        JsonValue::Number(n) => {\n'
    '            if n.is_nan() || n.is_infinite() {\n'
    '                // ECMAScript: NaN and Infinity are serialised as "null".\n'
    '                Ok(Some("null".to_string()))\n'
    '            } else {\n'
    '                Ok(Some(format_number(*n)))\n'
    '            }\n'
    '        }'
)
assert old2 in t, 'number branch not found'
t = t.replace(old2, new2, 1)
print('2. Replaced number branch')

# 3. Add format_number function before stringify_string
old3 = (
    '    }\n'
    '}\n'
    '\n'
    '/// Produce the JSON representation of a string, escaping per RFC 8259 §7.\n'
    'fn stringify_string(s: &str) -> String {'
)
new3 = (
    '    }\n'
    '}\n'
    '\n'
    '/// Format an `f64` as ECMAScript would represent the number.\n'
    '///\n'
    '/// Key differences from Rust\'s `Display`:\n'
    '/// - `-0.0` is formatted as `"0"` (no negative sign).\n'
    '/// - Whole numbers below `EXPONENTIAL_THRESHOLD` (10²¹) use integer notation.\n'
    '/// - Numbers ≥ 10²¹ use exponential notation with an explicit `+` sign\n'
    '///   (`e+21` not `e21`).\n'
    'pub fn format_number(n: f64) -> String {\n'
    '    // Negative zero → "0"\n'
    '    if n == 0.0 {\n'
    '        return "0".to_string();\n'
    '    }\n'
    '    // Whole numbers below the threshold → integer notation.\n'
    '    if n.fract() == 0.0 && n.abs() < EXPONENTIAL_THRESHOLD {\n'
    '        return format!("{}", n as i64);\n'
    '    }\n'
    '    // Everything else: delegate to Rust\'s formatter, then normalise the\n'
    '    // exponent marker from Rust\'s `e` to ECMAScript\'s `e+`.\n'
    '    let s = format!("{n}");\n'
    '    // Rust emits `1e21` whereas ES requires `1e+21`.\n'
    '    if let Some(pos) = s.find(\'e\') {\n'
    '        let (mantissa, exp_part) = s.split_at(pos);\n'
    '        let exp = &exp_part[1..]; // skip the \'e\'\n'
    '        if exp.starts_with(\'-\') {\n'
    '            return format!("{mantissa}e{exp}");\n'
    '        }\n'
    '        return format!("{mantissa}e+{exp}");\n'
    '    }\n'
    '    s\n'
    '}\n'
    '\n'
    '/// Produce the JSON representation of a string, escaping per RFC 8259 §7.\n'
    'fn stringify_string(s: &str) -> String {'
)
# Only insert if format_number fn is NOT already defined
if 'pub fn format_number' not in t:
    assert old3 in t, f'stringify_string anchor not found'
    t = t.replace(old3, new3, 1)
    print('3. Added format_number function')
else:
    print('3. format_number already present, skipping')

# 4. Update js_value_to_json signatures
old4 = (
    'pub fn js_value_to_json(value: &JsValue) -> StatorResult<Option<JsonValue>> {\n'
    '    js_value_to_json_inner(value, &mut HashSet::new())\n'
    '}\n'
    '\n'
    'fn js_value_to_json_inner(\n'
    '    value: &JsValue,\n'
    '    seen: &mut HashSet<usize>,\n'
    ') -> StatorResult<Option<JsonValue>> {'
)
new4 = (
    'pub fn js_value_to_json(value: &JsValue) -> StatorResult<Option<JsonValue>> {\n'
    '    js_value_to_json_inner(value, &mut HashSet::new(), "")\n'
    '}\n'
    '\n'
    '/// Like [`js_value_to_json`] but allows specifying the `key` that will be\n'
    '/// forwarded to a `toJSON` method if the object has one.\n'
    'pub fn js_value_to_json_with_key(value: &JsValue, key: &str) -> StatorResult<Option<JsonValue>> {\n'
    '    js_value_to_json_inner(value, &mut HashSet::new(), key)\n'
    '}\n'
    '\n'
    'fn js_value_to_json_inner(\n'
    '    value: &JsValue,\n'
    '    seen: &mut HashSet<usize>,\n'
    '    key: &str,\n'
    ') -> StatorResult<Option<JsonValue>> {'
)
if old4 in t:
    t = t.replace(old4, new4, 1)
    print('4. Updated js_value_to_json signatures')
else:
    print('4. Signature already updated or not found')

# 5. Update Array branch recursive call
old5 = (
    '            for item in items.borrow().iter() {\n'
    '                let json_item = js_value_to_json_inner(item, seen)?;'
)
new5 = (
    '            for (i, item) in items.borrow().iter().enumerate() {\n'
    '                let json_item = js_value_to_json_inner(item, seen, &i.to_string())?;'
)
if old5 in t:
    t = t.replace(old5, new5, 1)
    print('5. Updated Array branch')
else:
    print('5. Array branch already updated or not found')

# 6. Update PlainObject branch (toJSON + enumerable_iter)
old6 = (
    '        JsValue::PlainObject(map) => {\n'
    '            // §25.5.2 step 2: if the object has a callable `toJSON` property,\n'
    '            // invoke it and serialise the return value instead.\n'
    '            let to_json_fn = map.borrow().get("toJSON").and_then(|v| {\n'
    '                if let JsValue::NativeFunction(f) = v {\n'
    '                    Some(f.clone())\n'
    '                } else {\n'
    '                    None\n'
    '                }\n'
    '            });\n'
    '            if let Some(f) = to_json_fn {\n'
    '                let result = f(vec![JsValue::String(String::new().into())])?;\n'
    '                return js_value_to_json_inner(&result, seen);\n'
    '            }\n'
    '\n'
    '            let ptr = Rc::as_ptr(map) as usize;\n'
    '            if seen.contains(&ptr) {\n'
    '                return Err(StatorError::TypeError(\n'
    '                    "Converting circular structure to JSON".to_string(),\n'
    '                ));\n'
    '            }\n'
    '            seen.insert(ptr);\n'
    '            let mut entries: Vec<(String, JsonValue)> = Vec::new();\n'
    '            // §25.5.2 step 6: only enumerable own properties are serialised.\n'
    '            for (k, v) in map.borrow().enumerable_iter() {\n'
    '                if let Some(jv) = js_value_to_json_inner(v, seen)? {\n'
    '                    entries.push((k.clone(), jv));\n'
    '                }\n'
    '            }\n'
    '            seen.remove(&ptr);\n'
    '            Ok(Some(JsonValue::Object(Rc::new(RefCell::new(entries)))))\n'
    '        }'
)
new6 = (
    '        JsValue::PlainObject(map) => {\n'
    '            // §25.5.2 step 2: if the object has a callable `toJSON` property,\n'
    '            // invoke it and serialise the return value instead.  We check for\n'
    '            // any callable (NativeFunction *or* Function bytecode).\n'
    '            let to_json_val = map.borrow().get("toJSON").cloned();\n'
    '            let is_to_json_callable = matches!(\n'
    '                &to_json_val,\n'
    '                Some(JsValue::NativeFunction(_)) | Some(JsValue::Function(_))\n'
    '            );\n'
    '            if is_to_json_callable\n'
    '                && let Some(callee) = to_json_val\n'
    '            {\n'
    '                let key_js = JsValue::String(key.to_string().into());\n'
    '                let result =\n'
    '                    crate::interpreter::dispatch_call_value(&callee, vec![key_js])?;\n'
    '                return js_value_to_json_inner(&result, seen, key);\n'
    '            }\n'
    '\n'
    '            let ptr = Rc::as_ptr(map) as usize;\n'
    '            if seen.contains(&ptr) {\n'
    '                return Err(StatorError::TypeError(\n'
    '                    "Converting circular structure to JSON".to_string(),\n'
    '                ));\n'
    '            }\n'
    '            seen.insert(ptr);\n'
    '            let mut entries: Vec<(String, JsonValue)> = Vec::new();\n'
    '            // §25.5.2 step 6: only enumerable own properties are serialised.\n'
    '            for (k, v) in map.borrow().enumerable_iter() {\n'
    '                if let Some(jv) = js_value_to_json_inner(v, seen, k)? {\n'
    '                    entries.push((k.clone(), jv));\n'
    '                }\n'
    '            }\n'
    '            seen.remove(&ptr);\n'
    '            Ok(Some(JsonValue::Object(Rc::new(RefCell::new(entries)))))\n'
    '        }'
)
if old6 in t:
    t = t.replace(old6, new6, 1)
    print('6. Updated PlainObject branch')
else:
    print('6. PlainObject branch already updated or not found, trying alternate')
    # Maybe it was already partially updated
    if 'js_value_to_json_inner(&result, seen)' in t:
        print('   Found old 2-arg call, needs manual fix')
    if 'js_value_to_json_inner(v, seen)?' in t:
        print('   Found old 2-arg enumerable call')

p.write_text(t, encoding='utf-8')
print('\nAll done!')
