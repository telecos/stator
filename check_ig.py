import pathlib
t = pathlib.Path('crates/stator_core/src/builtins/install_globals.rs').read_text('utf-8')
idx = t.find('"stringify"')
section = t[idx:idx+2000]
print('has is_callable:', 'is_callable' in section)
print('has NativeFunction(f):', 'NativeFunction(f)' in section)
print('has repl_arg:', 'repl_arg' in section)
print('has is_array_replacer:', 'is_array_replacer' in section)
