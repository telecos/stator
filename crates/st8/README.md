# st8

Command-line shell for the Stator JavaScript engine.

`st8` is analogous to V8's `d8`: it can execute JavaScript files, evaluate
inline snippets, and start a Chrome DevTools Protocol inspector endpoint for
debugging.

## Usage

```text
st8 <file.js>
st8 -e "1 + 2"
st8 --inspect <file.js>
st8 --inspect-brk <file.js>
st8 --inspect=9333 <file.js>
st8 --emit-snapshot=<path>
st8 --snapshot=<path> <file.js>
st8 --jit-stats <file.js>
```

## Package contents

The published package includes only the shell source, this README, and Cargo
metadata. Engine internals live in the `stator_jse` crate, and the C ABI for
embedders lives in `stator_jse_ffi`.
