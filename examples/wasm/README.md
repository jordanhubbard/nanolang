# WebAssembly Examples

I compile these examples with my direct WASM backend. They use no native
modules, browser APIs, SDL, or OpenGL.

```bash
make stage1
./bin/nanoc examples/wasm/wasm_arithmetic.nano --target wasm --no-sourcemap -o /tmp/wasm_arithmetic.wasm
wasm-validate /tmp/wasm_arithmetic.wasm
wasm-interp /tmp/wasm_arithmetic.wasm --run-all-exports
```

`wasm-interp` reports `main() => i64:31`. The module exports its functions; it
does not define WASI `_start`, so use an export-aware runtime rather than trying
to launch it as a WASI command.

My current direct backend supports scalar `int`, `float`, and `bool` values,
local functions, recursion, mutable locals, `if`, `while`, arithmetic,
comparisons, and boolean operators. It rejects `for`, imports, extern calls,
strings, arrays, tuples, records, unions, `match`, I/O, and string conversion.

Shadows run in my host interpreter before I emit the WASM file. The validator
and runtime command above test the emitted file separately.
