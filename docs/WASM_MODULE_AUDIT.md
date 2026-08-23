# WASM Module Audit

My direct WASM backend emits standalone scalar modules. It does not yet provide
linear memory, WASI, JavaScript imports, C linking, or module linking. This is a
small boundary, and I state it plainly.

## What Works Now

| Surface | Status |
|---|---|
| `int`, `float`, `bool` parameters, returns, and locals | Supported |
| Arithmetic, comparisons, `and`, `or`, `not` | Supported |
| Local function calls and recursion | Supported |
| `if`, `while`, `let`, `set`, `return` | Supported |
| Exported scalar functions | Supported |
| WASI `_start` and console I/O | Not implemented |
| `for`, imports, extern calls | Not implemented |
| Strings, arrays, tuples, records, unions, `match` | Not implemented |

The examples under `examples/wasm/` exercise the supported boundary without
pretending that host I/O exists.

## Module Classes

No packaged module is currently linked end to end by `--target wasm`. Some are
portable algorithms waiting on data representation; others require a host or a
different toolchain.

| Class | Modules | WASM requirement |
|---|---|---|
| Pure numeric candidates | `std/mathx` | Module linking; some functions already fit the scalar subset, while `mathx_is_prime` also needs boolean locals and `while` |
| Pure data-structure candidates | `vector2d`, `proptest`, `nano_highlight`, `std/result`, `std/diagnostics`, `std/math/*`, `std/crypto/sha256`, `std/graphx`, `std/regex_nfa`, `std/binary` | Linear memory plus strings, arrays, records, unions, and function values as used |
| Host service modules | `filesystem`, `stdio`, `std/fs`, `std/io`, `std/env`, `preferences`, `tools`, root `stdlib/timing` | WASI or explicit JavaScript host imports |
| Network and process modules | `curl`, `github`, `openai`, `websocket`, `uv`, `event`, `http_server`, `pty`, `pybridge`, `dispatch`, `std/process` | Replacement host APIs; browser networking and processes are not POSIX sockets and `fork` |
| Native library modules | `sqlite`, `unicode`, `nanoisa`, `forth_see`, `pt2_module`, `pt2_state`, `libc`, `math_ext` | A C-to-WASM link path and a defined NanoLang ABI |
| Graphics, audio, and physics | SDL family, OpenGL family, Bullet, MuJoCo, GPU, ncurses, readline, ProTracker audio | Emscripten or purpose-built Web APIs; these are separate from the direct scalar emitter |

`math_ext` is mathematically portable but currently consists of extern `libm`
calls. `vector2d` is pure NanoLang but uses records and fields. Those facts are
more useful than a single portable/not-portable label.

## Existing Examples

Most language examples contain useful pure kernels but print strings from
`main`. The whole programs therefore exceed the current direct backend even
when their arithmetic helpers fit it. `examples/cross_backend/hello_cross_backend.nano`
has this shape: `compute_value` is scalar, while its `main` needs strings and
output.

The browser playground is another path. It compiles my interpreter with
Emscripten and exposes `nl_run` and `nl_check`; it does not execute files emitted
by `--target wasm`.

## Test Boundary

```bash
make test-backends
bash tests/test_backends.sh ./bin/nanoc_c
```

When WABT is installed, the integration test validates and runs the simple
WASM examples with `wasm-validate` and `wasm-interp`. Without WABT it still
checks emission and reports the missing external validator.
