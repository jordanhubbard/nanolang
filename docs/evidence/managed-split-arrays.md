# My managed split-array acceptance

I qualify task27e597 at frozen head `d0841511`, based on main `b28d9c8b`.
My [contract](../NANOISA_MANAGED_SPLIT_ARRAYS.md) admits STR_SPLIT, ARR_GET and
ARR_LEN only. Split arrays own copied string children; tags and stable handles
survive calls, explicit/implicit returns, locals, globals and error cleanup.
I preserve missing TAG_VOID, default array casts, identity equality and same-tag
ordering0. Unsupported construction/mutation/other collection shapes remain
refused. My compiler asserts the private array tag equals ISA TAG_ARRAY=7.

On Linux ARM64 I passed:

```sh
NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13 \
make test-llvm-literal-strings test-llvm-managed-strings test-verifier-profiles
```

The complete log is `/tmp/nanolang-managed-split-full-final.log`:

| Gate | Completed result |
| --- | --- |
| Scalar globals | 11 methods, 7.077s |
| Literal strings | 9 methods, 23.730s |
| Runtime package | 2 methods, 1.970s |
| Shared string-array core | 2 methods, 1.780s |
| Existing string core | 3 methods, 2.592s |
| Managed strings including six split methods | 45 methods, 51.348s |
| Shared verifier profiles | 1 method, 0.349s |

My split cases run ordinary VM controls, verify actual emitted native/wasm32
LLVM with opt, execute instrumented native code under ASan/UBSan, and run
import-free Wasm with Node and Wasmtime. They cover empty source/delimiter,
leading/repeated/trailing delimiters, nonoverlap, NUL/high bytes, missing and
large/negative indices, retained children after root release, exact array tags,
array parameters/results, initializer-result cleanup and default scalar casts.
Global arrays persist across repeated entry; two fresh Wasm instances begin
with zero live objects. Wrong-tag errors preserve previous global writes and
clear frame roots. Unsupported ARR_NEW/PUSH/SET/POP retain previous outputs.

Core controls exercise every bounded allocation budget through partial split
construction, descriptor growth, equal input handles and unchanged output on
failure. Emitted native allocation controls retain a previous string global
through failure and recover in the same instance. The real 1MiB Wasm cap forces
partial split failure, releases all partial children, preserves an earlier array
global and permits repeat entry/disposal. Physical allocation events and string
interning need not match the VM; observable bytes/tags/lifetimes do.

The contract preserves the original private-tag mismatch log and the two stale
pre-admission refusal logs. I corrected the tag against the ISA and retained the
old split programs as positive controls; unsupported-array refusal assertions
remain. I do not claim those initial runs passed.

I changed no source-language compiler and claim no new compiler bootstrap.
Parent488, parent51da, arbitrary collection shapes/cycles, Darwin sanitizer7ba
and historical evaluator791a remain open. This child does not complete them.

I restacked onto main `6e371b1d` after PR690's scalar reconstruction changes.
Integrated implementation `02219481` retains identical runtime/lowering/profile,
package inputs and LLVM/managed/profile test bytes to frozen `d0841511`.
The inherited Makefile change only adds the reconstruction truthiness test to
its reconstruction target. No affected runtime rebuild is needed;
`git diff --check` passes and the completed gate remains applicable.
