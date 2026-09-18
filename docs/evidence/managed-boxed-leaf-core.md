# My boxed leaf-array foundation acceptance

I qualify taskcb827048 at frozen source `3ec6e10b`, based on main `539072d4`.
My [contract](../NANOISA_MANAGED_BOXED_LEAVES.md) defines private exact-tag/value
ownership and transactional promotion. I add no opcode, verifier-profile,
FrameOutput or executable admission.

My generic leaf arrays preserve scalar bits and own string children. Promotion
moves existing string edges into boxed storage without changing their reference
counts or stable array handle. Append/set prepare storage, retain the new child,
then commit; replacement publishes before releasing the old child. Retained get
and transferring pop have distinct ownership. Legacy string-only accessors refuse
promoted descriptors; current admitted split modules never perform promotion.
Capacity never shrinks during a write, and each promoted element grows from an
8-byte handle to a 16-byte value, so the checked byte-accounting delta is nonnegative.

I passed this frozen Linux ARM64 gate:

```sh
NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13 \
make test-llvm-literal-strings test-llvm-managed-strings test-verifier-profiles
```

`/tmp/nanolang-boxed-leaf-full.log` records:

| Gate | Completed result |
| --- | --- |
| Scalar globals | 11 methods, 6.826s |
| Literal strings | 9 methods, 23.860s |
| Runtime package | 2 methods, 1.995s |
| String/boxed-leaf array core | 2 methods, 2.089s |
| Existing string core | 3 methods, 2.699s |
| Managed string/split operations | 45 methods, 53.292s |
| Shared verifier profiles | 1 method, 0.318s |

The expanded array core compiles verified native LLVM, executes production and
testing builds with ASan/UBSan, and compiles verified import-free wasm32 LLVM.
Node executes two fresh instances with three rounds per group; Wasmtime executes
each exported group. New leaf groups cover exact NaN/signed-zero/integer bits,
scalar/string mixtures, duplicate children, same-child replacement, table growth,
alias mutation, promotion success/rollback, retained gets, transferred pops,
missing values, unchanged refused outputs and independent terminal disposal.
One hundred creation/promotion/release rounds keep memory pages stable. The real
1MiB Wasm bound forces finite append refusal without losing child owners; teardown
returns allocator object accounting to zero. No historical failure is replayed.

Production package checks regenerate target-specific ABI and source/header hashes
and retain no testing hooks. I changed no source-language compiler and claim no
new bootstrap. Parent488 retains required packed coercion, shape eligibility,
nested/nominal child traversal, cycle collection and mutable opcode lowering.
Parent51da, Darwin sanitizer7ba and evaluator791a remain open.

I restacked onto main `d1d3c9a6` after reconstruction/component-entry changes.
Integrated implementation `b548aec7` retains identical managed source/header,
package inputs and array fixtures/harness to frozen `3ec6e10b`. Inherited Makefile
changes affect reconstruction and component entry checks, not this runtime gate.
No affected rebuild is needed; `git diff --check` passes and the completed gate
remains applicable.
