# My private packed scalar core evidence

I recorded task `task_bdaf1bb74e9749bf8cc1ccaad21af56c` and the storage/coercion
contract in `6cc4bb64` before implementation. My frozen production and test pin
is `9bb88d3c`; my base is `3e4e5a89`. This is a private core prerequisite under
aggregate parent488 and managed parent51da, with no opcode/profile admission.

I preserve the declared int/float/bool/U8 kind, exact matching payload bits,
and only the audited U8-to-int, int-to-U8 and int-to-float coercions. I retain
stable handle aliases across buffer/table growth, publish writes after checked
allocation, and reclaim packed storage without child ownership. Unsupported
private writes leave storage and string owners unchanged. The VM's raw
mismatched union-member fallbacks remain a separate semantics/shape obligation;
I do not execute them to infer a portable contract.

My first focused reference link omitted `src/nanoisa/isa.c` and stopped with an
undefined `isa_tag_name` reference. I retained that log at
`/tmp/nanolang-packed-scalar-focused.log`, added the actual dependency, then
passed the corrected focused gate in 3.088 seconds. This was a test link failure,
not evidence of a runtime defect.

My checked VM reference emits 563 ordinary vectors: signed integer endpoints,
negative byte wrapping, all 256 byte values, bools, signed zero, subnormal and
finite maximum float bits, quiet NaN payloads, and integer precision/tie cases
around 2^53. I compare these exact tags/bits in verified native LLVM IR under
ASan/UBSan and import-free wasm32. I use default nearest-even rounding; I do
not mutate or claim equivalence under other rounding modes.

Production and testing builds both run. Node checks zero imports and repeats
all core groups three times in each of two fresh instances; Wasmtime also runs
each export. Deterministic allocator controls cover descriptor/buffer/growth
failure rollback, retained alias mutation, table relocation, missing/empty
results, independent contexts and terminal disposal. One hundred allocate/free
rounds retain a stable page count, and a real one-MiB Wasm limit preserves prior
values after failed growth before complete teardown.

I do not claim new source bootstrap evidence for this runtime-only change.
Nested/nominal child tracing, cycles, static write eligibility and mutable
opcode lowering remain open, as do Darwin sanitizer7ba and evaluator791a.

My full affected gate on the frozen source passed in
`/tmp/nanolang-packed-scalar-full.log`:

| Gate | Methods | Seconds |
| --- | ---: | ---: |
| Scalar globals | 11 | 7.138 |
| Literal strings | 9 | 24.816 |
| Runtime package | 2 | 2.341 |
| String/boxed/packed array core | 3 | 5.340 |
| Existing string core | 3 | 3.103 |
| Managed target operations | 45 | 60.886 |
| Shared profiles | 1 | 0.375 |

I used `NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13`
with `make test-llvm-literal-strings test-llvm-managed-strings test-verifier-profiles`.
Independent production review found no scoped blocker. This is tested Linux
native/Wasm evidence, not a full platform or release acceptance claim.
