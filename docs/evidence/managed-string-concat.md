# My consuming concatenation checkpoint

I test the standalone helper under task_b1cc086f8cdf476cb0814f5ade9a15b1.
This checkpoint does not admit computed-string bytecode or complete frame
cleanup, target IR packaging, substring or conversion work.

I consume exactly one owner per input and publish the result only after
allocation and input release complete. I cover equal handles with an external
alias, literal/managed combinations, empty and embedded-NUL bytes, descriptor
growth, separate byte/table allocation failures, retained aliases, failure
consuming the last owner, and 2,000 repeated empty concatenations with one live
result and unchanged Wasm page count.

My frozen helper source passes `make test-managed-string-core` with GCC
(three methods, 1.257 seconds) and Clang 23 development build with the installed GCC 13 runtime
path (three methods, 1.333 seconds). Both native runs use ASan/UBSan and leak
checks. Node runs five runtime groups across two fresh instances and three
rounds per instance; Wasmtime runs each group. Production native/Wasm smoke
builds include concatenation without test hooks. Wasm declares zero imports;
its runtime LLVM IR passes `opt -passes=verify`.

I retain the logs as `/tmp/nanolang-managed-concat-gcc-final.log` and
`/tmp/nanolang-managed-concat-clang-final.log`. Existing allocator pressure,
coalescing and lifecycle checks remain in the same gate. These runtime tests
do not establish emitted-program ownership until the next integration stage.

My non-admitting IR packaging prototype passes
`NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13 make test-managed-runtime-package`.
I compile native and wasm32 variants separately, verify both with `opt`, retain
source hashes, compiler version, flags, target triple/layout and IR hashes in
the generated manifest, and compare two complete generations for exact bytes.
I compile a C consumer of the generated header and compare its dumped bytes
with both original IR modules. Each dumped module accepts an appended scalar
application calling the reserved-name helper: native execution and import-free
Node/Wasmtime execution pass. This establishes a packaging mechanism, not
application frame cleanup or allocation through emitted bytecode.

The target is deliberately standalone and rebuilt on every request; compiler
or flag changes cannot silently reuse its output. I have not added a runtime
package dependency or profile change to the public translator yet. My first
Wasm packaging fixture used `main`, which Clang treats as an ABI-special entry;
I changed the fixture to the intended freestanding `nano_package` export.

After integrating enum PR631 at `d9105a48`, I rerun the unchanged core plus
packaging gate. The packaging method passes in 0.442 seconds and the three
core methods pass in 1.298 seconds. The package manifest now also hashes its
generator script. Log: `/tmp/nanolang-managed-concat-integrated-final.log`.

## My private module ABI continuation

After PR632, `managed_module.c` packages one runtime instance and its first-error
latch. It exposes scalar ABI helpers for begin/finish/dispose, retain/release,
consuming concat, length and byte ordering. Compile-time layout assertions
validate the constant-view adapter. The package manifest includes both module
and allocator sources. This code still has no public translator caller.

`make test-managed-runtime-package` passes two methods with the native Clang
runtime selection recorded above. The new native sanitizer and import-free
Node/Wasmtime control exercises first-error preservation, rejected nested entry
and busy disposal without erasing the caller status, a retained root across an
error and next invocation, exact NUL-bearing concat, final release, idempotent
disposal and refusal after disposal. These are helper ABI checks; emitted-frame
cleanup and bytecode admission remain unfinished.

## My emitted ownership and instruction gate

Source/test checkpoint `cc1f2ec9` connects the target runtime to LLVM/Wasm.
I preserve the original scalar and literal profile selectors and add the
managed selector for STR_CONCAT and string/numeric ADD. Target, profile and
reserved-name checks precede output. Substring and conversions remain refused.

My seven emitted-program methods pass in 5.058 seconds. They compare ordinary
VM execution, native LLVM and import-free Wasm, including NUL/empty content,
DUP/SWAP aliases, branch joins, 2,000 loop iterations, ten recursive calls,
explicit/implicit helper returns, numeric/enum boundaries, initializer results,
last-write globals, repeated/fresh instances and terminal disposal. Six callee
assertion/type failures unwind frames while preserving earlier global writes.
Deterministic native byte/table allocation failure returns MEMORY and permits
a later successful entry. A real one-megabyte Wasm maximum rejects growth;
five subsequent attempts reuse storage with no live frame strings or page
increase. These allocator-limit errors are recoverable statuses, not traps.

I mark the emitted LLVM functions for ASan, run LLVM's ASan instrumentation
pass, verify instrumentation is present, emit an object, and link it with an
ASan/UBSan harness and leak interception. Merely passing sanitizer flags while
linking existing IR would not establish that its loads/stores were instrumented.
My standalone C core retains its separate native ASan/UBSan gate.

The reviewed conditional-branch path checks status after releasing its
condition before dispatching an edge. Only the exported legacy entry calls
llvm.trap; managed frames report and clean errors first. The dispose-before-
first-entry defect from review is fixed and tested in private native/Wasm
helpers and the public Wasm export. It remains tracked as
`task_c4c2b24bfcc3451d8bd7b89897f5cdab` until this change merges.

Final scoped commands use the explicit native Clang support-file selection
shown above:

- `make test-llvm-managed-strings test-verifier-profiles`: seven emitted methods,
  two package/module methods, three core methods and thirteen shared-profile
  decisions pass. Logs: `/tmp/nanolang-managed-reviewed-final.log`.
- `make test-verifier`: all 96 ordinary verifier tests pass, including cleanup;
  log `/tmp/nanolang-managed-final-focused.log`.
- Adjacent LLVM/float (16 methods), scalar globals (11), literal strings (9),
  and generic numeric/enum (12) groups pass. The 39-method Wasm/scalar batch
  had one obsolete concat-refusal expectation; its other 38 methods passed.
  I retained refusal coverage using unsupported substring and reran all seven
  generic-comparison methods successfully in 7.095 seconds. Logs:
  `/tmp/nanolang-managed-adjacent-gates.log` and
  `/tmp/nanolang-managed-comparisons-final.log`.

The numeric managed-profile fixture initially declared a union count instead
of an enum count; its legitimate profile refusal was a fixture error. The
corrected fixture passes numeric/enum VM/native/Wasm checks. No eligibility
rule was relaxed to accommodate that error. This is Linux ARM64 and Wasm
execution evidence; I do not claim Darwin execution or full release acceptance.

After rebasing onto main through PR634 (`38f29203`), my production files are
unchanged from `cc1f2ec9` (rebased source commit `efbc9201`). The integrated
package/core/profile/verifier and seven emitted methods pass again; emitted
methods take 4.974 seconds in `/tmp/nanolang-managed-integrated-main634.log`.
I then add explicit and implicit dynamic return-tag cleanup controls plus
allocation failure inside a callee with a retained global owner. All nine
emitted methods pass in 6.321 seconds, including ASan-instrumented generated
functions, in `/tmp/nanolang-managed-complete-ownership.log`.
