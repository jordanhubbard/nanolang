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
(three methods, 1.257 seconds) and Clang 18 with the installed GCC 13 runtime
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
