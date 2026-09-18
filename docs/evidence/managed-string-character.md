# My managed character-byte evidence

I implement `task_32b265f80d604826adc9ac29716598ab` under my pre-code
[character contract](../NANOISA_MANAGED_CHARACTER.md), after the merged VM
operand cleanup prerequisite62caf/PR670. My frozen implementation `32ba87f1`
became `4017b272` after integration of main `fe4d1181`. All `src/nanoisa`,
`src/nanovm` and affected character/profile tests compare byte-identically;
the intervening canonical source-frontend work does not change these gates.

I passed three focused methods in 2.262 seconds. My six byte strings and nine
indices cover 54 pairs: empty text, embedded NUL, bytes128/255, a long string,
INT64_MIN/MAX, negative and exact range edges. Literal and computed strings
agree on actual VM, ASan-instrumented emitted native LLVM, and import-free
Node/Wasmtime. Bool/U8/float/void/enum/string index tags retain the VM's
existing fallback to zero. A string used as both operands retains its global
alias after both transient owners are released. Calls, reentry and disposal
preserve the same contract.

My direct native ASan/UBSan and Wasm core checks disable allocation, retain
two source references, and check unchanged owner counts after byte access.
Invalid API flag and null-output controls preserve that state; output changes
only on success. Corrected-source VM/native/Wasm source-type refusals release
the dynamic string index while retaining the committed global. This is
ordinary lifecycle acceptance, not replay of any historical failed artifact.

I passed `make test-llvm-literal-strings test-llvm-managed-strings
 test-verifier-profiles`: 11 global, 9 literal, 2 target packaging, 3 runtime
core, 31 managed methods and the shared profile/publication test. The managed
methods took 29.595 seconds. Native gates used
`NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13`.
Logs: `/tmp/nanolang-managed-character-focused.log` and
`/tmp/nanolang-managed-character-full.log`. Parent independent production
review found no scoped blocker.

Only my managed profile admits STR_CHAR_AT. Scalar/literal selectors and
unsupported-operation output preservation remain checked. The helper borrows
without allocating; ordinary frame cleanup releases both operands. Source
builtin typing is separate. This runtime-only slice makes no new compiler
bootstrap claim, and leaves full runtime51da, Darwin7ba and evaluator791a open.
