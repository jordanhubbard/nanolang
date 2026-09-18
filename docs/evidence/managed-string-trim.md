# My managed trim evidence

I implement `task_bc2bd84520124b17aa4dfad09fb663dd` under my pre-code
[trim contract](../NANOISA_MANAGED_TRIM.md). My frozen implementation was
`4a959768`; integrating main `e2f55f7f` produced `17cfdb28`. The complete
`src/nanoisa`, `src/nanovm` and affected trim/global/profile test files compare
byte-identically across that integration. Main's intervening reconstruction
change does not alter this runtime gate.

I passed these actual Linux ARM64 gates:

- Three focused trim methods, 2.425 seconds: native ASan/UBSan core and
  import-free Node/Wasmtime core; actual VM, sanitized emitted LLVM and Wasm.
- `make test-llvm-literal-strings test-llvm-managed-strings test-verifier-profiles`:
  11 global, 9 literal, 2 packaging, 3 core, 28 managed methods and the shared
  profile/publication test passed. The managed methods took 27.230 seconds.
  Native compilation used
  `NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13`.

My 12 byte cases include empty/all-whitespace/unchanged text, all four trimmed
ASCII whitespace bytes, retained vertical tab/form feed, embedded NUL,
non-ASCII bytes and a 1024-byte interior. Literal and computed inputs preserve
exact bytes through a called helper, globals and repeated entry. Direct core
checks establish distinct live result handles even when no bytes change,
one surviving input alias, and complete disposal. Native and Wasm allocator
failure controls preserve the output sentinel while consuming exactly one
input owner; null-output controls also consume that owner without touching
its alias. Emitted native allocation failure unwinds the helper, preserves
the committed global and permits corrected reentry. Dynamic wrong-tag tests
preserve that global and clear transient owners on native/Wasm.

My managed profile alone admits STR_TRIM. Scalar/literal selectors retain
refusal. I updated an adjacent generic unsupported-operation fixture from
newly admitted trim to still unsupported uppercase; previous-output checks
remain. The package gate checks target-specific embedded IR and zero Wasm
imports. Parent independent production review found no scoped blocker.

I retain logs `/tmp/nanolang-managed-trim-focused.log` and
`/tmp/nanolang-managed-trim-full.log`. I did not replay failed historical
artifacts. Full managed parent51da, Darwin managed sanitizer7ba and historical
evaluator791a remain open. My separately recorded char-index owner child62caf
is not implemented here. This does not complete target or release acceptance.
