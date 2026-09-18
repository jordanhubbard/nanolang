# My bounded scalar formatting evidence

I implement child `task_074f76d564d145b88554a66b4fcfb204` after PR647
(`cb204da1`). Contract `eec551b1` preceded code. My managed-runtime parent and
portable floating formatter/parser tasks remain open.

I produce exact VM int/U8 decimal, bool true/false and void/enum empty strings.
My unsigned negative magnitude avoids negating INT64_MIN; a 20-byte scratch
buffer holds the longest admitted output. Runtime allocation publishes one
checked owner. String CAST_STRING returns its existing value, transferring
exactly one owner without allocation or retain/release churn.

My closed-profile scan refuses CAST_STRING modules with any current floating
instruction or float parameter/result annotation, including unreachable helper
bodies. Existing float modules without CAST_STRING remain admitted. The
scalar/literal-only API profiles retain refusal. I do not approximate `%g`.

My frozen Linux ARM64 gates passed in `/tmp/nanolang-managed-format-gate.log`:

- 11 global methods, 6.934 seconds.
- 9 literal-string methods, 21.795 seconds.
- 2 runtime package methods, 0.916 seconds.
- 3 runtime core methods, 1.671 seconds.
- 16 emitted managed methods, 11.633 seconds.
- 16 shared profile cases, 0.242 seconds for their containing method.

New controls compare actual VM/native LLVM/import-free Wasm outputs for eleven
scalar cases and literal/dynamic string identity. They include signed endpoints,
U8 endpoints, both booleans, void/enum empty output and NUL-bearing strings.
Calls, local/global aliases and twenty repeated entries preserve exact live
counts. Native public entry recovers from byte/table allocation failures;
allocation-disabled repeated identity calls succeed without allocating.
Runtime core tests exercise allocator failure/recovery on native and Wasm.
Native generated IR receives the explicit ASan pass and sanitizer/leak checks;
Node and Wasmtime run the runtime, and emitted Wasm checks zero imports.

Fifteen ordinary unused-helper fixtures separately cover PUSH_F64, CAST_FLOAT,
every admitted typed F64 operation, and float parameter/result annotations.
Each still executes normally in the VM, while both target publishers refuse
and preserve prior output. Existing managed decimal tests retain non-formatting
float conversion and cleanup controls. Historical failed artifacts remain
unexecuted. I claim no Darwin acceptance or full managed-runtime completion.
