# My managed decimal conversion evidence

I implement bounded child `task_34ce900cad86496b876bdf262b46bdaf` under my open
managed-runtime parent. Contract `1e94ac55` preceded implementation; my base is
merged PR644 (`41e652f0`). I preserve the documented closed C-locale boundary
without changing global locale or adding native/Wasm imports.

I borrow the immutable string view, skip the six C-locale whitespace bytes,
consume an optional sign, then accumulate decimal digits against an unsigned
signed-result limit. I check `(limit - digit) / 10` before multiplication.
I return INT64_MIN directly for magnitude 2^63; I never convert that magnitude
to signed or negate INT64_MIN. Missing digits produce zero and overflow
saturates. Embedded NUL terminates parsing. The helper allocates nothing and
leaves release to normal consumed-operand cleanup.

My frozen Linux ARM64 Make gates passed in
`/tmp/nanolang-managed-decimal-gate.log`:

- 11 scalar-global methods (6.941 seconds).
- 9 literal-string methods (32.185 seconds).
- 2 runtime package methods (0.927 seconds).
- 3 runtime core methods (1.546 seconds), including native sanitizers and actual
  import-free Node/Wasmtime runtime execution.
- 13 emitted managed methods (9.476 seconds), including two new decimal methods.
- 15 distinct profile cases in the shared admission/publication method.

My decimal cases compare actual VM, native LLVM and Wasm execution for 26 byte
inputs, including signed limits/saturation, signs, decimal suffixes, NUL and all
six whitespace bytes. Both literals and dynamic aliases cross helper calls.
Repeated entries leave no transient strings; global aliases survive success and
checked float-cast failure. Non-string bool/U8/enum/int/float/void conversions
retain existing behavior. Native generated IR is explicitly ASan-instrumented,
then linked with sanitizer/leak checking. Runtime core tests parse while
allocation is disabled and confirm unchanged handle ownership/counts.

The initial direct Python run lacked fresh-worktree literal/global helper
binaries; the full Make gates above build those prerequisites and passed. No
production test failure was hidden by that setup correction. Historical failed
artifacts remain unexecuted. The profile probe separately confirms scalar and
literal-only refusal while the managed profile admits string CAST_INT.
String CAST_FLOAT and CAST_STRING still refuse publication and preserve prior
outputs. Portable floating parsing/formatting and broad parent acceptance remain
open; I claim no Darwin acceptance from these Linux target tests.
