# My filled-array constructor lowering

On 2026-09-17 I lowered two-argument `array_new(count, fill)` for the array
shapes already supported by my emitter: integer, string, enum and finite
record elements. I evaluate count then fill once, including zero counts,
keep anonymous temporary slots, and preserve the C-seed loop and element tags.
I refuse unsupported arities, count types and fill shapes before publication.

I also check nonnegative counts in both bytecode frontends after evaluating
both operands. My previous C-seed loop silently returned an empty array for
negative counts. Compiler temporary names could also shadow legal source
bindings such as `__anew_fill__`; anonymous bindings repair that collision.

Validation:

- `make -j8 test-nanoisa-src-nano`: 86 checks and 42 integration methods pass.
- Sixteen exact C-seed opcode comparisons cover all seven fixture functions.
- Both modules verify and run in NanoVM and strict C11 native AOT, including
  nested construction, zero-count effects, record/string transport, and source
  bindings that used to collide with compiler temporary names.
- Seven unsupported operand cases are refused.
- Both frontends' modules reject negative counts under VM and AOT. A file
  written by the fill operand confirms count then fill ran before the trap.
- `make -j8 test-nanovirt`: 89 checks pass.

Logs: `/tmp/nanolang-filled-second.log` and
`/tmp/nanolang-filled-nanovirt.log`. The earlier failed collision regression
is retained in `/tmp/nanolang-filled-targeted.log`.

Native C source transpilation received its companion evaluation/count repair
in PR427 (`task_dbdd78cd11a6412aa0904d82ea013434`). This gate does not claim aggregate
allocation cleanup; the existing returned-record-array task remains open.

A freshly built canonical compiler advances to `unsupported local type
array<bool>`, recorded as the next emitter continuation. Its probe is
`/tmp/nanolang-canonical-after-filled-probe.log`. Full compiler emission and
bytecode bootstrap remain unfinished.
