# My declared enum values

On 2026-09-17 I lowered declared enum members from the names and signed values
retained by my parser. I do not rescan source tokens. Enum locals, parameters,
results and record fields use integer storage, as do supported enum arrays and
lists. Integer-compatible assignments retain the C-seed source contract.

My fixture checks explicit negative and noncontiguous values, automatic
increments, scalar calls, record fields, array returns and list access. Fourteen
C-seed bytecode comparisons pass; `make -j8 test-nanoisa-src-nano` passes
86 checks and 34 integration methods. Both modules run under NanoVM and strict
C11 AOT. I reject unknown members, ambiguous duplicate enum declarations, and
an integer local used as a member receiver even when its name matches an enum.
A local or global value takes precedence over an enum constant receiver.

My C seed currently infers a struct for `at(array<Enum>)`: assigning that result
to a declared enum local fails. Direct equality uses generic `EQ`, while my
emitter retains the enum type and emits `I64_EQ`. Task
`task_d2b541318e964ea4946542bffe3990ae` tracks this metadata/parity repair.
The exact comparison fixture retains enum-array construction, transport and
length checks; a separate variant executes direct element assertions through
both C-seed and self-hosted VM/native paths. I do not claim exact instruction
identity for that variant.

Full compiler emission and matching bytecode bootstrap remain unfinished.

A fresh C-seed-hosted canonical driver now first refuses
`undefined function list_CompilerDiagnostic_set`, recorded under
`task_0b7d7221b3fd44d5a8055a84e72397a1`. The actual debugger probe is
`/tmp/nanolang-canonical-after-enums-probe.log`; no compiler module is published.
