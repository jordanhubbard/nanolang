# My alphanumeric classification host boundary

I map the existing `is_alnum(int) -> bool` builtin to the empty-library
`vm_is_alnum` host import. I preserve its existing one-integer argument and
boolean result metadata. My VM and native adapters already implement this
contract; I do not alter their classification semantics.

My emitter gate passes 86 comparison checks and 57 regression methods. The
new fixture compares four function bodies and ordered imports with my C-seed
emitter (11 checks). Both modules execute in my VM and generated native code.
The assertions cover the ASCII digit/upper/lower range endpoints, neighboring
punctuation, underscore, negative input, out-of-range input, direct returns,
and once-only operand evaluation.

Four malformed calls refuse output. Three additional malformed extern
contracts reject an incorrect result, parameter type, or arity. The focused
signature regression passes after adding these cases. Logs:
`/tmp/nanolang-is-alnum-gate.log` and
`/tmp/nanolang-is-alnum-signatures.log`.

A fresh canonical driver builds after restacking onto `4081b77d`. Actual
whole-compiler emission advances to `undefined function str_starts_with`
(`/tmp/nanolang-alnum-fullcompiler-probe.log`). I track prefix/suffix lowering
as `task_891724de0cb5454eb0a575b099aba188`; complete compiler emission and the bytecode fixed point remain
unestablished.

The final restacked emitter gate also passes 86 checks and all 57 methods,
including the added malformed declarations
(`/tmp/nanolang-is-alnum-final-gate.log`).
