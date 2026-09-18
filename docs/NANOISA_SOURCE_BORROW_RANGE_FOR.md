# My owned-source range loops

MAC `task_212a2ec1f7374734b90a23abf00a3211`.

I admit direct builtin `range(start, end)` in my closed
source-borrow for profile. I evaluate exact integer bounds once, start then
end, before introducing the immutable lexical index. I iterate upward with
an exclusive end; start greater than or equal to end means zero iterations.
This matches my existing interpreter/native range-loop specialization.

I use scalar bound/index slots and existing verified comparison, addition
and jumps. I do not admit general array iteration. I introduce no trapping
increment: an entered iteration proves index < end <= INT64_MAX, so adding
one is representable. Continue performs that increment once; break skips it
and exits the innermost loop. Nested loops retain separate targets.

I preserve exact incoming owner/disposal facts, explicit consumption of local
source owners, terminal-holder cleanup and lexical names/restoration. Bounds
may use already-supported borrowed calls, preserving their evaluation order.
I refuse qualified/shadowed user range calls, noninteger bounds and unsupported
iterables. Selected shadows remain mandatory; helper owners, deeper calls and
implicit drops remain outside my profile. No runtime/verifier authority changes.

I require fresh bootstrap, paired C/Stage1/Stage2 code/contracts/names,
zero/entered/nested loops, endpoint effects and lexical shadowing,
break/continue/return coexistence, ordinary refusals preserving output, and
VM/sanitized native parity plus existing affine/lifecycle checks.

My live ledger has completed bounded source parity `20048` and recovery
`c60a`; that status does not establish their broad historical descriptions or
complete the normative ownership contract. One-IR `ed702`, borrow `718` and
release-equivalence `28f2` remain open. I do not replay historical failures.

My direct range identity is checked within the existing closed declaration
profile: no imports, global lets, externs or additional functions are admitted.
Both producers also refuse a local/formal named range, a sole helper named
range, qualified/indirect callee syntax and borrowed-call syntax here. I do not
infer builtin identity from an arbitrary imported or shadowed printed name.

My first full gate passed 25 of 26 methods but rejected the new one-bound
fixture with E003. My initial contract incorrectly inferred checked-source
support from raw lowering. The quick reference, registered builtin and
interpreter loop contract require two bounds. I retain that log at
`/tmp/nanolang-borrow-range-for-gate.log`, correct this slice to two-bound
source grammar, and preserve one-bound refusal. Cross-layer arity audit
`task_a1c30f5fe3a740c9906e150200d05af8` remains separate; I do not broaden the
checker or claim the raw extension is a supported source form.
