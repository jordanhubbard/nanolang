# My native variant guard facts

I retain a conservative variant number through aggregate constructors, local
copies, direct calls, returns and tail calls. A pending fact is not a value;
conflicting or untracked producers become unknown. I promote unresolved facts
to unknown and propagate them before the final diagnostic pass. Address-taken
functions have unknown parameter variants because direct calls are not their
complete caller set.

Within a function without backward jumps, an exact integer comparison of a
known variant with a literal can establish that a conditional jump is taken.
Its fallthrough block then cannot execute. I suppress the incompatible scalar
return diagnostic only in that block, and reset this fact at every jump target.
I still classify and emit every instruction, including runtime return-tag
checks. I do not remove branches or treat unrelated aggregate layouts as equal.

This is bounded reachability evidence, not a general control-flow refinement
analysis. Backward-flow functions, unknown callers and guarded blocks reached
through another jump keep the existing conservative diagnostic. A discarded
function reference is conservatively enough to invalidate direct-caller facts.
The accepted alias-return fixture, both equality opcodes, both conditional
jump directions and tail calls must execute in VM/native controls. Reachable
wrong-tag returns, conflicting callers and a jump bypassing the guard must
still refuse without replacing prior output.

## Selected payload layouts

I also track a tag comparison against the same retained operand-stack value.
`DUP` preserves that identity. An equality or inequality branch can select a
constructor on its matching edge; a fresh local load does not inherit the old
stack value's guard. Reassigning that local therefore cannot authorize a
projection from its replacement. Joins discard comparison relationships, and
unknown address-taken callers do not supply missing payload layouts.

For a union storage family containing record payloads and multiple
payload-bearing tags, I retain separate payload shapes under each constructor
tag. A single payload-bearing tag keeps its flat layout beside unit tags.
I propagate this storage choice through copies, calls, returns, globals and
stack joins. A selected projection
uses the corresponding payload view, with my existing managed snapshots and
runtime tag checks. Scalar-only families retain their finite tagged carriers.
A guard can prove a tag before caller facts arrive; it cannot invent that tag's
payload fields. Incompatible fields within one constructor still conflict.

My source controls cover different record layouts, record/string/unit families,
and retained managed payloads across global reassignment and allocation churn.
My raw controls retain wrong-field, wrong-tag, missing-guard and bypass refusals.
This does not establish general local-load refinement, constructor provenance
through every nested aggregate/array, or resource-bearing union ownership.

I retain nested constructor metadata through ordinary record fields and selected
variant payload fields. Copies share representation facts while keeping guard
witnesses tied to their own values. An unknown enclosing producer also makes
its projected constructor facts unknown. A record carrier alone does not choose
between a plain-record shape and a constructor-indexed shape; producer and
projection evidence make that choice.
