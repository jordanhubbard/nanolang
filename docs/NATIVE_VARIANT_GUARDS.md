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
