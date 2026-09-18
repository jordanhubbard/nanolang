# I discard underscore payload patterns

I treat `Variant(_)` as a discarded payload, with no new binding. An ordinary
outer `let _: int = 41` remains visible in the arm; without an outer name, `_`
is undefined. Wildcard arms and ordinary underscore declarations keep their
existing behavior. Resource-bearing payloads still require explicit consumption.

My C checker and interpreter now skip discarded bindings. My C expression
emitter and self-hosted native declaration/environment agree with the existing
NanoVirt and self-hosted checker discard behavior. The declaration helper has
an inline shadow. The repeatable target is `make test-underscore-payload`.

At production source `fbb4c898`, a fresh default three-stage bootstrap passed.
Nine new/adjacent underscore methods passed in 30.819 seconds, exercising the
interpreter, C seed, Stage1, Stage2 and NanoVirt/VM. Three adjacent ownership
methods passed in 7.645 seconds: ignored resources and wildcard-owned payloads
remain rejected, and ordinary selected payloads still execute. Guarded arms are
checked on the C/interpreter/VM route; this does not add guarded selfhost syntax.
The undefined-name checks also require the name diagnostic and preserve output.

I retained these initial failures: the first runner omitted `bin/nano` from its
build; the expanded source used direct selected-union constructors that Stage1
and Stage2 reject despite C/interpreter/VM acceptance; the first adjacent runner
named a nonexistent unittest class. Corrected explicit typed union locals pass.
The separate direct-constructor context issue remains open as
`task_6961296c51014326bb3a532c33fab2e0`; it is not an underscore-pattern repair.
Logs are `/tmp/nanolang-underscore-*.log`. I made a final whitespace cleanup and
removed a redundant C condition without changing the emitted behavior.
