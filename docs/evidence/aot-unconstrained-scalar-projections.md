# I preserve tags on unconstrained scalar projections

My native compiler aborted in `parser_is_at_end`: a token field had integer
storage kind 0 and payload 24, but its projection checked for kind 2. Kind 2 is
my internal unknown marker, not runtime storage. Generic comparison correctly
does not constrain both operands to the same type; guessing integer storage
would make heterogeneous comparisons incorrect.

I now read unresolved scalar fields as tagged values. I accept native integer,
boolean and string storage, and tagged optional scalar storage. I check bounds,
supported storage, optional tags and non-null string pointers. Unsupported
aggregate storage still traps; this is not a universal aggregate boxing ABI.
Existing projections with known storage keep their exact checks.

I mark roots of unresolved projection shapes after solving the graph. Local
copies sharing those roots use tagged storage instead of the integer fallback.
I do not change those shapes into optional constraints or weaken exact type
unification. My regression covers two function orders and direct/local-copy
paths, with seven valid scalar cases and five invalid cases per combination:
28 successful runs and 20 required traps.

## I checked these gates

- `make test-nvm2c`: 1,670 AOT and 1,073 shape checks pass.
- `make test-nvm2c-sanitizers`: fresh ASan/UBSan objects pass the same checks.
  Leak detection remains disabled by that gate.
- `make test-one-ir-compiler`: 17 of 18 methods pass. The full compiler builds
  and runs `--help`, but compiling hello still fails.
- `git diff --check`: passes.

Fresh standalone emission and a retained run of the exact compiler acceptance
method both get past the original projection check. LLDB reports a stack-probe
fault in `nl_parser_with_position`, reached through `nl_parser_advance` and the
expression parser roughly seventeen frames deep. The host stack limit is
8,372,224 bytes. Both debug and non-debug O0 builds reproduce the stack failure.
The full suite logs SIGABRT while the isolated method logs SIGSEGV; I do not
claim to have reconciled that distinction. Neither result passes acceptance.

I track this fix in MAC `task_45fedd409e1447089dad970396b0a075`, and the next
native execution work in `task_e81212c768d148639429d1d2be9f826d`. Claims remain
unavailable for my registered worker; I preserve evidence without force-closing
tasks. Compiler correctness, stack bounds and release readiness remain open.
