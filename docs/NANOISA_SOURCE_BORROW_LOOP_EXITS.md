# My exact borrowed-source loop exits

MAC `task_92608ddd872940d2b72e24841b598c31`, under my still-open One-IR
ownership and release-equivalence parents `ed702` and `28f2`.

I admit break and continue only inside the innermost supported while loop.
My source checkers already require each such edge to preserve incoming owner
state. My producers retain that exact state, including pending-disposal
provenance. I do not weaken my verifier's ownership, reference or region joins.

Before a loop transfer, every source owner introduced inside that loop must
have been explicitly consumed. I may drain only holders already marked by a
complete destructive pattern. I close lexical names for the exiting scopes
and preserve outer bindings. A continue targets condition evaluation; a break
targets the loop exit. Nested loops keep separate targets and owner snapshots.
The zero-iteration exit preserves the same incoming owners.

I represent non-fallthrough explicitly so branch joins consider only reaching
arms, without confusing a loop transfer with a function return. Borrow calls
finish their regions before subsequent source statements. Selected shadows
use identical lowering and remain mandatory. Unsupported helper owners,
partial moves, implicit drops and deeper call graphs remain refused.

I require fresh bootstrap, paired C/Stage1/Stage2 instructions, contracts and
lexical names, nested and zero/entered loops, both conditional arm orders,
return coexistence, exact ordinary ownership refusals preserving old output,
and VM/sanitized native values. Existing affine/reference gates remain required.
My source checker parity parent `20048` is completed; this new source-emission
slice does not complete the broader One-IR or release acceptance contracts.
