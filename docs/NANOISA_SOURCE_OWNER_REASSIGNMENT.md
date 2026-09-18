# My consumed-local owner reassignment

MAC `task_227b7b89a64d4e768705a9bd2c56430e`, after my exact resource-path joins.

I admit `set destination source` only when destination is an explicitly
mutable, consumed, nonborrowed resource local and source is a live nonborrowed
whole owner with the identical nominal layout. I evaluate the source move
before restoring destination liveness, using existing `OWN_MOVE_LOCAL` and
`OWN_STORE_LOCAL`. The source stays moved; neither slot retains pending
terminal-disposal provenance. I reuse the destination's physical slot and
lexical name interval without publishing a new binding.

My existing exact reaching-arm and loop-backedge checks remain unchanged.
A loop must restore its incoming owner state, including the zero-iteration
path. New branch owners must still be consumed explicitly. Every selected
shadow is lowered under these same rules or publication fails.

I refuse live-owner overwrite, immutable or borrowed destinations, moved or
wrong-nominal sources, constructor assignment, partial field assignment and
helper-owned locals. I introduce no implicit disposal or runtime authority.

I require paired C/selfhost code, ownership/layout metadata and advisory names;
canonical Stage1/Stage2 and selected-shadow acceptance; ordinary VM and
sanitized native execution. Positive cases cover leaf/nested owners, both
branch orders and zero/entered loops. Refusal controls preserve the previous
output and require ownership, type or bounded-emitter diagnostics.
