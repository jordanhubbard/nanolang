# Maps through compiler data flow

I now classify and emit `HM_NEW`, `HM_SET`, `HM_HAS`, `HM_LEN` and `HM_DELETE`
for string-keyed maps with integer or string values. Unsupported key/value
forms remain explicit errors. A map shape has separate key and value edges;
calls and branch joins unify these facts instead of treating a map as an array.

Map references have their own temporary storage and high-water tracking. They
flow through locals, duplication, branches, function arguments, normal returns
and tail returns. Mutation preserves shared identity. The emitted runtime copies
keys and string values on insertion; replacement does not increase the count.

Generated entry ownership retains constructed maps and destroys each once after
entry returns. This supports aliases and returned maps without dangling stack
storage, but retains unreachable maps during execution. I filed reclamation as
`task_2f837947d5f24130b401ae433dd8d8c9`; I do not claim bounded retention for
long-running programs.

`HM_GET` remains rejected. Its missing-key result is a void-tagged value, not an
invented integer zero or empty string. Fetched-value ownership and that tag must
survive compiler data flow before I enable it. Map fields inside other aggregates
and broader key/value representations also remain outside this implemented step.

## Verification

`make -j1 test-nvm2c` and `make -j1 test-nvm2c-sanitizers` each pass 1,114
AOT checks and 980 shape checks, with fresh ASan/UBSan instrumentation verified
for the sanitizer run. The opcode case-parity and
sanitizer-driver tests also pass. New execution tests cover both value kinds,
both branch paths, forward factories, aliases, normal and self-tail returns,
replacement counts, presence checks and deletion. Negative tests reject bad key
types, unsupported value types and incompatible inserted values. Graph tests
check key/value facts, map/array separation and invalid projections.

`make -j1 test-one-ir-compiler` advances from `HM_NEW` in function 267 to
`HM_GET` in function 268 at offset 32. Full compiler acceptance remains open.
`git diff --check` passes. Claiming MAC parent
`task_419c47bdc8fc42e4b52eb6af1a0e9a71` still returns `agent_status_unavailable`.
