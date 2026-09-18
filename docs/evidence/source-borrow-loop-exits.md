# My source-borrow loop-exit evidence

MAC `task_92608ddd872940d2b72e24841b598c31`, PR685.

I recorded my contract at `c0771b4c` before implementation. My paired producer
checkpoint is `f7e0d728` (C change `200dc511`); later changes are documentation.
My current ledger audit found source-checker parity parent `20048` completed,
while full One-IR ownership `ed702` and release equivalence `28f2` remain blocked.
I do not equate this bounded producer slice with those full acceptance gates.

I passed `make -j4 test-source-borrow-emission` from a fresh isolated tree:
bootstrap completed and all 24 paired methods passed in 282.741 seconds. The
new loop-exit method exercises both branch choices, nested innermost targets,
zero iterations, helper mutation, explicit transient owner consumption and
return/break coexistence. C, both selfhost-produced emitters and both canonical
stages retain exact dumped code, layouts, ownership and local-name metadata.
Names stripping and mandatory shadow modules preserve VM/sanitized native
execution. My existing loop-control refusals now require changed ownership;
I do not retain an obsolete blanket refusal for supported break/continue.

I also passed `make -j4 test-affine-bytecode test-owned-assertions`: 441 and 751
affine bytecode checks plus 959 owned assertion lifecycle checks. These gates
retain exact owner/reference/region verification and terminal cleanup.

I retain `/tmp/nanolang-borrow-loop-exits-gate.log` and
`/tmp/nanolang-borrow-loop-exits-authority.log`. No production source changed
during qualification, no verifier/runtime authority changed, and no historical
failed compiler artifact was replayed. General helper-owned locals, partial
moves, deeper calls and implicit source disposal remain outside this contract.
