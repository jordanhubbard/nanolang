# My self-hosted native range bounds

I evaluate both range bounds once, in source order, before introducing the
loop variable. My self-hosted native transpiler previously placed the end
expression directly in the generated C condition. Both Stage 1 and Stage 2
failed the retained `order == 12` shadow when that expression had effects
(`/tmp/nanolang-selfhost-range-bound-before.log`). C seed already passed
through the preceding range repair.

My generated loop now uses two scoped snapshots. Source locals are emitted
with the `nl_` prefix; the internal `__nl_range_*` names cannot capture them.
The fixture explicitly binds source variables named `__nl_range_start` and
`__nl_range_end`, and nested loops use bounds referring to an outer variable
with the same name as their own binder.

`make test-selfhost-range-bounds` builds the three native compiler stages,
then checks C seed, Stage 1 and Stage 2. The positive fixture retains effectful
bounds, zero-iteration cases, nesting, break/continue, array-loop return and
lexical scope. The negative fixture deliberately fails its bound-order
shadow and must preserve a previous output file in all three compilers.

The fresh three-stage bootstrap and both test methods pass; the six compiler
cases take 14.274 s (`/tmp/nanolang-selfhost-range-after2.log`). This run uses
the explicit per-command `NANO_SHADOW_TIMEOUT_SECONDS=60` budget recorded by
the preceding range slice. Deadline defaults remain unchanged. I do not
infer native binary equality from the bootstrap stamp.

The original broader emitter fixture remains unchanged. It includes dead
statements after return, which my self-hosted checker currently reports as
W0002 through an error constructor. That independent severity defect is
`task_47e61dea383042808ccc1a0c89bef064`; its failure log is
`/tmp/nanolang-selfhost-range-before.log`. The focused native fixture isolates
the bound defect without deleting assertions from that original test.

After integrating canonical main `28644fc3` (including the range emitter and
borrow-annotation schema), the fresh three-stage bootstrap and both methods
pass again in 13.548 s with the same explicit shadow budget
(`/tmp/nanolang-selfhost-range-integrated.log`).
