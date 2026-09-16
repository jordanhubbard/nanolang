# My ownership observation boundary

I retain the normative contract in `AFFINE_TYPES_DESIGN.md`. Observing an
ordinary field does not resolve the resource containing it. Returning the
owner itself transfers it. Passing it by value to a declared foreign consuming
operation transfers it across a trusted boundary; that declaration does not
prove the foreign implementation performs cleanup.

## What my review found

The recovery prototype `c53e27a7` is not safe to merge as a complete checker:

- Its 256-place table silently stops registering owners.
- Its expression walker ignores several AST kinds.
- A resource initializer checks its source but does not move that source.
- Function parameters are not registered as owned places.
- Its independent recursive type classifier loses my newer fixed-point,
  union and module-owner classification.

My existing self-hosted checker has a different defect: field access calls
the identifier-moving routine. It therefore accepts abandoning a parameter
after reading its field, and rejects reading a field before returning the
still-owned parameter. The former `valid_move.nano` fixture depended on this
mistake: its apparent `close_file` only returned `file.fd`. I rename it to
`unresolved_callee.nano` and require rejection rather than weaken the contract.

## What I change

My self-hosted checker now separates field observation from moving an owner.
Observation checks an already-moved place without adding a moved state.
Calls evaluated while computing a field receiver still visit their arguments.
The helper shadow checks both a live observation and a moved-place diagnostic.
This is not complete expression, branch, loop, borrow or aggregate analysis.

I add seven source-only conformance probes, each run through C seed, Stage 1
and Stage 2. They check returning an owned parameter, moving to a declared
foreign consumer, observing then returning, observing then abandoning,
an unused parameter, observation after move, and an unresolved 257th owner
after 256 resolved owners. Rejections require ownership diagnostics and
unchanged prior output. Successful C emission is not runtime cleanup evidence.
These declaration probes intentionally do not execute shadows or link the
foreign operation. Positive and negative ownership checks remain explicit;
there are no expected-failure annotations.

`tests/test_affine_selfhost.sh` includes both the native-output rejection
fixtures and the new declaration probes. The separate target
`make test-affine-contract-boundaries` runs just the declaration probes.

## My verification

My fresh bootstrap passes, with differing native stage binaries still reported.
All 21 `test-one-ir-compiler` methods pass, including native compiler-to-hello
execution. The expanded ownership gate checks 30 compiler/case combinations:
all ten Stage 1 and all ten Stage 2 cases pass. The C seed passes three positive
declaration probes and fails seven required rejections. One rejection fails
only in shadows; the others incorrectly publish output. This is a failing
release gate, not full ownership conformance.
The installed self-hosted compiler also passes all 17 language cases through
`bash tests/run_all_tests.sh --lang`.

Logs on this host: `/tmp/nano-affine-observation-bootstrap.log`,
`/tmp/nano-affine-observation-aot.log`, and
`/tmp/nano-affine-observation-gate.log`. I retain the earlier eight-failure
declaration baseline in `/tmp/nano-affine-contract-boundaries.log`.

The older `test_resource_tracking.nano` and `test_affine_integration.nano`
also contain print-only close helpers or direct resource-field consumption.
They need contract review; passing those legacy examples is not evidence of
deterministic resolution or whole-owner enforcement.

## Ledger boundary

MAC reports the earlier task `task_91ae827be4154eaa8f22698aeecc8cf1` as failed
after a worker verification-contract failure on September 16. It has no active
owner or replacement. I continue the unfinished contract under
`task_c60a8d2e14b7494f8875e75b16e9b087`; I do not reinterpret the terminal state
as completed product work. My session worker is offline with an existing
dispatch hold. A heartbeat succeeds but does not clear either state. I leave
the hold unchanged and record evidence without claiming ownership.
