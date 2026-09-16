# My PR #269 reconciliation

PR #269 has head `287c17c4629c3ac2c772f206769dc460d8f6ee54` and parent
`51760b94`, which is already in integration history. The patch supports flat
record/variant parameters and results; it does not complete aggregate breadth.

Integration's `1d97e312` and successors already provide the intended flat
behavior with monotonic call facts, field conflict rejection, convergence
failure detection and a final classification pass. They retain runtime field
guards, variant tags, recursive results, caller/callee order independence and
control-flow joins. I do not replace them with the PR's bounded-pass field
overwrites or its old fallback of unresolved local kinds to integers.

My existing `test_aggregate_call_facts` covers record and variant parameters,
results and tail-call chains in both function orders, preserving string and
integer fields. The unrepresentable-facts tests reject conflicting fields and
unresolved aggregate construction. I retain one additional PR case explicitly:
an unused parameter receives an integer and a record in separate calls. The
translator must reject that kind conflict even without a field access.

I add the PR head as an actual merge ancestor, retain current production code,
and keep the broader declared-layout task unfinished.

## Verification and limits

`make test-nvm2c` passes 609 checks after adding the parameter-kind case; the
baseline current suite passed 608. The harness compiles and executes generated
C for its positive fixtures and checks translator rejection for the new case.
The final macOS log is `/tmp/nanolang-pr269-final.log`.

This is structured-C subset evidence, not complete VM/AOT equivalence. Nested
aggregates, richer/path-varying field storage, arrays across aggregate
signatures, separately linked metadata and declared-layout coverage remain
open under MAC `task_a4fde0d59ad24fe18c285a76ad58c176`. I do not close that task
because the older PR was reconciled. The PR stays open until integration is
landed on main or explicitly superseded there.
