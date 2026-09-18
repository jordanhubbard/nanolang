# My ownership-preserving source control-flow acceptance

I track this slice as MAC `task_89c4d7131a664f5a8aadbbbbc4b5389e`, from main
`47375f64` after PR #614. Contract `a9d4fa94` precedes production `579b656b`.
My [bounded contract](../NANOISA_SOURCE_BORROW_CONTROL_FLOW.md) admits actual
if/else branches and while backedges around checked borrowed calls. Existing
scalar locals, owned roots and formal references keep the same authoritative
join state. My runtime, verifier, schema and ownership formats are unchanged.

Both producers require Boolean conditions and emit the same conditional
branches and jumps. Calls in conditions end their regions before branching;
calls in bodies end before the following branch or backedge. I refuse local
declarations, construction, destructuring, moves and early exits within those
bodies. Neither expression emitter emits ownership transfers: helper actuals
must be explicit borrows of exact live root-local paths. I retain the existing
source ownership checker and final affine verifier; I introduce no implicit
drops, dummy initialization or relaxed joins.

I measured a fresh default three-stage bootstrap on the new production source:
`/tmp/nanolang-source-control-bootstrap.log`. All fifteen paired source methods
then passed in 162.101 seconds in `/tmp/nanolang-source-control-paired.log`.
This includes unchanged source-borrow controls, exact C/raw-selfhost/Stage1/
Stage2 canonical dumps and selected-shadow dumps, stripped-name controls and
VM plus ASan/UBSan/LSan native execution. The target also passed 123 lexical
name checks, five name-allocation and twenty marker-allocation boundaries.
I reused the completed bootstrap with `make -o bootstrap`; I skipped no test.

The new exclusive fixture executes both branch directions, no-else branches,
zero-iteration loops, nested loops and helper-body control flow. It observes
mutations through nested caller paths. A helper call in a loop condition also
executes on the final failed condition check; post-loop values distinguish
that behavior. Selected shadows exercise the same operations. Depth 32 is
accepted and depth 33 is refused without replacing output.

Twelve new refusal cases cover branch/loop local initialization, attempted owner
moves/destructuring, returns, break/continue, non-Boolean conditions, changed
scalar assignment types and a false selected shadow. Canonical outputs remain
unchanged on refusal; raw producers refuse the structural cases. The old
blanket loop refusal is replaced by a loop-local-initialization refusal because
ordinary loops are now deliberately supported.

A subsequent test-only addition at `e75c7a88` adds shared-path calls, Boolean
loop state and read-only post-loop owner values. I rerun that positive method
explicitly because the earlier test process had already loaded its harness.
Broader path-sensitive owner movement, lexical initialization and early-exit
resource flow remain outside this slice and do not become complete here.
