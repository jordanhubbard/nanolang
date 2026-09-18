# My invocation-scoped owned verification reuse

I track `task_5b3e7272ec3f4ae9ad8eb8ffb15a3565` within the pending bounded
value-call graph prerequisite. The fresh60-second ordinary marker run at
`70dcc94f` completes ordinary/refusal/four-function preambles and eight-function
verification, then reaches artifact generation before its bound. It never
reaches API execution. I preserve its log/status/hash at
`/tmp/nanolang-owned-value-graph-phase.{log,json}`. This localizes phases; it
does not establish repeated verification as the sole cause of earlier timeouts.

## My static repeated-work finding

In `src/nanoisa/verifier.c`, `verify_structure(mod,false)` validates all
function declarations, code ranges, parameter/result declarations, ownership
metadata and other retained contracts. If ownership metadata requires runtime
authority, it analyzes every function and calls `nvm_verify_owned_module`.
Owned admission independently validates structure in affine-only mode, graph,
parameter/local/layout eligibility, every opcode and each function's affine
stack, initialization, region, reference and ownership obligations.

`nvm_verify` first completes that work, then calls `verify_function_impl` for
every function. Each call repeats structure and full owned admission. The
existing approved-owned branch already returns before the ordinary stack/type
walk because affine analysis supplies the applicable proof. I propose reusing
that completed positive admission inside this single public invocation.

## My bounded change

I add a private optional output to structural verification. It starts false
and becomes true only at successful completion after mandatory owned admission
and all remaining structural/metadata checks. Early errors leave it false and
retain their existing failure result. I do not set it from `ownership_size` or
a declaration tag alone.

`nvm_verify` may return its successful structural result when this flag proves
that the complete owned-module contract has already passed. Otherwise it
retains every existing per-function validation. The public single-function
entry path may use the same flag after it independently establishes structure;
it still checks the requested function index, linked-count restriction and
maximum-stack output before taking the existing owned fast path.

A module with nonzero ownership metadata but `needs_ownership=false` is not
approved by this flag. Its existing owned-admission probe and ordinary
per-function fallback remain intact. This distinction prevents advisory or
non-authorizing metadata from skipping ordinary operand, stack or type checks.

Every public verifier call establishes its own proof. I retain all checks in
`nvm_verify_owned_module`, `nvm_verify_affine_function`, ordinary and linked
entry points; linked ownership refusal is unchanged. I neither persist a flag
on a module nor retain a cache across calls, mutations or allocations. During
one verifier call I only read the caller's module; verification invokes no
user callback that mutates it. Concurrent mutation is not a new supported
contract. A later invocation examines the current module afresh.

I preserve failure propagation, output initialization and allocation cleanup.
Removing repeated allocations may change when an injected allocation budget
is exhausted, but never turns an incomplete first proof into acceptance. I do
not change graph bounds, supported opcodes, signatures, dataflow, ownership or
runtime authority. This is verification-work reuse, not new language admission.

## My acceptance

I retain ordinary and owned verifier positives/refusals, single-function
index/max-stack behavior, linked refusal, exact authority and allocation gates.
I qualify fresh corrected ordinary graph fixtures through all four VM APIs and
native targets. The same fully instrumented corpus and bounds remain separate
from the preserved incomplete runs. A short phase-marked check first establishes
whether the repaired path progresses; I do not relabel the old logs or infer a
performance result from the source change alone.

This contract awaits review before production edits. The main graph PR remains
a draft until its qualification is resolved; the unchanged affine example and
owned/void/string/source prerequisites remain open.
