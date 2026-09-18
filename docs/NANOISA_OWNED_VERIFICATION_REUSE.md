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

The first reviewed production checkpoint is6456be51. A fresh60-second marker
run completes artifact generation and reaches API0, improving progress without
completing qualification. Further static inspection shows vm_init calls
vm_recompute_verified, which uses nvm_verify_linked with zero linked modules.
That entry still repeats admission. My reviewed companion lets that entry use
its invocation's completed positive proof only when linked_count==0. Its
linked_count>0 refusal and ordinary/advisory fallback remain unchanged. I add
explicit zero-link positive/refusal/fallback controls before final qualification.
I preserve the superseded partition as interrupted, with no passing claim.

Both invocation-local reuse contracts are reviewed before their source edits. The main graph PR remains
a draft until its qualification is resolved; the unchanged affine example and
owned/void/string/source prerequisites remain open.

My next diagnostic checkpoint keeps production at 246f1464 and preserves all
previous runs. I timestamp each captured stderr line with the runner's monotonic
elapsed time. A separate generated copy of vm.c adds only markers at API entry,
frame preparation, dispatch entry, and around its owned/linked verifier calls.
The wrappers return the original result unchanged. My fixtures, assertions,
argument order, and production source remain unchanged. I run only ordinary
case 0 once with a 60-second bound and preserve its source, binary, hashes, log,
and status under /tmp/nanolang-owned-timed-diagnostic. This localizes phase cost
and completed call counts; it does not establish a new proof-reuse contract or
qualify the full instrumented corpus.

## Public invocation runtime proof proposal

I propose this runtime change only after the timed diagnostic at f78705c4 and
its recorded outcome at 749ffb27. I have not implemented it. The investigation
remains task_5b3e7272ec3f4ae9ad8eb8ffb15a3565, a prerequisite of my graph task.

I keep a private proof on the C invocation stack, never in VmState, NvmModule,
a global, a reference activation, or a persistent analysis cache. A proof is
initially absent. I establish it only after a fresh complete successful
nvm_verify_owned_module for the exact current standalone root module, preserving
ownership-contract validation and the existing needs=false advisory fallback.
A nonzero ownership section, vm->verified, or a previous successful invocation
cannot establish it. Unsupported ownership still fails closed.

My eligible invocation has module==root_module, zero linked modules, no callback
runtime, no opcode tracing, and no active inherited reference activation. Its
module storage remains alive and immutable throughout the synchronous call,
as required for execution of decoded instructions. Ordinary value-stack and
heap changes are not module mutations. I do not claim thread synchronization or
support concurrent mutation. I perform no module-mutating operation while this
proof is usable. Callable entry additionally requires its resolved target to be
the same module; another target uses the existing independently checked path.

I pass this private proof explicitly through private invocation/core helpers.
The unchanged public vm_core_execute always starts without a proof and performs
fresh checks, including when a host separately resumes it after a yield. I do
not change its signature or expose a trust token to callers. The private core
may skip only the repeated full module admission, using the already established
positive fact to select owned execution. It retains every live activation,
frame, generation, origin, region, descriptor, operand, and transfer check.
Before each private use I require the same root/current module, no links, and
no callback runtime or tracing. A mismatch discards the proof and uses fresh
existing checks; pointer equality alone never creates a proof.

I preserve the existing public API boundaries as follows:

| Entry | Proof and preparation boundary |
|---|---|
| vm_call_function | Fresh admission after existing host-entry constraints; private checked call receives this invocation's proof. |
| vm_invoke | Fresh admission before argument preparation; private call path may receive it without repeating admission, while retaining host-entry, argument, frame and activation checks. |
| vm_invoke_callable | Fresh admission and exact callable target check; only the same standalone root can receive the proof. Target switches retain the old independently checked path. |
| vm_execute | Existing entry/initializer selection remains; its call_function invocation establishes a fresh proof. Owned admission excludes __init__, so no proof is carried between initializer and main. |

I retain fn_idx/arity/local-count checks, argument snapshot and retain order,
stack/frame reservation, activation floors, output initialization, call-depth
bounds, and complete positional owned-call preflight. Frame preparation does
not execute user code in this eligible path. I neither memoize argument checks
nor reuse a callee reference context. Each actual call still assigns a fresh
monotonic generation after successful preflight; return clears that frame's
context, and equal frame/local slots in sibling calls remain distinct.

My complete core/trap audit gives the following scope rules:

- The only production caller of vm_core_execute is vm_call_function_impl;
  direct test/host callers retain its public independently checked behavior.
- OP_ASSERT always returns TRAP_ASSERT. A true assertion releases its scalar
  condition internally and resumes the private core with the same proof. It
  performs no callback, module change, output operation, or host return.
- A false assertion and TRAP_ERROR end this invocation; the existing outer
  cleanup clears contexts and releases actual owners. No proof survives.
- TRAP_NONE and TRAP_HALT end the invocation, with existing return cleanup.
  No proof survives successful return either.
- TRAP_YIELD currently arises from the callback budget. Callback-enabled
  invocations are ineligible. Before any callback pump, I discard any proof;
  an unexpected yield also discards it before continuing through existing
  behavior. I do not carry a proof across host-visible core resumptions.
- TRAP_EXTERN_CALL and TRAP_PRINT are excluded by owned admission. Their
  generic handlers remain unchanged and discard any proof before external
  dispatch or output. No proof crosses FFI, callback, output-hook execution,
  module resolution that changes the target, or a recursive public API call.
- Internal owned CALL/RET stays inside the admitted complete graph. Runtime
  argument/layout checks, exact exit consumption, per-frame reference origins,
  context generations and failure cleanup remain unchanged. CALL_REF keeps
  its separate admitted profile and the same live authority checks.

I will qualify all four APIs with successful repeated true assertions, false
assertion cleanup, repeated invocations and sibling same-slot generations.
Private test instrumentation will count admissions so separate public calls
must establish separate proofs. I will retain direct-core yield/resume and
callback/ordinary execution gates. Static verification controls will check a
changed module between invocations without executing a refused module. Existing
allocation/preflight controls must continue to pass. The next timed corrected
case will use a new executable and artifact directory; all earlier 60-second
artifacts remain untouched. Full instrumented qualification remains separate
and incomplete until its unchanged corpus actually completes.
