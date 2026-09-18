# My portable mutable-array eligibility analysis

I prepare matched mutable-array lowering under aggregate parent488 and managed
parent51da. This child supplies a read-only, non-admitting analysis. My ordinary
verifier and current LLVM/Wasm profiles keep their existing behavior. A passing
analysis report does not authorize an opcode, provide an ownership runtime, or
establish full collection support.

## My source of truth

I inspected `src/nanovm/heap.c` (`vm_array_new`, `packed_store`, `packed_load`,
`vm_array_push`, `vm_array_set`, `vm_array_get`, `vm_array_pop`) and the matching
handlers in `src/nanovm/vm.c`. Creation fixes the VM's packed-versus-boxed
representation. Only int, U8, float and bool use packed storage. Other declared
kinds use boxed values; their declared kind does not restrict later child tags.
Push retains a boxed child, set transfers it, get borrows it before the opcode
retains, and pop transfers it. Get/pop may return void. I preserve these rules
in the later adapter contract rather than treating my private borrowed-set API
as an already matching opcode.

My existing `verifier_types.c` deliberately widens tags to unknown and does not
track local/global origins. My `nvm2c_shape` graph solves native representation
constraints; its exact-kind unification is not a mutable VM origin analysis.
I keep both unchanged and build a separate private analysis API.

## My abstract state

Each value has a finite set of possible tags and a set of array allocation-site
origins. Unknown/unresolved is distinct from the empty, not-yet-reached state.
A tag saying array without authoritative origins is unresolved, not a boxed
leaf array. I do not turn a signature's `TAG_ARRAY` into a storage proof.

Each origin identifies a function/instruction allocation site and retains its
fixed declared kind and storage class. ARR_NEW int/U8/float/bool creates that
packed class; other supported leaf declarations create boxed storage. STR_SPLIT
creates boxed string storage with initial string children. I retain declared
kinds separately from content tag sets, including void/string/enum declarations.
My first prospective constructors admit only leaf declaration tags; nested or
nominal declarations remain unresolved even when currently empty.

For boxed origins I accumulate every possible written child tag monotonically.
I never reset that summary on a second allocation at the same site. Distinct
runtime objects allocated at one site share a conservative summary: this may
reject a correlated program, but cannot erase a possible write. Every alias
carries the origin set, so mutation through any local/global/call alias affects
subsequent reads through all aliases. I use weak summary updates, not a claim
that the abstract site denotes exactly one runtime object.

CFG joins union tags and origins. Locals begin void. Globals begin void and
accumulate all possible writes, including writes before a later runtime error.
I seed the first initializer and entry selected by the existing module contract;
entry re-execution retains global summaries. I union their effects and do not
claim a must-write initialization or precise call-order proof. The later runtime
adapter must preserve the actual initializer-before-entry order. I conservatively include both
initial and later states rather than assuming a fresh instance on each call.

Direct calls propagate actual argument facts into context-insensitive callee
parameter summaries and propagate result/effect summaries back. Recursive SCCs,
loop backedges and caller/global dependencies iterate to a fixed point. A new
possible argument or write requeues affected analyses. I do not evaluate final
eligibility until summaries stabilize; provisional bottom facts prove nothing.
Uncalled functions with unproved array parameters cannot receive an eligible
mutation report. Unknown calls, imports, closures, host inputs and unsupported
escapes remain unresolved. This child has no linked-module analysis.

## My transfer and eligibility obligations

| Operation | Required abstract result/effect |
| --- | --- |
| ARR_NEW | Fresh site identity with fixed declared storage; boxed contents start empty |
| STR_SPLIT | Boxed string origin; existing string children remain possible |
| DUP, local/global load/store, call/return | Preserve and conservatively combine origin sets |
| ARR_PUSH / ARR_SET | Check every possible receiver origin and input tag; add boxed child tags; result aliases the receiver |
| ARR_GET / ARR_POP | Packed declared tag or boxed accumulated child tags, always union void; pop never narrows shared content summary |
| ARR_LEN | Integer result; no child-shape change |
| Scalar/string instructions | Explicit successful-result tag transfer; unsupported or unknown transfers never become an invented exact scalar |

For packed writes, every possible pair must belong to the reviewed finite
matrix: exact canonical tag, U8-to-int, int-to-U8 modulo256, or int-to-float
under default nearest-even rounding. I retain exact matching float payloads
and never classify raw mismatched union-member reads as portable coercion.
For boxed writes, every possible child must be void/int/U8/float/bool/enum/string,
regardless of the array's declared boxed kind. I do not reject a heterogeneous
leaf write merely because it differs from a declaration or previous value.
Array/record/map/closure children remain unresolved until tracing and cycle
semantics qualify; a tag union containing any such possibility is unresolved.

Receiver/index tag uncertainty is reported explicitly for later adapter policy.
It is not silently discarded as an impossible path, nor converted into a new
ordinary-verifier error. This analysis does not establish bounds or suppress
existing runtime wrong-tag/missing-index/error behavior. Correlation-sensitive
branch refinement and strong updates are outside the first implementation.

## My API, limits and integration boundary

I will add private `managed_array_shapes.c/.h` with a result that distinguishes
eligible, unresolved, invalid ordinary bytecode, and analysis allocation/limit
failure. Diagnostics identify function/PC and the unmet origin/tag obligation.
The caller owns a successful report; destroy handles partial construction. The
analysis does not mutate the module or write executable output.

The first implementation bounds origins at 64, functions at 256, each function's
locals/stack at 256, global slots at 256, total decoded instructions at 65536, and
stored abstract-state cells at 1048576. Checked multiplication precedes allocation.
Exceeding a cap is an explicit analysis-limit result, never eligible. These are
private analysis limits, not a new language or VM limit. Finite monotone tag and
origin sets bound convergence; queued flags bound the worklist. I publish no
partial successful report after allocation or validation failure.

A later reviewed adapter must combine this report with structural verification,
matched ownership/status cleanup, generic GET on promoted split arrays, exact
VM initial capacity/growth and set/pop transfer semantics before any opcode
admission changes. This report alone does not widen `nvm_verify_profile`.

## My required evidence

I require analyzer tests for each packed matrix pair, boxed mixed leaves,
fixed storage despite content joins, local/global aliases, different origins,
repeated same-site allocation, branches, loop backedges, reordered call sites,
recursive SCCs, helper results and initializer/reentry effects. Negative
eligibility cases cover unproved origin, unsupported packed pair, nested child,
escape/unknown transfer and explicit analysis limits. I use ordinary checked
source fixtures and static reports, not execution of raw mismatched union paths.

Selected eligible programs run in the VM as reference controls. My standalone
analysis API and tests run with native sanitizers and preserve input modules.
Existing verifier profiles and prior-output refusals remain unchanged. Existing
managed package/string/split/core gates must pass. I audit explicit build source
lists before linking any new API into existing consumers; the first private
report test can link it directly without adding a production dependency.

I keep raw-write semantics, nested/nominal tracing, cycles, adapter/lowering
acceptance and full applicable-language coverage open under parents488/51da.
Darwin sanitizer7ba and evaluator791a remain distinct historical obligations.

## My consuming profile integration

My analysis API still reports shape evidence only. The mutable-array extension
now calls it from CLOSED_MANAGED_STRINGS after ordinary/profile verification.
That consumer admits only its separately implemented instruction helpers; a
successful report never removes runtime type, bounds or ownership checks.
