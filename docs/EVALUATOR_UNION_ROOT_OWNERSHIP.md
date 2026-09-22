# I retain evaluator-created union roots until Environment teardown

I extend task992713's source audit before implementation. The original495 report
has five direct48-byte create_union stacks: staged argument and two shadow-local
routes in match_wildcards_follow_lexical_order, handler_return_expression_order,
and union_types. These share the two constructor routes in eval_expression.
The third production constructor is builtin_result_map in eval/eval_io.c. Raw
create_union also has direct callers in test_env_scoping and the File fixture;
those callers retain their explicit ownership and must not be adopted by lookup.

I propose a separate Environment registry of newly constructed UnionValue roots.
An evaluator-specific construction helper constructs once, then publishes its
unique root; failure to publish frees only that unpublished constructor's owned
storage and refuses. I register at each of the three actual construction sites,
not at arbitrary symbol assignment, match projection, unwrap or parameter binding.
Aliases keep the same pointer; scope exit and assignment do not shorten its life.
Environment teardown retires each registered root exactly once.

The constructor owns its root, union/variant names, field-name strings, field
vectors and copied STRING/STRUCT/TUPLE payload storage. It borrows other payload
leaves, including nested union roots, arrays, opaque pointers and callables. The
registry destructor follows precisely those existing clone boundaries; it does
not recursively free borrowed union or callable pointers. Existing record/tuple
snapshot destruction already preserves these borrowed leaves. I must verify
create_struct's nested copying matches that destructor before implementation.

I keep this registry separate from record_result_index. Simply adding VAL_UNION
to that index would make call_function's independently owned record-return route
invoke the existing shallow union clone and falsely claim independent ownership.
Current public union returns and union leaves remain Environment-borrowed, as do
other pre-existing reference leaves; no lifetime beyond Environment destruction
is newly promised. Pending task leases still prevent Environment destruction.
I will audit task clone/drop and returned composite users against that boundary.

Direct empty-literal match cleanup must exclude registered roots while preserving
its existing raw-owned helper controls. The terminal worker's exact raw cleanup
must similarly leave registered roots for Environment teardown. Neither change
can make an unknown borrowed root owned. Existing File fixture direct constructors
and explicit shell cleanup stay unchanged.

Controls must cover repeated aliases and reassignment, nested unions sharing one
child, copied string/record/tuple leaves, borrowed array/callback leaves, returned
unions across local scope, both parsed constructor routes and result_map. Registry
publication failures must leave the Environment unchanged and preserve recovery.
Raw caller-owned constructor controls and exact terminal fatal diagnostic remain.
No affected execution precedes source and fixture review. This repair alone does
not close the other original full evaluator leak stacks or the complete SDK gate.

## I preserve completed task graph owners separately from arguments

The initial pending-lease argument was insufficient: coro_run drops its argument
bundle immediately after storing DONE, while the result remains readable until
explicit release. A shallow union, array or nested callable leaf can therefore
outlive that argument lease. I do not activate registry destruction on that basis.

I propose an additive contextual-owned spawn API, preserving legacy spawn and
spawn_owned signatures and behavior. The new API accepts a separately acquired
result-owner token and trusted drop hook. Failed enqueue transfers neither bundle
nor token. The scheduler stores the owner independently of arg and never delays
coro_drop_argument. Evaluator enqueue acquires a second Environment lease for the
result owner; every failure releases that extra lease while preserving the
existing caller-owned bundle failure path.

For successful DONE (including early complete), result-owner lifetime lasts
through nano_coro_release. Release keeps the existing active latch, clears stored
hooks before calling them, drops the owned result first and releases its graph
owner afterward. The same owner covers result_clone/result_copy while their
active latch forbids release. A pending cancellation has no result: drop argument
then result owner under the active latch, preserving existing cancellation
controls which permit Environment destruction immediately afterward. An ERROR
callback similarly drops its returned temporary while the owner is live, then
argument and owner, because no completed result is published. Early completion
followed by callback return retains its original completed result and owner;
only the ignored return is dropped, under the existing rule.

This token protects every Environment-borrowed leaf, independent of top-level
tag: union, array, opaque/reference or callable leaves in records/tuples as well
as direct values. It does not falsely deep-copy those leaves. A copied result's
owned record/tuple/string/callable storage remains caller-owned, but borrowed
leaves still require its Environment to remain alive; the new API must document
this distinct contract rather than inherit spawn_owned's blanket independent
copy wording. Immediate await already copies into the same live Environment
before release. Cross-Environment publication requires an explicit retained
owner, not merely a successful shallow clone, and remains a consumer audit gate.

Additive controls must observe unchanged argument-drop timing, completed owner
retention, result-drop-before-owner order, clone failure without owner loss,
early complete/error, active-hook reentry, pending cancellation, stale release,
full queue/failed enqueue recovery, and same-Environment immediate await. Actual
union/callable/array-leaf results must stay readable after argument drop and must
prevent Environment destruction until completed slot release. Existing scheduler
and lease assertions remain intact. No runtime execution precedes source review.

## I authorize evaluator result access by a retained Environment identity

I explicitly close the previous implicit cross-Environment gap: evaluator
coro_result and await must verify task ownership before asking the scheduler to
clone or run the target. Evaluating the handle expression once is necessary;
only after that expression returns a handle do I check its owner. Foreign or raw
C tasks refuse with a precise evaluator diagnostic. Raw C scheduler result,
await, spawn and owned-spawn APIs retain their current behavior. Direct async
calls create and await their task in the same Environment and pass this check.
No cross-Environment graph transfer is claimed.

A raw Environment pointer is insufficient for identity after scalar completion:
its address can be reused. I propose a separately allocated reference-counted
identity token, retained by its Environment and each contextual task. Tokens are
never reused while a task retains them. The task compares its token with the
current Environment token; it need not dereference a destroyed Environment.
Environment destruction releases its token reference, not task references.
Failed preparation leaves no published task/token or leaked lease. Reference
count overflow is checked before transfer.

The original test_eval_coroutine_spawn_and_run includes scheduler_run in source,
but run_ctx_init may only register main: it does not prove an actual scalar DONE
transition. I retain that test and add an actual checked call which observes DONE
before freeing its Environment, without first releasing its scalar task handle.
Existing generic owned-scheduler controls also require argument leases to end
at completion; those controls keep using the unchanged old API. I will preserve
both behaviors. Contextual tasks retain their identity token until release, but
may relinquish the separate result lease at completion only with a narrow proof:
actual returned value is scalar and the snapshotted declared result type is the
matching INT/U8/ENUM, BOOL, FLOAT or VOID category. TYPE_OPAQUE or absent/unknown
annotations never gain this proof merely because their runtime tag is VAL_INT.
All other results conservatively retain the lease through disposal, including
arrays, unions, callables and composite values. No recursive graph walk is needed
for early lease release. This may retain leaf-free records longer; their explicit
release contract is documented instead of asserting they are scalar.

A trusted contextual settlement hook runs after callback completion and temporary
return disposal, before argument drop. It can relinquish only the extra result
lease; identity remains. Cancellation and ERROR similarly relinquish the lease
without releasing the identity needed for later handle rejection. Final owner
drop releases a live result lease if any, then the token reference, without
accessing a former Environment after an early scalar lease release. DONE graph
results drop their result while the lease remains live, then release the owner.
The active latch protects every hook and publication order.

Additional controls retain original scalar teardown and old-API tests, then
check same-Environment access, foreign READY refusal without running callback,
foreign DONE refusal without cloning, raw-task refusal only through evaluator,
identity survival after scalar owner destruction/address reuse, and explicit
release of graph results. Copy failures preserve output and retained ownership.
These are the final proposed contracts for review; implementation and execution
remain held until review, and full original leak closure remains separate.

## I checkpoint the task lifetime prerequisite separately

My first implementation adds only contextual task lifetime/identity support; it
does not activate union registry teardown. I snapshot the selected function's
coarse return type before recursive argument evaluation, avoiding a new fallible
lookup/cache publication during enqueue. Ordinary owned scheduler APIs remain
unchanged. The parsed control actually checks and invokes scalar through the
real bundle, observes DONE and value7, frees its Environment before task release,
and proves a new Environment cannot match the retained token. A string task
conservatively retains its lease through release. Fresh process refusal controls
retain normal atexit cleanup and require exact diagnostic/exit1 for foreign READY,
foreign DONE and raw C task access through evaluator. READY refusal checks that
the actual checked scalar's global counter stayed zero.

Additive owning controls keep the original allocation sweeps and scheduler cases,
exercise a record containing borrowed union/array and Environment-owned callable
leaves after argument drop, early completion, ERROR, cancellation, clone failure,
active-hook refusal and stale release. Matching scalar categories and opaque,
unknown and mismatched tags distinguish early lease release from retention. Two
new Environment/token allocation failures and identity-reference overflow are
checked before execution. The fixture wrapper follows root's reviewed two-argument
scope-retirement API; its existing bool parameter remains for old callers.

Strict C99 syntax with warnings as errors passes for scheduler and fixture TUs;
production env/eval syntax and Python syntax pass. No runtime gate has run. Full
source/control review precedes the separately planned union ownership activation.
