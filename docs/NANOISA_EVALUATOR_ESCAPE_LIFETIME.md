# My evaluator escape lifetime proposal

I record this proposal before implementing either escape correction. My owners
are `task_c3e8419dec9b8ccbcd6e1f99e5dbe5ec` (tuple results) and
`task_f8b5ecacb7e4bcc1542712b4652308ad` (deferred environments), both dependencies
of `task_7b805000dfda4da386b55d4691e8c647`. I have not executed a faulty case or
qualified my current list storage draft. I preserve full 5.1 acceptance.

## My observed tuple boundary

`create_tuple` copies strings and shallow-copies other Values. `free_tuple`
releases strings only. `eval_expression(AST_TUPLE_LITERAL)` publishes those
copies, and `call_function_at` can return a tuple. A tuple containing a list
result can therefore carry my Environment-owned record pointer beyond teardown.
Cloning only a top-level VAL_STRUCT at `call_function` is insufficient. A record
may itself contain a tuple, so treating nested tuples as borrowed also fails to
establish a complete record/tuple value graph.

I propose extending my checked clone/discard pair to the value graph consisting
of strings, records and tuples. Records and tuples recursively own copies of
those three kinds; scalar leaves copy exactly. Arrays, dynamic arrays, list
handles, callable signatures and foreign references retain their separate
existing ownership contracts. I do not claim those referenced resources become
independent of their provider by cloning a surrounding record. Resource-containing
records remain subject to the existing source ownership restrictions.

My API proposal is `bool env_clone_value_snapshot(Value source, Value *out)` and
`void env_discard_value_snapshot(Value owned)`. Failure preserves `*out` and
cleans all partial owned nodes. The discard API is for a value returned by the
matching clone, not an arbitrary borrowed Value. Every recursive node checks
count, pointer and allocation-size facts; the explicit depth bound is 128.
Strings copy their bytes, including the existing NUL-terminated convention.
I do not relabel a recursion or allocation failure as a successful VOID value.

`create_struct` and `create_tuple` will use this common copy policy, and their
matching destructors will follow it. I must audit all construction/free sites
before replacing either destructor: a tuple not created by this policy cannot
be silently treated as owning a formerly borrowed record. Binding installation,
reassignment, parameter staging and return copying must agree on whether each
tuple is owned. Copy-before-release is mandatory, including self-assignment and
nested field assignment. A public `call_function` result owns the complete
string/record/tuple value graph and can be destroyed by the paired public helper
after its Environment is gone. Internal calls may publish the same graph through
an Environment-owned snapshot entry. REPL results remain borrowed until printed
within the live Environment. I do not add an ownership bit to historical Values.

My tests must cover both direct and indirect calls, records inside nested tuples,
tuples inside records, shared input aliases, result use after Environment/AST
teardown, all measured allocation prefixes, and zero/double-free prevention.
Existing tuple, record, field, array-reference and callable controls stay intact.
This checkpoint must not change numeric enum conversion or list nominal identity.

## My observed deferred boundary

`CoroCallArgs` stores a borrowed Environment pointer. `spawn` queues that bundle
in the global scheduler; it does not immediately execute it. `nano_coro_release`
refuses READY/active tasks and releases only terminal scheduler metadata. There
is no current argument-drop callback or cancellation API. `free_environment`
does not check pending callbacks. The REPL and compiler teardown paths call it
without implicitly draining the scheduler. An Environment also borrows function
AST bodies: merely delaying its heap free while its owner frees the AST is not
a complete lease of executable input.

I propose an explicit checked evaluation lease rather than running user code
from teardown. `bool env_acquire_evaluation_lease(Environment *)` increments a
checked count before enqueue; `void env_release_evaluation_lease(Environment *)`
releases exactly that lease after completion/cancellation. An acquisition failure
occurs before enqueue and preserves the prepared bundle's caller ownership.
A preflight `bool env_can_destroy(Environment *)` reports no outstanding leases.
The existing void `free_environment` must fail with a defined nonzero evaluator
error before changing any storage when a lease exists. The exact proposed
message is: `I cannot destroy an Environment with pending evaluator tasks.`
The normal owner therefore cannot proceed past teardown to free its borrowed AST
while a queued task remains. The established API precondition still requires AST
providers to remain alive until successful Environment destruction; arbitrary
external early `free_ast` is not made safe by an Environment counter.

I propose `nano_coro_spawn_owned(CoroFn, void *, CoroDropFn)` for bundles whose
scheduler owns a drop action after successful enqueue. The existing
`nano_coro_spawn` retains its borrowed-argument behavior by passing no drop action.
The scheduler detaches the owned argument/drop pair before invoking its drop
callback, exactly once after a callback returns on DONE or ERROR, or after a new
checked cancellation of a non-active READY task. Cancellation must not invoke the
user task. Active/RUNNING/SUSPENDED cancellation refuses unchanged. Existing
release continues to refuse active/pending tasks; terminal release must never
repeat a detached drop. Failed enqueue never transfers ownership. Unsupported
nonterminal callback returns need an explicit terminal/error transition before
drop, rather than retain a READY task with a freed argument.

The evaluator's bundle drop will free its staged argument value graphs and name,
then release its exact Environment lease. Its trampoline will stop freeing that
bundle directly. Argument records/tuples/strings must be staged left-to-right
before later evaluation can invalidate an earlier argument, with rollback of
already staged copies. The result must pass through the owned public escape
boundary before the lease is released. A scheduler result remains governed by
its existing borrowed-result API until a separately explicit ownership transfer
or release contract is defined; I must not free a result still returned by
`coro_result` or `await`. This last result-drop policy is part of source review,
not an assumption of the lease counter.

Nested environments each have their own lease count. Cancelling a task for A
cannot release B's lease, nested await keeps the outer callback's lease alive,
and destroying B while only A has tasks remains allowed. Failed spawn, callback
error, cancellation, repeated cancellation/release, nested await, terminal result
use after Environment teardown, and pending destruction refusal are required
controls. No destructor drains tasks; no global scheduler reset silently loses
owned arguments. Until this contract and its source are reviewed, retirement
storage supports only in-Environment borrowing and is not complete async proof.

## My scheduler result ownership decision for review

I refine the proposed owned scheduler entry to carry two trusted C cleanup hooks
and one checked clone hook: argument drop `void (*)(void *)`, result drop
`void (*)(Value)`, and result clone `bool (*)(Value, Value *)`. The legacy
`nano_coro_spawn` installs none and keeps its borrowed argument/result contract.
The evaluator uses the owned entry with `env_clone_value_snapshot` and
`env_discard_value_snapshot`; these hooks are storage operations, not NanoLang
user functions. Successful enqueue transfers the prepared argument bundle.
The owned callback returns an owned result value graph. Normal completion moves
that result into the slot, whose sole owner lasts until successful terminal
release. Error completion discards the callback's owned return instead of
publishing it. Argument cleanup happens after result publication/discard, so its
Environment lease cannot end while the result still borrows that Environment.

`nano_coro_complete(value)` on an owned RUNNING task clones its borrowed input
through the slot's checked clone hook before publishing DONE. Clone failure
leaves the input untouched and makes the task ERROR with a defined allocation
failure; it does not publish VOID success. On return from that callback, its
separately owned return value is discarded because the early completion already
owns an independent copy. A second completion preserves the first terminal and
does not take ownership of the rejected input. The same terminal ordering applies
to `nano_coro_error`. Both execution sites—`nano_scheduler_step` and the direct
READY-target path in `nano_coro_await_id`—must call one shared completion/drop
helper. No path can bypass the owned bundle cleanup.

Existing `nano_coro_result` and `nano_coro_await_id` remain borrowed views of a
live terminal slot. They are valid until its successful release; they do not
permit a caller to keep owned pointers after release. I add checked owned-task
copy APIs `bool nano_coro_result_copy(int, Value *)` and
`bool nano_coro_await_copy(int, Value *)`. They require a DONE task with a clone
hook, preserve output on failure, and return a genuinely independent snapshot.
The evaluator's `coro_result`, AST_AWAIT and automatic async-call wait use those
copy APIs. Record/tuple copies then transfer into the live caller Environment's
snapshot registry; independently owned strings follow the existing string return
boundary. The automatic async path copies before releasing its completed slot.
Public `call_function` clones its own escape graph before returning to an embedder.
Repeated copies are independent; release frees only the slot-owned original.

Cancellation is a new checked action for non-active READY tasks. It detaches and
drops their owned arguments, records a terminal cancellation error, and never
executes the task or any NanoLang cleanup callback. Active/SUSPENDED tasks refuse
unchanged. Their terminal metadata must still be explicitly released, which
frees a slot-owned result at most once. A terminal result is never exposed while
its ownership is being dropped. Legacy borrowed-result tasks preserve their old
release behavior. Global initialization remains idempotent and cannot erase live
owned slots. Out-of-process termination cannot promise in-process cleanup; no
exception/longjmp cleanup contract is inferred from normal C returns.

## My actual teardown caller audit

I checked all `free_environment` callers under `src/`, not only the REPL. Several
already free AST providers first, so a check only inside `free_environment`
would be too late. Before lease implementation I require
`void env_require_destroyable(Environment *)`, which prints exactly
`I cannot destroy an Environment with pending evaluator tasks.\n` to stderr and
calls `exit(1)` when its checked lease count is nonzero. It performs no mutation,
callback, scheduler step, or AST cleanup. `free_environment` calls it first, and
AST/cache-owning callers must invoke it before their first provider destruction.
A separate `env_can_destroy` query permits embedding code to cancel/wait explicitly
before taking this fatal wrapper path; elapsed time never releases a lease.

| Current caller | Observed provider cleanup order and required placement |
| --- | --- |
| `src/main.c` | 21 Environment teardown sites; success and multiple failures free `program` first. Preflight precedes AST/token/cache cleanup whenever an Environment exists. |
| `src/nano_main.c` | Normal path explicitly drains tasks before cleanup, then frees AST first. Preserve that explicit run phase; preflight still precedes AST cleanup and covers early returns. |
| `src/dap_server.c` | `run_program`/`call_function` can enqueue; final AST/token cleanup precedes Environment cleanup. Preflight must come first and must not execute tasks. |
| `src/repl.c` | Persistent teardown frees Environment before tracked ASTs; Environment's own preflight is early enough. Temporary checker Environment also remains checked. |
| `src/lsp_server.c` | `doc_free_compiled` frees the document AST before its Environment. Preflight precedes AST release despite the ordinary checker-only route. |
| `src/main_stage1_5.c`, `src/nanovirt/main.c` | Final cleanup frees AST first; check before that provider release and audit every failure branch. No evaluator-task success is inferred for these compiler paths. |
| `src/wasm_interface.c` | Compile/typecheck paths free AST/tokens before Environment. Preflight must precede those operations on every live Environment path. |
| `src/module.c` | Private module compilation can clear its temporary module cache before freeing `module_env`; preflight must precede clearing that cache. Saved outer caches/environments retain their distinct ownership; no global lease release. |
| `src/transpiler.c` | The search hit is an ownership comment, not a teardown call. |

The unchanged AST-first metadata lifecycle tests have no pending evaluation lease
and remain valid. A caller that independently destroys borrowed ASTs while a
lease exists violates the documented provider lifetime precondition; the shipped
callers above will not do so after the source correction. This audit does not
claim a general external allocator or AST reference-counting retrofit.

## My tuple source checkpoint and consumer audit

I found one TupleValue allocator and one destructor under `src/`: `create_tuple`
and `free_tuple` in `env.c`; the evaluator's empty/nonempty tuple literal paths
are their only constructor callers. No other source or test C constructor
manually allocates TupleValue storage in this tree. I now route both constructors
and both destructors (record/tuple) through one checked value-graph pair. Every
partial record field or tuple element is counted only when cleanup can safely
visit it. Output remains unchanged until the entire root is complete. The depth
bound is 128 value nodes along a path, including record, tuple and scalar leaves.

I extend symbol installation/reassignment to copy tuples before publication or
old-value release. Environment destruction discards binding-owned tuples;
scope truncation retires them by the same single-owner rule as records. Both
function return paths snapshot record/tuple graphs, and public `call_function`
clones a borrowed arena root into independently owned storage. `free_tuple` now
owns its nested record/tuple/string graph because every constructor/binding uses
that exact copy policy. The snapshot destructor is also the public matching
release operation. No legacy Value ownership flag is introduced.

Tuple index reads copy string elements and snapshot record/tuple elements.
Existing explicit borrowing accepts an identifier/field place rooted at a record;
a tuple index is not an accepted borrow place, so this change does not silently
copy a legal mutable tuple-element borrow. Union construction already clones
record/string fields; it now also clones tuple fields so the union cannot retain
an arena tuple. I do not claim complete union reclamation from that copy boundary.
Static and dynamic array record paths use `create_struct`, inheriting the same
nested value graph policy while keeping their existing array identity contract.

The independent pending-argument finding is corrected at all three evaluator
call loops: each captures the formal kind before evaluation, stages a by-value
record/tuple immediately, and preserves explicit borrow formals. Builtins with
NULL parameter arrays retain their valid special handling. Ordinary record,
union, tuple and spread construction stage fields/elements before evaluating the
next value; spread bases are preserved before overrides. This snapshot storage
is cumulative until Environment teardown, not expression-bounded. No synchronous
callback, old binary, source program, compiler or fixture was executed here.

The deferred lease/result decision above remains proposed, not implemented in
this tuple checkpoint. Its accepted-flow lifetime remains a qualification
prerequisite, and its future source must include actual teardown caller ordering.

### My string staging correction before execution

Independent review of `b1118ad7c` found that my staging wrapper still skipped
standalone VAL_STRING even though the graph contract includes strings. I will
include strings in the snapshot registry and same call/aggregate staging sites.
A binding receiving a registry string must clone rather than consume it; public
results must clone borrowed arena strings. Reassignment must copy before release,
including self-aliasing. String snapshots have one registry owner and live until
Environment teardown. This does not claim cleanup of every older evaluator
expression temporary. Explicit borrow formals remain un-copied.

### My immediate async borrow argument boundary

My automatic async call waits on a newly enqueued READY task in this sequential
scheduler. Its first await iteration invokes that target through the common C
runner, which cannot return until the target callback and its argument drop have
unwound. Yield does not suspend; a nested cycle/error still returns through the
same runner and drops the bundle. Early terminal state does not clear `active`.
My automatic route copies the result and requires successful terminal release
before returning to its caller; release refuses an active task. Any await/copy/
release failure exits with an explicit nonzero evaluator error, so it cannot
return normally while a queued borrowed argument outlives the caller activation.
Failed preparation/enqueue transfers no bundle and preserves the old synchronous
fallback after cleanup. This is a normal-return proof, not longjmp/thread safety.

The bundle records ownership per argument. Explicit borrowed formals of this
immediate-await route retain identity and are not discarded with copied arguments.
Ordinary staged arguments own their copied graphs. Explicit deferred `spawn`
rejects a target with borrowed formals before enqueue with
`I cannot enqueue a deferred borrowed argument.` and exit(1); the source checker
already restricts explicit borrow expressions to declared direct-call arguments.
Top-level callable arguments copy their name and complete signature using my
checked snapshot copier with partial rollback. This does not change the legacy
compiler metadata allocation policy. Reference fields outside the record/tuple/string
owned graph retain their separate ownership contract.

### My shared-cache prerequisite

The actual startup cache clears and private saved-cache swaps require more than
local Environment preflights. I filed task_60bef9462e22d7eb1724212ad1811803 and
recorded exact generation ownership in NANOISA_EVALUATOR_CACHE_LEASES.md before
cache implementation. My current source does not install a global no-leases
restriction and does not yet claim complete cache-provider lifetime. This is a
required source/qualification dependency, not a postponed optional enhancement.

### My owned scheduler source checkpoint

I implement owned enqueue/cancel/release plus checked result-copy APIs alongside
the unchanged borrowed legacy spawn/result/await APIs. Both execution routes use
`coro_run`. Its active latch remains set through callback return, rejected-return
result discard and detached argument drop. Release and cancellation also protect
their cleanup with the active latch, and detach slot ownership before calling
trusted storage hooks. Early complete clones a borrowed input; normal return
moves its owned result; error/early completion discards the separate owned return.
The first terminal is retained. A returned nonterminal state becomes an error.
No cancellation or destructor executes the queued task.

Evaluator bundles own staged arguments and one checked Environment lease after
successful preparation. Failure rolls back copied arguments; successful enqueue
transfers the bundle to scheduler cleanup. Both explicit spawn and automatic
async calls use this route. Result copy precedes task release. Standalone callable
arguments/results copy and release their owned name/signature metadata with a
separate hook, because the public function-call boundary already owns those
fields; nested callable/reference fields keep their separate borrowed contract.
My subsequent checked-callable correction below replaces the initially retained
fatal signature allocator in this task snapshot hook.

`env_require_destroyable` now runs before all audited local AST/cache teardown
sites, including main's 21 cleanup branches, DAP/LSP, nano/nanovirt, browser,
Stage1.5 and private module_env cleanup. Environment teardown repeats the check
before mutation. REPL's existing Environment-first order remains, and its temporary
Environment is checked before unlinking its parent. Shared-cache generation
protection is still the separate, explicitly open design above.

My string staging correction uses the same cumulative registry. Bindings receiving
an arena string copy it before taking ownership. Reassignment copies string bytes
before releasing old storage, so later-argument rebinding cannot free an earlier
staged argument. Public arena strings copy at the escape boundary. I have not
built, executed, qualified, or reproduced a historical faulty case in this source
checkpoint. The cache prerequisite and independent review still block gates.

### My checked callable snapshot correction

The 6a118 source review identified a fatal legacy signature allocation inside
`eval_owned_task_clone`, whose contract requires checked failure. I replace only
that hook with `copy_function_signature_checked`. My new env-owned include copies
all signature parameter/return metadata and every TypeInfo name, parameter, array,
tuple, row, type variable and nested callable subtree. I preserve scalar flags,
counts, NULL optional arrays and duplicate annotations. I do not borrow a nested
metadata node or use a same-pointer shortcut.

I reject negative counts, unrepresentable allocation products and depth 128
before allocating the rejected node. Every allocated node starts with NULL
pointers; existing recursive destructors reclaim each partial graph. A false
return leaves the caller's output unchanged, including allocation failures after
a successful sibling copy. A NULL source is a successful NULL copy. The task
hook releases its staged name if signature copying fails and publishes its Value
only after both succeed. No fatal metadata allocator is called by this copier.

This is source-reviewed work pending review of the correction and fixtures.
I still require allocation-prefix recovery, nested annotation preservation,
output sentinels, depth/count refusal and scheduler result cleanup controls.
I do not claim recoverable allocation throughout the compiler's unrelated legacy
metadata consumers, and no test or build has run at this checkpoint.
