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
