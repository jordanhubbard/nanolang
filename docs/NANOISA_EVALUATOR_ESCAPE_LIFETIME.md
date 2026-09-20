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
