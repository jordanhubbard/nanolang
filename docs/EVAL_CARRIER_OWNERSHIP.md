# I establish roots before reclaiming carriers

I distinguish three facts in the current1d2e922c48 interpreter:

1. An owning record binding copies its input; replacing the binding releases
   that copy. A borrowed C/API binding keeps the caller's pointer and owns none
   of its storage. My #1207 guards retain this distinction.
2. Both generated typed-list push/get dispatch paths store and return the same
   StructValue pointer, encoded as int64 in an ELEM_INT DynArray. Neither a
   numeric Value tag nor the array's element tag identifies that edge. Generic
   DynArray integer traversal therefore cannot discover this record root.
3. Scope/call cleanup drops aggregate bindings; aggregate constructors and
   returned/manual clones have different copy rules. Missing cleanup leaks,
   while the already-active replacement cleanup invalidates typed-list roots.
   Adding recursive frees at scope exit would add this failure to more paths.

My native API replacement and checked source/shadow observations now establish
fact3 directly. The no-replacement API control passes ASan/UBSan+LSan and
asserts both the input-to-binding copy and binding-to-list pointer identity.
The source failure occurs while copying the value returned by list get into
another ordinary binding. Copying list get cannot repair a header already
freed at replacement; copying on push would change the retained API contract.

My next implementation must account for fresh source allocations independently
of the public Value/container/FFI layouts and legacy caller-created carriers.
It must record the typed-list edge at the actual producer, not guess that an
arbitrary integer is a pointer. The root set must include owned environment
bindings, active evaluation/argument temporaries, returned/host-owned values,
aggregate descendants, typed-list entries, and coroutine/handler lifetimes.
Borrowed roots prevent premature destruction but never grant ownership of
caller-provided storage. Duplicate roots must not imply duplicate destruction.
Popping a frame is not proof that its allocations are unreachable.

I must account for every existing early-discard operation before registering a
carrier: fresh literal retirement, partial-array rollback, array removal,
ordinary binding replacement and environment teardown. Existing raw legacy
cleanup must not leave a stale registry entry or reclaim a registered object
twice. The source-specific constructor/return clone sites must preserve actual
owned versus shared fields rather than pretending create_struct, create_union,
create_tuple and the two manual clones share one layout/lifetime policy.

I do not attach a record-release callback to generic GC shutdown without
checking its order. gc_shutdown currently walks allocation-list nodes while
saving next before a finalizer, and managed STRING child storage can be visited
before a containing malloc-backed record. A new ownership design must handle
both ordinary list release and shutdown, rather than assuming a finalizer has
safe access to every child or may recursively release the saved next object.
This is an implementation constraint from current source, not a claim that an
uninstalled callback currently fails.

I will first require direct ownership/provenance accounting and actual escaped
alias controls, then reclaim unreachable frame carriers, then rerun the full
original compiler/source/platform gates. I retain the1937/82 and3720/159 full
LSan failures. This boundary document and a red witness are not acceptance of
#1211/#1116/#1140 or the release.

## I account for allocations before adapting the evaluator

My private `src/eval/carrier_heap.c` engine owns only allocations explicitly
registered with a destructor and an optional edge tracer. It keeps a private
pointer index and an iterative mark queue; it changes no Value/container/FFI
layout. An unknown root or edge grants no ownership. Duplicate registration,
a missing destructor and mutation/reentrant collection during callbacks are
refused. Allocation failure leaves the payload with its current owner; failed
collection leaves every registered allocation intact. A tracer returning false
reports an incomplete walk and aborts before any destruction; a later complete
walk can retry against the unchanged graph.

My tracers enumerate valid registered edges. Shared nodes and reachable cycles
keep their exact addresses; unreachable cycles retire once without recursive
child destruction. Destructors release only their own carrier/scalar storage,
while registered children have independent entries. An adapter traversing an
unregistered legacy container must inspect its valid edges explicitly; this
engine cannot infer them from arbitrary integers or pointers.

`nl_carrier_heap_forget` transfers one allocation out without destroying it.
My adapter must explicitly transfer every owned descendant crossing a host or
interpreter boundary and remove entries before existing early manual discard.
A transferred graph can be readopted by the same or another heap without a
payload copy. The accounting invariant is registered = destroyed + transferred
+ live. Heap teardown destroys every remaining entry and releases its index.
The engine is per interpreter and does not provide concurrent mutation safety.

My focused normal and ASan/UBSan+LSan tests cover explicit replacement/list
roots, sharing/cycles, caller-owned unknown storage, transfer/early discard,
readoption and cross-heap transfer, reentrant mutation refusals, incomplete-trace rollback/recovery, a65536-node
chain, allocation/removal cohorts and six actual metadata allocation failures.
The graph fixture balances67667 payload objects and their owned text buffers;
the fault fixture separately verifies every registry metadata allocation is
released. These tests exercise the actual private engine, not the interpreter's
source producers. I add their target to ordinary units, but link the engine
only into this focused target until its adapter is ready.

The evaluator/environment/GC paths are still unchanged. Actual source allocation
registration, evaluation/argument roots, typed-list and async lifetime hooks,
host transfers and coordinated early-discard adapters remain required before
collection is enabled. The retained #1211 source UAF and both full-source LSan
failures remain open; an engine unit pass does not accept those source paths.
