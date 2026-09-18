# My generated graph collection boundary contract

I record task4070da262 after graph provenance PR727 on main `3057bec8`.
The private collector and separate graph query are prerequisites, not executable
nested-array support. I implement and qualify the runtime preparation below
before changing admission or claiming emitted cyclic-program acceptance.

## My ownership audit

My emitted values carry one owner per dynamic stack/local/global value. LOAD
and DUP retain; stores transfer the popped owner before releasing the previous
slot. CALL transfers argument owners into callee parameter locals. Explicit
and implicit return move the result out before releasing remaining frame roots;
a successful heap result stays owned while held in the caller's SSA value.
Error cleanup releases the remaining operand stack and all locals. Completed
helper calls either return an owned result or release transferred operands.
Literal preparation keeps original counted inputs rooted until success.

The collector derives external roots from actual counts minus internal edges;
it does not need to read compiler stack memory. The ownership invariant must
hold for every live temporary as well as stack/local/global slots. Static graph
origins establish possible shapes, not these runtime counts. Collection must
never run inside retain/publication/release transactions or while a temporary
exists without an owner. Runtime helpers remain synchronous, with no callbacks
or concurrent collection.

## My allocation-pressure prerequisite

The current explicit collector allocates three scratch arrays on each call.
Calling it only after allocator exhaustion could fail to obtain the scratch
needed to reclaim garbage. I will not add collection to the low-level allocator
or retry it from a partially committed helper.

I add an opt-in private prepared-collection mode. Enabling it reserves aligned
trial-count/mark/queue workspace sufficient for the current descriptor capacity.
A checked contiguous allocation is acceptable: uint64 trial entries first,
byte marks next, uint32 queue at an explicitly aligned offset. I check every
capacity+1, multiplication, alignment and total-size operation against native64
and wasm32 bounds before allocation. Status values remain unchanged.

Once enabled, every descriptor-table growth prepares its replacement table
and sufficient replacement workspace before publishing either. Failure frees
only newly prepared allocations and leaves the old table, workspace, owners,
free list and caller output intact. Capacity accounting includes the temporary
old-plus-new peak; success is not promised when that peak cannot fit. Input
byte buffers prepared by the caller remain borrowed until slot publication.
Existing non-prepared runtimes keep their allocation behavior.

Prepared collection uses the same validated marking/commit algorithm with this
workspace and allocates nothing. I retain standalone `nms_collect` and its
failure-atomic scratch contract. Reserved workspace and descriptors are bounded
by table high-water capacity, reused across collections, and freed at terminal
disposal; I do not describe this bookkeeping as currently live object bytes.
This is reclaiming object storage, not a grow-only object arena.

## My module adapters and first error

I keep existing module begin/finish ABI and behavior for leaf profiles. A
separate private graph-begin adapter performs ordinary begin, enables prepared
collection only after successful entry, and publishes preparation failure
through the existing first-error accumulator. Refused nested entry must neither
reset a suspended caller's error nor alter its graph/workspace.

A graph collection adapter runs only on an initialized active prepared instance.
It preserves the first existing error. A graph-finish adapter runs after all
entry frames and transient return owners are released, collects with global
owners still held, then uses ordinary finish to return the original status or
the first collection failure. An error path may reclaim unreachable cycles
without replacing the error that caused unwinding. Terminal disposal remains
permitted only after active entry ends and invalidates all context-local roots.

## My future emitted safe points

After runtime qualification and independent review, a separately reviewed
admission/lowering step will select graph begin/finish and collection before
potentially allocating instructions, before any operands are popped. The initial
policy is deterministic pre-instruction collection, not an allocation-failure
retry: unreachable prior cycles are removed before the operation needs storage.
Each collection status branches through existing frame cleanup before executing
the instruction. Repeated loops cannot accumulate dead cycles across successive
allocation instructions.

The audited allocation candidates are ARR_NEW/PUSH/SET/LITERAL/SLICE,
STR_SPLIT/CONCAT/SUBSTR/REPLACE/TRIM/TO_LOWER/TO_UPPER/FROM_INT/FROM_FLOAT,
CAST_STRING and generic ADD. Conservative precollection is allowed when a
particular path does not allocate. Constants, borrowed reads/comparisons,
reference retains and numeric parsing do not allocate managed object storage.
Direct CALL needs no extra collector between argument transfer and callee entry;
callee allocation instructions provide their own safe points. Future admitting
new helpers must update this explicit audit, not assume the list is exhaustive
for an expanded ISA.

At module completion, graph finish collects cycles made unreachable by final
frame cleanup even if no later allocation occurs. A heap helper result remains
owned through caller transfer; persistent globals retain identity and preceding
writes across errors/reentry. Failure cannot leave partially owned graph roots
or trap before cleanup. The conservative initial schedule has no throughput or
VM collector-timing equivalence claim.

## My implementation checkpoints and acceptance

1. Implement transactional reusable workspace and allocation-free prepared
   collection, with private core/source review. Test initial preparation and
   every descriptor/workspace growth allocation failure, preserved outputs,
   accounting, old graph roots and later recovery. Preserve standalone collector
   and all leaf runtime behavior.
2. Qualify private module adapters with exact owned values in separately linked
   native LLVM and import-free Wasm harnesses: stack-like retained roots,
   transferred call/result owners, globals, self/mutual cycles, repeated bounded
   graph creation, allocator pressure, first-error preservation, refused nested
   entry, fresh versus repeated instances and terminal disposal. These harnesses
   qualify the runtime ABI/root discipline, not a bypassed executable profile.
3. Propose the final graph-profile conjunction and actual generated safe-point
   integration for independent review. Only after the preceding lifetime gates
   pass may that step add matched nested admission and ordinary VM/native/Wasm
   bytecode acceptance. No test-only verifier bypass or alternate unsafe emitter
   is part of this plan.

I require exact package regeneration/source hashes, production and testing
native/sanitizer and wasm32 links, finite-memory repeated reclamation, unchanged
leaf/profile/output-refusal gates, and existing graph provenance/collector tests.
Full aggregate488/managed51da, nominal/map/callable graphs, remaining platforms
and release criteria remain open.
