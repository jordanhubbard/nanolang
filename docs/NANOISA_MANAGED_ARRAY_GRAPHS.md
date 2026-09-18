# My non-admitting nested-array graph foundation

I record child `task_52d03b0ef98e452c9bc7b5d5591e843a` under aggregate488,
after main `03779c4a` (merged leaf literal/slice721). My full aggregate and
managed-runtime parents remain open. I change no opcode, wire metadata, shared
profile or source admission in this foundation.

## My audited gap

My roadmap explicitly requires heap-bearing collection identity and an explicit
cycle policy. `src/nanovm/heap.c` stores complete boxed values, retains children
on append and shallow slice, and transfers a popped owner. Its VM uses reference
counts plus cycle collection. A boxed declaration does not restrict every child
to that declaration. My managed `value_valid` currently accepts only scalar and
string tags; its final array release recursively visits string children only.
Merely accepting tag ARRAY would introduce unqualified recursion and leaked
cycles. My next prerequisite is owned graph storage and reclamation, before
shape analysis can admit general nested array programs.

## My exact representation and ownership

I retain `NmsValue`'s existing uint64 payload/uint32 tag ABI and context-local
stable handle table. ARRAY tag7 denotes a live array handle, never a numeric
payload or a string slot. I keep packed int/U8/float/bool storage unchanged and
refuse array inputs to packed writes under the existing private coercion matrix.
Boxed values can own scalar/string/array children, including duplicate edges,
self-edges, shared subgraphs and cycles. Record/map/callable/foreign children
remain required later work, outside this exact descriptor family.

I distinguish graph-capable boxed storage in naming and documentation, preserving
the existing numeric slot-kind ABI (an alias/rename of BOXED_LEAF_ARRAY3 is
sufficient). I retain declared element identity; declaration ARRAY uses boxed
storage. Existing declarations remain unchanged. Legacy STRING_ARRAY promotion
moves existing edges into boxed storage and preserves its handle and aliases.
Legacy string-only accessors keep their documented TYPE result after promotion.

Every borrowed append/set validates the incoming tag and live handle, prepares
storage, retains the new edge, publishes it, then releases the replaced edge.
Allocation/retain failure leaves contents, declaration, identity, refcounts and
outputs unchanged. Self-replacement retains before releasing. GET returns one
retained child owner; POP removes the edge and transfers its owner. Literal and
slice preparation retain array edges as well as strings and roll back only
prepared edges. A shallow slice has independent outer storage but shares nested
child identity, matching the VM. No slot pointer survives table growth.

## My iterative zero-reference release

I replace recursive child release with an allocation-free worklist of descriptors
whose reference count reaches zero. I may reuse their existing `next_free` field
while they are unavailable to live-handle lookup, but I do not publish them to the
allocator free list until their outgoing edges have been removed and storage
freed. Each edge decrements its child once; a child reaching zero joins the
worklist once. Scalars and literal strings have no dynamic edges. Duplicate
edges, long chains and shared descendants do not use the native/Wasm call stack
recursively. Pure cycles keep positive counts and wait for explicit collection.

This operation is synchronous, single-threaded and invokes no callbacks or
allocations. I preserve the existing status ABI and private valid-owner contract.
Terminal context disposal remains an unconditional one-buffer-per-slot teardown,
invalidating all context-local handles without traversing cycles recursively.

## My explicit failure-atomic collector

I add a private `nms_collect` operation, not a language instruction. I allocate
bounded-by-table-capacity scratch for trial counts, mark bits and a slot-index
work queue. Every size/alignment calculation is checked before allocation; a
scratch failure returns MEMORY with graph contents, owners and outputs unchanged.
I do not impose a new fixed object cap or add unmanaged growing storage.

I first validate live descriptor kinds, child handles and edge accounting without
mutating the live graph. I copy reference counts and subtract one trial count
per dynamic internal edge. Underflow or inconsistent private state returns STATE
before reclamation. Positive remaining counts identify external owners: frame,
global, retained API result and caller-held owners are all represented by the
existing reference counts. I mark those roots and their transitive outgoing
edges with a bounded iterative queue; each live slot is queued at most once.

After successful validation and marking, I commit without further allocation or
fallible helpers. Unmarked slots are unreachable from every external owner. I
remove their outgoing edges to surviving marked slots, then free every unmarked
buffer once and return its descriptor to the free list. Edges entirely inside
the dead set are not recursively released. Reachable cycles, duplicate edges,
shared dynamic strings and incoming edges from another surviving array remain
live. I update live byte/object accounting exactly and free all scratch.

I require no descriptor generation change: this remains the existing private
API where an owner keeps its handle alive and released handles cannot be reused
by the caller. Cross-context handles and concurrent mutation remain unsupported.
This collector is synchronous and must not run in the middle of a graph
mutation's commit. Stable external owners remain valid across collection.

## My ordered checkpoints and acceptance

1. Implement exact graph-tag validation, iterative zero-reference release and
   transactional nested edges/copies, with source review before widening any
   public profile. Retain the current scalar/string regression suite.
2. Implement the explicit collector and prove its operational invariants through
   ordinary graph controls: rooted/unrooted self and mutual cycles, duplicate
   edges, live shared children, cycles reached by globals/API owners, long chains,
   shallow copies and transferred POP owners. Deterministic scratch failures
   retain the graph and permit successful later collection.
3. Qualify production and test builds on native LLVM with sanitizers and
   import-free Wasm, packaged ABI/hash regeneration, allocator reuse/coalescing,
   descriptor relocation, repeated bounded-live graph creation/collection and
   terminal disposal. Compare ordinary graph mutation/identity with the VM;
   collector scheduling or exact allocation counts need not match.

My later admission work remains required and separate: origin graphs and child
provenance through GET/POP/joins/calls/global reentry, truthful fixed packed versus
boxed storage, matched root transfer and collection safe points under allocation
pressure and repeated module entry. Current leaf modules cannot construct nested
arrays and keep their admission. This private explicit collector alone does not
establish bounded storage for emitted cyclic programs or close full coverage.
