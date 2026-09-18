# My ordinary managed record foundation

I propose task_2dcceeb7ef38459093baaf52cf642239 under aggregate488 after
array graph admission732 on main `25a685ad`. This is a private descriptor/storage
prerequisite. I do not change opcode admission, source emission or the wire format.

## My audited next dependency

My managed profile now supports shape-checked array graphs, but still refuses
nominal tables/layouts and STRUCT/AGG operations. The VM represents records as
shared `VmStruct` heap objects with a definition index, fixed field count and
boxed values. AGG_PACK record and STRUCT_LITERAL allocate before popping their
counted fields; GET retains a child before releasing the receiver; SET publishes
the transferred value before releasing the previous child and returns the same
receiver. A stack-value copy cannot replace these aliases.

Retained LAYOUTS already survive canonical transport. Their global index is not
a VM record definition index: the latter is the ordinal among STRUCT layouts.
Same-shaped records remain different definitions. Nested metadata indices refer
to earlier global layouts. The decoder checks closure and tag range; it does not
by itself establish that each nested kind matches its field tag, nor that ordinary
bytecode field writes preserve a source annotation. Resource completeness flags
are a separate ownership contract and cannot be borrowed for ordinary records.

I also found a distinct static guard gap: OP_STRUCT_NEW publishes the result of
`vm_struct_new` without a NULL check, unlike STRUCT_LITERAL/AGG_PACK. I record
`task_66983a5c5f9a4c869fc726824cf6a773` before any repair or ordinary fault test.
No failed artifact is executed. Legacy STRUCT_NEW remains outside this proposed
foundation/admission; its checked guard is required before later matched support.

## My authoritative descriptor checkpoint

I add a host-side read-only ordinary-record plan from actual retained canonical
layout bytes and existing module counts. Count-only bridge placeholders do not
become complete descriptions. Explicit retained zero-field records remain valid.
I preserve a bidirectional global-layout/record-ordinal map, field order, declared
field tags/nested indices and distinct record identity. Optional names remain
advisory; identical names/shapes never merge two record definitions.

The initial complete descriptor kinds are ordinary records containing supported
scalar/string/array fields and earlier ordinary-record fields. I validate nested
record kind/identity explicitly. Unsupported union/tuple/map/callable field kinds
remain unresolved, with no inferred shape from their names or absent metadata.
Array child types are not invented from a bare ARRAY tag; later graph provenance
must establish possible children. I retain enum metadata without using it to
invent runtime enum identity that TAG_ENUM does not carry.

The plan distinguishes absence/unresolved, invalid data, finite limits and
allocation failure; failure leaves caller output and borrowed module unchanged.
I use the existing retained table/codec and do not allocate a new wire schema.
A compact immutable runtime descriptor view carries record ordinal, global-layout
identity and fixed field count. A target adapter must retain exact integer widths
and must not embed host pointers into Wasm data. Host plan memory has explicit
ownership/free; private runtime descriptor storage is borrowed for its entire
instance lifetime and cannot be rebound after record objects are published.

## My shared-table record checkpoint

I add a distinct record slot kind with explicit definition identity and fixed
boxed field storage. TAG_STRUCT remains the authoritative ISA tag8. Existing
strings, packed arrays and boxed arrays keep their kinds and behavior; record
metadata is not hidden in array capacity or an unrelated element-tag field.

Private construction borrows an ordered value vector, checks the descriptor
identity/count, prepares storage, retains each supported child and publishes one
stable handle only after success. Any failed allocation/retain releases only
prepared edges and leaves inputs/output/table state intact. Slot-table growth
continues to prepare collector workspace transactionally when enabled.

GET borrows the receiver and returns one retained child owner. SET borrows both
receiver and replacement, retains before publication, installs the new edge
before releasing the previous edge, and preserves receiver identity. Bounds/type
failures leave receiver, replacement and caller output unchanged. Runtime boxed
field values keep their actual tags/bits; the storage core must not pretend that
an annotation proves their dynamic tag or introduce ad hoc scalar coercion.
Future execution preflight separately checks the subset of VM-accepted writes.

I extend supported private child ownership to scalar/string/array/record values.
Arrays can retain records and records can retain arrays/records. Iterative
zero-ref release and both standalone/prepared collection traverse each actual
edge exactly once. The collector still validates/allocates before mutation;
prepared collection allocates nothing. Mixed cycles, repeated child aliases,
self replacement and dead-to-live edges preserve counts, and disposal frees each
object once. Invalid tags/handles remain checked API failures, not raw memory
interpretation. No union/tuple/map/callable slot kind is implied.

## My ordered acceptance and later obligations

1. Review and qualify the non-admitting descriptor plan: interleaved layout
   kinds, per-kind/global mapping, same-shaped distinct records, explicit empty
   records, nested records, unsupported fields, exact canonical bytes, bounded
   allocation failures and unchanged module/output. Audit every explicit build
   source list if a new host translation unit is introduced.
2. Independently review record storage and traversal before target execution.
   Qualify private native LLVM/sanitizer and import-free Wasm: exact scalar/float
   bits, strings, ordered fields, shared mutation, retained GET, SET aliasing,
   transactional descriptor/buffer/table/workspace failure, mixed record-array
   cycles, external roots and repeated bounded-live reclamation/disposal.
   Pin production/testing package ABI and clean regeneration hashes.
3. Preserve all existing array/string runtime and public profile/refusal gates.
   Use fresh ordinary VM record/array controls for identity and mutation after
   any required defensive prerequisite, never a pre-fix fault reproduction.
4. Before record opcode admission, record/review a separate complete field-origin
   analysis and matched lowering contract. It must connect authoritative producer
   layout identity, dynamic field effects through calls/globals/joins, counted
   literal roots, shared GET/SET semantics and generated prepared safe points.
   Ordinary source records, heap-bearing fields and arrays of records are required
   later acceptance, not closed by this private API.

Retained declaration layouts are acyclic; runtime alias cycles through boxed
containers are a different graph. Recursive nominal metadata needs its own
schema/producer decision if existing retained layouts cannot express it. Union,
tuple, map and callable transport, host capabilities, remaining platforms and
full aggregate/managed/release scope remain required and open.
