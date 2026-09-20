# I project complete mixed ownership declarations

I continue task_f36b179a0f2b4a1b99c29ccd2af66f99 and the full15f/488 scope.
My qualified private ordinary-array query validates both version3 extensions
but returns UNKNOWN when a union is present. My older public union accessor
refuses ARRAY_FIELDS before returning a variant. I must never remove either
refusal by skipping the other extension. This checkpoint supplies one owned
whole-envelope declaration plan before any new execution profile is enabled.

## My boundary

I retain the existing public validator, path/union accessors and executable
selectors unchanged during this checkpoint. A new internal header exposes an
opaque NvmOwnershipDeclarationPlan and copied numeric getters. Its preparation
returns PREPARED, UNKNOWN, INVALID, LIMIT or MEMORY and preserves the caller's
output for every non-PREPARED result. It owns no pointer into the source module.
Free(NULL) is harmless. Getter errors leave complete output values unchanged.
This is a shared core query, not a new wire format or proof of instruction flow.

I support COMPLETE ordinary record DAG declarations, flat scalar/string ARRAY
bindings and the already qualified exact scalar/string union variant slices in
one retained layout table. Global layout indices and per-kind ordinals stay
separate. A record ARRAY field still has TAG_ARRAY/NO_INDEX and exactly one
binding. A union does not acquire ARRAY payload support from an unrelated record
binding. Resource-bearing declarations, borrowed/reference descriptors, nested
ARRAY/STRUCT element types, incomplete facts, imports, module references, service
or passive contracts remain UNKNOWN after applicable structural validation.
Malformed framing, reserved bits, cross-table identities, missing/duplicate
bindings inside a present ARRAY_FIELDS body, truncated/trailing bytes and incompatible field contracts remain
INVALID. Limits remain explicit and do not become UNKNOWN authority. An absent array
extension with incomplete declaration facts retains UNKNOWN, as in the old query.

## My shared reader

I reuse the private checked retained-layout decoder, common descriptor/path/
extension reader and ARRAY_FIELDS validator from the ordinary-array query.
I factor construction of its owned declaration staging from the final ordinary-
only eligibility decision. The existing ordinary query keeps all its decisions,
including UNKNOWN for a union. I add no second envelope cursor or permissive
transport mode. Qualified union layout flags are exactly zero: COMPLETE is restricted to
STRUCT by the existing reader. A valid union sets the existing needs flag, so
I classify eligibility by exact layout/descriptor facts; I do not clear needs
or demand COMPLETE from a union. The old ordinary-only query keeps UNKNOWN.

Both extension bodies must validate before either projection
can be published. An ARRAY_FIELDS error cannot leave a usable union projection,
and a UNION_VARIANTS error cannot leave a usable array projection.

A descriptor whose legacy borrowed-root validation is ambiguous may record
UNKNOWN without aborting the new mixed staging pass after its full eight bytes
were consumed. I continue checking independently readable later descriptors
and extension bodies so a deterministic malformed suffix cannot hide behind
that subset decision. Cursor/framing failure or an undecodable allocation
failure still stops when further validation has no established input shape.
Old public and ordinary-query early-return/status decisions stay unchanged.

I reuse union_facts_read for copied variants. I extend that same checked reader
with optional bounded count/copy outputs. Its first whole-validation pass
counts rows; one later shared pass populates bounded staging; it does not call the older
public union accessor, which intentionally refuses the mixed envelope. All
allocations, fields and variant rows are charged before publication. The plan
retains the existing 256-layout, 65536-field/binding, 4096-type, 64-type-depth,
1048576-step and16MiB simultaneous-storage limits, with checked arithmetic.
Variant rows are additionally capped at65536 and charged to the same work/heap
budgets. I charge both validation/count and copy passes, including pairwise
name comparisons, before execution; I never parse the whole table once per
variant. I include transient staging in that budget. The common existing
256-variant limit per union remains. Allocation failure frees every prefix.
Ambiguous legacy decoder/reference allocation failure retains conservative
UNKNOWN classification unless an explicit hook establishes MEMORY.

My getters return counts; a copied layout kind, name index, flags and field
count; copied field and type/binding rows; and an exact variant fact by union
ordinal and variant. Numeric name indices are advisory identifiers, not copied
source text or nominal authority. Exact retained layout and ordinal mapping
supply identity. No getter publishes an executable-eligible bit.

## My qualification

I require real mixed envelopes with multiple concrete unions before and after
ordinary records, forward record DAGs and all five flat element tags. I compare
every copied layout/field/type/binding/variant fact, destroy all input storage,
and repeat every getter. I keep old ordinary/public union acceptance and refusal
controls beside the new positive query. I corrupt each extension independently,
its framing/revision/order/size/alignment, both cross-table mappings, function
signatures/descriptors and path rows. Every failed preparation/getter must leave
sentinels unchanged. Resource/nested/import/service/passive cases must retain
their stated UNKNOWN result, including malformed cases that must not be hidden
by early subset refusal. Boundary budgets, all allocation prefixes in persistent
and transient modes, zero-live cleanup and independent fresh recovery are required.

I review production and fixtures before running fresh Linux/Darwin ordinary and
scoped sanitizer checks, then retain unchanged ownership, affine-union and
ordinary-array neighbors with exact tool/source/product provenance.

## My required continuation

I separately audit every public retained-layout, verifier, VM, native, LLVM and
Wasm consumer before enabling the new grammar at an executable entry. Whole
metadata validation does not establish array element provenance, mutation
safety, alias lifetime, owner transfer or generated graph cleanup. Those origin
and runtime obligations precede paired source publication and all selected
shadows. Generic/imported/nested/recursive/tuple/map/callable graph scope remains
required by5.1; this intermediate projection closes none of those parents.

## My mixed forward-layout prerequisite

Static implementation review found that the existing private ARRAY decoder's
forward-edge path intentionally requires every layout to be STRUCT. A scalar
union beside a forward ordinary record would fail that earlier profile before
whole-envelope projection. I add a separate explicit mixed private decoder
entry using the same bounded copy/graph walker: STRUCT nodes may reference only
STRUCT nodes; UNION nodes carry only scalar/string NO_INDEX fields. The shared
iterative walk checks every record edge and rejects cycles. Existing public
and ordinary-array decoder profiles remain byte-for-byte decisions, including
their previous refusals. The new projection alone selects the mixed profile.
Resource ordering and union variant completeness remain the common ownership
reader's independent responsibility. This amendment precedes decoder changes;
its new profile must receive the same allocation, malformed-cross-kind and
forward-order controls before acceptance.
