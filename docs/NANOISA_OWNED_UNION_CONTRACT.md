# My owned generic unions

I extend my canonical owned profile to preserve the existing accepted programs
in `tests/test_generic_selected_ownership.py` and
`tests/test_generic_selected_patterns.py`. MAC
`task_d44b2db373d38a942db3e8c4567b8044` owns this work. PR #522 and v5.1 remain
blocked until this feature and the other integrated acceptance gates pass.
This document specifies required behavior. My format-4 declaration transport is
implemented. Selected VM/native transfers now execute for the bounded standalone
owned profile. My self-hosted and C-based `nano_virt` source producers now emit these transfers.
Scalar global effects and integrated release qualification remain incomplete.

## My historical starting boundary

At `7793ff5fe`, `nb_union_supported` accepts only scalar concrete arguments and
scalar substituted fields. `nb_union_type_identity` cannot identify a resource
record argument. `nb_union_expr` loads union aliases with `LOAD_LOCAL` and packs
scalar fields with `AGG_PACK`. Widening just that predicate would copy owners.

My retained ownership contract has the same limit. `check_layouts` in
`ownership_contracts.c` permits COMPLETE/RESOURCE flags only on records;
`descriptor` admits union descriptors only with zero layout flags;
`unions_read` requires scalar payload fields with no child layout. Private
mixed declaration decoding also specifies scalar unions. Each boundary needs
an explicit extension before resource union facts are executable.

My affine bytecode analysis rejects owned tokens in `AGG_PACK` and
`MATCH_TAG`. `OWN_PACK` and `OWN_UNPACK_LOCAL` consult whole-record fields.
The VM's owner move/unpack path requires `TAG_STRUCT`; native stack analysis
sizes unpack from the entire declared record. A union's flattened declaration
contains fields from every variant, while its runtime value contains only the
selected variant's fields. Those counts are not interchangeable.

## My concrete identity and layout

I resolve each concrete union and record argument to its bound declaration,
including module owner and concrete generic arguments. Equal spelling is not
identity. Unresolved arguments, ambiguous declarations, recursive payloads and
resource-bearing collections retain checked refusals until their own contracts
are implemented. Unused type arguments do not introduce stored ownership.

I classify the substituted stored payload graph. A union with an owner in any
variant has an affine whole-value contract, even when its current variant is
ordinary or empty. Selecting an ordinary variant discharges the outer move;
it creates no resource field obligation. Selecting an owned variant transfers
exactly that variant's obligations. Nested unions retain their own identity,
layout and variant; the outer selection cannot establish an inner variant.

I retain the declaration's flattened field table plus variant offset/count
facts. Each aggregate field names its exact child layout. Resource flags must
agree with transitive stored ownership. My validator checks child kind, bounds,
acyclic layout ordering, complete classification and variant coverage before
using a descriptor. I preserve the existing scalar wire grammar and introduce
an explicit versioned admission boundary for any expanded grammar. Unknown
versions and incomplete facts refuse before publication.

## My format-4 declaration transport

Ownership format 4 keeps the format-3 framing: version and layout count, padded
layout flags, function/local descriptors, bounded format-2 path suffix, and
ordered revision-1 extensions. `UNION_VARIANTS` keeps the same ordinal, layout,
variant-name and field-slice encoding. The new version changes the admissible
layout graph and descriptors; older versions retain their prior grammar.

Every layout is a COMPLETE record or union. Children precede their parents;
STRUCT and UNION field tags must name that exact child kind. Scalar and STRING
fields have no child index. Resource records may introduce an obligation;
unions have the RESOURCE flag exactly when a stored child has it. Every parent
of a resource child must propagate that flag. An unused type argument has no
edge in this table. Aggregate descriptors require an exact layout index.

My public declaration validator, authority queries and variant queries read
these facts. Binary serialization/deserialization and unverified textual
assembly retain them. Ordinary private declaration plans keep their old
profile. The owner-array router leaves format 4 to this validator. My shared
verifier checks selected affine transfers before admitting the bounded standalone
owned runtime. A declaration alone does not grant execution: the transport-only
fixture still refuses because it contains no owned transfer.

My VM and native backend transfer whole unions and detach exactly the selected
payload after checking its identity, variant, count and unique shell. VM union
ordinals are resolved to retained layout indices before checking locals,
arguments and results. Stack capacity is established before VM detachment.
Native emission uses the selected count rather than the flattened union layout.
The `test-owned-union-runtime` target checks resource, ordinary, empty, nested,
STRING and call/return cases, allocation failures, assertion cleanup and public
refusals. My self-hosted producer retains aggregate payload identities and ownership flags,
moves aliases and arguments, and emits selected extraction for complete patterns.
Fresh C-seed/Stage1/Stage2 generic acceptance and paired raw source execution pass.
My C-based `nano_virt` producer still has a scalar-only union boundary; owned
programs with mutable scalar globals also remain refused. Neither the raw runtime
checks nor the generic acceptance establish complete frontend or release parity.


## My transfer and selection

Construction evaluates each field once in source order, then packs declaration
order. Resource children are moved into temporary slots and then into the
selected payload; ordinary children retain ordinary value semantics. A failed
construction or trap releases each runtime root once. A successful move clears
its source slot. Aliases, parameters and returns carry the exact union layout
and whole-value ownership, including empty variants.

Matching evaluates its scrutinee once into an owned slot. Tag inspection is a
non-escaping observation; it cannot copy or consume the payload. On the matching
edge I retain the selected variant for that exact live slot generation.
Reassignment, consumption or an incompatible join invalidates selection.
An observation cannot outlive a move. A branch fact about an unrelated stack
copy must not authorize unpacking a local.

Complete destructuring consumes the selected outer slot and transfers each
selected field once. I require the correct union declaration, concrete instance,
variant, and every declared field exactly once. Partial projections cannot
move a resource field. Empty and ignored ordinary arms still consume the outer
union shell; they need no fabricated payload owners. Continuing arms agree on
outer owner state. Terminal arms satisfy their exit obligations independently.
Guarded resource matches retain their existing refusal boundary.

`OWN_UNPACK_VARIANT` has primary byte `0x97`, followed by three little-endian
`u16` operands: source local, selected variant, and payload field count. It pops
no stack values and pushes exactly `count` fields in declaration order. The
count must equal the retained selected variant slice; it is not the flattened
union declaration's field count. Existing record unpack encodings keep their
meaning. Schema generation and codec tests retain the seven exact bytes.

My affine analysis treats `LOAD_LOCAL` of a resource union as a rooted
observation. `MATCH_TAG` refines that exact local on the successful edge.
Moving, replacing or unpacking an observed local refuses; `POP` can discard
the union observation without consuming its owner. Extraction requires all
observations to have ended and the local still to carry the selected variant.
It consumes the local and creates exact typed payload tokens. Nested union
payloads start with an unknown inner variant. Whole stack moves clear local
selection, so reuse cannot inherit a stale arm. Live owner slots cannot be
overwritten; joins must agree on owner liveness and can only weaken variant
knowledge. The old scalar union rules retain their existing semantics.

My local-normalized transfer APIs reject wrong counts, wrong nominal layouts,
duplicate resource inputs, live owner destinations and unselected extraction
without changing their input state. The stack caller retains each resulting
resource obligation. Bytecode worklist and allocation refusals prevent a failed
analysis from supplying execution authority. VM/native extraction uses the same verified selected field count. Public execution
requires the complete standalone owned-module verifier conjunction.

## My execution and cleanup

My VM and native translator implement the same root transfers. Ownership is
not inferred from a non-null record pointer or reference count. Runtime tag,
layout, variant and field-count checks protect the boundary before mutation.
Allocation failure before transfer leaves sources intact; after transfer,
cleanup visits only the destination roots actually established. Unpacked
children are detached before destroying the union shell. Traps and early
returns unwind current roots without double release or retained stale aliases.
Ordinary GC emission is not a fallback for an unsupported owned program.

## My acceptance order

1. I qualify transport round trips and malformed flags, child identities,
   variant slices, cycles, truncation and unknown versions. Failed queries
   preserve their outputs. Declaration acceptance alone grants no execution.
2. I qualify raw affine admission and refusal for moves, selected extraction,
   stale observations, duplicate consumption, joins and nested unknown variants.
3. I execute the admitted raw modules in both VM dispatch modes and native C.
   Allocation/trap/early-return controls run with ASan, UBSan and leak checks.
   Native publication refusals preserve an existing output artifact.
4. I run all ten accepted selected-ownership methods: resource/ordinary arms,
   ignored ordinary arms, empty arms, alias/return, constructor return and
   argument, conditional return, block match return, nested generic transfer,
   and an outer owner consumed in every arm. I retain every existing negative
   method, including dropped owners, duplicate use, partial moves, incomplete
   matches, disagreeing joins, guards, collections and unresolved tuples.
5. I rebuild both canonical compiler stages, repeat the complete selected
   pattern/ownership suites, and qualify integrated bootstrap, fixed points
   and platform/sanitizer gates. Focused passes do not close PR #522.
