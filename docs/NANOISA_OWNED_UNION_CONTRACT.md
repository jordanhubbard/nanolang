# My owned generic unions

I extend my canonical owned profile to preserve the existing accepted programs
in `tests/test_generic_selected_ownership.py` and
`tests/test_generic_selected_patterns.py`. MAC
`task_d44b2db373d38a942db3e8c4567b8044` owns this work. PR #522 and v5.1 remain
blocked until this feature and the other integrated acceptance gates pass.
This document specifies required behavior; it does not claim implementation.

## My current boundary

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

Selected extraction needs an explicit, statically checkable variant and field
count at each instruction. I must specify its encoding before implementation;
I must not overload the existing record unpack by treating the full union field
table as one runtime payload. The verifier, decoder, native stack analysis and
VM must agree on that encoding and stack effect. Invalid or stale selection,
wrong child layouts, mismatched counts and repeated extraction refuse. Existing
record instruction encodings retain their semantics.

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
