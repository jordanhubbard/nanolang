# My paired Samples source prerequisite

Task `task_e64a2673b2b344b487746458d4d1e6ac` belongs to mixed parent4be.
My runtime dependency is actual PR819 merge
`d0de3d23a730c66531f2ed2c5d972215302ebe19`. This checkpoint is a contract and
static audit only. I have not changed a producer, built a new compiler, or
executed a newly admitted source program.

## My unchanged acceptance

I retain `tests/test_owned_record_patterns.py::test_ordinary_inferred_field_and_alias`
exactly, including its full PREFIX:

```nano
resource struct Handle { fd: int }
fn close(owned: Handle) -> int { let Handle { fd } = owned return fd }
shadow close { assert (== (close Handle { fd: 7 }) 7) }
struct Samples { values: array<float> }
fn main() -> int {
    let record: Samples = Samples { values: [1.5, 2.5] }
    let values = record.values
    let alias = values
    assert (== (at alias 1) 2.5)
    return 0
}
shadow main { assert (== (main) 0) }
```

I qualify C-seed, Stage1, Stage2 and NanoVirt. Every selected close/main shadow
remains in the same checked selection and synthetic entry graph. I do not strip
PREFIX, split owner and ordinary execution, change the example, or treat raw
emitter output as verified publication.

## My static boundaries

`src/nanovirt/borrow_codegen.inc` currently requires explicit/transitive resource
records, uses OWN_* for STRUCT lets, admits only scalar local expressions, and
runs nvm_verify_owned_module after general verification. The last check deliberately
refuses mixed modules despite PR819's separate complete conjunction.

`src_nano/compiler/nanoisa_borrows.nano` likewise emits ownership flag03 for every
record, rejects array<float> in nb_local_tag, and treats STRUCT identifier values
as owners/references. Its field expression path uses resource-place/reference
operations. Merely adding an ARRAY tag to nb_local_tag would leave these other
boundaries wrong and could widen unrelated field/signature authority.

I change only the closed consuming value-graph source path. Existing borrowed
CALL_REF graphs and old pure-owned/STRING paths keep their own checks. Whole-source
checking, reachable selection policy and mandatory shadows remain prerequisites.

## My first source shape

I admit finite nongeneric ordinary records whose fields are exact flat
array<float> values, plus their constructor, direct field projection, local
bindings/aliases and existing exact integer `at` reads. Explicitly typed empty
arrays may use the constructor's exact expected array<float> field type. I do not
infer FLOAT from an ARRAY wire tag or choose an element type for an untyped empty
array. Every literal element must have checked FLOAT type; unsupported coercions
remain refusals.

Existing supported scalar operations, lexical scopes and consuming owner calls
remain available under their prior contracts. I preserve the eight-function
complete acyclic graph bound including synthetic shadow entry, 256 local slots,
existing field/layout/depth limits and exact scalar entry result. I do not add
ordinary managed parameters/results, array mutation builtins, nested ordinary
record source fields, general collections, generic records, imports, externs,
globals, callbacks or recursion in this first slice. Unsupported selected code
refuses the full publication rather than taking a fallback.

Owner declarations keep their established classification and field policy. A
record containing an owner is an owner, never an ordinary managed row. ARRAY
inside an explicit/transitive owner (including Bundle) remains child430220;
FLOAT owner fields remain refused. Existing STRING-owner acceptance stays on its
qualified path; combining it with ordinary managed arrays is not implied here.

## My exact source facts and wire identity

I add explicit per-layout/per-local category facts: scalar, owner, ordinary
record, or exact flat FLOAT array. Original declaration/global layout indices
remain stable in both producers. Source struct ordinals are distinct from any
compact runtime managed mapping. Runtime mappings come only from fresh proof.

Owner rows retain COMPLETE|RESOURCE (03); proved source ordinary candidates
retain COMPLETE (01). An ordinary ARRAY field uses the existing ARRAY/NO_INDEX
wire descriptor. The source compiler separately carries exact FLOAT element
facts through constructors, field results, aliases and lexical bindings. Wire
ARRAY alone grants no element authority. Local ordinary STRUCT descriptors retain
the exact original nominal layout; ARRAY locals retain ARRAY/NO_INDEX plus their
compile-time element fact. No new schema, compact nominal replacement or fake
VOID local is introduced.

Checked binding identity and initializer-before-binding order win over names.
A same-shaped distinct nominal remains distinct. Explicit annotations must agree
with inferred checked facts; popped lexical bindings restore the outer fact.
Builtins such as `at` resolve only after existing lexical/declaration shadowing
rules. I retain advisory local-name intervals and deterministic executable-literal
versus name ordering, including selected shadows and unnamed compiler temporaries.

## My instructions and lifetime

Ordinary record construction evaluates each supplied field once in source order,
validates exact field names/count/types, then uses existing temporary slots to
pack declaration order with AGG_PACK and original nominal identity. I retain
unknown/missing/duplicate field refusals. Owner construction stays OWN_PACK and
named owner moves stay OWN_MOVE_LOCAL/OWN_STORE_LOCAL with all existing live,
mode, nominal, disposal and observation checks.

Ordinary values use LOAD_LOCAL/STORE_LOCAL, AGG_GET and ARR_LITERAL/ARR_NEW/ARR_GET.
Aliases retain the same ordinary value; they never enter owner transfer, borrow,
explicit owner disposal or disposal_pending logic. Scope/terminal cleanup follows
the qualified ordinary runtime roots, while source owner consumption stays exact.
No implicit owner drop or relaxed branch/backedge obligation follows.

An ARR_GET can dynamically produce FLOAT or VOID. I preserve that runtime fact;
I do not synthesize a default value or declare the runtime operand unconditionally
FLOAT. Typed consumers use the qualified tag checks and failure cleanup. Generic
comparison behavior must match the selected existing instruction semantics.
Tests distinguish missing-read failure/false assertion from an invented numeric
result, and check owner/ordinary cleanup on the accepted error path.

## My final publication conjunction

For a positively classified mixed value graph, I require the explicit mixed
candidate and fresh successful public mixed verification after exact emission.
Candidate detection alone is not authority. I do not retain the incompatible
owned-only verifier as the acceptance gate for this new route, nor remove it from
old routes. Metadata-only success and failed mixed validation cannot fall back
to ordinary lowering. Selfhost checked assembly/native publication must run the
same public conjunction; raw text production remains a separate capability.

Module bytes, layout flags, function/local descriptors, selected-shadow graph
and canonical disassembly must agree across C and both selfhost producers.
The independent shape and affine proof still checks all emitted functions and
scalar obligations. Source guesses never replace it. Later required-service
metadata rejection must precede this admission route and survive transport.

## My review and qualification order

1. I obtain contract review before producer edits, then send paired production
   checkpoints before executing newly admitted source.
2. I build fresh canonical C-seed/Stage1/Stage2 and emitter/shadow tools after the
   source changes; no old compiler/cache is relabeled. I keep qualified runtime
   trees immutable and preserve the first terminal of each gate.
3. I run the unchanged full PREFIX case through all four drivers, VM/native
   execution, canonical module equality and complete shadow selection. A false
   close shadow and false main shadow must stop publication and preserve an
   existing output artifact; an owner-free selected subset does not replace the
   full required acceptance.
4. I cover constructor field order/once-only evaluation, empty contextual arrays,
   inferred field/alias facts, lexical shadow restoration, distinct nominals,
   checked missing-read consumers and correct runtime cleanup. Negative controls
   cover mixed element types, unknown/duplicate/missing fields, untyped empty
   arrays, forbidden owner ARRAY fields, ordinary managed signatures, wrong
   owner transfers and unsupported selected shadows, with exact diagnostic phase
   and prior-output preservation.
5. I retain runtime819, private non-admission reports, owned/STRING/reference,
   lexical-name and source-borrow adjacency. I seal source/tool/object identities
   and evidence before canonical review. Source acceptance closes only this child;
   owner Bundle, broader source/managed/ownership parents, Darwin/full product
   qualification and release publication remain separately open.

## My paired production checkpoint

I now distinguish ordinary record rows from owners in both source producers.
My ordinary constructors evaluate fields once into typed hidden locals, then
pack declaration order with AGG_PACK. My exact FLOAT arrays use ARR_LITERAL 3;
field projection and aliases retain ARRAY locals, and unshadowed `at` uses raw
ARR_GET followed by the existing checked scalar consumers. I do not replace a
missing element with a FLOAT default.

I exclude ordinary rows from owner cleanup, resource places, ownership joins,
owner moves and record signatures. I retain COMPLETE 01 versus owner 03 and the
original layout indices. I refuse managed binding assignments, owner ARRAY
fields and mixed reference signatures. My C publication path uses the fresh
public mixed conjunction for a positive mixed candidate, retaining the old
owned-only verifier on other routes. My selfhost output remains subject to the
same assembler/public verification boundary.

I have inspected this checkpoint and checked whitespace only. I have not built
these producers or executed their new source/shadows. Their fresh bootstrap,
paired equality, false-shadow/output guards and runtime qualification remain
pending independent source review. My runtime and service guard files are
unchanged; I will integrate canonical service retention before qualification.
