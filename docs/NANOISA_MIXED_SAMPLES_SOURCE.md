# My paired Samples source prerequisite

Task `task_e64a2673b2b344b487746458d4d1e6ac` belongs to mixed parent4be.
My runtime dependency is actual PR819 merge
`d0de3d23a730c66531f2ed2c5d972215302ebe19`. My initial checkpoint was a contract and static audit. I record subsequent
production, successful bootstrap and the first checked source refusal below.

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

At that checkpoint I had inspected the source and checked whitespace only. I had not built
these producers or executed their new source/shadows. Their fresh bootstrap,
paired equality, false-shadow/output guards and runtime qualification remain
pending independent source review. My runtime and service guard files are
unchanged; I will integrate canonical service retention before qualification.

My first integrated bootstrap at `5ccfe04b6c0af9145a78523742e03499feb99a47`
includes canonical required-service guards from PR821 without changing either
reviewed producer. Fresh bootstrap passed in 268.797 seconds and tool/probe
setup passed in 25.821 seconds. Source maps, head and tracked tree remained
unchanged across both phases. I retain those reports separately from subsequent
source acceptance. My focused source fixtures now include the unchanged original
four-driver method, all selected shadows, paired dumps/names/stripped execution,
constructor order, aliases, contextual empty fields and retained refusals.

## My complete scalar-leaf pattern correction contract

Child `task_c935734a6a0e44dbbdbce12fcea31a9e` records the first source gate's
checked refusal at frozen `b92753166931a5ba75030d629b9abc762fb9c94f`.
The fresh five emitter/shadow drivers were built in 176.131 seconds. The first
unchanged original Samples/PREFIX publication then refused in 0.005 seconds:
function 1, offset 6 has no closed mixed transfer. No module was published or
executed. I retain the terminal, logs and maps in
[evidence/mixed-samples-source-first](evidence/mixed-samples-source-first/manifest.json).

My existing scalar-leaf destructive pattern lowering validates the entire
pattern, then marks a live leaf holder pending disposal. Its subsequent synthetic
field bindings use REGION_BEGIN / REF_GET before final owner cleanup. Those
reference opcodes remain outside the qualified mixed value graph. My nested
owner pattern already uses exact OWN_UNPACK_LOCAL with hidden field slots.

For mixed value graphs only, after successful full, distinct, exact pattern
validation, I will use that same explicit OWN_UNPACK_LOCAL path for scalar leaves.
It consumes the named/destructured shell once, clears its live/pending state,
and stores every declared field in typed hidden locals in the established stack
order. Subsequent synthetic pattern projections bind those exact parent/field
slots; scalar values may be loaded, but the original owner remains dead. The
source pattern's field ordering does not reorder its initializer or evaluation.
The 256-local bound and all nominal, field-count, duplicate/missing-field guards
remain. Failure before complete validation never authorizes a partial unpack.

I change neither general owner observations nor reference admission, implicit
drops, owner ARRAY fields or runtime authority. Pure-owned and CALL_REF source
paths retain their previous pending-disposal lowering. Existing nested paths
continue to use their established unpack behavior. I require paired bytecode and
local-name parity for the full original PREFIX, reordered leaf patterns, owner
use after consumption refusals, invalid pattern/output guards and old profiles.
A fresh bootstrap must include both corrected producers before acceptance.

My first source-run maps distinguish compiler inputs from setup outputs:
versioned sources/head stayed identical, the five generated producer binaries
stayed identical after setup, but obj/nano_modules cache objects were rebuilt and
new cache objects appeared while the emitters were compiled. I retain their
before/after hashes; I do not claim that the complete object maps are unchanged.
Subsequent qualification will capture a separate post-setup object/tool boundary.


## My corrected Linux qualification

At frozen `70c511dad439f77eba15fe0bdc189e90ff9c9486`, I qualify the reviewed
paired correction on canonical PR821 service guards plus PR820/822 integration.
My [Linux seal](evidence/mixed-samples-source-linux/manifest.json) retains
53 reports, exact source/tool inventories and retained artifact hashes.
Fresh bootstrap passes in 270.113 seconds; tool setup passes in 26.522 seconds.
Fresh emitter/shadow setup takes 177.373 seconds. Four focused methods pass with
GCC in 129.169 seconds and again with Clang in 84.309 seconds; 58 existing
borrow/STRING source methods pass in 265.019 seconds. These are 66 method
executions, including the unchanged original Samples/PREFIX method and every
selected shadow. I retain dump/name/stripped parity, constructor evaluation
order, aliases, contextual empty fields and output-preserving typed refusals.

Versioned sources and generated producer binaries stay unchanged. Emitter setup
rebuilds compiler-cache objects, so I do not claim equality across that setup.
All three subsequent source phases preserve every post-setup object/tool input.
My ordinary record fields are exact FLOAT arrays and existing INT/BOOL scalars;
this does not admit managed signatures, owner ARRAY fields or borrowed mixtures.

My mixed runtime/lifecycle/allocation gate passes in 34.301 seconds: 2,006
lifecycle checks, 13,358 heap checks across 488 budgets with 440 injected faults,
and 127 admission checks. Existing owned graph/result gates pass in 17.810
seconds. My first service preparation stops before assertions because Clang
reports ambiguous GCC installation selection under strict warnings. I preserve
that 0.215-second status2 separately. With the existing native-only
`NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13`,
the unchanged service gate passes in 3.673 seconds: three methods containing
725, 655 and 212 checks. Wasm flags and product sources are unchanged.

This seal is Linux evidence only. I retain the original checked source refusal
and setup-cache limits. Darwin same-pin qualification, canonical source merge,
parent4be, owner ARRAY430220 and full product/release acceptance remain open.

My bounded Linux integration at `6a492a7aab1184e22a0fd15935293738ec7a58dd`
adds canonical private owner-array descriptors and private File values, preserving
both producer files and all `.nano` sources. I rebuild affected C tools,
including the C producer, in 19.716 seconds. Mixed descriptor/proof/composition,
admission and VM/native runtime gates pass in 35.005 seconds; service gates pass
in 3.672 seconds; the unchanged original Samples/PREFIX shadow/parity method
passes in 28.186 seconds. Source and retained Stage1/Stage2/raw emitter hashes
remain identical. My [integration seal](evidence/mixed-samples-source-integration/manifest.json)
keeps this bounded check distinct from the full frozen70c qualification.
I do not repeat bootstrap for these private C-only additions.

After private origin-query PR829, my final provider integration at
`fbd3b5a9a494e8529d4c301098f23df670555735` adds only its retained-layout include,
private query and target. Existing `isa.o` supplies its decoder/info references.
Affected C producer, VM/native/LLVM/HL tools and metadata probe link in 20.567
seconds. The original Samples/PREFIX/all-shadow/parity method passes in 28.285
seconds. Sources and retained stages remain identical across this gate; source
producers, selectors and mixed runtime are unchanged from the preceding seal.
My [provider seal](evidence/mixed-samples-source-provider/manifest.json) preserves
this narrow acceptance without relabeling it as a fresh bootstrap or full suite.

## My same-source Darwin qualification

I qualify frozen `70c511dad439f77eba15fe0bdc189e90ff9c9486` in a separate
Darwin tree. Explicit Apple Clang/SDK bootstrap passes in 456.417 seconds;
tool/probe setup passes in 121.878 seconds. Fresh emitter/shadow setup takes
319.195 seconds. Homebrew LLVM23 passes the four focused methods in102.960
seconds and all58 existing borrow/STRING methods in660.013 seconds:62 method
executions,1083.972 seconds total. Original Samples/PREFIX and every selected
shadow remain unchanged. Both assertion phases preserve every post-setup input,
with no added inputs; producers and versioned sources remain identical.

I preserve distinct runtime/compiler-selection outcomes under child68db:

- My first mixed runtime/heap/admission gate passes in25.782 seconds with Apple
  ASan/UBSan and explicit root/allocation assertions. Its existing Apple policy
  disables leak detection; I do not call this a Homebrew LSan pass.
- The following owned gate stops in14.020 seconds because make's `CC = cc`
  overrides my environment selection and Apple rejects graph leak detection.
  The original result fixture reports `compiler=cc detect_leaks=0`.
- In another same-source tree, explicit command-line Homebrew CC makes owned
  graph/results pass in29.948 seconds; the result log confirms `detect_leaks=1`.
  The subsequent service gate stops in7.569 seconds because its independent
  compiler selector still defaults to `cc`. I preserve that second terminal.
- With every relevant selector explicit, fresh service fixtures pass in5.979
  seconds: three methods with725/655/212 checks. A separate previously unmeasured
  Homebrew mixed-native configuration passes in23.887 seconds: all twelve cases,
  O0/O2, ASan/UBSan/LSan, allocation faults and root assertions. Prepared source,
  tool and object inputs stay identical across both corrected direct-Python gates.

I change no product source, fixture assertion, sanitizer requirement or deadline
for these corrections. I retain new artifact directories and never rerun the
preserved failing binaries. Bootstrap and passed source/heap/admission gates are
not repeated. The [Darwin seal](evidence/mixed-samples-source-darwin/manifest.json)
contains76 reports and separate successful/failed artifact inventories; its
archive transfer hash is independently checked locally.

My source runner hashes the actual Homebrew compiler before emitter setup.
Bootstrap records its explicit `/usr/bin` invocation tools and SDK selection.
Supplemental SDK/config/tool hashes are captured during emitter setup, and
xcrun-resolved Xcode backend hashes during later source regression; I do not
relabel those supplemental snapshots as before-bootstrap evidence.

Linux final private-provider integration remains separately qualified. These
bounded source and runtime checks do not close parent4be, owner ARRAY430220,
the separate embedded-runtime Darwin fixture taska522 or full product/release
acceptance. PR828 merged at `6a993be5511297cb2cf6793d3a8aae0b6a75f8e2`; e64/c935/68db are reconciled complete from canonical ancestry.

My final canonical authority-provider integration at
`29ab099b5bec7eb3c2082fc4c9c504e2226d1da4` includes private query PR830 without
changing either source producer or public routing. Its private includes add no
conflicting preprocessor names. Affected C tools link in20.818seconds, existing
mixed composition/admission controls pass in1.468seconds, and the original
Samples/PREFIX/all-shadow/parity method passes in28.886seconds. Versioned source,
retained stages and host tools remain unchanged across this bounded gate. My
[authority integration seal](evidence/mixed-samples-source-authority/manifest.json)
keeps it distinct from frozen70c platform qualification.
