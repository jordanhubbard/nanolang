# My borrowed helper local-owner reconstruction contract

I retain my design checkpoints from aggregate `6a58133c9`, rebased onto
`3446ec677` after FLOAT integration. My reviewed implementation is isolated on
`design/5.1-borrowed-helper-local-owners`. The design chronology below records
what required proof before admission; my implementation and measured limits
are recorded at the end. Full #999 remains open.

## My accepted source and retained evidence

I retain `tests/nanoisa/fixtures/source_borrow_helper_owners.nano` unchanged.
Its ordinary entry returns INT zero. `combine(target: &mut Leaf, source: &Leaf)`
has exact modes [2,1], scalar-leaf nominal parameters and an INT result.
Leaf has INT value and BOOL active; Pair owns two distinct Leaf children.
Helper-local nested field observations require v2 paths even though the caller
passes two distinct local roots. I do not infer path operands from module version.

`test_helper_owned_locals_preserve_formals_and_shadows` in
`tests/test_source_borrow_emission.py` retains the unchanged source, an alternate
with `if false` and rank bound zero, and the former-refusal variant of
`source_borrow_resource_paths.nano`: its single-exclusive bump helper constructs
and consumes a local Leaf before updating its borrowed view. That last variant
is an existing [2] profile composition neighbor, not evidence for a [2,1] signature.
The test compares NanoVirt/Stage1/Stage2 and raw producer canonical output,
checks parameter/name transport, executes VM/native, and independently produces
and executes selected shadows. My historical receipt is
`docs/evidence/checked-owner-selection/9b87b18c/source-borrows.log:47-48`:
these two methods pass within 46 methods in376.325s. This is retained evidence,
not a newly executed baseline or current output-local measurement.

## My proposed authority and lifetime boundary

I propose the exact two-function entry0/helper1 v2 [2,1] profile, retaining
existing [2] neighbors separately. I preserve exact nominal layouts, complete
original verifier admission and all current function/instruction/depth/local
limits. I admit no deeper calls, owned formals/results, local-owner writes,
partial moves, escaping references or arbitrary signature widening.

My helper inherits descriptor0 exclusive and descriptor1 shared. A local-owner
observation uses descriptor2 in one complete BEGIN/borrow/REF_GET/END region;
local/path operands come from their exact original opcode and validator.
`src/nanovirt/borrow_codegen.inc:808` establishes this source lowering.
The enclosing caller region remains live across the helper invocation. Local
observations cannot overwrite, end or redirect inherited descriptors. Both
caller roots are distinct; exclusive/shared overlap remains refused.

My owned-live state excludes borrowed formals. Helper-local origins are scoped
to that activation and never alias same-numbered caller slots. Packing first
and second into Pair transfers their exact children; destructive unpack returns
those identities, not replacement owners of the same layout. Branch-local
`source` declarations are distinct slots and origins from the shared formal.
Both arms consume them before joining; inherited source remains unchanged.
The rank loop constructs and consumes item each iteration, retains incoming
left/right origins unchanged, and exports no body-local owner. Every successful
return consumes all helper owners. Assertion failure is a trap with established
cleanup semantics, not successful affine consumption.

## My reviewed witness interface

My first actual call provides literal target(3,true), source(4,false). The
unchanged helper returns7 then11 on repeated calls and preserves source(4,false).
Its assertions depend on shared INT4, local fields100/true and8/false,
branch1/false or2/true, and an item field equal to the index snapshot taken
before increment. They do not constrain the changing exclusive target value.
A representative first-call witness is therefore plausible for both variants;
this source argument alone is not admission authority. My bounded static proof
implements the following rules before either output is published.

I require a reviewed, bounded static provenance interface before admission:
exact shared INT/BOOL literals from the actual first-call owner snapshots;
constructor/read/move/unpack scalar provenance; equality of the same immutable
index snapshot; Boolean constants/NOT and exact equality. It must reject an
unproved assertion rather than execute the helper to guess a witness. Branch
proofs merge only equal surviving provenance. Borrowed writes invalidate affected
facts; runtime assertions remain at every original position on every invocation.
A later call with source.value different from4 must still fail, even after a
valid selected witness. I do not invent zero owners: source.value==4 rejects them.

### My proposed finite proof representation

I propose a flat immutable token table, not recursively expanded expressions.
Each scalar fact is unknown, an exact typed INT/BOOL literal, or a typed token
identifier. Tokens identify a checked scalar definition/snapshot by function,
activation namespace, instruction position and lexical loop context. Copying a
snapshot copies its token; it does not re-read mutable storage. A scalar store
updates only that local's current fact. Source names never identify facts.
A recomputed arithmetic result gets a fresh opaque token, even when its source
spelling repeats. I do not equate arithmetic expressions or evaluate arbitrary
helper instructions; the existing exact rank proof remains separate authority.

Owner field facts are keyed by activation namespace, actual owner local slot,
immutable generation/origin token and exact nominal field position. Pack captures
operand facts in original evaluation order. Move transfers the same generation;
unpack transfers exact child generations and captured scalar facts. A new pack
in a consumed slot creates a new generation. A view resolves the current exact
owner generation; inherited borrowed roots use a separate caller namespace.
Shared literal witness facts attach only to the selected actual caller origins.
Exclusive writes invalidate affected field facts before any later observation.
I never infer equality from the same slot reused across generations.

I prove equality only between exact literals of the same type or identical
non-unknown typed tokens. A comparison result may become a BOOL literal from
those rules; NOT may invert a proved BOOL literal. Other Boolean outcomes stay
unknown and cannot justify a generated passing assertion. Snapshot creation,
owner capture and consumption retain the original instruction order. I reject
unproved assertions without dropping their runtime checks from emitted output.

At a branch join I intersect identical facts from continuing predecessors,
including exact owner origins; no union, guessed equality or name matching is
allowed. A loop entry carries only established invariant facts. The body proof
uses a fresh symbolic iteration namespace: an item field captured from index
and the unchanged current index share a token only within that iteration.
Increment changes the current index fact. Body snapshots and body-local owner
generations expire at the backedge; they cannot establish cross-iteration facts.
Only unchanged invariant facts survive exit, intersected with the zero-trip
entry state. The rank/work checker proves iteration bounds separately, without
unrolling the witness proof or treating termination as assertion validity.

I propose explicit reconstruction-proof caps: at most4096 allocated token
records,4096 simultaneously retained fact bindings across all saved branch and
loop states,65536 charged traversal/copy/merge operations, and nesting depth32.
These are new conservative witness-analysis limits, not VM limits. Every
instruction or IR statement visit, operand/field inspection, binding copied,
branch-merge comparison and token creation consumes one charged operation;
revisits and saved-state copies are charged again. I check before allocating or
inserting, and refuse atomically on exhaustion. Flat token comparison is constant
size; owner traversal follows the already validated depth/layout bounds and
charges each field. No recursive symbolic expression or implicit uncharged
state cloning is permitted. These caps remain subject to actual boundary controls before admission.
My implementation draft conservatively counts cumulative retained binding
allocations against4096, including saved-state copies, rather than reclaiming
that allowance when a state dies. This may refuse below the peak-live cap.
The retained-state cap is not a claim that every Python allocation is counted:
transient expression operand staging is independently bounded to two operands,
constructor field staging to sixteen fields and nesting to32, before those
lists are constructed. Retained insertions/copies reserve their allowance
before mutation; operand/field inspection also consumes traversal work.

Current `multi_witness` only proves direct shared-BOOL assertions and straight
line scalar operations. I do not broaden that rule by merely accepting [2,1].
The new generated shadow must own and consume its two actual snapshot fixtures;
its purpose is consistency, not branch coverage, universal correctness or
recovery of lost original shadows. Original selected-shadow modules remain
separately produced accepted inputs with separately measured cardinality.

## My structured proof and output obligations

I reuse immutable owner-origin and initialization proofs before rendering,
including exact branch survivors and loop backedge restoration. The helper's
single nonnested rank loop must retain exact dominating initialization,
nonoverflow bound and one increment on every continuing edge. I retain the
existing4096 logical-work policy and guarded-ADD work accounting, counting
owner construction/destruction, every local observation, assertions and both
condition and body costs. This is a conservative reconstruction proof policy,
not a new VM runtime limit. The zero-trip variant must not gain initialization.

I first measure both unchanged/alternate ordinary modules and independently
produced original shadows, then ordinary reconstructed entry/helper/shadow
local counts through actual regenerated products. At design time, feasibility under128 locals
and the existing two-function borrowed admission was unmeasured. I preserve
all assertions if a bound fails and stop for representation review. I do not
automatically extend finite rendering patterns or borrow old shadow cost formulas.
Original VM/native, reconstructed C/Nano and regenerated sanitized native must
agree on return values, repeated mutation, unchanged shared fields and cleanup.
Old profiles must retain exact output bytes.

## My required controls before any support claim

- I preserve wrong mode/nominal endpoint, wrong operand arity, interior targets,
  formal descriptor overwrite and attempts to move/store borrowed formals.
- I refuse leaked/moved-again helper owners, wrong child identities, changed
  incoming origin at a backedge, body-local escape and uninitialized zero-trip use.
- I preserve inherited reads/writes after descriptor2 observations, repeated
  invocation generations and same-numbered caller/helper local separation.
- I qualify unchanged and else/zero-loop source variants; actual assertions
  include100/8/1/2/index and source4, with target7/11 for combine. The historical
  former-refusal single-[2] variant remains a separate open composition; this
  exact [2,1] change does not claim to reconstruct it.
- I reject invalid or nonliteral witnesses and unsupported assertion provenance;
  a later false assertion must preserve prior effects and terminate correctly.
- I retain original-verifier refusals separately from defensive facts checks,
  complete literal/type/count bounds, atomic old-output preservation and old
  byte goldens. Full required rows9538/10409 and #999 remain open.

## My reviewed representation routing after the first refusal

I retain the actual ordinary Nano refusal at estimated49/179 locals. The
reviewed correction opts only the new-profile Nano helper index1 into existing
`plan_reassignment_nano`, after proving the original immutable origin/work/
witness IR. I leave entry index0, C and old profiles unchanged. I add no planner
patterns, scratch ADD bank or extra assertion folding. The diagnostic plan
uses5 constructor bundles,6 pack/store,10 move/store,10 read/store fusions,
10 unused scalar-unpack copies and3 sole-use snapshot/assert substitutions.
All151 statements are covered and all19 assertions remain. Helper120 is only
an estimator hypothesis until ordinary regenerated products measure it.

## My implemented representation and measured boundary

I prove original helper and caller owner origins, helper rank/work and the
first-call witness before rendering. Before a symbolic loop iteration I
invalidate every loop-written scalar and borrowed field recursively, retaining
only invariants and fresh same-iteration snapshots. The retained draft failures
show why the separate rank proof cannot establish those invariants.

My first ordinary Nano publication refused at the unchanged128-local bound:
entry49/helper179 were syntax estimates, not regenerated counts. After review
I reuse only the existing finite rendering planner on a cloned Nano helper1;
caller0, C output, guarded ADD implementation and old profiles remain unchanged.
I introduce no new planning patterns or scratch bank. All151 helper statements
are covered and all19 assertions remain. The measured regenerated ordinary
entry/helper counts are49/120. The independent consistency shadow measures34/120;
its existing38-local estimate is conservatively four slots higher for this
two-parameter profile. I retain that overestimate rather than weaken the cap.
Original ordinary and independently produced original-shadow inputs each have
exactly two functions with12/41 locals. Original shadows are separate executable
inputs; generated consistency shadows do not recover source shadows or claim
branch coverage. Exact current qualification and provider pins accompany my
final receipt.
