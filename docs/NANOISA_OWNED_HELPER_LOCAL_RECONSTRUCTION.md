# My borrowed helper local-owner reconstruction design

I own this design-only #999 checkpoint on
`design/5.1-borrowed-helper-local-owners`, based on aggregate `6a58133c9`.
I change no admission, schema, producer, verifier or renderer here. The FLOAT
component owns shared implementation files and the build slot. My witness
interface awaits review; baseline execution and implementation remain pending.

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

## My unresolved witness interface

My first actual call provides literal target(3,true), source(4,false). The
unchanged helper returns7 then11 on repeated calls and preserves source(4,false).
Its assertions depend on shared INT4, local fields100/true and8/false,
branch1/false or2/true, and an item field equal to the index snapshot taken
before increment. They do not constrain the changing exclusive target value.
A representative first-call witness is therefore plausible for both variants;
this source argument is not yet an implemented proof.

I require a reviewed, bounded static provenance interface before admission:
exact shared INT/BOOL literals from the actual first-call owner snapshots;
constructor/read/move/unpack scalar provenance; equality of the same immutable
index snapshot; Boolean constants/NOT and exact equality. It must reject an
unproved assertion rather than execute the helper to guess a witness. Branch
proofs merge only equal surviving provenance. Borrowed writes invalidate affected
facts; runtime assertions remain at every original position on every invocation.
A later call with source.value different from4 must still fail, even after a
valid selected witness. I do not invent zero owners: source.value==4 rejects them.

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
local counts through actual regenerated products. Feasibility under128 locals
and the existing two-function borrowed admission is **unmeasured**. I preserve
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
- I test unchanged, else/zero-loop and former-refusal source variants; actual
  assertions include100/8/1/2/index and source4, with target7/11 for combine.
- I reject invalid or nonliteral witnesses and unsupported assertion provenance;
  a later false assertion must preserve prior effects and terminate correctly.
- I retain original-verifier refusals separately from defensive facts checks,
  complete literal/type/count bounds, atomic old-output preservation and old
  byte goldens. Full required rows9538/10409 and #999 remain open.
