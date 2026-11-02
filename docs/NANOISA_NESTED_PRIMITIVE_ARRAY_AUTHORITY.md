# My bounded depth-two primitive array authority contract

I own issue1001's isolated `fix/5.1-nested-primitive-authority` from5bb06c298.
I extend only exact `array<array<u8>>`, `array<array<float>>` and
`array<array<bool>>` fields, completing the five primitive depth-two leaves
beside already-qualified INT and STRING. I do not extend depth, callables,
resources, arbitrary recursive shapes, foreign operations or cycle admission.
The coordinator owns combined raw bootstrap and platform qualification.

## Accepted-source evidence and missing authority

My FLOAT record/global/result fixture in
`tests/nanoisa/fixtures/nested_array_contexts.nano` has retained paired VM/native
Linux/Darwin evidence in `docs/evidence/emitter-nested-contexts-20261009`.
My BOOL record-field fixture in `test_ordinary_declaration_provenance.py` passed
actual VM/native execution in the nested STRING receipt, while omitting authority.
My checked nested U8 literal `[[300], []]` preserves44 and an empty child in
`test_selfhost_array_literal_context.py`; retained hosted Linux evidence is in
`docs/evidence/imported-global-context-20261019/hosted-linux.log`. This local-array
case does not establish U8 record-field authority. I require that missing exact
record-field path before publishing the new descriptor.

## Representation and unchanged budgets

I preserve the existing INT bit7 and STRING bit8 normalized obligations; new
U8/FLOAT/BOOL obligations use independent uint16 bits9/10/11. They are not tag
masks, wire tags or wildcard ARRAY claims. Original descriptor rows remain
available and each ARRAY row refers to exactly one primitive leaf row.

`sizeof(Analysis)` itself is budget charged, so I must not add pointer members or
unconditionally widen field arrays. I retain the old one-byte field allocation
and existing optional STRING bitmap. Three named private reserved byte codes
represent the new nested leaves; only `record_field_element` decodes them into
normalized uint16 obligations. I audit all raw reads and writes so these codes
never enter a tag mask. I reuse the existing nested_arrays byte as explicit
profile-presence flags, preserve its size/alignment, and retain all existing
truth tests as nested-presence tests. A new leaf is permitted in unbound nested
origins only when its exact descriptor is present. I do not enable arbitrary
nested child tags simply because some ARRAY descriptor exists.

I compare old flat, INT and STRING peak bytes, work, allocation counts and exact
LIMIT boundaries before/after; old inputs allocate no new storage and incur no
new charged scans. New inputs use the existing bounded nested scan and field
storage. Typed MEMORY, LIMIT, INVALID and UNRESOLVED results remain distinct.

## Proof, runtime and transport

For every outer origin I join surviving children and all_writes, then require
one exact primitive inner tag. Empty outer arrays, detached/replaced aliases,
forward calls/results/globals and admitted acyclic tails carry the obligation.
INT/U8/FLOAT/BOOL/STRING alias collisions refuse even when machine storage widths
coincide. Each copied-plan validator independently decodes original leaf rows.
Public verifier, assembler and decoded transport independently rederive the
complete proof; no previous eligibility result is an execution token.

Private/public runtime checks retain the exact inner tag, valid packed storage,
length/capacity and owner lifetime. U8/BOOL use existing byte storage and FLOAT
uses existing double storage. U8 contextual narrowing belongs to existing
producer/runtime semantics: accepted `[[300], []]` emits an INT literal into a
U8 array literal and reads back U8 44. A constraint-local origin bit permits
historical INT writes only for children reached from exact nested U8 authority;
the declared/required/read mask remains U8, all unknown/origin checks remain,
and old flat U8 permissions do not change. FLOAT integer-literal narrowing is
not added: the attempted source fixture is rejected by the existing checker. FLOAT tests include signed zero, infinities and
NaN payload bit preservation through copy/slice/aliases where the existing
operations promise bit-preserving storage. A numeric equality assertion alone
does not qualify these values. Packed scalar leaves add no heap children or
release depth. Outer duplicated child ownership retains cumulative-headroom,
allocation-failure cleanup and independent valid retry.

Canonical native forward-result writes must preserve unknown receiver storage
until exact child shape convergence. I add only proved U8ARR/FARR/BARR to the
existing bounded ARR/SARR case if tests demonstrate it is needed; I do not
admit FNARR, arbitrary NARR depth or callable payloads.

## Ordered acceptance

1. Record/review this contract and capture old exact resource counts before edits.
2. Qualify each new primitive descriptor and complete proof: wrong leaf/ref/depth,
   mixed five-leaf graphs, crossing aliases, detached writes, empty/refill,
   malformed encoding and exact typed resource boundaries.
3. Qualify actual public/private VM, public assembly/load and copied plans; retain
   U8 record values and FLOAT bit observations, out-of-range errors, owner
   headroom, failure prefixes, release cleanup and recovery.
4. Qualify canonical native O0/O2 ASan/UBSan programs with forced collection and
   array/record aliases, calls/tails, mutation and distinct primitive storage.
5. Only then extend paired C/Nano source serializers. Keep old flat/INT/STRING
   wire bytes and row ordering unchanged, exact mixed leaf referents, checked
   imported declaration provenance, selected-shadow classification, full-module
   optional omission and hard-error/prior-output contracts.
6. Retain source and wrapper-failure evidence, then hand off a reviewed component
   for fresh aggregate generations and Linux/Darwin gates. Full issue1001 stays
   open; untyped empty replacement inference remains separately tracked.

## I reconcile this contract against the current aggregate

At7f3fb215e I retain a new original-source baseline in `evidence/nested-primitive-current-baseline-20261021/`. The unchanged BOOL record fixture and unchanged FLOAT contexts fixture, plus an explicit U8 record fixture preserving300→44 and an empty inner array, all compile/verify/execute through seed and both installed stages and O1 ASan/UBSan native outputs. All nine cases omit ordinary authority. This establishes the previously missing U8 record-field accepted path; it does not establish descriptor/runtime proof.

The old471738dd6 worktree still contains five uncompiled consumer draft files. I preserve them untouched and hash them in this baseline; I do not claim their correctness or silently rebase them. Before implementation I must review the delta against the current proof/runtime, recapture current flat/INT/STRING allocation/work/peak limits, and retain exact old boundary behavior. No new publisher, native shape, runtime or schema is admitted by this baseline. The current aggregate remains frozen while hosted capture/sanitizer/coverage qualification proceeds.
