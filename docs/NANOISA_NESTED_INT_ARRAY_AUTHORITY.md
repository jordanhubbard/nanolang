# My bounded nested integer-array field authority

I implement one bounded required source-authority slice under #1001, based on
`0f9ca2300`. My isolated owner is `workspace_repairs` on
`fix/5.1-nested-array-authority`. This contract was reviewed before admission changes. My focused local receipt
records implementation coverage separately from outstanding aggregate qualification.

## Exact boundary

I retain the existing flat INT/U8/FLOAT/BOOL/STRING field contracts and add only
`array<array<int>>` ordinary record fields. My accepted source example already
executes without optional authority in `test_ordinary_declaration_provenance.py`.
I do not extend arbitrary nested depth, nested strings, record-element arrays,
resources, foreign calls, callable fields, recursive nominal declarations or
cyclic tail edges here. I preserve old non-array and flat-array behavior.

My v3 ARRAY_FIELDS grammar expresses the retained binding: the field
binds an ARRAY type row whose referent is an INT row with NO_INDEX. I preserve
that wire representation, original runtime layout ordinals, and exact checked
source-owner provenance. I do not infer element authority from a bare ARRAY tag
or advisory source text. Valid but unsupported descriptors remain UNKNOWN or
UNRESOLVED; malformed extents/referents/cycles remain INVALID, budget exhaustion
LIMIT and allocation failure MEMORY.

## Proof representation checkpoint

I retain each field's copied element type index rather than collapsing its
contract to one tag byte. During preparation I normalize the checked descriptor
into one internal requirement: each existing flat leaf, or the distinct
ARRAY_OF_INT requirement. Normalization is structural: duplicate equivalent
wire rows mean the same requirement, while different referents never become
equal merely because both outer tags are ARRAY. I preserve the original copied
type rows and indices for transport/execution-plan inspection.

I extend the existing per-origin requirement mask with one distinct nested-int
bit. The existing exact flat masks remain intact. An outer constrained origin
must declare ARRAY, have no unknown contents, and have only array child origins.
Every child origin must declare INT with only INT writes/content and no heap
child origins. I apply the INT obligation to each child origin, including
children reached through aliases or later writes. Conflicting obligations,
non-array children, missing child origins for nonempty contents, mixed depth and
unknown origins refuse publication. Empty outer arrays need no invented child
origin; their exact declared ARRAY tag and all-writes closure must still prove
that every possible later child is an INT array.

I keep the current graph transfer for calls/results/globals, field mutation,
array read/write/pop/slice and terminal tail transfer. I inspect both
`children` and `all_writes`: forgetting a removed child must not erase an alias
write obligation. I evaluate these constraints after convergence, charging
bounded scans to the selected profile's existing work/storage budget. I do not
turn on broader graph admission by changing a tag whitelist alone. Arrays not
attached to a field remain subject to the closed selected profile: only the
old flat arrays or exact depth-two INT graphs may enter this slice.

I review the representation change before changing eligibility. No query result
is a reusable authorization token: public verification, save/load, canonical
assembly and each runtime independently derive the selected proof from the
current module. Eligible descriptor projection alone is not full flow proof.

## Execution and lifetime

I extend copied private VM field facts with the normalized nested obligation
and independently validate its original type rows. Actual values must have a
boxed ARRAY owner whose child values are valid INT-array owners; INT children
use the existing packed representation. I preserve alias identity and require
retain headroom before mutations, including repeated shared children.

My private iterative release stack currently assumes at most256 record layouts
plus one array. I explicitly recalculate its maximum for two array layers and
keep a proved bound; I do not retain a stale258 assertion. Depth-two arrays
cannot contain records or outer arrays, so this slice introduces no new heap
cycles. Existing record DAG policy remains unchanged.

My actual public VM uses the ordinary handlers, separately from the private
copied-plan runner. I qualify both paths and their cleanup, including array
replacement, slicing, duplicate child references, out-of-range operations and
forced allocation failures. A private-only preflight is not public evidence.

My native lowerer already represents nested arrays as NARR kind16, storing a
child storage kind and child handle, and its root walker follows those handles
from canonical record fields. That is groundwork, not proof of selected
canonical-record lowering. I inspect field-read/write shape propagation and
qualify shared inner identity with forced collection in O0/O2 ASan/UBSan C
products. I preserve the existing representation and explicit cycle policy.

## Ordered acceptance

1. I review this contract and normalized obligation representation before
   changing admission; record ownership in #1001 and this branch's roadmap.
2. I qualify descriptor grammar/projection and whole-flow proof with positive
   exact depth-two fixtures, alias writes and independent forged/mismatch
   controls. I keep flat and non-array regressions unchanged.
3. I qualify copied-plan validation, actual public and private VM execution,
   independent save/load/assembly proof reconstruction, native canonical-record
   lowering/roots, and typed work/storage/allocation failure behavior.
4. I extend both source publishers only after consumer qualification. I retain
   checked declaration identity and whole-module host-query staging, independent
   shadow eligibility, unsupported omission and hard-error prior-output rules.
5. I retain focused seed/C-hosted paired source receipts and failure-prefix
   evidence. Fresh raw compiler generations and exact Linux/Darwin qualification
   remain aggregate-owned gates; I do not start a competing bootstrap.

My positives include empty/nonempty outer and inner arrays, duplicate shared
inner arrays, mutation through original and field aliases, field replacement
while old aliases survive, pop/slice, returned values, globals, ordinary
calls and already-admitted acyclic tails. My controls include wrong inner leaf,
conflicting alias writes after binding, deeper nesting, forged/missing type
rows, cycles, unproved contents, out-of-range behavior and existing unsupported
profiles. Allocation-prefix checks require an actually reached injection,
unchanged prior outputs/results where promised, drained roots, independent
successful recovery and counted cleanup. Native forced-GC checks must observe
alias mutation rather than merely successful process exit.

This slice does not close all #1001 criteria. I retain the separately required
composite/recursive authority rows and distinguish missing implementation from
fresh-generation/platform qualification.

My first public consumer run confirmed ARR_REMOVE is outside the existing
selected profile, including flat arrays. I preserve it as an explicit refusal
control; this slice does not add that opcode or substitute unproved semantics.
