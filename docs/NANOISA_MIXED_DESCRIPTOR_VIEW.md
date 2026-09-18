# I describe mixed layouts before granting authority

I record task_20d4e142080a449e9098e36c244f31a8 under mixed-value parent
`task_4be28fef163f42069064357639b3b5cc`, against canonical `5079d92e`.
This is my preimplementation contract. My [parent proposal](NANOISA_MIXED_OWNED_MANAGED_VALUES.md)
retains the complete unchanged Samples/PREFIX module and every selected shadow.
PR798 completed only my operand-stack binary64 prerequisites. Ordinary managed
values alongside owners, managed fields inside owners, and full product acceptance
remain open. My separate FLOAT-local/source-unsafe lane grants no authority here.

## My first checkpoint describes facts only

I introduce a private, owned, immutable descriptor view, provisionally named
`NvmMixedLayoutView`. Its successful result means that I described the retained
bytes; it does not mean that I may execute the module. I do not return an existing
`NvmRecordPlan` or change its ORDINARY authority contract. I leave
`check_layout_facts`, `nvm_ownership_layout_authorities`, public verification,
managed selectors, source guards, and runtime dispatch unchanged.

I decode retained layout rows and ownership declarations without calling an
executable verifier. I preserve names, field order, exact tags, numeric references,
flags, function/local/parameter/result positions and reference paths. I validate
existing version/count/padding/index/tag rules, complete/resource consistency,
resource prior-order rules, and exact end of input. I do not ignore malformed
unused rows. I keep existing bounded layout/field/path counts and reject overflow
before allocation. Unsupported well-formed facts remain UNKNOWN; malformed bytes
are invalid. Allocation failure and limits have distinct outcomes. Absent ownership
metadata cannot establish a positive ordinary or resource classification.

My classification is explicit:

| Class | Descriptive fact; never executable admission |
| --- | --- |
| RESOURCE | Exact complete resource declaration and existing scalar-tree/prior-order constraints hold. |
| ORDINARY_STRUCTURAL | Exact complete nonresource record has supported scalar/string fields or recursively structural ordinary children, with no owned or unresolved child. |
| PENDING_ARRAY_PROOF | An otherwise eligible ordinary record contains an ARRAY field or a pending ordinary child; exact element/child provenance still needs proof. |
| UNKNOWN | Incomplete or unsupported facts cannot establish either ordinary or resource authority. |

I require ARRAY fields to retain their existing no-layout-index encoding. I do
not infer element type from a field name, source spelling, generic ARRAY tag,
resource absence, or a successful decode. Pending propagates through containing
ordinary records. An unknown child prevents positive classification; a resource
child cannot become ordinary through a pending classification. I retain the
existing resource field restrictions. I make no new wire version, feature bit,
reserved flag, or layout-index interpretation.

## My identities remain distinct

I retain three indices rather than substituting one for another:

* Source struct ordinal counts every STRUCT row in original per-kind order.
* Global layout index addresses the original retained layout table.
* Compact managed ordinal addresses only eligible ordinary rows in a private view.

I provide checked source-record-to-global and global-to-source-record maps for
all STRUCT rows. Global-to-managed and managed-to-global maps include only
ORDINARY_STRUCTURAL rows, ordered by original global index. Resource, pending,
unknown and nonstruct rows use NO_INDEX in the managed map. Even structural rows
in this descriptive view cannot be bound for execution until the later combined
proof succeeds.

For example, an unrelated nonstruct row at global0, scalar ordinary record at
global1, Handle at global2, and Samples at global3 give source struct ordinals
0, 1, 2 for those three records. Only the scalar ordinary record has managed
ordinal0. Handle remains RESOURCE, Samples remains PENDING_ARRAY_PROOF, and both
retain NO_INDEX in the managed map. After a later positive Samples proof, a new
certified view may give Samples managed ordinal1 while retaining global3/source2.
I never change the module's AGG_PACK operands, nominal IDs or OWN instructions.
I never clone metadata with RESOURCE cleared to satisfy an ordinary binder.

I build the view privately and publish it only after complete success. Failure
leaves the caller's previous output and module bytes unchanged. The view owns its
decoded storage and has an explicit destructor; input pointers do not outlive the
query. Repeated description has no module cache or persistent trust state.

## My next proof consumes these facts without recursion

I separate three dependencies: structural extraction, finite closed transfer
analysis, and final executable conjunction. Transfer analysis consumes immutable
facts without calling public nvm_verify or a selector that calls it. Shared
ownership validation never calls shape analysis. If I factor a syntactic decoder,
existing validators must retain their acceptance and authority outputs.

My subsequent proof identifies allocation origins by function and decoded PC,
with bounded origin sets, exact possible tags and an explicit unknown bit. Typed
float array construction must prove its flat element shape even when empty.
Copies preserve origin identity; record construction and projection propagate
field origins. Joins union every reaching alternative; writes must preserve the
proved element/field contract for all aliases. Repeated allocation at one site is
an abstraction over every instance, never proof that only one object exists.
I use a bounded deduplicated worklist and conservative refusal on its finite
iteration/cell/origin limits. Loops require convergence, not a one-pass guess.

I prove all required functions and selected shadows in the existing bounded
acyclic owned graph. Owner/reference facts remain separate exact obligations;
ordinary provenance cannot weaken their joins or convert observations into
ordinary values. Unknown sources, imports, callbacks, externally injected arrays,
owned array elements and unsupported record alternatives refuse this first slice.
Its ordinary records form a finite noncyclic graph ending in flat scalar arrays.
Ordinary managed parameter/results remain separate: Samples and close need only
the already supported scalar/owned signatures.

I derive shape from existing bytecode allocation/field origins if that suffices.
If it does not, I retain a pending refusal and record the missing transport
contract before proposing a format change. The final mixed gate explicitly
requires structural/instruction verification, exact affine obligations and the
positive ordinary provenance certificate. The old authority query alone continues
to refuse ARRAY metadata. A failed old or new verifier is not a fallback route.

Raw ARR_GET returns VOID for a missing index. My later transfer analysis must
preserve FLOAT-or-VOID possibilities unless it independently proves bounds; my
native operation must match that behavior. I do not turn absent elements into
zero, fabricate an exact FLOAT result, or invent an index trap. Subsequent
consumers and assertions retain their actual VM checks.

## My runtime dependency preserves every root

The later native carrier distinguishes scalar, affine owner, owner observation,
and ordinary managed handle. I reuse NmsRuntime's qualified ordinary record and
flat-array storage while retaining existing affine owner records and reference
contexts. One fresh managed context belongs to each public invocation and is
shared across its helper frames. Its certified ordinary descriptor table preserves
original global identity through compact ordinals and remains immutable/alive
until context disposal.

Each ordinary stack/local reference owns one root. LOAD/DUP/projection retain;
STORE secures the replacement before releasing the old root; POP, scope exit and
frame cleanup release exactly once. Aliases share mutations, not snapshots.
Constructor inputs remain rooted until successful child retention and publication;
partial failure releases only acquired roots and preserves unpublished outputs.
Cleanup audits include stack/local slots, prepared arguments, transient a/c
carriers, partial containers and pending results. I release all frame roots before
disposing the invocation context. I claim no general cycle collector.

Existing owner argument validation, frame reservation, once-only transfer,
reference generations and caller-held borrow survival remain live checks.
Managed allocations cannot retire caller references. No source owner is silently
dropped during ordinary cleanup. Future ordinary managed results need their own
exact identity/preflight/publication contract; this checkpoint adds none.

## My acceptance is staged

For this first descriptive checkpoint I test exact maps with interleaved kinds,
resource and structural rows, pending Samples and nested pending children;
version/count/tag/path validation; missing/unknown facts; output atomicity;
allocation cleanup; and deterministic repeated queries. I test that existing
public validators/selectors retain their prior decisions. I execute no pending
mixed module and claim no source admission from these description tests.

The later proof/runtime checkpoints require independent source-order, alias,
join/loop, wrong-shape, wrong-nominal and owned-child refusal controls; exact
roundtrip; all four VM APIs; native parity; and measured allocation/lifetime
cleanup with caller owners/references still live. Only after those prerequisites
may both source producers lower the unchanged Samples/PREFIX program and full
selected-shadow graph. False mandatory shadows must still fail, and rejected
compilation must preserve previous output. Bundle managed fields inside affine
owners remain a separate required contract. Historical950f failures remain sealed.

My first reader bounds its owned transport copy to 16 MiB before materialization.
Exceeding that private budget returns LIMIT; it changes no shared wire-format
acceptance. I reuse the existing allocation-free retained preflight unchanged,
including its exact forward-layout family, and leave nonrecord rows UNKNOWN.

## My prereview descriptor correction

My first unexecuted reader omitted the existing borrowed-formal root-place check.
I require the materialized second pass to call nvm_reference_place_valid with the
same zero-depth place as the shared ownership descriptor validator. A complete
resource tree may contain nested resources, but a borrowed referent must have
only scalar fields with NO_INDEX. Both shared and exclusive modes retain that
restriction. I add query-only rejection controls for nested resource referents,
alongside accepted leaf modes and accepted unborrowed nested resource declarations.

The prior-only layout codec permits TAG_STRUCT with NO_INDEX. I distinguish that
structural encoding from declaration claims: an incomplete row with that missing
edge remains UNKNOWN; a COMPLETE row claiming it is an invalid declaration.
A present edge must address a STRUCT row. An ordinary parent's correctly indexed
but incomplete child propagates UNKNOWN, including through an otherwise pending
array parent. This is deliberately descriptive uncertainty, not successful shared
ownership validation. RESOURCE still requires a complete scalar tree and rejects
that incomplete child. I retain the codec's stricter existing forward-family
rules and do not change any shared validator to accommodate these descriptions.
