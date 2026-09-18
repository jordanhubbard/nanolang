# I execute exact forward ordinary record DAGs

I propose `task_1fefd3f1f1c14798ad43e7e2daaa66b7` from canonical96f29b4d,
after the reviewed transport/publisher785 and managed execution789 checkpoints.
This contract precedes production. I preserve the qualified789 tree and tools.
My ordinary authority15f, aggregate488 and managed51da parents remain open.

## My concrete remaining restriction

`nvm_v2_layouts_decode` already accepts a table containing forward edges only
when every layout is STRUCT, every field is int/U8/float/bool/string or an exact
STRUCT reference, and the complete declaration graph is acyclic. Its iterative
three-color walk checks disconnected components too. Prior-only tables retain
the older path and allocation behavior. This structural fact grants no authority.

`ownership_contracts.c` independently accepts COMPLETE ordinary forward facts.
Any RESOURCE flag keeps every nested edge prior-only, including edges in UNKNOWN
or disconnected definitions. `nvm_verify_owned_module` also explicitly retains
prior-only edges even for COMPLETE ordinary layouts used with owned instructions.
I do not change either boundary.

The remaining managed restriction is `record_preflight` in `retained_layouts.c`:
it rejects every nested index greater than or equal to its containing layout.
The descriptor query performs this allocation-free check before the older codec,
so a later codec TRUNCATED error can be classified as allocation MEMORY rather
than confused with invalid input. Simply removing the index comparison would
lose part of that preflight guarantee; my extension replaces it with a complete
bounded check of the newly accepted graph family.

## My descriptor preflight and publication

I retain existing name, padding, tag, exact byte length, per-kind count and field
count checks. Every nested index must be in range and cannot name its own layout.
I detect whether any edge points to a later declaration without changing indices.
For a prior-only table I preserve the existing checks and allocation sequence,
including descriptive UNKNOWN authority, interleaved enum entries and explicit
empty-versus-placeholder behavior.

For a forward-containing table I add allocation-free validation equivalent to
the codec's existing forward family:

- Every retained layout must be STRUCT; no enum/union/tuple or mixed table enters
  this new path. Int/U8/float/bool/string fields have NO_INDEX. STRUCT fields name
  one exact in-range STRUCT layout. VOID, array and unknown field tags do not
  enter the forward family.
- I retain bounded per-layout field offsets/counts from the byte scan, then use
  fixed-size colors and an explicit DFS stack. I scan every disconnected root,
  reject a gray-edge cycle, and finish each layout once. No recursive C call or
  dynamic scratch allocation is necessary within the existing256-layout limit.
- Cursor offsets use bounded checked byte reads. Field cursors are wide enough
  for all65535 fields in an entry. The existing65536 total retained-field limit
  is checked before the graph walk; no caller-controlled stack size is used.
- Only after this check succeeds do I call the existing allocating codec. Its
  extra forward color/stack allocations are already bounded by the checked count.
  A codec allocation failure retains MEMORY classification. Structural/schema
  failures are INVALID, cap failures LIMIT, and unsupported authority UNRESOLVED.
  I preserve the existing ambiguous ownership-validator failure as UNRESOLVED;
  this child does not solve that separately recorded diagnostic limitation.

A forward plan is published only after the existing full ownership query proves
COMPLETE ORDINARY authority for every record. Absent ownership payload is
UNRESOLVED for this new forward path, rather than publishing a new UNKNOWN plan.
The old prior-only descriptive UNKNOWN result remains unchanged. Unknown facts,
resource flags and reference/owned execution contracts supply no permission.
Every failure leaves the caller's plan pointer and borrowed module unchanged and
releases private codec/maps/plan allocations. I neither reorder nor synthesize
nominal identities, signatures or local tags.

This is a descriptor implementation change in the existing translation unit;
I add no wire format, exported ABI field, source compiler change or link-source
dependency. I update the obsolete prior-record diagnostic in the analysis to
state the exact accepted acyclic ordinary schema after the descriptor qualifies.

## My analysis and runtime order audit

`prepare_records` already requires ORDINARY authority and no affine/reference
execution. `record_site` resolves the executable per-kind ordinal through
`record_to_layout`; `record_value_matches` compares each possible child origin's
exact global layout index with `nested_idx`. It does not require a smaller index.
GET/SET use each receiver's indexed field schema. The existing monotone weak
summaries, bottom/unknown distinction, call/recursive-SCC/global joins and the
shared64-origin and1048576 abstract-cell caps remain unchanged. Forward declaration
order does not replace those instruction-flow obligations.

`nvm_select_managed_heap` already requires both the descriptor and field report
before selecting record execution. I preserve that conjunction and its failure
atomicity; there is no fallback to a weaker mode after a record failure. This
child deliberately extends only the previously refused exact forward family.
CLOSED_SCALAR and CLOSED_LITERAL_STRINGS still reject nominal modules.

Runtime `NmsRecordDescriptor` contains only global_layout_index and field_count.
Its table is emitted in existing record-ordinal/declaration order. `nms_bind_records`
requires ascending declaration indices, not topological edge order; all-record
forward tables satisfy this without sorting. Record slots retain their ordinal
across table movement. Creation retains already-live child handles transactionally;
GET/SET and graph collection traverse actual tagged value edges, not declaration
order. No core, module-adapter or emitted instruction change is expected.

I retain789 counted stack/local/global/call roots, preallocation safe points,
acquired-bit entry cleanup, first-error behavior, terminal disposal and exact
record identity. Exact acyclic nominal schemas still cannot encode a record
cycle through these qualified fields. Previously qualified independent array
graphs may coexist under the unchanged analysis, but array-valued record fields,
record-valued array elements and new mixed heap shapes remain refused.

## My review and actual acceptance

I send the bounded descriptor/diagnostic production checkpoint for independent
review before executing fresh fixtures. Then I qualify:

1. Descriptor/query controls for forward chains, diamonds, disconnected components,
   empty/string/scalar leaves, permuted declaration order and distinct same-shaped
   records. I compare exact ordinal/global-index/field-origin reports and unchanged
   module bytes; I do not compare only successful status. Existing prior-only
   results, allocation budgets and placeholder distinctions remain controls.
2. Checked structural/authority refusals for cycles, self edges, wrong field tags,
   mixed layout kinds, absent/UNKNOWN/resource authority and wrong nominal writes.
   Safe allocation controls cover codec table/field/DFS scratch and owned plan/map
   publication, preserving output pointers on failure. The preflight must reject
   invalid structure before allocating, without replaying historical failures.
3. Normal verified VM, actual generated native LLVM with sanitizer harnesses,
   import-free Node/Wasmtime and public Wasm publication for ordinary forward
   records. Fixtures exercise construction before parent use, shared children,
   GET temporary lifetime, SET alias visibility, reordered calls/results, globals,
   same-instance/fresh-instance reentry, exact binary64 bits, allocation pressure
   and bounded-live cleanup. Existing normal source producers may supply forward
   facts, but no new bootstrap claim follows from this runtime-only source delta.
4. Replace only the exact newly admitted forward-layout refusal fixture from789
   with paired execution. Preserve neighboring unknown/wrong-nominal/array-field
   refusals, source/output preservation, resource-table and owned-profile prior-only
   controls. Run affected descriptor/field-origin/selector/profile and existing
   managed record/private-runtime regressions; no unchanged full corpus repetition
   is required without a new failure or shared runtime source change.

I freeze source and tools for each gate, retain first terminal outcomes and
actual source pins, seal the report hashes, and land through reviewed canonical
integration. This closes only the forward ordinary DAG execution gap. Broader
array/generic/import authority, mixed heap storage and full runtime/release scope
remain in their original parent tasks.
