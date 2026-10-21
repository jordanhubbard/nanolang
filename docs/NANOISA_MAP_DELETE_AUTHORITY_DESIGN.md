# Exact STRING-to-INT map removal: review checkpoint

I base this contract-only proposal on d5bc83d23 and issue #1001. No implementation, new baseline run or admission is claimed. The existing paired publication component is complete locally; fresh generations and current-head platform acceptance remain independent.

## Evidence and exact scope

My exact candidate is the existing `integer` variant in `tests/test_map_field_operations.py::test_paired_map_remove_preserves_void_and_aliases`: read unchanged `tests/unit/test_map_field_operations.nano`, replace `HashMap<string,string>` with `HashMap<string,int>` and `"kept"` with `42`. All original operations, assertions and shadows remain. The retained `docs/evidence/map-remove-lowering-20261021/README.md` records seed/fresh C-hosted Nano verification, VM and O0/O2 sanitized native success, alongside aliases, once-only key evaluation and missing removal. Its test explicitly expects absent ownership. This is accepted-source evidence, not a new measured selected-authority result.

The unchanged checked-in STRING-to-STRING fixture is broader: it needs a separate stored-value ownership contract as well as DELETE. I do not include it here. Other pairs, HM_KEYS/VALUES, standalone map globals/parameters/results, native dynamic-string families outside the current target, STRUCT/nonidentity native restrictions and all existing limits remain unchanged.

## Existing semantics and missing invariant

`vm.c` HM_DELETE consumes key and receiver, invokes `vm_hashmap_delete`, releases the temporary key, and returns the same receiver edge. Both lowerers POP that returned map for source-level VOID. `heap.c` removes a found entry by clearing its key/value, setting state2, adjusting live/tombstone counts, then releasing the stored edges. Absent removal changes nothing. Aliases observe mutation of the same object; no replacement map is returned.

Native `nmap_delete` uses a chained bucket representation, unlinks the matching entry, frees its copied key/value and entry, decrements count and returns the same map. No tombstone exists there. Its C-string equality remains restricted by the current native NUL-free producer closure; VM counted-string keys are independent. Existing generic native behavior is not selected-target qualification.

Selected VM adapters and validators currently require zero tombstones. `record_map_heap.inc` rejects state2 in full topology validation and probes; `heap.c` private DAG release asserts zero tombstones and occupied state1. Ordinary import and private observations use this same topology contract. Merely admitting the opcode would violate these invariants after the first successful removal.

## Proposed policy requiring review

I propose explicit tombstone support for selected maps, matching the ordinary representation rather than shifting unrelated entries or rebuilding a map during deletion. State0 remains empty, state1 is an exact STRING-to-INT live entry, and state2 is a tombstone with VOID key/value. Counts must satisfy live+tombstones <= buckets; a charged full scan independently checks both counts. Unknown states and non-VOID tombstones refuse. Lookup continues through state2, stops at state0, and remains bounded by bucket_count. For each live entry, charged hash/equality/reachability validation must prove it is the first equal live key reachable from its hash; tombstones do not erase duplicate checks. Existing counted-byte/hash, header, managed-pointer and owning-heap preconditions remain.

Insertion must reuse a tombstone with exact count adjustment, account for live+tombstones in growth decisions, and preserve retain-before-publish/headroom guarantees. Rehash may discard tombstones only on successful commit; failure preserves the original table and accounting, including any earlier committed deletion; it does not roll deletion back. Removal itself needs no allocation or additional retain: validate before mutation, clear stored edges before release, and transfer the existing receiver operand. Equal-content but distinct query keys release only their own consumed edge in the handler; the stored key edge is released exactly once. With identical stored/query identity, all lookup/comparison reads finish before stored-edge release and the still-owned query operand protects the later handler release; no post-release key read is permitted. Ordinary collector buffering remains intact. Private teardown skips proven empty/tombstone entries and releases live STRING keys inline; no release-stack depth increase or new scalar-owner rule is implied.

Old no-tombstone byte/work/allocation charges and allowances must remain exact, including the six legacy proof budgets and existing map preparation/query budgets. New tombstone traversal must be explicitly bounded and charged only on the new path; no hidden unconditional scan, allocation, object-size change or cap increase is authorized.

This changes a shared representation invariant and therefore needs explicit old-route, private and public qualification before any opcode enrollment. I do not silently relax existing header predicates.

## Proof and immutable correspondence

The existing opcode set is NEW/SET/GET/HAS/LEN. A new DELETE operation would require exact known receiver origins and exact STRING key. Its ISA result is that same HASHMAP origin set, not VOID; source POP remains unchanged. No presence fact follows from key spelling or HAS. GET retains INT|VOID, including after removal.

Deletion must not subtract anything from historical map key/value writes, record fields, array children, globals, calls or return summaries. A removed or detached object remains subject to every past/future write obligation and alias constraint. No static origin is recycled, and no runtime pointer becomes a static origin. The shared64 origin namespace and existing bounds remain.

Correspondence must be updated consistently in structure admission, mutable transfer, frozen execution recipes, independent copied-image proof, both preparations, private and borrowed invocation preflight, runtime dispatch, and the shared native target classifier. The invalid-base-recipe sentinel stays distinct. Every encoded instruction including unreachable DELETE must be checked. Unsupported-before-malformed query semantics and publication hard-failure/omission distinctions stay unchanged.

## Qualification before publication

1. Direct heap controls: counted-NUL VM keys, colliding keys before/after removed buckets, duplicate detection through tombstones, all-tombstone and wraparound probes, missing/repeated deletion, reinsert and growth. Test identical stored/query identity and independently allocated equal-content keys, saturated references, buffered ordinary aliases and private final teardown. Corruption tests restore valid storage before cleanup.
2. Import/observation: fresh invocation of a valid tombstoned record-map graph, exact charged lookup/copy/identity queries, malformed count/state/key/hash/topology refusal without normalization, output preservation and exact budget boundaries. Preserve ordinary GC metadata and external aliases on refusal.
3. Allocation failures: deletion itself has no allocator claim beyond verified path; insertion/rehash after deletion must exercise every reached transient/persistent failure and independent retry. Keep heap statistics and independent live-payload checks distinct.
4. Execute exact source variant and independent original shadows using current producers and explicit annotations before publication. Both VM dispatches, private/public roots, repeated calls, first-error unwind and alias survival; native O0/O2 sanitizers/forced GC, exact failure diagnostics and generated allocation prefixes. Assert ISA returned receiver versus source VOID and absent-key semantics separately.
5. Re-run old private/public map and array neighbors, byte goldens, six deterministic proof budgets and structure sizes. Only after consumer gates pass may complete staged publisher query admit this candidate; no syntax-filter or new wire schema is required. Fresh-stage/platform runs remain separate acceptance gates.

## Proposed file boundaries

Heap policy and tests: `src/nanovm/record_map_heap.inc`, `heap.c`, internal checked-adapter declarations, import/observation tests. Proof/recipes: `src/nanoisa/record_map_structure_private.h`, `managed_array_shapes.c`, map execution recipe definitions and copied target classifier. Execution correspondence: `record_map_common.inc`, private/public preparation/context includes and `vm.c`. Native runtime deletion already exists; only independently justified target/shape correspondence changes should be considered after qualification, not a blanket generic runtime rewrite. Publishers should retain existing complete-query dispatch; expectation changes follow consumers. Exact code edits and numeric charge changes require a later source review.
