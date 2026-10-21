# I propose exact STRING-to-INT map consumers

I own this contract-only #1001 checkpoint on `design/5.1-map-consumer-contract`, from aggregate `0df9575bb`, including declaration `2810567e6`. I change no admission, interpreter, native emitter, publisher or schema. The declaration phase's copied exact facts are not execution authority. This proposal requires review before implementation.

## I preserve the accepted input and limits

My first unchanged anchor is `tests/nanoisa/fixtures/map_record_fields.nano`: CollectResult holds ARRAY(STRING), MAP(STRING,INT), BOOL; collect returns a record, forward preserves its identity, remember receives a record/string/int and returns VOID, and main observes mutation through the original alias. Ordinary and independently selected original-shadow inputs remain distinct qualification products. I do not infer support for standalone map parameters/results/globals, map arrays, nested maps, arbitrary pairs or DELETE. Coarse local HASHMAP descriptors remain permitted, with exact identity supplied by instructions and flow, never by a local name. STRING-to-STRING and map_remove parity remain separate from selected authority.

Kind5/revision1 and simultaneous extensions<=4 remain unchanged. All existing record/array depth, layout, field, function, stack, origin, instruction and budget limits remain. Map origins consume the existing shared 64-bit origin namespace; they do not get an extra64 identities. Existing supported record/array/control/call compositions retain their proof obligations. The first added opcode set is exactly HM_NEW, HM_SET, HM_GET, HM_HAS, HM_LEN. Every instruction is structurally checked, including unreachable instructions; an unsupported map opcode refuses the whole selected profile.

## I make the proof context explicit

The following are proposed private interfaces, not implemented exported APIs:

```c
typedef struct NvmRecordMapContext NvmRecordMapContext;
typedef struct NvmRecordMapOrigins NvmRecordMapOrigins;
typedef struct NvmRecordMapExecutionPlan NvmRecordMapExecutionPlan;
typedef struct NvmRecordMapImage NvmRecordMapImage;
typedef struct VmRecordMapPrivate VmRecordMapPrivate;
NvmArrayEligibilityResult nvm_analyze_record_map_origins(
    const NvmModule *, NvmRecordMapOrigins **);
NvmArrayEligibilityResult nvm_prepare_record_map_execution(
    const NvmModule *, NvmRecordMapExecutionPlan **);
NvmArrayEligibilityResult nvm_record_map_image_copy(
    const NvmRecordMapExecutionPlan *, NvmRecordMapImage **);
NvmArrayEligibilityResult vm_record_map_private_create(
    const NvmModule *, VmRecordMapPrivate **);
VmResult vm_record_map_private_run(VmRecordMapPrivate *);
```

Every owning type gets its matching free/destroy. Every constructor initializes caller output only on complete success. INVALID, UNRESOLVED, LIMIT and MEMORY remain distinct. Map-only wrappers own an unchanged base object and separately owned context, never place pointers in padding/reserved fields, and never consult a global registry. No caller-supplied declaration plan is trusted as proof.

A context contains an owned declaration plan, per-origin exact key/value tags and historical key/value Value summaries, and per-field pair identifiers. These are explicit arrays allocated only on the map route. The unchanged Analysis/Value/Function/GlobalFlow/base-plan/image/private-runtime structures acquire no members. Internal map-aware variants of structure preparation, analysis, transfer, global_refine, record_value_matches, final_check and publication receive `NvmRecordMapContext *` explicitly; old entry points call them with NULL. Signature changes must reach every recursive helper/call site, not store context in a spare flag or recover it from a module address.

Proposed immutable queries copy facts, not borrowed context addresses: map-origin(index)->{origin ordinal,key tag,value tag}; map-field(layout,field)->exact pair; historical-write(origin)->key/value tag sets plus unknown flags. Original origin ordinals index the same record/array/map namespace. The map result wrapper exposes its existing base record/array report only for its lifetime. A coarse HASHMAP tag without nonempty known map-origin bits is UNRESOLVED.

## I require monotone origin and write closure

- HM_NEW requires encoded exact STRING,INT and creates one origin at its existing instruction identity. Repeated execution of the same site shares a static origin summary, not a runtime object identity. All writes from every activation/iteration accumulate in that summary.
- HM_SET consumes map,key,value and returns the same possible map origins. Every possible receiver must be a known map origin with the same exact pair. Key must be exactly STRING and value exactly INT, with unknown/mixed alternatives refused. Historical key/value facts grow monotonically; overwriting a record field or replacing a map entry never erases earlier writes.
- HM_GET/HAS require exact STRING keys and exact receiver origins. GET returns INT|VOID, including an absent key; HAS returns BOOL and LEN INT. HAS does not prove a subsequent GET present. No map GET produces nested heap origins in this slice.
- Pack/get/set of a record MAP field validates each possible origin against the declaration pair. Record field historical unions retain detached maps after replacement. Pair annotations constrain existing origins; they cannot manufacture or relabel one.
- LOAD/STORE/DUP/SWAP, joins, arguments, results, tail arguments and globals preserve the same origin bits. `global_refine` currently retains only ARRAY/RECORD bits (`managed_array_shapes.c`); its explicit map-context branch must preserve HASHMAP origins only when the refined physical tag includes HASHMAP. No normalized ARRAY mask is reused for map pairs.
- Calls pass record origins and exact scalar arguments through existing seed/result/global fixed points. Scalar STRING keys carry their existing tag proof, not string-content interpretation. VOID remember calls retain global/heap effects. A returned record's map edges remain in the result summary. Missing return summaries or unknown required global/result origins refuse; an unused unconstrained global is not itself proof of a violation.
- Zero iterations preserve initial roots; entered/final-false paths join all writes. Branches and recursive/tail summaries cannot discard historical origins. Existing graph policy stays unchanged; this is no new SCC admission.

## I account for storage without changing old receipts

Map declaration accounting remains its reviewed2MiB/4Mi logical-work policy. The new proof context uses existing record-array64MiB/16777216-step accounting for every allocation and traversal on the selected map route. No increase is proposed. Its ledger must include wrapper sizeof, copied declaration ownership, per-origin pair descriptors (at most64), two Value summaries per origin, at most65536 per-field pair indices, and overlap while constructing copied results. If declaration storage is separately budgeted, its live bytes must also be reserved in the enclosing peak; success under two independent caps must not hide their sum.

All indexed/field queries precharge the published bounded query cost (20 for a field lookup, including misses). Fixed-point revisits and each origin scan are charged; used tables do not waive work. Integer multiplication/addition for allocation sizes is checked before allocation. Publication and image copying reserve both source and destination lifetimes. Execution/image preparation retains its existing128MiB/33554432 limits, including added pair tables and validation work. Runtime dynamic-map growth is the existing heap allocation policy, not a newly invented element cap. Actual sizeof/peak/work boundaries and six old receipts must be measured before admission. This document does not claim an estimate as a passing budget result.

## I separate private preparation from actual public execution

Existing public `vm_execute_record_arrays` admits then releases its execution plan and runs the shared handlers with `VmOwnedInvocationProof`; it does not use the private copied-image runner. Therefore adding a private wrapper alone cannot qualify public maps.

The proposed public map entry owns a validated map execution plan for the full initializer+entry invocation. It passes a non-owning const map context explicitly through the scoped-call/execute-function chain and shared dispatch preflight, beside existing admission arguments; no VmState field or stale retained pointer is added. The context dies only after stack/frame cleanup. Retained/linked/callback entry routes retain their existing refusal boundary unless independently qualified; a selected-map failure is terminal and never retries legacy execution. Private execution instead owns its copied module, copied context, decode/dispatch image and VmState and must independently revalidate them. Its run/destruction cannot depend on the original module remaining alive.

The public classification audit includes every current `nvm_record_array_candidate` use: module construction, v2 load/save/conversion, assembler final verify, verifier variants, query/publisher validation, nvm2c, VM load/set-module/scoped/retained/linked dispatch and vm_execute. Map framing precedes array/legacy selection, including mixed ARRAY+MAP anchors. ABSENT is only a classification result; malformed metadata must still fail semantic validation. Kind5 presence with unsupported pair is selected-and-refused, never absent.

Proposed map-aware structure and type-verifier variants permit only these exact opcodes and descriptor rules after complete declaration validation. Snapshot recipes copy actual operand bytes, instruction boundaries, stack effects and pair facts; a map sidecar does not excuse missing instruction coverage. New map recipes require TAGS, staged operands, retain-before-release and first-error cleanup; no private recipe enum change alone enables public dispatch.

## I preserve real key lifetime and physical invariants

Current `vm_hashmap_set` first searches a slot; replacement retains the new value then releases the old value and keeps the original stored key. Insertion may resize before assigning/retaining key and value. Shared HM_SET releases popped key/value and returns the original map; GET retains the result before releasing map/key. Missing GET is VOID. These observed source paths are requirements to qualify, not permission to change them silently.

Selected preflight must independently validate HASHMAP object/tag, exact key/value types, nonzero valid bucket capacity, checked entry storage/count/tombstone arithmetic and occupied-entry STRING/INT tags. It must account for map->STRING heap edges in traversal, retain headroom, GC and teardown, including duplicated interned keys and equal bytes from distinct strings. Replacing an equal key must not leak the incoming key or release the stored key. Refcount headroom is checked before any irrevocable insertion/retain; resize failure and duplicate-key replacement get independent failure-prefix tests. If current shared handlers lack required preflight, retain the actual defect for review rather than infer safety from legacy success.

Private observations need explicit map identity/count/pair and key lookup APIs (byte pointer plus exact length), not an invented array-index path. An absent observation reports absent, distinct from malformed/error, and never supplies INT0. Returned string bytes are copied into caller storage; identity tokens remain epoch-scoped. Before choosing a native query ABI, confirm its string representation supports the same accepted byte domain; embedded NUL must not be silently treated as equivalent by a C-string key API.

Canonical native `HM_NEW`/GET/SET currently uses `nmap_owned_new`, `nmap_owned_get`, `nmap_set` and typed map slots. Exact selected admission must precede those existing operations. Record forwarding and map identity must survive forced GC and original-module destruction. We do not claim the private VM's heap layout is the native representation. Missing GET, duplicate key equality, hash collision, rehash, alias writes and error prefix are compared against the original VM at O0/O2 under sanitizers.

## I stage qualification before publishers

1. Read-only declaration/classification and original verifier refusal remain until map structure/proof tests pass. Test exact creation pair, wrong/mixed historical writes, unknown origins, record replacement with detached alias, call/result/global convergence and shared64-origin exhaustion/recovery; preserve old six budgets exactly.
2. Copied plan/image tests mutate/free original code, strings and metadata; corrupt pair/recipe/ordinal independently; require atomic output preservation and allocation-prefix cleanup at every new site.
3. Private and actual public VM tests cover unchanged ordinary and original-shadow anchor, empty/missing maps, replacement/insertion/rehash, duplicate interned keys, independent equal strings, aliases, global/call histories, false ASSERT after map mutation, bounds failure in the anchor's ARRAY neighbor, prior published private result and retry. Global mutations need not roll back. Destruction must leave zero owned objects.
4. Canonical native independently executes the same actual inputs O0/O2 with forced GC and exact identity/failure assertions. Transport includes binary/canonical text and stripped advisory names with literal contents unchanged. Original selected-map modules must not pass via legacy fallback.
5. Only after consumers are green propose paired publishers for exact STRING->INT, with full unchanged anchor and source/query pins. No publisher expectation changes occur in this contract checkpoint. DELETE, other pairs and full #1001 closure remain outside this phase.

The principal review gates are explicit context threading without old-object growth; required public retain preflight; STRING key byte-domain/native agreement; and actual map-only budget fit. I make no implementation or execution claim from this design.
