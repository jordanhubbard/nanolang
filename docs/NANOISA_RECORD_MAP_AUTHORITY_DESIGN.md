# I propose bounded ordinary map-field authority

I retain a design-only #1001 checkpoint from `878c2bb83` on `design/5.1-record-map-authority`. I change no decoder, publisher, verifier, runtime or admission here. My [baseline](evidence/record-map-baseline-20261021/README.md) establishes the unchanged paired STRING→INT anchor. The unchanged STRING→STRING Holder refuses `map_remove` in the three Nano producers; I retain that separate frontend boundary.

## I preserve the actual anchor

`tests/nanoisa/fixtures/map_record_fields.nano` combines `array<string>`, `HashMap<string,int>` and BOOL fields in `CollectResult`, returns and forwards that record, mutates its map through an alias and a VOID helper, and observes the original record. I must preserve the whole source, calls, aliases and assertions. I do not infer INT keys, STRING values, nested maps, map enumeration or map-bearing resource ownership from this acceptance.

## I compare two explicit grammars

My preferred candidate is a separate mandatory-understanding `MAP_FIELDS` extension. Kind 5 is an available candidate, not assigned by this document. I retain ownership v3 and the simultaneous extension-count ceiling of four: ARRAY plus MAP fits the anchor, but all five kinds together would explicitly refuse. Existing kinds 1–4 and their bytes remain unchanged. Old readers reject unknown kind 5; a five-record payload also exceeds their count ceiling. I must test both boundaries rather than silently increase `MAX_EXTENSIONS`.

The candidate payload has a bounded scalar type table, an exact pair table containing two u32 type indexes (key, value), and sorted field bindings `(layout:u32, field:u16, reserved:u16, pair:u32)`. Scalar type rows may reuse the existing eight-byte scalar encoding with NO_INDEX referents, but have their own explicit table identity. The first admitted pair is exactly STRING→INT. Every complete HASHMAP field needs one binding; duplicate, missing, wrong-tag and out-of-range bindings refuse. Equal pairs may share a pair index; nominal field identity remains `(layout, field)`. No advisory name supplies authority. Exact ceilings, checked size arithmetic and charge order require review before implementation; they are not inferred from the ARRAY table's ceilings.

The alternative is ARRAY_FIELDS revision 2 with explicit key/value rows and map bindings. Old readers already reject revision 2, so this can fail closed, but it couples independent map declarations to array grammar and needs revision-aware queries throughout. Existing eight-byte ARRAY rows have only one referent: neither alternative may reinterpret its reserved bits, overload an ARRAY referent as a pair, or accept HASHMAP without both exact endpoints. I prefer the separate extension because the anchor naturally retains its unchanged ARRAY declaration alongside the new pair declaration.

## I keep old allocation identities intact

`NvmOwnershipExtensions` in `ownership_contracts.c:158` is a call-local four-view structure, not embedded in the charged analysis/plan. A map view can instead be returned by an explicit optional output parameter while existing callers pass NULL; the full reader must still validate every present extension. This avoids even enlarging the old local view aggregate. It does not by itself solve map proof storage.

I must preserve these existing charged allocation sites:

| Existing allocation | Existing accounting | Proposed constraint |
|---|---|---|
| `Analysis` | `managed_array_shapes.c`, sizeof before allocation | no added fields or unconditional allocation |
| `NvmOrdinaryArrayAuthority` and layout/type/binding tables | `ownership_array_fields.inc:35–42` | old sizeof and old table rows unchanged |
| `NvmOwnershipDeclarationPlan` and union rows | `ownership_declaration_projection.inc:34–48` | no added map pointer; old preflight unchanged |
| `NvmRecordArrayStructure` | structure preparation charges sizeof | no added map pointer |
| execution plan, snapshot and image | `managed_record_array_execution.inc`, sizeof plus copied storage/work | old layouts, copy path and charges unchanged |

A candidate map-only context owns validated pair/binding tables, origin-pair facts and map field/call/global relations. It is allocated only after detecting and fully framing MAP_FIELDS, with checked count×width, temporary-overlap reservations and explicit failure cleanup. A separate wrapper can own the unchanged ordinary plan plus this context on the new route; old callers keep their old object. I must not hide a map pointer in padding, an ARRAY referent, a global registry or advisory metadata.

## I require explicit proof threading before implementation

The proposed interface is a private optional map-proof context passed explicitly through the new map route's transfer, join, summary and final-constraint functions. Existing routes pass NULL and retain their old charges. A declaration query returns an exact field pair only after validating the whole module; a map-origin query returns the original creation identity plus exact key/value pair, never just TAG_HASHMAP.

The context must connect to the existing value-origin set without adding unconditional per-Value storage: map origins must participate in the same bounded identity namespace and alias sets, with map-only per-origin facts outside Analysis. This is a proof obligation, not yet an implemented guarantee. Before admission I must show all relevant joins preserve the map identities, record pack/get/set links retain their field pair, and call arguments/results and global summaries carry those same origins. If the existing Value representation cannot carry these sets exactly, I stop for representation review rather than launder them through a coarse tag or widen old storage.

For each new HM_NEW origin I retain its exact opcode pair. Every historical/current HM_SET key and value must match that pair; DELETE cannot erase earlier invalid writes. Record field bindings constrain every possible origin, including aliases detached by later replacement. Missing HM_GET remains the existing VOID result, never an invented INT. Unsupported or unknown origins and unresolved return/global summaries refuse the whole optional authority. The first opcode scope is the anchor's NEW/SET/HAS/GET/LEN and its existing record/call operations; other map operations require separate evidence.

The next review must contain a concrete signature/call-site list and allocation ledger for this threading, including the copied private execution image. I have not established a complete implementation merely by choosing a sidecar. Existing six-profile budget receipts must remain byte-for-byte unchanged; new map-only work/allocation exhaustion must refuse atomically and recover independently.

## I separate declarations from runtime authority

The ordinary VM already has typed, refcounted maps and canonical native execution already handles STRING→INT/STRING maps, but selected record-array consumers currently exclude maps. I require independent private validation of map tags, key/value types, storage/count/capacity and aliases before selected execution. Native roots and teardown must retain the same map identity across record forwarding and calls. Map strings, replaced entries, deleted entries and allocation failures need exact lifetime coverage; an empty map cannot prove a missing pair by observation.

Qualification must cover original binary/text transport and stripped names, malformed pair/binding data and old-reader failure, public/private execution and copied-plan mutation isolation, all-writes conflicts, empty/nonempty maps, shared field aliases, detached historical aliases, call/global joins, allocation prefixes, teardown and strict sanitized native O0/O2. I update paired publisher expectations only after consumer gates pass. I retain old ARRAY bytes, all old required budget boundaries and the original fixture throughout. Full #1001 remains open.

## I locate the concrete threading barriers

I audited the following existing call chain after the initial checkpoint. These are proposed private interfaces, not new public authority tokens or implemented signatures:

| Existing boundary | Required map-only context | Old-route constraint |
|---|---|---|
| `analyze` → `nvm_record_array_structure_prepare` | detect/validate map framing before the raw opcode/signature whitelist; pass exact context to a private preparation variant | old entry remains a NULL-context wrapper |
| structure preparation → `nvm_prepare_ownership_declarations` → `oaa_prepare` | validate complete HASHMAP fields against the separate pair/binding table while preserving the existing ARRAY reader | no unbound HASHMAP admitted by the old reader |
| structure preparation → `verify_function_body` and mixed type verifier | allow only exact map opcode transfers and declared signature tags for the map route | original module range/contracts/body validation still mandatory |
| `prepare_records` → `ra_prepare_records` | require every HASHMAP field's exact pair; retain array/record checks | no coarse tag-only field acceptance |
| instruction-site scan → `walk` | reserve HM_NEW in the existing 64-origin namespace; transfer NEW/SET/HAS/GET/LEN with exact pair facts | no extra namespace or tag-as-origin substitute |
| `record_value_matches`, `global_refine`, `seed`, `join`, `global_return`, `final_check` | preserve map origin bits in record/call/global flow and check exact pair on every possible origin | no unknown or missing origin becomes typed by a field annotation |
| `ra_publish` / origin getters | copy map pair/origin/field facts into separately owned map result storage | old origins object size and reports unchanged |
| `nvm_prepare_record_array_execution` → snapshot → `analyze` → recipes | revalidate the copied module with its copied map context; include exact HM recipe coverage | no trusted borrowed pointer to mutable caller metadata |
| image copy/bound → `vm_ra_layouts`, `vm_ra_value_valid`, dispatch | obtain pair facts from the map image and independently validate physical maps and exact operations | old execution/image/private-runtime sizeof and entry behavior unchanged |

`Value` already contains a uint64 origin set and uint16 tag set. That permits a candidate map origin representation without changing its size, provided I extend *every* kind-sensitive filter only under the explicit map context. In particular, `global_refine` currently retains ARRAY/RECORD origins only; silently leaving it unchanged would lose map identity. `seed`, `join` and global summaries copy/merge Value, but that mechanical propagation is not sufficient until all producers, filters and terminal checks are audited. HM_GET returns INT-or-VOID under the initial pair, so it introduces no map-valued result origin.

I propose map-specific owning wrappers around the unchanged structure, origin report, execution plan/image and VM preparation objects. Each wrapper owns its sidecar explicitly and delegates destruction to both halves. A map-aware private function receives the sidecar as a separate argument; no old object acquires a pointer. Exact wrapper types, signature list, constructor order and cleanup graph remain the next review deliverable: existing snapshot analysis currently constructs and destroys its own structure, so the new route cannot simply bolt a context onto a completed old plan that would already have refused.

My incremental allocation ledger is: framed pair/binding table copies; per-origin pair and historical key/value facts (bounded by the same 64 origins); per-field pair-index table (bounded by existing field cells); map-only structure/result wrappers; separately copied execution/image pair data; and runtime per-field pair lookup/validation scratch. I must charge each allocation before construction, including overlap while analysis publishes and while the execution image is copied. I must charge context validation and full instruction rescans to the new route. Old NULL routes allocate none of these and retain existing sizeof-based charges. I have not chosen byte ceilings or claimed that these additional structures fit existing map-free budgets.

DELETE remains outside the first opcode profile: preserving historical writes despite removal is a future proof obligation, not permission to publish authority for the unchanged STRING→STRING fixture that Nano currently refuses.
