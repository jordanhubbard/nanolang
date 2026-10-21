# Bounded union-field consumer interface

Status: checkpoint1 constructor/query implementation qualified after reviewed design134f8c5ed; receipt `evidence/union-field-recipes-20261022/README.md`. Image/runtime/native/public/publisher sections remain design proposals requiring separate review. The qualified proof and declaration budgets remain unchanged. The only structural profile is one nominal union with an INT-payload variant and an empty variant, contained in one nominal record. Names and match metadata are not authority. None exhaustive-match payload access and field-read aliases contaminated by later Some/None history remain conservative refusals, not completed general union support.

## Actual source boundaries

| Site | Current behavior | Required new-only boundary |
|---|---|---|
| `record_union_structure.inc`, `union_field_analysis.inc` | Typed wrapper, historical proof, AGG_TAG and terminal HALT | Remain descriptive; executable consumers require their own constructors |
| `managed_union_field_proof_private.h` | Copied origins/globals/results/stacks and declarations | No per-instruction receiver subset is exported today |
| `managed_record_array_execution.inc:rae_instructions_using` | Copies operands/CFG and uses array recipe enum | Ordinary fallthrough rule would give HALT a successor; do not reuse unchanged for union HALT |
| `managed_record_map_execution.inc:rme_prepare_boundary` | Snapshot, single analysis, copied rows | Pattern for ownership only; union must not pass through map/array facts or a NULL declaration projection |
| `record_map_snapshot.inc:rme_image_copy_reserved` | Exact retained cost plus independent destination proof | Separate union image with actual retained overlap and no source-proof reuse |
| `record_map_borrowed_run.inc:vm_rmb_call_root` | ASSERT resumes; every other non-NONE trap, including HALT, refuses | Union HALT supervisor disposition must be deliberate |
| `vm.c` AGG_PACK/GET/SET/TAG | Ordinary ordinal storage, transfer/retain/release, unchecked generic retain saturation | Typed union preflight and checked transfer before pops/mutation |
| `vm.c` dispatch top | Context checks before table lookup, preflight before ip/profile/handler mutation | New mutually exclusive call-local typed context, no fallback on invalid non-NULL view |
| `nvm2c.c` simulation/emission | Variant payload boxing and kind/tag checks exist | Canonical union shape, nominal mapping and root lifetime still unqualified |
| `record_map_native_target.inc` | Native map descriptor ordinal==layout restriction | Cannot be reused for record ordinal0/layout1 union profile |

## First implementation checkpoint: constructor and copied queries only

Proposed opaque `NvmUnionFieldExecutionPlan`, module-taking `nvm_prepare_union_field_execution(const NvmModule *, NvmUnionFieldExecutionPlan **)`, destructor and copied count/function/parameter/instruction/declaration/origin/global/result/byte getters. No public execution function takes this plan. No getter returns an old array carrier, mutable snapshot, proof owner or Analysis pointer.

The plan owns an independently copied module snapshot, one successful union proof, copied function/parameter rows and decoded recipe rows. It may internally own an unchanged array execution carrier solely for common snapshot storage, with `origins==NULL`; no old origin/declaration getter or publication routine may receive that carrier. This choice needs a concrete owning-TU allocation ledger before source. A distinct owner with duplicated bounded snapshot storage is an alternative if enforcing those invariants becomes harder than sharing helpers. No new field enters old Analysis, Value, VmState, VmHeap, old plan, or old view.

A missing interface must be supplied by the same successful analysis: exact final site obligations for sensitive aggregate operations. Proposed private output rows contain `(function, pc)`, copied receiver/value `NvmRecordValueOrigins`, exact nominal ordinal/layout where applicable, and operation discriminator. Rows contain no mutable Values or runtime pointers. They publish only after final completeness checking and full historical constraints. Constructor then joins them to decoded `(function,pc)` rows, checks uniqueness and exact opcode, and rejects missing obligations at a reachable sensitive site. Missing propagated state never certifies unreachable code. All bytecode, including dead instructions, still receives structural operand/edge validation; execution of a row without its required proved site obligation traps.

Old proof callers supply NULL for this new-only output and retain literal old allocations/work. New plan preparation captures stacks and site facts during one analysis, not a second analysis with divergent summaries. A successful descriptive proof alone does not allocate or publish recipes. Any late snapshot/correspondence/recipe failure destroys both outputs and preserves the caller's previous plan.

Each new output capacity is charged before allocation using actual `sizeof` (including Value-independent copied rows), with exact zero-count behavior. A full 64MiB proof reserve remains live while snapshot/rows survive, under unchanged execution128MiB/33,554,432-work caps. No fit claim is made before measurement. Snapshot string terminators, optional parameter tables, zero-arity allocated rows, function/code capacities and decoder overlap must be listed explicitly. Immutable getters are bounded copies with explicit caller precharge; no hidden mutable budget. An allocation-free retained-cost helper must match an independent live-payload oracle before image budgets rely on it.

## Recipes and all-256 audit

`AGG_VARIANT` is an aggregate kind, not a new opcode. Exact `OP_AGG_PACK` operands are kind:u8, nominal:u32, variant:u16, count:u16. `AGG_GET`/`AGG_SET` index is u16; `AGG_TAG` has no operand. CALL/TAIL callee is u32, branches signed i32 relative to instruction start. Copy operand types and encoded bits from InstructionInfo, check exact decoded length, branch boundaries and callee signatures; never truncate the nominal operand to u16.

| Operation | Copied obligation and dynamic transfer |
|---|---|
| PACK RECORD | Exact record ordinal→layout, variant0, one UNION field; consumes input edge into new record |
| PACK VARIANT | Exact union ordinal→layout and variant; payload alternative count1/INT, empty count0; no union-payload mutation |
| GET record field | Receiver record nominal and field index; exact historical UNION origins; checked retain of selected child before receiver release |
| GET union payload | All proved receiver origins payload-bearing, exact union nominal/variant/count and INT field; consumes receiver and produces scalar |
| SET record field | Exact record and UNION replacement; full historical obligations and aliases; transfer new input before releasing old field |
| TAG | Exact UNION shell and valid variant; consumes union, returns integer variant tag |
| HALT | Terminal, no CFG successor/callee-return/result summary; distinct recipe |

A new recipe discriminator must be outside every valid old recipe with static assertions; do not make AGG_TAG or HALT a permissive scalar fallback. A full256 table comparison must establish structure predicate ↔ recipe generator ↔ each selected executor/native target policy. Current union structure is the old array predicate plus TAG and HALT, with additional PACK-kind restrictions; that syntactic superset does not prove arrays/resources executable under this first union profile. Final proof and target policies remain additional checks. Audit every explicit supported-count guard, not just increment a constant. Old full256 arrays/maps sets and their charges must remain exact. Constructor corruption tests independently mutate opcode, operand width/bits, pops/pushes, successor/terminal, callee, nominal/layout/variant and receiver/result masks.

The existing generic recipe decoder would assign HALT fallthrough because its exclusion list is JMP/RET/TAIL. The union path must exclude HALT before final edge validation. This needs a new-only decode policy or a separately reviewed NULL-preserving helper argument; post-publication repair is forbidden.

## Copied image and lifetime descriptors

Proposed `nvm_union_field_image_copy(const NvmUnionFieldExecutionPlan *, NvmUnionFieldImage **)`, destructor and copied getters. The source plan is only input data: reconstruct its owned module snapshot and run the new module-taking constructor independently. Compare source/destination metadata, signatures, recipes and exact nominal descriptors; source corruption must not be treated as trusted authority. Source may be destroyed after successful copy. Image owns only the newly proved plan, never a borrowed source proof.

Descriptors keep record ordinal0→layout1 and union ordinal0→layout0 separately. Union descriptor includes both exact variant identities/counts and payload INT constraint; record descriptor binds its UNION field to the union layout. Reversed variant order must work. Origins retain full historical fields/payloads and shared64 identity space. Neither descriptor ordinal nor runtime pointer is a substitute for a static origin.

Image budgeting reserves actual retained source allocations plus complete destination snapshot/proof/recipe transient overlap, not two nominal cap reservations added blindly. Cost helpers scan checked counts with prepaid logical visits, checked arithmetic and no allocations; internal immutable allocation metadata is a C precondition, not arbitrary pointer-extent validation. Every failure leaves source and previous destination usable.

## Scoped VM proposal, after constructor/image qualification

A private module-taking borrowed-state test entry precedes any public selector. It constructs and owns one fresh invocation preparation through import validation, initializer, entry, every ASSERT resume, final cleanup and destruction. Module must equal vm.module and vm.root_module; module/caches/heap are exclusively stable during invocation. No caller-supplied plan, persistent VmState token, observation result or relaxed map-context cast authorizes execution.

The ordinary heap convention remains `VmStruct.def_idx=record ordinal`, `VmUnion.def_idx=union ordinal`. Descriptor mapping selects layout for checks only. `vm_union_new` allocates a header and `calloc(field_count, NanoValue)`; empty variants permit implementation-defined zero-count allocation and need exact independent allocator accounting. `release_union` already releases each field. First-profile INT payloads introduce no recursive union children, but containing record aliases and ordinary suspect-buffer metadata remain significant. Private DAG policy must not be installed on public VmState.

Fresh imports validate all readable global-capacity slots, not just global_count. Copied historical global obligations already exist; unknown summaries refuse. Imported heap occurrences need the same conservative singleton structural assignment policy as reviewed map import, with shared physical aliases agreeing on a singleton and full history. Positive refs/header/nominal/variant/count/payload are checked; GC-only buffer members get membership checks rather than invented origins. Ordinary buffered BLACK-at-zero and PURPLE states must follow actual collector invariants. Unrelated cycles must not be normalized or force-collected to obtain success. Failed preparation is a const query except its own allocations.

Prepare the conservative whole-frame stack bound (`VM_MAX_FRAMES*512+1`) only after independently confirming local/operand/frame limits. Actual reserve mirrors ordinary doubling/fallback/no-growth behavior and rechecks original stack identity/capacity before allocation. Byte reservation includes live old/new overlap; work covers possible old-byte copy plus bounded rounding, not uninitialized new capacity. Invalid nonzero-capacity NULL storage refuses. Reserve failure preserves guest values and prior stack backing/capacity; success may enlarge capacity before guest execution.

Runtime context validation precedes frame/function/table indexing. Exact decoded recipe correspondence precedes ip advance, profiling or mutation. Invalid non-NULL union context traps; all old NULL wrappers retain behavior. Fused dispatch (e.g. LOAD_LOCAL+AGG_GET) must either be disabled for the new context or independently perform both instructions' complete checks; bypassing preflight is forbidden. Call-local view retains active_root/root_base, updated separately for initializer and entry and on root tail replacement.

Zero-arity INT entry and zero-arity VOID/INT initializer remain the proposed selected host contract. Initializer residue is drained before entry. Caller must consume the successful entry result before another invocation. Helpers may return UNION/STRUCT; this does not imply heap-valued host entry support. CALL/TAIL stage all operands, count duplicate retained owners cumulatively, check capacity/frame/nominal tags before mutation, preserve simultaneous argument replacement and root identity across >16 staging. GET must retain before consuming the last record receiver; SET transfers its rooted input before releasing the prior field. Saturated retains trap before mutation. Existing generic handlers' unchecked retains/pushes are not fixed merely by proving bytecode.

ASSERT consumes its condition exactly once; true resumes the same view and root, false preserves the first error and selected-route unwind drains frames/stack while keeping globals and external aliases. This is selected cleanup, not generic VM's leftover-frame behavior. Import rejection leaves caller state unchanged. Runtime failure does not roll back aliased globals. View invalidation precedes preparation destruction on every exit.

HALT is a separate policy decision requiring explicit review: generic public supervisor currently returns VM_OK without ordinary return checks; map supervisor rejects it. Proposed union behavior is terminal whole invocation, selected owned-stack/frame cleanup, preserved globals, no fabricated entry result, and no entry execution after initializer HALT. An internal root outcome distinguishes RETURN/HALT/ERROR; a caller must not mistake HALT for a successful INT return. Native translation must agree on whole-program termination. This proposal is not yet baseline-qualified and must not silently become ordinary RET or invalid-opcode behavior.

## Native intersection and staged enrollment

The module-taking native route owns fresh plan then independently reproved image, and exposes no caller-supplied trusted-image API. Eligibility remains separate from declaration authority and VM proof. The existing map native ordinal==layout guard cannot admit this profile: record0/layout1 is mandatory. A union-specific target must map both nominal namespaces exactly and check every descriptor, not only encountered constructors.

`nvm2c.c` currently chooses canonical records only for record arrays/maps. Variant simulation boxes INT through OPTIONAL/VARIANT_SCALAR; AGG_TAG checks runtime AGG_VARIANT kind; record fields may use REC snapshots. These are groundwork, not evidence that canonical Outer→Box alias identity or sole-root GC lifetime works. New target/shape transfer must distinguish record and union nominal spaces in PACK, nested field checks and SET. It must not pass a union ordinal to a record-layout helper. Preserve kind/tag and empty representation, helper CALL/TAIL return roots, and existing first-error checks. Canonical helpers/accounting must be enabled even for unused declarations, with old output bytes unchanged.

Native HALT currently has unwind and non-unwind paths with different result handling; the target must establish the reviewed whole-invocation policy, not inherit fabricated zero INT return accidentally. String-producing restrictions, STRUCT-vs-AGG transport and other existing native limitations require explicit target policy rather than assertion of universal native parity. First actual tests remain six unchanged originals plus reversed ordinals/variants, empty tag, aliases/replacement, helper/tail sole-root forced GC and O0/O2 allocator failure controls.

Implementation checkpoints are separately reviewed: (1) same-analysis site export + recipe constructor/getters, (2) cost oracle + independently copied image, (3) ordinary import/preparation and checked transfer tests, (4) private borrowed invocation/dispatch including HALT policy, (5) independent native target/shape/lifetime, (6) atomic public verification/selection/transport, and only then paired publisher design. At every earlier checkpoint public execution remains refused. No new wire grammar or publisher widening is proposed here.

## Checkpoint 1 concrete refinement for source review

I choose the existing map-style snapshot carrier reuse, restricted to the owning analysis TU. No header exposes its pointer. Its `origins` stays NULL for its entire lifetime; its destructor releases snapshot and base instruction storage only. Union proof/site ownership is separate. Only raw snapshot/count/function/parameter/byte utilities may be shared; old origin/declaration accessors, `ra_prepare_records`, `ra_publish` and old whole-plan constructors are excluded.

Proposed exact new types (names remain internal until reviewed):

```c
enum NvmUnionSiteState { UFS_NOT_REQUIRED=0, UFS_NOT_ESTABLISHED=1, UFS_PROVED=2 };
enum NvmUnionSiteOperation {
 UFS_NONE=0, UFS_RECORD_NEW, UFS_RECORD_PACK, UFS_VARIANT_PACK,
 UFS_RECORD_GET, UFS_PAYLOAD_GET, UFS_RECORD_SET, UFS_TAG, UFS_HALT
};
typedef struct {
 uint32_t function, pc;
 NvmRecordValueOrigins receiver, input, result;
 uint32_t nominal, layout;
 uint16_t member;  /* field index for access, field count for construction */
 uint8_t variant; /* 0/1 only for variant construction/payload access;
                      UINT8_MAX when not applicable */
 uint8_t operation, state, reserved[3];
} NvmUnionFieldSite; /* LP64 prediction72; actual sizeof is charged */
typedef struct {
 NvmUnionFieldSite *rows;
 uint32_t count, reserved;
} NvmUnionFieldSiteExport; /* stack-local ownership envelope, LP64 16 */
struct NvmUnionFieldExecutionPlan {
 NvmRecordArrayExecutionPlan *carrier; /* origins always NULL */
 NvmUnionFieldProof *proof;
 NvmUnionFieldSite *sites;
 uint32_t site_count, reserved;
}; /* LP64 prediction32; no other owner allocation */
```

`NvmRecordValueOrigins` remains16 bytes. Padding/reserved bytes are zeroed, never used for flags. These LP64 figures are layout derivations, not measured compiler results; implementation must assert/measure them and charge portable sizeof. A public copied instruction query returns base recipe plus site by value; it does not allocate a second expanded instruction array. Separate immutable getters retain four-probe bounds per copied getter, including misses. The base carrier's existing size is charged as `sizeof(NvmRecordArrayExecutionPlan)` without modifying that type; no unmeasured numeric constant replaces it.

There is one dense site row per decoded instruction, in function then byte-offset order, with `I<=65536`, `1<=F<=256`, `G<=256`, `O<=64`, code<=16MiB and existing string/section caps. Fixed complete declaration counts remain two layouts, one record and one union with two variants. Dense rows avoid a hidden site-index map and make recipe correspondence O(I). Zero instructions allocate no site array; actual eligible root requirements remain separate. Unused receiver/input/result summaries are zero, not TAG_VOID facts. Nominal/layout/member sentinels are UINT32_MAX/UINT32_MAX/UINT16_MAX when not applicable.

Common operations have NOT_REQUIRED/NONE rows with only function/pc set. Sensitive decoded instructions with no final `seen` state have NOT_ESTABLISHED and an opcode-compatible operation (AGG_GET whose receiver kind is not established uses NONE until refusal at execution); summaries remain zero and sentinels unchanged. This is absence of an established dynamic obligation, never proof of CFG unreachability. No consumer may execute that row. Sensitive seen instructions must be PROVED after the existing complete final check; a seen deferred/unknown site prevents all output publication. HALT PROVED has no receiver/input/result and no nominal, variant or member; it establishes only a terminal instruction, not a function result.

For PACK, input is the sole payload/field summary (zero for empty); result is the exact allocation-site origin. RECORD_NEW follows actual proof-supported initialization and may not invent a UNION field from VOID. GET has exact receiver and selected result; PAYLOAD_GET requires one exact payload variant across all receiver origins. SET has exact receiver, input replacement and result receiver, preserving full histories. TAG has exact receiver, scalar INT result and no variant filter (both variant alternatives allowed). Scalar summaries have no origin bits, unknown is always zero in PROVED rows, and copied masks must match frozen kind/nominal rows. Variant spelling/order is never consulted.

### Same-analysis output transaction

The new call is private/static in the owning TU, not a trusted caller-plan API:

```c
static NvmArrayEligibilityResult analyze_union_sites(
 const NvmModule *module, NvmUnionFieldProof **proof_out,
 NvmUnionFieldSiteExport *sites_out);
```

Both writable outputs must be nonoverlapping with each other and all input/owner storage; explicit output range overlap refuses. They are caller-visible outputs only and never accepted as execution inputs. Both are unchanged on every failure. The normal descriptive proof entry retains its existing signature and passes NULL for optional sites internally. No new persistent Analysis/context field or old prepared_stacks use is needed: completed proof already copies exact stacks, and the new constructor copies those with four probes/function after success.

Internally rename the analysis implementation once, retain its old exact-signature wrapper forwarding NULL, and branch only when union proof plus sites output are requested. After current final checking, allocate/copy sites into a local unpublished envelope while declaration/context/state data remain alive. Then freeze proof into a local pointer. On site/freeze failure, free local rows/proof and normal analysis owners. Only after both succeed assign the two outputs, with no fallible work between assignments. Existing proof-only union path calls its original freeze branch without site counting, allocation or charge. All map/array callers remain NULL; old six budgets and current proof-only totals/prefix counts must remain identical.

The constructor itself receives these two outputs in private locals. It joins every row with decoded recipes and runs full snapshot equality before assigning its single plan output. Late recipe/equality failure destroys both, including the transferred declaration owner. A site-array allocation failure cannot leave a published proof behind. The new private transaction requires direct prior-two-owner sentinel tests, not only constructor output checks.

### Exact owning allocation and reservation ledger

Let C be the unchanged carrier sizeof, R the unchanged base-instruction sizeof, Q the new owner sizeof32 on the measured LP64 ABI, T the site sizeof72 prediction, D the declaration retained cost112 on the qualified ABI. Let S be exact snapshot allocation payload:

`F*sizeof(NvmFunctionEntry) + code_size + layout_size + ownership_size + metadata_count*sizeof(NvmMetadataEntry) + debug_count*sizeof(NvmDebugEntry) + strings*(sizeof(uint32_t)+sizeof(char*)) + sum(length_i+1) + optional(F*sizeof(uint8_t*)) + sum(present_parameter_row_i ? max(arity_i,1) : 0)`.

Sections/header/scalar capacities are inside carrier snapshot, not separately allocated. Logical capacities copied into metadata do not expand owned allocation extents. Every zero-count snapshot allocation is skipped except a present zero-arity parameter row, which owns one zero byte exactly as `rae_snapshot` currently implements. Proof retained payload is `P = sizeof(proof_owner)+D+O*sizeof(origin_row)+(G+F)*sizeof(NvmRecordValueOrigins)+F*sizeof(uint16_t)` (qualified LP64 `184+40O+16G+18F`). Source code/module is borrowed only during construction and not counted as new ownership.

Final retained plan payload is exactly `Q+C+S+I*R+I*T+P`. It includes both existing base instruction rows and dense sites; no expanded duplicate recipe table. Temporary declaration/analysis/decode/state/report overlap remains inside the unchanged64MiB proof reserve. Site rows are allocated with `analysis_alloc` and therefore consume that reserve while analysis lives, then remain owned. The execution budget conservatively also charges their retained `I*T` after successful analysis, with no refund: this covers their overlap with subsequent decoded recipes. P likewise remains covered by the full proof reserve, never separately counted twice as a simultaneous full proof. The later image copy uses exact retained formula, not the reserve, for its source overlap.

Initial execution reservation: bytes=`64MiB+65536+Q+C`; work=`16777216+65536+Q+C`. Existing snapshot allocations charge copied/initialized bytes as both bytes and work. The fixed65536 is the established constructor scratch reserve, not a new automatic array. After proof: charge `I*T` bytes for retained sites (copy already performed under proof work), 4F work before stack getters, 256 work before opcode audit, 32 work before at-most-eight declaration/mapping queries, then unchanged decoder count/copy/edge schedules and snapshot-equality charges. Final counts/rows correspondence adds 16I work before its scan. All checked additions refuse before dependent allocations; unchanged execution128MiB/33,554,432-work limits may reject large inputs below independent count caps.

Optional analysis export schedule is separate: validate summed instruction counts with F prepaid visits; precharge `512*I` logical work before semantic row scans, then `analysis_alloc(I,sizeof(site))` charges initialization bytes/work. The512 envelope covers row/header/state checks, up to three64-origin summary validations, at most eight four-probe declaration reads and constant-width copying; no dynamic string or unbounded child scan is allowed in this envelope. A helper that charges its own existing semantic validation adds those charges explicitly rather than reusing a hidden credit. Histories are read, not recomputed or mutated. Constructor fit and new exact budget thresholds require measured tests; no cap increase is proposed.

### Exact syntactic recipe partition

Read-only enumeration of current predicate names gives94 array opcodes and96 union-structure opcodes (adding AGG_TAG and HALT); this is a source audit, not compiled all256 evidence. The mandatory compiled comparison must test all256 byte values independently and retain exact differences. No HM opcode is included.

Sensitive recipes are exactly STRUCT_NEW, STRUCT_LITERAL, STRUCT_GET, STRUCT_SET, AGG_PACK, AGG_GET, AGG_SET, AGG_TAG, HALT. AGG_PACK chooses record versus variant only from exact kind operand and declaration/proof facts; AGG_GET chooses record versus payload only from PROVED receiver facts. These nine use a dedicated invalid-old-recipe sentinel and the site discriminator. HALT has zero successors irrespective of InstructionInfo's generic default. STRUCT/AGG union payload SET never receives PROVED operation.

The other87 syntactically accepted operations use exactly `rae_old_recipe`, including its existing tag/retain/staging flags: primitive constants, arithmetic/comparison/logical/casts, string operations, local/global load/store, DUP/SWAP/POP/ASSERT, branches, CALL/TAIL/RET and array primitives. Their NOT_REQUIRED site row does not waive runtime tagged checks, checked retained-owner headroom or final proof restrictions. In particular array-producing operations do not acquire union-profile origin authority merely because they have a syntactic recipe. Later target/executor qualification can conservatively exclude unsupported common operation domains explicitly, without silently modifying declaration or proof claims.

The new-only decoder should parameterize only its terminal policy where necessary; the existing NULL/default decoder excludes only JMP/RET/TAIL from fallthrough, while union policy also excludes HALT. Existing old recipe wrappers continue their original whitelist. Exact operand/pops/pushes/CFG/global bounds still run for every instruction, including NOT_ESTABLISHED sites. Constructor tests cover all256 entries, both variant orders, absent/proved/malformed site rows, source destruction, allocator prefixes and exact unchanged old proof/plan bytes and work.


Source-review refinements for checkpoint1: constructor correspondence now needs a prepaid fixed64-row frozen-origin cache plus record/per-variant masks. Charge four for counts,5O for origin getter/row visits, O for completed-mask validation and80I for correspondence (16 constant checks plus at most64 origin visits). PACK result must be the exact singleton matching function/pc/kind/nominal/layout/variant; GET masks must match historical fields or exact payload-variant subsets. This replaces the earlier16I-only comparison schedule. The fixed cache uses actual `sizeof(UfeOrigins)` within the existing65536 scratch reserve and must be measured before qualification; no persistent owner field changes.

Before analysis, validate each copied function name index and charge length+1 before rejecting embedded NUL, exactly matching the existing map constructor. This prevents strcmp initializer recognition in proof from disagreeing with counted8-byte recipe initializer recognition. The old descriptive proof path is unchanged. Constructor intentionally remains narrower for STRUCT_GET on UNION (which proof can describe): only AGG_GET union payload is proposed for this recipe profile; retain a named refusal until separately qualified. No source/runtime parity is claimed from the constructor alone.

Checkpoint1 measured plan32/site72/carrier13384/fixed cache2592 bytes. The exact constructor and two-output allocation/limit/old-route evidence is retained in the recipe receipt; these measurements do not qualify any image or executor.
