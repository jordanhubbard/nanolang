# I propose an independently validated copied map image

This is a contract-only follow-on to constructor/query component `6940bd144`. I propose no runtime dispatch, public selection, native translation or publisher changes. Existing public selected-map refusal remains required. The next implementation requires review of this ownership/cost interface first.

## I rebuild authority from the owned snapshot

A new opaque `NvmRecordMapImage` owns one newly constructed `NvmRecordMapExecutionPlan`. It never borrows the source plan's carrier, frozen context, base report, declaration table or recipe rows. The copy constructor uses the source plan's immutable carrier snapshot as input to the same private map-plan constructor, which snapshots it again, independently runs complete structure/origin proof, and reconstructs every recipe. It does not clone an old proof bit or use source origin rows as acceptance authority.

Thus each plan keeps its own carrier origins NULL and its own frozen context as sole owner of that plan's base report. The source plan may be destroyed immediately after a successful image copy. Image destruction frees its plan once and then its wrapper. A failed image construction preserves the caller's existing image pointer and never modifies the source. No third image snapshot is created inside the wrapper.

The new image may expose a const module borrow only in its private header, for the eventual private consumer; this borrow is valid only until image destruction and carries no public executable authority. No mutable module/facts pointer, generic caller trust token or API for host-supplied physical maps is added. Full public load/verify/native refusal remains in this phase.

## I count actual retained owners before a second proof

A source plan's historical peak reservation is not its current resident size. Adding two128MiB preparation ceilings would either double-count destroyed analysis or silently change a limit. Instead I propose a new private retained-storage cost query, with explicit bounded traversal charging and no fields added to any old type.

The retained payload sum includes each allocation exactly once:

| Owner | Retained allocations |
| --- | --- |
| Map plan | Its wrapper, unchanged carrier, carrier's base instruction rows, augmented map instruction rows |
| Carrier snapshot | Existing copied function rows, code, ownership/layout bytes, metadata/debug rows, string lengths and pointers, each counted string plus terminator, optional parameter-row pointer table and each present row (including existing one-byte zero-arity rows) |
| Frozen facts | Frozen wrapper, eight-byte map-origin rows, base-origin wrapper and fixed required-mask array inside it |
| Base report | Fixed record report with its64 inline origins, flattened field-value allocation |
| Base declarations | Ownership declaration wrapper and union rows; ordinary-authority wrapper with inline flags; layout rows and each layout's field rows; exact array type/binding allocations |
| Map declarations | Sidecar wrapper with base pointer NULL, scalar/pair/binding allocations, fixed257-entry starts allocation |

The source plan has no mutable Analysis, Function states, historical Value arrays or temporary decoder boundary buffers left alive. Those are not included in retained bytes. The frozen map sidecar's `base==NULL` and carrier's `origins==NULL` must be checked before counting; violation refuses rather than counting or freeing a shared owner twice. All products/sums use checked uint64 arithmetic and validate existing count ceilings/pointer presence. No hidden padding is reused for accounting.

Opaque declaration objects require private cost helpers implemented in their owning TU, not guessed sizeof values in the execution module. Proposed `nvm_ownership_declarations_retained_cost` and `nvm_map_declarations_retained_cost` return copied byte totals and charged logical work using a supplied private budget; they allocate nothing and preserve outputs on failure. The map helper requires the detached sidecar form and never recursively counts a base. The ordinary helper counts its owned nested fields using bounded layout traversal. These helpers do not change old constructors or their budget receipts.

The top-level source cost walk is also explicit: it is supplied a private preparation budget, charges each examined row before traversal and then publishes a byte total only on success. No mutable budget pointer is stored in an image/plan or exposed through ordinary immutable getters. A fixed precharge for maximum256 layout rows plus maximum256 function rows and the already bounded string count can be calculated from immutable top-level counts before walking. The exact per-row charge schedule must be written and tested with implementation; no uncharged scan or machine-instruction claim is allowed.

## I keep the existing preparation ceilings

The image operation uses the existing128MiB/33554432-work constructor ceiling for simultaneous source-retained bytes plus image preparation. It computes source retained bytes before allocating the image. It then pre-reserves source bytes plus image-wrapper sizeof through the reviewed positive-only `rme_prepare_reserved`, and pre-reserves source-cost traversal work plus image-wrapper zeroing work. The called constructor still reserves the full64MiB/16777216 proof allowance and charges its new snapshot, carrier and rows normally. Therefore old source storage, new snapshot, mutable proof and new frozen publication overlap all fit one checked envelope; no source credit is refunded during construction.

No promise is made that every accepted plan fits image creation. A large source plan can be valid yet fail this additional simultaneous-storage bound with LIMIT, unchanged prior image and a valid still-queryable source. This is a conservative new private preparation refusal, not a guest runtime limit. Source construction's already completed historical work is not charged again; only the new source cost scan and actual new proof/copy work are charged. The old array image and runtime budgets remain unchanged.

## I propose explicit private APIs

```c
typedef struct NvmRecordMapImage NvmRecordMapImage;
typedef struct {
    uint64_t source_retained_bytes;
    uint64_t preparation_bytes_reserved;
    uint64_t preparation_work_reserved;
} NvmRecordMapImageCost;
NvmArrayEligibilityResult nvm_record_map_image_copy(
    const NvmRecordMapExecutionPlan *, NvmRecordMapImage **);
void nvm_record_map_image_free(NvmRecordMapImage *);
bool nvm_record_map_image_cost(const NvmRecordMapImage *, NvmRecordMapImageCost *);
/* Private borrow only, no public execution admission. */
const NvmModule *nvm_record_map_image_borrow(const NvmRecordMapImage *);
```

The image's typed counts/instruction/map-field/map-origin queries forward through its independently owned plan and copy results. No source-plan pointer is retained. Tests compare all stable logical fields, recipes, declarations and snapshot bytes between source and image; measured budget counters are intentionally different because they account for different overlap. Independent proof of the copied snapshot is the acceptance authority, not equality with a possibly stale source summary.

## I require independent validation and failure evidence

Before any runtime proposal: exercise all four retained actual original anchor modules with test-only map declarations; destroy originals and source plans before querying images; compare exact code/layout/ownership/string bytes and full opcode/operand/edge recipes; check GET INT|VOID and map endpoint types; preserve public verifier/native/serialization refusal. Add whitebox corruption controls to the source snapshot (malformed kind5, wrong pair, historical write conflict, unknown receiver, invalid branch/operand) and require the fresh proof to refuse. Test corrupted source owner/count bookkeeping separately as private defensive checks, not purported public attacks.

Instrument every image-copy allocation prefix with prior-image preservation, source query survival, zero leaks and independent retry. Measure retained totals against tracked live payload before copy, and peak image operation against the combined reservation. Qualify exact byte/work boundary and one-unit-over refusal using the same explicit positive pre-reservation principle. Include maximum bounded tables/strings and arithmetic overflow before allocation. Preserve all six old proof budgets and old array execution/image neighbors unchanged.

Physical hashmap validation and execution remain subsequent review: tags/counts/capacity alone do not establish uniqueness/reachability under the actual hash probe sequence. Later import/observation validation must establish lookup topology, because hm_resize assumes no duplicate equal keys. Key retain headroom, resize failure/alias semantics, absent GET, closed invocation lifetime and native key byte closure are not discharged by a copied image.

## I fix the retained-size helpers and logical charge schedule

The constructor audit confirms exact allocation capacities, not just wire counts. `layouts_decode_profile` allocates `count` layout rows and exactly `field_count` field rows, skipping both zero cases. Its forward-graph colors/stack are freed before return. Ordinary ARRAY type/binding allocations and union rows skip zero counts and have no retained spare capacity. Map scalar/pair/binding counts are nonzero and exact; starts always allocates257 uint32 entries, even when fewer layouts exist. Frozen map rows and report fields skip zero counts. Every wrapper/report uses exactly one sizeof allocation.

`rae_copy` and `rae_zero` skip zero byte counts. Snapshot strings allocate each counted length+1, including empty strings. The optional parameter-pointer table allocates exactly function_count pointers when present; each present zero-arity row uniquely allocates one byte. An absent row allocates nothing. The carrier's function/stack/descriptor fixed arrays and the report's64 origins/required masks are inline and must not be charged a second time. Both instruction arrays allocate exact total instruction count, with no growth capacity. Every mutable VmDecodedFunction instruction capacity (minimum16, doubling), byte-index array, decoder boundary buffer, Value/state queue and proof structure is destroyed before plan publication; none is retained in this ledger. Proof reservation continues to cover their earlier peak.

I propose a new private header shared only by the owning TUs:

```c
typedef struct {
    uint64_t used, limit; /* logical work only; no owned storage */
    bool limited;
} NvmMapCopyWork;
typedef enum { NVM_MAP_COST_OK, NVM_MAP_COST_INVALID,
               NVM_MAP_COST_LIMIT } NvmMapCostStatus;
NvmMapCostStatus nvm_ownership_declarations_retained_cost(
    const NvmOwnershipDeclarationPlan *, NvmMapCopyWork *, uint64_t *bytes);
NvmMapCostStatus nvm_map_declarations_retained_cost(
    const NvmMapDeclarationPlan *, NvmMapCopyWork *, uint64_t *bytes);
```

These two helpers are defined in `ownership_contracts.c` through its existing owning includes, where the real private types and constructor rules are visible. The second requires detached `base==NULL`. Static helpers `rme_frozen_retained_cost` and `rme_plan_retained_cost` live in the analysis/execution TU; no old public type gains a cost field. `bytes` outputs are published once at success. Failure may consume work already spent and set the caller's limited flag; it never modifies the plan or bytes output. The work object and output must be nonoverlapping objects. These are private call-local accounting arguments, not stored contexts or trusted admission tokens.

The schedule is defined in logical probes, not CPU operations:

- Every helper entry first charges16 before reading any owning object. It checks pointer presence, fixed count ceilings, detached ownership where applicable, and top-level scalar relationships.
- Every allocation term charges4 before inspecting its pointer/count and performing checked multiplication plus accumulation. It is charged even for an absent/zero allocation; that term contributes zero bytes. Count/pointer consistency follows the audited constructor rule. A byte-sum overflow yields LIMIT, malformed count/pointer relationships INVALID.
- Each visited layout, function or string row charges8 before dereferencing that row. Its allocation term then charges4 separately. No field contents, instruction rows, scalar/pair/binding contents or string bytes are scanned merely to measure storage; their complete semantics are independently re-proved by the destination constructor. Layout field_count is read once and its aggregate total remains≤65536. Function local/arity counts retain≤256; snapshot strings retain≤4096 and lengths use checked +1.

Successful declaration cost is `40 + 12*L`: entry16, six allocation terms (ownership wrapper, ordinary wrapper, layout rows, union rows, ARRAY types, ARRAY bindings), and L layout-row visits plus their field-array terms. Detached map declaration cost is36: entry16 and five terms (sidecar, scalars, pairs, bindings, fixed starts). Frozen cost is `112 + 12*L`: entry16, five terms (frozen wrapper, map-origin rows, base wrapper, fixed report, report fields), then both declaration helpers. The report field allocation uses its actual field_value_count with existing FIELD_CELLS≤65536, not the declaration-field count; repeated record origins can own distinct field summary slices.

Plan cost is `180 + 12*(L + F + S)`: entry16, thirteen terms (map wrapper, carrier, two instruction arrays, snapshot function/code/layout/ownership/metadata/debug/string-length/string-pointer/parameter-pointer allocations), F function-row visits plus each optional parameter-row term, S string-row visits plus each terminated-string term, then frozen cost. The fixed header/sections and all carrier inline arrays are already included in carrier sizeof. No decoded capacity, source module capacity field, allocator overhead or old peak reservation is substituted for retained payload. Counts are checked before using them as loop limits, and every row is charged before access. The implementation must retain these exact schedules or return for review if an additional traversal becomes necessary.

The image constructor initializes this call-local work allowance to the unchanged33554432 preparation limit, calls the retained-cost helper before allocating an image, and passes the actual consumed work plus image-wrapper initialization to positive pre-reservation. The new constructor reserves/proves/copies as already specified. There is no uncharged preliminary loop to calculate the cost of the cost walk. Work exhaustion inside any helper leaves byte output unchanged. The resulting image cost records this actual schedule plus constructor reservations; source construction's prior historical work is not repeated in the accounting.

The independent payload oracle uses existing `record_array_alloc.c` tracking: immediately after successful plan publication and all temporaries are gone, `ra_bytes` must equal reported retained bytes exactly, and the helper call must not increase ra_calls/ra_live. Fixtures must cover absent/present parameter tables, a present zero-arity one-byte row, empty strings, absent optional rows, zero-field unrelated layouts, actual map rows and full array/declaration combinations. The tracker records requested payload bytes, excluding allocator bookkeeping; a separate allocator-minimum or alignment guess would be incorrect. Image tests compare the source's still-live payload plus destination actual peak to the combined reservation, then free the source and compare destination retained payload independently. All allocation-prefix tests preserve prior output, source queries, exact cleanup and retry.
