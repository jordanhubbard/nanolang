# I specify the proposed MAP_FIELDS declaration phase

I refine [the map representation alternatives](NANOISA_RECORD_MAP_AUTHORITY_DESIGN.md) for independent review. This is design only: I assign no production constant and change no admission. My first executable pair remains STRING→INT from unchanged `map_record_fields.nano`; other structurally valid pairs grant no executable support.

## I propose exact bytes

I propose ownership v3 extension kind **5**, revision **1**, retaining the existing simultaneous extension count **1..4**, strictly increasing unique kinds and four-byte framing alignment. Five simultaneous kinds refuse. Unknown kinds/revisions, truncated frames, nonzero framing padding and trailing bytes refuse. Old readers reject kind 5; no reader may interpret its bytes as ARRAY_FIELDS.

All multibyte integers use the existing little-endian v2 cursor convention. The payload is exactly:

1. `u32 scalar_count` in 1..5.
2. `scalar_count` rows: `u8 tag, u8 reserved0, u16 reserved1, u32 referent`. Both reserved values are zero and referent is NO_INDEX. Tags are strictly increasing and drawn only from INT, U8, FLOAT, BOOL, STRING. No VOID/ENUM/ARRAY/STRUCT/HASHMAP row exists in this revision.
3. `u32 pair_count` in 1..25.
4. `pair_count` rows: `u32 key_type, u32 value_type`. Both index this payload's scalar table. Rows are strictly lexicographically increasing `(key_type,value_type)`; identical pairs cannot repeat.
5. `u32 binding_count` in 1..65536.
6. `binding_count` rows: `u32 layout, u16 field, u16 reserved, u32 pair`. Reserved is zero. Rows strictly increase by `(layout,field)`; a field cannot bind twice even to the same pair. Pair indexes this payload's pair table.

The payload length is exactly `12 + 8*scalar_count + 8*pair_count + 12*binding_count`, at most **786684 bytes**. I validate counts, overflow and complete byte extents before allocating any table. There is no variable text or inline key/value datum. I require every scalar row to participate in a pair and every pair to participate in a binding; unreachable extra declarations refuse. This makes the minimal anchor's STRING→INT pair unambiguous without requiring a universal table of unused types.

Bindings address the complete ownership layout table, bounded by the existing 256 layouts and 65536 total fields. Every bound field must be TAG_HASHMAP/NO_INDEX in a COMPLETE ordinary STRUCT. Every HASHMAP field in a complete layout must have exactly one binding. HASHMAP in an UNKNOWN, RESOURCE or UNION layout refuses in this declaration profile; I do not infer an endpoint from runtime contents. Other fields retain their existing exact validators, including the independently framed ARRAY_FIELDS extension. A standalone HASHMAP parameter/result/global descriptor is outside this first profile; the unchanged anchor transports maps through bound record fields. Scalar pairs other than STRING→INT may be described, but the selected executable query must return UNRESOLVED for them before any execution route is chosen.

## I define owned declaration interfaces

I propose a new header `ownership_map_declarations.h` with an opaque `NvmMapDeclarationPlan`, not a pointer added to old declaration structs. The following signatures and results are the proposed review boundary:

```c
typedef enum {
    NVM_MAP_ABSENT, NVM_MAP_DESCRIBED, NVM_MAP_INVALID,
    NVM_MAP_LIMIT, NVM_MAP_MEMORY
} NvmMapDeclarationStatus;
typedef struct {
    NvmMapDeclarationStatus status;
    const char *message;
} NvmMapDeclarationResult;
typedef struct { uint32_t scalars, pairs, bindings; } NvmMapDeclarationCounts;
typedef struct { uint32_t key_type, value_type; } NvmMapDeclarationPair;
typedef struct {
    uint32_t layout;
    uint16_t field;
    uint32_t pair;
} NvmMapDeclarationBinding;

NvmMapDeclarationResult nvm_prepare_map_declarations(
    const NvmModule *, NvmMapDeclarationPlan **out);
void nvm_map_declarations_free(NvmMapDeclarationPlan *);
bool nvm_map_declarations_counts(const NvmMapDeclarationPlan *,
    NvmMapDeclarationCounts *out);
bool nvm_map_declarations_scalar(const NvmMapDeclarationPlan *,
    uint32_t index, uint8_t *tag);
bool nvm_map_declarations_pair(const NvmMapDeclarationPlan *,
    uint32_t index, NvmMapDeclarationPair *out);
bool nvm_map_declarations_binding(const NvmMapDeclarationPlan *,
    uint32_t index, NvmMapDeclarationBinding *out);
bool nvm_map_declarations_field(const NvmMapDeclarationPlan *,
    uint32_t layout, uint16_t field, NvmMapDeclarationPair *out);
```

DESCRIBED owns copied numeric facts plus the unchanged base declaration object, independent of source bytes. The opaque map plan is a wrapper owning those two halves; old plans do not own map objects. The field getter returns the pair's table indexes, so callers must obtain exact tags from the same plan rather than comparing indexes across plans. Failed getters preserve output. Public getters are immutable, allocation-free and independently bounded: counts/scalar/pair/binding perform constant work; field lookup uses two retained layout-boundary reads, at most 17 binary-search comparisons for at most 65536 bindings, and one pair lookup. I publish a fixed conservative field-query charge of 20 work units and a fixed charge of 1 for each other getter. A cumulative proof caller must precharge that amount in its own budget before invoking the getter, including failed queries; a public standalone caller receives only the per-call bound. No plan contains a mutable hidden query budget. All unsuccessful constructors, including ABSENT, preserve `*out`; only DESCRIBED publishes a plan. Free accepts NULL and destroys sidecar tables, the owned base plan, then wrapper. No getter grants executable admission. An initial pair-support check belongs to future executable admission and must reject every bound non-STRING→INT pair.

Preparation validates complete layouts, ownership flags/signatures/paths and all present extension frames and semantics. It cannot produce a successful map plan from an independently validated fragment of malformed metadata. Its private base-declaration preparation takes an explicit map declaration view to permit exactly bound HASHMAP fields; ordinary callers pass NULL and retain their old checks. It must not call the unchanged rejecting base constructor first and then reinterpret its failure as ABSENT.

## I separate classification from authority

I propose an allocation-free private route classifier with result `ABSENT`, `PRESENT`, `INVALID` or `LIMIT`, never a boolean that treats every error as absence. It scans bounded ownership framing and layout extents, not map runtime data. PRESENT means a kind-5 frame or HASHMAP layout field was observed; it is only a requirement to call complete map preparation. A HASHMAP field without MAP_FIELDS reaches that preparation and refuses missing bindings. A kind-5 frame without fields refuses unused/empty bindings. Truncated/duplicate/unknown extension framing returns INVALID even if kind 5 has not yet been reached. An over-limit count/scan returns LIMIT. A malformed map payload may remain PRESENT at this framing phase, but can never return ABSENT.

The first classifier case is a metadata-free legacy module: when both ownership and layout payloads are absent, it reports ABSENT without inspecting legacy executable record/map contents or inventing authority. A module with either metadata payload present enters the metadata scan; it cannot use that legacy case to hide a malformed or incomplete pair of tables. In that metadata-bearing case, ABSENT requires a successful bounded scan showing no kind-5 frame and no HASHMAP field; ordinary validation remains mandatory before execution. Any existing ownership payload with malformed framing remains a refusal, not a legacy retry. This phase adds no map support to public execution: until the later consumer integration is reviewed, PRESENT must be rejected at public selected execution boundaries. The old ARRAY detector's early return cannot take precedence over this classification.

## I bound and charge the new phase

I retain all old base object sizes and their exact 16MiB declaration and 1048576-step allowances. A map-selected transaction reserves that existing base declaration allowance and an additional **2MiB map-only byte allowance plus 4194304 map-only work units**. These are proposed conservative declaration-policy ceilings, not runtime map-size limits. Old absent-map transactions reserve no additional bytes/work and retain old allocation order. A public classifier's added framing work is separately bounded and never smuggled into the old six-profile charged counters.

The map-only ledger charges, before allocation: sizeof wrapper; exact scalar/pair/binding table capacities using checked `count*sizeof(element)`; a 257-entry u32 layout-to-binding start table; the actual bounded temporary scratch object described below; and every simultaneously live temporary table. The wrapper holds the old base pointer and the new tables only on this route. If implementation needs another temporary or copy, it must reserve its full capacity and overlap here before construction. No allocation is hidden in old Analysis, declaration, structure, execution or image sizes. Copying this plan into later execution/image wrappers has a separate ledger and is not authorized by this declaration ceiling.

For map work I charge every byte consumed in the two permitted payload scans, every scalar/pair/binding row validation, each referenced-index lookup, each layout and field visited for binding completeness, every ordering/comparison step and the fixed worst-case precharge for each getter invoked by a cumulative proof caller. I use the sorted tables for linear completeness validation; field queries use the retained per-layout start/end interval and bounded binary search, with the caller precharging 20 units rather than mutating an immutable plan. I permit no uncharged repeated full-table search. Before each subtraction/addition I check remaining budget; count products and total payload length use checked wide arithmetic before conversion to size_t. Budget failure releases staged allocations and preserves output; retry with independent fresh budget must succeed when inputs fit.

I require no 64KiB automatic scratch for this algorithm. Sorted completeness uses a single monotonically advancing binding cursor and two used-row sets of exactly 5 and 25 bytes. The concrete new temporary object contains those sets, two `NvmV2Cursor` values (framing and row scan), one scalar row, one pair row, one binding row, previous ordering keys, and scalar/pair/binding/layout/field counters. I charge its actual `sizeof` before use and retain that reservation for the transaction; I do not allocate full-table duplicate scratch or a per-field automatic array. The implementation checkpoint must show the final object's fields/sizeof and any additional new call-chain temporary reservation. Existing base preparation keeps its old independent scratch/accounting and is not charged twice.

The maximum payload is below half of the proposed work allowance, but this does not prove the complete transaction fits: actual max-layout/max-binding and allocation-prefix tests must measure the full implementation, including existing base preparation. Any insufficiency returns LIMIT rather than silently enlarging a ceiling. The existing six map-free allocation/work receipts must remain exactly unchanged.

## I enumerate declaration integration sites

`ownership_contracts.h/.c` needs the reviewed kind/revision, explicit optional map view output from extension framing, and map-aware complete layout checks; the old four-view aggregate remains unchanged. `ownership_layouts_private.h` and its implementation need a private explicit map mode that accepts only HASHMAP/NO_INDEX leaves while preserving all old DAG and layout bounds. `ownership_declaration_projection.inc` needs a private preparation variant receiving validated map bindings, with the old public entry delegating with NULL. The new map declaration file owns pair-table validation, copied wrapper and getters.

Before public admission changes, the complete declaration validators used by `nvm_v2_module.c`, `nvm_v2_convert.c`, assembler and source query paths must agree on kind 5's mandatory-understanding behavior. Structural support cannot be used as executable support: verifier, VM and native selection must still refuse PRESENT until the separately reviewed map proof/runtime routes exist. Publishers stay unchanged until those routes and all consumers qualify. I keep the first instruction set exactly NEW/SET/HAS/GET/LEN; DELETE and the STRING→STRING `map_remove` source boundary remain outside.

Required declaration tests cover exact binary/text roundtrip, old-reader rejection, all reserved fields/padding, count and length overflow/truncation, strict ordering/duplicate rows, wrong and missing field bindings, unused scalar/pair rows, unsupported-but-wellformed pairs, malformed classifier prefixes and later frames, max-four versus five extensions, unchanged prior output on every failure, every allocation prefix, byte/work boundary refusal and recovery, and old map-free bytes/budgets. I do not authorize implementation or claim a runtime result with this document.


## I bound the declaration implementation phase

After review, the proposed implementation scope is limited to:

- `ownership_contracts.h/.c`: kind 5/revision 1 recognition with max-four count unchanged; complete framing and explicit optional map view; old four-view storage unchanged; a private map-aware complete-layout check.
- `ownership_layouts_private.h` and `nvm_v2_layouts.c`: an explicit map-enabled entry accepting only HASHMAP/NO_INDEX leaves, preserving old entries and exact layout/DAG bounds.
- `ordinary_array_authority.inc` and `ownership_declaration_projection.inc`: private `oaa_prepare`/base-preparation variants receiving validated map declarations, with the existing entry passing NULL; no old struct-size change.
- New `ownership_map_declarations.h` and private implementation: complete payload validation, owned wrapper/tables, immutable getters, the bounded route classifier and explicit status/cleanup behavior.
- Focused declaration tests and their existing narrow test-target wiring: malformed bytes, exact pairs/bindings, classifier precedence, atomic output, allocation/work boundaries, and map-free byte/size/budget controls.

This phase must not enable publishers, change Analysis/Value/origin transfer, extend execution recipes, or admit map execution in the verifier/VM/native backend. If recognizing kind 5 in common declaration validation could allow any existing public path to execute selected map metadata, implementation must first add a fail-closed selected-map refusal at that path and test it; it may not rely on an incidental later error. The new focused harnesses are proposed as `tests/nanoisa/test_map_declarations.c` and `tests/test_map_declarations.py`, following `test_ownership_declaration_projection` allocation instrumentation. Existing neighbors are `test_ownership_contracts`, `test_ownership_layouts_private`, `test_ordinary_array_authority`, `test_ownership_declaration_projection` and the six record-array budget probes. `Makefile.gnu` changes are limited to new include dependencies and a narrow declaration test target; no frontend prerequisites. Production proof/runtime wrappers and public selection support remain separately reviewed future work.
