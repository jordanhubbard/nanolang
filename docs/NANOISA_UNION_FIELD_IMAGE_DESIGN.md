# Independently re-proved union image

Status: design-only checkpoint after constructor568ac06c1. No image/runtime/native/public/publisher code or build is authorized. Constructor-only plan ownership and the original proof conservative precision boundaries remain unchanged. This proposal chooses strict source-fact correspondence rather than silently repairing a corrupted source plan.

## Actual reconstruction boundary

The current map image implementation in `record_map_snapshot.inc:rme_image_copy_reserved` measures retained source payload, adds image overhead, and invokes a fresh module-taking constructor on `source->carrier->snapshot`. Its input facts are not execution authority. It does not compare all old source recipes to the rederived plan. The union image should add that explicit comparison because the new dense site state, variant masks and HALT terminal recipes are part of its copied contract.

No recipe or site serialization format is introduced. Source code, counted strings, signatures, layouts, ownership metadata, debug/metadata rows and existing nominal counts already reside in the owned snapshot. A new independent snapshot is constructed from those bytes; declaration validation, complete historical analysis, final-site export, opcode decoding and correspondence run again. Source recipes/proof are used only for bounded retention accounting and post-reproof equality. They never seed destination analysis or supply a validated bit.

Proposed private API:

```c
typedef struct NvmUnionFieldImage NvmUnionFieldImage;
typedef struct {
 uint64_t source_retained_bytes,preparation_bytes_reserved,preparation_work_reserved;
} NvmUnionFieldImageCost;
NvmArrayEligibilityResult nvm_union_field_image_copy(
 const NvmUnionFieldExecutionPlan *,NvmUnionFieldImage **);
void nvm_union_field_image_free(NvmUnionFieldImage *);
bool nvm_union_field_image_cost(const NvmUnionFieldImage *,NvmUnionFieldImageCost *);
/* The same copied instruction/function/parameter/declaration/origin/global/
 * result/byte accessors as the execution plan; no mutable or carrier getter. */
```

Owner is one plan pointer and the three cost counters, measured-layout prediction32 bytes on the qualified ABI. It owns only a new complete destination plan. No public API accepts an image as execution authority. This checkpoint does not need a borrowed module or plan getter; later consumers can review a narrowly scoped immutable borrow if required. Numeric copied values survive image destruction; byte-copy calls copy into caller storage.

Source is immutable managed constructor-owned storage throughout the call. Writable output is disjoint from source and all owned storage as a C precondition; no arbitrary allocation-extent validator is claimed. Failure frees all partially built destination objects and preserves source, prior destination pointer and existing output queries. Success publishes only after reproof and semantic comparison; the caller may then destroy the source/module independently.

## Owning-TU retained-cost interfaces

Use the already private checked arithmetic/work carrier in `record_map_cost_private.h` without changing its layout or existing functions. Its map-prefixed name conveys no map authority. Proposed new internal `union_field_cost_private.h` declares only the cross-TU declaration cost helper; proof/plan helpers remain static inside `managed_array_shapes.c`:

```c
NvmMapCostStatus nvm_union_declarations_retained_cost(
 const NvmUnionFieldDeclarations *,NvmMapCopyWork *,uint64_t *);
static NvmMapCostStatus ufe_proof_retained_cost(
 const NvmUnionFieldProof *,NvmMapCopyWork *,uint64_t *);
static NvmMapCostStatus ufe_plan_retained_cost(
 const NvmUnionFieldExecutionPlan *,NvmMapCopyWork *,uint64_t *);
```

All helpers allocate nothing, charge before scans, preserve byte output on failure, and consume the supplied work even on failure. Work and output cannot overlap each other or owner storage. Explicit range checks can reject the two writable envelopes overlapping; general managed input disjointness remains a precondition. Counts are checked before pointer traversal; no helper treats metadata capacity as allocated capacity. Redundant count/signature/range checks catch underreporting but cannot establish forged pointer allocation extent.

Declaration owns exactly `sizeof(owner)+2*sizeof(NvmV2Layout)+2*sizeof(NvmV2LayoutField)` =112 bytes on the qualified ABI. It has no separate variant or flags allocation. Check count2, each field_count1, non-NULL arrays, exact ordinal/layout mapping, complete flags and two inline variant count identities before reporting. The loop is fixed2, not a general layout traversal. Work schedule:32 header/inline guards +four allocation terms at4 each +two layout visits at8 each =64 logical units.

Proof owns `sizeof(proof)+D+O*sizeof(origin)+(G+F)*sizeof(value)+F*sizeof(uint16_t)`, qualified `184+40O+16G+18F`. Check O<=64,G<=256,1<=F<=256, exact pointer/count correspondence and non-NULL declaration owner. Work:16 header guards +five allocation terms at4 +declaration64 =100. No mutable Value storage or temporary context remains. Full fact semantics are checked by reproof/comparison, not claimed by cost counting.

Plan retains exactly `Q+C+S+I*R+I*T+P`, from the constructor contract: Q32, C13384, T72, R actual unchanged base recipe sizeof, P proof cost, S actual snapshot capacities. It owns no origin cache (that2592-byte cache was automatic scratch) or second expanded recipe table. Check carrier.originsNULL, site_count==instruction_count<=65536, copied counts versus snapshot functions/strings/sections/metadata/debug, proof functions/globals versus carrier counts, and reserved fields zero. During each charged function visit check copied signature arity/local_count, contiguous instruction_start/count range and total; arity/local_count<=256 before calculating row allocations. Optional parameter table owns F pointers and each present row owns max(arity,1) bytes. Each string owns length+1 including an empty string. Sections are embedded. All checked multiplication/addition precedes allocation terms.

Plan work schedule:32 header guards +13 allocation terms at4 (owner/carrier/base rows/sites and nine snapshot arrays) +20F (16 function/range checks plus one parameter term) +12S_count (8 length checks plus one string term) +proof100 =184+20F+12S_count. Zero/absent arrays still take their bounded term check; they contribute zero bytes. This is a logical accounting schedule, not allocator wall time. Independent live-payload oracles must match every helper, including zero globals, empty string, present zero-arity row and optional metadata/debug tables, before image code relies on them.

## Exact comparison and overlap

After independent destination construction, compare semantic counts and copied data field by field, never `memcmp` structs with padding. Allocation/budget diagnostic counters differ due image pre-reservation and are deliberately excluded; this exclusion cannot hide semantic count differences. Compare:

- Snapshot bytes and logical metadata/signatures/parameter presence: the destination constructor's existing `rae_same` already checks these against source snapshot while the source is stable. Do not call `rae_same` on the source carrier (it mutates budget counters).
- Every execution count including initializer, entry, frame limit, section/metadata/debug and record counts; every function signature, parameter-table presence, instruction range and maximum_stack; every parameter tag; record ordinal0/layout1 and union ordinal0/layout0 mappings and both variant facts.
- Every base recipe: function/pc/next/successors/callee, opcode/recipe/pops/pushes/obligations, operand count/types/bits.
- Every dense site: state/discriminator, function/pc, nominal/layout/member/variant, receiver/input/result origins/tags/unknown and zero reserved bytes. NOT_ESTABLISHED must remain NOT_ESTABLISHED; it is not proof of runtime unreachability.
- Every historical origin/member, global and function result plus stack fact; all complete declaration rows/fields/variants/nominal mappings. Reversed variant order is compared by actual identity, not spelling.

Precompute the comparison allowance from validated source counts before destination allocation, and precharge it before any semantic row traversal. Count validation and the bounded arity visits used to derive it belong to the cost pass above; corrupt destination counts must equal those validated source counts before any comparison loop. The exact proposed envelope is:

- Fixed header/count pairs: `32+sizeof(NvmRecordArrayExecutionCounts)+sizeof(NvmUnionFieldProofCounts)+sizeof(NvmHeader)`; three getter pairs cost24, with8 fixed guards. Compare only semantic fields, not reservation counters or padding.
- Functions: `F*(16+sizeof(NvmRecordArrayExecutionFunction))`; two getters cost8, eight fixed loop/presence checks, and the struct byte extent bounds scalar field comparisons. Parameters: `9*sum(arity)` for two four-probe getters and one tag comparison, only for present parameter rows. Arity is counted conservatively even when the optional table is absent.
- Instructions/sites: `I*(24+sizeof(NvmRecordArrayExecutionInstruction)+sizeof(NvmUnionFieldSite))`; two getters cost8,16 fixed guards, and byte-extent bounds for explicit field comparisons. Both active and unused operand/successor slots are compared; no padding is read.
- Origins: `O*(8+sizeof(NvmUnionFieldOrigin))`; globals/results: `(G+F)*(8+sizeof(NvmRecordValueOrigins))`; stacks: `10F` for two getters and bounded comparison.
- Eight declaration pairs: two nominal mappings (record ordinal0 and union ordinal0), two layouts (indexes0 and1), two union variants (ordinals0 and1), and two fields (layout0/field0 and layout1/field0). Charge `2*(8+sizeof(uint32_t))+2*(8+sizeof(NvmDeclarationLayout))+2*(8+sizeof(NvmUnionVariantFact))+2*(8+sizeof(NvmV2LayoutField))`. Each8 covers its getter pair; typed byte extents bound explicit comparisons. Variant order is not assumed to identify the payload alternative.

Use checked multiplication and addition throughout. Snapshot byte/string traversal is already charged by destination snapshot construction and `rae_same`; the extra correspondence pass does not rescan those bytes. Source allocation capacities are counted independently of reservation counters. Carrier embedded descriptors, function stacks and fixed arrays are included exactly once in `sizeof(carrier)`, never added as hypothetical heap arrays.

Let R_source be actual retained cost, W_cost measured cost work, H=image sizeof32, W_cmp the prepaid comparison envelope. Destination constructor's initial reservation remains `B0=64MiB+65536+sizeof(plan)+sizeof(carrier)` and `W0=16777216+65536+sizeof(plan)+sizeof(carrier)`. Before allocating image, check `R_source+H <= 128MiB-B0` and `W_cost+H+W_cmp <= 33554432-W0`. Then a new **static owning-TU** `ufe_prepare_reserved(module,out,extra_bytes,extra_work)` adds only these positive reservations before its first allocation; current public-private constructor remains exact0/0 wrapper. This interface is justified by real source/destination overlap, not introduced only for tests. Checked additions form each extra, then cap-minus-base comparisons reject overflow or excess before allocation. The image header zero allocation is charged once in both extras; source retention is charged once in bytes, cost traversal and comparison once in work. Source retention is not zeroed or recopied by this reservation. No refund or negative credit exists. Old plan constructor0/0 must retain exact54-prefix/byte/work behavior.

The source plan and image header stay live while destination snapshot, proof/context/site export and recipe arrays coexist. The unchanged full proof reserve covers that destination analysis overlap; actual R_source, not an old128MiB peak, covers source retention. No blind128+128 addition and no second copy of borrowed source payload. Comparison uses automatic row scratch within existing65536 reservation, with actual maximum frames measured. Prepaid comparison work is never charged twice; runtime getters remain immutable.

The image starts with a NULL plan. The constructor publishes a local destination plan only on complete success; comparison failure frees that local plan through its sole destructor. After comparison, transfer it into the image and clear the local pointer, then publish the image output. Image destruction calls the plan destructor once (which owns carrier, proof and sites separately) and frees only the image header. No ownership or pointer is transferred from the source.

On destination failure preserve its actual INVALID/UNRESOLVED/LIMIT/MEMORY classification. On post-reproof source-fact mismatch return INVALID and destroy the new plan. Do not allow a valid destination to silently sanitize source recipe/site corruption. On success image cost reports R_source and destination final reservations including the positive overlap/comparison reserve. It does not claim actual peak equals reservation.

## Proposed bounded next implementation checkpoint

First implement only retained-cost helpers and their independent no-allocation byte/work oracle, then the internal positive-reservation wrapper and image copy/semantic comparison with copied queries. Keep old constructor/public routes literal. Required source review precedes a bounded build.

Tests: six unchanged originals, source/module destruction, exact copied bytes and semantic rows, reversed variant order, NOT_ESTABLISHED/HALT terminal stability; source recipe/operand/site/nominal/variant/mask corruption rejected after independent reproof; source snapshot wrong payload/history/nominal and malformed metadata refused by actual destination proof; real prior-image/source survival through every transient/persistent allocation denial; exact positive overlap cap/+1/overflow without raising caps; independent retained/peak oracle including optional/zero-count edges. Public verifier/native remain refused for original and reconstructed modules; descriptive conversion is not execution admission. Old constructor842/3234, proof338/393, old image/recipe neighbors and six resource rows remain required compatibility evidence. No frontend/bootstrap, VM/native run, wire schema or publisher change is part of this checkpoint.
