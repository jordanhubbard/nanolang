# Private union VM preparation

I propose preparation only after image75f811. I have not implemented or measured this owner. The existing declaration, proof, recipe and image receipts remain evidence for their own phases. I do not infer execution from a prepared owner. This contract belongs to #1001.

## Interface and ownership

```c
typedef struct VmUnionFieldPrivate VmUnionFieldPrivate;
NvmArrayEligibilityResult vm_union_field_private_prepare(
    const NvmModule *module, VmUnionFieldPrivate **out);
void vm_union_field_private_destroy(VmUnionFieldPrivate *owner);
bool vm_union_field_private_preparation(
    const VmUnionFieldPrivate *owner, VmUnionFieldPreparation *out);
```

The module-taking constructor is the only construction entry. Input module storage remains exclusively stable for the call. Writable output is disjoint from all input/owner storage as a C precondition; I do not claim arbitrary pointer validation. Failure preserves the prior output and input module, and frees every partially owned allocation. Success publishes once. Preparation statistics are copied values, not mutable VM, plan, image or execution handles. Destroy(NULL) is valid; destruction requires the creating thread and, in a later execution phase, an idle owner.

The owner embeds one ordinary `VmState`, one owned image pointer, copied counts, function rows[256], two layout rows, two variant rows, two field rows, and explicit record/union nominal-to-layout mappings. It owns one dense instruction/site copy and byte/work accounting/status fields. Exact field order and `sizeof(owner)` must be measured at source qualification, not inferred from an existing map owner. It does not contain a second proof, carrier, origin allocation or borrowed source-plan pointer. Fixed tables count once in sizeof(owner). Preparation alone needs no result, epoch, busy flag or invocation view; those belong to a later reviewed runtime extension.

I preserve record nominal ordinal0 → layout1 and union nominal ordinal0 → layout0. Variant identity follows the copied declaration, including reversed payload/empty ordering. No name allowlist is involved. The first shape remains one record field containing one nominal union with one INT payload alternative and one empty alternative. Union payload writes, recursive payloads, arrays and maps are outside this preparation profile. Conservative Some/None and historical field alias refusals remain unchanged.

## Actual setup boundary

`vm_init` in `src/nanovm/vm.c` allocates an initial stack, reads environment settings, builds constants/decode/dispatch and invokes `vm_recompute_verified` → `nvm_verify_linked`. I do not use it or any public load/execute entry for this private setup. In particular, public rejection of the six record-containing originals is not an error to suppress or reinterpret.

My proposed owning-TU sequence is:

1. Run `nvm_prepare_union_field_execution(module,&source)` with its unchanged default budget. Measure bounded runtime requirements from its copied getters.
2. Independently reconstruct the image using the positive runtime reservation below. After image success, free source. The original module may then be destroyed independently.
3. Allocate zeroed owner, install creating thread and all coprocessor fd/pid sentinels(-1), and transfer image ownership. Install only image-bound immutable module/root borrows. Leave verified false, references/links/callbacks/trace absent, and private_record_dag false.
4. Copy function, declaration and dense recipe/site rows through immutable getters. Call `vm_decode_module`, set validity only on success, select `vm_dispatch_profile_none()`, then call `vm_dispatch_build_module`. Decode/build validate instruction framing/control-flow, not selected union execution authority.
5. Check decoded/dispatch correspondence against the copied recipes: function counts/ranges, all opcode/operand widths and values, stack effects, branch targets/successors, sensitive site states and exact nominal mappings. Scan all256 opcode values against the qualified96-opcode constructor set. NOT_ESTABLISHED never becomes an executable row. Require no fused instruction, empty call-handle caches and exact same-module CALL/TAIL target correspondence. Do not call module resolution or invent resolved module pointers.
6. Allocate the proven whole-frame stack capacity and exact global rows, all VOID-initialized. Call `vm_heap_init` and then the existing static `vm_module_constants_build`. No guest opcode runs and no container is constructed.
7. Publish only after every correspondence/storage check succeeds.

The copied module borrow is a new explicitly internal image-owner accessor, scoped by a private header to preparation. It returns `const NvmModule *`, valid only while that image lives. It does not expose the plan/carrier, accept caller-supplied validated state, or authorize public execution. No preparation API accepts an image supplied by callers.

## Private image reservation boundary

I propose an internal owning-TU entry declared only in that private header:

```c
NvmArrayEligibilityResult nvm_union_field_image_copy_reserved(
    const NvmUnionFieldExecutionPlan *source,
    uint64_t extra_bytes, uint64_t extra_work,
    NvmUnionFieldImage **out);
const NvmModule *nvm_union_field_image_module_internal(
    const NvmUnionFieldImage *image);
```

The existing `nvm_union_field_image_copy(source,out)` remains a literal0/0 wrapper. Extras are positive conservative reservations, never refunds. Checked addition includes extras in the existing source-retained/image-header/comparison charges, then cap-minus-initial checks occur before the first image allocation. Overflow and over-cap requests return LIMIT with output unchanged. Fresh destination construction still independently proves the snapshot and compares all source facts; extras cannot skip proof or correspondence.

Existing image0/0 allocation order,55 prefix positions and default byte/work results must remain unchanged. Existing constructor0/0 keeps54 positions, atomic proof/site38 positions, and all old recipe/proof budgets. No source/caller budget token becomes authority. Image statistics include extras when nonzero and must be labeled reserved allowances rather than actual allocated payload.

## Cardinalities and retained allocations

All counts are checked before traversal or multiplication. I require 1<=F<=256, I<=65536, G<=256, O<=64, S<=NVM_MAX_STRINGS, frame_limit==VM_MAX_FRAMES, valid entry/initializer indices, and exact two-layout/two-variant/one-record/one-union shape. Per function: arity<=local_count<=256, maximum_stack<=256, result_count<=1, upvalues0, code length L_f<=16MiB, contiguous instruction ranges summing to I. The source constructor already validates module-wide bounds and counted names; preparation still checks copied count identities before using them. The image reproof remains independent.

All size products/additions must fit uint64_t, SIZE_MAX and unchanged128MiB bytes/33,554,432 logical-work caps. Allocation rounding must not overflow. `K=VM_MAX_FRAMES*512+1`; current1024 frames gives524289 NanoValues. This conservative frame bound covers local256 plus operand256 for every frame and one result edge. It is not an entry-only stack estimate. No stack realloc or growth is allowed in preparation.

Let Q=sizeof(new owner), E=sizeof(NvmUnionFieldExecutionInstruction), D=sizeof(VmDecodedInstruction), J=sizeof(VmDispatchInstruction), V=sizeof(NanoValue), P=sizeof(void*). Let C_f be the geometric decoder capacities16,32,... through the first capacity >=I_f; C_f is empty when I_f=0. Let H be intern capacities256,512,... sufficient for S distinct strings, using the actual pre-insertion threshold `4*existing_count >= 3*capacity`. Initial256 is always included, even S=0. The runtime byte reserve is:

```
B_runtime = Q + I*E + K*V + G*V
          + F*(sizeof(VmDecodedFunction)+sizeof(VmDispatchFunction))
          + sum_f[(L_f+1)*(1+2*sizeof(uint32_t))
                  + I_f*J + D*sum(C_f)]
          + S*sizeof(VmString*) + sizeof(VmString*)*sum(H)
          + sum_s[sizeof(VmString)+length_s+1]
          + 65536
```

The final65536 is a separately named conservative automatic scratch allowance, not retained heap. I measure actual per-frame and whole-call-stack extents before qualification. Descriptor rows and fixed functions are already in Q. The decoded capacity sum conservatively covers every realloc old/new overlap; retained bytes contain only the final capacity. Intern capacity sum covers successful old/new bucket overlap. Failed growth allocation preserves the prior table and may be retried; failed allocator internals are not claimed as allocated payload or machine-operation bounds.

Zero cases are explicit: globals absent at G=0; constant pointer array absent at S=0; decode/dispatch function tables nonempty because F>=1. Boundary, instruction-index and dispatch-offset arrays each have L_f+1 entries even for empty code. Decoder and dispatch instruction allocations are absent for I_f=0. Each string reserves its terminal byte, including empty strings. The owned image retains its own optional present-zero-arity parameter row allocation of1; runtime borrows that snapshot and does not allocate another parameter table. All snapshot/debug/metadata storage is counted by image retention, not duplicated in runtime accounting.

I reserve each constant independently even when interning deduplicates equal strings. The constants table owns one retained edge per index. Real retained payload and peak must be measured independently, including deduplicated objects, rather than equated to this conservative reserve. No cycle buffer is allocated by preparation: only strings exist. Later ordinary containers and collector storage require separate runtime accounting/qualification.

## Logical work schedule

Work is a bounded logical preparation schedule, not runtime fuel or allocator timing. Checked arithmetic and precharge precede each loop. Runtime requirements are accumulated locally; the full allowance is passed once as image extra_work. Later operations debit that allowance for consistency, never add it again to the image reservation.

I propose this explicit envelope:

```
W_measure = 64 + 32*F + 8*S
W_copy    = 64 + F*(16+sizeof(function_row))
            + I*(4+E) + 8*(4+max_declaration_row_size)
W_decode  = sum_f[4*L_f + 64*I_f + (L_f+1)]
W_check   = 256 + 64*F
            + sum_f[(L_f+1) + I_f*(256+E+D+J)]
W_strings = sum_s[(length_s+1)*(s+8)+16]
            + sum_successful_growth_capacity_c[c+S]
W_runtime = B_runtime + W_measure + W_copy + W_decode
            + W_check + W_strings
```

Here function_row is the copied execution-function type. Eight declaration copies are two nominal mappings, two layouts, two variants and two fields; max_declaration_row_size bounds their typed output extents. The fixed64 copy work covers header/count queries and fixed policy guards. Each getter costs its documented four probes; copied byte extents and fixed loop guards are explicit conservative extras. W_measure is checked/precharged before its bounded function/string scans; it covers count/header checks and length/function getters used to derive sizes. It is included once in W_runtime even though performed before image allocation.

B_runtime in work conservatively prepays zero initialization, decoder realloc copies and copied allocation payload extents, plus scratch allowance. No uninitialized stack growth is involved: this constructor uses calloc. W_decode reuses the existing preparation decoder envelope; it is not a claim about every C instruction. W_check prepays boundary arrays, all fixed operand/successor slots and96-of256 predicate comparison, copied sites and decoded/dispatch rows before traversal. Any future additional loop must acquire its own term, not spend an unspecified remainder.

For strings, index s has at most s existing intern-chain nodes. Each adversarial same-hash/same-length comparison examines at most length_s bytes, plus bounded node guards; hash/copy/terminal initialization are covered by the additional8 factors and allocation extent. Growth walks c buckets and at most S strings on each successful resize; the capacity list is checked before looping. Failed calloc does not traverse or rehash. No open-addressing s+8 assumption is used: the actual intern table is chained. Counted NUL strings remain ordinary VM string storage; this preparation does not make a native byte-domain claim.

No fit claim is made before actual six-original measurements. If this conservative schedule refuses a valid input, I retain that result for review; I do not raise caps or silently remove terms.

## Whole-call peak and cleanup

I report three phases, with independent allocator oracles:

1. Module→source plan: the unchanged constructor's complete proof/snapshot peak and work occur before runtime estimation. They are not hidden inside image statistics.
2. Source retained plus image header plus independent destination construction: existing full proof reserve covers destination analysis, while exact source retained bytes cover source overlap. Positive B_runtime additionally reserves the future runtime/scratch envelope. This intentionally over-reserves memory that is not all live in phase2; it never discounts source overlap.
3. Source freed; retained image plus owner, decoded/dispatch tables, stack/globals and constants: actual peak is image retention plus actual runtime allocations and transient realloc/growth overlaps. No source-plan retention remains.

Full measured peak is max(phase1 peak, phase2 peak, phase3 peak), not the sum of three historical maxima and not only phase3. Logical total work is the bounded sum of source-constructor work, existing image cost/reproof/comparison work and W_runtime. Source construction and image construction each retain their existing independent caps; I do not claim one shared total33,554,432 budget across both proofs. The combined image+runtime reservation must fit its unchanged cap. Measured counters must distinguish these quantities.

On any failure, destroy only initialized owners. `vm_decode_module` and `vm_dispatch_build_module` clean their partial tables; validity flags publish only on success. `vm_module_constants_build` cleans its partially populated table; equal constants hold separate edges and must release once per index. Owner destruction calls ordinary `vm_destroy` before freeing copied rows/image. That releases frames, globals and stack, decoded/dispatch tables, constants and heap, while image strings/code remain alive. Then free copied rows, image and owner. A freshly zeroed owner gets thread/fd sentinels before any failure path can invoke vm_destroy. Never free a borrowed image module separately. No guest result or collector container exists in this checkpoint.

## Later execution boundary, not this patch

A later typed call-local view must validate before IP/profile mutation and before dispatch-table indexing; it must preserve genuine bufferedPURPLE and unbufferedBLACK/PURPLE containers, require stringBLACK/unbuffered, and leave ordinary GC metadata intact. All repeated union/record/string retains must preflight cumulatively. No fused path may skip a sensitive recipe. Scope must be invalidated/restored on every exit, with initializer and entry roots reset independently and root-tail/ASSERT resume identity kept invocation-local. No persistent VmState context pointer or heap-policy flag is proposed.

HALT is not decided by adding an outcome enum. Ordinary baseline and separate supervisor tests must establish its distinction from return/error before execution implementation. Likewise public borrowed-state execution, native nominal/canonical representation, transport enrollment and publishers remain separate. The existing Some-only/history precision boundaries remain explicit follow-on obligations.

## First qualification gate after source review

I require strict default/private macro compilation and exact owner/table sizeof measurements, then all six retained originals through preparation only, with module/source destruction and copied preparation queries. Test every allocation prefix in transient/persistent denial modes, prior owner sentinel, retry, independent retained/peak bytes, constant dedup and intern growth, S=0/empty strings/optional parameter rows, malformed copied/decode/call-cache correspondence, source limits and positive-extra overflow/cap refusals. Keep image0/0 55 positions, constructor54/38 positions, proof338/393 and six legacy budget rows unchanged. These tests must not call a new executor or claim public/native acceptance.
