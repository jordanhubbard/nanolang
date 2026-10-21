# I design a closed STRING-to-INT map invocation

Status: design-only above copied-image75bf0ef55; #1001 remains open. I own this audit and propose interfaces for review before runtime changes. The first source remains unchanged `tests/nanoisa/fixtures/map_record_fields.nano`; the four retained ordinary inputs and private annotation provenance remain explicit. Only HM_NEW/SET/GET/HAS/LEN are selected. DELETE, map signatures/globals, native translation and producer publication are separate gates.

## I retain immutable authority for the whole invocation

Current `vm.c` routes public execution through `vm_execute` → `vm_execute_record_arrays` → `vm_call_function_scoped` → `vm_core_execute_scoped`. The array public wrapper discards its plan before dispatch; this is not a suitable lifetime model for new map facts. The private array path instead owns plan/image/VM in `VmRecordArrayPrivate`, enters `vm_ra_call_root`, and invokes the same core dispatcher with an explicit private context. Its context is validated on every entry/resume; no pointer is stored in VmState.

I propose a distinct opaque `VmRecordMapPrivate`, not an enlarged old owner or a cast to it. It owns one independently reproved map image, its VmState, copied runtime layout/recipe tables, root/result state and preparation accounting. The source plan used to construct the image is freed after preparation; the image remains the sole long-lived owner of its copied plan/facts. A private lifetime-bound const plan accessor may expose image queries internally, without transferring ownership or granting executable authority. Runtime tables copy only the facts needed for dispatch; they never retain Analysis, Value summaries or proof budgets.

I propose one internal invocation view whose lifetime is bounded by its caller:

```
typedef struct VmRecordMapInvocation VmRecordMapInvocation;
static bool vm_rm_context(const VmRecordMapInvocation *, const VmState *);
static VmResult vm_rm_preflight(VmRecordMapInvocation *,
                               const VmDispatchInstruction *);
static VmTrap vm_core_execute_scoped(..., VmRecordArrayPrivate *,
                                    VmRecordMapInvocation *);
```

The view borrows exact VM/module identities, immutable image/plan queries, decoded-instruction correspondence and runtime layout tables. It owns no report/module/heap edge. Old call sites pass NULL. Exactly one selected context may be present; there is no fallback to the legacy path after map selection. The signature must be threaded through root call supervision and every ASSERT resume, rather than reconstructed from a boolean proof bit. Existing old private owner sizes and old NULL-route budgets remain unchanged. Shared scalar handlers remain the implementation of arithmetic, calls and control flow; this is not a second interpreter.

The private public-facing test API is create/run/destroy/stats plus copied observations, analogous to the existing private array API, behind an explicit private build macro. Create leaves output unchanged on failure; run rejects busy/wrong-thread state; destroy cannot run during an active invocation. Observation identities are instance+epoch scoped and never export mutable map pointers or entry pointers. Map field observation returns copied count/key/value information through bounded index/key queries, not a heap pointer.

## I stage actual public selection separately

The eventual public chain must be audited together: map extension classification → load/save/asm/verify/query → selected execution preparation → `vm_execute` root wrapper → root/core scoped dispatch. MALFORMED never becomes ABSENT. Well-formed unsupported pairs remain refused. Private runtime green does not authorize public structural acceptance. `vm_create`/ownership gates, direct `vm_call_function`, external/linked module calls, callbacks and opcode-trace entry points must all distinguish selected maps before the existing array/legacy routes. A call to a helper cannot bypass the closed initializer+entry contract.

The eventual public wrapper owns a newly prepared map plan until initializer, entry, all assertion pauses and failure cleanup complete. It checks exact root module, idle frames/stack, no references/activation floor, no linked modules/callbacks/tracing, and ready counted string constants. It creates a call-local invocation view; no VmState pointer to that view persists. Public callbacks/reentry are refused, not allowed to observe a temporary view. Initializer result is consumed as the existing public API specifies; entry result remains rooted until normal publication/return. Errors unwind frames/operands before freeing the plan. The private run retains prior published result across failure; globals need not roll back. Both routes retain first-error status and destroy every owned heap edge exactly once.

No public selector/admission change belongs to the first private runtime implementation. Native still refuses map authority until its independent consumer is qualified. A public verifier change cannot accidentally make `nvm2c` accept through an unrelated generic route.

## I establish map storage topology, not just tags

`heap.c` uses counted STRING hashing/equality in `hm_find_slot`; its bounded linear probe stops at an empty bucket, remembers tombstones, and returns the first equal key. `hm_resize` builds replacement storage and transfers entries without retain/release, but assumes occupied keys are unique and reachable; it ignores `found` while reinserting. Therefore header tags/counts alone are insufficient physical validation.

The closed route permits only maps created by HM_NEW and mutated by checked HM_SET. It imports no host map. NEW establishes nonnull allocated storage, the existing initial capacity, zero occupied/tombstone counts and exact STRING/INT tags. SET preserves unique keys, probe reachability, exact occupied count and empty/tombstone state rules. No DELETE is admitted, so selected-created maps retain zero tombstones. Aliases refer to the same live map; no host or callback can mutate it between preflight and handler. Historical origin proof remains static authority for allowed field stores; a physical pointer is not assigned a new static origin by its address.

An independent physical boundary validator must check header identity/refcount, exact tags, checked bucket allocation arithmetic and capacity, legal state bytes, occupied count, STRING key validity and INT values. For each occupied bucket it must use the same probe relation and require lookup returns that exact bucket. This simultaneously rejects duplicate equal keys and entries unreachable behind an empty bucket. Merely checking a key can be found is insufficient. This diagnostic/import validation is explicitly charged, including byte hashing/equality and every probe; it may be quadratic in capacity and must fail before exceeding the boundary budget. It is not run on every opcode and never silently grants trust to a forged allocation extent.

The first route accepts no imported physical maps. Thus normal dispatch relies on construction/checked-mutation invariants, with constant-time header/tag/count arithmetic at each operation. Existing lookup is bounded by capacity; rehash visits old capacity and its insertion probes. I propose no cumulative runtime work limit or new map-size cap: preparation limits remain preparation limits. Observation/import validation gets an explicit call-local work limit, distinct from guest execution. If a checked handler cannot preserve topology inductively, it must refuse rather than invoke an unbounded hidden validation scan.

## I check before pops, retains and table commitment

Map preflight must verify decoded recipe identity, frame/operand boundaries, exact receiver pair and operand tags before any pop or mutation. GET result remains INT|VOID, including after HAS. No presence inference is allowed from literal spelling or a previous HAS. GET retains no heap result under this pair; it releases map/key inputs and publishes INT or tagged VOID. HAS returns BOOL; LEN returns INT and consumes its receiver. NEW allocation failure publishes nothing.

SET receives map/key/value in original order. Existing replacement retains the new value before replacing/releasing old value; INT makes both operations trivial here. It does not retain a replacement key. Insertion stores and retains one STRING key, so checked headroom must precede any growth/table mutation. A private checked heap adapter should share `hm_find_slot` and `hm_resize`, not duplicate their hash/probe implementation. Its invocation-local probe result may only be used immediately before mutation without reentry; resize invalidates the slot and requires a new lookup. A conservative retain check on replacement would change behavior at maximum key refcount, so insertion vs replacement must be established exactly. Duplicate interned operand aliases and the map's stored key retain must be counted distinctly.

The scoped checked SET failure path releases each popped input once and publishes no map result. Allocation failure before resize commit preserves the old table and keys. Resize only replaces storage after successful complete rehash; on success, the subsequent insertion must be guaranteed by topology/capacity or reported as a separately reviewed invariant defect. A failure must not be described as full table rollback if allocation/topology already committed. Existing ordinary heap handlers are not rewritten incidentally; the selected adapter and shared primitive changes, if needed, require old-route tests.

## I close a concrete private teardown prerequisite

Current `vm_private_dag_release` in `heap.c` handles only STRUCT/ARRAY parent frames and inline STRING. A HASHMAP currently reaches an assertion. Reusing that heap policy without a map-specific release case is unsound. For exact STRING→INT only, a dead map can release each occupied STRING key inline, skip scalar INT values, free bucket storage and then the map. It needs no additional parent frame because map keys cannot own records/maps. The existing259-entry record/array release stack and old heap struct size can remain unchanged. Counts/stats must use actual bucket capacity, not element count. No recursive vm_release call may corrupt the active iterative parent walk. Maximum record-chain→map→STRING teardown, shared keys and maps, failed insertion/growth, intern-table unlinking and forced collection need actual qualification. General map-value heap objects remain excluded.

Ordinary public `release_hashmap` and cycle collection already know maps, but that is not evidence for the private DAG policy. Public forced-GC tests must independently establish key identity and cleanup.

## I separate preparation, observation and guest costs

The private runtime constructor must present an exact allocation ledger before implementation: owner, image retained payload, copied runtime facts/layout tables, decoded/dispatch capacities, stack/frame/global/constants storage and any optional validator workspace. Reuse the qualified retained-cost helper for image/plan overlap; do not reserve two historical128MiB peaks or count a borrowed plan twice. Every allocation is charged before it occurs, output remains atomic, and failure frees all partial owners. Old private-array preparation bytes/work remain byte-identical under NULL map context.

Immutable query bounds remain declaration20/frozen-field24/frozen-origin72 logical work; charge preparation visits before query calls or copy needed pairs once into immutable runtime tables. Normal handler lookup/rehash work follows existing map semantics and is not debited from a preparation counter. Observation validation separately charges table/byte/probe traversal and preserves caller outputs on LIMIT. Exact numeric runtime-constructor ledger is a prerequisite for its source checkpoint, not inferred from current recipe/image costs.

## I require the following evidence before public admission

1. Unchanged STRING→INT anchor through actual private dispatcher, both switch/computed-goto, initializer+entry and true/false assertion resume. Public/native/serialization refusal remains tested during private phase.
2. GET miss stays VOID; zero/empty keys, counted embedded-NUL VM keys and equal-prefix distinct keys preserve exact bytes. Native NUL refusal remains the retained target boundary; no broad native STRING claim.
3. Distinct and shared maps/keys across record fields, local/global/call aliases, replacement, insertion and growth; original write history and field identities remain enforced.
4. All preparation and actual map/string allocation prefixes, duplicate-key/refcount headroom, exact prior result preservation, first-error cleanup and independent retry. Corrupt topology, duplicate equal keys, unreachable buckets, illegal states/counts/pairs refuse before mutation.
5. Source/image destruction independence, no stale context through ASSERT, return, failure, repeat run or destruction; direct helper/callback/linked/reentrant entry refusal.
6. Maximum bounded record DAG teardown with map terminal, old259 stack unchanged, all owners released, old array/recipe/image budgets unchanged and old dispatch neighbors green.

Likely implementation files are new map runtime private header/prepare/run includes, explicit optional context threading in `vm.c`, narrowly scoped checked map primitives/private DAG teardown in `heap.c`, private copied-image query accessors if necessary, and dedicated tests. Public admission files, `nvm2c`, source publishers and declaration grammar are deliberately separate later changes. This document authorizes none of those edits.

## I refine the constructor boundary and exact charge ledger

The smallest proposed first API takes an already prepared immutable source plan:

```
NvmArrayEligibilityResult vm_record_map_private_create(
    const NvmRecordMapExecutionPlan *source, VmRecordMapPrivate **out);
```

The source is borrowed only during construction, never retained after success. This avoids silently performing two full proof reservations inside a module-to-runtime convenience constructor. The constructor independently re-proves an image and may then discard every source pointer. A caller may destroy source and original module immediately. A future module/public wrapper must account its preceding proof separately and is not included in this first API.

Let F/I/L/E/S/G be validated function/instruction/layout/field/string/global counts, c[f] code length, i[f] decoded instruction count, and len[s] counted string bytes. All count guards and checked arithmetic precede loops. Preparation caps remain128MiB bytes/33554432 logical work. The new private owner has fixed arrays for256 function signatures/layouts/runtime nominal IDs, an embedded VmState and invocation view, and scalar lifecycle state. Its exact sizeof is charged once; existing owner/VM/heap structs do not grow. No per-origin mutable runtime registry is allocated.

The additional-runtime reservation R is the checked sum below; zero-count allocations remain absent, positive allocations use exact requested capacity. Each allocation also charges its bytes as logical initialization work. Fixed owner arrays are included only in sizeof(owner).

| Allocation | Reserved requested bytes |
| --- | --- |
| New owner | sizeof(VmRecordMapPrivate), including embedded VmState/frame array and fixed256 tables |
| Copied augmented instructions | I*sizeof(NvmRecordMapExecutionInstruction) |
| Runtime field rows | E*sizeof(VmRecordMapRuntimeField), with exact declaration, array depth/leaf and map key/value pair tags |
| Decoded/dispatch function tables | F*(sizeof(VmDecodedFunction)+sizeof(VmDispatchFunction)) |
| Per-function offset tables | (c[f]+1)*(1+4+4), matching boundary/index/dispatch arrays |
| Per-function decoded allocation overlap | sum(cap*sizeof(VmDecodedInstruction)), cap=16,32,... first cap>=i[f]; zero if i[f]=0 |
| Per-function dispatch instructions | i[f]*sizeof(VmDispatchInstruction) |
| Root stack | (VM_MAX_FRAMES*512+1)*sizeof(NanoValue), unchanged existing capacity |
| Globals | G*sizeof(NanoValue) |
| Constant string pointer table | S*sizeof(VmString*) |
| Intern table overlap | sum(cap*sizeof(VmString*)), cap=256,512,... sufficient for S at existing3/4 load threshold |
| Interned constant strings | sum(sizeof(VmString)+len[s]+1), conservative even for duplicate bytes |

The decoder overlap sum is deliberate; it covers realloc old/new coexistence without assuming in-place realloc. Constants reserve every original string even if interning reuses storage. No map bucket is allocated during preparation: guest HM_NEW allocations retain ordinary runtime failure semantics. The exact runtime field struct and compiler sizeof values are measured before implementation credit. No arbitrary alignment or allocator-overhead estimate is included.

The pre-scan obtains immutable counts/signatures/string lengths only via bounded source getters. Charge16 entry work,32 per function,16 per layout/field/array binding visit,24 per map field query and8 per string before access, plus actual retained-cost helper schedule. Array chains remain at most3; each traversed type row is charged before access. Allocation byte charges above are separate from scan charges. Decoder work adds sum(4*c[f]+64*i[f]); correspondence work adds256+128*I+32*F+sum(c[f]+1); constants collision/intern work adds sum((len[s]+1)*(s+8)+16) plus the explicit successful-rehash traversal reserve defined below. All arithmetic is checked and the complete positive reservation is validated before allocations/decoder/constant building. No counters are hidden in immutable getters.

The image constructor must receive positive pre-reservation Rbytes/Rwork, including owner and complete runtime preparation, before it begins reproof. Proposed private API is `nvm_record_map_image_copy_reserved(source, extra_bytes, extra_work, out)` with the same checked positive-only semantics as the already tested internal image helper; no credit/refund or executable token is exposed. Its ordinary copy wrapper remains0/0. Cost scan work inside that constructor is still charged there; the runtime pre-scan does not call it and then discard/reset its work. Total reported preparation reservation includes exact retained source+image proof/copy overlap+R, so source remains covered throughout runtime setup even if no longer needed afterward. This conservatively sums runtime buffers with image peak; it does not add two128MiB historical maxima. All errors leave out unchanged and source queries valid. An allocation oracle must bound actual combined live payload and independently compare retained runtime bytes after source destruction.

## I define checked heap operations and observation limits

Proposed private adapter statuses are `VM_RM_HEAP_OK`, `INVALID`, `MEMORY`, `RETAIN_OVERFLOW`, `LIMIT`; LIMIT applies only to explicit observation validation, never ordinary guest lookup/SET. All adapters are internal, not user-supplied proof APIs:

```
VmRmHeapStatus vm_rm_heap_set(VmHeap *, VmHashMap *, NanoValue key,
                              NanoValue value); /* borrowed inputs */
VmRmHeapStatus vm_rm_heap_get(const VmHashMap *, NanoValue key,
                              NanoValue *out); /* borrowed INT|VOID result */
VmRmHeapStatus vm_rm_heap_has(const VmHashMap *, NanoValue key, bool *out);
VmRmHeapStatus vm_rm_heap_validate(const VmHashMap *,
                                  VmRmObservationWork *work);
```

The setter shares existing hash/probe/rehash primitives, checks exact tags/header, performs lookup, then checks retain headroom only for insertion before resize. It performs no callback/reentry. Borrowed inputs remain owned by VM stack until the handler consumes them; the adapter never releases caller operands. On failure the handler releases popped operands once; on success SET returns the original map edge. No slot pointer survives resize. Get/has preserve output on INVALID; absent GET succeeds with VOID. Error mapping is MEMORY→VM_ERR_MEMORY and invalid/retain-overflow→the selected route's existing type/headroom error, with the first failure preserved. A topology invariant failure after a successful resize is not mislabeled allocator failure or promised rollback; induction and corruption controls must make it unreachable for admitted storage.

Observation validation has a call-local maximum33554432 logical units, no allocation and no persistent mutation. It charges16 for entry/header checks;8 before each bucket visit; for every occupied STRING key, len+1 before recomputing/validating its counted hash and exact string header; and4 before each probe plus compared_length+1 before any equality byte comparison. Count/state/occupied checks and exact-bucket lookup use this same charged traversal. Work/output structures must not overlap map storage. Overflow/LIMIT preserves observation outputs. It rejects duplicate keys, unreachable entries and tombstones on this no-DELETE route. Normal dispatch never spends this observation budget. It may inspect only constructor-owned live allocation metadata; it is not a memory-safe validator for forged arbitrary pointers.

The existing heap stats require a separate retained baseline: `hm_resize` has no heap argument and does not record bucket growth, whereas `release_hashmap` counts current bucket capacity as freed. A tiny original grow→release probe must distinguish this accounting discrepancy from ownership failure before any shared stats correction. Independent allocation/live-owner tracking remains mandatory and must not be weakened to match inaccurate counters.

## I bind correspondence and field pairs across control transfer

Preparation compares every original instruction, including dead rows, to both decoded and dispatch rows: opcode/operands, byte/next offsets, pop/push counts, current function, direct callee, branch successors, resolved target and no super-op/cached foreign call handle. The opcode universe combines the old94 explicit recipes with exactly the five map recipes; map rows carry the invalid-base sentinel plus exact map operation and masks, including GET INT|VOID. No numeric opcode gap becomes a default admission.

Before each dispatch, the invocation validates current frame module/function/signature/base and maps the actual decoded byte offset to the copied recipe. CALL pushes a frame whose direct callee/signature match the immutable row; the next dispatch checks the new frame. TAIL changes the frame but the next dispatch again verifies actual function/module/local and operand bounds against that function's copied rows. ASSERT supervision retains the same live invocation view; resumption verifies the same module and the now-current instruction offset. A cached recipe pointer is never carried blindly through a call, tail replacement or assertion pause.

Runtime nominal layout IDs map to copied declaration rows established during preparation. For STRUCT/AGG field read/write, the exact receiver runtime layout and field ordinal select the copied field pair; a HASHMAP value must physically be STRING→INT and come from the closed construction/mutation domain. Static origin histories already proved every reaching field value; runtime checks do not equate a pointer with a static origin ordinal or create authority from pointer identity. All maps in this first executable profile have the single exact pair, so there is no ambiguous runtime pair selection to infer. Wrong receiver/layout/field/pair is rejected before overwriting/releasing the prior field. Aliasing itself remains permitted by the static proof and shared mutable map semantics.


## I charge chained interning and reuse the qualified decoder schedule

String constants use **chained** buckets (`src/nanovm/heap.c:70–118`), unlike the map's open-addressed buckets. With zero-based constant index s, at most s earlier unique nodes exist. Lookup visits at most s nodes even if every hash collides. A logical chain probe includes the fixed hash/length comparison; at most len[s] byte comparisons follow it. Thus s*(len[s]+1) bounds lookup. The remaining 8*(len[s]+1)+16 reserves counted hashing, copying/terminator handling, fixed constructor/retain checks, growth-attempt setup and insertion. These are bounded logical operations, not machine instructions or allocator-internal work. Duplicate constants only reduce nodes/allocations; they still receive the full charge.

I additionally reserve sum(C+S) over each possible successful intern-table doubling whose old capacity is C (256,512,... below the final reserved capacity). Each successful growth scans C old bucket heads and at most S existing nodes, using their cached hashes; each capacity can grow successfully only once. Initial table creation has no old heads/nodes. Failed growth does not traverse or change the table, but may leave it denser: therefore I do **not** replace S with 3*C/4 or assume a successful previous growth. At most S insertion attempts occur; their fixed failed-attempt handling is already in the per-string reserve. Requested allocation-byte charges cover new bucket initialization and conservative old/new coexistence. Allocation failure internals are outside logical work accounting. All terms are checked before allocation; overflow or the unchanged cap refuses preparation. This additional traversal charge is new-map-only; no old constant budget changes.

For decoding I reuse exactly the existing qualified `vm_ra_decoder_budget` schedule in `src/nanovm/record_array_vm_prepare.inc:323–340`: function-table bytes, offset/dispatch bytes, every decoded realloc capacity, then 4*c+64*i logical work per function. This is an existing conservative preparation policy, not a new assertion about 64 CPU instructions. Its covered passes are byte decoding and boundary/index construction, target resolution, dispatch-row copying, dispatch-target resolution and the fusion scan (which still visits rows with all fusion flags disabled). Copied/inited allocation capacities are charged separately in bytes. The independent map correspondence pass keeps its separate 256+128*I+32*F+sum(c+1) schedule; it is not claimed to fit inside the decoder charge. Implementation must reuse that schedule and qualify unchanged old-path budget receipts, rather than substitute a smaller guessed instruction count.

## I keep selected context and actual retained edges explicit

The source sites below are an implementation checklist, not execution authorization:

| Current site in src/nanovm/vm.c | Required separate map route |
| --- | --- |
| vm_core_execute_scoped:1644, selection:1649–1725 | Accept a call-local map view; old array owner remains NULL, with a distinct selection flag. Selected-map failure cannot fall through ordinary admission. |
| instruction preflight:1987 and retain preflight:2052 | Check the map recipe/frame first, then every actual retained edge before the shared handler pops or mutates. |
| CALL frame statistics:3178 | Use a map-owner accessor; never dereference record_array for a map invocation. |
| STRUCT_NEW:4223, STRUCT_LITERAL:4274, AGG_PACK:4401 | Resolve runtime layout through a map-view accessor bound to the copied image, not record_array->record_layouts. |
| HM_NEW/GET/SET/HAS/LEN:4499–4616 | Use checked map adapters and exact pair/operand/result correspondence; DELETE/KEYS/VALUES remain refused. |
| scoped call/root wrappers and ASSERT resume | Thread the same live map view through initialization, entry, CALL/TAIL and every resume; unwind before destroying its owner. |

Separate internal accessors are proposed as `vm_rm_runtime_layout(view, ordinal, out)` and `vm_rm_note_frame_count(view, count)`. Both first check the view/VM/image identity; layout lookup is bounded and preserves out on failure. They neither cast the old array owner nor establish authority from a heap pointer.

The map retain preflight must cover STRING, ARRAY, HASHMAP **and record shells** on actual retaining LOAD_LOCAL/LOAD_GLOBAL/DUP/AGG_GET and any admitted array read/copy operation. Existing array-only final tag checks at1534–1605 are insufficient. Transfer-only stores, packs and returns must not gain spurious retains. CALL argument moves and TAIL staging must be audited separately against their actual handler retain/transfer rules; repeated identical arguments require the exact number of new edges, not one check per distinct pointer. Map SET checks the one new key edge only after determining insertion; replacement needs none. GET's existing vm_retain is trivial for the selected INT|VOID result. These checks must precede mutation and preserve existing error/unwind behavior; qualification includes maximum-refcount shells/maps, repeated interned keys and staging aliases.

## I retain the original heap accounting prerequisite

The authorized existing-provider probe in `/private/tmp/nanolang-map-runtime-audit-20261021` inserted 64 unique STRING keys into an INT map, grew buckets16→128 and released/collected it. Actual output was allocated2646, freed-before0, objects65; then allocated2646, freed-after7126, objects0. The difference4480 is exactly (128-16)*sizeof(VmHMEntry). Existing hm_resize omits allocation-stat growth while release_hashmap reports current capacity. Build/run exited0; the first two harness compile failures (size_t printf format, then missing retained-link globals) are retained separately. This is accounting evidence, not independent proof of no leaks: runtime qualification still requires allocation interception/live-payload balance, prior-result survival, balanced retry and destructor checks. No shared stats correction is included in this contract checkpoint.
