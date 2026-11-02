# Private map invocation and dispatch checkpoint

Status: design-only after heapf8c2f127f (aggregatead5332c58). I own the #1001 private invocation audit. This document proposes the next implementation boundary; it changes no source or admission. The first executable input is the unchanged STRING→INT map_record_fields fixture, using its retained original modules and independently validated private annotation. Public load/verify/execution, native translation and publishers continue refusing selected map metadata.

Review approved the source-only run/context/dispatch draft. Observation APIs below remain deferred: the first implementation exposes only stats and uses whitebox copied-state qualification. Source review precedes the first unchanged switch-dispatch gate. The supervisor must reset active_root and root_base separately for initializer and entry; cleanup invalidates the view before publishing idle state while retaining DAG teardown policy.

## Owner and call-local view

The existing prepared owner exclusively owns the image, which owns the plan and copied proof. Its plan pointer remains a const borrow. Decoded/dispatch tables, runtime fields, constants, stack and globals already belong to the owner. No Analysis, mutable historical Value, source-module pointer or static-origin token is reconstructed from a runtime pointer.

Proposed runtime-only APIs, behind NANO_RECORD_MAP_PRIVATE_RUNTIME, are:

```
VmResult vm_record_map_private_run(VmRecordMapPrivate *);
bool vm_record_map_private_stats(const VmRecordMapPrivate *, VmRecordMapPrivateStats *);
bool vm_record_map_private_observe(const VmRecordMapPrivate *, VmRecordMapRoot,
    uint32_t global, const uint32_t *path, uint16_t path_count,
    VmRmObservationWork *, VmRecordMapObservation *);
VmRmHeapStatus vm_record_map_private_lookup(const VmRecordMapPrivate *,
    VmRecordMapRoot, uint32_t global, const uint32_t *path, uint16_t path_count,
    const void *key_bytes, uint32_t key_length, VmRmObservationWork *,
    VmRecordMapScalar *);
```

Root is published result or a bounded global index. Observation copies tag/scalar bits, count, layout and an instance/epoch-local identity; no live pointer is returned. Lookup returns copied INT or VOID, not an owner. It accepts counted bytes solely as an observation query, never as an imported guest STRING/map. Paths traverse record fields and existing exact array elements; a map terminates the path. Path length stays bounded by the existing258 observation limit. Output/work storage must not overlap the owned graph. Outputs remain unchanged on failure. Observation charges each path step, physical validation and actual counted-key probe/comparison before work; it reuses the33554432 observation cap, not guest fuel. Missing lookup is success with VOID. Exact public struct layouts/query charge schedule must be frozen before this observation implementation; whitebox tests may inspect copied fields meanwhile, but may not call that a public observation API.

The owner adds runtime-only epoch, result, last_status, maximum_frames, busy and has_result fields, under the new runtime macro. Existing preparation-only and array-only owner sizes remain unchanged. Actual new sizeof and reservation are measured before qualification. The root/result owns exactly one edge when present. Destroy rejects wrong-thread or busy state and retires result, globals, constants and remaining roots before releasing image storage. No borrowed plan is freed twice.

The invocation view is automatic storage in run, not an embedded persistent view:

```
typedef struct {
    VmRecordMapPrivate *owner;       /* borrowed, busy for this invocation */
    VmState *vm;                    /* exactly &owner->vm */
    const NvmModule *module;        /* image-owned */
    uint64_t epoch;
    uint32_t active_root, root_base;
    bool active;
} VmRecordMapInvocation;
```

Only the root supervisor creates/invalidates it. It is passed through initializer, entry, every real core dispatch and ASSERT resume. Nothing stores it in VmState, heap, decoded rows, callbacks or global storage. Epoch cannot wrap. Busy/wrong-thread/reentry, active stack/frames, foreign root/module, callback/linked/trace state, fused dispatch and selected-context overlap refuse before entry. There is no asynchronous suspension or external trap service. After cleanup the view is invalidated and busy cleared, on success and failure.

## Actual core call sites and old-array isolation

Extend only the internal vm_core_execute_scoped signature with an optional map-view parameter. The ordinary core wrapper (vm.c4942), ordinary call supervisor (5129), and record_array_vm_run.inc pass NULL. New map root supervision passes NULL old-array/ordinary/owned contexts plus its exact view. Reject simultaneous map and old-array/ordinary/owned contexts rather than choosing precedence. Public calls do not synthesize a map view.

Keep old record_execution semantics array-only. Introduce a separate map_execution discriminator. Admission may be bypassed only when one exact private context has passed its own checks; a malformed map view never falls back to ordinary verification. Do not broaden record_execution then dereference record_array when it is NULL.

Known shared sites requiring explicit two-context branches are: pre-dispatch validation (1990), retain preflight (2055), maximum-frame accounting (3181), STRUCT_NEW/STRUCT_LITERAL record ordinal→layout (4226/4277), and AGG_PACK record ordinal→layout (4404). Use separate checked map accessors for layout mapping/frame accounting. Old-array pointer/table access and calls remain unchanged on the NULL map route. A map operation's invalid-base-recipe sentinel never enters the old recipe switch as a valid array recipe.

## Correspondence and preflight before effects

vm_rm_context requires prepared owner, exact busy epoch/view, exact image module/root, !verified, owned decoded/dispatch tables with fusion disabled, fixed stack capacity, bounded frame count and exact active-root identity. Every frame refers to the same image; no closure, owned callable, effect activation, references, linked module or callback is allowed. Validated construction plus restricted handlers establishes the closed heap graph; header checks alone do not authorize a forged host graph.

vm_rm_preflight runs before IP advance or any pop/store/retain. It resolves the current function-relative byte offset through the copied decoder index, checks bounds before indexing, then compares opcode, all operands, byte offset/length and next/target boundaries with the immutable augmented recipe. It checks the actual current frame function/local count/base/module, exact operand region, pop/push counts, scalar tag obligations and output capacity. Map rows require the exact map operation, invalid base sentinel, STRING key mask, INT value mask and prescribed result mask. GET remains INT|VOID, including after HAS; no key-presence inference. Old rows require NVM_RM_NONE and their exact valid old recipe.

Runtime values are checked against copied nominal record rows, exact field tags/referents, array depth/leaf and map STRING→INT pair. A map shell's address is only a physical alias identity for refcount/observation checks, never a proof origin. Static historical map writes and record-field closure remain authority from the independently reproved image. Record field stores use copied pair rows; plain map locals do not invent a new pair or escape through excluded map parameters/results/globals.

The fast map path relies on construction induction: initial roots contain only validated constants/scalars; NEW constructs exact empty maps; checked SET alone changes tables; records/locals/calls transfer existing edges; no import/callback/DELETE exists. Topology validation is required for explicit observation and defensive boundary tests, not rescanned with a new guest budget before every operation. Corrupt fixture graphs must refuse at the explicit boundary and be restored before teardown. No claim is made that arbitrary invalid pointers can be safely examined.

## Retains, CALL/TAIL and returns

Map-specific retain checks cover actual edges for PUSH_STR, DUP, LOAD_LOCAL/GLOBAL, record/array field loads, array push/slice, and all existing supported record construction/update paths. Generic old vm_record_array_retain_preflight only checks ARRAY/STRING at its final guard and therefore cannot by itself protect MAP/STRUCT shells. The new preflight must check the actual next retain of every heap tag it permits, including duplicate children near saturation; no blanket reuse silently omits map shells.

CALL arguments already occupy rooted stack positions and transfer into new locals without extra retain. Validate exact signature/arity, operand count, frame capacity and copied maximum stack before mutation; then use the real shared CALL handler. New frame identity comes from the same image. Return validates zero/one declared results and physical value before removing locals; the shared RET transfers the result edge without inventing a retain. Initializer results below the entry base are not entry operands.

TAIL uses the real handler: stage every argument in original order, retain all before releasing old locals/operands, then install arguments. Check repeated identities collectively against UINT32_MAX, not one at a time against the same old count. More than16 arguments uses the existing fallible temporary allocation; failure leaves rooted arguments for unwind. Exact signature/result and maximum-stack checks precede staging. Root-frame tail changes update view.active_root only after successful frame replacement; ASSERT resume validates that new identity, not the original entry function. Helper tails cannot alter root identity. No new recursion or termination claim follows: only graph forms already proven by the map plan can execute.

## Map handlers and failure cleanup

Only the five map recipes branch to the checked adapters when map_execution is true. Old public handlers remain unchanged otherwise.

| Recipe | Stack ownership and result |
| --- | --- |
| NEW | Validate encoded STRING/INT pair and output room; allocate checked empty map; publish one new owner edge. |
| SET | Pop value,key,map only after preflight. Adapter borrows them. On success consume scalar value/key and transfer original map edge to output. On adapter failure release every popped edge once and report its exact status. Replacement performs no key retain; insertion checks one key headroom before growth. |
| GET | Pop key/map; checked lookup returns scalar INT or VOID; consume key/map once, publish scalar with no fabricated owner retain. |
| HAS | Pop key/map, consume once, publish BOOL. |
| LEN | Pop map, consume once, publish INT. |

Adapter MEMORY and RETAIN_OVERFLOW map to VM_ERR_MEMORY, INVALID to VM_ERR_TYPE_ERROR. LIMIT is observation-only and is not a possible fast handler result. No adapter status is ignored or treated as legacy success. Allocation failure after pops follows the actual handler's ownership cleanup; preflight failure before pops leaves stack roots to unwind. No handler may overwrite the first VM error while releasing roots.

The private DAG release policy activates only after copied layouts/recipes/constants and closed root state pass preparation/runtime checks. Once live map DAG roots exist, it stays active through result/global retention, repeat runs and final destruction; returning from run must not switch those objects to a different release policy. The flag alone is not dispatch authority. All VM-layout-dependent providers consistently enable preparation+runtime+map-heap+array-runtime macros. Old builds retain their exact structs and behavior.

## Root supervision and publication

run starts with idle stack/frames and preserved prior result/globals. After epoch/busy setup it executes the selected zero-arity initializer, if any, then zero-arity entry using real scoped dispatch. For each root separately, the supervisor sets active_root to that function and root_base to the current stack size. The initializer's permitted result remains below entry root_base, matching existing execution semantics. True ASSERT consumes its condition once and resumes with the same live view; false ASSERT returns VM_ERR_ASSERT_FAILED and never resumes. Any PRINT/FFI/service/callback trap is unexpected and refuses; the admitted recipe set cannot silently request a host service.

Success requires the exact final result count/tag/value and no frames. Detach the new result edge, unwind initializer residue and frame roots, then atomically replace the published result and release its predecessor. Failure preserves the previous published result edge and identity; it does not promise an immutable snapshot of a map that aliases a mutated global. Globals are intentionally not transactional and may reflect earlier stores. Unwind clears all pending operands, locals and callable roots, resets idle frame/IP/module state, and cannot allocate or replace first-error status. Failed-run globals remain valid closed graph roots. Destroy eventually releases them. Repeated success/failure/retry must have balanced independent allocation/owner accounting.

## Costs, source scope and qualification

No guest work/fuel limit is added. Existing128MiB preparation and33554432 logical limits remain. Runtime owner additions and simultaneous automatic invocation view are positively reserved by the existing constructor/image-overlap schedule, with exact sizeof measured before publication; no negative credit or hidden persistent pointer. Decoder/recipe/layout/constants allocations and old preparation-only budgets stay unchanged. No new heap allocation is needed merely to create the call-local view. Existing guest map growth and >16 tail staging remain actual runtime allocations, separately fault-injected.

Proposed source scope after review: private runtime header, runtime-only owner fields/preparation reservation, new private run/context/preflight include, explicit optional-argument threading and scoped handler/accessor branches in vm.c, and focused tests. Heap adapters are reused without another rewrite. No ownership verifier, metadata grammar, public selector, native backend or source publisher changes.

Required first gate: unchanged original modules through actual private switch dispatcher with existing assertions, copied source/module destroyed, exact result and all roots reclaimed. Then computed-goto parity; initializer result offset; true/false ASSERT resume; missing GET and zero/count-NUL keys; distinct/shared record/map/key aliases; replacement/growth; direct and self-tail where already proven including swapped distinct owners/>16 arguments; prior-result and first-error preservation; every real map/string/tail allocation prefix with independent retry. Test explicit wrong-context/frame/recipe/epoch refusal, map/record headroom, topology corruption at boundary, busy/thread/reentry, maximum release depth, all old NULL-route budgets and array/ordinary neighbors. Continue testing public/native/load refusal. Observation implementation is separately reviewed against its concrete structs/charged walk before relying on it as a black-box oracle.
