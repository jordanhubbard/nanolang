# Private borrowed-state map activation and shared dispatch

This design follows preparation component84768a3ba and
[NANOISA_MAP_PUBLIC_CONTEXT_DESIGN.md](NANOISA_MAP_PUBLIC_CONTEXT_DESIGN.md).
It proposes actual execution through a private test entry over an ordinary
caller-owned VmState. Public vm_execute/nvm2c/load/save/assembler/publisher
selection remains refused. No caller supplies a preparation, import report,
trusted context, or executable plan. The completed observation/preparation APIs
remain non-executable.

## Exact entry and ownership

The proposed macro-private header exposes only:

```c
VmResult vm_record_map_borrowed_execute_private(
    const NvmModule *module, VmState *vm);
```

It requires module==vm->module==vm->root_module. The caller owns the module,
decoded/dispatch caches, constants, stack, globals and ordinary heap throughout.
Managed allocation extents, balanced owner references, exclusive owner-thread
access and stable source/caches are C API preconditions. No callbacks, external
calls, linked modules, effects, reference activations, closures, trace callback,
foreign thread or reentry are authorized. The exact99-opcode proof whitelist and
whole-module closure establish the executable subset; a valid data declaration
alone is insufficient. The public verified bit is not authority.

The entry privately calls vm_record_map_public_prepare once. Its owned boundary
plan remains live through import validation, both roots, assertion resumptions,
all nested direct calls/tails, cleanup and result validation. It owns immutable
copies only. The entry never frees the original module or caller storage. It
never creates a temporary private owner and copies a scalar result out of it.
The entry remains zero-arity with an INT result; an initializer remains zero-arity
with a VOID or INT result, as required by the existing preparation checks. No
heap-valued host entry result is admitted. Success leaves the declared INT entry
result on the actual caller stack. Existing
vm_get_result remains a borrowed peek; the caller must consume/release that result
before invoking the gate again. Supported helper/intermediate heap, record and
array values retain their real owner edge and identity. There is no persistent selected-map flag or context in VmState.

A stack-local view retains the already budgeted40-byte layout:

```c
typedef struct {
    const VmRecordMapPublicPreparation *preparation;
    VmState *vm;
    const NvmModule *module;
    uint32_t active_root, root_base;
    bool active;
} VmRecordMapBorrowedInvocation;
```

The view expires before the entry returns or frees preparation. It is never
returned, stored in VmState, or accepted by an external execution API. Root
coordinates reset independently for initializer and entry. A root tail updates
active_root; root_base remains the original caller-stack base. No pointer to a
root local outlives its synchronous call. Existing private-owner epochs remain
unchanged; this view has no persistent epoch token to replay.

## Activation and reservation

Preparation is const and has already independently validated ordinary imports
using the same boundary plan. Before allocating stack storage the entry rechecks
thread/root/idle identity and exact stack pointer and capacity against
preparation.original_stack/original_stack_capacity. It also rechecks globals and
cache/root headers needed to exclude an intervening invocation. Exclusive
stability is still required; these checks do not implement concurrent validation.

Only then call the existing stack_reserve for R=VM_MAX_FRAMES*512+1 values. Its
rounding must produce the preparation's recorded rounded_stack_capacity. Capacity
already covering R has no realloc. The preparation charged full possible old+new
storage overlap, but only possible copied old bytes+64 logical work; newly reserved
capacity is not initialized by realloc. No allocation or budget credit is invented
for the no-growth case. Overflow or reserve failure occurs before guest mutation.
On failed realloc the original pointer/capacity and all owners survive. Returning
a diagnostic may update last_error/error_msg; it does not clear globals/results
or normalize collector state. The const preparation API itself remains fully
nonmutating, including diagnostics.

After successful reserve, no handler in this closed graph needs stack growth:
each of at most1024 frames uses at most256 locals and256 operands, with an extra
result slot. CALL checks frame count and complete local+maximum-stack room before
mutation; TAIL replaces a frame and checks the replacement footprint. The
initializer result is drained before entry, so their root maxima do not add.
The view checks capacity against the recorded rounded value and non-NULL backing;
exclusive execution and the whole-frame bound keep the stack pointer stable after
activation. Preparation's original_stack is used only for pre-reserve comparison,
never dereferenced after realloc. No new active-stack pointer or size is added to
the view or old owner structures.

A later runtime failure is different from preparation/realloc failure. Ordinary
heap allocations, successful global stores and alias mutations may already exist.
Selected cleanup drains live execution roots but does not transactionally roll
back payloads, reset globals or restore the old stack allocation address.

## Typed core threading and immutable correspondence

The proposed internal core receives a sixth, distinct typed borrowed-map view:

```c
static VmTrap vm_core_execute_views(VmState *,
    const VmOwnedInvocationProof *, VmOrdinaryAdmission *,
    VmRecordArrayPrivate *, VmRecordMapInvocation *,
    VmRecordMapBorrowedInvocation *);
```

Keep the current five-argument vm_core_execute_context as an exact wrapper passing
NULL for the sixth argument. Keep the current four-argument scoped wrapper and
all old callers intact. Old NULL routes and old private-map routing do not change.
A borrowed-map view must be mutually exclusive with all other proofs/contexts;
an invalid non-NULL view fails closed before classification/dispatch. No failed
map context falls back to ordinary legacy execution. A separate
borrowed_map_execution flag must not accidentally enable map_execution's private
ordinal remapping, embedded-owner counters or DAG policy.

Preparation has checked all instruction and cache rows, including unreachable
ones. Before each handler mutates stack/heap, borrowed preflight rechecks the
selected immutable row: active frame/function/root coordinates, exact opcode and
operand widths/bits, recipe/obligations, call target/parameter signature,
successors/branch/next index, no resolved foreign call handle or superinstruction,
local/operand counts, frame cap and result footprint. CALL/TAIL/ASSERT do not
create a new proof. A successful ASSERT resumes with the same view and current
root identity. No public selector or generic host helper entry can obtain this
view.

Do not reuse vm_rmx_context by relaxing its tests. It requires embedded VmState,
private DAG policy and exact private stack size; it interprets record def_idx as
layout and globals by count. New typed helpers borrow preparation and ordinary
VmState. Pure recipe decoding, operand/signature equality, scalar-input checks and
retain arithmetic may be reused without changing their old semantics. Complete
physical value checks need ordinary wrappers; private !buffered checks stay intact.

## Ordinary heap, nominal identity and map adapters

The heap remains ordinary and cycle-aware throughout. Never set private_record_dag,
clear the suspect buffer, force a collection to make imports fit, or rewrite
collector colours. Fresh import validation establishes actual membership/topology
and conservative singleton obligations at the boundary. Valid positive buffered
PURPLE and unbuffered PURPLE remain accepted; unrelated zero-ref buffered BLACK
collector entries remain collector-owned. Guest allocations/stores follow the
proved closed graph and normal retain/release induction. There is no static-origin
ID assigned from a pointer, and no second singleton import check after initializer
or each allocation; guest historical alternatives can legitimately be non-singleton.

Ordinary STRUCT_LITERAL and AGG_RECORD store operand ordinal in VmStruct.def_idx.
Resolve ordinal through preparation.record_layouts for field/schema checks only.
For nested nominal fields, map the actual child's ordinal to layout and compare
that layout to the exact declared nested referent. Never replace def_idx with the
layout. Nonidentity ordinal/layout fixtures are mandatory. Field reads check the
selected value's exact scalar, nominal, array depth/leaf or map pair obligation;
field stores check the replacement before popping and preserve transfer semantics.
Complete imported graphs were checked once; runtime shell/selected-edge checks
must not accidentally revalidate uncharged unrelated siblings as a host query.
Runtime validation is not a new guest work-fuel policy.

The five map operations need distinct typed ordinary adapters under the private
qualification macro. They accept the established ordinary collector header states,
while current vm_rm_heap_* private adapters retain their existing header policy.
Suggested internal declarations use the existing VmRmHeapStatus and normal borrowed
operand conventions:

```c
VmRmHeapStatus vm_rm_ordinary_new(VmHeap *, VmHashMap **);
VmRmHeapStatus vm_rm_ordinary_set(VmHeap *, VmHashMap *, NanoValue key, NanoValue value);
VmRmHeapStatus vm_rm_ordinary_get(const VmHeap *, const VmHashMap *, NanoValue key, NanoValue *);
VmRmHeapStatus vm_rm_ordinary_has(const VmHeap *, const VmHashMap *, NanoValue key, bool *);
VmRmHeapStatus vm_rm_ordinary_len(const VmHeap *, const VmHashMap *, uint32_t *);
```

These are heap implementation interfaces, not host execution authority. Fast
lookup requires closed topology from this invocation's fresh import validator or
constructor and preserved by the exact setters. No exported bool/trust token can
skip validation. Common static implementation may share probe/commit code behind
separate typed wrappers. Do not rescan every full table before each operation or
reuse an old observation report as authority. Initial bucket/hash/reachability/
duplicate validation remains charged by preparation. Lookup/growth use existing
bounded table algorithms and guest allocation behavior; no runtime work budget is
added in this slice.

NEW constructs STRING→INT only. SET distinguishes replacement before insertion:
replacement changes the scalar value, with no key retain and no growth; even a
max-refcount lookup key can replace an equal key. Insertion checks key retain
headroom before growth, preserves the old table on failed allocation/rehash,
retains the one new STRING edge, then publishes the bucket/count with no remaining
fallible operation. Existing net-capacity statistics remain unchanged. Distinct
allocated equal-content strings, interned duplicates, collisions and counted-NUL
keys require actual tests. No STRING→STRING or DELETE scope is added.

GET returns copied INT or tagged VOID for absence; HAS does not refine GET to INT.
LEN returns the actual count. SET transfers the popped map owner to its result;
GET/HAS/LEN release consumed receiver/key owners exactly once. The instruction
adapter either performs fallible work while operands remain stack roots or stages
and releases every popped operand on every failure. All cheap tag/context/stack/
retain checks occur before pops. Helpers do not install DAG policy. Existing shared
ordinary release and vm_destroy reclaim keys, buffered containers and cycles.

## Exact recipe and retained-edge obligations

The allowed set remains exactly vm_rm_recipe's99 opcodes, checked against all256
possible byte values and immutable execution recipes. Map rows retain the invalid
base-recipe sentinel and separate operation/pair/result facts. No generic fallback
is authorized. The complete groups are:

```text
PUSH_I64 PUSH_U8 PUSH_F64 PUSH_BOOL PUSH_VOID PUSH_STR
NOP I64_ADD I64_SUB I64_MUL I64_DIV_S I64_REM_S
I64_NEG F64_TO_BITS F64_FROM_BITS F64_ADD F64_SUB F64_MUL
F64_DIV F64_NEG CAST_INT CAST_FLOAT CAST_BOOL CAST_U8
TYPE_CHECK EQ NE LT LE GT
GE I64_EQ I64_NE I64_LT_S I64_LE_S I64_GT_S
I64_GE_S F64_EQ F64_NE F64_LT F64_LE F64_GT
F64_GE BOOL_AND BOOL_OR BOOL_NOT AND OR
NOT STR_LEN STR_CHAR_AT ARR_LEN STR_EQ STR_CONTAINS
STR_STARTS_WITH STR_ENDS_WITH CAST_STRING STR_CONCAT STR_SUBSTR STR_TRIM
STR_TO_LOWER STR_TO_UPPER STR_REPLACE STR_FROM_INT STR_FROM_FLOAT STR_SPLIT
LOAD_LOCAL LOAD_GLOBAL STORE_LOCAL STORE_GLOBAL DUP SWAP
POP ASSERT JMP JMP_TRUE JMP_FALSE CALL
TAIL_CALL RET STRUCT_NEW STRUCT_LITERAL AGG_PACK ARR_NEW
ARR_LITERAL STRUCT_GET AGG_GET ARR_GET STRUCT_SET AGG_SET
ARR_SET ARR_PUSH ARR_POP ARR_SLICE HM_NEW HM_SET
HM_GET HM_HAS HM_LEN
```

| Family | Required preflight and actual ownership effect |
| --- | --- |
| Scalar/constants/string operations | Preserve original total/trap semantics. Typed scalar checks reject wrong heap operands before shared handlers pop them without releasing. PUSH_STR checks one extra retain; dynamic VM strings remain counted-byte values. Native NUL restrictions are unrelated to this VM gate. |
| LOAD_LOCAL/LOAD_GLOBAL/DUP | Validate actual source and one retained owner, including records and maps as well as arrays/strings. LOAD_GLOBAL uses allocated capacity, never high-water count. |
| STORE_LOCAL/STORE_GLOBAL | Transfer the stack edge into the destination before releasing its previous edge. No redundant retain. Global growth must fail before popping. Bound index by the copied global closure and VM_MAX_GLOBALS; preserve high-water/count semantics and do not clear globals. |
| Record constructors | Allocate before popping fields; preserve source-order ownership transfer. Exact ordinal, count and each declared field obligation are checked. No retain is added for a transfer. |
| Record GET/SET | Bounds/nominal check first. GET retains the selected field before releasing receiver; SET transfers replacement and releases previous field. Headroom follows actual retained edges, not every heap operand. |
| Array constructors/mutations | Preserve existing packed conversions, depth/leaf proof and actual handler bounds. Retains for PUSH and selected GET are checked; literal/store transfers stay transfers. ARR_GET's missing/invalid-index VOID behavior is not replaced with ARR_SET's bounds error. |
| ARR_SLICE | Count cumulative retained aliases among selected STRING/ARRAY children before constructing the copy; one-at-a-time max checks are insufficient. Preserve existing clamp/default-index behavior. |
| CALL | Arguments transfer from operand stack into callee locals. Check arity/types, exact cached target, frame cap, local and maximum operand footprint before publishing frame. No closure/effect ownership or host entry is admitted. |
| TAIL_CALL | Require exact remaining arguments and result signature. Check cumulative repeated-owner retains before staging; >16-argument staging allocation fails before old frame release. Preserve distinct-identity argument order and same-root result signature; update call-local active_root for root tails. |
| RET/ASSERT/branches | Exact result count/tag; no extra operands. ASSERT transfers condition to supervisor, which releases it exactly once and resumes on true with unchanged context. False ASSERT preserves first error and invokes selected cleanup. Branch successors remain copied exact byte targets. |
| HM_NEW/SET/GET/HAS/LEN | Exact STRING→INT pair, sentinel recipe and separate result mask. GET remains INT|VOID. Ordinary typed adapters preserve topology and the actual retained/consumed edges described above. |

The current vm_record_array_retain_preflight can contribute its already-qualified
array slice arithmetic, but alone it covers only ARRAY/STRING single retains and
omits ordinary STRUCT_GET/general MAP owners. Borrowed-map preflight must cover
all selected record/map retained edges. CALL does not acquire an extra edge;
TAIL does, once per staged argument, so duplicate owners are counted cumulatively.
Do not add blanket max-ref refusals to transfer-only operations.

## Root supervision and failure cleanup

A small new root supervisor uses the real shared core; it is not a bytecode
interpreter. Initialize a real root frame with ordinary module/ordinal conventions,
VOID locals and zero closure/effect/callable owners only after stack reserve and
preflight. Each root sets active_root/root_base independently. Execute initializer
first if present; validate its result signature and release all initializer residue
before creating entry. Entry success leaves its declared result on the caller
stack. No global reset occurs between roots or across calls.

ASSERT conditions are released once before either resumption or failure. Other
traps outside NONE/ASSERT/ERROR refuse within the closed profile; they do not invoke
a host callback. On error preserve the first VM error code/message and relevant
instruction location, clear reference activations, release stack roots down to the
root base, release any remaining frame-owned callable edges defensively, reset
frame_count/activation_floor, and prune effects exactly as selected-array cleanup.
Do not copy private vm_rmx_unwind's unconditional current_fn/ip reset or private
result publication. Imported globals and independent caller aliases survive;
mutations already made through those aliases remain visible. Diagnostic changes
and successful capacity growth are not rolled back. The view is invalidated on
every exit before preparation is freed and before the caller observes idle state.

The entry requires an empty stack/frame on entry, like selected-array invocation.
It therefore has no separate previously published result to retain. An unconsumed
prior stack result refuses before new guest execution. Ordinary legacy baseline
false-ASSERT stack3/frame1 is comparison evidence, not cleanup behavior to copy.
Selected failures must leave empty execution stack/frames while preserving globals
and permitting ordinary GC/destruction and a subsequent independent invocation.

## Concrete shared-handler hazards and reuse limits

| Current site | Required disposition for this separate route |
| --- | --- |
| vm_core_execute_context admission block and instruction preflight | Add a separate typed borrowed mode; invalid context never reaches legacy admission. Preserve current five/four-argument wrappers and old NULL behavior. |
| vm_call_function_impl | It resolves module calls before entry, potentially changing caches. Do not call it as a shortcut after immutable correspondence. A bounded new supervisor creates the real zero-arity root and uses the typed core directly. |
| vm_rmx_context/value_valid/field_accepts | Private embedded-owner, DAG, fixed-capacity, !buffered and layout-index assumptions are incompatible. New ordinary wrappers use validated collector state and ordinal mapping. |
| STRUCT_NEW/LITERAL and AGG_PACK private branches | Existing private branches rewrite ordinal to layout. Borrowed mode must never take them. Test ordinal0/layout1 and another nonzero ordinal, not only identity layouts. |
| LOAD_GLOBAL/STORE_GLOBAL | Private preflight checks global_count, ordinary reads capacity. Use capacity for reads and copied proof bounds for stores; let normal store grow before pop and update high-water count. Do not enable owned_execution's forced-GC/global-clear path. |
| Ordinary HM_GET/SET/HAS/LEN | They pop before generic operations. Selected typed adapter must own staging, actual release order and exact failure status. Generic SET does not prevent refcount saturation; stack reserve cannot repair that. |
| Shared stack_push sites | Some pushes ignore returned allocation status. Whole-invocation reserve plus exact per-op/frame footprint makes allocation impossible there; first independent reserve must be real and checked. No fixed private-capacity equality shortcut. |
| CALL/TAIL | CALL publishes only after frame reserve; TAIL retains all arguments before releasing old frame. Check cumulative duplicates and >16 staging failure, with two distinct swapped identities as well as duplicate owners. Update only this route's call-local root. |
| Typed scalar handlers | Several error after pop without releasing a wrong-tag heap operand. Reject through preflight while all owners remain rooted; preserve original scalar error and missing-value semantics. |
| RET/ASSERT and unexpected traps | Real core may return for supervision. Keep the same owned preparation live, release transferred condition once, and retain first failure rather than invoking generic callback/effect handling. |

Packed conversion deserves a separate exact acceptance control before any parity
claim. heap.c packed_store supports INT→U8 wrapping and INT→FLOAT conversion;
that runtime fact does not itself authorize either in every proved record field.
The current ordinary record-array proof has an explicit converted_children
exception for exact bound nested U8 children. Its reviewed unchanged U8 source
uses300 and produces44. In contrast current vm_rmx_preflight and private DAG
array push require exact input element tags. Copying those checks would reject
that ordinary accepted composition. The borrowed preflight must be reviewed
against a source-derived exact nested-U8 map/record composition and the existing
proof's required write mask; no generic numeric whitelist or FLOAT field widening
is implied. First unchanged map anchors can qualify independently, but later
public selection cannot silently claim full array parity without this gate.

## Small implementation and qualification checkpoints

1. **Ordinary heap adapters only.** Separate typed ordinary header/set/get/has/len
   wrappers, retaining private interfaces and both heap layouts. Direct tests use
   actual valid buffered/unbuffered maps after full import/topology validation,
   counted-NUL keys, independently allocated equal-content strings, collisions,
   replacement at MAX, insertion MAX before growth, partial construction and all
   allocation prefixes. Check exact net-growth statistics and ordinary GC cleanup.
   No shared handler, core selector or execution entry is enabled at this checkpoint.
2. **Private borrowed activation and shared dispatch.** Freeze source for review:
   new call-local view/root supervisor, real reserved stack activation, typed
   six-argument core and preserved wrappers, separate ordinary preflight/HM path.
   No public candidate routing. First bounded gate runs all four unchanged original
   map inputs through this private entry and verifies actual results/assertions,
   stack-result lifetime and ordinary heap cleanup. Stop on the first target,
   conversion, context, allocation or ownership discrepancy before broader tests.
3. **Complete private invocation qualification.** Run switch/computed-goto with all
   four source-global inputs too. Test consumed and unconsumed repeated results,
   initializer scalar residue, initializer/entry root tails followed by ASSERT,
   nested calls, distinct and duplicate >16 arguments, frame cap, nonidentity
   ordinals, field replacement and retained external aliases, valid buffered
   inputs, wrong recipe/context/parameter/cache refusal, missing GET and actual
   false ASSERT/bounds failures. First-error selected unwind leaves stack/frames
   empty and globals/alias mutations intact; subsequent invocation and ordinary
   vm_destroy/GC balance allocations. Include representative primitive/nested
   field compositions and the exact accepted U8 conversion before broader parity.

All initial preparation allocations, stack realloc and executed heap/argument
allocations need independent transient/persistent prefixes with recovery. A failed
stack realloc preserves original stack identity/capacity; later errors may retain
successfully grown capacity. Observe source/caches/preparation identity through
assertion resumptions and every cleanup path. Failure tests require exact statuses
and sanitizer-clean stderr, not arbitrary nonzero exits. Preserve the old required
private runtime/import/observation/array tests, ordinary NULL routes, public refusal
and all six old allocation/work budgets. No VmState/VmHeap member or legacy budget
changes are proposed. No new frontend build/bootstrap is necessary for this slice.

Public selector/load/native/publisher integration remains a later explicit review,
even if this private entry passes. This document authorizes no production change;
implementation begins only after the interface and checkpoint scope are approved.

The reviewed checkpoint1 refinement gives all five ordinary adapters a heap
argument (const for reads), requiring !gc_running and !private_record_dag.
Managed map/key ownership by that heap remains a C precondition, not a registry
lookup. Existing private fast adapters retain their exact source and policy.
