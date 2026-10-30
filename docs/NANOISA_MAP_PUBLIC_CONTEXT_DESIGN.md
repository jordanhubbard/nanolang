# I propose a borrowed-state map invocation context

Design-only #1001 checkpoint based on aggregate e5164fe74. This refines
NANOISA_MAP_PUBLIC_INVOCATION_DESIGN.md after the boundary/import components. No
source, admission, runtime, native, loader or publisher change is authorized here.
The four original modules and four source-global baselines retain their exact
producer receipts; the standalone import query is not execution authority.

## Concrete existing paths

`vm_init` borrows the original module and owns ordinary stack/globals/constants,
decoded functions and dispatch tables. It sizes globals from decoded references;
LOAD_GLOBAL reads global_capacity, whereas STORE_GLOBAL grows capacity and advances
count. The module and managed heap storage must remain live and exclusively
accessible throughout an invocation. No concurrent host mutation is supported.

`vm_execute_record_arrays` prepares then frees its plan, creates an ordinary-array
proof, calls the initializer through `vm_call_function_scoped`, drains initializer
results, then calls entry. Selected scoped failure releases new stack/frame owners,
restores activation floor and prunes effects; globals are preserved. It does not
apply the separate typed-owned global-clear path. Ordinary generic vm_execute's
retained assertion failure stack3/frame1 is comparison evidence, not this selected
cleanup contract. Entry results remain on the public stack; vm_get_result borrows
rather than consumes. The caller must consume/release the result before another
selected invocation. Retained host aliases remain valid across initializer root
replacement and runtime failure; mutable aliased payloads are not rolled back.

The private map supervisor instead owns VmState, an image and a persistent result.
Its context requires `vm == &owner->vm`, image module identity, fixed stack capacity
and private_record_dag. Its record handlers store layout IDs; ordinary record
handlers store record ordinals. Its map adapters reject buffered maps. Those are
separate contracts: relaxing these checks is not the public integration interface.

## Proposed internal APIs and lifetime

The following names are proposed, not exported APIs or implemented declarations:

```c
typedef struct VmRecordMapPublicPreparation VmRecordMapPublicPreparation;
typedef struct VmRecordMapPublicView VmRecordMapPublicView;
static NvmArrayEligibilityResult vm_rmp_prepare(
    const VmState *, VmRecordMapPublicPreparation **out);
static void vm_rmp_free(VmRecordMapPublicPreparation *);
static VmRmImportStatus vm_rmp_import(
    const VmRecordMapPublicPreparation *, const VmState *);
static VmResult vm_rmp_preflight(
    VmRecordMapPublicView *, const VmDispatchInstruction *);
static VmResult vm_rmp_call_root(VmRecordMapPublicView *, uint32_t function);
static void vm_rmp_unwind(VmRecordMapPublicView *);
static VmResult vm_execute_record_maps(VmState *);
```

Preparation owns exactly one fresh NvmRecordMapBoundaryPlan and its immutable
execution-plan borrow, plus copied function/recipe/layout/field tables. It contains
no VmState, VmHeap or result/global owner. All allocations are owned explicitly;
old structs and NULL paths remain unchanged. Preparation validates complete
original module correspondence and every decoded/dispatch row, not only reached
instructions; cached call handles/superinstructions must match the supported
unfused closed route. Constants must have exact count, tag, counted length, hash,
terminator and bytes matching the original module, not merely non-NULL pointers.

The view borrows preparation, caller VmState and original root module. It holds
active_root, root_base and active-state identity only for this invocation. It is
never installed in VmState, exposed to callers, returned as a token or reused.
Core integration uses a separate typed ordinary-map context, mutually exclusive
with existing array/private/owned contexts; old scoped wrappers pass NULL.

The static import helper invokes the same reviewed import walker with this owned
boundary plan. It does not call the module-taking observation query and construct
a second proof. The complete capacity scan, collector membership, singleton
assignment, historical fields/children and counted-map topology checks occur
before any guest instruction. The table is transient, not an origin registry.
No host callback, extern, module link, effect, coroutine or tracing boundary can
intervene between validation and execution. ASSERT supervision is internal and
resumes this same active view. A fresh invocation repeats proof/import validation.
No second import scan is imposed between initializer and entry: closed guest
transitions are justified by the same whole-program proof, not by pretending that
newly produced objects recover unique historical allocation provenance.

Preparation/query refusal frees only preparation temporaries; caller roots,
collector metadata and prior output remain untouched. The eventual execution
adapter may record its error in VmState as other public entrypoints do. A successful
stack reserve may change stack address/capacity; it never consumes existing owners.
Runtime failure has selected-route unwind semantics, not bytewise VM rollback.

## Required shared-handler preflight changes

1. **Context and caches.** Reject linked/callback/effect/reference/trace/reentrant
   state and GC-in-progress. Keep ordinary heap policy false. Match frame module,
   signature, local count, PC, instruction operands/widths, successor indices,
   recipe, operation and stack effects against retained immutable rows before
   pops/writes. Neither vm->verified nor a previous import report is authority.
   Normal vm_init starts with fusion off; explicit incompatible cache profiles
   refuse until separately qualified. Per-instruction checks retain CALL/TAIL/RET
   correspondence after each ASSERT resume.
2. **Globals.** LOAD checks actual capacity and retain headroom. STORE checks its
   proved index, ordinary storage capacity/growth and existing owner before pop;
   it uses the actual prior slot and advances count only at ordinary commit.
   It never resets globals or invents VOID for allocated storage above count.
3. **Nominal records.** Physical def_idx is an ordinal. Query descriptor(ordinal),
   require its layout equal the proved layout and use that layout for field
   obligations. New STRUCT/AGG_RECORD stores keep the original ordinal. Existing
   private ordinal-to-layout rewrite remains private-only. Nonidentity ordinal0/
   layout1 is a required executed fixture, not merely a query positive.
4. **Ordinary container headers.** Valid buffered PURPLE and unbuffered PURPLE
   headers are allowed as established by import validation and ordinary heap
   transitions. Do not force collection, clear flags or enable private DAG policy.
   GC-only entries carry membership, not static origins. Closed valid operations
   preserve topology; runtime boundaries check selected shell/field obligations
   without uncharged repeated whole-graph scans.
5. **Map operations.** NEW/SET/GET/HAS/LEN only. Add distinct typed ordinary
   checked adapters or static shared implementation; no exported header-skip
   boolean. Full topology is established at import and preserved by NEW/SET.
   SET replacement retains no key; insertion checks headroom before fallible
   growth, retains the key before publishing, and has no subsequent fallible
   step. Ordinary buffering/refcount release behavior is preserved. GET keeps
   INT|VOID, including missing keys after HAS; no presence inference from spelling.
   Popped operands are released exactly once on failure; returned receiver
   ownership transfers to stack on successful SET.
6. **Retains and calls.** Check STRING/ARRAY/RECORD/MAP owners, not only old array
   retain kinds. CALL transfers existing argument stack ownership. TAIL stages
   all arguments before releasing the old frame, counting duplicate physical
   owners cumulatively, including >16 argument allocation failure. Root TAIL
   changes the view's active_root; each initializer/entry establishes its own
   root_base. ARR_SLICE counts repeated child retains; field loads and constants
   check the actual retained value. Never check a record shell when the operation
   retains its field instead.
7. **Scalar traps.** Reuse exact scalar-input checks before handlers that pop a
   wrong-tag heap operand without releasing it. Preserve ordinary polymorphic
   truthiness/comparison and existing numeric error order. Map absence cannot be
   narrowed to INT to make a later scalar operator appear safe.

## Whole-invocation stack reserve

Several real handlers ignore stack_push failure. Private map execution avoids
this through complete preallocation; a public adapter cannot rely on optimistic
per-function capacity or dynamic push failures after mutation.

Retained proof bounds each activation's locals and temporary operand stack by256
each, with at mostVM_MAX_FRAMES activations. Direct calls retain caller frames;
tail calls replace a frame. A conservative simultaneous bound is therefore
`VM_MAX_FRAMES * (256 + 256) + 1` NanoValues, with checked arithmetic. The extra
slot covers a root result. Initializer residue is drained before entry, so their
root maxima are alternatives, not two accumulated frame stacks. The design does
not sum every function's maximum or assume tail recursion adds frames.

Before guest execution, ordinary stack_reserve must reserve this bound (or a
separately proven tighter bound), accepting capacity >= required rather than
private fixed-capacity equality. The actual existing doubling/rounding algorithm
must be reflected in preparation accounting. Charge requested new capacity plus
live-old/new overlap; caller-retained storage is separately identified. Failed
allocation preserves old stack storage and roots. Successful reserve may replace
storage, but all subsequent scoped pushes fit, including CALL local initialization
and tail replacement. Callback/reentry remains refused. This reservation is a
host preparation limit, not new guest work/fuel semantics.

## Supervision and cleanup

Require empty stack/frames and ordinary idle state before preparation. Run
initializer first when present; reset active_root/root_base independently for
initializer and entry and release initializer residue before entry. Root ASSERT
returns a condition whose owner is released once; false sets the exact assertion
failure, true resumes the same view. No user code runs in that supervision loop.

On error, release invocation stack roots and outstanding frame callable owners,
restore frame/floor state and prune scoped effects as the selected-array route
does. Preserve first error and diagnostic location; do not reset globals, destroy
the borrowed VM, fabricate a private result or roll back aliased mutations. The
view is invalidated before freeing preparation and returning idle control. Host
aliases and globals remain ordinary GC roots; vm_destroy uses its ordinary path.
On success, entry's declared result remains on the public stack. There is no
persistent private result transaction or automatic consumption on the next call.

## Review prerequisites and staged qualification

Exact owner sizeof, copied row capacities, zero-count allocation behavior, full
boundary reserve overlap, immutable table scan charges, constant/cached-decode
validation schedule and rounded stack realloc ledger are still to be finalized
before implementation. Existing64-origin/256-function/depth/byte/work caps remain
unchanged. This document does not authorize a public selector change.

First prepare/context-only gates must preserve old NULL APIs/budgets and output
atomicity. A subsequent private module-taking borrowed-state test entry can then
qualify actual shared dispatch before public selection: all four originals and
four source-global inputs, repeated initializer/result ownership, host mutation
and singleton refusal, nonidentity nominal mapping, buffered aliases, counted-NUL
keys, CALL/TAIL (>16 and duplicate owners), initializer/root-tail ASSERT resume,
missing GET/falseassert/bounds-first-error, allocation prefixes and ordinary GC.
Full caller objects may be reused only under fresh validation; no successful
observation result can bypass it. Public verifier/load/save/native/publisher
selection remains jointly gated in a later reviewed phase.

## Numeric preparation ledger proposed for independent review

The original module, decoded/dispatch caches, constant objects and all caller heap
storage are exclusively stable for the entire invocation. Managed allocation
extent and ownership remain C API preconditions. Correspondence checks supplement
these preconditions; they do not make concurrent mutation or forged pointers safe.
The first executable test entry remains private and module-taking, constructing
its own preparation; public vm_execute selection stays closed.

I propose this exact preparation layout, with 64-bit-pointer ABI sizes derived
from current declarations. These are design values to assert and measure on the
first authorized build, not claimed new compiled measurements:

```c
struct VmRecordMapPublicPreparation {
    NvmRecordMapBoundaryPlan *boundary;             /* owns */
    const NvmRecordMapExecutionPlan *plan;          /* boundary-owned borrow */
    const NvmModule *module;                       /* caller-owned borrow */
    const NanoValue *original_stack;               /* pre-reserve identity */
    NvmRecordMapExecutionInstruction *instructions; /* owns I rows */
    VmRecordMapRuntimeField *fields;               /* owns E rows */
    NvmRecordMapExecutionCounts counts;
    NvmRecordArrayExecutionFunction functions[256];
    VmRecordMapRuntimeLayout layouts[256];
    uint32_t record_layouts[256];
    uint32_t layout_count, field_count;
    uint32_t original_stack_capacity, rounded_stack_capacity;
    uint64_t bytes, work;
};
```

Six pointers48, full counts96, functions256*36, layouts256*16, ordinal table256*4,
four uint32 fields16 and two uint64 counters16 yield **14512 bytes**. Instructions
are80 bytes per row; runtime fields20 per row. A separate stack-local view with
three pointers, two uint32 root coordinates and active bool is40 bytes. No VmState
or VmHeap member changes. Owners/tables publish only after complete validation;
zero table counts allocate nothing, never an undocumented one-element minimum.

Let B/W be the boundary owner's complete reported reservation, I total instructions,
E total declared fields, F functions, L layouts, A array bindings, P parameter-tag
slots, S strings, C=sum(function code lengths+1), and Z=sum(string lengths).
Counts, multiplication and sums are checked before loops or allocations. Retain
full B/W, including its proof/snapshot overlap; never replace it by retained bytes
or refund its analysis reservation.

Preparation adds these positive reservations:

| Item | Bytes | Logical work before work occurs |
| --- | ---: | ---: |
| Preparation allocation |14512|14512 zeroed bytes|
| Instruction allocation/copy |80I|80I zeroed bytes +4I getter probes|
| Field allocation |20E|20E zeroed bytes|
| Stack-local view |40|40 initialized bytes|
| Explicit preparation scratch |1812|1812 initialized/reserved payload bytes|
| Function and parameter copy |0|32F+4P|
| Existing physical layout/rank copy |0|24L+40E+64A, plus1280 DFS initialization already in scratch|
| Complete opcode/decoded/dispatch correspondence |0|256+128I+32F, matching existing private coverage policy|
| Raw code/offset-index bytes |0|10C|
| Existing constant validation |0|32S+2Z|
| Planned ordinary stack growth |old_bytes+new_bytes if growth, otherwise0|old_bytes+64 if growth|

The1812-byte scratch enumerates a256-byte comparison buffer, existing1280-byte
layout DFS storage, execution counts96, boundary counts24, declaration counts20,
header32, expected recipe72, descriptor12, array type8 and array binding12.
This is logical payload accounting, not a claim about compiler stack padding.
No decoder, dispatch cache or intern table is newly constructed; these existing
caller-owned objects are inspected. Accordingly there is no new intern resize
schedule to infer or duplicate. Complete cached decoded rows are charged by the
existing128I logical comparison policy; code/offset bytes additionally precharge
10C (raw code, boundary byte and two uint32 index maps, including endpoint).
Constant validation precharges its complete32+2*length schedule per string before
hash and byte comparison, then checks terminator/header/count/identity. Counted
NUL remains valid for VM constants. A bounded256-byte snapshot buffer avoids any
length-sized temporary allocation.

The exact same import walker then starts from B/W plus all preparation additions,
using its approved positive-only reserve seam. Its4568-byte copied-row/state scratch,
16-byte identity buckets (initial16, doubling at half occupancy), old/new table
allocation overlap, hash probes, collector entries, global-capacity visits, selected
fields/array elements and map byte/probe traversal add to those same counters.
It does not reset to the standalone boundary budget, silently use another work
limit, or run a second proof. On success the preparation records the final shared
counters; on failure all preparation/boundary/table allocations are freed. The
standalone observation query continues supplying0/0 and keeps its current schedule.

For stack growth, R=1024*512+1=524289 values. If the current valid stack capacity
already covers R, no new stack allocation is planned. Otherwise reproduce exactly
stack_reserve's checked doubling starting at the existing capacity (4096 for normal
vm_init); use R when rounded capacity is unrepresentable. With the normal capacity,
the rounded target is1048576 values: new_bytes16777216 and old_bytes65536. Both
are charged before the post-preparation allocator, even though old storage is
caller-owned and realloc might grow in place. The64 logical overhead covers the
bounded doubling loop and allocation commit; possible copied old payload is charged
by old_bytes. Actual stack_reserve uses realloc and does not initialize the newly
reserved capacity, so new_bytes is a live-allocation charge, not a zeroing-work
charge. This is a bounded logical allocator-call policy, not a bound on allocator
internals. Later guest local initialization belongs to execution, not preparation.
For the normal capacity the stack work addition is65600, not16842816. The retained
import example16864804 plus this addition is16930404 before the remaining new
preparation work; this is arithmetic feasibility only, not an actual constructor
qualification. Overflow, cap excess or impossible size refuses before allocation.

A nonzero stack capacity with NULL backing refuses as malformed entry; no
reserve operation is used to manufacture backing legitimacy. A zero-capacity
NULL stack follows the existing initial-capacity path if all other owner
preconditions hold. Prepare remains const and allocates no caller stack. After complete preparation
and import, a separate activation step rechecks unchanged stack pointer/capacity
and requests the already-reserved growth. A failed realloc preserves prior caller
storage/owners; a successful realloc changes only storage/capacity before root
activation. No guest work has occurred. Runtime allocation failures later use
selected-route unwind, preserve globals and ordinary collector semantics, and
never refund preparation costs or reinterpret them as guest instruction fuel.

### Initial boundary reservation interface refinement

Before proof allocations, the fixed14512 owner +40 view +1812 preparation scratch
and the checked planned stack old/new reserve must already be charged. I propose
an internal positive-only reserved wrapper around the existing rmb_prepare_reserved
in its owning translation unit. It forwards only additional nonnegative byte/work
reservations and returns the same boundary owner; no trusted execution token or
credit/refund is introduced. The existing module-taking boundary constructor keeps
0/0 and its exact old schedule. The private owning-TU declaration is:

```c
NvmArrayEligibilityResult nvm_prepare_record_map_boundary_reserved_private(
    const NvmModule *module, NvmRecordMapBoundaryPlan **out,
    uint64_t extra_bytes, uint64_t extra_work);
```

It validates checked additions and unchanged caps before proof allocations. Only
the private VM preparation constructor needs this construction API; no borrowed
plan execution entry is exported. The old public constructor remains the0/0
wrapper. Source implementation awaits preparation-slice authorization.

The returned B/W then already includes these fixed additions: do not add them a
second time. Dynamic80I instruction and20E field allocations and exact copy/cache/
constant scan charges are checked after complete count discovery and before their
allocations/scans. The same import walker starts from augmented B/W plus only the
remaining dynamic additions, then adds its own scratch/table/traversal charges.
If dynamic totals exceed the cap, the earlier boundary proof allocations are
reversibly freed; output and caller state stay unchanged. No claim is made that
all LIMIT failures occur before any boundary allocation. This preserves both the
full initial proof overlap and a single successful boundary analysis.


### Charge order and count discovery

The initial reservation also includes64 logical header/count probes. This is a
fixed conservative query policy, not a claim of64 machine instructions. It pays
for bounded scalar header/count getters and validating their ceilings before any
count-driven scan. Fixed owner/view/scratch initialization and planned stack
reserve are included in the initial B/W, even if their allocations occur later.
They are never added again. Positive dynamic charges accumulate as follows:

1. Validate bounded F/I/L/A/S counts with the prepaid header probes. Check all
   arithmetic and representable allocation sizes before allocation or traversal.
   Precharge80I bytes and80I+4I work before instruction allocation/getter visits.
2. Precharge32F before function-row traversal. During each charged row visit,
   validate arity/local/code bounds, then precharge4*arity before parameter visits.
   P is accumulated with checked addition during these visits; no separate unpaid
   parameter-count pass is performed.
3. Precharge24L before layout visits. Accumulate E from checked per-layout field
   counts within those visits, then reserve20E bytes and20E zeroing work before
   allocating the field table. Precharge40E and64A before respective field and
   binding scans. The1280-byte DFS initialization is already prepaid scratch.
4. Precharge256+128I+32F before complete coverage/cache correspondence. During
   each charged function-row visit, check code length and precharge10*(length+1)
   before scanning raw code and offset/index bytes. C is a ledger sum, not an
   additional uncharged discovery scan.
5. Precharge32S before constant-header visits. At each charged header, validate
   length/pointer preconditions and precharge2*length before hashing/comparing
   bytes. Terminator/header checks fit the32-per-row charge. Z likewise requires
   no separate uncharged discovery scan.
6. Pass only accumulated dynamic additions to the import reserve seam atop the
   already augmented boundary B/W. The existing walker adds its own scratch and
   every table/probe/edge/byte charge before the corresponding work. No initial
   fixed reserve is repeated and no successful proof reserve is refunded.

Each LIMIT/MEMORY/INVALID exit frees temporary owners and preserves the caller VM
and output pointer. Final measured reservations and independent allocation-oracle
coverage remain required; these formulas alone do not qualify any original input.
