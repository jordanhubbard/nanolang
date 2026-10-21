# I propose invocation-scoped public map execution

This is a design-only #1001 checkpoint on `design/5.1-map-public-invocation`, based
on aggregate ecbf0c0ec. No admission, runtime, proof, native, loader or publisher
change is authorized by this document. It refines NANOISA_MAP_CONSUMER_DESIGN.md
following the private consumer qualification. Numeric storage/work schedules and
actual repeated-source baselines remain review prerequisites.

## Existing contracts and exact call chain

- `vm_init` (vm.c:522) borrows the original module/root and owns constants,
  decoded instructions, dispatch tables, stack, globals and heap. Original module
  lifetime remains the caller's responsibility; this is not a copied private owner.
- `vm_execute` (5662) currently selects `vm_execute_record_arrays` first. That
  function (5633) requires an idle root, empty stack, no references/callbacks/links/
  tracing and ready constants, admits then frees the array plan, and calls the
  initializer and entry through `vm_call_function_scoped` (5348).
- `vm_call_function_scoped` saves the base/frame/floor, calls
  `vm_call_function_impl`, and unwinds new stack/frame owners on error. The latter
  resumes the scoped core after ASSERT; root TAIL updates active-root identity.
  Its ordinary-array proof is not map authority.
- Initializer stack residue is explicitly released before entry on the public
  array route. Entry's result remains on the public stack. `vm_get_result` (1403)
  merely borrows/peeks; it neither removes nor retains the result. A repeated
  selected invocation requires the caller to consume/release the prior stack
  owner first. There is no private persistent result slot or prior-result publish
  transaction in public VmState.
- Globals survive success and failure; the separate typed-owned global clearing
  branch does not apply to ordinary arrays. `vm_destroy` (575) releases frames,
  globals, stack and constants and destroys the ordinary cycle-aware heap.
- Public array entry does not validate preexisting global heap graphs. LOAD_GLOBAL
  (2475) checks capacity and retains; selected retain preflight (1552) checks
  headroom. Neither establishes map probing topology or imported nominal shape.
  This is an observed existing boundary, not permission to reuse it for maps.

## What the proof actually knows

`managed_array_shapes.c` initializes historical globals to VOID at 965. Each
STORE_GLOBAL merges into that monotone historical vector (653), while assigning
its current path value and invalidating old refinements. On every fixed-point
pass, initializer is seeded from the historical vector; entry receives its
returned exit vector, or the historical vector directly without initializer
(974–979). CALL/TAIL seed current path globals and merge returned summaries only
for transitive-written slots (718). TYPE_CHECK refines current snapshots; it is
not a permanent assertion about a slot.

Consequently, the proof includes values previously stored by this module, not
only values established during the current invocation. It does not include
arbitrary host-injected heap shapes. The present frozen execution API exposes
only global count, not this per-slot historical boundary. Runtime pointers must
never be converted to static allocation-site origin IDs.

## Proposed immutable boundary facts and ownership

I propose a **separate opt-in owner**, not extra fields in old Analysis, Value,
VmState, VmHeap, frozen-map or execution-plan structures:

```
MapPublicPreparation owns existing map plan + boundary global rows + runtime tables
MapPublicInvocation borrows preparation + caller VmState + original module
```

The public preparation path asks the existing single successful map analysis to
copy its final historical global summaries in the same transaction. Old query/
plan/image wrappers pass NULL for this optional output and retain exact sizes,
allocation schedules and bytes. No second incomplete analysis or caller-supplied
trusted plan is accepted. The new owner frees every detached output on failure.

Each global row reuses existing `NvmRecordValueOrigins` (origins, tags, unknown),
with `sizeof==16` asserted for this supported ABI. Zero-initialize the copied row,
strip Analysis Value relation/relation_tag and never copy mutable Value storage.
There are at most256 slots (at most4096 row bytes); zero slots allocate no rows.
Copy validates count, tag domain, origin range/kind and unknown status before
publication. Existing final_check already rejects historical global unknown
before freeze; unknown==0 is a checked invariant, not a newly admitted state.
VOID is allowed exactly when the historical tag set contains VOID, including
fresh globals. Standalone HASHMAP globals remain refused by existing map_final.
The separate new wrapper and actual row capacities are charged; existing Analysis
remains39840 and existing six budget receipts must remain byte-identical.

The owner also owns or borrows-for-its-own-lifetime the existing immutable exact
origin/field/array obligations and map pair rows. No Value, mutable Analysis or
source-global pointer survives. A private copy getter costs one checked indexed
row access under an explicit caller charge; it preserves output on miss and has
no hidden mutable budget. The exact helper signatures and owning TU placement
must be reviewed before code. Old copied-image APIs need no new global rows.

New allocations are map-public-only, charged against unchanged execution limits
with retained proof/plan plus table/scratch overlap. Row copy charges count before
scanning and checked count*16 before allocation. Header sizeof, actual allocated
capacities and alignment need a measured ledger; 4096 is row payload, not a total
constructor estimate. Existing private preparation must retain its current cost.

## Per-invocation boundary validation

Each invocation reproves the original module, validates original decode/dispatch/
constant correspondence against its owned snapshots, and validates **every current
global** before a guest instruction or irreversible state mutation. Root/module
identity, owner thread, idle frames/stack/references/handlers, no callbacks/links/
trace and coherent counts/capacities are mandatory. Constants need exact counted
bytes, not merely the existing non-NULL readiness test. No proof survives return.

The boundary graph consists of managed objects referenced by globals. Ordinary
VmState's managed-pointer/ownership API precondition remains; a C client forging
arbitrary invalid addresses cannot be made safe by dereferencing a tag. Within
valid managed storage, corrupted shape/topology is a controlled refusal.

### Conservative singleton-obligation proposal (not yet approved)

Current structural shape alone is insufficient: host-created aliases can combine
origins with different future all-write obligations. I withdraw the earlier
structural-candidate-intersection solver. I propose this narrower import policy
for baseline/review, not an assertion of parity for all previously accepted graphs.

For each root or child occurrence, require its actual tag in the historical row.
Scalar/VOID values need no origin. For a heap value, select only origins of that
physical kind from the row and require **exactly one** before structural matching;
zero or ambiguous alternatives refuse. This singleton is a proof obligation used
by boundary validation, not a discovered allocation provenance. Do not choose an
origin merely because its current shape happens to match.

An invocation-local identity table records `(physical identity, singleton
obligation, visiting/complete)`. On another occurrence of the same object, require
the SAME singleton obligation and all exact occurrence descriptors; distinct
origin requirements refuse even when current shapes match. Distinct physical
objects may satisfy the same singleton (repeated allocations at one site). A
visiting revisit refuses a cycle; a completed revisit may reuse the same verified
obligation only after charging its lookup and checking the occurrence descriptor.
No union/intersection of disjoint origin IDs is called structural compatibility.
The table does not escape, grant runtime allocation identity or change the heap.

Validate records against exact nominal layout, field count, declaration and the
complete per-origin historical field summaries, recursing through each field.
`write_record` merges every constructor/replacement write into these summaries.
Validate arrays against exact declared packed/boxed representation, leaf/depth,
required_elements, and complete child summaries, recursing into actual children.
`write_array` merges every boxed write into child tags/origins; packed arrays retain
exact declared runtime tags and existing reviewed packed-conversion semantics.
Thus an imported object obeys the same future-read/write abstraction used for its
one obligation. A host alias that would introduce a new cross-origin connection
is refused instead of silently weakening the analysis. Existing proven source
aliases whose occurrence sets are ambiguous may also refuse this conservative
first boundary; the baseline must expose that limitation before implementation.

Maps require exact STRING→INT pair and full occupied-entry/count/hash/probe/unique
key topology validation. Keys are immutable counted strings; duplicated interned
keys across maps remain allowed and do not acquire allocation-site obligations.
Missing GET stays VOID. Strings require valid ownership, counted length, checked
allocation-size arithmetic and hash/byte consistency; embedded NUL is legal here.
Expected occurrence field obligations remain conjunctive, including nominal child
and exact array depth/leaf; they cannot be waived by seeing an identity before.

Nominal record rank and array depth bound a traversal path; ordinary imported cycles
refuse selected map execution while ordinary GC remains responsible for cleanup.
The 64 static-site bound is not a 64 physical-object bound. The identity table must
grow within a checked, charged preparation budget. Charge every root, field,
element, byte, identity-table probe and map topology comparison before reading it,
plus old/new table overlap before growth. Its concrete capacity/load-factor and
numeric ledger remain review work. No guest fuel or silent runtime element cap is
introduced; repeated invocation may explicitly hit this boundary's preparation
limit after an otherwise successful guest run.

Broader multi-origin imported graphs would need independently reviewed alias-effect
closure/re-analysis that propagates future writes across introduced aliases. It is
not authorized by this singleton proposal or by physical shape matching.

## Borrowed-state execution and cleanup

After preparation/validation, create a call-local view containing immutable table
borrow, VmState pointer, original module identity, active-root/base and active flag.
Do not reuse `VmRecordMapPrivate` with a fake embedded VmState. Refactor shared
checked preflight access behind immutable table accessors, keeping old array and
private-map ownership assertions separate. Public capacity is dynamic; check and
reserve actual frame/local/maximum-stack requirements, not private fixed capacity.

Keep ordinary heap release and cycle collection; never set private_record_dag on
this externally accessible heap. Checked map adapters need a separately reviewed
internal build surface independent of the private teardown macro. They establish
new-map topology and preserve validated topology through exact insertion and
replacement. Validate retained global graphs afresh next invocation; callbacks,
concurrent mutation, linked calls, external direct/core/helper entry and reentry
are refused. No persistent trust flag or registered VmState pointer is introduced.

Initializer and entry independently reset active root/base. Drain initializer
residue before entry. Successful ASSERT resumes the same view; CALL/TAIL preserve
actual recipe/frame/stack correspondence, including root-tail identity and >16
stable argument staging. Retain checks cover cumulative duplicate references,
map insertion keys, loaded globals/fields and replacement order before pops or
publication. No blanket proof of refcount headroom follows from type checking.

On preparation/boundary failure, preserve stack/global values, identities and
refcounts; permit the normal error status/message update, with temporary allocation
balance. On guest failure, invalidate the view before exposing idle state; unwind
only the invocation's frames/stack owners and keep first error. Globals and heap
mutations already performed remain, including mutations visible through an alias
held by the host. No deep transaction is promised. On success, leave the real
entry result on the public stack. Preparation tables are freed only after cleanup;
results/globals depend on ordinary heap ownership, not the table owner.

## Required baselines and proposed controls

The [ordinary baseline](evidence/map-public-invocation-baseline-20261021/README.md)
now runs all four unchanged `map_record_fields.nano` modules twice in one public
VmState with explicit result consumption (84 checks). The existing unannotated
private alias-failure byte fixture succeeds once then fails ASSERT with global
identity/value2 preserved. Its ordinary failure leaves stack3/frame1 until destroy;
selected immediate unwind would deliberately follow the selected-array contract.

The new minimal source-global acceptance probe is accepted/verified by all four
retained producers and executes twice (53 checks each). Initializer replaces the
global root on each call; an explicitly retained prior record/map alias survives
until released. No singleton boundary or new authority was used in these tests.

Qualification must include matching distinct objects, repeated shared global/field
aliases, wrong nominal layout/map pair/array depth/child tags, standalone map global,
unknown summaries, VOID allowed/disallowed, corrupt map bucket topology/duplicate
keys/counts/hash, counted-NUL keys, cycles, retained-counter saturation, stale
constants/decode/module bytes, and initializer failure before entry. Imported
matching objects are accepted only if the reviewed structural boundary rule proves
them; arbitrary pointer provenance is never asserted. Every refusal preserves
preexisting owners, including cyclic objects later destroyed by ordinary GC.
Test failure after mutation, repeated entry after host edits, source lifetime,
all allocation prefixes, both dispatches and exact old NULL-route budgets.

Public selection precedence and loader/verifier changes remain a separate reviewed
step. Malformed/present map metadata must never fall back to arrays/legacy, but
this contract itself changes no selector. Public native and publishers stay closed.

## Proposed concrete optional interfaces and bounded validation schedule

These are internal declarations for review, not new public caller authority:

```c
/* Owning proof TU: new opt-in constructor, original module only. */
typedef struct NvmRecordMapBoundaryPlan NvmRecordMapBoundaryPlan;
NvmArrayEligibilityResult nvm_prepare_record_map_boundary(
    const NvmModule *, NvmRecordMapBoundaryPlan **out);
void nvm_record_map_boundary_free(NvmRecordMapBoundaryPlan *);
bool nvm_record_map_boundary_global(const NvmRecordMapBoundaryPlan *,
    uint32_t slot, NvmRecordValueOrigins *out);
/* Internal immutable borrow; never an executable user-supplied token. */
const NvmRecordMapExecutionPlan *nvm_record_map_boundary_execution(
    const NvmRecordMapBoundaryPlan *);
```

Proposed owner fields are plan pointer, global-row pointer, u32 count/reserved,
u64 charged bytes/work (40 bytes on this ABI, to assert/measure). Existing map
constructor remains a wrapper over a common helper with boundary output NULL;
only the new path requests the row copy during the same successful analysis.
No map plan/frozen/image size changes. Publish the new owner only after complete
plan/snapshot/recipe construction and row consistency; every output remains intact
on failure. The new constructor owns the plan exactly once. Getter misses preserve
output; a valid indexed global getter is precharged four logical units by its
caller. Work is a named conservative schedule, not machine instruction counting.

The common helper must reserve the extra40+16G live bytes before its owning
allocations and the optional copy work before row visits. Full analysis reservation
remains64MiB/16777216 work inside unchanged128MiB/33554432 preparation caps. It
cannot refund this reserve just because the particular analysis used less. Charge
owner/row zeroing by actual allocated byte count plus8G for validation/copy. Existing
plan retained-cost helpers give exact live payload before boundary validation; no
second128MiB blind reservation. Internal analyze accepts optional detached rows
alongside frozen output, and frees both on any later constructor failure.

The invocation need not duplicate the private owner's 256-function arrays. Query
immutable instruction/function/descriptor rows into bounded stack copies through
an internal context adapter. The private path can keep its existing direct-table
accessor; public path reads the retained boundary plan. Its proposed call-local
view contains three pointers (VmState, boundary plan, original module), copied
NvmRecordArrayExecutionCounts, u32 root/base and active flag. Charge actual sizeof
(view) and queried row scratch before use; no view pointer is stored in VmState.

For the singleton import validator I propose:

- Identity row `{const void *object; uint32_t obligation; uint8_t color;
  uint8_t reserved[3];}`:16 bytes on this ABI. Empty pointer marks unused; no
  tombstones. Start16 buckets; grow by2 before insertion would exceed1/2 load.
  Linear probing uses a hash of uintptr_t solely for runtime identity. Bound each
  probe by capacity and charge2 logical units before inspecting its bucket.
- Each growth checks doubling/size overflow, reserves **both** old and new table
  payload while rehashing, and charges new zeroed bytes plus old capacity before
  scanning, then2 per reinsert probe. Failed growth keeps old table/state intact.
  Maximum physical objects is derived from remaining preparation bytes/work,
  never assumed64. Exceeding budget returns LIMIT before uncharged traversal.
- DFS frame `{const void *object; uint32_t obligation,next_child;}`:16 bytes.
  A fixed259-frame scratch (4144 bytes) is reserved/charged before traversal;
  bounds checks refuse overflow. Acyclic record rank<=256 plus array depth<=3
  provides the existing structural depth ceiling, not an assertion that every
 259-node graph is constructible. Actual maximum fixture must be measured.
- Before header/global scan charge32 plus8 per live global slot. Check
  global_count<=capacity<=VM_MAX_GLOBALS and valid storage. Slots outside copied
  global count must be VOID; no non-VOID external root is silently ignored.
  Charge64 before each static-origin-kind filter, four per copied summary getter,
  and eight before each physical field/element visit. Record/array header checks
  receive16 before inspection; selected descriptor/origin getters receive their
  documented bounds in addition to the visit charge.
- Charge each standalone string's counted length+16 before byte hashing/inspection;
  validate ownership and checked size before payload traversal. STRING identity is
  immutable data, not a static heap-origin assignment. Repeated strings may be
  rescanned with charges; no uncharged dedup cache.
- Map topology uses the existing exact charged validator schedule:16 header,
 8 per bucket, length+1 for key hash,4 per probe, and compared length+1 (or1 for
  unequal lengths) before equality. Bridge its work accounting to the same remaining
  invocation-preparation budget; do not give each map a fresh full budget. Do not
  call the uncharged fast lookup before topology has been established in this call.

The ledger bounds preparation-owned allocations and borrowed traversal scratch,
not the caller's already-live guest heap. Retained caller heap/stack/constants stay
owned by VmState and must be reported separately in allocator qualification. Any
new stack-reserve allocation remains the existing VM allocation contract with
preflight before operand mutation; it is not hidden in this proof ledger. No guest
operation decrements the boundary-validation work budget after execution starts.

All numeric sizeof assertions, zero-count minima, exact-cap/+1/overflow behavior,
identity-table collision/growth prefixes and source/prior-state preservation must
be measured before this proposal authorizes implementation. Baselines establish
required lifecycle, not that these proposed table/alias bounds are already green.

### Ordinary collector prerequisite retained before runtime work

`vm_release` marks still-referenced containers as suspect in the ordinary heap.
Therefore legitimate public maps/records can have buffered=true between operations
or invocations. The current private map header/value checks require !buffered
because private DAG teardown never enters that collector. Public execution cannot
reuse those guards unchanged. A later reviewed ordinary-heap adapter must accept
valid buffered containers while preserving suspect-buffer/color metadata and
requiring gc_running=false and valid managed ownership. Running collection before
admission merely to erase this distinction would mutate a failed boundary's state
and is not proposed. No adapter or dispatch change belongs to the summary slice.

The proposed259 traversal-frame upper bound refers to at most256 nominal record
nodes followed by at most3 array levels. A map endpoint instead needs at most
256record+1map; STRING keys/scalars are validated inline and acquire no static
origin assignment or DFS frame. These are separate paths, not256records+3arrays+
map: map arrays are outside this profile. Actual positive depth must be measured.

### Boundary-summary implementation charge clarification

I charge the optional row allocation inside the existing 64 MiB analysis reserve.
Before its scan I charge128 logical probes for two origin-kind masks, then8G
for exact tag/unknown/kind validation and copying; scalar and string rows carry no
heap origins. The masks retain only ARRAY and STRUCT sites, never map sites.
The execution constructor reserves the40-byte wrapper before proof. Once G is
known, it additionally reserves16G retained row bytes and24G work before recipe
publication, preserving the full proof reserve rather than refunding it. A late
LIMIT may therefore follow reversible proof allocations; all detached rows and
plans are freed and the caller output remains unchanged. Old NULL exports incur
none of these charges.
