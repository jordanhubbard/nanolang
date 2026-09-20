# I preserve physical File lifetimes across checked loops

I refine dependency2 of task_15a92c930af7433e9e25b41c7c5c761f after
merged PR897. My [execution contract](NANOISA_FILE_CYCLIC_EXECUTION.md)
continues to require matched VM/native loops, finite fuel and later public
conjunction. This document requests source-checkpoint approval, not execution
or public admission. Public895 byte APIs, installed headers, native ABI1 and
`NvmFileRuntimeReport` stay unchanged. Closed indirect calls belong to the
separate root-owned2c135 design; richer borrowing and source remain required.

## I separate plan and report kinds

I add an internal acyclic/cyclic discriminator to the opaque runtime and retain
one owning plan pointer of the matching kind. Existing create and plan getters
keep their old contract; the old plan getter returns NULL for cyclic contexts.
A new source-private header proposes these exact shapes:

```c
#define NVM_FILE_CYCLIC_RUNTIME_REVISION 1u
#define NVM_FILE_CYCLIC_FUEL_MAX UINT64_C(1000000)
#define NVM_FILE_CYCLIC_FUEL_DEFAULT UINT64_C(100000)
typedef struct {
    uint32_t revision;
    uint64_t instruction_limit;
} NvmFileCyclicOptions;
typedef struct {
    uint32_t revision;
    NvmFileRuntimeReport runtime;
    uint64_t instruction_limit, instructions_started;
    bool fuel_exhausted;
} NvmFileCyclicExecutionReport;
typedef struct {
    uint32_t revision;
    NvmFileRuntimeFrameView frame;
    uint8_t variant;
    bool instruction_open;
} NvmFileCyclicFrameView;
```

I propose `nvm_file_runtime_cyclic_create(bytes,size,mode,options,out)`, a copying
`nvm_file_runtime_cyclic_frame_view`, `nvm_file_runtime_cyclic_enter`, and distinct
cyclic finish/destroy returning the new report. Enter opens exactly the current
instruction after validation and charging. Existing frame primitives dispatch
internally by plan kind; they never cast cyclic facts to an acyclic report.
Old finish/destroy reject a cyclic context without mutation; the matching
cyclic destructor remains available after every create/begin failure. A cyclic
report getter is copying and nonterminal; all invalid getters preserve outputs.
Old methods on old contexts retain their original behavior.

The proposed private prototypes make ownership and report choice explicit:

```c
NvmFileRuntimeStatus nvm_file_runtime_cyclic_create(
    const uint8_t *, size_t, NvmFileRuntimeMode,
    const NvmFileCyclicOptions *, NvmFileRuntime **);
const NvmFileCyclicHostedPlan *nvm_file_runtime_cyclic_plan(
    const NvmFileRuntime *);
bool nvm_file_runtime_cyclic_frame_view(
    const NvmFileRuntime *, NvmFileCyclicFrameView *);
NvmFileRuntimeStatus nvm_file_runtime_cyclic_enter(NvmFileRuntime *);
bool nvm_file_runtime_cyclic_report(
    const NvmFileRuntime *, NvmFileCyclicExecutionReport *);
NvmFileCyclicExecutionReport nvm_file_runtime_cyclic_finish(
    NvmFileRuntime *, NvmFileRuntimeView *);
NvmFileCyclicExecutionReport nvm_file_runtime_cyclic_destroy(
    NvmFileRuntime **, NvmFileRuntimeView *);
bool nvm_file_runtime_cyclic_abi(uint32_t revision, size_t options_size,
    size_t report_size, size_t view_size, size_t frame_size);
```

The plan getter is borrowed, const and invalid after runtime destruction; it
returns NULL for old contexts. Every proposed name remains source-private.

Creation requires nonnull options, revision1 and limit0..1000000. It copies
options, freshly prepares the complete hosted plan and owns all bytes. It
preallocates bounded witness storage with checked multiplication/addition under
64MiB runtime storage, separately from hosted preparation/retention bounds.
No storage scales with fuel or completed iterations. Failure preserves `*out`
and occurs before begin/acquisition. Begin remains separate; limit0 may acquire
a context but never starts an instruction or File service. No new installed
header, public selector, CLI option or export registration belongs here.

## I validate the exact selected variant

Each internal frame adds a variant byte and an instruction-open bit. Root and
callee entry use the retained parameter seed ordinal0; this is an explicit
hosted invariant, not a default for other instruction sites. A suspended caller
retains its call site, variant and return successor. Its instruction is charged
once before argument staging; entering a callee never resets the context counter.

I preallocate a scratch witness table for at most256 canonical owners plus the
existing bounded formal anchors, reference slots and region depths. Each witness
stores the complete `NlFileValue` identity, not just a physical slot index.
For the selected input variant, validation proceeds deterministically:

1. Compare original function/local declarations, stack extent, initialized bits,
   category/mode/global/catalog identity and Result arm. Abstract UNKNOWN arm
   allows either valid physical arm; it does not allow an uninitialized value.
   Empty abstract locals/stack slots and all idle staging slots must be empty.
2. Walk locals then operands in index order. Bind each ordinary owner label to
   its full invocation/slot/generation tuple. Reject multiple owning roots for
   one tuple, two labels for one owner, missing labels and nonowning values
   claiming owner authority. Formal anchors map through their ancestor reference
   and do not create an owning root in the callee.
3. Validate every physical owner against the core without mutation or host I/O.
   A narrow internal validator reuses `fv_resolve`/`fv_borrow` for exact live
   kind/generation/epoch checks; its source-private declaration exposes neither
   raw host token nor a mint operation. OpenResult arms use the existing checked
   view. This addition is reviewed independently of any public header surface.
4. Match each ordered abstract region to the actual nonzero physical region at
   that depth. Match each live reference's local or formal origin, canonical
   owner label, physical owner tuple, originating reference index and epoch.
   Forwarded formals must resolve to the same live ancestor origin. No formal
   alias may end the originating borrow. Dead references must be fully empty.
5. Reject any residual owner/reference/region outside the validated frame
   relationships. Waiting ancestor frames retain their own rooted values and
   suspended call obligations; they are not mistaken for current-frame extras.

The tables are scratch evidence, not handles. They are rebuilt only at checked
boundaries; no physical generation or borrow epoch is overwritten by canonical
labels. A missing core validation primitive is not replaced with slot equality.
No host access is needed for witness checks.

For ordinary completion I require an open instruction, its exact decoded
successor ordinal and an existing edge bit. The edge's stored variant selects
one target; I validate the post-operation physical state against that target
before committing frame position. I never scan alternatives for a convenient
match. The eventual VM/native matcher must still establish the actual scalar
predicate or Result arm; the carrier alone does not prove branch evaluation.

Call/return retains existing prevalidation, staging and failure roots. A call
validates exact arguments and callee declarations before moving any ownership,
then enters seed0. Return validates exact result and exit obligations before
moving to rooted caller staging. The caller resumes through its saved variant's
edge, with no second CALL charge. A failed partial transfer drains caller,
callee, staging and pending result through the existing cleanup order. Cyclic
operations outside an open instruction refuse, except lifecycle cleanup and
read-only accessors; frame bookkeeping does not create extra fuel charges.

## I recycle storage without recycling identity

Static inspection finds reuse already implemented below the carrier:
`nsi_cap.c` private retirement frees slots after consume/transfer while keeping
the table's monotonic generation counter; private mint/transfer refuse at
UINT32_MAX. `nsi_file_values.c` preserves each empty slot's generation and borrow
epoch, skips exhausted empty slots, increments generation on acquisition/move/
take-Ok and invalidates the source. Existing File/service invocation counters
refuse at UINT64_MAX. Carrier region IDs increment and refuse before wrap.
I propose no reset, wider counter, public capability-policy change or new slot
allocator. If qualification reveals an actual missing reuse path, I retain its
first terminal and obtain a separate reviewed correction.

An exhausted empty value slot is retired for that invocation: acquisition may
use another eligible empty slot, but cannot wrap that slot's counters. Exhaustion
of every eligible slot refuses before acquisition. Move/take-Ok at generation
UINT64_MAX and borrow at epoch UINT64_MAX preserve input/output on refusal.
Region exhaustion preserves the region stack. Capability exhaustion may remain
an existing checked OpenResult.Error with no live stream, rather than becoming
a fabricated fuel error; I preserve each layer's existing status/result mapping.
Cleanup/drop of a live exhausted-generation owner remains possible and must not
require another generation increment. A stale copy after close/reuse cannot
resolve even when its slot number is reused.

The abstract256-owner bound does not promise more than64 simultaneous physical
File/OpenResult slots. I retain existing capacity behavior. More than64 total
acquisitions with at most one or two live owners must succeed when host service
operations succeed; neither query-site identity nor completed acquisitions
consume an ever-growing runtime table.

## I charge and report one invocation budget

The runtime stores copied limit, started counter and fuel-exhausted flag once.
`cyclic_enter` first verifies phase/site/variant/witness and that no instruction
is already open. It records the original function/byte offset. If started equals
limit, it records the first LIMIT with fuel_exhausted=true, opens nothing and
performs no instruction effect. Otherwise it increments once and opens the
instruction. A repeated enter while open is STATE without another charge.

Successful next closes the instruction; CALL closes its charged call before
starting the callee, and RET closes before caller/root completion. Callee RET
and caller continuation do not recharge CALL. Every decoded opcode, including
NOP, branch, CALL and RET, costs one. Failed effects after enter remain counted.
Initializer completion neither resets the budget nor charges entry setup.
No helper accepts a private budget override. Counters cannot overflow because
started never exceeds the validated one-million bound.

The first error remains authoritative. Witness/type/state errors detected before
charge do not consume fuel; an independently reached LIMIT does not set the fuel
flag. A fuel failure does not overwrite an earlier assertion or cleanup result.
Finish drains all references/roots without fuel, then disposes the core and
publishes a scalar only on complete clean success. Repeated finish preserves the
cached report; destroy returns that report before freeing the context. BUSY or
wrong-kind calls do not consume or destroy an active invocation. Invalid options
return revision1, INVALID, acquired=false, started0, fuel_exhausted=false and
leave scalar output untouched; their report retains the supplied limit when
available. The new semantic ABI checks revision and all new struct sizes; old
ABI1 remains exact. Fuel is not a wall-clock or hostile-I/O timeout.

## I review and qualify in dependency order

I request source review of private headers, kind-aware preparation/fact access,
pure core validation, witness checks, frame transfers and charge/report state
before fixtures or execution. This first carrier checkpoint uses explicit
checked frame operations; it does not expose a cyclic dispatcher. Complete
macro-private VM and real generated native functions/labels/direct calls follow
as separate reviewed checkpoints using this same protocol and report. Public
conjunction follows only after both targets qualify; acyclic public refusal of
cyclic modules remains a mandatory neighbor throughout.

Carrier fixtures must preserve all acyclic carrier/frame/VM/native controls and
exercise zero-iteration versus backedge initialization, two-owner swap via empty
temporary, replacement after consume, held-reference refusal, nested forwarded
borrow, absent refined edge, wrong variant, exact nominal mismatch and stale
physical witnesses. Allocation prefixes and single faults prove atomic create
and failure cleanup, with live-root accounting before terminal disposal.

Finite fuel tests include0,1,N-1,N,N+1, exact initializer+entry+callee sums,
repeat-enter/no-open refusal, wrong successor, effect failure after charge,
first-error retention and cleanup at zero remaining fuel. Test-only white-box
counter controls prove near/max generation, epoch, region and invocation
exhaustion without billions of iterations or production test hooks. Reuse tests
perform at least257 acquisitions/consume cycles and retain stale copies across
slot reuse. Host fault/close failure reports stay observable.

Later matched VM/native O0/O2 tests execute the same complete loop corpus,
including service-error arms, original nominal permutations and initializer/
helper fuel boundaries, on Linux and Darwin ordinary and supported sanitizer
configurations. They compare exact result/status/site/fuel/cleanup and prove no
service effect for an uncharged instruction. Native output contains real
functions/gotos/direct calls, not an embedded bytecode interpreter. Closed
indirect, richer borrowing, paired source/all shadows and full release remain
open and are not discharged by this bounded carrier checkpoint.
