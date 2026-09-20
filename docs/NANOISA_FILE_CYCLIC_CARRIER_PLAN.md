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
wrong-kind calls do not consume or destroy an active invocation. Cyclic create returns only INVALID for invalid options and preserves its
context output. The eventual execution adapters, not create or finish(NULL),
construct revision1 INVALID reports with acquired=false, started0 and
fuel_exhausted=false, retaining a supplied limit when available and leaving
scalar output untouched. The new semantic ABI checks revision and all new struct sizes; old
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

## I retain my first unexecuted carrier source checkpoint

I implement the proposed source-private header and explicit plan kind in
`file_runtime.c`, with common fact getters that copy declarations/storage but
never create an old hosted authority. The new facts and carrier includes own
cyclic creation, physical witnesses, fuel and report lifecycle. Old acyclic
entry points retain their plan path. Wrong-kind old finish/destroy return before
cleanup or free; the cyclic destructor owns its separate plan and witness arena.

Pure `nsi_file_values_internal.h` validators check live generations, owner kinds,
borrow epochs and exact live/borrowed slot masks without host I/O. The carrier
compares these masks with all rooted owners/originating references so terminal
core disposal cannot conceal an unrooted live value at a checked boundary.
OpenResult arm checks consult `nl_file_open_view`, not its cached carrier arm.
Underlying slot retirement, generation/epoch counters and public declarations
remain unchanged. The new private declarations are not installed.

Frame transfer changes select only the retained edge variant, validate the full
current/ancestor physical relationships, preserve argument/return roots and
close an instruction before the next charged entry. Common mutation primitives
require an open cyclic instruction; the bounded internal return transfer can
move its already-rooted result after removing the callee frame. This is still
a carrier protocol, not proof that an external C caller evaluated an opcode or
branch correctly. Matched VM/native implementations remain required.

The Make dependency addition covers both carrier object owners and the private
value validator header. I have only reviewed source and checked whitespace;
no fixture, compilation or cyclic/service execution occurs at this checkpoint.
Root source review precedes fixture preparation and fresh qualification.

## I prepare my carrier fixture before execution

Root reviewed the complete carrier source at `d4070d672`. I now add a fixture
checkpoint, without building or running it. I retain the entire old manual
carrier and frame corpus, then manually drive decoded, prepared instruction
sites through the new cyclic protocol. Selecting VM/native arena storage is
not cyclic VM or generated-native execution; those matched adapters remain a
later dependency.

My new cases cover both arena modes and permuted nominal identities, copied
input independence, zero and 258 loop iterations, exact instruction counts,
zero fuel and exhaustion before service/after rooted results/inside borrowed
helper frames, direct entry and initializer completion, nested borrowed calls,
owned arguments/results, and two physical owners swapped through an empty local.
I check the exhausted function/byte offset, first-error persistence, no host
acquisition at an uncharged service, and cleanup failures secondary to fuel.
A held borrow prevents moving its owner. Positive completion checks empty core
owner/borrow masks and empty frame/region state before terminal disposal.

Instrumented cases retain actual core owners while forging only the carrier's
same-slot generation or borrow epoch, poison a variant or idle staging slot,
create an unrooted core owner, and choose the wrong actual OpenResult arm.
I restore deliberate fixture corruption before cleanup when necessary; refusal
must already have occurred. Separate core-owning fixture helpers seed valid
handle/slot pairs at generation and borrow-epoch exhaustion, retire empty slots,
and check carrier region exhaustion. I keep capability-generation and context
identity overflow in the unchanged `test-nsi-file-values` neighbor, rather than
claiming the new carrier fixture independently repeats those controls.

I measure the allocation-attempt count of successful creation plus begin, then
inject every persistent and single-failure prefix through that measured count.
Both creation and begin must expose failures, failed publication keeps its
sentinel, and each prefix restores the retained allocation baseline and host
state before a fresh recovery context. This scope instruments the rebuilt query,
carrier and private core providers; ordinary linked common objects are not a
whole-program sanitizer claim. Linked mode uses separately compiled production
providers, while instrumented mode includes the real carrier/core owning source
for physical assertions and boundary-counter setup.

I also set the project allocation budget to zero after creation/begin, through
all entered instructions, physical witness checks, backedges and final return
in both the zero-iteration and 258-iteration cases. I require no failed allocation
attempt and unchanged retained allocation counts/bytes before restoring the
budget for finish/destroy. This does not instrument host libc allocations.
The separate creation assertion compares retained bytes with the storage bound;
it does not measure or claim a transient preparation peak.

My runner reuses the reviewed cyclic-query command driver: file-backed output,
240-second command bound, bounded TERM/KILL group cleanup even after leader
exit, launch/cleanup errors and terminal JSON retained. I explicitly clear
`LSAN_OPTIONS`; strict supported sanitizer runs retain leak detection. No
historical failing binary is replayed. Before gates I freeze the complete
source/provider/tool inputs and obtain fixture review. Planned Linux GCC/Clang
and Darwin Apple/Homebrew ordinary and supported sanitizer configurations retain
their exact compiler flags, SDK and partial-instrumentation attribution.
Unchanged acyclic carrier/frame, core-value, private VM/native and public
refusal/positive neighbors remain required alongside the cyclic query/hosted
neighbors. I preserve the first unexpected terminal before any correction.
No installed ABI, public admission, cyclic dispatcher or generated-native
execution is added by this fixture checkpoint.

## I preserve my first carrier fixture terminal

At frozen `c5d048fd8`, fresh Linux setup passed in 21.569 seconds and configuration
passed in 0.114 seconds. The first ordinary fixture compiled, passed the complete
retained carrier (84,506 checks) and frame (42,381 checks) suites, then failed at
the inherited `frame_local` helper. I retain the 4.424-second failed phase under
`/tmp/nanolang-file-cyclic-carrier-c5d-linux` and the original produced bytes under
`/tmp/nano-file-cyclic-runtime-ehwku8ns`. No later phase or Darwin gate ran.

The terminal does not print its new-case name. Static inspection identifies the
earliest matching misuse: the held-owner test expects BORROWED, then requests
its local root through `frame_local`; that active-frame query correctly rejects
a non-OK report. The later generation-LIMIT test repeats the same misuse.
Under repair child `1a19c4ff`, I snapshot each exact local root before refusal,
then use the existing root-value view afterward. I retain both the error and
preserved-owner predicates. I add unbuffered case-start markers so later first
terminals identify the begun case. This is a fixture correction, not a relaxed
runtime accessor. Fresh corrected products require reviewed fixture bytes;
original c5d products and first-terminal attribution remain immutable.

## I retain the qualified carrier scope

At unchanged carrier production `d4070d672` and corrected fixture `ac8772ae8`,
Linux GCC13/Clang ordinary and scoped sanitizer carrier gates pass. Darwin
Apple/Homebrew ordinary and Homebrew scoped sanitizer carrier gates pass.
Each instrumented run retains 442,427 new carrier checks, 202 measured
creation/begin allocation attempts, and 398 creation plus six begin refusals
with recovery. Each linked run retains 225,049 new checks. The zero-project-
allocation loop blocks pass; neither transient preparation peak nor whole-
program instrumentation is claimed.

Core-value, cyclic query/hosted, private VM and actual generated-native neighbors
pass on both hosts. Linux's full public suite passes. Darwin's public CLI and
instrumented corpus pass, then its linked method's historical `git show` fails
because the source archive has no `.git`. Under `c4c58d62`, I retain that terminal
and qualify only the remaining linked method in a fresh artifact directory,
using read-only `/Users/jkh/Src/nanolang/.git`, frozen ac877 worktree, exact
`f1606e2c84e67491e9652a5bf71944d235216d95` historical commit and
`732f38decf290ab7e3288d7f77421848aa235c24` source blob. The corrected method passes;
its Git executable/blob and source/tool/provider endpoints remain separate.
Public package tests rebuild providers, so their endpoint changes are recorded
rather than described as global object immutability.

The paired seal preserves both first terminals and all produced bytes. My next
ready integration includes canonical899/900: the new private indirect query
factors declaration/code readers with explicit false-mode wrappers for existing
acyclic/cyclic paths. Carrier/core/VM/native/public execution sources stay exact.
Root approved a separate bounded integration matrix with fresh affected providers,
ordinary carrier, cyclic-hosted and indirect queries, plus public refusal/archive
boundary controls on both hosts. Original sanitizer evidence retains its original
ac877 attribution. No full bootstrap or repeated sanitizer matrix is inferred.
Full cyclic dispatch, genuine generated-native loops, matched public conjunction,
indirect/richer-borrow execution and source acceptance remain unfinished.
