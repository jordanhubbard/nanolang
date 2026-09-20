# My private ordinary mixed VM activation

I continue actual PR933 merge `6b44b302509545373a65520a9d7ff54639e60e93`
under f36b/15f/488. The copied plan and counted storage milestones do not admit
execution. This is a preimplementation contract for the first actual consumer.
Matching generated C, native LLVM and Wasm remain required before any public
selection. I keep the complete domain and all later graph/source obligations in
[my conjunction contract](NANOISA_MIXED_GENERATED_CONJUNCTION.md).

## Distinct authority and lifetime

I do not reuse `VmOwnedInvocationProof`, `owner_arrays`, a public `verified` bit,
or an ordinary classification cache as authority. Those owned paths select
move/reference handlers and a different frame bound. I introduce a distinct
source-private opaque ordinary mixed instance, with entry symbols present only
under `NANO_RECORD_ARRAY_PRIVATE_RUNTIME` in the qualification build. Production
public wrappers remain unchanged and continue refusing this new mixed profile.

The proposed interface separates preparation from execution:

```
typedef struct VmRecordArrayPrivate VmRecordArrayPrivate;
NvmArrayEligibilityResult vm_record_array_private_create(
    const NvmModule *original, VmRecordArrayPrivate **out);
VmResult vm_record_array_private_run(VmRecordArrayPrivate *instance);
void vm_record_array_private_destroy(VmRecordArrayPrivate *instance);
```

The complete source checkpoint defines copied result/statistics getters before
fixtures. No caller obtains a mutable VmState, proof, module or plan pointer.
Create borrows stable original input only for its synchronous duration, prepares
a fresh plan and compares every copied defined field and byte with the original.
It independently materializes a complete execution module from the plan getters
without dropping sections, authority rows, sidecar presence or counted strings.
The original can be destroyed after create. The plan, module and checked decoded
facts remain owned by the instance through final teardown; no original pointer
is used during execution. This materialization is an exact copy, not a rewritten
module passed to old public verification to manufacture acceptance.

Create checks all service/owned exclusions through the fresh query. It completes
all allocation, decoded coverage, descriptor and identity agreement before it
instantiates VM constants, globals or executes an instruction. Failure preserves
`*out` and releases every partial object. Actual VM initialization failure is
MEMORY where caused by allocation and INVALID where a decoded/coverage identity
has disagreed; it cannot publish a partly initialized instance. I factor any
necessary vm_init setup mechanically, keeping old public setup decisions intact.
I do not set vm->verified to evade an old refusal.

An invocation-local internal context ties the exact instance, VM address, copied
module, retained plan and checked decoded tables together. Only the private
instance runner constructs it. The actual dispatch checks that correspondence
before entering or resuming. No exported API accepts a context or boolean bypass.
Callbacks, external/module calls, trace callbacks, linked modules and reentrant
entry remain excluded by this closed query domain. A busy run fails before any
root or prior-result mutation. This private API is single-owner-thread; it does
not promise concurrent lock-free use or permit destroy during an active run.

## Complete correspondence before execution

I compare all 93 accepted opcode recipes and the entire numeric opcode space
against an explicit VM coverage table. I compare every decoded instruction,
including unreachable rows: opcode, exact immediate bits/types, original function
and relative PC, next PC, branch successors, callee, dynamic stack effects and
root obligations. A supported recipe without its actual implementation refuses
create. A malformed later row cannot hide behind a successful entry function.

Decoded direct-call links and dispatch/fusion tables retain original instruction
identity. Every fused component has the same checked source row and complete
root transaction as unfused execution. If a fusion lacks that correspondence,
I use the existing unfused representation for this private route and record it;
I do not silently claim fused coverage. Both actual switch dispatch and computed
goto must execute the same profile. Macro selection and generated binaries are
recorded independently, not inferred from a nominal Make target name.

Record compact ordinals map to exact global layout indices and field counts.
Runtime record identity, field access and array tag checks use those exact facts,
never equal-shaped structural substitution. All five flat element tags retain
independent runtime tags and FLOAT bits. Interleaved unused union declarations
remain fully validated; executing unions and nested arrays stay the separately
required future extension, not misread as ordinary records.

## Real handlers and owning roots

I execute the real NanoValue/VM heap instruction handlers, with a distinct private
admission context. I do not embed a second bytecode interpreter or cast an NmsValue
handle into a VM object. Existing public/owned behavior stays on its old paths.
Every reused handler receives a complete root audit against the plan's recipe;
handler gaps get reviewed source corrections before fixtures, not exemptions.

Loads and DUP retain before publishing. Local/global overwrite stages the new
root before releasing the old one. Field/array GET retains the child before
consuming the receiver. Constructors keep all source-order operands rooted until
allocation and every child retain succeed. SET/PUSH retain new edges before old
edge release and preserve receiver/result identity. A failed transaction either
has not changed the destination or has a documented committed first-error state;
no abandoned intermediate owns a hidden edge. Copy/slice creates distinct runtime
storage; allocation-site summaries never serve as runtime object identity.

Wrong-tag CAST_U8 and all other scalar/tag consumers release consumed heap values
exactly once before unwinding. Integer division, optional indexing, assertion and
bounds behavior follow the actual existing ISA/VM semantics; I do not invent a
new error because an enum has a suggestive name. First actual runtime failure
wins. I retain the original nonvoid and implicit-return depth/tag checks.

Call arguments remain in caller roots until the callee frame and stack capacity
are reserved. Transfer clears the previous owner slot; return stages the result
before releasing callee locals and all dead operands. Error unwind walks actual
live stack/frame ranges once, including suspended callers and staged arguments.
Globals and any previously published result stay separately rooted. The instance
owns its result; successful run atomically replaces it only after full frame
cleanup. Failed run preserves the previously published result, while retaining
any already committed global effects according to normal execution semantics.

Result getters expose copied scalar bits, counted STRING bytes and read-only
nominal/array shape and child observations with instance-scoped opaque identity
numbers; no unmanaged heap owner escapes. Identity distinguishes aliases from
independent copies. Getter failure preserves caller output. Complete definitions
and overflow/exhaustion behavior are reviewed in the source/API checkpoint. This
private inspection surface does not add a public heap-result ABI. Destroy releases
result, globals, constants and residual activation roots before destroying heap,
decoders, copied module and plan. Partial create teardown uses the same ownership
ledger; NULL destroy is harmless. A live instance cannot be destroyed twice.

I run the exact selected initializer before the entry on each invocation, matching
ordinary vm_execute order. Initializer and root share the same globals and first
error. Failed initialization never enters the root. Globals survive repeated runs
and are released at disposal. A nonzero-arity entry retains ordinary missing-host-
argument refusal; preparation does not silently alter its signature or code.

## Bounds, recursion and allocation

The static plan keeps its existing 128MiB/33554432-step ceiling. New preparation
storage and work get separate checked counters and a concrete field-by-field
allocation/scan table in the source checkpoint; the total includes overlap with
the retained plan and independently materialized module/decoded/dispatch storage.
No automatic recursive traversal or uncharged repeated coverage scan is allowed.
I propose a further 128MiB preparation reservation and 33554432 steps, making
256MiB/67108864 combined conservative ceilings. These limits constrain host
preparation workspace, not an arbitrary new language heap lifetime.

Dynamic frames retain VM_MAX_FRAMES=1024, distinct from256 static functions.
The ordinary VM dispatch already uses explicit frames; no C recursion is needed
for bytecode CALL. For each frame the query bounds locals and operand maximum to
256 each. I reserve checked capacity for1024 times the maximum per-frame
locals-plus-operands, plus one staged root result, before running: at most524289
NanoValue slots. This is capacity, not fictitious initialized owners. I audit
all caller/callee slot offsets against the actual shared-stack calling convention;
no handler may publish beyond this checked extent. The source checkpoint must
prove the formula covers every transfer, including zero-local/zero-result calls.
A call beyond1024 records VM_ERR_CALL_DEPTH before consuming staged arguments and
unwinds all active roots. No lower owned-profile cap or unchecked host recursion
may substitute. Frame and root cleanup itself must use bounded iteration.

Heap allocation remains checked by the actual VM allocator and ISA container
limits; I do not impose an unreviewed small total heap cap. Instrumented fixture
hooks measure attempted requests, one-shot and persistent failure, current/peak
live payload, and independent recovery. Reference-count overflow, allocation
failure and collector workspace failure must not publish partial edges or lose
caller roots. Collector safe points occur only with every operand/caller/global
root published, never midway through replacement or call transfer. Any existing
collector recursion limitation must be recorded and repaired or explicitly
bounded by the admitted DAG shape before execution approval.

I introduce no instruction fuel or finite loop count into ordinary semantics.
A program may diverge. Test supervisors use explicit bounded deadlines and retain
partial output/cleanup terminals; that external bound is not a successful runtime
result or a language-level fuel trap. Recursive depth is finite as above; loop
termination is not inferred from a finite abstract query.

## Required fixture and matching-consumer gates

Before execution I submit the complete source, all moved/shared handler checks,
API definitions and fixtures for independent review. The same original modules
must exercise both dispatches, all93 operations, complete old neighbors and
counted root failure paths. Positive programs cover every element tag, exact
FLOAT/STRING bits, nominal equal-shape distinctions, constructors and aliases,
replacement/growth/copy, branch joins/backedges, direct calls, initializer/globals
and repeated entry/disposal. Frame1024 succeeds where the program returns;
frame1025 fails with complete roots cleaned. Heap-result inspection must assert
real alias identity and payload, not just a scalar final return.

Negative controls alter a later instruction, edge, descriptor, nominal identity,
stack fact or copied input and must fail before constants or execution begin.
Allocation sweeps cover every measured preparation and runtime allocation prefix,
normal/assert/tag/bounds failure, initializer failure and nested-call cleanup,
with exact status and fresh recovery. Old service/owned/private/public selectors
retain their prior refusals. Tests must identify normal and test-hook binaries,
retained provider closure and instrumentation limits on Linux and Darwin.

After this private VM checkpoint I implement real generated C and LLVM/Wasm with
matching explicit frame/root semantics and full differential controls. I do not
open public admission after VM-only success, narrow the query to easy opcodes,
or close full mixed/source parents. Whole matched consumer qualification and
installed runtime/archive correspondence remain prerequisites for public routing.

## My private acyclic heap policy

For this exact first query I validate the nominal record declaration DAG again
from the copied descriptor edges before enabling my instance-local heap policy.
I do not infer acyclicity from the origin summary. Every constructed or updated
record edge must match its exact declared child layout; every array edge must
have a flat scalar/string tag. Copy and string-split paths preserve those checks.
A heap path therefore contains at most256 distinct record layouts, followed by
at most one leaf array and one string. Interleaved union declarations do not add
executable heap nodes in this profile.

I disable suspect buffering only for this private instance. I release its graph
with an explicit258-entry traversal stack, retaining each parent until its child
edges have been discharged. The stack is part of the instance's heap storage;
release allocates nothing and does not recurse through the host C stack. This
also avoids collection during a partially completed opcode transaction. Ordinary
heaps retain their existing collector behavior. Nested/cyclic full graphs still
require the separate collector and safe-point implementation in my parent scope.

My eligible STRUCT_NEW remains zero-field only: the existing query compares its
implicit constructed count0 with the exact declaration field count. I map that
constructor's compact identity without expanding its admitted field count.

## My first source checkpoint

My implementation is in `record_array_vm_prepare.inc` and
`record_array_vm_run.inc`, included by the actual `vm.c` only under
`NANO_RECORD_ARRAY_PRIVATE_RUNTIME`. Every VM translation unit in that private
build must use the same macro: it adds instance-local fields to `VmHeap`.
Ordinary VM builds have neither the private activation definitions nor those
heap fields. The image materializer is a source-private query helper; it grants
no runtime authority on its own and is not an installed API header.

I preserve old admission wrappers and pass NULL private context from every old
core entry. The private core entry requires this exact instance, its busy run,
owned copied module, decoded/dispatch storage and unverified ordinary VM state.
It does not set the old verified flag or an owned activation proof. I select no
fusion; both actual dispatch implementations still execute the shared handlers.

I compare all256 opcode decisions and require exactly93 matching recipes. Before
constants I compare every declared instruction, including unreachable rows:
exact operand bits/types/width, dynamic pops/pushes, recipe/check obligations,
fallthrough/branch/call targets, decoder boundary/index maps and unfused dispatch
indices. My image copy checks the complete defined module facts against the
retained plan; I keep both owners until instance disposal.

My extra reservation table is conservative and cumulative. Freed preparation
buffers do not refund it, so overlap never borrows space from another phase.

| Storage or scan | Reservation before allocation or scan |
| --- | --- |
| Retained execution plan | Existing128MiB/33554432-step query ceiling and reported reserved counts |
| Private instance, including1024 actual frames and258 iterative release entries | Exact `sizeof(VmRecordArrayPrivate)` bytes and zeroing work |
| Independently owned image | Fixed owner, all logical copied rows, string terminators, optional parameter rows; exact snapshot-copy and defined-field-comparison work plus both cost scans |
| Runtime copied instruction/field tables | Checked count times owning C type size, plus copy/getter work |
| Global layouts and compact-record map | Fixed256-row arrays in the instance; all declaration/binding/field scans charged |
| Explicit nominal DFS temporary | Exact256 color bytes plus256 pairs of uint16, with one bounded traversal of all fields and layout nodes |
| Decoder and dispatch function tables | Function count times actual owning types |
| Decoder instructions | Sum of every16/doubled reallocation capacity times actual row size, conservatively covering old/new overlap |
| Decoder and dispatch byte maps | Nine bytes times `(code_length+1)` per function; all boundary scans charged |
| Unfused dispatch instructions | Exact instruction count times actual row size |
| Decode/project/compare work | Reserved map zeroing and allocation-copy bytes, code scans, full instruction comparisons and complete opcode-space comparison |
| Operand/local roots |524289 actual `NanoValue` slots, reserved and zeroed before execution |
| Globals | Exact inferred global count times `NanoValue` |
| Constants and intern table | Pointer table, every full string allocation including duplicate upper bounds, all doubling bucket capacities and collision-chain comparison upper bounds |

The private reservation cannot exceed128MiB/33554432 steps. It overlaps the
retained query plan, so reported combined preparation never exceeds
256MiB/67108864 steps. This is a preparation bound, not an execution fuel policy.
The source checkpoint has not yet been built or qualified.

For stack capacity I partition the actual shared stack at active frame bases.
A suspended caller contributes its locals plus remaining operands; arguments
already occupy the next callee segment and are not counted twice. Each segment
has at most256 locals and256 operands. CALL replaces its argument suffix with
the callee's at-most256 locals only after the1024-frame check. RET removes the
callee segment before placing its at-most-one result in the caller. The selected
initializer can leave one result below the entry root, exactly as ordinary
`vm_execute` does; this is the extra slot. Thus every live stack offset is below
`1024*512+1`, including zero-local/zero-result calls and implicit returns. Return
staging is an existing local C value before publication, not an extra live VM
frame. I check the fixed capacity and dynamic operand/transfer bounds at runtime.

My instance owns a previously published result separately from this stack.
Failure drains the actual live stack once and clears suspended callable slots;
globals and the prior result survive. Successful entry stages its result, drains
initializer results, and only then replaces the prior result. Every invocation
increments an epoch before effects; UINT64_MAX refuses before root mutation.
Observation identities compare heap aliases only within one idle instance epoch;
they are never accepted as inputs or dereferenced on behalf of a caller. Getters
copy value bits, shapes or counted bytes and preserve output on refusal. Busy,
wrong-thread and invalid instance operations refuse; destruction of a busy or
wrong-thread instance leaves ownership with the caller.

I keep allocation-free iterative release limited to actual validated DAG edges.
Private string interning and boxed-array push/copy check retain overflow before
publication. Slice rollback releases only successfully retained elements. All
constructors and mutations validate exact physical record layout/field shape and
flat element tags before touching an edge. These changes are instance-local;
ordinary heaps preserve their historical reference and collection behavior.

## My private fixture checkpoint

I keep the complete fixture unexecuted until source review. My two Python
methods build linked and allocation-observed variants in both actual VM
instruction dispatch modes, retaining compiler commands, preprocessor selection,
executables, file-backed output and bounded process-group cleanup. Every VM
layout-dependent provider is rebuilt with the private heap macro; I separately
label the ordinary common-provider objects. The selected query, decoder,
dispatch and VM allocations are observed, not every allocation in common code.

My independent literal table covers all93 recipe/obligation rows and every
numeric opcode. An observed VM instruction profile must report every one of
those93 operations retired by the actual runtime corpus. I independently alter
non-first copied facts, decoded boundaries, signature and dispatch mappings;
these altered preparations are checked and restored without executing them.
My scalar family cases use actual heap operands and require exact TYPE_ERROR,
unchanged retained heap counts and a clean second failure; this qualifies the
reviewed tag-preflight repair, never the unfixed historical source.

My graph cases cover all five flat element tags, counted embedded-NUL strings,
interleaved union/record declarations, global and owner-field aliases, distinct
array slices, append/set/pop, both aggregate constructor/get/set spellings and
the zero-field NEW control. Original input buffers are overwritten and freed
before execution. I check first-success result replacement, first-error scalar
and heap-result retention, committed globals and an intentionally visible
mutation through a retained result alias. Heap arguments transfer through a
real helper CALL/RET. Initializer success leaves an owned result prefix for
entry cleanup; initializer assertion failure never enters main.

The recursion fixtures keep a real heap local in every helper frame. Root plus
1023 helpers succeeds at1024 active frames, with and without an initializer
prefix; one more call must report CALL_DEPTH and dispose every live frame root.
I also check explicit VOID and implicit nonvoid results, preparation refusal of
missing result operands, wrong-thread/busy/epoch boundaries and retain overflow.

All256 nominal rows form a checked DAG. Actual bytecode constructs63 record
levels plus a flat string array, preserving the original64-origin query limit.
A separately labeled heap-layer case uses the same checked256-row descriptors
to construct256 records plus the array/string tail and releases it with all
project allocations denied. This checks maximum iterative release depth; it
does not claim bytecode admission beyond64 origins or general cyclic graphs.

For every measured required preparation allocation and every measured runtime
allocation in the retained graph transaction I inject both a single failure and
persistent failure. Preparation requires exact MEMORY with output untouched;
runtime requires exact MEMORY with the prior result still owned. Each refusal
is followed by complete disposal, zero observed live objects/bytes and an
independent fresh success. The measured requested-payload peak is compared with
the combined reserved preparation bound; allocator metadata and common-provider
allocations are outside this measurement. Exact private byte/work ceilings and
overflow checks must refuse without an allocation attempt.

My pre-run static wire audit corrected the zero-field fixture's extension size
offset: the48-byte tail has an8-byte header, so the size field is at end minus44,
not at its kind/revision word. No malformed fixture was executed or qualified.
