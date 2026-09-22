# My capture binding wire and ownership contract

I specify this extension under task_af8091f571a842bc90656e2c7f19b68e. It refines
[my shared-capture design](NANOISA_SHARED_MUTABLE_CAPTURES.md) before production
changes. The identifiers below are proposed against main81454fdc5 and my
f195b843e source branch. They are not implemented or qualified by this document.

## One atomic construction operation

I construct a closure from a checked binding-site descriptor. I do not push
internal cell references onto the ordinary value stack. One operation stages
the whole environment, including any newly boxed locals, then commits it.
This replaces the preliminary design's separate capture/forward operations;
forwarding is explicit in the descriptor rather than an intermediate value.

I retain ordinary NanoValue tags and ordinary foreign signatures unchanged.
I represent each private cell with an owned, one-slot VmTuple. The capture mode
and binding sidecar identify its internal role; its wrapper never becomes an
ordinary source value accepted by arrays, casts, comparisons, generic stack
operators, foreign calls or serialized constants. An immutable source tuple
captured in mode0 remains an ordinary tuple. A mode1 wrapper contains the source
value, which can itself be a tuple. I do not introduce a new public value tag or
a new private collector kind for this representation.

## Container and exact payload

I use the existing NVM v2 container, with required feature bit10 (`0x400`) and
section kind15 (`0x0f`), named CAPTURE_BINDINGS. Existing readers reject the
unknown bit/section. Its payload version is1. I do not reinterpret v1 modules,
existing v2 modules without the feature, or the fixed-width FUNCTIONS table.
The section and feature must occur together. All integers below are unsigned
little-endian. Every reserved byte must be zero; trailing bytes are refused.

The section starts with `version:u16, reserved:u16, function_count:u32,
site_count:u32`. There follows exactly one function record in function-index
order for every FUNCTIONS entry, then exactly `site_count` site records.

Each function record is `function_index:u32, local_count:u16,
upvalue_count:u16`, followed by `local_count` local-mode bytes and
`upvalue_count` capture-mode bytes. Both counts must equal FUNCTIONS. A local
mode is0 for an immutable binding or compiler temporary,1 for a mutable binding.
A capture mode is0 for an immutable copied value,1 for a shared mutable cell.
Other values are refused. Parameters occupy the first arity slots and get their
declared binding mode. A source binding's mode never depends on whether a
particular capturing function writes it. Full nominal/type and affine facts
remain the authority of their existing contracts; these bytes do not grant
permission to capture a resource, borrowed reference or owner.

Each site record is `owner_function:u32, instruction_offset:u32,
target_function:u32, capture_count:u16, reserved:u16`, followed by
`capture_count` entries of `source_kind:u8, mode:u8, source_slot:u16`.
The instruction offset is relative to the owner's first code byte. Records
are strictly ordered by `(owner_function,instruction_offset)`, with no duplicate
site. All indices/counts must be in range. `source_kind`0 names an owner local;
1 names an owner upvalue. The mode must equal both that source binding's mode
and the corresponding target upvalue's mode. The count must equal the target's
upvalue count. Slot order is target capture order. Repeated source bindings
are permitted and must share the same cell when mutable.

Counts are bounded by the actual containing sections, existing function/local
limits and checked allocation arithmetic, not by untrusted multiplication.
The reader validates the complete payload before installing it. A failed read
leaves no partial module state. Writers retain canonical record order and exact
payload version; the loader, execution-module projection and linker preserve
every mode and remap owner/target function indices and code offsets correctly.
Dropping this section during projection is a refusal, never a copied-capture
fallback. Canonical bytecode fixed-point comparison includes this payload.

## Instructions and verification

I propose these currently unused primary opcode bytes. Schema generation,
decoder, assembler, disassembler and all consumers must agree before emission.

| Opcode | Encoding after opcode byte | Stack effect | Meaning |
| --- | --- | --- | --- |
| BIND_INIT_LOCAL `0x97` | `slot:u16` | value to empty | Start a fresh dynamic local binding, dropping its prior local reference. |
| BIND_CLEAR_LOCAL `0x98` | `slot:u16` | unchanged | End the local binding without changing an escaped cell's value. |
| CLOSURE_BIND `0x99` | `site_index:u32` | empty to closure | Atomically copy or share the exact site environment. |

Each instruction requires CAPTURE_BINDINGS. Each CLOSURE_BIND's site must name
the current function and its exact instruction boundary. Every site must name
exactly that instruction; no unused or multiply claimed sites are accepted.
The target is read from the descriptor, not guessed from stack values.

In a module with this feature, LOAD_LOCAL reads the binding value and
STORE_LOCAL assigns that value, updating its cell if boxed. STORE_LOCAL cannot
initialize a missing binding or assign an immutable binding. BIND_INIT_LOCAL
is the declaration/temporary path; BIND_CLEAR_LOCAL is the cleanup path.
Initialization establishes a new identity even when a loop executes the same
instruction and physical slot again. Clearing an already clear slot is allowed
for merged cleanup paths. Loading, assigning or capturing a clear slot is
refused. Parameters begin initialized; other locals begin clear.

LOAD_UPVALUE reads either a copied value or a cell value according to the
target's exact metadata. STORE_UPVALUE requires shared mode and replaces the
cell value. Its existing depth operand must still be zero. In modules without
the feature these existing operations retain their old interpretation.
CLOSURE_NEW is refused in a feature-bearing module: producers use CLOSURE_BIND
for immutable and mutable environments alike, including empty environments.
This keeps environment shape explicit without changing legacy CLOSURE_NEW.

The verifier tracks definite initialization through every reachable control
edge, including handlers and loop backedges. An operation requiring a binding
must find it initialized on all incoming paths. An initialization can replace
an initialized slot; it creates a new binding, never an assignment. Branches
may merge boxed and unboxed instances of the same initialized binding because
ordinary operations observe the same source value semantics. Mode metadata is
fixed for a slot; a producer needing a different mode allocates a different slot.

Direct, indirect, tail, linked and callback entry paths require the exact target
environment shape. A raw function reference cannot enter a function requiring
captures. Closure identity includes its module and validated target; equal
capture counts alone do not establish compatibility. Unsupported consumers
must refuse before publishing runnable output. Complete VM/C/LLVM/Wasm support
remains required for5.1; an interim refusal does not complete this feature.

## Runtime representation and transaction

The concrete implementation uses private per-activation binding state alongside
ordinary locals: initialization state and an optional owned cell reference.
An unboxed initialized local owns its ordinary local value. A boxed local owns
one cell reference; the cell owns the value and the ordinary slot holds VOID.
That VOID is internal storage, not an initialized source binding's value.
An environment slot owns either its copied ordinary value or one cell reference,
with a separately checked mode. No frame stores a pointer into another frame's
ordinary stack. Effect-handler access uses the existing lexical owner mapping;
it does not allocate a second identity for the same owner binding.

CLOSURE_BIND first checks all descriptors, initialization, target shape, stack
capacity and ownership restrictions without mutation. It allocates a private
environment and staging storage with checked size arithmetic. It allocates at
most one new cell per distinct unboxed mutable source local. Each staged new
cell initially owns VOID; its source value remains owned by the unchanged local
until commit. Already boxed locals and shared
upvalues retain their existing cells; immutable sources retain their values.
Repeated references to a new cell share the staged identity.

No new cell is attached to a local until every allocation and retain obligation
has succeeded. Private allocations remain rooted while staging; a collector
must not reclaim them or their values during later allocation. Failure releases
every staged edge and allocation, leaving the operand stack, local values,
existing cells and prior closures unchanged. The retained first error survives
cleanup. Retrying from a handler sees the same prior binding identities.

Commit performs no allocation, callback, collection or other fallible action.
It moves each old ordinary local value into its new cell, replaces that local
slot with VOID, transfers the staged owner edge to its local binding, and
publishes the fully owned closure on
the already reserved operand stack. No partially initialized environment is
observable. There is no general rollback of user code: this transaction covers
only construction of this environment.

## Audited heap representation and remaining frame audit

At f195b843e, `vm_tuple_new` allocates an owned zeroed tuple and accounts its
bytes/object count. `release_tuple` visits every element; `release_closure`
visits every capture. In `heap_cycles.c`, both read-only `for_each_child` and
mutating `for_each_child_slot` already visit tuple elements and closure captures.
The suspect buffer reconstructs the existing heap tag from the object header.
Storing a mode1 capture as an internal `val_tuple(cell)` therefore preserves
these existing collector edges. The binding sidecar owns the same internal
wrapper edge through the ordinary retain/release functions; it never publishes
that wrapper as a language result. This source audit establishes the existing
traversal paths, not passing cycle or lifetime tests for the new feature.

I keep `VmClosure.captures` as owned NanoValue slots. Mode0 stores a copied
ordinary value; mode1 stores a checked nonnull, one-element tuple wrapper.
The executing module's validated target metadata distinguishes these roles.
LOAD/STORE_UPVALUE and capture forwarding must check the shape before using a
mode1 wrapper, including entry through the public callback path. Ordinary tuple
operators never receive the wrapper. Destruction and cycle traversal need not
infer capture mode to release its edge correctly. Existing copied closures
retain their current layout and release behavior.

The frame audit must cover every site creating/removing `VmCallFrame`, including
OP_CALL, OP_CALL_INDIRECT, OP_TAIL_CALL, linked calls, reference-call admission,
OP_PERFORM, OP_EFFECT_RESUME, effect-origin return, ordinary return, runtime
error cleanup, `vm_push_entry_frame`, nested/public callback cleanup and final
`vm_execute` cleanup. `effect_local_index` currently maps to a lexical owner's
ordinary slot; the new binding lookup must use that same owner for its sidecar.
A sidecar borrowed by an effect activation must not be freed as a new owner.
Ownership-profile and private mixed-runtime entry paths must refuse unsupported
combined modes before execution until their full joint rules are qualified.

## Lifetime obligations

### Effect activation binding ownership

Each effect activation owns a distinct binding state for its physical local
array. It borrows its lexical owner's callable environment, but never copies
that owner's binding-state ownership. The existing effect owner mapping redirects
prefix-local access to the same lexical binding and cell. Handler parameters and
later temporary/local slots belong to the new activation. Nested activations
follow the same owner chain for each requested slot; they do not share an entire
local-state array indiscriminately.

I prepare a state with one explicit initialized range: start and count within
the complete local count. Ordinary calls use start0 and arity; effect calls use
the handler's parameter_start and parameter_count. I check the range with
subtraction before allocation, copy every validated mode, and publish only after
successful allocation. Other slots begin clear. Preparation neither reads nor
moves arguments. The caller reserves stack capacity and prepares this state
before moving effect operands; on refusal, operands and existing frames stay
owned and unchanged. The subsequent publication contains no fallible operation.

On activation destruction I release only that activation's state and physical
locals. I resolve access to an owner state before operations, without retaining
addresses into the movable stack. A resumed or lexically returned value retains
its independent operand ownership before any local state is destroyed. Cleanup
must cover every existing return, tail replacement, effect resume/unwind, public
entry failure and VM destruction path before this feature is admitted.

My range-construction controls cover nonzero parameter starts, empty ranges at
the end, oversized ranges, invalid modes, allocation refusal, output and heap
preservation, managed parameters and full cleanup. They supplement existing
ordinary entry/storage controls; they do not establish effect execution parity.

### Complete frame cleanup plumbing

Before feature admission, I add an optional owned binding-state pointer to each
VmCallFrame. Legacy entries initialize it to NULL; effect frame copies explicitly
reset it, so the borrowed lexical closure never duplicates state ownership.
Reference-call entries already zero the complete frame. New states will be
published only after the future entry checks and preparation succeed.

I detach a state before destroying it against that frame's current physical
locals, while the stack storage and every lexical owner still exist. The state
leaves those locals VOID, so existing stack cleanup cannot release them twice.
I preserve independent result/argument operands before this step. For a range
of departing frames I destroy states from newest to oldest. I do not release a
lexical owner's state merely because a handler activation exits.

The concrete cleanup inventory is vm_destroy; tail replacement after retaining
arguments; EFFECT_RESUME after popping its result; lexical effect RET for all
frames above the owner; ordinary/implicit return after removing result operands;
vm_core_execute mixed failure; vm_call_function_scoped owned failure;
vm_invoke_callable nested cleanup; vm_invoke outer cleanup; and the private
mixed vm_ra_unwind path, whose root entry already zeroes its whole frame. Each range cleanup
runs before the existing operand-stack drain. Existing callable release behavior
remains at its current sites. Every ordinary, indirect, linked, public and copied
effect entry initializes the optional pointer, including reused frame slots.

The Makefile VM source closure acquires binding_state.c. The private mixed VM's
explicit provider list must acquire that same source; fixtures using the complete
Makefile object closure retain their existing selection. I require the complete
VM unit gate and real trapped ordinary/effect frame cleanup controls, with
explicitly prepared states and managed values, before qualifying this plumbing.
These controls establish lifecycle cleanup, not capture opcode entry or source
semantics. I retain ordinary refusal until full verification and all consumers
are implemented.

### Entry environment and upvalue access checkpoint

I validate a closure against its executing module's immutable target contract:
module identity, function index, exact capture count and every capture mode.
A raw function supplies no closure and is valid only for zero captures. Shared
slots contain live one-element internal tuple cells; copied slots contain the
ordinary value, including an ordinary source tuple. Header shape cannot grant
capture authority: the caller must supply the exact validated target metadata,
not a newly invented or mutable mode array.

Entry validation checks the whole environment. Per-upvalue access rechecks
identity, count, selected index/mode and cell shape. Reads retain only the
ordinary source value and publish on success. Assignment requires shared mode
and an independently owned incoming operand; it moves that operand into the
cell, clears the incoming storage, then releases the previous value. Inputs and
outputs must not alias closure/cell storage. The closure and original owners
remain rooted during the call. These helpers do not grant public entry, infer
target identity, reserve stack capacity or replace verifier initialization facts.

### Resolved-source construction checkpoint

My internal construction helper accepts a readable borrowed source array, exact
target capture modes, and the validated target's module/function identity. Each
local source names its owning binding state, current locals array and slot.
Each forwarded source supplies a borrowed ordinary value or shared cell and
its already validated mode. Inputs stay live and unchanged during the call;
no callback or stack growth occurs in the helper. Different references to the
same state must supply the same current locals array. Outputs cannot alias any
input or replace an unreleased owner. Wire/site identity, source accessibility,
target metadata and result-stack reservation remain caller obligations.

I resolve every source before acquiring edges. A work-budgeted scan unifies
repeated mutable locals by state and slot, including locals mapped through an
effect owner. Scratch is counted as accounted heap storage while live. I check
the combined remaining closure/cell bytes and cumulative accounting counters
before allocating them. A private new cell initially owns VOID and one staged
owner edge; every corresponding environment slot then acquires its own edge.
Existing cells and immutable values acquire checked references separately.

Failure releases the private environment, staged cell owners and scratch.
I reverse the transaction's acquired references directly, without ordinary
cycle-suspect bookkeeping: each referent still has its original rooted owner
or the staged new-cell owner. I then free the unpublished environment and
VOID cells with exact accounting. No collector allocation or callback occurs
during rollback. I reserve room for at most one diagnostic release per source
before acquiring any reference; rollback cannot overflow that counter.
It may advance diagnostic allocation/retain/release counters, but preserves
existing source values, cell identities, reference counts and the output.
Successful commit moves each unique local's value into its new cell and
transfers the staged cell owner to the local sidecar. No release, allocation,
callback or collection occurs during this publication. Scratch is then freed.
This helper does not admit the wire feature, enter a frame, or establish
definite initialization through instruction control flow.

My corrected f0ff868e2 helper passes438 atomic-construction checks alongside
77 storage checks in seven Linux/Darwin ordinary and ASan/UBSan configurations.
I retain [raw evidence and input identities](evidence/capture-bindings/atomic/checks.json).
Independent static review caught ordinary-release rollback side effects before
the first constructor executed; the tested correction bypasses cycle-suspect
bookkeeping only while reversing its own private acquisitions. Actual frame
integration and source/backend acceptance remain unqualified.

| Event | Required ownership behavior |
| --- | --- |
| Read local/upvalue | Retain the ordinary value for the result; do not expose its cell. |
| Assign shared binding | Own the incoming value before releasing the previous cell value, including self-assignment. |
| Initialize local | Transfer the incoming value into a fresh unboxed binding; release the old local value or cell edge. |
| Clear local | Release only that local's value/cell edge and mark it clear. Escaped cell values remain intact. |
| Enter ordinary call | Move or retain arguments according to the existing call path; each parameter is a fresh binding. |
| Tail call | Retain/move arguments and callable before dropping the outgoing bindings; no argument may depend on a destroyed cell edge. |
| Return/trap/cancel/unwind | Release each initialized frame binding once and the frame's owned callable under its existing contract. |
| Suspend/resume effect | Retain suspended binding owners; resumed access uses their original identities. No duplicate release on handler exit. |
| Destroy closure | Release every copied value or cell edge once, including repeated edges to the same cell. |
| Collect cycle | Traverse closure-to-cell and cell-to-value edges in trial deletion, scanning and final destruction; reclaim closure/cell cycles. |

Frame sidecars, environments, cells and temporary transaction storage count
toward the same enforced heap/allocation accounting. I audit every teardown and
call path before running changed heap code. Per-platform allocation-failure
controls must include failure after an earlier cell was staged, duplicate
captures, an already boxed sibling, aliases through a grandparent and managed
payload replacement. Existing byte conversion and effect-count assertions stay
unchanged. I require actual cycle reclamation, escaped lifetime and repeated-loop
identity evidence, not only a successful single closure call.

This is the proposed wire and ownership contract for independent review. It does
not claim a schema implementation, heap repair, source qualification or release.

## Payload reader checkpoint

My next codec step writes the same canonical payload from borrowed descriptor
tables. I require valid readable arrays for their declared extents, as with
other in-process compiler APIs. I check counts, null pointers and arithmetic
before allocating scratch. I include scratch and the reader's temporary index
tables in one caller budget, then validate the staged bytes with the existing
reader. I publish the owned byte buffer and length only after complete success;
both output locations must be distinct from inputs and each other. Failure
preserves both outputs and frees every temporary allocation. This writer does
not grant instruction or execution authority.

My first implementation isolates the payload codec in `capture_bindings.c`.
It checks canonical function/site order, exact table counts, modes, source and
target identities, source-slot bounds, code-span minimums, reserved bytes and
complete input consumption. Combined index-table allocation has an explicit
caller budget. It borrows immutable payload bytes, frees its partial tables on
failure and preserves the caller's output object until the full decode succeeds.

`make test-capture-bindings` exercises copied/shared/duplicate/forwarded source
descriptors, every input truncation, malformed records and modes, exact allocation
budget endpoints, both table-allocation failures and independent recovery. My independently reviewed60a8b4d1c checkpoint passes714 checks in each of
seven selected configurations: Linux GCC/Clang ordinary and ASan/UBSan, Darwin
Apple/Homebrew Clang ordinary and Homebrew ASan/UBSan. All seven source files
retain exact before/after bytes and modes. I retain the first Clang toolchain
selection failure and Darwin extraction-mode refusal alongside corrected
[evidence](evidence/capture-bindings/payload/checks.json). No warnings, assertions
or source code were changed to obtain these results.

This codec does not yet validate opcode boundaries, establish definite binding
initialization, emit a section or admit its feature in the container/VM. Existing
readers still refuse the proposed feature. Integration must retain that refusal
until the schema, verifier and consumers implement the complete contract; a
successful structural decode alone is never execution authority.

My independently reviewed writer checkpoint46718646a now passes803 checks in
each of seven Linux/Darwin ordinary and ASan/UBSan configurations. The writer
reproduces the canonical fixture bytes, rejects invalid descriptor facts,
preserves both outputs at budget boundaries and all three allocation failures,
and recovers with no live temporary storage. Exact source bytes/modes remain
unchanged before and after every platform run. I retain commands, tools, raw
terminals and identities in [writer evidence](evidence/capture-bindings/writer/checks.json).
This qualifies only the payload codec; schema, instruction verification,
producer integration and closure execution remain open.

## Structural instruction checkpoint plan

I next register the three exact opcodes in the shared schema and C enum. My
ordinary verifier explicitly refuses them while consumers remain incomplete.
I add a separate allocation-free structural pass over successfully decoded,
unchanged capture tables and module bytes. It decodes every complete instruction
with the shared ISA decoder, checks local/upvalue bounds and shared-store modes,
rejects legacy closure construction, and consumes ordered sites one-to-one at
their exact owner/offset with matching encoded site indices. Every site must be
consumed. Its work budget charges one unit per function and code byte before
walking that function, including unreachable instructions. A failure publishes
no proof or changed module state.

This pass does not establish stack shape, ordinary table operands, definite
initialization, effect edges or runtime ownership. The eventual admission path
must combine those checks; I do not call this structural result permission to
execute. Tests retain existing payload checks and add real encode/decode,
truncation, mode/slot/site mismatches and exact work-budget endpoints. Separate
ordinary-verifier controls must show these newly recognized bytes still refuse.

My independently reviewed2d867 structure checkpoint required a direct isa.h
include correction after the first Linux/Darwin compilations stopped before
execution. Corrected d8fc passes977 payload/structure checks across seven
ordinary/sanitized configurations and33 schema tests on each host. Fresh
ordinary verifier builds pass97 tests each, including explicit refusal of all
three new operations; verifier allocation-cleanup controls pass too. I retain
both first compiler terminals, commands and exact source identities in
[structural evidence](evidence/capture-bindings/structure/checks.json).
Definite initialization and complete execution admission remain unimplemented.

## Activation binding storage plan

I audit the current VM before storing local references. Its stack can relocate;
my binding state therefore owns slot metadata and optional cell edges, never a
saved pointer into stack storage. Each operation receives the current locals
array. A single checked allocation stores the state and all slot records, with
copied immutable/shared modes and initialization flags. The first arity slots
start initialized; other slots start clear. The caller still owns actual local
values and supplies zeroed clear slots at frame entry.

A read retains the selected ordinary value only after checking reference-count
capacity; a saturated reference count refuses without changing output or
ownership. I do not use the existing unchecked vm_retain increment as a fallible
transaction. Assignment requires an initialized shared slot. Initialization
starts a fresh binding even if the slot was already initialized. Both consume
an independently owned incoming operand only after validating the operation;
its storage must not alias the local/cell/output storage. Clear is idempotent.
Replacement/clear detach the old local or cell edge before vm_release can run
the cycle collector. Destruction clears each local/cell exactly once, leaves
ordinary locals VOID for the existing stack teardown, then frees the state.

State bytes contribute to VmHeap allocated/freed/allocation-call statistics but
are not a traced heap object. The explicit creation budget bounds current
accounted live heap bytes plus the new state; it does not claim to limit process
resident memory, intern-table storage or all existing heap allocations. I check
counter and size arithmetic before allocation. Existing heap allocation APIs do
not enforce a global memory limit; closure transaction admission must account
for its cells, environment and scratch explicitly before invoking them.

Effect-local access currently follows effect_owner while the requested index is
below effect_local_start. The eventual binding resolver must follow that same
chain and choose that owner's state; each handler activation owns only its own
parameter/temporary slots. My existing stack-height handler edge resets operand
height but does not establish binding initialization or resumed lexical state.
I must implement that separate analysis before granting capture execution.

This checkpoint implements storage operations only. It does not attach cells,
construct closures, change frame layout, admit new opcodes or weaken the pending
full verifier/runtime/C/LLVM/Wasm obligations.
