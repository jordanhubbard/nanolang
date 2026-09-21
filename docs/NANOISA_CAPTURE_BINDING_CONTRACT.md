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
Cells have a private heap object kind and explicit internal edges. They never
become values accepted by arrays, casts, comparisons, generic stack operators,
foreign calls or serialized constants. The collector must understand those
edges even though no public NanoValue tag denotes a cell.

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
most one new cell per distinct unboxed mutable source local, retaining that
local's current value into the private cell. Already boxed locals and shared
upvalues retain their existing cells; immutable sources retain their values.
Repeated references to a new cell share the staged identity.

No new cell is attached to a local until every allocation and retain obligation
has succeeded. Private allocations remain rooted while staging; a collector
must not reclaim them or their values during later allocation. Failure releases
every staged edge and allocation, leaving the operand stack, local values,
existing cells and prior closures unchanged. The retained first error survives
cleanup. Retrying from a handler sees the same prior binding identities.

Commit performs no allocation, callback, collection or other fallible action.
It transfers each new cell's owner edge to its local binding, removes the
redundant ordinary local value edge, and publishes the fully owned closure on
the already reserved operand stack. No partially initialized environment is
observable. There is no general rollback of user code: this transaction covers
only construction of this environment.

## Lifetime obligations

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
