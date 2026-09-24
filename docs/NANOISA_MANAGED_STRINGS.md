# My managed-string runtime contract

I recorded this pre-code contract under `task_51da49b39230468784da3481b893563b`,
after literal-string PR #622 at main `58e0353d`. My core, emitted cleanup,
concat, substring and portable conversion children are now merged through
PR662. The sections describing the initial boundary retain their historical
meaning; current admission is in [my implemented subset](NANOISA_LLVM_MANAGED_STRINGS.md).
Darwin managed acceptance7ba and full applicable-language LLVM/Wasm coverage
remain open. I change no bytecode or source syntax.

## My original pre-implementation boundary

My LLVM values carry `{i64 payload, i8 tag}`. Literal-string payloads are
constant-pool index plus one; my descriptors retain target-native pointers and
explicit byte lengths. Static bytes last for the module instance. The closed
literal profile refuses ADD/CAST_INT/CAST_FLOAT in string-bearing modules and
all allocating string instructions. My ordinary scalar profile stays separate.

`src/nanovm/heap.c` establishes immutable strings with uint32 byte lengths;
`vm.c` retains values on LOAD/DUP and transfers/releases them on stores, calls
and consumption. My generated LLVM currently uses immediate `llvm.trap` for
failed checks and has no cleanup protocol. `scripts/nvm2wasm.py` currently links
no allocator, libc or host imports. I cannot widen admission until both
representation and terminal cleanup exist.

## My handles and object ownership

I preserve static handles unchanged. Bit 63 distinguishes a managed handle;
the remaining payload identifies a slot in a module-owned descriptor table.
I initially cap the slot index to uint32 and validate nonzero/live slots. This
is an implementation/resource bound, not a declaration that large programs are
outside the language. A descriptor owns an immutable byte allocation, its
uint32 length and a checked uint64 reference count. Bytes include a trailing
zero for later conversion adapters; length, equality and order still include
embedded zero bytes. The trailing zero is not part of the language value.

Descriptors can move when their table grows; handles do not. Operations retain
input ownership and resolve their byte pointers again after any table growth.
A byte allocation is not resized after publication. Reused descriptor slots
are allowed only after the last owned reference is released. No operation in
this closed profile fabricates a string tag from an integer or exposes handles
to a host. This is not a claim that arbitrary forged values are safe.

Static handles need no retain/release. A new managed result starts with one
owned reference. The active descriptor table and allocator are module-instance
state, not process-wide state shared among unrelated modules. Strings have no
child references, so reference counting is sufficient for this acyclic child;
that statement does not establish a policy for aggregate cycles.

| Boundary | My ownership action |
| --- | --- |
| PUSH_STR | Push an immortal static handle. |
| LOAD_LOCAL / LOAD_GLOBAL / DUP | Retain one managed reference for the additional value. |
| POP | Release the consumed value. |
| STORE_LOCAL / STORE_GLOBAL | Transfer the popped reference into the slot; release its previous value. |
| SWAP | Transfer positions without changing counts. |
| CALL | Transfer popped argument references into callee parameter locals. |
| RET or implicit return | Move the one declared result out before releasing remaining operand values and all callee locals. |
| Operation consuming operands | Its helper owns popped operands and releases them exactly once on success or failure. |
| Comparison, truthiness, length, numeric/tag error | Release consumed managed values even when the result is scalar or an error. |
| Module disposal | Release globals, then descriptor-table/allocator bookkeeping after all frames have unwound. |

An allocation or table-growth failure publishes no result/descriptor. Previous
global writes remain observable, matching sequential execution; I do not roll
back an entire invocation. A destination whose producing instruction failed
keeps its preceding value. Refcount overflow is a checked runtime failure,
not wrapping ownership state.

## My portable allocation boundary

I propose one emitted runtime interface with two explicitly selected backends.
The native LLVM backend declares the fixed platform libc malloc/free ABI and
links it normally; these are documented runtime dependencies, not arbitrary
module imports. My generated `.ll` contains the ownership/runtime helpers so
`lli` and native Clang execution require no unpublished project object file.
Native allocator lowering pins the selected target size/alignment ABI (the
initial native host builds are 64-bit); an unknown ABI is refused rather than
assuming an arbitrary cross-target override matches host size_t. Sizes use
checked conversion to that target size type. I avoid realloc: a grown
descriptor table is allocated/copied first and committed only on success.

My Wasm backend remains import-free. I emit a reclaiming allocator using its
own linear-memory range starting at the linker's heap boundary, after static
storage and the reserved stack. It uses 16-byte alignment, checked block sizes,
an address-ordered free list, splitting when a usable remainder exists and
coalescing adjacent free blocks. A request first reuses free storage. Only when
no free block fits does it request enough additional pages with memory.grow,
then coalesce the new contiguous range. I calculate byte/page ends in widened
integers and check wasm32 addressability before converting offsets to pointers.
Failed growth leaves the old free list and live allocations unchanged.

This is a proposed allocator algorithm to be reviewed and tested, not permission
to substitute a bump-only arena. Wasm pages do not shrink; free blocks are
reusable within the instance and memory leaves the host only when that instance
is dropped. Live-byte/object counts and page high-water measurements must
distinguish reclaiming reuse from merely retaining every allocation.

I add an explicit native/wasm32 runtime-target selector to the emitting API/CLI;
existing APIs default to native. The Wasm wrapper selects wasm32 and retains
its no-unresolved-import link policy. Runtime helper names are reserved against
custom entry names. I test the selected LLVM memory intrinsics and linker heap
symbol with the installed native/wasm32 toolchain before using them in product
lowering. I do not infer allocator availability from a successful scalar link.
No host memory allocator, WASI or third-party allocator is silently assumed.

The module instance is single-threaded and non-reentrant at its exported entry
boundary, as the existing closed global model is. An active-entry guard rejects
nested host entry/disposal without touching the suspended invocation. Internal
ordinary recursion remains supported. This guard is not a concurrency protocol.

## My failure and teardown ABI

Managed functions return an internal value-plus-status result, including void
helpers. Failed callee status propagates through each caller's cleanup block.
All generated language checks (including ASSERT and numeric tag checks) use
this path; an inner `llvm.trap` must not skip owners. Helpers that have popped
operands consume/release those operands before returning failure, while the
function cleanup releases only values it still owns. I must test both explicit
and implicit returns and joins; runtime ownership follows actual execution.

For a managed target I propose these host wrappers:

- `nano_try_entry() -> i64`: high 32 bits are status; low 32 bits carry the
  existing int/bool entry result when status is zero, otherwise zero. This
  avoids an unchecked caller-provided output pointer in Wasm. Status version 1
  is 0=OK, 1=TYPE, 2=ASSERT, 3=MEMORY, 4=BUSY, 5=DISPOSED, 6=STATE.
- Existing `nano_entry() -> i32` (or the selected named entry) preserves its
  successful result and raises its existing target trap on failure, but only
  after structured frame cleanup has returned. Previously committed globals
  remain owned and may be observed by another non-reentrant invocation.
- `nano_dispose() -> i32`: release instance-owned globals and managed storage,
  mark the instance disposed, and return OK. Repetition is harmless. While an
  invocation is active it returns BUSY without mutating state. Entry after
  disposal returns DISPOSED through the status wrapper. It does not create a
  fresh instance; the host constructs a new instance for fresh semantics.
- A generated native executable `main` disposes its one instance before normal
  exit or final error termination. A reusable native named entry, like Wasm,
  retains globals until its host explicitly disposes it.

Unexpected engine/OS faults such as an exhausted target call stack are not
recoverable language statuses. They can bypass cleanup; the host must discard
that instance/process rather than infer a successful unwinding. Normal language
errors and allocator refusal are handled by the status path. In Wasm, dropping
the instance releases its page backing; disposal alone cannot shrink pages.
This distinction must remain explicit in documentation and tests.

## My operation sequence

1. I implement/test the allocator, stable handles and ownership/status helpers
   without changing executable admission. Deterministic allocation failures
   cover byte allocation and descriptor growth independently. No failed helper
   may publish a half-initialized descriptor or damage an existing alias.
2. I wire actual stack/local/global/call/return cleanup and admit STR_CONCAT and
   string/string generic ADD together with numeric ADD. I retain exact bytes,
   UINT32 length checks, immutable aliases and existing mixed-invalid-tag errors.
   The old conservative ADD refusal can disappear only in the matched managed
   profile. I retain the static-only and numeric profile APIs separately.
3. I admit substring after the VM reference contract is repaired and tested.
   Static review found unchecked uint32 end clipping in `vm_string_substr` and
   no explicit allocation-result error in OP_STR_SUBSTR, unlike CONCAT.
   `task_ce840367841a4bdb94ab69fd2446b635` records the defensive prerequisite;
   I do not reproduce malformed crashes or copy that uncertainty into a target.
   Ordinary byte slicing and the VM's existing index conversion must be pinned.
4. I implement conversion adapters before lifting CAST_INT/CAST_FLOAT refusals
   or admitting CAST_STRING. VM integer/float parsing currently uses strtoll/
   strtod; float formatting uses snprintf `%g`. A portable Wasm implementation
   needs pinned parsing/formatting, overflow, signed zero, NaN/infinity and
   embedded-NUL behavior. Calling undeclared libc is not a solution. If that
   policy exposes platform-dependent VM behavior, I record and reconcile it
   explicitly rather than invent a target-only conversion rule.

The managed-string parent stays open until these promised operation/lifetime
requirements are met. Aggregate and host-linkage tasks remain dependent on the
runtime contract; neither reference counting for immutable strings nor this
closed module ABI completes their alias/cycle/foreign-ownership requirements.

My Wasm runtime supplies bytewise `memcpy` and `memset` for aggregate operations
introduced by the C toolchain, including unoptimized builds. Their volatile
accesses prevent recursive lowering to the same library routines. These exact
names join my reserved runtime entry names on both targets. I do not import a
host libc or claim support for other unspecified library calls.

## My acceptance evidence

I compare ordinary valid programs across VM, LLVM before/after optimization,
native execution and Wasm. Cases include NUL/empty/multibyte bytes, equal content
from separate allocations, repeated concatenation with overwritten owners,
returned aliases surviving caller/callee cleanup, globals across repeated
entries, multiple scalar/string branches and explicit/implicit/recursive calls.

I test reclamation through allocation/live-byte counters, reuse after adjacent
frees, fragmentation/coalescing, table growth with old aliases, Wasm page growth
and bounded-page failure, and native allocation-failure injection. I measure
steady bounded live workloads rather than accepting a short test of a leak.
I test nested-call errors, failed initializers, failed casts/assertions, failure
after a committed global write, repeated invocation after handled failure,
idempotent disposal and independent fresh instances. Allocator and generated
native runtime code receive appropriate sanitizer checks; linked engine code
is not automatically instrumented by those options.

I retain output publication checks, ordinary verifier behavior, current scalar
and literal-profile gates, and rejection of unsupported heap/import/nominal,
reference/passive and signature families. These are concrete evidence for
existing equivalence and ownership obligations, not new release criteria.

## My first implementation child

Task `task_bfe3bb8672c04bcea56dede3f531aee7` implements the reusable runtime core
in portable freestanding C, compiled for native and wasm32 in its own gate.
This keeps allocator/refcount logic reviewable without hand-maintaining two
implementations. It does not yet connect that core to emitted application IR;
the later integration must incorporate its validated target IR/helpers in the
published product and preserve the standalone/native/Wasm contract above.
No external runtime object is silently made a requirement of current outputs.

The core owns descriptor/byte allocations and exposes explicit status results,
retain/release/view/creation and terminal disposal. Function-frame/global
ownership remains a caller obligation until lowering is wired. Deterministic
allocation-failure controls are test-build-only. The Wasm allocator is private
to one Wasm module instance; multiple test contexts in that instance may share
its free pool, while descriptor handles remain context-local. Independent Wasm
instances own independent memory and free pools. Successful temporary page
reservation before a later failure may raise memory high-water, but no failed
operation publishes a live handle or changes a pre-existing descriptor.

The runtime-core checkpoint is tested in `docs/evidence/managed-string-core.md`.
Its C layouts are private target-specific implementation types: I must not cast
my existing LLVM `%S` descriptor directly to `NmsView`, whose length field is
uint32. Later integration must use generated target types or explicit field
adapters, and must wire the reserved-entry predicate before introducing those
runtime symbols to application output. PR628 left translators unchanged; the
[managed concat continuation](NANOISA_LLVM_MANAGED_STRINGS.md) connects these
helpers to emitted frame/global ownership and bounded instruction admission.
