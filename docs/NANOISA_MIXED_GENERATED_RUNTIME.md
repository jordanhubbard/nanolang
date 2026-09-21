# My shared generated mixed runtime

I prepare this design from reviewed PR936 head `d6179a7be0d564858c7f303c457e9ebeae50f606`.
I have verified its actual merge `8f2a6c874cf8c64266d65744529fe20d24eeb9fd`;
root independently audited the query and VM evidence before merging.
I extend `NANOISA_MIXED_GENERATED_CONJUNCTION.md` and retain the complete goals of
`task_f36b179a0f2b4a1b99c29ccd2af66f99`,
`task_15f955fae5cf402d92bf88794122e9a2` and
`task_488a05eb5e2a417caf83a8353363a30d`. This checkpoint changes documentation only.

## Actual implementation boundary

My qualified private VM executes the complete current copied query domain using
real handlers. I now need generated operations with the same values, ownership,
control flow and repeated-entry semantics. My existing `nvm2llvm.c` emits recursive
host calls at OP_CALL and reuses old public verifier/heap selection. Its
`nvm2llvm_managed.inc` calls the singleton `managed_module.c` adapter. My
`nvm2c.c` has separate typed/owned/service routes and allocation sweeps. None of
these is automatically a consumer of the new copied execution plan.

I add a separate private generated runtime and emission entry points; I do not
replace an old verifier result with a success bit or redirect old profiles.
I use `managed_strings.c` counted primitives, including VM-policy arrays, with
an explicit instance. I retain the old singleton adapter and exported ABI
unchanged. I do not embed NanoVM, bytecode decoding or an opcode interpreter in
my generated product.

## First production checkpoint: instance and operation interface

I first implement `record_array_generated_private.h/.c`: a source-private,
macro-gated instance supporting an immutable generated descriptor, explicit
frame/root storage, checked value transactions and status reporting. Before
execution, its complete source review supplies exact C definitions, fieldwise
ABI agreement, allocation/work table and a 93-row helper/emission map. This is
a necessary runtime foundation; manual calls do not qualify an emitter.

My generator consumes only a freshly prepared owned execution plan. It stages
an exact immutable copy of strings, signatures, layout/field/type/binding facts,
origin constraints, all instruction facts and direct control-flow successors.
Preparation compares these defined fields and bytes with the plan, including
all unused functions. No checksum substitutes for agreement. Generated product
storage owns its tables through instance disposal; host plan storage may then
be freed. A runtime ABI revision plus exact sizes/offsets and operation coverage
must agree before creating a counted instance or publishing an output pointer.

I keep NmsValue separate from NanoValue. My private execution status distinguishes
TYPE, BOUNDS, ASSERT, MEMORY, arithmetic failure, frame overflow, BUSY and invalid
state; I map actual core status explicitly and preserve the first error. I do
not stretch the existing NmsStatus enum or its packed result convention. Reports
copy defined fields and leave prior user result unchanged on failure. Successful
heap results own one instance-local root until replacement/disposal; views never
outlive that root. Cleanup errors do not overwrite the first execution error.

I reserve 1024 frame records and a checked root capacity derived from the VM's
524289 bound, with separate globals, retained result and constructor scratch
counted explicitly. I show every call/return/initializer offset in the source
checkpoint rather than treating that number as a new proof. Root slots begin
VOID; moves clear sources, copies retain, and overwrite acquires before release.
The initializer and root execute sequentially in the same instance. Committed
globals survive a failed invocation exactly as in the qualified VM, while stack,
locals and staging unwind. Prior returned values remain owned until successful
replacement. Reentry while active returns BUSY without inspecting replacement
tables, clearing state or acquiring cleanup responsibility.

My qualified VM already reserves two separate preparation domains: up to
128MiB and33554432 steps for the copied plan, plus up to128MiB and33554432
steps for the private consumer. Its reported sums therefore permit256MiB and
67108864 steps. My earlier combined128MiB design sentence was inconsistent
with that implemented boundary. I preserve the two domains in generated
preparation rather than narrowing otherwise qualified plans. I charge copied
tables, instance/frame/root capacity, emission staging and all overlapping
consumer temporaries to the consumer domain; I report both components and
their checked sum. I release a reservation only after ownership ends. Retaining
a plan and duplicating its tables does not permit charging those copies to the
already reserved plan domain. If full generated preparation exceeds the second
domain, I retain the concrete conflict for review before changing a bound.
Runtime heap allocation follows existing checked core capacity/overflow rules;
I introduce no arbitrary execution fuel or host recursion dependence. Wasm
memory limits are explicit target configuration and exhaustion is MEMORY with
complete cleanup, not permission to exclude eligible programs at preparation.

## Ownership and collector correspondence

I validate descriptor DAG, nominal receiver identity, declared field tags and
flat element constraints at every create/get/update/copy boundary. Scalar type
preflight precedes consuming operands, including all INT/FLOAT/BOOL operators
and CAST_U8. Helpers that consume their arguments on failure are distinguished
from borrowing helpers in the full operation table; generated slots are cleared
exactly when consumption occurs. Values are never hidden in untracked C/LLVM
temporaries across a call, allocation, collection or error branch.

Counted record/array/string roots remain external references while collection
runs. I prepare collector workspace before any graph-changing operation and
collect only with all suspended caller roots visible. Failed workspace growth
is an execution failure before mutation. I inspect the core release traversal
and prove or repair its stack bound for the actual accepted nominal DAG before
qualification; VM's instance-local acyclic release policy is not a proof about
this different allocator. I preserve cycles/nested-array refusal in this first
existing query domain, without claiming that its scalar origins establish
physical acyclicity. General graph collection remains a required extension.

## Generated C, then shared LLVM and Wasm lowering

My first emitter checkpoint follows the reviewed runtime foundation and emits
all 93 operations into actual C labels/operators. One generated function per
source function executes basic blocks until CALL, RET or failure. A bounded
outer scheduler selects a generated function/resume label; CALL saves an exact
continuation, checks/reserves a frame before transferring arguments and returns
to the scheduler. RET moves the result before clearing the frame. This scheduler
selects generated control continuations, never reads code bytes or dispatches
opcodes. It bounds host call depth independently of 1024 language frames.

I then generate the equivalent LLVM blocks/functions against the same instance
API, frame layout and operation ledger, using explicit native/wasm32 layouts.
Native and Wasm share lowering; their target ABI, pointer widths and artifacts
are independently verified. No new route uses old recursive generated calls.
Wasm links the actual import-free counted runtime and bounded linear memory;
C and native LLVM link only the required runtime objects, with no VM symbols.
I inspect all actual93 operation cases and all256 opcode decisions before output.
Unsupported new opcodes refuse atomically until this shared conjunction expands.

Emission is transactional into a private buffer, with checked growth and an
explicit output-byte/work bound in the emitter source checkpoint. Every source
function and unreachable label must be covered before publication. Actual stream
I/O failure is reported honestly; CLI file publication uses temporary files and
atomic replacement. Private APIs precede any new CLI/public selector.

## Acceptance and subsequent full product dependencies

I first test the runtime interface's acquired-entry protocol, exact fields,
1024/1025 frame boundary, root movement and finite allocation prefixes (persistent
and one-shot) with fresh recovery. This does not count as generated acceptance.
Then I translate the unchanged private VM corpus through C O0/O2, native LLVM
O0/O2 and Wasm O0/O2 on both supported engines. I compare observable payload bits,
identity/alias relationships, globals, prior results, statuses and cleanup rather
than demanding identical allocation counts across different heaps.

I require all five element tags, all93 operations, initializer failures, recursive
frames, branch/backedge joins, partial constructors/copies, wrong-tag roots,
repeated entries, retained results, source-input destruction and output sentinels.
Allocation sweeps cover actual generated execution and collection preparation.
Strict supported sanitizers identify which runtime and generated TUs/IR are
instrumented. Installed-only isolated linkage verifies package symbols and absence
of testing hooks/VM dispatch. Existing public/refusal/owned/File routes retain
unchanged acceptance. First failures remain recorded; no gate narrows the corpus.

Only after the VM/C/LLVM/Wasm conjunction passes do I propose explicit public
selection with unchanged old route precedence and exact whole-module authority.
Paired C/Nano producers must then preserve declaration/binding metadata across
imports, generics, closures, shadows and source-order evaluation, and exercise
actual bootstrap/compiler workloads through the selected route. A private
fixture is not that bootstrap. Full executed scalar unions, nested and cyclic
mixed graphs, indirect calls and richer source graphs remain required parent
work, with their own complete ownership/query/collector/consumer conjunction.
I neither mark those complete nor substitute the present flat-array domain for
them. Existing refusals remain until each matching extension qualifies.

## Ledger availability

On this checkpoint's first ledger read, `mac task show` failed because its local
login tunnel port34113 was occupied by an unmanaged process. I retain the three
existing parent identities above and do not disturb the listener. This does not
claim creation of a new child task; I will attach this checkpoint when ledger
access is restored.
