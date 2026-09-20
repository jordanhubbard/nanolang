# I execute checked File CODE through a private VM adapter

I record task_e105bb912f9e44cd84e16ce78358b508 under runtime
82ffef0affec2b23c4390931370f6774. This is the second matched target checkpoint
in [my runtime contract](NANOISA_FILE_PRIVATE_RUNTIME.md), after
[checked frames](NANOISA_FILE_RUNTIME_FRAMES.md). I request independent review
before implementation. Frame PR879 is separately qualified and awaiting merge;
this document neither changes its frozen production nor claims VM execution.

## My private entry and preparation

I propose `src/nanovm/file_vm_private.h` and `.c`, exposing a single private
`nvm_file_vm_execute(bytes, size, scalar_output)` returning the existing
`NvmFileRuntimeReport`. Declarations and implementation require the explicit
`NVM_FILE_VM_PRIVATE` build macro. A normal build provides no executable adapter
symbol or implicit fallback. I link it only in explicit private qualification
targets. I do not change vm_execute, vm_invoke, public readiness, converters,
FFI, wrapper generation or installed CLI selection. All public File refusals
remain authoritative, including bare/truncated File opcode claims.

The input is immutable for the call; output is disjoint from input and private
storage. Calls and the underlying private APIs require external serialization.
This is not thread-safe, and BUSY is not a mutex. Each invocation creates its
own runtime from fresh serialized v2 bytes in VM mode. It owns its plan and
arenas; there is no caller NvmModule, cached authority, externally supplied
mutable frame or borrowed decoded pointer. Unsupported metadata, instructions,
unknown pending masks or handler mismatch refuse before `begin` and host access.

Before begin I walk every prepared function and decoded instruction, including
unreached instructions and uncalled bodies. I require exact supported opcode
coverage. For reachable body facts I validate cleanup classification and known
obligation kind/target/mask against that opcode. I do not rewrite the immutable
query's pending masks into an authority certificate. Preparation failure returns
a deterministic no-acquisition report and preserves scalar output. If runtime
creation succeeded but adapter coverage fails, I destroy only this fresh context.

I retain the exact hosted entry/initializer selection and signatures. After
begin I start the selected root with `frame_start`. Actual RET completes an
initializer through the existing cleanup snapshot before entry can start.
The initializer must have zero results and exact VOID result metadata; scalar
and owning initializer results are refused. Root completion receives NO_SLOT. Both roots use one invocation identity.
Entry publishes only its exact INT or BOOL result after clean terminal finish.

## My complete first instruction inventory

I match the current `file_code_supported` list; no new opcode admission is part
of this checkpoint. The body query's narrower category rules still apply.

| Instructions | Actual execution |
| --- | --- |
| NOP | Advance the checked fallthrough. |
| PUSH_I64, PUSH_BOOL, PUSH_VOID | Create the exact scalar in an empty next operand slot. VOID is initialized and distinct from an empty slot. |
| DUP, POP | Copy or drop only the passive top value permitted by body analysis. Never duplicate an owner or formal. |
| LOAD_LOCAL, STORE_LOCAL | Load an initialized exact passive local; use checked frame store for consumption/overwrite. |
| OWN_MOVE_LOCAL, OWN_STORE_LOCAL | Move the exact File/OpenResult through existing generation-checked roots; no owning C temporary. |
| ADD, SUB, MUL, DIV, MOD, NEG | Exact INT operations, matching the typed variants below. |
| I64_ADD, I64_SUB, I64_MUL, I64_DIV_S, I64_REM_S, I64_NEG | Unsigned intermediate modulo arithmetic; total signed division/remainder. |
| EQ, NE | Same-tag INT or BOOL equality only. |
| LT, LE, GT, GE, I64_EQ, I64_NE, I64_LT_S, I64_LE_S, I64_GT_S, I64_GE_S | Exact signed INT comparison producing canonical BOOL. |
| AND, OR, NOT | Exact BOOL operations producing canonical BOOL. |
| ASSERT | Consume exact BOOL; false records ASSERT at this decoded site and enters terminal cleanup. |
| JMP, JMP_TRUE, JMP_FALSE | Select the actual checked target/fallthrough; conditional forms consume exact BOOL. |
| AGG_PACK, UNION_CONSTRUCT | Build only the exact passive catalog record/Result from declared operand order; no fabricated File/OpenResult. |
| AGG_GET, UNION_FIELD | Consume/project the permitted passive record or known Result arm through the existing carrier helper. |
| AGG_TAG, UNION_TAG | Observe an actual passive scalar Result arm and replace it with INT0 for Ok or INT1 for Error. No owner projection. |
| REGION_BEGIN, REGION_END, BORROW_LOCAL_EXCLUSIVE, FILE_END_BORROW | Use exact current frame helpers and region/formal-origin floors, then advance. |
| FILE_RESULT_BRANCH | Observe the actual local Result arm without consuming it: successor0 is Ok/fallthrough, successor1 is Error/target. |
| FILE_RESULT_TAKE | Check the encoded arm against the actual local arm, consume the local once and publish its exact payload. |
| FILE_DROP_LOCAL, FILE_DROP_STACK | Drop the exact tracked root through the existing core; preserve any cleanup error. |
| FILE_SERVICE | Execute the exact validated import/catalog operation, live reference/owner and byte-domain checks, described below. |
| CALL, CALL_REF | Use checked frame_call; preserve all-arguments-before-any-local staging and exact reference slice. The next iteration executes the actual callee body. |
| RET | Use checked frame_return; stage the result before clearing the callee, resume the exact caller site or complete the selected root. |

I use explicit INT64_MIN/-1 and zero-divisor branches: division by zero returns0,
MIN/-1 returns MIN, remainder by zero or MIN/-1 returns0. Add/sub/mul/neg use
uint64 intermediates and bit-preserving signed representation without signed
arithmetic overflow. I evaluate each operand once. There is no ENUM coercion,
FLOAT, STRING, array, indirect call, loop or generic owner-shell operation in
this first path. These exclusions match current private preparation, not a
reduction of the full release's later acceptance requirements.

## My physical roots and step protocol

The iterative dispatcher keeps only copied instruction/body/frame facts,
passive value views and bounded arrays of at most seven root indices. It does
not use C recursion for calls or allocate a second frame/argument arena.
Every frame index comes from the existing checked getters and VM suffix layout.
Before a handler I verify current mode, function/instruction, reachable fact,
input stack and site; error reports retain function and byte offset.

Passive arithmetic snapshots are not owners. I check types before clearing any
operand, consume the exact input roots, then install the passive result in the
lowest consumed slot. Construction/projection already support their explicitly
checked in-place forms. Every successfully completed ordinary operation calls
frame_next, whose output-stack/empty-staging check remains mandatory. CALL and
RET change frames through their dedicated helpers and do not also advance.

A service that consumes a top operand needs an empty result root even when its
output reuses that slot. I use the current frame's already reserved last staging
slot for this result, never an untracked temporary. If needed I add a checked
`frame_scratch` getter that returns this empty reserved slot with current-frame
validation; it allocates nothing and cannot expose an owner handle. After a
successful service, I move the staged result into the now-empty operand result
slot before frame_next. A later move failure leaves the successful service
result rooted for global cleanup. The same rule applies to any handler needing
an out-of-place temporary. Every normal advance requires scratch empty.

I execute each CFG edge from its real predicate or Result arm, never from a
query's assumed known arm. The copied decoded successor table determines branch
indices; no raw byte-pointer arithmetic occurs during dispatch. Acyclic bodies
and calls bound each individual path, but repeated calls may expand total work.
I add a checked uint64 executed-step counter that fails LIMIT only on counter
exhaustion; it does not impose an undocumented small instruction budget or
claim a practical execution-time bound. No callback or cancellation API is
introduced in this slice.

## My service and pending-obligation conjunction

I retain all original pending masks in the immutable hosted/body facts;
NvmFileRuntimeReport has no pending-mask field. Before begin, every reachable
mask must be covered by this actual adapter and its qualified carrier/core;
unknown bits or an unexpected obligation are UNRESOLVED.

| Pending check | Concrete enforcement |
| --- | --- |
| BINDING | Fresh hosted exact catalog/import identity; copied site agrees with checked target and current service ordinal. |
| INVOCATION | Fresh acquired File-values context shared only by this initializer/entry invocation. |
| LIVENESS and RIGHTS | Existing service/core validates live invocation, slot/generation, receiver and required rights immediately before stream access. |
| BORROW | Frame reference mapping and core exclusive epoch; formal aliases never own or end caller epochs. |
| BYTE | Existing write_byte INT0..255 check precedes narrowing and stream access; rejected range retains File and produces Argument Error. |
| CALLEE | Body preparation checked all callees; actual CALL executes that exact body through checked frames. No opaque callback substitutes for it. |
| RESULT | Exact catalog carrier mapping, actual Result arm/take, staged result ownership and clean scalar host publication. |
| CLEANUP | Every error funnels through finish/destroy; aliases clear, origins end, all actual value roots drain before core disposal. |

Temp reserves output before acquisition. Write/rewind/read use the actual frame
reference. Write consumes only its scalar byte operand, while close consumes
its File operand on both accepted Ok and Error outcomes. Rejected stale/borrowed
inputs remain rejected before any close. Read preserves byte/eof and canonical
zero at EOF. Both update-stream direction restrictions remain unchanged.
Service host errors are Result data; runtime protocol/type/generation errors
stop execution. A handled accepted-close Error still makes initializer completion
or final cleanup fail according to the existing core's retained report.

On any handler failure I record the first status/site once and stop dispatch.
I do not retry normal mutators after sticky failure or invent owner rollback.
Finish/destroy drain staged arguments/results, operands, locals and references;
secondary close errors cannot replace the first failure. No File/OpenResult
crosses host output. I return a report even on preparation failure; scalar output
changes only after fully successful execution and cleanup. Repeated invocations
use fresh identities; no stale context is revived.

## My storage and qualification order

All heap storage remains in the existing runtime's checked64MiB bound, including
full File-values/service/capability storage once. The adapter adds only fixed
passive automatic storage, whose actual sizeof and maximum seven-index scratch
I record in the source checkpoint. If a new owning field is needed, its actual
sizeof participates in the owning runtime budget before implementation review.
The step loop allocates nothing after begin; libc's internal FILE allocation is
still outside the project-owned arena bound and is not misreported as bounded RSS.

1. I submit this contract and complete opcode/mask table for review.
2. I implement the private adapter, any nonallocating frame scratch getter and
   explicit private build linkage; I send the full source/allocation table before
   fixtures execute a service. Default provider builds must not depend on the
   macro-gated symbol; if shared provider lists change, I audit Make/module/
   wrapper/Forth closure together before qualification.
3. I prepare fresh serialized-v2 fixtures that actually execute temp, write,
   rewind, read/EOF and close, exact numeric extremes and both real branch arms,
   nested helpers/aliases, overlapping arguments/returns, every supported opcode,
   initializer ordering and cleanup failure suppression. Unknown masks/opcodes,
   malformed bytes and old public routes refuse before host/loading attempts.
4. I review full allocation-prefix/transient, acquisition and modeled read/write/
   seek/close failures, first/secondary error/site, partial owner transfer, sentinel
   descriptors and clean-only output controls before running. Fault hooks remain
   explicitly distinguished from actual linked libc observations. I reuse old
   carrier/frame/refusal fixtures without weakening their assertions.
5. I freeze source/tools, qualify ordinary and strict GCC/Clang sanitizers on
   Linux and explicit Apple/Homebrew Clang on puck, retain first terminals and
   every phase's objects/binaries, then seal evidence and submit a reviewable PR.

This child can establish actual private VM dispatch only. Generated native
functions/labels (not an embedded interpreter), public selection, paired source
and mandatory shadow execution, installed routes, loops, indirect calls and
richer borrowed calls remain concrete required runtime82ff/full File/full5.1
work. Their checkpoints remain independent of this bounded implementation.

## My first production checkpoint

My scalar output C type is exactly `NvmFileRuntimeView *`, required non-NULL.
The private declaration and entire implementation are gated by
`NVM_FILE_VM_PRIVATE`; no normal provider manifest or public route changes.
I add one nonallocating checked frame scratch getter, returning only the empty
last staging index. No context/arena struct changes or heap allocations occur.

| Storage | Owner and bound |
| --- | --- |
| Runtime, hosted plan, values/references/regions/frames and File core | Existing create/begin/destroy ownership and checked64MiB bound; unchanged concrete structs. |
| Execute loop | One runtime pointer, borrowed const plan pointer, copied frame/instruction/body fact, root index/status and uint64 counter; fixed automatic passive storage. |
| Numeric handler | Two uint32 root indices, two passive RuntimeViews and scalar operands/result. No owner is stored in these views. |
| Constructor handler | At most seven uint32 root indices; construction consumes the existing arena roots in place. |
| Service temporary | Existing last stage slot; success moves it to the operand slot, failure leaves it tracked for terminal cleanup. |
| Result/report | Existing passive RuntimeView publication and RuntimeReport; only destroy after clean complete entry writes caller output. |

These are exact C element types/counts, not a measured compiler stack-frame ABI
or a claim that libc's FILE memory enters the arena ceiling. No recursion or
variable-size automatic allocation is introduced. A prebegin coverage refusal
destroys the fresh unacquired context without publishing output and returns its
own no-acquisition UNRESOLVED report; it cannot finish another invocation.
