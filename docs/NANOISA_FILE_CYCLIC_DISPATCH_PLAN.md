# My private cyclic File dispatch checkpoint

I refine dependency 3 of [my cyclic execution contract](NANOISA_FILE_CYCLIC_EXECUTION.md)
under `task_15a92c930af7433e9e25b41c7c5c761f`. My starting point is actual
PR902 merge `9c90a55e1a08c79878b5219ca34b0c14a883401f`: the copied hosted plan
and manually exercised physical carrier are qualified. Actual cyclic VM dispatch
and generated native control flow are not yet qualified. This document proposes
those private adapters; it is not their implementation or public admission.

## My concrete interfaces

I add source-private headers and macro-gated translation units:

```c
/* src/nanovm/file_vm_cyclic_private.h; NVM_FILE_CYCLIC_VM_PRIVATE */
NvmFileCyclicExecutionReport nvm_file_vm_cyclic_execute(
    const uint8_t *bytes, size_t size,
    const NvmFileCyclicOptions *options, NvmFileRuntimeView *out);

/* src/nanoisa/nvm2c_file_cyclic_private.h;
 * NVM_FILE_CYCLIC_NATIVE_PRIVATE */
NvmFileRuntimeStatus nvm2c_file_cyclic_private_emit(
    const uint8_t *bytes, size_t size, char **out,
    char *err, size_t err_size);

/* Emitted only in the private generated translation unit. */
NvmFileCyclicExecutionReport nvm_file_native_cyclic_execute(
    const NvmFileCyclicOptions *options, NvmFileRuntimeView *out);
```

I include the existing private cyclic runtime types. I require valid disjoint
output storage and explicit non-NULL options, with revision 1 and a limit in
`[0,1000000]`. The existing default constant is a caller convenience, not an
implicit unlimited or missing-options mode. Emission has no fuel option and
performs no acquisition. Each invocation freshly prepares the serialized bytes,
checks complete adapter coverage, begins one context, and eventually destroys it.

I keep the old private/public byte APIs, `NvmFileRuntimeReport`, native ABI1,
installed archive/header set, CLI, and all public cyclic refusals unchanged.
I do not reinterpret an acyclic plan or cast opaque plan pointers. Private
adapters link explicit reviewed provider lists; generated native drivers have
no VM dispatch object. I add explicit Make dependencies for each included source
and header. The new private targets do not enter public/default selectors.

I add a separate private cyclic native semantic guard, named
`nvm_file_runtime_cyclic_native_abi`, with revision and `sizeof` arguments for
options, cyclic report, cyclic frame view, ordinary frame view and value view.
Revision 1 commits to exact edge-selected variants, charge-before-effects,
shared invocation fuel, checked return staging and cleanup-before-publication.
It delegates the existing carrier size/revision check and also checks the value
view extent. A generated program checks this guard before context creation;
no old native ABI number or public export changes.

## My complete coverage check

I use the retained original function/instruction identities and copied decoded
operands. I enumerate every function and instruction, including unreachable
instructions, before beginning the runtime. I reject unsupported opcodes even
when dead. For every reachable instruction I enumerate **all** retained variants
and require valid logical obligations, supported cleanup/exit kind, input/output
stack counts, exact service/call identity, rights and runtime-check masks.
I retain zero-variant dead labels as explicit refusals. I do not choose one
variant as representative of the others.

I verify every successor ordinal and each variant's exact successor variant,
including backedges; calls use the original callee and that function's retained
entry seed. I check original import identity by the cyclic hosted mapping
(catalog ordinal to original import), not the acyclic inverse accessor.
Missing, duplicate, contradictory or unsupported coverage returns UNRESOLVED
before acquisition. Malformed serialized input and preparation MEMORY/LIMIT
retain their existing classifications. I do not convert allocation failure into
successful partial coverage.

My shared-operation factoring is deliberately smaller than my authority layer.
The existing VM `fvm_step` and native `fn_instruction` already operate on physical
frame-relative slots. I may extract those operations into common internal
helpers, with explicit checked service catalog ordinal and generated callee
calling convention. The acyclic caller supplies its old facts and counter;
the cyclic caller supplies only its distinct checked facts and carrier fuel.
No mode inference through a NULL plan, fallback, or acyclic authority adapter is
permitted. I preserve the old whitelist, numeric results and public outcomes.

## My opcode-to-carrier mapping

I use exactly this existing File operation set. This is a coverage inventory,
not authority to accept operations the cyclic query refuses.

| Decoded operations | Physical operation and continuation |
| --- | --- |
| NOP, JMP | No data change; `frame_next(c,0)` selects the retained edge. |
| PUSH_I64, PUSH_BOOL, PUSH_VOID | Reserve the next operand, create the exact scalar, then next edge 0. |
| DUP, POP | Copy or drop only the already supported nonowner/nonformal value; then edge 0. |
| LOAD_LOCAL, OWN_MOVE_LOCAL | Resolve exact local and next operand; respectively copy nonowner or move owner, preserving existing initialized/formal checks. |
| STORE_LOCAL, OWN_STORE_LOCAL | Use checked `frame_store`, then edge 0. |
| REGION_BEGIN, REGION_END | Use the matching frame region primitive, then edge 0. |
| BORROW_LOCAL_EXCLUSIVE, FILE_END_BORROW | Use frame borrow/end-reference with exact original indices and live core epoch; then edge 0. |
| FILE_DROP_LOCAL, FILE_DROP_STACK | Drop the exact physical root, then edge 0. |
| FILE_SERVICE | Resolve checked original import/catalog ordinal, reference and consuming argument; invoke existing service into frame scratch; move its rooted result to the reserved operand before edge 0. |
| FILE_RESULT_BRANCH | Read actual core-backed arm; edge 0 is Ok, edge 1 Error; do not consume the result. |
| FILE_RESULT_TAKE | Take the recorded arm from the exact local into the reserved operand; then edge 0. |
| JMP_TRUE, JMP_FALSE | Require BOOL, consume it; edge 1 is taken when its truth value equals the opcode's condition, otherwise edge 0. |
| ASSERT | Require and consume BOOL; false is sticky ASSERT with cleanup, true takes edge 0. |
| CALL, CALL_REF | `frame_call` checks/stages transfer and suspends the caller; callee entry is the retained seed; its checked return performs caller resumption. |
| RET | `frame_return` checks complete exit, roots/stages the result, and restores caller or completes the current root. No extra `frame_next`. |
| ADD, SUB, MUL, DIV, MOD, NEG; I64_ADD, I64_SUB, I64_MUL, I64_DIV_S, I64_REM_S, I64_NEG | Existing exact INT operations: wrapping unsigned arithmetic/bit conversion, zero divisor returns 0, INT64_MIN/-1 division returns INT64_MIN and remainder 0. Drop operands and root result before edge 0. |
| EQ, NE, LT, LE, GT, GE; I64_EQ, I64_NE, I64_LT_S, I64_LE_S, I64_GT_S, I64_GE_S | Existing scalar comparison rules, including generic EQ/NE BOOL only when both operands are BOOL; root BOOL result then edge 0. |
| AND, OR, NOT | Existing exact BOOL operations; root result then edge 0. |
| AGG_PACK, UNION_CONSTRUCT | Construct only the admitted catalog value/variant with ordered passive operands; root result then edge 0. |
| AGG_GET, UNION_FIELD | Existing checked catalog projection into the same operand root, then edge 0. |
| AGG_TAG, UNION_TAG | Existing nonowner/nonformal scalar-result arm extraction into INT, then edge 0. |

I keep FUNCREF/CALL_INDIRECT, recursion, FLOAT/ARRAY/general heap operations,
unknown opcodes, and richer borrow profiles refused here. The independently
owned indirect composition work needs a later complete runtime conjunction.

## My VM loop and native labels

For the VM, I start the current root frame, obtain a cyclic frame view and its
exact selected input variant, validate adapter mode/function/instruction/stack,
and call `cyclic_enter` once. Only successful entry permits the mapped operation.
After a noncall operation, `frame_next` checks the complete output witness and
selects the exact retained destination variant. The next iteration reads that
variant; it never scans for a convenient matching alternative.

A direct call enters the callee through the carrier. Its RET restores the
caller's retained successor without charging CALL twice. A root RET finishes
the initializer or entry through the carrier's existing completion protocol.
After initializer completion I start the entry frame in the **same** context.
Entry completion leaves no current root. I reject successful-looking termination
with an unexpected remaining frame/root or staging state.

For native output, I generate a real C function per original function and a
label per decoded instruction. Existing operations already use runtime frame
stack counts, so I do not duplicate instruction bodies per abstract variant.
At every label I fetch the cyclic frame view, check native mode and exact
function/instruction, retrieve its selected variant and stack requirement, then
call `cyclic_enter`. All possible selected variants were covered before begin.
Generated arithmetic remains C operations; branches remain direct gotos;
CALL/CALL_REF remain direct generated function calls. A backedge is a goto
within its function, not recursive C invocation. Callee graph depth remains
bounded by the checked nonrecursive graph (at most 64 functions).

The native operation emitter uses the same service/import facts, physical
primitives and numeric semantics as the VM. It emits no decoded-opcode dispatch
loop and no embedded bytecode interpreter. Its per-function interface carries
only the context; it has no independent `steps` pointer. The carrier owns fuel.

I generate a semantic agreement function covering copied serialized bytes,
startup bounds/entry/initializer, callee order, nominal types/imports, every
function/local/instruction/operand/successor, variant counts and full variant
facts, local/stack/reference/region relations, exact edge variants and all
obligation fields. Comparisons use defined fields, not padding or only a hash.
The freshly prepared cyclic plan must agree before begin. Emission output is
bounded by the existing 128 MiB builder limit, with overflow checks and atomic
`char **out` publication; failure frees all temporary plans/buffers.

## My one charge and one cleanup protocol

I preserve the carrier's actual physical generation/epoch validation and exact
edge protocol. At instruction entry I validate the complete current witness,
including ancestors, staged arguments/results and core live coverage, before
charging. When `instructions_started == instruction_limit`, I report LIMIT at
the current original function/byte offset, set `fuel_exhausted`, and execute no
part of that instruction. Every executed opcode, including NOP, CALL and RET,
costs one. Call resumption, witness checks and cleanup cost zero. Neither a
backedge, helper, initializer transition nor failure path resets fuel.

I construct an early failure report explicitly if no context exists: revision1,
requested limit when options are supplied, zero started instructions, false
fuel exhaustion, no function/instruction site, and exact preparation/ABI status.
NULL options give reported limit0 and INVALID. I do not use `finish(NULL)` to
invent knowledge of the caller's options. A started context supplies its report;
first failure and secondary cleanup errors remain separate. I always run cyclic
destroy after begin failure, dispatch failure or successful root completion.
The caller's scalar output changes only after clean terminal validation and
cleanup. Error, fuel exhaustion and close failure preserve its prior bytes.

## My review and acceptance order

1. I publish the complete VM/native source delta and explicit provider lists for
   review, including any shared factoring and all generated agreement fields.
2. I prepare one identical serialized corpus for VM and real native O0/O2,
   then submit fixtures and bounded retaining runners before execution.
3. I freeze fresh Linux/puck source, selected compilers, actual providers and
   tool identities. I retain first unexpected terminals before corrections.
4. I qualify ordinary and supported scoped sanitizer configurations, reporting
   exactly which carrier/core/common/generated objects are instrumented.

I retain every acceptance in my parent execution contract. Concretely, I cover
zero/one/many/nested iterations, break/continue/early return, multiple exits,
initializer plus entry plus lower-index helper sharing fuel, both Result arms,
owner/reference across backedges, two-owner swap, consume/replace and at least
258 repeated acquisitions with generation reuse. I measure a path's exact N
charges: N succeeds, N-1 fails at the expected site, zero has no service effect.
A structurally returning but actually looping path exhausts fuel while holding
an owner/reference and cleans both. I inspect roots before disposal sweeps.

I exercise argument staging, callee failure and returned-owner staging; assertion,
host and secondary close failure; preparation/begin allocation failure with
fresh recovery; ABI and full-fact mismatch before acquisition; emitter output
sentinels, allocation and size limits; wrong variant/arm and held-reference
refusal. I retain exact report/result/host-event comparisons across VM/native,
not only equal exit codes. No per-instruction project allocations are expected
from preallocated carrier witnesses; instrumentation must measure that claim.

I preserve the complete old acyclic/private/public corpus and public refusal of
the same cyclic wires. I do not claim cyclic public admission, installed APIs,
source loops, indirect or richer borrow execution, LLVM/Wasm, full-product or
parent completion from this private checkpoint. Their remaining dependencies
stay recorded under 15a92/72556/6931 and the original parent contracts.
