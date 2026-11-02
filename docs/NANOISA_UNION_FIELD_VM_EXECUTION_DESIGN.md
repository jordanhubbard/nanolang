# Private union invocation: outcome and scoped lifetime proposal

Design only, following preparation `5b899da28d4b8abd245203f700a448bea78f5cb2` and aggregate `6a88ce502`. No executor implementation, new public selector, native admission or publisher change is authorized by this document. The ordinary baseline is retained in [the receipt](evidence/union-field-ordinary-halt-baseline-20261022/README.md).

## Outcome and output transaction

Proposed private module-owned preparation consumer:

```c
typedef enum {
    VM_UNION_FIELD_RETURNED,
    VM_UNION_FIELD_HALTED
} VmUnionFieldRunOutcome;
typedef struct {
    VmUnionFieldRunOutcome outcome;
    NanoValue result; /* RETURNED: exact INT; HALTED: VOID, no owned edge. */
} VmUnionFieldRunResult;
VmResult vm_union_field_private_execute(
    VmUnionFieldPrivate *owner, VmUnionFieldRunResult *out);
```

This is a proposal, not an exported API. The caller supplies only the prepared owner and output, never a trusted view, module, proof or image. Root entry remains zero-arity/exact one INT result. Helper heap values are not host results. RETURNED requires actual normal root RET and exact one INT result; HALTED represents terminal activation HALT and carries VOID, never a manufactured INT or a stack-residue result. Error is the returned VmResult and leaves every byte of `out` unchanged. Successful publication is one final struct assignment after cleanup; the complete result is initialized locally so padding does not expose uninitialized data. Known owner/output overlap is refused and the documented C precondition requires disjoint writable output from all owned/rooted storage; it does not claim arbitrary heap-interior detection.

There is no new persistent result edge: an earlier caller-owned scalar result remains unchanged on failure. Guest globals persist across invocations and retain committed mutations on errors/HALT; cleanup is not a deep transaction. Public/native/source consumers remain refused for the selected record-containing union metadata. Existing constructor facts and typed copied image are authority only for this separate proposed closed consumer.

## Initializer, root, and cleanup policy

The ordinary baseline proves initializer HALT does not stop entry execution. Preserve that observable sequencing: run initializer if present; either normal return or terminal HALT ends that initializer activation; release all its stack residue and frames, then run entry exactly once. A false ASSERT or another error stops before entry. Normal initializer VOID/INT results are drained as in selected map/array supervision. This deliberately uses clean selected activation boundaries rather than copying ordinary stale-frame behavior; the baseline's global marker order is preserved, but raw public stack/frame leftovers are not a selected-owner API promise.

HALT in an entry or any helper reached by CALL/TAIL terminates the entire current root activation, not just the innermost helper. Do not resume caller bytecode after that trap. Entry HALT publishes HALTED/VOID after unwinding that activation. Normal entry RET publishes RETURNED/exact INT after draining owned activation state. A prior output is never inspected to determine a new outcome. Bare-HALT-only INT source eligibility remains the existing proof/recipe boundary; runtime policy does not broaden it. Use already-admitted branch-return/HALT fixtures for future selected positive tests.

ASSERT is the sole resumable exposed trap in the closed scope: release its condition once; true resumes with unchanged invocation identity and updated IP, false preserves the exact first assertion error and unwinds. Unsupported effects/traps fail closed. The core preserves an explicit returned-versus-HALT outcome until supervisor policy consumes it; do not collapse both to VM_OK prematurely.

On every completion/error: detach/invalidate the invocation view before exposing idle owner; release all transient stack values and frame-owned callables exactly once, clear activation/reference/effect state, restore image-bound module/root identities and saved bookkeeping, and leave ordinary GC metadata and retained globals intact. Do not force GC merely to normalize headers. Preserve the first error during cleanup. Pre-activation validation failure leaves owner/global state and output unchanged; runtime failure permits already-committed guest global/field mutations. Owner remains reusable only after complete cleanup. No persistent VmState context pointer or heap policy is introduced. Exact owner reentrancy/state fields, if any are needed, require separate layout/accounting review before code.

## Typed synchronous scope and preflight

A distinct invocation-local view borrows the prepared owner/image module and carries active flag, active-root function and root stack base. It is never supplied by an external caller or stored in VmState. Extend the innermost typed core with an explicit optional union scope; all old wrappers forward NULL literally. Active scope and all other selected contexts are mutually exclusive. Invalid non-NULL context traps, never falls through to legacy dispatch.

Validate frame/current-function bounds and exact instruction/site correspondence before dispatch indexing, IP advance or profile mutation. The prepared image can contain dormant NOT_ESTABLISHED rows; only an attempt to execute such a sensitive site refuses. HALT has no successor. Decoded/fused dispatch remains disabled for this scope; no guessed alternate/fused path bypasses preflight.

Reset active-root/root-base independently for initializer and entry. Root TAIL updates that local identity; nested calls do not. ASSERT resume validates the same live scope/root identity. Root local/max-operand and full stack/frame bounds are checked before mutation; caller frame and tail staging overlap remain included. No optimistic per-function-only capacity claim.

Preflight counts every actual retain occurrence across LOAD/DUP/global read/field GET/TAIL, grouping equal managed pointers to check cumulative headroom before any mutation. Direct CALL transfers argument edges without retaining them in this closure-free profile. Include staged duplicate arguments beyond the inline staging threshold. PACK/SET transfer existing edges; do not double-retain them. Record GET returning a UNION checks result headroom before releasing parent; exact INT union payload GET creates no heap edge. Ordinary `vm_retain` itself increments unchecked, so preparation alone cannot replace these runtime checks.

Managed container headers permit positive reference counts with buffered PURPLE or unbuffered BLACK/PURPLE; strings are BLACK/unbuffered. Reject active collection; preserve genuine collector buffer membership and counts. Genuine managed storage, owning heap and stable exclusive execution are preconditions, not inferred from headers. Constructor-owned constants plus closed typed transitions establish the initial/ongoing graph; use shallow shells and selected edges, not repeated uncharged subtree scans. Record ordinal0 maps to layout1; union ordinal0 maps to layout0. Never remap ordinary record def_idx to layout1. Both variant orders remain meaningful. Ordinary release handles record-to-union-to-INT/empty depth; do not reuse the private map/array DAG release validator.

## Qualification required before implementation claims

The nine ordinary baseline cases establish behavior only for their exact ordinary modules. Future selected controls must independently prove admissibility, execute actual switch and computed-goto cores, and test initializer return/HALT with entry marker ordering, caller HALT terminality, root tail identity, successful and failed ASSERT resumes, unchanged output on all errors, HALTED VOID even with INT residue, repeated clean invocations, and global committed-alias survival. Add exact last-owner GET, repeated owner saturation, allocation prefix cleanup and ordinary GC destruction. No fresh frontend is needed for the first synthetic consumer gate. Original six retained source modules remain separate actual-source execution obligations.

The proof's conservative None-match and historical field-read alias refusals remain explicit follow-on boundaries; this consumer proposal neither refines tag branches nor introduces temporal snapshots. All old owner/Analysis/VM/heap sizes, old NULL routes, constructor0/0 metrics and six old resource rows must remain unchanged. New scope or owner bookkeeping and work schedules must be concretely measured/reviewed before source implementation.

## Concrete internal interface checkpoint

Keep `VmUnionFieldPrivate` exactly unchanged (measured807232B); reuse its prepared image, copied tables, VmState and `prepared` flag. No epoch, busy flag, result slot, persistent scope pointer or heap policy is needed for the first synchronous route. The lifetime precondition forbids concurrent execute/destroy and any external mutation of opaque owner storage. Same-thread validation, no callbacks/effects/foreign calls, and an established active frame exclude reentry; no authorized host call occurs in the small interval before frame installation or after frame removal. This is a closed C API precondition, not synchronization against hostile concurrent memory access.

Proposed internal types and functions (all owning VM TU only):

```c
typedef struct {
    VmUnionFieldPrivate *owner; /* borrowed */
    VmState *vm;               /* exactly &owner->vm */
    const NvmModule *module;   /* immutable image-bound borrow */
    uint32_t active_root, root_base;
    bool active;
} VmUnionFieldInvocation; /* predicted40B LP64, must measure */

static bool vm_uf_idle(const VmUnionFieldPrivate *);
static bool vm_uf_context(const VmUnionFieldInvocation *, const VmState *);
static VmResult vm_uf_preflight(const VmUnionFieldInvocation *,
                               const VmDispatchInstruction *);
static VmTrap vm_core_execute_union(VmState *,
    const VmOwnedInvocationProof *, VmOrdinaryAdmission *,
    VmRecordArrayPrivate *, VmRecordMapInvocation *,
    VmRecordMapBorrowedInvocation *, uint8_t map_value_tag,
    VmUnionFieldInvocation *);
static VmResult vm_uf_call_root(VmUnionFieldInvocation *, uint32_t function,
                               VmUnionFieldRunOutcome *outcome);
static void vm_uf_unwind(VmUnionFieldInvocation *);
```

The existing seven-argument `vm_core_execute_pair` becomes a literal NULL-union forwarding wrapper to the internal eight-argument function, preserving all older wrappers/callers and macro availability. The new private supervisor alone passes a non-NULL union view; every existing proof/ordinary/array/map context must be NULL there, and `map_value_tag` is VOID. Union context validation runs before ordinary admission fallback, before function/frame indexing, and again before each dispatch lookup. Generic `owned_execution` remains false: reference activation, private DAG remapping, affine call/return behavior and owned global reset paths must not be accidentally selected. Unselected behavior remains literal.

`vm_uf_call_root` publishes its local outcome only on complete normal-return or HALT termination. It does not hide an assertion condition inside that output. A normal result remains rooted on the VM stack until the outer supervisor copies its scalar and cleans the activation. The root descriptor is saved independently from changing `active_root` so tail replacement is checked against the original root result contract as well as target signature. The outer supervisor uses one local view across initializer/entry, with independent root/base resets and cleanup between them; final output is assigned only after the view is invalidated.

At idle require prepared owner, same owner thread, exact module/root image borrow, unverified public flag, valid decode/dispatch and no fusion, exact reserved backing capacities/pointers, zero stack/frame/activation floor, zero handler/reference/callee-reference activation and reference generation, no linked modules/callbacks/trace, no collection and no private DAG policy. Reject stale halt request rather than treating it as an execution outcome. Constant table correspondence remains constructor-owned and immutable. Globals may be non-VOID and ordinary containers may be buffered/PURPLE; neither is an idle rejection. Their graph is retained from prior closed execution, not imported from a caller. Owner, image, constants and descriptor tables never leave the owning TU through an executable trust API.

At active context require1..VM_MAX_FRAMES, bounded current function and frame indexes before dereference, exact module for all examined frames, no closure/upvalues/effect owner, VOID owned callable, active root identity and root stack base, local/operand extents within reserved stack, and current instruction boundary in the corresponding copied function. Validate site receiver kind/nominal/variant and selected member, not pointer-as-static-origin. Full history proof plus closed transitions supplies provenance; runtime masks are compared against immutable copied facts, never reconstructed from pointer identities. Payload access additionally requires the proven payload-bearing variant and actual field count1. Record replacement accepts only the complete historical union obligation; union-payload mutation stays refused.

Preflight may use the image's copied instruction getter (documented four logical probes) once per active instruction, then compare named base/site fields against the prepared row and actual decoded instruction. This is a separate runtime validation operation, not a charge silently deducted from preparation's work reservation. Fixed function/descriptor lookups and at most256-argument duplicate scans are explicitly bounded; no runtime fuel or wall-clock limit is claimed. Do not rescan the complete module/proof per instruction. A full256-opcode audit remains required for the new runtime recipe classification;96 recognized constructor opcodes do not imply96 reachable runtime shapes. Array-producing/use recipes with no qualified array origins fail closed if reached; dormant rows alone do not reject the owner. No array/map/callable expansion occurs.

## Exact retain multiplicities and transfer table

The table describes actual shared handlers on the admitted closed shape, not hypothetical generic behavior. Bounds/type/shape checks precede the described mutation. Heap retain headroom uses `ref_count <= UINT32_MAX - occurrences` with checked occurrence counts. Refusal happens before IP/profile mutation where the next handler retain is explicit; dynamic interning retains are checked inside the existing string helper and may fail after input pops, followed by ordinary supervisor cleanup.

| Operation | New retained edges per instruction | Transfer/release obligation |
|---|---|---|
| PUSH_STR |1 on the selected constant string | Borrow constant then publish owned stack edge. |
| DUP |1 on the actual top owner | Original stack edge stays rooted. |
| LOAD_LOCAL / LOAD_GLOBAL |1 on selected existing value if heap-owned | Local/global edge stays rooted; global index uses capacity, not count alone. |
| STRUCT_GET / AGG_GET of record |1 on selected UNION field | Retain result before releasing parent operand; exact nominal/field checked first. |
| AGG_GET of variant |0 (payload exact INT) | Validate payload-bearing variant; release union operand. |
| CALL |0 argument retains | Existing argument stack edges become callee local edges; nonclosure callee has VOID callable. Check target/parameter tags, frame capacity and locals+maximum operands. Do not reject MAX-ref arguments merely for CALL. |
| TAIL_CALL |1 per heap argument occurrence, cumulatively grouped by identical pointer | Retain all temporary staged arguments before releasing old locals/operands; transfer staged edges to new locals. More than16 uses bounded allocated staging. No extra callable retain because upvalues/closures are excluded. |
| STRUCT_LITERAL / AGG_PACK record or variant |0 existing-edge retains | Allocate shell/field storage before popping; move operands into fields; fresh shell contributes its initial owned edge. Empty variant storage/cleanup follows actual allocator contract. |
| STRUCT_SET / AGG_SET record |0 | Move new value into field, release previous field after replacement, transfer receiver operand back to stack. Same-value identity remains safe because incoming edge already exists. UNION receiver refuses. |
| STORE_LOCAL / STORE_GLOBAL |0 | Move operand to destination and release prior destination; capacity already reserved, historical global/field proof preserved. |
| RET |0 result retains | Move result operand aside, release locals and frame callable, publish same edge into caller/result stack. Validate exact result count/tag before removal. |
| SWAP |0 | Rearrange two existing edges only. POP releases one. |
| AGG_TAG |0 | Read tag before releasing union, produce scalar INT. |
| ASSERT / conditional branches / TYPE_CHECK |0 | Pop condition/value; branches/type-check release in handler, ASSERT transfers condition to supervisor for exactly one release. |
| HALT |0 | No result production; supervisor owns all remaining stack/frame cleanup. |
| scalar arithmetic/comparison/boolean/casts and string predicates |0 explicit heap retains | Preserve existing typed operands/releases and scalar result semantics; invalid domains fail before unsafe scalar union reads. |
| STR_CONCAT/SUBSTR/TRIM/TO_LOWER/TO_UPPER/REPLACE/FROM_INT/FROM_FLOAT and scalar-to-STRING casts |At most1 returned interned string edge, checked by `vm_string_new` | Existing helper returns a fresh ref1 string or checks existing refcount!=MAX before increment. Input releases remain actual handler order; OOM/saturation must not leak popped inputs. CAST_STRING on STRING transfers its input edge with0 retain. |

PICK/ROLL/ROT3, indirect/closure/upvalue operations, native effects, and direct legacy UNION_* opcodes are not silently added to the96-opcode constructor predicate. Generic handlers having a retain does not establish enrollment. STR_SPLIT and array operations cannot produce qualified live array origins in the current union proof; do not reuse the old array recursive preflight to make them executable. Tests must check exact recognized sets and active-site refusal rather than increment a count alone.

## Cleanup ownership table

| Owned value/storage | Owner at a trap/boundary | Required handling |
|---|---|---|
| ASSERT condition already popped | local VmTrap in root supervisor | Release exactly once on true/false; no stack cleanup may release it again. Error message captured before cleanup. |
| Active frame owned_callable | the frame | Closed scope requires VOID, but unwind releases any actual established frame edge once before clearing; invalid arbitrary pointers are outside managed-input preconditions. Normal RET already clears it. |
| Tail staging buffer before successful malloc | none | Failure leaves argument stack/old frame rooted and unchanged. |
| Tail staged argument edges after retain | handler-local staging | All checked retains complete without failure; old frame release then transfers each edge to reserved stack and frees only staging bytes. No external trap between retain and transfer. If implementation adds any failure here, it must release every staged edge before returning. |
| Operand/local stack edges | VmState stack, including post-HALT residue | Unwind pops/releases each live slot exactly once. Normal-result scalar may be copied before drain; heap helper results stay rooted through ordinary return transfer. |
| Global edges | VmState globals | Never rollback/reset during scoped cleanup; committed replacements remain. Destroy later releases them under ordinary GC. No new forced collection solely to clear PURPLE/buffered metadata. |
| Constants/intern table | constructor-owned module constants / ordinary heap | Invocation cleanup never frees constants/table. Existing string helper owns temporary byte buffers until intern result acquired; destructor retains original order. |
| Copied image/tables | prepared owner | Never freed by invocation; destroy only after idle, under exclusive lifetime precondition. |
| Local scope and output staging | stack-local supervisor | Invalidate scope on every exit before returning; commit output only success, otherwise preserve prior bytes. No persistent identity token escapes. |

No constructor reservation changes are needed for this proposed40B local view or bounded inline argument staging; actual per-frame scratch and call depth still require measurement and review against the provisional scratch allowance. Allocated tail staging is guest runtime allocation, with failure-prefix qualification, not part of retained preparation payload. Do not label unmeasured runtime allocations as covered by the constructor's heap peak oracle. These interfaces remain a source-review checkpoint, not implementation approval.

### Independent execution signature gate

Preparation success does not establish the host root signature. Before clearing errors, installing a frame, changing stack/globals or publishing any output, execution independently requires an in-range entry with zero arity, exact result_count1 and result_tag INT. If an initializer exists it must be in range, zero arity, and either result_count0/result_tag VOID or result_count1/result_tag INT. Reject every other prepared signature without altering owner/output; do not borrow the old array admission guard implicitly. Tests must first establish preparation eligibility for wrong-entry and non-scalar-initializer fixtures, then independently require execution refusal and valid-owner recovery. This scalar initializer boundary does not change declaration/proof eligibility.

All non-NULL union entry validation also precedes core early exits for halt_requested, empty frames and end-of-code. Such exits cannot authorize a stale/inactive scope or skip the selected root outcome checks. Existing NULL routes retain their existing order/behavior.

Source checkpoint clarification: OP_STRUCT_NEW is recognized syntactically but `record_site` compares its actual zero-field construction with the declared one-field containing record and refuses preparation. The private runtime explicitly refuses NEW rather than treating it as a literal or reading a nonexistent operand. No new positive NEW support is claimed. Per-invocation identity validation compares one copied counts row, at most256 function rows, and two variant rows to the immutable image before activation; each getter costs four logical probes plus fixed named-field comparisons. This new runtime validation schedule does not alter constructor reservation counters.
