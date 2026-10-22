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

Preflight counts every actual retain occurrence across LOAD/DUP/global read/field GET/CALL/TAIL, grouping equal managed pointers to check cumulative headroom before any mutation. Include staged duplicate arguments beyond the inline staging threshold. PACK/SET transfer existing edges; do not double-retain them. Record GET returning a UNION checks result headroom before releasing parent; exact INT union payload GET creates no heap edge. Ordinary `vm_retain` itself increments unchecked, so preparation alone cannot replace these runtime checks.

Managed container headers permit positive reference counts with buffered PURPLE or unbuffered BLACK/PURPLE; strings are BLACK/unbuffered. Reject active collection; preserve genuine collector buffer membership and counts. Genuine managed storage, owning heap and stable exclusive execution are preconditions, not inferred from headers. Constructor-owned constants plus closed typed transitions establish the initial/ongoing graph; use shallow shells and selected edges, not repeated uncharged subtree scans. Record ordinal0 maps to layout1; union ordinal0 maps to layout0. Never remap ordinary record def_idx to layout1. Both variant orders remain meaningful. Ordinary release handles record-to-union-to-INT/empty depth; do not reuse the private map/array DAG release validator.

## Qualification required before implementation claims

The nine ordinary baseline cases establish behavior only for their exact ordinary modules. Future selected controls must independently prove admissibility, execute actual switch and computed-goto cores, and test initializer return/HALT with entry marker ordering, caller HALT terminality, root tail identity, successful and failed ASSERT resumes, unchanged output on all errors, HALTED VOID even with INT residue, repeated clean invocations, and global committed-alias survival. Add exact last-owner GET, repeated owner saturation, allocation prefix cleanup and ordinary GC destruction. No fresh frontend is needed for the first synthetic consumer gate. Original six retained source modules remain separate actual-source execution obligations.

The proof's conservative None-match and historical field-read alias refusals remain explicit follow-on boundaries; this consumer proposal neither refines tag branches nor introduces temporal snapshots. All old owner/Analysis/VM/heap sizes, old NULL routes, constructor0/0 metrics and six old resource rows must remain unchanged. New scope or owner bookkeeping and work schedules must be concretely measured/reviewed before source implementation.
