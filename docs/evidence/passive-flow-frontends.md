# My scalar flow frontend boundary

I track this slice as `task_a1bedc94472e491ea40ae8773f3b7ca0`, after the
structured flow marker prerequisite in #547. My source AST retains binding
order. My graph helpers deduplicate reads against the full block name set and
repeatedly select the lowest source ID whose dependencies are complete.

```nano
fn diamond(x: int) -> int {
    flow {
        let total: int = (+ left right)
        let left: int = (* base base)
        let right: int = (+ base base)
        let base: int = (+ x 1)
    }
    return total
}
shadow diamond { assert (== (diamond 2) 15) }
```

My source IDs are `total=0`, `left=1`, `right=2`, `base=3`; my serial order is
`3, 1, 2, 0`. The repeated `base` reads produce one dependency per consumer.
Both my C-seed and self-hosted emitters retain source-ID node arrays and actual
instruction ranges in version-2 passive records. I reuse the existing verifier
and wire format without changing their acceptance rules.

My C environment records a graph visibility start separately from definition
coordinates. Only completed dependencies enter the checking environment;
subsequent emitters resolve those same bindings using the graph start. File and
scope-end filters remain in force. Ordinary source-order blocks do not acquire
forward references. `flow` remains an ordinary identifier except for contextual
`flow { ... }` at a statement boundary.

I accept immutable scalar lets, forward/diamond dependencies, stable ties,
scalar type inference, parameter proofs and checked closed scalar calls. I
retain refusal of cycles, duplicate names, mutable bindings, statements,
nested blocks, effects, aggregates and unproved local inputs in bytecode
records. Pure-function rules and scheduling machinery do not change.

## Retained call boundary

I initially combined multiple flow blocks with ordinary helper calls in their
owning function. The existing verifier refused publication: calls outside node
ranges are explicitly outside the closed-node contract. The original ordinary
source remains `/tmp/nanolang-passive-owner-calls-boundary.nano`; this does not
establish an unsafe execution. I record the continuation as
`task_eccab63d320946d283342bb8fc8eb18f` and test the refusal without weakening it.
My accepted fixture retains multiple blocks in a scalar helper and calls that
helper from a function with no passive block.

## Validation

On source `bf61838dd2d46b1663f9394f3ab2a43741260b69`, integrated with main
`71e433b6`, `/tmp/nanolang-flow-final3-gates.log` exited zero:

- Fresh native C-seed → Stage1 → Stage2 bootstrap and configured smoke checks.
- 86 baseline comparison checks and 85 emitter/artifact/shadow methods
  (110.325 seconds), including exact normalized code and passive bytes for
  seven named functions and six flow records, plus verified VM/strict native output.
- Six flow frontend methods (41.150 seconds), five existing par frontend methods
  (37.430 seconds), and 20 passive metadata methods (1.996 seconds).
- Indexed environment regression, including preserved definition locations,
  graph lexical start, file/scope bounds and an ordinary forward-reference control.
- Clean generated compiler schema and 33 schema tests (1.368 seconds).
- All 79 C typechecker cases.

These are native bootstrap and bounded parity checks, not a new bytecode
compiler fixed-point claim. Earlier checkpoints and failures remain in
`/tmp/nanolang-flow-integrated-gates.log`, `/tmp/nanolang-flow-final-gates.log`
and `/tmp/nanolang-flow-final2-gates.log`.

The expanded inferred-type fixture first exposed the raw
self-hosted let emitter requiring an annotation after flow had inferred a scalar
type. I now pass that checked scalar type through shared typed-let lowering;
ordinary raw unannotated locals remain outside this change. My regression sources are
`tests/nanoisa/fixtures/passive_flow.nano`,
`tests/test_passive_flow_frontends.py`, and the exact paired metadata method in
`tests/test_nanoisa_flat_records.py`. They cover native C-seed/Stage1/Stage2,
canonical bound module owners, raw self-hosted emission, exact function code and
metadata, verified VM/strict native execution, and output preservation on
refusal. I do not infer full passive scheduling or arbitrary capture support
from this bounded slice.

A new helper shadow originally declared two `let _` locals. C-seed built it,
but Stage1 emitted two `nl__` C locals and Stage2's native shadow build refused
them. I retain `/tmp/nanolang-flow-final2-gates.log` and record the general
discard-local boundary as `task_31bb6c9161ad4498a085905d26d1f666`. The helper
shadow now asserts the reset return value directly; its checks remain intact.
