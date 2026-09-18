# My original frontend and runtime acceptance reconciliation

I audit canonical `1e9d9519f89a63ef443500a7d8156554400add0e` on
2026-09-18 under task_9e43514b701c4869bf67c3d8d803662a. My
[original task clauses](evidence/original-acceptance-reconciliation/task-clauses.json)
are live MAC snapshots, not a new expansion of those tasks. This is a read-only
evidence audit: I run no historical compiler or failed artifact and make no new
current-head test claim. Later product and release gates remain independent.

## My call-scoped borrow task718

The original task begins with nominal resource parameters and explicit `&` and
`&mut` call arguments. It explicitly assigns later IR ownership representation
to ed702. It requires actual caller identity and paired C-seed/Stage1/Stage2
acceptance, not a value-copy surrogate. I map every stated case below.

| Original clause | Merged evidence and executable assertion |
| --- | --- |
| Retained annotations and explicit arguments | [Annotation checkpoint](evidence/call-scoped-borrow-annotations.md), `tests/test_borrow_annotations.py`; shared/exclusive suites exercise the admitted syntax after the initial refusal-only prerequisite. |
| Read shared owner twice, then consume | `test_repeated_shared_aliases_and_forwarding` in `tests/test_shared_borrows.py` reads, aliases the same owner twice, forwards and finally consumes it. |
| Exclusive mutation visible to caller, then consume | `test_mutation_is_visible_to_caller_and_forwarded_reference` in `tests/test_exclusive_borrows.py` observes 11→12→13 through caller and forwarded references, then consumes 13. Generated pointer-signature checks and the environment identity fixture exclude an invisible value-copy substitute. |
| No consume, overwrite or escape through shared borrow | Shared `test_borrowed_parameter_cannot_move_escape_or_overwrite`; exclusive mutation-needs-capability and later-argument shared-hold tests. |
| Exclusive/shared overlap and move during live argument borrow | Exclusive `test_overlapping_arguments_and_later_reads_are_rejected`; shared `test_argument_order_holds_owner_until_call`, including conditional owner overwrite. |
| Use after move, stored/escaping references, annotation mismatch | Shared `test_borrow_requires_live_explicit_named_owner`; exclusive stored-reference and owned-return refusal methods; annotation and borrowed-callback refusal controls. |
| Preserve nominal identity and ordinary by-value behavior | Shared imported same-spelling/nominal-shape refusal methods; environment identity fixture checks borrowed identity versus independent ordinary copies; exclusive ordinary/owned callback positive control. |
| Paired actual behavior and checked refusals | [Exclusive integrated evidence](evidence/call-scoped-exclusive-borrows.md): fresh default-budget bootstrap and 36 exclusive/shared/annotation/callback methods across C seed, Stage1 and Stage2 at `1bc18a1e`; parser/typechecker, 45 environment checks, ten lexical methods and generated-schema gates. Rejections require checker diagnostics and old-output preservation. |

Shared PR508 (`984f8fff`), exclusive PR517 (`3703847d`) and the later nested-place
PR527 (`f229173a`) are canonical ancestors. [Shared evidence](evidence/call-scoped-shared-borrows.md)
retains the earlier deadline observation without attributing its cause. I can
reconcile original718 after this audit merges. Its stale shared-only task note
is not an outstanding exclusive implementation requirement. This conclusion
does not implement arbitrary stored/generic/foreign callback references, close
ed702, or close the full affine matrix28.

## My frontend tasks c60 and20048

The live ledger marks both tasks completed. I preserve that history but do not
replace their descriptions with the completion bit.

The [C-seed flow checkpoint](evidence/affine-c-seed-flow.md) replaces the old
identifier-state prototype with growable function flow, parameters, observation
versus move, scope/branch/loop checks and allocation-failure controls. Its first
paired run retained 27 failures; that report remains historical evidence.
The [subsequent selfhost flow checkpoint](evidence/affine-selfhost-flow.md)
records corrected paired flow: fresh bootstrap, 21 combined ownership methods,
ten lexical methods and 21 compiler-to-native methods. Those controls cover the
original pinned declarations/moves/consuming calls/rejection corpus and extend
it to branch joins, early returns, loops, reassignment, hidden outer obligations,
short-circuit paths and recursive record/union classification. Actual source is
`src/resource_flow.c` and the `resource_flow_*` functions in `src_nano/typecheck.nano`.

This reconciles20048's pinned frontend/bootstrap corpus, while preserving the
explicit supported-language boundary. It does not claim every future affine
form, inferred ownership fact or cleanup route merely from a static decision.

The broader c60 description also asks for emitted ownership facts and a full
positive/negative matrix. Its historical completion does **not** independently
close those clauses. Subsequent whole-owner, borrow and IR checkpoints advance
them, but the full conjunction remains explicitly under
`task_ed70242ac4d83be7b2327da7ece387ad` and
`task_28f2fb4b1f3c8a5ce93df628bb569d76`. I correct the roadmap's claim that c60
itself is still open; I keep the full normative acceptance checkbox open. No
new duplicate implementation task or fabricated all-language matrix is needed.

## My private record foundation2dc

The original task is explicitly non-admitting. [Descriptor evidence](evidence/managed-record-plan.md)
and [private storage evidence](evidence/managed-record-storage.md) satisfy its
two ordered checkpoints:

- Global retained-layout indices remain distinct from per-kind record ordinals;
  explicit empty definitions, placeholders, interleaved kinds, nominal identity,
  nested indices, owned plan lifetime and canonical bytes are checked.
- Dedicated stable record slots retain exact scalar/string/array/record children;
  construction rolls back, GET retains, SET retains before replacing, and mixed
  record/array edges participate in iterative release and cycle collection.
- Actual native sanitizers and import-free Wasm qualify descriptor binding,
  repeated aliases, dead-to-live edges, table/workspace allocation failures,
  a 4000-record chain and bounded repeated cycles. Production/package ABI and
  old public profile refusals remain checked.

PR733 (`11d64547`) and PR737 (`f922299a`) are canonical ancestors. The task is
already completed; the unchecked aggregate roadmap foundation row was stale.
Ordinary authority15f, field origins and generated nominal admission remain
separate requirements. I do not infer storage authority from field shape.

## My original AOT runtime707

This task records a concrete implementation checkpoint as done in tree: dynamic
arrays, packed strings, self-tail-call lowering, heap records, map growth and
whole-record calls, with an AOT compiler reaching merge/lex/parse/typecheck.
Its stated acceptance is not the complete final product cutover.

The later [native full-source fixed point](evidence/native-full-fixedpoint-host-owned.md)
and its [machine-readable manifest](evidence/native-full-fixedpoint-host-owned.json)
exercise the complete compiler through normal NanoISA emission, `nvm2c`, native
linking and two native self-generations, followed by verified hello execution.
All recorded stages exit0. Initial and both generated modules are378144 bytes;
raw generation-one/two equality and immutable host closure equality pass.
The post-run integrity snapshot records unchanged source/helper/translator/
native executable/libraries. This directly exceeds the old merge/lex/parse/
typecheck milestone without requiring obsolete internal helper names forever.
Current `src/nanoisa/nvm2c.c` retains dynamic aggregate storage, whole-record
values/calls and `L_tco`; subsequent shape/lifetime repairs are independently
qualified improvements, not a missing initial runtime implementation.

The qualified source pin `2f50d7a0` is not itself a canonical ancestor. Its code
base `61b0aedc` is; their entire diff is five roadmap lines, with no production
change. I retain that distinction rather than invent a final-main bootstrap.
The original707 runtime checkpoint can be reconciled after this audit merges.
PR522's default-product transition and fresh final candidate/platform gates
remain open and cannot inherit this older fixed point automatically.

## My remaining parents

I leave full affine28, ownership transport ed702, real-handle d03c, ordinary
record authority15f, aggregate488, managed51da and product/release acceptance
open. Completed original718/707 and the bounded frontend/private-record rows
remove stale bookkeeping; they do not remove any of those explicit obligations.
