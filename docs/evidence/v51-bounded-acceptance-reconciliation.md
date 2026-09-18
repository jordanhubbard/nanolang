# My bounded v5.1 acceptance reconciliation

I audited merged main `4c2ae731` and the retained evidence below. I change no
compiler code and run no executable acceptance probes in this reconciliation.
My documentation task is `task_0f0655717d984eae8bf6c82a109b1229`.

## Executable reconstruction

PR604's tested head `9d76aedc` supplies the second executable high-level spike
and the published finding. I retain eight original modules, eight VM runs,
eight strict native C executions and 24 C-seed/Stage1/Stage2 native executions
of reconstructed NanoLang. Twenty refusal decisions preserve previous output.
My [finding](../NANOISA_HL_ROUNDTRIP.md) and
[measured evidence](nanoisa-scalar-reconstruction.md) distinguish the tested
closed scalar grammar from general reconstruction. Neither output embeds an
interpreter. The original assembly source is removed before reconstruction.
Fixture shadows are supplied test assertions, not recovered original tests.

I mark only the spike and finding milestones complete. Parent
`task_4bd034f6029b7458201db74e2c3aeb32`, general structured reconstruction,
complete frontend facts and broader target coverage remain open. I introduce
no original-shadow-retention release gate.

## Qualified callback signatures

The earlier qualified-import mismatch is historical. The merged call-context
repair and integrated evidence (`c765bb7b`, `675cbbef`) check ordinary indirect
and qualified arguments in both source checkers. A fresh native bootstrap,
parser/typechecker suites and 35 paired methods passed, including qualified
execution/rejection and previous-output preservation. The completed child is
`task_3ca0e46fbbc64aa8bc39bfdaf65b8833`.

I retain [the original failure and corrected acceptance](callback-signature-boundaries.md).
The ordinary signature row can close; complete callback ownership and parent
`task_e05a42e2e09b47cc9c53fa6923eeeaef` do not close by implication.

## C-seed ownership wording

The old `task_c4e2f078cef8c4e461f0de3711c8a2b9` is cancelled. The current
checker is the merged growable `src/resource_flow.c` pass, not the unmerged
fixed-capacity `c53e27a7` recovery prototype. Its
[C-seed checkpoint](affine-c-seed-flow.md) tests 300 owners, capacity rejection
and allocation-failure cleanup. The subsequent
[self-hosted flow checkpoint](affine-selfhost-flow.md) records paired lexical,
move, exit, branch and loop decisions. Later nominal and selected-payload
children extend those checkpoints; none establishes the entire affine matrix.

I update the historical wording while leaving the full ownership row open.
The complete contract, both producers, IR transport, runtime behavior and
real-handle migration remain separate obligations, including
`task_c60a8d2e14b7494f8875e75b16e9b087` and
`task_28f2fb4b1f3c8a5ce93df628bb569d76`.

## Bounded passive acceptance

The live completed task `task_90b123edcc301b464a031c55e4ba1a11` requires
pure/effect-checked independence, deterministic serial reference semantics,
NanoISA eligibility records, agreement between both source frontends, and
serial VM/AOT behavior with unsafe dependency/effect/resource refusals.
It explicitly excludes scheduler optimization, async I/O, SoA and hardware
speedup claims.

I reconcile that bounded row against these merged layers:

- [Closed purity](closed-purity-foundation.md): transitive body inspection,
  mutable/aliased aggregate read boundaries, exact declaration/source identity
  and explicit extern refusal. An annotation alone is not an IR effect fact.
- [Passive records](passive-scalar-metadata.md) and
  [guarded scalar inputs](passive-guarded-inputs.md): verified ranges, dependency
  graphs, scalar provenance, versioned input guards, exact canonical transport,
  malformed claim refusal and serial VM/native execution.
- [Both par frontends](passive-par-frontends.md): closed scalar calls and
  immutable parameter inputs, independent bindings, effect/resource exclusions,
  exact producer code/metadata parity and native execution.
- [Both flow frontends](passive-flow-frontends.md): forward dependencies,
  stable source-ID topological order, cycle/duplicate/effect refusals and
  exact producer metadata. The integrated gate reports 86 baseline comparisons,
  85 emitter methods, six flow methods, five par methods and 20 metadata methods.
- [Owner calls](passive-owner-calls.md): ordinary calls around passive blocks
  preserve serial order without admitting effectful calls inside pure nodes.

The earlier record-only documents describe their own checkpoint limitations;
later frontend and call evidence supplies those bounded continuations. I do
not rewrite historical results as though every layer existed from the start.

This satisfies the written closed scalar passive slice. Broader captures,
resource permissions, trusted foreign summaries and full applicable-language
coverage remain outside this completed child. I claim neither a parallel
scheduler nor hardware acceleration. I keep the full release parents open.
