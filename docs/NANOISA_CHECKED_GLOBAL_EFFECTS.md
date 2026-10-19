# My invocation-local checked global effects contract

I own #998's `fix/5.1-checked-global-effects` fromdfc8708b6. My primitive
worktree remains untouched; this is a separate compiler advisory-metadata task.

## Boundary and interfaces

Canonical `nanoc_v06` currently checks the merged parser, emits program text,
then emits and executes selected shadows before publishing. Its legacy union
merge branch is dormant because import_count is fixed at zero. Actual globals
are built by `typecheck_phase_with_shadows` using bound declaration names; the
current advisory query discards them by supplying globals=[]. Immutable scalar
reads therefore report UNKNOWN instead of0 and mutable reads UNKNOWN instead of2.

I factor a checker core with an optional invocation-local summary sink. Existing
public checking wrappers retain their behavior and request no summaries. A
checked-summary helper returns the normal TypecheckPhaseOutput plus bounded
per-original-function masks, function identity snapshots and owner epochs. It
computes these only after all diagnostics, resource checks and selected-shadow
checks succeed, using the actual bound Symbol array while that array is local.
No Symbol array or persistent Parser field is retained.

Codegen owns the public check-and-emit bridge. The bridge itself invokes checking;
it never accepts trusted masks from its public caller. It synchronously consumes
its internal summary result to generate program and optional selected-shadow
text, returning only phase diagnostics, assembly strings and lowering failure.
The canonical callsite invokes this bridge once, preserving diagnostic remap,
selected-shadow execution, output preservation and existing error distinctions.
No summary context escapes that invocation, including on early failure.

Internal module/shadow helpers accept the scoped summary value explicitly.
A private function-header helper accepts one resolved mask; existing raw
nisa_emit_function continues its old no-global analysis. Only original ordinary
queue entries with exact function owner identity, saved owner epoch and original
index receive checked masks. Generic-active, specialized, lexical lambda and
synthetic functions remain UNKNOWN where already unsupported; no function name
or body-index match substitutes for identity. Empty checked summaries mean
UNKNOWN (bounded refusal), distinct from absent context in a raw wrapper.

Existing raw parser/program/shadow/bound wrappers retain no-global inference.
The standalone nanoisa_emit driver currently parses and calls these wrappers;
it does not gain actual-global facts merely by being a producer. A separate
checked entry may be qualified later, but this change does not silently alter
standalone checking/admission. The ordinary publisher's bound flag is not evidence
of checked effects.

## Mutation audit

The active canonical check-to-emit path has no semantic parser mutation. Emitter
state resets, queues and local generic/lambda ASTFunction copies are not parser
rewrites. Production codegen and imported ownership/borrow/shape helpers must be
checked for indirect calls into check_function/check_expression or semantic type
inference, not only direct parser setters. If a path can recheck or mutate shared
semantic nodes, it receives UNKNOWN or is excluded until the invariant is closed.
Program text is completed before shadow generation; both consume the same exact
checked parser synchronously. The context is never cached for a later invocation.

## Work and failure bounds

I preserve existing raw behavior and declared-pure validation. For the new
checked context, oversized modules short-circuit UNKNOWN before per-function
scans or snapshot allocation. At most64 analyzed bodies, bounded total function
and global entries, and at most4096 analysis steps per function are accepted.
Global lookup visits consume the advisory budget; exhausted work returnsUNKNOWN.
The declared-pure path uses an empty budget and keeps its previous semantics.
Bounds must cover preparation/lookup loops as well as recursive expression visits.
No effect mask authorizes optimization, execution, ownership or nominal admission.

Failure returns normal diagnostics with no program/shadow text. Lowering failure
retains the original diagnostic distinction and never publishes output. An
ordinary global's checked name/type/mutability is authoritative; AST spelling,
annotations and mutable process-global caches are not substitutes.

## Acceptance

I require original seed and changed Nano immutable-global0/mutable-global2 masks
with unchanged VM/native outcomes; calls through multiple bodies; local shadowing;
imported same-spelled bound globals; copied/unrelated function identity refusal;
failed check and lowering paths preserving prior output; checked then raw and
checked then failed invocations without leakage; generic/synthetic UNKNOWN;
unchanged raw no-global masks and affine-use facts; exact small work-limit
refusal/recovery; and oversized-module fast UNKNOWN. Selected-shadow diagnostics
and execution remain separate from program text creation. Fresh raw-generation
and target gates remain aggregate work, not inferred from a C-hosted build.
