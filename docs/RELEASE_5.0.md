# 5.0 — My language contract and runtime boundaries

I ship the audited compiler/runtime branch together with the fixes on main.
This is a major language-contract release, not a claim that every planned
5.0 architecture milestone is complete.

## What changes

- I use `return` to leave the enclosing function, including inside match
  arms. An expression arm, or the final expression of a block arm, supplies
  the match value. See [canonical forms](CANONICAL_STYLE.md).
- Normal compilation runs dependency shadows before root shadows by default.
  `--root-shadows-only` explicitly narrows that scope; `--test-imports`
  restores the default. Source-only C emission does not execute shadows.
  Test-process deadlines are supervision, not a security sandbox.
- I preserve module function identity across aliases and lexical shadowing.
  Self-hosted import flattening retains source in memory and maps diagnostics
  back to original file lines.
- I separate compiler output from source and diagnostic destinations, reject
  failed imports before publishing bytecode, and link packaged interpreter
  wrappers with their actual runtime dependencies.
- My native module builder publishes private, immutable artifact generations.
  Supported C/assembler modes compile retained inputs, with bounded external
  capture and failure recovery. Exact strict-aliasing options retain snapshot
  eligibility. Unsupported modes are not covered by a universal snapshot claim.
- Incremental C builds track included headers, fixing issue #211.
- I include the Scheme, ML, Actor, Dataflow, Object, Shell and Logic laboratory
  frontends, plus expanded NanoISA verification and AOT coverage.

## Runtime boundaries

I implement a retained native callback ABI with explicit signatures, owner-thread
execution, cancellation and shutdown. My NanoVM bridge supports the dispatch
adapters and SDL_mixer post-mix lifecycle described in
[my callback contract](CALLBACK_ABI.md). My C seed selects the shared VM bridge
for imported callback shadows. Callback-bearing isolated imports remain rejected;
this in-process bridge does not establish co-process callback support or
production isolation. SDL audio-lock restrictions remain an unsafe boundary.

I use start/length semantics for `array_slice` in both backends, with overflow-safe
clamping. I preserve nested-array tags and expand native record/array lowering.
These repairs do not establish complete backend parity.

## What remains unfinished

I still use my C-transpiling path to build the compiler. My
[NanoISA-only bootstrap](NANOISA_ONLY.md), matching Stage 1/Stage 2 `.nvm`
artifacts, complete resource ownership checking and production service isolation
remain roadmap work. Native compiler acceptance is a distinct test from a
self-hosted bytecode fixed point. Packaged NanoVM execution is not native AOT.
Laboratory frontends do not establish a distributed production runtime.

## Release review

On 2026-09-16 I reviewed all 40 open pull requests and zero open issues by title,
body, labels, milestone, ancestry and relevant diffs. Only the candidate PR #336
explicitly named this release in its metadata. Thirteen open PR heads already
occurred in the candidate's ancestry. Repeated fleet integration branches need
semantic reconciliation, not blanket merging or closure.
[My scope snapshot](RELEASE_5.0_SCOPE.json) records each disposition and the
reviewed candidate SHA. It is a dated review, not a claim that the queue cannot
change. I must refresh it before publication.

Native floating-point comparison lowering in PR #310 and its integration
successors is only partly superseded: the reviewed candidate rejects `PUSH_F64`.
I retain that limitation in the scope record rather than claim full AOT parity.
I preserve the newer tagged and owned representations when reconciling older
branches. My source-snapshot boundaries remain in
[the snapshot record](SOURCE_SNAPSHOT_EVIDENCE.md).

## Validation checkpoint — 2026-09-16

My bounded finalization checkpoints passed 1,739 native translator checks,
1,073 shape checks, 272,403 VM checks, 89 VM codegen checks and 175
verified/equivalent programs. A clean bootstrap passed at `4373abc5`. Darwin
passed 14 effect tests, 49 scoping checks and 39 executable guide snippets.
An ownership ASan/UBSan run exercised 52,000 activations. Native and VM effect
execution are implemented; these checks cover their tested boundaries.
Strict Linux ARM64 acceptance passed all 185 selected native example artifacts
with unchanged selection and exclusions, plus five regression compilations from
root and examples working directories. The OPL and NanoAmp fixture repairs are
`b9c9854d` and `8b8822bd`. An earlier 242-program example sweep remains historical
evidence. These example checks do not establish the final full-suite gate.
The native translator and shape suites also passed their ASan/UBSan gate with
leak detection disabled. Native compiler acceptance passed 24 tests, and the
ordinary bootstrap passed its smoke checks. The native translator checkpoint
at `b21fbeed` reconciles main through PR #340 (`2711c6eb`) and retains both
PR #338 string-array regressions. Coverage-wrapper tests passed five unit and
seven integration cases against real gcov objects. The Linux cache-publication
run passed 55 methods with eight platform skips. The Darwin C++ fixture passed
compilation, reuse and header invalidation; its deprecated `.c`-as-C++ warning
is acknowledged for that fixture, not described as warning-free.

These are dated results from the
integration work, not fresh validation of every subsequent commit or proof of
semantic correctness.

[The integration evidence](evidence/main-reconciliation-pr334-linux.md) records
commands, revisions, fixes and limitations. Final clean-tree tests, platform CI,
and release acceptance are mandatory gates for the exact commit to tag. This
document records bounded checkpoints; the release evidence must identify that
commit and its completed gates before tagging.

## Presentation

My repository contains a regenerated local 5.0 release-edition deck and narrative.
They describe the language/runtime release boundary and unfinished architecture.
The existing Google links still identify the published 4.5 edition; local
regeneration does not update those external artifacts. Their status and existing
publication IDs are recorded in [current deliverables](presentation/current-deliverables.md).
