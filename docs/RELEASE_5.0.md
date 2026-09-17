# Published 5.0.0 — My language contract and runtime boundaries

I published the audited compiler/runtime branch together with the fixes on main.
This is a major language-contract release, not a claim that every planned
5.0 architecture milestone is complete.

## Full-roadmap scope selected after publication

My user subsequently selected the full 5.0 roadmap, including NanoISA-only
compilation and matching compiler bytecode. This document records the narrower
published artifact; it does not replace the acceptance criteria in
[my roadmap](ROADMAP.md) and [One IR contract](NANOISA_ONLY.md). Those criteria
remain mandatory. I preserve this tag as history and will publish new evidence
only after the full contract passes. My next release, v5.1.0, must satisfy
that full contract; it is not limited to the native map lifetime repair.

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

I execute synchronous effect handlers in my native C source backend and NanoVM.
A final handler expression resumes the perform; an explicit return exits its
lexical function. Native C rejects nonlocal returns across foreign callbacks;
NanoVM does not unwind across externally entered callback activations. My
`nvm2c` AOT translator rejects effect opcodes. I record the tested contracts in
[my native effect evidence](evidence/native-effects-linux.md) and
[my VM effect evidence](evidence/vm-effect-dispatch.md).

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

On 2026-09-16 I reviewed the open pull requests and issues by title, body,
labels, milestone, ancestry and relevant diffs. Only candidate PR #336 explicitly
named this release in its metadata. My refreshed
[scope snapshot](RELEASE_5.0_SCOPE.json) records the reviewed heads, dispositions
and candidate revision after main incorporated PRs #353 and #357. Superseded PR closure
remains conditional on #336 landing with accepted gates and unchanged heads.
The snapshot is a dated review; the published release evidence must also record
the final queue check.

I incorporate `PUSH_F64`, numeric comparisons, float locals and direct/tail-call
argument transport from the reconciled comparison work. Exact-bit constants
preserve nonfinite inputs, and comparisons retain NanoVM equality and ordering.
This does not establish float arithmetic, float result or aggregate parity.
The focused reconciliation passes 1,761 translator and 1,076 shape checks;
[its evidence](evidence/main-reconciliation-pr357.md) states the tested limits.

I preserve unfinished native AOT parity work. Array-read branches also add
`CAST_BOOL` and, in some branches, `PUSH_VOID`; their common missing-read repairs
alone do not justify closing those branches as superseded.
I preserve the newer tagged and owned representations when reconciling older
branches. My source-snapshot boundaries remain in
[the snapshot record](SOURCE_SNAPSHOT_EVIDENCE.md).

## Validation checkpoint — 2026-09-16

My bounded finalization checkpoints passed 1,745 native translator checks,
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
run passed 47 methods, with eight platform/compiler-specific skips. The Darwin C++ fixture passed
compilation, reuse and header invalidation; its deprecated `.c`-as-C++ warning
is acknowledged for that fixture, not described as warning-free.

At `094cfa80`, a fresh Linux build passes and the full test run
passes the translator, shape, VM and 175-program verifier/equivalence gates,
then stops on two obsolete OPL exclusions after all 225 corpus programs pass.
The repaired VM example gate now covers 244 eligible sources and four verified
exclusions. Native array search also passes a reproduced x86 argument-order
regression, and all 27 FFI tests pass with default sanitizer ODR checking.
Darwin compiles all 187 targets selected after installing the actual OpenGL
and readline dependencies, including the nine targets absent from the earlier
178-target selection. Five example regressions pass. These are compilation
results, not a graphical runtime claim. All 12 document-pair tests pass,
including real authoring regeneration. The hosted sanitizer build and bootstrap
pass with an explicit bounded shadow budget. I correct the strict CI dependency
list to install GLFW and GLEW; automatic package installation stays disabled.
My public compiler version header and package metadata now both identify 5.0.0,
and release tests keep future versions synchronized.

These are dated results from the integration work, not a claim that the final
full-suite and CI gates have already completed. My published release evidence
must identify their tested revisions and final outcomes before publication.

[The integration evidence](evidence/main-reconciliation-pr334-linux.md) records
commands, revisions, fixes and limitations. Final clean-tree tests, platform CI,
and release acceptance are mandatory gates for the exact commit to tag. This
document records bounded checkpoints. My GitHub release carries the final
validation record, identifying that revision, completed gates and final scope
review; I publish it only after those gates pass.

## Presentation

My repository contains a regenerated local 5.0 release-edition deck and narrative.
They describe the language/runtime release boundary and unfinished architecture.
The existing Google links still identify the published 4.5 edition; local
regeneration does not update those external artifacts. Their status and existing
publication IDs are recorded in [current deliverables](presentation/current-deliverables.md).
