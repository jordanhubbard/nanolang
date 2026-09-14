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

## What I do not claim

I still use my C-transpiling path to build the compiler. The NanoISA-only
bootstrap in [my architecture plan](NANOISA_ONLY.md) remains a target, not
the implementation shipped here. Packaged NanoVM execution is not native AOT.
Backend parity, complete resource ownership checking, broader input snapshots,
and production service isolation remain bounded work on [my roadmap](ROADMAP.md).
Laboratory frontends do not establish a distributed production runtime.
In particular, bytecode `array_slice` currently treats its third argument
as an end index, while the C path treats it as a length. Nonzero-start
slice parity remains follow-up work; this cut does not claim that parity.

## Release review

I reviewed every open GitHub issue and PR by title, body, labels and milestone.
There were no open issues and twelve unscoped PRs. I leave those PRs visible;
I do not close them merely to empty the queue.
[The scope snapshot](RELEASE_5.0_SCOPE.json) records that review.

I reconcile main's source-provenance fix with the newer module binder, retain
the immutable cache implementation instead of restoring its obsolete public
hash-cache API, and preserve main's source-identity and alias-shadow regressions.
My release evidence distinguishes this integration run from earlier cache
fault-injection runs in [the snapshot record](SOURCE_SNAPSHOT_EVIDENCE.md).

The user explicitly authorized release despite the MAC fleet dispatch hold.
That hold can prevent task claims and closure; it is not a compiler test.
I do not clear it or report held tasks as completed.

## Validation

On Darwin arm64 with Apple clang 21.0.0, my clean build passed. The release
test run exposed two stale shadow fixtures: missing assembly result fields
and an uninitialized emitter-local environment. I corrected both fixtures;
the NanoISA module gate and the 86-check source-emitter comparison pass.
The AOT suite passes 375 checks. Stage-1 self-hosted source provenance and
nine module-binding tests pass; the cross-backend language-claims suite
passes 17 tests, and C-seed import shadows pass nine.

The language-claims rerun explicitly uses the freshly built stage-1
self-hosted compiler. An initial invocation failed because it expected the
stage-2 binary removed by cleaning. I do not count that as a stage-2 bootstrap.
Some native links emit an Apple SDK text-stub warning; I do not describe
those links as warning-free.

The complete test gate is still pending while this release is prepared.

A subsequent integration scan reported 205 passes and 13 failures. The
failures exposed dormant dependency shadows and two missing execution paths.
I corrected filesystem imports, mutable-map expectations, floating-point
tolerances, JSON length spelling, OPL fixtures, and property-counterexample
expectations. I added interpreter support for mixed static/dynamic nested
arrays and epoch-millisecond timing, and preserved nested-array literal tags
in bytecode. The affected programs pass direct retests. My new nested-array
regression checks empty inner arrays, three levels, aliasing, appends and
indexed writes through C compilation/shadows and NanoVM with tracing disabled.
Opcode tracing isolated the separate slice-convention limitation noted above.

## Presentation acknowledgement

I retain the existing **4.5 edition** deck and narrative as historical
artifacts. I have not rebuilt or republished them as 5.0. This release note,
the canonical style guide and the roadmap describe the 5.0 boundary instead.
I acknowledge the presentation freshness gate on that basis. Google Drive
publication remains a separate, explicitly authorized operation.
