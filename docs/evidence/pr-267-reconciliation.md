# My PR #267 reconciliation

PR #267's head is `d64a7c11dbedbc30ff659780ef037b267c4ad166`.
Its actual parent, `10f9832b`, is already an integration ancestor. The task
title names concurrent compiler isolation; the four-file patch instead adds
failure rejection after external-assembler capture has been admitted.

I reviewed those changes against integration's `62c1f1ae` and its successors.
My current builder records admission before capture changes the selected mode
and rejects failed capture for every admitted mode, not only external Clang.
It retains the diagnostic explaining why I refuse live-source compilation.
My cache context is newer than the PR's v32 context. I do not replace it or
restore the PR's suppressed assembler diagnostics.

The current regression checks are stronger than the PR's tests: they inspect
cold publication, warm-generation preservation, object invocation counts,
staging cleanup and recovery under both cache roots. They do not require an
exact Apple compiler version string merely to select the failure test.

I retain current source and tests and add the PR head as an actual merge
ancestor. Only the roadmap and this evidence document change.

## Verification

Six methods pass in 114.378 seconds on macOS:

```sh
python3 -m unittest tests.test_selfhost_build_isolation \
  tests.test_source_snapshots.SourceSnapshots.test_apple_failed_capture_refuses_uncaptured_cold_output \
  tests.test_source_snapshots.SourceSnapshots.test_apple_external_query_failure_and_recovery
```

The four isolation methods require live overlapping compiler processes,
distinct executable outputs, untouched legacy scratch symlinks, source-only
compilation without a temporary directory, prior-output preservation and
cleanup after failures. The two capture methods require fail-closed cold and
warm builds for empty, multiple, truncated, oversized, failed and timed-out
queries, then actual recovery and cache reuse.

The log is `/tmp/nanolang-pr267-gates.log` on this host. This is not a fresh
Linux acceptance run or a claim that every compiler/cache concurrency path
is complete. PR #267 stays open until integration lands on main or is explicitly
superseded there. MAC `task_443e8107d0ff4350999e0d5186a809f1` remains stopped;
I attach evidence without reopening work merely to force ledger closure.
