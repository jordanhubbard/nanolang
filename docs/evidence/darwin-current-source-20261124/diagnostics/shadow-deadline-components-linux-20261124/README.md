# My late-reap deadline repair checkpoint

I repair [#1249](https://github.com/jordanhubbard/nanolang/issues/1249) at source `386cf15309a524078bf1ad2ee36566d65ea43735`. My [unchanged actual C-seed counterexample](../shadow-late-exit-linux-20261124/README.md) publishes a native product after a late shadow finishes under a one-second deadline. Two new regressions then reproduce late acceptance in both my actual C seed and a compiled consumer of the shipped runtime header (two failures, zero errors).

I check my monotonic deadline even when `waitpid` has just reaped the child. My C seed retains the resulting clock/timeout flags for its existing refusal path. My runtime runner refuses on clock failure or expiry, and does not kill/reap a child already returned by waitpid. Receipt, clean exit, clock-error and original budget guards remain. I append my two regressions to the existing completion-observation Make cohort without removing either original module or changing assertions/deadlines.

## Actual component qualification

I build all four tools from zero provider copies in fresh normal/SAN/O3 roots. Each mode passes the complete original completion-observation/tracing cohort plus my two regressions (12 methods), C-seed import shadows (10), and VM/runtime supervision (6). These are 26 original methods plus two new controls per mode, with zero skips. The new C-seed control preserves prior output, records late completion and a clean child exit, and requires the parent to refuse with its original one-second control deadline. The runtime control likewise requires refusal. I do not alter ordinary default10 or any bootstrap budget.

| Mode | Four-tool build seconds | Available original/new groups | Required Stage2/native gates |
|---|---:|---|---|
| normal | 47.068 | 12 + 10 + 6 PASS | original wrapper FAIL: missing Stage2 |
| san | 154.617 | 12 + 10 + 6 PASS | not qualified: fresh Stage2 still required |
| production | 106.719 | 12 + 10 + 6 PASS | not qualified: fresh Stage2 still required |

Source, C-seed pre/post image and system-tool/library guards match. All native tools retain their actual ARM64 ELF images. The sanitizer build uses the original ASan/UBSan flags and leak detection; the new expected-negative cases and complete wrapper stderr contain no unexpected sanitizer report. I retain source snapshots, four-tool outputs, original streams, new regression products and source/native identities. Original fixtures retain their own temporary-file cleanup; I do not claim that every original temporary product survives.

## Retained incomplete gates

My first complete cold normal build and two new controls pass, but the original observation class refuses setup because `.test_output` is absent. I preserve that full attempt and add only directory creation to the existing Make recipe. In a new cold normal root the 12/10/6 cohorts pass; the broader original `test-native-shadows` wrapper then fails with22 missing-provider errors because `bin/nanoc_stage2` is absent. Its emitter companion also requires this fresh Stage2 and is not run afterward. I preserve the failed wrapper, do not filter its errors into a pass, and do not supply an older Stage2. SAN/O3 qualify only the complete available 12/10/6 cohorts in new roots; original Stage2/native-emitter qualification remains required.

I will produce fresh exact-source Stage2 through the existing Make bootstrap with its declared1800/30 budgets, then run those original native gates. That separate bootstrap cannot change the [retained ordinary full600/default10 source refusal](../fresh-record-adoption-linux-20261124/README.md): all1306 shadows complete with zero assertion failures, but the parent refuses at its deadline and publishes no seed. I make no unchanged full-source retry or historical cause/performance claim. Exact paired Darwin, private full-source, raw/bootstrap/installed and full69-entry release acceptance remain open. #1249 remains open for its complete route/platform gates.

## Independent read

I retain 24863 file records across the red regression, missing-output setup, failed required normal wrapper, and complete SAN/O3 component roots. My manifest SHA-256 is `0cf805018fde466b9d908e328fc01bab1fe3216b2ab22e463a81de78f7c22b5b`. Chunk references bind all unchanged parent archives and actual compressed/raw bytes. Everything stays in the existing excluded host-local evidence path.

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/shadow-deadline-components-linux-20261124/read_archive.py
```

My reader verifies all bytes, selected source inputs against the exact repair commit, unchanged original guard inputs, actual native architecture/images, red and green late-reap outcomes, prior-output preservation, complete 12/10/6 reports, and the explicit failed/missing Stage2 boundary. It requires the referenced archives and Git objects and does not claim a hermetic operating-system/header archive.
