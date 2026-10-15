# I attribute bounded component shadow execution

My d6ec53cd7 Linux [x64 build](https://github.com/jordanhubbard/nanolang/actions/runs/37957439641/job/113911425640)
and [Docs build](https://github.com/jordanhubbard/nanolang/actions/runs/37957439641/job/113911425696)
stop nanoisa_emitter shadows at 60 seconds. The
[sanitizer job](https://github.com/jordanhubbard/nanolang/actions/runs/37957439641/job/113911425633)
stops typecheck shadows at that deadline. A cold stage1 links bin/nanoc to the
C seed; a bootstrapped tree selects the self-hosted product. I do not use one
route's timing as qualification for the other.

I add opt-in, flushed JSON start/completion records under NANO_SHADOW_TRACE.
They retain source path, declaration line, target, monotonic elapsed seconds
and assertion failures. A missing clock yields -1, not an invented duration.
The existing outer supervisor still owns exactly the same whole-suite deadline.
I enable this existing trace option in CI, retain component logs per worktree,
and upload them through the existing failed-build artifact steps.

`make -j4 test-cseed-shadow-trace` passes all three regressions (`tests.log`):
success and opt-in behavior, assertion failure with completion, and a real
one-second timeout retaining the last start without a false completion. Both
failure routes preserve the existing output. Quoted filenames round-trip through
JSON. This is attribution and retention, not a repair or waiver of the CI timeout.

## I size CI's finite deadline from retained progress

The next [Linux trace](https://github.com/jordanhubbard/nanolang/actions/runs/37960808642/job/113922857890) reaches nb_union_layout_bytes after passing the preceding shadows. Thirty-one matching completed shadows have a median Linux/Darwin duration ratio of 2.041 (range 1.891–2.185); the full local 661-shadow baseline sums to 44.955 seconds. I retain the individual comparisons in ci-timing-comparison.json. This is evidence of aggregate runtime exceeding the old 60-second budget, not successful completion on Linux.

I select 120 seconds for ordinary CI and 300 seconds for sanitizer/coverage jobs, which run instrumented builds. Both are within the existing 1–300 second opt-in range. Compiler defaults, selected shadow sets, assertions, prior-output preservation and failure behavior are unchanged. The new CI run must still complete successfully.

## I discard an unconvincing field-lookup optimization

A fixed per-thread field-slot hint passed the full evaluator suite, layout-change controls and all 661 emitter shadows. Its first run summed to 41.549 seconds, compared with the initial 44.955-second baseline. Rebuilding the baseline against the identical other C objects and repeating it yielded 42.010 seconds. I retain the comparison in field-hint-comparison.json and remove the candidate: the roughly one-percent difference does not justify additional interpreter state. No field-cache code or test-only hook remains in the product.
