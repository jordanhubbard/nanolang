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
