# Hosted full-suite budget

I keep every full-suite test and give its hosted invocation a finite budget.
At candidate `86aa2006`, my Ubuntu ARM job
[104950451685](https://github.com/jordanhubbard/nanolang/actions/runs/35142570442/job/104950451685)
entered `test-impl` at 2026-09-16 19:49:52 UTC and exhausted its 1,800-second
make deadline at 20:19:52 UTC:

```text
make: *** [Makefile.gnu:2382: test] Alarm clock
```

My Jackson Forth checks were still progressing. Memory-Allocation reported
zero errors and Locals began immediately before the runner stopped. This was
the make deadline, before the job's former 45-minute limit.

I now invoke the full hosted suite with `TEST_TIMEOUT=3600` and cap its job at
75 minutes to include the JUnit corpus and guide validation. I retain all
steps, matrix platforms, test selections, instrumentation and shadow budgets.
My normal make default remains 1,800 seconds. These limits do not turn a
timeout into success; fresh exact-head CI must complete its tests and steps.

Focused validation parses the workflow and confirms that only this job limit
and the full-suite command change. The previous successful coverage and
sanitizer jobs do not establish acceptance of the next full-suite revision.
