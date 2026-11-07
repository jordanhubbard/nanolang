# Combined fast-gate budget measurement

I fail the complete same-worker budget at exact source `6959a9762c4aa18b4d9b12bec06b85d09ae1266a` in [run 42406792194](https://github.com/jordanhubbard/nanolang/actions/runs/42406792194). The runner annotation explicitly reports the configured 25-minute measurement limit. Final job duration is 1,517 seconds, exceeding the unchanged 1,200-second acceptance budget. My [terminal summary](terminal-summary.json) and `terminal-run/` retain exact job/step states, limit annotations, resources, selected compiler identities, the full Make log and partial full-command report.

My actual complete Make path runs with `FAST_GATE_TEST_PROFILES=1` at the ordinary four-worker default on one observed four-CPU host. Lexer, prerequisite and original refusal controls pass. Clang completes all 1,727 compiles with zero failures in 741.728 seconds. GCC is still compiling when the job limit cancels it, and the final bootstrap-tool control is not reached. I do not infer a completed GCC result from partial progress. The report still lists 230 unreviewed C files.

The opt-in switch preserves the default while full C-profile coverage and budget acceptance remain open. My [actual Make previews](make-switch.json) verify both selections without executing compilers. I keep every ordinary/sanitizer/debug/Fortify profile. I need a measured performance/parallel-execution repair plus the remaining coverage before promotion; increasing a timeout or dropping work does not meet the criterion.

This is a demonstrated whole-job budget failure with retained partial diagnostics. It does not establish a C correctness failure or a native lifecycle receipt. Existing native/qualification handles remain untouched.
