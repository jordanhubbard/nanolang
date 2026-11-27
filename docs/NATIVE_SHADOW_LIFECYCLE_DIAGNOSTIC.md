# My bounded Darwin shadow lifecycle observation

I track this diagnostic under #1253/#1065. My current
[original emitter refusal](evidence/c-repairs-linux-20261126/darwin-current-native-failure-45970514720/README.md)
completes all1,139 selected shadow assertions in its bounded trace, then records
a parent deadline failure. The existing markers do not establish when the
process became waitable. I preserve that failed qualification.

My separate branch-scoped `C seed shadow lifecycle diagnostic` workflow runs on
a hosted Xcode27 worker. It restores the exact saved6e09 artifact ZIP only after
checking its digest. All original non-document product Git entries must match;
I verify the actual compiler/runtime images, source inputs, generated index,
six host libraries and selected external tools before using the saved compiler.
I copy the exact diagnostic source and toolchain records into its output.
Restoring a saved image does not create a new bootstrap qualification.

My small native observer first passes three real child-process controls:
successful exit, a marker followed by a deliberately late clean exit, and a
marker followed by blocking until the control parent kills and reaps its child.
I require a live-process sample, stable birth identity, an independently checked
wait status and the expected process-event/timeout distinction. A failed
control stops before the compiler observation.

For the compiler observation, I register a process event for the actual shadow
child after its existing dispatch marker and remain idle through shadow bodies.
The existing flush marker triggers a bounded three-second observation of
`proc_pid_rusage`, BSD process state and `EVFILT_PROC/NOTE_EXIT`. The observer
neither signals nor reaps that child. I retain raw absolute/monotonic clock
brackets, birth identity, syscall errors and process-event dequeue times.
The compiler retains its ordinary ten-second child/parent supervisor and its
separate300-second command bound. Only one compiler observation runs.

I interpret those records within these limits:

- My resource and BSD-state calls are separate bracketed reads.
- My process-event dequeue time bounds when I observed the event; scheduling
  gaps are still uncertainty.
- A cached `ri_proc_exit_abstime` alone does not prove completed teardown or
  waitable exit. The inspected Apple source assigns it before later termination
  and zombie transitions. I must qualify the actual target behavior.
- Sampling and trace output can perturb timing. A diagnostic is not a passing
  original emitter gate or an exclusive wall-time attribution.

My API references are Apple's
[libproc declarations](https://github.com/apple-oss-distributions/xnu/blob/main/libsyscall/wrappers/libproc/libproc.h),
[resource query](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_resource.c)
and [exit path](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_exit.c).
Those reference sources are not an identity claim for the hosted kernel.

My workflow uploads the exact source/compiler ZIP, controls, native observer,
full compiler trace, lifecycle records and before/after identities. Missing
flush/event observations fail the diagnostic boundary. No source compiler,
supervisor decision, original test assertion, qualification workflow or release
selection changes. My original emitter12 and all69 release requirements remain
open until their actual gates pass.

## My first calibration refusal

I retain run45982190971: the native successful control exits cleanly and its
NOTE_EXIT receipt follows its marker0.192ms later. Its marker occurs178.168ms
after the pre-fork origin, outside my mistaken pre-marker100ms assertion.
No compiler observation runs. I measure success/late exit tails from the
actual marker: success remains below100ms, late remains at least150ms.
The marker retains its20ms minimum, controls retain their5s command bound,
and all product300/10 boundaries remain unchanged. This diagnostic is not
qualification.

I retain [both native control refusals](evidence/c-repairs-linux-20261126/lifecycle-control-refusals-20261127/README.md).
The second control starts its blocked window before the child marker and kills
the child without a marker after102.881ms. No compiler observation runs.
I correct this in isolated diagnostic branch
`work/5.1-cseed-lifecycle-controls-20261127`, source5ca56494a9dc8eb84c66da61402343a2a34994f2.
I arm before releasing the owned child and trigger after its marker, preserving
all control and product bounds and strengthening ordering/duration checks.
Native control qualification is pending in run45991837855. The diagnostic
branch retains the exact restored original C product tree; I do not relax its
guard for newer pure-helper source additions on my release branch.
