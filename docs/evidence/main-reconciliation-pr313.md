# My main reconciliation boundary through PR #313

My integration branch already contains main through `b37136cc`, via merge
`26397722`. I inspected eight later commits through `931517fb`. A read-only
merge preview reports nine conflicting files. I have not completed that merge.

The two latest commits concern uncalled record-parameter functions (#311) and
projected array fields (#313). I ran independent native probes before adopting
their changes. My existing dynamic representation passes the eleven-field
array-record probe. The uncalled function fails at `AGG_PACK` because its
unconstrained parameter has no concrete storage shape.

I now compute a function-level call closure rooted at entry and the first
`__init__` function, matching my VM's two invocation roots. I follow direct
calls and tail calls. I require concrete packed-field layouts for this closure.
I omit an uncalled function whose packed layout remains unresolved, then omit
its uncalled callers to avoid dangling C references. I retain representable
functions: dropping every uncalled body broke 101 existing native boundary
subcases that independently exercise generated helpers. I retain whole-module
structural, instruction and type checking; this is not permission to hide
malformed or unsupported bytecode in uncalled functions. Instruction-level
dead-call pruning remains conservative.

The initializer root matters: a call from `__init__` can initialize a global
without appearing in entry's call graph. My regression exercises that helper
call and asserts the resulting global value. The incoming main algorithm only
starts from entry, so taking it unchanged would lose this integration behavior.

`test_incoming_main_record_shapes_execute` exercises four standalone C11
compilation/execution probes: uncalled record parameter, array-valued record
field, initializer-only call chain and uncalled direct/tail-call wrappers.
The packed-field unit cases now
distinguish uncalled unknown layouts from reachable unknown layouts; conflicts
remain rejected rather than assigned guessed representations.

The final `make -j1 test-nvm2c` passes 1,673 translator checks and 1,073 shape
checks. The complete compiler-to-native module passes all 22 methods, including
the existing native helper/tag/representation probes. Fresh
`make -j1 test-nvm2c-sanitizers` repeats the translator and shape suites with
verified ASan/UBSan instrumentation and passes. That runner disables leak
detection; these results do not establish leak freedom.

The 2026-09-16 GitHub inspection returned no open issues and 21 open PRs. None
had a release label, milestone or explicit 5.0 title/body scope. My creator's
all-branches request still requires their reconciliation. I have not closed
them to satisfy a label-only gate. Query snapshots are recorded locally as
`/tmp/nano-release-open-prs-20260916.json` and
`/tmp/nano-release-open-issues-20260916.json`; these are observations, not a
claim that remote state remains unchanged.

Remaining reconciliation includes portable write-failure injection and the
other overlapping main changes. MAC `task_cffdafd16e641ac417ccfddb962534b9`.
