# Functional-array driver enum leak

I retain CI run 35967867927, job 107530646541, at `e70c0de46`.
Instrumented C-seed driver compilation fails with 10,136 leaked bytes in 77
allocations. `type_check_module` allocates enum variant values twice and loses
the first allocation. The original functional-array gate must pass with both
compiler and generated-program leak detection enabled before I close
`task_b3580edb93ad45c9a2233c6667ba0798`.

## Diagnosis and correction

I retain the checked enum-values allocation and fill that buffer instead of
allocating a second one. The environment still frees it. A fresh unoptimized
Darwin Clang ASan/UBSan compiler reproduces the exact hosted leak before the
fix. The optimized `-O1` control passes; that does not establish absence of the
leak in the unoptimized CI configuration. The enum-only correction passes the
original nine-method functional-array gate with compiler and generated-program
leak detection enabled.

Expanded enum coverage exposes a separate 40-byte leak in three generic-call
names. The checker allocates `AST_CALL.concrete_func_name`; I release it in
`free_ast` alongside the other owned call metadata. The separate task is
`task_9f20623802694b778d3e10dfb4fe6486`. My AST clone starts zeroed and does not
share this annotation. I retain the expanded suite's first failure.

Initial scratch-path compiler probes cannot locate runtime headers. Their logs
are retained as setup failures. The same binaries located under repository
`bin/` reproduce the imported-enum leak before correction and pass after it.
No test assertions, deadlines or leak settings change.

The retained units-01 log is a separate full-bootstrap shadow timeout at
60 seconds, tracked by `task_4251e719e9634aa7b597dbdd080b6b9b`. These allocation
repairs do not claim to resolve that timeout or complete hosted qualification.

## Combined qualification

All 35 original methods pass with both fixes: nine functional-array, eight
single-letter enum and eighteen generic-function methods. The C seed is a
fresh unoptimized Clang ASan/UBSan build, with leak detection enabled. Generated
functional-array programs retain their own ASan/UBSan and leak checks. The
generic and enum methods use the instrumented C seed; this run does not
instrument every generated generic/enum executable or requalify native stages.
The driver build retains its 180-second bound and the CI shadow allowance
remains 60 seconds. `final-O0.json` records source and compiler hashes.

MAC rejects direct completion from the tasks' current failed/running states.
I retain those responses; passing repository checks do not mean the ledger
tasks are closed.
