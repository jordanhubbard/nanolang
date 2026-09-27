# My interpreter stack repair

MAC `task_e43e2a52806f4609bf05de6c41e263b2` tracks the stack-guard failure retained
in my preceding shadow-termination evidence. I move builtin dispatch into a
helper that returns before a user function body begins, preserving dispatch
order, argument evaluation and the declared `array_push` override. I dispatch
calls, borrows and prefix operators before entering the remaining expression
handler. I do not change the shadow deadline or skip shadows.

The same final regression fixture fails all three methods with the preceding
fully instrumented C seed: direct and function-value recursion at depth 128
crash, and the deliberately failing shadow cannot reach its assertion. With
the repair, all three pass. The assertion failure preserves the prior artifact.
My intermediate builtin-only split still crashed; I retain its terminal. The
first complete split reached the correct assertion diagnostic but the new test
expected the wrong text; I retain that terminal and correct the expectation to
the existing named-shadow failure message, explicitly excluding signal failure.

Both C seeds use Homebrew LLVM at `-O0` with ASan/UBSan on every compiler C
object. External libraries and generated module objects remain ordinary, and
leak detection stays disabled as in the hosted sanitizer worker. My retained
Darwin arm64 disassembly shows `eval_call_impl`'s reservation decreasing from
43,424 to 21,760 bytes and expression dispatch from 22,400 to 400 bytes, before
alignment/save overhead. This is a measured build property, not a portable
bound or a promise of unlimited recursion.

The full instrumented compiler run passes all 898 selected shadows with the
unchanged 60-second shadow deadline and successfully emits Stage 1. Total build
wall time is 131.70 seconds; that includes module and native C compilation and
is distinct from the supervised shadow interval. This passes the local failure
case. It does not qualify the separate Linux timeout or all hosted partitions.

Compressed logs preserve exact bytes, with uncompressed hashes in `logs.json`.
The ordinary `make test-eval test-interpreter-recursive-stack` owning gate
passes, including fresh native bootstrap, every interpreter unit control and
all three new methods. The native stage binaries differ; both pass the configured
smoke test. I do not present that as canonical bytecode fixed-point evidence.
All nine sanitizer inventory methods pass. Final-source fixed points and
complete release gates remain required.
