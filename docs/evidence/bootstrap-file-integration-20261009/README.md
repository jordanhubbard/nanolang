# I bootstrap the compiler with File publication support

I repair two bootstrap integration boundaries under #982 on base `56f67be84`.
My current compiler imports `file_product`, but my bootstrap host-module list
omitted it. I add that declared module to host preparation and immutable input
retention. I also give the complete compiler shadow suite an explicit
30-second bootstrap deadline, with a validated 1–300-second environment/CLI
override and receipt consistency checks. Ordinary compiler and File-publication
defaults remain unchanged. I retain the previous default-deadline failure in
[my complete driver evidence](../file-complete-drivers-20261009/README.md).

## Actual Darwin bootstrap

I run `make -f Makefile.gnu bootstrap3` with Homebrew LLVM Clang, the installed
OpenSSL 3.6.5 prefix and `NANO_SHADOW_TRACE=1`. The complete Make invocation
exits zero. I execute both generations in NanoVM, verify their bytecode,
translate each native compiler, compile and execute hello programs, compare
raw Stage 1/Stage 2 modules, and install Stage 2. The final Make gate also
compiles a hello program with `nanoc_c` temporarily removed.

My two self-hosted modules are byte-identical: 630,624 bytes, SHA-256
`c4aa4320225675f5bb009624e7f4facc4f708a8df0286b0e130853cc715b7ed2`.
I retain the complete receipt, source/tool/host/artifact hashes, each command's
log and terminal status. `summary.json` gives the phase timings; the first VM
generation takes 573.511 seconds and the second 796.438 seconds. Their native
builds take 39.309 and 19.185 seconds. I observe no generated-native-code
violation marker. Declared native host artifact work remains allowed; this
guard is a build-provenance check, not a security sandbox.

My receipt records the 30-second shadow budget and the separate 1,800-second
per-command deadline. I retain exact six-library closure identity across seed,
Stage 1, Stage 2 and final verification. I do not normalize module differences,
skip selected shadows, substitute native generation for VM generation, or
compare native binary hashes as a fixed-point criterion.

## Boundary checks

Nine bootstrap boundary/native-guard tests pass. They cover the actual File
host inclusion and retained snapshots, allowed native host compilation,
generated-product refusal, changed sources/tools/host/artifacts, inherited and
explicit deadline settings, child environment propagation, receipt mismatch,
and native link flags for instrumented runtime objects. Two deadline tests pass
again after I bound oversized numeric input before conversion. Thirteen
installation-message and Make dependency tests pass after their synthetic
receipt includes the required deadline setting.

I retain those test logs alongside the actual bootstrap logs. The native links
report duplicate `-lm` warnings; they exit successfully. This is an incremental
Darwin bootstrap, not clean-environment reproducibility, Linux qualification,
or a proof of compiler semantic correctness.

## File execution through the self-hosted generations

I run the existing complete eleven-method `NanoServiceDriver` corpus separately
with each generation's module and native executable. Each method exercises
both compiler forms. I set `NANO_SHADOW_TIMEOUT_SECONDS=10` for these runs;
the bootstrap's larger compiler-suite budget does not carry into normal File
publication checks. All eleven methods pass for Stage 1 in 212.073 seconds and for Stage 2 in
211.231 seconds. I retain separate terminal logs and the runner results; the
receipt identifies both modules, both native compilers and their host closure.
I recheck all recorded source, tool, host and compiler artifact hashes after
the File tests.

GitHub DNS failures prevent publishing this batch and refreshing remote CI or
worker status. File indirect execution, Linux/Darwin full service equivalence,
the remaining compiler/backend/network requirements and the full 5.1 release
remain open.
