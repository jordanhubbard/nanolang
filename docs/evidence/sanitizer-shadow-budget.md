# My instrumented shadow budget

CI run `35132089446`, Memory Sanitizers job `104915293711`, built the
ASan/UBSan C seed and compiled the parser and typechecker components. The
transpiler component then failed with `I stopped shadow execution after 10
seconds.` The log reports no sanitizer violation at that failure. My C-seed
and VM supervisors both hardcoded that deadline; the outer Makefile timeout
could not extend it.

I now accept `NANO_SHADOW_TIMEOUT_SECONDS` as decimal seconds from 1 through
300, defaulting to 10 when absent. Empty, signed, fractional, zero, overflowing
and out-of-range settings fail closed. C-seed, VM and generated native shadow
supervisors share the setting. Child alarms and parent monotonic supervision
retain the same budget, and completion markers, process status and assertions
still govern publication. My sanitizer CI bootstrap steps explicitly select
60 seconds. Unit and negative-test steps retain the default. I did not change
coverage budgets: its inspected job was still running without this failure.

On Linux ARM64 I built fresh objects with GCC, `-O3`,
`-fsanitize=address,undefined` and `-fno-omit-frame-pointer`. The unchanged C
seed compiled `src_nano/transpiler_driver.nano` in 19.33 seconds total with the
default budget. This host does not reproduce CI's ten-second shadow failure;
total compilation includes parsing and native compilation outside the shadow
phase. With the patched seed, the real fixture fails at a one-second shadow
budget after 12.56 seconds total and publishes no artifact. With a 60-second
budget it completes in 19.19 seconds total, publishes its executable, and
reports no ASan/UBSan error. Leak detection was disabled for these compiler
runs, matching CI; this is not leak-freedom evidence.

`tests/test_shadow_runner.py` checks the default and finite upper bound,
invalid settings, early exit/exec/abort refusal, a hung child, a two-second
fixture refused at one second and completed at three, and that supervised
children are reaped. Focused instrumented C-seed and VM probes additionally
check successful publication, one-second infinite-shadow refusal, and invalid
configuration refusal with no artifact. These bounded checks do not establish
full sanitizer CI success; that exact-commit gate must still run.

The CI negative-test steps also pin `NANOLANG_COMPILER=./bin/nanoc_c`, using
the independently repaired negative runner's explicit compiler selection.
They must not accidentally test the self-hosted driver after bootstrap changes
the default symlink. Explicit self-hosted acceptance remains separate.

MAC: `task_6a2f75e259e84f09bbec61554b9ff9a7`; compiler-selection follow-up:
`task_9bdaa5f642b04645948f37acc282ea41`.
