# I qualify canonical runtime consumers

I audit every Python reference to `nano_aot_runtime.o` after the [filesystem link repair](../filesystem-sanitizer-link/README.md). String, product and One IR consumers omit selected runtime-link flags; module-facts also ignores the configured leak-capable compiler. The first 16-method run retains eight failures with an actually instrumented runtime object.

I share compiler and link selection in `tests/native_toolchain.py`. Compiler precedence is `NANO_NATIVE_TEST_CC`, `NANO_CC`, `CC`, then `cc`; argument splitting retains configured compiler arguments. Link precedence is `NANO_ARTIFACT_LDFLAGS`, `NANO_LDFLAGS`, then `LDFLAGS`, including explicit empty overrides. I preserve strict warnings, semantic assertions, native/VM execution, allocation checks and all command deadlines.

The module-facts VM trace identifies a separate stale fixture expectation: `/var/...` versus the correct canonical `/private/var/...`. I retain exact path equality using the resolved dependency and add an explicit symlink-import control for the self-hosted producer. All nine module-facts methods pass after this correction.

## Verified scope

All 18 selected original consumer methods plus the new symlink control (19 total) pass with an actual ASan/UBSan runtime object and generated-native link instrumentation. The same 19 pass with the ordinary runtime restored byte-for-byte. Compiler selection stays on Homebrew Clang for the leak-requiring module-facts checks. I separately remove compiler/link overrides and pass the seven string/product/filesystem methods with the default compiler and ordinary runtime. The native compiler stages, VM and translator used by this local check are ordinary; I do not claim all providers are instrumented.

The instrumented runtime is linked from the fresh `-O0` ASan/UBSan dynamic-array, GC and GC-struct objects recorded in the [lookup evidence](../function-lookup-index/README.md). Generated native checks use `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and `UBSAN_OPTIONS=halt_on_error=1`. The manifest identifies source/provider hashes and exact runtime restoration.

## Remaining full-gate failures

The expanded 45-method run fails in 21 subcases. Two are the module-path expectations subsequently corrected above. Eighteen representation/allocation failures also reproduce with the unchanged parent One IR test source and the same Clang selected through a temporary `cc` wrapper: C-seed full-compiler optional/record conversion, twelve generic-array argument ABI controls, three record-temporary allocation controls and two tagged projections through locals. I retain this independent baseline; compiler selection is not their demonstrated cause.

The other full-run failure reaches the default ten-second shadow deadline with the instrumented C seed. That first diagnostic omitted CI's existing `NANO_SHADOW_TIMEOUT_SECONDS=60`; I preserve this setup difference and require the exact CI configuration for further qualification. I do not extend any test or production deadline.

`task_0bad55314e3f45799c3a9dbb219c65d0` tracks the consumer qualification. `task_b90e9d06d86e4e5a97b3eb8218ebfac0` tracks the remaining complete One IR gate. Generated failure artifacts are retained under `/tmp/pr522-runtime-consumers/retained/`. This checkpoint changes test harnesses and one fixture expectation, not compiler semantics. It does not qualify the complete One IR suite, hosted partitions or release.
