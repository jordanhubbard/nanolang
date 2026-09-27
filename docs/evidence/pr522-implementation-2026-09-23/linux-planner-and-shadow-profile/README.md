# Linux planner qualification and compiler-shadow diagnosis

I retain evidence against base `dbc6d31b2`. `identity.json` identifies the changed sources and the two instrumented Linux compiler binaries. I keep #522 draft: these local gates do not establish final hosted acceptance or canonical bytecode fixed points.

## Qualified repairs

I reproduce the hosted strict GCC edge-table warning with the original planner and pass the identical compilation after retaining the validated function count in an immutable local (`linux-gcc-before-after.log`). I preserve the eight-function limit. New fixtures exercise the highest target bit, reversed declaration order, transitive cycles and refusal at nine functions. The first full Linux gate also identifies three misleadingly indented fixture assertions; I separate their unconditional checks without suppressing warnings.

I pass `make -j1 test-affine-bytecode test-owned-indirect-calls` on Ubuntu 24.04 aarch64, GCC 13.3.0 (`linux-owned-corrected.log`): 6,258 affine assertions, 6,779 allocation/visit-failure checks and 66 indirect-call cases with 8,246 VM assertions. Generated native programs retain ASan/UBSan, leak detection, target/tag guards and allocation-failure cleanup. My ordinary Darwin boundary gate passes the same affine counts. My private Homebrew Clang ASan/UBSan affine objects also pass both complete affine gates with `ASAN_OPTIONS=detect_leaks=1` (`darwin-affine-instrumented.log`).

I reproduce the Darwin shared-harness failure under default `cc`: the native runtime rejects leak detection. The harness now honors `NANOLANG_GUARD_SAN_CC`, falling back to `CC`/`cc`, and explicitly rejects sanitizer recovery. With Homebrew Clang, `make -j1 test-owned-indirect-calls test-owned-value-graphs` passes all 66 indirect cases and all 12 graph cases (2,169 VM assertions), including generated-native sanitizer checks. These fixture generators and CLI runtimes are ordinary builds; only the separately named private affine gate instruments its implementation objects.

## Unresolved shadow timeout

I build the complete Linux C seed in private directories with GCC 13.3.0, `-g -fsanitize=address,undefined -fno-omit-frame-pointer`, no optimization, and matching sanitizer link flags. I retain its build log. I place the executable under `/work/bin` so module-root resolution matches the repository. The command is:

```sh
ASAN_OPTIONS=detect_leaks=0 NANO_SHADOW_TIMEOUT_SECONDS=60 \
NANO_MODULE_PATH=modules NANO_BUILD_CACHE=/work/profile-module-cache \
/usr/bin/time -v bin/nanoc_pr522_profile src_nano/nanoc_v06.nano --verbose -o /work/profile-stage1
```

I preserve the existing 60-second shadow deadline and every shadow. The unsampled baseline exits 1 after shadow timeout; total command wall time is 89.98 seconds. A separate fully instrumented Darwin compiler passes all 902 shadows under the same deadline (`darwin-shadow-correct-root.log`). Total command time includes work outside the supervised shadow child.

Thirty Linux GDB diagnostic samples reach `__asan::FakeStack::Allocate` in 22 top frames, frequently through small value constructors. Sampling perturbs execution and is diagnostic only. `value_return_probe.c` and its GCC assembly compare named local returns with compound literals: the latter omit constructor fake-stack allocations in this controlled probe.

I test that change in the full compiler (`constructor-experiment.patch`). The unsampled experiment also exceeds the unchanged deadline and exits 1, with total wall time 91.59 seconds. Eight subsequent diagnostic stacks still repeatedly reach fake-stack allocation through other interpreter functions. I revert the production experiment. I do not claim it improves the workload or explains the timeout. I retain the patch, both compiler hashes, full terminals and sample scripts for the next measurement. Neither run disables use-after-return checks.

The isolated Linux container uses an ordinary Ubuntu 24.04 image on a dedicated Colima aarch64 VM; it is a local reproduction, not the hosted x64 worker. Imported native modules and generated native output are ordinary builds. I have not qualified the entire final platform/sanitizer matrix or canonical `.nvm` fixed points here.
