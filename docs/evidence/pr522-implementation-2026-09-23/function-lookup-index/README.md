# I bound interpreter function lookup

I retain the hosted Forth-worker preparation failure at candidate `5016427`: the compiler stops interpreted shadows at the original 60-second deadline, before Forth tests run. The CI merge commit has no file differences from the candidate. My prior [instrumented profile](../bootstrap-shadow-profile/README.md) places 746 of 2,791 main-thread samples in `env_get_function`; it does not establish the Linux timeout cause.

I index function names by hash and slot. I retain first-declaration selection, current-module preference, qualified namespace ownership, builtin precedence and the module-local `array_push` exception. I read mutable function metadata from its slot, so same-name REPL replacement remains visible. Name-changing external writes explicitly invalidate the optional index. Index allocation failure retains linear lookup. I release the index with its environment.

## Local evidence

- In the same 4,096-function workload, 2,000 successful/missing queries require 8,393,000 name comparisons before correction and 201,000 afterward. Both counts include the unchanged builtin scan. This measures comparisons, not wall-clock speedup. The baseline is the parent commit's `src/env.c`; both harnesses use identical declarations and queries for `function_0` and `ordinary_missing_function`.
- The existing environment-index target passes with new function-resolution controls: table growth, duplicate declarations across modules, global preference, unrelated-module fallback, qualified owner/export refusal, builtin precedence, local push/extern selection, same-name slot replacement, changed module metadata, explicit rename invalidation, null-name slots and failure of all three initial index allocations. The final test binary passes under ASan/UBSan with leak detection enabled.
- All original typechecker unit tests pass under unoptimized ASan/UBSan, use-after-return detection and the CI leak setting (`detect_leaks=0`). This does not claim a typechecker leak gate.
- The complete compiler source builds successfully under an unoptimized instrumented C seed in 135.890 seconds. Mandatory interpreted shadows retain their 60-second limit. The outer command bound is 600 seconds; `compiler-shadow.json` records configuration and hashes. I do not infer a matched timing improvement from the earlier 150-second profile, which used a different build-cache setting.
- All 35 original functional-array, single-letter-enum and generic-function methods pass in 148.558 seconds with compiler leak detection. Functional-array generated programs also retain ASan/UBSan and leak checks. The other generated executables and existing native stages are not requalified by this run. Original compilation deadlines remain unchanged.

I build all instrumented objects fresh in `/tmp/pr522-function-index/obj` using Homebrew Clang, `-O0 -fsanitize=address,undefined -fno-omit-frame-pointer`. The compiler hash is unchanged between the full compiler workload and compatibility checks. The compiler-run manifest also records an intermediate `src/repl.c` hash: that file is not part of the compiler binary, and its unnecessary invalidation edit was reverted afterward. Final compiler source hashes are in `manifest.json`.

## Separate finding and remaining acceptance

The standalone C REPL target fails with four pre-existing compile errors: absent `Environment.parent`, `g_mod_count` and `cmd_modules`. Strict syntax checking of unchanged parent source reproduces them. I retain both logs and track restoration as `task_b3732aa2105b471bb4d6bb73c5c26840`; I do not claim a passing REPL executable.

Function-lookup work is `task_90609aa45d3340de8daeb55a5b644498`. Final hosted deadlines, all original partitions/platforms, and canonical fixed points on the changed compiler source remain required. The prior-source native fixed-point process continues separately; it is not acceptance of this index.
