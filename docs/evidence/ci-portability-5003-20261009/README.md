# Compiler portability repair after 5003d1590

I retain the two build failures from [CI run 38009049399](https://github.com/jordanhubbard/nanolang/actions/runs/38009049399)
and [guide run 38009049407](https://github.com/jordanhubbard/nanolang/actions/runs/38009049407)
at `5003d15907641f85bba0f5c5e8cb063db602c9cd`. I track this repair under
[#982](https://github.com/jordanhubbard/nanolang/issues/982).

My retained excerpts cover nine failed main-CI jobs and the guide failure.
GCC rejects three misleadingly indented control-flow statements in
`src/service_lowering.c`. Clang rejects boolean bitwise cleanup in
`src/runtime/service_product.c`. I put each unconditional statement on its own
line and spell out all three cleanup attempts. I preserve later removal
attempts when an earlier one fails. I keep warnings as errors.

I pass syntax checks with installed Apple Clang and Homebrew LLVM Clang,
including `-Wbitwise-instead-of-logical`. My installed Apple compiler did not
reproduce the original warning before the repair; CI is the failure evidence.
I have no local Linux GCC result: Docker socket access was denied.

I compile the two changed objects and link separate `nanoc_c` and `nano_virt`
executables under `/private/tmp/nanolang-ci-portability-isolated`, using the
Makefile's compiler and linker arguments and existing objects for unchanged
code. I retain those commands in `isolated-build.log`. This is an incremental
Darwin qualification, not a clean build or a self-hosted fixed point.
My first temporary command-extraction script stopped after compiling both
objects; I corrected extraction of the conditional link recipe and completed
both links. No product change was needed for that harness error.

I run three existing `ServiceDrivers` methods with only `setUp`'s driver paths
redirected to those isolated executables. I set `NANO_ROOT` to this checkout
and `NANO_NATIVE_TEST_CC` to `/opt/homebrew/opt/llvm/bin/clang`:

- `test_grant_failure_aliases_and_compiler_failure_preserve_output`
- `test_generated_shadows_byte_parity_and_native_runtime_grant`
- `test_shared_borrow_shadow_failure_preserves_output`

I retain their terminal results in `driver-tests.log`. They check grant and
compiler failures, protected output aliases, staging residue, generated shadow
selection, byte parity, native invocation grants, and failed shared-reference
shadows preserving prior output. I do not claim the complete driver suite.

The original Make invocation also progressed to compiler links, but its tool
handle was lost before a terminal result was captured. I do not use that log
as proof of successful completion or restart it on an observation timeout.

GitHub DNS/API access failed during follow-up verification. I cannot refresh
the previous sanitizer job's state or claim a passing replacement CI run.
The repair's CI acceptance and the full 5.1 release remain open.
