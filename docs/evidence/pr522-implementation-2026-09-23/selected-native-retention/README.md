# My selected native compiler in retention tests

The broader One IR run completes 76 methods in 423.747 seconds. All 31
`OneIrCompiler` methods pass, including C-seed and self-hosted full compiler
translation, native compilation and generated hello products. Sixty subcases
in the neighboring modules fail before their assertions because Apple's
sanitizer reports that `detect_leaks=1` is unsupported on this platform.
Those fixtures hardcode `cc` or inspect only `CC`, ignoring my explicitly
selected Homebrew Clang.

I use the existing shared `native_cc()` selection in those fixtures, preserving
the explicit `NANOLANG_GUARD_SAN_CC` override. No sanitizer, leak, performance,
retention, failure, stack or semantic assertion is removed. My first corrected
45-method run clears those runtime failures and exposes two strict-Clang
warnings for counters that a fixture increments but never measures. I inject
only the requested counters and verify each instrumentation hook occurs once.
The copied-read scan/visit bounds and storage-growth scan assertion remain.

The final 45-method rerun is retained after completion. My production compiler
source is unchanged from `df364a501`; these are fixture/toolchain corrections.
This evidence does not qualify the separately tracked hosted setter or nominal
compatibility failures, nor final-source fixed points or complete hosted CI.

My final thirteen-module retention/allocation rerun passes **45 methods in
70.551 seconds** with Homebrew Clang and the original ASan/UBSan/leak controls.
All **16 canonical runtime-consumer methods** also pass in 4.478 seconds.
Together with the 31 passing One IR methods in the retained broader run,
these cover every method in that 76-method owning invocation; I do not label
the original failed invocation a pass. No production source changed between
those runs. The host AOT runtime is ordinary in this qualification; the C seed
retains its previously recorded ASan/UBSan build. Final hosted and complete
instrumented-host-runtime qualification remain separate.
