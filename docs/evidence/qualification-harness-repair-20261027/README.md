# Qualification harness repairs, October 27

I retain the two failed gates from PR #995 at
`b7c20ea901a930d211612ec9d6cc4bdda9d8bcdb`. I track the repairs in
[#1036](https://github.com/jordanhubbard/nanolang/issues/1036) and
[#1037](https://github.com/jordanhubbard/nanolang/issues/1037); full release
acceptance remains under #976.

## Retained failures

- [Memory Sanitizers job122356428817](https://github.com/jordanhubbard/nanolang/actions/runs/40527688607/job/122356428817)
  fails compilation of `test_disasm_roundtrip.c` at its derived `sprintf`
  destinations, with GCC `-Werror=format-overflow` null-destination diagnostics.
  The allocation already had a null guard. I do not relabel this as a proven
  missing allocation guard, runtime sanitizer finding or infrastructure failure.
  [Exact diagnostic lines](retained-sanitizer-diagnostic.txt) retain the failure.
- [Xcode compiler-contracts job122479317295](https://github.com/jordanhubbard/nanolang/actions/runs/40527689123/job/122479317295)
  passes109 of110 cases; the3500-call native string safepoint case exceeds the
  harness's90-second sanitized C compilation deadline. I retain its complete
  small [compiler-managed log](xcode-compiler-managed.log). This does not prove
  a semantic defect or an infrastructure cause.

I downloaded only these small retained ZIP artifacts, preserving their hashes:

| Artifact | Bytes | SHA-256 |
| --- | --- | --- |
| verifier-corpus-sanitizers,12774812415 |107868|333d18510debfcd76bafbd47cc81774289488bb72e6d289af4bb3cdf0f03dbe2|
| qualification-compiler-contracts-xcode-27,12778883419 |2045079|8ced9964e9abe06af5137a15a8cf22e55a5761c0f703b8c0173f18dfd04c7b0c|

## Changes and scoped checks

I bound fixture formatting with remaining capacity and check every append.
Allocation absence, exhausted capacity and truncation refuse construction;
exact-fit and recovery cases retain the terminator and correct length.
I replace the long-string prefix's remaining deprecated `sprintf` with checked
`snprintf`. I preserve the2200-symbol workload and strict warnings.

I route the string-retention harness through the existing `run_captured`
process-group supervisor. Only sanitized C compilation selects `build=True`
and the existing positive `CMD_TIMEOUT` budget (default600 seconds). Assembly,
translation, VM and native execution retain90-second deadlines. I retain
all3500 calls, sanitizer flags, retention limits and assertions.

On macOS arm64, using freshly built exact-source C-hosted tools:

- `make -f Makefile.gnu test-disasm-roundtrip CC=clang` with
  `CFLAGS='-Wall -Wextra -Werror -std=c99 -O3 -g -Isrc -D_GNU_SOURCE -fsanitize=address,undefined -fno-omit-frame-pointer'`
  and `LDFLAGS='-lm -fsanitize=address,undefined'`:220 checks pass.
- `gcc-16 -Wall -Wextra -Werror -std=c99 -O3 -g -Isrc -Isrc/nanoisa -D_GNU_SOURCE -fsanitize=address,undefined -fno-omit-frame-pointer -c tests/nanoisa/test_disasm_roundtrip.c -o obj/test_disasm_roundtrip_gcc16.o`: passes with no diagnostics, producing a154312-byte object.
  This executes optimized GCC16 compilation on macOS, not the original Linux/glibc compiler.
- `python3 -m unittest -v tests.test_native_string_retention.NativeStringHarnessControls tests.test_native_process`:
  six controls pass in5.567s, including actual process-group timeout cleanup
  and build-budget versus guest-deadline checks.
- `python3 -m unittest -v tests.test_native_string_retention.NativeStringRetention.test_call_safepoint_collects_without_backwards_branch`:
  passes in37.790s; measured `string_peak_bytes=65550`.

I built only the C-hosted tools needed by this case, without a bootstrap.
Initial local setup failures are retained outside the repository: mixing
instrumented objects with an uninstrumented link omitted sanitizer runtimes,
and the first case invocation lacked `bin/nanoisa`. Explicit sanitizer link
flags and the exact `bin/nanoisa` target corrected those setup errors before
the successful run. I do not report those setup attempts as product defects.

I preserved both active older qualification jobs, prior source, outputs and
caches. I did not rerun an active CI job. Exact new-head Linux sanitizer and
hosted Xcode qualification remain pending; local checks do not close them.
