# My public C Darwin policy qualification

I qualify canonical main a59858424169ca52ed10f099c5692dd88e6636d4 in a fresh
isolated Darwin checkout. I build C-seed, interpreter, canonical producer, VM
and native translator with Apple Clang in4.297s. Generated programs and API
fixtures explicitly use Homebrew LLVM23.1.1 with ASan/UBSan and
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1`.

All16 methods pass in37.971s: eight public C binary64 methods, four signed
nonfinite formatting/ownership methods and four string equality methods.
The suite retains strict C99/C11 O0/O2, contraction controls, exact result/input
bits, independent output bytes, once evaluation, globals/entry/fields, stable
conversion aliases, cleanup/failure controls, private names, prior-output
preservation and ordinary interpreter/verified VM/sanitized native comparison.
My retained `qualification-runner.py` and `invocation.json` record the exact suite CC/ASAN_OPTIONS assignments and separate build compiler selection. Nothing is skipped or weakened for Darwin. Before/after source/tool hashes match.
I independently compare all eight transferred reports with remote SHA256; the
[sealed report directory](public-c-darwin-policy/) retains command/status,
inventory, logs and complete hashes.

## My completed formatting boundary

These Darwin public-C checks complete the last platform observation for signed
nonfinite parent `task_e92a45b66a104e9ba3854cd5f994df8b`, together with:

- PR745 [VM/native/shared formatter evidence](signed-nan-format.md), including
  signed quiet/signaling NaNs and infinities, exact bits and sanitized range checks;
- PR739 [all71 managed checks](managed-allocation-portability.md) passing on both
  Linux and Darwin with unchanged allocation, sanitizer and memory limits;
- PR752 [legacy interpreter and paired runtime evidence](legacy-signed-nan-format.md),
  with actual Linux/Darwin bootstrap, supported scalar/array/generic/format routes
  and retained finite conventions; and
- PR749 [public C formatting evidence](public-c-nonfinite-format.md) passing
  Linux GCC/Clang, plus the fresh Darwin observations above.

I preserve all original host-oracle, declaration-order and fixture/setup failures.
Earlier evidence documents correctly described e92 as open at their checkpoints;
this document records the subsequent completion, not a rewrite of those results.
Canonical merge precedes parent reconciliation. Scalar arithmetic5009, full
managed/ownership/public-C portability, product522 and release gates remain open.
This signed-text completion does not claim general literal byte transport or
close the separately recorded escape-normalization task.
