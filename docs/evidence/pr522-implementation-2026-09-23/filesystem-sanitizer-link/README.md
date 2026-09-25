# I retain sanitizer linkage in the filesystem gate

Hosted run `35971818336`, units-01 job `107543731151`, passes bootstrap and then fails the original filesystem native-product link: hardcoded `cc` omits the ASan/UBSan runtime required by `bin/nano_aot_runtime.o`. I retain its complete log.

I reproduce that failure locally with the unchanged test and a runtime object freshly linked from the unoptimized instrumented `dyn_array.o`, `gc.o` and `gc_struct.o` used in the [function-index checks](../function-lookup-index/README.md). I do not substitute an ordinary runtime to obtain an instrumented pass.

The corrected test selects `NANO_CC`, then `CC`, then `cc`, and splits compiler commands and `NANO_LDFLAGS` into arguments. I retain every source assertion, VM verification/execution, native execution, strict C warning and original 180-second command deadline.

Both original methods pass with Homebrew Clang, `NANO_LDFLAGS=-fsanitize=address,undefined`, `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and `UBSAN_OPTIONS=halt_on_error=1`. I then restore the ordinary runtime byte-for-byte, remove those overrides, and both methods pass with default compiler selection. The native-stage compiler, VM and translator executables in this local run are ordinary; the generated native product and its runtime are instrumented in the sanitizer case. Full hosted partition acceptance remains pending.

I retain the MAC close response for `task_22673ccbd07e4577b342b24af67da14b`; repository verification does not override the ledger lifecycle.
