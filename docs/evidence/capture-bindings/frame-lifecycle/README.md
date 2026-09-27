# My optional frame binding lifecycle gate

I qualify source8225b1d73 in seven fresh configurations: Linux GCC and Clang,
ordinary and ASan/UBSan, and Darwin Apple Clang ordinary plus Homebrew Clang
ordinary and ASan/UBSan. Each complete test-nanovm gate passes274729 VM assertions
and the existing substring, callback, heap-allocation and stack-allocation controls.
Each configuration also compiles the private record-array VM path. O3 and all
original assertions remain enabled. All989 source/driver inputs match afterward
on both hosts. I select the installed macOS26.2 SDK explicitly.

My production ownership change is19ab5a368: every departing frame releases its
optional owned binding state before draining physical locals, including private
mixed unwind; copied effect activations do not duplicate ownership. The actual
trapped-frame controls cover managed parameter return, tail transfer, destroy,
effect resume and lexical return. Production entry does not yet allocate this
state or admit new capture opcodes. Full capture semantics remain open.

The first19ab gate passes four ordinary configurations, then Darwin sanitizer
main exhausts the stack before the first unit test. Disassembly retains its
approximately126MiB frame. I deliberately stop dependent Linux sanitizer
compilation, retaining its -15 terminal, rather than attribute a product failure.
Both original input maps remain unchanged after their terminals.

My corrected RUN_TEST dispatch uses a volatile typed function pointer, keeping
test frames separate. Independent source review precedes fresh qualification.
A nonexecuted corrected Darwin sanitizer object shows main reserving80 bytes;
only then do I run the complete corrected gates. I preserve first and corrected
reports losslessly, with raw/stored hashes in checks.json. The first failed
Darwin binary and corrected preflight object remain in persistent qualification
storage, not in this report bundle. Passing ordinary test executables are removed
by the existing Make recipe; I do not claim to retain those deleted products.

These are bounded lifecycle and runner checks. Canonical integration and the
full compiler, bootstrap, backend, SDK and release gates remain open.
