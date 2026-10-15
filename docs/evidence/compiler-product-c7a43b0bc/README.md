# My clean self-hosted named-container compiler gate

I ran `make -j2 test-one-ir-compiler` in a clean detached checkout at
`c7a43b0bc6ed3032ec25695f3d56971320736b3a`. All 95 compiler-product methods pass;
the complete Make command exits zero after 782.792 seconds. My source remains
clean at the same commit. The manifest retains the log hash and terminal state.

I used Homebrew LLVM through the explicit `cc` wrapper directory, with the clone
as `NANOLANG_ROOT` and its own module cache. No prepared self-hosted compiler or
translator override supplied the compiler under test. This run includes the
self-hosted named-container producer repair; it predates captured-closure target
analysis and native environment lowering. My raw fixed-point and complete 5.1
release qualification remain separate gates.
