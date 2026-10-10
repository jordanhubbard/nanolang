# I replace my legacy match backstop fixture with NanoISA

My first fixture build fails because imported private emitter globals are unavailable. I retain that failure. The correction establishes an integer parameter/return context through the normal function emitter, then invokes the same case emitter as the checked path without its coverage gate.

The corrected backstop method passes on Darwin in 76.010 seconds. Both incomplete integer expression and statement matches produce verified modules. Their matching inputs execute successfully in NanoVM and strict LLVM Clang C11 with ASan/UBSan. Their unmatched inputs produce the exact VM assertion diagnostic and native SIGABRT with the NanoISA invariant diagnostic, without sanitizer errors. The fixture no longer imports `transpiler.nano`.

I preserve the prior valid-input/terminal-miss distinction and native abort assertions, add VM verification/execution, and document the backend-specific diagnostic change in `docs/CANONICAL_MATCH_GUARDS.md`. My case emitter retains the false assertion and terminal halt. Coverage remains mandatory through every normal emitter entry.

I rerun all four focused component methods with a newly built emitter driver and normal dependency shadows; their full log is retained here. Fresh installed Stage1/Stage2, the complete canonical corpus and adjacent tests remain pending. I add an explicit Make gate and retain issue #981 as open.
