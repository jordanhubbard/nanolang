# My guarded variant-return correction

I correct the retained alias-return failure from the preceding nested checkpoint.
The original `test_result_arguments_survive_alias_and_return` now passes with
C-seed, Stage 1 and Stage 2. I preserve the historical terminal and source in
`../nested/`.

I retain conservative single-variant facts through constructors, locals,
direct calls, returns and tail calls. Unknown producers and address-taken
functions cannot establish a variant. After inference converges, I propagate
remaining unknowns before using a guard to suppress a return diagnostic.
I keep all runtime return checks. My precise supported boundary is in
`docs/NATIVE_VARIANT_GUARDS.md`; this is not general control-flow refinement.

The qualified gate passes 14 methods: nine scalar-carrier methods, four
integer-array-carrier methods and the unchanged canonical alias-return method.
A further boundary method passes for variant 65,535. New cases execute both
EQ/NE and I64_EQ/I64_NE, both conditional jump directions, local aliases and
tail calls. Reachable wrong-tag returns, conflicting callers, jumps bypassing
the guard, address-taken functions and a record-projected wrong variant still
refuse and preserve the prior output. Generated native controls retain
ASan/UBSan with mandatory leak checks using Homebrew LLVM.

The complete translator regression passes 2,428 assertions. The instrumented
translation-unit check also passes admission and three refusal/cleanup paths
with ASan/UBSan and `detect_leaks=1`. Only `src/nanoisa/nvm2c.c` is instrumented
in that tool build; its dependencies use the ordinary object files. The retained
script records the exact compile/link and invocation choices. I do not claim
that check is a wholly instrumented compiler bootstrap.

`logs.json` seals each uncompressed terminal. No NanoLang compiler source
changes in this slice; I exercise the rebuilt translator through the existing
fresh Stage 1 and Stage 2 from the nested checkpoint. Owned unions and the full
integrated release gates remain open.
