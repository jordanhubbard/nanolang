# My rejected-declaration warning repair

I reject a reserved builtin before registering its Function. My later missing-shadow warning still reads that absent Function and crashes. I guard the warning lookup result; my existing rejection flags and diagnostics continue to return an ordinary checking failure.

I preserve the unchanged retained C-seed GDB SIGSEGV, and the new regressions fail on the prior C component: four invalid CLI cases signal, while the legal renamed native program passes. My six direct checker cases cover accepted root/module declarations, orphan shadows and reserved declarations with/without shadows. The corrected CLI returns ordinary nonzero errors, preserves both original diagnostics, preserves existing output bytes, publishes no absent output, and compiles and executes the legal renamed control.

My normal build, both new CLI methods, existing checker-context/dispatch/STRING/index/evaluator/scope controls pass (137.494 seconds). My separate full ASan/UBSan object and compiler-input closure passes both checker units with detect_leaks=1 (65.277 seconds), with source/compiler guards unchanged. Reverting only the warning predicate in a temporary sanitizer object reproduces a Function null-member diagnostic and rejects the new unchanged assertions. I preserve baseline/candidate source snapshots, commands, streams and provider identities; cached baseline unit objects are not a whole historical-build qualification.

I add both regression targets to test-units and the original Linux/Darwin C-repair workflow. This local component does not close #1267, either C repair issue, platform qualification or the full69 release contract. I retain those gates and the pending frozen CAST_STRING expectation reconciliation.

Run `python3 verify.py --workspace /path/on/large/volume` to verify the receipt without executing a product. The reader binds the original crashes, candidate cases, source-only guard change and sanitizer/mutation streams.
