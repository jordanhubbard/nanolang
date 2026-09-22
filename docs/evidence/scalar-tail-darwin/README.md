# My Darwin scalar tail qualification

At frozen `1f352e543`, all six scalar tail methods pass in ordinary and sanitizer lanes, in 20.265 and 30.417 seconds. Each lane builds its own backend tools first. My generated C is checked with AddressSanitizer, UndefinedBehaviorSanitizer and leak detection; native LLVM IR uses AddressSanitizer. The VM, lli and Wasm paths remain ordinary. This is bounded scalar-tail acceptance, not a complete compiler or release gate.

I preserve four earlier terminals: an incorrect make target, the missing default Homebrew SDK, and two Python stack-limit setup refusals before bounded program execution. The corrected Darwin launcher sets and reads back both soft and hard 512 KiB limits before execution. The six methods retain self/mutual tail recursion, ordinary callers, initialization, tagged carriers and atomic refusals.

My report bundles preserve exact remote bytes and original remote paths. Each object index maps hashes to the verified local copy under `/home/jkh/nanolang-qualification/puck-tail-evidence`; I rehashed every referenced object after transfer. The bundles do not embed the object store or all temporary generated programs. Source/tool endpoints are equal for every phase; all leaders were reaped and no tracked descendants or process groups remain. Earlier setup failures remain separate from product results.
