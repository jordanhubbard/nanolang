# My first LLVM translator boundary

I translate a verified v2 module directly to LLVM IR. My initial contract covers integer/bool/void scalar storage, integer/bool results, locals, direct calls, branches and assertions. I retain value tags through calls and joins and check typed operations at runtime when verification cannot establish their inputs. My integer arithmetic wraps at 64 bits; division by zero returns zero, and INT64_MIN divided by -1 returns INT64_MIN. I use explicit control before division to avoid LLVM poison.

I reject imports, linked modules, heap values, nominal layouts, ownership/passive metadata and unsupported instructions. I do not embed NanoVM or resurrect AST target paths. I publish named output only after verification, profile checks and complete emission succeed, using an exclusive temporary beside the destination. Existing output and source survive failures.

My first same-module gate runs VM, C AOT and LLVM on calls, branch effects, loops, arithmetic boundaries and boolean tags. A passing scalar gate does not complete my required full-language LLVM or Wasm release scope. Wasm remains unimplemented.

## My bounded evidence

`make nvm2llvm nanoisa_dump` builds only the translator and module tools. `make test-nvm2llvm` requires LLVM's `llvm-as`, `lli`, `opt`, `llc` and a native C linker; missing tools fail this explicit execution gate. It does not run implicitly as an optional pass in the ordinary suite.

Eight focused methods pass on Linux ARM64 with installed LLVM 23.0.0git. The same modules run in NanoVM, native C, unoptimized LLVM, optimized LLVM and linked LLVM machine code. I test wraparound, zero/overflow division and remainder, recursion, loops, stack joins, argument order, bool tags, runtime type rejection and atomic failure preservation. The negative bool-to-int call traps in VM/LLVM; my C translator rejects it earlier, which I record as refusal rather than claiming a native execution. A development PUSH_VOID fixture also exceeds the C translator's current subset; the common parity fixture stays integer/bool.

The eight methods pass in 3.897 seconds, and again in 6.702 seconds with the new host translator sources instrumented by ASan/UBSan/LSan (linked existing ISA objects are not fully instrumented). Logs remain `/tmp/nanolang-llvm-foundation-complete-tests.log` and `/tmp/nanolang-llvm-foundation-sanitizer-tests.log`. I retain initial assembly-fixture syntax failures and the new translator's corrected relative-branch offset failure in the earlier numbered logs. VM/C comparison tools come from the already tested host-ownership checkout; no full compiler build was run for this slice. Darwin and other LLVM versions are unrun evidence, not claimed coverage.

I do not yet lower generic arithmetic/comparison, floating point, globals, tail calls, strings, aggregate allocation, imports, linked modules, effects or resource/passive contracts here. Those remain explicit refusals or signature/profile exclusions. This foundation does not close the full LLVM release requirement, and I have not implemented Wasm.
