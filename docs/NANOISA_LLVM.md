# My first LLVM translator boundary

I translate a verified v2 module directly to LLVM IR. My initial contract covers integer/bool/void scalar storage, integer/bool results, locals, direct calls, branches and assertions. I retain value tags through calls and joins and check typed operations at runtime when verification cannot establish their inputs. My integer arithmetic wraps at 64 bits; division by zero returns zero, and INT64_MIN divided by -1 returns INT64_MIN. I use explicit control before division to avoid LLVM poison.

I reject imports, linked modules, heap values, nominal layouts, ownership/passive metadata, named module initializers, implicit exits and unsupported instructions. Non-entry functions must end with an explicit return or branch; I retain dynamic declared-result tag checks on every return. I do not embed NanoVM or resurrect AST target paths. I publish named output only after verification, profile checks and complete emission succeed, using an exclusive temporary beside the destination. Existing output and source survive failures.

My first same-module gate runs VM, C AOT and LLVM on calls, branch effects, loops, arithmetic boundaries and boolean tags. A passing scalar gate does not complete my required full-language LLVM or Wasm release scope. Wasm remains unimplemented.

## My bounded evidence

`make nvm2llvm nanoisa_dump` builds only the translator and module tools. `make test-nvm2llvm` requires LLVM's `llvm-as`, `lli`, `opt`, `llc` and a native C linker; missing tools fail this explicit execution gate. It does not run implicitly as an optional pass in the ordinary suite.

Eleven focused methods pass on Linux ARM64 with installed LLVM 23.0.0git. The same modules run in NanoVM, native C, unoptimized LLVM, optimized LLVM and linked LLVM machine code. I test wraparound, zero/overflow division and remainder, recursion, loops, stack joins, argument order, bool tags, runtime type rejection and atomic failure preservation. The negative bool-to-int call traps in VM/LLVM; my C translator rejects it earlier, which I record as refusal rather than claiming a native execution. A development PUSH_VOID fixture also exceeds the C translator's current subset; the common parity fixture stays integer/bool.

The final eleven methods pass in 6.098 seconds, and again in 5.292 seconds with the new host translator sources instrumented by ASan/UBSan/LSan (linked existing ISA objects are not fully instrumented). Logs remain `/tmp/nanolang-llvm-result-tag-tests.log` and `/tmp/nanolang-llvm-final-sanitizer-tests.log`. I retain initial assembly-fixture syntax failures and the new translator's corrected relative-branch offset failure in the earlier numbered logs. VM/C comparison tools come from the already tested host-ownership checkout; no full compiler build was run for this slice. Darwin and other LLVM versions are unrun evidence, not claimed coverage.

I do not yet lower generic arithmetic/comparison, floating point, globals, tail calls, strings, aggregate allocation, imports, linked modules, effects or resource/passive contracts here. Those remain explicit refusals or signature/profile exclusions. This foundation does not close the full LLVM release requirement, and I have not implemented Wasm.

My initial implicit-return fixture exposed a separate VM caller-resumption defect: a nested implicit return ended execution before the caller resumed. Task `task_4b3800f46af143fbb171f9565f92b8e0` retains `/tmp/nanolang-llvm-implicit-tests.log`. Entry implicit return executes in VM but C AOT refuses it; task `task_8a18a76c86884299ac3f7880ea617978` retains `/tmp/nanolang-llvm-profile-tests.log`. I refuse all such exits in this foundation rather than claim common coverage. My tests verify both refusals and a VM/LLVM runtime-unknown declared-result tag rejection. Named initializers are also explicitly refused because VM executes `__init__` before entry and this profile has no initializer contract.

My default executable entry is `main`. `--entry-name nano_NAME` selects an
ASCII identifier in my reserved target-entry namespace; other names are
refused before publication. My Wasm translator uses this API to avoid the
wasm32 C-main startup convention. The underlying scalar result remains i32
at the host entry boundary.
