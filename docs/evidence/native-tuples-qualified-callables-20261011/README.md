# My native tuple and qualified callable evidence

I continue #976 from 021c7d7a3. I substitute concrete tuple layouts inside native generic bodies and retain imported tuple parameter/result declarations. I collect their nested callable and tuple dependencies before emitting native signatures.

I resolve qualified function values through the current module owner in both C checking and self-hosted checking/lowering. Visible local receivers retain field-access behavior. I preserve contextual generic signatures for declared locals and deferred generic arguments, and refuse incompatible parameter, result and arity signatures before replacing prior output.

I copy callback names and complete signatures into interpreter tuple storage and copy them again when projecting a callback. Returning a generic callback parameter inside a nested tuple no longer leaves its metadata owned by the returning call frame. Array elements retain their existing alias behavior. I do not claim general recursive tuple reclamation from this change.

My complete tuple/generic suite passes 40 methods through C NanoISA, C native and the newly built self-hosted seed (c-suite.log, native-suite.log, selfhost-seed-suite.log). The four new methods cover imported qualified/selective tuple callables, nested generic scalar/string/float/callback/array tuples, repeated callback projections after reassignment, local receiver shadowing, contextual imported generic callbacks and incompatible signatures in unused bodies. The NanoISA route verifies and executes each positive module in NanoVM and generates and executes strict ASan/UBSan C AOT products using Homebrew LLVM.

My four changed C implementation objects—environment, evaluator, typechecker and transpiler—are separately rebuilt with ASan/UBSan (sanitizer-commands.json). All 40 native methods pass with these objects and a 65,520 KiB stack limit (asan-suite-64m.log). I disable leak detection for this interpreter audit; it does not establish leak-free interpreter allocation. A direct tuple-lifetime.c check independently releases the original callback and then checks both retained recursive signatures before freeing the tuple under ASan/UBSan.

I retain the initial sanitizer failures. A compiler stored outside my bin directory first looked for runtime sources beneath /private/tmp; moving that same instrumented binary into bin corrects this test setup. At the shell's 8,176 KiB stack limit, 39 methods pass but the unchanged 5,000-call recursion control stops during its main shadow. A diagnostic-only supervisor observes signal 4, not an ASan/UBSan memory report. The identical control passes at 65,520 KiB. This establishes stack-limit sensitivity, not the exact faulting instruction; I leave that boundary visible in my roadmap.

My environment gate passes 60 C checks and ten lexical-scope methods. The separate evaluator and transpiler gate passes, including both assert-literal methods. My broader typechecker unit gate fails at the enum reduce control. Diagnostic checking finds an enum initializer parameter without its declaration name; a scratch two-site parameter metadata correction compiles and executes enum-reduce.nano. I keep that correction separate from the audited source and retain the unresolved production work on my roadmap. I do not report the entire unit gate as passing.

My checker and emitter component builds execute their shadows; the changed-function shadow presence check also passes. I retain earlier failures, including the first fixture's unparenthesized nested tuple indices, the initial qualified callback refusal, and the old self-hosted checker's generic callback refusal.

The module-owned equal-name union collision remains open. These checks do not establish full 5.1 completion or final Linux/Darwin candidate qualification.

My unchanged-source compiler-product audit passes all eleven steps in 581.173 seconds (product/manifest.json). The resulting native self-hosted compiler passes all 40 tuple/generic methods (selfhost-product-suite.log). I retain the exact source-input hashes and patch. The enum metadata probe is not part of that audited compiler source.
