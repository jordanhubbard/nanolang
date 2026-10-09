# My C companion acquisition

At 1aac35b56 I retain immutable companions in the actual C import loader under
canonical source origins. I validate full-width origin indices and complete
string counts, reuse the first snapshot for an unchanged declaration, refuse
inconsistent rebinding and release successful prefixes with the environment.
I do not remove the unresolved-service checker/lowering refusal.

My [loader gates](loader-tests.log) pass the parser/binder/recursive-loader
fixture plus ten C-seed/NanoVirt CLI cases. I check same-basename origins,
symlinked source identity, sixteen-origin capacity, full-width invalid indices,
changed input after acquisition, inconsistent companion rebind, a failed second
acquisition, strict invalid/missing/symlink companions and prior-output retention.

My [selected sanitizer fixture](loader-sanitizer.log) rebuilds the actual loader,
environment destructor, snapshot reader, strict preparer, NSI/catalog parser,
cJSON and UTF-8 with LLVM ASan/UBSan and leak detection. Other common/runtime
objects remain ordinary. The instrumented lifetime/refusal fixture passes.

The common consumer tests pass 52 environment checks, ten lexical-scope tests,
native StringBuilder and two literal-emission methods, and the complete evaluator
suite. Their [combined first run](initial-neighbors.log) then reaches AOT tests,
where Apple cc refuses leak detection. An environment-only compiler override is
also ineffective because Make assigns CC. The corrected command supplies
`make CC=/opt/homebrew/opt/llvm/bin/clang test-nvm2c` and
`NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang`: all prerequisite suites and
[2,438 core AOT checks](aot-corrected.log) pass. I retain both unsuccessful
configurations; no assertion or sanitizer is removed.

My compiler-input archive contains the strict reader/provider closure once;
common-only and standalone NanoISA consumers both link successfully. This is
Darwin evidence. Final paired bootstrap, Linux, complete namespace/nominal
binding and executable File source admission remain open under #989/#976.
