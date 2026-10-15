# My C ownership effect producer checkpoint

I lower source effect declarations, performs and handlers into exact EFFECTS ownership contracts and verified NanoISA. I move resource arguments and results, isolate handler resource parameters, retain scalar captures and lexical exits, and finish consumed destructuring shells before suspension. I preserve canonical selected-union pattern identities. I accept public resource declarations in my C parser and resolve ownership helper calls by defining program or the resolved function body instead of spelling alone.

My final `local-calls.log` passes four test methods, including the reused thirteen-group resource frontend matrix, fourteen execution cases, imported resource/effect identities with same-named public/private helpers and dependency shadows, and ambiguous operation refusal with prior output preservation. Every positive source compiles selected shadows, verifies and executes in NanoVM, then executes generated C under strict warnings, ASan, UBSan and leak detection using Homebrew LLVM. This qualifies generated native execution; I do not claim full compiler instrumentation.

My `selected.log` passes 49 neighboring selected/generic ownership and pattern methods with `NANOLANG_AFFINE_COMPILERS=nanoc_c`. My parser gate passes. My runtime gate rebuilds affine verification and executes the existing twenty VM/native cases with allocation-failure cleanup checks. These are targeted checks, not full release qualification.

I retain earlier failures: unsupported declarations, native no-perform warning, uncanonicalized union patterns, void expression returns, public resource grammar, and name-only helper collisions. The initially ambiguous imported operation test used identical unqualified operation names; I preserve its refusal as a negative control and use distinct operation names in the positive imported-identity fixture. This follows the existing unique-effect inference contract.

I reproduce the source gate with `make -f Makefile.gnu test-owned-effect-c-source`, the parser gate with `make -f Makefile.gnu test-parser`, and the runtime gate with `make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang test-affine-effect-runtime`.

I still require independent Nano producer execution and fresh compiler-product qualification. Global/effect combinations, collection signatures, borrowed continuations and broader 5.1 acceptance remain open. I retain the bounded existing ownership profile; this checkpoint does not establish full language coverage.
