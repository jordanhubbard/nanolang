# I qualify the pinned native compiler product on Darwin

I run `CMD_TIMEOUT=1800 python3 -m unittest -v tests.test_one_ir_compiler.OneIrCompiler.test_compiler_bytecode_to_native_to_program` in an isolated checkout at `a36cf09d1`. I use the copied, hash-recorded providers from the completed `4cfad3737` raw bootstrap. The intervening commit changes the relocated compiler capture environment, not compiler product source.

I pass the complete method in **1052.879 seconds**. It translates compiler bytecode to a native compiler, runs its closure/managed-array/generic-environment checks, rebuilds the full compiler source with that native compiler, verifies the rebuilt module, and qualifies the resulting hello products. I retain the successful terminal and exact provider hashes here. The harness uses temporary product directories; I do not claim retained generated compiler binaries.

My first attempt fails after 103.836 seconds because the sparse checkout omits `docs/evidence/captured-closure-baseline-20261008/capture.nano`. I retain that failure and provision the exact tracked directory before retrying. I change no assertion, compiler behavior, or guest deadline.

This is Darwin evidence at the stated source/provider pins. It is not final-candidate Linux evidence or a result for the later combined U8, nested STRING and dispatch changes. The instrumented full-generation diagnostic is separate and does not inherit this pass.
