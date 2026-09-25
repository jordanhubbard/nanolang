# Scalar boolean conversion qualification

I reproduced all four recorded native-stage failures at `1971a80182becf8f8e10c17d4297d878c1be6c66`: my canonical emitter rejected `cast_bool` before execution or the expected failing shadow. The original four-method test run is retained in `baseline.log`.

I recognize undeclared scalar `cast_bool` calls, infer their boolean result, require exactly one int/u8/float/bool operand, and emit `CAST_BOOL`. I preserve fractional floats instead of truncating through `CAST_INT`. My emitter shadows check arity/type refusals and preserve declared functions named `cast_bool`.

My expanded source fixture exercises zero/nonzero u8, zero/negative integers, booleans, fractional positive/negative floats and signed floating zero. The added driver test executes the verified module in NanoVM and its standalone native translation. `vm-native.log` records the ordinary pass. `vm-native-instrumented.log` records the same native translation with ASan/UBSan, leak detection and stack-use-after-return detection. The emitter, VM and host runtime objects in that run are ordinary builds; this does not qualify the complete instrumented host-runtime gate.

Commands:

```sh
NANO_CC=/opt/homebrew/opt/llvm/bin/clang NANO_SHADOW_TIMEOUT_SECONDS=60 make -j2 bootstrap3
NANO_CC=/opt/homebrew/opt/llvm/bin/clang NANO_SHADOW_TIMEOUT_SECONDS=60 make nanoisa_emit
NANO_CC=/opt/homebrew/opt/llvm/bin/clang python3 -m unittest tests.test_nanoisa_emit_driver.NanoisaEmitDriver.test_scalar_boolean_casts_agree_in_vm_and_native -v
NANO_CC=/opt/homebrew/opt/llvm/bin/clang NANO_NATIVE_TEST_CC='/opt/homebrew/opt/llvm/bin/clang -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer' ASAN_OPTIONS=detect_leaks=1:detect_stack_use_after_return=1 UBSAN_OPTIONS=halt_on_error=1 python3 -m unittest tests.test_nanoisa_emit_driver.NanoisaEmitDriver.test_scalar_boolean_casts_agree_in_vm_and_native -v
```

I track string conversion separately as `task_ab9646880e7f483a9f7bc359876f0415`: the interpreter recognizes only textual `true` and `1`, whereas the legacy VM source frontend emits raw pointer truthiness. Native AOT refuses string truthiness. This scalar correction does not settle that source-language discrepancy or change the raw ISA.

Final hosted acceptance, final-source fixed points and the remaining PR gates are still open.

My fresh bootstrap passes both stages, installed execution and the no-C-seed smoke check (`bootstrap.log`). These stages were built from the corrected canonical emitter using the pre-float-helper C seed; the subsequent C-seed helper correction is separately rebuilt and tested. I do not claim final-source fixed points from this bootstrap.

The expanded runtime fixture exposed a C-seed-only defect: interpreter shadows passed, but the native product truncated `0.5` through an integer helper (`cseed-execution.log`). I now select a double-argument helper for the builtin float call, retaining existing ordered argument evaluation. Declared functions and captured callees do not select that helper. I track this repair as `task_8193e913303f428f8392d1da906ea7ea`. The first helper-selection attempt did not recognize registered builtin function entries; `native-stages.log` retains that failure. The corrected selection recognizes a bodyless, non-extern builtin entry.

All four owning methods pass against the rebuilt C seed and both native stages (`native-stages-final.log`, 2.966 seconds). They also pass with the unoptimized private ASan/UBSan seed and instrumented generated products (`native-stages-instrumented.log`); leak and stack-use-after-return detection are enabled. The ordinary seed was restored byte-for-byte afterward. Both native stage compiler executables remain ordinary builds. All eleven emitter-driver methods pass (`emitter-suite.log`, 2.948 seconds).

The owning ordinary command sets `NANO_CC` and `NANOLANG_U8_C_COMPILERS` to Homebrew Clang and `NANOLANG_U8_SANITIZER_LEAKS=1`, then runs `python3 -m unittest tests.test_selfhost_native_u8 -v`. The instrumented run adds `NANO_CFLAGS='-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer'`, `NANO_LDFLAGS='-fsanitize=address,undefined'`, `ASAN_OPTIONS=detect_leaks=1:detect_stack_use_after_return=1`, and `UBSAN_OPTIONS=halt_on_error=1`. Original per-test deadlines and publication assertions remain unchanged.
