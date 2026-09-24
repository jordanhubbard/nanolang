# Primitive-list payload qualification

I reproduced all ten nominal-order subcase failures at `1d20a33ace24b26a42a069fdbfc80cfface1a4f5` (`baseline.log`). Four are missing cycle refusals, four reject primitive-list union locals, and two fail translation of an integer-list record field.

I traced the record failure to `AGG_PACK` field 0 in function 2 at offset 17. My canonical `list_int_new` emitted `ARR_NEW 1`, which native translation retains as a legacy polymorphic empty-array placeholder. The resulting array element shape remained unresolved through the record projection. I changed only that constructor to `ARR_LITERAL 1 0` in the retained assembly; assembly, native translation, strict Clang compilation, native execution and VM execution all succeeded. `list-record.nasm`, `typed-list.nasm` and `translate.log` retain that diagnosis.

My source emitter now preserves the known integer-list type with an explicitly typed empty literal. Other list tags retain their existing exact constructors. I also admit `List<int>` and `List<string>` as union payloads and concrete generic arguments. This does not change raw ISA placeholder compatibility or admit resource-bearing list elements.

All twelve emitter-driver methods pass (`emitter-suite.log`). The new regression mutates a list through a record alias and reads the same content through a generic union and the original list. Both integer and string cases execute in NanoVM and standalone native output. Native output also passes ASan/UBSan with leak and stack-use-after-return checks (`list-payload-instrumented.log`); compiler executables and VM runtime are ordinary builds.

I retain missing by-value-cycle refusals as incomplete work in `task_2a1eaa9ecdc94d489c5f731d4daf416d`. Final hosted acceptance and final-source fixed points remain open.

Fresh bootstrap passes both stages, installed execution and the no-C-seed smoke check (`bootstrap.log`). The unchanged ten-method nominal-order suite now has four failing subcases, all in the two cycle-refusal methods; its eight positive methods pass (`native-stages.log`, 5.676 seconds). The two list-field methods also pass on both stages with instrumented native products (`native-stages-instrumented.log`, 3.519 seconds). I did not mark the complete nominal-order gate passed.

Commands:

```sh
NANO_CC=/opt/homebrew/opt/llvm/bin/clang NANO_SHADOW_TIMEOUT_SECONDS=60 make nanoisa_emit
NANO_CC=/opt/homebrew/opt/llvm/bin/clang NANO_SHADOW_TIMEOUT_SECONDS=60 make -j2 bootstrap3
NANO_CC=/opt/homebrew/opt/llvm/bin/clang python3 -m unittest tests.test_nanoisa_emit_driver -v
NANO_CC=/opt/homebrew/opt/llvm/bin/clang python3 -m unittest tests.test_native_nominal_order -v
NANO_CC=/opt/homebrew/opt/llvm/bin/clang NANO_CFLAGS='-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer' NANO_LDFLAGS='-fsanitize=address,undefined' ASAN_OPTIONS=detect_leaks=1:detect_stack_use_after_return=1 UBSAN_OPTIONS=halt_on_error=1 python3 -m unittest tests.test_native_nominal_order.NativeNominalOrder.test_primitive_list_union_fields_keep_runtime_typedefs tests.test_native_nominal_order.NativeNominalOrder.test_primitive_list_record_fields_keep_runtime_typedefs -v
NANO_CC=/opt/homebrew/opt/llvm/bin/clang NANO_NATIVE_TEST_CC='/opt/homebrew/opt/llvm/bin/clang -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer' ASAN_OPTIONS=detect_leaks=1:detect_stack_use_after_return=1 UBSAN_OPTIONS=halt_on_error=1 python3 -m unittest tests.test_nanoisa_emit_driver.NanoisaEmitDriver.test_primitive_lists_retain_identity_in_records_and_generic_unions -v
```
