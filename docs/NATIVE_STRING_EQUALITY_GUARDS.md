# My native tagged string equality guard contract

I track `task_39453e3f2c76454dab9afd3e346c2483` before this repair.
My enum same-module gate fails GCC O2 compilation at nvalue_equal's fallback
strcmp: constant propagation reports a possibly null argument under Werror.
I retain `/tmp/nanolang-llvm-enum-focused.log` and the full gate failure.

I preserve existing tag comparison, numeric equality and array/map pointer
identity. After those cases, I match VM string equality: identical pointers
compare equal, either missing pointer otherwise compares unequal, and only
two nonnull distinct pointers reach content comparison. I do not suppress
warnings, assume nonnull input or admit new heap kinds.

I require ordinary enum/numeric mixed comparison modules at strict GCC/Clang
O2, VM parity, and generated native ASan/UBSan/LSan. An isolated repaired-helper
control checks null/null, null/text, shared pointers and distinct same-content
strings against the VM value helper. No retained failing product is executed.

On main `5193858f`, production `3e7763ef` passes three strict GCC O2 methods
in 1.340 seconds with ASan/UBSan/LSan. Twenty-three Clang methods pass in
16.699 seconds: three new equality methods plus ordinary enum, typed-enum and
tagged arithmetic controls. The isolated generated-helper harness compares
36 pairs against the actual VM val_equal implementation, including null/null,
null/text, identical pointers, distinct equal text, unequal text and empty text.
I retain `/tmp/nanolang-string-equality-{build,gcc,clang}.log`. This repairs the
observed compile prerequisite; full LLVM/Wasm enum acceptance remains separate.
