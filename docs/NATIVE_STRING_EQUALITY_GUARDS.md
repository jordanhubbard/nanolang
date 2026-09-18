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
