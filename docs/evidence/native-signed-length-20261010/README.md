# My signed native string length

I map str_length to my existing int64_t-returning nl_cstr_length runtime
helper. My former mapping to C strlen returned size_t, causing signed index
comparisons to fail strict C compilation and promoting negative arithmetic to
unsigned values. My retained baseline reproduces three sign-comparison errors.

I pass both methods in tests.test_native_string_length after rebuilding
bin/nanoc_c and nano_virt. The source compares a negative index with a string
length in both operand orders, subtracts one from an empty string length,
checks the negative result and checks ordinary length. I execute mandatory
shadows and the native product, plus verified NanoVM and generated C with
LLVM ASan/UBSan. The existing evaluator suite, including registry lookup,
also passes. The generated C sanitizer run links the ordinary runtime object.

Commands:

```
make -f Makefile.gnu -j4 bin/nanoc_c nano_virt
NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang python3 -m unittest tests.test_native_string_length -v
make -f Makefile.gnu -j4 test-eval
```

This closes the native signed-length defect under #976; it does not qualify
the full 5.1 release candidate or its platform matrix.
