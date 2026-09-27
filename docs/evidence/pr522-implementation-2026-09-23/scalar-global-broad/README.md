# My completed scalar-global broad gate

I retain the corrected source-borrow and scalar-union gate after checkpoint
`180422cd0`. All 60 unittest methods pass: 56 source-borrow methods and four
scalar-union methods. I run:

```sh
NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang \
python3 -m unittest -v tests.test_source_borrow_emission tests.test_affine_scalar_union_source
```

I keep leak detection enabled. The first run selected Apple's sanitizer runtime
and failed 43 assertions with its unsupported-leak-detection diagnostic; its
full terminal remains in the adjacent optional-records checkpoint. The corrected
run takes 1,785.275 seconds and exits zero.

I preserve the normal compiler sources and binaries for this run. Concurrent
optional-record and record/union source prototypes use isolated files and
executables. This gate qualifies the scalar-global checkpoint; it does not
qualify the later source shape/checker/parser repairs or the full release.
