# My first useful interpreter source checkpoint

At `ee286d683f2834d7179b208d89ae0804beea2e0a`, Apple run [42512291803](https://github.com/jordanhubbard/nanolang/actions/runs/42512291803) passes my standalone dispatcher/GC sanitizer controls, all five VM methods, native extreme-bound controls, ignored-invalid-call publication refusals and lexical callee identity. Both all-builtin/U8 source builds complete interpreter shadows and reach C compilation, where the unchanged warning policy rejects my fixture's binary == binary self-comparison. No native all-builtin production program executes from those two attempts.

I retain exact early artifact text/hash metadata. I correct the fixture to compare distinct bindings that alias the same owner, preserving the identity assertion and compiler warnings. Actual O0/O2 native execution remains required. This is a fixture/compiler boundary, not infrastructure or full runtime qualification.
