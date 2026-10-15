# My native callable-analysis checkpoint

I resolve module-local function target sets before native representation
inference. My ordinary focused gate passes 204 checks. Including the archived
source-generated returned-function module passes 300 checks, including all four
expected call-site target sets. Fresh address/undefined-behavior sanitizer objects
pass the same fixture with leak detection enabled. My manifest binds source and
log hashes and verifies actual ASan/UBSan object instrumentation.

I retain the initial missing-header build, decoded-branch-index failure and
unsupported-opcode diagnostic regression. I correct the header and branch
accessor, and run the seeded analysis only when a module contains both function
references and indirect calls. The existing unsupported-opcode assertion remains
unchanged. My final broader native gate passes all 2,431 structured-C checks, 2,527 shape checks, the callable checks and adjacent control methods; I retain it as `native.log`.

The same original three native returned-call methods still fail during native
representation classification. Target analysis does not yet implement function
storage or checked native dispatch. I retain their actual failure log rather than
claiming language parity or release readiness.

My reproducible focused commands are:

```sh
make test-nvm2c-callables
obj/test_nvm2c_callables docs/evidence/computed-call-lowering-20261007/returned-functions.nasm
ASAN_OPTIONS=detect_leaks=1 make -j2 \
  'CC=/opt/homebrew/opt/llvm/bin/clang -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer' \
  OBJ_DIR=/private/tmp/nanolang-callable-analysis-sanitized/obj \
  BIN_DIR=/private/tmp/nanolang-callable-analysis-sanitized/bin test-nvm2c-callables
ASAN_OPTIONS=detect_leaks=1 /private/tmp/nanolang-callable-analysis-sanitized/obj/test_nvm2c_callables \
  docs/evidence/computed-call-lowering-20261007/returned-functions.nasm
```

My broader native gate uses `make -j2 test-nvm2c` with an explicit Homebrew LLVM
`cc` symlink directory first on PATH. My sanitizer driver now checks instrumentation
of `nvm2c_callables.o` alongside the translator and shape solver.
