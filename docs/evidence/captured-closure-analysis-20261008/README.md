# My captured-callable target constraints

I extend my native target-analysis pass with closure creation and flattened
upvalue loads/stores. I retain capture provenance through returned nested
closures, later writes, captured array aliases and multiple instances of one
body. I conservatively union instances for analysis; runtime environments must
still remain distinct. Invalid target indices, capture counts, depths, slots
and stack underflow are refused.

My complete `make -j2 test-nvm2c` run passes all 2,431 native checks.
My fresh normal and ASan/UBSan/LSan focused fixtures each pass 379 checks. Adding
the unchanged returned-function source assembly passes 475 checks on both builds.
My first extended alias fixture reused a local for unrelated callable values,
while asserting separate target sets. My flow-insensitive analysis correctly
merged those assignments. I retain its four failed assertions and use distinct
locals to test producer isolation; I did not weaken the target-set assertions.
The earlier 268-check log predates that added alias/instance fixture.

```sh
make test-nvm2c-callables
obj/test_nvm2c_callables docs/evidence/computed-call-lowering-20261007/returned-functions.nasm
ASAN_OPTIONS=detect_leaks=1 make -j2 \
  'CC=/opt/homebrew/opt/llvm/bin/clang -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer' \
  OBJ_DIR=/private/tmp/nanolang-closure-analysis-sanitized/obj \
  BIN_DIR=/private/tmp/nanolang-closure-analysis-sanitized/bin test-nvm2c-callables
ASAN_OPTIONS=detect_leaks=1 /private/tmp/nanolang-closure-analysis-sanitized/obj/test_nvm2c_callables \
  docs/evidence/computed-call-lowering-20261007/returned-functions.nasm
```

I retain [the source-level failure baseline](../captured-closure-baseline-20261008/README.md).
This pass does not add native environment allocation, capture roots or reclamation,
and does not repair my self-hosted lexical scope. Those remain required before
captured-closure execution or full 5.1 acceptance.
