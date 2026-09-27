# My runtime callback alias repair

I preserve an immutable runtime callback local's declared signature and value
without requiring a static function declaration. I propagate an exact target
only when known. I still check the complete local signature and evaluate the
initializer before introducing its name. Functional array operations retain
their separate static-target admission checks.

Fresh `make -j2 bootstrap` passes, including selected dependency/root shadows,
both compiler stage smoke tests and the installed compiler with the C seed
removed. Native binary equality is not established by this gate. I retain the
first failed run: my new shadow initially confused a parameter ASTLet with a
body declaration. Selecting through the function body corrected the fixture.

The complete command `python3 -m unittest -v tests.test_declared_array_push_identity
tests.test_generic_function_values tests.test_cseed_union_signatures` passes
all 34 methods ordinarily (31.813 seconds) and with generated-product
instrumentation (61.646 seconds). All original six declared-push methods remain
unchanged; my seventh method checks two runtime targets through one alias chain.
The compiler stages themselves are ordinary bootstrap products.

For generated-product instrumentation I selected:

```
NANO_CC=/opt/homebrew/opt/llvm/bin/clang
NANO_CFLAGS=-O0 -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer
NANO_LDFLAGS=-fsanitize=address,undefined -L/opt/homebrew/opt/openssl@3/lib
ASAN_OPTIONS=detect_leaks=1:detect_stack_use_after_return=1
UBSAN_OPTIONS=halt_on_error=1
NANO_SHADOW_TIMEOUT_SECONDS=60
```

The shadow deadline matches CI; individual test compiler deadlines are unchanged.
The separate exact-scalar-callback evidence now also includes the passing fresh
ASan/UBSan translator gate (2,582 assertions).

My first two-target fixture also tested a parameter and local with the same
name. The C seed accepts it but emits a C redeclaration. I preserve that source
as same-name-reproducer.nano and its failure log under separate
`task_b417476726ad47e58cd2bbe6c299e15b`. The final two-target regression uses a
distinct local name, matching the original reported contract. The producer
shadow still checks same-name initializer binding. I do not claim C-seed
same-name alias support. The transition log used a rebuilt Stage 1 and the
previous Stage 2; final ordinary/instrumented logs use both rebuilt stages.

This locally repairs the original failures in task_560bf9f1fca645d7aff9b71004ee3859.
Final hosted qualification, the separately recorded defects and all remaining
#522 acceptance requirements are still open.
