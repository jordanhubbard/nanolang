# I converge callback targets before refusing them

I retain the preceding function-order diagnosis in `../generic-callback-trace`.
My classifier used to reject an indirect call before a later function could
establish its parameter representation. I defer only the missing-candidate
refusal until the existing inference reaches its final pass. Missing targets
still refuse before C publication; ambiguity and stack checks remain active.
I do not reorder functions or broaden the scalar-result boundary.

I add all six declaration orders for one record callback, execute each native
result, and retain a wrong-argument call with no matching target after
convergence. The complete ordinary translator gate passes 2,441 assertions.
The complete fresh ASan/UBSan translator gate also passes 2,441 assertions and
checks actual instrumentation symbols in the translator and shape objects.
Its existing driver disables leak detection; it does not establish leak freedom.

The complete original generic suite runs 18 methods and now has ten native
failures, down from fourteen: generic parameters and indirect record literals
pass on both native stages. Generic results, local/forwarded generic values,
nested arrays and imported nested-array callbacks still fail in both stages.
I preserve every original positive and negative assertion.

Four source methods (both restored positives and wrong nominal/generic argument
controls) pass across the C seed, Stage1 and Stage2 with generated programs
instrumented by Homebrew Clang using ASan/UBSan and leak detection. Their compiler
executables and external libraries remain ordinary. Exact command flags are:

```
CC=/opt/homebrew/opt/llvm/bin/clang
NANO_CFLAGS=-O1 -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
```

## I isolate the sanitizer archive

My first bootstrap fails linking `nano_vm`: the sanitizer driver uses private
object and binary directories but leaves `lib/libnano_file_runtime.a` shared.
That archive has ASan/UBSan references; its restored ordinary counterpart has
neither. I retain the failure and symbol/hash observations. The first fully
instrumented translator gate itself passes, but overwrites this ordinary build
artifact.

I give each sanitizer invocation its own `FILE_PUBLIC_LIBRARY` under the same
private temporary root and extend the isolation driver test. All three driver
methods pass. A second complete instrumented gate passes with the private
archive, while the restored ordinary archive retains its exact before/after
SHA-256. This is a demonstrated build-artifact collision and repair, not an
unexplained infrastructure retry. MAC task:
`task_c21164320d83478799f6c56c922426e3`.

Fresh corrected bootstrap passes, including both native stage smoke tests.
Native binaries differ; this does not establish canonical bytecode fixed points.
The parent generic-callback task remains open:
`task_915f6994f07742cea3c280c3e6743d14`. Target provenance, aggregate callback
results, resource ownership transfer, Linux shadow timeouts, complete hosted
gates and final-source canonical fixed points remain required for #522.
