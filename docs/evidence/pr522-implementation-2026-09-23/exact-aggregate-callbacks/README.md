# I qualify exact aggregate callback targets

My [contract](../../../NANOISA_EXACT_AGGREGATE_CALLBACKS.md) states the supported
single-target, forward-only call profile and the remaining broader boundaries.
I carry pending, exact and mixed/unproved target identity separately, preserve
it through local and interprocedural flow, and use my existing direct-call
shape and cleanup checks for an established aggregate/array target. Native
output guards the evaluated target identity. I do not infer an aggregate result
from an unrelated scalar function with matching arity.

The original 18-method generic function-value suite passes ordinarily across
the C seed and both native stages, restoring its ten remaining native failures.
The complete translator gate passes 2,446 assertions ordinarily and with fresh
ASan/UBSan instrumentation. The sanitizer driver verifies actual translator and
shape-object instrumentation and disables leak detection as documented.

New raw-module controls cover callback selector effects, returned identities,
local storage, forwarding, tail calls, worker/main declaration order, an
unrelated float-result decoy, and a mismatched array-argument tag. All six
function-table permutations of the original nested-array module assemble,
execute in NanoVM, translate, compile with strict warnings and ASan/UBSan, and
execute with leak detection. I retain the exact commands and outcomes.

I separately execute the existing named-callback refusal methods for local/
computed functional reduction, global mutation aliases and u8. All pass their
original specific diagnostic and prior-output checks. I do not count this as
rerunning that suite's separately built emitter matrix.

## I preserve explicit zero-argument C signatures

The first broader generated-program sanitizer run exposes three C-seed-only
function type mismatches in generic-result, local and forwarded callbacks.
The callback typedef has `(void)` while the generated declaration and definition
have `()`. I retain that 27-method failure, the generated C and an isolated
old executable that exits on UBSan. The same source rebuilt with explicit
`(void)` declarations and definitions executes successfully.

I apply explicit empty parameter lists to language functions, generic instances
and imported prototypes. The corrected 27-method generic/purity/returned-call
run passes with generated programs instrumented by Homebrew Clang using
ASan/UBSan, no sanitizer recovery and leak detection. Compiler executables and
external libraries remain ordinary. Exact flags are:

```
CC=/opt/homebrew/opt/llvm/bin/clang
NANO_CFLAGS=-O1 -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
```

The owning `make test-transpiler` gate triggers fresh bootstrap after the C seed
change. Both native-stage smoke tests and the no-C-seed check pass, followed by
the StringBuilder boundary tests and two assertion-literal methods. Native
binaries differ; I do not claim canonical bytecode fixed points. The failed
`make nanoc_c` invocation is retained as a command-selection error; the actual
build target is `make bin/nanoc_c`.

The fresh-stage repetition passes all 27 methods (70.401 seconds). The generic
suite selects the C seed and both native stages; the adjacent purity and
returned-call suites use their default compiler selection. The complete
13-method resource-callback suite retains exactly four failures: fixed resource
parameters and results in each native stage. I retain those terminals.
General multiple-target and backedge callback inference, resource ownership
transfer, Linux shadow timeouts, hosted acceptance and final-source fixed points
remain outside this qualification.
#522 remains draft.

My C signature task description records these verified repairs, but the MAC
hub rejects a direct failed-to-completed transition. I do not claim it is closed.
