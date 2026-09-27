# My C-seed lexical storage names

I repair `task_b417476726ad47e58cd2bbe6c299e15b` from source `73d736b40`.
My prior C output redeclares a callback parameter in its own scope:
`BinaryOp_0 callback = callback;`. `before.log` retains that refusal.
`original-reproducer.log` compiles and runs the unchanged two-target source
successfully with the repair.

I allocate distinct native storage for shadowed locals. I publish the new
name after emitting its initializer, retain source names in checked metadata,
restore mappings when scopes end, and mask outer names for loop and match
bindings. Effect capture addresses use the same mapping. Existing underscore
storage names retain their prefix and cleanup behavior.

## Passing checks

- `ordinary-neighbors.log`: all 22 lexical-name, lexical-scope and native-effect
  methods pass. `instrumented-neighbors.log` passes the same 22 methods with
  Homebrew LLVM, ASan/UBSan, leak detection and stack-use-after-return checks.
- `compiler-selection-failure.log`: my first instrumented command selected
  Homebrew LLVM only for generated products. The direct C effect harness used
  Apple Clang and rejected `detect_leaks=1`. I retain this command error and
  correct `CC` as well as `NANO_CC`, keeping leak checks enabled.
- `final-regression.log`: the final regression passes with and without `--tco`.
  It checks repeated parameter/local callback aliases, two runtime targets,
  mutable nested shadowing, loop masking, guarded/unguarded statement and
  expression matches, scope restoration and integer same-name initializers.
  The final instrumented 45-method command also passes this regression.
- `bootstrap-transpiler.log`: fresh Stage 1 and Stage 2 builds and both smoke
  tests pass, including the compiler running with the C seed removed. The full
  transpiler gate and both assertion-literal tests pass. Binary equality and
  current canonical fixed points are separate requirements.

My product instrumentation environment is:

```
CC=/opt/homebrew/opt/llvm/bin/clang
NANO_CC=/opt/homebrew/opt/llvm/bin/clang
NANO_CFLAGS=-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all
NANO_LDFLAGS=-fsanitize=address,undefined
ASAN_OPTIONS=detect_leaks=1:detect_stack_use_after_return=1
UBSAN_OPTIONS=halt_on_error=1
```

The compiler stages and previously linked runtime/provider objects are not
thereby fully instrumented. Existing test and compiler deadlines are unchanged.

## Broader failures retained

`stage-neighbors.log` runs all 45 methods from the binding-name, underscore,
match-expression, declared-array-push, generic-function-value and C-seed-union-
signature suites. Four subtests fail in the unchanged underscore suite: both
native stages refuse integer f-string conversion before reaching its normal
or intentionally failing shadow. `stage-instrumented.log` retains those four
failures plus two C-seed vectorization failures. The callback, signature and
match-expression suites pass in both commands. Neither complete 45-method run
is green.

I relink the exact `73d736b40` transpiler sources against the unchanged compiler
objects, replacing only the transpiler object, and rebuild Stage 1. This is a
relinked control, not a second full checkout. Module NanoLang sources are
freshly transpiled by `compile_module_to_object`; the checked-in compiler and
runtime sources otherwise match the baseline. `inputs.json` retains hashes.
The first baseline executable location outside the repository could not find
runtime headers; `baseline-link-and-location-failure.log` retains that setup
failure. Placing the same seed in the repository's bin directory permits the
baseline build (`baseline-stage-build.log`).

- `baseline-underscore.log` reproduces the canonical `str_concat` refusal.
  `fstring-isolation.json` shows that integer interpolation fails with both `_`
  and an ordinary name, while repeated string bindings succeed. The lexer
  wraps interpolation in `to_string`; my next diagnostic step is its canonical typing and lowering. This remains
  `task_42ae2ce1524b49ac8c55ec4e74287307`.
- `baseline-vectorization.log` reproduces the C-seed LLVM optimization refusal
  under the same sanitizer flags and `-Werror`. I retain all diagnostics and
  flags under `task_31904c9a48804f94aa81b2559a97ee88`; suppressing the warning
  does not count as repairing emitted optimization requests.

#522 remains draft. These local lexical repairs do not close the retained
failures, full platform/sanitizer partitions, or remaining acceptance gates.
The MAC task retains its earlier failed lifecycle until normal review.
