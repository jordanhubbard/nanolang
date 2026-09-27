# Scalar string conversion and C-seed loop directives

I retain this repair against `87436b7dc637d5e6b36d8e0a43005d08c5946a8c`.
`source.json` records source hashes and the host/toolchain. My previous
[matched baseline](../cseed-binding-names) records canonical integer-f-string
refusals and LLVM sanitizer vectorization failures before this correction.

I lower `to_string` and `cast_string` for int, u8, float, bool, string and
ordinary enums. I evaluate the operand once, preserve declared functions and
local callable bindings, and reuse my float formatting rule, including `-0.0`.
I leave vectorization to the C compiler: my C-seed array-loop emitter has no
proof of loop independence to justify a forced directive. I retain optimization
flags, `-Werror`, assertions and sanitizer settings.

## Verification

- `bootstrap.log`: fresh compiler generations and smoke checks pass.
- `emitter.log`: the complete `make test-transpiler test-nanoisa-src-nano` gate passes after rebuilding both stages: StringBuilder checks, two transpiler methods, 86 bytecode comparisons and all 91 emitter methods (390.991 seconds for the Python emitter suite).
- `final-seed.log`: rebuild after removing a stale comment passes.
- `neighbors-corrected.log`: all 49 ordinary methods pass (47.960 seconds).
- `neighbors-sanitized.log`: all 49 methods pass (118.918 seconds).
- `vector-check.log`: the two targeted C-seed methods pass under instrumentation.

I run the 49-method group with:

```sh
python3 -m unittest tests.test_canonical_string_conversion \
  tests.test_cseed_binding_names tests.test_native_underscore_bindings \
  tests.test_match_expression_scope tests.test_declared_array_push_identity \
  tests.test_generic_function_values tests.test_cseed_union_signatures
```

For the instrumented run I set both `CC` and `NANO_CC` to
`/opt/homebrew/opt/llvm/bin/clang`, `NANO_CFLAGS` to
`-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all`,
`NANO_LDFLAGS=-fsanitize=address,undefined`,
`ASAN_OPTIONS=detect_leaks=1:detect_stack_use_after_return=1`, and
`UBSAN_OPTIONS=halt_on_error=1`. This instruments generated native products;
it does not establish that every compiler, VM or linked runtime unit is
instrumented. Compiler source stayed fixed during these tests; the owning
make gate rebuilt the same stages concurrently.

My new tests cover scalar values, float formatting, left-to-right evaluation,
name binding, invalid arity and prior-output preservation. The original
underscore/shadow controls remain unchanged. `neighbors.log` and `focused.log`
retain the initial fixture error: `byte` was a reserved type name in the C seed.
I renamed that local to `octet`; I did not change its assertion.

## Remaining compatibility work

`comprehensive.log` retains an additional failure: the unchanged
`tests/unit/test_fstring_comprehensive.nano` passes through the C seed, then
canonical Stage 1 refuses `undefined function str_equals`. I track that under
`task_2cf42468d6094fa9b8396fa9de6e75e3`; this fixture is not qualified yet.

`aggregate-baseline.json` retains `(to_string [1, 2])`: the C seed prints the
expected representation, while the earlier canonical Stage 2 refuses it.
Aggregate formatting remains required under
`task_fe7abd6028d14ae387e70e0e83b8885e`. My scalar helper deliberately refuses
non-scalar operands; that is a boundary, not aggregate support.

These results do not close PR522. Full Linux/hosted qualification, final-source
fixed points, complete applicable LLVM/Wasm coverage, and the remaining
acceptance ledger still require evidence.

`hosted-checks.json` records the pushed baseline head's CI observation.
`hosted-units00.log` is the complete log for job `107795464069` in run
`36047708438`. Four global-resource boundary subtests fail because native
translation refuses aggregate globals. `global-boundary.log` reproduces all
four failures locally (15 methods, 4.672 seconds), including ordinary union
globals and an unused resource type parameter. I attach these to the existing
aggregate-global transport task `task_95796f5f49564ed4a911fd05a1aac5b4`.
Neither result establishes a sanitizer memory defect; both show a translation
refusal. Resource refusal controls pass. The hosted snapshot is not final CI
qualification.

MAC rejects direct `open -> completed` transitions for both repaired tasks.
I retain the closure responses and attach evidence without forcing lifecycle
completion or triggering an automatic merge.
