# String equality in canonical products

I repair `task_2cf42468d6094fa9b8396fa9de6e75e3` against `0d28c919b`.
The [prior checkpoint](../scalar-string-conversion/comprehensive.log) retains
my original `undefined function str_equals` failure in the unchanged
`tests/unit/test_fstring_comprehensive.nano`.

I map the source builtin to `STR_EQ`, matching my C frontend. My checked
two-string lowering preserves operand order and requires exact arity/types.
Declared functions and local function values retain their own behavior.
I reject a non-callable bound value instead of substituting the builtin.

My first lowering attempt passed the VM route but exposed missing `STR_EQ`
support in native translation. `neighbors.log` retains all four failing
native subtests. I then added classifier and emitter cases to `nvm2c` and
classified the result as bool. Native comparison uses string contents, within
my existing string representation and embedded-NUL refusal boundary.

## Verification

- `bootstrap.log`: fresh compiler stages and smoke checks pass with the final
  frontend source, before the native translator correction.
- `emitter.log`: 86 bytecode comparisons and all 91 emitter methods pass
  (342.102 seconds for the Python suite). This gate uses the final frontend
  source and the preceding translator; it does not cover the new opcode.
- `translator.log`: final translator gate passes all 2,637 translator and
  1,614 shape assertions, plus opcode-coverage and sanitizer-driver checks.
- `translator-sanitized.log`: all 2,637 translator and 1,614 shape assertions pass in fresh isolated ASan/UBSan builds, with leak and use-after-return checks enabled; the driver verifies instrumentation symbols in translator and shape objects.
- `neighbors-corrected.log`: all 53 ordinary neighboring methods pass
  (53.261 seconds), using both final compiler stages and the corrected translator.
- `source-sanitized.log`: all eight string-conversion/equality methods pass
  (17.306 seconds), including the unchanged comprehensive f-string fixture
  through C seed and both canonical native/VM routes.

My raw translator edge tests now include `STR_EQ` alongside prefix/suffix
comparisons: empty strings, unequal lengths, equal contents, UTF-8 and control
bytes. My boolean producer check observes the result tag. Source tests also
check single left-to-right evaluation, direct returned bools, declared/local
bindings, invalid operands/arity, and prior-output preservation.

The ordinary neighboring command is:

```sh
python3 -m unittest tests.test_canonical_string_conversion \
  tests.test_cseed_binding_names tests.test_native_underscore_bindings \
  tests.test_match_expression_scope tests.test_declared_array_push_identity \
  tests.test_generic_function_values tests.test_cseed_union_signatures
```

For the eight-method source sanitizer run I set both `CC` and `NANO_CC` to
`/opt/homebrew/opt/llvm/bin/clang`, `NANO_CFLAGS` to
`-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all`,
`NANO_LDFLAGS=-fsanitize=address,undefined`,
`ASAN_OPTIONS=detect_leaks=1:detect_stack_use_after_return=1`, and
`UBSAN_OPTIONS=halt_on_error=1`, then run
`python3 -m unittest tests.test_canonical_string_conversion`.
This instruments generated native products, not every compiler/VM unit.
I preserve the gate's assertions, `-Werror` and deadlines.

`source.json` records source hashes and toolchain. These local results do not
close the full PR522 acceptance ledger. Ordinary aggregate globals, aggregate
formatting, full platform qualification, final fixed points and applicable
LLVM/Wasm coverage remain required.

I run the isolated translator instrumentation with:

```sh
ASAN_OPTIONS=detect_leaks=1:detect_stack_use_after_return=1 \
UBSAN_OPTIONS=halt_on_error=1 \
python3 scripts/run_nvm2c_sanitizers.py --cc /opt/homebrew/opt/llvm/bin/clang
```

The driver uses fresh object, binary and runtime-library directories and
instruments its generated C child commands. It removes those directories
on completion. The native-invariant diagnostics in the logs belong to
negative tests; both complete translator runs finish with zero failed checks.

MAC rejects direct `open -> completed` closure. I retain that response and
attach the verified evidence without bypassing its review lifecycle.
