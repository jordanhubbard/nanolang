# My native callable execution checkpoint

I now execute named function values and computed callees through native C calls.
My original three returned-call methods pass, together with sixteen VM/native
parity methods and three native sanitizer methods: 22 methods in 4.988 seconds.
The native gate passes 2,431 checks and the shape solver passes 2,535 checks,
including an address/undefined-behavior/leak sanitizer run.

I preserve function tags and target identity through locals, globals, arguments
and returns. Indirect calls reuse direct-call argument conversion and record
pointer transport. I test zero-valued target IDs, forged numeric/void tags,
negative/out-of-set targets, sixty possible targets and record/array values across
observed garbage collections. The sanitizer tests retain strict C11 warnings and
check that invalid callables trap at native guards rather than sanitizer errors.

I retain the initial return-shape failure, integer local/heap-root assumptions,
and cumulative temporary exhaustion across exclusive dispatch arms. The original
acceptance assertions remain unchanged. My native parity subclass now executes
the existing function-value assertions through the native product, including
prior-output preservation for rejected source.

The source producer used in the focused tests is
`/private/tmp/nanolang-computed-calls-final`, built from the unchanged function
lowering at `4f56b353c`; the tests use the current checkout's translator. This is
not clean final-revision release qualification. A fresh clean compiler-product
run is required for the committed native changes.

My container fixture is a remaining release blocker. The C seed compiles it and
NanoVM executes it, while my self-hosted producer refuses `unsupported local type
Ops` and the native translator refuses the function-valued aggregate field. I
retain the actual source, module, commands, hashes and terminals. The separate
`o.op` call spelling is rejected by the C seed too; that spelling is not evidence
of a regression. The accepted container fixture calls a function copied from its
record field and selects a function from an array.

I reproduce the focused product checks with:

```sh
NANOLANG_SELFHOST_COMPILER=/private/tmp/nanolang-computed-calls-final \
  python3 -m unittest -v tests.test_selfhost_returned_calls tests.test_selfhost_function_values tests.test_native_callables
make -j2 test-nvm2c
```

I place an explicit Homebrew LLVM `cc` symlink directory first on PATH for these
sanitizer tests. `make test-native-callables` runs the three native safety methods;
my complete compiler-product target now includes them as well.
