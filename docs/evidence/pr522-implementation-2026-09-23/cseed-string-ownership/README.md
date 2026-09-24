# My C-seed string ownership repair

I continue `task_02204077a4d34d4aa5ce20ef7054e113` from pushed source
`54355fc66`. Its instrumented Stage 1 smoke retained four owned
`path_canonical` results, [4,100 bytes total](../cseed-list-ownership/instrumented-stage-smoke.log).

I add explicitly declared, same-source typed string-release companions to my
C-seed ownership boundary. My [contract](../../../CSEED_FOREIGN_STRING_OWNERSHIP.md)
states the accepted parameter types, alias lifetime, explicit-release behavior,
null handling and process-exit cleanup. My filesystem provider ABI is unchanged.

## Retained intermediate evidence

- `initial-ownership-tests.log`: eight methods pass before broadening the real
  path test to all six helpers.
- `pre-gc-stage-build.log`, `pre-gc-stage-smoke.log`, and
  `pre-gc-stage-inputs.json`: the fresh instrumented compiler and original
  120-second hello smoke pass after raw foreign-string adoption. This source
  predates the defensive null-parameter guards and GC shutdown addition.
- `pre-gc-source-hashes.json` and `pre-gc-install.log`: after adding those guards,
  actual instrumented install/uninstall passes in 434.736 seconds. It exercises
  external scalar compilation through absolute, relative, PATH and symlink
  invocation, an invalid explicit override and prior-output preservation.
- `gc-facade-failure.log`: the expanded strict-root test exposes 237 bytes in
  eight allocations from four GC-managed path facades. I file
  `task_c0a10b52512d4fd28c4a01c8728a1648` and register checked native GC shutdown.
  I retain the same test and leak configuration.

## Corrected ownership tests

`ownership-tests.log` records all eight methods passing in 69.571 seconds after
GC shutdown. The counted provider checks function-value invocation, release
through a function value, returned aliases, borrowed literals, null results and
actual exit cleanup. The real filesystem case checks all six path helpers.
The shared owner and capture cases retain allocation/registration failure,
concurrency, nested list and hashmap aliases, cross-module release and exit
checks. Generated products use ASan/UBSan/leak/UAR with conservative globals,
stacks and registers excluded from leak roots. The runtime owner harness
compiles its runtime sources with instrumentation directly.

`hook-control.nano` and `hook-controls.log` retain passing `--profile` and
`--trace` runs with the same strict ASan/UBSan/leak/UAR configuration. Both
produce their diagnostic output and exit successfully. I have not qualified
the separate gprof execution path on this host.

I do not claim every linked runtime or provider object is instrumented merely
because generated products carry sanitizer flags. Full hosted partitions and
Linux qualification remain separate obligations.

## Final-source compiler and package qualification

`installed-instrumented.log` passes the actual installed-package test in
443.166 seconds after rebuilding both compiler stages with the recorded
ASan/UBSan product flags. `stage-smoke.log` and `stage-inputs.json` retain a
separate passing original 120-second Stage 1 hello smoke, compiler hash,
source hashes and exact environment. Leak/UAR checks remain enabled.

`final-gates.log` passes `make test-transpiler test-gc-struct test-dyn-array`,
including both assertion-literal methods, all 10 GC-struct tests and all 27
dynamic-array tests. This command uses ordinary host build flags; it does not
replace the preceding instrumented package check or claim a second fresh
ordinary bootstrap. I retain the broader platform/partition obligations.

The original matched Darwin compiler/install leak failure is corrected. My
MAC lifecycle records still require their normal review transition; local
passing evidence does not bypass that process.

## Remaining PR work

`callback-still-fails.log` reproduces the same-name callback C redeclaration
with this repaired seed. It remains under
`task_b417476726ad47e58cd2bbe6c299e15b`. My lexical native-name mapping handles
only `_` bindings today; ordinary shadowed names need distinct storage names
while their initializers still observe the previous binding.

#522 remains draft. These repairs do not establish current fixed points,
complete LLVM/Wasm coverage, or all final hosted partitions.
