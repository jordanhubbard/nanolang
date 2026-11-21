# My checked global-binding checkpoint

I implement #1147 in `0c8a282fc` and add explicit entry-point shadows in the
test-only commit `0cf0765e3`. I resolve checked global reads, writes and
function-valued calls by their declaration identity. I select current evaluated
storage, including qualified imports, while retaining ordinary local lookup.
I preserve selected callees before evaluating their arguments. I copy borrowed
global function values when a new binding needs its own callable metadata.

I do not change call-frame cleanup or claim that #986/#1149 is complete.

## My passing checks

| Gate | Linux ARM64 | Darwin ARM64 |
| --- | --- | --- |
| Original evaluator, typechecker, environment/scoping and module metadata gates | Pass | Pass |
| Original imported-global suite, C producer | 29 methods pass | 29 methods pass |
| Original imported-global suite, development NanoLang producer | 29 methods pass | 29 methods pass |
| Original scope witness and new runtime collisions | Five routes pass | Five routes pass |
| Qualified global storage, retained callable and callee-before-argument order | Four NanoISA routes pass | Four NanoISA routes pass |
| Actual C-interpreter qualified-global shadow report | Seven shadows pass | Seven shadows pass |
| Focused C-interpreter scalar/STRING ownership, ASan/UBSan | Pass, including default leak detection | Pass; I do not claim Darwin LeakSanitizer coverage |
| Combined decoder and scope gate at implementation pin | Seven methods pass | Seven methods pass |
| Schema and embedded-identifier controls | Pass | Pass |

My combined gate retains all5486 synthetic decoder cases per route, six unchanged
example products,256 primary widths and256 File scans. It takes81.605s on Linux
and54.334s on Darwin. Those observed wall times are not a performance comparison.
Five module extension profiles remain pending. Decoder success does not grant
instruction or host execution authority.

The five routes are C-seed native and both C/NanoLang producers in NanoVM and
instrumented generated C. The qualified-global fixture has four passing NanoISA
routes because the legacy C-native route still fails under #986. I retain its
separate actual interpreter report, failed native compilation and unchanged
prior output; I do not count it as a fifth passing native route.

After the test-only entry shadows, I rerun the original evaluator, two scope
methods, focused ownership unit and actual qualified shadow/publication control
on both hosts. The two-method scope runs take4.311s and5.459s. The unchanged
implementation's full decoder matrix is retained at `0c8a282fc`; the final
scope-fixture/source manifests are at `0cf0765e3`.

My runtime controls retain arrays, mutable scalar and STRING globals, caller
locals with the same spellings, named/function-valued calls, recursive calls,
early returns, returned callable values and a live local capture. The unchanged
minimal witness is loaded losslessly from the preceding instruction evidence.
I retain the original array_push API assertions, including intentionally
injected global and local VOID/integer values.

Both development NanoLang drivers are built from `src_nano/nanoc_v06.nano`
using the C seed, without changing shadow deadlines. They establish the stated
source-produced routes, not raw Stage1/Stage2 equality or installed/no-C-seed
qualification. Generated native C is instrumented; its existing AOT runtime
object is ordinary. Unit-test products are temporary and not archived.

## My retained failures

- My original production baseline plus the new unchanged witness fails actual
  evaluator execution at the caller's empty array. The original decoder failure
  and prior-output refusal remain in `nanolang-instructions-20261120`.
- My first candidate bypasses an existing array_push API path and fails the
  unchanged `result.type == VAL_VOID` assertion. I preserve that log, restore
  the existing dispatch path and pass the original assertions without edits.
- My qualified fixture passes actual C-interpreter shadows but the legacy
  AST-to-C backend emits an undeclared module alias and an invalid callable
  spelling. I track this under existing #986. Each host's final report records
  all seven shadows complete, native compilation nonzero and prior bytes intact.
  I do not weaken this native product requirement or close #986.
- My fresh full-provider Clang ASan/UBSan evaluator run fails LeakSanitizer:
  29347 bytes in578 allocations. A separately built unchanged `cf925d416`
  worktree also fails:28879 bytes in569 allocations. Their test sets and AST
  sizes differ; these totals do not measure an ownership regression or gain.
  The reports include effect declaration/environment metadata, callable/array
  allocations and the terminal-match fatal-child path. I file #1149 and keep
  #1140/#1116/#1053 obligations open. Neither run establishes full sanitizer
  acceptance or complete original-suite execution under instrumentation.
- My first focused ownership fixture omits `main` and fails checking. I retain
  that setup failure, add the required entry and pass the corrected strict
  focused ASan/UBSan unit. I do not suppress leak detection or change the
  original full-suite assertions to obtain that component result.

The unchanged sanitizer baseline has a retained commit, clean tracked state,
source hashes and executable hash before I remove its task-created worktree.
I retain complete build/diagnostic logs, not its executable. My early candidate's
failed full-suite executable was replaced by a later ordinary gate, so I do not
claim an archived image/hash or an exact final-source receipt for that early
run. The final focused sanitizer product and source hashes are retained in the
post-run manifest; its full provider graph was built in a separate object tree.

## My commands

```sh
make -j8 test-eval test-env-scoping test-typechecker test-module-metadata \
  test-eval-global-binding-ownership bin/nanoc_c nano_virt nano_vm nvm2c nvm2c-runtime
bin/nanoc_c src_nano/nanoc_v06.nano -o .test_output/global-bindings-source-compiler
NANOLANG_TRANSLATOR_SOURCE_COMPILER="$PWD/.test_output/global-bindings-source-compiler" \
  python3 -m unittest -v tests.test_eval_global_bindings \
    tests.test_nanolang_translator_container tests.test_nanolang_translator_payloads \
    tests.test_nanolang_translator_module tests.test_nanolang_translator_instructions
NANO_IMPORTED_GLOBAL_COMPILER="$PWD/bin/nano_virt" \
  python3 -m unittest -v tests.test_imported_globals
NANO_IMPORTED_GLOBAL_COMPILER="$PWD/.test_output/global-bindings-source-compiler" \
  python3 -m unittest -v tests.test_imported_globals
/usr/bin/python3 -m unittest tests.test_nanoisa_schema
make test-embedded-nano-identifiers
```

For the fresh Linux sanitizer builds I select
`CC='clang --gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13'`,
`CFLAGS='-Wall -Wextra -Werror -std=c99 -g -O1 -fPIC -Isrc -D_GNU_SOURCE -fsanitize=address,undefined -fno-omit-frame-pointer'`
and `LDFLAGS='-lm -lcrypto -lffi -rdynamic -fsanitize=address,undefined'`.
The candidate uses `OBJ_DIR=.test_output/obj-global-bindings-asan` and
`COMPILER_INPUT_ARCHIVE=.test_output/global-bindings-inputs-asan.a`.
The unchanged detached baseline uses its own default object/archive tree.
The full target is `test-eval`; the focused target is
`test-eval-global-binding-ownership`. Darwin uses the same focused flags with
its native compiler and `LDFLAGS='-lm -fsanitize=address,undefined'`.
Its PATH begins `/opt/homebrew/bin:/usr/local/bin`.

`global-bindings-qualified-report.py` extracts the actual qualified fixture,
executes the C seed with its machine-readable shadow report and retains the
known legacy publication refusal. `global-bindings-manifest.py` records post-run
source/tool/platform observations, not a complete before/after SDK closure.
`retained-inputs.json` binds each compressed log, source and report to its
uncompressed size/SHA256 and compressed SHA256. The final manifests have matching
source hashes on both hosts. I retain initial and final Darwin manifests so the
implementation and test-only pins stay distinct.

Canonical integration and broader #1147 acceptance remain open, along with the
legacy native #986 route, full memory qualification, #1148, all extension and
instruction/host admission, pure NanoLang C emission, installed/bootstrap
cutover and the complete #976 release contract.
