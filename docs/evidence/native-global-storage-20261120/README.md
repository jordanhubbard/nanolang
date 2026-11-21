# My native global-storage checkpoint

I implement this checkpoint in `602dc495e` on
`feat/5.1-nanolang-translator`, above the existing callable-ownership stack.
I track public imported globals under #986 and the translator under #1142.
My earlier [interpreter binding evidence](../global-bindings-20261120/README.md)
remains a separate checkpoint.

I retain module-local storage and export typed accessors for public globals.
Checked declaration identity selects the accessor; aliases share one location.
I snapshot function-valued globals before evaluating their arguments. I retain
the checked aggregate and nominal types when generating reads, writes and calls.

I initialize direct imports before a module's own runtime initializers and guard
each initialization once. This includes private state used by public functions.
My shared parsed-AST classifier keeps C-backed modules with NanoLang globals in
the object graph. I use canonical source identities for storage, initializer
and serialized metadata symbols, without changing reported module names.
Native module objects use the existing canonical-directory SHA256 namespace
under `obj/nano_modules/`, preserving private temporary compilation and atomic
publication. Equal basenames in distinct directories no longer overwrite an
object or collide at link time.

My full NanoLang driver now rejects duplicate public metadata identities at its
merge boundary, as the C loader already does. I keep the inventory lossless and
retain its query-time controls. The new validator has its own useful shadow;
the merge shadow also checks the original ambiguous-identity negative source.

## My qualification

| Check | Linux ARM64 | Darwin ARM64 |
| --- | --- | --- |
| Original transpiler, checker, metadata, environment/scoping and evaluator gates | Pass | Pass |
| Imported globals through C producer, verified VM and instrumented generated C | 32 methods pass | 32 methods pass |
| Imported globals through freshly built NanoLang driver, verified VM and instrumented generated C | 32 methods pass | 32 methods pass |
| Imported globals through freshly built NanoLang driver, native output | 32 methods pass | 32 methods pass |
| C-seed native imported-global suite | 30 pass, 2 shadow refusals | 30 pass, 2 shadow refusals |
| C-seed native suite with ASan/UBSan product compilation | Same 30 pass, same 2 shadow refusals | Same 30 pass, same 2 shadow refusals |
| Original module compiler invocation controls | 24 methods pass | 24 methods pass |
| Original introspection suite, C and NanoLang producers | 8 methods per producer pass | 8 methods per producer pass |
| Combined decoder and global-binding gate | 7 methods pass | 7 methods pass |
| Schema and embedded-identifier gates | 33 and 5 methods pass | 33 and 5 methods pass |
| Existing terminal-backstop method and nested-object collector | Pass | Pass |

I preserve the original 29 imported-global methods and add three: private-state
initialization before root initialization, distinct storage for same-basename
modules, and C-backed extern/public-global storage. The basename method also
retains a negative case: without distinct declared public metadata names, the
existing ambiguity policy requires refusal and prior-output preservation.

The combined decoder gate takes115.616s on Linux and60.094s on Darwin; these
observations are not a performance comparison. It retains all5486 synthetic cases per route, six
unchanged example products,256 primary widths and256 File scans. Both scope
methods now pass all five routes, including the unchanged qualified fixture's
formerly refused C-seed native product. Five extension profiles remain pending;
decoding does not authorize execution.

The two C-seed failures are `test_captured_local_shadows_imported_alias` and
`test_imported_closure_global_capture`. Both stop in interpreter shadow execution
before publishing native output. I retain them without skips, changed assertions
or an all-green suite claim. They belong to the existing #1089 legacy capture
compatibility work; I do not add that optional AST-to-C extension to the One IR
release scope. All32 methods pass through the stated NanoISA and source-driver
routes. #986 still requires canonical integration and installed Stage1/Stage2,
raw fixed-point and complete acceptance evidence.

## My retained failures

I retain complete initial and corrected logs, including:

- The unchanged qualified native fixture's undeclared alias/call failure.
- My first imported-global run's aggregate type and nominal field failures,
  alongside the two existing capture refusals.
- A misplaced unittest method's discovery failure, then the corrected private
  state test observing0 instead of41 before ordered initialization is fixed.
- Same-basename module objects overwriting one pathname, then the distinct
  serialized metadata symbol collision exposed after object paths are corrected.
- My new basename fixture initially expecting success without distinct public
  identities. The C loader correctly refuses it; I correct that expectation.
- The NanoLang driver then unexpectedly publishing that negative source. I add
  merge-boundary validation and retain both refusal and positive controls.
- An unused unadmitted helper extern in my initial C-backed fixture. I retain
  the C translator's exact host-ABI refusal and use the admitted `get_argc` ABI
  in that storage fixture. I do not relax host admission.
- The terminal-backstop check stopping because `bin/nanoisa` was absent, and
  my incorrect `make nanoisa` target, before using `make bin/nanoisa`.

My metadata-unit insertion also stops on an ambiguous source anchor before
writing; I correct the anchor to the actual C entry. That mechanical edit has
no standalone retained diagnostic log. I do not reconstruct missing temporary
sources and describe them as exact historical artifacts.

## My commands and evidence limits

```sh
make -j8 test-transpiler test-typechecker test-module-metadata test-env-scoping \
  test-eval bin/nanoc_c nano_virt nano_vm nvm2c nvm2c-runtime
bin/nanoc_c src_nano/nanoc_v06.nano -o .test_output/legacy-globals-source-final
NANO_IMPORTED_GLOBAL_COMPILER="$PWD/bin/nano_virt" \
  python3 -m unittest -v tests.test_imported_globals
NANO_IMPORTED_GLOBAL_COMPILER="$PWD/.test_output/legacy-globals-source-final" \
  python3 -m unittest -v tests.test_imported_globals
NANO_IMPORTED_GLOBAL_ROUTE=native \
  NANO_IMPORTED_GLOBAL_COMPILER="$PWD/.test_output/legacy-globals-source-final" \
  python3 -m unittest -v tests.test_imported_globals
NANO_IMPORTED_GLOBAL_ROUTE=native NANO_IMPORTED_GLOBAL_COMPILER="$PWD/bin/nanoc_c" \
  python3 -m unittest -v tests.test_imported_globals
NANO_CC='cc -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer' \
  NANO_IMPORTED_GLOBAL_ROUTE=native NANO_IMPORTED_GLOBAL_COMPILER="$PWD/bin/nanoc_c" \
  python3 -m unittest -v tests.test_imported_globals
python3 -m unittest -v tests.test_module_compile_invocation
NANOLANG_TRANSLATOR_SOURCE_COMPILER="$PWD/.test_output/legacy-globals-source-final" \
  python3 -m unittest -v tests.test_nanolang_translator_container \
    tests.test_nanolang_translator_payloads tests.test_nanolang_translator_module \
    tests.test_nanolang_translator_instructions tests.test_eval_global_bindings
NANOLANG_INTROSPECTION_COMPILER="$PWD/.test_output/legacy-globals-source-final" \
  python3 -m unittest -v tests.test_nanoisa_introspection
make bin/nanoisa
python3 -m unittest -v \
  tests.test_canonical_match_guards.CanonicalMatchGuards.test_unchecked_generated_terminal_backstops
```

The introspection suite also runs with `NANOLANG_INTROSPECTION_COMPILER` set to
`bin/nano_virt`. Linux's initial native runs use the synonymous `legacy` route
selector. Darwin prepends `/opt/homebrew/bin:/usr/local/bin` to PATH and runs
the33 schema tests with `/usr/bin/python3`; both hosts run
`make test-embedded-nano-identifiers`.

I build the development driver with the C seed and use the existing C
translator in these reference routes. These checks do not establish raw
Stage1/Stage2 equality, installed/no-C-seed execution or NanoLang C emission.
Generated-C decoder/imported-global routes instrument the generated program
while using an ordinary AOT runtime object. The separate C-seed native sanitizer
route instruments its generated main/modules/runtime compilation, not the
compiler process or every external/system library. Linux retains default leak
detection; I do not claim Darwin LeakSanitizer coverage. The full original
interpreter sanitizer failure under #1149 remains open.

My source/tool manifests are post-run observations at the implementation pin,
not a complete before/after SDK closure. Unit-test products and collector
binary archives remain temporary; I retain the source tests, diagnostic logs and
final terminal-backstop command/terminal/hash receipts, without the binary files
those receipts describe. Both hosts' final manifests have identical source hashes.
`retained-inputs.json` binds each compressed input to its uncompressed length
and SHA256 and its compressed SHA256. Source snapshots describe the final
checkpoint, not every intermediate failed edit.

I keep #986/#1147 integration, #1148/#1149 and all original release obligations
open. My next #1142 work is extension/profile admission and then representation
analysis and C emission in NanoLang. Product selection remains unchanged.
