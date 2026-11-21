# My complete core/passive module decisions

I implement this #1142 boundary in `f9373961b` on
`feat/5.1-nanolang-translator`, draft PR #1144. `decode_core` now handles
section 12 through the existing whole-module deserialization contract when no
unimplemented ownership/service/private-layout profile is selected. I retain
those four profiles as pending; I do not remove them from release scope.

I retain the exact passive payload in `ModuleView.passive`. I preserve section
framing, core payload/reference checks, feature agreement, conversion and
passive validation in that order. Empty passive sections fail framing rather
than becoming absent claims. Missing feature bits fail before conversion.

The current C bridge refuses non-string constants and duplicate length/byte
string identities, including embedded zeros. I implement those decisions in
NanoLang before narrowing code/function/debug fields and function result
counts. I reconstruct parameter views and evaluate canonical retained-layout
size, names, kinds and DAG requirements from already parsed rows. I do not
materialize an old C module or confuse ignored table tails with canonical
retained bytes. Existing callback checks cover callback construction's
nonallocation refusals. Ordinary import construction retains its distinct
result-count behavior.

With no ownership/service payload selected, the C shared ownership declaration
validator returns success for that absent contract. I then run the complete
NanoLang passive validator. I apply this conversion path only to nonempty
passive payloads; extension-free core modules retain the existing decoder's
nonconversion behavior. No C analysis helper makes these NanoLang decisions.

## My paired checks

| Check | Linux ARM64 | Darwin ARM64 |
| --- | --- | --- |
| Complete passive-bearing module decisions and exact claim bytes | 2240 cases match on five routes | 2240 cases match on five routes |
| Explicit module status expectations against actual C deserialization | 472 pass | 472 pass |
| Unchanged par/flow products from both producers | Four modules pass | Four modules pass |
| Updated core decoder and global-scope gate | Seven methods pass | Seven methods pass |
| Prior complete passive and call-analysis gates | Three methods pass | Three methods pass |
| Schema gate | 33 methods pass with system Python | 33 methods pass with system Python |
| Embedded-identifier gate | Five methods pass | Five methods pass |

The updated core gate retains 5489 cases per route: 165 container, 1585 payload,
267 module and 3472 primary instruction cases. It includes six unchanged example
products, 256 widths, 256 File scans and both scope fixtures through five routes.
The module corpus gains three actual passive framing/feature decisions and
retains four explicit pending profiles. Historical five-profile evidence remains
at its original pins.

The prior passive gate reruns all 2183 claim cases and the 1498 callee cases,
plus 512 versioned opcode decisions per route and four unchanged producer
products through verified VM and instrumented native execution. This includes
the actual node/call/instruction/storage/worklist limits and shared-cache order.
My added/changed NanoLang functions have useful shadows, exercised in builds.

The five routes are C-seed native and both C/NanoLang producers through NanoVM
and instrumented generated C. The new module gate compiles unchanged par/flow
sources with both producers, runs ordinary verification and actual VM execution
with original outputs 13/15, then compares their module decisions and every
retained claim byte through all five routes. The prior passive gate additionally
executes their instrumented generated-C products.

I retain paired controls showing conversion is conditional: extension-free core
modules may transport duplicates, non-string constants, wide debug offsets and
wide result counts, while the passive conversion path preserves its additional
refusals. I keep embedded-zero/binary/prefix string identities, exact result/debug
width boundaries, import/function differences, repeated source metadata keys,
retained layout tails and section/reference/feature/conversion error precedence.
Wrapping the complete claim corpus also checks actual wire signature errors
before passive validation, rather than copying standalone boolean expectations.

## My commands and setup correction

```sh
make -j8 obj/nvm2c_module_reference
NANOLANG_TRANSLATOR_SOURCE_COMPILER="$PWD/.test_output/legacy-globals-source-final" \
  python3 -m unittest -v tests.test_nanolang_translator_passive_module
NANOLANG_TRANSLATOR_SOURCE_COMPILER="$PWD/.test_output/legacy-globals-source-final" \
  python3 -m unittest -v tests.test_nanolang_translator_container \
    tests.test_nanolang_translator_payloads tests.test_nanolang_translator_module \
    tests.test_nanolang_translator_instructions tests.test_eval_global_bindings
NANOLANG_TRANSLATOR_SOURCE_COMPILER="$PWD/.test_output/legacy-globals-source-final" \
  python3 -m unittest -v tests.test_nanolang_translator_passive \
    tests.test_nanolang_translator_passive_calls
/usr/bin/python3 -m unittest -v tests.test_nanoisa_schema
make test-embedded-nano-identifiers
PYTHONPATH="$PWD" python3 .test_output/passive-module-collect.py --corpus
```

Darwin prepends `/opt/homebrew/bin:/usr/local/bin` to PATH and combines the new
module and updated core/scope suites into one eight-method run. Linux retains
separate logs for those groups; both retain a separate three-method prior gate.
Their observed runtimes are in the complete logs, not performance comparisons.
`make test-nanolang-translator-passive-module` supplies the required build tools
for the ordinary three-route test; the environment variable adds both source-
producer routes.

My first Linux schema command uses the default Linuxbrew Python 3.15, which lacks
PyYAML. It fails at import before discovering the 33 cases. I retain that log,
select the already installed system Python and pass all 33 without changing the
schema, installing packages or modifying compiler/translator source. The
embedded gate reports 3286 fragments, zero reserved names and 135 existing
lexer-refused fragments; executable gates remain responsible for their behavior.

## My evidence boundaries

The whole-module C oracle calls `nvm_v2_module_deserialize` directly. Its new
`--passive` mode reports the actual result and retained payload bytes; it does
not implement another conversion. Failed reports expose no partial claim bytes.
The NanoLang driver independently reports the same fields from its module view.
My whole-module C oracle is ordinary, not a full-provider sanitizer build.

I reuse the C-seed-built full NanoLang driver from 602dc495e; compiler frontends
are unchanged since that pin. Each producer imports and compiles the new module
implementation. Generated-C routes instrument generated code and link an
ordinary AOT runtime. Linux retains default leak detection; I do not claim
Darwin LeakSanitizer coverage, allocation-failure simulation, complete leak
freedom or a full SDK closure. The separate #1149 memory gate remains open.

Post-run manifests at f9373961b have identical source hashes. Independent corpus
collection regenerates all 2240 synthetic modules, runs the actual C oracle and
retains every input byte, explicit status and exact result/payload report. Those
complete records match across hosts. Temporary unittest images and actual
producer modules are not archived; I retain their unchanged source fixtures and
run logs. `retained-inputs.json` binds each compressed input to its uncompressed
length/SHA256 and compressed SHA256.

Successful module decoding still does not verify all instructions, stack/type
flows or host authority. Ordinary malformed branches/calls retain their separate
verification boundary. Ownership, service and private array/map profiles,
representation analysis, C emission in NanoLang, installed/raw bootstrap
cutover and the complete original #976 release scope remain open. I change no
product selection.
