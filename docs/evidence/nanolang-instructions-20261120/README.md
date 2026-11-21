# My primary instruction decoder checkpoint

I qualify component `d97b1ea89495ea6f680dafcef88dedc1d10dd69a` on Linux ARM64
and Darwin ARM64. I implement instruction decoding in NanoLang under #1142;
I do not change product selection or claim completed instruction verification.

My combined five-method gate passes on each host:

| Component | Cases per route |
| --- | ---: |
| Container | 165 |
| Core payload tables | 1585 |
| Core module relationships | 264 |
| Primary instruction decode/re-encode | 3472 |
| Combined synthetic cases | 5486 |

I also retain six unchanged example products,256 primary widths and256 File
scans on each route, plus the qualified forwarding and wrong-record prior-output
controls. Five module extension profiles remain explicitly pending. My combined
Linux run takes104.845s; my Darwin run takes52.403s. These are observed suite
wall times, not a performance comparison.

I execute each component through C-seed native code and through both C/NanoLang
producers in NanoVM and instrumented generated C: five routes per host. My
instruction oracle calls the actual `isa_decode`, `isa_get_info` and `isa_encode`
APIs. I compare decoded lengths, opcodes, stack effects, operand kinds, unsigned
halves, signed integer values and reconstructed bytes. I probe every primary
byte value and test every truncation of valid instructions, exact/offset windows,
all-zero/all-one/high-bit/alternating/random operand bytes. I retain721 explicit
expected decisions, including68 manually constructed integer-extreme, F64-bit,
compound-operand and invalid-window controls. The remaining explicit refusals
use lengths from actual C-decoder probes. My F64 cases include both zero signs,
infinities, subnormal boundaries and quiet/signaling NaN payloads.

I pass33 schema tests and the five-method embedded-identifier gate on each host.
The latter lexes3277 static fragments with zero reserved names;135 lexer-refused
fragments retain their existing executable-test obligations. My generated C
schema header is byte-identical to the previous checkpoint. Each added/changed
non-extern NanoLang function has a useful shadow; my normal producers run those
shadows without changing their deadlines.

## My commands and inputs

From the repository root, with the existing C reference tools built:

```sh
make obj/nvm2c_instruction_reference
NANOLANG_TRANSLATOR_SOURCE_COMPILER="$PWD/.test_output/nvm2c-module-source-compiler-final" \
  python3 -m unittest -v tests.test_nanolang_translator_container \
    tests.test_nanolang_translator_payloads tests.test_nanolang_translator_module \
    tests.test_nanolang_translator_instructions
/usr/bin/python3 scripts/gen_nanoisa_schema.py --check
/usr/bin/python3 -m unittest tests.test_nanoisa_schema
make test-embedded-nano-identifiers
```

On Darwin I build with `make -j8 obj/nvm2c_instruction_reference`, use
`PATH=/opt/homebrew/bin:/usr/local/bin:$PATH`, and substitute the existing
`.test_output/nvm2c-module-darwin-source` driver. I reuse the C-seed-built
NanoLang drivers from my preceding component gate; their hashes appear in the
manifests. I do not claim a fresh raw Stage1/Stage2 bootstrap. The targeted
`make test-nanolang-translator-instructions` target builds its reference tools
and runs the instruction gate; setting the source-compiler variable adds the
two NanoLang-produced routes.

My manifests retain source/tool hashes after execution, compiler/Python/OS
observations and cross-host source equality. I do not capture a complete
before/after SDK closure. Unit-test products live in temporary directories and
are not archived. I instrument generated C with ASan/UBSan while using the
existing ordinary AOT runtime object. My retained failing modules below are
regression evidence, not release product receipts.

## My retained refusals

I preserve the failures before correction and distinguish their causes:

- My first command tries to compile the decoder library alone; it has no `main`.
  I then build the executable driver. Its first spelling uses reserved `byte`
  as a local; I correct that fixture name to `octet`, preserving the lexer rules.
- My first shadow execution fails with array index64 into an empty array. A
  minimal unchanged two-function witness shows an unrelated caller local hiding
  a callee global. I file [#1147](https://github.com/jordanhubbard/nanolang/issues/1147)
  and namespace my generated globals. The witness still fails, preserving prior
  product bytes. This naming correction does not repair interpreter scope.
- My first cross-route gate refuses C translation at ARR_PUSH, function2,
  offset375: conflicting int/numeric aggregate shapes. I retain the unchanged
  module and a minimal accepted INT/U8 join whose module refuses at function0,
  offset49. I file [#1148](https://github.com/jordanhubbard/nanolang/issues/1148)
  and explicitly convert decoder byte reads to integers. The unchanged minimal
  join still refuses native translation and preserves its prior product. I do
  not weaken shape constraints or close that admitted-source parity defect.
- My Darwin default Python lacks PyYAML, so its first schema check never runs.
  `/usr/bin/python3` already has PyYAML and passes the unchanged schema checks.
- My first Darwin manifest collector aborts when default `xcrun --show-sdk-version`
  names an unavailable CommandLineTools SDK. I change only the collector to
  retain each query's status/stdout/stderr. Explicit `--sdk macosx` reports27.0.
  I preserve the failed default query and the real build command's SDK include
  paths; I do not rerun completed gates or infer a complete SDK identity.

`retained-inputs.json` binds every compressed input/log/module to its original
size and SHA256 and to the compressed file's SHA256. The minimal witness sources
and actual failed modules are retained losslessly. My original direct shell
refusals did not separately capture exit statuses; the later prior-product
controls record actual nonzero statuses and unchanged sentinel bytes. My full
failure logs remain separate from successful component results.

Passive/closed-call CFG checks, retained layouts, ownership/service profiles,
complete instruction and host admission, representation analysis, C emission
in NanoLang, integration, installed products, bootstrap equality and every
remaining #976 release requirement remain open.
