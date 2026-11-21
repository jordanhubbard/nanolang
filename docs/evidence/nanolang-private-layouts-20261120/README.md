# My private retained-layout structure profiles

I implement this #1142 prerequisite in `a139ebd10` on
`feat/5.1-nanolang-translator`, draft PR #1144. My new
`src_nano/nvm2c/private_layouts.nano` checks the complete private transport
preflight before using the shared payload parser. I preserve the exact
256-layout, 65536-field and 16777216-byte bounds, complete input consumption,
and error precedence before semantic row/field checks.

I extend the shared forward-graph checker with the three existing C profiles.
ARRAY permits scalar/string/array leaves in record DAGs. Mixed additionally
permits scalar union nodes beside those record DAGs; record edges must still
connect records, and arrays remain record fields. MAP additionally permits map
leaves in records. I preserve the iterative cycle check and disconnected-node
rules. Prior-only tables retain their broader transport grammar. My public
decoder selects its original profile and retains permitted table tails.

These NanoLang outputs are immutable byte views. Callers keep the input bytes
immutable; I do not claim the C API's independent numeric allocation ownership.
Successful structure decoding grants no element type, ordinary/resource
classification, reference eligibility or executable authority. Private
declaration projections, whole-module conversion and ordinary instruction
verification remain required. All four pending whole-module profiles remain
pending after this prerequisite.

## My differential corpus

| Check | Linux ARM64 | Darwin ARM64 |
| --- | --- | --- |
| Public/private exact-byte differential gate | 1748 cases × four profiles × five routes pass | 1748 cases × four profiles × five routes pass |
| Explicit status expectations | 4640 pass | 4640 pass |
| Complete C layout/cursor provider ASan/UBSan oracle | All four profile reports match ordinary C | All four profile reports match ordinary C |
| Original public payload codecs | 1585 cases per route pass | 1585 cases per route pass |
| Core/passive module and shared-ownership regressions | Five methods pass | Five methods pass |
| Embedded identifiers | Five methods pass | Five methods pass |

I compare 1748 cases per profile across public, ARRAY, mixed and MAP decoding:
6992 complete status/count/length/re-encoded-byte reports per execution route.
1160 cases carry explicit expectations for all four profiles, giving 4640
explicit C status controls. Every successful C encoding must equal its actual
input prefix; private success must consume the entire input. Every NanoLang
report must match the complete C report, without checksums or sampling.

I retain all 256 layout kinds and all 256 tags with prior and forward controls,
all prior kind/tag combinations, scalar/array/map leaves, disconnected unions,
exact nominal edges, shared diamonds, cycles, 256-node forward/prior chains,
raw unsigned name indices, reserved fields and rich truncations/mutations.
The actual 255/256/257 layout, 65535/65536/65537 field and 16 MiB/16 MiB+1 byte
controls retain both accepted boundaries and exact refusals. Private transport
preflight still precedes bad kinds, reserved fields and graph failures, while
the public minimum-header and field-array prechecks keep their original order.

My C oracle calls the actual public/private APIs and re-encoder. It seeds each
private output with a sentinel and requires every refusal to preserve it. It
also compares both mixed entry points and checks that the detailed APIs reset
their allocation-refusal sink in these ordinary runs. I do not simulate OOM or
infer precise allocation failures from a legacy TRUNCATED result.

My test driver uses balanced hexadecimal joins for the complete large field
tables, preserving exact bytes without quadratic whole-report concatenation.
The product decoder remains iterative. Added and changed NanoLang functions
have useful shadows exercised in the fixture builds.

My original regressions retain 1585 public codec cases, 267 complete core module
cases with six unchanged example products/256 widths/256 File scans, 2240
passive-bearing module decisions and exact claim bytes with four unchanged
producer products, and 2350 shared ownership status/authority decisions per
route. Both hosts' independently regenerated corpus bytes/reports and tested
source hashes match exactly. The embedded gate retains 3287 static fragments,
zero reserved declarations and 135 existing lexer-refused fragments; executable
checks remain responsible for their parsing/runtime behavior.

## My retained corrections

My first driver has a malformed output expression and fails parsing before
publication. I retain its source/log, separate the output fields and later use
balanced exact-byte reports. I change no language rule.

The first full run has two wrong explicit C expectations: a two-row header
claim with only one empty row fails the public minimum-header check before its
bad kind or reserved byte. All five actual routes match the C reports in that
run, but the two explicit controls fail, so the run is not green. I retain the
complete FAILED result and first test source. I correct those expectations to
TRUNCATED and add separate full-first-row controls demonstrating the later
private-preflight/public-semantic ordering. The final corpus has two additional
cases; I retain both behaviors rather than dropping the original controls.

## My commands

```sh
make -j8 obj/nvm2c_private_layout_reference
NANOLANG_TRANSLATOR_SOURCE_COMPILER="$PWD/.test_output/ownership-source-final" \
  python3 -m unittest -v tests.test_nanolang_translator_private_layouts
NANOLANG_TRANSLATOR_SOURCE_COMPILER="$PWD/.test_output/ownership-source-final" \
  python3 -m unittest -v tests.test_nanolang_translator_payloads
NANOLANG_TRANSLATOR_SOURCE_COMPILER="$PWD/.test_output/ownership-source-final" \
  python3 -m unittest -v tests.test_nanolang_translator_module \
    tests.test_nanolang_translator_passive_module \
    tests.test_nanolang_translator_ownership
cc -std=c99 -g -O1 -fsanitize=address,undefined -fno-sanitize-recover=all \
  tests/nvm2c_private_layout_reference.c src/nanoisa/nvm_v2_layouts.c \
  src/nanoisa/nvm_v2_cursor.c -o .test_output/private-layout-reference-asan
PYTHONPATH="$PWD" python3 .test_output/private-layout-collect.py --asan
PYTHONPATH="$PWD" python3 .test_output/private-layout-collect.py --corpus
make test-embedded-nano-identifiers
```

Darwin prepends `/opt/homebrew/bin:/usr/local/bin` to PATH and combines the
public payload and dependent module/ownership regressions into one six-method
run. Linux retains separate one-method and five-method runs. I reuse each
host's C-seed-built full NanoLang compiler at `7cd173b30`; compiler frontends
are unchanged, and each fixture build imports the new decoder implementation.
That reuse does not establish raw Stage1/Stage2 equality or installed cutover.

## My evidence boundaries

The five routes are C-seed native plus C/NanoLang producers in NanoVM and
instrumented generated C. Generated C links an ordinary AOT runtime. My
separate C oracle sanitizer image instruments the complete layout/cursor
providers and wrapper. Linux retains default leak detection; I claim no Darwin
LeakSanitizer coverage or complete leak freedom.

I independently regenerate all synthetic input bytes and retain their explicit
expectations and complete C reports. Source/tool manifests are post-run
observations, not a complete SDK closure. Temporary unittest products are not
archived. `retained-inputs.json` binds each compressed input to its raw
length/SHA256 and compressed SHA256. I preserve failed setup/expectation runs
beside final results.

I change no product selection and close no parent issue. Full private
declarations, service/profile conversion, representation analysis, C emission
in NanoLang, canonical integration, installed/raw bootstrap and the complete
original #976 release scope remain open.
