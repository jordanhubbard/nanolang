# My complete NanoLang passive-claim validator

I implement this #1142 prerequisite in `eb70aa3f6` on
`feat/5.1-nanolang-translator`, draft PR #1144. My implementation is
`src_nano/nvm2c/passive.nano`; my production C validator is unchanged.
This completes the `nvm_passive_valid` boundary under its bounded, narrow
function/parameter representation. It does not complete v2 module conversion
or executable admission.

I validate both record versions, exact payload consumption, sorted dependency
and input lists, guarded immutable scalar inputs, distinct result locals and
the stable lowest-source-index topological order. Physical node ranges must
partition each block exactly. Version 2 checks forward-only node CFGs,
instruction boundaries, stack joins and the existing 65536-instruction node
limit. Version 1 keeps its straight-line behavior and its lack of that node
instruction limit. I also scan the entire owner function for interval-entry,
opcode and external-input-write rules.

I collect concrete callee roots in block, selected-node and instruction order.
`closed_sequence` checks them with one shared private cache before I return
success. Resetting that cache for each node would change the existing analysis result under the depth limit. I retain a 65-callee chain accepted after its suffix
has been checked, and the same chain refused when checked first. I preserve the
separate depth-before-cache control. These are limits on active analysis, not
a runtime stack-depth guarantee or a termination theorem.

Deferring the ordered callee batch until the other claim checks pass changes
no boolean result. I return no intermediate successful claim and expose no
caller-writable checked-state cache. Ordinary direct calls outside node ranges
are not summarized as pure by this validator.

## My paired qualification

| Check | Linux ARM64 | Darwin ARM64 |
| --- | --- | --- |
| Full passive-record differential corpus, five routes | 2183 cases match | 2183 cases match |
| Explicit record expectations against actual C validation | 414 pass | 414 pass |
| Unchanged producer products | Four par/flow products pass | Four par/flow products pass |
| Earlier closed-call regression, five routes | 1498 cases and 512 opcode/profile decisions match | 1498 cases and 512 opcode/profile decisions match |
| Actual C passive validator and decoder, ASan/UBSan | All 2183 cases pass | All 2183 cases pass |
| Original input/call/internal-CFG/directive suites | All 25 methods pass | All 25 methods pass |
| Embedded-identifier gate | Five methods pass | Five methods pass |

The final three-method differential/product/call run takes 244.370s on Linux
and 122.984s on Darwin. These observations are not a performance comparison.
My first Linux record-only run passes the same 2183 cases in 73.921s before I add
the original-product method; the implementation does not change between runs.

The five validator routes are C-seed native plus both C/NanoLang producers in
NanoVM and instrumented generated C. My fixtures execute the actual NanoLang
validator; no foreign analysis helper decides their results. Every added or
changed named NanoLang function has a useful shadow, run during fixture builds.

The corpus includes tag values 0 through 255; malformed/truncated/trailing
records; effect/resource claims; missing, duplicated, reversed and unused
reads/dependencies; forward source references and cycles; exact topological
partitions; duplicate result locals; guarded input ordering and immutable
parameters; node branch targets, unreachable gaps, joins and stores; whole-owner
interval restrictions; both versions' opcode behavior; direct-call purity and
shared-cache order. I retain exact/over-limit node cases under both versions:
65536 instructions pass both, while 65537 pass version 1 and fail version 2.
The earlier callee suite re-exercises the actual depth/instruction/storage/
worklist limits after the shared-batch entry is added.

I compile the unchanged `passive_par.nano` and `passive_flow.nano` fixtures with
both the C and development NanoLang producer. All four actual modules pass the
existing whole-module decoder, ordinary VM verification, VM execution and
instrumented generated-C execution. Expected main outputs remain `13` and `15`.
I extract the actual passive bytes, code offsets, function shapes and parameter
tags without renumbering them, then check each extracted claim through all five
validator routes. These are source-produced products, not synthetic claims
described as compiler output.

## My verified boundary

`tests/nvm2c_passive_reference.c` calls production `nvm_passive_valid` directly.
Its test envelope supplies explicit code, claim and bounded function/parameter
descriptors. It is not a second NanoISA wire format or a C reimplementation of
the analysis. Its narrow-field and parameter-length checks precede both routes.
The independent sanitizer build instruments the wrapper, production passive
analysis and actual ISA decoder.

I preserve explicit controls where passive validation succeeds but ordinary
module verification remains necessary: a branch targeting outside the module,
an invalid ordinary call outside a node, and an owner ending at its passive
block without a return. An absent claim also does not verify arbitrary code.
I do not reinterpret those boundary decisions as authority to run the module.
`decode_core` still reports section 12 pending. Its whole-module conversion,
signature/reference/ownership and remaining profile checks must be implemented
before that pending result can become a complete module decision.

## My commands and retained evidence

```sh
make -j8 obj/nvm2c_passive_reference
NANOLANG_TRANSLATOR_SOURCE_COMPILER="$PWD/.test_output/legacy-globals-source-final" \
  python3 -m unittest -v tests.test_nanolang_translator_passive \
    tests.test_nanolang_translator_passive_calls
python3 -m unittest -v tests.test_passive_inputs tests.test_passive_calls \
  tests.test_passive_internal_cfg tests.test_passive_directives
cc -std=c99 -Wall -Wextra -Werror -O1 -g \
  -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer \
  -Isrc/nanoisa tests/nvm2c_passive_reference.c src/nanoisa/passive.c \
  src/nanoisa/isa.c -o .test_output/passive-record-reference-asan
PYTHONPATH="$PWD" python3 .test_output/passive-record-collect.py --corpus
make test-embedded-nano-identifiers
```

`make test-nanolang-translator-passive` declares the required tools for the
ordinary three-route gate; the source-compiler environment variable enables
both additional NanoLang-producer routes. Darwin prepends
`/opt/homebrew/bin:/usr/local/bin` to PATH. Both hosts reuse the full
C-seed-built NanoLang driver from 602dc495e; compiler frontends are unchanged
since that pin. The new validator is imported and compiled from its new source
by each producer. This is not installed/no-C-seed or raw fixed-point evidence.

Generated-C routes instrument generated code and link an ordinary AOT runtime.
Linux retains default leak detection; I do not claim Darwin LeakSanitizer
coverage, allocation-failure simulation, a whole-process memory cap, complete
leak freedom or service isolation. The original full evaluator sanitizer
failure under #1149 remains open.

Post-run manifests at eb70aa3f6 have identical source hashes. They record tool
hashes and explicit SDK observations, not a complete before/after SDK closure.
Unittest products, including the four producer modules and their native images,
are temporary and not archived. I retain their source fixtures and actual run
logs. The independent corpus collector deterministically regenerates all 2183
synthetic inputs, runs the actual C sanitizer oracle and records every input
byte, explicit expectation and C decision. Those complete records match across
hosts. `retained-inputs.json` binds each compressed source/log/manifest/corpus
to its uncompressed length and SHA256 and compressed SHA256.

I keep whole-module conversion, ownership/service/retained-layout profiles,
ordinary instruction/host admission, representation analysis, NanoLang C
emission, product and bootstrap cutover and every remaining #976 release
requirement open. Product selection is unchanged.
