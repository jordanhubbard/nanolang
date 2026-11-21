# My shared ownership whole-module conversion evidence

I implement this checkpoint at `a471982ca` under #1142 and draft PR #1144.
I carry ownership section13 through `module.decode_core`, preserve its feature
relationship and construct the shared ownership model after core conversion.
My public MAP quarantine runs before table decoding. Selected private profiles
remain pending; invalid selected identities cannot fall back to shared checks.

## My conversion contract

I retain core table/reference/feature precedence, byte-exact string identity and
type checks, narrow code/function/debug fields, parameter and first-result tags,
canonical retained layouts and per-kind counts. I remove tolerated layout table
tails by re-encoding, and retain an empty layout table only when its feature is
present. Owner-route errors precede shared declaration validation. Ownership
validation precedes passive validation when both payloads are present.

This matches the shared C conversion boundary. It does not run ordinary
instruction verification. I deliberately retain a module with opcode255 that
both decoders accept structurally. Neither result authorizes its execution.

## My paired checks

I retain2531 synthetic whole modules, twenty explicit status expectations and
fourteen actual products from seven unchanged sources through both producers.
Across five routes on each ARM64 host, every completed status and exact
ownership/passive byte report matches the actual C deserializer. Eight inputs
remain pending on every route: six synthetic private-profile controls and two
original nested-map modules. I retain their C results rather than discarding
them or counting pending as successful admission.

The routes are direct C-seed native execution plus C/NanoLang producers through
NanoVM and ASan/UBSan-instrumented generated C. The linked AOT runtime and C
whole-module/routing reference providers are ordinary builds. I do not claim a
sanitized full C decoder, allocation-denial coverage for NanoLang, arbitrary C
pointer emulation, complete leak freedom or Darwin LSan coverage.

My corpus wraps the complete2350-model shared declaration corpus in real v2
containers. Its original declaration expectations are not reused blindly:
whole-module tables, references, features, route selection and conversion can
return earlier errors. Additional controls cover missing ownership/retained
features, empty/truncated sections in both directory orders, invalid routing
versions, exact debug narrowing, constant/refusal precedence, canonical layout
tails, ownership/passive co-profiles, every ownership truncation and128 fixed
mutations. A private pending result requires an explicit ARRAY/MAP layout scope
or actual C MAP selection, with a separate selected-kind5 control; names do not
select the allowance.

I compile and execute seven original sources through both producers, preserving
their assertions, stdout and main results. The two `map_nested_u8_borrowed`
products remain pending in the new decoder. The other twelve complete shared
conversion. All fourteen retain their original VM results, including
`first\nsecond\n`, `right\nsemantic-source-parity-pass\n` and the mixed
owner's main result7. I retain actual module/model bytes, source/product hashes,
VM results and complete C claims/routing reports. Their foreign-context carrier
prefixes are verified zero; a nonzero context requires a separate carrier.

| Check | Linux ARM64 | Darwin ARM64 |
| --- | --- | --- |
| Ownership module/status/exact bytes,2545 inputs | One method passes; eight pending | One method passes; eight pending |
| Core module and passive-module regressions | Three methods pass | Three methods pass |
| Embedded identifiers | Five methods pass | Five methods pass |

The core gate retains269 decisions, six original products,256 widths,256 File
scans and three explicit pending profile controls. Nominal argument refusals
preserve prior outputs through all three producers. The passive module gate
retains2240 decisions,472 explicit expectations and four original products.
Mandatory shadows run through fixture compilation. Identifier checks retain
3287 fragments, zero reserved declarations and135 existing lexer-refused
fragments, whose executable tests retain responsibility for their behavior.

My first Linux run passes2531 synthetic decisions with six pending controls.
I then replace its name-based pending allowance with an explicit scope plus
actual C routing result and add original producer modules. I retain the first
source/log beside the stricter final gate. There is no failed product check in
this checkpoint, and I do not invent one. My optional completion observer first
refuses an unquoted timeout clause, then times out with a line-anchored pattern.
The quoted clause and unanchored completion marker return satisfied. I retain
these observer outcomes separately from actual test-process exit statuses.

## My reproduction and retained inputs

```sh
make obj/nvm2c_module_reference obj/nvm2c_declaration_reference
NANOLANG_TRANSLATOR_SOURCE_COMPILER="$PWD/.test_output/ownership-source-final" \
  python3 -m unittest -v tests.test_nanolang_translator_ownership_module \
    tests.test_nanolang_translator_module tests.test_nanolang_translator_passive_module
make test-embedded-nano-identifiers
```

Linux runs the new gate separately from the three regression methods; Darwin
runs the combined four methods. Darwin prepends
`/opt/homebrew/bin:/usr/local/bin` to PATH. I reuse the genuine C-seed-built
NanoLang development compiler from `7cd173b30`; each image hash matches its
previous paired ownership receipt. These images are not installed stages or a
raw bootstrap fixed point. Every fixture imports the current module decoder.

I retain independently regenerated synthetic bytes/reports from both hosts,
source/tool/environment manifests, actual producer artifacts, build/check logs,
collector source and terminal results. Independent synthetic bytes/reports and
437 source hashes match across hosts. `retained-inputs.json` binds every input
to compressed and raw SHA256/size. Temporary decoder-driver executables are not
archived. I compare immutable normal narrow carriers and exact retained claims;
C allocation failure is outside that model. The canonical-layout tail controls
check conversion decisions; the byte report covers ownership and passive
payloads, not an arbitrary C object graph.

Service14, complete private flow admission, ordinary executable verification,
representation analysis, C emission in NanoLang, canonical integration,
installed/raw bootstrap and every remaining original #976 requirement stay
open. I change no executable product selection and close no parent issue.
