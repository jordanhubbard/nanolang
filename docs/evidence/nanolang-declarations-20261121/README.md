# My private declaration projections

I implement this #1142 dependency in `b50039864` on
`feat/5.1-nanolang-translator`, draft PR #1144. My NanoLang ARRAY, mixed and
MAP projection model preserves their different described, unknown, absent,
invalid and limit outcomes. I retain declaration order and failure precedence;
an unknown field does not hide a later malformed declaration.

`array_declarations.nano` validates exact element/binding extents, 4096 types,
65536 bindings, iterative coloring with cached depth64 bounds, recognized
primitive nesting and complete array-field coverage. `map_declarations.nano`
preserves the separate framing classifier, canonical scalar/pair inventories,
field coverage and map work charges. `declarations.nano` preserves retained
names/kinds/counts, signature/local descriptors, conservative descriptor/union
precharges, union partitions, foreign-profile refusals and complete publication.
My `variant` and `map_field` queries expose checked facts only after success.

I share private layout/descriptor and optional MAP-extension framing helpers
with the existing ownership reader. Its public entry points retain their
original profile and MAP refusal. My added suffix helper also preserves sticky
cursor failures. All added and changed NanoLang functions have useful shadows.

## My paired checks

| Check | Linux ARM64 | Darwin ARM64 |
| --- | --- | --- |
| Complete declaration/query differential reports | 3988 models × three profiles × five routes pass | 3988 models × three profiles × five routes pass |
| Explicit C status expectations | 1728 pass | 1728 pass |
| Instrumented C declaration providers | 3974 synthetic models per profile match ordinary C | 3974 synthetic models per profile match ordinary C |
| Unchanged source products and original VM results | Seven sources × two producers pass | Seven sources × two producers pass |
| Original ordinary/mixed/map C suites | All three targets pass | All three targets pass |
| Public shared ownership regression | 2350 decisions per route and slice refusals pass | 2350 decisions per route and slice refusals pass |
| Embedded identifiers | Five methods pass | Five methods pass |

The five routes are C-seed native and C/NanoLang producers in NanoVM and
instrumented generated C. The final differential corpus comprises 3974
synthetic models and fourteen actual producer models. I compare every returned
layout/field, element, binding, union variant, map scalar/pair/binding and map
field query. Balanced numeric joins retain the complete maximum-size reports;
I use no checksum or sampled field comparison.

I retain the earlier 2350 shared-declaration inputs through each private
classifier, every element/map scalar tag, forward/prior cached type chains,
cycles, recognized versus unknown nesting, foreign contracts, raw corruption
and the actual 4096-type/65536-binding/depth64 boundaries. Both array and map
65536-binding positive tables are materialized and queried. Descriptor work
approaches1048576; separate controls accept exactly1048576 and refuse1048577.
The map payload controls reach exactly4194304 charged work before semantic
failure, then cross it. Input ownership bytes exercise16MiB/16MiB+1. Ambiguous
borrow roots retain ordinary UNKNOWN while mixed readers continue to detect a
later malformed suffix. Map parameters/results retain their distinct refusals;
ordinary map locals are not rejected by that parameter/result rule.

My C oracle calls the actual constructors/getters. It checks failure-atomic
sentinel outputs, both public and split map constructors, and invalid query
outputs. The separate sanitizer image instruments the owning declaration TU,
retained-layout/reference helpers, cursor, layout decoder and ISA providers.
Other linked providers and producer-model extraction are ordinary. The original
C suites separately retain allocator denial/retry controls: eight ordinary,
thirteen mixed and twelve map allocation positions. Those C controls do not
establish NanoLang allocation-denial behavior.

## My actual producer products

I compile these unchanged sources with both C and NanoLang producers:

- `array_record_fields.nano`
- `empty_array_append.nano`
- `affine_scalar_union_instances.nano`
- `string_int_maps.nano`
- `string_string_maps.nano`
- `map_nested_u8_borrowed.nano`
- `owned_mixed/basic.nano`

I run each original NanoISA product in NanoVM. The empty-array fixture retains
`first`/`second` output order; the affine fixture retains its two output lines;
the original mixed owner returns7. I require those exact results, rather than
treating every nonzero main result as a compiler failure. All original source
assertions and selected shadows remain intact.

The C test extractor materializes each actual producer module through the real
v2 decoder/bridge, then records its raw declaration inputs, strings, narrow
function carriers and code. It supplies no query facts or execution plan to the
NanoLang projection. All fourteen models then pass the same three-profile,
five-route comparisons. This is not pure whole-module loader qualification.
I retain the actual modules, extracted model bytes, hashes and original VM
results independently on each host; host metadata can differ.

## My retained corrections

My first C oracle build fails strict `-Werror=misleading-indentation`. I retain
its source/log and separate its statements without relaxing compiler flags.
The first full run matches C reports on all actual routes but fails two explicit
expectations: a COMPLETE map field without a map contract is INVALID before
the ordinary/mixed reader can return UNKNOWN. I retain the FAILED run and first
test source, correct those expectations and preserve their original inputs.

My first producer probe also expects empty stdout from the unchanged
empty-array fixture. Its actual source prints `first` and `second`. I retain
that failure and require those exact lines in their original order. I alter no
fixture or product assertion. I add distinct ambiguous-borrow and map-signature
controls before the final run; the initial3969-model corpus and final3974-model
synthetic corpus remain distinguishable.

## My commands and limits

```sh
make -j8 obj/nvm2c_declaration_reference obj/nvm2c_declaration_reference_asan
NANOLANG_TRANSLATOR_SOURCE_COMPILER="$PWD/.test_output/ownership-source-final" \
  python3 -m unittest -v tests.test_nanolang_translator_declarations
NANOLANG_TRANSLATOR_SOURCE_COMPILER="$PWD/.test_output/ownership-source-final" \
  python3 -m unittest -v tests.test_nanolang_translator_ownership
make -j8 test-ordinary-array-authority test-ownership-declaration-projection \
  test-map-declarations
make test-embedded-nano-identifiers
PYTHONPATH="$PWD" python3 .test_output/declarations-collect.py --corpus
PYTHONPATH="$PWD" python3 .test_output/declarations-collect.py --products \
  .test_output/declarations-linux-final.log
```

Darwin prepends `/opt/homebrew/bin:/usr/local/bin` to PATH and combines both
Python suites into one four-method run. I reuse each host's qualified
C-seed-built NanoLang compiler at7cd173b30; both image hashes match the prior
paired ownership receipt. Each producer imports the new projection code.
These are genuine development compiler paths, not installed-stage aliases.

I require immutable input and return immutable views/numeric facts. I do not
emulate arbitrary C pointers, allocator layout or copied-buffer ownership.
Under the fixed cardinality ceilings on these64-bit hosts, the default C
16MiB ordinary/mixed and2MiB map copy limits cannot add a refusal. The original
map unit observes maximum copied bytes787873 and work3607975, with scratch128.
My NanoLang representation has different heap costs; I preserve logical work
and declaration bounds without claiming the same heap footprint. Allocation
denial is not injected into NanoLang. Linux retains default leak detection;
I claim no Darwin LeakSanitizer coverage or complete leak freedom.

I retain every synthetic input and complete C report, paired source/tool
manifests, actual producer artifacts, failures and terminal results. The
manifests are post-run observations, not a full SDK closure. Temporary decoder
driver images are not archived. `retained-inputs.json` binds compressed inputs
to their raw length/SHA256 and compressed SHA256.

Declaration success grants no flow, borrow, initialization, handler or execution
authority. All four whole-module profile controls remain pending. Service/
ownership/private module conversion, ordinary/private executable verification,
representation analysis, C emission in NanoLang, canonical integration,
installed/raw bootstrap and the complete original #976 release scope remain
required. I change no product selection and close no parent issue.
