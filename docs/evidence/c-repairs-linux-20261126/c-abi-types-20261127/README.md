# My C ABI selector component

I port the five value selectors aggregate_value_tag, aggregate_kind_for_tag,
c_result_type, result_is_i64 and c_local_type to pure
`src_nano/nvm2c/c_abi_types.nano` under #1142. I preserve their exact C strings,
flags and fallbacks. These facts grant no module, payload, profile, ownership
or executable admission. I do not implement checked table/pointer indexing
or full module representation through a spelling. No C product or compiler
selection changes.

My fresh original C tools build36.357857s on Linux ARM64. The four-method
permanent gate passes23.349s/all45 commands at original600/60/default10.
Ordinary/sanitized actual C and five NanoLang routes agree on:

- All256x256 result-count/tag pairs at a fixed array kind and all256x256
  result-count/array-kind pairs at ARRAY, with their256 duplicate rows removed.
  All256 local kinds and256 aggregate tags bring the matrix to131328 inputs.
- All6144 additional ignored-array-context controls,18 malformed packet
  fixtures, and good/refused/good/empty prior-output neighbors.
- Imported shadows through C-seed native and both producer VM paths.

I check every result against independent exact strings/flags. Zero-count or
VOID precedes multiple-result refusal; only count-one INT/BOOL sets the i64
fact. I preserve unsupported array storage and unknown result tags, default
local int64_t, and default aggregate TUPLE kind alongside a separate aggregate
predicate. The matrix covers complete branch input domains, not all256-cubed
result triples. My C observer compiles the actual five unchanged selectors
and word-array dependency, with the real ISA/function-entry headers and only
the minimal array-results owner field they read. This does not test the full
C owner structure's binary layout.

My source compiler is the actual qualified Stage2 image SHA-256
`cc94609575ad0d626e696449f486ed04336dff6e0887113542b09bce222a4820`,
bound to native manifest4c3fef1f342e7a6b2d0fe912100f9b1d299455d4db3db2d89783ab2b6c00dd8f.
All source/tool/qualified-producer/receipt/index/six-host guards match. Shadow
policy and five embedded identifier methods plus the full fixture audit pass.
Set the source compiler explicitly to reproduce the full five-route gate:

```sh
NANOLANG_TRANSLATOR_SOURCE_COMPILER=/path/to/qualified/bin/nanoc_stage2 \
  make -f Makefile.gnu test-nanolang-translator-c-abi-types
```

The shared route harness runs only three routes without that selection.

I retain my first zero-method setup failure: the new observer includes the
real function-entry header but omits isa.h, so TAG/AGG constants are undefined.
The corrected observer adds only that original header; every matrix/oracle,
C product byte, assertion and warning flag stays unchanged. My reader checks
the exact one-file/one-include correction and both guarded terminals.

I seal235 records/141 chunks/176220318 raw bytes, including both terminals,
exact new sources, actual helper headers, ordinary/sanitized reference
binaries, all packet/report bytes, generated products, selected C tools,
policies and producer receipts. Manifest SHA-256:
`40c1d41a253b9fa34df9e1bb98710820da9e0a28c9faf98dc3b63cf2c0aa4c75`.
My reader verifies4686 unchanged Git sources, all bytes/guards and all45
command streams. It replays962724 report lines through the original packet
builders/oracles, then replays the original imported-shadow method's exact
eight captured commands/assertions. Product execution is forbidden during
all four method replays. The initial report reader and strengthened complete
reader both pass; an altered manifest refuses before reconstruction.

```sh
python3 docs/evidence/c-repairs-linux-20261126/c-abi-types-20261127/read_c_abi_types.py \
  --archive docs/evidence/c-repairs-linux-20261126/c-abi-types-20261127/archive \
  --repo . --workspace /path/to/mounted/volume
```

This accepts only these ABI selections. Complete module representation,
declarations/body/runtime/profile/ownership/host emission, current paired
native/emitter qualification, frozen CAST_STRING reconciliation, installed
pure-translator cutover, exact-candidate CI and all69 release requirements
remain mandatory. I merge, tag and publish no release from this receipt.

My follow-up reader also requires first-run selected tools after the refused
build to match every selected tool before the corrected run. This inter-run
check passes without any product rerun or archive mutation.
