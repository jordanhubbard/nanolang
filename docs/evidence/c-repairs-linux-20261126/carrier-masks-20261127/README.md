# My exhaustive scalar-carrier component

I port scalar_kind_tags, boxed_carrier_tags and optional_scalar_tags to pure
`src_nano/nvm2c/carrier_masks.nano` under #1142, after my storage-kind
component. These facts grant no payload, shape, profile, ownership or
executable admission. No C product source or compiler selection changes.

My fresh original C tool build passes36.505444s on Linux ARM64. The permanent
three-method gate passes12.280s, all38 commands, at original600/60/default10.
I compare all256 kind bytes and65536 masks across ordinary/sanitized actual C
and five NanoLang routes: C-seed native and both producers' VM/sanitized-native
paths. I compile the actual unchanged three C helpers and their two
variant-mask dependencies from the pinned C source with its real ISA header.
The original storage-kind source and tests stay unchanged.

Independent exact sets check all five known scalar mappings and251 unknown
bytes (0x8000),258 boxed masks and four optional masks. I preserve the
reference's TAG_BOOL upper cutoff: optional string/enum/function masks do not
satisfy it. All15 malformed packets refuse before reports; good/refused/good/
empty neighbors preserve prior outputs. Imported shadows pass through C-seed
native and both producer VM routes. Repository shadow policy and five embedded
identifier methods plus the full fixture audit pass.

My source compiler is the actual qualified Stage2 image SHA-256
`cc94609575ad0d626e696449f486ed04336dff6e0887113542b09bce222a4820`,
bound to native manifest4c3fef1f342e7a6b2d0fe912100f9b1d299455d4db3db2d89783ab2b6c00dd8f.
All source/tool/qualified-producer/receipt/index/six-host guards match. For a
fresh full five-route run, set the compiler explicitly:

```sh
NANOLANG_TRANSLATOR_SOURCE_COMPILER=/path/to/qualified/bin/nanoc_stage2 \
  make -f Makefile.gnu test-nanolang-translator-carrier-masks
```

The shared route harness runs three routes when no NanoLang source compiler
is selected. That default does not establish five-route acceptance.

I seal176 records/112 chunks/150609124 raw bytes, including exact new sources,
all packet/report bytes, actual helper header/reference binaries, generated
products, selected fresh C tools, policies and immutable producer receipts.
Manifest SHA-256:
`c9f83eb87d97c185f49a63aeca25206db75f83341d82c33ce9ac9aae09c99a11`.
My reader checks4682 unchanged Git sources, all bytes/guards, then replays
460810 report lines using the original packet builders/assertions with process
execution forbidden. Its first invocation retains a reader-only stale
worktree-name refusal; the corrected exact name precedes acceptance. An
altered manifest refuses before reconstruction.

```sh
python3 docs/evidence/c-repairs-linux-20261126/carrier-masks-20261127/read_carrier_masks.py \
  --archive docs/evidence/c-repairs-linux-20261126/carrier-masks-20261127/archive \
  --repo . --workspace /path/to/mounted/volume
```

This accepts only these scalar-carrier facts. Full module representation,
body/runtime/profile/ownership/host emission, paired current native-emitter
qualification, frozen CAST_STRING reconciliation, installed pure-translator
cutover, exact-candidate CI and all69 release requirements remain mandatory.
No release is merged, tagged or published by this receipt.
