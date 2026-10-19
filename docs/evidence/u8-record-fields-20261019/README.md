# I classify exact U8 record fields

I base this isolated slice on frozen aggregate `4cfad3737`. I record [ownership before source edits](https://github.com/jordanhubbard/nanolang/issues/1001#issuecomment-6262202165). The integration owner independently reviewed my exact predicate and focused controls without a blocking finding.

I add only exact `u8` recognition in `nisa_field_shape_supported_seen`. My old List<Fact> refusal follows that recursive predicate into its omitted scalar field case; array<u8>, local u8, ordinary layout TAG_U8 and expected byte value emission already exist. I do not change literal coercion, type matching, solver conversions, ownership or verifier rules. Shadows cover direct/nested/List byte records and unchanged unknown/cyclic spelling refusal.

I retain original C/Stage1/Stage2 refusals and provider hashes in [the prior U8 probe bundle](../optional-float-join-20261019/nano-optional-u8-typed-20261019.json). Those retained Nano stages predate this change. I build the changed C-hosted Nano compiler serially with:

```sh
NANO_SHADOW_TIMEOUT_SECONDS=300 bin/nanoc_c src_nano/nanoc_v06.nano -o bin/nanoc_u8_fields
NANO_OPTIONAL_RECORD_JOIN_COMPILERS="$PWD/bin/nano_virt $PWD/bin/nanoc_u8_fields" python3 -m unittest -v tests.test_u8_record_fields
```

Compiler imported/root shadows and all seven focused methods pass. The final run takes 10.306 seconds: 12 original verified VM modules and 24 O0/O2 ASan/UBSan native runs. Direct/nested/List fields preserve typed 0/255 values and record update independence. Actual direct-only modules publish one two-field layout with exact TAG_U8 and no nested layout indices; this assertion does not depend on List metadata authority. Conditional List record present/empty/unchecked cases pass; raw missing reads fail VM assertion/native checked invariant without sanitizer findings. Wrong field kinds, explicit U8 local literal 256, unconsumed resource and raw Nano integer field refusals preserve prior output.

I retain two earlier failures. A shadow local named `byte` conflicts with the C parser's reserved alias; I renamed it `octet`. The attempted shared direct-field literal-256 refusal exposed a separate gap: C accepts that raw field initializer, while Nano refuses it. The retained C dump contains PUSH_I64 256 with TAG_U8 layout, and the original fixture does not read the field. I make no runtime-safety or modulo claim for it. Both producers reject explicit `let bad:u8=256`; Nano still refuses raw integer record fields 0/255/256 at its exact matching gate. This contextual literal gap is separately tracked under #1001 and ROADMAP, not qualified by this change.

Each receipt JSON retains source, provider hashes, commands, status/output and module hashes from its printed artifact directory. I build no fresh Stage1/Stage2 and run no full bootstrap. Full #1001 and aggregate/platform qualification remain open.
