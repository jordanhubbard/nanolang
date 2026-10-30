# I retain exact U8 field contexts

I work from `381093ad9` after the aggregate source pin `4cfad3737`, on `fix/5.1-u8-field-context`. I record [ownership before edits](https://github.com/jordanhubbard/nanolang/issues/1001#issuecomment-6262401220), the [evaluator defect before correction](https://github.com/jordanhubbard/nanolang/issues/1001#issuecomment-6262544896), and the [strict C boundary](https://github.com/jordanhubbard/nanolang/issues/1001#issuecomment-6262666646). The integration owner independently reviewed source routing, evaluator cleanup and anonymous staging tag retention without a blocking finding.

My [contract](../../NANOISA_U8_FIELD_CONTEXT.md) uses exact checked record field identity, literal 0..255 range checks and existing computed INT modulo conversion. I preserve raw expression types and evaluate direct U8-bearing constructor operands in written order once, then pack in declaration order; spread evaluates its base before written overrides. I apply the same declared U8 boundary after single evaluation in seed shadows. I do not change global type matching, ownership, solver rules or verifier admission.

I build serially, without a full bootstrap:

```sh
make -f Makefile.gnu -j1 nano_virt bin/nanoc_c
NANO_SHADOW_TIMEOUT_SECONDS=300 bin/nanoc_c src_nano/nanoc_v06.nano -o bin/nanoc_u8_context
NANO_OPTIONAL_RECORD_JOIN_COMPILERS="$PWD/bin/nano_virt $PWD/bin/nanoc_u8_context" NANO_U8_COMPILERS="$PWD/bin/nano_virt:$PWD/bin/nanoc_u8_context" python3 -m unittest -v tests.test_u8_field_context tests.test_u8_record_fields tests.test_computed_u8_context
```

The changed Nano producer's imported/root shadow build passes. My final 15 methods pass in 24.379 seconds. Actual C and changed C-hosted Nano modules run through the original VM verifier and VM, nvm2c, and O0/O2 ASan/UBSan binaries. Direct/spread, nested constructor and call argument fixtures assert side-effect order and single evaluation, exact 0/255 and signed-boundary values. Imported same-short-name records retain distinct checked identities and pass legacy native shadows/execution. Existing direct/nested/List byte-field cases, checked raw-missing failures, exact TAG_U8 layouts, wrong-type/range refusals preserving prior output, and unchanged resource refusal pass. The neighboring computed-U8 contexts also pass. The raw registry unit refuses missing, ambiguous and mismatched declaration identity.

I retain failures rather than replacing them:

- `c-first.log`: actual legacy shadow assertions exposed missing evaluator field narrowing; the narrow resolved TYPE_U8/VAL_INT correction passes the same shadows.
- `c-after-evaluator.log`: original strict fixture refuses unsupported cast_int; this is a call-profile boundary, not a narrowing defect.
- `c-bounded-profiles.log`: strict computed constant expression 256 emits defined implicit uint8_t conversion but clang reports `-Wconstant-conversion` under `-Werror`. I leave that strict emitter warning limitation open. My strict pure numeric qualification uses function-result operands instead; canonical and legacy constant-expression tests remain.
- `c-numeric-profiles.log`: strict generated unused runtime helpers trigger warnings; its final O0/O2 sanitized compilation uses `-Wno-unused-function` with other `-Wall -Wextra -Werror` checks retained.
- `focused.log`: new C scratch slots lacked tag facts and caused omission of the prior exact-layout metadata. The correction registers only their exact stored types through the existing helper; `focused-final.log` passes the unchanged layout assertion.

Strict C compound initializer operand order remains unspecified; I do not claim strict constructor-order parity or add a statement-only workaround. Strict spread remains refused. Canonical C AOT is nvm2c; its direct/spread ordering is exercised above. I retain all artifact paths, source, command/status/output and module hashes in JSON receipts; current producer/tool hashes are in `providers.json`. These producers are changed-source C-hosted builds, not fresh self-hosted Stage1/Stage2 products. Fresh aggregate stages, platforms, strict ordering/warning limitations and full #1001 remain open.
