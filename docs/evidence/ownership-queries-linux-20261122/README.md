# My Linux ownership-query checkpoint

I publish source `6e218f2289ace10c9cc35cfe3e4a0021462386db` for #1142/#1203/#1204
on PR1168. I implement pure metadata queries for paths, globals, effect
signatures and element tables, array-constructor bindings and union variants.
I validate complete declarations before selecting facts and preserve the
public APIs' precondition order. These facts grant no execution authority.

My 3,080 complete reports match on actual C, ASan/UBSan C, C-seed native and
both C-produced and NanoLang-produced NanoVM products. The complete report
SHA256 is `b5973145e3390a20d652a55102bc12b925892c221507c78b261566a7ee9071a0`.
I retain all original declaration carriers, all 64 pointer-presence combinations
for each query, byte-exact effect-key aliases, missing/binary keys, capacity
and count-only contracts, truncations, transport refusals and the declared
path/global/effect/type/parameter/variant/constructor bounds. My C observer
checks unchanged output on failure and trailing canaries on successful copies.
My pure driver checks input immutability and copied path independence.

I found and repaired #1203: my public union-variant query admitted the internal
validation-only `NO_INDEX` sentinel, then published an uninitialized local fact.
I now refuse that ordinal before publication. The new check fails against my
retained original C implementation and passes after the repair. All 2,444
ownership-contract checks, including the new check, pass. I preserve internal
validation-only sentinel use.

My full query target **fails** under #1204. Both required canonical C AOT
entries refuse `path`, function8/offset545, with `narr_t versus nmap_value` at
result field2 while slicing a selected nested array. Both failures preserve
binary prior output. I retain the first reserved-identifier parse error and
an incorrect empty-UNION fixture expectation separately; complete declaration
validation requires SECTION_TYPE8 for that fixture. Neither correction removes
the array copy or weakens an assertion/deadline.

My source compiler is the retained native 09cab Stage2 product, SHA256
`346dc0d73af5960dea863fa724a0b1845f5613dff10b8661180ed74bf4eb3d83`.
My native toolchain is GCC13.3/as2.42 on Linux ARM64. My five identifier controls
pass; the actual lexer checks 3,446 fragments with zero reserved names and
retains 135 existing lexer refusals. My named-function shadow policy passes.
I record post-observation source/tool identities and preserve them through
final controls/export; I claim no new complete host/bootstrap closure check.

I leave Puck-owned `nvm2c.c` unchanged at SHA256
`12d43399e8b3fcda44964f90830d8aa3b24c637df31f0ce2f151527756372d67`.
The #1202 caller proposal remains unapplied. I rebuild the shared ownership
API repair normally; I compile no modified copy/slice caller candidate. Five-route query acceptance,
#1203's full source-route continuation, frame/whole-function CFG, pure C
emission, Darwin, later-source bootstrap and full release acceptance remain
open. I claim no historical timeout cause or performance improvement.

I retain 15,689 records, 207,895,121 logical bytes and 1,371 Git source identities.
I retain every original/failed/final corpus, product, command stream, source
snapshot and tool image in my content-addressed manifest. My reader checks
all archived hashes, Git source identities, independently expected results,
complete report equality, the original/fixed sentinel regression and required
AOT refusals:

```sh
python3 docs/evidence/ownership-queries-linux-20261122/read_archive.py
```

The printed integrity status records two required refusals. It is not query
acceptance or release acceptance. My NAS/migration/TCC operational pause,
frozen roots, original failure records, dirty audit and stashes remain intact.
