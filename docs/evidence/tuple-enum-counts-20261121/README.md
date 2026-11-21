# My tuple and enum count correction

Under #1160/#1142, I count wire enum kind 3 in `ownership.retained`,
`module.convert_claims` and `declarations.prepare`. Wire tuple kind 1 contributes
no enum. My old Python carrier default mirrored the same wrong discriminant;
I correct its independent count order to `(0, 3, 2)` and add explicit counts.
I add useful retained/projection shadows and preserve all original bounds,
malformed-input expectations, ownership flags and profile refusals.

My actual pre-correction NanoLang driver disagrees with the production C oracle
on 12 of 18 explicit-count carriers across ownership versions 1/2/3. It accepts
invalid enum/tuple counts and refuses valid ones; two invalid version3 inputs
also incorrectly return verifier-required success. I retain all 18 original
comparisons, the actual old driver and the four initial boundary witnesses.
`initial-setup.json` records the earlier invalid COMPLETE-flag setup; those
initial overwritten carrier bytes are not retained or represented as valid
witnesses. UNKNOWN flags expose the actual count disagreement.

| Original complete gate | Linux ARM64 | Darwin ARM64 |
| --- | --- | --- |
| Ownership/declaration/core-module methods | 6 pass, 674.846s | 6 pass, 234.991s |
| Shared ownership reports per five routes | 2,368; 1,433 explicit | Same |
| Actual-provider sanitizer reports per three profiles | 3,998 per profile | Same |
| Declaration/query reports per profile/five routes | 4,005 | Same |
| Core-module reports / original products / widths / File scans | 273 / 6 / 256 / 256 | Same |
| Whole ownership-module gate | 2,563 cases/five routes, 279.153s | 2,556 cases/five routes, 273.380s |
| Explicitly pending private ownership-module profiles | 8 | 7 |

I add 18 direct, six projection and four full-module count controls. My first
six-method gate includes the original seven C-produced declaration fixtures.
Its drivers use all five C/NanoLang execution routes. I then discover that the
fixture producer helper still omits the explicit raw compiler-module option;
I route it through my existing shared command selector. A separate retained
14-product gate executes seven C and seven actual NanoLang producer modules,
checks their VM results, and compares complete reports through all three
profiles and five routes on each host. I do not relabel the first4,005-report
gate as4,012 reports.

Linux's subsequent full ownership-module gate uses that corrected14-product
helper. Darwin's earlier full gate includes seven C products, followed by a
separate14-product full-module/claim comparison through five routes. These
account for the seven-case difference and one additional pending private input.
Private ARRAY/MAP profiles still return the quantified pending boundary; they
are not successful admission. The core-module gate also retains its three
separate pending profiles.

My paired audit checks1,334 identical source hashes across both hosts and the
retained current files,18 independently expected C reports, every retained
14-product/driver hash, all original suite terminals and matching producer
source identities. I compile the corrected library sources with the earlier
actual verified raw seeds retained in my
[seed receipt](../nanolang-service-v2-seed-20261121/README.md); this does not
create or claim a new compiler bootstrap. Generated C drivers use ASan/UBSan
and ordinary AOT runtime objects. The separate declaration oracle instruments
its real declaration/layout/retained/reference/cursor/ISA providers; other
common objects remain ordinary. I make no full-runtime sanitizer claim.

`manifest.json` retains214 files,115,922,297 raw bytes and8,949,146 compressed
bytes, with lossless members and raw/compressed SHA-256 identities. It includes
actual old/corrected drivers, modules, models, reports, collectors, source deltas,
source/tool manifests and test logs. Whole-module service integration,
instruction lifetimes, pure translator emission/cutover, native stages and the
full release remain open.
