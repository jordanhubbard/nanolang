# I qualify the private raw file-service codec

I freeze source and tests at `f4381337c`; production is exactly reviewed
`fa708d9c1`, after contract5ede and raw-boundary clarification42bc.
[My transport contract](../NANOISA_FILE_SERVICE_TRANSPORT.md) remains staged:
this checkpoint contributes only the raw codec to task6833551, not its later
module retention/refusal integration or full6931/d03c/ed702 acceptance.

| First frozen gate | Result | Outer seconds |
| --- | --- | ---: |
| Linux GCC strict sanitizer method | PASS | 0.227 |
| Linux Clang23 strict sanitizer method | PASS | 0.282 |
| Linux normal GCC target | PASS | 0.202 |
| Linux adjacent v2 header/import suites | PASS | 7.816 |
| Darwin Homebrew Clang23 strict sanitizer method | PASS | 1.966 |
| Darwin normal actual Apple Clang target | PASS | 0.780 |
| Darwin adjacent v2 header/import suites | PASS | 6.529 |

Each codec method passes586 assertions against literal56-byte golden data,
including high-bit indices, UINT32_MAX-1, every short input/output size, long
input and SIZE_MAX length, null arguments, size-only mode, field-byte mutations,
reserved flags, ordered ordinals, version/catalog/endianness refusals, duplicate
and reserved indices, unchanged output/size sentinels and bytes beyond the written
payload. Exact input/output overlap is tested in both directions; decoding also
survives subsequent input overwrite. Large valid indices deliberately remain
accepted by the raw codec because import-table bounds belong to cross-section
validation. No allocation occurs, so there is no invented codec OOM gate.

The actual production C is separately linked into the fixture. Sanitizer runs
retain strict C11 warnings, ASan/UBSan and leak detection. Normal test-service-
bindings uses project CC/CFLAGS/LDFLAGS and joins test-units; the explicit sanitizer
target keeps compiler selection separate. I ran the selected target, not full
units. The unchanged adjacent suites pass29 header and38 imports/link/metadata/
debug controls on both platforms; they do not know the proposed new feature.

I seal26 reports in [report-sha256.json](service-bindings-codec/report-sha256.json).
Both hosts retain2,201 tracked inputs equal before/after and current, plus six
actual tool identities each, including Homebrew and actual Apple Clang on Darwin.
Manifests retain commands, statuses, timing and log hashes. The separate artifact
inventory is explicitly post-run executable/build/run identity, not a before/after
binary inventory. Linux frozen source remains at
`/home/jkh/Src/nanolang-service-codec-qualified`; Darwin at
`/private/tmp/nanolang-service-codec-f438`. No failing gate was observed.

No module known-mask, section/import reader, bridge, public verifier, service
runtime or generated-source path changes in this checkpoint. No service operation
executes and no public owned output is admitted. The next combined module
retention/consumer-refusal production still requires independent source review.

## My additive Socket integration

I preserve the frozenf438 tree and merge canonical Socket PR815 atd8caf70ff in a
separate integration tree. The sole conflict is adjacent Makefile target blocks;
I retain both complete recipes and both test-units prerequisites. My two codec
production files, two fixture files and full codec recipes remain byte-identical
to the reviewed d314 checkpoint. The roadmap merge is automatic. Socket adds its
own independently qualified sources/tests/docs; I do not claim the old inventory
covers those new files or that Makefile.gnu remains byte-identical. No new codec
execution is justified by this additive integration, and no gate is rerun.
