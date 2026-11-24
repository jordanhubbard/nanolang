# My Linux native-publication checkpoint

I retain source **af4583722bf2876297c7fe645579d002b80ddece** under #1237,
with implementation commits 33946fd483 and af4583722b on PR1168.
My focused native-publication checks pass. My whole-release acceptance is
**false**, and I retain the separate #1242 source-C failure.

I give the configured native compiler a provisional executable beside the
requested destination. I require a nonempty regular executable with one link,
finish temporary-C cleanup, recheck source identities and rename only after
success. I preserve the original compilation flags and shadow deadlines.
I check a partially loaded import graph before writing import-error diagnostics.
[My publication contract](../../NATIVE_PUBLICATION.md) records stable-filesystem,
coverage, inspection-output and crash-recovery boundaries.

| Actual final-source check | Result |
| --- | --- |
| Ordinary publication suite | 18 methods pass, no skips |
| Private-collector publication suite | Same 18 methods pass, no skips |
| Actual selected-compiler Make target | Same 18 methods pass |
| Ordinary/private original enum suites | 21 methods pass in each mode |
| Original phase/shadow/import neighbors | 15 pass, one source-C method fails |
| Independent real gcov consumers | Both note/data observations pass |
| Embedded-fixture policy and its controls | Pass |
| Fresh main.c normal/sanitized/optimized compilation | Pass with original flags |
| Five public tools and nine library guards | Match retained originals |

My publication suite retains real C compiler and missing-symbol linker failures,
partial-output and false-success compiler controls, symlink-candidate refusal,
cleanup failure, successful replacement and native execution, file modes,
output symlinks/directories, root/dependency path/symlink/hard-link aliases,
kept/saved C and actual coverage output. My ordinary/private compiler images
reuse retained provider objects; compiling main.c in three profiles does not
establish a fresh complete provider build or a complete instrumented-driver run.

I preserve three independent original source-loss observations: the linker
removes prior output after one successful shadow; source equal to output is
replaced with executable bytes; and a later missing import lets diagnostic JSON
overwrite an already loaded dependency. Every input is an owned fixture. The
final tests preserve both sources, prior bytes and permissions. I retain the
initial 14- and 17-method phases, all native products, commands and streams.

The original 16-method neighbor suite remains unsuccessful. Its source-only
qualified import method refuses with `I require the exact emitted declaration
for this qualified C call.` The retained predecessor and final candidate both
reproduce that refusal. A separate same-source observation preserves old output
and produces no shadow report on both images. I track this under #1242; I do not
weaken its assertion or present the other 15 methods as a green complete suite.

I preserve the earlier full-source deadline failures in my unchanged
[preceding partial receipt](../public-c-fixture-recovery-linux-20261123/README.md).
I admit no unchanged full-source retry, assign no historical cause or speedup,
and claim no fresh shared-provider, current-head Darwin, raw bootstrap,
installed-product, default aggregate ownership or whole-release acceptance.
All 69 release scope entries and the existing operational pauses remain in force.

My independent reader passes **4,351 records / 1,469 Git inputs / 108 native
images**. It verifies the exact Git source pin, compressed object identities,
actual command outcomes, original test-source continuity, failed neighbor and
predecessor observations, public guards and explicit false acceptance fields.
My first package inventory check rejected superseded unreferenced compressed
objects. I retain that diagnostic and the superseded local package objects;
my final archive contains exactly its referenced compressed objects. This was
an archive-inventory correction, not a native retry or source-test change.

```sh
python3 docs/evidence/native-output-publication-linux-20261123/read_archive.py
```
