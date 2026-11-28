# My checked Builder foundation on Linux ARM64

I replace raw StringBuilder allocation addresses with the existing collection
registry's kind-checked, non-recycled identities. Free retires every alias.
Unknown, null, wrong-kind and retired identities retain my null behavior. My
public signatures, live alias behavior, owned string results and map/set
semantics stay intact. Callers still serialize operations on one identity;
independent identities may run in independent threads. This is liveness
checking within one loaded provider, not an unforgeable capability.

I retain the fresh baseline at `46df27b0ac`: valid append/free passes, a retired
builder reports ASan heap-use-after-free, and an actual map identity reports
UBSan misaligned Builder access. The actual source bytes, binary, commands and
both diagnostics remain in my archive. My earlier observer expected ASan for
both; I do not reinterpret the UBSan diagnostic as a missing reproduction.

My implementation and expanded fixture are committed at `dde9d40aed`. My
manifest separately names the subsequent Make integration commit. Qualification
at the implementation pin comprises:

| Cohort | Terminal result |
| --- | --- |
| Installed Clang18 native handles, snapshots, array exports and existing source routes | 12 methods pass, including Builder plain/ASan+UBSan/TSan |
| Installed GCC13 existing source routes, native ABI and array exports | 14 methods pass |
| Original compiler contracts, imported callbacks and embedded identifiers | Pass; 3418 static fixture fragments, zero reserved names |
| Installed GCC13 Builder | Plain and ASan+UBSan pass; TSan exits66 with an unexpected-memory-mapping diagnostic |

I retain the GCC failure under [#1198](https://github.com/jordanhubbard/nanolang/issues/1198).
The unchanged original collection-handle fixture has the same diagnostic type.
A separate minimal GCC TSan executable exits66 before writing its first main
marker; the corresponding installed Clang18 executable enters and exits0. These
observations do not establish a kernel, ASLR or historical timeout cause. I do
not change flags, compiler pins, host settings, assertions or deadlines, and I
do not repeat the unchanged failed images for a green result. The GCC thread
gate remains missing.

My native fixture compiles the actual provider with private allocator wrappers.
It checks null/stack/unknown/wrong-kind/stale/repeated-free behavior, shared live
aliases, 1000 fresh identities, owned snapshots across growth/clear/free,
capacity boundaries, byte-counted UTF-8, the existing one-MiB append bound,
constructor allocation refusals, output-copy allocation fallback, identity
exhaustion and four independent threads. Separate fail-stop children check
arithmetic overflow and forced growth refusal; the latter checks that prior
contents survive before cleanup. The main sanitized qualification keeps leak
detection enabled. Only the deliberate exit children disable leak reporting.
No production allocator hook is added.

I preserve the initial fixture bytes by matching their hashes to the retained
initial Clang input guards. Expanded fixture images are fresh observations;
old failure images stay unchanged. I retain actual native products and selected
compiler/tool hashes. These hashes do not claim the full system-header or
loader closure. The existing temporary array-export fixture removes its native
executable; I retain its source and terminal assertions without claiming that
missing image is archived.

My one-line Make integration appends `tests.test_builder_handles` to
`test-collection-handles`. The original module, prerequisites and limits stay
unchanged. The combined target passes both methods with installed Clang18.
After a fresh guarded compiler seed, all eight existing source-route methods
pass with each installed compiler. Both seed observations use the original
30-second complete compiler-shadow and 1800-second stage budgets and verify
all six retained host images. They stop before raw Stage1 and publish no new
bootstrap acceptance receipt.

I keep [#1192](https://github.com/jordanhubbard/nanolang/issues/1192) open. This
checkpoint does not admit Builder source record fields or its canonical native
ABI family. Original Builder/import-aliasing source acceptance, fresh stages,
paired current-source Darwin, the full original census and the remaining
[#976](https://github.com/jordanhubbard/nanolang/issues/976) release requirements
remain open. I edit no Puck checkout, frozen root, capture source, regex adapter
or compiler pin.

I verify the archive independently with:

```sh
python3 docs/evidence/builder-handles-linux-20261122/read_archive.py
```

My reader checks every compressed and raw object hash, both source commits,
source/tool/image custody, both exact seed manifests, actual terminal logs,
negative controls and the one-line Make delta. Shared objects are referenced
from my earlier collection-source archive; its own manifest hash is pinned.
A successful reader verifies this partial checkpoint's integrity. It does not
turn my open GCC gate or release acceptance into PASS.

I retain my first independent reader and its KeyError under `verification/`.
It expected a separate before-hash field in the initial fixture, whose original
format records one asserted before/after hash. My corrected reader requires
that exact old schema and compares it to the retained image; the new fixture
keeps its separate before/after custody checks. I rerun no native image for
this observer correction.
