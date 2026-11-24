# My current GC registry refusal checkpoint

I integrate the exact reviewed GC correction and allocation-failure test from
`0d3f591b1ec5194550478eba3b3a27195855d614` into current source
`8d970c29319878fbd27b95349a2ed43b10759b1c`, under #1121 / PR1168.
The old PR1122, its branch, previous qualification and security history remain
separate and unchanged. This is a correctness repair, not a timeout optimization.

```sh
python3 docs/evidence/gc-registry-current-linux-20261124/read_archive.py
```

My independent reader verifies 4,154 records, 3,519 Git inputs and 63 retained
ARM64 native images. I require compressed/decompressed hashes and sizes, exact
current Git sources, byte identity with the reviewed GC/test origin, the precise
four-line Make addition, every original result and the failed predecessor.
Identical provider/image inputs reuse the preceding parser/source-C archives.

## My retained failure and correction

My fresh actual-current-source ASan/UBSan control fails at `owner == NULL`.
Refusing only the second allocation leaves a returned payload whose pointer
registry entry does not exist. I retain the full original source, exact test,
sole include-path mapping, commands, executable and streams before integration.

I apply the reviewed two GC source hunks. `gc_hash_add` reports success/failure;
`gc_alloc` registers the new pointer before linking its owner or charging any
statistics. A registry refusal frees only the unpublished allocation and returns
NULL. Existing owners, list links, statistics and finalizers remain untouched.
My public header, separate registry-entry allocation and size-overflow refusal
remain unchanged. I preserve the distinct existing finalizer visibility of
ordinary release and shutdown.

The permanent original matrix checks header and registry failure for ordinary,
STRING and opaque constructors; exact live-allocation/byte accounting; complete
statistics/list/owner preservation; successful retries; retain/release;
finalizers; repeated shutdown; and zero-payload reinitialization failure/retry.
It runs the actual production GC through controlled allocation, not a replacement
pointer registry. Its original test bytes match the reviewed commit exactly.

## My actual checks

| Check | Result |
| --- | --- |
| GCC normal / ASan+UBSan / O3 | All 12 allocation-matrix/original-GC build and execution commands pass |
| Installed supported Clang 18 normal / ASan+UBSan / O3 | All 12 corresponding commands pass |
| Original GC controls | All 13 methods pass in each compiler/profile |
| Environment C unit | All 60 checks pass in normal, sanitizer and O3 profiles |
| Original lexical-scope Python suite | All 10 methods pass with the newly linked GC-corrected compiler, preserving 120s/10s limits |
| Actual Make sanitizer target | Original GC suite plus exact retained allocation matrix pass |
| Original BSTRING sanitizer target | Pass |
| Original record-field sanitizer target | Pass, including 2,000 exact lookups/comparisons |
| Original STRING-bounds sanitizer target | Pass |
| Source freeze | All 3,501 selected tracked inputs match their source commit |
| Tool continuity | All 24 inherited actual compiler/loader images match the final observation |
| Public products | Five existing compiler/VM images and nine runtime libraries unchanged |

My matrix sanitizer builds include actual ASan/UBSan instrumentation and enable
leak detection with halt-on-error. The actual Make recipes retain their original
flags, assertions and Linux leak detection. I verify sanitizer symbols in the
executed GCC and Clang images.

The direct GC, BSTRING, record-field and STRING-bounds targets compile their
actual source providers fresh. My environment units rebuild GC and the current
C backend in all three profiles and reuse the already qualified matching shared
provider objects for the rest of that closure. I retain every provider hash and
link command. I do not claim a fresh complete provider build or a fully
instrumented compiler CLI. The lexical methods use the separately linked normal
compiler with current main/backend objects and the corrected GC.

I extend the existing `test-gc-shutdown-sanitizers` recipe by precisely the four
reviewed allocation-matrix compile/run lines. Its original dependencies, GC
method selection, flags, platform leak-detection choice and `test-units` membership
remain. I change no other recipe.

## My remaining acceptance

The allocation defect is repaired and its current focused original controls pass.
I keep #1121 open for its broader current-source/platform qualification. This
receipt does not merge or disposition the older PR1122 security check. Current
hosted CI, full compiler-source default10 admission, paired Darwin, raw-stage
bootstrap, installed execution and all 69 release requirements remain separate.

I make no unchanged full-source retry or deadline increase, assign no historical
timeout cause, claim no performance improvement and enable no ordinary ownership
policy. The preceding [bounded parser observation](../parser-shadow-cost-linux-20261124/README.md)
remains exact to its earlier source/providers; these GC changes do not rewrite it.
#1229/#1241/#1116/#1213 and default ownership stay open.

I preserve Puck's nvm2c/regex/merge/capture/CI lane, all frozen roots, primary audit
and existing stashes. NAS/TCC/migration pauses remain unchanged. My local native
controllers and their children are terminal; I hold no Darwin reservation.
