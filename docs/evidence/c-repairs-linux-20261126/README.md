# My C repair checkpoint — 2026-11-26

I own [#1262](https://github.com/jordanhubbard/nanolang/issues/1262) and
[#1265](https://github.com/jordanhubbard/nanolang/issues/1265) under the explicit
handoff. My starting source is `6b543189f9de7ce0895aa0bcc70084aab560cea2`.
I retain all 69 release requirements and every original acceptance deadline.
This checkpoint does not qualify or publish 5.1.

I repair deferred array-valued writes, independent numeric join storage,
checked shared-record array views and the present result of checked slicing.
My focused regressions pass 80 array-write cases, 28 numeric join paths,
16 static refusals, 250 expected runtime traps and 40 record-return cases.
My final guarded candidate passes all eleven original private/public/cache,
solver and five-route callable methods. I preserve the enclosing fourteen-method
observer failure: 31 supplementary subcases hit a text-stream logging error.
The three companion methods pass separately after that observer-only correction,
with identical source, tool and image guards across both runs.

I retain every backend compiler invocation and run independent captured
C-to-assembly inputs with at most eight workers. My real-process regressions
pass 18 subcases across three methods. The complete original bytecode-shadow,
module-cache, source-snapshot and link target passes 284 reported tests with
34 existing platform skips. These results precede the final slice correction;
my module-builder implementation is unchanged.

| Identical captured inputs and compiler arguments | Serial | Eight workers | Assembly bytes |
| --- | ---: | ---: | --- |
| 81 file-product units | 17.367427 s | 7.606510 s | Identical |
| 42 NanoISA units | 2.618612 s | 0.816396 s | Identical |

I retain the first comparison's subsequent optional-field observer error and
the separate corrected NanoISA comparison. The direct-module warm lookup
profile keeps all 331/331/172/172 compiler invocations and returns the same
pinned prefix paths and bytes. Its wrapper and execution layer differ from
the original VM diagnostic; I do not claim an original Darwin bootstrap gain.

My broad native target reaches 2,456 structured-C passes and two failing old
array CAST_STRING refusal assertions. The integrated
[#1193](https://github.com/jordanhubbard/nanolang/issues/1193) contract and its
newer tests require VM-compatible empty text. I preserve the failure and
[propose a test-only reconciliation](proposed-cast-string-test.patch).
I have requested an explicit decision before replacing those assertions.
I do not report this target as passing.

I seal 58,823 records in [my archive](archive/manifest.json), manifest SHA-256
`bed29fb26bfcd0ca4a8894d683fd7398a05634c080f219ed12d9a03c6bc3a872`.
My [independent reader](read_repairs.py) verifies all 9,818 compressed chunks,
3,933 guarded source files, retained source overlays, images, commands and reports.
I retain the raw root on the mounted fleet volume as well.

```sh
python3 docs/evidence/c-repairs-linux-20261126/read_repairs.py \
  --archive docs/evidence/c-repairs-linux-20261126/archive --repo .
```

My private/public/cache protocols preserve the original 1,001-frame report hash
and exercise 308/845/3,317 additional cases across six producer/execution routes.
My expanded five-method regression passes in 60.162 seconds: all original
80 array positives, 16 refusals, 28 numeric paths and 250 runtime traps remain;
60 new nonempty VM/native content cases and 36 new refusal controls pass.
I retain all three failed new-oracle proposals and the text-logger failure.
The original combined observer and the broad structured-C gate remain failed;
component acceptance does not qualify my release or check current liveness.

My branch-scoped `C repair original qualification` workflow requires hosted
workers before provisioning and runs separate Linux/Darwin bootstrap and cache
jobs. It preserves the default 600-second command, 1,800-second bootstrap-stage,
30-second compiler-shadow and ten-second ordinary-shadow limits, records source,
selected tool/image guards, and retains terminal logs, products and host closures.
A workflow definition is not a passing run. Full original bootstrap/raw equality,
installation/no-C-seed/native compiler/emitter acceptance, paired platforms,
installed pure-translator cutover, required exact-candidate CI and every other
5.1 requirement remain open.

## My first hosted and live-cohort refusals

I pushed the repair code as `d90b3feb9aab481387e9317b0a620ee071a1314b`.
[My first hosted run](https://github.com/jordanhubbard/nanolang/actions/runs/45948712006)
is terminal with four failed jobs. Linux records three cache-control failures
and a seed refusal after 1.268 seconds. Both real baseline and bounded probes
reject the stock GNU assembler 2.46 on madmax; my new workflow omitted the
canonical release workflow's existing verified capture-assembler selection.
I restore that action without changing product admission or any deadline.
The first seed log was not uploaded; I do not infer its exact cause from the
cache control. I correct retention of `obj/bootstrap-nanoisa` and
`obj/module_cache` before further hosted qualification.

Both Darwin jobs refuse admission before native builds. One records
3,582,410,752 free bytes of 192,749,240,320 after the full checkout. I preserve
the 20 GiB and ten-percent headroom requirements and use a Darwin product/test
checkout excluding historical `docs/evidence`. The original bootstrap/cache
and native-shadow gate inputs do not reference that archive tree. I preserve
all original tests, require all 36 native and 12 emitter methods, and keep
Linux's full checkout for its verified assembler profile. I provision the
pinned Apple toolchain and required libraries without an unused Homebrew LLVM.

My first sealed local cohort completes the private method, then loses the
external F28 producer root during the public method. My publication logical
symlink also disappears. I stop my exact observer, retain its incomplete
terminal and reports without claiming final guards, and restore my verified
publication alias. I do not assign a remover or a product cause.

I reconstruct 7,045 exact prerequisite files from published manifest
`3c2f440e292cb2c0c54f8b3d344b239c927223bed505c874e90831b842f1c5d7`
onto the mounted volume. The F28 image and all six host hashes match; its help
command succeeds without changing those hosts. This reconstructs a prerequisite,
not a new bootstrap acceptance. I also recover all seven original failing NVM
inputs from their pinned archives and verify their original hashes; the final
C tool translates all seven. My new complete cohort includes the six host
libraries in its tool guards and completes with all guards matching.

## My second hosted refusal

[My second run](https://github.com/jordanhubbard/nanolang/actions/runs/45952317501)
clears Darwin storage admission. Its tool-evidence copier then fails when
`shutil.copy2` attempts to copy protected assembler file flags. I copy exact
bytes and verify their hashes without requiring those system flags. Linux's
cache gate retains one controlled-boundary failure, awaiting diagnosis.
The Linux bootstrap job remains separately observed; neither platform is
qualified by these local component results.

I reproduce that Linux boundary failure on madmax and capture its child stderr:
GCC15 rejects the fixture's undeclared `usleep` under captured compilation.
Defining `_DEFAULT_SOURCE` before its headers exposes the existing declaration.
The original method then passes in0.161s with the pinned source-fixed assembler,
all ready/release assertions and the original ten-second boundary unchanged.
My hosted cache jobs also run all five expanded array regression methods.
The full hosted suites still have to pass.

The second Linux bootstrap job completes all17 original steps, raw Stage1/Stage2
equality, installation and no-C-seed acceptance. All36 native methods then pass.
The emitter companion discovers zero methods: `nanoc_c` stops during setup's
source shadows at the original ten-second limit. I retain that failure and
localize it with existing opt-in tracing before any repair. Twelve emitter
methods and the full hosted qualification remain unaccepted.

The exact hosted C-seed image completes all1,138 selected emitter-source shadows
on madmax under an opt-in trace, totaling5.038530s in per-shadow intervals and
completing under the ordinary ten-second supervisor. This does not establish
the hosted timeout cause or pass the twelve-method suite. I retain a bounded,
failure-only hosted trace for that original setup input, with process-group
cleanup and no changed deadlines.

I publish [my independently checked hosted Linux bootstrap component](hosted-bootstrap-45952317501/README.md), retaining all3,161 actual artifact records, the emitter setup refusal and the missing generated-index evidence gap.

## My native followups — 2026-11-27

My original hosted Linux and Darwin bootstrap steps pass in run45958449179.
Linux's complete cache target reports284 tests/34 existing skips; all worker3
and expanded array5 methods pass. Native acceptance remains open. Linux runs
native36 successfully, then Stage2 exceeds300s in emitter setup. Darwin's native36
has one exact-stderr failure from the duplicate `-lm` warning; its emitter suite
is not reached. My original assertions and deadlines remain.

I retain a separate original madmax terminal: bootstrap1021.362555s, native36
34.370s, then the same native Stage2 setup timeout300s. Selected source/tool
guards match. A bounded CPU profile retains14002 samples/no lost samples;
23% sampled user CPU is memset, with allocation and root-table costs as well.
I do not call those samples exclusive wall durations or a paired timing result.
The actual compiler uses legacy inline records with80 slots per storage arm.

My reversible generated-C experiment shares only mutually exclusive pointer
arms and retains independent scalar bits. All12 unchanged emitter methods pass
234.859s through132 commands. Both628200-byte emitted modules verify and share
SHA-256 `4db7d935ac0dd8f4a248f80e2c1515ac4701d12af602ad4ca0facc5ef321605a`.
I preserve the first comparison: it omits Make's exported `NANO_BUILD_CACHE` and
differs only in three host-library paths. The corrected comparison restores the
original cache directory and requires exact bytes.

My isolated generator candidate retains canonical storage and checks legacy
pointer-carrier sizes at compile time. All broad prerequisites pass; structured
C remains2456 passes/two unchanged CAST_STRING failures. A new sanitized control
retains six mixed pointer fields, boxed scalar bits and a NaN payload. My Darwin
followup omits the redundant generated math flag while retaining every caller
flag; actual bounded Darwin math linkage and the new selected VM shadows pass.
The control compiler is Apple clang21.0.0 build2100.3.34.2; the pinned hosted
build2100.4.1.1 still requires native qualification.

I retain the new raw originals, profiles, prototypes, comparisons and preflight
logs under the mounted volume's `nanolang/tmp/c-repairs-followups-20261126`.
Those raw followups are not yet a published component archive. The existing
sealed archives remain unchanged. I require fresh current-source original
bootstrap/native36/emitter12, paired platforms and all69 release requirements.
