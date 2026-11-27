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
An earlier candidate passes the complete five-route callable and solver suites.
My final combined candidate's guarded private/public/cache/solver/callable
cohort is still running; those earlier passes do not replace it.

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

I retain terminal logs, failed candidates, exact overlays, products, compiler
observations and the running cohort under the mounted volume at
`/home/jkh/.local/share/fleet-migration/puck-volume/nanolang/tmp/c-repairs-20261126`.
I will seal the terminal component archive and its independent reader after
the cohort finishes. The raw root is retained; an archive is not yet claimed.

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
libraries in its tool guards and is still running.
