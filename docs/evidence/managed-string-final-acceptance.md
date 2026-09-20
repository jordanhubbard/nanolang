# I qualify my original managed-string contract

I complete the measured acceptance in task_5792220dc3654ddcbe7e47ec0253f8ea
for the original string lifetime contract51da. My compiler/runtime production
is unchanged. I add an explicit emitted-IR optimization selector, a failed
managed initializer control, strict supported Darwin leak checking and a
retained qualification runner. Independent source and fixture review preceded
execution. Publication still requires canonical integration and review.

## My actual acceptance matrix

| Phase | Linux host | Methods | Seconds | Darwin host | Methods | Seconds |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Original emitted string/conversion modules | sparky |23|99.480|CXWWHGGJX0|23|145.467|
| `default<O2>` emitted string/conversion modules | sparky |23|129.495|CXWWHGGJX0|23|181.234|
| Core allocator and embedded runtime package | sparky |5|15.133|CXWWHGGJX0|5|21.306|
| Scalar globals, literal strings, enum/numeric and verifier neighbors | sparky |42|425.700|puck|42|340.809|

All eight phases pass with no skipped methods, unexpected failures, timeout,
unconfirmed cleanup or changed source/tool/provider input maps. Times include
retention overhead. Original/O2/core use frozen971b075a2; corrected neighbors
use76debee8f. Each preparation builds fresh providers before the gate. The unused
76de CX preparation also passes; its neighboring suite is not run there because
free space is less than the observed size of the Linux retained suite. Its
artifacts remain intact. I do not present these as eight phases on one Mac.

Both optimization routes execute the same complete string and conversion
modules. I verify original and selected IR, optimize before explicit native
ASan instrumentation, and run the selected Wasm through Node and Wasmtime.
The separately published Wasm command remains an additional control. I retain
every original lifetime/publication assertion and explicitly check failed
initializer cleanup: committed global bytes survive, temporary local bytes do
not, entry does not run, and disposal releases the remaining global.

My unchanged core/package methods cover allocation/refcount failure, reuse,
growth, fragmentation, live-byte accounting, embedded runtime bytes and real
native/Wasm links. This package phase does not perform a prefix installation.
The emitted methods retain aliases, calls/returns, globals, reentry, content,
casts and terminal cleanup. My source contract and selected-method inventories
provide the exact scope; these totals are method invocations, not independent
source programs. Neighbor native sanitizer link flags and the default-main
control do not prove explicit IR instrumentation. The separate marked-IR
methods carry that instrumentation claim; linked engines are not thereby
instrumented.

## I retain the first terminals and exact corrections

The first Linux preparation atcd312 fails before any fixture because Clang23
selects GCC14 with missing C++ headers under strict warnings. I select the
installed GCC13 explicitly for native compilation and preserve Wasm flags.
No warning is suppressed. Exact compiler/tool/library hashes and environment
selection are retained for every host. Puck uses a task-local copy of the
qualified CX Wasmtime48.0.2 binary, SHA256
`130b4a32cecb619d52aefedda39c6733f00ee7460bbe8748f99eac72143c980c`;
its system-library dependencies and executable hash were checked before use.
I do not change a global tool installation.

The original971b Linux neighbor run passes four methods, then encounters an
obsolete refusal for `ARR_NEW 5; STORE_GLOBAL 0`. Corrected342 passes all scalar
globals and stops at method19 on the literal fixture's obsolete `ARR_NEW 5; POP`
refusal. Static review finds two analogous enum-conversion refusals. These
operations were already admitted by earlier merged production. I preserve each
exact program as positive VM/original+optimized LLVM/native/Wasm execution,
replace and execute both translator output sentinels, and retain the real
initializer/type/import/tail-call and verifier negative controls. Taska8bc
records these fixture corrections before execution. Both failed directories,
logs, modules and overwritten outputs remain retained.

Two initial Darwin audit attempts are interrupted after prolonged archive
file-open waits; a process sample locates the wait in `open`. Bounded audits
with per-object progress subsequently rehash the complete evidence successfully.
No checksum mismatch was established and no qualification test was rerun by
those audits. I preserve the interruption record rather than assign an
unproved infrastructure cause. A setup fetch initially addresses a clone's
local origin, which lacks the branch; the explicit GitHub fetch succeeds before
checkout or preparation. Both setup outcomes are retained.

## My sealed reports and independently checked bytes

My [matrix](managed-string-final-acceptance/summary.json) records all statuses,
source pins, report-manifest digests and retention locations. Each phase contains
the exact selected methods, log, host/tool selection, complete report manifest
and independent audit. The audits rehash every uniquely referenced archived
object and current immutable input, compare Git source identities, inspect
each command's input equality and confirmed cleanup, and check individual test
outcomes. Expected nonzero commands remain visible as negative controls.

The complete per-command maps, original products and content-addressed objects
remain at the recorded locations; the repository contains summary reports and
indexes rather than claiming to embed all artifact bytes. The retained runner
uses stat-keyed cached digests between commands and fresh phase endpoint hashes.
I do not describe those cached command maps as repeated independent byte reads.
The selected tool/library inventory also does not claim every transitive loader
or compiler dependency.

An additional complete original-Darwin backup is retained on sparky at
`/tmp/nanolang-managed-string-darwin-retained/nanolang-managed-string-971b-darwin-original.tar.gz`.
I independently read all 8,844 archived entries, 4,582 reports and 3,902 objects;
its report manifest matches the original audit. The 357,089,280-byte archive has
SHA256 `1cb72893aeb2cff65e9ccc6b638addc8e7b7179336b1ec076388c236cc56af64`.
The original CX directory remains intact.

This evidence satisfies the original bounded managed-string acceptance after
canonical integration. Aggregate/cycle488, host linkage2d2, full applicable
language coverage, historical evaluator incident791a, NanoISA-only bootstrap
and the full5.1 release remain separate. A passing string matrix does not
authorize publication.

## I check canonical integration separately

At e4f67f981 I merge canonical5d1d and preserve both roadmap sections from the
only textual conflict. Fresh provider preparation and all23 original methods
pass on Linux and puck. Independent audits verify4,582 reports and456 command
pairs on each host, plus3,864/3,914 archived objects respectively. My
[integration matrix](managed-string-current-integration/summary.json) retains
actual timings, statuses and exact pins.

Main then gains the independently qualified public File opt-ins in PR895 at
97546742a. I merge that change automatically at7d0b661e6. Independent review
finds unchanged default CLI bodies and managed routes; the new File paths have
their own paired integration/refusal/wrapper evidence. My
[identity record](managed-string-current-integration/integration-identity.json)
lists the complete incoming path set and56 identical managed source/fixture
blobs. I retain the e4f execution attribution; this later merge is a reviewed
composition, not a new run of those binaries. No new bootstrap or full release
claim follows.
