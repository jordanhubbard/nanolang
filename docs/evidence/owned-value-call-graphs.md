# My bounded owned value-call graph evidence

My final integrated qualification at42ce55c2 passes all listed ordinary and
instrumented gates. I preserve earlier incomplete outcomes below as history;
they are not the final acceptance result. The last section records exact pins.

I track `task_4ce5cfc5b8034949852255d7307c9f91` under the unchanged affine
example blocker `task_c4351c720aee424ea9b90187e51a08f2`. My reviewed contract
is [the first runtime prerequisite](../NANOISA_AFFINE_EXAMPLE_PREREQUISITES.md).
My production checkpoint is `fea6639f`; it builds both runtime and translator.

I retain eight functions/frames, complete acyclic direct-call validation,
mode-zero scalar/resource arguments and one scalar result. My entry is not a
call target. Each VM activation has its own reference context and generation;
my native value helpers have private local origins and share the monotonic
generation counter. I preserve the separate borrowed CALL_REF profile.

My new positive fixtures include four-function and eight-function graphs,
zero-argument entry/wrapper calls, two distinct nominal resource parameters,
repeated calls and sibling targets. Callers retain a hold on an unrelated
owner while nested helpers use identical local/reference indices. All four VM
APIs run each case twice, with an assertion failure at each of the eight
active depths, complete root/context cleanup and subsequent entry. An ordinary
non-owned ten-frame chain retains generic dispatch behavior.

My deeper preflight fixture fails contract allocation, the last positional
nominal check, and stack reservation before the fifth activation. All four
APIs retain the generation before that activation, unwind earlier owners and
contexts, and then complete a normal retry. These are distinct from native
owner allocation controls. I do not claim exhaustive allocation-site coverage.

My first test build retained a misleading-indentation warning; I separated
its test statements. My first deeper stack fault fixture set capacity 64,
which forced growth during preparation of two arguments before the intended
callee preflight. I corrected capacity to 66: four sixteen-local frames plus
two arguments fit, while the fifth frame does not. The corrected fixture
passes 338 checks. I preserve the initial assertion log at
`/tmp/nanolang-owned-value-graph-preflight.log` and the corrected run at
`/tmp/nanolang-owned-value-graph-preflight-corrected.log`; I do not attribute
that test setup mistake to runtime behavior. My initial combined gate also
found the old verifier diagnostic assertion still expecting an entry-to-helper
message. It now requires the new checked-acyclic-call diagnostic; the same
refusal remains required.

My original ordinary full run is
`/tmp/nanolang-owned-value-graph-full.log`. The new paired gate passes 1847
checks plus 338 preflight checks. Each of ten native cases uses strict C
warnings, ASan/UBSan/LSan, repeated invocation and injected owner allocation
failure with no retained native roots. My unchanged production and tests pass the existing single/multiple consuming
(3091/4542 lifecycle checks), borrowed (1548 and 2004), helper-local (1309),
assertion (959), affine state (314/343), affine bytecode (441/751), verifier 96,
VM 274541, native 2422 and shape 1365 gates, including their recorded allocation
controls. The full command is:

```
make -j4 test-owned-value-graphs test-consuming-calls test-multiple-consuming-calls \
  test-caller-references test-multi-caller-references test-helper-local-owners \
  test-owned-assertions test-affine-state test-affine-bytecode test-verifier \
  test-nvm2c test-nanovm
```

My first Clang attempt stops in its driver on `-Wgcc-install-dir-libstdcxx`
before compiling generated C. I retain
`/tmp/nanolang-owned-value-graph-clang.log`. An explicit existing GCC13
selection, `clang --gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13`, passes
the same ten strict/sanitized native cases in 5.367 seconds without warning
suppression (`/tmp/nanolang-owned-value-graph-clang-pinned.log`).

I restack onto canonical 55208c47 at ae2a83e0. All nine changed production
files and the graph fixtures are byte-identical to tested 8349b23e. The incoming
changes are separate managed-array graphs, reconstruction and evidence. My
rebuilt integrated tools pass the paired graph/preflight gate again
(`/tmp/nanolang-owned-value-graph-integrated.log`). Their SHA256 hashes are:

| Tool | SHA256 |
| --- | --- |
| bin/nano_vm | cbe18a4e8703e6681586bfb114bcef6e71fcba60c1778e610e6fe6a0321e332c |
| bin/nvm2c | be3bf4f9718cbc5241e40d11ba6af7f28fcdd7d90e20a4de505a6c657e1c3357 |
| obj/test_owned_value_graphs | ea674f770e7d79eb0a239b8f83194e0ea30e5faf10ca0e511f990193c65e3c12 |
| obj/test_owned_value_graph_preflight | dd5d0e650446ba178bf9a1cecd1d1ef2fe6bfc389097bd4b2f45c535cb618563 |

My separately instrumented O0 VM/NanoISA run reaches its 600-second bound and
remains incomplete. I retain `/tmp/nanolang-owned-value-graph-sanitizers.log`;
there is no passing claim for that run and no demonstrated correctness failure.
I preserve all ten cases, all four APIs and two repeats in a separate O2
ASan/UBSan/LSan build. Its log is
`/tmp/nanolang-owned-value-graph-sanitizers-optimized.log`; it also reaches the
600-second bound and remains incomplete. After both runs terminate, I add an
optional case selector to the harness. The default full gate still passes
1847 checks and 338 preflight checks unchanged. A separate partition invokes
each of the ten identical cases with all four APIs, two repeats and every
assertion, at most two processes at once and 300 seconds per case. Per-case logs
and statuses are in `/tmp/nanolang-owned-value-graph-sanitizer-parts`; their
aggregate is incomplete: cases0/1 each reach300.066 seconds; cases2/3
were automatically launched before the stop instruction and are interrupted;
cases4..9 are not launched. A current ordinary-case stack attach is denied
under ptrace restrictions, so I obtain no stack evidence. Each selected-case
invocation still executes the unchanged ordinary-chain, graph-refusal and
four-function verification preambles, so these timeouts do not localize work
to the selected cases. I track investigation
`task_5b3e7272ec3f4ae9ad8eb8ffb15a3565` before any repair.

My static runner audit finds the expected ASan8/UBSan1 ELF dependencies and
recompilation of all linked NanoVM/NanoISA source objects. Ordinary compiler
support objects remain normally built, as already disclosed. I have not
established a tool or runtime defect. My next bounded diagnostic keeps all
fixtures/assertions and emits optional phase markers before/after each common
preamble and fixture/verification/artifact/API phase. One corrected ordinary
case may then localize progress under an explicit bound; I do not schedule
another whole corpus or reduce coverage to obtain a pass. I do not relabel either full-run timeout as a pass. Both builds instrument the test driver and every linked NanoVM and
NanoISA object, with ordinary compiler/runtime support objects linked normally.
I do not claim that the entire legacy compiler is instrumented.

I do not claim source admission, owned/void results, string/PRINT effects or
restoration of the example from this runtime prerequisite. I do not replay
its frozen failure artifact. The parent release blocker remains open.

### Timed diagnostic after zero-link verification reuse

I retain production at 246f1464 and commit the diagnostic contract at f78705c4.
My generated diagnostic VM copy changes only stderr markers and wrappers that
return each verifier result unchanged. My runner timestamps captured lines with
monotonic elapsed time. Fixtures and assertions remain unchanged. I preserve
script `/tmp/nanolang-owned-timed-diagnostic.py`, generated source, executable,
`timed.log`, and `status.json` under `/tmp/nanolang-owned-timed-diagnostic`.
The executable SHA256 is
`1b08a6ee7ab9c3e4c4545047612b6938f691220339b865fc4e166a22d316e310`;
the generated diagnostic source SHA256 is
`b0c470d04d7fdc2f57e52546b9d78a9661ce61e68d8d77ecb32da5fc073cce08`.

My single ordinary case0 run reaches its 60-second bound in 60.030 seconds and
is incomplete. I do not report an API completion or instrumented corpus pass.
My observed phase intervals are:

| Phase | Elapsed interval | Duration |
|---|---|---|
| Eight-function verification | 1.928–6.156 seconds | 4.228 seconds |
| Artifact generation | 6.156–30.494 seconds | 24.337 seconds |
| VM initialization linked verification | 30.496–33.268 seconds | 2.772 seconds |
| API0/repeat0 | Begins at 33.268 seconds | Incomplete at deadline |

During that API interval I observe 18 owned-admission calls begin and 17
complete. The completed calls total 25.154 seconds, with individual durations
1.465–1.564 seconds. My markers show three admissions on each core reentry.
Static source inspection explains that OP_ASSERT returns TRAP_ASSERT even for
true conditions; the ordinary outer handler then resumes core execution.
This establishes substantial repeated verification cost in the measured API
portion, separately from artifact generation. It does not establish a sole
cause for earlier timeouts, a runtime nontermination defect, or permission to
reuse proofs across mutable execution contexts. I require a separately reviewed
contract before any additional reuse change.

My final production246f existing full regression run also completes successfully:
274541 VM checks, 2422 native checks, and 1365 shape checks. I preserve
`/tmp/nanolang-owned-zero-link-full.log` separately from the incomplete diagnostic.
The focused246f log retains 69 verification-reuse controls, 1847 graph checks,
338 preflight controls, and adjacent authority gates. At this diagnostic checkpoint my PR remains a draft.

### Completed runtime-invocation qualification

I qualify reviewed production071c126c at frozen source
871b638dbef236d8d24f3be4dba9ce5bc024659b. My private proof lives only on a
single synchronous invocation's C stack. My 529 admission-boundary checks count
exactly one full runtime admission for each accepted or false-ASSERT invocation,
across all four APIs and two repetitions. Later public calls establish new
proofs. My changed-declaration control requires refusal before activation; it
does not promise arbitrary mutable bytecode or decoded-cache replacement.
I preserve its initial overly strict counter assertion in
`/tmp/nanolang-owned-invocation-proof-focused.log` and track its correction as
task_6b0e7040a5534b1cb14da14f8291623b. Declaration validation may refuse before
full admission, so that refusal permits zero or one admission; accepted calls
still require exactly one.

My corrected ordinary gate passes 1847 graph, 338 preflight, 529 proof-boundary,
and 69 verifier-reuse checks. GCC and pinned Clang generated-native cases pass;
the Clang ten-case test completes in3.308 seconds. The full regression passes
274541 VM, 2422 native, 1365 shape, 96 verifier and 28 VM-FFI checks, alongside
all existing affine/owned/borrowed/helper/consuming authority and allocation
gates. Logs use the `/tmp/nanolang-owned-invocation-proof-` prefix:
`corrected.log`, `clang.log`, `regression.log`, and `public-boundaries.log`.

I now complete the unchanged ten-case instrumented corpus, keeping every case's
four APIs, two repeats and lifecycle assertions. Every linked NanoVM/NanoISA
object and the driver is freshly built with ASan/UBSan at O2, with leak checking
enabled; common legacy compiler/runtime objects retain their ordinary build.
At most two cases run concurrently, each with a300-second bound. All ten exit0
without sanitizer findings. The aggregate finishes in230.170 seconds; cases0–8
take45.681–46.485 seconds, and case9 takes7.833 seconds. I retain the instrumented
build and executable under `/tmp/nanolang-owned-invocation-proof-sanitizers-build`,
with executable SHA256
`b769c3d14ab04c11e58922a45a70e37364dbe38490cd35997c993d5545ca69fa`.
Per-case logs/status and the complete summary remain under
`/tmp/nanolang-owned-invocation-proof-sanitizer-parts`; build/runner scripts and
logs share the invocation-proof prefix.

This completed partition aggregate is separate from every earlier incomplete
full run, timed diagnostic and interrupted partition. I neither relabel those
outcomes nor claim the original600-second runs passed. The measured correction
removes repeated admission across internal assertion resumptions; independent
artifact generation still has measurable cost. I next integrate current main
before declaring the PR merge-ready.

### Final integration on canonical main

I merge canonical main25a685ad into my branch at
42ce55c233ae99e276f30e0187b46e0a1bbf9309. The only conflict is additive Makefile
content: I retain my graph target and incoming binary64 dependency/flags. The
incoming float arithmetic and managed profile code merges separately from my
reviewed ownership/proof logic. I preserve both and rebuild affected tools.

My final own-tool ordinary gate again passes 1847 graph, 338 preflight,
529 admission-boundary, 69 verifier-reuse, 274541 VM, 2422 native, 1365 shape,
96 verifier and 28 VM-FFI checks, plus selected owned assertion, multi-borrowed
and multi-consuming authority/allocation gates. I retain
`/tmp/nanolang-owned-value-graphs-final-integrated.log`. The final pinned Clang
suite also passes in3.412 seconds, recorded in
`/tmp/nanolang-owned-value-graphs-final-clang.log`.

I rebuild every instrumented VM/NanoISA object and driver on this integrated
source, preserving incoming VM arithmetic flags. The unchanged complete corpus
again passes all ten cases/all four APIs/two repeats in230.119 seconds, with
at most two processes and a300-second limit per case. No sanitizer findings
occur. My exact executable SHA256 is
`8ee333125940e4be7099a70111bb4ceb77d11494830c7a78591a70f151fb012e`.
I preserve the build under `/tmp/nanolang-owned-final-sanitizers-build`, all
per-case logs/status and summary under `/tmp/nanolang-owned-final-sanitizer-parts`,
and runner/build output in `/tmp/nanolang-owned-final-sanitizers.log`.

This is the final bounded runtime acceptance. Documentation-only publication
follows this source pin. The unchanged affine example, source admission,
owned/void results and string/PRINT prerequisites remain open. I reconcile only
the three bounded tasks after canonical merge; I make no release claim.
