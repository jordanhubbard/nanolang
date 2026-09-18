# My bounded owned value-call graph evidence

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

My current ordinary full run is
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
aggregate remains pending. I do not relabel either full-run timeout as a pass. Both builds instrument the test driver and every linked NanoVM and
NanoISA object, with ordinary compiler/runtime support objects linked normally.
I do not claim that the entire legacy compiler is instrumented.

I do not claim source admission, owned/void results, string/PRINT effects or
restoration of the example from this runtime prerequisite. I do not replay
its frozen failure artifact. The parent release blocker remains open.
