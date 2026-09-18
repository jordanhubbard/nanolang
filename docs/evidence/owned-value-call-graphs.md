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
its test statements. My first deeper stack fault fixture set capacity64,
which forced growth during preparation of two arguments before the intended
callee preflight. I corrected capacity to66: four sixteen-local frames plus
two arguments fit, while the fifth frame does not. The corrected fixture
passes338 checks. I preserve the initial assertion log at
`/tmp/nanolang-owned-value-graph-preflight.log` and the corrected run at
`/tmp/nanolang-owned-value-graph-preflight-corrected.log`; I do not attribute
that test setup mistake to runtime behavior. My initial combined gate also
found the old verifier diagnostic assertion still expecting an entry-to-helper
message. It now requires the new checked-acyclic-call diagnostic; the same
refusal remains required.

My current ordinary full run is
`/tmp/nanolang-owned-value-graph-full.log`. The new paired gate passes1847
checks plus338 preflight checks. Each of ten native cases uses strict C
warnings, ASan/UBSan/LSan, repeated invocation and injected owner allocation
failure with no retained native roots. Existing single/multiple consuming,
borrowed, helper-local, assertion, affine analysis, verifier and VM gates are
being qualified together. My separately instrumented VM/NanoISA run is
`/tmp/nanolang-owned-value-graph-sanitizers.log`; its result remains pending.

I do not claim source admission, owned/void results, string/PRINT effects or
restoration of the example from this runtime prerequisite. I do not replay
its frozen failure artifact. The parent release blocker remains open.
