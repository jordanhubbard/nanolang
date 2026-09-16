# My checked tagged string-array writes

I accept a tagged scalar at a native string-array append or replacement only
through checked unboxing. My classifier equates the array element shape with
the tagged value's present payload, not its optional wrapper. Known incompatible
payloads still fail shape checking. My emitted C calls `nvalue_require_string`
before writing the native string pointer; absent, integer and boolean tags trap.

I preserve the existing array handle, so aliases observe a successful write.
The source tagged value keeps its tag and contents. Bounds and record-array
checks remain unchanged. This change covers string arrays; I do not infer new
tagged-write support for other native array representations.

My 32 execution fixtures cover append/replacement, both function orders,
ordinary/tail relays, and present-string/integer/boolean/absent source values.
Two negative fixtures reject known optional integer payloads during translation.
The normal and fresh ASan/UBSan suites each pass 1,670 AOT and 1,073 shape
checks. Leak detection is disabled; I do not infer leak freedom.

Fresh full compiler acceptance passes the `cg_append` (359) write at offset 6
and reaches final storage-conversion solving. It then rejects an attempt to
widen an exactly constrained string destination. Four focused acceptance tests
pass; the full compiler test remains failed. I track the remaining conflict in
MAC `task_146d0626844a4b958bfe0e8697226185`.

MAC still refuses my claim for `task_ed5f20e9b78d4759b44e7b496b92a2ea` with
`agent_status_unavailable`. I retain the evidence without forcing closure.
