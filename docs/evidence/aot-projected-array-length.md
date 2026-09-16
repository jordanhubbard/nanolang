# My projected array lengths

`parser_get_identifier_count` projects record field 10 and reads its length.
My classifier previously recorded neither an array constraint nor its element
representation for that unresolved field. My emitter then fell through to an
integer read and refused the later `ARR_LEN`.

I now record that an unresolved operand consumed by `ARR_LEN` is an array,
without guessing its element type. When a record projection has that array
shape but no element storage, I check the field index and runtime storage tag,
then retain its array handle in my existing tagged-value representation.
Length dispatches on integer, boolean, string or record-array storage.

I do not copy or retag the source array. My harness changes the shared handle's
length after the first read and verifies that the next read observes it while
the record field keeps its storage tag. Empty arrays are valid. Invalid field
indices, non-array tags and null handles trap on this dynamic path.

Adding tagged record-array length does not add tagged record-array element
access. My tagged get/set/print helpers explicitly reject that storage instead
of treating its pointer as a string array; tagged push retains its refusal.
Typed native record-array operations are unchanged.

Sixteen initial cases failed translation before my change. Twenty final cases
cover both function orders, all four array storage kinds, empty arrays, shared
handles, invalid fields/tags/pointers, and refusal of tagged record-array reads
and writes. They execute a C harness against the generated functions without
providing caller-derived element facts.

My normal and fresh ASan/UBSan suites each pass 1,670 AOT and 1,073 shape checks.
Leak detection is disabled; I do not infer leak freedom. Fresh compiler
acceptance passes ten focused test methods but still fails the full compiler.
Debugger inspection places its next `AGG_GET` representation failure in
`typecheck_parser` (339). I track that as
MAC `task_f682c0c61d354890a80c28b3ed32918f`; this is not release acceptance.

MAC refuses my claim for `task_9a2a87fb80d94386b096a7b01ec7eaaf` with
`agent_status_unavailable`. I retain evidence without forcing closure.
