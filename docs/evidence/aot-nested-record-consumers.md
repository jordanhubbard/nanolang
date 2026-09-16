# My nested record consumers

My `AGG_GET` classifier marked its operand's local origin as a record but did
not constrain the operand's shape. A nested projection has no local origin.
Its output could therefore retain an unknown representation even when the next
instruction required a record, and native emission fell back to an integer
field read before refusing the second projection.

I now require the consumed shape itself to be a record. This does not infer
the nested fields' types. Their producers and consumers supply those facts.
My existing field bounds, storage-tag and non-null nested-record checks remain
in the emitted C.

All sixteen reduced cases failed translation before the change. Six positive
cases now execute nested integer, boolean and string reads in both function
orders. Ten negative cases trap on outer/inner tag mismatches, null nested
records, and outer/inner bounds failures. The native harness deliberately
provides no caller-derived field facts to the classifier.

My normal and fresh ASan/UBSan suites each pass 1,670 AOT and 1,073 shape
checks. Leak detection is disabled; I do not infer leak freedom.

Fresh compiler acceptance passes eleven focused test methods. It passes the
previous `typecheck_parser` (339) projection failure, then stops in
`generate_enum_definitions_from_tokens` (394): `STORE_LOCAL` expects a string
but receives another emitted representation. I track that as
MAC `task_34d2d9345d584ff78fab59261eb112d6`. Full compiler and release acceptance
remain incomplete.

MAC refuses my claim for `task_f682c0c61d354890a80c28b3ed32918f` with
`agent_status_unavailable`. I retain evidence without forcing closure.
