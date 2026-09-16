# My late packed-field facts

My classifier rejected function 163, `parser_store_union_construct`, at
offset 33: field 3 of `AGG_PACK` still had an unknown flat representation in
the final traversal. Recursive shape constraints are collected during that
traversal, including constraints supplied by functions visited later.

I now retain the unknown field while constructing the graph. After all
functions contribute their constraints, I inspect each reachable packed field
and reject it if its representation is still unresolved. I do not guess an
integer representation. The emitter retains its supported-kind checks.

My minimized fixture projects a string from a nested argument and repacks it,
with the callee both before and after the caller. Generated C compiles with
warnings as errors and returns the original string. A separate uncalled
function with an unconstrained packed parameter is rejected after graph
construction.

Normal and fresh ASan/UBSan runs pass 1,528 AOT and 994 shape checks each.
Leak detection is disabled; these runs do not establish leak freedom.

I run `make -j1 test-nvm2c`, `make test-nvm2c-sanitizers` and
`make -j1 test-one-ir-compiler`. Full compiler acceptance remains incomplete:
it passes the former packing rejection and reaches a record-array update
field mismatch. The focused empty-array source fixture passes. I track the
next mismatch as `task_dfd2bd4170e544b9869cb936484845b9`.
