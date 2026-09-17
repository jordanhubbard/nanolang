# My scalar expression types

I infer types for supported literals, locals, flat-record fields, declared
function results, emitted string/bool builtins, lengths and binary expressions.
I now lower projected string comparisons and concatenated `str_substring` or
`int_to_string` results. Bool equality uses generic `EQ`/`NE`, matching the C
seed; I retain rejection of mixed scalar equality.

`make -j8 test-nanoisa-src-nano` passes 86 baseline checks and six focused
Python cases on Linux ARM64. The new positive case compares ten named
module/function checks and verifies/executes both C-seed and self-hosted
modules in NanoVM and through strict C11 AOT. Four mixed scalar expressions
are refused without output. Existing flat-record, void, nested-record refusal
and invalid-return cases remain green. The two previously pending expression
probes are now covered by `scalar_expression_types.nano`, so I removed the
obsolete pending files.

I reran emission of `src_nano/nanoc_v06.nano`. Its
`c_source_output_path` concatenation now passes. The first rejection is
`statement outside the pinned subset`, reached at a `break` in `parse_options`.
The loop-control task is recorded in my roadmap; full compiler emission and
canonical bytecode bootstrap equality remain open.

Task `task_8e367aeda3394b0bb1ed1c37f56edeef` records this slice. The earlier
flat-record evidence describes the historical boundary before this change.
