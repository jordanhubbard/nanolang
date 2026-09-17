# My element access result types

On 2026-09-16 I added element-result inference for `at`, `array_get` and
typed-list getters. I use the container's declared type, retaining string,
integer and supported record results through comparisons, concatenation and
field access. Scalar array literals, array-valued function calls and list
constructors supply the same expression type information.

I check two-argument access, an integer index and a supported container. A typed
list accessor must match its element type. I reject mismatches before output.

`make -j8 test-nanoisa-src-nano` passes 86 checks and 23 integration methods.
The added fixture contributes 10 C-seed bytecode comparisons and runs under
NanoVM and strict C11 AOT. It covers direct string array/list results, literal
arrays, array-returning calls, integer arithmetic and a record-list getter
followed by field projection. Four malformed access programs refuse output.

The full compiler boundary measured after global initialization remains
array-bearing `MergeResult`; this expression repair does not close full
compiler emission or the matching bytecode bootstrap gate.
