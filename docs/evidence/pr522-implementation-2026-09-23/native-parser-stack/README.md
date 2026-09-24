# Generated native parser stack measurements

At `e08f2c5fe`, I retain a current C-seed compiler module and its generated standalone C, compile it at strict `-O0` with ASan/UBSan, frame pointers and `-fstack-usage`, then link the fresh instrumented host runtime. Running that compiler on hello reproduces the stack overflow. Source/object hashes and build commands identify the private artifacts.

The compiler reports 2,726,432 stack bytes for `nl_parse_primary` and 717,280 for `nl_parse_expression_recursive`. Record locals and explicit record temporaries are already heap-backed; generated internal calls still pass and return the full record struct by value. This module's record representation has 77 field slots.

My private generated-C prototype changes internal record parameters to const pointers and record results to output pointers. Each callee copies its input into its own existing record-local storage; it writes the return snapshot after normal cleanup. The script asserts the expected generated call shapes and changes no production translator source.

With the same strict compiler and instrumentation flags, measured frames drop to 1,069,440 bytes and 420,288 bytes respectively. The resulting instrumented compiler compiles hello and its output executes successfully, with leak and stack-use-after-return detection retained. No stack limit or deadline is raised. This is a diagnostic prototype, not production qualification; helper calls still pass records by value and further frame reduction may be needed.

I require production lowering, value/alias lifetime controls, indirect/tail-call handling and both complete compiler paths before closing `task_ec9e85c7c9ec4521bde658929f8a5e2f`. The full prior One IR terminal remains in `../instrumented-host-runtime`.
