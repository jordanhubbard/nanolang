# Generated native parser stack measurements

At `e08f2c5fe`, I retain a current C-seed compiler module and its generated standalone C, compile it at strict `-O0` with ASan/UBSan, frame pointers and `-fstack-usage`, then link the fresh instrumented host runtime. Running that compiler on hello reproduces the stack overflow. Source/object hashes and build commands identify the private artifacts.

The compiler reports 2,726,432 stack bytes for `nl_parse_primary` and 717,280 for `nl_parse_expression_recursive`. Record locals and explicit record temporaries are already heap-backed; generated internal calls still pass and return the full record struct by value. This module's record representation has 77 field slots.

My private generated-C prototype changes internal record parameters to const pointers and record results to output pointers. Each callee copies its input into its own existing record-local storage; it writes the return snapshot after normal cleanup. The script asserts the expected generated call shapes and changes no production translator source.

With the same strict compiler and instrumentation flags, measured frames drop to 1,069,440 bytes and 420,288 bytes respectively. The resulting instrumented compiler compiles hello and its output executes successfully, with leak and stack-use-after-return detection retained. No stack limit or deadline is raised. This is a diagnostic prototype, not production qualification; helper calls still pass records by value and further frame reduction may be needed.

I require production lowering, value/alias lifetime controls, indirect/tail-call handling and both complete compiler paths before closing `task_ec9e85c7c9ec4521bde658929f8a5e2f`. The full prior One IR terminal remains in `../instrumented-host-runtime`.

## Production lowering

I now emit const-pointer record arguments and output-pointer record results for internal generated calls. Each callee retains its own input copy and return snapshot. Direct calls, resolved aggregate callbacks, scalar indirect calls with record arguments, and cross-function tail calls use this convention; self-tail restarts keep their existing snapshots. Artifact exports and raw NanoISA semantics retain their contracts.

My production generated compiler measures 1,069,440 bytes for `nl_parse_primary` and 420,288 for `nl_parse_expression_recursive`, matching the prototype under the same strict O0 ASan/UBSan frame-usage command. `compiler-production.su.gz` and `production-hashes.txt` identify this measurement. These frames remain large; this is measured reduction, not a general bound on recursive input depth.

The unchanged 31-method ordinary One IR suite passes in 382.418 seconds. My new regression makes 600 calls through a 75-field record, checks each returned field, and runs with a 2 MiB stack limit. The parent translator fails this same fixture with SIGSEGV; the production translator passes ordinarily and with ASan/UBSan, leak and stack-use-after-return checks. `qualify_call_bound.py` retains the matched comparison procedure. My existing translator assertions also pass (2,558).

I update handwritten C probes only to call the new internal signatures. Their value, storage-tag, alias, cleanup and allocation-failure assertions remain intact. The complete instrumented suite includes the new regression and retains all existing methods and deadlines.

The complete instrumented One IR run passes all 32 methods in 417.283 seconds, including both complete compiler paths and the real std artifact. I retain its terminal in `production-one-ir-instrumented.log`; the runner and instrumented host-runtime build provenance remain in `../instrumented-host-runtime`. Final hosted platform qualification remains separate.
