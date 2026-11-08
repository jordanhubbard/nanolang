# My first format implementation unit refusal

I retain failed [run 42470662957](https://github.com/jordanhubbard/nanolang/actions/runs/42470662957) at `45b40552b5f24683a8bfc862edff0365a48bb0c5`. Both original C typechecker unit suites fail at the first positive `tc_format_template` assertion, line 106, for the unchanged variadic string/int example. The focused format and other operand methods are skipped after that prerequisite failure; this pin establishes no new format runtime acceptance.

My new declared-function guard incorrectly required absent parameter metadata for a synthetic builtin. The actual `env_get_function_at` cache retains format's fixed STRING parameter. Commit `3b73f2552` removes only that incorrect parameter premise, distinguishing executable bodies and extern declarations instead. Every original unit assertion remains. Its corrected runtime results are a separate checkpoint.

Both jobs still pass the independently selected complete translator gate: 2,443 execution, 3,866 shape and 464 callable checks, zero failures, with its original adjacent suites. Those passes do not override the failed prerequisite or skipped source suite.

Both ZIPs pass CRC and have complete uploaded-file SHA-256 inventories and deterministic exact-text bundles. Both embedded translator TAR hashes and inventories independently verify, with empty inventories after normal successful cleanup. Uploaded binary bytes remain private; this does not retain every cleaned native product or establish filesystem modes or symlinks through direct ZIP custody.
