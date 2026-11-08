# My first c5 source-emitter attempts

I retain early metadata from Linux run 42515795661 and Apple run 42515796795 at c5bcf5899. Both Linux jobs reach the new source-emitter gate after prior component checks, then fail before building that emitter: my Python harness invokes rg, which is not installed on the hosted worker. This is a harness dependency error, not a demonstrated source lowering failure. I replace the subprocess inventory with Python filesystem traversal and require actual corrected execution.

Apple stops the checker at the unchanged ten-second shadow deadline before the source-emitter gate runs. I observe 695 starts and 694 completions with no assertion failure and 8.446655 summed body seconds. My first recorder missed the format_error_elm_style completion because its diagnostic text shares the event line; I correct tagged-event extraction rather than treating that interleaved completion as absent. My summary retains the exact selection/count and summed body durations. I do not classify the deadline as infrastructure or infer its root cause from successful older pins.

I CRC-check the early artifacts in RAM and retain all exact text, inventories and ZIP digests. The original ZIPs are not saved; I do not claim native product custody or complete job acceptance from these early receipts. Both live jobs continue their always-run translator/alias gates.
