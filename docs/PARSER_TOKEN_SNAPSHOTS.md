# My parser token snapshot contract

I measured the unchanged candidate cce540 parser_driver with the retained instrumented Cseed:336 shadows complete in45.903seconds under the original60-second selection deadline. Two original520-statement block controls take34.977seconds. I retain that measurement at9ca6cf6db; the original full typecheck preparation deadline remains failed.

Before implementation under task_f78ce9cfafe14a89b785a67d04afe6e0, I select this narrow change:

- I fetch at most one current token per parser_current invocation. I preserve the exact zero-valued EOF sentinel when position reaches token_count or the stored token is EOF. Real tokens keep all five fields. The public helper signature stays unchanged.
- I retain one current-token snapshot within each block-loop iteration and parse_statement invocation. No parser mutation or user callback occurs between the reads I remove. I invoke owned-pattern recognition only for an actual LET token, the same first condition checked inside that helper.
- I replace only the block loops' trivial parser_has_error calls with their identical has_error field reads. I retain all parser-state transitions, list appends, node identity, error propagation and declaration dispatch.
- I preserve every byte of both520-statement shadow bodies, including shaped declaration and incomplete-input assertions. I add current-token empty/end/embedded-EOF/real-token controls and actual statement dispatch/refusal controls.

My block loops are already iterative. Native list reads are constant-time and pushes grow geometrically. The evaluator deliberately copies record parameters and snapshots returned records; Parser has77 fields. I remove redundant pure calls without changing those ownership rules, adding a global cache, retaining a token across parser advancement, changing input sizes or raising any deadline.

I require independent source review before corrected execution. Qualification first checks the unchanged parser component plus additive shadows with exact retained providers and timing attribution. Full candidate bootstrap and CI remain separate acceptance obligations; I will claim a speedup only from actual corrected measurement.
