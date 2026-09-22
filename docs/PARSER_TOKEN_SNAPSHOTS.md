# My parser token snapshot contract

I measured the unchanged candidate cce540 parser_driver with the retained instrumented Cseed:336 shadows complete in45.903seconds under the original60-second selection deadline. Two original520-statement block controls take34.977seconds. I retain that measurement at9ca6cf6db; the original full typecheck preparation deadline remains failed.

Before implementation under task_f78ce9cfafe14a89b785a67d04afe6e0, I select this narrow change:

- I fetch at most one current token per parser_current invocation. I preserve the exact zero-valued EOF sentinel when position reaches token_count or the stored token is EOF. Real tokens keep all five fields. The public helper signature stays unchanged.
- I retain one current-token snapshot within each block-loop iteration and parse_statement invocation. No parser mutation or user callback occurs between the reads I remove. I invoke owned-pattern recognition only for an actual LET token, the same first condition checked inside that helper.
- I replace only the block loops' trivial parser_has_error calls with their identical has_error field reads. I retain all parser-state transitions, list appends, node identity, error propagation and declaration dispatch.
- I preserve every byte of both520-statement shadow bodies, including shaped declaration and incomplete-input assertions. I add current-token empty/end/embedded-EOF/real-token controls and actual statement dispatch/refusal controls.

My block loops are already iterative. Native list reads are constant-time and pushes grow geometrically. The evaluator deliberately copies record parameters and snapshots returned records; Parser has77 fields. I remove redundant pure calls without changing those ownership rules, adding a global cache, retaining a token across parser advancement, changing input sizes or raising any deadline.

I require independent source review before corrected execution. Qualification first checks the unchanged parser component plus additive shadows with exact retained providers and timing attribution. Full candidate bootstrap and CI remain separate acceptance obligations; I will claim a speedup only from actual corrected measurement.

## My required-expression EOF correction

Before correcting task_03c26b0bc49d4dbc9b193a83fb51d11d, I retain the corrected-staging1ee terminal:53.504seconds, no timeout, all336 original shadow entries completed. Only my additive malformed statement assertion fails. The source `assert` calls parse_expression_recursive at EOF with left_parsed=false; its existing EOF branch returns an unchanged non-error parser, allowing a missing condition to reach parser_store_assert. I do not remove the new assertion.

I audited generic-expression callers. Optional return checks EOF and closing brace before calling expression parsing. Optional match guard checks EOF and absence of IF before parsing its required condition. Empty call, tuple and array forms branch before requesting an element expression. Let/set/assert, conditions, iterator sources, contracts, field initializers and arguments require an expression. The wrapper always starts with left_parsed=false; recursive calls use true after parsing an actual operand. The separate historical nanoc_integrated source owns a different parser implementation and is unchanged.

I change only the EOF branch: preserve the current parser after a parsed left operand; otherwise return parser_with_error. Existing incoming errors remain unchanged. Additive controls cover an empty required expression, completed scalar expression at EOF, ordinary bare return at EOF/closing brace and absent match guard. The original malformed statement assertion and both520-statement controls remain byte-exact. Source review precedes execution.

The failed run separately measures the original block and unsafe-block shadows at13.884 and14.042seconds, both with zero assertion failures, versus17.424 and17.552seconds in the earlier unchanged sample. Total shadow bodies are37.831seconds versus45.894. This comparison does not make the failing fixture or full CI preparation pass.
