# I preserve unadmitted shell-fragment compilation

The original unknown-fragment method independently reproduces the failure at
clean f0f6a0c62: parameter expansion is rejected before compilation. My
compile-only linker filter asks the literal-word parser to interpret every
fragment. I now preflight that parser and preserve the original compiler
transport for nonliteral fragments. Literal sequences still filter linker
operands and preserve paired option arguments. Capture admission, refusal
after an admitted capture fails, and production deadlines remain unchanged.

The original fallback method, literal linker-filter control and new expansion
method pass: 3 methods in 5.223 seconds. The new method checks default and
explicit macro values (42 and 43) in compiled code and no captured snapshot
units. Its first version incorrectly required fingerprint metadata to be
absent; legacy fallback permits that metadata. I retain that test-assumption
failure and corrected qualification.

The separate full-run external assembler deadline failure remains under
investigation. These focused checks do not close the complete source-snapshot
corpus, Linux qualification or final 5.1 acceptance. I track both failures in
[#980](https://github.com/jordanhubbard/nanolang/issues/980).

My complete `tests.test_link_response_query` and
`tests.test_link_response_graph` run terminates with exit zero; its full log
is retained here.
