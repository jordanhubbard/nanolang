# My Linux match qualification

I pass all 122 evaluator cases and all eight C-seed match-totality tests at `7a21096f86c3a51f83e38672484e694b90c5aa72`. My production code is unchanged from PR #855 at `d0fbd9b1bda184d54563599f2c6f23b41e6ed66d`. I compare every tracked source, 155 retained providers, six tool identities, and HEAD before and after the final run; all remain unchanged.

I retain each first terminal in its own directory. My first run stopped because the unchecked-match fixture did not register its function before bypassing the typechecker. My corrected run passed that backstop, then stopped at a stale `reduce` argument order. After correcting that order and executing its existing shadow, my next run stopped because my runner used a checkout without the relative FFI test library. My final run uses the original provider checkout as the evaluator working directory. I preserve all assertions and use a fresh fixture binary after each correction.

My repeated wildcard controls check lexical guard order, selected body, and once-only scrutinee evaluation. My terminal backstop registers only its unchecked function, drains stderr to EOF with EINTR handling, checks normal failure status, and checks the diagnostic. My handler fixture retains its required fallback and original result assertions.

The manifest seals copied reports, maps, runner sources, and raw logs. The retained binaries remain at the paths in the binary maps. Raw failure logs preserve their original formatting. I do not claim Darwin qualification, a fresh current bootstrap, all compiler routes, or release readiness from these results.
