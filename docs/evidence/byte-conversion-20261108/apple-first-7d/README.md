# My Apple checker deadline failure

Run [42497933793](https://github.com/jordanhubbard/nanolang/actions/runs/42497933793) at `7d020ab46f370ac51d1a52716048f8c41085e4f4` is terminal with failure. Worker headroom, the pinned Apple toolchain and original C lexer/parser/typechecker prerequisites pass. Checker creation stops at the unchanged ten-second dependency-shadow deadline, before publishing any checker product or executing the operand suite.

The existing trace records 687 completed shadows, zero assertion failures and 9.905652 seconds of completed body time. It then starts `check_array_sort_contracts` without recording completion. The byte-conversion and main entry shadows remain unexecuted. The body sum excludes selection/setup/reporting and the interrupted body's elapsed time; I do not invent completion or call this infrastructure.

The independent original translator gate passes all 2458 assertions and both six-method array alias routes pass. Early artifact 13624686173 and final artifact 13624895693 pass CRC and full file-hash verification. I retain eight early texts and 25 final files (24 exact texts), plus the manifest-verified empty cleaned translator TAR. These artifacts contain no accepted native checker product and do not qualify Apple byte conversion.

Under #1065 I retain this prefix, a discarded local lookup experiment and an actual current-C-seed CPU sample before selecting an optimization. I preserve all assertions, the ten-second deadline and the original platform/toolchain profile. Linux component qualification, fresh-stage results and final release acceptance remain separate.
