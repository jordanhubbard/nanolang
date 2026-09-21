# My PR939 hosted CI failures

My independent VM teardown gate passed on Linux and Darwin, and PR939 merged.
Broader hosted CI is not green. I retain the downloaded job logs verbatim in
lossless gzip files, with original and compressed hashes in checks.json.

Linux x64 and ARM64 build jobs each reach176 selected verifier-corpus sources,
174 verified,2 failed and0 skipped. Both name tests/test_u8_basic.nano and
tests/token_value_bytes.nano. The detailed compile logs are only named under
worker /tmp/nano-verify-corpus paths; these downloaded logs do not contain their
contents. Existing tasks c6b2a040c1434fc784a9d46c02a4981e and
7b805000dfda4da386b55d4691e8c647 keep those source requirements open. I do not
infer that this new occurrence has the same cause merely from its filenames.

Coverage instead fails three ArtifactImports methods while linking native
products against instrumented bin/nano_aot_runtime.o: unresolved __gcov_init,
__gcov_exit and __gcov_merge_add. The fixture hardcodes native link arguments
without the selected coverage link flags. Task_b9e2f08776a64a67879453d7d6e6052d
requires the compatible compiler/link closure and unchanged semantic controls.

The CI failure-log step reads .test_output/*.compile.log, while the verifier
corpus retains its diagnostics in a different temporary tree. Task_13ddb5e14be94d4fba5cf4e4ee4c6c0f
tracks lossless hosted failure-artifact retention. Neither issue is labeled an
infrastructure failure. Full5.1 integration and publication remain open.
