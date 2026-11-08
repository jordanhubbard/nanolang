# My terminal 9c original-route XML checkpoint

Run [42492701660](https://github.com/jordanhubbard/nanolang/actions/runs/42492701660) at `9c6312a3d6ab42e65b43d34385a34052a38c8e42` is terminal with failure in the complete original corpus step. Both original seeds pass 237/237. Both fresh stages pass 207/237, with the same 30 failures and no skips. All four ordered selections match each other and the preceding f16 checkpoint exactly.

`comparison.json` verifies that test_stdlib_arrays is the only cleared failure from f16 in both stages; no new failure appears. `route-results.json` retains every selected name, class, status and complete failure diagnostic. The whole original array fixture now passes through both fresh stages; later byte/environment work is qualified separately. This does not establish a passing complete corpus or the original budget acceptance.

`exact-route-xml.json.gz` retains the exact four XML files with their individual sizes and SHA-256 hashes. I inspect the entire hosted 51308067-byte ZIP in memory and verify its complete CRC and SHA-256 `372f50559e28427dcb2533acb081419c6925e9e2c7c82141fff3bea877eb3dc6`. Artifact 13626800155 contains 8450 entries. Local disk pressure prevents retaining a further full ZIP copy here; this receipt retains only route XML and does not independently verify actual product bytes, other text entries or embedded TAR manifests.

The earlier CLI step passes exact 2/0/1/2 outcomes through all four providers, with 22 markers per fresh stage, as retained in ../../checker-phase-profile-20261108/9c-cli. Actual full-artifact product custody, later sources, integrated platform/full-route/native SDK and release acceptance remain open.
