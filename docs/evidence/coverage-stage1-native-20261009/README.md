# My bootstrap coverage-link correction

Coverage job113771001335 from CI run37915636041 uses head `191a825b8` (tested merge `146b6db293cbfbae0f03604c88123440bcff9130`). The corrected five-method emitter component gate passes in 117.136 seconds. The subsequent bootstrap emits and verifies Stage1 and translates it to C, then native linking exits1 after 48.677 seconds.

I recover artifact11610685540 directly from its upload ID and verify its zip checksum against the upload log. `hosted-native.log` reports undefined `__gcov_init`, `__gcov_exit` and `__gcov_merge_add` in the covered runtime object. `hosted-manifest.json` records empty linker flags; the native command omits the coverage flags even though Make exports them through `LDFLAGS`.

I reproduce the configuration defect independently on Darwin with a real coverage-instrumented C object. My regression calls the real Stage1 configuration and native command, while isolating translation and compiler-smoke setup. The unchanged implementation fails with missing LLVM coverage-runtime symbols. I now fall back to effective `LDFLAGS` when `NANO_LDFLAGS` is absent or empty, preserving a nonempty explicit override. Both inherited-flag and explicit-override native products link and execute their runtime assertion. All six bootstrap-boundary/native-guard tests pass in 1.045 seconds.

I do not claim a fresh hosted coverage pass from this focused check. The full Darwin bootstrap receipt in `../header-tokens-20261009/qualified-bootstrap` pins `e1fa92e03` before this script change; I preserve that boundary. Final candidate bootstrap and hosted/platform qualification remain under #982/#976.
