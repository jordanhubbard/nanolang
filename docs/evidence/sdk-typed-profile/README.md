# I qualify the private typed declaration profile

I qualify frozen `8954a54c6538c0ac750392866790e5a2c6cf0477` on Linux GCC and Darwin Apple Clang ordinary builds, then Linux GCC and Darwin Homebrew Clang ASan/UBSan/LSan builds. Each lane rebuilds its providers and runs the original declaration fixture plus typed revision2 controls.

Each lane passes 2,189 linked and 25,493 allocation-instrumented mixed declaration checks, alongside the original ordinary-array checks (464 linked and 2,398 instrumented). Controls retain exact signatures, matching layout kinds, mixed storage graph cycles/refusals, bounded unresolved provider references, original revision2 refusals, depth64 success/depth65 refusal, independent copied output, and every measured allocation prefix in both failure modes with fresh recovery. Budget preservation compares fields explicitly, never padding.

My 124 reports retain source/provider/tool endpoint equality, exact commands and flags, all inner leader/group cleanup and successful outer terminals. `seal.json` records member sizes and SHA256 values; `scope.json` records frozen Git blobs; `run.py` is the actual driver. Original 240-second subprocess and 1,200-second preparation bounds remain unchanged. No first product failure occurred in this gate.

This private profile copies numeric declarations only. I have not resolved provider-dependent facts, enabled any executable reader, or completed generated adapters, lifetime validation or installed SDK acceptance.
