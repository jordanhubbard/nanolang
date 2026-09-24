# My repaired-compiler integration of the private generated consumers

I apply the source, test and runtime-packaging delta from PR945
`57542f703b16f92951e54d43962e68035d00fca1` to an isolated checkout of
`b34fbab63`. I preserve the repaired compiler and both sets of Make targets.
All 21 recorded incoming source/build files still match their pre-run hashes
and the copies integrated into the repair branch. The subsequent C-seed
nominal-name correction is outside this unchanged private-consumer delta.
I have not merged or closed PR945.

On Darwin ARM64 with selected Homebrew Clang, the complete
`make -j2 test-record-array-generated` gate passes both methods in 272.658
seconds. It retains all 93 emission recipes, 256 decisions, 98 emission
allocation positions and all 73 unchanged generated products through O0/O2
linked and observed configurations. Fault injection and independent recovery
remain enabled. This ordinary integration does not substitute for the incoming
PR's separately pinned seven-configuration sanitizer evidence.

The complete `make -j2 test-record-array-llvm` gate passes all four methods
in 762.760 seconds. It checks factored C bytes against the captured C corpus,
emission boundaries, native LLVM and Wasm through Node and Wasmtime at both
optimizations, with linked/observed products and unchanged fault coverage.
The explicit tool selections and each child command are retained in reports.
Both original managed-runtime package methods then pass with Homebrew Clang,
ASan/UBSan, leak and use-after-return checks for their instrumented native
lifecycle control, real native/Wasm links and byte-reproducible packages.

I archive 6,656 generated-C reports and 25,411 LLVM/Wasm reports, including
all command statuses and outputs. Every archived report is rehashed against
the inventory after archive creation. The inventories also hash every actual
product and object retained at the local paths in `manifest.json`; those
binaries remain local rather than being duplicated in Git. The existing
240-second child bounds and every original assertion remain unchanged.

Fresh Linux integration and affected query/VM neighbors remain required.
Installed publication of the complete generated consumer, public selection,
paired source/bootstrap and full union/nested/cyclic/indirect graph coverage
remain required parent work. This private patch is not release admission or
full LLVM/Wasm completion. The declared module-artifact and callback-binding
repairs also remain open on the canonical compiler path.
