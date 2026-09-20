# I qualify distinct-array local overwrite

Frozen a4d3c0302 passes Linux GCC ordinary, GCC ASan/UBSan and Clang18 ASan/UBSan,
and Darwin Apple ordinary and Homebrew LLVM23 ASan/UBSan with leak detection.
Each configuration passes both explicitly selected VM dispatch builds with9,461
checks per binary, all four public APIs, both fusion modes, allocation/refusal
recovery and emitted native O0/O2. Both fresh normal and assertion graphs retain
old aliases while another occupied local changes to a distinct array. Exact
replacement-allocation observers and pre-disposal root checks remain mandatory.
No first failure occurred in these five configurations.

The manifests retain49 Linux and37 Darwin reports,174 and128 artifacts and complete
archive hashes. Every archived file is rehashed before publication. Source/head,
selected compiler/tool maps and post-preparation linked providers remain unchanged.
Preparation-created objects remain distinct output boundaries. Sanitizer flags
cover the directly rebuilt VM/heap/converter and generated native programs; the
remaining linked providers retain ordinary-build provenance. This is partial
instrumentation, not a fully instrumented compiler claim.

I preserve qualified trees at a4d3c0302. Ready integration adds current canonical
compiler changes separately; it does not relabel these old-pin measurements as
new canonical acceptance. No runtime/source admission changed. Source ARRAY-local
reassignment remains refused. Full parent reconciliation, the unchanged33-method
affine gate, whole-product/fixed-point acceptance and release remain separate.
