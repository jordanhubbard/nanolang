# My native artifact link qualification

I qualify source883dac9d5 with the ten unchanged ArtifactImports methods in
seven configurations: Linux ordinary GCC, coverage GCC, ASan/UBSan GCC and
Clang; Darwin ordinary Apple Clang, coverage Apple Clang, and ASan/UBSan
Homebrew Clang. All70 method executions pass. Native sanitizer runs enable
leak detection and halt on undefined behavior. Ordinary source producers remain
separately pinned through NANO_CC; the fixture selects its native compiler
through NANO_NATIVE_TEST_CC or CC and retains LDFLAGS.

Both fresh hosts then pass the full test-nvm2c target:2422 structured-C checks,
1365 shape checks and existing opcode/sanitizer-driver controls. Darwin uses
Homebrew Clang with the installed SDK explicitly. Existing argc-only, argv-only,
combined argument boundaries and generated-main wrappers remain unchanged.
All2695 selected source inputs match before/after on both hosts. The AOT runtime
is rebuilt separately for each instrumentation configuration; its objects and
actual native link commands are retained. These are fixture/provider checks,
not a claim that every source producer was itself sanitizer-instrumented.

My first6fda fixture patch passes Linux ordinary, coverage and GCC sanitizer.
Multiword CC in Linux Clang and Darwin first attempts prevents the existing
module cache from establishing tool identity and produces distinct generation
paths. I preserve those failures, then pin the ordinary producer compiler
separately without weakening import identity assertions. Darwin ordinary and
coverage then pass; Apple sanitizer explicitly refuses leak detection before
program execution. Homebrew retains leak checks and exposes strict unused
argument-global diagnostics in all five generated native units.

My870 emission correction passes all seven artifact configurations but changes
an imported generated-main signature. The full Linux backend gate retains2421
passes and one wrapper compile failure. Corrected883 preserves that signature,
discards unused parameters explicitly, and emits each argument global/assignment
only for a consuming host helper. The fresh complete gates above then pass.

I retain5036 reports losslessly in nine verified tar/gzip bundles. checks.json
records each bundle/member, stored hash and original bytes/hash (large members
may themselves be gzip-compressed). Its path field is the logical report path,
not a separate Git file. The3051 captured products occupy1324516089 bytes in
persistent local qualification storage, indexed separately; they are not
vendored here. Temporary fixture trees were copied before cleanup. This does
not claim a complete archive of external module caches or deleted backend-test
executables. The first Linux input-map delta is its deliberately later roadmap
note; final883 source maps are exact. Original hosted failures remain in the
separate PR939 CI evidence.

Tasks b9e2f08776a64a67879453d7d6e6052d andbd50601a87444d93aabf751f9f33b963
await canonical integration. Full5.1 compiler/bootstrap/backend/SDK/release
qualification remains open.

After merging current main (the independently qualified empty-append guard and
CI log retention), fresh9576 ordinary providers and all ten original artifact
methods pass on Linux and Darwin. Both2695-input maps remain exact. I retain
these separate [integration reports](integration/checks.json); earlier sanitizer
qualification remains attributed to883, not relabeled as integrated evidence.
