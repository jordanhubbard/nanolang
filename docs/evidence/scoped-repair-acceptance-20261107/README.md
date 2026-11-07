# Scoped repair acceptance

I verify that every selected source file from my qualified #1045/#1073 version/hybrid repair (`35dca0dc02f6fd27320cd610295d924b7c7c2133`) and #1078 path/fixture repair (`d9d553576`) has identical bytes in my current branch and integration `435dae34cc2ce74d5210806d26ef1ebb9f9e4b7a`. My [identity comparison](source-identities.json) records each SHA-256 and qualified source.

For #1045 I retain the actual two-build before/after control in `docs/evidence/compiler-version-reproducibility-20261107/`: the original differs under two SOURCE_DATE_EPOCH values; the corrected actual compiler reports match, shared providers retain bytes/mtimes, and six release-version/package controls pass. I have a roadmap acceptance row. This meets the issue's three listed criteria; full binary reproducibility and path-dependent diagnostics remain separate under #1039.

For #1073 the same evidence retains the original strict GCC errors and records corrected GCC 16 and Apple Clang C11/O3/Werror checks. The source comparison verifies that the tested entry point reaches my integration unchanged. I qualify the three argument/format diagnostics; I do not qualify a legacy hybrid bootstrap or runtime.

For #1078 I retain the original capacity mutation, corrected capacity/hash native and ASan/UBSan controls, 2,380 owned-union assertions and all five strict GCC profile results in `docs/evidence/gcc-test-path-bounds-20261107/`. My subsequent exact-source hosted audit at `7db768cc6` passes all 1,483 selected compiles per compiler, including the repaired fixtures, while preserving assertions and fault positions. I qualify the issue's capacity/hash/fixture criteria. The broader #1046 inventory and combined runtime budget remain open.

I reuse these retained actual results and verify their source bytes; I do not claim new local compilation on an overloaded worker. Main merge and the full integration/native/release gates remain pending independently of these scoped issue closures.
