# My source-snapshot recovery

My first `make -j2 test-bytecode-shadows test-parser-parenthesized` exits 2.
The bytecode-shadow 40 and cache-publication 56 methods pass; four Linux-link
checks skip on Darwin, and the compiled parenthesized-parser assertions pass.
The source-snapshot phase runs 126 methods in 4,287.554 seconds: 22 failures,
113 errors, 19 skips. All 135 failure/error reports are classified: 115 contain
explicit disk-full errors, while 20 reject `-Xlinker -lm` during compile-only
jobs under `-Werror`. I retain the entire terminal and classification.
This development run spans source changes and disk recovery; it is not a
clean-pin acceptance result. The Make target stops before later link suites.

I remove linker operands from compile-only fallback jobs regardless of whether
I captured linker provenance. I retain paired non-linker operands, including
a literal include directory named `-Xlinker`. Link jobs keep their arguments.
The uncaptured-linker control executes the library and verifies that missing
provenance still prevents cache publication. I do not disable warnings.

The corrected compile path exposes a second refusal: my selected Apple linker
reports version `27037.1`, while grammar admission named only `1267`. I add
this exact version alongside the prior one; unknown versions, including
`27037.2`, remain refused. Actual link-response/query/argument controls pass
49 methods in 110.783 seconds. The uncaptured control and full split-assembler
search/recovery method pass in 103.714 seconds. The latter covers common,
platform and package flags with integrated/external assembly and private/shared
caches, retaining source selection and recovery assertions.

Commands: `make -j2 obj/test_module_generation_probe`, then Python unittest
modules `tests.test_link_response_query`, `tests.test_link_response_graph`,
`tests.test_link_argument_transport`; focused `SourceSnapshots` methods
`test_uncaptured_linker_flags_stay_out_of_compile_jobs` and
`test_split_assembler_search_order_phases_and_recovery`. I retain all logs.

My remaining affected standalone/shared assembly cases pass: four methods,
16 subcases, 351.058 seconds. I retain `affected-cases.log` and its runner.
My full 127-method source-snapshot rerun is running in the clean qualification
checkout at `f0f6a0c62`; its terminal result remains open. I preserve qualification
clones and their host-library paths; no source corpus assertion is weakened.
