# My independent LLVM evidence integrity review

I independently read the actual Git report blobs at
`3d4afccb726e18c8b59813c1de1b710e3b56636f` and every object in the retained
LLVM artifact store. My [audit](audit.py) passes [these checks](checks.json):
637 reports, 37,086 objects containing 5,820,577,421 bytes, 136,103 artifact
references and 90 equal source/tool before-and-after pairs. I also require the
exact report and object sets; a missing or extra entry fails this audit.

My 46 terminal reports include preserved unsuccessful historical attempts.
Their presence is evidence retention, not a claim that all attempts passed.
This review establishes committed bytes and retained object identity. It does
not establish semantic corpus coverage, current-main integration, full backend
parity or release readiness. Those require their own completed acceptance.

My separate [native coverage audit](native-coverage.py) now verifies all73
products in linked and observed modes at O0 and O2 across seven native
configurations. Each complete configuration has292 retained executables,
2,432 checked build/tool/execution terminals,146 complete allocation coverage
records,676 fault workers and18,976 recoveries. I check the actual invoked
executable path, linked observation output, excluded VM symbols and every
worker's contiguous range and exact two-mode recovery output.

Linux Clang sanitizer O0 comes from the completed portion of the interrupted
combined run; O2 comes from its separately attributed continuation. I exclude
the interrupted partial O2 work from coverage. The [results](native-coverage.json)
keep both sources explicit. Startup, emission, Wasm, package, normalized corpus
equivalence and current-main integration require separate evidence.

My [first audit script](native-coverage-first.py) stopped on Darwin's `.dSYM`
debug companion sharing the executable basename. The corrected lookup excludes
those debug-directory entries from command-product lookup, while the complete
integrity audit still verifies their bytes. No product, fixture, execution or
assertion changed to resolve this report-schema mistake.

My [Wasm coverage audit](wasm-coverage.py) independently verifies both retained
host runs. Each has292 main modules plus two limited-memory modules,3,698
selected build/tool/engine terminals,2,524 actual engine invocations,292 complete
fault coverage records,1,352 workers and37,952 recoveries. Both Wasmtime and
Node run all73 linked/observed cases at O0 and O2. I inspect actual module import
sections, invoked paths, numeric outputs, memory growth and limited-memory
refusal, and every contiguous fault range. My [results](wasm-coverage.json)
exclude separate startup/ABI/emission/package and current integration claims.
