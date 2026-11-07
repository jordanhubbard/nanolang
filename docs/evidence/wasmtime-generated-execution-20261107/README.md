# Wasmtime generated-program dependency qualification

I preserve [failed run 42405633652](https://github.com/jordanhubbard/nanolang/actions/runs/42405633652) at `8264642567952468b949c2dfc5c73751573fd497`. Resource checks, prerequisite installation, the actual checksum-pinned Wasmtime installation/version check, and real compiler/tool identity capture pass. My actual VM/C/LLVM/Wasm producers build successfully.

My qualification recorder then asks sha256sum for `bin/nanoisa_dump`, which does not exist: `nanoisa_dump` is a Make target that produces `bin/nanoisa`. This is my workflow recording defect. The actual producer build succeeds, but the original scalar-Wasm test step is skipped after that recorder error. I retain the exact job/step states, tool identities, installer output, worker observation and full compressed build log under `first-run/`.

I correct the recorder to hash the actual `bin/nanoisa` once. I preserve every original scalar execution, two-engine, trap, source-preservation and prior-output assertion. A new exact hosted result is still required before I close #1060. I do not call this a product failure, native lifecycle receipt or new mandatory optional-translator release gate.
