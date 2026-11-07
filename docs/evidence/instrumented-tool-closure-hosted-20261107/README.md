# Hosted instrumented tool closure

I qualify #1075 at integration commit `435dae34cc2ce74d5210806d26ef1ebb9f9e4b7a` in [immutable image job 128477711608](https://github.com/jordanhubbard/nanolang/actions/runs/42397278342/job/128477711608). My [job record](hosted-job.json) preserves the exact head, completed successful steps and observation while the parent workflow is still running. My [selected smoke output](selected-smoke.log) preserves the real consumer results.

I install LLVM, LLD and Node from my existing signed timestamp-pinned Ubuntu repository, then require actual version commands for Clang, llvm-as, lli, opt, llc, wasm-ld, Node and Wasmtime during construction. Inside the transported and verified image I generate real LLVM IR, verify and assemble it, lower it with llc, link Wasm with wasm-ld, and execute the module in both Wasmtime and Node. My sanitizer/coverage probes and immutable-root, transport and checkpoint identity controls also pass.

I close the missing-image-tools defect from this exact hosted image and consumer evidence. I do not claim a complete unit suite, instrumented bootstrap, optional translator implementation or release qualification from these small controls. The parent CI and qualification workflows remain independent requirements; #1060 still needs its own generated-program execution evidence.
