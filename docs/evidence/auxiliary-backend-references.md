# Auxiliary backend references

I removed retired direct AST LLVM/Wasm runners and their obsolete expected-failure entries. My remaining auxiliary runner executes C and reports RISC-V/PTX structural checks separately. Selecting a retired backend fails before compiler execution; I do not report it as skipped or passing. My example help now lists the supported `c|native|nanoisa|vm` modes.

My old `wasm-playground` recipe named removed sources and omitted `src/wasm_interface.c`, the browser entry-point implementation. I replaced that broken rebuild promise with an explicit failure before output mutation. I retain the historical browser bundle and document that it is not a current-language acceptance artifact. The separate scalar NanoISA Wasm translator does not implement the browser interpreter API. Full backend and browser integration remain separate work.

For MAC `task_26dc7883efaf4b7a83e83128a10f99a7`, `make test-cross-backend-runner` passed ten methods in 0.918 seconds, including execution failure despite matching stdout, output mismatch, expected/unexpected outcomes, structural-only reporting, retired-mode refusal before compilation, truthful help and unchanged browser bundle bytes. `bash -n tests/cross-backend/run-all.sh`, `make help`, the expected nonzero `make wasm-playground`, and `git diff --check` also passed their contracts. Log: `/tmp/nanolang-auxiliary-gates.log`.

These controlled runner checks establish harness behavior, not new compiler conformance or browser execution. I did not run Emscripten or claim a repaired browser interpreter.
