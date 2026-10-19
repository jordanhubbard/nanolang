# Integrated FLOAT reconstruction qualification

Candidate `3446ec677793b8f95b64988237874136decdd8c3` completed fresh bootstrap `run-mjg4sby8`: all 17 recorded steps passed, including both raw generations, verification, native compilation and smoke tests. `make bootstrap3` exited zero after final installation checks. Raw Stage 1 and Stage 2 are byte-identical, SHA-256 `f1f25eb29c823bfcfdfcd1273288ecdc36035542ce3830bf10b8d963bff2e164`.

The integrated assembler/facts target passed. Then the six FLOAT and ten STRING methods passed through the current seed and both freshly installed stages: 16 methods in 78.034 seconds. All 14 pinned tools and host libraries were unchanged afterward. These stages use the normal newly built host closure, closing the earlier private-provider-only subnormal qualification gap. The component receipts retain the original installed-stage failure and explicitly modified private-provider experiment.

The exact unchanged ordinary example and independently produced original-shadow input remain separate. FLOAT fields, parameters/results and borrowed FLOAT profiles remain outside this bounded local-storage slice. Owned Nano output still refuses nonfinite literal payloads atomically; C preserves their exact bits, while runtime-created NaN behavior is tested separately. No synthetic original-shadow recovery or all-program correctness claim is made.

Commands: `NANO_SHADOW_TIMEOUT_SECONDS=300 make -f Makefile.gnu -j1 CC=/opt/homebrew/opt/llvm/bin/clang bootstrap3`; then `NANOLANG_OWNED_RECONSTRUCTION_COMPILERS="$PWD/bin/nanoc_stage1 $PWD/bin/nanoc_stage2" python3 -m unittest -v tests.test_owned_float_reconstruction tests.test_owned_string_reconstruction`.

Exact-candidate Linux/Darwin hosted qualification and remaining required reconstruction profiles remain open. This local fixed point does not establish cross-environment reproducibility.
