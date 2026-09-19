# My true-switch private owner ARRAY supplement

I correct PR836 dispatch attribution without replacing its original seals. At frozen35a3472bc I explicitly define NANO_NO_COMPUTED_GOTO, capture compiler preprocessor macros and assert that NANO_COMPUTED_GOTO is absent. Each run executes the unchanged seven-case VM fixture with fused/unfused observations, allocation failures/recovery, exact graph/reference/byte baselines and growth-accounting controls.

- gcc: PASS, 3.019s,4739 checks.
- gcc-san: PASS, 9.681s,4739 checks.
- clang-san: PASS, 10.107s,4739 checks.

I retain exact [reports and artifacts](private-owned-array-true-switch.json), unchanged836 runtime/C fixture sources and copied qualified providers. This supplement does not repeat native or computed-goto execution. GCC/Clang sanitizer instrumentation remains limited to private VM/translator/heap/fixture units; linked providers are ordinary qualified objects. All source/tool before/after identities match. Public activation and source/mutation acceptance remain open.
