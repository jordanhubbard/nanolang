# My combined affine and array qualification

I qualify aggregate `28e50b802` on Darwin. My compiler production inputs were frozen at `57652fc51`; the subsequent field-test linker correction changed only tests and evidence. Fresh bootstrap `run-m17ygl7e` passed all 17 steps, including generation, verification, translation, native compilation and smoke execution. Raw Stage 1 and Stage 2 modules are byte-identical. My installed compiler also compiled and executed hello with the C seed removed.

I supplied all three absolute producer paths to the declared-purity, inferred-effects, match-coverage and affine-completion suites. All 20 methods passed in 94.492 seconds, including their VM, sanitized-native and actual parser/checker controls. I retain the full bootstrap manifest, logs and provider hashes here.

This receipt qualifies these combined compiler inputs locally. Hosted exact-candidate Linux/Darwin checks remain pending. The next source-array publisher and owned reconstruction changes require another fresh combined qualification; this receipt does not cover those later changes. Broader per-use affine facts, tail transfers and release obligations remain open.
