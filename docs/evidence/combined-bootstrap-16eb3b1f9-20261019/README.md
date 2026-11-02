# I retain my combined mutation bootstrap

At `16eb3b1f9` all 17 raw bootstrap steps passed. Stage 1 and Stage 2 are byte-identical with SHA256 `03f6b51b37909a9c9aa6b7a26e69efecc11303f7776029dc56771fd648f769bb`. Native hello checks and the installed compiler check with the C seed removed also passed. The bootstrap installs raw modules in `bin/nanoc_stage1.nvm` and `bin/nanoc_stage2.nvm`; I record their actual installed hashes.

I then passed all 27 methods in 50.708 seconds across the original array-field setter regression, contextual nested-array mutation, and flat/nested array source publication. The matrix uses NanoVirt and both freshly installed stages (the original setter suite also uses the C native compiler). It covers the previously failing untyped empty replacement, all primitive element contexts, once-only evaluation order, wrong-storage refusal with old-output preservation, checked imports and retained authority bytes. Actual VM and generated sanitized native execution remain asserted by their respective suites.

These are pinned Darwin results. Full paired File parser qualification is running separately; hosted Linux/Darwin and remaining release scope stay open. I do not infer full 5.1 readiness from this checkpoint.

I normalize trailing display whitespace in copied logs. `retained-original-hashes.json` identifies changed copies and preserved original paths/hashes; manifest log hashes refer to those originals.
