# I lower scalar match guards and block values

I retain my original named-union and integer/wildcard fixtures from `tests/test_canonical_match_guards.py`. The baseline named-union fixture passes both producers, while self-hosted integer/wildcard lowering refuses it. After scalar lowering, that original fixture passes; my added block-local fixture still fails. I retain both failures.

My correction passes all four focused methods in 2.435 seconds on Darwin: original named-union effects; original integer/wildcard effects, early return and loop exits; block-local inferred values and scope plus union wildcards; and five guard/domain/coverage refusals. Each positive fixture runs through both the C-seed NanoVirt and a C-seed-built self-hosted emitter, verified NanoVM execution, nvm2c, and strict LLVM Clang C11 with ASan/UBSan and leak detection. C-seed refusal preserves previous output; component refusal emits no module.

I built the driver with the primary `bin/nanoc_c`, from this worktree, with normal dependency shadows. Consumer tools are from the separately qualified `0273c40fd` array worktree. Exact source and driver hashes are in `tested-inputs.json`; these are component results, not fresh installed-stage or Linux qualification.

Issue #981 remains open. I still must migrate the unchecked legacy backstop fixture, update the installed wildcard profile expectation, run the complete canonical corpus and fresh platform gates, and review expression forms beyond these scalar fixtures. I do not replace any original assertions or claim release completion.
