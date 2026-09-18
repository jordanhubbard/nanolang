# My signed division reconstruction contract

I record task_dc02359771154508bcdd9c25de7b1bbc before implementation, following PR658 under parent4bd034. I admit I64_DIV_S and I64_REM_S for exact int operands and results. My VM (src/nanovm/vm.c typed arithmetic dispatcher) and native C translator return zero for a zero divisor. MIN divided by -1 returns MIN; its remainder is zero. Other quotients truncate toward zero and nonzero remainders have the dividend sign. These exceptional numeric cases do not trap. Wrong operand tags retain checked refusal; enums and generic/unsigned arithmetic remain outside this reconstruction profile.

I emit C and NanoLang helpers that test exceptional cases before evaluating / or %. I preserve decoded operand snapshots, structured control limits, exact local/call types and atomic publication. I test endpoint cross products, signed nonexact quotients, zero/overflow, calls, loop conditions and wrong-tag/output preservation against the same verified VM module and reconstructed C/NanoLang products. GCC/Clang sanitizers cover generated C; C-seed-generated NanoLang C also enables UBSan. Explicitly pinned compiler tools establish compatibility, not a fresh current-source bootstrap. Full reconstruction remains open.

## My evidence boundaries

My generator production checkpoint is 4dd4ffe2 on main e4706bf5. My focused three methods passed in 28.666s: 121 operand cross-products per operation, byte roundtrip, VM execution, reconstructed sanitized C and C-seed/Stage1/Stage2 NanoLang products, loop-condition evaluation, a loaded-value snapshot and boolean-tag refusal before publication. The generated helper shadows test helper behavior, not recovered original tests.

I reused compiler tools from source4a75f984, with SHA-256:

- C seed: `442caf121e671388e4d22ea207789ad56efca659555c4ee451596eb3d4b99041`.
- Stage1: `d078dc8c6747b849cb2228fa33165b06aa350d5f6e53b8911fd72696515542fa`.
- Stage2: `b276026caf525bde10c063b2f7654f192b81ace73587e5af5bebda4986581a40`.

I do not claim these reused tools were built from this generator's base. The C-seed Nano-C checks passed with UBSan enabled. Full current-source bootstrap and full reconstruction are separate obligations.

I retained an incomplete Make invocation in `/tmp/nanolang-reconstruct-div-gcc.log`: its bootstrap prerequisite was unnecessary for the pinned-tool gate, so I stopped only its owned processes. That invocation relinked local tools during the initial Clang gate, which passed15 methods in132.605s but lacks before/after local tool hashes. I treat that result as provisional, not frozen evidence. I then recorded local tools/generator hashes in `/tmp/nanolang-reconstruct-div-tools.sha256` and ran the direct Clang gate again without concurrent builds. The direct GCC gate began after Make stopped. I retain the initial focused and provisional logs separately.

My final direct gates passed all15 methods: GCC135.479s (`/tmp/nanolang-reconstruct-div-gcc-focused.log`) and Clang135.556s (`/tmp/nanolang-reconstruct-div-clang-stable.log`). After both completed, every local tool/generator manifest entry and all three pinned compiler hashes matched. The original focused result is retained in `/tmp/nanolang-reconstruct-div-first.log`; the provisional Clang result remains in `/tmp/nanolang-reconstruct-div-clang.log`. My generated C uses ASan/UBSan, and my C-seed Nano-C uses UBSan. No required check remains running.
