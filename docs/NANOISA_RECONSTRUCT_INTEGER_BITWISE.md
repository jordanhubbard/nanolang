# My integer bitwise reconstruction contract

I record task_c4188559900c4a48887c5c29f77b6ed5 before code, following PR661 under parent4bd034. I admit I64_AND, I64_OR, I64_XOR and I64_INVERT for exact int operands/results. My VM applies the operations to unsigned64 bit patterns and returns an integer carrying those bits. I retain this behavior with portable unsigned C operations followed by my representable signed conversion.

My NanoLang binary helpers inspect64 successive low bits, combine boolean selections, and accumulate powers of two through my existing total-add helper. I advance operands with my tested logical-right-shift helper using count1. Inversion uses -1-a, whose result is representable for every signed64 input. I emit only needed helpers and their dependencies, each with meaningful generated shadows. The binary helpers take64 steps (each right shift takes one step); this is not machine-bitwise performance parity or reconstructed original source tests.

I preserve exact tags, operand snapshots, structured-region limits and atomic refusal. I require endpoint and alternating-pattern pairs, calls, a loop/snapshot case, wrong-tag output preservation, canonical byte roundtrip and same-module VM/C/pinned C-seed/Stage1/Stage2 execution. Generated C enables ASan/UBSan; C-seed Nano-C enables UBSan. Tool hashes must remain unchanged. Pinned compiler compatibility does not establish current-source bootstrap. Unsigned arithmetic/comparisons, other values and full reconstruction remain separate.

## My bounded evidence

My generator production checkpoint is f9b234f4 on main4f715b09. Four focused methods passed45.060s (`/tmp/nanolang-reconstruct-bitwise-first.log`). They exercise363 binary endpoint/pattern pairs,22 inversion/double-inversion checks, direct calls, loop/snapshot behavior, byte roundtrip and four exact-tag refusals preserving previous output. Each binary operation and inversion has an independent module to exercise only-needed helper emission. The Python oracle uses arbitrary-precision bitwise operations followed by explicit64-bit wrapping. My generated helper shadows test helper behavior; they do not recover original source shadows.

My reused compiler source pin is4a75f984, with SHA-256:

- C seed: `442caf121e671388e4d22ea207789ad56efca659555c4ee451596eb3d4b99041`.
- Stage1: `d078dc8c6747b849cb2228fa33165b06aa350d5f6e53b8911fd72696515542fa`.
- Stage2: `b276026caf525bde10c063b2f7654f192b81ace73587e5af5bebda4986581a40`.

I recorded the local assembler, VM, facts tool, reconstruction CLI and generator hashes before testing in `/tmp/nanolang-reconstruct-bitwise-tools.sha256`. I run direct unittest gates without concurrent builds. Generated C uses ASan/UBSan; C-seed Nano-C uses UBSan. This is pinned-tool compatibility evidence, not current-main compiler bootstrap or full reconstruction acceptance.

My combined22-method gates passed: GCC214.911s (`/tmp/nanolang-reconstruct-bitwise-gcc.log`) and Clang204.734s (`/tmp/nanolang-reconstruct-bitwise-clang.log`). Every local tool/generator hash and all three pinned compiler hashes matched afterward. C-seed Nano-C checks passed with UBSan enabled; no required gate remains running. These results establish this bounded typed-bitwise profile, not full parent4bd034.
