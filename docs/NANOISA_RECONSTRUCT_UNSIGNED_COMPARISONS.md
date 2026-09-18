# My unsigned comparison reconstruction contract

I record task_d4c7fd42960d4f218249555b7e268a70 before code, after PR663 under parent4bd034. I admit I64_LT_U, I64_LE_U, I64_GT_U and I64_GE_U for exact integer operands and boolean results. My VM compares the operands as uint64 bit patterns. My emitted C does the same. My NanoLang helpers first order opposite-sign halves (negative signed values have the larger unsigned values), then compare signed operands within the same half. No numeric conversion, floating rounding or arithmetic overflow is needed.

I retain boolean result identity through locals, calls, boolean operators and structured branches/loops. I preserve operand snapshots, exact types, region bounds, only-needed helpers and atomic output. I require endpoint/alternating-pattern cross-products, direct bool-return helpers, loop conditions, snapshots, wrong-tag refusal and byte roundtrip, paired through VM/C/pinned C-seed/Stage1/Stage2. Generated C uses ASan/UBSan; C-seed Nano-C uses UBSan. Tool hashes are recorded before tests and checked afterward. Pinned-tool compatibility is not current-source bootstrap. Unsigned division/remainder, generic comparisons and full reconstruction remain separate.

## My evidence and retained correction

My corrected generator checkpoint is ab5c186b on maincc9c7a1d. The corrected focused3-method gate passed43.956s (`/tmp/nanolang-reconstruct-ucompare-corrected.log`), covering484 endpoint/pattern pairs, direct bool-return calls, sign-boundary loop/snapshot behavior, canonical byte roundtrip and input/result-tag publication refusals. Four comparison families have separate modules, testing only-needed helper emission. My oracle compares Python integers normalized into0..2^64-1. Generated helper shadows are helper tests, not original reconstructed shadows.

I retain the first10.492s run in `/tmp/nanolang-reconstruct-ucompare-first.log`. Five positive subcases hit pinned Cseed4a75f984's strict C parentheses warning on nested boolean comparisons. I filed current-main audit task_de7d1397f86940f8b759ae07eb46820f, then used named bool sign intermediates without changing any compiler. Four negative subcases incorrectly expected assembler refusal of a declared int return carrying a bool. I corrected them to test exact reconstruction refusal with both previous C/Nano outputs preserved; those modules are not executed. This does not establish a new VM/runtime failure.

My reused compiler source pin is4a75f984, with SHA-256:

- C seed: `442caf121e671388e4d22ea207789ad56efca659555c4ee451596eb3d4b99041`.
- Stage1: `d078dc8c6747b849cb2228fa33165b06aa350d5f6e53b8911fd72696515542fa`.
- Stage2: `b276026caf525bde10c063b2f7654f192b81ace73587e5af5bebda4986581a40`.

I retained initial local tool/generator hashes in `/tmp/nanolang-reconstruct-ucompare-tools.sha256` and recorded corrected hashes before corrected tests in `/tmp/nanolang-reconstruct-ucompare-corrected-tools.sha256`. Direct unittest gates run without concurrent builds. Generated C enables ASan/UBSan; C-seed Nano-C enables UBSan. This is pinned-tool compatibility, not current-main compiler bootstrap or broad reconstruction acceptance.

My combined25-method gates passed: GCC268.987s (`/tmp/nanolang-reconstruct-ucompare-gcc.log`) and Clang255.836s (`/tmp/nanolang-reconstruct-ucompare-clang.log`). All corrected local tool/generator hashes and all three pinned compiler hashes matched afterward. C-seed Nano-C UBSan checks completed; no required gate remains running. Full reconstruction and current-source qualification of the pinned parentheses observation remain open.
