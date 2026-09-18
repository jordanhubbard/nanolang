# My integer shift reconstruction contract

I record task_417225d8ac7e49278c25eee9ed1291b9 before code, after PR659 under parent4bd034. I admit I64_SHL, I64_SHR_S and I64_SHR_U for exact int inputs/results. I use the low six bits of the count, including negative counts, as my VM does. Left shift wraps modulo2^64, signed right shift extends the sign bit, and unsigned right shift fills with zeros. My VM uses host signed C right shift for SHR_S; my emitted C independently implements arithmetic shift with unsigned operations and explicit sign fill. I test VM parity on the stated host rather than asserting all C hosts implement signed shift identically.

My emitted C masks counts before shifting and never shifts by64 or shifts a negative signed value. My NanoLang helpers normalize signed count remainder into0..63 and take at most63 arithmetic steps. Left shift uses my total add helper; right shift halves toward negative infinity for signed values, then removes sign extension for logical shifts using representable intermediates. I do not assume a source bitwise operator exists. This finite cost is not machine-shift performance parity.

I preserve exact tags, operand snapshots, structured regions and atomic output. I require same-module VM/C/pinned C-seed/Stage1/Stage2 tests for MIN/MAX, negative values/counts, counts0/1/63/64/65 and large signed counts, calls, loops and refusal/output preservation. Generated C enables ASan/UBSan; C-seed Nano-C enables UBSan. Pinned tools establish compatibility, not a fresh current-source bootstrap. Bitwise AND/OR/XOR/INVERT, unsigned arithmetic and full reconstruction remain separate.

## My bounded evidence

My generator production checkpoint is162895d9 on main4ad97303. Three focused methods passed37.247s in `/tmp/nanolang-reconstruct-shifts-first.log`. They compare351 value/count combinations (117 per operation), calls, a shift loop condition, loaded-operand snapshots, exact-tag refusal and byte roundtrip. My Python oracle uses arbitrary-precision signed arithmetic and explicit64-bit wrapping. Each family has a separate module, so right-shift-only source also checks that unrelated helpers are not emitted. Generated helper shadows validate my helpers; they do not reconstruct original tests.

My reused compiler source pin is4a75f984, with SHA-256:

- C seed: `442caf121e671388e4d22ea207789ad56efca659555c4ee451596eb3d4b99041`.
- Stage1: `d078dc8c6747b849cb2228fa33165b06aa350d5f6e53b8911fd72696515542fa`.
- Stage2: `b276026caf525bde10c063b2f7654f192b81ace73587e5af5bebda4986581a40`.

I recorded the local assembler, VM, facts tool, reconstruction CLI and generator hashes before testing in `/tmp/nanolang-reconstruct-shifts-tools.sha256`. I run direct unittest commands without Make/bootstrap or concurrent rebuilds. This establishes pinned-tool compatibility, not current-main compiler bootstrap. My reconstructed C enables ASan/UBSan, and C-seed Nano-C enables UBSan.

My combined18-method gates passed: GCC173.983s (`/tmp/nanolang-reconstruct-shifts-gcc.log`) and Clang167.175s (`/tmp/nanolang-reconstruct-shifts-clang.log`). Every local tool/generator hash and all three compiler hashes matched after both runs. No required gate remains running. These tests cover the admitted typed integer/boolean reconstruction grammar; they do not close full parent4bd034 or establish all-host VM signed-shift portability.
