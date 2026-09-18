# My binary64 operand facts

I add `f64_bits` to an instruction object exactly when its first operand has
`OPERAND_F64` encoding. Its value is a JSON string of exactly 16 lowercase
hexadecimal digits, most significant digit first, without a prefix. For example,
negative zero is `"8000000000000000"`. This is an additive facts-schema field;
I leave the existing keys and integer `arg` interpretation unchanged. The
legacy `arg` is zero for this operand and is not a float value.

I decode the little-endian operand with my existing decoder, copy its binary64
storage into `uint64_t` with `memcpy`, then format the integer bits. I do no
floating arithmetic or decimal formatting. The field retains signed zeros,
finite values, infinities, and quiet/signaling NaN payload and sign bits.

These facts do not grant executable admission. I retain my bounded int/bool
function signatures, entry restrictions, instruction whitelist and atomic
source publication. `PUSH_F64` still refuses reconstructed C and NanoLang
output. I do not claim float source constants, arithmetic or comparisons.

My next source contract must distinguish exact bit transport from arithmetic
results. Typed F64 comparisons use IEEE unordered behavior; generic float
ordering has a separate VM contract. Typed F64 division returns positive zero
for either zero divisor. Signed-zero behavior, operation rounding, NaN result
policy and conversions need explicit acceptance before admission.

At the facts checkpoint, my canonical disassembler used `%.17g` for float
operands and could not preserve every NaN payload/signaling encoding. My
[subsequent exact-token repair](NANOISA_F64_TEXT.md) addresses that boundary
under `task_50ba3bd3018f438a93113891d48e1387`.
My ordinary facts tests supply valid operand bits directly and do not claim
canonical text or executable float roundtrip.

I run `make test-reconstruction-binary64-facts` for 17 binary64 patterns,
unchanged integer/boolean facts and previous-output preservation for both
source targets. No copied compiler is required for this facts-only gate.

## My measured checks

At source base `66508fe4`, my two focused methods pass with normal GCC, then
strict `-O2 -Wall -Wextra -Werror` GCC and Clang builds instrumenting
`hl_facts_main.c` with ASan/UBSan. Shared decoder/loader objects remain normal
builds. Logs are `/tmp/nanolang-binary64-facts-corrected.log`,
`/tmp/nanolang-binary64-facts-gcc.log`, and
`/tmp/nanolang-binary64-facts-clang.log`.

My initial fixture changed valid operand bits but omitted recomputing the
serialized module checksum; my loader rejected all 17 altered modules before
fact extraction. I retain `/tmp/nanolang-binary64-facts-test.log`. I corrected
only fixture serialization and did not change the production implementation
to obtain the passing results. The source-refusal method passed initially.
I did not execute any float module or invoke historical compiler artifacts.
