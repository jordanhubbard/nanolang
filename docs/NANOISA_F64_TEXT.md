# My exact F64 assembly token

I record task `task_50ba3bd3018f438a93113891d48e1387` before code.
My canonical F64 operand is `bits:` followed immediately by exactly 16 ASCII
hexadecimal digits, most significant digit first. Input accepts either digit
case; output uses lowercase. The prefix is lowercase and case-sensitive.

I accumulate unsigned integer bits and copy them with `memcpy`; I do not
convert the number numerically. All binary64 encodings are valid operands,
including signed zeros, infinities and quiet/signaling NaN payloads. I leave
execution semantics unchanged. Canonical output uses this token for every F64
operand. Existing decimal/strtod input remains accepted with its current
conversion/error policy. Invalid length, nonhex digits, unknown prefixes and
trailing operand text remain parser refusals. My existing line preprocessing
strips unquoted `;` and `#` comments before operand parsing, including adjacent
comments without whitespace; output publication remains
atomic.

I require exact module bytes after dump/reassembly for ordinary valid bit
patterns, uppercase input normalization, legacy text controls, existing
roundtrip consumers, and previous-output preservation on refusals. This
contract does not admit float source reconstruction or change its whitelist.
I do not use historical failed compiler artifacts.

## My measured acceptance

My base is `5faaf843`; contract `7e07e967` precedes production `4a2c7a53`.
I tested 17 exact encodings through assembly, canonical disassembly, reassembly,
full module-byte equality and the verified facts reader. Five legacy decimal
inputs retain their values. Eleven normal parser refusals preserve previous
output and require an ordinary positive error exit without sanitizer reports.
My final delimiter controls cover whitespace and adjacent `;`/`#` comments.
The existing reconstructed-source float refusal remains tested for both targets.

The initial five focused methods pass normal GCC (0.206s), instrumented GCC
(1.194s), and instrumented Clang (1.255s). My final six-method Clang gate adds
comment/delimiter controls. Both instrumented builds pass all 210 existing
canonical roundtrip checks. Additional Clang consumer gates pass 2,856 NanoISA
checks and 20 v2 emit/load checks. I instrument the changed assembler and
disassembler objects, facts main and test harnesses with ASan/UBSan and strict
warnings; other linked runtime objects retain normal builds. I make no fully
instrumented-runtime claim.

I retain logs under `/tmp/nanolang-f64-text-`: `focused.log`,
`gcc-sanitized.log`, `clang-sanitized.log`, `delimiters-final.log`,
`gcc-sanitized-build.log`, `clang-sanitized-build.log`, and `consumers.log`.
The initial `build.log` records a missing `<inttypes.h>` include; I corrected it
before executing tests. No historical compiler or failed artifact was replayed.
