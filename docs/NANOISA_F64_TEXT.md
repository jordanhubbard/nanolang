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
trailing operand text remain parser refusals; output publication remains
atomic.

I require exact module bytes after dump/reassembly for ordinary valid bit
patterns, uppercase input normalization, legacy text controls, existing
roundtrip consumers, and previous-output preservation on refusals. This
contract does not admit float source reconstruction or change its whitelist.
I do not use historical failed compiler artifacts.
