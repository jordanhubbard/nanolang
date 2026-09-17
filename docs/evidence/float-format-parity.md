# My float formatting contract

I retain the existing `float_to_string` contract: libc `%g` precision and
spelling, with `.0` appended only when the result contains none of `.`, `e`,
`n`, or `i`. This keeps whole floats distinct from integers while leaving
fractional/exponent forms and infinities/NaNs intact. Generic `CAST_STRING`
remains unchanged.

My C-seed and self-hosted NanoISA emitters evaluate the operand once, cast it,
then use existing string and branch instructions to apply that suffix rule.
My self-hosted native runtime helper follows the same rule as my C reference
interpreter and native helper. I add no instruction or host ABI.

My focused fixture covers positive/negative zero, unary versus subtraction,
whole and fractional values, six-significant-digit precision, small and large
exponents, runtime overflow to positive/negative infinity, runtime infinity
subtraction to NaN, and once-only operand evaluation. Five functions compare
exactly against C-seed bytecode (12 checks); both modules verify and run in
NanoVM, and my C reference normal compilation/shadows and executable pass.
The permanent tests also compare printed nonfinite results between paths.

A fresh native three-stage bootstrap and paired C-seed/Stage1/Stage2 formatting
gate are in progress. Native AOT acceptance remains dependent on typed-F64
support (`task_fd4c63cf9f3e46f09ece380ce00c7a58`) and the already recorded
`JMP_TRUE` companion (`task_211f22859e164287a07a63cba74ace5b`). I keep the native
checks enabled and do not claim acceptance until they pass.

Task: `task_29976241a36244f7b0ce4ad75cb10b3f`. Local evidence:
`/tmp/nanolang-float-format-exact.log`, `/tmp/nanolang-float-format-native.log`.
