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

My fixture covers positive/negative zero, unary versus subtraction, whole and
fractional values, six-significant-digit precision, small and large exponents,
runtime overflow to positive/negative infinity, runtime infinity subtraction to
NaN, and once-only operand evaluation. Five functions compare exactly against
C-seed bytecode (12 checks); both modules verify and run in NanoVM and native
code. Printed nonfinite results agree exactly with C-seed native execution.

With PR510's scalar emitter and merged native companions PR513/516, the full
paired emitter gate passes 86 existing comparisons and 82 Python methods in
95.640 seconds, with native checks enabled. A fresh native three-stage bootstrap
passes at the default deadline; the paired C-seed/Stage1/Stage2 float tests pass
(two methods, six compiler cases, 15.901 seconds). The final rebase onto merged
PR510 (`8e56e9cf`) changes only inherited evidence, not this tested production
source. Schema regeneration retains both unary and borrow metadata.

Task: `task_29976241a36244f7b0ce4ad75cb10b3f`. Local evidence:
`/tmp/nanolang-float-format-exact.log`, `/tmp/nanolang-format-final516-gate.log`,
and `/tmp/nanolang-format-bootstrap-final.log`.
