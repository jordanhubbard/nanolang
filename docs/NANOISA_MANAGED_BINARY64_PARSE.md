# My exact portable binary64 parser contract

I record parser `task_4d2f69a19d754ac88876f93a0913d1fb` and cross-host NaN
normalization `task_89fd02e9aa4d4a4faca33ba5fbc2b703` before implementation.
My managed parent and full release remain open.

I preserve the closed C-locale/default-nearest-even strtod value rules: six ASCII
whitespace bytes, optional sign, decimal or hexadecimal significand with one
point, optional complete decimal e/E or binary p/P exponent, ignored suffix,
NUL termination, signed zero, case-insensitive inf/infinity/nan, and no-digit
positive zero. An incomplete exponent leaves the significand value unchanged.
A 0x prefix selects hexadecimal only when a hexadecimal digit follows before
or immediately after the point. I introduce no locale/rounding mutation.

For finite values I retain 800 significant decimal digits or 32 hexadecimal
digits plus a sticky nonzero suffix bit. Every binary64 rounding midpoint is
a dyadic number with at most 54 numerator bits and denominator at most 2^1075;
for a negative binary exponent its decimal coefficient is less than
2^54 * 5^1075. I bound 2^54 < 10^17 and
5^1075 = (5^10)^107 * 5^5 < 10^749 * 10^4 = 10^753,
so that coefficient has at most 770 significant decimal digits. Midpoints
with nonnegative binary exponent have at most 309 decimal digits. Both bounds
are strictly below my 800 retained digits. A midpoint cannot lie strictly
inside the decimal interval selected by that prefix: it would need more than
800 significant digits. A finite discarded tail cannot reach the upper
endpoint. Only an exact lower-endpoint tie needs the sticky bit to distinguish
an exact tie from a value strictly above it. The 32 hexadecimal digits retain
128 bits, likewise more than the 54 significant midpoint bits.
I count omitted/fractional digits and saturate enormous explicit exponents far
beyond any cancellation possible within a uint32 byte length.

I convert the retained exact rational using checked fixed-capacity base-2^32
integers, comparing shifted numerator/denominator to find the binary exponent,
then dividing for the 53-bit normal significand or fixed 2^-1074 subnormal quantum.
Remainder comparison and sticky implement nearest-even, including the
subnormal/normal and finite/infinity transitions. Decimal exponent shortcuts
apply only where all values are certainly zero or infinite. Finite relevant
powers need fewer than 4096 bits; I check every capacity operation and report
an internal status rather than silently truncating. Parsing allocates nothing.

My original ordinary reference logs are
`/tmp/nanolang-managed-parser-linux.txt` and
`/tmp/nanolang-managed-parser-darwin.txt`. They demonstrate a value difference:
`nan(184467440737095516160000)` has bits `7fffffffffffffff` on Linux ARM64 and
`7ff8000000000000` on puck/Darwin. I explicitly normalize to the existing Linux
policy: valid decimal/octal/hex numeric NaN payloads saturate to unsigned64,
then mask payload bits and set the quiet bit; invalid payload names use zero.
The leading sign remains. This changes Darwin's overflowing decimal payload
case deliberately; endptr-only differences are irrelevant to CAST_FLOAT.

One pure parser source serves VM CAST_FLOAT, static-string and boxed native C
CAST_FLOAT, and managed LLVM/Wasm. I embed generated source in standalone C
output, check generation drift, hash package inputs and wire header dependencies.
String operands remain borrowed while parsing and are released once by existing
frame cleanup. I lift target CAST_FLOAT refusal only after matched finite,
subnormal/tie/overflow, syntax, NaN-policy and ownership gates pass.

Legacy AST conversion routes in stdlib_runtime.c, transpiler.nano and eval.c
remain explicit companion `task_9e93c1badb1a4da093a737b3a2c15ef7`; their strict
endptr checks must be preserved before full source-route parity is claimed.
I test ordinary reference inputs and long precision/cancellation controls on
native/Wasm and the actual VM/C target paths, preserving historical artifacts
without executing them. I claim only measured target acceptance.
