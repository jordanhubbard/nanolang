# My byte-to-string conversion contract

I track task_d08968be827a4d26942123825229ac8e after the scalar U8 foundation.
CAST_STRING of a TAG_U8 value produces its unsigned decimal representation,
with no sign, prefix, padding or leading zeroes (except the value zero itself).
This agrees with existing byte display and CAST_INT followed by CAST_STRING.
The result has TAG_STRING and uses the same owned allocation and reference
lifecycle as an integer conversion. The consumed byte has no heap reference.

I reuse my existing VM integer-string allocation and native managed numeric
string helper. I preserve aliases, returned strings and subsequent collection;
I do not introduce an allocation outside those lifetime systems. I test all
256 ordinary values and retained strings across conversion churn, including
native sanitizer/leak checks. LLVM/Wasm continue to refuse CAST_STRING because
their current profile excludes strings; failed publication preserves output.

The earlier empty-string VM fallback is retained in the preceding U8 task's
failure logs. It is not my intended numeric string conversion contract.
