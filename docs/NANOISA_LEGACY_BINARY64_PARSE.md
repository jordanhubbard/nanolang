# My legacy binary64 parsing companion

I implement required task `task_9e93c1badb1a4da093a737b3a2c15ef7` after PR657.
My canonical VM/native/LLVM/Wasm value parser remains the source of truth;
this companion preserves each existing legacy route's validation policy.

## My observed routes

- My AST evaluator's `cast_float` accepts a string only when conversion consumes
  at least one byte and the next byte is NUL. Invalid or partial conversion
  emits a diagnostic and returns positive zero. I preserve that distinction.
- My AST evaluator's `string_to_float` ignores the consumed suffix. My C-seed
  and self-hosted generated `string_to_float` helpers do the same. I preserve
  prefix conversion, including no-digit positive zero and embedded-NUL stopping.
- My compiled source `cast_float` helpers support numeric inputs already; I do
  not add a string type-admission path in this companion.

## My consumed-byte contract

I add an optional consumed-byte result to the shared parser while keeping the
existing value-only API. On success the endpoint is within the supplied byte
view. No conversion reports offset zero, even after skipped whitespace/sign.
A valid decimal/hexadecimal significand consumes a complete exponent only;
`1e+` stops after `1`, `0x1p-` after `0x1`, and a zero significand still consumes
its valid exponent. A suffix, whitespace or NUL ends conversion.

I consume `infinity` when complete, otherwise `inf`. I consume a `nan(...)`
suffix only when the closing parenthesis follows ASCII letters, digits or
underscores (including an empty payload); otherwise I stop after `nan`.
This pins the existing Linux endpoint policy alongside the portable NaN value
policy. Darwin's previously observed `nan(+1)` endpoint differs; I explicitly
normalize that token boundary rather than silently preserving host variation.
Strict evaluation still rejects every nonempty unconsumed suffix.

I publish value and endpoint only after successful checked parsing. I retain
the fixed-capacity arithmetic, rounding proof, uint32 byte-view bound, and
allocation-free behavior. C-string adapters check length before narrowing and
copy binary64 bits with memcpy. They do not mutate locale or rounding state.

## My implementation and acceptance order

1. I add checked endpoint reporting, regenerate embedded source and verify exact
   value/endpoint pairs for ordinary decimal/hex, zero exponents, no-digit,
   whitespace, NUL, suffix, infinity and NaN cases. The value-only corpus remains
   unchanged and passing.
2. I share a C-string adapter between evaluator and both generated legacy
   runtimes. Prefix helpers preserve prefix behavior; strict cast retains its
   consumed-all check and error result. Header/build dependencies stay explicit.
3. I require meaningful C-seed/Stage1/Stage2 source conversion controls and
   evaluator strict acceptance/refusal controls, generated runtime checks,
   fresh bootstrap, parser/typechecker/evaluator gates and affected canonical
   parser/package regressions. Platform scope is stated per actual target.

Darwin managed sanitizer task7ba remains open. This companion does not complete
my full managed runtime, all-platform acceptance or release scope.
