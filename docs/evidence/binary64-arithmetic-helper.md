# My shared scalar arithmetic helper evidence

I completed only stage 1 of `task_50090616040044dfa3bba1ab93a9f6d1`.
That task and full reconstruction remain open. Policy/inventory contracts
`baa865b4` and `0cfbd4cf` preceded production `b53be06d`; review correction
`3345afdb` uses the existing nano_rt_ prefix instead of colliding with ordinary
f64_* source function names. No existing backend calls this header yet.

My shared C header has four scalar binary helpers, a volatile double operation
boundary, exact integer NaN normalization and signed-zero divisor precedence.
Its emitted standalone text is generated from the entire header, including
target and optimization guards. The generation check and actual C string output
both match the canonical header exactly. I add no float reconstruction admission.

My first frozen harness `a83b5683` passed two methods with GCC in 0.883 seconds
and Clang in 1.161 seconds. After both runs ended, I added the contraction-enabled
mode and froze final harness `673392ce`; final GCC passed in 1.131 seconds and
Clang in 1.382 seconds. No source or test changed while a gate was active.

Each final compiler run executes direct and emitted helper copies at O0, O2,
O3 with explicit ffp-contract=fast, and O2 with LTO. Normal modes explicitly
set ffp-contract=off and fno-fast-math. All modes use strict warnings and
ASan/UBSan with nonrecovering errors; Clang LTO uses lld. Each executable passed
133 exact arithmetic bit checks. Runtime volatile input bits prevent constant
substitution from replacing the rounding cases.

I tested signed quiet/signaling NaNs in both operand positions, differing NaN
payloads, invalid infinity operations, signed-zero divisor precedence including
NaN numerators, unchanged input/transport/negation bits, signed-zero results,
halfway ties, gradual subnormal results, overflow, division rounding and a
multiply/subtract chain that distinguishes contraction. I check FE_TONEAREST
without changing the caller's environment. Direct C symbols nl_f64_add/sub/mul/div
coexist with the runtime helpers. Actual paired source-name coverage remains
required when stage 3 integrates the emitters; this scaffold is not that gate.

Compile-only controls reject fast-math, finite-math-only, a simulated extended
evaluation target and an incompatible double precision declaration. The latter
two are isolated macro controls, not claims of running another target. Current
compiler results qualify Linux ARM64 GCC/Clang only, not universal hosts.

I retained exact source/log hashes in [my manifest](binary64-arithmetic-helper.json)
and per-command fixtures under `/tmp/nano-binary64-arithmetic-*`. No gate failed.
The four logs use `/tmp/nanolang-binary64-arithmetic-` plus `gcc.log`, `clang.log`,
`final-gcc.log` and `final-clang.log`. Immutable PR720 producer/import hashes
still match; those compilers were not used or rebuilt in these helper gates.

Stage 2 must integrate VM/native/LLVM/Wasm and test all scalar dispatch routes.
Stage 3 must integrate main/optimized interpreter and both legacy C emitters,
repair raw scalar division and qualify a fresh bootstrap. Actual driver flags
and generated products remain obligations of those stages. Array task3717 and
external math contracts retain the explicit scope in my policy document.
I make no backend-wide, source-wide, reconstruction or release completion claim.

A final static contract check tightened storage equality to explicit eight-byte
double and uint64_t at `e0e31b9f`. With the same frozen `673392ce` harness, all
two methods passed again on GCC in 1.158 seconds and Clang in 1.335 seconds,
including all four optimization modes and direct/emitted forms. I retain prior
and final source/log hashes; final logs end in `storage-gcc.log` and
`storage-clang.log`. No live gate overlapped the correction.
