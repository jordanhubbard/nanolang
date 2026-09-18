# My portable binary64 formatting evidence

I implement required formatter task `task_4fa62bcd01324cdfa0612d278d3bbaf0`
under my open managed-runtime parent. My base is merged PR648 (`297a8b3b`);
contract `e65e13b3` preceded implementation.

I decode exact binary64 bits into a bounded decimal integer coefficient, round
once to six significant digits with ties-to-even, and adjust an exponent carry
before selecting `%g` notation. Checked coefficient growth stays within 1100
one-byte decimal digits; multiplier/carry bounds are at most five/four. Fixed
notation indexes at most six initialized leading digits, and the longest output
fits well within the 32-byte buffer. Intermediate conversion allocates nothing;
final publication uses the existing managed allocator/status/cleanup contract.

My closed C-locale/default-rounding reference corpus contains 2,077 values:
29 selected boundaries plus 2,048 deterministic binary64 bit patterns. I generate
expected bytes with ordinary native snprintf, then run the identical data through
my actual core under native sanitizers, import-free Node and Wasmtime. Cases
include both zeros, min subnormal/max finite, infinities/NaNs, positive/negative
ties and carries across `9.999995e-5` and `999999.5` notation boundaries. All match.
I also compare the 29 selected values through real VM/native LLVM/Wasm programs,
including float helper calls, globals/reentry and final string reclamation.
Public native byte/table allocation failures recover cleanly; Wasm ordinary
allocation/cleanup and existing runtime failure/reuse groups remain green.

My frozen Linux ARM64 Make gates passed in
`/tmp/nanolang-managed-binary64-final.log`:

- 11 global methods, 7.107 seconds.
- 9 literal-string methods, 21.779 seconds.
- 2 package methods, 1.004 seconds.
- 3 core methods, 1.654 seconds.
- 19 emitted managed/reference methods, 15.505 seconds.
- 17 shared profile cases, 0.260 seconds for their containing method.

I remove the temporary floating CAST_STRING exclusion. Prior unused-helper
controls now assert successful publication for floating instructions/signatures;
CAST_FLOAT in string-bearing modules remains refused until the separately
recorded portable parser lands, and prior output remains preserved on refusal.
Scalar/literal-only profile decisions are unchanged. Native generated program IR
receives explicit ASan instrumentation, while the core is compiled with
ASan/UBSan and leak checking. Wasm tests enforce zero imports. I do not mutate
locale/rounding or execute historical failed artifacts, and I claim no Darwin,
nondefault-rounding or full managed-parent/release acceptance.
