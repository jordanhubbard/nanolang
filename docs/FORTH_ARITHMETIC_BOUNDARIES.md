# My Forth arithmetic boundaries

I continue `task_94534bf8291349a380ed1c88a93e3cb0` and the independently
identified division task `task_bbeaab3cfaa9cbc34398272b8a1947ea` from actual
PR943 merge `92720ef41d5050b54214b27a0b91d8d0c51d1ea4`. This is my policy and
source audit before correction. I do not execute known undefined boundaries.
My earlier signed packing, D2* and decimal repairs remain qualified separately.

## Representation and accepted arithmetic

My cell contains 64 bits. My double is low cell followed by high cell on the
Forth stack, with 128-bit two's-complement interpretation when signed. I retain
my existing uint64-to-int64 cell-bit conversion policy on qualified platforms.
Unsigned intermediates define wraparound; I do not rely on signed C overflow.

| Words | Result policy | Existing implementation and required work |
| --- | --- | --- |
| +, -, *, 1+, 1-, 2*, NEGATE, ABS, +! | modulo2^64; minimum negation/absolute-value retains its bit pattern | Existing VM unsigned arithmetic/minimum guard and Forth definitions already implement this; qualify boundary controls without changing VM semantics. |
| D+, D-, DNEGATE, DABS, M+, D2* | modulo2^128; DABS minimum retains its unsigned magnitude bit pattern | Replace remaining signed int128 addition/subtraction/negation with unsigned128 operations. Sign-extend M+ input before unsigned addition. D2* already uses unsigned carry. |
| 2/, D2/ | arithmetic shift by one, rounded toward negative infinity | D2/ must explicitly move high-to-low carry and reinsert the high sign bit with unsigned shifts. Existing 2/ VM behavior stays unchanged and gets exact controls. |
| D<, D=, D0<, D0=, DMIN, DMAX, DU< | signed/unsigned comparison as named | Exact signed packing is representable; no overflowing arithmetic is needed. |
| M*, UM* | exact signed/unsigned64-by64 product into128 bits | Existing VM wide multiply results remain exact; cover all sign/extrema combinations. |
| D>S | retain low cell | Existing explicit DROP policy remains; this is not checked division. |
| >NUMBER and pictured unsigned division | modulo unsigned accumulation / exact quotient-remainder | Existing unsigned operations remain; exercise minimum-double D./D.R to cover DABS magnitude. |

The full Forth task includes other word sets. This table does not claim that
floating transcendental arithmetic, storage addresses or loop control has been
qualified by integer tests.

## Checked division

I keep SM/REM truncation toward zero and FM/MOD flooring. /MOD, / and MOD remain
floored, and */MOD and */ continue composing exact M* with FM/MOD. My Forth
owning route chooses catchable -10 for a zero divisor and -11 when the final
quotient does not fit its declared cell range. These choices replace previously
inconsistent ambiguous-condition behavior; they are an intentional Forth policy
change, not a change to ordinary NanoVM division.

The standard leaves zero-divisor and out-of-range signed quotients ambiguous
for [SM/REM](https://forth-standard.org/standard/core/SMDivREM) and
[FM/MOD](https://forth-standard.org/standard/core/FMDivMOD). I select the assigned
[exception codes](https://forth-standard.org/standard/exception) explicitly.

I compute signed division using unsigned magnitudes, never signed min/-1:

1. Form the exact128-bit numerator bit pattern. If negative, obtain its
   magnitude with unsigned complement-plus-one. Obtain divisor magnitude using
   my existing safe64-bit unsigned absolute-value helper.
2. Divide unsigned128 by unsigned64. The quotient magnitude is at most2^127;
   the remainder is less than the divisor magnitude.
3. For a negative floored result with nonzero remainder, increment quotient
   magnitude and use divisor-magnitude minus remainder. Check the increment
   before performing it. Symmetric remainder takes the numerator sign; floored
   remainder takes the divisor sign.
4. Admit quotient magnitude at most2^63 for a negative result or2^63-1 for a
   nonnegative result. Materialize minimum as INT64_MIN explicitly; do not negate
   an unrepresentable signed value. Remainder magnitude is at most2^63-1.
5. Publish remainder then quotient only after every arithmetic check succeeds.

This handles minimum128 divided by -1 without evaluating the undefined signed
operation. It also rejects ordinary oversized quotients that previously narrowed
silently. UM/MOD uses unsigned quotient arithmetic but rejects quotient above
UINT64_MAX and zero divisor with the same selected codes.

My current /MOD primitive directly composes VM division/remainder and inherits
VM's zero-divisor and minimum/-1 results. I replace only this Forth primitive
with an owning runtime operation using sign extension and the common checked
floored helper. Existing / and MOD definitions continue calling it. An appended
internal host-operation identifier preserves earlier identifier values; this is
not a new NanoISA opcode or public ABI.

## M*/ and double conversion

My existing M*/ decomposes an exact192-bit magnitude product into64-bit limbs.
I retain that algorithm and its nonzero signed-divisor extension, including
negative divisors already accepted. The standard requires a positive divisor
but permits implementation behavior outside it; my extension is explicit.
[M*/](https://forth-standard.org/standard/double/MTimesDiv) defines a signed
double quotient, so I check positive magnitude against2^127-1 and negative
magnitude against2^127 after any floor increment. A wider quotient from the
192/64 helper becomes -11, not a generic failure or truncated double. I check
before incrementing; negative publication uses unsigned complement-plus-one.
Zero divisor becomes -10. Product limb carries remain exact and unsigned.

My adjacent F>D path currently casts any double to signed int128. I include its
boundary guard because it creates the same signed-double representation:
nonfinite input throws -46; finite input outside [-2^127,2^127) throws -11.
Inside that interval, the existing truncation toward zero is representable.
I use exactly representable binary power-of-two bounds, not a rounded conversion
of INT128_MAX. After numeric checks I reserve two available data-stack slots before
publishing either cell; insufficient space throws -3. This explicit precondition
repairs the statically found one-slot partial-publication path. D>F remains its existing finite conversion with ordinary precision
loss. These guards do not claim a complete floating-point word-set audit.

## Failure, lifetime and implementation boundaries

I use the existing session throw field and halt protocol. Arithmetic failure
publishes no quotient/remainder pair or converted result. CATCH restores its
existing data/return/float/source/control depths; I do not strengthen its
contract to promise arbitrary restored stack contents. The caller can discard
restored arguments and execute a fresh successful operation. Uncaught errors
remain actual session errors and cannot masquerade as a successful zero result.
I preserve existing underflow behavior separately from divisor-zero checks:
a failed pop must not be mislabeled -10.

All helpers have fixed-size scalar locals, no new heap allocation, no mutable
global state, and no unbounded loop. I audit each scalar conversion, shift,
quotient adjustment and product carry before source review. No generic FFI,
VM arithmetic handler, source compiler or generated backend change is needed.

## Acceptance before closing the arithmetic tasks

I add independent exact two-cell results for zero, allones, sign boundaries,
minimum and maximum, cross-cell carries and wraparound in every modular word.
I cover all quotient/remainder sign combinations and exact/nonexact division;
minimum128/-1 and values immediately inside/outside each signed/unsigned quotient
range; floor adjustment that crosses the permitted boundary; zero divisors;
192-bit M*/ intermediate products and signed-double result endpoints.
F>D controls include signed zero, fractions, exact lower bound, representable
neighbors of both bounds, infinities and NaNs. I check CATCH codes/depths and
fresh recovery independently of direct failure. I retain unchanged original
session, Core and Jackson Double assertions, and existing float conversion
neighbors. No faulty pre-correction boundary is run.

Fresh both-host ordinary and strict supported UBSan configurations must retain
all inputs, actual tool selectors, raw outputs and first terminals. I capture
inner executables before Make cleanup for this new checkpoint; I do not relabel
the deleted binaries from PR943 as recovered. Actual source/fixture review
precedes execution. Full Forth and 5.1 parents remain open until their remaining
requirements are independently qualified and merged.
