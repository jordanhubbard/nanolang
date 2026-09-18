# My portable binary64 formatting contract

I record required task `task_4fa62bcd01324cdfa0612d278d3bbaf0` under my open
managed-runtime parent, after merged scalar formatting PR648. My VM uses
`snprintf("%g")`, with six significant digits, in the closed C locale and default
round-to-nearest/ties-to-even environment. I do not mutate global locale or
rounding state. Hosts that change them remain outside this closed contract.

I decode the binary64 sign, exponent and significand as integers. For finite
nonzero values I construct an exact decimal coefficient: multiply the integer
significand by powers of two for nonnegative binary exponent, or powers of five
and retain the negative decimal scale otherwise. Binary exponent magnitude is
at most 1074; the exact coefficient fits fewer than 800 decimal digits. I reserve
1100 checked decimal digits, with bounded small-integer carry, and no heap
allocation for intermediate conversion.

I round the exact coefficient once to six significant digits, using remaining
digits to distinguish above-half from exact ties and the retained final digit
for ties-to-even. A carry advances the decimal exponent before I select `%g`
fixed versus scientific notation (scientific below -4 or at least 6). I remove
insignificant fractional trailing zeros, preserve needed integer zeros and
pad exponent magnitude to at least two digits. Signed zero and nonfinite values
retain the ordinary VM reference spelling, including sign. Final bytes use the
existing checked managed allocator; integer/string ownership rules stay intact.

I require direct review of coefficient bounds, rounding and output capacity,
then actual native/Wasm equality with ordinary VM/native snprintf references.
Cases include min subnormal/max finite, both zero signs, infinities/NaNs,
rounding ties, carry crossing notation boundaries, deterministic ordinary
binary64 bit-pattern samples, and native/public allocation cleanup/recovery.
Wasm must remain import-free; a native-generated reference corpus may be fed to
both targets without supplying a formatting host import. I preserve previous
failure artifacts without executing them.

Only after those gates do I lift the bounded floating CAST_STRING exclusion.
String CAST_FLOAT remains refused pending parser task
`task_4d2f69a19d754ac88876f93a0913d1fb`. Full managed parent/release acceptance
remains open. I claim only targets actually exercised.
