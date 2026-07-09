\ test_arithmetic.fs — Core arithmetic tests (ANS Forth / Forth 2012 subset)
\ Based on the forth-standard.org test suite, adapted for NanoLang Forth.
\ Floored division semantics (Forth 83 / ANS FM/MOD).

testing +
T{ 0 5 + -> 5 }T
T{ -7 3 + -> -4 }T
T{ 0 -5 + -> -5 }T
T{ -7 -3 + -> -10 }T
T{ 0 0 + -> 0 }T
T{ 1 2 + -> 3 }T

testing -
T{ 0 5 - -> -5 }T
T{ 5 3 - -> 2 }T
T{ -7 3 - -> -10 }T
T{ 0 -5 - -> 5 }T
T{ 5 5 - -> 0 }T

testing *
T{ 0 0 * -> 0 }T
T{ 0 1 * -> 0 }T
T{ 1 2 * -> 2 }T
T{ 3 3 * -> 9 }T
T{ -3 3 * -> -9 }T
T{ 3 -3 * -> -9 }T
T{ -3 -3 * -> 9 }T

testing / (floored)
T{ 1 1 / -> 1 }T
T{ 2 1 / -> 2 }T
T{ -1 1 / -> -1 }T
T{ 1 -1 / -> -1 }T
T{ 2 2 / -> 1 }T
T{ -6 3 / -> -2 }T
T{ -7 2 / -> -4 }T

testing mod (floored)
T{ 7 3 mod -> 1 }T
T{ -7 3 mod -> 2 }T
T{ 7 -3 mod -> -2 }T
T{ 0 3 mod -> 0 }T

testing /mod (floored: rem quot)
T{ 7 2 /mod -> 1 3 }T
T{ -7 2 /mod -> 1 -4 }T
T{ 7 -2 /mod -> -1 -4 }T

testing */
T{ 2 3 4 */ -> 1 }T
T{ 2 3 2 */ -> 3 }T
T{ -5 2 3 */ -> -4 }T

testing */mod
T{ 5 3 2 */mod -> 1 7 }T

testing abs
T{ 0 abs -> 0 }T
T{ 1 abs -> 1 }T
T{ -1 abs -> 1 }T
T{ -100 abs -> 100 }T

testing negate
T{ 0 negate -> 0 }T
T{ 1 negate -> -1 }T
T{ -1 negate -> 1 }T
T{ 2 negate -> -2 }T

testing max
T{ 1 2 max -> 2 }T
T{ 2 1 max -> 2 }T
T{ -1 0 max -> 0 }T
T{ 0 -1 max -> 0 }T
T{ 1 1 max -> 1 }T

testing min
T{ 1 2 min -> 1 }T
T{ 2 1 min -> 1 }T
T{ -1 0 min -> -1 }T
T{ 0 -1 min -> -1 }T
T{ 1 1 min -> 1 }T

testing 1+ 1-
T{ 0 1+ -> 1 }T
T{ -1 1+ -> 0 }T
T{ 1 1+ -> 2 }T
T{ 0 1- -> -1 }T
T{ 1 1- -> 0 }T
T{ 2 1- -> 1 }T

testing 2* 2/
T{ 0 2* -> 0 }T
T{ 1 2* -> 2 }T
T{ -1 2* -> -2 }T
T{ 3 2* -> 6 }T
T{ 0 2/ -> 0 }T
T{ 2 2/ -> 1 }T
T{ 4 2/ -> 2 }T
T{ -2 2/ -> -1 }T

testing fm/mod (floored: same as /mod)
T{ 7 2 fm/mod -> 1 3 }T
T{ -7 2 fm/mod -> 1 -4 }T

testing sm/rem (symmetric: truncated toward zero)
T{ 7 2 sm/rem -> 1 3 }T
T{ -7 2 sm/rem -> -1 -3 }T
T{ 7 -2 sm/rem -> 1 -3 }T

testing m*
T{ 0 0 m* -> 0 0 }T
T{ 1 1 m* -> 1 0 }T
T{ 2 3 m* -> 6 0 }T
T{ -1 1 m* -> -1 -1 }T

testing um*
T{ 0 0 um* -> 0 0 }T
T{ 1 1 um* -> 1 0 }T
T{ 3 4 um* -> 12 0 }T

test-summary
