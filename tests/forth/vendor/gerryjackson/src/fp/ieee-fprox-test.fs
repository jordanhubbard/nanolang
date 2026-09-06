(   Title:  IEEE F~ Tests
     File:  ieee-fprox-test.fs
   Author:  David N. Williams
  Version:  0.5.1
  License:  Public Domain
  Revised:  July 1, 2009

Version 0.5.1
 1Jul09 * Inserted FABS in definitions of +inf and +0.

Version 0.5.0
29Jun09 * Started.
30Jun09 * Released.
)

CR .( Running ieee-fprox-test.fs)
CR .( --------------------------) CR

\ Causes pfe to load floating point:
s" FLOATING-EXT" environment? [IF] drop [THEN]

\ s" ttester.fs" included
decimal
true verbose !

: ?.cr  ( -- )  verbose @ IF cr THEN ;
?.cr

\ The ttester default for EXACT? is TRUE.  Uncomment the
\ following line if your system needs it to be FALSE:
\ SET-NEAR

variable #errors  0 #errors !

:noname  ( c-addr u -- )
(
Display an error message followed by the line that had the
error.
)
  1 #errors +! error1 ; error-xt !

: ?.errors  ( -- )  verbose @ IF ." #ERRORS: " #errors @ . THEN ;

[UNDEFINED] \\ [IF]  \ for debugging
: \\  ( -- )  -1 parse 2drop BEGIN refill 0= UNTIL ; [THEN]

\ FABS should be superfluous in these:

0e fabs       fconstant +0 
+0 fnegate    fconstant -0
1e 0e f/ fabs fconstant +inf
+inf fnegate  fconstant -inf

\ FABS is not superflous here, because the sign of 0/0 is not
\ specified by IEEE, and is actually different in Mac OS X
\ ppc/intel (+/-), both for gforth and pfe:

0e 0e f/ fabs fconstant +nan
+nan fnegate  fconstant -nan

TESTING equality of floating-point encoding

t{ +0 +0  +0 f~ -> true }t
t{ +0 -0  +0 f~ -> false }t
t{ -0 +0  +0 f~ -> false }t
t{ -0 -0  +0 f~ -> true }t

t{  7e -2e  +0 f~ -> false }t
t{ -2e  7e  +0 f~ -> false }t
t{  7e  7e  +0 f~ -> true }t
   
t{  7e +inf  +0 f~ -> false }t
t{ +inf 7e   +0 f~ -> false }t
t{  7e -inf  +0 f~ -> false }t
t{ -inf 7e   +0 f~ -> false }t

t{ +inf +inf  +0 f~ -> true }t
t{ +inf -inf  +0 f~ -> false }t
t{ -inf +inf  +0 f~ -> false }t
t{ -inf -inf  +0 f~ -> true }t

t{ +nan 7e   +0 f~ -> false }t
t{ -nan 7e   +0 f~ -> false }t
t{  7e +nan  +0 f~ -> false }t
t{  7e -nan  +0 f~ -> false }t

t{ +nan +nan  +0 f~ -> true }t
t{ -nan +nan  +0 f~ -> false }t
t{ +nan -nan  +0 f~ -> false }t
t{ -nan -nan  +0 f~ -> true }t

t{ +inf +nan  +0 f~ -> false }t
t{ -inf +nan  +0 f~ -> false }t
t{ +inf -nan  +0 f~ -> false }t
t{ -inf -nan  +0 f~ -> false }t

t{ +nan +inf  +0 f~ -> false }t
t{ -nan +inf  +0 f~ -> false }t
t{ +nan -inf  +0 f~ -> false }t
t{ -nan -inf  +0 f~ -> false }t


t{ +0 +0  -0 f~ -> true }t
t{ +0 -0  -0 f~ -> false }t
t{ -0 +0  -0 f~ -> false }t
t{ -0 -0  -0 f~ -> true }t

t{  7e -2e  -0 f~ -> false }t
t{ -2e  7e  -0 f~ -> false }t
t{  7e  7e  -0 f~ -> true }t

t{  7e +inf  -0 f~ -> false }t
t{ +inf 7e   -0 f~ -> false }t
t{  7e -inf  -0 f~ -> false }t
t{ -inf 7e   -0 f~ -> false }t

t{ +inf -inf  -0 f~ -> false }t
t{ -inf +inf  -0 f~ -> false }t
t{ -inf -inf  -0 f~ -> true }t
t{ +inf +inf  -0 f~ -> true }t

t{ +nan 7e   -0 f~ -> false }t
t{ -nan 7e   -0 f~ -> false }t
t{  7e +nan  -0 f~ -> false }t
t{  7e -nan  -0 f~ -> false }t

t{ +nan +nan  -0 f~ -> true }t
t{ -nan +nan  -0 f~ -> false }t
t{ +nan -nan  -0 f~ -> false }t
t{ -nan -nan  -0 f~ -> true }t

t{ +inf +nan  -0 f~ -> false }t
t{ -inf +nan  -0 f~ -> false }t
t{ +inf -nan  -0 f~ -> false }t
t{ -inf -nan  -0 f~ -> false }t

t{ +nan +inf  -0 f~ -> false }t
t{ -nan +inf  -0 f~ -> false }t
t{ +nan -inf  -0 f~ -> false }t
t{ -nan -inf  -0 f~ -> false }t

TESTING absolute tolerance

t{ +0 +0  7e f~ -> true }t
t{ +0 -0  7e f~ -> true }t
t{ -0 +0  7e f~ -> true }t
t{ -0 -0  7e f~ -> true }t

t{  7e +inf  7e f~ -> false }t
t{ +inf 7e   7e f~ -> false }t
t{  7e -inf  7e f~ -> false }t
t{ -inf 7e   7e f~ -> false }t

t{ +inf +inf  7e f~ -> false }t
t{ +inf -inf  7e f~ -> false }t
t{ -inf +inf  7e f~ -> false }t
t{ -inf -inf  7e f~ -> false }t

t{ +nan 7e   7e f~ -> false }t
t{ -nan 7e   7e f~ -> false }t
t{  7e +nan  7e f~ -> false }t
t{  7e -nan  7e f~ -> false }t

t{ +nan +nan  7e f~ -> false }t
t{ -nan +nan  7e f~ -> false }t
t{ +nan -nan  7e f~ -> false }t
t{ -nan -nan  7e f~ -> false }t

t{ +inf +nan  7e f~ -> false }t
t{ -inf +nan  7e f~ -> false }t
t{ +inf -nan  7e f~ -> false }t
t{ -inf -nan  7e f~ -> false }t

t{ +nan +inf  7e f~ -> false }t
t{ -nan +inf  7e f~ -> false }t
t{ +nan -inf  7e f~ -> false }t
t{ -nan -inf  7e f~ -> false }t

t{ +0 +0  +inf f~ -> true }t
t{ +0 -0  +inf f~ -> true }t
t{ -0 +0  +inf f~ -> true }t
t{ -0 -0  +inf f~ -> true }t

t{  7e -2e  +inf f~ -> true }t
t{ -2e  7e  +inf f~ -> true }t
t{  7e  7e  +inf f~ -> true }t

t{  7e +inf  +inf f~ -> false }t
t{ +inf 7e   +inf f~ -> false }t
t{  7e -inf  +inf f~ -> false }t
t{ -inf 7e   +inf f~ -> false }t

t{ +inf +inf  +inf f~ -> false }t
t{ +inf -inf  +inf f~ -> false }t
t{ -inf +inf  +inf f~ -> false }t
t{ -inf -inf  +inf f~ -> false }t

t{ +nan 7e   +inf f~ -> false }t
t{ -nan 7e   +inf f~ -> false }t
t{  7e +nan  +inf f~ -> false }t
t{  7e -nan  +inf f~ -> false }t

t{ +nan +nan  +inf f~ -> false }t
t{ -nan +nan  +inf f~ -> false }t
t{ +nan -nan  +inf f~ -> false }t
t{ -nan -nan  +inf f~ -> false }t

t{ +inf +nan  +inf f~ -> false }t
t{ -inf +nan  +inf f~ -> false }t
t{ +inf -nan  +inf f~ -> false }t
t{ -inf -nan  +inf f~ -> false }t

t{ +nan +inf  +inf f~ -> false }t
t{ -nan +inf  +inf f~ -> false }t
t{ +nan -inf  +inf f~ -> false }t
t{ -nan -inf  +inf f~ -> false }t

t{ +0 +0  +nan f~ -> false }t
t{ +0 -0  +nan f~ -> false }t
t{ -0 +0  +nan f~ -> false }t
t{ -0 -0  +nan f~ -> false }t

t{  7e -2e  +nan f~ -> false }t
t{ -2e  7e  +nan f~ -> false }t
t{  7e  7e  +nan f~ -> false }t

t{  7e +inf  +nan f~ -> false }t
t{ +inf 7e   +nan f~ -> false }t
t{  7e -inf  +nan f~ -> false }t
t{ -inf 7e   +nan f~ -> false }t

t{ +inf +inf  +nan f~ -> false }t
t{ +inf -inf  +nan f~ -> false }t
t{ -inf +inf  +nan f~ -> false }t
t{ -inf -inf  +nan f~ -> false }t

t{ +nan 7e   +nan f~ -> false }t
t{ -nan 7e   +nan f~ -> false }t
t{  7e +nan  +nan f~ -> false }t
t{  7e -nan  +nan f~ -> false }t

t{ +nan +nan  +nan f~ -> false }t
t{ -nan +nan  +nan f~ -> false }t
t{ +nan -nan  +nan f~ -> false }t
t{ -nan -nan  +nan f~ -> false }t

t{ +inf +nan  +nan f~ -> false }t
t{ -inf +nan  +nan f~ -> false }t
t{ +inf -nan  +nan f~ -> false }t
t{ -inf -nan  +nan f~ -> false }t

t{ +nan +inf  +nan f~ -> false }t
t{ -nan +inf  +nan f~ -> false }t
t{ +nan -inf  +nan f~ -> false }t
t{ -nan -inf  +nan f~ -> false }t

TESTING relative tolerance

t{ +0 +0  -7e f~ -> false }t
t{ +0 -0  -7e f~ -> false }t
t{ -0 +0  -7e f~ -> false }t
t{ -0 -0  -7e f~ -> false }t

t{  7e +inf  -7e f~ -> false }t
t{ +inf 7e   -7e f~ -> false }t
t{  7e -inf  -7e f~ -> false }t
t{ -inf 7e   -7e f~ -> false }t

t{ +inf +inf  -7e f~ -> false }t
t{ +inf -inf  -7e f~ -> false }t
t{ -inf +inf  -7e f~ -> false }t
t{ -inf -inf  -7e f~ -> false }t

t{ +nan 7e   -7e f~ -> false }t
t{ -nan 7e   -7e f~ -> false }t
t{  7e +nan  -7e f~ -> false }t
t{  7e -nan  -7e f~ -> false }t

t{ +nan +nan  -7e f~ -> false }t
t{ -nan +nan  -7e f~ -> false }t
t{ +nan -nan  -7e f~ -> false }t
t{ -nan -nan  -7e f~ -> false }t

t{ +inf +nan  -7e f~ -> false }t
t{ -inf +nan  -7e f~ -> false }t
t{ +inf -nan  -7e f~ -> false }t
t{ -inf -nan  -7e f~ -> false }t

t{ +nan +inf  -7e f~ -> false }t
t{ -nan +inf  -7e f~ -> false }t
t{ +nan -inf  -7e f~ -> false }t
t{ -nan -inf  -7e f~ -> false }t

t{ +0 +0  -inf f~ -> false }t
t{ +0 -0  -inf f~ -> false }t
t{ -0 +0  -inf f~ -> false }t
t{ -0 -0  -inf f~ -> false }t

t{  7e -2e  -inf f~ -> true }t
t{ -2e  7e  -inf f~ -> true }t
t{  7e  7e  -inf f~ -> true }t

t{  7e +inf  -inf f~ -> false }t
t{ +inf 7e   -inf f~ -> false }t
t{  7e -inf  -inf f~ -> false }t
t{ -inf 7e   -inf f~ -> false }t

t{ +inf +inf  -inf f~ -> false }t
t{ +inf -inf  -inf f~ -> false }t
t{ -inf +inf  -inf f~ -> false }t
t{ -inf -inf  -inf f~ -> false }t

t{ +nan 7e   -inf f~ -> false }t
t{ -nan 7e   -inf f~ -> false }t
t{  7e +nan  -inf f~ -> false }t
t{  7e -nan  -inf f~ -> false }t

t{ +nan +nan  -inf f~ -> false }t
t{ -nan +nan  -inf f~ -> false }t
t{ +nan -nan  -inf f~ -> false }t
t{ -nan -nan  -inf f~ -> false }t

t{ +inf +nan  -inf f~ -> false }t
t{ -inf +nan  -inf f~ -> false }t
t{ +inf -nan  -inf f~ -> false }t
t{ -inf -nan  -inf f~ -> false }t

t{ +nan +inf  -inf f~ -> false }t
t{ -nan +inf  -inf f~ -> false }t
t{ +nan -inf  -inf f~ -> false }t
t{ -nan -inf  -inf f~ -> false }t

t{ +0 +0  +nan f~ -> false }t
t{ +0 -0  +nan f~ -> false }t
t{ -0 +0  +nan f~ -> false }t
t{ -0 -0  +nan f~ -> false }t

t{  7e -2e  -nan f~ -> false }t
t{ -2e  7e  -nan f~ -> false }t
t{  7e  7e  -nan f~ -> false }t

t{  7e +inf  -nan f~ -> false }t
t{ +inf 7e   -nan f~ -> false }t
t{  7e -inf  -nan f~ -> false }t
t{ -inf 7e   -nan f~ -> false }t

t{ +inf +inf  -nan f~ -> false }t
t{ +inf -inf  -nan f~ -> false }t
t{ -inf +inf  -nan f~ -> false }t
t{ -inf -inf  -nan f~ -> false }t

t{ +nan 7e   -nan f~ -> false }t
t{ -nan 7e   -nan f~ -> false }t
t{  7e +nan  -nan f~ -> false }t
t{  7e -nan  -nan f~ -> false }t

t{ +nan +nan  -nan f~ -> false }t
t{ -nan +nan  -nan f~ -> false }t
t{ +nan -nan  -nan f~ -> false }t
t{ -nan -nan  -nan f~ -> false }t

t{ +inf +nan  -nan f~ -> false }t
t{ -inf +nan  -nan f~ -> false }t
t{ +inf -nan  -nan f~ -> false }t
t{ -inf -nan  -nan f~ -> false }t

t{ +nan +inf  -nan f~ -> false }t
t{ -nan +inf  -nan f~ -> false }t
t{ +nan -inf  -nan f~ -> false }t
t{ -nan -inf  -nan f~ -> false }t

?.errors ?.cr

CR .( End of ieee-fprox-test.fs) CR