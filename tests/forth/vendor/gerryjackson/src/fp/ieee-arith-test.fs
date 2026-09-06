(         Title:  IEEE Special Data Arithmetic Tests
           File:  ieee-arith~test.fs
         Author:  David N. Williams
        Version:  0.5.2
        License:  Public Domain
  Last revision:  October 2, 2009

The last revision date may reflect cosmetic changes not logged
below.

Version 0.5.2
 2Oct09 * Made compilation conditional for +INF, -INF, +NAN,
          -NAN.

Version 0.5.1
11Jul09 * Fixed tests with two NaN inputs so the output can be
          either of the two, as allowed by IEEE.  Solution
          suggested by Marcel Hendrix.

Version 0.5.0
29Jun09 * Started
 1Jul09 * Mostly finished.
 8Jul09 * Added FNAN? for FSQRT nans.
)

CR .( Running ieee-arith-test.fs)
CR .( --------------------------) CR

\ for pfe:
\ loadm ieeefp
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

[UNDEFINED] +inf [IF]
  1e 0e f/ fabs fconstant +inf
  +inf fnegate  fconstant -inf
[THEN]

[UNDEFINED] +nan [IF]
\ FABS is not superflous here, because the sign of 0/0 is not
\ specified by IEEE, and is actually different in Mac OS X
\ ppc/intel (+/-), both for gforth and pfe.  Note that IEEE-2008
\ does not require that 0/0 be a nan with zero load:
  0e 0e f/ fabs fconstant +nan
  +nan fnegate  fconstant -nan
[THEN]

\ The following huge kludge is just for testing sqrt(-1)!
\ But maybe FNAN? can also be used to improve some of the other
\ tests.
[UNDEFINED] fnan? [IF]
  : fdatum=  ( f: r1 r2 -- s: flag )  0e f~ ;

  \ maybe this can be improved
  : fnan?  ( f: r -- s: isNaN? )
    fdup +0 fdatum=
    fdup -0 fdatum= or
    fdup -inf fdatum= or
    fdup +inf fdatum= or IF fdrop false EXIT THEN
    +inf f< 0= ;
    
  \ borrowed from ieee-fprox-test.fs  
  t{ +0 +0     fdatum= -> true }t
  t{ +0 -0     fdatum= -> false }t
  t{ -0 +0     fdatum= -> false }t
  t{ -0 -0     fdatum= -> true }t
  t{  7e -2e   fdatum= -> false }t
  t{ -2e  7e   fdatum= -> false }t
  t{  7e  7e   fdatum= -> true }t
  t{  7e +inf  fdatum= -> false }t
  t{ +inf 7e   fdatum= -> false }t
  t{  7e -inf  fdatum= -> false }t
  t{ -inf 7e   fdatum= -> false }t
  t{ +inf +inf fdatum= -> true }t
  t{ +inf -inf fdatum= -> false }t
  t{ -inf +inf fdatum= -> false }t
  t{ -inf -inf fdatum= -> true }t
  t{ +nan 7e   fdatum= -> false }t
  t{ -nan 7e   fdatum= -> false }t
  t{  7e +nan  fdatum= -> false }t
  t{  7e -nan  fdatum= -> false }t
  t{ +nan +nan fdatum= -> true }t
  t{ -nan +nan fdatum= -> false }t
  t{ +nan -nan fdatum= -> false }t
  t{ -nan -nan fdatum= -> true }t
  t{ +inf +nan fdatum= -> false }t
  t{ -inf +nan fdatum= -> false }t
  t{ +inf -nan fdatum= -> false }t
  t{ -inf -nan fdatum= -> false }t
  t{ +nan +inf fdatum= -> false }t
  t{ -nan +inf fdatum= -> false }t
  t{ +nan -inf fdatum= -> false }t
  t{ -nan -inf fdatum= -> false }t
  t{ 1e +inf f< -> true }t
  t{ +0   fnan? -> false }t
  t{ -0   fnan? -> false }t
  t{ +inf fnan? -> false }t
  t{ -inf fnan? -> false }t
  t{  1e  fnan? -> false }t
  t{ +nan fnan? -> true }t
  t{ -nan fnan? -> true }t
  #errors @ 0= 

\ Note that the above is not airtight.  We really need to know
\ that the IEEE special data are correct.

[ELSE] true [THEN]
constant VALID-FNAN?-DEFINED

TESTING F+

\ IEEE-2008 6.3

t{ +0 +0 f+ -> +0 }t
t{ +0 -0 f+ -> +0 }t
t{ -0 +0 f+ -> +0 }t
t{ -0 -0 f+ -> -0 }t

t{ +nan  2e  f+ -> +nan }t
t{ -nan  2e  f+ -> -nan }t
t{  3e  +nan f+ -> +nan }t
t{  3e  -nan f+ -> -nan }t

t{ +nan +nan f+ -> +nan }t
t{ -nan +nan f+ fabs -> +nan }t
t{ +nan -nan f+ fabs -> +nan }t
t{ -nan -nan f+ -> -nan }t

t{  2e  +inf f+ -> +inf }t
t{ +inf  7e  f+ -> +inf }t
t{  2e  -inf f+ -> -inf }t
t{ -inf  7e  f+ -> -inf }t

t{ +nan +inf f+ -> +nan }t
t{ +inf +nan f+ -> +nan }t
t{ +nan -inf f+ -> +nan }t
t{ -inf +nan f+ -> +nan }t
t{ -nan +inf f+ -> -nan }t
t{ +inf -nan f+ -> -nan }t
t{ -nan -inf f+ -> -nan }t
t{ -inf -nan f+ -> -nan }t

t{ +inf +inf f+      -> +inf }t
t{ +inf -inf f+ fabs -> +nan }t
t{ -inf +inf f+ fabs -> +nan }t
t{ -inf -inf f+      -> -inf }t

TESTING F-

t{ +0 +0 f- -> +0 }t
t{ +0 -0 f- -> +0 }t
t{ -0 +0 f- -> -0 }t
t{ -0 -0 f- -> +0 }t

t{ +nan  2e  f- -> +nan }t
t{ -nan  2e  f- -> -nan }t
t{  3e  +nan f- -> +nan }t
t{  3e  -nan f- -> -nan }t

t{ +nan +nan f-      -> +nan }t
t{ -nan +nan f- fabs -> +nan }t
t{ +nan -nan f- fabs -> +nan }t
t{ -nan -nan f-      -> -nan }t

t{  2e  +inf f- -> -inf }t
t{ +inf  7e  f- -> +inf }t
t{  2e  -inf f- -> +inf }t
t{ -inf  7e  f- -> -inf }t

t{ +nan +inf f- -> +nan }t
t{ +inf +nan f- -> +nan }t
t{ +nan -inf f- -> +nan }t
t{ -inf +nan f- -> +nan }t
t{ -nan +inf f- -> -nan }t
t{ +inf -nan f- -> -nan }t
t{ -nan -inf f- -> -nan }t
t{ -inf -nan f- -> -nan }t

t{ +inf +inf f- fabs -> +nan }t
t{ +inf -inf f-      -> +inf }t
t{ -inf +inf f-      -> -inf }t
t{ -inf -inf f- fabs -> +nan }t

TESTING F*

t{ +0 +0 f* -> +0 }t
t{ +0 -0 f* -> -0 }t
t{ -0 +0 f* -> -0 }t
t{ -0 -0 f* -> +0 }t

t{ +0  2e f* -> +0 }t
t{ -0  2e f* -> -0 }t
t{ +0 -2e f* -> -0 }t
t{ -0 -2e f* -> +0 }t
t{  2e +0 f* -> +0 }t
t{  2e -0 f* -> -0 }t
t{ -2e +0 f* -> -0 }t
t{ -2e -0 f* -> +0 }t

t{ +nan  2e  f* -> +nan }t
t{ -nan  2e  f* -> -nan }t
t{  3e  +nan f* -> +nan }t
t{  3e  -nan f* -> -nan }t

t{ +nan +nan f*      -> +nan }t
t{ -nan +nan f* fabs -> +nan }t
t{ +nan -nan f* fabs -> +nan }t
t{ -nan -nan f*      -> -nan }t

t{  2e  +inf f* -> +inf }t
t{ +inf  7e  f* -> +inf }t
t{  2e  -inf f* -> -inf }t
t{ -inf  7e  f* -> -inf }t

t{ +nan +inf f* -> +nan }t
t{ +inf +nan f* -> +nan }t
t{ +nan -inf f* -> +nan }t
t{ -inf +nan f* -> +nan }t
t{ -nan +inf f* -> -nan }t
t{ +inf -nan f* -> -nan }t
t{ -nan -inf f* -> -nan }t
t{ -inf -nan f* -> -nan }t

t{ +inf +inf f* -> +inf }t
t{ +inf -inf f* -> -inf }t
t{ -inf +inf f* -> -inf }t

TESTING F/

t{ +0 +0 f/ fabs -> +nan }t
t{ +0 -0 f/ fabs -> +nan }t
t{ -0 +0 f/ fabs -> +nan }t
t{ -0 -0 f/ fabs -> +nan }t

t{ +0  2e f/ -> +0 }t
t{ -0  2e f/ -> -0 }t
t{ +0 -2e f/ -> -0 }t
t{ -0 -2e f/ -> +0 }t
t{  2e +0 f/ -> +inf }t
t{  2e -0 f/ -> -inf }t
t{ -2e +0 f/ -> -inf }t
t{ -2e -0 f/ -> +inf }t

t{ +nan  2e  f/ -> +nan }t
t{ -nan  2e  f/ -> -nan }t
t{  3e  +nan f/ -> +nan }t
t{  3e  -nan f/ -> -nan }t

t{ +nan +nan f/      -> +nan }t
t{ -nan +nan f/ fabs -> +nan }t
t{ +nan -nan f/ fabs -> +nan }t
t{ -nan -nan f/      -> -nan }t

t{  2e  +inf f/ -> +0 }t
t{ +inf  7e  f/ -> +inf }t
t{  2e  -inf f/ -> -0 }t
t{ -inf  7e  f/ -> -inf }t

t{ +nan +inf f/ -> +nan }t
t{ +inf +nan f/ -> +nan }t
t{ +nan -inf f/ -> +nan }t
t{ -inf +nan f/ -> +nan }t
t{ -nan +inf f/ -> -nan }t
t{ +inf -nan f/ -> -nan }t
t{ -nan -inf f/ -> -nan }t
t{ -inf -nan f/ -> -nan }t

t{ +inf +inf f/ fabs -> +nan }t
t{ +inf -inf f/ fabs -> +nan }t
t{ -inf +inf f/ fabs -> +nan }t

TESTING FSQRT

t{ +0   fsqrt -> +0 }t
t{ -0   fsqrt -> -0 }t
t{ +inf fsqrt -> +inf }t
t{ -inf fsqrt -> -1e fsqrt }t
t{ +nan fsqrt -> +nan }t
t{ -nan fsqrt -> -nan }t
VALID-FNAN?-DEFINED [IF]
t{ -1e  fsqrt fnan? -> true }t
[ELSE]
verbose @ [IF] .( NOT TESTING -1E FSQRT) cr [THEN]
[THEN]

?.errors ?.cr

CR
CR .( End of ieee-arith-test.fs) CR

