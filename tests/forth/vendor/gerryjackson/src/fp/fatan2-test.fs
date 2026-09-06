(   Title:  FATAN2 tests
     File:  fatan2-test.fs
   Author:  David N. Williams
  Version:  0.1.0
  License:  Public Domain
  Revised:  March 27, 2009

Version 0.1.0
14Mar09 * Started.
19Mar09 * Added angles to complete the circle.
23Mar09 * Added definitions of fractions of pi in terms of pi,
	  which avoids most of the decimal to binary
	  conversions.  Suggested by Bernd Paysan.
27Mar09 * Added link to Krishna Myneni's kforth version.

This file tests the ISO C / Single UNIX 3 specification for
the ANS Forth Floating-Ext word:
)
\ FATAN2 ( f: y x -- radians ) or ( y x -- radians )
(
It expects the return value to be the principal angle, and
includes a number of tests with +/-0, +/-NaN, and +/-Inf that
are not required by the ANS Forth 94 standard.

A version that works with kForth can be found here:

  ftp://ccreweb.org/software/kforth/examples/fatan2-test.4th
)

CR .( Running fatan2-test.fs)
CR .( ----------------------) CR

\ for pfe:
s" FLOATING-EXT" environment? [IF] drop [THEN]

[UNDEFINED]  NaN [IF]  0e 0e f/ fconstant  NaN [THEN]
[UNDEFINED] +Inf [IF]  1e 0e f/ fconstant +Inf [THEN]
[UNDEFINED] -Inf [IF] -1e 0e f/ fconstant -Inf [THEN]

\ s" ttester.fs" included
true verbose !
decimal

\ The ttester default for EXACT? is TRUE.  Uncomment the
\ following line if your system needs it to be FALSE:
\ SET-NEAR

variable #errors    0 #errors !

:noname  ( c-addr u -- )
(
Display an error message followed by the line that had the
error.
)
  1 #errors +! error1 ; error-xt !

[UNDEFINED] \\ [IF]
  : \\  ( -- )  -1 parse 2drop BEGIN refill 0= UNTIL ; [THEN]

[UNDEFINED] pi [IF]
 0.3141592653589793238463E1 fconstant  pi
[THEN]
[UNDEFINED] -pi [IF]
 pi fnegate fconstant -pi
[THEN]

false [IF]
 0.7853981633974483096157E0 fconstant   pi/4
-0.7853981633974483096157E0 fconstant  -pi/4
 0.1570796326794896619231E1 fconstant   pi/2
-0.1570796326794896619231E1 fconstant  -pi/2
 0.4712388980384689857694E1 fconstant  3pi/2
 0.2356194490192344928847E1 fconstant  3pi/4
-0.2356194490192344928847E1 fconstant -3pi/4
[ELSE]
 pi 4e f/   fconstant   pi/4
-pi 4e f/   fconstant  -pi/4
 pi 2e f/   fconstant   pi/2
-pi 2e f/   fconstant  -pi/2
 pi/2 3e f* fconstant  3pi/2
 pi/4 3e f* fconstant  3pi/4
-pi/4 3e f* fconstant -3pi/4
[THEN]

verbose @ [IF]
:noname  ( -- fp.separate? )
  depth >r 1e depth >r fdrop 2r> = ; execute
cr .( floating-point and data stacks )
[IF] .( *separate*) [ELSE] .( *not separate*) [THEN]
cr
[THEN]

testing normal values

\ Input x-y values taken from Ed, comp.lang.forth,
\ "Re: Ambiguity in FATAN2", 14Mar09, 17Mar09:
\   y   x              rad          deg
t{  0e  1e fatan2 ->   0e   }t    \   0
t{  1e  1e fatan2 ->   pi/4 }t    \  45
t{  1e  0e fatan2 ->   pi/2 }t    \  90
t{ -1e -1e fatan2 -> -3pi/4 }t    \ 135
t{  0e -1e fatan2 ->   pi   }t    \ 180
t{ -1e  1e fatan2 ->  -pi/4 }t    \ 225
t{ -1e  0e fatan2 ->  -pi/2 }t    \ 270
t{ -1e  1e fatan2 ->  -pi/4 }t    \ 315

testing Single UNIX 3 special values spec

\ ISO C / Single UNIX Specification Version 3:
\   http://www.unix.org/single_unix_specification/
\ Select "Topic", then "Math Interfaces", then "atan2()":
\   http://www.opengroup.org/onlinepubs/009695399/functions/atan2f.html

\ If y is +/-0 and x is < 0, +/-pi shall be returned.
t{  0e -1e fatan2 ->  pi }t
t{ -0e -1e fatan2 -> -pi }t

\ If y is +/-0 and x is > 0, +/-0 shall be returned.
t{  0e  1e fatan2 ->  0e }t
t{ -0e  1e fatan2 -> -0e }t

\ If y is < 0 and x is +/-0, -pi/2 shall be returned.
t{ -1e  0e fatan2 -> -pi/2 }t
t{ -1e -0e fatan2 -> -pi/2 }t

\ If y is > 0 and x is +/-0, pi/2 shall be returned.
t{  1e  0e fatan2 -> pi/2 }t
t{  1e -0e fatan2 -> pi/2 }t

testing Single UNIX 3 special values optional spec

\ Optional ISO C / single UNIX specs:

\ If either x or y is NaN, a NaN shall be returned.
t{ NaN  1e fatan2 -> NaN }t
t{  1e NaN fatan2 -> NaN }t
t{ NaN NaN fatan2 -> NaN }t

\ If y is +/-0 and x is -0, +/-pi shall be returned.
t{  0e -0e fatan2 ->  pi }t
t{ -0e -0e fatan2 -> -pi }t

\ If y is +/-0 and x is +0, +/-0 shall be returned.
t{  0e  0e fatan2 -> +0e }t
t{ -0e  0e fatan2 -> -0e }t

\ For finite values of +/-y > 0, if x is -Inf, +/-pi shall be returned.
t{  1e -Inf fatan2 ->  pi }t
t{ -1e -Inf fatan2 -> -pi }t

\ For finite values of +/-y > 0, if x is +Inf, +/-0 shall be returned.
t{  1e +Inf fatan2 -> +0e }t
t{ -1e +Inf fatan2 -> -0e }t

\ For finite values of x, if y is +/-Inf, +/-pi/2 shall be returned.
t{ +Inf  1e fatan2 ->  pi/2 }t
t{ +Inf -1e fatan2 ->  pi/2 }t
t{ +Inf  0e fatan2 ->  pi/2 }t
t{ +Inf -0e fatan2 ->  pi/2 }t
t{ -Inf  1e fatan2 -> -pi/2 }t
t{ -Inf -1e fatan2 -> -pi/2 }t
t{ -Inf  0e fatan2 -> -pi/2 }t
t{ -Inf -0e fatan2 -> -pi/2 }t

\ If y is +/-Inf and x is -Inf, +/-3pi/4 shall be returned.
t{ +Inf -Inf fatan2 ->  3pi/4 }t
t{ -Inf -Inf fatan2 -> -3pi/4 }t

\ If y is +/-Inf and x is +Inf, +/-pi/4 shall be returned.
t{ +Inf +Inf fatan2 ->  pi/4 }t
t{ -Inf +Inf fatan2 -> -pi/4 }t

verbose @ [IF]
cr .( #ERRORS: ) #errors @ . cr
[THEN]

CR
CR .( End of fatan2-test.fs) CR