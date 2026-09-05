\ From A.K. comp.lang.forth 2/4/2012
\ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

\ Below is some pretty old stuff for testing sum FP words. Don't know if it
\ is "compliant" or consistent or strictly mathematically correct. But perhaps
\ somebody might be willing to brush it up.

0 [if]
Changes made by Gerry Jackson
   1. Various as marked for GForth
   2. Tests with FLOATS Faligned etc removed as size of a float is
      implementation defined, therefore original tests not portable
   3. The FS. FE. and F. tests have been kept although different systems
      have a different interpretation of 'significant digits' in SET-PRECISION
[then]

\ THIS PROGRAM TESTS SOME OF THE FLOATING-POINT WORDS OF A MINFORTH SYSTEM

\ REQUIRE ttester.fs

CR .( Running ak-fp-test.fth)
CR .( ----------------------) CR  CR

TESTING --- MINFORTH FLOATING-POINT WORDS ---

\ : EMPTY-FPSTACK     \ ( F: ... -- ) EMPTY FLOATING-POINT STACK
\  BEGIN FDEPTH WHILE FDROP REPEAT ;

\ WARNING OFF
\ : }T FDEPTH 0<> ABORT" FP-STACK IS NOT EMPTY" }T ;
\ WARNING ON

\ EMPTY-FPSTACK
DECIMAL

\ ------------------------------------------------------------------------
TESTING BASIC FUNCTIONS

T{ -> }T        \ START    WITH CLEAN SLATE

T{ 0. d>f fdepth f>d -> 1 0. }T \ test the words used in EMPTY-FPSTACK
T{ 0. d>f fdrop fdepth -> 0 }T
T{ 0. d>f 0. d>f fdrop fdepth f>d -> 1 0. }T

T{ 0. d>f f>d -> 0. }T
T{ 1. d>f f>d -> 1. }T
T{ -1. d>f f>d -> -1. }T

\ IEEE754 max value (52 bit mantissa)
T{ hex fffffffffffff. decimal d>f f>d -> hex fffffffffffff. decimal }T

\ ------------------------------------------------------------------------
TESTING STACK OPERATIONS

T{ 1. d>f 2. d>f fswap f>d f>d -> 1. 2. }T
T{ 1. 2. 3. d>f d>f d>f frot f>d f>d f>d -> 3. 1. 2. }T
T{ -7. d>f fdup f>d f>d -> -7. -7. }T
T{ -4. d>f -2. d>f fover f>d f>d f>d -> -4. -2. -4. }T

\ ------------------------------------------------------------------------
TESTING BASIC ARITHMETICS

T{ 1. d>f fdup f+ f>d -> 2. }T
T{ -2. d>f -3. d>f f+ f>d -> -5. }T

T{ 1. d>f fdup f- f>d -> 0. }T
T{ -2. d>f -3. d>f f- f>d -> 1. }T

T{ 0. d>f fnegate f>d -> 0. }T
T{ 7. d>f fnegate f>d -> -7. }T
T{ -3. d>f fnegate f>d -> 3. }T

T{ 2. d>f fdup f* f>d -> 4. }T
T{ -3. d>f fdup f* f>d -> 9. }T
T{ -2. d>f 3. d>f f* f>d -> -6. }T
T{ 5. d>f -2. d>f f* f>d -> -10. }T
T{ 0. d>f -1. d>f f* f>d -> 0. }T
T{ 3. d>f 0. d>f f* f>d -> 0. }T

T{ 10. d>f 5. d>f f/ f>d -> 2. }T
T{ -33. d>f 11. d>f f/ f>d -> -3. }T
T{ 33. d>f -3. d>f f/ f>d -> -11. }T
T{ -14. d>f -7. d>f f/ f>d -> 2. }T
T{ 0. d>f 2. d>f f/ f>d -> 0. }T

\ ------------------------------------------------------------------------
TESTING COMPARISONS

T{ 1. d>f f0< -> false }T
T{ 0. d>f f0< -> false }T
T{ -1. d>f f0< -> true }T

T{ 0. d>f f0= -> true }T
T{ 1. d>f f0= -> false }T
T{ -2. d>f f0= -> false }T

T{ 1. d>f 2. d>f f< -> true }T
T{ 2. d>f 0. d>f f< -> false }T
T{ 0. d>f -2. d>f f< -> false }T
T{ -3. d>f -2. d>f f< -> true }T

T{ 1. d>f fabs f>d -> 1. }T
T{ -2. d>f fabs f>d -> 2. }T

T{ -2. d>f 1. d>f fmax f>d -> 1. }T
T{ -1. d>f -2. d>f fmax f>d -> -1. }T
T{ -1. d>f 0. d>f fmax f>d -> 0. }T
T{ 0. d>f -1. d>f fmax f>d -> 0. }T
T{ 1. d>f 2. d>f fmin f>d -> 1. }T
T{ -3. d>f -4. d>f fmin f>d -> -4. }T
T{ 1. d>f 0. d>f fmin f>d -> 0. }T
T{ 0. d>f -2. d>f fmin f>d -> -2. }T

T{ 10. d>f 11. d>f 1. d>f f~ -> false }T
T{ -10. d>f -11. d>f 2. d>f f~ -> true }T
T{ 1. d>f 2. d>f 1. d>f f~ -> false }T

T{ 0. d>f fdup fdup f~ -> true }T
T{ 1. d>f fdup 0. d>f f~ -> true }T
T{ -2. d>f fdup 0. d>f f~ -> true }T
T{ 3. d>f 4. d>f 0. d>f f~ -> false }T

T{ 2. d>f 3. d>f -1. d>f 5. d>f f/ f~ -> false }T
T{ 2. d>f 3. d>f -2. d>f 5. d>f f/ f~ -> true }T
T{ -2. d>f -3. d>f -1. d>f 5. d>f f/ f~ -> false }T
T{ -2. d>f -3. d>f -2. d>f 5. d>f f/ f~ -> true }T

\ ------------------------------------------------------------------------
TESTING MEMORY ACCESS, FLITERAL, FCONSTANT, FVARIABLE

VARIABLE FMEM 2 CELLS ALLOT

T{ 1. d>f fmem f! -> }T
T{ fmem f@ f>d -> 1. }T

FMEM 2 CELLS ERASE
T{ -2. d>f fmem sf! -> }T
T{ fmem sf@ f>d -> -2. }T
T{ fmem cell+ @ -> 0 }T

T{ 3. d>f fmem df! -> }T
T{ fmem df@ f>d -> 3. }T

: FT1 [ -2. d>f ] fliteral ;
T{ ft1 f>d -> -2. }T

-3. D>F FCONSTANT FT2
T{ ft2 f>d -> -3. }T

FVARIABLE FT4
T{ -4. d>f ft4 f! -> }T
T{ ft4 f@ f>d -> -4. }T

0 [if]  Tests removed as size of a float is implementation dependent
T{ 0 floats -> 0 }T
T{ 1 floats -> 8 }T
T{ -1 floats -> -8 }T
T{ 0 sfloats -> 0 }T
T{ 1 sfloats -> 4 }T
T{ -1 sfloats -> -4 }T
T{ 0 dfloats -> 0 }T
T{ 1 dfloats -> 8 }T
T{ -1 dfloats -> -8 }T
T{ 0 float+ -> 8 }T
T{ 0 sfloat+ -> 4 }T
T{ 0 dfloat+ -> 8 }T

T{ 8 faligned -> 8 }T
T{ 9 faligned -> 16 }T
T{ 10 faligned -> 16 }T
T{ 11 faligned -> 16 }T
T{ 12 faligned -> 16 }T
T{ 13 faligned -> 16 }T
T{ 14 faligned -> 16 }T
T{ 15 faligned -> 16 }T
T{ 16 faligned -> 16 }T
T{ 17 faligned -> 24 }T

T{ 4 sfaligned -> 4 }T
T{ 5 sfaligned -> 8 }T
T{ 6 sfaligned -> 8 }T
T{ 7 sfaligned -> 8 }T
T{ 8 sfaligned -> 8 }T
T{ 9 sfaligned -> 12 }T

T{ 8 dfaligned -> 8 }T
T{ 9 dfaligned -> 16 }T
T{ 10 dfaligned -> 16 }T
T{ 11 dfaligned -> 16 }T
T{ 12 dfaligned -> 16 }T
T{ 13 dfaligned -> 16 }T
T{ 14 dfaligned -> 16 }T
T{ 15 dfaligned -> 16 }T
T{ 16 dfaligned -> 16 }T
T{ 17 dfaligned -> 24 }T

T{ 0 c, falign here 7 and -> 0 }T
T{ 0 c, dfalign here 7 and -> 0 }T
T{ 0 c, sfalign here 3 and -> 0 }T
T{ falign here falign here = -> true }T
T{ dfalign here dfalign here = -> true }T
T{ sfalign here sfalign here = -> true }T
[THEN]
T{ 0  floats -> 0 }T
T{ 0 sfloats -> 0 }T
T{ 0 dfloats -> 0 }T

\ ------------------------------------------------------------------------
TESTING NUMBER INPUT
\ Leading and trailing spaces removed from strings because ANS Forth
\ does not specify they should be ignored

T{ s" ." >float -> false }T
T{ s" .E" >float -> false }T
T{ s" +.E+0" >float -> false }T
T{ s" E" >float -> false }T
T{ s" 0E" >float f>d -> true 0. }T
T{ s" 1E" >float f>d -> true 1. }T
T{ s" 1.E" >float f>d -> true 1. }T
T{ s" 1.E0" >float f>d -> true 1. }T
T{ s" 1.2E+1" >float f>d -> true 12. }T
T{ s" +1.2E1" >float f>d -> true 12. }T
T{ s" 120E-1" >float f>d -> true 12. }T
T{ s" -120E-1" >float f>d -> true -12. }T

T{ s" 1F" >float -> false }T  \ check against C floats
T{ s" 1D" >float f>d -> true 1. }T
T{ s" -1D" >float f>d -> true -1. }T

T{ s" 1EE" >float -> false }T
T{ s" 1DD" >float -> false }T
T{ s" 1E1E" >float -> false }T
T{ s" 1E 1E" >float -> false }T
T{ s" 1E  1E" >float -> false }T

T{ pad 0 >float f>d -> true 0. }T
T{ s"   " >float f>d -> true 0. }T  \ special case

T{ s" 2.0D0" >float f>d -> true 2. }T     \ Modified to make it compliant
T{ s" 2.0d+0" >float f>d -> true 2. }T    \ Modified to make it compliant
T{ s" -2.0e-0" >float f>d -> true -2. }T  \ Modified to make it compliant

T{ 1.0E0 f>d -> 1. }T
T{ -2.0E1 f>d -> -20. }T
T{ 200.0E-1 f>d -> 20. }T                 \ Modified to make it compliant
T{ +300.0E+02 f>d -> 30000. }T            \ Modified to make it compliant
T{ 10E f>d -> 10. }T
T{ -10E-1 f>d -> -1. }T

\ ------------------------------------------------------------------------
TESTING FRACTIONAL ARITHMETICS
DECIMAL

: F=    ( r1 r2 -- flag, true if exact identitiy )
  0E f~ ;

: TF=   ( r1 r2 -- flag, true if abs. error < 0.00005 )
  0.00005E f~ ;

T{ 1.E 1.00005E tf= -> false }T
T{ 1.E 1.00004E tf= -> true }T

T{ 3.33333E 6.66666E f+ 10E tf= -> true }T
T{ 10E 6.66666E f- 3.33333E tf= -> true }T
T{ 2E 0.02E f* 0.04E tf= -> true }T
T{ 10E 3E f/ 3.3333E tf= -> true }T
T{ -3E-3 fnegate +3E-3 f= -> true }T

T{ 2E floor 2E f= -> true }T
T{ 1.5E floor 1E f= -> true }T
T{ -0.5E floor -1E f= -> true }T
T{ 0E floor 0E f= -> true }T
T{ -0E floor -0E f= -> true }T            \ Modified to agree with GForth

T{ 2E fround 2E f= -> true }T
T{ 1.5E fround 2E f= -> true }T
T{ 1.4999E fround 1E f= -> true }T
T{ -0.4999E fround -0E f= -> true }T      \ Modified to agree with GForth
T{ -0.5001E fround -1E f= -> true }T
\ T{ 2.5E fround 2E f= -> true }T

T{ 4E fsqrt 2E tf= -> true }T
T{ 2E fsqrt 1.4142E tf= -> true }T
T{ 0E fsqrt 0E f= -> true }T
T{ 1E fsqrt 1E f= -> true }T

\ ------------------------------------------------------------------------
TESTING TRIGONOMETRIC FUNCTIONS

[UNDEFINED] PI [IF]
   3.1415926535897932384626433832795E FCONSTANT PI
[THEN]
PI   0.5E F* FCONSTANT PI2/
PI2/ 0.5E F* FCONSTANT PI4/

T{ 0E fsin 0E f= -> true }T
T{ pi fsin 0E tf= -> true }T
T{ pi2/ fsin 1E tf= -> true }T
T{ pi4/ fsin 0.7071E tf= -> true }T
T{ pi fnegate fsin 0E tf= -> true }T
T{ pi2/ fnegate fsin 1E fnegate tf= -> true }T
T{ pi4/ fnegate fsin -0.7071E tf= -> true }T
T{ 10E fsin -0.5440E tf= -> true }T

T{ 0E fcos 1E f= -> true }T
T{ pi fcos 1E fnegate tf= -> true }T
T{ pi2/ fcos 0E tf= -> true }T
T{ pi4/ fcos 0.7071E tf= -> true }T
T{ pi fnegate fcos 1E fnegate tf= -> true }T
T{ pi2/ fnegate fcos 0E tf= -> true }T
T{ pi4/ fnegate fcos 0.7071E tf= -> true }T
T{ 10E fcos -0.8391E tf= -> true }T

T{ 0E fsincos 1E f= 0E f= -> true true }T
T{ pi4/ fsincos f- 0E tf= -> true }T
T{ 2.3562E fsincos f+ 0E tf= -> true }T

T{ 0E ftan 0E f= -> true }T
T{ pi ftan 0E tf= -> true }T
T{ pi4/ ftan 1E tf= -> true }T
T{ pi 6E f/ ftan 0.57735E tf= -> true }T
T{ pi fnegate ftan 0E tf= -> true }T
T{ pi 6E f/ fnegate ftan -0.57735E tf= -> true }T
T{ pi4/ fnegate ftan 1E fnegate tf= -> true }T
T{ 10E ftan 0.6484E tf= -> true }T

T{ 0E fasin 0E f= -> true }T
T{ 0.5E fasin pi f/ 0.1667E tf= -> true }T
T{ 1E fasin pi f/ 0.5E tf= -> true }T
T{ -1E fasin pi f/ -0.5E tf= -> true }T

T{ 1E facos 0E tf= -> true }T
T{ 0.5E facos pi f/ 0.3333E tf= -> true }T
T{ 0E facos pi f/ 0.5E tf= -> true }T
T{ -1E facos pi tf= -> true }T

T{ 0E fatan 0E f= -> true }T
T{ 1E fatan 0.7854E tf= -> true }T
T{ 0.5E fatan 0.4636E tf= -> true }T
T{ -1E fatan -0.7854E tf= -> true }T

T{ 0E 1E fatan2 0E f= -> true }T
T{ 1E 1E fatan2 0.7854E tf= -> true }T
T{ -1E 1E fatan2 -0.7854E tf= -> true }T
T{ -1E -1E fatan2 -2.3562E tf= -> true }T
T{ 1E -1E fatan2 2.3562E tf= -> true }T

\ ------------------------------------------------------------------------
TESTING EXPONENTIAL AND LOGARITHMIC FUNCTIONS

T{ 0E fexp 1E f= -> true }T
T{ 1E fexp 2.7183E tf= -> true }T
T{ -1E fexp 0.3679E tf= -> true }T

T{ 0E fexpm1 0E f= -> true }T
T{ 1E fexpm1 1.7183E tf= -> true }T
T{ -1E fexpm1 -0.6321E tf= -> true }T

T{ 1E fln 0E f= -> true }T
T{ 2.7183E fln 1E tf= -> true }T
T{ 0.36788E fln -1E tf= -> true }T

T{ 1E flog 0E f= -> true }T
T{ 0.1E flog -1E tf= -> true }T
T{ 10E flog 1E tf= -> true }T

T{ 0E flnp1 0E f= -> true }T
T{ 1E flnp1 0.6931E tf= -> true }T
T{ -0.63212E flnp1 -1E tf= -> true }T

T{ 1E 0E f** 1E f= -> true }T
T{ 2E 2E f** 4E tf= -> true }T
T{ 2E 0.5E f** 1.4142E tf= -> true }T

T{ 0E falog 1E f= -> true }T
T{ 1E falog 10E tf= -> true }T
T{ -1E falog 0.1E tf= -> true }T

\ ------------------------------------------------------------------------
TESTING HYPERBOLIC FUNCTIONS

T{ 0E fsinh 0E f= -> true }T
T{ -1E fsinh -1.1752E tf= -> true }T
T{ 1E fsinh 1.1752E tf= -> true }T

T{ 0E fcosh 1E f= -> true }T
T{ 1E fcosh 1.5431E tf= -> true }T
T{ -1E fcosh 1.5431E tf= -> true }T

T{ 0E ftanh 0E f= -> true }T
T{ 1E ftanh 0.7616E tf= -> true }T
T{ -1E ftanh -0.7616E tf= -> true }T

T{ 0E fasinh 0E f= -> true }T
T{ -1E fasinh -0.8814E tf= -> true }T
T{ 1E fasinh 0.8814E tf= -> true }T

T{ 1E facosh 0E f= -> true }T
T{ 2E facosh 1.317E tf= -> true }T

\ ------------------------------------------------------------------------
TESTING NUMBER OUTPUT

CREATE FBUF 20 ALLOT
FBUF 20 ERASE

T{ 1E fbuf 5 represent -> 1 0 true }T           \ Modified to agree with GForth
T{ s" 10000" fbuf 5 compare -> 0 }T
\ T{ fbuf 5 + c@ -> 0 }T                        \ Not required

T{ -1E fbuf 5 represent -> 1 -1 true }T
T{ s" 10000" fbuf 5 compare -> 0 }T

T{ 100E 3E f/ fbuf 5 represent -> 2 0 true }T   \ Modified to agree with GForth
T{ s" 33333" fbuf 5 compare -> 0 }T
T{ 0.02E 3E f/ fbuf 5 represent -> -2 0 true }T \ Modified to agree with GForth
T{ s" 66667" fbuf 5 compare -> 0 }T

CR .( CHECKING FS. )

: YSS  \ ( a u -- ) type a new output check line
cr ." You might see " type ."  : " ;

5 SET-PRECISION

1E S" 1.0000E0 " YSS FS.         \ Modified to agree with GForth & Win32 Forth
20E S" 2.0000E1 " YSS FS.        \ Modified to agree with GForth & Win32 Forth
0.02E S" 2.0000E-2" YSS FS.      \ Modified to agree with GForth & Win32 Forth
-333.E2 S" -3.3300E4" YSS FS.    \ Modified to agree with GForth & Win32 Forth
10E 3E F/ S" 3.3333E0 " YSS FS.
0.2E 3E F/ S" 6.6667E-2" YSS FS.

CR .( CHECKING FE. )

1E S" 1.0000E0 " YSS FE.         \ Modified to agree with GForth & Win32 Forth
20E S" 20.000E0 " YSS FE.        \ Modified to agree with GForth & Win32 Forth
300E S" 300.00E0 " YSS FE.       \ Modified to agree with GForth & Win32 Forth
4000E S" 4.0000E3 " YSS FE.      \ Modified to agree with GForth & Win32 Forth
1E 3E F/ S" 333.33E-3" YSS FE.
2E4 3E f/ S" 6.6667E3 " YSS FE.

CR .( CHECKING F. )

1E3 S" 1000.  " YSS F.
1.1E3 S" 1100.  " YSS F.
1E 3E F/ S" 0.33333" YSS F.
200E 3E F/ S" 66.667 " YSS F.
0.000234E S" 0.00023" YSS F.     \ Modified to agree with GForth & Win32 Forth
0.000236E S" 0.00024" YSS F.     \ Modified to agree with GForth & Win32 Forth
\ -------------------------------------------------------------------------------------

CR
CR .( End of ak-fp-test.fth) CR
