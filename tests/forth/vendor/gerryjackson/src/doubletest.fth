\ To test the ANS Forth Double-Number word set and double number extensions

\ This program was written by Gerry Jackson in 2006, with contributions from
\ others where indicated, and is in the public domain - it can be distributed
\ and/or modified in any way but please retain this notice.

\ This program is distributed in the hope that it will be useful,
\ but WITHOUT ANY WARRANTY; without even the implied warranty of
\ MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

\ The tests are not claimed to be comprehensive or correct 
\ ------------------------------------------------------------------------------
\ Version 0.13  Assumptions and dependencies changed
\         0.12  1 August 2015 test D< acts on MS cells of double word
\         0.11  7 April 2015 2VALUE tested
\         0.6   1 April 2012 Tests placed in the public domain.
\               Immediate 2CONSTANTs and 2VARIABLEs tested
\         0.5   20 November 2009 Various constants renamed to avoid
\               redefinition warnings. <TRUE> and <FALSE> replaced
\               with TRUE and FALSE
\         0.4   6 March 2009 { and } replaced with T{ and }T
\               Tests rewritten to be independent of word size and
\               tests re-ordered
\         0.3   20 April 2007 ANS Forth words changed to upper case
\         0.2   30 Oct 2006 Updated following GForth test to include
\               various constants from core.fr
\         0.1   Oct 2006 First version released
\ ------------------------------------------------------------------------------
\ The tests are based on John Hayes test program for the core word set

\ Words tested in this file are:
\     2CONSTANT 2LITERAL 2VARIABLE D+ D- D. D.R D0< D0= D2* D2/
\     D< D= D>S DABS DMAX DMIN DNEGATE M*/ M+ 2ROT DU<
\ Also tests the interpreter and compiler reading a double number
\ ------------------------------------------------------------------------------
\ Assumptions and dependencies:
\     - tester.fr (or ttester.fs), errorreport.fth and utilities.fth have been
\       included prior to this file
\     - the Core word set is available and tested
\ ------------------------------------------------------------------------------
\ Constant definitions

DECIMAL
0 INVERT        CONSTANT 1SD
1SD 1 RSHIFT    CONSTANT MAX-INTD   \ 01...1
MAX-INTD INVERT CONSTANT MIN-INTD   \ 10...0
MAX-INTD 2/     CONSTANT HI-INT     \ 001...1
MIN-INTD 2/     CONSTANT LO-INT     \ 110...1

\ ------------------------------------------------------------------------------
TESTING interpreter and compiler reading double numbers, with/without prefixes

T{ 1. -> 1 0 }T
T{ -2. -> -2 -1 }T
T{ : RDL1 3. ; RDL1 -> 3 0 }T
T{ : RDL2 -4. ; RDL2 -> -4 -1 }T

VARIABLE OLD-DBASE
DECIMAL BASE @ OLD-DBASE !
T{ #12346789. -> 12346789. }T
T{ #-12346789. -> -12346789. }T
T{ $12aBcDeF. -> 313249263. }T
T{ $-12AbCdEf. -> -313249263. }T
T{ %10010110. -> 150. }T
T{ %-10010110. -> -150. }T
\ Check BASE is unchanged
T{ BASE @ OLD-DBASE @ = -> <TRUE> }T

\ Repeat in Hex mode
16 OLD-DBASE ! 16 BASE !
T{ #12346789. -> BC65A5. }T
T{ #-12346789. -> -BC65A5. }T
T{ $12aBcDeF. -> 12AbCdeF. }T
T{ $-12AbCdEf. -> -12ABCDef. }T
T{ %10010110. -> 96. }T
T{ %-10010110. -> -96. }T
\ Check BASE is unchanged
T{ BASE @ OLD-DBASE @ = -> <TRUE> }T   \ 2

DECIMAL
\ Check number prefixes in compile mode
T{ : dnmp  #8327. $-2cbe. %011010111. ; dnmp -> 8327. -11454. 215. }T

\ ------------------------------------------------------------------------------
TESTING 2CONSTANT

T{ 1 2 2CONSTANT 2C1 -> }T
T{ 2C1 -> 1 2 }T
T{ : CD1 2C1 ; -> }T
T{ CD1 -> 1 2 }T
T{ : CD2 2CONSTANT ; -> }T
T{ -1 -2 CD2 2C2 -> }T
T{ 2C2 -> -1 -2 }T
T{ 4 5 2CONSTANT 2C3 IMMEDIATE 2C3 -> 4 5 }T
T{ : CD6 2C3 2LITERAL ; CD6 -> 4 5 }T

\ ------------------------------------------------------------------------------
\ Some 2CONSTANTs for the following tests

1SD MAX-INTD 2CONSTANT MAX-2INT  \ 01...1
0   MIN-INTD 2CONSTANT MIN-2INT  \ 10...0
MAX-2INT 2/  2CONSTANT HI-2INT   \ 001...1
MIN-2INT 2/  2CONSTANT LO-2INT   \ 110...0

\ ------------------------------------------------------------------------------
TESTING DNEGATE

T{ 0. DNEGATE -> 0. }T
T{ 1. DNEGATE -> -1. }T
T{ -1. DNEGATE -> 1. }T
T{ MAX-2INT DNEGATE -> MIN-2INT SWAP 1+ SWAP }T
T{ MIN-2INT SWAP 1+ SWAP DNEGATE -> MAX-2INT }T

\ ------------------------------------------------------------------------------
TESTING D+ with small integers

T{  0.  5. D+ ->  5. }T
T{ -5.  0. D+ -> -5. }T
T{  1.  2. D+ ->  3. }T
T{  1. -2. D+ -> -1. }T
T{ -1.  2. D+ ->  1. }T
T{ -1. -2. D+ -> -3. }T
T{ -1.  1. D+ ->  0. }T

TESTING D+ with mid range integers

T{  0  0  0  5 D+ ->  0  5 }T
T{ -1  5  0  0 D+ -> -1  5 }T
T{  0  0  0 -5 D+ ->  0 -5 }T
T{  0 -5 -1  0 D+ -> -1 -5 }T
T{  0  1  0  2 D+ ->  0  3 }T
T{ -1  1  0 -2 D+ -> -1 -1 }T
T{  0 -1  0  2 D+ ->  0  1 }T
T{  0 -1 -1 -2 D+ -> -1 -3 }T
T{ -1 -1  0  1 D+ -> -1  0 }T
T{ MIN-INTD 0 2DUP D+ -> 0 1 }T
T{ MIN-INTD S>D MIN-INTD 0 D+ -> 0 0 }T

TESTING D+ with large double integers

T{ HI-2INT 1. D+ -> 0 HI-INT 1+ }T
T{ HI-2INT 2DUP D+ -> 1SD 1- MAX-INTD }T
T{ MAX-2INT MIN-2INT D+ -> -1. }T
T{ MAX-2INT LO-2INT D+ -> HI-2INT }T
T{ HI-2INT MIN-2INT D+ 1. D+ -> LO-2INT }T
T{ LO-2INT 2DUP D+ -> MIN-2INT }T

\ ------------------------------------------------------------------------------
TESTING D- with small integers

T{  0.  5. D- -> -5. }T
T{  5.  0. D- ->  5. }T
T{  0. -5. D- ->  5. }T
T{  1.  2. D- -> -1. }T
T{  1. -2. D- ->  3. }T
T{ -1.  2. D- -> -3. }T
T{ -1. -2. D- ->  1. }T
T{ -1. -1. D- ->  0. }T

TESTING D- with mid-range integers

T{  0  0  0  5 D- ->  0 -5 }T
T{ -1  5  0  0 D- -> -1  5 }T
T{  0  0 -1 -5 D- ->  1  4 }T
T{  0 -5  0  0 D- ->  0 -5 }T
T{ -1  1  0  2 D- -> -1 -1 }T
T{  0  1 -1 -2 D- ->  1  2 }T
T{  0 -1  0  2 D- ->  0 -3 }T
T{  0 -1  0 -2 D- ->  0  1 }T
T{  0  0  0  1 D- ->  0 -1 }T
T{ MIN-INTD 0 2DUP D- -> 0. }T
T{ MIN-INTD S>D MAX-INTD 0 D- -> 1 1SD }T

TESTING D- with large integers

T{ MAX-2INT MAX-2INT D- -> 0. }T
T{ MIN-2INT MIN-2INT D- -> 0. }T
T{ MAX-2INT HI-2INT  D- -> LO-2INT DNEGATE }T
T{ HI-2INT  LO-2INT  D- -> MAX-2INT }T
T{ LO-2INT  HI-2INT  D- -> MIN-2INT 1. D+ }T
T{ MIN-2INT MIN-2INT D- -> 0. }T
T{ MIN-2INT LO-2INT  D- -> LO-2INT }T

\ ------------------------------------------------------------------------------
TESTING D0< D0=

T{ 0. D0< -> FALSE }T
T{ 1. D0< -> FALSE }T
T{ MIN-INTD 0 D0< -> FALSE }T
T{ 0 MAX-INTD D0< -> FALSE }T
T{ MAX-2INT  D0< -> FALSE }T
T{ -1. D0< -> TRUE }T
T{ MIN-2INT D0< -> TRUE }T

T{ 1. D0= -> FALSE }T
T{ MIN-INTD 0 D0= -> FALSE }T
T{ MAX-2INT  D0= -> FALSE }T
T{ -1 MAX-INTD D0= -> FALSE }T
T{ 0. D0= -> TRUE }T
T{ -1. D0= -> FALSE }T
T{ 0 MIN-INTD D0= -> FALSE }T

\ ------------------------------------------------------------------------------
TESTING D2* D2/

T{ 0. D2* -> 0. D2* }T
T{ MIN-INTD 0 D2* -> 0 1 }T
T{ HI-2INT D2* -> MAX-2INT 1. D- }T
T{ LO-2INT D2* -> MIN-2INT }T

T{ 0. D2/ -> 0. }T
T{ 1. D2/ -> 0. }T
T{ 0 1 D2/ -> MIN-INTD 0 }T
T{ MAX-2INT D2/ -> HI-2INT }T
T{ -1. D2/ -> -1. }T
T{ MIN-2INT D2/ -> LO-2INT }T

\ ------------------------------------------------------------------------------
TESTING D< D=

T{  0.  1. D< -> TRUE  }T
T{  0.  0. D< -> FALSE }T
T{  1.  0. D< -> FALSE }T
T{ -1.  1. D< -> TRUE  }T
T{ -1.  0. D< -> TRUE  }T
T{ -2. -1. D< -> TRUE  }T
T{ -1. -2. D< -> FALSE }T
T{ 0 1   1. D< -> FALSE }T  \ Suggested by Helmut Eller
T{ 1.  0 1  D< -> TRUE  }T
T{ 0 -1 1 -2 D< -> FALSE }T
T{ 1 -2 0 -1 D< -> TRUE  }T
T{ -1. MAX-2INT D< -> TRUE }T
T{ MIN-2INT MAX-2INT D< -> TRUE }T
T{ MAX-2INT -1. D< -> FALSE }T
T{ MAX-2INT MIN-2INT D< -> FALSE }T
T{ MAX-2INT 2DUP -1. D+ D< -> FALSE }T
T{ MIN-2INT 2DUP  1. D+ D< -> TRUE  }T
T{ MAX-INTD S>D 2DUP 1. D+ D< -> TRUE }T \ Ensure D< acts on MS cells 

T{ -1. -1. D= -> TRUE  }T
T{ -1.  0. D= -> FALSE }T
T{ -1.  1. D= -> FALSE }T
T{  0. -1. D= -> FALSE }T
T{  0.  0. D= -> TRUE  }T
T{  0.  1. D= -> FALSE }T
T{  1. -1. D= -> FALSE }T
T{  1.  0. D= -> FALSE }T
T{  1.  1. D= -> TRUE  }T

T{ 0 -1 0 -1 D= -> TRUE  }T
T{ 0 -1 0  0 D= -> FALSE }T
T{ 0 -1 0  1 D= -> FALSE }T
T{ 0  0 0 -1 D= -> FALSE }T
T{ 0  0 0  0 D= -> TRUE  }T
T{ 0  0 0  1 D= -> FALSE }T
T{ 0  1 0 -1 D= -> FALSE }T
T{ 0  1 0  0 D= -> FALSE }T
T{ 0  1 0  1 D= -> TRUE  }T

T{ MAX-2INT MIN-2INT D= -> FALSE }T
T{ MAX-2INT 0. D= -> FALSE }T
T{ MAX-2INT MAX-2INT D= -> TRUE }T
T{ MAX-2INT HI-2INT  D= -> FALSE }T
T{ MAX-2INT MIN-2INT D= -> FALSE }T
T{ MIN-2INT MIN-2INT D= -> TRUE }T
T{ MIN-2INT LO-2INT  D=  -> FALSE }T
T{ MIN-2INT MAX-2INT D= -> FALSE }T

\ ------------------------------------------------------------------------------
TESTING 2LITERAL 2VARIABLE

T{ : CD3 [ MAX-2INT ] 2LITERAL ; -> }T
T{ CD3 -> MAX-2INT }T
T{ 2VARIABLE 2V1 -> }T
T{ 0. 2V1 2! -> }T
T{ 2V1 2@ -> 0. }T
T{ -1 -2 2V1 2! -> }T
T{ 2V1 2@ -> -1 -2 }T
T{ : CD4 2VARIABLE ; -> }T
T{ CD4 2V2 -> }T
T{ : CD5 2V2 2! ; -> }T
T{ -2 -1 CD5 -> }T
T{ 2V2 2@ -> -2 -1 }T
T{ 2VARIABLE 2V3 IMMEDIATE 5 6 2V3 2! -> }T
T{ 2V3 2@ -> 5 6 }T
T{ : CD7 2V3 [ 2@ ] 2LITERAL ; CD7 -> 5 6 }T
T{ : CD8 [ 6 7 ] 2V3 [ 2! ] ; 2V3 2@ -> 6 7 }T

\ ------------------------------------------------------------------------------
TESTING DMAX DMIN

T{  1.  2. DMAX -> 2. }T
T{  1.  0. DMAX -> 1. }T
T{  1. -1. DMAX -> 1. }T
T{  1.  1. DMAX -> 1. }T
T{  0.  1. DMAX -> 1. }T
T{  0. -1. DMAX -> 0. }T
T{ -1.  1. DMAX -> 1. }T
T{ -1. -2. DMAX -> -1. }T

T{ MAX-2INT HI-2INT  DMAX -> MAX-2INT }T
T{ MAX-2INT MIN-2INT DMAX -> MAX-2INT }T
T{ MIN-2INT MAX-2INT DMAX -> MAX-2INT }T
T{ MIN-2INT LO-2INT  DMAX -> LO-2INT  }T

T{ MAX-2INT  1. DMAX -> MAX-2INT }T
T{ MAX-2INT -1. DMAX -> MAX-2INT }T
T{ MIN-2INT  1. DMAX ->  1. }T
T{ MIN-2INT -1. DMAX -> -1. }T


T{  1.  2. DMIN ->  1. }T
T{  1.  0. DMIN ->  0. }T
T{  1. -1. DMIN -> -1. }T
T{  1.  1. DMIN ->  1. }T
T{  0.  1. DMIN ->  0. }T
T{  0. -1. DMIN -> -1. }T
T{ -1.  1. DMIN -> -1. }T
T{ -1. -2. DMIN -> -2. }T

T{ MAX-2INT HI-2INT  DMIN -> HI-2INT  }T
T{ MAX-2INT MIN-2INT DMIN -> MIN-2INT }T
T{ MIN-2INT MAX-2INT DMIN -> MIN-2INT }T
T{ MIN-2INT LO-2INT  DMIN -> MIN-2INT }T

T{ MAX-2INT  1. DMIN ->  1. }T
T{ MAX-2INT -1. DMIN -> -1. }T
T{ MIN-2INT  1. DMIN -> MIN-2INT }T
T{ MIN-2INT -1. DMIN -> MIN-2INT }T

\ ------------------------------------------------------------------------------
TESTING D>S DABS

T{  1234  0 D>S ->  1234 }T
T{ -1234 -1 D>S -> -1234 }T
T{ MAX-INTD  0 D>S -> MAX-INTD }T
T{ MIN-INTD -1 D>S -> MIN-INTD }T

T{  1. DABS -> 1. }T
T{ -1. DABS -> 1. }T
T{ MAX-2INT DABS -> MAX-2INT }T
T{ MIN-2INT 1. D+ DABS -> MAX-2INT }T

\ ------------------------------------------------------------------------------
TESTING M+ M*/

T{ HI-2INT   1 M+ -> HI-2INT   1. D+ }T
T{ MAX-2INT -1 M+ -> MAX-2INT -1. D+ }T
T{ MIN-2INT  1 M+ -> MIN-2INT  1. D+ }T
T{ LO-2INT  -1 M+ -> LO-2INT  -1. D+ }T

\ To correct the result if the division is floored, only used when
\ necessary i.e. negative quotient and remainder <> 0

: ?FLOORED [ -3 2 / -2 = ] LITERAL IF 1. D- THEN ;

T{  5.  7 11 M*/ ->  3. }T
T{  5. -7 11 M*/ -> -3. ?FLOORED }T    \ FLOORED -4.
T{ -5.  7 11 M*/ -> -3. ?FLOORED }T    \ FLOORED -4.
T{ -5. -7 11 M*/ ->  3. }T
T{ MAX-2INT  8 16 M*/ -> HI-2INT }T
T{ MAX-2INT -8 16 M*/ -> HI-2INT DNEGATE ?FLOORED }T  \ FLOORED SUBTRACT 1
T{ MIN-2INT  8 16 M*/ -> LO-2INT }T
T{ MIN-2INT -8 16 M*/ -> LO-2INT DNEGATE }T
T{ MAX-2INT MAX-INTD MAX-INTD M*/ -> MAX-2INT }T
T{ MAX-2INT MAX-INTD 2/ MAX-INTD M*/ -> MAX-INTD 1- HI-2INT NIP }T
T{ MIN-2INT LO-2INT NIP 1+ DUP 1- NEGATE M*/ -> 0 MAX-INTD 1- }T
T{ MIN-2INT LO-2INT NIP 1- MAX-INTD M*/ -> MIN-INTD 3 + HI-2INT NIP 2 + }T
T{ MAX-2INT LO-2INT NIP DUP NEGATE M*/ -> MAX-2INT DNEGATE }T
T{ MIN-2INT MAX-INTD DUP M*/ -> MIN-2INT }T

\ ------------------------------------------------------------------------------
TESTING D. D.R

\ Create some large double numbers
MAX-2INT 71 73 M*/ 2CONSTANT DBL1
MIN-2INT 73 79 M*/ 2CONSTANT DBL2

: D>ASCII  ( D -- CADDR U )
   DUP >R <# DABS #S R> SIGN #>    ( -- CADDR1 U )
   HERE SWAP 2DUP 2>R CHARS DUP ALLOT MOVE 2R>
;

DBL1 D>ASCII 2CONSTANT "DBL1"
DBL2 D>ASCII 2CONSTANT "DBL2"

: DOUBLEOUTPUT
   CR ." You should see lines duplicated:" CR
   5 SPACES "DBL1" TYPE CR
   5 SPACES DBL1 D. CR
   8 SPACES "DBL1" DUP >R TYPE CR
   5 SPACES DBL1 R> 3 + D.R CR
   5 SPACES "DBL2" TYPE CR
   5 SPACES DBL2 D. CR
   10 SPACES "DBL2" DUP >R TYPE CR
   5 SPACES DBL2 R> 5 + D.R CR
;

T{ DOUBLEOUTPUT -> }T

\ ------------------------------------------------------------------------------
TESTING 2ROT DU< (Double Number extension words)

T{ 1. 2. 3. 2ROT -> 2. 3. 1. }T
T{ MAX-2INT MIN-2INT 1. 2ROT -> MIN-2INT 1. MAX-2INT }T

T{  1.  1. DU< -> FALSE }T
T{  1. -1. DU< -> TRUE  }T
T{ -1.  1. DU< -> FALSE }T
T{ -1. -2. DU< -> FALSE }T
T{ 0 1   1. DU< -> FALSE }T
T{ 1.  0 1  DU< -> TRUE  }T
T{ 0 -1 1 -2 DU< -> FALSE }T
T{ 1 -2 0 -1 DU< -> TRUE  }T

T{ MAX-2INT HI-2INT  DU< -> FALSE }T
T{ HI-2INT  MAX-2INT DU< -> TRUE  }T
T{ MAX-2INT MIN-2INT DU< -> TRUE }T
T{ MIN-2INT MAX-2INT DU< -> FALSE }T
T{ MIN-2INT LO-2INT  DU< -> TRUE }T

\ ------------------------------------------------------------------------------
TESTING 2VALUE

T{ 1111 2222 2VALUE 2VAL -> }T
T{ 2VAL -> 1111 2222 }T
T{ 3333 4444 TO 2VAL -> }T
T{ 2VAL -> 3333 4444 }T
T{ : TO-2VAL TO 2VAL ; 5555 6666 TO-2VAL -> }T
T{ 2VAL -> 5555 6666 }T

\ ------------------------------------------------------------------------------

DOUBLE-ERRORS SET-ERROR-COUNT

CR .( End of Double-Number word tests) CR

