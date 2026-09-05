\ To test the ANS Forth Core Extension word set

\ This program was written by Gerry Jackson in 2006, with contributions from
\ others where indicated, and is in the public domain - it can be distributed
\ and/or modified in any way but please retain this notice.

\ This program is distributed in the hope that it will be useful,
\ but WITHOUT ANY WARRANTY; without even the implied warranty of
\ MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

\ The tests are not claimed to be comprehensive or correct 

\ ------------------------------------------------------------------------------
\ Version 0.15 1 August 2025 Added two tests to VALUE
\         0.14 21 July 2022 Updated first line of BUFFER: test as recommended
\              in issue 32
\         0.13 28 October 2015
\              Replace <FALSE> and <TRUE> with FALSE and TRUE to avoid
\              dependence on Core tests
\              Moved SAVE-INPUT and RESTORE-INPUT tests in a file to filetest.fth
\              Use of 2VARIABLE (from optional wordset) replaced with CREATE.
\              Minor lower to upper case conversions.
\              Calls to COMPARE replaced by S= (in utilities.fth) to avoid use
\              of a word from an optional word set.
\              UNUSED tests revised as UNUSED UNUSED = may return FALSE when an
\              implementation has the data stack sharing unused dataspace.
\              Double number input dependency removed from the HOLDS tests.
\              Minor case sensitivities removed in definition names.
\         0.11 25 April 2015
\              Added tests for PARSE-NAME HOLDS BUFFER:
\              S\" tests added
\              DEFER IS ACTION-OF DEFER! DEFER@ tests added
\              Empty CASE statement test added
\              [COMPILE] tests removed because it is obsolescent in Forth 2012
\         0.10 1 August 2014
\             Added tests contributed by James Bowman for:
\                <> U> 0<> 0> NIP TUCK ROLL PICK 2>R 2R@ 2R>
\                HEX WITHIN UNUSED AGAIN MARKER
\             Added tests for:
\                .R U.R ERASE PAD REFILL SOURCE-ID 
\             Removed ABORT from NeverExecuted to enable Win32
\             to continue after failure of RESTORE-INPUT.
\             Removed max-intx which is no longer used.
\         0.7 6 June 2012 Extra CASE test added
\         0.6 1 April 2012 Tests placed in the public domain.
\             SAVE-INPUT & RESTORE-INPUT tests, position
\             of T{ moved so that tests work with ttester.fs
\             CONVERT test deleted - obsolete word removed from Forth 200X
\             IMMEDIATE VALUEs tested
\             RECURSE with :NONAME tested
\             PARSE and .( tested
\             Parsing behaviour of C" added
\         0.5 14 September 2011 Removed the double [ELSE] from the
\             initial SAVE-INPUT & RESTORE-INPUT test
\         0.4 30 November 2009  max-int replaced with max-intx to
\             avoid redefinition warnings.
\         0.3  6 March 2009 { and } replaced with T{ and }T
\                           CONVERT test now independent of cell size
\         0.2  20 April 2007 ANS Forth words changed to upper case
\                            Tests qd3 to qd6 by Reinhold Straub
\         0.1  Oct 2006 First version released
\ -----------------------------------------------------------------------------
\ The tests are based on John Hayes test program for the core word set

\ Words tested in this file are:
\     .( .R 0<> 0> 2>R 2R> 2R@ :NONAME <> ?DO AGAIN C" CASE COMPILE, ENDCASE
\     ENDOF ERASE FALSE HEX MARKER NIP OF PAD PARSE PICK REFILL
\     RESTORE-INPUT ROLL SAVE-INPUT SOURCE-ID TO TRUE TUCK U.R U> UNUSED
\     VALUE WITHIN [COMPILE]

\ Words not tested or partially tested:
\     \ because it has been extensively used already and is, hence, unnecessary
\     REFILL and SOURCE-ID from the user input device which are not possible
\     when testing from a file such as this one
\     UNUSED (partially tested) as the value returned is system dependent
\     Obsolescent words #TIB CONVERT EXPECT QUERY SPAN TIB as they have been
\     removed from the Forth 2012 standard

\ Results from words that output to the user output device have to visually
\ checked for correctness. These are .R U.R .(

\ -----------------------------------------------------------------------------
\ Assumptions & dependencies:
\     - tester.fr (or ttester.fs), errorreport.fth and utilities.fth have been
\       included prior to this file
\     - the Core word set available
\ -----------------------------------------------------------------------------
TESTING Core Extension words

DECIMAL

TESTING TRUE FALSE

T{ TRUE  -> 0 INVERT }T
T{ FALSE -> 0 }T

\ -----------------------------------------------------------------------------
TESTING <> U>   (contributed by James Bowman)

T{ 0 0 <> -> FALSE }T
T{ 1 1 <> -> FALSE }T
T{ -1 -1 <> -> FALSE }T
T{ 1 0 <> -> TRUE }T
T{ -1 0 <> -> TRUE }T
T{ 0 1 <> -> TRUE }T
T{ 0 -1 <> -> TRUE }T

T{ 0 1 U> -> FALSE }T
T{ 1 2 U> -> FALSE }T
T{ 0 MID-UINT U> -> FALSE }T
T{ 0 MAX-UINT U> -> FALSE }T
T{ MID-UINT MAX-UINT U> -> FALSE }T
T{ 0 0 U> -> FALSE }T
T{ 1 1 U> -> FALSE }T
T{ 1 0 U> -> TRUE }T
T{ 2 1 U> -> TRUE }T
T{ MID-UINT 0 U> -> TRUE }T
T{ MAX-UINT 0 U> -> TRUE }T
T{ MAX-UINT MID-UINT U> -> TRUE }T

\ -----------------------------------------------------------------------------
TESTING 0<> 0>   (contributed by James Bowman)

T{ 0 0<> -> FALSE }T
T{ 1 0<> -> TRUE }T
T{ 2 0<> -> TRUE }T
T{ -1 0<> -> TRUE }T
T{ MAX-UINT 0<> -> TRUE }T
T{ MIN-INT 0<> -> TRUE }T
T{ MAX-INT 0<> -> TRUE }T

T{ 0 0> -> FALSE }T
T{ -1 0> -> FALSE }T
T{ MIN-INT 0> -> FALSE }T
T{ 1 0> -> TRUE }T
T{ MAX-INT 0> -> TRUE }T

\ -----------------------------------------------------------------------------
TESTING NIP TUCK ROLL PICK   (contributed by James Bowman)

T{ 1 2 NIP -> 2 }T
T{ 1 2 3 NIP -> 1 3 }T

T{ 1 2 TUCK -> 2 1 2 }T
T{ 1 2 3 TUCK -> 1 3 2 3 }T

T{ : RO5 100 200 300 400 500 ; -> }T
T{ RO5 3 ROLL -> 100 300 400 500 200 }T
T{ RO5 2 ROLL -> RO5 ROT }T
T{ RO5 1 ROLL -> RO5 SWAP }T
T{ RO5 0 ROLL -> RO5 }T

T{ RO5 2 PICK -> 100 200 300 400 500 300 }T
T{ RO5 1 PICK -> RO5 OVER }T
T{ RO5 0 PICK -> RO5 DUP }T

\ -----------------------------------------------------------------------------
TESTING 2>R 2R@ 2R>   (contributed by James Bowman)

T{ : RR0 2>R 100 R> R> ; -> }T
T{ 300 400 RR0 -> 100 400 300 }T
T{ 200 300 400 RR0 -> 200 100 400 300 }T

T{ : RR1 2>R 100 2R@ R> R> ; -> }T
T{ 300 400 RR1 -> 100 300 400 400 300 }T
T{ 200 300 400 RR1 -> 200 100 300 400 400 300 }T

T{ : RR2 2>R 100 2R> ; -> }T
T{ 300 400 RR2 -> 100 300 400 }T
T{ 200 300 400 RR2 -> 200 100 300 400 }T

\ -----------------------------------------------------------------------------
TESTING HEX   (contributed by James Bowman)

T{ BASE @ HEX BASE @ DECIMAL BASE @ - SWAP BASE ! -> 6 }T

\ -----------------------------------------------------------------------------
TESTING WITHIN   (contributed by James Bowman)

T{ 0 0 0 WITHIN -> FALSE }T
T{ 0 0 MID-UINT WITHIN -> TRUE }T
T{ 0 0 MID-UINT+1 WITHIN -> TRUE }T
T{ 0 0 MAX-UINT WITHIN -> TRUE }T
T{ 0 MID-UINT 0 WITHIN -> FALSE }T
T{ 0 MID-UINT MID-UINT WITHIN -> FALSE }T
T{ 0 MID-UINT MID-UINT+1 WITHIN -> FALSE }T
T{ 0 MID-UINT MAX-UINT WITHIN -> FALSE }T
T{ 0 MID-UINT+1 0 WITHIN -> FALSE }T
T{ 0 MID-UINT+1 MID-UINT WITHIN -> TRUE }T
T{ 0 MID-UINT+1 MID-UINT+1 WITHIN -> FALSE }T
T{ 0 MID-UINT+1 MAX-UINT WITHIN -> FALSE }T
T{ 0 MAX-UINT 0 WITHIN -> FALSE }T
T{ 0 MAX-UINT MID-UINT WITHIN -> TRUE }T
T{ 0 MAX-UINT MID-UINT+1 WITHIN -> TRUE }T
T{ 0 MAX-UINT MAX-UINT WITHIN -> FALSE }T
T{ MID-UINT 0 0 WITHIN -> FALSE }T
T{ MID-UINT 0 MID-UINT WITHIN -> FALSE }T
T{ MID-UINT 0 MID-UINT+1 WITHIN -> TRUE }T
T{ MID-UINT 0 MAX-UINT WITHIN -> TRUE }T
T{ MID-UINT MID-UINT 0 WITHIN -> TRUE }T
T{ MID-UINT MID-UINT MID-UINT WITHIN -> FALSE }T
T{ MID-UINT MID-UINT MID-UINT+1 WITHIN -> TRUE }T
T{ MID-UINT MID-UINT MAX-UINT WITHIN -> TRUE }T
T{ MID-UINT MID-UINT+1 0 WITHIN -> FALSE }T
T{ MID-UINT MID-UINT+1 MID-UINT WITHIN -> FALSE }T
T{ MID-UINT MID-UINT+1 MID-UINT+1 WITHIN -> FALSE }T
T{ MID-UINT MID-UINT+1 MAX-UINT WITHIN -> FALSE }T
T{ MID-UINT MAX-UINT 0 WITHIN -> FALSE }T
T{ MID-UINT MAX-UINT MID-UINT WITHIN -> FALSE }T
T{ MID-UINT MAX-UINT MID-UINT+1 WITHIN -> TRUE }T
T{ MID-UINT MAX-UINT MAX-UINT WITHIN -> FALSE }T
T{ MID-UINT+1 0 0 WITHIN -> FALSE }T
T{ MID-UINT+1 0 MID-UINT WITHIN -> FALSE }T
T{ MID-UINT+1 0 MID-UINT+1 WITHIN -> FALSE }T
T{ MID-UINT+1 0 MAX-UINT WITHIN -> TRUE }T
T{ MID-UINT+1 MID-UINT 0 WITHIN -> TRUE }T
T{ MID-UINT+1 MID-UINT MID-UINT WITHIN -> FALSE }T
T{ MID-UINT+1 MID-UINT MID-UINT+1 WITHIN -> FALSE }T
T{ MID-UINT+1 MID-UINT MAX-UINT WITHIN -> TRUE }T
T{ MID-UINT+1 MID-UINT+1 0 WITHIN -> TRUE }T
T{ MID-UINT+1 MID-UINT+1 MID-UINT WITHIN -> TRUE }T
T{ MID-UINT+1 MID-UINT+1 MID-UINT+1 WITHIN -> FALSE }T
T{ MID-UINT+1 MID-UINT+1 MAX-UINT WITHIN -> TRUE }T
T{ MID-UINT+1 MAX-UINT 0 WITHIN -> FALSE }T
T{ MID-UINT+1 MAX-UINT MID-UINT WITHIN -> FALSE }T
T{ MID-UINT+1 MAX-UINT MID-UINT+1 WITHIN -> FALSE }T
T{ MID-UINT+1 MAX-UINT MAX-UINT WITHIN -> FALSE }T
T{ MAX-UINT 0 0 WITHIN -> FALSE }T
T{ MAX-UINT 0 MID-UINT WITHIN -> FALSE }T
T{ MAX-UINT 0 MID-UINT+1 WITHIN -> FALSE }T
T{ MAX-UINT 0 MAX-UINT WITHIN -> FALSE }T
T{ MAX-UINT MID-UINT 0 WITHIN -> TRUE }T
T{ MAX-UINT MID-UINT MID-UINT WITHIN -> FALSE }T
T{ MAX-UINT MID-UINT MID-UINT+1 WITHIN -> FALSE }T
T{ MAX-UINT MID-UINT MAX-UINT WITHIN -> FALSE }T
T{ MAX-UINT MID-UINT+1 0 WITHIN -> TRUE }T
T{ MAX-UINT MID-UINT+1 MID-UINT WITHIN -> TRUE }T
T{ MAX-UINT MID-UINT+1 MID-UINT+1 WITHIN -> FALSE }T
T{ MAX-UINT MID-UINT+1 MAX-UINT WITHIN -> FALSE }T
T{ MAX-UINT MAX-UINT 0 WITHIN -> TRUE }T
T{ MAX-UINT MAX-UINT MID-UINT WITHIN -> TRUE }T
T{ MAX-UINT MAX-UINT MID-UINT+1 WITHIN -> TRUE }T
T{ MAX-UINT MAX-UINT MAX-UINT WITHIN -> FALSE }T

T{ MIN-INT MIN-INT MIN-INT WITHIN -> FALSE }T
T{ MIN-INT MIN-INT 0 WITHIN -> TRUE }T
T{ MIN-INT MIN-INT 1 WITHIN -> TRUE }T
T{ MIN-INT MIN-INT MAX-INT WITHIN -> TRUE }T
T{ MIN-INT 0 MIN-INT WITHIN -> FALSE }T
T{ MIN-INT 0 0 WITHIN -> FALSE }T
T{ MIN-INT 0 1 WITHIN -> FALSE }T
T{ MIN-INT 0 MAX-INT WITHIN -> FALSE }T
T{ MIN-INT 1 MIN-INT WITHIN -> FALSE }T
T{ MIN-INT 1 0 WITHIN -> TRUE }T
T{ MIN-INT 1 1 WITHIN -> FALSE }T
T{ MIN-INT 1 MAX-INT WITHIN -> FALSE }T
T{ MIN-INT MAX-INT MIN-INT WITHIN -> FALSE }T
T{ MIN-INT MAX-INT 0 WITHIN -> TRUE }T
T{ MIN-INT MAX-INT 1 WITHIN -> TRUE }T
T{ MIN-INT MAX-INT MAX-INT WITHIN -> FALSE }T
T{ 0 MIN-INT MIN-INT WITHIN -> FALSE }T
T{ 0 MIN-INT 0 WITHIN -> FALSE }T
T{ 0 MIN-INT 1 WITHIN -> TRUE }T
T{ 0 MIN-INT MAX-INT WITHIN -> TRUE }T
T{ 0 0 MIN-INT WITHIN -> TRUE }T
T{ 0 0 0 WITHIN -> FALSE }T
T{ 0 0 1 WITHIN -> TRUE }T
T{ 0 0 MAX-INT WITHIN -> TRUE }T
T{ 0 1 MIN-INT WITHIN -> FALSE }T
T{ 0 1 0 WITHIN -> FALSE }T
T{ 0 1 1 WITHIN -> FALSE }T
T{ 0 1 MAX-INT WITHIN -> FALSE }T
T{ 0 MAX-INT MIN-INT WITHIN -> FALSE }T
T{ 0 MAX-INT 0 WITHIN -> FALSE }T
T{ 0 MAX-INT 1 WITHIN -> TRUE }T
T{ 0 MAX-INT MAX-INT WITHIN -> FALSE }T
T{ 1 MIN-INT MIN-INT WITHIN -> FALSE }T
T{ 1 MIN-INT 0 WITHIN -> FALSE }T
T{ 1 MIN-INT 1 WITHIN -> FALSE }T
T{ 1 MIN-INT MAX-INT WITHIN -> TRUE }T
T{ 1 0 MIN-INT WITHIN -> TRUE }T
T{ 1 0 0 WITHIN -> FALSE }T
T{ 1 0 1 WITHIN -> FALSE }T
T{ 1 0 MAX-INT WITHIN -> TRUE }T
T{ 1 1 MIN-INT WITHIN -> TRUE }T
T{ 1 1 0 WITHIN -> TRUE }T
T{ 1 1 1 WITHIN -> FALSE }T
T{ 1 1 MAX-INT WITHIN -> TRUE }T
T{ 1 MAX-INT MIN-INT WITHIN -> FALSE }T
T{ 1 MAX-INT 0 WITHIN -> FALSE }T
T{ 1 MAX-INT 1 WITHIN -> FALSE }T
T{ 1 MAX-INT MAX-INT WITHIN -> FALSE }T
T{ MAX-INT MIN-INT MIN-INT WITHIN -> FALSE }T
T{ MAX-INT MIN-INT 0 WITHIN -> FALSE }T
T{ MAX-INT MIN-INT 1 WITHIN -> FALSE }T
T{ MAX-INT MIN-INT MAX-INT WITHIN -> FALSE }T
T{ MAX-INT 0 MIN-INT WITHIN -> TRUE }T
T{ MAX-INT 0 0 WITHIN -> FALSE }T
T{ MAX-INT 0 1 WITHIN -> FALSE }T
T{ MAX-INT 0 MAX-INT WITHIN -> FALSE }T
T{ MAX-INT 1 MIN-INT WITHIN -> TRUE }T
T{ MAX-INT 1 0 WITHIN -> TRUE }T
T{ MAX-INT 1 1 WITHIN -> FALSE }T
T{ MAX-INT 1 MAX-INT WITHIN -> FALSE }T
T{ MAX-INT MAX-INT MIN-INT WITHIN -> TRUE }T
T{ MAX-INT MAX-INT 0 WITHIN -> TRUE }T
T{ MAX-INT MAX-INT 1 WITHIN -> TRUE }T
T{ MAX-INT MAX-INT MAX-INT WITHIN -> FALSE }T

\ -----------------------------------------------------------------------------
TESTING UNUSED  (contributed by James Bowman & Peter Knaggs)

VARIABLE UNUSED0
T{ UNUSED DROP -> }T                  
T{ ALIGN UNUSED UNUSED0 ! 0 , UNUSED CELL+ UNUSED0 @ = -> TRUE }T
T{ UNUSED UNUSED0 ! 0 C, UNUSED CHAR+ UNUSED0 @ =
         -> TRUE }T  \ aligned -> unaligned
T{ UNUSED UNUSED0 ! 0 C, UNUSED CHAR+ UNUSED0 @ = -> TRUE }T  \ unaligned -> ?

\ -----------------------------------------------------------------------------
TESTING AGAIN   (contributed by James Bowman)

T{ : AG0 701 BEGIN DUP 7 MOD 0= IF EXIT THEN 1+ AGAIN ; -> }T
T{ AG0 -> 707 }T

\ -----------------------------------------------------------------------------
TESTING MARKER   (contributed by James Bowman)

T{ : MA? BL WORD FIND NIP 0<> ; -> }T
T{ MARKER MA0 -> }T
T{ : MA1 111 ; -> }T
T{ MARKER MA2 -> }T
T{ : MA1 222 ; -> }T
T{ MA? MA0 MA? MA1 MA? MA2 -> TRUE TRUE TRUE }T
T{ MA1 MA2 MA1 -> 222 111 }T
T{ MA? MA0 MA? MA1 MA? MA2 -> TRUE TRUE FALSE }T
T{ MA0 -> }T
T{ MA? MA0 MA? MA1 MA? MA2 -> FALSE FALSE FALSE }T

\ -----------------------------------------------------------------------------
TESTING ?DO

: QD ?DO I LOOP ;
T{ 789 789 QD -> }T
T{ -9876 -9876 QD -> }T
T{ 5 0 QD -> 0 1 2 3 4 }T

: QD1 ?DO I 10 +LOOP ;
T{ 50 1 QD1 -> 1 11 21 31 41 }T
T{ 50 0 QD1 -> 0 10 20 30 40 }T

: QD2 ?DO I 3 > IF LEAVE ELSE I THEN LOOP ;
T{ 5 -1 QD2 -> -1 0 1 2 3 }T

: QD3 ?DO I 1 +LOOP ;
T{ 4  4 QD3 -> }T
T{ 4  1 QD3 -> 1 2 3 }T
T{ 2 -1 QD3 -> -1 0 1 }T

: QD4 ?DO I -1 +LOOP ;
T{  4 4 QD4 -> }T
T{  1 4 QD4 -> 4 3 2 1 }T
T{ -1 2 QD4 -> 2 1 0 -1 }T

: QD5 ?DO I -10 +LOOP ;
T{   1 50 QD5 -> 50 40 30 20 10 }T
T{   0 50 QD5 -> 50 40 30 20 10 0 }T
T{ -25 10 QD5 -> 10 0 -10 -20 }T

VARIABLE ITERS
VARIABLE INCRMNT

: QD6 ( limit start increment -- )
   INCRMNT !
   0 ITERS !
   ?DO
      1 ITERS +!
      I
      ITERS @  6 = IF LEAVE THEN
      INCRMNT @
   +LOOP ITERS @
;

T{  4  4 -1 QD6 -> 0 }T
T{  1  4 -1 QD6 -> 4 3 2 1 4 }T
T{  4  1 -1 QD6 -> 1 0 -1 -2 -3 -4 6 }T
T{  4  1  0 QD6 -> 1 1 1 1 1 1 6 }T
T{  0  0  0 QD6 -> 0 }T
T{  1  4  0 QD6 -> 4 4 4 4 4 4 6 }T
T{  1  4  1 QD6 -> 4 5 6 7 8 9 6 }T
T{  4  1  1 QD6 -> 1 2 3 3 }T
T{  4  4  1 QD6 -> 0 }T
T{  2 -1 -1 QD6 -> -1 -2 -3 -4 -5 -6 6 }T
T{ -1  2 -1 QD6 -> 2 1 0 -1 4 }T
T{  2 -1  0 QD6 -> -1 -1 -1 -1 -1 -1 6 }T
T{ -1  2  0 QD6 -> 2 2 2 2 2 2 6 }T
T{ -1  2  1 QD6 -> 2 3 4 5 6 7 6 }T
T{  2 -1  1 QD6 -> -1 0 1 3 }T

\ -----------------------------------------------------------------------------
TESTING BUFFER:

T{ 2 CELLS BUFFER: BUF:TEST -> }T
T{ BUF:TEST DUP ALIGNED = -> TRUE }T
T{ 111 BUF:TEST ! 222 BUF:TEST CELL+ ! -> }T
T{ BUF:TEST @ BUF:TEST CELL+ @ -> 111 222 }T

\ -----------------------------------------------------------------------------
TESTING VALUE TO

T{ 111 VALUE VAL1 -999 VALUE VAL2 -> }T
T{ VAL1 -> 111 }T
T{ VAL2 -> -999 }T
T{ 222 TO VAL1 -> }T
T{ VAL1 -> 222 }T
T{ : VD1 VAL1 ; -> }T
T{ VD1 -> 222 }T
T{ : VD2 TO VAL2 ; -> }T
T{ VAL2 -> -999 }T
T{ -333 VD2 -> }T
T{ VAL2 -> -333 }T
T{ VAL1 -> 222 }T
T{ 444 TO VAL1 -> }T
T{ VD1 -> 444 }T
T{ 123 VALUE VAL3 IMMEDIATE VAL3 -> 123 }T
T{ : VD3 VAL3 LITERAL ; VD3 -> 123 }T

\ -----------------------------------------------------------------------------
TESTING CASE OF ENDOF ENDCASE

: CS1 CASE 1 OF 111 ENDOF
           2 OF 222 ENDOF
           3 OF 333 ENDOF
           >R 999 R>
      ENDCASE
;

T{ 1 CS1 -> 111 }T
T{ 2 CS1 -> 222 }T
T{ 3 CS1 -> 333 }T
T{ 4 CS1 -> 999 }T

\ Nested CASE's

: CS2 >R CASE -1 OF CASE R@ 1 OF 100 ENDOF
                            2 OF 200 ENDOF
                           >R -300 R>
                    ENDCASE
                 ENDOF
              -2 OF CASE R@ 1 OF -99  ENDOF
                            >R -199 R>
                    ENDCASE
                 ENDOF
                 >R 299 R>
         ENDCASE R> DROP
;

T{ -1 1 CS2 ->  100 }T
T{ -1 2 CS2 ->  200 }T
T{ -1 3 CS2 -> -300 }T
T{ -2 1 CS2 -> -99  }T
T{ -2 2 CS2 -> -199 }T
T{  0 2 CS2 ->  299 }T

\ Boolean short circuiting using CASE

: CS3  ( N1 -- N2 )
   CASE 1- FALSE OF 11 ENDOF
        1- FALSE OF 22 ENDOF
        1- FALSE OF 33 ENDOF
        44 SWAP
   ENDCASE
;

T{ 1 CS3 -> 11 }T
T{ 2 CS3 -> 22 }T
T{ 3 CS3 -> 33 }T
T{ 9 CS3 -> 44 }T

\ Empty CASE statements with/without default

T{ : CS4 CASE ENDCASE ; 1 CS4 -> }T
T{ : CS5 CASE 2 SWAP ENDCASE ; 1 CS5 -> 2 }T
T{ : CS6 CASE 1 OF ENDOF 2 ENDCASE ; 1 CS6 -> }T
T{ : CS7 CASE 3 OF ENDOF 2 ENDCASE ; 1 CS7 -> 1 }T

\ -----------------------------------------------------------------------------
TESTING :NONAME RECURSE

VARIABLE NN1
VARIABLE NN2
:NONAME 1234 ; NN1 !
:NONAME 9876 ; NN2 !
T{ NN1 @ EXECUTE -> 1234 }T
T{ NN2 @ EXECUTE -> 9876 }T

T{ :NONAME ( n -- 0,1,..n ) DUP IF DUP >R 1- RECURSE R> THEN ;
   CONSTANT RN1 -> }T
T{ 0 RN1 EXECUTE -> 0 }T
T{ 4 RN1 EXECUTE -> 0 1 2 3 4 }T

:NONAME  ( n -- n1 )    \ Multiple RECURSEs in one definition
   1- DUP
   CASE 0 OF EXIT ENDOF
        1 OF 11 SWAP RECURSE ENDOF
        2 OF 22 SWAP RECURSE ENDOF
        3 OF 33 SWAP RECURSE ENDOF
        DROP ABS RECURSE EXIT
   ENDCASE
; CONSTANT RN2

T{  1 RN2 EXECUTE -> 0 }T
T{  2 RN2 EXECUTE -> 11 0 }T
T{  4 RN2 EXECUTE -> 33 22 11 0 }T
T{ 25 RN2 EXECUTE -> 33 22 11 0 }T

\ -----------------------------------------------------------------------------
TESTING C"

T{ : CQ1 C" 123" ; -> }T
T{ CQ1 COUNT EVALUATE -> 123 }T
T{ : CQ2 C" " ; -> }T
T{ CQ2 COUNT EVALUATE -> }T
T{ : CQ3 C" 2345"COUNT EVALUATE ; CQ3 -> 2345 }T

\ -----------------------------------------------------------------------------
TESTING COMPILE,

:NONAME DUP + ; CONSTANT DUP+
T{ : Q DUP+ COMPILE, ; -> }T
T{ : AS1 [ Q ] ; -> }T
T{ 123 AS1 -> 246 }T

\ -----------------------------------------------------------------------------
\ Cannot automatically test SAVE-INPUT and RESTORE-INPUT from a console source

TESTING SAVE-INPUT and RESTORE-INPUT with a string source

VARIABLE SI_INC 0 SI_INC !

: SI1
   SI_INC @ >IN +!
   15 SI_INC !
;

: S$ S" SAVE-INPUT SI1 RESTORE-INPUT 12345" ;

T{ S$ EVALUATE SI_INC @ -> 0 2345 15 }T

\ -----------------------------------------------------------------------------
TESTING .(

CR CR .( Output from .() 
T{ CR .( You should see -9876: ) -9876 . -> }T
T{ CR .( and again: ).( -9876)CR -> }T

CR CR .( On the next 2 lines you should see First then Second messages:)
T{ : DOTP  CR ." Second message via ." [CHAR] " EMIT    \ Check .( is immediate
     [ CR ] .( First message via .( ) ; DOTP -> }T
CR CR
T{ : IMM? BL WORD FIND NIP ; IMM? .( -> 1 }T

\ -----------------------------------------------------------------------------
TESTING .R and U.R - has to handle different cell sizes

\ Create some large integers just below/above MAX and Min INTs
MAX-INT 73 79 */ CONSTANT LI1
MIN-INT 71 73 */ CONSTANT LI2

LI1 0 <# #S #> NIP CONSTANT LENLI1

: (.R&U.R)  ( u1 u2 -- )  \ u1 <= string length, u2 is required indentation
   TUCK + >R
   LI1 OVER SPACES  . CR R@    LI1 SWAP  .R CR
   LI2 OVER SPACES  . CR R@ 1+ LI2 SWAP  .R CR
   LI1 OVER SPACES U. CR R@    LI1 SWAP U.R CR
   LI2 SWAP SPACES U. CR R>    LI2 SWAP U.R CR
;

: .R&U.R  ( -- )
   CR ." You should see lines duplicated:" CR
   ." indented by 0 spaces" CR 0      0 (.R&U.R) CR
   ." indented by 0 spaces" CR LENLI1 0 (.R&U.R) CR \ Just fits required width
   ." indented by 5 spaces" CR LENLI1 5 (.R&U.R) CR
;

CR CR .( Output from .R and U.R)
T{ .R&U.R -> }T

\ -----------------------------------------------------------------------------
TESTING PAD ERASE
\ Must handle different size characters i.e. 1 CHARS >= 1 

84 CONSTANT CHARS/PAD      \ Minimum size of PAD in chars
CHARS/PAD CHARS CONSTANT AUS/PAD
: CHECKPAD  ( caddr u ch -- f )  \ f = TRUE if u chars = ch
   SWAP 0
   ?DO
      OVER I CHARS + C@ OVER <>
      IF 2DROP UNLOOP FALSE EXIT THEN
   LOOP  
   2DROP TRUE
;

T{ PAD DROP -> }T
T{ 0 INVERT PAD C! -> }T
T{ PAD C@ CONSTANT MAXCHAR -> }T
T{ PAD CHARS/PAD 2DUP MAXCHAR FILL MAXCHAR CHECKPAD -> TRUE }T
T{ PAD CHARS/PAD 2DUP CHARS ERASE 0 CHECKPAD -> TRUE }T
T{ PAD CHARS/PAD 2DUP MAXCHAR FILL PAD 0 ERASE MAXCHAR CHECKPAD -> TRUE }T
T{ PAD 43 CHARS + 9 CHARS ERASE -> }T
T{ PAD 43 MAXCHAR CHECKPAD -> TRUE }T
T{ PAD 43 CHARS + 9 0 CHECKPAD -> TRUE }T
T{ PAD 52 CHARS + CHARS/PAD 52 - MAXCHAR CHECKPAD -> TRUE }T

\ Check that use of WORD and pictured numeric output do not corrupt PAD
\ Minimum size of buffers for these are 33 chars and (2*n)+2 chars respectively
\ where n is number of bits per cell

PAD CHARS/PAD ERASE
2 BASE !
MAX-UINT MAX-UINT <# #S CHAR 1 DUP HOLD HOLD #> 2DROP
DECIMAL
BL WORD 12345678123456781234567812345678 DROP
T{ PAD CHARS/PAD 0 CHECKPAD -> TRUE }T

\ -----------------------------------------------------------------------------
TESTING PARSE

T{ CHAR | PARSE 1234| DUP ROT ROT EVALUATE -> 4 1234 }T
T{ CHAR ^ PARSE  23 45 ^ DUP ROT ROT EVALUATE -> 7 23 45 }T
: PA1 [CHAR] $ PARSE DUP >R PAD SWAP CHARS MOVE PAD R> ;
T{ PA1 3456
   DUP ROT ROT EVALUATE -> 4 3456 }T
T{ CHAR A PARSE A SWAP DROP -> 0 }T
T{ CHAR Z PARSE
   SWAP DROP -> 0 }T
T{ CHAR " PARSE 4567 "DUP ROT ROT EVALUATE -> 5 4567 }T
 
\ -----------------------------------------------------------------------------
TESTING PARSE-NAME  (Forth 2012)
\ Adapted from the PARSE-NAME RfD tests

T{ PARSE-NAME abcd  STR1  S= -> TRUE }T        \ No leading spaces
T{ PARSE-NAME      abcde STR2 S= -> TRUE }T    \ Leading spaces

\ Test empty parse area, new lines are necessary
T{ PARSE-NAME
  NIP -> 0 }T
\ Empty parse area with spaces after PARSE-NAME
T{ PARSE-NAME         
  NIP -> 0 }T

T{ : PARSE-NAME-TEST ( "name1" "name2" -- n )
    PARSE-NAME PARSE-NAME S= ; -> }T
T{ PARSE-NAME-TEST abcd abcd  -> TRUE }T
T{ PARSE-NAME-TEST abcd   abcd  -> TRUE }T  \ Leading spaces
T{ PARSE-NAME-TEST abcde abcdf -> FALSE }T
T{ PARSE-NAME-TEST abcdf abcde -> FALSE }T
T{ PARSE-NAME-TEST abcde abcde
   -> TRUE }T         \ Parse to end of line
T{ PARSE-NAME-TEST abcde           abcde         
   -> TRUE }T         \ Leading and trailing spaces

\ -----------------------------------------------------------------------------
TESTING DEFER DEFER@ DEFER! IS ACTION-OF (Forth 2012)
\ Adapted from the Forth 200X RfD tests

T{ DEFER DEFER1 -> }T
T{ : MY-DEFER DEFER ; -> }T
T{ : IS-DEFER1 IS DEFER1 ; -> }T
T{ : ACTION-DEFER1 ACTION-OF DEFER1 ; -> }T
T{ : DEF! DEFER! ; -> }T
T{ : DEF@ DEFER@ ; -> }T

T{ ' * ' DEFER1 DEFER! -> }T
T{ 2 3 DEFER1 -> 6 }T
T{ ' DEFER1 DEFER@ -> ' * }T
T{ ' DEFER1 DEF@ -> ' * }T
T{ ACTION-OF DEFER1 -> ' * }T
T{ ACTION-DEFER1 -> ' * }T
T{ ' + IS DEFER1 -> }T
T{ 1 2 DEFER1 -> 3 }T
T{ ' DEFER1 DEFER@ -> ' + }T
T{ ' DEFER1 DEF@ -> ' + }T
T{ ACTION-OF DEFER1 -> ' + }T
T{ ACTION-DEFER1 -> ' + }T
T{ ' - IS-DEFER1 -> }T
T{ 1 2 DEFER1 -> -1 }T
T{ ' DEFER1 DEFER@ -> ' - }T
T{ ' DEFER1 DEF@ -> ' - }T
T{ ACTION-OF DEFER1 -> ' - }T
T{ ACTION-DEFER1 -> ' - }T

T{ MY-DEFER DEFER2 -> }T
T{ ' DUP IS DEFER2 -> }T
T{ 1 DEFER2 -> 1 1 }T

\ -----------------------------------------------------------------------------
TESTING HOLDS  (Forth 2012)

: HTEST S" Testing HOLDS" ;
: HTEST2 S" works" ;
: HTEST3 S" Testing HOLDS works 123" ;
T{ 0 0 <#  HTEST HOLDS #> HTEST S= -> TRUE }T
T{ 123 0 <# #S BL HOLD HTEST2 HOLDS BL HOLD HTEST HOLDS #>
   HTEST3 S= -> TRUE }T
T{ : HLD HOLDS ; -> }T
T{ 0 0 <#  HTEST HLD #> HTEST S= -> TRUE }T

\ -----------------------------------------------------------------------------
TESTING REFILL SOURCE-ID
\ REFILL and SOURCE-ID from the user input device can't be tested from a file,
\ can only be tested from a string via EVALUATE

T{ : RF1  S" REFILL" EVALUATE ; RF1 -> FALSE }T
T{ : SID1  S" SOURCE-ID" EVALUATE ; SID1 -> -1 }T

\ ------------------------------------------------------------------------------
TESTING S\"  (Forth 2012 compilation mode)
\ Extended the Forth 200X RfD tests
\ Note this tests the Core Ext definition of S\" which has unedfined
\ interpretation semantics. S\" in interpretation mode is tested in the tests on
\ the File-Access word set

T{ : SSQ1 S\" abc" S" abc" S= ; -> }T  \ No escapes
T{ SSQ1 -> TRUE }T
T{ : SSQ2 S\" " ; SSQ2 SWAP DROP -> 0 }T    \ Empty string

T{ : SSQ3 S\" \a\b\e\f\l\m\q\r\t\v\x0F0\x1Fa\xaBx\z\"\\" ; -> }T
T{ SSQ3 SWAP DROP          ->  20 }T    \ String length
T{ SSQ3 DROP            C@ ->   7 }T    \ \a   BEL  Bell
T{ SSQ3 DROP  1 CHARS + C@ ->   8 }T    \ \b   BS   Backspace
T{ SSQ3 DROP  2 CHARS + C@ ->  27 }T    \ \e   ESC  Escape
T{ SSQ3 DROP  3 CHARS + C@ ->  12 }T    \ \f   FF   Form feed
T{ SSQ3 DROP  4 CHARS + C@ ->  10 }T    \ \l   LF   Line feed
T{ SSQ3 DROP  5 CHARS + C@ ->  13 }T    \ \m        CR of CR/LF pair
T{ SSQ3 DROP  6 CHARS + C@ ->  10 }T    \           LF of CR/LF pair
T{ SSQ3 DROP  7 CHARS + C@ ->  34 }T    \ \q   "    Double Quote
T{ SSQ3 DROP  8 CHARS + C@ ->  13 }T    \ \r   CR   Carriage Return
T{ SSQ3 DROP  9 CHARS + C@ ->   9 }T    \ \t   TAB  Horizontal Tab
T{ SSQ3 DROP 10 CHARS + C@ ->  11 }T    \ \v   VT   Vertical Tab
T{ SSQ3 DROP 11 CHARS + C@ ->  15 }T    \ \x0F      Given Char
T{ SSQ3 DROP 12 CHARS + C@ ->  48 }T    \ 0    0    Digit follow on
T{ SSQ3 DROP 13 CHARS + C@ ->  31 }T    \ \x1F      Given Char
T{ SSQ3 DROP 14 CHARS + C@ ->  97 }T    \ a    a    Hex follow on
T{ SSQ3 DROP 15 CHARS + C@ -> 171 }T    \ \xaB      Insensitive Given Char
T{ SSQ3 DROP 16 CHARS + C@ -> 120 }T    \ x    x    Non hex follow on
T{ SSQ3 DROP 17 CHARS + C@ ->   0 }T    \ \z   NUL  No Character
T{ SSQ3 DROP 18 CHARS + C@ ->  34 }T    \ \"   "    Double Quote
T{ SSQ3 DROP 19 CHARS + C@ ->  92 }T    \ \\   \    Back Slash

\ The above does not test \n as this is a system dependent value.
\ Check it displays a new line
CR .( The next test should display:)
CR .( One line...)
CR .( another line)
T{ : SSQ4 S\" \nOne line...\nanotherLine\n" TYPE ; SSQ4 -> }T

\ Test bare escapable characters appear as themselves
T{ : SSQ5 S\" abeflmnqrtvxz" S" abeflmnqrtvxz" S= ; SSQ5 -> TRUE }T

T{ : SSQ6 S\" a\""2DROP 1111 ; SSQ6 -> 1111 }T \ Parsing behaviour

T{ : SSQ7  S\" 111 : SSQ8 S\\\" 222\" EVALUATE ; SSQ8 333" EVALUATE ; -> }T
T{ SSQ7 -> 111 222 333 }T
T{ : SSQ9  S\" 11 : SSQ10 S\\\" \\x32\\x32\" EVALUATE ; SSQ10 33" EVALUATE ; -> }T
T{ SSQ9 -> 11 22 33 }T

\ -----------------------------------------------------------------------------
CORE-EXT-ERRORS SET-ERROR-COUNT

CR .( End of Core Extension word tests) CR


