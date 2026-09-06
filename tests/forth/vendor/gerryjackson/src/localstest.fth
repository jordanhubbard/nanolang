\ To test the ANS Forth and Forth 2012 Locals word set

\ This program was written by Gerry Jackson in 2015 and is in the public domain
\ - it can be distributed and/or modified in any way but please retain this
\ notice.

\ This program is distributed in the hope that it will be useful,
\ but WITHOUT ANY WARRANTY; without even the implied warranty of
\ MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

\ The tests are not claimed to be comprehensive or correct 

\ ------------------------------------------------------------------------------
\ Version 0.13 13 Nov 2015 Priority of locals tests made conditional on the
\              the required search-order words being available
\         0.11 25 April 2015 Initial release

\ ------------------------------------------------------------------------------
\ The tests are based on John Hayes test program for the core word set
\ and requires those files to have been loaded

\ Words tested in this file are:
\     {: TO (LOCAL) 

\ Words not tested:
\     LOCALS|  (designated obsolescent in Forth 2012)
\ ------------------------------------------------------------------------------
\ Assumptions, dependencies and notes:
\     - tester.fr (or ttester.fs), errorreport.fth and utilities.fth have been
\       included prior to this file
\     - the Core word set is available and tested
\     - some tests at the end require the following words from the Search-Order
\       word set WORDLIST GET-CURRENT SET-CURRENT GET-ORDER SET-ORDER PREVIOUS.
\       If any these are not available the tests will be ignored.
\ ------------------------------------------------------------------------------

TESTING Locals word set

DECIMAL

\ Syntax is : foo ... {: <args>* [| <vals>*] [-- <out>*] :} ... ;
\ <arg>s are initialised from the data stack
\ <val>s are uninitialised
\ <out>s are ignored (treated as a comment)

TESTING null locals

T{ : LT0 {: :} ; 0 LT0 -> 0 }T
T{ : LT1 {: | :} ; 1 LT1 -> 1 }T
T{ : LT2 {: -- :} ; 2 LT2 -> 2 }T
T{ : LT3 {: | -- :} ; 3 LT3 -> 3 }T

TESTING <arg>s and TO <arg>

T{ : LT4 {: A :} ; 4 LT4 -> }T
T{ : LT5 {: A :} A ; 5 LT5 -> 5 }T
T{ : LT6 DEPTH {: A B :} DEPTH A B ; 6 LT6 -> 0 6 1 }T
T{ : LT7 {: A B :} B A ; 7 8 LT7 -> 8 7 }T
T{ : LT8 {: A B :} B A 11 TO A A B 12 TO B B A ; 9 10 LT8 -> 10 9 11 10 12 11 }T
T{ : LT9 2DUP + {: A B C :} C B A ; 13 14 LT9 -> 27 14 13 }T

TESTING | <val>s and TO <val>s
T{ : LT10 {: A B | :} B 2* A + ; 15 16 LT10 -> 47 }T
T{ : LT11 {: A | B :} A 2* ; 17 18 LT11 -> 17 36 }T
T{ : LT12 {: A | B C :} 20 TO B A 21 TO A 22 TO C A C B ; 19 LT12 -> 19 21 22 20 }T
T{ : LT13 {: | A :} ; 23 LT13 -> 23 }T
T{ : LT14 {: | A B :} 24 TO B 25 TO A A B ; 26 LT14 -> 26 25 24 }T

TESTING -- ignores everything up to :}
T{ : LT15 {: -- DUP SWAP OVER :} DUP 28 SWAP OVER ; 27 LT15 -> 27 28 27 28 }T
T{ : LT16 {: | A -- this should be ignored :} TO A A + ; 29 30 LT16 -> 59 }T
T{ : LT17 {: A -- A + 1 :} A 1+ ; 31 LT17 -> 32 }T
T{ : LT18 {: A | B -- 2A+B :} TO B A 2* B + ; 33 34 LT18 -> 101 }T

TESTING local names supersede global names and numbers
T{ : LT19 {: DUP DROP | SWAP -- OVER :} 35 TO SWAP SWAP DUP DROP OVER ; -> }T
T{ 36 37 38 LT19 -> 36 35 37 38 37 }T
T{ HEX : LT20 {: BEAD DEAF :} DEAF BEAD ; BEEF DEAD LT20 -> DEAD BEEF }T DECIMAL

TESTING definition with locals calling another with same name locals
T{ : LT21 {: A | B :} 39 TO B A B ; -> }T
T{ : LT22 {: B | A :} 40 TO A A 2* B 2* LT21 A B ; -> }T
T{ 41 LT22 -> 80 82 39 40 41 }T

TESTING locals in :NONAME & DOES>
T{ 42 43 :NONAME {: W X | Y -- DUP :} 44 TO Y X W Y DUP ; EXECUTE -> 43 42 44 44 }T
T{ : LT23 {: P Q :} CREATE P Q 2* + ,
                    DOES> @ ROT ROT {: P Q | R -- DUP :} TO R Q R P ; -> }T
T{ 45 46 LT23 LT24 -> }T
T{ 47 48 LT24 -> 48 137 47 }T

TESTING locals in control structures
T{ : LT25 {: A B :} IF A ELSE B THEN ; -1 50 51 LT25 -> 50 }T
T{ 0 52 53 LT25 -> 53 }T
T{ : LT26 {: A :} 0 BEGIN A WHILE 2 + A 1- TO A REPEAT ; -> }T
T{ 5 LT26 -> 10 }T
T{ : LT27 {: A :} 0 BEGIN A 1- TO A 3 + A 0= UNTIL ; -> }T
T{ 5 LT27 -> 15 }T
T{ : LT28 1+ {: A B :} B A DO I LOOP ; 54 58 LT28 -> 54 55 56 57 58 }T
T{ : LT29 {: I J :} 2 0 DO 5 3 DO I J LOOP LOOP ; -> }T
T{ 59 60 LT29 -> 59 60 59 60 59 60 59 60 }T


TESTING recursion with locals
T{ : LT30 {: A B :} A 0> IF A B * A 1- B 10 * RECURSE A B THEN ; -> }T
T{ 3 10 LT30 -> 30 200 1000 1 1000 2 100 3 10 }T

TESTING system supplies at least 16 locals

: LOC-ENVQ S" #LOCALS" ENVIRONMENT? ;
T{ LOC-ENVQ SWAP 15 > -> TRUE TRUE }T
T{ : LT31 {: A B C D E F G H I J K L M N O P :}
             P O N M L K J I H G F E D C B A ; -> }T
T{ 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 LT31
          -> 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 }T

TESTING (LOCAL)
T{ : LOCAL BL WORD COUNT (LOCAL) ; IMMEDIATE -> }T
T{ : END-LOCALS 99 0 (LOCAL) ; IMMEDIATE     -> }T
: LT32 LOCAL A LOCAL B LOCAL C END-LOCALS A B C ; 61 62 63 LT32 -> 63 62 61 }T

\ ------------------------------------------------------------------------------
\ These tests require Search-order words WORDLIST GET-CURRENT SET-CURRENT
\ GET-ORDER SET-ORDER PREVIOUS. If any of these are not available the following
\ tests will be ignored except for the simple test
[?UNDEF] WORDLIST \? [?UNDEF] GET-CURRENT \? [?UNDEF] SET-CURRENT
\? [?UNDEF] GET-ORDER \? [?UNDEF] SET-ORDER

\? TESTING that local names are always found first & that they are not available
\ after the end of a definition.

\ Simple test
: LT36 68 ;
T{ : LT37 {: LT36 :} LT36 ; 69 LT37 LT36 -> 69 68 }T 

\? WORDLIST CONSTANT LTWL1
\? WORDLIST CONSTANT LTWL2
\? GET-CURRENT LTWL1 SET-CURRENT
\? : LT33 64 ;       \ Define LT33 in LTWL1 wordlist
\? LTWL2 SET-CURRENT
\? : LT33 65 ;       \ Redefine LT33 in LTWL2 wordlist
\? SET-CURRENT
\? : ALSO-LTWL  ( wid -- )  >R GET-ORDER R> SWAP 1+ SET-ORDER ;
\? LTWL1 ALSO-LTWL   \ Add LTWL1 to search-order
\? T{ : LT34 {: LT33 :} LT33 ; 66 LT34 LT33 -> 66 64 }T
\? T{ : LT35 {: LT33 :} LT33 LTWL2 ALSO-LTWL LT33 PREVIOUS LT33 PREVIOUS LT33 ;
\?    -> }T
\ If the next test fails the system may be left with LTWL2 and/or LTWL1 in the
\ search order
\? T{ 67 LT35 -> 67 67 67 67 }T
[?ELSE]
\? CR CR
\? .( Some search-order words not present - priority of Locals not fully tested)
\? CR
[?THEN]

\ ------------------------------------------------------------------------------

LOCALS-ERRORS SET-ERROR-COUNT    \ For final error report

CR .( End of Locals word set tests. ) .S
