\ To test part of the Forth 2012 Facility word set

\ This program was written by Gerry Jackson in 2015, with contributions from
\ others where indicated, and is in the public domain - it can be distributed
\ and/or modified in any way but please retain this notice.

\ This program is distributed in the hope that it will be useful,
\ but WITHOUT ANY WARRANTY; without even the implied warranty of
\ MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

\ The tests are not claimed to be comprehensive or correct 

\ ------------------------------------------------------------------------------
\ Version 0.13 Assumptions and dependencies added
\         0.11 25 April 2015 Added tests for BEGIN-STRUCTURE END-STRUCTURE +FIELD
\              FIELD: CFIELD:
\ -----------------------------------------------------------------------------
\ The tests are based on John Hayes test program for the core word set

\ Words tested in this file are: +FIELD BEGIN-STRUCTURE CFIELD: END-STRUCTURE
\      FIELD:

\ ------------------------------------------------------------------------------
\ Assumptions and dependencies:
\     - tester.fr (or ttester.fs), errorreport.fth and utilities.fth have been
\       included prior to this file
\     - the Core word set is available and tested
\ -----------------------------------------------------------------------------
TESTING Facility words

DECIMAL
\ -----------------------------------------------------------------------------
TESTING BEGIN-STRUCTURE END-STRUCTURE +FIELD

T{ BEGIN-STRUCTURE STRCT1
   END-STRUCTURE   -> }T
T{ STRCT1 -> 0 }T

T{ BEGIN-STRUCTURE STRCT2
      1 CHARS +FIELD F21
      2 CHARS +FIELD F22
      0 +FIELD F23
      1 CELLS +FIELD F24
   END-STRUCTURE   -> }T

T{ STRCT2 -> 3 chars 1 cells + }T   \ +FIELD doesn't align
T{ 0 F21 -> 0 }T
T{ 0 F22 -> 1 }T
T{ 0 F23 -> 3 }T
T{ 0 F24 -> 3 }T
T{ 5 F23 -> 8 }T

T{ CREATE S21 STRCT2 ALLOT -> }T
T{ 11 S21 F21 C! -> }T
T{ 22 S21 F22 C! -> }T
T{ 33 S21 F23 C! -> }T
T{ S21 F23 C@ -> 33 }T
T{ 44 S21 F24 C! -> }T
T{ S21 F21 C@ -> 11 }T
T{ S21 F22 C@ -> 22 }T
T{ S21 F23 C@ -> 44 }T
T{ S21 F24 C@ -> 44 }T

T{ CREATE S22 STRCT2 ALLOT -> }T
T{ 55 S22 F21 C! -> }T
T{ 66 S22 F22 C! -> }T
T{ S21 F21 C@ -> 11 }T
T{ S21 F22 C@ -> 22 }T
T{ S22 F21 C@ -> 55 }T
T{ S22 F22 C@ -> 66 }T

TESTING FIELD: CFIELD:

T{ BEGIN-STRUCTURE STRCT3
      FIELD:  F31
      FIELD:  F32
      CFIELD: CF31
      CFIELD: CF32
      CFIELD: CF33
      FIELD:  F33
   END-STRUCTURE -> }T

T{ 0 F31  CELL+ -> 0 F32  }T
T{ 0 CF31 CHAR+ -> 0 CF32 }T
T{ 0 CF32 CHAR+ -> 0 CF33 }T
T{ 0 CF33 CHAR+ ALIGNED -> 0 F33 }T
T{ 0 F33 ALIGNED -> 0 F33 }T


T{ CREATE S31 STRCT3 ALLOT -> }T
T{ 1 S31 F31   ! -> }T
T{ 2 S31 F32   ! -> }T
T{ 3 S31 CF31 C! -> }T
T{ 4 S31 CF32 C! -> }T
T{ 5 S31 F33   ! -> }T
T{ S31 F31   @ -> 1 }T
T{ S31 F32   @ -> 2 }T
T{ S31 CF31 C@ -> 3 }T
T{ S31 CF32 C@ -> 4 }T
T{ S31 F33   @ -> 5 }T

TESTING Nested structures

T{ BEGIN-STRUCTURE STRCT4
      STRCT2 +FIELD F41
      ALIGNED STRCT3 +FIELD F42
      3 +FIELD F43
      STRCT2 +FIELD F44
   END-STRUCTURE        -> }T
T{ STRCT4 -> STRCT2 ALIGNED STRCT3 + 3 + STRCT2 + }T

T{ CREATE S41 STRCT4 ALLOT -> }T
T{ 21 S41 F41 F21  C! -> }T
T{ 22 S41 F41 F22  C! -> }T
T{ 23 S41 F41 F23  C! -> }T
T{ 24 S41 F42 F31   ! -> }T
T{ 25 S41 F42 F32   ! -> }T
T{ 26 S41 F42 CF31 C! -> }T
T{ 27 S41 F42 CF32 C! -> }T
T{ 28 S41 F42 CF33 C! -> }T
T{ 29 S41 F42 F33   ! -> }T
T{ 30 S41 F44 F21  C! -> }T
T{ 31 S41 F44 F22  C! -> }T
T{ 32 S41 F44 F23  C! -> }T

T{ S41 F41 F21  C@ -> 21 }T
T{ S41 F41 F22  C@ -> 22 }T
T{ S41 F41 F23  C@ -> 23 }T
T{ S41 F42 F31   @ -> 24 }T
T{ S41 F42 F32   @ -> 25 }T
T{ S41 F42 CF31 C@ -> 26 }T
T{ S41 F42 CF32 C@ -> 27 }T
T{ S41 F42 CF33 C@ -> 28 }T
T{ S41 F42 F33   @ -> 29 }T
T{ S41 F44 F21  C@ -> 30 }T
T{ S41 F44 F22  C@ -> 31 }T
T{ S41 F44 F23  C@ -> 32 }T

\ -----------------------------------------------------------------------------

FACILITY-ERRORS SET-ERROR-COUNT

CR .( End of Facility word tests) CR
