\ To test the ANS Forth Memory-Allocation word set

\ This program was written by Gerry Jackson in 2006, with contributions from
\ others where indicated, and is in the public domain - it can be distributed
\ and/or modified in any way but please retain this notice.

\ This program is distributed in the hope that it will be useful,
\ but WITHOUT ANY WARRANTY; without even the implied warranty of
\ MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

\ The tests are not claimed to be comprehensive or correct 

\ ------------------------------------------------------------------------------
\ Version 0.11 25 April 2015 Now checks memory region is unchanged following a
\              RESIZE. @ and ! in allocated memory.
\         0.8 10 January 2013, Added CHARS and CHAR+ where necessary to correct
\             the assumption that 1 CHARS = 1
\         0.7 1 April 2012  Tests placed in the public domain.
\         0.6 30 January 2011 CHECKMEM modified to work with ttester.fs
\         0.5 30 November 2009 <FALSE> replaced with FALSE
\         0.4 9 March 2009 Aligned test improved and data space pointer tested
\         0.3 6 March 2009 { and } replaced with T{ and }T
\         0.2 20 April 2007  ANS Forth words changed to upper case
\         0.1 October 2006 First version released

\ ------------------------------------------------------------------------------
\ The tests are based on John Hayes test program for the core word set

\ Words tested in this file are:
\     ALLOCATE FREE RESIZE
\     
\ ------------------------------------------------------------------------------
\ Assumptions, dependencies and notes:
\     - tester.fr (or ttester.fs), errorreport.fth and utilities.fth have been
\       included prior to this file
\     - the Core word set is available and tested
\     - that 'addr -1 ALLOCATE' and 'addr -1 RESIZE' will return an error
\     - testing FREE failing is not done as it is likely to crash the system
\ ------------------------------------------------------------------------------

TESTING Memory-Allocation word set

DECIMAL

\ ------------------------------------------------------------------------------
TESTING ALLOCATE FREE RESIZE

VARIABLE ADDR1
VARIABLE DATSP

HERE DATSP !
T{ 100 ALLOCATE SWAP ADDR1 ! -> 0 }T
T{ ADDR1 @ ALIGNED -> ADDR1 @ }T   \ Test address is aligned
T{ HERE -> DATSP @ }T            \ Check data space pointer is unchanged
T{ ADDR1 @ FREE -> 0 }T

T{ 99 ALLOCATE SWAP ADDR1 ! -> 0 }T
T{ ADDR1 @ ALIGNED -> ADDR1 @ }T
T{ ADDR1 @ FREE -> 0 }T

T{ 50 CHARS ALLOCATE SWAP ADDR1 ! -> 0 }T

: WRITEMEM 0 DO I 1+ OVER C! CHAR+ LOOP DROP ;   ( ad n -- )

\ CHECKMEM is defined this way to maintain compatibility with both
\ tester.fr and ttester.fs which differ in their definitions of T{

: CHECKMEM  ( ad n --- )
   0
   DO
      >R
      T{ R@ C@ -> R> I 1+ SWAP >R }T
      R> CHAR+
   LOOP
   DROP
;

ADDR1 @ 50 WRITEMEM ADDR1 @ 50 CHECKMEM

T{ ADDR1 @ 28 CHARS RESIZE SWAP ADDR1 ! -> 0 }T
ADDR1 @ 28 CHECKMEM

T{ ADDR1 @ 200 CHARS RESIZE SWAP ADDR1 ! -> 0 }T
ADDR1 @ 28 CHECKMEM

\ ------------------------------------------------------------------------------
TESTING failure of RESIZE and ALLOCATE (unlikely to be enough memory)

\ This test relies on the previous test having passed

VARIABLE RESIZE-OK
T{ ADDR1 @ -1 CHARS RESIZE 0= DUP RESIZE-OK ! -> ADDR1 @ FALSE }T

\ Check unRESIZEd allocation is unchanged following RESIZE failure 
: MEM?  RESIZE-OK @ 0= IF ADDR1 @ 28 CHECKMEM THEN ;   \ Avoid using [IF]
MEM?

T{ ADDR1 @ FREE -> 0 }T   \ Tidy up

T{ -1 ALLOCATE SWAP DROP 0= -> FALSE }T      \ Memory allocate failed

\ ------------------------------------------------------------------------------
TESTING @  and ! work in ALLOCATEd memory (provided by Peter Knaggs)

: WRITE-CELL-MEM ( ADDR N -- )
  1+ 1 DO I OVER ! CELL+ LOOP DROP
;

: CHECK-CELL-MEM ( ADDR N -- )
  1+ 1 DO
    I SWAP >R >R
    T{ R> ( I ) -> R@ ( ADDR ) @ }T
    R> CELL+
  LOOP DROP
;

\ Cell based access to the heap

T{ 50 CELLS ALLOCATE SWAP ADDR1 ! -> 0 }T
ADDR1 @ 50 WRITE-CELL-MEM
ADDR1 @ 50 CHECK-CELL-MEM

\ ------------------------------------------------------------------------------

MEMORY-ERRORS SET-ERROR-COUNT

CR .( End of Memory-Allocation word tests) CR
