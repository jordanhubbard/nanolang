\ To test the ANS Forth Exception word set and extension words

\ This program was written by Gerry Jackson in 2006, with contributions from
\ others where indicated, and is in the public domain - it can be distributed
\ and/or modified in any way but please retain this notice.

\ This program is distributed in the hope that it will be useful,
\ but WITHOUT ANY WARRANTY; without even the implied warranty of
\ MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

\ The tests are not claimed to be comprehensive or correct 

\ ------------------------------------------------------------------------------
\ Version 0.13 13 Nov 2015 C6 rewritten to avoid use of CASE etc and hence
\              dependence on the Core extension word set.     
\         0.4 1 April 2012  Tests placed in the public domain.
\         0.3 6 March 2009 { and } replaced with T{ and }T
\         0.2 20 April 2007 ANS Forth words changed to upper case
\         0.1 Oct 2006 First version released

\ ------------------------------------------------------------------------------
\ The tests are based on John Hayes test program for the core word set
\
\ Words tested in this file are:
\     CATCH THROW ABORT ABORT"
\
\ ------------------------------------------------------------------------------
\ Assumptions and dependencies:
\     - the forth system under test throws an exception with throw
\       code -13 for a word not found by the text interpreter. The
\       undefined word used is $$qweqweqwert$$,  if this happens to be
\       a valid word in your system change the definition of t7 below
\     - tester.fr (or ttester.fs), errorreport.fth and utilities.fth have been
\       included prior to this file
\     - the Core word set available and tested
\     - CASE, OF, ENDOF and ENDCASE from the core extension wordset
\       are present and work correctly
\ ------------------------------------------------------------------------------
TESTING CATCH THROW

DECIMAL

: T1 9 ;
: C1 1 2 3 ['] T1 CATCH ;
T{ C1 -> 1 2 3 9 0 }T         \ No THROW executed

: T2 8 0 THROW ;
: C2 1 2 ['] T2 CATCH ;
T{ C2 -> 1 2 8 0 }T            \ 0 THROW does nothing

: T3 7 8 9 99 THROW ;
: C3 1 2 ['] T3 CATCH ;
T{ C3 -> 1 2 99 }T            \ Restores stack to CATCH depth

: T4 1- DUP 0> IF RECURSE ELSE 999 THROW -222 THEN ;
: C4 3 4 5 10 ['] T4 CATCH -111 ;
T{ C4 -> 3 4 5 0 999 -111 }T   \ Test return stack unwinding

: T5 2DROP 2DROP 9999 THROW ;
: C5 1 2 3 4 ['] T5 CATCH            \ Test depth restored correctly
   DEPTH >R DROP 2DROP 2DROP R> ;   \ after stack has been emptied
T{ C5 -> 5 }T

\ ------------------------------------------------------------------------------
TESTING ABORT ABORT"

-1  CONSTANT EXC_ABORT
-2  CONSTANT EXC_ABORT"
-13 CONSTANT EXC_UNDEF
: T6 ABORT ;

\ The 77 in T10 is necessary for the second ABORT" test as the data stack
\ is restored to a depth of 2 when THROW is executed. The 77 ensures the top
\ of stack value is known for the results check

: T10 77 SWAP ABORT" This should not be displayed" ;
: C6 CATCH
   >R   R@ EXC_ABORT  = IF 11
   ELSE R@ EXC_ABORT" = IF 12
   ELSE R@ EXC_UNDEF  = IF 13
   THEN THEN THEN R> DROP
;

T{ 1 2 ' T6 C6  -> 1 2 11 }T     \ Test that ABORT is caught
T{ 3 0 ' T10 C6 -> 3 77 }T       \ ABORT" does nothing
T{ 4 5 ' T10 C6 -> 4 77 12 }T    \ ABORT" caught, no message

\ ------------------------------------------------------------------------------
TESTING a system generated exception

: T7 S" 333 $$QWEQWEQWERT$$ 334" EVALUATE 335 ;
: T8 S" 222 T7 223" EVALUATE 224 ;
: T9 S" 111 112 T8 113" EVALUATE 114 ;

T{ 6 7 ' T9 C6 3 -> 6 7 13 3 }T         \ Test unlinking of sources

\ ------------------------------------------------------------------------------

EXCEPTION-ERRORS SET-ERROR-COUNT

CR .( End of Exception word tests) CR

