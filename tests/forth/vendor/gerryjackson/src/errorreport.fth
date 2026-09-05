\ To collect and report on the number of errors resulting from running the
\ ANS Forth and Forth 2012 test programs

\ This program was written by Gerry Jackson in 2015, and is in the public
\ domain - it can be distributed and/or modified in any way but please
\ retain this notice.

\ This program is distributed in the hope that it will be useful,
\ but WITHOUT ANY WARRANTY; without even the implied warranty of
\ MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

\ ------------------------------------------------------------------------------
\ This file is INCLUDED after Core tests are complete and only uses Core words
\ already tested. The purpose of this file is to count errors in test results
\ and present them as a summary at the end of the tests.

DECIMAL

VARIABLE TOTAL-ERRORS

: ERROR-COUNT  ( "name" n1 -- n2 )  \ n2 = n1 + 1cell
   CREATE  DUP , CELL+
   DOES>  ( -- offset ) @     \ offset in address units
;

0     \ Offset into ERRORS[] array
ERROR-COUNT CORE-ERRORS          ERROR-COUNT CORE-EXT-ERRORS
ERROR-COUNT DOUBLE-ERRORS        ERROR-COUNT EXCEPTION-ERRORS
ERROR-COUNT FACILITY-ERRORS      ERROR-COUNT FILE-ERRORS
ERROR-COUNT LOCALS-ERRORS        ERROR-COUNT MEMORY-ERRORS
ERROR-COUNT SEARCHORDER-ERRORS   ERROR-COUNT STRING-ERRORS
ERROR-COUNT TOOLS-ERRORS         ERROR-COUNT BLOCK-ERRORS
CREATE ERRORS[] DUP ALLOT CONSTANT #ERROR-COUNTS

\ SET-ERROR-COUNT called at the end of each test file with its own offset into
\ the ERRORS[] array. #ERRORS is in files tester.fr and ttester.fs

: SET-ERROR-COUNT  ( offset -- )
   #ERRORS @ SWAP ERRORS[] + !
   #ERRORS @ TOTAL-ERRORS +!
   0 #ERRORS !
;

: INIT-ERRORS  ( -- )
   ERRORS[] #ERROR-COUNTS OVER + SWAP DO -1 I ! 1 CELLS +LOOP
   0 TOTAL-ERRORS !
   CORE-ERRORS SET-ERROR-COUNT
;

INIT-ERRORS

\ Report summary of errors

25 CONSTANT MARGIN

: SHOW-ERROR-LINE  ( n caddr u -- )
   CR SWAP OVER TYPE MARGIN - ABS >R
   DUP -1 = IF DROP R> 1- SPACES ." -" ELSE
   R> .R THEN
;

: SHOW-ERROR-COUNT  ( caddr u offset -- )
   ERRORS[] + @ ROT ROT SHOW-ERROR-LINE
;

: HLINE  ( -- )  CR ." ---------------------------"  ;

: REPORT-ERRORS
   HLINE
   CR 8 SPACES ." Error Report"
   CR ." Word Set" 13 SPACES ." Errors"
   HLINE
   S" Core" CORE-ERRORS SHOW-ERROR-COUNT
   S" Core extension" CORE-EXT-ERRORS SHOW-ERROR-COUNT
   S" Block" BLOCK-ERRORS SHOW-ERROR-COUNT
   S" Double number" DOUBLE-ERRORS SHOW-ERROR-COUNT
   S" Exception" EXCEPTION-ERRORS SHOW-ERROR-COUNT
   S" Facility" FACILITY-ERRORS SHOW-ERROR-COUNT
   S" File-access" FILE-ERRORS SHOW-ERROR-COUNT
   S" Locals"    LOCALS-ERRORS SHOW-ERROR-COUNT
   S" Memory-allocation" MEMORY-ERRORS SHOW-ERROR-COUNT
   S" Programming-tools" TOOLS-ERRORS SHOW-ERROR-COUNT
   S" Search-order" SEARCHORDER-ERRORS SHOW-ERROR-COUNT
   S" String" STRING-ERRORS SHOW-ERROR-COUNT
   HLINE
   TOTAL-ERRORS @ S" Total" SHOW-ERROR-LINE
   HLINE CR CR
;
