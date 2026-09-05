\ To test the ANS Forth search-order word set and search order extensions

\ This program was written by Gerry Jackson in 2006, with contributions from
\ others where indicated, and is in the public domain - it can be distributed
\ and/or modified in any way but please retain this notice.

\ This program is distributed in the hope that it will be useful,
\ but WITHOUT ANY WARRANTY; without even the implied warranty of
\ MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

\ The tests are not claimed to be comprehensive or correct 

\ ------------------------------------------------------------------------------
\ Version 0.13  Replaced 2 instances of ?DO with DO
\               Interpretive use of S" replaced by $" from utilities.fth
\         0.10 3 August 2014 Name changes to remove redefinition messages
\               "list" changed to "wordlist" in message for ORDER tests
\         0.5 1 April 2012  Tests placed in the public domain.
\         0.4 6 March 2009 { and } replaced with T{ and }T
\         0.3 20 April 2007 ANS Forth words changed to upper case
\         0.2 30 Oct 2006 updated following GForth tests to get
\             initial search order into a known state
\         0.1 Oct 2006 First version released

\ ------------------------------------------------------------------------------
\ The tests are based on John Hayes test program for the core word set

\ Words tested in this file are:
\     FORTH-WORDLIST GET-ORDER SET-ORDER ALSO ONLY FORTH GET-CURRENT
\     SET-CURRENT DEFINITIONS PREVIOUS SEARCH-WORDLIST WORDLIST FIND
\ Words not fully tested:
\     ORDER only tests that it executes, display is implementation
\           dependent and should be visually inspected

\ ------------------------------------------------------------------------------
\ Assumptions, dependencies and notes:
\     - tester.fr (or ttester.fs), errorreport.fth and utilities.fth have been
\       included prior to this file
\     - the Core word set is available and tested
\     - that ONLY FORTH DEFINITIONS will work at the start of the file
\       to ensure the search order is in a known state
\ ------------------------------------------------------------------------------

ONLY FORTH DEFINITIONS

TESTING Search-order word set

DECIMAL

VARIABLE WID1  VARIABLE WID2

\ Only execute SAVE-ORDERLIST once
: SAVE-ORDERLIST ( widn ... wid1 n -> ) DUP , ?DUP IF 0 DO , LOOP THEN ;

\ ------------------------------------------------------------------------------
TESTING FORTH-WORDLIST GET-ORDER SET-ORDER

T{ FORTH-WORDLIST WID1 ! -> }T

CREATE ORDER-LIST

T{ GET-ORDER SAVE-ORDERLIST -> }T

: GET-ORDERLIST  ( -- widn ... wid1 n )
   ORDER-LIST DUP @ CELLS DUP  ( -- ad n )
   IF
      OVER + DO I @ -1 CELLS +LOOP
   ELSE
      2DROP 0
   THEN
;

T{ GET-ORDER OVER -> GET-ORDER WID1 @ }T \ Forth wordlist at top
T{ GET-ORDER SET-ORDER -> }T             \ Effectively noop
T{ GET-ORDER -> GET-ORDERLIST }T         \ Check nothing changed
T{ GET-ORDERLIST DROP GET-ORDERLIST 2* SET-ORDER -> }T
T{ GET-ORDER -> GET-ORDERLIST DROP GET-ORDERLIST 2* }T
T{ GET-ORDERLIST SET-ORDER GET-ORDER -> GET-ORDERLIST }T

\ ------------------------------------------------------------------------------
TESTING ALSO ONLY FORTH

T{ ALSO GET-ORDER -> GET-ORDERLIST OVER SWAP 1+ }T
T{ ONLY FORTH GET-ORDER -> GET-ORDERLIST }T    \ See assumptions above

\ ------------------------------------------------------------------------------
TESTING GET-CURRENT SET-CURRENT WORDLIST (simple)

T{ GET-CURRENT -> WID1 @ }T        \ See assumptions above
T{ WORDLIST WID2 ! -> }T
T{ WID2 @ SET-CURRENT -> }T
T{ GET-CURRENT -> WID2 @ }T
T{ WID1 @ SET-CURRENT -> }T

\ ------------------------------------------------------------------------------
TESTING minimum search order list contains FORTH-WORDLIST and SET-ORDER

: SO1 SET-ORDER ;    \ In case it is unavailable in the forth wordlist

T{ ONLY FORTH-WORDLIST 1 SET-ORDER GET-ORDERLIST SO1 -> }T
T{ GET-ORDER -> GET-ORDERLIST }T

\ ------------------------------------------------------------------------------
TESTING GET-ORDER SET-ORDER with 0 and -1 number of wids argument

: SO2A GET-ORDER GET-ORDERLIST SET-ORDER ; \  To recover search order
: SO2 0 SET-ORDER SO2A ;

T{ SO2 -> 0 }T         \ 0 set-order leaves an empty search order

: SO3 -1 SET-ORDER SO2A ;
: SO4 ONLY SO2A ;

T{ SO3 -> SO4 }T       \ -1 SET-ORDER = ONLY

\ ------------------------------------------------------------------------------
TESTING DEFINITIONS PREVIOUS

T{ ONLY FORTH DEFINITIONS -> }T
T{ GET-CURRENT -> FORTH-WORDLIST }T
T{ GET-ORDER WID2 @ SWAP 1+ SET-ORDER DEFINITIONS GET-CURRENT -> WID2 @ }T
T{ GET-ORDER -> GET-ORDERLIST WID2 @ SWAP 1+ }T
T{ PREVIOUS GET-ORDER -> GET-ORDERLIST }T
T{ DEFINITIONS GET-CURRENT -> FORTH-WORDLIST }T

\ ------------------------------------------------------------------------------
TESTING SEARCH-WORDLIST WORDLIST FIND

ONLY FORTH DEFINITIONS
VARIABLE XT  ' DUP XT !
VARIABLE XTI ' ( XTI !    \ Immediate word

\ $" is an equivalent to S" in interpreter mode. It is defined in the file
\ utilities.fth and used to avoid relying on a File-Access word set extension

T{ $" DUP" WID1 @ SEARCH-WORDLIST -> XT  @ -1 }T
T{ $" ("   WID1 @ SEARCH-WORDLIST -> XTI @  1 }T
T{ $" DUP" WID2 @ SEARCH-WORDLIST ->        0 }T

: C"DUP" C" DUP" ;
: C"("  C" (" ;
: C"X" C" UNKNOWN WORD"  ;

T{ C"DUP" FIND -> XT  @ -1 }T
T{ C"("  FIND -> XTI @  1 }T
T{ C"X"   FIND -> C"X"   0 }T

\ ------------------------------------------------------------------------------
TESTING new definitions are put into the correct wordlist

: ALSOWID2 ALSO GET-ORDER WID2 @ ROT DROP SWAP SET-ORDER ;
ALSOWID2
: W2 1234  ;
DEFINITIONS
: W2 -9876 ; IMMEDIATE

ONLY FORTH
T{ W2 -> 1234 }T
DEFINITIONS
T{ W2 -> 1234 }T
ALSOWID2
T{ W2 -> -9876 }T
DEFINITIONS
T{ W2 -> -9876 }T

ONLY FORTH DEFINITIONS

: SO5  DUP IF SWAP EXECUTE THEN ;

T{ $" W2" WID1 @ SEARCH-WORDLIST SO5 -> -1  1234 }T
T{ $" W2" WID2 @ SEARCH-WORDLIST SO5 ->  1 -9876 }T

: C"W2" C" W2" ;
T{ ALSOWID2 C"W2" FIND SO5 ->  1 -9876 }T
T{ PREVIOUS C"W2" FIND SO5 -> -1  1234 }T

\ ------------------------------------------------------------------------------
TESTING ORDER  \ Should display search order and compilation wordlist

CR .( ONLY FORTH DEFINITIONS search order and compilation wordlist) CR
T{ ONLY FORTH DEFINITIONS ORDER -> }T

CR .( Plus another unnamed wordlist at the head of the search order) CR
T{ ALSOWID2 DEFINITIONS ORDER -> }T

\ ------------------------------------------------------------------------------

SEARCHORDER-ERRORS SET-ERROR-COUNT

CR .( End of Search Order word tests) CR

ONLY FORTH DEFINITIONS      \ Leave search order in the standard state
