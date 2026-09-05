\ This file contains the code for ttester, a utility for testing Forth words,
\ as developed by several authors (see below), together with some explanations
\ of its use.

\ ttester is based on the original tester suite by Hayes:
\ From: John Hayes S1I
\ Subject: tester.fr
\ Date: Mon, 27 Nov 95 13:10:09 PST  
\ (C) 1995 JOHNS HOPKINS UNIVERSITY / APPLIED PHYSICS LABORATORY
\ MAY BE DISTRIBUTED FREELY AS LONG AS THIS COPYRIGHT NOTICE REMAINS.
\ VERSION 1.1
\ All the subsequent changes have been placed in the public domain.
\ The primary changes from the original are the replacement of "{" by "T{"
\ and "}" by "}T" (to avoid conflicts with the uses of { for locals and }
\ for FSL arrays), modifications so that the stack is allowed to be non-empty
\ before T{, and extensions for the handling of floating point tests.
\ Code for testing equality of floating point values comes
\ from ftester.fs written by David N. Williams, based on the idea of
\ approximate equality in Dirk Zoller's float.4th.
\ Further revisions were provided by Anton Ertl, including the ability
\ to handle either integrated or separate floating point stacks.
\ Revision history and possibly newer versions can be found at
\ http://www.complang.tuwien.ac.at/cvsweb/cgi-bin/cvsweb/gforth/test/ttester.fs
\ Explanatory material and minor reformatting (no code changes) by
\ C. G. Montgomery March 2009, with helpful comments from David Williams
\ and Krishna Myneni.

\ Usage:

\ The basic usage takes the form  T{ <code> -> <expected stack> }T .
\ This executes  <code>  and compares the resulting stack contents with
\ the  <expected stack>  values, and reports any discrepancy between the
\ two sets of values.
\ For example:
\ T{ 1 2 3 swap -> 1 3 2 }T  ok
\ T{ 1 2 3 swap -> 1 2 2 }T INCORRECT RESULT: T{ 1 2 3 swap -> 1 2 2 }T ok
\ T{ 1 2 3 swap -> 1 2 }T WRONG NUMBER OF RESULTS: T{ 1 2 3 swap -> 1 2 }T ok

\ Floating point testing can involve further complications.  The code
\ attempts to determine whether floating-point support is present, and
\ if so, whether there is a separate floating-point stack, and behave
\ accordingly.  The CONSTANTs HAS-FLOATING and HAS-FLOATING-STACK
\ contain the results of its efforts, so the behavior of the code can
\ be modified by the user if necessary.

\ Then there are the perennial issues of floating point value
\ comparisons.  Exact equality is specified by SET-EXACT (the
\ default).  If approximate equality tests are desired, execute
\ SET-NEAR .  Then the FVARIABLEs REL-NEAR (default 1E-12) and
\ ABS-NEAR (default 0E) contain the values to be used in comparisons
\ by the (internal) word FNEARLY= .

\ When there is not a separate floating point stack and you want to
\ use approximate equality for FP values, it is necessary to identify
\ which stack items are floating point quantities.  This can be done
\ by replacing the closing }T with a version that specifies this, such
\ as RRXR}T which identifies the stack picture ( r r x r ).  The code
\ provides such words for all combinations of R and X with up to four
\ stack items.  They can be used with either an integrated or separate
\ floating point stacks. Adding more if you need them is
\ straightforward; see the examples in the source.  Here is an example
\ which also illustrates controlling the precision of comparisons:

\   SET-NEAR
\   1E-6 REL-NEAR F!
\   T{ S" 3.14159E" >FLOAT -> -1E FACOS TRUE RX}T

\ The word ERROR is now vectored, so that its action can be changed by
\ the user (for example, to add a counter for the number of errors).
\ The default action ERROR1 can be used as a factor in the display of
\ error reports.

\ Loading ttester.fs does not change BASE.  Remember that floating point input
\ is ambiguous if the base is not decimal.

\ The file defines some 70 words in all, but in most cases only the
\ ones mentioned above will be needed for successful testing.

BASE @
DECIMAL

VARIABLE ACTUAL-DEPTH			\ stack record
CREATE ACTUAL-RESULTS 32 CELLS ALLOT
VARIABLE START-DEPTH
VARIABLE XCURSOR      \ for ...}T
VARIABLE ERROR-XT

: ERROR ERROR-XT @ EXECUTE ;   \ for vectoring of error reporting

: "FLOATING" S" FLOATING" ;    \ only compiled S" in CORE
: "FLOATING-STACK" S" FLOATING-STACK" ;
"FLOATING" ENVIRONMENT? [IF]
    [IF]
        TRUE
    [ELSE]
        FALSE
    [THEN]
[ELSE]
    FALSE
[THEN] CONSTANT HAS-FLOATING
"FLOATING-STACK" ENVIRONMENT? [IF]
    [IF]
        TRUE
    [ELSE]
        FALSE
    [THEN]
[ELSE]            \ We don't know whether the FP stack is separate.
    HAS-FLOATING  \ If we have FLOATING, we assume it is.
[THEN] CONSTANT HAS-FLOATING-STACK

HAS-FLOATING [IF]
    \ Set the following to the relative and absolute tolerances you
    \ want for approximate float equality, to be used with F~ in
    \ FNEARLY=.  Keep the signs, because F~ needs them.
    FVARIABLE REL-NEAR 1E-12 REL-NEAR F!
    FVARIABLE ABS-NEAR 0E    ABS-NEAR F!

    \ When EXACT? is TRUE, }F uses FEXACTLY=, otherwise FNEARLY=.
    
    TRUE VALUE EXACT?
    : SET-EXACT  ( -- )   TRUE TO EXACT? ;
    : SET-NEAR   ( -- )  FALSE TO EXACT? ;

    : FEXACTLY=  ( F: X Y -- S: FLAG )
        (
        Leave TRUE if the two floats are identical.
        )
        0E F~ ;
    
    : FABS=  ( F: X Y -- S: FLAG )
        (
        Leave TRUE if the two floats are equal within the tolerance
        stored in ABS-NEAR.
        )
        ABS-NEAR F@ F~ ;
    
    : FREL=  ( F: X Y -- S: FLAG )
        (
        Leave TRUE if the two floats are relatively equal based on the
        tolerance stored in ABS-NEAR.
        )
        REL-NEAR F@ FNEGATE F~ ;

    : F2DUP  FOVER FOVER ;
    : F2DROP FDROP FDROP ;
    
    : FNEARLY=  ( F: X Y -- S: FLAG )
        (
        Leave TRUE if the two floats are nearly equal.  This is a 
        refinement of Dirk Zoller's FEQ to also allow X = Y, including
        both zero, or to allow approximately equality when X and Y are too
        small to satisfy the relative approximation mode in the F~ 
        specification.
        )
        F2DUP FEXACTLY= IF F2DROP TRUE EXIT THEN
        F2DUP FREL=     IF F2DROP TRUE EXIT THEN
        FABS= ;

    : FCONF= ( R1 R2 -- F )
        EXACT? IF
            FEXACTLY=
        ELSE
            FNEARLY=
        THEN ;
[THEN]

HAS-FLOATING-STACK [IF]
    VARIABLE ACTUAL-FDEPTH
    CREATE ACTUAL-FRESULTS 32 FLOATS ALLOT
    VARIABLE START-FDEPTH
    VARIABLE FCURSOR

    : EMPTY-FSTACK ( ... -- ... )
        FDEPTH START-FDEPTH @ < IF
            FDEPTH START-FDEPTH @ SWAP DO 0E LOOP
        THEN
        FDEPTH START-FDEPTH @ > IF
            FDEPTH START-FDEPTH @ DO FDROP LOOP
        THEN ;
    
    : F{ ( -- )
        FDEPTH START-FDEPTH ! 0 FCURSOR ! ;

    : F-> ( ... -- ... )
        FDEPTH DUP ACTUAL-FDEPTH !
        START-FDEPTH @ > IF
            FDEPTH START-FDEPTH @ - 0 DO ACTUAL-FRESULTS I FLOATS + F! LOOP
        THEN ;

    : F} ( ... -- ... )
        FDEPTH ACTUAL-FDEPTH @ = IF
            FDEPTH START-FDEPTH @ > IF
                FDEPTH START-FDEPTH @ - 0 DO
                    ACTUAL-FRESULTS I FLOATS + F@ FCONF= INVERT IF
                        S" INCORRECT FP RESULT: " ERROR LEAVE
                    THEN
                LOOP
            THEN
        ELSE
            S" WRONG NUMBER OF FP RESULTS: " ERROR
        THEN ;

    : F...}T ( -- )
        FCURSOR @ START-FDEPTH @ + ACTUAL-FDEPTH @ <> IF
            S" NUMBER OF FLOAT RESULTS BEFORE '->' DOES NOT MATCH ...}T SPECIFICATION: " ERROR
        ELSE FDEPTH START-FDEPTH @ = 0= IF
            S" NUMBER OF FLOAT RESULTS BEFORE AND AFTER '->' DOES NOT MATCH: " ERROR
        THEN THEN ;

    
    : FTESTER ( R -- )
        FDEPTH 0= ACTUAL-FDEPTH @ FCURSOR @ START-FDEPTH @ + 1+ < OR IF
            S" NUMBER OF FLOAT RESULTS AFTER '->' BELOW ...}T SPECIFICATION: " ERROR 
        ELSE ACTUAL-FRESULTS FCURSOR @ FLOATS + F@ FCONF= 0= IF
            S" INCORRECT FP RESULT: " ERROR
        THEN THEN
        1 FCURSOR +! ;
        
[ELSE]
    : EMPTY-FSTACK ;
    : F{ ;
    : F-> ;
    : F} ;
    : F...}T ;

    HAS-FLOATING [IF]
    : COMPUTE-CELLS-PER-FP ( -- U )
        DEPTH 0E DEPTH 1- >R FDROP R> SWAP - ;

    COMPUTE-CELLS-PER-FP CONSTANT CELLS-PER-FP

    : FTESTER ( R -- )
        DEPTH CELLS-PER-FP < ACTUAL-DEPTH @ XCURSOR @ START-DEPTH @ + CELLS-PER-FP + < OR IF
            S" NUMBER OF RESULTS AFTER '->' BELOW ...}T SPECIFICATION: " ERROR EXIT
        ELSE ACTUAL-RESULTS XCURSOR @ CELLS + F@ FCONF= 0= IF
            S" INCORRECT FP RESULT: " ERROR
        THEN THEN
        CELLS-PER-FP XCURSOR +! ;
    [THEN]
[THEN]    

: EMPTY-STACK	\ ( ... -- ) empty stack; handles underflowed stack too.
    DEPTH START-DEPTH @ < IF
        DEPTH START-DEPTH @ SWAP DO 0 LOOP
    THEN
    DEPTH START-DEPTH @ > IF
        DEPTH START-DEPTH @ DO DROP LOOP
    THEN
    EMPTY-FSTACK ;

: ERROR1	\ ( C-ADDR U -- ) display an error message 
		\ followed by the line that had the error.
   TYPE SOURCE TYPE CR			\ display line corresponding to error
   EMPTY-STACK				\ throw away everything else
;

' ERROR1 ERROR-XT !

: T{		\ ( -- ) syntactic sugar.
   DEPTH START-DEPTH ! 0 XCURSOR ! F{ ;

: ->		\ ( ... -- ) record depth and contents of stack.
   DEPTH DUP ACTUAL-DEPTH !		\ record depth
   START-DEPTH @ > IF		\ if there is something on the stack
       DEPTH START-DEPTH @ - 0 DO ACTUAL-RESULTS I CELLS + ! LOOP \ save them
   THEN
   F-> ;

: }T		\ ( ... -- ) COMPARE STACK (EXPECTED) CONTENTS WITH SAVED
		\ (ACTUAL) CONTENTS.
   DEPTH ACTUAL-DEPTH @ = IF		\ if depths match
      DEPTH START-DEPTH @ > IF		\ if there is something on the stack
         DEPTH START-DEPTH @ - 0 DO	\ for each stack item
	    ACTUAL-RESULTS I CELLS + @	\ compare actual with expected
	    <> IF S" INCORRECT RESULT: " ERROR LEAVE THEN
	 LOOP
      THEN
   ELSE					\ depth mismatch
      S" WRONG NUMBER OF RESULTS: " ERROR
   THEN
   F} ;

: ...}T ( -- )
    XCURSOR @ START-DEPTH @ + ACTUAL-DEPTH @ <> IF
        S" NUMBER OF CELL RESULTS BEFORE '->' DOES NOT MATCH ...}T SPECIFICATION: " ERROR
    ELSE DEPTH START-DEPTH @ = 0= IF
        S" NUMBER OF CELL RESULTS BEFORE AND AFTER '->' DOES NOT MATCH: " ERROR
    THEN THEN
    F...}T ;

: XTESTER ( X -- )
    DEPTH 0= ACTUAL-DEPTH @ XCURSOR @ START-DEPTH @ + 1+ < OR IF
        S" NUMBER OF CELL RESULTS AFTER '->' BELOW ...}T SPECIFICATION: " ERROR EXIT
    ELSE ACTUAL-RESULTS XCURSOR @ CELLS + @ <> IF
        S" INCORRECT CELL RESULT: " ERROR
    THEN THEN
    1 XCURSOR +! ;

: X}T XTESTER ...}T ;
: XX}T XTESTER XTESTER ...}T ;
: XXX}T XTESTER XTESTER XTESTER ...}T ;
: XXXX}T XTESTER XTESTER XTESTER XTESTER ...}T ;

HAS-FLOATING [IF]
: R}T FTESTER ...}T ;
: XR}T FTESTER XTESTER ...}T ;
: RX}T XTESTER FTESTER ...}T ;
: RR}T FTESTER FTESTER ...}T ;
: XXR}T FTESTER XTESTER XTESTER ...}T ;
: XRX}T XTESTER FTESTER XTESTER ...}T ;
: XRR}T FTESTER FTESTER XTESTER ...}T ;
: RXX}T XTESTER XTESTER FTESTER ...}T ;
: RXR}T FTESTER XTESTER FTESTER ...}T ;
: RRX}T XTESTER FTESTER FTESTER ...}T ;
: RRR}T FTESTER FTESTER FTESTER ...}T ;
: XXXR}T FTESTER XTESTER XTESTER XTESTER ...}T ;
: XXRX}T XTESTER FTESTER XTESTER XTESTER ...}T ;
: XXRR}T FTESTER FTESTER XTESTER XTESTER ...}T ;
: XRXX}T XTESTER XTESTER FTESTER XTESTER ...}T ;
: XRXR}T FTESTER XTESTER FTESTER XTESTER ...}T ;
: XRRX}T XTESTER FTESTER FTESTER XTESTER ...}T ;
: XRRR}T FTESTER FTESTER FTESTER XTESTER ...}T ;
: RXXX}T XTESTER XTESTER XTESTER FTESTER ...}T ;
: RXXR}T FTESTER XTESTER XTESTER FTESTER ...}T ;
: RXRX}T XTESTER FTESTER XTESTER FTESTER ...}T ;
: RXRR}T FTESTER FTESTER XTESTER FTESTER ...}T ;
: RRXX}T XTESTER XTESTER FTESTER FTESTER ...}T ;
: RRXR}T FTESTER XTESTER FTESTER FTESTER ...}T ;
: RRRX}T XTESTER FTESTER FTESTER FTESTER ...}T ;
: RRRR}T FTESTER FTESTER FTESTER FTESTER ...}T ;
[THEN]

\ Set the following flag to TRUE for more verbose output; this may
\ allow you to tell which test caused your system to hang.
VARIABLE VERBOSE
   TRUE VERBOSE !

: TESTING	\ ( -- ) TALKING COMMENT.
   SOURCE VERBOSE @
   IF DUP >R TYPE CR R> >IN !
   ELSE >IN ! DROP
   THEN ;

BASE !
\ end of ttester.fs
