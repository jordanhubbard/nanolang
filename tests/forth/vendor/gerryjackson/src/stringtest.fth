\ To test the ANS Forth String word set

\ This program was written by Gerry Jackson in 2006, with contributions from
\ others where indicated, and is in the public domain - it can be distributed
\ and/or modified in any way but please retain this notice.

\ This program is distributed in the hope that it will be useful,
\ but WITHOUT ANY WARRANTY; without even the implied warranty of
\ MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

\ The tests are not claimed to be comprehensive or correct 

\ ------------------------------------------------------------------------------
\ Version 0.13 13 Nov 2015 Interpretive use of S" replaced by $" from
\                          utilities.fth
\         0.11 25 April 2015 Tests for REPLACES SUBSTITUTE UNESCAPE added
\         0.6 1 April 2012 Tests placed in the public domain.
\         0.5 29 April 2010 Added tests for SEARCH and COMPARE with
\             all strings zero length (suggested by Krishna Myneni).
\             SLITERAL test amended in line with comp.lang.forth
\             discussion
\         0.4 30 November 2009 <TRUE> and <FALSE> replaced with TRUE
\             and FALSE
\         0.3 6 March 2009 { and } replaced with T{ and }T
\         0.2 20 April 2007 ANS Forth words changed to upper case
\         0.1 Oct 2006 First version released

\ ------------------------------------------------------------------------------
\ The tests are based on John Hayes test program for the core word set
\ and requires those files to have been loaded

\ Words tested in this file are:
\     -TRAILING /STRING BLANK CMOVE CMOVE> COMPARE SEARCH SLITERAL
\     REPLACES SUBSTITUTE UNESCAPE
\
\ ------------------------------------------------------------------------------
\ Assumptions, dependencies and notes:
\     - tester.fr (or ttester.fs), errorreport.fth and utilities.fth have been
\       included prior to this file
\     - the Core word set is available and tested
\     - COMPARE is case sensitive
\ ------------------------------------------------------------------------------

TESTING String word set

DECIMAL

T{ :  S1 S" abcdefghijklmnopqrstuvwxyz" ; -> }T
T{ :  S2 S" abc"   ; -> }T
T{ :  S3 S" jklmn" ; -> }T
T{ :  S4 S" z"     ; -> }T
T{ :  S5 S" mnoq"  ; -> }T
T{ :  S6 S" 12345" ; -> }T
T{ :  S7 S" "      ; -> }T
T{ :  S8 S" abc  " ; -> }T
T{ :  S9 S"      " ; -> }T
T{ : S10 S"    a " ; -> }T

\ ------------------------------------------------------------------------------
TESTING -TRAILING

T{  S1 -TRAILING -> S1 }T
T{  S8 -TRAILING -> S8 2 - }T
T{  S7 -TRAILING -> S7 }T
T{  S9 -TRAILING -> S9 DROP 0 }T
T{ S10 -TRAILING -> S10 1- }T

\ ------------------------------------------------------------------------------
TESTING /STRING

T{ S1  5 /STRING -> S1 SWAP 5 + SWAP 5 - }T
T{ S1 10 /STRING -4 /STRING -> S1 6 /STRING }T
T{ S1  0 /STRING -> S1 }T

\ ------------------------------------------------------------------------------
TESTING SEARCH

T{ S1 S2 SEARCH -> S1 TRUE }T
T{ S1 S3 SEARCH -> S1  9 /STRING TRUE }T
T{ S1 S4 SEARCH -> S1 25 /STRING TRUE }T
T{ S1 S5 SEARCH -> S1 FALSE }T
T{ S1 S6 SEARCH -> S1 FALSE }T
T{ S1 S7 SEARCH -> S1 TRUE }T
T{ S7 PAD 0 SEARCH -> S7 TRUE }T

\ ------------------------------------------------------------------------------
TESTING COMPARE

T{ S1 S1 COMPARE -> 0 }T
T{ S1 PAD SWAP CMOVE -> }T
T{ S1 PAD OVER COMPARE -> 0 }T
T{ S1 PAD 6 COMPARE -> 1 }T
T{ PAD 10 S1 COMPARE -> -1 }T
T{ S1 PAD 0 COMPARE -> 1 }T
T{ PAD 0 S1 COMPARE -> -1 }T
T{ S1 S6 COMPARE ->  1 }T
T{ S6 S1 COMPARE -> -1 }T
T{ S7 PAD 0 COMPARE -> 0 }T

T{ S1 $" abdde"  COMPARE -> -1 }T
T{ S1 $" abbde"  COMPARE ->  1 }T
T{ S1 $" abcdf"  COMPARE -> -1 }T
T{ S1 $" abcdee" COMPARE ->  1 }T

: S11 S" 0abc" ;
: S12 S" 0aBc" ;

T{ S11 S12  COMPARE -> 1 }T
T{ S12 S11  COMPARE -> -1 }T

\ ------------------------------------------------------------------------------
TESTING CMOVE and CMOVE>

PAD 30 CHARS 0 FILL
T{ S1 PAD SWAP CMOVE -> }T
T{ S1 PAD S1 SWAP DROP COMPARE -> 0 }T
T{ S6 PAD 10 CHARS + SWAP CMOVE -> }T
T{ $" abcdefghij12345pqrstuvwxyz" PAD S1 SWAP DROP COMPARE -> 0 }T
T{ PAD 15 CHARS + PAD CHAR+ 6 CMOVE -> }T
T{ $" apqrstuhij12345pqrstuvwxyz" PAD 26 COMPARE -> 0 }T
T{ PAD PAD 3 CHARS + 7 CMOVE -> }T
T{ $" apqapqapqa12345pqrstuvwxyz" PAD 26 COMPARE -> 0 }T
T{ PAD PAD CHAR+ 10 CMOVE -> }T
T{ $" aaaaaaaaaaa2345pqrstuvwxyz" PAD 26 COMPARE -> 0 }T
T{ S7 PAD 14 CHARS + SWAP CMOVE -> }T
T{ $" aaaaaaaaaaa2345pqrstuvwxyz" PAD 26 COMPARE -> 0 }T

PAD 30 CHARS 0 FILL

T{ S1 PAD SWAP CMOVE> -> }T
T{ S1 PAD S1 SWAP DROP COMPARE -> 0 }T
T{ S6 PAD 10 CHARS + SWAP CMOVE> -> }T
T{ $" abcdefghij12345pqrstuvwxyz" PAD S1 SWAP DROP COMPARE -> 0 }T
T{ PAD 15 CHARS + PAD CHAR+ 6 CMOVE> -> }T
T{ $" apqrstuhij12345pqrstuvwxyz" PAD 26 COMPARE -> 0 }T
T{ PAD 13 CHARS + PAD 10 CHARS + 7 CMOVE> -> }T
T{ $" apqrstuhijtrstrstrstuvwxyz" PAD 26 COMPARE -> 0 }T
T{ PAD 12 CHARS + PAD 11 CHARS + 10 CMOVE> -> }T
T{ $" apqrstuhijtvvvvvvvvvvvwxyz" PAD 26 COMPARE -> 0 }T
T{ S7 PAD 14 CHARS + SWAP CMOVE> -> }T
T{ $" apqrstuhijtvvvvvvvvvvvwxyz" PAD 26 COMPARE -> 0 }T

\ ------------------------------------------------------------------------------
TESTING BLANK

: S13 S" aaaaa      a" ;   \ Don't move this down as it might corrupt PAD

T{ PAD 25 CHAR a FILL -> }T
T{ PAD 5 CHARS + 6 BLANK -> }T
T{ PAD 12 S13 COMPARE -> 0 }T

\ ------------------------------------------------------------------------------
TESTING SLITERAL

T{ HERE DUP S1 DUP ALLOT ROT SWAP CMOVE S1 SWAP DROP 2CONSTANT S1A -> }T
T{ : S14 [ S1A ] SLITERAL ; -> }T
T{ S1A S14 COMPARE -> 0 }T
T{ S1A DROP S14 DROP = -> FALSE }T

\ ------------------------------------------------------------------------------
TESTING UNESCAPE

CREATE SUBBUF 48 CHARS ALLOT

\ $CHECK AND $CHECKN return f = 0 if caddr1 = SUBBUF and string1 = string2
: $CHECK   ( caddr1 u1 caddr2 u2 -- f )  2SWAP OVER SUBBUF <> >R COMPARE R> or ;
: $CHECKN  ( caddr1 u1 n caddr2 u2 -- f n )  ROT >R $CHECK R> ;

T{ 123 SUBBUF C! $" " SUBBUF UNESCAPE SUBBUF 0 $CHECK -> FALSE }T
T{ SUBBUF C@ -> 123 }T
T{ $" unchanged" SUBBUF UNESCAPE $" unchanged" $CHECK -> FALSE }T
T{ $" %" SUBBUF UNESCAPE $" %%" $CHECK -> FALSE }T
T{ $" %%%" SUBBUF UNESCAPE $" %%%%%%" $CHECK -> FALSE }T
T{ $" abc%def" SUBBUF UNESCAPE $" abc%%def" $CHECK -> FALSE }T
T{ : TEST-UNESCAPE S" %abc%def%%ghi%" SUBBUF UNESCAPE ; -> }T \ Compile check
T{ TEST-UNESCAPE $" %%abc%%def%%%%ghi%%" $CHECK -> FALSE }T

TESTING SUBSTITUTE REPLACES

T{ $" abcdef" SUBBUF 20 SUBSTITUTE $" abcdef" $CHECKN -> FALSE 0 }T \ Unchanged
T{ $" " SUBBUF 20 SUBSTITUTE $" " $CHECKN -> FALSE 0 }T    \ Zero length string
T{ $" %%" SUBBUF 20 SUBSTITUTE $" %" $CHECKN -> FALSE 0 }T        \ %% --> %
T{ $" %%%%%%" SUBBUF 25 SUBSTITUTE $" %%%" $CHECKN -> FALSE 0 }T
T{ $" %%%%%%%" SUBBUF 25 SUBSTITUTE $" %%%%" $CHECKN -> FALSE 0 }T \ Odd no. %'s

: MAC1 S" mac1" ;  : MAC2 S" mac2" ;  : MAC3 S" mac3" ;

T{ $" wxyz" MAC1 REPLACES -> }T
T{ $" %mac1%" SUBBUF 20 SUBSTITUTE $" wxyz" $CHECKN -> FALSE 1 }T
T{ $" abc%mac1%d" SUBBUF 20 SUBSTITUTE $" abcwxyzd" $CHECKN -> FALSE 1 }T
T{ : SUBST SUBBUF 20 SUBSTITUTE ; -> }T   \ Check it compiles
T{ $" defg%mac1%hi" SUBST $" defgwxyzhi" $CHECKN -> FALSE 1 }T
T{ $" 12" MAC2 REPLACES -> }T
T{ $" %mac1%mac2" SUBBUF 20 SUBSTITUTE $" wxyzmac2" $CHECKN -> FALSE 1 }T
T{ $" abc %mac2% def%mac1%gh" SUBBUF 20 SUBSTITUTE $" abc 12 defwxyzgh" $CHECKN
      -> FALSE 2 }T
T{ : REPL  ( caddr1 u1 "name" -- )  PARSE-NAME REPLACES ; -> }T
T{ $" " REPL MAC3 -> }T    \ Check compiled version
T{ $" abc%mac3%def%mac1%gh" SUBBUF 20 SUBSTITUTE $" abcdefwxyzgh" $CHECKN
      -> FALSE 2 }T      \ Zero length string substituted
T{ $" %mac3%" SUBBUF 10 SUBSTITUTE $" " $CHECKN
      -> FALSE 1 }T      \ Zero length string substituted
T{ $" abc%%mac1%%%mac2%" SUBBUF 20 SUBSTITUTE $" abc%mac1%12" $CHECKN
      -> FALSE 1 }T   \ Check substitution is single pass
T{ $" %mac3%" MAC3 REPLACES -> }T
T{ $" a%mac3%b" SUBBUF 20 SUBSTITUTE $" a%mac3%b" $CHECKN
      -> FALSE 1 }T    \ Check non-recursive
T{ $" %%" MAC3 REPLACES -> }T
T{ $" abc%mac1%de%mac3%g%mac2%%%%mac1%hij" SUBBUF 30 SUBSTITUTE
      $" abcwxyzde%%g12%wxyzhij" $CHECKN -> FALSE 4 }T
T{ $" ab%mac4%c" SUBBUF 20 SUBSTITUTE $" ab%mac4%c" $CHECKN
      -> FALSE 0 }T   \ Non-substitution name passed unchanged
T{ $" %mac2%%mac5%" SUBBUF 20 SUBSTITUTE $" 12%mac5%" $CHECKN
      -> FALSE 1 }T   \ Non-substitution name passed unchanged
T{ $" %mac5%" SUBBUF 20 SUBSTITUTE $" %mac5%" $CHECKN
      -> FALSE 0 }T   \ Non-substitution name passed unchanged

\ Check UNESCAPE SUBSTITUTE leaves a string unchanged
T{ $" %mac1%" SUBBUF 30 CHARS + UNESCAPE SUBBUF 10 SUBSTITUTE $" %mac1%" $CHECKN
   -> FALSE 0 }T

\ Check with odd numbers of % characters, last is passed unchanged
T{ $" %" SUBBUF 10 SUBSTITUTE $" %" $CHECKN -> FALSE 0 }T
T{ $" %abc" SUBBUF 10 SUBSTITUTE $" %abc" $CHECKN -> FALSE 0 }T
T{ $" abc%" SUBBUF 10 SUBSTITUTE $" abc%" $CHECKN -> FALSE 0 }T
T{ $" abc%mac1" SUBBUF 10 SUBSTITUTE $" abc%mac1" $CHECKN -> FALSE 0 }T
T{ $" abc%mac1%d%%e%mac2%%mac3" SUBBUF 20 SUBSTITUTE
      $" abcwxyzd%e12%mac3" $CHECKN -> FALSE 2 }T

\ Check for errors
T{ $" abcd" SUBBUF 4 SUBSTITUTE $" abcd" $CHECKN -> FALSE 0 }T  \ Just fits
T{ $" abcd" SUBBUF 3 SUBSTITUTE ROT ROT 2DROP 0< -> TRUE }T     \ Just too long
T{ $" abcd" SUBBUF 0 SUBSTITUTE ROT ROT 2DROP 0< -> TRUE }T
T{ $" zyxwvutsr" MAC3 REPLACES -> }T
T{ $" abc%mac3%d" SUBBUF 10 SUBSTITUTE ROT ROT 2DROP 0< -> TRUE }T

\ Conditional test for overlapping strings, including the case where
\ caddr1 = caddr2. If a system cannot handle overlapping strings it should
\ return n < 0 with (caddr2 u2) undefined. If it can handle them correctly
\ it should return the usual results for success. The following definitions
\ apply the appropriate tests depending on whether n < 0 or not.
\ The overlapping SUBSTITUTE tests:
\     succeed if SUBSTITUTE returns an error i.e. n<0
\     fail if n is incorrect 
\     fail if the result string is at the incorrect addresses
\     fail if the result string is incorrect
\ Note that variables are used to avoid complicated stack manipulations

VARIABLE sdest       \ Holds dest address for SUBSTITUTE
20 constant ssize
2VARIABLE $sresult   VARIABLE #subst   \ Hold output from SUBSTITUTE

\ sinit set ups addresses and inputs for SUBSTITUTE and saves the
\ output destination for check-subst
\     srcn and destn are offsets into the substitution buffer subbuf
\     (caddr1 u1) is the input string for SUBSTITUTE

: sinit  ( caddr1 u1 srcn destn -- src u1 dest size )
   CHARS subbuf + sdest !        ( -- caddr1 u1 srcn )
   CHARS subbuf + 2DUP 2>R       ( -- caddr1 u1 src ) ( R: -- u1 src )
   SWAP CHARS MOVE               ( -- )
   R> R> sdest @ ssize           ( -- src u1 dest size) ( R: -- )
;

\ In check-subst
\     (caddr1 u1) is the expected result from SUBSTITUTE
\     n is the expected n if SUBSTITUTE succeeded with overlapping buffers

: check-subst  ( caddr1 u1 n -- f )
   #subst @ 0<
   IF DROP 2DROP TRUE EXIT THEN  \ SUBSTITUTE failed, test succeeds
   #subst @ = >R
   $sresult CELL+ @ sdest @ = R> AND
   IF $sresult 2@ COMPARE 0= EXIT THEN
   2DROP FALSE                   \ Fails if #subst or result address is wrong
;

\ Testing the helpers sinit and check-subst

T{ $" abcde" 2 6 sinit -> subbuf 2 chars + 5 subbuf 6 chars + ssize }T
T{ $" abcde" subbuf 2 chars + over compare -> 0 }T
T{ sdest @ -> subbuf 6 chars + }T

T{ -78 #subst ! 0 0 0 check-subst -> TRUE }T
T{ 5 #subst ! $" def" over sdest ! 2dup $sresult 2! 5 check-subst -> TRUE }T
T{ 5 #subst ! $" def" over sdest ! 2dup $sresult 2! 4 check-subst -> FALSE }T
T{ 5 #subst ! $" def" over sdest ! 2dup 1+ $sresult 2! 5 check-subst -> FALSE }T
T{ 5 #subst ! $" def" over sdest ! 2dup 1- $sresult 2! 3 check-subst -> FALSE }T

\ Testing overlapping SUBSTITUTE

: do-subst  ( caddr1 u1 n1 n2 -- )
   sinit SUBSTITUTE #subst ! $sresult 2!
;

T{ $" zyxwvut" MAC3 REPLACES -> }T
T{ $" zyx"     MAC2 REPLACES -> }T

T{ $" a%mac3%b" 0 9 do-subst $" azyxwvutb" 1 check-subst -> TRUE }T
T{ $" c%mac3%d" 0 3 do-subst $" czyxwvutd" 1 check-subst -> TRUE }T
T{ $" e%mac2%f" 0 3 do-subst $" ezyxf"     1 check-subst -> TRUE }T
T{ $" abcdefgh" 0 0 do-subst $" abcdefgh"  0 check-subst -> TRUE }T
T{ $" i%mac3%j" 3 0 do-subst $" izyxwvutj" 1 check-subst -> TRUE }T
T{ $" k%mac3%l" 9 0 do-subst $" kzyxwvutl" 1 check-subst -> TRUE }T

\ Simulating a failing overlapping SUBSTITUTE

T{ $" pqrst" 2dup 0 0 do-subst -78 #subst ! 0 check-subst -> TRUE }T

\ Using SUBSTITUTE to define a name whose (caddr u) is on the stack
: $CREATE  ( caddr u -- )
   S" name" REPLACES          ( -- )
   S" CREATE %name%" SUBBUF 40 SUBSTITUTE
   0 > IF EVALUATE THEN
;
t{ $" SUBST2" $CREATE 123 , -> }t
t{ SUBST2 @ -> 123 }t

\ ------------------------------------------------------------------------------

STRING-ERRORS SET-ERROR-COUNT

CR .( End of String word tests) CR
