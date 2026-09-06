CR CR SOURCE TYPE ( Preliminary test ) CR
SOURCE ( These lines test SOURCE, TYPE, CR and parenthetic comments ) TYPE CR
( The next line of output should be blank to test CR ) SOURCE TYPE CR CR

( It is now assumed that SOURCE, TYPE, CR and comments work. SOURCE and      )
( TYPE will be used to report test passes until something better can be      )
( defined to report errors. Until then reporting failures will depend on the )
( system under test and will usually be via reporting an unrecognised word   )
( or possibly the system crashing. Tests will be numbered by #n from now on  )
( to assist fault finding. Test successes will be indicated by               )
( 'Pass: #n ...' and failures by 'Error: #n ...'                             )

( Initial tests of >IN +! and 1+ )
( Check that n >IN +! acts as an interpretive IF, where n >= 0 )
( Pass #1: testing 0 >IN +! ) 0 >IN +! SOURCE TYPE CR
( Pass #2: testing 1 >IN +! ) 1 >IN +! xSOURCE TYPE CR
( Pass #3: testing 1+ ) 1 1+ >IN +! xxSOURCE TYPE CR

( Test results can now be reported using the >IN +! trick to skip )
( 1 or more characters )

( The value of BASE is unknown so it is not safe to use digits > 1, therefore )
( it will be set it to binary and then decimal, this also tests @ and ! )

( Pass #4: testing @ ! BASE ) 0 1+ 1+ BASE ! BASE @ >IN +! xxSOURCE TYPE CR
( Set BASE to decimal ) 1010 BASE !
( Pass #5: testing decimal BASE ) BASE @ >IN +! xxxxxxxxxxSOURCE TYPE CR

( Now in decimal mode and digits >1 can be used )

( A better error reporting word is needed, much like .( which can't  )
( be used as it is in the Core Extension word set, similarly PARSE can't be )
( used either, only WORD is available to parse a message and must be used   )
( in a colon definition. Therefore a simple colon definition is tested next )

( Pass #6: testing : ; ) : .SRC SOURCE TYPE CR ; 6 >IN +! xxxxxx.SRC
( Pass #7: testing number input ) 19 >IN +! xxxxxxxxxxxxxxxxxxx.SRC

( VARIABLE is now tested as one will be used instead of DROP e.g. Y ! )

( Pass #8: testing VARIABLE ) VARIABLE Y 2 Y ! Y @ >IN +! xx.SRC

: MSG 41 WORD COUNT ;  ( 41 is the ASCII code for right parenthesis )
( The next tests MSG leaves 2 items on the data stack )
( Pass #9: testing WORD COUNT ) 5 MSG abcdef) Y ! Y ! >IN +! xxxxx.SRC
( Pass #10: testing WORD COUNT ) MSG ab) >IN +! xxY ! .SRC

( For reporting success .MSG( is now defined )
: .MSG( MSG TYPE ; .MSG( Pass #11: testing WORD COUNT .MSG) CR

( To define an error reporting word, = 2* AND will be needed, test them first )
( This assumes 2's complement arithmetic )
1 1 = 1+ 1+ >IN +! x.MSG( Pass #12: testing = returns all 1's for true) CR
1 0 = 1+ >IN +! x.MSG( Pass #13: testing = returns 0 for false) CR
1 1 = -1 = 1+ 1+ >IN +! x.MSG( Pass #14: testing -1 interpreted correctly) CR

1 2* >IN +! xx.MSG( Pass #15: testing 2*) CR
-1 2* 1+ 1+ 1+ >IN +! x.MSG( Pass #16: testing 2*) CR

-1 -1 AND 1+ 1+ >IN +! x.MSG( Pass #17: testing AND) CR
-1  0 AND 1+ >IN +! x.MSG( Pass #18: testing AND) CR
6  -1 AND >IN +! xxxxxx.MSG( Pass #19: testing AND) CR

( Define ~ to use as a 'to end of line' comment. \ cannot be used as it a )
( Core Extension word )
: ~  ( -- )  SOURCE >IN ! Y ! ;

( Rather than relying on a pass message test words can now be defined to )
( report errors in the event of a failure. For convenience words ?T~ and )
( ?F~ are defined together with a helper ?~~ to test for TRUE and FALSE  )
( Usage is: <test> ?T~ Error #n: <message>                               )
( Success makes >IN index the ~ in ?T~ or ?F~ to skip the error message. )
( Hence it is essential there is only 1 space between ?T~ and Error      )

: ?~~  ( -1 | 0 -- )  2* >IN +! ;
: ?F~ ( f -- )   0 = ?~~ ;
: ?T~ ( f -- )  -1 = ?~~ ;

( Errors will be counted )
VARIABLE #ERRS 0 #ERRS !
: Error  1 #ERRS +! -6 >IN +! .MSG( CR ;
: Pass  -1 #ERRS +! 1 >IN +! Error ;  ~ Pass is defined solely to test Error

-1 ?F~ Pass #20: testing ?F~ ?~~ Pass Error
-1 ?T~ Error #1: testing ?T~ ?~~ ~

0  0 = 0= ?F~ Error #2: testing 0=
1  0 = 0= ?T~ Error #3: testing 0=
-1 0 = 0= ?T~ Error #4: testing 0=

0  0 = ?T~ Error #5: testing =
0  1 = ?F~ Error #6: testing =
1  0 = ?F~ Error #7: testing =
-1 1 = ?F~ Error #8: testing =
1 -1 = ?F~ Error #9: testing =

-1 0< ?T~ Error #10: testing 0<
0  0< ?F~ Error #11: testing 0<
1  0< ?F~ Error #12: testing 0<

 DEPTH 1+ DEPTH = ?~~ Error #13: testing DEPTH
 ~ Up to now whether the data stack was empty or not hasn't mattered as
 ~ long as it didn't overflow. Now it will be emptied - also
 ~ removing any unreported underflow
 DEPTH 0< 0= 1+ >IN +! ~ 0 0 >IN ! Remove any underflow
 DEPTH 0= 1+ >IN +! ~ Y !  0 >IN ! Empty the stack
 DEPTH 0= ?T~ Error #14: data stack not emptied 

 4 -5 SWAP 4 = SWAP -5 = = ?T~ Error #15: testing SWAP
 111 222 333 444
 DEPTH 4 = ?T~ Error #16: testing DEPTH
 444 = SWAP 333 = = DEPTH 3 = = ?T~ Error #17: testing SWAP DEPTH
 222 = SWAP 111 = = DEPTH 1 = = ?T~ Error #18: testing SWAP DEPTH
 DEPTH 0= ?T~ Error #19: testing DEPTH = 0

~ From now on the stack is expected to be empty after a test so
~ ?~ will be defined to include a check on the stack depth. Note
~ that ?~~ was defined and used earlier instead of ?~ to avoid
~ (irritating) redefinition messages that many systems display had
~ ?~ simply been redefined

: ?~  ( -1 | 0 -- )  DEPTH 1 = AND ?~~ ; ~ -1 test success, 0 test failure

123 -1 ?~ Pass #21: testing ?~
Y !   ~ equivalent to DROP

~ Testing the remaining Core words used in the Hayes tester, with the above
~ definitions these are straightforward

1 DROP DEPTH 0= ?~ Error #20: testing DROP
123 DUP  = ?~ Error #21: testing DUP
123 ?DUP = ?~ Error #22: testing ?DUP
0  ?DUP 0= ?~ Error #23: testing ?DUP
123  111  + 234  = ?~ Error #24: testing +
123  -111 + 12   = ?~ Error #25: testing +
-123 111  + -12  = ?~ Error #26: testing +
-123 -111 + -234 = ?~ Error #27: testing +
-1 NEGATE 1 = ?~ Error #28: testing NEGATE
0  NEGATE 0=  ?~ Error #29: testing NEGATE
987 NEGATE -987 = ?~ Error #30: testing NEGATE
HERE DEPTH SWAP DROP 1 = ?~ Error #31: testing HERE
CREATE TST1 HERE TST1 = ?~ Error #32: testing CREATE HERE
16  ALLOT HERE TST1 NEGATE + 16 = ?~ Error #33: testing ALLOT
-16 ALLOT HERE TST1 = ?~ Error #34: testing ALLOT
0 CELLS 0= ?~ Error #35: testing CELLS
1 CELLS ALLOT HERE TST1 NEGATE + VARIABLE CSZ CSZ !
CSZ @ 0= 0= ?~ Error #36: testing CELLS
3 CELLS CSZ @ DUP 2* + = ?~ Error #37: testing CELLS
-3 CELLS CSZ @ DUP 2* + + 0= ?~ Error #38: testing CELLS
: TST2  ( f -- n )  DUP IF 1+ THEN ;
0 TST2 0=  ?~ Error #39: testing IF THEN
1 TST2 2 = ?~ Error #40: testing IF THEN
: TST3  ( n1 -- n2 )  IF 123 ELSE 234 THEN ;
0 TST3 234 = ?~ Error #41: testing IF ELSE THEN
1 TST3 123 = ?~ Error #42: testing IF ELSE THEN
: TST4  ( -- n )  0 5 0 DO 1+ LOOP ;
TST4 5 = ?~ Error #43: testing DO LOOP
: TST5  ( -- n )  0 10 0 DO I + LOOP ;
TST5 45 = ?~ Error #44: testing I
: TST6  ( -- n )  0 10 0 DO DUP 5 = IF LEAVE ELSE 1+ THEN LOOP ;
TST6 5 = ?~ Error #45: testing LEAVE
: TST7  ( -- n1 n2 ) 123 >R 234 R> ;
TST7 NEGATE + 111 = ?~ Error #46: testing >R R>
: TST8  ( -- ch )  [CHAR] A ;
TST8 65 = ?~ Error #47: testing [CHAR]
: TST9  ( -- )  [CHAR] s [CHAR] s [CHAR] a [CHAR] P 4 0 DO EMIT LOOP ;
TST9 .MSG(  #22: testing EMIT) CR
: TST10  ( -- )  S" Pass #23: testing S" TYPE [CHAR] " EMIT CR ; TST10

~ The Hayes core test core.fr uses CONSTANT before it is tested therefore
~ we test CONSTANT here

1234 CONSTANT CTEST
CTEST 1234 = ?~ Error #48: testing CONSTANT

~ The Hayes tester uses some words from the Core extension word set
~ These will be conditionally defined following definition of a
~ word called ?DEFINED to determine whether these are already defined

VARIABLE TIMM1 0 TIMM1 !
: TIMM2  123 TIMM1 ! ; IMMEDIATE
: TIMM3 TIMM2 ; TIMM1 @ 123 = ?~ Error #49: testing IMMEDIATE

: ?DEFINED  ( "name" -- 0 | -1 )  32 WORD FIND SWAP DROP 0= 0= ;
?DEFINED SWAP ?~ Error #50: testing FIND ?DEFINED
?DEFINED <<no-such-word-hopefully>> 0= ?~ Error #51 testing FIND ?DEFINED

?DEFINED \ ?~ : \ ~ ; IMMEDIATE 
\ Error #52: testing \
: TIMM4  \ Error #53: testing \ is IMMEDIATE
;

~ TRUE and FALSE are defined as colon definitions as they have been used
~ more than CONSTANT above

?DEFINED TRUE  ?~ : TRUE 1 NEGATE ;
?DEFINED FALSE ?~ : FALSE 0 ;
?DEFINED HEX   ?~ : HEX 16 BASE ! ;

TRUE -1 = ?~ Error #54: testing TRUE
FALSE 0=  ?~ Error #55: testing FALSE
10 HEX 0A = ?~ Error #56: testing HEX
AB 0A BASE ! 171 = ?~ Error #57: testing hex number

~ Delete the ~ on the next 2 lines to check the final error report
~ Error #998: testing a deliberate failure
~ Error #999: testing a deliberate failure

~ Describe the messages that should be seen. The previously defined .MSG(
~ can be used for text messages

CR .MSG( Results: ) CR
CR .MSG( Pass messages #1 to #23 should be displayed above)
CR .MSG( and no error messages) CR

~ Finally display a message giving the number of tests that failed.
~ This is complicated by the fact that untested words including .( ." and .
~ cannot be used. Also more colon definitions shouldn't be defined than are
~ needed. To display a number, note that the number of errors will have
~ one or two digits at most and an interpretive loop can be used to
~ display those.

CR
0 #ERRS @
~ Loop to calculate the 10's digit (if any)
DUP NEGATE 9 + 0< NEGATE >IN +! ( -10 + SWAP 1+ SWAP 0 >IN ! )
~ Display the error count
SWAP ?DUP 0= 1+ >IN +! ( 48 + EMIT ( ) 48 + EMIT

.MSG(  test) #ERRS @ 1 = 1+ >IN +! ~ .MSG( s)
.MSG(  failed out of 57 additional tests) CR

CR CR .MSG( --- End of Preliminary Tests --- ) CR
