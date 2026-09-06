\ to-float-test.fs  
\
\ Test Forth-94 compliance for >FLOAT
\
\ by "Ed" on comp.lang.forth
\
\ Revisions:
\   2009-05-07  ed; created
\   2009-05-08  km; modified to use ttester.fs; the ttester
\                   tests have the added feature that they
\                   verify not only the flag returned by
\                   >FLOAT, but also the floating point value.
\   2010-04-25  km; additional tests to cover some cases not
\                   checked earlier.

\ s" ans-words" included

CR .( Running to-float-test.4th)
CR .( -------------------------) CR

0 [IF]  \ original code 

: CHK ( addr len flag )
  >R CR [CHAR] " EMIT 2DUP TYPE [CHAR] " EMIT
  8 OVER - SPACES  >FLOAT DUP >R IF FDROP THEN R>
  ." --> " DUP IF ." TRUE " ELSE ." FALSE" THEN
  R> - IF ."   *fail* " ELSE ."   pass " THEN ;

: TEST ( -- )
  CR ." Checking >FLOAT Forth-94 compliance ..." CR
  S" ."    FALSE CHK
  S" E"    FALSE CHK
  S" .E"   FALSE CHK
  S" .E-"  FALSE CHK
  S" +"    FALSE CHK
  S" -"    FALSE CHK
  S"  9"   FALSE CHK
  S" 9 "   FALSE CHK
  S" "     TRUE CHK
  S"    "  TRUE CHK
  S" 1+1"  TRUE CHK
  S" 1-1"  TRUE CHK
  S" 9"    TRUE CHK
  S" 9."   TRUE CHK
  S" .9"   TRUE CHK
  S" 9E"   TRUE CHK
  S" 9e+"  TRUE CHK
  S" 9d-"  TRUE CHK
;

TEST

[ELSE]

\ s" ttester" included

variable #errors    0 #errors !
: noname  ( c-addr u -- ) 1 #errors +! error1 ; ' noname error-xt !
: ?.errors  ( -- )  verbose @ IF ." #ERRORS: " #errors @ . THEN ;
: ?.cr  ( -- )  verbose @ IF cr THEN ;
true verbose !

TESTING >FLOAT
DECIMAL
SET-EXACT
t{  S" ."    >FLOAT  ->   FALSE     }t
t{  S" E"    >FLOAT  ->   FALSE     }t
t{  S" .E"   >FLOAT  ->   FALSE     }t
t{  S" .E-"  >FLOAT  ->   FALSE     }t
t{  S" +"    >FLOAT  ->   FALSE     }t
t{  S" -"    >FLOAT  ->   FALSE     }t
t{  S"  9"   >FLOAT  ->   FALSE     }t    \ Leading space
t{  S" 9 "   >FLOAT  ->   FALSE     }t    \ Trailing space
t{  S" "     >FLOAT  ->   0E TRUE   rx}t 
t{  S"    "  >FLOAT  ->   0E TRUE   rx}t
t{  S" 1+1"  >FLOAT  ->   10E TRUE  rx}t
t{  S" 1-1"  >FLOAT  ->   0.1E TRUE rx}t
t{  S" 9"    >FLOAT  ->   9E TRUE   rx}t
t{  S" 9."   >FLOAT  ->   9E TRUE   rx}t
t{  S" .9"   >FLOAT  ->   0.9E TRUE rx}t
t{  S" 9E"   >FLOAT  ->   9E TRUE   rx}t
t{  S" 9e+"  >FLOAT  ->   9E TRUE   rx}t
t{  S" 9d-"  >FLOAT  ->   9E TRUE   rx}t

\ Additional tests
t{  S" -35E2"     >FLOAT  ->  -3500E TRUE  rx}t
t{  S" -35.E2"    >FLOAT  ->  -3500E TRUE  rx}t
t{  S" -35.0E2"   >FLOAT  ->  -3500E TRUE  rx}t
t{  S" -35.0E+2"  >FLOAT  ->  -3500E TRUE  rx}t
t{  S" -35.0E+02" >FLOAT  ->  -3500E TRUE  rx}t
t{  S" 35.E+2"    >FLOAT  ->   3500E TRUE  rx}t
t{  S" +35.E+2"   >FLOAT  ->   3500E TRUE  rx}t   
t{  S" -35.+2"    >FLOAT  ->  -3500E TRUE  rx}t
t{  S" +35.+2"    >FLOAT  ->   3500E TRUE  rx}t
t{  S" -.35+4"    >FLOAT  ->  -3500E TRUE  rx}t
t{  S" +.35+4"    >FLOAT  ->   3500E TRUE  rx}t
t{  S" .35E4"     >FLOAT  ->   3500E TRUE  rx}t
t{  S" 0.35E4"    >FLOAT  ->   3500E TRUE  rx}t
t{  S" +0.35E4"   >FLOAT  ->   3500E TRUE  rx}t
t{  S" -0.35E4"   >FLOAT  ->  -3500E TRUE  rx}t
t{  S" -350000-2" >FLOAT  ->  -3500E TRUE  rx}t
t{  S" 350000E-2" >FLOAT  ->   3500E TRUE  rx}t

?.cr ?.errors ?.cr
 
[THEN]

CR .( End of to-float-test.4th) CR
