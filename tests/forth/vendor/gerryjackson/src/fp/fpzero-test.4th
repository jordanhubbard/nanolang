\ fpzero-test.4th
\
\ Check whether or not basic operations with floating point signed zero in a
\ Forth system are compatible with IEEE 754 arithmetic
\
\ Krishna Myneni
\
\ Revisions:
\   2009-05-05  km; created
\
\ Notes:
\
\ 1. Based on the C program zerosdq.c, from
\
\    http://www.math.utah.edu/~beebe/software/ieee/#testing-is-necessary
\
\ 2. This Forth program makes no assumptions about the internal representation
\    of floating point numbers, unlike the original C program, which assumes an 
\    IEEE format.
\ 
\ 3. Several additional tests are included in the Forth version.

\ s" ans-words" included 
\ s" ttester"   included

CR .( Running fpzero-test.4th)
CR .( -----------------------) CR

true verbose !
decimal

variable #errors    0 #errors !

: noname  ( c-addr u -- | Keep a cumulative error count )
  1 #errors +! error1 ;  ' noname error-xt !

-0E 0E 0E F~ [IF]
   
   cr cr .( ** System does not support floating point signed zero. **)
   cr    .( ** Therefore these tests have been skipped **) cr
[ELSE]

verbose @ [IF]
  cr cr .( System supports fp signed zero. )
[THEN]

SET-EXACT

t{  0E  FNEGATE       ->  -0E     }t
t{ -0E  FABS          ->   0E     }t
t{  0E  F0=           ->   TRUE   }t 
t{ -0E  F0=           ->   TRUE   }t
t{ -0E  0E F<         ->   FALSE  }t
t{  0E -0E F<         ->   FALSE  }t
t{ -0E  0E F>         ->   FALSE  }t
t{  0E -0E F>         ->   FALSE  }t
t{  0E  0E F-         ->   0E     }t
t{  0E FNEGATE 0E F-  ->  -0E     }t
t{  0E  1E F*         ->   0E     }t
t{  0E -1E F*         ->  -0E     }t

verbose @ [IF]
cr .( #ERRORS: ) #errors @ . cr
[THEN]

CR .( End of fpzero-test.4th) CR
[THEN]

