\ test_rstack.fs — Return stack tests
\ Note: >r/r>/r@ are safe at top level in this interpreter.

testing >r r>
T{ 1 >r r> -> 1 }T
T{ -1 >r r> -> -1 }T
T{ 3 >r 4 r> -> 4 3 }T

testing r@
T{ 1 >r r@ r> -> 1 1 }T
T{ 5 >r r@ r@ r> drop -> 5 5 }T

testing 2>r 2r>
T{ 1 2 2>r 2r> -> 1 2 }T
T{ -1 0 2>r 2r> -> -1 0 }T

testing 2r@
T{ 1 2 2>r 2r@ 2r> -> 1 2 1 2 }T

testing >r inside words
: rt-word1  >r 1 r> + ;
T{ 3 rt-word1 -> 4 }T

: rt-word2  >r >r r> r> ;
T{ 1 2 rt-word2 -> 1 2 }T

test-summary
