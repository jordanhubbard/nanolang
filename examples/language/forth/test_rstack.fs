\ test_rstack.fs — Return stack tests
\ Interpretation semantics of >R / R> / R@ are undefined (Forth 2012).
\ These cases compile them inside colon definitions.

testing >r r>
: rs-id  >r r> ;
T{ 1 rs-id -> 1 }T
T{ -1 rs-id -> -1 }T
: rs-nest  >r 4 r> ;
T{ 3 rs-nest -> 4 3 }T

testing r@
: rs-fetch  >r r@ r> ;
T{ 1 rs-fetch -> 1 1 }T
: rs-fetch2  >r r@ r@ r> drop ;
T{ 5 rs-fetch2 -> 5 5 }T

testing 2>r 2r>
: rs-2id  2>r 2r> ;
T{ 1 2 rs-2id -> 1 2 }T
T{ -1 0 rs-2id -> -1 0 }T

testing 2r@
: rs-2fetch  2>r 2r@ 2r> ;
T{ 1 2 rs-2fetch -> 1 2 1 2 }T

testing >r inside words
: rt-word1  >r 1 r> + ;
T{ 3 rt-word1 -> 4 }T

: rt-word2  >r >r r> r> ;
T{ 1 2 rt-word2 -> 1 2 }T

test-summary
