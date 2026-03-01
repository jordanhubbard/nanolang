\ test_words.fs — User-defined word tests

testing basic word definition
: w-double  2 * ;
T{ 3 w-double -> 6 }T
T{ -5 w-double -> -10 }T

: w-square  dup * ;
T{ 4 w-square -> 16 }T
T{ 0 w-square -> 0 }T

: w-cube  dup w-square * ;
T{ 3 w-cube -> 27 }T

testing word calling word
: w-sum-sq  w-square swap w-square + ;
T{ 3 4 w-sum-sq -> 25 }T

testing constants in words
10 constant ten
: w-times-ten  ten * ;
T{ 5 w-times-ten -> 50 }T
T{ -3 w-times-ten -> -30 }T

testing variables in words
variable wv1
: w-store-add  wv1 ! wv1 @ + ;
T{ 10 5 w-store-add -> 15 }T
T{ 0 3 w-store-add -> 3 }T

testing word redefinition (latest wins)
: w-val  1 ;
T{ w-val -> 1 }T
: w-val  2 ;
T{ w-val -> 2 }T

testing mutual recursion via exit
: w-even  dup 0 = if drop -1 exit then 1 - w-odd ;
: w-odd   dup 0 = if drop 0 exit then 1 - w-even ;
T{ 0 w-even -> -1 }T
T{ 1 w-even -> 0 }T
T{ 4 w-even -> -1 }T
T{ 3 w-even -> 0 }T

testing hex and decimal in words
: w-base-test  hex ff decimal ;
T{ w-base-test 1 + -> 256 }T

testing begin/until in words
: w-loop  0 begin 1+ dup 5 = until ;
T{ w-loop -> 5 }T

test-summary
