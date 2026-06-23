\ test_control.fs — Control flow tests

testing if then
T{ 1 if 42 then -> 42 }T
T{ 0 if 42 then -> }T
T{ -1 if 99 then -> 99 }T

testing if else then
T{ 1 if 1 else 2 then -> 1 }T
T{ 0 if 1 else 2 then -> 2 }T
T{ -1 if 1 else 2 then -> 1 }T

testing begin until
: bu-test  0 begin 1+ dup 5 = until ;
T{ bu-test -> 5 }T

: bu-count  5 begin 1- dup 0 = until ;
T{ bu-count -> 0 }T

testing begin while repeat
: bwr-test  0 begin dup 5 < while 1+ repeat ;
T{ bwr-test -> 5 }T

testing do loop
: dl-test  0 5 0 do 1+ loop ;
T{ dl-test -> 5 }T

: dl-index  5 0 do i loop ;
T{ dl-index -> 0 1 2 3 4 }T

testing do +loop
: dpl-test  10 0 do i 2 +loop ;
T{ dpl-test -> 0 2 4 6 8 }T

: dpl-neg  0 5 do i -1 +loop ;
T{ dpl-neg -> 5 4 3 2 1 0 }T

testing ?do
: qdl-zero  5 5 ?do i loop ;
T{ qdl-zero -> }T

: qdl-one  6 5 ?do i loop ;
T{ qdl-one -> 5 }T

: qdl-count  3 0 ?do i loop ;
T{ qdl-count -> 0 1 2 }T

testing leave
: leave-test  10 0 do i dup 3 = if drop leave then loop ;
T{ leave-test -> 0 1 2 }T

testing exit
: exit-test  dup 0 < if drop -1 exit then 1 ;
T{ 1 exit-test -> 1 1 }T
T{ -1 exit-test -> -1 }T

testing recurse (factorial)
: fact  dup 1 > if dup 1 - fact * then ;
T{ 0 fact -> 0 }T
T{ 1 fact -> 1 }T
T{ 5 fact -> 120 }T
T{ 6 fact -> 720 }T

testing nested loops with j
: nl-test  3 0 do 3 0 do j i * loop loop ;
T{ nl-test -> 0 0 0 0 1 2 0 2 4 }T

test-summary
