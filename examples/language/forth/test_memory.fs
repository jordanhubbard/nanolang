\ test_memory.fs — Memory, variable, constant, and cell tests

testing variable @ !
variable mv1
T{ 0 mv1 ! mv1 @ -> 0 }T
T{ 1 mv1 ! mv1 @ -> 1 }T
T{ -1 mv1 ! mv1 @ -> -1 }T
T{ 42 mv1 ! mv1 @ -> 42 }T

testing +!
variable mv2
T{ 0 mv2 ! 5 mv2 +! mv2 @ -> 5 }T
T{ 5 mv2 ! 3 mv2 +! mv2 @ -> 8 }T
T{ 0 mv2 ! -1 mv2 +! mv2 @ -> -1 }T

testing constant
5 constant mk5
-3 constant mkm3
0 constant mk0
T{ mk5 -> 5 }T
T{ mkm3 -> -3 }T
T{ mk0 -> 0 }T

testing 2! 2@
create dbl1 0 , 0 ,
T{ 0 0 dbl1 2! dbl1 2@ -> 0 0 }T
T{ 1 2 dbl1 2! dbl1 2@ -> 1 2 }T
T{ -1 -2 dbl1 2! dbl1 2@ -> -1 -2 }T

testing cells cell+
T{ 0 cells -> 0 }T
T{ 1 cells -> 8 }T
T{ 5 cells -> 40 }T
T{ 0 cell+ -> 8 }T
T{ 3 cell+ -> 11 }T

testing allot here
T{ here 1 allot here swap - -> 1 }T
T{ here 5 allot here swap - -> 5 }T

testing , (comma)
create comma_base 1 , 2 , 3 ,
T{ comma_base @ -> 1 }T
T{ comma_base cell+ @ -> 2 }T
T{ comma_base cell+ cell+ @ -> 3 }T

test-summary
