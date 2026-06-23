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
here 2 allot constant dbl1
T{ 0 0 dbl1 2! dbl1 2@ -> 0 0 }T
T{ 1 2 dbl1 2! dbl1 2@ -> 1 2 }T
T{ -1 -2 dbl1 2! dbl1 2@ -> -1 -2 }T

testing cells cell+
T{ 0 cells -> 0 }T
T{ 1 cells -> 1 }T
T{ 5 cells -> 5 }T
T{ 0 cell+ -> 1 }T
T{ 3 cell+ -> 4 }T

testing allot here
here constant ha1
1 allot
here constant ha2
T{ ha2 ha1 - -> 1 }T
5 allot
here constant ha3
T{ ha3 ha2 - -> 5 }T

testing , (comma)
here constant comma_base
1 ,
2 ,
3 ,
T{ comma_base @ -> 1 }T
T{ comma_base cell+ @ -> 2 }T
T{ comma_base cell+ cell+ @ -> 3 }T

test-summary
