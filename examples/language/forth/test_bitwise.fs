\ test_bitwise.fs — Bitwise and shift word tests

testing and
T{ 0 0 and -> 0 }T
T{ 0 -1 and -> 0 }T
T{ -1 0 and -> 0 }T
T{ -1 -1 and -> -1 }T
T{ 1 3 and -> 1 }T
T{ 2 3 and -> 2 }T
T{ 6 3 and -> 2 }T

testing or
T{ 0 0 or -> 0 }T
T{ 0 -1 or -> -1 }T
T{ -1 0 or -> -1 }T
T{ -1 -1 or -> -1 }T
T{ 1 2 or -> 3 }T
T{ 4 3 or -> 7 }T

testing xor
T{ 0 0 xor -> 0 }T
T{ 0 -1 xor -> -1 }T
T{ -1 0 xor -> -1 }T
T{ -1 -1 xor -> 0 }T
T{ 1 3 xor -> 2 }T
T{ 5 3 xor -> 6 }T

testing invert
T{ 0 invert -> -1 }T
T{ -1 invert -> 0 }T
T{ 1 invert -> -2 }T
T{ -2 invert -> 1 }T

testing lshift
T{ 1 0 lshift -> 1 }T
T{ 1 1 lshift -> 2 }T
T{ 1 2 lshift -> 4 }T
T{ 1 8 lshift -> 256 }T
T{ 2 1 lshift -> 4 }T

testing rshift
T{ 1 0 rshift -> 1 }T
T{ 2 1 rshift -> 1 }T
T{ 4 2 rshift -> 1 }T
T{ 256 8 rshift -> 1 }T
T{ 255 4 rshift -> 15 }T

test-summary
