\ test_compare.fs — Comparison word tests (TRUE = -1, FALSE = 0)

testing =
T{ 0 0 = -> -1 }T
T{ 1 1 = -> -1 }T
T{ -1 -1 = -> -1 }T
T{ 0 1 = -> 0 }T
T{ 1 0 = -> 0 }T
T{ 1 -1 = -> 0 }T

testing <>
T{ 0 1 <> -> -1 }T
T{ 1 -1 <> -> -1 }T
T{ 0 0 <> -> 0 }T
T{ 1 1 <> -> 0 }T

testing <
T{ 0 1 < -> -1 }T
T{ 1 2 < -> -1 }T
T{ -1 0 < -> -1 }T
T{ 0 0 < -> 0 }T
T{ 1 0 < -> 0 }T
T{ 0 -1 < -> 0 }T

testing >
T{ 1 0 > -> -1 }T
T{ 2 1 > -> -1 }T
T{ 0 -1 > -> -1 }T
T{ 0 0 > -> 0 }T
T{ 0 1 > -> 0 }T

testing u<
T{ 1 2 u< -> -1 }T
T{ 2 1 u< -> 0 }T
T{ 0 1 u< -> -1 }T
T{ 1 1 u< -> 0 }T
T{ -1 -1 u< -> 0 }T

testing u>
T{ 2 1 u> -> -1 }T
T{ 1 2 u> -> 0 }T
T{ 0 0 u> -> 0 }T
T{ 1 0 u> -> -1 }T

testing 0=
T{ 0 0= -> -1 }T
T{ 1 0= -> 0 }T
T{ -1 0= -> 0 }T

testing 0<>
T{ 0 0<> -> 0 }T
T{ 1 0<> -> -1 }T
T{ -1 0<> -> -1 }T

testing 0<
T{ 0 0< -> 0 }T
T{ -1 0< -> -1 }T
T{ 1 0< -> 0 }T

testing 0>
T{ 0 0> -> 0 }T
T{ 1 0> -> -1 }T
T{ -1 0> -> 0 }T

testing true false
T{ true -> -1 }T
T{ false -> 0 }T

testing within
T{ 1 1 3 within -> -1 }T
T{ 2 1 3 within -> -1 }T
T{ 3 1 3 within -> 0 }T
T{ 0 1 3 within -> 0 }T
T{ 0 0 0 within -> 0 }T
T{ 0 0 1 within -> -1 }T

test-summary
