# My byte type alias source contract

I treat the bare `byte` type spelling as my existing `u8` type. I normalize that spelling within recursive array and function annotations. Qualified nominal names keep their identity. My C lexer already classifies the bare spelling as the same primitive type.

I preserve my existing `u8` literal range checks, computed modulo-256 conversion and exact array storage. This alias correction does not change general scalar compatibility or the separate `u8` argument to `int` mismatch in `std/binary`. I keep the complete original `test_u8_basic.nano` fixture and its shadows unchanged.

I require actual source VM/native acceptance and actual native checker controls, with fresh-stage and integrated platform results recorded separately.
