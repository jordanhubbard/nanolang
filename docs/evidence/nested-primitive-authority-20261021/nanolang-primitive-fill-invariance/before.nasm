.string "floating_rows"
.string "integer_rows"
.string "float_box"
.string "int_box"
.string "main"
.string "__init__"
.string "nano.purity.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00"
.string "nano.affine.v1"
.string "nano.effects.v1"
.string "\x01\x00\x00\x00\x00\x00\x00\x00"
.string "\x02\x00\x00\x00\x00\x00\x00\x00"
.string "\x03\x00\x00\x00\x00\x00\x00\x00"
.string "\x04\x00\x00\x00\x00\x00\x00\x00"
.string "\x04\x00\x00\x00\x02\x00\x00\x00"
.string "tests/nanoisa/fixtures/nested_array_contexts.nano"
.string "nano.local.v1"
.string "\x04\x00\x00\x00\x0c\x00\x00\x00\n\x02\x00\x00^\x02\x00\x00integer"
.string "\x04\x00\x00\x00\x0b\x00\x00\x00\x02\x02\x00\x00^\x02\x00\x00floating"
.string "\x04\x00\x00\x00\n\x00\x00\x00\x98\x01\x00\x00^\x02\x00\x00integers"
.string "\x04\x00\x00\x00\x05\x00\x00\x00,\x01\x00\x00^\x02\x00\x00floats"
.string "\x04\x00\x00\x00\x00\x00\x00\x00\x14\x00\x00\x00^\x02\x00\x00inferred"
.string "nano.source_file"

.metadata 6 7
.metadata 8 7
.metadata 9 7
.metadata 6 10
.metadata 8 10
.metadata 9 10
.metadata 6 11
.metadata 8 11
.metadata 9 11
.metadata 6 12
.metadata 8 12
.metadata 9 12
.metadata 6 13
.metadata 8 13
.metadata 9 14
.metadata 16 17
.metadata 16 18
.metadata 16 19
.metadata 16 20
.metadata 16 21
.metadata 22 15

.types 2 0 0

.flag debug_info

.debug 0 6 45
.debug 18 9 42
.debug 36 12 31
.debug 64 18 27
.debug 92 25 5
.debug 112 26 5
.debug 147 27 5
.debug 183 28 5
.debug 212 29 5
.debug 248 30 5
.debug 284 31 5
.debug 392 32 5
.debug 500 33 5
.debug 515 34 5
.debug 549 35 5
.debug 564 36 5
.debug 598 37 5
.debug 606 38 5
.debug 614 39 5
.debug 651 40 5
.debug 688 41 5

.entry 4

.function floating_rows 0 0 0 array 1
  PUSH_F64 bits:3ff8000000000000
  ARR_LITERAL 3 1
  ARR_LITERAL 7 1
  RET
.end

.function integer_rows 0 0 0 array 1
  PUSH_I64 1
  ARR_LITERAL 1 1
  ARR_LITERAL 7 1
  RET
.end

.function float_box 0 0 0 struct 1
  PUSH_F64 bits:3ff8000000000000
  ARR_LITERAL 3 1
  ARR_LITERAL 7 1
  AGG_PACK 0 0 0 1
  RET
.end

.function int_box 0 0 0 struct 1
  PUSH_I64 1
  ARR_LITERAL 1 1
  ARR_LITERAL 7 1
  AGG_PACK 0 1 0 1
  RET
.end

.function main 0 13 0 int 1
  PUSH_F64 bits:3ff8000000000000
  ARR_LITERAL 3 1
  ARR_LITERAL 7 1
  STORE_LOCAL 0
  LOAD_LOCAL 0
  PUSH_I64 0
  ARR_GET
  PUSH_I64 0
  ARR_GET
  CAST_FLOAT
  PUSH_F64 bits:3ff8000000000000
  F64_EQ
  ASSERT
  LOAD_GLOBAL 0
  PUSH_I64 0
  ARR_GET
  PUSH_I64 0
  ARR_GET
  PUSH_F64 bits:3ff8000000000000
  F64_EQ
  ASSERT
  PUSH_I64 1
  ARR_LITERAL 1 1
  ARR_LITERAL 7 1
  ARR_LEN
  PUSH_I64 1
  I64_EQ
  ASSERT
  CALL 0
  PUSH_I64 0
  ARR_GET
  PUSH_I64 0
  ARR_GET
  PUSH_F64 bits:3ff8000000000000
  F64_EQ
  ASSERT
  CALL 1
  PUSH_I64 0
  ARR_GET
  PUSH_I64 0
  ARR_GET
  PUSH_I64 1
  I64_EQ
  ASSERT
  PUSH_I64 2
  STORE_LOCAL 1
  PUSH_F64 bits:3ff8000000000000
  ARR_LITERAL 3 1
  STORE_LOCAL 2
  LOAD_LOCAL 1
  PUSH_I64 0
  I64_GE_S
  ASSERT
  ARR_NEW 7
  STORE_LOCAL 3
  PUSH_I64 0
  STORE_LOCAL 4
L1:
  LOAD_LOCAL 4
  LOAD_LOCAL 1
  LT
  JMP_FALSE L0
  LOAD_LOCAL 3
  LOAD_LOCAL 2
  ARR_PUSH
  STORE_LOCAL 3
  LOAD_LOCAL 4
  PUSH_I64 1
  I64_ADD
  STORE_LOCAL 4
  JMP L1
L0:
  LOAD_LOCAL 3
  STORE_LOCAL 5
  PUSH_I64 2
  STORE_LOCAL 6
  PUSH_I64 1
  ARR_LITERAL 1 1
  STORE_LOCAL 7
  LOAD_LOCAL 6
  PUSH_I64 0
  I64_GE_S
  ASSERT
  ARR_NEW 7
  STORE_LOCAL 8
  PUSH_I64 0
  STORE_LOCAL 9
L3:
  LOAD_LOCAL 9
  LOAD_LOCAL 6
  LT
  JMP_FALSE L2
  LOAD_LOCAL 8
  LOAD_LOCAL 7
  ARR_PUSH
  STORE_LOCAL 8
  LOAD_LOCAL 9
  PUSH_I64 1
  I64_ADD
  STORE_LOCAL 9
  JMP L3
L2:
  LOAD_LOCAL 8
  STORE_LOCAL 10
  LOAD_LOCAL 5
  ARR_LEN
  PUSH_I64 2
  I64_EQ
  ASSERT
  LOAD_LOCAL 5
  PUSH_I64 1
  ARR_GET
  PUSH_I64 0
  ARR_GET
  PUSH_F64 bits:3ff8000000000000
  F64_EQ
  ASSERT
  LOAD_LOCAL 10
  ARR_LEN
  PUSH_I64 2
  I64_EQ
  ASSERT
  LOAD_LOCAL 10
  PUSH_I64 1
  ARR_GET
  PUSH_I64 0
  ARR_GET
  PUSH_I64 1
  I64_EQ
  ASSERT
  CALL 2
  STORE_LOCAL 11
  CALL 3
  STORE_LOCAL 12
  LOAD_LOCAL 11
  AGG_GET 0
  PUSH_I64 0
  ARR_GET
  PUSH_I64 0
  ARR_GET
  PUSH_F64 bits:3ff8000000000000
  F64_EQ
  ASSERT
  LOAD_LOCAL 12
  AGG_GET 0
  PUSH_I64 0
  ARR_GET
  PUSH_I64 0
  ARR_GET
  PUSH_I64 1
  I64_EQ
  ASSERT
  PUSH_I64 0
  RET
.end

.function __init__ 0 0 0 void 0
  PUSH_F64 bits:3ff8000000000000
  ARR_LITERAL 3 1
  ARR_LITERAL 7 1
  STORE_GLOBAL 0
  RET
.end

