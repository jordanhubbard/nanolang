.string "main"
.string "nano.purity.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00"
.string "nano.affine.v1"
.string "nano.effects.v1"
.string "\x00\x00\x00\x00\x02\x00\x00\x00"
.string "nano.local.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00\x14\x00\x00\x00]\x02\x00\x00inferred"
.string "\x00\x00\x00\x00\x05\x00\x00\x00+\x01\x00\x00]\x02\x00\x00floats"
.string "\x00\x00\x00\x00\n\x00\x00\x00\x97\x01\x00\x00]\x02\x00\x00integers"
.string "\x00\x00\x00\x00\x0b\x00\x00\x00\x01\x02\x00\x00]\x02\x00\x00floating"
.string "\x00\x00\x00\x00\x0c\x00\x00\x00\t\x02\x00\x00]\x02\x00\x00integer"
.string "floating_rows"
.string "\x01\x00\x00\x00\x00\x00\x00\x00"
.string "integer_rows"
.string "\x02\x00\x00\x00\x00\x00\x00\x00"
.string "float_box"
.string "\x03\x00\x00\x00\x00\x00\x00\x00"
.string "int_box"
.string "\x04\x00\x00\x00\x00\x00\x00\x00"
.string "__init__"

.metadata 1 2
.metadata 3 2
.metadata 4 5
.metadata 6 7
.metadata 6 8
.metadata 6 9
.metadata 6 10
.metadata 6 11
.metadata 1 13
.metadata 3 13
.metadata 4 13
.metadata 1 15
.metadata 3 15
.metadata 4 15
.metadata 1 17
.metadata 3 17
.metadata 4 17
.metadata 1 19
.metadata 3 19
.metadata 4 19

.types 2 0 0

.entry 0

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
  CALL 1
  PUSH_I64 0
  ARR_GET
  PUSH_I64 0
  ARR_GET
  PUSH_F64 bits:3ff8000000000000
  F64_EQ
  ASSERT
  CALL 2
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
  CALL 3
  STORE_LOCAL 11
  CALL 4
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

.function __init__ 0 0 0 void 0
  PUSH_F64 bits:3ff8000000000000
  ARR_LITERAL 3 1
  ARR_LITERAL 7 1
  STORE_GLOBAL 0
  RET
.end

