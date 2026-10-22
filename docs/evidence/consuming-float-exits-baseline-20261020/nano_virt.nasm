.string "Handle"
.string "fd"
.string "main"
.string "close"
.string "worker"
.string "nano.local.v1"
.string "\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x00\x00\x00owned"
.string "\x01\x00\x00\x00\x02\x00\x00\x00\x15\x00\x00\x00\x1d\x00\x00\x00fd"
.string "\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00C\x01\x00\x00owned"
.string "\x02\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00C\x01\x00\x00stop"
.string "\x02\x00\x00\x00\x02\x00\x00\x00\x0c\x00\x00\x00C\x01\x00\x00value"
.string "\x02\x00\x00\x00\x03\x00\x00\x00\x12\x00\x00\x00C\x01\x00\x00copy"
.string "\x02\x00\x00\x00\x04\x00\x00\x006\x00\x00\x00C\x01\x00\x00index"
.string "\x02\x00\x00\x00\x05\x00\x00\x00\xbe\x00\x00\x00C\x01\x00\x00value"
.string "\x02\x00\x00\x00\x06\x00\x00\x00\xd8\x00\x00\x00C\x01\x00\x00inf"
.string "\x02\x00\x00\x00\x07\x00\x00\x00\x14\x01\x00\x00C\x01\x00\x00nan"
.string "nano.affine.v1"
.string "\x00\x00\x00\x00\x01\x00\x00\x00"
.string "\x01\x00\x00\x00\x01\x00\x00\x00"
.string "\x02\x00\x00\x00\x01\x00\x00\x00"
.string "nano.affine-use.v1"
.string "\x01\x00\x00\x00\x02\x00\x00\x006\x00\x00\x00\x02\x01\x00\x00B\x00\x00\x00\x05\x00\x00\x00/private/tmp/consuming-float-exits-baseline-20261020/original.nanoowned"
.string "\x02\x00\x00\x00 \x00\x00\x00\x17\x00\x00\x00\x02\x01\x00\x00B\x00\x00\x00\x05\x00\x00\x00/private/tmp/consuming-float-exits-baseline-20261020/original.nanoowned"

.metadata 5 6
.metadata 5 7
.metadata 5 8
.metadata 5 9
.metadata 5 10
.metadata 5 11
.metadata 5 12
.metadata 5 13
.metadata 5 14
.metadata 5 15
.metadata 16 17
.metadata 16 18
.metadata 16 19
.metadata 20 21
.metadata 20 22

.types 1 0 0

.layouts "01000000000001000000000001000000ffffffff01000000"

.ownership "010000000100000003000000030000000200000001000000ffffffff01000000"
.ownership "ffffffff01000000ffffffff0300010001000000ffffffff0800000000000000"
.ownership "080000000000000001000000ffffffff0800020001000000ffffffff08000000"
.ownership "0000000004000000ffffffff03000000ffffffff03000000ffffffff01000000"
.ownership "ffffffff03000000ffffffff03000000ffffffff03000000ffffffff"

.entry 0

.function main 0 2 0 int 1
  PUSH_I64 8
  STORE_LOCAL 0
  LOAD_LOCAL 0
  OWN_PACK 0
  PUSH_BOOL 0
  CALL 2
  PUSH_I64 8
  EQ
  ASSERT
  PUSH_I64 9
  STORE_LOCAL 1
  LOAD_LOCAL 1
  OWN_PACK 0
  PUSH_BOOL 1
  CALL 2
  PUSH_I64 9
  EQ
  ASSERT
  PUSH_I64 0
  RET
.end

.function close 1 3 0 int 1
  OWN_MOVE_LOCAL 0
  OWN_STORE_LOCAL 1
  REGION_BEGIN
  BORROW_LOCAL_SHARED 1 1
  REF_GET 1 0
  REGION_END
  STORE_LOCAL 2
  LOAD_LOCAL 2
  OWN_UNPACK_LOCAL 1
  POP
  RET
.end

.parameters 1 struct

.function worker 2 8 0 int 1
  PUSH_F64 bits:3ff8000000000000
  STORE_LOCAL 2
  LOAD_LOCAL 2
  STORE_LOCAL 3
L1:
  PUSH_BOOL 0
  JMP_FALSE L0
  PUSH_F64 bits:4058c00000000000
  STORE_LOCAL 2
  JMP L1
L0:
  PUSH_I64 0
  STORE_LOCAL 4
L4:
  LOAD_LOCAL 4
  PUSH_I64 3
  LT
  JMP_FALSE L2
  LOAD_LOCAL 4
  PUSH_I64 1
  ADD
  STORE_LOCAL 4
  LOAD_LOCAL 1
  JMP_FALSE L3
  JMP L2
L3:
  LOAD_LOCAL 2
  PUSH_F64 bits:3ff0000000000000
  F64_ADD
  STORE_LOCAL 2
  JMP L4
L2:
  LOAD_LOCAL 3
  PUSH_F64 bits:3ff8000000000000
  F64_EQ
  ASSERT
  LOAD_LOCAL 1
  JMP_FALSE L5
  LOAD_LOCAL 2
  PUSH_F64 bits:3ff8000000000000
  F64_EQ
  ASSERT
  JMP L6
L5:
  LOAD_LOCAL 2
  PUSH_F64 bits:4012000000000000
  F64_EQ
  ASSERT
L6:
  PUSH_F64 bits:4004000000000000
  F64_NEG
  STORE_LOCAL 5
  LOAD_LOCAL 5
  PUSH_F64 bits:0000000000000000
  F64_LT
  ASSERT
  PUSH_F64 bits:4202a05f20000000
  STORE_LOCAL 6
  LOAD_LOCAL 6
  LOAD_LOCAL 6
  F64_MUL
  STORE_LOCAL 6
  LOAD_LOCAL 6
  LOAD_LOCAL 6
  F64_MUL
  STORE_LOCAL 6
  LOAD_LOCAL 6
  LOAD_LOCAL 6
  F64_MUL
  STORE_LOCAL 6
  LOAD_LOCAL 6
  LOAD_LOCAL 6
  F64_MUL
  STORE_LOCAL 6
  LOAD_LOCAL 6
  LOAD_LOCAL 6
  F64_MUL
  STORE_LOCAL 6
  LOAD_LOCAL 6
  LOAD_LOCAL 6
  F64_SUB
  STORE_LOCAL 7
  LOAD_LOCAL 7
  LOAD_LOCAL 7
  F64_NE
  ASSERT
  LOAD_LOCAL 7
  PUSH_F64 bits:3ff0000000000000
  F64_LE
  NOT
  ASSERT
  LOAD_LOCAL 7
  PUSH_F64 bits:3ff0000000000000
  F64_GE
  NOT
  ASSERT
  OWN_MOVE_LOCAL 0
  CALL 1
  RET
.end

.parameters 2 struct bool

