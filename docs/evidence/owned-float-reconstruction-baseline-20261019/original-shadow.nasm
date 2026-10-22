.string "Handle"
.string "fd"
.string "__nanoisa_shadow_entry"
.string "close"
.string "main"
.string "nanolang.shadow.offset.0"
.string "nanolang.shadow.offset.37"
.string "nano.local.v1"
.string "\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x00\x00\x00owned"
.string "\x01\x00\x00\x00\x02\x00\x00\x00\x15\x00\x00\x00\x1d\x00\x00\x00fd"
.string "\x02\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x00\x00\\\x00\x00\x00fd"
.string "\x02\x00\x00\x00\x03\x00\x00\x002\x00\x00\x00D\x00\x00\x00fd"
.string "nano.affine.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00"
.string "\x01\x00\x00\x00\x00\x00\x00\x00"
.string "\x02\x00\x00\x00\x00\x00\x00\x00"

.metadata 5 3
.metadata 6 4
.metadata 7 8
.metadata 7 9
.metadata 7 10
.metadata 7 11
.metadata 12 13
.metadata 12 14
.metadata 12 15

.types 1 0 0

.layouts "01000000000001000000000001000000ffffffff01000000"

.ownership "010000000100000003000000030000000100000001000000ffffffff01000000"
.ownership "ffffffff0300010001000000ffffffff08000000000000000800000000000000"
.ownership "01000000ffffffff0400000001000000ffffffff03000000ffffffff01000000"
.ownership "ffffffff080000000000000001000000ffffffff"

.entry 0

.function __nanoisa_shadow_entry 0 1 0 int 1
  NOP
  PUSH_I64 7
  STORE_LOCAL 0
  LOAD_LOCAL 0
  OWN_PACK 0
  CALL 1
  PUSH_I64 7
  EQ
  ASSERT
  NOP
  CALL 2
  PUSH_I64 0
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

.function main 0 4 0 int 1
  PUSH_F64 bits:4004000000000000
  STORE_LOCAL 0
  PUSH_I64 42
  STORE_LOCAL 1
  LOAD_LOCAL 1
  OWN_PACK 0
  OWN_STORE_LOCAL 2
  REGION_BEGIN
  BORROW_LOCAL_SHARED 0 2
  REF_GET 0 0
  REGION_END
  STORE_LOCAL 3
  LOAD_LOCAL 3
  PUSH_I64 42
  EQ
  ASSERT
  OWN_UNPACK_LOCAL 2
  POP
  LOAD_LOCAL 0
  PUSH_F64 bits:4004000000000000
  F64_EQ
  ASSERT
  PUSH_I64 0
  RET
.end

