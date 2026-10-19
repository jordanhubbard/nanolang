.string "Handle"
.string "fd"
.string "Empty"
.string "__nanoisa_shadow_entry"
.string "close"
.string "main"
.string "nanolang.shadow.offset.0"
.string "nanolang.shadow.offset.37"
.string "nano.local.v1"
.string "\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x00\x00\x00owned"
.string "\x01\x00\x00\x00\x02\x00\x00\x00\x15\x00\x00\x00\x1d\x00\x00\x00fd"
.string "\x02\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00\x1b\x00\x00\x00value"
.string "nano.affine.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00"
.string "\x01\x00\x00\x00\x00\x00\x00\x00"
.string "\x02\x00\x00\x00\x00\x00\x00\x00"

.metadata 6 4
.metadata 7 5
.metadata 8 9
.metadata 8 10
.metadata 8 11
.metadata 12 13
.metadata 12 14
.metadata 12 15

.types 2 0 0

.layouts "02000000000001000000000001000000ffffffff010000000000000002000000"

.ownership "010000000200000003030000030000000100000001000000ffffffff01000000"
.ownership "ffffffff0300010001000000ffffffff08000000000000000800000000000000"
.ownership "01000000ffffffff0200000001000000ffffffff080000000100000008000000"
.ownership "01000000"

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

.function main 0 2 0 int 1
  OWN_PACK 1
  OWN_STORE_LOCAL 0
  OWN_MOVE_LOCAL 0
  OWN_STORE_LOCAL 1
  PUSH_I64 0
  OWN_UNPACK_LOCAL 1
  RET
.end

