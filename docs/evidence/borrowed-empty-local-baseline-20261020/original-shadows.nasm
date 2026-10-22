.string "Empty"
.string "Counter"
.string "value"
.string "active"
.string "__nanoisa_shadow_entry"
.string "update"
.string "nanolang.shadow.offset.0"
.string "nanolang.shadow.offset.154"
.string "main"
.string "nano.local.v1"
.string "\x00\x00\x00\x00\x02\x00\x00\x00 \x00\x00\x00\x9a\x00\x00\x00owner"
.string "\x00\x00\x00\x00\x04\x00\x00\x00s\x00\x00\x00\x9a\x00\x00\x00value"
.string "\x00\x00\x00\x00\x05\x00\x00\x00\x82\x00\x00\x00\x9a\x00\x00\x00active"
.string "\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00%\x00\x00\x00view"
.string "nano.affine.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00"
.string "\x01\x00\x00\x00\x00\x00\x00\x00"

.metadata 6 5
.metadata 7 8
.metadata 9 10
.metadata 9 11
.metadata 9 12
.metadata 9 13
.metadata 14 15
.metadata 14 16

.types 2 0 0

.layouts "020000000000000000000000000002000100000001000000ffffffff02000000"
.layouts "04000000ffffffff03000000"

.ownership "010000000200000003030000020000000600000001000000ffffffff01000000"
.ownership "ffffffff04000000ffffffff0800000001000000080000000100000001000000"
.ownership "ffffffff04000000ffffffff0100010001000000ffffffff0802000001000000"

.entry 0

.function __nanoisa_shadow_entry 0 6 0 int 1
  NOP
  PUSH_I64 7
  STORE_LOCAL 0
  PUSH_BOOL 1
  STORE_LOCAL 1
  LOAD_LOCAL 0
  LOAD_LOCAL 1
  OWN_PACK 1
  OWN_STORE_LOCAL 2
  REGION_BEGIN
  BORROW_LOCAL_EXCLUSIVE 0 2
  CALL_REF 1 0
  REGION_END
  PUSH_I64 10
  EQ
  ASSERT
  REGION_BEGIN
  BORROW_LOCAL_SHARED 0 2
  REF_GET 0 0
  REGION_END
  PUSH_I64 10
  EQ
  ASSERT
  REGION_BEGIN
  BORROW_LOCAL_SHARED 0 2
  REF_GET 0 1
  REGION_END
  NOT
  ASSERT
  OWN_MOVE_LOCAL 2
  OWN_STORE_LOCAL 3
  REGION_BEGIN
  BORROW_LOCAL_SHARED 0 3
  REF_GET 0 0
  REGION_END
  STORE_LOCAL 4
  REGION_BEGIN
  BORROW_LOCAL_SHARED 0 3
  REF_GET 0 1
  REGION_END
  STORE_LOCAL 5
  LOAD_LOCAL 4
  PUSH_I64 10
  EQ
  ASSERT
  LOAD_LOCAL 5
  NOT
  ASSERT
  OWN_UNPACK_LOCAL 3
  POP
  POP
  NOP
  PUSH_BOOL 1
  ASSERT
  PUSH_I64 0
  RET
.end

.function update 1 1 0 int 1
  REF_GET 0 0
  PUSH_I64 3
  ADD
  REF_SET 0 0
  REF_GET 0 1
  NOT
  REF_SET 0 1
  REF_GET 0 0
  RET
.end

.parameters 1 struct

