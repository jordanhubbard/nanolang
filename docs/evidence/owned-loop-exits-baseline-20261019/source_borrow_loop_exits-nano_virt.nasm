.string "Counter"
.string "value"
.string "active"
.string "main"
.string "bump"
.string "nano.local.v1"
.string "\x00\x00\x00\x00\x02\x00\x00\x00\x1f\x00\x00\x00\xcf\x01\x00\x00owner"
.string "\x00\x00\x00\x00\x03\x00\x00\x00+\x00\x00\x00\xcf\x01\x00\x00i"
.string "\x00\x00\x00\x00\x06\x00\x00\x00f\x00\x00\x00=\x01\x00\x00transient"
.string "\x00\x00\x00\x00\x08\x00\x00\x00{\x00\x00\x00=\x01\x00\x00value"
.string "\x00\x00\x00\x00\t\x00\x00\x00\x8a\x00\x00\x00=\x01\x00\x00active"
.string "\x00\x00\x00\x00\n\x00\x00\x00\xda\x00\x00\x00=\x01\x00\x00j"
.string "\x00\x00\x00\x00\x0c\x00\x00\x00}\x01\x00\x00\xcf\x01\x00\x00value"
.string "\x00\x00\x00\x00\r\x00\x00\x00\x8c\x01\x00\x00\xcf\x01\x00\x00active"
.string "\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00d\x00\x00\x00view"
.string "\x01\x00\x00\x00\x01\x00\x00\x00\x0c\x00\x00\x00d\x00\x00\x00turns"
.string "nano.affine.v1"
.string "\x00\x00\x00\x00\x01\x00\x00\x00"
.string "\x01\x00\x00\x00\x01\x00\x00\x00"
.string "nano.affine-use.v1"
.string "\x00\x00\x00\x00*\x00\x00\x00#\x00\x00\x00\x02\x01\x00\x00`\x00\x00\x00\t\x00\x00\x00/Users/jkh/Src/nanolang-owned-nested-depth2/tests/nanoisa/fixtures/source_borrow_loop_exits.nanotransient"
.string "\x00\x00\x00\x004\x00\x00\x00\x19\x00\x00\x00\x01\x01\x00\x00`\x00\x00\x00\x05\x00\x00\x00/Users/jkh/Src/nanolang-owned-nested-depth2/tests/nanoisa/fixtures/source_borrow_loop_exits.nanoowner"
.string "\x00\x00\x00\x009\x00\x00\x00\"\x00\x00\x00\x02\x01\x00\x00`\x00\x00\x00\x05\x00\x00\x00/Users/jkh/Src/nanolang-owned-nested-depth2/tests/nanoisa/fixtures/source_borrow_loop_exits.nanoowner"
.string "\x01\x00\x00\x00\x07\x00\x00\x00\x15\x00\x00\x00\x01\x01\x00\x00`\x00\x00\x00\x04\x00\x00\x00/Users/jkh/Src/nanolang-owned-nested-depth2/tests/nanoisa/fixtures/source_borrow_loop_exits.nanoview"
.string "\x01\x00\x00\x00\n\x00\x00\x00\t\x00\x00\x00\x01\x01\x00\x00`\x00\x00\x00\x04\x00\x00\x00/Users/jkh/Src/nanolang-owned-nested-depth2/tests/nanoisa/fixtures/source_borrow_loop_exits.nanoview"

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
.metadata 19 20
.metadata 19 21
.metadata 19 22
.metadata 19 23
.metadata 19 24

.types 1 0 0

.layouts "01000000000002000000000001000000ffffffff0100000004000000ffffffff"
.layouts "02000000"

.ownership "010000000100000003000000020000000e00000001000000ffffffff01000000"
.ownership "ffffffff04000000ffffffff080000000000000001000000ffffffff01000000"
.ownership "ffffffff04000000ffffffff0800000000000000080000000000000001000000"
.ownership "ffffffff04000000ffffffff01000000ffffffff080000000000000001000000"
.ownership "ffffffff04000000ffffffff0200010001000000ffffffff0802000000000000"
.ownership "01000000ffffffff"

.entry 0

.function main 0 14 0 int 1
  PUSH_I64 0
  STORE_LOCAL 0
  PUSH_BOOL 1
  STORE_LOCAL 1
  LOAD_LOCAL 0
  LOAD_LOCAL 1
  OWN_PACK 0
  OWN_STORE_LOCAL 2
  PUSH_I64 0
  STORE_LOCAL 3
L2:
  LOAD_LOCAL 3
  PUSH_I64 4
  LT
  JMP_FALSE L0
  LOAD_LOCAL 3
  PUSH_I64 1
  ADD
  STORE_LOCAL 3
  LOAD_LOCAL 3
  STORE_LOCAL 4
  PUSH_BOOL 1
  STORE_LOCAL 5
  LOAD_LOCAL 4
  LOAD_LOCAL 5
  OWN_PACK 0
  OWN_STORE_LOCAL 6
  OWN_MOVE_LOCAL 6
  OWN_STORE_LOCAL 7
  REGION_BEGIN
  BORROW_LOCAL_SHARED 0 7
  REF_GET 0 0
  REGION_END
  STORE_LOCAL 8
  REGION_BEGIN
  BORROW_LOCAL_SHARED 0 7
  REF_GET 0 1
  REGION_END
  STORE_LOCAL 9
  LOAD_LOCAL 8
  LOAD_LOCAL 3
  EQ
  ASSERT
  LOAD_LOCAL 9
  ASSERT
  LOAD_LOCAL 3
  PUSH_I64 1
  EQ
  JMP_FALSE L1
  OWN_UNPACK_LOCAL 7
  POP
  POP
  JMP L2
L1:
  LOAD_LOCAL 3
  PUSH_I64 4
  EQ
  JMP_FALSE L3
  OWN_UNPACK_LOCAL 7
  POP
  POP
  JMP L0
L3:
  PUSH_I64 0
  STORE_LOCAL 10
L6:
  LOAD_LOCAL 10
  PUSH_I64 3
  LT
  JMP_FALSE L4
  LOAD_LOCAL 10
  PUSH_I64 1
  ADD
  STORE_LOCAL 10
  PUSH_BOOL 1
  JMP_FALSE L5
  JMP L4
L5:
  JMP L6
L4:
  LOAD_LOCAL 10
  PUSH_I64 1
  EQ
  ASSERT
  REGION_BEGIN
  BORROW_LOCAL_EXCLUSIVE 0 2
  CALL_REF 1 0
  REGION_END
  LOAD_LOCAL 3
  PUSH_I64 1
  SUB
  EQ
  ASSERT
  OWN_UNPACK_LOCAL 7
  POP
  POP
  JMP L2
L0:
  PUSH_BOOL 0
  JMP_FALSE L7
  JMP L7
L7:
  PUSH_BOOL 0
  JMP_FALSE L8
  JMP L7
L8:
  LOAD_LOCAL 3
  PUSH_I64 4
  EQ
  ASSERT
  OWN_MOVE_LOCAL 2
  OWN_STORE_LOCAL 11
  REGION_BEGIN
  BORROW_LOCAL_SHARED 0 11
  REF_GET 0 0
  REGION_END
  STORE_LOCAL 12
  REGION_BEGIN
  BORROW_LOCAL_SHARED 0 11
  REF_GET 0 1
  REGION_END
  STORE_LOCAL 13
  LOAD_LOCAL 12
  PUSH_I64 2
  EQ
  ASSERT
  LOAD_LOCAL 13
  ASSERT
  PUSH_BOOL 1
  JMP_FALSE L9
  PUSH_BOOL 1
  JMP_FALSE L10
  PUSH_I64 0
  OWN_UNPACK_LOCAL 11
  POP
  POP
  RET
L10:
  JMP L9
L9:
  PUSH_I64 0
  OWN_UNPACK_LOCAL 11
  POP
  POP
  RET
.end

.function bump 1 2 0 int 1
  PUSH_I64 0
  STORE_LOCAL 1
L2:
  LOAD_LOCAL 1
  PUSH_I64 3
  LT
  JMP_FALSE L0
  LOAD_LOCAL 1
  PUSH_I64 1
  ADD
  STORE_LOCAL 1
  LOAD_LOCAL 1
  PUSH_I64 1
  EQ
  JMP_FALSE L1
  JMP L2
L1:
  REF_GET 0 0
  PUSH_I64 1
  ADD
  REF_SET 0 0
  JMP L0
L0:
  REF_GET 0 0
  RET
.end

.parameters 1 struct

