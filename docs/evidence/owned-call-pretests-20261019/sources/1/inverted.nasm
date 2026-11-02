.string "Box"
.string "value"
.string "flag"
.string "main"
.string "pick"
.string "nano.local.v1"
.string "\x00\x00\x00\x00\x02\x00\x00\x00-\x00\x00\x00\xc4\x00\x00\x00zero"
.string "\x00\x00\x00\x00\x05\x00\x00\x00Z\x00\x00\x00\xc4\x00\x00\x00one"
.string "\x00\x00\x00\x00\x08\x00\x00\x00\x87\x00\x00\x00\xc4\x00\x00\x00many"
.string "\x00\x00\x00\x00\x0b\x00\x00\x00\xb4\x00\x00\x00\xc4\x00\x00\x00disabled"
.string "\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00|\x00\x00\x00owner"
.string "\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00|\x00\x00\x00count"
.string "\x01\x00\x00\x00\x02\x00\x00\x00\x0c\x00\x00\x00|\x00\x00\x00i"
.string "\x01\x00\x00\x00\x04\x00\x00\x00`\x00\x00\x00|\x00\x00\x00value"
.string "\x01\x00\x00\x00\x05\x00\x00\x00o\x00\x00\x00|\x00\x00\x00flag"
.string "nano.affine.v1"
.string "\x00\x00\x00\x00\x01\x00\x00\x00"
.string "\x01\x00\x00\x00\x01\x00\x00\x00"
.string "nano.affine-use.v1"
.string "\x01\x00\x00\x00\x04\x00\x00\x00\r\x00\x00\x00\x01\x01\x00\x00h\x00\x00\x00\x05\x00\x00\x00/Users/jkh/Src/nanolang-owned-nested-depth2/tests/nanoisa/fixtures/owned_observation_loops/multiple.nanoowner"
.string "\x01\x00\x00\x00\x04\x00\x00\x00.\x00\x00\x00\x01\x01\x00\x00h\x00\x00\x00\x05\x00\x00\x00/Users/jkh/Src/nanolang-owned-nested-depth2/tests/nanoisa/fixtures/owned_observation_loops/multiple.nanoowner"
.string "\x01\x00\x00\x00\x05\x00\x00\x00\x1c\x00\x00\x00\x02\x01\x00\x00h\x00\x00\x00\x05\x00\x00\x00/Users/jkh/Src/nanolang-owned-nested-depth2/tests/nanoisa/fixtures/owned_observation_loops/multiple.nanoowner"

.metadata 5 6
.metadata 5 7
.metadata 5 8
.metadata 5 9
.metadata 5 10
.metadata 5 11
.metadata 5 12
.metadata 5 13
.metadata 5 14
.metadata 15 16
.metadata 15 17
.metadata 18 19
.metadata 18 20
.metadata 18 21

.types 1 0 0

.layouts "01000000000002000000000001000000ffffffff0100000004000000ffffffff"
.layouts "02000000"

.ownership "010000000100000003000000020000000c00000001000000ffffffff01000000"
.ownership "ffffffff04000000ffffffff01000000ffffffff01000000ffffffff04000000"
.ownership "ffffffff01000000ffffffff01000000ffffffff04000000ffffffff01000000"
.ownership "ffffffff01000000ffffffff04000000ffffffff01000000ffffffff06000200"
.ownership "01000000ffffffff080000000000000001000000ffffffff01000000ffffffff"
.ownership "080000000000000001000000ffffffff04000000ffffffff"

.entry 0

.function main 0 12 0 int 1
  PUSH_I64 3
  STORE_LOCAL 0
  PUSH_BOOL 1
  STORE_LOCAL 1
  LOAD_LOCAL 0
  LOAD_LOCAL 1
  OWN_PACK 0
  PUSH_I64 0
  CALL 1
  STORE_LOCAL 2
  PUSH_I64 3
  STORE_LOCAL 3
  PUSH_BOOL 1
  STORE_LOCAL 4
  LOAD_LOCAL 3
  LOAD_LOCAL 4
  OWN_PACK 0
  PUSH_I64 1
  CALL 1
  STORE_LOCAL 5
  PUSH_I64 3
  STORE_LOCAL 6
  PUSH_BOOL 1
  STORE_LOCAL 7
  LOAD_LOCAL 6
  LOAD_LOCAL 7
  OWN_PACK 0
  PUSH_I64 5
  CALL 1
  STORE_LOCAL 8
  PUSH_I64 3
  STORE_LOCAL 9
  PUSH_BOOL 0
  STORE_LOCAL 10
  LOAD_LOCAL 9
  LOAD_LOCAL 10
  OWN_PACK 0
  PUSH_I64 5
  CALL 1
  STORE_LOCAL 11
  LOAD_LOCAL 2
  LOAD_LOCAL 5
  ADD
  LOAD_LOCAL 8
  LOAD_LOCAL 11
  ADD
  ADD
  RET
.end

.function pick 2 6 0 int 1
  PUSH_I64 0
  STORE_LOCAL 2
L1:
  REGION_BEGIN
  BORROW_LOCAL_SHARED 2 0
  REF_GET 2 1
  REGION_END
  LOAD_LOCAL 2
  LOAD_LOCAL 1
  LT
  LOAD_LOCAL 2
  REGION_BEGIN
  BORROW_LOCAL_SHARED 2 0
  REF_GET 2 0
  REGION_END
  LT
  AND
  AND
  NOT
  JMP_TRUE L0
  LOAD_LOCAL 2
  PUSH_I64 1
  ADD
  STORE_LOCAL 2
  JMP L1
L0:
  OWN_MOVE_LOCAL 0
  OWN_STORE_LOCAL 3
  REGION_BEGIN
  BORROW_LOCAL_SHARED 2 3
  REF_GET 2 0
  REGION_END
  STORE_LOCAL 4
  REGION_BEGIN
  BORROW_LOCAL_SHARED 2 3
  REF_GET 2 1
  REGION_END
  STORE_LOCAL 5
  LOAD_LOCAL 4
  LOAD_LOCAL 2
  ADD
  OWN_UNPACK_LOCAL 3
  POP
  POP
  RET
.end

.parameters 1 struct int

