.string "FileHandle"
.string "fd"
.string "main"
.string "open_file"
.string "close_file"
.string "audit.log"
.string "resource closed"
.string "nano.local.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00\r\x00\x00\x00\x1f\x00\x00\x00f"
.string "\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x19\x00\x00\x00path"
.string "\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00.\x00\x00\x00_f"
.string "\x02\x00\x00\x00\x02\x00\x00\x00\x15\x00\x00\x00.\x00\x00\x00fd"
.string "nano.affine.v1"
.string "\x00\x00\x00\x00\x01\x00\x00\x00"
.string "\x01\x00\x00\x00\x01\x00\x00\x00"
.string "\x02\x00\x00\x00\x01\x00\x00\x00"
.string "nano.affine-use.v1"
.string "\x00\x00\x00\x00)\x00\x00\x00\x11\x00\x00\x00\x02\x02\x00\x00Z\x00\x00\x00\x01\x00\x00\x00/Users/jkh/Src/nanolang-owned-nested-depth2/examples/language/nl_affine_resource_demo.nanof"
.string "\x02\x00\x00\x00\x1d\x00\x00\x00\x1d\x00\x00\x00\x02\x02\x00\x00Z\x00\x00\x00\x02\x00\x00\x00/Users/jkh/Src/nanolang-owned-nested-depth2/examples/language/nl_affine_resource_demo.nano_f"

.metadata 7 8
.metadata 7 9
.metadata 7 10
.metadata 7 11
.metadata 12 13
.metadata 12 14
.metadata 12 15
.metadata 16 17
.metadata 16 18

.types 1 0 0

.layouts "01000000000001000000000001000000ffffffff01000000"

.ownership "010000000100000003000000030000000100000001000000ffffffff08000000"
.ownership "0000000002000100080000000000000005000000ffffffff01000000ffffffff"
.ownership "0300010000000000ffffffff0800000000000000080000000000000001000000"
.ownership "ffffffff"

.entry 0

.function main 0 1 0 int 1
  PUSH_STR 5
  CALL 1
  OWN_STORE_LOCAL 0
  OWN_MOVE_LOCAL 0
  CALL 2
  PUSH_I64 0
  RET
.end

.function open_file 1 2 0 struct 1
  LOAD_LOCAL 0
  PRINTLN
  PUSH_I64 42
  STORE_LOCAL 1
  LOAD_LOCAL 1
  OWN_PACK 0
  RET
.end

.parameters 1 string

.function close_file 1 3 0 void 0
  OWN_MOVE_LOCAL 0
  OWN_STORE_LOCAL 1
  REGION_BEGIN
  BORROW_LOCAL_SHARED 1 1
  REF_GET 1 0
  REGION_END
  STORE_LOCAL 2
  LOAD_LOCAL 2
  PUSH_I64 0
  GT
  ASSERT
  PUSH_STR 6
  PRINTLN
  OWN_UNPACK_LOCAL 1
  POP
  RET
.end

.parameters 2 struct

