.string "FileHandle"
.string "fd"
.string "__nanoisa_shadow_entry"
.string "open_file"
.string "close_file"
.string "main"
.string "audit.log"
.string "resource closed"
.string "nanolang.shadow.offset.0"
.string "nanolang.shadow.offset.22"
.string "nanolang.shadow.offset.54"
.string "nano.local.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00\x0e\x00\x00\x00\x16\x00\x00\x00f"
.string "\x00\x00\x00\x00\x02\x00\x00\x00.\x00\x00\x006\x00\x00\x00f"
.string "\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x19\x00\x00\x00path"
.string "\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00.\x00\x00\x00_f"
.string "\x02\x00\x00\x00\x02\x00\x00\x00\x15\x00\x00\x00.\x00\x00\x00fd"
.string "\x03\x00\x00\x00\x00\x00\x00\x00\r\x00\x00\x00\x1f\x00\x00\x00f"
.string "nano.affine.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00"
.string "\x01\x00\x00\x00\x00\x00\x00\x00"
.string "\x02\x00\x00\x00\x00\x00\x00\x00"
.string "\x03\x00\x00\x00\x00\x00\x00\x00"

.metadata 8 3
.metadata 9 4
.metadata 10 5
.metadata 11 12
.metadata 11 13
.metadata 11 14
.metadata 11 15
.metadata 11 16
.metadata 11 17
.metadata 18 19
.metadata 18 20
.metadata 18 21
.metadata 18 22

.types 1 0 0

.layouts "01000000000001000000000001000000ffffffff01000000"

.ownership "010000000100000003000000040000000300000001000000ffffffff08000000"
.ownership "0000000001000000ffffffff0800000000000000020001000800000000000000"
.ownership "05000000ffffffff01000000ffffffff0300010000000000ffffffff08000000"
.ownership "00000000080000000000000001000000ffffffff0100000001000000ffffffff"
.ownership "0800000000000000"

.entry 0

.function __nanoisa_shadow_entry 0 3 0 int 1
  NOP
  PUSH_STR 6
  CALL 1
  OWN_STORE_LOCAL 0
  OWN_MOVE_LOCAL 0
  CALL 2
  NOP
  PUSH_I64 1
  STORE_LOCAL 1
  LOAD_LOCAL 1
  OWN_PACK 0
  OWN_STORE_LOCAL 2
  OWN_MOVE_LOCAL 2
  CALL 2
  NOP
  CALL 3
  PUSH_I64 0
  EQ
  ASSERT
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
  PUSH_STR 7
  PRINTLN
  OWN_UNPACK_LOCAL 1
  POP
  RET
.end

.parameters 2 struct

.function main 0 1 0 int 1
  PUSH_STR 6
  CALL 1
  OWN_STORE_LOCAL 0
  OWN_MOVE_LOCAL 0
  CALL 2
  PUSH_I64 0
  RET
.end

