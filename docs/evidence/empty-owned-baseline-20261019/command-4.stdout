.string "Handle"
.string "fd"
.string "Empty"
.string "main"
.string "close"
.string "nano.local.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00\x1b\x00\x00\x00value"
.string "\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x00\x00\x00owned"
.string "\x01\x00\x00\x00\x02\x00\x00\x00\x15\x00\x00\x00\x1d\x00\x00\x00fd"
.string "nano.affine.v1"
.string "\x00\x00\x00\x00\x01\x00\x00\x00"
.string "\x01\x00\x00\x00\x01\x00\x00\x00"
.string "nano.affine-use.v1"
.string "\x00\x00\x00\x00\x05\x00\x00\x00?\x00\x00\x00\x02\x01\x00\x008\x00\x00\x00\x05\x00\x00\x00/private/tmp/empty-owned-baseline-20261019/original.nanovalue"
.string "\x01\x00\x00\x00\x02\x00\x00\x006\x00\x00\x00\x02\x01\x00\x008\x00\x00\x00\x05\x00\x00\x00/private/tmp/empty-owned-baseline-20261019/original.nanoowned"

.metadata 5 6
.metadata 5 7
.metadata 5 8
.metadata 9 10
.metadata 9 11
.metadata 12 13
.metadata 12 14

.types 2 0 0

.layouts "02000000000001000000000001000000ffffffff010000000000000002000000"

.ownership "010000000200000003030000020000000200000001000000ffffffff08000000"
.ownership "0100000008000000010000000300010001000000ffffffff0800000000000000"
.ownership "080000000000000001000000ffffffff"

.entry 0

.function main 0 2 0 int 1
  OWN_PACK 1
  OWN_STORE_LOCAL 0
  OWN_MOVE_LOCAL 0
  OWN_STORE_LOCAL 1
  PUSH_I64 0
  OWN_UNPACK_LOCAL 1
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

