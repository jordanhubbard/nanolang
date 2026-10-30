.string "Handle"
.string "payload"
.string "__nano_record_7269676874_Handle"
.string "main"
.string "take"
.string "run"
.string "$import.3.0"
.string "$import.4.0"
.string "nano.local.v1"
.string "\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x000\x00\x00\x00value"
.string "\x01\x00\x00\x00\x01\x00\x00\x00\x0f\x00\x00\x000\x00\x00\x00observed"
.string "\x01\x00\x00\x00\x03\x00\x00\x00$\x00\x00\x000\x00\x00\x00payload"
.string "\x02\x00\x00\x00\x01\x00\x00\x00\x17\x00\x00\x00 \x00\x00\x00value"
.string "\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x000\x00\x00\x00value"
.string "\x03\x00\x00\x00\x01\x00\x00\x00\x0f\x00\x00\x000\x00\x00\x00observed"
.string "\x03\x00\x00\x00\x03\x00\x00\x00$\x00\x00\x000\x00\x00\x00payload"
.string "\x04\x00\x00\x00\x01\x00\x00\x00\x17\x00\x00\x00 \x00\x00\x00value"
.string "nano.affine.v1"
.string "\x00\x00\x00\x00\x01\x00\x00\x00"
.string "\x01\x00\x00\x00\x01\x00\x00\x00"
.string "\x02\x00\x00\x00\x01\x00\x00\x00"
.string "\x03\x00\x00\x00\x01\x00\x00\x00"
.string "\x04\x00\x00\x00\x01\x00\x00\x00"
.string "nano.affine-use.v1"
.string "\x01\x00\x00\x00\x02\x00\x00\x00.\x00\x00\x00\x01\x01\x00\x005\x00\x00\x00\x05\x00\x00\x00/private/tmp/nanolang-imported-affine-repro/left.nanovalue"
.string "\x01\x00\x00\x00\x02\x00\x00\x00P\x00\x00\x00\x02\x01\x00\x005\x00\x00\x00\x05\x00\x00\x00/private/tmp/nanolang-imported-affine-repro/left.nanovalue"
.string "\x02\x00\x00\x00\x04\x00\x00\x00D\x00\x00\x00\x02\x01\x00\x005\x00\x00\x00\x05\x00\x00\x00/private/tmp/nanolang-imported-affine-repro/left.nanovalue"
.string "\x03\x00\x00\x00\x02\x00\x00\x00.\x00\x00\x00\x01\x01\x00\x006\x00\x00\x00\x05\x00\x00\x00/private/tmp/nanolang-imported-affine-repro/right.nanovalue"
.string "\x03\x00\x00\x00\x02\x00\x00\x00P\x00\x00\x00\x02\x01\x00\x006\x00\x00\x00\x05\x00\x00\x00/private/tmp/nanolang-imported-affine-repro/right.nanovalue"
.string "\x04\x00\x00\x00\x04\x00\x00\x00D\x00\x00\x00\x02\x01\x00\x006\x00\x00\x00\x05\x00\x00\x00/private/tmp/nanolang-imported-affine-repro/right.nanovalue"

.metadata 8 9
.metadata 8 10
.metadata 8 11
.metadata 8 12
.metadata 8 13
.metadata 8 14
.metadata 8 15
.metadata 8 16
.metadata 17 18
.metadata 17 19
.metadata 17 20
.metadata 17 21
.metadata 17 22
.metadata 23 24
.metadata 23 25
.metadata 23 26
.metadata 23 27
.metadata 23 28
.metadata 23 29

.types 2 0 0

.layouts "02000000000001000000000001000000ffffffff010000000000010002000000"
.layouts "01000000ffffffff01000000"

.ownership "010000000200000003030000050000000000000001000000ffffffff04000100"
.ownership "01000000ffffffff080000000000000001000000ffffffff0800000000000000"
.ownership "01000000ffffffff0200000001000000ffffffff01000000ffffffff08000000"
.ownership "000000000400010001000000ffffffff080000000100000001000000ffffffff"
.ownership "080000000100000001000000ffffffff0200000001000000ffffffff01000000"
.ownership "ffffffff0800000001000000"

.entry 0

.function main 0 0 0 int 1
  CALL 2
  PUSH_I64 14
  EQ
  ASSERT
  CALL 4
  PUSH_I64 18
  EQ
  ASSERT
  PUSH_I64 0
  RET
.end

.function take 1 4 0 int 1
  REGION_BEGIN
  BORROW_LOCAL_SHARED 1 0
  REF_GET 1 0
  REGION_END
  STORE_LOCAL 1
  OWN_MOVE_LOCAL 0
  OWN_STORE_LOCAL 2
  REGION_BEGIN
  BORROW_LOCAL_SHARED 1 2
  REF_GET 1 0
  REGION_END
  STORE_LOCAL 3
  LOAD_LOCAL 1
  LOAD_LOCAL 3
  ADD
  OWN_UNPACK_LOCAL 2
  POP
  RET
.end

.parameters 1 struct

.function run 0 2 0 int 1
  PUSH_I64 7
  STORE_LOCAL 0
  LOAD_LOCAL 0
  OWN_PACK 0
  OWN_STORE_LOCAL 1
  OWN_MOVE_LOCAL 1
  CALL 1
  RET
.end

.function $import.3.0 1 4 0 int 1
  REGION_BEGIN
  BORROW_LOCAL_SHARED 1 0
  REF_GET 1 0
  REGION_END
  STORE_LOCAL 1
  OWN_MOVE_LOCAL 0
  OWN_STORE_LOCAL 2
  REGION_BEGIN
  BORROW_LOCAL_SHARED 1 2
  REF_GET 1 0
  REGION_END
  STORE_LOCAL 3
  LOAD_LOCAL 1
  LOAD_LOCAL 3
  ADD
  OWN_UNPACK_LOCAL 2
  POP
  RET
.end

.parameters 3 struct

.function $import.4.0 0 2 0 int 1
  PUSH_I64 9
  STORE_LOCAL 0
  LOAD_LOCAL 0
  OWN_PACK 1
  OWN_STORE_LOCAL 1
  OWN_MOVE_LOCAL 1
  CALL 3
  RET
.end

