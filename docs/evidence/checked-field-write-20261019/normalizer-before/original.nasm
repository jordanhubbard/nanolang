.string "Counter"
.string "value"
.string "active"
.string "main"
.string "update"
.string "nano.local.v1"
.string "\x00\x00\x00\x00\x02\x00\x00\x00\x1f\x00\x00\x00\x8a\x00\x00\x00owner"
.string "\x00\x00\x00\x00\x04\x00\x00\x00Z\x00\x00\x00\x8a\x00\x00\x00value"
.string "\x00\x00\x00\x00\x05\x00\x00\x00i\x00\x00\x00\x8a\x00\x00\x00active"
.string "\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00%\x00\x00\x00view"
.string "nano.affine.v1"
.string "\x00\x00\x00\x00\x01\x00\x00\x00"
.string "\x01\x00\x00\x00\x01\x00\x00\x00"
.string "nano.affine-use.v1"
.string "\x00\x00\x00\x00\x12\x00\x00\x00\x1a\x00\x00\x00\x01\x01\x00\x00?\x00\x00\x00\x05\x00\x00\x00/private/tmp/nanolang-field-write-normalizer-before/source.nanoowner"
.string "\x00\x00\x00\x00\x13\x00\x00\x00\t\x00\x00\x00\x01\x01\x00\x00?\x00\x00\x00\x05\x00\x00\x00/private/tmp/nanolang-field-write-normalizer-before/source.nanoowner"
.string "\x00\x00\x00\x00\x14\x00\x00\x00\"\x00\x00\x00\x02\x01\x00\x00?\x00\x00\x00\x05\x00\x00\x00/private/tmp/nanolang-field-write-normalizer-before/source.nanoowner"
.string "\x01\x00\x00\x00\x03\x00\x00\x00\x14\x00\x00\x00\x01\x01\x00\x00?\x00\x00\x00\x04\x00\x00\x00/private/tmp/nanolang-field-write-normalizer-before/source.nanoview"
.string "\x01\x00\x00\x00\x04\x00\x00\x00\x17\x00\x00\x00\x01\x01\x00\x00?\x00\x00\x00\x04\x00\x00\x00/private/tmp/nanolang-field-write-normalizer-before/source.nanoview"
.string "\x01\x00\x00\x00\x05\x00\x00\x00\t\x00\x00\x00\x01\x01\x00\x00?\x00\x00\x00\x04\x00\x00\x00/private/tmp/nanolang-field-write-normalizer-before/source.nanoview"
.string "nano.borrow-place.v1"
.string "\x00\x00\x00\x00\x12\x00\x00\x00\r\x00\x00\x00\x12\x00\x00\x00\x15\x00\x00\x00\x00\x00\x00\x00\x02\x01\x00\x00?\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00/private/tmp/nanolang-field-write-normalizer-before/source.nanoowner"
.string "nano.field-read.v1"
.string "\x00\x00\x00\x00\x13\x00\x00\x00\x0e\x00\x00\x00\x01\x01\x00\x00?\x00\x00\x00\x05\x00\x00\x00\x07\x00\x00\x00/private/tmp/nanolang-field-write-normalizer-before/source.nanoowner.active"
.string "\x01\x00\x00\x00\x03\x00\x00\x00\x18\x00\x00\x00\x03\x01\x00\x00?\x00\x00\x00\x04\x00\x00\x00\x06\x00\x00\x00/private/tmp/nanolang-field-write-normalizer-before/source.nanoview.value"
.string "\x01\x00\x00\x00\x04\x00\x00\x00\x1b\x00\x00\x00\x03\x01\x00\x00?\x00\x00\x00\x04\x00\x00\x00\x07\x00\x00\x00/private/tmp/nanolang-field-write-normalizer-before/source.nanoview.active"
.string "\x01\x00\x00\x00\x05\x00\x00\x00\r\x00\x00\x00\x03\x01\x00\x00?\x00\x00\x00\x04\x00\x00\x00\x06\x00\x00\x00/private/tmp/nanolang-field-write-normalizer-before/source.nanoview.value"
.string "nano.field-write.v1"
.string "\x01\x00\x00\x00\x03\x00\x00\x00\x02\x00\x00\x00\x01\x01\x00\x00?\x00\x00\x00\x04\x00\x00\x00\x06\x00\x00\x00/private/tmp/nanolang-field-write-normalizer-before/source.nanoview.value"
.string "\x01\x00\x00\x00\x04\x00\x00\x00\x02\x00\x00\x00\x01\x01\x00\x00?\x00\x00\x00\x04\x00\x00\x00\x07\x00\x00\x00/private/tmp/nanolang-field-write-normalizer-before/source.nanoview.active"

.metadata 5 6
.metadata 5 7
.metadata 5 8
.metadata 5 9
.metadata 10 11
.metadata 10 12
.metadata 13 14
.metadata 13 15
.metadata 13 16
.metadata 13 17
.metadata 13 18
.metadata 13 19
.metadata 20 21
.metadata 22 23
.metadata 22 24
.metadata 22 25
.metadata 22 26
.metadata 27 28
.metadata 27 29

.types 1 0 0

.layouts "01000000000002000000000001000000ffffffff0100000004000000ffffffff"
.layouts "02000000"

.ownership "010000000100000003000000020000000600000001000000ffffffff01000000"
.ownership "ffffffff04000000ffffffff0800000000000000080000000000000001000000"
.ownership "ffffffff04000000ffffffff0100010001000000ffffffff0802000000000000"

.entry 0

.function main 0 6 0 int 1
  PUSH_I64 12
  STORE_LOCAL 0
  PUSH_BOOL 0
  STORE_LOCAL 1
  LOAD_LOCAL 0
  LOAD_LOCAL 1
  OWN_PACK 0
  OWN_STORE_LOCAL 2
  REGION_BEGIN
  BORROW_LOCAL_EXCLUSIVE 0 2
  CALL_REF 1 0
  REGION_END
  PUSH_I64 15
  EQ
  ASSERT
  REGION_BEGIN
  BORROW_LOCAL_SHARED 0 2
  REF_GET 0 1
  REGION_END
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
  PUSH_I64 15
  EQ
  ASSERT
  LOAD_LOCAL 5
  ASSERT
  PUSH_I64 0
  OWN_UNPACK_LOCAL 3
  POP
  POP
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

