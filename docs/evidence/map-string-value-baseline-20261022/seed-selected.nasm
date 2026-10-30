.string "exercise"
.string "main"
.string "nano.purity.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00"
.string "nano.affine.v1"
.string "nano.effects.v1"
.string "\x00\x00\x00\x00\x08\x00\x00\x00"
.string "key"
.string "kept"
.string "\x01\x00\x00\x00\x00\x00\x00\x00"
.string "\x01\x00\x00\x00\x08\x00\x00\x00"
.string "/private/tmp/nanolang-map-string-value-baseline-20261022/unchanged.nano"
.string "nano.local.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\x00\x00y\x00\x00\x00h"
.string "nano.source_file"

.metadata 2 3
.metadata 4 3
.metadata 5 6
.metadata 2 9
.metadata 4 9
.metadata 5 10
.metadata 12 13
.metadata 14 11

.types 1 0 0

.flag debug_info

.debug 0 3 2
.debug 16 4 2
.debug 34 5 2
.debug 47 6 2
.debug 66 7 2
.debug 84 8 2
.debug 97 9 2
.debug 111 10 2
.debug 121 13 20

.entry 1

.function exercise 0 1 0 int 1
  HM_NEW 5 5
  AGG_PACK 0 0 0 1
  STORE_LOCAL 0
  LOAD_LOCAL 0
  AGG_GET 0
  PUSH_STR 7
  PUSH_STR 8
  HM_SET
  POP
  LOAD_LOCAL 0
  AGG_GET 0
  PUSH_STR 7
  HM_HAS
  ASSERT
  LOAD_LOCAL 0
  AGG_GET 0
  PUSH_STR 7
  HM_GET
  PUSH_STR 8
  EQ
  ASSERT
  LOAD_LOCAL 0
  AGG_GET 0
  HM_LEN
  PUSH_I64 1
  I64_EQ
  ASSERT
  LOAD_LOCAL 0
  AGG_GET 0
  PUSH_STR 7
  HM_DELETE
  POP
  LOAD_LOCAL 0
  AGG_GET 0
  PUSH_STR 7
  HM_HAS
  BOOL_NOT
  ASSERT
  PUSH_I64 0
  RET
.end

.function main 0 0 0 int 1
  TAIL_CALL 0
.end


.layouts "0100000000000100ffffffff0d000000ffffffffffffffff"
.ownership "030000000100000001000000020000000100000001000000ffffffff00000000ffffffff0000000001000000ffffffff04000000000000000100000005000100280000000100000005000000ffffffff01000000000000000000000001000000000000000000000000000000"
