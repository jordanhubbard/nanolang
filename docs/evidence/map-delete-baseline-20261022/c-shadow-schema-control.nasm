.string "exercise"
.string "main"
.string "nano.purity.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00"
.string "nano.affine.v1"
.string "nano.effects.v1"
.string "\x00\x00\x00\x00\x08\x00\x00\x00"
.string "key"
.string "\x01\x00\x00\x00\x00\x00\x00\x00"
.string "\x01\x00\x00\x00\x08\x00\x00\x00"
.string "$shadow_0_exercise"
.string "\x02\x00\x00\x00\x00\x00\x00\x00"
.string "\x02\x00\x00\x00\x08\x00\x00\x00"
.string "$shadow_1_main"
.string "\x03\x00\x00\x00\x00\x00\x00\x00"
.string "\x03\x00\x00\x00\x08\x00\x00\x00"
.string "$shadow_entry"
.string "/private/tmp/nanolang-map-delete-baseline-20261022/integer.nano"
.string "nano.local.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\x00\x00\x81\x00\x00\x00h"

.metadata 2 3
.metadata 4 3
.metadata 5 6
.metadata 2 8
.metadata 4 8
.metadata 5 9
.metadata 2 11
.metadata 4 11
.metadata 5 12
.metadata 2 14
.metadata 4 14
.metadata 5 15
.metadata 18 19

.types 1 0 0

.flag debug_info

.debug 0 3 2
.debug 16 4 2
.debug 38 5 2
.debug 51 6 2
.debug 74 7 2
.debug 92 8 2
.debug 105 9 2
.debug 119 10 2
.debug 129 13 20
.debug 134 12 19
.debug 151 14 15

.entry 4

.function exercise 0 1 0 int 1
  HM_NEW 5 1
  AGG_PACK 0 0 0 1
  STORE_LOCAL 0
  LOAD_LOCAL 0
  AGG_GET 0
  PUSH_STR 7
  PUSH_I64 42
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
  PUSH_I64 42
  I64_EQ
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
  HM_HAS
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

.function $shadow_0_exercise 0 0 0 void 0
  CALL 0
  PUSH_I64 0
  I64_EQ
  ASSERT
  RET
.end

.function $shadow_1_main 0 0 0 void 0
  CALL 1
  PUSH_I64 0
  I64_EQ
  ASSERT
  RET
.end

.function $shadow_entry 0 0 0 int 1
  CALL 2
  CALL 3
  PUSH_I64 0
  RET
.end


.layouts "0100000000000100ffffffff0d000000ffffffffffffffff"
.ownership "030000000100000001000000050000000100000001000000ffffffff00000000ffffffff0000000001000000ffffffff0000000000000000ffffffff0000000000000000ffffffff0000000001000000ffffffff04000000000000000100000005000100300000000200000001000000ffffffff05000000ffffffff01000000010000000000000001000000000000000000000000000000"
