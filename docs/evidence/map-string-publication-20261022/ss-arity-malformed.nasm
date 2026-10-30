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
.string "nano.local.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\x00\x00{\x00\x00\x00h"

.metadata 2 3
.metadata 4 3
.metadata 5 6
.metadata 2 9
.metadata 4 9
.metadata 5 10
.metadata 11 12

.types 1 0 0

.flag debug_info

.debug 0 3 2
.debug 16 4 2
.debug 35 5 2
.debug 48 6 2
.debug 67 7 2
.debug 85 8 2
.debug 99 9 2
.debug 113 10 2
.debug 123 13 20

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

