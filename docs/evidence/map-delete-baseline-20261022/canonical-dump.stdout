.string "key"
.string "main"
.string "nano.purity.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00"
.string "nano.affine.v1"
.string "nano.effects.v1"
.string "\x00\x00\x00\x00\x08\x00\x00\x00"
.string "exercise"
.string "\x01\x00\x00\x00\x00\x00\x00\x00"
.string "\x01\x00\x00\x00\x08\x00\x00\x00"
.string "nano.local.v1"
.string "\x01\x00\x00\x00\x00\x00\x00\x00\x10\x00\x00\x00\x81\x00\x00\x00h"

.metadata 2 3
.metadata 4 3
.metadata 5 6
.metadata 2 8
.metadata 4 8
.metadata 5 9
.metadata 10 11

.types 1 0 0

.entry 0

.function main 0 0 0 int 1
  TAIL_CALL 1
.end

.function exercise 0 1 0 int 1
  HM_NEW 5 1
  AGG_PACK 0 0 0 1
  STORE_LOCAL 0
  LOAD_LOCAL 0
  AGG_GET 0
  PUSH_STR 0
  PUSH_I64 42
  HM_SET
  POP
  LOAD_LOCAL 0
  AGG_GET 0
  PUSH_STR 0
  HM_HAS
  ASSERT
  LOAD_LOCAL 0
  AGG_GET 0
  PUSH_STR 0
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
  PUSH_STR 0
  HM_DELETE
  POP
  LOAD_LOCAL 0
  AGG_GET 0
  PUSH_STR 0
  HM_HAS
  BOOL_NOT
  ASSERT
  PUSH_I64 0
  RET
.end

