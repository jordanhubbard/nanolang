.string "key"
.string "__nanoisa_shadow_0"
.string "nano.purity.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00"
.string "nano.affine.v1"
.string "nano.effects.v1"
.string "\x00\x00\x00\x00\x08\x00\x00\x00"
.string "__nanoisa_shadow_1"
.string "\x01\x00\x00\x00\x00\x00\x00\x00"
.string "\x01\x00\x00\x00\x08\x00\x00\x00"
.string "__nanoisa_shadow_entry"
.string "nanolang.shadow.offset.34"
.string "exercise"
.string "nanolang.shadow.offset.40"
.string "main"
.string "\x03\x00\x00\x00\x00\x00\x00\x00"
.string "\x03\x00\x00\x00\x08\x00\x00\x00"
.string "nano.local.v1"
.string "\x03\x00\x00\x00\x00\x00\x00\x00\x10\x00\x00\x00\x81\x00\x00\x00h"
.string "\x04\x00\x00\x00\x00\x00\x00\x00"
.string "\x04\x00\x00\x00\x08\x00\x00\x00"

.metadata 2 3
.metadata 4 3
.metadata 5 6
.metadata 2 8
.metadata 4 8
.metadata 5 9
.metadata 11 12
.metadata 13 14
.metadata 2 15
.metadata 4 15
.metadata 5 16
.metadata 17 18
.metadata 2 19
.metadata 4 19
.metadata 5 20

.types 1 0 0

.entry 2

.function __nanoisa_shadow_0 0 0 0 void 0
  CALL 3
  PUSH_I64 0
  I64_EQ
  ASSERT
  RET
.end

.function __nanoisa_shadow_1 0 0 0 void 0
  CALL 4
  PUSH_I64 0
  I64_EQ
  ASSERT
  RET
.end

.function __nanoisa_shadow_entry 0 0 0 int 1
  NOP
  CALL 0
  NOP
  CALL 1
  PUSH_I64 0
  RET
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
  HM_HAS
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

.function main 0 0 0 int 1
  TAIL_CALL 3
.end


.layouts "0100000000000100ffffffff0d000000ffffffffffffffff"
.ownership "030000000100000001000000050000000000000000000000ffffffff0000000000000000ffffffff0000000001000000ffffffff0100000001000000ffffffff00000000ffffffff0000000001000000ffffffff04000000000000000100000005000100300000000200000001000000ffffffff05000000ffffffff01000000010000000000000001000000000000000000000000000000"
