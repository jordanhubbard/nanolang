.string "main"
.string "nano.purity.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00"
.string "nano.affine.v1"
.string "nano.effects.v1"
.string "nano.local.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00\x19\x00\x00\x00>\x00\x00\x00entries"

.metadata 1 2
.metadata 3 2
.metadata 4 2
.metadata 5 6

.types 1 0 0

.entry 0

.function main 0 1 0 int 1
  PUSH_BOOL 0
  PUSH_BOOL 1
  ARR_LITERAL 4 2
  ARR_LITERAL 7 1
  AGG_PACK 0 0 0 1
  STORE_LOCAL 0
  LOAD_LOCAL 0
  AGG_GET 0
  PUSH_I64 0
  ARR_GET
  PUSH_I64 1
  ARR_GET
  ASSERT
  PUSH_I64 0
  RET
.end

