.string "main"
.string "nano.purity.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00"
.string "nano.affine.v1"
.string "nano.effects.v1"
.string "\x00\x00\x00\x00\x08\x00\x00\x00"
.string "read"
.string "\x01\x00\x00\x00\x00\x00\x00\x00"
.string "\x01\x00\x00\x00\x08\x00\x00\x00"
.string "__init__"

.metadata 1 2
.metadata 3 2
.metadata 4 5
.metadata 1 7
.metadata 3 7
.metadata 4 8

.entry 0

.function main 0 0 0 int 1
  CALL 1
  PUSH_I64 7
  I64_SUB
  RET
.end

.function read 0 0 0 int 1
  LOAD_GLOBAL 0
  RET
.end

.function __init__ 0 0 0 void 0
  PUSH_I64 7
  STORE_GLOBAL 0
  RET
.end

