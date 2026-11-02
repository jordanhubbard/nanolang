.string "read"
.string "main"
.string "__init__"
.string "nano.purity.v1"
.string "\x00\x00\x00\x00\x00\x00\x00\x00"
.string "nano.affine.v1"
.string "nano.effects.v1"
.string "\x00\x00\x00\x00\x02\x00\x00\x00"
.string "\x01\x00\x00\x00\x00\x00\x00\x00"
.string "\x01\x00\x00\x00\x02\x00\x00\x00"
.string "/private/tmp/nanolang-effects-global-map-20261019/mutable.nano"
.string "nano.source_file"

.metadata 3 4
.metadata 5 4
.metadata 6 7
.metadata 3 8
.metadata 5 8
.metadata 6 9
.metadata 11 10

.flag debug_info

.debug 0 2 16
.debug 6 4 16

.entry 1

.function read 0 0 0 int 1
  LOAD_GLOBAL 0
  RET
.end

.function main 0 0 0 int 1
  CALL 0
  PUSH_I64 7
  I64_SUB
  RET
.end

.function __init__ 0 0 0 void 0
  PUSH_I64 7
  STORE_GLOBAL 0
  RET
.end
