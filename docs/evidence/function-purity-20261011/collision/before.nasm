.string "value"
.string "main"
.string "nano.purity.v1"
.string "\x02\x00\x00\x00\x00\x00\x00\x00"
.string "\x00\x00\x00\x00\x01\x00\x00\x00"
.string "\x01\x00\x00\x00\x00\x00\x00\x00"
.string "right"
.string "/private/tmp/nl51-purity/collision/case.nano"
.string "nano.local.v1"
.string "\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\x00\x00\x00x"
.string "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00x"
.string "nano.source_file"

.metadata 2 3
.metadata 2 4
.metadata 2 5
.metadata 8 9
.metadata 8 10
.metadata 11 7

.flag debug_info

.debug 0 1 31
.debug 8 1 26
.debug 14 1 44
.debug 18 3 16

.entry 2

.function value 1 1 0 int 1
  LOAD_LOCAL 0
  LOAD_LOCAL 0
  I64_ADD
  RET
.end

.parameters 0 int

.function value 1 1 0 int 1
  PUSH_STR 6
  PRINTLN
  LOAD_LOCAL 0
  RET
.end

.parameters 1 int

.function main 0 0 0 int 1
  PUSH_I64 4
  CALL 0
  PUSH_I64 9
  CALL 1
  I64_ADD
  RET
.end

