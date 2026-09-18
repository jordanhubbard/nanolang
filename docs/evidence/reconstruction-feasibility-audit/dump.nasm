.string "combine"
.string "main"
.string "/tmp/nano-feasibility-current-ho3ajwip/original.nano"
.string "nano.local.v1"
.string "\x01\x00\x00\x00\x01\x00\x00\x00!\x00\x00\x00e\x00\x00\x00result"
.string "\x01\x00\x00\x00\x00\x00\x00\x00\r\x00\x00\x00e\x00\x00\x00input"
.string "\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00right"
.string "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00left"
.string "nano.source_file"

.metadata 3 4
.metadata 3 5
.metadata 3 6
.metadata 3 7
.metadata 8 2

.flag debug_info

.debug 0 1 50
.debug 8 4 5
.debug 21 5 5
.debug 41 6 5
.debug 60 6 56
.debug 60 6 58
.debug 70 7 5
.debug 89 7 55
.debug 89 7 57
.debug 99 8 5

.entry 1

.function combine 2 2 0 float 1
  LOAD_LOCAL 0
  LOAD_LOCAL 1
  F64_ADD
  RET
.end

.parameters 0 float float

.function main 0 2 0 int 1
  PUSH_I64 9221120237041090602
  F64_FROM_BITS
  STORE_LOCAL 0
  LOAD_LOCAL 0
  PUSH_F64 bits:3ff0000000000000
  CALL 0
  STORE_LOCAL 1
  LOAD_LOCAL 1
  F64_TO_BITS
  PUSH_I64 9221120237041090560
  I64_NE
  JMP_FALSE L0
  PUSH_I64 1
  RET
L0:
  LOAD_LOCAL 0
  F64_TO_BITS
  PUSH_I64 9221120237041090602
  I64_NE
  JMP_FALSE L1
  PUSH_I64 2
  RET
L1:
  PUSH_I64 0
  RET
.end

