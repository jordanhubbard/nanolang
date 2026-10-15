.string "callee"
.string "argument"
.string "main"
.string "choose"
.string "nano.local.v1"
.string "\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x14\x00\x00\x00add"
.string "record_callee"
.string "record_argument"
.string "add_one"
.string "\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0e\x00\x00\x00value"
.string "double"
.string "\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0e\x00\x00\x00value"

.metadata 4 5
.metadata 4 9
.metadata 4 11

.entry 0

.function main 0 4 0 int 1
  PUSH_BOOL 1
  CALL 1
  STORE_LOCAL 0
  PUSH_I64 7
  LOAD_LOCAL 0
  CALL_INDIRECT 1 1
  PUSH_I64 8
  I64_EQ
  ASSERT
  PUSH_BOOL 0
  CALL 1
  STORE_LOCAL 1
  PUSH_BOOL 1
  CALL 1
  STORE_LOCAL 2
  PUSH_I64 4
  LOAD_LOCAL 2
  CALL_INDIRECT 1 1
  LOAD_LOCAL 1
  CALL_INDIRECT 1 1
  PUSH_I64 10
  I64_EQ
  ASSERT
  CALL 2
  STORE_LOCAL 3
  CALL 3
  LOAD_LOCAL 3
  CALL_INDIRECT 1 1
  PUSH_I64 10
  I64_EQ
  ASSERT
  PUSH_I64 0
  RET
.end

.function choose 1 1 0 function 1
  LOAD_LOCAL 0
  JMP_FALSE L0
  FUNCREF 4
  RET
L0:
  FUNCREF 5
  RET
.end

.parameters 1 bool

.function record_callee 0 0 0 function 1
  PUSH_STR 0
  PRINTLN
  FUNCREF 4
  RET
.end

.function record_argument 0 0 0 int 1
  PUSH_STR 1
  PRINTLN
  PUSH_I64 9
  RET
.end

.function add_one 1 1 0 int 1
  LOAD_LOCAL 0
  PUSH_I64 1
  I64_ADD
  RET
.end

.parameters 4 int

.function double 1 1 0 int 1
  LOAD_LOCAL 0
  PUSH_I64 2
  I64_MUL
  RET
.end

.parameters 5 int

