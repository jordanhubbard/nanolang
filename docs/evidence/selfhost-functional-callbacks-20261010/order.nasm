.string "source"
.string "initial"
.string "combine"
.string "callback"
.string "main"
.string "__init__"
.string "/private/tmp/nl51-selfhost-functional/test_callback_expression_order_and_length_snapshot.nano"
.string "nano.local.v1"
.string "\x02\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00b"
.string "\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00a"
.string "nano.source_file"

.metadata 7 8
.metadata 7 9
.metadata 10 6

.flag debug_info

.debug 42 3 25
.debug 72 3 54
.debug 78 5 19
.debug 108 5 48
.debug 118 7 30
.debug 126 10 1
.debug 156 10 30
.debug 172 10 52
.debug 178 13 1
.debug 192 14 1
.debug 303 15 1
.debug 319 16 1

.entry 4

.function source 0 0 0 array 1
  LOAD_GLOBAL 1
  PUSH_I64 10
  I64_MUL
  PUSH_I64 1
  I64_ADD
  STORE_GLOBAL 1
  LOAD_GLOBAL 0
  RET
.end

.function initial 0 0 0 int 1
  LOAD_GLOBAL 1
  PUSH_I64 10
  I64_MUL
  PUSH_I64 2
  I64_ADD
  STORE_GLOBAL 1
  PUSH_I64 10
  RET
.end

.function combine 2 2 0 int 1
  LOAD_LOCAL 0
  LOAD_LOCAL 1
  I64_ADD
  RET
.end

.parameters 2 int int

.function callback 0 0 0 function 1
  LOAD_GLOBAL 1
  PUSH_I64 10
  I64_MUL
  PUSH_I64 3
  I64_ADD
  STORE_GLOBAL 1
  LOAD_GLOBAL 0
  PUSH_I64 4
  ARR_PUSH
  POP
  FUNCREF 2
  RET
.end

.function main 0 5 0 int 1
  PUSH_I64 0
  STORE_GLOBAL 1
  CALL 0
  STORE_LOCAL 0
  CALL 1
  STORE_LOCAL 1
  CALL 3
  STORE_LOCAL 2
  LOAD_LOCAL 0
  ARR_LEN
  STORE_LOCAL 3
  PUSH_I64 0
  STORE_LOCAL 4
L1:
  LOAD_LOCAL 4
  LOAD_LOCAL 3
  LT
  JMP_FALSE L0
  LOAD_LOCAL 1
  LOAD_LOCAL 0
  LOAD_LOCAL 4
  ARR_GET
  LOAD_LOCAL 2
  CALL_INDIRECT 2 1
  STORE_LOCAL 1
  LOAD_LOCAL 4
  PUSH_I64 1
  I64_ADD
  STORE_LOCAL 4
  JMP L1
L0:
  LOAD_LOCAL 1
  PUSH_I64 19
  I64_EQ
  ASSERT
  LOAD_GLOBAL 1
  PUSH_I64 123
  I64_EQ
  ASSERT
  PUSH_I64 0
  RET
.end

.function __init__ 0 0 0 void 0
  PUSH_I64 2
  PUSH_I64 3
  ARR_LITERAL 1 2
  STORE_GLOBAL 0
  PUSH_I64 0
  STORE_GLOBAL 1
  RET
.end
