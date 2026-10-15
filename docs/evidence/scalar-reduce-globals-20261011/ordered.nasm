.string "source"
.string "initial"
.string "combine"
.string "callback"
.string "main"
.string "__init__"
.string "/var/folders/9z/xpmfgw8j09l4g6wwxrxtt97w0000gn/T/nano-reduce-source-keyh0jh8/ordinary.nano"
.string "nano.local.v1"
.string "\x04\x00\x00\x00\x05\x00\x00\x00\x87\x00\x00\x00\xc1\x00\x00\x00result"
.string "\x02\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00&\x00\x00\x00b"
.string "\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00&\x00\x00\x00a"
.string "nano.source_file"

.metadata 7 8
.metadata 7 9
.metadata 7 10
.metadata 11 6

.flag debug_info

.debug 24 3 27
.debug 54 3 58
.debug 60 5 21
.debug 90 5 52
.debug 110 5 87
.debug 120 7 36
.debug 150 7 67
.debug 158 9 39
.debug 188 9 70
.debug 194 12 2
.debug 208 13 2
.debug 226 14 2
.debug 329 15 2
.debug 344 16 2
.debug 360 17 2
.debug 377 18 2

.entry 4

.function source 0 0 0 array 1
  LOAD_GLOBAL 0
  PUSH_I64 10
  I64_MUL
  PUSH_I64 1
  I64_ADD
  STORE_GLOBAL 0
  LOAD_GLOBAL 1
  RET
.end

.function initial 0 0 0 float 1
  LOAD_GLOBAL 0
  PUSH_I64 10
  I64_MUL
  PUSH_I64 2
  I64_ADD
  STORE_GLOBAL 0
  LOAD_GLOBAL 1
  PUSH_F64 bits:4008000000000000
  ARR_PUSH
  STORE_GLOBAL 1
  PUSH_F64 bits:4010000000000000
  RET
.end

.function combine 2 2 0 float 1
  LOAD_GLOBAL 0
  PUSH_I64 10
  I64_MUL
  PUSH_I64 4
  I64_ADD
  STORE_GLOBAL 0
  LOAD_LOCAL 0
  LOAD_LOCAL 1
  F64_ADD
  RET
.end

.parameters 2 float float

.function callback 0 0 0 function 1
  LOAD_GLOBAL 0
  PUSH_I64 10
  I64_MUL
  PUSH_I64 3
  I64_ADD
  STORE_GLOBAL 0
  FUNCREF 2
  RET
.end

.function main 0 6 0 int 1
  PUSH_I64 0
  STORE_GLOBAL 0
  PUSH_F64 bits:4000000000000000
  ARR_LITERAL 3 1
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
  STORE_LOCAL 5
  LOAD_LOCAL 5
  F64_TO_BITS
  PUSH_I64 4621256167635550208
  I64_EQ
  ASSERT
  LOAD_GLOBAL 0
  PUSH_I64 12344
  I64_EQ
  ASSERT
  LOAD_GLOBAL 1
  ARR_LEN
  PUSH_I64 2
  I64_EQ
  ASSERT
  PUSH_I64 0
  RET
.end

.function __init__ 0 0 0 void 0
  PUSH_I64 0
  STORE_GLOBAL 0
  ARR_LITERAL 3 0
  STORE_GLOBAL 1
  RET
.end

