.string "receiver"
.string "value"
.string "position"
.string "main"
.string "__init__"
.string "/Users/jkh/nanolang-qualification/byte-array-879-puck/ordinary/original-mutation/source.nano"
.string "nano.source_file"

.metadata 6 5

.flag debug_info

.debug 15 2 30
.debug 45 2 59
.debug 54 2 82
.debug 58 4 21
.debug 88 4 50
.debug 98 6 24
.debug 128 6 53
.debug 138 9 2
.debug 152 10 2
.debug 167 11 2
.debug 183 12 2
.debug 198 13 2
.debug 223 14 2
.debug 237 15 2
.debug 253 16 2
.debug 269 17 2
.debug 294 18 2

.entry 3

.function receiver 0 1 0 array 1
  LOAD_GLOBAL 0
  PUSH_I64 10
  I64_MUL
  PUSH_I64 1
  I64_ADD
  STORE_GLOBAL 0
  PUSH_U8 1
  ARR_LITERAL 2 1
  STORE_LOCAL 0
  LOAD_LOCAL 0
  RET
.end

.function value 0 0 0 int 1
  LOAD_GLOBAL 0
  PUSH_I64 10
  I64_MUL
  PUSH_I64 2
  I64_ADD
  STORE_GLOBAL 0
  PUSH_I64 258
  RET
.end

.function position 0 0 0 int 1
  LOAD_GLOBAL 0
  PUSH_I64 10
  I64_MUL
  PUSH_I64 3
  I64_ADD
  STORE_GLOBAL 0
  PUSH_I64 0
  RET
.end

.function main 0 1 0 int 1
  PUSH_I64 0
  STORE_GLOBAL 0
  CALL 0
  CALL 1
  CAST_U8
  ARR_PUSH
  STORE_LOCAL 0
  LOAD_GLOBAL 0
  PUSH_I64 12
  I64_EQ
  ASSERT
  LOAD_LOCAL 0
  ARR_LEN
  PUSH_I64 2
  I64_EQ
  ASSERT
  LOAD_LOCAL 0
  PUSH_I64 1
  ARR_GET
  CAST_INT
  PUSH_I64 2
  I64_EQ
  ASSERT
  PUSH_I64 0
  STORE_GLOBAL 0
  LOAD_LOCAL 0
  CALL 2
  CALL 1
  CAST_U8
  ARR_SET
  POP
  LOAD_GLOBAL 0
  PUSH_I64 32
  I64_EQ
  ASSERT
  LOAD_LOCAL 0
  PUSH_I64 0
  ARR_GET
  CAST_INT
  PUSH_I64 2
  I64_EQ
  ASSERT
  PUSH_I64 0
  RET
.end

.function __init__ 0 0 0 void 0
  PUSH_I64 0
  STORE_GLOBAL 0
  RET
.end

