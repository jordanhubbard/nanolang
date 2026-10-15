.types 0 0 0
.entry 0

.function main 0 0 0 int 1
  LOAD_GLOBAL 0
  PUSH_I64 0
  ARR_GET
  CAST_INT
  PUSH_I64 44
  I64_EQ
  ASSERT
  PUSH_I64 0
  RET
.end

.function __init__ 0 0 0 void 0
  PUSH_I64 300
  ARR_LITERAL 2 1
  STORE_GLOBAL 0
  RET
.end

