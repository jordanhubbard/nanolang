.types 0 0 0
.entry 0

.function main 0 0 0 int 1
  CALL bytes
  PUSH_I64 0
  ARR_GET
  CAST_INT
  PUSH_I64 44
  I64_EQ
  ASSERT
  PUSH_I64 0
  RET
.end

.function bytes 0 0 0 array 1
  PUSH_I64 300
  ARR_LITERAL 2 1
  RET
.end

