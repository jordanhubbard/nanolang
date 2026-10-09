.types 0 0 0
.entry 0

.string s0 " Choice<int>"
.string s1 "Choice"

.function main 0 2 0 int 1
  PUSH_STR s0
  STORE_LOCAL 0
.local_begin 0 "spelling"
  LOAD_LOCAL 0
  PUSH_I64 0
  PUSH_I64 7
  STR_SUBSTR
  STR_TRIM
  ARR_LITERAL 5 1
  STORE_LOCAL 1
  LOAD_LOCAL 1
  PUSH_I64 0
  ARR_GET
  PUSH_STR s1
  EQ
  ASSERT
  PUSH_I64 0
  RET
.local_end 0
.end

