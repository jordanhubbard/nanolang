.string "main"
.string "read"
.string "choice_size"

.types 1 0 1

.entry 0

.function main 0 4 0 int 1
  ARR_NEW 1
  STORE_LOCAL 0
  LOAD_LOCAL 0
  PUSH_I64 5
  AGG_PACK 0 0 0 2
  STORE_LOCAL 1
  LOAD_LOCAL 1
  AGG_GET 0
  PUSH_I64 7
  ARR_PUSH
  POP
  LOAD_LOCAL 1
  AGG_GET 0
  PUSH_I64 9
  ARR_PUSH
  POP
  LOAD_LOCAL 1
  CALL 1
  PUSH_I64 9
  I64_EQ
  ASSERT
  LOAD_LOCAL 0
  ARR_LEN
  PUSH_I64 2
  I64_EQ
  ASSERT
  LOAD_LOCAL 1
  STORE_LOCAL 2
  LOAD_LOCAL 2
  AGG_PACK 1 0 0 1
  STORE_LOCAL 3
  LOAD_LOCAL 3
  CALL 2
  PUSH_I64 7
  I64_EQ
  ASSERT
  PUSH_I64 0
  RET
.end

.function read 1 1 0 int 1
  LOAD_LOCAL 0
  AGG_GET 0
  PUSH_I64 1
  ARR_GET
  RET
.end

.parameters 1 void

.function choice_size 1 2 0 int 1
  LOAD_LOCAL 0
  DUP
  AGG_TAG
  PUSH_I64 0
  EQ
  JMP_FALSE L0
  DUP
  STORE_LOCAL 1
  POP
  LOAD_LOCAL 1
  AGG_GET 0
  AGG_GET 0
  ARR_LEN
  LOAD_LOCAL 1
  AGG_GET 0
  AGG_GET 1
  I64_ADD
  RET
L0:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
.end

.parameters 2 void

