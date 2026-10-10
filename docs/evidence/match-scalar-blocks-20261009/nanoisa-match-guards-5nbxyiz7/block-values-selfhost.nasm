.types 0 0 1
.entry 0

.function main 0 3 0 int 1
  PUSH_I64 0
  STORE_GLOBAL 0
  CALL fresh
  PUSH_I64 2
  PUSH_BOOL 0
  CALL gate
  JMP_FALSE L1
  POP
  PUSH_I64 99
  JMP L0
L1:
  DUP
  AGG_TAG
  PUSH_I64 0
  EQ
  JMP_FALSE L2
  DUP
  STORE_LOCAL 0
  PUSH_I64 3
  PUSH_BOOL 1
  CALL gate
  JMP_FALSE L2
  POP
  LOAD_LOCAL 0
  AGG_GET 0
  STORE_LOCAL 1
.local_begin 1 "local"
  LOAD_LOCAL 1
  PUSH_I64 1
  I64_ADD
.local_end 1
  JMP L0
L2:
  POP
  PUSH_I64 0
  JMP L0
L3:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
L0:
  STORE_LOCAL 2
.local_begin 2 "value"
  LOAD_LOCAL 2
  PUSH_I64 8
  I64_EQ
  ASSERT
  LOAD_GLOBAL 0
  PUSH_I64 123
  I64_EQ
  ASSERT
  PUSH_I64 1
  CALL choose
  PUSH_I64 7
  I64_EQ
  ASSERT
  PUSH_I64 2
  CALL choose
  PUSH_I64 7
  I64_EQ
  ASSERT
  PUSH_I64 0
  RET
.local_end 2
.end

.function fresh 0 2 0 union 1
  PUSH_I64 1
  CALL mark
  STORE_LOCAL 0
.local_begin 0 "ignored"
  PUSH_I64 7
  STORE_LOCAL 1
  LOAD_LOCAL 1
  AGG_PACK 1 0 0 1
  RET
.local_end 0
.end

.function gate 2 3 0 bool 1
.local_begin 0 "n"
.local_begin 1 "answer"
  LOAD_LOCAL 0
  CALL mark
  STORE_LOCAL 2
.local_begin 2 "ignored"
  LOAD_LOCAL 1
  RET
.local_end 2
.end

.function choose 1 6 0 int 1
.local_begin 0 "n"
  PUSH_I64 40
  STORE_LOCAL 1
.local_begin 1 "local"
  LOAD_LOCAL 0
  DUP
  PUSH_I64 1
  EQ
  JMP_FALSE L5
  POP
  LOAD_LOCAL 0
  PUSH_I64 0
  I64_GT_S
  JMP_FALSE L6
  PUSH_I64 7
  RET
L6:
  PUSH_I64 8
  STORE_LOCAL 2
.local_begin 2 "local"
  LOAD_LOCAL 2
  PUSH_I64 1
  I64_ADD
.local_end 2
  JMP L4
L5:
  POP
  LOAD_LOCAL 0
  PUSH_I64 2
  I64_ADD
  STORE_LOCAL 3
.local_begin 3 "local"
  LOAD_LOCAL 3
  PUSH_I64 3
  I64_ADD
  STORE_LOCAL 4
.local_begin 4 "next"
  LOAD_LOCAL 4
.local_end 3
.local_end 4
  JMP L4
L7:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
L4:
  STORE_LOCAL 5
.local_begin 5 "result"
  LOAD_LOCAL 1
  PUSH_I64 40
  I64_EQ
  ASSERT
  LOAD_LOCAL 5
  RET
.local_end 1
.local_end 5
.end

.function mark 1 1 0 int 1
.local_begin 0 "n"
  LOAD_GLOBAL 0
  PUSH_I64 10
  I64_MUL
  LOAD_LOCAL 0
  I64_ADD
  STORE_GLOBAL 0
  LOAD_LOCAL 0
  RET
.end

.function __init__ 0 0 0 void 0
  PUSH_I64 0
  STORE_GLOBAL 0
  RET
.end

