.types 0 0 0
.entry 0
.function main 0 0 0 int 1
  PUSH_I64 2
  CALL probe
  PUSH_I64 7
  I64_SUB
  RET
.end
.function probe 1 1 0 int 1
  LOAD_LOCAL 0
  DUP
  PUSH_I64 1
  EQ
  JMP_FALSE L1
  PUSH_BOOL 1
  JMP_FALSE L1
  POP
  PUSH_I64 7
  JMP L0
L1:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
L0:
  RET
.end
