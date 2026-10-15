.types 0 0 1
.entry 4

.function __nanoisa_shadow_0 0 3 0 void 0
  CALL make_value
  STORE_LOCAL 0
  LOAD_LOCAL 0
  DUP
  AGG_TAG
  PUSH_I64 0
  EQ
  JMP_FALSE L1
  DUP
  STORE_LOCAL 1
  POP
  LOAD_LOCAL 1
  AGG_GET 0
  PUSH_I64 7
  I64_EQ
  ASSERT
  JMP L0
L1:
  DUP
  AGG_TAG
  PUSH_I64 1
  EQ
  JMP_FALSE L2
  DUP
  STORE_LOCAL 2
  POP
  PUSH_BOOL 0
  ASSERT
  JMP L0
L2:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
L0:
  RET
.end
.function __nanoisa_shadow_1 0 0 0 void 0
  PUSH_I64 0
  STORE_GLOBAL 0
  CALL expression
  PUSH_I64 7
  I64_EQ
  ASSERT
  LOAD_GLOBAL 0
  PUSH_I64 1
  I64_EQ
  ASSERT
  RET
.end
.function __nanoisa_shadow_2 0 0 0 void 0
  PUSH_I64 0
  STORE_GLOBAL 0
  CALL statement
  PUSH_I64 7
  I64_EQ
  ASSERT
  LOAD_GLOBAL 0
  PUSH_I64 1
  I64_EQ
  ASSERT
  RET
.end
.function __nanoisa_shadow_3 0 0 0 void 0
  CALL main
  PUSH_I64 0
  I64_EQ
  ASSERT
  RET
.end
.function __nanoisa_shadow_entry 0 0 0 int 1
  .shadow "make_value"
  CALL __nanoisa_shadow_0
  .shadow "expression"
  CALL __nanoisa_shadow_1
  .shadow "statement"
  CALL __nanoisa_shadow_2
  .shadow "main"
  CALL __nanoisa_shadow_3
  PUSH_I64 0
  RET
.end
.function make_value 0 1 0 union 1
  LOAD_GLOBAL 0
  PUSH_I64 1
  I64_ADD
  STORE_GLOBAL 0
  PUSH_I64 7
  STORE_LOCAL 0
  LOAD_LOCAL 0
  AGG_PACK 1 0 0 1
  RET
.end
.function expression 0 2 0 int 1
  CALL make_value
  DUP
  AGG_TAG
  PUSH_I64 0
  EQ
  JMP_FALSE L4
  DUP
  STORE_LOCAL 0
  POP
  LOAD_LOCAL 0
  AGG_GET 0
  JMP L3
L4:
  DUP
  AGG_TAG
  PUSH_I64 1
  EQ
  JMP_FALSE L5
  DUP
  STORE_LOCAL 1
  POP
  PUSH_I64 0
  JMP L3
L5:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
L3:
  RET
.end
.function statement 0 2 0 int 1
  CALL make_value
  DUP
  AGG_TAG
  PUSH_I64 0
  EQ
  JMP_FALSE L7
  DUP
  STORE_LOCAL 0
  POP
  LOAD_LOCAL 0
  AGG_GET 0
  RET
L7:
  DUP
  AGG_TAG
  PUSH_I64 1
  EQ
  JMP_FALSE L8
  DUP
  STORE_LOCAL 1
  POP
  PUSH_I64 0
  RET
L8:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
L6:
.end
.function main 0 0 0 int 1
  PUSH_I64 0
  STORE_GLOBAL 0
  CALL expression
  PUSH_I64 7
  I64_EQ
  ASSERT
  LOAD_GLOBAL 0
  PUSH_I64 1
  I64_EQ
  ASSERT
  PUSH_I64 0
  STORE_GLOBAL 0
  CALL statement
  PUSH_I64 7
  I64_EQ
  ASSERT
  LOAD_GLOBAL 0
  PUSH_I64 1
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

