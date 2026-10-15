.string "main"
.string "read"
.string "nano.local.v1"
.string "\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa7\x00\x00\x00box"
.string "\x01\x00\x00\x00\x01\x00\x00\x00\x0c\x00\x00\x00\xa7\x00\x00\x00item"
.string "\x01\x00\x00\x00\x05\x00\x00\x00\x95\x00\x00\x00\xa7\x00\x00\x00result"

.metadata 2 3
.metadata 2 4
.metadata 2 5

.types 0 0 1

.entry 0

.function main 0 1 0 int 1
  PUSH_I64 7
  STORE_LOCAL 0
  LOAD_LOCAL 0
  AGG_PACK 1 0 0 1
  CALL 1
  PUSH_I64 7
  I64_EQ
  ASSERT
  AGG_PACK 1 0 1 0
  CALL 1
  PUSH_I64 1
  I64_EQ
  ASSERT
  PUSH_I64 0
  RET
.end

.function read 1 6 0 int 1
  PUSH_I64 99
  STORE_LOCAL 1
  LOAD_LOCAL 0
  DUP
  AGG_TAG
  PUSH_I64 0
  EQ
  JMP_FALSE L0
  DUP
  STORE_LOCAL 2
  LOAD_LOCAL 2
  AGG_GET 0
  PUSH_I64 0
  I64_LT_S
  JMP_FALSE L0
  POP
  PUSH_I64 0
  JMP L1
L0:
  DUP
  AGG_TAG
  PUSH_I64 0
  EQ
  JMP_FALSE L2
  DUP
  STORE_LOCAL 3
  POP
  LOAD_LOCAL 3
  AGG_GET 0
  JMP L1
L2:
  DUP
  AGG_TAG
  PUSH_I64 1
  EQ
  JMP_FALSE L3
  DUP
  STORE_LOCAL 4
  POP
  PUSH_I64 1
  JMP L1
L3:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
L1:
  STORE_LOCAL 5
  LOAD_LOCAL 1
  PUSH_I64 99
  I64_EQ
  ASSERT
  LOAD_LOCAL 5
  RET
.end

.parameters 1 void

