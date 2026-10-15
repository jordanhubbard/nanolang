.string "read"
.string "main"
.string "/private/tmp/nl51-pattern-names/match.nano"
.string "nano.local.v1"
.string "\x00\x00\x00\x00\x04\x00\x00\x00\x91\x00\x00\x00\xa3\x00\x00\x00result"
.string "\x00\x00\x00\x00\x03\x00\x00\x00]\x00\x00\x00d\x00\x00\x00item"
.string "\x00\x00\x00\x00\x02\x00\x00\x00$\x00\x00\x00C\x00\x00\x00item"
.string "\x00\x00\x00\x00\x01\x00\x00\x00\x0c\x00\x00\x00\xa3\x00\x00\x00item"
.string "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa3\x00\x00\x00box"
.string "nano.source_file"

.metadata 9 2

.types 0 0 1

.flag debug_info

.debug 0 3 2
.debug 12 4 2
.debug 145 9 2
.debug 159 10 2
.debug 163 13 17
.debug 204 13 57
.debug 230 13 90

.entry 1

.function read 1 5 0 int 1
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
  POP
  PUSH_I64 1
  JMP L1
L3:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
L1:
  STORE_LOCAL 4
  LOAD_LOCAL 1
  PUSH_I64 99
  I64_EQ
  ASSERT
  LOAD_LOCAL 4
  RET
.end

.parameters 0 union

.function main 0 1 0 int 1
  PUSH_I64 7
  STORE_LOCAL 0
  LOAD_LOCAL 0
  AGG_PACK 1 0 0 1
  CALL 0
  PUSH_I64 7
  I64_EQ
  ASSERT
  AGG_PACK 1 0 1 0
  CALL 0
  PUSH_I64 1
  I64_EQ
  ASSERT
  PUSH_I64 0
  RET
.end

