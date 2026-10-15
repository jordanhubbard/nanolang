.types 1 0 0
.entry 0
.string "Handle"
.string "fd"
.string "main"
.string "identity"
.string "read"
.layouts "01000000000001000000000001000000ffffffff01000000"
.ownership "030000000100000003000000030000000500000001000000ffffffff01000000ffffffff0800000000000000080000000000000001000000ffffffff01000000ffffffff0100010001000000ffffffff01000000ffffffff0000000001000000ffffffff040000000000000001000000030001000c0000000100000001010000ffffffff"
.function main 0 5 0 int 1
  PUSH_I64 4
  STORE_GLOBAL 0
  PUSH_I64 7
  STORE_LOCAL 0
  LOAD_LOCAL 0
  OWN_PACK 0
  OWN_STORE_LOCAL 1
  .local_begin 1 "local_owner"
  OWN_MOVE_LOCAL 1
  OWN_STORE_LOCAL 2
  REGION_BEGIN
  BORROW_LOCAL_SHARED 0 2
  REF_GET 0 0
  REGION_END
  STORE_LOCAL 3
  .local_begin 3 "fd"
  LOAD_LOCAL 3
  PUSH_I64 7
  EQ
  ASSERT
  PUSH_I64 9
  CALL 1
  PUSH_I64 9
  EQ
  ASSERT
  nb_control_0:
  PUSH_BOOL 1
  JMP_FALSE nb_control_1
  PUSH_I64 100
  STORE_LOCAL 4
  .local_begin 4 "counter"
  PUSH_I64 101
  STORE_LOCAL 4
  LOAD_LOCAL 4
  PUSH_I64 101
  EQ
  ASSERT
  .local_end 4
  JMP nb_control_2
  nb_control_1:
  nb_control_2:
  LOAD_GLOBAL 0
  PUSH_I64 4
  EQ
  ASSERT
  PUSH_I64 5
  STORE_GLOBAL 0
  CALL 2
  PUSH_I64 5
  EQ
  ASSERT
  PUSH_I64 0
  OWN_UNPACK_LOCAL 2
  POP
  RET
  .local_end 1
  .local_end 3
.end
.function identity 1 1 0 int 1
  .local_begin 0 "counter"
  LOAD_LOCAL 0
  RET
  .local_end 0
.end
.function read 0 0 0 int 1
  LOAD_GLOBAL 0
  RET
.end
.parameters 1 int
.parameters 2
