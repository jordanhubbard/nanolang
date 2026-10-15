.types 1 0 1
.entry 0
.string "Handle"
.string "fd"
.string "Result<Handle,string>"
.string "value"
.string "error"
.string "main"
.string "close_handle"
.string "consume"
.string "empty"
.string "Ok"
.string "Err"
.layouts "02000000000001000000000001000000ffffffff01000000020002000200000008000000000000000300000005000000ffffffff04000000"
.ownership "030000000200000003030000030000000300000001000000ffffffff01000000ffffffff080000000000000008000000000000000300010001000000ffffffff0800000000000000080000000000000001000000ffffffff0700010001000000ffffffff0a000000010000000a00000001000000080000000000000008000000000000000a0000000100000005000000ffffffff05000000ffffffff040000000000000001000000010001001c00000001000000010000000200000009000000000001000a00000001000100"
.function main 0 3 0 int 1
  PUSH_I64 9
  STORE_LOCAL 0
  LOAD_LOCAL 0
  OWN_PACK 0
  OWN_STORE_LOCAL 1
  .local_begin 1 "owner"
  OWN_MOVE_LOCAL 1
  OWN_STORE_LOCAL 2
  OWN_MOVE_LOCAL 2
  AGG_PACK 1 0 0 1
  CALL 2
  PUSH_I64 9
  SUB
  RET
  .local_end 1
.end
.function close_handle 1 3 0 int 1
  .local_begin 0 "owner"
  OWN_MOVE_LOCAL 0
  OWN_STORE_LOCAL 1
  REGION_BEGIN
  BORROW_LOCAL_SHARED 1 1
  REF_GET 1 0
  REGION_END
  STORE_LOCAL 2
  .local_begin 2 "fd"
  LOAD_LOCAL 2
  OWN_UNPACK_LOCAL 1
  POP
  RET
  .local_end 0
  .local_end 2
.end
.function consume 1 7 0 int 1
  .local_begin 0 "result"
  OWN_MOVE_LOCAL 0
  MATCH_TAG 0 nb_control_1
  JMP nb_control_2
  nb_control_1:
  OWN_STORE_LOCAL 1
  .local_begin 1 "payload"
  OWN_UNPACK_LOCAL 1
  OWN_STORE_LOCAL 2
  OWN_MOVE_LOCAL 2
  OWN_STORE_LOCAL 3
  .local_begin 3 "value"
  OWN_MOVE_LOCAL 3
  CALL 1
  RET
  .local_end 3
  .local_end 1
  nb_control_2:
  MATCH_TAG 1 nb_control_3
  JMP nb_control_4
  nb_control_3:
  OWN_STORE_LOCAL 4
  OWN_UNPACK_LOCAL 4
  STORE_LOCAL 5
  .local_begin 4 "payload"
  LOAD_LOCAL 5
  STORE_LOCAL 6
  .local_begin 6 "error"
  LOAD_LOCAL 6
  PUSH_STR 8
  EQ
  ASSERT
  PUSH_I64 0
  RET
  .local_end 6
  .local_end 4
  nb_control_4:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
  nb_control_0:
  .local_end 0
.end
.parameters 1 struct
.parameters 2 union
