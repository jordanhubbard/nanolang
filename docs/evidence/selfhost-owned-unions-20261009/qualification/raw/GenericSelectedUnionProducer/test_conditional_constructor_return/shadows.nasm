.types 1 0 1
.entry 0
.string "Handle"
.string "fd"
.string "Result<Handle,string>"
.string "value"
.string "error"
.string "__nanoisa_shadow_entry"
.string "close_handle"
.string "consume"
.string "choose"
.string "main"
.string "empty"
.string "Ok"
.string "Err"
.layouts "02000000000001000000000001000000ffffffff01000000020002000200000008000000000000000300000005000000ffffffff04000000"
.ownership "030000000200000003030000050000000900000001000000ffffffff01000000ffffffff01000000ffffffff080000000000000008000000000000000a0000000100000005000000ffffffff0a0000000100000001000000ffffffff01000000ffffffff0300010001000000ffffffff0800000000000000080000000000000001000000ffffffff0700010001000000ffffffff0a000000010000000a00000001000000080000000000000008000000000000000a0000000100000005000000ffffffff05000000ffffffff040002000a00000001000000080000000000000004000000ffffffff080000000000000005000000ffffffff0100000001000000ffffffff01000000ffffffff040000000000000001000000010001001c0000000100000001000000020000000b000000000001000c00000001000100"
.function __nanoisa_shadow_entry 0 9 0 int 1
  .shadow "close_handle"
  PUSH_I64 7
  STORE_LOCAL 0
  LOAD_LOCAL 0
  OWN_PACK 0
  CALL 1
  PUSH_I64 7
  EQ
  ASSERT
  .shadow "consume"
  PUSH_I64 7
  STORE_LOCAL 1
  LOAD_LOCAL 1
  OWN_PACK 0
  OWN_STORE_LOCAL 2
  .local_begin 2 "owner"
  OWN_MOVE_LOCAL 2
  OWN_STORE_LOCAL 3
  OWN_MOVE_LOCAL 3
  AGG_PACK 1 0 0 1
  OWN_STORE_LOCAL 4
  .local_begin 4 "value"
  OWN_MOVE_LOCAL 4
  CALL 2
  PUSH_I64 7
  EQ
  ASSERT
  PUSH_STR 10
  STORE_LOCAL 5
  LOAD_LOCAL 5
  AGG_PACK 1 0 1 1
  OWN_STORE_LOCAL 6
  .local_begin 6 "empty"
  OWN_MOVE_LOCAL 6
  CALL 2
  PUSH_I64 0
  EQ
  ASSERT
  .local_end 2
  .local_end 4
  .local_end 6
  .shadow "choose"
  PUSH_I64 7
  STORE_LOCAL 7
  LOAD_LOCAL 7
  OWN_PACK 0
  PUSH_BOOL 1
  CALL 3
  CALL 2
  PUSH_I64 7
  EQ
  ASSERT
  PUSH_I64 8
  STORE_LOCAL 8
  LOAD_LOCAL 8
  OWN_PACK 0
  PUSH_BOOL 0
  CALL 3
  CALL 2
  PUSH_I64 0
  EQ
  ASSERT
  .shadow "main"
  CALL 4
  PUSH_I64 0
  EQ
  ASSERT
  PUSH_I64 0
  RET
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
  PUSH_STR 10
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
.function choose 2 4 0 union 1
  .local_begin 0 "owner"
  .local_begin 1 "keep"
  nb_control_5:
  LOAD_LOCAL 1
  JMP_FALSE nb_control_6
  OWN_MOVE_LOCAL 0
  OWN_STORE_LOCAL 2
  OWN_MOVE_LOCAL 2
  AGG_PACK 1 0 0 1
  RET
  nb_control_6:
  OWN_MOVE_LOCAL 0
  CALL 1
  POP
  PUSH_STR 10
  STORE_LOCAL 3
  LOAD_LOCAL 3
  AGG_PACK 1 0 1 1
  RET
  nb_control_7:
  .local_end 0
  .local_end 1
.end
.function main 0 1 0 int 1
  PUSH_I64 9
  STORE_LOCAL 0
  LOAD_LOCAL 0
  OWN_PACK 0
  PUSH_BOOL 1
  CALL 3
  CALL 2
  PUSH_I64 9
  SUB
  RET
.end
.parameters 1 struct
.parameters 2 union
.parameters 3 struct bool
.parameters 4
