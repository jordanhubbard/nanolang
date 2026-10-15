.types 1 0 2
.entry 0
.string "Handle"
.string "fd"
.string "Result<Handle,string>"
.string "value"
.string "error"
.string "Box<Result<Handle,string>>"
.string "__nanoisa_shadow_entry"
.string "close_handle"
.string "consume"
.string "outer"
.string "main"
.string "empty"
.string "Ok"
.string "Err"
.string "Some"
.string "None"
.layouts "03000000000001000000000001000000ffffffff01000000020002000200000008000000000000000300000005000000ffffffff0400000002000100050000000a0000000100000003000000"
.ownership "030000000300000003030300050000000d00000001000000ffffffff01000000ffffffff01000000ffffffff080000000000000008000000000000000a0000000100000005000000ffffffff0a0000000100000001000000ffffffff080000000000000008000000000000000a000000010000000a000000010000000a000000020000000300010001000000ffffffff0800000000000000080000000000000001000000ffffffff0700010001000000ffffffff0a000000010000000a00000001000000080000000000000008000000000000000a0000000100000005000000ffffffff05000000ffffffff0500010001000000ffffffff0a000000020000000a000000020000000a000000010000000a000000010000000a000000020000000000000001000000ffffffff04000000000000000100000001000100340000000200000001000000020000000c000000000001000d0000000100010002000000020000000e000000000001000f00000001000000"
.function __nanoisa_shadow_entry 0 13 0 int 1
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
  PUSH_STR 11
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
  .shadow "outer"
  PUSH_I64 7
  STORE_LOCAL 7
  LOAD_LOCAL 7
  OWN_PACK 0
  OWN_STORE_LOCAL 8
  .local_begin 8 "owner"
  OWN_MOVE_LOCAL 8
  OWN_STORE_LOCAL 9
  OWN_MOVE_LOCAL 9
  AGG_PACK 1 0 0 1
  OWN_STORE_LOCAL 10
  .local_begin 10 "result"
  OWN_MOVE_LOCAL 10
  OWN_STORE_LOCAL 11
  OWN_MOVE_LOCAL 11
  AGG_PACK 1 1 0 1
  OWN_STORE_LOCAL 12
  .local_begin 12 "boxed"
  OWN_MOVE_LOCAL 12
  CALL 3
  PUSH_I64 7
  EQ
  ASSERT
  .local_end 8
  .local_end 10
  .local_end 12
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
  PUSH_STR 11
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
.function outer 1 5 0 int 1
  .local_begin 0 "boxed"
  OWN_MOVE_LOCAL 0
  MATCH_TAG 0 nb_control_6
  JMP nb_control_7
  nb_control_6:
  OWN_STORE_LOCAL 1
  .local_begin 1 "payload"
  OWN_UNPACK_LOCAL 1
  OWN_STORE_LOCAL 2
  OWN_MOVE_LOCAL 2
  OWN_STORE_LOCAL 3
  .local_begin 3 "value"
  OWN_MOVE_LOCAL 3
  CALL 2
  RET
  .local_end 3
  .local_end 1
  nb_control_7:
  MATCH_TAG 1 nb_control_8
  JMP nb_control_9
  nb_control_8:
  OWN_STORE_LOCAL 4
  OWN_UNPACK_LOCAL 4
  .local_begin 4 "payload"
  PUSH_I64 0
  RET
  .local_end 4
  nb_control_9:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
  nb_control_5:
  .local_end 0
.end
.function main 0 0 0 int 1
  PUSH_I64 0
  RET
.end
.parameters 1 struct
.parameters 2 union
.parameters 3 union
.parameters 4
