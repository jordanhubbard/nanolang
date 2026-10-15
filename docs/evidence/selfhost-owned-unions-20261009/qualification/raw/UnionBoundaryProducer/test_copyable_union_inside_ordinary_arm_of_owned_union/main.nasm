.types 1 0 2
.entry 0
.string "Handle"
.string "fd"
.string "Box<int>"
.string "item"
.string "Choice"
.string "owner"
.string "inner"
.string "main"
.string "close_handle"
.string "consume"
.string "Some"
.string "None"
.string "Owned"
.string "Plain"
.layouts "03000000000001000000000001000000ffffffff01000000020001000200000001000000ffffffff0300000002000200040000000800000000000000050000000a0000000100000006000000"
.ownership "030000000300000003010300030000000500000001000000ffffffff01000000ffffffff0a000000010000000a0000000100000001000000ffffffff08000000000000000300010001000000ffffffff0800000000000000080000000000000001000000ffffffff0900010001000000ffffffff0a000000020000000a00000002000000080000000000000008000000000000000a000000020000000a000000010000000a000000010000000a000000010000000a0000000100000004000000000000000100000001000100340000000200000001000000020000000a000000000001000b0000000100000002000000020000000c000000000001000d00000001000100"
.function main 0 5 0 int 1
  PUSH_I64 9
  STORE_LOCAL 0
  LOAD_LOCAL 0
  AGG_PACK 1 0 0 1
  STORE_LOCAL 1
  .local_begin 1 "inner"
  LOAD_LOCAL 1
  STORE_LOCAL 2
  LOAD_LOCAL 2
  AGG_PACK 1 1 1 1
  CALL 2
  PUSH_I64 9
  EQ
  ASSERT
  PUSH_I64 7
  STORE_LOCAL 3
  LOAD_LOCAL 3
  OWN_PACK 0
  OWN_STORE_LOCAL 4
  OWN_MOVE_LOCAL 4
  AGG_PACK 1 1 0 1
  CALL 2
  PUSH_I64 7
  EQ
  ASSERT
  PUSH_I64 0
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
.function consume 1 9 0 int 1
  .local_begin 0 "value"
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
  .local_begin 3 "owner"
  OWN_MOVE_LOCAL 3
  CALL 1
  .local_end 1
  .local_end 3
  JMP nb_control_0
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
  .local_begin 6 "inner"
  LOAD_LOCAL 6
  MATCH_TAG 0 nb_control_6
  JMP nb_control_7
  nb_control_6:
  DUP
  STORE_LOCAL 7
  .local_begin 7 "item"
  POP
  LOAD_LOCAL 7
  AGG_GET 0
  .local_end 7
  JMP nb_control_5
  nb_control_7:
  MATCH_TAG 1 nb_control_8
  JMP nb_control_9
  nb_control_8:
  DUP
  STORE_LOCAL 8
  .local_begin 8 "empty"
  POP
  PUSH_I64 0
  .local_end 8
  JMP nb_control_5
  nb_control_9:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
  nb_control_5:
  .local_end 4
  .local_end 6
  JMP nb_control_0
  nb_control_4:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
  nb_control_0:
  RET
  .local_end 0
.end
.parameters 1 struct
.parameters 2 union
