.types 1 0 1
.entry 0
.string "Handle"
.string "fd"
.string "Choice"
.string "left"
.string "right"
.string "label"
.string "number"
.string "main"
.string "close_handle"
.string "consume"
.string "kept"
.string "Some"
.string "Plain"
.string "None"
.layouts "02000000000001000000000001000000ffffffff01000000020004000200000008000000000000000300000008000000000000000400000005000000ffffffff0500000001000000ffffffff06000000"
.ownership "030000000200000003030000030000000700000001000000ffffffff01000000ffffffff080000000000000001000000ffffffff080000000000000005000000ffffffff0a0000000100000001000000ffffffff0300010001000000ffffffff0800000000000000080000000000000001000000ffffffff0d00020001000000ffffffff0a00000001000000080000000000000001000000ffffffff0a000000010000000800000000000000080000000000000005000000ffffffff0800000000000000080000000000000005000000ffffffff0a0000000100000001000000ffffffff0a0000000100000004000000000000000100000001000100240000000100000001000000030000000b000000000003000c000000030001000d00000004000000"
.function main 0 7 0 int 1
  PUSH_I64 3
  STORE_LOCAL 0
  LOAD_LOCAL 0
  OWN_PACK 0
  OWN_STORE_LOCAL 1
  PUSH_I64 4
  STORE_LOCAL 2
  LOAD_LOCAL 2
  OWN_PACK 0
  OWN_STORE_LOCAL 3
  PUSH_STR 10
  STORE_LOCAL 4
  OWN_MOVE_LOCAL 1
  OWN_MOVE_LOCAL 3
  LOAD_LOCAL 4
  AGG_PACK 1 0 0 3
  OWN_STORE_LOCAL 5
  .local_begin 5 "value"
  OWN_MOVE_LOCAL 5
  PUSH_I64 2
  STORE_LOCAL 6
  LOAD_LOCAL 6
  OWN_PACK 0
  CALL 2
  PUSH_I64 9
  SUB
  RET
  .local_end 5
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
.function consume 2 13 0 int 1
  .local_begin 0 "value"
  .local_begin 1 "extra"
  PUSH_I64 0
  STORE_LOCAL 2
  .local_begin 2 "total"
  OWN_MOVE_LOCAL 0
  MATCH_TAG 0 nb_control_1
  JMP nb_control_2
  nb_control_1:
  OWN_STORE_LOCAL 3
  .local_begin 3 "payload"
  OWN_UNPACK_LOCAL 3
  STORE_LOCAL 6
  OWN_STORE_LOCAL 5
  OWN_STORE_LOCAL 4
  OWN_MOVE_LOCAL 4
  OWN_STORE_LOCAL 7
  .local_begin 7 "left"
  OWN_MOVE_LOCAL 5
  OWN_STORE_LOCAL 8
  .local_begin 8 "right"
  LOAD_LOCAL 6
  STORE_LOCAL 9
  .local_begin 9 "label"
  OWN_MOVE_LOCAL 7
  CALL 1
  OWN_MOVE_LOCAL 8
  CALL 1
  ADD
  STORE_LOCAL 2
  .local_end 7
  .local_end 8
  .local_end 9
  .local_end 3
  JMP nb_control_0
  nb_control_2:
  MATCH_TAG 1 nb_control_3
  JMP nb_control_4
  nb_control_3:
  OWN_STORE_LOCAL 10
  OWN_UNPACK_LOCAL 10
  STORE_LOCAL 11
  .local_begin 10 "payload"
  LOAD_LOCAL 11
  STORE_LOCAL 2
  .local_end 10
  JMP nb_control_0
  nb_control_4:
  MATCH_TAG 2 nb_control_5
  JMP nb_control_6
  nb_control_5:
  OWN_STORE_LOCAL 12
  OWN_UNPACK_LOCAL 12
  .local_begin 12 "payload"
  .local_end 12
  JMP nb_control_0
  nb_control_6:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
  nb_control_0:
  LOAD_LOCAL 2
  OWN_MOVE_LOCAL 1
  CALL 1
  ADD
  RET
  .local_end 0
  .local_end 1
  .local_end 2
.end
.parameters 1 struct
.parameters 2 union struct
