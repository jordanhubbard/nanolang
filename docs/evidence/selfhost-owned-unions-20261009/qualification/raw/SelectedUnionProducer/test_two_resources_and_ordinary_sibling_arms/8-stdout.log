.types 1 0 1
.entry 0
.string "Handle"
.string "fd"
.string "Choice"
.string "left"
.string "right"
.string "label"
.string "number"
.string "__nanoisa_shadow_entry"
.string "close_handle"
.string "consume"
.string "main"
.string "kept"
.string "Some"
.string "Plain"
.string "None"
.layouts "02000000000001000000000001000000ffffffff01000000020004000200000008000000000000000300000008000000000000000400000005000000ffffffff0500000001000000ffffffff06000000"
.ownership "030000000200000003030000040000000a00000001000000ffffffff01000000ffffffff01000000ffffffff080000000000000001000000ffffffff080000000000000005000000ffffffff0a000000010000000a0000000100000001000000ffffffff0a000000010000000300010001000000ffffffff0800000000000000080000000000000001000000ffffffff0b00010001000000ffffffff0a000000010000000a000000010000000800000000000000080000000000000005000000ffffffff080000000000000005000000ffffffff08000000000000000a0000000100000001000000ffffffff0a000000010000000600000001000000ffffffff01000000ffffffff080000000000000001000000ffffffff080000000000000005000000ffffffff0a0000000100000004000000000000000100000001000100240000000100000001000000030000000c000000000003000d000000030001000e00000004000000"
.function __nanoisa_shadow_entry 0 10 0 int 1
  .shadow "close_handle"
  PUSH_I64 3
  STORE_LOCAL 0
  LOAD_LOCAL 0
  OWN_PACK 0
  CALL 1
  PUSH_I64 3
  EQ
  ASSERT
  .shadow "consume"
  PUSH_I64 3
  STORE_LOCAL 1
  LOAD_LOCAL 1
  OWN_PACK 0
  OWN_STORE_LOCAL 2
  PUSH_I64 4
  STORE_LOCAL 3
  LOAD_LOCAL 3
  OWN_PACK 0
  OWN_STORE_LOCAL 4
  PUSH_STR 11
  STORE_LOCAL 5
  OWN_MOVE_LOCAL 2
  OWN_MOVE_LOCAL 4
  LOAD_LOCAL 5
  AGG_PACK 1 0 0 3
  OWN_STORE_LOCAL 6
  .local_begin 6 "value"
  OWN_MOVE_LOCAL 6
  CALL 2
  PUSH_I64 7
  EQ
  ASSERT
  AGG_PACK 1 0 2 0
  OWN_STORE_LOCAL 7
  .local_begin 7 "empty"
  OWN_MOVE_LOCAL 7
  CALL 2
  PUSH_I64 0
  EQ
  ASSERT
  PUSH_I64 7
  STORE_LOCAL 8
  LOAD_LOCAL 8
  AGG_PACK 1 0 1 1
  OWN_STORE_LOCAL 9
  .local_begin 9 "plain"
  OWN_MOVE_LOCAL 9
  CALL 2
  PUSH_I64 7
  EQ
  ASSERT
  .local_end 6
  .local_end 7
  .local_end 9
  .shadow "main"
  CALL 3
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
.function consume 1 11 0 int 1
  .local_begin 0 "value"
  OWN_MOVE_LOCAL 0
  MATCH_TAG 0 nb_control_1
  JMP nb_control_2
  nb_control_1:
  OWN_STORE_LOCAL 1
  .local_begin 1 "payload"
  OWN_UNPACK_LOCAL 1
  STORE_LOCAL 4
  OWN_STORE_LOCAL 3
  OWN_STORE_LOCAL 2
  OWN_MOVE_LOCAL 3
  OWN_STORE_LOCAL 5
  .local_begin 5 "right"
  LOAD_LOCAL 4
  STORE_LOCAL 6
  .local_begin 6 "label"
  OWN_MOVE_LOCAL 2
  OWN_STORE_LOCAL 7
  .local_begin 7 "left"
  LOAD_LOCAL 6
  PUSH_STR 11
  EQ
  ASSERT
  OWN_MOVE_LOCAL 7
  CALL 1
  OWN_MOVE_LOCAL 5
  CALL 1
  ADD
  RET
  .local_end 5
  .local_end 6
  .local_end 7
  .local_end 1
  nb_control_2:
  MATCH_TAG 1 nb_control_3
  JMP nb_control_4
  nb_control_3:
  OWN_STORE_LOCAL 8
  OWN_UNPACK_LOCAL 8
  STORE_LOCAL 9
  .local_begin 8 "payload"
  LOAD_LOCAL 9
  RET
  .local_end 8
  nb_control_4:
  MATCH_TAG 2 nb_control_5
  JMP nb_control_6
  nb_control_5:
  OWN_STORE_LOCAL 10
  OWN_UNPACK_LOCAL 10
  .local_begin 10 "payload"
  PUSH_I64 0
  RET
  .local_end 10
  nb_control_6:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
  nb_control_0:
  .local_end 0
.end
.function main 0 6 0 int 1
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
  PUSH_STR 11
  STORE_LOCAL 4
  OWN_MOVE_LOCAL 1
  OWN_MOVE_LOCAL 3
  LOAD_LOCAL 4
  AGG_PACK 1 0 0 3
  OWN_STORE_LOCAL 5
  .local_begin 5 "value"
  OWN_MOVE_LOCAL 5
  CALL 2
  PUSH_I64 7
  SUB
  RET
  .local_end 5
.end
.parameters 1 struct
.parameters 2 union
.parameters 3
