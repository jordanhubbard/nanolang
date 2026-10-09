.types 2 0 1
.entry 0
.string "Handle"
.string "fd"
.string "Pair"
.string "owner"
.string "label"
.string "Choice"
.string "pair"
.string "__nanoisa_shadow_entry"
.string "close_handle"
.string "consume"
.string "main"
.string "kept"
.string "Some"
.string "None"
.layouts "03000000000001000000000001000000ffffffff01000000000002000200000008000000000000000300000005000000ffffffff040000000200010005000000080000000100000006000000"
.ownership "030000000300000003030300040000000600000001000000ffffffff01000000ffffffff01000000ffffffff080000000000000005000000ffffffff08000000010000000a000000020000000300010001000000ffffffff0800000000000000080000000000000001000000ffffffff0a00010001000000ffffffff0a000000020000000a00000002000000080000000100000008000000010000000800000001000000080000000000000005000000ffffffff080000000000000005000000ffffffff0a000000020000000500000001000000ffffffff01000000ffffffff080000000000000005000000ffffffff08000000010000000a00000002000000040000000000000001000000010001001c0000000100000002000000020000000c000000000001000d00000001000000"
.function __nanoisa_shadow_entry 0 6 0 int 1
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
  PUSH_STR 11
  STORE_LOCAL 3
  OWN_MOVE_LOCAL 2
  LOAD_LOCAL 3
  OWN_PACK 1
  OWN_STORE_LOCAL 4
  OWN_MOVE_LOCAL 4
  AGG_PACK 1 0 0 1
  OWN_STORE_LOCAL 5
  .local_begin 5 "value"
  OWN_MOVE_LOCAL 5
  CALL 2
  PUSH_I64 7
  EQ
  ASSERT
  .local_end 5
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
.function consume 1 10 0 int 1
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
  .local_begin 3 "pair"
  OWN_MOVE_LOCAL 3
  OWN_STORE_LOCAL 4
  OWN_UNPACK_LOCAL 4
  STORE_LOCAL 6
  OWN_STORE_LOCAL 5
  OWN_MOVE_LOCAL 5
  OWN_STORE_LOCAL 7
  .local_begin 7 "owner"
  LOAD_LOCAL 6
  STORE_LOCAL 8
  .local_begin 8 "label"
  LOAD_LOCAL 8
  PUSH_STR 11
  EQ
  ASSERT
  OWN_MOVE_LOCAL 7
  CALL 1
  RET
  .local_end 3
  .local_end 7
  .local_end 8
  .local_end 1
  nb_control_2:
  MATCH_TAG 1 nb_control_3
  JMP nb_control_4
  nb_control_3:
  OWN_STORE_LOCAL 9
  OWN_UNPACK_LOCAL 9
  .local_begin 9 "payload"
  PUSH_I64 0
  RET
  .local_end 9
  nb_control_4:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
  nb_control_0:
  .local_end 0
.end
.function main 0 5 0 int 1
  PUSH_I64 7
  STORE_LOCAL 0
  LOAD_LOCAL 0
  OWN_PACK 0
  OWN_STORE_LOCAL 1
  PUSH_STR 11
  STORE_LOCAL 2
  OWN_MOVE_LOCAL 1
  LOAD_LOCAL 2
  OWN_PACK 1
  OWN_STORE_LOCAL 3
  OWN_MOVE_LOCAL 3
  AGG_PACK 1 0 0 1
  OWN_STORE_LOCAL 4
  .local_begin 4 "value"
  OWN_MOVE_LOCAL 4
  CALL 2
  PUSH_I64 7
  SUB
  RET
  .local_end 4
.end
.parameters 1 struct
.parameters 2 union
.parameters 3
