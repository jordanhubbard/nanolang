.types 1 0 1
.entry 0
.string "Handle"
.string "fd"
.string "Choice"
.string "value"
.string "__nanoisa_shadow_entry"
.string "main"
.string "Some"
.string "None"
.layouts "02000000000001000000000001000000ffffffff01000000020001000200000001000000ffffffff03000000"
.ownership "030000000200000003010000020000000100000001000000ffffffff01000000ffffffff0800000001000000ffffffff01000000ffffffff0800000000000000080000000000000001000000ffffffff0a000000010000000a000000010000000a000000010000000a00000001000000040000000000000002000000010001001c00000001000000010000000200000006000000000001000700000001000000030001000c000000010000000a01000001000000"
.function __nanoisa_shadow_entry 0 1 0 int 1
  PUSH_I64 7
  STORE_LOCAL 0
  LOAD_LOCAL 0
  AGG_PACK 1 0 0 1
  STORE_GLOBAL 0
  .shadow "main"
  CALL 1
  PUSH_I64 0
  EQ
  ASSERT
  PUSH_I64 0
  RET
.end
.function main 0 8 0 int 1
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
  LOAD_GLOBAL 0
  MATCH_TAG 0 nb_control_1
  JMP nb_control_2
  nb_control_1:
  DUP
  STORE_LOCAL 4
  .local_begin 4 "payload"
  POP
  LOAD_LOCAL 4
  AGG_GET 0
  PUSH_I64 7
  EQ
  ASSERT
  .local_end 4
  JMP nb_control_0
  nb_control_2:
  MATCH_TAG 1 nb_control_3
  JMP nb_control_4
  nb_control_3:
  DUP
  STORE_LOCAL 5
  .local_begin 5 "payload"
  POP
  PUSH_BOOL 0
  ASSERT
  .local_end 5
  JMP nb_control_0
  nb_control_4:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
  nb_control_0:
  AGG_PACK 1 0 1 0
  STORE_GLOBAL 0
  LOAD_GLOBAL 0
  MATCH_TAG 0 nb_control_6
  JMP nb_control_7
  nb_control_6:
  DUP
  STORE_LOCAL 6
  .local_begin 6 "payload"
  POP
  PUSH_BOOL 0
  ASSERT
  .local_end 6
  JMP nb_control_5
  nb_control_7:
  MATCH_TAG 1 nb_control_8
  JMP nb_control_9
  nb_control_8:
  DUP
  STORE_LOCAL 7
  .local_begin 7 "payload"
  POP
  PUSH_BOOL 1
  ASSERT
  .local_end 7
  JMP nb_control_5
  nb_control_9:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
  nb_control_5:
  PUSH_I64 0
  OWN_UNPACK_LOCAL 2
  POP
  RET
  .local_end 1
  .local_end 3
.end
.parameters 1
