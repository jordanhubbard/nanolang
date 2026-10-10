.types 1 0 1
.entry 0
.string "Handle"
.string "fd"
.string "Choice"
.string "owner"
.string "__nanoisa_shadow_entry"
.string "make_value"
.string "consume"
.string "main"
.string "Some"
.string "None"
.layouts "02000000000001000000000001000000ffffffff010000000200010002000000080000000000000003000000"
.ownership "030000000200000003030000040000000700000001000000ffffffff0a000000010000000a0000000100000008000000000000000800000000000000080000000000000001000000ffffffff0a00000001000000020000000a0000000100000001000000ffffffff08000000000000000600000001000000ffffffff0a0000000100000008000000000000000800000000000000080000000000000001000000ffffffff0a000000010000000000000001000000ffffffff040000000000000002000000010001001c00000001000000010000000200000008000000000001000900000001000000030001000c0000000100000001010000ffffffff"
.function __nanoisa_shadow_entry 0 7 0 int 1
  PUSH_I64 0
  STORE_GLOBAL 0
  .shadow "make_value"
  PUSH_I64 0
  STORE_GLOBAL 0
  CALL 1
  OWN_STORE_LOCAL 0
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
  OWN_STORE_LOCAL 4
  REGION_BEGIN
  BORROW_LOCAL_SHARED 0 4
  REF_GET 0 0
  REGION_END
  STORE_LOCAL 5
  .local_begin 5 "fd"
  LOAD_LOCAL 5
  PUSH_I64 7
  EQ
  ASSERT
  OWN_UNPACK_LOCAL 4
  POP
  .local_end 3
  .local_end 5
  .local_end 1
  JMP nb_control_0
  nb_control_2:
  MATCH_TAG 1 nb_control_3
  JMP nb_control_4
  nb_control_3:
  OWN_STORE_LOCAL 6
  OWN_UNPACK_LOCAL 6
  .local_begin 6 "payload"
  PUSH_BOOL 0
  ASSERT
  .local_end 6
  JMP nb_control_0
  nb_control_4:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
  nb_control_0:
  LOAD_GLOBAL 0
  PUSH_I64 1
  EQ
  ASSERT
  .local_end 0
  .shadow "consume"
  PUSH_I64 0
  STORE_GLOBAL 0
  CALL 2
  PUSH_I64 7
  EQ
  ASSERT
  LOAD_GLOBAL 0
  PUSH_I64 1
  EQ
  ASSERT
  .shadow "main"
  CALL 3
  PUSH_I64 0
  EQ
  ASSERT
  PUSH_I64 0
  RET
.end
.function make_value 0 2 0 union 1
  LOAD_GLOBAL 0
  PUSH_I64 1
  ADD
  STORE_GLOBAL 0
  PUSH_I64 7
  STORE_LOCAL 0
  LOAD_LOCAL 0
  OWN_PACK 0
  OWN_STORE_LOCAL 1
  OWN_MOVE_LOCAL 1
  AGG_PACK 1 0 0 1
  RET
.end
.function consume 0 6 0 int 1
  CALL 1
  MATCH_TAG 0 nb_control_6
  JMP nb_control_7
  nb_control_6:
  OWN_STORE_LOCAL 0
  .local_begin 0 "payload"
  OWN_UNPACK_LOCAL 0
  OWN_STORE_LOCAL 1
  OWN_MOVE_LOCAL 1
  OWN_STORE_LOCAL 2
  .local_begin 2 "owner"
  OWN_MOVE_LOCAL 2
  OWN_STORE_LOCAL 3
  REGION_BEGIN
  BORROW_LOCAL_SHARED 0 3
  REF_GET 0 0
  REGION_END
  STORE_LOCAL 4
  .local_begin 4 "fd"
  LOAD_LOCAL 4
  OWN_UNPACK_LOCAL 3
  POP
  RET
  .local_end 2
  .local_end 4
  .local_end 0
  nb_control_7:
  MATCH_TAG 1 nb_control_8
  JMP nb_control_9
  nb_control_8:
  OWN_STORE_LOCAL 5
  OWN_UNPACK_LOCAL 5
  .local_begin 5 "payload"
  PUSH_I64 0
  RET
  .local_end 5
  nb_control_9:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
  nb_control_5:
.end
.function main 0 0 0 int 1
  PUSH_I64 0
  STORE_GLOBAL 0
  CALL 2
  PUSH_I64 7
  EQ
  ASSERT
  LOAD_GLOBAL 0
  PUSH_I64 1
  EQ
  ASSERT
  PUSH_I64 0
  RET
.end
.parameters 1
.parameters 2
.parameters 3
