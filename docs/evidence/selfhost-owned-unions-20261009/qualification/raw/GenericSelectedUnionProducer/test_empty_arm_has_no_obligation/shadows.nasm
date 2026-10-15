.types 1 0 1
.entry 0
.string "Handle"
.string "fd"
.string "Box<Handle>"
.string "value"
.string "__nanoisa_shadow_entry"
.string "close_handle"
.string "consume"
.string "main"
.string "Some"
.string "None"
.layouts "02000000000001000000000001000000ffffffff010000000200010002000000080000000000000003000000"
.ownership "030000000200000003030000040000000600000001000000ffffffff01000000ffffffff0a0000000100000001000000ffffffff080000000000000008000000000000000a000000010000000300010001000000ffffffff0800000000000000080000000000000001000000ffffffff0500010001000000ffffffff0a000000010000000a00000001000000080000000000000008000000000000000a000000010000000000000001000000ffffffff040000000000000001000000010001001c00000001000000010000000200000008000000000001000900000001000000"
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
  AGG_PACK 1 0 1 0
  OWN_STORE_LOCAL 1
  .local_begin 1 "empty"
  OWN_MOVE_LOCAL 1
  CALL 2
  PUSH_I64 0
  EQ
  ASSERT
  PUSH_I64 7
  STORE_LOCAL 2
  LOAD_LOCAL 2
  OWN_PACK 0
  OWN_STORE_LOCAL 3
  .local_begin 3 "owner"
  OWN_MOVE_LOCAL 3
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
  .local_end 1
  .local_end 3
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
.function consume 1 5 0 int 1
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
  .local_begin 4 "payload"
  PUSH_I64 0
  RET
  .local_end 4
  nb_control_4:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
  nb_control_0:
  .local_end 0
.end
.function main 0 0 0 int 1
  PUSH_I64 0
  RET
.end
.parameters 1 struct
.parameters 2 union
.parameters 3
