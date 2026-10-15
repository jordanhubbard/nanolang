.types 1 0 1
.entry 0
.string "Handle"
.string "fd"
.string "Marker<Handle>"
.string "number"
.string "__nanoisa_shadow_entry"
.string "close_handle"
.string "read"
.string "main"
.string "Value"
.layouts "02000000000001000000000001000000ffffffff01000000020001000200000001000000ffffffff03000000"
.ownership "030000000200000003000000040000000300000001000000ffffffff01000000ffffffff01000000ffffffff0a000000010000000300010001000000ffffffff0800000000000000080000000000000001000000ffffffff0200010001000000ffffffff0a000000010000000a000000010000000400000001000000ffffffff01000000ffffffff0a000000010000000a0000000100000001000000ffffffff04000000000000000100000001000100140000000100000001000000010000000800000000000100"
.function __nanoisa_shadow_entry 0 3 0 int 1
  .shadow "close_handle"
  PUSH_I64 7
  STORE_LOCAL 0
  LOAD_LOCAL 0
  OWN_PACK 0
  CALL 1
  PUSH_I64 7
  EQ
  ASSERT
  .shadow "read"
  PUSH_I64 4
  STORE_LOCAL 1
  LOAD_LOCAL 1
  AGG_PACK 1 0 0 1
  STORE_LOCAL 2
  .local_begin 2 "marker"
  LOAD_LOCAL 2
  CALL 2
  PUSH_I64 4
  EQ
  ASSERT
  LOAD_LOCAL 2
  CALL 2
  PUSH_I64 4
  EQ
  ASSERT
  .local_end 2
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
.function read 1 2 0 int 1
  .local_begin 0 "value"
  LOAD_LOCAL 0
  MATCH_TAG 0 nb_control_1
  JMP nb_control_2
  nb_control_1:
  DUP
  STORE_LOCAL 1
  .local_begin 1 "payload"
  POP
  LOAD_LOCAL 1
  AGG_GET 0
  .local_end 1
  JMP nb_control_0
  nb_control_2:
  POP
  PUSH_BOOL 0
  ASSERT
  HALT
  nb_control_0:
  RET
  .local_end 0
.end
.function main 0 4 0 int 1
  PUSH_I64 4
  STORE_LOCAL 0
  LOAD_LOCAL 0
  AGG_PACK 1 0 0 1
  STORE_LOCAL 1
  .local_begin 1 "marker"
  LOAD_LOCAL 1
  STORE_LOCAL 2
  .local_begin 2 "copied"
  LOAD_LOCAL 1
  CALL 2
  LOAD_LOCAL 2
  CALL 2
  ADD
  PUSH_I64 8
  EQ
  ASSERT
  PUSH_I64 7
  STORE_LOCAL 3
  LOAD_LOCAL 3
  OWN_PACK 0
  CALL 1
  PUSH_I64 7
  EQ
  ASSERT
  PUSH_I64 0
  RET
  .local_end 1
  .local_end 2
.end
.parameters 1 struct
.parameters 2 union
.parameters 3
