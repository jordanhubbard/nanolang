.types 1 0 1
.entry 0
.string "Handle"
.string "fd"
.string "Marker<Handle>"
.string "number"
.string "main"
.string "close_handle"
.string "read"
.string "Value"
.layouts "02000000000001000000000001000000ffffffff01000000020001000200000001000000ffffffff03000000"
.ownership "030000000200000003000000030000000400000001000000ffffffff01000000ffffffff0a000000010000000a0000000100000001000000ffffffff0300010001000000ffffffff0800000000000000080000000000000001000000ffffffff0200010001000000ffffffff0a000000010000000a0000000100000004000000000000000100000001000100140000000100000001000000010000000700000000000100"
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
.parameters 1 struct
.parameters 2 union
