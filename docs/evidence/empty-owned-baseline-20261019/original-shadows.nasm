.types 2 0 0
.entry 0
.string "Handle"
.string "fd"
.string "Empty"
.string "__nanoisa_shadow_entry"
.string "close"
.string "main"
.layouts "02000000000001000000000001000000ffffffff010000000000000002000000"
.ownership "010000000200000003030000030000000100000001000000ffffffff01000000ffffffff0300010001000000ffffffff0800000000000000080000000000000001000000ffffffff0200000001000000ffffffff08000000010000000800000001000000"
.function __nanoisa_shadow_entry 0 1 0 int 1
  .shadow "close"
  PUSH_I64 7
  STORE_LOCAL 0
  LOAD_LOCAL 0
  OWN_PACK 0
  CALL 1
  PUSH_I64 7
  EQ
  ASSERT
  .shadow "main"
  CALL 2
  PUSH_I64 0
  EQ
  ASSERT
  PUSH_I64 0
  RET
.end
.function close 1 3 0 int 1
  .local_begin 0 "owned"
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
.function main 0 2 0 int 1
  OWN_PACK 1
  OWN_STORE_LOCAL 0
  .local_begin 0 "value"
  OWN_MOVE_LOCAL 0
  OWN_STORE_LOCAL 1
  PUSH_I64 0
  OWN_UNPACK_LOCAL 1
  RET
  .local_end 0
.end
.affine_completion 0 0
.affine_completion 1 0
.parameters 1 struct
.affine_completion 2 0
.parameters 2
