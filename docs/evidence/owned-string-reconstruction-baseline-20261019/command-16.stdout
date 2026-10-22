.types 1 0 0
.entry 0
.string "FileHandle"
.string "fd"
.string "__nanoisa_shadow_entry"
.string "open_file"
.string "close_file"
.string "main"
.string "audit.log"
.string "resource closed"
.layouts "01000000000001000000000001000000ffffffff01000000"
.ownership "010000000100000003000000040000000300000001000000ffffffff080000000000000001000000ffffffff080000000000000002000100080000000000000005000000ffffffff01000000ffffffff0300010000000000ffffffff0800000000000000080000000000000001000000ffffffff0100000001000000ffffffff0800000000000000"
.function __nanoisa_shadow_entry 0 3 0 int 1
  .shadow "open_file"
  PUSH_STR 6
  CALL 1
  OWN_STORE_LOCAL 0
  .local_begin 0 "f"
  OWN_MOVE_LOCAL 0
  CALL 2
  .local_end 0
  .shadow "close_file"
  PUSH_I64 1
  STORE_LOCAL 1
  LOAD_LOCAL 1
  OWN_PACK 0
  OWN_STORE_LOCAL 2
  .local_begin 2 "f"
  OWN_MOVE_LOCAL 2
  CALL 2
  .local_end 2
  .shadow "main"
  CALL 3
  PUSH_I64 0
  EQ
  ASSERT
  PUSH_I64 0
  RET
.end
.function open_file 1 2 0 struct 1
  .local_begin 0 "path"
  LOAD_LOCAL 0
  PRINTLN
  PUSH_I64 42
  STORE_LOCAL 1
  LOAD_LOCAL 1
  OWN_PACK 0
  RET
  .local_end 0
.end
.function close_file 1 3 0 void 0
  .local_begin 0 "_f"
  OWN_MOVE_LOCAL 0
  OWN_STORE_LOCAL 1
  REGION_BEGIN
  BORROW_LOCAL_SHARED 1 1
  REF_GET 1 0
  REGION_END
  STORE_LOCAL 2
  .local_begin 2 "fd"
  LOAD_LOCAL 2
  PUSH_I64 0
  GT
  ASSERT
  PUSH_STR 7
  PRINTLN
  OWN_UNPACK_LOCAL 1
  POP
  RET
  .local_end 0
  .local_end 2
.end
.function main 0 1 0 int 1
  PUSH_STR 6
  CALL 1
  OWN_STORE_LOCAL 0
  .local_begin 0 "f"
  OWN_MOVE_LOCAL 0
  CALL 2
  PUSH_I64 0
  RET
  .local_end 0
.end
.affine_completion 0 0
.affine_completion 1 0
.parameters 1 string
.affine_completion 2 0
.parameters 2 struct
.affine_completion 3 0
.parameters 3
