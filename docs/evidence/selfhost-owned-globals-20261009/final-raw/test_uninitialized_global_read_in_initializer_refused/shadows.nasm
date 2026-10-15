.types 1 0 0
.entry 0
.string "Handle"
.string "fd"
.string "__nanoisa_shadow_entry"
.string "read"
.string "main"
.layouts "01000000000001000000000001000000ffffffff01000000"
.ownership "030000000100000003000000030000000000000001000000ffffffff0000000001000000ffffffff0400000001000000ffffffff01000000ffffffff0800000000000000080000000000000001000000ffffffff04000000000000000100000003000100140000000200000001000000ffffffff01000000ffffffff"
.function __nanoisa_shadow_entry 0 0 0 int 1
  CALL 1
  STORE_GLOBAL 0
  PUSH_I64 7
  STORE_GLOBAL 1
  .shadow "read"
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
.function read 0 0 0 int 1
  LOAD_GLOBAL 1
  RET
.end
.function main 0 4 0 int 1
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
  PUSH_I64 7
  EQ
  ASSERT
  PUSH_I64 0
  OWN_UNPACK_LOCAL 2
  POP
  RET
  .local_end 1
  .local_end 3
.end
.parameters 1
.parameters 2
