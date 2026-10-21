.types 1 0 0
.entry 0
.string "Leaf"
.string "value"
.string "__nanoisa_shadow_entry"
.string "scalar"
.string "make"
.string "forward"
.string "take"
.string "main"
.string "A"
.string "B"
.string "C"
.layouts "01000000000001000000000001000000ffffffff01000000"
.ownership "010000000100000003000000060000000000000001000000ffffffff0200020001000000ffffffff01000000ffffffff05000000ffffffff03000200080000000000000001000000ffffffff05000000ffffffff01000000ffffffff01000100080000000000000008000000000000000800030001000000ffffffff0800000000000000080000000000000001000000ffffffff080000000000000001000000ffffffff01000000ffffffff080000000000000001000000ffffffff0300000001000000ffffffff01000000ffffffff01000000ffffffff01000000ffffffff"
.function __nanoisa_shadow_entry 0 0 0 int 1
  .shadow "scalar"
  PUSH_BOOL 1
  ASSERT
  .shadow "make"
  PUSH_BOOL 1
  ASSERT
  .shadow "forward"
  PUSH_BOOL 1
  ASSERT
  .shadow "take"
  PUSH_BOOL 1
  ASSERT
  .shadow "main"
  CALL 5
  PUSH_I64 0
  EQ
  ASSERT
  PUSH_I64 0
  RET
.end
.function scalar 2 2 0 int 1
  .local_begin 0 "value"
  .local_begin 1 "text"
  LOAD_LOCAL 1
  PRINT
  LOAD_LOCAL 0
  RET
  .local_end 0
  .local_end 1
.end
.function make 2 3 0 struct 1
  .local_begin 0 "value"
  .local_begin 1 "text"
  LOAD_LOCAL 1
  PRINT
  LOAD_LOCAL 0
  STORE_LOCAL 2
  LOAD_LOCAL 2
  OWN_PACK 0
  RET
  .local_end 0
  .local_end 1
.end
.function forward 1 1 0 struct 1
  .local_begin 0 "owner"
  OWN_MOVE_LOCAL 0
  RET
  .local_end 0
.end
.function take 3 8 0 int 1
  .local_begin 0 "first"
  .local_begin 1 "second"
  .local_begin 2 "last"
  OWN_MOVE_LOCAL 0
  OWN_STORE_LOCAL 3
  REGION_BEGIN
  BORROW_LOCAL_SHARED 3 3
  REF_GET 3 0
  REGION_END
  STORE_LOCAL 4
  .local_begin 4 "value"
  LOAD_LOCAL 4
  STORE_LOCAL 5
  .local_begin 5 "left"
  OWN_MOVE_LOCAL 1
  OWN_STORE_LOCAL 6
  REGION_BEGIN
  BORROW_LOCAL_SHARED 3 6
  REF_GET 3 0
  REGION_END
  STORE_LOCAL 7
  .local_begin 7 "value"
  LOAD_LOCAL 2
  PUSH_I64 3
  EQ
  ASSERT
  LOAD_LOCAL 5
  LOAD_LOCAL 7
  ADD
  OWN_UNPACK_LOCAL 3
  POP
  OWN_UNPACK_LOCAL 6
  POP
  RET
  .local_end 0
  .local_end 1
  .local_end 2
  .local_end 4
  .local_end 5
  .local_end 7
.end
.function main 0 3 0 int 1
  PUSH_I64 0
  PUSH_I64 3
  STORE_LOCAL 0
  STORE_LOCAL 1
  .local_begin 1 "index"
  nb_control_0:
  LOAD_LOCAL 1
  LOAD_LOCAL 0
  LT
  JMP_FALSE nb_control_1
  PUSH_I64 1
  PUSH_STR 8
  CALL 1
  STORE_LOCAL 2
  LOAD_LOCAL 2
  OWN_PACK 0
  PUSH_I64 2
  PUSH_STR 9
  CALL 2
  CALL 3
  PUSH_I64 3
  PUSH_STR 10
  CALL 1
  CALL 4
  PUSH_I64 3
  EQ
  ASSERT
  LOAD_LOCAL 1
  PUSH_I64 1
  ADD
  STORE_LOCAL 1
  JMP nb_control_0
  nb_control_1:
  .local_end 1
  PUSH_I64 0
  RET
.end
.affine_completion 0 0
.affine_completion 1 0
.parameters 1 int string
.affine_completion 2 0
.parameters 2 int string
.affine_completion 3 0
.parameters 3 struct
.affine_completion 4 0
.parameters 4 struct struct int
.affine_completion 5 0
.parameters 5
