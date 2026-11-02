.types 3 0 0
.entry 0
.string "Leaf"
.string "value"
.string "Inner"
.string "leaf"
.string "yes"
.string "Outer"
.string "inner"
.string "extra"
.string "__nanoisa_shadow_entry"
.string "scalar"
.string "make"
.string "relay"
.string "take"
.string "main"
.string "A"
.string "B"
.layouts "03000000000001000000000001000000ffffffff01000000000002000200000008000000000000000300000004000000ffffffff04000000000002000500000008000000010000000600000001000000ffffffff07000000"
.ownership "010000000300000003030300060000000000000001000000ffffffff0200020001000000ffffffff01000000ffffffff05000000ffffffff09000100080000000200000004000000ffffffff01000000ffffffff0800000000000000080000000000000004000000ffffffff0800000001000000080000000100000001000000ffffffff080000000200000001000100080000000200000008000000020000000d00010001000000ffffffff08000000020000000800000002000000080000000100000001000000ffffffff080000000100000001000000ffffffff0800000001000000080000000000000004000000ffffffff080000000000000004000000ffffffff080000000000000001000000ffffffff0300000001000000ffffffff01000000ffffffff01000000ffffffff0800000002000000"
.function __nanoisa_shadow_entry 0 0 0 int 1
  .shadow "scalar"
  PUSH_BOOL 1
  ASSERT
  .shadow "make"
  PUSH_BOOL 1
  ASSERT
  .shadow "relay"
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
.function make 1 9 0 struct 1
  .local_begin 0 "flag"
  PUSH_I64 7
  PUSH_STR 14
  CALL 1
  STORE_LOCAL 1
  LOAD_LOCAL 1
  OWN_PACK 0
  OWN_STORE_LOCAL 2
  .local_begin 2 "leaf"
  OWN_MOVE_LOCAL 2
  OWN_STORE_LOCAL 3
  PUSH_BOOL 1
  STORE_LOCAL 4
  OWN_MOVE_LOCAL 3
  LOAD_LOCAL 4
  OWN_PACK 1
  OWN_STORE_LOCAL 5
  .local_begin 5 "inner"
  OWN_MOVE_LOCAL 5
  OWN_STORE_LOCAL 6
  PUSH_I64 9
  PUSH_STR 15
  CALL 1
  STORE_LOCAL 7
  OWN_MOVE_LOCAL 6
  LOAD_LOCAL 7
  OWN_PACK 2
  OWN_STORE_LOCAL 8
  .local_begin 8 "owner"
  nb_control_0:
  LOAD_LOCAL 0
  JMP_FALSE nb_control_1
  OWN_MOVE_LOCAL 8
  RET
  nb_control_1:
  OWN_MOVE_LOCAL 8
  RET
  nb_control_2:
  .local_end 0
  .local_end 2
  .local_end 5
  .local_end 8
.end
.function relay 1 1 0 struct 1
  .local_begin 0 "owner"
  OWN_MOVE_LOCAL 0
  RET
  .local_end 0
.end
.function take 1 13 0 int 1
  .local_begin 0 "owner"
  OWN_MOVE_LOCAL 0
  OWN_STORE_LOCAL 1
  OWN_UNPACK_LOCAL 1
  STORE_LOCAL 3
  OWN_STORE_LOCAL 2
  OWN_MOVE_LOCAL 2
  OWN_STORE_LOCAL 4
  .local_begin 4 "inner"
  LOAD_LOCAL 3
  STORE_LOCAL 5
  .local_begin 5 "extra"
  OWN_MOVE_LOCAL 4
  OWN_STORE_LOCAL 6
  OWN_UNPACK_LOCAL 6
  STORE_LOCAL 8
  OWN_STORE_LOCAL 7
  OWN_MOVE_LOCAL 7
  OWN_STORE_LOCAL 9
  .local_begin 9 "leaf"
  LOAD_LOCAL 8
  STORE_LOCAL 10
  .local_begin 10 "yes"
  OWN_MOVE_LOCAL 9
  OWN_STORE_LOCAL 11
  REGION_BEGIN
  BORROW_LOCAL_SHARED 1 11
  REF_GET 1 0
  REGION_END
  STORE_LOCAL 12
  .local_begin 12 "value"
  LOAD_LOCAL 10
  ASSERT
  LOAD_LOCAL 12
  LOAD_LOCAL 5
  ADD
  OWN_UNPACK_LOCAL 11
  POP
  RET
  .local_end 0
  .local_end 4
  .local_end 5
  .local_end 9
  .local_end 10
  .local_end 12
.end
.function main 0 3 0 int 1
  PUSH_I64 0
  PUSH_I64 2
  STORE_LOCAL 0
  STORE_LOCAL 1
  .local_begin 1 "index"
  nb_control_3:
  LOAD_LOCAL 1
  LOAD_LOCAL 0
  LT
  JMP_FALSE nb_control_4
  LOAD_LOCAL 1
  PUSH_I64 0
  EQ
  CALL 2
  CALL 3
  OWN_STORE_LOCAL 2
  .local_begin 2 "owner"
  OWN_MOVE_LOCAL 2
  CALL 4
  PUSH_I64 16
  EQ
  ASSERT
  .local_end 2
  LOAD_LOCAL 1
  PUSH_I64 1
  ADD
  STORE_LOCAL 1
  JMP nb_control_3
  nb_control_4:
  .local_end 1
  PUSH_I64 0
  RET
.end
.affine_completion 0 0
.affine_completion 1 0
.parameters 1 int string
.affine_completion 2 0
.parameters 2 bool
.affine_completion 3 0
.parameters 3 struct
.affine_completion 4 0
.parameters 4 struct
.affine_completion 5 0
.parameters 5
