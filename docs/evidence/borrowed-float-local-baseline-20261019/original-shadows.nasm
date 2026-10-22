.types 2 0 0
.entry 0
.string "Leaf"
.string "value"
.string "enabled"
.string "Pair"
.string "left"
.string "right"
.string "__nanoisa_shadow_entry"
.string "step"
.layouts "02000000000002000000000001000000ffffffff0100000004000000ffffffff020000000000020003000000080000000000000004000000080000000000000005000000"
.ownership "020000000200000003030000020000001800000001000000ffffffff01000000ffffffff04000000ffffffff080000000000000001000000ffffffff04000000ffffffff080000000000000008000000000000000800000000000000080000000100000001000000ffffffff01000000ffffffff01000000ffffffff03000000ffffffff08000000010000000800000000000000080000000000000008000000000000000800000000000000080000000000000001000000ffffffff04000000ffffffff080000000000000001000000ffffffff04000000ffffffff0300010001000000ffffffff080200000000000001000000ffffffff01000000ffffffff0200000001000000000000000100000001000000"
.function __nanoisa_shadow_entry 0 24 0 int 1
  .shadow "step"
  PUSH_I64 0
  STORE_LOCAL 0
  PUSH_BOOL 1
  STORE_LOCAL 1
  LOAD_LOCAL 0
  LOAD_LOCAL 1
  OWN_PACK 0
  OWN_STORE_LOCAL 2
  .local_begin 2 "left"
  PUSH_I64 100
  STORE_LOCAL 3
  PUSH_BOOL 0
  STORE_LOCAL 4
  LOAD_LOCAL 3
  LOAD_LOCAL 4
  OWN_PACK 0
  OWN_STORE_LOCAL 5
  .local_begin 5 "right"
  OWN_MOVE_LOCAL 2
  OWN_STORE_LOCAL 6
  OWN_MOVE_LOCAL 5
  OWN_STORE_LOCAL 7
  OWN_MOVE_LOCAL 6
  OWN_MOVE_LOCAL 7
  OWN_PACK 1
  OWN_STORE_LOCAL 8
  .local_begin 8 "root"
  PUSH_I64 0
  STORE_LOCAL 9
  .local_begin 9 "i"
  PUSH_I64 0
  STORE_LOCAL 10
  .local_begin 10 "j"
  PUSH_I64 0
  STORE_LOCAL 11
  .local_begin 11 "total"
  nb_control_0:
  LOAD_LOCAL 9
  PUSH_I64 3
  LT
  JMP_FALSE nb_control_1
  PUSH_I64 0
  STORE_LOCAL 10
  nb_control_3:
  LOAD_LOCAL 10
  PUSH_I64 2
  LT
  JMP_FALSE nb_control_4
  nb_control_6:
  LOAD_LOCAL 10
  PUSH_I64 0
  EQ
  JMP_FALSE nb_control_7
  LOAD_LOCAL 11
  REGION_BEGIN
  BORROW_PATH_EXCLUSIVE 0 8 0
  CALL_REF 1 0
  REGION_END
  ADD
  STORE_LOCAL 11
  JMP nb_control_8
  nb_control_7:
  LOAD_LOCAL 11
  REGION_BEGIN
  BORROW_PATH_EXCLUSIVE 0 8 1
  CALL_REF 1 0
  REGION_END
  ADD
  STORE_LOCAL 11
  nb_control_8:
  LOAD_LOCAL 10
  PUSH_I64 1
  ADD
  STORE_LOCAL 10
  JMP nb_control_3
  nb_control_4:
  LOAD_LOCAL 9
  PUSH_I64 1
  ADD
  STORE_LOCAL 9
  JMP nb_control_0
  nb_control_1:
  LOAD_LOCAL 11
  PUSH_I64 9
  EQ
  ASSERT
  nb_control_9:
  PUSH_BOOL 0
  JMP_FALSE nb_control_10
  REGION_BEGIN
  BORROW_PATH_EXCLUSIVE 0 8 0
  CALL_REF 1 0
  REGION_END
  POP
  JMP nb_control_9
  nb_control_10:
  nb_control_12:
  PUSH_BOOL 0
  JMP_FALSE nb_control_13
  PUSH_BOOL 0
  ASSERT
  JMP nb_control_14
  nb_control_13:
  LOAD_LOCAL 11
  PUSH_I64 9
  EQ
  ASSERT
  nb_control_14:
  nb_control_15:
  PUSH_BOOL 1
  JMP_FALSE nb_control_16
  REGION_BEGIN
  BORROW_PATH_SHARED 0 8 0
  REF_GET 0 0
  REGION_END
  PUSH_I64 6
  EQ
  ASSERT
  JMP nb_control_17
  nb_control_16:
  nb_control_17:
  nb_control_18:
  REGION_BEGIN
  BORROW_PATH_EXCLUSIVE 0 8 0
  CALL_REF 1 0
  REGION_END
  PUSH_I64 10
  LT
  JMP_FALSE nb_control_19
  LOAD_LOCAL 11
  PUSH_I64 1
  ADD
  STORE_LOCAL 11
  JMP nb_control_18
  nb_control_19:
  REGION_BEGIN
  BORROW_PATH_SHARED 0 8 0
  REF_GET 0 0
  REGION_END
  PUSH_I64 10
  EQ
  ASSERT
  LOAD_LOCAL 11
  PUSH_I64 10
  EQ
  ASSERT
  nb_control_21:
  REGION_BEGIN
  BORROW_PATH_EXCLUSIVE 0 8 0
  CALL_REF 1 0
  REGION_END
  PUSH_I64 10
  GT
  JMP_FALSE nb_control_22
  PUSH_F64 2.0
  STORE_LOCAL 12
  .local_begin 12 "value"
  LOAD_LOCAL 12
  PUSH_F64 2.0
  F64_EQ
  ASSERT
  LOAD_LOCAL 11
  PUSH_I64 2
  ADD
  STORE_LOCAL 11
  .local_end 12
  JMP nb_control_23
  nb_control_22:
  PUSH_BOOL 0
  ASSERT
  nb_control_23:
  LOAD_LOCAL 11
  PUSH_I64 12
  EQ
  ASSERT
  OWN_MOVE_LOCAL 8
  OWN_STORE_LOCAL 13
  OWN_UNPACK_LOCAL 13
  OWN_STORE_LOCAL 15
  OWN_STORE_LOCAL 14
  OWN_MOVE_LOCAL 15
  OWN_STORE_LOCAL 16
  .local_begin 16 "right"
  OWN_MOVE_LOCAL 14
  OWN_STORE_LOCAL 17
  .local_begin 17 "left"
  OWN_MOVE_LOCAL 17
  OWN_STORE_LOCAL 18
  REGION_BEGIN
  BORROW_LOCAL_SHARED 0 18
  REF_GET 0 0
  REGION_END
  STORE_LOCAL 19
  .local_begin 19 "value"
  REGION_BEGIN
  BORROW_LOCAL_SHARED 0 18
  REF_GET 0 1
  REGION_END
  STORE_LOCAL 20
  .local_begin 20 "enabled"
  LOAD_LOCAL 19
  PUSH_I64 12
  EQ
  ASSERT
  LOAD_LOCAL 20
  ASSERT
  OWN_MOVE_LOCAL 16
  OWN_STORE_LOCAL 21
  REGION_BEGIN
  BORROW_LOCAL_SHARED 0 21
  REF_GET 0 0
  REGION_END
  STORE_LOCAL 22
  .local_begin 22 "value"
  REGION_BEGIN
  BORROW_LOCAL_SHARED 0 21
  REF_GET 0 1
  REGION_END
  STORE_LOCAL 23
  .local_begin 23 "enabled"
  LOAD_LOCAL 22
  PUSH_I64 100
  EQ
  ASSERT
  LOAD_LOCAL 23
  NOT
  ASSERT
  OWN_UNPACK_LOCAL 18
  POP
  POP
  OWN_UNPACK_LOCAL 21
  POP
  POP
  .local_end 2
  .local_end 5
  .local_end 8
  .local_end 9
  .local_end 10
  .local_end 11
  .local_end 16
  .local_end 17
  .local_end 19
  .local_end 20
  .local_end 22
  .local_end 23
  .shadow "main"
  nb_control_24:
  PUSH_BOOL 1
  JMP_FALSE nb_control_25
  PUSH_BOOL 1
  ASSERT
  JMP nb_control_26
  nb_control_25:
  PUSH_BOOL 0
  ASSERT
  nb_control_26:
  PUSH_I64 0
  RET
.end
.function step 1 3 0 int 1
  .local_begin 0 "view"
  PUSH_I64 0
  STORE_LOCAL 1
  .local_begin 1 "result"
  PUSH_I64 0
  STORE_LOCAL 2
  .local_begin 2 "index"
  nb_control_27:
  REF_GET 0 1
  JMP_FALSE nb_control_28
  nb_control_30:
  LOAD_LOCAL 2
  PUSH_I64 2
  LT
  JMP_FALSE nb_control_31
  REF_GET 0 0
  PUSH_I64 1
  ADD
  REF_SET 0 0
  LOAD_LOCAL 2
  PUSH_I64 1
  ADD
  STORE_LOCAL 2
  JMP nb_control_30
  nb_control_31:
  REF_GET 0 0
  STORE_LOCAL 1
  JMP nb_control_29
  nb_control_28:
  PUSH_I64 -1
  STORE_LOCAL 1
  nb_control_29:
  LOAD_LOCAL 1
  RET
  .local_end 0
  .local_end 1
  .local_end 2
.end
.affine_completion 0 0
.affine_completion 1 0
.parameters 1 struct
