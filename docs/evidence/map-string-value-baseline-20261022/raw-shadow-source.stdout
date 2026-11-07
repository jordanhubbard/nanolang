.types 1 0 0
.entry 2

.string s0 "key"
.string s1 "kept"
.function __nanoisa_shadow_0 0 0 0 void 0
.function_purity 0
.function_affine 0
.function_effects 8
  CALL exercise
  PUSH_I64 0
  I64_EQ
  ASSERT
  RET
.end
.function __nanoisa_shadow_1 0 0 0 void 0
.function_purity 0
.function_affine 0
.function_effects 8
  CALL main
  PUSH_I64 0
  I64_EQ
  ASSERT
  RET
.end
.function __nanoisa_shadow_entry 0 0 0 int 1
  .shadow "exercise"
  CALL __nanoisa_shadow_0
  .shadow "main"
  CALL __nanoisa_shadow_1
  PUSH_I64 0
  RET
.end
.function exercise 0 1 0 int 1
.function_purity 0
.function_affine 0
.function_effects 8
  HM_NEW 5 5
  AGG_PACK 0 0 0 1
  STORE_LOCAL 0
.local_begin 0 "h"
  LOAD_LOCAL 0
  AGG_GET 0
  PUSH_STR s0
  PUSH_STR s1
  HM_SET
  POP
  LOAD_LOCAL 0
  AGG_GET 0
  PUSH_STR s0
  HM_HAS
  ASSERT
  LOAD_LOCAL 0
  AGG_GET 0
  PUSH_STR s0
  HM_GET
  PUSH_STR s1
  EQ
  ASSERT
  LOAD_LOCAL 0
  AGG_GET 0
  HM_LEN
  PUSH_I64 1
  I64_EQ
  ASSERT
  LOAD_LOCAL 0
  AGG_GET 0
  PUSH_STR s0
  HM_DELETE
  POP
  LOAD_LOCAL 0
  AGG_GET 0
  PUSH_STR s0
  HM_HAS
  BOOL_NOT
  ASSERT
  PUSH_I64 0
  RET
.local_end 0
.end
.function main 0 0 0 int 1
.function_purity 0
.function_affine 0
.function_effects 8
  TAIL_CALL exercise
.end
