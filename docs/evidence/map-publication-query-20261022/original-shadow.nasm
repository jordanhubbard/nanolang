.types 1 0 0
.entry 4

.string s0 "first"
.string s1 "second"
.function __nanoisa_shadow_0 0 1 0 void 0
.function_purity 0
.function_affine 0
.function_effects 8
  CALL collect
  STORE_LOCAL 0
.local_begin 0 "result"
  LOAD_LOCAL 0
  AGG_GET 2
  ASSERT
.local_end 0
  RET
.end
.function __nanoisa_shadow_1 0 2 0 void 0
.function_purity 0
.function_affine 0
.function_effects 8
  CALL collect
  CALL forward
  STORE_LOCAL 0
.local_begin 0 "result"
  LOAD_LOCAL 0
  AGG_GET 1
  STORE_LOCAL 1
.local_begin 1 "values"
  LOAD_LOCAL 1
  PUSH_STR s0
  HM_GET
  PUSH_I64 1
  I64_EQ
  ASSERT
.local_end 0
.local_end 1
  RET
.end
.function __nanoisa_shadow_2 0 2 0 void 0
.function_purity 0
.function_affine 0
.function_effects 8
  CALL collect
  STORE_LOCAL 0
.local_begin 0 "result"
  LOAD_LOCAL 0
  PUSH_STR s1
  PUSH_I64 2
  CALL remember
  LOAD_LOCAL 0
  AGG_GET 1
  STORE_LOCAL 1
.local_begin 1 "values"
  LOAD_LOCAL 1
  PUSH_STR s1
  HM_GET
  PUSH_I64 2
  I64_EQ
  ASSERT
.local_end 0
.local_end 1
  RET
.end
.function __nanoisa_shadow_3 0 0 0 void 0
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
  .shadow "collect"
  CALL __nanoisa_shadow_0
  .shadow "forward"
  CALL __nanoisa_shadow_1
  .shadow "remember"
  CALL __nanoisa_shadow_2
  .shadow "main"
  CALL __nanoisa_shadow_3
  PUSH_I64 0
  RET
.end
.function collect 0 1 0 struct 1
.function_purity 0
.function_affine 0
.function_effects 8
  HM_NEW 5 1
  STORE_LOCAL 0
.local_begin 0 "values"
  LOAD_LOCAL 0
  PUSH_STR s0
  PUSH_I64 1
  HM_SET
  POP
  PUSH_STR s0
  ARR_LITERAL 5 1
  LOAD_LOCAL 0
  PUSH_BOOL 1
  AGG_PACK 0 0 0 3
  RET
.local_end 0
.end
.function forward 1 1 0 struct 1
.function_purity 0
.function_affine 0
.function_effects 2
.local_begin 0 "result"
  LOAD_LOCAL 0
  RET
.end
.function remember 3 4 0 void 0
.function_purity 0
.function_affine 0
.function_effects 10
.local_begin 0 "result"
.local_begin 1 "key"
.local_begin 2 "value"
  LOAD_LOCAL 0
  AGG_GET 1
  STORE_LOCAL 3
.local_begin 3 "values"
  LOAD_LOCAL 3
  LOAD_LOCAL 1
  LOAD_LOCAL 2
  HM_SET
  POP
.local_end 3
  RET
.end
.function main 0 3 0 int 1
.function_purity 0
.function_affine 0
.function_effects 10
  CALL collect
  STORE_LOCAL 0
.local_begin 0 "original"
  LOAD_LOCAL 0
  CALL forward
  STORE_LOCAL 1
.local_begin 1 "alias"
  LOAD_LOCAL 1
  PUSH_STR s1
  PUSH_I64 2
  CALL remember
  LOAD_LOCAL 0
  AGG_GET 2
  ASSERT
  LOAD_LOCAL 0
  AGG_GET 0
  PUSH_I64 0
  ARR_GET
  PUSH_STR s0
  EQ
  ASSERT
  LOAD_LOCAL 0
  AGG_GET 1
  STORE_LOCAL 2
.local_begin 2 "values"
  LOAD_LOCAL 2
  PUSH_STR s1
  HM_HAS
  ASSERT
  LOAD_LOCAL 2
  PUSH_STR s1
  HM_GET
  PUSH_I64 2
  I64_EQ
  ASSERT
  LOAD_LOCAL 1
  AGG_GET 1
  HM_LEN
  PUSH_I64 2
  I64_EQ
  ASSERT
  PUSH_I64 0
  RET
.local_end 0
.local_end 1
.local_end 2
.end
