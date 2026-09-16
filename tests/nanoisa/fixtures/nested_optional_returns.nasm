.string text "present"
.types 2 0 0
.entry main
.function main 0 0 0 int 1
  PUSH_BOOL 1
  CALL choose
  AGG_GET 0
  AGG_GET 0
  PUSH_STR text
  EQ
  ASSERT
  PUSH_BOOL 0
  CALL choose
  AGG_GET 0
  AGG_GET 0
  TYPE_CHECK 0
  ASSERT
  PUSH_I64 0
  RET
.end
.function choose 1 1 0 struct 1
  LOAD_LOCAL 0
  JMP_FALSE absent
  TAIL_CALL present
absent:
  TAIL_CALL missing
.end
.function present 0 0 0 struct 1
  PUSH_STR text
  AGG_PACK 0 0 0 1
  AGG_PACK 0 1 0 1
  RET
.end
.function missing 0 0 0 struct 1
  LOAD_GLOBAL 0
  AGG_PACK 0 0 0 1
  AGG_PACK 0 1 0 1
  RET
.end
