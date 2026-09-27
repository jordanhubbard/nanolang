.string text "kept"
.entry main
.function main 0 1 0 int 1
PUSH_STR text
AGG_PACK 0 0 0 1
ARR_LITERAL 8 1
AGG_PACK 0 0 0 1
STORE_LOCAL 0
LOAD_LOCAL 0
CALL project
STORE_LOCAL 0
LOAD_LOCAL 0
CALL rewrite
POP
PUSH_STR text
ARR_LITERAL 5 1
PUSH_I64 0
ARR_GET
CALL length
POP
PUSH_I64 0
RET
.end
.function project 1 3 0 struct 1
LOAD_LOCAL 0
AGG_GET 0
PUSH_I64 0
ARR_GET
AGG_GET 0
STORE_LOCAL 1
LOAD_LOCAL 1
CALL length
POP
LOAD_LOCAL 1
CALL length
PUSH_I64 4
EQ
ASSERT
LOAD_LOCAL 0
RET
.end
.function rewrite 1 1 0 struct 1
LOAD_LOCAL 0
AGG_GET 0
PUSH_I64 0
PUSH_STR text
CALL identity
AGG_PACK 0 0 0 1
ARR_SET
AGG_PACK 0 0 0 1
RET
.end
.function length 1 1 0 int 1
LOAD_LOCAL 0
STR_LEN
RET
.end
.function identity 1 1 0 string 1
LOAD_LOCAL 0
RET
.end
