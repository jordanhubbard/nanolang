.string item "kept"
.entry main
.function before 1 1 0 array 1
PUSH_STR item
ARR_LITERAL 5 1
PUSH_I64 0
ARR_GET
LOAD_LOCAL 0
PUSH_I64 1
CALL collect
RET
.end
.function collect 3 3 0 array 1
LOAD_LOCAL 2
PUSH_I64 0
I64_GT_S
JMP_FALSE append
LOAD_LOCAL 0
LOAD_LOCAL 1
LOAD_LOCAL 2
PUSH_I64 1
I64_SUB
CALL collect
RET
append:
LOAD_LOCAL 1
LOAD_LOCAL 0
ARR_PUSH
RET
.end
.function main 0 1 0 int 1
ARR_LITERAL 5 0
CALL before
STORE_LOCAL 0
PUSH_STR item
LOAD_LOCAL 0
PUSH_I64 1
CALL collect
STORE_LOCAL 0
LOAD_LOCAL 0
ARR_LEN
PUSH_I64 2
I64_EQ
ASSERT
LOAD_LOCAL 0
PUSH_I64 1
ARR_GET
PUSH_STR item
EQ
ASSERT
PUSH_I64 0
RET
.end
.parameters 0 array
.parameters 1 string array int
