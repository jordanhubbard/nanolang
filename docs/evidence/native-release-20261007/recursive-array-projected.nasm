.string empty ""
.string item "kept"
.entry main
.function is_empty 1 1 0 bool 1
LOAD_LOCAL 0
STR_LEN
PUSH_I64 0
I64_EQ
RET
.end
.function parts 0 0 0 array 1
ARR_LITERAL 5 0
RET
.end
.function before 1 1 0 array 1
PUSH_STR item
ARR_LITERAL 5 1
PUSH_I64 0
ARR_GET
LOAD_LOCAL 0
CALL collect
RET
.end
.function collect 2 5 0 array 1
LOAD_LOCAL 0
CALL is_empty
JMP_FALSE work
LOAD_LOCAL 1
RET
work:
CALL parts
STORE_LOCAL 2
LOAD_LOCAL 1
STORE_LOCAL 3
PUSH_I64 0
STORE_LOCAL 4
loop:
LOAD_LOCAL 4
LOAD_LOCAL 2
ARR_LEN
I64_LT_S
JMP_FALSE done
LOAD_LOCAL 2
LOAD_LOCAL 4
ARR_GET
LOAD_LOCAL 3
CALL collect
STORE_LOCAL 3
LOAD_LOCAL 4
PUSH_I64 1
I64_ADD
STORE_LOCAL 4
JMP loop
done:
LOAD_LOCAL 3
LOAD_LOCAL 0
ARR_PUSH
RET
.end
.function main 0 1 0 int 1
ARR_LITERAL 5 0
STORE_LOCAL 0
PUSH_STR item
LOAD_LOCAL 0
CALL collect
STORE_LOCAL 0
PUSH_STR item
LOAD_LOCAL 0
CALL collect
STORE_LOCAL 0
LOAD_LOCAL 0
ARR_LEN
PUSH_I64 2
I64_EQ
ASSERT
PUSH_I64 0
RET
.end
.parameters 0 string
.parameters 2 array
.parameters 3 string array
