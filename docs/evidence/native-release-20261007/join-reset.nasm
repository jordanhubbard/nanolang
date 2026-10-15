.types 1 0 0
.entry main
.function choose 1 1 0 bool 1
PUSH_BOOL 0
DUP
JMP_TRUE done
POP
LOAD_LOCAL 0
PUSH_I64 0
ARR_GET
done:
RET
.end
.function main 0 0 0 int 1
PUSH_I64 7
AGG_PACK 0 0 0 1
ARR_LITERAL 8 1
STORE_GLOBAL 0
PUSH_BOOL 1
ARR_LITERAL 4 1
CALL choose
ASSERT
PUSH_I64 0
RET
.end
.parameters 0 array
