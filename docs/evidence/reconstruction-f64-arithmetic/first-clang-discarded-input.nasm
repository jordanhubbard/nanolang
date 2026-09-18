.entry main
.function main 0 0 0 int 1
PUSH_F64 bits:3ff0000000000000
CALL relay
PUSH_F64 bits:4000000000000000
CALL relay
F64_ADD
POP
PUSH_I64 0
RET
.end
.function relay 1 1 0 float 1
.parameters relay float
LOAD_LOCAL 0
RET
.end
