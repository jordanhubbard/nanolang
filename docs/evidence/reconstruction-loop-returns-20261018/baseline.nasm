.entry main
.function main 0 0 0 int 1
PUSH_I64 0
CALL walk
RET
.end
.function walk 1 1 0 int 1
loop:
LOAD_LOCAL 0
PUSH_I64 3
I64_LT_S
JMP_FALSE done
PUSH_I64 7
RET
JMP loop
done:
LOAD_LOCAL 0
RET
.end
.parameters walk int
