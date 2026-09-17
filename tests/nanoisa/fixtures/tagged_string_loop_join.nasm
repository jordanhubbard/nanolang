.entry main
.string key "key"
.string text "text"
.function main 0 1 0 int 1
PUSH_I64 0
STORE_LOCAL 0
PUSH_STR text
loop:
POP
LOAD_LOCAL 0
PUSH_I64 1
I64_ADD
DUP
STORE_LOCAL 0
PUSH_I64 2
I64_LT_S
JMP_FALSE done
HM_NEW 5 5
PUSH_STR key
PUSH_STR text
HM_SET
PUSH_STR key
HM_GET
JMP loop
done:
PUSH_I64 0
RET
.end
