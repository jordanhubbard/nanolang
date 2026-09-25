#include "nanolang.h"
Value named_void(void) {
    Value v;
    v.type=VAL_VOID;v.is_return=false;v.is_break=false;v.is_continue=false;
    return v;
}
Value literal_void(void) { return (Value){.type=VAL_VOID}; }
Value named_int(int64_t n) {
    Value v;
    v.type=VAL_INT;v.is_return=false;v.is_break=false;v.is_continue=false;v.as.int_val=n;
    return v;
}
Value literal_int(int64_t n) { return (Value){.type=VAL_INT,.as.int_val=n}; }
