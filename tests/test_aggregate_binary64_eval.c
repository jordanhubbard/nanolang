/* I exercise both evaluator array representations, including legacy VAL_ARRAY. */
#define main nano_full_eval_test_main
#include "test_eval.c"
#undef main
#include "../src/binary64_bits.h"
typedef struct { unsigned operation; uint64_t left, right, expected; } ArrayCase;
static const ArrayCase cases[] = {
    {0, UINT64_C(0x7ff0000000000001), UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff8000000000000)},
    {0, UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff0000000000001), UINT64_C(0x7ff8000000000000)},
    {0, UINT64_C(0xfff0000000000042), UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff8000000000000)},
    {0, UINT64_C(0x3ff0000000000000), UINT64_C(0xfff0000000000042), UINT64_C(0x7ff8000000000000)},
    {0, UINT64_C(0x7ff8123456789abc), UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff8000000000000)},
    {0, UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff8123456789abc), UINT64_C(0x7ff8000000000000)},
    {0, UINT64_C(0xfff8000000001234), UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff8000000000000)},
    {0, UINT64_C(0x3ff0000000000000), UINT64_C(0xfff8000000001234), UINT64_C(0x7ff8000000000000)},
    {1, UINT64_C(0x7ff0000000000001), UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff8000000000000)},
    {1, UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff0000000000001), UINT64_C(0x7ff8000000000000)},
    {1, UINT64_C(0xfff0000000000042), UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff8000000000000)},
    {1, UINT64_C(0x3ff0000000000000), UINT64_C(0xfff0000000000042), UINT64_C(0x7ff8000000000000)},
    {1, UINT64_C(0x7ff8123456789abc), UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff8000000000000)},
    {1, UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff8123456789abc), UINT64_C(0x7ff8000000000000)},
    {1, UINT64_C(0xfff8000000001234), UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff8000000000000)},
    {1, UINT64_C(0x3ff0000000000000), UINT64_C(0xfff8000000001234), UINT64_C(0x7ff8000000000000)},
    {2, UINT64_C(0x7ff0000000000001), UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff8000000000000)},
    {2, UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff0000000000001), UINT64_C(0x7ff8000000000000)},
    {2, UINT64_C(0xfff0000000000042), UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff8000000000000)},
    {2, UINT64_C(0x3ff0000000000000), UINT64_C(0xfff0000000000042), UINT64_C(0x7ff8000000000000)},
    {2, UINT64_C(0x7ff8123456789abc), UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff8000000000000)},
    {2, UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff8123456789abc), UINT64_C(0x7ff8000000000000)},
    {2, UINT64_C(0xfff8000000001234), UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff8000000000000)},
    {2, UINT64_C(0x3ff0000000000000), UINT64_C(0xfff8000000001234), UINT64_C(0x7ff8000000000000)},
    {3, UINT64_C(0x7ff0000000000001), UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff8000000000000)},
    {3, UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff0000000000001), UINT64_C(0x7ff8000000000000)},
    {3, UINT64_C(0xfff0000000000042), UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff8000000000000)},
    {3, UINT64_C(0x3ff0000000000000), UINT64_C(0xfff0000000000042), UINT64_C(0x7ff8000000000000)},
    {3, UINT64_C(0x7ff8123456789abc), UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff8000000000000)},
    {3, UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff8123456789abc), UINT64_C(0x7ff8000000000000)},
    {3, UINT64_C(0xfff8000000001234), UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff8000000000000)},
    {3, UINT64_C(0x3ff0000000000000), UINT64_C(0xfff8000000001234), UINT64_C(0x7ff8000000000000)},
    {3, UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000)},
    {3, UINT64_C(0x0000000000000000), UINT64_C(0x8000000000000000), UINT64_C(0x0000000000000000)},
    {3, UINT64_C(0x8000000000000000), UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000)},
    {3, UINT64_C(0x8000000000000000), UINT64_C(0x8000000000000000), UINT64_C(0x0000000000000000)},
    {3, UINT64_C(0x3ff0000000000000), UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000)},
    {3, UINT64_C(0x3ff0000000000000), UINT64_C(0x8000000000000000), UINT64_C(0x0000000000000000)},
    {3, UINT64_C(0x7ff0000000000000), UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000)},
    {3, UINT64_C(0x7ff0000000000000), UINT64_C(0x8000000000000000), UINT64_C(0x0000000000000000)},
    {3, UINT64_C(0xfff0000000000000), UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000)},
    {3, UINT64_C(0xfff0000000000000), UINT64_C(0x8000000000000000), UINT64_C(0x0000000000000000)},
    {3, UINT64_C(0x7ff0000000000001), UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000)},
    {3, UINT64_C(0x7ff0000000000001), UINT64_C(0x8000000000000000), UINT64_C(0x0000000000000000)},
    {3, UINT64_C(0xfff0000000000042), UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000)},
    {3, UINT64_C(0xfff0000000000042), UINT64_C(0x8000000000000000), UINT64_C(0x0000000000000000)},
    {3, UINT64_C(0x7ff8123456789abc), UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000)},
    {3, UINT64_C(0x7ff8123456789abc), UINT64_C(0x8000000000000000), UINT64_C(0x0000000000000000)},
    {3, UINT64_C(0xfff8000000001234), UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000)},
    {3, UINT64_C(0xfff8000000001234), UINT64_C(0x8000000000000000), UINT64_C(0x0000000000000000)},
    {0, UINT64_C(0x7ff0000000000000), UINT64_C(0xfff0000000000000), UINT64_C(0x7ff8000000000000)},
    {1, UINT64_C(0x7ff0000000000000), UINT64_C(0x7ff0000000000000), UINT64_C(0x7ff8000000000000)},
    {2, UINT64_C(0x0000000000000000), UINT64_C(0x7ff0000000000000), UINT64_C(0x7ff8000000000000)},
    {2, UINT64_C(0x7ff0000000000000), UINT64_C(0x8000000000000000), UINT64_C(0x7ff8000000000000)},
    {3, UINT64_C(0x7ff0000000000000), UINT64_C(0x7ff0000000000000), UINT64_C(0x7ff8000000000000)},
    {0, UINT64_C(0x8000000000000000), UINT64_C(0x8000000000000000), UINT64_C(0x8000000000000000)},
    {0, UINT64_C(0x8000000000000000), UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000)},
    {1, UINT64_C(0x8000000000000000), UINT64_C(0x0000000000000000), UINT64_C(0x8000000000000000)},
    {2, UINT64_C(0x8000000000000000), UINT64_C(0x3ff0000000000000), UINT64_C(0x8000000000000000)},
    {3, UINT64_C(0x8000000000000000), UINT64_C(0x3ff0000000000000), UINT64_C(0x8000000000000000)},
    {0, UINT64_C(0x3ff0000000000000), UINT64_C(0x3ca0000000000000), UINT64_C(0x3ff0000000000000)},
    {0, UINT64_C(0x3ff0000000000001), UINT64_C(0x3ca0000000000000), UINT64_C(0x3ff0000000000002)},
    {2, UINT64_C(0x0000000000000001), UINT64_C(0x3fe0000000000000), UINT64_C(0x0000000000000000)},
    {2, UINT64_C(0x0000000000000003), UINT64_C(0x3fe0000000000000), UINT64_C(0x0000000000000002)},
    {2, UINT64_C(0x8000000000000001), UINT64_C(0x3fe0000000000000), UINT64_C(0x8000000000000000)},
    {0, UINT64_C(0x0000000000000001), UINT64_C(0x0000000000000001), UINT64_C(0x0000000000000002)},
    {0, UINT64_C(0x7fefffffffffffff), UINT64_C(0x7fefffffffffffff), UINT64_C(0x7ff0000000000000)},
    {3, UINT64_C(0x3ff0000000000000), UINT64_C(0x4008000000000000), UINT64_C(0x3fd5555555555555)},
};
static double from_bits(uint64_t bits) { double value; memcpy(&value,&bits,sizeof value); return value; }
static Value input_array(bool dynamic, double value, int length) {
    if (dynamic) {
        Value result=create_void(); result.type=VAL_DYN_ARRAY;
        result.as.dyn_array_val=dyn_array_new(ELEM_FLOAT);
        if(length) result.as.dyn_array_val=dyn_array_push_float(result.as.dyn_array_val,value);
        return result;
    }
    Value result=create_array(VAL_FLOAT,length,length);
    if(length) ((double *)result.as.array_val->data)[0]=value;
    return result;
}
static void observe(Value value, int length, uint64_t expected) {
    double actual=0;
    if(value.type==VAL_DYN_ARRAY) {
        ASSERT(dyn_array_get_elem_type(value.as.dyn_array_val)==ELEM_FLOAT);
        ASSERT(dyn_array_length(value.as.dyn_array_val)==length);
        if(length) actual=dyn_array_get_float(value.as.dyn_array_val,0);
    } else {
        ASSERT(value.type==VAL_ARRAY); ASSERT(value.as.array_val->element_type==VAL_FLOAT);
        ASSERT(value.as.array_val->length==length);
        if(length) actual=((double *)value.as.array_val->data)[0];
    }
    uint64_t bits;memcpy(&bits,&actual,sizeof bits);
    if(length) ASSERT(bits==expected);
}
int main(void) {
    const char operations[]="+-*/";
    unsigned observations=0;
    for(unsigned op=0;op<4;op++) {
        char source[2048];
        snprintf(source,sizeof source,
            "fn pair(a:array<float>,b:array<float>)->array<float>{return (%c a b)}\n"
            "shadow pair {assert (== (array_length (pair [1.0] [1.0])) 1)}\n"
            "fn left(a:float,b:array<float>)->array<float>{return (%c a b)}\n"
            "shadow left {assert (== (array_length (left 1.0 [1.0])) 1)}\n"
            "fn right(a:array<float>,b:float)->array<float>{return (%c a b)}\n"
            "shadow right {assert (== (array_length (right [1.0] 1.0)) 1)}\n"
            "fn main()->int{return 0}\n",operations[op],operations[op],operations[op]);
        RunCtx ctx;ASSERT(run_ctx_init(&ctx,source));
        for(unsigned dynamic=0;dynamic<2;dynamic++) {
            for(unsigned i=0;i<sizeof cases/sizeof cases[0];i++) {
                const ArrayCase *c=&cases[i];if(c->operation!=op)continue;
                double x=from_bits(c->left),y=from_bits(c->right);
                Value a=input_array(dynamic,x,1), b=input_array(dynamic,y,1);
                Value pair[]={a,b}, left[]={create_float(x),b}, right[]={a,create_float(y)};
                observe(call_function("pair",pair,2,ctx.env),1,c->expected);
                observe(call_function("left",left,2,ctx.env),1,c->expected);
                observe(call_function("right",right,2,ctx.env),1,c->expected);
                observe(a,1,c->left);observe(b,1,c->right);observations+=3;
            }
            Value empty=input_array(dynamic,0,0);
            Value pair[]={empty,empty},left[]={create_float(0),empty},right[]={empty,create_float(0)};
            observe(call_function("pair",pair,2,ctx.env),0,0);
            observe(call_function("left",left,2,ctx.env),0,0);
            observe(call_function("right",right,2,ctx.env),0,0);
        }
        run_ctx_free(&ctx);
    }
    printf("I retained %u exact array arithmetic results and 24 empty results.\n",observations);
    return 0;
}
