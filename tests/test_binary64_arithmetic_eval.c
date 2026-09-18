/* I exercise both optimized callback evaluators with actual dynamic arrays. */
#define main nano_full_eval_test_main
#include "test_eval.c"
#undef main
#include "../src/binary64_bits.h"
int main(void) {
    const char operations[] = "+-*/";
    for (unsigned op = 0; op < sizeof operations - 1; ++op) {
        char source[2048];
        snprintf(source, sizeof source,
            "fn single(x:float)->float{return (%c x 1.0)}\n"
            "shadow single {assert true}\n"
            "fn pair(a:float,b:float)->float{return (%c a b)}\n"
            "shadow pair {assert true}\n"
            "fn mapped(xs:array<float>)->int{let ys:array<float> =(map xs single) return (float_to_bits (at ys 0))}\n"
            "shadow mapped {assert true}\n"
            "fn reduced(xs:array<float>)->int{return (float_to_bits (reduce xs 1.0 pair))}\n"
            "shadow reduced {assert true}\n"
            "fn zero(x:float)->float{return (/ x 0.0)}\n"
            "shadow zero {assert true}\n"
            "fn total(xs:array<float>)->int{let ys:array<float> =(map xs zero) return (float_to_bits (at ys 0))}\n"
            "shadow total {assert true}\n"
            "fn main()->int{return 0}\n", operations[op], operations[op]);
        RunCtx ctx;
        ASSERT(run_ctx_init(&ctx, source));
        uint64_t patterns[] = {UINT64_C(0x7ff0000000000001), UINT64_C(0xfff8000000000042)};
        for (unsigned i = 0; i < sizeof patterns / sizeof patterns[0]; ++i) {
            double value;
            memcpy(&value, &patterns[i], sizeof value);
            DynArray *dynamic = dyn_array_new(ELEM_FLOAT);
            dynamic = dyn_array_push_float(dynamic, value);
            Value array = create_void();
            array.type = VAL_DYN_ARRAY;
            array.as.dyn_array_val = dynamic;
            Value mapped = call_function("mapped", &array, 1, ctx.env);
            ASSERT(mapped.type == VAL_INT);
            ASSERT_EQ(mapped.as.int_val, INT64_C(9221120237041090560));
            Value reduced = call_function("reduced", &array, 1, ctx.env);
            ASSERT(reduced.type == VAL_INT);
            ASSERT_EQ(reduced.as.int_val, INT64_C(9221120237041090560));
            Value zero = call_function("total", &array, 1, ctx.env);
            ASSERT(zero.type == VAL_INT);
            ASSERT_EQ(zero.as.int_val, 0);
            uint64_t after;
            value = dyn_array_get_float(dynamic, 0);
            memcpy(&after, &value, sizeof after);
            ASSERT(after == patterns[i]);
        }
        run_ctx_free(&ctx);
    }
    puts("I retained 24 optimized callback results and 8 input patterns.");
    return 0;
}
