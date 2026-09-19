/* I exercise bit identities through the tree interpreter's public calls. */
#define main nano_full_eval_test_main
#include "test_eval.c"
#undef main
#include "../src/binary64_bits.h"
#include <fenv.h>
int main(void) {
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,
        "fn copy(bits:int)->int{return (float_to_bits (float_from_bits bits))}\n"
        "fn main()->int{return 0}\n"));
    uint64_t patterns[] = {0, UINT64_C(0x8000000000000000), 1,
        UINT64_C(0x8000000000000001), UINT64_C(0x7fefffffffffffff),
        UINT64_C(0x7ff0000000000000), UINT64_C(0xfff0000000000000),
        UINT64_C(0x7ff0000000000001), UINT64_C(0xfff0000000000042),
        UINT64_C(0x7ff8000000000042), UINT64_C(0xfff8000000000042)};
    for (unsigned i = 0; i < sizeof patterns / sizeof patterns[0]; ++i) {
        uint64_t bits = patterns[i];
        int64_t signed_bits = bits <= INT64_MAX ? (int64_t)bits :
                             -1 - (int64_t)(UINT64_MAX - bits);
        Value arg = create_int(signed_bits);
        ASSERT(feclearexcept(FE_ALL_EXCEPT) == 0);
        Value result = call_function("copy", &arg, 1, ctx.env);
        ASSERT(fetestexcept(FE_ALL_EXCEPT) == 0);
        ASSERT(result.type == VAL_INT);
        ASSERT_EQ(result.as.int_val, signed_bits);
    }
    run_ctx_free(&ctx);
    puts("I retained 11 interpreter bit patterns.");
    return 0;
}
