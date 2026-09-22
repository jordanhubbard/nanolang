/* I check source evaluator negation with independently written endpoints. */
#define main nano_full_eval_test_main
#include "test_eval.c"
#undef main

int main(void) {
    const int64_t input[] = {INT64_MIN, -INT64_MAX, -1, 0, 1, INT64_MAX};
    const int64_t expected[] = {INT64_MIN, INT64_MAX, 1, 0, -1, -INT64_MAX};
    const char *source =
        "fn single(x:int)->int{return (- x)}\n"
        "shadow single {assert (== (single 1) -1)}\n"
        "fn pair(a:int,b:int)->int{return (- b)}\n"
        "shadow pair {assert (== (pair 3 1) -1)}\n"
        "fn vector(xs:array<int>)->array<int>{return (- xs)}\n"
        "shadow vector {assert (== (at (vector [1]) 0) -1)}\n"
        "fn mapped(xs:array<int>)->array<int>{return (map xs single)}\n"
        "shadow mapped {assert (== (at (mapped [1]) 0) -1)}\n"
        "fn reduced(xs:array<int>)->int{return (reduce xs 0 pair)}\n"
        "shadow reduced {assert (== (reduced [1]) -1)}\n"
        "fn main()->int{return 0}\n";
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx, source));
    for (unsigned i=0;i<sizeof input/sizeof input[0];++i) {
        Value scalar=create_int(input[i]);
        Value result=call_function("single",&scalar,1,ctx.env);
        ASSERT(result.type==VAL_INT); ASSERT_EQ(result.as.int_val,expected[i]);
        ASSERT_EQ(scalar.as.int_val,input[i]);
        Value fixed=create_array(VAL_INT,1,1);
        ((long long *)fixed.as.array_val->data)[0]=input[i];
        result=call_function("vector",&fixed,1,ctx.env);
        ASSERT(result.type==VAL_ARRAY);
        ASSERT_EQ(((long long *)result.as.array_val->data)[0],expected[i]);
        ASSERT_EQ(((long long *)fixed.as.array_val->data)[0],input[i]);
        free(result.as.array_val->data); free(result.as.array_val);
        free(fixed.as.array_val->data); free(fixed.as.array_val);
        DynArray *dynamic=dyn_array_new(ELEM_INT);
        dynamic=dyn_array_push_int(dynamic,input[i]);
        Value array=create_void(); array.type=VAL_DYN_ARRAY; array.as.dyn_array_val=dynamic;
        const char *routes[]={"vector","mapped"};
        for (unsigned j=0;j<2;++j) {
            result=call_function(routes[j],&array,1,ctx.env);
            ASSERT(result.type==VAL_DYN_ARRAY);
            ASSERT_EQ(dyn_array_get_int(result.as.dyn_array_val,0),expected[i]);
            ASSERT_EQ(dyn_array_get_int(dynamic,0),input[i]);
            gc_release(result.as.dyn_array_val);
        }
        result=call_function("reduced",&array,1,ctx.env);
        ASSERT(result.type==VAL_INT); ASSERT_EQ(result.as.int_val,expected[i]);
        ASSERT_EQ(dyn_array_get_int(dynamic,0),input[i]);
        gc_release(dynamic);
    }
    run_ctx_free(&ctx);
    puts("I retain 30 exact integer negation results and unchanged inputs across five evaluator paths.");
    return 0;
}
