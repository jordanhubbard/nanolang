/* I compare actual evaluator routes with independent integer endpoints. */
#define main nano_full_eval_test_main
#include "test_eval.c"
#undef main

static Value fixture_array(int64_t value, int dynamic) {
    if (dynamic) {
        DynArray *a=dyn_array_new(ELEM_INT);
        a=dyn_array_push_int(a,value);
        Value result=create_void(); result.type=VAL_DYN_ARRAY; result.as.dyn_array_val=a;
        return result;
    }
    Value result=create_array(VAL_INT,1,1);
    ((long long *)result.as.array_val->data)[0]=value;
    return result;
}
static int64_t fixture_element(Value value) {
    ASSERT(value.type==VAL_ARRAY || value.type==VAL_DYN_ARRAY);
    return value.type==VAL_ARRAY ? ((long long *)value.as.array_val->data)[0]
        : dyn_array_get_int(value.as.dyn_array_val,0);
}
static void fixture_release(Value value) {
    if (value.type==VAL_DYN_ARRAY) gc_release(value.as.dyn_array_val);
    else { free(value.as.array_val->data); free(value.as.array_val); }
}
static void fixture_release_result(Value value) {
    /* Fixed results borrow ctx; dynamic results keep their existing GC contract. */
    if (value.type==VAL_DYN_ARRAY) gc_release(value.as.dyn_array_val);
}
int main(void) {
    const struct { int64_t a,b,result[5]; } cases[]={
        {INT64_MIN,-1,{INT64_MAX,INT64_MIN+1,INT64_MIN,INT64_MIN,0}},
        {INT64_MAX,1,{INT64_MIN,INT64_MAX-1,INT64_MAX,INT64_MAX,0}},
        {INT64_MAX,2,{INT64_MIN+1,INT64_MAX-2,-2,INT64_C(4611686018427387903),1}},
        {INT64_MIN,2,{INT64_MIN+2,INT64_MAX-1,0,-INT64_C(4611686018427387904),0}},
        {-7,2,{-5,-9,-14,-3,-1}}, {7,-2,{5,9,-14,-3,1}},
        {7,0,{7,7,0,0,0}}, {INT64_MIN,0,{INT64_MIN,INT64_MIN,0,0,0}},
        {-1,-1,{-2,0,1,1,0}}
    };
    const char operators[]="+-*/%";
    unsigned observations=0;
    for (unsigned row=0;row<sizeof cases/sizeof cases[0];++row) {
        for (unsigned op=0;op<5;++op) {
            char source[4096]; char c=operators[op];
            snprintf(source,sizeof source,
                "fn pair(a:int,b:int)->int{return (%c a b)}\n"
                "shadow pair {assert (== (pair 0 1) %d)}\n"
                "fn single(a:int)->int{return (%c a %lld)}\n"
                "shadow single {assert (== (single 0) (pair 0 %lld))}\n"
                "fn arrays(a:array<int>,b:array<int>)->array<int>{return (%c a b)}\n"
                "shadow arrays {assert (== (array_length (arrays [1] [1])) 1)}\n"
                "fn right(a:array<int>,b:int)->array<int>{return (%c a b)}\n"
                "shadow right {assert (== (array_length (right [1] 1)) 1)}\n"
                "fn left(a:int,b:array<int>)->array<int>{return (%c a b)}\n"
                "shadow left {assert (== (array_length (left 1 [1])) 1)}\n"
                "fn mapped(a:array<int>)->array<int>{return (map a single)}\n"
                "shadow mapped {assert (== (array_length (mapped [1])) 1)}\n"
                "fn reduced(a:int,b:array<int>)->int{return (reduce b a pair)}\n"
                "shadow reduced {assert (== (reduced 1 [1]) (pair 1 1))}\n"
                "fn main()->int{return 0}\n",
                c,op==0?1:op==1?-1:0,c,(long long)cases[row].b,(long long)cases[row].b,c,c,c);
            RunCtx ctx; ASSERT(run_ctx_init(&ctx,source));
            Value args[]={create_int(cases[row].a),create_int(cases[row].b)};
            Value result=call_function("pair",args,2,ctx.env);
            ASSERT(result.type==VAL_INT); ASSERT_EQ(result.as.int_val,cases[row].result[op]); ++observations;
            for (int dynamic=0;dynamic<2;++dynamic) {
                Value a=fixture_array(cases[row].a,dynamic), b=fixture_array(cases[row].b,dynamic);
                Value pair[]={a,b}; result=call_function("arrays",pair,2,ctx.env);
                ASSERT_EQ(fixture_element(result),cases[row].result[op]); fixture_release_result(result); ++observations;
                pair[0]=a; pair[1]=args[1]; result=call_function("right",pair,2,ctx.env);
                ASSERT_EQ(fixture_element(result),cases[row].result[op]); fixture_release_result(result); ++observations;
                pair[0]=args[0]; pair[1]=b; result=call_function("left",pair,2,ctx.env);
                ASSERT_EQ(fixture_element(result),cases[row].result[op]); fixture_release_result(result); ++observations;
                if (dynamic) {
                    result=call_function("mapped",&a,1,ctx.env);
                    ASSERT_EQ(fixture_element(result),cases[row].result[op]); fixture_release_result(result); ++observations;
                    result=call_function("reduced",pair,2,ctx.env);
                    ASSERT(result.type==VAL_INT); ASSERT_EQ(result.as.int_val,cases[row].result[op]); ++observations;
                }
                ASSERT_EQ(fixture_element(a),cases[row].a); ASSERT_EQ(fixture_element(b),cases[row].b);
                fixture_release(a); fixture_release(b);
            }
            run_ctx_free(&ctx);
        }
    }
    ASSERT_EQ(observations,405);
    puts("I retain 405 exact binary integer results across nine evaluator routes with unchanged inputs.");
    return 0;
}
