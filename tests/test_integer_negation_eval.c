/* I check source evaluator negation with independently written endpoints. */
#define main nano_full_eval_test_main
#include "test_eval.c"
#undef main

static int callback_snapshot_failure;
static unsigned callback_snapshot_calls;
char *nano_test_callback_name(const char *name) {
    ++callback_snapshot_calls;
    if (callback_snapshot_failure) {
        if (callback_snapshot_failure > 0) --callback_snapshot_failure;
        return NULL;
    }
    return strdup(name);
}

/* I distinguish temporary declaration callbacks from caller-owned borrowed
 * callbacks, including every synchronous alias and invalid-array return. */
static void callback_ownership(void) {
    const char *source =
        "fn identity(x:int)->int{return x}\n"
        "shadow identity{assert (== (identity 3) 3)}\n"
        "fn keep(x:int)->bool{return (> x 0)}\n"
        "shadow keep{assert (keep 3)}\n"
        "fn take(a:int,b:int)->int{return b}\n"
        "shadow take{assert (== (take 1 3) 3)}\n"
        "fn array_push(x:int)->int{return x}\n"
        "shadow array_push{assert (== (array_push 3) 3)}\n"
        "fn main()->int{return 0}\n";
    RunCtx ctx;
    ASSERT(run_ctx_init(&ctx,source));
    const char *names[]={"map","array_map","filter","array_filter","reduce","array_fold"};
    for (unsigned n=0;n<6;++n) {
        bool reduce=n>=4;
        const char *callback=n<2 ? "identity" : n<4 ? "keep" : "take";
        Type types[]={TYPE_INT,TYPE_INT};
        Value borrowed=create_function(callback,create_function_signature(types,reduce?2:1,
            n>=2 && n<4 ? TYPE_BOOL : TYPE_INT));
        env_define_var(ctx.env,"saved",TYPE_FUNCTION,false,borrowed);
        Value shadowed=create_function(callback,copy_function_signature(borrowed.as.function_val.signature));
        env_define_var(ctx.env,"array_push",TYPE_FUNCTION,false,shadowed);
        env_define_var(ctx.env,"array_push",TYPE_FUNCTION,false,create_void());
        env_get_var(ctx.env,"array_push")->def_line=1;
        DynArray *input=dyn_array_new(ELEM_INT);
        input=dyn_array_push_int(input,3);
        Value array=create_void(); array.type=VAL_DYN_ARRAY; array.as.dyn_array_val=input;
        env_define_var(ctx.env,"input",TYPE_ARRAY,false,array);
        ASTNode receiver={0},initial={0},function={0},call={0};
        receiver.type=AST_IDENTIFIER; receiver.as.identifier="input";
        initial.type=AST_NUMBER; initial.as.number=0;
        function.type=AST_IDENTIFIER;
        ASTNode *arguments[]={&receiver,reduce?&initial:&function,&function};
        call.type=AST_CALL; call.as.call.name=(char *)names[n];
        call.as.call.args=arguments; call.as.call.arg_count=reduce?3:2;
        for (int saved=0;saved<3;++saved) {
            function.as.identifier=(char *)(saved==2?"array_push":saved?"saved":callback);
            Value result=repl_eval_node(&call,ctx.env);
            if (reduce) { ASSERT(result.type==VAL_INT); ASSERT_EQ(result.as.int_val,3); }
            else { ASSERT(result.type==VAL_DYN_ARRAY); ASSERT_EQ(dyn_array_get_int(result.as.dyn_array_val,0),3); gc_release(result.as.dyn_array_val); }
            ASSERT(strcmp(borrowed.as.function_val.function_name,callback)==0);
            ASSERT_EQ(borrowed.as.function_val.signature->param_count,reduce?2:1);
            receiver.type=AST_NUMBER; receiver.as.number=7;
            suppress_stderr(); result=repl_eval_node(&call,ctx.env); restore_stderr();
            ASSERT(result.type==VAL_VOID);
            receiver.type=AST_IDENTIFIER; receiver.as.identifier="input";
        }
        /* The Environment owns saved; I retain ownership of the array input. */
        env_get_var(ctx.env,"input")->value=create_void();
        gc_release(input);
    }
    run_ctx_free(&ctx);
    puts("I preserve owned and borrowed callbacks across six aliases and error returns.");
}

/* I keep one callback identity for the entire loop even when its body
 * replaces the global binding. I refuse the one snapshot allocation before
 * running any callback, then recover independently with unchanged inputs. */
static void callback_mutation_and_failure(void) {
    const char *names[]={"map","array_map","filter","array_filter","reduce","array_fold"};
    for (unsigned n=0;n<6;++n) for (int borrowed=0;borrowed<2;++borrowed)
    for (int persistent=0;persistent<2;++persistent) {
        bool reduce=n>=4, filter=n>=2 && n<4;
        const char *parameters=reduce?"a:int,b:int":"x:int";
        const char *signature=reduce?"fn(int,int)->int":filter?"fn(int)->bool":"fn(int)->int";
        const char *returns=filter?"bool":"int";
        const char *changed=reduce?"b":filter?"true":"(- x)";
        const char *steady=reduce?"a":filter?"false":"x";
        char source[2048];
        snprintf(source,sizeof source,
            "let mut selected:%s=change\n"
            "fn change(%s)->%s{set selected stable return %s}\n"
            "shadow change{assert (== (change %s) %s)}\n"
            "fn stable(%s)->%s{return %s}\n"
            "shadow stable{assert (== (stable %s) %s)}\n"
            "fn main()->int{return 0}\n",
            signature,parameters,returns,changed,reduce?"1 3":"1",reduce?"3":filter?"true":"-1",
            parameters,returns,steady,reduce?"1 3":"1",filter?"false":"1");
        RunCtx ctx; ASSERT(run_ctx_init(&ctx,source));
        Symbol *selected=env_get_var(ctx.env,"selected"); ASSERT(selected);
        Value original=selected->value; ASSERT(original.type==VAL_FUNCTION);
        DynArray *input=dyn_array_new(ELEM_INT);
        input=dyn_array_push_int(input,1); input=dyn_array_push_int(input,3);
        Value array=create_void(); array.type=VAL_DYN_ARRAY; array.as.dyn_array_val=input;
        env_define_var(ctx.env,"input",TYPE_ARRAY,false,array);
        ASTNode receiver={0},initial={0},function={0},call={0};
        receiver.type=AST_IDENTIFIER; receiver.as.identifier="input";
        initial.type=AST_NUMBER; initial.as.number=0;
        function.type=AST_IDENTIFIER; function.as.identifier=borrowed?"selected":"change";
        ASTNode *arguments[]={&receiver,reduce?&initial:&function,&function};
        call.type=AST_CALL; call.as.call.name=(char *)names[n];
        call.as.call.args=arguments; call.as.call.arg_count=reduce?3:2;
        callback_snapshot_calls=0; callback_snapshot_failure=persistent?-1:1;
        Value result=repl_eval_node(&call,ctx.env);
        ASSERT(result.type==VAL_VOID); ASSERT_EQ(callback_snapshot_calls,1);
        selected=env_get_var(ctx.env,"selected");
        ASSERT(selected->value.as.function_val.function_name==original.as.function_val.function_name);
        ASSERT(selected->value.as.function_val.signature==original.as.function_val.signature);
        ASSERT(strcmp(original.as.function_val.function_name,"change")==0);
        callback_snapshot_failure=0;
        result=repl_eval_node(&call,ctx.env);
        ASSERT_EQ(callback_snapshot_calls,2);
        if (reduce) { ASSERT(result.type==VAL_INT); ASSERT_EQ(result.as.int_val,3); }
        else {
            ASSERT(result.type==VAL_DYN_ARRAY); ASSERT_EQ(dyn_array_length(result.as.dyn_array_val),2);
            ASSERT_EQ(dyn_array_get_int(result.as.dyn_array_val,0),filter?1:-1);
            ASSERT_EQ(dyn_array_get_int(result.as.dyn_array_val,1),filter?3:-3);
            gc_release(result.as.dyn_array_val);
        }
        selected=env_get_var(ctx.env,"selected");
        ASSERT(strcmp(selected->value.as.function_val.function_name,"stable")==0);
        ASSERT_EQ(dyn_array_get_int(input,0),1); ASSERT_EQ(dyn_array_get_int(input,1),3);
        env_get_var(ctx.env,"input")->value=create_void(); gc_release(input);
        run_ctx_free(&ctx);
    }
    puts("I retain callback loop identity, snapshot refusal and fresh recovery in 24 cases.");
}

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
        /* My returned fixed array borrows ctx; fixed remains caller-owned. */
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
    callback_ownership();
    callback_mutation_and_failure();
    puts("I retain 30 exact integer negation results and unchanged inputs across five evaluator paths.");
    return 0;
}
