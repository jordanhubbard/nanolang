#include "../src/eval/carrier_runtime.h"
#include "../src/tracing.h"
#define main original_eval_suite_main
#include "test_eval.c"
#undef main
static size_t peak_live;
static void sample_statement(ASTNode *stmt, Environment *env) {
    (void)stmt;
    size_t live = env_carrier_stats(env).live;
    if (live > peak_live) peak_live = live;
}
int main(void) {
    ASSERT_EQ(setvbuf(stdout,NULL,_IONBF,0),0);
    FILE *f = fopen("tests/fixtures/carrier_active_roots.nano","rb"); ASSERT(f);
    char source[16384]; size_t n=fread(source,1,sizeof source-1,f); ASSERT(feof(f)); fclose(f);source[n]=0;
    RunCtx ctx={0};ctx.tokens=tokenize(source,&ctx.token_count);ASSERT(ctx.tokens);
    ctx.program=parse_program(ctx.tokens,ctx.token_count);ASSERT(ctx.program);
    clear_module_cache();ctx.env=create_environment();typecheck_set_current_file("<test>");
    ASSERT(type_check(ctx.program,ctx.env));ASSERT(env_carrier_enable(ctx.env));
    ASSERT(env_carrier_enable_collection(ctx.env,NULL,0));ASSERT(run_program(ctx.program,ctx.env));
    extern void (*g_dap_statement_hook)(ASTNode *,Environment *);
    g_dap_statement_hook=sample_statement;
    ASSERT(run_shadow_tests(ctx.program,ctx.env,false));
    NlCarrierHeapStats start=env_carrier_stats(ctx.env);
    for(int i=0;i<64;++i) { Value v=call_function("root_siblings",NULL,0,ctx.env);ASSERT_EQ(v.type,VAL_INT);ASSERT_EQ(v.as.int_val,16); }
    NlCarrierHeapStats middle=env_carrier_stats(ctx.env);
    for(int i=0;i<64;++i) { Value v=call_function("root_siblings",NULL,0,ctx.env);ASSERT_EQ(v.type,VAL_INT);ASSERT_EQ(v.as.int_val,16); }
    NlCarrierHeapStats end=env_carrier_stats(ctx.env);
    ASSERT(middle.destroyed>start.destroyed);ASSERT(end.destroyed>middle.destroyed);
    ASSERT_EQ(middle.live,end.live);ASSERT(peak_live<=32);
    printf("I bound actual source calls: start=%zu after64=%zu after128=%zu peak=%zu destroyed=%zu.\n",start.live,middle.live,end.live,peak_live,end.destroyed);
    g_dap_statement_hook=NULL;run_ctx_free(&ctx);
    NlCarrierHeapStats retired=env_carrier_last_retired_stats();ASSERT_EQ(retired.live,0);
    ASSERT_EQ(retired.registered,retired.destroyed+retired.transferred);
    return 0;
}
