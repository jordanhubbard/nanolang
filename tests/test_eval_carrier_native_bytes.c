#include "../src/nanolang.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define NANO_SCRATCH_ASAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(NANO_SCRATCH_ASAN)
extern size_t __sanitizer_get_current_allocated_bytes(void);
static size_t allocated(void) { return __sanitizer_get_current_allocated_bytes(); }
#elif defined(__GLIBC__)
#include <malloc.h>
static size_t allocated(void) { struct mallinfo2 m=mallinfo2(); return m.uordblks + m.hblkhd; }
#else
int g_argc;
char **g_argv;
int main(void) { puts("I skip native allocation accounting without glibc or ASan statistics.");return 0; }
#endif
#if defined(__GLIBC__) || defined(__SANITIZE_ADDRESS__) || defined(NANO_SCRATCH_ASAN)
static size_t first_bytes,middle_bytes,last_bytes,observed_calls;
static Value observed_call_function(const char *,Value *,int,Environment *);
#define call_function observed_call_function
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
int fixture_main(void) {
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

#undef main
#undef call_function
static Value observed_call_function(const char *name,Value *args,int count,Environment *env) {
    bool chosen=strcmp(name,"root_siblings")==0;
    if(chosen && !observed_calls) first_bytes=allocated();
    Value result=call_function(name,args,count,env);
    if(chosen) { observed_calls++;if(observed_calls==64)middle_bytes=allocated();if(observed_calls==128)last_bytes=allocated(); }
    return result;
}
int main(void) {
    size_t before=allocated();
    void *block=malloc(2*1024*1024);if(!block)return 92;
    ((volatile unsigned char *)block)[0]=1;((volatile unsigned char *)block)[2*1024*1024-1]=2;
    size_t during=allocated();free(block);size_t after=allocated();
    if(during<before+2*1024*1024 || after>before+4096)return 93;
    int result=fixture_main();
    printf("I observe native allocation: positive=%zu before=%zu after64=%zu after128=%zu delta=%lld calls=%zu.\n",during-before,first_bytes,middle_bytes,last_bytes,(long long)last_bytes-(long long)middle_bytes,observed_calls);
    if(observed_calls!=128 || last_bytes>middle_bytes+1024)return 94;
    return result;
}

#endif
