/* I qualify STRING references without granting ownership of aggregate storage. */
#include "../src/eval/carrier_runtime.h"
#define main original_eval_suite_main
#include "test_eval.c"
#undef main

static void mode_boundaries(void) {
    ASSERT(!env_carrier_enable_strings(NULL));
    ASSERT(!env_carrier_enable(NULL));
    Environment *env=create_environment();
    ASSERT(!env_carrier_enabled(env));ASSERT(!env_carrier_scopes_enabled(env));
    ASSERT(env_carrier_enable_strings(env));ASSERT(env_carrier_enable_strings(env));
    ASSERT(!env_carrier_enabled(env));ASSERT(env_carrier_scopes_enabled(env));
    ASSERT(!env_carrier_enable_collection(env,NULL,0));
    Value array=create_array(VAL_INT,1,1);((long long *)array.as.array_val->data)[0]=42;
    env_carrier_track(env,array,false);ASSERT(!env_carrier_owns(env,array));
    ASSERT(env_carrier_collect(env));ASSERT_EQ(env_carrier_stats(env).registered,0);
    ASSERT_EQ(((long long *)array.as.array_val->data)[0],42);
    free(array.as.array_val->data);free(array.as.array_val);
    env_carrier_enter(env);
    Value temporary=create_string("temporary");env_carrier_track_string(env,temporary);
    ASSERT(!env_carrier_enable(env));ASSERT(!env_carrier_enabled(env));
    env_carrier_leave(env);ASSERT_EQ(gc_get_stats().num_objects,0);
    ASSERT(env_carrier_enable(env));ASSERT(env_carrier_enabled(env));
    ASSERT(env_carrier_enable_strings(env));ASSERT(env_carrier_enable(env));
    array=create_array(VAL_INT,1,1);env_carrier_track(env,array,false);
    ASSERT(env_carrier_owns(env,array));ASSERT_EQ(env_carrier_stats(env).registered,1);
    ASSERT(env_carrier_enable_collection(env,NULL,0));ASSERT(env_carrier_collect(env));
    ASSERT_EQ(env_carrier_stats(env).destroyed,1);ASSERT_EQ(env_carrier_stats(env).live,0);
    free_environment(env);ASSERT_EQ(env_carrier_last_string_stats().live,0);
    puts("I keep STRING-only ownership separate and upgrade only outside active frames.");
}
static void source_fixture(const char *path,bool carriers) {
    FILE *f=fopen(path,"rb");ASSERT(f);char source[16384];size_t n=fread(source,1,sizeof source-1,f);ASSERT(feof(f));fclose(f);source[n]=0;
    RunCtx ctx={0};ctx.tokens=tokenize(source,&ctx.token_count);ASSERT(ctx.tokens);ctx.program=parse_program(ctx.tokens,ctx.token_count);ASSERT(ctx.program);
    clear_module_cache();ctx.env=create_environment();typecheck_set_current_file("<test>");ASSERT(type_check(ctx.program,ctx.env));
    ASSERT(env_carrier_enable_strings(ctx.env));
    if(carriers){ASSERT(env_carrier_enable(ctx.env));ASSERT(env_carrier_enable_collection(ctx.env,NULL,0));}
    ASSERT(run_program(ctx.program,ctx.env));ASSERT(run_shadow_tests(ctx.program,ctx.env,false));
    if(strstr(path,"carrier_string_async.nano")) {
        Value caller=create_string("caller");char *identity=caller.as.string_val;
        Value result=call_function("string_async_echo",&caller,1,ctx.env);
        ASSERT_EQ(result.type,VAL_STRING);ASSERT(result.as.string_val!=identity);
        ASSERT(strcmp(result.as.string_val,"caller")==0);gc_release(result.as.string_val);
        ASSERT(caller.as.string_val==identity);ASSERT(strcmp(identity,"caller")==0);
        ASSERT_EQ(gc_get_header(identity)->ref_count,1);gc_release(identity);
    }
    if(!carriers){ASSERT(!env_carrier_enabled(ctx.env));ASSERT_EQ(env_carrier_stats(ctx.env).registered,0);}
    run_ctx_free(&ctx);NlCarrierStringStats stats=env_carrier_last_string_stats();
    ASSERT_EQ(stats.live,0);ASSERT_EQ(stats.registered,stats.released+stats.exported);
    ASSERT_EQ(gc_get_stats().num_objects,0);ASSERT_EQ(gc_get_stats().current_usage,0);
    printf("I qualify %s mode=%s registered=%zu released=%zu exported=%zu live=%zu.\n",path,carriers?"full":"strings",stats.registered,stats.released,stats.exported,stats.live);
}
int main(int argc,char **argv) {
    ASSERT_EQ(setvbuf(stdout,NULL,_IONBF,0),0);mode_boundaries();
    bool carriers=argc==2 && strcmp(argv[1],"--full")==0;
    if(argc==2 && strcmp(argv[1],"--duplicate")==0){
        source_fixture("tests/fixtures/carrier_string_duplicate.nano",false);return 0;
    }
    source_fixture("tests/fixtures/carrier_string_sibling.nano",carriers);
    if(carriers) source_fixture("tests/fixtures/carrier_string_duplicate.nano",true);
    source_fixture("tests/fixtures/carrier_string_async.nano",carriers);
    return 0;
}
