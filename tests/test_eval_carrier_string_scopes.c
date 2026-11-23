/* I distinguish source temporary references from caller-owned STRING values. */
#include "../src/eval/carrier_runtime.h"
#include "../src/eval/eval_string.h"
#define main original_eval_suite_main
#include "test_eval.c"
#undef main

static void library_buffers(void) {
    Value left = create_string("left"), right = create_string("right");
    Value args[] = {left, right, create_void()};
    Value joined = builtin_str_concat(args);
    ASSERT_EQ(joined.type, VAL_STRING); ASSERT(strcmp(joined.as.string_val,"leftright")==0);
    ASSERT(strcmp(left.as.string_val,"left")==0); ASSERT(strcmp(right.as.string_val,"right")==0);
    gc_release(joined.as.string_val);
    args[1] = create_int(1); args[2] = create_int(2);
    Value part = builtin_str_substring(args);
    ASSERT_EQ(part.type, VAL_STRING); ASSERT(strcmp(part.as.string_val,"ef")==0);
    gc_release(part.as.string_val); gc_release(left.as.string_val); gc_release(right.as.string_val);
    puts("I preserve caller strings and retire both library result owners.");
}
static void field_refusal(void) {
    FILE *file=fopen("tests/fixtures/carrier_string_field_invalid.nano","rb");ASSERT(file);
    char source[4096];size_t n=fread(source,1,sizeof source-1,file);ASSERT(feof(file));fclose(file);source[n]=0;
    RunCtx ctx={0};ctx.tokens=tokenize(source,&ctx.token_count);ASSERT(ctx.tokens);
    ctx.program=parse_program(ctx.tokens,ctx.token_count);ASSERT(ctx.program);
    clear_module_cache();ctx.env=create_environment();typecheck_set_current_file("<test>");
    ASSERT(!type_check(ctx.program,ctx.env));run_ctx_free(&ctx);
    puts("I preserve the exclusive borrowed-owner refusal for STRING field mutation.");
}
int main(int argc, char **argv) {
    ASSERT_EQ(setvbuf(stdout,NULL,_IONBF,0),0);
    library_buffers();
    if (argc==2 && strcmp(argv[1],"--library-only")==0) return 0;
    field_refusal();
    FILE *file=fopen("tests/fixtures/carrier_string_scopes.nano","rb");ASSERT(file);
    char source[16384];size_t n=fread(source,1,sizeof source-1,file);ASSERT(feof(file));fclose(file);source[n]=0;
    RunCtx ctx={0};ctx.tokens=tokenize(source,&ctx.token_count);ASSERT(ctx.tokens);
    ctx.program=parse_program(ctx.tokens,ctx.token_count);ASSERT(ctx.program);
    clear_module_cache();ctx.env=create_environment();typecheck_set_current_file("<test>");
    ASSERT(type_check(ctx.program,ctx.env));ASSERT(env_carrier_enable(ctx.env));
    ASSERT(env_carrier_enable_collection(ctx.env,NULL,0));ASSERT(run_program(ctx.program,ctx.env));
    ASSERT(run_shadow_tests(ctx.program,ctx.env,false));
    Value caller=create_string("caller");char *identity=caller.as.string_val;
    Value returned=call_function("string_scope_return",&caller,1,ctx.env);
    ASSERT_EQ(returned.type,VAL_STRING);ASSERT(returned.as.string_val!=identity);
    ASSERT(strcmp(returned.as.string_val,"caller")==0);gc_release(returned.as.string_val);
    ASSERT(caller.as.string_val==identity);ASSERT(strcmp(identity,"caller")==0);
    ASSERT_EQ(gc_get_header(identity)->ref_count,1);gc_release(identity);
    const char *names[]={"string_scope_bindings","string_scope_owner_rebind","string_scope_helpers","string_scope_loop","string_scope_sibling","string_scope_argument"};
    const long long expected[]={13,6,8,9,7,7};
    GCStats baseline=gc_get_stats(),previous={0};
    size_t roots=0,root_bytes=0;
    for(int i=0;i<ctx.env->symbol_count;++i) {
        Value value=ctx.env->symbols[i].value;
        if(value.type!=VAL_STRING)continue;
        ASSERT(gc_is_managed(value.as.string_val));bool known=false;
        for(int k=0;k<i;++k) if(ctx.env->symbols[k].value.type==VAL_STRING &&
            ctx.env->symbols[k].value.as.string_val==value.as.string_val)known=true;
        if(!known){roots++;root_bytes+=gc_get_header(value.as.string_val)->size;}
    }
    ASSERT(roots>0);ASSERT_EQ(baseline.num_objects,roots);ASSERT_EQ(baseline.current_usage,root_bytes);
    for(int round=0;round<2;++round) {
        for(int i=0;i<64;++i)for(size_t f=0;f<sizeof names/sizeof *names;++f) {
            Value value=call_function(names[f],NULL,0,ctx.env);ASSERT_EQ(value.type,VAL_INT);ASSERT_EQ(value.as.int_val,expected[f]);
        }
        GCStats stats=gc_get_stats();
        ASSERT_EQ(stats.num_objects,baseline.num_objects);ASSERT_EQ(stats.current_usage,baseline.current_usage);
        if (round) { ASSERT_EQ(stats.current_usage,previous.current_usage); ASSERT_EQ(stats.num_objects,previous.num_objects); }
        previous=stats;
        printf("I observe live managed storage after batch%d: bytes=%zu objects=%zu.\n",round+1,stats.current_usage,stats.num_objects);
    }
    run_ctx_free(&ctx);NlCarrierStringStats stats=env_carrier_last_string_stats();
    ASSERT_EQ(gc_get_stats().num_objects,0);ASSERT_EQ(gc_get_stats().current_usage,0);
    ASSERT_EQ(stats.live,0);ASSERT_EQ(stats.registered,stats.released+stats.exported);
    ASSERT(stats.registered>0);ASSERT(stats.exported>0);
    printf("I retire explicit fresh STRING references: registered=%zu released=%zu exported=%zu live=%zu.\n",stats.registered,stats.released,stats.exported,stats.live);
    return 0;
}
