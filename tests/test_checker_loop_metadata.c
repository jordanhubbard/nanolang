/* I own copied loop trees and preserve ordinary AST metadata after env teardown. */
#include "../src/eval/carrier_runtime.h"
#define main original_eval_suite_main
#include "test_eval.c"
#undef main
int main(int argc, char **argv) {
    bool runtime = argc==2;
    bool private = runtime && strcmp(argv[1],"--private-runtime")==0;
    ASSERT(argc==1 || (runtime && (private || strcmp(argv[1],"--legacy-runtime")==0)));
    FILE *f=fopen("tests/fixtures/checker_loop_metadata.nano","rb");ASSERT(f);
    char source[8192];size_t n=fread(source,1,sizeof source-1,f);ASSERT(feof(f));fclose(f);source[n]=0;
    for(int round=0;round<8;++round) {
        RunCtx ctx={0};ctx.tokens=tokenize(source,&ctx.token_count);ASSERT(ctx.tokens);
        ctx.program=parse_program(ctx.tokens,ctx.token_count);ASSERT(ctx.program);
        clear_module_cache();ctx.env=create_environment();typecheck_set_current_file("<test>");
        ASSERT(type_check(ctx.program,ctx.env));
        Symbol *row=env_get_var(ctx.env,"row");ASSERT(row);ASSERT(row->type_info);
        TypeInfo *copied=row->type_info;ASSERT_EQ(copied->base_type,TYPE_ARRAY);
        ASSERT(copied->element_type);ASSERT_EQ(copied->element_type->base_type,TYPE_INT);
        Function *fn=env_get_function(ctx.env,"checker_loop_borrowed");ASSERT(fn);
        ASSERT(fn->params[0].type_info);TypeInfo *borrowed=fn->params[0].type_info->element_type;
        Symbol *text=env_get_var(ctx.env,"text");ASSERT(text);ASSERT(text->type_info==borrowed);
        for(int i=0;i<128;++i) { char name[64];snprintf(name,sizeof name,"metadata_growth_%d",i);env_define_var(ctx.env,name,TYPE_INT,false,create_int(i)); }
        ASSERT_EQ(copied->element_type->base_type,TYPE_INT);
        if (runtime) {
            if (private) { ASSERT(env_carrier_enable(ctx.env));ASSERT(env_carrier_enable_collection(ctx.env,NULL,0)); }
            ASSERT(run_program(ctx.program,ctx.env));ASSERT(run_shadow_tests(ctx.program,ctx.env,false));
        }
        free_environment(ctx.env);ctx.env=NULL;
        ASSERT_EQ(borrowed->base_type,TYPE_STRING);
        free_ast(ctx.program);free_tokens(ctx.tokens,ctx.token_count);
    }
    printf("I preserve inferred nested-loop trees and borrowed AST annotations through eight full teardowns; mode=%s.\n",private ? "private runtime with all32 shadows" : runtime ? "legacy runtime with all32 shadows" : "checker only");
    return 0;
}
