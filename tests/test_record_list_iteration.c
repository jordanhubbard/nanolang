/* I keep checker metadata owned and runtime loop values borrowed. */
#include "../src/eval/carrier_runtime.h"
#define main original_eval_suite_main
#include "test_eval.c"
#undef main
int main(int argc,char **argv) {
    bool runtime=argc==2;ASSERT(argc==1 || (runtime && strcmp(argv[1],"--runtime")==0));
    ASSERT_EQ(setvbuf(stdout,NULL,_IONBF,0),0);
    FILE *f=fopen("tests/fixtures/record_list_iteration.nano","rb");ASSERT(f);char source[16384];size_t n=fread(source,1,sizeof source-1,f);ASSERT(feof(f));fclose(f);source[n]=0;
    for(int round=0;round<4;round++) {
        RunCtx ctx={0};ctx.tokens=tokenize(source,&ctx.token_count);ASSERT(ctx.tokens);ctx.program=parse_program(ctx.tokens,ctx.token_count);ASSERT(ctx.program);clear_module_cache();ctx.env=create_environment();typecheck_set_current_file("<test>");ASSERT(type_check(ctx.program,ctx.env));ASSERT(type_check_root_shadows(ctx.program,ctx.env));
        Symbol *outer=env_get_var(ctx.env,"outer");ASSERT(outer);ASSERT_EQ(outer->type,TYPE_STRUCT);ASSERT(outer->type_info);ASSERT_EQ(outer->type_info->base_type,TYPE_STRUCT);ASSERT(strcmp(outer->type_info->generic_name,"LoopItem")==0);
        TypeInfo *owned=outer->type_info;char *borrowed=NULL;
        for(int i=0;i<ctx.program->as.program.count;i++) {ASTNode *node=ctx.program->as.program.items[i];if(node->type==AST_FUNCTION && strcmp(node->as.function.name,"loop_sum")==0)borrowed=node->as.function.params[0].struct_type_name;}
        ASSERT(borrowed);ASSERT(owned->generic_name!=borrowed);
        for(int i=0;i<128;i++) {char name[64];snprintf(name,sizeof name,"loop_growth_%d",i);env_define_var(ctx.env,name,TYPE_INT,false,create_int(i));}
        ASSERT(strcmp(owned->generic_name,"LoopItem")==0);
        if(runtime) {
            ASSERT(env_carrier_enable(ctx.env));ASSERT(env_carrier_enable_collection(ctx.env,NULL,0));ASSERT(run_program(ctx.program,ctx.env));ASSERT(run_shadow_tests(ctx.program,ctx.env,false));ASSERT(env_carrier_collect(ctx.env));ASSERT_EQ(env_carrier_stats(ctx.env).live,0);
            Value fields[]={create_int(41),create_string("kept")};char *names[]={"number","text"};
            StructValue record={0};record.struct_name="LoopItem";record.field_count=2;record.field_names=names;record.field_values=fields;
            DynArray *list=dyn_array_new(ELEM_INT);dyn_array_push_int(list,(intptr_t)&record);
            NlCarrierRef root=nl_carrier_record_list(list);ASSERT(env_carrier_enable_collection(ctx.env,&root,1));
            size_t registered=env_carrier_stats(ctx.env).registered;char *identity=fields[1].as.string_val;
            Value argument=create_int((intptr_t)list);
            for(int i=0;i<64;i++) {Value result=call_function("loop_sum",&argument,1,ctx.env);ASSERT_EQ(result.type,VAL_INT);ASSERT_EQ(result.as.int_val,41);ASSERT_EQ(dyn_array_get_int(list,0),(intptr_t)&record);ASSERT(fields[1].as.string_val==identity);ASSERT(strcmp(identity,"kept")==0);}
            ASSERT_EQ(env_carrier_stats(ctx.env).registered,registered);ASSERT_EQ(gc_get_header(identity)->ref_count,1);
            if(round%2==0) {gc_release(list);ASSERT(env_carrier_enable_collection(ctx.env,NULL,0));ASSERT(env_carrier_collect(ctx.env));}
            free_environment(ctx.env);ctx.env=NULL;
            ASSERT(strcmp(identity,"kept")==0);ASSERT_EQ(record.field_values[0].as.int_val,41);
            if(round%2)gc_release(list);
            gc_release(identity);ASSERT_EQ(gc_get_stats().num_objects,0);ASSERT_EQ(gc_get_stats().current_usage,0);ASSERT_EQ(env_carrier_last_retired_stats().live,0);
        } else {free_environment(ctx.env);ctx.env=NULL;}
        ASSERT(strcmp(borrowed,"LoopItem")==0);free_ast(ctx.program);free_tokens(ctx.tokens,ctx.token_count);clear_module_cache();
    }
    printf("I qualify four complete record-list contexts: mode=%s; borrowed caller storage survives.\n",runtime?"private runtime":"checker");return 0;
}
