/* I run accepted enum values and list roots through the actual interpreter. */
#include "../src/nanolang.h"
#include "../src/eval/carrier_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"I fail %s at line %d.\n",#c,__LINE__);exit(1); } } while (0)
int g_argc;
char **g_argv;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }
static void fixture(const char *path, bool pressure) {
    FILE *file=fopen(path,"rb");CHECK(file);CHECK(fseek(file,0,SEEK_END)==0);
    long length=ftell(file);CHECK(length>0);rewind(file);
    char *source=malloc((size_t)length+1);CHECK(source);
    CHECK(fread(source,1,(size_t)length,file)==(size_t)length);CHECK(fclose(file)==0);source[length]=0;
    int count=0;Token *tokens=tokenize(source,&count);CHECK(tokens);
    ASTNode *program=parse_program(tokens,count);CHECK(program);
    Environment *env=create_environment();typecheck_set_current_file(path);
    CHECK(type_check(program,env));CHECK(type_check_root_shadows(program,env));
    CHECK(env_carrier_enable(env));CHECK(env_carrier_enable_collection(env,NULL,0));
    CHECK(run_program(program,env));CHECK(run_shadow_tests(program,env,false));
    if (pressure) for (int i=0;i<128;++i) {
        Value result=call_function("enum_hold",NULL,0,env);
        CHECK(result.type==VAL_INT && result.as.int_val==781);
        result=call_function("enum_field",NULL,0,env);
        CHECK(result.type==VAL_INT && result.as.int_val==-3);
        result=call_function("enum_scalar_values",NULL,0,env);
        CHECK(result.type==VAL_INT && result.as.int_val==17);
        CHECK(env_carrier_collect(env));CHECK(env_carrier_stats(env).live==0);
        CHECK(env_carrier_consumed_count(env)==0);
    }
    if (pressure) {
        List_int *caller=list_int_new();list_int_push(caller,17);
        NlCarrierRef host=nl_carrier_legacy_scalar_list(caller);
        CHECK(env_carrier_begin_list_borrow(env,host));
        CHECK(env_carrier_enable_collection(env,&host,1));
        Value args[]={create_int((intptr_t)caller),create_int(5)};
        Value result=call_function("enum_size",args,2,env);
        CHECK(result.type==VAL_INT && result.as.int_val==6);
        CHECK(env_carrier_collect(env));CHECK(env_carrier_consumed_count(env)==1);
        CHECK(env_carrier_enable_collection(env,NULL,0));
        CHECK(env_carrier_collect(env));CHECK(env_carrier_consumed_count(env)==0);
    }
    free_environment(env);free_ast(program);free_tokens(tokens,count);free(source);clear_module_cache();
    CHECK(env_carrier_last_retired_stats().live==0);
    CHECK(gc_get_stats().num_objects==0 && gc_get_stats().current_usage==0);
}
int main(void) {
    fixture("tests/nanoisa/fixtures/enum_values.nano",false);
    fixture("tests/nanoisa/fixtures/carrier_enum_pressure.nano",true);
    puts("I preserve enum get/pop, parameters/results, sibling pressure and record fields across 128 rounds.");
    return 0;
}
