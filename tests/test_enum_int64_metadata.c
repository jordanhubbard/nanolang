/* I retain full-width enum metadata independently of source list storage. */
#include "../src/nanolang.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"I fail %s at line %d.\n",#c,__LINE__);exit(1); } } while (0)
int g_argc;
char **g_argv;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }
int main(void) {
    const char *source="enum Edge { Min = -9223372036854775808, MinNext, Max = 9223372036854775807, Reset = -4294967297, ResetNext }\nfn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n";
    const long long expected[]={LLONG_MIN,LLONG_MIN+1,LLONG_MAX,-4294967297LL,-4294967296LL};
    int count=0;Token *tokens=tokenize(source,&count);CHECK(tokens);
    ASTNode *program=parse_program(tokens,count);CHECK(program);
    ASTNode *definition=program->as.program.items[0];CHECK(definition->type==AST_ENUM_DEF);
    CHECK(sizeof *definition->as.enum_def.variant_values==8);
    for (int i=0;i<5;++i)CHECK(definition->as.enum_def.variant_values[i]==expected[i]);
    Environment *env=create_environment();typecheck_set_current_file("<enum-int64>");
    CHECK(type_check(program,env));CHECK(type_check_root_shadows(program,env));
    EnumDef *registered=env_get_enum(env,"Edge");CHECK(registered && registered->variant_count==5);
    for (int i=0;i<5;++i)CHECK(registered->variant_values[i]==expected[i]);
    CHECK(env_get_enum_variant(env,"Min")==LLONG_MIN);
    CHECK(env_get_enum_variant(env,"Max")==LLONG_MAX);
    ModuleMetadata *meta=extract_module_metadata(env,"wide");CHECK(meta && meta->enum_count==1);
    CHECK(meta->enums[0].variant_values!=registered->variant_values);
    for (int i=0;i<5;++i)CHECK(meta->enums[0].variant_values[i]==expected[i]);
    free_module_metadata(meta);
    CHECK(run_program(program,env));CHECK(run_shadow_tests(program,env,false));
    registered=env_get_enum(env,"Edge");CHECK(registered);
    for (int i=0;i<5;++i)CHECK(registered->variant_values[i]==expected[i]);
    free_environment(env);free_ast(program);free_tokens(tokens,count);clear_module_cache();
    CHECK(gc_get_stats().num_objects==0 && gc_get_stats().current_usage==0);
    puts("I preserve signed64 AST, checked/runtime declarations, direct enum lookup and independent module metadata copies.");
    return 0;
}
