/* I observe per-walk metadata lifetime and optional-allocation fallback. */
#include "../src/nanolang.h"
#include "../src/eval/carrier_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"I fail %s at line %d.\n",#c,__LINE__);exit(1); } } while (0)
static int declaration_budget=-1;
static size_t declaration_failures;
static void *declaration_calloc(size_t count,size_t bytes) {
    if(declaration_budget==0) {declaration_failures++;return NULL;}
    if(declaration_budget>0)declaration_budget--;
    return calloc(count,bytes);
}
#define calloc declaration_calloc
#include "../src/eval/carrier_runtime.c"
#undef calloc
int g_argc;
char **g_argv;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }
static Value item(Environment *env) {
    Value text=create_string("kept");char *name="text";
    Value result=create_struct("WalkItem",&name,&text,1);gc_release(text.as.string_val);
    env_carrier_track(env,result,false);return result;
}
static Value list(Environment *env,Value value) {
    List_int *header=list_int_new();list_int_push(header,(intptr_t)value.as.struct_val);
    env_carrier_track_legacy_list(env,header);return create_int((intptr_t)header);
}
static void declaration(Environment *env,const char *owner,Type type) {
    StructDef def={0};def.name=strdup("WalkShape");def.module_name=strdup(owner);def.field_count=1;
    if(!strcmp(owner,"one"))def.original_name=strdup("OriginalShape");
    def.field_names=malloc(sizeof *def.field_names);def.field_names[0]=strdup("items");
    def.field_types=malloc(sizeof *def.field_types);def.field_types[0]=type;
    env_define_struct(env,def);
}
int main(void) {
    Environment *env=create_environment();CHECK(env_carrier_enable(env));
    CHECK(env_carrier_enable_collection(env,NULL,0));
    Value empty=create_struct("Warm",NULL,NULL,0);env_carrier_track(env,empty,false);
    env_define_var(env,"warm",TYPE_BORROW_SHARED,true,empty);CHECK(env_carrier_collect(env));
    env_get_var(env,"warm")->value=create_void();CHECK(env_carrier_collect(env));
    declaration(env,"two",TYPE_INT);
    for(int i=0;i<12;++i) {
        char name[64];snprintf(name,sizeof name,"WalkPadding%d",i);
        StructDef padding={0};padding.name=strdup(name);env_define_struct(env,padding);
    }
    declaration(env,"one",TYPE_LIST_GENERIC);
    env->current_module="one";
    Value child=item(env),encoded=list(env,child);char *name="items";
    Value parent=create_struct("WalkShape",&name,&encoded,1);env_carrier_track(env,parent,false);
    env_define_var(env,"selected",TYPE_BORROW_SHARED,true,parent);
    declaration_budget=1;
    CHECK(env_carrier_collect(env));
    declaration_budget=-1;
    CHECK(declaration_failures>0 && env_carrier_stats(env).live==3);
    CHECK(!strcmp(child.as.struct_val->field_values[0].as.string_val,"kept"));
    CHECK(env_carrier_collect(env));CHECK(env_carrier_stats(env).live==3);
    StructDef *first=env_get_struct_owned(env,"WalkShape","one");CHECK(first);
    Type *replacement=malloc(sizeof *replacement);replacement[0]=TYPE_INT;
    free(first->field_types);first->field_types=replacement;
    CHECK(env_carrier_collect(env));CHECK(env_carrier_stats(env).live==1);
    parent.as.struct_val->field_values[0]=create_int(0);
    first->field_types[0]=TYPE_LIST_GENERIC;
    child=item(env);encoded=list(env,child);parent.as.struct_val->field_values[0]=encoded;
    CHECK(env_carrier_collect(env));CHECK(env_carrier_stats(env).live==3);
    env->current_module="two";
    CHECK(env_carrier_collect(env));CHECK(env_carrier_stats(env).live==1);
    parent.as.struct_val->field_values[0]=create_int(0);
    env->current_module="one";
    child=item(env);encoded=list(env,child);parent.as.struct_val->field_values[0]=encoded;
    CHECK(env_carrier_collect(env));CHECK(env_carrier_stats(env).live==3);
    Value alias=create_struct("OriginalShape",&name,&encoded,1);env_carrier_track(env,alias,false);
    env_define_var(env,"original_alias",TYPE_BORROW_SHARED,true,alias);
    CHECK(env_carrier_collect(env));CHECK(env_carrier_stats(env).live==4);
    env->current_module="two";
    CHECK(env_carrier_collect(env));CHECK(env_carrier_stats(env).live==4);
    env_get_var(env,"original_alias")->value=create_void();
    CHECK(env_carrier_collect(env));CHECK(env_carrier_stats(env).live==1);
    parent.as.struct_val->field_values[0]=create_int(0);
    env->current_module="one";
    child=item(env);encoded=list(env,child);
    Value unknown=create_struct("missing.WalkShape",&name,&encoded,1);env_carrier_track(env,unknown,false);
    env_define_var(env,"missing_namespace",TYPE_BORROW_SHARED,true,unknown);
    CHECK(env_carrier_collect(env));CHECK(env_carrier_stats(env).live==2);
    env_get_var(env,"missing_namespace")->value=create_void();
    CHECK(env_carrier_collect(env));CHECK(env_carrier_stats(env).live==1);
    child=item(env);encoded=list(env,child);parent.as.struct_val->field_values[0]=encoded;
    CHECK(env_carrier_collect(env));CHECK(env_carrier_stats(env).live==3);
    parent.as.struct_val->field_values[0]=create_bool(true);
    CHECK(!env_carrier_collect(env));CHECK(env_carrier_stats(env).live==3);
    parent.as.struct_val->field_values[0]=encoded;
    first->field_types[0]=TYPE_INT;
    CHECK(env_carrier_collect(env));CHECK(env_carrier_stats(env).live==1);
    parent.as.struct_val->field_values[0]=create_int(0);
    first->field_types[0]=TYPE_LIST_GENERIC;
    child=item(env);encoded=list(env,child);parent.as.struct_val->field_values[0]=encoded;
    env->current_module="two";
    env_carrier_export(env,parent,false);
    CHECK(env_carrier_stats(env).transferred==1);
    env_get_var(env,"selected")->value=create_void();
    CHECK(env_carrier_collect(env));CHECK(env_carrier_stats(env).live==0);
    env->current_module=NULL;free_environment(env);
    release_struct_carrier_names(parent.as.struct_val);
    free(parent.as.struct_val->field_values);free(parent.as.struct_val);
    CHECK(gc_get_stats().num_objects==0 && gc_get_stats().current_usage==0);
    puts("I preserve current module/type metadata across successful/failed walks and transfer, release no caller payload through metadata, and retain list roots when optional declaration allocation fails.");
    return 0;
}
