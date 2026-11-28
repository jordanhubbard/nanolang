/* I own generated descriptor storage while declarations keep borrowing ASTs. */
#include "nanolang.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int g_argc;
char **g_argv;
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"I fail generated-list metadata check %s at line %d.\n",#c,__LINE__);exit(1); } } while (0)
int main(void) {
    for (int round=0; round<8; ++round) {
        Environment *env=create_environment(); CHECK(env);
        char source_name[]="TrackedItem";
        env_register_list_instantiation(env,source_name);
        int count=env->function_count;
        env_register_list_instantiation(env,source_name); CHECK(env->function_count==count);
        source_name[0]='X';
        Function *fresh=env_get_function(env,"List_TrackedItem_new"); CHECK(fresh);
        CHECK(fresh->is_extern && fresh->param_count==0 && fresh->return_type==TYPE_LIST_GENERIC);
        Function *push=env_get_function(env,"List_TrackedItem_push"); CHECK(push);
        CHECK(push->param_count==2 && push->params[0].type==TYPE_LIST_GENERIC);
        CHECK(push->params[1].type==TYPE_STRUCT && !strcmp(push->params[1].struct_type_name,"TrackedItem"));
        CHECK(!strcmp(push->params[0].name,"list") && !strcmp(push->params[1].name,"value"));
        Function saved=*push;
        Function *get=env_get_function(env,"List_TrackedItem_get"); CHECK(get);
        CHECK(get->param_count==2 && get->params[1].type==TYPE_INT);
        CHECK(get->return_type==TYPE_STRUCT && !strcmp(get->return_struct_type_name,"TrackedItem"));
        Function *length=env_get_function(env,"List_TrackedItem_length"); CHECK(length);
        CHECK(length->param_count==1 && length->return_type==TYPE_INT);
        for (int i=0;i<12;++i) { char name[32];snprintf(name,sizeof name,"Extra%d",i);env_register_list_instantiation(env,name); }
        CHECK(!strcmp(saved.name,"List_TrackedItem_push")); CHECK(!strcmp(saved.params[1].struct_type_name,"TrackedItem"));
        char borrowed_name[]="borrowed_declaration", borrowed_parameter[]="argument";
        Parameter parameter={0};parameter.name=borrowed_parameter;parameter.type=TYPE_INT;
        Function declaration={0};declaration.name=borrowed_name;declaration.params=&parameter;declaration.param_count=1;
        declaration.is_extern=true;declaration.return_type=TYPE_INT;env_define_function(env,declaration);
        free_environment(env);
        CHECK(!strcmp(borrowed_name,"borrowed_declaration") && !strcmp(borrowed_parameter,"argument"));
    }
    puts("I preserve four generated list signatures, duplicate admission, source independence and borrowed declarations across eight full metadata teardowns.");
    return 0;
}
