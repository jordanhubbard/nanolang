#include "nanolang.h"
#include "service_bodies.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
static long allocations_before_failure=-1;
static void *body_calloc(size_t count, size_t size) {
    if(allocations_before_failure==0) return NULL;
    if(allocations_before_failure>0) --allocations_before_failure;
    return calloc(count,size);
}
#define calloc body_calloc
#include "../src/service_bodies.c"
#undef calloc
int g_argc; char **g_argv;
int main(int argc, char **argv) {
    assert(argc==3);
    FILE *file=fopen(argv[1],"rb"); assert(file);
    assert(!fseek(file,0,SEEK_END)); long size=ftell(file); assert(size>=0); rewind(file);
    char *source=calloc((size_t)size+1,1); assert(source);
    assert(fread(source,1,(size_t)size,file)==(size_t)size); fclose(file);
    int count=0; Token *tokens=tokenize(source,&count); assert(tokens);
    ASTNode *root=parse_program(tokens,count); assert(root);
    Environment *env=create_environment(); ModuleList *modules=create_module_list();
    assert(!process_imports(root,env,modules,argv[1]));
    const NlServiceBodyCheck *check=env->service_bodies;
    assert(check);
    printf("STATUS %u FUNCTIONS %zu SHADOWS %zu FACTS %zu\n",check->status,check->functions,check->shadows,check->count);
    if(check->diagnostic) puts(check->diagnostic);
    assert(check->status==(unsigned)atoi(argv[2]));
    for(size_t i=0;i<check->count;++i) {
        const NlServiceBodyFact *fact=&check->facts[i];
        if(fact->declaration) {
            const NlServiceName *row=nl_service_namespace_name(env->service_namespace,fact->declaration-1);
            assert(row);
            printf("CALL %s %u %u\n",row->name,fact->type.service_ordinal,fact->borrow_mode);
        }
    }
    for(long prefix=0;prefix<4;++prefix) {
        allocations_before_failure=prefix;
        NlServiceBodyCheck *candidate=nl_service_check_bodies(env->service_namespace);
        allocations_before_failure=-1;
        if(prefix<3) assert(!candidate);
        else { assert(candidate && candidate->status==check->status); nl_service_body_check_free(candidate); }
        assert(env->service_bodies==check);
    }
    free_environment(env); free_ast(root); free_tokens(tokens,count); free(source);
    free_module_list(modules); clear_module_cache();
    return 0;
}
