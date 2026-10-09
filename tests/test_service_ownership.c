#include "nanolang.h"
#include "service_ownership.h"
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
#include "../src/service_ownership.c"
#undef calloc
int g_argc; char **g_argv;
static void check_limits(void) {
    NlServiceOwnershipCheck out={0};
    SoCheck check={.out=&out}; SoState state={.next=true};
    ASTNode node={.type=AST_BOOL};
    out.count=SO_FACTS; so_note(&check,&node,NL_SERVICE_MOVE,1,0); assert(out.status==3);
    out=(NlServiceOwnershipCheck){0}; check.next_binding=SO_FACTS;
    so_bind(&check,&state,&node,"f",1,0); assert(out.status==3 && state.count==0);
    out=(NlServiceOwnershipCheck){0}; check.next_binding=0; state.count=SO_LOCALS;
    so_bind(&check,&state,&node,"f",1,0); assert(out.status==3 && state.count==SO_LOCALS);
    out=(NlServiceOwnershipCheck){0}; state.count=0; check.depth=SO_DEPTH;
    so_expr(&check,&state,&node,false); assert(out.status==3);
}
int main(int argc, char **argv) {
    assert(argc==3);
    check_limits();
    FILE *file=fopen(argv[1],"rb"); assert(file);
    assert(!fseek(file,0,SEEK_END)); long size=ftell(file); assert(size>=0); rewind(file);
    char *source=calloc((size_t)size+1,1); assert(source);
    assert(fread(source,1,(size_t)size,file)==(size_t)size); fclose(file);
    int count=0; Token *tokens=tokenize(source,&count); assert(tokens);
    ASTNode *root=parse_program(tokens,count); assert(root);
    Environment *env=create_environment(); ModuleList *modules=create_module_list();
    assert(!process_imports(root,env,modules,argv[1]));
    assert(env->service_bodies && env->service_bodies->status==0);
    const NlServiceOwnershipCheck *check=env->service_ownership;
    assert(check);
    printf("STATUS %u FUNCTIONS %zu SHADOWS %zu FACTS %zu\n",check->status,check->functions,check->shadows,check->count);
    if(check->diagnostic) puts(check->diagnostic);
    fflush(stdout);
    assert(check->status==(unsigned)atoi(argv[2]));
    for(size_t i=0;i<check->count;++i) {
        const NlServiceOwnershipFact *fact=&check->facts[i];
        printf("FLOW %u %u %u\n",fact->action,fact->binding,fact->mode);
    }
    if(check->status==0) {
        bool succeeded=false;
        for(long prefix=0;prefix<1024;++prefix) {
            allocations_before_failure=prefix;
            NlServiceOwnershipCheck *candidate=nl_service_check_ownership(env->service_namespace,env->service_bodies);
            allocations_before_failure=-1;
            if(candidate && candidate->status==0) {
                assert(candidate->count==check->count);
                succeeded=true;
            } else if(candidate) assert(candidate->status==3);
            nl_service_ownership_free(candidate);
            assert(env->service_ownership==check);
            if(succeeded) break;
        }
        assert(succeeded);
    }
    free_environment(env); free_ast(root); free_tokens(tokens,count); free(source);
    free_module_list(modules); clear_module_cache();
    return 0;
}
