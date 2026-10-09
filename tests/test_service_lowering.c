#include "nanolang.h"
#include "service_lowering.h"
#include "nanoisa/file_cyclic_public.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
static long fail_after=-1;
static void *lower_malloc(size_t size) {
    if(fail_after==0)return NULL;
    if(fail_after>0)--fail_after;
    return malloc(size);
}
static void *lower_calloc(size_t count,size_t size) {
    if(fail_after==0)return NULL;
    if(fail_after>0)--fail_after;
    return calloc(count,size);
}
#define malloc lower_malloc
#define calloc lower_calloc
#include "../src/service_lowering.c"
#undef malloc
#undef calloc
int g_argc;char **g_argv;
static unsigned descriptor_count(void) {
    unsigned count=0;for(int i=0;i<1024;i++)count+=fcntl(i,F_GETFD)!=-1;return count;
}
int main(int argc,char **argv) {
    assert(argc==4 || argc==5);
    bool refusal=argc==5 && !strncmp(argv[4],"lower:",6);
    unsigned expected=argc==5?(unsigned)atoi(argv[4]+(refusal?6:0)):0;
    FILE *file=fopen(argv[1],"rb");assert(file);
    assert(!fseek(file,0,SEEK_END));long size=ftell(file);assert(size>=0);rewind(file);
    char *source=calloc((size_t)size+1,1);assert(source);
    assert(fread(source,1,(size_t)size,file)==(size_t)size);assert(!fclose(file));
    int count=0;Token *tokens=tokenize(source,&count);assert(tokens);
    ASTNode *root=parse_program(tokens,count);assert(root);
    Environment *env=create_environment();ModuleList *modules=create_module_list();
    assert(!process_imports(root,env,modules,argv[1]));
    assert(env->service_bodies && env->service_bodies->status==0);
    assert(env->service_ownership && env->service_ownership->status==0);
    const ASTNode *selection=NULL;
    if(strcmp(argv[2],"main"))for(int i=0;i<root->as.program.count;i++) {
        const ASTNode *node=root->as.program.items[i];
        if(node->type==AST_SHADOW && !strcmp(node->as.shadow.function_name,argv[2]))selection=node;
    }
    assert(selection || !strcmp(argv[2],"main"));
    NvmModule *module=NULL;
    NlServiceLoweringResult result=nl_service_lower(env->service_namespace,env->service_bodies,env->service_ownership,selection,&module);
    if(result.status)fprintf(stderr,"LOWER %u %d:%d %s\n",result.status,result.line,result.column,result.diagnostic);
    if(refusal) {
        assert(result.status==expected && !module);
        printf("REFUSAL %u\n",result.status);goto done;
    }
    assert(!result.status && module);
    NlServiceBodyCheck bad_body=*env->service_bodies;bad_body.status=1;
    NvmModule *sentinel=module;
    assert(nl_service_lower(env->service_namespace,&bad_body,env->service_ownership,selection,&sentinel).status==1 && sentinel==module);
    NlServiceOwnershipCheck bad_owner=*env->service_ownership;bad_owner.status=1;
    assert(nl_service_lower(env->service_namespace,env->service_bodies,&bad_owner,selection,&sentinel).status==1 && sentinel==module);
    ASTNode absent={0};
    assert(nl_service_lower(env->service_namespace,env->service_bodies,env->service_ownership,&absent,&sentinel).status==1 && sentinel==module);
    bool recovered=false;
    for(long prefix=0;prefix<8;prefix++) {
        fail_after=prefix;sentinel=module;
        NlServiceLoweringResult attempt=nl_service_lower(env->service_namespace,env->service_bodies,env->service_ownership,selection,&sentinel);
        fail_after=-1;
        if(!attempt.status){assert(sentinel!=module);nvm_module_free(sentinel);recovered=true;break;}
        assert(attempt.status==4 && sentinel==module);
    }
    assert(recovered);
    uint8_t *bytes=NULL;size_t length=0;result=nl_service_serialize(module,&bytes,&length);
    if(result.status)fprintf(stderr,"SERIALIZE %u %s\n",result.status,result.diagnostic);
    assert(!result.status);
    uint8_t *prior=bytes;size_t prior_size=length;
    fail_after=0;
    assert(nl_service_serialize(module,&prior,&prior_size).status==4 && prior==bytes && prior_size==length);
    fail_after=-1;
    NvmFileHostGrant *grant=NULL;assert(nvm_file_host_grant_create_temporary_files(&grant)==NVM_FILE_HOST_OK);
    NvmFileCyclicOptions options={NVM_FILE_CYCLIC_RUNTIME_REVISION,100000};NvmFileScalar scalar={0};
    unsigned descriptors=descriptor_count();
    NvmFileCyclicExecutionReport denied=nvm_file_execute_cyclic_bytes(NULL,bytes,length,&options,&scalar);
    assert(denied.runtime.status==NVM_FILE_RUNTIME_INVALID && !denied.runtime.acquired);
    assert(scalar.tag==0 && scalar.value==0 && descriptors==descriptor_count());
    NvmFileCyclicExecutionReport report=nvm_file_execute_cyclic_bytes(grant,bytes,length,&options,&scalar);
    assert(descriptors==descriptor_count());
    printf("EXEC %u VALUE %lld FUNCTIONS %u BYTES %zu\n",report.runtime.status,(long long)scalar.value,module->function_count,length);fflush(stdout);
    assert(report.runtime.status==expected);
    assert(!report.runtime.cleanup.cleanup_failures);
    char diagnostic[256],*native=NULL;
    assert(nvm2c_emit_file_cyclic_bytes(bytes,length,"source",&native,diagnostic,sizeof diagnostic)==NVM_FILE_RUNTIME_OK);
    file=fopen(argv[3],"wb");assert(file);assert(fwrite(native,1,strlen(native),file)==strlen(native));assert(!fclose(file));
    free(native);free(bytes);nvm_module_free(module);
    assert(nvm_file_host_grant_destroy(&grant)==NVM_FILE_HOST_OK);
done:
    free_environment(env);free_ast(root);free_tokens(tokens,count);free(source);free_module_list(modules);clear_module_cache();
    return 0;
}
