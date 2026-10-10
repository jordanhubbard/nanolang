#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1
#include "service_driver.h"
#include "service_lowering.h"
#include "nanoisa/file_source_snapshot.h"
#include <sys/stat.h>
#include <unistd.h>

char *nl_service_product_root(const char *executable) {
    const char *configured=getenv("NANO_ROOT");
    if(configured && *configured)return realpath(configured,NULL);
    char *path=strchr(executable,'/')?realpath(executable,NULL):NULL;
    if(!path && !strchr(executable,'/')) {
        const char *search=getenv("PATH");
        for(const char *part=search;part;) {
            const char *end=strchr(part,':');size_t size=end?(size_t)(end-part):strlen(part);
            size_t length=strlen(executable);
            if(size>SIZE_MAX-length-3)return NULL;
            char *candidate=malloc(size+length+3);if(!candidate)return NULL;
            if(size){memcpy(candidate,part,size);candidate[size]=0;}else strcpy(candidate,".");
            strcat(candidate,"/");strcat(candidate,executable);
            if(access(candidate,X_OK)==0)path=realpath(candidate,NULL);
            free(candidate);if(path)break;
            part=end?end+1:NULL;
        }
    }
    if(!path)return NULL;
    char *slash=strrchr(path,'/');if(!slash){free(path);return NULL;}*slash=0;
    slash=strrchr(path,'/');if(!slash){free(path);return NULL;}*slash=0;
    return path;
}
static bool same_file(const char *a,const char *b) {
    struct stat left,right;
    return a && b && stat(a,&left)==0 && stat(b,&right)==0 &&
        left.st_dev==right.st_dev && left.st_ino==right.st_ino;
}
static bool lower(Environment *env,const ASTNode *selection,uint8_t **bytes,size_t *size) {
    NvmModule *module=NULL;
    NlServiceLoweringResult result=nl_service_lower(env->service_namespace,env->service_bodies,
        env->service_ownership,selection,&module);
    if(!result.status)result=nl_service_serialize(module,bytes,size);
    nvm_module_free(module);
    if(result.status)fprintf(stderr,"I cannot lower File source at %d:%d: %s\n",
        result.line,result.column,result.diagnostic);
    return result.status==0;
}
int nl_service_compile(ASTNode *root,Environment *env,bool include_imports,
                       const NlServiceProductOptions *options) {
    if(!root || !env || !options || !env->service_namespace)return 1;
    size_t count=0;
    for(uint32_t owner=0;nl_service_namespace_program(env->service_namespace,owner);owner++) {
        if(same_file(options->output,nl_service_namespace_module(env->service_namespace,owner))) {
            fputs("I will not replace a File source input.\n",stderr);return 1;
        }
        const ASTNode *program=nl_service_namespace_program(env->service_namespace,owner);
        if(!include_imports && program!=root)continue;
        for(int i=0;i<program->as.program.count;i++)count+=program->as.program.items[i]->type==AST_SHADOW;
    }
    for(size_t i=0;i<nl_file_source_snapshot_count(env->service_inputs);i++) {
        size_t size=0;
        const unsigned char *path=nl_file_source_snapshot_bytes(env->service_inputs,i,0,&size);
        if(!path || !size)return 1;
        char *copy=malloc(size+1);if(!copy)return 1;
        memcpy(copy,path,size);copy[size]=0;
        bool alias=same_file(options->output,copy);free(copy);
        if(alias){fputs("I will not replace a File companion input.\n",stderr);return 1;}
    }
    if(count>4096)return 1;
    NlServiceShadow *shadows=calloc(count?count:1,sizeof *shadows);
    if(!shadows)return 1;
    uint8_t *main_bytes=NULL;size_t main_size=0,selected=0;int result=1;
    if(!lower(env,NULL,&main_bytes,&main_size))goto done;
    for(uint32_t owner=0;nl_service_namespace_program(env->service_namespace,owner);owner++) {
        const ASTNode *program=nl_service_namespace_program(env->service_namespace,owner);
        if(!include_imports && program!=root)continue;
        for(int i=0;i<program->as.program.count;i++) {
            const ASTNode *node=program->as.program.items[i];if(node->type!=AST_SHADOW)continue;
            uint8_t *bytes=NULL;size_t size=0;
            if(!lower(env,node,&bytes,&size))goto done;
            shadows[selected++]=(NlServiceShadow){bytes,size,
                nl_service_namespace_module(env->service_namespace,owner),node->as.shadow.function_name};
        }
    }
    result=nl_service_publish(main_bytes,main_size,shadows,count,options);
done:
    for(size_t i=0;i<selected;i++)free((void *)shadows[i].bytes);
    free(shadows);free(main_bytes);return result;
}
