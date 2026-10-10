#include "nanolang.h"
#include "service_lowering.h"
#include "nanoisa/file_cyclic_public.h"
#include "nanoisa/socket_indirect_native_public.h"
#include "nanoisa/services_indirect_native_public.h"
#include "runtime/service_shadows.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
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
static void reject_reference_map(NvmModule *module) {
    NvmFileCyclicReport *sentinel=(NvmFileCyclicReport *)(uintptr_t)1;
    assert(nvm_file_cyclic_analyze(module,&sentinel)!=NVM_FILE_FLOW_OK);
    assert(sentinel==(NvmFileCyclicReport *)(uintptr_t)1);
}
static void check_reference_maps(NvmModule *module) {
    unsigned checked=0;
    for(uint32_t f=0;f<module->function_count;f++) {
        NvmFunctionEntry fn=module->functions[f];uint32_t pc=0;uint16_t instruction=0;
        while(pc<fn.code_length) {
            uint8_t *code=module->code+fn.code_offset+pc;DecodedInstruction decoded;
            uint32_t width=isa_decode(code,fn.code_length-pc,&decoded);assert(width);
            if(decoded.opcode==OP_FILE_CALL_REFS) {
                uint32_t index=decoded.operands[1].u32,size=module->string_lengths[index];
                uint8_t *map=(uint8_t *)module->strings[index],saved[SL_LOCALS*2];
                assert(size && size<=sizeof saved);memcpy(saved,map,size);
                NvmFileCyclicReport *query=NULL;assert(nvm_file_cyclic_analyze(module,&query)==NVM_FILE_FLOW_OK);
                NvmFileCodeInstruction retained;assert(nvm_file_cyclic_instruction(query,f,instruction,&retained));
                module->string_lengths[index]=size-1;reject_reference_map(module);module->string_lengths[index]=size;
                int first=-1;
                for(uint32_t i=0;i<size/2;i++) {
                    uint16_t value=(uint16_t)saved[2*i]|((uint16_t)saved[2*i+1]<<8);
                    if(value==UINT16_MAX) {map[2*i]=0;map[2*i+1]=0;reject_reference_map(module);}
                    else {
                        map[2*i]=0;map[2*i+1]=1;reject_reference_map(module);
                        map[2*i]=255;map[2*i+1]=255;reject_reference_map(module);
                        if(first>=0){
                            map[2*i]=saved[2*first];map[2*i+1]=saved[2*first+1];
                            NvmFileFlowDeclaration a,b;
                            assert(nvm_file_cyclic_local(query,decoded.operands[0].u32,(uint16_t)first,&a));
                            assert(nvm_file_cyclic_local(query,decoded.operands[0].u32,(uint16_t)i,&b));
                            if(a.mode==1 && b.mode==1){
                                NvmFileCyclicReport *alias=NULL;
                                assert(nvm_file_cyclic_analyze(module,&alias)==NVM_FILE_FLOW_OK);
                                nvm_file_cyclic_free(alias);
                            } else reject_reference_map(module);
                        }
                        else first=(int)i;
                    }
                    memcpy(map,saved,size);
                }
                assert(first>=0);
                map[2*first]^=1;
                NvmFileCodeInstruction copy;assert(nvm_file_cyclic_instruction(query,f,instruction,&copy));
                assert(!memcmp(copy.call_references,retained.call_references,sizeof copy.call_references));
                memcpy(map,saved,size);nvm_file_cyclic_free(query);
                sl_wr32(code+5,module->string_count);reject_reference_map(module);sl_wr32(code+5,index);
                checked++;
            }
            pc+=width;instruction++;
        }
    }
    if(checked)printf("REFERENCE MAPS %u: malformed lengths, slots, value sentinels, overlap and owned copies checked\n",checked);
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
    if (!strcmp(argv[2],"all")) {
        NlServiceShadow shadows[64];size_t selected=0;
        for(uint32_t owner=0;nl_service_namespace_program(env->service_namespace,owner);owner++) {
            const ASTNode *program=nl_service_namespace_program(env->service_namespace,owner);
            for(int i=0;i<program->as.program.count;i++) {
                const ASTNode *node=program->as.program.items[i];
                if(node->type!=AST_SHADOW)continue;
                assert(selected<64);NvmModule *lowered=NULL;uint8_t *wire=NULL;size_t wire_size=0;
                assert(!nl_service_lower(env->service_namespace,env->service_bodies,
                    env->service_ownership,node,&lowered).status);
                assert(!nl_service_serialize(lowered,&wire,&wire_size).status);
                nvm_module_free(lowered);
                shadows[selected++]=(NlServiceShadow){wire,wire_size,
                    nl_service_namespace_module(env->service_namespace,owner),node->as.shadow.function_name};
            }
        }
        NlServiceShadowReport tested=nl_service_run_shadows(shadows,selected,true,argv[3]);
        assert(tested.status==NL_SERVICE_SHADOW_OK && tested.completed==selected);
        for(size_t i=0;i<selected;i++)free((void *)shadows[i].bytes);
        printf("SHADOWS %zu\n",selected);goto done;
    }
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
    if(module->service_size==NVM_FILE_NOMINAL_BYTES)check_reference_maps(module);
    uint8_t *bytes=NULL;size_t length=0;result=nl_service_serialize(module,&bytes,&length);
    if(result.status)fprintf(stderr,"SERIALIZE %u %s\n",result.status,result.diagnostic);
    assert(!result.status);
    uint8_t *prior=bytes;size_t prior_size=length;
    recovered=false;
    for(long allocation=0;allocation<16;allocation++) {
        fail_after=allocation;
        NlServiceLoweringResult attempt=nl_service_serialize(module,&prior,&prior_size);
        fail_after=-1;
        if(!attempt.status) {
            assert(prior!=bytes && prior_size==length && !memcmp(prior,bytes,length));
            free(prior);recovered=true;break;
        }
        assert(attempt.status==4 && prior==bytes && prior_size==length);
    }
    assert(recovered);
    if(module->service_size>NVM_SOCKET_NOMINAL_BYTES) {
        char wire_path[8192];assert(snprintf(wire_path,sizeof wire_path,"%s.nvm",argv[3])<(int)sizeof wire_path);
        file=fopen(wire_path,"wb");assert(file);assert(fwrite(bytes,1,length,file)==length);assert(!fclose(file));
        NvmMultiNominalBindings bindings={0};assert(nvm_multi_nominal_decode(module->service_data,module->service_size,&bindings)==NVM_SERVICE_OK);
        NvmServicesHostPolicy policies[64];
        for(uint32_t i=0;i<bindings.count;i++)policies[i]=(NvmServicesHostPolicy){(NvmServicesHostCatalog)bindings.instances[i].catalog,true};
        NvmServicesHostGrant *grant=NULL;assert(nvm_services_host_grant_create(policies,bindings.count,&grant)==NVM_SERVICES_HOST_OK);
        NvmServicesIndirectOptions options={1,100000};NvmServicesScalar scalar={0};
        unsigned descriptors=descriptor_count();
        NvmServicesIndirectExecutionReport report=nvm_services_execute_indirect_bytes(grant,bytes,length,&options,&scalar);
        assert(report.runtime.status==expected && !report.runtime.cleanup.cleanup_failures && descriptors==descriptor_count());
        printf("EXEC %u VALUE %lld\n",report.runtime.status,(long long)scalar.value);
        char diagnostic[256],*native=NULL;
        assert(nvm2c_emit_services_indirect_bytes(bytes,length,"source",&native,diagnostic,sizeof diagnostic)==NVM_SERVICES_RUNTIME_OK);
        file=fopen(argv[3],"wb");assert(file);assert(fwrite(native,1,strlen(native),file)==strlen(native));assert(!fclose(file));
        free(native);free(bytes);nvm_module_free(module);
        assert(nvm_services_host_grant_destroy(&grant)==NVM_SERVICES_HOST_OK);goto done;
    }
    if(module->service_size==NVM_SOCKET_NOMINAL_BYTES) {
        char wire_path[8192];assert(snprintf(wire_path,sizeof wire_path,"%s.nvm",argv[3])<(int)sizeof wire_path);
        file=fopen(wire_path,"wb");assert(file);assert(fwrite(bytes,1,length,file)==length);assert(!fclose(file));
        NvmSocketHostGrant *grant=NULL;assert(nvm_socket_host_grant_create_tcp_connections(&grant)==NVM_SOCKET_HOST_OK);
        NvmSocketIndirectOptions options={1,100000};NvmSocketScalar scalar={0};
        unsigned descriptors=descriptor_count();
        NvmSocketIndirectExecutionReport denied=nvm_socket_execute_indirect_bytes(NULL,bytes,length,&options,&scalar);
        assert(denied.runtime.status==NVM_SOCKET_RUNTIME_INVALID && !denied.runtime.acquired);
        NvmSocketIndirectExecutionReport report=nvm_socket_execute_indirect_bytes(grant,bytes,length,&options,&scalar);
        assert(report.runtime.status==expected && !report.runtime.cleanup.cleanup_failures && descriptors==descriptor_count());
        printf("EXEC %u VALUE %lld\n",report.runtime.status,(long long)scalar.value);
        char diagnostic[256],*native=NULL;
        assert(nvm2c_emit_socket_indirect_bytes(bytes,length,"source",&native,diagnostic,sizeof diagnostic)==NVM_SOCKET_RUNTIME_OK);
        file=fopen(argv[3],"wb");assert(file);assert(fwrite(native,1,strlen(native),file)==strlen(native));assert(!fclose(file));
        free(native);free(bytes);nvm_module_free(module);
        assert(nvm_socket_host_grant_destroy(&grant)==NVM_SOCKET_HOST_OK);goto done;
    }
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
    if (selection) {
        char log_path[4096];
        assert(snprintf(log_path,sizeof log_path,"%s.shadows",argv[3]) < (int)sizeof log_path);
        assert(unlink(log_path)==0 || errno==ENOENT);
        NlServiceShadow shadow={bytes,length,argv[1],argv[2]};
        NlServiceShadowReport tested=nl_service_run_shadows(&shadow,1,false,log_path);
        assert(tested.status==NL_SERVICE_SHADOW_DENIED && access(log_path,F_OK)!=0);
        tested=nl_service_run_shadows(&shadow,1,true,log_path);
        assert(tested.status==(expected?NL_SERVICE_SHADOW_FAILED:NL_SERVICE_SHADOW_OK));
        assert(tested.completed==(expected?0u:1u));
        assert(unlink(log_path)==0);
    }
    char diagnostic[256],*native=NULL;
    assert(nvm2c_emit_file_cyclic_bytes(bytes,length,"source",&native,diagnostic,sizeof diagnostic)==NVM_FILE_RUNTIME_OK);
    file=fopen(argv[3],"wb");assert(file);assert(fwrite(native,1,strlen(native),file)==strlen(native));assert(!fclose(file));
    free(native);free(bytes);nvm_module_free(module);
    assert(nvm_file_host_grant_destroy(&grant)==NVM_FILE_HOST_OK);
done:
    free_environment(env);free_ast(root);free_tokens(tokens,count);free(source);free_module_list(modules);clear_module_cache();
    return 0;
}
