#define SERVICES_DISPATCH_NO_MAIN
#include "test_services_dispatch.c"
#include "../../src/nanoisa/file_host_grant.h"
#define SERVICES_PUBLIC_EMITTER_TEST
#include "services_public_host.h"
static uint8_t *public_wire;static size_t public_size;
static NvmServicesIndirectExecutionReport execute(NvmServicesHostGrant *grant,const NvmServicesIndirectOptions *options,NvmServicesScalar *out){
 return nvm_services_execute_indirect_bytes(grant,public_wire,public_size,options,out);
}
static void grant_boundaries(void){
 NvmServicesHostGrant *g=NULL;NvmServicesHostPolicy p[64];
 for(unsigned i=0;i<64;i++)p[i]=(NvmServicesHostPolicy){i%2?NVM_SERVICES_HOST_TCP:NVM_SERVICES_HOST_FILE,true};
 CHECK(nvm_services_host_grant_create(NULL,1,&g)==NVM_SERVICES_HOST_INVALID && !g);
 CHECK(nvm_services_host_grant_create(p,0,&g)==NVM_SERVICES_HOST_INVALID && !g);
 CHECK(nvm_services_host_grant_create(p,65,&g)==NVM_SERVICES_HOST_INVALID && !g);
 p[0].catalog=(NvmServicesHostCatalog)99;
 CHECK(nvm_services_host_grant_create(p,1,&g)==NVM_SERVICES_HOST_INVALID && !g);p[0].catalog=NVM_SERVICES_HOST_FILE;
 CHECK(nvm_services_host_grant_create(p,64,&g)==NVM_SERVICES_HOST_OK);
 CHECK(nvm_services_host_enter(g,2,NVM_SERVICES_HOST_CATALOG)==NVM_SERVICES_HOST_UNRESOLVED);
 CHECK(nvm_services_host_enter(g,1,1)==NVM_SERVICES_HOST_UNRESOLVED);
 CHECK(nvm_services_host_grant_revoke_instance(g,63)==NVM_SERVICES_HOST_OK);
 CHECK(nvm_services_host_grant_revoke_instance(g,63)==NVM_SERVICES_HOST_OK);
 CHECK(nvm_services_host_grant_destroy(&g)==NVM_SERVICES_HOST_OK && !g);
 CHECK(nvm_services_host_grant_destroy(&g)==NVM_SERVICES_HOST_OK);
 NvmFileHostGrant *foreign=NULL;CHECK(nvm_file_host_grant_create_temporary_files(&foreign)==NVM_FILE_HOST_OK);
 CHECK(nvm_services_host_enter((const NvmServicesHostGrant *)foreign,1,NVM_SERVICES_HOST_CATALOG)==NVM_SERVICES_HOST_UNRESOLVED);
 CHECK(nvm_services_host_grant_revoke((NvmServicesHostGrant *)foreign)==NVM_SERVICES_HOST_UNRESOLVED);
 CHECK(nvm_file_host_grant_destroy(&foreign)==NVM_FILE_HOST_OK && !foreign);
#ifdef SERVICE_ALLOC_TEST
 budget=0;CHECK(nvm_services_host_grant_create(p,64,&g)==NVM_SERVICES_HOST_MEMORY && !g);budget=-1;
 budget=1;CHECK(nvm_services_host_grant_create(p,64,&g)==NVM_SERVICES_HOST_OK);budget=-1;
 CHECK(nvm_services_host_grant_destroy(&g)==NVM_SERVICES_HOST_OK && !g);
#endif
}
int main(int argc,char **argv){
 grant_boundaries();CHECK(argc==8);unsigned mode=(unsigned)strtoul(argv[1],NULL,10),port=(unsigned)strtoul(argv[2],NULL,10);bool ipv6=atoi(argv[3])!=0;
 NvmModule *m=program(mode,port,ipv6);NvmV2Module v={0};CHECK(nvm_v2_from_nvm_module(m,&v)==NVM_V2_OK);
 CHECK(nvm_v2_module_serialize(&v,NULL,0,&public_size)==NVM_V2_OK);public_wire=malloc(public_size);CHECK(public_wire);
 CHECK(nvm_v2_module_serialize(&v,public_wire,public_size,&public_size)==NVM_V2_OK);nvm_v2_module_free(&v);nvm_module_free(m);
 char *generated=NULL,diagnostic[256];CHECK(nvm2c_emit_services_indirect_bytes(public_wire,public_size,"test",&generated,diagnostic,sizeof diagnostic)==NVM_SERVICES_RUNTIME_OK);
 FILE *f=fopen(argv[4],"w");CHECK(f);CHECK(fputs(generated,f)>=0 && !fclose(f));free(generated);
 f=fopen(argv[5],"wb");CHECK(f);CHECK(fwrite(public_wire,1,public_size,f)==public_size && !fclose(f));
 NvmServicesHostPolicy policies[]={{NVM_SERVICES_HOST_FILE,true},{NVM_SERVICES_HOST_TCP,true},{NVM_SERVICES_HOST_FILE,true}};
 NvmServicesHostGrant *grant=NULL;CHECK(nvm_services_host_grant_create(policies,3,&grant)==NVM_SERVICES_HOST_OK);
 NvmServicesIndirectOptions options={1,0};NvmServicesScalar preserved={TAG_INT,999};
 NvmServicesIndirectExecutionReport refused=nvm_services_execute_indirect_bytes(grant,public_wire,public_size-1,&options,&preserved);
 CHECK(refused.runtime.status!=NVM_SERVICES_RUNTIME_OK && !refused.runtime.acquired && preserved.value==999);
 CHECK(nvm_services_host_grant_destroy(&grant)==NVM_SERVICES_HOST_OK && !grant);
 /* Query emission and malformed-byte refusal acquire no resource. */
 CHECK(!public_file_opens && !public_tcp_opens);
 char *sentinel=(void *)&checks;
 CHECK(nvm2c_emit_services_indirect_bytes(public_wire,public_size,"not-valid",&sentinel,diagnostic,sizeof diagnostic)==NVM_SERVICES_RUNTIME_INVALID && sentinel==(void *)&checks);
 unsigned grantmode=(unsigned)strtoul(argv[6],NULL,10);uint64_t fuel=(uint64_t)strtoull(argv[7],NULL,10);
 int result=public_run(execute,grantmode,fuel);free(public_wire);return result;
}
