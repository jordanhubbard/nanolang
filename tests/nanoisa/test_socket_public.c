#include "../../src/nanoisa/socket_indirect_public.h"
#include "../../src/nanoisa/socket_host_grant_internal.h"
#include "../../src/nanoisa/file_host_grant_internal.h"
#include <pthread.h>
#include <stdlib.h>
static NvmSocketHostGrant *public_grant;
static void public_busy_probe(void);
#define SOCKET_DISPATCH_ON_OPEN() public_busy_probe()
#define SOCKET_DISPATCH_MAIN socket_fixture_main
#define nvm2c_socket_indirect_private_emit public_emit_bridge
#define nvm_socket_vm_indirect_execute public_vm_bridge
#include "test_socket_dispatch.c"
#undef nvm2c_socket_indirect_private_emit
#undef nvm_socket_vm_indirect_execute
static void public_busy_probe(void){
 NvmSocketIndirectExecutionReport r=nvm_socket_execute_indirect_bytes((void *)1,(void *)1,1,(void *)1,(void *)1);
 CHECK(r.runtime.status==NVM_SOCKET_RUNTIME_BUSY && !r.runtime.acquired && !r.instruction_limit && !r.instructions_started);
 CHECK(nvm2c_emit_socket_indirect_bytes((void *)1,1,(void *)1,(void *)1,(void *)1,1)==NVM_SOCKET_RUNTIME_BUSY);
 CHECK(nvm_socket_host_grant_create_tcp_connections((void *)1)==NVM_SOCKET_HOST_BUSY);
 CHECK(nvm_socket_host_grant_revoke((void *)1)==NVM_SOCKET_HOST_BUSY);
 CHECK(nvm_socket_host_grant_destroy((void *)1)==NVM_SOCKET_HOST_BUSY);
 CHECK(nvm_file_host_enter_query()==NVM_FILE_HOST_BUSY);
 CHECK(nvm_file_host_grant_create_temporary_files((void *)1)==NVM_FILE_HOST_BUSY);
}
static void *contender(void *unused){(void)unused;public_busy_probe();return NULL;}
NvmSocketRuntimeStatus public_emit_bridge(const uint8_t *bytes,size_t size,char **out,char *err,size_t n){
 return nvm2c_emit_socket_indirect_bytes(bytes,size,"test",out,err,n);
}
NvmSocketIndirectExecutionReport public_vm_bridge(const uint8_t *bytes,size_t size,const NvmSocketIndirectOptions *options,NvmSocketRuntimeView *out){
 NvmSocketScalar scalar={TAG_INT,12345};
 NvmSocketIndirectExecutionReport r=nvm_socket_execute_indirect_bytes(public_grant,bytes,size,options,&scalar);
 if(r.runtime.status==NVM_SOCKET_RUNTIME_OK){*out=(NvmSocketRuntimeView){.initialized=true,.fields=1,.values={scalar.value}};out->type.tag=scalar.tag;}
 else CHECK(scalar.tag==TAG_INT && scalar.value==12345);
 return r;
}
int main(int argc,char **argv){
 NvmSocketHostGrant *grant=NULL;CHECK(nvm_socket_host_grant_create_tcp_connections(&grant)==NVM_SOCKET_HOST_OK);
 CHECK(nvm_socket_host_enter(grant,2,NVM_SOCKET_HOST_CATALOG)==NVM_SOCKET_HOST_UNRESOLVED);
 CHECK(nvm_socket_host_enter(grant,NVM_SOCKET_HOST_ABI,1)==NVM_SOCKET_HOST_UNRESOLVED);
 CHECK(nvm_file_host_enter_query()==NVM_FILE_HOST_OK);public_busy_probe();pthread_t thread;
 CHECK(pthread_create(&thread,NULL,contender,NULL)==0 && pthread_join(thread,NULL)==0);nvm_file_host_leave();
 CHECK(nvm_socket_host_enter_query()==NVM_SOCKET_HOST_OK);public_busy_probe();nvm_socket_host_leave();
 int mode=argc==6?atoi(argv[5]):0;public_grant=mode==1?NULL:grant;
 if(mode==2){CHECK(nvm_socket_host_grant_revoke(grant)==NVM_SOCKET_HOST_OK);CHECK(nvm_socket_host_grant_revoke(grant)==NVM_SOCKET_HOST_OK);}
 if(mode==3)CHECK(nvm_file_host_enter_query()==NVM_FILE_HOST_OK);
 int result=socket_fixture_main(argc==6?5:argc,argv);
 if(mode==3)nvm_file_host_leave();
 CHECK(nvm_socket_host_grant_destroy(&grant)==NVM_SOCKET_HOST_OK && !grant);
 CHECK(nvm_socket_host_grant_destroy(&grant)==NVM_SOCKET_HOST_OK);return result;
}
