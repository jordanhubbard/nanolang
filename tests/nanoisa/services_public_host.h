#include <nanolang/services/nanoisa/services_indirect_native_public.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/socket.h>
#include <unistd.h>
static unsigned public_file_opens,public_file_closes,public_tcp_opens,public_tcp_closes;
static bool public_close_fault;
typedef NvmServicesIndirectExecutionReport (*PublicExecution)(NvmServicesHostGrant *,const NvmServicesIndirectOptions *,NvmServicesScalar *);
static PublicExecution public_execution;
static void public_require(bool condition){if(!condition)abort();}
static void public_busy(void){
 NvmServicesIndirectExecutionReport r=public_execution((void *)1,(void *)1,(void *)1);
 public_require(r.runtime.status==NVM_SERVICES_RUNTIME_BUSY && !r.runtime.acquired && !r.instruction_limit);
 public_require(nvm_services_host_grant_create((void *)1,SIZE_MAX,(void *)1)==NVM_SERVICES_HOST_BUSY);
 public_require(nvm_services_host_grant_revoke((void *)1)==NVM_SERVICES_HOST_BUSY);
 public_require(nvm_services_host_grant_revoke_instance((void *)1,SIZE_MAX)==NVM_SERVICES_HOST_BUSY);
 public_require(nvm_services_host_grant_destroy((void *)1)==NVM_SERVICES_HOST_BUSY);
#ifdef SERVICES_PUBLIC_EMITTER_TEST
 public_require(nvm2c_emit_services_indirect_bytes((void *)1,SIZE_MAX,(void *)1,(void *)1,(void *)1,SIZE_MAX)==NVM_SERVICES_RUNTIME_BUSY);
#endif
}
FILE *services_public_tmpfile(void){public_busy();FILE *f=tmpfile();if(f)public_file_opens++;return f;}
int services_public_fclose(FILE *f){int r=fclose(f);public_file_closes++;if(public_close_fault){errno=EIO;return EOF;}return r;}
int services_public_socket(int domain,int type,int protocol){public_busy();int fd=socket(domain,type,protocol);if(fd>=0)public_tcp_opens++;return fd;}
int services_public_close(int fd){int r=close(fd);public_tcp_closes++;if(public_close_fault){errno=EIO;return -1;}return r;}
static int public_run(PublicExecution execute,unsigned mode,uint64_t fuel){
 public_execution=execute;
 NvmServicesHostPolicy policies[]={{NVM_SERVICES_HOST_FILE,true},{NVM_SERVICES_HOST_TCP,true},{NVM_SERVICES_HOST_FILE,true}};
 NvmServicesHostGrant *grant=NULL;
 if(mode==4)policies[2].allowed=false;
 if(mode==5)policies[1].allowed=false;
 if(mode==6)policies[1].catalog=NVM_SERVICES_HOST_FILE;
 public_require(nvm_services_host_grant_create(policies,mode==7?2:3,&grant)==NVM_SERVICES_HOST_OK);
 /* I copy policy inputs, rather than borrowing their later contents. */
 policies[0].catalog=NVM_SERVICES_HOST_TCP;policies[0].allowed=false;
 if(mode==2)public_require(nvm_services_host_grant_revoke(grant)==NVM_SERVICES_HOST_OK);
 if(mode==8)public_require(nvm_services_host_grant_revoke_instance(grant,0)==NVM_SERVICES_HOST_OK);
 public_require(nvm_services_host_grant_revoke_instance(grant,64)==NVM_SERVICES_HOST_INVALID);
 if(mode==3){public_require(nvm_services_host_enter_query()==NVM_SERVICES_HOST_OK);public_busy();}
 public_close_fault=mode==9;
 NvmServicesIndirectOptions options={1,fuel};NvmServicesScalar out={TAG_INT,999};
 NvmServicesIndirectExecutionReport r=execute(mode==1?NULL:grant,&options,&out);
 if(mode==3)nvm_services_host_leave();
 public_require(nvm_services_host_grant_destroy(&grant)==NVM_SERVICES_HOST_OK && !grant);
 public_require(public_file_opens==public_file_closes && public_tcp_opens==public_tcp_closes);
 printf("{\"status\":%u,\"acquired\":%u,\"steps\":%llu,\"fuel\":%u,\"cleanup\":%llu,\"value\":%lld,\"tag\":%u,\"files\":%u,\"sockets\":%u}\n",
 r.runtime.status,r.runtime.acquired,(unsigned long long)r.instructions_started,r.fuel_exhausted,
 (unsigned long long)r.runtime.cleanup.cleanup_failures,(long long)out.value,out.tag,public_file_opens,public_tcp_opens);
 return 0;
}
