#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1
#include "runtime/service_shadows.h"
#include "runtime/service_policy.h"
#include "nanoisa/services_indirect_public.h"
#include "nanoisa/file_indirect_public.h"
#include "nanoisa/socket_indirect_public.h"
#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* I isolate supervisor faults from bytecode semantics. The source lowerer
 * integration corpus separately executes the real File consumer. */
struct NvmFileHostGrant { unsigned generation; bool revoked; };
static struct NvmFileHostGrant policy;
static unsigned created, disposed;
static uint8_t fault;
static char descendant_marker[512];
NvmFileHostStatus nvm_file_host_grant_create_temporary_files(NvmFileHostGrant **out) {
    if(fault==9)return NVM_FILE_HOST_MEMORY;
    assert(created == disposed);
    policy=(struct NvmFileHostGrant){++created,false}; *out=&policy;
    return NVM_FILE_HOST_OK;
}
NvmFileHostStatus nvm_file_host_grant_revoke(NvmFileHostGrant *grant) {
    assert(grant==&policy && !grant->revoked); grant->revoked=true;
    return fault==10?NVM_FILE_HOST_STATE:NVM_FILE_HOST_OK;
}
NvmFileHostStatus nvm_file_host_grant_destroy(NvmFileHostGrant **grant) {
    if(fault==11)return NVM_FILE_HOST_BUSY;
    assert(*grant==&policy && policy.revoked); ++disposed; *grant=NULL;
    return NVM_FILE_HOST_OK;
}
NvmFileIndirectExecutionReport nvm_file_execute_indirect_bytes(NvmFileHostGrant *grant,
        const uint8_t *bytes,size_t size,const NvmFileIndirectOptions *options,NvmFileScalar *scalar) {
    assert(grant==&policy && !grant->revoked && size==1);
    assert(options->revision==1 && options->instruction_limit>0);
    NvmFileIndirectExecutionReport report={0};
    report.runtime.acquired=true;
    switch (*bytes) {
    case 1: report.runtime.status=NVM_FILE_RUNTIME_ASSERT; break;
    case 2: { struct timespec pause={0,600000000}; nanosleep(&pause,NULL); break; }
    case 3: _exit(0);
    case 4: raise(SIGKILL); break;
    case 5: report.runtime.cleanup.cleanup_failures=1; break;
    case 6: report.runtime.acquired=false; break;
    case 7: scalar->value=9; break;
    case 8: {
        pid_t child=fork(); assert(child>=0);
        if (!child) {
            struct timespec pause={1,0};nanosleep(&pause,NULL);
            FILE *marker=fopen(descendant_marker,"w");
            if(marker)fclose(marker);
            _exit(0);
        }
        raise(SIGKILL);break;
    }
    }
    return report;
}
struct NvmSocketHostGrant { unsigned placeholder; };
static struct NvmSocketHostGrant tcp_policy;
NvmSocketHostStatus nvm_socket_host_grant_create_tcp_connections(NvmSocketHostGrant **out) {
    NvmFileHostGrant *file=NULL;
    NvmFileHostStatus status=nvm_file_host_grant_create_temporary_files(&file);
    if(status!=NVM_FILE_HOST_OK)return NVM_SOCKET_HOST_MEMORY;
    *out=&tcp_policy;return NVM_SOCKET_HOST_OK;
}
NvmSocketHostStatus nvm_socket_host_grant_revoke(NvmSocketHostGrant *grant) {
    assert(grant==&tcp_policy);
    return nvm_file_host_grant_revoke(&policy)==NVM_FILE_HOST_OK?NVM_SOCKET_HOST_OK:NVM_SOCKET_HOST_STATE;
}
NvmSocketHostStatus nvm_socket_host_grant_destroy(NvmSocketHostGrant **grant) {
    assert(*grant==&tcp_policy);NvmFileHostGrant *file=&policy;
    if(nvm_file_host_grant_destroy(&file)!=NVM_FILE_HOST_OK)return NVM_SOCKET_HOST_BUSY;
    *grant=NULL;return NVM_SOCKET_HOST_OK;
}
NvmSocketIndirectExecutionReport nvm_socket_execute_indirect_bytes(NvmSocketHostGrant *grant,
    const uint8_t *bytes,size_t size,const NvmSocketIndirectOptions *options,NvmSocketScalar *scalar) {
    assert(grant==&tcp_policy);
    NvmFileIndirectOptions file_options={options->revision,options->instruction_limit};
    NvmFileScalar value={0};
    NvmFileIndirectExecutionReport result=nvm_file_execute_indirect_bytes(&policy,bytes,size,&file_options,&value);
    NvmSocketIndirectExecutionReport report={0};
    report.runtime.status=(NvmSocketRuntimeStatus)result.runtime.status;
    report.runtime.acquired=result.runtime.acquired;
    report.runtime.cleanup.cleanup_failures=result.runtime.cleanup.cleanup_failures;
    scalar->value=value.value;return report;
}
bool nl_service_policy_read(const uint8_t *bytes,size_t size,bool files,bool tcp,NlServicePolicy *out) {
    if(!bytes || size!=1 || *bytes==12)return false;
    *out=(NlServicePolicy){.profile=3,.count=3,.allowed=files&&tcp,.requires_file=true,.requires_tcp=true,
        .instances={{NVM_SERVICES_HOST_FILE,files},{NVM_SERVICES_HOST_TCP,tcp},{NVM_SERVICES_HOST_FILE,files}}};
    return true;
}
struct NvmServicesHostGrant { unsigned placeholder; };
static struct NvmServicesHostGrant mixed_policy;
NvmServicesHostStatus nvm_services_host_grant_create(const NvmServicesHostPolicy *items,size_t count,NvmServicesHostGrant **out) {
    assert(count==3 && items[0].catalog==NVM_SERVICES_HOST_FILE && items[1].catalog==NVM_SERVICES_HOST_TCP && items[2].catalog==NVM_SERVICES_HOST_FILE);
    assert(items[0].allowed && items[1].allowed && items[2].allowed);
    NvmFileHostGrant *file=NULL;
    if(nvm_file_host_grant_create_temporary_files(&file)!=NVM_FILE_HOST_OK)return NVM_SERVICES_HOST_MEMORY;
    *out=&mixed_policy;return NVM_SERVICES_HOST_OK;
}
NvmServicesHostStatus nvm_services_host_grant_revoke(NvmServicesHostGrant *grant) {
    assert(grant==&mixed_policy);
    return nvm_file_host_grant_revoke(&policy)==NVM_FILE_HOST_OK?NVM_SERVICES_HOST_OK:NVM_SERVICES_HOST_STATE;
}
NvmServicesHostStatus nvm_services_host_grant_destroy(NvmServicesHostGrant **grant) {
    assert(*grant==&mixed_policy);NvmFileHostGrant *file=&policy;
    if(nvm_file_host_grant_destroy(&file)!=NVM_FILE_HOST_OK)return NVM_SERVICES_HOST_BUSY;
    *grant=NULL;return NVM_SERVICES_HOST_OK;
}
NvmServicesIndirectExecutionReport nvm_services_execute_indirect_bytes(NvmServicesHostGrant *grant,
    const uint8_t *bytes,size_t size,const NvmServicesIndirectOptions *options,NvmServicesScalar *scalar) {
    assert(grant==&mixed_policy);
    NvmFileIndirectOptions file_options={options->revision,options->instruction_limit};NvmFileScalar value={0};
    NvmFileIndirectExecutionReport result=nvm_file_execute_indirect_bytes(&policy,bytes,size,&file_options,&value);
    NvmServicesIndirectExecutionReport report={0};report.runtime.status=(NvmServicesRuntimeStatus)result.runtime.status;
    report.runtime.acquired=result.runtime.acquired;report.runtime.cleanup.cleanup_failures=result.runtime.cleanup.cleanup_failures;
    scalar->value=value.value;return report;
}
static unsigned occurrences(const char *text,const char *needle) {
    unsigned count=0;const char *p=text;
    while((p=strstr(p,needle))) {++count;p+=strlen(needle);}return count;
}
int main(void) {
    char directory[]="/tmp/nano-service-supervisor-XXXXXX";
    assert(mkdtemp(directory));
    char path[512];assert(snprintf(path,sizeof path,"%s/records",directory)<(int)sizeof path);
    assert(snprintf(descendant_marker,sizeof descendant_marker,"%s/escaped",directory)<(int)sizeof descendant_marker);
    uint8_t code=0;NlServiceShadow suite[2]={{&code,1,"root\norigin","temp"},{&code,1,"import","close"}};
    NlServiceShadowReport report=nl_service_run_shadows(suite,2,false,path);
    assert(report.status==NL_SERVICE_SHADOW_DENIED && access(path,F_OK)!=0);
    assert(setenv("NANO_SHADOW_TIMEOUT_SECONDS","1",1)==0);
    report=nl_service_run_shadows(suite,2,true,path);
    assert(report.status==NL_SERVICE_SHADOW_OK && report.completed==2);
    struct stat st;assert(stat(path,&st)==0 && (st.st_mode&0777)==0600);
    FILE *file=fopen(path,"rb");assert(file);char log[2048]={0};
    assert(fread(log,1,sizeof log-1,file)>0 && fclose(file)==0);
    assert(occurrences(log,"SELECT ")==2 && occurrences(log,"START ")==2 && occurrences(log,"DONE ")==2);
    assert(strstr(log,"726f6f740a6f726967696e") && !strstr(log,"root\norigin"));
    report=nl_service_run_shadows(suite,2,true,path);
    assert(report.status==NL_SERVICE_SHADOW_SYSTEM);
    assert(stat(path,&st)==0 && st.st_size==(off_t)strlen(log));
    assert(unlink(path)==0);
    for(code=1;code<=11;code++) {
        fault=code;
        report=nl_service_run_shadows(suite,2,true,path);
        NlServiceShadowStatus expected=code==2?NL_SERVICE_SHADOW_TIMEOUT:
            (code==3||code==4||code==8)?NL_SERVICE_SHADOW_SYSTEM:NL_SERVICE_SHADOW_FAILED;
        assert(report.status==expected && report.completed==0);
        file=fopen(path,"rb");assert(file);memset(log,0,sizeof log);
        assert(fread(log,1,sizeof log-1,file)>0 && fclose(file)==0);
        /* Two individually sub-deadline invocations still time out as a suite. */
        assert(occurrences(log,"DONE ")== (code==2?1u:0u));
        assert(unlink(path)==0);
    }
    fault=0;
    struct timespec settle={1,200000000};nanosleep(&settle,NULL);
    assert(access(descendant_marker,F_OK)!=0);
    assert(setenv("NANO_SHADOW_TIMEOUT_SECONDS","0",1)==0);
    assert(nl_service_run_shadows(suite,2,true,path).status==NL_SERVICE_SHADOW_INVALID);
    assert(access(path,F_OK)!=0);assert(unsetenv("NANO_SHADOW_TIMEOUT_SECONDS")==0);
    assert(nl_service_run_shadows(NULL,1,true,path).status==NL_SERVICE_SHADOW_INVALID);
    assert(symlink("absent",path)==0);
    assert(nl_service_run_shadows(suite,2,true,path).status==NL_SERVICE_SHADOW_SYSTEM);
    assert(unlink(path)==0);
    assert(nl_service_run_shadows(NULL,0,false,path).status==NL_SERVICE_SHADOW_OK);
    assert(unlink(path)==0);
    assert(nl_service_run_catalog_shadows(suite,2,2,false,path).status==NL_SERVICE_SHADOW_DENIED);
    assert(nl_service_run_catalog_shadows(suite,2,3,true,path).status==NL_SERVICE_SHADOW_INVALID);
    assert(setenv("NANO_SHADOW_TIMEOUT_SECONDS","1",1)==0);
    for(code=0;code<=11;code++) {
        fault=code;
        report=nl_service_run_catalog_shadows(suite,2,2,true,path);
        NlServiceShadowStatus expected=code==0?NL_SERVICE_SHADOW_OK:code==2?NL_SERVICE_SHADOW_TIMEOUT:
            (code==3||code==4||code==8)?NL_SERVICE_SHADOW_SYSTEM:NL_SERVICE_SHADOW_FAILED;
        assert(report.status==expected && report.completed==(code==0?2u:0u));
        assert(unlink(path)==0);
    }
    fault=0;code=0;
    assert(nl_service_run_mixed_shadows(suite,2,false,true,path).status==NL_SERVICE_SHADOW_DENIED);
    assert(nl_service_run_mixed_shadows(suite,2,true,false,path).status==NL_SERVICE_SHADOW_DENIED);
    assert(access(path,F_OK)!=0);
    code=12;assert(nl_service_run_mixed_shadows(suite,2,true,true,path).status==NL_SERVICE_SHADOW_INVALID);
    assert(access(path,F_OK)!=0);
    for(code=0;code<=11;code++) {
        fault=code;report=nl_service_run_mixed_shadows(suite,2,true,true,path);
        NlServiceShadowStatus expected=code==0?NL_SERVICE_SHADOW_OK:code==2?NL_SERVICE_SHADOW_TIMEOUT:
            (code==3||code==4||code==8)?NL_SERVICE_SHADOW_SYSTEM:NL_SERVICE_SHADOW_FAILED;
        assert(report.status==expected && report.completed==(code==0?2u:0u));assert(unlink(path)==0);
    }
    assert(unsetenv("NANO_SHADOW_TIMEOUT_SECONDS")==0);
    assert(rmdir(directory)==0);

    puts("I pass File/TCP/mixed shadow supervision, whole-suite timeout and refusal controls.");
    return 0;
}
