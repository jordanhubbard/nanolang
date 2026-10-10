#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1
#include "service_product.h"
#include "service_policy.h"
#include "../nanoisa/services_indirect_public.h"
#include "../nanoisa/file_indirect_public.h"
#include "../nanoisa/socket_indirect_public.h"
#include "../nanoisa/nvm_v2_sections.h"
#include "../shell_path.h"
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>

static char *joined(const char *base, const char *name) {
    size_t a=strlen(base), b=strlen(name);
    if(a>SIZE_MAX-b-2)return NULL;
    char *path=malloc(a+b+2);
    if(path)snprintf(path,a+b+2,"%s/%s",base,name);
    return path;
}
static bool write_bytes(int fd,const void *bytes,size_t size) {
    const uint8_t *p=bytes;
    while(size) {
        ssize_t n=write(fd,p,size);
        if(n<0 && errno==EINTR)continue;
        if(n<=0)return false;
        p+=n;size-=(size_t)n;
    }
    return fsync(fd)==0;
}
static bool remove_file(const char *path) {
    return !path || unlink(path)==0 || errno==ENOENT;
}
/* I bound external native compilation separately from the shadow deadline. */
static bool compile_native(const char *command) {
    struct timespec start,now,pause={0,10000000};
    if(clock_gettime(CLOCK_MONOTONIC,&start))return false;
    pid_t child=fork();
    if(child==0) {
        if(setpgid(0,0))_exit(1);
        execl("/bin/sh","sh","-c",command,(char *)NULL);_exit(1);
    }
    if(child<0)return false;
    (void)setpgid(child,child);
    int status=0;bool ok=false;
    for(;;) {
        pid_t done=waitpid(child,&status,WNOHANG);
        if(done==child){ok=WIFEXITED(status)&&WEXITSTATUS(status)==0;break;}
        if(done<0 && errno!=EINTR)break;
        if(clock_gettime(CLOCK_MONOTONIC,&now) || now.tv_sec-start.tv_sec>=120)break;
        nanosleep(&pause,NULL);
    }
    bool terminated=kill(-child,SIGKILL)==0 || errno==ESRCH;
    if(!ok) {
        (void)kill(child,SIGKILL);
        /* I never wait indefinitely for native compiler cleanup. */
        for(unsigned i=0;i<100;i++) {
            pid_t done=waitpid(child,&status,WNOHANG);
            if(done==child || (done<0 && errno==ECHILD))break;
            nanosleep(&pause,NULL);
        }
    }
    return ok && terminated;
}

static const char launcher[] =
"\n#include <stdio.h>\n"
"int main(int argc,char **argv) {\n"
" if(argc!=2 || strcmp(argv[1],\"--allow-temporary-files\")) {\n"
"  fputs(\"I require --allow-temporary-files for this invocation.\\n\",stderr); return 1; }\n"
" NvmFileHostGrant *grant=NULL;\n"
" if(nvm_file_host_grant_create_temporary_files(&grant)!=NVM_FILE_HOST_OK)return 1;\n"
" NvmFileIndirectOptions options={NVM_FILE_INDIRECT_RUNTIME_REVISION,NVM_FILE_INDIRECT_FUEL_MAX};\n"
" NvmFileScalar scalar={0};\n"
" NvmFileIndirectExecutionReport result=nvm_file_indirect_program_product(grant,&options,&scalar);\n"
" NvmFileHostStatus revoked=nvm_file_host_grant_revoke(grant);\n"
" NvmFileHostStatus destroyed=nvm_file_host_grant_destroy(&grant);\n"
" if(result.runtime.status!=NVM_FILE_RUNTIME_OK || !result.runtime.acquired ||\n"
" result.runtime.cleanup.cleanup_failures || revoked!=NVM_FILE_HOST_OK || destroyed!=NVM_FILE_HOST_OK)return 1;\n"
" return (int)((uint64_t)scalar.value & 255u);\n}\n";

static const char socket_launcher[] =
"\n#include <stdio.h>\n"
"int main(int argc,char **argv) {\n"
" if(argc!=2 || strcmp(argv[1],\"--allow-tcp-connections\")) {\n"
"  fputs(\"I require --allow-tcp-connections for this invocation.\\n\",stderr); return 1; }\n"
" NvmSocketHostGrant *grant=NULL;\n"
" if(nvm_socket_host_grant_create_tcp_connections(&grant)!=NVM_SOCKET_HOST_OK)return 1;\n"
" NvmSocketIndirectOptions options={NVM_SOCKET_INDIRECT_RUNTIME_REVISION,NVM_SOCKET_INDIRECT_FUEL_MAX};\n"
" NvmSocketScalar scalar={0};\n"
" NvmSocketIndirectExecutionReport result=nvm_socket_indirect_program_product(grant,&options,&scalar);\n"
" NvmSocketHostStatus revoked=nvm_socket_host_grant_revoke(grant);\n"
" NvmSocketHostStatus destroyed=nvm_socket_host_grant_destroy(&grant);\n"
" if(result.runtime.status!=NVM_SOCKET_RUNTIME_OK || !result.runtime.acquired ||\n"
" result.runtime.cleanup.cleanup_failures || revoked!=NVM_SOCKET_HOST_OK || destroyed!=NVM_SOCKET_HOST_OK)return 1;\n"
" return (int)((uint64_t)scalar.value & 255u);\n}\n";

static bool write_mixed_launcher(int fd,const NlServicePolicy *policy) {
    char catalogs[256];size_t used=0;
    for(size_t i=0;i<policy->count;i++) {
        int n=snprintf(catalogs+used,sizeof catalogs-used,"%s%u",i?",":"",(unsigned)policy->instances[i].catalog);
        if(n<0 || (size_t)n>=sizeof catalogs-used)return false;
        used+=(size_t)n;
    }
    static const char prefix[]="\n#include <stdio.h>\nstatic const unsigned product_catalogs[]={";
    static const char suffix[]=
"};\nint main(int argc,char **argv) {\n"
" bool files=false,tcp=false;\n"
" for(int i=1;i<argc;i++) {\n"
"  if(!strcmp(argv[i],\"--allow-temporary-files\") && !files)files=true;\n"
"  else if(!strcmp(argv[i],\"--allow-tcp-connections\") && !tcp)tcp=true;\n"
"  else return 1;\n"
" }\n"
" NvmServicesHostPolicy policies[sizeof product_catalogs/sizeof product_catalogs[0]];\n"
" for(size_t i=0;i<sizeof policies/sizeof policies[0];i++) {\n"
"  bool allowed=product_catalogs[i]==1?files:tcp;\n"
"  if(!allowed){fprintf(stderr,\"I require %s for this invocation.\\n\",product_catalogs[i]==1?\"--allow-temporary-files\":\"--allow-tcp-connections\");return 1;}\n"
"  policies[i]=(NvmServicesHostPolicy){(NvmServicesHostCatalog)product_catalogs[i],allowed};\n"
" }\n"
" NvmServicesHostGrant *grant=NULL;\n"
" if(nvm_services_host_grant_create(policies,sizeof policies/sizeof policies[0],&grant)!=NVM_SERVICES_HOST_OK)return 1;\n"
" NvmServicesIndirectOptions options={1,NVM_SERVICES_INDIRECT_FUEL_MAX};NvmServicesScalar scalar={0};\n"
" NvmServicesIndirectExecutionReport result=nvm_services_indirect_program_product(grant,&options,&scalar);\n"
" NvmServicesHostStatus revoked=nvm_services_host_grant_revoke(grant);\n"
" NvmServicesHostStatus destroyed=nvm_services_host_grant_destroy(&grant);\n"
" if(result.runtime.status!=NVM_SERVICES_RUNTIME_OK || !result.runtime.acquired || result.runtime.cleanup.cleanup_failures || revoked!=NVM_SERVICES_HOST_OK || destroyed!=NVM_SERVICES_HOST_OK || grant)return 1;\n"
" return (int)((uint64_t)scalar.value&255u);\n}\n";
    return write_bytes(fd,prefix,strlen(prefix)) && write_bytes(fd,catalogs,used) && write_bytes(fd,suffix,strlen(suffix));
}

int nl_service_publish(const uint8_t *bytes,size_t size,const NlServiceShadow *shadows,
                       size_t count,const NlServiceProductOptions *options) {
    if(!bytes || !size || !options || (!shadows&&count) || !options->root ||
       (options->output && !*options->output))return 1;
    NlServicePolicy policy;
    if(!nl_service_policy_read(bytes,size,options->allow_temporary_files,options->allow_tcp_connections,&policy))return 1;
    unsigned catalog=policy.profile;bool tcp=catalog==2,mixed=catalog==3;
    bool allowed=policy.allowed;
    if((count || options->run) && !allowed) {
        if(policy.requires_file && !options->allow_temporary_files)fputs("I require --allow-temporary-files for selected service shadows or execution.\n",stderr);
        if(policy.requires_tcp && !options->allow_tcp_connections)fputs("I require --allow-tcp-connections for selected service shadows or execution.\n",stderr);
        return 1;
    }
    /* Translation validates the main module without executing or granting it. */
    char *native=NULL,diagnostic[256];
    unsigned emitted=mixed?(unsigned)nvm2c_emit_services_indirect_bytes(bytes,size,"product",&native,diagnostic,sizeof diagnostic):tcp?(unsigned)nvm2c_emit_socket_indirect_bytes(bytes,size,"product",&native,diagnostic,sizeof diagnostic):
        (unsigned)nvm2c_emit_file_indirect_bytes(bytes,size,"product",&native,diagnostic,sizeof diagnostic);
    if(emitted!=0) {
        fprintf(stderr,"I cannot validate service output: %s\n",diagnostic);return 1;
    }
    char *parent=strdup(options->output?options->output:"/tmp/nano-file-run");
    if(!parent){free(native);return 1;}
    char *slash=strrchr(parent,'/');
    if(!slash)strcpy(parent,".");else if(slash==parent)slash[1]=0;else *slash=0;
    char *directory=joined(parent,".nano-file.XXXXXX");
    char *product=joined(parent,".nano-publish.XXXXXX");
    free(parent);
    int result=1,product_fd=-1;bool staged=false,product_created=false;
    char *source=NULL,*log=NULL,*include=NULL,*link=NULL,*root_source=NULL,*archive=NULL;
    if(!directory || !product || !mkdtemp(directory))goto done;
    staged=true;
    source=joined(directory,"program.c");log=joined(directory,"shadows.log");
    include=joined(directory,"nanolang");link=include?joined(include,mixed?"services":tcp?"socket":"file"):NULL;
    char *root=realpath(options->root,NULL);
    if(!root)goto done;
    root_source=joined(root,"src");archive=joined(root,mixed?"lib/libnano_services_runtime.a":tcp?"lib/libnano_socket_runtime.a":"lib/libnano_file_runtime.a");free(root);
    if(!source || !log || !include || !link || !root_source || !archive)goto done;
    struct stat headers;
    if(stat(root_source,&headers) || !S_ISDIR(headers.st_mode)) {
        free(root_source);root_source=NULL;
        root=realpath(options->root,NULL);
        if(root){root_source=joined(root,mixed?"include/nanolang/services":tcp?"include/nanolang/socket":"include/nanolang/file");free(root);}
        if(!root_source || stat(root_source,&headers) || !S_ISDIR(headers.st_mode))goto done;
    }
    NlServiceShadowReport tested=mixed?nl_service_run_mixed_shadows(shadows,count,options->allow_temporary_files,options->allow_tcp_connections,log):nl_service_run_catalog_shadows(shadows,count,catalog,allowed,log);
    FILE *records=fopen(log,"rb");
    if(records) {
        char buffer[4096];size_t n;
        while((n=fread(buffer,1,sizeof buffer,records)))fwrite(buffer,1,n,stderr);
        bool bad=ferror(records)!=0;
        if(fclose(records))bad=true;
        if(bad)goto done;
    } else goto done;
    if(tested.status!=NL_SERVICE_SHADOW_OK || tested.completed!=count) {
        fprintf(stderr,"I will not publish after %s shadow failure (status %u).\n",mixed?"mixed-service":tcp?"TCP":"File",tested.status);goto done;
    }
    if(options->output) {
        product_fd=mkstemp(product);if(product_fd<0)goto done;
        product_created=true;
        if(options->emit_nvm) {
            bool written=write_bytes(product_fd,bytes,size);
            if(close(product_fd))written=false;
            product_fd=-1;if(!written)goto done;
        } else {
            if(close(product_fd)){product_fd=-1;goto done;}product_fd=-1;
            int fd=open(source,O_CREAT|O_EXCL|O_WRONLY,0600);if(fd<0)goto done;
            bool written=write_bytes(fd,native,strlen(native)) && (mixed?write_mixed_launcher(fd,&policy):write_bytes(fd,tcp?socket_launcher:launcher,strlen(tcp?socket_launcher:launcher)));
            if(close(fd))written=false;
            if(!written || mkdir(include,0700) || symlink(root_source,link))goto done;
            char *qsource=module_quote_path(source),*qproduct=module_quote_path(product);
            char *qinclude=module_quote_path(directory),*qarchive=module_quote_path(archive);
            char *command=NULL;
            bool ready=qsource&&qproduct&&qinclude&&qarchive;
            if(ready)ready=module_append_fragment(&command,options->compiler?options->compiler:"cc") &&
                module_append_fragment(&command,"-std=c11") &&
                module_append_fragment(&command,options->cflags?options->cflags:"-O1") &&
                module_append_fragment(&command,"-I") && module_append_fragment(&command,qinclude) &&
                module_append_fragment(&command,qsource) && module_append_fragment(&command,qarchive) &&
                module_append_fragment(&command,"-lm") &&
                module_append_fragment(&command,options->ldflags?options->ldflags:"") &&
                module_append_fragment(&command,"-o") && module_append_fragment(&command,qproduct);
            bool compiled=ready && compile_native(command);
            free(command);free(qsource);free(qproduct);free(qinclude);free(qarchive);
            if(!compiled || chmod(product,0700))goto done;
            int completed=open(product,O_RDONLY);
            if(completed<0)goto done;
            bool synced=fsync(completed)==0;
            if(close(completed))synced=false;
            if(!synced)goto done;
        }
    }
    if(options->run) {
        if(mixed) {
        NvmServicesHostGrant *grant=NULL;
        if(nvm_services_host_grant_create(policy.instances,policy.count,&grant)!=NVM_SERVICES_HOST_OK)goto done;
        NvmServicesScalar scalar={0};NvmServicesIndirectOptions execution={1,NVM_SERVICES_INDIRECT_FUEL_MAX};
        NvmServicesIndirectExecutionReport report=nvm_services_execute_indirect_bytes(grant,bytes,size,&execution,&scalar);
        NvmServicesHostStatus revoked=nvm_services_host_grant_revoke(grant);
        NvmServicesHostStatus destroyed=nvm_services_host_grant_destroy(&grant);
        if(report.runtime.status!=NVM_SERVICES_RUNTIME_OK || !report.runtime.acquired || report.runtime.cleanup.cleanup_failures ||
           revoked!=NVM_SERVICES_HOST_OK || destroyed!=NVM_SERVICES_HOST_OK || grant)goto done;
        result=(int)((uint64_t)scalar.value&255u);
        } else if(tcp) {
        NvmSocketHostGrant *grant=NULL;
        if(nvm_socket_host_grant_create_tcp_connections(&grant)!=NVM_SOCKET_HOST_OK)goto done;
        NvmSocketScalar scalar={0};NvmSocketIndirectOptions execution={1,NVM_SOCKET_INDIRECT_FUEL_MAX};
        NvmSocketIndirectExecutionReport report=nvm_socket_execute_indirect_bytes(grant,bytes,size,&execution,&scalar);
        NvmSocketHostStatus revoked=nvm_socket_host_grant_revoke(grant);
        NvmSocketHostStatus destroyed=nvm_socket_host_grant_destroy(&grant);
        if(report.runtime.status!=NVM_SOCKET_RUNTIME_OK || !report.runtime.acquired || report.runtime.cleanup.cleanup_failures ||
           revoked!=NVM_SOCKET_HOST_OK || destroyed!=NVM_SOCKET_HOST_OK || grant)goto done;
        result=(int)((uint64_t)scalar.value&255u);
        } else {
        NvmFileHostGrant *grant=NULL;
        if(nvm_file_host_grant_create_temporary_files(&grant)!=NVM_FILE_HOST_OK)goto done;
        NvmFileScalar scalar={0};NvmFileIndirectOptions execution={1,NVM_FILE_INDIRECT_FUEL_MAX};
        NvmFileIndirectExecutionReport report=nvm_file_execute_indirect_bytes(grant,bytes,size,&execution,&scalar);
        NvmFileHostStatus revoked=nvm_file_host_grant_revoke(grant);
        NvmFileHostStatus destroyed=nvm_file_host_grant_destroy(&grant);
        if(report.runtime.status!=NVM_FILE_RUNTIME_OK || !report.runtime.acquired || report.runtime.cleanup.cleanup_failures ||
           revoked!=NVM_FILE_HOST_OK || destroyed!=NVM_FILE_HOST_OK || grant)goto done;
        result=(int)((uint64_t)scalar.value&255u);
        }
    } else result=0;
    /* I finish all auxiliary cleanup before the final atomic replacement. */
    if(!remove_file(source) || !remove_file(log) || !remove_file(link) ||
       (rmdir(include)&&errno!=ENOENT) || rmdir(directory)) {result=1;goto done;}
    staged=false;
    if(options->output && rename(product,options->output))result=1;
done:
    if(product_fd>=0)close(product_fd);
    if(staged) {
        bool cleaned=remove_file(source);
        if(!remove_file(log))cleaned=false;
        if(!remove_file(link))cleaned=false;
        if(include && rmdir(include) && errno!=ENOENT)cleaned=false;
        if(rmdir(directory))cleaned=false;
        if(!cleaned)fputs("I could not clean private service staging.\n",stderr);
    }
    if(product_created && !remove_file(product))fputs("I could not remove provisional service output.\n",stderr);
    free(source);free(log);free(include);free(link);free(root_source);free(archive);
    free(directory);free(product);free(native);
    return result;
}
