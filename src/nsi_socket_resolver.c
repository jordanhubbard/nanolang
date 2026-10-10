#define _POSIX_C_SOURCE 200809L
#include "nsi_socket_resolver_wire.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netdb.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;
static bool lookup_clock(int64_t *out) {
    struct timespec t;
    if(clock_gettime(CLOCK_MONOTONIC,&t))return false;
    *out=(int64_t)t.tv_sec*1000+t.tv_nsec/1000000;return true;
}
static bool lookup_decode(const unsigned char *p,uint16_t port,
    NlSocketResolveResult *result,NlSocketResolution *out) {
    if(memcmp(p,"NLR1",4))return false;
    uint32_t status=nl_resolve_get(p+4),count=nl_resolve_get(p+16);
    int error=(int32_t)nl_resolve_get(p+8),host=(int32_t)nl_resolve_get(p+12);
    if(status!=NL_SOCKET_OK && status!=NL_SOCKET_IO && status!=NL_SOCKET_MEMORY &&
       status!=NL_SOCKET_WOULD_BLOCK && status!=NL_SOCKET_CAPACITY && status!=NL_SOCKET_LIMIT)return false;
    if(host<0 || (error!=EAI_SYSTEM && host) || count>NL_SOCKET_RESOLVE_MAX ||
       (status==NL_SOCKET_OK?(error || host || !count):count!=0))return false;
    if(error) {
        uint32_t expected=error==EAI_MEMORY?NL_SOCKET_MEMORY:error==EAI_AGAIN?NL_SOCKET_WOULD_BLOCK:NL_SOCKET_IO;
        if(status!=expected)return false;
    } else if(status==NL_SOCKET_MEMORY || status==NL_SOCKET_WOULD_BLOCK)return false;
    NlSocketResolution value={0};value.count=count;
    for(unsigned i=0;i<NL_SOCKET_RESOLVE_MAX;i++) {
        const unsigned char *a=p+20+i*28;
        if(i>=count) { for(unsigned j=0;j<28;j++)if(a[j])return false;continue; }
        uint32_t family=nl_resolve_get(a),wire_port=nl_resolve_get(a+4),scope=nl_resolve_get(a+8);
        if((family!=NL_SOCKET_IPV4 && family!=NL_SOCKET_IPV6) || wire_port!=port)return false;
        if(family==NL_SOCKET_IPV4) {
            if(scope)return false;
            for(unsigned j=4;j<16;j++)if(a[12+j])return false;
        }
        value.addresses[i]=(NlSocketAddress){.family=(NlSocketFamily)family,.port=port,.scope_id=scope};
        memcpy(value.addresses[i].address,a+12,16);
        for(unsigned j=0;j<i;j++) {
            const NlSocketAddress *old=&value.addresses[j];
            if(old->family==(NlSocketFamily)family && old->scope_id==scope && !memcmp(old->address,a+12,16))return false;
        }
    }
    *result=(NlSocketResolveResult){(NlSocketStatus)status,error,host};*out=value;return true;
}
static bool lookup_pipe(int p[2]) {
    if(pipe(p))return false;
    for(unsigned i=0;i<2;i++) {
        if(p[i]<3) {
            int fd=fcntl(p[i],F_DUPFD_CLOEXEC,3);
            if(fd<0)goto fail;
            close(p[i]);p[i]=fd;
        } else if(fcntl(p[i],F_SETFD,FD_CLOEXEC)<0)goto fail;
    }
    if(fcntl(p[0],F_SETFL,O_NONBLOCK)<0)goto fail;
    return true;
fail: {
    int saved=errno;close(p[0]);close(p[1]);errno=saved;return false;
}
}
NlSocketLookupResult nl_socket_resolve_tcp_supervised(NlSocketService *service,
    const char *host,size_t length,uint16_t port,bool allow_lookup,
    const char *helper,unsigned timeout_ms,NlSocketResolution *out) {
    NlSocketLookupResult r={.supervision=NL_LOOKUP_COMPLETE};
    /* I validate caller storage through the existing no-lookup boundary before
     * copying a hostname or starting a process. Numeric success publishes here. */
    r.resolver=nl_socket_resolve_tcp(service,host,length,port,false,out);
    if(r.resolver.status!=NL_SOCKET_RIGHTS || !allow_lookup)return r;
    r.resolver=(NlSocketResolveResult){.status=NL_SOCKET_IO};
    if(!helper || helper[0]!='/' || !timeout_ms || timeout_ms>60000) {
        r.supervision=NL_LOOKUP_ARGUMENT;return r;
    }
    int64_t start;
    if(!lookup_clock(&start)){r.supervision=NL_LOOKUP_SYSTEM;r.supervisor_errno=errno;return r;}
    int64_t deadline=start+timeout_ms;
    int pipefd[2];
    if(!lookup_pipe(pipefd)){r.supervision=NL_LOOKUP_SYSTEM;r.supervisor_errno=errno;return r;}
    posix_spawn_file_actions_t actions;posix_spawnattr_t attr;
    bool have_actions=false,have_attr=false;int error;
    pid_t child=-1;
    char name[NL_SOCKET_HOST_MAX+1],port_text[6];
    memcpy(name,host,length);name[length]=0;snprintf(port_text,sizeof port_text,"%u",(unsigned)port);
    char *argv[]={(char *)helper,name,port_text,NULL};
    if((error=posix_spawn_file_actions_init(&actions)))goto spawn_done;
    have_actions=true;
    if((error=posix_spawnattr_init(&attr)))goto spawn_done;
    have_attr=true;
    if((error=posix_spawn_file_actions_adddup2(&actions,pipefd[1],STDOUT_FILENO)) ||
       (error=posix_spawn_file_actions_addclose(&actions,pipefd[0])) ||
       (error=posix_spawn_file_actions_addclose(&actions,pipefd[1])) ||
       (error=posix_spawnattr_setpgroup(&attr,0)) ||
       (error=posix_spawnattr_setflags(&attr,POSIX_SPAWN_SETPGROUP)))goto spawn_done;
    error=posix_spawn(&child,helper,&actions,&attr,argv,environ);
spawn_done:
    if(have_actions)posix_spawn_file_actions_destroy(&actions);
    if(have_attr)posix_spawnattr_destroy(&attr);
    close(pipefd[1]);
    if(error){close(pipefd[0]);r.supervision=NL_LOOKUP_SYSTEM;r.supervisor_errno=error;return r;}
    unsigned char wire[NL_RESOLVE_WIRE_SIZE+1];size_t used=0;bool eof=false,reaped=false;
    for(;;) {
        int64_t now;
        if(!lookup_clock(&now)){r.supervision=NL_LOOKUP_SYSTEM;r.supervisor_errno=errno;break;}
        if(now>=deadline){r.supervision=NL_LOOKUP_TIMEOUT;break;}
        if(!eof) {
            ssize_t n=read(pipefd[0],wire+used,sizeof wire-used);
            if(n>0) {
                used+=(size_t)n;
                if(used>NL_RESOLVE_WIRE_SIZE){r.supervision=NL_LOOKUP_PROTOCOL;break;}
                continue;
            }
            if(n==0)eof=true;
            else if(errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR) {
                r.supervision=NL_LOOKUP_SYSTEM;r.supervisor_errno=errno;break;
            }
        }
        if(eof) {
            int status;pid_t done=waitpid(child,&status,WNOHANG);
            if(done==child) {
                reaped=true;
                if(!WIFEXITED(status) || WEXITSTATUS(status)!=0){r.supervision=NL_LOOKUP_CHILD;break;}
                NlSocketResolution value;
                if(used!=NL_RESOLVE_WIRE_SIZE || !lookup_decode(wire,port,&r.resolver,&value)) {
                    r.supervision=NL_LOOKUP_PROTOCOL;break;
                }
                if(r.resolver.status==NL_SOCKET_OK)*out=value;
                break;
            }
            if(done<0 && errno!=EINTR){r.supervision=NL_LOOKUP_SYSTEM;r.supervisor_errno=errno;break;}
        }
        int ms=(int)(deadline-now);if(ms>10)ms=10;
        struct pollfd fd={pipefd[0],POLLIN,0};
        if(poll(eof?NULL:&fd,eof?0:1,ms)<0 && errno!=EINTR) {
            r.supervision=NL_LOOKUP_SYSTEM;r.supervisor_errno=errno;break;
        }
    }
    close(pipefd[0]);
    if(!reaped) {
        /* I keep the child unreaped until EOF so its process-group identity
         * cannot be recycled while a descendant still holds the pipe. */
        if(kill(-child,SIGKILL)<0 && errno!=ESRCH && !r.supervisor_errno)r.supervisor_errno=errno;
        int status;pid_t done;
        do { done=waitpid(child,&status,0); } while(done<0 && errno==EINTR);
        if(done<0 && !r.supervisor_errno)r.supervisor_errno=errno;
    }
    return r;
}
