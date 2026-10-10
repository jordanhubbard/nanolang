#define _POSIX_C_SOURCE 200809L
#include "../src/nsi_socket_resolver_wire.h"
#include <errno.h>
#include <netdb.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);exit(1);} } while(0)

static int worker(const char *name,unsigned port) {
    if(!strcmp(name,"hang.test")) { for(;;)pause(); }
    if(!strcmp(name,"crash.test"))return 7;
    if(!strcmp(name,"signal.test")) { raise(SIGKILL);return 8; }
    NlSocketResolution value={.count=1};
    value.addresses[0]=(NlSocketAddress){.family=NL_SOCKET_IPV4,.address={127,0,0,1},.port=(uint16_t)port};
    NlSocketResolveResult result={.status=NL_SOCKET_OK};
    if(!strcmp(name,"error.test"))result=(NlSocketResolveResult){NL_SOCKET_IO,EAI_SYSTEM,EACCES};
    unsigned char wire[NL_RESOLVE_WIRE_SIZE+1];nl_resolve_encode(wire,result,&value);wire[NL_RESOLVE_WIRE_SIZE]=0;
    size_t size=NL_RESOLVE_WIRE_SIZE;
    if(!strcmp(name,"short.test"))size--;
    if(!strcmp(name,"extra.test"))size++;
    if(!strcmp(name,"version.test"))wire[3]='2';
    if(!strcmp(name,"family.test"))nl_resolve_put(wire+20,99);
    if(!strcmp(name,"port.test"))nl_resolve_put(wire+24,port+1);
    if(!strcmp(name,"count.test"))nl_resolve_put(wire+16,17);
    if(!strcmp(name,"tail.test"))wire[NL_RESOLVE_WIRE_SIZE-1]=1;
    if(!strcmp(name,"errno.test"))nl_resolve_put(wire+12,EACCES);
    if(!strcmp(name,"status.test"))nl_resolve_put(wire+4,99);
    if(!strcmp(name,"domain.test")) {
        nl_resolve_encode(wire,(NlSocketResolveResult){NL_SOCKET_MEMORY,EAI_SYSTEM,EACCES},&value);
    }
    if(!strcmp(name,"duplicate.test")) { nl_resolve_put(wire+16,2);memcpy(wire+48,wire+20,28); }
    CHECK(write(1,wire,size)==(ssize_t)size);
    if(!strcmp(name,"exit.test"))return 3;
    if(!strcmp(name,"eofhang.test")) { close(1);for(;;)pause(); }
    if(!strcmp(name,"pipehang.test")) { for(;;)pause(); }
    return 0;
}
static int64_t milliseconds(void) {
    struct timespec t;CHECK(!clock_gettime(CLOCK_MONOTONIC,&t));return (int64_t)t.tv_sec*1000+t.tv_nsec/1000000;
}
int main(int argc,char **argv) {
    if(argc==3)return worker(argv[1],(unsigned)strtoul(argv[2],NULL,10));
    CHECK(argc==2 && argv[0][0]=='/' && argv[1][0]=='/');
    NlSocketService *s=NULL;CHECK(nl_socket_service_create(&s).status==NL_SOCKET_OK);
    NlSocketPair pair;CHECK(nl_socket_acquire_pair(s,NL_CAP_READ|NL_CAP_WRITE,NL_CAP_READ|NL_CAP_WRITE,&pair).status==NL_SOCKET_OK);
    NlSocketResolution out,before;memset(&before,0x5a,sizeof before);out=before;
    NlSocketLookupResult r=nl_socket_resolve_tcp_supervised(s,"127.0.0.1",9,80,false,NULL,0,&out);
    CHECK(r.supervision==NL_LOOKUP_COMPLETE && r.resolver.status==NL_SOCKET_OK && out.count==1);
    out=before;r=nl_socket_resolve_tcp_supervised(s,"localhost",9,80,false,NULL,0,&out);
    CHECK(r.supervision==NL_LOOKUP_COMPLETE && r.resolver.status==NL_SOCKET_RIGHTS && !memcmp(&out,&before,sizeof out));
    r=nl_socket_resolve_tcp_supervised(s,"localhost",9,80,true,argv[1],3000,&out);
    CHECK(r.supervision==NL_LOOKUP_COMPLETE && r.resolver.status==NL_SOCKET_OK && out.count && out.count<=16);
    for(size_t i=0;i<out.count;i++)CHECK(out.addresses[i].port==80);
    const char *bad[]={"short.test","extra.test","version.test","family.test","port.test","count.test","tail.test","errno.test","duplicate.test","status.test","domain.test"};
    for(unsigned i=0;i<sizeof bad/sizeof *bad;i++) {
        out=before;r=nl_socket_resolve_tcp_supervised(s,bad[i],strlen(bad[i]),80,true,argv[0],2000,&out);
        CHECK(r.supervision==NL_LOOKUP_PROTOCOL && !memcmp(&out,&before,sizeof out));
    }
    const char *hung[]={"hang.test","eofhang.test","pipehang.test"};
    for(unsigned i=0;i<3;i++) {
        out=before;int64_t start=milliseconds();
        r=nl_socket_resolve_tcp_supervised(s,hung[i],strlen(hung[i]),80,true,argv[0],80,&out);
        CHECK(r.supervision==NL_LOOKUP_TIMEOUT && milliseconds()-start<2000 && !memcmp(&out,&before,sizeof out));
        int status;CHECK(waitpid(-1,&status,WNOHANG)==-1 && errno==ECHILD);
    }
    const char *failed[]={"crash.test","exit.test","signal.test"};
    for(unsigned i=0;i<3;i++) {
        out=before;r=nl_socket_resolve_tcp_supervised(s,failed[i],strlen(failed[i]),80,true,argv[0],2000,&out);
        CHECK(r.supervision==NL_LOOKUP_CHILD && !memcmp(&out,&before,sizeof out));
    }
    out=before;r=nl_socket_resolve_tcp_supervised(s,"error.test",10,80,true,argv[0],2000,&out);
    CHECK(r.supervision==NL_LOOKUP_COMPLETE && r.resolver.status==NL_SOCKET_IO && r.resolver.resolver_error==EAI_SYSTEM && r.resolver.host_errno==EACCES && !memcmp(&out,&before,sizeof out));
    r=nl_socket_resolve_tcp_supervised(s,"localhost",9,80,true,"/nonexistent-nanolang-resolver",100,&out);
    CHECK(r.supervision==NL_LOOKUP_SYSTEM && r.supervisor_errno==ENOENT && !memcmp(&out,&before,sizeof out));
    r=nl_socket_resolve_tcp_supervised(s,"localhost",9,80,true,argv[1],0,&out);
    CHECK(r.supervision==NL_LOOKUP_ARGUMENT && !memcmp(&out,&before,sizeof out));
    r=nl_socket_resolve_tcp_supervised(s,"localhost",9,80,true,"relative",1,&out);
    CHECK(r.supervision==NL_LOOKUP_ARGUMENT && !memcmp(&out,&before,sizeof out));
    r=nl_socket_resolve_tcp_supervised(s,"localhost",9,80,true,argv[1],60001,&out);
    CHECK(r.supervision==NL_LOOKUP_ARGUMENT && !memcmp(&out,&before,sizeof out));
    for(int fd=0;fd<3;fd++) {
        int saved=dup(fd);CHECK(saved>=0);CHECK(!close(fd));
        r=nl_socket_resolve_tcp_supervised(s,"valid.test",10,80,true,argv[0],2000,&out);
        CHECK(dup2(saved,fd)==fd);CHECK(!close(saved));
        CHECK(r.supervision==NL_LOOKUP_COMPLETE && r.resolver.status==NL_SOCKET_OK && out.count==1);
    }
    CHECK(nl_socket_send_byte(s,&pair.endpoints[0],42).status==NL_SOCKET_OK);
    uint8_t byte=0;CHECK(nl_socket_receive_byte(s,&pair.endpoints[1],&byte).status==NL_SOCKET_OK && byte==42);
    CHECK(nl_socket_service_destroy(s).status==NL_SOCKET_OK);
    puts("PASS supervised localhost, authority, deadline/reaping, worker protocol and preserved socket owners");
    return 0;
}
