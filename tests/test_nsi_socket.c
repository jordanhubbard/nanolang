/* I use real local sockets with deterministic boundary failures, never raw FFI. */
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static unsigned checks, live_allocations, host_opens, host_closes, close_attempts;
static unsigned manual_closes, io_calls, config_calls;
static int descriptors[128], last_pair[2], fail_pair, fail_config, fail_send, fail_recv;
static long allocation_budget = -1;
static int close_faults[4], close_fault_index;
#define CHECK(x) do { checks++; if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(1); } } while (0)
static void *checked_calloc(size_t n, size_t size) {
    if (!allocation_budget) return NULL;
    if (allocation_budget > 0) allocation_budget--;
    void *p=calloc(n,size);if(p)live_allocations++;return p;
}
static void checked_free(void *p) {if(p){CHECK(live_allocations);live_allocations--;}free(p);errno=ERANGE;}
static unsigned tracked_fd(int fd) {
    unsigned i=0;while(i<128 && descriptors[i]!=fd)i++;CHECK(i<128);return i;
}
static void real_close(int fd, bool manual) {
    unsigned i=tracked_fd(fd);CHECK(close(fd)==0);descriptors[i]=-1;host_closes++;
    if(manual)manual_closes++;
}
static int checked_socketpair(int domain,int type,int protocol,int pair[2]) {
    if(fail_pair){errno=EMFILE;return -1;}
    int rc=socketpair(domain,type,protocol,pair);if(rc)return rc;
    for(unsigned n=0;n<2;n++){unsigned i=0;while(i<128 && descriptors[i]>=0)i++;CHECK(i<128);descriptors[i]=pair[n];host_opens++;last_pair[n]=pair[n];}
    return 0;
}
static int checked_fcntl(int fd,int command,...) {
    config_calls++;if(fail_config && config_calls==(unsigned)fail_config){errno=EACCES;return -1;}
    if(command==F_GETFL || command==F_GETFD)return fcntl(fd,command);
    va_list args;va_start(args,command);int value=va_arg(args,int);va_end(args);return fcntl(fd,command,value);
}
#ifdef __APPLE__
static int checked_setsockopt(int fd,int level,int name,const void *value,socklen_t size) {
    config_calls++;if(fail_config && config_calls==(unsigned)fail_config){errno=EACCES;return -1;}
    return setsockopt(fd,level,name,value,size);
}
#endif
static ssize_t checked_send(int fd,const void *bytes,size_t count,int flags) {
    io_calls++;if(fail_send){errno=fail_send;return -1;}return send(fd,bytes,count,flags);
}
static ssize_t checked_recv(int fd,void *bytes,size_t count,int flags) {
    io_calls++;if(fail_recv){errno=fail_recv;return -1;}return recv(fd,bytes,count,flags);
}
static int checked_close(int fd) {
    close_attempts++;(void)tracked_fd(fd);
    int injected=close_fault_index<4?close_faults[close_fault_index++]:0;
    if(injected<0){errno=-injected;return -1;} /* No real close: harness must recover it explicitly. */
    real_close(fd,false);
    if(injected){errno=injected;return -1;} /* Real close completed before injected report. */
    return 0;
}
#define calloc checked_calloc
#define free checked_free
#define socketpair checked_socketpair
#define fcntl checked_fcntl
#ifdef __APPLE__
#define setsockopt checked_setsockopt
#endif
#define send checked_send
#define recv checked_recv
#define close checked_close
#include "../src/nsi_cap.c"
#include "../src/nsi_socket.c"
#undef calloc
#undef free
#undef socketpair
#undef fcntl
#ifdef __APPLE__
#undef setsockopt
#endif
#undef send
#undef recv
#undef close

static void faults(int first,int second) {
    memset(close_faults,0,sizeof close_faults);close_faults[0]=first;close_faults[1]=second;close_fault_index=0;
}
static NlSocketService *create(void) {NlSocketService *s=NULL;CHECK(nl_socket_service_create(&s).status==NL_SOCKET_OK);CHECK(s);return s;}
static NlSocketPair pair(NlSocketService *s,uint32_t a,uint32_t b) {
    NlSocketPair p;CHECK(nl_socket_acquire_pair(s,a,b,&p).status==NL_SOCKET_OK);return p;
}
static void empty(void) {
    CHECK(!live_allocations);CHECK(host_opens==host_closes);
    for(unsigned i=0;i<128;i++)CHECK(descriptors[i]==-1);
}
static void send_receive(NlSocketService *s,const NlSocketToken *a,const NlSocketToken *b,uint8_t value) {
    NlSocketResult r=nl_socket_send_byte(s,a,value);CHECK(r.status==NL_SOCKET_OK && r.bytes==1 && !r.consumed);
    uint8_t byte=99;r=nl_socket_receive_byte(s,b,&byte);CHECK(r.status==NL_SOCKET_OK && r.bytes==1 && !r.eof && byte==value);
}
static void io_and_identity(void) {
    puts("I check real NUL/bidirectional/EOF/would-block and exact Socket identity.");fflush(stdout);
    FILE *sentinel=tmpfile();CHECK(sentinel);int sentinel_fd=fileno(sentinel);
    NlSocketService *s=create(),*other=create();NlSocketPair p=pair(s,SOCKET_RIGHTS,SOCKET_RIGHTS);
    for(unsigned i=0;i<2;i++){int fd=s->sockets[p.endpoints[i].cap.slot].fd;CHECK(fcntl(fd,F_GETFL)&O_NONBLOCK);CHECK(fcntl(fd,F_GETFD)&FD_CLOEXEC);}
    uint8_t byte=88;NlSocketResult r=nl_socket_receive_byte(s,&p.endpoints[0],&byte);CHECK(r.status==NL_SOCKET_WOULD_BLOCK && !r.eof && r.bytes==0 && byte==88);
    send_receive(s,&p.endpoints[0],&p.endpoints[1],0);send_receive(s,&p.endpoints[1],&p.endpoints[0],255);
    unsigned before=io_calls;CHECK(nl_socket_receive_byte(other,&p.endpoints[0],&byte).status==NL_SOCKET_TOKEN && io_calls==before && byte==88);
    NlSocketToken invalid=p.endpoints[0];invalid.cap.generation++;CHECK(nl_socket_send_byte(s,&invalid,1).status==NL_SOCKET_TOKEN && io_calls==before);
    NlCapSlot *slot=&s->caps->slots[p.endpoints[0].cap.slot];char saved=slot->type_id[0];slot->type_id[0]='X';CHECK(nl_socket_consume_close(s,&p.endpoints[0]).status==NL_SOCKET_TOKEN);slot->type_id[0]=saved;
    saved=slot->service_id[0];slot->service_id[0]='X';CHECK(nl_socket_consume_close(s,&p.endpoints[0]).status==NL_SOCKET_TOKEN);slot->service_id[0]=saved;
    NlSocketToken stale=p.endpoints[0];r=nl_socket_transfer(s,&p.endpoints[0],&p.endpoints[0]);CHECK(r.status==NL_SOCKET_OK && r.consumed);CHECK(nl_socket_send_byte(s,&stale,1).status==NL_SOCKET_TOKEN);
    send_receive(s,&p.endpoints[0],&p.endpoints[1],17);
    /* Both token edges overlap valid object bytes; no invalid pointer is used. */
    NlSocketToken snapshot=p.endpoints[0];before=io_calls;
    CHECK(nl_socket_receive_byte(s,&p.endpoints[0],(uint8_t *)&p.endpoints[0]).status==NL_SOCKET_ARGUMENT);
    CHECK(nl_socket_receive_byte(s,&p.endpoints[0],((uint8_t *)&p.endpoints[0])+sizeof(NlSocketToken)-1).status==NL_SOCKET_ARGUMENT);
    CHECK(io_calls==before && !memcmp(&snapshot,&p.endpoints[0],sizeof snapshot));
    int fd=s->sockets[p.endpoints[0].cap.slot].fd;r=nl_socket_consume_close(s,&p.endpoints[0]);CHECK(r.status==NL_SOCKET_OK && r.consumed && r.close_attempts==1 && r.closed_count==1 && !r.closure_unknown);
    errno=0;CHECK(fcntl(fd,F_GETFD)==-1 && errno==EBADF);before=close_attempts;CHECK(nl_socket_consume_close(s,&p.endpoints[0]).status==NL_SOCKET_TOKEN && close_attempts==before);
    byte=88;r=nl_socket_receive_byte(s,&p.endpoints[1],&byte);CHECK(r.status==NL_SOCKET_EOF && r.eof && !r.bytes && byte==0);
    struct sigaction old_action,current;CHECK(sigaction(SIGPIPE,NULL,&old_action)==0);
    r=nl_socket_send_byte(s,&p.endpoints[1],1);CHECK(r.status==NL_SOCKET_IO && !r.bytes && !r.consumed);
    CHECK(sigaction(SIGPIPE,NULL,&current)==0 && current.sa_handler==old_action.sa_handler);
    CHECK(fcntl(sentinel_fd,F_GETFD)>=0);CHECK(fputs("sentinel",sentinel)>=0 && fflush(sentinel)==0);
    CHECK(nl_socket_service_destroy(other).status==NL_SOCKET_OK);CHECK(nl_socket_service_dispose(s).status==NL_SOCKET_OK);
    CHECK(nl_socket_send_byte(s,&p.endpoints[1],1).status==NL_SOCKET_DISPOSED);CHECK(nl_socket_service_dispose(s).status==NL_SOCKET_OK);CHECK(nl_socket_service_destroy(s).status==NL_SOCKET_OK);
    CHECK(fclose(sentinel)==0);empty();
}
static void rights_and_io_errors(void) {
    puts("I check rights, interrupted calls and byte output retention.");fflush(stdout);
    NlSocketService *s=create();NlSocketPair p=pair(s,NL_CAP_READ,NL_CAP_WRITE);uint8_t byte=91;unsigned before=io_calls;
    CHECK(nl_socket_send_byte(s,&p.endpoints[0],1).status==NL_SOCKET_RIGHTS);CHECK(nl_socket_receive_byte(s,&p.endpoints[1],&byte).status==NL_SOCKET_RIGHTS);CHECK(io_calls==before && byte==91);
    NlSocketToken out=p.endpoints[1],saved=out;CHECK(nl_socket_transfer(s,&p.endpoints[0],&out).status==NL_SOCKET_RIGHTS);CHECK(!memcmp(&out,&saved,sizeof out));
    send_receive(s,&p.endpoints[1],&p.endpoints[0],10);
    for(unsigned i=0;i<3;i++){int errors[]={EINTR,EAGAIN,EIO};NlSocketStatus expected[]={NL_SOCKET_INTERRUPTED,NL_SOCKET_WOULD_BLOCK,NL_SOCKET_IO};fail_send=errors[i];NlSocketResult r=nl_socket_send_byte(s,&p.endpoints[1],3);CHECK(r.status==expected[i] && r.host_errno==errors[i] && !r.bytes && !r.consumed);fail_send=0;fail_recv=errors[i];r=nl_socket_receive_byte(s,&p.endpoints[0],&byte);CHECK(r.status==expected[i] && r.host_errno==errors[i] && byte==91 && !r.eof);fail_recv=0;}
    CHECK(nl_socket_consume_close(s,&p.endpoints[0]).status==NL_SOCKET_OK);CHECK(nl_socket_service_destroy(s).status==NL_SOCKET_OK);empty();
}
static void capacity_and_reuse(void) {
    puts("I check pair atomicity, aliased transfer capacity, generations and reuse.");fflush(stdout);
    NlSocketService *s=create();NlSocketPair all[NL_CAP_PRIVATE_SLOTS/2];
    for(unsigned i=0;i<NL_CAP_PRIVATE_SLOTS/2;i++)all[i]=pair(s,SOCKET_RIGHTS,SOCKET_RIGHTS);
    NlSocketPair out=all[1],saved=out;unsigned opens=host_opens;
    CHECK(nl_socket_acquire_pair(s,SOCKET_RIGHTS,SOCKET_RIGHTS,&out).status==NL_SOCKET_CAPACITY && host_opens==opens);CHECK(!memcmp(&out,&saved,sizeof out));
    NlSocketToken original=all[0].endpoints[0];CHECK(nl_socket_transfer(s,&all[0].endpoints[0],&all[0].endpoints[0]).status==NL_SOCKET_CAPACITY);CHECK(!memcmp(&original,&all[0].endpoints[0],sizeof original));
    NlSocketToken other=all[1].endpoints[0],keep=other;CHECK(nl_socket_transfer(s,&all[0].endpoints[0],&other).status==NL_SOCKET_CAPACITY);CHECK(!memcmp(&other,&keep,sizeof other));send_receive(s,&all[0].endpoints[0],&all[0].endpoints[1],8);
    CHECK(nl_socket_consume_close(s,&all[1].endpoints[0]).status==NL_SOCKET_OK);CHECK(nl_socket_transfer(s,&all[0].endpoints[0],&all[0].endpoints[0]).status==NL_SOCKET_OK);CHECK(nl_socket_consume_close(s,&original).status==NL_SOCKET_TOKEN);
    CHECK(nl_socket_service_destroy(s).status==NL_SOCKET_OK);empty();
    s=create();NlSocketToken previous={0};
    for(unsigned i=0;i<96;i++){NlSocketPair p=pair(s,SOCKET_RIGHTS,SOCKET_RIGHTS);if(i){CHECK(p.endpoints[0].cap.generation>previous.cap.generation);CHECK(nl_socket_consume_close(s,&previous).status==NL_SOCKET_TOKEN);}previous=p.endpoints[0];for(unsigned j=0;j<2;j++)CHECK(nl_socket_consume_close(s,&p.endpoints[j]).status==NL_SOCKET_OK);}
    NlSocketPair live=pair(s,SOCKET_RIGHTS,SOCKET_RIGHTS);s->caps->next_generation=UINT32_MAX-1;out=live;saved=out;opens=host_opens;
    CHECK(nl_socket_acquire_pair(s,SOCKET_RIGHTS,SOCKET_RIGHTS,&out).status==NL_SOCKET_LIMIT && host_opens==opens);CHECK(!memcmp(&out,&saved,sizeof out));unsigned used=0;for(unsigned i=0;i<NL_CAP_PRIVATE_SLOTS;i++)used+=s->caps->slots[i].used!=0;CHECK(used==2);
    original=live.endpoints[0];CHECK(nl_socket_transfer(s,&live.endpoints[0],&live.endpoints[0]).status==NL_SOCKET_LIMIT);CHECK(!memcmp(&original,&live.endpoints[0],sizeof original));send_receive(s,&live.endpoints[0],&live.endpoints[1],9);
    CHECK(nl_socket_service_destroy(s).status==NL_SOCKET_OK);empty();
    uint64_t counter=socket_context_counter;socket_context_counter=UINT64_MAX;NlSocketService *sentinel=(NlSocketService *)(uintptr_t)1;CHECK(nl_socket_service_create(&sentinel).status==NL_SOCKET_LIMIT && sentinel==(NlSocketService *)(uintptr_t)1);socket_context_counter=counter;
}
static void setup_and_allocation_faults(void) {
    puts("I check every platform setup step and complete pair rollback.");fflush(stdout);
    for(long budget=0;budget<3;budget++){allocation_budget=budget;NlSocketService *s=(NlSocketService *)(uintptr_t)1;NlSocketResult r=nl_socket_service_create(&s);allocation_budget=-1;if(budget<2){CHECK(r.status==NL_SOCKET_MEMORY && s==(NlSocketService *)(uintptr_t)1);}else{CHECK(r.status==NL_SOCKET_OK);CHECK(nl_socket_service_destroy(s).status==NL_SOCKET_OK);}empty();}
    unsigned steps=8;
#ifdef __APPLE__
    steps=10;
#endif
    for(unsigned step=0;step<=steps;step++){
        printf("I inject acquisition step %u/%u.\n",step,steps);fflush(stdout);
        NlSocketService *s=create();NlSocketPair live=pair(s,SOCKET_RIGHTS,SOCKET_RIGHTS),out=live,saved=out;
        config_calls=0;fail_config=(int)step;fail_pair=step==0;unsigned opens=host_opens,closes=host_closes;
        NlSocketResult r=nl_socket_acquire_pair(s,SOCKET_RIGHTS,SOCKET_RIGHTS,&out);fail_config=fail_pair=0;
        CHECK(r.status==NL_SOCKET_IO && !r.consumed && !memcmp(&out,&saved,sizeof out));
        CHECK(r.close_attempts==(step?2u:0u) && r.closed_count==r.close_attempts && !r.closure_unknown);
        CHECK(host_opens==opens+(step?2u:0u) && host_closes==closes+(step?2u:0u));
        if(step)for(unsigned n=0;n<2;n++){errno=0;CHECK(fcntl(last_pair[n],F_GETFD)==-1 && errno==EBADF);}
        send_receive(s,&live.endpoints[0],&live.endpoints[1],42);NlSocketPair fresh=pair(s,SOCKET_RIGHTS,SOCKET_RIGHTS);send_receive(s,&fresh.endpoints[1],&fresh.endpoints[0],43);
        CHECK(nl_socket_service_destroy(s).status==NL_SOCKET_OK);empty();
    }
}
static void close_outcomes(void) {
    puts("I distinguish real close, injected reports and deliberately unclosed fault descriptors.");fflush(stdout);
    NlSocketService *s=create();NlSocketPair p=pair(s,SOCKET_RIGHTS,SOCKET_RIGHTS);int fd=s->sockets[p.endpoints[0].cap.slot].fd;
    faults(EINTR,0);NlSocketResult r=nl_socket_consume_close(s,&p.endpoints[0]);CHECK(r.status==NL_SOCKET_IO && r.host_errno==EINTR && r.consumed && r.close_attempts==1 && !r.closed_count && r.closure_unknown);errno=0;CHECK(fcntl(fd,F_GETFD)==-1 && errno==EBADF);
    faults(ENOSPC,0);r=nl_socket_service_dispose(s);CHECK(r.status==NL_SOCKET_IO && r.host_errno==ENOSPC && r.close_attempts==1 && !r.closed_count && r.closure_unknown);
    faults(0,0);r=nl_socket_service_destroy(s);CHECK(r.status==NL_SOCKET_IO && r.host_errno==EINTR && !r.close_attempts && r.closure_unknown);empty();
    s=create();p=pair(s,SOCKET_RIGHTS,SOCKET_RIGHTS);fd=s->sockets[p.endpoints[0].cap.slot].fd;faults(-EINTR,0);unsigned closes=host_closes;
    r=nl_socket_consume_close(s,&p.endpoints[0]);CHECK(r.status==NL_SOCKET_IO && r.closure_unknown && host_closes==closes && fcntl(fd,F_GETFD)>=0);CHECK(nl_socket_consume_close(s,&p.endpoints[0]).status==NL_SOCKET_TOKEN);
    faults(0,0);r=nl_socket_service_destroy(s);CHECK(r.status==NL_SOCKET_IO && r.host_errno==EINTR && r.closure_unknown && r.closed_count==1);CHECK(fcntl(fd,F_GETFD)>=0);real_close(fd,true);empty();
    s=create();p=pair(s,SOCKET_RIGHTS,SOCKET_RIGHTS);faults(EIO,ENOSPC);r=nl_socket_service_destroy(s);CHECK(r.status==NL_SOCKET_IO && r.host_errno==EIO && r.cleanup_failed && r.cleanup_errno==ENOSPC && r.close_attempts==2 && r.closure_unknown);faults(0,0);empty();
    s=create();NlSocketPair saved=p;config_calls=0;fail_config=1;faults(EIO,ENOSPC);r=nl_socket_acquire_pair(s,SOCKET_RIGHTS,SOCKET_RIGHTS,&p);fail_config=0;CHECK(r.status==NL_SOCKET_IO && r.host_errno==EACCES && r.cleanup_failed && r.cleanup_errno==EIO && r.close_attempts==2 && r.closure_unknown && !memcmp(&p,&saved,sizeof p));faults(0,0);CHECK(nl_socket_service_destroy(s).status==NL_SOCKET_IO);empty();
}
int main(void) {
    CHECK(signal(SIGPIPE,SIG_DFL)!=SIG_ERR); /* I qualify the real default-disposition path in this test process. */
    for(unsigned i=0;i<128;i++)descriptors[i]=-1;
    io_and_identity();rights_and_io_errors();capacity_and_reuse();setup_and_allocation_faults();close_outcomes();
    printf("PASS %u Socket checks; real opens=%u closes=%u; adapter close attempts=%u; harness-only recovery closes=%u\n",checks,host_opens,host_closes,close_attempts,manual_closes);return 0;
}
