#include "../src/nsi_services_values.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

static unsigned checks;
#define CHECK(x) do { checks++;if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);exit(1);} } while(0)
#define OK(x) CHECK((x)==NL_SERVICES_VALUE_OK)
static long budget=-1,allocations;
#ifdef SERVICES_ALLOC_TEST
static bool fail_cleanup;
static unsigned file_closes,tcp_closes;
int services_test_close(int fd) {
    int result=close(fd);tcp_closes++;
    if(fail_cleanup){errno=EIO;return -1;}return result;
}
int services_test_fclose(FILE *stream) {
    int result=fclose(stream);file_closes++;
    if(fail_cleanup){errno=EIO;return EOF;}return result;
}
#endif
void *services_test_calloc(size_t n,size_t width) {
    if(!budget)return NULL;
    if(budget>0)budget--;
    void *p=calloc(n,width);if(p)allocations++;return p;
}
void services_test_free(void *p) { if(p)allocations--;free(p); }
static const NlServicesCatalog catalogs[]={NL_SERVICES_FILE,NL_SERVICES_TCP,NL_SERVICES_FILE};
static NlServicesValues *create(void) {
    NlServicesValues *c=NULL;OK(nl_services_values_create(catalogs,3,&c));return c;
}
static void clean(NlServicesValues **c,NlServicesValueStatus status) {
    NlServicesFinish report;OK(nl_services_values_destroy(c,status,&report));
    CHECK(!*c && report.execution==status && report.count==3 && !report.cleanup_failures);
    for(unsigned i=0;i<3;i++)CHECK(report.instances[i].catalog==catalogs[i]);
}
static NlServicesValue acquire(NlServicesValues *c,uint32_t i,const NlSocketEndpoint *endpoint) {
    NlServicesValue result={0},owner={0};NlServicesOpenView view;
    OK(nl_services_values_acquire(c,i,endpoint,&result));
    OK(nl_services_value_validate(c,&result,true));
    CHECK(nl_services_value_validate(c,&result,false)==NL_SERVICES_VALUE_TYPE);
    OK(nl_services_value_view(c,&result,&view));CHECK(view.ok && view.catalog==catalogs[i]);
    NlServicesBorrow premature={0};
    CHECK(nl_services_value_borrow(c,&result,&premature)==NL_SERVICES_VALUE_TYPE && !premature.instance);
    NlServicesOpenView sentinel={.catalog=99},saved=sentinel;
    CHECK(nl_services_value_take_error(c,&result,&sentinel)==NL_SERVICES_VALUE_TYPE);
    CHECK(!memcmp(&sentinel,&saved,sizeof saved));
    NlServicesValue old=result;OK(nl_services_value_take_ok(c,&result,&owner));CHECK(!result.instance);
    CHECK(nl_services_value_drop(c,&old)==NL_SERVICES_VALUE_STALE);
    return owner;
}
static int listener(bool ipv6,NlSocketEndpoint *endpoint) {
    int fd=socket(ipv6?AF_INET6:AF_INET,SOCK_STREAM,IPPROTO_TCP);CHECK(fd>=0);
    CHECK(fcntl(fd,F_SETFL,O_NONBLOCK)==0);
    *endpoint=(NlSocketEndpoint){.family=ipv6?6:4};
    if(ipv6) {
        struct sockaddr_in6 a={.sin6_family=AF_INET6,.sin6_addr=IN6ADDR_LOOPBACK_INIT};
        CHECK(bind(fd,(struct sockaddr *)&a,sizeof a)==0);socklen_t n=sizeof a;
        CHECK(getsockname(fd,(struct sockaddr *)&a,&n)==0);endpoint->address3=1;endpoint->port=ntohs(a.sin6_port);
    } else {
        struct sockaddr_in a={.sin_family=AF_INET,.sin_addr={.s_addr=htonl(INADDR_LOOPBACK)}};
        CHECK(bind(fd,(struct sockaddr *)&a,sizeof a)==0);socklen_t n=sizeof a;
        CHECK(getsockname(fd,(struct sockaddr *)&a,&n)==0);endpoint->address0=0x7f000001;endpoint->port=ntohs(a.sin_port);
    }
    CHECK(listen(fd,4)==0);return fd;
}
static NlServicesScalarResult tcp_wait(NlServicesValues *c,NlServicesBorrow *b,unsigned method) {
    NlServicesScalarResult result;
    for(unsigned i=0;i<2000;i++) {
        OK(nl_services_value_call(c,1,method,b,0,&result));CHECK(result.catalog==NL_SERVICES_TCP);
        if(result.result.tcp.detail.status!=NL_SOCKET_WOULD_BLOCK && result.result.tcp.detail.status!=NL_SOCKET_INTERRUPTED)return result;
        usleep(1000);
    }
    CHECK(!"I exceeded my bounded loopback wait");return (NlServicesScalarResult){0};
}
static void lifecycle(bool ipv6,bool abandon) {
    NlSocketEndpoint endpoint;int server=listener(ipv6,&endpoint);
    NlServicesValues *c=create(),*other=create();
    NlServicesValue values[3]={acquire(c,0,NULL),acquire(c,1,&endpoint),acquire(c,2,NULL)};
    NlServicesBorrow borrows[3]={{0}};
    for(unsigned i=0;i<3;i++) {
        NlServicesValue old=values[i],moved={0};
        OK(nl_services_value_move(c,&values[i],&moved));CHECK(!values[i].instance);
        CHECK(nl_services_value_drop(c,&old)==NL_SERVICES_VALUE_STALE);
        CHECK(nl_services_value_drop(other,&moved)==NL_SERVICES_VALUE_STALE);
        values[i]=moved;OK(nl_services_value_validate(c,&values[i],false));
        CHECK(nl_services_value_validate(c,&values[i],true)==NL_SERVICES_VALUE_TYPE);
        OK(nl_services_value_borrow(c,&values[i],&borrows[i]));
        OK(nl_services_borrow_validate(c,&borrows[i]));
        uint64_t owners=0,borrowed=0;
        OK(nl_services_values_live_slots(c,i,&owners,&borrowed));CHECK(owners && owners==borrowed);
        CHECK(nl_services_values_live_slots(c,i,&owners,&owners)==NL_SERVICES_VALUE_ARGUMENT);
        NlServicesValue blocked={0};
        CHECK(nl_services_value_move(c,&values[i],&blocked)==NL_SERVICES_VALUE_BORROWED && !blocked.instance);
        CHECK(nl_services_value_drop(c,&values[i])==NL_SERVICES_VALUE_BORROWED);
        NlServicesScalarResult out={.catalog=99},saved=out;
        for(unsigned j=0;j<3;j++)if(j!=i) {
            CHECK(nl_services_value_call(c,j,1,&borrows[i],42,&out)==NL_SERVICES_VALUE_TYPE);
            CHECK(nl_services_value_close(c,j,&values[i],&out)==NL_SERVICES_VALUE_TYPE);
            CHECK(!memcmp(&out,&saved,sizeof out));
        }
        CHECK(nl_services_value_close(c,i,&values[i],&out)==NL_SERVICES_VALUE_BORROWED);
        CHECK(!memcmp(&out,&saved,sizeof out));
    }
    NlServicesScalarResult r=tcp_wait(c,&borrows[1],2);CHECK(r.result.tcp.ok);
    struct pollfd wait={.fd=server,.events=POLLIN};CHECK(poll(&wait,1,2000)==1);
    int peer=accept(server,NULL,NULL);CHECK(peer>=0);CHECK(fcntl(peer,F_SETFL,O_NONBLOCK)==0);
    for(unsigned i=0;i<3;i++) {
        OK(nl_services_value_call(c,i,1,&borrows[i],80+i,&r));
        CHECK(i==1?r.result.tcp.ok:r.result.file.ok);
    }
    wait=(struct pollfd){.fd=peer,.events=POLLIN};CHECK(poll(&wait,1,2000)==1);
    unsigned char byte=0;CHECK(recv(peer,&byte,1,0)==1 && byte==81);
    byte=231;CHECK(send(peer,&byte,1,0)==1);
    r=tcp_wait(c,&borrows[1],3);CHECK(r.result.tcp.ok && r.result.tcp.value==231);
    for(unsigned i=0;i<3;i+=2) {
        OK(nl_services_value_call(c,i,2,&borrows[i],0,&r));CHECK(r.result.file.ok);
        OK(nl_services_value_call(c,i,3,&borrows[i],0,&r));CHECK(r.result.file.ok && r.result.file.value==80+i);
    }
    /* Relabeling a handle cannot select the other catalog or repeated File core. */
    NlServicesValue forged=values[0];forged.instance=2;
    CHECK(nl_services_value_drop(c,&forged)==NL_SERVICES_VALUE_TYPE);
    forged.instance=3;CHECK(nl_services_value_drop(c,&forged)==NL_SERVICES_VALUE_STALE);
    NlServicesBorrow forged_borrow=borrows[0];forged_borrow.instance=2;
    CHECK(nl_services_value_call(c,1,1,&forged_borrow,4,&r)==NL_SERVICES_VALUE_TYPE);
    forged_borrow.instance=3;CHECK(nl_services_value_call(c,2,1,&forged_borrow,4,&r)==NL_SERVICES_VALUE_STALE);
    if(!abandon)for(unsigned i=0;i<3;i++) {
        NlServicesBorrow old=borrows[i];OK(nl_services_value_end_borrow(c,&borrows[i]));
        CHECK(!borrows[i].instance);
        CHECK(nl_services_borrow_validate(c,&old)==NL_SERVICES_VALUE_STALE);
        CHECK(nl_services_value_call(c,i,1,&old,4,&r)==NL_SERVICES_VALUE_STALE);
        OK(nl_services_value_close(c,i,&values[i],&r));CHECK(!values[i].instance);
        CHECK(i==1?r.result.tcp.ok:r.result.file.ok);
    }
    NlServicesValueStatus execution=abandon?NL_SERVICES_VALUE_STATE:NL_SERVICES_VALUE_OK;
    NlServicesFinish first,second;
    OK(nl_services_values_finish(c,execution,&first));CHECK(!first.cleanup_failures);
    OK(nl_services_values_finish(c,NL_SERVICES_VALUE_MEMORY,&second));CHECK(!memcmp(&first,&second,sizeof first));
    CHECK(nl_services_value_call(c,1,3,&borrows[1],0,&r)==NL_SERVICES_VALUE_DISPOSED);
    wait=(struct pollfd){.fd=peer,.events=POLLIN};CHECK(poll(&wait,1,2000)==1);
    CHECK(recv(peer,&byte,1,0)==0);CHECK(close(peer)==0 && close(server)==0);
    clean(&c,execution);clean(&other,NL_SERVICES_VALUE_OK);
}
static void boundaries(void) {
    size_t bound=123;NlServicesCatalog bad[]={NL_SERVICES_FILE,99};NlServicesValues *c=NULL;
    CHECK(!nl_services_values_storage_bound(bad,2,&bound) && bound==123);
    CHECK(nl_services_values_create(bad,2,&c)==NL_SERVICES_VALUE_ARGUMENT && !c);
    CHECK(nl_services_values_create(catalogs,0,&c)==NL_SERVICES_VALUE_ARGUMENT);
    CHECK(nl_services_values_create(catalogs,65,&c)==NL_SERVICES_VALUE_ARGUMENT);
    CHECK(nl_services_values_storage_bound(catalogs,3,&bound) && bound>123);
    c=create();NlServicesValue v={0};NlSocketEndpoint bad_endpoint={0};
    CHECK(nl_services_values_create(catalogs,3,&c)==NL_SERVICES_VALUE_ARGUMENT && c);
    NlServicesValue full[NL_FILE_VALUE_SLOTS]={{0}};
    for(unsigned i=0;i<NL_FILE_VALUE_SLOTS;i++)OK(nl_services_values_acquire(c,0,NULL,&full[i]));
    CHECK(nl_services_values_acquire(c,0,NULL,&v)==NL_SERVICES_VALUE_LIMIT && !v.instance);
    OK(nl_services_values_acquire(c,2,NULL,&v));OK(nl_services_value_drop(c,&v));
    for(unsigned i=0;i<NL_FILE_VALUE_SLOTS;i++)OK(nl_services_value_drop(c,&full[i]));
    CHECK(nl_services_values_acquire(c,0,&bad_endpoint,&v)==NL_SERVICES_VALUE_ARGUMENT && !v.instance);
    CHECK(nl_services_values_acquire(c,1,NULL,&v)==NL_SERVICES_VALUE_ARGUMENT && !v.instance);
    OK(nl_services_values_acquire(c,1,&bad_endpoint,&v));NlServicesOpenView view;
    OK(nl_services_value_view(c,&v,&view));CHECK(!view.ok && view.error.tcp.status==NL_SOCKET_ARGUMENT);
    OK(nl_services_value_take_error(c,&v,&view));CHECK(!v.instance && !view.ok && view.catalog==NL_SERVICES_TCP);
    NlServicesFinish report;
    CHECK(nl_services_values_destroy(&c,(NlServicesValueStatus)999,&report)==NL_SERVICES_VALUE_ARGUMENT && c);
    clean(&c,NL_SERVICES_VALUE_OK);
    NlServicesCatalog all[64];for(unsigned i=0;i<64;i++)all[i]=i%2?NL_SERVICES_TCP:NL_SERVICES_FILE;
    OK(nl_services_values_create(all,64,&c));
    for(unsigned i=0;i<64;i++) {
        OK(nl_services_values_acquire(c,i,i%2?&bad_endpoint:NULL,&v));v=(NlServicesValue){0};
    }
    OK(nl_services_values_destroy(&c,NL_SERVICES_VALUE_STATE,&report));CHECK(report.count==64 && !report.cleanup_failures);
#ifdef SERVICES_ALLOC_TEST
    for(long fail=0;fail<=193;fail++) {
        CHECK(!allocations);budget=fail;
        NlServicesValueStatus status=nl_services_values_create(all,64,&c);budget=-1;
        if(fail<193)CHECK(status==NL_SERVICES_VALUE_MEMORY && !c);
        else {OK(status);OK(nl_services_values_destroy(&c,NL_SERVICES_VALUE_OK,&report));CHECK(report.count==64 && !report.cleanup_failures);}
        CHECK(!allocations);
    }
#endif
}
#ifdef SERVICES_ALLOC_TEST
static void cleanup_errors(void) {
    NlSocketEndpoint endpoint;int server=listener(false,&endpoint);
    NlServicesValues *c=create();
    NlServicesValue v=acquire(c,0,NULL);(void)v;
    v=acquire(c,1,&endpoint);v=acquire(c,2,NULL);
    unsigned files=file_closes,sockets=tcp_closes;
    fail_cleanup=true;NlServicesFinish first,second;
    OK(nl_services_values_finish(c,NL_SERVICES_VALUE_STATE,&first));
    fail_cleanup=false;
    /* TCP retains both the close error and its terminal ambiguous-close history. */
    CHECK(first.instances[0].finish.file.cleanup_failures==1);
    CHECK(first.instances[1].finish.tcp.cleanup_failures==2);
    CHECK(first.instances[2].finish.file.cleanup_failures==1);
    CHECK(first.cleanup_failures==4 && file_closes==files+2 && tcp_closes==sockets+1);
    CHECK(first.instances[0].finish.file.first_cleanup.host_errno==EIO);
    CHECK(first.instances[1].finish.tcp.first_cleanup.host_errno==EIO);
    CHECK(first.instances[1].finish.tcp.first_cleanup.closure_unknown);
    CHECK(first.instances[2].finish.file.first_cleanup.host_errno==EIO);
    OK(nl_services_values_destroy(&c,NL_SERVICES_VALUE_OK,&second));
    CHECK(!memcmp(&first,&second,sizeof first));
    CHECK(file_closes==files+2 && tcp_closes==sockets+1 && !allocations);
    CHECK(close(server)==0);
}
#endif
int main(void) {
    boundaries();lifecycle(false,false);lifecycle(true,false);lifecycle(false,true);lifecycle(true,true);
    #ifdef SERVICES_ALLOC_TEST
    cleanup_errors();
    #endif
    CHECK(!allocations);printf("PASS %u mixed instance value checks\n",checks);return 0;
}
