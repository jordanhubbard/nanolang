/* I inject the actual getentropy boundary on Darwin and Linux. */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__APPLE__)
#include <sys/random.h>
#else
#include <unistd.h>
#endif
static unsigned calls,fail_call,file_closes;
static int failure,zero;
static uint64_t fixed;
static int checked_getentropy(void *out,size_t size) {
    calls++;
    if(failure || calls==fail_call){memset(out,0x5a,size);errno=EIO;return -1;}
    if(zero){memset(out,0,size);return 0;}
    if(fixed){if(size!=sizeof(fixed))abort();memcpy(out,&fixed,size);return 0;}
    return getentropy(out,size);
}
#define getentropy checked_getentropy
#include "../src/nsi_cap.c"
#undef getentropy
static int checked_fclose(FILE *stream) {file_closes++;return fclose(stream);}
#define fclose checked_fclose
#include "../src/nsi_file.c"
#undef fclose
#include "../src/nsi_socket.c"
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);exit(1);}}while(0)
static void unchanged(NlCapTable *t,const NlCapSlot *slots,uint32_t generation) {
    CHECK(!memcmp(t->slots,slots,sizeof(t->slots)) && t->next_generation==generation);
}
static void adapter_failures(void) {
    NlFileService *file=NULL;CHECK(nl_file_service_create(&file).status==NL_FILE_OK);
    NlFileToken token,prior;memset(&token,0xa5,sizeof(token));prior=token;
    failure=1;unsigned closes=file_closes;
    NlFileResult fr=nl_file_acquire_temp(file,NL_CAP_READ|NL_CAP_TRANSFER,&token);
    CHECK(fr.status==NL_FILE_IO && !fr.cleanup_failed && file_closes==closes+1);
    CHECK(!memcmp(&token,&prior,sizeof(token)));
    failure=0;
    CHECK(nl_file_acquire_temp(file,NL_CAP_READ|NL_CAP_TRANSFER,&token).status==NL_FILE_OK);
    prior=token;failure=1;
    CHECK(nl_file_transfer(file,&token,&token).status==NL_FILE_IO);
    CHECK(!memcmp(&token,&prior,sizeof(token)));
    failure=0;CHECK(nl_file_service_destroy(file).status==NL_FILE_OK);
    NlSocketService *socket=NULL;CHECK(nl_socket_service_create(&socket).status==NL_SOCKET_OK);
    NlSocketPair pair,before;memset(&pair,0xa5,sizeof(pair));before=pair;
    fail_call=calls+2;
    NlSocketResult sr=nl_socket_acquire_pair(socket,NL_CAP_READ|NL_CAP_TRANSFER,NL_CAP_WRITE,&pair);
    CHECK(sr.status==NL_SOCKET_IO && !sr.close_attempts && !sr.cleanup_failed);
    CHECK(!memcmp(&pair,&before,sizeof(pair)));
    for(unsigned i=0;i<NL_CAP_SLOTS;i++)CHECK(!socket->caps->slots[i].used);
    fail_call=0;
    CHECK(nl_socket_acquire_pair(socket,NL_CAP_READ|NL_CAP_TRANSFER,NL_CAP_WRITE,&pair).status==NL_SOCKET_OK);
    NlSocketToken old=pair.endpoints[0];failure=1;
    CHECK(nl_socket_transfer(socket,&pair.endpoints[0],&pair.endpoints[0]).status==NL_SOCKET_IO);
    CHECK(!memcmp(&pair.endpoints[0],&old,sizeof(old)));
    failure=0;sr=nl_socket_service_destroy(socket);
    CHECK(sr.status==NL_SOCKET_OK && sr.closed_count==2);
    puts("PASS File rollback and Socket partial-mint/transfer owner preservation");
}
int main(void) {
    NlCapTable *t=nl_cap_table_create();CHECK(t);
    NlCap destination,before,owner;memset(&destination,0xa5,sizeof(destination));before=destination;
    NlCapSlot slots[NL_CAP_SLOTS];memcpy(slots,t->slots,sizeof(slots));
    failure=1;unsigned prior=calls;
    CHECK(nl_cap_mint(t,"owner","service",NL_CAP_READ,0,NULL,&destination)==NL_CAP_ERR_ENTROPY);
    CHECK(calls==prior+1 && !memcmp(&destination,&before,sizeof(before)));unchanged(t,slots,0);
    failure=0;zero=1;prior=calls;
    CHECK(nl_cap_private_mint(t,"owner","service",NL_CAP_READ,&destination)==NL_CAP_ERR_ENTROPY);
    CHECK(calls==prior+4 && !memcmp(&destination,&before,sizeof(before)));unchanged(t,slots,0);
    zero=0;
    CHECK(nl_cap_private_mint(t,"owner","service",NL_CAP_READ|NL_CAP_TRANSFER|NL_CAP_DELEGATE,&owner)==NL_CAP_OK);
    memcpy(slots,t->slots,sizeof(slots));uint32_t generation=t->next_generation;
    failure=1;
    CHECK(nl_cap_transfer(t,&owner,&destination)==NL_CAP_ERR_ENTROPY);
    CHECK(!memcmp(&destination,&before,sizeof(before)));unchanged(t,slots,generation);
    NlCap old=owner;
    CHECK(nl_cap_private_transfer(t,&owner,&owner)==NL_CAP_ERR_ENTROPY);
    CHECK(!memcmp(&owner,&old,sizeof(owner)));unchanged(t,slots,generation);
    CHECK(nl_cap_attenuate(t,&owner,NL_CAP_READ,&destination)==NL_CAP_ERR_ENTROPY);
    CHECK(!memcmp(&destination,&before,sizeof(before)));unchanged(t,slots,generation);
    CHECK(nl_cap_check(t,&owner,NL_CAP_READ)==NL_CAP_OK);
    uint64_t cell=UINT64_MAX;
    CHECK(nl_cap_forth_bind(t,&owner,&cell)==NL_CAP_ERR_ENTROPY && cell==UINT64_MAX);
    CHECK(!t->forth_used[0]);
    failure=0;zero=1;prior=calls;
    CHECK(nl_cap_forth_bind(t,&owner,&cell)==NL_CAP_ERR_ENTROPY && cell==UINT64_MAX);
    CHECK(calls==prior+4 && !t->forth_used[0]);
    zero=0;fixed=123;prior=calls;
    CHECK(nl_cap_forth_bind(t,&owner,&cell)==NL_CAP_ERR_ENTROPY && cell==UINT64_MAX);
    CHECK(calls==prior+4 && !t->forth_used[0]);
    fixed=UINT64_C(0x123456789abcdef0);
    CHECK(nl_cap_forth_bind(t,&owner,&cell)==NL_CAP_OK && cell==fixed);
    uint64_t another=UINT64_MAX;prior=calls;
    CHECK(nl_cap_forth_bind(t,&owner,&another)==NL_CAP_ERR_ENTROPY && another==UINT64_MAX);
    CHECK(calls==prior+4 && !t->forth_used[1]);
    CHECK(nl_cap_forth_lookup(t,cell,&destination)==NL_CAP_OK && !memcmp(&destination,&owner,sizeof(owner)));
    fixed=0;
    CHECK(nl_cap_private_transfer(t,&owner,&destination)==NL_CAP_OK);
    CHECK(nl_cap_check(t,&owner,NL_CAP_READ)!=NL_CAP_OK);
    CHECK(nl_cap_check(t,&destination,NL_CAP_READ)==NL_CAP_OK);
    CHECK(nl_cap_forth_lookup(t,cell,&owner)!=NL_CAP_OK);
    CHECK(nl_cap_private_consume(t,&destination)==NL_CAP_OK);
    nl_cap_table_destroy(t);
    adapter_failures();
    puts("PASS entropy failure/zero/low/collision refusal, preserved owners and recovery");
    return 0;
}
