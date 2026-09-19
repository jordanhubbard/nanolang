#include "../src/nsi_socket.h"
#include <stdio.h>
#include <stdlib.h>
#define CHECK(x) do { if (!(x)) {fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);exit(1);} } while(0)
int main(void) {
    NlSocketService *s=NULL;CHECK(nl_socket_service_create(&s).status==NL_SOCKET_OK);
    NlSocketPair pair;CHECK(nl_socket_acquire_pair(s,NL_CAP_READ|NL_CAP_WRITE|NL_CAP_TRANSFER,NL_CAP_READ|NL_CAP_WRITE,&pair).status==NL_SOCKET_OK);
    uint8_t byte=17;CHECK(nl_socket_receive_byte(s,&pair.endpoints[0],&byte).status==NL_SOCKET_WOULD_BLOCK && byte==17);
    CHECK(nl_socket_send_byte(s,&pair.endpoints[0],0).bytes==1);CHECK(nl_socket_receive_byte(s,&pair.endpoints[1],&byte).status==NL_SOCKET_OK && byte==0);
    CHECK(nl_socket_send_byte(s,&pair.endpoints[1],255).bytes==1);CHECK(nl_socket_receive_byte(s,&pair.endpoints[0],&byte).status==NL_SOCKET_OK && byte==255);
    NlSocketToken prior=pair.endpoints[0];CHECK(nl_socket_transfer(s,&pair.endpoints[0],&pair.endpoints[0]).status==NL_SOCKET_OK);CHECK(nl_socket_consume_close(s,&prior).status==NL_SOCKET_TOKEN);
    NlSocketResult r=nl_socket_consume_close(s,&pair.endpoints[0]);CHECK(r.status==NL_SOCKET_OK && r.closed_count==1 && r.consumed);
    byte=17;r=nl_socket_receive_byte(s,&pair.endpoints[1],&byte);CHECK(r.status==NL_SOCKET_EOF && r.eof && byte==0);
    CHECK(nl_socket_send_byte(s,&pair.endpoints[1],1).status==NL_SOCKET_IO);r=nl_socket_service_destroy(s);CHECK(r.status==NL_SOCKET_OK && r.closed_count==1);
    puts("PASS ordinary linked Socket NUL/byte/EOF/transfer/close");return 0;
}
