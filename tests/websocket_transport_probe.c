#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "../src/nsi_websocket_transport.h"
#ifdef NL_WS_INJECT
static int refuse_copy;
static void *controlled_malloc(size_t size) { return refuse_copy ? NULL : malloc(size); }
#define malloc controlled_malloc
#include "../src/nsi_websocket_transport.c"
#undef malloc
#endif

static void unchanged(NlWsMessage m) {
    assert(m.binary && m.bytes == NULL && m.length == 123);
}
int main(int argc,char **argv) {
    assert(argc==3);
    NlWsTransportPolicy policy={true,false,getenv("NANOLANG_RESOLVER"),60000};
    NlWsTransport *c=NULL;
    const char *host="ws://localhost:80/";
    NlWsTransportResult r=nl_ws_transport_connect(host,strlen(host),&policy,1000,&c);
    assert(r.status==NL_WS_TRANSPORT_RIGHTS && c==NULL);
    policy.allow_network=false;
    r=nl_ws_transport_connect(argv[1],strlen(argv[1]),&policy,1000,&c);
    assert(r.status==NL_WS_TRANSPORT_RIGHTS && c==NULL);
    policy.allow_network=true;policy.allow_lookup=true;
    r=nl_ws_transport_connect(argv[1],strlen(argv[1]),&policy,0,&c);
    assert(r.status==NL_WS_TRANSPORT_TIMEOUT && c==NULL);
    r=nl_ws_transport_connect(argv[1],strlen(argv[1]),&policy,60001,&c);
    assert(r.status==NL_WS_TRANSPORT_LIMIT && c==NULL);
    r=nl_ws_transport_connect("ws://a\0b",8,&policy,1000,&c);
    assert(r.status==NL_WS_TRANSPORT_ARGUMENT && c==NULL);
    r=nl_ws_transport_connect(argv[1],strlen(argv[1]),&policy,2000,&c);
    assert(r.status==NL_WS_TRANSPORT_OK && c && !r.terminal);
    NlWsMessage message={true,NULL,123};
    if(!strcmp(argv[2],"counted")) {
        const unsigned char binary[]={0,255,1};
        r=nl_ws_transport_send(c,false,"\xff",1,1000);
        assert(r.status==NL_WS_TRANSPORT_ARGUMENT && !r.terminal);
        r=nl_ws_transport_send(c,false,"a\0b",3,1000);
        assert(r.status==NL_WS_TRANSPORT_OK && r.bytes==3);
        r=nl_ws_transport_send(c,true,binary,3,1000);
        assert(r.status==NL_WS_TRANSPORT_OK && r.bytes==3);
        r=nl_ws_transport_send(c,false,NULL,0,1000);
        assert(r.status==NL_WS_TRANSPORT_OK && r.bytes==0);
#ifdef NL_WS_INJECT
        refuse_copy=1;
        r=nl_ws_transport_receive(c,1000,&message);
        assert(r.status==NL_WS_TRANSPORT_MEMORY && !r.terminal);unchanged(message);
        refuse_copy=0;
#endif
        r=nl_ws_transport_receive(c,1000,&message);
        assert(r.status==NL_WS_TRANSPORT_OK && !message.binary && message.length==3);
        assert(!memcmp(message.bytes,"a\0b",3));
        NlWsMessage second={0},empty={0};
        r=nl_ws_transport_receive(c,1000,&second);
        assert(r.status==NL_WS_TRANSPORT_OK && second.binary && second.length==3);
        assert(!memcmp(second.bytes,binary,3));
        r=nl_ws_transport_receive(c,1000,&empty);
        assert(r.status==NL_WS_TRANSPORT_OK && !empty.binary && !empty.length && empty.bytes);
        NlWsMessage sentinel={true,NULL,123};
        r=nl_ws_transport_receive(c,1000,&sentinel);
        assert(r.status==NL_WS_TRANSPORT_CLOSED && r.terminal && r.close_code==1000);unchanged(sentinel);
        r=nl_ws_transport_close(c,1000);assert(r.status==NL_WS_TRANSPORT_OK && r.terminal);
        assert(!memcmp(message.bytes,"a\0b",3) && !memcmp(second.bytes,binary,3));
        nl_ws_message_free(&message);nl_ws_message_free(&second);nl_ws_message_free(&empty);
    } else if(!strcmp(argv[2],"partial")) {
        r=nl_ws_transport_receive(c,30,&message);
        assert(r.status==NL_WS_TRANSPORT_TIMEOUT && !r.terminal);unchanged(message);
        r=nl_ws_transport_receive(c,1000,&message);
        assert(r.status==NL_WS_TRANSPORT_OK && message.length==5 && !memcmp(message.bytes,"hello",5));
        nl_ws_message_free(&message);
        r=nl_ws_transport_close(c,1000);assert(r.status==NL_WS_TRANSPORT_OK);
    } else if(!strcmp(argv[2],"invalid")) {
        r=nl_ws_transport_receive(c,1000,&message);
        assert(r.status==NL_WS_TRANSPORT_PROTOCOL && r.terminal);unchanged(message);
        r=nl_ws_transport_close(c,1000);assert(r.status==NL_WS_TRANSPORT_OK);
    } else if(!strcmp(argv[2],"close-invalid")) {
        r=nl_ws_transport_close(c,60001);assert(r.status==NL_WS_TRANSPORT_LIMIT && r.terminal);
    } else if(!strcmp(argv[2],"close-timeout")) {
        r=nl_ws_transport_close(c,30);assert(r.status==NL_WS_TRANSPORT_TIMEOUT && r.terminal);
    } else if(!strcmp(argv[2],"send-partial")) {
        unsigned char *large=malloc(1024*1024);assert(large);memset(large,'x',1024*1024);
        r=nl_ws_transport_send(c,true,large,1024*1024,0);free(large);
        assert(r.status==NL_WS_TRANSPORT_TIMEOUT && r.terminal && !nl_ws_transport_connected(c));
        r=nl_ws_transport_close(c,1000);assert(r.status==NL_WS_TRANSPORT_OK);
    } else assert(0);
    return 0;
}
