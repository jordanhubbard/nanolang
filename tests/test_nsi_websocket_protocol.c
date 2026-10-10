#include "../src/nsi_websocket_protocol.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) {fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);exit(1);} } while(0)
#ifdef NL_WS_PROTOCOL_INSTRUMENT
static int allocation_failure;
static void *ws_test_calloc(size_t n,size_t size) {return allocation_failure ? NULL : calloc(n,size);}
static void *ws_test_realloc(void *p,size_t size) {return allocation_failure ? NULL : realloc(p,size);}
#define calloc ws_test_calloc
#define realloc ws_test_realloc
#include "../src/nsi_websocket_protocol.c"
#undef calloc
#undef realloc
static void allocation_refusals(void) {
    NlWsDecoder *d=NULL;allocation_failure=1;
    CHECK(nl_ws_decoder_create(&d)==NL_WS_MEMORY && !d);
    allocation_failure=0;CHECK(nl_ws_decoder_create(&d)==NL_WS_OK);
    allocation_failure=1;
    const uint8_t frame[]={0x81,2,'a','b'};size_t used=999;
    NlWsEvent event={NL_WS_PONG,NULL,777};
    CHECK(nl_ws_decoder_feed(d,frame,sizeof(frame),&used,&event)==NL_WS_MEMORY && used==2);
    CHECK(event.opcode==NL_WS_PONG && event.length==777);
    allocation_failure=0;
    CHECK(nl_ws_decoder_feed(d,frame,sizeof(frame),&used,&event)==NL_WS_MEMORY && !used);
    nl_ws_decoder_destroy(d);
    puts("PASS allocation refusal and terminal decoder cleanup");
}
#endif
static NlWsDecoder *decoder(void) {NlWsDecoder *d=NULL;CHECK(nl_ws_decoder_create(&d)==NL_WS_OK);return d;}
static void handshake(void) {
    char key[25],accept[29];
    CHECK(nl_ws_handshake_key((const uint8_t *)"the sample nonce",key,accept)==NL_WS_OK);
    CHECK(!strcmp(key,"dGhlIHNhbXBsZSBub25jZQ=="));
    CHECK(!strcmp(accept,"s3pPLMBiTxaQ9kYGzzhZRbK+xOo="));
    const char *good="HTTP/1.1 101 Switching Protocols\r\nUpgrade: WebSocket\r\nConnection: keep-alive, Upgrade\r\nSec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n";
    size_t used=999,n=strlen(good);
    for (size_t i=0;i<n;i++) {CHECK(nl_ws_upgrade_validate(good,i,accept,&used)==NL_WS_MORE);CHECK(used==999);}
    CHECK(nl_ws_upgrade_validate(good,n,accept,&used)==NL_WS_OK && used==n);
    char coalesced[512];memcpy(coalesced,good,n);coalesced[n]=(char)0x81;coalesced[n+1]=0;
    CHECK(nl_ws_upgrade_validate(coalesced,n+2,accept,&used)==NL_WS_OK && used==n);
    const char *bad[]={
        "HTTP/1.1 403 Forbidden\r\nX-Trace: 101\r\n\r\n",
        "HTTP/1.1 101 Switching\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: wrong\r\n\r\n",
        "HTTP/1.1 101 Switching\r\nUpgrade: websocket\r\nConnection: xUpgrade\r\nSec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n",
        "HTTP/1.1 101 Switching\r\nUpgrade: websocket\r\nConnection: Upgrade,\r\nSec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n",
        "HTTP/1.1 101 Switching\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\nSec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n",
        "HTTP/1.1 101 Switching\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\nSec-WebSocket-Extensions: permessage-deflate\r\n\r\n",
        "HTTP/1.1 101 Switching\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\nSec-WebSocket-Protocol: chat\r\n\r\n",
        "HTTP/1.1 101 Switching\r\n Upgrade: websocket\r\nConnection: Upgrade\r\n\r\n",
        "HTTP/1.1 101 Switching\r\nUpgrade : websocket\r\nConnection: Upgrade\r\n\r\n",
        "HTTP/1.1 101 Switching\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n"
    };
    for (size_t i=0;i<sizeof(bad)/sizeof(*bad);i++) {
        used=999;CHECK(nl_ws_upgrade_validate(bad[i],strlen(bad[i]),accept,&used)==NL_WS_PROTOCOL);CHECK(used==999);
    }
    char huge[NL_WS_HEADER_MAX];memset(huge,'a',sizeof(huge));
    CHECK(nl_ws_upgrade_validate(huge,sizeof(huge),accept,&used)==NL_WS_LIMIT);
    uint8_t zeros[16]={0};CHECK(nl_ws_handshake_key(zeros,key,accept)==NL_WS_OK);
    CHECK(!strcmp(key,"AAAAAAAAAAAAAAAAAAAAAA=="));
    puts("PASS nonce/accept known vectors and exact bounded HTTP upgrade");
}
static void messages(void) {
    const uint8_t stream[]={0x01,2,'h','e',0x89,3,'p',0,'g',0x00,1,'l',0x80,2,'l','o',0x82,2,0xff,0,0x88,2,3,232};
    for (size_t split=1;split<=sizeof(stream);split++) {
        NlWsDecoder *d=decoder();size_t at=0;unsigned events=0;
        while (at<sizeof(stream)) {
            size_t n=sizeof(stream)-at;if(n>split)n=split;
            size_t used=999;NlWsEvent e={0};
            NlWsStatus s=nl_ws_decoder_feed(d,stream+at,n,&used,&e);
            CHECK(used && used<=n);at+=used;
            CHECK(s==NL_WS_MORE || s==NL_WS_EVENT);
            if(s==NL_WS_EVENT) {
                if(events==0) CHECK(e.opcode==NL_WS_PING && e.length==3 && !memcmp(e.bytes,"p\0g",3));
                if(events==1) CHECK(e.opcode==NL_WS_TEXT && e.length==5 && !memcmp(e.bytes,"hello",5));
                if(events==2) CHECK(e.opcode==NL_WS_BINARY && e.length==2 && e.bytes[0]==255 && e.bytes[1]==0);
                if(events==3) CHECK(e.opcode==NL_WS_CLOSE && e.length==2 && e.bytes[1]==232);
                events++;
            }
        }
        CHECK(events==4 && nl_ws_decoder_eof(d)==NL_WS_CLOSED);
        nl_ws_decoder_destroy(d);
    }
    const uint8_t utf8[]={0x01,1,0xe2,0x80,2,0x82,0xac};
    NlWsDecoder *d=decoder();size_t used;NlWsEvent e;
    CHECK(nl_ws_decoder_feed(d,utf8,sizeof(utf8),&used,&e)==NL_WS_EVENT);
    CHECK(e.opcode==NL_WS_TEXT && e.length==3 && !memcmp(e.bytes,"\xe2\x82\xac",3));
    CHECK(nl_ws_decoder_eof(d)==NL_WS_PROTOCOL);
    nl_ws_decoder_destroy(d);
    const uint8_t empty[]={0x81,0,0x89,0};d=decoder();
    CHECK(nl_ws_decoder_feed(d,empty,sizeof(empty),&used,&e)==NL_WS_EVENT && used==2 && !e.length && e.opcode==NL_WS_TEXT);
    CHECK(nl_ws_decoder_feed(d,empty+2,2,&used,&e)==NL_WS_EVENT && !e.length && e.opcode==NL_WS_PING);
    nl_ws_decoder_destroy(d);
    puts("PASS all stream splits, fragments, interleaved controls, binary/NUL and UTF-8");
}
static void refusals(void) {
    static const uint8_t cases[][12]={
        {0xc1,0},{0x81,0x80},{0x83,0},{0x80,0},{0x09,0},{0x89,126},
        {0x81,126,0,125},{0x81,127,0,0,0,0,0,0,255,255},
        {0x81,127,128,0,0,0,0,0,0,0},
        {0x81,2,0xc0,0xaf},{0x81,3,0xed,0xa0,0x80},{0x81,1,0xe2},
        {0x88,1,0},{0x88,2,3,237},{0x88,2,3,238},{0x88,2,3,247},
        {0x88,3,3,232,0xff},{0x01,0,0x81,0}
    };
    static const size_t sizes[]={2,2,2,2,2,2,4,10,10,4,5,3,3,4,4,4,5,4};
    for(size_t i=0;i<sizeof(sizes)/sizeof(*sizes);i++) {
        NlWsDecoder *d=decoder();size_t used=999;NlWsEvent e={NL_WS_PONG,NULL,777};
        CHECK(nl_ws_decoder_feed(d,cases[i],sizes[i],&used,&e)==NL_WS_PROTOCOL);
        CHECK(e.opcode==NL_WS_PONG && e.length==777);
        CHECK(nl_ws_decoder_feed(d,(const uint8_t *)"",0,&used,&e)==NL_WS_PROTOCOL && !used);
        nl_ws_decoder_destroy(d);
    }
    uint8_t oversized[]={0x82,127,0,0,0,0,0,0x10,0,1};
    NlWsDecoder *d=decoder();size_t used;NlWsEvent e;
    CHECK(nl_ws_decoder_feed(d,oversized,sizeof(oversized),&used,&e)==NL_WS_LIMIT);
    nl_ws_decoder_destroy(d);
    const uint8_t partial[]={0x81,126,0};d=decoder();
    CHECK(nl_ws_decoder_feed(d,partial,sizeof(partial),&used,&e)==NL_WS_MORE);
    CHECK(nl_ws_decoder_eof(d)==NL_WS_PROTOCOL);nl_ws_decoder_destroy(d);
    puts("PASS masked/RSV/opcode/length/UTF-8/close/fragment/EOF refusals");
}
static void encoding(void) {
    size_t lengths[]={0,1,125,126,65535,65536,NL_WS_MESSAGE_MAX};
    uint8_t *data=malloc(NL_WS_MESSAGE_MAX),*wire=malloc(NL_WS_MESSAGE_MAX+14),*server=malloc(NL_WS_MESSAGE_MAX+10);
    CHECK(data && wire && server);for(size_t i=0;i<NL_WS_MESSAGE_MAX;i++)data[i]=(uint8_t)i;
    const uint8_t mask[]={0x37,0xfa,0x21,0x3d};
    for(size_t x=0;x<sizeof(lengths)/sizeof(*lengths);x++) {
        size_t n=lengths[x],written=999,prefix=n<126?2:n<=65535?4:10;
        memset(wire,0xa5,NL_WS_MESSAGE_MAX+14);
        CHECK(nl_ws_client_frame(NL_WS_BINARY,data,n,mask,wire,n+prefix+3,&written)==NL_WS_LIMIT && written==999 && wire[0]==0xa5);
        CHECK(nl_ws_client_frame(NL_WS_BINARY,data,n,mask,wire,NL_WS_MESSAGE_MAX+14,&written)==NL_WS_OK);
        CHECK(written==n+prefix+4 && wire[0]==0x82 && (wire[1]&0x80));
        CHECK(!memcmp(wire+prefix,mask,4));
        memcpy(server,wire,prefix);server[1]&=127;
        for(size_t i=0;i<n;i++) {CHECK((wire[prefix+4+i]^mask[i%4])==data[i]);server[prefix+i]=data[i];}
        NlWsDecoder *d=decoder();size_t at=0;NlWsEvent e={0};NlWsStatus status=NL_WS_MORE;
        while(at<n+prefix) {
            size_t size=n+prefix-at;if(size>NL_WS_FEED_MAX)size=NL_WS_FEED_MAX;
            size_t used;status=nl_ws_decoder_feed(d,server+at,size,&used,&e);CHECK(used==size);at+=used;
            CHECK(status==(at==n+prefix?NL_WS_EVENT:NL_WS_MORE));
        }
        CHECK(status==NL_WS_EVENT && e.length==n && (!n || !memcmp(e.bytes,data,n)));
        nl_ws_decoder_destroy(d);
    }
    size_t written;CHECK(nl_ws_client_frame(NL_WS_PONG,(const uint8_t *)"abc",3,mask,wire,20,&written)==NL_WS_OK && written==9);
    CHECK(wire[0]==0x8a && wire[1]==0x83);
    CHECK(nl_ws_client_frame(NL_WS_CLOSE,NULL,0,mask,wire,20,&written)==NL_WS_OK && written==6 && wire[1]==0x80);
    CHECK(nl_ws_client_frame(NL_WS_TEXT,(const uint8_t *)"\xc0\xaf",2,mask,wire,20,&written)==NL_WS_PROTOCOL);
    free(data);free(wire);free(server);
    puts("PASS masked client frames at every length boundary through 1 MiB");
}
int main(void) {
#ifdef NL_WS_PROTOCOL_INSTRUMENT
    allocation_refusals();
#endif
    handshake();messages();refusals();encoding();return 0;
}
