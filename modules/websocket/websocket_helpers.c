#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "websocket_helpers.h"
#include "../../src/nsi_socket.h"
#include "../../src/nsi_websocket_protocol.h"
#include <openssl/rand.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define WS_CONTEXTS 64u
#define WS_DEADLINE_MS 10000
#define WS_MAX_TIMEOUT_MS 60000

typedef struct {
    int64_t identity;
    NlSocketService *service;
    NlSocketToken socket;
    NlWsDecoder *decoder;
    bool connected,cleanup_failed,close_sent,peer_closed,closing;
    char error[128];
    char *message;
    size_t message_capacity;
    uint8_t incoming[4096];
    size_t incoming_at,incoming_size;
} WsCtx;
static WsCtx contexts[WS_CONTEXTS];
static int64_t next_identity;
static atomic_flag ws_gate=ATOMIC_FLAG_INIT;
static bool ws_enter(void) {return !atomic_flag_test_and_set(&ws_gate);}
static void ws_leave(void) {atomic_flag_clear(&ws_gate);}
static WsCtx *ws_lookup(int64_t handle) {
    if (handle<=0) return NULL;
    for (unsigned i=0;i<WS_CONTEXTS;i++) if(contexts[i].identity==handle) return &contexts[i];
    return NULL;
}
static int64_t ws_now(void) {
    struct timespec ts;
    if(clock_gettime(CLOCK_MONOTONIC,&ts)!=0) return -1;
    return (int64_t)ts.tv_sec*1000+ts.tv_nsec/1000000;
}
static bool ws_wait(int64_t deadline) {
    int64_t now=ws_now();
    if(now<0 || now>=deadline) return false;
    const struct timespec pause={0,1000000};
    (void)nanosleep(&pause,NULL);
    return true;
}
static bool ws_release_socket(WsCtx *c) {
    c->connected=false;
    if(!c->service) return !c->cleanup_failed;
    NlSocketResult r=nl_socket_service_destroy(c->service);
    c->service=NULL;
    c->cleanup_failed |= r.status!=NL_SOCKET_OK || r.cleanup_failed || r.closure_unknown;
    return !c->cleanup_failed;
}
static void ws_fail(WsCtx *c,const char *message) {
    (void)snprintf(c->error,sizeof(c->error),"%s",message);
    (void)ws_release_socket(c);
}
static void ws_destroy(WsCtx *c) {
    (void)ws_release_socket(c);
    nl_ws_decoder_destroy(c->decoder);free(c->message);
    memset(c,0,sizeof(*c));
}
static bool ws_send_bytes(WsCtx *c,const uint8_t *bytes,size_t size,int64_t deadline) {
    size_t at=0;
    while(at<size) {
        int64_t now=ws_now();if(now<0 || (at && now>=deadline)) return false;
        size_t n=size-at;if(n>NL_SOCKET_IO_MAX)n=NL_SOCKET_IO_MAX;
        NlSocketResult r=nl_socket_send(c->service,&c->socket,bytes+at,n);
        if(r.status==NL_SOCKET_OK && r.bytes) {at+=r.bytes;continue;}
        if((r.status!=NL_SOCKET_WOULD_BLOCK && r.status!=NL_SOCKET_INTERRUPTED) || !ws_wait(deadline)) return false;
    }
    return true;
}
static bool ws_send_frame(WsCtx *c,NlWsOpcode opcode,const uint8_t *bytes,size_t size,int64_t deadline) {
    if(size>NL_WS_MESSAGE_MAX) return false;
    uint8_t mask[4];
    if(RAND_bytes(mask,sizeof(mask))!=1) return false;
    size_t capacity=size+NL_WS_FRAME_OVERHEAD,written=0;
    uint8_t *frame=malloc(capacity);
    if(!frame)return false;
    bool ok=nl_ws_client_frame(opcode,bytes,size,mask,frame,capacity,&written)==NL_WS_OK &&
        ws_send_bytes(c,frame,written,deadline);
    free(frame);
    if(ok && opcode==NL_WS_CLOSE)c->close_sent=true;
    return ok;
}
typedef struct {char host[254],authority[280],path[1536];uint16_t port;} WsUrl;
static bool ws_url(const char *text,WsUrl *out) {
    if(!text || strncmp(text,"ws://",5)) return false; /* I refuse WSS until TLS exists. */
    size_t size=strnlen(text,2048);if(size==2048) return false;
    for(size_t i=5;i<size;i++) if((unsigned char)text[i]<=32 || (unsigned char)text[i]>=127 || text[i]=='#')return false;
    const char *start=text+5,*end=start;
    while(*end && *end!='/' && *end!='?')end++;
    size_t authority=(size_t)(end-start);
    if(!authority || authority>=sizeof(out->authority) || memchr(start,'@',authority))return false;
    const char *host=start,*host_end=end,*port=NULL;
    if(*host=='[') {
        host++;
        host_end=memchr(host,']',(size_t)(end-host));
        if(!host_end || !memchr(host,':',(size_t)(host_end-host)))return false;
        if(host_end+1<end) {if(host_end[1]!=':')return false;port=host_end+2;}
    } else {
        const char *colon=memchr(host,':',(size_t)(end-host));
        if(colon){host_end=colon;port=colon+1;}
    }
    size_t length=(size_t)(host_end-host);
    if(!length || length>=sizeof(out->host))return false;
    unsigned number=80;
    if(port) {
        if(port==end)return false;
        number=0;
        for(const char *p=port;p<end;p++) {if(*p<'0' || *p>'9')return false;number=number*10+(unsigned)(*p-'0');if(number>65535)return false;}
        if(!number)return false;
    }
    size_t path_length=strlen(end),prefix=*end=='?'?1:0;
    if(path_length+prefix>=sizeof(out->path))return false;
    memset(out,0,sizeof(*out));out->port=(uint16_t)number;
    memcpy(out->host,host,length);memcpy(out->authority,start,authority);
    if(!*end)strcpy(out->path,"/");
    else {if(prefix)out->path[0]='/';memcpy(out->path+prefix,end,path_length);}
    return true;
}
static bool ws_connect_socket(WsCtx *c,const WsUrl *url,int64_t deadline) {
    if(nl_socket_service_create(&c->service).status!=NL_SOCKET_OK)return false;
    NlSocketResolution addresses;
    /* This legacy unsafe wrapper explicitly requests host lookup. Public service
     * DNS authority and deadline supervision remain separate required work. */
    if(nl_socket_resolve_tcp(c->service,url->host,strlen(url->host),url->port,true,&addresses).status!=NL_SOCKET_OK)return false;
    for(size_t i=0;i<addresses.count;i++) {
        int64_t now=ws_now();if(now<0 || now>=deadline)return false;
        NlSocketResult r=nl_socket_acquire_tcp(c->service,&addresses.addresses[i],NL_CAP_READ|NL_CAP_WRITE,&c->socket);
        if(r.closure_unknown || r.cleanup_failed)return false;
        if(r.status!=NL_SOCKET_OK)continue;
        do {
            r=nl_socket_finish_connect(c->service,&c->socket);
            if(r.status==NL_SOCKET_OK)return true;
        } while((r.status==NL_SOCKET_WOULD_BLOCK || r.status==NL_SOCKET_INTERRUPTED) && ws_wait(deadline));
        r=nl_socket_consume_close(c->service,&c->socket);
        if(r.status!=NL_SOCKET_OK)return false;
    }
    return false;
}
static bool ws_handshake(WsCtx *c,const WsUrl *url,int64_t deadline) {
    uint8_t nonce[16];char key[25],accept[29],request[2048];
    if(RAND_bytes(nonce,sizeof(nonce))!=1 || nl_ws_handshake_key(nonce,key,accept)!=NL_WS_OK)return false;
    int size=snprintf(request,sizeof(request),"GET %s HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n",url->path,url->authority,key);
    if(size<0 || (size_t)size>=sizeof(request) || !ws_send_bytes(c,(const uint8_t *)request,(size_t)size,deadline))return false;
    char response[NL_WS_HEADER_MAX];size_t total=0;
    while(total<sizeof(response)) {
        int64_t now=ws_now();if(now<0 || now>=deadline)return false;
        /* I stop exactly at the header boundary, retaining subsequent frame bytes
         * in the socket. The buffer decoder handles coalescing after upgrade. */
        NlSocketResult r=nl_socket_receive(c->service,&c->socket,response+total,1);
        if(r.status==NL_SOCKET_OK && r.bytes==1) {
            total++;
            if(total>=4 && !memcmp(response+total-4,"\r\n\r\n",4)) {
                size_t consumed=0;
                return nl_ws_upgrade_validate(response,total,accept,&consumed)==NL_WS_OK && consumed==total;
            }
        } else if((r.status!=NL_SOCKET_WOULD_BLOCK && r.status!=NL_SOCKET_INTERRUPTED) || !ws_wait(deadline))return false;
    }
    return false;
}
int64_t nl_ws_connect(const char *text) {
    if(!ws_enter())return 0;
    WsUrl url;WsCtx *c=NULL;int64_t result=0,now=ws_now();
    if(now<0 || next_identity==INT64_MAX || !ws_url(text,&url))goto done;
    for(unsigned i=0;i<WS_CONTEXTS;i++)if(!contexts[i].identity){c=&contexts[i];break;}
    if(!c)goto done;
    if(nl_ws_decoder_create(&c->decoder)!=NL_WS_OK || !ws_connect_socket(c,&url,now+WS_DEADLINE_MS) ||
        !ws_handshake(c,&url,now+WS_DEADLINE_MS)){ws_destroy(c);goto done;}
    c->identity=++next_identity;c->connected=true;result=c->identity;
done:ws_leave();return result;
}
int64_t nl_ws_send(int64_t handle,const char *message) {
    if(!ws_enter())return -1;
    WsCtx *c=ws_lookup(handle);int64_t result=-1,now=ws_now();
    if(c && c->connected && now>=0) {
        if(!message)message="";
        size_t size=strnlen(message,NL_WS_MESSAGE_MAX+1);
        if(ws_send_frame(c,NL_WS_TEXT,(const uint8_t *)message,size,now+WS_DEADLINE_MS))result=0;
        else ws_fail(c,"I could not send a complete valid WebSocket frame.");
    }
    ws_leave();return result;
}
static const char *ws_receive(WsCtx *c,int64_t timeout) {
    int64_t now=ws_now();if(now<0)return "";
    int64_t deadline=now+timeout;
    for(;;) {
        if(c->incoming_at<c->incoming_size) {
            size_t consumed=0;NlWsEvent event;
            NlWsStatus status=nl_ws_decoder_feed(c->decoder,c->incoming+c->incoming_at,
                c->incoming_size-c->incoming_at,&consumed,&event);
            c->incoming_at+=consumed;
            if(status==NL_WS_EVENT) {
                if(event.opcode==NL_WS_PING) {
                    if(!ws_send_frame(c,NL_WS_PONG,event.bytes,event.length,deadline))goto failed;
                } else if(event.opcode==NL_WS_CLOSE) {
                    c->peer_closed=true;
                    if(!c->close_sent)(void)ws_send_frame(c,NL_WS_CLOSE,event.bytes,event.length,deadline);
                    (void)ws_release_socket(c);return "";
                } else if(event.opcode==NL_WS_TEXT && !c->closing) {
                    if(event.length && memchr(event.bytes,0,event.length)) {
                        ws_fail(c,"I cannot expose NUL text through my legacy C-string API.");return "";
                    }
                    if(event.length+1>c->message_capacity) {
                        char *next=realloc(c->message,event.length+1);if(!next)goto failed;
                        c->message=next;c->message_capacity=event.length+1;
                    }
                    if(event.length)memcpy(c->message,event.bytes,event.length);
                    c->message[event.length]=0;return c->message;
                }
            } else if(status!=NL_WS_MORE)goto failed;
        } else {
            c->incoming_at=c->incoming_size=0;
            NlSocketResult r=nl_socket_receive(c->service,&c->socket,c->incoming,sizeof(c->incoming));
            if(r.status==NL_SOCKET_OK && r.bytes){c->incoming_size=r.bytes;continue;}
            if(r.status==NL_SOCKET_EOF) {
                (void)nl_ws_decoder_eof(c->decoder);
                ws_fail(c,"I received EOF before a WebSocket close frame.");return "";
            }
            if(r.status!=NL_SOCKET_WOULD_BLOCK && r.status!=NL_SOCKET_INTERRUPTED)goto failed;
            if(!ws_wait(deadline))return "";
        }
        now=ws_now();if(now<0 || now>=deadline)return "";
    }
failed:ws_fail(c,"I refused an invalid or incomplete WebSocket operation.");return "";
}
const char *nl_ws_receive_timeout(int64_t handle,int64_t timeout_ms) {
    if(timeout_ms<0 || timeout_ms>WS_MAX_TIMEOUT_MS || !ws_enter())return "";
    WsCtx *c=ws_lookup(handle);
    const char *result=c && c->connected ? ws_receive(c,timeout_ms) : "";
    ws_leave();return result;
}
const char *nl_ws_receive(int64_t handle) {return nl_ws_receive_timeout(handle,WS_DEADLINE_MS);}
int64_t nl_ws_close(int64_t handle) {
    if(!ws_enter())return -1;
    WsCtx *c=ws_lookup(handle);int64_t result=-1,now=ws_now();
    if(c) {
        bool handshake=true;
        if(c->connected) {
            c->closing=true;
            handshake=now>=0 && ws_send_frame(c,NL_WS_CLOSE,NULL,0,now+WS_DEADLINE_MS);
            if(handshake) {
                int64_t after=ws_now();
                if(after>=0 && after<now+WS_DEADLINE_MS)
                    (void)ws_receive(c,now+WS_DEADLINE_MS-after);
                handshake=c->peer_closed;
            }
        }
        bool closed=ws_release_socket(c);ws_destroy(c);result=handshake && closed ? 0 : -1;
    }
    ws_leave();return result;
}
int64_t nl_ws_is_connected(int64_t handle) {
    if(!ws_enter())return 0;
    WsCtx *c=ws_lookup(handle);int64_t result=c && c->connected;
    ws_leave();return result;
}
const char *nl_ws_last_error(int64_t handle) {
    if(!ws_enter())return "I am handling another WebSocket operation.";
    WsCtx *c=ws_lookup(handle);const char *result=c ? c->error : "I require a live WebSocket handle.";
    ws_leave();return result;
}
