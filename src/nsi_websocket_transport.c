#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "nsi_websocket_transport.h"
#include "nsi_socket.h"
#include "nsi_socket_resolver.h"
#include "nsi_websocket_protocol.h"
#include <openssl/rand.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct NlWsTransport {
    NlSocketService *service;NlSocketToken socket;NlWsDecoder *decoder;
    unsigned max_timeout_ms;
    bool connected,close_sent,peer_closed,closing,pending;
    bool cleanup_failed,closure_unknown;
    int cleanup_errno,close_code;
    NlWsTransportResult last;
    NlWsEvent event;
    uint8_t incoming[4096];size_t incoming_at,incoming_size;
};
bool nl_ws_transport_storage_bound(size_t *out) {
    size_t decoder,socket;
    if(!out || !nl_ws_decoder_storage_bound(&decoder) || !nl_socket_service_storage_bound(&socket))return false;
    size_t fixed=sizeof(NlWsTransport)+NL_WS_MESSAGE_MAX+NL_WS_FRAME_OVERHEAD;
    if(decoder>SIZE_MAX-fixed || socket>SIZE_MAX-fixed-decoder)return false;
    *out=fixed+decoder+socket;return true;
}
static int64_t ws_now(void) {
    struct timespec ts;if(clock_gettime(CLOCK_MONOTONIC,&ts))return -1;
    return (int64_t)ts.tv_sec*1000+ts.tv_nsec/1000000;
}
static bool ws_wait(int64_t deadline) {
    int64_t now=ws_now();if(now<0 || now>=deadline)return false;
    const struct timespec pause={0,1000000};(void)nanosleep(&pause,NULL);return true;
}
static NlWsTransportResult ws_result(NlWsTransport *c,NlWsTransportStatus status) {
    if(!c)return (NlWsTransportResult){.status=status};
    c->last.status=status;c->last.terminal=!c->connected;c->last.close_code=c->close_code;
    c->last.cleanup_errno=c->cleanup_errno;c->last.cleanup_failed=c->cleanup_failed;
    c->last.closure_unknown=c->closure_unknown;return c->last;
}
static void ws_socket_result(NlWsTransport *c,NlSocketResult r) {
    c->last.host_errno=r.host_errno;
    if(!c->cleanup_errno)c->cleanup_errno=r.cleanup_errno;
    if(!c->cleanup_errno && (r.cleanup_failed || r.closure_unknown))c->cleanup_errno=r.host_errno;
    c->cleanup_failed|=r.cleanup_failed;c->closure_unknown|=r.closure_unknown;
}
static void ws_release(NlWsTransport *c) {
    c->connected=false;
    if(c->service) {
        NlSocketResult r=nl_socket_service_destroy(c->service);c->service=NULL;
        int prior=c->last.host_errno;ws_socket_result(c,r);if(prior)c->last.host_errno=prior;
        c->cleanup_failed|=r.status!=NL_SOCKET_OK;
    }
}
static NlWsTransportResult ws_fail(NlWsTransport *c,NlWsTransportStatus status) {
    ws_release(c);return ws_result(c,status);
}
static NlWsTransportStatus ws_socket_status(NlSocketStatus status) {
    return status==NL_SOCKET_MEMORY?NL_WS_TRANSPORT_MEMORY:
        status==NL_SOCKET_LIMIT || status==NL_SOCKET_CAPACITY?NL_WS_TRANSPORT_LIMIT:NL_WS_TRANSPORT_IO;
}
static bool ws_deadline(NlWsTransport *c,unsigned timeout,int64_t *deadline) {
    if(timeout>c->max_timeout_ms || timeout>60000){ws_result(c,NL_WS_TRANSPORT_LIMIT);return false;}
    int64_t now=ws_now();if(now<0){ws_result(c,NL_WS_TRANSPORT_IO);return false;}
    *deadline=now+timeout;return true;
}
static bool ws_send_bytes(NlWsTransport *c,const uint8_t *bytes,size_t length,int64_t deadline) {
    size_t at=0;
    while(at<length) {
        int64_t now=ws_now();
        if(now<0 || (at && now>=deadline)){ws_result(c,now<0?NL_WS_TRANSPORT_IO:NL_WS_TRANSPORT_TIMEOUT);return false;}
        size_t count=length-at;if(count>NL_SOCKET_IO_MAX)count=NL_SOCKET_IO_MAX;
        NlSocketResult r=nl_socket_send(c->service,&c->socket,bytes+at,count);
        if(r.status==NL_SOCKET_OK && r.bytes){at+=r.bytes;continue;}
        if(r.status!=NL_SOCKET_WOULD_BLOCK && r.status!=NL_SOCKET_INTERRUPTED) {
            ws_socket_result(c,r);ws_result(c,ws_socket_status(r.status));return false;
        }
        if(!ws_wait(deadline)){ws_result(c,NL_WS_TRANSPORT_TIMEOUT);return false;}
    }
    return true;
}
static bool ws_send_frame(NlWsTransport *c,NlWsOpcode opcode,const uint8_t *bytes,size_t length,int64_t deadline) {
    if(length>NL_WS_MESSAGE_MAX){ws_result(c,NL_WS_TRANSPORT_LIMIT);return false;}
    uint8_t mask[4];if(RAND_bytes(mask,sizeof mask)!=1){ws_result(c,NL_WS_TRANSPORT_CRYPTO);return false;}
    size_t capacity=length+NL_WS_FRAME_OVERHEAD,written=0;
    uint8_t *frame=malloc(capacity);if(!frame){ws_result(c,NL_WS_TRANSPORT_MEMORY);return false;}
    NlWsStatus encoded=nl_ws_client_frame(opcode,bytes,length,mask,frame,capacity,&written);
    bool ok=false;
    if(encoded!=NL_WS_OK)ws_result(c,NL_WS_TRANSPORT_ARGUMENT);
    else {ok=ws_send_bytes(c,frame,written,deadline);if(!ok)ws_release(c);}
    free(frame);if(ok && opcode==NL_WS_CLOSE)c->close_sent=true;return ok;
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


static bool ws_connect_socket(NlWsTransport *c,const WsUrl *url,
    const NlWsTransportPolicy *policy,int64_t deadline) {
    NlSocketResult created=nl_socket_service_create(&c->service);
    if(created.status!=NL_SOCKET_OK){ws_socket_result(c,created);ws_result(c,ws_socket_status(created.status));return false;}
    NlSocketResolution addresses;
    NlSocketResolveResult numeric=nl_socket_resolve_tcp(c->service,url->host,strlen(url->host),url->port,false,&addresses);
    if(numeric.status==NL_SOCKET_RIGHTS) {
        if(!policy->allow_lookup){ws_result(c,NL_WS_TRANSPORT_RIGHTS);return false;}
        int64_t now=ws_now();if(now<0 || now>=deadline){ws_result(c,NL_WS_TRANSPORT_TIMEOUT);return false;}
        NlSocketLookupResult lookup=nl_socket_resolve_tcp_supervised(c->service,url->host,strlen(url->host),url->port,true,
            policy->resolver_helper,(unsigned)(deadline-now),&addresses);
        c->last.host_errno=lookup.supervision==NL_LOOKUP_COMPLETE?lookup.resolver.host_errno:lookup.supervisor_errno;c->last.resolver_error=lookup.resolver.resolver_error;
        c->last.supervisor_status=lookup.supervision;
        if(lookup.supervision!=NL_LOOKUP_COMPLETE || lookup.resolver.status!=NL_SOCKET_OK) {
            ws_result(c,lookup.supervision==NL_LOOKUP_TIMEOUT?NL_WS_TRANSPORT_TIMEOUT:
                lookup.supervision==NL_LOOKUP_ARGUMENT?NL_WS_TRANSPORT_ARGUMENT:ws_socket_status(lookup.resolver.status));return false;
        }
    } else if(numeric.status!=NL_SOCKET_OK){ws_result(c,NL_WS_TRANSPORT_ARGUMENT);return false;}
    for(size_t i=0;i<addresses.count;i++) {
        int64_t now=ws_now();if(now<0 || now>=deadline){ws_result(c,NL_WS_TRANSPORT_TIMEOUT);return false;}
        NlSocketResult r=nl_socket_acquire_tcp(c->service,&addresses.addresses[i],NL_CAP_READ|NL_CAP_WRITE,&c->socket);
        if(r.cleanup_failed || r.closure_unknown){ws_socket_result(c,r);ws_result(c,NL_WS_TRANSPORT_IO);return false;}
        if(r.status!=NL_SOCKET_OK){ws_socket_result(c,r);ws_result(c,ws_socket_status(r.status));continue;}
        do {
            r=nl_socket_finish_connect(c->service,&c->socket);
            if(r.status==NL_SOCKET_OK)return true;
        } while((r.status==NL_SOCKET_WOULD_BLOCK || r.status==NL_SOCKET_INTERRUPTED) && ws_wait(deadline));
        ws_socket_result(c,r);ws_result(c,r.status==NL_SOCKET_WOULD_BLOCK || r.status==NL_SOCKET_INTERRUPTED?NL_WS_TRANSPORT_TIMEOUT:ws_socket_status(r.status));
        r=nl_socket_consume_close(c->service,&c->socket);
        if(r.status!=NL_SOCKET_OK){ws_socket_result(c,r);ws_result(c,NL_WS_TRANSPORT_IO);return false;}
    }
    return false;
}
static bool ws_handshake(NlWsTransport *c,const WsUrl *url,int64_t deadline) {
    uint8_t nonce[16];char key[25],accept[29],request[2048];
    if(RAND_bytes(nonce,sizeof nonce)!=1 || nl_ws_handshake_key(nonce,key,accept)!=NL_WS_OK){ws_result(c,NL_WS_TRANSPORT_CRYPTO);return false;}
    int size=snprintf(request,sizeof request,"GET %s HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n",url->path,url->authority,key);
    if(size<0 || (size_t)size>=sizeof request){ws_result(c,NL_WS_TRANSPORT_LIMIT);return false;}
    if(!ws_send_bytes(c,(const uint8_t *)request,(size_t)size,deadline))return false;
    char response[NL_WS_HEADER_MAX];size_t total=0;
    while(total<sizeof response) {
        int64_t now=ws_now();if(now<0 || now>=deadline){ws_result(c,NL_WS_TRANSPORT_TIMEOUT);return false;}
        NlSocketResult r=nl_socket_receive(c->service,&c->socket,response+total,1);
        if(r.status==NL_SOCKET_OK && r.bytes==1) {
            total++;
            if(total>=4 && !memcmp(response+total-4,"\r\n\r\n",4)) {
                size_t consumed=0;
                if(nl_ws_upgrade_validate(response,total,accept,&consumed)==NL_WS_OK && consumed==total)return true;
                ws_result(c,NL_WS_TRANSPORT_PROTOCOL);return false;
            }
        } else if(r.status!=NL_SOCKET_WOULD_BLOCK && r.status!=NL_SOCKET_INTERRUPTED) {
            ws_socket_result(c,r);ws_result(c,NL_WS_TRANSPORT_IO);return false;
        } else if(!ws_wait(deadline)){ws_result(c,NL_WS_TRANSPORT_TIMEOUT);return false;}
    }
    ws_result(c,NL_WS_TRANSPORT_LIMIT);return false;
}
NlWsTransportResult nl_ws_transport_connect(const void *url,size_t length,
    const NlWsTransportPolicy *policy,unsigned timeout,NlWsTransport **out) {
    NlWsTransportResult result={.status=NL_WS_TRANSPORT_ARGUMENT};
    if(!out || !policy || !url || !length || length>=2048 || memchr(url,0,length))return result;
    if(!policy->allow_network){result.status=NL_WS_TRANSPORT_RIGHTS;return result;}
    if(policy->max_timeout_ms>60000 || timeout>policy->max_timeout_ms){result.status=NL_WS_TRANSPORT_LIMIT;return result;}
    char text[2048];memcpy(text,url,length);text[length]=0;WsUrl parsed;
    if(!ws_url(text,&parsed))return result;
    if(!timeout){result.status=NL_WS_TRANSPORT_TIMEOUT;return result;}
    int64_t now=ws_now();if(now<0){result.status=NL_WS_TRANSPORT_IO;return result;}
    NlWsTransport *c=calloc(1,sizeof *c);if(!c){result.status=NL_WS_TRANSPORT_MEMORY;return result;}
    c->max_timeout_ms=policy->max_timeout_ms;
    NlWsStatus status=nl_ws_decoder_create(&c->decoder);
    if(status!=NL_WS_OK)ws_result(c,status==NL_WS_MEMORY?NL_WS_TRANSPORT_MEMORY:NL_WS_TRANSPORT_IO);
    else if(ws_connect_socket(c,&parsed,policy,now+timeout) && ws_handshake(c,&parsed,now+timeout)) {
        c->last=(NlWsTransportResult){0};c->connected=true;*out=c;return ws_result(c,NL_WS_TRANSPORT_OK);
    }
    ws_release(c);result=ws_result(c,c->last.status);nl_ws_decoder_destroy(c->decoder);free(c);return result;
}
NlWsTransportResult nl_ws_transport_send(NlWsTransport *c,bool binary,const void *bytes,size_t length,unsigned timeout) {
    if(!c)return ws_result(NULL,NL_WS_TRANSPORT_ARGUMENT);
    c->last=(NlWsTransportResult){0};
    if(!c->connected)return ws_result(c,NL_WS_TRANSPORT_CLOSED);
    if(!bytes && length)return ws_result(c,NL_WS_TRANSPORT_ARGUMENT);
    int64_t deadline;if(!ws_deadline(c,timeout,&deadline))return c->last;
    if(!ws_send_frame(c,binary?NL_WS_BINARY:NL_WS_TEXT,bytes,length,deadline))return ws_result(c,c->last.status);
    c->last.bytes=length;return ws_result(c,NL_WS_TRANSPORT_OK);
}
static NlWsTransportResult ws_receive(NlWsTransport *c,int64_t deadline,NlWsMessage *out) {
    for(;;) {
        if(c->pending) {
            if(c->closing)c->pending=false;
            else {
                unsigned char *copy=malloc(c->event.length+1);
                if(!copy)return ws_result(c,NL_WS_TRANSPORT_MEMORY);
                if(c->event.length)memcpy(copy,c->event.bytes,c->event.length);
                copy[c->event.length]=0;
                *out=(NlWsMessage){c->event.opcode==NL_WS_BINARY,copy,c->event.length};
                c->pending=false;c->last.bytes=out->length;return ws_result(c,NL_WS_TRANSPORT_OK);
            }
        }
        if(c->incoming_at<c->incoming_size) {
            size_t consumed=0;NlWsEvent event;
            NlWsStatus status=nl_ws_decoder_feed(c->decoder,c->incoming+c->incoming_at,c->incoming_size-c->incoming_at,&consumed,&event);
            c->incoming_at+=consumed;
            if(status==NL_WS_EVENT) {
                if(event.opcode==NL_WS_PING) {
                    if(!ws_send_frame(c,NL_WS_PONG,event.bytes,event.length,deadline))return ws_fail(c,c->last.status);
                } else if(event.opcode==NL_WS_CLOSE) {
                    c->peer_closed=true;c->close_code=event.length>=2?(int)event.bytes[0]*256+event.bytes[1]:0;
                    bool replied=c->close_sent || ws_send_frame(c,NL_WS_CLOSE,event.bytes,event.length,deadline);
                    return ws_fail(c,replied?NL_WS_TRANSPORT_CLOSED:c->last.status);
                } else if((event.opcode==NL_WS_TEXT || event.opcode==NL_WS_BINARY) && !c->closing) {
                    c->event=event;c->pending=true;continue;
                }
            } else if(status!=NL_WS_MORE)return ws_fail(c,status==NL_WS_MEMORY?NL_WS_TRANSPORT_MEMORY:NL_WS_TRANSPORT_PROTOCOL);
        } else {
            c->incoming_at=c->incoming_size=0;
            NlSocketResult r=nl_socket_receive(c->service,&c->socket,c->incoming,sizeof c->incoming);
            if(r.status==NL_SOCKET_OK && r.bytes){c->incoming_size=r.bytes;continue;}
            if(r.status==NL_SOCKET_EOF){(void)nl_ws_decoder_eof(c->decoder);return ws_fail(c,NL_WS_TRANSPORT_PROTOCOL);}
            if(r.status!=NL_SOCKET_WOULD_BLOCK && r.status!=NL_SOCKET_INTERRUPTED){ws_socket_result(c,r);return ws_fail(c,ws_socket_status(r.status));}
            if(!ws_wait(deadline))return ws_result(c,NL_WS_TRANSPORT_TIMEOUT);
        }
        int64_t now=ws_now();if(now<0)return ws_fail(c,NL_WS_TRANSPORT_IO);
        if(now>=deadline)return ws_result(c,NL_WS_TRANSPORT_TIMEOUT);
    }
}
NlWsTransportResult nl_ws_transport_receive(NlWsTransport *c,unsigned timeout,NlWsMessage *out) {
    if(!c || !out)return ws_result(c,NL_WS_TRANSPORT_ARGUMENT);
    c->last=(NlWsTransportResult){0};if(!c->connected)return ws_result(c,NL_WS_TRANSPORT_CLOSED);
    int64_t deadline;if(!ws_deadline(c,timeout,&deadline))return c->last;
    return ws_receive(c,deadline,out);
}
void nl_ws_message_free(NlWsMessage *message) {
    if(message){free(message->bytes);*message=(NlWsMessage){0};}
}
NlWsTransportResult nl_ws_transport_abort(NlWsTransport *c) {
    if(!c)return ws_result(NULL,NL_WS_TRANSPORT_ARGUMENT);
    return ws_fail(c,NL_WS_TRANSPORT_PROTOCOL);
}
NlWsTransportResult nl_ws_transport_close(NlWsTransport *c,unsigned timeout) {
    if(!c)return ws_result(NULL,NL_WS_TRANSPORT_ARGUMENT);
    c->last=(NlWsTransportResult){0};NlWsTransportStatus status=NL_WS_TRANSPORT_OK;int64_t deadline;
    if(!ws_deadline(c,timeout,&deadline))status=c->last.status;
    else if(c->connected) {
        c->closing=true;c->pending=false;
        if(!ws_send_frame(c,NL_WS_CLOSE,NULL,0,deadline))status=c->last.status;
        else {
            NlWsMessage ignored;NlWsTransportResult r=ws_receive(c,deadline,&ignored);
            if(!c->peer_closed)status=r.status;
        }
    }
    ws_release(c);if(c->cleanup_failed && status==NL_WS_TRANSPORT_OK)status=NL_WS_TRANSPORT_IO;
    NlWsTransportResult result=ws_result(c,status);nl_ws_decoder_destroy(c->decoder);free(c);return result;
}
bool nl_ws_transport_connected(const NlWsTransport *c) { return c && c->connected; }
const char *nl_ws_transport_error(const NlWsTransport *c) {
    if(!c)return "I require an owned WebSocket connection.";
    static const char *messages[]={"","I require valid WebSocket arguments.","I require WebSocket network or lookup authority.",
        "I could not allocate WebSocket storage.","I exceeded my WebSocket bound.","I reached my WebSocket deadline.",
        "I could not complete WebSocket host I/O.","I refused an invalid WebSocket protocol operation.",
        "I could not obtain WebSocket cryptographic randomness.","I have closed this WebSocket transport."};
    return messages[c->last.status];
}
