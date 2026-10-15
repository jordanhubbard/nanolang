#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "websocket_helpers.h"
#include "../../src/nsi_websocket_transport.h"
#include "../../src/nsi_websocket_protocol.h"
#include "../../src/nsi_socket_resolver.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define WS_CONTEXTS 64u
#define WS_DEADLINE_MS 10000u
#define WS_MAX_TIMEOUT_MS 60000u
typedef struct { int64_t identity; NlWsTransport *transport; unsigned char *message; } WsLegacy;
static WsLegacy contexts[WS_CONTEXTS];
static int64_t next_identity;
static atomic_flag ws_gate=ATOMIC_FLAG_INIT;
static bool ws_enter(void){return !atomic_flag_test_and_set(&ws_gate);}
static void ws_leave(void){atomic_flag_clear(&ws_gate);}
static WsLegacy *ws_lookup(int64_t handle){
    if(handle<=0)return NULL;
    for(unsigned i=0;i<WS_CONTEXTS;i++)if(contexts[i].identity==handle)return &contexts[i];
    return NULL;
}
static int64_t ws_now(void){
    struct timespec ts;if(clock_gettime(CLOCK_MONOTONIC,&ts))return -1;
    return (int64_t)ts.tv_sec*1000+ts.tv_nsec/1000000;
}
int64_t nl_ws_connect(const char *url){
    if(!ws_enter())return 0;
    int64_t result=0,start=ws_now();WsLegacy *slot=NULL;
    if(!url || start<0 || next_identity==INT64_MAX)goto done;
    for(unsigned i=0;i<WS_CONTEXTS;i++)if(!contexts[i].identity){slot=&contexts[i];break;}
    if(!slot)goto done;
    char helper[4096];const char *selected=nl_socket_resolver_path(helper,sizeof helper)?helper:NULL;
    int64_t now=ws_now();if(now<0 || now>=start+WS_DEADLINE_MS)goto done;
    NlWsTransportPolicy policy={true,true,selected,WS_MAX_TIMEOUT_MS};
    NlWsTransport *transport=NULL;
    NlWsTransportResult r=nl_ws_transport_connect(url,strnlen(url,2048),&policy,(unsigned)(start+WS_DEADLINE_MS-now),&transport);
    if(r.status==NL_WS_TRANSPORT_OK){slot->transport=transport;slot->identity=++next_identity;result=slot->identity;}
done:ws_leave();return result;
}
int64_t nl_ws_send(int64_t handle,const char *message){
    if(!ws_enter())return -1;
    WsLegacy *slot=ws_lookup(handle);int64_t result=-1;
    if(slot){
        if(!message)message="";
        NlWsTransportResult r=nl_ws_transport_send(slot->transport,false,message,strnlen(message,NL_WS_MESSAGE_MAX+1),WS_DEADLINE_MS);
        if(r.status==NL_WS_TRANSPORT_OK)result=0;
        else (void)nl_ws_transport_abort(slot->transport);
    }
    ws_leave();return result;
}
const char *nl_ws_receive_timeout(int64_t handle,int64_t timeout){
    if(timeout<0 || timeout>WS_MAX_TIMEOUT_MS || !ws_enter())return "";
    WsLegacy *slot=ws_lookup(handle);const char *result="";int64_t start=ws_now();
    if(slot && start>=0){
        int64_t deadline=start+timeout;
        for(;;){
            int64_t now=ws_now();if(now<0)break;
            unsigned remaining=now<deadline?(unsigned)(deadline-now):0;
            NlWsMessage message={0};NlWsTransportResult r=nl_ws_transport_receive(slot->transport,remaining,&message);
            if(r.status!=NL_WS_TRANSPORT_OK)break;
            if(message.binary){nl_ws_message_free(&message);if(ws_now()>=deadline)break;continue;}
            if(message.length && memchr(message.bytes,0,message.length)){
                nl_ws_message_free(&message);(void)nl_ws_transport_abort(slot->transport);break;
            }
            free(slot->message);slot->message=message.bytes;result=(const char *)slot->message;break;
        }
    }
    ws_leave();return result;
}
const char *nl_ws_receive(int64_t handle){return nl_ws_receive_timeout(handle,WS_DEADLINE_MS);}
int64_t nl_ws_close(int64_t handle){
    if(!ws_enter())return -1;
    WsLegacy *slot=ws_lookup(handle);int64_t result=-1;
    if(slot){
        slot->identity=0;
        NlWsTransportResult r=nl_ws_transport_close(slot->transport,WS_DEADLINE_MS);
        free(slot->message);memset(slot,0,sizeof *slot);result=r.status==NL_WS_TRANSPORT_OK?0:-1;
    }
    ws_leave();return result;
}
int64_t nl_ws_is_connected(int64_t handle){
    if(!ws_enter())return 0;
    WsLegacy *slot=ws_lookup(handle);int64_t result=slot && nl_ws_transport_connected(slot->transport);
    ws_leave();return result;
}
const char *nl_ws_last_error(int64_t handle){
    if(!ws_enter())return "I am handling another WebSocket operation.";
    WsLegacy *slot=ws_lookup(handle);
    const char *result=slot?nl_ws_transport_error(slot->transport):"I require a live WebSocket handle.";
    ws_leave();return result;
}
