#include "nsi_websocket_plan.h"
#include "nsi_service_catalog_internal.h"
#include "nsi_cap.h"
#include <stdlib.h>

#define IFACE "nsi:nanolang/websocket"
#define ID(s) IFACE "#" s
#define INT "nsi:core/int"
#define BOOL "nsi:core/bool"
#define STRING "nsi:core/string"
#define COUNT(a) (sizeof(a)/sizeof((a)[0]))
#define MEMBER(t,n,k) {ID(t "." n),n,k,NL_SERVICE_DOMAIN_NONE}
static const NlServicePlanMember error_fields[]={
    MEMBER("WebSocketError","status",INT),
    MEMBER("WebSocketError","host_errno",INT),
    MEMBER("WebSocketError","resolver_error",INT),
    MEMBER("WebSocketError","supervisor_status",INT),
    MEMBER("WebSocketError","close_code",INT),
    MEMBER("WebSocketError","cleanup_errno",INT),
    MEMBER("WebSocketError","cleanup_failed",BOOL),
    MEMBER("WebSocketError","closure_unknown",BOOL),
    MEMBER("WebSocketError","terminal",BOOL)
};
static const NlServicePlanMember message_fields[]={
    MEMBER("Message","binary",BOOL),MEMBER("Message","data",STRING)
};
#define RESULT(t,ok) static const NlServicePlanMember t##_cases[]={ \
    MEMBER(#t,"Ok",ok),MEMBER(#t,"Error",ID("WebSocketError")) }
RESULT(ConnectResult,ID("Connection"));
RESULT(SendResult,INT);
RESULT(ReceiveResult,ID("Message"));
RESULT(CloseResult,NULL);
#define TYPE(t,k,a) {ID(#t),#t,k,a,COUNT(a)}
static const NlServicePlanType types[]={
    {ID("Connection"),"Connection",NL_NSI_TYPE_RESOURCE,NULL,0},
    TYPE(WebSocketError,NL_NSI_TYPE_RECORD,error_fields),
    TYPE(Message,NL_NSI_TYPE_RECORD,message_fields),
    TYPE(ConnectResult,NL_NSI_TYPE_VARIANT,ConnectResult_cases),
    TYPE(SendResult,NL_NSI_TYPE_VARIANT,SendResult_cases),
    TYPE(ReceiveResult,NL_NSI_TYPE_VARIANT,ReceiveResult_cases),
    TYPE(CloseResult,NL_NSI_TYPE_VARIANT,CloseResult_cases)
};
#define PARAM(m,n,t,dir,own,life,mut,dom) {ID(m "." n),n,t,dir,own,life,mut,dom}
#define INPUT(m,n,t) PARAM(m,n,t,NL_NSI_DIR_IN,NL_NSI_OWN_COPY,NL_NSI_LIFE_CALL,NL_NSI_MUT_IMMUTABLE,NL_SERVICE_DOMAIN_NONE)
#define TIMEOUT(m) PARAM(m,"timeout_ms",INT,NL_NSI_DIR_IN,NL_NSI_OWN_COPY,NL_NSI_LIFE_CALL,NL_NSI_MUT_IMMUTABLE,NL_SERVICE_DOMAIN_TIMEOUT_MS)
#define BORROW(m) PARAM(m,"connection",ID("Connection"),NL_NSI_DIR_IN,NL_NSI_OWN_BORROW,NL_NSI_LIFE_CALL,NL_NSI_MUT_MUTABLE,NL_SERVICE_DOMAIN_NONE)
#define RETURN(m,t) PARAM(m,"result",ID(t),NL_NSI_DIR_RETURN,NL_NSI_OWN_COPY,NL_NSI_LIFE_CALLER,NL_NSI_MUT_IMMUTABLE,NL_SERVICE_DOMAIN_NONE)
static const NlServicePlanParam connect_params[]={
    INPUT("connect","url",STRING),TIMEOUT("connect"),
    PARAM("connect","result",ID("ConnectResult"),NL_NSI_DIR_RETURN,NL_NSI_OWN_TRANSFER,NL_NSI_LIFE_RESOURCE,NL_NSI_MUT_IMMUTABLE,NL_SERVICE_DOMAIN_NONE)
};
static const NlServicePlanParam send_params[]={BORROW("send"),INPUT("send","message",ID("Message")),TIMEOUT("send"),RETURN("send","SendResult")};
static const NlServicePlanParam receive_params[]={BORROW("receive"),TIMEOUT("receive"),RETURN("receive","ReceiveResult")};
static const NlServicePlanParam close_params[]={
    PARAM("close","connection",ID("Connection"),NL_NSI_DIR_IN,NL_NSI_OWN_TRANSFER,NL_NSI_LIFE_CALLEE,NL_NSI_MUT_IMMUTABLE,NL_SERVICE_DOMAIN_NONE),
    TIMEOUT("close"),RETURN("close","CloseResult")
};
#define METHOD(n,rights,acquired,mode,p,state,owned) \
    {ID(n),n,"nsi_nanolang_websocket_" n,"nsi.websocket.v1." n,1,rights,acquired,mode,p,COUNT(p),{{state,owned},{state,NULL}}}
static const NlServicePlanMethod methods[]={
    METHOD("connect",0,NL_CAP_READ|NL_CAP_WRITE|NL_CAP_TRANSFER,NL_SERVICE_INPUT_NONE,connect_params,NL_SERVICE_OWNER_NONE,ID("Connection")),
    METHOD("send",NL_CAP_WRITE,0,NL_SERVICE_INPUT_EXCLUSIVE,send_params,NL_SERVICE_OWNER_PRESERVED,NULL),
    METHOD("receive",NL_CAP_READ,0,NL_SERVICE_INPUT_EXCLUSIVE,receive_params,NL_SERVICE_OWNER_PRESERVED,NULL),
    METHOD("close",0,0,NL_SERVICE_INPUT_CONSUME,close_params,NL_SERVICE_OWNER_CONSUMED,NULL)
};
static const NlServicePlanCapability capabilities[]={
    {"cap:nanolang/websocket.connect","connect"},
    {"cap:nanolang/net.lookup","lookup"}
};
_Static_assert(COUNT(types)==NL_WEBSOCKET_PLAN_TYPES,"I retain all WebSocket types");
_Static_assert(COUNT(methods)==NL_WEBSOCKET_PLAN_METHODS,"I retain all WebSocket methods");
_Static_assert(COUNT(capabilities)==NL_WEBSOCKET_PLAN_CAPABILITIES,"I separate network and DNS authority");
static const NlServiceCatalog catalog={IFACE,"websocket",ID("io"),"io","1",capabilities,COUNT(capabilities),methods,COUNT(methods),types,COUNT(types)};
struct NlWebSocketPlan { const NlServiceCatalog *catalog; };
NlWebSocketPlanStatus nl_websocket_plan_build(const NlNsi *n,NlWebSocketPlan **out) {
    if(!out || !nl_service_catalog_document_equal(n,&catalog))return NL_WEBSOCKET_PLAN_INVALID;
    NlWebSocketPlan *p=malloc(sizeof *p);if(!p)return NL_WEBSOCKET_PLAN_MEMORY;
    p->catalog=&catalog;*out=p;return NL_WEBSOCKET_PLAN_OK;
}
void nl_websocket_plan_free(NlWebSocketPlan *p) { free(p); }
size_t nl_websocket_plan_storage_size(void) { return sizeof(NlWebSocketPlan); }
const char *nl_websocket_plan_interface(const NlWebSocketPlan *p) { return p?p->catalog->interface_id:NULL; }
size_t nl_websocket_plan_method_count(const NlWebSocketPlan *p) { return p?p->catalog->method_count:0; }
size_t nl_websocket_plan_type_count(const NlWebSocketPlan *p) { return p?p->catalog->type_count:0; }
size_t nl_websocket_plan_capability_count(const NlWebSocketPlan *p) { return p?p->catalog->capability_count:0; }
const NlServicePlanMethod *nl_websocket_plan_method(const NlWebSocketPlan *p,size_t i) { return p && i<p->catalog->method_count?&p->catalog->methods[i]:NULL; }
const NlServicePlanType *nl_websocket_plan_type(const NlWebSocketPlan *p,size_t i) { return p && i<p->catalog->type_count?&p->catalog->types[i]:NULL; }
const NlServicePlanCapability *nl_websocket_plan_capability(const NlWebSocketPlan *p,size_t i) { return p && i<p->catalog->capability_count?&p->catalog->capabilities[i]:NULL; }
const char *nl_websocket_catalog_interface(void) { return IFACE; }
const NlServicePlanMethod *nl_websocket_catalog_method(size_t i) { return i<COUNT(methods)?&methods[i]:NULL; }
const NlServicePlanType *nl_websocket_catalog_type(size_t i) { return i<COUNT(types)?&types[i]:NULL; }
const NlServicePlanCapability *nl_websocket_catalog_capability(size_t i) { return i<COUNT(capabilities)?&capabilities[i]:NULL; }
