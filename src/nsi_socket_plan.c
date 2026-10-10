#include "nsi_socket_plan.h"
#include "nsi_service_catalog_internal.h"
#include "nsi_cap.h"
#include <stdlib.h>

#define IFACE "nsi:nanolang/net"
#define ID(s) IFACE "#" s
#define INT "nsi:core/int"
#define BOOL "nsi:core/bool"
#define COUNT(a) (sizeof(a) / sizeof((a)[0]))
#define MEMBER(t,n,k,d) {ID(t "." n), n, k, d}
static const NlServicePlanMember error_fields[] = {
    MEMBER("SocketError", "status", INT, NL_SERVICE_DOMAIN_NONE),
    MEMBER("SocketError", "host_errno", INT, NL_SERVICE_DOMAIN_NONE),
    MEMBER("SocketError", "cleanup_errno", INT, NL_SERVICE_DOMAIN_NONE),
    MEMBER("SocketError", "bytes", INT, NL_SERVICE_DOMAIN_NONE),
    MEMBER("SocketError", "close_attempts", INT, NL_SERVICE_DOMAIN_NONE),
    MEMBER("SocketError", "closed_count", INT, NL_SERVICE_DOMAIN_NONE),
    MEMBER("SocketError", "eof", BOOL, NL_SERVICE_DOMAIN_NONE),
    MEMBER("SocketError", "consumed", BOOL, NL_SERVICE_DOMAIN_NONE),
    MEMBER("SocketError", "cleanup_failed", BOOL, NL_SERVICE_DOMAIN_NONE),
    MEMBER("SocketError", "closure_unknown", BOOL, NL_SERVICE_DOMAIN_NONE),
    MEMBER("SocketError", "connect_pending", BOOL, NL_SERVICE_DOMAIN_NONE)
};
static const NlServicePlanMember byte_fields[] = {
    MEMBER("ReadByte", "value", INT, NL_SERVICE_DOMAIN_BYTE_INT),
    MEMBER("ReadByte", "eof", BOOL, NL_SERVICE_DOMAIN_NONE)
};
static const NlServicePlanMember endpoint_fields[] = {
    MEMBER("Endpoint", "family", INT, NL_SERVICE_DOMAIN_IP_FAMILY),
    MEMBER("Endpoint", "address0", INT, NL_SERVICE_DOMAIN_U32_INT),
    MEMBER("Endpoint", "address1", INT, NL_SERVICE_DOMAIN_U32_INT),
    MEMBER("Endpoint", "address2", INT, NL_SERVICE_DOMAIN_U32_INT),
    MEMBER("Endpoint", "address3", INT, NL_SERVICE_DOMAIN_U32_INT),
    MEMBER("Endpoint", "port", INT, NL_SERVICE_DOMAIN_PORT_INT),
    MEMBER("Endpoint", "scope_id", INT, NL_SERVICE_DOMAIN_U32_INT)
};
#define RESULT(t,ok) static const NlServicePlanMember t##_cases[] = { \
    MEMBER(#t, "Ok", ok, NL_SERVICE_DOMAIN_NONE), \
    MEMBER(#t, "Error", ID("SocketError"), NL_SERVICE_DOMAIN_NONE) }
RESULT(ConnectResult, ID("Conn"));
RESULT(SendResult, INT);
RESULT(ConnectStatus, NULL);
RESULT(ReceiveResult, ID("ReadByte"));
RESULT(CloseResult, NULL);
#define TYPE(t,k,a) {ID(#t), #t, k, a, COUNT(a)}
static const NlServicePlanType types[] = {
    {ID("Conn"), "Conn", NL_NSI_TYPE_RESOURCE, NULL, 0},
    TYPE(SocketError, NL_NSI_TYPE_RECORD, error_fields),
    TYPE(ReadByte, NL_NSI_TYPE_RECORD, byte_fields),
    TYPE(ConnectResult, NL_NSI_TYPE_VARIANT, ConnectResult_cases),
    TYPE(SendResult, NL_NSI_TYPE_VARIANT, SendResult_cases),
    TYPE(ConnectStatus, NL_NSI_TYPE_VARIANT, ConnectStatus_cases),
    TYPE(ReceiveResult, NL_NSI_TYPE_VARIANT, ReceiveResult_cases),
    TYPE(CloseResult, NL_NSI_TYPE_VARIANT, CloseResult_cases),
    TYPE(Endpoint, NL_NSI_TYPE_RECORD, endpoint_fields)
};
#define PARAM(m,n,t,dir,own,life,mut,dom) {ID(m "." n), n, t, dir, own, life, mut, dom}
#define BORROW(m) PARAM(m,"connection",ID("Conn"),NL_NSI_DIR_IN,NL_NSI_OWN_BORROW,NL_NSI_LIFE_CALL,NL_NSI_MUT_MUTABLE,NL_SERVICE_DOMAIN_NONE)
#define RETURN(m,t) PARAM(m,"result",ID(t),NL_NSI_DIR_RETURN,NL_NSI_OWN_COPY,NL_NSI_LIFE_CALLER,NL_NSI_MUT_IMMUTABLE,NL_SERVICE_DOMAIN_NONE)
static const NlServicePlanParam begin_params[] = {
    PARAM("begin_connect","endpoint",ID("Endpoint"),NL_NSI_DIR_IN,NL_NSI_OWN_COPY,NL_NSI_LIFE_CALL,NL_NSI_MUT_IMMUTABLE,NL_SERVICE_DOMAIN_NONE),
    PARAM("begin_connect","result",ID("ConnectResult"),NL_NSI_DIR_RETURN,NL_NSI_OWN_TRANSFER,NL_NSI_LIFE_RESOURCE,NL_NSI_MUT_IMMUTABLE,NL_SERVICE_DOMAIN_NONE)
};
static const NlServicePlanParam send_params[] = {
    BORROW("send_byte"),
    PARAM("send_byte","value",INT,NL_NSI_DIR_IN,NL_NSI_OWN_COPY,NL_NSI_LIFE_CALL,NL_NSI_MUT_IMMUTABLE,NL_SERVICE_DOMAIN_BYTE_INT),
    RETURN("send_byte","SendResult")
};
static const NlServicePlanParam finish_params[] = {BORROW("finish_connect"), RETURN("finish_connect","ConnectStatus")};
static const NlServicePlanParam receive_params[] = {BORROW("receive_byte"), RETURN("receive_byte","ReceiveResult")};
static const NlServicePlanParam close_params[] = {
    PARAM("close","connection",ID("Conn"),NL_NSI_DIR_IN,NL_NSI_OWN_TRANSFER,NL_NSI_LIFE_CALLEE,NL_NSI_MUT_IMMUTABLE,NL_SERVICE_DOMAIN_NONE),
    RETURN("close","CloseResult")
};
#define METHOD(n,rights,acquired,mode,p,state,owned) \
    {ID(n), n, "nsi_nanolang_net_" n, "nsi.tcp.v1." n, 1, rights, acquired, mode, p, COUNT(p), {{state, owned}, {state, NULL}}}
static const NlServicePlanMethod methods[] = {
    METHOD("begin_connect",0,NL_CAP_READ|NL_CAP_WRITE|NL_CAP_TRANSFER,NL_SERVICE_INPUT_NONE,begin_params,NL_SERVICE_OWNER_NONE,ID("Conn")),
    METHOD("send_byte",NL_CAP_WRITE,0,NL_SERVICE_INPUT_EXCLUSIVE,send_params,NL_SERVICE_OWNER_PRESERVED,NULL),
    METHOD("finish_connect",0,0,NL_SERVICE_INPUT_EXCLUSIVE,finish_params,NL_SERVICE_OWNER_PRESERVED,NULL),
    METHOD("receive_byte",NL_CAP_READ,0,NL_SERVICE_INPUT_EXCLUSIVE,receive_params,NL_SERVICE_OWNER_PRESERVED,NULL),
    METHOD("close",0,0,NL_SERVICE_INPUT_CONSUME,close_params,NL_SERVICE_OWNER_CONSUMED,NULL)
};
_Static_assert(COUNT(methods) == NL_SOCKET_PLAN_METHODS, "I require all TCP methods");
_Static_assert(COUNT(types) == NL_SOCKET_PLAN_TYPES, "I require all TCP types");
static const NlServiceCatalog catalog = {
    IFACE, "net", ID("io"), "io", "1", "cap:nanolang/net.connect", "connect",
    methods, COUNT(methods), types, COUNT(types)
};
struct NlSocketPlan { const NlServiceCatalog *catalog; };
NlSocketPlanStatus nl_socket_plan_build(const NlNsi *n, NlSocketPlan **out) {
    if (!out || !nl_service_catalog_document_equal(n, &catalog)) return NL_SOCKET_PLAN_INVALID;
    NlSocketPlan *p = malloc(sizeof(*p));
    if (!p) return NL_SOCKET_PLAN_MEMORY;
    p->catalog = &catalog;
    *out = p;
    return NL_SOCKET_PLAN_OK;
}
void nl_socket_plan_free(NlSocketPlan *p) { free(p); }
size_t nl_socket_plan_storage_size(void) { return sizeof(NlSocketPlan); }
const char *nl_socket_plan_interface(const NlSocketPlan *p) { return p ? p->catalog->interface_id : NULL; }
size_t nl_socket_plan_method_count(const NlSocketPlan *p) { return p ? p->catalog->method_count : 0; }
size_t nl_socket_plan_type_count(const NlSocketPlan *p) { return p ? p->catalog->type_count : 0; }
const NlServicePlanMethod *nl_socket_plan_method(const NlSocketPlan *p, size_t i) {
    return p && i < p->catalog->method_count ? &p->catalog->methods[i] : NULL;
}
const NlServicePlanType *nl_socket_plan_type(const NlSocketPlan *p, size_t i) {
    return p && i < p->catalog->type_count ? &p->catalog->types[i] : NULL;
}
const char *nl_socket_catalog_interface(void) { return IFACE; }
const NlServicePlanMethod *nl_socket_catalog_method(size_t i) { return i < COUNT(methods) ? &methods[i] : NULL; }
const NlServicePlanType *nl_socket_catalog_type(size_t i) { return i < COUNT(types) ? &types[i] : NULL; }
