#ifndef NL_NSI_SOCKET_RESOLVER_WIRE_H
#define NL_NSI_SOCKET_RESOLVER_WIRE_H
#include "nsi_socket_resolver.h"
#include <string.h>

/* I exchange fixed-width big-endian fields, not C structure padding or pointers. */
#define NL_RESOLVE_WIRE_SIZE (20u + NL_SOCKET_RESOLVE_MAX * 28u)
static inline uint32_t nl_resolve_get(const unsigned char *p) {
    return (uint32_t)p[0]<<24 | (uint32_t)p[1]<<16 | (uint32_t)p[2]<<8 | p[3];
}
static inline void nl_resolve_put(unsigned char *p,uint32_t v) {
    p[0]=(unsigned char)(v>>24);p[1]=(unsigned char)(v>>16);
    p[2]=(unsigned char)(v>>8);p[3]=(unsigned char)v;
}
static inline void nl_resolve_encode(unsigned char *p,NlSocketResolveResult r,
    const NlSocketResolution *value) {
    memset(p,0,NL_RESOLVE_WIRE_SIZE);memcpy(p,"NLR1",4);
    nl_resolve_put(p+4,(uint32_t)r.status);nl_resolve_put(p+8,(uint32_t)r.resolver_error);
    nl_resolve_put(p+12,(uint32_t)r.host_errno);
    if(r.status!=NL_SOCKET_OK)return;
    nl_resolve_put(p+16,(uint32_t)value->count);
    for(size_t i=0;i<value->count;i++) {
        unsigned char *a=p+20+i*28;
        nl_resolve_put(a,(uint32_t)value->addresses[i].family);
        nl_resolve_put(a+4,value->addresses[i].port);
        nl_resolve_put(a+8,value->addresses[i].scope_id);
        memcpy(a+12,value->addresses[i].address,16);
    }
}
#endif
