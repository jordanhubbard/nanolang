#include "service_bindings.h"
#include <string.h>

static uint16_t service_u16(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8));
}
static uint32_t service_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static void service_put16(uint8_t *p,uint16_t value) {
    p[0]=(uint8_t)value;p[1]=(uint8_t)(value >> 8);
}
static void service_put32(uint8_t *p,uint32_t value) {
    p[0]=(uint8_t)value;p[1]=(uint8_t)(value >> 8);
    p[2]=(uint8_t)(value >> 16);p[3]=(uint8_t)(value >> 24);
}
NvmServiceResult nvm_service_bindings_check(const NvmServiceBindings *value) {
    if(!value) return NVM_SERVICE_ARGUMENT;
    for(size_t i=0;i<NVM_SERVICE_BINDING_COUNT;i++) {
        if(value->imports[i]==UINT32_MAX) return NVM_SERVICE_INDEX;
        for(size_t j=0;j<i;j++)
            if(value->imports[i]==value->imports[j]) return NVM_SERVICE_INDEX;
    }
    return NVM_SERVICE_OK;
}
NvmServiceResult nvm_service_bindings_decode(const uint8_t *bytes,size_t size,
                                             NvmServiceBindings *out) {
    if(!bytes || !out) return NVM_SERVICE_ARGUMENT;
    if(size!=NVM_SERVICE_BINDING_BYTES) return NVM_SERVICE_SIZE;
    if(service_u16(bytes)!=NVM_SERVICE_BINDING_VERSION) return NVM_SERVICE_VERSION;
    if(service_u16(bytes+2)!=NVM_SERVICE_BINDING_CATALOG_FILE) return NVM_SERVICE_CATALOG;
    if(service_u32(bytes+4)!=NVM_SERVICE_BINDING_COUNT) return NVM_SERVICE_COUNT;
    if(service_u32(bytes+8) || service_u32(bytes+12)) return NVM_SERVICE_RESERVED;
    NvmServiceBindings staged;
    for(size_t i=0;i<NVM_SERVICE_BINDING_COUNT;i++) {
        size_t offset=16+8*i;
        if(service_u32(bytes+offset)!=i) return NVM_SERVICE_ORDINAL;
        staged.imports[i]=service_u32(bytes+offset+4);
    }
    NvmServiceResult status=nvm_service_bindings_check(&staged);
    if(status!=NVM_SERVICE_OK) return status;
    *out=staged;
    return NVM_SERVICE_OK;
}
NvmServiceResult nvm_service_bindings_encode(const NvmServiceBindings *value,
                                             uint8_t *bytes,size_t capacity,size_t *size) {
    if(!size) return NVM_SERVICE_ARGUMENT;
    NvmServiceResult status=nvm_service_bindings_check(value);
    if(status!=NVM_SERVICE_OK) return status;
    if(bytes && capacity<NVM_SERVICE_BINDING_BYTES) return NVM_SERVICE_SIZE;
    if(!bytes) { *size=NVM_SERVICE_BINDING_BYTES;return NVM_SERVICE_OK; }
    uint8_t staged[NVM_SERVICE_BINDING_BYTES]={0};
    service_put16(staged,NVM_SERVICE_BINDING_VERSION);
    service_put16(staged+2,NVM_SERVICE_BINDING_CATALOG_FILE);
    service_put32(staged+4,NVM_SERVICE_BINDING_COUNT);
    for(size_t i=0;i<NVM_SERVICE_BINDING_COUNT;i++) {
        size_t offset=16+8*i;
        service_put32(staged+offset,(uint32_t)i);
        service_put32(staged+offset+4,value->imports[i]);
    }
    memcpy(bytes,staged,sizeof(staged));
    *size=sizeof(staged);
    return NVM_SERVICE_OK;
}
