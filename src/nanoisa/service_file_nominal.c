#include "service_file_nominal.h"
#include <string.h>

static uint16_t get16(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0]|((uint16_t)p[1]<<8));
}
static uint32_t get32(const uint8_t *p) {
    return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
}
static void put16(uint8_t *p,uint16_t n) { p[0]=(uint8_t)n;p[1]=(uint8_t)(n>>8); }
static void put32(uint8_t *p,uint32_t n) {
    p[0]=(uint8_t)n;p[1]=(uint8_t)(n>>8);p[2]=(uint8_t)(n>>16);p[3]=(uint8_t)(n>>24);
}
static bool indices(const uint32_t *p,size_t count) {
    for(size_t i=0;i<count;i++) {
        if(p[i]==UINT32_MAX)return false;
        for(size_t j=0;j<i;j++)if(p[j]==p[i])return false;
    }
    return true;
}
NvmServiceResult nvm_file_nominal_check(const NvmFileNominalBindings *value) {
    if(!value)return NVM_SERVICE_ARGUMENT;
    return indices(value->imports,NVM_SERVICE_BINDING_COUNT) &&
        indices(value->layouts,NVM_FILE_NOMINAL_TYPES)?NVM_SERVICE_OK:NVM_SERVICE_INDEX;
}
NvmServiceResult nvm_file_nominal_decode(const uint8_t *bytes,size_t size,NvmFileNominalBindings *out) {
    if(!bytes || !out)return NVM_SERVICE_ARGUMENT;
    if(size!=NVM_FILE_NOMINAL_BYTES)return NVM_SERVICE_SIZE;
    if(get16(bytes)!=NVM_FILE_NOMINAL_VERSION)return NVM_SERVICE_VERSION;
    if(get16(bytes+2)!=NVM_SERVICE_BINDING_CATALOG_FILE)return NVM_SERVICE_CATALOG;
    if(get32(bytes+4)!=NVM_SERVICE_BINDING_COUNT || get32(bytes+8)!=NVM_FILE_NOMINAL_TYPES)
        return NVM_SERVICE_COUNT;
    if(get32(bytes+12))return NVM_SERVICE_RESERVED;
    NvmFileNominalBindings value;
    for(size_t i=0;i<NVM_SERVICE_BINDING_COUNT;i++) {
        if(get32(bytes+16+8*i)!=i)return NVM_SERVICE_ORDINAL;
        value.imports[i]=get32(bytes+20+8*i);
    }
    for(size_t i=0;i<NVM_FILE_NOMINAL_TYPES;i++) {
        if(get32(bytes+56+8*i)!=i)return NVM_SERVICE_ORDINAL;
        value.layouts[i]=get32(bytes+60+8*i);
    }
    NvmServiceResult status=nvm_file_nominal_check(&value);
    if(status!=NVM_SERVICE_OK)return status;
    *out=value;return NVM_SERVICE_OK;
}
NvmServiceResult nvm_file_nominal_encode(const NvmFileNominalBindings *value,uint8_t *bytes,
                                       size_t capacity,size_t *size) {
    if(!size)return NVM_SERVICE_ARGUMENT;
    NvmServiceResult status=nvm_file_nominal_check(value);
    if(status!=NVM_SERVICE_OK)return status;
    if(bytes && capacity<NVM_FILE_NOMINAL_BYTES)return NVM_SERVICE_SIZE;
    if(!bytes){*size=NVM_FILE_NOMINAL_BYTES;return NVM_SERVICE_OK;}
    uint8_t staged[NVM_FILE_NOMINAL_BYTES]={0};
    put16(staged,NVM_FILE_NOMINAL_VERSION);put16(staged+2,NVM_SERVICE_BINDING_CATALOG_FILE);
    put32(staged+4,NVM_SERVICE_BINDING_COUNT);put32(staged+8,NVM_FILE_NOMINAL_TYPES);
    for(size_t i=0;i<NVM_SERVICE_BINDING_COUNT;i++) {
        put32(staged+16+8*i,(uint32_t)i);put32(staged+20+8*i,value->imports[i]);
    }
    for(size_t i=0;i<NVM_FILE_NOMINAL_TYPES;i++) {
        put32(staged+56+8*i,(uint32_t)i);put32(staged+60+8*i,value->layouts[i]);
    }
    memcpy(bytes,staged,sizeof staged);*size=sizeof staged;return NVM_SERVICE_OK;
}
