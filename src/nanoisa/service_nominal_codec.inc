/* I instantiate this bounded engine with one immutable catalog configuration. */
#include <string.h>
_Static_assert(NOMINAL_BYTES==16u+8u*(NOMINAL_METHODS+NOMINAL_TYPES),
    "I require an exact nominal table extent");

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
NvmServiceResult NOMINAL_FN(check)(const NominalBindings *value) {
    if(!value)return NVM_SERVICE_ARGUMENT;
    return indices(value->imports,NOMINAL_METHODS) &&
        indices(value->layouts,NOMINAL_TYPES)?NVM_SERVICE_OK:NVM_SERVICE_INDEX;
}
NvmServiceResult NOMINAL_FN(decode)(const uint8_t *bytes,size_t size,NominalBindings *out) {
    if(!bytes || !out)return NVM_SERVICE_ARGUMENT;
    if(size!=NOMINAL_BYTES)return NVM_SERVICE_SIZE;
    if(get16(bytes)!=NOMINAL_VERSION)return NVM_SERVICE_VERSION;
    if(get16(bytes+2)!=NOMINAL_CATALOG_ID)return NVM_SERVICE_CATALOG;
    if(get32(bytes+4)!=NOMINAL_METHODS || get32(bytes+8)!=NOMINAL_TYPES)
        return NVM_SERVICE_COUNT;
    if(get32(bytes+12))return NVM_SERVICE_RESERVED;
    NominalBindings value;
    for(size_t i=0;i<NOMINAL_METHODS;i++) {
        if(get32(bytes+16+8*i)!=i)return NVM_SERVICE_ORDINAL;
        value.imports[i]=get32(bytes+20+8*i);
    }
    for(size_t i=0;i<NOMINAL_TYPES;i++) {
        if(get32(bytes+16+8*NOMINAL_METHODS+8*i)!=i)return NVM_SERVICE_ORDINAL;
        value.layouts[i]=get32(bytes+20+8*NOMINAL_METHODS+8*i);
    }
    NvmServiceResult status=NOMINAL_FN(check)(&value);
    if(status!=NVM_SERVICE_OK)return status;
    *out=value;return NVM_SERVICE_OK;
}
NvmServiceResult NOMINAL_FN(encode)(const NominalBindings *value,uint8_t *bytes,
                                       size_t capacity,size_t *size) {
    if(!size)return NVM_SERVICE_ARGUMENT;
    NvmServiceResult status=NOMINAL_FN(check)(value);
    if(status!=NVM_SERVICE_OK)return status;
    if(bytes && capacity<NOMINAL_BYTES)return NVM_SERVICE_SIZE;
    if(!bytes){*size=NOMINAL_BYTES;return NVM_SERVICE_OK;}
    uint8_t staged[NOMINAL_BYTES]={0};
    put16(staged,NOMINAL_VERSION);put16(staged+2,NOMINAL_CATALOG_ID);
    put32(staged+4,NOMINAL_METHODS);put32(staged+8,NOMINAL_TYPES);
    for(size_t i=0;i<NOMINAL_METHODS;i++) {
        put32(staged+16+8*i,(uint32_t)i);put32(staged+20+8*i,value->imports[i]);
    }
    for(size_t i=0;i<NOMINAL_TYPES;i++) {
        put32(staged+16+8*NOMINAL_METHODS+8*i,(uint32_t)i);put32(staged+20+8*NOMINAL_METHODS+8*i,value->layouts[i]);
    }
    memcpy(bytes,staged,sizeof staged);*size=sizeof staged;return NVM_SERVICE_OK;
}
