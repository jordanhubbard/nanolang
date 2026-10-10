#include "service_multi_nominal.h"
#include <string.h>
static uint16_t get16(const uint8_t *p) {return (uint16_t)(p[0]|((uint16_t)p[1]<<8));}
static uint32_t get32(const uint8_t *p) {return p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);}
static void put32(uint8_t *p,uint32_t v) {for(unsigned i=0;i<4;i++)p[i]=(uint8_t)(v>>(8*i));}
uint32_t nvm_multi_nominal_catalog_types(uint32_t catalog) {
    return catalog==1?8:catalog==2?9:catalog==3?7:0;
}
uint32_t nvm_multi_nominal_catalog_methods(uint32_t catalog) {
    return catalog==1 || catalog==2?5:catalog==3?4:0;
}
NvmServiceResult nvm_multi_nominal_check(const NvmMultiNominalBindings *b) {
    if(!b)return NVM_SERVICE_ARGUMENT;
    if(!b->count || b->count>NVM_MULTI_NOMINAL_MAX_INSTANCES)return NVM_SERVICE_COUNT;
    for(uint32_t i=0;i<b->count;i++) {
        const NvmServiceInstance *v=&b->instances[i];
        unsigned types=nvm_multi_nominal_catalog_types(v->catalog);
        unsigned methods=nvm_multi_nominal_catalog_methods(v->catalog);
        if(!types || !methods)return NVM_SERVICE_CATALOG;
        for(unsigned k=types;k<9;k++)if(v->layouts[k]!=UINT32_MAX)return NVM_SERVICE_RESERVED;
        for(unsigned k=methods;k<5;k++)if(v->imports[k]!=UINT32_MAX)return NVM_SERVICE_RESERVED;
        for(unsigned table=0;table<2;table++) {
            const uint32_t *indices=table?v->layouts:v->imports;
            unsigned count=table?types:methods;
            for(unsigned k=0;k<count;k++) {
                if(indices[k]==UINT32_MAX)return NVM_SERVICE_INDEX;
                for(uint32_t j=0;j<=i;j++) {
                    const NvmServiceInstance *prior=&b->instances[j];
                    const uint32_t *other=table?prior->layouts:prior->imports;
                    unsigned end=j==i?k:(table?nvm_multi_nominal_catalog_types(prior->catalog):nvm_multi_nominal_catalog_methods(prior->catalog));
                    for(unsigned n=0;n<end;n++)if(indices[k]==other[n])return NVM_SERVICE_INDEX;
                }
            }
        }
    }
    return NVM_SERVICE_OK;
}
NvmServiceResult nvm_multi_nominal_decode(const uint8_t *p,size_t n,NvmMultiNominalBindings *out) {
    if(!p || !out)return NVM_SERVICE_ARGUMENT;
    if(n<16 || n>NVM_MULTI_NOMINAL_MAX_BYTES)return NVM_SERVICE_SIZE;
    if(get16(p)!=NVM_MULTI_NOMINAL_VERSION)return NVM_SERVICE_VERSION;
    if(get16(p+2) || get32(p+12))return NVM_SERVICE_RESERVED;
    uint32_t count=get32(p+4);
    if(!count || count>NVM_MULTI_NOMINAL_MAX_INSTANCES)return NVM_SERVICE_COUNT;
    if(get32(p+8)!=NVM_MULTI_NOMINAL_ENTRY_BYTES || n!=16+(size_t)count*64)return NVM_SERVICE_SIZE;
    NvmMultiNominalBindings b={0};b.count=count;
    for(uint32_t i=0;i<count;i++) {
        const uint8_t *v=p+16+64*i;NvmServiceInstance *d=&b.instances[i];
        if(get32(v)!=i)return NVM_SERVICE_ORDINAL;
        d->catalog=get16(v+4);
        if(get16(v+6))return NVM_SERVICE_RESERVED;
        for(unsigned j=0;j<5;j++)d->imports[j]=get32(v+8+4*j);
        for(unsigned j=0;j<9;j++)d->layouts[j]=get32(v+28+4*j);
    }
    NvmServiceResult r=nvm_multi_nominal_check(&b);
    if(r==NVM_SERVICE_OK)*out=b;
    return r;
}
NvmServiceResult nvm_multi_nominal_encode(const NvmMultiNominalBindings *b,uint8_t *out,size_t capacity,size_t *size) {
    if(!size)return NVM_SERVICE_ARGUMENT;
    NvmServiceResult r=nvm_multi_nominal_check(b);if(r!=NVM_SERVICE_OK)return r;
    size_t n=16+(size_t)b->count*64;
    if(!out){*size=n;return NVM_SERVICE_OK;}
    if(capacity<n)return NVM_SERVICE_SIZE;
    uint8_t bytes[NVM_MULTI_NOMINAL_MAX_BYTES]={0};bytes[0]=NVM_MULTI_NOMINAL_VERSION;
    put32(bytes+4,b->count);put32(bytes+8,NVM_MULTI_NOMINAL_ENTRY_BYTES);
    for(uint32_t i=0;i<b->count;i++) {
        uint8_t *p=bytes+16+64*i;const NvmServiceInstance *v=&b->instances[i];
        put32(p,i);p[4]=(uint8_t)v->catalog;
        for(unsigned j=0;j<5;j++)put32(p+8+4*j,v->imports[j]);
        for(unsigned j=0;j<9;j++)put32(p+28+4*j,v->layouts[j]);
    }
    memcpy(out,bytes,n);*size=n;return NVM_SERVICE_OK;
}
