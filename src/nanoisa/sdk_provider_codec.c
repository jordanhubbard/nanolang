#include "sdk_provider_codec.h"
#include <stdlib.h>
#include <string.h>
struct NvmSdkProviderTransport {
    uint32_t counts[5];
    size_t offsets[5];
    size_t size;
    uint8_t data[];
};
static const uint32_t widths[5] = {32, 32, 24, 24, 4};
static const uint32_t ceilings[5] = {4096, 4096, 4096, 65536, 65536};
static uint32_t get32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
static void put32(uint8_t *p, uint32_t v) {
    p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); p[2]=(uint8_t)(v>>16); p[3]=(uint8_t)(v>>24);
}
static bool slice(uint32_t first, uint32_t count, uint32_t total) {
    return first<=total && count<=total-first;
}
static NvmSdkResult shape(const uint32_t counts[5], size_t offsets[5], size_t *size) {
    size_t n=NVM_SDK_PROVIDER_HEADER_BYTES;
    for (unsigned i=0;i<5;i++) {
        if (counts[i]>ceilings[i]) return NVM_SDK_LIMIT;
        if (counts[i]>(NVM_SDK_PROVIDER_MAX_BYTES-n)/widths[i]) return NVM_SDK_LIMIT;
        offsets[i]=n; n+=(size_t)counts[i]*widths[i];
    }
    *size=n; return NVM_SDK_OK;
}
/* I check reserved bytes and internal table slices before allocation. External
 * indices remain uninterpreted until the shared declaration/module validator. */
static NvmSdkResult validate(const uint8_t *p, size_t size,
                            uint32_t counts[5], size_t offsets[5]) {
    if (!p || size<NVM_SDK_PROVIDER_HEADER_BYTES) return NVM_SDK_INVALID;
    if (size>NVM_SDK_PROVIDER_MAX_BYTES) return NVM_SDK_LIMIT;
    if (get32(p)!=NVM_SDK_PROVIDER_REVISION || get32(p+4) || get32(p+28)) return NVM_SDK_INVALID;
    for(unsigned i=0;i<5;i++) counts[i]=get32(p+8+4*i);
    size_t expected; NvmSdkResult r=shape(counts,offsets,&expected);
    if(r!=NVM_SDK_OK) return r;
    if(size!=expected) return NVM_SDK_INVALID;
    for(uint32_t i=0;i<counts[0];i++) {
        const uint8_t *q=p+offsets[0]+(size_t)i*32;
        uint32_t kind=get32(q+8),layout=get32(q+12);
        if(kind>NVM_SDK_NOMINAL_OPAQUE || get32(q+24) || get32(q+28) ||
           !slice(get32(q+16),get32(q+20),counts[4])) return NVM_SDK_INVALID;
        if((kind==NVM_SDK_NOMINAL_OPAQUE)!=(layout==UINT32_MAX)) return NVM_SDK_INVALID;
    }
    for(uint32_t i=0;i<counts[1];i++) {
        const uint8_t *q=p+offsets[1]+(size_t)i*32;
        if(get32(q+24)||get32(q+28)) return NVM_SDK_INVALID;
    }
    for(uint32_t i=0;i<counts[2];i++) {
        const uint8_t *q=p+offsets[2]+(size_t)i*24;
        if(get32(q+20) || get32(q+8)>UINT16_MAX || get32(q+16)>UINT16_MAX ||
           !slice(get32(q+4),get32(q+8),counts[4]) ||
           !slice(get32(q+12),get32(q+16),counts[4])) return NVM_SDK_INVALID;
    }
    for(uint32_t i=0;i<counts[3];i++) {
        const uint8_t *q=p+offsets[3]+(size_t)i*24;
        uint32_t kind=get32(q),slot=get32(q+8),detail=get32(q+12),provider=get32(q+16);
        if(kind>NVM_SDK_BIND_FIELD || get32(q+20)) return NVM_SDK_INVALID;
        if(kind==NVM_SDK_BIND_FIELD) {
            if(slot>UINT16_MAX || provider!=UINT32_MAX) return NVM_SDK_INVALID;
            /* detail refers to the shared type pool, not this codec's tables. */
        } else {
            if(slot!=UINT32_MAX || detail>=counts[2]) return NVM_SDK_INVALID;
            if(kind==NVM_SDK_BIND_IMPORT ? provider>=counts[1] : provider!=UINT32_MAX) return NVM_SDK_INVALID;
        }
    }
    return NVM_SDK_OK;
}
NvmSdkResult nvm_sdk_provider_decode(const uint8_t *p,size_t size,size_t limit,
                                    NvmSdkProviderTransport **out) {
    if(!out) return NVM_SDK_INVALID;
    uint32_t counts[5]; size_t offsets[5];
    NvmSdkResult r=validate(p,size,counts,offsets); if(r!=NVM_SDK_OK)return r;
    if(limit>NVM_SDK_PROVIDER_MAX_BYTES)limit=NVM_SDK_PROVIDER_MAX_BYTES;
    if(sizeof(NvmSdkProviderTransport)>limit || size>limit-sizeof(NvmSdkProviderTransport))return NVM_SDK_LIMIT;
    NvmSdkProviderTransport *t=malloc(sizeof *t+size); if(!t)return NVM_SDK_MEMORY;
    memcpy(t->counts,counts,sizeof counts); memcpy(t->offsets,offsets,sizeof offsets);
    t->size=size; memcpy(t->data,p,size); *out=t; return NVM_SDK_OK;
}
void nvm_sdk_provider_transport_free(NvmSdkProviderTransport *p) { free(p); }
static void words(uint8_t *p,const uint32_t *values,unsigned n) {
    for(unsigned i=0;i<n;i++)put32(p+4*i,values[i]);
}
NvmSdkResult nvm_sdk_provider_encode(const NvmSdkProviderRows *rows,size_t limit,
                                    uint8_t **data,size_t *size) {
    if(!rows||!data||!size)return NVM_SDK_INVALID;
    uint32_t counts[5]={rows->nominal_count,rows->provider_count,rows->signature_count,rows->binding_count,rows->reference_count};
    if((counts[0]&&!rows->nominals)||(counts[1]&&!rows->providers)||
       (counts[2]&&!rows->signatures)||(counts[3]&&!rows->bindings)||(counts[4]&&!rows->references))return NVM_SDK_INVALID;
    size_t offsets[5],bytes;NvmSdkResult r=shape(counts,offsets,&bytes);if(r!=NVM_SDK_OK)return r;
    if(limit>NVM_SDK_PROVIDER_MAX_BYTES)limit=NVM_SDK_PROVIDER_MAX_BYTES;
    if(bytes>limit)return NVM_SDK_LIMIT;
    uint8_t *p=calloc(1,bytes);if(!p)return NVM_SDK_MEMORY;
    put32(p,NVM_SDK_PROVIDER_REVISION);words(p+8,counts,5);
    for(uint32_t i=0;i<counts[0];i++) {
        NvmSdkNominalRow a=rows->nominals[i];uint32_t v[]={a.owner,a.name,a.kind,a.layout,a.argument_first,a.argument_count};
        words(p+offsets[0]+(size_t)i*32,v,6);
    }
    for(uint32_t i=0;i<counts[1];i++) {
        NvmSdkProviderRow a=rows->providers[i];uint32_t v[]={a.module,a.abi,a.target,a.artifact_digest,a.generation_digest,a.library};
        words(p+offsets[1]+(size_t)i*32,v,6);
    }
    for(uint32_t i=0;i<counts[2];i++) {
        NvmSdkSignatureRow a=rows->signatures[i];uint32_t v[]={a.coarse_signature,a.parameter_first,a.parameter_count,a.result_first,a.result_count};
        words(p+offsets[2]+(size_t)i*24,v,5);
    }
    for(uint32_t i=0;i<counts[3];i++) {
        NvmSdkBindingRow a=rows->bindings[i];uint32_t v[]={a.kind,a.subject,a.slot,a.detail,a.provider};
        words(p+offsets[3]+(size_t)i*24,v,5);
    }
    words(p+offsets[4],rows->references,counts[4]);
    uint32_t checked[5];size_t checked_offsets[5];r=validate(p,bytes,checked,checked_offsets);
    if(r!=NVM_SDK_OK){free(p);return r;}
    *data=p;*size=bytes;return NVM_SDK_OK;
}
static const uint8_t *row(const NvmSdkProviderTransport *p,unsigned table,uint32_t i) {
    return p&&i<p->counts[table]?p->data+p->offsets[table]+(size_t)i*widths[table]:NULL;
}
bool nvm_sdk_provider_counts(const NvmSdkProviderTransport *p,uint32_t out[5]) {
    if(!p||!out)return false;
    memcpy(out,p->counts,sizeof p->counts);return true;
}
bool nvm_sdk_provider_nominal(const NvmSdkProviderTransport *p,uint32_t i,NvmSdkNominalRow *out) {
    const uint8_t *q=row(p,0,i);if(!q||!out)return false;
    *out=(NvmSdkNominalRow){get32(q),get32(q+4),get32(q+8),get32(q+12),get32(q+16),get32(q+20)};return true;
}
bool nvm_sdk_provider_requirement(const NvmSdkProviderTransport *p,uint32_t i,NvmSdkProviderRow *out) {
    const uint8_t *q=row(p,1,i);if(!q||!out)return false;
    *out=(NvmSdkProviderRow){get32(q),get32(q+4),get32(q+8),get32(q+12),get32(q+16),get32(q+20)};return true;
}
bool nvm_sdk_provider_signature(const NvmSdkProviderTransport *p,uint32_t i,NvmSdkSignatureRow *out) {
    const uint8_t *q=row(p,2,i);if(!q||!out)return false;
    *out=(NvmSdkSignatureRow){get32(q),get32(q+4),get32(q+8),get32(q+12),get32(q+16)};return true;
}
bool nvm_sdk_provider_binding(const NvmSdkProviderTransport *p,uint32_t i,NvmSdkBindingRow *out) {
    const uint8_t *q=row(p,3,i);if(!q||!out)return false;
    *out=(NvmSdkBindingRow){get32(q),get32(q+4),get32(q+8),get32(q+12),get32(q+16)};return true;
}
bool nvm_sdk_provider_reference(const NvmSdkProviderTransport *p,uint32_t i,uint32_t *out) {
    const uint8_t *q=row(p,4,i);if(!q||!out)return false;*out=get32(q);return true;
}
